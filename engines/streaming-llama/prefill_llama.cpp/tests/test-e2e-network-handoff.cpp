// E2E test: prefill → KV export → TCP send → receive → import → decode → token relay
// Uses a small model (Ministral 3B Q4_K_S) in a single process with threads.
//
// Usage: test-e2e-network-handoff -m /path/to/model.gguf
//
// Requires LLAMA_PREFILL_TB_ENABLE=1 for the transport sender.

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "llama-kv-artifact.h"
#include "llama-kv-receiver.h"
#include "llama-token-stream.h"
#include "llama-tb-transport.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#undef NDEBUG
#include <cassert>

namespace fs = std::filesystem;

static std::string find_model_arg(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        if ((std::string(argv[i]) == "-m" || std::string(argv[i]) == "--model") && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return "";
}

int main(int argc, char ** argv) {
    std::string model_path = find_model_arg(argc, argv);
    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m /path/to/model.gguf\n", argv[0]);
        fprintf(stderr, "Skipping E2E test (no model specified)\n");
        return 0;
    }

    fprintf(stderr, "=== E2E Network Handoff Test ===\n");
    fprintf(stderr, "Model: %s\n", model_path.c_str());

    // Enable TB transport
#ifdef _WIN32
    _putenv_s("LLAMA_PREFILL_TB_ENABLE", "1");
#else
    setenv("LLAMA_PREFILL_TB_ENABLE", "1", 1);
#endif

    const int KV_PORT = 19300;
    const int TOKEN_PORT = 19301;
    const int N_CTX = 512;
    const int N_PREDICT = 10;
    const char * PROMPT = "The capital of France is";

    llama_backend_init();

    // --- STEP 1: Load model and prefill on "RTX side" ---
    fprintf(stderr, "\n[step 1] Loading model for prefill...\n");
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU prefill (simulates streaming mode)
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    assert(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_CTX;
    cparams.n_batch = 128;
    cparams.n_threads = 4;
    llama_context * ctx_prefill = llama_init_from_model(model, cparams);
    assert(ctx_prefill);

    // Tokenize prompt
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(N_CTX);
    int n_tokens = llama_tokenize(vocab, PROMPT, (int)strlen(PROMPT),
                                   tokens.data(), (int)tokens.size(), true, true);
    assert(n_tokens > 0);
    tokens.resize(n_tokens);
    fprintf(stderr, "[step 1] Tokenized: %d tokens\n", n_tokens);

    // Prefill
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
    }
    int rc = llama_decode(ctx_prefill, batch);
    assert(rc == 0);
    fprintf(stderr, "[step 1] Prefill complete\n");

    // --- STEP 2: Export KV artifact ---
    fprintf(stderr, "\n[step 2] Exporting KV state...\n");
    size_t state_size = llama_state_seq_get_size(ctx_prefill, 0);
    assert(state_size > 0);
    std::vector<uint8_t> state_data(state_size);
    size_t written = llama_state_seq_get_data(ctx_prefill, state_data.data(), state_data.size(), 0);
    assert(written > 0);
    state_data.resize(written);
    fprintf(stderr, "[step 2] KV state: %zu bytes\n", written);

    // Write artifact to temp file
    auto artifact_path = (fs::temp_directory_path() / "test_e2e_handoff.kvbin").string();
    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 1;
    meta.n_layers = (uint32_t) llama_model_n_layer(model);
    meta.n_ctx = N_CTX;
    meta.token_count = (uint32_t) n_tokens;
    meta.last_token_id = tokens.back();
    meta.type_k = 1; // fp16
    meta.type_v = 1;

    std::string error;
    assert(llama_kv_artifact_write(artifact_path, state_data, meta, nullptr, &error));
    fprintf(stderr, "[step 2] Artifact written: %s (%zu + %zu bytes)\n",
            artifact_path.c_str(), sizeof(meta), state_data.size());

    // --- STEP 3: Start receiver thread (simulates Mac) ---
    fprintf(stderr, "\n[step 3] Starting KV receiver + decode on port %d...\n", KV_PORT);

    std::atomic<bool> decode_done{false};
    std::vector<std::string> decoded_texts;
    int decode_token_count = 0;

    std::thread mac_thread([&]() {
        // Receive KV artifact over TCP
        llama_kv_receiver_config rcfg;
        rcfg.bind_host = "127.0.0.1";
        rcfg.port = KV_PORT;
        rcfg.accept_timeout_ms = 15000;
        rcfg.recv_timeout_ms = 15000;

        std::vector<uint8_t> raw_artifact;
        llama_kv_receiver_result rresult;
        std::string rerror;
        if (!llama_kv_receiver_accept_artifact(rcfg, raw_artifact, &rresult, &rerror)) {
            fprintf(stderr, "[mac] KV receive failed: %s\n", rerror.c_str());
            decode_done = true;
            return;
        }
        fprintf(stderr, "[mac] KV received: %lu bytes in %.1f ms\n",
                (unsigned long)rresult.bytes_received, rresult.receive_ms);

        // Parse artifact from memory
        std::vector<uint8_t> payload;
        llama_kv_artifact_metadata rmeta = {};
        if (!llama_kv_artifact_read_mem(raw_artifact.data(), raw_artifact.size(),
                                         payload, &rmeta, nullptr, &rerror)) {
            fprintf(stderr, "[mac] Artifact parse failed: %s\n", rerror.c_str());
            decode_done = true;
            return;
        }

        // Create decode context (simulates Mac side)
        llama_context_params dparams = llama_context_default_params();
        dparams.n_ctx = N_CTX;
        dparams.n_batch = 128;
        dparams.n_threads = 4;
        llama_context * ctx_decode = llama_init_from_model(model, dparams);
        if (!ctx_decode) {
            fprintf(stderr, "[mac] Failed to create decode context\n");
            decode_done = true;
            return;
        }

        // Import KV
        size_t consumed = llama_state_seq_set_data(ctx_decode, payload.data(), payload.size(), 0);
        if (consumed == 0) {
            fprintf(stderr, "[mac] KV import failed\n");
            llama_free(ctx_decode);
            decode_done = true;
            return;
        }
        fprintf(stderr, "[mac] KV imported: %zu bytes\n", consumed);

        // Start token stream server
        llama_token_stream_server server;
        if (!server.start("127.0.0.1", TOKEN_PORT)) {
            fprintf(stderr, "[mac] Token server start failed\n");
            llama_free(ctx_decode);
            decode_done = true;
            return;
        }
        fprintf(stderr, "[mac] Token stream on port %d, waiting for client...\n", TOKEN_PORT);

        if (!server.wait_for_client(10000)) {
            fprintf(stderr, "[mac] No token stream client\n");
            server.stop();
            llama_free(ctx_decode);
            decode_done = true;
            return;
        }

        llama_token_handshake hs = {};
        hs.version = 1;
        hs.status = "ready";
        hs.token_count = rmeta.token_count;
        server.send_handshake(hs);

        // Decode
        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        llama_token last_token = (rmeta.last_token_id >= 0)
            ? (llama_token) rmeta.last_token_id
            : llama_vocab_bos(vocab);

        llama_batch dbatch = llama_batch_init(1, 0, 1);

        for (int i = 0; i < N_PREDICT; i++) {
            common_batch_clear(dbatch);
            common_batch_add(dbatch, last_token, rmeta.token_count + i, {0}, true);

            if (llama_decode(ctx_decode, dbatch) != 0) {
                fprintf(stderr, "[mac] Decode failed at token %d\n", i);
                break;
            }

            llama_token new_token = llama_sampler_sample(smpl, ctx_decode, -1);
            if (llama_vocab_is_eog(vocab, new_token)) break;

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            std::string text(buf, n > 0 ? n : 0);

            llama_token_msg tmsg = {};
            tmsg.token_id = new_token;
            tmsg.text = text;
            tmsg.pos = rmeta.token_count + i;
            server.enqueue(tmsg);

            decoded_texts.push_back(text);
            last_token = new_token;
            decode_token_count++;
        }

        double tok_s = decode_token_count > 0 ? 100.0 : 0.0;  // placeholder
        server.send_done(decode_token_count, tok_s);

        llama_batch_free(dbatch);
        llama_sampler_free(smpl);
        server.stop();
        llama_free(ctx_decode);
        decode_done = true;
    });

    // Give receiver time to bind
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // --- STEP 4: Send artifact over TCP (simulates RTX sender) ---
    fprintf(stderr, "\n[step 4] Sending KV artifact to 127.0.0.1:%d...\n", KV_PORT);
    llama_tb_transfer_options opts;
    opts.transport_mode = "tcp";
    opts.kv_host = "127.0.0.1";
    opts.kv_port = KV_PORT;
    opts.session_id = "e2e_test";
    opts.chunk_bytes = 64 * 1024;
    opts.stream_count = 1;

    llama_tb_transfer_result send_result;
    std::string send_error;
    bool send_ok = llama_tb_transport_send_artifact(artifact_path, opts, &send_result, &send_error);
    if (!send_ok) {
        fprintf(stderr, "[rtx] Send failed: %s\n", send_error.c_str());
    }
    assert(send_ok);
    fprintf(stderr, "[step 4] Sent: %llu bytes, %.1f ms, %.3f Gbps\n",
            (unsigned long long) send_result.bytes_sent,
            send_result.transfer_ms, send_result.throughput_gbps);

    // --- STEP 5: Connect token stream client (simulates RTX relay) ---
    fprintf(stderr, "\n[step 5] Connecting to token stream on port %d...\n", TOKEN_PORT);
    llama_token_stream_client client;
    assert(client.connect("127.0.0.1", TOKEN_PORT, 10, 500));

    llama_token_handshake hs;
    assert(client.read_handshake(hs, 10000));
    assert(hs.version == 1);
    assert(hs.status == "ready");
    fprintf(stderr, "[step 5] Handshake OK (token_count=%u)\n", hs.token_count);

    std::vector<std::string> relayed_texts;
    while (true) {
        llama_token_msg msg;
        if (!client.read_msg(msg, 30000)) {
            fprintf(stderr, "[rtx] Token read timeout\n");
            break;
        }
        if (msg.done) {
            fprintf(stderr, "[step 5] Done: %d tokens decoded\n", msg.tokens_generated);
            break;
        }
        relayed_texts.push_back(msg.text);
    }
    client.disconnect();

    mac_thread.join();

    // --- STEP 6: Verify ---
    fprintf(stderr, "\n[step 6] Verification...\n");
    fprintf(stderr, "  Decoded tokens: %d\n", decode_token_count);
    fprintf(stderr, "  Relayed tokens: %zu\n", relayed_texts.size());
    fprintf(stderr, "  Generated text: ");
    for (auto & t : relayed_texts) fprintf(stderr, "%s", t.c_str());
    fprintf(stderr, "\n");

    assert(decode_token_count > 0);
    assert(relayed_texts.size() == decoded_texts.size());
    for (size_t i = 0; i < relayed_texts.size(); i++) {
        assert(relayed_texts[i] == decoded_texts[i]);
    }

    // Cleanup
    llama_batch_free(batch);
    llama_free(ctx_prefill);
    llama_model_free(model);
    llama_backend_free();
    fs::remove(artifact_path);

    fprintf(stderr, "\n=== E2E Network Handoff Test: PASSED ===\n");
    return 0;
}
