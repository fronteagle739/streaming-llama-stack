// End-to-end handoff test: prefill → export KV → import KV → decode → token stream.
// Runs entirely on localhost with a single model file.
// Uses the tinyllamas/stories15M-q4_0 model (same as other tests).
//
// Flow:
//   1. Load model, create context A, tokenize prompt, prefill
//   2. Export KV state from context A (as artifact payload)
//   3. Create context B, import KV state
//   4. Start token stream server
//   5. Decode on context B, stream tokens
//   6. Token stream client receives and verifies

#include "common.h"
#include "llama.h"
#include "llama-kv-artifact.h"
#include "llama-token-stream.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Batch helper — fill a batch for a single token at a given position.
static void batch_set_one(llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq_id, bool logits) {
    batch.n_tokens    = 1;
    batch.token[0]    = token;
    batch.pos[0]      = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq_id;
    batch.logits[0]   = logits ? 1 : 0;
}

// Batch helper — fill a batch with multiple tokens for prefill.
static void batch_fill_prefill(llama_batch & batch, const llama_token * tokens, int n, llama_seq_id seq_id) {
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = (i == n - 1) ? 1 : 0;  // only last token needs logits
    }
}

#undef NDEBUG
#include <cassert>

static std::string get_model_path(int argc, char ** argv) {
    // Accept model path as -m argument (like other tests)
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            return argv[i + 1];
        }
    }
    // Fallback: try environment
    const char * env = getenv("LLAMA_TEST_MODEL");
    if (env) return env;
    return "";
}

int main(int argc, char ** argv) {
    std::string model_path = get_model_path(argc, argv);
    if (model_path.empty()) {
        fprintf(stderr, "test-e2e-handoff: model path required (-m PATH)\n");
        return 1;
    }

    printf("test-e2e-handoff: model=%s\n", model_path.c_str());

    llama_backend_init();

    // -----------------------------------------------------------------------
    // Step 1: Load model + create context A + prefill
    // -----------------------------------------------------------------------
    printf("test-e2e-handoff: step 1 — load model and prefill\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU for testing
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    assert(model && "failed to load model");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = 256;
    ctx_params.n_batch  = 256;
    ctx_params.n_threads = 2;

    llama_context * ctx_a = llama_init_from_model(model, ctx_params);
    assert(ctx_a && "failed to create context A");

    // Tokenize a simple prompt
    const char * prompt = "Once upon a time, there was a";
    std::vector<llama_token> tokens(256);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                  tokens.data(), tokens.size(), true, true);
    assert(n_tokens > 0);
    tokens.resize(n_tokens);
    printf("  prompt tokens: %d\n", n_tokens);

    // Prefill (eval all prompt tokens)
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch_fill_prefill(batch, tokens.data(), n_tokens, 0);

    int rc = llama_decode(ctx_a, batch);
    assert(rc == 0 && "prefill decode failed");
    printf("  prefill complete\n");

    // -----------------------------------------------------------------------
    // Step 2: Export KV state from context A
    // -----------------------------------------------------------------------
    printf("test-e2e-handoff: step 2 — export KV state\n");

    // Get the state size and serialize
    size_t state_size = llama_state_seq_get_size(ctx_a, 0);
    assert(state_size > 0);
    std::vector<uint8_t> state_data(state_size);
    size_t written = llama_state_seq_get_data(ctx_a, state_data.data(), state_data.size(), 0);
    assert(written > 0 && written <= state_size);
    state_data.resize(written);
    printf("  KV state: %zu bytes\n", written);

    // Sample token from context A (greedy) → this is tok1 (first decode token)
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_token tok1 = llama_sampler_sample(smpl, ctx_a, -1);
    char tok1_buf[64];
    int tok1_len = llama_token_to_piece(vocab, tok1, tok1_buf, sizeof(tok1_buf), 0, true);
    std::string tok1_text(tok1_buf, tok1_len > 0 ? tok1_len : 0);
    printf("  tok1 (from ctx_a): %d (\"%s\")\n", tok1, tok1_text.c_str());

    // Now feed tok1 back into context A and get tok2 for baseline
    llama_sampler_free(smpl);

    llama_batch batch_a2 = llama_batch_init(1, 0, 1);
    batch_set_one(batch_a2, tok1, n_tokens, 0, true);
    rc = llama_decode(ctx_a, batch_a2);
    assert(rc == 0 && "ctx_a second decode failed");

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_token tok2_baseline = llama_sampler_sample(smpl, ctx_a, -1);
    char tok2_buf[64];
    int tok2_len = llama_token_to_piece(vocab, tok2_baseline, tok2_buf, sizeof(tok2_buf), 0, true);
    std::string tok2_text(tok2_buf, tok2_len > 0 ? tok2_len : 0);
    printf("  tok2 (from ctx_a): %d (\"%s\")\n", tok2_baseline, tok2_text.c_str());

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_batch_free(batch_a2);

    // -----------------------------------------------------------------------
    // Step 3: Create context B + import KV state
    // -----------------------------------------------------------------------
    printf("test-e2e-handoff: step 3 — create context B and import KV\n");

    llama_context * ctx_b = llama_init_from_model(model, ctx_params);
    assert(ctx_b && "failed to create context B");

    auto t_import_start = std::chrono::high_resolution_clock::now();
    size_t consumed = llama_state_seq_set_data(ctx_b, state_data.data(), state_data.size(), 0);
    auto t_import_end = std::chrono::high_resolution_clock::now();
    assert(consumed > 0 && "KV import failed");

    double import_ms = std::chrono::duration<double, std::milli>(t_import_end - t_import_start).count();
    printf("  import: %zu bytes in %.1f ms\n", consumed, import_ms);

    // -----------------------------------------------------------------------
    // Step 4: Verify imported state produces same tokens as original
    // -----------------------------------------------------------------------
    printf("test-e2e-handoff: step 4 — verify decoded tokens match\n");

    // Feed tok1 (same token context A produced) into context B after KV import.
    // Context B should produce the same tok2 as context A did.
    llama_batch batch_b = llama_batch_init(1, 0, 1);
    batch_set_one(batch_b, tok1, n_tokens, 0, true);

    rc = llama_decode(ctx_b, batch_b);
    assert(rc == 0 && "decode on context B failed");

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_token tok2_imported = llama_sampler_sample(smpl, ctx_b, -1);

    char tok2i_buf[64];
    int tok2i_len = llama_token_to_piece(vocab, tok2_imported, tok2i_buf, sizeof(tok2i_buf), 0, true);
    std::string tok2i_text(tok2i_buf, tok2i_len > 0 ? tok2i_len : 0);
    printf("  tok2 (from ctx_b): %d (\"%s\")\n", tok2_imported, tok2i_text.c_str());

    // After feeding the same tok1, both contexts must produce the same tok2.
    // This proves the KV state was imported correctly.
    assert(tok2_imported == tok2_baseline &&
           "MISMATCH: imported KV state produces different token than original!");
    printf("  MATCH — KV import verified\n");

    llama_sampler_free(smpl);

    // -----------------------------------------------------------------------
    // Step 5: Token stream E2E — decode on B, stream tokens to client
    // -----------------------------------------------------------------------
    printf("test-e2e-handoff: step 5 — token stream E2E\n");

    {
    llama_token_stream_server server;
    assert(server.start("127.0.0.1", 0));
    int port = server.bound_port();
    printf("  token stream on port %d\n", port);

    const int n_decode = 5;
    std::vector<llama_token_msg> received;

    // Client thread
    std::thread client_thread([port, &received]() {
        llama_token_stream_client client;
        assert(client.connect("127.0.0.1", port, 5, 200));

        llama_token_handshake hs = {};
        assert(client.read_handshake(hs, 10000));
        assert(hs.version == 1);
        assert(hs.status == "ready");

        for (;;) {
            llama_token_msg msg = {};
            assert(client.read_msg(msg, 30000));
            if (msg.done) break;
            received.push_back(msg);
        }
        client.disconnect();
    });

    assert(server.wait_for_client(5000));

    llama_token_handshake hs = {};
    hs.version = 1;
    hs.status  = "ready";
    hs.token_count = n_tokens;
    server.send_handshake(hs);

    // Decode n_decode tokens on context B, streaming each via token server
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_token cur_token = tok2_imported;
    int decode_pos = n_tokens + 1;  // we already decoded one token in step 4

    for (int i = 0; i < n_decode; i++) {
        batch_set_one(batch_b, cur_token, decode_pos + i, 0, true);

        rc = llama_decode(ctx_b, batch_b);
        assert(rc == 0);

        llama_token next = llama_sampler_sample(smpl, ctx_b, -1);

        char buf[256];
        int n = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
        std::string text(buf, n > 0 ? n : 0);

        llama_token_msg tmsg = {};
        tmsg.token_id = next;
        tmsg.text = text;
        tmsg.pos = decode_pos + i;
        server.enqueue(tmsg);

        cur_token = next;
    }

    server.send_done(n_decode, 0.0);
    llama_sampler_free(smpl);

    client_thread.join();
    server.stop();

    assert((int) received.size() == n_decode);
    printf("  received %d tokens via stream\n", n_decode);

    for (int i = 0; i < n_decode; i++) {
        printf("    [%d] id=%d text=\"%s\"\n",
               i, received[i].token_id, received[i].text.c_str());
    }
    }  // end token stream scope

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    llama_batch_free(batch_b);
    llama_free(ctx_a);
    llama_free(ctx_b);
    llama_model_free(model);
    llama_backend_free();

    printf("\ntest-e2e-handoff: all tests PASSED\n");
    return 0;
}
