// TDD: KV receiver tests — written before implementation.
// Loopback test: sender thread sends artifact via TBP1 frames,
// receiver thread accepts and reassembles.
// Uses small synthetic artifacts — no model needed.

#include "llama-kv-artifact.h"
#include "llama-kv-receiver.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#undef NDEBUG
#include <cassert>

namespace fs = std::filesystem;

// We need the sender to send an artifact over TCP.
// Import the send function from llama-tb-transport.
#include "llama-tb-transport.h"

static std::string tmp_path(const std::string & name) {
    fs::path p = fs::temp_directory_path() / ("test_kv_recv_" + name + ".bin");
    return p.string();
}

// ---------------------------------------------------------------------------
// Test: loopback — sender and receiver on localhost
// ---------------------------------------------------------------------------
static void test_loopback_basic() {
    printf("  loopback_basic ... ");

    // Create a small artifact file that the sender will transmit
    auto artifact_path = tmp_path("send");
    std::vector<uint8_t> payload(256);
    for (size_t i = 0; i < payload.size(); i++) payload[i] = (uint8_t)(i & 0xFF);

    llama_kv_artifact_metadata meta = {};
    meta.format_major      = 1;
    meta.format_minor      = 1;
    meta.n_layers          = 4;
    meta.token_count       = 100;
    meta.last_token_id     = 42;
    meta.v_trans           = 1;
    meta.model_fingerprint = 12345;

    std::string error;
    assert(llama_kv_artifact_write(artifact_path, payload, meta, nullptr, &error));

    // Read the file to know expected bytes
    std::ifstream ifs(artifact_path, std::ios::binary | std::ios::ate);
    size_t expected_file_size = (size_t) ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> expected_file(expected_file_size);
    ifs.read((char*)expected_file.data(), (std::streamsize)expected_file_size);
    ifs.close();

    // Pick a port
    const int port = 19100;  // test port, unlikely to conflict

    // Start receiver in a thread
    std::vector<uint8_t> received_artifact;
    llama_kv_receiver_result recv_result = {};
    std::string recv_error;
    bool recv_ok = false;

    std::thread receiver_thread([&]() {
        llama_kv_receiver_config cfg;
        cfg.bind_host = "127.0.0.1";
        cfg.port = port;
        cfg.accept_timeout_ms = 10000;
        cfg.recv_timeout_ms = 10000;
        recv_ok = llama_kv_receiver_accept_artifact(cfg, received_artifact, &recv_result, &recv_error);
    });

    // Give receiver a moment to bind
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Send via TB transport
    // Need LLAMA_PREFILL_TB_ENABLE=1 for transport to work
#ifdef _WIN32
    _putenv_s("LLAMA_PREFILL_TB_ENABLE", "1");
#else
    setenv("LLAMA_PREFILL_TB_ENABLE", "1", 1);
#endif

    llama_tb_transfer_options opts;
    opts.transport_mode = "tcp";
    opts.kv_host = "127.0.0.1";
    opts.kv_port = port;
    opts.session_id = "test_loopback";
    opts.chunk_bytes = 64;  // small chunks to exercise multi-chunk path
    opts.stream_count = 1;

    llama_tb_transfer_result send_result;
    std::string send_error;
    bool send_ok = llama_tb_transport_send_artifact(artifact_path, opts, &send_result, &send_error);

    receiver_thread.join();

    if (!send_ok) {
        fprintf(stderr, "send failed: %s\n", send_error.c_str());
    }
    if (!recv_ok) {
        fprintf(stderr, "recv failed: %s\n", recv_error.c_str());
    }

    // Verify
    assert(send_ok);
    assert(recv_ok);
    assert(received_artifact == expected_file);
    assert(recv_result.bytes_received == expected_file_size);
    assert(recv_result.chunks_received > 0);

    // Parse the received artifact from memory
    std::vector<uint8_t> parsed_payload;
    llama_kv_artifact_metadata parsed_meta = {};
    assert(llama_kv_artifact_read_mem(received_artifact.data(), received_artifact.size(),
                                       parsed_payload, &parsed_meta, nullptr, &error));
    assert(parsed_payload == payload);
    assert(parsed_meta.last_token_id == 42);
    assert(parsed_meta.v_trans == 1);
    assert(parsed_meta.model_fingerprint == 12345);

    fs::remove(artifact_path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: receiver timeout when no sender connects
// ---------------------------------------------------------------------------
static void test_receiver_timeout() {
    printf("  receiver_timeout ... ");

    llama_kv_receiver_config cfg;
    cfg.bind_host = "127.0.0.1";
    cfg.port = 19101;
    cfg.accept_timeout_ms = 500;  // very short timeout
    cfg.recv_timeout_ms = 500;

    std::vector<uint8_t> artifact;
    std::string error;
    bool ok = llama_kv_receiver_accept_artifact(cfg, artifact, nullptr, &error);
    assert(!ok);
    assert(artifact.empty());

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("test-kv-receiver: loopback tests\n");
    test_receiver_timeout();
    test_loopback_basic();

    printf("\ntest-kv-receiver: all tests OK\n");
    return 0;
}
