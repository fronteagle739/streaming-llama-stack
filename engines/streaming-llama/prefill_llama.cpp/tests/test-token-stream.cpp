// TDD: Token stream unit tests — written before implementation.
// Tests the 16k ring buffer, TCP JSON-lines protocol, version handshake,
// and backpressure behavior.

#include "llama-token-stream.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

#undef NDEBUG
#include <cassert>

// ---------------------------------------------------------------------------
// Ring buffer tests (no networking)
// ---------------------------------------------------------------------------

static void test_ring_basic() {
    printf("  ring_basic ... ");
    llama_token_ring ring;

    assert(ring.size() == 0);
    assert(ring.total_dropped() == 0);

    llama_token_msg msg = {};
    msg.token_id = 42;
    msg.text     = "hello";
    msg.pos      = 1;

    assert(ring.push(msg));  // not full yet
    assert(ring.size() == 1);

    llama_token_msg out = {};
    assert(ring.pop(out));
    assert(out.token_id == 42);
    assert(out.text == "hello");
    assert(out.pos == 1);
    assert(ring.size() == 0);

    // pop on empty → false
    assert(!ring.pop(out));

    printf("OK\n");
}

static void test_ring_fifo_order() {
    printf("  ring_fifo_order ... ");
    llama_token_ring ring;

    for (int i = 0; i < 100; i++) {
        llama_token_msg m = {};
        m.token_id = i;
        ring.push(m);
    }
    assert(ring.size() == 100);

    for (int i = 0; i < 100; i++) {
        llama_token_msg m = {};
        assert(ring.pop(m));
        assert(m.token_id == i);
    }
    assert(ring.size() == 0);

    printf("OK\n");
}

static void test_ring_overflow() {
    printf("  ring_overflow ... ");
    llama_token_ring ring;

    // fill to capacity
    for (size_t i = 0; i < llama_token_ring::CAPACITY; i++) {
        llama_token_msg m = {};
        m.token_id = (int32_t) i;
        assert(ring.push(m));  // returns true (not overwriting)
    }
    assert(ring.size() == llama_token_ring::CAPACITY);
    assert(ring.total_dropped() == 0);

    // one more → overwrites oldest, returns false
    llama_token_msg extra = {};
    extra.token_id = 99999;
    assert(!ring.push(extra));  // returns false = overwrite happened
    assert(ring.total_dropped() == 1);
    assert(ring.size() == llama_token_ring::CAPACITY);

    // oldest should now be token_id=1 (0 was overwritten)
    llama_token_msg out = {};
    assert(ring.pop(out));
    assert(out.token_id == 1);

    printf("OK\n");
}

static void test_ring_wrap_around() {
    printf("  ring_wrap_around ... ");
    llama_token_ring ring;

    // push and pop repeatedly to exercise wrap-around
    for (int cycle = 0; cycle < 3; cycle++) {
        for (int i = 0; i < 1000; i++) {
            llama_token_msg m = {};
            m.token_id = cycle * 1000 + i;
            ring.push(m);
        }
        for (int i = 0; i < 1000; i++) {
            llama_token_msg m = {};
            assert(ring.pop(m));
            assert(m.token_id == cycle * 1000 + i);
        }
        assert(ring.size() == 0);
    }

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// JSON serialization tests
// ---------------------------------------------------------------------------

static void test_json_token_roundtrip() {
    printf("  json_token_roundtrip ... ");

    llama_token_msg msg = {};
    msg.token_id = 42;
    msg.text     = "hello world";
    msg.pos      = 7;
    msg.timestamp_us = 1234567890;
    msg.dropped_count = 3;

    std::string json = llama_token_msg_to_json(msg);
    assert(!json.empty());

    llama_token_msg out = {};
    assert(llama_token_msg_from_json(json, out));
    assert(out.token_id == 42);
    assert(out.text == "hello world");
    assert(out.pos == 7);
    assert(out.timestamp_us == 1234567890);
    assert(out.dropped_count == 3);
    assert(!out.done);

    printf("OK\n");
}

static void test_json_done_roundtrip() {
    printf("  json_done_roundtrip ... ");

    llama_token_msg msg = {};
    msg.done = true;
    msg.tokens_generated = 500;
    msg.decode_tok_s = 42.5;
    msg.decode_elapsed_ms = 11900.0;
    msg.decode_intervals = 499;
    msg.decode_tok_s_early = 48.0;
    msg.decode_tok_s_middle = 42.0;
    msg.decode_tok_s_late = 36.0;
    msg.retained_token_count = 8192;
    msg.retained_prefix_hash = UINT64_MAX - 7;

    std::string json = llama_token_msg_to_json(msg);

    llama_token_msg out = {};
    assert(llama_token_msg_from_json(json, out));
    assert(out.done);
    assert(out.tokens_generated == 500);
    assert(out.decode_tok_s > 42.0 && out.decode_tok_s < 43.0);
    assert(out.decode_elapsed_ms > 11899.0 && out.decode_elapsed_ms < 11901.0);
    assert(out.decode_intervals == 499);
    assert(out.decode_tok_s_early == 48.0);
    assert(out.decode_tok_s_middle == 42.0);
    assert(out.decode_tok_s_late == 36.0);
    assert(out.retained_token_count == 8192);
    assert(out.retained_prefix_hash == ((UINT64_MAX - 7) & 0x7fffffffffffffffULL));

    printf("OK\n");
}

static void test_json_config_retained_roundtrip() {
    printf("  json_config_retained_roundtrip ... ");

    llama_decode_config cfg = {};
    cfg.stop_words = {"</tool_call>", "done"};
    cfg.max_tokens = 321;
    cfg.prompt_token_count = 77;
    cfg.prompt_hash = 0x12345678;
    cfg.reuse_retained_count = 4096;
    cfg.reuse_retained_prefix_hash = UINT64_MAX - 3;

    const std::string json = llama_decode_config_to_json(cfg);
    llama_decode_config out = {};
    assert(llama_decode_config_from_json(json, out));
    assert(out.stop_words == cfg.stop_words);
    assert(out.max_tokens == cfg.max_tokens);
    assert(out.prompt_token_count == cfg.prompt_token_count);
    assert(out.prompt_hash == cfg.prompt_hash);
    assert(out.reuse_retained_count == cfg.reuse_retained_count);
    assert(out.reuse_retained_prefix_hash ==
           ((UINT64_MAX - 3) & 0x7fffffffffffffffULL));

    llama_decode_config gate_off = {};
    const std::string gate_off_json = llama_decode_config_to_json(gate_off);
    assert(gate_off_json.find("reuse_retained_count") == std::string::npos);
    assert(gate_off_json.find("reuse_retained_prefix_hash") == std::string::npos);

    printf("OK\n");
}

static void test_decode_timing_first_to_last() {
    printf("  decode_timing_first_to_last ... ");

    // Seven emitted tokens define six decode intervals. The intervals deliberately
    // slow from 100/200 ms to 500/600 ms so the thirds expose the decay.
    const std::vector<int64_t> emitted_us = {
        1000000, 1100000, 1300000, 1600000, 2000000, 2500000, 3100000,
    };
    const llama_sustained_decode_timing timing =
        llama_sustained_decode_timing_from_timestamps(emitted_us);

    assert(timing.emitted_tokens == 7);
    assert(timing.decode_intervals == 6);
    assert(timing.elapsed_ms > 2099.0 && timing.elapsed_ms < 2101.0);
    assert(timing.tok_s > 2.85 && timing.tok_s < 2.86);       // 6 / 2.1 s
    assert(timing.early_tok_s > 6.66 && timing.early_tok_s < 6.67);   // 2 / 0.3 s
    assert(timing.middle_tok_s > 2.85 && timing.middle_tok_s < 2.86); // 2 / 0.7 s
    assert(timing.late_tok_s > 1.81 && timing.late_tok_s < 1.82);     // 2 / 1.1 s

    const llama_sustained_decode_timing one =
        llama_sustained_decode_timing_from_timestamps({123456});
    assert(one.emitted_tokens == 1);
    assert(one.decode_intervals == 0);
    assert(one.elapsed_ms == 0.0);
    assert(one.tok_s == 0.0);

    printf("OK\n");
}

static void test_json_handshake_roundtrip() {
    printf("  json_handshake_roundtrip ... ");

    llama_token_handshake hs = {};
    hs.version = 1;
    hs.status  = "ready";
    hs.model_fingerprint = 0xDEADBEEFCAFE;
    hs.token_count = 4096;

    std::string json = llama_token_handshake_to_json(hs);

    llama_token_handshake out = {};
    assert(llama_token_handshake_from_json(json, out));
    assert(out.version == 1);
    assert(out.status == "ready");
    assert(out.model_fingerprint == 0xDEADBEEFCAFE);
    assert(out.token_count == 4096);

    printf("OK\n");
}

static void test_json_text_escaping() {
    printf("  json_text_escaping ... ");

    llama_token_msg msg = {};
    msg.token_id = 1;
    msg.text = "line1\nline2\t\"quoted\"\\back";
    msg.pos = 0;

    std::string json = llama_token_msg_to_json(msg);
    llama_token_msg out = {};
    assert(llama_token_msg_from_json(json, out));
    assert(out.text == msg.text);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// TCP server/client integration tests
// ---------------------------------------------------------------------------

static void test_server_client_basic() {
    printf("  server_client_basic ... ");

    llama_token_stream_server server;
    assert(server.start("127.0.0.1", 0));  // port 0 = OS-assigned
    int port = server.bound_port();
    assert(port > 0);

    // client connects in a thread
    std::vector<llama_token_msg> received;
    std::thread client_thread([port, &received]() {
        llama_token_stream_client client;
        assert(client.connect("127.0.0.1", port, 5, 100));

        llama_token_handshake hs = {};
        assert(client.read_handshake(hs, 5000));
        assert(hs.version == 1);
        assert(hs.status == "ready");
        assert(hs.token_count == 100);

        for (;;) {
            llama_token_msg msg = {};
            assert(client.read_msg(msg, 5000));
            if (msg.done) break;
            received.push_back(msg);
        }
        client.disconnect();
    });

    assert(server.wait_for_client(5000));

    llama_token_handshake hs = {};
    hs.version = 1;
    hs.status  = "ready";
    hs.model_fingerprint = 0;
    hs.token_count = 100;
    assert(server.send_handshake(hs));

    // enqueue tokens
    for (int i = 0; i < 10; i++) {
        llama_token_msg msg = {};
        msg.token_id = i;
        msg.text = "tok" + std::to_string(i);
        msg.pos = 100 + i;
        assert(server.enqueue(msg));
    }

    assert(server.send_done(10, 50.0));

    client_thread.join();
    server.stop();

    assert(received.size() == 10);
    for (int i = 0; i < 10; i++) {
        assert(received[i].token_id == i);
        assert(received[i].text == "tok" + std::to_string(i));
    }

    printf("OK\n");
}

static void test_server_client_sustained_timing() {
    printf("  server_client_sustained_timing ... ");

    llama_token_stream_server server;
    assert(server.start("127.0.0.1", 0));
    const int port = server.bound_port();
    llama_token_msg done = {};

    std::thread client_thread([port, &done]() {
        llama_token_stream_client client;
        assert(client.connect("127.0.0.1", port, 5, 100));
        llama_token_handshake hs = {};
        assert(client.read_handshake(hs, 5000));
        for (;;) {
            llama_token_msg msg = {};
            assert(client.read_msg(msg, 5000));
            if (msg.done) {
                done = msg;
                break;
            }
        }
        client.disconnect();
    });

    assert(server.wait_for_client(5000));
    llama_token_handshake hs = {1, "ready", 0, 4};
    assert(server.send_handshake(hs));
    const llama_sustained_decode_timing timing =
        llama_sustained_decode_timing_from_timestamps({1000000, 1100000, 1300000, 1600000});
    assert(server.send_done(4, timing, LLAMA_STOP_LIMIT));

    client_thread.join();
    server.stop();
    assert(done.done);
    assert(done.tokens_generated == 4);
    assert(done.decode_intervals == 3);
    assert(done.decode_elapsed_ms > 599.0 && done.decode_elapsed_ms < 601.0);
    assert(done.decode_tok_s > 4.99 && done.decode_tok_s < 5.01);
    assert(done.decode_tok_s_early > 9.99 && done.decode_tok_s_early < 10.01);
    assert(done.decode_tok_s_middle > 4.99 && done.decode_tok_s_middle < 5.01);
    assert(done.decode_tok_s_late > 3.33 && done.decode_tok_s_late < 3.34);
    assert(done.stop_reason == LLAMA_STOP_LIMIT);

    printf("OK\n");
}

static void test_server_client_large_burst() {
    printf("  server_client_large_burst ... ");

    llama_token_stream_server server;
    assert(server.start("127.0.0.1", 0));
    int port = server.bound_port();

    const int N = 2000;
    std::vector<llama_token_msg> received;

    std::thread client_thread([port, &received, N]() {
        llama_token_stream_client client;
        assert(client.connect("127.0.0.1", port, 5, 100));

        llama_token_handshake hs = {};
        assert(client.read_handshake(hs, 5000));

        for (;;) {
            llama_token_msg msg = {};
            assert(client.read_msg(msg, 10000));
            if (msg.done) break;
            received.push_back(msg);
        }
        client.disconnect();
    });

    assert(server.wait_for_client(5000));

    llama_token_handshake hs = {1, "ready", 0, 0};
    assert(server.send_handshake(hs));

    for (int i = 0; i < N; i++) {
        llama_token_msg msg = {};
        msg.token_id = i;
        msg.text = "t" + std::to_string(i);
        msg.pos = i;
        server.enqueue(msg);
    }
    server.send_done(N, 100.0);

    client_thread.join();
    server.stop();

    assert((int) received.size() == N);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    printf("test-token-stream: ring buffer tests\n");
    test_ring_basic();
    test_ring_fifo_order();
    test_ring_overflow();
    test_ring_wrap_around();

    printf("test-token-stream: JSON serialization tests\n");
    test_json_token_roundtrip();
    test_json_done_roundtrip();
    test_decode_timing_first_to_last();
    test_json_config_retained_roundtrip();
    test_json_handshake_roundtrip();
    test_json_text_escaping();

    printf("test-token-stream: TCP server/client tests\n");
    test_server_client_basic();
    test_server_client_sustained_timing();
    test_server_client_large_burst();

    printf("\ntest-token-stream: all tests OK\n");
    return 0;
}
