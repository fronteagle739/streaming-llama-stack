#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Token message — a single decoded token or a "done" sentinel.
// ---------------------------------------------------------------------------

// Stop reasons for decode completion
enum llama_token_stop_reason {
    LLAMA_STOP_NONE      = 0,  // still generating
    LLAMA_STOP_EOS       = 1,  // end-of-generation token
    LLAMA_STOP_LIMIT     = 2,  // hit max_tokens / n_predict
    LLAMA_STOP_WORD      = 3,  // hit a stop word (e.g. tool call delimiter)
    LLAMA_STOP_CANCELLED = 4,  // client disconnected
};

// Decoder-only throughput measured across emitted-token intervals. For N emitted
// tokens there are N-1 decode intervals: first emission -> last emission. This
// deliberately excludes model load, prefill, KV transfer/import, TTFT and relay
// teardown. The thirds are non-overlapping contiguous interval ranges.
struct llama_sustained_decode_timing {
    int32_t emitted_tokens   = 0;
    int32_t decode_intervals = 0;
    double  elapsed_ms       = 0.0;
    double  tok_s            = 0.0;
    double  early_tok_s      = 0.0;
    double  middle_tok_s     = 0.0;
    double  late_tok_s       = 0.0;
};

llama_sustained_decode_timing llama_sustained_decode_timing_from_timestamps(
    const std::vector<int64_t> & emitted_timestamps_us);

struct llama_token_msg {
    int32_t  token_id       = 0;
    std::string text;
    int32_t  pos            = 0;
    int64_t  timestamp_us   = 0;
    bool     done           = false;
    int32_t  tokens_generated = 0;    // only meaningful when done==true
    double   decode_tok_s   = 0.0;    // strict first-emitted -> last-emitted throughput when done==true
    double   decode_elapsed_ms = 0.0; // strict first-emitted -> last-emitted interval
    int32_t  decode_intervals = 0;    // tokens_generated - 1 when at least two tokens were emitted
    double   decode_tok_s_early = 0.0;
    double   decode_tok_s_middle = 0.0;
    double   decode_tok_s_late = 0.0;
    int32_t  dropped_count  = 0;      // tokens dropped due to ring overflow
    int32_t  stop_reason    = 0;      // only meaningful when done==true (llama_token_stop_reason)
    std::string stopping_word;        // only meaningful when stop_reason==LLAMA_STOP_WORD
    // --- Mac-retained-prefix handle (Option B, cross-turn retained-KV; only meaningful when done==true) ---
    // After a PROMPT-DIRECT decode turn, the Mac reports the token prefix it now holds in its
    // Phase-74 retained cache (mlx_token_stream_server.py _cache_with_reuse/_save_retained_state).
    // The RTX host records this as a retained-KV candidate so the NEXT turn can route a cached-
    // prefix continuation (send only the delta) instead of re-shipping the whole growing prompt.
    // Purely additive: 0 => the Mac did not report a handle (legacy receiver) => today's behavior.
    uint32_t retained_token_count = 0;   // # tokens the Mac retains (prompt + generated response)
    uint64_t retained_prefix_hash = 0;   // FNV-1a over those token IDs (matches _kva_prompt_count_hash)
};

// ---------------------------------------------------------------------------
// Decode config — sent by RTX client to Mac server before handshake.
// Contains stop words and other per-request decode parameters.
// ---------------------------------------------------------------------------

struct llama_decode_config {
    std::vector<std::string> stop_words;  // antiprompt strings (e.g. "</tool_call>")
    int32_t max_tokens = -1;              // per-request decode cap; -1 means receiver default
    uint32_t prompt_token_count = 0;       // full prompt token count for KVA/config matching
    uint64_t prompt_hash = 0;              // FNV-1a over int32 LE prompt tokens
    // Option B cross-turn retained-KV (LLAMA_DSV4_MAC_RETAIN_HANDLE). When > 0 the RTX has
    // shipped ONLY the delta continuation tokens for this turn; the Mac must REUSE the first
    // N tokens of its Phase-74 retained cache as the prefix and treat the incoming tokens as
    // the continuation appended after them (do NOT run the LCP heuristic — the delta prompt
    // would score LCP~0 and reset). 0 => today's behavior (full prompt / LCP reuse).
    uint32_t reuse_retained_count = 0;
    // Expected FNV-1a hash for retained_tokens[:reuse_retained_count]. Kept to
    // 63 bits so it round-trips through signed JSON integer parsers.
    uint64_t reuse_retained_prefix_hash = 0;
};

// ---------------------------------------------------------------------------
// Version handshake — first message on the token stream connection.
// ---------------------------------------------------------------------------

struct llama_token_handshake {
    int32_t  version          = 1;
    std::string status;                  // "ready", "error", ...
    uint64_t model_fingerprint = 0;
    uint32_t token_count       = 0;     // prompt tokens loaded in KV
};

// ---------------------------------------------------------------------------
// JSON serialization (one JSON line per message, '\n' terminated).
// ---------------------------------------------------------------------------

std::string llama_token_msg_to_json(const llama_token_msg & msg);
bool        llama_token_msg_from_json(const std::string & json, llama_token_msg & msg);

std::string llama_token_handshake_to_json(const llama_token_handshake & hs);
bool        llama_token_handshake_from_json(const std::string & json, llama_token_handshake & hs);

std::string llama_decode_config_to_json(const llama_decode_config & cfg);
bool        llama_decode_config_from_json(const std::string & json, llama_decode_config & cfg);

// ---------------------------------------------------------------------------
// 16k ring buffer — lock-free-ish SPSC with mutex for simplicity.
// Push never blocks. When full, overwrites the oldest entry.
// ---------------------------------------------------------------------------

struct llama_token_ring {
    static constexpr size_t CAPACITY = 16384;

    llama_token_ring();

    // Push a message. Returns true normally, false if an overwrite occurred.
    bool push(const llama_token_msg & msg);

    // Pop a message. Returns false if empty.
    bool pop(llama_token_msg & msg);

    size_t  size() const;
    int32_t total_dropped() const;

private:
    std::vector<llama_token_msg> buf_;
    size_t  head_    = 0;   // next write position
    size_t  tail_    = 0;   // next read position
    size_t  count_   = 0;
    int32_t dropped_ = 0;
    mutable std::mutex mtx_;
};

// ---------------------------------------------------------------------------
// Server (Mac-side): accepts one TCP client, sends tokens via sender thread.
// ---------------------------------------------------------------------------

struct llama_token_stream_server {
    // Bind and listen. port==0 → OS-assigned.
    bool start(const std::string & host, int port);

    // Block until a client connects or timeout.
    bool wait_for_client(int timeout_ms = 30000);

    // Read decode config from client (stop words, etc). Call after wait_for_client.
    bool read_config(llama_decode_config & cfg, int timeout_ms = 10000);

    // Send the version handshake line.
    bool send_handshake(const llama_token_handshake & hs);

    // Enqueue a token into the ring buffer. The sender thread drains it.
    bool enqueue(const llama_token_msg & msg);

    // Convenience: enqueue a "done" message and flush.
    bool send_done(int32_t tokens_generated, double decode_tok_s,
                   int32_t stop_reason = LLAMA_STOP_EOS,
                   const std::string & stopping_word = "");
    bool send_done(int32_t tokens_generated, const llama_sustained_decode_timing & timing,
                   int32_t stop_reason = LLAMA_STOP_EOS,
                   const std::string & stopping_word = "");

    // Shut down listener, sender thread, close connections.
    void stop();

    // RAII, because the listening socket now outlives more code than it used to.
    //
    // The listener is bound at the START of a decode session (before the KV import, so the RTX
    // client does not poll a port that does not exist yet -- see decode-receiver/main.cpp). That
    // put a dozen `return -1` paths between the bind and the matching stop(), and a leaked
    // listener is not a quiet leak: the NEXT session's start() fails with EADDRINUSE, so one
    // failed import would take every subsequent turn down with it.
    //
    // stop() is idempotent and sentinel-guarded (SOCKET_INVALID / joinable()), so this is safe on
    // a server that was never started, and safe after an explicit stop() -- the call sites that
    // already stop() deliberately are left exactly as they are.
    ~llama_token_stream_server() { stop(); }

    // Check if the relay client is still connected (for cancel detection).
    bool is_client_connected() const;

    int bound_port() const;

private:
#ifdef _WIN32
    uintptr_t listen_fd_ = ~(uintptr_t)0;
    uintptr_t client_fd_ = ~(uintptr_t)0;
#else
    int  listen_fd_ = -1;
    int  client_fd_ = -1;
#endif
    int  port_      = 0;
    std::atomic<bool> running_{false};
    std::atomic<bool> done_queued_{false};
    llama_token_ring  ring_;
    std::thread       sender_;

    void sender_loop();
    bool send_line(const std::string & line);
};

// ---------------------------------------------------------------------------
// Client (RTX-side): connects to a token stream server and reads messages.
// ---------------------------------------------------------------------------

struct llama_token_stream_client {
    bool connect(const std::string & host, int port,
                 int retries = 5, int retry_ms = 500);

    // Send decode config to server (stop words, etc). Call after connect, before read_handshake.
    bool send_config(const llama_decode_config & cfg);

    bool read_handshake(llama_token_handshake & hs, int timeout_ms = 10000);
    bool read_msg(llama_token_msg & msg, int timeout_ms = 60000);

    void disconnect();

private:
#ifdef _WIN32
    uintptr_t fd_ = ~(uintptr_t)0;
#else
    int fd_ = -1;
#endif
    std::string buf_;   // accumulation buffer for partial reads

    bool read_line(std::string & line, int timeout_ms);
};
