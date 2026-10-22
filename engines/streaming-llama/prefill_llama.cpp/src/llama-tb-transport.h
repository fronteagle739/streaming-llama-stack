#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// xxhash64 of int32_t token array — implemented in llama-tb-transport.cpp.
// Used by decode-receiver (which compiles llama-tb-transport.cpp directly)
// for prefix cache snapshot insertion after local prefill.
uint64_t llama_tb_transport_hash_tokens(const int32_t * tokens, int32_t n_tokens);

// 8-byte fingerprint of model description string.
uint64_t llama_tb_transport_model_fingerprint(const char * model_desc);

struct llama_tb_transfer_options {
    std::string session_dir;
    std::string session_id;
    std::string endpoint;
    std::string transport_mode;
    std::string kv_host;
    int32_t     kv_port = 0;
    std::string kv_bind_addrs;
    std::string kv_peer_addrs;
    std::string kv_balance;
    bool        transport_fallback = false;
    std::string execution_mode;
    size_t      chunk_bytes = 4 * 1024 * 1024;
    size_t      max_inflight_bytes = 256 * 1024 * 1024;
    int32_t     stream_count = 1;
    int32_t     socket_send_buf = 0;
    int32_t     socket_recv_buf = 0;
    bool        progressive = false;
    int32_t     remote_nodes = 1;
};

struct llama_tb_transfer_result {
    std::string session_path;
    std::string transport_mode = "disabled";
    std::string transport_backend = "filesystem";
    uint64_t    bytes_sent = 0;
    uint32_t    chunks_sent = 0;
    bool        progressive = false;
    int32_t     stream_count = 1;
    int32_t     interface_count = 0;
    uint32_t    retransmit_chunks = 0;
    uint64_t    window_stalls_ms = 0;
    double      transfer_ms = 0.0;
    double      throughput_gbps = 0.0;
    uint64_t    first_chunk_send_unix_us = 0;
    uint64_t    last_chunk_send_unix_us = 0;
};

bool llama_tb_transport_enabled();

bool llama_tb_transport_send_artifact(const std::string &          artifact_path,
                                      const llama_tb_transfer_options & options,
                                      llama_tb_transfer_result *    result,
                                      std::string *                 error);

// In-memory variant of send_artifact: streams the artifact bytes from a
// caller-provided buffer over TCP/RDMA without touching the SSD. Used by
// the disk-wear-protection path (avoids per-request 60-1200 MB writes that
// would otherwise hit the device on every prefill).
bool llama_tb_transport_send_artifact_bytes(const uint8_t *              data,
                                             size_t                       size,
                                             const llama_tb_transfer_options & options,
                                             llama_tb_transfer_result *   result,
                                             std::string *                error);

// Pull-based variant (2026-09-10, streamed export): the artifact bytes are produced on demand by
// `pull`, which fills dst with the next bytes (at most cap; n_out == 0 means finished) and returns
// false on error. total_bytes must be the exact stream length (it goes into SESSION_START and
// KV_SEGMENT_BEGIN). Single stream only: the receiver reassembles chunks in arrival order.
using llama_tb_pull_fn = std::function<bool(uint8_t * dst, size_t cap, size_t & n_out, std::string * err)>;
bool llama_tb_transport_send_artifact_stream(uint64_t                     total_bytes,
                                             const llama_tb_pull_fn &     pull,
                                             const llama_tb_transfer_options & options,
                                             llama_tb_transfer_result *   result,
                                             std::string *                error);

// Delta shipping (2026-09-10): ask the decode receiver (on the KV port) how many leading tokens of
// `tokens` its resident cache holds. One short TCP connection: HELLO, QUERY_PREFIX_CACHE (the int32
// token ids), PREFIX_CACHE_REPLY (hit u8, cached_len u32, session_id u64). cached_len is 0 on a miss.
// A receiver that does not understand the query never replies; the timeout then means "ship all".
bool llama_tb_transport_query_prefix(const llama_tb_transfer_options & options,
                                     const int32_t *              tokens,
                                     size_t                       n_tokens,
                                     int                          timeout_ms,
                                     uint32_t &                   cached_len,
                                     std::string *                error);

// Send raw prompt tokens to the decode receiver for local prefill.
// Used for short prompts where RTX streaming prefill is not beneficial.
// The receiver detects mode=prompt and runs local prefill instead of KV import.
bool llama_tb_transport_send_prompt(const int32_t *               tokens,
                                    int32_t                       n_tokens,
                                    const llama_tb_transfer_options & options,
                                    llama_tb_transfer_result *    result,
                                    std::string *                 error);

// Mac-side: PROMPT-DIRECT KV-handshake control socket listener.
// Binds on bind_host:control_port (kv_port - 1), accepts one connection per call,
// reads a TBP_MSG_QUERY_PREFIX_CACHE frame, calls the lookup callback, sends reply.
// Returns true if the round-trip completed (hit or miss).
// Designed to be called from a background thread in a loop.
//
// lookup_fn: callback that returns true on cache hit, fills cached_len and session_id.
using llama_tb_prefix_cache_lookup_fn = std::function<bool(uint64_t prefix_hash,
                                                            uint32_t prefix_len_tokens,
                                                            uint64_t model_fingerprint,
                                                            uint32_t & out_cached_len,
                                                            uint64_t & out_session_id)>;

bool llama_tb_transport_control_server_accept_one(
    const std::string &                   bind_host,
    int                                   control_port,
    llama_tb_prefix_cache_lookup_fn       lookup_fn,
    std::string *                         error);

// Phase 6 helper: copies through an explicit host staging hop to emulate remote slab relay.
void llama_tb_transport_weight_relay_copy(const uint8_t *src, uint8_t *dst, size_t size);

// Phase 6 helper: obtain a weight chunk from the configured remote source mode.
// Returns true and fills dst on success.
bool llama_tb_transport_fetch_weight_chunk(const uint8_t * src, size_t size, std::vector<uint8_t> & dst, std::string * error);
