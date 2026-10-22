#pragma once

#include "llama-tb-transport.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

enum llama_decode_executor_kind {
    LLAMA_DECODE_EXECUTOR_LOCAL_CPU = 0,
    LLAMA_DECODE_EXECUTOR_LOCAL_GPU = 1,
    LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID = 2,
    LLAMA_DECODE_EXECUTOR_SPLIT_THUNDERBOLT = 3,
    LLAMA_DECODE_EXECUTOR_WIR_WAVEFRONT    = 4,
    LLAMA_DECODE_EXECUTOR_WIR_LOCAL        = 5,
    LLAMA_DECODE_EXECUTOR_STREAM_LOCAL     = 6,
};

struct llama_decode_layer_map_entry {
    std::string node;
    int32_t     layer_start = 0;  // inclusive
    int32_t     layer_end   = 0;  // exclusive
};

struct llama_decode_handoff_runtime {
    int32_t n_layers = 0;

    bool offload_kqv     = false;
    bool has_gpu_backend = false;
    int32_t model_gpu_layers = 0;

    int32_t decode_mode        = LLAMA_PREFILL_DECODE_MODE_AUTO;
    int32_t gpu_layers_hint    = -1;  // -1 = auto
    int32_t remote_layers_hint = 0;   // 0 = no remote split
    int32_t remote_nodes_hint  = 1;
    std::string remote_ranges_hint;   // explicit per-node ranges: node:start-end,node:start-end
    std::string remote_failover_policy = "reroute"; // reroute|local|fail

    int32_t transport_mode = LLAMA_PREFILL_TRANSPORT_MODE_DISABLED;
    int32_t execution_mode = LLAMA_PREFILL_EXECUTION_MODE_COUPLED;
    int32_t tb_chunk_bytes = 4 * 1024 * 1024;
    std::string transport_session_dir;
    std::string kv_transport;
    std::string tb_direct_endpoint;
    std::string kv_host;
    int32_t kv_port = 0;
    int32_t kv_streams = 0;
    int32_t kv_stream_chunk_bytes = 0;
    int32_t kv_max_inflight_bytes = 0;
    int32_t kv_socket_send_buf = 0;
    int32_t kv_socket_recv_buf = 0;
    std::string kv_bind_addrs;
    std::string kv_peer_addrs;
    std::string kv_balance;
    bool kv_transport_fallback = false;

    bool transport_required = false;

    // Token relay + KV sync (Phase 3)
    std::string token_stream_host;
    int32_t     token_stream_port      = 0;
    int32_t     kv_sync_batch_size     = 512;

    // Stop words for remote decode
    std::vector<std::string> stop_words;
    int32_t max_tokens = -1;
    uint32_t prompt_token_count = 0;
    uint64_t prompt_hash = 0;
    // Option B cross-turn retained-KV: when > 0, ship only the delta and tell the Mac to
    // reuse the first N tokens of its retained cache as the prefix (see llama_decode_config).
    uint32_t reuse_retained_count = 0;
    uint64_t reuse_retained_prefix_hash = 0;
};

struct llama_decode_handoff_plan {
    int32_t requested_mode = LLAMA_PREFILL_DECODE_MODE_AUTO;
    int32_t resolved_mode  = LLAMA_PREFILL_DECODE_MODE_AUTO;

    llama_decode_executor_kind executor_kind = LLAMA_DECODE_EXECUTOR_LOCAL_CPU;

    bool use_transport      = false;
    bool transport_required = false;
    bool fallback_applied   = false;
    int32_t remote_nodes    = 1;
    int32_t expected_gpu_layers = 0;
    int32_t expected_remote_layers = 0;
    int32_t transport_mode  = LLAMA_PREFILL_TRANSPORT_MODE_DISABLED;
    int32_t execution_mode  = LLAMA_PREFILL_EXECUTION_MODE_COUPLED;
    std::string remote_ranges;
    std::string remote_failover_policy = "reroute";
    int32_t tb_chunk_bytes  = 4 * 1024 * 1024;
    std::string transport_session_dir;
    std::string kv_transport;
    std::string tb_direct_endpoint;
    std::string kv_host;
    int32_t kv_port = 0;
    int32_t kv_streams = 0;
    int32_t kv_stream_chunk_bytes = 0;
    int32_t kv_max_inflight_bytes = 0;
    int32_t kv_socket_send_buf = 0;
    int32_t kv_socket_recv_buf = 0;
    std::string kv_bind_addrs;
    std::string kv_peer_addrs;
    std::string kv_balance;
    bool kv_transport_fallback = false;

    std::string fallback_reason;
    std::vector<llama_decode_layer_map_entry> layer_map;

    // Token relay + KV sync (Phase 3)
    std::string token_stream_host;
    int32_t     token_stream_port      = 0;
    int32_t     kv_sync_batch_size     = 512;

    // Stop words for remote decode (antiprompt strings, e.g. tool call delimiters)
    std::vector<std::string> stop_words;
    int32_t max_tokens = -1;
    uint32_t prompt_token_count = 0;
    uint64_t prompt_hash = 0;
    // Option B cross-turn retained-KV reuse-count instruction (see llama_decode_config).
    uint32_t reuse_retained_count = 0;
    uint64_t reuse_retained_prefix_hash = 0;
};

struct llama_decode_publish_diag {
    bool        transport_used = false;
    std::string transport_backend = "disabled";
    bool        progressive = false;
    uint64_t    bytes_sent = 0;
    uint32_t    chunks_sent = 0;
    int32_t     stream_count = 1;
    int32_t     interface_count = 0;
    uint32_t    retransmit_chunks = 0;
    uint64_t    window_stalls_ms = 0;
    double      transfer_ms = 0.0;
    double      throughput_gbps = 0.0;
    uint64_t    first_chunk_send_unix_us = 0;
    uint64_t    last_chunk_send_unix_us = 0;
};

struct llama_decode_relay_result {
    int32_t     tokens_received = 0;
    int32_t     tokens_synced   = 0;
    double      relay_ms        = 0.0;
    double      decode_tok_s    = 0.0;   // reported by Mac
    double      decode_elapsed_ms = 0.0; // Mac first-emitted -> last-emitted interval
    int32_t     decode_intervals = 0;
    double      decode_tok_s_early = 0.0;
    double      decode_tok_s_middle = 0.0;
    double      decode_tok_s_late = 0.0;
    int32_t     stop_reason     = 0;     // llama_token_stop_reason from Mac
    std::string stopping_word;           // stop word matched (if stop_reason == LLAMA_STOP_WORD)
    std::string status;
    // Mac-retained-prefix handle (Option B). 0 = the Mac reported no retained cache handle.
    uint32_t    retained_token_count = 0; // Mac Phase-74 retained prefix length (prompt + generated)
    uint64_t    retained_prefix_hash = 0; // FNV-1a over the retained prefix token IDs
    bool        retained_prefix_miss = false;
};

// Callback for token relay — called for each token received from Mac.
// token_id: the generated token, text: detokenized text, done: generation complete.
// Return false to stop relay early (e.g., user cancelled).
using llama_decode_token_callback = std::function<bool(int32_t token_id, const std::string & text, bool done)>;

class llama_decode_executor_i {
  public:
    virtual ~llama_decode_executor_i() = default;

    virtual const char * name() const = 0;
    virtual bool         available() const = 0;
    virtual bool begin_session(const llama_decode_handoff_plan & plan, std::string & status) = 0;
    virtual bool publish_kv_artifact(const std::string & artifact_path, const llama_decode_handoff_plan & plan, std::string & status,
                                     llama_decode_publish_diag * diag = nullptr) = 0;

    // In-memory publish path: transport KV bytes to the receiver directly
    // from a host buffer, skipping the SSD round-trip. Default implementation
    // returns false (use the file-based path); the thunderbolt-split executor
    // overrides this to call llama_tb_transport_send_artifact_bytes.
    virtual bool publish_kv_artifact_bytes(const uint8_t * data, size_t size,
                                           const llama_decode_handoff_plan & plan, std::string & status,
                                           llama_decode_publish_diag * diag = nullptr) {
        (void) data; (void) size; (void) plan; (void) diag;
        status = "publish_kv_artifact_bytes not supported for this executor; fall back to file path";
        return false;
    }
    // Pull-based publish (2026-09-10): the artifact bytes are produced on demand (the NVMe tier's row
    // files), so nothing the size of the artifact exists in RAM. Default: unsupported.
    virtual bool publish_kv_artifact_stream(uint64_t total_bytes, const llama_tb_pull_fn & pull,
                                            const llama_decode_handoff_plan & plan, std::string & status,
                                            llama_decode_publish_diag * diag = nullptr) {
        (void) total_bytes; (void) pull; (void) plan; (void) diag;
        status = "publish_kv_artifact_stream not supported for this executor";
        return false;
    }
    // Delta shipping (2026-09-10): how many leading tokens of this sequence the decoder already
    // holds resident. Default: unknown (ship everything).
    virtual bool query_decoder_prefix(const int32_t * tokens, size_t n_tokens,
                                      const llama_decode_handoff_plan & plan,
                                      uint32_t & cached_len, std::string & status) {
        (void) tokens; (void) n_tokens; (void) plan;
        cached_len = 0;
        status = "query_decoder_prefix not supported for this executor";
        return false;
    }

    // After publish, relay decoded tokens from Mac and optionally sync local KV.
    // ctx may be nullptr to skip KV sync. output may be nullptr to skip display.
    virtual llama_decode_relay_result relay_tokens(
        const llama_decode_handoff_plan & plan,
        struct llama_context * ctx,
        FILE * output) {
        (void)plan; (void)ctx; (void)output;
        llama_decode_relay_result r; r.status = "relay not supported for this executor"; return r;
    }

    // Callback-based relay: routes tokens through a callback instead of FILE*.
    // Used by llama-server to send tokens to HTTP clients via SSE.
    virtual llama_decode_relay_result relay_tokens_cb(
        const llama_decode_handoff_plan & plan,
        struct llama_context * ctx,
        llama_decode_token_callback token_cb) {
        // Default implementation: use the callback variant by wrapping relay_tokens
        (void)plan; (void)ctx; (void)token_cb;
        llama_decode_relay_result r; r.status = "relay_tokens_cb not supported for this executor"; return r;
    }
};

llama_decode_handoff_plan llama_decode_handoff_build_plan(const llama_decode_handoff_runtime & runtime);

std::unique_ptr<llama_decode_executor_i>
llama_decode_executor_create(const llama_decode_handoff_plan & plan, std::string * fallback_reason);

const char * llama_decode_executor_kind_name(llama_decode_executor_kind kind);
const char * llama_prefill_decode_mode_name(int32_t mode);

std::string llama_decode_handoff_plan_to_string(const llama_decode_handoff_plan & plan);
