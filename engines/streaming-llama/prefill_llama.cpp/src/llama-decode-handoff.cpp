#include "llama-decode-handoff.h"
#include "llama-tb-transport.h"
#include "llama-token-stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <set>
#include <sstream>
#include <utility>

static std::atomic<uint64_t> g_decode_handoff_session_nonce{0};

static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return std::max(lo, std::min(v, hi));
}

static bool is_env_truthy(const char * v) {
    if (!v) {
        return false;
    }

    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

static std::vector<int32_t> parse_csv_ints(const char * v) {
    std::vector<int32_t> out;
    if (!v || v[0] == '\0') {
        return out;
    }

    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        try {
            out.push_back((int32_t) std::stoi(tok));
        } catch (...) {
            // ignore malformed entries
        }
    }
    return out;
}

struct remote_range_spec {
    int32_t node  = 0;
    int32_t start = 0;
    int32_t end   = 0;
};

static std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

static std::string trim_copy(std::string value) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static std::string normalize_kv_transport_mode(std::string value) {
    value = to_lower_copy(trim_copy(std::move(value)));
    if (value.empty()) {
        return "";
    }
    if (value == "tb-direct") {
        return "rdma";
    }
    if (value == "tb-ethernet" || value == "ethernet") {
        return "tcp";
    }
    if (value == "auto" || value == "rdma" || value == "tcp" || value == "mixed" || value == "disabled") {
        return value;
    }
    return "";
}

static std::string normalize_failover_policy(const std::string & raw_value) {
    const std::string value = to_lower_copy(raw_value);
    if (value.empty()) {
        return "reroute";
    }
    if (value == "reroute" || value == "local" || value == "fail") {
        return value;
    }
    return "reroute";
}

static bool parse_remote_ranges_spec(
        const std::string & raw,
        std::vector<remote_range_spec> & out_specs,
        std::string & err) {
    out_specs.clear();
    if (raw.empty()) {
        return true;
    }

    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }

        const size_t colon = token.find(':');
        const size_t dash  = token.find('-', colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || dash == std::string::npos || dash <= colon + 1) {
            err = "invalid remote range token '" + token + "' (expected node:start-end)";
            return false;
        }

        try {
            const int32_t node  = (int32_t) std::stoi(token.substr(0, colon));
            const int32_t start = (int32_t) std::stoi(token.substr(colon + 1, dash - (colon + 1)));
            const int32_t end   = (int32_t) std::stoi(token.substr(dash + 1));
            if (node < 0) {
                err = "invalid remote node index in token '" + token + "'";
                return false;
            }
            if (end <= start) {
                err = "invalid remote range bounds in token '" + token + "'";
                return false;
            }
            out_specs.push_back({ node, start, end });
        } catch (...) {
            err = "invalid remote range token '" + token + "'";
            return false;
        }
    }

    return true;
}

static void push_span(std::vector<llama_decode_layer_map_entry> & layer_map,
                      const char *                                 node,
                      int32_t                                      start,
                      int32_t                                      end) {
    if (end <= start) {
        return;
    }
    layer_map.push_back({ node, start, end });
}

const char * llama_prefill_decode_mode_name(int32_t mode) {
    switch (mode) {
        case LLAMA_PREFILL_DECODE_MODE_AUTO:
            return "auto";
        case LLAMA_PREFILL_DECODE_MODE_CPU_KV:
            return "cpu_kv";
        case LLAMA_PREFILL_DECODE_MODE_GPU_KV:
            return "gpu_kv";
        case LLAMA_PREFILL_DECODE_MODE_HYBRID:
            return "hybrid";
        case LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT:
            return "split_thunderbolt";
        case LLAMA_PREFILL_DECODE_MODE_WIR_WAVEFRONT:
            return "wir_wavefront";
        case LLAMA_PREFILL_DECODE_MODE_WIR_LOCAL:
            return "wir_local";
        case LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL:
            return "stream_local";
        default:
            return "unknown";
    }
}

const char * llama_decode_executor_kind_name(llama_decode_executor_kind kind) {
    switch (kind) {
        case LLAMA_DECODE_EXECUTOR_LOCAL_CPU:
            return "local_cpu";
        case LLAMA_DECODE_EXECUTOR_LOCAL_GPU:
            return "local_gpu";
        case LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID:
            return "local_hybrid";
        case LLAMA_DECODE_EXECUTOR_SPLIT_THUNDERBOLT:
            return "split_thunderbolt";
        case LLAMA_DECODE_EXECUTOR_WIR_WAVEFRONT:
            return "wir_wavefront";
        case LLAMA_DECODE_EXECUTOR_WIR_LOCAL:
            return "wir_local";
        case LLAMA_DECODE_EXECUTOR_STREAM_LOCAL:
            return "stream_local";
        default:
            return "unknown";
    }
}

llama_decode_handoff_plan llama_decode_handoff_build_plan(const llama_decode_handoff_runtime & runtime) {
    llama_decode_handoff_plan plan;
    plan.requested_mode      = runtime.decode_mode;
    plan.resolved_mode       = runtime.decode_mode;
    plan.transport_required  = runtime.transport_required;
    plan.remote_nodes        = std::max(1, runtime.remote_nodes_hint);
    plan.remote_ranges       = runtime.remote_ranges_hint;
    plan.remote_failover_policy = normalize_failover_policy(runtime.remote_failover_policy);
    plan.transport_mode      = runtime.transport_mode;
    plan.execution_mode      = runtime.execution_mode;
    plan.tb_chunk_bytes      = runtime.tb_chunk_bytes > 0 ? runtime.tb_chunk_bytes : 4 * 1024 * 1024;
    plan.transport_session_dir = runtime.transport_session_dir;
    const std::string kv_transport_normalized = normalize_kv_transport_mode(runtime.kv_transport);
    plan.kv_transport        = kv_transport_normalized.empty() ? runtime.kv_transport : kv_transport_normalized;
    plan.tb_direct_endpoint  = runtime.tb_direct_endpoint;
    plan.kv_host             = runtime.kv_host;
    plan.kv_port             = runtime.kv_port;
    plan.kv_streams          = runtime.kv_streams;
    plan.kv_stream_chunk_bytes = runtime.kv_stream_chunk_bytes;
    plan.kv_max_inflight_bytes = runtime.kv_max_inflight_bytes;
    plan.kv_socket_send_buf  = runtime.kv_socket_send_buf;
    plan.kv_socket_recv_buf  = runtime.kv_socket_recv_buf;
    plan.kv_bind_addrs       = runtime.kv_bind_addrs;
    plan.kv_peer_addrs       = runtime.kv_peer_addrs;
    plan.kv_balance          = runtime.kv_balance;
    plan.kv_transport_fallback = runtime.kv_transport_fallback;
    plan.token_stream_host     = runtime.token_stream_host;
    plan.token_stream_port     = runtime.token_stream_port;
    plan.kv_sync_batch_size    = runtime.kv_sync_batch_size;
    plan.stop_words            = runtime.stop_words;
    plan.max_tokens            = runtime.max_tokens;
    plan.prompt_token_count    = runtime.prompt_token_count;
    plan.prompt_hash           = runtime.prompt_hash;
    plan.reuse_retained_count  = runtime.reuse_retained_count;
    plan.reuse_retained_prefix_hash = runtime.reuse_retained_prefix_hash & 0x7fffffffffffffffULL;
    const bool kv_transport_disabled = plan.kv_transport == "disabled";

    const int32_t n_layers = std::max(0, runtime.n_layers);
    const bool    can_gpu_backend = runtime.has_gpu_backend && runtime.offload_kqv;
    const bool    has_decode_gpu_residency = runtime.model_gpu_layers > 0;
    const bool    can_gpu = can_gpu_backend && has_decode_gpu_residency;

    if (plan.resolved_mode == LLAMA_PREFILL_DECODE_MODE_AUTO) {
        plan.resolved_mode = can_gpu ? LLAMA_PREFILL_DECODE_MODE_GPU_KV : LLAMA_PREFILL_DECODE_MODE_CPU_KV;
    }

    int32_t gpu_layers    = 0;
    int32_t remote_layers = 0;
    int32_t cpu_layers    = 0;
    auto pick_local_executor = [](int32_t gpu_layers_local, int32_t cpu_layers_local) {
        if (gpu_layers_local > 0 && cpu_layers_local > 0) {
            return LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID;
        }
        if (gpu_layers_local > 0) {
            return LLAMA_DECODE_EXECUTOR_LOCAL_GPU;
        }
        return LLAMA_DECODE_EXECUTOR_LOCAL_CPU;
    };

    switch (plan.resolved_mode) {
        case LLAMA_PREFILL_DECODE_MODE_CPU_KV:
            gpu_layers             = 0;
            remote_layers          = 0;
            cpu_layers             = n_layers;
            plan.executor_kind     = LLAMA_DECODE_EXECUTOR_LOCAL_CPU;
            plan.use_transport     = false;
            break;

        case LLAMA_PREFILL_DECODE_MODE_GPU_KV:
            gpu_layers             = can_gpu ? n_layers : 0;
            remote_layers          = 0;
            cpu_layers             = n_layers - gpu_layers;
            plan.executor_kind     = gpu_layers > 0 ? LLAMA_DECODE_EXECUTOR_LOCAL_GPU : LLAMA_DECODE_EXECUTOR_LOCAL_CPU;
            plan.use_transport     = false;
            if (!can_gpu && n_layers > 0) {
                plan.fallback_applied = true;
                plan.fallback_reason  = !can_gpu_backend
                    ? "gpu_kv requested but GPU decode backend is unavailable"
                    : "gpu_kv requested but model was loaded with n_gpu_layers=0; decode GPU residency is unavailable";
                plan.resolved_mode   = LLAMA_PREFILL_DECODE_MODE_CPU_KV;
            }
            break;

        case LLAMA_PREFILL_DECODE_MODE_HYBRID:
            gpu_layers             = can_gpu ? clamp_i32(runtime.gpu_layers_hint, 0, n_layers) : 0;
            if (can_gpu && runtime.gpu_layers_hint < 0) {
                // Hybrid auto mode must produce an actual split by default.
                // Reserve at least one layer on CPU and one on GPU when possible.
                if (n_layers <= 1) {
                    gpu_layers = n_layers;
                } else {
                    gpu_layers = clamp_i32(n_layers / 2, 1, n_layers - 1);
                }
            }
            remote_layers          = 0;
            cpu_layers             = n_layers - gpu_layers;
            plan.executor_kind     = pick_local_executor(gpu_layers, cpu_layers);
            plan.use_transport = false;
            if (!can_gpu && n_layers > 0) {
                plan.fallback_applied = true;
                plan.fallback_reason  = !can_gpu_backend
                    ? "hybrid requested but GPU decode backend is unavailable"
                    : "hybrid requested but model was loaded with n_gpu_layers=0; decode GPU residency is unavailable";
                plan.resolved_mode   = LLAMA_PREFILL_DECODE_MODE_CPU_KV;
                plan.executor_kind   = LLAMA_DECODE_EXECUTOR_LOCAL_CPU;
            }
            break;

        case LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT:
            remote_layers = clamp_i32(runtime.remote_layers_hint, 0, n_layers);
            if (can_gpu) {
                if (runtime.gpu_layers_hint >= 0) {
                    gpu_layers = clamp_i32(runtime.gpu_layers_hint, 0, n_layers);
                } else if (remote_layers > 0) {
                    gpu_layers = n_layers - remote_layers;
                } else {
                    // Prefer keeping all layers local unless caller explicitly requests a remote split.
                    gpu_layers = n_layers;
                }
            } else {
                gpu_layers = 0;
            }

            if (gpu_layers + remote_layers > n_layers) {
                remote_layers = n_layers - gpu_layers;
            }
            cpu_layers = n_layers - gpu_layers - remote_layers;
            if (cpu_layers < 0) {
                cpu_layers = 0;
            }

            if (!can_gpu && n_layers > 0 && gpu_layers > 0) {
                gpu_layers = 0;
                cpu_layers = n_layers - remote_layers;
                plan.fallback_applied = true;
                if (!plan.fallback_reason.empty()) {
                    plan.fallback_reason += "; ";
                }
                plan.fallback_reason += !can_gpu_backend
                    ? "split_thunderbolt requested but GPU decode backend is unavailable"
                    : "split_thunderbolt requested but model was loaded with n_gpu_layers=0; decode GPU residency is unavailable";
            }
            break;

        case LLAMA_PREFILL_DECODE_MODE_WIR_WAVEFRONT:
            // WIR mode: same transport as split_thunderbolt (send KV artifact to
            // remote Mac) but the Mac runs wavefront refinement instead of plain AR.
            // Layer split is identical to split_thunderbolt.
            remote_layers = clamp_i32(runtime.remote_layers_hint, 0, n_layers);
            if (remote_layers == 0) remote_layers = n_layers;  // default: all remote
            gpu_layers = 0;
            cpu_layers = n_layers - remote_layers;
            plan.use_transport = true;
            plan.executor_kind = LLAMA_DECODE_EXECUTOR_WIR_WAVEFRONT;
            break;

        case LLAMA_PREFILL_DECODE_MODE_WIR_LOCAL:
            // Local WIR: prefill on RTX, decode locally using WIR with a draft model.
            // No remote transport — everything stays on the local machine.
            // Uses GPU for both prefill and decode (model must fit in VRAM).
            gpu_layers = n_layers;
            cpu_layers = 0;
            remote_layers = 0;
            plan.use_transport = false;
            plan.executor_kind = LLAMA_DECODE_EXECUTOR_WIR_LOCAL;
            break;

        case LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL:
            // Local streaming decode re-drives the weight fetcher for small
            // autoregressive batches instead of requiring resident weights.
            gpu_layers = 0;
            cpu_layers = 0;
            remote_layers = 0;
            plan.use_transport = false;
            plan.executor_kind = LLAMA_DECODE_EXECUTOR_STREAM_LOCAL;
            break;

        case LLAMA_PREFILL_DECODE_MODE_AUTO:
        default:
            plan.resolved_mode   = can_gpu ? LLAMA_PREFILL_DECODE_MODE_GPU_KV : LLAMA_PREFILL_DECODE_MODE_CPU_KV;
            gpu_layers           = can_gpu ? n_layers : 0;
            remote_layers        = 0;
            cpu_layers           = n_layers - gpu_layers;
            plan.executor_kind   = gpu_layers > 0 ? LLAMA_DECODE_EXECUTOR_LOCAL_GPU : LLAMA_DECODE_EXECUTOR_LOCAL_CPU;
            plan.use_transport   = false;
            break;
    }

    auto append_fallback = [&](const std::string & reason) {
        if (reason.empty()) {
            return;
        }
        plan.fallback_applied = true;
        if (!plan.fallback_reason.empty()) {
            plan.fallback_reason += "; ";
        }
        plan.fallback_reason += reason;
    };

    struct remote_span {
        int32_t node  = 0;
        int32_t start = 0;
        int32_t end   = 0;
    };

    std::vector<remote_span> remote_spans;
    const bool split_remote_requested = plan.resolved_mode == LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT &&
        (runtime.remote_layers_hint > 0 || !plan.remote_ranges.empty());

    if (plan.resolved_mode == LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT &&
        (remote_layers > 0 || !plan.remote_ranges.empty())) {
        bool using_explicit_ranges = false;
        if (!plan.remote_ranges.empty()) {
            std::vector<remote_range_spec> parsed_ranges;
            std::string range_err;
            if (!parse_remote_ranges_spec(plan.remote_ranges, parsed_ranges, range_err)) {
                append_fallback("invalid remote ranges; using contiguous planner (" + range_err + ")");
            } else if (!parsed_ranges.empty()) {
                using_explicit_ranges = true;
                int32_t max_node = 0;
                for (const auto & range : parsed_ranges) {
                    max_node = std::max(max_node, range.node + 1);
                    remote_spans.push_back({ range.node, range.start, range.end });
                }
                plan.remote_nodes = std::max<int32_t>(plan.remote_nodes, std::max<int32_t>(1, max_node));

                if (runtime.gpu_layers_hint < 0) {
                    int32_t min_remote_start = n_layers;
                    for (const auto & range : parsed_ranges) {
                        min_remote_start = std::min(min_remote_start, range.start);
                    }
                    gpu_layers = clamp_i32(min_remote_start, 0, n_layers);
                }
            }
        }

        if (!using_explicit_ranges) {
            const int32_t remote_nodes = std::max(1, plan.remote_nodes);
            const auto node_weights_raw = parse_csv_ints(std::getenv("LLAMA_PREFILL_REMOTE_NODE_WEIGHTS"));
            std::vector<int32_t> node_weights(remote_nodes, 1);
            for (int32_t i = 0; i < remote_nodes && i < (int32_t) node_weights_raw.size(); ++i) {
                node_weights[i] = std::max<int32_t>(1, node_weights_raw[i]);
            }

            int32_t total_weight = 0;
            for (int32_t i = 0; i < remote_nodes; ++i) {
                total_weight += std::max<int32_t>(1, node_weights[i]);
            }

            int32_t cursor = clamp_i32(gpu_layers, 0, n_layers);
            int32_t remaining = clamp_i32(remote_layers, 0, n_layers - cursor);
            for (int32_t node_idx = 0; node_idx < remote_nodes && remaining > 0; ++node_idx) {
                int32_t span = 0;
                if (node_idx + 1 == remote_nodes) {
                    span = remaining;
                } else {
                    span = (int32_t) ((int64_t) remote_layers * node_weights[node_idx] / std::max<int32_t>(1, total_weight));
                    span = std::min(span, remaining);
                }
                if (span <= 0) {
                    continue;
                }
                remote_spans.push_back({ node_idx, cursor, cursor + span });
                cursor += span;
                remaining -= span;
            }
            if (remaining > 0 && !remote_spans.empty()) {
                remote_spans.push_back({ remote_spans.front().node, cursor, cursor + remaining });
            }
        }

        const auto unhealthy_nodes_raw = parse_csv_ints(std::getenv("LLAMA_PREFILL_TB_UNHEALTHY_NODES"));
        std::set<int32_t> unhealthy_nodes(unhealthy_nodes_raw.begin(), unhealthy_nodes_raw.end());

        if (!remote_spans.empty() && !unhealthy_nodes.empty()) {
            std::vector<int32_t> healthy_nodes;
            const int32_t node_bound = std::max<int32_t>(1, plan.remote_nodes);
            for (int32_t i = 0; i < node_bound; ++i) {
                if (unhealthy_nodes.find(i) == unhealthy_nodes.end()) {
                    healthy_nodes.push_back(i);
                }
            }

            // Explicit ranges can reference node indices beyond remote_nodes_hint.
            for (const auto & span : remote_spans) {
                if (span.node >= node_bound && unhealthy_nodes.find(span.node) == unhealthy_nodes.end()) {
                    healthy_nodes.push_back(span.node);
                }
            }
            std::sort(healthy_nodes.begin(), healthy_nodes.end());
            healthy_nodes.erase(std::unique(healthy_nodes.begin(), healthy_nodes.end()), healthy_nodes.end());

            if (plan.remote_failover_policy == "fail") {
                remote_spans.clear();
                append_fallback("remote node marked unhealthy and failover policy=fail; degrading split layers to local CPU");
            } else if (plan.remote_failover_policy == "local") {
                std::vector<remote_span> filtered;
                filtered.reserve(remote_spans.size());
                for (const auto & span : remote_spans) {
                    if (unhealthy_nodes.find(span.node) == unhealthy_nodes.end()) {
                        filtered.push_back(span);
                    }
                }
                if (filtered.size() != remote_spans.size()) {
                    append_fallback("remote node marked unhealthy and failover policy=local; degrading affected ranges to local CPU");
                }
                remote_spans = std::move(filtered);
            } else {
                if (healthy_nodes.empty()) {
                    remote_spans.clear();
                    append_fallback("all configured remote nodes marked unhealthy; degrading split layers to local CPU");
                } else {
                    size_t rr = 0;
                    bool rerouted = false;
                    for (auto & span : remote_spans) {
                        if (unhealthy_nodes.find(span.node) != unhealthy_nodes.end()) {
                            span.node = healthy_nodes[rr % healthy_nodes.size()];
                            rr++;
                            rerouted = true;
                        }
                    }
                    if (rerouted) {
                        append_fallback("rerouted unhealthy remote nodes across healthy remote peers");
                    }
                }
            }
        }
    }

    // Build per-layer ownership map.
    // owner: -2 local_gpu, -1 local_cpu, >=0 remote node index
    const int32_t gpu_prefix = clamp_i32(gpu_layers, 0, n_layers);
    std::vector<int32_t> owner((size_t) n_layers, -1);
    for (int32_t layer = 0; layer < gpu_prefix; ++layer) {
        owner[(size_t) layer] = -2;
    }

    bool clipped_remote_over_gpu = false;
    bool dropped_remote_overlap = false;
    for (const auto & span : remote_spans) {
        int32_t start = clamp_i32(span.start, 0, n_layers);
        int32_t end = clamp_i32(span.end, start, n_layers);
        if (end <= start) {
            continue;
        }

        for (int32_t layer = start; layer < end; ++layer) {
            if (owner[(size_t) layer] == -2) {
                clipped_remote_over_gpu = true;
                continue;
            }
            if (owner[(size_t) layer] >= 0) {
                dropped_remote_overlap = true;
                continue;
            }
            owner[(size_t) layer] = span.node;
        }
    }
    if (clipped_remote_over_gpu) {
        append_fallback("remote range overlapped local GPU prefix; clipped overlapping layers");
    }
    if (dropped_remote_overlap) {
        append_fallback("remote ranges overlapped; first assignment wins for overlapping layers");
    }

    // Emit contiguous layer map spans.
    plan.layer_map.clear();
    auto owner_name = [](int32_t layer_owner) {
        if (layer_owner == -2) {
            return std::string("local_gpu");
        }
        if (layer_owner == -1) {
            return std::string("local_cpu");
        }
        return std::string("tb_remote_") + std::to_string(layer_owner);
    };

    if (n_layers > 0) {
        int32_t start = 0;
        int32_t current_owner = owner[0];
        for (int32_t layer = 1; layer <= n_layers; ++layer) {
            const bool boundary = layer == n_layers || owner[(size_t) layer] != current_owner;
            if (!boundary) {
                continue;
            }
            const std::string node = owner_name(current_owner);
            push_span(plan.layer_map, node.c_str(), start, layer);
            start = layer;
            if (layer < n_layers) {
                current_owner = owner[(size_t) layer];
            }
        }
    }

    gpu_layers = 0;
    remote_layers = 0;
    cpu_layers = 0;
    std::set<int32_t> active_remote_nodes;
    for (int32_t layer_owner : owner) {
        if (layer_owner == -2) {
            gpu_layers++;
        } else if (layer_owner == -1) {
            cpu_layers++;
        } else {
            remote_layers++;
            active_remote_nodes.insert(layer_owner);
        }
    }
    if (!active_remote_nodes.empty()) {
        plan.remote_nodes = (int32_t) active_remote_nodes.size();
    }

    plan.expected_gpu_layers = gpu_layers;
    plan.expected_remote_layers = remote_layers;

    if (plan.resolved_mode == LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT) {
        plan.use_transport = remote_layers > 0 &&
                             plan.transport_mode != LLAMA_PREFILL_TRANSPORT_MODE_DISABLED &&
                             !kv_transport_disabled;
        if (plan.use_transport) {
            plan.executor_kind = LLAMA_DECODE_EXECUTOR_SPLIT_THUNDERBOLT;
        } else {
            plan.executor_kind = pick_local_executor(gpu_layers, cpu_layers);
            if (split_remote_requested) {
                if (remote_layers == 0) {
                    append_fallback("split_thunderbolt requested with no effective remote layers; using local decode");
                } else if (kv_transport_disabled) {
                    append_fallback("split_thunderbolt requested but kv transport is disabled; using local decode");
                } else {
                    append_fallback("split_thunderbolt requested but transport mode is disabled; using local decode");
                }
            }
        }
    } else {
        plan.use_transport = false;
        plan.expected_remote_layers = 0;
    }

    return plan;
}

// ---------------------------------------------------------------------------
// KV sync helper: batch-prefill received tokens into RTX's local KV cache.
// Returns number of tokens successfully synced.
// ---------------------------------------------------------------------------
static int32_t kv_sync_flush(struct llama_context * ctx,
                              const std::vector<int32_t> & token_ids,
                              int32_t start_pos) {
    if (!ctx || token_ids.empty()) return 0;

    const int32_t n = (int32_t) token_ids.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;

    for (int32_t i = 0; i < n; i++) {
        batch.token[i]    = (llama_token) token_ids[i];
        batch.pos[i]      = start_pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n - 1) ? 1 : 0;  // logits on last only
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        fprintf(stderr, "[kv-sync] batch prefill failed at pos %d (%d tokens)\n", start_pos, n);
        return 0;
    }
    return n;
}

class llama_decode_executor_local final : public llama_decode_executor_i {
  public:
    explicit llama_decode_executor_local(llama_decode_executor_kind kind) : kind_(kind) {}

    const char * name() const override { return llama_decode_executor_kind_name(kind_); }
    bool         available() const override { return true; }
    bool begin_session(const llama_decode_handoff_plan & plan, std::string & status) override {
        (void) plan;
        status = "local decode executor active";
        return true;
    }
    bool publish_kv_artifact(const std::string & artifact_path, const llama_decode_handoff_plan & plan, std::string & status,
                             llama_decode_publish_diag * diag = nullptr) override {
        (void) plan;
        if (diag) {
            *diag = llama_decode_publish_diag{};
            diag->transport_used = false;
            diag->transport_backend = "disabled";
        }
        status = "local executor retained KV artifact at " + artifact_path;
        return true;
    }

  private:
    llama_decode_executor_kind kind_;
};

class llama_decode_executor_tb_split final : public llama_decode_executor_i {
  public:
    const char * name() const override { return llama_decode_executor_kind_name(LLAMA_DECODE_EXECUTOR_SPLIT_THUNDERBOLT); }
    bool         available() const override { return llama_tb_transport_enabled(); }
    bool begin_session(const llama_decode_handoff_plan & plan, std::string & status) override {
        if (!available()) {
            status = "thunderbolt transport unavailable (set LLAMA_PREFILL_TB_ENABLE=1 to enable transport bridge)";
            return false;
        }
        status = "thunderbolt transport executor armed";
        (void) plan;
        return true;
    }

    static llama_tb_transfer_options build_opts(const llama_decode_handoff_plan & plan) {
        llama_tb_transfer_options opts;
        opts.progressive = plan.transport_mode == LLAMA_PREFILL_TRANSPORT_MODE_PROGRESSIVE;
        opts.chunk_bytes = plan.tb_chunk_bytes > 0 ? (size_t) plan.tb_chunk_bytes : 4 * 1024 * 1024;
        opts.remote_nodes = std::max(1, plan.remote_nodes);
        opts.execution_mode = plan.execution_mode == LLAMA_PREFILL_EXECUTION_MODE_DECOUPLED ? "decoupled" : "coupled";
        const long long session_ts_us = (long long) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const uint64_t session_nonce = g_decode_handoff_session_nonce.fetch_add(1, std::memory_order_relaxed);
        opts.session_id = "decode_handoff_" + opts.execution_mode + "_" + std::to_string(opts.remote_nodes) + "_" +
                          std::to_string(session_ts_us) + "_" + std::to_string(session_nonce);
        opts.session_dir = plan.transport_session_dir;
        opts.transport_mode = plan.kv_transport;
        opts.endpoint = plan.tb_direct_endpoint;
        opts.kv_host = plan.kv_host;
        opts.kv_port = plan.kv_port;
        opts.stream_count = plan.kv_streams;
        opts.chunk_bytes = plan.kv_stream_chunk_bytes > 0 ? (size_t) plan.kv_stream_chunk_bytes : opts.chunk_bytes;
        opts.max_inflight_bytes = plan.kv_max_inflight_bytes > 0 ? (size_t) plan.kv_max_inflight_bytes : opts.max_inflight_bytes;
        opts.socket_send_buf = plan.kv_socket_send_buf;
        opts.socket_recv_buf = plan.kv_socket_recv_buf;
        opts.kv_bind_addrs = plan.kv_bind_addrs;
        opts.kv_peer_addrs = plan.kv_peer_addrs;
        opts.kv_balance = plan.kv_balance;
        opts.transport_fallback = plan.kv_transport_fallback;
        return opts;
    }

    static void fill_diag_and_status(const llama_tb_transfer_result & res,
                                     const llama_decode_handoff_plan & plan,
                                     std::string & status,
                                     llama_decode_publish_diag * diag) {
        if (diag) {
            *diag = llama_decode_publish_diag{};
            diag->transport_used = true;
            diag->transport_backend = !res.transport_backend.empty()
                                          ? res.transport_backend
                                          : (!res.transport_mode.empty()
                                                 ? res.transport_mode
                                                 : (plan.kv_transport.empty() ? "auto" : plan.kv_transport));
            diag->progressive = res.progressive;
            diag->bytes_sent = res.bytes_sent;
            diag->chunks_sent = res.chunks_sent;
            diag->stream_count = res.stream_count;
            diag->interface_count = res.interface_count;
            diag->retransmit_chunks = res.retransmit_chunks;
            diag->window_stalls_ms = res.window_stalls_ms;
            diag->transfer_ms = res.transfer_ms;
            diag->throughput_gbps = res.throughput_gbps;
            diag->first_chunk_send_unix_us = res.first_chunk_send_unix_us;
            diag->last_chunk_send_unix_us = res.last_chunk_send_unix_us;
        }

        std::ostringstream oss;
        oss << "thunderbolt transfer complete: " << res.session_path
            << " (" << res.bytes_sent << " bytes, " << res.chunks_sent << " chunks, mode="
            << (res.progressive ? "progressive" : "bulk")
            << ", transport_backend=" << (!res.transport_backend.empty()
                                              ? res.transport_backend
                                              : (!res.transport_mode.empty()
                                                     ? res.transport_mode
                                                     : (plan.kv_transport.empty() ? "auto" : plan.kv_transport)))
            << ", transport_mode=" << (res.transport_mode.empty() ? "auto" : res.transport_mode)
            << ", transfer_ms=" << res.transfer_ms
            << ", throughput_gbps=" << res.throughput_gbps
            << ", stream_count=" << res.stream_count
            << ", interface_count=" << res.interface_count
            << ", retransmit_chunks=" << res.retransmit_chunks
            << ", window_stalls_ms=" << res.window_stalls_ms
            << ", first_chunk_send_unix_us=" << res.first_chunk_send_unix_us
            << ", last_chunk_send_unix_us=" << res.last_chunk_send_unix_us
            << ")";
        status = oss.str();
    }

    bool publish_kv_artifact(const std::string & artifact_path, const llama_decode_handoff_plan & plan, std::string & status,
                             llama_decode_publish_diag * diag = nullptr) override {
        const llama_tb_transfer_options opts = build_opts(plan);
        llama_tb_transfer_result res;
        std::string err;
        if (!llama_tb_transport_send_artifact(artifact_path, opts, &res, &err)) {
            status = err.empty() ? "thunderbolt publish failed" : err;
            return false;
        }
        if (res.transport_backend == "filesystem" || res.transport_mode == "disabled") {
            status = "transport publish did not use rdma/tcp backend (resolved mode=" +
                     (res.transport_mode.empty() ? std::string("auto") : res.transport_mode) + ")";
            return false;
        }
        fill_diag_and_status(res, plan, status, diag);
        return true;
    }

    bool publish_kv_artifact_bytes(const uint8_t * data, size_t size,
                                   const llama_decode_handoff_plan & plan, std::string & status,
                                   llama_decode_publish_diag * diag = nullptr) override {
        const llama_tb_transfer_options opts = build_opts(plan);
        llama_tb_transfer_result res;
        std::string err;
        if (!llama_tb_transport_send_artifact_bytes(data, size, opts, &res, &err)) {
            status = err.empty() ? "thunderbolt publish (bytes) failed" : err;
            return false;
        }
        if (res.transport_backend == "filesystem" || res.transport_mode == "disabled") {
            status = "transport publish did not use rdma/tcp backend (resolved mode=" +
                     (res.transport_mode.empty() ? std::string("auto") : res.transport_mode) + ")";
            return false;
        }
        fill_diag_and_status(res, plan, status, diag);
        return true;
    }
    bool query_decoder_prefix(const int32_t * tokens, size_t n_tokens,
                              const llama_decode_handoff_plan & plan,
                              uint32_t & cached_len, std::string & status) override {
        const llama_tb_transfer_options opts = build_opts(plan);
        static const int timeout_ms = []() {
            const char * e = std::getenv("LLAMA_KV_DELTA_QUERY_TIMEOUT_MS");
            const int v = e ? atoi(e) : 0;
            return v > 0 ? v : 3000;
        }();
        std::string err;
        if (!llama_tb_transport_query_prefix(opts, tokens, n_tokens, timeout_ms, cached_len, &err)) {
            status = err.empty() ? "prefix query failed" : err;
            cached_len = 0;
            return false;
        }
        status = "prefix query answered";
        return true;
    }
    bool publish_kv_artifact_stream(uint64_t total_bytes, const llama_tb_pull_fn & pull,
                                    const llama_decode_handoff_plan & plan, std::string & status,
                                    llama_decode_publish_diag * diag = nullptr) override {
        const llama_tb_transfer_options opts = build_opts(plan);
        llama_tb_transfer_result res;
        std::string err;
        if (!llama_tb_transport_send_artifact_stream(total_bytes, pull, opts, &res, &err)) {
            status = err.empty() ? "thunderbolt publish (stream) failed" : err;
            return false;
        }
        if (res.transport_backend == "filesystem" || res.transport_mode == "disabled") {
            status = "transport publish did not use rdma/tcp backend (resolved mode=" +
                     (res.transport_mode.empty() ? std::string("auto") : res.transport_mode) + ")";
            return false;
        }
        fill_diag_and_status(res, plan, status, diag);
        return true;
    }

    llama_decode_relay_result relay_tokens(
        const llama_decode_handoff_plan & plan,
        struct llama_context * ctx,
        FILE * output) override {

        llama_decode_relay_result result = {};
        auto t0 = std::chrono::steady_clock::now();

        if (plan.token_stream_host.empty() || plan.token_stream_port <= 0) {
            result.status = "no token stream endpoint configured";
            return result;
        }

        llama_token_stream_client client;
        if (!client.connect(plan.token_stream_host, plan.token_stream_port, 10, 1000)) {
            result.status = "failed to connect to token stream at " +
                            plan.token_stream_host + ":" + std::to_string(plan.token_stream_port);
            return result;
        }

        llama_token_handshake hs;
        if (!client.read_handshake(hs, 300000)) {
            client.disconnect();
            result.status = "token stream handshake timeout";
            return result;
        }
        if (hs.version == 1 && hs.status == "retained-prefix-miss") {
            client.disconnect();
            result.retained_prefix_miss = true;
            result.status = hs.status;
            return result;
        }
        if (hs.version != 1 || hs.status != "ready") {
            client.disconnect();
            result.status = "handshake failed: version=" + std::to_string(hs.version) +
                            " status=" + hs.status;
            return result;
        }

        // Relay loop
        std::vector<int32_t> sync_buffer;
        int32_t sync_base_pos = (int32_t) hs.token_count;

        while (true) {
            llama_token_msg msg;
            if (!client.read_msg(msg, 600000)) {
                result.status = "token stream read timeout after " +
                                std::to_string(result.tokens_received) + " tokens";
                break;
            }

            if (msg.done) {
                result.decode_tok_s = msg.decode_tok_s;
                result.decode_elapsed_ms = msg.decode_elapsed_ms;
                result.decode_intervals = msg.decode_intervals;
                result.decode_tok_s_early = msg.decode_tok_s_early;
                result.decode_tok_s_middle = msg.decode_tok_s_middle;
                result.decode_tok_s_late = msg.decode_tok_s_late;
                result.stop_reason = msg.stop_reason;
                result.stopping_word = msg.stopping_word;
                // Option B: carry the Mac-retained-prefix handle back to the caller.
                result.retained_token_count = msg.retained_token_count;
                result.retained_prefix_hash = msg.retained_prefix_hash;
                result.status = "relay complete";
                break;
            }

            // Display
            if (output) {
                fprintf(output, "%s", msg.text.c_str());
                fflush(output);
            }

            result.tokens_received++;
            sync_buffer.push_back(msg.token_id);

            // KV sync: batch-prefill every N tokens into local KV cache
            if (ctx && plan.kv_sync_batch_size > 0 &&
                (int32_t) sync_buffer.size() >= plan.kv_sync_batch_size) {
                result.tokens_synced += kv_sync_flush(ctx, sync_buffer, sync_base_pos + result.tokens_synced);
                sync_buffer.clear();
            }
        }

        // Flush remaining
        if (ctx && !sync_buffer.empty()) {
            result.tokens_synced += kv_sync_flush(ctx, sync_buffer, sync_base_pos + result.tokens_synced);
            sync_buffer.clear();
        }

        client.disconnect();

        auto t1 = std::chrono::steady_clock::now();
        result.relay_ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

        return result;
    }

    // Callback-based relay for server integration (SSE responses)
    llama_decode_relay_result relay_tokens_cb(
        const llama_decode_handoff_plan & plan,
        struct llama_context * ctx,
        llama_decode_token_callback token_cb) override {

        llama_decode_relay_result result = {};
        auto t0 = std::chrono::steady_clock::now();

        if (plan.token_stream_host.empty() || plan.token_stream_port <= 0) {
            result.status = "no token stream endpoint configured";
            return result;
        }

        fprintf(stderr, "*** RELAY_CB: connecting to %s:%d...\n",
                plan.token_stream_host.c_str(), plan.token_stream_port);
        fflush(stderr);
        llama_token_stream_client client;
        // Large KV imports (1.5 GB+) + tail prefill (13k+ tokens) can take
        // 90+ seconds on the Mac. Allow enough retries to cover this.
        if (!client.connect(plan.token_stream_host, plan.token_stream_port, 120, 2000)) {
            result.status = "failed to connect to token stream at " +
                            plan.token_stream_host + ":" + std::to_string(plan.token_stream_port);
            fprintf(stderr, "*** RELAY_CB: connection FAILED\n"); fflush(stderr);
            return result;
        }

        // Send decode config (stop words, etc.) to Mac before handshake
        if (!plan.stop_words.empty()) {
            llama_decode_config cfg;
            cfg.stop_words = plan.stop_words;
            cfg.max_tokens = plan.max_tokens;
            cfg.prompt_token_count = plan.prompt_token_count;
            cfg.prompt_hash = plan.prompt_hash;
            cfg.reuse_retained_count = plan.reuse_retained_count;
            cfg.reuse_retained_prefix_hash = plan.reuse_retained_prefix_hash;
            if (!client.send_config(cfg)) {
                fprintf(stderr, "*** RELAY_CB: failed to send config\n"); fflush(stderr);
            } else {
                fprintf(stderr, "*** RELAY_CB: sent %zu stop words to Mac\n", plan.stop_words.size());
                fflush(stderr);
            }
        } else {
            // Send empty config so Mac doesn't block waiting for it
            llama_decode_config cfg;
            cfg.max_tokens = plan.max_tokens;
            cfg.prompt_token_count = plan.prompt_token_count;
            cfg.prompt_hash = plan.prompt_hash;
            cfg.reuse_retained_count = plan.reuse_retained_count;
            cfg.reuse_retained_prefix_hash = plan.reuse_retained_prefix_hash;
            client.send_config(cfg);
        }

        const int hs_timeout_ms = 300000; // 5 min — Mac prefill can be slow for large prompts
        fprintf(stderr, "*** RELAY_CB: connected! Reading handshake (timeout=%dms)...\n", hs_timeout_ms); fflush(stderr);

        llama_token_handshake hs;
        if (!client.read_handshake(hs, hs_timeout_ms)) {
            client.disconnect();
            result.status = "token stream handshake timeout";
            fprintf(stderr, "*** RELAY_CB: handshake FAILED\n"); fflush(stderr);
            return result;
        }
        if (hs.version == 1 && hs.status == "retained-prefix-miss") {
            client.disconnect();
            result.retained_prefix_miss = true;
            result.status = hs.status;
            fprintf(stderr, "*** RELAY_CB: retained-prefix-miss NACK\n"); fflush(stderr);
            return result;
        }
        if (hs.version != 1 || hs.status != "ready") {
            client.disconnect();
            result.status = "handshake failed: version=" + std::to_string(hs.version) +
                            " status=" + hs.status;
            fprintf(stderr, "*** RELAY_CB: handshake REJECTED (version=%d status=%s)\n",
                    hs.version, hs.status.c_str()); fflush(stderr);
            return result;
        }
        fprintf(stderr, "*** RELAY_CB: handshake OK (tokens=%u), starting relay\n", hs.token_count);
        fflush(stderr);

        // Relay loop with callback
        while (true) {
            llama_token_msg msg;
            if (!client.read_msg(msg, 600000)) {
                result.status = "token stream read timeout";
                break;
            }

            if (msg.done) {
                result.decode_tok_s = msg.decode_tok_s;
                result.decode_elapsed_ms = msg.decode_elapsed_ms;
                result.decode_intervals = msg.decode_intervals;
                result.decode_tok_s_early = msg.decode_tok_s_early;
                result.decode_tok_s_middle = msg.decode_tok_s_middle;
                result.decode_tok_s_late = msg.decode_tok_s_late;
                result.stop_reason = msg.stop_reason;
                result.stopping_word = msg.stopping_word;
                // Option B: carry the Mac-retained-prefix handle back to the server.
                result.retained_token_count = msg.retained_token_count;
                result.retained_prefix_hash = msg.retained_prefix_hash;
                result.status = "relay complete";
                if (token_cb) token_cb(0, "", true);
                break;
            }

            result.tokens_received++;

            // Route token through callback (for server SSE)
            if (token_cb) {
                if (!token_cb(msg.token_id, msg.text, false)) {
                    result.status = "relay cancelled by callback";
                    break;
                }
            }
        }

        client.disconnect();
        auto t1 = std::chrono::steady_clock::now();
        result.relay_ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        return result;
    }
};

std::unique_ptr<llama_decode_executor_i>
llama_decode_executor_create(const llama_decode_handoff_plan & plan, std::string * fallback_reason) {
    auto set_reason = [&](std::string reason) {
        if (fallback_reason) {
            *fallback_reason = std::move(reason);
        }
    };

    switch (plan.executor_kind) {
        case LLAMA_DECODE_EXECUTOR_LOCAL_CPU:
            return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_LOCAL_CPU);
        case LLAMA_DECODE_EXECUTOR_LOCAL_GPU:
            return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_LOCAL_GPU);
        case LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID:
            return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID);
        case LLAMA_DECODE_EXECUTOR_SPLIT_THUNDERBOLT:
            {
                auto tb = std::make_unique<llama_decode_executor_tb_split>();
                if (tb->available()) {
                    return tb;
                }
                set_reason("thunderbolt transport executor unavailable; falling back to local hybrid");
                return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID);
            }
        case LLAMA_DECODE_EXECUTOR_WIR_WAVEFRONT:
            {
                // WIR uses the same transport as TB split (KV artifact + token relay)
                // but the remote Mac runs wavefront refinement instead of plain AR.
                // Reuse TB split executor for transport; WIR mode is selected on the
                // Mac receiver side via --wir flag.
                auto tb = std::make_unique<llama_decode_executor_tb_split>();
                if (tb->available()) {
                    return tb;
                }
                set_reason("wir_wavefront requires transport; falling back to local hybrid");
                return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID);
            }
        case LLAMA_DECODE_EXECUTOR_WIR_LOCAL:
            {
                // Local WIR: no transport needed. The server runs WIR decode in-process
                // after prefill. Uses a local executor that will be intercepted by the
                // server's slot processing to run WIR instead of standard AR decode.
                return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_WIR_LOCAL);
            }
        case LLAMA_DECODE_EXECUTOR_STREAM_LOCAL:
            {
                // Local streaming decode does not use handoff transport. The caller
                // re-enters decode_streaming() for AR decode batches after prefill.
                return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_STREAM_LOCAL);
            }
    }

    set_reason("unknown executor type; falling back to local CPU");
    return std::make_unique<llama_decode_executor_local>(LLAMA_DECODE_EXECUTOR_LOCAL_CPU);
}

std::string llama_decode_handoff_plan_to_string(const llama_decode_handoff_plan & plan) {
    std::ostringstream oss;
    oss << "decode handoff plan: requested=" << llama_prefill_decode_mode_name(plan.requested_mode)
        << ", resolved=" << llama_prefill_decode_mode_name(plan.resolved_mode)
        << ", executor=" << llama_decode_executor_kind_name(plan.executor_kind)
        << ", transport=" << (plan.use_transport ? "on" : "off")
        << ", transport_required=" << (plan.transport_required ? "true" : "false")
        << ", transport_mode="
        << (plan.transport_mode == LLAMA_PREFILL_TRANSPORT_MODE_PROGRESSIVE ? "progressive" :
            plan.transport_mode == LLAMA_PREFILL_TRANSPORT_MODE_BULK ? "bulk" : "disabled")
        << ", execution_mode="
        << (plan.execution_mode == LLAMA_PREFILL_EXECUTION_MODE_DECOUPLED ? "decoupled" : "coupled")
        << ", remote_nodes=" << plan.remote_nodes
        << ", remote_ranges=" << (plan.remote_ranges.empty() ? "<none>" : plan.remote_ranges)
        << ", remote_failover=" << plan.remote_failover_policy
        << ", expected_gpu_layers=" << plan.expected_gpu_layers
        << ", expected_remote_layers=" << plan.expected_remote_layers
        << ", session_dir=" << (plan.transport_session_dir.empty() ? "<default>" : plan.transport_session_dir)
        << ", kv_transport=" << (plan.kv_transport.empty() ? "<env/default>" : plan.kv_transport)
        << ", endpoint=" << (plan.tb_direct_endpoint.empty() ? "<env/default>" : plan.tb_direct_endpoint)
        << ", kv_host=" << (plan.kv_host.empty() ? "<env/default>" : plan.kv_host)
        << ", kv_port=" << plan.kv_port
        << ", kv_streams=" << plan.kv_streams
        << ", kv_balance=" << (plan.kv_balance.empty() ? "<env/default>" : plan.kv_balance)
        << ", kv_fallback=" << (plan.kv_transport_fallback ? "on" : "off")
        << ", map=";

    if (plan.layer_map.empty()) {
        oss << "<empty>";
    } else {
        for (size_t i = 0; i < plan.layer_map.size(); ++i) {
            const auto & e = plan.layer_map[i];
            if (i > 0) {
                oss << "|";
            }
            oss << e.node << "[" << e.layer_start << "," << e.layer_end << ")";
        }
    }
    if (plan.fallback_applied && !plan.fallback_reason.empty()) {
        oss << ", fallback=\"" << plan.fallback_reason << "\"";
    }

    return oss.str();
}
