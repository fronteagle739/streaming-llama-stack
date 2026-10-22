#include "llama-prefill-stream.h"

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-tb-transport.h"
#include "llama-weight-fetcher.h"
#include "llama-weight-fetcher-adapter.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <limits>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_POSIX_ADVISORY_INFO) || defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

template <typename F>
class llama_scope_exit {
  public:
    explicit llama_scope_exit(F && fn) : fn(std::forward<F>(fn)) {}
    ~llama_scope_exit() { fn(); }

  private:
    F fn;
};

struct llama_dsv4_route_trace_layer {
    bool                 captured = false;
    std::vector<int32_t> experts;
};

static void llama_dsv4_route_trace_append(
        const char *                                      path,
        int                                               pos,
        llama_token                                       token,
        const std::vector<llama_dsv4_route_trace_layer> & layers) {
    static std::mutex      trace_mutex;
    static std::atomic_int trace_step { 0 };

    std::lock_guard<std::mutex> lock(trace_mutex);
    FILE * fp = fopen(path, "ab+");
    if (!fp) {
        LLAMA_LOG_ERROR("%s: failed to open DSV4 route trace '%s'\n", __func__, path);
        return;
    }

    if (fseek(fp, 0, SEEK_END) == 0 && ftell(fp) == 0) {
        int top_k = 0;
        for (const auto & layer : layers) {
            if (layer.captured) {
                top_k = (int) layer.experts.size();
                break;
            }
        }
        fprintf(fp, "# llama-dsv4-route-trace-v1\tlayers=%zu\ttop_k=%d\n", layers.size(), top_k);
        fprintf(fp, "step\tpos\ttoken\troutes\n");
    }

    const int step = trace_step.fetch_add(1, std::memory_order_relaxed);
    fprintf(fp, "%d\t%d\t%d\t", step, pos, (int) token);
    for (size_t il = 0; il < layers.size(); ++il) {
        if (il > 0) {
            fputc(';', fp);
        }
        fprintf(fp, "%zu:", il);
        const auto & layer = layers[il];
        if (!layer.captured) {
            fputc('-', fp);
            continue;
        }
        for (size_t ie = 0; ie < layer.experts.size(); ++ie) {
            if (ie > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%d", layer.experts[ie]);
        }
    }
    fputc('\n', fp);
    fclose(fp);
}

struct llama_dsv4_lid_topk_trace_layer {
    bool                 captured = false;
    int64_t              n_lid = 0;
    int32_t              n_visible = -1;
    std::vector<int32_t> rows;
    // LLAMA_DSV4_LID_PROBE extension: the masked LID score vector read in the same
    // post-layer callback, enabling an in-process score-vs-selection consistency
    // check without the cb_eval hook (which splits/corrupts the FA-on decode graph).
    std::vector<float>   scores;
};

static void llama_dsv4_lid_topk_trace_append(
        const char *                                         path,
        int                                                  pos,
        bool                                                 force_bitonic,
        const std::vector<llama_dsv4_lid_topk_trace_layer> & layers) {
    static std::mutex      trace_mutex;
    static std::atomic_int trace_step { 0 };

    std::lock_guard<std::mutex> lock(trace_mutex);
    FILE * fp = fopen(path, "ab+");
    if (!fp) {
        LLAMA_LOG_ERROR("%s: failed to open DSV4 LID top-k trace '%s'\n", __func__, path);
        return;
    }

    if (fseek(fp, 0, SEEK_END) == 0 && ftell(fp) == 0) {
        fprintf(fp, "# LLAMA_DSV4_LID_TOPK_TRACE v2 (LID_PROBE columns appended when scores captured)\n");
        fprintf(fp, "step\tpos\tlayer\tn_lid\tn_visible\tK\tselected_impl\tselected_rows"
                    "\tsel_consist\tkth_score\tneedle_score\tneedle_rank\ttop1_score\tn_finite\n");
    }

    const int step = trace_step.fetch_add(1, std::memory_order_relaxed);
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & layer = layers[il];
        if (!layer.captured) {
            continue;
        }
        int64_t ncols_pad = 1;
        while (ncols_pad < layer.n_lid) {
            ncols_pad *= 2;
        }
        const bool bitonic_selected = force_bitonic && ncols_pad * (int64_t) sizeof(int) <= 48 * 1024;
        fprintf(fp, "%d\t%d\t%zu\t%lld\t%d\t%zu\t%s\t",
                step, pos, il, (long long) layer.n_lid, layer.n_visible,
                layer.rows.size(), bitonic_selected ? "bitonic" : "cub");
        for (size_t i = 0; i < layer.rows.size(); ++i) {
            if (i != 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%d", layer.rows[i]);
        }

        // LID_PROBE score columns: consistency of the top-k output against the score
        // vector read from the same completed layer graph. sel_consist = fraction of
        // selected rows whose score >= the K-th highest score (ties tolerated);
        // needle_* report the configured needle row's score-space rank.
        if (!layer.scores.empty()) {
            static const int32_t needle_row = []() {
                const char * e = std::getenv("LLAMA_DSV4_LID_NEEDLE_ROW");
                return e && e[0] != '\0' ? std::atoi(e) : -1;
            }();
            const size_t k = layer.rows.size();
            std::vector<float> ordered(layer.scores);
            size_t n_finite = 0;
            for (float & v : ordered) {
                if (std::isnan(v) || std::isinf(v)) { v = -std::numeric_limits<float>::infinity(); }
                else { n_finite++; }
            }
            std::vector<float> sorted(ordered);
            const size_t kth_idx = std::min(k, sorted.size()) - 1;
            std::nth_element(sorted.begin(), sorted.begin() + kth_idx, sorted.end(), std::greater<float>());
            const float kth_score = sorted[kth_idx];
            size_t consistent = 0;
            for (int32_t r : layer.rows) {
                if (r >= 0 && (size_t) r < ordered.size() && ordered[(size_t) r] >= kth_score) {
                    consistent++;
                }
            }
            float  needle_score = std::numeric_limits<float>::quiet_NaN();
            int    needle_rank  = -1;
            if (needle_row >= 0 && (size_t) needle_row < ordered.size()) {
                needle_score = ordered[(size_t) needle_row];
                needle_rank  = 0;
                for (float v : ordered) {
                    if (v > needle_score) { needle_rank++; }
                }
            }
            const float top1_score = *std::max_element(ordered.begin(), ordered.end());
            fprintf(fp, "\t%.4f\t%.6f\t%.6f\t%d\t%.6f\t%zu",
                    k ? (double) consistent / (double) k : 0.0,
                    (double) kth_score,
                    (double) needle_score, needle_rank,
                    (double) top1_score, n_finite);
        }
        fputc('\n', fp);
    }
    fclose(fp);
}

#ifdef GGML_USE_CUDA
class llama_cuda_backend_graph_disable_scope {
  public:
    llama_cuda_backend_graph_disable_scope(ggml_backend_t backend, bool enabled) : backend(backend) {
        if (enabled && backend && ggml_backend_is_cuda(backend)) {
            ggml_backend_cuda_graphs_disable_push(backend);
            active = true;
        }
    }

    ~llama_cuda_backend_graph_disable_scope() {
        if (active) {
            ggml_backend_cuda_graphs_disable_pop(backend);
        }
    }

    llama_cuda_backend_graph_disable_scope(const llama_cuda_backend_graph_disable_scope &) = delete;
    llama_cuda_backend_graph_disable_scope & operator=(const llama_cuda_backend_graph_disable_scope &) = delete;

  private:
    ggml_backend_t backend = nullptr;
    bool active = false;
};
#endif

struct llama_prefill_host_cache_span {
    uint8_t * addr = nullptr;
    size_t    len  = 0;
};

struct llama_prefill_host_cache_layer_plan {
    std::vector<llama_prefill_host_cache_span> spans;
    size_t                                     source_bytes = 0;
    size_t                                     advised_bytes = 0;
};

struct llama_prefill_host_cache_runtime {
    bool                                       enabled = false;
    int                                        window_layers = 0;
    size_t                                     window_bytes = 0;
    int                                        prefetch_distance = 1;
    bool                                       advice_supported = true;
    std::vector<llama_prefill_host_cache_layer_plan> layer_plans;
    std::vector<bool>                          prefetched;
    std::vector<bool>                          evicted;
    int                                        prefetched_layers = 0;
    size_t                                     prefetched_bytes = 0;
    int                                        evicted_layers = 0;
    size_t                                     evicted_bytes = 0;
    int                                        lookups = 0;
    int                                        hits = 0;
    int                                        misses = 0;
    float                                      storage_wait_ms = 0.0f;
};

static size_t llama_prefill_host_cache_page_size() {
#if defined(_POSIX_ADVISORY_INFO) || defined(__APPLE__) || defined(__linux__)
    static size_t page = 0;
    if (page == 0) {
        long v = sysconf(_SC_PAGESIZE);
        page = v > 0 ? (size_t) v : (size_t) 4096;
    }
    return page;
#else
    return (size_t) 4096;
#endif
}

static bool llama_prefill_host_cache_advise_span(const llama_prefill_host_cache_span & span, int advice, std::string * err) {
#if defined(_POSIX_ADVISORY_INFO) || defined(__APPLE__) || defined(__linux__)
    if (!span.addr || span.len == 0) {
        return true;
    }
    const size_t    page = llama_prefill_host_cache_page_size();
    const uintptr_t beg  = (uintptr_t) span.addr;
    const uintptr_t end  = beg + span.len;
    const uintptr_t aligned_beg = beg & ~(uintptr_t) (page - 1);
    const uintptr_t aligned_end = (end + page - 1) & ~(uintptr_t) (page - 1);
    if (aligned_end <= aligned_beg) {
        return true;
    }
    const int rc = posix_madvise((void *) aligned_beg, aligned_end - aligned_beg, advice);
    if (rc != 0) {
        if (err) {
            *err = format("posix_madvise failed rc=%d advice=%d addr=%p len=%zu", rc, advice, (void *) aligned_beg,
                          (size_t) (aligned_end - aligned_beg));
        }
        return false;
    }
    return true;
#else
    GGML_UNUSED(span);
    GGML_UNUSED(advice);
    if (err) {
        *err = "posix_madvise unavailable on this platform";
    }
    return false;
#endif
}

static void llama_prefill_build_host_cache_plans(const std::vector<layer_tensor_info> & layers,
                                                 llama_prefill_host_cache_runtime &      cache) {
    cache.layer_plans.clear();
    cache.layer_plans.resize(layers.size());

    for (size_t il = 0; il < layers.size(); ++il) {
        auto & plan = cache.layer_plans[il];
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        ranges.reserve(layers[il].entries.size());
        for (const auto & entry : layers[il].entries) {
            if (!entry.orig_data || entry.nbytes == 0) {
                continue;
            }
            const uintptr_t beg = (uintptr_t) entry.orig_data;
            const uintptr_t end = beg + entry.nbytes;
            if (end > beg) {
                ranges.emplace_back(beg, end);
                plan.source_bytes += entry.nbytes;
            }
        }
        if (ranges.empty()) {
            continue;
        }

        std::sort(ranges.begin(), ranges.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        std::vector<std::pair<uintptr_t, uintptr_t>> merged;
        merged.reserve(ranges.size());
        for (const auto & r : ranges) {
            if (merged.empty() || r.first > merged.back().second) {
                merged.push_back(r);
                continue;
            }
            merged.back().second = std::max(merged.back().second, r.second);
        }

        plan.spans.reserve(merged.size());
        for (const auto & r : merged) {
            if (r.second <= r.first) {
                continue;
            }
            const size_t len = r.second - r.first;
            plan.spans.push_back({ (uint8_t *) r.first, len });
            plan.advised_bytes += len;
        }
    }
}

static int llama_prefill_host_cache_window_end(const llama_prefill_host_cache_runtime & cache, int anchor_layer, int n_layer) {
    if (anchor_layer >= n_layer) {
        return n_layer - 1;
    }

    int end = n_layer - 1;
    if (cache.window_layers > 0) {
        end = std::min(end, anchor_layer + cache.window_layers - 1);
    }

    if (cache.window_bytes > 0 && anchor_layer < n_layer) {
        size_t budget = 0;
        int    budget_end = anchor_layer - 1;
        for (int il = anchor_layer; il <= end && il < n_layer; ++il) {
            const size_t layer_bytes = il < (int) cache.layer_plans.size() ? cache.layer_plans[il].advised_bytes : 0;
            if (budget > 0 && budget + layer_bytes > cache.window_bytes) {
                break;
            }
            budget += layer_bytes;
            budget_end = il;
            if (budget >= cache.window_bytes) {
                break;
            }
        }
        if (budget_end >= anchor_layer) {
            end = std::min(end, budget_end);
        } else {
            end = anchor_layer;
        }
    }

    return end;
}

static void llama_prefill_host_cache_prefetch_layer(llama_prefill_host_cache_runtime & cache, int il) {
    if (il < 0 || il >= (int) cache.layer_plans.size()) {
        return;
    }
    if (cache.prefetched[il] && !cache.evicted[il]) {
        return;
    }

    auto & plan = cache.layer_plans[il];
    for (const auto & span : plan.spans) {
        std::string err;
#if defined(POSIX_MADV_WILLNEED)
        const bool ok = llama_prefill_host_cache_advise_span(span, POSIX_MADV_WILLNEED, &err);
#elif defined(MADV_WILLNEED)
        const bool ok = llama_prefill_host_cache_advise_span(span, MADV_WILLNEED, &err);
#else
        const bool ok = true; // madvise not available on this platform
        GGML_UNUSED(span); GGML_UNUSED(err);
#endif
        if (!ok) {
            cache.advice_supported = false;
            LLAMA_LOG_WARN("%s: S3 host cache prefetch disabled after advise failure: %s\n", __func__,
                           err.empty() ? "<unknown>" : err.c_str());
            return;
        }
    }

    cache.prefetched[il] = true;
    cache.evicted[il]    = false;
    cache.prefetched_layers += 1;
    cache.prefetched_bytes += plan.advised_bytes;
}

static void llama_prefill_host_cache_evict_layer(llama_prefill_host_cache_runtime & cache, int il) {
    if (il < 0 || il >= (int) cache.layer_plans.size()) {
        return;
    }
    if (cache.evicted[il] || !cache.prefetched[il]) {
        return;
    }

    auto & plan = cache.layer_plans[il];
    for (const auto & span : plan.spans) {
        std::string err;
#if defined(POSIX_MADV_DONTNEED)
        const bool ok = llama_prefill_host_cache_advise_span(span, POSIX_MADV_DONTNEED, &err);
#elif defined(MADV_DONTNEED)
        const bool ok = llama_prefill_host_cache_advise_span(span, MADV_DONTNEED, &err);
#else
        const bool ok = true; // madvise not available on this platform
        GGML_UNUSED(span); GGML_UNUSED(err);
#endif
        if (!ok) {
            cache.advice_supported = false;
            LLAMA_LOG_WARN("%s: S3 host cache eviction disabled after advise failure: %s\n", __func__,
                           err.empty() ? "<unknown>" : err.c_str());
            return;
        }
    }

    cache.evicted[il] = true;
    cache.evicted_layers += 1;
    cache.evicted_bytes += plan.advised_bytes;
}

//
// Streaming Prefill — Phase 2.1: Pointer-Rerouted 2-Buffer GPU Rotation
//
// Architecture:
//   1. Model loaded with -ngl 0 → all weights in CPU heap RAM
//   2. Allocate 2 GPU buffers sized to hold the largest layer
//   3. Temporarily remap all weight tensor->buffer/data to GPU buffer
//      → ggml_backend_sched sees GPU-backed tensors → routes compute to Metal
//   4. Restore original CPU pointers after graph allocation
//   5. Per-layer callbacks memcpy weights from CPU heap → active GPU buffer
//   6. Ping-pong between GPU buffers A and B across layers
//
// Memory: model_heap(~70GB) + 2*max_layer(~2.2GB) + KV+sys(~15GB) ≈ 87GB
// No Triple-Allocation Trap. No SIGKILL on 128GB machines.
//

//
// Collect all non-null tensor pointers from a llama_layer.
// skip_ptrs: set of tensor pointers that are handled as persistent tensors
// (e.g., TENSOR_DUPLICATED shared tensors). These must NOT be remapped per-layer
// because all layers share the same ggml_tensor pointer — remapping would corrupt
// the data pointer for all layers when only the current layer's offset is valid.
//
static void collect_tensors_from_layer(const llama_layer & layer,
                                       std::vector<tensor_remap_entry> & out) {
// Macro to avoid repeating the null check + push_back pattern 100+ times
#define PUSH_IF(field)                            \
    if (layer.field) { \
        tensor_remap_entry e;                     \
        e.tensor      = layer.field;              \
        e.orig_buffer = layer.field->buffer;      \
        e.orig_data   = layer.field->data;        \
        /* no_alloc sentinel (void*)1 means no real data */\
        if ((uintptr_t)e.orig_data <= 1) e.orig_data = nullptr; \
        e.nbytes      = ggml_nbytes(layer.field); \
        out.push_back(e);                         \
    }

    // Attention normalization
    PUSH_IF(attn_norm)
    PUSH_IF(attn_norm_b)
    PUSH_IF(attn_norm_2)
    PUSH_IF(attn_norm_2_b)
    PUSH_IF(attn_q_norm)
    PUSH_IF(attn_q_norm_b)
    PUSH_IF(attn_k_norm)
    PUSH_IF(attn_k_norm_b)
    PUSH_IF(attn_out_norm)
    PUSH_IF(attn_out_norm_b)
    PUSH_IF(attn_q_a_norm)
    PUSH_IF(attn_kv_a_norm)
    PUSH_IF(attn_sub_norm)
    PUSH_IF(attn_post_norm)
    PUSH_IF(ffn_sub_norm)
    PUSH_IF(attn_norm_cross)
    PUSH_IF(attn_norm_enc)
    PUSH_IF(ssm_norm)
    PUSH_IF(ssm_dt_norm)
    PUSH_IF(ssm_b_norm)
    PUSH_IF(ssm_c_norm)

    // Attention weights
    PUSH_IF(wq)
    PUSH_IF(wk)
    PUSH_IF(wv)
    PUSH_IF(wo)
    PUSH_IF(wqkv)
    PUSH_IF(wq_a)
    PUSH_IF(wq_b)
    PUSH_IF(wkv_a_mqa)
    PUSH_IF(wkv_b)
    // DeepSeek-V4 uses the direct KV projection plus grouped output LoRA
    // tensors instead of the usual wkv_a_mqa/wkv_b/wo fields.  These must be
    // part of the streamed layer slab: in no-alloc mode an omitted tensor has
    // no backing storage, and the first HCA/attention kernel that consumes it
    // will dereference an invalid device address.
    PUSH_IF(wkv)
    PUSH_IF(wo_a)
    PUSH_IF(wo_b)
    PUSH_IF(wk_b)
    PUSH_IF(wv_b)
    PUSH_IF(wq_cross)
    PUSH_IF(wk_cross)
    PUSH_IF(wv_cross)
    PUSH_IF(wo_cross)
    PUSH_IF(wq_enc)
    PUSH_IF(wk_enc)
    PUSH_IF(wv_enc)
    PUSH_IF(wo_enc)
    PUSH_IF(wqkv_gate)

    // Attention biases
    PUSH_IF(bq)
    PUSH_IF(bk)
    PUSH_IF(bv)
    PUSH_IF(bo)
    PUSH_IF(bqkv)

    // Relative position bias
    PUSH_IF(attn_rel_b)
    PUSH_IF(attn_rel_b_enc)
    PUSH_IF(attn_rel_b_cross)

    // FFN normalization
    PUSH_IF(ffn_norm)
    PUSH_IF(ffn_norm_b)
    PUSH_IF(ffn_post_norm)
    PUSH_IF(layer_out_norm)
    PUSH_IF(layer_out_norm_b)
    PUSH_IF(ffn_norm_exps)
    PUSH_IF(ffn_norm_enc)

    // FFN weights
    PUSH_IF(ffn_gate)
    PUSH_IF(ffn_down)
    PUSH_IF(ffn_up)
    PUSH_IF(ffn_gate_enc)
    PUSH_IF(ffn_down_enc)
    PUSH_IF(ffn_up_enc)

    // MoE
    PUSH_IF(ffn_gate_inp)
    PUSH_IF(ffn_gate_exps)
    PUSH_IF(ffn_down_exps)
    PUSH_IF(ffn_up_exps)
    PUSH_IF(ffn_gate_up_exps)
    PUSH_IF(ffn_gate_inp_b)
    PUSH_IF(ffn_gate_exps_b)
    PUSH_IF(ffn_down_exps_b)
    PUSH_IF(ffn_up_exps_b)

    // Shared expert
    PUSH_IF(ffn_gate_inp_shexp)
    PUSH_IF(ffn_gate_shexp)
    PUSH_IF(ffn_down_shexp)
    PUSH_IF(ffn_up_shexp)

    // Adjugate experts
    PUSH_IF(ffn_gate_chexps)
    PUSH_IF(ffn_down_chexps)
    PUSH_IF(ffn_up_chexps)

    // FFN biases
    PUSH_IF(ffn_gate_b)
    PUSH_IF(ffn_down_b)
    PUSH_IF(ffn_up_b)
    PUSH_IF(ffn_act)
    PUSH_IF(ffn_exp_probs_b)
    PUSH_IF(ffn_gate_tid2eid)

    // DeepSeek-V4 hyper-connection and compressed-attention weights.  In
    // particular, hc_attn_{fn,scale,base} feed the first hc_pre affine and its
    // sigmoid at the start of every layer.  Leaving them out of this list made
    // WF/no-alloc layer-major prefill run that affine from unbacked tensors;
    // CUDA reported the resulting asynchronous fault at the following
    // SIGMOID (node_12 <- node_11).
    PUSH_IF(attn_kv_norm)
    PUSH_IF(hc_attn_fn)
    PUSH_IF(hc_attn_base)
    PUSH_IF(hc_attn_scale)
    PUSH_IF(hc_ffn_fn)
    PUSH_IF(hc_ffn_base)
    PUSH_IF(hc_ffn_scale)
    PUSH_IF(attn_comp_wkv)
    PUSH_IF(attn_comp_wgate)
    PUSH_IF(attn_comp_ape)
    PUSH_IF(attn_comp_norm)
    PUSH_IF(indexer_comp_wkv)
    PUSH_IF(indexer_comp_wgate)
    PUSH_IF(indexer_comp_ape)
    PUSH_IF(indexer_comp_norm)

    // SSM / Mamba
    PUSH_IF(ssm_in)
    PUSH_IF(ssm_x)
    PUSH_IF(ssm_dt)
    PUSH_IF(ssm_out)
    PUSH_IF(ssm_conv1d)
    PUSH_IF(ssm_a)
    PUSH_IF(ssm_d)
    PUSH_IF(ssm_conv1d_b)
    PUSH_IF(ssm_dt_b)
    PUSH_IF(ssm_beta_alpha)

    // RWKV
    PUSH_IF(time_mix_w1)
    PUSH_IF(time_mix_w2)
    PUSH_IF(time_mix_lerp_x)
    PUSH_IF(time_mix_lerp_w)
    PUSH_IF(time_mix_lerp_k)
    PUSH_IF(time_mix_lerp_v)
    PUSH_IF(time_mix_lerp_r)
    PUSH_IF(time_mix_lerp_g)
    PUSH_IF(time_mix_lerp_fused)
    PUSH_IF(time_mix_first)
    PUSH_IF(time_mix_decay)
    PUSH_IF(time_mix_decay_w1)
    PUSH_IF(time_mix_decay_w2)
    PUSH_IF(time_mix_key)
    PUSH_IF(time_mix_key_b)
    PUSH_IF(time_mix_value)
    PUSH_IF(time_mix_value_b)
    PUSH_IF(time_mix_receptance)
    PUSH_IF(time_mix_receptance_b)
    PUSH_IF(time_mix_gate)

    // RWKV7
    PUSH_IF(time_mix_w0)
    PUSH_IF(time_mix_a0)
    PUSH_IF(time_mix_a1)
    PUSH_IF(time_mix_a2)
    PUSH_IF(time_mix_v0)
    PUSH_IF(time_mix_v1)
    PUSH_IF(time_mix_v2)
    PUSH_IF(time_mix_g1)
    PUSH_IF(time_mix_g2)
    PUSH_IF(time_mix_k_k)
    PUSH_IF(time_mix_k_a)
    PUSH_IF(time_mix_r_k)
    PUSH_IF(time_mix_ln)
    PUSH_IF(time_mix_ln_b)
    PUSH_IF(time_mix_output)

    PUSH_IF(channel_mix_lerp_k)
    PUSH_IF(channel_mix_lerp_r)
    PUSH_IF(channel_mix_key)
    PUSH_IF(channel_mix_receptance)
    PUSH_IF(channel_mix_value)

    // Rope factors
    PUSH_IF(rope_long)
    PUSH_IF(rope_short)
    PUSH_IF(rope_freqs)

    // Bitnet scales
    PUSH_IF(wq_scale)
    PUSH_IF(wk_scale)
    PUSH_IF(wv_scale)
    PUSH_IF(wo_scale)
    PUSH_IF(ffn_gate_scale)
    PUSH_IF(ffn_up_scale)
    PUSH_IF(ffn_down_scale)

    // Altup & Laurel
    PUSH_IF(per_layer_inp_gate)
    PUSH_IF(per_layer_proj)
    PUSH_IF(per_layer_post_norm)
    PUSH_IF(altup_correct_coef)
    PUSH_IF(altup_correct_scale)
    PUSH_IF(altup_predict_coef)
    PUSH_IF(altup_router)
    PUSH_IF(altup_router_norm)
    PUSH_IF(laurel_l)
    PUSH_IF(laurel_r)
    PUSH_IF(laurel_post_norm)

    // OpenAI-MOE
    PUSH_IF(attn_sinks)

    // CogVLM
    PUSH_IF(visexp_attn_wqkv)
    PUSH_IF(visexp_attn_wo)
    PUSH_IF(visexp_ffn_gate)
    PUSH_IF(visexp_ffn_down)
    PUSH_IF(visexp_ffn_up)

    // xIELU activation parameters
    PUSH_IF(ffn_act_alpha_n)
    PUSH_IF(ffn_act_alpha_p)
    PUSH_IF(ffn_act_beta)
    PUSH_IF(ffn_act_eps)

    // Kimi Linear KDA
    PUSH_IF(ssm_q_conv)
    PUSH_IF(ssm_k_conv)
    PUSH_IF(ssm_v_conv)
    PUSH_IF(ssm_f_a)
    PUSH_IF(ssm_f_b)
    PUSH_IF(ssm_beta)
    PUSH_IF(ssm_alpha)
    PUSH_IF(ssm_g_a)
    PUSH_IF(ssm_g_b)
    PUSH_IF(ssm_o_norm)

    // DSA indexer tensors (GLM-5)
    PUSH_IF(indexer_k_norm)
    PUSH_IF(indexer_k_norm_b)
    PUSH_IF(indexer_proj)
    PUSH_IF(indexer_attn_k)
    PUSH_IF(indexer_attn_q_b)

    // V4.1 projections/gates are ordinary per-layer streamed weights. The
    // opaque Engram tables intentionally have no tensor in this list: their
    // selected row264 records use the bounded disk reader instead.
    PUSH_IF(engram_wkv)
    PUSH_IF(engram_q)
    PUSH_IF(engram_k)
    PUSH_IF(ffn_exp_probs_b_vl)

#undef PUSH_IF
}

//
// Layer Weight Size Calculation
//
size_t llama_layer_weight_bytes(const llama_model * model, int layer_idx) {
    if (!model || layer_idx < 0 || (size_t) layer_idx >= model->layers.size()) {
        return 0;
    }

    std::vector<tensor_remap_entry> entries;
    collect_tensors_from_layer(model->layers[layer_idx], entries);

    size_t total = 0;
    for (const auto & e : entries) {
        total += e.nbytes;
    }
    return total;
}

//
// Collect all per-layer tensor pointers and compute max layer size
//
void prefill_collect_layer_tensors(const llama_model *              model,
                                   std::vector<layer_tensor_info> & out_layers,
                                   size_t &                         out_max_layer_bytes) {
    const int n_layer = model->hparams.n_layer;
    out_layers.resize(n_layer);
    out_max_layer_bytes = 0;

    for (int il = 0; il < n_layer; ++il) {
        auto & info = out_layers[il];
        info.entries.clear();
        info.total_bytes = 0;

        collect_tensors_from_layer(model->layers[il], info.entries);

        for (const auto & e : info.entries) {
            info.total_bytes += e.nbytes;
        }

        if (info.total_bytes > out_max_layer_bytes) {
            out_max_layer_bytes = info.total_bytes;
        }
    }

    std::unordered_map<ggml_tensor *, int> first_seen_layer;
    first_seen_layer.reserve(n_layer * 128);
    int duplicate_count = 0;
    std::vector<std::string> duplicate_samples;
    duplicate_samples.reserve(8);

    for (int il = 0; il < n_layer; ++il) {
        for (const auto & e : out_layers[il].entries) {
            auto it = first_seen_layer.find(e.tensor);
            if (it == first_seen_layer.end()) {
                first_seen_layer.emplace(e.tensor, il);
                continue;
            }
            duplicate_count++;
            if ((int) duplicate_samples.size() < 8) {
                const char * tname = ggml_get_name(e.tensor);
                duplicate_samples.push_back(
                    format("%s first=L%d now=L%d", tname ? tname : "(unnamed)", it->second, il));
            }
        }
    }

    LLAMA_LOG_INFO("%s: collected %d layers, max layer = %.2f MB, total tensors per layer = %zu..%zu\n", __func__,
                   n_layer, out_max_layer_bytes / (1024.0 * 1024.0),
                   out_layers.empty() ? 0 : out_layers[0].entries.size(),
                   out_layers.empty() ? 0 : out_layers[n_layer - 1].entries.size());
    if (duplicate_count > 0) {
        std::string sample_joined;
        for (size_t i = 0; i < duplicate_samples.size(); ++i) {
            if (!sample_joined.empty()) {
                sample_joined += " | ";
            }
            sample_joined += duplicate_samples[i];
        }
        LLAMA_LOG_WARN("%s: detected %d cross-layer duplicate tensor references; overlap safety may be impacted%s%s\n",
                       __func__, duplicate_count, sample_joined.empty() ? "" : " (samples: ",
                       sample_joined.empty() ? "" : sample_joined.c_str());
        if (!sample_joined.empty()) {
            LLAMA_LOG_WARN("%s: duplicate sample details end\n", __func__);
        }
    }
}

//
// Classify a tensor as attention-only (true) or FFN/other (false).
// Used by split decode: RTX loads attention weights, Mac handles FFN/MoE.
//
static bool is_attention_tensor(const char * name) {
    if (!name) return false;
    // Skip the "blk.N." prefix to get the tensor type
    const char * dot = strrchr(name, '.');
    if (!dot) return false;
    const char * field = dot + 1;
    // Check previous segment for "blk.N" pattern
    const char * prev_dot = dot - 1;
    while (prev_dot > name && *prev_dot != '.') prev_dot--;
    if (prev_dot > name) prev_dot++;

    // Attention tensors: anything with "attn" in the name
    if (strstr(name, "attn"))  return true;
    // Rope factors
    if (strstr(name, "rope"))  return true;
    // Bitnet attention scales
    if (strstr(name, "wq_scale") || strstr(name, "wk_scale") ||
        strstr(name, "wv_scale") || strstr(name, "wo_scale")) return true;

    // NOT attention: ffn, ssm, time_mix, channel_mix, etc.
    return false;
}

//
// Collect attention-only tensors for split decode.
// Returns total attention weight bytes across all layers.
//
size_t prefill_collect_attention_tensors(const llama_model *              model,
                                         std::vector<layer_tensor_info> & out_layers) {
    const int n_layer = model->hparams.n_layer;
    out_layers.resize(n_layer);
    size_t total_attn_bytes = 0;

    for (int il = 0; il < n_layer; ++il) {
        auto & info = out_layers[il];
        info.entries.clear();
        info.total_bytes = 0;

        // Collect ALL tensors first
        std::vector<tensor_remap_entry> all_entries;
        collect_tensors_from_layer(model->layers[il], all_entries);

        // Filter to attention-only
        for (auto & e : all_entries) {
            const char * tname = e.tensor ? ggml_get_name(e.tensor) : nullptr;
            if (is_attention_tensor(tname)) {
                info.entries.push_back(std::move(e));
                info.total_bytes += e.nbytes;
            }
        }

        total_attn_bytes += info.total_bytes;
    }

    LLAMA_LOG_INFO("%s: collected attention tensors for %d layers, total = %.2f MB\n",
                   __func__, n_layer, total_attn_bytes / (1024.0 * 1024.0));
    return total_attn_bytes;
}

//
// Compute GPU alloc offsets — uses the ggml backend API to determine
// exact tensor placement including backend-specific padding.
// This mirrors what ggml_backend_alloc_ctx_tensors_from_buft does internally.
//
void prefill_compute_alloc_offsets(std::vector<layer_tensor_info> & layers,
                                   size_t &                         max_layer_bytes,
                                   ggml_backend_buffer_type_t       gpu_buft) {
    if (!gpu_buft) return;

    max_layer_bytes = 0;

    // Align tensor offsets — CUDA MUL_MAT_ID requires at least 256-byte alignment.
    // Use the backend's alignment, with 256 as minimum.
    const size_t backend_align = ggml_backend_buft_get_alignment(gpu_buft);
    const size_t TENSOR_ALIGNMENT = std::max(backend_align, (size_t)256);

    for (auto & info : layers) {
        size_t offset = 0;
        for (auto & e : info.entries) {
            // Get the backend's alloc size (includes padding like MATRIX_ROW_PADDING)
            e.alloc_size = e.tensor ? ggml_backend_buft_get_alloc_size(gpu_buft, e.tensor)
                                    : e.nbytes;
            offset = (offset + TENSOR_ALIGNMENT - 1) & ~(TENSOR_ALIGNMENT - 1);
            e.alloc_offset = offset;
            offset += e.alloc_size;
        }
        info.total_bytes = offset;
        if (offset > max_layer_bytes) max_layer_bytes = offset;
    }

    LLAMA_LOG_INFO("%s: computed alloc offsets for %zu layers, max=%.2f MB\n",
                   __func__, layers.size(), max_layer_bytes / (1024.0 * 1024.0));
}

//
// Allocate 2 GPU buffers for the largest layer
//
bool prefill_allocate_gpu_buffers(ggml_backend_t          backend,
                                  size_t                  max_layer_bytes,
                                  ggml_backend_buffer_t * out_buf_a,
                                  ggml_backend_buffer_t * out_buf_b) {
    // Get the GPU buffer type from the backend's device
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        LLAMA_LOG_ERROR("%s: backend has no device\n", __func__);
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft) {
        LLAMA_LOG_ERROR("%s: device has no buffer type\n", __func__);
        return false;
    }

    // Add 10% padding for alignment
    size_t alloc_size = max_layer_bytes + max_layer_bytes / 10;

    *out_buf_a = ggml_backend_buft_alloc_buffer(buft, alloc_size);
    if (!*out_buf_a) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffer A (%.2f MB)\n", __func__, alloc_size / (1024.0 * 1024.0));
        return false;
    }

    *out_buf_b = ggml_backend_buft_alloc_buffer(buft, alloc_size);
    if (!*out_buf_b) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffer B (%.2f MB)\n", __func__, alloc_size / (1024.0 * 1024.0));
        ggml_backend_buffer_free(*out_buf_a);
        *out_buf_a = nullptr;
        return false;
    }

    // Mark the rotating buffers as weights so the scheduler preferentially
    // assigns weight-consuming ops to the backend that owns these buffers.
    ggml_backend_buffer_set_usage(*out_buf_a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(*out_buf_b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    LLAMA_LOG_INFO("%s: allocated 2 GPU buffers: %.2f MB each (%.2f MB total)\n", __func__,
                   alloc_size / (1024.0 * 1024.0), 2.0 * alloc_size / (1024.0 * 1024.0));

    return true;
}

//
// Temporarily remap all weight tensors to point at a GPU buffer
//
// This is the core "trick": the ggml_backend_sched looks at tensor->buffer
// to decide which backend gets the compute. By setting all weight buffers
// to the GPU buffer, the scheduler will route all matmuls to Metal.
//
// The tensor->data field is set to a valid offset within the GPU buffer
// so that the scheduler can compute buffer sizes. The actual data content
// doesn't matter at this point — real data is uploaded per-layer later.
//
void prefill_remap_tensors_to_gpu(std::vector<layer_tensor_info> & layers, ggml_backend_buffer_t gpu_buf) {
    void * gpu_base = ggml_backend_buffer_get_base(gpu_buf);
    size_t gpu_size = ggml_backend_buffer_get_size(gpu_buf);

    int total_remapped = 0;

    for (auto & info : layers) {
        size_t offset = 0;
        for (auto & entry : info.entries) {
            // Save original pointers (already done in collect, but re-save in case
            // this is called after a previous remap/restore cycle)
            entry.orig_buffer = entry.tensor->buffer;
            entry.orig_data   = entry.tensor->data;
            if ((uintptr_t)entry.orig_data <= 1) entry.orig_data = nullptr; // no_alloc sentinel

            // Remap to GPU
            entry.tensor->buffer = gpu_buf;

            // Use pre-computed alloc_offset (from prefill_compute_alloc_offsets).
            // Falls back to sequential nbytes packing if offsets weren't computed.
            size_t data_offset = (entry.alloc_offset > 0 || entry.alloc_size > 0)
                               ? entry.alloc_offset % gpu_size
                               : offset % gpu_size;
            entry.tensor->data = (char *) gpu_base + data_offset;

            offset += (entry.alloc_size > 0) ? entry.alloc_size : entry.nbytes;
            total_remapped++;
        }
    }

    LLAMA_LOG_INFO("%s: remapped %d tensors across %zu layers to GPU buffer\n", __func__, total_remapped,
                   layers.size());
}

//
// Restore original CPU buffer/data pointers on all tensors
//
void prefill_restore_tensor_pointers(std::vector<layer_tensor_info> & layers) {
    for (auto & info : layers) {
        for (auto & entry : info.entries) {
            entry.tensor->buffer = entry.orig_buffer;
            entry.tensor->data   = entry.orig_data;
        }
    }
}

//
// Collect non-layer model tensors (embeddings, output head, norms) that must
// live on GPU persistently during streaming prefill.  These are NOT part of
// any per-layer rotation — they are uploaded once and stay on GPU.
//
static void collect_persistent_tensors(const llama_model *              model,
                                       std::vector<tensor_remap_entry> & out,
                                       size_t &                         out_total_bytes) {
    out.clear();
    out_total_bytes = 0;

#define PUSH_PERSIST(field)                                \
    if (model->field) {                                    \
        tensor_remap_entry e;                              \
        e.tensor      = model->field;                      \
        e.orig_buffer = model->field->buffer;              \
        e.orig_data   = model->field->data;                \
        /* no_alloc sentinel (void*)1 means no real data */\
        if ((uintptr_t)e.orig_data <= 1) e.orig_data = nullptr; \
        e.nbytes      = ggml_nbytes(model->field);         \
        out.push_back(e);                                  \
        out_total_bytes += e.nbytes;                       \
    }

    PUSH_PERSIST(tok_embd)
    PUSH_PERSIST(type_embd)
    PUSH_PERSIST(pos_embd)
    PUSH_PERSIST(tok_norm)
    PUSH_PERSIST(tok_norm_b)
    PUSH_PERSIST(output_norm)
    PUSH_PERSIST(output_norm_b)
    PUSH_PERSIST(output)
    PUSH_PERSIST(output_b)
    // DeepSeek-V4 applies one final hyper-connection reduction after the last
    // streamed layer.  It is model-global, so keep it resident with the output
    // head rather than rotating it through a per-layer slab.
    PUSH_PERSIST(hc_head_fn)
    PUSH_PERSIST(hc_head_base)
    PUSH_PERSIST(hc_head_scale)

    // DSV4 raw attention consumes the per-layer f32 sinks directly as FA src4.
    // Keep these tiny tensors resident for the whole pass so their buffer is
    // already GPU-backed when the scheduler evaluates FLASH_ATTN_EXT support.
    if (model->arch == LLM_ARCH_DEEPSEEK4) {
        for (const auto & layer : model->layers) {
            if (!layer.attn_sinks) {
                continue;
            }
            tensor_remap_entry e;
            e.tensor      = layer.attn_sinks;
            e.orig_buffer = layer.attn_sinks->buffer;
            e.orig_data   = layer.attn_sinks->data;
            if ((uintptr_t) e.orig_data <= 1) e.orig_data = nullptr;
            e.nbytes      = ggml_nbytes(layer.attn_sinks);
            out.push_back(e);
            out_total_bytes += e.nbytes;
        }
    }

#undef PUSH_PERSIST
}

//
// Upload one layer's weights from CPU heap into a GPU buffer
//
// This is the hot path during the per-layer forward pass.
// For each tensor in the layer:
//   1. Bind tensor->buffer/data to packed offsets in gpu_buf
//   2. Upload tensor bytes via backend transfer API
//
// Using ggml_backend_tensor_set() keeps this backend-agnostic:
// - Metal (Apple unified memory)
// - CUDA (host->device copy)
// - Vulkan/OpenCL/SYCL backends that implement set_tensor
//
static float prefill_upload_layer(std::vector<tensor_remap_entry> & layer_entries,
                                  ggml_backend_buffer_t             gpu_buf,
                                  size_t                            slab_bytes,
                                  int                               layer_idx,
                                  bool                              trace_swap,
                                  bool                              remote_weight_relay) {
    auto t_start = std::chrono::high_resolution_clock::now();

    void * gpu_base = ggml_backend_buffer_get_base(gpu_buf);
    size_t offset   = 0;
    const size_t chunk_bytes = slab_bytes == 0 ? std::numeric_limits<size_t>::max() : slab_bytes;
    if (trace_swap) {
        LLAMA_LOG_INFO("%s: [SWAP_TRACE] upload begin layer %d buf=%p entries=%zu\n",
                       __func__, layer_idx, (void *) gpu_buf, layer_entries.size());
    }

    for (auto & entry : layer_entries) {
        // Redirect tensor to read from GPU buffer
        entry.tensor->buffer = gpu_buf;
        entry.tensor->data   = (char *) gpu_base + offset;

        // Copy weight data from CPU heap to device buffer. When slab_bytes is set,
        // upload in fixed chunks so staging/upload behavior can be tuned at runtime.
        size_t copied = 0;
        while (copied < entry.nbytes) {
            const size_t n = std::min(chunk_bytes, entry.nbytes - copied);
            const uint8_t *src = (const uint8_t *) entry.orig_data + copied;
            if (remote_weight_relay) {
                std::vector<uint8_t> remote_chunk;
                std::string fetch_error;
                if (llama_tb_transport_fetch_weight_chunk(src, n, remote_chunk, &fetch_error) &&
                    remote_chunk.size() == n) {
                    ggml_backend_tensor_set(entry.tensor, remote_chunk.data(), copied, n);
                } else {
                    static bool warned_remote_fetch_fallback = false;
                    if (!warned_remote_fetch_fallback) {
                        warned_remote_fetch_fallback = true;
                        LLAMA_LOG_WARN("%s: remote weight fetch failed, falling back to local chunk copy: %s\n",
                                       __func__, fetch_error.empty() ? "<unknown>" : fetch_error.c_str());
                    }
                    ggml_backend_tensor_set(entry.tensor, src, copied, n);
                }
            } else {
                ggml_backend_tensor_set(entry.tensor, src, copied, n);
            }
            copied += n;
        }

        offset += entry.nbytes;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    const float upload_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    if (trace_swap) {
        LLAMA_LOG_INFO("%s: [SWAP_TRACE] upload end layer %d buf=%p bytes=%.2fMB time=%.2fms\n",
                       __func__, layer_idx, (void *) gpu_buf, offset / (1024.0 * 1024.0), upload_ms);
    }
    return upload_ms;
}

// ---------------------------------------------------------------------------
// Staged upload: 2-phase (CPU staging to pinned, then async H2D on copy stream)
//
// Phase 1 (CPU): memcpy from mmap → pinned staging buffer  (blocks host)
// Phase 2 (GPU): cudaMemcpyAsync pinned → GPU on copy stream (non-blocking)
//
// After this returns, the H2D transfer is IN FLIGHT on the copy stream.
// The caller must call ggml_backend_cuda_prefill_wait_upload() on the
// compute stream before the layer's kernels execute.
// ---------------------------------------------------------------------------
#ifdef GGML_USE_CUDA
static float prefill_upload_layer_staged(
        std::vector<tensor_remap_entry> & layer_entries,
        ggml_backend_buffer_t             gpu_buf,
        void *                            pinned_staging,
        void *                            pipeline,
        int                               layer_idx,
        bool                              trace_swap) {
    auto t_start = std::chrono::high_resolution_clock::now();

    void * gpu_base = ggml_backend_buffer_get_base(gpu_buf);
    size_t offset   = 0;

    // Phase 1: remap tensor pointers and stage ALL weight data into pinned buffer
    for (auto & entry : layer_entries) {
        entry.tensor->buffer = gpu_buf;
        entry.tensor->data   = (char *) gpu_base + offset;

        std::memcpy((char *) pinned_staging + offset, entry.orig_data, entry.nbytes);
        offset += entry.nbytes;
    }

    auto t_staged = std::chrono::high_resolution_clock::now();

    // Phase 2: single large async H2D on copy stream
    ggml_backend_cuda_prefill_upload_async(pipeline, gpu_base, pinned_staging, offset);

    auto t_end = std::chrono::high_resolution_clock::now();
    const float stage_ms = std::chrono::duration<float, std::milli>(t_staged - t_start).count();
    const float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    if (trace_swap) {
        LLAMA_LOG_INFO("%s: [SWAP_TRACE] staged upload layer %d: %.2f MB, stage=%.2fms, enqueue=%.2fms\n",
                       __func__, layer_idx, offset / (1024.0 * 1024.0), stage_ms, total_ms - stage_ms);
    }
    return total_ms;
}
#endif

//
// Streaming Availability Check
//
bool llama_prefill_stream_available(const llama_context * ctx) {
    GGML_UNUSED(ctx);
    return true;
}

//
// Main Streaming Prefill Entry Point — Phase 2.1: Pointer-Rerouted
//
// This function is called from decode_streaming(). It:
//   1. Collects all per-layer tensor pointers
//   2. Allocates 2 GPU buffers for the largest layer
//   3. Remaps all tensors → GPU, builds graph, restores CPU pointers
//   4. Per-layer callbacks upload weights and ping-pong GPU buffers
//
int llama_prefill_stream(llama_context *                     ctx,
                         const llama_batch &                 batch,
                         const llama_prefill_stream_params & params,
                         llama_prefill_stream_result *       result) {
    if (!result) {
        return -1;
    }

    result->layer_metrics.clear();
    result->used_streaming = false;
    result->host_cache_enabled = false;
    result->host_cache_advice_supported = true;
    result->host_prefetch_layers = 0;
    result->host_prefetch_bytes = 0;
    result->host_evict_layers = 0;
    result->host_evict_bytes = 0;
    result->host_cache_lookups = 0;
    result->host_cache_hits = 0;
    result->host_cache_misses = 0;
    result->host_cache_hit_rate = 0.0f;
    result->host_storage_wait_ms = 0.0f;

    // Skip streaming for tiny batches — uploading weights for a few tokens
    // is slower than normal decode. BUT: when the weight fetcher is active
    // (wf_active), normal decode uses sentinel data and produces garbage.
    // In that case, we MUST stream regardless of batch size.
    const int min_streaming_tokens = std::max(1, params.min_streaming_tokens);
    const bool wf_active = ctx->get_model().hparams.wf_active;
    if (!params.force_streaming && batch.n_tokens < min_streaming_tokens && !wf_active) {
        LLAMA_LOG_DEBUG("%s: skipping streaming for small batch (%d tokens < %d min)\n",
                       __func__, batch.n_tokens, min_streaming_tokens);
        result->used_streaming = false;
        return 1;  // signal caller to use normal decode
    }

    LLAMA_LOG_INFO("%s: streaming prefill %d tokens\n", __func__, batch.n_tokens);

    auto t_total_start = std::chrono::high_resolution_clock::now();

    const llama_model * model = llama_get_model(ctx);
    if (!model) {
        LLAMA_LOG_ERROR("%s: [DIAG] llama_get_model returned null!\n", __func__);
        return -1;
    }
    const int n_layer = model->hparams.n_layer;

    const char * dsv4_route_trace_path = std::getenv("LLAMA_DSV4_ROUTE_TRACE");
    const bool dsv4_route_trace_enabled =
            dsv4_route_trace_path != nullptr && dsv4_route_trace_path[0] != '\0' &&
            model->arch == LLM_ARCH_DEEPSEEK4 &&
            (params.route_trace_decode_step || params.route_trace_prompt_last) &&
            batch.token != nullptr;
    std::vector<llama_dsv4_route_trace_layer> dsv4_route_trace_layers(
            dsv4_route_trace_enabled ? (size_t) n_layer : 0);

    // LLAMA_DSV4_LID_PROBE is the standalone path for THIS (post-layer-callback) trace:
    // unlike LLAMA_DSV4_LID_TOPK_TRACE it does NOT also install the llama-context cb_eval
    // hook, whose graph-splitting corrupts the FA-on stream_local decode. Prefer it.
    const char * dsv4_lid_topk_trace_path = std::getenv("LLAMA_DSV4_LID_PROBE");
    if (!dsv4_lid_topk_trace_path || dsv4_lid_topk_trace_path[0] == '\0') {
        dsv4_lid_topk_trace_path = std::getenv("LLAMA_DSV4_LID_TOPK_TRACE");
    }
    const bool dsv4_lid_topk_trace_enabled =
            dsv4_lid_topk_trace_path != nullptr && dsv4_lid_topk_trace_path[0] != '\0' &&
            model->arch == LLM_ARCH_DEEPSEEK4 && params.route_trace_decode_step &&
            batch.n_tokens == 1;
    std::vector<llama_dsv4_lid_topk_trace_layer> dsv4_lid_topk_trace_layers(
            dsv4_lid_topk_trace_enabled ? (size_t) n_layer : 0);

    const char * force_bitonic_env = std::getenv("LLAMA_DSV4_TOPK_FORCE_BITONIC");
    const bool force_bitonic = force_bitonic_env != nullptr &&
            force_bitonic_env[0] != '\0' && force_bitonic_env[0] != '0';

    LLAMA_LOG_INFO("%s: [DIAG] model has %d layers\n", __func__, n_layer);

    const bool enable_overlap = params.enable_overlap;
    bool remote_weight_relay = params.remote_weight_relay;
    if (remote_weight_relay && !llama_tb_transport_enabled()) {
        if (params.remote_weight_hybrid) {
            LLAMA_LOG_WARN("%s: remote weight relay requested but transport is unavailable; using local weight uploads\n", __func__);
            remote_weight_relay = false;
        } else {
            LLAMA_LOG_WARN("%s: remote weight relay requested without active transport; using loopback relay copy path\n", __func__);
        }
    }
    const char * swap_trace_env = std::getenv("LLAMA_PREFILL_SWAP_TRACE");
    const bool trace_swap = swap_trace_env != nullptr && swap_trace_env[0] != '\0' && std::strcmp(swap_trace_env, "0") != 0;

    const char * prefill_debug_env = std::getenv("LLAMA_PREFILL_DEBUG");
    const int    prefill_debug     = prefill_debug_env ? atoi(prefill_debug_env) : 0;
    const size_t slab_bytes = params.slab_bytes;
    int prefetch_distance = std::max(1, params.prefetch_distance);
    const int host_cache_prefetch_distance = std::max(1, params.host_cache_prefetch_distance);
    llama_prefill_host_cache_runtime host_cache;
    host_cache.window_layers = std::max(0, params.host_cache_window_layers);
    host_cache.window_bytes  = params.host_cache_window_bytes;
    host_cache.prefetch_distance = host_cache_prefetch_distance;
    host_cache.enabled = host_cache.window_layers > 0 || host_cache.window_bytes > 0;
    result->host_cache_enabled = host_cache.enabled;
    if (params.n_buffers != 2) {
        LLAMA_LOG_WARN("%s: only 2-buffer ping-pong is currently implemented (requested %d, using 2)\n", __func__,
                       params.n_buffers);
    }
    if (prefetch_distance != 1) {
        LLAMA_LOG_WARN("%s: prefetch_distance=%d requested, clamping to 1 for 2-buffer ping-pong\n", __func__,
                       prefetch_distance);
        prefetch_distance = 1;
    }

    // ====================================================================
    // Step 1: Collect per-layer tensor info and compute max layer size
    // ====================================================================

    std::vector<layer_tensor_info> layer_infos;
    size_t                         max_layer_bytes = 0;
    prefill_collect_layer_tensors(model, layer_infos, max_layer_bytes);

    // DSV4 attention sinks are scheduling inputs to FLASH_ATTN_EXT, not bulk
    // rotating weights. They are uploaded into the persistent GPU buffer below;
    // remove them from each layer slab so later WF remaps cannot put src4 back
    // onto a transient/CPU-described tensor at graph construction time.
    if (model->arch == LLM_ARCH_DEEPSEEK4) {
        max_layer_bytes = 0;
        for (size_t il = 0; il < layer_infos.size(); ++il) {
            auto & info = layer_infos[il];
            ggml_tensor * sinks = il < model->layers.size() ? model->layers[il].attn_sinks : nullptr;
            if (sinks) {
                info.entries.erase(std::remove_if(info.entries.begin(), info.entries.end(),
                        [&](const tensor_remap_entry & entry) { return entry.tensor == sinks; }),
                        info.entries.end());
            }
            info.total_bytes = 0;
            for (const auto & entry : info.entries) {
                info.total_bytes += entry.nbytes;
            }
            max_layer_bytes = std::max(max_layer_bytes, info.total_bytes);
        }
    }

    LLAMA_LOG_INFO("%s: [DIAG] tensor collection done, max_layer=%.2f MB, layers=%zu\n", __func__,
                   max_layer_bytes / (1024.0 * 1024.0), layer_infos.size());

    // Phase 1 diagnostic: measure attention vs FFN/MoE split per layer
    if (!layer_infos.empty()) {
        size_t attn_total = 0, ffn_total = 0, max_attn = 0, max_ffn = 0;
        for (int il = 0; il < (int)layer_infos.size(); il++) {
            size_t attn_bytes = 0, ffn_bytes = 0;
            for (auto & e : layer_infos[il].entries) {
                const char * name = e.tensor->name;
                if (!name) continue;
                // Attention: attn_*, wq*, wk*, wv*, wo*
                if (strstr(name, "attn_") || strncmp(name, "wq", 2) == 0 ||
                    strncmp(name, "wk", 2) == 0 || strncmp(name, "wv", 2) == 0 ||
                    strncmp(name, "wo", 2) == 0 || strstr(name, "bq") ||
                    strstr(name, "bk") || strstr(name, "bv") || strstr(name, "bo")) {
                    attn_bytes += e.nbytes;
                } else {
                    ffn_bytes += e.nbytes;
                }
            }
            attn_total += attn_bytes;
            ffn_total += ffn_bytes;
            if (attn_bytes > max_attn) max_attn = attn_bytes;
            if (ffn_bytes > max_ffn) max_ffn = ffn_bytes;
        }
    }

    // ---- Sub-layer splitting for MoE expert pipelining ----
    // LLAMA_WF_LAYER_SPLITS=N splits each model layer into N virtual WF layers.
    // Group 0: attention + shared expert + router + experts [0, n_exp/N)
    // Group 1..N-1: experts [k*n_exp/N, (k+1)*n_exp/N)
    // The WF buffer only needs to hold the largest sub-layer.
    const char * splits_env = std::getenv("LLAMA_WF_LAYER_SPLITS");
    int n_sublayers = (splits_env && splits_env[0] != '\0') ? std::atoi(splits_env) : 1;
    if (n_sublayers < 1) n_sublayers = 1;
    if (n_sublayers > 8) n_sublayers = 8;

    std::vector<layer_tensor_info> sublayer_infos;  // flat: [il*N+s] for model layer il, sub-layer s
    size_t max_sublayer_bytes = max_layer_bytes;

    if (n_sublayers > 1 && !layer_infos.empty()) {
        sublayer_infos.resize(n_layer * n_sublayers);
        max_sublayer_bytes = 0;

        for (int il = 0; il < n_layer; il++) {
            auto & info = layer_infos[il];

            // Classify entries: non-expert go to group 0, expert tensors split evenly
            std::vector<tensor_remap_entry> non_expert;
            std::vector<tensor_remap_entry> expert;

            for (auto & e : info.entries) {
                const char * name = e.tensor->name;
                // Expert tensors: ffn_{gate,down,up}_exps (packed 3D, dim 2 = n_experts)
                if (name && (strstr(name, "ffn_gate_exps") || strstr(name, "ffn_down_exps") ||
                             strstr(name, "ffn_up_exps"))) {
                    expert.push_back(e);
                } else {
                    non_expert.push_back(e);
                }
            }

            if (expert.empty()) {
                // Non-MoE layer (e.g., layer 0) — all in group 0, rest empty
                sublayer_infos[il * n_sublayers] = info;
                if (info.total_bytes > max_sublayer_bytes) max_sublayer_bytes = info.total_bytes;
                for (int s = 1; s < n_sublayers; s++) {
                    sublayer_infos[il * n_sublayers + s].total_bytes = 0;
                }
            } else {
                // MoE layer: determine n_experts from first expert tensor's ne[2]
                int n_experts = (int)expert[0].tensor->ne[2];
                int experts_per_group = (n_experts + n_sublayers - 1) / n_sublayers;

                // For now, we can't slice packed 3D expert tensors at the WF level.
                // Instead, put ALL expert tensors in each sub-layer and split the
                // file read range. The WF reads a contiguous byte range per span.
                //
                // Each expert tensor is [dim0, dim1, n_experts] — experts are
                // contiguous in memory along dimension 2. Expert k starts at
                // offset k * (dim0 * dim1 * type_size) within the tensor.
                size_t expert_elem_size = ggml_row_size(expert[0].tensor->type, expert[0].tensor->ne[0]);
                size_t expert_stride = expert_elem_size * expert[0].tensor->ne[1]; // bytes per expert

                for (int s = 0; s < n_sublayers; s++) {
                    auto & sub = sublayer_infos[il * n_sublayers + s];
                    sub.total_bytes = 0;

                    // Group 0 gets non-expert tensors
                    if (s == 0) {
                        sub.entries = non_expert;
                        for (auto & e : sub.entries) sub.total_bytes += e.nbytes;
                    }

                    // Each group gets its share of expert tensors
                    int exp_start = s * experts_per_group;
                    int exp_end   = std::min(exp_start + experts_per_group, n_experts);
                    if (exp_start < n_experts) {
                        for (auto & exp_entry : expert) {
                            // Create a virtual entry for this expert slice
                            tensor_remap_entry slice = exp_entry;
                            // Adjust data pointer and size for this expert range
                            size_t s_offset = (size_t)exp_start * expert_stride;
                            size_t slice_bytes  = (size_t)(exp_end - exp_start) * expert_stride;
                            slice.orig_data = exp_entry.orig_data
                                ? (char *)exp_entry.orig_data + s_offset : nullptr;
                            slice.nbytes = slice_bytes;
                            slice.slice_offset = s_offset;
                            sub.entries.push_back(slice);
                            sub.total_bytes += slice_bytes;
                        }
                    }

                    if (sub.total_bytes > max_sublayer_bytes) max_sublayer_bytes = sub.total_bytes;
                }
            }
        }

    }
    if (host_cache.enabled) {
        llama_prefill_build_host_cache_plans(layer_infos, host_cache);
        host_cache.prefetched.assign(n_layer, false);
        host_cache.evicted.assign(n_layer, false);
        LLAMA_LOG_INFO("%s: S3 host moving-window active (layers=%d, bytes=%.2fMB, prefetch_distance=%d)\n",
                       __func__, host_cache.window_layers, host_cache.window_bytes / (1024.0 * 1024.0),
                       host_cache.prefetch_distance);
        const int initial_anchor = std::min(n_layer, std::max(0, host_cache.prefetch_distance - 1));
        if (initial_anchor < n_layer) {
            const int initial_end = llama_prefill_host_cache_window_end(host_cache, initial_anchor, n_layer);
            for (int il = initial_anchor; il <= initial_end; ++il) {
                if (!host_cache.advice_supported) {
                    break;
                }
                llama_prefill_host_cache_prefetch_layer(host_cache, il);
            }
        }
    }

    // ====================================================================
    // Step 2: Allocate 2 GPU buffers sized for the largest layer
    // ====================================================================

    // Get the first GPU backend from the context
    ggml_backend_t gpu_backend = nullptr;
    {
        auto * sched = ctx->get_sched();
        if (!sched) {
            LLAMA_LOG_ERROR("%s: [DIAG] get_sched() returned null!\n", __func__);
            return -1;
        }
        // Try to find a non-CPU backend (Metal)
        int n_backends = ggml_backend_sched_get_n_backends(sched);
        LLAMA_LOG_INFO("%s: [DIAG] scheduler has %d backends\n", __func__, n_backends);
        for (int i = 0; i < n_backends; ++i) {
            ggml_backend_t b    = ggml_backend_sched_get_backend(sched, i);
            const char *   name = ggml_backend_name(b);
            LLAMA_LOG_INFO("%s: [DIAG]   backend[%d] = %s\n", __func__, i, name ? name : "(null)");
            if (name && strstr(name, "CPU") == nullptr && strstr(name, "cpu") == nullptr) {
                gpu_backend = b;
                break;
            }
        }
        if (!gpu_backend && n_backends > 0) {
            // Fallback: use backend 0 even if it's CPU (will fail on buffer alloc)
            gpu_backend = ggml_backend_sched_get_backend(sched, 0);
            LLAMA_LOG_WARN("%s: [DIAG] no non-CPU backend found, using backend[0]\n", __func__);
        }
    }

    if (!gpu_backend) {
        LLAMA_LOG_ERROR("%s: no GPU backend available\n", __func__);
        return -1;
    }
    LLAMA_LOG_INFO("%s: [DIAG] using backend: %s\n", __func__, ggml_backend_name(gpu_backend));

#ifdef GGML_USE_CUDA
    llama_cuda_backend_graph_disable_scope cuda_graph_disable_scope(gpu_backend, params.disable_cuda_graphs);
    if (params.disable_cuda_graphs && ggml_backend_is_cuda(gpu_backend)) {
        LLAMA_LOG_INFO("%s: CUDA graphs disabled for this streaming pass\n", __func__);
    }
#endif

    // Compute GPU alloc offsets using the backend's buffer type.
    // This sets alloc_size, alloc_offset, and re-computes total_bytes
    // to include backend-specific padding (e.g., MATRIX_ROW_PADDING for CUDA).
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
        ggml_backend_buffer_type_t gpu_buft = dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
        if (gpu_buft) {
            prefill_compute_alloc_offsets(layer_infos, max_layer_bytes, gpu_buft);
        }
    }

    ggml_backend_buffer_t gpu_buf_a = nullptr;
    ggml_backend_buffer_t gpu_buf_b = nullptr;

    // When the weight fetcher is active, it has its own GPU buffers for DMA.
    // The gpu_buf_a/b only need to exist as ggml buffer wrappers (for scheduler
    // backend assignment). Allocate minimal size (1MB) to save VRAM.
    // Without WF, allocate full-size for direct tensor uploads.
    size_t gpu_buf_alloc_size = max_layer_bytes;

    // Phase 328 (2026-05-05): if persistent buffers exist from a prior call,
    // reuse them. LLAMA_PERSIST_GPU_BUFS=1 enables this. The size must match
    // (model-dependent) — invalidate persistent buffers if size changes.
    const char * persist_env = std::getenv("LLAMA_PERSIST_GPU_BUFS");
    const bool persist_bufs = persist_env && persist_env[0] && persist_env[0] != '0';
    bool reused_persist_a = false;
    bool reused_persist_b = false;
    if (persist_bufs) {
        ggml_backend_buffer_t pa = ctx->get_persist_gpu_buf_a();
        ggml_backend_buffer_t pb = ctx->get_persist_gpu_buf_b();
        size_t ps = ctx->get_persist_gpu_buf_size();
        if (pa && ps == gpu_buf_alloc_size) {
            gpu_buf_a = pa;
            reused_persist_a = true;
            if (pb) {
                gpu_buf_b = pb;
                reused_persist_b = true;
            }
            LLAMA_LOG_INFO("%s: Phase 328 - reusing persistent gpu_buf_a/b (size=%.2f MB)\n",
                           __func__, gpu_buf_alloc_size / (1024.0 * 1024.0));
        } else if (pa && ps != gpu_buf_alloc_size) {
            // Size mismatch: free old buffers, will allocate new ones below
            LLAMA_LOG_INFO("%s: Phase 328 - persistent buffer size mismatch (%.2f vs %.2f MB), reallocating\n",
                           __func__, ps / (1024.0 * 1024.0), gpu_buf_alloc_size / (1024.0 * 1024.0));
            ggml_backend_buffer_free(pa);
            if (pb) ggml_backend_buffer_free(pb);
            ctx->set_persist_gpu_buf_a(nullptr);
            ctx->set_persist_gpu_buf_b(nullptr);
            ctx->set_persist_gpu_buf_size(0);
        }
    }

    {
        size_t vf = 0, vt = 0;
        if (ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend)) {
            ggml_backend_dev_memory(dev, &vf, &vt);
        }
        fprintf(stderr, "*** PRE_WF_ALLOC: VRAM free=%.0f MB total=%.0f MB, need=%.0f MB × %d bufs\n",
                vf/(1024.0*1024.0), vt/(1024.0*1024.0),
                gpu_buf_alloc_size/(1024.0*1024.0), gpu_buf_b ? 2 : 1);
        fflush(stderr);
    }
    // Allocate a small guard buffer to prevent CUDA allocator from placing
    // WF compute buffers overlapping with the non-layer tensor buffer.
    // Without this guard, cudaMalloc can return addresses that overlap by ~1 MB
    // due to alignment rounding in the ggml buffer layer vs CUDA's allocator.
    ggml_backend_buffer_t guard_buf = nullptr;
    {
        auto buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(gpu_backend));
        guard_buf = ggml_backend_buft_alloc_buffer(buft, 2 * 1024 * 1024);  // 2 MB guard
    }

    // Respect LLAMA_WF_GPU_BUFS: allocate 1 or 2 rotation buffers
    const char * gpu_bufs_env = std::getenv("LLAMA_WF_GPU_BUFS");
    const int n_gpu_bufs = (gpu_bufs_env && gpu_bufs_env[0] == '1') ? 1 : 2;
    // R3 DS-direct edit 18: when LLAMA_WF_DS_DIRECT, the compute buffer (gpu_buf_a) is the WF's PERMANENT
    // D3D12-heap compute buffer, allocated inside wf_create/wf_dstorage_init. We must NOT cudaMalloc one here
    // (would waste 8.5GB + be clobbered); instead we wrap the WF heap ptr as gpu_buf_a AFTER wf_create below.
    const char * ds_direct_env = std::getenv("LLAMA_WF_DS_DIRECT");
    const char * wf_env_chk     = std::getenv("LLAMA_PREFILL_WF");
    const bool wf_on_chk        = wf_env_chk && wf_env_chk[0] && std::strcmp(wf_env_chk, "0") != 0;
    // DS-direct is a WF-only feature: only defer the alloc when the WF path is actually requested,
    // otherwise gpu_buf_a would be left null with no WF to provide the heap.
    const bool ds_direct = ds_direct_env && ds_direct_env[0] && ds_direct_env[0] != '0' && wf_on_chk;
    // R3 DS-direct (corrected): the COMPUTE buffer is allocated NORMALLY (cudaMalloc) even under ds_direct —
    // it must NOT be a D3D12-imported heap (that double-accounts VRAM and OOMs). Only the WF's PREFETCH is a
    // D3D heap. So we DO allocate gpu_buf_a here and wire it to the WF via wf_set_gpu_buf_ptr (legacy path).
    if (reused_persist_a) {
        // Phase 328: skip allocation, persistent buffers already assigned
        LLAMA_LOG_INFO("%s: skipping fresh GPU buffer alloc (Phase 328 persist mode)\n", __func__);
    } else if (n_gpu_bufs == 1 || ds_direct) {
        // Single buffer mode: allocate only buf_a, leave buf_b null.
        // R3 DS-direct: only buf_a is used (the WF computes from gpu_bufs[0]); a second rotation buffer
        // (buf_b) would waste a full ~8.6GB of VRAM. Force single-buffer alloc under ds_direct.
        auto buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(gpu_backend));
        size_t alloc_size = gpu_buf_alloc_size + gpu_buf_alloc_size / 10;
        gpu_buf_a = ggml_backend_buft_alloc_buffer(buft, alloc_size);
        gpu_buf_b = nullptr;
        if (!gpu_buf_a) {
            LLAMA_LOG_ERROR("%s: failed to allocate single GPU buffer (%.2f MB)\n", __func__, alloc_size / (1024.0 * 1024.0));
            if (guard_buf) ggml_backend_buffer_free(guard_buf);
            return -1;
        }
        ggml_backend_buffer_set_usage(gpu_buf_a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        LLAMA_LOG_INFO("%s: allocated 1 GPU buffer: %.2f MB\n", __func__, alloc_size / (1024.0 * 1024.0));
    } else if (!prefill_allocate_gpu_buffers(gpu_backend, gpu_buf_alloc_size, &gpu_buf_a, &gpu_buf_b)) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffers\n", __func__);
        if (guard_buf) ggml_backend_buffer_free(guard_buf);
        return -1;
    }
    // Free guard — its purpose was just to push the allocator past any overlap
    if (guard_buf) ggml_backend_buffer_free(guard_buf);

    // Phase 328: store newly-allocated buffers as persistent if env enabled
    // (R3 DS-direct: skip — gpu_buf_a is the WF's permanent heap, not a ggml-owned buffer to persist here.)
    if (persist_bufs && !reused_persist_a && !ds_direct) {
        ctx->set_persist_gpu_buf_a(gpu_buf_a);
        ctx->set_persist_gpu_buf_b(gpu_buf_b);
        ctx->set_persist_gpu_buf_size(gpu_buf_alloc_size);
        LLAMA_LOG_INFO("%s: Phase 328 - stored as persistent gpu_buf_a/b (size=%.2f MB)\n",
                       __func__, gpu_buf_alloc_size / (1024.0 * 1024.0));
    }

    // These are declared here (before the pipeline init and cleanup lambda) so
    // they're in scope for both the pipeline init block and the cleanup lambda.
    void * cuda_pipeline = nullptr;
    void * pinned_a      = nullptr;
    void * pinned_b      = nullptr;

    LLAMA_LOG_INFO("%s: [DIAG] GPU buffers allocated successfully\n", __func__);

    // ====================================================================
    // Step 2a-WF: Weight fetcher pipeline (optional, env-controlled)
    //
    // When LLAMA_PREFILL_WF=1, use the weight fetcher for NVMe→GPU DMA
    // instead of the mmap→pinned→H2D pipeline.  The WF manages its own
    // GPU buffers and I/O threads.  We still use gpu_buf_a/b for the
    // scheduler remap trick (tensor->buffer must be a recognized GPU
    // buffer for dispatch routing).
    // ====================================================================

    wf_context * wf = nullptr;
    {
        const char * wf_env = std::getenv("LLAMA_PREFILL_WF");
        const bool use_wf_requested = wf_env && wf_env[0] != '\0' && std::strcmp(wf_env, "0") != 0;

        if (use_wf_requested && !model->primary_model_path.empty()) {
            // Phase 250 (2026-05-03): WF cache reuse opt-out for safety.
            // Diagnosis: cached WF retains DirectStorage / cuFile descriptors
            // bound to the previous request's GPU buffer VAs. New request
            // allocates fresh ggml GPU buffers at potentially-different VAs;
            // wf_set_gpu_buf_ptr only updates top-level base pointers, not
            // the underlying registered descriptors. In-flight DMA from the
            // re-started pipeline writes to STALE VAs → MMU fault → GSP
            // wedge → "GPU is lost" with full unrecoverability (Windows
            // can't even Disable-PnpDevice the device after this).
            //
            // Symptom: ub=4096 path crashes at second-or-later request that
            // gets a fresh buffer alloc that diverges from the previous
            // request's VA. Reproducible at ~8.5k token prompts on Kimi K2.6.
            //
            // Default: cache reuse OFF for safety. Set
            // LLAMA_PREFILL_WF_REUSE_CACHED_DANGEROUS=1 to opt back in for
            // bench-only, single-shot workloads where the bug doesn't fire.
            const char * wf_reuse_env = std::getenv("LLAMA_PREFILL_WF_REUSE_CACHED_DANGEROUS");
            const bool allow_cache_reuse = wf_reuse_env && wf_reuse_env[0] != '\0'
                                       && std::strcmp(wf_reuse_env, "0") != 0;
            wf = allow_cache_reuse ? (wf_context *)ctx->get_cached_wf() : nullptr;
            if (wf) {
                // Each streaming request allocates fresh ggml GPU buffers. When
                // reusing the cached WF, re-point it at the current request's
                // buffer bases before restarting the pipeline.
                if (gpu_buf_a) {
                    wf_set_gpu_buf_ptr(wf, 0, ggml_backend_buffer_get_base(gpu_buf_a));
                }
                if (gpu_buf_b) {
                    wf_set_gpu_buf_ptr(wf, 1, ggml_backend_buffer_get_base(gpu_buf_b));
                }
                // Reset pipeline for next pass through the layers
                wf_end_pass(wf);
                LLAMA_LOG_INFO("%s: reusing cached weight fetcher for batch %d\n", __func__, batch.n_tokens);
            } else {
                const char * wf_trace_env = std::getenv("LLAMA_PREFILL_WF_TRACE");
                const bool wf_trace = wf_trace_env && wf_trace_env[0] != '\0'
                                   && std::strcmp(wf_trace_env, "0") != 0;

                // Build split paths array for the adapter.
                // split_file_paths[0] is the primary file (already passed as model_path),
                // so skip it — the WF opens model_path separately as ds_files[0].
                // file_idx from the model loader: 0=primary, 1..N=splits.
                // ds_files layout: [0]=model_path, [1..N]=split_ptrs[0..N-1].
                // So split_ptrs must start from split_file_paths[1].
                std::vector<const char *> split_ptrs;
                for (size_t i = 1; i < model->split_file_paths.size(); i++) {
                    split_ptrs.push_back(model->split_file_paths[i].c_str());
                }

                // Convert model weight locations to adapter format
                std::unordered_map<std::string, wf_adapter_weight_loc> weight_locs;
                for (const auto & [name, loc] : model->weight_file_locs) {
                    weight_locs[name] = { loc.file_idx, loc.file_offset };
                }

                // Use sub-layer infos if splitting, else original layer infos
                auto & wf_infos = (n_sublayers > 1 && !sublayer_infos.empty())
                                  ? sublayer_infos : layer_infos;
                size_t wf_max_bytes = (n_sublayers > 1) ? max_sublayer_bytes : max_layer_bytes;

                wf = wf_adapter_create(
                    wf_infos, wf_max_bytes,
                    model->primary_model_path.c_str(),
                    split_ptrs.empty() ? nullptr : split_ptrs.data(),
                    (int)split_ptrs.size(),
                    gpu_backend,
                    wf_trace,
                    &weight_locs);

                if (wf) {
                    // R3 DS-direct (corrected): the COMPUTE buffer (gpu_buf_a) is a normal cudaMalloc ggml buffer
                    // (allocated above) for BOTH legacy and DS-direct — only the WF's PREFETCH is a D3D heap. So
                    // we wire gpu_buf_a to the WF the same way in both modes (no heap-wrap; that double-accounted).
                    // Set WF compute buffer ptrs to the ggml buffers.
                    if (gpu_buf_a) {
                        wf_set_gpu_buf_ptr(wf, 0, ggml_backend_buffer_get_base(gpu_buf_a));
                    }
                    if (gpu_buf_b) {
                        wf_set_gpu_buf_ptr(wf, 1, ggml_backend_buffer_get_base(gpu_buf_b));
                    }
                    wf_transfer_mode eff = wf_get_effective_mode(wf);
                    const char * mode_names[] = { "mmap", "direct_io", "gds", "dstorage", "auto" };
                    LLAMA_LOG_INFO("%s: weight fetcher active (mode=%s)\n", __func__,
                                   (int)eff < 5 ? mode_names[(int)eff] : "unknown");
                    // Phase 250 (2026-05-03): only cache when explicitly opted-in.
                    // Default (cache OFF) means each request gets a fresh WF
                    // and the cleanup path at end of request will wf_destroy()
                    // it (because get_cached_wf() != wf when cached_wf is null).
                    if (allow_cache_reuse) {
                        ctx->set_cached_wf(wf);
                    }
                } else {
                    LLAMA_LOG_WARN("%s: weight fetcher creation failed, falling back to mmap upload\n", __func__);
                }
            }
        } else if (use_wf_requested) {
            LLAMA_LOG_WARN("%s: LLAMA_PREFILL_WF=1 but no model path stored, falling back to mmap upload\n", __func__);
        }
    }
    const bool use_wf = (wf != nullptr);
    bool wf_started = false;

    // ---- Layer-major mode (env-controlled) ----
    // When LLAMA_PREFILL_LAYER_MAJOR=1 and WF is active, invert the compute
    // loop: for each layer, process ALL ubatches, then move to the next layer.
    // This streams each layer's weights exactly once instead of N times.
    const char * lm_env = std::getenv("LLAMA_PREFILL_LAYER_MAJOR");
    const bool use_layer_major = use_wf && lm_env && lm_env[0] != '\0' && std::strcmp(lm_env, "0") != 0;
    if (use_layer_major) {
        wf_set_layer_major_mode(wf, true);
        LLAMA_LOG_INFO("%s: layer-major ubatch processing ENABLED\n", __func__);
    }

    // ---- Hidden state storage mode (env-controlled) ----
    // LLAMA_PREFILL_HIDDEN_MODE controls where hidden states are stored between
    // layers in layer-major mode:
    //   "cpu"    or "0" = CPU RAM, synchronous PCIe transfers (default)
    //   "gpu"    or "1" = GPU buffer, GPU→GPU copies (~900 GB/s)
    //   "dblbuf" or "2" = Double-buffer: 2 GPU staging tensors + CPU RAM
    llama_hidden_mode hidden_mode = LLAMA_HIDDEN_CPU;
    if (use_layer_major) {
        const char * hm_env = std::getenv("LLAMA_PREFILL_HIDDEN_MODE");
        if (hm_env && hm_env[0] != '\0') {
            if (std::strcmp(hm_env, "gpu") == 0 || std::strcmp(hm_env, "1") == 0) {
                hidden_mode = LLAMA_HIDDEN_GPU;
            } else if (std::strcmp(hm_env, "dblbuf") == 0 || std::strcmp(hm_env, "2") == 0) {
                hidden_mode = LLAMA_HIDDEN_DBLBUF;
            }
        }
        const char * hm_names[] = { "cpu", "gpu", "dblbuf" };
        LLAMA_LOG_INFO("%s: hidden state mode: %s\n", __func__, hm_names[hidden_mode]);
    }

    // ---- KV cache location mode (env-controlled) ----
    // LLAMA_PREFILL_KV_MODE=gpu|ram|auto is the explicit placement control.
    // LLAMA_PREFILL_KV_RAM remains as a legacy alias (1|ram => RAM, 0|gpu => GPU).
    // Only meaningful in layer-major mode.
    llama_kv_mode kv_mode = LLAMA_KV_GPU;
    if (use_layer_major) {
        const auto kv_cfg = llama_prefill_resolve_kv_mode_from_env();
        kv_mode = kv_cfg.mode;
        const char * kv_names[] = { "gpu", "ram" };
        fprintf(stderr, "*** PREFILL: KV cache mode: %s (forced=%d source=%s value=%s)\n",
                kv_names[kv_mode], (int) kv_cfg.forced, kv_cfg.source,
                kv_cfg.value ? kv_cfg.value : "NULL");
        if (kv_cfg.explicit_env && !kv_cfg.recognized) {
            fprintf(stderr, "*** PREFILL: unrecognized %s=%s, defaulting to gpu\n",
                    kv_cfg.source, kv_cfg.value ? kv_cfg.value : "NULL");
        }
    }

    // NVMe→RAM→GPU bandwidth tests removed — results in docs/nvme-ram-gpu-benchmarks.md
    // Best: IOCP 25.5 GB/s NVMe→RAM, 19.8 GB/s pipelined two-hop, DS ~20 GB/s.

    // ====================================================================
    // Step 2a: CUDA dual-stream pipeline + pinned staging buffers
    //
    // Allocate a dedicated copy stream and pinned (page-locked) host staging
    // buffers so that H2D transfers are truly asynchronous and can overlap
    // with compute on a separate stream.  Three pipeline stages:
    //   Stage 1 (CPU thread) : memcpy mmap → pinned staging   [layer N+2]
    //   Stage 2 (copy stream): cudaMemcpyAsync pinned → GPU   [layer N+1]
    //   Stage 3 (compute)    : kernel execution                [layer N]
    //
    // Skipped when the weight fetcher is active — WF manages its own
    // DMA pipeline (DirectStorage / direct I/O → GPU).
    // ====================================================================

#ifdef GGML_USE_CUDA
    if (!use_wf && ggml_backend_is_cuda(gpu_backend)) {
        cuda_pipeline = ggml_backend_cuda_prefill_pipeline_init(gpu_backend);
        if (cuda_pipeline) {
            pinned_a = ggml_backend_cuda_prefill_pinned_alloc(max_layer_bytes + max_layer_bytes / 10);
            pinned_b = ggml_backend_cuda_prefill_pinned_alloc(max_layer_bytes + max_layer_bytes / 10);
            if (!pinned_a || !pinned_b) {
                LLAMA_LOG_WARN("%s: pinned alloc failed (%.2f MB each), falling back to pageable uploads\n",
                               __func__, max_layer_bytes / (1024.0 * 1024.0));
                if (pinned_a) { ggml_backend_cuda_prefill_pinned_free(pinned_a); pinned_a = nullptr; }
                if (pinned_b) { ggml_backend_cuda_prefill_pinned_free(pinned_b); pinned_b = nullptr; }
                ggml_backend_cuda_prefill_pipeline_free(cuda_pipeline);
                cuda_pipeline = nullptr;
            } else {
                if (prefill_debug >= 1) {
                    fprintf(stderr, "*** PREFILL PIPELINE: dual-stream enabled, 2x %.2f MB pinned staging ***\n",
                            max_layer_bytes / (1024.0 * 1024.0));
                }
            }
        }
    }
#endif
    const bool use_pipeline = (cuda_pipeline != nullptr);

    // ====================================================================
    // Step 2b: Persistent GPU buffer for non-layer tensors
    //
    // Embeddings (tok_embd) and the output head (output_norm, output) are
    // NOT part of the per-layer rotation.  They must live on GPU for the
    // entire prefill pass so the scheduler routes embedding lookup and
    // output projection to the GPU backend.  Without this, the output
    // projection (vocab_size × hidden_dim matmul for all tokens) runs on
    // CPU — catastrophically slow for large batches.
    // ====================================================================

    ggml_backend_buffer_t gpu_buf_persist = nullptr;
    std::vector<tensor_remap_entry> persist_entries;
    bool persist_remap_applied = false;
    bool persist_load_failed = false;
    std::vector<uint8_t> persist_file_buf; // temp buffer for file-based loading
    // Pristine tensor->data per persist entry, snapshotted BEFORE the file-load
    // path may overwrite entry.orig_data with a pointer into persist_file_buf
    // (a function-local freed on return). Teardown restores from THIS, never
    // from the transient orig_data, so the model tensor is never left dangling
    // into freed heap across chunk boundaries. See 0x47c4b3 chunk-boundary AV.
    std::vector<void *> persist_restore_data;
    {
        size_t persist_total_bytes = 0;
        collect_persistent_tensors(model, persist_entries, persist_total_bytes);

        if (persist_total_bytes > 0) {
            // Snapshot the pristine tensor->data (nullptr for no_alloc entries)
            // for every persistent tensor BEFORE the file-load path below may
            // redirect entry.orig_data into the function-local persist_file_buf.
            // Teardown restores from this snapshot, so on chunk re-entry the
            // model tensor still shows its original nullptr sentinel and
            // re-triggers need_file_load / re-reads from file instead of
            // uploading from freed heap. Root cause of the 0x47c4b3 AV.
            persist_restore_data.clear();
            persist_restore_data.reserve(persist_entries.size());
            for (const auto & e : persist_entries) {
                persist_restore_data.push_back(e.orig_data);
            }
            // When no_alloc mode is active (LLAMA_PREFILL_WF=1), tensor data
            // pointers are NULL because mmap was skipped.  Load persistent
            // tensor data directly from the GGUF file into a temp CPU buffer.
            bool need_file_load = false;
            for (const auto & entry : persist_entries) {
                if (entry.orig_data == nullptr) {
                    need_file_load = true;
                    break;
                }
            }

            if (need_file_load && model->primary_model_path.empty()) {
                LLAMA_LOG_ERROR("%s: cannot load persistent tensors: primary model path is empty\n", __func__);
                persist_load_failed = true;
            } else if (need_file_load) {
                persist_file_buf.resize(persist_total_bytes);
                size_t buf_offset = 0;
                size_t loaded_tensors = 0;
                size_t loaded_bytes = 0;

                for (auto & entry : persist_entries) {
                    if (entry.orig_data != nullptr) {
                        buf_offset += entry.nbytes;
                        continue;
                    }

                    const char * tname = entry.tensor->name;
                    auto it = model->weight_file_locs.find(tname);
                    if (it == model->weight_file_locs.end()) {
                        LLAMA_LOG_ERROR("%s: persistent tensor '%s' not found in weight_file_locs\n",
                                       __func__, tname ? tname : "(null)");
                        persist_load_failed = true;
                        break;
                    }

                    const auto & loc = it->second;
                    // TASK-A ROOT CAUSE FIX (2026-07-21): weight_file_locs file_idx indexes
                    // split_file_paths DIRECTLY — split_file_paths[0] IS the primary file
                    // (llama.cpp:879 stores the loader's splits list, which includes the main
                    // file at [0]; same contract as the WF adapter and L1 dense-resident).
                    // The old `file_idx - 1` mapping read shard N's offsets from shard N-1:
                    // every persistent tensor in a non-first shard (first victim:
                    // blk.20.attn_sinks, shard 00002) loaded garbage under keep_mmap=0 —
                    // 2 NaN sink logits => 2 NaN heads in the joint CSA+LID flash attention
                    // => hidden 100% NaN from layer 20 => every streamed prompt decoded
                    // '<DDD'. keep_mmap=1 masked it (mmap view short-circuits the file read).
                    const std::string & fpath =
                        (loc.file_idx < model->split_file_paths.size())
                            ? model->split_file_paths[loc.file_idx]
                            : model->primary_model_path;

                    FILE * f = fopen(fpath.c_str(), "rb");
                    if (!f) {
                        LLAMA_LOG_ERROR("%s: failed to open '%s' for persistent tensor '%s'\n",
                                       __func__, fpath.c_str(), tname);
                        persist_load_failed = true;
                        break;
                    }

#ifdef _WIN32
                    const int seek_rc = _fseeki64(f, (int64_t)loc.file_offset, SEEK_SET);
#else
                    const int seek_rc = fseeko(f, (off_t)loc.file_offset, SEEK_SET);
#endif
                    if (seek_rc != 0) {
                        LLAMA_LOG_ERROR("%s: failed to seek '%s' to offset %zu for persistent tensor '%s'\n",
                                       __func__, fpath.c_str(), loc.file_offset, tname);
                        fclose(f);
                        persist_load_failed = true;
                        break;
                    }
                    size_t nread = fread(persist_file_buf.data() + buf_offset, 1, entry.nbytes, f);
                    fclose(f);

                    if (nread != entry.nbytes) {
                        LLAMA_LOG_ERROR("%s: short read for '%s': got %zu / %zu bytes\n",
                                       __func__, tname, nread, entry.nbytes);
                        persist_load_failed = true;
                        break;
                    }

                    entry.orig_data = persist_file_buf.data() + buf_offset;
                    buf_offset += entry.nbytes;
                    loaded_tensors++;
                    loaded_bytes += entry.nbytes;
                }

                LLAMA_LOG_INFO("%s: loaded %zu persistent tensors from file (%.2f MB)\n",
                               __func__, loaded_tensors,
                               loaded_bytes / (1024.0 * 1024.0));
            }

            if (!persist_load_failed) {
                // Use the same buffer type as the ping-pong buffers
                ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
                ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
                if (buft) {
                    // 10% padding for alignment, matching ping-pong buffer allocation
                    const size_t alloc_size = persist_total_bytes + persist_total_bytes / 10;
                    gpu_buf_persist = ggml_backend_buft_alloc_buffer(buft, alloc_size);
                }
                if (!gpu_buf_persist) {
                    LLAMA_LOG_WARN("%s: failed to allocate persistent GPU buffer (%.2f MB), "
                                   "non-layer ops will run on CPU\n",
                                   __func__, persist_total_bytes / (1024.0 * 1024.0));
                } else {
                    ggml_backend_buffer_set_usage(gpu_buf_persist, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    void * persist_base = ggml_backend_buffer_get_base(gpu_buf_persist);
                    size_t offset = 0;
                    for (auto & entry : persist_entries) {
                        entry.tensor->buffer = gpu_buf_persist;
                        entry.tensor->data   = (char *) persist_base + offset;
                        ggml_backend_tensor_set(entry.tensor, entry.orig_data, 0, entry.nbytes);
                        offset += entry.nbytes;
                    }
                    persist_remap_applied = true;
                    LLAMA_LOG_INFO("%s: uploaded %zu persistent tensors (%.2f MB) to GPU\n",
                                   __func__, persist_entries.size(),
                                   persist_total_bytes / (1024.0 * 1024.0));
                }
            }
        }
    }

    // ====================================================================
    // Step 3: Remap → Build Graph → Restore
    //
    // We temporarily make all weight tensors look GPU-resident so the
    // scheduler routes compute to GPU. The callbacks then keep each layer's
    // pointers redirected to the active ping-pong buffer.
    // ====================================================================

    // For WF path: SKIP the GPU remap. Leave weight tensors on CPU buffer
    // so the scheduler creates input_cpy tensors (proper GPU copies with
    // alloc_size padding and init_tensor handling). This matches exactly
    // what the non-streaming mmap path does — the scheduler handles everything.
    // The per-layer callback sets tensor->data to WF GPU buffer, but keeps
    // tensor->buffer as CPU. The scheduler copies data GPU→GPU transparently.
    if (persist_load_failed) {
        LLAMA_LOG_ERROR("%s: persistent tensor file load failed; aborting streaming prefill\n", __func__);
    } else if (use_wf) {
        void * wf_buf_base = wf_get_gpu_buf_ptr(wf, 0);
        size_t wf_buf_cap  = wf_get_gpu_buf_capacity(wf);
        if (!wf_buf_base || wf_buf_cap == 0) {
            // wf_buf_cap is the divisor in offset % wf_buf_cap. A zero value
            // means per-slot initialization was skipped or failed. Leave the
            // tensors untouched and diagnose the lifecycle error without
            // raising STATUS_INTEGER_DIVIDE_BY_ZERO.
            LLAMA_LOG_ERROR("%s: invalid WF GPU buffer state (base=%p capacity=%zu); "
                            "skipping WF tensor remap\n",
                            __func__, wf_buf_base, wf_buf_cap);
        } else {
            for (auto & info : layer_infos) {
                size_t offset = 0;
                for (auto & entry : info.entries) {
                    entry.orig_buffer = entry.tensor->buffer;
                    entry.orig_data   = entry.tensor->data;
                    if ((uintptr_t)entry.orig_data <= 1) entry.orig_data = nullptr;
                    entry.tensor->buffer = gpu_buf_a;
                    // Point to WF buffer at the correct per-tensor offset.
                    // All layers share the same WF buffer (reused per layer),
                    // so all layers' tensors point to overlapping ranges.
                    // This tells the allocator to avoid the WF buffer range.
                    size_t data_offset = offset % wf_buf_cap;
                    entry.tensor->data = (char *)wf_buf_base + data_offset;
                    offset += entry.nbytes;
                }
            }
            LLAMA_LOG_INFO("%s: [DIAG] tensor remap to WF buffer complete\n", __func__);
        }
    } else {
        prefill_remap_tensors_to_gpu(layer_infos, gpu_buf_a);
        LLAMA_LOG_INFO("%s: [DIAG] tensor remap complete\n", __func__);
    }

    std::vector<float> upload_times(n_layer, 0.0f);
    std::vector<float> stall_times(n_layer, 0.0f);
    std::vector<float> compute_times(n_layer, 0.0f);

    struct pending_upload_t {
        int               layer_idx = -1;
        bool              active    = false;
        std::future<float> fut;
    };

    pending_upload_t pending_upload;
    std::vector<bool> layer_loaded(n_layer, false);
    auto elapsed_ms = [&]() -> float {
        return std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t_total_start).count();
    };

    bool remap_applied       = !persist_load_failed;
    bool callbacks_registered = false;
    const bool saved_streaming_flag = ctx->is_prefill_streaming();
    const char * force_single_env = std::getenv("LLAMA_PREFILL_FORCE_SINGLE_BUFFER");
    const bool force_single_buffer =
        (force_single_env != nullptr && force_single_env[0] != '\0' && std::strcmp(force_single_env, "0") != 0) ||
        (gpu_buf_b == nullptr);  // Single buffer mode (LLAMA_WF_GPU_BUFS=1)

    const auto cleanup = llama_scope_exit([&]() {
        if (callbacks_registered) {
            ctx->set_per_layer_callbacks(nullptr, nullptr);
            ctx->set_lid_topk_callback(nullptr);
        }
        if (pending_upload.active) {
            pending_upload.fut.wait();
            pending_upload.active = false;
        }
        if (remap_applied) {
            prefill_restore_tensor_pointers(layer_infos);
        }
        if (persist_remap_applied) {
            for (size_t i = 0; i < persist_entries.size(); ++i) {
                auto & entry = persist_entries[i];
                entry.tensor->buffer = entry.orig_buffer;
                // Restore the pristine pre-file-load pointer, NOT the transient
                // persist_file_buf pointer (freed on return). Prevents the model
                // tensor from dangling into freed heap on the next chunk's
                // re-entry (0x47c4b3 chunk-boundary AV in cuMemcpyHtoDAsync).
                entry.tensor->data = (i < persist_restore_data.size())
                    ? persist_restore_data[i]
                    : entry.orig_data;
            }
        }
        // CRITICAL: synchronize the GPU before freeing buffers.
        // CUDA kernels are asynchronous — without this, freeing a buffer
        // while a kernel still reads from it causes "illegal memory access"
        // that surfaces as a latent error on the NEXT CUDA call.
        if (gpu_backend) {
            ggml_backend_synchronize(gpu_backend);
        }
        // Shut down weight fetcher pipeline (if active)
        if (wf) {
            wf_stop(wf);
            wf_quiesce(wf);
            wf_print_stats(wf);
            // Free the prefetch buffer to release VRAM for next request.
            // The WF itself stays cached for reuse, but the prefetch buffer
            // is re-allocated each pass based on available VRAM.
            wf_free_prefetch(wf);
            // Don't destroy if cached — will be reused by next batch or
            // destroyed on context teardown. DirectStorage resources can't
            // be freed mid-session without CUDA interop errors.
            if (ctx->get_cached_wf() != wf) {
                wf_destroy(wf);
            }
            wf = nullptr;
        }
        // Prevent graph reuse with stale GPU buffer pointers.
        // The previous graph may reference the streaming GPU buffers we're
        // about to free.  Only resets gf_res_prev — does NOT trigger
        // sched_reserve (which would destroy and recreate the scheduler,
        // causing cuBLAS crashes if another streaming prefill follows).
        ctx->invalidate_graph();
        if (cuda_pipeline) {
#ifdef GGML_USE_CUDA
            ggml_backend_cuda_prefill_sync_copy_stream(cuda_pipeline);
            ggml_backend_cuda_prefill_pipeline_free(cuda_pipeline);
#endif
            cuda_pipeline = nullptr;
        }
#ifdef GGML_USE_CUDA
        if (pinned_a) { ggml_backend_cuda_prefill_pinned_free(pinned_a); pinned_a = nullptr; }
        if (pinned_b) { ggml_backend_cuda_prefill_pinned_free(pinned_b); pinned_b = nullptr; }
#endif
        {
            size_t free_before = 0, total = 0;
            wf_cuda_get_vram_info(&free_before, &total);
            // Phase 328: skip free if buffers are persisted across requests
            if (!persist_bufs) {
                if (gpu_buf_a) ggml_backend_buffer_free(gpu_buf_a);
                if (gpu_buf_b) ggml_backend_buffer_free(gpu_buf_b);
            } else {
                LLAMA_LOG_INFO("%s: Phase 328 - SKIPPING gpu_buf_a/b free (persist mode)\n", __func__);
            }
            if (gpu_buf_persist) ggml_backend_buffer_free(gpu_buf_persist);
            size_t free_after_bufs = 0;
            wf_cuda_get_vram_info(&free_after_bufs, &total);
            // Phase 296 (2026-05-04): release the CUDA backend's per-stream
            // memory pool too. The legacy pool (used when GGML_CUDA_NO_VMM=1)
            // caches up to MAX_BUFFERS=256 freed compute buffers indefinitely,
            // accumulating 20+ GB across streaming-prefill cycles. Without this
            // the second consecutive prefill request OOMs at PRE_WF_ALLOC.
            size_t pool_released = 0;
            // Phase 2.3b (2026-05-12): snapshot AFTER pool_clear and AFTER
            // graphs_clear separately to bisect the 500 MB/req leak at MAX ctx.
            // (Phase 296+316i collapse them; we need them split for diagnosis.)
            size_t free_after_pool = 0;
#ifdef GGML_USE_CUDA
            if (gpu_backend) {
                pool_released = ggml_backend_cuda_pool_clear(gpu_backend);
                wf_cuda_get_vram_info(&free_after_pool, &total);
                // Phase 316i (2026-05-04): clear the cuda_graphs map too.
                // The map holds node-ptr keys from the prior request's graph;
                // those pointers reference compute-graph memory that's about
                // to be freed by ggml_backend_buffer_free above. Without this
                // clear, the next request's graph node may reuse a stale ptr
                // address, hit the cached entry, and access freed memory
                // (illegal memory access at request 4 symptom). Clearing
                // forces a fresh entry for every node in the next request.
                ggml_backend_cuda_graphs_clear(gpu_backend);
            }
#endif
            size_t free_after = 0;
            wf_cuda_get_vram_info(&free_after, &total);
            // Phase 299: report per-process cuda buffer accounting too,
            // so we can spot leaks at the ggml-wrapper level (in addition to
            // the CUDA driver's free VRAM stats).
            int64_t cuda_alloc_total = 0;
            int64_t cuda_free_total  = 0;
            int64_t cuda_live_count  = 0;
            int64_t dev_malloc_bytes = 0;
            int64_t dev_malloc_count = 0;
            int64_t cuda_graphs_total = 0;
            int64_t event_creates = 0;
            int64_t event_destroys = 0;
#ifdef GGML_USE_CUDA
            cuda_alloc_total = ggml_backend_cuda_buffer_alloc_total();
            cuda_free_total  = ggml_backend_cuda_buffer_free_total();
            cuda_live_count  = ggml_backend_cuda_buffer_live_count();
            dev_malloc_bytes = ggml_cuda_device_malloc_total_bytes();
            dev_malloc_count = ggml_cuda_device_malloc_total_count();
            cuda_graphs_total = ggml_backend_cuda_graphs_total_created(gpu_backend);
            event_creates  = ggml_cuda_event_create_total();
            event_destroys = ggml_cuda_event_destroy_total();
#endif
            // Phase 2.3b: VRAM trace bisect — show free at all four cleanup points
            //   start -> [free gpu_buf_a/b/persist] -> after_bufs
            //         -> [pool_clear]              -> after_pool
            //         -> [graphs_clear]            -> after
            // A residue between any two = the leaker.
            fprintf(stderr, "*** BUF_FREE: VRAM: %zuMB -> bufs:%zuMB -> pool:%zuMB -> graphs:%zuMB | pool_released=%zuMB | cuda_bufs diff=%lldMB | events leak=%lld | graphs=%lld\n",
                    free_before/(1024*1024),
                    free_after_bufs/(1024*1024),
                    free_after_pool/(1024*1024),
                    free_after/(1024*1024),
                    pool_released/(1024*1024),
                    (long long)((cuda_alloc_total - cuda_free_total)/(1024*1024)),
                    (long long)(event_creates - event_destroys),
                    (long long)cuda_graphs_total);
            fflush(stderr);
        }
        // DO NOT clear KV-RAM state between passes. The staging buffer and
        // kv_ram_layers persist for the next decode_layer_major call.
        // Clearing forces re-init which double-frees the GPU KV buffer
        // (first freed in pass 1's init, freed again in pass 2's init).
        // The kv_ram_already_init flag in decode_layer_major correctly
        // reuses the existing allocation when kv_ram is non-empty.
        ctx->set_prefill_streaming(saved_streaming_flag);
    });

    if (persist_load_failed) {
        return -1;
    }

    auto get_buf_for_layer = [&](int il) -> ggml_backend_buffer_t {
        if (force_single_buffer) {
            return gpu_buf_a;
        }
        return (il % 2 == 0) ? gpu_buf_a : gpu_buf_b;
    };

    auto get_pinned_for_layer = [&](int il) -> void * {
        if (!use_pipeline) { return nullptr; }
        if (force_single_buffer) { return pinned_a; }
        return (il % 2 == 0) ? pinned_a : pinned_b;
    };

    if (force_single_buffer) {
        LLAMA_LOG_WARN("%s: LLAMA_PREFILL_FORCE_SINGLE_BUFFER enabled; all layers will upload into buffer A\n", __func__);
    }

    auto upload_sync = [&](int il) {
        if (trace_swap) {
            LLAMA_LOG_INFO("%s: [SWAP_TRACE] sync upload layer %d start @ %.2fms\n", __func__, il, elapsed_ms());
        }
#ifdef GGML_USE_CUDA
        if (use_pipeline) {
            // Staged path: memcpy mmap → pinned (CPU), then async H2D on copy stream.
            // Do NOT block-wait here — the caller will call wait_upload() to
            // insert a GPU-side event sync before the layer's kernels execute.
            upload_times[il] = prefill_upload_layer_staged(
                layer_infos[il].entries, get_buf_for_layer(il),
                get_pinned_for_layer(il), cuda_pipeline, il, trace_swap);
        } else
#endif
        {
            upload_times[il] = prefill_upload_layer(layer_infos[il].entries, get_buf_for_layer(il), slab_bytes, il, trace_swap,
                                                    remote_weight_relay);
        }
        layer_loaded[il] = true;
        if (trace_swap) {
            LLAMA_LOG_INFO("%s: [SWAP_TRACE] sync upload layer %d done @ %.2fms\n", __func__, il, elapsed_ms());
        }
    };

    auto schedule_async_upload = [&](int il) {
        if (il < 0 || il >= n_layer || layer_loaded[il]) {
            return;
        }
        if (trace_swap) {
            LLAMA_LOG_INFO("%s: [SWAP_TRACE] async schedule layer %d @ %.2fms\n", __func__, il, elapsed_ms());
        }
        pending_upload.layer_idx = il;
        pending_upload.active    = true;
#ifdef GGML_USE_CUDA
        if (use_pipeline) {
            // 3-stage pipeline: CPU thread does staging + enqueues async H2D
            pending_upload.fut = std::async(
                std::launch::async,
                [&layer_infos, &get_buf_for_layer, &get_pinned_for_layer, &cuda_pipeline, il, trace_swap]() {
                return prefill_upload_layer_staged(
                    layer_infos[il].entries, get_buf_for_layer(il),
                    get_pinned_for_layer(il), cuda_pipeline, il, trace_swap);
            });
        } else
#endif
        {
            pending_upload.fut = std::async(
                std::launch::async, [&layer_infos, &get_buf_for_layer, il, slab_bytes, trace_swap, remote_weight_relay]() {
                return prefill_upload_layer(layer_infos[il].entries, get_buf_for_layer(il), slab_bytes, il, trace_swap,
                                            remote_weight_relay);
            });
        }
    };

    auto schedule_next_for_layer = [&](int il, const char * where) {
        const int next_il = il + prefetch_distance;
        if (!enable_overlap || next_il >= n_layer || layer_loaded[next_il]) {
            return;
        }
        // MoE selective mode: don't prefetch expert sub-layers ahead of time.
        // The active expert list must be set BEFORE fetching, which requires
        // the router (sub-layer 0) to complete first. Prefetching sub-layer 1
        // before the router runs would fetch ALL experts (defeating the purpose).
        if (n_sublayers > 1 && use_wf && wf_get_moe_mode(wf) == WF_MOE_FETCH_ACTIVE) {
            int next_sub = next_il % n_sublayers;
            if (next_sub > 0) {
                // Expert sub-layer — defer until active experts are known
                return;
            }
        }
        if (pending_upload.active) {
            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] async skip next=%d at %s (pending layer %d) @ %.2fms\n",
                               __func__, next_il, where ? where : "unknown", pending_upload.layer_idx, elapsed_ms());
            }
            return;
        }
        schedule_async_upload(next_il);
    };

    auto slide_host_cache_window = [&](int il) {
        if (!host_cache.enabled || !host_cache.advice_supported) {
            return;
        }
        const int anchor = il + host_cache.prefetch_distance;
        if (anchor < n_layer) {
            const int end = llama_prefill_host_cache_window_end(host_cache, anchor, n_layer);
            for (int want = anchor; want <= end; ++want) {
                if (!host_cache.advice_supported) {
                    break;
                }
                llama_prefill_host_cache_prefetch_layer(host_cache, want);
            }
        }
        const int evict_upto = std::min(n_layer, std::max(0, anchor));
        for (int old = 0; old < evict_upto; ++old) {
            if (!host_cache.advice_supported) {
                break;
            }
            llama_prefill_host_cache_evict_layer(host_cache, old);
        }
    };

    auto pre_layer_cb = [&](int il, int /*n_layer_total*/) {
        if (prefill_debug >= 2) {
            fprintf(stderr, "*** PRE_LAYER_CB: layer %d/%d entering ***\n", il, n_layer);
            if (il >= 0 && il < n_layer) {
                const auto & entries = layer_infos[il].entries;
                for (size_t i = 0; i < std::min(entries.size(), (size_t)2); i++) {
                    const auto & e = entries[i];
                    fprintf(stderr, "***   weight[%zu]: name=\"%s\" buf=%s data=%p (gpu_buf_a=%p)\n",
                            i, e.tensor->name ? e.tensor->name : "(null)",
                            e.tensor->buffer ? ggml_backend_buffer_name(e.tensor->buffer) : "NULL",
                            e.tensor->data,
                            gpu_buf_a ? ggml_backend_buffer_get_base(gpu_buf_a) : nullptr);
                }
            }
        }

        if (host_cache.enabled) {
            const bool prefetched = il >= 0 && il < (int) host_cache.prefetched.size() ? host_cache.prefetched[il] : false;
            const bool evicted = il >= 0 && il < (int) host_cache.evicted.size() ? host_cache.evicted[il] : false;
            const bool resident = prefetched && !evicted;
            host_cache.lookups += 1;
            if (resident) {
                host_cache.hits += 1;
            } else {
                host_cache.misses += 1;
            }
        }

        // ---- Weight fetcher path: NVMe → GPU via DirectStorage / direct I/O ----
        if (use_wf) {
            // Start only after persistent-tensor setup, tensor remapping, graph
            // build, and scheduler allocation are complete. Starting at WF
            // creation let chunk N+1's reused-staging DMA overlap those setup
            // allocations/uploads, widening the nvcuda interop race window.
            if (!wf_started) {
                wf_start(wf);
                wf_started = true;
                LLAMA_LOG_INFO("%s: weight fetcher pipeline started at first layer callback\n", __func__);
            }

            int model_layer = (n_sublayers > 1) ? (il / n_sublayers) : il;
            int sub_idx     = (n_sublayers > 1) ? (il % n_sublayers) : 0;

            // Sub-layer swap: release previous sub-layer's buffer and schedule this one
            if (sub_idx > 0) {
                int prev_vl = il - 1;
                wf_cuda_record_compute_done(wf, prev_vl);

                // ---- MoE selective expert streaming ----
                // After the router sub-layer (sub_idx 0) computes, the
                // post_layer_cb has already read back active expert indices
                // and called wf_set_active_experts(). The WF will now skip
                // inactive expert weights on fetch, saving ~90% I/O per MoE layer.

                wf_swap_sublayer(wf, prev_vl, il);
            }

            auto t_wait_start = std::chrono::high_resolution_clock::now();
            void * gpu_ptr = wf_wait_layer_gpu(wf, il);
            auto t_wait_end = std::chrono::high_resolution_clock::now();

            stall_times[model_layer] = std::chrono::duration<float, std::milli>(t_wait_end - t_wait_start).count();

            if (gpu_ptr) {
                // Use sub-layer entries if splitting, else full layer entries
                auto & remap_entries = (n_sublayers > 1 && !sublayer_infos.empty())
                                       ? sublayer_infos[il].entries
                                       : layer_infos[model_layer].entries;
                // WF_BUFID_FIX_2026_07_07: get_buf_for_layer(model_layer) guesses the
                // physical ping-pong buffer from model_layer%2 parity. That guess is
                // WRONG for the WF path whenever the real per-virtual-layer buffer
                // (assigned by wf_find_free_gpu_buf's first-free scan inside
                // llama-weight-fetcher.cpp) diverges from that parity -- e.g. a MoE
                // router sub-layer and its expert sub-layer share the SAME model_layer
                // (so get_buf_for_layer returns the SAME ggml buffer for both) but are
                // routinely assigned to DIFFERENT physical WF buffers (il alternates,
                // model_layer doesn't). entry.tensor->data is already correct (it comes
                // straight from gpu_ptr, the real address), but entry.tensor->buffer was
                // being stamped with the WRONG ggml_backend_buffer_t. Resolve identity
                // from the actual returned pointer instead of the parity guess so
                // tensor->buffer always names the buffer tensor->data really lives in.
                ggml_backend_buffer_t wf_actual_buf = gpu_buf_a;
                if (gpu_buf_b && gpu_ptr == ggml_backend_buffer_get_base(gpu_buf_b)) {
                    wf_actual_buf = gpu_buf_b;
                }
                if (prefill_debug >= 1) {
                    ggml_backend_buffer_t wf_guessed_buf = get_buf_for_layer(model_layer);
                    if (wf_guessed_buf != wf_actual_buf) {
                        fprintf(stderr, "*** WF_BUFID_FIX: il=%d model_layer=%d sub_idx=%d "
                                        "guessed(parity)=%p actual=%p (MISMATCH corrected)\n",
                                il, model_layer, sub_idx, (void *) wf_guessed_buf, (void *) wf_actual_buf);
                    }
                }
                wf_adapter_remap_tensors(remap_entries, gpu_ptr, wf_actual_buf);
                wf_cuda_wait_upload(wf, il);
                if (std::getenv("LLAMA_PREFILL_WF_HARD_SYNC")) {
                    ggml_backend_synchronize(gpu_backend);
                }

                // WF_BUFDIAG (LLAMA_WF_BUFDIAG=1): per-(layer,pass) buffer-content checksum to LOCALIZE the
                // WF>=2 double-buffer DATA corruption (proven a data bug: HARD_SYNC did not fix it). Weights are
                // deterministic, so a given (il) must hash IDENTICALLY across WF=1 and WF=2; the FIRST layer whose
                // hash diverges is the bug locus. If the SAME il's hash changes across ubatch passes, the buffer was
                // recycled/overwritten between a layer's ubatches. Cheap (3 small D2H reads), env-gated, no effect off.
#ifdef GGML_USE_CUDA
                if (std::getenv("LLAMA_WF_BUFDIAG") && gpu_ptr) {
                    ggml_backend_synchronize(gpu_backend);
                    const size_t lb = layer_infos[model_layer].total_bytes;
                    const int wf_bi = (gpu_buf_b && gpu_ptr == ggml_backend_buffer_get_base(gpu_buf_b)) ? 1 : 0;
                    auto hash_at = [&](size_t off) -> uint64_t {
                        unsigned char b[256] = {0};
                        const size_t n = (lb > off + sizeof(b)) ? sizeof(b) : (lb > off ? lb - off : 0);
                        if (n == 0) { return 0; }
                        cudaMemcpy(b, (const char *) gpu_ptr + off, n, cudaMemcpyDeviceToHost);
                        uint64_t h = 1469598103934665603ULL;
                        for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
                        return h;
                    };
                    const uint64_t h0 = hash_at(0);
                    const uint64_t hm = hash_at(lb / 2);
                    const uint64_t he = hash_at(lb > 256 ? lb - 256 : 0);
                    fprintf(stderr, "*** WF_BUFDIAG il=%d ml=%d bi=%d bytes=%zu h0=%016llx hm=%016llx he=%016llx\n",
                            il, model_layer, wf_bi, lb,
                            (unsigned long long) h0, (unsigned long long) hm, (unsigned long long) he);
                    fflush(stderr);
                }
#endif

                // Export tensor order file on first pass — extractor uses this
                // to write .bin files in compute-buffer order
                {
                    static const char * order_dir = std::getenv("LLAMA_PREFILL_DUMP_ORDER");
                    static bool order_dumped = false;
                    if (order_dir && order_dir[0] && !order_dumped) {
                        // Write tensor order for each layer (first pass only)
                        char path[512];
                        snprintf(path, sizeof(path), "%s/layer_%03d.order", order_dir, model_layer);
                        FILE * ofp = fopen(path, "w");
                        if (ofp) {
                            for (auto & e : remap_entries) {
                                if (e.tensor && e.tensor->name)
                                    fprintf(ofp, "%s %zu\n", e.tensor->name, e.nbytes);
                            }
                            fclose(ofp);
                        }
                        // Mark done after last layer
                        if (model_layer == (int)layer_infos.size() - 1) order_dumped = true;
                    }
                }

                // Dump layer weights to .bin file (compute-buffer order)
                // Activated by LLAMA_PREFILL_DUMP_LAYERS env var
                {
                    static const char * dump_dir = std::getenv("LLAMA_PREFILL_DUMP_LAYERS");
                    if (dump_dir && dump_dir[0] && gpu_ptr) {
                        {
                            size_t layer_bytes = layer_infos[model_layer].total_bytes;
                            char path[512];
                            snprintf(path, sizeof(path), "%s/layer_%03d.bin", dump_dir, model_layer);
                            // D2H copy from GPU buffer
                            std::vector<uint8_t> host_buf(layer_bytes);
                            ggml_backend_synchronize(gpu_backend);
#ifdef GGML_USE_CUDA
                            cudaMemcpy(host_buf.data(), gpu_ptr, layer_bytes, cudaMemcpyDeviceToHost);
#else
                            memcpy(host_buf.data(), gpu_ptr, layer_bytes);
#endif
                            FILE * fp = fopen(path, "wb");
                            if (fp) {
                                fwrite(host_buf.data(), 1, layer_bytes, fp);
                                fclose(fp);
                                if (model_layer % 10 == 0 || model_layer < 5)
                                    fprintf(stderr, "*** DUMP: layer %d → %s (%zu bytes)\n",
                                            model_layer, path, layer_bytes);
                            }
                        }
                    }
                }
            } else {
                LLAMA_LOG_ERROR("%s: wf_wait_layer_gpu returned null for layer %d (virtual)\n", __func__, il);
            }

            layer_loaded[model_layer] = true;
            upload_times[model_layer] = 0.0f;
            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] wf layer %d ready, stall=%.2fms @ %.2fms\n",
                               __func__, il, stall_times[il], elapsed_ms());
            }
            return;
        }

        if (enable_overlap && pending_upload.active && pending_upload.layer_idx == il) {
            auto t_wait_start = std::chrono::high_resolution_clock::now();
            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] pre-layer wait begin layer %d @ %.2fms\n", __func__, il, elapsed_ms());
            }
            // Wait for the async CPU thread to finish staging + CUDA enqueue
            pending_upload.fut.wait();
            auto t_wait_end = std::chrono::high_resolution_clock::now();
            stall_times[il] = std::chrono::duration<float, std::milli>(t_wait_end - t_wait_start).count();
            upload_times[il]  = pending_upload.fut.get();
            pending_upload.active = false;
            layer_loaded[il]      = true;
            host_cache.storage_wait_ms += stall_times[il];

#ifdef GGML_USE_CUDA
            // Pipeline: make the compute stream wait for the H2D copy to finish
            // on the copy stream.  This is GPU-side sync (non-blocking on host).
            if (use_pipeline) {
                ggml_backend_cuda_prefill_wait_upload(cuda_pipeline);
            }
#endif

            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] pre-layer wait end layer %d stall=%.2fms @ %.2fms\n",
                               __func__, il, stall_times[il], elapsed_ms());
            }
            // Launch upload of the following layer while this layer computes.
            schedule_next_for_layer(il, "pre_layer_cb(wait-path)");
            slide_host_cache_window(il);
            return;
        }

        upload_sync(il);
        stall_times[il] = 0.0f;
#ifdef GGML_USE_CUDA
        // For sync path with pipeline: the H2D is already enqueued on
        // copy stream.  Insert event wait so compute stream waits for it.
        if (use_pipeline) {
            ggml_backend_cuda_prefill_wait_upload(cuda_pipeline);
        }
#endif
        if (prefill_debug >= 2) {
            fprintf(stderr, "*** PRE_LAYER_CB: layer %d upload done ***\n", il);
        }
        // Launch upload of the following layer while this layer computes.
        schedule_next_for_layer(il, "pre_layer_cb(sync-path)");
        slide_host_cache_window(il);
    };

    auto capture_dsv4_route_layer = [&](int model_layer_idx) {
        if (!dsv4_route_trace_enabled || model_layer_idx < 0 || model_layer_idx >= n_layer) {
            return;
        }

        auto * res = ctx->get_graph_result();
        ggml_tensor * topk_tensor = nullptr;
        if (res) {
            const std::string topk_name = "ffn_moe_topk-" + std::to_string(model_layer_idx);
            topk_tensor = ggml_graph_get_tensor(res->get_gf(), topk_name.c_str());
        }

        if (!topk_tensor || topk_tensor->type != GGML_TYPE_I32 ||
                topk_tensor->ne[0] <= 0 || topk_tensor->ne[1] <= 0) {
            return;
        }

        // The mid-layer callback runs at ffn_moe_topk completion, before
        // downstream expert nodes can reuse scheduler scratch. Pay the
        // synchronization/readback cost only while route tracing is enabled.
        if (gpu_backend) {
            ggml_backend_synchronize(gpu_backend);
        }

        const size_t n_used = (size_t) topk_tensor->ne[0];
        const size_t token_index = params.route_trace_prompt_last
                ? (size_t) topk_tensor->ne[1] - 1 : 0;
        const size_t token_offset = token_index * topk_tensor->nb[1];
        const size_t span = (n_used - 1) * topk_tensor->nb[0] + sizeof(int32_t);
        std::vector<uint8_t> raw(span);
        ggml_backend_tensor_get(topk_tensor, raw.data(), token_offset, span);

        auto & layer_trace = dsv4_route_trace_layers[(size_t) model_layer_idx];
        layer_trace.experts.resize(n_used);
        for (size_t ie = 0; ie < n_used; ++ie) {
            memcpy(&layer_trace.experts[ie], raw.data() + ie * topk_tensor->nb[0], sizeof(int32_t));
        }
        layer_trace.captured = true;
    };

    // LLAMA_DSV4_COMP_DUMP=<prefix>: raw-byte dumps of the LID compression-chain tensors
    // (lid_state_kv / lid_state_score_ape / lid_state_compress_rot) as they complete inside
    // the layer-major executor. Bug#2 round-5 oracle: the Python side recomputes the pooling
    // chain from these and bisects the first diverging op. Capped per-name to bound size.
    static const char * dsv4_comp_dump_prefix = std::getenv("LLAMA_DSV4_COMP_DUMP");
    auto capture_dsv4_comp_dump = [&](ggml_tensor * t) {
        static int comp_dump_entry_logged = 0;
        if (comp_dump_entry_logged < 6) {
            LLAMA_LOG_INFO("COMP_DUMP: capture entry '%s' buffer=%p\n",
                    t ? t->name : "(null)", t ? (void *) t->buffer : nullptr);
            comp_dump_entry_logged++;
        }
        if (!dsv4_comp_dump_prefix || !dsv4_comp_dump_prefix[0] || !t || !t->buffer) {
            return;
        }
        static std::mutex dump_mutex;
        static std::map<std::string, int> dump_counts;
        std::lock_guard<std::mutex> lock(dump_mutex);
        const std::string name = t->name;
        int & count = dump_counts[name];
        // 2026-08-28: the cap was a hardcoded 3 described as "ubatch 0 + margin", and it really is
        // ubatch 0 ONLY -- f0 and f1 came back BYTE-IDENTICAL, so every tensor measurement in the
        // needle investigation describes the first ubatch. The failure is now known to depend on
        // which ubatch a fact was prefilled in (only the FINAL ubatch's content is retrievable), so
        // the later firings are the ones that matter. Configurable; default unchanged at 3.
        //
        // Identify the ubatch from the manifest afterwards: `csa_k_all` has ne2 = raw + compressed,
        // which grows 512 -> 768 as the raw cache grows, so the firing carrying the larger geometry
        // is a later ubatch.
        static const int dump_max_firings = []() {
            const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP_FIRINGS");
            const int v = (e && e[0]) ? std::atoi(e) : 3;
            return v > 0 ? v : 3;
        }();
        if (count >= dump_max_firings) {
            return;
        }
        if (gpu_backend) {
            ggml_backend_synchronize(gpu_backend);
        }
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        char fname[1024];
        snprintf(fname, sizeof(fname), "%s.%s.f%d.bin", dsv4_comp_dump_prefix, name.c_str(), count);
        FILE * fb = fopen(fname, "wb");
        if (fb) {
            fwrite(raw.data(), 1, nbytes, fb);
            fclose(fb);
        }
        char mname[1024];
        snprintf(mname, sizeof(mname), "%s.manifest.tsv", dsv4_comp_dump_prefix);
        FILE * fm = fopen(mname, "ab+");
        if (fm) {
            fprintf(fm, "%s\t%d\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%zu\t%zu\t%zu\t%zu\n",
                    name.c_str(), count, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], nbytes);
            fclose(fm);
        }
        count++;
    };

    auto capture_dsv4_lid_topk_layer = [&](int model_layer_idx, ggml_tensor * topk_tensor) {
        // COMP_DUMP nodes arrive through the same hook; branch by name before the
        // top-k-specific gating below.
        if (topk_tensor && topk_tensor->name[0] != '\0' &&
                strncmp(topk_tensor->name, "lid_state_", 10) == 0) {
            capture_dsv4_comp_dump(topk_tensor);
            return;
        }
        if (!dsv4_lid_topk_trace_enabled || model_layer_idx < 0 || model_layer_idx >= n_layer ||
                !topk_tensor || topk_tensor->type != GGML_TYPE_I32 || topk_tensor->ne[0] <= 0) {
            return;
        }

        // This callback runs immediately after lid_top_k-<il>. Synchronization and
        // strided readback are intentionally paid only while tracing is enabled.
        if (gpu_backend) {
            ggml_backend_synchronize(gpu_backend);
        }

        ggml_tensor * topk_op = topk_tensor;
        while (topk_op && topk_op->op != GGML_OP_TOP_K) {
            topk_op = topk_op->src[0];
        }
        const int64_t n_lid = topk_op && topk_op->src[0] ? topk_op->src[0]->ne[0] : 0;

        int32_t n_visible = -1;
        auto * res = ctx->get_graph_result();
        if (res) {
            if (ggml_tensor * visible = ggml_graph_get_tensor(res->get_gf(), "dsv4_lid_n_visible")) {
                if (visible->type == GGML_TYPE_I32 && ggml_nelements(visible) > 0) {
                    ggml_backend_tensor_get(visible, &n_visible, 0, sizeof(n_visible));
                }
            }
        }

        const size_t k = (size_t) topk_tensor->ne[0];
        const size_t span = (k - 1) * topk_tensor->nb[0] + sizeof(int32_t);
        std::vector<uint8_t> raw(span);
        ggml_backend_tensor_get(topk_tensor, raw.data(), 0, span);

        auto & layer_trace = dsv4_lid_topk_trace_layers[(size_t) model_layer_idx];
        layer_trace.n_lid = n_lid;
        layer_trace.n_visible = n_visible;
        layer_trace.rows.resize(k);
        for (size_t i = 0; i < k; ++i) {
            memcpy(&layer_trace.rows[i], raw.data() + i * topk_tensor->nb[0], sizeof(int32_t));
        }

        // LID_PROBE: also read the masked score vector this top-k consumed, from the
        // same completed layer graph (no cb_eval, no graph splitting). First token's
        // column only; strided F32 readback.
        layer_trace.scores.clear();
        if (res && n_lid > 0) {
            const std::string score_name = "lid_score_masked-" + std::to_string(model_layer_idx);
            if (ggml_tensor * score_t = ggml_graph_get_tensor(res->get_gf(), score_name.c_str())) {
                if (score_t->type == GGML_TYPE_F32 && score_t->ne[0] >= n_lid) {
                    const size_t sspan = ((size_t) n_lid - 1) * score_t->nb[0] + sizeof(float);
                    std::vector<uint8_t> sraw(sspan);
                    ggml_backend_tensor_get(score_t, sraw.data(), 0, sspan);
                    layer_trace.scores.resize((size_t) n_lid);
                    for (int64_t i = 0; i < n_lid; ++i) {
                        memcpy(&layer_trace.scores[(size_t) i], sraw.data() + (size_t) i * score_t->nb[0], sizeof(float));
                    }
                }
            }
        }
        layer_trace.captured = true;
    };

    auto post_layer_cb = [&](int il, int /*n_layer_total*/) {

        // COMP_DUMP: prefill ubatches run through the DLM's own layer loop (they never
        // reach graph_compute_per_layer's node scan), so hook the dump here — the same
        // post-layer readback pattern the MoE route-trace oracle uses. Layer 2 only.
        if (dsv4_comp_dump_prefix && dsv4_comp_dump_prefix[0]) {
            const int model_layer_idx = (n_sublayers > 1) ? il / n_sublayers : il;
            const int sub_idx         = (n_sublayers > 1) ? il % n_sublayers : 0;
            // Layer 2 by default, but EVERY dump taken so far was layer 2 -- a divergence that only
            // begins deeper in the stack was structurally invisible. LLAMA_DSV4_COMP_DUMP_LAYER
            // moves it; the pins in deepseek-v4.cpp read the same variable so both sides agree.
            static const int dump_layer = []() {
                const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP_LAYER");
                return (e && e[0]) ? std::atoi(e) : 2;
            }();
            if (model_layer_idx == dump_layer && sub_idx == n_sublayers - 1) {
                if (auto * res = ctx->get_graph_result()) {
                    static const char * dump_bases[] = {
                        "lid_state_kv", "lid_state_score_ape", "lid_state_compress_rot", "lid_q_rot",
                        "csa_state_kv", "csa_state_score_ape", "csa_state_compress_final",
                        "csa_state_compress_pooled", "lid_state_compress_pooled",
                        "csa_comp_k", "csa_k_all", "csa_lid_kq_mask", "csa_comp_input",
                        "csa_pre_q", "csa_pre_out",
                        "csa_state_pos", "csa_state_score_pre_ape", "csa_ape_rows", "csa_raw_mask", "csa_raw_mask_leaf", "csa_raw_mask_cnv",
                        // HCA (odd layers 3..41, ratio 128). Never dumped before 2026-08-28 -- the
                        // whole investigation sampled CSA layers only, so half the compressed
                        // attention had no coverage. Its soft_max reduces over 128 rows where CSA's
                        // reduces over 8, so it is a genuinely different CUDA kernel shape.
                        "hca_values", "hca_scores",
                        "hca_state_compress_weights", "hca_state_compress_pooled",
                    };
                    // The streamed WEIGHTS themselves, not just the activations. E1 exonerated
                    // decode-side consumption, so the suspect is the content both consumers read --
                    // i.e. what prefill computed. These compressor tensors are streamed per layer
                    // (collect_tensors_from_layer PUSH_IF attn_comp_* / indexer_comp_*), so if
                    // streaming ever delivers stale or wrong bytes, the compressed rows are built
                    // from them and every downstream check still passes: the pooling would
                    // faithfully pool wrong inputs, which is exactly why "pooling is exact vs numpy"
                    // did not clear this. Dumping the on-device bytes lets them be diffed against
                    // the GGUF file offline -- the Mac recalls from the SAME file, so the file is
                    // good and only delivery is in question.
                    // GGUF names, not the llama_layer FIELD names -- they differ
                    // (`attn_comp_wkv` is stored as `blk.N.attn_compressor_kv`, per
                    // llama-arch.cpp's LLM_TENSOR_* table). Using the field names silently dumped
                    // nothing for the compressor set while the two `indexer.*` ones, whose field
                    // and GGUF names happen to match, appeared and made it look like it worked.
                    static const char * weight_bases[] = {
                        "attn_compressor_kv", "attn_compressor_gate", "attn_compressor_ape",
                        "attn_compressor_norm", "indexer_compressor_kv", "indexer_compressor_gate",
                        "indexer_compressor_ape", "indexer_compressor_norm",
                        "indexer.proj", "indexer.attn_q_b", "indexer.attn_k", "indexer.k_norm",
                    };
                    char wname_buf[GGML_MAX_NAME];
                    for (const char * wb : weight_bases) {
                        snprintf(wname_buf, sizeof(wname_buf), "blk.%d.%s.weight", dump_layer, wb);
                        if (ggml_tensor * wt = ggml_graph_get_tensor(res->get_gf(), wname_buf)) {
                            capture_dsv4_comp_dump(wt);
                        }
                    }
                    char dump_name_buf[GGML_MAX_NAME];
                    for (const char * base_nm : dump_bases) {
                        snprintf(dump_name_buf, sizeof(dump_name_buf), "%s-%d", base_nm, dump_layer);
                        const char * nm = dump_name_buf;
                        if (ggml_tensor * t = ggml_graph_get_tensor(res->get_gf(), nm)) {
                            capture_dsv4_comp_dump(t);
                        } else {
                            static int miss_logged = 0;
                            if (miss_logged < 6) {
                                LLAMA_LOG_INFO("COMP_DUMP: tensor '%s' not in layer-%d graph\n", nm, dump_layer);
                                miss_logged++;
                            }
                        }
                    }
                }
            }
        }

        if (use_wf) {
            wf_cuda_record_compute_done(wf, il);

            // ---- MoE selective expert streaming ----
            // After sub-layer 0 (router) computes, read back the top-K expert
            // selections and tell the WF which experts to fetch for the expert
            // sub-layers.  This saves ~90% of MoE I/O per layer.
            if (n_sublayers > 1 && wf_get_moe_mode(wf) == WF_MOE_FETCH_ACTIVE) {
                int sub_idx = il % n_sublayers;
                int model_layer_idx = il / n_sublayers;

                if (sub_idx == 0) {
                    // Router sub-layer just finished. Find the topk tensor in the graph.
                    // The graph result holds the last-built graph with all nodes.
                    auto * res = ctx->get_graph_result();
                    ggml_tensor * topk_tensor = nullptr;

                    if (res) {
                        // Search the graph for ffn_moe_topk-{layer}
                        std::string topk_name = "ffn_moe_topk-" + std::to_string(model_layer_idx);
                        topk_tensor = ggml_graph_get_tensor(res->get_gf(), topk_name.c_str());
                    }

                    if (topk_tensor && topk_tensor->ne[0] > 0) {
                        // GPU sync to ensure router output is ready
                        if (gpu_backend) {
                            ggml_backend_synchronize(gpu_backend);
                        }

                        {
                            int ne0 = (int)topk_tensor->ne[0], ne1 = (int)topk_tensor->ne[1];
                            int total = ne0 * ne1;
                            std::vector<int32_t> dbg(total);
                            ggml_backend_tensor_get(topk_tensor, dbg.data(), 0, total * sizeof(int32_t));
                            fprintf(stderr, "[moe-debug] layer %d: ne=[%d,%d] ids=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                                    model_layer_idx, ne0, ne1,
                                    total > 0 ? dbg[0] : -1, total > 1 ? dbg[1] : -1,
                                    total > 2 ? dbg[2] : -1, total > 3 ? dbg[3] : -1,
                                    total > 4 ? dbg[4] : -1, total > 5 ? dbg[5] : -1,
                                    total > 6 ? dbg[6] : -1, total > 7 ? dbg[7] : -1);
                        }

                        const int n_expert_used = (int)topk_tensor->ne[0];
                        const int n_tok         = (int)topk_tensor->ne[1];
                        std::vector<int32_t> expert_ids(n_expert_used * n_tok);

                        // D2H copy (tiny: 8 experts × N tokens × 4 bytes)
                        ggml_backend_tensor_get(topk_tensor, expert_ids.data(), 0,
                                                n_expert_used * n_tok * sizeof(int32_t));

                        // Collect unique expert indices
                        const int max_experts = (int)model->hparams.n_expert;
                        std::vector<bool> seen(max_experts, false);
                        std::vector<int> active;
                        for (int i = 0; i < n_expert_used * n_tok; i++) {
                            int eid = expert_ids[i];
                            if (eid >= 0 && eid < max_experts && !seen[eid]) {
                                seen[eid] = true;
                                active.push_back(eid);
                            }
                        }

                        // Set active experts for all expert sub-layers of this model layer
                        for (int s = 1; s < n_sublayers; s++) {
                            int vl = model_layer_idx * n_sublayers + s;
                            wf_set_active_experts(wf, vl, active.data(), (int)active.size());
                        }

                        if (prefill_debug >= 1) {
                            fprintf(stderr, "[moe-select] layer %d: %d/%d experts active across %d tokens (%.0f%% I/O saved)\n",
                                    model_layer_idx, (int)active.size(), max_experts, n_tok,
                                    100.0 * (1.0 - (double)active.size() / (double)max_experts));
                        }
                    }
                }
            }

            wf_release_layer(wf, il);
            if (wf_get_n_gpu_bufs(wf) <= 1) {
                wf_prefetch_next_layer(wf, il);
            }
        }

        // Map virtual layer to model layer for reporting
        int model_il = (n_sublayers > 1) ? (il / n_sublayers) : il;

        // On the last layer, report GPU VRAM usage breakdown
        if (model_il == n_layer - 1) {
            size_t vram_free = 0, vram_total = 0;
            if (ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend)) {
                ggml_backend_dev_memory(dev, &vram_free, &vram_total);
                size_t vram_used = vram_total - vram_free;

                // Get actual WF buffer info from the weight fetcher
                int n_wf_bufs = use_wf ? wf_get_n_gpu_bufs(wf) : 0;
                size_t wf_buf_cap = use_wf ? wf_get_gpu_buf_capacity(wf) : 0;
                size_t wf_total = (size_t)n_wf_bufs * wf_buf_cap;
                size_t remainder = vram_used > wf_total ? vram_used - wf_total : 0;

                // Find max layer size for reference
                size_t max_lb = 0;
                for (int i = 0; i < n_layer; i++) {
                    if (layer_infos[i].total_bytes > max_lb) max_lb = layer_infos[i].total_bytes;
                }

                fprintf(stderr, "\n");
                fprintf(stderr, "*** VRAM REPORT (during last layer compute) ***\n");
                fprintf(stderr, "*** VRAM Total:    %8.2f MB\n", vram_total / (1024.0 * 1024.0));
                fprintf(stderr, "*** VRAM Used:     %8.2f MB\n", vram_used  / (1024.0 * 1024.0));
                fprintf(stderr, "*** VRAM Free:     %8.2f MB\n", vram_free  / (1024.0 * 1024.0));
                if (use_wf) {
                    fprintf(stderr, "*** VRAM WF bufs:  %8.2f MB  (%d × %.2f MB, max_layer=%.2f MB)\n",
                            wf_total / (1024.0 * 1024.0),
                            n_wf_bufs,
                            wf_buf_cap / (1024.0 * 1024.0),
                            max_lb / (1024.0 * 1024.0));
                }
                fprintf(stderr, "*** VRAM Other:    %8.2f MB  (KV cache + compute scratch + non-layer tensors)\n",
                        remainder / (1024.0 * 1024.0));
                fprintf(stderr, "***\n");
            }
        }

        if (prefill_debug >= 2) {
            fprintf(stderr, "*** LAYER %d/%d done ***\n", il, n_layer);
        }
        GGML_UNUSED(il);
    };

    // Mid-layer callback: fires after ffn_moe_topk computes (router done).
    // Reads expert IDs and sets active expert filter for selective streaming.
    auto mid_layer_cb = [&](int il, int /*n_layer_total*/) {
        capture_dsv4_route_layer(il);

        if (!use_wf || wf_get_moe_mode(wf) != WF_MOE_FETCH_ACTIVE) return;

        auto * res = ctx->get_graph_result();
        if (!res) return;

        std::string topk_name = "ffn_moe_topk-" + std::to_string(il);
        ggml_tensor * topk = ggml_graph_get_tensor(res->get_gf(), topk_name.c_str());
        if (!topk || topk->ne[0] <= 0) return;

        const int ne0 = (int)topk->ne[0]; // n_expert_used
        const int ne1 = (int)topk->ne[1]; // n_tokens
        std::vector<int32_t> ids(ne0 * ne1);
        ggml_backend_tensor_get(topk, ids.data(), 0, ne0 * ne1 * sizeof(int32_t));

        const int max_exp = (int)model->hparams.n_expert;
        std::vector<bool> seen(max_exp, false);
        std::vector<int> active;
        for (int i = 0; i < ne0 * ne1; i++) {
            int eid = ids[i];
            if (eid >= 0 && eid < max_exp && !seen[eid]) {
                seen[eid] = true;
                active.push_back(eid);
            }
        }

        // Set active experts for this layer in the WF
        wf_set_active_experts(wf, il, active.data(), (int)active.size());
    };

    // Register callbacks and sub-layer count in the context
    ctx->set_per_layer_callbacks(pre_layer_cb, post_layer_cb, mid_layer_cb);
    if (dsv4_lid_topk_trace_enabled ||
            (dsv4_comp_dump_prefix && dsv4_comp_dump_prefix[0] != '\0' &&
             model->arch == LLM_ARCH_DEEPSEEK4)) {
        ctx->set_lid_topk_callback(capture_dsv4_lid_topk_layer);
    }
    ctx->set_n_sublayers(n_sublayers);
    // C1: expose the active WF (non-owning) so decode_layer_major can reset+re-drive it per ubatch
    // under LLAMA_PREFILL_FASTPATH_MULTIUB. Cleared is unnecessary (re-set each prefill; non-owning).
    ctx->set_prefill_active_wf(wf);
    callbacks_registered = true;
    if (prefill_debug >= 1) {
        fprintf(stderr, "*** PREFILL: callbacks registered, calling llama_decode ***\n");
    }

    auto t_compute_start = std::chrono::high_resolution_clock::now();

    // Temporarily disable recursive streaming dispatch while we run the
    // orchestrated decode path with per-layer callbacks.
    ctx->set_prefill_streaming(false);
    ctx->set_inside_streaming_decode(true);

    // CRITICAL: suppress sched_reserve() and decode handoff during our inner
    // llama_decode() call.
    //
    // Between consecutive streaming prefill batches, arm_decode_handoff_after_prefill()
    // sets decode_handoff_pending = true.  If the inner llama_decode()'s
    // maybe_apply_decode_handoff() fires, it calls migrate_kv_cache_to_cpu() which:
    //   (a) replaces memory with a CPU-backed module
    //   (b) sets cparams.offload_kqv = false
    //   (c) sets sched_need_reserve = true
    //
    // (a)+(b) cause the NEXT streaming batch's graph to have CPU-backed recurrent
    // state tensors.  GPU force assigns all nodes to GPU, so CUDA kernels try to
    // write to CPU memory → "illegal memory access" crash.
    //
    // (c) triggers sched_reserve() which destroys and recreates the scheduler,
    // wiping compute buffer allocations while streaming GPU buffers are live →
    // cuBLAS "internal operation failed" crash.
    //
    // The guard saves both flags, clears them, and restores after the inner
    // decode returns.  The actual handoff is deferred to the first non-streaming
    // decode batch (which enters decode() → maybe_apply_decode_handoff()).
    auto guard = ctx->suppress_sched_and_handoff();

    llama_batch batch_copy = batch;  // llama_decode takes by value

    // When WF is active, skip the CPU-blocking ggml_backend_synchronize() in
    // the eval callback loop.  GPU sync is handled by CUDA events instead:
    //   - compute_done event recorded in post_layer_cb
    //   - wf_find_free_gpu_buf checks compute_done before buffer reuse
    //   - wf_cuda_wait_upload ensures DMA is complete before compute starts
    if (use_wf) {
        auto * sched = ctx->get_sched();
        if (sched) {
            ggml_backend_sched_set_eval_skip_sync(sched, true);
        }
    }

    int ret;
    bool n_p_eval_already_fixed = false;
    const int chunk_pos0 = batch_copy.pos ? batch_copy.pos[0] : 0;
    const int total_after_chunk = chunk_pos0 + batch_copy.n_tokens;
    int prefix_tokens = batch_copy.n_tokens;
    int tail_tokens = 0;

    // Phase 142 round-transition probe (env-gated).
    // Print at every llama_prefill_stream entry so we can see what state
    // round 2+ inherits from round 1's tail-deferred handoff. Cheap to leave
    // in (one fprintf per call) and toggleable.
    if (std::getenv("LLAMA_PREFILL_ROUND_PROBE") != nullptr) {
        const uint32_t hf_streamed = ctx->get_prefill_handoff_streamed_prefix_count();
        const int first_token_id = (batch_copy.token && batch_copy.n_tokens > 0)
                                   ? (int) batch_copy.token[0]
                                   : -1;
        fprintf(stderr,
                "[ROUND_PROBE] enter streaming_prefill: chunk_pos0=%d n_tokens=%d total_after=%d "
                "prev_streamed_prefix=%u first_tok=%d fa_enabled=%d\n",
                chunk_pos0, batch_copy.n_tokens, total_after_chunk,
                hf_streamed, first_token_id,
                ctx->is_flash_attn_enabled() ? 1 : 0);
        fflush(stderr);
    }
    ctx->set_prefill_kv_layout_flash_attn(ctx->is_flash_attn_enabled());
    ctx->clear_prefill_tail_tokens();
    if (batch_copy.token && batch_copy.n_tokens > 0) {
        if (chunk_pos0 == 0) {
            ctx->clear_prefill_handoff_prompt();
        }
        ctx->set_prefill_handoff_prompt_token_range(batch_copy.token, (size_t) batch_copy.n_tokens, (uint32_t) chunk_pos0);
    }
    const int ubatch_size = std::max(1, (int) ctx->n_ubatch());
    const bool flash_attn_saved = ctx->is_flash_attn_enabled();
    const bool flash_attn_layout_saved = ctx->get_prefill_kv_layout_flash_attn();
    // Always 256-align: MLA models require it, non-MLA models accept it
    // (256 is a multiple of 64, the previous non-MLA value). Using a single
    // constant eliminates the [880, 1500]-tail-band silent corruption that
    // surfaced when 64-aligned tails were handed to Mac for re-prefill on
    // MLA models. Memory: feedback_no_weight_quant_blame.md.
    const int fa_tail_align = 256;
    // 2026-06-01 vision diag: an embd (image) batch gets 256-aligned-split into an
    // FA-prefix + a separate FA-tail decode pass (graph rebuild). Suspected to corrupt
    // the tail image cells' K. LLAMA_PREFILL_EMBD_NO_SPLIT=1 runs the whole embd batch
    // as ONE FA-on pass (no split) so all image cells get consistent FA-on K layout.
    const bool embd_no_split = (batch_copy.embd != nullptr) &&
            (std::getenv("LLAMA_PREFILL_EMBD_NO_SPLIT") != nullptr);
    // Keep the prompt-tail and one-token decode graphs on the same DSV4 FA path.
    const bool keep_dsv4_stream_local_fa =
            model->arch == LLM_ARCH_DEEPSEEK4 &&
            ctx->get_prefill_decode_mode() == LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
    // NULL TEST (LLAMA_PREFILL_NO_FA_TAIL_SPLIT, default off): run the whole batch as ONE pass.
    //
    // The FA-tail split runs prefix and tail as two separate decode_layer_major calls -- see the
    // comment immediately below, which records that "only the LAST decode's logits remain
    // accessible" and skips the split for multi-logits batches for exactly that reason. The DSV4
    // needle failure has the SAME shape for a different consumer: only content prefilled in the
    // final pass is retrievable (9 prompts; distance-from-end and last-N-tokens both eliminated).
    // `fa_tail_align = 256` is also precisely the 256 granularity the failures track, and it is
    // independent of n_ubatch -- which is why lowering n_ubatch did not move the boundary.
    //
    // The existing escape hatch is scoped to image batches (`embd != nullptr`), so it cannot test a
    // token batch. This one is general and default-off, so it changes nothing unless set.
    //   recall RETURNS with it on -> the FA prefix/tail split is the cause
    //   still MISSES              -> the loss is at the ubatch level inside a single decode
    const bool force_no_fa_tail_split =
            std::getenv("LLAMA_PREFILL_NO_FA_TAIL_SPLIT") != nullptr;
    bool has_fa_tail = use_layer_major &&
            ctx->is_flash_attn_enabled() &&
            batch_copy.n_tokens > 0 &&
            !embd_no_split &&
            !force_no_fa_tail_split &&
            (((total_after_chunk % fa_tail_align) != 0) || ((batch_copy.n_tokens % ubatch_size) != 0));

    // Phase 312 (2026-05-04): when the batch has multiple logits=1 positions
    // (WIR refine pass, etc.), the FA-tail split runs prefix and tail as two
    // separate decode_layer_major calls. Each call wipes output_ids via
    // output_reserve, so only the LAST decode's logits remain accessible.
    // For multi-logits batches the split corrupts coherence; keep the batch
    // whole and let the caller's alignment determine FA on/off.
    int n_logits_marked = 0;
    if (has_fa_tail && batch_copy.logits && batch_copy.n_tokens > 1) {
        for (int i = 0; i < batch_copy.n_tokens && n_logits_marked < 2; i++) {
            if (batch_copy.logits[i]) n_logits_marked++;
        }
        if (n_logits_marked > 1) {
            LLAMA_LOG_INFO("%s: multi-logits batch (%d tokens, >=2 marked) — skipping FA-tail split to preserve per-position logits\n",
                           __func__, batch_copy.n_tokens);
            has_fa_tail = false;
            const bool no_fa_disable = std::getenv("LLAMA_PREFILL_NO_FA_DISABLE") != nullptr;
            if (!no_fa_disable && (total_after_chunk % fa_tail_align != 0)) {
                ctx->set_flash_attn_enabled(false);
                ctx->set_prefill_kv_layout_flash_attn(false);
                ctx->invalidate_graph();
            }
        }
    }

    bool use_safe_full_no_fa = false;
    if (has_fa_tail) {
        // [TAG_GLM_DSA_UBATCH_ALIGN_2026_06_24] env-gated (LLAMA_PREFILL_UBATCH_ALIGN_PREFIX): align the streamed FA
        // prefix to the UBATCH, not just fa_tail_align(256). A PARTIAL last ubatch triggers the SM120 NCOLS=64 VKQ
        // flash-attn kernel corruption on GLM-5.2 (DKQ=576 MLA): streamed_prefix % ubatch != 0 -> garbled/NaN;
        // == 0 (full last ubatch) -> coherent. ubatch_size is already 256-aligned so this stays FA-legal; the larger
        // sub-ubatch suffix is handed to the Mac tail (which re-prefills it correctly).
        int fa_prefix_align = fa_tail_align;
        {
            static const bool ub_align_prefix = std::getenv("LLAMA_PREFILL_UBATCH_ALIGN_PREFIX") != nullptr;
            if (ub_align_prefix && (int)ubatch_size >= fa_tail_align && ((int)ubatch_size % fa_tail_align) == 0) {
                fa_prefix_align = (int)ubatch_size;
            }
        }
        const int aligned_total = (total_after_chunk / fa_prefix_align) * fa_prefix_align;
        const int aligned_prefix = std::max(0, std::min(batch_copy.n_tokens, aligned_total - chunk_pos0));
        // The FA kernels need an aligned token count, not necessarily an exact
        // multiple of n_ubatch. Keeping the aligned final microbatch on RTX is
        // the point of dynamic ubatch: e.g. 7*4096 + 1024 should prefill on RTX,
        // leaving only the sub-alignment suffix for the receiver.
        prefix_tokens = std::max(0, std::min(batch_copy.n_tokens, aligned_prefix));
        tail_tokens = batch_copy.n_tokens - prefix_tokens;
        if (prefix_tokens != aligned_prefix) {
            LLAMA_LOG_INFO("%s: FA split: clamped prefix from %d to %d for FA alignment (align=%d ub=%d)\n",
                           __func__, aligned_prefix, prefix_tokens, fa_tail_align, ubatch_size);
        }
        // If no FA prefix possible (batch smaller than ubatch), disable FA entirely
        // and process normally without the split. The whole batch will run non-FA.
        // MUST restore FA state after processing (see restore block below).
        if (prefix_tokens == 0) {
            // Phase 81: when LLAMA_PREFILL_NO_FA_DISABLE=1, force the entire batch
            // through the FA-on path EVEN IF the size isn't 256-aligned. The MLA FA
            // kernel may degrade on the unaligned tail but the K cells remain
            // FA-on-compatible so the Mac decoder doesn't see garbage. (FA-off
            // produces numerically-different K values that the Mac FA decoder can't
            // use coherently for MLA.)
            const bool no_fa_disable =
                    std::getenv("LLAMA_PREFILL_NO_FA_DISABLE") != nullptr ||
                    keep_dsv4_stream_local_fa;
            if (no_fa_disable) {
                LLAMA_LOG_INFO("%s: FA split: prefix=0, KEEPING FA on unaligned batch (%d tokens, align=%d)\n",
                               __func__, batch_copy.n_tokens, fa_tail_align);
                prefix_tokens = batch_copy.n_tokens;
                tail_tokens = 0;
            } else {
                LLAMA_LOG_INFO("%s: FA split: prefix=0, disabling FA for entire batch (%d tokens, align=%d)\n",
                               __func__, batch_copy.n_tokens, fa_tail_align);
                ctx->set_flash_attn_enabled(false);
                ctx->set_prefill_kv_layout_flash_attn(false);
                ctx->invalidate_graph();
                prefix_tokens = batch_copy.n_tokens;
                tail_tokens = 0;
            }
        }
        // The hybrid FA-prefix + local-tail path works well when the tail is
        // genuinely short on long prompts. For short 1-3 ubatch prompts, the
        // hybrid handoff has been less stable than exporting the whole prompt
        // after a conservative no-FA pass. Keep the long-prompt FA path for
        // larger aligned prefixes so 16k+ prompts still hit the fast path.
        // For multi-pass prompts (chunk_pos0 > 0), always use the FA prefix +
        // tail handoff path. The safe-no-FA fallback disables flash attention
        // which explodes the graph compute buffer (n_kv grows across passes) and
        // OOMs on the second pass. Tail tokens are handed to the Mac receiver.
        if (chunk_pos0 > 0) {
            use_safe_full_no_fa = false;
        } else {
            // Use the safe-no-FA fallback only when it is genuinely needed:
            // either FA has no prefix to work on (prefix=0 was already handled
            // above by disabling FA on the whole batch), or the Mac-handed
            // tail would exceed one ubatch. The FA-prefix + Mac-tail handoff
            // path works for any prefix >= 1 ubatch with tail < 1 ubatch, and
            // is the coherent path (safe-no-FA produces `@@@@` corruption on
            // short-prompt streaming runs with GLM-5.1 MLA). Previously the
            // fallback was gated by `prefix_tokens <= 3 * ubatch_size` which
            // pre-empted the good path for any streaming prompt under ~4.6k
            // tokens.
            use_safe_full_no_fa = tail_tokens > ubatch_size;
        }
    }
    const bool decode_handoff_configured =
            ctx->is_prefill_only() ||
            ctx->has_decode_handoff_pending() ||
            ctx->has_transport_plan() ||
            ctx->get_prefill_decode_mode() == LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT ||
            ctx->get_prefill_transport_mode() != LLAMA_PREFILL_TRANSPORT_MODE_DISABLED ||
            ctx->get_prefill_kv_port() > 0 ||
            ctx->get_token_stream_port() > 0;
    const bool is_final_prompt_chunk =
            ctx->is_prefill_only() ||
            batch_copy.logits == nullptr ||
            (batch_copy.n_tokens > 0 && batch_copy.logits[batch_copy.n_tokens - 1]);

    // When prefix=0 (batch < ubatch), the FA split can't produce an FA prefix.
    // Fall through to normal processing — the server's prompt-direct path or
    // non-FA streaming will handle it. Don't use the tail handoff with no prefix.
    // Env gate LLAMA_PREFILL_NO_REMOTE_TAIL_HANDOFF=1 forces the local safe-tail
    // fallback (RTX runs the unaligned tail with FA off, exports full KV) so
    // Mac never re-prefills cross-stack — eliminates divergence at the cost of
    // one slower tail ubatch on RTX. Required for coherent output on prompts
    // where total_token_count % 256 != 0.
    const bool no_remote_tail_env = std::getenv("LLAMA_PREFILL_NO_REMOTE_TAIL_HANDOFF") != nullptr;
    // 2026-05-31 vision: an embd batch (image embeddings; batch.embd != null,
    // batch.token == null) CANNOT have its unaligned tail deferred — neither to the
    // Mac (the receiver has no pixel data and the prompt export breaks at the image
    // LLAMA_TOKEN_NULL) nor to "the next RTX batch" (the next batch is the text suffix
    // at a later position; the deferred image cells [prefix..end] would be orphaned and
    // left as all-zero K rows, which the KV-RAM guard then rejects). Force the tail to
    // run locally on RTX so every image cell gets a K write in this same decode.
    const bool is_embd_batch = (batch_copy.embd != nullptr);
    const bool use_remote_tail_handoff =
            has_fa_tail &&
            is_final_prompt_chunk &&
            !use_safe_full_no_fa &&
            !no_remote_tail_env &&
            !is_embd_batch &&
            prefix_tokens > 0 &&
            tail_tokens > 0 &&
            tail_tokens < ubatch_size &&
            decode_handoff_configured;
    const bool defer_tail_to_next_rtx_batch =
            has_fa_tail &&
            !is_final_prompt_chunk &&
            !use_safe_full_no_fa &&
            !is_embd_batch &&
            prefix_tokens > 0 &&
            tail_tokens > 0;
    const bool tail_not_run_on_rtx = use_remote_tail_handoff || defer_tail_to_next_rtx_batch;
    const int rtx_prefill_tokens = tail_not_run_on_rtx ? prefix_tokens : batch_copy.n_tokens;
    const char * tail_mode =
            tail_tokens <= 0               ? "none" :
            use_safe_full_no_fa            ? "full-no-fa" :
            use_remote_tail_handoff        ? "handoff" :
            defer_tail_to_next_rtx_batch   ? "deferred" :
                                             "local-no-fa";

    const bool logits_delivery_debug = std::getenv("LLAMA_STREAM_LOCAL_LOGITS_DEBUG") != nullptr;
    if (logits_delivery_debug) {
        int marked = 0;
        for (int i = 0; batch_copy.logits && i < batch_copy.n_tokens; ++i) {
            marked += batch_copy.logits[i] ? 1 : 0;
        }
        fprintf(stderr,
                "[LOGITS_DELIVERY] stream-select n_tokens=%d pos0=%d marked=%d last_mark=%d "
                "final=%d fa_tail=%d prefix=%d tail=%d mode=%s handoff_cfg=%d remote=%d defer=%d\n",
                batch_copy.n_tokens, chunk_pos0, marked,
                batch_copy.logits && batch_copy.n_tokens > 0 ? (int) batch_copy.logits[batch_copy.n_tokens - 1] : -1,
                (int) is_final_prompt_chunk, (int) has_fa_tail, prefix_tokens, tail_tokens, tail_mode,
                (int) decode_handoff_configured, (int) use_remote_tail_handoff,
                (int) defer_tail_to_next_rtx_batch);
        fflush(stderr);
    }

    auto make_batch_view = [&](llama_batch & src, int offset, int n_tokens) -> llama_batch {
        return llama_batch {
            /*.n_tokens =*/ n_tokens,
            /*.token    =*/ src.token ? src.token + offset : nullptr,
            /*.embd     =*/ src.embd ? src.embd + (size_t) offset * model->hparams.n_embd : nullptr,
            /*.pos      =*/ src.pos ? src.pos + offset : nullptr,
            /*.n_seq_id =*/ src.n_seq_id ? src.n_seq_id + offset : nullptr,
            /*.seq_id   =*/ src.seq_id ? src.seq_id + offset : nullptr,
            /*.logits   =*/ src.logits ? src.logits + offset : nullptr,
        };
    };

    auto run_decode = [&](llama_batch chunk, const char * label) -> int {
        if (logits_delivery_debug) {
            int marked = 0;
            for (int i = 0; chunk.logits && i < chunk.n_tokens; ++i) {
                marked += chunk.logits[i] ? 1 : 0;
            }
            fprintf(stderr,
                    "[LOGITS_DELIVERY] inner-enter label=%s n_tokens=%d pos0=%d marked=%d last_mark=%d\n",
                    label, chunk.n_tokens, chunk.pos && chunk.n_tokens > 0 ? chunk.pos[0] : -1, marked,
                    chunk.logits && chunk.n_tokens > 0 ? (int) chunk.logits[chunk.n_tokens - 1] : -1);
            fflush(stderr);
        }

        int decode_ret;
        if (use_layer_major) {
            if (prefill_debug >= 1) {
                fprintf(stderr, "*** PREFILL: calling decode_layer_major(%s) n_tokens=%d fa=%s ***\n",
                               label, chunk.n_tokens, ctx->is_flash_attn_enabled() ? "on" : "off");
            }
            decode_ret = ctx->decode_layer_major(chunk, hidden_mode, kv_mode);
        } else {
            if (prefill_debug >= 1) {
                fprintf(stderr, "*** PREFILL: calling llama_decode(%s) n_tokens=%d overlap=%s skip_sync=%s fa=%s ***\n",
                               label, chunk.n_tokens, enable_overlap ? "on" : "off", use_wf ? "on" : "off",
                               ctx->is_flash_attn_enabled() ? "on" : "off");
            }
            decode_ret = llama_decode(ctx, chunk);
        }

        if (logits_delivery_debug) {
            fprintf(stderr, "[LOGITS_DELIVERY] inner-exit label=%s ret=%d\n", label, decode_ret);
            fflush(stderr);
        }
        return decode_ret;
    };

    if (use_safe_full_no_fa) {
        const bool flash_attn_orig = ctx->is_flash_attn_enabled();
        const bool force_safe_inputs_orig = ctx->get_force_safe_mla_multiubatch_inputs();
        const bool kv_layout_flash_attn_orig = ctx->get_prefill_kv_layout_flash_attn();
        LLAMA_LOG_INFO("%s: FA split: prefix=%d tail=%d is too imbalanced for hybrid handoff; running full batch with flash_attn disabled\n",
                       __func__, prefix_tokens, tail_tokens);
        ctx->set_force_safe_mla_multiubatch_inputs(true);
        ctx->set_flash_attn_enabled(false);
        ctx->set_prefill_kv_layout_flash_attn(false);
        ctx->invalidate_graph();
        ret = run_decode(batch_copy, "full-safe-no-fa");
        ctx->set_flash_attn_enabled(flash_attn_orig);
        ctx->set_force_safe_mla_multiubatch_inputs(force_safe_inputs_orig);
        if (ret < 0) {
            ctx->set_prefill_kv_layout_flash_attn(kv_layout_flash_attn_orig);
        }
        ctx->invalidate_graph();
    } else if (has_fa_tail && prefix_tokens > 0) {
        const bool flash_attn_orig = ctx->is_flash_attn_enabled();

        LLAMA_LOG_INFO("%s: FA split: pos0=%d prefix=%d tail=%d total_after=%d align=%d (%s)\n",
                       __func__, chunk_pos0, prefix_tokens, tail_tokens, total_after_chunk,
                       fa_tail_align,
                       use_remote_tail_handoff ? "tail handed to receiver" :
                       defer_tail_to_next_rtx_batch ? "tail deferred to next RTX batch" :
                       "tail runs locally with flash_attn disabled");

        ret = 0;
        if (prefix_tokens > 0) {
            ret = run_decode(make_batch_view(batch_copy, 0, prefix_tokens), "fa-prefix");
        }

        if (ret >= 0 && tail_tokens > 0) {
            if (use_remote_tail_handoff) {
                ctx->set_prefill_handoff_streamed_prefix_count((uint32_t) (chunk_pos0 + prefix_tokens));
                // decode_layer_major() already updated n_p_eval / last_token_id
                // for the aligned FA prefix inside the context.
                n_p_eval_already_fixed = true;
                ret = 0;
            } else if (defer_tail_to_next_rtx_batch) {
                ctx->set_prefill_handoff_streamed_prefix_count((uint32_t) (chunk_pos0 + prefix_tokens));
                ctx->update_n_p_eval(chunk_pos0 + prefix_tokens);
                n_p_eval_already_fixed = true;
                ret = 0;
            } else {
                if (use_wf && prefix_tokens > 0) {
                    LLAMA_LOG_INFO("%s: FA tail fallback: restarting weight fetcher between prefix and tail passes\n",
                                   __func__);
                    wf_end_pass(wf);
                    wf_start(wf);
                }

                // Phase 81 extended: respect LLAMA_PREFILL_NO_FA_DISABLE here too.
                // Phase 81 originally only guarded the prefix=0 branch (line ~2371).
                // The prefix>0 && tail>0 branch unconditionally disabled FA for the
                // unaligned tail, producing numerically-different K cells than the
                // FA-on prefix and corrupting Mac MLX MLA decode (`_TKS_TKS_…` for
                // 5743-tok / 111-tail prompts). The MLA FA kernel handles unaligned
                // input fine; with NO_FA_DISABLE=1 keep FA on for the tail too so
                // the K cells stay FA-on-compatible.
                const bool no_fa_disable_tail =
                        std::getenv("LLAMA_PREFILL_NO_FA_DISABLE") != nullptr ||
                        keep_dsv4_stream_local_fa;
                if (!no_fa_disable_tail) {
                    ctx->set_flash_attn_enabled(false);
                    ctx->invalidate_graph();
                } else {
                    LLAMA_LOG_INFO("%s: FA tail fallback: KEEPING FA on for unaligned tail (%d tokens) per LLAMA_PREFILL_NO_FA_DISABLE=1\n",
                                   __func__, tail_tokens);
                    // Phase 196 (2026-05-02): force graph rebuild for the tail call.
                    // Without this, the tail decode_layer_major call reuses the prefix
                    // graph (built for n_ubatch sized SET_ROWS, e.g. 2048) and
                    // dispatches (n_ubatch - tail_tokens) stale k_idxs row writes —
                    // the 256-alignment bug that causes broken-band garbage on every
                    // n_in % 256 != 0 prompt going through streaming-inject.
                    // FA-disable branch above already invalidates as a side effect of
                    // toggling FA; this branch keeps FA on but still must invalidate
                    // for the tail's smaller batch size.
                    ctx->invalidate_graph();
                }
                const int tail_ret = run_decode(make_batch_view(batch_copy, prefix_tokens, tail_tokens), "safe-tail");
                if (tail_ret >= 0) {
                    // decode_layer_major() rewrites n_p_eval to the tail chunk length.
                    // Restore the full prompt-visible token count for artifact export.
                    ctx->update_n_p_eval(chunk_pos0 + prefix_tokens);
                    ctx->set_prefill_handoff_streamed_prefix_count((uint32_t) total_after_chunk);
                    n_p_eval_already_fixed = true;
                    // Successful inner passes still need to report streaming success
                    // to the outer decode path, which expects 0 rather than
                    // decode_layer_major()'s positive token count.
                    ret = 0;
                } else {
                    ret = tail_ret;
                }
            }
        }

        if (!use_remote_tail_handoff) {
            ctx->set_flash_attn_enabled(flash_attn_orig);
            ctx->invalidate_graph();
        }
        if (ret >= 0) {
            ret = 0;
        }
    } else {
        ret = run_decode(batch_copy, "full");
    }
    if (ret >= 0 && batch_copy.token && batch_copy.n_tokens > 0 && !use_remote_tail_handoff && !defer_tail_to_next_rtx_batch) {
        ctx->set_prefill_handoff_streamed_prefix_count((uint32_t) total_after_chunk);
        fprintf(stderr, "*** STREAMED_PREFIX_UPDATE: total_after_chunk=%d (chunk_pos0=%d batch_n_tokens=%d use_remote=%d defer=%d)\n",
                total_after_chunk, chunk_pos0, batch_copy.n_tokens, (int)use_remote_tail_handoff, (int)defer_tail_to_next_rtx_batch);
        fflush(stderr);
    } else {
        fprintf(stderr, "*** STREAMED_PREFIX_NOT_UPDATED: ret=%d batch.token=%p batch.n_tokens=%d use_remote=%d defer=%d\n",
                ret, (void*)batch_copy.token, batch_copy.n_tokens, (int)use_remote_tail_handoff, (int)defer_tail_to_next_rtx_batch);
        fflush(stderr);
    }
    ctx->set_inside_streaming_decode(false);

    // Mark streaming prefill as done so subsequent llama_decode() calls
    // (e.g. generation, chat-template re-eval) don't create a second WF
    // and OOM on large models.
    if (ret >= 0 && use_wf) {
        ctx->set_wf_prefill_done(true);
    }

    // Update n_p_eval so KV artifact export knows the token count.
    // decode_layer_major returns the number of tokens processed but
    // doesn't update n_p_eval (that's done in the standard path).
    if (ret > 0 && !n_p_eval_already_fixed) {
        ctx->update_n_p_eval(ret);
    }

    if (prefill_debug >= 1) {
        fprintf(stderr, "*** PREFILL: llama_decode returned %d ***\n", ret);
    }

    // Restore skip_eval_sync and do a final GPU sync to ensure the last
    // layer's compute is done before we read logits / KV cache.
    if (use_wf) {
        auto * sched = ctx->get_sched();
        if (sched) {
            ggml_backend_sched_set_eval_skip_sync(sched, false);
        }
        // Final sync via the backend API (works for CUDA, Metal, etc.)
        ggml_backend_synchronize(gpu_backend);
    }

    if (ret >= 0 && dsv4_route_trace_enabled) {
        int captured_layers = 0;
        for (const auto & layer : dsv4_route_trace_layers) {
            captured_layers += layer.captured ? 1 : 0;
        }
        if (captured_layers != n_layer) {
            LLAMA_LOG_WARN("%s: DSV4 route trace captured %d/%d layers\n",
                    __func__, captured_layers, n_layer);
        }
        const int trace_token_index = params.route_trace_prompt_last ? batch.n_tokens - 1 : 0;
        const int trace_pos = batch.pos ? batch.pos[trace_token_index] : -1;
        llama_dsv4_route_trace_append(
                dsv4_route_trace_path, trace_pos, batch.token[trace_token_index], dsv4_route_trace_layers);
    }

    if (ret >= 0 && dsv4_lid_topk_trace_enabled) {
        int captured_layers = 0;
        for (const auto & layer : dsv4_lid_topk_trace_layers) {
            captured_layers += layer.captured ? 1 : 0;
        }
        if (captured_layers == 0) {
            LLAMA_LOG_WARN("%s: DSV4 LID top-k trace captured no layers\n", __func__);
        }
        const int trace_pos = batch.pos ? batch.pos[0] : -1;
        llama_dsv4_lid_topk_trace_append(
                dsv4_lid_topk_trace_path, trace_pos, force_bitonic, dsv4_lid_topk_trace_layers);
    }

    // Restore flags so the handoff/reserve happens at the right time
    // (after all streaming prefill batches are done).
    ctx->restore_sched_and_handoff(guard);

    auto  t_compute_end = std::chrono::high_resolution_clock::now();
    float compute_ms    = std::chrono::duration<float, std::milli>(t_compute_end - t_compute_start).count();
    const float stream_pass_s = compute_ms / 1000.0f;
    const float rtx_prefill_tok_s = stream_pass_s > 0.0f ? rtx_prefill_tokens / stream_pass_s : 0.0f;
    const float equiv_prefill_tok_s = stream_pass_s > 0.0f ? batch_copy.n_tokens / stream_pass_s : 0.0f;
    fprintf(stderr,
            "*** PREFILL_EQUIV: prompt_tokens=%d rtx_tokens=%d prefix=%d tail=%d tail_mode=%s wall=%.2fms rtx_tok_s=%.1f equiv_tok_s=%.1f ***\n",
            batch_copy.n_tokens, rtx_prefill_tokens, prefix_tokens, tail_tokens, tail_mode,
            compute_ms, rtx_prefill_tok_s, equiv_prefill_tok_s);
    fflush(stderr);

    if (pending_upload.active) {
        const int il = pending_upload.layer_idx;
        if (il >= 0 && il < n_layer) {
            upload_times[il] = pending_upload.fut.get();
            layer_loaded[il] = true;
        } else {
            pending_upload.fut.wait();
        }
        pending_upload.active = false;
    }

    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: streaming path failed (ret=%d) — returning error (no fallback)\n", __func__, ret);

        // Clean up callbacks and tensor pointers
        ctx->set_per_layer_callbacks(nullptr, nullptr);
        ctx->set_lid_topk_callback(nullptr);
        callbacks_registered = false;
        prefill_restore_tensor_pointers(layer_infos);
        remap_applied = false;
        ctx->set_prefill_streaming(false);

        result->layer_metrics.clear();
        result->used_streaming = false;
        result->n_tokens_processed = 0;
        result->total_time_ms =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
        result->total_tok_s = 0;
        // Restore FA state before returning
        ctx->set_flash_attn_enabled(flash_attn_saved);
        ctx->set_prefill_kv_layout_flash_attn(flash_attn_layout_saved);
        ctx->invalidate_graph();

        // Phase 252 (2026-05-03): MISSING CLEANUP on error path was leaking
        // gpu_buf_a/b (~17GB), pinned_a/b (~17GB host), cuda_pipeline streams,
        // and WF resources on every failed request. After 1-2 failures the
        // server OOMs on the next request's GPU buffer alloc and crashes.
        // Mirroring the success-path cleanup at lines 1729-1781.
        if (gpu_backend) {
            ggml_backend_synchronize(gpu_backend);
        }
        if (wf) {
            wf_stop(wf);
            wf_quiesce(wf);
            wf_free_prefetch(wf);
            // Default behavior is no-cache (Phase 250), so destroy unconditionally
            // unless the user opted in to cache reuse and stored it.
            if (ctx->get_cached_wf() != wf) {
                wf_destroy(wf);
            }
            wf = nullptr;
        }
        if (cuda_pipeline) {
#ifdef GGML_USE_CUDA
            ggml_backend_cuda_prefill_sync_copy_stream(cuda_pipeline);
            ggml_backend_cuda_prefill_pipeline_free(cuda_pipeline);
#endif
            cuda_pipeline = nullptr;
        }
#ifdef GGML_USE_CUDA
        if (pinned_a) { ggml_backend_cuda_prefill_pinned_free(pinned_a); pinned_a = nullptr; }
        if (pinned_b) { ggml_backend_cuda_prefill_pinned_free(pinned_b); pinned_b = nullptr; }
#endif
        // Phase 328: skip free in persist mode (this is the failure-path
        // cleanup; persistent buffers stay alive for the next attempt)
        if (!persist_bufs) {
            if (gpu_buf_a) { ggml_backend_buffer_free(gpu_buf_a); gpu_buf_a = nullptr; }
            if (gpu_buf_b) { ggml_backend_buffer_free(gpu_buf_b); gpu_buf_b = nullptr; }
        }
        if (gpu_buf_persist) { ggml_backend_buffer_free(gpu_buf_persist); gpu_buf_persist = nullptr; }
        return ret;  // return the error, don't fallback
    }

    compute_times = ctx->get_last_layer_compute_times_ms();
    if ((int) compute_times.size() != n_layer) {
        compute_times.assign(n_layer, 0.0f);
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();

    // ====================================================================
    // Step 5: Build metrics and report
    // ====================================================================

    size_t total_bytes     = 0;
    float  total_upload_ms = 0;
    for (int il = 0; il < n_layer; ++il) {
        llama_layer_metrics m;
        m.layer_idx       = il;
        m.weight_bytes    = layer_infos[il].total_bytes;
        m.compute_time_ms = compute_times[il];
        m.load_time_ms    = upload_times[il];
        m.stall_time_ms   = stall_times[il];
        result->layer_metrics.push_back(m);
        total_bytes += m.weight_bytes;
        total_upload_ms += upload_times[il];
    }

    result->total_time_ms      = std::chrono::duration<float, std::milli>(t_total_end - t_total_start).count();
    result->n_tokens_processed = batch.n_tokens;
    result->total_tok_s    = (result->total_time_ms > 0) ? batch.n_tokens / (result->total_time_ms / 1000.0f) : 0.0f;
    result->used_streaming = true;
    result->host_cache_enabled = host_cache.enabled;
    result->host_cache_advice_supported = host_cache.advice_supported;
    result->host_prefetch_layers = host_cache.prefetched_layers;
    result->host_prefetch_bytes  = host_cache.prefetched_bytes;
    result->host_evict_layers = host_cache.evicted_layers;
    result->host_evict_bytes  = host_cache.evicted_bytes;
    result->host_cache_lookups = host_cache.lookups;
    result->host_cache_hits = host_cache.hits;
    result->host_cache_misses = host_cache.misses;
    result->host_cache_hit_rate = host_cache.lookups > 0
        ? (100.0f * ((float) host_cache.hits / (float) host_cache.lookups))
        : 0.0f;
    result->host_storage_wait_ms = host_cache.storage_wait_ms;

    // Summary
    float upload_bw = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_upload_ms / 1000.0);

    float total_compute_ms = 0;
    for (int il = 0; il < n_layer; ++il) { total_compute_ms += compute_times[il]; }
    float total_stall_ms = 0;
    for (int il = 0; il < n_layer; ++il) { total_stall_ms += stall_times[il]; }

    if (prefill_debug >= 1) {
        fprintf(stderr, "\n*** PREFILL SUMMARY ***\n");
        fprintf(stderr, "*** pipeline: %s ***\n",
                use_wf ? "WEIGHT-FETCHER (NVMe → GPU DMA)" :
                use_pipeline ? "DUAL-STREAM (pinned + copy stream)" : "legacy (pageable, compute stream)");
        fprintf(stderr, "*** layers: %d, weight data: %.2f GB ***\n", n_layer,
                       total_bytes / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "*** total upload: %.2f ms (%.2f GB/s effective) ***\n", total_upload_ms, upload_bw);
        fprintf(stderr, "*** total compute (per-layer sum): %.2f ms ***\n", total_compute_ms);
        fprintf(stderr, "*** total stall (waiting for async): %.2f ms ***\n", total_stall_ms);
        fprintf(stderr, "*** forward pass (wall clock): %.2f ms ***\n", compute_ms);
        fprintf(stderr, "*** overhead (wall - upload - compute): %.2f ms ***\n",
                       compute_ms - total_upload_ms - total_compute_ms);
        fprintf(stderr, "*** total: %d tokens, %.2f ms, %.2f tok/s ***\n", result->n_tokens_processed,
                       result->total_time_ms, result->total_tok_s);
    }

    // (RAM prefetch two-hop test runs at init and prints its own results)

    LLAMA_LOG_INFO("%s: === Phase 2.1 Pointer-Rerouted Summary ===\n", __func__);
    LLAMA_LOG_INFO("%s: layers: %d, weight data: %.2f GB\n", __func__, n_layer,
                   total_bytes / (1024.0 * 1024.0 * 1024.0));
    LLAMA_LOG_INFO("%s: total upload: %.2f ms (%.2f GB/s effective)\n", __func__, total_upload_ms, upload_bw);
    LLAMA_LOG_INFO("%s: forward pass: %.2f ms\n", __func__, compute_ms);
    LLAMA_LOG_INFO("%s: total: %d tokens, %.2f ms, %.2f tok/s\n", __func__, result->n_tokens_processed,
                   result->total_time_ms, result->total_tok_s);
    if (host_cache.enabled) {
        LLAMA_LOG_INFO("%s: S3 host moving-window activity: prefetched=%d layers (%.2f MB), evicted=%d layers (%.2f MB), lookups=%d hits=%d misses=%d hit_rate=%.1f%% storage_wait_ms=%.2f advise_supported=%s\n",
                       __func__, host_cache.prefetched_layers, host_cache.prefetched_bytes / (1024.0 * 1024.0),
                       host_cache.evicted_layers, host_cache.evicted_bytes / (1024.0 * 1024.0),
                       host_cache.lookups, host_cache.hits, host_cache.misses,
                       host_cache.lookups > 0 ? (100.0f * ((float) host_cache.hits / (float) host_cache.lookups)) : 0.0f,
                       host_cache.storage_wait_ms,
                       host_cache.advice_supported ? "true" : "false");
    }

    if (params.telemetry) {
        LLAMA_LOG_INFO("%s: per-layer upload times:\n", __func__);
        for (int il = 0; il < n_layer; ++il) {
            LLAMA_LOG_INFO("  layer %3d: upload=%.2fms bytes=%.2fMB\n", il, upload_times[il],
                           layer_infos[il].total_bytes / (1024.0 * 1024.0));
        }
    }

    // Restore FA state for subsequent requests (prefix=0 path may have disabled it)
    ctx->set_flash_attn_enabled(flash_attn_saved);
    ctx->set_prefill_kv_layout_flash_attn(flash_attn_layout_saved);
    if (flash_attn_saved != ctx->is_flash_attn_enabled()) {
        ctx->invalidate_graph();
    }

    return ret;
}
