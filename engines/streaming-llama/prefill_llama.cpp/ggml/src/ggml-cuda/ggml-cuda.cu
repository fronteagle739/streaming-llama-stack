#include "ggml-cuda.h"
#include "ggml-ds4-native-half.h"
#include "ggml-dsv41-dense-half.h"
#include "ggml-dsv41-hc-half.h"
#include "ggml-dsv41-native-fp4.h"
#include "ggml-dsv41-index-cublas.h"
#include "ggml-dsv41-expert-cublas.h"
#include "ggml-cuda/dsv41-expert-cublas.cuh"
#include "ggml-dsv41-hc-expand.h"
#include "ggml-cuda/dsv41-ds4-hc-expand.cuh"
#include "ggml-cuda/ds4split-native-half-witness.h"
#include "ggml-cuda/ds4split-q4k-half-mmid.cuh"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include <thread>   // std::this_thread::sleep_for (LLAMA_CUDA_PACE_SLEEP_US)
#include <chrono>

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/diffusion-sampling.cuh"
#include "ggml-cuda/dsa-indexer-score.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-act-qat.cuh"
#include "ggml-cuda/dsv41-ops.cuh"
#include "ggml-cuda/flash-attn-sparse.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmid.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmq_id.cuh"
#include "ggml-dsv41-expert-routing.h"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/quantize_id.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

// ---------------------------------------------------------------------------
// Lightweight per-op CUDA profiling  (LLAMA_CUDA_OP_PROFILE=1 to enable)
// ---------------------------------------------------------------------------
#include <cstring>

static bool g_op_profile_enabled = false;

struct OpProfile {
    std::atomic<int64_t> count{0};
    // We accumulate elapsed_ms in a double protected by a mutex to avoid
    // atomic floating-point issues without pulling in libatomic.
    double   total_ms = 0.0;
    std::mutex mtx;
    void add(double ms) {
        count.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mtx);
        total_ms += ms;
    }
};

// Indexed by GGML_OP enum value.  GGML_OP_COUNT is the sentinel.
static OpProfile g_op_profiles[GGML_OP_COUNT];

// Optional bounded semantic-node split of the SAME per-op event measurements.
// Names, operand origins and shapes distinguish dense/attention/HC/index ops;
// repeated instances across tiles aggregate without a graph-pointer key.
static bool g_node_profile_enabled = false;
struct NodeProfile { int64_t count=0; double total_ms=0; };
static std::mutex g_node_profiles_mutex;
static std::unordered_map<std::string,NodeProfile> g_node_profiles;
static constexpr size_t node_profile_limit=32768;
static const ggml_tensor * node_profile_origin(const ggml_tensor *t) {
    for(int depth=0;t&&t->src[0]&&depth<12;++depth){
        switch(t->op){case GGML_OP_VIEW:case GGML_OP_RESHAPE:case GGML_OP_PERMUTE:case GGML_OP_TRANSPOSE:case GGML_OP_CONT:case GGML_OP_CPY:t=t->src[0];break;
            default:return t;}
    }return t;
}
static std::string node_profile_key(const ggml_tensor *t) {
    const auto*a=t->src[0],*b=t->src[1],*oa=node_profile_origin(a),*ob=node_profile_origin(b);
    char key[1024];const int used=snprintf(key,sizeof(key),
        "%s|node=%s|a=%s|b=%s|type=%s|dst=%lld,%lld,%lld,%lld|ashape=%lld,%lld,%lld,%lld|bshape=%lld,%lld,%lld,%lld",
        ggml_op_name(t->op),t->name,oa?oa->name:"-",ob?ob->name:"-",ggml_type_name(t->type),
        (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
        a?(long long)a->ne[0]:0,a?(long long)a->ne[1]:0,a?(long long)a->ne[2]:0,a?(long long)a->ne[3]:0,
        b?(long long)b->ne[0]:0,b?(long long)b->ne[1]:0,b?(long long)b->ne[2]:0,b?(long long)b->ne[3]:0);
    GGML_ASSERT(used>=0&&size_t(used)<sizeof(key));
    for(int i=0;i<used;++i)if(key[i]=='\n'||key[i]=='\r'||key[i]=='\t')key[i]='_';
    return key;
}
static void node_profile_add(const std::string&key,double ms){
    if(key.empty())return;std::lock_guard<std::mutex> lock(g_node_profiles_mutex);
    auto found=g_node_profiles.find(key);
    if(found==g_node_profiles.end()){GGML_ASSERT(g_node_profiles.size()<node_profile_limit);found=g_node_profiles.emplace(key,NodeProfile{}).first;}
    ++found->second.count;found->second.total_ms+=ms;
}

// MUL_MAT_ID route counters
static std::atomic<int64_t> g_mmid_route_mmq_id{0};
static std::atomic<int64_t> g_mmid_route_mmvq{0};
static std::atomic<int64_t> g_mmid_route_mmvf{0};
static std::atomic<int64_t> g_mmid_route_mmq{0};
static std::atomic<int64_t> g_mmid_route_mmf{0};
static std::atomic<int64_t> g_mmid_route_cublas{0};  // fallback sort+scatter path

// Forward decl — actual public C wrapper at the bottom (after
// cuda_op_profile_dump's definition).

static void cuda_op_profile_dump() {
    if (!g_op_profile_enabled) return;

    // Collect non-zero ops into a sortable vector.
    struct Entry { const char * name; int64_t count; double total_ms; };
    std::vector<Entry> entries;
    for (int op = 0; op < (int)GGML_OP_COUNT; ++op) {
        if (g_op_profiles[op].count.load(std::memory_order_relaxed) == 0) continue;
        entries.push_back({ggml_op_name((ggml_op)op),
                           g_op_profiles[op].count.load(std::memory_order_relaxed),
                           g_op_profiles[op].total_ms});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry & a, const Entry & b){ return a.total_ms > b.total_ms; });

    fprintf(stderr, "\n=== CUDA OP PROFILE (top %zu of %zu ops) ===\n",
            entries.size(), entries.size());
    fprintf(stderr, "%-30s %8s %12s %10s\n", "op", "count", "total_ms", "avg_ms");
    for (size_t i = 0; i < entries.size() && i < 30; ++i) {
        const auto & e = entries[i];
        fprintf(stderr, "%-30s %8" PRId64 " %12.2f %10.4f\n",
                e.name, e.count, e.total_ms, e.total_ms / (double)e.count);
    }

    // MUL_MAT_ID route breakdown
    auto mm_id  = g_mmid_route_mmq_id.load();
    auto mm_vq  = g_mmid_route_mmvq.load();
    auto mm_vf  = g_mmid_route_mmvf.load();
    auto mm_q   = g_mmid_route_mmq.load();
    auto mm_f   = g_mmid_route_mmf.load();
    auto mm_cub = g_mmid_route_cublas.load();
    fprintf(stderr, "\n=== MUL_MAT_ID route breakdown ===\n");
    fprintf(stderr, "  mmq_id  : %" PRId64 "\n", mm_id);
    fprintf(stderr, "  mmvq    : %" PRId64 "\n", mm_vq);
    fprintf(stderr, "  mmvf    : %" PRId64 "\n", mm_vf);
    fprintf(stderr, "  mmq     : %" PRId64 "\n", mm_q);
    fprintf(stderr, "  mmf     : %" PRId64 "\n", mm_f);
    fprintf(stderr, "  cublas  : %" PRId64 "\n", mm_cub);
    fprintf(stderr, "=== END CUDA OP PROFILE ===\n\n");
    if(g_node_profile_enabled){
        std::vector<std::pair<std::string,NodeProfile>> rows;
        {std::lock_guard<std::mutex> lock(g_node_profiles_mutex);rows.assign(g_node_profiles.begin(),g_node_profiles.end());}
        std::sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){return a.second.total_ms>b.second.total_ms;});
        fprintf(stderr,"DS4PERF_NODE_PROFILE_HEADER\tcount\tgpu_ms\top|node|origins|shapes cap=%zu\n",node_profile_limit);
        for(const auto&r:rows)fprintf(stderr,"DS4PERF_NODE_PROFILE\t%" PRId64 "\t%.9f\t%s\n",r.second.count,r.second.total_ms,r.first.c_str());
        fprintf(stderr,"DS4PERF_NODE_PROFILE_END\trows=%zu\n",rows.size());
    }
    fflush(stderr);  // 2026-07-10: the DUMP_EACH/atexit output is lost on Windows force-kill without an explicit flush
}

// Public C wrapper (exported from ggml-cuda.dll via GGML_BACKEND_API).
// Phase 1.3: callers in llama.dll invoke this after each prefill when
// LLAMA_CUDA_OP_PROFILE_DUMP_EACH=1 so we capture per-op CUDA timings
// without depending on the Windows-flaky atexit path.
extern "C" GGML_BACKEND_API void ggml_cuda_op_profile_dump_now() {
    cuda_op_profile_dump();
}

// LLAMA_DLM_REMAINDER_TIMERS=2 uses a separate profiler from the legacy
// LLAMA_CUDA_OP_PROFILE path above. The legacy path synchronizes measured
// operations and changes the scheduler-drain attribution this probe observes.
// This sampler records one event pair per 64 dispatches (plus each family's
// first dispatch), reuses a small ring, and only harvests already-complete
// events. It never synchronizes and records nothing during stream capture.
// Mode 1 is intentionally CPU-only and never enters this CUDA-event path.
static std::atomic<bool> g_dlm_remainder_profile_enabled{false};

static constexpr uint64_t DLM_REMAINDER_SAMPLE_EVERY = 64;
static constexpr size_t   DLM_REMAINDER_EVENT_SLOTS  = 64;

struct dlm_remainder_gpu_family_stat {
    uint64_t dispatches = 0;
    uint64_t samples = 0;
    double sampled_ms = 0.0;
};

struct dlm_remainder_gpu_slot {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    uint64_t epoch = 0;
    int op = -1;
    bool pending = false;
};

struct dlm_remainder_gpu_state {
    std::array<dlm_remainder_gpu_family_stat, GGML_OP_COUNT> families = {};
    std::array<dlm_remainder_gpu_slot, DLM_REMAINDER_EVENT_SLOTS> slots = {};
    uint64_t epoch = 0;
    size_t next_slot = 0;
    uint64_t capture_skips = 0;
    uint64_t ring_full_drops = 0;
    uint64_t event_errors = 0;
    uint64_t abandoned_event_pairs = 0;
    bool active = false;
};

static thread_local dlm_remainder_gpu_state g_dlm_remainder_gpu_state;

static void dlm_remainder_gpu_release_events(dlm_remainder_gpu_state & state) {
    for (auto & slot : state.slots) {
        if (slot.stop) {
            if (cudaEventDestroy(slot.stop) != cudaSuccess) {
                (void) cudaGetLastError();
                ++state.event_errors;
            }
        }
        if (slot.start) {
            if (cudaEventDestroy(slot.start) != cudaSuccess) {
                (void) cudaGetLastError();
                ++state.event_errors;
            }
        }
        slot = {};
    }
    state.next_slot = 0;
    state.active = false;
}

static void dlm_remainder_gpu_abandon_stale_events(dlm_remainder_gpu_state & state) {
    // reset() can be reached after an earlier chunk aborted before dump_now().
    // Those handles may belong to a CUDA context that has already been torn
    // down. Do not call into the driver with them: that exact stale-handle
    // pattern has produced nvcuda64.dll AVs on Windows. The normal successful
    // path destroys every pair in dump_now(); this is diagnostic-only fallout.
    for (auto & slot : state.slots) {
        if (slot.start || slot.stop) {
            ++state.abandoned_event_pairs;
        }
        slot = {};
    }
    state.next_slot = 0;
    state.active = false;
}

static bool dlm_remainder_gpu_harvest_slot(
        dlm_remainder_gpu_state & state,
        dlm_remainder_gpu_slot & slot) {
    if (!slot.pending) {
        return true;
    }

    const cudaError_t query = cudaEventQuery(slot.stop);
    if (query == cudaErrorNotReady) {
        return false;
    }
    if (query != cudaSuccess) {
        (void) cudaGetLastError();
        ++state.event_errors;
        slot.pending = false;
        return true;
    }

    float elapsed_ms = 0.0f;
    const cudaError_t elapsed = cudaEventElapsedTime(&elapsed_ms, slot.start, slot.stop);
    if (elapsed == cudaSuccess && slot.epoch == state.epoch &&
            slot.op >= 0 && slot.op < (int) GGML_OP_COUNT) {
        auto & family = state.families[(size_t) slot.op];
        ++family.samples;
        family.sampled_ms += (double) elapsed_ms;
    } else if (elapsed != cudaSuccess) {
        (void) cudaGetLastError();
        ++state.event_errors;
    }
    slot.pending = false;
    return true;
}

static dlm_remainder_gpu_slot * dlm_remainder_gpu_acquire_slot(
        dlm_remainder_gpu_state & state) {
    for (size_t n = 0; n < DLM_REMAINDER_EVENT_SLOTS; ++n) {
        const size_t idx = (state.next_slot + n) % DLM_REMAINDER_EVENT_SLOTS;
        auto & slot = state.slots[idx];
        if (!dlm_remainder_gpu_harvest_slot(state, slot)) {
            continue;
        }
        state.next_slot = (idx + 1) % DLM_REMAINDER_EVENT_SLOTS;
        return &slot;
    }
    ++state.ring_full_drops;
    return nullptr;
}

struct CudaDlmRemainderTimer {
    dlm_remainder_gpu_slot * slot = nullptr;
    cudaStream_t stream = nullptr;

    CudaDlmRemainderTimer(cudaStream_t stream_in, int op) : stream(stream_in) {
        if (!g_dlm_remainder_profile_enabled.load(std::memory_order_relaxed) ||
                g_op_profile_enabled || op < 0 || op >= (int) GGML_OP_COUNT) {
            return;
        }

        auto & state = g_dlm_remainder_gpu_state;
        auto & family = state.families[(size_t) op];
        const uint64_t dispatch = ++family.dispatches;
        if (dispatch != 1 && (dispatch % DLM_REMAINDER_SAMPLE_EVERY) != 0) {
            return;
        }

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t capture_err = cudaStreamIsCapturing(stream, &capture_status);
        if (capture_err != cudaSuccess || capture_status != cudaStreamCaptureStatusNone) {
            if (capture_err != cudaSuccess) {
                (void) cudaGetLastError();
                ++state.event_errors;
            } else {
                ++state.capture_skips;
            }
            return;
        }

        slot = dlm_remainder_gpu_acquire_slot(state);
        if (!slot) {
            return;
        }
        if (!slot->start || !slot->stop) {
            ++state.event_errors;
            slot = nullptr;
            return;
        }
        slot->epoch = state.epoch;
        slot->op = op;
        if (cudaEventRecord(slot->start, stream) != cudaSuccess) {
            (void) cudaGetLastError();
            ++state.event_errors;
            slot = nullptr;
        }
    }

    ~CudaDlmRemainderTimer() {
        if (!slot) {
            return;
        }
        auto & state = g_dlm_remainder_gpu_state;
        if (cudaEventRecord(slot->stop, stream) == cudaSuccess) {
            slot->pending = true;
        } else {
            (void) cudaGetLastError();
            ++state.event_errors;
        }
    }
};

extern "C" GGML_BACKEND_API void ggml_cuda_dlm_remainder_profile_reset() {
    const char * env = std::getenv("LLAMA_DLM_REMAINDER_TIMERS");
    const int mode = env && env[0] != '\0' ? std::atoi(env) : 0;
    const bool enabled = mode >= 2;
    g_dlm_remainder_profile_enabled.store(enabled, std::memory_order_relaxed);
    if (!enabled) {
        return;
    }

    auto & state = g_dlm_remainder_gpu_state;
    if (state.active) {
        dlm_remainder_gpu_abandon_stale_events(state);
    }
    ++state.epoch;
    if (state.epoch == 0) {
        ++state.epoch;
    }
    state.families = {};
    state.capture_skips = 0;
    state.ring_full_drops = 0;
    state.event_errors = 0;
    for (auto & slot : state.slots) {
        if (slot.start && slot.stop) {
            continue;
        }
        cudaError_t err = cudaEventCreate(&slot.start);
        if (err == cudaSuccess) {
            err = cudaEventCreate(&slot.stop);
        }
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            if (slot.start && !slot.stop) {
                (void) cudaEventDestroy(slot.start);
                slot.start = nullptr;
            }
            ++state.event_errors;
        }
    }
    state.active = true;
}

extern "C" GGML_BACKEND_API void ggml_cuda_dlm_remainder_profile_dump_now() {
    if (!g_dlm_remainder_profile_enabled.load(std::memory_order_relaxed)) {
        return;
    }

    auto & state = g_dlm_remainder_gpu_state;
    uint64_t pending = 0;
    for (auto & slot : state.slots) {
        if (!dlm_remainder_gpu_harvest_slot(state, slot)) {
            ++pending;
        }
    }

    struct entry {
        int op;
        uint64_t dispatches;
        uint64_t samples;
        double sampled_ms;
        double estimated_ms;
    };
    std::vector<entry> entries;
    for (int op = 0; op < (int) GGML_OP_COUNT; ++op) {
        const auto & family = state.families[(size_t) op];
        if (family.dispatches == 0 || family.samples == 0) {
            continue;
        }
        const double estimated_ms = family.sampled_ms *
                (double) family.dispatches / (double) family.samples;
        entries.push_back({ op, family.dispatches, family.samples,
                family.sampled_ms, estimated_ms });
    }
    std::sort(entries.begin(), entries.end(), [](const entry & a, const entry & b) {
        return a.estimated_ms > b.estimated_ms;
    });

    fprintf(stderr,
            "*** DLM_REMAINDER_GPU: mode=2 sample_every=%llu completed_families=%zu pending=%llu capture_skips=%llu ring_drops=%llu errors=%llu abandoned_event_pairs=%llu%s\n",
            (unsigned long long) DLM_REMAINDER_SAMPLE_EVERY, entries.size(),
            (unsigned long long) pending,
            (unsigned long long) state.capture_skips,
            (unsigned long long) state.ring_full_drops,
            (unsigned long long) state.event_errors,
            (unsigned long long) state.abandoned_event_pairs,
            g_op_profile_enabled ? " legacy_op_profile_active" : "");
    for (const auto & item : entries) {
        fprintf(stderr,
                "*** DLM_REMAINDER_GPU: op=%-22s dispatches=%llu samples=%llu sampled_ms=%.3f est_gpu_ms=%.3f sample_avg_ms=%.4f\n",
                ggml_op_name((ggml_op) item.op),
                (unsigned long long) item.dispatches,
                (unsigned long long) item.samples,
                item.sampled_ms, item.estimated_ms,
                item.sampled_ms / (double) item.samples);
    }
    fflush(stderr);

    // The layer-major caller invokes dump_now() after the chunk's final
    // scheduler/KV synchronization and before graph/stream teardown. Destroy
    // the event ring here so no CUDA handle crosses a chunk/context boundary.
    dlm_remainder_gpu_release_events(state);
    g_dlm_remainder_profile_enabled.store(false, std::memory_order_relaxed);
}

// RAII cudaEvent timer — records elapsed GPU time between construction and destruction.
// The stop event is recorded lazily on the stream and measured on destruction via synchronize.
// 2026-05-29: PACE_EVERY_N=4 isolated the per-op cudaEventSynchronize as cheap
// (~0.8s total); the OP_PROFILE crutch's ~6s @20k overhead is almost entirely
// per-op cudaEventCreate/Destroy (8500 ops x4 driver allocs). LLAMA_CUDA_REUSE_EVENTS=1
// reuses two static events (created once) instead of create/destroy-per-op, keeping
// the per-op sync that paces DStorage. Recovers ~5s -> compute ~30.4s -> ~650 @20k.
// Safe because the per-op cudaEventSynchronize consumes the event before the next op
// reuses it (eval_cb is single-threaded on the compute thread). Default off.
static bool g_reuse_op_events = []() {
    const char * e = std::getenv("LLAMA_CUDA_REUSE_EVENTS");
    return e && e[0] && e[0] != '0';
}();
struct CudaOpTimer {
    cudaEvent_t start_ev = nullptr;
    cudaEvent_t stop_ev  = nullptr;
    cudaStream_t stream  = nullptr;
    int op_index         = 0;
    bool owns_events     = true;  // false when borrowed from the reuse-pool
    std::string node_key;

    CudaOpTimer(cudaStream_t s, int op,const ggml_tensor *node=nullptr) : stream(s), op_index(op) {
        if(g_node_profile_enabled&&node)node_key=node_profile_key(node);
        if (g_reuse_op_events) {
            static cudaEvent_t s_start = nullptr, s_stop = nullptr;
            if (!s_start) { cudaEventCreate(&s_start); cudaEventCreate(&s_stop); }
            start_ev = s_start; stop_ev = s_stop; owns_events = false;
        } else {
            cudaEventCreate(&start_ev);
            cudaEventCreate(&stop_ev);
        }
        cudaEventRecord(start_ev, stream);
    }
    ~CudaOpTimer() {
        if (!start_ev) return;
        cudaEventRecord(stop_ev, stream);
        // Phase 1.2 v14 (2026-05-11): only synchronize when the stream is NOT in
        // graph-capture mode. cudaEventSynchronize is illegal during cudaStream*
        // capture and crashes cudaStreamEndCapture later with "operation failed
        // due to a previous error during capture" (validated at n_ubatches=1 fast
        // path with OP_PROFILE=1, ggml-cuda.cu:4624). For multi-ubatch (no graph
        // capture) the per-op host-block here is what gives the WF dma thread CPU
        // time to overlap NVMe with compute — the actual mechanism behind the
        // 547+ tok/s breakthrough vs 465 baseline at Kimi 32k/8ub.
        //
        // 2026-05-29: the per-op cudaEventSynchronize is BOTH the pacing (cedes the
        // CUDA driver lock to the WF DMA thread) AND ~6s of overhead at 20k (~8500
        // ops). Measured: removing it entirely (OP_PROFILE=0) starves the DMA
        // (wf 22ms->9486ms, 533->438 tps). LLAMA_CUDA_PACE_EVERY_N>1 syncs only
        // every Nth op: the host bursts N-1 launches then blocks once, keeping the
        // DMA fed at ~1/N the sync cost. N=1 (default) == original per-op behavior
        // (zero behavior change). Tune N in {2,4,8} for max tok/s while wf stall
        // stays low and coherence holds. Goal: recover the 533->~685 @20k gap.
        static const int pace_every_n = []() {
            const char * e = std::getenv("LLAMA_CUDA_PACE_EVERY_N");
            int n = (e && e[0]) ? atoi(e) : 1;
            return n >= 1 ? n : 1;
        }();
        // 2026-05-29: the crutch's real benefit is keeping the NVMe FETCH fast (17 vs
        // 11.7 GB/s) by BLOCKING the host so DirectStorage's CPU I/O threads get
        // scheduled (not CUDA-lock). LLAMA_CUDA_PACE_SLEEP_US>0 replaces the per-op
        // cudaEventSynchronize (which waits the op's full GPU time, ~6s total) with a
        // per-op std::this_thread::sleep_for(us) that yields the CPU to DStorage at a
        // tiny fraction of the cost (~8500*us). If it keeps fetch fast -> compute-bound
        // ~29.3s -> ~678 @20k. Default 0 = original cudaEventSync behavior.
        static const int pace_sleep_us = []() {
            const char * e = std::getenv("LLAMA_CUDA_PACE_SLEEP_US");
            int n = (e && e[0]) ? atoi(e) : 0;
            return n > 0 ? n : 0;
        }();
        static std::atomic<uint64_t> pace_ctr{0};
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap_status);
        // In reuse mode the static event MUST be consumed (synced) before the next op
        // reuses it, so force per-op sync (ignore pace_every_n / sleep).
        const bool do_sync = (cap_status == cudaStreamCaptureStatusNone) &&
            (g_reuse_op_events ||
             ((pace_ctr.fetch_add(1, std::memory_order_relaxed) % (uint64_t)pace_every_n) == 0));
        if (do_sync) {
            if (pace_sleep_us > 0 && !g_reuse_op_events) {
                // Cheap CPU-yield pace: free the core for DStorage I/O threads WITHOUT
                // waiting the op's full GPU time. Skip the (now-invalid) elapsed read.
                std::this_thread::sleep_for(std::chrono::microseconds(pace_sleep_us));
            } else {
                cudaEventSynchronize(stop_ev);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, start_ev, stop_ev);
                g_op_profiles[op_index].add((double)ms);
                node_profile_add(node_key,(double)ms);
            }
        }
        // (If capturing: skip the sync + timing read. The events are still
        // recorded into the graph and will execute during replay; we just don't
        // get accurate per-op host timing for captured ops.)
        if (owns_events) {
            cudaEventDestroy(start_ev);
            cudaEventDestroy(stop_ev);
        }
    }
};
// ---------------------------------------------------------------------------
// End profiling infrastructure
// ---------------------------------------------------------------------------

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

// Phase 300 (2026-05-04): track all cudaMalloc calls through ggml_cuda_device_malloc.
// This catches both wrapper allocations AND pool's direct allocations,
// pinpointing exactly where VRAM is being consumed across cycles.
static std::atomic<int64_t> g_cuda_device_malloc_bytes{0};
static std::atomic<int64_t> g_cuda_device_malloc_count{0};
extern "C" int64_t ggml_cuda_device_malloc_total_bytes() { return g_cuda_device_malloc_bytes.load(); }
extern "C" int64_t ggml_cuda_device_malloc_total_count() { return g_cuda_device_malloc_count.load(); }

// DS4COMBO fast switch (2026-10-23): an optional process-wide switch arena.
// When enabled before the first device allocation, every ggml_cuda_device_malloc
// block gets its own VA range inside one cuMemAddressReserve span, backed by
// cuMemCreate/cuMemMap. suspend() unmaps and releases ALL physical memory of
// live blocks and registered VMM pools while every VA stays reserved; resume()
// creates, maps and zeroes the same ranges again, so device pointers held by
// buffers, gallocr and CUDA graphs stay valid. Contents are NOT preserved:
// callers re-import state after resume. Disabled by default: unchanged path.
namespace ds4_switch {
struct block { size_t size; CUmemGenericAllocationHandle handle; bool mapped; };
struct pool_range { CUdeviceptr * addr; size_t * size; int device; };
static std::mutex mutex;
static bool enabled = false, suspended = false;
static int arena_device = 0;
static CUdeviceptr base = 0;
static size_t reserved = 0, next = 0, granularity = 0;
static std::map<CUdeviceptr, block> blocks;
static std::vector<pool_range> pools;

static CUmemAllocationProp prop(int device) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = device;
    return p;
}
static bool map_range(CUdeviceptr va, size_t size, int device, CUmemGenericAllocationHandle * keep) {
    CUmemAllocationProp p = prop(device);
    CUmemGenericAllocationHandle h;
    if (cuMemCreate(&h, size, &p, 0) != CUDA_SUCCESS) return false;
    if (cuMemMap(va, size, 0, h, 0) != CUDA_SUCCESS) { cuMemRelease(h); return false; }
    CUmemAccessDesc a = {};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = device;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(va, size, &a, 1) != CUDA_SUCCESS) { cuMemUnmap(va, size); cuMemRelease(h); return false; }
    if (keep) *keep = h; else cuMemRelease(h);
    return true;
}
static void register_pool(CUdeviceptr * addr, size_t * size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    if (enabled) pools.push_back({addr, size, device});
}
static void unregister_pool(CUdeviceptr * addr) {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t i = 0; i < pools.size(); ++i) if (pools[i].addr == addr) { pools.erase(pools.begin() + i); return; }
}
static void require_resident(const char * what) {
    if (suspended) {
        fprintf(stderr, "DS4COMBO_SWITCH_ARENA_REFUSED operation=%s reason=suspended\n", what);
        fflush(stderr);
        GGML_ABORT("ggml CUDA switch arena is suspended; no device allocation is possible");
    }
}
static cudaError_t alloc(void ** ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    require_resident("alloc");
    if (device != arena_device) return cudaErrorInvalidDevice;
    const size_t rounded = granularity * ((size + granularity - 1) / granularity);
    if (rounded > reserved - next) {
        fprintf(stderr, "DS4COMBO_SWITCH_ARENA_EXHAUSTED request=%zu used_va=%zu reserved_va=%zu\n", rounded, next, reserved);
        fflush(stderr);
        return cudaErrorMemoryAllocation;
    }
    const CUdeviceptr va = base + next;
    block b = {rounded, 0, true};
    if (!map_range(va, rounded, device, &b.handle)) return cudaErrorMemoryAllocation;
    next += rounded;
    blocks[va] = b;
    *ptr = (void *) va;
    return cudaSuccess;
}
static bool owns(const void * ptr) {
    const CUdeviceptr p = (CUdeviceptr) ptr;
    return enabled && p >= base && p < base + reserved;
}
static cudaError_t free_block(void * ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = blocks.find((CUdeviceptr) ptr);
    if (it == blocks.end()) return cudaErrorInvalidDevicePointer;
    (void) cudaDeviceSynchronize();
    if (it->second.mapped) {
        if (cuMemUnmap(it->first, it->second.size) != CUDA_SUCCESS || cuMemRelease(it->second.handle) != CUDA_SUCCESS) return cudaErrorUnknown;
    }
    blocks.erase(it);   // the VA range is never reused within the reservation
    return cudaSuccess;
}
} // namespace ds4_switch

extern "C" bool ggml_backend_cuda_switch_arena_enable(int device, size_t va_bytes) {
    std::lock_guard<std::mutex> lock(ds4_switch::mutex);
    if (ds4_switch::enabled || g_cuda_device_malloc_count.load() != 0 || !va_bytes) return false;
    ggml_cuda_set_device(device);
    CUmemAllocationProp p = ds4_switch::prop(device);
    size_t g = 0;
    if (cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || !g) return false;
    const size_t bytes = g * ((va_bytes + g - 1) / g);
    if (cuMemAddressReserve(&ds4_switch::base, bytes, 0, 0, 0) != CUDA_SUCCESS) return false;
    ds4_switch::granularity = g; ds4_switch::reserved = bytes; ds4_switch::next = 0; ds4_switch::arena_device = device;
    ds4_switch::enabled = true; ds4_switch::suspended = false;
    fprintf(stderr, "DS4COMBO_SWITCH_ARENA_ENABLED device=%d va_bytes=%zu granularity=%zu base=0x%llx\n", device, bytes, g, (unsigned long long) ds4_switch::base);
    fflush(stderr);
    return true;
}

extern "C" bool ggml_backend_cuda_switch_arena_suspend(size_t * released_bytes, int * released_blocks) {
    std::lock_guard<std::mutex> lock(ds4_switch::mutex);
    if (!ds4_switch::enabled || ds4_switch::suspended) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    size_t bytes = 0; int count = 0;
    for (auto & kv : ds4_switch::blocks) {
        if (!kv.second.mapped) continue;
        if (cuMemUnmap(kv.first, kv.second.size) != CUDA_SUCCESS || cuMemRelease(kv.second.handle) != CUDA_SUCCESS) return false;
        kv.second.mapped = false; kv.second.handle = 0; bytes += kv.second.size; ++count;
    }
    for (auto & pool : ds4_switch::pools) {
        if (*pool.addr && *pool.size) {
            if (cuMemUnmap(*pool.addr, *pool.size) != CUDA_SUCCESS) return false;   // pool handles were released at map time
            bytes += *pool.size; ++count;
        }
    }
    ds4_switch::suspended = true;
    if (released_bytes) *released_bytes = bytes;
    if (released_blocks) *released_blocks = count;
    return true;
}

extern "C" bool ggml_backend_cuda_switch_arena_resume(size_t * mapped_bytes, int * mapped_blocks) {
    std::lock_guard<std::mutex> lock(ds4_switch::mutex);
    if (!ds4_switch::enabled || !ds4_switch::suspended) return false;
    ggml_cuda_set_device(ds4_switch::arena_device);
    size_t bytes = 0; int count = 0;
    for (auto & kv : ds4_switch::blocks) {
        if (!ds4_switch::map_range(kv.first, kv.second.size, ds4_switch::arena_device, &kv.second.handle)) return false;
        kv.second.mapped = true;
        if (cuMemsetD8(kv.first, 0, kv.second.size) != CUDA_SUCCESS) return false;
        bytes += kv.second.size; ++count;
    }
    for (auto & pool : ds4_switch::pools) {
        if (*pool.addr && *pool.size) {
            if (!ds4_switch::map_range(*pool.addr, *pool.size, pool.device, nullptr)) return false;
            if (cuMemsetD8(*pool.addr, 0, *pool.size) != CUDA_SUCCESS) return false;
            bytes += *pool.size; ++count;
        }
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) return false;
    ds4_switch::suspended = false;
    if (mapped_bytes) *mapped_bytes = bytes;
    if (mapped_blocks) *mapped_blocks = count;
    return true;
}

extern "C" bool ggml_backend_cuda_switch_arena_status(size_t * live_bytes, int * live_blocks, bool * is_suspended) {
    std::lock_guard<std::mutex> lock(ds4_switch::mutex);
    size_t bytes = 0; int count = 0;
    for (auto & kv : ds4_switch::blocks) { bytes += kv.second.size; ++count; }
    for (auto & pool : ds4_switch::pools) { bytes += *pool.size; ++count; }
    if (live_bytes) *live_bytes = bytes;
    if (live_blocks) *live_blocks = count;
    if (is_suspended) *is_suspended = ds4_switch::suspended;
    return ds4_switch::enabled;
}

// Frees a block that ggml_cuda_device_malloc returned, through the arena when it owns it.
static cudaError_t ggml_cuda_device_free_owned(void * ptr) {
    if (ds4_switch::owns(ptr)) return ds4_switch::free_block(ptr);
    return cudaFree(ptr);
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            CUDA_CHECK(cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else if (ds4_switch::enabled) {
        err = ds4_switch::alloc(ptr, size, device);
    } else {
        err = cudaMalloc(ptr, size);
    }
    if (err == cudaSuccess) {
        g_cuda_device_malloc_bytes.fetch_add((int64_t) size, std::memory_order_relaxed);
        g_cuda_device_malloc_count.fetch_add(1, std::memory_order_relaxed);
        const char * trace_env = std::getenv("LLAMA_CUDA_TRACE_DEVICE_MALLOC");
        if (trace_env && trace_env[0] != '\0' && trace_env[0] != '0' && size > 1024 * 1024) {
            fprintf(stderr, "*** DEV_MALLOC: size=%.2f MiB ptr=%p dev=%d (cum_count=%lld cum_bytes=%lld MB)\n",
                    size / 1024.0 / 1024.0, *ptr, device,
                    (long long) g_cuda_device_malloc_count.load(),
                    (long long)(g_cuda_device_malloc_bytes.load() / (1024 * 1024)));
            fflush(stderr);
        }
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)
#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(ggml_cuda_device_free_owned(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        // DIAG: log every cudaMalloc from this pool. Useful to see how many
        // fire per compute so we know if warm-up is stabilizing sizes.
        if (getenv("LLAMA_SPLIT_CAP_TRACE")) {
            fprintf(stderr, "[split_cap_trace] pool_alloc cudaMalloc(size=%zu, look_ahead=%zu)\n",
                    size, look_ahead_size);
            fflush(stderr);
        }
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_free_owned(ptr));
        pool_size -= size;
    }

    // Phase 296: release all cached freed buffers back to the CUDA driver.
    void clear() override {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(ggml_cuda_device_free_owned(b.ptr));
                pool_size -= b.size;
                b.ptr = nullptr;
                b.size = 0;
            }
        }
    }

    size_t pool_size_bytes() const override { return pool_size; }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        ds4_switch::unregister_pool(&pool_addr);
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            ds4_switch::require_resident("vmm_pool_grow");
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // DIAG: log every VMM pool growth so we can see if warm-up
            // has stabilized the pool for split-decode capture.
            if (getenv("LLAMA_SPLIT_CAP_TRACE")) {
                fprintf(stderr, "[split_cap_trace] vmm_pool grow: requested=%zu avail=%zu reserve=%zu (pool_size=%zu -> %zu)\n",
                        size, avail, reserve_size, pool_size, pool_size + reserve_size);
                fflush(stderr);
            }

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
                ds4_switch::register_pool(&pool_addr, &pool_size, device);
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            // Phase 80: cuMemSetAccess fails with CUDA_ERROR_NOT_READY on Windows
            // when the device has pending async operations from prior streams.
            // Synchronise the device first so the pool growth can complete.
            (void) cudaDeviceSynchronize();
            CU_CHECK(cuMemSetAccess((CUdeviceptr)((char *)(pool_addr) + pool_size), reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }

    // Phase 296: VMM pool. The reserve_size growth allocations stay mapped;
    // pool_used is the bump pointer. Resetting pool_used reclaims pool space
    // for new allocations without unmapping (which would invalidate descriptors).
    void clear() override {
        pool_used = 0;
    }

    size_t pool_size_bytes() const override { return pool_size; }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
    // Phase 80: env override to force the legacy pool. The VMM pool's
    // cuMemSetAccess fails with "device not ready" on Windows when growing
    // past the initial reserve, blocking dynamic n_ubatch resize. The
    // legacy pool uses cudaMalloc directly and has no growth restriction.
    const char * no_vmm = std::getenv("GGML_CUDA_NO_VMM");
    if (no_vmm && no_vmm[0] != '0' && no_vmm[0] != '\0') {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
    }
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (glm_topk_gather_masked_slots != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(glm_topk_gather_masked_slots));
        glm_topk_gather_masked_slots = nullptr;
        glm_topk_gather_mask_counter_active = false;
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }
}


// cuda buffer

// Phase 299 (2026-05-04): track per-process CUDA buffer alloc/free totals to
// surface leaks across streaming-prefill cycles. Wrapper-level counters are
// the only reliable signal because the leak hides in many sub-100MB allocs.
static std::atomic<int64_t> g_cuda_buf_alloc_bytes{0};
static std::atomic<int64_t> g_cuda_buf_free_bytes{0};
static std::atomic<int64_t> g_cuda_buf_live_count{0};

extern "C" int64_t ggml_backend_cuda_buffer_alloc_total() { return g_cuda_buf_alloc_bytes.load(); }
extern "C" int64_t ggml_backend_cuda_buffer_free_total()  { return g_cuda_buf_free_bytes.load(); }
extern "C" int64_t ggml_backend_cuda_buffer_live_count()  { return g_cuda_buf_live_count.load(); }

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    size_t alloc_size = 0;
    std::string name;
    // R3 dual-use: when false, the dtor does NOT cudaFree(dev_ptr) — the pointer
    // is owned externally (e.g. a D3D12-heap import via cudaExternalMemoryGetMappedBuffer)
    // and is permanent / freed by its real owner. Default true preserves all
    // existing cudaMalloc-backed buffers unchanged.
    bool owns_ptr = true;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr, size_t alloc_size = 0) :
        device(device), dev_ptr(dev_ptr), alloc_size(alloc_size),
        name(GGML_CUDA_NAME + std::to_string(device)) {
        g_cuda_buf_alloc_bytes.fetch_add((int64_t) alloc_size, std::memory_order_relaxed);
        g_cuda_buf_live_count.fetch_add(1, std::memory_order_relaxed);
    }

    ~ggml_backend_cuda_buffer_context() {
        if (owns_ptr) {
            CUDA_CHECK(ggml_cuda_device_free_owned(dev_ptr));
        }
        g_cuda_buf_free_bytes.fetch_add((int64_t) alloc_size, std::memory_order_relaxed);
        g_cuda_buf_live_count.fetch_sub(1, std::memory_order_relaxed);
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

// When the caller has an active cudaStreamBeginCapture, the per-thread default
// stream (cudaStreamPerThread) is NOT part of the capture — any async copy on
// it during capture raises cudaErrorInvalidArgument. Split-decode's graph-capture
// path (llama_split_graph_capture_layer) sets this TLS override to the capturing
// stream before running compute; buffer_set_tensor / buffer_get_tensor /
// buffer_memset_tensor then route their copies through that stream so they're
// captured too. Setting it back to nullptr restores default behavior.
thread_local cudaStream_t g_cuda_tensor_copy_stream_override = nullptr;

// Pinned-memory staging arena. CUDA also requires the host source of an async
// memcpy during graph capture to be PINNED. Pageable memory (std::vector etc)
// is rejected with cudaErrorInvalidArgument. When the stream override is
// active AND the src buffer isn't already pinned, we stage through this
// thread-local pinned arena so the async copy is captureable.
struct cuda_tensor_copy_pinned_arena {
    uint8_t * base = nullptr;
    size_t    capacity = 0;
    size_t    tail = 0;

    void ensure(size_t need) {
        if (tail + need <= capacity) return;
        // Grow: at least 1 MiB, or next power-of-two beyond tail+need.
        size_t new_cap = capacity == 0 ? (1 << 20) : capacity * 2;
        while (new_cap < tail + need) new_cap *= 2;
        uint8_t * new_base = nullptr;
        if (cudaHostAlloc((void **)&new_base, new_cap, cudaHostAllocDefault) != cudaSuccess) {
            // Alloc failed — caller falls back to error path (src stays pageable).
            return;
        }
        if (base) {
            if (tail > 0) memcpy(new_base, base, tail);
            cudaFreeHost(base);
        }
        base = new_base;
        capacity = new_cap;
    }

    uint8_t * stage(const void * src, size_t n) {
        // 16-byte align each chunk so tensor-reads don't cross cacheline awkward.
        tail = (tail + 15) & ~size_t(15);
        ensure(n);
        if (!base || tail + n > capacity) return nullptr;
        memcpy(base + tail, src, n);
        uint8_t * out = base + tail;
        tail += n;
        return out;
    }

    void reset() { tail = 0; }

    ~cuda_tensor_copy_pinned_arena() {
        if (base) cudaFreeHost(base);
    }
};
thread_local cuda_tensor_copy_pinned_arena g_cuda_tensor_copy_pinned;

// Per-capture arena staging log. Records `{tensor_name, arena_offset, size}`
// for every set_tensor that staged into the pinned arena during a capture
// window. Populated in ggml_backend_cuda_buffer_set_tensor when the override
// is active. Cleared on ggml_cuda_set_tensor_copy_stream(nullptr) alongside
// the arena reset. Used by split-decode Phase 2d.C1 to locate inp_pos /
// inp_kq_mask offsets inside the pinned arena after capture completes.
struct cuda_tensor_staging_record {
    std::string name;
    size_t      offset;
    size_t      size;
};
thread_local std::vector<cuda_tensor_staging_record> g_cuda_tensor_staging_log;

void ggml_cuda_set_tensor_copy_stream(void * stream) {
    g_cuda_tensor_copy_stream_override = (cudaStream_t)stream;
    // When turning OFF the override (end of capture), reset the arena so
    // subsequent captures start fresh. The pinned allocation stays.
    if (stream == nullptr) {
        g_cuda_tensor_copy_pinned.reset();
        g_cuda_tensor_staging_log.clear();
    } else {
        // Pre-allocate arena BEFORE capture begins. cudaHostAlloc during an
        // active stream capture is generally disallowed; calling ensure() here
        // forces any needed allocation to happen now.
        g_cuda_tensor_copy_pinned.ensure(1 << 20);  // 1 MiB initial
        g_cuda_tensor_staging_log.clear();
    }
}

void * ggml_cuda_get_tensor_copy_pinned_base(void) {
    return g_cuda_tensor_copy_pinned.base;
}

// Return the arena offset of the most-recent staging where tensor name CONTAINS
// `needle` as a substring, or SIZE_MAX if not found. The substring match handles
// the scheduler's wrapper names like "CUDA0#inp_pos#3" for cross-split inputs.
size_t ggml_cuda_get_last_staged_offset(const char * needle) {
    if (!needle) return (size_t)-1;
    for (auto it = g_cuda_tensor_staging_log.rbegin();
              it != g_cuda_tensor_staging_log.rend(); ++it) {
        if (it->name.find(needle) != std::string::npos) return it->offset;
    }
    return (size_t)-1;
}

// Variant that also returns the staged byte count via *out_size.
size_t ggml_cuda_get_last_staged_offset_size(const char * needle, size_t * out_size) {
    if (out_size) *out_size = 0;
    if (!needle) return (size_t)-1;
    for (auto it = g_cuda_tensor_staging_log.rbegin();
              it != g_cuda_tensor_staging_log.rend(); ++it) {
        if (it->name.find(needle) != std::string::npos) {
            if (out_size) *out_size = it->size;
            return it->offset;
        }
    }
    return (size_t)-1;
}

static inline cudaStream_t ggml_cuda_tensor_copy_stream() {
    return g_cuda_tensor_copy_stream_override ? g_cuda_tensor_copy_stream_override : cudaStreamPerThread;
}

// Returns true iff `p` is pinned/device/unified memory (captureable src).
static bool ggml_cuda_ptr_is_captureable(const void * p) {
    cudaPointerAttributes attr{};
    cudaError_t rc = cudaPointerGetAttributes(&attr, p);
    if (rc != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        return false;
    }
    // cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2, cudaMemoryTypeManaged = 3
    // cudaMemoryTypeUnregistered = 0 — pageable, NOT captureable.
    return attr.type != cudaMemoryTypeUnregistered;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    const cudaStream_t s = ggml_cuda_tensor_copy_stream();
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, s));
    if (s == cudaStreamPerThread) CUDA_CHECK(cudaStreamSynchronize(s));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    const cudaStream_t s = ggml_cuda_tensor_copy_stream();
    const void * src = data;
    // If we're routing through an override stream (stream capture active)
    // and the caller's src memory is pageable, cudaMemcpyAsync will reject
    // it with cudaErrorInvalidArgument. Stage through the thread-local
    // pinned arena so the captured memcpy reads from pinned memory.
    if (s != cudaStreamPerThread && !ggml_cuda_ptr_is_captureable(data)) {
        const uint8_t * staged = g_cuda_tensor_copy_pinned.stage(data, size);
        if (staged) {
            src = staged;
            // Record so split-decode can query arena offsets after capture.
            cuda_tensor_staging_record rec;
            rec.name   = tensor->name ? tensor->name : "";
            rec.offset = (size_t)(staged - g_cuda_tensor_copy_pinned.base);
            rec.size   = size;
            g_cuda_tensor_staging_log.push_back(std::move(rec));
        }
    }
    if (std::getenv("LLAMA_CUDA_TENSOR_SET_TRACE") != nullptr) {
        fprintf(stderr, "buffer_set_tensor: tensor='%s' size=%zu offset=%zu stream=%p "
                "src_orig=%p src_used=%p (staged=%d)\n",
                tensor->name ? tensor->name : "?", size, offset, (void *)s,
                data, src, (int)(src != data));
    }
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, src, size, cudaMemcpyHostToDevice, s));
    // Only sync if we're on the default per-thread stream. When an override
    // is active, the caller (e.g. stream capture) controls synchronization.
    if (s == cudaStreamPerThread) CUDA_CHECK(cudaStreamSynchronize(s));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    const cudaStream_t s = ggml_cuda_tensor_copy_stream();
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, s));
    if (s == cudaStreamPerThread) CUDA_CHECK(cudaStreamSynchronize(s));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (ggml_backend_buffer_is_cuda(src_buf)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src_buf->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst_buf->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    // Phase 298 (2026-05-04): log every CUDA buffer-type allocation > 100 MB
    // to identify per-request leaks. Gated on env to not spam regular runs.
    if (size > 100ull * 1024 * 1024) {
        const char * trace_env = std::getenv("LLAMA_CUDA_TRACE_BUFFER_ALLOC");
        if (trace_env && trace_env[0] != '\0' && trace_env[0] != '0') {
            size_t free_mb = 0, total_mb = 0;
            size_t free_b = 0, total_b = 0;
            (void) cudaMemGetInfo(&free_b, &total_b);
            free_mb = free_b / (1024 * 1024);
            total_mb = total_b / (1024 * 1024);
            fprintf(stderr, "*** ALLOC_TRACE: ggml_backend_cuda_buffer_type_alloc_buffer size=%.2f MiB ptr=%p (VRAM free=%zu/%zu MB)\n",
                    size / 1024.0 / 1024.0, dev_ptr, free_mb, total_mb);
            fflush(stderr);
        }
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr, size);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

// R3 dual-use ping-pong (USER design): wrap an externally-owned CUDA pointer
// (a contiguous view of a D3D12 heap imported via cudaImportExternalMemory +
// cudaExternalMemoryGetMappedBuffer) as a first-class ggml CUDA buffer. The
// returned buffer behaves exactly like a cudaMalloc-backed one for compute
// (get_base/set_tensor/cpy_tensor/clear all use ctx->dev_ptr) but its
// free_buffer will NOT cudaFree the pointer (owns_ptr=false) — the D3D12 heap
// is the real owner and is permanent across requests. This lets DirectStorage
// write weights DIRECTLY into the compute buffer (no staging, no D2D).
// Mirrors ggml_backend_cpu_buffer_from_ptr (ggml-backend.cpp) for the CUDA backend.
ggml_backend_buffer_t ggml_backend_cuda_buffer_from_ptr(int device, void * ptr, size_t size) {
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(device, ptr, size);
    ctx->owns_ptr = false;  // external D3D12-heap owner; never cudaFree here
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }

        const int cc = ggml_cuda_info().devices[id].cc;
        row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
    }
    return row_rounding;
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    int main_device;
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    std::string name;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(ggml_cuda_device_free_owned(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};


static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, size, id));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

static const char * ggml_backend_cuda_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::pair<int, std::array<float, GGML_CUDA_MAX_DEVICES>>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find({main_device, tensor_split_arr});
    if (it != buft_map.end()) {
        return &it->second;
    }
    auto * ctx = new ggml_backend_cuda_split_buffer_type_context{
        main_device,
        tensor_split_arr,
        GGML_CUDA_NAME + std::to_string(main_device) + "_Split",
    };

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;

    const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);

    const bool use_fp16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT;

    if (supports_bf16 && src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), ne, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff*src1_ncols, stream);
    } else if (fast_fp16_hardware_available(cc) && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f16, src0_ptr,      CUDA_R_16F, ne00,
                                    src1_ptr,      CUDA_R_16F, ne10,
                        &beta_f16,  dst_f16.get(), CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
            to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        }
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}

static void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);

        for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != main_device && id_other != main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
                    if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    } else {
                        // reset the error
                        (void)cudaGetLastError();
                    }
                } else {
                    cudaError_t err = cudaDeviceDisablePeerAccess(id_other);
                    if (err != cudaErrorPeerAccessNotEnabled) {
                        CUDA_CHECK(err);
                    } else {
                        // reset the error
                        (void)cudaGetLastError();
                    }
                }
            }
        }
    }

    ggml_cuda_set_device(main_device);
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;

    GGML_UNUSED(main_device);
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int64_t i02_divisor = ne12 / ne02;
    const int64_t i03_divisor = ne13 / ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));
    GGML_ASSERT(!(split && ne03 < ne13));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer.
            // This buffer needs to be cleared entirely because multiple regions will function as padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            const size_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(
                    dev[id].src1_ddf, nullptr, dev[id].src1_ddq, src0->type, ne10,
                    nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                    src1_padded_col_size, ne11, ne12, ne13, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                const size_t nbytes_src0_matrix = ne01*ne00*src0_ts / src0_bs;
                char  *  src0_dd_i =  dev[id].src0_dd + ((i03/i03_divisor)*ne02 + (i02/i02_divisor)) * nbytes_src0_matrix;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(
                        src1_ddf_i, nullptr, src1_ddq_i, src0->type, ne10, ne10, ne11*ne10, ne12*ne11*ne10,
                        src1_padded_col_size, src1_ncols, 1, 1, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i03 % i03_divisor == 0 && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src0_dd_i, src0, i03/i03_divisor, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
                        CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                            dhf_dst_i, ctx.device, ne0*sizeof(float), dst_dd_i, id, row_diff*sizeof(float), row_diff*sizeof(float), src1_ncols, stream));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        cu_compute_type = CUBLAS_COMPUTE_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        const int64_t smb = ne12 == 1 ? s13       : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (glu->src[0] != ffn_gate_bias || glu->src[1] != ffn_up_bias) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up || ffn_up_bias->src[1] == ffn_up;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate || ffn_gate_bias->src[1] == ffn_gate;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up || ffn_gate_bias->src[0] != ffn_gate) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    } else {
        if (glu->src[0] != ffn_gate && glu->src[1] != ffn_up) {
            return false;
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (ffn_up->src[2] && (ffn_up->src[2] != ffn_gate->src[2])) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 3> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    const bool split = ggml_backend_buft_is_cuda_split(ffn_up->src[0]->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(ffn_gate->src[0]->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    if (ggml_ds4_native_half_requested(tensor)) return false;
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    if (ggml_ds4_native_half_requested(tensor)) return false;
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return use_mul_mat_vec_q;
}

#include "ggml-dsv41-split-f32.h"
#include "dsv41-split-f32.cuh"
static std::atomic<uint64_t> dsv41_split_f32_queued{0};
void ggml_cuda_dsv41_split_f32_record_queued(){dsv41_split_f32_queued.fetch_add(1,std::memory_order_release);}
extern "C" uint64_t ggml_backend_cuda_dsv41_split_f32_queued(void){return dsv41_split_f32_queued.load(std::memory_order_acquire);}
static std::atomic<uint64_t> dsv41_dense_half_queued{0};
static std::atomic<uint64_t> dsv41_hc_half_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_hc_half_queued(void){return dsv41_hc_half_queued.load(std::memory_order_acquire);}
extern "C" uint64_t ggml_backend_cuda_dsv41_dense_half_queued(void){return dsv41_dense_half_queued.load(std::memory_order_acquire);}
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if(ggml_dsv41_split_f32_requested(dst)){
        GGML_ASSERT(ggml_dsv41_split_f32_shape(dst)&&src0==dst->src[0]&&src1==dst->src[1]);
        GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200);
        const int status=ggml_cuda_dsv41_split_f32(1,int(src0->ne[1]),int(src1->ne[1]),int(src0->ne[0]),
            (const float*)src0->data,int(src0->nb[1]/4),int64_t(src0->nb[2]/4),
            (const float*)src1->data,int(src1->nb[1]/4),int64_t(src1->nb[2]/4),
            (float*)dst->data,int(dst->ne[0]),int64_t(dst->nb[2]/4),int(dst->ne[2]),ctx.stream());
        GGML_ASSERT(status==0);CUDA_CHECK(cudaGetLastError());
        dsv41_split_f32_queued.fetch_add(1,std::memory_order_release);return;
    }
    if(ggml_dsv41_hc_half_requested(dst)){
        GGML_ASSERT(ggml_dsv41_hc_half_valid(dst)&&src0==dst->src[0]&&src1==dst->src[1]);
        GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200);
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx,src0,src1,dst);
        dsv41_hc_half_queued.fetch_add(1,std::memory_order_release);return;
    }
    if(ggml_dsv41_dense_half_requested(dst)){
        GGML_ASSERT(ggml_dsv41_dense_half_shape(dst)&&src0==dst->src[0]&&src1==dst->src[1]);
        GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200);
        // Reuse the existing F16 A/B, COMPUTE_32F and F32-output path.
        // This explicit tag does not affect HC, router or MUL_MAT_ID dispatch.
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx,src0,src1,dst);
        dsv41_dense_half_queued.fetch_add(1,std::memory_order_release);return;
    }
    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_mul_mat_vec_f = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const bool force_cublas_prec =
        ggml_is_quantized(src0->type) &&
        ggml_get_op_params_i32(dst, 0) == GGML_PREC_F32;
    if (force_cublas_prec) {
        use_mul_mat_vec_q = false;
        use_mul_mat_q     = false;
    }

    bool any_gpus_with_slow_fp16 = false;

    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc            = ggml_cuda_info().devices[id].cc;
            const int warp_size     = ggml_cuda_info().devices[id].warp_size;
            use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
            use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
            use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
        }
    } else {
        const int cc            = ggml_cuda_info().devices[ctx.device].cc;
        const int warp_size     = ggml_cuda_info().devices[ctx.device].warp_size;
        use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
        use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
        use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (std::getenv("LLAMA_CUDA_MMQ_TRACE") && (use_mul_mat_q || use_mul_mat_vec_q)) {
        fprintf(stderr,
            "[mmq-trace] dst=%s src0=%s type=%s src0_ne=[%lld,%lld,%lld,%lld] src1=%s type=%s src1_ne=[%lld,%lld,%lld,%lld] dst_ne=[%lld,%lld,%lld,%lld] vec=%d mmq=%d prec=%d\n",
            dst->name, src0->name, ggml_type_name(src0->type),
            (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
            src1->name, ggml_type_name(src1->type),
            (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
            (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
            use_mul_mat_vec_q ? 1 : 0, use_mul_mat_q ? 1 : 0, ggml_get_op_params_i32(dst, 0));
        fflush(stderr);
    }

    //TODO update for generic tensor parallelism
    const int cc                 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16  = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32  = src0->type == GGML_TYPE_F32;

    if (!split && use_mul_mat_vec_f) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_f) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_vec_q) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_q) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec_f) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_f, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

#include "ds4split-mmid-observer.cuh"
#include "ggml-dsv41-shared-sparse.h"
#include "dsv41-ds4-token-tile.cuh"
#include "ggml-dsv41-sparse-replay.h"
#include "ds4split-expert-projection.cuh"

static ds4split_native_half_witness::counters ds4_native_half_actual;
extern "C" bool ggml_backend_cuda_ds4_native_half_begin(ggml_ds4_native_half_stats * out){return out&&ds4_native_half_actual.begin(*out);}
extern "C" bool ggml_backend_cuda_ds4_native_half_complete(const ggml_ds4_native_half_stats * before,ggml_ds4_native_half_stats * after){return before&&after&&ds4_native_half_actual.complete(*before,*after);}
extern "C" void ggml_backend_cuda_ds4_native_half_snapshot(ggml_ds4_native_half_stats * out){GGML_ASSERT(out);ds4_native_half_actual.snapshot(*out);}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if(ggml_dsv41_native_fp4_requested(dst)){
        GGML_ASSERT(ggml_dsv41_native_fp4_valid(dst)&&blackwell_mma_available(ggml_cuda_info().devices[ctx.device].cc));
        // Existing generic routed MMQ includes native FP4 activation conversion
        // and Blackwell FP4 MMA. The preferred mmq_id path uses Q8_1 instead.
        ggml_cuda_mul_mat_q(ctx,dst->src[0],dst->src[1],dst->src[2],dst);return;
    }
    if(ggml_dsv41_expert_cublas_requested(dst)){ggml_cuda_op_dsv41_expert_cublas(ctx,dst);return;}
    // Explicit graph-native-half nodes never enter TEST projection/hash observers.
    // All untagged nodes continue through the original code below unchanged.
    if (ggml_ds4_native_half_requested(dst)) {
        const int cc=ggml_cuda_info().devices[ctx.device].cc;
        GGML_ASSERT(ggml_ds4_native_half_shape_valid(dst) && cc==1200);
        for (int i=0;i<3;++i) {
            const auto * source=dst->src[i];
            GGML_ASSERT(source->buffer && ggml_backend_buft_is_cuda(source->buffer->buft) &&
                !ggml_backend_buft_is_cuda_split(source->buffer->buft));
            const auto * placement=static_cast<ggml_backend_cuda_buffer_type_context *>(source->buffer->buft->context);
            GGML_ASSERT(placement->device==ctx.device);
        }
        GGML_ASSERT(dst->buffer && ggml_backend_buft_is_cuda(dst->buffer->buft));
        const auto * destination_placement=static_cast<ggml_backend_cuda_buffer_type_context *>(dst->buffer->buft->context);
        GGML_ASSERT(destination_placement->device==ctx.device);
        GGML_ASSERT(ggml_cuda_supports_q4k_half_mmid(dst->src[0],dst->src[1],dst->src[2],dst,cc));
        GGML_ASSERT(ds4_native_half_actual.active.load(std::memory_order_acquire));
        ds4split_q4k_half_launch actual{};
        ggml_cuda_mul_mat_q4k_half_id(ctx,dst->src[0],dst->src[1],dst->src[2],dst,&actual);
        bool first=false;GGML_ASSERT(ds4_native_half_actual.queue(dst->op_params[3],first));
        if(first)fprintf(stderr,"DS4_NATIVE_HALF_FIRST_QUEUED slot=%d family=%d k=%lld n=%lld tokens=%lld warps=%d task_bound=%lld workspace=%llu completion=queued_only\n",
            dst->op_params[3],actual.kernel_family,(long long)dst->src[0]->ne[0],(long long)dst->ne[0],(long long)actual.tokens,actual.nwarps,(long long)actual.maximum_tasks,(unsigned long long)actual.workspace_requested_bytes);
        return;
    }
    ds4split_projection_candidate::call_scope projection(ctx, dst);
    if (projection.run()) return;
    ds4split_projection_candidate::finish_after_original_scope([&]() {
    ds4split_mmid::call_scope mmid_capture(ctx, dst);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) && "mul_mat_id does not support split buffers");

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // bug#2 stream_local (2026-07-18): ggml_cuda_mul_mat_id, unlike ggml_cuda_mul_mat, did NOT honor the
    // GGML_PREC_F32 op-param, so ggml_mul_mat_set_prec() on an expert matmul was a no-op. At decode
    // (ne2<=4) quantized experts route to MMVQ, at prefill to MMQ/mmq_id - different quantized kernels.
    // Mirror ggml_cuda_mul_mat's force_cublas_prec: when GGML_PREC_F32 is requested on a quantized expert
    // matmul, SKIP all quantized kernels (MMVQ/MMQ/mmq_id/mmf) and fall through to the dequant->F32 cuBLAS
    // path, so a DSV4 nt==1 decode (LLAMA_DSV4_DECODE_F32_MOE=1) can bypass the single-column MMVQ kernel.
    const bool force_cublas_prec_id =
        ggml_is_quantized(src0->type) && ggml_get_op_params_i32(dst, 0) == GGML_PREC_F32;
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && !force_cublas_prec_id) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                if (ne2 <= 4) {
                    if (g_op_profile_enabled) g_mmid_route_mmvq.fetch_add(1, std::memory_order_relaxed);
                    mmid_capture.route("mmvq");
                    projection.route("mmvq");
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    mmid_capture.finish();
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    if (g_op_profile_enabled) g_mmid_route_mmvf.fetch_add(1, std::memory_order_relaxed);
                    mmid_capture.route("mmvf");
                    projection.route("mmvf");
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    mmid_capture.finish();
                    return;
                }
            }
        }

        if (ggml_cuda_can_use_mmq_id(src0->type, cc, ne11)) {
            cudaStream_t stream = ctx.stream();
            const int64_t n_expert_used = ids->ne[0];
            const int64_t ne_get_rows = ne12 * n_expert_used;
            const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
            const size_t nbytes_ids = (2*ne_get_rows + ne02 + 1) * sizeof(int32_t);
            const size_t nbytes_src1_q8_1 = ne_get_rows*ne10_padded * sizeof(block_q8_1)/QK8_1
                + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);

            ggml_cuda_pool_alloc<char> ids_buf(ctx.pool());
            const bool reuse_route=ggml_dsv41_routing_map_reuse_requested(dst);
            char*route_data=reuse_route?ggml_cuda_dsv41_routing_map(ctx,dst):ids_buf.alloc(nbytes_ids);
            ggml_cuda_pool_alloc<char> src1q_buf(ctx.pool(), nbytes_src1_q8_1);

            int32_t * ids_src1 = reinterpret_cast<int32_t *>(route_data);
            int32_t * ids_dst = ids_src1 + ne_get_rows;
            int32_t * expert_bounds = ids_dst + ne_get_rows;

            if(reuse_route){}else if(ggml_dsv41_sorted_routing_requested(dst)){
                compute_row_ids_count_prefix_scatter(ctx,(const int32_t *)ids->data,ids_src1,ids_dst,expert_bounds,
                    ne02,ne12,n_expert_used,ne11,nb11,nb12,ids->nb[1],stream);
            }else compute_row_ids((const int32_t *) ids->data, ids_src1, ids_dst, expert_bounds,
                ne02, ne12, n_expert_used, ne11, nb11, nb12, ids->nb[1], stream);

            const size_t ts_src1 = ggml_type_size(src1->type);
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            quantize_mmq_q8_1_cuda_id((const float *) src1->data, ids_src1, src1q_buf.get(), src0->type,
                ne10, s11, s12, s13, ne10_padded, ne_get_rows, 1, 1, stream);
            CUDA_CHECK(cudaGetLastError());

            if (g_op_profile_enabled) g_mmid_route_mmq_id.fetch_add(1, std::memory_order_relaxed);
            mmid_capture.route("mmq_id");
            projection.route("mmq_id");
            mmid_capture.packed(route_data, nbytes_ids, src1q_buf.get(),
                size_t(ne_get_rows)*size_t(ne10_padded)*sizeof(block_q8_1)/QK8_1,
                nbytes_src1_q8_1, ne10_padded);
            ggml_cuda_mul_mat_q_id(ctx, src0, src1, ids, dst, route_data, src1q_buf.get());
            mmid_capture.finish();
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            if (g_op_profile_enabled) g_mmid_route_mmq.fetch_add(1, std::memory_order_relaxed);
            mmid_capture.route("mmq");
            projection.route("mmq");
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            mmid_capture.finish();
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            if (g_op_profile_enabled) g_mmid_route_mmf.fetch_add(1, std::memory_order_relaxed);
            mmid_capture.route("mmf");
            projection.route("mmf");
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            mmid_capture.finish();
            return;
        }
    }
    if (g_op_profile_enabled) g_mmid_route_cublas.fetch_add(1, std::memory_order_relaxed);
    mmid_capture.route("sort_scatter_fallback_not_proof_of_cublas");
    projection.route("sort_scatter_fallback_not_proof_of_cublas");

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    // CPU-side expert ID sorting (ported from upstream llama.cpp).
    // Builds ids_to_sorted and ids_from_sorted mappings on host, copies to device.
    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                }
            }
        }
    }
    if (ids_to_sorted_host.size() != size_t(ne_get_rows)) {
        fprintf(stderr, "*** MUL_MAT_ID: ids_to_sorted=%zu ne_get_rows=%lld ne02=%lld ne12=%lld n_expert_used=%lld ne11=%lld\n",
                ids_to_sorted_host.size(), (long long)ne_get_rows, (long long)ne02, (long long)ne12,
                (long long)n_expert_used, (long long)ne11);
        GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));
    }

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
    mmid_capture.finish();
    }, [&]() { projection.finish(); });
}

// [DSA_KIDX_STORE_DIAG] The GLM-DSA indexer-K store is where the LLAMA_PREFILL_KV_RAM=0 prefill dies:
//     ggml_cuda_compute_forward: SET_ROWS failed dst='indexer_k_store-0'
//         src0='indexer_k_full-0 (view)' src1='CUDA0#k_idxs#0'
// SET_ROWS scatters src0's rows into dst at the row indices held in src1, so an illegal access is
// either an OUT-OF-RANGE index or a bad src0/dst mapping. The indices must be read BEFORE the launch:
// once the IMA fires the CUDA context is poisoned and any later copy fails or returns garbage.
// Env-gated (LLAMA_DSA_KIDX_STORE_DIAG=1), name-filtered, and capped, so it costs one predictable
// branch when off and cannot flood a 78-layer x 7-ubatch prefill when on.
static void ggml_cuda_diag_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    static const bool diag_on = []() {
        const char * e = std::getenv("LLAMA_DSA_KIDX_STORE_DIAG");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    if (!diag_on || dst == nullptr) {
        return;
    }
    const char * name = ggml_get_name(dst);
    if (name == nullptr || strncmp(name, "indexer_k_store", 15) != 0) {
        return;
    }
    static int printed = 0;
    if (printed >= 8) {
        return;
    }
    ++printed;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    long long idx_min = 0, idx_max = -1;
    size_t n_idx = (src1 != nullptr) ? (size_t) ggml_nelements(src1) : 0;
    bool idx_read = false;
    if (src1 != nullptr && src1->type == GGML_TYPE_I64 && n_idx > 0 && src1->data != nullptr) {
        std::vector<int64_t> host(n_idx);
        if (cudaMemcpyAsync(host.data(), src1->data, n_idx * sizeof(int64_t),
                            cudaMemcpyDeviceToHost, ctx.stream()) == cudaSuccess &&
            cudaStreamSynchronize(ctx.stream()) == cudaSuccess) {
            idx_min = idx_max = host[0];
            for (size_t i = 1; i < n_idx; ++i) {
                if (host[i] < idx_min) idx_min = host[i];
                if (host[i] > idx_max) idx_max = host[i];
            }
            idx_read = true;
        }
    }

    const long long dst_rows = (long long) dst->ne[1];
    fprintf(stderr,
            "*** DSA_KIDX_STORE_DIAG: dst='%s' buf=%s data=%p ne=[%lld,%lld,%lld,%lld] nb1=%zu bytes=%zu "
            "| src0='%s' ne=[%lld,%lld,%lld,%lld] cont=%d "
            "| src1='%s' type=%d n_idx=%zu read=%d idx_min=%lld idx_max=%lld "
            "| IN_RANGE=%d\n",
            name,
            (dst->buffer != nullptr) ? ggml_backend_buffer_name(dst->buffer) : "(null)",
            dst->data,
            (long long) dst->ne[0], dst_rows, (long long) dst->ne[2], (long long) dst->ne[3],
            dst->nb[1], ggml_nbytes(dst),
            (src0 != nullptr && ggml_get_name(src0) != nullptr) ? ggml_get_name(src0) : "(null)",
            (src0 != nullptr) ? (long long) src0->ne[0] : -1,
            (src0 != nullptr) ? (long long) src0->ne[1] : -1,
            (src0 != nullptr) ? (long long) src0->ne[2] : -1,
            (src0 != nullptr) ? (long long) src0->ne[3] : -1,
            (src0 != nullptr) ? (int) ggml_is_contiguous(src0) : -1,
            (src1 != nullptr && ggml_get_name(src1) != nullptr) ? ggml_get_name(src1) : "(null)",
            (src1 != nullptr) ? (int) src1->type : -1,
            n_idx, (int) idx_read, idx_min, idx_max,
            (idx_read && idx_max < dst_rows && idx_min >= 0) ? 1 : 0);
    fflush(stderr);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // why is this here instead of mul_mat?
    if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->buft)) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    // Per-op timing: optionally time every dispatched op.
    // CudaOpTimer RAII records start on construction, stop+sync on destruction.
    // timer_ptr is null when profiling is disabled — zero overhead on the hot path.
    std::unique_ptr<CudaOpTimer> timer_ptr;
    if (g_op_profile_enabled && dst->op < GGML_OP_COUNT) {
        timer_ptr = std::make_unique<CudaOpTimer>(ctx.stream(), (int)dst->op,dst);
    }
    CudaDlmRemainderTimer remainder_timer(ctx.stream(), (int) dst->op);

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_diag_set_rows(ctx, dst);
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_op_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_MAP_CUSTOM3:
            if(!ggml_dsv41_hc_expand_requested(dst))return false;
            ggml_cuda_op_dsv41_ds4_hc_expand(ctx,dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_DSV4_ACT_QAT:
            ggml_cuda_op_dsv4_act_qat(ctx, dst);
            break;
        case GGML_OP_DSV41_ENGRAM_ADD:
            ggml_cuda_op_dsv41_engram_add(ctx, dst);
            break;
        case GGML_OP_DSV41_POOL2:
            ggml_cuda_op_dsv41_pool2(ctx, dst);
            break;
        case GGML_OP_DSV41_ROPE:
            ggml_cuda_op_dsv41_rope(ctx, dst);
            break;
        case GGML_OP_DSV41_RAW_COMMIT:
            ggml_cuda_op_dsv41_raw_commit(ctx, dst);
            break;
        case GGML_OP_DSV41_INDEX_SCORE:
            if(ggml_dsv41_index_cublas_requested(dst))ggml_cuda_op_dsv41_index_score_cublas(ctx,dst);
            else ggml_cuda_op_dsv41_index_score(ctx, dst);
            break;
        case GGML_OP_DSV41_CANDIDATE_BLOCKS:
            ggml_cuda_op_dsv41_candidate_blocks(ctx, dst);
            break;
        case GGML_OP_DSV41_SELECTED_MASK:
            ggml_cuda_op_dsv41_selected_mask(ctx, dst);
            break;
        case GGML_OP_DSV41_MASKED_HALF:
            ggml_cuda_op_dsv41_masked_half(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT_SPARSE:
            if(ggml_dsv41_token_tile_requested(dst)){
                ggml_cuda_op_dsv41_ds4_token_tile(ctx,dst);break;
            }
            if(ggml_dsv41_shared_sparse_requested(dst)){
                GGML_ASSERT(ggml_dsv41_shared_sparse_valid(dst));
                GGML_ASSERT(ggml_dsv41_cuda_sparse_f32_replay(
                    (const float*)dst->src[0]->data,(const float*)dst->src[1]->data,
                    (const int32_t*)dst->src[3]->data,(const float*)dst->src[4]->data,
                    (const float*)dst->src[5]->data,(float*)dst->data,
                    int(dst->ne[2]),int(dst->src[1]->ne[2]),int(dst->src[3]->ne[0]),int(dst->ne[1]),ctx.stream()));
                break;
            }
            ggml_cuda_op_flash_attn_ext_sparse(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_DSA_INDEXER_SCORE:
            ggml_cuda_op_dsa_indexer_score(ctx, dst);
            break;
        case GGML_OP_DSA_INDEXER_TOPK:
            ggml_cuda_op_dsa_indexer_topk(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        GGML_LOG_ERROR("%s: %s failed dst='%s' src0='%s' src1='%s'\n",
                __func__,
                ggml_op_desc(dst),
                ggml_get_name(dst),
                src0 ? ggml_get_name(src0) : "<null>",
                src1 ? ggml_get_name(src1) : "<null>");
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    // cudaStreamSynchronize on a stream that's currently capturing raises
    // "operation not permitted when stream is capturing". During capture
    // the semantics of "synchronize" don't apply — work is being recorded,
    // not executed. Skip the call in that case; ggml's callers of
    // synchronize() during capture are redundant (everything's already
    // ordered via the capture's dependency graph).
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cuda_ctx->stream(), &cap_status) == cudaSuccess
        && cap_status == cudaStreamCaptureStatusActive) {
        return;
    }

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";
    const std::string nemotron_h_block_out_prefix = "nemotron_h_block_out";
    const std::string mamba2_y_add_d_prefix = "mamba2_y_add_d";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
            use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
        }

        if (ggml_ds4_native_half_requested(node) || (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] != 1)) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
        }

        if (node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&
            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
            strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(), ffn_moe_gate_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_up_bias_prefix.c_str(), ffn_moe_up_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_down_bias_prefix.c_str(), ffn_moe_down_bias_prefix.size()) != 0 &&
            strncmp(node->name, nemotron_h_block_out_prefix.c_str(), nemotron_h_block_out_prefix.size()) != 0 &&
            strncmp(node->name, mamba2_y_add_d_prefix.c_str(), mamba2_y_add_d_prefix.size()) != 0) {
            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
            // by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static void ggml_cuda_graph_node_set_properties(ggml_cuda_graph_node_properties * props, ggml_tensor * node) {
    memset(props, 0, sizeof(ggml_cuda_graph_node_properties));
    props->node_data = node->data;
    props->node_op = node->op;
    props->node_type = node->type;
    props->flags = node->flags;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        props->ne[i] = node->ne[i];
        props->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!node->src[i]) {
            continue;
        }

        props->src_data[i] = node->src[i]->data;
    }
    memcpy(props->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_cuda_graph_node_properties_match(ggml_tensor * node, ggml_cuda_graph_node_properties * props) {
    if (node->data != props->node_data && node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != props->node_op) {
        return false;
    }

    if (node->type != props->node_type) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != props->ne[i]) {
            return false;
        }
        if (node->nb[i] != props->nb[i]) {
            return false;
        }
    }

    if (node->op != GGML_OP_VIEW) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (!node->src[i]) {
                if (props->src_data[i] != nullptr) {
                    return false;
                }
                continue;
            }

            if (node->src[i]->data != props->src_data[i]) {
                return false;
            }
        }
    }

    if (memcmp(props->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) != (props->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return false;
    }

    return true;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->instance == nullptr) {
        res = true;
    }

    // Check if the graph size has changed
    if (graph->props.size() != (size_t)cgraph->n_nodes) {
        res = true;
        graph->props.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    std::unordered_set<ggml_tensor *> seen_node;
    std::vector<ggml_tensor *> srcs_extra;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool props_match = true;

        seen_node.insert(cgraph->nodes[i]);

        if (!res) {
            props_match = ggml_cuda_graph_node_properties_match(cgraph->nodes[i], &graph->props[i]);
        }
        if (!props_match) {
            res = true;
        }
        ggml_cuda_graph_node_set_properties(&graph->props[i], cgraph->nodes[i]);

        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            ggml_tensor * src = cgraph->nodes[i]->src[src_idx];
            if (src && seen_node.find(src) == seen_node.end()) {
                srcs_extra.push_back(src);
            }
        }
    }

    if (graph->extra.size() != (size_t) srcs_extra.size()) {
        res = true;
        graph->extra.resize(srcs_extra.size());
    }

    for (size_t i = 0; i < srcs_extra.size(); ++i) {
        bool props_match = true;

        if (!res) {
            props_match = ggml_cuda_graph_node_properties_match(srcs_extra[i], &graph->extra[i]);
        }

        if (!props_match) {
            res = true;
        }
        ggml_cuda_graph_node_set_properties(&graph->extra[i], srcs_extra[i]);
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_can_fuse_range(const struct ggml_cgraph * cgraph, int start_idx, int end_idx, std::initializer_list<int> outputs) {
    const int count = end_idx - start_idx + 1;
    if (count <= 0 || count >= 32 || end_idx >= cgraph->n_nodes) {
        return false;
    }

    for(int ds4_i=start_idx;ds4_i<=end_idx;++ds4_i)
        if(ds4_i>=0&&ggml_ds4_native_half_requested(cgraph->nodes[ds4_i]))return false;

    int idxs[32];
    enum ggml_op ops[32];
    int out_nodes[4];
    int n_out_nodes = 0;

    for (int i = 0; i < count; ++i) {
        idxs[i] = start_idx + i;
        ops[i]  = cgraph->nodes[start_idx + i]->op;
    }

    for (const int output : outputs) {
        if (n_out_nodes >= 4) {
            return false;
        }
        out_nodes[n_out_nodes++] = output;
    }

    return ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, n_out_nodes);
}

static bool ggml_cuda_match_bailingmoev2(const struct ggml_cgraph * cgraph, int node_idx, int & topk_view_idx, int & weights_idx) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  node  = nodes[node_idx];

    if (node->op != GGML_OP_UNARY || ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    const int limit = std::min(cgraph->n_nodes, node_idx + 24);
    for (int idx = node_idx + 1; idx < limit; ++idx) {
        ggml_tensor * topk_view = nodes[idx];
        if (topk_view->op != GGML_OP_VIEW) {
            continue;
        }

        ggml_tensor * topk = topk_view->src[0];
        if (!topk || topk->op != GGML_OP_ARGSORT) {
            continue;
        }

        ggml_tensor * masked_probs = topk->src[0];
        if (!masked_probs || masked_probs->op != GGML_OP_RESHAPE) {
            continue;
        }

        ggml_tensor * set_rows = masked_probs->src[0];
        if (!set_rows || set_rows->op != GGML_OP_SET_ROWS) {
            continue;
        }

        ggml_tensor * fill = set_rows->src[2];
        if (!fill || fill->op != GGML_OP_FILL) {
            continue;
        }

        ggml_tensor * selection_groups = fill->src[0];
        if (!selection_groups || selection_groups->op != GGML_OP_RESHAPE) {
            continue;
        }

        ggml_tensor * selection_input = selection_groups->src[0];
        if (!selection_input || selection_input->op != GGML_OP_ADD || selection_input->src[0] != node) {
            continue;
        }

        ggml_tensor * expert_groups_view = set_rows->src[1];
        if (!expert_groups_view || expert_groups_view->op != GGML_OP_VIEW) {
            continue;
        }

        ggml_tensor * expert_groups = expert_groups_view->src[0];
        if (!expert_groups || expert_groups->op != GGML_OP_ARGSORT) {
            continue;
        }

        int candidate_weights_idx = -1;
        const int weights_limit = std::min(cgraph->n_nodes, idx + 4);
        for (int w = idx + 1; w < weights_limit; ++w) {
            ggml_tensor * weights = nodes[w];
            if (weights->op != GGML_OP_GET_ROWS || weights->src[1] != topk_view) {
                continue;
            }

            ggml_tensor * probs_reshaped = weights->src[0];
            if (!probs_reshaped || probs_reshaped->op != GGML_OP_RESHAPE || probs_reshaped->src[0] != node) {
                continue;
            }

            candidate_weights_idx = w;
            break;
        }

        if (candidate_weights_idx < 0) {
            continue;
        }

        if (!ggml_cuda_can_fuse_range(cgraph, node_idx, candidate_weights_idx, { idx, candidate_weights_idx })) {
            continue;
        }

        topk_view_idx = idx;
        weights_idx   = candidate_weights_idx;
        return true;
    }

    return false;
}

static bool ggml_cuda_match_glm45moe(const struct ggml_cgraph * cgraph, int node_idx, int & topk_view_idx, int & weights_idx) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  node  = nodes[node_idx];

    if (node->op != GGML_OP_UNARY || ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    const int limit = std::min(cgraph->n_nodes, node_idx + 8);
    for (int idx = node_idx + 1; idx < limit; ++idx) {
        ggml_tensor * topk_view = nodes[idx];
        if (topk_view->op != GGML_OP_VIEW) {
            continue;
        }

        ggml_tensor * topk = topk_view->src[0];
        if (!topk || topk->op != GGML_OP_ARGSORT) {
            continue;
        }

        ggml_tensor * add = topk->src[0];
        if (!add || add->op != GGML_OP_ADD || add->src[0] != node) {
            continue;
        }

        if (idx + 1 >= cgraph->n_nodes) {
            continue;
        }

        ggml_tensor * weights = nodes[idx + 1];
        if (weights->op != GGML_OP_GET_ROWS || weights->src[1] != topk_view) {
            continue;
        }

        ggml_tensor * probs_reshaped = weights->src[0];
        if (!probs_reshaped || probs_reshaped->op != GGML_OP_RESHAPE || probs_reshaped->src[0] != node) {
            continue;
        }

        if (!ggml_cuda_can_fuse_range(cgraph, node_idx, idx + 1, { idx, idx + 1 })) {
            continue;
        }

        topk_view_idx = idx;
        weights_idx   = idx + 1;
        return true;
    }

    return false;
}

static bool ggml_cuda_match_openai_experts(const struct ggml_cgraph * cgraph, int node_idx, int & topk_view_idx, int & softmax_idx, int & weights_idx) {
    if (node_idx + 5 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes;

    ggml_tensor * topk      = nodes[node_idx];
    ggml_tensor * topk_view = nodes[node_idx + 1];
    ggml_tensor * get_rows  = nodes[node_idx + 2];
    ggml_tensor * reshape   = nodes[node_idx + 3];
    ggml_tensor * softmax   = nodes[node_idx + 4];
    ggml_tensor * weights   = nodes[node_idx + 5];

    if (topk->op != GGML_OP_ARGSORT) {
        return false;
    }

    if (topk_view->op != GGML_OP_VIEW || topk_view->src[0] != topk) {
        return false;
    }

    if (get_rows->op != GGML_OP_GET_ROWS || get_rows->src[1] != topk_view) {
        return false;
    }

    if (reshape->op != GGML_OP_RESHAPE || reshape->src[0] != get_rows) {
        return false;
    }

    if (softmax->op != GGML_OP_SOFT_MAX || softmax->src[0] != reshape) {
        return false;
    }

    if (weights->op != GGML_OP_RESHAPE || weights->src[0] != softmax) {
        return false;
    }

    if (!ggml_cuda_can_fuse_range(cgraph, node_idx, node_idx + 5, { node_idx + 1, node_idx + 5 })) {
        return false;
    }

    topk_view_idx = node_idx + 1;
    softmax_idx   = node_idx + 4;
    weights_idx   = node_idx + 5;
    return true;
}

static bool moe_grouped_enabled = []() {
    const char * env = std::getenv("LLAMA_MOE_GROUPED_EXPERTS");
    return env && env[0] != '\0' && env[0] != '0';
}();

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        args.sigmoid = true;
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
    for(size_t ds4_i=0;node_idx>=0&&ds4_i<ops.size()&&node_idx+int(ds4_i)<cgraph->n_nodes;++ds4_i)
        if(ggml_ds4_native_half_requested(cgraph->nodes[node_idx+int(ds4_i)]))return false;

#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            return true;
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            return true;
        }
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            return true;
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contigous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

static bool ggml_cuda_can_fuse_mul_add(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL, GGML_OP_ADD }, { node_idx + 1 })) {
        return false;
    }

    const ggml_tensor * mul = cgraph->nodes[node_idx];
    const ggml_tensor * add = cgraph->nodes[node_idx + 1];
    if (add->src[0] != mul && add->src[1] != mul) {
        return false;
    }

    const ggml_tensor * addend = add->src[0] == mul ? add->src[1] : add->src[0];
    if (mul->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        addend->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_are_same_shape(mul, add) || !ggml_are_same_shape(mul->src[0], add) ||
        !ggml_are_same_shape(addend, add) || !ggml_is_contiguous_rows(mul->src[0]) ||
        !ggml_is_contiguous(addend) || !ggml_is_contiguous(add) ||
        !ggml_can_repeat(mul->src[1], add)) {
        return false;
    }

    // k_mul_add_bcast_f32 dereferences both mul operands as float*, and
    // ggml_cuda_op_fused_mul_add asserts nb[0] == sizeof(float) on each.  src[0] is already
    // covered by ggml_is_contiguous_rows above; src[1] was not constrained at all, so a broadcast
    // operand with a non-unit dim-0 stride reached the assert and ABORTED the process instead of
    // falling back.  Seen in stream_local decode: binbcast.cu:546 killed the server after a clean
    // streaming prefill (identically with LLAMA_PREFILL_KV_RAM=0 and =1).  Declining to fuse here
    // sends the pair down the generic unfused MUL/ADD path, which handles any stride.
    if (mul->src[1]->nb[0] != ggml_type_size(mul->src[1]->type)) {
        return false;
    }

    // k_mul_add_bcast_f32 maps dimensions 1 and 2*3 directly to CUDA grid
    // dimensions.  Keep the generic path for shapes outside those limits.
    return add->ne[1] <= 65535 && add->ne[2] * add->ne[3] <= 65535;
}

static const ggml_tensor * ggml_cuda_scalar_addend(const ggml_tensor * add, const ggml_tensor * value) {
    if (add->src[0] == value && add->src[1] != nullptr && ggml_nelements(add->src[1]) == 1) {
        return add->src[1];
    }
    if (add->src[1] == value && add->src[0] != nullptr && ggml_nelements(add->src[0]) == 1) {
        return add->src[0];
    }
    return nullptr;
}

static bool ggml_cuda_is_swap_01_view(const ggml_tensor * permute, const ggml_tensor * src) {
    return permute->op == GGML_OP_PERMUTE && permute->src[0] == src &&
        permute->ne[0] == src->ne[1] && permute->ne[1] == src->ne[0] &&
        permute->ne[2] == src->ne[2] && permute->ne[3] == src->ne[3] &&
        permute->nb[0] == src->nb[1] && permute->nb[1] == src->nb[0] &&
        permute->nb[2] == src->nb[2] && permute->nb[3] == src->nb[3];
}

struct ggml_cuda_sum_rows_div_args {
    const ggml_tensor * src        = nullptr;
    const ggml_tensor * eps        = nullptr;
    ggml_tensor       * dst        = nullptr;
    int                 node_count = 0;
    bool                columnwise = false;
};

static bool ggml_cuda_match_sum_rows_div(
        const ggml_cgraph             * cgraph,
        int                             node_idx,
        ggml_cuda_sum_rows_div_args   & args) {
    // Row normalization: x / (sum_rows(x) + eps).
    if (ggml_can_fuse_subgraph(cgraph, node_idx,
            { GGML_OP_SUM_ROWS, GGML_OP_ADD, GGML_OP_DIV }, { node_idx + 2 })) {
        const ggml_tensor * sum = cgraph->nodes[node_idx];
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];
        ggml_tensor       * div = cgraph->nodes[node_idx + 2];
        const ggml_tensor * eps = ggml_cuda_scalar_addend(add, sum);

        if (eps != nullptr && div->src[0] == sum->src[0] && div->src[1] == add) {
            args.src        = sum->src[0];
            args.eps        = eps;
            args.dst        = div;
            args.node_count = 3;
            args.columnwise = false;
        }
    }

    // Column normalization as emitted by DeepSeek-V4 Sinkhorn:
    //   t = cont(permute(x, 1, 0, 2, 3)); s = sum_rows(t) + eps;
    //   y = x / permute(s, 1, 0, 2, 3)
    // The fused kernel reads x with the transposed reduction indexing and
    // removes the otherwise dominant CONT layout copy.
    if (args.src == nullptr && ggml_can_fuse_subgraph(cgraph, node_idx,
            { GGML_OP_CONT, GGML_OP_SUM_ROWS, GGML_OP_ADD, GGML_OP_PERMUTE, GGML_OP_DIV },
            { node_idx + 4 })) {
        const ggml_tensor * cont       = cgraph->nodes[node_idx];
        const ggml_tensor * sum        = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * denom_perm = cgraph->nodes[node_idx + 3];
        ggml_tensor       * div        = cgraph->nodes[node_idx + 4];
        const ggml_tensor * src        = cont->src[0] != nullptr ? cont->src[0]->src[0] : nullptr;
        const ggml_tensor * eps        = ggml_cuda_scalar_addend(add, sum);

        if (src != nullptr && ggml_cuda_is_swap_01_view(cont->src[0], src) &&
            sum->src[0] == cont && ggml_cuda_is_swap_01_view(denom_perm, add) &&
            eps != nullptr && div->src[0] == src && div->src[1] == denom_perm) {
            args.src        = src;
            args.eps        = eps;
            args.dst        = div;
            args.node_count = 5;
            args.columnwise = true;
        }
    }

    if (args.src == nullptr) {
        return false;
    }

    if (args.src->type != GGML_TYPE_F32 || args.eps->type != GGML_TYPE_F32 ||
        args.dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(args.src) ||
        !ggml_is_contiguous(args.dst) || !ggml_are_same_shape(args.src, args.dst)) {
        return false;
    }

    const int64_t nrows = args.columnwise ?
        args.src->ne[0] * args.src->ne[2] * args.src->ne[3] : ggml_nrows(args.src);
    return args.src->ne[0] <= 2147483647 && args.src->ne[1] <= 2147483647 &&
        nrows <= 2147483647;
}

struct ggml_cuda_lid_score_reduce_args {
    const ggml_tensor * kq         = nullptr;
    const ggml_tensor * weights    = nullptr;
    ggml_tensor       * dst        = nullptr;
    float               scale      = 0.0f;
    int                 node_count = 0;
};

static bool ggml_cuda_lid_fusion_trace_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_CUDA_FUSION_TRACE");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}

struct ggml_cuda_lid_fusion_trace_state {
    int emitted    = 0;
    int suppressed = 0;
};

static thread_local ggml_cuda_lid_fusion_trace_state g_lid_fusion_trace;

static void ggml_cuda_lid_fusion_trace_begin_graph() {
    if (ggml_cuda_lid_fusion_trace_enabled()) {
        g_lid_fusion_trace = {};
    }
}

static void ggml_cuda_lid_fusion_trace_end_graph() {
    if (ggml_cuda_lid_fusion_trace_enabled() && g_lid_fusion_trace.suppressed > 0) {
        fprintf(stderr, "[fusion-trace] lid_score_reduce: %d rejects suppressed\n",
                g_lid_fusion_trace.suppressed);
        fflush(stderr);
    }
}

static bool ggml_cuda_lid_fusion_trace_reject(
        const ggml_cgraph * cgraph, int node_idx, const char * checkpoint, const char * format, ...) {
    if (!ggml_cuda_lid_fusion_trace_enabled()) {
        return false;
    }

    constexpr int max_lines = 200;
    if (g_lid_fusion_trace.emitted >= max_lines) {
        ++g_lid_fusion_trace.suppressed;
        return false;
    }

    const ggml_tensor * node = node_idx >= 0 && node_idx < cgraph->n_nodes ? cgraph->nodes[node_idx] : nullptr;
    fprintf(stderr, "[fusion-trace] lid_score_reduce reject@%s: node=%d op=%s reason=",
            checkpoint, node_idx, node != nullptr ? ggml_op_name(node->op) : "out-of-range");
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    fflush(stderr);
    ++g_lid_fusion_trace.emitted;
    return false;
}

static void ggml_cuda_lid_fusion_trace_match(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_lid_fusion_trace_enabled()) {
        return;
    }
    if (g_lid_fusion_trace.emitted >= 200) {
        return;
    }
    const ggml_tensor * node = cgraph->nodes[node_idx];
    fprintf(stderr, "[fusion-trace] lid_score_reduce match: node=%d op=%s name=%s\n",
            node_idx, ggml_op_name(node->op), ggml_get_name(node));
    fflush(stderr);
    ++g_lid_fusion_trace.emitted;
}

static bool ggml_cuda_lid_node_is_in_window(int candidate_idx, int node_idx, int node_count) {
    return candidate_idx >= node_idx && candidate_idx < node_idx + node_count;
}

static bool ggml_cuda_lid_can_fuse_subgraph(
        const ggml_cgraph * cgraph, int node_idx, int node_count, const ggml_op * ops, int output_idx) {
    // LLAMA_DSV4_LID_FUSE=0 falls back to the portable ggml-op decomposition this fusion replaces.
    //
    // Why this knob exists: the Mac's llama-decode-receiver runs THIS SAME FORK's DSV4 code on Metal,
    // resident, and RECALLS a mid-prompt needle at 596 / 2862 / 11330 tokens. The RTX, CUDA, running
    // the same source and (tested) the same 153 GB GGUF, does not. Metal has no fused LID kernel and
    // takes the portable chain; CUDA matches this private seven-node tail and replaces it. So the
    // fused kernel is a prime suspect, and this makes it an A/B instead of an argument -- with the
    // fusion off, CUDA runs the same math Metal does.
    static const bool fuse_enabled = []() {
        const char * e = std::getenv("LLAMA_DSV4_LID_FUSE");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    if (!fuse_enabled) {
        return false;
    }
    if (node_idx + node_count > cgraph->n_nodes) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "window-bounds",
                "need %d nodes through index %d but graph has %d", node_count,
                node_idx + node_count - 1, cgraph->n_nodes);
    }

    for (int i = 0; i < node_count; ++i) {
        const int candidate_idx = node_idx + i;
        const ggml_tensor * node = cgraph->nodes[candidate_idx];
        if (node->op != ops[i]) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "op-sequence",
                    "position=%d graph-node=%d expected=%s actual=%s actual-name=%s", i,
                    candidate_idx, ggml_op_name(ops[i]), ggml_op_name(node->op), ggml_get_name(node));
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "compute-flag",
                    "position=%d graph-node=%d op=%s name=%s missing GGML_TENSOR_FLAG_COMPUTE",
                    i, candidate_idx, ggml_op_name(node->op), ggml_get_name(node));
        }
        if (node == cgraph->nodes[output_idx]) {
            continue;
        }
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-flag",
                    "position=%d graph-node=%d op=%s name=%s is externally visible",
                    i, candidate_idx, ggml_op_name(node->op), ggml_get_name(node));
        }

        int subgraph_uses = 0;
        for (int j = i + 1; j < node_count; ++j) {
            const ggml_tensor * consumer = cgraph->nodes[node_idx + j];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                subgraph_uses += consumer->src[src_idx] == node;
            }
        }
        const int total_uses = ggml_node_get_use_count(cgraph, candidate_idx);
        if (subgraph_uses != total_uses) {
            int extra_idx = -1;
            int extra_src = -1;
            const ggml_tensor * extra = nullptr;
            for (int j = 0; j < cgraph->n_nodes && extra == nullptr; ++j) {
                const bool expected_consumer = j > candidate_idx &&
                    ggml_cuda_lid_node_is_in_window(j, node_idx, node_count);
                for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                    if (cgraph->nodes[j]->src[src_idx] == node && !expected_consumer) {
                        extra_idx = j;
                        extra_src = src_idx;
                        extra = cgraph->nodes[j];
                        break;
                    }
                }
            }
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "liveness",
                    "position=%d graph-node=%d producer-op=%s producer-name=%s internal-uses=%d total-uses=%d "
                    "extra-consumer-node=%d extra-consumer-src=%d extra-consumer-op=%s extra-consumer-name=%s",
                    i, candidate_idx, ggml_op_name(node->op), ggml_get_name(node), subgraph_uses, total_uses,
                    extra_idx, extra_src, extra != nullptr ? ggml_op_name(extra->op) : "not-found",
                    extra != nullptr ? ggml_get_name(extra) : "not-found");
        }

        for (const ggml_tensor * view_src = node->view_src; view_src != nullptr; view_src = view_src->view_src) {
            bool found = false;
            for (int j = 0; j < node_count; ++j) {
                found = found || cgraph->nodes[node_idx + j] == view_src;
            }
            if (!found) {
                return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "view-liveness",
                        "position=%d graph-node=%d op=%s name=%s external-view-src-op=%s external-view-src-name=%s",
                        i, candidate_idx, ggml_op_name(node->op), ggml_get_name(node),
                        ggml_op_name(view_src->op), ggml_get_name(view_src));
            }
        }
    }
    return true;
}

static bool ggml_cuda_match_lid_score_reduce(
        const ggml_cgraph                * cgraph,
        int                                node_idx,
        ggml_cuda_lid_score_reduce_args  & args) {
    // Exact DeepSeek-V4 portable score chain (PERMUTE views are inspected but
    // do no device work):
    //   cont(permute(kq, 2,1,0,3)) -> relu
    //   scale(weights) -> mul -> sum_rows -> cont(permute(..., 2,1,0,3)).
    // All intermediates must be private to the chain before they are elided.
    constexpr int node_count = 7;
    const ggml_op ops[node_count] = {
        GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL,
        GGML_OP_SUM_ROWS, GGML_OP_PERMUTE, GGML_OP_CONT,
    };
    const int output = node_idx + node_count - 1;
    if (!ggml_cuda_lid_fusion_trace_enabled()) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops, &output, 1)) {
            return false;
        }
    } else if (!ggml_cuda_lid_can_fuse_subgraph(cgraph, node_idx, node_count, ops, output)) {
        return false;
    }

    const ggml_tensor * cont_in  = cgraph->nodes[node_idx + 0];
    const ggml_tensor * relu     = cgraph->nodes[node_idx + 1];
    const ggml_tensor * scaled_w = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 3];
    const ggml_tensor * sum      = cgraph->nodes[node_idx + 4];
    const ggml_tensor * perm_out = cgraph->nodes[node_idx + 5];
    ggml_tensor       * cont_out = cgraph->nodes[node_idx + 6];
    const ggml_tensor * perm_in  = cont_in->src[0];
    const ggml_tensor * kq       = perm_in != nullptr ? perm_in->src[0] : nullptr;
    const ggml_tensor * weights  = scaled_w->src[0];

    if (perm_in == nullptr) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-in-null",
                "cont input has no source");
    }
    if (kq == nullptr) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-null",
                "permute input has no source");
    }
    if (weights == nullptr) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-null",
                "scale input has no source");
    }
    if (perm_in->op != GGML_OP_PERMUTE) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-in-op",
                "expected=%s actual=%s name=%s", ggml_op_name(GGML_OP_PERMUTE),
                ggml_op_name(perm_in->op), ggml_get_name(perm_in));
    }
    if (perm_in->src[0] != kq) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-in-source",
                "expected-kq=%s actual-source=%s", ggml_get_name(kq),
                perm_in->src[0] != nullptr ? ggml_get_name(perm_in->src[0]) : "null");
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const int source_dim = d == 0 ? 2 : (d == 2 ? 0 : d);
        if (perm_in->ne[d] != kq->ne[source_dim]) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-in-shape",
                    "dim=%d expected=%" PRId64 " actual=%" PRId64, d, kq->ne[source_dim], perm_in->ne[d]);
        }
        if (perm_in->nb[d] != kq->nb[source_dim]) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-in-stride",
                    "dim=%d expected=%zu actual=%zu", d, kq->nb[source_dim], perm_in->nb[d]);
        }
    }
    if (cont_in->src[0] != perm_in) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "cont-in-source",
                "expected=%s actual=%s", ggml_get_name(perm_in),
                cont_in->src[0] != nullptr ? ggml_get_name(cont_in->src[0]) : "null");
    }
    if (ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "relu-kind",
                "expected=%d actual=%d", (int) GGML_UNARY_OP_RELU, (int) ggml_get_unary_op(relu));
    }
    if (relu->src[0] != cont_in) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "relu-source",
                "expected=%s actual=%s", ggml_get_name(cont_in),
                relu->src[0] != nullptr ? ggml_get_name(relu->src[0]) : "null");
    }
    if (mul->src[0] != relu) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "mul-source-0",
                "expected=%s actual=%s", ggml_get_name(relu),
                mul->src[0] != nullptr ? ggml_get_name(mul->src[0]) : "null");
    }
    if (mul->src[1] != scaled_w) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "mul-source-1",
                "expected=%s actual=%s", ggml_get_name(scaled_w),
                mul->src[1] != nullptr ? ggml_get_name(mul->src[1]) : "null");
    }
    if (sum->src[0] != mul) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "sum-source",
                "expected=%s actual=%s", ggml_get_name(mul),
                sum->src[0] != nullptr ? ggml_get_name(sum->src[0]) : "null");
    }
    if (perm_out->op != GGML_OP_PERMUTE) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-out-op",
                "expected=%s actual=%s name=%s", ggml_op_name(GGML_OP_PERMUTE),
                ggml_op_name(perm_out->op), ggml_get_name(perm_out));
    }
    if (perm_out->src[0] != sum) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-out-source",
                "expected=%s actual=%s", ggml_get_name(sum),
                perm_out->src[0] != nullptr ? ggml_get_name(perm_out->src[0]) : "null");
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const int source_dim = d == 0 ? 2 : (d == 2 ? 0 : d);
        if (perm_out->ne[d] != sum->ne[source_dim]) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-out-shape",
                    "dim=%d expected=%" PRId64 " actual=%" PRId64, d, sum->ne[source_dim], perm_out->ne[d]);
        }
        if (perm_out->nb[d] != sum->nb[source_dim]) {
            return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "perm-out-stride",
                    "dim=%d expected=%zu actual=%zu", d, sum->nb[source_dim], perm_out->nb[d]);
        }
    }
    if (cont_out->src[0] != perm_out) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "cont-out-source",
                "expected=%s actual=%s", ggml_get_name(perm_out),
                cont_out->src[0] != nullptr ? ggml_get_name(cont_out->src[0]) : "null");
    }

    const float scale = ggml_get_op_params_f32(scaled_w, 0);
    const float bias  = ggml_get_op_params_f32(scaled_w, 1);

    // The fused kernel applies the SCALE op parameter and runtime head count
    // directly, so exact model-specific constants are unnecessary.  Require
    // sane positive values in addition to the complete dependency/layout chain.
    if (!(scale > 0.0f && scale < 1.0f)) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "scale-value",
                "expected=(0,1) actual=%.9g", scale);
    }
    if (bias != 0.0f) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "scale-bias",
                "expected=0 actual=%.9g", bias);
    }
    if (kq->ne[2] <= 0) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-heads",
                "expected=>0 actual=%" PRId64, kq->ne[2]);
    }
    if (kq->ne[3] != 1 && kq->ne[3] != 2) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-streams",
                "expected=1-or-2 actual=%" PRId64, kq->ne[3]);
    }
    if (kq->ne[0] <= 0) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-n-lid-positive",
                "expected=>0 actual=%" PRId64, kq->ne[0]);
    }
    if (kq->ne[1] <= 0) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-token-positive",
                "expected=>0 actual=%" PRId64, kq->ne[1]);
    }
    if (kq->ne[0] > 2147483647) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-n-lid-limit",
                "expected<=2147483647 actual=%" PRId64, kq->ne[0]);
    }
    if (kq->ne[1] > 65535) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-token-grid-limit",
                "expected<=65535 actual=%" PRId64, kq->ne[1]);
    }
    if (kq->type != GGML_TYPE_F32) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-type",
                "expected=%s actual=%s", ggml_type_name(GGML_TYPE_F32), ggml_type_name(kq->type));
    }
    if (weights->type != GGML_TYPE_F32) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-type",
                "expected=%s actual=%s", ggml_type_name(GGML_TYPE_F32), ggml_type_name(weights->type));
    }
    if (cont_out->type != GGML_TYPE_F32) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-type",
                "expected=%s actual=%s", ggml_type_name(GGML_TYPE_F32), ggml_type_name(cont_out->type));
    }
    if (!ggml_is_contiguous(kq)) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "kq-contiguous",
                "strides=[%zu,%zu,%zu,%zu] shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                kq->nb[0], kq->nb[1], kq->nb[2], kq->nb[3], kq->ne[0], kq->ne[1], kq->ne[2], kq->ne[3]);
    }
    if (!ggml_is_contiguous(weights)) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-contiguous",
                "strides=[%zu,%zu,%zu,%zu] shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                weights->nb[0], weights->nb[1], weights->nb[2], weights->nb[3],
                weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3]);
    }
    if (!ggml_is_contiguous(cont_out)) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-contiguous",
                "strides=[%zu,%zu,%zu,%zu] shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                cont_out->nb[0], cont_out->nb[1], cont_out->nb[2], cont_out->nb[3],
                cont_out->ne[0], cont_out->ne[1], cont_out->ne[2], cont_out->ne[3]);
    }
    if (weights->ne[0] != kq->ne[2]) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-head-shape",
                "expected=%" PRId64 " actual=%" PRId64, kq->ne[2], weights->ne[0]);
    }
    if (weights->ne[1] != kq->ne[1]) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-token-shape",
                "expected=%" PRId64 " actual=%" PRId64, kq->ne[1], weights->ne[1]);
    }
    if (weights->ne[2] != 1) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-ne2",
                "expected=1 actual=%" PRId64, weights->ne[2]);
    }
    if (weights->ne[3] != kq->ne[3]) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "weights-stream-shape",
                "expected=%" PRId64 " actual=%" PRId64, kq->ne[3], weights->ne[3]);
    }
    if (cont_out->ne[0] != kq->ne[0]) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-n-lid-shape",
                "expected=%" PRId64 " actual=%" PRId64, kq->ne[0], cont_out->ne[0]);
    }
    if (cont_out->ne[1] != kq->ne[1]) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-token-shape",
                "expected=%" PRId64 " actual=%" PRId64, kq->ne[1], cont_out->ne[1]);
    }
    if (cont_out->ne[2] != 1) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-ne2",
                "expected=1 actual=%" PRId64, cont_out->ne[2]);
    }
    if (cont_out->ne[3] != kq->ne[3]) {
        return ggml_cuda_lid_fusion_trace_reject(cgraph, node_idx, "output-stream-shape",
                "expected=%" PRId64 " actual=%" PRId64, kq->ne[3], cont_out->ne[3]);
    }

    args.kq         = kq;
    args.weights    = weights;
    args.dst        = cont_out;
    args.scale      = scale;
    args.node_count = node_count;
    ggml_cuda_lid_fusion_trace_match(cgraph, node_idx);
    return true;
}

enum ggml_cuda_fusion_family {
    GGML_CUDA_FUSION_BAILINGMOEV2,
    GGML_CUDA_FUSION_GLM45MOE,
    GGML_CUDA_FUSION_OPENAI_EXPERTS,
    GGML_CUDA_FUSION_TOPK_MOE,
    GGML_CUDA_FUSION_ROPE_SET_ROWS,
    GGML_CUDA_FUSION_LID_SCORE_REDUCE,
    GGML_CUDA_FUSION_SUM_ROWS_DIV,
    GGML_CUDA_FUSION_SOFTPLUS_SQRT,
    GGML_CUDA_FUSION_MUL_ADD,
    GGML_CUDA_FUSION_ADD_CHAIN,
    GGML_CUDA_FUSION_MM_GLU_BIAS,
    GGML_CUDA_FUSION_MM_GLU,
    GGML_CUDA_FUSION_MM_BIAS,
    GGML_CUDA_FUSION_RMS_NORM_MUL_ADD,
    GGML_CUDA_FUSION_RMS_NORM_MUL,
    GGML_CUDA_FUSION_SOFTCAP,
    GGML_CUDA_FUSION_COUNT,
};

static constexpr const char * ggml_cuda_fusion_family_names[GGML_CUDA_FUSION_COUNT] = {
    "bailingmoev2",
    "glm45moe",
    "openai_experts",
    "topk_moe",
    "rope_set_rows",
    "lid_score_reduce",
    "sum_rows_div",
    "softplus_sqrt",
    "mul_add",
    "add_chain",
    "mm_glu_bias",
    "mm_glu",
    "mm_bias",
    "rms_norm_mul_add",
    "rms_norm_mul",
    "softcap",
};

static std::array<std::atomic<uint64_t>, GGML_CUDA_FUSION_COUNT> g_cuda_fusion_counts = {};
static std::atomic<uint64_t> g_cuda_fusion_count_graphs{0};

static bool ggml_cuda_fusion_counts_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_CUDA_FUSION_COUNTS");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

static void ggml_cuda_fusion_count(ggml_cuda_fusion_family family) {
    if (ggml_cuda_fusion_counts_enabled()) {
        g_cuda_fusion_counts[(size_t) family].fetch_add(1, std::memory_order_relaxed);
    }
}

static void ggml_cuda_fusion_counts_maybe_dump() {
    if (!ggml_cuda_fusion_counts_enabled()) {
        return;
    }

    const uint64_t graphs = g_cuda_fusion_count_graphs.fetch_add(1, std::memory_order_relaxed) + 1;
    if (graphs != 1 && (graphs % 256) != 0) {
        return;
    }

    std::string line = "*** CUDA_FUSION_COUNTS: graphs=" + std::to_string(graphs);
    for (size_t family = 0; family < GGML_CUDA_FUSION_COUNT; ++family) {
        line += " ";
        line += ggml_cuda_fusion_family_names[family];
        line += ":";
        line += std::to_string(g_cuda_fusion_counts[family].load(std::memory_order_relaxed));
    }
    fprintf(stderr, "%s\n", line.c_str());
    fflush(stderr);
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    ggml_cuda_dsv41_routing_scope sorted_routing_scope(*cuda_ctx,cgraph,!use_cuda_graph);
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    ggml_cuda_lid_fusion_trace_begin_graph();
    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

#ifdef GGML_CUDA_DEBUG
                const int nodes_fused = i - prev_i - 1;
                if (nodes_fused > 0) {
                    GGML_LOG_INFO("nodes_fused: %d\n", nodes_fused);
                }
#endif
                prev_i = i;

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // start of fusion operations
                static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
                if (!disable_fusion) {
                    if (moe_grouped_enabled) {
                        int topk_view_idx = -1;
                        int weights_idx   = -1;

                        if (ggml_cuda_match_bailingmoev2(cgraph, i, topk_view_idx, weights_idx) &&
                                ggml_cuda_should_use_topk_moe(node, node->src[0], cgraph->nodes[weights_idx], cgraph->nodes[topk_view_idx])) {
                            cuda_bailingmoev2_experts(*cuda_ctx, cgraph->nodes[weights_idx], cgraph->nodes[topk_view_idx]);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_BAILINGMOEV2);
                            i = weights_idx;
                            continue;
                        }

                        if (ggml_cuda_match_glm45moe(cgraph, i, topk_view_idx, weights_idx) &&
                                ggml_cuda_should_use_topk_moe(node, node->src[0], cgraph->nodes[weights_idx], cgraph->nodes[topk_view_idx])) {
                            cuda_glm45moe_experts(*cuda_ctx, cgraph->nodes[weights_idx], cgraph->nodes[topk_view_idx]);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_GLM45MOE);
                            i = weights_idx;
                            continue;
                        }

                        int softmax_idx = -1;
                        if (ggml_cuda_match_openai_experts(cgraph, i, topk_view_idx, softmax_idx, weights_idx) &&
                                ggml_cuda_should_use_topk_moe(cgraph->nodes[softmax_idx], node->src[0], cgraph->nodes[weights_idx], cgraph->nodes[topk_view_idx])) {
                            cuda_openai_experts(*cuda_ctx, node, cgraph->nodes[softmax_idx]);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_OPENAI_EXPERTS);
                            i = weights_idx;
                            continue;
                        }
                    }

                    ggml_cuda_topk_moe_args args;

                    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                        cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                        const bool can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);

                        std::vector<ggml_op> ops;

                        if (can_fuse) {
                            const ggml_tensor * logits  = node->src[0];
                            ggml_tensor *       weights = nullptr;
                            ggml_tensor *       ids     = nullptr;
                            const ggml_tensor * bias    = nullptr;
                            const ggml_tensor * clamp   = nullptr;
                            const ggml_tensor * scale   = nullptr;

                            if (!args.delayed_softmax) {
                                ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
                                int     out_nodes[2];  // nodes which can't be elided

                                if (args.prob_bias) {
                                    bias = cgraph->nodes[i + 2]->src[1];
                                    ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT,
                                                            GGML_OP_VIEW, GGML_OP_GET_ROWS });
                                    out_nodes[0] = i + 4;
                                    ids          = cgraph->nodes[i + 4];
                                } else {
                                    ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                            GGML_OP_GET_ROWS });
                                    out_nodes[0] = i + 3;
                                    ids          = cgraph->nodes[i + 3];
                                }

                                if (args.norm) {
                                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
                                                            GGML_OP_DIV, GGML_OP_RESHAPE });
                                    clamp = cgraph->nodes[i + ops.size() - 3];
                                }
                                if (args.scale) {
                                    ops.insert(ops.end(), { GGML_OP_SCALE });
                                    scale = cgraph->nodes[i + ops.size() - 1];
                                }

                                weights      = cgraph->nodes[i + ops.size() - 1];
                                out_nodes[1] = i + ops.size() - 1;

                                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {
                                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                                    ggml_cuda_fusion_count(GGML_CUDA_FUSION_TOPK_MOE);
                                    i += ops.size() - 1;
                                    continue;
                                }
                            } else if (!args.norm && !args.prob_bias) {
                                //special case gpt-oss, no norm, no bias.
                                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
                                                        GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                                weights                     = cgraph->nodes[i + 5];
                                ids                         = cgraph->nodes[i + 1];
                                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                                int out_nodes[2] = { i + 1, i + 5 };
                                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids)) {
                                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                                    ggml_cuda_fusion_count(GGML_CUDA_FUSION_TOPK_MOE);
                                    i += ops.size() - 1;
                                    continue;
                                }
                            }
                        }
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
                        ggml_tensor * rope = cgraph->nodes[i];
                        ggml_tensor * set_rows = cgraph->nodes[i + 2];

                        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
                        ggml_cuda_fusion_count(GGML_CUDA_FUSION_ROPE_SET_ROWS);

                        i += 2;
                        continue;
                    }

                    if (node->op == GGML_OP_CONT) {
                        ggml_cuda_lid_score_reduce_args lid_args;
                        if (ggml_cuda_match_lid_score_reduce(cgraph, i, lid_args)) {
                            ggml_cuda_op_fused_lid_score_reduce(
                                *cuda_ctx, lid_args.kq, lid_args.weights, lid_args.dst, lid_args.scale);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_LID_SCORE_REDUCE);
                            i += lid_args.node_count - 1;
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_CONT || node->op == GGML_OP_SUM_ROWS) {
                        ggml_cuda_sum_rows_div_args norm_args;
                        if (ggml_cuda_match_sum_rows_div(cgraph, i, norm_args)) {
                            ggml_cuda_op_fused_sum_rows_div(
                                *cuda_ctx, norm_args.src, norm_args.eps, norm_args.dst, norm_args.columnwise);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_SUM_ROWS_DIV);
                            i += norm_args.node_count - 1;
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SOFTPLUS &&
                        ggml_can_fuse_subgraph(cgraph, i,
                            { GGML_OP_UNARY, GGML_OP_SQRT }, { i + 1 }) &&
                        cgraph->nodes[i + 1]->src[0] == node && node->type == GGML_TYPE_F32 &&
                        cgraph->nodes[i + 1]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[0]) &&
                        ggml_is_contiguous(cgraph->nodes[i + 1])) {
                        ggml_cuda_op_fused_softplus_sqrt(*cuda_ctx, cgraph->nodes[i + 1]);
                        ggml_cuda_fusion_count(GGML_CUDA_FUSION_SOFTPLUS_SQRT);
                        i += 1;
                        continue;
                    }

                    if (node->op == GGML_OP_MUL && ggml_cuda_can_fuse_mul_add(cgraph, i)) {
                        ggml_cuda_op_fused_mul_add(*cuda_ctx, node, cgraph->nodes[i + 1]);
                        ggml_cuda_fusion_count(GGML_CUDA_FUSION_MUL_ADD);
                        i += 1;
                        continue;
                    }

                    if (node->op == GGML_OP_ADD) {
                        int n_fuse = 0;
                        ggml_op ops[8];
                        std::fill(ops, ops + 8, GGML_OP_ADD);

                        for (; n_fuse <= 6; ++n_fuse){
                            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                                break;
                            }
                            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                                break;
                            }
                            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                                break;
                            }
                        }

                        n_fuse++;

                        if (n_fuse > 1) {
                            for (int j = 0; j < n_fuse - 1; ++j) {
                                node->src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
                            }
                            cgraph->nodes[i + n_fuse - 1]->data = node->data;
                            ggml_cuda_op_fused_add(*cuda_ctx, node, n_fuse);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_ADD_CHAIN);
                            i += n_fuse - 1;

                            continue;
                        }
                    }

                    bool fused_mul_mat_vec = false;
                    int fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
                            ggml_tensor * glu         = cgraph->nodes[i + 4];
                            ggml_tensor * gate_bias_n = glu->src[0];
                            ggml_tensor * up_bias_n   = glu->src[1];

                            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
                            ggml_tensor * gate_n      = nullptr;
                            ggml_tensor * up_n        = nullptr;

                            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                                gate_n = cgraph->nodes[i];
                                up_n   = cgraph->nodes[i + 2];
                            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                                gate_n = cgraph->nodes[i + 2];
                                up_n   = cgraph->nodes[i];
                            } else {
                                continue;
                            }

                            auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) {
                                if (op_bias == GGML_OP_ADD) {
                                    if (bias_node->src[0] == mul_node) {
                                        return bias_node->src[1];
                                    }
                                    if (bias_node->src[1] == mul_node) {
                                        return bias_node->src[0];
                                    }
                                    return (ggml_tensor *) nullptr;
                                }
                                GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
                                GGML_ASSERT(bias_node->src[0] == mul_node);
                                return bias_node->src[1];
                            };

                            ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
                            ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

                            if (!up_bias_tensor || !gate_bias_tensor) {
                                continue;
                            }

                            // we don't support repeating adds
                            if (bias_op == GGML_OP_ADD &&
                                (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                 !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                                continue;
                            }

                            const ggml_tensor * src0 = up_n->src[0];
                            const ggml_tensor * src1 = up_n->src[1];
                            const ggml_tensor * ids  = up_n->src[2];

                            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate      = gate_n->src[0];
                                fusion_data.x_bias    = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op    = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                ggml_cuda_fusion_count(GGML_CUDA_FUSION_MM_GLU_BIAS);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }

                            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate      = gate_n->src[0];
                                fusion_data.x_bias    = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op    = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                ggml_cuda_fusion_count(GGML_CUDA_FUSION_MM_GLU_BIAS);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }
                        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
                            ggml_tensor * glu  = cgraph->nodes[i + 2];
                            ggml_tensor * gate = glu->src[0];
                            ggml_tensor * up   = glu->src[1];

                            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1])
                                || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

                            if (!ok) continue;

                            const ggml_tensor * src0 = up->src[0];
                            const ggml_tensor * src1 = up->src[1];
                            const ggml_tensor * ids  = up->src[2];

                            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate   = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                ggml_cuda_fusion_count(GGML_CUDA_FUSION_MM_GLU);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }

                            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate   = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                ggml_cuda_fusion_count(GGML_CUDA_FUSION_MM_GLU);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    fused_mul_mat_vec = false;
                    fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
                            continue;
                        }

                        ggml_tensor * mm_node   = cgraph->nodes[i];
                        ggml_tensor * bias_node = cgraph->nodes[i + 1];

                        ggml_tensor * bias_tensor = nullptr;
                        if (bias_op == GGML_OP_ADD) {
                            if (bias_node->src[0] == mm_node) {
                                bias_tensor = bias_node->src[1];
                            } else if (bias_node->src[1] == mm_node) {
                                bias_tensor = bias_node->src[0];
                            } else {
                                continue;
                            }
                        } else {
                            if (bias_node->src[0] != mm_node) {
                                continue;
                            }
                            bias_tensor = bias_node->src[1];
                        }

                        const ggml_tensor * src0 = mm_node->src[0];
                        const ggml_tensor * src1 = mm_node->src[1];
                        const ggml_tensor * ids  = mm_node->src[2];

                        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
                            continue;
                        }

                        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
                            continue;
                        }

                        ggml_cuda_mm_fusion_args_host fusion_data{};
                        fusion_data.x_bias = bias_tensor;

                        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
                            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_MM_BIAS);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }

                        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
                            ggml_cuda_fusion_count(GGML_CUDA_FUSION_MM_BIAS);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
                        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
                        ggml_cuda_fusion_count(GGML_CUDA_FUSION_RMS_NORM_MUL_ADD);
                        i += 2;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
                        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i+1]);
                        ggml_cuda_fusion_count(GGML_CUDA_FUSION_RMS_NORM_MUL);
                        i++;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
                        i += 2;
                        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i], node);
                        ggml_cuda_fusion_count(GGML_CUDA_FUSION_SOFTCAP);
                        continue;
                    }
                } else if (node->op == GGML_OP_CONT) {
                    ggml_cuda_lid_fusion_trace_reject(cgraph, i, "feature-gate",
                            "GGML_CUDA_DISABLE_FUSION is set");
                }
#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft) || (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
#else
        graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
#endif // USE_CUDA_GRAPH
    }
    ggml_cuda_lid_fusion_trace_end_graph();

#ifdef USE_CUDA_GRAPH
    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
    }
#else
    GGML_UNUSED(graph_key);
#endif  // USE_CUDA_GRAPH
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    const bool cuda_graphs_forced_disabled =
        cuda_ctx->cuda_graph_disable_depth.load(std::memory_order_acquire) > 0;

    if (!cuda_graphs_forced_disabled) {
        graph_key = ggml_cuda_graph_get_key(cgraph);

        use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->is_enabled()) {
            cuda_graph_update_required = ggml_cuda_graph_update_required(cuda_ctx, cgraph);
            use_cuda_graph             = ggml_cuda_graph_check_compability(cgraph);

            graph->record_update(use_cuda_graph, cuda_graph_update_required);
        }
    }
    // Runtime kill-switches force the node-by-node eval path.
    // GGML_CUDA_DISABLE_GRAPHS is retained for diagnostics; normal callers should
    // use ggml_backend_cuda_graphs_disable_push/pop for scoped runtime control.
    static const bool cuda_graphs_disabled = (std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
    if (cuda_graphs_disabled || cuda_graphs_forced_disabled) {
        use_cuda_graph = false;
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    ggml_cuda_fusion_counts_maybe_dump();

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

#ifdef USE_CUDA_GRAPH
    // [2026-07-28 codex-v4 hardening] Streaming prefill disables CUDA graphs via the
    // scoped cuda_graph_disable_depth counter rather than GGML_CUDA_DISABLE_GRAPHS.
    // Do not leave the auxiliary-stream graph optimizer active while graph execution
    // itself is force-disabled: graph_compute() would take the non-graph path while
    // still consuming concurrent_events produced here (multi-stream launches with no
    // capture-time ordering guarantees).
    const bool cuda_graphs_forced_disabled =
        cuda_ctx->cuda_graph_disable_depth.load(std::memory_order_acquire) > 0;
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph =
        !cuda_graphs_forced_disabled && ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
        return;
    }

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}

GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return nullptr;
    }
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    return (void *)cuda_ctx->stream();
}

extern "C" GGML_BACKEND_API void ggml_backend_cuda_glm_topk_gather_mask_begin(
        ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    GGML_ASSERT(cuda_ctx != nullptr);
    GGML_ASSERT(!cuda_ctx->glm_topk_gather_mask_counter_active);

    ggml_cuda_set_device(cuda_ctx->device);
    if (cuda_ctx->glm_topk_gather_masked_slots == nullptr) {
        CUDA_CHECK(cudaMalloc(
            (void **) &cuda_ctx->glm_topk_gather_masked_slots,
            sizeof(*cuda_ctx->glm_topk_gather_masked_slots)));
    }
    CUDA_CHECK(cudaMemset(
        cuda_ctx->glm_topk_gather_masked_slots, 0,
        sizeof(*cuda_ctx->glm_topk_gather_masked_slots)));
    cuda_ctx->glm_topk_gather_mask_counter_active = true;
}

extern "C" GGML_BACKEND_API int64_t ggml_backend_cuda_glm_topk_gather_mask_end(
        ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return 0;
    }
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!cuda_ctx || !cuda_ctx->glm_topk_gather_masked_slots ||
            !cuda_ctx->glm_topk_gather_mask_counter_active) {
        return 0;
    }

    ggml_cuda_set_device(cuda_ctx->device);
    unsigned long long masked_slots = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        &masked_slots, cuda_ctx->glm_topk_gather_masked_slots,
        sizeof(masked_slots), cudaMemcpyDeviceToHost));
    cuda_ctx->glm_topk_gather_mask_counter_active = false;
    return (int64_t) masked_slots;
}

// Phase 296 (2026-05-04): release all cached freed buffers from per-stream pools.
// Also synchronises the device first so any pending kernels finish before cudaFree.
size_t ggml_backend_cuda_pool_clear(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return 0;
    }
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!cuda_ctx) return 0;
    ggml_cuda_set_device(cuda_ctx->device);
    cudaDeviceSynchronize();
    return cuda_ctx->clear_pools();
}

// Phase 302 (2026-05-04): cuda_graphs counter via Meyers-singleton atomic
// in common.cuh. No class member changes, no concurrency races on the counter
// itself (the map insertion is still racy but that was already so).
size_t ggml_backend_cuda_graphs_count(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return 0;
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
#ifdef USE_CUDA_GRAPH
    return cuda_ctx ? cuda_ctx->cuda_graphs.size() : 0;
#else
    (void)cuda_ctx;
    return 0;
#endif
}
size_t ggml_backend_cuda_graphs_clear(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return 0;
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!cuda_ctx) return 0;
#ifdef USE_CUDA_GRAPH
    // Phase 316i (2026-05-04): drop all stale cuda_graphs entries between
    // requests. Each entry holds node pointers from the prior request's
    // graph; those pointers reference freed compute-graph memory now.
    // Re-using a stale entry on a colliding new node ptr causes illegal
    // memory access (Phase 316h symptom). Clearing forces a fresh entry
    // for every node in the next request.
    ggml_cuda_set_device(cuda_ctx->device);
    cudaDeviceSynchronize();
    size_t n = cuda_ctx->cuda_graphs.size();
    cuda_ctx->cuda_graphs.clear();
    fprintf(stderr, "*** GRAPHS_CLEAR: dropped %zu stale entries\n", n);
    fflush(stderr);
    return n;
#else
    return 0;
#endif
}
int64_t ggml_backend_cuda_graphs_total_created(ggml_backend_t /*backend*/) {
    return ggml_cuda_graphs_created_counter().load(std::memory_order_relaxed);
}

void ggml_backend_cuda_graphs_disable_push(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return;
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!cuda_ctx) return;
#ifdef USE_CUDA_GRAPH
    cuda_ctx->cuda_graph_disable_depth.fetch_add(1, std::memory_order_acq_rel);
#else
    (void) cuda_ctx;
#endif
}

void ggml_backend_cuda_graphs_disable_pop(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return;
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!cuda_ctx) return;
#ifdef USE_CUDA_GRAPH
    int prev = cuda_ctx->cuda_graph_disable_depth.load(std::memory_order_acquire);
    while (prev > 0 && !cuda_ctx->cuda_graph_disable_depth.compare_exchange_weak(
               prev, prev - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    if (prev <= 0) {
        GGML_LOG_WARN("%s: CUDA graph disable pop without matching push\n", __func__);
    }
#else
    (void) cuda_ctx;
#endif
}

bool ggml_backend_cuda_graphs_are_disabled(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return false;
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!cuda_ctx) return false;
#ifdef USE_CUDA_GRAPH
    return cuda_ctx->cuda_graph_disable_depth.load(std::memory_order_acquire) > 0;
#else
    (void) cuda_ctx;
    return false;
#endif
}

int64_t ggml_cuda_event_create_total() {
    return ggml_cuda_event_create_counter().load(std::memory_order_relaxed);
}
int64_t ggml_cuda_stream_create_total() {
    return ggml_cuda_stream_create_counter().load(std::memory_order_relaxed);
}
int64_t ggml_cuda_event_destroy_total() {
    return ggml_cuda_event_destroy_counter().load(std::memory_order_relaxed);
}
void ggml_cuda_event_create_count_inc() {
    ggml_cuda_event_create_counter().fetch_add(1, std::memory_order_relaxed);
}
void ggml_cuda_event_destroy_count_inc() {
    ggml_cuda_event_destroy_counter().fetch_add(1, std::memory_order_relaxed);
}
void ggml_cuda_stream_create_count_inc() {
    ggml_cuda_stream_create_counter().fetch_add(1, std::memory_order_relaxed);
}
int64_t ggml_cuda_pinned_alloc_total_bytes() {
    return ggml_cuda_pinned_alloc_bytes_counter().load(std::memory_order_relaxed);
}
int64_t ggml_cuda_pinned_free_total_bytes() {
    return ggml_cuda_pinned_free_bytes_counter().load(std::memory_order_relaxed);
}
void ggml_cuda_pinned_alloc_track(size_t bytes) {
    ggml_cuda_pinned_alloc_bytes_counter().fetch_add((int64_t) bytes, std::memory_order_relaxed);
}
void ggml_cuda_pinned_free_track(size_t bytes) {
    ggml_cuda_pinned_free_bytes_counter().fetch_add((int64_t) bytes, std::memory_order_relaxed);
}

// ============================================================================
// Prefill streaming pipeline: dual-stream overlap (copy + compute)
// ============================================================================

struct ggml_cuda_prefill_pipeline {
    cudaStream_t copy_stream;
    cudaStream_t compute_stream;
    cudaEvent_t  upload_done;
    int          device;
};

void * ggml_backend_cuda_prefill_pipeline_init(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return nullptr;
    }
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);

    auto * p = new ggml_cuda_prefill_pipeline;
    p->device         = cuda_ctx->device;
    p->compute_stream = cuda_ctx->stream();

    ggml_cuda_set_device(p->device);
    // Clear any stale CUDA errors from a previous streaming prefill so
    // the first CUDA_CHECK in this pipeline doesn't inherit them.
    (void)cudaGetLastError();
    CUDA_CHECK(cudaStreamCreateWithFlags(&p->copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&p->upload_done, cudaEventDisableTiming));

    GGML_LOG_INFO("%s: created copy stream %p + event for device %d (compute stream %p)\n",
                  __func__, (void *)p->copy_stream, p->device, (void *)p->compute_stream);
    return p;
}

void ggml_backend_cuda_prefill_pipeline_free(void * pipeline) {
    if (!pipeline) {
        return;
    }
    auto * p = static_cast<ggml_cuda_prefill_pipeline *>(pipeline);
    ggml_cuda_set_device(p->device);
    // Full device sync to catch ANY pending async errors (compute stream,
    // copy stream, cuBLAS internal streams, etc.).  Per-stream sync can
    // miss errors from streams we don't own.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        GGML_LOG_WARN("%s: cudaDeviceSynchronize found latent error: %s\n",
                      __func__, cudaGetErrorString(err));
        // Clear the error so subsequent CUDA calls don't inherit it
        (void)cudaGetLastError();
    }
    cudaStreamDestroy(p->copy_stream);
    cudaEventDestroy(p->upload_done);
    delete p;
}

// Phase 316b (2026-05-04): private ptr->size map for prefill pinned alloc/free
// so the free path can log the matching size without callers passing it.
static std::mutex & ggml_cuda_pin_size_mtx() {
    static std::mutex m;
    return m;
}
static std::unordered_map<void *, size_t> & ggml_cuda_pin_size_map() {
    static std::unordered_map<void *, size_t> m;
    return m;
}

void * ggml_backend_cuda_prefill_pinned_alloc(size_t size) {
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMallocHost(%.2f MB) failed: %s\n",
                      __func__, size / (1024.0 * 1024.0), cudaGetErrorString(err));
        return nullptr;
    }
    ggml_cuda_pinned_alloc_track(size);
    {
        std::lock_guard<std::mutex> lk(ggml_cuda_pin_size_mtx());
        ggml_cuda_pin_size_map()[ptr] = size;
    }
    fprintf(stderr, "*** PIN_ALLOC: %p +%.1fMB (cum_alloc=%lldMB cum_free=%lldMB)\n",
            ptr, size / (1024.0 * 1024.0),
            (long long)(ggml_cuda_pinned_alloc_total_bytes() / (1024 * 1024)),
            (long long)(ggml_cuda_pinned_free_total_bytes() / (1024 * 1024)));
    fflush(stderr);
    return ptr;
}

void ggml_backend_cuda_prefill_pinned_free(void * ptr) {
    if (!ptr) {
        return;
    }
    size_t sz = 0;
    {
        std::lock_guard<std::mutex> lk(ggml_cuda_pin_size_mtx());
        auto & m = ggml_cuda_pin_size_map();
        auto it = m.find(ptr);
        if (it != m.end()) {
            sz = it->second;
            m.erase(it);
        }
    }
    cudaFreeHost(ptr);
    if (sz > 0) {
        ggml_cuda_pinned_free_track(sz);
    }
    fprintf(stderr, "*** PIN_FREE: %p -%.1fMB (cum_alloc=%lldMB cum_free=%lldMB)\n",
            ptr, sz / (1024.0 * 1024.0),
            (long long)(ggml_cuda_pinned_alloc_total_bytes() / (1024 * 1024)),
            (long long)(ggml_cuda_pinned_free_total_bytes() / (1024 * 1024)));
    fflush(stderr);
}

void ggml_backend_cuda_prefill_upload_async(void * pipeline,
                                             void * dst_device,
                                             const void * src_pinned,
                                             size_t size) {
    auto * p = static_cast<ggml_cuda_prefill_pipeline *>(pipeline);
    // Check for stale CUDA errors before enqueuing — a previous streaming
    // prefill may have left a latent error that would confusingly surface here.
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
        GGML_LOG_WARN("%s: clearing stale CUDA error before upload: %s\n",
                      __func__, cudaGetErrorString(prev_err));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst_device, src_pinned, size,
                               cudaMemcpyHostToDevice, p->copy_stream));
}

void ggml_backend_cuda_prefill_wait_upload(void * pipeline) {
    auto * p = static_cast<ggml_cuda_prefill_pipeline *>(pipeline);
    // Record where the copy stream is, then make the compute stream wait
    CUDA_CHECK(cudaEventRecord(p->upload_done, p->copy_stream));
    CUDA_CHECK(cudaStreamWaitEvent(p->compute_stream, p->upload_done, 0));
}

void ggml_backend_cuda_prefill_sync_copy_stream(void * pipeline) {
    auto * p = static_cast<ggml_cuda_prefill_pipeline *>(pipeline);
    CUDA_CHECK(cudaStreamSynchronize(p->copy_stream));
}

// [2026-07-28 STAGING_SYNC probe — GLM-DSA multi-ubatch race A/B] These helpers issue
// SYNCHRONOUS cudaMemcpy on the legacy NULL stream. ggml's compute streams are created
// cudaStreamNonBlocking (common.cuh:1476), so the legacy stream has NO ordering edge
// against them: a staging copy can overwrite gallocr-reused device memory while an
// in-flight compute kernel still reads it (consumer-reads-while-writer-writes — the
// serialization-masked IMA class). LLAMA_PREFILL_STAGING_SYNC=1 drains the device
// before each such copy to A/B-confirm the class; default OFF = prior behavior.
// Marker string for post-build DLL grep: STAGING_SYNC_PROBE_V1
static bool ggml_cuda_staging_sync_enabled(void) {
    static const bool en = [] {
        const char * e = getenv("LLAMA_PREFILL_STAGING_SYNC");
        const bool on = e != nullptr && e[0] != '\0' && e[0] != '0';
        if (on) {
            fprintf(stderr, "*** STAGING_SYNC_PROBE_V1 ACTIVE: device-drain before sync staging memcpys\n");
            fflush(stderr);
        }
        return on;
    }();
    return en;
}

void ggml_backend_cuda_memcpy_d2d(void * dst, const void * src, size_t nbytes) {
    if (ggml_cuda_staging_sync_enabled()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
}

void ggml_backend_cuda_memcpy_d2h(void * dst_host, const void * src_device, size_t nbytes) {
    if (ggml_cuda_staging_sync_enabled()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(dst_host, src_device, nbytes, cudaMemcpyDeviceToHost));
}

void ggml_backend_cuda_memcpy_h2d(void * dst_device, const void * src_host, size_t nbytes) {
    if (ggml_cuda_staging_sync_enabled()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(dst_device, src_host, nbytes, cudaMemcpyHostToDevice));
}

void ggml_backend_cuda_memcpy_d2d_async(void * dst, const void * src, size_t nbytes, void * stream_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaMemcpyAsync(dst, const_cast<void*>(src), nbytes, cudaMemcpyDeviceToDevice, stream));
}

void ggml_backend_cuda_sync_stream(void * stream_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ggml_backend_cuda_device_synchronize(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Causal mask kernel — generates mask directly on GPU, replacing CPU fill + H2D copy.
// pos_base is the absolute position of the ubatch's first query token.
static __global__ void kernel_causal_mask(float * __restrict__ mask, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n_kv_dim || i >= n_tokens) return;
    const bool visible = j < n_kv_visible;
    mask[j + i * n_kv_dim] = (visible && j <= pos_base + i) ? 0.0f : -INFINITY;
}

static __global__ void kernel_causal_mask_f16(uint16_t * __restrict__ mask, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n_kv_dim || i >= n_tokens) return;
    const bool visible = j < n_kv_visible;
    mask[j + i * n_kv_dim] = (visible && j <= pos_base + i) ? uint16_t(0x0000) : uint16_t(0xfc00);
}

void ggml_cuda_generate_causal_mask(void * mask_gpu, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base, void * stream_ptr) {
    if (!mask_gpu || n_kv_dim <= 0 || n_kv_visible <= 0 || n_tokens <= 0) return;
    dim3 block(32, 8);
    dim3 grid((n_kv_dim + 31) / 32, (n_tokens + 7) / 8);
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    kernel_causal_mask<<<grid, block, 0, stream>>>((float *)mask_gpu, n_kv_dim, n_kv_visible, n_tokens, pos_base);
}

void ggml_cuda_generate_causal_mask_f16(void * mask_gpu, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base, void * stream_ptr) {
    if (!mask_gpu || n_kv_dim <= 0 || n_kv_visible <= 0 || n_tokens <= 0) return;
    dim3 block(32, 8);
    dim3 grid((n_kv_dim + 31) / 32, (n_tokens + 7) / 8);
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    kernel_causal_mask_f16<<<grid, block, 0, stream>>>((uint16_t *)mask_gpu, n_kv_dim, n_kv_visible, n_tokens, pos_base);
}

void ggml_backend_cuda_launch_host_fn(void * stream_ptr, ggml_cuda_host_fn fn, void * user_data) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaLaunchHostFunc(stream, (cudaHostFn_t)fn, user_data));
}

void * ggml_backend_cuda_event_create(void) {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return (void *)event;
}

void ggml_backend_cuda_event_destroy(void * event_ptr) {
    if (event_ptr) {
        CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event_ptr));
    }
}

void ggml_backend_cuda_event_record(void * event_ptr, void * stream_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event_ptr, stream));
}

void ggml_backend_cuda_event_synchronize(void * event_ptr) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event_ptr));
}

void ggml_backend_cuda_stream_wait_event(void * stream_ptr, void * event_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaStreamWaitEvent(stream, (cudaEvent_t)event_ptr, 0));
}

void * ggml_backend_cuda_device_malloc(size_t nbytes) {
    void * ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMalloc(%zu) failed: %s\n", __func__, nbytes, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void ggml_backend_cuda_device_free(void * ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Generate causal attention mask directly on GPU.
// mask[token_i * n_kv + kv_j] = (kv_j <= pos_base + token_i) ? 0.0f : -INFINITY
static __global__ void k_generate_causal_mask(float * mask, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x; // kv index
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // token index
    if (j >= n_kv_dim || i >= n_tokens) return;
    const bool visible = j < n_kv_visible;
    mask[i * n_kv_dim + j] = (visible && j <= pos_base + i) ? 0.0f : -INFINITY;
}

void ggml_backend_cuda_generate_causal_mask(void * mask_gpu, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base, void * stream_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    dim3 block(32, 8);
    dim3 grid((n_kv_dim + block.x - 1) / block.x, (n_tokens + block.y - 1) / block.y);
    k_generate_causal_mask<<<grid, block, 0, stream>>>((float *)mask_gpu, n_kv_dim, n_kv_visible, n_tokens, pos_base);
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemGetInfo(free, total));

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx->device));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__)

}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    if (ggml_ds4_native_half_requested(op)) {
        return ggml_ds4_native_half_shape_valid(op) && ggml_cuda_info().devices[dev_ctx->device].cc==1200;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->buffer && ggml_backend_buft_is_cuda_split(a->buffer->buft)) {
                    if (a->ne[2] > 1 || a->ne[3] > 1) {
                        return false;
                    }
                    // for small weight matrices the active device can end up without any rows, don't use row split in those cases
                    // this avoids some edge cases (and the performance would not be good anyways)
                    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) a->buffer->buft->context;
                    int64_t row_low;
                    int64_t row_high;
                    get_row_split(&row_low, &row_high, a, buft_ctx->tensor_split, dev_ctx->device);
                    if (row_low == row_high) {
                        return false;
                    }
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    case GGML_TYPE_Q4_K:  // added 2026-04-19 via k_get_rows_q4_K
                    case GGML_TYPE_Q5_K:
                        // K-quant get_rows kernels require ne00 to be a
                        // multiple of QK_K (256). Check row size fits.
                        return (op->src[0]->ne[0] % 256) == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                const ggml_type type = op->type;
                return op->src[0]->type == type &&
                       op->src[1]->type == type &&
                       ggml_blck_size(type) == 1;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SSM_SCAN: {
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_2D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_MAP_CUSTOM3:
            return ggml_dsv41_hc_expand_valid(op);
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_ACT_QAT:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_DSV41_ENGRAM_ADD:
        case GGML_OP_DSV41_POOL2:
            for (int i = 0; i < 4; ++i) {
                if (op->src[i] && (op->src[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(op->src[i]))) return false;
            }
            return op->op != GGML_OP_DSV41_ENGRAM_ADD || !op->src[4] ||
                (op->src[4]->type == GGML_TYPE_I8 && ggml_is_contiguous(op->src[4]));
        case GGML_OP_DSV41_ROPE:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[1]) &&
                op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
        case GGML_OP_DSV41_RAW_COMMIT:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]) &&
                op->src[2]->type == GGML_TYPE_F32;
        case GGML_OP_DSV41_INDEX_SCORE:
            for(int i=0;i<3;++i)if(op->src[i]->type!=GGML_TYPE_F32 || !ggml_is_contiguous(op->src[i]))return false;
            return op->src[3]->type==GGML_TYPE_I32 && ggml_is_contiguous(op->src[3]) &&
                (!op->src[4] || (op->src[4]->type==GGML_TYPE_F32 && ggml_is_contiguous(op->src[4])));
        case GGML_OP_DSV41_CANDIDATE_BLOCKS:
            return op->src[0]->type==GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1]->type==GGML_TYPE_I32 && ggml_is_contiguous(op->src[1]);
        case GGML_OP_DSV41_SELECTED_MASK:
            return op->src[0]->type==GGML_TYPE_I32 && ggml_is_contiguous(op->src[0]) && op->src[1]->type==GGML_TYPE_I32 && ggml_is_contiguous(op->src[1]) &&
                (!op->src[2] || (op->src[2]->type==GGML_TYPE_F32 && ggml_is_contiguous(op->src[2])));
        case GGML_OP_DSV41_MASKED_HALF:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_FLASH_ATTN_EXT_SPARSE:
            if(ggml_dsv41_token_tile_requested(op))return !GGML_CUDA_CC_IS_AMD(ggml_cuda_info().devices[dev_ctx->device].cc)&&
                ggml_cuda_info().devices[dev_ctx->device].cc>=GGML_CUDA_CC_AMPERE&&ggml_dsv41_token_tile_valid(op);
            if(ggml_dsv41_shared_sparse_requested(op))return ggml_dsv41_shared_sparse_valid(op);
            return op->type == GGML_TYPE_F32
                && op->src[0]->type == GGML_TYPE_F32
                && (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_Q8_0)
                && (op->src[2]->type == GGML_TYPE_F16 || op->src[2]->type == GGML_TYPE_Q8_0)
                && op->src[3]->type == GGML_TYPE_I32
                && (op->src[4] == nullptr || op->src[4]->type == GGML_TYPE_F16)
                && op->src[0]->ne[0] == op->src[1]->ne[0]
                && op->src[1]->ne[1] == 1
                && op->src[2]->ne[1] == 1
                && op->src[1]->ne[2] == op->src[2]->ne[2]
                && op->src[1]->ne[3] == 1
                && op->src[2]->ne[3] == 1
                && op->src[2]->ne[0] <= 1024
                && op->src[3]->ne[0] == ggml_get_op_params_i32(op, 2)
                && op->src[3]->ne[0] <= op->src[1]->ne[2]
                && op->src[3]->ne[1] >= op->src[0]->ne[2]*op->src[0]->ne[3]
                && (op->src[4] == nullptr ||
                    (op->src[4]->ne[0] == op->src[3]->ne[0] &&
                     (op->src[4]->ne[1] >= op->src[0]->ne[2]*op->src[0]->ne[3] ||
                      op->src[4]->ne[3] >= op->src[0]->ne[2]*op->src[0]->ne[3])));
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_DSA_INDEXER_SCORE:
            // shape constraints: D == 128 (matches GLM-DSA indexer head_size); H typically 32.
            // Strictly require contiguous F16 k_cache + F32 q + F32 gates.
            return op->src[0]->type == GGML_TYPE_F16
                && op->src[1]->type == GGML_TYPE_F32
                && op->src[2]->type == GGML_TYPE_F32
                && op->src[0]->ne[0] == 128
                && ggml_is_contiguous(op->src[0])
                && ggml_is_contiguous(op->src[1])
                && ggml_is_contiguous(op->src[2]);
        case GGML_OP_DSA_INDEXER_TOPK:
            return op->type == GGML_TYPE_I32
                && op->src[0]->type == GGML_TYPE_F16
                && op->src[1]->type == GGML_TYPE_F32
                && op->src[2]->type == GGML_TYPE_F32
                && op->src[0]->ne[0] == 128
                && op->ne[0] <= op->src[0]->ne[1]
                && ggml_is_contiguous(op->src[0])
                && ggml_is_contiguous(op->src[1])
                && ggml_is_contiguous(op->src[2])
                // [DSA_TOPK_CAUSAL] optional query-position src for causal score masking
                && (op->src[3] == nullptr ||
                    (op->src[3]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[3])));

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (((ggml_backend_buft_is_cuda(buft) || ggml_backend_buft_is_cuda_split(buft)) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft)));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
        features.push_back({ "PEER_MAX_BATCH_SIZE", STRINGIFY(GGML_CUDA_PEER_MAX_BATCH_SIZE) });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_cuda_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_cuda_diffusion_sample") == 0) {
        return (void *)ggml_cuda_diffusion_sample;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);

                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                char pci_bus_id[16] = {};
                snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                dev_ctx->pci_bus_id = pci_bus_id;
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    // One-time profiling initialisation (safe to call multiple times).
    static std::once_flag s_profile_init;
    std::call_once(s_profile_init, [](){
        const char * env = getenv("LLAMA_CUDA_OP_PROFILE");
        if (env && env[0] == '1') {
            g_op_profile_enabled = true;
            const char *nodes=getenv("LLAMA_CUDA_OP_PROFILE_NODES");g_node_profile_enabled=nodes&&nodes[0]=='1';
            std::atexit(cuda_op_profile_dump);
            fprintf(stderr, "[CUDA OP PROFILE] enabled — will dump at exit\n");
            fflush(stderr);  // 2026-07-10: flush so this enable-marker is observable right after startup (env-propagation check)
        }
    });

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)

#if defined(_WIN32)
#include "ds4split-projection-fixture.cuh"
#endif
