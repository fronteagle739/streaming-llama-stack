#include "mmq_id_common.cuh"
#include "mmq_id.cuh"
#include "ggml-dsv41-expert-routing.h"
#include "quantize_id.cuh"
// NOTE (iter-33 port): DO NOT include "quantize.cuh" here — it includes mmq.cuh which redefines
// all the load_tiles/vec_dot functions already defined in mmq_id_common.cuh, causing ODR violations.
// Forward-declare only the one function we need, using OUR tree's 13-param signature.
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

#include <vector>
#include <climits>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mmq_ids_helper_store {
    uint32_t data;

    __device__ mmq_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mmq_ids_helper_store) == 4, "unexpected size for mmq_ids_helper_store");

namespace {

// Routing is stable for a given expert weight tensor across ubatches in the measured
// DSV4 workload. Cache the one-time D2H classification by weight address so the hot
// path remains entirely asynchronous after the first ubatch.
struct mmq_id_tile_cache_key {
    const void * data;
    int device;

    bool operator==(const mmq_id_tile_cache_key & other) const {
        return data == other.data && device == other.device;
    }
};

struct mmq_id_tile_cache_key_hash {
    size_t operator()(const mmq_id_tile_cache_key & key) const {
        return std::hash<const void *>{}(key.data) ^ (std::hash<int>{}(key.device) << 1);
    }
};

struct mmq_id_tile_cache_entry {
    int tile = 0; // Zero means use the generic selector.
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    int64_t ne02 = 0;
    int64_t ne12 = 0;
    uint32_t uses_since_measure = 0;
    uint64_t last_used = 0;
};

static std::mutex mmq_id_tile_cache_mutex;
static std::unordered_map<mmq_id_tile_cache_key, mmq_id_tile_cache_entry,
        mmq_id_tile_cache_key_hash> mmq_id_tile_cache;
static uint64_t mmq_id_tile_cache_clock = 0;

static constexpr size_t MMQ_ID_TILE_CACHE_MAX_ENTRIES = 512;
static constexpr uint32_t MMQ_ID_TILE_CACHE_RECHECK_USES = 256;

static bool mmq_id_env_enabled(const char * name, bool default_value) {
    const char * value = getenv(name);
    return value ? value[0] != '0' : default_value;
}

static int mmq_id_env_int(const char * name, int default_value) {
    const char * value = getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    return end != value && parsed >= 0 && parsed <= INT_MAX ? (int) parsed : default_value;
}

static int mmq_id_choose_tile(const std::vector<int32_t> & bounds) {
    const int64_t n_experts = (int64_t) bounds.size() - 1;
    if (n_experts <= 0) {
        return 0;
    }

    int64_t cols_total = 0;
    int n_nonzero = 0;
    for (int64_t i = 0; i < n_experts; ++i) {
        const int cols = bounds[i + 1] - bounds[i];
        cols_total += cols;
        n_nonzero += cols > 0;
    }

    // Regime A from the production trace: at least 75% of experts receive work,
    // while each non-empty expert averages at most 24 columns. Mid/collapsed
    // routing deliberately falls through to the generic large-tile selector.
    if (n_nonzero * 4 < n_experts * 3 || cols_total > (int64_t) n_nonzero * 24) {
        return 0;
    }

    // Cost is padded work adjusted for measured small-tile efficiency plus an
    // eight-column-equivalent scheduling/fixup cost per logical expert block.
    // SM120 measurements on the DSV4 Q2_K/Q3_K pair calibrate the relative
    // kernel-efficiency scales: 16 -> 55%, 24 -> 70%, 32 -> 100%.
    static constexpr int candidates[] = {16, 24, 32};
    static constexpr int efficiency[] = {55, 70, 100};
    int best_tile = 0;
    int64_t best_cost = INT64_MAX;
    for (int ci = 0; ci < 3; ++ci) {
        const int tile = candidates[ci];
        int64_t blocks = 0;
        for (int64_t i = 0; i < n_experts; ++i) {
            const int cols = bounds[i + 1] - bounds[i];
            blocks += (cols + tile - 1) / tile;
        }
        const int64_t padded_cols = blocks * tile;
        const int64_t cost = (padded_cols * 100 + efficiency[ci] - 1) / efficiency[ci] + blocks * 8;
        if (cost < best_cost) {
            best_cost = cost;
            best_tile = tile;
        }
    }
    return best_tile;
}

static int mmq_id_adaptive_tile(
        const ggml_tensor * src0, const int32_t * expert_bounds,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne12,
        int cc, cudaStream_t stream, std::vector<int32_t> * measured_bounds) {
    static const bool enabled = mmq_id_env_enabled("LLAMA_MMQ_ID_ADAPTIVE", true);
    if (!enabled || !GGML_CUDA_CC_IS_NVIDIA(cc) ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_BLACKWELL ||
            (src0->type != GGML_TYPE_Q2_K && src0->type != GGML_TYPE_Q3_K)) {
        return 0;
    }

    static const int force_tile = mmq_id_env_int("LLAMA_MMQ_ID_ADAPTIVE_FORCE_TILE", 0);
    if (force_tile != 0) {
        GGML_ASSERT(force_tile == 16 || force_tile == 24 || force_tile == 32);
        return force_tile;
    }

    const mmq_id_tile_cache_key key = {src0->data, ggml_cuda_get_device()};
    static const int recheck_uses = mmq_id_env_int(
        "LLAMA_MMQ_ID_ADAPTIVE_RECHECK", MMQ_ID_TILE_CACHE_RECHECK_USES);

    {
        std::lock_guard<std::mutex> lock(mmq_id_tile_cache_mutex);
        const auto it = mmq_id_tile_cache.find(key);
        if (it != mmq_id_tile_cache.end()) {
            mmq_id_tile_cache_entry & entry = it->second;
            entry.last_used = ++mmq_id_tile_cache_clock;
            const bool same_shape = entry.type == src0->type && entry.ne00 == ne00 &&
                entry.ne01 == ne01 && entry.ne02 == ne02 && entry.ne12 == ne12;
            if (same_shape && (recheck_uses == 0 || entry.uses_since_measure++ < (uint32_t) recheck_uses)) {
                return entry.tile;
            }
        }
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return 0;
    }

    measured_bounds->resize(ne02 + 1);
    CUDA_CHECK(cudaMemcpyAsync(measured_bounds->data(), expert_bounds,
        (ne02 + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int tile = mmq_id_choose_tile(*measured_bounds);

    {
        std::lock_guard<std::mutex> lock(mmq_id_tile_cache_mutex);
        if (mmq_id_tile_cache.find(key) == mmq_id_tile_cache.end() &&
                mmq_id_tile_cache.size() >= MMQ_ID_TILE_CACHE_MAX_ENTRIES) {
            auto oldest = mmq_id_tile_cache.begin();
            for (auto it = mmq_id_tile_cache.begin(); it != mmq_id_tile_cache.end(); ++it) {
                if (it->second.last_used < oldest->second.last_used) {
                    oldest = it;
                }
            }
            mmq_id_tile_cache.erase(oldest);
        }
        mmq_id_tile_cache[key] = {
            tile, src0->type, ne00, ne01, ne02, ne12, 0, ++mmq_id_tile_cache_clock};
    }

    static const bool trace = mmq_id_env_enabled("LLAMA_MMQ_ID_ADAPTIVE_TRACE", false);
    if (trace) {
        int n_nonzero = 0;
        int64_t cols_total = 0;
        for (int64_t i = 0; i < ne02; ++i) {
            const int cols = (*measured_bounds)[i + 1] - (*measured_bounds)[i];
            n_nonzero += cols > 0;
            cols_total += cols;
        }
        fprintf(stderr, "MMQ_ID_ADAPT data=%p type=%s experts=%lld tokens=%lld nz=%d mean_nz=%.1f tile=%d\n",
            src0->data, ggml_type_name(src0->type), (long long) ne02, (long long) ne12,
            n_nonzero, n_nonzero ? (double) cols_total / n_nonzero : 0.0, tile);
    }
    return tile;
}

} // namespace

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mmq_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mmq_ids_helper[];
    mmq_ids_helper_store * store = (mmq_ids_helper_store *) data_mmq_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mmq_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= offset) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mmq_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mmq_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < gridDim.x - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mmq_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mmq_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mmq_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_get_physical_warp_size_host(); //ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mmq_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mmq_ids_helper_store);
    mmq_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

static void ggml_cuda_mul_mat_q_switch_type_id(ggml_backend_cuda_context & ctx, const mmq_args_id & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case_id<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case_id<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case_id<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case_id<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        // case GGML_TYPE_Q6_0: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_Q6_0> /* not supported in this tree */(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case_id<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case_id<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case_id<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case_id<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case_id<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case_id<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case_id<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case_id<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case_id<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case_id<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case_id<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case_id<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case_id<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ1_S_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ1_S_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case_id<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case_id<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ2_KS: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ2_KS> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ2_KL: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ2_KL> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ2_K: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ2_K> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ2_K_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ2_K_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ3_K: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ3_K> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ3_K_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ3_K_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ3_KS: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ3_KS> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ4_KSS: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ4_KSS> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ4_KS: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ4_KS> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ4_KS_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ4_KS_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ4_K: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ4_K> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ4_K_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ4_K_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ5_KS: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ5_KS> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ5_KS_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ5_KS_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ5_K: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ5_K> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ5_K_R4: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ5_K_R4> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ6_K: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ6_K> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ1_KT: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ1_KT> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ2_KT: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ2_KT> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ3_KT: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ3_KT> /* not supported in this tree */(ctx, args, stream);
            break;
        // case GGML_TYPE_IQ4_KT: /* not supported in this tree */
            // mul_mat_q_case_id<GGML_TYPE_IQ4_KT> /* not supported in this tree */(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Reused from ds4-win b12b8ab3248293e12a987f21769f3ed9656f6b3c; see source receipt.
__global__ static void dsv41_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert, uint32_t used, uint32_t si1) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[(pair/used)*si1+pair%used];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void dsv41_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[n_total_expert] = sum;
    }
}

__global__ static void dsv41_scatter_sorted_pairs_kernel(
        int32_t *ids_src1, int32_t *ids_dst,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert, uint32_t used, uint32_t si1, uint32_t channels, uint32_t sis1) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[(pair/used)*si1+pair%used];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    ids_src1[pos] = (pair/used)*sis1+(pair%used)%channels;
    ids_dst[pos] = pair;
}


void compute_row_ids_count_prefix_scatter(ggml_backend_cuda_context &ctx,const int32_t *ids,int32_t *ids_src1,int32_t *ids_dst,int32_t *expert_bounds,
        int64_t ne02,int64_t ne12,int64_t used,int64_t ne11,int64_t nb11,int64_t nb12,int64_t nb21,cudaStream_t stream){
    GGML_ASSERT(ne02>0&&ne02<=1024&&ne12>0&&ne12<=1048576&&used>0&&used<=32&&ne11>0&&ne11<=used&&nb11>0&&nb12%nb11==0&&nb21%4==0);
    const uint32_t pairs=uint32_t(ne12*used),experts=uint32_t(ne02),si1=uint32_t(nb21/4),sis1=uint32_t(nb12/nb11);
    ggml_cuda_pool_alloc<uint32_t> counts(ctx.pool(),experts),cursors(ctx.pool(),experts);
    CUDA_CHECK(cudaMemsetAsync(counts.get(),0,experts*sizeof(uint32_t),stream));
    dsv41_count_sorted_pairs_kernel<<<(pairs+255)/256,256,0,stream>>>(counts.get(),ids,pairs,experts,uint32_t(used),si1);
    dsv41_prefix_sorted_pairs_kernel<<<1,32,0,stream>>>(reinterpret_cast<uint32_t*>(expert_bounds),cursors.get(),counts.get(),experts);
    dsv41_scatter_sorted_pairs_kernel<<<(pairs+255)/256,256,0,stream>>>(ids_src1,ids_dst,cursors.get(),ids,pairs,experts,uint32_t(used),si1,uint32_t(ne11),sis1);
    CUDA_CHECK(cudaGetLastError());
    std::fprintf(stderr,"DS4PERF_SORTED_ROUTING tokens=%lld experts=%lld used=%lld shared_bytes=0 temporary_bytes=%llu ds4_count_prefix_scatter=1 atomic_pair_order=1\n",(long long)ne12,(long long)ne02,(long long)used,(unsigned long long)2*experts*sizeof(uint32_t));
}

namespace {
struct dsv41_routing_map_state{
    ggml_backend_cuda_context&ctx;ggml_tensor*products[3];const ggml_tensor*ids=nullptr;const void*data=nullptr;
    ggml_cuda_pool_alloc<char>gate,down;size_t pairs=0,bytes=0;unsigned calls=0;
    dsv41_routing_map_state(ggml_backend_cuda_context&c,ggml_tensor**nodes):ctx(c),gate(c.pool()),down(c.pool()){
        for(unsigned i=0;i<3;++i)products[i]=nodes[i];ids=nodes[0]->src[2];pairs=size_t(ids->ne[0])*ids->ne[1];bytes=(2*pairs+385)*sizeof(int32_t);
        gate.alloc(bytes);down.alloc(bytes);
    }
};
thread_local dsv41_routing_map_state*active_dsv41_map=nullptr;
}
ggml_cuda_dsv41_routing_scope::ggml_cuda_dsv41_routing_scope(ggml_backend_cuda_context&ctx,ggml_cgraph*g,bool uncaptured){
    bool any=false;for(int i=0;i<ggml_graph_n_nodes(g);++i)any|=ggml_dsv41_routing_map_reuse_requested(ggml_graph_node(g,i));
    if(!any)return;
    ggml_tensor*nodes[3]={};GGML_ASSERT(uncaptured&&!active_dsv41_map&&ctx.curr_stream_no==0&&ggml_dsv41_routing_triplet(g,nodes));
    auto*value=new dsv41_routing_map_state(ctx,nodes);active_dsv41_map=value;state=value;
}
ggml_cuda_dsv41_routing_scope::~ggml_cuda_dsv41_routing_scope(){
    if(!state)return;auto*s=static_cast<dsv41_routing_map_state*>(state);
    GGML_ASSERT(active_dsv41_map==s&&s->ctx.curr_stream_no==0&&s->calls==3);
    std::fprintf(stderr,"DS4PERF_ROUTING_SCOPE_COMPLETE tokens=%lld products=3 builders=1 encoded_maps=2 stream=0 graph_lifetime=1\n",(long long)s->ids->ne[1]);
    // Every consumer was queued on this same stream before returning pool credit.
    // Original stream-local pool ordering protects the still-asynchronous kernels.
    active_dsv41_map=nullptr;delete s;
}
char*ggml_cuda_dsv41_routing_map(ggml_backend_cuda_context&ctx,const ggml_tensor*dst){
    auto*s=active_dsv41_map;GGML_ASSERT(s&&&s->ctx==&ctx&&ctx.curr_stream_no==0&&s->calls<3&&s->products[s->calls]==dst&&dst->src[2]==s->ids);
    const auto*src=dst->src[1];const auto*ids=dst->src[2];auto*base=reinterpret_cast<int32_t*>(s->gate.get());
    if(!s->calls){
        s->data=ids->data;
        compute_row_ids_count_prefix_scatter(ctx,static_cast<const int32_t*>(ids->data),base,base+s->pairs,base+2*s->pairs,
            384,ids->ne[1],6,src->ne[1],src->nb[1],src->nb[2],ids->nb[1],ctx.stream());
        // The down source row is the canonical pair ID. Derive its existing
        // contiguous API encoding from this SAME built permutation and bounds.
        CUDA_CHECK(cudaMemcpyAsync(s->down.get(),base+s->pairs,s->pairs*4,cudaMemcpyDeviceToDevice,ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(s->down.get()+s->pairs*4,base+s->pairs,(s->pairs+385)*4,cudaMemcpyDeviceToDevice,ctx.stream()));
    }else GGML_ASSERT(ids->data==s->data);
    std::fprintf(stderr,"DS4PERF_ROUTING_MAP_USE tokens=%lld product=%u reused=%u same_ID_source=1 stream=0\n",(long long)ids->ne[1],s->calls,unsigned(s->calls>0));
    return s->calls++==2?s->down.get():s->gate.get();
}

void compute_row_ids(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int64_t ne02, int64_t ne12, int64_t n_expert_used, int64_t ne11, int64_t nb11, int64_t nb12, int64_t nb21,
        cudaStream_t stream) {

    const int si1  = nb21 / sizeof(int);
    const int sis1 = nb12 / nb11;

    switch (n_expert_used) {
        case  2:
            launch_mmq_ids_helper< 2> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case  4:
            launch_mmq_ids_helper< 4> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case  6:
            launch_mmq_ids_helper< 6> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case  8:
            launch_mmq_ids_helper< 8> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case 16:
            launch_mmq_ids_helper<16> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case 32:
            launch_mmq_ids_helper<32> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        default:
            launch_mmq_ids_helper< 0> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_q_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids_tensor, ggml_tensor * dst, char * ids_data, char * src1_quantized_data) {
    GGML_ASSERT(       src1->type == GGML_TYPE_F32);
    GGML_ASSERT(       dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids_tensor || ids_tensor->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    //const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    //GGML_ASSERT(       nb00       == ts_src0);
    GGML_ASSERT(       nb10       == ts_src1);
    GGML_ASSERT(       nb0        == ts_dst);
    GGML_ASSERT(!ids_tensor || ids_tensor->nb[0] == ggml_type_size(ids_tensor->type));

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1];// / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2];// / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3];// / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    if (!ids_tensor) {

        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        if (!src1_quantized_data) {
            const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1
                                          + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
            src1_q8_1.alloc(nbytes_src1_q8_1);
            // NOTE (iter-33 port): adapted to our tree's 13-param quantize_mmq_q8_1_cuda API.
            // Original ik_llama 8-param call: (src1_d, vy, ne10, ne11, 1, ne10_padded, type, stream)
            quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
                ne10, ne10, ne10*ne11, ne10*ne11, ne10_padded, ne11, 1, 1, stream);
            CUDA_CHECK(cudaGetLastError());
            src1_quantized_data = src1_q8_1.get();
        }

        const int64_t s12 = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args_id args = {
            src0_d, src0->type, (const int *)src1_quantized_data, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1, 0};

        ggml_cuda_mul_mat_q_switch_type_id(ctx, args, stream);
        return;
    }

    const int64_t n_expert_used = ids_tensor->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1_local(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst_local(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds_local(ctx.pool());

    int32_t * ids_src1, *ids_dst, *expert_bounds;
    if (ids_data) {
        ids_src1 = (int32_t *)ids_data;
        ids_dst  = ids_src1 + ne_get_rows;
        expert_bounds = ids_dst + ne_get_rows;
    }
    else {
        GGML_ASSERT(ids_tensor->nb[0] == ggml_element_size(ids_tensor));

        ids_src1_local.alloc(ne_get_rows);
        ids_dst_local.alloc(ne_get_rows);
        expert_bounds_local.alloc(ne02 + 1);

        ids_src1 = ids_src1_local.get();
        ids_dst  = ids_dst_local.get();
        expert_bounds = expert_bounds_local.get();

        const int si1  = ids_tensor->nb[1] / ggml_element_size(ids_tensor);
        const int sis1 = nb12 / nb11;

        switch (n_expert_used) {
            case  2:
                launch_mmq_ids_helper< 2> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case  4:
                launch_mmq_ids_helper< 4> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case  6:
                launch_mmq_ids_helper< 6> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case  8:
                launch_mmq_ids_helper< 8> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case 16:
                launch_mmq_ids_helper<16> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case 32:
                launch_mmq_ids_helper<32> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            default:
                launch_mmq_ids_helper< 0> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<int32_t> measured_bounds;
    const int adaptive_mmq_x = mmq_id_adaptive_tile(
        src0, expert_bounds, ne00, ne01, ne02, ne12, cc, stream, &measured_bounds);

    // Diagnostic: per-expert routed-column histogram (LLAMA_MMQ_ID_COL_HISTO=1).
    // Gates any tile re-tune: the 2026-07-12 sparse-tile refutation showed uniform synthetic
    // routing in the harness inverts on real (skewed) routing, same GPU. Prints first 24 calls
    // then every 100th. D2H sync per printed call - diagnostics only, never leave enabled.
    {
        static const bool histo_enabled = [] {
            const char * v = getenv("LLAMA_MMQ_ID_COL_HISTO");
            return v && v[0] == '1';
        }();
        if (histo_enabled) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            static std::atomic<int> histo_call {0};
            const int call = histo_call.fetch_add(1, std::memory_order_relaxed);
            if (cap == cudaStreamCaptureStatusNone && (call < 24 || call % 100 == 0)) {
                std::vector<int32_t> hb;
                if (!measured_bounds.empty()) {
                    hb = measured_bounds;
                } else {
                    hb.resize(ne02 + 1);
                    CUDA_CHECK(cudaMemcpyAsync(hb.data(), expert_bounds, (ne02 + 1)*sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                std::vector<int> cols(ne02);
                for (int64_t i = 0; i < ne02; i++) {
                    cols[i] = hb[i + 1] - hb[i];
                }
                std::vector<int> nz;
                nz.reserve(ne02);
                int cmax = 0;
                int64_t csum = 0;
                for (int c : cols) {
                    if (c > 0) { nz.push_back(c); }
                    cmax  = c > cmax ? c : cmax;
                    csum += c;
                }
                std::sort(nz.begin(), nz.end());
                const int n_nz  = (int) nz.size();
                const int p50   = n_nz ? nz[n_nz/2] : 0;
                const int p90   = n_nz ? nz[(int)(n_nz*0.90)] : 0;
                const int p99   = n_nz ? nz[(int)(n_nz*0.99)] : 0;
                // Tile-relevant: total blocks a given mmq_x would launch across experts.
                auto blocks_at = [&](int tx) {
                    int64_t b = 0;
                    for (int c : cols) { b += (c + tx - 1)/tx; }
                    return (long long) b;
                };
                fprintf(stderr, "MMQ_ID_HISTO call=%d type=%s experts=%lld used=%lld tokens=%lld "
                    "cols_total=%lld nz=%d mean_nz=%.1f p50=%d p90=%d p99=%d max=%d "
                    "blocks@8=%lld @16=%lld @24=%lld @32=%lld @64=%lld @88=%lld\n",
                    call, ggml_type_name(src0->type), (long long) ne02, (long long) n_expert_used,
                    (long long) ne12, (long long) csum, n_nz, n_nz ? (double) csum/n_nz : 0.0,
                    p50, p90, p99, cmax,
                    blocks_at(8), blocks_at(16), blocks_at(24), blocks_at(32), blocks_at(64), blocks_at(88));
            }
        }
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    const size_t nbytes_src1_q8_1 = ne11_flat*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    char * src1_q8_1;

    if (src1_quantized_data) {
        src1_q8_1 = src1_quantized_data;
    } else {

        src1_q8_1_local.alloc(nbytes_src1_q8_1);
        src1_q8_1 = src1_q8_1_local.get();

        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_mmq_q8_1_cuda_id(src1_d, ids_src1, src1_q8_1, src0->type,
            ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args_id args = {
        src0_d, src0->type, (const int *) src1_q8_1, ids_dst, expert_bounds, dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12, adaptive_mmq_x};

    //printf("ne00 = %ld, ne01 = %ld, ne_get_rows = %ld, s01 = %ld, s1 = %ld\n", ne00, ne01, ne_get_rows, s01, s1);
    //printf("ne02 = %ld, s02 = %ld, s12 = %ld, s2 = %ld\n", ne02, s02, s12, s2);
    //printf("ne03 = %ld, s03 = %ld, s13 = %ld, s3 = %ld\n", ne03, s03, s13, s3);

    ggml_cuda_mul_mat_q_switch_type_id(ctx, args, stream);
}

bool ggml_cuda_can_use_mmq_id(enum ggml_type type, int cc, int64_t ne11) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type);
    GGML_UNUSED(cc);
    GGML_UNUSED(ne11);
    return false;
#endif

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        // case GGML_TYPE_Q6_0: /* not supported in this tree */
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        // case GGML_TYPE_IQ1_S_R4: /* not supported in this tree */
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        // case GGML_TYPE_IQ2_KS: /* not supported in this tree */
        // case GGML_TYPE_IQ2_KL: /* not supported in this tree */
        // case GGML_TYPE_IQ2_K: /* not supported in this tree */
        // case GGML_TYPE_IQ2_K_R4: /* not supported in this tree */
        // case GGML_TYPE_IQ3_KS: /* not supported in this tree */
        // case GGML_TYPE_IQ3_K: /* not supported in this tree */
        // case GGML_TYPE_IQ3_K_R4: /* not supported in this tree */
        // case GGML_TYPE_IQ4_KSS: /* not supported in this tree */
        // case GGML_TYPE_IQ4_KS: /* not supported in this tree */
        // case GGML_TYPE_IQ4_KS_R4: /* not supported in this tree */
        // case GGML_TYPE_IQ4_K: /* not supported in this tree */
        // case GGML_TYPE_IQ4_K_R4: /* not supported in this tree */
        // case GGML_TYPE_IQ5_KS: /* not supported in this tree */
        // case GGML_TYPE_IQ5_KS_R4: /* not supported in this tree */
        // case GGML_TYPE_IQ5_K: /* not supported in this tree */
        // case GGML_TYPE_IQ5_K_R4: /* not supported in this tree */
        // case GGML_TYPE_IQ6_K: /* not supported in this tree */
        // case GGML_TYPE_IQ1_KT: /* not supported in this tree */
        // case GGML_TYPE_IQ2_KT: /* not supported in this tree */
        // case GGML_TYPE_IQ3_KT: /* not supported in this tree */
        // case GGML_TYPE_IQ4_KT: /* not supported in this tree */
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

   if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        // NOTE (iter-33 port): GGML_TYPE_Q6_0 not in our tree; omitted from condition.
        if (ne11 <= 128 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0
                        || type == GGML_TYPE_Q5_1 /* || type == GGML_TYPE_Q6_0 */) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    return (!GGML_CUDA_CC_IS_RDNA4(cc) && !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;

}
