// DSA fused indexer-scoring kernel — replaces the HEAD_CHUNK loop in
// src/models/deepseek2.cpp:504-558 (~50 ggml ops × 78 layers × 22 ubatches
// at 40k = tens of thousands of kernel launches → 281 s of GPU time on
// RTX 5090). Fuses mul_mat → scale → relu → mul → sum_rows into a single
// CUDA kernel call per (layer, ubatch).
//
// Input shapes (contiguous):
//   k_cache: F16 [D, n_kv]         (idx_k_cache, post-RMS-norm + RoPE)
//   q_full:  F32 [D, H, n_q]       (idx_q_full, post-RoPE+concat)
//   gates:   F32 [H, n_q]          (head_gates, already pre-scaled by 1/sqrt(H))
// Output:
//   scores:  F32 [n_kv, n_q]
// Param:
//   softmax_scale = 1/sqrt(D)      (in dst->op_params[0])
//
// Math (per output cell):
//   scores[k_t, q_t]
//     = sum_h(relu(softmax_scale * sum_d(q_full[d,h,q_t] * k_cache[d,k_t])) * gates[h, q_t])
//
// v2 layout (one warp = one (k_t, q_t) cell):
//   - Block dim = 32 threads = 1 warp
//   - Grid = (n_kv, n_q) blocks
//   - Each thread holds D/32 d-elements in registers (D=128 → 4 elements/thread)
//   - Per-h: per-thread macs → warp-shuffle reduce → lane 0 applies relu + gate +
//     accumulates the score in a register
//   - No __syncthreads() in the inner loop (warp-only). No __shared__ memory.
//
// Versus v1 (which used 128 threads + smem + 64 syncthreads per block per ubatch)
// this is ~30× lighter per cell while staying fully correct.

#include "dsa-indexer-score.cuh"
#include "convert.cuh"
#include "ggml-dsv41-index-cublas.h"
#include "../ggml-dsv41-ops.h"
#include <atomic>
#include <algorithm>
#include <math_constants.h>
#include <climits>

#include <cfloat>
#include <cstdint>
#include <vector>
#include "dsv41-index-fp4.cuh"

#define DSA_INDEXER_SCORE_D       128
#define DSA_INDEXER_SCORE_BLOCK   32   // = 1 warp
#define DSA_INDEXER_SCORE_LANES   (DSA_INDEXER_SCORE_D / DSA_INDEXER_SCORE_BLOCK)  // 4
#define DSA_INDEXER_TOPK_BLOCK    256
#define DSA_INDEXER_TOPK_WARPS    8
#define DSA_INDEXER_GEMM_HEAD_CHUNK 8

template<int D, int BLOCK_DIM, int LANES>
static __global__ void dsa_indexer_score_kernel_v2(
        const half  * __restrict__ K,       // [D, n_kv]  fp16
        const float * __restrict__ Q,       // [D, H, n_q] fp32
        const float * __restrict__ GATES,   // [H, n_q]    fp32
        float       * __restrict__ S,       // [n_kv, n_q] fp32
        const int H,
        const int n_kv,
        const int n_q,
        const float scale) {
    static_assert(BLOCK_DIM == 32, "v2 kernel uses one warp per cell");
    static_assert(D == BLOCK_DIM * LANES, "D must equal BLOCK_DIM * LANES");

    const int kt  = blockIdx.x;
    const int qt  = blockIdx.y;
    const int tid = threadIdx.x;  // 0..31

    if (kt >= n_kv || qt >= n_q) return;

    // Load D/32 k-elements into thread registers (one fp16 → fp32 conversion each)
    float my_sk[LANES];
    #pragma unroll
    for (int i = 0; i < LANES; ++i) {
        const int d = tid + i * BLOCK_DIM;
        my_sk[i] = __half2float(K[(int64_t)d + (int64_t)kt * D]);
    }

    float my_score = 0.0f;

    for (int h = 0; h < H; ++h) {
        // Per-thread partial dot
        float p = 0.0f;
        #pragma unroll
        for (int i = 0; i < LANES; ++i) {
            const int d = tid + i * BLOCK_DIM;
            const float qv = Q[(int64_t)d + (int64_t)h * D + (int64_t)qt * D * H];
            p += qv * my_sk[i];
        }

        // Warp-shuffle reduction (no smem, no sync)
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            p += __shfl_down_sync(0xffffffff, p, o);
        }
        // After the reduce, lane 0 holds the full dot.
        if (tid == 0) {
            float dot = p * scale;
            if (dot > 0.0f) {
                my_score += dot * GATES[(int64_t)h + (int64_t)qt * H];
            }
        }
    }

    if (tid == 0) {
        S[(int64_t)kt + (int64_t)qt * n_kv] = my_score;
    }
}

// v2b kernel: 4 q-cells per block sharing 1 K vector via shmem.
// Block: 4 warps (128 threads); each warp w handles q_t = qt_base + w.
// Reduces grid count 4× and K bandwidth 4× vs v2. Q loads unchanged.
template<int D, int Q_TILE>
static __global__ void dsa_indexer_score_kernel_v2b(
        const half  * __restrict__ K,
        const float * __restrict__ Q,
        const float * __restrict__ GATES,
        float       * __restrict__ S,
        const int H,
        const int n_kv,
        const int n_q,
        const float scale) {
    static_assert(Q_TILE == 4, "v2b kernel uses Q_TILE=4");
    const int kt      = blockIdx.x;
    const int qt_base = blockIdx.y * Q_TILE;
    const int tid     = threadIdx.x;       // 0..127
    const int warp    = tid >> 5;          // 0..3 (which q within tile)
    const int lane    = tid & 31;          // 0..31 (D-lane within warp)

    if (kt >= n_kv) return;
    const int qt = qt_base + warp;

    // Cooperatively load K[*, kt] into shmem (128 fp16 → fp32, 1 per thread).
    __shared__ float sK[D];
    sK[tid] = __half2float(K[(int64_t)tid + (int64_t)kt * D]);
    __syncthreads();

    if (qt >= n_q) return;

    // Each warp processes its assigned qt. lane = D-element index.
    constexpr int LANES = D / 32;
    float my_sk[LANES];
    #pragma unroll
    for (int i = 0; i < LANES; ++i) {
        my_sk[i] = sK[lane + i * 32];
    }

    float my_score = 0.0f;
    for (int h = 0; h < H; ++h) {
        float p = 0.0f;
        #pragma unroll
        for (int i = 0; i < LANES; ++i) {
            const int d = lane + i * 32;
            const float qv = Q[(int64_t)d + (int64_t)h * D + (int64_t)qt * D * H];
            p += qv * my_sk[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            p += __shfl_down_sync(0xffffffff, p, o);
        }
        if (lane == 0) {
            float dot = p * scale;
            if (dot > 0.0f) {
                my_score += dot * GATES[(int64_t)h + (int64_t)qt * H];
            }
        }
    }

    if (lane == 0) {
        S[(int64_t)kt + (int64_t)qt * n_kv] = my_score;
    }
}

void ggml_cuda_op_dsa_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cache = dst->src[0];
    const ggml_tensor * q_full  = dst->src[1];
    const ggml_tensor * gates   = dst->src[2];

    GGML_ASSERT(k_cache->type == GGML_TYPE_F16);
    GGML_ASSERT(q_full->type  == GGML_TYPE_F32);
    GGML_ASSERT(gates->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(k_cache));
    GGML_ASSERT(ggml_is_contiguous(q_full));
    GGML_ASSERT(ggml_is_contiguous(gates));

    const int D    = (int) k_cache->ne[0];
    const int n_kv = (int) k_cache->ne[1];
    const int H    = (int) q_full->ne[1];
    const int n_q  = (int) q_full->ne[2];

    GGML_ASSERT(D == DSA_INDEXER_SCORE_D);  // v2 supports only D=128 (matches GLM-DSA)
    GGML_ASSERT(D == (int) q_full->ne[0]);
    GGML_ASSERT(H == (int) gates->ne[0]);
    GGML_ASSERT(n_q == (int) gates->ne[1]);
    GGML_ASSERT(n_kv == (int) dst->ne[0]);
    GGML_ASSERT(n_q  == (int) dst->ne[1]);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();

    const half  * K_d = (const half  *) k_cache->data;
    const float * Q_d = (const float *) q_full->data;
    const float * G_d = (const float *) gates->data;
    float       * S_d = (float       *) dst->data;

    dim3 grid(n_kv, n_q, 1);
    dim3 block(DSA_INDEXER_SCORE_BLOCK, 1, 1);

    dsa_indexer_score_kernel_v2<DSA_INDEXER_SCORE_D, DSA_INDEXER_SCORE_BLOCK, DSA_INDEXER_SCORE_LANES>
        <<<grid, block, 0, stream>>>(K_d, Q_d, G_d, S_d, H, n_kv, n_q, scale);
}

template<int D, int BLOCK_DIM, int WARPS, int LANES>
static __global__ void dsa_indexer_score_kernel_v3(
        const half  * __restrict__ K,       // [D, n_kv]   fp16
        const float * __restrict__ Q,       // [D, H, n_q] fp32
        const float * __restrict__ GATES,   // [H, n_q]    fp32
        float       * __restrict__ S,       // [n_kv, n_q] fp32 scratch
        const int H,
        const int n_kv,
        const int n_q,
        const float scale) {
    static_assert(BLOCK_DIM == WARPS * 32, "v3 kernel uses one warp per k in an 8-k tile");
    static_assert(D == 32 * LANES, "D must equal warp_size * LANES");

    const int qt   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (qt >= n_q) {
        return;
    }

    extern __shared__ float smem[];
    float * sQ = smem;
    float * sG = sQ + D * H;

    for (int i = tid; i < D * H; i += BLOCK_DIM) {
        sQ[i] = Q[(int64_t)i + (int64_t)qt * D * H];
    }
    for (int h = tid; h < H; h += BLOCK_DIM) {
        sG[h] = GATES[(int64_t)h + (int64_t)qt * H];
    }
    __syncthreads();

    for (int base = 0; base < n_kv; base += WARPS) {
        const int kt = base + warp;

        if (kt < n_kv) {
            float my_k[LANES];
            #pragma unroll
            for (int i = 0; i < LANES; ++i) {
                const int d = lane + i * 32;
                my_k[i] = __half2float(K[(int64_t)d + (int64_t)kt * D]);
            }

            float score = 0.0f;
            for (int h = 0; h < H; ++h) {
                float p = 0.0f;
                #pragma unroll
                for (int i = 0; i < LANES; ++i) {
                    const int d = lane + i * 32;
                    p += sQ[d + h * D] * my_k[i];
                }

                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    p += __shfl_down_sync(0xffffffff, p, o);
                }

                if (lane == 0) {
                    const float dot = p * scale;
                    if (dot > 0.0f) {
                        score += dot * sG[h];
                    }
                }
            }

            if (lane == 0) {
                S[(int64_t)kt + (int64_t)qt * n_kv] = score;
            }
        }
    }
}

static __device__ __forceinline__ uint32_t dsa_float_to_ordered_key(const float v) {
    const uint32_t bits = __float_as_uint(v);
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

static __device__ __forceinline__ int dsa_i_min(const int a, const int b) {
    return a < b ? a : b;
}

static __device__ __forceinline__ int dsa_i_max(const int a, const int b) {
    return a > b ? a : b;
}

// Exact radix top-k over a contiguous candidate range [lo, hi). The caller must
// ensure all threads in the block call this together; it uses block-wide sync.
static __device__ void dsa_topk_select_range(
        const float * __restrict__ col,
        int         * __restrict__ out,
        const int lo,
        const int hi,
        const int take,
        const int tid,
        int      * s_hist,
        uint32_t * s_prefix,
        uint32_t * s_prefix_mask,
        uint32_t * s_threshold,
        int      * s_kth,
        int      * s_out_count) {
    if (take <= 0 || hi <= lo) {
        return;
    }

    if (tid == 0) {
        *s_prefix      = 0;
        *s_prefix_mask = 0;
        *s_kth         = take;
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += blockDim.x) {
            s_hist[i] = 0;
        }
        __syncthreads();

        const uint32_t prefix      = *s_prefix;
        const uint32_t prefix_mask = *s_prefix_mask;
        for (int k = lo + tid; k < hi; k += blockDim.x) {
            const uint32_t key = dsa_float_to_ordered_key(col[k]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&s_hist[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int skipped = 0;
            int chosen  = 0;
            for (int bucket = 255; bucket >= 0; --bucket) {
                const int count = s_hist[bucket];
                if (skipped + count >= *s_kth) {
                    chosen = bucket;
                    break;
                }
                skipped += count;
            }

            *s_kth         -= skipped;
            *s_prefix      |= (uint32_t) chosen << shift;
            *s_prefix_mask |= 0xffu << shift;
        }
        __syncthreads();
    }

    if (tid == 0) {
        *s_threshold = *s_prefix;
        *s_out_count = 0;
    }
    __syncthreads();

    const uint32_t threshold = *s_threshold;

    for (int k = lo + tid; k < hi; k += blockDim.x) {
        const uint32_t key = dsa_float_to_ordered_key(col[k]);
        if (key > threshold) {
            const int pos = atomicAdd(s_out_count, 1);
            if (pos < take) {
                out[pos] = k;
            }
        }
    }
    __syncthreads();

    for (int k = lo + tid; k < hi; k += blockDim.x) {
        const uint32_t key = dsa_float_to_ordered_key(col[k]);
        if (key == threshold) {
            const int pos = atomicAdd(s_out_count, 1);
            if (pos < take) {
                out[pos] = k;
            }
        }
    }
    __syncthreads();
}

// [DSA_TOPK_CAUSAL 2026-07-03] Position-causal score masking BEFORE top-k selection.
// ROOT-CAUSE FIX for the GLM-5.2 DSA multi-ubatch prefill content corruption: the
// indexer scored the FULL n_kv cell range with no causal mask (iter-34 "tradeoff"),
// so cache cells NOT YET WRITTEN for the current (layer, ubatch) — zeros at layer 0,
// the PREVIOUS layer's K in the shared layer-major staging slab at layers 1+ —
// competed for the fixed top-k budget. The pos-causal ATTENTION mask correctly
// suppressed their contribution, but it cannot restore the causally-valid keys they
// CROWDED OUT of the top-k set (worst for early tokens, whose few valid keys can
// also score <= 0 and rank below thousands of zero/stale cells) -> attention lost
// most of its true context -> cumulative corruption (needle MISS, decorrelated KV).
// Masking score[k, q] = -FLT_MAX for k > pos[q] makes every causally-valid cell
// rank strictly above every invalid one: queries with < top_k valid keys get ALL
// of them (sparse == dense there, mathematically); later queries do true DSA
// selection among VALID keys only. Surplus slots still fill with masked cells,
// which the downstream pos-causal attention mask excludes exactly.
static __global__ void dsa_causal_mask_scores_kernel(
        float         * __restrict__ S,    // [n_kv, n_q]
        const int32_t * __restrict__ pos,  // [>= n_q] query positions
        const int n_kv,
        const int n_q) {
    const int qt = blockIdx.x;
    if (qt >= n_q) {
        return;
    }
    const int p = pos[qt];
    float * col = S + (int64_t)qt * n_kv;
    for (int k = threadIdx.x; k < n_kv; k += blockDim.x) {
        if (k > p) {
            col[k] = -FLT_MAX;
        }
    }
}

static __global__ void dsa_topk_select_kernel(
        const float * __restrict__ scores,   // [n_kv, n_q]
        int   * __restrict__ indices,  // [top_k, n_q]
        const int n_kv,
        const int n_q,
        const int top_k) {
    const int qt  = blockIdx.x;
    const int tid = threadIdx.x;

    if (qt >= n_q) {
        return;
    }

    const float * col = scores  + (int64_t)qt * n_kv;
    int         * out = indices + (int64_t)qt * top_k;

    __shared__ int      s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_prefix_mask;
    __shared__ uint32_t s_threshold;
    __shared__ int      s_kth;
    __shared__ int      s_out_count;

    dsa_topk_select_range(col, out, 0, n_kv, top_k, tid, s_hist,
        &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
}

// Env-gated coverage selector for long-context DSA FULL reuse. It keeps the
// same top_k budget but reserves quota across causal position bands, preventing
// one recent-heavy global top-k set from dropping old/middle evidence that
// shared sparse layers cannot recover.
static __global__ void dsa_topk_select_coverage_kernel(
        const float   * __restrict__ scores,   // [n_kv, n_q]
        const int32_t * __restrict__ pos,      // optional [>= n_q]
        int           * __restrict__ indices,  // [top_k, n_q]
        const int n_kv,
        const int n_q,
        const int top_k,
        const int n_bands) {
    const int qt  = blockIdx.x;
    const int tid = threadIdx.x;

    if (qt >= n_q) {
        return;
    }

    const float * col = scores  + (int64_t)qt * n_kv;
    int         * out = indices + (int64_t)qt * top_k;

    int valid_len = n_kv;
    if (pos != nullptr) {
        valid_len = pos[qt] + 1;
        valid_len = dsa_i_max(0, dsa_i_min(valid_len, n_kv));
    }

    if (valid_len <= 0) {
        for (int i = tid; i < top_k; i += blockDim.x) {
            out[i] = 0;
        }
        return;
    }

    // If the valid causal window is smaller than top_k, output every valid key
    // and pad with the first future key when available. The downstream causal
    // mask excludes those pads. This mirrors the old masked-top-k behavior
    // without letting -FLT_MAX future ties determine the useful prefix.
    if (valid_len <= top_k) {
        const int pad = valid_len < n_kv ? valid_len : valid_len - 1;
        for (int i = tid; i < top_k; i += blockDim.x) {
            out[i] = i < valid_len ? i : pad;
        }
        return;
    }

    __shared__ int      s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_prefix_mask;
    __shared__ uint32_t s_threshold;
    __shared__ int      s_kth;
    __shared__ int      s_out_count;

    // Near-dense rows do not need coverage rebalancing, and band quotas can
    // over-constrain tiny ranges. Use exact global selection there.
    if (n_bands <= 1 || valid_len < top_k * 2) {
        dsa_topk_select_range(col, out, 0, valid_len, top_k, tid, s_hist,
            &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
        return;
    }

    const int bands = dsa_i_min(dsa_i_min(n_bands, top_k), valid_len);
    const int base  = top_k / bands;
    const int rem   = top_k - base * bands;

    for (int b = 0; b < bands; ++b) {
        const int lo = (int)(((int64_t) valid_len * b) / bands);
        const int hi = (int)(((int64_t) valid_len * (b + 1)) / bands);
        const int take = base + (b < rem ? 1 : 0);
        const int out0 = b * base + dsa_i_min(b, rem);

        dsa_topk_select_range(col, out + out0, lo, hi, take, tid, s_hist,
            &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
    }
}

// [DSA coverage-weighted 2026-07-08] Env-gated strict-5/5 weighted coverage selector. Uniform bands cap
// at 4/5 (fine bands miss the oldest needle; coarse bands lump the middle). Non-uniform ranges + quotas fit
// the empirical constraints: oldest OCTILE [0,1/8) gets 3/8*K (targets the earliest needle in a small range),
// three middle quarters get 1/8*K each (256 proven enough for 25/50/75), recent octile [7/8,1) gets 1/4*K
// (proven needed for 95%). K=2048 -> 768+256+256+256+512=2048. Disjoint ranges => radix work ~= 4*valid_len
// (same speed). Marker: LLAMA_DSA_TOPK_COVERAGE_WEIGHTED. Default off => SEMI/FULL unchanged. Design: Codex xhigh.
static __global__ void dsa_topk_select_weighted_coverage_kernel(
        const float   * __restrict__ scores,   // [n_kv, n_q]
        const int32_t * __restrict__ pos,      // optional [>= n_q]
        int           * __restrict__ indices,  // [top_k, n_q]
        const int n_kv,
        const int n_q,
        const int top_k) {
    const int qt  = blockIdx.x;
    const int tid = threadIdx.x;

    if (qt >= n_q) {
        return;
    }

    const float * col = scores  + (int64_t)qt * n_kv;
    int         * out = indices + (int64_t)qt * top_k;

    int valid_len = n_kv;
    if (pos != nullptr) {
        valid_len = pos[qt] + 1;
        valid_len = dsa_i_max(0, dsa_i_min(valid_len, n_kv));
    }

    if (valid_len <= 0) {
        for (int i = tid; i < top_k; i += blockDim.x) {
            out[i] = 0;
        }
        return;
    }

    if (valid_len <= top_k) {
        const int pad = valid_len < n_kv ? valid_len : valid_len - 1;
        for (int i = tid; i < top_k; i += blockDim.x) {
            out[i] = i < valid_len ? i : pad;
        }
        return;
    }

    __shared__ int      s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_prefix_mask;
    __shared__ uint32_t s_threshold;
    __shared__ int      s_kth;
    __shared__ int      s_out_count;

    // The weighted profile is intended for long-context rows. For shorter rows,
    // small edge bands can become over-constrained; exact global top-k is safer.
    if (top_k < 8 || valid_len < top_k * 8) {
        dsa_topk_select_range(col, out, 0, valid_len, top_k, tid, s_hist,
            &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
        return;
    }

    const int b0 = (int)(((int64_t) valid_len * 1) / 8);
    const int b1 = (int)(((int64_t) valid_len * 3) / 8);
    const int b2 = (int)(((int64_t) valid_len * 5) / 8);
    const int b3 = (int)(((int64_t) valid_len * 7) / 8);

    const int q0 = (top_k * 3) / 8;
    const int q4 = top_k / 4;
    const int q_mid_total = top_k - q0 - q4;
    const int q1 = q_mid_total / 3;
    const int q2 = q_mid_total / 3;
    const int q3 = q_mid_total - q1 - q2;

    int out0 = 0;
    dsa_topk_select_range(col, out + out0, 0,  b0,        q0, tid, s_hist,
        &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
    out0 += q0;

    dsa_topk_select_range(col, out + out0, b0, b1,        q1, tid, s_hist,
        &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
    out0 += q1;

    dsa_topk_select_range(col, out + out0, b1, b2,        q2, tid, s_hist,
        &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
    out0 += q2;

    dsa_topk_select_range(col, out + out0, b2, b3,        q3, tid, s_hist,
        &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
    out0 += q3;

    dsa_topk_select_range(col, out + out0, b3, valid_len, q4, tid, s_hist,
        &s_prefix, &s_prefix_mask, &s_threshold, &s_kth, &s_out_count);
}

static __global__ void dsa_pack_q_chunk_f16_kernel(
        const float * __restrict__ Q,       // [D, H, n_q]
        half        * __restrict__ QH,      // [D, Hc, n_q], compact
        const int D,
        const int H,
        const int n_q,
        const int h0,
        const int Hc) {
    const int64_t total = (int64_t) D * Hc * n_q;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) blockDim.x * gridDim.x) {
        const int d = (int) (i % D);
        const int h = (int) ((i / D) % Hc);
        const int q = (int) (i / ((int64_t) D * Hc));
        QH[i] = __float2half(Q[(int64_t) d + (int64_t) (h0 + h) * D + (int64_t) q * D * H]);
    }
}

template<bool NATIVE_ORDER=false,int HEAD_CHUNK=DSA_INDEXER_GEMM_HEAD_CHUNK>
static __global__ void dsa_accum_scores_from_gemm_kernel(
        const float * __restrict__ dot,     // [Ks, Hc, n_q] column-major
        const float * __restrict__ gates,   // [H, n_q]
        float       * __restrict__ scores,  // [n_kv, n_q]
        const int n_kv,
        const int n_q,
        const int H,
        const int h0,
        const int Hc,
        const float scale,
        const int k_offset,                 // starting k row in scores
        const int Ks) {                     // number of k rows in this slab
    const int64_t total = (int64_t) Ks * n_q;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) blockDim.x * gridDim.x) {
        const int k = (int) (i % Ks);
        const int q = (int) (i / Ks);
        const int64_t s_idx = (int64_t) (k_offset + k) + (int64_t) q * n_kv;

        float acc = scores[s_idx];
        #pragma unroll
        for (int h = 0; h < HEAD_CHUNK; ++h) {
            if (h < Hc) {
                const float raw=dot[(int64_t) k + (int64_t) (h + q * Hc) * Ks];
                if constexpr(NATIVE_ORDER){
                    const float v=ggml_dsv41_relu(__fmul_rn(raw,scale));
                    acc=__fmaf_rn(v,gates[(int64_t)(h0+h)+(int64_t)q*H],acc);
                }else{
                    const float v=raw*scale;
                    if(v>0.0f)acc+=v*gates[(int64_t)(h0+h)+(int64_t)q*H];
                }
            }
        }
        scores[s_idx] = acc;
    }
}

static void dsa_indexer_score_cublas(
        ggml_backend_cuda_context & ctx,
        const half  * K_d,
        const float * Q_d,
        const float * G_d,
        float       * S_d,
        const int D,
        const int H,
        const int n_kv,
        const int n_q,
        const float scale,
        const int32_t * P_h,   // [DSA_SCORE_CAUSAL_TILE] optional host-side pos; nullptr = no tiling
        cudaStream_t stream,int explicit_k_slab=0,int explicit_nq_slab=0,bool native_order=false) {
    GGML_ASSERT(D == DSA_INDEXER_SCORE_D);
    GGML_ASSERT(H % DSA_INDEXER_GEMM_HEAD_CHUNK == 0);

    static const int k_slab_env = []() {
        const char * env = getenv("LLAMA_DSA_SCORE_CUBLAS_K_SLAB");
        return env ? atoi(env) : 0;
    }();
    const int k_slab = explicit_k_slab>0?explicit_k_slab:(k_slab_env > 0 ? k_slab_env : n_kv);
    const int dot_rows = (k_slab < n_kv) ? k_slab : n_kv;

    // 2026-07-02 (DSA OOM fix, Job 2): the transient dot / Q_chunk pool buffers
    // scaled with the FULL ubatch n_q, so at ctx>=32768 with a single large
    // ubatch (n_q ~= 4044) dot_alloc = k_slab * HEAD_CHUNK * n_q * 4B reached
    // ~8.5 GB (k_slab=16384) and, stacked with WF + the compute buffer,
    // contributed to the ~29 GB cudaMalloc OOM on the 32 GB 5090. K_SLAB only
    // tiled the k dimension; it did nothing for n_q. Chunk the query dimension
    // too: process n_q_slab tokens per pass so the transient is bounded
    // independent of ubatch size. Math is unchanged (each (h0,k0,q0) tile is
    // computed exactly once; accum writes disjoint score columns). Tune with
    // LLAMA_DSA_SCORE_CUBLAS_NQ_SLAB (default 1024 -> ~2.1 GB transient at
    // k_slab=16384; 0 or negative = no q-chunking = legacy behavior).
    static const int nq_slab_env = []() {
        const char * env = getenv("LLAMA_DSA_SCORE_CUBLAS_NQ_SLAB");
        return env ? atoi(env) : 1024;  // default: bound the transient
    }();
    const int requested_nq_slab=explicit_nq_slab>0?explicit_nq_slab:nq_slab_env;
    const int nq_slab = (requested_nq_slab > 0 && requested_nq_slab < n_q) ? requested_nq_slab : n_q;

    ggml_cuda_pool_alloc<half>  Q_chunk_alloc(ctx.pool(), (size_t) D * DSA_INDEXER_GEMM_HEAD_CHUNK * nq_slab);
    ggml_cuda_pool_alloc<float> dot_alloc(ctx.pool(), (size_t) dot_rows * DSA_INDEXER_GEMM_HEAD_CHUNK * nq_slab);

    half  * Q_chunk = Q_chunk_alloc.get();
    float * dot     = dot_alloc.get();

    CUDA_CHECK(cudaMemsetAsync(S_d, 0, (size_t) n_kv * n_q * sizeof(float), stream));
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int pack_block = 256;
    const int accum_block = 256;

    for (int q0 = 0; q0 < n_q; q0 += nq_slab) {
        const int Qs = (nq_slab < n_q - q0) ? nq_slab : n_q - q0;
        // Query-token slab: Q_d is [D, H, n_q]; the +q0*H column offset selects
        // this slab's queries. Scores for these queries land in S_d columns
        // [q0, q0+Qs) via the +q0*n_kv offset baked into S_slab.
        const float * Q_slab_ptr = Q_d + (size_t) q0 * D * H;
        const float * G_slab_ptr = G_d + (size_t) q0 * H;
        float       * S_slab_ptr = S_d + (size_t) q0 * n_kv;

        // [DSA_SCORE_CAUSAL_TILE 2026-07-10, Codex xhigh] when P_h is provided, skip GEMM/accum for
        // key rows > the max causal pos in this query slab. Those cells are k>pos[q] for EVERY query
        // here, so they stay 0 (memset) and the later dsa_causal_mask_scores_kernel rewrites them to
        // -FLT_MAX exactly as before => bitwise-identical S_d, all top-k selectors unchanged. Boundary
        // waste (queries with pos<max) is masked as today. Saves ~upper-triangle when queries span ctx.
        int k_limit = n_kv;
        if (P_h != nullptr) {
            int max_pos = -1;
            for (int q = 0; q < Qs; ++q) {
                int p = P_h[q0 + q];
                if (p >= n_kv) { p = n_kv - 1; }
                if (p > max_pos) { max_pos = p; }
            }
            k_limit = max_pos + 1;
            if (k_limit <= 0) { continue; }
        }

        for (int h0 = 0; h0 < H; h0 += DSA_INDEXER_GEMM_HEAD_CHUNK) {
            const int Hc = (DSA_INDEXER_GEMM_HEAD_CHUNK < H - h0) ? DSA_INDEXER_GEMM_HEAD_CHUNK : H - h0;
            const int64_t pack_total = (int64_t) D * Hc * Qs;
            const int64_t pack_grid_i64 = (pack_total + pack_block - 1) / pack_block;
            const int pack_grid = (int) (pack_grid_i64 > 65535 ? 65535 : pack_grid_i64);
            dsa_pack_q_chunk_f16_kernel<<<pack_grid, pack_block, 0, stream>>>(Q_slab_ptr, Q_chunk, D, H, Qs, h0, Hc);

            for (int k0 = 0; k0 < k_limit; k0 += k_slab) {
                const int Ks = (k_slab < k_limit - k0) ? k_slab : k_limit - k0;
                const half * K_slab_ptr = K_d + (size_t) k0 * D;

                CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                        Ks, Hc * Qs, D,
                        &alpha, K_slab_ptr, CUDA_R_16F, D,
                                Q_chunk,    CUDA_R_16F, D,
                        &beta,  dot,        CUDA_R_32F, Ks,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

                const int64_t accum_total = (int64_t) Ks * Qs;
                const int64_t accum_grid_i64 = (accum_total + accum_block - 1) / accum_block;
                const int accum_grid = (int) (accum_grid_i64 > 65535 ? 65535 : accum_grid_i64);
                // n_q passed as Qs so the accum kernel indexes dot / gates / scores
                // within this query slab; S_slab_ptr already offsets to column q0.
                if(native_order)dsa_accum_scores_from_gemm_kernel<true><<<accum_grid,accum_block,0,stream>>>(
                    dot,G_slab_ptr,S_slab_ptr,n_kv,Qs,H,h0,Hc,scale,k0,Ks);
                else dsa_accum_scores_from_gemm_kernel<false><<<accum_grid,accum_block,0,stream>>>(
                    dot,G_slab_ptr,S_slab_ptr,n_kv,Qs,H,h0,Hc,scale,k0,Ks);
            }
        }
    }
}

static __global__ void dsv41_index_half_exact_kernel(const float*q,const float*k,const float*w,const int32_t*visible,
        uint64_t nq,uint64_t nk,uint64_t nw,uint64_t tokens,int rows,uint32_t *bad){
    const uint64_t maximum=nq>nk?nq:nk;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<maximum;i+=uint64_t(gridDim.x)*blockDim.x){
        if(i<nq){const float v=q[i];if(!isfinite(v)||__float_as_uint(__half2float(__float2half_rn(v)))!=__float_as_uint(v))atomicOr(bad,1u);}
        if(i<nk){const float v=k[i];if(!isfinite(v)||__float_as_uint(__half2float(__float2half_rn(v)))!=__float_as_uint(v))atomicOr(bad,2u);}
        if(i<nw){const float v=w[i];if(!isfinite(v)||(v!=0.f&&(fabsf(v)<0x1p-32f||fabsf(v)>0x1p32f)))atomicOr(bad,4u);}
        if(i<tokens&&(visible[i]<0||visible[i]>rows))atomicOr(bad,8u);
    }
}
// Each admitted128-value row is an integer vector with magnitudes<256 in
// units2^L. Hence each dot has |integer partial sum|<128*255^2<2^23:
// products and every F32 partial sum are exact, independent of reduction order.
static __global__ void dsv41_index_exact_dot_span_kernel(const float*q,const float*k,uint32_t qrows,uint32_t krows,uint32_t*bad){
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x*4+warp;
    if(row>=qrows+krows)return;
    const float*base=row<qrows?q+uint64_t(row)*128:k+uint64_t(row-qrows)*128;
    int low=INT_MAX,high=INT_MIN;
    for(unsigned d=lane;d<128;d+=32){const uint32_t bits=__float_as_uint(base[d])&0x7fffffffu;
        if(bits){const int e=int(bits>>23)-127;const uint32_t mant=(bits&0x7fffffu)|0x800000u;
            low=min(low,e-23+__ffs(int(mant))-1);high=max(high,e);}}
    for(unsigned d=16;d;d/=2){low=min(low,__shfl_down_sync(0xffffffffu,low,d));high=max(high,__shfl_down_sync(0xffffffffu,high,d));}
    if(!lane&&low!=INT_MAX&&high-low>7)atomicOr(bad,16u);
}
static __global__ void dsv41_index_apply_mask_kernel(float*out,const int32_t*visible,const float*mask,int rows,int tokens){
    const uint64_t total=uint64_t(rows)*tokens;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=uint64_t(gridDim.x)*blockDim.x){
        const int row=int(i%rows),token=int(i/rows);
        if(row>=visible[token]||(mask&&mask[uint64_t(token)*((rows+7)/8)+row/8]!=0.f))out[i]=-CUDART_INF_F;
    }
}
static std::atomic<uint64_t> dsv41_index_cublas_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_index_cublas_queued(void){return dsv41_index_cublas_queued.load(std::memory_order_acquire);}
void ggml_cuda_op_dsv41_index_score_cublas(ggml_backend_cuda_context&ctx,ggml_tensor*dst){
    GGML_ASSERT(ggml_dsv41_index_cublas_requested(dst)&&ggml_dsv41_index_cublas_shape(dst));
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200);
    const auto*q=dst->src[0],*weights=dst->src[1],*keys=dst->src[2],*visible=dst->src[3],*mask=dst->src[4];
    const int rows=int(keys->ne[1]),tokens=int(q->ne[2]);const auto stream=ctx.stream();
    const auto nq=uint64_t(ggml_nelements(q)),nk=uint64_t(ggml_nelements(keys)),nw=uint64_t(ggml_nelements(weights));
    if(ggml_dsv41_index_f32_requested(dst)){
        // Reuse GLM's GEMM scorer and ordered F32 head reduction. Q and K are
        // consumed directly as F32; no conversion, packing, or TF32 math.
        constexpr int D=128,H=32,q_slab=64,k_slab=8192;
        const int Qs_max=std::min(tokens,q_slab),Ks_max=std::min(rows,k_slab);
        const uint64_t dot_elements=uint64_t(Qs_max)*H*Ks_max;
        GGML_ASSERT(dot_elements*sizeof(float)<=64ull*1024*1024);
        ggml_cuda_pool_alloc<float> dot(ctx.pool(),dot_elements);
        CUDA_CHECK(cudaMemsetAsync(dst->data,0,size_t(rows)*tokens*sizeof(float),stream));
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(),stream));
        const float alpha=1.f,beta=0.f;
        for(int q0=0;q0<tokens;q0+=q_slab){
            const int Qs=std::min(tokens-q0,q_slab),ratio=dst->op_params[6];
            GGML_ASSERT(ratio==0||ratio==1||ratio==2);
            const int key_offset=int(ggml_dsv41_index_window_offset(dst));
            const int k_limit=ratio?std::max(0,std::min(rows,(dst->op_params[5]+q0+Qs)/ratio-key_offset)):rows;
            for(int k0=0;k0<k_limit;k0+=k_slab){
                const int Ks=std::min(k_limit-k0,k_slab);
                CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),CUBLAS_OP_T,CUBLAS_OP_N,
                    Ks,H*Qs,D,&alpha,(const float*)keys->data+size_t(k0)*D,CUDA_R_32F,D,
                    (const float*)q->data+size_t(q0)*H*D,CUDA_R_32F,D,
                    &beta,dot.get(),CUDA_R_32F,Ks,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
                const int blocks=int((uint64_t(Ks)*Qs+255)/256);
                dsa_accum_scores_from_gemm_kernel<true,H><<<blocks,256,0,stream>>>(dot.get(),
                    (const float*)weights->data+size_t(q0)*H,(float*)dst->data+size_t(q0)*rows,
                    rows,Qs,H,0,H,1.f/64.f,k0,Ks);
            }
        }
        const auto blocks=unsigned(std::min<uint64_t>(65535,(uint64_t(rows)*tokens+255)/256));
        dsv41_index_apply_mask_kernel<<<blocks,256,0,stream>>>((float*)dst->data,(const int32_t*)visible->data,
            mask?(const float*)mask->data:nullptr,rows,tokens);
        CUDA_CHECK(cudaGetLastError());dsv41_index_cublas_queued.fetch_add(1,std::memory_order_release);return;
    }
    if(ggml_dsv41_index_bf16_requested(dst)){
        if(dsv41_index_fp4(ctx,dst)){dsv41_index_cublas_queued.fetch_add(1,std::memory_order_release);return;}
        // Host-only adapter: reuse ggml's BF16 conversion, cuBLAS GEMM and
        // GLM's existing ordered accumulation/mask kernels. No new GPU kernel.
        constexpr int D=128,H=32,q_slab=256,Hc=DSA_INDEXER_GEMM_HEAD_CHUNK;
        static_assert(H%Hc==0,"Complete GLM head chunks required");
        const int Qs_max=std::min(tokens,q_slab);
        const uint64_t q_elements=uint64_t(D)*Hc*Qs_max;
        // Retune the existing slabs inside the same64MiB reserve. No extra
        // buffers: leave room for canonical-key conversion and packed queries.
        constexpr uint64_t ceiling=64ull*1024*1024;
        const uint64_t packed_bytes=(q_elements+nk)*sizeof(nv_bfloat16);
        GGML_ASSERT(packed_bytes<ceiling);
        const uint64_t max_keys=(ceiling-packed_bytes)/(uint64_t(Qs_max)*Hc*sizeof(float));
        const int k_slab=int(std::min<uint64_t>(8192,max_keys/256*256));
        GGML_ASSERT(k_slab>=256);
        const int Ks_max=std::min(rows,k_slab);
        const uint64_t dot_elements=uint64_t(Qs_max)*Ks_max*Hc;
        const uint64_t temporary=(q_elements+nk)*sizeof(nv_bfloat16)+dot_elements*sizeof(float);
        GGML_ASSERT(temporary<=64ull*1024*1024&&"DSV41_INDEX_BF16_WORKSPACE_LIMIT");
        ggml_cuda_pool_alloc<nv_bfloat16> q_bf16(ctx.pool(),q_elements),k_bf16(ctx.pool(),nk);
        ggml_cuda_pool_alloc<float> dot(ctx.pool(),dot_elements);
        const auto convert=ggml_get_to_bf16_cuda(GGML_TYPE_F32);GGML_ASSERT(convert);
        const auto pack=ggml_get_to_bf16_nc_cuda(GGML_TYPE_F32);GGML_ASSERT(pack);
        convert(keys->data,k_bf16.get(),nk,stream);
        CUDA_CHECK(cudaMemsetAsync(dst->data,0,size_t(rows)*tokens*sizeof(float),stream));
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(),stream));
        const float alpha=1.f,beta=0.f;
        for(int q0=0;q0<tokens;q0+=q_slab){
            const int Qs=std::min(tokens-q0,q_slab);
            // Same causal slab exclusion as the existing GLM scorer above.
            // The executor supplied the position/ratio proof and uploads the
            // matching visibility vector; final per-query masking is unchanged.
            const int ratio=dst->op_params[6];
            GGML_ASSERT(ratio==0||ratio==1||ratio==2);
            const int key_offset=int(ggml_dsv41_index_window_offset(dst));
            const int k_limit=ratio?std::max(0,std::min(rows,(dst->op_params[5]+q0+Qs)/ratio-key_offset)):rows;
            if(k_limit<=0)continue;
            for(int h0=0;h0<H;h0+=Hc){
             pack((const float*)q->data+(size_t(q0)*H+h0)*D,q_bf16.get(),D,Hc,Qs,1,D,D*H,0,stream);
              for(int k0=0;k0<k_limit;k0+=k_slab){
                const int Ks=std::min(k_limit-k0,k_slab);
                CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),CUBLAS_OP_T,CUBLAS_OP_N,
                    Ks,Hc*Qs,D,&alpha,k_bf16.get()+size_t(k0)*D,CUDA_R_16BF,D,
                    q_bf16.get(),CUDA_R_16BF,D,&beta,dot.get(),CUDA_R_32F,Ks,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                const int blocks=int((uint64_t(Ks)*Qs+255)/256);
                dsa_accum_scores_from_gemm_kernel<true><<<blocks,256,0,stream>>>(dot.get(),
                    (const float*)weights->data+size_t(q0)*H,(float*)dst->data+size_t(q0)*rows,
                    rows,Qs,H,h0,Hc,1.f/64.f,k0,Ks);
             }
            }
        }
        const auto blocks=unsigned(std::min<uint64_t>(65535,(uint64_t(rows)*tokens+255)/256));
        dsv41_index_apply_mask_kernel<<<blocks,256,0,stream>>>((float*)dst->data,(const int32_t*)visible->data,
            mask?(const float*)mask->data:nullptr,rows,tokens);
        CUDA_CHECK(cudaGetLastError());dsv41_index_cublas_queued.fetch_add(1,std::memory_order_release);return;
    }
    ggml_cuda_pool_alloc<uint32_t> flag(ctx.pool(),1);
    CUDA_CHECK(cudaMemsetAsync(flag.get(),0,4,stream));
    const auto blocks=unsigned(std::min<uint64_t>(65535,(std::max(nq,nk)+255)/256));
    dsv41_index_half_exact_kernel<<<blocks,256,0,stream>>>((const float*)q->data,(const float*)keys->data,
        (const float*)weights->data,(const int32_t*)visible->data,nq,nk,nw,tokens,rows,flag.get());
    const unsigned qrows=unsigned(nq/128),krows=unsigned(nk/128);
    dsv41_index_exact_dot_span_kernel<<<(qrows+krows+3)/4,128,0,stream>>>((const float*)q->data,(const float*)keys->data,qrows,krows,flag.get());
    uint32_t invalid=0;CUDA_CHECK(cudaMemcpyAsync(&invalid,flag.get(),4,cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // This is a representation-preserving reuse experiment, never another
    // activation quantizer. Non-exact/overflowing/subnormal input fails loudly.
    GGML_ASSERT(invalid==0&&"DSV41_INDEX_CUBLAS_EXACT_DOT_OR_VISIBILITY_ADMISSION_REFUSED");
    ggml_cuda_pool_alloc<half> half_keys(ctx.pool(),nk);
    const auto convert=ggml_get_to_fp16_cuda(GGML_TYPE_F32);GGML_ASSERT(convert);
    convert(keys->data,half_keys.get(),nk,stream);
    // Reuse GLM's existing head-chunk GEMM and ordered head accumulation.
    // Fixed slabs bound DOT to2MiB and Q_chunk to512KiB; no GLM env override.
    dsa_indexer_score_cublas(ctx,half_keys.get(),(const float*)q->data,(const float*)weights->data,
        (float*)dst->data,128,32,rows,tokens,1.f/64.f,nullptr,stream,256,256,true);
    const auto mask_blocks=unsigned(std::min<uint64_t>(65535,(uint64_t(rows)*tokens+255)/256));
    dsv41_index_apply_mask_kernel<<<mask_blocks,256,0,stream>>>((float*)dst->data,(const int32_t*)visible->data,
        mask?(const float*)mask->data:nullptr,rows,tokens);
    dsv41_index_cublas_queued.fetch_add(1,std::memory_order_release);
}

void ggml_cuda_op_dsa_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cache = dst->src[0];
    const ggml_tensor * q_full  = dst->src[1];
    const ggml_tensor * gates   = dst->src[2];
    const ggml_tensor * pos     = dst->src[3];  // optional [DSA_TOPK_CAUSAL]

    GGML_ASSERT(k_cache->type == GGML_TYPE_F16);
    GGML_ASSERT(q_full->type  == GGML_TYPE_F32);
    GGML_ASSERT(gates->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(k_cache));
    GGML_ASSERT(ggml_is_contiguous(q_full));
    GGML_ASSERT(ggml_is_contiguous(gates));

    const int D     = (int) k_cache->ne[0];
    const int n_kv  = (int) k_cache->ne[1];
    const int H     = (int) q_full->ne[1];
    const int n_q   = (int) q_full->ne[2];
    const int top_k = (int) dst->ne[0];

    GGML_ASSERT(D == DSA_INDEXER_SCORE_D);
    GGML_ASSERT(D == (int) q_full->ne[0]);
    GGML_ASSERT(H == (int) gates->ne[0]);
    GGML_ASSERT(n_q == (int) gates->ne[1]);
    GGML_ASSERT(top_k <= n_kv);
    GGML_ASSERT(n_q == (int) dst->ne[1]);

    float scale;
    int32_t top_k_param;
    memcpy(&scale, dst->op_params, sizeof(float));
    memcpy(&top_k_param, dst->op_params + 1, sizeof(int32_t));
    GGML_ASSERT(top_k_param == top_k);

    cudaStream_t stream = ctx.stream();

    const half  * K_d = (const half  *) k_cache->data;
    const float * Q_d = (const float *) q_full->data;
    const float * G_d = (const float *) gates->data;
    int         * I_d = (int         *) dst->data;

    ggml_cuda_pool_alloc<float> scratch_alloc(ctx.pool(), (size_t) n_kv * n_q);
    float * S_d = scratch_alloc.get();

    dim3 grid(n_q, 1, 1);
    dim3 block(DSA_INDEXER_TOPK_BLOCK, 1, 1);

    // [DSA_TOPK_CAUSAL 2026-07-03] pos for the post-score causal mask; hoisted above the score call so
    // the optional causal-tiling (below) can also use it. pos in-bounds read only (ne[0]>=n_q asserted).
    const int32_t * P_d = nullptr;
    if (pos != nullptr) {
        GGML_ASSERT(pos->type == GGML_TYPE_I32);
        GGML_ASSERT(pos->ne[0] >= n_q);
        P_d = (const int32_t *) pos->data;
    }

    // [DSA_SCORE_CAUSAL_TILE 2026-07-10] env-gated, default-off, bitwise-lossless: copy pos to host so
    // dsa_indexer_score_cublas can skip GEMM/accum for causally-invalid key rows (see its comment).
    static const bool causal_tile_env = []() {
        const char * env = getenv("LLAMA_DSA_SCORE_CAUSAL_TILE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();

    static const bool use_cublas_score = []() {
        const char * env = getenv("LLAMA_DSA_SCORE_CUBLAS");
        return env != nullptr && env[0] != '0';
    }();

    if (use_cublas_score && (H % DSA_INDEXER_GEMM_HEAD_CHUNK) == 0) {
        std::vector<int32_t> pos_h;
        const int32_t * P_h = nullptr;
        if (causal_tile_env && P_d != nullptr) {
            pos_h.resize((size_t) n_q);
            CUDA_CHECK(cudaMemcpyAsync(pos_h.data(), P_d, (size_t) n_q * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            P_h = pos_h.data();
        }
        dsa_indexer_score_cublas(ctx, K_d, Q_d, G_d, S_d, D, H, n_kv, n_q, scale, P_h, stream);
    } else {
        const size_t shmem = ((size_t) D * H + H) * sizeof(float);
        dsa_indexer_score_kernel_v3<DSA_INDEXER_SCORE_D, DSA_INDEXER_TOPK_BLOCK, DSA_INDEXER_TOPK_WARPS, DSA_INDEXER_SCORE_LANES>
            <<<grid, block, shmem, stream>>>(K_d, Q_d, G_d, S_d, H, n_kv, n_q, scale);
    }

    // P_d hoisted above (before the score call). Apply the post-score causal mask.
    if (pos != nullptr) {
        dsa_causal_mask_scores_kernel<<<grid, block, 0, stream>>>(S_d, P_d, n_kv, n_q);
    }

    // [DSA coverage 2026-07-08] Env-gated stratified coverage top-k: reserves per-causal-band
    // quota so old/middle content-relevant keys are not crowded out of the reused sparse top-k
    // by recent keys (fixes FULL-mode long-context needle drop at fixed K). Default OFF ->
    // byte-identical to the exact-global dsa_topk_select_kernel. Marker: LLAMA_DSA_TOPK_COVERAGE_BANDS.
    static const int coverage_bands_env = []() {
        const char * env = getenv("LLAMA_DSA_TOPK_COVERAGE_BANDS");
        const int v = env ? atoi(env) : 0;
        return v > 1 ? (v > 64 ? 64 : v) : 0;
    }();

    static const bool coverage_weighted_env = []() {
        const char * env = getenv("LLAMA_DSA_TOPK_COVERAGE_WEIGHTED");
        return env != nullptr && env[0] != '0';
    }();

    if (coverage_weighted_env) {
        dsa_topk_select_weighted_coverage_kernel<<<grid, block, 0, stream>>>(
            S_d, P_d, I_d, n_kv, n_q, top_k);
    } else if (coverage_bands_env > 1) {
        int bands = coverage_bands_env;
        if (bands > top_k) {
            bands = top_k;
        }
        dsa_topk_select_coverage_kernel<<<grid, block, 0, stream>>>(
            S_d, P_d, I_d, n_kv, n_q, top_k, bands);
    } else {
        dsa_topk_select_kernel<<<grid, block, 0, stream>>>(S_d, I_d, n_kv, n_q, top_k);
    }
}
