#include "dispatch-counter.cuh"
// Sparse flash-attention over full K/V caches plus per-query top-k indices.
// v4/v5: custom scalar WMMA kernels (scalar gather, no tensor cores)
// v6:    patched dense MMA kernel with per-row indices gather (tensor cores preserved)

// [FA_VKQ_F32_ACC_SPARSE 2026-07-07] The v6 sparse kernel reuses the dense MMA template
// flash_attn_ext_f16_sparse (fattn-mma-f16.cuh). On SM120 (Blackwell) with the MLA sparse
// shape DKQ=576/DV=512/ncols=64, the cumulative half2 VKQ accumulator saturates over the
// 50+ MLA layers → attractor-collapse garbage (observed 2026-07-07: -fa on all-sparse v6 =
// MISS `)0)` at 216 tok/s while the unfused path is coherent at 136). Phase22 angle47's fix
// gates an fp32 VKQ accumulator behind `#if defined(LLAMA_FA_VKQ_F32_ACC)` — a COMPILE-TIME
// macro (NOT a runtime env), so no launcher flag could ever enable it. Define it here (before
// fattn-mma-f16.cuh) so the sparse template instantiation uses fp32 accumulation. The header's
// use_f32_acc condition (DKQ==576 && DV==512 && ncols==64 && cols_per_warp!=8) restricts it to
// exactly this MLA sparse shape; all other shapes keep the half2 path unchanged.
// [2026-08-27] DISABLED HERE TOO. The fp32 VKQ accumulator is not the fix for this shape --
// it is the defect. test-backend-ops on hsk=576/hsv=512 (the same MLA shape this file
// instantiates) goes 86/144 -> **144/144, zero failures** when it is off, and the dense
// end-to-end path goes from token soup to exact needles at 53k. See fattn-mma-f16.cuh.
//
// NOTE the trap this file created: disabling the macro in fattn-mma-f16.cuh does NOT reach
// the sparse instantiation, because this translation unit defines it ITSELF before the
// include. Both had to be turned off.
// #ifndef LLAMA_FA_VKQ_F32_ACC
// #define LLAMA_FA_VKQ_F32_ACC 1
// #endif

#include "flash-attn-sparse.cuh"
#include "fattn-mma-f16.cuh"   // for flash_attn_ext_f16_sparse + shared utilities
#include "dequantize.cuh"
#include "ggml-dsv41-sparse-replay.h"
#include <atomic>

#include <mma.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#define SPARSE_FA_BLOCK  256
#define SPARSE_FA_MAX_DV 1024
#define SPARSE_FA_V5_K_TILE_N 32
#define SPARSE_FA_V5_MAX_SMEM (100*1024)
#define SPARSE_FA_V5_MMA_M 16
#define SPARSE_FA_V5_MMA_N 16
#define SPARSE_FA_V5_MMA_K 16

static __device__ __forceinline__ float sparse_fa_load_f32(
        const char * base, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    const char * ptr = base + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
    return __ldg((const float *) ptr);
}

static __device__ __forceinline__ float sparse_fa_load_f16(
        const char * base, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    const char * ptr = base + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
    return __half2float(__ldg((const half *) ptr));
}

static __device__ __forceinline__ float sparse_fa_load_q8_0(
        const char * base, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
        size_t nb1, size_t nb2, size_t nb3) {
    const char * row = base + i1*nb1 + i2*nb2 + i3*nb3;
    const block_q8_0 * blocks = (const block_q8_0 *) row;
    const block_q8_0 & block = blocks[i0 / QK8_0];
    const float d = block.d;
    return d * (float) block.qs[i0 % QK8_0];
}

static __device__ __forceinline__ float sparse_fa_load_kv(
        const char * base, int type, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    if (type == GGML_TYPE_Q8_0) {
        return sparse_fa_load_q8_0(base, i0, i1, i2, i3, nb1, nb2, nb3);
    }
    return sparse_fa_load_f16(base, i0, i1, i2, i3, nb0, nb1, nb2, nb3);
}

static __device__ __forceinline__ int32_t sparse_fa_load_i32(
        const char * base, int64_t i0, int64_t i1, size_t nb0, size_t nb1) {
    const char * ptr = base + i0*nb0 + i1*nb1;
    return __ldg((const int32_t *) ptr);
}

static __host__ __device__ __forceinline__ size_t sparse_fa_align_up(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

// [SPARSE_FA_V6_TILE_ALIGNED_GRID 2026-07-10, Codex xhigh] Pick a block count that DIVIDES
// ntiles_total so the v6 sparse-FA launch avoids the flash_attn_stream_k_fixup seam path
// (which otherwise runs on nearly every ub=512 sparse launch since max_blocks_per_sm*nsm
// rarely divides ntiles_total). Prefers a divisor near max_blocks that still keeps all SMs fed.
static int sparse_fa_v6_pick_tile_aligned_blocks(
        const int ntiles_total,
        const int max_blocks,
        const int nsm) {
    if (ntiles_total <= 0 || max_blocks <= 0) {
        return max_blocks;
    }

    if (ntiles_total <= max_blocks) {
        return ntiles_total;
    }

    int lower = 0;
    for (int d = max_blocks; d >= 1; --d) {
        if (ntiles_total % d == 0) {
            lower = d;
            break;
        }
    }

    int upper = 0;
    for (int d = max_blocks + 1; d <= ntiles_total; ++d) {
        if (ntiles_total % d == 0) {
            upper = d;
            break;
        }
    }

    if (lower == 0 && upper == 0) {
        return max_blocks;
    }
    if (lower == 0) {
        return upper;
    }
    if (upper == 0) {
        return lower;
    }

    // Avoid picking a divisor that leaves SMs idle when a nearby larger divisor
    // can keep all SMs fed and still eliminates the stream-K seam/fixup path.
    if (lower < nsm) {
        return upper;
    }

    const int lower_delta = max_blocks - lower;
    const int upper_delta = upper - max_blocks;
    return upper_delta < lower_delta ? upper : lower;
}

static size_t sparse_fa_v5_smem_size(const int DKQ, const int DV) {
    size_t offset = 0;
    offset = sparse_fa_align_up(offset, 16); offset += (size_t) SPARSE_FA_V5_MMA_M * DKQ * sizeof(half);
    offset = sparse_fa_align_up(offset, 16); offset += (size_t) DKQ * SPARSE_FA_V5_K_TILE_N * sizeof(half);
    offset = sparse_fa_align_up(offset, 16); offset += (size_t) DV * sizeof(float);
    offset = sparse_fa_align_up(offset, 16); offset += SPARSE_FA_BLOCK * sizeof(float);
    offset = sparse_fa_align_up(offset, 16); offset += SPARSE_FA_V5_K_TILE_N * sizeof(float);
    offset = sparse_fa_align_up(offset, 16); offset += (size_t) (SPARSE_FA_V5_K_TILE_N / SPARSE_FA_V5_MMA_N) *
        SPARSE_FA_V5_MMA_M * SPARSE_FA_V5_MMA_N * sizeof(float);
    offset = sparse_fa_align_up(offset, 16); offset += SPARSE_FA_V5_K_TILE_N * sizeof(int);
    return sparse_fa_align_up(offset, 16);
}

static __global__ void flash_attn_ext_sparse_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ indices,
        const char * __restrict__ mask,
        char       * __restrict__ dst,
        const int DKQ,
        const int DV,
        const int n_head,
        const int n_q,
        const int n_stream,
        const int n_kv,
        const int top_k,
        const int k_type,
        const int v_type,
        const float scale,
        const size_t q_nb0, const size_t q_nb1, const size_t q_nb2, const size_t q_nb3,
        const size_t k_nb0, const size_t k_nb1, const size_t k_nb2, const size_t k_nb3,
        const size_t v_nb0, const size_t v_nb1, const size_t v_nb2, const size_t v_nb3,
        const size_t i_nb0, const size_t i_nb1,
        const size_t m_nb0, const size_t m_nbq,
        const size_t o_nb0, const size_t o_nb1, const size_t o_nb2, const size_t o_nb3) {
    const int q_stream = (int) blockIdx.x;
    const int h        = (int) blockIdx.y;
    const int tid      = (int) threadIdx.x;

    const int qt = q_stream % n_q;
    const int st = q_stream / n_q;

    if (h >= n_head || qt >= n_q || st >= n_stream) {
        return;
    }

    __shared__ float s_reduce[SPARSE_FA_BLOCK];
    __shared__ float s_acc[SPARSE_FA_MAX_DV];
    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;
    __shared__ float s_p;
    __shared__ int   s_k_idx;

    for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
        s_acc[dv] = 0.0f;
    }
    if (tid == 0) {
        s_m = -FLT_MAX;
        s_l = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < top_k; ++i) {
        if (tid == 0) {
            s_k_idx = sparse_fa_load_i32(indices, i, q_stream, i_nb0, i_nb1);
        }
        __syncthreads();

        float partial = 0.0f;
        if (s_k_idx >= 0 && s_k_idx < n_kv) {
            for (int d = tid; d < DKQ; d += SPARSE_FA_BLOCK) {
                const float qv = sparse_fa_load_f32(q, d, h, qt, st, q_nb0, q_nb1, q_nb2, q_nb3);
                const float kv = sparse_fa_load_kv(k, k_type, d, 0, s_k_idx, 0, k_nb0, k_nb1, k_nb2, k_nb3);
                partial += qv * kv;
            }
        }

        s_reduce[tid] = partial;
        __syncthreads();

        for (int stride = SPARSE_FA_BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float score = s_k_idx >= 0 && s_k_idx < n_kv ? s_reduce[0] * scale : -FLT_MAX;
            if (mask != nullptr) {
                score += sparse_fa_load_f16(mask, i, q_stream, 0, 0, m_nb0, m_nbq, 0, 0);
            }

            if (score < -1.0e30f) {
                s_alpha = 1.0f;
                s_p     = 0.0f;
            } else {
                const float m_new = fmaxf(s_m, score);
                s_alpha = s_l == 0.0f ? 0.0f : expf(s_m - m_new);
                s_p     = expf(score - m_new);
                s_l     = s_l * s_alpha + s_p;
                s_m     = m_new;
            }
        }
        __syncthreads();

        if (s_p != 0.0f && s_k_idx >= 0 && s_k_idx < n_kv) {
            for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
                const float vv = sparse_fa_load_kv(v, v_type, dv, 0, s_k_idx, 0, v_nb0, v_nb1, v_nb2, v_nb3);
                s_acc[dv] = s_acc[dv] * s_alpha + s_p * vv;
            }
        } else if (s_alpha != 1.0f) {
            for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
                s_acc[dv] *= s_alpha;
            }
        }
        __syncthreads();
    }

    const float norm = s_l > 0.0f ? 1.0f / s_l : 0.0f;
    for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
        char * out_ptr = dst + (int64_t) dv*o_nb0 + (int64_t) h*o_nb1 + (int64_t) qt*o_nb2 + (int64_t) st*o_nb3;
        *((float *) out_ptr) = s_acc[dv] * norm;
    }
}

template<bool native_f32 = false>
static __global__ void flash_attn_ext_sparse_kernel_v5(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ indices,
        const char * __restrict__ mask,
        char       * __restrict__ dst,
        const int DKQ,
        const int DV,
        const int n_head,
        const int n_q,
        const int n_stream,
        const int n_kv,
        const int top_k,
        const int k_type,
        const int v_type,
        const float scale,
        const size_t q_nb0, const size_t q_nb1, const size_t q_nb2, const size_t q_nb3,
        const size_t k_nb0, const size_t k_nb1, const size_t k_nb2, const size_t k_nb3,
        const size_t v_nb0, const size_t v_nb1, const size_t v_nb2, const size_t v_nb3,
        const size_t i_nb0, const size_t i_nb1,
        const size_t m_nb0, const size_t m_nbq,
        const size_t o_nb0, const size_t o_nb1, const size_t o_nb2, const size_t o_nb3,
        const float * sinks = nullptr) {
    const int q_stream = (int) blockIdx.x;
    const int h        = (int) blockIdx.y;
    const int tid      = (int) threadIdx.x;
    const int warp     = tid >> 5;

    const int qt = q_stream % n_q;
    const int st = q_stream / n_q;

    if (h >= n_head || qt >= n_q || st >= n_stream) {
        return;
    }

    extern __shared__ __align__(16) unsigned char s_buf[];
    size_t offset = 0;
    offset = sparse_fa_align_up(offset, 16);
    half * s_q_mma = (half *) (s_buf + offset);
    offset += (size_t) SPARSE_FA_V5_MMA_M * DKQ * sizeof(half);
    offset = sparse_fa_align_up(offset, 16);
    half * s_k = (half *) (s_buf + offset);
    offset += (size_t) DKQ * SPARSE_FA_V5_K_TILE_N * sizeof(half);
    offset = sparse_fa_align_up(offset, 16);
    float * s_acc = (float *) (s_buf + offset);
    offset += (size_t) DV * sizeof(float);
    offset = sparse_fa_align_up(offset, 16);
    float * s_reduce = (float *) (s_buf + offset);
    offset += SPARSE_FA_BLOCK * sizeof(float);
    offset = sparse_fa_align_up(offset, 16);
    float * s_scores = (float *) (s_buf + offset);
    offset += SPARSE_FA_V5_K_TILE_N * sizeof(float);
    offset = sparse_fa_align_up(offset, 16);
    float * s_mma_scores = (float *) (s_buf + offset);
    offset += (size_t) (SPARSE_FA_V5_K_TILE_N / SPARSE_FA_V5_MMA_N) *
        SPARSE_FA_V5_MMA_M * SPARSE_FA_V5_MMA_N * sizeof(float);
    offset = sparse_fa_align_up(offset, 16);
    int * s_idx = (int *) (s_buf + offset);

    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;

    for (int linear = tid; linear < SPARSE_FA_V5_MMA_M * DKQ; linear += SPARSE_FA_BLOCK) {
        const int row = linear / DKQ;
        const int d   = linear - row * DKQ;
        const float qv = row == 0 ? sparse_fa_load_f32(q, d, h, qt, st, q_nb0, q_nb1, q_nb2, q_nb3) : 0.0f;
        s_q_mma[linear] = __float2half_rn(qv);
    }
    for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
        s_acc[dv] = 0.0f;
    }
    if (tid == 0) {
        s_m = native_f32 ? sinks[h] : -FLT_MAX;
        s_l = native_f32 ? 1.0f : 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < top_k; chunk += SPARSE_FA_V5_K_TILE_N) {
        const int tile_count = SPARSE_FA_V5_K_TILE_N < top_k - chunk ? SPARSE_FA_V5_K_TILE_N : top_k - chunk;

        if (tid < SPARSE_FA_V5_K_TILE_N) {
            s_idx[tid] = tid < tile_count ? sparse_fa_load_i32(indices, chunk + tid, q_stream, i_nb0, i_nb1) : -1;
            if constexpr (native_f32) {
                // Skip masked rows before loading canonical KV, as native does.
                if (tid < tile_count && !isfinite(sparse_fa_load_f32(mask, chunk + tid, q_stream, 0, 0, m_nb0, m_nbq, 0, 0))) s_idx[tid] = -1;
            }
        }
        __syncthreads();

        for (int linear = tid; linear < SPARSE_FA_V5_K_TILE_N * DKQ; linear += SPARSE_FA_BLOCK) {
            const int ki = linear / DKQ;
            const int d  = linear - ki * DKQ;
            const int k_idx = s_idx[ki];
            float kv = 0.0f;
            if (k_idx >= 0 && k_idx < n_kv) {
                if constexpr (native_f32) kv = sparse_fa_load_f32(k, d, 0, k_idx, 0, k_nb0, k_nb1, k_nb2, k_nb3);
                else kv = sparse_fa_load_kv(k, k_type, d, 0, k_idx, 0, k_nb0, k_nb1, k_nb2, k_nb3);
            }
            s_k[(size_t) ki * DKQ + d] = __float2half_rn(kv);
        }

        __syncthreads();

        if (warp < SPARSE_FA_V5_K_TILE_N / SPARSE_FA_V5_MMA_N) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, SPARSE_FA_V5_MMA_M, SPARSE_FA_V5_MMA_N, SPARSE_FA_V5_MMA_K,
                half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, SPARSE_FA_V5_MMA_M, SPARSE_FA_V5_MMA_N, SPARSE_FA_V5_MMA_K,
                half, nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, SPARSE_FA_V5_MMA_M, SPARSE_FA_V5_MMA_N, SPARSE_FA_V5_MMA_K,
                float> c_frag;
            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

            const int n0 = warp * SPARSE_FA_V5_MMA_N;
            for (int kk = 0; kk < DKQ; kk += SPARSE_FA_V5_MMA_K) {
                nvcuda::wmma::load_matrix_sync(a_frag, s_q_mma + kk, DKQ);
                nvcuda::wmma::load_matrix_sync(b_frag, s_k + (size_t) n0 * DKQ + kk, DKQ);
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            nvcuda::wmma::store_matrix_sync(
                s_mma_scores + (size_t) warp * SPARSE_FA_V5_MMA_M * SPARSE_FA_V5_MMA_N,
                c_frag, SPARSE_FA_V5_MMA_N, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < tile_count) {
            const int k_idx = s_idx[tid];
            const int tile = tid / SPARSE_FA_V5_MMA_N;
            const int col  = tid - tile * SPARSE_FA_V5_MMA_N;
            float score = k_idx >= 0 && k_idx < n_kv ?
                s_mma_scores[(size_t) tile * SPARSE_FA_V5_MMA_M * SPARSE_FA_V5_MMA_N + col] * scale : -FLT_MAX;
            if (mask != nullptr) {
                if constexpr (native_f32) score += sparse_fa_load_f32(mask, chunk + tid, q_stream, 0, 0, m_nb0, m_nbq, 0, 0);
                else score += sparse_fa_load_f16(mask, chunk + tid, q_stream, 0, 0, m_nb0, m_nbq, 0, 0);
            }
            s_scores[tid] = score;
        }
        __syncthreads();

        if (tid == 0) {
            float m_tile = -FLT_MAX;
            for (int ki = 0; ki < tile_count; ++ki) {
                m_tile = fmaxf(m_tile, s_scores[ki]);
            }

            const float m_new = fmaxf(s_m, m_tile);
            s_alpha = s_l == 0.0f ? 0.0f : expf(s_m - m_new);

            float l_tile = 0.0f;
            for (int ki = 0; ki < tile_count; ++ki) {
                const float p = s_scores[ki] < -1.0e30f ? 0.0f : expf(s_scores[ki] - m_new);
                s_scores[ki] = p;
                l_tile += p;
            }

            s_l = s_l * s_alpha + l_tile;
            s_m = m_new;
        }
        __syncthreads();

        for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
            float acc = s_acc[dv] * s_alpha;
            for (int ki = 0; ki < tile_count; ++ki) {
                const float p = s_scores[ki];
                if (p == 0.0f) {
                    continue;
                }
                const int k_idx = s_idx[ki];
                float vv;
                if constexpr (native_f32) vv = __half2float(__float2half_rn(sparse_fa_load_f32(v, dv, 0, k_idx, 0, v_nb0, v_nb1, v_nb2, v_nb3)));
                else vv = sparse_fa_load_kv(v, v_type, dv, 0, k_idx, 0, v_nb0, v_nb1, v_nb2, v_nb3);
                acc += p * vv;
            }
            s_acc[dv] = acc;
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_reduce[0] = s_l > 0.0f ? 1.0f / s_l : 0.0f;
    }
    __syncthreads();

    const float norm = s_reduce[0];
    for (int dv = tid; dv < DV; dv += SPARSE_FA_BLOCK) {
        char * out_ptr = dst + (int64_t) dv*o_nb0 + (int64_t) h*o_nb1 + (int64_t) qt*o_nb2 + (int64_t) st*o_nb3;
        *((float *) out_ptr) = s_acc[dv] * norm;
    }
}

// [GLM_TOPK_GATHER_MASK_V1]
// Prepare exactly the top-k cells consumed by sparse FA. Invalid physical row
// indices stay invalid (-1) and receive an IEEE-f16 -inf score mask; the sparse
// K/V loaders zero-fill -1 rows instead of dereferencing row 0. If a query has
// no usable key after invalid and causal masking, slot 0 becomes the query's
// causal-safe diagonal row so the softmax denominator remains nonzero.
static __global__ void glm_topk_gather_mask_kernel(
        const int32_t * __restrict__ src,
        int32_t * __restrict__ dst,
        const int n_queries,
        const int top_k,
        const int indices_stride,
        const int phys_rows,
        const int n_q,
        char * __restrict__ mask,
        const int64_t mask_slot_stride,
        const int64_t mask_q_stride,
        unsigned long long * __restrict__ masked_slots) {
    const int qi = (int) blockIdx.x;
    if (qi >= n_queries) {
        return;
    }

    __shared__ int s_usable;
    if (threadIdx.x == 0) {
        s_usable = 0;
    }
    __syncthreads();

    for (int slot = (int) threadIdx.x; slot < top_k; slot += (int) blockDim.x) {
        const int64_t cell = (int64_t) qi * indices_stride + slot;
        const int32_t value = src[cell];
        const bool valid = (uint32_t) value < (uint32_t) phys_rows;
        bool causally_unmasked = true;
        if (mask != nullptr) {
            const char * mask_cell =
                mask + (int64_t) qi * mask_q_stride +
                (int64_t) slot * mask_slot_stride;
            causally_unmasked = *((const uint16_t *) mask_cell) != 0xfc00u;
        }

        dst[cell] = valid ? value : -1;
        if (valid && causally_unmasked) {
            atomicExch(&s_usable, 1);
        } else if (!valid) {
            if (mask != nullptr) {
                char * mask_cell =
                    mask + (int64_t) qi * mask_q_stride +
                    (int64_t) slot * mask_slot_stride;
                *((uint16_t *) mask_cell) = 0xfc00u;
            }
            if (masked_slots != nullptr) {
                atomicAdd(masked_slots, 1ULL);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_usable == 0) {
        const int query_in_stream = qi % n_q;
        int diagonal = phys_rows - n_q + query_in_stream;
        if (diagonal < 0) {
            diagonal = 0;
        } else if (diagonal >= phys_rows) {
            diagonal = phys_rows - 1;
        }
        dst[(int64_t) qi * indices_stride] = diagonal;
        if (mask != nullptr) {
            char * mask_cell = mask + (int64_t) qi * mask_q_stride;
            *((uint16_t *) mask_cell) = 0x0000u;
        }
    }
}

void ggml_cuda_op_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * indices = dst->src[3];
    const ggml_tensor * mask    = dst->src[4];

    GGML_ASSERT(q->type       == GGML_TYPE_F32);
    GGML_ASSERT(k->type       == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(v->type       == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(mask == nullptr || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    const int DKQ      = (int) q->ne[0];
    const int n_head   = (int) q->ne[1];
    const int n_q      = (int) q->ne[2];
    const int n_stream = (int) q->ne[3];
    const int DV       = (int) v->ne[0];
    const int n_kv     = (int) k->ne[2];

    float scale;
    float max_bias;
    int32_t top_k;
    memcpy(&scale,    (const float *)   dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)   dst->op_params + 1, sizeof(float));
    memcpy(&top_k,    (const int32_t *) dst->op_params + 2, sizeof(int32_t));

    GGML_ASSERT(max_bias == 0.0f);
    GGML_ASSERT(DKQ == (int) k->ne[0]);
    GGML_ASSERT(k->ne[1] == 1 && v->ne[1] == 1);
    GGML_ASSERT(k->ne[2] == v->ne[2]);
    GGML_ASSERT(k->ne[3] == 1 && v->ne[3] == 1);
    GGML_ASSERT(DV <= SPARSE_FA_MAX_DV);
    GGML_ASSERT(top_k > 0 && top_k == (int) indices->ne[0] && top_k <= n_kv);
    GGML_ASSERT(indices->ne[1] >= (int64_t) n_q*n_stream);
    GGML_ASSERT(mask == nullptr || (mask->ne[0] == top_k &&
        (mask->ne[1] >= (int64_t) n_q*n_stream || mask->ne[3] >= (int64_t) n_q*n_stream)));
    GGML_ASSERT(dst->ne[0] == DV && dst->ne[1] == n_head && dst->ne[2] == n_q && dst->ne[3] == n_stream);

    // [GLM_TOPK_WRITE_COVERAGE_V1]
    //
    // Layer-major GLM-DSA gives indexer_topk_indices_stable-* request-owned
    // storage.  Its graph CONT is semantically an identity copy, but the
    // per-ubatch staging save happens only after this sparse consumer returns.
    // If the sliced graph leaves any destination cells untouched, neither the
    // slot-level "written" bit nor the later save can repair the read that is
    // already about to happen.  Re-materialize the identity at the actual
    // consumer boundary, on the consumer stream, over the complete CONT
    // extent.  FULL sources are the freshly selected top-k tensor; SHARED
    // sources are the per-ubatch FULL tensor restored by the staging protocol.
    // Thus the copy also makes every subsequent FULL staging save complete.
    static const bool glm_topk_write_coverage = []() {
        const char * env = getenv("LLAMA_PREFILL_GLM_TOPK_WRITE_COVERAGE");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    static constexpr const char * glm_stable_prefix =
        "indexer_topk_indices_stable-";
    const bool glm_stable_indices =
        strncmp(indices->name, glm_stable_prefix,
                strlen(glm_stable_prefix)) == 0;
    if (glm_topk_write_coverage && glm_stable_indices) {
        const ggml_tensor * source = dst->src[6];
        GGML_ASSERT(source != nullptr);
        GGML_ASSERT(source == indices->src[0]);
        GGML_ASSERT(source->type == GGML_TYPE_I32);
        GGML_ASSERT(source->data != nullptr);
        GGML_ASSERT(ggml_is_contiguous(source));
        GGML_ASSERT(ggml_is_contiguous(indices));
        GGML_ASSERT(ggml_nbytes(source) == ggml_nbytes(indices));
        if (source->data != indices->data) {
            CUDA_CHECK(cudaMemcpyAsync(
                indices->data, source->data, ggml_nbytes(indices),
                cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr,
                    "*** GLM_TOPK_WRITE_COVERAGE_V1 CUDA_ACTIVE: "
                    "tensor='%s' source=%p destination=%p bytes=%lld "
                    "stream=%p\n",
                    indices->name, source->data, indices->data,
                    (long long) ggml_nbytes(indices),
                    (void *) ctx.stream());
            fflush(stderr);
        }
    }

    const dim3 grid((unsigned) (n_q*n_stream), (unsigned) n_head, 1);
    const dim3 block(SPARSE_FA_BLOCK, 1, 1);
    const size_t mask_q_stride = mask == nullptr ? 0 :
        (mask->ne[1] >= (int64_t) n_q*n_stream ? mask->nb[1] : mask->nb[3]);

    static const bool glm_topk_gather_mask = []() {
        const char * env = getenv("LLAMA_PREFILL_GLM_TOPK_GATHER_MASK");
        return !(env && env[0] == '0');
    }();
    const bool glm_topk_gather_mask_active =
        glm_topk_gather_mask && glm_stable_indices &&
        ctx.glm_topk_gather_masked_slots != nullptr &&
        ctx.glm_topk_gather_mask_counter_active;

    ggml_cuda_pool_alloc<int32_t> prepared_indices(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> prepared_mask(ctx.pool());
    const int32_t * gather_indices = (const int32_t *) indices->data;
    const char * gather_mask = mask ? (const char *) mask->data : nullptr;
    if (glm_topk_gather_mask_active) {
        GGML_ASSERT(indices->nb[0] == sizeof(int32_t));
        const int indices_stride =
            (int) (indices->nb[1] / sizeof(int32_t));
        GGML_ASSERT(indices_stride >= top_k);
        const size_t index_words =
            (size_t) (n_q*n_stream - 1) * (size_t) indices_stride +
            (size_t) top_k;
        prepared_indices.alloc(index_words);

        if (mask != nullptr) {
            const size_t mask_bytes =
                (size_t) (n_q*n_stream - 1) * mask_q_stride +
                (size_t) (top_k - 1) * mask->nb[0] +
                sizeof(uint16_t);
            prepared_mask.alloc(mask_bytes);
            CUDA_CHECK(cudaMemcpyAsync(
                prepared_mask.get(), mask->data, mask_bytes,
                cudaMemcpyDeviceToDevice, ctx.stream()));
            gather_mask = (const char *) prepared_mask.get();
        }

        glm_topk_gather_mask_kernel<<<
            (unsigned) (n_q*n_stream), SPARSE_FA_BLOCK, 0, ctx.stream()>>>(
                (const int32_t *) indices->data,
                prepared_indices.get(),
                n_q*n_stream,
                top_k,
                indices_stride,
                n_kv,
                n_q,
                mask ? (char *) prepared_mask.get() : nullptr,
                mask ? (int64_t) mask->nb[0] : 0,
                (int64_t) mask_q_stride,
                ctx.glm_topk_gather_masked_slots);
        CUDA_CHECK(cudaGetLastError());
        gather_indices = prepared_indices.get();
    }

    static const bool use_v6 = []() {
        const char * env = std::getenv("LLAMA_MLA_SPARSE_FA_V6");
        return env && env[0] != '\0' && env[0] != '0';
    }();

    if (use_v6) {
        ggml_cuda_op_flash_attn_ext_sparse_v6(
            ctx, dst,
            glm_topk_gather_mask_active ? gather_indices : nullptr,
            glm_topk_gather_mask_active ? gather_mask : nullptr);
        return;
    }

    static const bool use_v5 = []() {
        const char * env = std::getenv("LLAMA_MLA_SPARSE_FA_V5");
        return env && env[0] != '\0' && env[0] != '0';
    }();

    if (use_v5) {
        const size_t smem_v5 = sparse_fa_v5_smem_size(DKQ, DV);
        if (DKQ % SPARSE_FA_V5_MMA_K == 0 && smem_v5 <= SPARSE_FA_V5_MAX_SMEM) {
            CUDA_SET_SHARED_MEMORY_LIMIT(flash_attn_ext_sparse_kernel_v5<false>, (int) smem_v5);
            flash_attn_ext_sparse_kernel_v5<false><<<grid, block, smem_v5, ctx.stream()>>>(
                (const char *) q->data,
                (const char *) k->data,
                (const char *) v->data,
                (const char *) gather_indices,
                gather_mask,
                (char *) dst->data,
                DKQ, DV, n_head, n_q, n_stream, n_kv, top_k, (int) k->type, (int) v->type, scale,
                q->nb[0], q->nb[1], q->nb[2], q->nb[3],
                k->nb[0], k->nb[1], k->nb[2], k->nb[3],
                v->nb[0], v->nb[1], v->nb[2], v->nb[3],
                indices->nb[0], indices->nb[1],
                mask ? mask->nb[0] : 0, mask_q_stride,
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            return;
        }

        static bool warned_v5_smem = false;
        if (!warned_v5_smem) {
            fprintf(stderr,
                    "[sparse-fa] LLAMA_MLA_SPARSE_FA_V5=1 requested but DKQ=%d, shared memory need=%zu bytes; falling back to v4\n",
                    DKQ, smem_v5);
            warned_v5_smem = true;
        }
    }

    flash_attn_ext_sparse_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const char *) q->data,
        (const char *) k->data,
        (const char *) v->data,
        (const char *) gather_indices,
        gather_mask,
        (char *) dst->data,
        DKQ, DV, n_head, n_q, n_stream, n_kv, top_k, (int) k->type, (int) v->type, scale,
        q->nb[0], q->nb[1], q->nb[2], q->nb[3],
        k->nb[0], k->nb[1], k->nb[2], k->nb[3],
        v->nb[0], v->nb[1], v->nb[2], v->nb[3],
        indices->nb[0], indices->nb[1],
        mask ? mask->nb[0] : 0, mask_q_stride,
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
}

// ============================================================
// V6: dense MMA kernel with per-row indices gather
// ============================================================

// [SPARSE_IDX_SANITIZE_MASK_V1]
// Clamp an invalid index to row 0 and mask the corresponding top-k slot.
// The mask must be repaired here because it was constructed from the raw index:
// clamping an invalid-but-causally-unmasked index alone exposes row 0.
static __global__ void sparse_idx_sanitize_kernel(
        const int32_t * __restrict__ src, int32_t * __restrict__ dst,
        const int64_t n, const int phys_rows,
        char * __restrict__ mask,
        const int64_t indices_stride,
        const int64_t mask_q_stride,
        const int top_k,
        const int n_q) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x * gridDim.x) {
        const int32_t v = src[i];
        const bool valid = (uint32_t) v < (uint32_t) phys_rows;
        dst[i] = valid ? v : 0;

        if (!valid && mask != nullptr) {
            const int64_t qi   = i / indices_stride;
            const int64_t slot = i - qi * indices_stride;
            if (qi < n_q && slot < top_k) {
                // IEEE binary16 negative infinity.
                ((uint16_t *) (mask + qi * mask_q_stride))[slot] = 0xfc00u;
            }
        }
    }
}

void ggml_cuda_op_flash_attn_ext_sparse_v6(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const int32_t * prepared_indices,
        const char * prepared_mask) {
    // Sparse op layout: src[0]=Q(F32), src[1]=K_full(F16/Q8), src[2]=V_full(F16/Q8),
    //                   src[3]=indices(I32), src[4]=mask(F16/nullptr)
    //
    // IMPORTANT Q layout note: the sparse op uses [DKQ, n_head, n_q, n_stream] for Q,
    // while the dense MMA kernel expects [DKQ, n_q, n_head, n_seq].
    // We adapt by swapping the strides and dimensions ne[1]/ne[2] (ne01/ne02) in the launch.

    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * indices = dst->src[3];
    const ggml_tensor * mask    = dst->src[4];

    GGML_ASSERT(q->type       == GGML_TYPE_F32);
    GGML_ASSERT(k->type       == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(v->type       == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(mask == nullptr || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    // Sparse op Q layout: [DKQ, n_head, n_q, n_stream]
    constexpr int DKQ = 576;
    constexpr int DV  = 512;
    const int n_head   = (int) q->ne[1];  // Q heads (sparse layout)
    const int n_q      = (int) q->ne[2];  // Q tokens (sparse layout)
    const int n_stream = (int) q->ne[3];

    float scale;
    int32_t top_k;
    memcpy(&scale, (const float *)   dst->op_params + 0, sizeof(float));
    memcpy(&top_k, (const int32_t *) dst->op_params + 2, sizeof(int32_t));

    // V6 only handles the MLA configuration: DKQ=576, DV=512, single KV head, single stream
    GGML_ASSERT(q->ne[0] == DKQ && v->ne[0] == DV);
    GGML_ASSERT(k->ne[1] == 1 && v->ne[1] == 1);
    GGML_ASSERT(k->ne[3] == 1 && v->ne[3] == 1);
    GGML_ASSERT(n_stream == 1);
    GGML_ASSERT(top_k > 0 && top_k == (int) indices->ne[0]);
    GGML_ASSERT(indices->nb[0] == sizeof(int32_t));
    GGML_ASSERT(indices->ne[1] >= n_q);
    GGML_ASSERT(mask != nullptr);
    GGML_ASSERT(mask->ne[0] == top_k);
    GGML_ASSERT(mask->ne[1] >= n_q || mask->ne[3] >= n_q);
    GGML_ASSERT(mask->nb[0] == ggml_element_size(mask));

    cudaStream_t stream = ctx.stream();

    // [2026-07-28 DSA_SERIALIZE probe, codex-v4] Diagnostic only: drain every producer
    // stream immediately before the index-gather consumer. Default OFF.
    //   clean      => an upstream producer/consumer stream edge is missing
    //   error HERE => an upstream producer already issued the invalid access
    //   IMA later  => investigate the V6 MMA/index-gather kernel itself
    // Marker for post-build DLL grep: DSA_SERIALIZE_PROBE_V1
    static const bool dsa_serialize = []() {
        const char * e = getenv("LLAMA_DSA_SERIALIZE");
        const bool on = e != nullptr && e[0] != '\0' && e[0] != '0';
        if (on) {
            fprintf(stderr, "*** DSA_SERIALIZE_PROBE_V1 ACTIVE: device-drain before sparse-FA v6\n");
            fflush(stderr);
        }
        return on;
    }();
    if (dsa_serialize) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // [2026-07-28 INDEX AUDIT probe] LLAMA_DSA_INDEX_AUDIT=1: sync-copy the indices
    // tensor to host before the gather and report invalid entries (value outside
    // [0, n_kv_rows)) per launch. Names WHERE the garbage indices come from:
    // padded/unwritten q rows vs corrupt real rows. Diagnostic only, default OFF.
    // Marker: DSA_INDEX_AUDIT_V1
    static const int dsa_index_audit = []() {
        const char * e = getenv("LLAMA_DSA_INDEX_AUDIT");
        const int n = (e && e[0] != '\0') ? atoi(e) : 0;
        if (n > 0) {
            fprintf(stderr, "*** DSA_INDEX_AUDIT_V1 ACTIVE: reporting first %d offending launches\n", n);
            fflush(stderr);
        }
        return n;
    }();
    if (dsa_index_audit > 0) {
        static int audit_last_kv_rows = -1;
        static int audit_reports = 0;
        static long long audit_launch = 0;
        audit_launch++;
        const int   kv_rows = (int) k->ne[2];  // physical rows in the sparse K view
        const int   nq_i    = (int) q->ne[2];
        const int   tk      = (int) indices->ne[0];
        const int64_t n_idx = (int64_t) tk * indices->ne[1];
        if (audit_reports < dsa_index_audit) {
            std::vector<int32_t> h_idx((size_t) n_idx);
            CUDA_CHECK(cudaMemcpyAsync(h_idx.data(), indices->data, (size_t) n_idx * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, ctx.stream()));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            const bool domain_changed = kv_rows != audit_last_kv_rows;
            audit_last_kv_rows = kv_rows;
            long long ninv_real = 0, ninv_pad = 0; int first_q = -1, first_slot = -1, first_val = 0;
            int max_valid_real = -1;
            for (int64_t qi = 0; qi < indices->ne[1]; ++qi) {
                for (int si = 0; si < tk; ++si) {
                    const int32_t vidx = h_idx[(size_t)(qi * tk + si)];
                    const bool invalid = (uint32_t) vidx >= (uint32_t) kv_rows;
                    if (qi < nq_i && !invalid && vidx > max_valid_real) {
                        max_valid_real = vidx;
                    }
                    if (invalid) {
                        if (qi < nq_i) { ninv_real++; } else { ninv_pad++; }
                        if (first_q < 0) { first_q = (int) qi; first_slot = si; first_val = vidx; }
                    }
                }
            }
            if (ninv_real + ninv_pad > 0) {
                audit_reports++;
                fprintf(stderr,
                        "*** DSA_INDEX_AUDIT launch=%lld node=%s n_q=%d idx_rows=%lld top_k=%d kv_rows=%d "
                        "max_valid_real=%d INVALID real=%lld pad=%lld first(q=%d slot=%d val=%d)\n",
                        audit_launch,
                        dst->name[0] ? dst->name : "(unnamed)",
                        nq_i, (long long) indices->ne[1], tk, kv_rows,
                        max_valid_real, ninv_real, ninv_pad, first_q, first_slot, first_val);
                fflush(stderr);
            } else if (domain_changed) {
                // One line per row-domain transition distinguishes absolute cache
                // rows from invocation-local rows without logging every layer.
                fprintf(stderr,
                        "*** DSA_INDEX_AUDIT_DOMAIN launch=%lld n_q=%d idx_rows=%lld top_k=%d "
                        "kv_rows=%d max_valid_real=%d\n",
                        audit_launch, nq_i, (long long) indices->ne[1], tk,
                        kv_rows, max_valid_real);
                fflush(stderr);
            }
        }
    }

    ggml_cuda_pool & pool = ctx.pool();

    // Convert K/V to F16 if needed
    ggml_cuda_pool_alloc<half> K_f16(pool);
    ggml_cuda_pool_alloc<half> V_f16(pool);

    const char * K_data = (const char *) k->data;
    size_t K_nb1 = k->nb[1], K_nb2 = k->nb[2], K_nb3 = k->nb[3];

    const char * V_data = (const char *) v->data;
    size_t V_nb1 = v->nb[1], V_nb2 = v->nb[2], V_nb3 = v->nb[3];

    if (k->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(k->type);
        const size_t ts = ggml_type_size(k->type);

        K_f16.alloc(ggml_nelements(k));
        if (ggml_is_contiguously_allocated(k)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(k->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(k), stream);

            K_nb1 = k->nb[1] * bs * sizeof(half) / ts;
            K_nb2 = k->nb[2] * bs * sizeof(half) / ts;
            K_nb3 = k->nb[3] * bs * sizeof(half) / ts;
        } else {
            GGML_ASSERT(k->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(k->type);
            GGML_ASSERT(to_fp16 != nullptr);
            const int64_t s01 = K_nb1 / ts;
            const int64_t s02 = K_nb2 / ts;
            const int64_t s03 = K_nb3 / ts;
            to_fp16(K_data, K_f16.ptr, k->ne[0], k->ne[1], k->ne[2], k->ne[3], s01, s02, s03, stream);

            K_nb1 = k->ne[0] * sizeof(half);
            K_nb2 = k->ne[1] * K_nb1;
            K_nb3 = k->ne[2] * K_nb2;
        }
        K_data = (const char *) K_f16.ptr;
    }

    // V is often a view of K (MLA K-reuse). Detect and point to converted K data.
    const bool V_is_K_view = (v->view_src && (v->view_src == k ||
        (v->view_src == k->view_src && v->view_offs == k->view_offs)));

    if (V_is_K_view) {
        GGML_ASSERT(v->view_offs == 0);
        V_data = K_data;
        V_nb1  = K_nb1;
        V_nb2  = K_nb2;
        V_nb3  = K_nb3;
    } else if (v->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(v->type);
        const size_t ts = ggml_type_size(v->type);

        V_f16.alloc(ggml_nelements(v));
        if (ggml_is_contiguously_allocated(v)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(v->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16(V_data, V_f16.ptr, ggml_nelements(v), stream);

            V_nb1 = v->nb[1] * bs * sizeof(half) / ts;
            V_nb2 = v->nb[2] * bs * sizeof(half) / ts;
            V_nb3 = v->nb[3] * bs * sizeof(half) / ts;
        } else {
            GGML_ASSERT(v->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(v->type);
            GGML_ASSERT(to_fp16 != nullptr);
            const int64_t s01 = V_nb1 / ts;
            const int64_t s02 = V_nb2 / ts;
            const int64_t s03 = V_nb3 / ts;
            to_fp16(V_data, V_f16.ptr, v->ne[0], v->ne[1], v->ne[2], v->ne[3], s01, s02, s03, stream);

            V_nb1 = v->ne[0] * sizeof(half);
            V_nb2 = v->ne[1] * V_nb1;
            V_nb3 = v->ne[2] * V_nb2;
        }
        V_data = (const char *) V_f16.ptr;
    }

    // indices: [top_k, n_q] with nb[0]=sizeof(int32), nb[1]=top_k*sizeof(int32)
    // For query token jt: indices at indices_ptr + jt * top_k
    const int * indices_ptr = prepared_indices ?
        (const int *) prepared_indices : (const int *) indices->data;
    const char * mask_ptr = prepared_mask ?
        prepared_mask : (const char *) mask->data;
    const int32_t indices_stride = (int32_t) (indices->nb[1] / sizeof(int32_t));

    // [2026-07-28 SPARSE_IDX_SANITIZE_V1] Legacy fallback for non-GLM callers
    // and GLM_TOPK_GATHER_MASK_V1=0. The production GLM path arrives with
    // prepared -1 indices and bypasses this clamp-to-row-0 sanitizer.
    const int phys_kv_rows = (int) k->ne[2];
    const size_t mask_q_stride =
        mask->ne[1] >= n_q ? mask->nb[1] : mask->nb[3];
    static const bool sanitize_invalid_mask = []() {
        const char * e = std::getenv("LLAMA_DSA_SANITIZE_INVALID_MASK");
        // Safe default ON; =0 restores the old index-only sanitizer for A/B.
        return !(e != nullptr && e[0] == '0');
    }();
    ggml_cuda_pool_alloc<int32_t> indices_sane(ctx.pool());
    if (prepared_indices == nullptr) {
        const int64_t idx_rows  = indices->ne[1];
        const int64_t idx_total = (int64_t) indices_stride * idx_rows;
        indices_sane.alloc((size_t) idx_total);
        sparse_idx_sanitize_kernel<<<(unsigned) ((idx_total + 255) / 256), 256, 0, ctx.stream()>>>(
            (const int32_t *) indices->data, indices_sane.get(), idx_total, phys_kv_rows,
            sanitize_invalid_mask ? (char *) mask_ptr : nullptr,
            indices_stride,
            (int64_t) mask_q_stride,
            top_k, n_q);
        CUDA_CHECK(cudaGetLastError());
        indices_ptr = (const int *) indices_sane.get();
    }

    // Dense MMA kernel expects Q in [DKQ, n_q, n_head, n_seq] layout.
    // Sparse op gives Q in [DKQ, n_head, n_q, n_seq] layout.
    // Adapt by swapping ne[1]/ne[2] and nb[1]/nb[2] before passing to kernel.
    //   dense ne01 (Q token count) = sparse q->ne[2] = n_q
    //   dense ne02 (Q head count)  = sparse q->ne[1] = n_head
    //   dense nb01 (token stride)  = sparse q->nb[2]
    //   dense nb02 (head stride)   = sparse q->nb[1]
    // With K having ne[2]=1 head: gqa_ratio = ne02 / K->ne[2] = n_head / 1 = n_head.

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // [GLM_DSA_DETERMINISTIC_SPARSE_FA_V1]
    // GLM-DSA marks this op deterministic by default. Pick a block count that
    // divides ntiles_total so no block boundary can split a query/head tile:
    // every softmax/V reduction then runs in one block with a fixed order and
    // flash_attn_stream_k_fixup is not launched. The op flag is absent for
    // other architectures, preserving their existing launch byte-for-byte.
    // LLAMA_MLA_SPARSE_FA_V6_TILE_ALIGNED_GRID=0 omits the GLM op flag and
    // restores the legacy occupancy grid plus stream-K seam/fixup for A/B.
    const bool deterministic_stream_k =
        ggml_get_op_params_i32(dst, 3) != 0;
    if (deterministic_stream_k) {
        static bool deterministic_logged = false;
        if (!deterministic_logged) {
            deterministic_logged = true;
            fprintf(stderr,
                "*** GLM_DSA_DETERMINISTIC_SPARSE_FA_V1 ACTIVE: "
                "tile-aligned grid; stream-K fixup disabled\n");
            fflush(stderr);
        }
    }

    // Generic opt-in remains available to non-GLM sparse-v6 callers.
    //   LLAMA_MLA_SPARSE_FA_V6_TILE_ALIGNED_GRID=1 : use the same aligned grid.
    //   LLAMA_MLA_SPARSE_FA_V6_NCOLS64=1 : instantiate ncols2=64 (1 head-tile => gather sparse K/V
    //     ONCE for all 64 heads instead of twice for 2x32; auto-enables fp32 VKQ acc via the
    //     use_f32_acc template gate). RISK: prior SM120 multi-ubatch garble (Phase22 angle53) -
    //     hypothesis is it was the stream-K seam, so this path also forces the tile-aligned grid.
    //     MUST validate recall/coherence, not just speed. Also forces tile-aligned grid.
    static const bool v6_tile_aligned_grid_env = []() {
        const char * env = getenv("LLAMA_MLA_SPARSE_FA_V6_TILE_ALIGNED_GRID");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    static const bool v6_ncols64_env = []() {
        const char * env = getenv("LLAMA_MLA_SPARSE_FA_V6_NCOLS64");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();

    auto launch_sparse_v6 = [&](auto ncols2_tag) {
        constexpr int ncols1 = 1;   // one Q token per tile
        constexpr int ncols2 = decltype(ncols2_tag)::value;
        constexpr int ncols  = ncols1 * ncols2;
        constexpr bool use_logit_softcap = false;

        const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
        const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
        const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
        const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
        const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
        const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
        const int nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);
        const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));
        const int nwarps         = nthreads / WARP_SIZE;

        const size_t nbytes_shared_KV_1stage = (size_t) nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t nbytes_shared_KV_2stage = (size_t) nbatch_fa *        (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t nbytes_shared_Q         = (size_t) ncols       * (DKQ/2 + 4) * sizeof(half2);
        const size_t nbytes_shared_mask      = (size_t) ncols1      * (nbatch_fa/2 + 4) * sizeof(half2);
        const size_t nbytes_shared_combine   = (size_t) nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
        const size_t nbytes_shared_KV        = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;
        const size_t nbytes_shared_total     = std::max(nbytes_shared_combine, Q_in_reg ?
            std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask) :
                     nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

        const int ne01_dense   = n_q;        // token count -> iter_j axis
        const int ne02_dense   = n_head;     // head count  -> gqa_ratio axis
        const int iter_j       = (ne01_dense + ncols1 - 1) / ncols1;
        const int gqa_ratio    = ne02_dense; // n_head / 1 (one KV head)
        const int iter_z_gqa   = (gqa_ratio + ncols2 - 1) / ncols2;
        const int ntiles_total = iter_j * iter_z_gqa * 1 /* ne12 */ * 1 /* ne03 */;

        auto kernel_fn = V_is_K_view
            ? flash_attn_ext_f16_sparse<DKQ, DV, ncols1, ncols2, use_logit_softcap, true>
            : flash_attn_ext_f16_sparse<DKQ, DV, ncols1, ncols2, use_logit_softcap, false>;

        const dim3 block_dim(WARP_SIZE, nwarps, 1);
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes_shared_total));

        int max_blocks_per_sm = 1;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel_fn,
            block_dim.x * block_dim.y * block_dim.z, nbytes_shared_total));
        GGML_ASSERT(max_blocks_per_sm > 0);

        const int max_blocks = max_blocks_per_sm * nsm;
        const bool tile_aligned_grid =
            deterministic_stream_k ||
            v6_tile_aligned_grid_env ||
            v6_ncols64_env;
        int launch_blocks = tile_aligned_grid ?
            sparse_fa_v6_pick_tile_aligned_blocks(ntiles_total, max_blocks, nsm) : max_blocks;
        // The deterministic GLM contract must survive Release/NDEBUG builds:
        // if the divisor picker ever returns an invalid grid, fall back to one
        // block per tile instead of relying on the assertion below.
        const bool forced_one_block_per_tile =
            deterministic_stream_k &&
            (launch_blocks <= 0 || ntiles_total % launch_blocks != 0);
        if (forced_one_block_per_tile) {
            launch_blocks = ntiles_total;
        }
        const dim3 blocks_num((unsigned) launch_blocks, 1, 1);

        // Allocate fixup buffer when SMs don't evenly divide tiles
        ggml_cuda_pool_alloc<float2> dst_meta(pool);
        const bool need_fixup = (ntiles_total % (int) blocks_num.x != 0);
        GGML_ASSERT(!deterministic_stream_k || !need_fixup);
        static const bool glm_sparse_fa_v6_launch_trace = []() {
            const char * env =
                getenv("LLAMA_GLM_SPARSE_FA_V6_LAUNCH_TRACE");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }();
        const bool fixup_launched = need_fixup;
        if (need_fixup) {
            dst_meta.alloc((size_t) blocks_num.x * ncols * (2 + DV/2));
        }

        const float max_bias      = 0.0f;
        const float logit_softcap = 0.0f;
        const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));
        const float m0 = 1.0f, m1 = 1.0f;

        const uint3 ne01_kparam = init_fastdiv_values(ne01_dense);

        const int32_t Q_nb01 = (int32_t) q->nb[2];  // token stride
        const int32_t Q_nb02 = (int32_t) q->nb[1];  // head stride
        const int32_t Q_nb03 = (int32_t) q->nb[3];  // seq stride

        const int32_t K_ne11 = top_k;
        const int32_t K_nb11 = (int32_t) K_nb2;
        const int32_t K_nb12 = (int32_t) K_nb1;
        const int64_t K_nb13 = (int64_t) K_nb3;

        const int32_t V_nb21 = (int32_t) V_nb2;
        const int32_t V_nb22 = (int32_t) V_nb1;
        const int64_t V_nb23 = (int64_t) V_nb3;

        kernel_fn<<<blocks_num, block_dim, nbytes_shared_total, stream>>>(
            (const char *) q->data,
            K_data,
            V_data,
            mask_ptr,
            indices_ptr,
            indices_stride,
            nullptr,    // sinks: not used
            nullptr,    // KV_max: not used
            (float *) dst->data,
            need_fixup ? dst_meta.ptr : nullptr,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            (int32_t) DKQ, ne01_kparam, (int32_t) ne02_dense, (int32_t) n_stream,
            Q_nb01, Q_nb02, Q_nb03,
            (int32_t) DKQ, K_ne11, (int32_t) 1 /* ne12 */, (int32_t) 1 /* ne13 */,
            K_nb11, K_nb12, K_nb13,
            V_nb21, V_nb22, V_nb23,
            (int32_t) top_k, (int32_t) n_q, (int32_t) n_stream,
            (int32_t) mask_q_stride, (int32_t) 0, (int64_t) 0
        );
        CUDA_CHECK(cudaGetLastError());

        // Fixup pass if needed (stream-K seam)
        if (fixup_launched) {
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};
            flash_attn_stream_k_fixup<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, stream>>>
                ((float *) dst->data, dst_meta.ptr,
                 ne01_dense, ne02_dense, (int) n_stream, top_k, 1 /* ne12 */, nbatch_fa);
        }
        CUDA_CHECK(cudaGetLastError());
        if (glm_sparse_fa_v6_launch_trace) {
            fprintf(stderr,
                "GLM_SPARSE_FA_V6_LAUNCH_V1 "
                "k_ne11=%lld n_q=%d top_k=%d ncols=%d "
                "ntiles_total=%d blocks_num.x=%u need_fixup=%d "
                "deterministic_stream_k=%d forced_one_block_per_tile=%d "
                "fixup_launched=%d\n",
                (long long) K_ne11, n_q, top_k, ncols,
                ntiles_total, blocks_num.x, need_fixup ? 1 : 0,
                deterministic_stream_k ? 1 : 0,
                forced_one_block_per_tile ? 1 : 0,
                fixup_launched ? 1 : 0);
            fflush(stderr);
        }
    };

    if (v6_ncols64_env) {
        launch_sparse_v6(std::integral_constant<int, 64>{});
    } else {
        launch_sparse_v6(std::integral_constant<int, 32>{});
    }
}

// Isolated native same-input replay only. The production graph never calls it.
// Reuse GLM v5 WMMA QK plus F32 online P/V; canonical KV remains F32.
static ggml_cuda_dispatch_counter dsv41_sparse_replay_queued{0};
extern "C" uint64_t ggml_dsv41_cuda_sparse_f32_replay_queued() { return dsv41_sparse_replay_queued.load(); }
extern "C" bool ggml_dsv41_cuda_sparse_f32_replay(const float * q, const float * kv,
        const int32_t * ids, const float * mask, const float * sinks, float * out,
        int queries, int rows, int topk, int heads, void * stream) {
    if (!q || !kv || !ids || !mask || !sinks || !out || queries<1 || queries>4096 ||
        rows<1 || rows>131072 || topk<1 || topk>640 || heads<1 || heads>64) return false;
    const size_t smem=sparse_fa_v5_smem_size(512,512);
    CUDA_SET_SHARED_MEMORY_LIMIT(flash_attn_ext_sparse_kernel_v5<true>, (int)smem);
    flash_attn_ext_sparse_kernel_v5<true><<<dim3(queries,heads),SPARSE_FA_BLOCK,smem,static_cast<cudaStream_t>(stream)>>>(
        (const char*)q,(const char*)kv,(const char*)kv,(const char*)ids,(const char*)mask,(char*)out,
        512,512,heads,queries,1,rows,topk,GGML_TYPE_F32,GGML_TYPE_F32,1.f/sqrtf(512.f),
        4,512*4,size_t(512)*heads*4,0,
        4,512*4,512*4,0, 4,512*4,512*4,0,
        4,size_t(topk)*4,4,size_t(topk)*4,
        4,512*4,size_t(512)*heads*4,0,sinks);
    CUDA_CHECK(cudaGetLastError());dsv41_sparse_replay_queued.fetch_add(1);return true;
}
