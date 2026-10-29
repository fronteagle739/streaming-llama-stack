/*
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek
Copyright (c) 2026 The ds4-win contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
// Existing ds4-win token-tile CUDA bodies, copied verbatim.
// Ordered per-query graph records replace bitmap-union construction so long
// contexts do not require a context-sized shared-memory bitmap. Each query
// sees each selected ID once; records for other queries carry disjoint masks.
#include "dsv41-ds4-token-tile.cuh"
#include "ggml-dsv41-shared-sparse.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <type_traits>
/* -------------------------------------------------------------------------
 * DS4_ATTN_TOKENTILE: token-tile HMMA indexed attention (opt-in).
 * Ported from the pass-12 standalone prototype with fixed STAGE_ROWS=32.
 * Four-token tiles reduce the selected-row union while eight heads keep the
 * 32-row MMA shape full.
 */

static constexpr uint32_t kTTTileTokens = 4u;
static constexpr uint32_t kTTG = 8u;
static constexpr uint32_t kTTM = 32u;
static constexpr uint32_t kTTStageRows = 32u;
static constexpr uint32_t kTTRawWindow = 128u;
static constexpr uint32_t kTTHeadDim = 512u;
static constexpr uint32_t kTTWarps = 16u;
static constexpr uint32_t kTTThreads = 512u;
#include "dsv41-ds4-token-union.cuh"
static constexpr uint32_t kTTScoreKQuarters = 4u;
static constexpr uint32_t kTTScoreKSliceDim = kTTHeadDim / kTTScoreKQuarters;
static constexpr uint32_t kTTScoreKStepsPerQuarter = kTTScoreKSliceDim / 16u;
static constexpr uint32_t kTTRecordRingPlanes = 4u;
static constexpr uint32_t kTTProbStride = 40u;
static constexpr uint32_t kTTRingChunkBytes = 16u;
static constexpr uint32_t kTTRingChunksPerRow =
    (kTTHeadDim * sizeof(half)) / kTTRingChunkBytes;
static constexpr size_t kTTSmemHardCap = 90ull * 1024ull;

static_assert(kTTM == kTTTileTokens * kTTG, "token-tile M must be 4 tokens x G8");
static_assert(kTTProbStride == kTTStageRows + 8u, "token-tile prob stride changed");
static_assert(kTTScoreKSliceDim % 16u == 0, "token-tile score K split changed");
static_assert(kTTRingChunksPerRow == 64u, "token-tile KV ring expects 64 chunks");
static_assert(sizeof(int2) == 8u, "token-tile union record must remain 8 bytes");

template <uint32_t TT_STAGE_ROWS>
struct tt_TokentileLayout {
    static constexpr uint32_t prob_stride = TT_STAGE_ROWS + 8u;
    static constexpr uint32_t ring_plane_bytes =
        TT_STAGE_ROWS * kTTRingChunksPerRow * kTTRingChunkBytes;
    static constexpr uint32_t ring_plane_elems = ring_plane_bytes / sizeof(half);
};

__device__ static float tt_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float tt_warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ uint32_t tt_lane_id(void) {
    return threadIdx.x & 31u;
}

__device__ __forceinline__ uint32_t tt_warp_id(void) {
    return threadIdx.x >> 5u;
}

__device__ __forceinline__ int tt_mma_c_i(uint32_t lane, int l) {
    return ((l >> 1) << 3) + (int)(lane >> 2);
}

__device__ __forceinline__ int tt_mma_c_j(uint32_t lane, int l) {
    return (int)((lane & 3u) << 1) + (l & 1);
}

__device__ __forceinline__ unsigned tt_smem_addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t tt_ring_off_bytes(uint32_t row, uint32_t c) {
    return (row * kTTRingChunksPerRow + (c ^ (row & 7u))) * kTTRingChunkBytes;
}

__device__ __forceinline__ void tt_ldmatrix_x4_addr(uint32_t (&r)[4], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x4(uint32_t (&r)[4], const void *p) {
    tt_ldmatrix_x4_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_ldmatrix_x2(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_trans_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_mma_m16n8k16_f16_f32(
        float *d,
        const uint32_t (&a)[4],
        const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t d0 = __float_as_uint(d[0]);
    uint32_t d1 = __float_as_uint(d[1]);
    uint32_t d2 = __float_as_uint(d[2]);
    uint32_t d3 = __float_as_uint(d[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0);
    d[1] = __uint_as_float(d1);
    d[2] = __uint_as_float(d2);
    d[3] = __uint_as_float(d3);
#else
    (void)a;
    (void)b;
#endif
}

__device__ __forceinline__ unsigned char *tt_align16(unsigned char *p) {
    uintptr_t x = reinterpret_cast<uintptr_t>(p);
    x = (x + 15u) & ~uintptr_t(15u);
    return reinterpret_cast<unsigned char *>(x);
}

__device__ __forceinline__ void tt_zero_16B(void *dst) {
    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
}

__device__ __forceinline__ void tt_zero_8B(void *dst) {
    *reinterpret_cast<int2 *>(dst) = make_int2(0, 0);
}

__device__ __forceinline__ void tt_cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_16B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_16B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_8B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_8B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void tt_cp_async_wait_group(void) {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ uint32_t tt_score_partial_slot(
        uint32_t kq,
        uint32_t m,
        uint32_t r) {
    return (kq + r + m) & 3u;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_store_score_partial(
        float4 * __restrict__ partials,
        uint32_t kq,
        uint32_t m,
        uint32_t r,
        float v) {
    float *dst = &partials[m * TT_STAGE_ROWS + r].x;
    dst[tt_score_partial_slot(kq, m, r)] = v;
}

__device__ __forceinline__ float4 tt_load_score_partial_record(
        const float4 * __restrict__ p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float x, y, z, w;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "r"(tt_smem_addr(p)));
    return make_float4(x, y, z, w);
#else
    return *p;
#endif
}

__device__ __forceinline__ void tt_issue_cp_async_row(
        half * __restrict__ dst,
        uint32_t rr,
        uint32_t lane16,
        const half * __restrict__ src,
        bool live) {
    const char *src_b = reinterpret_cast<const char *>(src);
    char *dst_b = reinterpret_cast<char *>(dst);

#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t chunk = lane16 + i * 16u;
        char *db = dst_b + tt_ring_off_bytes(rr, chunk);
        tt_cp_async_16B(db, live ? static_cast<const void *>(src_b + chunk * 16u)
                                 : static_cast<const void *>(db),
                        live);
    }
}

__device__ __forceinline__ uint32_t tt_stage_raw_rows(
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count) {
    uint32_t raw_rows = 0;
    if (row0 < raw_union_count) {
        const uint32_t raw_left = raw_union_count - row0;
        raw_rows = raw_left < nr ? raw_left : nr;
    }
    return raw_rows;
}


__global__ static void __launch_bounds__(256, 1) attention_tokentile_raw_mirror_kernel(
        half *dst,
        const float *raw_kv,
        const int32_t *seq_id,
        uint32_t tt_run_pos0,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (d0 >= head_dim) return;
    half *dst_row = dst + (uint64_t)row * head_dim;
    if (row < raw_row_min) {
        dst_row[d0] = __float2half(0.0f);
        if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const int64_t p = (int64_t)tt_run_pos0 - (int64_t)(kTTRawWindow - 1u) + (int64_t)row;
    uint32_t slot = 0u;
    if (seq_id) {
        slot = (uint32_t)seq_id[0] * raw_cap + (uint32_t)((uint64_t)p % raw_cap);
    } else {
        const uint32_t rel = (uint32_t)(p - (int64_t)first_raw_pos);
        slot = (raw_start + rel) % raw_cap;
    }
    const float *src = raw_kv + (uint64_t)slot * head_dim;
    dst_row[d0] = __float2half(src[d0]);
    if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(src[d0 + 1u]);
}

__global__ static void __launch_bounds__(256, 1) attention_tokentile_comp_mirror_kernel(
        half *dst,
        const float *comp_kv,
        uint32_t n_comp,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t c4 = threadIdx.x;
    if (row >= n_comp || c4 >= (head_dim >> 2u)) return;
    const float4 v = ((const float4 *)(comp_kv + (uint64_t)row * head_dim))[c4];
    half *out = dst + (uint64_t)row * head_dim + (c4 << 2u);
    out[0] = __float2half(v.x);
    out[1] = __float2half(v.y);
    out[2] = __float2half(v.z);
    out[3] = __float2half(v.w);
}


template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_record_stage_cp_async(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile) {
    static_assert(TT_STAGE_ROWS == 32u, "token-tile record issue is fixed at R32");
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;
    const bool comp_live = active && rr >= raw_rows && rr < nr;
    if (comp_live && lane16 == 0u) {
        int2 *dst = rec_plane + rr;
        const uint32_t ci = row0 + rr - raw_union_count;
        tt_cp_async_8B(dst, union_records_tile + ci, true);
    }
}

template <uint32_t TT_STAGE_ROWS, bool USE_SMEM_RECORDS>
__device__ __forceinline__ void tt_issue_kv_stage_cp_async(
        half * __restrict__ dst,
        const int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile,
        uint32_t tile_base,
        const half * __restrict__ raw_kv,
        const half * __restrict__ comp_kv,
        uint32_t tid) {
    constexpr uint32_t kCp16PerRow = (kTTHeadDim * sizeof(half)) / 16u;
    static_assert(kCp16PerRow == 64u, "expected 64 cp.async chunks per f16 KV row");
    static_assert(TT_STAGE_ROWS == 32u, "token-tile KV issue is fixed at R32");
    (void)tid;
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;

    if (active && rr < raw_rows) {
        const uint32_t sr = row0 + rr;
        const half *src = raw_kv + (uint64_t)(tile_base + sr) * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= raw_rows && rr < nr) {
        uint32_t comp_id = 0u;
        if (USE_SMEM_RECORDS) {
            comp_id = (uint32_t)rec_plane[rr].x;
        } else {
            const uint32_t ci = row0 + rr - raw_union_count;
            comp_id = (uint32_t)union_records_tile[ci].x;
        }
        const half *src = comp_kv + (uint64_t)comp_id * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= nr) {
        tt_issue_cp_async_row(dst, rr, lane16, NULL, false);
    }
}

template <uint32_t TT_G>
__device__ __forceinline__ void tt_load_score_q_frag(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const float * __restrict__ q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kTTG = TT_G, kTTM = kTTTileTokens * TT_G;
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t k_base = kq * kTTScoreKSliceDim;
    const uint32_t lane = tt_lane_id();
    const uint32_t a_group = lane >> 2u;
    const uint32_t a_col_pair = (lane & 3u) << 1u;
#pragma unroll
    for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
        const uint32_t k0 = k_base + kt * 16u;
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint32_t m = mtile * 16u + a_group + ((r & 1u) ? 8u : 0u);
            const uint32_t tok = m / kTTG;
            const uint32_t h = m - tok * kTTG;
            const uint32_t gt = tile_base + tok;
            const uint32_t gh = head_base + h;
            const uint32_t d = k0 + a_col_pair + ((r & 2u) ? 8u : 0u);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (gt < n_tokens && gh < n_head) {
                const float *q_row = q + ((uint64_t)gt * n_head + gh) * kTTHeadDim;
                x0 = q_row[d];
                x1 = q_row[d + 1u];
            }
            const half2 packed = __floats2half2_rn(x0, x1);
            q_frag[kt][r] =
                (uint32_t)__half_as_ushort(__low2half(packed)) |
                ((uint32_t)__half_as_ushort(__high2half(packed)) << 16);
        }
    }
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
__device__ __forceinline__ void tt_hmma_score_stage(
        float4 * __restrict__ partial_scores,
        const uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ kv_cur,
        uint32_t nr,
        float score_scale) {
    constexpr uint32_t kTTM = kTTTileTokens * TT_G;
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    constexpr uint32_t kNtiles = TT_STAGE_ROWS / 8u;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t lane = tt_lane_id();
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t score_row_lane = lane & 7u;
    const uint32_t score_chunk_lane = (lane >> 3u) & 1u;
    const uint32_t score_chunk_base = kq * (kTTScoreKSliceDim / 8u) + score_chunk_lane;
#pragma unroll
    for (uint32_t ntile = 0; ntile < kNtiles; ++ntile) {
        const uint32_t row_base = ntile * 8u;
        if (row_base < nr) {
            float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint32_t score_row = row_base + score_row_lane;
#pragma unroll
            for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
                uint32_t b[2];
                tt_ldmatrix_x2_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(score_row, score_chunk_base + kt * 2u));
                tt_mma_m16n8k16_f16_f32(s_frag, q_frag[kt], b);
            }
            const uint32_t m0 = mtile * 16u + (lane >> 2u);
            const uint32_t r0 = row_base + ((lane & 3u) << 1u);
            const uint32_t r1 = r0 + 1u;
            const uint32_t m1 = m0 + 8u;
            if (r0 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r0, s_frag[0] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r0, s_frag[2] * score_scale);
            }
            if (r1 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r1, s_frag[1] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r1, s_frag[3] * score_scale);
            }
        }
    }
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
__device__ __forceinline__ void tt_softmax_stage(
        half * __restrict__ probs,
        float * __restrict__ stage_rescale,
        float * __restrict__ max_s,
        float * __restrict__ sum_s,
        const float4 * __restrict__ scores,
        const int2 * __restrict__ records,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        uint32_t tile_count,
        uint32_t tile_base,
        uint32_t raw_row_min) {
    constexpr uint32_t kTTG = TT_G, kTTM = kTTTileTokens * TT_G;
    constexpr uint32_t kMPerWarp = kTTM / kTTWarps;
    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static_assert(kTTM % kTTWarps == 0u, "softmax maps integral m rows per warp");
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t sr = row0 + lane;
    const bool lane_live = lane < TT_STAGE_ROWS && lane < nr;
    const bool raw_slot = lane_live && sr < raw_union_count;
    uint16_t comp_mask = 0u;
    if (lane_live && !raw_slot) {
        comp_mask = (uint16_t)records[lane].y;
    }
    const uint32_t prob_lane_base = warp * kProbStride + lane;

#pragma unroll
    for (uint32_t mi = 0; mi < kMPerWarp; ++mi) {
        const uint32_t m = warp + mi * kTTWarps;
        const uint32_t tok = m / kTTG;
        const bool valid_token = tok < tile_count;
        const uint32_t score_idx = m * TT_STAGE_ROWS + lane;
        const uint32_t prob_idx = prob_lane_base + mi * kTTWarps * kProbStride;
        float score = -INFINITY;
        if (lane_live && valid_token) {
            const bool selected = raw_slot
                ? (((uint32_t)(sr - tok) < kTTRawWindow) && (tile_base + sr >= raw_row_min))
                : ((comp_mask & (uint16_t)(1u << tok)) != 0u);
            if (selected) {
                const float4 parts = tt_load_score_partial_record(scores + score_idx);
                const float s01 = parts.x + parts.y;
                const float s23 = parts.z + parts.w;
                score = s01 + s23;
            }
        }

        const float stage_m = tt_warp_max_f32(score);
        const float old_m = max_s[m];
        const float new_m = fmaxf(old_m, stage_m);
        float old_scale = 1.0f;
        if (new_m != -INFINITY) {
            old_scale = (old_m == -INFINITY) ? 0.0f : expf(old_m - new_m);
        }
        const float row_scale = (score == -INFINITY || new_m == -INFINITY)
            ? 0.0f
            : expf(score - new_m);
        const float stage_sum = tt_warp_sum_f32(row_scale);

        if (lane < TT_STAGE_ROWS) {
            probs[prob_idx] = __float2half(lane < nr ? row_scale : 0.0f);
        }
        if (lane == 0u) {
            max_s[m] = new_m;
            sum_s[m] = sum_s[m] * old_scale + stage_sum;
            stage_rescale[m] = old_scale;
        }
    }
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
__device__ __forceinline__ void tt_pv_mma_stage(
        float (&o_acc)[64],
        const half * __restrict__ probs,
        const float * __restrict__ stage_rescale,
        const half * __restrict__ kv_cur) {
    constexpr uint32_t kTTM = kTTTileTokens * TT_G;
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = TT_G == 16u ? 0u : 8u;
    constexpr uint32_t kPvNTiles = TT_G == 16u ? 4u : 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
        const float *scale0 = stage_rescale + mtile * 16u + (lane >> 2u);
        const float *scale1 = scale0 + 8u;
        const float rs0 = *scale0;
        const float rs1 = *scale1;
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
            const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
            o_acc[idx + 0u] *= rs0;
            o_acc[idx + 1u] *= rs0;
            o_acc[idx + 2u] *= rs1;
            o_acc[idx + 3u] *= rs1;
        }
    }

    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    constexpr unsigned kPvAStepBytes = 16u * sizeof(half);
    constexpr unsigned kPvMtileBytes = 16u * kProbStride * sizeof(half);
    const unsigned probs_lane_base =
        tt_smem_addr(probs) +
        (unsigned)(((lane & 15u) * (kProbStride / 2u) +
                    (lane >> 4u) * 4u) * sizeof(uint32_t));
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t pv_row_lane = lane & 15u;
    const uint32_t pv_chunk_base = pv_warp * kPvNTiles;
#pragma unroll
    for (uint32_t kt = 0; kt < TT_STAGE_ROWS / 16u; ++kt) {
        const unsigned probs_kt_base = probs_lane_base + (unsigned)(kt * kPvAStepBytes);
        const uint32_t pv_row = kt * 16u + pv_row_lane;
#pragma unroll
        for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
            uint32_t a[4];
            tt_ldmatrix_x4_addr(a, probs_kt_base + (unsigned)(mtile * kPvMtileBytes));
#pragma unroll
            for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
                uint32_t b[2];
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
                tt_ldmatrix_x2_trans_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(pv_row, pv_chunk_base + ntile));
                tt_mma_m16n8k16_f16_f32(o_acc + idx, a, b);
            }
        }
    }
}

template <uint32_t TT_G>
__device__ __forceinline__ void tt_pv_mma_epilogue(
        const float (&o_acc)[64],
        const float * __restrict__ final_scale,
        float * __restrict__ heads,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kTTG = TT_G, kTTM = kTTTileTokens * TT_G;
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = TT_G == 16u ? 0u : 8u;
    constexpr uint32_t kPvNTiles = TT_G == 16u ? 4u : 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2) + (uint32_t)l;
                const uint32_t m = mtile * 16u + (uint32_t)tt_mma_c_i(lane, l);
                const uint32_t tok = m / kTTG;
                const uint32_t h = m - tok * kTTG;
                const uint32_t gt = tile_base + tok;
                const uint32_t gh = head_base + h;
                const uint32_t d =
                    pv_warp * (kPvNTiles * 8u) + ntile * 8u + (uint32_t)tt_mma_c_j(lane, l);
                if (gt < n_tokens && gh < n_head) {
                    heads[((uint64_t)gt * n_head + gh) * kTTHeadDim + d] =
                        o_acc[idx] * final_scale[m];
                }
            }
        }
    }
}

template <uint32_t TT_G>
__global__ static void __launch_bounds__(512, 1) attention_tokentile_hmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min) {
    static_assert(TT_G == 8u || TT_G == 16u, "supported token-tile head groups");
    constexpr uint32_t kTTG = TT_G, kTTM = kTTTileTokens * TT_G;
    constexpr uint32_t kKvElems = tt_TokentileLayout<kTTStageRows>::ring_plane_elems;
    constexpr uint32_t kProbStride = tt_TokentileLayout<kTTStageRows>::prob_stride;
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t head_group = blockIdx.y;
    const uint32_t tile_base = tile_idx * kTTTileTokens;
    const uint32_t head_base = head_group * kTTG;
    if (tile_base >= n_tokens) {
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t raw_union_count = tile_count + kTTRawWindow - 1u;
    const uint32_t comp_union_count = union_counts[tile_idx];
    const uint32_t n_score = raw_union_count + comp_union_count;
    const uint64_t union_tile_off = (uint64_t)tile_idx * rec_stride;
    const int2 * __restrict__ union_records_tile = union_records + union_tile_off;
    const float score_scale = rsqrtf((float)kTTHeadDim);

    extern __shared__ unsigned char smem[];
    unsigned char *p = tt_align16(smem);
    half *kv_h = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * tt_TokentileLayout<kTTStageRows>::ring_plane_bytes);
    half *probs = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * kTTM * kProbStride * sizeof(half));
    float4 *score_scratch = reinterpret_cast<float4 *>(p);
    // G16 scores occupy the inactive 32KiB KV plane only after its PV reads.
    if constexpr (TT_G == 8u) p = tt_align16(p + kTTM * kTTStageRows * sizeof(float4));
    float *max_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *sum_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *stage_rescale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *final_scale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    int2 *rec_ring = reinterpret_cast<int2 *>(p);
    p = tt_align16(p + kTTRecordRingPlanes * kTTStageRows * sizeof(int2));

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        max_s[m] = -INFINITY;
        sum_s[m] = 0.0f;
        stage_rescale[m] = 1.0f;
        final_scale[m] = 0.0f;
    }
    __syncthreads();

    union tt_TokentileRoleRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[64];
    };
    struct tt_TokentileBothRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[64];
    };
    typename std::conditional<TT_G == 16u, tt_TokentileBothRegs, tt_TokentileRoleRegs>::type role_regs;
    tt_load_score_q_frag<TT_G>(
        role_regs.score_q_frag, q, n_tokens, n_head, tile_base, head_base);
    if (TT_G == 16u || tt_warp_id() >= 8u) {
#pragma unroll
        for (uint32_t i = 0; i < 64u; ++i) {
            role_regs.o_acc[i] = 0.0f;
        }
    }

    uint32_t cur = 0u;
    uint32_t free = 1u;
    if (n_score != 0u) {
        const uint32_t nr0 = n_score < kTTStageRows ? n_score : kTTStageRows;
        tt_issue_kv_stage_cp_async<kTTStageRows, false>(
            kv_h + cur * kKvElems,
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile,
            tile_base,
            raw_kv,
            comp_kv,
            tid);
        tt_issue_record_stage_cp_async<kTTStageRows>(
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile);
        if (kTTStageRows < n_score) {
            const uint32_t nr1 =
                n_score - kTTStageRows < kTTStageRows ? n_score - kTTStageRows : kTTStageRows;
            tt_issue_record_stage_cp_async<kTTStageRows>(
                rec_ring + kTTStageRows,
                kTTStageRows,
                nr1,
                raw_union_count,
                union_records_tile);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    uint32_t prob_cur = 0u;
    for (uint32_t row0 = 0u; row0 < n_score; row0 += kTTStageRows) {
        const uint32_t nr = n_score - row0 < kTTStageRows ? n_score - row0 : kTTStageRows;
        half *kv_cur = kv_h + cur * kKvElems;
        half *kv_free = kv_h + free * kKvElems;

        if constexpr (TT_G == 8u) {
            tt_hmma_score_stage<kTTStageRows, TT_G>(
                score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
        }
        if (row0 != 0u) {
            const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
            const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
            tt_pv_mma_stage<kTTStageRows, TT_G>(
                role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
        }
        __syncthreads();

        if constexpr (TT_G == 16u) {
            score_scratch = reinterpret_cast<float4 *>(kv_free);
            tt_hmma_score_stage<kTTStageRows, TT_G>(
                score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
            __syncthreads();
            tt_softmax_stage<kTTStageRows, TT_G>(
                probs + prob_cur * kTTM * kProbStride, stage_rescale, max_s, sum_s,
                score_scratch, rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
                row0, nr, raw_union_count, tile_count, tile_base, raw_row_min);
            // All score readers finish before cp.async overwrites this plane.
            __syncthreads();
        }

        const uint32_t next_row0 = row0 + kTTStageRows;
        const bool has_next = next_row0 < n_score;
        if (has_next) {
            const uint32_t next_nr =
                n_score - next_row0 < kTTStageRows ? n_score - next_row0 : kTTStageRows;
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_free,
                rec_ring + (((row0 / kTTStageRows) + 1u) & 3u) * kTTStageRows,
                next_row0,
                next_nr,
                raw_union_count,
                union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            const uint32_t prefetch_row0 = next_row0 + kTTStageRows;
            if (prefetch_row0 < n_score) {
                const uint32_t prefetch_nr =
                    n_score - prefetch_row0 < kTTStageRows
                        ? n_score - prefetch_row0
                        : kTTStageRows;
                tt_issue_record_stage_cp_async<kTTStageRows>(
                    rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                    prefetch_row0,
                    prefetch_nr,
                    raw_union_count,
                    union_records_tile);
            }
            tt_cp_async_commit();
        }

        if constexpr (TT_G == 8u) {
        half *probs_cur = probs + prob_cur * kTTM * kProbStride;
        tt_softmax_stage<kTTStageRows, TT_G>(
            probs_cur,
            stage_rescale,
            max_s,
            sum_s,
            score_scratch,
            rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
            row0,
            nr,
            raw_union_count,
            tile_count,
            tile_base,
            raw_row_min);
        }
        tt_cp_async_wait_group<0>();
        __syncthreads();
        cur ^= 1u;
        free ^= 1u;
        prob_cur ^= 1u;
    }

    if (n_score != 0u) {
        const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
        const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
        tt_pv_mma_stage<kTTStageRows, TT_G>(
            role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
    }
    __syncthreads();

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        const uint32_t tok = m / kTTG;
        const uint32_t h = m - tok * kTTG;
        const uint32_t gt = tile_base + tok;
        const uint32_t gh = head_base + h;
        if (gt < n_tokens && gh < n_head) {
            const float sink = sinks[gh];
            const float old_m = max_s[m];
            const float new_m = fmaxf(old_m, sink);
            const float old_scale = old_m == -INFINITY ? 0.0f : expf(old_m - new_m);
            const float sink_scale = expf(sink - new_m);
            const float den = sum_s[m] * old_scale + sink_scale;
            final_scale[m] = den == 0.0f ? 0.0f : old_scale / den;
        } else {
            final_scale[m] = 0.0f;
        }
    }
    __syncthreads();

    tt_pv_mma_epilogue<TT_G>(
        role_regs.o_acc, final_scale, heads, n_tokens, n_head, tile_base, head_base);
}

constexpr size_t tt_align16_const(size_t x) {
    return (x + 15u) & ~size_t(15u);
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
struct tt_TokentileSmemBudget {
    static constexpr uint32_t M = kTTTileTokens * TT_G;
    static constexpr uint32_t prob_stride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static constexpr size_t q_bytes = 0;
    static constexpr size_t ring_bytes = 2ull * tt_TokentileLayout<TT_STAGE_ROWS>::ring_plane_bytes;
    static constexpr size_t p_bytes = 2ull * M * prob_stride * sizeof(half);
    static constexpr size_t partial_records = (size_t)M * TT_STAGE_ROWS;
    static constexpr size_t partial_bytes = TT_G == 16u ? 0u : partial_records * sizeof(float4);
    static constexpr size_t stats_bytes = 4ull * M * sizeof(float);
    static constexpr size_t record_bytes =
        (size_t)kTTRecordRingPlanes * TT_STAGE_ROWS * sizeof(int2);
    static constexpr size_t total =
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(q_bytes) + ring_bytes) + p_bytes) +
        partial_bytes) + stats_bytes) + record_bytes);
};

static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::p_bytes == 5120ull,
              "M32/R32 P double-buffer must use R+8 stride");
static_assert(sizeof(float4) == 16u, "score partial records must stay 16 bytes");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::partial_bytes == 16ull * 1024ull,
              "M32/R32 score partial records must be M*R float4");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::record_bytes == 1024ull,
              "M32/R32 record ring must be four R-row int2 planes");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::total == 88576ull,
              "M32/R32 total dynamic shared memory changed unexpectedly");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::total <= kTTSmemHardCap,
              "token-tile dynamic shared memory must stay under the 90 KiB pass gate");
static_assert(64u * kTTStageRows * sizeof(float4) == tt_TokentileLayout<kTTStageRows>::ring_plane_bytes,
              "G16 score records must exactly fit an inactive KV plane");
static_assert(tt_TokentileSmemBudget<kTTStageRows, 16u>::total == 77824ull,
              "G16 aliased score shared-memory budget changed");

static std::atomic<uint64_t> ds4_token_tile_queued{0};
static std::atomic<uint64_t> ds4_token_union_queued{0};
static std::atomic<uint64_t> ds4_token_tile_g16_queued{0};
static std::atomic<uint64_t> ds4_token_f16_table_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_token_f16_table_queued(void){return ds4_token_f16_table_queued.load(std::memory_order_acquire);}
extern "C" uint64_t ggml_backend_cuda_dsv41_token_tile_g16_queued(void){return ds4_token_tile_g16_queued.load(std::memory_order_acquire);}
extern "C" uint64_t ggml_backend_cuda_dsv41_token_union_queued(void){return ds4_token_union_queued.load(std::memory_order_acquire);}
extern "C" uint64_t ggml_backend_cuda_dsv41_token_tile_queued(void){return ds4_token_tile_queued.load(std::memory_order_acquire);}
#include "dsv41-tilelang.cuh"
void ggml_cuda_op_dsv41_ds4_token_tile(ggml_backend_cuda_context &ctx,ggml_tensor *dst){
    GGML_ASSERT(ggml_dsv41_token_tile_valid(dst));
    const int cc=ggml_cuda_info().devices[ctx.device].cc;
    GGML_ASSERT(!GGML_CUDA_CC_IS_AMD(cc)&&cc>=GGML_CUDA_CC_AMPERE);
    const auto*q=dst->src[0];const auto*raw=dst->src[1];const auto*comp=dst->src[2];
    const uint32_t count=uint32_t(q->ne[2]),start=uint32_t(dst->op_params[3]);
    const uint32_t tiles=(count+3)/4,raw_rows=count+127,raw_min=127-std::min(start,127u);
    const uint32_t first_raw=start-std::min(start,127u),comp_rows=uint32_t(comp->ne[1]);
    const uint64_t raw_elems=uint64_t(raw_rows)*512,comp_elems=uint64_t(comp_rows)*512;
    const bool direct_f16=dst->op_params[7]==1;
    GGML_ASSERT((raw_elems+(direct_f16?0:comp_elems))*sizeof(half)<=256ull*1024*1024);
    ggml_cuda_pool_alloc<half> raw_half(ctx.pool(),raw_elems),comp_half(ctx.pool());
    if(!direct_f16)comp_half.alloc(comp_elems);
    const half *comp_ptr=direct_f16?(const half*)comp->data:comp_half.get();
    const auto stream=ctx.stream();
    attention_tokentile_raw_mirror_kernel<<<raw_rows,256,0,stream>>>(
        raw_half.get(),(const float*)raw->data,nullptr,start,count,uint32_t(raw->ne[1]),0,first_raw,raw_min,512);
    if(!direct_f16)attention_tokentile_comp_mirror_kernel<<<comp_rows,256,0,stream>>>(comp_half.get(),(const float*)comp->data,comp_rows,512);
    CUDA_CHECK(cudaGetLastError());
    const bool merge=dst->op_params[5]&&dst->op_params[2];
    if(merge){
        const uint32_t actual_rows=comp_rows-1u;
        const size_t bitmap_bytes=size_t((actual_rows+7u)>>3u)*sizeof(uint32_t);
        GGML_ASSERT(actual_rows&&actual_rows<=131072&&bitmap_bytes<=65536);
        CUDA_CHECK(cudaFuncSetAttribute(attention_tokentile_union_build_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
        attention_tokentile_union_build_kernel<<<tiles,512,bitmap_bytes,stream>>>(
            (int2*)dst->src[3]->data,(uint32_t*)dst->src[4]->data,(const int32_t*)dst->src[6]->data,
            nullptr,start,count,uint32_t(dst->op_params[2]),uint32_t(dst->op_params[6]),actual_rows,uint32_t(dst->src[3]->ne[1]));
        CUDA_CHECK(cudaGetLastError());ds4_token_union_queued.fetch_add(1,std::memory_order_release);
    }
    const char *tilelang_env=std::getenv("LLAMA_DSV41_TILELANG_ATTENTION");
    if(tilelang_env&&tilelang_env[0]=='1'&&!tilelang_env[1]&&comp_rows<=131072){
        dsv41_tilelang_attention(ctx,dst,raw_half.get(),comp_ptr+(merge?512:0),raw_rows,raw_min,comp_rows-(merge?1u:0u));
        ds4_token_tile_queued.fetch_add(1,std::memory_order_release);
        if(direct_f16)ds4_token_f16_table_queued.fetch_add(1,std::memory_order_release);
        return;
    }
    const char *g16_env=std::getenv("LLAMA_DSV41_ATTN_G16");
    const bool g16=g16_env&&g16_env[0]=='1'&&g16_env[1]=='\0';
    if(g16){
    CUDA_SET_SHARED_MEMORY_LIMIT(attention_tokentile_hmma_kernel<16u>, (tt_TokentileSmemBudget<kTTStageRows,16u>::total));
    attention_tokentile_hmma_kernel<16u><<<dim3(tiles,4),512,tt_TokentileSmemBudget<kTTStageRows,16u>::total,stream>>>(
        (float*)dst->data,(const float*)dst->src[5]->data,(const float*)q->data,raw_half.get(),comp_ptr+(merge?512:0),
        (const int2*)dst->src[3]->data,(const uint32_t*)dst->src[4]->data,uint32_t(dst->src[3]->ne[1]),count,64,raw_min);
    ds4_token_tile_g16_queued.fetch_add(1,std::memory_order_release);
    }else{
    CUDA_CHECK(cudaFuncSetAttribute(attention_tokentile_hmma_kernel<8u>,cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(tt_TokentileSmemBudget<kTTStageRows,kTTG>::total)));
    attention_tokentile_hmma_kernel<8u><<<dim3(tiles,8),512,tt_TokentileSmemBudget<kTTStageRows,kTTG>::total,stream>>>(
        (float*)dst->data,(const float*)dst->src[5]->data,(const float*)q->data,raw_half.get(),comp_ptr+(merge?512:0),
        (const int2*)dst->src[3]->data,(const uint32_t*)dst->src[4]->data,uint32_t(dst->src[3]->ne[1]),count,64,raw_min);
    }
    CUDA_CHECK(cudaGetLastError());
    ds4_token_tile_queued.fetch_add(1,std::memory_order_release);
    if(direct_f16)ds4_token_f16_table_queued.fetch_add(1,std::memory_order_release);
}
