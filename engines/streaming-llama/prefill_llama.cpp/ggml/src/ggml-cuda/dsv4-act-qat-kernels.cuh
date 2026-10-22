#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "../ggml-dsv41-quant.h"

// DeepSeek-V4 QAT activation-simulation round trip.
//
// Must stay bit-comparable with the CPU reference in ggml-cpu/ops.cpp, which is itself a port of
// ds4.c (dsv4_fp8_kv_quantize_row_inplace_cpu / dsv4_fp4_act_quantize_row_inplace_cpu). This is a
// SELECTION-CRITICAL transform: it decides which compressed rows the model attends to, so a kernel
// that is merely "close" reorders the top-k and silently loses mid-prompt content -- the exact
// failure this op exists to fix. Every constant below is the format's, not a tunable.

static __device__ __forceinline__ float dsv4_e2m1_value(int i) {
    switch (i & 7) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        default: return 6.0f;
    }
}

// Nearest E2M1 magnitude, ties to EVEN index. Linear search over 8 values mirrors ds4.c exactly;
// a branchy "closed form" would be faster and is not worth a tie-break divergence here.
static __device__ __forceinline__ float dsv4_e2m1_round(float v) {
    const float sign = v < 0.0f ? -1.0f : 1.0f;
    const float ax   = fabsf(v);

    int   best      = 0;
    float best_diff = fabsf(ax - dsv4_e2m1_value(0));
#pragma unroll
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - dsv4_e2m1_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best      = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1_value(best);
}

// Nearest representable E4M3FN (no infinities, max finite 448), ties to even mantissa.
static __device__ __forceinline__ float dsv4_e4m3_round(float v) {
    const float sign = v < 0.0f ? -1.0f : 1.0f;
    float       ax   = fabsf(v);

    if (!(ax > 0.0f)) {
        return 0.0f * sign;
    }
    if (ax >= 448.0f) {
        return sign * 448.0f;
    }

    const float min_normal = 0.015625f;   // 2^-6
    float step;
    if (ax < min_normal) {
        step = 0.001953125f;              // 2^-9, subnormal step
    } else {
        int e;
        frexpf(ax, &e);                   // ax = m * 2^e, m in [0.5,1)
        step = ldexpf(1.0f, (e - 1) - 3); // 3 mantissa bits
    }

    const float q = ax / step;
    float       r = floorf(q);
    const float d = q - r;
    if (d > 0.5f || (d == 0.5f && fmodf(r, 2.0f) != 0.0f)) {
        r += 1.0f;
    }
    float out = r * step;
    if (out > 448.0f) {
        out = 448.0f;
    }
    return sign * out;
}

// One block per (row, quant-block). BLK threads cooperate on the amax reduction so the per-block
// scale is computed once and shared -- the scale is per BLOCK, not per element, and computing it
// independently per thread would be both slower and a correctness hazard if any thread disagreed.
template <int BLK, bool IS_FP8>
static __global__ void dsv4_act_qat_kernel(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t ne0,
        const int64_t n_q,
        const int64_t nrow) {

    // GRID-Y OVERFLOW FIX 2026-08-28. This was `row = blockIdx.y` with the launcher putting the
    // full row count in gridDim.y, which CUDA caps at 65535. The DSV4 indexer has 64 heads, so
    // nrow = 64 * n_tokens and a 1024-token ubatch is 65536 rows -- over the cap by ONE. Every
    // prefill with ubatch >= 1024 therefore died in cudaLaunchKernel with "invalid argument",
    // while short prompts (267 tok = 17088 rows) passed. Rows now spill into gridDim.z.
    const int64_t row  = (int64_t) blockIdx.y + (int64_t) blockIdx.z * (int64_t) gridDim.y;
    if (row >= nrow) {
        return;
    }
    const int64_t off  = (int64_t) blockIdx.x * BLK;
    const int     tid  = threadIdx.x;

    const float * srow = src + row*ne0;
    float       * drow = dst + row*ne0;

    // Tail beyond the quantised span (the RoPE part for FP8) is copied through untouched.
    if (off >= n_q) {
        for (int64_t i = off + tid; i < ne0; i += BLK) {
            drow[i] = srow[i];
        }
        return;
    }

    const float x = srow[off + tid];

    __shared__ float s_amax[BLK];
    s_amax[tid] = fabsf(x);
    __syncthreads();
#pragma unroll
    for (int s = BLK/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + s]);
        }
        __syncthreads();
    }

    const float amax_floor = IS_FP8 ? 1.0e-4f : 7.052966104933725e-38f;
    const float sat        = IS_FP8 ? 448.0f  : 6.0f;

    float amax = s_amax[0];
    if (amax < amax_floor) {
        amax = amax_floor;
    }
    const float scale = ldexpf(1.0f, (int) ceilf(log2f(amax / sat)));

    float v = x / scale;
    v = fminf(fmaxf(v, -sat), sat);

    drow[off + tid] = (IS_FP8 ? dsv4_e4m3_round(v) : dsv4_e2m1_round(v)) * scale;
}

// V4.1 has no indexer Hadamard. Its FP4/E4M3 compressed-KV format uses
// sixteen-value blocks, quantizes the RoPE tail, and preserves every BF16
// boundary. Keep it a separate kernel so the established V4 path is unchanged.
static __global__ void dsv41_bf16_vector4_kernel(const float4 *src, float4 *dst, int64_t vectors) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    const float4 v = src[i]; // Read every member before writing an aliased vector.
    dst[i] = make_float4(ggml_dsv41_bf16(v.x), ggml_dsv41_bf16(v.y),
                         ggml_dsv41_bf16(v.z), ggml_dsv41_bf16(v.w));
}

template <int BLK, int FORMAT>
static __global__ void dsv41_act_qat_kernel(
        // FORMAT0 is an independent elementwise round-trip and may reuse its
        // input after gallocr proves last readership. Do not promise no-alias.
        const float * src, float * dst,
        int64_t width, int64_t rows) {
    const int64_t row = int64_t(blockIdx.y) + int64_t(blockIdx.z) * gridDim.y;
    if (row >= rows) return;
    const int64_t column = int64_t(blockIdx.x) * BLK + threadIdx.x;
    const bool valid = column < width;
    const float value = valid ? ggml_dsv41_bf16(src[row * width + column]) : 0.f;
    if (FORMAT == 0) {
        if (valid) dst[row * width + column] = value;
        return;
    }
    __shared__ float maxima[BLK];
    maxima[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (int step = BLK / 2; step; step >>= 1) {
        if (threadIdx.x < step) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
        __syncthreads();
    }
    const float scale = ggml_dsv41_scale(maxima[0], FORMAT);
    if (valid) dst[row * width + column] = ggml_dsv41_quantized(value, scale, FORMAT);
}
