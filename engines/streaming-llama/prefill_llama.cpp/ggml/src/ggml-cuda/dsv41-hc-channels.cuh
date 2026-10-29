#pragma once
#include <cuda_runtime.h>
#include "../ggml-dsv41-quant.h"

// One lane owns the same embedding coordinate in all four HC outputs. Each
// residual load is shared by four independent, source-ordered accumulations.
// Explicit RN instructions preserve the unfused graph even with --fmad=true.
// OPUS22: R/O = float or uint16_t (BF16 bits). A BF16 residual widens exactly;
// a BF16 output stores the high half of the already BF16-rounded F32 value.
__device__ __forceinline__ float dsv41_hc_load(float v) { return v; }
__device__ __forceinline__ float dsv41_hc_load(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ void dsv41_hc_store(float *p, float v) { *p = v; }
__device__ __forceinline__ void dsv41_hc_store(uint16_t *p, float v) { *p = uint16_t(__float_as_uint(v) >> 16); }
template<bool BF16, typename R = float, typename O = float>
__global__ void dsv41_hc_channels_kernel(O *out, const float *block,
        const R *residual, const float *mix, uint32_t width, uint32_t tokens) {
    const uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= uint64_t(width) * tokens) return;
    const uint32_t d = gid % width, t = gid / width;
    const float *m = mix + uint64_t(t) * 20;
    const float b = block[gid];
    float acc[4];
#pragma unroll
    for (int dst = 0; dst < 4; ++dst) acc[dst] = __fmul_rn(b, m[dst]);
#pragma unroll
    for (int src = 0; src < 4; ++src) {
        const float v = dsv41_hc_load(residual[(uint64_t(t) * 4 + src) * width + d]);
#pragma unroll
        for (int dst = 0; dst < 4; ++dst)
            acc[dst] = __fadd_rn(acc[dst], __fmul_rn(m[4 + src * 4 + dst], v));
    }
#pragma unroll
    for (int dst = 0; dst < 4; ++dst)
        dsv41_hc_store(out + (uint64_t(t) * 4 + dst) * width + d, BF16 ? ggml_dsv41_bf16(acc[dst]) : acc[dst]);
}
