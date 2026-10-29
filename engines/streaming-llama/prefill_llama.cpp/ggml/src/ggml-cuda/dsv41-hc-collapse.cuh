#pragma once
#include "../ggml-dsv41-quant.h"
// The unfused expression rounds each product and each left-associated sum.
// Explicit RN instructions preserve that contract even in FMA-enabled builds.
__device__ __forceinline__ float dsv41_hc_collapse_load(float v) { return v; }
__device__ __forceinline__ float dsv41_hc_collapse_load(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
template<typename R>
static __global__ void dsv41_hc_collapse_kernel(const R *r, const float *p,
        float *out, int64_t width, int64_t rows) {
    const int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (at>=width*rows) return;
    const int64_t row=at/width, d=at%width;
    float sum=__fmul_rn(dsv41_hc_collapse_load(r[(row*4)*width+d]),p[row*4]);
    #pragma unroll
    for (int c=1;c<4;++c) sum=__fadd_rn(sum,__fmul_rn(dsv41_hc_collapse_load(r[(row*4+c)*width+d]),p[row*4+c]));
    out[at]=ggml_dsv41_bf16(sum);
}
