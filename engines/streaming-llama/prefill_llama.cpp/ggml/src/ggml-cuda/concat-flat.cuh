#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Each vector remains within one input span; arbitrary shapes use scalar
// copies. Flattening packs short rows into full blocks without changing bits.
template<bool VECTOR>
__global__ void concat_f32_flat(const float *x, const float *y, float *out,
        int64_t xspan, int64_t yspan, int64_t count) {
    constexpr int lanes = VECTOR ? 4 : 1;
    const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * lanes;
    if (i >= count) return;
    const int64_t span = xspan + yspan, outer = i / span, inner = i % span;
    const float *src = inner < xspan ? x + outer * xspan + inner : y + outer * yspan + inner - xspan;
    if constexpr (VECTOR) *reinterpret_cast<float4 *>(out + i) = *reinterpret_cast<const float4 *>(src);
    else out[i] = *src;
}
