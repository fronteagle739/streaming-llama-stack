#pragma once
#include <cuda_runtime.h>
#include <cstdint>
// ABI 2: GGML ordinary MXFP4 blocks, packed expert routes, original F32 input/output.
// Weights are only rearranged; activations use E4M3 plus E8M0 per 32 values.
using slstack_flashinfer_mxfp4_fn = int (*)(const void *, const float *,
    const int32_t *, const int32_t *, const int32_t *, float *,
    int, int, int, int64_t, int64_t, int64_t, uint64_t, cudaStream_t);
