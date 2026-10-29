#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "../prefill_llama.cpp/ggml/src/ggml-dsv41-flashinfer-workspace.h"
// ABI 4: up to 16384 tokens (98304 routes), canonical MXFP4, F32 input/output.
// Weights are only rearranged; activations use E4M3 plus E8M0 per 32 values.
// Workspace is caller-owned, 256-byte aligned, live until stream completion.
using slstack_flashinfer_mxfp4_fn = int (*)(const void *, const float *,
    const int32_t *, const int32_t *, const int32_t *, float *,
    int, int, int, int64_t, int64_t, int64_t, void *, uint64_t, cudaStream_t);
