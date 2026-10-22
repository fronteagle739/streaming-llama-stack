#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

// Experimental, explicitly selected fused three-product TF32 GEMM. Original
// F32 inputs are split inside the upstream CUTLASS mainloop. No operand bank,
// split workspace, host staging, or three separate GEMMs. Not bit-exact F32.
int ggml_cuda_dsv41_split_f32(int tile, int m, int n, int k,
    const float * a, int lda, int64_t batch_a,
    const float * b, int ldb, int64_t batch_b,
    float * c, int ldc, int64_t batch_c, int batches, cudaStream_t stream);

// PV consumes the existing column-major value view without a transpose copy.
int ggml_cuda_dsv41_split_f32_pv(int m, int n, int k,
    const float * a, int lda, int64_t batch_a,
    const float * b, int ldb, int64_t batch_b,
    float * c, int ldc, int64_t batch_c, int batches, cudaStream_t stream);
void ggml_cuda_dsv41_split_f32_record_queued();
