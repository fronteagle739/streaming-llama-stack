#include "common.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);
void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_sum_rows_div(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor         * src,
        const ggml_tensor         * eps,
        ggml_tensor               * dst,
        bool                        columnwise);

void ggml_cuda_op_fused_lid_score_reduce(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor         * kq,
        const ggml_tensor         * weights,
        ggml_tensor               * dst,
        float                       scale);
