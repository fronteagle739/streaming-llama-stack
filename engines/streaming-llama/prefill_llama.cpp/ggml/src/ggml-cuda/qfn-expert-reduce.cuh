// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-qfn-expert-reduce.h"

// Indexing follows ds4_qwen4_cuda.cuh::moe_reduce. Explicit rounding keeps
// QFN's materialized F32 products and its slot-ascending addition order.
static __global__ void qfn_expert_reduce_kernel(const char * x, const char * w, float * out,
        int64_t E, int64_t K, int64_t T, size_t x0, size_t x1, size_t x2, size_t w1, size_t w2) {
    const int64_t count = E*T;
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x)*blockDim.x) {
        const int64_t t = i/E, d = i%E;
        const char * row = x + t*x2 + d*x0;
        const char * weights = w + t*w2;
        float sum = __fmul_rn(*reinterpret_cast<const float *>(row),
                              *reinterpret_cast<const float *>(weights));
        for (int64_t k = 1; k < K; ++k)
            sum = __fadd_rn(sum, __fmul_rn(*reinterpret_cast<const float *>(row+k*x1),
                                          *reinterpret_cast<const float *>(weights+k*w1)));
        out[i] = sum;
    }
}
static void ggml_cuda_op_qfn_expert_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_qfn_expert_reduce_valid(dst));
    const auto * x = dst->src[0]; const auto * w = dst->src[1];
    const int blocks = int(std::min<int64_t>((ggml_nelements(dst)+255)/256, 65535));
    qfn_expert_reduce_kernel<<<blocks,256,0,ctx.stream()>>>(
        static_cast<const char *>(x->data), static_cast<const char *>(w->data),
        static_cast<float *>(dst->data), x->ne[0], x->ne[1], x->ne[2],
        x->nb[0], x->nb[1], x->nb[2], w->nb[1], w->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
