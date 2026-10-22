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
// Verbatim ds4 hc_expand body. Compile this TU with --fmad=false to retain
// the original graph's separate multiply/add rounding. BF16 boundaries remain
// in the existing graph operator. No new GPU body or scratch allocation.
#include "dsv41-ds4-hc-expand.cuh"
#include "ggml-dsv41-hc-expand.h"
#include "dsv4-act-qat-kernels.cuh"
#include <atomic>
__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add,
        int has_add2) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) {
        float add_v = block_add[(uint64_t)t * n_embd + d];
        if (has_add2) add_v += block_add2[(uint64_t)t * n_embd + d];
        block_v += add_v;
    }
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}


__global__ static void hc_expand_bf16_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add,
        int has_add2) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) {
        float add_v = block_add[(uint64_t)t * n_embd + d];
        if (has_add2) add_v += block_add2[(uint64_t)t * n_embd + d];
        block_v += add_v;
    }
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = ggml_dsv41_bf16(acc);
}


static std::atomic<uint64_t> hc_expand_queued{0},hc_bf16_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_hc_bf16_queued(void){return hc_bf16_queued.load();}
extern "C" uint64_t ggml_backend_cuda_dsv41_hc_expand_queued(void){return hc_expand_queued.load();}
void ggml_cuda_op_dsv41_ds4_hc_expand(ggml_backend_cuda_context&ctx,ggml_tensor*dst){
 GGML_ASSERT(ggml_dsv41_hc_expand_valid(dst));
 const uint32_t width=uint32_t(dst->ne[0]),tokens=uint32_t(dst->ne[2]);
 const auto*mix=(const float*)dst->src[2]->data;
  if(ggml_dsv41_hc_bf16_requested(dst)){
   hc_expand_bf16_kernel<<<unsigned((uint64_t(width)*4*tokens+255)/256),256,0,ctx.stream()>>>(
    (float*)dst->data,(const float*)dst->src[1]->data,nullptr,nullptr,(const float*)dst->src[0]->data,
    mix,mix+4,width,4,tokens,20,20,0,0);
   hc_bf16_queued.fetch_add(1);
  }else{
 hc_expand_kernel<<<unsigned((uint64_t(width)*4*tokens+255)/256),256,0,ctx.stream()>>>(
  (float*)dst->data,(const float*)dst->src[1]->data,nullptr,nullptr,(const float*)dst->src[0]->data,
  mix,mix+4,width,4,tokens,20,20,0,0);
  }
 CUDA_CHECK(cudaGetLastError());hc_expand_queued.fetch_add(1);
}
