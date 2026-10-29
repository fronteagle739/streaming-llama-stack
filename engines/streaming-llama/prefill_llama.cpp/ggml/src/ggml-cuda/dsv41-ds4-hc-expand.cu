#include "dispatch-counter.cuh"
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
// The F32 path retains the ds4 hc_expand body and --fmad=false contract.
// The BF16 path shares residual loads across four channels, using explicit RN
// multiply/add instructions and the identical BF16 boundary. No workspace.
#include "dsv41-ds4-hc-expand.cuh"
#include "ggml-dsv41-hc-expand.h"
#include "dsv4-act-qat-kernels.cuh"
#include "dsv41-hc-channels.cuh"
#include <atomic>
#include "ggml-dsv41-hc-collapse.h"
#include "dsv41-hc-collapse.cuh"
void ggml_cuda_op_dsv41_hc_collapse(ggml_backend_cuda_context&ctx,ggml_tensor*dst) {
    GGML_ASSERT(ggml_dsv41_hc_collapse_valid(dst));
    const unsigned blocks=unsigned((ggml_nelements(dst)+255)/256);
    if(dst->src[0]->type==GGML_TYPE_BF16)dsv41_hc_collapse_kernel<uint16_t><<<blocks,256,0,ctx.stream()>>>(
        (const uint16_t*)dst->src[0]->data,(const float*)dst->src[1]->data,(float*)dst->data,dst->ne[0],dst->ne[1]);
    else dsv41_hc_collapse_kernel<float><<<blocks,256,0,ctx.stream()>>>(
        (const float*)dst->src[0]->data,(const float*)dst->src[1]->data,
        (float*)dst->data,dst->ne[0],dst->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
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


static ggml_cuda_dispatch_counter hc_expand_queued{0},hc_bf16_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_hc_bf16_queued(void){return hc_bf16_queued.load();}
extern "C" uint64_t ggml_backend_cuda_dsv41_hc_expand_queued(void){return hc_expand_queued.load();}
void ggml_cuda_op_dsv41_ds4_hc_expand(ggml_backend_cuda_context&ctx,ggml_tensor*dst){
 GGML_ASSERT(ggml_dsv41_hc_expand_valid(dst));
 const uint32_t width=uint32_t(dst->ne[0]),tokens=uint32_t(dst->ne[2]);
 const auto*mix=(const float*)dst->src[2]->data;
  if(ggml_dsv41_hc_bf16_requested(dst)){
   const unsigned blocks=unsigned((uint64_t(width)*tokens+255)/256);const auto*block=(const float*)dst->src[1]->data;
   const bool r16=dst->src[0]->type==GGML_TYPE_BF16,o16=dst->type==GGML_TYPE_BF16;
   if(r16&&o16)dsv41_hc_channels_kernel<true,uint16_t,uint16_t><<<blocks,256,0,ctx.stream()>>>((uint16_t*)dst->data,block,(const uint16_t*)dst->src[0]->data,mix,width,tokens);
   else if(r16)dsv41_hc_channels_kernel<true,uint16_t,float><<<blocks,256,0,ctx.stream()>>>((float*)dst->data,block,(const uint16_t*)dst->src[0]->data,mix,width,tokens);
   else if(o16)dsv41_hc_channels_kernel<true,float,uint16_t><<<blocks,256,0,ctx.stream()>>>((uint16_t*)dst->data,block,(const float*)dst->src[0]->data,mix,width,tokens);
   else dsv41_hc_channels_kernel<true><<<blocks,256,0,ctx.stream()>>>((float*)dst->data,block,(const float*)dst->src[0]->data,mix,width,tokens);
   hc_bf16_queued.fetch_add(1);
  }else{
 GGML_ASSERT(dst->type==GGML_TYPE_F32&&dst->src[0]->type==GGML_TYPE_F32);
 hc_expand_kernel<<<unsigned((uint64_t(width)*4*tokens+255)/256),256,0,ctx.stream()>>>(
  (float*)dst->data,(const float*)dst->src[1]->data,nullptr,nullptr,(const float*)dst->src[0]->data,
  mix,mix+4,width,4,tokens,20,20,0,0);
  }
 CUDA_CHECK(cudaGetLastError());hc_expand_queued.fetch_add(1);
}
