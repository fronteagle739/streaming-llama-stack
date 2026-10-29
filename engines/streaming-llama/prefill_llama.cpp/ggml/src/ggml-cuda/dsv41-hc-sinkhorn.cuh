// SPDX-License-Identifier: MIT
// Copyright (c) 2023 DeepSeek
// Derived from inference/kernel.py:hc_split_sinkhorn_kernel at
// 1bae0e1088993b9b82de08f59de5be5a34e42170. See LICENSE.DeepSeek.
// TileLang0.1.15's generated SM120 CUDA is specialized to hc=4,20 iterations,
// eps=1e-6. Its XOR reductions are expanded here (row:2,1; column:8,4),
// and the three outputs share one ggml allocation. No TileLang runtime needed.
#pragma once
#include "common.cuh"
#include "ggml-dsv41-hc-sinkhorn.h"

static __device__ __forceinline__ float dsv41_hc_sum_row(float x){
    x+=__shfl_xor_sync(0xffffffffu,x,2);return x+__shfl_xor_sync(0xffffffffu,x,1);
}
static __device__ __forceinline__ float dsv41_hc_sum_column(float x){
    x+=__shfl_xor_sync(0xffffffffu,x,8);return x+__shfl_xor_sync(0xffffffffu,x,4);
}
template<bool native_math> static __device__ __forceinline__ float dsv41_hc_affine(float x,float s,float b){
    if constexpr(native_math)return __fadd_rn(__fmul_rn(x,s),b);
    return x*s+b;
}
template<bool native_math> static __device__ __forceinline__ float dsv41_hc_exp(float x){
    if constexpr(native_math)return __expf(x);
    return expf(x);
}
template<bool native_math> static __device__ __forceinline__ float dsv41_hc_div(float x,float y){
    if constexpr(native_math)return __fdividef(x,y);
    return x/y;
}
template<bool native_math> static __global__ __launch_bounds__(64,1) void dsv41_hc_sinkhorn(
        float *out,const float *mixes,const float *scale,const float *base,int rows){
    __shared__ float shared[24];
    const int lane=threadIdx.x,row=blockIdx.x;
    if(lane<24)shared[lane]=mixes[int64_t(row)*24+lane];
    __syncthreads();
    if(lane<4){
        out[int64_t(row)*4+lane]=dsv41_hc_div<native_math>(1.f,1.f+dsv41_hc_exp<native_math>(-dsv41_hc_affine<native_math>(shared[lane],scale[0],base[lane])))+1e-6f;
        out[int64_t(rows)*4+int64_t(row)*4+lane]=2.f*dsv41_hc_div<native_math>(1.f,1.f+dsv41_hc_exp<native_math>(-dsv41_hc_affine<native_math>(shared[lane+4],scale[1],base[lane+4])));
    }
    const int c=(lane&15)+8;
    float value=dsv41_hc_affine<native_math>(shared[c],scale[2],base[c]);
    float maximum=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,2));
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,1));
    value=dsv41_hc_exp<native_math>(value-maximum);
    if constexpr(native_math)value=__fmul_rn(value,dsv41_hc_div<true>(1.f,dsv41_hc_sum_row(value)))+1e-6f;
    else value=value/dsv41_hc_sum_row(value)+1e-6f;
    value=dsv41_hc_div<native_math>(value,dsv41_hc_sum_column(value)+1e-6f);
    #pragma unroll 1
    for(int i=0;i<19;++i){
        value=dsv41_hc_div<native_math>(value,dsv41_hc_sum_row(value)+1e-6f);
        value=dsv41_hc_div<native_math>(value,dsv41_hc_sum_column(value)+1e-6f);
    }
    if(lane<16)out[int64_t(rows)*8+int64_t(row)*16+lane]=value;
}
static void ggml_cuda_op_dsv41_hc_sinkhorn(ggml_backend_cuda_context &ctx,ggml_tensor *dst){
    GGML_ASSERT(ggml_dsv41_hc_sinkhorn_valid(dst)&&ggml_cuda_info().devices[ctx.device].cc==1200);
    const int rows=int(dst->ne[1]);
    if(dst->op_params[6]==2)
        dsv41_hc_sinkhorn<true><<<rows,64,0,ctx.stream()>>>((float*)dst->data,(const float*)dst->src[0]->data,
            (const float*)dst->src[1]->data,(const float*)dst->src[2]->data,rows);
    else dsv41_hc_sinkhorn<false><<<rows,64,0,ctx.stream()>>>((float*)dst->data,(const float*)dst->src[0]->data,
            (const float*)dst->src[1]->data,(const float*)dst->src[2]->data,rows);
    CUDA_CHECK(cudaGetLastError());
}
