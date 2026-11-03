// SPDX-License-Identifier: MIT
// DSV4.1 dsv41-index-fp4.cuh packing/MMA fragment order, adapted to QFN H4.
// Explicit lossy experiment: QFN has no FP4 index QAT. Quantize actual Q/K to
// E2M1/E8M0 per32; retain every pool, actual scales/strides and the QFN selector.
#pragma once
#include "mma.cuh"
#include "../ggml-dsv41-quant.h"

template<typename T, bool query>
static __global__ void qfn_index_fp4_pack(const char *input, uint32_t *codes,
        uint32_t *scales, int rows, uint64_t head_stride, uint64_t row_stride,
        uint64_t stream_stride) {
    const int row=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32, s=blockIdx.y;
    if(row>=rows)return;
    const int head=query?row%16:0, token=query?row/16:row;
    const T *src=(const T *)(input+uint64_t(s)*stream_stride+
        uint64_t(token)*row_stride+uint64_t(head)*head_stride);
    const uint64_t outrow=uint64_t(s)*rows+row;
    uint32_t scale_word=0;
    for(int group=0;group<4;++group){
        const float x=query&&head>=4?0.f:ggml_cuda_cast<float>(src[group*32+lane]);
        float maximum=fabsf(x);
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
        const float scale=ggml_dsv41_scale(maximum,2);
        const float value=ggml_dsv41_e2m1(__fdiv_rn(fabsf(x),scale));
        int code=0;for(int c=1;c<8;++c)if(value==ggml_dsv41_e2m1_value(c))code=c;
        code|=signbit(x)?8:0;
        uint32_t word=0;
        for(int d=0;d<8;++d)word|=uint32_t(__shfl_sync(0xffffffff,code,(lane/8)*8+d))<<(4*d);
        if(lane%8==0)codes[outrow*16+group*4+lane/8]=word;
        scale_word|=((__float_as_uint(scale)>>23)&255)<<(group*8);
    }
    if(!lane)scales[outrow]=scale_word;
}

static __global__ void qfn_index_fp4_scores(float *out,const uint32_t *query,const uint32_t *keys,
        const uint32_t *qs,const uint32_t *ks,const char *weights,int rows,int tokens,
        uint64_t wt,uint64_t ws,uint64_t ot,uint64_t os,float se,float sh){
    using namespace ggml_cuda_mma;
    const int lane=threadIdx.x,warp=threadIdx.y,token=blockIdx.y,s=blockIdx.z;
    const int first=(blockIdx.x*4+warp)*8;
    query+=uint64_t(s)*tokens*16*16;qs+=uint64_t(s)*tokens*16;
    keys+=uint64_t(s)*rows*16;ks+=uint64_t(s)*rows;
    __shared__ float dots[4][16][8];
    tile<16,8,float> c;
    for(int k=0;k<2;++k){
        tile<16,8,int>a;tile<8,8,int>b;
        for(int l=0;l<a.ne;++l)a.x[l]=int(query[(uint64_t(token)*16+(l%2)*8+lane/4)*16+k*8+(l/2)*4+lane%4]);
        for(int l=0;l<b.ne;++l){const int key=first+b.get_i(l);
            b.x[l]=key<rows?int(keys[uint64_t(key)*16+k*8+b.get_j(l)]):0;}
        const int qrow=lane/4+(lane%2)*8,key=first+lane/4;
        const uint32_t as=(qs[token*16+qrow]>>(k*16))&65535;
        const uint32_t bs=key<rows?(ks[key]>>(k*16))&65535:0;
        mma_block_scaled(c,a,b,as,bs);
    }
    for(int l=0;l<c.ne;++l)dots[warp][c.get_i(l)][c.get_j(l)]=c.x[l];
    __syncwarp();
    if(lane<8&&first+lane<rows){float result=0;
        const float *w=(const float *)(weights+uint64_t(s)*ws+uint64_t(token)*wt);
        for(int h=0;h<4;++h)result+=fmaxf(dots[warp][h][lane]*se,0.f)*w[h];
        float *dst=(float *)((char *)out+uint64_t(s)*os+uint64_t(token)*ot);
        dst[first+lane]=result*sh;
    }
}

static void qfn_index_fp4(ggml_backend_cuda_context &ctx,ggml_tensor *t){
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200);
    const auto *q=t->src[0],*k=t->src[1],*w=t->src[2];
    const int nt=int(q->ne[2]),np=int(k->ne[2]),ns=int(q->ne[3]);
    ggml_cuda_pool_alloc<uint32_t> qc(ctx.pool(),uint64_t(ns)*nt*16*16),
        kc(ctx.pool(),uint64_t(ns)*np*16),qs(ctx.pool(),uint64_t(ns)*nt*16),ks(ctx.pool(),uint64_t(ns)*np);
    const auto stream=ctx.stream();
    qfn_index_fp4_pack<float,true><<<dim3((nt*16+3)/4,ns),128,0,stream>>>(
        (const char *)q->data,qc.get(),qs.get(),nt*16,q->nb[1],q->nb[2],q->nb[3]);
    if(k->type==GGML_TYPE_F16)
        qfn_index_fp4_pack<half,false><<<dim3((np+3)/4,ns),128,0,stream>>>(
            (const char *)k->data,kc.get(),ks.get(),np,0,k->nb[2],k->nb[3]);
    else qfn_index_fp4_pack<float,false><<<dim3((np+3)/4,ns),128,0,stream>>>(
            (const char *)k->data,kc.get(),ks.get(),np,0,k->nb[2],k->nb[3]);
    float se,sh;memcpy(&se,t->op_params+2,4);memcpy(&sh,t->op_params+3,4);
    qfn_index_fp4_scores<<<dim3((np+31)/32,nt,ns),dim3(32,4),0,stream>>>(
        (float *)t->data,qc.get(),kc.get(),qs.get(),ks.get(),(const char *)w->data,
        np,nt,w->nb[1],w->nb[3],t->nb[1],t->nb[3],se,sh);
    CUDA_CHECK(cudaGetLastError());
}
