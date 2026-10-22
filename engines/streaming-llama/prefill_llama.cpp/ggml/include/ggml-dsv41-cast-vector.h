#pragma once
#include "ggml.h"
#include <stdint.h>

// Reject tails, unaligned rows, unsupported type pairs, overflow and overlap.
// Strided inputs use a conservative bounding span; destinations are compact.
static inline bool ggml_dsv41_cast_vector_geometry(enum ggml_type src,enum ggml_type dst,
        const void *x,void *y,int64_t n0,int64_t n1,int64_t n2,int64_t n3,
        int64_t s1,int64_t s2,int64_t s3) {
    const bool from_float=src==GGML_TYPE_F32&&(dst==GGML_TYPE_F16||dst==GGML_TYPE_BF16);
    const bool to_float=dst==GGML_TYPE_F32&&(src==GGML_TYPE_F16||src==GGML_TYPE_BF16);
    if(!(from_float||to_float)||!x||!y||n0<=0||n1<=0||n2<=0||n3<=0||n0%4||s1<0||s2<0||s3<0||s1%4||s2%4||s3%4)return false;
    const uint64_t a=from_float?4:2,b=to_float?4:2,limit=UINTPTR_MAX;
    const uintptr_t xp=(uintptr_t)x,yp=(uintptr_t)y;
    if(xp%(4*a)||yp%(4*b))return false;
    uint64_t count=(uint64_t)n0,span=(uint64_t)n0;
    const int64_t dims[3]={n1,n2,n3},strides[3]={s1,s2,s3};
    for(int i=0;i<3;++i){
        if((uint64_t)dims[i]>limit/count)return false;count*=dims[i];
        if(strides[i]&&(uint64_t)(dims[i]-1)>(limit-span)/(uint64_t)strides[i])return false;
        span+=(uint64_t)(dims[i]-1)*(uint64_t)strides[i];
    }
    if(span>limit/a||count>limit/b||xp>limit-span*a||yp>limit-count*b)return false;
    return xp+span*a<=yp||yp+count*b<=xp;
}
#ifdef __cplusplus
extern "C" {
#endif
GGML_API uint64_t ggml_backend_cuda_dsv41_cast_vector_queued(void);
GGML_API uint64_t ggml_backend_cuda_dsv41_cast_vector_elements(void);
GGML_API uint64_t ggml_backend_cuda_dsv41_cast_vector_nc_queued(void);
#ifdef __cplusplus
}
#endif
