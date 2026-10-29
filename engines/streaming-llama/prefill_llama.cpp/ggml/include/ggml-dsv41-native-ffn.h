// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-backend.h"
#include <cstring>
#include <cmath>
inline bool ggml_dsv41_native_ffn_requested(const ggml_tensor *t){
    return t&&t->op==GGML_OP_CUSTOM&&t->op_params[0]==0x46494e31&&t->op_params[1]==1;
}
inline bool ggml_dsv41_native_ffn_valid(const ggml_tensor *t){
    if(!ggml_dsv41_native_ffn_requested(t)||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||
       t->ne[0]!=5120||t->ne[1]<1||t->ne[1]>16384||t->ne[2]!=1||t->ne[3]!=1)return false;
    const auto *x=t->src[0],*weights=t->src[1],*ids=t->src[2];
    if(!x||!weights||!ids||x->type!=GGML_TYPE_F32||x->ne[0]!=5120||x->ne[1]!=t->ne[1]||
       x->ne[2]!=1||x->ne[3]!=1||weights->type!=GGML_TYPE_F32||ids->type!=GGML_TYPE_I32||
       weights->ne[0]!=6||weights->ne[1]!=t->ne[1]||weights->ne[2]!=1||weights->ne[3]!=1||
       std::memcmp(weights->ne,ids->ne,sizeof(ids->ne))||!ggml_is_contiguous(x)||!ggml_is_contiguous(weights)||!ggml_is_contiguous(ids))return false;
    for(int i=3;i<6;++i){const auto *w=t->src[i];
        if(!w||w->type!=GGML_TYPE_MXFP4||!ggml_is_contiguous(w)||w->ne[2]!=384||w->ne[3]!=1||
           w->ne[0]!=(i==5?2304:5120)||w->ne[1]!=(i==5?5120:2304))return false;
    }
    float limit;std::memcpy(&limit,t->op_params+2,sizeof(limit));
    return std::isfinite(limit)&&limit>0;
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_native_ffn_queued(void);
