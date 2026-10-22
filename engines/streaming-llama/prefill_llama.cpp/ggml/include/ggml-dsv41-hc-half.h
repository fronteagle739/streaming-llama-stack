#pragma once
// Explicit HC projection selection; existing cuBLAS F16/F16/F32 only.
#include "ggml-backend.h"
#include <cstring>
inline bool ggml_dsv41_hc_half_requested(const ggml_tensor *t){
    return t&&t->op==GGML_OP_MUL_MAT&&t->op_params[2]==0x48434648&&t->op_params[3]==1;
}
inline bool ggml_dsv41_hc_half_valid(const ggml_tensor *t){
    if(!t||t->op!=GGML_OP_MUL_MAT||t->type!=GGML_TYPE_F32||t->op_params[0]!=GGML_PREC_F32)return false;
    const auto *w=t->src[0],*x=t->src[1];
    if(!w||!x||w->type!=GGML_TYPE_F16||x->type!=GGML_TYPE_F16||w->ne[0]!=20480||w->ne[1]!=24||
        w->ne[2]!=1||w->ne[3]!=1||x->ne[0]!=20480||x->ne[1]<=8||x->ne[1]>4096||x->ne[2]!=1||x->ne[3]!=1||
        !ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(t))return false;
    if(std::strncmp(w->name,"blk.",4))return false;
    const char *suffixes[]={".hc_attn_fn.weight",".hc_ffn_fn.weight"};const size_t length=std::strlen(w->name);
    for(const char *suffix:suffixes){const size_t n=std::strlen(suffix);if(length>=n&&!std::strcmp(w->name+length-n,suffix))return true;}
    return false;
}
inline bool ggml_dsv41_hc_half_tag(ggml_tensor *t){
    if(!ggml_dsv41_hc_half_valid(t))return false;t->op_params[2]=0x48434648;t->op_params[3]=1;return true;
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_hc_half_queued(void);
