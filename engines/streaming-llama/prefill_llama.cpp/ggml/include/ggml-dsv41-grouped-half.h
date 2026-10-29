#pragma once
// OPUS22 r10: grouped F16 dense projection (attn_output_a, 8 groups) as ONE node.
// src[0] = F32 RHS [K, rows, groups] (any strides with nb0 = 4), src[1+g] = the
// group's contiguous F16 weight [K, N]. dst = contiguous F32 [N, rows, groups].
// The CUDA op runs, per group, exactly the cuBLAS call the tagged F16 dense
// MUL_MAT would run, writing that group's slice of dst, so the result is
// bit-identical to the per-group products while the concat tree is removed.
#include "ggml.h"
inline bool ggml_dsv41_grouped_half_requested(const ggml_tensor *t){
    return t&&t->op==GGML_OP_CUSTOM&&t->op_params[0]==0x47525031&&t->op_params[1]==1;
}
inline bool ggml_dsv41_grouped_half_valid(const ggml_tensor *t){
    if(!ggml_dsv41_grouped_half_requested(t)||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t))return false;
    const int64_t groups=t->op_params[2];
    const ggml_tensor *x=t->src[0];
    if(groups<2||groups+1>GGML_MAX_SRC||!x||x->type!=GGML_TYPE_F32||x->nb[0]!=sizeof(float)||
       x->ne[2]!=groups||x->ne[3]!=1||t->ne[1]!=x->ne[1]||t->ne[2]!=groups||t->ne[3]!=1)return false;
    for(int64_t g=0;g<groups;++g){const ggml_tensor *w=t->src[1+g];
        if(!w||w->type!=GGML_TYPE_F16||!ggml_is_contiguous(w)||w->ne[0]!=x->ne[0]||w->ne[1]!=t->ne[0]||w->ne[2]!=1||w->ne[3]!=1)return false;}
    return true;
}
