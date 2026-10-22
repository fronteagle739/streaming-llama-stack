#pragma once
#include "ggml-backend.h"
#include "ggml-dsv41-pv-view-f32.h"
#include <cstdint>

// Only explicitly tagged native attention products. No global F32 override.
inline bool ggml_dsv41_split_f32_requested(const ggml_tensor * t) {
    return t && (t->op==GGML_OP_MUL_MAT||t->op==GGML_OP_OUT_PROD) && t->op_params[4]==0x53335446 && t->op_params[5]==1;
}
inline bool ggml_dsv41_split_f32_shape(const ggml_tensor * t) {
    if(!t || t->op!=GGML_OP_MUL_MAT || t->type!=GGML_TYPE_F32 ||
       t->op_params[0]!=GGML_PREC_F32 || t->view_src || t->src[2])return false;
    const auto * a=t->src[0], * b=t->src[1];
    if(!a||!b||a->type!=GGML_TYPE_F32||b->type!=GGML_TYPE_F32||
       !ggml_is_contiguous(a)||!ggml_is_contiguous(b)||!ggml_is_contiguous(t))return false;
    const auto k=a->ne[0],m=a->ne[1],n=b->ne[1],batch=b->ne[2];
    if(k!=b->ne[0]||n!=64||batch<1||batch>4096||a->ne[2]!=batch||
       a->ne[3]!=1||b->ne[3]!=1||t->ne[3]!=1||t->ne[0]!=m||t->ne[1]!=n||t->ne[2]!=batch)return false;
    // QK or PV with the bounded raw/selected-key table. The one-token warmup
    // and any unaligned key tail explicitly retain their original GEMM.
    return m%4==0&&k%4==0&&((k==512&&m>=4&&m<=640)||(m==512&&k>=4&&k<=640));
}
inline bool ggml_dsv41_split_f32_tag(ggml_tensor * t) {
    if(!ggml_dsv41_split_f32_shape(t))return false;
    t->op_params[4]=0x53335446;t->op_params[5]=1;return true;
}
inline bool ggml_dsv41_split_f32_pv_shape(const ggml_tensor * t) {
    return t&&!t->view_src&&!t->src[2]&&ggml_dsv41_pv_view_f32_valid(t)&&
        t->src[0]->ne[1]%4==0&&t->src[1]->ne[0]==64&&
        t->src[1]->nb[0]==sizeof(float)*size_t(t->src[1]->ne[1]);
}
inline bool ggml_dsv41_split_f32_pv_tag(ggml_tensor * t) {
    if(!ggml_dsv41_split_f32_pv_shape(t))return false;
    t->op_params[4]=0x53335446;t->op_params[5]=1;return true;
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_split_f32_queued(void);
