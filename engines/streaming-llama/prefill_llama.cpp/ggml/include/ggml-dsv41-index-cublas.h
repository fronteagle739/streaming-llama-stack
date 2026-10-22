#pragma once
#include "ggml-backend.h"
#include "ggml-dsv41-index-window.h"
// Explicit isolated arithmetic experiment; no ambient GLM option is adopted.
inline bool ggml_dsv41_index_cublas_requested(const ggml_tensor *t){
    return t&&t->op==GGML_OP_DSV41_INDEX_SCORE&&t->op_params[2]==0x44344943&&t->op_params[3]==1;
}
inline bool ggml_dsv41_index_cublas_shape(const ggml_tensor *t){
    if(!t||t->op!=GGML_OP_DSV41_INDEX_SCORE||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t))return false;
    const auto*q=t->src[0],*w=t->src[1],*k=t->src[2],*v=t->src[3],*m=t->src[4];
    if(!q||!w||!k||!v)return false;
    const int64_t n=q->ne[2],rows=k->ne[1];
    return q->type==GGML_TYPE_F32&&w->type==GGML_TYPE_F32&&k->type==GGML_TYPE_F32&&v->type==GGML_TYPE_I32&&
        q->ne[0]==128&&q->ne[1]==32&&q->ne[3]==1&&n>0&&n<=4096&&
        k->ne[0]==128&&rows>0&&rows<=131072&&k->ne[2]==1&&k->ne[3]==1&&
        w->ne[0]==32&&w->ne[1]==n&&w->ne[2]==1&&w->ne[3]==1&&
        ggml_nelements(v)==n&&t->ne[0]==rows&&t->ne[1]==n&&t->ne[2]==1&&t->ne[3]==1&&
        ggml_is_contiguous(q)&&ggml_is_contiguous(w)&&ggml_is_contiguous(k)&&ggml_is_contiguous(v)&&
        (!m||(m->type==GGML_TYPE_F32&&m->ne[0]==(rows+7)/8&&m->ne[1]==n&&m->ne[2]==1&&m->ne[3]==1&&ggml_is_contiguous(m)));
}
inline bool ggml_dsv41_index_cublas_tag(ggml_tensor *t){
    if(!ggml_dsv41_index_cublas_shape(t))return false;
    t->op_params[2]=0x44344943;t->op_params[3]=1;return true;
}
inline bool ggml_dsv41_index_bf16_requested(const ggml_tensor *t){return ggml_dsv41_index_cublas_requested(t)&&t->op_params[4]==1;}
inline bool ggml_dsv41_index_bf16_tag(ggml_tensor *t){if(!ggml_dsv41_index_cublas_tag(t))return false;t->op_params[4]=1;return true;}
// Internal contiguous-position proof supplied by the tile executor which also
// uploads visible[t]=(start+t+1)/ratio. Ordinary score callers keep full rows.
inline bool ggml_dsv41_index_bf16_causal_tag(ggml_tensor*t,uint32_t start,uint32_t ratio){
    if(!ggml_dsv41_index_bf16_requested(t)||(ratio!=1&&ratio!=2)||
        uint64_t(start)+uint64_t(t->src[0]->ne[2])>1048576)return false;
    if(t->op_params[8]&&(!ggml_dsv41_index_window_requested(t)||!ggml_dsv41_index_window_shape(t)))return false;
    t->op_params[5]=int32_t(start);t->op_params[6]=int32_t(ratio);return true;
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_index_cublas_queued(void);

inline bool ggml_dsv41_index_f32_requested(const ggml_tensor*t){return ggml_dsv41_index_cublas_requested(t)&&t->op_params[4]==2;}
inline bool ggml_dsv41_index_f32_tag(ggml_tensor*t){if(!ggml_dsv41_index_cublas_tag(t))return false;t->op_params[4]=2;return true;}
inline bool ggml_dsv41_index_f32_causal_tag(ggml_tensor*t,uint32_t start,uint32_t ratio){
    if(!ggml_dsv41_index_f32_requested(t)||(ratio!=1&&ratio!=2)||uint64_t(start)+uint64_t(t->src[0]->ne[2])>1048576)return false;
    if(t->op_params[8]&&(!ggml_dsv41_index_window_requested(t)||!ggml_dsv41_index_window_shape(t)))return false;
    t->op_params[5]=int32_t(start);t->op_params[6]=int32_t(ratio);return true;
}
