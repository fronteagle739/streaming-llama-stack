#pragma once
#include "ggml-backend.h"
// Private key-window metadata; score callers upload LOCAL visibility, while
// candidate-block callers retain GLOBAL visibility for the mandatory final
// block marker. Windows are aligned to the existing8192-key score slabs.
inline bool ggml_dsv41_index_window_requested(const ggml_tensor*t){return t&&t->op_params[8]==0x4453574e;}
inline bool ggml_dsv41_index_window_shape(const ggml_tensor*t){
    if(!t||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||t->op_params[7]<0||t->op_params[7]%8192)return false;
    const ggml_tensor*s=nullptr,*visible=nullptr;
    if(t->op==GGML_OP_DSV41_INDEX_SCORE){s=t;visible=t->src[3];}
    else if(t->op==GGML_OP_DSV41_CANDIDATE_BLOCKS){s=t->src[0];visible=t->src[1];}
    else return false;
    return s&&s->type==GGML_TYPE_F32&&ggml_is_contiguous(s)&&s->ne[0]>0&&s->ne[0]<=8192&&s->ne[1]>0&&s->ne[1]<=4096&&
        s->ne[2]==1&&s->ne[3]==1&&uint64_t(t->op_params[7])+uint64_t(s->ne[0])<=1048576&&visible&&
        visible->type==GGML_TYPE_I32&&ggml_is_contiguous(visible)&&ggml_nelements(visible)==s->ne[1]&&
        (t->op!=GGML_OP_DSV41_CANDIDATE_BLOCKS||(t->ne[0]==(s->ne[0]+7)/8&&t->ne[1]==s->ne[1]&&t->ne[2]==1&&t->ne[3]==1));
}
inline bool ggml_dsv41_index_window_tag(ggml_tensor*t,uint32_t first){
    if(!t||first>1048576||first%8192||(t->op_params[8]&& !ggml_dsv41_index_window_requested(t)))return false;
    const int32_t old=t->op_params[7];t->op_params[7]=int32_t(first);
    if(!ggml_dsv41_index_window_shape(t)){t->op_params[7]=old;return false;}
    t->op_params[8]=0x4453574e;return true;
}
inline uint32_t ggml_dsv41_index_window_offset(const ggml_tensor*t){
    GGML_ASSERT(t);if(!t->op_params[8])return 0;
    GGML_ASSERT(ggml_dsv41_index_window_requested(t)&&ggml_dsv41_index_window_shape(t));return uint32_t(t->op_params[7]);
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_candidate_window_queued(void);
