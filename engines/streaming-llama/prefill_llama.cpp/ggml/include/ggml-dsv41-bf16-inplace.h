#pragma once
#include "ggml.h"

// Private opt-in for the exact per-element BF16 round-trip only. No group
// quantization, external buffers or output lifetime rules are relaxed.
#define GGML_DSV41_BF16_INPLACE_MAGIC 0x49424631
static inline bool ggml_dsv41_bf16_inplace_requested(const struct ggml_tensor *t){
    if(!(t&&t->op==GGML_OP_DSV4_ACT_QAT&&t->op_params[0]==GGML_DSV41_QAT_BF16&&t->op_params[1]==0&&t->op_params[7]==GGML_DSV41_BF16_INPLACE_MAGIC&&
        t->src[0]&&t->type==GGML_TYPE_F32&&t->src[0]->type==GGML_TYPE_F32&&ggml_is_contiguous(t)&&ggml_is_contiguous(t->src[0])))return false;
    for(int d=0;d<GGML_MAX_DIMS;++d)if(t->ne[d]!=t->src[0]->ne[d]||t->nb[d]!=t->src[0]->nb[d])return false;
    return true;
}
static inline bool ggml_dsv41_tag_bf16_inplace(struct ggml_tensor *t){
    if(!t||t->op!=GGML_OP_DSV4_ACT_QAT||t->op_params[0]!=GGML_DSV41_QAT_BF16||t->op_params[1]!=0||t->op_params[7])return false;
    t->op_params[7]=GGML_DSV41_BF16_INPLACE_MAGIC;
    if(ggml_dsv41_bf16_inplace_requested(t))return true;
    t->op_params[7]=0;return false;
}
