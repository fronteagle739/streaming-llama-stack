#pragma once
#include "ggml.h"
#define GGML_DSV41_HC_SINKHORN_MAGIC 0x44534853
static inline bool ggml_dsv41_hc_sinkhorn_requested(const struct ggml_tensor *t){
    return t&&t->op==GGML_OP_MAP_CUSTOM3&&t->op_params[7]==GGML_DSV41_HC_SINKHORN_MAGIC;
}
static inline bool ggml_dsv41_hc_sinkhorn_valid(const struct ggml_tensor *t){
    if(!ggml_dsv41_hc_sinkhorn_requested(t)||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||t->op_params[6]<0||t->op_params[6]>2)return false;
    for(int i=0;i<3;++i)if(!t->src[i]||t->src[i]->type!=GGML_TYPE_F32||!ggml_is_contiguous(t->src[i]))return false;
    const struct ggml_tensor *mix=t->src[0],*scale=t->src[1],*base=t->src[2];
    return mix->ne[0]==24&&mix->ne[1]>0&&mix->ne[1]<=65536&&mix->ne[2]==1&&mix->ne[3]==1&&
        ggml_nelements(scale)==3&&ggml_nelements(base)==24&&ggml_are_same_shape(t,mix);
}
