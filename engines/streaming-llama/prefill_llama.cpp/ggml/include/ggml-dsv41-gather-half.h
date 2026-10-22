#pragma once
#include "ggml.h"

// Optional src3 is the compressed F32/BF16 table. IDs address the logical
// raw+compressed concatenation; no full joined table needs to be materialized.
#define GGML_DSV41_GATHER_HALF_MAGIC 0x47483131
inline bool ggml_dsv41_gather_half_requested(const ggml_tensor * t) {
    return t && t->op == GGML_OP_DSV41_MASKED_HALF && t->op_params[6] == GGML_DSV41_GATHER_HALF_MAGIC;
}
inline bool ggml_dsv41_gather_half_valid(const ggml_tensor * t) {
    if (!ggml_dsv41_gather_half_requested(t) || t->op_params[7] || t->view_src ||
        t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    const auto * table=t->src[0], * mask=t->src[1], * ids=t->src[2];
    const auto * compressed=t->src[3];
    if (!table || !mask || !ids || table==t || mask==t || ids==t || compressed==t) return false;
    for(int i=4;i<GGML_MAX_SRC;++i)if(t->src[i])return false;
    if(compressed && ((compressed->type!=GGML_TYPE_F32&&compressed->type!=GGML_TYPE_BF16)||
        compressed->ne[0]!=512||compressed->ne[1]<=0||compressed->ne[1]>1048576||
        compressed->ne[2]!=1||compressed->ne[3]!=1||!ggml_is_contiguous(compressed)||
        table->ne[1]>4096+128))return false;
    return table->type==GGML_TYPE_F32 && table->ne[0]==512 && table->ne[1]>0 && table->ne[1]<=1048576+4096+128 &&
        table->ne[2]==1 && table->ne[3]==1 && ggml_is_contiguous(table) &&
        ids->type==GGML_TYPE_I32 && ids->ne[0]>0 && ids->ne[0]<=640 && ids->ne[1]>0 && ids->ne[1]<=4096 &&
        ids->ne[2]==1 && ids->ne[3]==1 && ggml_is_contiguous(ids) &&
        mask->type==GGML_TYPE_F32 && ggml_are_same_shape(ids,mask) && ggml_is_contiguous(mask) &&
        t->ne[0]==512 && t->ne[1]==ids->ne[0] && t->ne[2]==ids->ne[1] && t->ne[3]==1;
}
inline ggml_tensor * ggml_dsv41_gather_half(ggml_context * ctx, ggml_tensor * table, ggml_tensor * ids, ggml_tensor * mask,ggml_tensor * compressed=nullptr) {
    GGML_ASSERT(ctx && table && ids && mask);
    auto * out=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,table->ne[0],ids->ne[0],ids->ne[1]);
    out->op=GGML_OP_DSV41_MASKED_HALF;out->src[0]=table;out->src[1]=mask;out->src[2]=ids;out->src[3]=compressed;
    out->op_params[6]=GGML_DSV41_GATHER_HALF_MAGIC;
    GGML_ASSERT(ggml_dsv41_gather_half_valid(out));
    return out;
}
extern "C" uint64_t ggml_backend_cuda_dsv41_gather_half_queued(void);
extern "C" uint64_t ggml_backend_cuda_dsv41_gather_half_split_queued(void);
