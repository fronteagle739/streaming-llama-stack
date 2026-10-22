#pragma once
#include "ggml.h"

// Private, explicit DSV4.1 attention layout. OUT_PROD already represents A*B^T;
// B is a transposed view of contiguous F32 probabilities. Canonical K/V, half
// rounding, scores, softmax and sinks retain their existing types/operations.
#define GGML_DSV41_PV_VIEW_F32_MAGIC 0x41505632
static inline bool ggml_dsv41_pv_view_f32_requested(const struct ggml_tensor * t) {
    return t && t->op == GGML_OP_OUT_PROD && t->op_params[15] == GGML_DSV41_PV_VIEW_F32_MAGIC;
}
static inline bool ggml_dsv41_pv_view_f32_valid(const struct ggml_tensor * t) {
    if (!ggml_dsv41_pv_view_f32_requested(t) || !t->src[0] || !t->src[1]) return false;
    const struct ggml_tensor * a=t->src[0], * b=t->src[1];
    return a->type==GGML_TYPE_F32 && b->type==GGML_TYPE_F32 && t->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(a) && ggml_is_contiguous(t) &&
        a->ne[0]==512 && a->ne[1]>=1 && a->ne[1]<=640 && a->ne[2]>=1 && a->ne[2]<=4096 && a->ne[3]==1 &&
        b->ne[0]>=1 && b->ne[0]<=64 && b->ne[1]==a->ne[1] && b->ne[2]==a->ne[2] && b->ne[3]==1 &&
        b->nb[1]==sizeof(float) &&
        ((b->nb[0]==sizeof(float)*(size_t)b->ne[1] &&
          b->nb[2]==sizeof(float)*(size_t)b->ne[0]*(size_t)b->ne[1]) ||
         (b->nb[0]==sizeof(float)*(size_t)b->ne[1]*(size_t)b->ne[2] &&
          b->nb[2]==sizeof(float)*(size_t)b->ne[1])) &&
        b->nb[3]==sizeof(float)*(size_t)b->ne[0]*(size_t)b->ne[1]*(size_t)b->ne[2] &&
        t->ne[0]==a->ne[0] && t->ne[1]==b->ne[0] && t->ne[2]==b->ne[2] && t->ne[3]==1;
}
static inline struct ggml_tensor * ggml_dsv41_pv_view_f32(struct ggml_context * ctx,
        struct ggml_tensor * values, struct ggml_tensor * probabilities) {
    GGML_ASSERT(ggml_is_contiguous(values));
    struct ggml_tensor * out=ggml_out_prod(ctx,values,ggml_permute(ctx,probabilities,1,0,2,3));
    out->op_params[15]=GGML_DSV41_PV_VIEW_F32_MAGIC;
    GGML_ASSERT(ggml_dsv41_pv_view_f32_valid(out));
    return out;
}
