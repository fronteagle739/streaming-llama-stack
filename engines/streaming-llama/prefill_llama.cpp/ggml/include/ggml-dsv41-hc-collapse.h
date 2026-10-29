#pragma once
#include "ggml.h"
#define GGML_DSV41_HC_COLLAPSE_MAGIC 0x4843434c
static inline bool ggml_dsv41_hc_collapse_requested(const struct ggml_tensor *t) {
    return t && t->op == GGML_OP_MAP_CUSTOM3 && t->op_params[7] == GGML_DSV41_HC_COLLAPSE_MAGIC;
}
static inline bool ggml_dsv41_hc_collapse_valid(const struct ggml_tensor *t) {
    if (!ggml_dsv41_hc_collapse_requested(t) || t->type != GGML_TYPE_F32) return false;
    const struct ggml_tensor *r=t->src[0], *p=t->src[1];
    return r && p && t->src[2]==p && r->type==GGML_TYPE_F32 && p->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(r) && ggml_is_contiguous(p) && ggml_is_contiguous(t) &&
        r->ne[0]>0 && r->ne[1]==4 && r->ne[2]>0 && r->ne[3]==1 &&
        p->ne[0]==4 && p->ne[1]==r->ne[2] && p->ne[2]==1 && p->ne[3]==1 &&
        t->ne[0]==r->ne[0] && t->ne[1]==r->ne[2] && t->ne[2]==1 && t->ne[3]==1;
}
