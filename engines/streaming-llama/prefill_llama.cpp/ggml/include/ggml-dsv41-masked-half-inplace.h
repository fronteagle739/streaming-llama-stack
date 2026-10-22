#pragma once
#include "ggml.h"

// Permission only: the allocator retains ownership, output/view/layout and
// last-reader checks. Both existing backends read/write one independent F32
// element and use no restrict-qualified input/output pointers. Never alias mask.
#define GGML_DSV41_MASKED_HALF_INPLACE_MAGIC 0x494d4831
static inline bool ggml_dsv41_masked_half_inplace_requested(const struct ggml_tensor * t) {
    if (!(t && t->op == GGML_OP_DSV41_MASKED_HALF &&
          t->op_params[7] == GGML_DSV41_MASKED_HALF_INPLACE_MAGIC &&
          t->src[0] && t->src[1] && t->type == GGML_TYPE_F32 &&
          t->src[0]->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_F32 &&
          ggml_is_contiguous(t) && ggml_is_contiguous(t->src[0]) && ggml_is_contiguous(t->src[1]))) return false;
    if (t->ne[0] < 1 || t->ne[0] > 512 || t->ne[1] < 1 || t->ne[1] > 640 ||
        t->ne[2] < 1 || t->ne[2] > 4096 || t->ne[3] != 1 ||
        t->src[1]->ne[0] != t->ne[1] || t->src[1]->ne[1] != t->ne[2] ||
        t->src[1]->ne[2] != 1 || t->src[1]->ne[3] != 1) return false;
    for (int d = 0; d < GGML_MAX_DIMS; ++d)
        if (t->ne[d] != t->src[0]->ne[d] || t->nb[d] != t->src[0]->nb[d]) return false;
    return true;
}
static inline bool ggml_dsv41_tag_masked_half_inplace(struct ggml_tensor * t) {
    if (!t || t->op != GGML_OP_DSV41_MASKED_HALF || t->op_params[7]) return false;
    t->op_params[7] = GGML_DSV41_MASKED_HALF_INPLACE_MAGIC;
    if (ggml_dsv41_masked_half_inplace_requested(t)) return true;
    t->op_params[7] = 0;
    return false;
}
