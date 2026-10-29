#pragma once
#include "ggml.h"

// Private permission for the V4.1 tail-pair operator only. The allocator still
// owns the last-reader, output, view, buffer and layout lifetime proof.
#define GGML_DSV41_ROPE_INPLACE_MAGIC 0x49525031
// op_params[1]: round every input element to BF16 first, exactly as a
// preceding QAT_BF16 round trip would (rope(bf16(x)) in one pass).
#define GGML_DSV41_ROPE_ROUND_INPUT 0x52424631
static inline bool ggml_dsv41_rope_inplace_requested(const struct ggml_tensor *t) {
    if (!(t && t->op == GGML_OP_DSV41_ROPE && t->op_params[7] == GGML_DSV41_ROPE_INPLACE_MAGIC &&
          t->src[0] && t->src[1] && t->src[2] && t->type == GGML_TYPE_F32 &&
          t->src[0]->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_I32 &&
          t->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
          ggml_is_contiguous(t->src[0]) && ggml_is_contiguous(t->src[1]) &&
          ggml_is_contiguous(t->src[2]))) return false;
    if ((t->ne[0] != 64 && t->ne[0] != 128 && t->ne[0] != 512) ||
        t->ne[1] < 1 || t->ne[1] > 64 || t->ne[2] < 1 || t->ne[2] > 4096 || t->ne[3] != 1 ||
        ggml_nelements(t->src[1]) != t->ne[2] || ggml_nelements(t->src[2]) != 32 ||
        (t->op_params[0] != 0 && t->op_params[0] != 1)) return false;
    for (int d = 0; d < GGML_MAX_DIMS; ++d)
        if (t->ne[d] != t->src[0]->ne[d] || t->nb[d] != t->src[0]->nb[d]) return false;
    return true;
}
static inline bool ggml_dsv41_tag_rope_inplace(struct ggml_tensor *t) {
    if (!t || t->op != GGML_OP_DSV41_ROPE || t->op_params[7]) return false;
    t->op_params[7] = GGML_DSV41_ROPE_INPLACE_MAGIC;
    if (ggml_dsv41_rope_inplace_requested(t)) return true;
    t->op_params[7] = 0;
    return false;
}
#ifdef __cplusplus
extern "C" {
#endif
GGML_API uint64_t ggml_backend_cuda_dsv41_rope_inplace_queued(void);
#ifdef __cplusplus
}
#endif
