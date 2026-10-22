#pragma once
#include "ggml.h"
#include <stdint.h>
static inline bool ggml_dsv41_bf16_vector_eligible(const struct ggml_tensor *t) {
    return t && t->op == GGML_OP_DSV4_ACT_QAT && t->op_params[0] == GGML_DSV41_QAT_BF16 &&
        t->op_params[1] == 0 && t->src[0] && t->type == GGML_TYPE_F32 &&
        t->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        ggml_is_contiguous(t->src[0]) && ggml_are_same_shape(t,t->src[0]) &&
        ggml_nelements(t) > 0 && ggml_nelements(t) % 4 == 0 && t->data && t->src[0]->data &&
        ((uintptr_t)t->data % 16) == 0 && ((uintptr_t)t->src[0]->data % 16) == 0;
}
#ifdef __cplusplus
extern "C" {
#endif
GGML_API uint64_t ggml_backend_cuda_dsv41_bf16_vector_queued(void);
#ifdef __cplusplus
}
#endif
