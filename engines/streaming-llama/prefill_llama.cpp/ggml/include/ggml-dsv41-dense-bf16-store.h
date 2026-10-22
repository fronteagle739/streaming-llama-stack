#pragma once
#include "ggml.h"
#include <stdint.h>
#include <string.h>

#define GGML_DSV41_DENSE_BF16_STORE_MAGIC 0x44424631
static inline bool ggml_dsv41_dense_bf16_store_requested(const struct ggml_tensor *t) {
    if (!(t && t->op == GGML_OP_MUL_MAT && t->op_params[6] == 0x51384131 &&
          t->op_params[7] == GGML_DSV41_DENSE_BF16_STORE_MAGIC && t->src[0] && t->src[1] &&
          t->type == GGML_TYPE_F32 && t->src[0]->type == GGML_TYPE_Q8_0 &&
          t->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
          ggml_is_contiguous(t->src[0]) && t->src[1]->nb[0] == sizeof(float))) return false;
    return t->src[0]->ne[0] > 0 && t->src[0]->ne[0] <= 16384 &&
        t->src[0]->ne[1] > 0 && t->src[0]->ne[1] <= 32768 &&
        t->src[0]->ne[2] == 1 && t->src[0]->ne[3] == 1 &&
        t->src[1]->ne[0] == t->src[0]->ne[0] && t->src[1]->ne[1] > 8 &&
        t->src[1]->ne[1] <= 4096 && t->src[1]->ne[2] == 1 && t->src[1]->ne[3] == 1;
}

// The sole F32 product reader must be its BF16 boundary. Other readers and raw
// capture outputs retain the original two operations. The boundary remains a
// live view of the identically rounded producer store, including output flags.
static inline unsigned ggml_dsv41_fuse_dense_bf16_stores(struct ggml_cgraph *graph) {
    unsigned fused = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        struct ggml_tensor *q = ggml_graph_node(graph,i), *p = q->src[0];
        if (q->op != GGML_OP_DSV4_ACT_QAT || q->op_params[0] != GGML_DSV41_QAT_BF16 ||
            q->op_params[1] != 0 || !p || p->op != GGML_OP_MUL_MAT || p->op_params[7] ||
            (p->flags & GGML_TENSOR_FLAG_OUTPUT) || p->view_src || q->view_src ||
            !ggml_is_contiguous(q) || !ggml_are_same_shape(q,p)) continue;
        unsigned users = 0;
        for (int j = 0; j < ggml_graph_n_nodes(graph); ++j)
            for (int s = 0; s < GGML_MAX_SRC; ++s) users += ggml_graph_node(graph,j)->src[s] == p;
        if (users != 1) continue;
        p->op_params[7] = GGML_DSV41_DENSE_BF16_STORE_MAGIC;
        if (!ggml_dsv41_dense_bf16_store_requested(p)) { p->op_params[7] = 0; continue; }
        q->op = GGML_OP_VIEW; q->view_src = p; q->view_offs = 0;
        memset(q->op_params,0,sizeof(q->op_params));
        ++fused;
    }
    return fused;
}
#ifdef __cplusplus
extern "C" {
#endif
GGML_API uint64_t ggml_backend_cuda_dsv41_dense_bf16_store_queued(void);
#ifdef __cplusplus
}
#endif
