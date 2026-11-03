// SPDX-License-Identifier: MIT
#pragma once
#include "ggml.h"

// F32 expert outputs [E,K,T] and router weights [1,K,T]. Preserve the
// separate MUL then slot-ascending ADD arithmetic, including signed zero.
inline bool ggml_qfn_expert_reduce_requested(const ggml_tensor * t) {
    return t && t->op == GGML_OP_CUSTOM && t->op_params[0] == 0x51465244 && t->op_params[1] == 1;
}
inline bool ggml_qfn_expert_reduce_inputs_valid(const ggml_tensor * x, const ggml_tensor * w) {
    if (!x || !w || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
        x->ne[0] < 1 || x->ne[1] < 1 || x->ne[2] < 1 || x->ne[3] != 1 ||
        w->ne[0] != 1 || w->ne[1] != x->ne[1] || w->ne[2] != x->ne[2] || w->ne[3] != 1)
        return false;
    // Views are supported: the CUDA consumer uses all three byte strides.
    for (int d = 0; d < 3; ++d)
        if (x->nb[d] % sizeof(float) || w->nb[d] % sizeof(float)) return false;
    return true;
}
inline bool ggml_qfn_expert_reduce_valid(const ggml_tensor * t) {
    if (!(ggml_qfn_expert_reduce_requested(t) && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        ggml_qfn_expert_reduce_inputs_valid(t->src[0], t->src[1]) &&
        t->ne[0] == t->src[0]->ne[0] && t->ne[1] == t->src[0]->ne[2] &&
        t->ne[2] == 1 && t->ne[3] == 1)) return false;
    for (int i = 2; i < GGML_MAX_SRC; ++i) if (t->src[i]) return false;
    return true;
}
inline ggml_tensor * ggml_qfn_expert_reduce(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    GGML_ASSERT(ggml_qfn_expert_reduce_inputs_valid(x, w));
    auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[2]);
    t->op = GGML_OP_CUSTOM;
    t->op_params[0] = 0x51465244;
    t->op_params[1] = 1;
    t->src[0] = x; t->src[1] = w;
    ggml_set_name(t, "qfn_expert_weighted_reduce");
    GGML_ASSERT(ggml_qfn_expert_reduce_valid(t));
    return t;
}
