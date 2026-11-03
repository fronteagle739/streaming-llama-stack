// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

inline bool ggml_qfn_qsa_score_requested(const ggml_tensor * t) {
    return t && t->op == GGML_OP_CUSTOM && t->op_params[0] == 0x514e5153 && (t->op_params[1] == 1 || t->op_params[1] == 2 || t->op_params[1] == 3);
}
inline bool ggml_qfn_qsa_score_span(const ggml_tensor * t) {
    size_t span = ggml_type_size(t->type);
    const size_t element = span;
    if (t->data && uintptr_t(t->data) % element) return false;
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] < 1 || t->nb[i] % element ||
            uint64_t(t->ne[i] - 1) > (SIZE_MAX - span) / (t->nb[i] ? t->nb[i] : 1)) return false;
        span += size_t(t->ne[i] - 1) * t->nb[i];
    }
    return !t->data || uintptr_t(t->data) <= UINTPTR_MAX - span;
}
inline bool ggml_qfn_qsa_score_valid(const ggml_tensor * t) {
    if (!ggml_qfn_qsa_score_requested(t) || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    const auto * q = t->src[0], * k = t->src[1], * w = t->src[2];
    if (!q || !k || !w || q->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
        (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) ||
        q->ne[0] != 128 || q->ne[1] != 4 || q->ne[2] < 1 || q->ne[2] > 1048560 ||
        q->ne[3] < 1 || q->ne[3] > 65535 || k->ne[0] != 128 || k->ne[1] != 1 ||
        k->ne[2] < 1 || k->ne[2] > INT32_MAX || k->ne[3] != q->ne[3] ||
        w->ne[0] != 4 || w->ne[1] != q->ne[2] || w->ne[2] != 1 || w->ne[3] != q->ne[3] ||
        t->ne[0] != k->ne[2] || t->ne[1] != q->ne[2] || t->ne[2] != 1 || t->ne[3] != q->ne[3] ||
        q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) || w->nb[0] != sizeof(float)) return false;
    if (t->op_params[1] == 3) {
        const auto *pos=t->src[3];
        if (!pos || pos->type!=GGML_TYPE_I32 || !ggml_is_contiguous(pos) || ggml_nelements(pos)!=q->ne[2] ||
            t->op_params[4]<1 || t->op_params[5]<1 || !ggml_qfn_qsa_score_span(pos)) return false;
    }
    float a, b; std::memcpy(&a, t->op_params + 2, sizeof a); std::memcpy(&b, t->op_params + 3, sizeof b);
    return std::isfinite(a) && std::isfinite(b) && ggml_qfn_qsa_score_span(q) &&
           ggml_qfn_qsa_score_span(k) && ggml_qfn_qsa_score_span(w) && ggml_qfn_qsa_score_span(t);
}
// Same full score shape as scalar lightning_indexer. All supplied outer strides survive.
inline ggml_tensor * ggml_qfn_qsa_score(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                      ggml_tensor * w, float a, float b) {
    auto * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k->ne[2], q->ne[2], 1, q->ne[3]);
    t->op = GGML_OP_CUSTOM; t->op_params[0] = 0x514e5153; t->op_params[1] = 1;
    std::memcpy(t->op_params + 2, &a, sizeof a); std::memcpy(t->op_params + 3, &b, sizeof b);
    t->src[0] = q; t->src[1] = k; t->src[2] = w;
    GGML_ASSERT(ggml_qfn_qsa_score_valid(t)); return t;
}
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_qsa_score_available(char * error, size_t capacity);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_qsa_score_prepare(void * stream);
// Queued dispatches, including successful graph replays through the shared capture ledger.
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_qfn_qsa_score_queued(void);
