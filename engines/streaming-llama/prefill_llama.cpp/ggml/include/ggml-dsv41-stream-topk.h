#pragma once
#include "ggml-backend.h"
// Internal bounded streaming merge: the existing large-input selector's
// score-descending/index-ascending order is required even for a small merge.
// NaNs are explicitly refused; their legacy comparison network is not a
// total order and therefore cannot support a local-topK merge proof.
inline bool ggml_dsv41_stream_topk_shape(const ggml_tensor*t){
    const auto*s=t?t->src[0]:nullptr;
    return t&&t->op==GGML_OP_TOP_K&&t->type==GGML_TYPE_I32&&s&&s->type==GGML_TYPE_F32&&
        ggml_is_contiguous(s)&&s->ne[0]>0&&s->ne[0]<=8192&&s->ne[1]>0&&s->ne[1]<=4096&&
        s->ne[2]==1&&s->ne[3]==1&&t->ne[0]>0&&t->ne[0]<=2048&&t->ne[0]<=s->ne[0]&&
        t->ne[1]==s->ne[1]&&t->ne[2]==1&&t->ne[3]==1;
}
inline bool ggml_dsv41_stream_topk_requested(const ggml_tensor*t){return t&&t->op==GGML_OP_TOP_K&&t->op_params[0]==0x4453544b&&t->op_params[1]==1;}
inline bool ggml_dsv41_stream_topk_tag(ggml_tensor*t){if(!ggml_dsv41_stream_topk_shape(t))return false;t->op_params[0]=0x4453544b;t->op_params[1]=1;return true;}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_stream_topk_queued(void);
