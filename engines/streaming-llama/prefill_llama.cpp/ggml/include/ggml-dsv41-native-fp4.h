#pragma once
// Explicit operator experiment. No model weights are converted by this API.
#include "ggml-backend.h"
inline bool ggml_dsv41_native_fp4_requested(const ggml_tensor *t){
    return t&&t->op==GGML_OP_MUL_MAT_ID&&t->op_params[2]==0x46345034&&t->op_params[3]==1;
}
inline bool ggml_dsv41_native_fp4_valid(const ggml_tensor *t){
    if(!ggml_dsv41_native_fp4_requested(t)||t->type!=GGML_TYPE_F32||t->op_params[0]!=GGML_PREC_DEFAULT)return false;
    const auto*w=t->src[0],*x=t->src[1],*ids=t->src[2];
    return w&&x&&ids&&w->type==GGML_TYPE_MXFP4&&x->type==GGML_TYPE_F32&&ids->type==GGML_TYPE_I32&&
        ((w->ne[0]==5120&&w->ne[1]==2304)||(w->ne[0]==2304&&w->ne[1]==5120))&&w->ne[2]==384&&w->ne[3]==1&&
        x->ne[0]==w->ne[0]&&(x->ne[1]==1||x->ne[1]==6)&&x->ne[2]>4&&x->ne[2]<=4096&&x->ne[3]==1&&
        ids->ne[0]==6&&ids->ne[1]==x->ne[2]&&ids->ne[2]==1&&ids->ne[3]==1&&
        t->ne[0]==w->ne[1]&&t->ne[1]==6&&t->ne[2]==x->ne[2]&&t->ne[3]==1&&
        ggml_is_contiguous(w)&&ggml_is_contiguous(x)&&ggml_is_contiguous(ids)&&ggml_is_contiguous(t);
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_native_mxfp4_routed_queued(void);
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_native_mxfp4_dense_queued(void);
// Reuses the existing FP4 activation quantizer with ordinary17-byte weight
// blocks as its destination. Caller owns disjoint CUDA input/output/scratch.
// No allocation, model edit, disk I/O or implicit stream synchronization.
extern "C" GGML_BACKEND_API int ggml_backend_cuda_dsv41_pack_weights_mxfp4(
    const ggml_tensor *source,ggml_tensor *destination,void *scratch,size_t scratch_bytes,void *stream);
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_mxfp4_weight_packs_queued(void);

// Explicit disjoint Q4_K source-loader fusion; same FP4 rounding rules.
extern "C" GGML_BACKEND_API int ggml_backend_cuda_dsv41_pack_q4k_mxfp4_fused(
    const ggml_tensor *source,ggml_tensor *destination,void *stream);
