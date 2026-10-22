#pragma once
#include "ggml.h"
#define GGML_DSV41_EXPERT_CUBLAS_MAGIC 0x44534543
#ifdef __cplusplus
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_expert_cublas_queued(void);
#endif
static inline bool ggml_dsv41_expert_cublas_requested(const struct ggml_tensor*t){
 return t&&t->op==GGML_OP_MUL_MAT_ID&&t->op_params[4]==GGML_DSV41_EXPERT_CUBLAS_MAGIC;
}
static inline bool ggml_dsv41_expert_cublas_valid(const struct ggml_tensor*t){
 if(!t||t->op!=GGML_OP_MUL_MAT_ID||t->type!=GGML_TYPE_F32)return false;
 const struct ggml_tensor*w=t->src[0],*x=t->src[1],*ids=t->src[2];
 if(!w||!x||!ids||!ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(ids))return false;
 const int64_t k=w->ne[0],n=w->ne[1],tokens=x->ne[2];
 return w->type==GGML_TYPE_Q4_K&&w->ne[2]==384&&w->ne[3]==1&&
  ((k==5120&&n==2304&&x->ne[1]==1)||(k==2304&&n==5120&&x->ne[1]==6))&&
  x->type==GGML_TYPE_F32&&x->ne[0]==k&&tokens>0&&tokens<=4096&&x->ne[3]==1&&
  ids->type==GGML_TYPE_I32&&ids->ne[0]==6&&ids->ne[1]==tokens&&ids->ne[2]==1&&ids->ne[3]==1&&
  t->ne[0]==n&&t->ne[1]==6&&t->ne[2]==tokens&&t->ne[3]==1;
}
static inline bool ggml_dsv41_expert_cublas_tag(struct ggml_tensor*t){
 if(!ggml_dsv41_expert_cublas_valid(t))return false;t->op_params[4]=GGML_DSV41_EXPERT_CUBLAS_MAGIC;return true;
}
static inline uint64_t ggml_dsv41_expert_cublas_workspace(uint32_t tokens){
 // Worst of gate/up/down: one dequantized expert, sorted F16 operands and
 // F32 results, two maps, and2MiB pool rounding. Never a full F16 expert bank.
 return 2ull*5120*2304+uint64_t(tokens)*6*(2*5120+4*5120+8)+2ull*1024*1024;
}
