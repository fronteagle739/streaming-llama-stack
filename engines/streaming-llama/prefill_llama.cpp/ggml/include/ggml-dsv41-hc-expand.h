#pragma once
#include "ggml.h"
#define GGML_DSV41_HC_EXPAND_MAGIC 0x44534845
#define GGML_DSV41_HC_BF16_MAGIC 0x42463146
#ifdef __cplusplus
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_hc_expand_queued(void);
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_hc_bf16_queued(void);
#endif
static inline bool ggml_dsv41_hc_expand_requested(const struct ggml_tensor*t){return t&&t->op==GGML_OP_MAP_CUSTOM3&&t->op_params[7]==GGML_DSV41_HC_EXPAND_MAGIC;}
static inline bool ggml_dsv41_hc_expand_valid(const struct ggml_tensor*t){
 if(!ggml_dsv41_hc_expand_requested(t)||t->type!=GGML_TYPE_F32||(t->op_params[6]!=0&&t->op_params[6]!=GGML_DSV41_HC_BF16_MAGIC))return false;
 for(int i=0;i<3;++i)if(!t->src[i]||t->src[i]->type!=GGML_TYPE_F32||!ggml_is_contiguous(t->src[i]))return false;
 const struct ggml_tensor*r=t->src[0],*b=t->src[1],*m=t->src[2];const int64_t width=r->ne[0],tokens=r->ne[2];
 return width>0&&width<=5120&&tokens>0&&tokens<=4096&&r->ne[1]==4&&r->ne[3]==1&&
  b->ne[0]==width&&b->ne[1]==tokens&&b->ne[2]==1&&b->ne[3]==1&&
  m->ne[0]==20&&m->ne[1]==tokens&&m->ne[2]==1&&m->ne[3]==1&&ggml_are_same_shape(t,r);
}

static inline bool ggml_dsv41_hc_bf16_requested(const struct ggml_tensor*t){return ggml_dsv41_hc_expand_valid(t)&&t->op_params[6]==GGML_DSV41_HC_BF16_MAGIC;}
