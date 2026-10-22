#pragma once
#include "ggml.h"
#define GGML_DSV41_SHARED_SPARSE_MAGIC 0x44534635
#define GGML_DSV41_TOKEN_TILE_MAGIC 0x44535454
#ifdef __cplusplus
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_token_tile_queued(void);
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_token_union_queued(void);
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_token_f16_table_queued(void);
#endif
static inline bool ggml_dsv41_token_tile_requested(const struct ggml_tensor *t){
    return t&&t->op==GGML_OP_FLASH_ATTN_EXT_SPARSE&&t->op_params[4]==GGML_DSV41_TOKEN_TILE_MAGIC;
}
static inline bool ggml_dsv41_token_tile_valid(const struct ggml_tensor *t){
    if(!ggml_dsv41_token_tile_requested(t)||t->type!=GGML_TYPE_F32)return false;
    for(int i=0;i<6;++i)if(!t->src[i]||!ggml_is_contiguous(t->src[i]))return false;
    const struct ggml_tensor *q=t->src[0],*r=t->src[1],*c=t->src[2],*rec=t->src[3],*n=t->src[4],*s=t->src[5];
    const int64_t count=q->ne[2],start=t->op_params[3],top=t->op_params[2],tiles=(count+3)/4;
    if(t->op_params[5]!=0&&t->op_params[5]!=1)return false;
    if(t->op_params[7]!=0&&t->op_params[7]!=1)return false;
    const bool direct_f16=t->op_params[7]==1;
    if(c->type!=(direct_f16?GGML_TYPE_F16:GGML_TYPE_F32)||c->ne[1]>(direct_f16?1048577:131073))return false;
    // The existing shared bitmap union is bounded independently of KV format.
    // Long callers must explicitly request the existing unmerged records path.
    if(t->op_params[5]&&top&&c->ne[1]>131073)return false;
    if(t->op_params[5]&&top){const struct ggml_tensor *ids=t->src[6];
        if(!ids||ids->type!=GGML_TYPE_I32||!ggml_is_contiguous(ids)||ids->ne[0]!=top||ids->ne[1]!=count||ids->ne[2]!=1||ids->ne[3]!=1||
            (t->op_params[6]!=1&&t->op_params[6]!=2))return false;}
    return q->type==GGML_TYPE_F32&&q->ne[0]==512&&q->ne[1]==64&&count>0&&count<=4096&&q->ne[3]==1&&
        start>=0&&start<=1048576&&r->type==GGML_TYPE_F32&&r->ne[0]==512&&r->ne[1]==count+(start<127?start:127)&&r->ne[2]==1&&r->ne[3]==1&&
        c->ne[0]==512&&c->ne[1]>0&&c->ne[2]==1&&c->ne[3]==1&&
        top>=0&&top<=512&&rec->type==GGML_TYPE_I32&&rec->ne[0]==2&&rec->ne[1]==(top?4*top:1)&&rec->ne[2]==tiles&&rec->ne[3]==1&&
        n->type==GGML_TYPE_I32&&ggml_nelements(n)==tiles&&s->type==GGML_TYPE_F32&&ggml_nelements(s)==64&&ggml_are_same_shape(q,t);
}
// Explicit private graph tag; original sparse dispatch remains unchanged.
static inline bool ggml_dsv41_shared_sparse_requested(const struct ggml_tensor *t){
    return t&&t->op==GGML_OP_FLASH_ATTN_EXT_SPARSE&&t->op_params[4]==GGML_DSV41_SHARED_SPARSE_MAGIC;
}
static inline bool ggml_dsv41_shared_sparse_valid(const struct ggml_tensor *t){
    if(!ggml_dsv41_shared_sparse_requested(t)||t->type!=GGML_TYPE_F32)return false;
    for(int i=0;i<6;++i)if(!t->src[i]||!ggml_is_contiguous(t->src[i]))return false;
    const struct ggml_tensor *q=t->src[0],*k=t->src[1],*ids=t->src[3],*m=t->src[4],*s=t->src[5];
    return q->type==GGML_TYPE_F32&&k->type==GGML_TYPE_F32&&t->src[2]==k&&ids->type==GGML_TYPE_I32&&
      m->type==GGML_TYPE_F32&&s->type==GGML_TYPE_F32&&q->ne[0]==512&&q->ne[1]>0&&q->ne[1]<=64&&
      q->ne[2]>0&&q->ne[2]<=4096&&q->ne[3]==1&&k->ne[0]==512&&k->ne[1]==1&&k->ne[2]>0&&k->ne[2]<=131072&&k->ne[3]==1&&
      ids->ne[0]>0&&ids->ne[0]<=640&&ids->ne[1]==q->ne[2]&&ids->ne[2]==1&&ids->ne[3]==1&&
      ggml_are_same_shape(ids,m)&&ggml_nelements(s)==q->ne[1]&&ggml_are_same_shape(q,t)&&t->op_params[2]==ids->ne[0];
}
