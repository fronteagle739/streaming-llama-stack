#pragma once
// Explicit DS4.1 Q4_K weights / half-RHS expert contract. No environment selector.
// MUL_MAT_ID currently uses only op_params[0] (precision); retain it as zero.
#include "ggml.h"
#include <string.h>
enum ggml_ds4_native_half_slot { GGML_DS4_NATIVE_HALF_GATE=1, GGML_DS4_NATIVE_HALF_UP=2, GGML_DS4_NATIVE_HALF_DOWN=3 };
static inline bool ggml_ds4_native_half_requested(const struct ggml_tensor * t) {
    return t &&
        (t->op_params[1]==0x44533448 || t->op_params[4]==0x3bacabb7);
}
static inline bool ggml_ds4_native_half_metadata_valid(const struct ggml_tensor * t) {
    if(!ggml_ds4_native_half_requested(t)||t->op!=GGML_OP_MUL_MAT_ID||t->op_params[0]!=0||t->op_params[1]!=0x44533448||
       t->op_params[2]!=1||t->op_params[3]<1||t->op_params[3]>3||t->op_params[4]!=0x3bacabb7)return false;
    for(unsigned i=5;i<GGML_MAX_OP_PARAMS/sizeof(int32_t);++i)if(t->op_params[i]!=0)return false;
    return true;
}
// Returns false without modifying any byte on malformed geometry/metadata.
static inline bool ggml_ds4_native_half_shape_valid(const struct ggml_tensor * t) {
    if(!ggml_ds4_native_half_metadata_valid(t)||t->type!=GGML_TYPE_F32)return false;
    const struct ggml_tensor * w=t->src[0],*x=t->src[1],*ids=t->src[2];
    if(!w||!x||!ids||w->type!=GGML_TYPE_Q4_K||x->type!=GGML_TYPE_F32||ids->type!=GGML_TYPE_I32)return false;
    const bool down=t->op_params[3]==GGML_DS4_NATIVE_HALF_DOWN;
    const int64_t k=down?2304:5120,n=down?5120:2304,c=down?6:1,tokens=ids->ne[1];
    if(tokens<1||tokens>4096||w->ne[0]!=k||w->ne[1]!=n||w->ne[2]!=384||w->ne[3]!=1||
       x->ne[0]!=k||x->ne[1]!=c||x->ne[2]!=tokens||x->ne[3]!=1||
       ids->ne[0]!=6||ids->ne[2]!=1||ids->ne[3]!=1||t->ne[0]!=n||t->ne[1]!=6||t->ne[2]!=tokens||t->ne[3]!=1)return false;
    return w->nb[0]==144&&w->nb[1]==(size_t)(k/256)*144&&w->nb[2]==6635520&&w->nb[3]==(size_t)384*6635520&&
       x->nb[0]==4&&x->nb[1]==(size_t)k*4&&x->nb[2]==(size_t)k*c*4&&x->nb[3]==(size_t)k*c*tokens*4&&
       ids->nb[0]==4&&ids->nb[1]==24&&ids->nb[2]==(size_t)tokens*24&&ids->nb[3]==(size_t)tokens*24&&
       t->nb[0]==4&&t->nb[1]==(size_t)n*4&&t->nb[2]==(size_t)n*24&&t->nb[3]==(size_t)n*24*tokens;
}
static inline bool ggml_ds4_native_half_tag(struct ggml_tensor * t,enum ggml_ds4_native_half_slot slot) {
    if(!t||t->op!=GGML_OP_MUL_MAT_ID||slot<1||slot>3)return false;
    for(unsigned i=0;i<GGML_MAX_OP_PARAMS/sizeof(int32_t);++i)if(t->op_params[i]!=0)return false;
    t->op_params[1]=0x44533448;t->op_params[2]=1;t->op_params[3]=(int32_t)slot;t->op_params[4]=0x3bacabb7;
    if(ggml_ds4_native_half_shape_valid(t))return true;
    memset(t->op_params,0,sizeof(t->op_params));return false;
}
