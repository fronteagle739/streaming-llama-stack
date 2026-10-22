#pragma once
#include "ggml-dsv41-index-window.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

// Recover the query partition from the original uploaded visibility owner.
// Long key windows wrap that partition in the existing exact local-visibility
// expression. Recognize only that expression; do not infer ownership by shape.
inline bool llama_dsv41_index_query_offset(const ggml_tensor * score,
        const ggml_tensor * owner,uint32_t count,uint32_t attention_rows,uint32_t & offset) {
    if(!score||score->op!=GGML_OP_DSV41_INDEX_SCORE||!owner||!count||!attention_rows||
       count>16384||attention_rows>4096||owner->type!=GGML_TYPE_I32||
       owner->ne[0]!=count||ggml_nelements(owner)!=count||!ggml_is_contiguous(owner))return false;
    const ggml_tensor * v=score->src[3];
    if(!v||v->type!=GGML_TYPE_I32||!ggml_is_contiguous(v))return false;
    auto same_shape=[](const ggml_tensor*a,const ggml_tensor*b){
        return a&&b&&!std::memcmp(a->ne,b->ne,sizeof(a->ne))&&ggml_is_contiguous(a)&&ggml_is_contiguous(b);
    };
    auto scalar_pair=[](const ggml_tensor*t,float a,float b){
        float p[2];std::memcpy(p,t->op_params,sizeof(p));return p[0]==a&&p[1]==b;
    };
    if(score->op_params[8]){
        if(!ggml_dsv41_index_window_requested(score)||!ggml_dsv41_index_window_shape(score)||
           v->op!=GGML_OP_CPY||v->src[1]!=v||v->view_src)return false;
        const auto * clamp=v->src[0];
        const auto * scale=clamp?clamp->src[0]:nullptr;
        const auto * cast=scale?scale->src[0]:nullptr;
        const auto * original=cast?cast->src[0]:nullptr;
        if(!clamp||!scale||!cast||!original||clamp->op!=GGML_OP_CLAMP||
           scale->op!=GGML_OP_SCALE||cast->op!=GGML_OP_CPY||cast->src[1]!=cast||
           clamp->type!=GGML_TYPE_F32||scale->type!=GGML_TYPE_F32||cast->type!=GGML_TYPE_F32||
           original->type!=GGML_TYPE_I32||clamp->view_src!=scale||clamp->view_offs||
           scale->view_src||cast->view_src||!same_shape(v,clamp)||!same_shape(v,scale)||
           !same_shape(v,cast)||!same_shape(v,original)||
           !scalar_pair(clamp,0.f,float(score->ne[0]))||
           !scalar_pair(scale,1.f,-float(score->op_params[7])))return false;
        for(int i=1;i<GGML_MAX_SRC;++i)if(clamp->src[i]||scale->src[i])return false;
        for(int i=2;i<GGML_MAX_SRC;++i)if(v->src[i]||cast->src[i])return false;
        v=original;
    }
    uint64_t first=0;
    if(v!=owner){
        if(v->op!=GGML_OP_VIEW||v->view_src!=owner||v->src[0]!=owner||
           v->type!=GGML_TYPE_I32||v->view_offs%sizeof(int32_t)||!ggml_is_contiguous(v))return false;
        first=v->view_offs/sizeof(int32_t);
    }
    if(first>=count||first%attention_rows||v->ne[0]!=std::min<uint64_t>(attention_rows,count-first)||
       uint64_t(ggml_nelements(v))!=std::min<uint64_t>(attention_rows,count-first)||
       ggml_nelements(score->src[3])!=ggml_nelements(v))return false;
    offset=uint32_t(first);return true;
}
