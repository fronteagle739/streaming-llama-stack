#pragma once
#include "ggml.h"
#include <cstring>

// Prove an unchanged compact-ID tensor from its actual graph sources. Paired
// graphs may concatenate the exact two ordered views of the original input.
inline bool llama_dsv41_compact_ids_unchanged(const ggml_tensor *value,const ggml_tensor *input){
    if(!value||!input||input->type!=GGML_TYPE_I32)return false;
    auto strip=[](const ggml_tensor *t){
        for(unsigned n=0;t&&n<8&&t->op==GGML_OP_CONT;++n){const auto*s=t->src[0];
            if(!s||t->type!=s->type||std::memcmp(t->ne,s->ne,sizeof(t->ne))||std::memcmp(t->nb,s->nb,sizeof(t->nb)))return static_cast<const ggml_tensor*>(nullptr);
            for(unsigned j=1;j<GGML_MAX_SRC;++j)if(t->src[j])return static_cast<const ggml_tensor*>(nullptr);t=s;}
        return t;
    };
    value=strip(value);if(value==input)return true;
    if(!value||value->type!=input->type||std::memcmp(value->ne,input->ne,sizeof(value->ne))||input->ne[2]!=1||input->ne[3]!=1)return false;
    int64_t cursor=0;unsigned leaves=0;
    auto visit=[&](auto&&self,const ggml_tensor*t,unsigned depth)->bool{
        t=strip(t);if(!t||depth>4||t->type!=input->type)return false;
        if(t->op==GGML_OP_CONCAT){
            if(t->op_params[0]!=1||t->ne[0]!=input->ne[0]||t->ne[2]!=1||t->ne[3]!=1)return false;
            if(!t->src[0]||!t->src[1]||t->src[0]->ne[1]+t->src[1]->ne[1]!=t->ne[1])return false;
            for(unsigned j=2;j<GGML_MAX_SRC;++j)if(t->src[j])return false;
            return self(self,t->src[0],depth+1)&&self(self,t->src[1],depth+1);
        }
        if(++leaves>4||t->op!=GGML_OP_VIEW||t->view_src!=input||t->src[0]!=input||t->ne[0]!=input->ne[0]||t->ne[1]<=0||t->ne[2]!=1||t->ne[3]!=1||
            t->nb[0]!=input->nb[0]||t->nb[1]!=input->nb[1]||t->view_offs!=uint64_t(cursor)*input->nb[1])return false;
        cursor+=t->ne[1];return cursor<=input->ne[1];
    };
    return visit(visit,value,0)&&leaves>=2&&cursor==input->ne[1];
}
