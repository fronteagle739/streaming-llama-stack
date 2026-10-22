#pragma once
// Explicit offline DS4 dense projection tag. No global precision override.
#include "ggml-backend.h"
#include <cstring>

inline bool ggml_dsv41_dense_half_requested(const ggml_tensor *t){
    return t&&t->op==GGML_OP_MUL_MAT&&t->op_params[2]==0x44344846&&t->op_params[3]==1;
}
inline bool ggml_dsv41_dense_half_shape(const ggml_tensor *t){
    if(!t||t->op!=GGML_OP_MUL_MAT||t->type!=GGML_TYPE_F32||t->op_params[0]!=GGML_PREC_F32)return false;
    const auto *w=t->src[0],*x=t->src[1];
    return w&&x&&w->type==GGML_TYPE_F16&&x->type==GGML_TYPE_F32&&
        w->ne[0]>0&&w->ne[0]<=16384&&w->ne[1]>0&&w->ne[1]<=32768&&
        x->ne[0]==w->ne[0]&&x->ne[1]>8&&x->ne[1]<=4096&&
        w->ne[2]==1&&w->ne[3]==1&&x->ne[2]==1&&x->ne[3]==1&&
        ggml_is_contiguous(w)&&ggml_is_contiguous(t)&&x->nb[0]==sizeof(float)&&
        x->nb[1]%sizeof(float)==0;
}
inline bool ggml_dsv41_dense_half_tag(ggml_tensor *t){
    if(!ggml_dsv41_dense_half_shape(t))return false;
    const ggml_tensor *root=t->src[0];
    for(unsigned depth=0;depth<32;++depth){
        if(root->view_src){root=root->view_src;continue;}
        if(root->op==GGML_OP_CPY&&root->src[0]){root=root->src[0];continue;}
        break;
    }
    if(root->op!=GGML_OP_NONE||!root->buffer||!root->data)return false;
    const char *name=root->name;
    if(std::strncmp(name,"blk.",4)!=0)return false;
    const char *end=name+std::strlen(name);
    const char *allowed[]={".attn_q_a.weight",".attn_q_b.weight",".attn_kv.weight",
        ".attn_output_a.weight",".attn_output_b.weight",".attn_compressor_kv.weight",
        ".attn_compressor_gate.weight",".indexer.attn_k.weight",".indexer.attn_q_b.weight",
        ".indexer.proj.weight",".ffn_gate_shexp.weight",".ffn_up_shexp.weight",
        ".ffn_down_shexp.weight",".engram_kv.weight"};
    bool accepted=false;
    for(const char *suffix:allowed){const auto n=std::strlen(suffix);
        if(size_t(end-name)>=n&&std::strcmp(end-n,suffix)==0){accepted=true;break;}}
    // HC projection, router, attention products and the output head are absent
    // from the allowlist. Tagged expert MUL_MAT_ID nodes cannot match this API.
    if(!accepted)return false;
    t->op_params[2]=0x44344846;t->op_params[3]=1;return true;
}
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_dense_half_queued(void);
