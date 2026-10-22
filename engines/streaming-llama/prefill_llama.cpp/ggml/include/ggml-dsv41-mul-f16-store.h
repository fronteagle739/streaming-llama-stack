#pragma once
#include "ggml.h"
#include <stdint.h>
#include <string.h>

#define GGML_DSV41_MUL_F16_STORE_MAGIC 0x4d483031
static inline bool ggml_dsv41_mul_f16_store_requested(const struct ggml_tensor*t){
    return t&&t->op==GGML_OP_MUL&&t->op_params[7]==GGML_DSV41_MUL_F16_STORE_MAGIC&&
        t->type==GGML_TYPE_F32&&t->src[0]&&t->src[1]&&
        t->src[0]->op==GGML_OP_GLU&&t->src[0]->type==GGML_TYPE_F32&&t->src[1]->type==GGML_TYPE_F32&&
        ggml_are_same_shape(t,t->src[0])&&ggml_is_contiguous(t)&&
        ggml_is_contiguous(t->src[0])&&ggml_is_contiguous(t->src[1])&&
        t->ne[0]>0&&t->ne[0]<=16384&&t->ne[1]==6&&t->ne[2]>0&&t->ne[2]<=16384&&t->ne[3]==1&&
        t->src[1]->ne[0]==1&&t->src[1]->ne[1]==6&&t->src[1]->ne[2]==t->ne[2]&&t->src[1]->ne[3]==1;
}

// Fresh, sole-reader MUL -> cast(F16) -> cast(F32). Represent the internal
// half value as its exact F32 expansion after the producer rounds its store.
// Raw product/half capture outputs and any additional readers forbid folding.
static inline unsigned ggml_dsv41_fuse_mul_f16_stores(struct ggml_cgraph*g){
    unsigned fused=0;
    for(int i=0;i<ggml_graph_n_nodes(g);++i){
        struct ggml_tensor*r=ggml_graph_node(g,i),*h=r->src[0],*p=h?h->src[0]:NULL;
        if(r->op!=GGML_OP_CPY||r->type!=GGML_TYPE_F32||r->src[1]!=r||!h||
           h->op!=GGML_OP_CPY||h->type!=GGML_TYPE_F16||h->src[1]!=h||!p||p->op!=GGML_OP_MUL||
           p->type!=GGML_TYPE_F32||p->op_params[7]||
           (p->flags&GGML_TENSOR_FLAG_OUTPUT)||(h->flags&GGML_TENSOR_FLAG_OUTPUT)||
           p->view_src||h->view_src||r->view_src||p->data||h->data||r->data||
           !ggml_is_contiguous(h)||!ggml_is_contiguous(r)||!ggml_are_same_shape(p,h)||!ggml_are_same_shape(p,r))continue;
        unsigned pu=0,hu=0;bool pn=false,hn=false;
        for(int j=0;j<ggml_graph_n_nodes(g);++j){
            const struct ggml_tensor*n=ggml_graph_node(g,j);pn|=n==p;hn|=n==h;
            for(int s=0;s<GGML_MAX_SRC;++s){pu+=n!=p&&n->src[s]==p;hu+=n!=h&&n->src[s]==h;}
        }
        if(!pn||!hn||pu!=1||hu!=1)continue;
        p->op_params[7]=GGML_DSV41_MUL_F16_STORE_MAGIC;
        if(!ggml_dsv41_mul_f16_store_requested(p)){p->op_params[7]=0;continue;}
        h->op=GGML_OP_VIEW;h->type=GGML_TYPE_F32;h->view_src=p;h->view_offs=0;
        memcpy(h->nb,p->nb,sizeof(h->nb));memset(h->op_params,0,sizeof(h->op_params));
        for(int s=1;s<GGML_MAX_SRC;++s)h->src[s]=NULL;
        r->op=GGML_OP_VIEW;r->view_src=p;r->view_offs=0;memset(r->op_params,0,sizeof(r->op_params));
        for(int s=1;s<GGML_MAX_SRC;++s)r->src[s]=NULL;
        ++fused;
    }
    return fused;
}
#ifdef __cplusplus
extern "C" {
#endif
GGML_API uint64_t ggml_backend_cuda_dsv41_mul_f16_store_queued(void);
#ifdef __cplusplus
}
#endif
