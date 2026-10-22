#pragma once
#include "ggml.h"
// Private explicit routing-builder tag; no arithmetic/precision selection.
inline bool ggml_dsv41_sorted_routing_requested(const ggml_tensor*t){return t&&t->op==GGML_OP_MUL_MAT_ID&&t->op_params[6]==0x53525431;}
inline bool ggml_dsv41_sorted_routing_tag(ggml_tensor*t,bool reuse=false){
    if(!t||t->op!=GGML_OP_MUL_MAT_ID||!t->src[0]||!t->src[1]||!t->src[2]||t->src[0]->type!=GGML_TYPE_Q4_K||t->src[0]->ne[2]!=384||
       t->src[1]->type!=GGML_TYPE_F32||t->src[2]->type!=GGML_TYPE_I32||t->src[2]->ne[0]!=6||t->src[2]->ne[1]<5||t->src[2]->ne[1]>16384)return false;
    t->op_params[6]=0x53525431;t->op_params[7]=reuse?0x4d415031:0;return true;
}

inline bool ggml_dsv41_routing_map_reuse_requested(const ggml_tensor*t){return ggml_dsv41_sorted_routing_requested(t)&&t->op_params[7]==0x4d415031;}
inline bool ggml_dsv41_routing_triplet(ggml_cgraph*g,ggml_tensor**out){
    ggml_tensor*nodes[3]={};unsigned n=0;
    for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(ggml_dsv41_routing_map_reuse_requested(t)){if(n==3)return false;nodes[n++]=t;}}
    if(n!=3)return false;
    const auto*a=nodes[0],*b=nodes[1],*d=nodes[2];
    if(a->src[2]!=b->src[2]||a->src[2]!=d->src[2]||a->src[1]!=b->src[1]||a->src[2]->ne[0]!=6||a->src[2]->ne[1]<5||a->src[2]->ne[1]>16384)return false;
    for(auto*t:nodes)if(t->src[0]->type!=GGML_TYPE_Q4_K||t->src[0]->ne[2]!=384||t->src[1]->type!=GGML_TYPE_F32||t->src[1]->ne[2]!=a->src[2]->ne[1]||t->src[1]->ne[3]!=1)return false;
    if(a->src[0]->ne[0]!=5120||b->src[0]->ne[0]!=5120||a->src[0]->ne[1]!=2304||b->src[0]->ne[1]!=2304||
       a->src[1]->ne[0]!=5120||a->src[1]->ne[1]!=1||d->src[0]->ne[0]!=2304||d->src[0]->ne[1]!=5120||d->src[1]->ne[0]!=2304||d->src[1]->ne[1]!=6)return false;
    for(unsigned i=0;i<3;++i)out[i]=nodes[i];return true;
}
struct ggml_backend_cuda_context;
class ggml_cuda_dsv41_routing_scope{
public:
    ggml_cuda_dsv41_routing_scope(ggml_backend_cuda_context&,ggml_cgraph*,bool uncaptured);
    ~ggml_cuda_dsv41_routing_scope();
    ggml_cuda_dsv41_routing_scope(const ggml_cuda_dsv41_routing_scope&)=delete;
    ggml_cuda_dsv41_routing_scope&operator=(const ggml_cuda_dsv41_routing_scope&)=delete;
private:void*state=nullptr;
};
char*ggml_cuda_dsv41_routing_map(ggml_backend_cuda_context&,const ggml_tensor*);
