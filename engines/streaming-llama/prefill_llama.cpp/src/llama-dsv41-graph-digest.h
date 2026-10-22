#pragma once
#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-backend.h"
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
// Diagnostic structure fingerprint only; never a numerical qualification hash.
inline uint64_t llama_dsv41_graph_digest(ggml_context *ctx,ggml_cgraph *graph,bool allocated=false){
    std::vector<const ggml_tensor*> all;
    std::unordered_map<const ggml_tensor*,uint32_t> ids;
    for(auto *t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)){
        all.push_back(t);ids.emplace(t,uint32_t(all.size()));
    }
    uint64_t h=14695981039346656037ull;
    auto bytes=[&](const void *p,size_t n){auto *s=static_cast<const uint8_t*>(p);for(size_t i=0;i<n;++i){h^=s[i];h*=1099511628211ull;}};
    auto value=[&](uint64_t x){bytes(&x,sizeof(x));};
    auto reference=[&](const ggml_tensor *t){
        if(!t){value(0);return;}
        auto it=ids.find(t);if(it!=ids.end()){value(it->second);return;}
        value(UINT64_MAX);bytes(t->name,std::strlen(t->name));
        value(t->type);bytes(t->ne,sizeof(t->ne));bytes(t->nb,sizeof(t->nb));
    };
    value(all.size());
    for(auto *t:all){
        value(t->type);bytes(t->ne,sizeof(t->ne));bytes(t->nb,sizeof(t->nb));value(t->op);
        bytes(t->op_params,sizeof(t->op_params));value(uint32_t(t->flags));value(t->view_offs);
        value(t->data!=nullptr);value(t->buffer!=nullptr);
        if(allocated&&t->buffer&&t->data){value(uintptr_t(t->data)-uintptr_t(ggml_backend_buffer_get_base(t->buffer)));value(ggml_backend_buffer_get_size(t->buffer));}
        value(t->data&&t->view_src&&t->view_src->data?uintptr_t(t->data)-uintptr_t(t->view_src->data):0);
        for(auto *src:t->src)reference(src);reference(t->view_src);
    }
    value(ggml_graph_n_nodes(graph));
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)reference(ggml_graph_node(graph,i));
    return h;
}
