#pragma once
// LLAMADECODE nondeterminism hunt (diagnostic only, LLAMADECODE_HASH_DUMP=1).
// Hashes what a layer graph consumes before compute (every non-weight leaf, i.e.
// uploaded inputs, resident banks and persistent caches) and what it produced
// after compute (kept outputs, persistent caches again, the staged weights; for
// the 384-expert tensors only the routed experts' slices). Two runs of the same
// request are then diffed line by line to find the first divergent (step, layer,
// tensor). Reading back synchronizes the backend; a race that disappears under
// this mode is itself a localization (an async producer the compute did not wait on).
#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>
#include <algorithm>

inline bool llama_dsv41_hash_dump_enabled(){
    static const bool on=[]{const char*e=std::getenv("LLAMADECODE_HASH_DUMP");return e&&e[0]=='1'&&!e[1];}();
    return on;
}
inline uint64_t llama_dsv41_hash_bytes(const void *data,size_t bytes,uint64_t h=1469598103934665603ull){
    const auto *p=static_cast<const unsigned char*>(data);size_t i=0;
    for(;i+8<=bytes;i+=8){uint64_t w;std::memcpy(&w,p+i,8);h^=w;h*=1099511628211ull;h^=h>>29;}
    for(;i<bytes;++i){h^=p[i];h*=1099511628211ull;}
    return h;
}
inline uint64_t llama_dsv41_hash_tensor(const ggml_tensor *t,size_t offset,size_t bytes,std::vector<unsigned char>&scratch){
    scratch.resize(bytes);ggml_backend_tensor_get(const_cast<ggml_tensor*>(t),scratch.data(),offset,bytes);
    return llama_dsv41_hash_bytes(scratch.data(),bytes);
}
// Leaves reachable from the graph (graph->leafs plus op-NONE sources of nodes).
inline std::vector<ggml_tensor*> llama_dsv41_hash_leaves(ggml_cgraph *graph){
    std::unordered_set<ggml_tensor*>seen;std::vector<ggml_tensor*>out;
    auto add=[&](ggml_tensor*t){if(t&&t->op==GGML_OP_NONE&&t->buffer&&t->data&&seen.insert(t).second)out.push_back(t);};
    for(int i=0;i<graph->n_leafs;++i)add(graph->leafs[i]);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);for(auto*s:n->src)if(s){add(s);if(s->view_src)add(s->view_src);}}
    return out;
}
inline bool llama_dsv41_hash_is_weight(const ggml_tensor *t){return !std::strncmp(t->name,"blk.",4)||!std::strcmp(t->name,"output.weight")||!std::strcmp(t->name,"output_norm.weight")||!std::strcmp(t->name,"token_embd.weight");}
// phase: "in" (before compute: non-weight leaves) or "out" (after compute).
inline void llama_dsv41_hash_phase(const char *phase,uint32_t layer,uint32_t position,uint32_t count,ggml_cgraph *graph,
        const std::vector<ggml_tensor*> &outputs,const std::vector<ggml_tensor*> &routed,const std::vector<int32_t> &experts){
    std::vector<unsigned char>scratch;const size_t cap=64ull*1024*1024;
    for(auto*t:llama_dsv41_hash_leaves(graph)){
        const bool weight=llama_dsv41_hash_is_weight(t);
        if(!std::strcmp(phase,"in")&&weight)continue;
        if(weight){
            bool is_routed=false;for(auto*r:routed)if(r==t)is_routed=true;
            if(is_routed){
                if(count>8)continue; // prefill tiles: routed weights are not hashed (cost)
                std::vector<int32_t>unique;for(const auto e:experts)if(std::find(unique.begin(),unique.end(),e)==unique.end())unique.push_back(e);
                std::sort(unique.begin(),unique.end());
                uint64_t h=1469598103934665603ull;
                for(const auto e:unique){if(e<0||e>=int32_t(t->ne[2]))continue;scratch.resize(t->nb[2]);
                    ggml_backend_tensor_get(t,scratch.data(),size_t(e)*t->nb[2],t->nb[2]);h=llama_dsv41_hash_bytes(scratch.data(),scratch.size(),h);}
                std::fprintf(stderr,"LLAMADECODE_HASH %s pos=%u count=%u layer=%u kind=weight_routed name=%s hash=%016llx\n",phase,position,count,layer,t->name,(unsigned long long)h);
                continue;
            }
        }
        const size_t bytes=std::min(ggml_nbytes(t),cap);
        std::fprintf(stderr,"LLAMADECODE_HASH %s pos=%u count=%u layer=%u kind=%s name=%s bytes=%zu hash=%016llx\n",phase,position,count,layer,
            weight?"weight":"leaf",t->name[0]?t->name:"(unnamed)",bytes,(unsigned long long)llama_dsv41_hash_tensor(t,0,bytes,scratch));
    }
    for(size_t i=0;i<outputs.size();++i)if(outputs[i]&&outputs[i]->buffer&&outputs[i]->data){
        const size_t bytes=std::min(ggml_nbytes(outputs[i]),cap);
        std::fprintf(stderr,"LLAMADECODE_HASH %s pos=%u count=%u layer=%u kind=output index=%zu name=%s bytes=%zu hash=%016llx\n",phase,position,count,layer,i,
            outputs[i]->name,bytes,(unsigned long long)llama_dsv41_hash_tensor(outputs[i],0,bytes,scratch));
    }
    std::fflush(stderr);
}
