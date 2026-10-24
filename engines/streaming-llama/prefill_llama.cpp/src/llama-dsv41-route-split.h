#pragma once
// LLAMADECODE B1: stable partition of a layer graph into nodes independent of
// the router's selected experts, then the nodes that depend on them. Applied
// BEFORE allocation so the lifetime allocator plans the order that executes.
// Topological order holds inside both parts (a node independent of the router
// cannot consume a dependent one).
#include "ggml.h"
#include "../ggml/src/ggml-impl.h"
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

// Returns the count of router-independent nodes (the split point), or -1 with why.
inline int llama_dsv41_route_partition(ggml_cgraph *graph,const char *router_name,std::string &why){
    const int nodes=ggml_graph_n_nodes(graph);ggml_tensor *router=nullptr;
    for(int i=0;i<nodes;++i){auto *n=ggml_graph_node(graph,i);
        if(!std::strcmp(ggml_get_name(n),router_name)){if(router){why="Routed decode split found two routers";return -1;}router=n;}}
    if(!router){why="Routed decode split found no router";return -1;}
    std::unordered_set<const ggml_tensor*>dependent;std::vector<ggml_tensor*>first,second;
    for(int i=0;i<nodes;++i){auto *n=ggml_graph_node(graph,i);bool d=false;
        for(const auto *src:n->src)if(src&&(src==router||dependent.count(src)))d=true;
        if(n->view_src&&(n->view_src==router||dependent.count(n->view_src)))d=true;
        if(d){dependent.insert(n);second.push_back(n);}else first.push_back(n);}
    if(second.empty()){why="Routed decode split has no routed consumers";return -1;}
    int at=0;for(auto *n:first)graph->nodes[at++]=n;for(auto *n:second)graph->nodes[at++]=n;
    return int(first.size());
}
