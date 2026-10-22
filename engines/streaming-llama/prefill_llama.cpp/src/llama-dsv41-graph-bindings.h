#pragma once
#include "../ggml/include/ggml.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
// Capture BEFORE gallocr assigns scratch addresses. Restore that exact state
// before re-planning: persistent aliases remain bound, scratch tensors unbound.
class llama_dsv41_graph_bindings {
    struct saved {ggml_tensor *tensor;ggml_tensor initial;};
    std::vector<saved> tensors;
    uint32_t original_start=0,rows=0;
public:
    void capture(const std::vector<ggml_tensor*> &all,uint32_t start,uint32_t count){
        static_assert(std::is_trivially_copyable<ggml_tensor>::value,"GGML tensor snapshot requires plain metadata");
        if(!tensors.empty()||all.empty()||all.size()>16384||!count||start>=1048576||count>1048576-start)
            throw std::runtime_error("Invalid initial layer-graph binding snapshot");
        original_start=start;rows=count;tensors.reserve(all.size());
        for(auto *t:all){if(!t)throw std::runtime_error("Null graph tensor snapshot");tensors.push_back({t,*t});
            if(t->op==GGML_OP_DSV41_RAW_COMMIT){uint32_t raw=0;std::memcpy(&raw,t->op_params,4);
                if(!t->src[0]||raw<start||raw>=uint64_t(start)+count||t->src[0]->ne[1]<=0||
                   t->src[0]->ne[1]>32||uint64_t(raw)+t->src[0]->ne[1]>uint64_t(start)+count)
                    throw std::runtime_error("Raw-ring commit escaped graph tile");}}
    }
    void restore(uint32_t start,uint32_t count){
        if(tensors.empty()||count!=rows||start>=1048576||count>1048576-start)
            throw std::runtime_error("Layer graph binding restore changed tile geometry");
        for(auto &s:tensors)*s.tensor=s.initial;
        for(auto &s:tensors)if(s.tensor->op==GGML_OP_DSV41_RAW_COMMIT){
            uint32_t raw=0;std::memcpy(&raw,s.initial.op_params,4);raw=start+(raw-original_start);
            std::memcpy(s.tensor->op_params,&raw,4);
        }
    }
    uint64_t bytes()const{return tensors.capacity()*sizeof(saved);}
};
