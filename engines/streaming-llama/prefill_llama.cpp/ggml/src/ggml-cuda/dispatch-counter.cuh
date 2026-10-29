#pragma once
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>
using ggml_cuda_dispatch_deltas=std::vector<std::pair<std::atomic<uint64_t>*,uint64_t>>;
extern thread_local ggml_cuda_dispatch_deltas *ggml_cuda_capturing_dispatch;
// Queue witnesses describe submitted work, including successful graph replays.
// Capture records actual dispatcher increments; replay applies that same ledger
// only after cudaGraphLaunch succeeds. This does not synthesize kernel coverage.
struct ggml_cuda_dispatch_counter : std::atomic<uint64_t> {
    explicit ggml_cuda_dispatch_counter(uint64_t value):std::atomic<uint64_t>(value) {}
    uint64_t fetch_add(uint64_t value,std::memory_order order=std::memory_order_seq_cst) {
        if(ggml_cuda_capturing_dispatch) {
            auto &deltas=*ggml_cuda_capturing_dispatch;
            auto it=deltas.begin();
            for(;it!=deltas.end();++it)if(it->first==this){it->second+=value;break;}
            if(it==deltas.end())deltas.emplace_back(this,value);
        }
        return std::atomic<uint64_t>::fetch_add(value,order);
    }
};
