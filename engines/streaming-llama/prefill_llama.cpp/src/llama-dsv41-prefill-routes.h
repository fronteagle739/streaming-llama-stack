#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llama_dsv41_prefill_routes {
struct range { uint32_t first,count; };
struct reads {
    std::array<bool,384> missing{};
    std::vector<range> ranges;
    uint32_t active=0,additional=0;
};
inline reads plan(const std::array<bool,384>&loaded,const std::vector<int32_t>&ids){
    reads out;
    if(ids.empty()||ids.size()%6)throw std::runtime_error("Prefill routing requires six IDs per row");
    for(auto id:ids){if(id<0||id>=384)throw std::runtime_error("Prefill routed expert ID out of range");out.missing[size_t(id)]=true;}
    for(size_t i=0;i<384;++i){out.active+=out.missing[i];out.missing[i]=out.missing[i]&&!loaded[i];out.additional+=out.missing[i];}
    for(uint32_t first=0;first<384;){if(!out.missing[first]){++first;continue;}uint32_t end=first+1;while(end<384&&out.missing[end])++end;
        out.ranges.push_back({first,end-first});first=end;}
    return out;
}
inline void commit(std::array<bool,384>&loaded,const reads&read){
    for(size_t i=0;i<384;++i)loaded[i]=loaded[i]||read.missing[i];
}
}
