#include "llama-dsv41-prefill-routes.h"
#include <cassert>
#include <iostream>
using namespace llama_dsv41_prefill_routes;
int main(){
    std::array<bool,384>loaded{};
    auto first=plan(loaded,{0,1,2,2,383,383});
    assert(first.active==4&&first.additional==4&&first.ranges.size()==2);
    assert(first.ranges[0].first==0&&first.ranges[0].count==3);
    assert(first.ranges[1].first==383&&first.ranges[1].count==1);
    // Planning (and a failed fetch) cannot publish resident bytes.
    assert(!loaded[0]&&!loaded[383]);commit(loaded,first);
    auto second=plan(loaded,{0,1,2,3,4,383});
    assert(second.active==6&&second.additional==2&&second.ranges.size()==1);
    assert(second.ranges[0].first==3&&second.ranges[0].count==2);commit(loaded,second);
    auto hit=plan(loaded,{0,1,2,3,4,383});assert(hit.additional==0&&hit.ranges.empty());
    // Independent banks/layers must not inherit the prior layer's bitmap.
    loaded.fill(false);assert(plan(loaded,{0,1,2,3,4,383}).additional==6);
    for(const auto&bad:std::vector<std::vector<int32_t>>{{},{0},{0,1,2,3,4,-1},{0,1,2,3,4,384}}){
        bool failed=false;try{plan(loaded,bad);}catch(const std::runtime_error&){failed=true;}assert(failed);
        for(auto present:loaded)assert(!present);
    }
    std::cout<<"Routed prefill: dedup, ranges, hits, bank reset and failure publication pass\n";
}
