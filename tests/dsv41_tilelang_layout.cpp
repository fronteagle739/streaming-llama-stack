#include "ggml-dsv41-tilelang-layout.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <vector>
int main(){
    std::mt19937 rng(17);uint64_t cases=0;
    for(uint32_t start:{0u,1u,126u,127u,128u,4097u,100000u})
    for(uint32_t queries:{1u,2u,3u,4u,5u,31u,128u,129u})
    for(bool merged:{false,true}){
        const uint32_t tiles=(queries+3)/4,stride=2048,raw_rows=queries+127,raw_min=127-std::min(start,127u),comp_rows=1025;
        std::vector<ggml_dsv41_tilelang_record> records(uint64_t(tiles)*stride);
        std::vector<uint32_t>counts(tiles);std::vector<std::set<int32_t>> expected(queries);
        for(uint32_t tile=0;tile<tiles;++tile){
            std::array<uint32_t,1025> membership{};
            for(uint32_t local=0;local<4&&tile*4+local<queries;++local){
                const uint32_t q=tile*4+local,selected=rng()%513;std::set<uint32_t>chosen;
                while(chosen.size()<selected)chosen.insert(1+rng()%1024);
                for(uint32_t id:chosen){expected[q].insert(int32_t(raw_rows+id));
                    if(merged)membership[id]|=1u<<local;
                    else records[uint64_t(tile)*stride+counts[tile]++]={int32_t(id),int32_t(1u<<local)};
                }
            }
            if(merged)for(uint32_t id=1;id<comp_rows;++id)if(membership[id])records[uint64_t(tile)*stride+counts[tile]++]={int32_t(id),int32_t(membership[id])};
        }
        for(uint32_t q=0;q<queries;++q){std::array<int32_t,640>ids{};
            assert(ggml_dsv41_tilelang_ids(ids.data(),q,queries,raw_rows,raw_min,comp_rows,records.data(),counts.data(),stride));
            // Oracle uses absolute causal positions, not the packed raw offsets.
            std::set<int32_t>want=expected[q];
            for(int64_t p=int64_t(start)+q-127;p<=int64_t(start)+q;++p)if(p>=0)want.insert(int32_t(p-int64_t(start)+127));
            std::set<int32_t>actual;for(auto id:ids)if(id>=0)assert(actual.insert(id).second);
            assert(actual==want);++cases;
        }
    }
    std::array<int32_t,640>out{};ggml_dsv41_tilelang_record record{-1,1};uint32_t count=1;
    assert(!ggml_dsv41_tilelang_ids(out.data(),0,1,128,127,1,&record,&count,1));
    record={1,1};assert(!ggml_dsv41_tilelang_ids(out.data(),0,1,128,127,1,&record,&count,1));
    record={0,1};count=2;assert(!ggml_dsv41_tilelang_ids(out.data(),0,1,128,127,1,&record,&count,1));
    std::cout<<"PASS "<<cases<<" query layouts, causal windows, merged/per-query records and invalid extents\n";
}
