#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-engram-plan.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
int main(){
    namespace ep=llama_dsv41_engram_plan;
    std::vector<uint8_t> file(1<<20);std::mt19937 rng(13);
    for(auto&b:file)b=uint8_t(rng());
    const uint64_t base=1031;
    std::vector<uint32_t> ids;
    for(uint32_t i=0;i<3500;++i)if(rng()%7==0)ids.push_back(i);
    ids.insert(ids.begin()+1,ids[1]);
    std::vector<uint8_t> hits((ids.size()+7)/8);
    for(size_t i=0;i<ids.size();++i)if(i%3==0)hits[i/8]|=1u<<(i%8);
    uint64_t totals[2]{};
    for(size_t arm=0;arm<2;++arm){
        const uint32_t alignment=arm?512:4096;
        const auto plan=ep::build(ids.data(),ids.size(),hits.data(),base,264,65536,alignment);
        size_t row=0;
        for(const auto&e:plan.extents){
            assert(e.file_offset%alignment==0&&e.bytes%alignment==0&&e.bytes<=65536);
            assert(e.file_offset+e.bytes<=file.size());
            for(uint32_t n=0;n<e.rows;++n,++row){
                const auto&r=plan.rows[row];assert(!ep::is_hit(hits.data(),r.index));
                assert(r.offset+264<=e.bytes);
                assert(!std::memcmp(file.data()+e.file_offset+r.offset,file.data()+base+uint64_t(ids[r.index])*264,264));
            }
        }
        assert(row==plan.rows.size());totals[arm]=plan.bytes;
    }
    assert(totals[1]<totals[0]);
    const uint32_t crossing=1;const auto cross=ep::build(&crossing,1,nullptr,0,264,65536,512);
    assert(cross.extents.size()==1&&cross.extents[0].bytes==1024);
    const uint8_t all_hit=1;
    assert(ep::build(&crossing,1,&all_hit,0,264,65536,512).extents.empty());
    for(uint32_t bad:{0u,256u,1024u,8192u}){
        bool threw=false;
        try{ep::build(&crossing,1,nullptr,0,264,65536,bad);}
        catch(const std::invalid_argument&){threw=true;}
        assert(threw);
    }
    std::cout<<"Engram exact sector recovery, crossing row, hits, duplicates PASS; 4K_bytes="<<totals[0]<<" 512B_bytes="<<totals[1]<<"\n";
}
