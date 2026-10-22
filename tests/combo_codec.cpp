#include "llama-dsv41-delta.h"
#include "dsv41-native-c-artifact.h"
#include <cassert>
#include <cstdio>
#include <cstring>
extern "C" int combo_plan(uint32_t,uint32_t,uint32_t,uint32_t,dsv41_c_plan*);
extern "C" int combo_header(const dsv41_c_plan*,uint64_t,uint64_t,uint8_t*);
int main(){
    unsigned valid=0,rejected=0;
    for(uint32_t mode:{1u,3u})for(uint32_t context:{256u,262144u,1000001u,1048576u})
    for(uint32_t base:{1u,2u,3u,31u,32u,63u,64u,126u,127u,128u,129u,255u,4095u,49151u,262140u,999999u,1048574u})
    for(uint32_t added:{1u,2u,3u,31u,64u,127u,128u,129u,49151u,49152u,49153u}){
        dsv41_c_plan c{};const uint32_t target=base+added;
        bool ok=combo_plan(mode,context,base,target,&c)!=0;
        if(base>=context||target>=context||added>49152){assert(!ok);++rejected;continue;}
        assert(ok);++valid;
        llama_dsv41::wire_frontier b{context,base,mode,mode==3?base:0,mode==3?std::min(base,128u):0};
        llama_dsv41::wire_frontier t{context,target,mode,mode==3?target:0,mode==3?std::min(target,128u):0};
        llama_dsv41::delta_plan oracle;llama_dsv41::wire_plan full;std::string error;
        assert(llama_dsv41::make_delta_plan(b,t,oracle,error));
        assert(llama_dsv41::make_wire_plan(t,full,error));
        assert(c.mode==mode&&c.span_count==full.spans.size()&&c.slice_count==oracle.slices.size());
        assert(c.state_bytes==oracle.state_bytes);
        assert(c.artifact_bytes==204+(mode==3?60:52)+uint64_t(added)*4+129280*4+c.state_bytes);
        for(uint32_t i=0;i<c.span_count;++i){const auto&s=full.spans[i];
            assert(c.spans[i].kind==uint32_t(s.kind)&&c.spans[i].index==s.index&&c.spans[i].bytes==s.bytes);
            uint32_t found=999;assert(dsv41_c_find(c.spans,c.span_count,uint32_t(s.kind),s.index,&found));assert(found==i);
        }
        for(uint32_t i=0;i<c.slice_count;++i){const auto&r=oracle.slices[i];const auto&s=c.spans[c.slices[i].span];
            assert(s.kind==uint32_t(r.span.kind)&&s.index==r.span.index);
            assert(c.slices[i].offset==r.offset&&c.slices[i].bytes==r.bytes);
        }
        uint8_t header[264]{};assert(combo_header(&c,17,23,header)==(mode==3?60:52));
        assert(dsv41_c_get32(header+204+12)==mode);
        assert(dsv41_c_get32(header+204+28)==target);
    }
    for(uint32_t mode:{0u,2u,4u}){dsv41_c_plan c{};assert(!combo_plan(mode,262144,32,64,&c));++rejected;}
    dsv41_c_plan c{};assert(!combo_plan(1,262144,0,1,&c));++rejected;
    assert(!combo_plan(1,262144,64,32,&c));++rejected;
    assert(!combo_plan(1,1048577,32,64,&c));++rejected;
    std::printf("PASS: %u valid mode1/mode3 planner cases, %u refusals; owner identities and slices exact\n",valid,rejected);
}
