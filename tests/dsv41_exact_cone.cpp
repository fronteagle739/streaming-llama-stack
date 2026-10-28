// Independent causal-window oracle: exact dependency coverage, not GPU kernel
// qualification. Shared compressed keys/candidates are immutable layer20 state.
#include "llama-dsv41-yoco.h"
#include "llama-dsv41-prefill-plan.h"
#include <array>
#include <deque>
#include <vector>
#include <cstdio>
#include <stdexcept>
namespace y=llama_dsv41_yoco;
using row=std::array<uint64_t,4>;
static void check(bool ok){if(!ok)throw std::runtime_error("cone oracle mismatch");}
struct result {std::vector<row> hidden;std::array<std::deque<uint64_t>,19> raw;};
static result run(uint32_t n,uint32_t keep,bool image){
    result out;out.hidden.resize(n);std::vector<uint64_t> compressed(n);
    for(uint32_t i=0;i<n;++i){for(uint32_t c=0;c<4;++c)out.hidden[i][c]=(i*17+c*23+1)%1009;
        if(image&&(i==n/2||i==n-130||i==n-2500))out.hidden[i]={81,7,930,256};
        for(auto v:out.hidden[i])compressed[i]+=v;}
    const uint32_t first=keep&&n>keep?n-keep:0;
    for(uint32_t layer=21;layer<40;++layer){uint64_t sum=0;auto&ring=out.raw[layer-21];
        for(uint32_t tile=first;tile<n;tile+=127){ // deliberately ragged/non-window aligned
            for(uint32_t i=tile;i<std::min(n,tile+127);++i){
                uint64_t key=0;for(auto v:out.hidden[i])key+=v;
                ring.push_back(key);sum+=key;if(ring.size()>128){sum-=ring.front();ring.pop_front();}
                const uint64_t selected=compressed[(uint64_t(i)*37+layer)%(i+1)];
                for(auto&v:out.hidden[i])v=(v+sum+selected+layer)%104729;
            }
        }
    }
    return out;
}
int main(){
    check(y::exact_keep==2541);
    for(uint32_t n:{128u,2541u,2542u,4097u,8193u})for(bool image:{false,true}){
        auto full=run(n,0,image),cone=run(n,y::exact_keep,image);
        check(full.raw==cone.raw);
        for(uint32_t i=n-65;i<n;++i)check(full.hidden[i]==cone.hidden[i]);
        auto p=y::schedule(y::exact_keep,y::operation::prefill,image,0,n,0,n);
        check(!p.approximate&&p.layers==40&&p.first==(n>2541?n-2541:0));
        check(p.exact_cone==(n>2541));
    }
    check(run(8193,0,false).hidden.back()!=run(8193,140,false).hidden.back());
    for(uint32_t chunks:{1u,2u,3u,4u}){
        auto extents=llama_dsv41_prefill::plan_sweeps(400000,chunks*49152+12288,chunks*49152,chunks*49152+12288);
        uint32_t encoder=0,decoder=0;
        for(const auto&e:extents){auto p=y::schedule(2541,y::operation::prefill,true,0,400000,e.first,e.first+e.count);
            encoder+=e.count;decoder+=p.encoder_only?0:e.count-p.first;
            check(p.encoder_only?p.layers==21:p.layers==40);
        }
        check(encoder==400000&&decoder==2541);
    }
    check(llama_dsv41_prefill::tile_count(43459,46000,4096,45056)==1597);
    check(llama_dsv41_prefill::tile_count(45056,46000,4096,45056)==944);
    check(!y::may_export(2541)&&!y::may_rewind(2541));
    for(uint32_t n:{2542u,4096u,8192u,49555u}){
        auto p=y::schedule(2541,y::operation::prefill,false,0,n,0,n);y::align_exact(p,4096);
        check(p.first%4096==0&&p.first<=n-2541&&n-p.first<2541+4096);
    }
    bool refused=false;try{y::schedule(2541,y::operation::prefill,false,0,100000,0,99000);}catch(...){refused=true;}
    check(refused);
    std::puts("PASS exact cone: final128 windows at every replay layer, final65 hidden rows, image rows, grouped passes and spill boundary");
}
