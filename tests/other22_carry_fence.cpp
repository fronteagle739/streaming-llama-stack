#define DS4PERF_DIRECT_CARRY_TEST
#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-carry.h"
#include <cstdio>
#include <cstdlib>
struct ds4perf_direct_carry_fixture {
    static auto completion(const llama_dsv41_carry_store::ram_lease &lease,std::shared_ptr<bool> fence){
        llama_dsv41_carry_store::ram_completion out(lease,true,true);out.deferred=std::move(fence);return out;
    }
};
static void require(bool x,const char *why){if(!x){std::fprintf(stderr,"FAIL %s\n",why);std::exit(2);}}
int main(){
    for(bool certified:{false,true}){
        llama_dsv41_carry_store store;llama_dsv41_carry_store::config cfg;
        cfg.tokens=2;cfg.max_tile_tokens=32;cfg.resident_ram=true;cfg.direct_ram_tiles=true;cfg.admitted_host_bytes=16ull<<20;
        std::string error;require(store.create(cfg,error),error.c_str());
        llama_dsv41_carry_store::tile seed;seed.start=0;seed.count=2;seed.residual.resize(2*20480);seed.previous_pre.resize(8);
        require(store.seed(seed,error),error.c_str());require(store.finish_seed(error),error.c_str());
        require(store.begin_layer(0,0,0,error),error.c_str());
        auto fence=std::make_shared<bool>(false);
        for(unsigned i=0;i<2;++i){llama_dsv41_carry_store::ram_lease lease;
            require(store.acquire_ram_tile(i,1,lease,error),error.c_str());
            auto done=ds4perf_direct_carry_fixture::completion(lease,fence);
            require(store.complete_ram_tile(lease,done,error),error.c_str());}
        *fence=certified;
        require(store.commit_layer(error)==certified,"pending GPU tiles must never publish a carry layer");
        if(!certified)require(error.find("uncertified")!=std::string::npos,"specific missing-certificate error");
    }
    std::puts("PASS uncertified CUDA carry rejected; certified layer commits");
}
