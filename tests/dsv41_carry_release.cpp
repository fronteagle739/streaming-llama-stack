// CPU-only: a sealed certificate survives physical row release, and a new
// prefill reacquires the same zeroed bank without stale leases or accounting.
#include "llama-dsv41-carry.h"
#include "llama-dsv41-locked-ram.h"
#include <cstdio>
#include <stdexcept>

static void check(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    using store=llama_dsv41_carry_store;std::string error;
    const auto before=llama_dsv41_locked_ram::state().bytes;
    {
        auto bank=store::reserve_ram(1,64ull<<20,32);
        store::config cfg;cfg.tokens=1;cfg.resident_ram=true;cfg.direct_ram_tiles=true;
        cfg.shared_ram=bank;cfg.admitted_host_bytes=64ull<<20;
        store carry;check(carry.create(cfg,error),error);
        store::tile seed;seed.start=0;seed.count=1;
        seed.residual.assign(5120*4,.25f);seed.previous_pre.assign(4,.5f);
        check(carry.seed(seed,error),error);check(carry.finish_seed(error),error);
        for(uint32_t layer=0;layer<40;++layer){
            check(carry.begin_layer(layer,0,0,error),error);store::tile row;
            check(carry.read_tile(0,1,row,error),error);check(row.residual==seed.residual,"carry bytes changed");
            check(carry.write_tile(row,error),error);check(carry.commit_layer(error),error);
        }
        store::tile final;check(carry.read_final_tile(0,1,final,error),error);
        check(carry.certify_frontier(error),error);check(carry.frontier_ready(),"missing sealed certificate");
        const auto before_release=llama_dsv41_locked_ram::state().bytes;
        check(carry.release_completed_ram(error),error);bank->suspend_pages();
        check(carry.frontier_ready()&&!carry.failed(),"releasing hidden rows invalidated native frontier");
        check(bank->suspended()&&llama_dsv41_locked_ram::state().bytes+bank->size()==before_release,"physical credit was not released");
        check(carry.release_completed_ram(error),error); // idempotent
        bank->resume_pages();check(!bank->suspended(),"carry did not recommit");
        check(carry.retire_shared_ram(error),error);check(!carry.frontier_ready(),"retired certificate remained valid");
        {store next;check(next.create(cfg,error),error);check(next.seed(seed,error),error);}
        // Delete a suspended bank too: its bytes were already uncharged.
        bank->suspend_pages();cfg.shared_ram.reset();bank.reset();
    }
    check(llama_dsv41_locked_ram::state().bytes==before,"suspended deletion underflowed RAM accounting");
    {store incomplete;check(!incomplete.release_completed_ram(error),"uncertified carry released pages");}
    std::puts("PASS carry certificate, 40 byte-preserving layers, release/recommit/reseed, suspended destruction, incomplete refusal");
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
