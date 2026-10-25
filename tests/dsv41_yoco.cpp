#include "llama-dsv41-yoco.h"
#include "llama-dsv41-plan.h"
#include "llama-dsv41-prefill-plan.h"
#include "llama-dsv41-dspark-capture.h"
#include "llama-dsv41-carry.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace y=llama_dsv41_yoco;
static unsigned checks=0;
static void expect(bool ok){++checks;if(!ok){std::fprintf(stderr,"YOCO check %u failed\n",checks);std::exit(1);}}
template<class F>static void refuses(F f){bool failed=false;try{f();}catch(...){failed=true;}expect(failed);}
static void seed(llama_dsv41_carry_store&carry,const llama_dsv41_carry_store::config&cfg){
    std::string error;expect(carry.create(cfg,error));
    for(uint32_t start=0;start<cfg.tokens;start+=32){
        llama_dsv41_carry_store::tile t;t.start=start;t.count=std::min(32u,cfg.tokens-start);
        t.residual.assign(size_t(t.count)*5120*4,0.f);t.previous_pre.assign(t.count*4,0.f);
        expect(carry.seed(t,error));
    }
    expect(carry.finish_seed(error));
}
static void run_layer(llama_dsv41_carry_store&carry,uint32_t layer,uint32_t first,uint32_t end){
    std::string error;expect(carry.begin_layer(layer,0,0,error));
    for(uint32_t start=first;start<end;){
        const uint32_t n=std::min(32-start%32,end-start);
        llama_dsv41_carry_store::tile t;expect(carry.read_tile(start,n,t,error));
        expect(carry.write_tile(t,error));start+=n;
    }
    expect(carry.commit_layer(error));
}
int main(){
    for(uint32_t n:{1u,140u,141u,43000u,103189u,999423u}){
        auto extents=llama_dsv41_prefill::plan_sweeps(n,61440,49152,61440);
        expect(!extents.empty());
        uint32_t encoder=0,decoder=0,passes=0;
        for(const auto&e:extents){
            const auto off=y::schedule(0,y::operation::prefill,false,0,n,e.first,e.first+e.count);
            expect(off.layers==40&&off.first==0&&off.count==e.count&&!off.approximate&&!off.encoder_only);
            for(uint32_t l=0;l<40;++l)expect(y::first_row(off,l)==0);
            const auto on=y::schedule(140,y::operation::prefill,false,0,n,e.first,e.first+e.count);
            encoder+=on.count;
            if(!on.encoder_only){decoder+=on.count-on.first;++passes;}
            expect(y::first_row(on,20)==0);
            if(on.encoder_only)expect(on.layers==21);
        }
        expect(encoder==n&&decoder==std::min(n,140u)&&passes==1);
        for(auto op:{y::operation::decode,y::operation::verify,y::operation::append_decode,y::operation::all_logits}){
            const auto p=y::schedule(140,op,false,0,n,0,n);
            expect(p.first==0&&p.layers==40&&!p.approximate);
        }
        const auto image=y::schedule(140,y::operation::prefill,true,0,n,0,n);
        expect(image.layers==40&&image.first==0&&!image.approximate);
    }
    // A logical append retains absolute positions and only its own tail.
    const auto append=y::schedule(140,y::operation::prefill,false,50000,153189,148304,153189);
    expect(append.first==4745&&append.count-append.first==140);
    expect(!y::may_export(140)&&!y::may_rewind(140)&&y::may_export(0)&&y::may_rewind(0));
    refuses([]{y::validate(139);});refuses([]{y::validate(141);});
    refuses([]{y::schedule(140,y::operation::prefill,false,0,50000,0,49900);});
    // Every visible raw key belongs to the replay suffix, while absolute
    // positions, full compressed publication and visibility are untouched.
    for(uint32_t begin:{1u,128u,42860u,103049u}){
        llama_dsv41_tile_plan p(begin,140,1048576,1,4096);
        const auto positions=p.positions,visible=p.compressed_visible,publication=p.compressed_positions;
        p.restrict_raw(begin,4096);
        expect(p.raw_start==0&&p.history_rows==0&&p.raw_rows==128&&p.history_ids.empty());
        expect(p.positions==positions&&p.compressed_visible==visible&&p.compressed_positions==publication);
        for(uint32_t row=0;row<140;++row)for(uint32_t i=0;i<p.raw_rows;++i){
            const auto at=size_t(row)*p.raw_rows+i;
            if(std::isfinite(p.raw_mask[at]))expect(p.raw_ids[at]>=0&&uint32_t(p.raw_ids[at])<=row);
        }
    }
    llama_dsv41_tile_plan second(103081,32,1048576,1,4096);second.restrict_raw(103049,4096);
    expect(second.history_rows==32&&second.history_ids.front()==103049%128&&second.raw_start==32);
    llama_dsv41_dspark_capture::completion captures;
    for(uint32_t l=0;l<3;++l){auto t=captures.begin_replay(l,103049,140);expect(!captures.ready(103189));captures.commit(t);}
    expect(captures.ready(103189));
    refuses([&]{captures.begin_replay(0,103190,127);});
    {
        llama_dsv41_carry_store carry;llama_dsv41_carry_store::config cfg;
        cfg.resident_ram=true;cfg.tokens=256;cfg.max_tile_tokens=512;cfg.admitted_host_bytes=256ull<<20;cfg.yoco_first=116;
        seed(carry,cfg);
        for(uint32_t l=0;l<40;++l){run_layer(carry,l,l>20?116:0,256);expect(!carry.frontier_ready());}
        std::string error;llama_dsv41_carry_store::tile last;
        expect(carry.read_final_tile(255,1,last,error));expect(carry.certify_frontier(error));expect(carry.frontier_ready());
    }
    {
        llama_dsv41_carry_store carry;llama_dsv41_carry_store::config cfg;
        cfg.resident_ram=true;cfg.tokens=256;cfg.max_tile_tokens=512;cfg.admitted_host_bytes=256ull<<20;cfg.yoco_first=116;
        seed(carry,cfg);
        for(uint32_t l=0;l<40;++l)run_layer(carry,l,l>20?116:0,256);
        std::string error;llama_dsv41_carry_store::tile stale;
        expect(!carry.read_final_tile(0,1,stale,error));expect(carry.failed());expect(!carry.certify_frontier(error));
    }
    {
        // An internal encoder endpoint can retire its RAM lease, but cannot
        // certify a head or grant completed decoder-layer credit.
        llama_dsv41_carry_store carry;llama_dsv41_carry_store::config cfg;
        cfg.resident_ram=true;cfg.direct_ram_tiles=true;cfg.tokens=256;cfg.max_tile_tokens=512;cfg.admitted_host_bytes=256ull<<20;
        cfg.shared_ram=llama_dsv41_carry_store::reserve_ram(cfg.tokens,cfg.admitted_host_bytes,cfg.max_tile_tokens);
        seed(carry,cfg);
        for(uint32_t l=0;l<21;++l)run_layer(carry,l,0,256);
        expect(carry.completed_layers()==21&&!carry.frontier_ready());
        std::string error;expect(carry.retire_yoco_encoder(error));expect(!carry.frontier_ready());
    }
    std::printf("PASS YOCO %u checks\n",checks);
}
