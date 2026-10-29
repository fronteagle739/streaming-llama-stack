#include "llama-dsv41-tail-hint.h"
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <future>
#include <memory>
using namespace llama_dsv41_tail_hint;
static void check(bool ok,const char*why){if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
int main(){
    const key current{22,0,7};
    std::vector<int32_t>ids;for(int i=0;i<96;++i)ids.push_back(i);
    auto p=plan(current,ids);check(p.reads.additional==24,"bounded24 unique IDs");
    std::array<bool,384>loaded{};
    check(!p.adopt(current,loaded),"incomplete/failed transfer cannot publish");
    p.completed=true;
    for(const auto wrong:{key{23,0,7},key{22,1,7},key{22,0,8},key{22,0,0}})
        check(!p.adopt(wrong,loaded),"stage/bank/pass generation mismatch refused");
    for(auto x:loaded)check(!x,"failed adoption leaves demand state empty");
    check(p.adopt(current,loaded),"matching completed generation adopted");
    auto demand=llama_dsv41_prefill_routes::plan(loaded,{0,1,23,24,25,383});
    check(demand.additional==3&&demand.missing[24]&&demand.missing[25]&&demand.missing[383],"all unpredicted actual IDs fetched");
    llama_dsv41_prefill_routes::commit(loaded,demand);
    check(llama_dsv41_prefill_routes::plan(loaded,{0,1,23,24,25,383}).additional==0,"actual misses become hits only after completion");
    loaded.fill(false);check(!p.adopt({22,0,9},loaded),"restart never adopts old generation");
    auto duplicate=plan(current,{3,3,3,3,383,383});check(duplicate.reads.additional==2,"duplicate predictions deduplicated");
    for(const auto bad:{key{21,0,1},key{41,0,1},key{22,2,1},key{22,0,0}}){
        bool refused=false;try{plan(bad,ids);}catch(const std::runtime_error&){refused=true;}check(refused,"invalid target refused");
    }
    for(const auto bad:std::vector<std::vector<int32_t>>{{},{0},{0,1,2,3,4,-1},{0,1,2,3,4,384}}){
        bool refused=false;try{plan(current,bad);}catch(const std::runtime_error&){refused=true;}check(refused,"invalid routing refused");
    }
    // Exercise the same worker owner used by the provider. finish publishes
    // completion, propagates errors and is idempotent; destruction joins even
    // when the caller abandons the result during another failure.
    {
        job successful;successful.prediction=plan(current,{1,2,3,4,5,6});
        successful.start([](job&j){j.bytes=123;j.fence=7;});successful.finish();successful.finish();
        check(successful.prediction.completed&&successful.bytes==123,"completed worker joined and published");
        bool refused=false;try{successful.start([](job&){});}catch(const std::runtime_error&){refused=true;}
        check(refused,"only one bounded worker per job");
    }
    {
        job failed;failed.prediction=plan(current,{1,2,3,4,5,6});
        failed.start([](job&){throw std::runtime_error("injected storage failure");});
        bool refused=false;try{failed.finish();}catch(const std::runtime_error&){refused=true;}
        check(refused&&!failed.prediction.completed,"worker failure propagates without publication");
        check(!failed.prediction.adopt(current,loaded),"failed worker cannot become demand hits");
    }
    {
        std::atomic<bool>done{false};std::promise<void>entered;auto ready=entered.get_future();
        auto abandoned=std::make_unique<job>();
        abandoned->start([&](job&){entered.set_value();done.store(true);});ready.get();
        abandoned.reset();check(done.load(),"retirement joins before callback state is released");
    }
    std::puts("PASS bounded exact tail hint: generation/admission, completion, retirement and authoritative demand misses");
}
