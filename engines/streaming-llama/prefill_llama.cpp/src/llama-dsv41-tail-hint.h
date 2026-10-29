#pragma once
#include "llama-dsv41-prefill-routes.h"
#include <algorithm>
#include <exception>
#include <thread>
#include <utility>

// One bounded prediction. Predictions never replace actual routing: the demand
// planner still fetches every ID not present in a successfully adopted bitmap.
namespace llama_dsv41_tail_hint {
struct key {
    uint32_t stage=0,bank=0;
    uint64_t generation=0;
};
inline bool matches(const key&a,const key&b){
    return a.generation&&a.stage==b.stage&&a.bank==b.bank&&a.generation==b.generation;
}
struct prediction {
    key target;
    llama_dsv41_prefill_routes::reads reads;
    bool completed=false;
    bool adopt(const key&current,std::array<bool,384>&loaded)const{
        if(!completed||!matches(target,current))return false;
        llama_dsv41_prefill_routes::commit(loaded,reads);return true;
    }
};
inline prediction plan(key target,const std::vector<int32_t>&ids){
    if(target.stage<22||target.stage>40||target.bank>1||!target.generation)
        throw std::runtime_error("Tail hint requires a live next-layer bank generation");
    // Four complete routing rows, at most24 experts /72 extent ranges. Never
    // copy the complete prompt's routing array or allocate a weight buffer.
    if(ids.empty()||ids.size()%6)throw std::runtime_error("Tail hint requires complete routing rows");
    const std::vector<int32_t>bounded(ids.begin(),ids.begin()+std::min<size_t>(24,ids.size()));
    prediction out;out.target=target;
    out.reads=llama_dsv41_prefill_routes::plan({},bounded);return out;
}
// Own the completion boundary used by release, restart and destruction. The
// callback owns no provider lock; only finish() permits failure/publication to
// be observed. Destruction joins even when another request failure is unwinding.
struct job {
    llama_dsv41_tail_hint::prediction prediction;
    uint64_t fence=0,bytes=0;double milliseconds=0;
    double begin_epoch_ms=0,end_epoch_ms=0,exposed_join_ms=0;
    std::exception_ptr failure;
    job()=default;
    job(const job&)=delete;
    job&operator=(const job&)=delete;
    ~job(){join();}
    template<class F>void start(F fn){
        if(worker.joinable()||started)throw std::runtime_error("Tail hint job started twice");
        started=true;
        worker=std::thread([this,fn=std::move(fn)]()mutable{
            try{fn(*this);this->prediction.completed=true;}
            catch(...){failure=std::current_exception();}
        });
    }
    void finish(){join();if(failure)std::rethrow_exception(failure);}
private:
    bool started=false;
    std::thread worker;
    void join(){if(worker.joinable())worker.join();}
};
}
