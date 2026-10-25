#pragma once

#include "llama-dsv41-wire.h"
#include "boundary-capture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// LLAMADECODE incremental owner verification (producer-local; the wire
// owner_sha256 values are unchanged).
//
// Each owner digest is, as before, the plain SHA-256 of that span's canonical
// F32 bytes. compressed/index spans are append-only: row i is final once the
// frontier passes it and only the rows [old_live, new_live) are ever added
// (make_delta_plan writes exactly that tail). The memo keeps, per
// compressed/index owner, the SHA-256 running state after its first L bytes.
// Extending that state over the appended bytes [L, bytes) and finalizing a
// copy is exactly SHA-256 of the whole span: the same value a full re-hash and
// the Mac receiver compute, so integrity of everything hashed is unchanged and
// the cost is O(appended rows). Ring/pool/DSpark owners (bounded, <=128 rows,
// rewritten in place) are re-hashed in full every time.
//
// Trust boundary: rows below L are not re-read. They are covered by the
// digest verified when they were hashed; any path that can rewrite them
// (rewind, restore/replace, clear, suspend, failure) must invalidate().
// LLAMADECODE_OWNER_VERIFY=full keeps the previous full re-hash everywhere.
//
// Speed (digest values unchanged): the hash is dsv41_boundary::sha256_native
// (Windows CNG; SHA extensions where present). Spans are independent, so up to
// max_workers workers hash them in parallel, largest remaining work first; the
// output stays in plan.spans order. Each worker is a reader thread and a hasher
// thread over two `chunk`-byte slots (double buffering: chunk i+1 is read while
// chunk i is hashed). Staging = workers * 2 * chunk bytes (32 MiB at 8 workers),
// caller-owned pinned host memory when set_staging() was given, else pageable.
// The reader must be safe to call from several threads at once.
namespace llama_dsv41 {

inline bool owner_verify_incremental(){
    static const bool on=[]{const char*e=std::getenv("LLAMADECODE_OWNER_VERIFY");
        return !(e&&std::strcmp(e,"full")==0);}();
    return on;
}

class owner_digest_memo {
public:
    using reader=std::function<bool(const state_span&,uint64_t,void*,size_t)>;
    void invalidate(){valid=false;for(auto&e:entries)e.have=false;}
    bool ready()const{return valid;}
    uint32_t position()const{return memo_position;}
    // Digests in plan.spans order. incremental=false re-hashes every byte
    // (and refreshes the memo). hashed_bytes reports the bytes actually read.
    static constexpr size_t chunk=2u<<20;      // bytes per read slot
    static constexpr unsigned max_workers=8;
    static constexpr size_t staging_bytes(unsigned workers=max_workers){return size_t(workers)*2*chunk;}
    // Caller-owned staging (e.g. pinned host memory); the worker count is capped
    // to staging/(2*chunk). Without it each pass uses its own pageable buffer.
    void set_staging(unsigned char*p,size_t bytes){staging=p;staging_size=bytes;}
    // Upper bound of the pageable fallback buffer (no staging). Default: the
    // full staging_bytes(); the cache lowers it to 2 MiB until admission.
    void set_fallback_limit(size_t bytes){fallback_limit=bytes;}
    std::vector<std::string> digests(const wire_plan&plan,const reader&read,bool incremental,uint64_t*hashed_bytes=nullptr){
        const auto&f=plan.frontier;
        const bool use=incremental&&valid&&f.context==memo_context&&f.mode==memo_mode&&f.position>=memo_position;
        valid=false; // only a fully completed pass re-arms the memo
        const size_t n=plan.spans.size();
        std::vector<std::string> out(n);
        std::vector<dsv41_boundary::sha256_native*> hash(n,nullptr);
        std::vector<dsv41_boundary::sha256_native> local;local.reserve(n);
        std::vector<uint64_t> from(n,0);
        for(size_t i=0;i<n;++i){
            const auto&s=plan.spans[i];const int k=slot(s);
            if(k>=0){
                auto&e=entries[size_t(k)];
                if(use&&e.have&&e.bytes<=s.bytes)from[i]=e.bytes;else e.hash=dsv41_boundary::sha256_native{};
                e.have=false;hash[i]=&e.hash; // re-armed only when this pass completes
            }else{local.emplace_back();hash[i]=&local.back();}
        }
        std::vector<size_t> order(n);std::iota(order.begin(),order.end(),size_t(0));
        auto work=[&](size_t i){return plan.spans[i].bytes-from[i];};
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return work(a)>work(b);});
        uint64_t biggest=n?work(order[0]):0;
        unsigned workers=std::max(1u,std::thread::hardware_concurrency()/2);
        workers=unsigned(std::min<size_t>({size_t(workers),size_t(max_workers),std::max<size_t>(n,1)}));
        size_t slot_bytes=chunk;std::vector<unsigned char> own;unsigned char*base=staging;
        if(base&&staging_size>=2*chunk)workers=unsigned(std::min<size_t>(workers,staging_size/(2*chunk)));
        else{slot_bytes=size_t(std::min<uint64_t>(chunk,std::max<uint64_t>(64,(biggest+63)/64*64)));
            slot_bytes=std::max<size_t>(64,std::min(slot_bytes,fallback_limit/2/64*64));
            workers=unsigned(std::max<size_t>(1,std::min<size_t>(workers,fallback_limit/(2*slot_bytes))));
            own.resize(size_t(workers)*2*slot_bytes);base=own.data();}
        struct pipe{std::mutex m;std::condition_variable cv;struct item{bool full=false,last=false;size_t task=0,bytes=0;}slot[2];};
        std::vector<pipe> pipes(workers);
        std::atomic<size_t> next{0};std::atomic<bool> failed{false};std::atomic<uint64_t> hashed{0};
        std::mutex failure_lock;std::string failure;
        auto fail=[&](const std::string&why){{std::lock_guard<std::mutex>l(failure_lock);if(failure.empty())failure=why;}
            failed=true;for(auto&p:pipes){std::lock_guard<std::mutex>l(p.m);p.cv.notify_all();}};
        constexpr size_t end_marker=~size_t(0);
        auto produce=[&](pipe&p,unsigned char*buf){
            try{
                size_t w=0;
                auto put=[&](size_t task,size_t bytes,bool last){
                    {std::lock_guard<std::mutex>l(p.m);auto&x=p.slot[w%2];x.full=true;x.last=last;x.task=task;x.bytes=bytes;}
                    p.cv.notify_all();++w;};
                auto wait_free=[&]{std::unique_lock<std::mutex>l(p.m);p.cv.wait(l,[&]{return !p.slot[w%2].full||failed;});return !failed;};
                for(;;){
                    const size_t q=next.fetch_add(1);if(q>=n)break;
                    const size_t i=order[q];const auto&s=plan.spans[i];uint64_t off=from[i];
                    do{
                        if(!wait_free())return;
                        const size_t m=size_t(std::min<uint64_t>(slot_bytes,s.bytes-off));
                        if(m&&!read(s,off,buf+(w%2)*slot_bytes,m)){fail("Owner digest span read failed");return;}
                        off+=m;put(i,m,off>=s.bytes);
                    }while(off<s.bytes);
                }
                if(wait_free())put(end_marker,0,true);
            }catch(const std::exception&e){fail(e.what());}catch(...){fail("Owner digest reader failed");}
        };
        auto consume=[&](pipe&p,unsigned char*buf){
            try{
                for(size_t r=0;;++r){
                    pipe::item x;
                    {std::unique_lock<std::mutex>l(p.m);p.cv.wait(l,[&]{return p.slot[r%2].full||failed;});if(failed)return;x=p.slot[r%2];}
                    if(x.task==end_marker)return;
                    if(x.bytes){hash[x.task]->update(buf+(r%2)*slot_bytes,x.bytes);hashed+=x.bytes;}
                    if(x.last)out[x.task]=hash[x.task]->hex();
                    {std::lock_guard<std::mutex>l(p.m);p.slot[r%2].full=false;}
                    p.cv.notify_all();
                }
            }catch(const std::exception&e){fail(e.what());}catch(...){fail("Owner digest hash failed");}
        };
        {
            std::vector<std::thread> threads;threads.reserve(size_t(workers)*2);
            for(unsigned t=0;t<workers;++t){
                unsigned char*buf=base+size_t(t)*2*slot_bytes;
                threads.emplace_back(produce,std::ref(pipes[t]),buf);threads.emplace_back(consume,std::ref(pipes[t]),buf);
            }
            for(auto&t:threads)t.join();
        }
        if(failed)throw std::runtime_error(failure.empty()?"Owner digest span read failed":failure);
        for(size_t i=0;i<n;++i){const int k=slot(plan.spans[i]);if(k>=0){entries[size_t(k)].have=true;entries[size_t(k)].bytes=plan.spans[i].bytes;}}
        memo_context=f.context;memo_mode=f.mode;memo_position=f.position;valid=true;
        last_workers=workers;
        if(hashed_bytes)*hashed_bytes=hashed;
        return out;
    }
    unsigned workers_used()const{return last_workers;}
private:
    unsigned char*staging=nullptr;size_t staging_size=0,fallback_limit=staging_bytes();unsigned last_workers=0;
    static int slot(const state_span&s){
        if(s.index>=4)return -1;
        if(s.kind==state_kind::compressed)return int(s.index);
        if(s.kind==state_kind::index)return 4+int(s.index);
        return -1;
    }
    struct entry{bool have=false;uint64_t bytes=0;dsv41_boundary::sha256_native hash;};
    std::array<entry,8> entries{};
    bool valid=false;
    uint32_t memo_context=0,memo_mode=0,memo_position=0;
};

} // namespace llama_dsv41
