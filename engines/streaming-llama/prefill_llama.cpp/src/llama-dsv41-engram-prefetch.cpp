// IoRing declarations are isolated to this TU. Runtime functions are resolved
// only for an explicitly enabled cache; ordinary paths gain no Win11 import.
#ifdef _WIN32
#include "llama-dsv41-layer-log.h"
#include <sdkddkver.h>
#if !defined(NTDDI_VERSION) || NTDDI_VERSION < NTDDI_WIN10_CO
#undef NTDDI_VERSION
#define NTDDI_VERSION NTDDI_WIN10_CO
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <ioringapi.h>
#include <malloc.h>
#endif
#include "llama-dsv41-engram-prefetch.h"
#include "llama-dsv41-engram-plan.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
// LLAMADECODE_ENGRAM_EARLY=1: the early multi-ring loader. Flag off: no caller
// reaches prepare_early/load_early and the weight-stream marker is never set.
static std::atomic<bool> llama_dsv41_engram_weight_pass{false};
bool llama_dsv41_engram_early_enabled(){
    static const bool on=[]{const char*e=std::getenv("LLAMADECODE_ENGRAM_EARLY");return e&&std::strcmp(e,"1")==0;}();return on;
}
void llama_dsv41_engram_weight_stream(bool active){llama_dsv41_engram_weight_pass.store(active,std::memory_order_relaxed);}
#ifdef _WIN32
// Submit/cleanup contracts: https://learn.microsoft.com/windows/win32/api/ioringapi/nf-ioringapi-submitioring
// https://learn.microsoft.com/windows/win32/api/ioringapi/nf-ioringapi-closeioring
struct llama_dsv41_engram_prompt_cache::implementation {
    static constexpr uint32_t depth=256,page_size=8192,row_bytes=264;
    struct table {std::vector<uint32_t> ids;std::vector<uint8_t> data;};
    std::array<table,2> tables;
    std::array<uint64_t,2> bases{};
    bool attempted=false,prepared=false,load_attempted=false;
    // Tables whose rows are complete: 0, 1 (table 0) or 2 (both). Release
    // stores publish the rows; read() and wait_table() acquire them.
    std::atomic<uint32_t> ready_tables{0};
    std::atomic<bool> failed{false};
    uint64_t host_plan=0;
    // LLAMADECODE_ENGRAM_EARLY only.
    bool early_mode=false;
    uint32_t early_rings=0,early_small=0,early_large=0,early_batch=0;
    uint64_t early_arena=0;
    std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior;
    static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
    static uint64_t add(uint64_t a,uint64_t b){require(a<=UINT64_MAX-b,"Engram cache plan overflow");return a+b;}
    struct api_table {
        decltype(&::CreateIoRing) CreateIoRing=nullptr;
        decltype(&::IsIoRingOpSupported) IsIoRingOpSupported=nullptr;
        decltype(&::SetIoRingCompletionEvent) SetIoRingCompletionEvent=nullptr;
        decltype(&::BuildIoRingRegisterFileHandles) BuildIoRingRegisterFileHandles=nullptr;
        decltype(&::BuildIoRingRegisterBuffers) BuildIoRingRegisterBuffers=nullptr;
        decltype(&::BuildIoRingReadFile) BuildIoRingReadFile=nullptr;
        decltype(&::SubmitIoRing) SubmitIoRing=nullptr;
        decltype(&::PopIoRingCompletion) PopIoRingCompletion=nullptr;
        decltype(&::CloseIoRing) CloseIoRing=nullptr;
        api_table(){
            const auto module=GetModuleHandleW(L"KernelBase.dll");require(module,"Required system IoRing module unavailable");
            CreateIoRing=reinterpret_cast<decltype(CreateIoRing)>(GetProcAddress(module,"CreateIoRing"));
            IsIoRingOpSupported=reinterpret_cast<decltype(IsIoRingOpSupported)>(GetProcAddress(module,"IsIoRingOpSupported"));
            SetIoRingCompletionEvent=reinterpret_cast<decltype(SetIoRingCompletionEvent)>(GetProcAddress(module,"SetIoRingCompletionEvent"));
            BuildIoRingRegisterFileHandles=reinterpret_cast<decltype(BuildIoRingRegisterFileHandles)>(GetProcAddress(module,"BuildIoRingRegisterFileHandles"));
            BuildIoRingRegisterBuffers=reinterpret_cast<decltype(BuildIoRingRegisterBuffers)>(GetProcAddress(module,"BuildIoRingRegisterBuffers"));
            BuildIoRingReadFile=reinterpret_cast<decltype(BuildIoRingReadFile)>(GetProcAddress(module,"BuildIoRingReadFile"));
            SubmitIoRing=reinterpret_cast<decltype(SubmitIoRing)>(GetProcAddress(module,"SubmitIoRing"));
            PopIoRingCompletion=reinterpret_cast<decltype(PopIoRingCompletion)>(GetProcAddress(module,"PopIoRingCompletion"));
            CloseIoRing=reinterpret_cast<decltype(CloseIoRing)>(GetProcAddress(module,"CloseIoRing"));
            require(CreateIoRing&&IsIoRingOpSupported&&SetIoRingCompletionEvent&&BuildIoRingRegisterFileHandles&&BuildIoRingRegisterBuffers&&BuildIoRingReadFile&&SubmitIoRing&&PopIoRingCompletion&&CloseIoRing,"Required IoRing APIs unavailable; no serial fallback");
        }
    } api;
    struct io_scope {
        api_table &api;
        explicit io_scope(api_table &a):api(a){}
        io_scope(const io_scope&)=delete;io_scope&operator=(const io_scope&)=delete;
        HIORING ring=nullptr;
        HANDLE file=INVALID_HANDLE_VALUE,event=nullptr;
        uint8_t* pages=nullptr;
        uint32_t submitted=0;
        [[noreturn]] static void unsafe_cleanup(){
            std::fputs("DSV41_ENGRAM_IORING_CLEANUP_REFUSED: owned process exits before releasing buffers with unconfirmed IO\n",stderr);
            std::fflush(stderr);TerminateProcess(GetCurrentProcess(),79);std::abort();
        }
        ~io_scope(){
            if(ring){
                if(submitted&&file!=INVALID_HANDLE_VALUE)CancelIoEx(file,nullptr);
                const auto deadline=GetTickCount64()+5000;
                while(submitted){
                    IORING_CQE c{};const auto hr=api.PopIoRingCompletion(ring,&c);
                    if(hr==S_OK){--submitted;continue;}
                    if(hr!=S_FALSE||GetTickCount64()>=deadline)unsafe_cleanup();
                    const auto status=WaitForSingleObject(event,50);
                    if(status!=WAIT_OBJECT_0&&status!=WAIT_TIMEOUT)unsafe_cleanup();
                }
                // Close abandons unsubmitted SQ entries, but does NOT cancel
                // submitted reads. Every submitted completion was drained above.
                if(FAILED(api.CloseIoRing(ring)))unsafe_cleanup();
            }
            if(event)CloseHandle(event);
            if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
            if(pages)_aligned_free(pages);
        }
    };
public:
    void prepare(const llama_dsv41::engram_layout& layout,const int32_t* tokens,uint32_t count,
                 const std::array<uint64_t,2>& offsets,uint64_t admitted_host_bytes,llama_dsv41::engram_history initial){
        require(!attempted&&tokens&&count&&count<=1048576,"Invalid or repeated whole-prompt Engram preparation");attempted=true;
        // Charge both full-capacity ID vectors before allocating them. Dedup
        // does not reduce their capacity, so the final plan retains that charge.
        const uint64_t id_limit=uint64_t(count)*48*sizeof(uint32_t);
        require(add(id_limit,4*1024*1024)<=admitted_host_bytes,"Engram prompt IDs exceed explicit provider host admission");
        for(auto& t:tables)t.ids.reserve(size_t(count)*24);
        llama_dsv41::engram_history history=initial;
        std::vector<llama_dsv41::engram_rows> rows(1024);
        std::array<uint8_t,1024> mask;mask.fill(1);
        std::string why;
        for(uint32_t start=0;start<count;start+=1024){
            const auto n=std::min(1024u,count-start);
            require(llama_dsv41::engram_hash(layout,history,tokens+start,mask.data(),n,rows.data(),why),"Whole-prompt native Engram hash failed");
            for(uint32_t i=0;i<n;++i)for(size_t table=0;table<2;++table)
                tables[table].ids.insert(tables[table].ids.end(),rows[i][table].begin(),rows[i][table].end());
        }
        host_plan=4*1024*1024; // bounded aligned slots, ring queues and metadata
        for(size_t i=0;i<2;++i){
            auto& t=tables[i];std::sort(t.ids.begin(),t.ids.end());
            t.ids.erase(std::unique(t.ids.begin(),t.ids.end()),t.ids.end());
            require(!t.ids.empty()&&uint64_t(t.ids.back())<layout.rows[i],"Prompt Engram ID exceeds original table");
            host_plan=add(host_plan,add(uint64_t(t.ids.capacity())*sizeof(uint32_t),uint64_t(t.ids.size())*row_bytes));
        }
        require(host_plan<=admitted_host_bytes,"Whole-prompt Engram rows exceed explicit host admission");
        bases=offsets;prepared=true;
    }
    uint64_t budget()const{return prepared?host_plan:0;}
    void load(const wchar_t* path,HANDLE verified_file,uint64_t verified_file_bytes,const std::function<bool()>& stopped){
        require(prepared&&!load_attempted&&path&&verified_file!=INVALID_HANDLE_VALUE,"Engram rows must be prepared/admitted once before load");load_attempted=true;
        require(!stopped||!stopped(),"Engram preload cancelled before IO");
        io_scope io(api);
        io.file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED|FILE_FLAG_RANDOM_ACCESS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(io.file!=INVALID_HANDLE_VALUE,"Cannot open original Engram file for its own IoRing");
        {   // LLAMADECODE: the overlapped prompt-cache reads yield to the weight
            // DirectStorage streaming only if LLAMADECODE_ENGRAM_IO_PRIORITY=low (measured: low priority starves
            // the reads, 43K layers 123 s vs 33 s; default stays normal).
            const char*e=std::getenv("LLAMADECODE_ENGRAM_IO_PRIORITY");
            FILE_IO_PRIORITY_HINT_INFO hint{};hint.PriorityHint=e&&!std::strcmp(e,"low")?IoPriorityHintLow:IoPriorityHintNormal; /* low measured 123 s vs 33 s (starved); normal is the default */
            const BOOL ok=SetFileInformationByHandle(io.file,FileIoPriorityHintInfo,&hint,sizeof(hint));
            static bool logged=false;if(!logged){logged=true;std::fprintf(stderr,"LLAMADECODE_ENGRAM_IO_PRIORITY hint=%s applied=%d\n",hint.PriorityHint==IoPriorityHintLow?"low":"normal",int(ok));}
        }
        FILE_ID_INFO a{},b{};
        require(GetFileInformationByHandleEx(verified_file,FileIdInfo,&a,sizeof(a))&&
                GetFileInformationByHandleEx(io.file,FileIdInfo,&b,sizeof(b))&&
                a.VolumeSerialNumber==b.VolumeSerialNumber&&!std::memcmp(a.FileId.Identifier,b.FileId.Identifier,16),
                "Engram IoRing descriptor differs from the verified model");
        IORING_CREATE_FLAGS flags{};
        require(SUCCEEDED(api.CreateIoRing(IORING_VERSION_1,flags,depth,2*depth,&io.ring)),"Required Engram IoRing unavailable; no serial fallback");
        require(api.IsIoRingOpSupported(io.ring,IORING_OP_READ),"Engram IoRing read operation unavailable");
        io.event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        require(io.event&&SUCCEEDED(api.SetIoRingCompletionEvent(io.ring,io.event)),"Cannot register Engram IoRing completion event");
        io.pages=static_cast<uint8_t*>(_aligned_malloc(uint64_t(depth)*page_size,4096));
        require(io.pages,"Cannot allocate admitted Engram IoRing slots");
        // Pin this small slot arena and register the read-only file once;
        // request builders then use indices rather than per-read raw MDLs.
        require(api.IsIoRingOpSupported(io.ring,IORING_OP_REGISTER_FILES)&&api.IsIoRingOpSupported(io.ring,IORING_OP_REGISTER_BUFFERS),
                "Required Engram IoRing resource registration unavailable");
        IORING_BUFFER_INFO registered{io.pages,depth*page_size};
        require(SUCCEEDED(api.BuildIoRingRegisterFileHandles(io.ring,1,&io.file,1))&&
                SUCCEEDED(api.BuildIoRingRegisterBuffers(io.ring,1,&registered,2)),"Cannot enqueue Engram IoRing registrations");
        UINT32 registered_count=0;
        const auto register_hr=api.SubmitIoRing(io.ring,0,0,&registered_count);
        if(register_hr==S_OK){if(registered_count!=2)io_scope::unsafe_cleanup();io.submitted=registered_count;}
        else throw std::runtime_error("Required Engram IoRing registration submit failed");
        bool registered_file=false,registered_buffer=false;
        const auto register_deadline=GetTickCount64()+5000;
        while(io.submitted){
            IORING_CQE c{};const auto hr=api.PopIoRingCompletion(io.ring,&c);
            if(hr==S_FALSE){
                require(GetTickCount64()<register_deadline,"Engram IoRing registration timeout");
                const auto wait=WaitForSingleObject(io.event,50);
                require(wait==WAIT_OBJECT_0||wait==WAIT_TIMEOUT,"Engram IoRing registration wait failed");continue;
            }
            require(hr==S_OK,"Cannot dequeue Engram IoRing registration");--io.submitted;
            require(SUCCEEDED(c.ResultCode),"Engram IoRing resource registration failed");
            if(c.UserData==1){require(!registered_file,"Duplicate Engram file registration");registered_file=true;}
            else if(c.UserData==2){require(!registered_buffer,"Duplicate Engram buffer registration");registered_buffer=true;}
            else throw std::runtime_error("Unknown Engram resource registration completion");
        }
        require(registered_file&&registered_buffer,"Missing Engram resource registration");
        for(auto& t:tables){t.data.resize(t.ids.size()*row_bytes);require(t.data.capacity()==t.data.size(),"Unexpected raw-row cache allocation capacity");}
        struct slot {uint32_t table=0,extent=0,skip=0;size_t row=0;UINT_PTR id=0;bool live=false;};
        std::array<slot,depth> slots;
        std::vector<uint32_t> staged;staged.reserve(depth);
        size_t next_row=0;uint32_t next_table=0;UINT_PTR sequence=0;
        uint64_t read_bytes=0,read_requests=0,submits=0;size_t table0_done=0;
        const char*batch_env=std::getenv("LLAMA_DSV41_ENGRAM_SUBMIT_BATCH");
        const bool batch_submits=batch_env&&std::strcmp(batch_env,"1")==0;
        auto issue=[&](uint32_t i){
            while(next_table<2&&next_row==tables[next_table].ids.size()){++next_table;next_row=0;}
            if(next_table==2)return false;
            auto& s=slots[i];require(!s.live,"Engram IoRing reused a live slot");
            s.table=next_table;s.row=next_row++;
            const uint64_t offset=add(bases[s.table],uint64_t(tables[s.table].ids[s.row])*row_bytes);
            const uint64_t begin=offset&~uint64_t(4095);s.skip=uint32_t(offset-begin);
            s.extent=(s.skip+row_bytes+4095)&~4095u;
            require(s.extent<=page_size&&begin<=verified_file_bytes&&s.extent<=verified_file_bytes-begin,"Engram aligned read exceeds original file");
            require(sequence<(UINTPTR_MAX>>9),"Engram operation nonce overflow");s.id=(++sequence<<9)|(i+1);
            require(SUCCEEDED(api.BuildIoRingReadFile(io.ring,IoRingHandleRefFromIndex(0),
                IoRingBufferRefFromIndexAndOffset(0,i*page_size),s.extent,begin,s.id,IOSQE_FLAGS_NONE)),
                "Cannot enqueue required Engram row read");
            s.live=true;staged.push_back(i);read_bytes+=s.extent;++read_requests;return true;
        };
        auto submit=[&]{
            if(staged.empty())return;
            UINT32 submitted=0;const auto hr=api.SubmitIoRing(io.ring,0,0,&submitted);
            // Per the API, non-timeout errors leave all entries in the SQ.
            // No wait was requested here, so only a full successful submit is valid.
            if(hr==S_OK){
                if(submitted!=staged.size())io_scope::unsafe_cleanup();
                io.submitted+=submitted;staged.clear();++submits;
            }else throw std::runtime_error("Required Engram IoRing Submit failed; no serial fallback");
        };
        const auto began=GetTickCount64();
        for(uint32_t i=0;i<depth;++i)if(!issue(i))break;
        submit();
        while(io.submitted){
            require((!stopped||!stopped())&&GetTickCount64()-began<120000,"Engram preload cancelled or exceeded120s");
            uint32_t completed=0;
            for(;completed<64;++completed){
                IORING_CQE c{};const auto hr=api.PopIoRingCompletion(io.ring,&c);
                if(hr==S_FALSE)break;
                require(hr==S_OK&&io.submitted,"Cannot dequeue Engram IoRing completion");
                --io.submitted;
                const auto id=uint32_t(c.UserData&511u);
                require(id&&id<=depth,"Engram completion has an unknown slot");
                auto& s=slots[id-1];
                require(s.live&&s.id==c.UserData,"Engram completion nonce/ownership differs");s.live=false;
                require(SUCCEEDED(c.ResultCode)&&c.Information==s.extent,"Engram IoRing read failed or was short");
                std::memcpy(tables[s.table].data.data()+s.row*row_bytes,io.pages+uint64_t(id-1)*page_size+s.skip,row_bytes);
                if(s.table==0&&++table0_done==tables[0].ids.size()){
                    ready_tables.store(1,std::memory_order_release);
                    llama_dsv41_layer_log("DS4PERF_ENGRAM_TABLE_LOADED table=0 rows=%zu wall_ms=%llu\n",tables[0].ids.size(),(unsigned long long)(GetTickCount64()-began));
                }
                issue(id-1);
            }
            require(io.submitted+staged.size()<=depth,"Engram queued and submitted slots exceed fixed depth");
            // Keep the same bounded slot ring. Amortize tiny completion groups,
            // but replenish before outstanding reads fall below half its depth.
            if(!batch_submits||staged.size()>=64||io.submitted<=depth/2)submit();
            if(!completed&&io.submitted){
                const auto wait=WaitForSingleObject(io.event,100);
                require(wait==WAIT_OBJECT_0||wait==WAIT_TIMEOUT,"Engram IoRing completion wait failed");
            }
        }
        require(next_table==2&&staged.empty()&&table0_done==tables[0].ids.size(),"Incomplete whole-prompt Engram preload");
        ready_tables.store(2,std::memory_order_release);
        if(batch_submits)llama_dsv41_layer_log("DS4PERF_ENGRAM_SUBMIT_BATCH requests=%llu submits=%llu depth=256 threshold=64 low_water=128 all_completions_drained=1\n",(unsigned long long)read_requests,(unsigned long long)submits);
        llama_dsv41_layer_log("[dsv41-engram-prefetch] table0_rows=%zu table1_rows=%zu host_plan_bytes=%llu read_bytes=%llu requests=%llu submits=%llu wall_ms=%llu\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)host_plan,(unsigned long long)read_bytes,
            (unsigned long long)read_requests,(unsigned long long)submits,(unsigned long long)(GetTickCount64()-began));
        std::fflush(stderr);
    }
    // ---- LLAMADECODE_ENGRAM_EARLY: same rows/ids/order as prepare(); the IO
    // arena is sized from the admission headroom left by the actual rows. ----
    static uint32_t env_u32(const char*name,uint32_t fallback,uint32_t lo,uint32_t hi){
        const char*e=std::getenv(name);if(!e||!*e)return fallback;
        char*end=nullptr;const unsigned long v=std::strtoul(e,&end,10);
        require(end&&!*end&&v>=lo&&v<=hi,"LLAMADECODE_ENGRAM_* setting out of range");return uint32_t(v);
    }
    static constexpr uint32_t small_slot=8192,large_slot=llama_dsv41_engram_plan::default_merge_cap;
    void prepare_early(const llama_dsv41::engram_layout& layout,const int32_t* tokens,uint32_t count,
                       const std::array<uint64_t,2>& offsets,uint64_t admitted,llama_dsv41::engram_history initial,
                       std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior_cache){
        prepare(layout,tokens,count,offsets,admitted,initial); // host_plan holds the fixed 4 MiB IO reserve
        early_mode=true;prior=std::move(prior_cache);
        early_rings=env_u32("LLAMADECODE_ENGRAM_RINGS",4,1,8);
        const uint32_t depth=env_u32("LLAMADECODE_ENGRAM_DEPTH",1024,32,4096);
        const uint64_t rows=uint64_t(tables[0].ids.size())+tables[1].ids.size();
        const uint64_t without=host_plan-4*1024*1024;
        // Slot tables, plan state and (with a prior cache) the hit bitmaps.
        const uint64_t meta=add(256*1024,add(uint64_t(early_rings)*(depth+depth/16+1)*64,prior?rows/8+64:0));
        require(add(without,meta)<admitted,"Early Engram metadata exceeds explicit host admission");
        const uint64_t per=(admitted-without-meta)/early_rings;
        // Per ring: n small (8 KiB) slots and max(1,n/16) large (64 KiB) slots.
        require(per>uint64_t(large_slot)+16*(small_slot+large_slot/16),"Early Engram IO arena below its admitted minimum");
        early_small=uint32_t(std::min<uint64_t>(depth,(per-large_slot)/(small_slot+large_slot/16)));
        early_large=std::max(1u,early_small/16);
        early_batch=std::min(env_u32("LLAMADECODE_ENGRAM_BATCH",256,1,4096),std::max(1u,early_small/2));
        early_arena=uint64_t(early_rings)*(uint64_t(early_small)*small_slot+uint64_t(early_large)*large_slot);
        host_plan=add(without,add(meta,early_arena));
        require(host_plan<=admitted,"Early Engram rows and IO arena exceed explicit host admission");
    }
    struct early_slot {uint64_t file_offset=0;uint32_t bytes=0,table=0,first=0,last=0,rows=0;UINT_PTR id=0;bool live=false;};
    struct early_ring {
        HIORING ring=nullptr;HANDLE event=nullptr;uint8_t* arena=nullptr;uint32_t submitted=0,staged=0;
        std::vector<early_slot> slots;std::vector<uint32_t> free_small,free_large;
    };
    struct early_scope {
        api_table& api;HANDLE file=INVALID_HANDLE_VALUE;std::vector<early_ring> rings;
        explicit early_scope(api_table& a):api(a){}
        early_scope(const early_scope&)=delete;early_scope&operator=(const early_scope&)=delete;
        ~early_scope(){
            bool any=false;for(const auto& r:rings)any=any||r.submitted;
            if(any&&file!=INVALID_HANDLE_VALUE)CancelIoEx(file,nullptr);
            const auto deadline=GetTickCount64()+5000;
            for(auto& r:rings){
                if(!r.ring)continue;
                while(r.submitted){
                    IORING_CQE c{};const auto hr=api.PopIoRingCompletion(r.ring,&c);
                    if(hr==S_OK){--r.submitted;continue;}
                    if(hr!=S_FALSE||GetTickCount64()>=deadline)io_scope::unsafe_cleanup();
                    const auto status=WaitForSingleObject(r.event,50);
                    if(status!=WAIT_OBJECT_0&&status!=WAIT_TIMEOUT)io_scope::unsafe_cleanup();
                }
                // Staged-but-unsubmitted entries are abandoned by Close; every
                // submitted read was drained above before its buffer is freed.
                if(FAILED(api.CloseIoRing(r.ring)))io_scope::unsafe_cleanup();
            }
            for(auto& r:rings){if(r.event)CloseHandle(r.event);if(r.arena)_aligned_free(r.arena);}
            if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
        }
    };
    void load_early(const wchar_t* path,HANDLE verified_file,uint64_t verified_file_bytes,const std::function<bool()>& stopped){
        namespace ep=llama_dsv41_engram_plan;
        require(prepared&&early_mode&&!load_attempted&&path&&verified_file!=INVALID_HANDLE_VALUE,"Early Engram rows must be prepared/admitted once before load");load_attempted=true;
        require(!stopped||!stopped(),"Engram preload cancelled before IO");
        using clock=std::chrono::steady_clock;const auto began=clock::now();
        auto since=[&](clock::time_point t){return std::chrono::duration<double,std::milli>(clock::now()-t).count();};
        const bool weight_at_start=llama_dsv41_engram_weight_pass.load(std::memory_order_relaxed);
        for(auto& t:tables){t.data.resize(t.ids.size()*row_bytes);require(t.data.capacity()==t.data.size(),"Unexpected raw-row cache allocation capacity");}
        // Previous-request reuse (one deep): rows the provider's last completed
        // cache already holds are copied, not read. The caller keeps that
        // cache's charge counted until this load ends.
        std::array<std::vector<uint8_t>,2> hits;std::array<uint64_t,2> hit_rows{};
        int reuse=prior?0:-1;
        if(prior){
            const implementation* q=prior->p.get();
            if(q&&q->ready_tables.load(std::memory_order_acquire)==2&&q->bases==bases){
                reuse=1;
                for(size_t t=0;t<2;++t){
                    const auto& mine=tables[t].ids;const auto& old=q->tables[t].ids;hits[t].assign((mine.size()+7)/8,0);
                    for(size_t i=0,j=0;i<mine.size()&&j<old.size();){
                        if(mine[i]<old[j])++i;else if(old[j]<mine[i])++j;
                        else{std::memcpy(tables[t].data.data()+i*row_bytes,q->tables[t].data.data()+j*row_bytes,row_bytes);
                             hits[t][i>>3]|=uint8_t(1u<<(i&7));++hit_rows[t];++i;++j;}
                    }
                }
            }
            prior.reset();
        }
        const std::array<uint64_t,2> need{tables[0].ids.size()-hit_rows[0],tables[1].ids.size()-hit_rows[1]};
        std::array<uint64_t,2> done{};
        double table0_ms=-1;
        if(!need[0]){ready_tables.store(1,std::memory_order_release);table0_ms=since(began);}
        early_scope io(api);
        io.file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED|FILE_FLAG_RANDOM_ACCESS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(io.file!=INVALID_HANDLE_VALUE,"Cannot open original Engram file for its own IoRings");
        {   const char*e=std::getenv("LLAMADECODE_ENGRAM_IO_PRIORITY");
            FILE_IO_PRIORITY_HINT_INFO hint{};hint.PriorityHint=e&&!std::strcmp(e,"low")?IoPriorityHintLow:IoPriorityHintNormal;
            const BOOL ok=SetFileInformationByHandle(io.file,FileIoPriorityHintInfo,&hint,sizeof(hint));
            std::fprintf(stderr,"LLAMADECODE_ENGRAM_IO_PRIORITY hint=%s applied=%d early=1\n",hint.PriorityHint==IoPriorityHintLow?"low":"normal",int(ok));
        }
        FILE_ID_INFO a{},b{};
        require(GetFileInformationByHandleEx(verified_file,FileIdInfo,&a,sizeof(a))&&
                GetFileInformationByHandleEx(io.file,FileIdInfo,&b,sizeof(b))&&
                a.VolumeSerialNumber==b.VolumeSerialNumber&&!std::memcmp(a.FileId.Identifier,b.FileId.Identifier,16),
                "Engram IoRing descriptor differs from the verified model");
        const uint32_t slots_per_ring=early_small+early_large;
        const uint64_t ring_arena=uint64_t(early_small)*small_slot+uint64_t(early_large)*large_slot;
        require(ring_arena<=UINT32_MAX,"Early Engram ring arena exceeds one registered buffer");
        auto slot_offset=[&](uint32_t s){return s<early_small?uint64_t(s)*small_slot:uint64_t(early_small)*small_slot+uint64_t(s-early_small)*large_slot;};
        io.rings.resize(early_rings);
        for(auto& r:io.rings){
            IORING_CREATE_FLAGS flags{};
            require(SUCCEEDED(api.CreateIoRing(IORING_VERSION_1,flags,slots_per_ring,2*slots_per_ring,&r.ring)),"Required Engram IoRing unavailable; no serial fallback");
            require(api.IsIoRingOpSupported(r.ring,IORING_OP_READ)&&api.IsIoRingOpSupported(r.ring,IORING_OP_REGISTER_FILES)&&
                    api.IsIoRingOpSupported(r.ring,IORING_OP_REGISTER_BUFFERS),"Required Engram IoRing operations unavailable");
            r.event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            require(r.event&&SUCCEEDED(api.SetIoRingCompletionEvent(r.ring,r.event)),"Cannot register Engram IoRing completion event");
            r.arena=static_cast<uint8_t*>(_aligned_malloc(ring_arena,4096));require(r.arena,"Cannot allocate admitted Engram IoRing arena");
            IORING_BUFFER_INFO registered{r.arena,UINT32(ring_arena)};
            require(SUCCEEDED(api.BuildIoRingRegisterFileHandles(r.ring,1,&io.file,1))&&
                    SUCCEEDED(api.BuildIoRingRegisterBuffers(r.ring,1,&registered,2)),"Cannot enqueue Engram IoRing registrations");
            UINT32 registered_count=0;
            const auto register_hr=api.SubmitIoRing(r.ring,0,0,&registered_count);
            if(register_hr==S_OK){if(registered_count!=2)io_scope::unsafe_cleanup();r.submitted=registered_count;}
            else throw std::runtime_error("Required Engram IoRing registration submit failed");
            bool registered_file=false,registered_buffer=false;
            const auto register_deadline=GetTickCount64()+5000;
            while(r.submitted){
                IORING_CQE c{};const auto hr=api.PopIoRingCompletion(r.ring,&c);
                if(hr==S_FALSE){
                    require(GetTickCount64()<register_deadline,"Engram IoRing registration timeout");
                    const auto wait=WaitForSingleObject(r.event,50);
                    require(wait==WAIT_OBJECT_0||wait==WAIT_TIMEOUT,"Engram IoRing registration wait failed");continue;
                }
                require(hr==S_OK,"Cannot dequeue Engram IoRing registration");--r.submitted;
                require(SUCCEEDED(c.ResultCode),"Engram IoRing resource registration failed");
                if(c.UserData==1){require(!registered_file,"Duplicate Engram file registration");registered_file=true;}
                else if(c.UserData==2){require(!registered_buffer,"Duplicate Engram buffer registration");registered_buffer=true;}
                else throw std::runtime_error("Unknown Engram resource registration completion");
            }
            require(registered_file&&registered_buffer,"Missing Engram resource registration");
            r.slots.resize(slots_per_ring);
            for(uint32_t s=early_small;s-->0;)r.free_small.push_back(s);
            for(uint32_t s=slots_per_ring;s-->early_small;)r.free_large.push_back(s);
        }
        const double setup_ms=since(began);
        // Plan stream: table 0's extents first, then table 1's, in file order.
        uint32_t cur_table=0;size_t cur_pos=0;bool have_pending=false;ep::extent pending;uint32_t pending_table=0;
        auto fetch=[&]{
            if(have_pending)return true;
            while(cur_table<2){
                const auto& t=tables[cur_table];
                if(ep::next_extent(t.ids.data(),t.ids.size(),hits[cur_table].empty()?nullptr:hits[cur_table].data(),cur_pos,bases[cur_table],row_bytes,large_slot,pending)){
                    pending_table=cur_table;have_pending=true;return true;}
                ++cur_table;cur_pos=0;
            }
            return false;
        };
        UINT_PTR sequence=0;uint64_t read_bytes=0,requests=0,submits=0,useful_rows=0,max_inflight=0,replanned=0;
        auto issue=[&](early_ring& r){
            while(fetch()){
                if(pending.bytes>small_slot&&r.free_large.empty()){
                    // No large slot free: wait for one while this ring still
                    // has a quarter of its slots in flight; otherwise re-plan
                    // the same rows from the same index at the small-slot cap
                    // (same rows land, more requests) instead of starving.
                    if(r.submitted+r.staged>=early_small/4)return;
                    const auto& t=tables[pending_table];cur_pos=pending.first;
                    require(ep::next_extent(t.ids.data(),t.ids.size(),hits[pending_table].empty()?nullptr:hits[pending_table].data(),cur_pos,bases[pending_table],row_bytes,small_slot,pending),"Engram re-plan lost its rows");
                    ++replanned;
                }
                auto& pool=pending.bytes>small_slot?r.free_large:r.free_small;
                if(pool.empty())return;
                const uint32_t s=pool.back();pool.pop_back();
                auto& sl=r.slots[s];require(!sl.live,"Engram IoRing reused a live slot");
                require(pending.file_offset<=verified_file_bytes&&pending.bytes<=verified_file_bytes-pending.file_offset,"Engram aligned read exceeds original file");
                require(sequence<(UINTPTR_MAX>>17),"Engram operation nonce overflow");
                sl.file_offset=pending.file_offset;sl.bytes=pending.bytes;sl.table=pending_table;sl.first=pending.first;sl.last=pending.last;sl.rows=pending.rows;
                sl.id=(++sequence<<16)|(s+1);
                require(SUCCEEDED(api.BuildIoRingReadFile(r.ring,IoRingHandleRefFromIndex(0),
                    IoRingBufferRefFromIndexAndOffset(0,UINT32(slot_offset(s))),sl.bytes,sl.file_offset,sl.id,IOSQE_FLAGS_NONE)),
                    "Cannot enqueue required Engram extent read");
                sl.live=true;++r.staged;read_bytes+=sl.bytes;++requests;useful_rows+=sl.rows;have_pending=false;
            }
        };
        auto submit=[&](early_ring& r,bool drain){
            if(!r.staged)return;
            // Always batched: submit a full batch, or whatever is staged once
            // the ring's in-flight reads fall to half its slots or the plan ends.
            if(!drain&&r.staged<early_batch&&r.submitted>slots_per_ring/2)return;
            UINT32 n=0;const auto hr=api.SubmitIoRing(r.ring,0,0,&n);
            // No wait requested: only a full successful submit is valid.
            if(hr==S_OK){if(n!=r.staged)io_scope::unsafe_cleanup();r.submitted+=n;r.staged=0;++submits;}
            else throw std::runtime_error("Required Engram IoRing Submit failed; no serial fallback");
        };
        std::vector<HANDLE> events;for(const auto& r:io.rings)events.push_back(r.event);
        auto last=clock::now();double overlap_ms=0;bool weight_active=weight_at_start;
        while(true){
            const auto now=clock::now();
            if(weight_active)overlap_ms+=std::chrono::duration<double,std::milli>(now-last).count();
            last=now;weight_active=llama_dsv41_engram_weight_pass.load(std::memory_order_relaxed);
            require((!stopped||!stopped())&&since(began)<120000,"Engram preload cancelled or exceeded120s");
            bool progress=false;
            for(auto& r:io.rings){
                for(uint32_t k=0;k<1024;++k){
                    IORING_CQE c{};const auto hr=api.PopIoRingCompletion(r.ring,&c);
                    if(hr==S_FALSE)break;
                    require(hr==S_OK&&r.submitted,"Cannot dequeue Engram IoRing completion");--r.submitted;
                    const uint32_t s1=uint32_t(c.UserData&0xFFFFu);
                    require(s1&&s1<=slots_per_ring,"Engram completion has an unknown slot");
                    auto& sl=r.slots[s1-1];
                    require(sl.live&&sl.id==c.UserData,"Engram completion nonce/ownership differs");sl.live=false;
                    require(SUCCEEDED(c.ResultCode)&&c.Information==sl.bytes,"Engram IoRing read failed or was short");
                    auto& t=tables[sl.table];const uint8_t* src=r.arena+slot_offset(s1-1);
                    const uint8_t* bits=hits[sl.table].empty()?nullptr:hits[sl.table].data();
                    for(size_t i=sl.first;i<=sl.last;++i){
                        if(ep::is_hit(bits,i))continue;
                        const uint64_t off=bases[sl.table]+uint64_t(t.ids[i])*row_bytes-sl.file_offset;
                        require(off+row_bytes<=sl.bytes,"Engram row outside its extent");
                        std::memcpy(t.data.data()+i*row_bytes,src+off,row_bytes);
                    }
                    done[sl.table]+=sl.rows;
                    (s1-1<early_small?r.free_small:r.free_large).push_back(s1-1);progress=true;
                }
            }
            if(ready_tables.load(std::memory_order_relaxed)==0&&done[0]==need[0]){
                ready_tables.store(1,std::memory_order_release);table0_ms=since(began);
                llama_dsv41_layer_log("DS4PERF_ENGRAM_TABLE_LOADED table=0 rows=%zu wall_ms=%llu early=1\n",tables[0].ids.size(),(unsigned long long)table0_ms);
            }
            for(auto& r:io.rings)issue(r);
            const bool plan_done=!fetch();
            uint64_t inflight=0;
            for(auto& r:io.rings){submit(r,plan_done);inflight+=r.submitted;}
            max_inflight=std::max(max_inflight,inflight);
            bool idle=plan_done;for(const auto& r:io.rings)idle=idle&&!r.staged&&!r.submitted;
            if(idle)break;
            if(!progress&&inflight){
                const auto wait=WaitForMultipleObjects(DWORD(events.size()),events.data(),FALSE,100);
                require(wait<WAIT_OBJECT_0+events.size()||wait==WAIT_TIMEOUT,"Engram IoRing completion wait failed");
            }
        }
        require(done[0]==need[0]&&done[1]==need[1]&&useful_rows==need[0]+need[1],"Incomplete whole-prompt Engram early preload");
        ready_tables.store(2,std::memory_order_release);
        const double wall=since(began);
        const uint64_t rows=tables[0].ids.size()+tables[1].ids.size();
        llama_dsv41_layer_log("LLAMADECODE_ENGRAM_EARLY_LOAD table0_rows=%zu table1_rows=%zu hit_rows=%llu reuse=%d extents=%llu merged_rows=%llu requests=%llu submits=%llu "
            "read_bytes=%llu useful_bytes=%llu rings=%u slots=%u large_slots=%u batch=%u replanned=%llu arena_bytes=%llu host_plan_bytes=%llu max_inflight=%llu "
            "setup_ms=%.3f table0_ms=%.3f wall_ms=%.3f weight_pass_at_start=%d weight_pass_overlap_ms=%.3f overlap_frac=%.3f\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)(hit_rows[0]+hit_rows[1]),reuse,(unsigned long long)requests,
            (unsigned long long)(useful_rows-requests),(unsigned long long)requests,(unsigned long long)submits,(unsigned long long)read_bytes,
            (unsigned long long)(rows*row_bytes),early_rings,early_small,early_large,early_batch,(unsigned long long)replanned,(unsigned long long)early_arena,(unsigned long long)host_plan,
            (unsigned long long)max_inflight,setup_ms,table0_ms,wall,int(weight_at_start),overlap_ms,wall>0?overlap_ms/wall:0.0);
        llama_dsv41_layer_log("[dsv41-engram-prefetch] table0_rows=%zu table1_rows=%zu host_plan_bytes=%llu read_bytes=%llu requests=%llu submits=%llu wall_ms=%llu early=1\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)host_plan,(unsigned long long)read_bytes,
            (unsigned long long)requests,(unsigned long long)submits,(unsigned long long)wall);
        std::fflush(stderr);
    }
    bool read(uint32_t table,uint64_t offset,void* destination,size_t bytes)const{
        if(table>=2||ready_tables.load(std::memory_order_acquire)<=table||bytes!=row_bytes||!destination||offset<bases[table]||(offset-bases[table])%row_bytes)return false;
        const auto id=(offset-bases[table])/row_bytes;if(id>UINT32_MAX)return false;
        const auto& t=tables[table];const auto found=std::lower_bound(t.ids.begin(),t.ids.end(),uint32_t(id));
        if(found==t.ids.end()||*found!=id)return false;
        std::memcpy(destination,t.data.data()+size_t(found-t.ids.begin())*row_bytes,row_bytes);return true;
    }
    void wait_table(uint32_t table,const std::function<bool()>& stopped)const{
        require(table<2&&prepared,"Engram table wait requires a prepared cache");
        while(ready_tables.load(std::memory_order_acquire)<=table){
            require(!failed.load(std::memory_order_acquire),"Whole-prompt Engram preload failed before this table was ready");
            require(!stopped||!stopped(),"Engram table wait cancelled");
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
};
#else
struct llama_dsv41_engram_prompt_cache::implementation {};
#endif
llama_dsv41_engram_prompt_cache::llama_dsv41_engram_prompt_cache():p(new implementation){}
llama_dsv41_engram_prompt_cache::~llama_dsv41_engram_prompt_cache()=default;
void llama_dsv41_engram_prompt_cache::prepare(const llama_dsv41::engram_layout &l,const int32_t*t,uint32_t n,const std::array<uint64_t,2>&o,uint64_t b,llama_dsv41::engram_history initial){
#ifdef _WIN32
 p->prepare(l,t,n,o,b,initial);
#else
 (void)l;(void)t;(void)n;(void)o;(void)b;throw std::runtime_error("Engram IoRing requires Windows");
#endif
}
uint64_t llama_dsv41_engram_prompt_cache::budget()const{
#ifdef _WIN32
 return p->budget();
#else
 return 0;
#endif
}
void llama_dsv41_engram_prompt_cache::prepare_early(const llama_dsv41::engram_layout &l,const int32_t*t,uint32_t n,const std::array<uint64_t,2>&o,uint64_t b,llama_dsv41::engram_history initial,
        std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior){
#ifdef _WIN32
 p->prepare_early(l,t,n,o,b,initial,std::move(prior));
#else
 (void)l;(void)t;(void)n;(void)o;(void)b;(void)initial;(void)prior;throw std::runtime_error("Engram IoRing requires Windows");
#endif
}
void llama_dsv41_engram_prompt_cache::load_early(const wchar_t*path,void*file,uint64_t n,const std::function<bool()>&stop){
#ifdef _WIN32
 try{p->load_early(path,static_cast<HANDLE>(file),n,stop);}
 catch(...){p->failed.store(true,std::memory_order_release);throw;}
#else
 (void)path;(void)file;(void)n;(void)stop;throw std::runtime_error("Engram IoRing requires Windows");
#endif
}
void llama_dsv41_engram_prompt_cache::load(const wchar_t*path,void*file,uint64_t n,const std::function<bool()>&stop){
#ifdef _WIN32
 try{p->load(path,static_cast<HANDLE>(file),n,stop);}
 catch(...){p->failed.store(true,std::memory_order_release);throw;}
#else
 (void)path;(void)file;(void)n;(void)stop;throw std::runtime_error("Engram IoRing requires Windows");
#endif
}
bool llama_dsv41_engram_prompt_cache::read(uint32_t table,uint64_t offset,void*dst,size_t n)const{
#ifdef _WIN32
 return p->read(table,offset,dst,n);
#else
 (void)table;(void)offset;(void)dst;(void)n;return false;
#endif
}
void llama_dsv41_engram_prompt_cache::wait_table(uint32_t table,const std::function<bool()>&stop)const{
#ifdef _WIN32
 p->wait_table(table,stop);
#else
 (void)table;(void)stop;throw std::runtime_error("Engram IoRing requires Windows");
#endif
}
