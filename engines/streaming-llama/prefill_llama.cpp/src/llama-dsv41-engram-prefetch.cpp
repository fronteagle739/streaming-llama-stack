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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifdef _WIN32
// Submit/cleanup contracts: https://learn.microsoft.com/windows/win32/api/ioringapi/nf-ioringapi-submitioring
// https://learn.microsoft.com/windows/win32/api/ioringapi/nf-ioringapi-closeioring
struct llama_dsv41_engram_prompt_cache::implementation {
    static constexpr uint32_t depth=256,page_size=8192,row_bytes=264;
    struct table {std::vector<uint32_t> ids;std::vector<uint8_t> data;};
    std::array<table,2> tables;
    std::array<uint64_t,2> bases{};
    bool attempted=false,prepared=false,load_attempted=false,ready=false;
    uint64_t host_plan=0;
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
        uint64_t read_bytes=0,read_requests=0,submits=0;
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
        require(next_table==2&&staged.empty(),"Incomplete whole-prompt Engram preload");
        ready=true;
        if(batch_submits)llama_dsv41_layer_log("DS4PERF_ENGRAM_SUBMIT_BATCH requests=%llu submits=%llu depth=256 threshold=64 low_water=128 all_completions_drained=1\n",(unsigned long long)read_requests,(unsigned long long)submits);
        llama_dsv41_layer_log("[dsv41-engram-prefetch] table0_rows=%zu table1_rows=%zu host_plan_bytes=%llu read_bytes=%llu requests=%llu submits=%llu wall_ms=%llu\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)host_plan,(unsigned long long)read_bytes,
            (unsigned long long)read_requests,(unsigned long long)submits,(unsigned long long)(GetTickCount64()-began));
        std::fflush(stderr);
    }
    bool read(uint32_t table,uint64_t offset,void* destination,size_t bytes)const{
        if(!ready||table>=2||bytes!=row_bytes||!destination||offset<bases[table]||(offset-bases[table])%row_bytes)return false;
        const auto id=(offset-bases[table])/row_bytes;if(id>UINT32_MAX)return false;
        const auto& t=tables[table];const auto found=std::lower_bound(t.ids.begin(),t.ids.end(),uint32_t(id));
        if(found==t.ids.end()||*found!=id)return false;
        std::memcpy(destination,t.data.data()+size_t(found-t.ids.begin())*row_bytes,row_bytes);return true;
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
void llama_dsv41_engram_prompt_cache::load(const wchar_t*path,void*file,uint64_t n,const std::function<bool()>&stop){
#ifdef _WIN32
 p->load(path,static_cast<HANDLE>(file),n,stop);
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
