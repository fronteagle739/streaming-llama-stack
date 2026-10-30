// Included by the Windows producer after its bounded native-file/hash helpers.
#include "live-session-checkpoint.h"
#include "llama-dsv41-artifact.h"
#include "llama-dsv41-layer-log.h"
#include "llama-dsv41-resident-append.h"
#include <thread>
#include <filesystem>
#include <optional>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>

namespace {
class session_output_redirect {
    HANDLE wire=nullptr;
public:
    ~session_output_redirect(){if(wire)CloseHandle(wire);}
    HANDLE get()const{return wire;}
    void bind(){
        const HANDLE output=GetStdHandle(STD_OUTPUT_HANDLE),input=GetStdHandle(STD_INPUT_HANDLE);
        require(!wire&&GetFileType(input)==FILE_TYPE_PIPE&&GetFileType(output)==FILE_TYPE_PIPE,"Live session requires private inherited pipes");
        require(DuplicateHandle(GetCurrentProcess(),output,GetCurrentProcess(),&wire,0,FALSE,DUPLICATE_SAME_ACCESS)!=0,"Cannot retain private session output");
        require(std::fflush(stdout)==0&&_dup2(_fileno(stderr),_fileno(stdout))==0&&SetStdHandle(STD_OUTPUT_HANDLE,GetStdHandle(STD_ERROR_HANDLE)),"Cannot isolate diagnostic stdout from session payload");
    }
};
class session_pipe {
    HANDLE input=GetStdHandle(STD_INPUT_HANDLE),output;
    std::array<unsigned char,65536>buffer{};size_t first=0,last=0;
    std::chrono::steady_clock::time_point deadline;
    uint64_t timeout_ms;
public:
    explicit session_pipe(uint64_t timeout,HANDLE output):output(output),timeout_ms(timeout){
        require(GetFileType(input)==FILE_TYPE_PIPE&&GetFileType(output)==FILE_TYPE_PIPE,"Session stdio requires private parent-owned pipes");renew();
    }
    void renew(){deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);}
    void check(){stop();require(std::chrono::steady_clock::now()<deadline,"Session private-pipe deadline expired");}
    void fill(){
        require(first==last,"Session pipe pending data overwrite");first=last=0;
        while(!last){check();DWORD available=0;require(PeekNamedPipe(input,nullptr,0,nullptr,&available,nullptr)!=0,"Session parent input pipe closed");
            if(!available){Sleep(10);continue;}DWORD got=0;
            require(ReadFile(input,buffer.data(),DWORD(std::min<size_t>(available,buffer.size())),&got,nullptr)&&got,"Session pipe read failed");last=got;}
    }
    void read(void*dst,size_t bytes){
        require(bytes<=65536,"Session read exceeds64KiB");auto*out=static_cast<unsigned char*>(dst);
        while(bytes){check();if(first==last)fill();const size_t n=std::min(bytes,last-first);std::memcpy(out,buffer.data()+first,n);out+=n;first+=n;bytes-=n;}
    }
    std::string line(){
        std::string out;out.reserve(16384);
        for(;;){if(first==last)fill();const auto*b=buffer.data()+first;const auto*e=static_cast<const unsigned char*>(std::memchr(b,'\n',last-first));
            const size_t n=e?size_t(e-b):last-first;require(out.size()+n<=1024*1024,"Session command exceeds1MiB");out.append(reinterpret_cast<const char*>(b),n);first+=n;
            if(e){++first;return out;}}
    }
    void write(const void*src,size_t bytes){
        require(bytes<=65536,"Session write exceeds64KiB");check();DWORD sent=0;
        // The parent drains this private pipe under the same kill-on-close Job
        // and operation deadline. Its cancellation terminates blocked writes.
        require(WriteFile(output,src,DWORD(bytes),&sent,nullptr)&&sent==bytes,"Session pipe write failed");
    }
    void send(const json&j){const auto text=j.dump()+"\n";require(text.size()<=65536,"Session reply metadata exceeds64KiB");write(text.data(),text.size());}
};
}
// DS4COMBO fast switch: the base reader needs session_pipe and is used by resume.
#include "combo-base.inc"
namespace {

using decode_quiet_scope=llama_dsv41_decode_quiet_scope; // per-layer logs opt-in during decode steps
// LLAMADECODE ds4-decode: LLAMADECODE_DS4_DECODE=1 runs each decode step with
// ds4-win's own single-token decode (ds4-decode.dll, built from the stack's
// engines/ds4-win with DS4_EXTERNAL_DECODE) on this owner's canonical state,
// resident weights and B2 expert cache. Prefill stays on the executor.
// LLAMADECODE cache growth: once the first one-token step has sized the
// decode graph, the VRAM freed by the decode-sized plan (minus the decode
// graph, the 1-row hidden bank and a growth margin; capped by the device's
// free memory less a floor) becomes the L2 expert tier. LLAMADECODE_L2=0 disables.
void local_video_memory(uint64_t&budget,uint64_t&usage){
    budget=usage=0;IDXGIFactory4*factory=nullptr;
    if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4),reinterpret_cast<void**>(&factory)))||!factory)return;
    uint64_t best=0;
    for(UINT i=0;;++i){IDXGIAdapter1*a=nullptr;if(factory->EnumAdapters1(i,&a)==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 d{};a->GetDesc1(&d);IDXGIAdapter3*a3=nullptr;
        if(d.DedicatedVideoMemory>best&&SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3),reinterpret_cast<void**>(&a3)))){
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if(SUCCEEDED(a3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info))){best=d.DedicatedVideoMemory;budget=info.Budget;usage=info.CurrentUsage;}
            a3->Release();}
        a->Release();}
    factory->Release();
}
// Physical device memory from NVML (the driver's own count; WDDM/CUDA free
// double-counts the imported D3D12 banks). Selected by the job's GPU UUID.
bool nvml_free_bytes(uint64_t&free_bytes,uint64_t&total_bytes){
    free_bytes=total_bytes=0;
    static HMODULE nvml=LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!nvml)return false;
    struct memory{unsigned long long total,free,used;};
    using init_t=int(*)();using by_uuid_t=int(*)(const char*,void**);using info_t=int(*)(void*,memory*);
    static auto init=reinterpret_cast<init_t>(GetProcAddress(nvml,"nvmlInit_v2"));
    static auto by_uuid=reinterpret_cast<by_uuid_t>(GetProcAddress(nvml,"nvmlDeviceGetHandleByUUID"));
    static auto info=reinterpret_cast<info_t>(GetProcAddress(nvml,"nvmlDeviceGetMemoryInfo"));
    static const int ready=init?init():-1;
    const char*uuid=std::getenv("CUDA_VISIBLE_DEVICES");
    if(ready!=0||!by_uuid||!info||!uuid||std::strncmp(uuid,"GPU-",4))return false;
    void*device=nullptr;memory m{};
    if(by_uuid(uuid,&device)!=0||info(device,&m)!=0)return false;
    free_bytes=m.free;total_bytes=m.total;return true;
}
// Physical VRAM reserve kept free beside the L2 (LLAMADECODE_VRAM_RESERVE_MIB, default 128 MiB;
// owner 2026-10-30: 128 MiB reserve + 64 MiB growth margin, 8/8 runs at 50K-200K without a breach).
uint64_t decode_vram_reserve(){static const uint64_t r=[]{const char*e=std::getenv("LLAMADECODE_VRAM_RESERVE_MIB");
    const long v=e&&*e?std::strtol(e,nullptr,10):128; /* measured: no transient dip below the post-L2 steady free (5 ms NVML, 43K run) */return uint64_t(std::max<long>(128,std::min<long>(v,8192)))<<20;}();return r;}
// Decode-graph growth margin beside the reserve (LLAMADECODE_VRAM_MARGIN_MIB, default 64 MiB; the
// measured post-L2 transient is ~25-30 MiB, and 128+0 breached the reserve in every run).
// Owner 2026-10-29: the safety margin may be cut but not zeroed; reserve+margin >= 128 MiB.
uint64_t decode_vram_margin(){static const uint64_t r=[]{const char*e=std::getenv("LLAMADECODE_VRAM_MARGIN_MIB");
    const long v=e&&*e?std::strtol(e,nullptr,10):64;return uint64_t(std::max<long>(0,std::min<long>(v,8192)))<<20;}();return r;}
// Headroom measurement: a 5 ms NVML sampler records the minimum free VRAM
// (transient cuBLAS/graph/WDDM/other-process use) once the L2 is filled.
struct vram_sampler{
    std::atomic<bool>stop{false},started{false};std::atomic<uint64_t>min_free{~0ull},samples{0};std::thread worker;
    void start(){bool no=false;if(!started.compare_exchange_strong(no,true))return;
        worker=std::thread([this]{while(!stop.load()){uint64_t f=0,t=0;if(nvml_free_bytes(f,t)){uint64_t m=min_free.load();while(f<m&&!min_free.compare_exchange_weak(m,f)){}++samples;}
            std::this_thread::sleep_for(std::chrono::milliseconds(5));}});}
    ~vram_sampler(){stop.store(true);if(worker.joinable())worker.join();}
};
vram_sampler&decode_vram_sampler(){static vram_sampler s;return s;}
// Runtime guard: refuse loudly (graceful error) if physical free VRAM falls
// below the reserve while the decode-sized plan holds the L2.
void check_decode_vram_reserve(){
    uint64_t f=0,t=0;if(!nvml_free_bytes(f,t))return;
    auto&s=decode_vram_sampler();
    if(s.started.load()){
        // WDDM demotion watch: DXGI local budget/usage, non-local (shared) usage, system RAM.
        uint64_t budget=0,usage=0,shared=0;local_video_memory(budget,usage);
        {IDXGIFactory4*fa=nullptr;if(SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory4),reinterpret_cast<void**>(&fa)))&&fa){uint64_t best=0;
            for(UINT i=0;;++i){IDXGIAdapter1*a=nullptr;if(fa->EnumAdapters1(i,&a)==DXGI_ERROR_NOT_FOUND)break;DXGI_ADAPTER_DESC1 d{};a->GetDesc1(&d);IDXGIAdapter3*a3=nullptr;
                if(d.DedicatedVideoMemory>best&&SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3),reinterpret_cast<void**>(&a3)))){DXGI_QUERY_VIDEO_MEMORY_INFO nl{};
                    if(SUCCEEDED(a3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nl))){best=d.DedicatedVideoMemory;shared=nl.CurrentUsage;}a3->Release();}
                a->Release();}fa->Release();}}
        MEMORYSTATUSEX ms{};ms.dwLength=sizeof(ms);GlobalMemoryStatusEx(&ms);
        std::fprintf(stderr,"LLAMADECODE_VRAM_HEADROOM nvml_free_now=%llu min_free_since_l2=%llu samples=%llu reserve=%llu dxgi_local_budget=%llu dxgi_local_usage=%llu dxgi_shared_usage=%llu ram_avail=%llu\n",
            (unsigned long long)f,(unsigned long long)s.min_free.load(),(unsigned long long)s.samples.load(),(unsigned long long)decode_vram_reserve(),
            (unsigned long long)budget,(unsigned long long)usage,(unsigned long long)shared,(unsigned long long)ms.ullAvailPhys);}
    if(f<decode_vram_reserve()){std::fprintf(stderr,"LLAMADECODE_VRAM_RESERVE_BREACH nvml_free=%llu reserve=%llu\n",(unsigned long long)f,(unsigned long long)decode_vram_reserve());std::fflush(stderr);
        throw std::runtime_error("Physical free VRAM fell below the decode VRAM reserve");}
}
void grow_decode_l2(llama_dsv41_sweep&sweep,const llama_dsv41_fixed_storage*fixed,ggml_backend_t backend){
    static const bool on=[]{const char*e=std::getenv("LLAMADECODE_L2");return !(e&&e[0]=='0'&&!e[1]);}();
    if(!on||!fixed||!fixed->decode_compact)return;
    auto*provider=sweep.provider_for_external_decode();
    if(!provider->decode_layout()||provider->cache_stats().l2_slots)return;
    // Owner rule: keep a 2 GiB physical reserve (NVML free already excludes
    // Studio's idle context) plus a decode-graph growth margin.
    const uint64_t margin=decode_vram_margin(),floor=decode_vram_reserve();
    const uint64_t freed=fixed->full_graph_reserve+fixed->full_hidden_bytes;
    const uint64_t now=ggml_gallocr_get_buffer_size(fixed->graph.get(),0)+fixed->hidden_bytes;
    (void)freed;(void)now;uint64_t offer=~0ull;
    size_t free_bytes=0,total=0;ggml_backend_cuda_get_device_memory(0,&free_bytes,&total);(void)backend; // the job sees one device
    // WDDM: the authoritative no-demotion room is this process's local-segment
    // budget minus its current usage (DXGI), on the largest adapter (the 5090).
    // WDDM's process budget and CUDA's free count the imported D3D12 banks
    // twice (diagnostic only). The cap is NVML's physical free memory, and the
    // offer never exceeds what the decode-sized plan freed from the job's own
    // startup-admitted plan; no NVML reading means no L2.
    uint64_t budget=0,usage=0;local_video_memory(budget,usage);
    uint64_t nv_free=0,nv_total=0;const bool nv=nvml_free_bytes(nv_free,nv_total);
    // LLAMADECODE growable context cache: the plan keeps the next growth step
    // free beside the reserve (current size is already mapped, so not in nv_free).
    const uint64_t next_step=sweep.context_next_step_bytes();
    offer=nv?std::min<uint64_t>(offer,llama_dsv41_ctx_cache::l2_offer(nv_free,floor,margin,next_step)):0;
    uint32_t slots=0;std::string error;success(provider->grow_decode_cache(offer,&slots,error),error);
    if(slots)decode_vram_sampler().start();
    uint64_t nv_after=0,nv_total_after=0;(void)nvml_free_bytes(nv_after,nv_total_after);
    std::fprintf(stderr,"LLAMADECODE_L2_OFFER freed=%llu decode_now=%llu cuda_free=%llu cuda_total=%llu dxgi_budget=%llu dxgi_usage=%llu nvml_free=%llu nvml_total=%llu nvml_free_after=%llu offered=%llu slots=%u next_step_reserved=%llu\n",(unsigned long long)freed,
        (unsigned long long)now,(unsigned long long)free_bytes,(unsigned long long)total,(unsigned long long)budget,(unsigned long long)usage,
        (unsigned long long)nv_free,(unsigned long long)nv_total,(unsigned long long)nv_after,(unsigned long long)offer,slots,(unsigned long long)next_step);std::fflush(stderr);
}
// LLAMADECODE growable context cache during decode: map the step that covers
// `tokens` (it comes out of the reserved next step), then shrink the L2 so the
// following step is reserved again (L2 rebuilt smaller; its contents restart).
void ensure_decode_context(llama_dsv41_sweep&sweep,const llama_dsv41_fixed_storage*fixed,ggml_backend_t backend,uint32_t tokens){
    bool grew=false;std::string error;success(sweep.ensure_context(tokens,"decode",&grew,error),error);
    if(!grew)return;
    auto*provider=sweep.provider_for_external_decode();
    const uint32_t before=provider->cache_stats().l2_slots;if(!before)return;
    success(provider->release_decode_cache(error),error);
    grow_decode_l2(sweep,fixed,backend);
    std::fprintf(stderr,"LLAMADECODE_CTX_CACHE_L2_SHRINK tokens=%u l2_slots=%u->%u next_step_reserved=%llu\n",tokens,before,
        provider->cache_stats().l2_slots,(unsigned long long)sweep.context_next_step_bytes());std::fflush(stderr);
}
bool ds4_decode_enabled(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_DS4_DECODE");return e&&e[0]=='1'&&!e[1];}();return on;}
// LLAMADECODE append-via-decode: with the ds4 decode on, a live prefill of at
// most LLAMADECODE_APPEND_DECODE_MAX tokens (default 0 = off; opt-in 96) is appended
// one token at a time through the ds4 decode step instead of a full weight sweep.
uint64_t append_decode_max(){static const uint64_t n=[]{const char*e=std::getenv("LLAMADECODE_APPEND_DECODE_MAX");
    return e&&*e?uint64_t(std::max<long long>(0,std::min<long long>(std::strtoll(e,nullptr,10),4096))):uint64_t(0);}();return n;} /* E2c measured faster but non-exact; owner requires opt-in until attribution/fix. */
// LLAMADECODE decode cancel: the client creates LLAMADECODE_DECODE_CANCEL_FILE
// while a decode op runs; the loop stops at the next token boundary exactly as at
// max_tokens (the last sampled token is returned, not evaluated). Cleared at
// every decode start and after a cancel.
// LLAMADECODE multi-row append (opt-in, LLAMADECODE_APPEND_MULTIROW=1): an append of
// 2..LLAMADECODE_APPEND_DECODE_MAX tokens runs as ONE ds4-win layer-major prefill
// (ds4x_prefill, native ds41_graph_prefill on a small workspace) instead of one ds4
// decode step per token; the routed experts of the rows are fetched as one union per layer.
bool append_multirow_enabled(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_APPEND_MULTIROW");return e&&e[0]=='1'&&!e[1];}();return on;}
// LLAMADECODE keep-open: after a prefill weight sweep rebuilt the banks, the ds4
// decoder stays open and only the moved resident weights are re-registered
// (default ON). LLAMADECODE_DS4X_KEEP_OPEN=0 restores x_close + x_open.
bool ds4x_keep_open(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_DS4X_KEEP_OPEN");const bool v=llama_dsv41_ds4x_keep::keep_open_from_env(e);
    std::fprintf(stderr,"LLAMADECODE_DS4X_KEEP_OPEN enabled=%d default=on (=0 disables)\n",int(v));std::fflush(stderr);return v;}();return on;}
const std::wstring&decode_cancel_file(){static const std::wstring p=[]{const char*e=std::getenv("LLAMADECODE_DECODE_CANCEL_FILE");
    return e&&*e?std::filesystem::u8path(e).wstring():std::wstring();}();return p;}
bool decode_cancel_requested(){const auto&p=decode_cancel_file();return !p.empty()&&GetFileAttributesW(p.c_str())!=INVALID_FILE_ATTRIBUTES;}
void clear_decode_cancel(){const auto&p=decode_cancel_file();if(!p.empty())DeleteFileW(p.c_str());}
struct ds4_external {
    HMODULE module=nullptr;bool open=false;
    int(*x_open)(const char*,uint32_t,char*,size_t)=nullptr;const void*(*x_map)()=nullptr;
    int(*x_register)(uint64_t,uint64_t,void*)=nullptr;
    int(*x_bind)(void*const*,void*const*,void*const*,const uint32_t*,void*const*,void*const*)=nullptr;
    int(*x_bind_ex)(void*const*,void*const*,void*const*,const uint32_t*,void*const*,void*const*,uint32_t)=nullptr; // bf16 owners (optional export)
    void(*x_hook)(int(*)(void*,uint32_t,const int32_t*,uint32_t,int32_t*,const char**,const char**,const char**,uint32_t*),void*)=nullptr;
    void(*x_hook_l2)(int(*)(void*,uint32_t,const int32_t*,uint32_t,int32_t*,const char**,uint64_t*,uint32_t*,uint32_t*),void*)=nullptr; // optional export
    int(*x_decode)(int,uint32_t,const int32_t*,uint32_t,float*)=nullptr;void(*x_close)()=nullptr;
    int(*x_affinity)(float,uint32_t,uint32_t,const char*,int(*)(void*,uint32_t,uint8_t*,uint32_t),void*)=nullptr;
    int(*x_prefill_ws)(uint32_t,char*,size_t)=nullptr;int(*x_prefill)(const int32_t*,uint32_t,uint32_t,const int32_t*,uint32_t,float*)=nullptr;
    int(*x_replace)(uint64_t,uint64_t,void*)=nullptr;uint64_t(*x_drop)()=nullptr; // keep-open (optional exports)
    ds4_external_host_cache_v1 host_cache{};ds4_external_host_cache_v2 host_cache2{};bool have_host2=false;
    void(*x_host_timing)(double*,uint64_t*)=nullptr;
    uint64_t host_limit=0;
    void grow_host(){if(host_cache.resize){std::string error;
        success(provider->configure_host_cache(host_cache,host_limit,llama_dsv41_host_memory::runtime_min_free_bytes,error,have_host2?&host_cache2:nullptr),error);}}
    uint32_t prefill_rows=0; // multi-row workspace rows (0 = per-token appends)
    llama_dsv41_weight_provider*provider=nullptr;uint64_t registered=0,registered_bytes=0;
    using resident=llama_dsv41_ds4x_keep::resident;std::vector<resident>table; // what ds4 holds registered
    // Per-turn handoff timing (decode-op start to the first token + L2), one line per op.
    struct handoff_times{clock_type::time_point op_started{};double inspect_ms=0,compact_ms=0,layout_ms=0,open_ms=0,register_ms=0,bind_ms=0;
        bool rebuilt=false,reopened=false,kept=false;const char*mode="none";uint64_t moved=0,unchanged=0;}handoff;
    void begin_op(clock_type::time_point started,double inspect_ms){handoff=handoff_times{};handoff.op_started=started;handoff.inspect_ms=inspect_ms;}
    std::vector<resident>resident_now(){std::vector<resident>now;std::string error;
        success(provider->decode_resident_tensors([&](uint64_t offset,uint64_t bytes,void*device){now.push_back({offset,bytes,device});return true;},error),error);return now;}
    // After a bank rebuild with the decoder kept open: move only what moved.
    void reregister(){
        const auto t0=clock_type::now();auto now=resident_now();
        const auto plan=llama_dsv41_ds4x_keep::reregister_plan(table,now);
        if(!plan.delta)(void)x_drop();   // layout changed: drop every range, register the new set
        for(const auto i:plan.moved){const auto&r=now[i];
            require((plan.delta?x_replace(r.offset,r.bytes,r.device):x_register(r.offset,r.bytes,r.device))==1,"ds4x refused a moved resident tensor");}
        uint64_t bytes=0;for(const auto&r:now)bytes+=r.bytes;
        const uint64_t moved=plan.moved.size(),unchanged=plan.unchanged;handoff.mode=plan.delta?"delta":"full";
        table=std::move(now);registered=table.size();registered_bytes=bytes;handoff.moved=moved;handoff.unchanged=unchanged;handoff.register_ms=ms(t0);
        std::fprintf(stderr,"LLAMADECODE_DS4X_REREGISTER mode=%s moved=%llu unchanged=%llu tensors=%llu bytes=%llu ms=%.3f\n",handoff.mode,(unsigned long long)moved,
            (unsigned long long)unchanged,(unsigned long long)registered,(unsigned long long)bytes,handoff.register_ms);std::fflush(stderr);
    }
    void report_handoff(uint32_t position,double first_token_ms,double l2_ms){
        const auto l2=provider?provider->cache_stats().l2_slots:0u;
        std::fprintf(stderr,"LLAMADECODE_DS4X_HANDOFF keep_open=%d layout_rebuilt=%d reopened=%d kept_open=%d register_mode=%s moved=%llu unchanged=%llu inspect_ms=%.3f compact_ms=%.3f layout_ms=%.3f open_ms=%.3f register_ms=%.3f bind_ms=%.3f first_token_ms=%.3f l2_ms=%.3f l2_slots=%u l2_contents=empty_victim_tier total_ms=%.3f position=%u\n",
            int(ds4x_keep_open()),int(handoff.rebuilt),int(handoff.reopened),int(handoff.kept),handoff.mode,(unsigned long long)handoff.moved,(unsigned long long)handoff.unchanged,
            handoff.inspect_ms,handoff.compact_ms,handoff.layout_ms,handoff.open_ms,handoff.register_ms,handoff.bind_ms,first_token_ms,l2_ms,l2,ms(handoff.op_started),position);std::fflush(stderr);
    }
    float quality=-1.f; // live quality-speed mean (session op); <0 = LLAMADECODE_DS4_BIAS / off
    static int residency_hook(void*user,uint32_t layer,uint8_t*tier,uint32_t experts){
        auto*self=static_cast<ds4_external*>(user);return self->provider&&self->provider->decode_residency(layer,tier,experts)?1:0;}
    static int expert_hook(void*user,uint32_t layer,const int32_t*ids,uint32_t n,int32_t*slots,const char**g,const char**u,const char**d,uint32_t*count){
        auto*self=static_cast<ds4_external*>(user);std::string why;
        if(!self->provider||!self->provider->external_expert_slots(layer,ids,n,slots,g,u,d,count,nullptr,why)){
            std::fprintf(stderr,"LLAMADECODE_DS4X_SLOTS_FAILED layer=%u reason=%s\n",layer,why.c_str());return 0;}
        return 1;
    }
    static int expert_hook_l2(void*user,uint32_t layer,const int32_t*ids,uint32_t n,int32_t*slots,const char**bases,uint64_t*l2_stride,uint32_t*l1,uint32_t*count){
        auto*self=static_cast<ds4_external*>(user);std::string why;
        if(!self->provider||!self->provider->external_expert_slots_l2(layer,ids,n,slots,bases,l2_stride,l1,count,nullptr,why)){
            std::fprintf(stderr,"LLAMADECODE_DS4X_SLOTS_FAILED layer=%u direct_l2=1 reason=%s\n",layer,why.c_str());return 0;}
        return 1;
    }
    template<class F>void bind(F&f,const char*name){f=reinterpret_cast<F>(GetProcAddress(module,name));require(f,"ds4-decode.dll export missing");}
    void load(){
        if(module)return;
        auto dll=executable().parent_path()/L"ds4-decode.dll";
        module=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        require(module,"Cannot load ds4-decode.dll beside the producer");
        bind(x_open,"ds4x_open");bind(x_map,"ds4x_model_map");bind(x_register,"ds4x_register_weight");bind(x_bind,"ds4x_bind_state");
        bind(x_hook,"ds4x_set_expert_hook");bind(x_decode,"ds4x_decode");bind(x_close,"ds4x_close");bind(x_affinity,"ds4x_set_affinity");
        x_host_timing=reinterpret_cast<decltype(x_host_timing)>(GetProcAddress(module,"ds4x_host_timing_v1"));
        const char*ram=std::getenv("LLAMADECODE_HOST_CACHE_MB");
        if(ram&&*ram&&std::strcmp(ram,"0")){
            int(*api)(ds4_external_host_cache_v1*)=nullptr;bind(api,"ds4x_host_cache_v1");
            require(api(&host_cache)==1,"ds4-decode RAM cache ABI refused");
            int(*api2)(ds4_external_host_cache_v2*)=reinterpret_cast<int(*)(ds4_external_host_cache_v2*)>(GetProcAddress(module,"ds4x_host_cache_v2"));
            have_host2=api2&&api2(&host_cache2)==1;
            const char*mode=std::getenv("LLAMADECODE_HOST_MODE");
            require(!mode||!std::strcmp(mode,"upload")||!std::strcmp(mode,"cpu")||!std::strcmp(mode,"mix"),"LLAMADECODE_HOST_MODE must be upload, cpu or mix");
            if(mode&&std::strcmp(mode,"upload")){
                int(*cpu)(ds4_external_dot)=nullptr;bind(cpu,"ds4x_host_cpu_v1");
                // The GPU producer does not otherwise initialize the CPU
                // backend. Its SIMD dot uses the CPU FP16 lookup table.
                ggml_cpu_init();
                const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot;
                std::array<uint8_t,144>qw{};std::array<uint8_t,292>qx{};
                const uint16_t half_one=ggml_fp32_to_fp16(1.0f);const float one=1.0f;
                std::memcpy(qw.data(),&half_one,2);std::memcpy(qx.data(),&one,4);
                for(size_t i=4;i<8;++i)qw[i]=1;for(size_t i=12;i<16;++i)qw[i]=1;
                std::fill(qw.begin()+16,qw.end(),uint8_t(0x11));
                std::fill(qx.begin()+4,qx.begin()+260,uint8_t(1));
                for(size_t i=260;i<292;i+=2){const int16_t sum=16;std::memcpy(qx.data()+i,&sum,2);}
                float known=0;dot(256,&known,0,qw.data(),0,qx.data(),0,1);
                require(std::isfinite(known)&&std::abs(known-256.0f)<0.001f,"CPU Q4_K dot known-answer check failed");
                require(cpu(dot)==1,"CPU expert kernel registration failed");
                std::fprintf(stderr,"LLAMADECODE_HOST_CPU_ISA avx2=%d avx512=%d\n",ggml_cpu_has_avx2(),ggml_cpu_has_avx512());
            }
        }
        if(append_multirow_enabled()){bind(x_prefill_ws,"ds4x_prefill_workspace");bind(x_prefill,"ds4x_prefill");}
        x_bind_ex=reinterpret_cast<decltype(x_bind_ex)>(GetProcAddress(module,"ds4x_bind_state_ex"));
        x_replace=reinterpret_cast<decltype(x_replace)>(GetProcAddress(module,"ds4x_replace_weight"));
        x_drop=reinterpret_cast<decltype(x_drop)>(GetProcAddress(module,"ds4x_drop_weights"));
        x_hook_l2=reinterpret_cast<decltype(x_hook_l2)>(GetProcAddress(module,"ds4x_set_expert_hook_l2"));
        if(ds4x_keep_open()&&(!x_replace||!x_drop)){std::fprintf(stderr,"LLAMADECODE_DS4X_KEEP_OPEN unavailable reason=ds4-decode.dll lacks ds4x_replace_weight/ds4x_drop_weights; x_close+x_open\n");std::fflush(stderr);}
    }
    bool can_keep()const{return ds4x_keep_open()&&x_replace&&x_drop;}
    // (Re)prepare on the current owner: decode layout, resident weights
    // registered at their device addresses, state views, expert hook.
    void prepare(llama_dsv41_sweep&sweep,const llama_dsv41_weight_provider::config&weights,uint32_t context){
        load();std::string error;MEMORYSTATUSEX before{};before.dwLength=sizeof(before);
        if(host_cache.resize)require(GlobalMemoryStatusEx(&before),"Prefill release RAM query failed");
        auto t0=clock_type::now();success(sweep.compact_for_external_decode(error,bool(host_cache.resize)),error);handoff.compact_ms=ms(t0);
        provider=sweep.provider_for_external_decode();
        host_limit=weights.job_host_bytes;
        if(host_cache.resize){
            success(provider->release_prefill_rows(error),error);
            MEMORYSTATUSEX after{};after.dwLength=sizeof(after);require(GlobalMemoryStatusEx(&after),"Decode RAM query failed");
            std::fprintf(stderr,"LLAMADECODE_PREFILL_HOST_RELEASE available_before=%llu available_after=%llu delta=%lld frontier_certificate_kept=1\n",
                (unsigned long long)before.ullAvailPhys,(unsigned long long)after.ullAvailPhys,(long long)after.ullAvailPhys-(long long)before.ullAvailPhys);
        }
        if(!provider->decode_layout()){
            handoff.rebuilt=true;
            // Banks were rebuilt by a prefill pass. Old path: close and reopen (re-register
            // everything). Keep-open: the decoder stays open; reregister() moves what moved.
            if(open&&!can_keep()){x_close();open=false;table.clear();handoff.reopened=true;}
            auto decode_weights=weights;decode_weights.decode_layout=true;
            t0=clock_type::now();success(provider->prepare_external_decode(decode_weights,error),error);handoff.layout_ms=ms(t0);
            if(open){handoff.kept=true;
                std::fprintf(stderr,"LLAMADECODE_DS4X_KEPT_OPEN ctx=%u reason=bank rebuild (prefill sweep); no x_close/x_open\n",context);std::fflush(stderr);
                reregister();}
        }
        if(!open){
            size_t free_before=0,free_after=0,total=0;ggml_backend_cuda_get_device_memory(0,&free_before,&total);
            t0=clock_type::now();
            char why[512]{};require(x_open(weights.model_path.c_str(),context,why,sizeof(why))==1,why[0]?why:"ds4x_open failed");open=true;
            ggml_backend_cuda_get_device_memory(0,&free_after,&total);
            std::fprintf(stderr,"LLAMADECODE_DS4X_DEVICE_DELTA bytes=%lld free_before=%llu free_after=%llu\n",(long long)free_before-(long long)free_after,
                (unsigned long long)free_before,(unsigned long long)free_after);std::fflush(stderr);
            registered=registered_bytes=0;prefill_rows=0;
            if(append_multirow_enabled()&&append_decode_max()>=2){
                // The native prefill workspace, sized to the largest small append (allocated before the L2 offer).
                char werr[512]{};const uint32_t rows=uint32_t(append_decode_max());
                require(x_prefill_ws(rows,werr,sizeof(werr))==1,werr[0]?werr:"ds4x_prefill_workspace failed");prefill_rows=rows;
                size_t free_ws=0;ggml_backend_cuda_get_device_memory(0,&free_ws,&total);
                std::fprintf(stderr,"LLAMADECODE_DS4X_PREFILL_ROWS rows=%u workspace_bytes=%lld\n",rows,(long long)free_after-(long long)free_ws);std::fflush(stderr);
            }
            handoff.open_ms=ms(t0);t0=clock_type::now();table.clear();
            success(provider->decode_resident_tensors([&](uint64_t offset,uint64_t bytes,void*device){
                if(!x_register(offset,bytes,device))return false;++registered;registered_bytes+=bytes;table.push_back({offset,bytes,device});return true;},error),error);
            handoff.register_ms=ms(t0);handoff.mode="open";handoff.moved=registered;
            std::fprintf(stderr,"LLAMADECODE_DS4X_OPENED ctx=%u open_ms=%.3f register_ms=%.3f tensors=%llu\n",context,handoff.open_ms,handoff.register_ms,(unsigned long long)registered);std::fflush(stderr);
        }
        t0=clock_type::now();
        llama_dsv41_sweep::device_state state;success(sweep.device_state_views(state,error),error);
        if(state.bf16){
            require(x_bind_ex,"ctx-cache-precision bf16 needs a ds4-decode.dll with ds4x_bind_state_ex (BF16 owner caches)");
            require(!append_multirow_enabled(),"ctx-cache-precision bf16 does not support LLAMADECODE_APPEND_MULTIROW=1 (ds4 multi-row prefill reads F32 owners)");
            require(x_bind_ex(state.raw.data(),state.compressed.data(),state.index.data(),state.rows.data(),state.pool_kv.data(),state.pool_score.data(),1)==1,
                "ds4x_bind_state_ex refused the canonical BF16 state");
        }else require(x_bind(state.raw.data(),state.compressed.data(),state.index.data(),state.rows.data(),state.pool_kv.data(),state.pool_score.data())==1,
            "ds4x_bind_state refused the canonical state");
        x_hook(&expert_hook,this);
        // Packed FP4 experts: the decoder reads second-tier (L2) hits in place.
        // LLAMADECODE_DIRECT_L2=0 keeps the copy-into-L1 path.
        if(x_hook_l2){static const bool direct=[]{const char*e=std::getenv("LLAMADECODE_DIRECT_L2");return !(e&&e[0]=='0'&&!e[1]);}();
            x_hook_l2(direct?&expert_hook_l2:nullptr,this);}
        // B5: LLAMADECODE_DS4_BIAS=<quality-speed mean> enables ds4-win's own
        // cache-affinity bias on this cache (selection only). Spread file and
        // warmup/ramp follow the native engine (DS4_CACHE_AFFINITY_*).
        const char*b=std::getenv("LLAMADECODE_DS4_BIAS");
        if(quality>=0.f||b){
            const float mean=quality>=0.f?quality:std::strtof(b,nullptr);const char*spread=std::getenv("LLAMADECODE_DS4_SPREAD");
            const char*w=std::getenv("DS4_CACHE_AFFINITY_WARMUP_TOKENS");const char*r=std::getenv("DS4_CACHE_AFFINITY_RAMP_TOKENS");
            require(x_affinity(mean,w?uint32_t(std::strtoul(w,nullptr,10)):100u,r?uint32_t(std::strtoul(r,nullptr,10)):8u,spread,&residency_hook,this)==1,
                "ds4x_set_affinity refused the bias configuration");
        }
        handoff.bind_ms=ms(t0);
        std::fprintf(stderr,"LLAMADECODE_DS4X_READY registered_tensors=%llu registered_bytes=%llu\n",(unsigned long long)registered,(unsigned long long)registered_bytes);
    }
    void step(llama_dsv41_sweep&sweep,ggml_backend_t backend,int32_t token,uint32_t position){
        std::string error;std::array<int32_t,3>tail{};success(sweep.engram_tail(tail,error),error);
        std::vector<float>logits(129280);
        ggml_backend_synchronize(backend);
        if(position%16==0)check_decode_vram_reserve();
        double f0[4]{};uint64_t r0=0,b0=0;wf_extent_fetch_stats(f0,&r0,&b0);
        const auto before=provider->cache_stats();const auto started=std::chrono::steady_clock::now();
        double cpu0[4]{};uint64_t count0[2]{};if(x_host_timing)x_host_timing(cpu0,count0);
        require(x_decode(token,position,tail.data(),3,logits.data())==1,"ds4-win decode step failed");
        const double decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        success(sweep.commit_external_token(token,logits,error),error);
        const auto after=provider->cache_stats();double f1[4]{};uint64_t r1=0,b1=0;wf_extent_fetch_stats(f1,&r1,&b1);
        double cpu1[4]{};uint64_t count1[2]{};if(x_host_timing)x_host_timing(cpu1,count1);
        std::fprintf(stderr,"LLAMADECODE_TIER_TIME position=%u hook_ms=%.4f store_ms=%.4f victim_sync_ms=%.4f upload_ms=%.4f overlap_ms=%.4f async_hits=%llu direct_hits=%llu dropped=%llu residency_ms=%.4f cpu_ms=%.4f cpu_wait_ms=%.4f cpu_transfer_ms=%.4f cpu_merge_ms=%.4f cpu_experts=%llu\n",position,
            after.hook_ms-before.hook_ms,after.host_store_ms-before.host_store_ms,after.host_sync_ms-before.host_sync_ms,
            after.host_upload_ms-before.host_upload_ms,after.host_overlap_ms-before.host_overlap_ms,
            (unsigned long long)(after.host_async_hits-before.host_async_hits),(unsigned long long)(after.host_direct_hits-before.host_direct_hits),(unsigned long long)(after.host_dropped-before.host_dropped),after.residency_ms-before.residency_ms,
            cpu1[0]-cpu0[0],cpu1[1]-cpu0[1],cpu1[2]-cpu0[2],cpu1[3]-cpu0[3],(unsigned long long)(count1[0]-count0[0]));
        // Opt-in bounded correctness trace. Original logits, sampled once per
        // 32 positions; the caller supplies a scratch path and compares only
        // identical token prefixes. No extra model evaluation or routing.
        const char*trace=std::getenv("LLAMADECODE_LOGITS_TRACE");
        if(trace&&*trace&&position%32==0){
            auto*f=std::fopen(trace,"ab");require(f,"Cannot open decode logits trace");
            const uint32_t header[3]={position,uint32_t(token),uint32_t(logits.size())};
            const bool ok=std::fwrite(header,sizeof(header),1,f)==1&&std::fwrite(logits.data(),sizeof(float),logits.size(),f)==logits.size();
            std::fclose(f);require(ok,"Decode logits trace write failed");
        }
        std::fprintf(stderr,"LLAMADECODE_DS4X_TOKEN position=%u decode_ms=%.3f ds_bytes=%llu ds_ms=%.3f l1_hits=%llu l2_hits=%llu misses=%llu l1_slots=%u l2_slots=%u ds_submit_ms=%.3f ds_fence_ms=%.3f ds_verify_ms=%.3f ds_cuda_ms=%.3f ds_requests=%llu ds_batches=%llu assign_ms=%.3f copy_sync_ms=%.3f l2_stored=%llu host_hits=%llu host_stored=%llu host_slots=%u host_bytes=%llu\n",position,decode_ms,
            (unsigned long long)(after.fetched_bytes-before.fetched_bytes),after.fetch_ms-before.fetch_ms,(unsigned long long)(after.hits-before.hits),
            (unsigned long long)(after.l2_hits-before.l2_hits),(unsigned long long)(after.misses-before.misses),after.slots,after.l2_slots,
            f1[0]-f0[0],f1[1]-f0[1],f1[2]-f0[2],f1[3]-f0[3],(unsigned long long)(r1-r0),(unsigned long long)(b1-b0),
            after.assign_ms-before.assign_ms,after.copy_sync_ms-before.copy_sync_ms,(unsigned long long)(after.l2_stored-before.l2_stored),
            (unsigned long long)(after.host_hits-before.host_hits),(unsigned long long)(after.host_stored-before.host_stored),after.host_slots,(unsigned long long)after.host_bytes);
    }
    // Multi-row append: `count` tokens at `position` in one ds4 prefill; the producer
    // commits them with the frontier logits (the per-token path's final state).
    void prefill_rows_step(llama_dsv41_sweep&sweep,ggml_backend_t backend,const std::vector<int32_t>&tokens,uint32_t position){
        std::string error;std::array<int32_t,3>tail{};success(sweep.engram_tail(tail,error),error);
        std::vector<float>logits(129280);
        ggml_backend_synchronize(backend);check_decode_vram_reserve();
        double f0[4]{};uint64_t r0=0,b0=0;wf_extent_fetch_stats(f0,&r0,&b0);
        const auto before=provider->cache_stats();const auto started=std::chrono::steady_clock::now();
        require(x_prefill(tokens.data(),uint32_t(tokens.size()),position,tail.data(),3,logits.data())==1,"ds4-win multi-row append failed");
        const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        success(sweep.commit_external_tokens(tokens.data(),tokens.size(),logits,error),error);
        const auto after=provider->cache_stats();double f1[4]{};uint64_t r1=0,b1=0;wf_extent_fetch_stats(f1,&r1,&b1);
        std::fprintf(stderr,"LLAMADECODE_DS4X_PREFILL rows=%zu position=%u prefill_ms=%.3f ds_bytes=%llu ds_ms=%.3f l1_hits=%llu l2_hits=%llu misses=%llu ds_requests=%llu ds_batches=%llu\n",
            tokens.size(),position,prefill_ms,(unsigned long long)(after.fetched_bytes-before.fetched_bytes),after.fetch_ms-before.fetch_ms,
            (unsigned long long)(after.hits-before.hits),(unsigned long long)(after.l2_hits-before.l2_hits),(unsigned long long)(after.misses-before.misses),
            (unsigned long long)(r1-r0),(unsigned long long)(b1-b0));
    }
    // The DLL stays mapped until process exit (never FreeLibrary): no ds4 code may
    // be unmapped under a thread or CUDA callback it started.
    ~ds4_external(){if(open&&x_close){x_hook(nullptr,nullptr);x_close();open=false;std::fprintf(stderr,"LLAMADECODE_DS4X_CLOSED\n");std::fflush(stderr);}}
};
// LLAMADECODE: decode steps use the routed-expert layout (B1). The explicit
// diagnostic LLAMADECODE_DECODE_LAYOUT=0 runs them on the whole-layer B0 layout.
bool routed_decode_layout(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_DECODE_LAYOUT");
    if(e&&e[0]=='0'&&!e[1]){std::fprintf(stderr,"LLAMADECODE_DECODE_LAYOUT=0 whole-layer B0 decode (diagnostic)\n");return false;}return true;}();return on;}
void run_live_session(std::unique_ptr<llama_dsv41_sweep>&sweep,
        const llama_dsv41_weight_provider::config&weights,const llama_dsv41_sweep::config&options,
        const std::string&session,uint64_t timeout_ms,uint32_t capacity,HANDLE output,uint64_t owner_epoch=0){
    using namespace llama_dsv41;using dsv41_live::checkpoint;
    require(sweep&&sweep->ready()&&options.fixed&&options.stateful_chunks,
        "Persistent session requires a completed fixed mode1/3 stateful owner");
    require(capacity&&capacity<=llama_dsv41_prefill::max_sweep_tokens&&options.fixed->tokens==capacity,"Session chunk reservation differs from fixed owner");
    const auto*model_sha=weights.verified_sha256.c_str();
    session_pipe io(timeout_ms,output);dsv41_live::sequence_guard guard(session,model_sha,owner_epoch);
    auto send=[&](json value){if(owner_epoch)value["owner_epoch"]=owner_epoch;io.send(value);};
    const uint64_t session_route=std::stoull(session.substr(0,16),nullptr,16),model_route=std::stoull(std::string(model_sha).substr(0,16),nullptr,16);
    require(session_route&&model_route,"Session routing fingerprint must be nonzero");
    checkpoint current,remote;bool have_remote=false,initial_sent=false,awaiting_ack=false,device_suspended=false;
    ds4_external ds4x; // LLAMADECODE ds4-decode (LLAMADECODE_DS4_DECODE=1)
    studio_generation generation;
    // One ds4 single-token step appending `token` at frontier+index (prepare at
    // index 0); shared by decode, decode_forced and the small-append prefill.
    auto ds4_step=[&](int32_t token,uint64_t index){
        if(index==0)ds4x.prepare(*sweep,weights,uint32_t(options.context));
        if(generation.active){
            generation.refresh();const float bias=generation.bias();
            // Manual changes and phase transitions take effect at this forward
            // token. No restarted warmup/ramp may defer a live slider update.
            if(index==0||bias!=ds4x.quality){
                const char*spread=std::getenv("LLAMADECODE_DS4_SPREAD");
                require(ds4x.x_affinity(bias,0,0,spread,&ds4_external::residency_hook,&ds4x)==1,"Live bias refused");
                ds4x.quality=bias;
                std::fprintf(stderr,"LLAMADECODE_LIVE_BIAS value=%.6g position=%u\n",bias,current.frontier.position+uint32_t(index));std::fflush(stderr);
            }
        }
        ensure_decode_context(*sweep,options.fixed.get(),weights.backend,current.frontier.position+uint32_t(index)+1);
        const auto t_token=clock_type::now();
        ds4x.step(*sweep,weights.backend,token,current.frontier.position+uint32_t(index));
        if(index==0){const double token_ms=ms(t_token);const auto t_l2=clock_type::now();grow_decode_l2(*sweep,options.fixed.get(),weights.backend);
            ds4x.grow_host(); // VRAM L2 is allocated before admitting the third tier
            ds4x.report_handoff(current.frontier.position,token_ms,ms(t_l2));}
        else if(index%16==0)check_decode_vram_reserve();
    };
    std::unique_ptr<llama_dsv41_sweep::rewind_anchor> rewind_anchor;
    checkpoint rewind_identity;
    // Incremental owner verify (default): tokens/logits from the payload as
    // before; owner SHA-256s from the sweep, reading only rows appended since
    // the last verified digest. `expect` (the checkpoint the caller compares
    // against) turns a mismatch into a full re-hash before it is reported.
    // LLAMADECODE_OWNER_VERIFY=full: the previous full payload capture.
    auto inspect=[&](const checkpoint*expect=nullptr){checkpoint c;std::string error;const bool incremental=owner_verify_incremental();
        success(sweep->inspect_local([&](wire_payload_reader&r,std::string&why){try{c=dsv41_live::capture(r,[]{return should_stop();},!incremental);return true;}catch(const std::exception&e){why=e.what();return false;}},error),error);
        if(incremental){
            const bool same=expect&&expect->frontier.context==c.frontier.context&&expect->frontier.position==c.frontier.position&&expect->frontier.mode==c.frontier.mode;
            success(sweep->owner_digests(c.owners,same?&expect->owners:nullptr,error),error);c=checkpoint::from_json(c.describe());
        }
        return c;};
    current=inspect();
    send({{"status","ready"},{"session",session},{"model_sha256",model_sha},{"next_sequence",guard.expected()},
        {"checkpoint",current.describe()},{"maximum_prefill_tokens",std::min(capacity,llama_dsv41_prefill::legacy_sweep_tokens)},{"maximum_prefill_sweep_tokens",capacity},{"maximum_decode_tokens",llama_dsv41::decode_delta_chunk_tokens},{"maximum_decode_delta_tokens",llama_dsv41::decode_delta_max_tokens},{"control_host_bytes",64*1024*1024},
        {"generation_control_version",1},{"local_snapshot_version",1},{"vision_prefill_version",1},{"yoco_keep",options.yoco_keep},{"yoco_state_version",options.yoco_remote_state_version},{"transport",options.yoco_keep&&!llama_dsv41_yoco::remote_compatible(options.yoco_keep,options.yoco_remote_state_version)?"local bounded state; remote export disabled":"private owned pipes; existing native payload and artifact v2.1 delta"}});
    for(;;){
        io.renew();auto command=dsv41_live::parse_control(io.line());guard.accept(command);if(owner_epoch)command.erase("owner_epoch");
        const auto operation=command.at("operation").get<std::string>();const auto sequence=dsv41_live::u64(command.at("sequence"));
        producer_abort::sequence.store(sequence);   // an abort during this operation names its sequence
        const auto started=clock_type::now();
        auto reply=[&](json extra){extra["status"]="complete";extra["sequence"]=sequence;extra["session"]=session;extra["model_sha256"]=model_sha;extra["checkpoint"]=current.describe();send(extra);};
        auto same_base=[&](){const auto base=checkpoint::from_json(command.at("base"));require(base.describe()==current.describe(),"Session base token/position/owner hashes differ");};
        if(operation=="close"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation"});reply({{"closed",true}});return;
        }else if(operation=="suspend"){
            // DS4COMBO fast switch: give back VRAM, DS heaps and carry RAM; keep
            // the process, CUDA context, model lease and plans. The frontier is
            // dropped: only resume (with a new native base) or close may follow.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base"});same_base();
            require(!device_suspended&&have_remote&&!awaiting_ack&&remote.describe()==current.describe(),"Suspend requires the exported frontier to be acknowledged by the peer");
            std::string error;success(sweep->suspend_device(error),error);device_suspended=true;
            size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(weights.backend),&free,&total);
            reply({{"suspended",true},{"device_free_bytes",free},{"device_total_bytes",total},{"suspend_ms",ms(started)}});
            continue;
        }else if(operation=="resume"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","native_base_contract_sha256","tokens"});
            require(device_suspended,"Resume requires a suspended session");
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=capacity,"Resume prefill token count exceeds reserved chunk");
            std::vector<int32_t>tokens;tokens.reserve(ids.size());for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Resume token out of vocabulary");tokens.push_back(int32_t(n));}
            std::string error;success(sweep->resume_device(error),error);const double resume_ms=ms(started);
            // The new base follows this command on the same pipe: contract line + payload.
            combo_base_input incoming(io,session,command.at("native_base_contract_sha256").get<std::string>(),owner_epoch,uint32_t(options.context),weights.verified_sha256);
            require(tokens.size()<uint64_t(options.context)-incoming.base.frontier.position,"Resume prefill exceeds context capacity");
            const double base_ms=ms(started)-resume_ms;
            auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
            next_options.yoco_logical_begin=next_options.yoco_logical_end=0;next_options.carry.tokens=uint32_t(tokens.size());if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=uint32_t(tokens.size());
            success(sweep->open(next_weights,next_options,error,previous.get(),nullptr,&incoming.base),error);incoming.finish();previous.reset();
            const double admission_ms=ms(started)-resume_ms-base_ms;
            success(sweep->run(tokens,[]{return should_stop();},error),error);const auto timing=sweep->timing();
            current=inspect();require(current.frontier.position==incoming.base.frontier.position+tokens.size(),"Resume prefill position did not advance exactly");
            have_remote=false;initial_sent=false;awaiting_ack=false;device_suspended=false;remote=checkpoint{};
            std::fprintf(stderr,"DS4COMBO_RESUME_COMPLETE resume_ms=%.3f base_import_ms=%.3f admission_ms=%.3f prompt_ms=%.3f tokens=%zu\n",resume_ms,base_ms,admission_ms,timing.prompt_processing_ms,tokens.size());
            reply({{"peer_synchronized",false},{"prompt_tokens",tokens.size()},{"resume_ms",resume_ms},{"base_import_ms",base_ms},{"admission_ms",admission_ms},
                {"prompt_processing_ms",timing.prompt_processing_ms},{"export_preparation_ms",timing.export_preparation_ms}});
            continue;
        }else if(device_suspended){
            require(false,"Suspended session accepts only resume or close");
        }else if(operation=="generation_start"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","settings"});
            generation.begin(command.at("settings"));
            reply({{"generation_control_version",1},{"revision",generation.settings.at("revision")},{"applies","next token boundary"}});
        }else if(operation=="quality_speed"){
            // LLAMADECODE B5: the live quality-speed control (ds4-win /v1/quality-speed
            // semantics: 0 = max quality, 1.36125 = max speed). Applies from the next
            // decode request; the bias keeps ds4-win's warmup/ramp per request.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","value"});
            const auto&v=command.at("value");require(v.is_number(),"Quality-speed value must be a number");
            const double value=v.get<double>();require(value>=0.0&&value<=1.36125,"Quality-speed value outside 0..1.36125");
            ds4x.quality=float(value);
            std::fprintf(stderr,"LLAMADECODE_QUALITY_SPEED value=%.6g applies=next_decode\n",value);std::fflush(stderr);
            reply({{"supported",ds4_decode_enabled()},{"value",value},{"applies","next decode request"},{"scope","session"}});
        }else if(operation=="checkpoint"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation"});
            require(inspect(&current).describe()==current.describe(),"Session idle owner changed");reply({{"peer_synchronized",have_remote&&!awaiting_ack&&remote.describe()==current.describe()}});
        }else if(operation=="save_rewind"||operation=="rewind"){
            dsv41_live::keys(command,operation=="save_rewind"?
                std::initializer_list<const char*>{"session","model_sha256","sequence","operation","base"}:
                std::initializer_list<const char*>{"session","model_sha256","sequence","operation","base","target"});same_base();
            require(have_remote&&!awaiting_ack&&remote.describe()==current.describe(),"Rewind operation before peer synchronization");
            require(inspect(&current).describe()==current.describe(),"Rewind current owner changed");std::string error;
            if(operation=="save_rewind"){
                rewind_anchor.reset(new llama_dsv41_sweep::rewind_anchor);
                success(sweep->capture_rewind_anchor(*rewind_anchor,error),error);rewind_identity=current;
            }else{
                require(bool(rewind_anchor)&&checkpoint::from_json(command.at("target")).describe()==rewind_identity.describe(),
                    "Rewind target is not this session's saved checkpoint");
                success(sweep->restore_rewind_anchor(*rewind_anchor,current.owners,error),error);
                current=inspect();require(current.describe()==rewind_identity.describe(),"Restored rewind checkpoint differs");
                awaiting_ack=true;
            }
            reply({{"peer_synchronized",!awaiting_ack}});
        }else if(operation=="export_local"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base"});same_base();
            require(!awaiting_ack,"Local snapshot while a peer transfer awaits acknowledgement");
            std::string error;uint64_t sent=0;std::array<unsigned char,65536>page{};
            success(sweep->inspect_local([&](wire_payload_reader&r,std::string&why){try{
                require(dsv41_live::capture(r,[]{return should_stop();}).describe()==current.describe(),"Local snapshot checkpoint changed");
                send({{"status","payload"},{"sequence",sequence},{"session",session},{"model_sha256",model_sha},
                    {"kind","local_snapshot_v1"},{"bytes",r.bytes()},{"base",nullptr},{"target",current.describe()},
                    {"yoco_keep",options.yoco_keep},{"cache_precision",options.compressed_cache_bf16?"bf16":"f32"}});
                while(sent<r.bytes()){const size_t n=size_t(std::min<uint64_t>(page.size(),r.bytes()-sent));
                    success(r.read(sent,page.data(),n,why),why);io.write(page.data(),n);sent+=n;}
                return true;
            }catch(const std::exception&e){why=e.what();return false;}},error),error);
            std::fprintf(stderr,"LLAMADECODE_LOCAL_SNAPSHOT yoco_keep=%u bytes=%llu milliseconds=%.3f\n",options.yoco_keep,(unsigned long long)sent,ms(started));
        }else if(operation=="export_initial"||operation=="export_delta"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base"});same_base();
            require(llama_dsv41_yoco::remote_compatible(options.yoco_keep,options.yoco_remote_state_version),"YOCO remote export requires matching keep/version peer negotiation");
            const bool initial=operation=="export_initial";
            require(!awaiting_ack,"Previous state transfer awaits an exact acknowledgement");
            require(initial?(!initial_sent&&!have_remote):(have_remote&&remote.frontier.position<current.frontier.position),"Full resend or delta without an acknowledged base refused");
            std::string error;uint64_t sent=0;std::array<unsigned char,65536>page{};
            success(sweep->export_native([&](wire_payload_reader&r,std::string&why){try{
                // PAIRFIX: the initial export verifies the header, tokens and logits
                // first, then each owner span's SHA-256 while its bytes stream (one
                // read instead of a full pre-hash pass plus the send). A mismatch
                // fails this export loudly after the stream; the receiver also
                // verifies every owner before it acknowledges. Deltas keep the
                // up-front full capture.
                if(initial){auto head=dsv41_live::capture(r,[]{return should_stop();},false);head.owners=current.owners;
                    require(head.describe()==current.describe(),"Export checkpoint changed");}
                else require(dsv41_live::capture(r,[]{return should_stop();}).describe()==current.describe(),"Export checkpoint changed");
                delta_artifact_sender delta;if(!initial)success(delta.init(remote.frontier,current.frontier,session_route,model_route,r,why),why);
                const uint64_t bytes=initial?r.bytes():delta.bytes();
                send({{"status","payload"},{"sequence",sequence},{"session",session},{"model_sha256",model_sha},
                    {"kind",initial?"native_initial":"artifact_delta_v2_1"},{"bytes",bytes},{"base",initial?json(nullptr):remote.describe()},{"target",current.describe()}});
                // Owner span boundaries in payload order (header, tokens, logits, spans).
                std::vector<uint64_t> span_begin;std::vector<dsv41_boundary::sha256_native> span_hash;size_t span_at=0;
                if(initial){llama_dsv41::wire_plan plan;std::string plan_error;require(llama_dsv41::make_wire_plan(current.frontier,plan,plan_error),plan_error.c_str());
                    uint64_t at=(current.frontier.mode==3?60:52)+uint64_t(current.frontier.position)*4+uint64_t(129280)*4;
                    for(const auto&s:plan.spans){span_begin.push_back(at);at+=s.bytes;}span_begin.push_back(at);
                    require(at==bytes&&span_begin.size()==current.owners.size()+1,"Export owner span plan differs from the payload");
                    span_hash.resize(current.owners.size());}
                while(sent<bytes){size_t n=size_t(std::min<uint64_t>(page.size(),bytes-sent));
                    // Never let one page straddle an owner boundary, so each span hashes exactly its bytes.
                    if(initial){while(span_at+1<span_begin.size()&&sent>=span_begin[span_at+1])++span_at;
                        if(sent>=span_begin[0]&&span_at+1<span_begin.size())n=size_t(std::min<uint64_t>(n,span_begin[span_at+1]-sent));
                        else if(sent<span_begin[0])n=size_t(std::min<uint64_t>(n,span_begin[0]-sent));}
                    success(initial?r.read(sent,page.data(),n,why):delta.read(sent,page.data(),n,why),why);
                    if(initial&&sent>=span_begin[0])span_hash[span_at].update(page.data(),n);
                    io.write(page.data(),n);sent+=n;}
                if(initial)for(size_t i=0;i<span_hash.size();++i)require(span_hash[i].hex()==current.owners[i],"Exported owner bytes differ from the verified checkpoint");
                return true;
            }catch(const std::exception&e){why=e.what();return false;}},error),error);
            initial_sent=true;awaiting_ack=true;
            std::fprintf(stderr,"DS4PERF_LIVE_EXPORT sequence=%llu initial=%u bytes=%llu milliseconds=%.3f\n",(unsigned long long)sequence,unsigned(initial),(unsigned long long)sent,ms(started));
            // Exactly one metadata line then the declared bytes; no trailer.
        }else if(operation=="acknowledge"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base"});same_base();
            require(awaiting_ack,"Unexpected session acknowledgement");remote=current;have_remote=true;awaiting_ack=false;reply({{"peer_synchronized",true}});
        }else if(operation=="apply_decode_delta"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","target","bytes"});same_base();
            require(have_remote&&!awaiting_ack&&remote.describe()==current.describe(),"Decode delta before peer synchronization");
            const auto target=checkpoint::from_json(command.at("target"));delta_plan plan;std::string error;
            success(make_delta_plan(current.frontier,target.frontier,plan,error),error);require(plan.token_count<=llama_dsv41::decode_delta_max_tokens,"Live decode delta exceeds the advertised maximum_decode_tokens");
            const auto bytes=dsv41_live::u64(command.at("bytes"));require(bytes<=llama_dsv41::decode_delta_byte_cap(plan.token_count),"Live decode artifact exceeds bounded native maximum");
            uint64_t received=0;delta_artifact_receiver incoming;
            auto stream=[&](uint64_t off,void*dst,size_t n,std::string&why){try{
                require(off==received&&n<=65536&&off<=bytes&&n<=bytes-off,"Live decode stream is not sequential/bounded");io.read(dst,n);received+=n;return true;
            }catch(const std::exception&e){why=e.what();return false;}};
            success(incoming.init(bytes,stream,current.frontier,session_route,model_route,error,llama_dsv41::decode_delta_max_tokens),error);
            require(incoming.plan().target.position==target.frontier.position&&incoming.plan().target.context==target.frontier.context&&
                incoming.plan().target.mode==target.frontier.mode&&incoming.plan().target.capture_end==target.frontier.capture_end&&incoming.plan().target.capture_length==target.frontier.capture_length,
                "Live decode artifact target differs from authenticated metadata");
            llama_dsv41_sweep::decode_delta update;update.target=target.frontier;update.evaluated_tokens=incoming.tokens();update.logits=incoming.logits();
            update.base_tokens_sha256=current.tokens_sha256;update.target_tokens_sha256=target.tokens_sha256;update.target_logits_sha256=target.logits_sha256;
            update.base_owner_sha256=current.owners;update.target_owner_sha256=target.owners;
            success(sweep->apply_decode_delta(update,[&](const state_span&s,uint64_t off,void*dst,size_t n){return incoming.read_span(s,off,dst,n,error);},error),error);
            require(received==bytes,"Live decode artifact was not fully consumed");current=target;remote=target;
            reply({{"peer_synchronized",true},{"received_bytes",received},{"evaluated_tokens",plan.token_count},{"RTX_decode",false},{"apply_ms",ms(started)}});
        }else if(operation=="decode"){
            // LLAMADECODE B0: greedy decode on the RTX in this process. Each
            // evaluated token is one stateful 1-row append through the same
            // owner transfer as "prefill" (existing sweep/carry/cache path).
            // A sampled stop token, or the token sampled at max_tokens, is
            // returned but NOT evaluated: the caller's next prefill carries it.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","max_tokens","stop_tokens"});same_base();
            require(!awaiting_ack,"Local decode while a state transfer awaits acknowledgement");
            {const auto t=clock_type::now();require(inspect(&current).describe()==current.describe(),"Local decode actual token/owner state differs from checkpoint");ds4x.begin_op(started,ms(t));}
            const auto maximum=dsv41_live::u64(command.at("max_tokens"));require(maximum&&maximum<=4096,"Local decode requires1..4096 max_tokens");
            const auto&stops=command.at("stop_tokens");require(stops.is_array()&&stops.size()<=16,"Local decode accepts at most16 stop tokens");
            std::vector<int32_t>stop_ids;for(const auto&v:stops){const auto n=dsv41_live::u64(v);require(n<129280,"Stop token out of vocabulary");stop_ids.push_back(int32_t(n));}
            require(uint64_t(current.frontier.position)+maximum<uint64_t(options.context),"Local decode exceeds context capacity");
            std::vector<int32_t>generated;std::vector<double>step_ms;std::string error;std::vector<float>logits;bool stopped=false,cancelled=false;
            clear_decode_cancel();
            generation.active=generation.configured;
            for(uint64_t index=0;index<maximum;++index){
                stop();const auto step_started=clock_type::now();
                success(sweep->completed_logits(logits,error),error);
                int32_t best=0;
                if(generation.active)best=generation.sample(logits,current.frontier.position+uint32_t(index));
                else for(int32_t i=1;i<int32_t(logits.size());++i)if(logits[size_t(i)]>logits[size_t(best)])best=i;
                generated.push_back(best);
                if(generation.active)generation.feed(best);
                if(std::find(stop_ids.begin(),stop_ids.end(),best)!=stop_ids.end()){stopped=true;break;}
                if(index+1==maximum)break;
                if(decode_cancel_requested()){cancelled=true;break;}   // same boundary as max_tokens
                if(ds4_decode_enabled()){
                    ds4_step(best,index);
                    step_ms.push_back(ms(step_started));
                    std::fprintf(stderr,"LLAMADECODE_STEP index=%llu token=%d position=%u step_ms=%.3f engine=ds4\n",(unsigned long long)index,best,
                        current.frontier.position+uint32_t(index)+1,step_ms.back());std::fflush(stderr);
                    continue;
                }
                auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
                next_options.yoco_logical_begin=next_options.yoco_logical_end=0;next_options.operation=llama_dsv41_yoco::operation::decode;next_options.carry.tokens=1;if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=1;
                next_weights.decode_layout=routed_decode_layout(); // LLAMADECODE B1: routed-expert fetch
                {decode_quiet_scope quiet;
                success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
                const std::vector<int32_t>one{best};success(sweep->run(one,[]{return should_stop();},error),error);}
                if(index==0)grow_decode_l2(*sweep,options.fixed.get(),weights.backend);else if(index%16==0)check_decode_vram_reserve();
                step_ms.push_back(ms(step_started));
                std::fprintf(stderr,"LLAMADECODE_STEP index=%llu token=%d position=%u step_ms=%.3f prompt_ms=%.3f\n",(unsigned long long)index,best,
                    current.frontier.position+uint32_t(index)+1,step_ms.back(),sweep->timing().prompt_processing_ms);std::fflush(stderr);
            }
            generation.active=false;
            if(cancelled)clear_decode_cancel();
            const auto t_end=clock_type::now();current=inspect();const double inspect_end_ms=ms(t_end);const uint32_t evaluated=uint32_t(step_ms.size());
            double total_ms=0;for(const double v:step_ms)total_ms+=v;
            std::fprintf(stderr,"LLAMADECODE_DECODE_WALL op_ms=%.3f steps_ms=%.3f inspect_start_ms=%.3f inspect_end_ms=%.3f\n",ms(started),total_ms,ds4x.handoff.inspect_ms,inspect_end_ms);std::fflush(stderr);
            std::fprintf(stderr,"LLAMADECODE_DECODE_COMPLETE generated=%zu evaluated=%u stopped=%d cancelled=%d position=%u decode_ms=%.3f tok_s=%.6f\n",generated.size(),evaluated,int(stopped),
                int(cancelled),current.frontier.position,total_ms,total_ms>0?evaluated*1000.0/total_ms:0.0);std::fflush(stderr);
            reply({{"peer_synchronized",have_remote&&remote.describe()==current.describe()},{"tokens",generated},{"evaluated_tokens",evaluated},{"stopped",stopped},
                {"cancelled",cancelled},{"step_ms",step_ms},{"decode_ms",total_ms},{"RTX_decode",true}});
        }else if(operation=="perfexp_kv_samples"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","path"});same_base();
            const char*gate=std::getenv("LLAMADECODE_PERFEXP_KV_SAMPLES");
            require(gate&&!std::strcmp(gate,"1"),"Experimental KV sampling is disabled");
            const auto path=std::filesystem::u8path(command.at("path").get<std::string>());
            require(path.is_absolute()&&!std::filesystem::exists(path),"KV sample path must be absolute and new");
            HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
            require(file!=INVALID_HANDLE_VALUE,"Cannot create KV sample file");
            struct sample_closer{HANDLE h;~sample_closer(){CloseHandle(h);}}closer{file};
            llama_dsv41::wire_plan plan;std::string error;
            success(llama_dsv41::make_wire_plan(current.frontier,plan,error),error);
            uint64_t cursor=52+uint64_t(current.frontier.position)*4+129280ull*4,output_offset=0;
            json segments=json::array();std::array<unsigned char,65536>buffer{};
            success(sweep->inspect_local([&](wire_payload_reader&r,std::string&why){
                for(const auto&span:plan.spans){
                    const bool kv=span.kind==llama_dsv41::state_kind::compressed,index=span.kind==llama_dsv41::state_kind::index;
                    if(kv||index){
                        const uint32_t width=kv?512:128;
                        const uint64_t bytes=std::min<uint64_t>(span.bytes,128ull*width*4);
                        segments.push_back({{"kind",kv?"compressed":"index"},{"index",span.index},{"width",width},
                                            {"rows",bytes/(width*4)},{"offset",output_offset},{"bytes",bytes}});
                        for(uint64_t off=0;off<bytes;off+=buffer.size()){
                            stop();const size_t n=size_t(std::min<uint64_t>(bytes-off,buffer.size()));
                            if(!r.read(cursor+span.bytes-bytes+off,buffer.data(),n,why))return false;
                            DWORD written=0;if(!WriteFile(file,buffer.data(),DWORD(n),&written,nullptr)||written!=n){why="KV sample write failed";return false;}
                        }
                        output_offset+=bytes;
                    }
                    cursor+=span.bytes;
                }
                return true;
            },error),error);
            require(segments.size()==8&&cursor==plan.bytes,"Incomplete KV sample layout");
            reply({{"experimental",true},{"position",current.frontier.position},{"bytes",output_offset},{"segments",segments},
                   {"cache_storage_changed",false}});
        }else if(operation=="decode_forced"){
            // LLAMADECODE B0/B1 gate: teacher-forced 1-row decode. For each i the
            // frontier's full F32 logits are written to logits_path (raw rows of
            // 129280 floats), then tokens[i] is appended; the last token is not
            // evaluated. Same per-token path as "decode"; only the token choice differs.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","tokens","logits_path"});same_base();
            require(!awaiting_ack,"Forced decode while a state transfer awaits acknowledgement");
            {const auto t=clock_type::now();require(inspect(&current).describe()==current.describe(),"Forced decode actual token/owner state differs from checkpoint");ds4x.begin_op(started,ms(t));}
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=4096,"Forced decode requires1..4096 tokens");
            std::vector<int32_t>forced;for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Forced token out of vocabulary");forced.push_back(int32_t(n));}
            require(uint64_t(current.frontier.position)+forced.size()<uint64_t(options.context),"Forced decode exceeds context capacity");
            const auto path=std::filesystem::u8path(command.at("logits_path").get<std::string>());
            require(path.is_absolute()&&!std::filesystem::exists(path),"Forced decode logits path must be absolute and new");
            HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
            require(file!=INVALID_HANDLE_VALUE,"Cannot create forced decode logits file");
            struct closer{HANDLE h;~closer(){if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}}close_file{file};
            std::vector<int32_t>argmax;std::vector<double>step_ms;std::string error;std::vector<float>logits;
            for(size_t index=0;index<forced.size();++index){
                stop();const auto step_started=clock_type::now();
                success(sweep->completed_logits(logits,error),error);
                int32_t best=0;for(int32_t i=1;i<int32_t(logits.size());++i)if(logits[size_t(i)]>logits[size_t(best)])best=i;
                argmax.push_back(best);
                const auto*bytes=reinterpret_cast<const char*>(logits.data());size_t left=logits.size()*sizeof(float);
                while(left){DWORD wrote=0;const DWORD n=DWORD(std::min<size_t>(left,1u<<24));require(WriteFile(file,bytes,n,&wrote,nullptr)&&wrote==n,"Forced decode logits write failed");bytes+=n;left-=n;}
                if(index+1==forced.size())break;
                if(ds4_decode_enabled()){
                    ds4_step(forced[index],index);
                    step_ms.push_back(ms(step_started));
                    std::fprintf(stderr,"LLAMADECODE_FORCED_STEP index=%zu forced=%d argmax=%d step_ms=%.3f engine=ds4\n",index,forced[index],best,step_ms.back());std::fflush(stderr);
                    continue;
                }
                auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
                next_options.yoco_logical_begin=next_options.yoco_logical_end=0;next_options.operation=llama_dsv41_yoco::operation::decode;next_options.carry.tokens=1;if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=1;
                next_weights.decode_layout=routed_decode_layout(); // LLAMADECODE B1: routed-expert fetch
                {decode_quiet_scope quiet;
                success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
                const std::vector<int32_t>one{forced[index]};success(sweep->run(one,[]{return should_stop();},error),error);}
                if(index==0)grow_decode_l2(*sweep,options.fixed.get(),weights.backend);else if(index%16==0)check_decode_vram_reserve();
                step_ms.push_back(ms(step_started));
                std::fprintf(stderr,"LLAMADECODE_FORCED_STEP index=%zu forced=%d argmax=%d step_ms=%.3f\n",index,forced[index],best,step_ms.back());std::fflush(stderr);
            }
            // Publish a completed dump only after releasing its exclusive
            // Windows writer handle; the receiver may open it immediately.
            require(CloseHandle(close_file.h)!=0,"Forced decode logits close failed");close_file.h=INVALID_HANDLE_VALUE;
            current=inspect();double total_ms=0;for(const double v:step_ms)total_ms+=v;
            reply({{"peer_synchronized",have_remote&&remote.describe()==current.describe()},{"argmax",argmax},{"vectors",argmax.size()},
                {"evaluated_tokens",step_ms.size()},{"step_ms",step_ms},{"decode_ms",total_ms},{"RTX_decode",true}});
        }else if(operation=="prefill"){
            if(command.contains("vision_spans_file"))
                dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","tokens","vision_spans_file"});
            else dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","tokens"});same_base();
            // LLAMADECODE B0: a local-only session (never exported to a peer)
            // may append without a peer; a peer-bound session keeps exact sync.
            const bool local_only=!have_remote&&!initial_sent&&!awaiting_ack;
            require(local_only||(have_remote&&!awaiting_ack&&remote.describe()==current.describe()),"Incremental prefill requires exact two-way synchronization");
            {const auto t=clock_type::now();require(inspect(&current).describe()==current.describe(),"Incremental prefill actual token/owner state differs from acknowledged checkpoint");ds4x.begin_op(started,ms(t));}
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=capacity,"Live prefill token count exceeds reserved chunk");
            require(ids.size()<uint64_t(options.context)-current.frontier.position,"Live prefill exceeds context capacity");
            std::vector<int32_t>tokens;tokens.reserve(ids.size());for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Live token out of vocabulary");tokens.push_back(int32_t(n));}
            auto next_images=options.image_spans;
            if(command.contains("vision_spans_file"))
                next_images=studio_vision_spans(command.at("vision_spans_file").get<std::string>(),uint32_t(options.context));
            // The borrowed decode bridge binds target-only owners. Mode3
            // needs the streaming path to maintain its complete DSpark ring.
            if(ds4_decode_enabled()&&!options.capture_dspark&&next_images.empty()&&tokens.size()<=append_decode_max()){
                // Small append: teacher-forced ds4 steps on the resident state (the
                // decode/decode_forced step), every token evaluated; same reply and
                // checkpoint as the sweep. The frontier logits come from ds4's decode kernels.
                if(append_multirow_enabled()&&tokens.size()>=2){
                    ds4x.prepare(*sweep,weights,uint32_t(options.context));require(ds4x.prefill_rows>=tokens.size(),"Multi-row append exceeds the ds4 prefill workspace");
                    stop();ensure_decode_context(*sweep,options.fixed.get(),weights.backend,current.frontier.position+uint32_t(tokens.size()));
                    ds4x.prefill_rows_step(*sweep,weights.backend,tokens,current.frontier.position);
                    grow_decode_l2(*sweep,options.fixed.get(),weights.backend);
                }else for(size_t index=0;index<tokens.size();++index){stop();ds4_step(tokens[index],index);}
                const double append_ms=ms(started);
                const uint32_t before=current.frontier.position;current=inspect();require(current.frontier.position==before+tokens.size(),"Live append position did not advance exactly");
                std::fprintf(stderr,"LLAMADECODE_APPEND_VIA_DECODE tokens=%zu ms=%.3f position=%u\n",tokens.size(),append_ms,current.frontier.position);std::fflush(stderr);
                reply({{"peer_synchronized",false},{"prompt_tokens",tokens.size()},{"prompt_processing_ms",append_ms},{"export_preparation_ms",0.0},{"append_via_decode",true},{"append_multirow",append_multirow_enabled()&&tokens.size()>=2}});
                continue;
            }
            // Return the decode-only host tier before even the speculative
            // Engram loader allocates an append's prefill rows.
            {std::string host_error;success(sweep->provider_for_external_decode()->release_host_cache(host_error),host_error);}
            // LLAMADECODE_ENGRAM_EARLY: this append's Engram rows start now, before
            // its open, start_request and layer-0 weight fetch (off: no-op).
            if(!command.contains("vision_spans_file")) {std::string early_error;success(sweep->start_append_engram(tokens,early_error),early_error);}
            bool resident_sweep=false;
            if(llama_dsv41_resident_append_enabled()&&append_decode_max()==0&&ds4_decode_enabled()&&!options.capture_dspark&&next_images.empty()&&tokens.size()<=llama_dsv41_resident_append_limit){
                auto *provider=sweep->provider_for_external_decode();
                resident_sweep=provider&&provider->decode_layout();
                if(resident_sweep){std::string why;success(provider->release_decode_cache(why),why);}
            }
            auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
            next_options.image_spans=std::move(next_images);next_options.images=!next_options.image_spans.empty();
            if(resident_sweep)next_weights.decode_layout=true;
            next_options.yoco_logical_begin=next_options.yoco_logical_end=0;next_options.carry.tokens=uint32_t(tokens.size());if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=uint32_t(tokens.size());
            std::string error;success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
            const double admission_ms=ms(started);success(sweep->run(tokens,[]{return should_stop();},error),error);const auto timing=sweep->timing();
            const uint32_t before=current.frontier.position;current=inspect();require(current.frontier.position==before+tokens.size(),"Live prefill position did not advance exactly");
            std::fprintf(stderr,"PERFEXP_RESIDENT_SWEEP enabled=%d tokens=%zu same_sweep_policy=1 l2_released_before_prefill_restore=1\n",int(resident_sweep),tokens.size());std::fflush(stderr);
            reply({{"peer_synchronized",false},{"prompt_tokens",tokens.size()},{"prompt_processing_ms",admission_ms+timing.prompt_processing_ms},{"export_preparation_ms",timing.export_preparation_ms},{"append_resident_sweep",resident_sweep}});
        }else require(false,"Unknown persistent session operation");
    }
}
}
