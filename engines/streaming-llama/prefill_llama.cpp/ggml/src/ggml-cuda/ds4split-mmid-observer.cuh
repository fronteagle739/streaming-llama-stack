#pragma once
// UNBUILT diagnostic-only source candidate. No dispatch/operand arithmetic changes.
// This header is included in ggml-cuda.cu after the ordinary matmul dispatcher.
// The SHA helper must be installed beside it as ds4split-mmid-sha256.inc.
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ds4split_mmid {
#if defined(_WIN32)
// Windows APIs are already used by this Windows CUDA backend; no BCrypt dependency.
constexpr uint64_t cap_limit = 32ull*1024*1024;
constexpr uint64_t arena_bytes = 4ull*1024*1024;
constexpr uint64_t host_admission = 8ull*1024*1024;
constexpr uint64_t ram_floor = 1024ull*1024*1024;
constexpr const char * source_base = "196889b86f0d01a4074cd2f9054af47e8abb193e";
inline const wchar_t * diagnostic_root() { const auto *p = _wgetenv(L"SLSTACK_DIAGNOSTIC_ROOT"); if (!p || !*p) GGML_ABORT("Set SLSTACK_DIAGNOSTIC_ROOT for diagnostics"); return p; }
inline void require(bool ok, const char * why) {
    if (!ok) GGML_ABORT("DS4SPLIT_MMID_CAPTURE_REFUSED: %s", why);
}
#if defined(DS4SPLIT_MMID_CPU_TEST)
#include "ds4split-mmid-sha256.inc"
#else
#include "ds4split-mmid-sha256.inc"
#endif
inline std::string sha(const void * p, size_t n) { digest d; d.update(p,n); return d.finish(); }
inline bool hex(const std::string & s, size_t n) {
    return s.size()==n && std::all_of(s.begin(),s.end(),[](unsigned char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
inline uint64_t integer(const std::string &s) {
    require(!s.empty()&&s.size()<=10,"invalid integer width");uint64_t v=0;
    for(unsigned char c:s){require(c>='0'&&c<='9',"invalid integer character");require(v<=(UINT64_MAX-(c-'0'))/10,"integer overflow");v=v*10+(c-'0');}return v;
}
inline void physical_ram(uint64_t pending=0) {
    MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
    require(GlobalMemoryStatusEx(&m)!=FALSE,"cannot read available physical RAM");
    require(pending<=UINT64_MAX-ram_floor && m.ullAvailPhys>=pending+ram_floor,"observer would cross physical RAM minus1GiB");
}
struct handle {
    HANDLE value=INVALID_HANDLE_VALUE;
    handle()=default;explicit handle(HANDLE h):value(h){}
    ~handle(){if(value!=INVALID_HANDLE_VALUE)CloseHandle(value);}
    handle(const handle&)=delete;handle&operator=(const handle&)=delete;
};
inline void ordinary(HANDLE h, bool directory) {
    BY_HANDLE_FILE_INFORMATION i{};
    require(h!=INVALID_HANDLE_VALUE&&GetFileInformationByHandle(h,&i)!=FALSE,"cannot inspect diagnostic file");
    require(!(i.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"reparse point in diagnostic path");
    require(bool(i.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)==directory,"diagnostic path kind differs");
    if(!directory)require(i.nNumberOfLinks==1,"diagnostic file must have one physical link");
}
inline std::wstring path(const std::string & name) {
    require(!name.empty()&&std::all_of(name.begin(),name.end(),[](unsigned char c){return (c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='.'||c=='-'||c=='_';}),"invalid internal diagnostic filename");
    return std::wstring(diagnostic_root())+L"\\"+std::wstring(name.begin(),name.end());
}
inline std::string read_owner_small(const std::string & name) {
    const auto p=path(name);handle h(CreateFileW(p.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr));ordinary(h.value,false);
    char bytes[4097];DWORD got=0;require(ReadFile(h.value,bytes,sizeof(bytes),&got,nullptr)!=FALSE&&got<=4096,"trusted observer file absent/too large");
    return std::string(bytes,bytes+got);
}
inline std::string module_sha(HMODULE module) {
    wchar_t p[32768];const DWORD len=GetModuleFileNameW(module,p,32768);require(len&&len<32768,"cannot identify actual executable/backend module");
    handle h(CreateFileW(p,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr));ordinary(h.value,false);
    LARGE_INTEGER bytes{};require(GetFileSizeEx(h.value,&bytes)!=FALSE&&bytes.QuadPart>0&&uint64_t(bytes.QuadPart)<=1024ull*1024*1024,"actual module exceeds1GiB diagnostic read bound");
    digest d;char buf[65536];uint64_t total=0;
    for(;;){DWORD n=0;require(ReadFile(h.value,buf,sizeof(buf),&n,nullptr)!=FALSE,"module hash read failed");if(!n)break;total+=n;require(total<=uint64_t(bytes.QuadPart),"module grew during hash");d.update(buf,n);}
    require(total==uint64_t(bytes.QuadPart),"module hash length changed");return d.finish();
}
inline std::map<std::string,std::string> parse(const std::string & raw) {
    require(!raw.empty()&&raw.size()<=4096&&raw.find('\0')==std::string::npos,"invalid observer config bytes");
    std::map<std::string,std::string> m;std::istringstream in(raw);std::string line;
    while(std::getline(in,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();auto eq=line.find('=');require(eq!=std::string::npos&&eq>0&&eq+1<line.size(),"expected exact observer key=value");require(m.emplace(line.substr(0,eq),line.substr(eq+1)).second,"duplicate observer config key");}
    const char * keys[]={"schema","enabled","nonce","source_base_revision","binary_sha256","backend_binary_sha256","variant_manifest_sha256","boundary_config_sha256","capture_bytes","existing_boundary_bytes","host_memory_budget_bytes"};
    require(m.size()==sizeof(keys)/sizeof(*keys),"missing/extra observer config key");for(auto k:keys)require(m.count(k)==1,"missing observer config key");return m;
}
struct profile {
    bool enabled=false;
    std::string nonce,binary_sha,backend_sha,manifest_sha,boundary_sha;
    uint64_t cap=0,existing=0;
};
inline profile settings(const std::map<std::string,std::string>&m) {
    require(m.at("schema")=="ds4split-mmid-capture-v1","wrong observer schema");
    require(m.at("enabled")=="0"||m.at("enabled")=="1","observer enabled must be exactly0 or1");
    profile p;p.enabled=m.at("enabled")=="1";p.nonce=m.at("nonce");
    require(hex(p.nonce,32)&&m.at("source_base_revision")==source_base,"observer nonce/source revision differs");
    for(auto k:{"binary_sha256","backend_binary_sha256","variant_manifest_sha256","boundary_config_sha256"})require(hex(m.at(k),64),"invalid observer SHA pin");
    p.binary_sha=m.at("binary_sha256");p.backend_sha=m.at("backend_binary_sha256");p.manifest_sha=m.at("variant_manifest_sha256");p.boundary_sha=m.at("boundary_config_sha256");
    p.cap=integer(m.at("capture_bytes"));p.existing=integer(m.at("existing_boundary_bytes"));
    require(p.cap&&p.cap<=cap_limit&&p.existing<=p.cap,"invalid shared32MiB capture admission");
    if(p.enabled){
        // Parent declares the expert20 profile and any bounded HC extras.
        // Attention/Q/head profiles instead require explicit enabled=0.
        require(p.existing>=13502464&&p.existing<=14ull*1024*1024,"enabled observer requires expert20/HC profile reservation between13502464B and14MiB");
        constexpr uint64_t observer_max=13327628+3*16384+4096;
        require(observer_max<=p.cap-p.existing,"enabled graph+MMID profile exceeds shared32MiB allowance");
        require(integer(m.at("host_memory_budget_bytes"))==host_admission,"enabled observer requires explicit8MiB additional host admission");
    }else{
        require(integer(m.at("host_memory_budget_bytes"))==0,"disabled observer must declare zero readback-arena host admission");
    }
    return p;
}
inline void validate_bindings(const profile&p,const std::string&boundary,const std::string&manifest,
        const std::string&actual_binary,const std::string&actual_backend) {
    require(sha(boundary.data(),boundary.size())==p.boundary_sha,"existing trusted boundary config hash differs");
    for(auto pair:{std::pair<const char*,std::string>{"nonce",p.nonce},{"binary_sha256",p.binary_sha},{"variant_manifest_sha256",p.manifest_sha}}){
        const std::string wanted=std::string(pair.first)+"="+pair.second;
        std::istringstream b(boundary);std::string line;int hits=0;while(std::getline(b,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();hits+=line==wanted;}require(hits==1,"observer and graph capture pins differ");}
    require(sha(manifest.data(),manifest.size())==p.manifest_sha,"source build manifest hash differs");
    require(actual_binary==p.binary_sha,"actual executable SHA differs from trusted observer config");
    require(actual_backend==p.backend_sha,"actual backend module SHA differs from trusted observer config");
}
struct owner_state {
    std::mutex mutex;
    std::string nonce, config_sha, binary_sha, backend_sha, manifest_sha, boundary_sha;
    uint64_t cap=0, used=0, existing=0;
    bool done[3]={false,false,false};bool in_call=false, initialized=false, enabled=false;
    std::vector<unsigned char> arena;
    std::vector<std::string> completed_records;
    // Held ancestors deny rename/delete for the lifetime of all exclusive outputs.
    std::array<HANDLE,3> ancestors{{INVALID_HANDLE_VALUE,INVALID_HANDLE_VALUE,INVALID_HANDLE_VALUE}};
    ~owner_state(){for(auto h:ancestors)if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}
    void charge(uint64_t n){require(used<=cap&&n<=cap-used,"aggregate graph+MMID capture exceeds32MiB cap");used+=n;}
    void allocate_readback_if_enabled(){if(enabled){physical_ram(host_admission);arena.resize(size_t(arena_bytes),0);}}
    void init() {
        if(initialized)return;physical_ram(256*1024);
        const wchar_t * dirs[]={diagnostic_root(),diagnostic_root(),diagnostic_root()};
        for(size_t i=0;i<3;++i){ancestors[i]=CreateFileW(dirs[i],FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);ordinary(ancestors[i],true);}
        const auto raw=read_owner_small("mmid-capture.config");const auto p=settings(parse(raw));config_sha=sha(raw.data(),raw.size());
        nonce=p.nonce;binary_sha=p.binary_sha;backend_sha=p.backend_sha;manifest_sha=p.manifest_sha;boundary_sha=p.boundary_sha;cap=p.cap;existing=p.existing;used=existing;enabled=p.enabled;
        const auto boundary=read_owner_small("boundary-capture.config"),manifest=read_owner_small("boundary-capture.source-manifest.json");
        const auto actual_binary=module_sha(nullptr);
        HMODULE module=nullptr;require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(&module_sha),&module)!=FALSE,"cannot identify actual observer backend module");
        validate_bindings(p,boundary,manifest,actual_binary,module_sha(module));
        allocate_readback_if_enabled();
        initialized=true;
    }
    void write_host(const std::string & name,const void * p,size_t n) {
        physical_ram();charge(n);const auto filename=path(name);handle out(CreateFileW(filename.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OPEN_REPARSE_POINT,nullptr));ordinary(out.value,false);
        const auto * b=static_cast<const unsigned char*>(p);size_t off=0;
        while(off<n){const DWORD take=DWORD(std::min<size_t>(65536,n-off));DWORD wrote=0;require(WriteFile(out.value,b+off,take,&wrote,nullptr)!=FALSE&&wrote==take,"exclusive observer output write failed");off+=take;}
        require(FlushFileBuffers(out.value)!=FALSE,"observer output flush failed");
    }
};
inline owner_state & state(){static owner_state s;return s;}
#if !defined(DS4SPLIT_MMID_CPU_TEST)
inline int target(const ggml_tensor *dst){
    if(!dst||!dst->src[0])return -1;
    const char *names[]={"blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight"};
    for(int i=0;i<3;++i)if(std::strcmp(dst->src[0]->name,names[i])==0)return i;return -1;
}
class call_scope {
    owner_state * s=nullptr;std::unique_lock<std::mutex> lock;
    ggml_backend_cuda_context * ctx=nullptr;ggml_tensor * dst=nullptr;int slot=-1;
    std::string stem, actual_route, records;
    bool packed_captured=false, finished=false;int64_t padded=0;uint64_t allocation=0,payload=0;
    std::string tensor_geometry(const ggml_tensor *t){
        std::ostringstream o;o<<"{\"type\":"<<int(t->type)<<",\"ne\":[";for(int i=0;i<4;++i)o<<(i?",":"")<<t->ne[i];o<<"],\"nb\":[";for(int i=0;i<4;++i)o<<(i?",":"")<<t->nb[i];o<<"]}";return o.str();
    }
    void dump(const char * tag,const void * dev,size_t n,const char *format){
        require(s&&dev&&n&&n<=s->arena.size(),"observer live span exceeds4MiB readback arena");physical_ram();
        CUDA_CHECK(cudaMemcpyAsync(s->arena.data(),dev,n,cudaMemcpyDeviceToHost,ctx->stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
        const std::string name=stem+"."+tag+".bin";const auto hash=sha(s->arena.data(),n);s->write_host(name,s->arena.data(),n);
        if(!records.empty())records+=",";records+="{\"name\":\""+name+"\",\"format\":\""+format+"\",\"bytes\":"+std::to_string(n)+",\"sha256\":\""+hash+"\"}";
    }
public:
    call_scope(ggml_backend_cuda_context & c,ggml_tensor *d):lock() {
        const char *flag=std::getenv("DSV41_BOUNDARY_CAPTURE");if(!flag||!*flag)return;
        require(std::strcmp(flag,"1")==0,"capture opt-in must be exactly1 or absent");
        slot=target(d);if(slot<0)return;
        s=&state();lock=std::unique_lock<std::mutex>(s->mutex);s->init();
        // Explicit disabled diagnostic profile still verified every trusted pin,
        // but allocates no arena and adds no CUDA sync, readback or output.
        if(!s->enabled){s=nullptr;lock.unlock();return;}
        require(!s->in_call&&!s->done[slot],"duplicate or concurrent layer0 expert capture");
        require(d->src[1]&&d->src[2]&&d->src[0]->type==GGML_TYPE_Q4_K&&d->src[1]->type==GGML_TYPE_F32&&d->type==GGML_TYPE_F32&&d->src[2]->type==GGML_TYPE_I32,"actual expert input types differ");
        auto*w=d->src[0];auto*x=d->src[1];auto*ids=d->src[2];
        const int64_t k=slot==2?2304:5120,out=slot==2?5120:2304,channels=slot==2?6:1;
        require(w->ne[0]==k&&w->ne[1]==out&&w->ne[2]==384&&w->ne[3]==1&&x->ne[0]==k&&x->ne[1]==channels&&x->ne[2]==32&&x->ne[3]==1,"actual expert matrix/activation geometry differs");
        require(d->ne[0]==out&&d->ne[1]==6&&d->ne[2]==32&&d->ne[3]==1&&ids->ne[0]==6&&ids->ne[1]==32&&ids->ne[2]==1&&ids->ne[3]==1,"actual expert output/IDs geometry differs");
        require(ggml_is_contiguous(x)&&ggml_is_contiguous(ids)&&ggml_is_contiguous(d),"actual expert logical strides are not expected contiguous spans");
        cudaStreamCaptureStatus cap=cudaStreamCaptureStatusNone;CUDA_CHECK(cudaStreamIsCapturing(c.stream(),&cap));require(cap==cudaStreamCaptureStatusNone,"diagnostic sync cannot run during CUDA graph capture");
        ctx=&c;dst=d;stem="mmid-"+s->nonce+"."+(slot==0?"gate":slot==1?"up":"down");s->in_call=true;
    }
    ~call_scope(){if(s&&!finished)require(false,"expert observer exited without completing actual dispatch");}
    void route(const char * name){if(!s)return;require(actual_route.empty(),"duplicate dispatch witness");actual_route=name;}
    void packed(const char *maps,size_t maps_bytes,const char *q,size_t defined,size_t allocated,int64_t padded_k){
        if(!s)return;require(actual_route=="mmq_id"&&!packed_captured,"packed buffers outside actual MMQ_ID route");
        require(maps_bytes==(2*192+384+1)*sizeof(int32_t)&&padded_k==(slot==2?2560:5120),"actual map/padded geometry differs");
        require(defined==size_t(192)*size_t(padded_k)*sizeof(block_q8_1)/QK8_1&&allocated>=defined,"actual quantized payload bounds differ");
        require(mmq_get_q8_1_ds_layout(dst->src[0]->type)==MMQ_Q8_1_DS_LAYOUT_DS4,"actual packed Q8_1 layout differs");
        padded=padded_k;allocation=allocated;payload=defined;
        // Wait/copy after the REAL quantizer on its real stream. No pool writes.
        dump("source_f32",dst->src[1]->data,ggml_nbytes(dst->src[1]),"ieee754-f32-le");
        dump("selected_i32",dst->src[2]->data,ggml_nbytes(dst->src[2]),"i32-le-top6-rank-token");
        dump("maps_i32",maps,maps_bytes,"i32-le-ids_src1_192-ids_dst_192-expert_bounds_385");
        dump("q8_1_mmq_ds4",q,defined,"mmq-ds4-transposed-128-values-144-bytes-half2-scale-sum");
        packed_captured=true;
    }
    void finish(){
        if(!s)return;require(!finished&&!actual_route.empty(),"missing actual dispatch route");
        // The original selected kernel already ran; observing output adds no math.
        dump("output_f32",dst->data,ggml_nbytes(dst),"ieee754-f32-le");
        std::ostringstream meta;meta<<"{\"schema\":\"ds4split-mmid-call-v1\",\"nonce\":\""<<s->nonce<<"\",\"source_base_revision\":\""<<source_base<<"\",\"config_sha256\":\""<<s->config_sha<<"\",\"binary_sha256\":\""<<s->binary_sha<<"\",\"backend_binary_sha256\":\""<<s->backend_sha<<"\",\"variant_manifest_sha256\":\""<<s->manifest_sha<<"\",\"boundary_config_sha256\":\""<<s->boundary_sha<<"\",\"weight_name\":\""<<dst->src[0]->name<<"\",\"weight\":"<<tensor_geometry(dst->src[0])<<",\"source\":"<<tensor_geometry(dst->src[1])<<",\"ids\":"<<tensor_geometry(dst->src[2])<<",\"output\":"<<tensor_geometry(dst)<<",\"op_precision\":"<<ggml_get_op_params_i32(dst,0)<<",\"device\":"<<ctx->device<<",\"cc\":"<<ggml_cuda_info().devices[ctx->device].cc<<",\"route\":\""<<actual_route<<"\",\"packed_captured\":"<<(packed_captured?"true":"false")<<",\"padded_k\":"<<padded<<",\"packed_defined_bytes\":"<<payload<<",\"packed_allocated_bytes\":"<<allocation<<",\"uninitialized_tail_captured\":false,\"diagnostic_stream_sync\":true,\"qualification_passed\":false,\"files\":["<<records<<"]}\n";
        // Stream identity is diagnostic only; all readbacks use this actual stream.
        std::string out=meta.str();const auto end=out.rfind("}\n");require(end!=std::string::npos,"invalid internal call metadata");
        out.insert(end,",\"stream_pointer_u64\":"+std::to_string(uint64_t(reinterpret_cast<uintptr_t>(ctx->stream()))));
        require(out.size()<=16384,"call manifest exceeds16KiB");s->write_host(stem+".json",out.data(),out.size());
        finished=true;s->in_call=false;s->done[slot]=true;s->completed_records.push_back(stem+".json");
        require(packed_captured&&actual_route=="mmq_id","actual expected MMQ_ID packed path was not observed; no alternate kernel forced");
        if(s->done[0]&&s->done[1]&&s->done[2]){
            std::ostringstream m;m<<"{\"schema\":\"ds4split-mmid-capture-v1\",\"nonce\":\""<<s->nonce<<"\",\"complete\":true,\"actual_dispatch_observed\":true,\"q8_1_activation_captured\":true,\"qualification_passed\":false,\"three_original_calls\":true,\"capture_config_sha256\":\""<<s->config_sha<<"\",\"reserved_existing_boundary_bytes\":"<<s->existing<<",\"aggregate_bytes_before_completion_manifest\":"<<s->used<<",\"cap_bytes\":"<<s->cap<<",\"files\":[";
            for(size_t i=0;i<s->completed_records.size();++i)m<<(i?",":"")<<"\""<<s->completed_records[i]<<"\"";m<<"]}\n";const auto b=m.str();require(b.size()<=4096,"completion manifest exceeds4KiB");s->write_host("mmid-"+s->nonce+".json",b.data(),b.size());
        }
    }
};
#endif // !DS4SPLIT_MMID_CPU_TEST
#else
class call_scope {
public:
    call_scope(ggml_backend_cuda_context &,ggml_tensor *){}
    void route(const char *){}
    void packed(const char *,size_t,const char *,size_t,size_t,int64_t){}
    void finish(){}
};
#endif
} // namespace ds4split_mmid
