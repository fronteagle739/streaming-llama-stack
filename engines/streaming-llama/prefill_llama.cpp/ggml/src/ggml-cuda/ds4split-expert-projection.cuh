#pragma once
// TEST-only complete expert projection diagnostic.
// Every arithmetic profile requires
// explicit trusted boundary/observer/projection configs pinned to the actual
// binary and source manifest. Include AFTER ds4split-mmid-observer.cuh.
// Default capture absent adds no config IO, readback, allocation or sync.
// Uses original provider Q4_K device ranges and actual CUDA operators only.
namespace ds4split_projection_candidate {
// The original observer owns the shared arena mutex until its lexical scope
// ends. Every original dispatch return must leave that scope before mixed
// completion acquires the same owner mutex. Also used by the CPU lifetime test.
template<class Original,class Completion>
inline void finish_after_original_scope(Original&&original,Completion&&completion){
    original();
    completion();
}
#if defined(_WIN32)
using ds4split_mmid::require;
constexpr uint64_t mib=1024ull*1024;
constexpr uint64_t half_matrix=23592960, float_matrix=47185920;
constexpr uint64_t rhs_max=655360, output_max=655360;
// Pinned model-free LAZY probe measured converter2MiB+firstMMF58MiB.
// Separate runtime/code first-use reserve; no additional weight/RHS matrix.
constexpr uint64_t mmf_explicit_arena=64*mib, mmf_runtime_code_reserve=64*mib;
constexpr uint64_t f32_explicit_arena=96*mib, cublas_opaque_reserve=96*mib;
// Pinned32-shape probe measured66MiB at handle first use;96MiB reserves30MiB margin.
constexpr uint64_t mmf_device_cap=mmf_explicit_arena+mmf_runtime_code_reserve, f32_device_cap=f32_explicit_arena+cublas_opaque_reserve;
constexpr uint64_t host_reserve=8*mib, host_arena_bytes=4*mib;
enum class arm { off, gateup_mmf, down_mmf, gateup_f32, down_f32, all_mmf, all_f32 };
struct golden { std::string q4_sha,half_sha; };
// Only the trusted loader may create the authorization token consumed below.
// In particular there is no environment switch for an arithmetic variant.
class authorization {
    arm value=arm::off;
    std::string nonce,config_sha,proof_sha;
    std::array<int32_t,192> expected_ids{};
    std::map<std::pair<int,int>,golden> half_proofs;
    friend authorization load_trusted();
    friend authorization authorize_fields(const std::string&,const ds4split_mmid::owner_state&,const std::string&);
    friend class execution;
    friend class call_scope;
public:
    authorization()=default;
#if defined(DS4SPLIT_PROJECTION_CPU_TEST)
    arm fixture_arm()const{return value;}
#endif
};
inline bool is_half(arm a){return a==arm::gateup_mmf||a==arm::down_mmf||a==arm::all_mmf;}
inline bool is_reference(arm a){return a==arm::gateup_f32||a==arm::down_f32||a==arm::all_f32;}
inline bool applies(arm a,int slot){return a==arm::all_mmf||a==arm::all_f32||(slot<2&&(a==arm::gateup_mmf||a==arm::gateup_f32))||(slot==2&&(a==arm::down_mmf||a==arm::down_f32));}
inline bool reference_override_name(std::string name){
    for(char&c:name)if(c>='a'&&c<='z')c=char(c-'a'+'A');
    const char*exact[]={"NVIDIA_TF32_OVERRIDE","CUBLAS_WORKSPACE_CONFIG","CUBLAS_MATH_MODE","CUBLAS_COMPUTE_TYPE","CUBLASLT_MATH_MODE","CUBLASLT_COMPUTE_TYPE"};
    for(auto key:exact)if(name==key)return true;
    const char*prefix[]={"CUBLAS_EMULATE_","CUBLAS_EMULATION_","CUBLAS_FIXEDPOINT_EMULATION_","CUBLASLT_EMULATE_","CUBLASLT_EMULATION_","CUBLASLT_FIXEDPOINT_EMULATION_"};
    for(auto key:prefix)if(name.compare(0,std::strlen(key),key)==0)return true;return false;
}
inline void reject_reference_overrides(arm a,const std::vector<std::string>&names){
    if(!is_reference(a))return;for(const auto&name:names)require(!reference_override_name(name),"pedantic F32 reference precision/emulation/workspace environment override present");
}
inline std::vector<std::string> actual_environment_names(){
    // Names only: no environment value is copied to a log or diagnostic record.
    struct snapshot {LPWCH p=GetEnvironmentStringsW();~snapshot(){if(p)FreeEnvironmentStringsW(p);}} env;
    require(env.p!=nullptr,"cannot inspect actual reference environment");std::vector<std::string> names;size_t scanned=0,total_names=0;
    for(const wchar_t*p=env.p;*p;){const wchar_t*end=p;while(*end){require(++scanned<=512*1024,"reference environment block exceeds1MiB bound");++end;}
        const wchar_t*eq=p;while(eq<end&&*eq!=L'=')++eq;
        if(eq>p&&eq<end){std::string n;for(auto c=p;c<eq;++c)n.push_back(*c<=127?char(*c):'?');total_names+=n.size();require(total_names<=256*1024,"reference environment names exceed256KiB");names.push_back(std::move(n));}
        p=end+1;
    }return names;
}
inline void verify_reference_environment(arm a){if(is_reference(a)){ds4split_mmid::physical_ram();reject_reference_overrides(a,actual_environment_names());}}
inline uint64_t capture_maximum(arm a){
    const uint64_t oldbytes[3]={3534596,3534596,6258436},newbytes[3]={2425600,2425600,5702400};uint64_t maximum=4096;
    for(int i=0;i<3;++i)maximum+=applies(a,i)?newbytes[i]+128*1024:oldbytes[i]+16384;
    return maximum;
}
inline std::array<std::vector<int>,384> ordered_columns(const int32_t*ids){
    std::array<std::vector<int>,384> columns;
    for(int t=0;t<32;++t)for(int r=0;r<6;++r){int e=ids[t*6+r];require(e>=0&&e<384,"projection ID bounds");for(int p=0;p<r;++p)require(ids[t*6+p]!=e,"projection duplicate selected expert");columns[size_t(e)].push_back(t*6+r);}
    return columns;
}

struct allocation_observation {
    uint64_t before_free,after_free,total_device,admitted,explicit_live,positive_global_delta;
    bool global_free_increased,accepted;
};
inline allocation_observation observe_allocation(uint64_t before,uint64_t after,uint64_t total,uint64_t admitted,uint64_t explicit_live){
    // cudaMemGetInfo is global. A desktop/context release may INCREASE free
    // bytes; that is zero net growth, never unsigned-underflow overbudget.
    const uint64_t delta=before>after?before-after:0;
    return {before,after,total,admitted,explicit_live,delta,after>before,
        before<=total&&after<=total&&explicit_live<=admitted&&delta<=admitted};
}
inline void require_allocation(const allocation_observation&o,const char*phase,int slot,int expert,int calls){
    if(!o.accepted){
        // One bounded numeric stderr witness before fatal refusal; no additional
        // capture arena/file, no cap increase, no claim of global-counter ownership.
        std::fprintf(stderr,"DS4SPLIT_PROJECTION_DEVICE_ACCOUNTING_REFUSED {\"phase\":\"%s\",\"slot\":%d,\"expert\":%d,\"completed_gpu_calls\":%d,\"before_free_bytes\":%llu,\"after_free_bytes\":%llu,\"total_device_bytes\":%llu,\"admitted_bytes\":%llu,\"known_explicit_live_bytes\":%llu,\"positive_global_delta_bytes\":%llu,\"global_free_increased\":%s,\"global_counter_proves_owned_footprint\":false}\n",phase,slot,expert,calls,
            static_cast<unsigned long long>(o.before_free),static_cast<unsigned long long>(o.after_free),static_cast<unsigned long long>(o.total_device),static_cast<unsigned long long>(o.admitted),static_cast<unsigned long long>(o.explicit_live),static_cast<unsigned long long>(o.positive_global_delta),o.global_free_increased?"true":"false");
        std::fflush(stderr);
    }
    require(o.accepted,"projection explicit allocation or positive global device delta exceeds declared cap (numeric witness above)");
}

inline std::string read_bounded(const char*name,size_t cap){
    const auto p=ds4split_mmid::path(name);
    ds4split_mmid::handle h(CreateFileW(p.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr));
    ds4split_mmid::ordinary(h.value,false);LARGE_INTEGER n{};
    require(GetFileSizeEx(h.value,&n)&&n.QuadPart>0&&uint64_t(n.QuadPart)<=cap,"projection trusted file size");
    std::string b(size_t(n.QuadPart),'\0');DWORD got=0;
    require(ReadFile(h.value,&b[0],DWORD(b.size()),&got,nullptr)&&got==b.size(),"projection trusted file read");return b;
}
inline std::map<std::string,std::string> parse_exact(const std::string&raw){
    require(raw.size()<=4096&&raw.find('\0')==std::string::npos,"projection config size/NUL");
    std::map<std::string,std::string> m;std::istringstream in(raw);std::string l;
    while(std::getline(in,l)){if(!l.empty()&&l.back()=='\r')l.pop_back();auto p=l.find('=');require(p>0&&p!=std::string::npos&&p+1<l.size(),"projection key=value");require(m.emplace(l.substr(0,p),l.substr(p+1)).second,"projection duplicate key");}
    const char*keys[]={"schema","arm","nonce","source_base_revision","binary_sha256","backend_binary_sha256","variant_manifest_sha256","boundary_config_sha256","capture_bytes","existing_boundary_bytes","host_memory_budget_bytes","device_workspace_bytes","native_dequant_proof_sha256","native_moe_shader_sha256","selected_ids_hex","selected_ids_sha256"};
    require(m.size()==16,"projection missing/extra config fields");for(auto k:keys)require(m.count(k)==1,"projection missing field");return m;
}
inline authorization authorize_fields(const std::string&raw,const ds4split_mmid::owner_state&s,const std::string&proof){
    authorization a;
    const auto m=parse_exact(raw);
    require(m.at("schema")=="ds4split-expert-projection-test-v1","projection schema");
    const auto name=m.at("arm");
    if(name=="off")a.value=arm::off;else if(name=="gateup-mmf")a.value=arm::gateup_mmf;else if(name=="down-mmf")a.value=arm::down_mmf;
    else if(name=="gateup-f32")a.value=arm::gateup_f32;else if(name=="down-f32")a.value=arm::down_f32;else if(name=="all-mmf")a.value=arm::all_mmf;else if(name=="all-f32")a.value=arm::all_f32;else require(false,"projection exact arm name");
    require(m.at("nonce")==s.nonce&&m.at("source_base_revision")==ds4split_mmid::source_base&&m.at("binary_sha256")==s.binary_sha&&m.at("backend_binary_sha256")==s.backend_sha&&m.at("variant_manifest_sha256")==s.manifest_sha&&m.at("boundary_config_sha256")==s.boundary_sha,"projection inherited trusted pins differ");
    require(ds4split_mmid::integer(m.at("capture_bytes"))==s.cap&&ds4split_mmid::integer(m.at("existing_boundary_bytes"))==s.existing,"projection shared capture reservation differs");
    if(a.value!=arm::off)require(s.existing<=s.cap&&capture_maximum(a.value)<=s.cap-s.existing,"mixed projection capture maximum exceeds32MiB");
    const uint64_t wanted=a.value==arm::off?0:is_half(a.value)?mmf_device_cap:f32_device_cap;
    if(a.value!=arm::off)require(s.enabled,"active projection candidate requires explicit existing expert observer enabled=1");
    require(ds4split_mmid::integer(m.at("device_workspace_bytes"))==wanted&&ds4split_mmid::integer(m.at("host_memory_budget_bytes"))==(a.value==arm::off?0:host_reserve),"projection explicit host/device budget differs");
    a.nonce=s.nonce;a.config_sha=ds4split_mmid::sha(raw.data(),raw.size());
    std::array<bool,384> selected_set{};size_t selected_count=0;
    if(a.value==arm::off){require(m.at("selected_ids_hex")=="none"&&m.at("selected_ids_sha256")=="none","off arm has no input readback/pins");}
    else{
        const auto encoded=m.at("selected_ids_hex");require(ds4split_mmid::hex(encoded,1536)&&ds4split_mmid::hex(m.at("selected_ids_sha256"),64),"active projection ordered IDs pin required");
        std::array<unsigned char,768> bytes{};auto nibble=[](char c){return c<='9'?c-'0':c-'a'+10;};
        for(size_t i=0;i<bytes.size();++i)bytes[i]=static_cast<unsigned char>((nibble(encoded[2*i])<<4)|nibble(encoded[2*i+1]));
        require(ds4split_mmid::sha(bytes.data(),bytes.size())==m.at("selected_ids_sha256"),"ordered IDs hex/hash differ");
        for(int i=0;i<192;++i){const auto*p=bytes.data()+4*i;const uint32_t e=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);require(e<384,"ordered ID bounds");a.expected_ids[i]=int32_t(e);if(!selected_set[e]){selected_set[e]=true;++selected_count;}for(int r=0;r<i%6;++r)require(a.expected_ids[i-i%6+r]!=int32_t(e),"ordered IDs duplicate expert in token");}
    }
    if(is_half(a.value)){
        require(m.at("native_moe_shader_sha256")=="19fd3a81b4e03568f5a5d2275b77a4faca9570c005936d9e5a57ea41c11bb669","native dequant shader pin differs");
        a.proof_sha=m.at("native_dequant_proof_sha256");require(ds4split_mmid::hex(a.proof_sha,64),"native half proof SHA required");
        require(!proof.empty()&&proof.size()<=256*1024&&proof.find('\0')==std::string::npos,"native half proof size/NUL");
        require(ds4split_mmid::sha(proof.data(),proof.size())==a.proof_sha,"native half proof file SHA differs");
        std::istringstream in(proof);std::string header;require(bool(std::getline(in,header)),"native half proof header");if(!header.empty()&&header.back()=='\r')header.pop_back();require(header=="ds4split-native-q4k-half-natural-layout-v1","native half proof schema");
        std::string line;
        while(std::getline(in,line)){
            if(!line.empty()&&line.back()=='\r')line.pop_back();require(!line.empty(),"empty native half proof entry");
            int slot=0,e=0;std::string q,h,extra;std::istringstream row(line);
            require(bool(row>>slot>>e>>q>>h)&&!(row>>extra),"native half proof must have exactly four columns");
            require(slot>=0&&slot<3&&e>=0&&e<384&&ds4split_mmid::hex(q,64)&&ds4split_mmid::hex(h,64),"native half proof entry");
            require(a.half_proofs.emplace(std::make_pair(slot,e),golden{q,h}).second,"duplicate native half proof entry");
        }
        require(!a.half_proofs.empty(),"empty native half proof");
        require(a.half_proofs.size()==3*selected_count,"native half proof must cover exactly all three matrices of every selected expert");
        for(int slot=0;slot<3;++slot)for(int e=0;e<384;++e)require(a.half_proofs.count({slot,e})==size_t(selected_set[e]),"native half proof incomplete selected set or extra expert");
    }else{
        require(m.at("native_dequant_proof_sha256")=="none"&&m.at("native_moe_shader_sha256")=="none","F32/off arm is not a native half proof");
    }
    return a;
}
inline authorization load_trusted(){
    const char*flag=std::getenv("DSV41_BOUNDARY_CAPTURE");if(!flag||!*flag)return authorization{};
    require(std::strcmp(flag,"1")==0,"projection requires existing exact trusted capture opt-in");
    auto&s=ds4split_mmid::state();std::lock_guard<std::mutex> lock(s.mutex);s.init();
    // Existing observer verifies actual binary/module + boundary/manifest SHA,
    // and holds non-reparse ancestors. CPU fixtures use the same pure settings.
    const auto raw=read_bounded("expert-projection.config",4096);const auto fields=parse_exact(raw);const auto arm_name=fields.at("arm");
    const auto proof=(arm_name=="gateup-mmf"||arm_name=="down-mmf"||arm_name=="all-mmf")?read_bounded("expert-native-dequant-proof.txt",256*1024):std::string();
    return authorize_fields(raw,s,proof);
}
struct process_state {
    std::mutex mutex;
    bool initialized=false,done[3]={false,false,false};
    authorization auth;
    std::string route[3],record[3];
};
inline process_state& process(){static process_state p;return p;}
#if !defined(DS4SPLIT_PROJECTION_CPU_TEST)
struct device_arena {
    char*p=nullptr;size_t bytes=0;cudaStream_t stream;
    explicit device_arena(size_t n,cudaStream_t s):bytes(n),stream(s){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p),n));}
    ~device_arena(){if(p){CUDA_CHECK(cudaStreamSynchronize(stream));CUDA_CHECK(cudaFree(p));}}
    device_arena(const device_arena&)=delete;
};
struct reference_handle {
    cublasHandle_t value=nullptr;cudaStream_t stream;
    explicit reference_handle(cudaStream_t s):stream(s){}
    void create(){CUBLAS_CHECK(cublasCreate(&value));}
    ~reference_handle(){if(value){CUDA_CHECK(cudaStreamSynchronize(stream));CUBLAS_CHECK(cublasDestroy(value));}}
    reference_handle(const reference_handle&)=delete;
};
struct reference_modes {
    cublasMath_t math=CUBLAS_DEFAULT_MATH;cublasAtomicsMode_t atomics=CUBLAS_ATOMICS_ALLOWED;
    cublasPointerMode_t pointer=CUBLAS_POINTER_MODE_DEVICE;cudaStream_t stream=nullptr;
    cublasEmulationStrategy_t emulation=CUBLAS_EMULATION_STRATEGY_DEFAULT;
    cudaEmulationSpecialValuesSupport special=CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    cudaEmulationMantissaControl mantissa=CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    int version=0,sm_target=0,max_mantissa=0,mantissa_offset=0;int*mantissa_pointer=nullptr;
    void observe(cublasHandle_t h,cudaStream_t expected){
        CUBLAS_CHECK(cublasGetMathMode(h,&math));CUBLAS_CHECK(cublasGetAtomicsMode(h,&atomics));CUBLAS_CHECK(cublasGetPointerMode(h,&pointer));CUBLAS_CHECK(cublasGetStream(h,&stream));CUBLAS_CHECK(cublasGetVersion(h,&version));CUBLAS_CHECK(cublasGetSmCountTarget(h,&sm_target));
        CUBLAS_CHECK(cublasGetEmulationStrategy(h,&emulation));CUBLAS_CHECK(cublasGetEmulationSpecialValuesSupport(h,&special));CUBLAS_CHECK(cublasGetFixedPointEmulationMantissaControl(h,&mantissa));CUBLAS_CHECK(cublasGetFixedPointEmulationMaxMantissaBitCount(h,&max_mantissa));CUBLAS_CHECK(cublasGetFixedPointEmulationMantissaBitOffset(h,&mantissa_offset));CUBLAS_CHECK(cublasGetFixedPointEmulationMantissaBitCountPointer(h,&mantissa_pointer));
        require(math==CUBLAS_PEDANTIC_MATH&&atomics==CUBLAS_ATOMICS_NOT_ALLOWED&&pointer==CUBLAS_POINTER_MODE_HOST&&stream==expected,"actual reference handle math/atomics/pointer/stream differs from declared pedantic contract");
    }
    std::string json()const{
        std::ostringstream o;o<<"{\"compute_type_argument\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",\"compute_type_argument_i32\":"<<int(CUBLAS_COMPUTE_32F_PEDANTIC)<<",\"math_getter_i32\":"<<int(math)<<",\"math_getter_name\":\"CUBLAS_PEDANTIC_MATH\",\"atomics_getter_i32\":"<<int(atomics)<<",\"pointer_getter_i32\":"<<int(pointer)<<",\"stream_getter_u64\":"<<uint64_t(reinterpret_cast<uintptr_t>(stream))<<",\"version_getter\":"<<version<<",\"sm_target_getter\":"<<sm_target<<",\"emulation_strategy_getter_i32\":"<<int(emulation)<<",\"emulation_special_values_getter_i32\":"<<int(special)<<",\"fixedpoint_mantissa_control_getter_i32\":"<<int(mantissa)<<",\"fixedpoint_max_mantissa_getter\":"<<max_mantissa<<",\"fixedpoint_mantissa_offset_getter\":"<<mantissa_offset<<",\"fixedpoint_mantissa_pointer_getter_u64\":"<<uint64_t(reinterpret_cast<uintptr_t>(mantissa_pointer))<<",\"precision_emulation_workspace_environment_overrides_absent\":true}";return o.str();
    }
};
inline ggml_tensor matrix(const ggml_tensor*base,ggml_type type,int64_t k,int64_t n,void*data){
    ggml_tensor t=*base;t.type=type;t.data=data;t.op=GGML_OP_NONE;
    // Direct MMF reads explicit data/ne/nb only. These are borrowed cudaMalloc
    // spans, never provider-owned tensors or scheduler-managed allocations.
    t.buffer=nullptr;t.view_src=nullptr;t.view_offs=0;t.extra=nullptr;
    for(int i=0;i<GGML_MAX_SRC;++i)t.src[i]=nullptr;
    t.ne[0]=k;t.ne[1]=n;t.ne[2]=t.ne[3]=1;
    t.nb[0]=ggml_type_size(type);for(int i=1;i<4;++i)t.nb[i]=t.nb[i-1]*t.ne[i-1];return t;
}
class execution {
    const authorization&a;ggml_backend_cuda_context&ctx;ggml_tensor*dst;int slot;
    // The selected arm bypasses original call_scope, so it can exclusively reuse
    // the already admitted 4MiB observer arena. No second host arena is allocated.
    std::vector<unsigned char>&readback;
    std::unique_lock<std::mutex> observer_lock;
    std::string gpu_sha(const void*p,size_t n){
        ds4split_mmid::digest digest;
        for(size_t off=0;off<n;){ds4split_mmid::physical_ram();size_t take=std::min(n-off,readback.size());CUDA_CHECK(cudaMemcpyAsync(readback.data(),static_cast<const char*>(p)+off,take,cudaMemcpyDeviceToHost,ctx.stream()));CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));digest.update(readback.data(),take);off+=take;}
        return digest.finish();
    }
    void capture_blob(const char*tag,const void*p,size_t n){
        require(n&&n<=readback.size(),"projection capture span exceeds4MiB arena");ds4split_mmid::physical_ram();
        CUDA_CHECK(cudaMemcpyAsync(readback.data(),p,n,cudaMemcpyDeviceToHost,ctx.stream()));CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        auto&s=ds4split_mmid::state();require(observer_lock.owns_lock(),"projection capture requires exclusive shared arena lock");
        s.write_host("projection-"+a.nonce+"."+std::to_string(slot)+"."+tag+".bin",readback.data(),n);
    }
public:
    execution(const authorization&auth,ggml_backend_cuda_context&c,ggml_tensor*d,int s):a(auth),ctx(c),dst(d),slot(s),readback(ds4split_mmid::state().arena),observer_lock(){}
    bool run(){
        if(!applies(a.value,slot))return false;
        verify_reference_environment(a.value);
        auto&owner=ds4split_mmid::state();observer_lock=std::unique_lock<std::mutex>(owner.mutex);
        require(owner.enabled&&!owner.in_call&&!owner.done[slot]&&readback.size()==host_arena_bytes,"projection shared observer arena/profile state");
        require(slot>=0&&slot<3&&ds4split_mmid::target(dst)==slot,"projection target identity");
        const auto*w=dst->src[0];const auto*x=dst->src[1];const auto*ids=dst->src[2];
        const int64_t k=slot==2?2304:5120,nout=slot==2?5120:2304,channels=slot==2?6:1;
        require(w&&x&&ids&&w->type==GGML_TYPE_Q4_K&&x->type==GGML_TYPE_F32&&ids->type==GGML_TYPE_I32&&dst->type==GGML_TYPE_F32,"projection original types");
        require(w->buffer&&ggml_backend_buft_is_cuda(w->buffer->buft)&&!ggml_backend_buft_is_cuda_split(w->buffer->buft),"projection requires original resident non-split CUDA provider weights");
        require(w->ne[0]==k&&w->ne[1]==nout&&w->ne[2]==384&&w->ne[3]==1&&x->ne[0]==k&&x->ne[1]==channels&&x->ne[2]==32&&x->ne[3]==1&&ids->ne[0]==6&&ids->ne[1]==32&&ids->ne[2]==1&&ids->ne[3]==1&&dst->ne[0]==nout&&dst->ne[1]==6&&dst->ne[2]==32&&dst->ne[3]==1,"projection exact count32/top6 geometry");
        require(ggml_is_contiguous(w)&&ggml_is_contiguous(x)&&ggml_is_contiguous(ids)&&ggml_is_contiguous(dst)&&ggml_get_op_params_i32(dst,0)==0,"projection strides/default precision");
        require(w->nb[2]==6635520&&k*nout*2==half_matrix,"projection original matrix byte range");
        cudaStreamCaptureStatus capture;CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(),&capture));require(capture==cudaStreamCaptureStatusNone,"projection diagnostic graph capture forbidden");
        ds4split_mmid::physical_ram(host_reserve);
        int32_t selected[192];CUDA_CHECK(cudaMemcpyAsync(selected,ids->data,sizeof(selected),cudaMemcpyDeviceToHost,ctx.stream()));CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        require(std::memcmp(selected,a.expected_ids.data(),sizeof(selected))==0,"actual GPU ordered IDs differ from trusted canonical seed");
        auto columns=ordered_columns(selected);
        const bool half=is_half(a.value);const size_t arena_size=half?mmf_explicit_arena:f32_explicit_arena,weight_bytes=half?half_matrix:float_matrix;
        size_t free0=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free0,&total));const size_t before_free=free0,admitted=half?mmf_device_cap:f32_device_cap;
        require(free0>=admitted+64*mib,"projection VRAM reserve admission refused");
        require_allocation(observe_allocation(before_free,before_free,total,admitted,arena_size),"before_arena",slot,-1,0);
        device_arena scratch(arena_size,ctx.stream());char*weight=scratch.p;float*rhs=reinterpret_cast<float*>(weight+weight_bytes);float*out=reinterpret_cast<float*>(reinterpret_cast<char*>(rhs)+rhs_max);
        require(weight_bytes+rhs_max+output_max<=arena_size,"projection exact arena accounting");
        reference_handle reference(ctx.stream());reference_modes modes;std::string initial_modes;
        if(!half){
            // Explicit F32 counterfactual only, never labelled native-equivalent.
            // Actual pinned probe completed32 shapes/67 calls: handle phase added
            // 66MiB globally, no later converter/GEMM/template growth. Reserve
            // 96MiB opaque/runtime separately from the unchanged96MiB arena.
            reference.create();CUBLAS_CHECK(cublasSetStream(reference.value,ctx.stream()));
            CUBLAS_CHECK(cublasSetMathMode(reference.value,CUBLAS_PEDANTIC_MATH));CUBLAS_CHECK(cublasSetAtomicsMode(reference.value,CUBLAS_ATOMICS_NOT_ALLOWED));CUBLAS_CHECK(cublasSetPointerMode(reference.value,CUBLAS_POINTER_MODE_HOST));
            char*workspace=reinterpret_cast<char*>(out)+output_max;
            CUBLAS_CHECK(cublasSetWorkspace(reference.value,workspace,32*mib));
            modes.observe(reference.value,ctx.stream());initial_modes=modes.json();
            CUDA_CHECK(cudaMemGetInfo(&free0,&total));require_allocation(observe_allocation(before_free,free0,total,admitted,scratch.bytes),"post_reference_handle",slot,-1,0);
        }
        std::ostringstream rows;int experts=0,calls=0;
        for(int e=0;e<384;++e){
            auto&cols=columns[size_t(e)];if(cols.empty())continue;require(cols.size()<=32,"projection per-expert columns bound");++experts;
            const char*original=static_cast<const char*>(w->data)+size_t(e)*w->nb[2];
            const auto q4hash=gpu_sha(original,size_t(w->nb[2]));
            if(half){auto convert=ggml_get_to_fp16_cuda(GGML_TYPE_Q4_K);require(convert,"projection Q4_K→F16 converter absent");convert(original,reinterpret_cast<::half*>(weight),k*nout,ctx.stream());}
            else{auto convert=ggml_get_to_fp32_cuda(GGML_TYPE_Q4_K);require(convert,"projection Q4_K→F32 converter absent");convert(original,reinterpret_cast<float*>(weight),k*nout,ctx.stream());}
            CUDA_CHECK(cudaGetLastError());const auto converted_hash=gpu_sha(weight,weight_bytes);
            if(half){auto p=a.half_proofs.find({slot,e});require(p!=a.half_proofs.end()&&p->second.q4_sha==q4hash&&p->second.half_sha==converted_hash,"actual CUDA F16 weight bits do not match pinned native Metal half proof");}
            // Normal non-ID MMF selector permits <=16 columns; batching preserves
            // the one weight matrix and allows the worst-case32 selected tokens.
            for(size_t start=0;start<cols.size();start+=16){
                const size_t nc=std::min<size_t>(16,cols.size()-start);
                for(size_t col=0;col<nc;++col){const int dstcol=cols[start+col],srcrow=channels==1?dstcol/6:dstcol;CUDA_CHECK(cudaMemcpyAsync(rhs+col*k,static_cast<const float*>(x->data)+size_t(srcrow)*k,size_t(k)*4,cudaMemcpyDeviceToDevice,ctx.stream()));}
                if(half){
                    auto wa=matrix(w,GGML_TYPE_F16,k,nout,weight);auto xa=matrix(x,GGML_TYPE_F32,k,nc,rhs);auto ya=matrix(dst,GGML_TYPE_F32,nout,nc,out);ya.src[0]=&wa;ya.src[1]=&xa;
                    require(ggml_cuda_should_use_mmf(GGML_TYPE_F16,ggml_cuda_info().devices[ctx.device].cc,ggml_cuda_info().devices[ctx.device].warp_size,wa.ne,wa.nb,int(nc),false),"per-expert actual MMF shape/CC unsupported; no alternate backend");
                    ggml_cuda_mul_mat_f(ctx,&wa,&xa,nullptr,&ya); // Actual existing CUDA half2 MMF kernel, F32 accumulator/output.
                }else{
                    verify_reference_environment(a.value);modes.observe(reference.value,ctx.stream());require(modes.json()==initial_modes,"reference actual handle getters changed before GEMM");
                    const float alpha=1,beta=0;
                    CUBLAS_CHECK(cublasGemmEx(reference.value,CUBLAS_OP_T,CUBLAS_OP_N,int(nout),int(nc),int(k),&alpha,weight,CUDA_R_32F,int(k),rhs,CUDA_R_32F,int(k),&beta,out,CUDA_R_32F,int(nout),CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
                }
                ++calls;CUDA_CHECK(cudaGetLastError());
                for(size_t col=0;col<nc;++col)CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(dst->data)+size_t(cols[start+col])*nout,out+col*nout,size_t(nout)*4,cudaMemcpyDeviceToDevice,ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream())); // Required before rhs/output/matrix reuse.
                CUDA_CHECK(cudaMemGetInfo(&free0,&total));require_allocation(observe_allocation(before_free,free0,total,admitted,scratch.bytes),"post_projection_batch",slot,e,calls);
            }
            if(experts>1)rows<<",";
            rows<<"{\"expert\":"<<e<<",\"source_q4k_bytes\":"<<w->nb[2]<<",\"source_q4k_offset_bytes\":"<<uint64_t(e)*w->nb[2]<<",\"source_q4k_sha256\":\""<<q4hash<<"\",\"actual_dequantized_weight_sha256\":\""<<converted_hash<<"\",\"columns\":[";
            for(size_t i=0;i<cols.size();++i)rows<<(i?",":"")<<cols[i];rows<<"]}";
        }
        require(experts>0&&calls>=experts&&calls<=2*experts,"projection actual call/selected expert witness");
        if(!half){verify_reference_environment(a.value);modes.observe(reference.value,ctx.stream());require(modes.json()==initial_modes,"reference actual handle getters changed after GEMMs");}
        const auto input_hash=gpu_sha(x->data,ggml_nbytes(x)),ids_hash=gpu_sha(ids->data,ggml_nbytes(ids));
        const auto output_hash=gpu_sha(dst->data,ggml_nbytes(dst));
        capture_blob("source_f32",x->data,ggml_nbytes(x));capture_blob("selected_i32",ids->data,ggml_nbytes(ids));capture_blob("output_f32",dst->data,ggml_nbytes(dst));
        std::ostringstream receipt;receipt<<"{\"schema\":\"ds4split-expert-projection-call-v1\",\"nonce\":\""<<a.nonce<<"\",\"config_sha256\":\""<<a.config_sha<<"\",\"slot\":"<<slot<<",\"actual_backend\":\""<<(half?"direct_per_expert_mmf_half2_float_accumulator":"direct_per_expert_cublas_compute32f_pedantic_counterfactual")<<"\",\"actual_gpu_calls\":"<<calls<<",\"one_matrix_reused\":true,\"device_admitted_bytes\":"<<admitted<<",\"host_admitted_bytes\":"<<host_reserve<<",\"output_f32_sha256\":\""<<output_hash<<"\",\"native_equivalence_claimed\":false,\"qualification_passed\":false,\"experts\":["<<rows.str()<<"]}\n";
        std::string bytes=receipt.str();const auto pos=bytes.rfind("}\n");require(pos!=std::string::npos,"projection internal JSON");
        const std::string prefix="projection-"+a.nonce+"."+std::to_string(slot);
        std::ostringstream envelope;
        envelope<<",\"reference_handle_modes\":"<<(half?"null":initial_modes)<<",\"source_f32_sha256\":\""<<input_hash<<"\",\"selected_i32_sha256\":\""<<ids_hash<<"\",\"weight_source_pointer_u64\":"<<uint64_t(reinterpret_cast<uintptr_t>(w->data))
            <<",\"device_explicit_arena_bytes\":"<<arena_size<<",\"first_use_runtime_code_reserve_bytes\":"<<(half?mmf_runtime_code_reserve:0)
            <<",\"cublas_handle_runtime_reserve_bytes\":"<<(half?0:cublas_opaque_reserve)
            <<",\"source_base_revision\":\""<<ds4split_mmid::source_base<<"\",\"observer_config_sha256\":\""<<owner.config_sha<<"\",\"capture_cap_bytes\":"<<owner.cap
            <<",\"source_shader_sha256\":\""<<(half?"19fd3a81b4e03568f5a5d2275b77a4faca9570c005936d9e5a57ea41c11bb669":"not-native-reference")
            <<"\",\"binary_sha256\":\""<<owner.binary_sha<<"\",\"backend_binary_sha256\":\""<<owner.backend_sha<<"\",\"variant_manifest_sha256\":\""<<owner.manifest_sha<<"\",\"boundary_config_sha256\":\""<<owner.boundary_sha<<"\",\"native_dequant_proof_sha256\":\""<<a.proof_sha<<"\",\"device\":"<<ctx.device<<",\"cc\":"<<ggml_cuda_info().devices[ctx.device].cc<<",\"stream_pointer_u64\":"<<uint64_t(reinterpret_cast<uintptr_t>(ctx.stream()))<<",\"weight_name\":\""<<w->name<<"\",\"op_precision\":"<<ggml_get_op_params_i32(dst,0)<<",\"tensor_geometry\":{";
        const ggml_tensor*tensors[]={w,x,ids,dst};const char*names[]={"weight","source","ids","output"};
        for(int t=0;t<4;++t){if(t)envelope<<",";envelope<<"\""<<names[t]<<"\":{\"type\":"<<int(tensors[t]->type)<<",\"ne\":[";for(int i=0;i<4;++i)envelope<<(i?",":"")<<tensors[t]->ne[i];envelope<<"],\"nb\":[";for(int i=0;i<4;++i)envelope<<(i?",":"")<<tensors[t]->nb[i];envelope<<"]}";}envelope<<"},\"files\":[";
        const char*tags[]={"source_f32","selected_i32","output_f32"};const size_t sizes[]={ggml_nbytes(x),ggml_nbytes(ids),ggml_nbytes(dst)};const std::string hashes[]={input_hash,ids_hash,output_hash};
        for(int i=0;i<3;++i)envelope<<(i?",":"")<<"{\"name\":\""<<prefix<<"."<<tags[i]<<".bin\",\"bytes\":"<<sizes[i]<<",\"sha256\":\""<<hashes[i]<<"\"}";envelope<<"]";
        bytes.insert(pos,envelope.str());
        require(bytes.size()<=128*1024,"projection receipt size");owner.write_host("projection-"+a.nonce+"."+std::to_string(slot)+".json",bytes.data(),bytes.size());return true;
    }
};
class call_scope {
    process_state*p=nullptr;ggml_backend_cuda_context*ctx=nullptr;ggml_tensor*dst=nullptr;
    int slot=-1;bool finished=false,projected=false;std::string actual_route;
    std::unique_lock<std::mutex> lock;
    static std::string slot_name(int i){return i==0?"gate":i==1?"up":"down";}
public:
    call_scope(ggml_backend_cuda_context&c,ggml_tensor*d):lock(){
        const char*f=std::getenv("DSV41_BOUNDARY_CAPTURE");if(!f||!*f)return;
        require(std::strcmp(f,"1")==0,"projection exact existing capture flag required");slot=ds4split_mmid::target(d);if(slot<0)return;
        p=&process();lock=std::unique_lock<std::mutex>(p->mutex,std::try_to_lock);require(lock.owns_lock(),"concurrent expert projection request refused");
        if(!p->initialized){p->auth=load_trusted();p->initialized=true;}
        auto&owner=ds4split_mmid::state();
        const auto raw=read_bounded("expert-projection.config",4096);
        require(ds4split_mmid::sha(raw.data(),raw.size())==p->auth.config_sha,"projection trusted config changed during process");
        for(auto pair:{std::pair<const char*,std::string>{"boundary-capture.config",owner.boundary_sha},{"boundary-capture.source-manifest.json",owner.manifest_sha}}){auto b=read_bounded(pair.first,4096);require(ds4split_mmid::sha(b.data(),b.size())==pair.second,"projection boundary/source pins changed during process");}
        if(p->auth.value==arm::off){p=nullptr;lock.unlock();return;}
        verify_reference_environment(p->auth.value);
        require(!p->done[slot],"duplicate expert projection request refused");
        // Exact binary mixed-profile accounting plus declared JSON maxima. This
        // is smaller than the original all-MMQ profile but checked independently.
        const uint64_t maximum=capture_maximum(p->auth.value);
        require(owner.existing<=owner.cap&&maximum<=owner.cap-owner.existing,"mixed projection capture maximum exceeds32MiB");
        ctx=&c;dst=d;
    }
    ~call_scope(){if(p&&!finished)require(false,"projection request exited without actual backend completion");}
    bool run(){
        if(!p||!applies(p->auth.value,slot))return false;
        {execution selected(p->auth,*ctx,dst,slot);require(selected.run(),"selected projection did not run");}
        projected=true;actual_route=is_half(p->auth.value)?"direct_per_expert_mmf_half2_float_accumulator":"direct_per_expert_cublas_compute32f_pedantic_counterfactual";
        // selected's shared observer mutex must release before mixed completion.
        finish();return true;
    }
    void route(const char*name){if(!p)return;require(actual_route.empty(),"duplicate projection original-route witness");actual_route=name;}
    void finish(){
        if(!p)return;require(!finished&&!actual_route.empty(),"projection missing actual route witness");
        auto&owner=ds4split_mmid::state();
        if(!projected)require(actual_route=="mmq_id"&&owner.done[slot],"unselected projection did not finish original observed MMQ_ID");
        p->done[slot]=true;p->route[slot]=actual_route;p->record[slot]=projected?"projection-"+p->auth.nonce+"."+std::to_string(slot)+".json":"mmid-"+p->auth.nonce+"."+slot_name(slot)+".json";finished=true;
        if(p->done[0]&&p->done[1]&&p->done[2]){
            std::lock_guard<std::mutex> observer(owner.mutex);
            std::ostringstream m;m<<"{\"schema\":\"ds4split-mixed-expert-projection-v1\",\"nonce\":\""<<p->auth.nonce<<"\",\"projection_config_sha256\":\""<<p->auth.config_sha<<"\",\"binary_sha256\":\""<<owner.binary_sha<<"\",\"backend_binary_sha256\":\""<<owner.backend_sha<<"\",\"variant_manifest_sha256\":\""<<owner.manifest_sha<<"\",\"boundary_config_sha256\":\""<<owner.boundary_sha<<"\",\"three_original_calls\":false,\"three_requested_expert_nodes\":true,\"actual_dispatch_observed\":true,\"qualification_passed\":false,\"native_equivalence_claimed\":false,\"existing_boundary_bytes\":"<<owner.existing<<",\"used_before_completion\":"<<owner.used<<",\"cap_bytes\":"<<owner.cap<<",\"calls\":[";
            for(int i=0;i<3;++i)m<<(i?",":"")<<"{\"slot\":"<<i<<",\"actual_route\":\""<<p->route[i]<<"\",\"record\":\""<<p->record[i]<<"\"}";m<<"]}\n";const auto b=m.str();require(b.size()<=4096,"mixed expert completion manifest exceeds4KiB");owner.write_host("projection-"+p->auth.nonce+".json",b.data(),b.size());
        }
    }
};
#endif // !DS4SPLIT_PROJECTION_CPU_TEST
#else
class call_scope {
public:
    call_scope(ggml_backend_cuda_context&,ggml_tensor*){}
    bool run(){return false;}
    void route(const char*){}
    void finish(){}
};
#endif
} // namespace ds4split_projection_candidate
