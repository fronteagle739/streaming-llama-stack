extern "C" unsigned long long ggml_backend_cuda_dsv41_pv_strided_queued(void);
#include "ram-output-budget.h"
#if defined(_WIN32) && defined(GGML_USE_CUDA)
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include "ggml-cuda.h"
#include "ggml-dsv41-gather-half.h"
#include "ggml-dsv41-split-f32.h"
extern "C" uint64_t ggml_backend_cuda_dsv41_pv_view_queued(void);
#include "ggml-dsv41-carry-copy.h"
#include "ggml-ds4-native-half-stats.h"
#include "llama-dsv41-sweep.h"
#include "profile-check.h"
#include "llama-dsv41-host-memory.h"
#include "llama-dsv41-native-columns-resource-proof.h"
#include "llama-dsv41-native-loader-resource-proof.h"
#include "llama-dsv41-native-rhs-pack-proof.h"
#ifndef DS4PERF_NATIVE_LOADER_UNROLL
#define DS4PERF_NATIVE_LOADER_UNROLL 0
#endif
static_assert(DS4PERF_NATIVE_LOADER_UNROLL==0||DS4PERF_NATIVE_LOADER_UNROLL==1,"Unqualified native loader build");
#ifndef DS4PERF_NATIVE_COLUMNS
#define DS4PERF_NATIVE_COLUMNS 16
#endif
static_assert(DS4PERF_NATIVE_COLUMNS==16||DS4PERF_NATIVE_COLUMNS==32,"Unqualified native expert column build");
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using json=nlohmann::json;
using clock_type=std::chrono::steady_clock;
constexpr uint64_t reserve=30ull*1024*1024*1024,receipt_limit=64*1024,chunk=1024*1024;
constexpr const char*model_name="DeepSeek-V4.1-Flash-1017-Q4-v41.gguf";
constexpr const char*model_sha="2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f";
constexpr uint64_t epoch_ticks=116444736000000000ull;
constexpr uint64_t gib=1024ull*1024*1024,cli_host_bytes=64*1024*1024;
std::atomic<bool>interrupted{false},memory_abort{false};
BOOL WINAPI interrupt(DWORD kind){if(kind==CTRL_C_EVENT||kind==CTRL_BREAK_EVENT||kind==CTRL_CLOSE_EVENT){interrupted.store(true);return TRUE;}return FALSE;}
void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
void success(bool value,const std::string&error){if(!value)throw std::runtime_error(error.empty()?"Offline producer operation failed":error);}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,"Producer capacity overflow");return a+b;}
double ms(clock_type::time_point start){return std::chrono::duration<double,std::milli>(clock_type::now()-start).count();}
bool should_stop(){
    if(!llama_dsv41_host_memory::live())memory_abort.store(true);
    return interrupted.load()||memory_abort.load();
}
void stop(){require(!should_stop(),memory_abort.load()?"Producer stopped at the1GiB physicalRAM/OScommit floor; incomplete output retained":"Producer cancelled; incomplete output retained and not certified");}
std::string utf8(const std::wstring&s){
    const int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,s.data(),int(s.size()),nullptr,0,nullptr,nullptr);
    require(n>0,"Invalid Unicode command argument");std::string out(n,'\0');
    require(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,s.data(),int(s.size()),out.data(),n,nullptr,nullptr)==n,"Cannot encode command argument");return out;
}
bool hex_sha(const std::string&s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](unsigned char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
uint64_t integer(const std::string&s){
    require(!s.empty()&&std::all_of(s.begin(),s.end(),[](unsigned char c){return c>='0'&&c<='9';}),"Expected unsigned decimal integer bytes/count");
    size_t used=0;const auto value=std::stoull(s,&used);require(used==s.size(),"Invalid integer option");return value;
}
uint64_t exact_u64(const json&v){require(v.is_number_unsigned(),"Receipt field is not an exact unsigned integer (floating-point receipts are rejected)");return v.get<uint64_t>();}
struct hash {
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE handle=nullptr;std::vector<uint8_t>object;
    hash(){
        require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"Cannot open native SHA256");
        DWORD bytes=0,n=0;require(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&bytes),sizeof(bytes),&n,0)>=0&&bytes<=65536,"Invalid SHA workspace");
        object.resize(bytes);require(BCryptCreateHash(algorithm,&handle,object.data(),bytes,nullptr,0,0)>=0,"Cannot create native SHA256");
    }
    ~hash(){if(handle)BCryptDestroyHash(handle);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);}
    void update(const void*data,size_t bytes){require(bytes<=chunk&&BCryptHashData(handle,(PUCHAR)data,ULONG(bytes),0)>=0,"Native SHA256 update failed");}
    std::string finish(){std::array<uint8_t,32>out{};require(BCryptFinishHash(handle,out.data(),ULONG(out.size()),0)>=0,"Native SHA256 finish failed");
        std::ostringstream text;text<<std::hex<<std::setfill('0');for(auto b:out)text<<std::setw(2)<<unsigned(b);return text.str();}
};
struct identity {uint64_t volume=0,size=0,mtime=0;std::array<uint8_t,16>id{};};
identity inspect(HANDLE file){
    FILE_ID_INFO id{};BY_HANDLE_FILE_INFORMATION info{};LARGE_INTEGER size{};
    require(GetFileInformationByHandleEx(file,FileIdInfo,&id,sizeof(id))&&GetFileInformationByHandle(file,&info)&&GetFileSizeEx(file,&size),"Cannot inspect native file identity");
    require(!(info.dwFileAttributes&(FILE_ATTRIBUTE_REPARSE_POINT|FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_SPARSE_FILE|FILE_ATTRIBUTE_COMPRESSED))&&info.nNumberOfLinks==1&&size.QuadPart>=0,
        "Producer requires a physical single-link ordinary file");
    identity out;out.volume=id.VolumeSerialNumber;out.size=uint64_t(size.QuadPart);std::memcpy(out.id.data(),id.FileId.Identifier,16);
    out.mtime=(uint64_t(info.ftLastWriteTime.dwHighDateTime)<<32)|info.ftLastWriteTime.dwLowDateTime;return out;
}
bool same(const identity&a,const identity&b){return a.volume==b.volume&&a.id==b.id&&a.size==b.size&&a.mtime==b.mtime;}
std::string id_hex(const std::array<uint8_t,16>&id){std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto b:id)s<<std::setw(2)<<unsigned(b);return s.str();}
json describe(const identity&id){return {{"volume_serial_u64",id.volume},{"file_id_le_hex",id_hex(id.id)},{"bytes",id.size},{"mtime_filetime_ticks",id.mtime}};}

// No unlink, replacement, truncation or cleanup operation exists here. The
// sole rename publishes our new flushed pending receipt with NO REPLACE.
// Held physical directory handles deny DELETE; before each native file action
// their identities are rebound to the current named chain as well.
class native_file {
    std::filesystem::path path;
    std::vector<HANDLE>parents;
    std::vector<std::filesystem::path>names;
    HANDLE file=INVALID_HANDLE_VALUE;
    identity owned{};
    uint64_t limit=0,pending=0,disk_floor=reserve;
    bool output=false,can_publish=false,flushed=false;
    void parents_ok()const{
        for(size_t i=0;i<parents.size();++i){
            BY_HANDLE_FILE_INFORMATION held{},named{};
            require(GetFileInformationByHandle(parents[i],&held)&&
                (held.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)&&!(held.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"Held output ancestor changed");
            HANDLE probe=CreateFileW(names[i].c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
            const bool valid=probe!=INVALID_HANDLE_VALUE&&GetFileInformationByHandle(probe,&named)&&
                (named.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)&&!(named.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT)&&
                held.dwVolumeSerialNumber==named.dwVolumeSerialNumber&&held.nFileIndexHigh==named.nFileIndexHigh&&held.nFileIndexLow==named.nFileIndexLow;
            if(probe!=INVALID_HANDLE_VALUE)CloseHandle(probe);require(valid,"Named output ancestor identity changed");
        }
    }
public:
    native_file()=default;
    ~native_file(){if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);for(auto h:parents)CloseHandle(h);}
    native_file(const native_file&)=delete;native_file&operator=(const native_file&)=delete;
    static uint64_t volume(const std::filesystem::path&p){
        require(p.is_absolute()&&p==p.lexically_normal()&&p.has_filename(),"Producer paths must be absolute and normalized");
        require(GetDriveTypeW(p.root_path().c_str())==DRIVE_FIXED,"Producer requires a fixed local volume");
        wchar_t device[32768]{};require(QueryDosDeviceW(p.root_name().c_str(),device,32768)&&std::wstring(device).find(L"\\Device\\HarddiskVolume")==0,
            "Producer requires a physical volume, not UNC/SUBST");
        HANDLE root=CreateFileW(p.root_path().c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(root!=INVALID_HANDLE_VALUE,"Cannot inspect producer volume");FILE_ID_INFO info{};
        const bool valid=GetFileInformationByHandleEx(root,FileIdInfo,&info,sizeof(info))!=0;CloseHandle(root);require(valid,"Cannot identify producer volume");return info.VolumeSerialNumber;
    }
    void open(const std::filesystem::path&p,bool create,uint64_t expected,uint64_t outstanding=0,bool publishable=false,uint64_t output_floor=reserve){
        require(file==INVALID_HANDLE_VALUE&&parents.empty(),"Native producer file is single-use");path=p;volume(path);
        require(path.filename().wstring().find(L':')==std::wstring::npos,"Alternate data streams are forbidden");
        std::vector<std::filesystem::path>chain;for(auto d=path.parent_path();;d=d.parent_path()){chain.push_back(d);if(d==d.root_path())break;}
        std::reverse(chain.begin(),chain.end());
        for(const auto&d:chain){
            HANDLE h=CreateFileW(d.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
            require(h!=INVALID_HANDLE_VALUE,"Cannot lease physical producer ancestor");parents.push_back(h);names.push_back(d);parents_ok();
        }
        limit=expected;pending=outstanding;disk_floor=output_floor;output=create;can_publish=publishable;parents_ok();if(create)budget();parents_ok();
        file=CreateFileW(path.c_str(),create?(GENERIC_READ|GENERIC_WRITE|(publishable?DELETE:0)):GENERIC_READ,FILE_SHARE_READ,nullptr,create?CREATE_NEW:OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"Cannot open exclusive-new output or read-only physical input");owned=inspect(file);
        require(create?owned.size==0:owned.size<=limit,"Producer file exceeds its bounded byte limit");check();
    }
    void check()const{parents_ok();require(same(owned,inspect(file)),"Native producer file identity/size/mtime changed");}
    void budget()const{
        parents_ok();ULARGE_INTEGER free{};require(GetDiskFreeSpaceExW(path.parent_path().c_str(),&free,nullptr,nullptr),"Cannot inspect actual output-volume free bytes");
        require(ds4perf_output_budget::fits(free.QuadPart,limit,owned.size,pending,disk_floor),"Producer output/carry/other pending bytes exceed actual-volume budget plus selected disk reserve");
    }
    void outstanding(uint64_t bytes){pending=bytes;}
    identity current()const{check();return owned;}
    size_t read(void*dst,size_t bytes){
        require(!output&&bytes<=chunk,"Invalid bounded input read");check();DWORD got=0;
        require(ReadFile(file,dst,DWORD(bytes),&got,nullptr),"Producer input read failed");check();return got;
    }
    void write(const void*src,size_t bytes){
        require(output&&bytes<=chunk&&owned.size<=limit&&bytes<=limit-owned.size,"Producer output exceeds planned bounded size");
        check();budget();flushed=false;DWORD wrote=0;require(WriteFile(file,src,DWORD(bytes),&wrote,nullptr)&&wrote==bytes,"Native output write failed; incomplete file retained");
        const auto after=inspect(file);require(after.volume==owned.volume&&after.id==owned.id&&after.size==owned.size+bytes,"Native output ownership or length changed");owned=after;check();
    }
    void flush(){check();require(FlushFileBuffers(file),"Native output flush failed; output is not certified");owned=inspect(file);check();flushed=true;}
    void finalize_read_only(const std::string&expected_sha){
        require(output&&flushed,"Payload must be flushed before closing its writer");check();const auto before=owned;
        require(CloseHandle(file),"Cannot close completed payload writer");file=INVALID_HANDLE_VALUE;parents_ok();
        file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"Cannot acquire completed payload readonly lease");const auto final=inspect(file);
        require(final.volume==before.volume&&final.id==before.id&&final.size==before.size,"Payload identity changed during writer-close/read-lease gap");
        owned=final;output=false;check();hash digest;std::vector<uint8_t>buffer(chunk);uint64_t count=0;
        for(;;){stop();const auto got=read(buffer.data(),buffer.size());if(!got)break;digest.update(buffer.data(),got);count+=got;}
        require(count==owned.size&&digest.finish()==expected_sha,"Closed payload differs from its incremental SHA; no receipt may publish");check();
    }
    void publish(const std::filesystem::path&target){
        require(output&&can_publish&&flushed,"Only an owned fully flushed pending receipt can publish");check();
        require(target.is_absolute()&&target==target.lexically_normal()&&target.has_filename()&&target.parent_path()==path.parent_path()&&
            target.filename().wstring().find(L':')==std::wstring::npos,"Receipt publication requires a sibling ordinary path");
        // Prepare every allocation/validation before publication. ReplaceIfExists
        // remains FALSE: a concurrent user file can never be overwritten.
        std::filesystem::path destination=target;const auto name=destination.wstring();
        // SetFileInformationByHandle reads the NUL-terminated FileName path.
        // Keep FileNameLength exclusive of NUL, but include NUL in the buffer.
        const size_t bytes=offsetof(FILE_RENAME_INFO,FileName)+(name.size()+1)*sizeof(wchar_t);
        require(bytes<=65536,"Receipt publish path exceeds bounded metadata");std::vector<uint8_t>buffer(bytes,0);
        auto*rename=reinterpret_cast<FILE_RENAME_INFO*>(buffer.data());rename->ReplaceIfExists=FALSE;rename->RootDirectory=nullptr;
        rename->FileNameLength=DWORD(name.size()*sizeof(wchar_t));std::memcpy(rename->FileName,name.data(),rename->FileNameLength);
        parents_ok();stop();require(SetFileInformationByHandle(file,FileRenameInfo,rename,DWORD(bytes)),"No-replace receipt publication failed; pending file retained");
        // No fallible operation follows the atomic publication.
        path.swap(destination);
    }
};
struct contents {std::vector<uint8_t>bytes;std::string sha;identity id;};
contents read_small(const std::filesystem::path&path,uint64_t maximum,bool retain=true){
    native_file file;file.open(path,false,maximum);contents out;out.id=file.current();hash digest;
    std::vector<uint8_t>buffer(chunk);uint64_t read=0;
    for(;;){stop();const auto count=file.read(buffer.data(),buffer.size());if(!count)break;read+=count;require(read<=maximum,"Input grew beyond bounded read admission");
        digest.update(buffer.data(),count);if(retain)out.bytes.insert(out.bytes.end(),buffer.begin(),buffer.begin()+count);}
    require(read==out.id.size,"Input byte count changed while reading");out.sha=digest.finish();return out;
}
std::filesystem::path executable(){wchar_t name[32768]{};const auto n=GetModuleFileNameW(nullptr,name,32768);require(n&&n<32768,"Cannot identify actual producer binary");return std::filesystem::path(std::wstring(name,n));}
uint64_t ram_admission(uint64_t admitted){MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory),"Cannot inspect available physical RAM");
    require(memory.ullAvailPhys>=add(admitted,gib),"Producer job host plan exceeds available physicalRAM minus1GiB");return memory.ullAvailPhys;}
const char*usage=
"Offline original-1017 native producer (Windows CUDA, explicit RTX5090 only)\n"
"Required --key value pairs:\n"
" --model --copy-receipt --tokens-le32 --cuda-device --context --build-sha\n"
" --provider-device-bytes --provider-host-bytes --cache-bytes --graph-bytes\n"
" --job-device-bytes --job-host-bytes --gpu-reserve-bytes --carry --carry-file-bytes --carry-host-bytes\n"
" --carry-outstanding-bytes --output-outstanding-bytes --payload --receipt\n"
"Optional --carry-storage file|ram|hybrid (default file; no automatic fallback).\n"
"Hybrid requires --carry-ram-prefix-tokens and --carry-disk-reserve-bytes; exclusive-new D: overflow, batched unbuffered tiles.\n"
"Optional --fetch-profile baseline|two-buffer-1500-auto|two-buffer-1500-auto-async (default baseline; qualification remains separate).\n"
"Optional --direct-carry-budget 0|1 (default0; pinned RAM leases, page-sized vector reads, explicit tighter capacity).\n"
"Optional --compute-tile 32|128|512|2048|4096 (default32; larger tiles require admitted RAM carry).\n"
"Optional --hidden-device-bytes <bytes> (default0; CUDA hidden bank with unchanged RAM mirror/checks).\n"
"Optional --dense-activation-q8-1 0|1 (default0; original Q8 dense weights, shared MMQ with Q8_1 activation rounding).\n"
"Optional --expert-weights-activations-fp4 0|1 (default0; hyper-fast only: leased expert FP4 weights and FP4 activations;<=4rows unchanged).\n"
"Optional --native-rhs-pack 0|1 (default0; qualified native half RHS packing,384MiB expert workspace).\n"
"Optional --pv-strided-probabilities 0|1 (default0; native F32 PV without probability copy).\n"
"Optional --reuse-child-raw-metadata 0|1 (default0; omit unused grouped child arrays).\n"
"Optional --unaligned-append-step-tokens 0|2048 (default0; explicit bounded continuation after Mac decode, unchanged fixed owners).\n"
"Optional --resident-pageable-copies 0|1 (default0; resident paired RAM mirrors use existing bounded backend vector copies).\n"
"Optional --tile-host-registration 0|1 (default0; paired full-resident plan registers only its current RAM lease).\n"
"Optional --pinned-device-bytes <bytes> (default0; native-expert direct pinned RAM copies/GPU validation; minimum67108864).\n"
"Optional --reuse-graph 0|1|2 (default0;1 keeps allocator;2 also reuses layer graph variants).\n"
"Optional --cast-device-bytes <bytes> (default0; reuse existing half weight casts in allocator mode1); --cast-audit 0|1.\n"
"Optional --engram-prefetch-bytes <bytes> (default0; whole-prompt raw row cache on its own IoRing).\n"
"Optional --prefetch-device-bytes <bytes> (default0; bounded next-tile pinned H2D overlap).\n"
"Optional --ds-gpu-ring-slots 0|2 (default0; one-shot TEST full-pass queue).\n"
"Optional --ds-submit-threads 0|1|2|4; --cuda-blocking-sync 0|1 (default0; scheduling TEST).\n"
"Optional --index-cublas-device-bytes <bytes> (default0; experimental exact-dot GLM reuse; minimum67108864).\n"
"RAM mode requires zero carry-file/outstanding bytes and admits the full bank in carry-host-bytes; --carry is unused.\n"
"build-sha is the expected SHA256 of this executable. All paths are absolute.\n"
"Outputs must not exist; incomplete outputs remain on failure. No preset is enabled.\n";
}

#include "live-session.inc"

int wmain(int argc,wchar_t**argv){
    // Backend is declared first so the sweep (which synchronizes on teardown)
    // is destroyed before its externally owned backend on every return path.
    ggml_backend_ptr backend;session_output_redirect session_output;
    std::unique_ptr<native_file>payload,receipt;std::unique_ptr<llama_dsv41_sweep>sweep;
    json report={{"schema","dsv41-offline-producer-v1"},{"status","failed"},{"runtime_qualified",false}};
    const auto process_started=clock_type::now();
    try{
        static std::array<char,1024*1024> stderr_storage;
        const char *buffer_env=std::getenv("DS4PERF_STDERR_BUFFER_BYTES");
        const bool buffered_stderr=buffer_env&&std::string(buffer_env)=="1048576";
        require(!buffer_env||buffered_stderr,"Diagnostic stderr buffer requires exactly1048576bytes");
        if(buffered_stderr)require(std::setvbuf(stderr,stderr_storage.data(),_IOFBF,stderr_storage.size())==0,"Cannot configure existing CRT stderr buffer");
        report["stderr_buffer_bytes"]=buffered_stderr?stderr_storage.size():0;
        if(argc==2&&std::wstring(argv[1])==L"--help"){std::cout<<usage;return 0;}
        if(argc==2&&std::wstring(argv[1])==L"--check-native-profiles"){slstack_native_profile_check();return 0;}
        const std::vector<std::string>keys={"--model","--copy-receipt","--tokens-le32","--cuda-device","--context","--build-sha",
            "--provider-device-bytes","--provider-host-bytes","--cache-bytes","--graph-bytes","--job-device-bytes","--job-host-bytes","--gpu-reserve-bytes",
            "--carry","--carry-file-bytes","--carry-host-bytes","--carry-outstanding-bytes","--output-outstanding-bytes","--payload","--receipt"};
        require(argc>=1+int(keys.size())*2&&argc<=125+int(keys.size())*2&&argc%2==1,"Expected every explicit required --key value pair; use --help");std::map<std::string,std::string>args;
        for(int i=1;i<argc;i+=2){auto key=utf8(argv[i]);require((key=="--unaligned-append-step-tokens"||key=="--reuse-child-raw-metadata"||key=="--session-stdio"||key=="--session-id"||key=="--session-control-host-bytes"||key=="--session-idle-timeout-ms"||key=="--native-rhs-pack"||key=="--split-f32-tensor-core-qk"||key=="--split-f32-tensor-core-pv"||key=="--fuse-kv-gather-half"||key=="--ds-zero-padding-only"||key=="--pv-strided-probabilities"||key=="--pv-view-f32"||key=="--compact-kv-gather"||key=="--masked-half-inplace"||key=="--native-loader-unroll"||key=="--native-expert-columns"||key=="--fixed-resident-plan"||key=="--export-hash-only"||key=="--backend-temporary-reserve-bytes"||key=="--compute-step-tokens"||key=="--prefill-chunk-tokens"||key=="--attention-kv-f16-table"||key=="--index-key-windows"||key=="--capture-dspark"||key=="--compressed-cache-bf16"||key=="--dense-activation-q8-1"||key=="--expert-weights-activations-fp4"||key=="--resident-pageable-copies"||key=="--tile-host-registration"||key=="--hc-bf16-epilogue"||key=="--reuse-expert-routing-map"||key=="--sort-expert-routing"||key=="--pair-ffn-tiles"||key=="--engram-tile-lookahead"||key=="--warmup-one-token"||key=="--batch-seed-ram"||key=="--dense-mxfp4"||key=="--hc-projection-f16-gemm"||key=="--merge-attention-selected-rows"||key=="--resident-metadata"||key=="--resident-compact-ids"||key=="--async-input-uploads"||key=="--resident-hidden-bf16"||key=="--resident-hidden-final-readback"||key=="--coalesce-dense-projections"||key=="--direct-carry-budget"||key=="--expert-f16-cublas"||key=="--token-tile-attention-f16-probabilities"||key=="--index-f32-cublas"||key=="--index-bf16"||key=="--dense-inputs-f16"||key=="--shared-flash-attention"||key=="--shared-sparse-attention"||key=="--carry-ram-prefix-tokens"||key=="--carry-disk-reserve-bytes"||key=="--carry-storage"||key=="--fetch-profile"||key=="--compute-tile"||key=="--hidden-device-bytes"||key=="--pinned-device-bytes"||key=="--reuse-graph"||key=="--cast-device-bytes"||key=="--cast-audit"||key=="--engram-prefetch-bytes"||key=="--ds-submit-threads"||key=="--cuda-blocking-sync"||key=="--index-cublas-device-bytes"||key=="--ds-gpu-ring-slots"||key=="--prefetch-device-bytes"||key=="--expert-profile"||key=="--expert-workspace-bytes"||key=="--expert-runtime-bytes"||std::find(keys.begin(),keys.end(),key)!=keys.end())&&!args.count(key),"Unknown or repeated producer option");args[key]=utf8(argv[i+1]);}
        for(const auto&key:keys)require(args.count(key),"Missing explicit producer option");
        const auto storage=args.count("--carry-storage")?args.at("--carry-storage"):"file";
        require(storage=="file"||storage=="ram"||storage=="hybrid","Expected explicit carry-storage file, ram or hybrid");
        const bool resident_ram=storage=="ram";
        require(!args.count("--direct-carry-budget")||args.at("--direct-carry-budget")=="0"||args.at("--direct-carry-budget")=="1","direct-carry-budget requires0/1");
        const bool direct_budget=args.count("--direct-carry-budget")&&args.at("--direct-carry-budget")=="1";
        require(!direct_budget||(resident_ram&&args.count("--pinned-device-bytes")&&integer(args.at("--pinned-device-bytes"))>=64ull*1024*1024),"Direct carry budget requires explicit pinned resident RAM; no file/hybrid fallback");
        report["direct_carry_budget"]={{"enabled",direct_budget},{"wide_tiles_borrowed_no_vector_allocation",direct_budget},{"vector_read_limit_tokens",direct_budget?32:0}};

        const bool hybrid=storage=="hybrid";
        require(hybrid?(args.count("--carry-ram-prefix-tokens")&&args.count("--carry-disk-reserve-bytes")):(!args.count("--carry-ram-prefix-tokens")&&!args.count("--carry-disk-reserve-bytes")),
            "Hybrid requires explicit prefix and disk reserve, forbidden for other modes");
        const auto ram_prefix=hybrid?integer(args.at("--carry-ram-prefix-tokens")):0;
        const auto disk_reserve=hybrid?integer(args.at("--carry-disk-reserve-bytes")):llama_dsv41_carry_store::reserve_bytes;
        require(ram_prefix<=UINT32_MAX,"Hybrid RAM prefix exceeds token bound");
        const auto tile_tokens=args.count("--compute-tile")?integer(args.at("--compute-tile")):32;
        require(tile_tokens==32||tile_tokens==128||tile_tokens==512||tile_tokens==2048||tile_tokens==4096||tile_tokens==8192||tile_tokens==16384,"Unsupported compute-tile");
        require(resident_ram||hybrid||tile_tokens==32,"Larger compute tiles require explicit RAM or hybrid carry");
        report["compute_tile_tokens"]=tile_tokens;
        const auto step_option=args.count("--compute-step-tokens")?integer(args.at("--compute-step-tokens")):0;
        require(step_option<=tile_tokens,"Logical compute step exceeds admitted tile maximum");
        const auto compute_step=step_option?step_option:tile_tokens;report["compute_step_tokens"]=compute_step;
        const uint64_t unaligned_append_step=args.count("--unaligned-append-step-tokens")?integer(args.at("--unaligned-append-step-tokens")):0;
        require(unaligned_append_step==0||unaligned_append_step==2048,"Unaligned append step must be0 or2048");
        const bool pair_ffn=args.count("--pair-ffn-tiles")&&args.at("--pair-ffn-tiles")=="1";
        require(!args.count("--pair-ffn-tiles")||args.at("--pair-ffn-tiles")=="0"||args.at("--pair-ffn-tiles")=="1","pair-ffn-tiles requires0/1");
        require(tile_tokens<=4096||(pair_ffn&&(tile_tokens==8192||tile_tokens==16384)),"8192 group requires explicit paired attention/FFN graph");
        report["fixed_request_allocations_in_open"]=true;
        require(!args.count("--async-input-uploads")||args.at("--async-input-uploads")=="0"||args.at("--async-input-uploads")=="1","async-input-uploads requires0/1");
        const auto fetch_profile=args.count("--fetch-profile")?args.at("--fetch-profile"):"baseline";
        require(fetch_profile=="baseline"||fetch_profile=="two-buffer-1500-auto"||fetch_profile=="two-buffer-1500-auto-async"||fetch_profile=="two-buffer-250-auto-async"||fetch_profile=="single-buffer-250-auto-async"||fetch_profile=="two-buffer-direct-auto-async","Unsupported explicit fetch-profile");
        report["declared_fetch_profile"]=fetch_profile;
        require(SetConsoleCtrlHandler(interrupt,TRUE),"Cannot install producer cancellation handler");
        auto path=[&](const char*key){return std::filesystem::u8path(args.at(key));};auto number=[&](const char*key){return integer(args.at(key));};
        llama_dsv41_expert_profile::policy expert;
        expert.profile=llama_dsv41_expert_profile::parse(args.count("--expert-profile")?args.at("--expert-profile").c_str():"existing");
        require(!args.count("--shared-sparse-attention")||args.at("--shared-sparse-attention")=="0"||args.at("--shared-sparse-attention")=="1","shared-sparse-attention requires0/1");
        if(args.count("--shared-sparse-attention")&&args.at("--shared-sparse-attention")=="1"){
            require(llama_dsv41_expert_profile::native(expert),"Shared sparse attention requires explicit admitted producer profile");
            expert.profile=llama_dsv41_expert_profile::activation_rounding_q8_1(expert)?llama_dsv41_expert_profile::kind::q8_1_shared_sparse:llama_dsv41_expert_profile::kind::native_half_shared_sparse;
        }
        require(!args.count("--shared-flash-attention")||args.at("--shared-flash-attention")=="0"||args.at("--shared-flash-attention")=="1","shared-flash-attention requires0/1");
        if(args.count("--shared-flash-attention")&&args.at("--shared-flash-attention")=="1"){
            require(llama_dsv41_expert_profile::native(expert)&&!llama_dsv41_expert_profile::shared_sparse(expert),"Shared flash attention and F32 sparse attention are distinct exclusive techniques");
            expert.profile=llama_dsv41_expert_profile::activation_rounding_q8_1(expert)?llama_dsv41_expert_profile::kind::q8_1_fattn:llama_dsv41_expert_profile::kind::native_half_fattn;
        }
        require(!args.count("--dense-inputs-f16")||args.at("--dense-inputs-f16")=="0"||args.at("--dense-inputs-f16")=="1","dense-inputs-f16 requires0/1");
        expert.dense_inputs_f16=args.count("--dense-inputs-f16")&&args.at("--dense-inputs-f16")=="1";
        require(!args.count("--dense-activation-q8-1")||args.at("--dense-activation-q8-1")=="0"||args.at("--dense-activation-q8-1")=="1","dense-activation-q8-1 requires0/1");
        expert.dense_activation_q8_1=args.count("--dense-activation-q8-1")&&args.at("--dense-activation-q8-1")=="1";
        require(!args.count("--index-bf16")||args.at("--index-bf16")=="0"||args.at("--index-bf16")=="1","index-bf16 requires0/1");
        expert.index_bf16=args.count("--index-bf16")&&args.at("--index-bf16")=="1";
        require(!args.count("--index-f32-cublas")||args.at("--index-f32-cublas")=="0"||args.at("--index-f32-cublas")=="1","index-f32-cublas requires 0/1");
        const bool index_f32=args.count("--index-f32-cublas")&&args.at("--index-f32-cublas")=="1";
        require(!index_f32||(!expert.index_bf16&&llama_dsv41_expert_profile::native(expert)&&tile_tokens<=4096&&
            args.count("--index-cublas-device-bytes")&&integer(args.at("--index-cublas-device-bytes"))>=64ull*1024*1024),
            "F32 cuBLAS indexer requires exclusive F32 operands, native expert profile, <=4K tile and explicit 64MiB workspace");
        require(_putenv_s("LLAMA_DSV41_INDEX_F32_CUBLAS",index_f32?"1":"0")==0,"Cannot bind explicit F32 index selection");
        report["indexer_f32_cublas"]={{"enabled",index_f32},{"operands","unconverted F32"},{"accumulation","pedantic F32 GEMM; original ordered F32 head reduction"},{"changed_dot_reduction",index_f32},{"numerically_qualified",false}};

        require(!expert.index_bf16||(args.count("--index-cublas-device-bytes")&&integer(args.at("--index-cublas-device-bytes"))>=64ull*1024*1024),"BF16 indexer requires its explicit existing64MiB device admission");
        require(!args.count("--token-tile-attention-f16-probabilities")||args.at("--token-tile-attention-f16-probabilities")=="0"||args.at("--token-tile-attention-f16-probabilities")=="1","token-tile-attention-f16-probabilities requires0/1");
        expert.token_tile_attention=args.count("--token-tile-attention-f16-probabilities")&&args.at("--token-tile-attention-f16-probabilities")=="1";
        require(!expert.token_tile_attention||(!llama_dsv41_expert_profile::shared_sparse(expert)&&!llama_dsv41_expert_profile::shared_flash_attention(expert)),"Token-tile attention is an explicit exclusive attention selection");
        require(!args.count("--expert-f16-cublas")||args.at("--expert-f16-cublas")=="0"||args.at("--expert-f16-cublas")=="1","expert-f16-cublas requires0/1");
        expert.expert_f16_cublas=args.count("--expert-f16-cublas")&&args.at("--expert-f16-cublas")=="1";
        report["expert_F16_cublas"]={{"enabled",expert.expert_f16_cublas},{"arithmetic","F16 weights and activations; F32 accumulation; changed summation order"},{"numerically_qualified",false}};
        report["speed_toggles"]={{"expert_activation_rounding_q8_1",llama_dsv41_expert_profile::activation_rounding_q8_1(expert)},
            {"shared_sparse_attention_f32",llama_dsv41_expert_profile::shared_sparse(expert)},
            {"shared_flash_attention",llama_dsv41_expert_profile::shared_flash_attention(expert)},
            {"token_tile_attention_f16_probabilities",expert.token_tile_attention},
            {"original_Q8_weights_dense_Q8_1_activation_rounding",expert.dense_activation_q8_1},{"dense_inputs_f16_fp32_accumulation",expert.dense_inputs_f16},{"indexer_bf16_operands_fp32_accumulation",expert.index_bf16},{"indexer_f32_cublas",index_f32}};
        const bool expert_options=args.count("--expert-profile")||args.count("--expert-workspace-bytes")||args.count("--expert-runtime-bytes");
        require(!expert_options||(args.count("--expert-profile")&&args.count("--expert-workspace-bytes")&&args.count("--expert-runtime-bytes")),"Explicit expert profile requires both pool and runtime byte declarations");
        expert.admitted_pool_bytes=expert_options?number("--expert-workspace-bytes"):0;
        expert.admitted_code_runtime_bytes=expert_options?number("--expert-runtime-bytes"):0;
        llama_dsv41_expert_profile::validate(expert,uint32_t(tile_tokens),std::getenv("DSV41_BOUNDARY_CAPTURE"));
        const auto native_columns=args.count("--native-expert-columns")?integer(args.at("--native-expert-columns")):16;
        require(native_columns==DS4PERF_NATIVE_COLUMNS,"Requested native expert columns do not match this explicitly built engine");
        if(native_columns!=16){
            using namespace llama_dsv41_native_columns_resource_proof;
            require(native_columns==columns&&operator_resource_fixture_passed&&tile_tokens<=maximum_tile_rows&&
                llama_dsv41_expert_profile::natural_half(expert)&&expert.admitted_pool_bytes>=minimum_pool_bytes&&
                expert.admitted_code_runtime_bytes>=minimum_runtime_bytes,"Explicit native32 schedule resource/policy admission refused");
        }
        report["native_expert_columns"]=native_columns;
        const auto native_loader=args.count("--native-loader-unroll")?integer(args.at("--native-loader-unroll")):0;
        require(native_loader==DS4PERF_NATIVE_LOADER_UNROLL,"Requested native loader schedule does not match this explicitly built engine");
        if(native_loader){
            using namespace llama_dsv41_native_loader_resource_proof;
            require(native_columns==columns&&native_loader==loader_unroll&&operator_resource_fixture_passed&&tile_tokens<=maximum_tile_rows&&
                llama_dsv41_expert_profile::natural_half(expert)&&expert.admitted_pool_bytes>=minimum_pool_bytes&&
                expert.admitted_code_runtime_bytes>=minimum_runtime_bytes,"Explicit native loader resource/policy admission refused");
        }
        report["native_loader_unroll"]=native_loader;
        const auto explicit_layout_option=[&](const char*key,const char*env){
            require(!args.count(key)||args.at(key)=="0"||args.at(key)=="1","Explicit producer options require0/1");
            const bool enabled=args.count(key)&&args.at(key)=="1";
            require(_putenv_s(env,enabled?"1":"0")==0,"Cannot bind explicit producer option");return enabled;
        };
        const bool rhs_pack=explicit_layout_option("--native-rhs-pack","LLAMA_DSV41_NATIVE_RHS_PACK");
        const bool reuse_child=explicit_layout_option("--reuse-child-raw-metadata","LLAMA_DSV41_REUSE_CHILD_RAW_METADATA");
        require(!reuse_child||(args.count("--pair-ffn-tiles")&&args.at("--pair-ffn-tiles")=="1"),"Child raw metadata reuse requires explicit grouped attention");
        report["reuse_child_raw_metadata"]=reuse_child;
        require(!rhs_pack||(llama_dsv41_expert_profile::natural_half(expert)&&!expert.expert_f16_cublas&&
            native_columns==16&&native_loader==0&&tile_tokens<=4096&&expert.admitted_pool_bytes>=384ull*1024*1024),
            "Native RHS packing requires qualified native half16-column arithmetic, at most4K rows and384MiB pool");
        report["native_rhs_pack"]=rhs_pack;
        report["native_panel_experts"]=rhs_pack?4:0;
        report["native_rhs_panel_columns"]=rhs_pack?32:0;
        const bool fused_gather=explicit_layout_option("--fuse-kv-gather-half","LLAMA_DSV41_GATHER_HALF");
        const bool split_f32_qk=explicit_layout_option("--split-f32-tensor-core-qk","LLAMA_DSV41_SPLIT_F32_QK");
        const bool split_f32_pv=explicit_layout_option("--split-f32-tensor-core-pv","LLAMA_DSV41_SPLIT_F32_PV");
        const bool split_f32=split_f32_qk||split_f32_pv;
        require(!split_f32||(!expert.token_tile_attention&&!llama_dsv41_expert_profile::shared_sparse(expert)&&!llama_dsv41_expert_profile::shared_flash_attention(expert)),"Split-F32 requires the native attention graph");
        const bool pv_view=explicit_layout_option("--pv-view-f32","LLAMA_DSV41_PV_VIEW_F32");
        const bool strided_pv=explicit_layout_option("--pv-strided-probabilities","LLAMA_DSV41_PV_STRIDED_PROBABILITIES");
        require(!strided_pv||(pv_view&&!split_f32_pv),"Strided PV requires native F32 PV view without split PV");
        report["pv_strided_probabilities"]=strided_pv;
        const bool compact_gather=explicit_layout_option("--compact-kv-gather","LLAMA_DSV41_COMPACT_KV_GATHER");
        const bool masked_inplace=explicit_layout_option("--masked-half-inplace","LLAMA_DSV41_MASKED_HALF_INPLACE");
        require(!fused_gather||compact_gather,"Fused masked gather requires explicit compact gather");
        require(!(fused_gather||pv_view||compact_gather||masked_inplace)||(tile_tokens<=4096&&llama_dsv41_expert_profile::natural_half(expert)&&
            !expert.token_tile_attention&&!llama_dsv41_expert_profile::shared_sparse(expert)&&!llama_dsv41_expert_profile::shared_flash_attention(expert)),
            "Explicit F32 attention layout requires native attention and at most4K tiles");
        report["attention_layout"]={{"fused_kv_gather_half",fused_gather},{"pv_view_f32",pv_view},{"compact_kv_gather",compact_gather},{"masked_half_inplace",masked_inplace},
            {"probabilities_and_sinks","unchanged F32"},{"numerically_qualified",false}};
        require(ggml_backend_cuda_dsv41_pv_view_queued()==0,"PV layout process witness must start zero");


        if(llama_dsv41_expert_profile::native(expert)){
        report["schema"]="dsv41-offline-producer-native-half-test-v1";
        report["expert_profile"]={{"name",llama_dsv41_expert_profile::name(expert)},
            {"explicit_pool_reserve_bytes",expert.admitted_pool_bytes},{"measured_code_runtime_reserve_bytes",expert.admitted_code_runtime_bytes},
            {"extra_device_bytes",llama_dsv41_expert_profile::extra_device(expert)},{"resource_fixture_receipt_sha256",llama_dsv41_expert_profile::resource_receipt(expert)},
            {"natural_inputs_no_replay",true},{"native_half_observer_arm_implemented",false},
            {"activation_rounding_q8_1",llama_dsv41_expert_profile::activation_rounding_q8_1(expert)},
            {"performance_resource_envelope_only",llama_dsv41_expert_profile::performance(expert)},
            {"maximum_resource_tile_rows",llama_dsv41_expert_profile::maximum_rows(expert)},
            {"numerical_qualification_passed",false}};
        }
        require(hex_sha(args.at("--build-sha")),"Expected lowercase64hex producer executable SHA256");
        const auto built=read_small(executable(),256ull*1024*1024,false);require(built.sha==args.at("--build-sha"),"Actual producer binary does not match caller buildSHA");
        report["producer_binary_sha256"]=built.sha;report["caller_expected_build_sha256"]=args.at("--build-sha");
        report["build_binding_limit"]="Executable SHA only; sharedDLL contents are not independently bound by this receipt";
        const auto copy=read_small(path("--copy-receipt"),chunk);const auto parsed=json::parse(copy.bytes.begin(),copy.bytes.end());
        const auto&entry=parsed.at("receipts").at(model_name);require(entry.at("sha256").is_string()&&entry.at("sha256").get<std::string>()==model_sha,"No completed original-1017 SHA receipt for main weights");
        const auto&id=entry.at("final_identity");const auto model_bytes=exact_u64(id.at("size")),model_volume=exact_u64(id.at("dev"));
        const auto inode=exact_u64(id.at("ino")),mtime_ns=exact_u64(id.at("mtime_ns"));
        require(mtime_ns/100<=UINT64_MAX-epoch_ticks,"Receipt FILETIME conversion overflow");
        auto tokenfile=read_small(path("--tokens-le32"),4ull*1048576);require(!tokenfile.bytes.empty()&&tokenfile.bytes.size()%4==0,"Expected nonempty raw LE32 token IDs");
        std::vector<int32_t>tokens(tokenfile.bytes.size()/4);for(size_t i=0;i<tokens.size();++i){const auto*p=tokenfile.bytes.data()+4*i;
            const uint32_t value=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);require(value<129280,"LE32 token ID outside original native vocabulary");tokens[i]=int32_t(value);}
        std::vector<uint8_t>().swap(tokenfile.bytes);
        const uint64_t chunk_option=args.count("--prefill-chunk-tokens")?number("--prefill-chunk-tokens"):0;
        require(chunk_option<=1048576,"Chunk tokens exceed context bound");
        const uint32_t chunk_tokens=uint32_t(chunk_option?std::min<uint64_t>(chunk_option,tokens.size()):tokens.size());
        require(!chunk_option||(resident_ram&&!hybrid&&!std::getenv("DSV41_BOUNDARY_CAPTURE")),"Stateful chunks require explicit RAM carry and no boundary capture");
        require(!chunk_option||!pair_ffn||tokens.size()<=chunk_tokens||chunk_tokens%4096==0||unaligned_append_step==2048,"Paired chunk size must align absolute attention positions");
        const auto context=number("--context");require(context<=1048576&&tokens.size()<context,"Native payload requires context room for the next decode token");
        require(!args.count("--session-stdio")||args.at("--session-stdio")=="0"||args.at("--session-stdio")=="1","session-stdio requires0/1");
        const bool live=args.count("--session-stdio")&&args.at("--session-stdio")=="1";
        const uint64_t control_host=live?64ull*1024*1024:0;
        const std::string session=live?args.at("--session-id"):"";
        const uint64_t session_timeout=live?number("--session-idle-timeout-ms"):0;
        const uint32_t reserved_chunk=live?uint32_t(chunk_option):chunk_tokens;
        if(live){
            dsv41_live::sequence_guard identity(session,model_sha);
            require(number("--session-control-host-bytes")==control_host&&session_timeout>=1000&&session_timeout<=3600000,
                "Session requires explicit16MiB control admission and1s..1h timeout");
            require(chunk_option&&chunk_option<=49152&&chunk_option<context&&resident_ram&&direct_budget&&
                (pair_ffn||(args.count("--fixed-resident-plan")&&args.at("--fixed-resident-plan")=="1")),
                "Session requires fixed RAM capacity1..49152 and stateful chunks");
            session_output.bind();
        }else require(!args.count("--session-id")&&!args.count("--session-control-host-bytes")&&!args.count("--session-idle-timeout-ms"),
            "Session options supplied with session disabled");
        report["persistent_session"]={{"enabled",live},{"reserved_chunk_tokens",reserved_chunk},{"control_host_bytes",control_host}};
        require(!args.count("--capture-dspark")||args.at("--capture-dspark")=="0"||args.at("--capture-dspark")=="1","capture-dspark requires0/1");
        const bool capture_dspark=args.count("--capture-dspark")&&args.at("--capture-dspark")=="1";
        require(!live||capture_dspark,"Live session requires DSpark mode3 capture");
        report["dspark_capture"]={{"enabled",capture_dspark},{"ring_bytes",capture_dspark?7864320ull:0ull},{"mode",capture_dspark?3:1},{"qualified",false}};
        llama_dsv41::wire_plan plan;std::string error;success(llama_dsv41::make_wire_plan({uint32_t(context),uint32_t(tokens.size()),capture_dspark?3u:1u,
            capture_dspark?uint32_t(tokens.size()):0u,capture_dspark?uint32_t(std::min<size_t>(tokens.size(),128)):0u},plan,error),error);
        require(!args.count("--export-hash-only")||args.at("--export-hash-only")=="0"||args.at("--export-hash-only")=="1","export-hash-only requires0/1");
        const bool hash_only=args.count("--export-hash-only")&&args.at("--export-hash-only")=="1";
        require(!live||hash_only,"Live initial payload must stream without a disk copy");
        const uint64_t backend_reserve=args.count("--backend-temporary-reserve-bytes")?number("--backend-temporary-reserve-bytes"):gib;
        require(backend_reserve==gib||(backend_reserve==768ull*1024*1024&&pair_ffn),"Backend reserve must be1GiB or explicit768MiB fixed grouped plan");
        const auto output_volume=native_file::volume(path("--payload"));
        require(!hybrid||(path("--carry").root_name()==L"D:"&&llama_dsv41_expert_profile::native(expert)&&disk_reserve==1024ull*1024*1024),
            "Hybrid TEST requires D: overflow, native experts and explicit1GiB disk reserve");
        const bool receipt_overlap=native_file::volume(path("--receipt"))==output_volume;
        const bool overlap=!resident_ram&&native_file::volume(path("--carry"))==output_volume;
        const auto carry_other=number("--carry-outstanding-bytes"),output_other=number("--output-outstanding-bytes");
        require(!resident_ram||(number("--carry-file-bytes")==0&&carry_other==0),
            "RAM carry has no file or pending disk bytes; admit its full bank as host memory");
        require(!overlap||carry_other==output_other,"Same-volume outstanding budgets must describe the same other pending payloads");
        llama_dsv41_carry_store::budget carry_budget;success(resident_ram?
            (direct_budget?llama_dsv41_carry_store::plan_resident_ram_direct(reserved_chunk,carry_budget,error,uint32_t(tile_tokens)):llama_dsv41_carry_store::plan_resident_ram(reserved_chunk,carry_budget,error,uint32_t(tile_tokens))):
            hybrid?llama_dsv41_carry_store::plan_hybrid(uint32_t(tokens.size()),uint32_t(ram_prefix),uint32_t(tile_tokens),0,disk_reserve,carry_budget,error):
            llama_dsv41_carry_store::plan(uint32_t(tokens.size()),0,carry_budget,error),error);
        require(resident_ram||carry_budget.disk_bytes<=number("--carry-file-bytes"),"Carry file exceeds explicit caller byte admission");
        require(carry_budget.bounded_host_bytes<=number("--carry-host-bytes"),"Carry bank/page/tile host memory exceeds explicit admission");
        const auto pending_output=add(output_other,overlap?carry_budget.disk_bytes:0);
        const auto output_floor=hybrid?disk_reserve:ds4perf_output_budget::floor(resident_ram,llama_dsv41_expert_profile::native(expert),
            number("--carry-file-bytes"),carry_other,output_other);
        const auto final_receipt=path("--receipt");require(final_receipt.filename().wstring().find(L':')==std::wstring::npos,"Receipt alternate data streams are forbidden");
        require(GetFileAttributesW(final_receipt.c_str())==INVALID_FILE_ATTRIBUTES&&GetLastError()==ERROR_FILE_NOT_FOUND,"Final receipt must not exist");
        const auto nonce=std::to_wstring(GetCurrentProcessId())+L"-"+std::to_wstring(clock_type::now().time_since_epoch().count());
        const auto pending_receipt=final_receipt.parent_path()/(final_receipt.filename().wstring()+L".pending-"+nonce);
        report["receipt_path"]=args.at("--receipt");report["pending_receipt_path"]=pending_receipt.u8string();
        if(!hash_only){payload.reset(new native_file);payload->open(path("--payload"),true,plan.bytes,add(pending_output,receipt_overlap?receipt_limit:0),false,output_floor);}
        receipt.reset(new native_file);receipt->open(pending_receipt,true,receipt_limit,
            receipt_overlap?add(pending_output,plan.bytes):0,true,output_floor);
        std::cerr<<"Pending receipt (retained on failure): "<<pending_receipt.u8string()<<"\n";
        report["model"]={{"path",args.at("--model")},{"sha256",model_sha},{"verified_bytes",model_bytes},{"receipt_identity",id},{"expected_filetime_ticks",mtime_ns/100+epoch_ticks}};
        report["copy_receipt_sha256"]=copy.sha;report["tokens"]={{"path",args.at("--tokens-le32")},{"sha256",tokenfile.sha},{"count",tokens.size()},{"encoding","LE32 unchanged IDs"}};
        report["payload"]={{"path",hash_only?json(nullptr):json(args.at("--payload"))},{"planned_bytes",plan.bytes},{"hash_only",hash_only}};
        report["carry_path"]=resident_ram?json(nullptr):json(args.at("--carry"));report["carry_storage"]=storage;
        report["carry_resident_bytes"]=carry_budget.resident_bytes;report["mode"]=1;
        report["carry_disk_bytes"]=carry_budget.disk_bytes;report["carry_ram_prefix_tokens"]=ram_prefix;
        const auto job_host=number("--job-host-bytes"),job_device=number("--job-device-bytes"),gpu_reserve=number("--gpu-reserve-bytes");
        require(job_host>cli_host_bytes+control_host&&gpu_reserve>=gib,"Producer requires its64MiB host overhead and explicitGPU reserve>=1GiB");
        require(llama_dsv41_host_memory::begin(job_host,job_device),
            "Producer physicalRAM/WDDMcommit/Job admission refused before CUDA initialization");
        report["telemetry"]["before_cuda_available_ram"]=ram_admission(job_host);
        const auto ordinal=number("--cuda-device");const int devices=ggml_backend_cuda_get_device_count();
        require(devices>0&&ordinal<16&&ordinal<uint64_t(devices),"Explicit CUDA device ordinal is unavailable");
        char description[256]{};ggml_backend_cuda_get_device_description(int(ordinal),description,sizeof(description));std::string gpu=description;
        require(gpu.find("RTX 5090")!=std::string::npos,"Selected explicit CUDA device is not RTX5090; no other GPU is substituted");
        report["gpu"]={{"cuda_ordinal",ordinal},{"description",gpu}};
        require(!args.count("--cuda-blocking-sync")||args.at("--cuda-blocking-sync")=="0"||args.at("--cuda-blocking-sync")=="1","cuda-blocking-sync requires0/1");
        if(args.count("--cuda-blocking-sync")&&args.at("--cuda-blocking-sync")=="1"){
            uint32_t before=0,after=0;require(ggml_dsv41_cuda_blocking_sync(int(ordinal),&before,&after)==0,"Explicit CUDA blocking sync scheduling was not applied");
            report["cuda_schedule_flags"]={{"before",before},{"after",after}};
        }
        size_t free=0,total=0;ggml_backend_cuda_get_device_memory(int(ordinal),&free,&total);
        report["telemetry"]["before_cuda_device_free"]=free;report["telemetry"]["device_total"]=total;
        std::fprintf(stderr,"DS4PERF_GPU_JOB_ADMISSION free=%llu total=%llu job=%llu reserve=%llu\n",(unsigned long long)free,(unsigned long long)total,(unsigned long long)job_device,(unsigned long long)gpu_reserve);
        require(uint64_t(free)>=add(job_device,gpu_reserve),"Producer job exceeds actual selectedGPU free bytes plus explicit reserve");
        backend.reset(ggml_backend_cuda_init(int(ordinal)));require(bool(backend),"Cannot initialize the explicitly selected RTX5090 backend");
        llama_dsv41_weight_provider::config weights;weights.model_path=args.at("--model");weights.verified_file_bytes=model_bytes;
        weights.verified_volume=model_volume;for(unsigned i=0;i<8;++i)weights.verified_file_id[i]=uint8_t(inode>>(i*8));
        weights.verified_mtime_ns=mtime_ns;weights.verified_sha256=model_sha;weights.backend=backend.get();
        weights.admitted_device_bytes=number("--provider-device-bytes");weights.admitted_host_bytes=number("--provider-host-bytes");weights.buffers=(fetch_profile=="baseline"||fetch_profile=="single-buffer-250-auto-async")?1u:2u;
        if(fetch_profile!="baseline"){
            weights.staging_limit_bytes=(fetch_profile=="two-buffer-250-auto-async"||fetch_profile=="single-buffer-250-auto-async"||fetch_profile=="two-buffer-direct-auto-async")?250ull*1024*1024:1500ull*1024*1024;
            weights.auto_request_size=true;
        }
        weights.async_staging_drain=fetch_profile=="two-buffer-1500-auto-async"||fetch_profile=="two-buffer-250-auto-async"||fetch_profile=="single-buffer-250-auto-async"||fetch_profile=="two-buffer-direct-auto-async";
        weights.compute_direct=fetch_profile=="two-buffer-direct-auto-async";
        report["weight_fetch_direct_compute"]=weights.compute_direct;
        report["weight_fetch_buffers"]=weights.buffers;report["weight_fetch_staging_bytes"]=weights.staging_limit_bytes;

        const auto ring_slots=args.count("--ds-gpu-ring-slots")?integer(args.at("--ds-gpu-ring-slots")):0;
        require(ring_slots==0||(ring_slots==2&&weights.buffers==2),"GPU ring requires0 or2 and explicit two-buffer TEST");
        weights.ds_gpu_ring_slots=uint32_t(ring_slots);if(ring_slots)weights.async_staging_drain=true;
        report["ds_gpu_ring_slots"]=ring_slots;
        weights.engram_prefetch_host_bytes=args.count("--engram-prefetch-bytes")?number("--engram-prefetch-bytes"):0;
        weights.engram_prefetch_tokens=weights.engram_prefetch_host_bytes?chunk_tokens:0;
        const uint64_t submit_threads=args.count("--ds-submit-threads")?number("--ds-submit-threads"):0;
        require(submit_threads==0||submit_threads==1||submit_threads==2||submit_threads==4,"ds-submit-threads requires0/1/2/4");
        require(!args.count("--ds-zero-padding-only")||args.at("--ds-zero-padding-only")=="0"||args.at("--ds-zero-padding-only")=="1","Padding-only zero requires0/1");
        weights.zero_padding_only=args.count("--ds-zero-padding-only")&&args.at("--ds-zero-padding-only")=="1";
        report["ds_zero_padding_only"]=weights.zero_padding_only;
        weights.ds_submit_threads=uint32_t(submit_threads);report["ds_submit_threads"]=submit_threads;
        report["engram_prefetch_host_bytes"]=weights.engram_prefetch_host_bytes;
        require(!args.count("--compressed-cache-bf16")||args.at("--compressed-cache-bf16")=="0"||args.at("--compressed-cache-bf16")=="1","compressed-cache-bf16 requires0/1");
        llama_dsv41_sweep::config options;options.compressed_cache_bf16=args.count("--compressed-cache-bf16")&&args.at("--compressed-cache-bf16")=="1";options.capture_dspark=capture_dspark;
        report["compressed_cache_bf16"]={{"enabled",options.compressed_cache_bf16},{"lossless_existing_QAT_BF16_values",true},{"canonical_wire_F32",true},{"raw_and_pool_tails_F32",true}};
        options.context=uint32_t(context);options.admitted_cache_bytes=number("--cache-bytes");options.admitted_graph_bytes=number("--graph-bytes");
        options.admitted_hidden_bytes=args.count("--hidden-device-bytes")?number("--hidden-device-bytes"):0;
        require(!args.count("--reuse-graph")||args.at("--reuse-graph")=="0"||args.at("--reuse-graph")=="1"||args.at("--reuse-graph")=="2","reuse-graph requires explicit0/1/2");
        options.reuse_graph=args.count("--reuse-graph")&&args.at("--reuse-graph")!="0";
        options.reuse_layer_graph=args.count("--reuse-graph")&&args.at("--reuse-graph")=="2";
        report["reuse_layer_graph_layouts"]=options.reuse_layer_graph;
        report["reuse_graph_allocation"]=options.reuse_graph;
        options.admitted_cast_device_bytes=args.count("--cast-device-bytes")?number("--cast-device-bytes"):0;
        require(!args.count("--cast-audit")||args.at("--cast-audit")=="0"||args.at("--cast-audit")=="1","cast-audit requires0 or1");
        options.audit_casts=args.count("--cast-audit")&&args.at("--cast-audit")=="1";
        require(!args.count("--coalesce-dense-projections")||args.at("--coalesce-dense-projections")=="0"||args.at("--coalesce-dense-projections")=="1","coalesce-dense-projections requires0/1");
        options.coalesce_dense=args.count("--coalesce-dense-projections")&&args.at("--coalesce-dense-projections")=="1";
        report["coalesce_dense_projections"]=options.coalesce_dense;
        require(!args.count("--merge-attention-selected-rows")||args.at("--merge-attention-selected-rows")=="0"||args.at("--merge-attention-selected-rows")=="1","merge-attention-selected-rows requires0/1");
        options.merge_attention_selected_rows=args.count("--merge-attention-selected-rows")&&args.at("--merge-attention-selected-rows")=="1";
        report["merge_attention_selected_rows"]=options.merge_attention_selected_rows;
        require(!args.count("--hc-projection-f16-gemm")||args.at("--hc-projection-f16-gemm")=="0"||args.at("--hc-projection-f16-gemm")=="1","hc-projection-f16-gemm requires0/1");
        options.hc_projection_f16_gemm=args.count("--hc-projection-f16-gemm")&&args.at("--hc-projection-f16-gemm")=="1";
        require(!args.count("--dense-mxfp4")||args.at("--dense-mxfp4")=="0"||args.at("--dense-mxfp4")=="1","dense-mxfp4 requires0/1");
        options.dense_mxfp4=args.count("--dense-mxfp4")&&args.at("--dense-mxfp4")=="1";
        require(!options.dense_mxfp4||expert.dense_inputs_f16,"Dense FP4 requires the explicit dense projection selection");
        report["dense_mxfp4"]=options.dense_mxfp4;
        require(!args.count("--batch-seed-ram")||args.at("--batch-seed-ram")=="0"||args.at("--batch-seed-ram")=="1","batch-seed-ram requires0/1");
        options.batch_seed_ram=args.count("--batch-seed-ram")&&args.at("--batch-seed-ram")=="1";
        report["batch_seed_ram"]=options.batch_seed_ram;
        options.unaligned_append_step_tokens=uint32_t(unaligned_append_step);
        report["unaligned_append_step_tokens"]=options.unaligned_append_step_tokens;
        options.compute_step_tokens=uint32_t(step_option);options.pair_ffn=pair_ffn;report["pair_ffn_tiles"]=pair_ffn;report["maximum_actual_attention_rows"]=pair_ffn?(llama_dsv41_expert_profile::natural_half(expert)?2048:4096):tile_tokens;
        report["activation_streaming_mode"]=pair_ffn&&(!args.count("--hidden-device-bytes")||number("--hidden-device-bytes")==0)?"existing pinned RAM layerwise tiles; separate copy stream and exact lookahead lease":"explicit existing device bank";
        report["hc_projection_f16_gemm"]=options.hc_projection_f16_gemm;
        options.admitted_index_device_bytes=args.count("--index-cublas-device-bytes")?number("--index-cublas-device-bytes"):0;
        report["index_cublas_experimental_device_bytes"]=options.admitted_index_device_bytes;
        options.admitted_pinned_device_bytes=args.count("--pinned-device-bytes")?number("--pinned-device-bytes"):0;
        options.admitted_prefetch_device_bytes=args.count("--prefetch-device-bytes")?number("--prefetch-device-bytes"):0;
        report["copy_prefetch_device_bytes"]=options.admitted_prefetch_device_bytes;
        options.admitted_device_bytes=job_device;options.admitted_host_bytes=job_host-cli_host_bytes-control_host;options.carry.path=args.at("--carry");options.carry.tokens=chunk_tokens;options.stateful_chunks=chunk_option!=0;
        options.carry.resident_ram=resident_ram;
        require(!args.count("--resident-hidden-final-readback")||args.at("--resident-hidden-final-readback")=="0"||args.at("--resident-hidden-final-readback")=="1","resident-hidden-final-readback requires0/1");
        options.carry.defer_hidden_readback=args.count("--resident-hidden-final-readback")&&args.at("--resident-hidden-final-readback")=="1";
        report["resident_hidden_final_readback"]=options.carry.defer_hidden_readback;
        require(!args.count("--resident-compact-ids")||args.at("--resident-compact-ids")=="0"||args.at("--resident-compact-ids")=="1","resident-compact-ids requires0/1");
        require(!args.count("--resident-metadata")||args.at("--resident-metadata")=="0"||args.at("--resident-metadata")=="1","resident-metadata requires0/1");
        options.resident_metadata=args.count("--resident-metadata")&&args.at("--resident-metadata")=="1";
        report["resident_metadata"]=options.resident_metadata;
        options.resident_ids=args.count("--resident-compact-ids")&&args.at("--resident-compact-ids")=="1";
        report["resident_compact_ids"]=options.resident_ids;
        options.async_inputs=args.count("--async-input-uploads")&&args.at("--async-input-uploads")=="1";
        report["async_input_uploads"]=options.async_inputs;
        options.carry.export_hidden_after_head=options.carry.defer_hidden_readback;
        report["hidden_export_after_head"]=options.carry.export_hidden_after_head;
        require(!args.count("--resident-hidden-bf16")||args.at("--resident-hidden-bf16")=="0"||args.at("--resident-hidden-bf16")=="1","resident-hidden-bf16 requires0/1");
        options.resident_bf16=args.count("--resident-hidden-bf16")&&args.at("--resident-hidden-bf16")=="1";
        require(!options.resident_bf16||(options.carry.defer_hidden_readback&&std::getenv("GGML_CUDA_DISABLE_GRAPHS")&&std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))=="1"),"Resident BF16 test requires explicit full resident readback and disabled CUDA capture");
        report["resident_hidden_storage"]=options.resident_bf16?"BF16 after validated existing BF16 boundary; layer0 input unchanged F32":"F32";
        options.carry.ram_prefix_tokens=uint32_t(ram_prefix);options.carry.disk_reserve_bytes=disk_reserve;
        if(const char *diag=std::getenv("DS4PERF_DIAGNOSTIC_LAYERS")){if(*diag){
            const auto layers=integer(diag);require(layers>0&&layers<40&&resident_ram&&llama_dsv41_expert_profile::native(expert),
                "Incomplete diagnostics require native RAM TEST and1..39 layers");
            options.diagnostic_layers=uint32_t(layers);report["intentional_incomplete_diagnostic_layers"]=layers;
            options.diagnostic_carry=[](uint32_t layer,uint32_t first,uint32_t count,const void *bank,uint64_t bytes,const ggml_tensor *raw){
                hash digest;const uint32_t tail=std::min(count,128u);
                for(uint32_t row=first+count-tail;row<first+count;++row){
                    const uint64_t at=4096+uint64_t(row/32)*llama_dsv41_carry_store::ram_page_bytes+uint64_t(row%32)*llama_dsv41_carry_store::ram_record_bytes;
                    require(at<=bytes&&81936<=bytes-at,"Diagnostic RAM tile span changed");digest.update(static_cast<const uint8_t*>(bank)+at,81936);
                }
                const auto sha=digest.finish();std::fprintf(stderr,"DS4PERF_CARRY_TRACE layer=%u start=%u count=%u tail_rows=%u sha256=%s",layer,first,count,tail,sha.c_str());
                std::fputc(10,stderr);std::fflush(stderr);
                require(raw&&raw->type==GGML_TYPE_F32&&ggml_nbytes(raw)<=262144,"Diagnostic raw-ring extent changed");
                std::vector<uint8_t> raw_bytes(ggml_nbytes(raw));ggml_backend_tensor_get(raw,raw_bytes.data(),0,raw_bytes.size());
                hash raw_digest;raw_digest.update(raw_bytes.data(),raw_bytes.size());const auto raw_sha=raw_digest.finish();
                std::fprintf(stderr,"DS4PERF_RAW_TRACE layer=%u start=%u count=%u sha256=%s",layer,first,count,raw_sha.c_str());
                std::fputc(10,stderr);std::fflush(stderr);
            };
        }}
        require(!args.count("--engram-tile-lookahead")||args.at("--engram-tile-lookahead")=="0"||args.at("--engram-tile-lookahead")=="1","engram-tile-lookahead requires0/1");
        options.engram_tile_lookahead=args.count("--engram-tile-lookahead")&&args.at("--engram-tile-lookahead")=="1";
        report["engram_tile_lookahead"]=options.engram_tile_lookahead;
        options.carry.max_tile_tokens=uint32_t(tile_tokens);options.carry.direct_ram_tiles=direct_budget;
        require(!args.count("--sort-expert-routing")||args.at("--sort-expert-routing")=="0"||args.at("--sort-expert-routing")=="1","sort-expert-routing requires0/1");
        expert.sorted_expert_routing=args.count("--sort-expert-routing")&&args.at("--sort-expert-routing")=="1";
        require(!args.count("--reuse-expert-routing-map")||args.at("--reuse-expert-routing-map")=="0"||args.at("--reuse-expert-routing-map")=="1","reuse-expert-routing-map requires0/1");
        expert.reuse_expert_routing_map=args.count("--reuse-expert-routing-map")&&args.at("--reuse-expert-routing-map")=="1";
        require(!args.count("--hc-bf16-epilogue")||args.at("--hc-bf16-epilogue")=="0"||args.at("--hc-bf16-epilogue")=="1","hc-bf16-epilogue requires0/1");
        expert.hc_bf16_epilogue=args.count("--hc-bf16-epilogue")&&args.at("--hc-bf16-epilogue")=="1";
        report["hc_bf16_epilogue"]=expert.hc_bf16_epilogue;
        require(!args.count("--expert-weights-activations-fp4")||args.at("--expert-weights-activations-fp4")=="0"||args.at("--expert-weights-activations-fp4")=="1","expert-weights-activations-fp4 requires0/1");
        expert.expert_mxfp4=args.count("--expert-weights-activations-fp4")&&args.at("--expert-weights-activations-fp4")=="1";
        require(!expert.expert_mxfp4||(options.dense_mxfp4&&options.reuse_graph&&!options.reuse_layer_graph&&!pair_ffn&&tile_tokens<=4096),"Expert FP4 requires existing dense FP4 scratch and allocator-only<=4K lane");
        report["expert_weights_activations_fp4"]={{"enabled",expert.expert_mxfp4},{"supersedes_Q8_1_for_routed_products_above4rows",expert.expert_mxfp4},{"canonical_model_file_unchanged",true},{"numerically_qualified",false}};
        require(!args.count("--index-key-windows")||args.at("--index-key-windows")=="0"||args.at("--index-key-windows")=="1","index-key-windows requires0/1");
        expert.index_key_windows=args.count("--index-key-windows")&&args.at("--index-key-windows")=="1";
        report["index_key_windows"]={{"enabled",expert.index_key_windows},{"key_slab_rows",8192},{"starts_above_keys",131072},{"single_query_full_score_ceiling_bytes",4194304},{"existing_score_and_topK_kernels",true},{"tagged_selector_NaNs_refused",true}};
        require(!args.count("--attention-kv-f16-table")||args.at("--attention-kv-f16-table")=="0"||args.at("--attention-kv-f16-table")=="1","attention-kv-f16-table requires0/1");
        expert.attention_kv_f16_table=args.count("--attention-kv-f16-table")&&args.at("--attention-kv-f16-table")=="1";
        require(!expert.attention_kv_f16_table||expert.token_tile_attention,"Direct F16 attention table requires token-tile attention");
        report["attention_kv_f16_table"]={{"enabled",expert.attention_kv_f16_table},{"canonical_cache_format_unchanged",true},{"existing_cast_and_HMMA_body",true},{"maximum_rows_including_dummy",1048577},{"long_context_requires_explicit_unmerged_records",true}};
        options.expert=expert;report["sort_expert_routing"]=expert.sorted_expert_routing;report["reuse_expert_routing_map"]=expert.reuse_expert_routing_map;
        options.carry.admitted_file_bytes=number("--carry-file-bytes");options.carry.admitted_host_bytes=number("--carry-host-bytes");
        options.carry.outstanding_bytes=add(carry_other,overlap?add(plan.bytes,receipt_limit):0);
        report["budgets"]={{"provider_device",weights.admitted_device_bytes},{"provider_host",weights.admitted_host_bytes},{"cache",options.admitted_cache_bytes},
            {"graph",options.admitted_graph_bytes},{"hidden_device",options.admitted_hidden_bytes},{"pinned_device",options.admitted_pinned_device_bytes},{"weight_cast_device",options.admitted_cast_device_bytes},{"audit_casts",options.audit_casts},{"job_device",job_device},{"job_host",job_host},{"cli_host_overhead",cli_host_bytes},{"gpu_reserve",gpu_reserve},{"carry_file",options.carry.admitted_file_bytes},
            {"carry_host",options.carry.admitted_host_bytes},{"carry_other_pending",carry_other},{"output_other_pending",output_other},{"actual_volume_overlap",overlap},{"receipt_output_volume_overlap",receipt_overlap},{"disk_reserve",output_floor}};
        if(llama_dsv41_expert_profile::native(expert)){
            report["budgets"]["expert_pool"]=expert.admitted_pool_bytes;
            report["budgets"]["expert_runtime"]=expert.admitted_code_runtime_bytes;
        }
        stop();report["telemetry"]["before_open_available_ram"]=ram_admission(job_host);
        ggml_backend_dev_memory(ggml_backend_get_device(backend.get()),&free,&total);report["telemetry"]["before_open_device_free"]=free;
        require(uint64_t(free)>=add(job_device,gpu_reserve),"CUDA context changed available device admission");sweep.reset(new llama_dsv41_sweep);
        report["phase_ms"]["input_identity_hash_budget_preflight"]=ms(process_started);
        if(llama_dsv41_expert_profile::natural_half(expert)){
            ggml_ds4_native_half_stats initial{};ggml_backend_cuda_ds4_native_half_snapshot(&initial);
            require(initial.completed_graphs==0&&initial.graph_active==0&&initial.queued[0]==0&&initial.queued[1]==0&&initial.queued[2]==0&&initial.completed[0]==0&&initial.completed[1]==0&&initial.completed[2]==0,"Native-half process witness must start zero");
        }
        if(expert.expert_f16_cublas)require(ggml_backend_cuda_dsv41_expert_cublas_queued()==0,"Expert cuBLAS process witness must start zero");
        require(!args.count("--warmup-one-token")||args.at("--warmup-one-token")=="0"||args.at("--warmup-one-token")=="1","warmup-one-token requires0/1");
        const bool warm=args.count("--warmup-one-token")&&args.at("--warmup-one-token")=="1";
        require(!args.count("--tile-host-registration")||args.at("--tile-host-registration")=="0"||args.at("--tile-host-registration")=="1","tile-host-registration requires0/1");
        const bool tile_registration=args.count("--tile-host-registration")&&args.at("--tile-host-registration")=="1";
        require(!args.count("--fixed-resident-plan")||args.at("--fixed-resident-plan")=="0"||args.at("--fixed-resident-plan")=="1","fixed-resident-plan requires0/1");
        const bool fixed_resident=args.count("--fixed-resident-plan")&&args.at("--fixed-resident-plan")=="1";
        require(!fixed_resident||(!pair_ffn&&tile_tokens<=4096&&warm&&tile_registration&&resident_ram&&direct_budget&&
            options.reuse_graph&&!options.reuse_layer_graph&&options.resident_bf16&&options.resident_ids&&options.resident_metadata&&
            options.carry.defer_hidden_readback&&!options.admitted_prefetch_device_bytes&&backend_reserve==gib),
            "Unpaired fixed plan requires explicit warm full-resident RAM tiles and unchanged1GiB backend reserve");
        report["unpaired_fixed_resident_plan"]=fixed_resident;
        require(!tile_registration||((pair_ffn||fixed_resident)&&((options.resident_bf16&&options.resident_ids&&options.resident_metadata)||
            (!options.admitted_hidden_bytes&&!options.resident_bf16&&!options.resident_ids&&!options.resident_metadata&&options.admitted_prefetch_device_bytes&&!options.carry.defer_hidden_readback))),"Tile host registration requires its explicit fixed resident or paired RAM-lookahead plan");
        require(!args.count("--resident-pageable-copies")||args.at("--resident-pageable-copies")=="0"||args.at("--resident-pageable-copies")=="1","resident-pageable-copies requires0/1");
        const bool pageable_copies=args.count("--resident-pageable-copies")&&args.at("--resident-pageable-copies")=="1";
        require(!pageable_copies||(!tile_registration&&pair_ffn&&options.resident_bf16&&options.resident_ids&&options.resident_metadata),"Pageable vector copies require exclusive full-resident pair plan");
        report["tile_host_registration"]=tile_registration;report["resident_pageable_copies"]=pageable_copies;
        if(pair_ffn||fixed_resident){
            require(warm,"Fixed reservation requires the actual persistent warmup");
            options.fixed.reset(new llama_dsv41_fixed_storage);
            options.fixed->unpaired_resident=fixed_resident;
            options.fixed->backend_temporary_reserve_bytes=backend_reserve;report["backend_temporary_reserve_bytes"]=backend_reserve;
            options.fixed->tokens=reserved_chunk;options.fixed->tile=options.carry.max_tile_tokens;options.fixed->context=options.context;
            options.fixed->bf16=options.resident_bf16;options.fixed->ids=options.resident_ids;options.fixed->metadata=options.resident_metadata;options.fixed->tile_registration=tile_registration;options.fixed->pageable_copies=pageable_copies;
            report["large_GPU_buffers_reserved_before_first_stage_lease"]=tile_tokens!=16384;
            report["graph_capacity_strategy"]=tile_tokens==16384?"existing gallocr grows to actual planned live bytes inside prompt clock":"explicit maximum reserved before first lease";
        }
        std::unique_ptr<llama_dsv41_sweep> warm_request;
        ggml_ds4_native_half_stats warm_native{};
        uint64_t warm_cublas=0;
        if(warm){
            require(resident_ram&&!options.diagnostic_layers&&!std::getenv("DSV41_BOUNDARY_CAPTURE"),"One-token warmup requires RAM carry and no boundary/incomplete diagnostics");
            auto warm_options=options;auto warm_weights=weights;
            warm_options.carry.tokens=1;warm_options.pair_ffn=false;
            warm_options.carry.max_tile_tokens=std::min(4096u,warm_options.carry.max_tile_tokens);
            warm_options.compute_step_tokens=std::min(warm_options.compute_step_tokens,warm_options.carry.max_tile_tokens);
            // Dense tensor-core tagging requires more than eight columns.
            // The real one-token request uses the existing narrow path.
            warm_options.expert.dense_inputs_f16=false;warm_options.expert.dense_activation_q8_1=false;warm_options.expert.sorted_expert_routing=false;warm_options.expert.reuse_expert_routing_map=false;
            warm_options.coalesce_dense=false;warm_options.dense_mxfp4=false;warm_options.expert.expert_mxfp4=false;
            if(expert.expert_mxfp4)std::fprintf(stderr,"DS4PERF_EXPERT_MXFP4_WARMUP_ORIGINAL tokens=1 large_batch_only_weight_pack=1 explicit=1\n");
            if(warm_weights.engram_prefetch_host_bytes)warm_weights.engram_prefetch_tokens=1;
            std::fprintf(stderr,"DS4PERF_WARMUP_BEGIN pid=%lu tokens=1 export=0\n",GetCurrentProcessId());std::fflush(stderr);
            const auto warm_started=clock_type::now();
            warm_request.reset(new llama_dsv41_sweep);
            success(warm_request->open(warm_weights,warm_options,error),error);
            success(warm_request->run(std::vector<int32_t>{tokens.front()},[]{return should_stop();},error),error);
            require(warm_request->ready(),"One-token warmup did not complete its actual40-layer prefill/head");
            ggml_backend_synchronize(backend.get());stop();
            report["phase_ms"]["one_token_warmup_and_retirement"]=ms(warm_started);
            report["warm_measurement"]={{"enabled",true},{"warmup_tokens",1},{"producer_pid",GetCurrentProcessId()},
                {"same_process_and_CUDA_backend",true},{"request_allocations_recreated",!bool(options.fixed)},{"request_tensor_contexts_recreated",true},{"physical_RAM_bank_reused",bool(options.fixed)},{"large_GPU_buffers_reused",bool(options.fixed)},{"provider_model_lease_heaps_requests_reused",true},{"warmup_exported",false},
                {"one_token_dense_path","Existing narrow path; F16 dense tags/coalescing/FP4 pack require large batch and are disabled for this one-token request"},
                {"scope","Prompt run only; model/provider/cache/resident allocation in open excluded; prompt graph work, Engram and required head included; bulk hidden export preparation, serialization and Mac transfer separate"}};
            ggml_backend_cuda_ds4_native_half_snapshot(&warm_native);warm_cublas=ggml_backend_cuda_dsv41_expert_cublas_queued();
        }
        const uint64_t warm_pv_view=ggml_backend_cuda_dsv41_pv_view_queued();
        const uint64_t warm_gather_half=ggml_backend_cuda_dsv41_gather_half_queued();
        const uint64_t warm_gather_split=ggml_backend_cuda_dsv41_gather_half_split_queued();
        const uint64_t warm_split_f32=ggml_backend_cuda_dsv41_split_f32_queued();
        const uint64_t warm_rhs_pack=ggml_backend_cuda_dsv41_rhs_pack_queued();
        auto started=clock_type::now();const bool opened=sweep->open(weights,options,error,warm_request.get());report["phase_ms"]["inspect_admit_allocate"]=ms(started);success(opened,error);warm_request.reset();stop();
        require(!chunk_option||options.reuse_graph&&!options.reuse_layer_graph,"Stateful CLI requires allocator-only graph mode");
        report["stateful_chunks"]={{"enabled",chunk_option!=0},{"maximum_chunk_tokens",chunk_tokens},{"total_context_tokens",tokens.size()},{"chunks",json::array()},{"canonical_cache_preserved",chunk_option!=0},{"token_history_host_charge_bytes",chunk_option?context*8:tokens.size()*8}};
        if(warm){std::fprintf(stderr,"DS4PERF_WARM_PROMPT_BEGIN pid=%lu tokens=%zu\n",GetCurrentProcessId(),tokens.size());std::fflush(stderr);}
        started=clock_type::now();double total_prompt_ms=0,total_export_ms=0,append_open_ms=0,total_token_prepare_ms=0;
        for(uint32_t first=0;first<tokens.size();first+=chunk_tokens){
            const uint32_t count=std::min(chunk_tokens,uint32_t(tokens.size())-first);
            if(first){
                auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);
                auto next_options=options;auto next_weights=weights;next_options.carry.tokens=count;
                if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=count;
                const auto append_started=clock_type::now();success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);
                require(!previous->ready(),"Consumed prefix still exports");
                unsigned retired_sends=0;std::string retired_error;require(!previous->export_native([&](llama_dsv41::wire_payload_reader&,std::string&){++retired_sends;return true;},retired_error)&&retired_sends==0,"Consumed prefix reached transport");
                previous.reset();append_open_ms+=ms(append_started);
            }
            const auto chunk_prepare_started=clock_type::now();
            const std::vector<int32_t> current(tokens.begin()+first,tokens.begin()+first+count);
            const double token_prepare_ms=ms(chunk_prepare_started);total_token_prepare_ms+=token_prepare_ms;total_prompt_ms+=token_prepare_ms;
            success(sweep->run(current,[]{return should_stop();},error),error);stop();
            const auto part=sweep->timing();require(part.prompt_processing_ms>0&&part.export_preparation_ms>=0,"Completed chunk omitted actual prompt/export timing boundaries");
            total_prompt_ms+=part.prompt_processing_ms;total_export_ms+=part.export_preparation_ms;
            report["stateful_chunks"]["chunks"].push_back({{"first",first},{"tokens",count},{"completed_context_tokens",first+count},{"prompt_processing_ms",part.prompt_processing_ms},{"export_preparation_ms",part.export_preparation_ms}});
            std::fprintf(stderr,"DS4PERF_STATEFUL_CHUNK_COMPLETE first=%u count=%u completed_context=%u processing_ms=%.6f canonical_cache_kept=%u\n",first,count,first+count,part.prompt_processing_ms,unsigned(first!=0));
        }
        const auto enclosing_run_ms=ms(started);report["phase_ms"]["append_owner_admission"]=append_open_ms;report["phase_ms"]["chunk_token_preparation"]=total_token_prepare_ms;
        // Append ownership/admission is required prompt work; include it.
        const llama_dsv41_sweep::run_timing run_timing{total_prompt_ms+append_open_ms,total_export_ms};
        report["phase_ms"]["embedding_40_layers_head"]=run_timing.prompt_processing_ms;
        report["phase_ms"]["final_hidden_export_preparation"]=run_timing.export_preparation_ms;
        report["phase_ms"]["run_dispatch_bookkeeping"]=std::max(0.0,enclosing_run_ms-run_timing.prompt_processing_ms-run_timing.export_preparation_ms);
        report["run_timing_scope"]="Actual sequential clock boundary after required head/sync; bulk hidden export readback and certification measured separately; no subtraction of aggregate GPU counters";
        if(warm){report["warm_measurement"]["prompt_processing_ms"]=report["phase_ms"]["embedding_40_layers_head"];std::fprintf(stderr,"DS4PERF_WARM_PROMPT_END pid=%lu tokens=%zu processing_ms=%.6f export_excluded=1\n",GetCurrentProcessId(),tokens.size(),report["phase_ms"]["embedding_40_layers_head"].get<double>());}
        if(llama_dsv41_expert_profile::natural_half(expert)){
            ggml_ds4_native_half_stats actual{};ggml_backend_cuda_ds4_native_half_snapshot(&actual);
            const uint64_t expected=40ull*([&]{uint64_t n=0;for(uint64_t first=0;first<tokens.size();first+=chunk_tokens)n+=(std::min<uint64_t>(chunk_tokens,tokens.size()-first)+compute_step-1)/compute_step;return n;}());
            require(actual.graph_active==0&&actual.completed_graphs-warm_native.completed_graphs==expected,"Native-half completed graph count differs from actual prefix/tiles");
            for(int i=0;i<3;++i)require(actual.queued[i]-warm_native.queued[i]==expected&&actual.completed[i]-warm_native.completed[i]==expected,"Native-half aggregate actual slot witness incomplete");
            report["expert_profile"]["actual_queued_per_slot"]={actual.queued[0]-warm_native.queued[0],actual.queued[1]-warm_native.queued[1],actual.queued[2]-warm_native.queued[2]};
            report["expert_profile"]["actual_completed_per_slot"]={actual.completed[0]-warm_native.completed[0],actual.completed[1]-warm_native.completed[1],actual.completed[2]-warm_native.completed[2]};
            report["expert_profile"]["actual_completed_graphs"]=actual.completed_graphs-warm_native.completed_graphs;
            report["expert_profile"]["expected_graphs"]=expected;
            report["expert_profile"]["actual_total_queued"]=expected*3;
            report["expert_profile"]["actual_total_completed"]=expected*3;
            report["expert_profile"]["dispatch_family"]="q4k_natural_half_rhs_mmid_v1";
            report["expert_profile"]["mathematical_source_sha256"]=llama_dsv41_native_half_resource_proof::mathematical_source_sha256;
            report["expert_profile"]["mathematical_object_sha256"]="09a90d07ce46717a25a3502bf8fbf68f6e587f85c134438adc86a49be344d068";
            report["expert_profile"]["native_half_matrix_proof_sha256"]="24231bd1932c637fd9344e7f780d9cf1352303d55a7af5a7c01f5933582a6025";
            const uint64_t rhs_actual=ggml_backend_cuda_dsv41_rhs_pack_queued()-warm_rhs_pack;
            require(rhs_actual==(rhs_pack?expected*3:0),"Native RHS packing completed dispatch count differs from explicit option");
            report["pv_strided_calls_after_completed_sweep"]=ggml_backend_cuda_dsv41_pv_strided_queued();
            report["expert_profile"]["prepacked_rhs_half"]=rhs_pack;
            report["expert_profile"]["prompt_rhs_pack_calls_after_completed_sweep"]=rhs_actual;
            report["expert_profile"]["warmup_rhs_pack_calls"]=warm_rhs_pack;
            if(rhs_pack){
                using namespace llama_dsv41_native_rhs_pack_proof;
                report["expert_profile"]["dispatch_family"]="q4k_natural_half_rhs_mmid_rhs_prepacked_half_panel4";
                report["expert_profile"]["panel_experts"]=4;
                report["expert_profile"]["mathematical_source_sha256"]=panel_source_sha256;
                report["expert_profile"]["mathematical_object_sha256"]=panel_object_sha256;
                report["expert_profile"]["dispatch_source_sha256"]=dispatch_source_sha256;
                report["expert_profile"]["dispatch_object_sha256"]=dispatch_object_sha256;
                report["expert_profile"]["baseline_native_half_matrix_proof_sha256"]=report["expert_profile"]["native_half_matrix_proof_sha256"];
                report["expert_profile"]["native_half_matrix_proof_sha256"]=nullptr;
                report["expert_profile"]["resource_fixture_receipt_sha256"]=fixture_sha256;
                report["expert_profile"]["operator_phase_sha256"]=phase_sha256;
                report["expert_profile"]["column_schedule"]=32;
                report["expert_profile"]["full_model_qualified"]=false;
            }
            if(native_loader){
                using namespace llama_dsv41_native_loader_resource_proof;
                report["expert_profile"]["dispatch_family"]="q4k_natural_half_rhs_mmid_loader_unroll1";
                report["expert_profile"]["mathematical_source_sha256"]=source_sha256;
                report["expert_profile"]["mathematical_object_sha256"]=object_sha256;
                report["expert_profile"]["baseline_native_half_matrix_proof_sha256"]=report["expert_profile"]["native_half_matrix_proof_sha256"];
                report["expert_profile"]["native_half_matrix_proof_sha256"]=nullptr;
                report["expert_profile"]["resource_fixture_receipt_sha256"]=fixture_sha256;
                report["expert_profile"]["operator_phase_sha256"]=phase_sha256;
                report["expert_profile"]["column_schedule"]=columns;
                report["expert_profile"]["loader_unroll"]=loader_unroll;
                report["expert_profile"]["full_model_qualified"]=full_model_qualified;
            }
            if(native_columns==32){
                using namespace llama_dsv41_native_columns_resource_proof;
                report["expert_profile"]["dispatch_family"]="q4k_natural_half_rhs_mmid_columns32_packed";
                report["expert_profile"]["mathematical_source_sha256"]=source_sha256;
                report["expert_profile"]["mathematical_object_sha256"]=object_sha256;
                report["expert_profile"]["baseline_native_half_matrix_proof_sha256"]=report["expert_profile"]["native_half_matrix_proof_sha256"];
                report["expert_profile"]["native_half_matrix_proof_sha256"]=nullptr;
                report["expert_profile"]["resource_fixture_receipt_sha256"]=fixture_sha256;
                report["expert_profile"]["operator_phase_sha256"]=phase_sha256;
                report["expert_profile"]["column_schedule"]=columns;
                report["expert_profile"]["full_model_qualified"]=full_model_qualified;
            }

        }
        {
            const uint64_t pv_actual=ggml_backend_cuda_dsv41_pv_view_queued()-warm_pv_view;
            const uint64_t pv_step=pair_ffn&&llama_dsv41_expert_profile::natural_half(expert)?2048:compute_step;
            const uint64_t pv_expected=pv_view?40ull*([&]{uint64_t n=0;for(uint64_t first=0;first<tokens.size();first+=chunk_tokens)n+=(std::min<uint64_t>(chunk_tokens,tokens.size()-first)+pv_step-1)/pv_step;return n;}()):0;
            require(pv_actual==pv_expected,"PV view actual dispatch count differs from completed graphs");
            report["attention_layout"]["prompt_pv_calls_after_completed_sweep"]=pv_actual;
            report["attention_layout"]["warmup_pv_calls"]=warm_pv_view;
            const uint64_t gather_actual=ggml_backend_cuda_dsv41_gather_half_queued()-warm_gather_half;
            const uint64_t gather_expected=fused_gather?40ull*([&]{uint64_t n=0;for(uint64_t first=0;first<tokens.size();first+=chunk_tokens)n+=(std::min<uint64_t>(chunk_tokens,tokens.size()-first)+pv_step-1)/pv_step;return n;}()):0;
            require(gather_actual==gather_expected,"Fused gather dispatch count differs from completed graphs");
            report["attention_layout"]["prompt_gather_half_calls_after_completed_sweep"]=gather_actual;
            const uint64_t actual_split_f32=ggml_backend_cuda_dsv41_split_f32_queued()-warm_split_f32;
            require(split_f32||actual_split_f32==0,"Split-F32 dispatched while both explicit options are off");
            report["split_f32_tensor_core_attention"]={{"enabled",split_f32},{"qk_enabled",split_f32_qk},{"pv_enabled",split_f32_pv},{"algorithm","CUTLASS fused three TF32 products; F32 accumulate"},{"tile","64x64x32"},{"prompt_completed_calls",actual_split_f32},{"warmup_calls",warm_split_f32},{"eligibility","native QK/PV F32 contiguous,64 heads,aligned4 key count<=640; other shapes retain SGEMM"},{"native_envelope_qualified",false}};
            report["attention_layout"]["warmup_gather_half_calls"]=warm_gather_half;
            const uint64_t split_actual=ggml_backend_cuda_dsv41_gather_half_split_queued()-warm_gather_split;
            const uint64_t split_expected=fused_gather?38ull*([&]{uint64_t n=0;for(uint64_t first=0;first<tokens.size();first+=chunk_tokens)n+=(std::min<uint64_t>(chunk_tokens,tokens.size()-first)+pv_step-1)/pv_step;return n;}()):0;
            require(tokens.size()<2||split_actual==split_expected,"Split-table gather count differs from compressed layers");
            report["attention_layout"]["prompt_split_table_calls_after_completed_sweep"]=split_actual;
            report["attention_layout"]["warmup_split_table_calls"]=warm_gather_split;
        }
        require(rhs_pack||ggml_backend_cuda_dsv41_rhs_pack_queued()==0,"Native RHS packing dispatched while disabled");
        if(llama_dsv41_expert_profile::activation_rounding_q8_1(expert)){
            report["expert_profile"]["dispatch_family"]="shared_mmq_id_q8_1";
            report["expert_profile"]["resource_fixture_receipt_sha256"]=nullptr;
            report["expert_profile"]["kernel_selection_note"]="Existing untagged MUL_MAT_ID dispatch; Q8_1 activation rounding enabled explicitly; no native-half numerical qualification";
        }
        if(expert.expert_f16_cublas){
            const uint64_t expected=120ull*([&]{uint64_t n=0;for(uint64_t first=0;first<tokens.size();first+=chunk_tokens)n+=(std::min<uint64_t>(chunk_tokens,tokens.size()-first)+compute_step-1)/compute_step;return n;}());
            require(ggml_backend_cuda_dsv41_expert_cublas_queued()-warm_cublas==expected,"F16 expert cuBLAS aggregate witness incomplete");
            report["expert_profile"]["name"]="f16_operands_f32_accumulation_cublas";
            report["expert_profile"]["dispatch_family"]="shared_sort_gather_cublas_F16_F16_F32_scatter";
            report["expert_profile"]["actual_total_completed"]=expected;
            report["expert_profile"]["resource_fixture_receipt_sha256"]=nullptr;
            report["expert_profile"]["kernel_selection_note"]="Bounded one-expert dequantization and existing conversion/gather/scatter kernels; no Q8_1 activations or full dequantized bank; unqualified TEST summation order";
        }
        // File carry has consumed its admitted disk extent; RAM carry never
        // reserves disk bytes. Neither remains a future output-volume payload.
        if(payload)payload->outstanding(add(output_other,receipt_overlap?receipt_limit:0));receipt->outstanding(receipt_overlap?output_other:0);
        hash payload_hash;uint64_t written=0;std::vector<uint8_t>buffer(chunk);started=clock_type::now();
        const bool exported=sweep->export_native([&](llama_dsv41::wire_payload_reader&reader,std::string&reason){
            try{require(reader.bytes()==plan.bytes,"Native export plan changed after admission");
                while(written<reader.bytes()){stop();const size_t count=size_t(std::min<uint64_t>(chunk,reader.bytes()-written));
                    success(reader.read(written,buffer.data(),count,reason),reason);if(payload)payload->write(buffer.data(),count);payload_hash.update(buffer.data(),count);written+=count;}
                if(payload)payload->flush();return true;
            }catch(const std::exception&failure){reason=failure.what();return false;}
        },error);report["phase_ms"]["native_payload_export_flush"]=ms(started);success(exported,error);stop();
        require(written==plan.bytes&&(hash_only||payload->current().size==plan.bytes),"Completed canonical payload byte count differs");
        const auto payload_sha=payload_hash.finish();started=clock_type::now();if(payload)payload->finalize_read_only(payload_sha);
        report["phase_ms"]["payload_close_readonly_rehash"]=ms(started);report["payload"]["sha256"]=payload_sha;
        report["payload"]["written_bytes"]=hash_only?0:written;report["payload"]["canonical_hashed_bytes"]=written;report["payload"]["identity"]=payload?describe(payload->current()):json(nullptr);
        report["status"]="complete";report["runtime_qualified"]=false;report["claim"]=hash_only?"Completed canonical frontier SHA; no payload file created or receiver parity claimed":"One measured offline producer artifact; independent receiver/output parity qualification is still required";
        report["residency_limit"]="Free-memory admission and allocation ceilings do not independently prove physicalVRAM residency under WDDM";
        const auto text=report.dump(2)+"\n";require(text.size()<=receipt_limit,"Producer receipt exceeds its bounded admission");receipt->write(text.data(),text.size());receipt->flush();stop();
        receipt->publish(final_receipt);
        if(live){run_live_session(sweep,weights,options,session,session_timeout,reserved_chunk,session_output.get());return 0;}
        std::fputs(hash_only?"Canonical native frontier hashed; receipt complete; no payload file created.\n":"Offline native payload and receipt completed; receiver/parity qualification remains pending.\n",stdout);return 0;
    }catch(const std::exception&failure){
        if(sweep)sweep->cancel();report["status"]=interrupted.load()?"cancelled":"failed";report["error"]=failure.what();report["memory_floor_abort"]=memory_abort.load();
        if(payload){try{report["payload_actual_identity"]=describe(payload->current());}catch(...){}}
        if(receipt){try{const auto text=report.dump(2)+"\n";if(receipt->current().size==0&&text.size()<=receipt_limit){receipt->write(text.data(),text.size());receipt->flush();}}catch(...){}}
        std::cerr<<"Offline producer failed: "<<failure.what()<<". Existing/unfinished files were preserved.\n";return interrupted.load()?130:1;
    }
}
#else
#include <cstdio>
int main(){std::fputs("Offline V4.1 producer requires the opt-in Windows CUDA build; no model or output was opened.\n",stderr);return 2;}
#endif
