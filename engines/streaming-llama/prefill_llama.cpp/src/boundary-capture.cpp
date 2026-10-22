#include "boundary-capture.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif
#ifndef DSV41_BOUNDARY_SOURCE_BASE_REVISION
#define DSV41_BOUNDARY_SOURCE_BASE_REVISION "UNATTESTED"
#endif

namespace dsv41_boundary {
namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(std::string("BOUNDARY_CAPTURE_REFUSED: ") + message);
}
bool hex(const std::string &s, size_t n) {
    return s.size() == n && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
uint32_t rotr(uint32_t v, unsigned n) { return (v >> n) | (v << (32 - n)); }
struct digest {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<unsigned char,64> block{};
    uint64_t total = 0;
    size_t pending = 0;
    void transform() {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (unsigned i=0;i<16;++i) w[i]=(uint32_t(block[4*i])<<24)|(uint32_t(block[4*i+1])<<16)|
            (uint32_t(block[4*i+2])<<8)|block[4*i+3];
        for (unsigned i=16;i<64;++i) {
            uint32_t a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(rotr(a,7)^rotr(a,18)^(a>>3))+w[i-7]+(rotr(b,17)^rotr(b,19)^(b>>10));
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for (unsigned i=0;i<64;++i) {
            uint32_t x=z+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            uint32_t y=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            z=g;g=f;f=e;e=d+x;d=c;c=b;b=a;a=x+y;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    }
    void update(const void *input, size_t n) {
        auto *data=static_cast<const unsigned char *>(input);
        require(n<=UINT64_MAX-total,"digest size overflow");total+=n;
        while(n) {size_t take=std::min(n,64-pending);std::memcpy(block.data()+pending,data,take);
            pending+=take;data+=take;n-=take;if(pending==64){transform();pending=0;}}
    }
    std::string finish() {
        require(total<=UINT64_MAX/8,"digest bit count overflow");uint64_t bits=total*8;
        unsigned char one=0x80,zero=0;update(&one,1);
        while(pending!=56)update(&zero,1);
        unsigned char end[8];for(unsigned i=0;i<8;++i)end[7-i]=(unsigned char)(bits>>(8*i));update(end,8);
        char out[65];for(unsigned i=0;i<8;++i)std::snprintf(out+8*i,9,"%08x",h[i]);return out;
    }
};
using json=nlohmann::json;
json strict_json(const std::string&text){
    std::vector<std::set<std::string>>keys;
    return json::parse(text,[&](int,json::parse_event_t event,json&value){
        if(event==json::parse_event_t::object_start)keys.emplace_back();
        else if(event==json::parse_event_t::key)require(!keys.empty()&&keys.back().insert(value.get<std::string>()).second,"duplicate replay JSON key");
        else if(event==json::parse_event_t::object_end){require(!keys.empty(),"replay JSON object state");keys.pop_back();}
        return true;
    });
}
thread_local capture *current = nullptr;
struct shape {const char *name;int64_t width,copies;ggml_type type=GGML_TYPE_F32;};
constexpr shape shapes[] = {
    {"q_after_rope_f32",512,64},{"attention_heads_f32",512,64},{"attention_heads_bf16",512,64},
    {"attention_sinks_f32",64,1},{"raw_ids",32,1,GGML_TYPE_I32},{"raw_mask_f32",32,1},
    {"hc_projection_f32",24,1},{"hc_post_f32",4,1},{"hc_comb_f32",4,4},
    {"common_residual",5120,4},{"previous_pre",4,1},{"collapsed_bf16",5120,1},
    {"weighted_norm_f32",5120,1},{"weighted_norm_bf16",5120,1},
    {"kv_projection_f32",512,1},{"kv_projection_bf16",512,1},
    {"kv_norm_f32",512,1},{"kv_norm_bf16",512,1},{"kv_rope_bf16",512,1},{"kv_fp8_e8m0",512,1},
    {"attention_block_bf16",5120,1},{"after_attention_hc_bf16",5120,4},
    {"ffn_weighted_norm_bf16",5120,1},{"route_ids",6,1,GGML_TYPE_I32},{"route_weights",6,1},
    {"routed_ffn_f32",5120,1},{"shared_ffn_bf16",5120,1},{"ffn_block_bf16",5120,1},{"final_hc_bf16",5120,4}};
const shape &expected(const std::string &name) {
    for(const auto &s:shapes)if(name==s.name)return s;
    throw std::runtime_error("BOUNDARY_CAPTURE_REFUSED: unknown stage "+name);
}
std::string read_small(const std::filesystem::path &path) {
    std::ifstream f(path,std::ios::binary);require(bool(f),"trusted config is absent");
    std::string data(4097,'\0');f.read(&data[0],4097);data.resize(size_t(f.gcount()));
    require(data.size()<=4096,"trusted config exceeds 4096 bytes");return data;
}
#if defined(_WIN32)
const wchar_t *diagnostic_root() { const auto *p = _wgetenv(L"SLSTACK_DIAGNOSTIC_ROOT"); require(p && *p, "Set SLSTACK_DIAGNOSTIC_ROOT for diagnostics"); return p; }
void physical_root() {
    for(const auto *path:{diagnostic_root()}) {
        const DWORD attr=GetFileAttributesW(path);
        require(attr!=INVALID_FILE_ATTRIBUTES&&(attr&FILE_ATTRIBUTE_DIRECTORY)&&
                !(attr&FILE_ATTRIBUTE_REPARSE_POINT),"evidence root is not an ordinary physical directory");
    }
}
std::string executable_digest() {
    wchar_t path[32768];DWORD count=GetModuleFileNameW(nullptr,path,32768);
    require(count&&count<32768,"cannot identify actual executable");
    std::ifstream f(std::filesystem::path(path),std::ios::binary);require(bool(f),"cannot read actual executable");
    digest d;char block[65536];uint64_t total=0;
    while(f.read(block,sizeof(block))||f.gcount()) {size_t n=size_t(f.gcount());total+=n;
        require(total<=256ull*1024*1024,"executable exceeds diagnostic bound");d.update(block,n);}
    require(f.eof(),"executable read failed");return d.finish();
}
FILE *exclusive(const std::wstring &name) {
    physical_root();auto path=std::filesystem::path(diagnostic_root())/name;
    HANDLE h=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    require(h!=INVALID_HANDLE_VALUE,"capture output exists or cannot be exclusively created");
    int fd=_open_osfhandle((intptr_t)h,_O_WRONLY|_O_BINARY);
    if(fd<0){CloseHandle(h);require(false,"capture descriptor conversion failed");}
    FILE *f=_fdopen(fd,"wb");if(!f){_close(fd);require(false,"capture stream conversion failed");}return f;
}
#endif
struct file_owner {
    FILE *fp=nullptr;
    ~file_owner(){if(fp)std::fclose(fp);}
    void close(){require(fp&&std::fflush(fp)==0,"capture flush failed");FILE *p=fp;fp=nullptr;
        require(std::fclose(p)==0,"capture close failed");}
};
}

std::string sha256(const void *data,size_t bytes){digest d;d.update(data,bytes);return d.finish();}
std::string sha256_stream(uint64_t bytes,const std::function<bool(uint64_t,void *,size_t)>&read){
    require(bool(read),"SHA256 stream requires a reader");
    digest d;std::array<unsigned char,65536> page{};
    for(uint64_t offset=0;offset<bytes;){
        const size_t count=size_t(std::min<uint64_t>(page.size(),bytes-offset));
        require(read(offset,page.data(),count),"SHA256 stream read failed");
        d.update(page.data(),count);offset+=count;
    }
    return d.finish();
}
config parse_config(const std::string &text) {
    require(text.size()<=4096&&!text.empty()&&text.find('\0')==std::string::npos,"invalid trusted config bytes");
    std::map<std::string,std::string> values;std::istringstream in(text);std::string line;
    while(std::getline(in,line)) {
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        const size_t eq=line.find('=');require(eq!=std::string::npos&&eq&&eq+1<line.size(),"expected exact key=value config line");
        require(values.emplace(line.substr(0,eq),line.substr(eq+1)).second,"duplicate config key");
    }
    const std::set<std::string> keys={"schema","nonce","source_base_revision","binary_sha256",
        "residual_sha256","previous_pre_sha256","variant_manifest_sha256","capture_bytes",
        "replay_arm","replay_inputs_sha256","weight_proof_sha256","attention_contract_sha256"};
    require(values.size()==keys.size()&&std::all_of(values.begin(),values.end(),[&](const auto &v){return keys.count(v.first)!=0;}),"unknown or missing config key");
    require(values["schema"]=="ds4split-boundary-capture-v4","wrong capture schema");
    config c;c.nonce=values["nonce"];c.source_base_revision=values["source_base_revision"];c.binary_sha256=values["binary_sha256"];
    c.variant_manifest_sha256=values["variant_manifest_sha256"];
    c.replay_arm=values["replay_arm"];c.replay_inputs_sha256=values["replay_inputs_sha256"];c.weight_proof_sha256=values["weight_proof_sha256"];
    c.attention_contract_sha256=values["attention_contract_sha256"];require(hex(c.attention_contract_sha256,64),"attention contract pin grammar");
    require((c.replay_arm=="attention"||c.replay_arm=="attention_capture")&&hex(c.replay_inputs_sha256,64)&&hex(c.weight_proof_sha256,64),"invalid replay profile/pins");
    c.residual_sha256=values["residual_sha256"];c.previous_pre_sha256=values["previous_pre_sha256"];
    require(hex(c.nonce,32)&&hex(c.source_base_revision,40)&&hex(c.binary_sha256,64)&&
            hex(c.residual_sha256,64)&&hex(c.previous_pre_sha256,64)&&hex(c.variant_manifest_sha256,64),"malformed input/source/binary pin");
    const auto &v=values["capture_bytes"];require(v.size()<=8&&std::all_of(v.begin(),v.end(),[](char x){return x>='0'&&x<='9';}),"invalid capture cap");
    c.capture_bytes=std::stoull(v);require(c.capture_bytes>0&&c.capture_bytes<=maximum_capture_bytes,"capture cap exceeds 32 MiB");return c;
}


#if defined(_WIN32)
struct held_source {
    HANDLE h=INVALID_HANDLE_VALUE;
    ~held_source(){if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}
    void open(const json&proof){
        const std::string path=proof.at("model_path").get<std::string>();
        require(path.size()<2048&&path.size()>3&&path[0]=='D'&&path[1]==':'&&(path[2]=='\\'||path[2]=='/'),"replay source must remain on D");
        // No write/delete sharing: the file identity cannot be replaced during proof verification.
        h=CreateFileW(std::filesystem::path(path).c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(h!=INVALID_HANDLE_VALUE,"cannot hold original-model source identity");
        wchar_t final_path[32768];const DWORD length=GetFinalPathNameByHandleW(h,final_path,32768,FILE_NAME_NORMALIZED|VOLUME_NAME_DOS);
        require(length>7&&length<32768&&final_path[4]==L'D'&&final_path[5]==L':'&&final_path[6]==L'\\',"held model must physically resolve on D");
        BY_HANDLE_FILE_INFORMATION old{};FILE_ID_INFO id{};LARGE_INTEGER size{};
        require(GetFileInformationByHandle(h,&old)&&!(old.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT)&&
            GetFileInformationByHandleEx(h,FileIdInfo,&id,sizeof(id))&&GetFileSizeEx(h,&size),"source identity read failed");
        char file_id[33]={};for(int i=0;i<16;++i)std::snprintf(file_id+2*i,3,"%02x",id.FileId.Identifier[i]);
        const auto&i=proof.at("model_identity");
        const uint64_t mtime=(uint64_t(old.ftLastWriteTime.dwHighDateTime)<<32)|old.ftLastWriteTime.dwLowDateTime;
        require(i.at("volume_serial_u64").get<uint64_t>()==id.VolumeSerialNumber&&i.at("file_id_le_hex").get<std::string>()==file_id&&
            i.at("bytes").get<uint64_t>()==uint64_t(size.QuadPart)&&i.at("mtime_filetime_ticks").get<uint64_t>()==mtime,"original-model identity mismatch");
    }
};
std::string read_bounded(const std::filesystem::path&path,size_t cap){
    const DWORD attributes=GetFileAttributesW(path.c_str());
    require(attributes!=INVALID_FILE_ATTRIBUTES&&!(attributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT)),"replay file is absent/not ordinary");
    std::ifstream f(path,std::ios::binary);require(bool(f),"replay file open failed");
    std::string data(cap+1,'\0');f.read(&data[0],std::streamsize(data.size()));data.resize(size_t(f.gcount()));
    require(data.size()<=cap&&f.eof(),"replay file exceeds explicit bound");return data;
}
#endif
struct capture::implementation {
    struct entry {ggml_tensor *original=nullptr,*retained=nullptr;std::string name;};
    bool enabled=false,sealed=false,read=false,test=false;
    uint32_t rows=0,norms=0,blocks=0,expansions=0;uint64_t extent=0;
    config options;std::string config_sha;
    std::vector<entry> entries;
    struct seed {std::string name,sha;ggml_type type;int64_t ne[4];std::string data;ggml_tensor*leaf=nullptr;};
    std::map<std::string,seed>seeds;
    json weight_proof,attention_contract,weight_records=json::array();std::set<std::string>verified_weights;
    std::set<int32_t>selected_experts;uint32_t prepares=0,projection_count=0,post_count=0,comb_count=0;
    bool uploaded=false,attention_contract_verified=false;uint64_t override_host_bytes=0,source_bytes_read=0;
#if defined(_WIN32)
    held_source source;
#endif

    std::vector<record> readback(ggml_backend_t backend,const block_sink &sink) {
        require(enabled&&sealed&&!read&&backend,"capture not retained, repeated, or backend absent");
        ggml_backend_synchronize(backend);std::vector<record> out;
        alignas(float) unsigned char data[65536];
        for(const auto &e:entries) {
            record r;r.stage=e.name;r.type=e.retained->type;r.bytes=ggml_nbytes(e.retained);
            std::copy(e.retained->ne,e.retained->ne+4,r.ne);digest d;
            for(uint64_t offset=0;offset<r.bytes;) {
                size_t n=size_t(std::min<uint64_t>(sizeof(data),r.bytes-offset));
                ggml_backend_tensor_get(e.retained,data,size_t(offset),n);
                for(size_t i=0;i<n;i+=4) {
                    if(r.type==GGML_TYPE_I32){int32_t value;std::memcpy(&value,data+i,4);
                        require(value>=0&&value<(r.stage=="raw_ids"?32:384),"captured I32 ID outside admitted table");}
                    else{float value;std::memcpy(&value,data+i,4);
                        if(r.stage=="raw_mask_f32")require(value==0.f||(std::isinf(value)&&value<0),"invalid causal mask value");
                        else require(std::isfinite(value),"nonfinite captured boundary");}
                    if(offset+i==94*4&&r.ne[0]>94){r.has_token0_dim94=true;std::memcpy(&r.token0_dim94_bits,data+i,4);}
                }
                d.update(data,n);sink(e.name.c_str(),data,n);offset+=n;
            }
            r.sha256=d.finish();out.push_back(r);
        }
        read=true;return out;
    }
};

capture::capture(std::unique_ptr<implementation> v):p(std::move(v)){}
capture::capture(uint32_t layer,uint32_t start,uint32_t count,const std::vector<float>&residual,
                 const std::vector<float>&pre):p(nullptr) {
    const char *flag=std::getenv("DSV41_BOUNDARY_CAPTURE");if(!flag||!*flag)return;
    require(std::strcmp(flag,"1")==0,"DSV41_BOUNDARY_CAPTURE must be exactly 1 or absent");
    if(layer!=0)return; // Other layers retain the original execution path.
    require(start==0&&count==32,"capture requires exact layer0/start0/32-row pilot");
    p=std::make_unique<implementation>();
#if defined(_WIN32)
    physical_root();std::string raw=read_small(std::filesystem::path(diagnostic_root())/L"boundary-capture.config");
    p->options=parse_config(raw);p->config_sha=sha256(raw.data(),raw.size());
    require(p->options.source_base_revision==DSV41_BOUNDARY_SOURCE_BASE_REVISION,"compiled source revision does not match trusted config");
    require(executable_digest()==p->options.binary_sha256,"actual capture executable differs from trusted config");
    // The owner build receipt binds this manifest to the observed binary. The
    // base revision alone is deliberately never described as the variant SHA.
    const std::string manifest=read_small(std::filesystem::path(diagnostic_root())/L"boundary-capture.source-manifest.json");
    require(sha256(manifest.data(),manifest.size())==p->options.variant_manifest_sha256,
            "owner capture-source build manifest changed or is absent");
    require(residual.size()==size_t(5120)*4*count&&pre.size()==size_t(4)*count,"common input geometry changed");
    require(sha256(residual.data(),residual.size()*4)==p->options.residual_sha256&&
            sha256(pre.data(),pre.size()*4)==p->options.previous_pre_sha256,"actual common input SHA differs from predeclared config");
    
    load_replay(read_small(std::filesystem::path(diagnostic_root())/L"boundary-replay.inputs.json"),
        read_bounded(std::filesystem::path(diagnostic_root())/L"boundary-replay.weights.json",256*1024),
        [&](const std::string&name,size_t bytes){return read_bounded(std::filesystem::path(diagnostic_root())/name,bytes);});
    p->enabled=true;p->rows=count;
#else
    require(false,"enabled producer capture requires Windows RTX fixture build");
#endif
}
void capture::load_replay(const std::string&inputs_raw,const std::string&proof_raw,
        const std::function<std::string(const std::string&,size_t)>&loader){
    require(inputs_raw.size()<=4096&&proof_raw.size()<=256*1024,"replay manifest bound");
#if defined(_WIN32)
    require(sha256(inputs_raw.data(),inputs_raw.size())==p->options.replay_inputs_sha256,"replay input manifest SHA mismatch");
    const auto inputs=strict_json(inputs_raw);
    require(inputs.size()==4&&inputs.at("schema")=="ds4split-replay-inputs-v1"&&inputs.at("arm")==p->options.replay_arm&&inputs.at("rows")==32,"replay input manifest contract");
    const std::vector<std::string>required=p->options.replay_arm=="attention"?std::vector<std::string>{"q_after_rope_f32","kv_fp8_e8m0"}:std::vector<std::string>{};
    require(inputs.at("seeds").is_object()&&inputs.at("seeds").size()==required.size(),"exclusive replay seed set");
    for(const auto&name:required){
        const auto&s=expected(name);const bool residual=s.copies>1;
        implementation::seed seed;seed.name=name;seed.type=s.type;seed.ne[0]=s.width;seed.ne[1]=residual?s.copies:32;seed.ne[2]=residual?32:1;seed.ne[3]=1;
        const auto&pin=inputs.at("seeds").at(name);require(pin.size()==3&&pin.at("ne")==json::array({seed.ne[0],seed.ne[1],seed.ne[2],seed.ne[3]})&&
            pin.at("format")==json(s.type==GGML_TYPE_I32?"i32-le":"ieee754-f32-le"),"replay seed type/axis order");
        seed.sha=pin.at("sha256").get<std::string>();require(hex(seed.sha,64),"replay seed SHA grammar");
        const uint64_t bytes=uint64_t(s.width)*s.copies*32*4;
        const std::string filename="replay-"+p->options.nonce+"."+name+(s.type==GGML_TYPE_I32?".i32":".f32");
        seed.data=loader(filename,size_t(bytes));
        require(seed.data.size()==bytes&&sha256(seed.data.data(),seed.data.size())==seed.sha,"exact pinned replay seed bytes");
        for(size_t off=0;off<seed.data.size();off+=4){
            if(s.type==GGML_TYPE_I32){int32_t value;std::memcpy(&value,seed.data.data()+off,4);require(value>=0&&value<384,"replay expert ID outside bank");p->selected_experts.insert(value);}
            else {float value;uint32_t bits;std::memcpy(&value,seed.data.data()+off,4);std::memcpy(&bits,seed.data.data()+off,4);
                require(std::isfinite(value),"nonfinite replay seed");
                if(name=="attention_block_bf16"||name=="ffn_weighted_norm_bf16")require((bits&0xffffu)==0,"replay BF16 seed has non-BF16 F32 word");}
        }
        p->override_host_bytes+=seed.data.size();p->seeds.emplace(name,std::move(seed));
    }
    require(p->override_host_bytes<=8ull*1024*1024,"explicit replay host seed bound");
    if(p->options.replay_arm=="hc")require(p->seeds.at("common_residual").sha==p->options.residual_sha256,"HC residual differs from canonical input pin");
    require(sha256(proof_raw.data(),proof_raw.size())==p->options.weight_proof_sha256,"weight proof manifest SHA mismatch");
    p->weight_proof=strict_json(proof_raw);
    require(p->weight_proof.size()==5&&p->weight_proof.at("schema")=="ds4split-replay-weight-proof-v1"&&
        p->weight_proof.at("arm")==p->options.replay_arm&&p->weight_proof.at("weights").is_array(),"weight proof manifest contract");
    p->source.open(p->weight_proof);

    const auto contract_raw=loader("boundary-attention.contract.json",4096);
    require(sha256(contract_raw.data(),contract_raw.size())==p->options.attention_contract_sha256,"attention contract SHA mismatch");
    p->attention_contract=strict_json(contract_raw);
    require(p->attention_contract.size()==9&&p->attention_contract.at("schema")=="ds4split-cold32-attention-contract-v2"&&p->attention_contract.at("rows")==32,"attention contract grammar");
    for(const auto*key:{"raw_ids_sha256","raw_mask_f32_sha256","native_mask_f16_sha256","native_runtime_sha256","native_receipt_sha256","sinks_sha256"})
        require(hex(p->attention_contract.at(key).get<std::string>(),64),"attention contract pin grammar");
    const auto native_raw=loader("boundary-attention.native.json",65536);
    require(sha256(native_raw.data(),native_raw.size())==p->attention_contract.at("native_receipt_sha256"),"complete original native receipt SHA mismatch");
    const auto receipt=strict_json(native_raw);const auto&runtime=p->attention_contract.at("native_runtime");
    const auto canonical_runtime=runtime.dump();
    require(sha256(canonical_runtime.data(),canonical_runtime.size())==p->attention_contract.at("native_runtime_sha256")&&
        runtime==receipt.at("raw_attention_parameters"),"canonical runtime object SHA or complete receipt object binding mismatch");
    const auto expected_runtime=strict_json(R"DS4ACTUAL({"bc_mask":false,"entry":"ds4_gpu_attention_decode_raw_batch_heads_tensor","has_kvpad":true,"head_dim":512,"heads":64,"ids":"implicit_raw_positions_0_through_31","internal_padding_bytes":135168,"kernel":"kernel_flash_attn_ext_f16_dk512_dv512","logical_mask_bytes":2048,"logical_mask_copied_from_padding":false,"logical_mask_stride_bytes":64,"mask":"actual_cpu_f16_mask","n_raw":32,"ncpsg":64,"noncausal":false,"nqptg":8,"nsg":8,"raw_cap":32,"raw_start":0,"rows":32,"sinks":"pinned_original_model","start":0,"window":128})DS4ACTUAL");
    require(runtime==expected_runtime,"actual native cold32 geometry/dispatch object changed; owner review required");
    require(receipt.at("schema")=="ds4split-native-layer0-replay-v1"&&receipt.at("complete")==true&&receipt.at("mode")=="attention_capture"&&
        receipt.at("layer")==0&&receipt.at("start")==0&&receipt.at("rows")==32&&receipt.at("kernel_replaced")==false&&
        receipt.at("prompt_sha256")=="8c6b8bef556f819920623362e38abd040f1d2faa90cf1fe863c4de1b3e067fee","original native cold canonical capture receipt contract");
    require(receipt.at("actual_attention_mask_f16_sha256")==p->attention_contract.at("native_mask_f16_sha256"),"actual native logical mask receipt binding");
    auto native_stage=[&](const char*name)->const json&{const auto&stages=receipt.at("stages");const json*found=nullptr;
        for(const auto&stage:stages)if(stage.at("stage")==name){require(!found,"duplicate native capture stage");found=&stage;}
        require(found!=nullptr,"missing native capture stage");return *found;};
    require(native_stage("attention_mask_f32").at("sha256")==p->attention_contract.at("raw_mask_f32_sha256")&&
        native_stage("attention_sinks_f32").at("sha256")==p->attention_contract.at("sinks_sha256"),"native mask/sinks stage receipt binding");
    if(p->options.replay_arm=="attention")for(const auto*name:{"q_after_rope_f32","kv_fp8_e8m0"})
        require(native_stage(name).at("sha256")==p->seeds.at(name).sha,"canonical Q/KV seed differs from complete native receipt");
    require(p->weight_proof.at("weights").size()==1&&p->weight_proof.at("weights")[0].at("name")=="attention_sinks"&&
        p->weight_proof.at("weights")[0].at("slices").size()==1&&p->weight_proof.at("weights")[0].at("slices")[0].at("sha256")==p->attention_contract.at("sinks_sha256"),
        "attention source sinks proof binding");
    // Backend still validates all module/binary/config pins at its original
    // call site; the attention profile independently refuses enabled1 here.
    std::istringstream observer(loader("mmid-capture.config",4096));std::string line;unsigned disabled=0,reservation=0;
    while(std::getline(observer,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(line=="enabled=0")++disabled;if(line=="existing_boundary_bytes=26099456")++reservation;
        require(line!="enabled=1","attention profile requires observer enabled=0");}
    require(disabled==1&&reservation==1,"explicit disabled observer and exact attention reservation required");


#else
    require(false,"replay loading requires Windows");
#endif
}
capture::~capture()=default;
bool capture::active()const{return p&&p->enabled;}
void capture::add(ggml_tensor *tensor,const char *stage_name) {
    if(!active())return;require(!p->sealed&&tensor&&stage_name,"stage after retention or absent tensor");
    const auto &s=expected(stage_name);
    require(tensor->type==s.type&&ggml_is_contiguous(tensor)&&tensor->ne[0]==s.width&&
        ggml_nelements(tensor)==s.width*s.copies*(std::strcmp(stage_name,"attention_sinks_f32")==0?1:p->rows),"boundary stage geometry/type changed");
    const bool residual=s.copies>1;
    const bool rotated=std::strcmp(stage_name,"kv_rope_bf16")==0||std::strcmp(stage_name,"kv_fp8_e8m0")==0;
    require(tensor->ne[1]==(std::strcmp(stage_name,"attention_sinks_f32")==0?1:residual?s.copies:rotated?1:p->rows)&&
            tensor->ne[2]==(residual||rotated?p->rows:1)&&tensor->ne[3]==1,
            "boundary token/copy axis order changed");
    require(std::none_of(p->entries.begin(),p->entries.end(),[&](const auto&e){return e.name==stage_name;}),"duplicate required boundary stage");
    const uint64_t n=ggml_nbytes(tensor);require(n<=p->options.capture_bytes-p->extent,"capture bytes exceed explicit cap");
    p->extent+=n;p->entries.push_back({tensor,nullptr,stage_name});
}
void capture::observe(ggml_tensor *tensor,const char *name) {
    if(!active())return;
    if(!std::strcmp(name,"dsv41_hc_mixes")){if(p->projection_count++==0)add(tensor,"hc_projection_f32");}
    else if(!std::strcmp(name,"dsv41_hc_post")){if(p->post_count++==0)add(tensor,"hc_post_f32");}
    else if(!std::strcmp(name,"dsv41_hc_comb")){if(p->comb_count++==0)add(tensor,"hc_comb_f32");}
    else if(!std::strcmp(name,"dsv41_hc_pre_bf16")) {
        if(std::none_of(p->entries.begin(),p->entries.end(),[](const auto&e){return e.name=="collapsed_bf16";}))add(tensor,"collapsed_bf16");
    }else if(!std::strcmp(name,"dsv41_weighted_norm_f32")) {
        ++p->norms;
        if(p->norms==1)add(tensor,"weighted_norm_f32");
        else if(p->norms==2)require(tensor->ne[0]==1280,"query norm ordering changed");
        else if(p->norms==3)add(tensor,"kv_norm_f32");
        else require(p->norms==4&&tensor->ne[0]==5120,"FFN norm ordering changed");
    }else if(!std::strcmp(name,"dsv41_weighted_norm_bf16")) {
        if(p->norms==1)add(tensor,"weighted_norm_bf16");
        else if(p->norms==3)add(tensor,"kv_norm_bf16");
        else if(p->norms==4)add(tensor,"ffn_weighted_norm_bf16");
    }else if(!std::strcmp(name,"dsv41_block_before_expand_bf16")){
        require(p->blocks<2,"unexpected extra block boundary");
        add(tensor,p->blocks++==0?"attention_block_bf16":"ffn_block_bf16");
    }else if(!std::strcmp(name,"dsv41_hc_expand_bf16")){
        require(p->expansions<2&&p->blocks==p->expansions+1,"HC expansion ordering changed");
        add(tensor,p->expansions++==0?"after_attention_hc_bf16":"final_hc_bf16");
    }
}
void capture::retain(ggml_context *ctx,ggml_cgraph *graph) {
    if(!active())return;require(!p->sealed&&ctx&&graph&&p->entries.size()==sizeof(shapes)/sizeof(*shapes)&&p->norms==4&&p->blocks==2&&p->expansions==2&&p->projection_count==2&&p->post_count==2&&p->comb_count==2,
        "required boundary outputs are missing");
    for(const auto&s:shapes)require(std::any_of(p->entries.begin(),p->entries.end(),[&](const auto&e){return e.name==s.name;}),"missing required boundary name");
    for(auto&e:p->entries){
        if(e.original->type==GGML_TYPE_I32){
            // The pinned CUDA backend rejects I32 DUP but supports I32 CPY.
            // Preserve the independent destination leaf as well as its copy node.
            auto *destination=ggml_dup_tensor(ctx,e.original);ggml_set_output(destination);
            e.retained=ggml_cpy(ctx,e.original,destination);
        }else e.retained=ggml_dup(ctx,e.original);
        ggml_set_output(e.retained);ggml_build_forward_expand(graph,e.retained);
    }
    p->sealed=true;
}
uint64_t capture::bytes()const{return p?p->extent:0;}
std::string capture::replay_arm()const{return active()?p->options.replay_arm:std::string();}

ggml_tensor*capture::replacement(ggml_context*ctx,ggml_tensor*original,const char*tag){
    if(!active())return original;
    std::string stage_name;
    if(p->options.replay_arm=="attention"&&std::strcmp(tag,"attention_query_input")==0)stage_name="q_after_rope_f32";
    else if(p->options.replay_arm=="attention"&&std::strcmp(tag,"attention_raw_kv_input")==0)stage_name="kv_fp8_e8m0";
    else if(std::strcmp(tag,"dsv41_hc_input_residual")==0){
        const auto index=p->prepares++;
        if(p->options.replay_arm=="hc"&&index==0)stage_name="common_residual";
    }else if(p->options.replay_arm=="hc"&&std::strcmp(tag,"dsv41_block_before_expand_bf16")==0&&p->blocks==0)stage_name="attention_block_bf16";
    else if(p->options.replay_arm=="expert"&&std::strcmp(tag,"expert_input")==0)stage_name="ffn_weighted_norm_bf16";
    else if(p->options.replay_arm=="expert"&&(std::strcmp(tag,"route_ids")==0||std::strcmp(tag,"route_weights")==0))stage_name=tag;
    if(stage_name.empty())return original;
    auto&seed=p->seeds.at(stage_name);
    require(ctx&&original&&!seed.leaf&&!p->sealed&&original->type==seed.type&&ggml_is_contiguous(original)&&
        std::memcmp(original->ne,seed.ne,sizeof(seed.ne))==0,"replay replacement geometry/order or repeated use");
    seed.leaf=ggml_new_tensor_4d(ctx,seed.type,seed.ne[0],seed.ne[1],seed.ne[2],seed.ne[3]);
    ggml_set_input(seed.leaf);ggml_set_output(seed.leaf);
    // The existing norm observation precedes the after-router expert injection.
    // Repoint it so its digest describes the actual common expert input.
    if(stage_name=="ffn_weighted_norm_bf16"||stage_name=="common_residual"||stage_name=="q_after_rope_f32"||stage_name=="kv_fp8_e8m0"){
        auto it=std::find_if(p->entries.begin(),p->entries.end(),[&](const auto&e){return e.name==stage_name;});
        require(it!=p->entries.end(),"missing pre-existing seed input observation");it->original=stage_name=="kv_fp8_e8m0"?ggml_reshape_3d(ctx,seed.leaf,512,1,32):seed.leaf;
    }
    return seed.leaf;
}
void capture::upload_replacements(){
    if(!active())return;require(p->sealed&&!p->uploaded&&p->attention_contract_verified,"attention replay upload/contract order");
    for(auto&item:p->seeds){auto&s=item.second;require(s.leaf&&s.leaf->buffer,"replay leaf escaped graph allocation");
        ggml_backend_tensor_set(s.leaf,s.data.data(),0,s.data.size());}
    p->uploaded=true;
}
void capture::verify_weight(const char*name,const ggml_tensor*tensor){
    if(!active())return;
#if defined(_WIN32)
    require(name&&tensor&&tensor->buffer&&tensor->data&&ggml_is_contiguous(tensor)&&!p->uploaded,"weight proof must precede replay compute/upload");
    auto&proof=p->weight_proof.at("weights");
    auto it=std::find_if(proof.begin(),proof.end(),[&](const json&v){return v.at("name")==name;});
    require(it!=proof.end()&&p->verified_weights.insert(name).second,"missing/duplicate staged weight proof");
    const auto&entry=*it;
    require(entry.size()==7&&entry.at("type")==int(tensor->type)&&entry.at("tensor_bytes")==ggml_nbytes(tensor)&&
        entry.at("ne")==json::array({tensor->ne[0],tensor->ne[1],tensor->ne[2],tensor->ne[3]})&&
        entry.at("nb")==json::array({tensor->nb[0],tensor->nb[1],tensor->nb[2],tensor->nb[3]}),"staged weight type/strides/extent mismatch");
    const bool selected=std::string(name)=="expert_gate"||std::string(name)=="expert_up"||std::string(name)=="expert_down";
    const auto&slices=entry.at("slices");require(slices.is_array()&&slices.size()==(selected?p->selected_experts.size():1)&&slices.size()<=192,"weight slice count exceeds selected union");
    uint64_t tensor_source=entry.at("tensor_source_offset").get<uint64_t>();std::set<int32_t>observed;
    const uint64_t source_bytes=p->weight_proof.at("model_identity").at("bytes").get<uint64_t>();
    alignas(float) char block[65536];auto selected_next=p->selected_experts.begin();
    for(const auto&slice:slices){
        require(slice.size()==5,"weight slice exact fields");
        int32_t expert=slice.at("expert").get<int32_t>();
        const uint64_t staged=slice.at("staged_offset").get<uint64_t>(),source=slice.at("source_offset").get<uint64_t>(),bytes=slice.at("bytes").get<uint64_t>();
        const uint64_t expected_staged=selected?uint64_t(expert)*tensor->nb[2]:0;
        const uint64_t expected_bytes=selected?tensor->nb[2]:ggml_nbytes(tensor);
        require((selected?(expert>=0&&p->selected_experts.count(expert)&&observed.insert(expert).second):expert==-1)&&
            staged==expected_staged&&bytes==expected_bytes&&staged<=ggml_nbytes(tensor)&&bytes<=ggml_nbytes(tensor)-staged&&
            tensor_source<=UINT64_MAX-staged&&source==tensor_source+staged&&source<=source_bytes&&bytes<=source_bytes-source,
            "weight slice source/staged span does not match selected original tensor");
        if(selected){require(selected_next!=p->selected_experts.end()&&expert==*selected_next,"expert proof must use sorted unique union");++selected_next;}
        constexpr uint64_t read_budget=4ull*1024*1024*1024;
        require(p->source_bytes_read<=read_budget&&bytes<=read_budget-p->source_bytes_read,"original-source range reads exceed explicit 4GiB stream budget");
        p->source_bytes_read+=bytes;
        digest source_hash,staged_hash;LARGE_INTEGER offset;offset.QuadPart=LONGLONG(source);
        require(source<=INT64_MAX&&SetFilePointerEx(p->source.h,offset,nullptr,FILE_BEGIN),"source slice seek");
        for(uint64_t at=0;at<bytes;){DWORD take=DWORD(std::min<uint64_t>(sizeof(block),bytes-at)),read=0;
            require(ReadFile(p->source.h,block,take,&read,nullptr)&&read==take,"bounded original-model slice read");source_hash.update(block,take);
            ggml_backend_tensor_get(tensor,block,size_t(staged+at),take);staged_hash.update(block,take);at+=take;}
        const std::string expected= slice.at("sha256").get<std::string>(),actual_source=source_hash.finish(),actual_staged=staged_hash.finish();
        require(hex(expected,64)&&actual_source==expected&&actual_staged==expected,"raw original-model/staged-device weight SHA mismatch");
        p->weight_records.push_back({{"name",name},{"expert",expert},{"type",int(tensor->type)},
            {"ne",entry.at("ne")},{"nb",entry.at("nb")},{"source_offset",source},{"staged_offset",staged},{"bytes",bytes},
            {"source_sha256",actual_source},{"staged_sha256",actual_staged}});
    }
    require(!selected||observed==p->selected_experts,"selected expert union incomplete");
#else
    require(false,"replay weight proof requires held Windows source identity");
#endif
}
void capture::verify_weight_proof_complete(){
    if(!active())return;
    const std::set<std::string>expected={"attention_sinks"};
    require(p->verified_weights==expected&&p->weight_proof.at("weights").size()==expected.size(),"exclusive replay weight proof incomplete");
}


void capture::verify_attention_contract(const std::vector<int32_t>&ids,const std::vector<float>&mask){
    if(!active())return;require(ids.size()==32*32&&mask.size()==32*32&&!p->attention_contract_verified,"attention arrays geometry/repeated verification");
    std::vector<uint16_t>half_mask(mask.size());
    for(size_t token=0;token<32;++token)for(size_t key=0;key<32;++key){const size_t at=token*32+key;const bool visible=key<=token;
        require(ids[at]==int32_t(visible?key:0),"actual raw gather IDs violate cold chronological plan");
        require(visible?mask[at]==0.f:(std::isinf(mask[at])&&mask[at]<0),"actual mask violates cold causal plan");
        half_mask[at]=visible?uint16_t(0):uint16_t(0xfc00);
    }
    require(sha256(ids.data(),ids.size()*4)==p->attention_contract.at("raw_ids_sha256")&&
        sha256(mask.data(),mask.size()*4)==p->attention_contract.at("raw_mask_f32_sha256")&&
        sha256(half_mask.data(),half_mask.size()*2)==p->attention_contract.at("native_mask_f16_sha256"),"actual IDs/F32 mask/native logical F16 mask SHA mismatch");
    p->attention_contract_verified=true;
}
void capture::write(ggml_backend_t backend,const ggml_tensor *kv_weight,const ggml_tensor *attn_norm,const ggml_tensor *kv_norm) {
    if(!active())return;
#if defined(_WIN32)
    require(!p->test&&p->sealed,"production capture is not sealed");require(p->uploaded&&p->attention_contract_verified,"attention overrides/contract not complete");physical_root();
    const std::string base="boundary-"+p->options.nonce;
    auto name=[&](const std::string&s){return std::wstring(s.begin(),s.end());};
    file_owner marker;marker.fp=exclusive(name(base+".json")); // Incomplete until all readbacks succeed.
    file_owner output;std::string previous;
    auto records=p->readback(backend,[&](const char *stage_name,const void*data,size_t n){
        if(previous!=stage_name){if(output.fp)output.close();previous=stage_name;output.fp=exclusive(name(base+"."+previous+((previous=="route_ids"||previous=="raw_ids")?".i32":".f32")));}
        require(std::fwrite(data,1,n,output.fp)==n,"boundary data write failed");
    });
    if(output.fp)output.close();
    auto input_digest=[&](const char *name,const std::string &pin){
        auto it=std::find_if(records.begin(),records.end(),[&](const auto&r){return r.stage==name;});
        require(it!=records.end()&&it->sha256==pin,"retained common input differs from admitted CPU bytes");
    };
    input_digest("common_residual",p->options.residual_sha256);
    input_digest("previous_pre",p->options.previous_pre_sha256);
    auto weight_digest=[&](const ggml_tensor *tensor){require(tensor&&tensor->data&&ggml_is_contiguous(tensor),"capture weight is absent/noncontiguous");
        uint64_t n=ggml_nbytes(tensor);require(n<=16ull*1024*1024,"diagnostic weight digest exceeds 16 MiB");
        digest d;char block[65536];for(uint64_t off=0;off<n;){size_t take=size_t(std::min<uint64_t>(sizeof(block),n-off));
            ggml_backend_tensor_get(tensor,block,size_t(off),take);d.update(block,take);off+=take;}return d.finish();};
    const auto kvsha=weight_digest(kv_weight),asha=weight_digest(attn_norm),nsha=weight_digest(kv_norm);
    std::fprintf(marker.fp,"{\"schema\":\"ds4split-producer-boundaries-v2\",\"complete\":true,\"qualification_passed\":false,"
        "\"same_input_replay_complete\":false,\"isolation_replay_complete\":false,\"layer\":0,\"start\":0,\"rows\":32,\"format\":\"typed-little-endian\","
        "\"capture_bytes\":%llu,\"source_base_revision\":\"%s\",\"binary_sha256\":\"%s\",\"config_sha256\":\"%s\",\"variant_manifest_sha256\":\"%s\","
        "\"residual_sha256\":\"%s\",\"previous_pre_sha256\":\"%s\",\"kv_weight_type\":%d,\"kv_weight_sha256\":\"%s\","
        "\"attention_norm_sha256\":\"%s\",\"kv_norm_sha256\":\"%s\",\"cuda_routed_dispatch_observed\":false,\"q8_1_activation_captured\":false,\"stages\":[",
        (unsigned long long)p->extent,p->options.source_base_revision.c_str(),p->options.binary_sha256.c_str(),p->config_sha.c_str(),p->options.variant_manifest_sha256.c_str(),
        p->options.residual_sha256.c_str(),p->options.previous_pre_sha256.c_str(),int(kv_weight->type),kvsha.c_str(),asha.c_str(),nsha.c_str());
    for(size_t i=0;i<records.size();++i){const auto&r=records[i];
        std::fprintf(marker.fp,"%s{\"stage\":\"%s\",\"format\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"bytes\":%llu,\"sha256\":\"%s\",\"has_token0_dim94\":%s,\"token0_dim94_bits\":%u}",
            i?",":"",r.stage.c_str(),r.type==GGML_TYPE_I32?"i32-le":"ieee754-f32-le",(long long)r.ne[0],(long long)r.ne[1],(long long)r.ne[2],(long long)r.ne[3],
            (unsigned long long)r.bytes,r.sha256.c_str(),r.has_token0_dim94?"true":"false",r.token0_dim94_bits);}
    require(std::fprintf(marker.fp,"],\"nonce\":\"%s\"}\n",p->options.nonce.c_str())>0&&!std::ferror(marker.fp),"capture completion receipt failed");marker.close();
    json seeds=json::array();for(const auto&item:p->seeds){const auto&s=item.second;
        const auto it=std::find_if(records.begin(),records.end(),[&](const auto&r){return r.stage==s.name;});
        require(it!=records.end()&&it->sha256==s.sha,"actual replay seed readback SHA mismatch");
        seeds.push_back({{"stage",s.name},{"sha256",s.sha},{"bytes",s.data.size()}});}
    const auto proof=json{{"schema","ds4split-isolated-replay-proof-v1"},{"arm",p->options.replay_arm},{"complete",true},
        {"qualification_passed",false},{"input_manifest_sha256",p->options.replay_inputs_sha256},
        {"weight_proof_sha256",p->options.weight_proof_sha256},{"model_identity",p->weight_proof.at("model_identity")},
        {"attention_contract",p->attention_contract},{"attention_contract_sha256",p->options.attention_contract_sha256},{"observer_enabled",0},{"override_host_bytes",p->override_host_bytes},{"source_read_buffer_bytes",65536},
        {"source_read_budget_bytes",4ull*1024*1024*1024},{"source_bytes_read",p->source_bytes_read},
        {"seeds",seeds},{"weights",p->weight_records}}.dump();
    file_owner replay;replay.fp=exclusive(name(base+".replay-proof.json"));require(std::fwrite(proof.data(),1,proof.size(),replay.fp)==proof.size(),"replay proof write");replay.close();

    std::fprintf(stderr,"BOUNDARY_CAPTURE_COMPLETE rows=32 bytes=%llu same_input_replay_complete=false qualification_passed=false\n",(unsigned long long)p->extent);
#else
    require(false,"production capture output requires Windows");
#endif
}
scope::scope(capture *value):previous(current){current=value&&value->active()?value:nullptr;}
scope::~scope(){current=previous;}
void stage(ggml_tensor *tensor,const char *name){if(current)current->add(tensor,name);}
ggml_tensor*replacement(ggml_context*ctx,ggml_tensor*tensor,const char*tag){return current?current->replacement(ctx,tensor,tag):tensor;}
#ifdef DSV41_BOUNDARY_CAPTURE_TEST
std::unique_ptr<capture> capture::for_test(uint32_t rows,uint64_t cap) {
    require(rows&&rows<=32&&cap&&cap<=maximum_capture_bytes,"invalid bounded test plan");
    auto p=std::make_unique<implementation>();p->enabled=true;p->test=true;p->rows=rows;p->options.capture_bytes=cap;
    return std::unique_ptr<capture>(new capture(std::move(p)));
}
std::unique_ptr<capture> capture::for_replay_test(const std::string&config_text,const std::string&inputs_text,
        const std::string&proof_text,const std::function<std::string(const std::string&,size_t)>&loader){
    auto p=std::make_unique<implementation>();p->options=parse_config(config_text);p->enabled=true;p->test=true;p->rows=32;
    auto result=std::unique_ptr<capture>(new capture(std::move(p)));result->load_replay(inputs_text,proof_text,loader);return result;
}
std::vector<record> capture::read_for_test(ggml_backend_t backend,const block_sink &sink){
    require(p->test,"test readback disabled for production capture");return p->readback(backend,sink);
}
#endif
}
