#include "llama-dsv41-artifact.h"
#include <algorithm>
#include <array>
#include <cstring>

namespace llama_dsv41 {
namespace {
constexpr size_t artifact_header_bytes=204;
constexpr size_t vocabulary=129280;
constexpr uint32_t flags=LLAMA_KV_ARTIFACT_FLAG_DELTA|LLAMA_KV_ARTIFACT_FLAG_NO_CRC|LLAMA_KV_ARTIFACT_FLAG_NATIVE_DSV41;
bool fail(std::string&e,const char*why){e=why;return false;}
size_t native_header_bytes(const wire_frontier&f){return f.mode==3?60:52;}
bool same(const state_span&a,const state_span&b){return a.kind==b.kind&&a.index==b.index&&a.bytes==b.bytes;}
bool native_header(const uint32_t*h,const wire_frontier&f){
    const std::array<uint32_t,13> expected{{0x34565344,2,f.context,f.mode,128,128,f.context+1,f.position,40,512,128,129280,0x413431}};
    return std::equal(expected.begin(),expected.end(),h)&&
        (f.mode!=3||(h[13]==f.capture_end&&h[14]==f.capture_length));
}
bool read_bounded(const artifact_read&source,uint64_t off,void*dst,size_t bytes,std::string&error){
    auto*out=static_cast<uint8_t*>(dst);
    while(bytes){const size_t n=std::min<size_t>(65536,bytes);
        if(!source(off,out,n,error))return false;off+=n;out+=n;bytes-=n;}
    return true;
}
}

bool delta_artifact_sender::init(const wire_frontier&base,const wire_frontier&target,
        uint64_t session,uint64_t model,wire_payload_reader&input,std::string&error){
    ready=false;total=0;header.clear();ranges.clear();source=nullptr;error.clear();
    delta_plan delta;wire_plan full;
    if(!session||!model)return fail(error,"Native delta artifact requires session/model routing identity");
    if(!make_delta_plan(base,target,delta,error)||!make_wire_plan(target,full,error))return false;
    if(delta.token_count>49152||input.bytes()!=full.bytes)return fail(error,"Native delta artifact exceeds chunk or source extent");
    const size_t nh=native_header_bytes(target);std::array<uint32_t,15> h{};
    if(!input.read(0,h.data(),nh,error))return false;
    if(!native_header(h.data(),target))return fail(error,"Native delta source header differs from declared target");
    uint64_t body=nh+uint64_t(delta.token_count)*4+vocabulary*4+delta.state_bytes;
    llama_kv_artifact_metadata meta;
    meta.format_major=1;meta.format_minor=10;meta.n_layers=40;meta.n_ctx=target.context;
    meta.token_count=target.position;meta.type_k=0;meta.type_v=0;meta.flags=flags;
    meta.model_fingerprint=model;meta.pd_session_id=session;meta.source_slot_id=0;
    meta.base_prefix_cells=base.position;
    if(!llama_kv_artifact_serialize_header(header,meta,body,0,0)||header.size()!=artifact_header_bytes)
        return fail(error,"Existing artifact v2.1 header extent changed");
    total=header.size();
    auto add=[&](uint64_t off,uint64_t bytes){if(bytes){ranges.push_back({total,off,bytes});total+=bytes;}};
    add(0,nh);add(nh+uint64_t(base.position)*4,uint64_t(delta.token_count)*4);
    add(nh+uint64_t(target.position)*4,vocabulary*4);
    uint64_t off=nh+uint64_t(target.position)*4+vocabulary*4;
    for(const auto&s:full.spans){
        for(const auto&r:delta.slices)if(same(s,r.span))add(off+r.offset,r.bytes);
        off+=s.bytes;
    }
    if(off!=input.bytes()||total!=header.size()+body||ranges.size()>99)
        return fail(error,"Native delta artifact coverage differs from its canonical plan");
    source=&input;ready=true;return true;
}

bool delta_artifact_sender::read(uint64_t offset,void*dst,size_t count,std::string&error){
    error.clear();
    if(!ready||!dst||!count||count>16*1024*1024||offset>total||count>total-offset)
        return fail(error,"Native delta artifact read outside immutable lease");
    auto*out=static_cast<uint8_t*>(dst);
    if(offset<header.size()){
        const size_t n=std::min<size_t>(count,header.size()-size_t(offset));
        std::memcpy(out,header.data()+offset,n);out+=n;offset+=n;count-=n;
    }
    for(const auto&r:ranges){
        if(!count)break;
        if(offset<r.offset||offset>=r.offset+r.bytes)continue;
        while(count&&offset<r.offset+r.bytes){
            const size_t n=size_t(std::min<uint64_t>(65536,std::min<uint64_t>(count,r.offset+r.bytes-offset)));
            if(!source->read(r.source+offset-r.offset,out,n,error)){ready=false;return false;}
            out+=n;offset+=n;count-=n;
        }
    }
    if(count){ready=false;return fail(error,"Native delta artifact range coverage incomplete");}
    return true;
}

bool delta_artifact_receiver::init(uint64_t bytes,const artifact_read&input,const wire_frontier&base,
        uint64_t session,uint64_t model,std::string&error,uint32_t token_limit){
    ready=false;delta={};offsets.clear();new_tokens.clear();new_logits.clear();source={};error.clear();
    if(!input||!session||!model||!token_limit||token_limit>49152||bytes<artifact_header_bytes+52)
        return fail(error,"Native delta receiver requires explicit bounded stream/session/model");
    std::array<uint8_t,artifact_header_bytes> header{};
    if(!input(0,header.data(),header.size(),error))return false;
    llama_kv_artifact_metadata meta;llama_kv_artifact_summary summary;size_t consumed=0;
    if(!llama_kv_artifact_parse_header(header.data(),header.size(),meta,summary,consumed,&error))return false;
    if(consumed!=header.size()||meta.format_major!=1||meta.format_minor!=10||meta.n_layers!=40||
        meta.n_ctx!=base.context||meta.flags!=flags||meta.base_prefix_cells!=base.position||
        meta.model_fingerprint!=model||meta.pd_session_id!=session||meta.source_slot_id!=0||
        meta.type_k||meta.type_v||meta.v_trans||meta.is_mla||meta.reserved_0||meta.last_token_id!=-1||
        meta.prompt_token_count||meta.prompt_text_bytes||meta.streamed_prefix_token_count||
        meta.img_cell_count||meta.img_n_embd||meta.img_pos_start||meta.mtp_hidden_n_tokens||meta.mtp_hidden_n_embd||
        summary.payload_crc32||summary.payload_bytes!=bytes-header.size())
        return fail(error,"Native delta artifact routing/geometry/flags mismatch");
    llama_kv_artifact_metadata canonical;
    canonical.format_major=1;canonical.format_minor=10;canonical.n_layers=40;
    canonical.n_ctx=base.context;canonical.token_count=meta.token_count;canonical.flags=flags;
    canonical.model_fingerprint=model;canonical.pd_session_id=session;canonical.source_slot_id=0;
    canonical.base_prefix_cells=base.position;
    std::vector<uint8_t> canonical_header;
    if(!llama_kv_artifact_serialize_header(canonical_header,canonical,summary.payload_bytes,0,0)||
        canonical_header.size()!=header.size()||!std::equal(header.begin(),header.end(),canonical_header.begin()))
        return fail(error,"Native delta artifact has noncanonical unused header fields");
    const size_t nh=native_header_bytes(base);std::array<uint32_t,15> h{};
    if(!input(header.size(),h.data(),nh,error))return false;
    const wire_frontier target{h[2],h[7],h[3],base.mode==3?h[13]:0u,base.mode==3?h[14]:0u};
    if(!native_header(h.data(),target)||meta.token_count!=target.position||
        !make_delta_plan(base,target,delta,error))return fail(error,"Native delta target header differs from artifact/base");
    if(delta.token_count>token_limit)return fail(error,"Native delta token extent exceeds explicit operation cap");
    const uint64_t expected=header.size()+nh+uint64_t(delta.token_count)*4+vocabulary*4+delta.state_bytes;
    if(expected!=bytes)return fail(error,"Native delta payload extent/trailing bytes mismatch");
    new_tokens.resize(delta.token_count);new_logits.resize(vocabulary);
    uint64_t off=header.size()+nh;
    if(!read_bounded(input,off,new_tokens.data(),new_tokens.size()*4,error))return false;off+=new_tokens.size()*4;
    if(!read_bounded(input,off,new_logits.data(),new_logits.size()*4,error))return false;off+=new_logits.size()*4;
    if(!std::all_of(new_tokens.begin(),new_tokens.end(),[](int32_t t){return t>=0&&uint32_t(t)<vocabulary;})||
        !std::all_of(new_logits.begin(),new_logits.end(),[](float x){uint32_t bits;std::memcpy(&bits,&x,4);return(bits&0x7f800000u)!=0x7f800000u;}))
        return fail(error,"Native delta contains invalid token IDs or nonfinite logits");
    for(const auto&r:delta.slices){offsets.push_back(off);off+=r.bytes;}
    if(off!=bytes)return fail(error,"Native delta state coverage mismatch");
    source=input;ready=true;return true;
}

bool delta_artifact_receiver::read_span(const state_span&span,uint64_t offset,void*dst,size_t count,std::string&error){
    error.clear();
    if(!ready||!dst||!count||count>65536||offset>span.bytes||count>span.bytes-offset)
        return fail(error,"Native delta state read exceeds bounded slice");
    for(size_t i=0;i<delta.slices.size();++i){const auto&r=delta.slices[i];
        if(same(span,r.span)&&offset>=r.offset&&offset-r.offset<=r.bytes&&count<=r.bytes-(offset-r.offset)){
            if(source(offsets[i]+offset-r.offset,dst,count,error))return true;
            ready=false;return false;
        }
    }
    return fail(error,"Native delta state read requests unsent base bytes");
}
}
