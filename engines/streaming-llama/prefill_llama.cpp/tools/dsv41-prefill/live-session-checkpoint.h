#pragma once
#include "llama-dsv41-wire.h"
#include "boundary-capture.h"
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <set>
#include <stdexcept>

// Metadata for the existing native payload and artifact-delta adapter. This
// does not own/cache a second copy of native KV or define another KV format.
namespace dsv41_live {
using json=nlohmann::json;
using namespace llama_dsv41;
inline void need(bool value,const std::string&why){if(!value)throw std::runtime_error(why);}
inline bool sha_valid(const std::string&s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});}
inline uint64_t u64(const json&v){need(v.is_number_unsigned(),"Session integer must be exact unsigned");return v.get<uint64_t>();}
inline void keys(const json&j,std::initializer_list<const char*> names){
    need(j.is_object()&&j.size()==names.size(),"Session object has unknown/missing fields");
    for(auto n:names)need(j.contains(n),"Session object is missing a field");
}
inline json parse_control(const std::string&s){
    need(!s.empty()&&s.size()<=1024*1024,"Session control exceeds1MiB");
    std::vector<std::set<std::string>>objects;size_t nodes=0;
    auto cb=[&](int depth,json::parse_event_t event,json&value){
        need(depth<=16&&++nodes<=200000,"Session JSON nesting/node bound");
        if(event==json::parse_event_t::object_start)objects.emplace_back();
        if(event==json::parse_event_t::key){need(!objects.empty(),"Session JSON key scope");need(objects.back().insert(value.get<std::string>()).second,"Duplicate session JSON key");}
        if(event==json::parse_event_t::object_end){need(!objects.empty(),"Session JSON object scope");objects.pop_back();}
        return true;
    };
    auto result=json::parse(s,cb);need(result.is_object(),"Session control must be an object");return result;
}
struct checkpoint {
    wire_frontier frontier;
    std::string tokens_sha256,logits_sha256;
    std::vector<std::string> owners;
    json describe()const{return {{"context",frontier.context},{"position",frontier.position},{"mode",frontier.mode},
        {"capture_end",frontier.capture_end},{"capture_length",frontier.capture_length},
        {"tokens_sha256",tokens_sha256},{"logits_sha256",logits_sha256},{"owner_sha256",owners}};}
    static checkpoint from_json(const json&j){
        keys(j,{"context","position","mode","capture_end","capture_length","tokens_sha256","logits_sha256","owner_sha256"});
        checkpoint out;auto word=[&](const char*k){auto v=u64(j.at(k));need(v<=UINT32_MAX,"Session frontier overflow");return uint32_t(v);};
        out.frontier={word("context"),word("position"),word("mode"),word("capture_end"),word("capture_length")};
        wire_plan p;std::string error;need(make_wire_plan(out.frontier,p,error),error);
        need(out.frontier.mode==3&&out.frontier.capture_end==out.frontier.position&&out.frontier.capture_length==std::min(out.frontier.position,128u),"Session requires completed DSpark mode3");
        out.tokens_sha256=j.at("tokens_sha256").get<std::string>();out.logits_sha256=j.at("logits_sha256").get<std::string>();
        need(sha_valid(out.tokens_sha256)&&sha_valid(out.logits_sha256),"Session token/logit SHA malformed");
        need(j.at("owner_sha256").is_array()&&j.at("owner_sha256").size()==p.spans.size(),"Session owner hash extent");
        for(const auto&h:j.at("owner_sha256")){auto s=h.get<std::string>();need(sha_valid(s),"Session owner SHA malformed");out.owners.push_back(s);}
        return out;
    }
};
inline checkpoint capture(wire_payload_reader&reader,const std::function<bool()>&stop){
    std::array<uint32_t,15>h{};std::string error;need(reader.read(0,h.data(),60,error),error);
    need(h[0]==0x34565344&&h[1]==2&&h[3]==3&&h[4]==128&&h[5]==128&&h[6]==h[2]+1&&h[8]==40&&h[9]==512&&h[10]==128&&h[11]==129280&&h[12]==0x413431,"Session native header mismatch");
    checkpoint out;out.frontier={h[2],h[7],h[3],h[13],h[14]};wire_plan p;
    need(make_wire_plan(out.frontier,p,error)&&p.bytes==reader.bytes(),"Session native length/plan mismatch");
    auto digest=[&](uint64_t offset,uint64_t bytes){return dsv41_boundary::sha256_stream(bytes,[&](uint64_t off,void*dst,size_t n){
        if(stop&&stop())return false;
        return reader.read(offset+off,dst,n,error);
    });};
    uint64_t offset=60;out.tokens_sha256=digest(offset,uint64_t(h[7])*4);offset+=uint64_t(h[7])*4;
    out.logits_sha256=digest(offset,129280*4);offset+=129280*4;
    for(const auto&s:p.spans){out.owners.push_back(digest(offset,s.bytes));offset+=s.bytes;}
    need(offset==reader.bytes(),"Session checkpoint coverage");return checkpoint::from_json(out.describe());
}
class sequence_guard {
    std::string session,model;uint64_t next=1;
public:
    sequence_guard(std::string session,std::string model):session(std::move(session)),model(std::move(model)){
        need(this->session.size()==32&&sha_valid(this->session+this->session)&&sha_valid(this->model),"Session/model identity malformed");
    }
    void accept(const json&command){
        need(command.is_object()&&command.at("session")==session&&command.at("model_sha256")==model,"Session/model identity mismatch");
        need(u64(command.at("sequence"))==next&&next!=UINT64_MAX,"Replayed/out-of-order session command");++next;
    }
    uint64_t expected()const{return next;}
};
}
