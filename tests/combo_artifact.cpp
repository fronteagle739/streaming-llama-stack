#include "llama-dsv41-artifact.h"
#include "dsv41-native-c-artifact.h"
#include "live-session-checkpoint.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" int combo_plan(uint32_t,uint32_t,uint32_t,uint32_t,dsv41_c_plan*);
extern "C" int combo_write(const dsv41_c_plan*,const int32_t*,const float*,dsv41_c_read_span,void*,dsv41_c_io,void*);
extern "C" int combo_read(uint32_t,uint32_t,uint32_t,uint64_t,dsv41_c_io,void*,int32_t*,float*,dsv41_c_plan*);
static void fill(uint32_t kind,uint32_t index,uint64_t offset,void*dst,size_t n){
    auto*p=static_cast<uint8_t*>(dst);for(size_t i=0;i<n;++i)p[i]=uint8_t((offset+i)*13+kind*37+index*17);
}
struct packet{std::vector<uint8_t>bytes;size_t cursor=0;};
static int append(void*ud,void*data,size_t n){auto*p=static_cast<packet*>(ud);assert(n<=65536);auto*b=static_cast<uint8_t*>(data);p->bytes.insert(p->bytes.end(),b,b+n);return 1;}
static int pull(void*ud,void*data,size_t n){auto*p=static_cast<packet*>(ud);assert(n<=65536);if(p->cursor>p->bytes.size()||n>p->bytes.size()-p->cursor)return 0;std::memcpy(data,p->bytes.data()+p->cursor,n);p->cursor+=n;return 1;}
static int read_c(void*ud,uint32_t owner,uint64_t off,void*dst,size_t n){auto*p=static_cast<dsv41_c_plan*>(ud);if(owner>=p->span_count||off>p->spans[owner].bytes||n>p->spans[owner].bytes-off||n>65536)return 0;const auto&s=p->spans[owner];fill(s.kind,s.index,off,dst,n);return 1;}
int main(){
    unsigned count=0;
    for(uint32_t mode:{1u,3u})for(auto range:{std::pair<uint32_t,uint32_t>{1,2},{31,32},{32,64},{127,129},{128,130},{255,258}}){
        uint32_t b=range.first,t=range.second;
        llama_dsv41::wire_frontier base{262144,b,mode,mode==3?b:0,mode==3?std::min(b,128u):0};
        llama_dsv41::wire_frontier target{262144,t,mode,mode==3?t:0,mode==3?std::min(t,128u):0};
        std::vector<int32_t>tokens(t);for(uint32_t i=0;i<t;++i)tokens[i]=int32_t(i);
        std::vector<float>logits(129280,.25f);std::string error;
        llama_dsv41::wire_payload_reader source;
        assert(source.init(target,tokens.data(),tokens.size(),logits.data(),logits.size(),
            [](const llama_dsv41::state_span&s,uint64_t off,void*dst,size_t n){fill(uint32_t(s.kind),s.index,off,dst,n);return true;},error));
        auto captured=dsv41_live::capture(source,{});
        assert(captured.frontier.mode==mode&&captured.frontier.position==t&&captured.owners.size()==(t%2?54:48)+(mode==3));
        assert(captured.tokens_sha256==dsv41_boundary::sha256(tokens.data(),tokens.size()*4));
        llama_dsv41::delta_artifact_sender sender;assert(sender.init(base,target,17,23,source,error));
        std::vector<uint8_t>expected(size_t(sender.bytes()));
        for(size_t off=0;off<expected.size();){size_t n=std::min<size_t>(65536,expected.size()-off);assert(sender.read(off,expected.data()+off,n,error));off+=n;}
        dsv41_c_plan c{};assert(combo_plan(mode,262144,b,t,&c));packet actual;
        assert(combo_write(&c,tokens.data()+b,logits.data(),read_c,&c,append,&actual));
        assert(actual.bytes==expected);
        std::vector<int32_t>received(t-b);std::vector<float>received_logits(129280);dsv41_c_plan received_plan{};
        assert(combo_read(mode,262144,b,actual.bytes.size(),pull,&actual,received.data(),received_logits.data(),&received_plan));
        assert(std::equal(received.begin(),received.end(),tokens.begin()+b)&&received_logits==logits);
        assert(actual.cursor==204+(mode==3?60:52)+(t-b)*4+129280*4);
        assert(received_plan.state_bytes==c.state_bytes);
        actual.bytes[52]^=1;actual.cursor=0;
        assert(!combo_read(mode,262144,b,actual.bytes.size(),pull,&actual,received.data(),received_logits.data(),&received_plan));
        assert(received_plan.slice_count==0);
        ++count;
    }
    dsv41_live::sequence_guard guard(std::string(32,'1'),std::string(64,'a'),42);
    dsv41_live::json command={{"session",std::string(32,'1')},{"model_sha256",std::string(64,'a')},{"sequence",1u},{"operation","checkpoint"}};
    bool refused=false;try{guard.accept(command);}catch(const std::exception&){refused=true;}
    assert(refused&&guard.expected()==1);
    command["owner_epoch"]=43u;refused=false;try{guard.accept(command);}catch(const std::exception&){refused=true;}
    assert(refused&&guard.expected()==1);
    command["owner_epoch"]=42.0;refused=false;try{guard.accept(command);}catch(const std::exception&){refused=true;}
    assert(refused&&guard.expected()==1);
    command["owner_epoch"]=42u;guard.accept(command);assert(guard.expected()==2);
    refused=false;try{guard.accept(command);}catch(const std::exception&){refused=true;}
    assert(refused&&guard.expected()==2);
    std::printf("PASS: %u complete C/C++ mode1/mode3 artifact packets byte-exact; target checkpoint headers/hashes and corrupted-model refusal\n",count);
    std::puts("PASS: owner epoch is exact, mandatory and replay protected without consuming refused sequences");
}
