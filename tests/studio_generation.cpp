// CPU regression for the actual native sampler/phase/live-publication path.
#define NOMINMAX
#include <windows.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include "../engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill/studio-generation.h"

static int checks=0;
static void check(bool ok){++checks;if(!ok)throw std::runtime_error("Native generation regression check "+std::to_string(checks));}
int main(int argc,char**argv)try{
    check(argc==2);std::filesystem::path root=std::filesystem::u8path(argv[1]);
    std::filesystem::create_directories(root);
    const auto vocab=root/"vocabulary.bin",control=root/"live.json";
    {std::ofstream f(vocab,std::ios::binary);uint32_t count=129280;f.write(reinterpret_cast<const char*>(&count),4);
        for(uint32_t i=0;i<count;i++){std::string s=i==13?"<":i==14?" calls>":i==15?"</":i==16?"```":i==17?"\n":i==18?"</think>":"x";
            uint32_t n=uint32_t(s.size());f.write(reinterpret_cast<const char*>(&n),4);f<<s;}}
    using json=nlohmann::json;
    json row={{"revision",1},{"per_phase",true},{"value",.35},{"sampler",{{"temperature",.6},{"top_p",.95},{"top_k",0},{"min_p",0}}}};
    row["phases"]=json::object();
    for(const char*p:ds4_quality_phase_names){row["phases"][p]=row["sampler"];row["phases"][p]["bias"]=.35;}
    auto publish=[&](){auto tmp=control;tmp+=L".tmp";{std::ofstream f(tmp);f<<row.dump();}check(MoveFileExW(tmp.c_str(),control.c_str(),MOVEFILE_REPLACE_EXISTING)!=0);};
    publish();studio_generation g;
    g.begin({{"control_file",control.u8string()},{"vocabulary_file",vocab.u8string()},{"reasoning",true},
        {"think_open",10},{"think_close",11},{"dsml",12},{"seed",42},{"overrides",json::object()}});
    check(g.phase()==DS4_QUALITY_REASONING);g.feed(18);check(g.phase()==DS4_QUALITY_REASONING); // literal spelling is not a special ID
    g.feed(11);check(g.phase()==DS4_QUALITY_ANSWER);
    g.feed(13);g.feed(12);g.feed(14);check(g.phase()==DS4_QUALITY_TOOL_CALL);
    g.feed(15);g.feed(12);g.feed(14);check(g.phase()==DS4_QUALITY_ANSWER);
    g.feed(17);g.feed(16);g.feed(17);check(g.phase()==DS4_QUALITY_CODE);g.feed(16);g.feed(17);check(g.phase()==DS4_QUALITY_ANSWER);
    const float logits[]={3,3,2};uint64_t a=42,b=42;
    check(slstack_sample_logits(logits,3,0,0,1,0,&a)==0); // stable greedy ties
    const float varied[]={1,3,2};bool different=false;
    for(int i=0;i<100;i++){
        check(slstack_sample_logits(varied,3,1,1,.95,0,&a)==1);
        uint64_t first=uint64_t(i+1),second=first;
        int x=slstack_sample_logits(varied,3,2,0,1,0,&first),y=slstack_sample_logits(varied,3,2,0,1,0,&second);
        check(x==y&&first==second);if(x!=1)different=true;
        check(slstack_sample_logits(varied,3,1,0,.01,0,&b)==1);
        check(slstack_sample_logits(varied,3,1,0,1,1,&b)==1);
    }
    check(different);
    g.sample({1,3,2},1);
    row["revision"]=2;row["per_phase"]=false;row["value"]=1.36125;row["sampler"]["temperature"]=0;publish();
    check(g.sample({1,3,2},2)==1);check(g.bias()>1.36f&&g.settings["revision"]==2);
    // Partial/unreadable publication never replaces the last validated state.
    {std::ofstream f(control);f<<"{";}g.refresh();check(g.settings["revision"]==2);
    std::puts("PASS: real native phase boundaries, seeded/top-k/top-p/min-p sampling, next-token revision, atomic-publication retry");
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
