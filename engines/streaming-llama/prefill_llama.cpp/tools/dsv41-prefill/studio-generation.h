#pragma once
// Reuse the native phase tracker and the Studio/native ds4 sampler. No CUDA
// state is serialized; the side channel contains only a few sampling numbers.
#include "../../../../ds4-win/ds4_quality_phase.h"
#include <fstream>
#include <cmath>
extern "C" int slstack_sample_logits(const float*,int,float,int,float,float,uint64_t*);

struct studio_generation {
    using json=nlohmann::json;
    bool configured=false,active=false;
    ds4_quality_phase_tracker tracker{};
    int think_open=0,think_close=0,dsml=0;
    uint64_t rng=1,reported_revision=0;
    int reported_phase=-1;
    std::vector<std::string> vocabulary;
    std::filesystem::path control;
    json settings,overrides;
    std::chrono::steady_clock::time_point unreadable{};

    static void check(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
    static double number(const json&j,const char*key,double lo,double hi,bool integer=false){
        check(j.contains(key)&&j[key].is_number(),"Missing sampler number");
        const double v=j[key].get<double>();
        check(std::isfinite(v)&&v>=lo&&v<=hi&&(!integer||std::floor(v)==v),"Invalid sampler number");return v;
    }
    static void sampler(const json&j,bool phase){
        check(j.is_object()&&j.size()==size_t(phase?5:4),"Invalid sampler fields");
        number(j,"temperature",0,2);number(j,"top_p",.001,1);number(j,"min_p",0,1);number(j,"top_k",0,1024,true);
        if(phase)number(j,"bias",0,1.36125);
    }
    static void validate(const json&j){
        check(j.is_object()&&j.size()==5&&j.contains("revision")&&j["revision"].is_number_unsigned()&&j["revision"].get<uint64_t>()>0,
              "Invalid generation revision");
        check(j.contains("per_phase")&&j["per_phase"].is_boolean(),"Invalid phase switch");
        number(j,"value",0,1.36125);sampler(j.at("sampler"),false);
        check(j.at("phases").is_object()&&j.at("phases").size()==4,"Invalid phase map");
        for(const char*p:ds4_quality_phase_names)sampler(j.at("phases").at(p),true);
    }
    void refresh(){
        if(!configured)return;
        try{
            // Share delete permits atomic replace while generation is reading.
            HANDLE f=CreateFileW(control.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
            check(f!=INVALID_HANDLE_VALUE,"Cannot read live generation control");
            char bytes[16384];DWORD got=0;const DWORD size=GetFileSize(f,nullptr);
            const bool ok=size>0&&size<sizeof(bytes)&&ReadFile(f,bytes,size,&got,nullptr)&&got==size;CloseHandle(f);
            check(ok,"Live generation control exceeds 16 KiB or is truncated");
            auto next=json::parse(bytes,bytes+got);validate(next);
            if(settings.is_null()||next.at("revision")>=settings.at("revision"))settings=std::move(next);
            unreadable={};
        }catch(...){
            if(settings.is_null())throw;
            const auto now=std::chrono::steady_clock::now();
            if(unreadable==std::chrono::steady_clock::time_point{})unreadable=now;
            if(now-unreadable>std::chrono::seconds(30))throw;
            // A short file replacement/antivirus race keeps the last complete
            // validated revision; retry at the very next token boundary.
        }
    }
    void begin(const json&j){
        check(j.is_object()&&j.size()==8,"Invalid generation command");
        auto path=std::filesystem::u8path(j.at("vocabulary_file").get<std::string>());
        if(vocabulary.empty()){
            std::ifstream f(path,std::ios::binary);check(bool(f),"Cannot read generation vocabulary");
            f.seekg(0,std::ios::end);const auto extent=f.tellg();check(extent>0&&extent<=16*1024*1024,"Vocabulary exceeds 16 MiB");f.seekg(0);
            uint32_t count=0;f.read(reinterpret_cast<char*>(&count),4);check(count==129280,"Wrong generation vocabulary");
            vocabulary.resize(count);
            for(auto&s:vocabulary){uint32_t n=0;f.read(reinterpret_cast<char*>(&n),4);check(bool(f)&&n<=65536,"Invalid vocabulary token");
                s.resize(n);f.read(s.data(),n);check(bool(f),"Truncated vocabulary");}
            check(f.peek()==EOF,"Trailing vocabulary data");
        }
        think_open=j.at("think_open").get<int>();think_close=j.at("think_close").get<int>();dsml=j.at("dsml").get<int>();
        check(think_open>=0&&think_open<129280&&think_close>=0&&think_close<129280&&dsml>=0&&dsml<129280,"Invalid phase token IDs");
        tracker={};tracker.reasoning=j.at("reasoning").get<bool>();rng=j.at("seed").get<uint64_t>();overrides=j.at("overrides");
        check(overrides.is_object()&&overrides.size()<=4,"Invalid sampler overrides");
        for(auto it=overrides.begin();it!=overrides.end();++it){
            const auto&k=it.key();check(k=="temperature"||k=="top_p"||k=="top_k"||k=="min_p","Unknown sampler override");
        }
        control=std::filesystem::u8path(j.at("control_file").get<std::string>());
        configured=true;reported_revision=0;reported_phase=-1;settings=nullptr;refresh();
    }
    int phase()const{return ds4_quality_phase_current(&tracker);}
    json knobs()const{
        json out=settings.at("per_phase").get<bool>()?settings.at("phases").at(ds4_quality_phase_names[phase()]):settings.at("sampler");
        if(!out.contains("bias"))out["bias"]=settings.at("value");
        for(auto it=overrides.begin();it!=overrides.end();++it)out[it.key()]=it.value();
        sampler(out,true);return out;
    }
    float bias()const{return knobs().at("bias").get<float>();}
    void feed(int token){
        check(token>=0&&token<int(vocabulary.size()),"Generated token outside vocabulary");
        const auto&s=vocabulary[size_t(token)];
        ds4_quality_phase_feed(&tracker,token,think_open,think_close,dsml,s.data(),s.size());
    }
    void report(uint32_t position,const json&k){
        const auto revision=settings.at("revision").get<uint64_t>();
        if(revision==reported_revision&&phase()==reported_phase)return;
        reported_revision=revision;reported_phase=phase();
        std::fprintf(stderr,"LLAMADECODE_GENERATION revision=%llu phase=%s bias=%.6g temperature=%.6g top_p=%.6g top_k=%d min_p=%.6g position=%u\n",
            (unsigned long long)revision,ds4_quality_phase_names[phase()],k.at("bias").get<double>(),k.at("temperature").get<double>(),
            k.at("top_p").get<double>(),k.at("top_k").get<int>(),k.at("min_p").get<double>(),position);std::fflush(stderr);
        auto state=k;state["revision"]=revision;state["phase"]=ds4_quality_phase_names[phase()];state["position"]=position;
        auto dst=control;dst+=L".applied.json";auto tmp=dst;tmp+=L".tmp";
        {std::ofstream f(tmp,std::ios::binary|std::ios::trunc);f<<state.dump();}
        MoveFileExW(tmp.c_str(),dst.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH);
    }
    int sample(const std::vector<float>&logits,uint32_t position){
        refresh();const auto k=knobs();report(position,k);
        return slstack_sample_logits(logits.data(),int(logits.size()),k.at("temperature").get<float>(),
            k.at("top_k").get<int>(),k.at("top_p").get<float>(),k.at("min_p").get<float>(),&rng);
    }
};
