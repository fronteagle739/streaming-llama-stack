// Included by the Windows producer after its bounded native-file/hash helpers.
#include "live-session-checkpoint.h"
#include "llama-dsv41-artifact.h"
#include <thread>
#include <io.h>

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

void run_live_session(std::unique_ptr<llama_dsv41_sweep>&sweep,
        const llama_dsv41_weight_provider::config&weights,const llama_dsv41_sweep::config&options,
        const std::string&session,uint64_t timeout_ms,uint32_t capacity,HANDLE output){
    using namespace llama_dsv41;using dsv41_live::checkpoint;
    require(sweep&&sweep->ready()&&options.fixed&&options.stateful_chunks&&options.capture_dspark,
        "Persistent session requires completed fixed DSpark stateful owner");
    require(capacity&&capacity<=49152&&options.fixed->tokens==capacity,"Session chunk reservation differs from fixed owner");
    session_pipe io(timeout_ms,output);dsv41_live::sequence_guard guard(session,model_sha);
    const uint64_t session_route=std::stoull(session.substr(0,16),nullptr,16),model_route=std::stoull(std::string(model_sha).substr(0,16),nullptr,16);
    require(session_route&&model_route,"Session routing fingerprint must be nonzero");
    checkpoint current,remote;bool have_remote=false,initial_sent=false,awaiting_ack=false;
    std::unique_ptr<llama_dsv41_sweep::rewind_anchor> rewind_anchor;
    checkpoint rewind_identity;
    auto inspect=[&](){checkpoint c;std::string error;
        success(sweep->export_native([&](wire_payload_reader&r,std::string&why){try{c=dsv41_live::capture(r,[]{return should_stop();});return true;}catch(const std::exception&e){why=e.what();return false;}},error),error);return c;};
    current=inspect();
    io.send({{"status","ready"},{"session",session},{"model_sha256",model_sha},{"next_sequence",guard.expected()},
        {"checkpoint",current.describe()},{"maximum_prefill_tokens",capacity},{"maximum_decode_tokens",128},{"control_host_bytes",64*1024*1024},
        {"transport","private owned pipes; existing native payload and artifact v2.1 delta"}});
    for(;;){
        io.renew();const auto command=dsv41_live::parse_control(io.line());guard.accept(command);
        const auto operation=command.at("operation").get<std::string>();const auto sequence=dsv41_live::u64(command.at("sequence"));
        const auto started=clock_type::now();
        auto reply=[&](json extra){extra["status"]="complete";extra["sequence"]=sequence;extra["session"]=session;extra["model_sha256"]=model_sha;extra["checkpoint"]=current.describe();io.send(extra);};
        auto same_base=[&](){const auto base=checkpoint::from_json(command.at("base"));require(base.describe()==current.describe(),"Session base token/position/owner hashes differ");};
        if(operation=="close"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation"});reply({{"closed",true}});return;
        }else if(operation=="checkpoint"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation"});
            require(inspect().describe()==current.describe(),"Session idle owner changed");reply({{"peer_synchronized",have_remote&&!awaiting_ack&&remote.describe()==current.describe()}});
        }else if(operation=="save_rewind"||operation=="rewind"){
            dsv41_live::keys(command,operation=="save_rewind"?
                std::initializer_list<const char*>{"session","model_sha256","sequence","operation","base"}:
                std::initializer_list<const char*>{"session","model_sha256","sequence","operation","base","target"});same_base();
            require(have_remote&&!awaiting_ack&&remote.describe()==current.describe(),"Rewind operation before peer synchronization");
            require(inspect().describe()==current.describe(),"Rewind current owner changed");std::string error;
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
        }else if(operation=="export_initial"||operation=="export_delta"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base"});same_base();
            const bool initial=operation=="export_initial";
            require(!awaiting_ack,"Previous state transfer awaits an exact acknowledgement");
            require(initial?(!initial_sent&&!have_remote):(have_remote&&remote.frontier.position<current.frontier.position),"Full resend or delta without an acknowledged base refused");
            std::string error;uint64_t sent=0;std::array<unsigned char,65536>page{};
            success(sweep->export_native([&](wire_payload_reader&r,std::string&why){try{
                require(dsv41_live::capture(r,[]{return should_stop();}).describe()==current.describe(),"Export checkpoint changed");
                delta_artifact_sender delta;if(!initial)success(delta.init(remote.frontier,current.frontier,session_route,model_route,r,why),why);
                const uint64_t bytes=initial?r.bytes():delta.bytes();
                io.send({{"status","payload"},{"sequence",sequence},{"session",session},{"model_sha256",model_sha},
                    {"kind",initial?"native_initial":"artifact_delta_v2_1"},{"bytes",bytes},{"base",initial?json(nullptr):remote.describe()},{"target",current.describe()}});
                while(sent<bytes){const size_t n=size_t(std::min<uint64_t>(page.size(),bytes-sent));
                    success(initial?r.read(sent,page.data(),n,why):delta.read(sent,page.data(),n,why),why);io.write(page.data(),n);sent+=n;}
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
            success(make_delta_plan(current.frontier,target.frontier,plan,error),error);require(plan.token_count<=128,"Live decode delta exceeds128 evaluated tokens");
            const auto bytes=dsv41_live::u64(command.at("bytes"));require(bytes<=64ull*1024*1024,"Live decode artifact exceeds bounded native maximum");
            uint64_t received=0;delta_artifact_receiver incoming;
            auto stream=[&](uint64_t off,void*dst,size_t n,std::string&why){try{
                require(off==received&&n<=65536&&off<=bytes&&n<=bytes-off,"Live decode stream is not sequential/bounded");io.read(dst,n);received+=n;return true;
            }catch(const std::exception&e){why=e.what();return false;}};
            success(incoming.init(bytes,stream,current.frontier,session_route,model_route,error),error);
            require(incoming.plan().target.position==target.frontier.position&&incoming.plan().target.context==target.frontier.context&&
                incoming.plan().target.mode==target.frontier.mode&&incoming.plan().target.capture_end==target.frontier.capture_end&&incoming.plan().target.capture_length==target.frontier.capture_length,
                "Live decode artifact target differs from authenticated metadata");
            llama_dsv41_sweep::decode_delta update;update.target=target.frontier;update.evaluated_tokens=incoming.tokens();update.logits=incoming.logits();
            update.base_tokens_sha256=current.tokens_sha256;update.target_tokens_sha256=target.tokens_sha256;update.target_logits_sha256=target.logits_sha256;
            update.base_owner_sha256=current.owners;update.target_owner_sha256=target.owners;
            success(sweep->apply_decode_delta(update,[&](const state_span&s,uint64_t off,void*dst,size_t n){return incoming.read_span(s,off,dst,n,error);},error),error);
            require(received==bytes,"Live decode artifact was not fully consumed");current=target;remote=target;
            reply({{"peer_synchronized",true},{"received_bytes",received},{"evaluated_tokens",plan.token_count},{"RTX_decode",false},{"apply_ms",ms(started)}});
        }else if(operation=="prefill"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","tokens"});same_base();
            require(have_remote&&!awaiting_ack&&remote.describe()==current.describe(),"Incremental prefill requires exact two-way synchronization");
            require(inspect().describe()==current.describe(),"Incremental prefill actual token/owner state differs from acknowledged checkpoint");
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=capacity,"Live prefill token count exceeds reserved chunk");
            require(ids.size()<uint64_t(options.context)-current.frontier.position,"Live prefill exceeds context capacity");
            std::vector<int32_t>tokens;tokens.reserve(ids.size());for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Live token out of vocabulary");tokens.push_back(int32_t(n));}
            auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
            next_options.carry.tokens=uint32_t(tokens.size());if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=uint32_t(tokens.size());
            std::string error;success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
            const double admission_ms=ms(started);success(sweep->run(tokens,[]{return should_stop();},error),error);const auto timing=sweep->timing();
            current=inspect();require(current.frontier.position==remote.frontier.position+tokens.size(),"Live prefill position did not advance exactly");
            reply({{"peer_synchronized",false},{"prompt_tokens",tokens.size()},{"prompt_processing_ms",admission_ms+timing.prompt_processing_ms},{"export_preparation_ms",timing.export_preparation_ms}});
        }else require(false,"Unknown persistent session operation");
    }
}
}
