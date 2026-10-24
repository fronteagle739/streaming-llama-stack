// Included by the Windows producer after its bounded native-file/hash helpers.
#include "live-session-checkpoint.h"
#include "llama-dsv41-artifact.h"
#include "llama-dsv41-layer-log.h"
#include <thread>
#include <filesystem>
#include <optional>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>

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
}
// DS4COMBO fast switch: the base reader needs session_pipe and is used by resume.
#include "combo-base.inc"
namespace {

using decode_quiet_scope=llama_dsv41_decode_quiet_scope; // per-layer logs opt-in during decode steps
// LLAMADECODE ds4-decode: LLAMADECODE_DS4_DECODE=1 runs each decode step with
// ds4-win's own single-token decode (ds4-decode.dll, built from the stack's
// engines/ds4-win with DS4_EXTERNAL_DECODE) on this owner's canonical state,
// resident weights and B2 expert cache. Prefill stays on the executor.
// LLAMADECODE cache growth: once the first one-token step has sized the
// decode graph, the VRAM freed by the decode-sized plan (minus the decode
// graph, the 1-row hidden bank and a growth margin; capped by the device's
// free memory less a floor) becomes the L2 expert tier. LLAMADECODE_L2=0 disables.
void grow_decode_l2(llama_dsv41_sweep&sweep,const llama_dsv41_fixed_storage*fixed,ggml_backend_t backend){
    static const bool on=[]{const char*e=std::getenv("LLAMADECODE_L2");return !(e&&e[0]=='0'&&!e[1]);}();
    if(!on||!fixed||!fixed->decode_compact)return;
    auto*provider=sweep.provider_for_external_decode();
    if(!provider->decode_layout()||provider->cache_stats().l2_slots)return;
    const uint64_t margin=768ull<<20,floor=1536ull<<20;
    const uint64_t freed=fixed->full_graph_reserve+fixed->full_hidden_bytes;
    const uint64_t now=ggml_gallocr_get_buffer_size(fixed->graph.get(),0)+fixed->hidden_bytes;
    uint64_t offer=freed>now+margin?freed-now-margin:0;
    size_t free_bytes=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(backend),&free_bytes,&total);
    offer=std::min<uint64_t>(offer,free_bytes>floor?free_bytes-floor:0);
    uint32_t slots=0;std::string error;success(provider->grow_decode_cache(offer,&slots,error),error);
    std::fprintf(stderr,"LLAMADECODE_L2_OFFER freed=%llu decode_now=%llu device_free=%llu offered=%llu slots=%u\n",(unsigned long long)freed,
        (unsigned long long)now,(unsigned long long)free_bytes,(unsigned long long)offer,slots);std::fflush(stderr);
}
bool ds4_decode_enabled(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_DS4_DECODE");return e&&e[0]=='1'&&!e[1];}();return on;}
struct ds4_external {
    HMODULE module=nullptr;bool open=false;
    int(*x_open)(const char*,uint32_t,char*,size_t)=nullptr;const void*(*x_map)()=nullptr;
    int(*x_register)(uint64_t,uint64_t,void*)=nullptr;
    int(*x_bind)(void*const*,void*const*,void*const*,const uint32_t*,void*const*,void*const*)=nullptr;
    void(*x_hook)(int(*)(void*,uint32_t,const int32_t*,uint32_t,int32_t*,const char**,const char**,const char**,uint32_t*),void*)=nullptr;
    int(*x_decode)(int,uint32_t,const int32_t*,uint32_t,float*)=nullptr;void(*x_close)()=nullptr;
    llama_dsv41_weight_provider*provider=nullptr;uint64_t registered=0,registered_bytes=0;
    static int expert_hook(void*user,uint32_t layer,const int32_t*ids,uint32_t n,int32_t*slots,const char**g,const char**u,const char**d,uint32_t*count){
        auto*self=static_cast<ds4_external*>(user);std::string why;
        if(!self->provider||!self->provider->external_expert_slots(layer,ids,n,slots,g,u,d,count,nullptr,why)){
            std::fprintf(stderr,"LLAMADECODE_DS4X_SLOTS_FAILED layer=%u reason=%s\n",layer,why.c_str());return 0;}
        return 1;
    }
    template<class F>void bind(F&f,const char*name){f=reinterpret_cast<F>(GetProcAddress(module,name));require(f,"ds4-decode.dll export missing");}
    void load(){
        if(module)return;
        auto dll=executable().parent_path()/L"ds4-decode.dll";
        module=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        require(module,"Cannot load ds4-decode.dll beside the producer");
        bind(x_open,"ds4x_open");bind(x_map,"ds4x_model_map");bind(x_register,"ds4x_register_weight");bind(x_bind,"ds4x_bind_state");
        bind(x_hook,"ds4x_set_expert_hook");bind(x_decode,"ds4x_decode");bind(x_close,"ds4x_close");
    }
    // (Re)prepare on the current owner: decode layout, resident weights
    // registered at their device addresses, state views, expert hook.
    void prepare(llama_dsv41_sweep&sweep,const llama_dsv41_weight_provider::config&weights,uint32_t context){
        load();std::string error;success(sweep.compact_for_external_decode(error),error);
        provider=sweep.provider_for_external_decode();
        if(!provider->decode_layout()){
            if(open){x_close();open=false;}   // banks were rebuilt by a prefill pass: re-register everything
            auto decode_weights=weights;decode_weights.decode_layout=true;
            success(provider->prepare_external_decode(decode_weights,error),error);
        }
        if(!open){
            size_t free_before=0,free_after=0,total=0;auto*dev=ggml_backend_get_device(weights.backend);ggml_backend_dev_memory(dev,&free_before,&total);
            char why[512]{};require(x_open(weights.model_path.c_str(),context,why,sizeof(why))==1,why[0]?why:"ds4x_open failed");open=true;
            ggml_backend_dev_memory(dev,&free_after,&total);
            std::fprintf(stderr,"LLAMADECODE_DS4X_DEVICE_DELTA bytes=%lld free_before=%llu free_after=%llu\n",(long long)free_before-(long long)free_after,
                (unsigned long long)free_before,(unsigned long long)free_after);std::fflush(stderr);
            registered=registered_bytes=0;
            success(provider->decode_resident_tensors([&](uint64_t offset,uint64_t bytes,void*device){
                if(!x_register(offset,bytes,device))return false;++registered;registered_bytes+=bytes;return true;},error),error);
        }
        llama_dsv41_sweep::device_state state;success(sweep.device_state_views(state,error),error);
        require(x_bind(state.raw.data(),state.compressed.data(),state.index.data(),state.rows.data(),state.pool_kv.data(),state.pool_score.data())==1,
            "ds4x_bind_state refused the canonical state");
        x_hook(&expert_hook,this);
        std::fprintf(stderr,"LLAMADECODE_DS4X_READY registered_tensors=%llu registered_bytes=%llu\n",(unsigned long long)registered,(unsigned long long)registered_bytes);
    }
    void step(llama_dsv41_sweep&sweep,ggml_backend_t backend,int32_t token,uint32_t position){
        std::string error;std::array<int32_t,3>tail{};success(sweep.engram_tail(tail,error),error);
        std::vector<float>logits(129280);
        ggml_backend_synchronize(backend);
        const auto before=provider->cache_stats();const auto started=std::chrono::steady_clock::now();
        require(x_decode(token,position,tail.data(),3,logits.data())==1,"ds4-win decode step failed");
        const double decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        success(sweep.commit_external_token(token,logits,error),error);
        const auto after=provider->cache_stats();
        std::fprintf(stderr,"LLAMADECODE_DS4X_TOKEN position=%u decode_ms=%.3f ds_bytes=%llu ds_ms=%.3f l1_hits=%llu l2_hits=%llu misses=%llu l1_slots=%u l2_slots=%u\n",position,decode_ms,
            (unsigned long long)(after.fetched_bytes-before.fetched_bytes),after.fetch_ms-before.fetch_ms,(unsigned long long)(after.hits-before.hits),
            (unsigned long long)(after.l2_hits-before.l2_hits),(unsigned long long)(after.misses-before.misses),after.slots,after.l2_slots);
    }
    // The DLL stays mapped until process exit (never FreeLibrary): no ds4 code may
    // be unmapped under a thread or CUDA callback it started.
    ~ds4_external(){if(open&&x_close){x_hook(nullptr,nullptr);x_close();open=false;std::fprintf(stderr,"LLAMADECODE_DS4X_CLOSED\n");std::fflush(stderr);}}
};
// LLAMADECODE: decode steps use the routed-expert layout (B1). The explicit
// diagnostic LLAMADECODE_DECODE_LAYOUT=0 runs them on the whole-layer B0 layout.
bool routed_decode_layout(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_DECODE_LAYOUT");
    if(e&&e[0]=='0'&&!e[1]){std::fprintf(stderr,"LLAMADECODE_DECODE_LAYOUT=0 whole-layer B0 decode (diagnostic)\n");return false;}return true;}();return on;}
void run_live_session(std::unique_ptr<llama_dsv41_sweep>&sweep,
        const llama_dsv41_weight_provider::config&weights,const llama_dsv41_sweep::config&options,
        const std::string&session,uint64_t timeout_ms,uint32_t capacity,HANDLE output,uint64_t owner_epoch=0){
    using namespace llama_dsv41;using dsv41_live::checkpoint;
    require(sweep&&sweep->ready()&&options.fixed&&options.stateful_chunks,
        "Persistent session requires a completed fixed mode1/3 stateful owner");
    require(capacity&&capacity<=49152&&options.fixed->tokens==capacity,"Session chunk reservation differs from fixed owner");
    session_pipe io(timeout_ms,output);dsv41_live::sequence_guard guard(session,model_sha,owner_epoch);
    auto send=[&](json value){if(owner_epoch)value["owner_epoch"]=owner_epoch;io.send(value);};
    const uint64_t session_route=std::stoull(session.substr(0,16),nullptr,16),model_route=std::stoull(std::string(model_sha).substr(0,16),nullptr,16);
    require(session_route&&model_route,"Session routing fingerprint must be nonzero");
    checkpoint current,remote;bool have_remote=false,initial_sent=false,awaiting_ack=false,device_suspended=false;
    ds4_external ds4x; // LLAMADECODE ds4-decode (LLAMADECODE_DS4_DECODE=1)
    std::unique_ptr<llama_dsv41_sweep::rewind_anchor> rewind_anchor;
    checkpoint rewind_identity;
    auto inspect=[&](){checkpoint c;std::string error;
        success(sweep->export_native([&](wire_payload_reader&r,std::string&why){try{c=dsv41_live::capture(r,[]{return should_stop();});return true;}catch(const std::exception&e){why=e.what();return false;}},error),error);return c;};
    current=inspect();
    send({{"status","ready"},{"session",session},{"model_sha256",model_sha},{"next_sequence",guard.expected()},
        {"checkpoint",current.describe()},{"maximum_prefill_tokens",capacity},{"maximum_decode_tokens",128},{"control_host_bytes",64*1024*1024},
        {"transport","private owned pipes; existing native payload and artifact v2.1 delta"}});
    for(;;){
        io.renew();auto command=dsv41_live::parse_control(io.line());guard.accept(command);if(owner_epoch)command.erase("owner_epoch");
        const auto operation=command.at("operation").get<std::string>();const auto sequence=dsv41_live::u64(command.at("sequence"));
        const auto started=clock_type::now();
        auto reply=[&](json extra){extra["status"]="complete";extra["sequence"]=sequence;extra["session"]=session;extra["model_sha256"]=model_sha;extra["checkpoint"]=current.describe();send(extra);};
        auto same_base=[&](){const auto base=checkpoint::from_json(command.at("base"));require(base.describe()==current.describe(),"Session base token/position/owner hashes differ");};
        if(operation=="close"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation"});reply({{"closed",true}});return;
        }else if(operation=="suspend"){
            // DS4COMBO fast switch: give back VRAM, DS heaps and carry RAM; keep
            // the process, CUDA context, model lease and plans. The frontier is
            // dropped: only resume (with a new native base) or close may follow.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base"});same_base();
            require(!device_suspended&&have_remote&&!awaiting_ack&&remote.describe()==current.describe(),"Suspend requires the exported frontier to be acknowledged by the peer");
            std::string error;success(sweep->suspend_device(error),error);device_suspended=true;
            size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(weights.backend),&free,&total);
            reply({{"suspended",true},{"device_free_bytes",free},{"device_total_bytes",total},{"suspend_ms",ms(started)}});
            continue;
        }else if(operation=="resume"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","native_base_contract_sha256","tokens"});
            require(device_suspended,"Resume requires a suspended session");
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=capacity,"Resume prefill token count exceeds reserved chunk");
            std::vector<int32_t>tokens;tokens.reserve(ids.size());for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Resume token out of vocabulary");tokens.push_back(int32_t(n));}
            std::string error;success(sweep->resume_device(error),error);const double resume_ms=ms(started);
            // The new base follows this command on the same pipe: contract line + payload.
            combo_base_input incoming(io,session,command.at("native_base_contract_sha256").get<std::string>(),owner_epoch,uint32_t(options.context));
            require(tokens.size()<uint64_t(options.context)-incoming.base.frontier.position,"Resume prefill exceeds context capacity");
            const double base_ms=ms(started)-resume_ms;
            auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
            next_options.carry.tokens=uint32_t(tokens.size());if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=uint32_t(tokens.size());
            success(sweep->open(next_weights,next_options,error,previous.get(),nullptr,&incoming.base),error);incoming.finish();previous.reset();
            const double admission_ms=ms(started)-resume_ms-base_ms;
            success(sweep->run(tokens,[]{return should_stop();},error),error);const auto timing=sweep->timing();
            current=inspect();require(current.frontier.position==incoming.base.frontier.position+tokens.size(),"Resume prefill position did not advance exactly");
            have_remote=false;initial_sent=false;awaiting_ack=false;device_suspended=false;remote=checkpoint{};
            std::fprintf(stderr,"DS4COMBO_RESUME_COMPLETE resume_ms=%.3f base_import_ms=%.3f admission_ms=%.3f prompt_ms=%.3f tokens=%zu\n",resume_ms,base_ms,admission_ms,timing.prompt_processing_ms,tokens.size());
            reply({{"peer_synchronized",false},{"prompt_tokens",tokens.size()},{"resume_ms",resume_ms},{"base_import_ms",base_ms},{"admission_ms",admission_ms},
                {"prompt_processing_ms",timing.prompt_processing_ms},{"export_preparation_ms",timing.export_preparation_ms}});
            continue;
        }else if(device_suspended){
            require(false,"Suspended session accepts only resume or close");
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
                send({{"status","payload"},{"sequence",sequence},{"session",session},{"model_sha256",model_sha},
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
        }else if(operation=="decode"){
            // LLAMADECODE B0: greedy decode on the RTX in this process. Each
            // evaluated token is one stateful 1-row append through the same
            // owner transfer as "prefill" (existing sweep/carry/cache path).
            // A sampled stop token, or the token sampled at max_tokens, is
            // returned but NOT evaluated: the caller's next prefill carries it.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","max_tokens","stop_tokens"});same_base();
            require(!awaiting_ack,"Local decode while a state transfer awaits acknowledgement");
            require(inspect().describe()==current.describe(),"Local decode actual token/owner state differs from checkpoint");
            const auto maximum=dsv41_live::u64(command.at("max_tokens"));require(maximum&&maximum<=4096,"Local decode requires1..4096 max_tokens");
            const auto&stops=command.at("stop_tokens");require(stops.is_array()&&stops.size()<=16,"Local decode accepts at most16 stop tokens");
            std::vector<int32_t>stop_ids;for(const auto&v:stops){const auto n=dsv41_live::u64(v);require(n<129280,"Stop token out of vocabulary");stop_ids.push_back(int32_t(n));}
            require(uint64_t(current.frontier.position)+maximum<uint64_t(options.context),"Local decode exceeds context capacity");
            std::vector<int32_t>generated;std::vector<double>step_ms;std::string error;std::vector<float>logits;bool stopped=false;
            for(uint64_t index=0;index<maximum;++index){
                stop();const auto step_started=clock_type::now();
                success(sweep->completed_logits(logits,error),error);
                int32_t best=0;for(int32_t i=1;i<int32_t(logits.size());++i)if(logits[size_t(i)]>logits[size_t(best)])best=i;
                generated.push_back(best);
                if(std::find(stop_ids.begin(),stop_ids.end(),best)!=stop_ids.end()){stopped=true;break;}
                if(index+1==maximum)break;
                if(ds4_decode_enabled()){
                    if(index==0)ds4x.prepare(*sweep,weights,uint32_t(options.context));
                    ds4x.step(*sweep,weights.backend,best,current.frontier.position+uint32_t(index));
                    if(index==0)grow_decode_l2(*sweep,options.fixed.get(),weights.backend);
                    step_ms.push_back(ms(step_started));
                    std::fprintf(stderr,"LLAMADECODE_STEP index=%llu token=%d position=%u step_ms=%.3f engine=ds4\n",(unsigned long long)index,best,
                        current.frontier.position+uint32_t(index)+1,step_ms.back());std::fflush(stderr);
                    continue;
                }
                auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
                next_options.carry.tokens=1;if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=1;
                next_weights.decode_layout=routed_decode_layout(); // LLAMADECODE B1: routed-expert fetch
                {decode_quiet_scope quiet;
                success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
                const std::vector<int32_t>one{best};success(sweep->run(one,[]{return should_stop();},error),error);}
                if(index==0)grow_decode_l2(*sweep,options.fixed.get(),weights.backend);
                step_ms.push_back(ms(step_started));
                std::fprintf(stderr,"LLAMADECODE_STEP index=%llu token=%d position=%u step_ms=%.3f prompt_ms=%.3f\n",(unsigned long long)index,best,
                    current.frontier.position+uint32_t(index)+1,step_ms.back(),sweep->timing().prompt_processing_ms);std::fflush(stderr);
            }
            current=inspect();const uint32_t evaluated=uint32_t(step_ms.size());
            double total_ms=0;for(const double v:step_ms)total_ms+=v;
            std::fprintf(stderr,"LLAMADECODE_DECODE_COMPLETE generated=%zu evaluated=%u stopped=%d position=%u decode_ms=%.3f tok_s=%.6f\n",generated.size(),evaluated,int(stopped),
                current.frontier.position,total_ms,total_ms>0?evaluated*1000.0/total_ms:0.0);std::fflush(stderr);
            reply({{"peer_synchronized",have_remote&&remote.describe()==current.describe()},{"tokens",generated},{"evaluated_tokens",evaluated},{"stopped",stopped},
                {"step_ms",step_ms},{"decode_ms",total_ms},{"RTX_decode",true}});
        }else if(operation=="decode_forced"){
            // LLAMADECODE B0/B1 gate: teacher-forced 1-row decode. For each i the
            // frontier's full F32 logits are written to logits_path (raw rows of
            // 129280 floats), then tokens[i] is appended; the last token is not
            // evaluated. Same per-token path as "decode"; only the token choice differs.
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","tokens","logits_path"});same_base();
            require(!awaiting_ack,"Forced decode while a state transfer awaits acknowledgement");
            require(inspect().describe()==current.describe(),"Forced decode actual token/owner state differs from checkpoint");
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=4096,"Forced decode requires1..4096 tokens");
            std::vector<int32_t>forced;for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Forced token out of vocabulary");forced.push_back(int32_t(n));}
            require(uint64_t(current.frontier.position)+forced.size()<uint64_t(options.context),"Forced decode exceeds context capacity");
            const auto path=std::filesystem::u8path(command.at("logits_path").get<std::string>());
            require(path.is_absolute()&&!std::filesystem::exists(path),"Forced decode logits path must be absolute and new");
            HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
            require(file!=INVALID_HANDLE_VALUE,"Cannot create forced decode logits file");
            struct closer{HANDLE h;~closer(){CloseHandle(h);}}close_file{file};
            std::vector<int32_t>argmax;std::vector<double>step_ms;std::string error;std::vector<float>logits;
            for(size_t index=0;index<forced.size();++index){
                stop();const auto step_started=clock_type::now();
                success(sweep->completed_logits(logits,error),error);
                int32_t best=0;for(int32_t i=1;i<int32_t(logits.size());++i)if(logits[size_t(i)]>logits[size_t(best)])best=i;
                argmax.push_back(best);
                const auto*bytes=reinterpret_cast<const char*>(logits.data());size_t left=logits.size()*sizeof(float);
                while(left){DWORD wrote=0;const DWORD n=DWORD(std::min<size_t>(left,1u<<24));require(WriteFile(file,bytes,n,&wrote,nullptr)&&wrote==n,"Forced decode logits write failed");bytes+=n;left-=n;}
                if(index+1==forced.size())break;
                if(ds4_decode_enabled()){
                    if(index==0)ds4x.prepare(*sweep,weights,uint32_t(options.context));
                    ds4x.step(*sweep,weights.backend,forced[index],current.frontier.position+uint32_t(index));
                    if(index==0)grow_decode_l2(*sweep,options.fixed.get(),weights.backend);
                    step_ms.push_back(ms(step_started));
                    std::fprintf(stderr,"LLAMADECODE_FORCED_STEP index=%zu forced=%d argmax=%d step_ms=%.3f engine=ds4\n",index,forced[index],best,step_ms.back());std::fflush(stderr);
                    continue;
                }
                auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
                next_options.carry.tokens=1;if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=1;
                next_weights.decode_layout=routed_decode_layout(); // LLAMADECODE B1: routed-expert fetch
                {decode_quiet_scope quiet;
                success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
                const std::vector<int32_t>one{forced[index]};success(sweep->run(one,[]{return should_stop();},error),error);}
                if(index==0)grow_decode_l2(*sweep,options.fixed.get(),weights.backend);
                step_ms.push_back(ms(step_started));
                std::fprintf(stderr,"LLAMADECODE_FORCED_STEP index=%zu forced=%d argmax=%d step_ms=%.3f\n",index,forced[index],best,step_ms.back());std::fflush(stderr);
            }
            current=inspect();double total_ms=0;for(const double v:step_ms)total_ms+=v;
            reply({{"peer_synchronized",have_remote&&remote.describe()==current.describe()},{"argmax",argmax},{"vectors",argmax.size()},
                {"evaluated_tokens",step_ms.size()},{"step_ms",step_ms},{"decode_ms",total_ms},{"RTX_decode",true}});
        }else if(operation=="prefill"){
            dsv41_live::keys(command,{"session","model_sha256","sequence","operation","base","tokens"});same_base();
            // LLAMADECODE B0: a local-only session (never exported to a peer)
            // may append without a peer; a peer-bound session keeps exact sync.
            const bool local_only=!have_remote&&!initial_sent&&!awaiting_ack;
            require(local_only||(have_remote&&!awaiting_ack&&remote.describe()==current.describe()),"Incremental prefill requires exact two-way synchronization");
            require(inspect().describe()==current.describe(),"Incremental prefill actual token/owner state differs from acknowledged checkpoint");
            const auto&ids=command.at("tokens");require(ids.is_array()&&!ids.empty()&&ids.size()<=capacity,"Live prefill token count exceeds reserved chunk");
            require(ids.size()<uint64_t(options.context)-current.frontier.position,"Live prefill exceeds context capacity");
            std::vector<int32_t>tokens;tokens.reserve(ids.size());for(const auto&v:ids){const auto n=dsv41_live::u64(v);require(n<129280,"Live token out of vocabulary");tokens.push_back(int32_t(n));}
            auto previous=std::move(sweep);sweep.reset(new llama_dsv41_sweep);auto next_options=options;auto next_weights=weights;
            next_options.carry.tokens=uint32_t(tokens.size());if(next_weights.engram_prefetch_host_bytes)next_weights.engram_prefetch_tokens=uint32_t(tokens.size());
            std::string error;success(sweep->open(next_weights,next_options,error,nullptr,previous.get()),error);require(!previous->ready(),"Transferred prefix still publishes");previous.reset();
            const double admission_ms=ms(started);success(sweep->run(tokens,[]{return should_stop();},error),error);const auto timing=sweep->timing();
            const uint32_t before=current.frontier.position;current=inspect();require(current.frontier.position==before+tokens.size(),"Live prefill position did not advance exactly");
            reply({{"peer_synchronized",false},{"prompt_tokens",tokens.size()},{"prompt_processing_ms",admission_ms+timing.prompt_processing_ms},{"export_preparation_ms",timing.export_preparation_ms}});
        }else require(false,"Unknown persistent session operation");
    }
}
}
