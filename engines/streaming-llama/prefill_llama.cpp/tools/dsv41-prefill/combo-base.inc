/* Native mode1 bootstrap over the existing bounded parent-owned pipe. The
 * caller pins the whole versioned contract on the command line after local
 * model/tokenizer/template/RoPE/epoch validation. State never touches disk. */
namespace {
class combo_base_input {
    std::optional<session_pipe> owned;
    session_pipe &io;
    llama_dsv41::wire_plan plan;
    uint32_t owner=0;uint64_t owner_offset=0,consumed=0;
public:
    llama_dsv41_sweep::native_base base;
    combo_base_input(uint64_t timeout,HANDLE output,const std::string&session,
            const std::string&contract_sha,uint64_t epoch,uint32_t context,const std::string&model_sha):io(owned.emplace(timeout,output)){
        load(session,contract_sha,epoch,context,model_sha);
    }
    // DS4COMBO fast switch: a later base arrives on the live session's own pipe.
    combo_base_input(session_pipe&shared,const std::string&session,
            const std::string&contract_sha,uint64_t epoch,uint32_t context,const std::string&model_sha):io(shared){
        load(session,contract_sha,epoch,context,model_sha);
    }
    combo_base_input(const combo_base_input&)=delete;combo_base_input&operator=(const combo_base_input&)=delete;
private:
    void load(const std::string&session,const std::string&contract_sha,uint64_t epoch,uint32_t context,const std::string&model_sha){
        using namespace dsv41_live;
        const auto line=io.line();
        need(sha_valid(contract_sha)&&dsv41_boundary::sha256(line.data(),line.size())==contract_sha,"Native base contract pin mismatch");
        const auto j=parse_control(line);
        auto standard=j;
        if(j.contains("local_state")){
            const auto&local=j.at("local_state");keys(local,{"version","yoco_keep","cache_precision","producer_sha256"});
            need(local.at("version")=="slstack-local-state/1","Unknown local snapshot version");
            const auto keep=u64(local.at("yoco_keep"));need(keep==0||keep==140,"Local YOCO mode invalid");
            const auto precision=local.at("cache_precision").get<std::string>();need(precision=="f32"||precision=="bf16","Local cache precision invalid");
            need(sha_valid(local.at("producer_sha256").get<std::string>()),"Local producer digest invalid");
            base.local_snapshot=true;base.yoco_keep=uint32_t(keep);base.cache_precision=precision;standard.erase("local_state");
        }
        keys(standard,{"version","mode","model_sha256","tokenizer_sha256","template_sha256","rope_sha256","kv_dtype","context","position","request_id","owner_epoch","generation","tokens_sha256","logits_sha256","owners","decode_provenance"});
        need(j.at("version")=="dsv41-cuda-target-state/1"&&u64(j.at("mode"))==1&&j.at("model_sha256")==model_sha&&
            j.at("request_id")==session&&u64(j.at("owner_epoch"))==epoch&&epoch&&u64(j.at("generation")),"Native base version/model/request/epoch mismatch");
        for(const auto*key:{"tokenizer_sha256","template_sha256","rope_sha256","tokens_sha256","logits_sha256"})
            need(sha_valid(j.at(key).get<std::string>()),"Native base digest malformed");
        need(j.at("kv_dtype")=="f32","Native base requires the qualified canonical F32 KV representation");
        need(j.at("decode_provenance")=="unbiased"||j.at("decode_provenance")=="manual-bias"||j.at("decode_provenance")=="phase-bias","Native base decode provenance missing");
        const uint64_t position=u64(j.at("position"));
        need(u64(j.at("context"))==context&&context<=(base.local_snapshot?1048576u:262144u)&&position&&position<context,"Native base context/position mismatch");
        base.frontier={context,uint32_t(position),1,0,0};std::string error;
        need(llama_dsv41::make_wire_plan(base.frontier,plan,error),error);
        need(j.at("owners").is_array()&&j.at("owners").size()==plan.spans.size(),"Native base owner count mismatch");
        const std::array<const char*,6> names{{"dspark","raw","compressed","index","pool_kv","pool_score"}};
        std::map<std::pair<std::string,uint64_t>,std::pair<uint64_t,std::string>> owners;
        for(const auto&row:j.at("owners")){
            keys(row,{"kind","index","bytes","sha256"});const auto digest=row.at("sha256").get<std::string>();
            need(sha_valid(digest),"Native base owner SHA malformed");
            need(owners.emplace(std::make_pair(row.at("kind").get<std::string>(),u64(row.at("index"))),
                std::make_pair(u64(row.at("bytes")),digest)).second,"Duplicate native base owner identity");
        }
        for(const auto&s:plan.spans){auto found=owners.find({names.at(size_t(s.kind)),s.index});
            need(found!=owners.end()&&found->second.first==s.bytes,"Native base kind/index/extent mismatch");
            base.owner_sha256.push_back(found->second.second);
        }
        std::array<uint32_t,13>h{};io.read(h.data(),52);
        const std::array<uint32_t,13>expected{{0x34565344,2,context,1,128,128,context+1,uint32_t(position),40,512,128,129280,0x413431}};
        need(h==expected,"Native base payload header mismatch");consumed=52;
        base.tokens.resize(size_t(position));base.logits.resize(129280);
        auto read=[&](void*dst,size_t n){auto*p=static_cast<unsigned char*>(dst);while(n){size_t count=std::min<size_t>(65536,n);io.read(p,count);p+=count;n-=count;consumed+=count;}};
        read(base.tokens.data(),base.tokens.size()*4);read(base.logits.data(),base.logits.size()*4);
        base.tokens_sha256=j.at("tokens_sha256").get<std::string>();base.logits_sha256=j.at("logits_sha256").get<std::string>();
        base.read=[this](const llama_dsv41::state_span&s,uint64_t offset,void*dst,size_t bytes){
            while(owner<plan.spans.size()&&owner_offset==plan.spans[owner].bytes){++owner;owner_offset=0;}
            if(owner>=plan.spans.size()||bytes>65536||offset!=owner_offset)return false;
            const auto&expected=plan.spans[owner];
            if(s.kind!=expected.kind||s.index!=expected.index||s.bytes!=expected.bytes||bytes>expected.bytes-owner_offset)return false;
            io.read(dst,bytes);owner_offset+=bytes;consumed+=bytes;return true;
        };
    }
public:
    void finish(){dsv41_live::need(consumed==plan.bytes,"Native base import left bytes unread");}
};
}
