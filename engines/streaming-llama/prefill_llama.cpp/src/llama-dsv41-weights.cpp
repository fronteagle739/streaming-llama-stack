#include "llama-dsv41-layer-log.h"
#include "llama-dsv41-weights.h"
#include <tuple>
#include "llama-dsv41-engram-prefetch.h"
#include "llama-weight-fetcher.h"
#include "llama-wf-bounded-policy.h"

#include <algorithm>
#include <unordered_map>
#include <list>
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <atomic>
#include <chrono>
#include <exception>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#if defined(LLAMA_DIRECTSTORAGE) && defined(GGML_USE_CUDA)
#include <dstorage.h>
static_assert(sizeof(DSTORAGE_REQUEST)+sizeof(int32_t)<=512,"Requalify strict request metadata budget for this SDK");
#endif
#endif
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif
#endif

namespace {
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
uint64_t add(uint64_t a,uint64_t b){require(a<=UINT64_MAX-b,"V4.1 weight capacity overflow");return a+b;}
uint64_t align(uint64_t value,uint64_t alignment){require(alignment&&!(alignment&(alignment-1)),"Invalid weight alignment");return add(value,alignment-1)&~(alignment-1);}
bool query(uint32_t l){return l==2||l==8||l==14||l==20||l==24||l==28||l==32||l==36;}
bool source(uint32_t l){return l==2||l==8||l==14||l==20;}
constexpr uint64_t mib=1024ull*1024;
constexpr uint64_t metadata_allowance=32*mib;
constexpr uint64_t factory_allowance=512*mib; // each system/upload tier, not a total factory cap
constexpr const char *original_sha="2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f";
}

struct llama_dsv41_weight_provider::implementation {
    struct entry{ggml_tensor*t;uint64_t file_offset,gpu_offset,allocation;};
    struct stage{std::vector<entry>entries;uint64_t bytes=0;};
    mutable std::mutex mutex;
    config options;
    capacity planned;
    bool inspected=false,allocated=false,poisoned=false,leased=false;
    uint32_t next=0;
    gguf_context_ptr gguf;
    ggml_context_ptr tensors;
    std::array<stage,42>stages;
    std::array<stage,82>decode_stages;bool decode_layout=false;bool experts_prebound=false;int dense_held=-1;
    uint64_t resident_dense_bytes=0,decode_bank0_resident=0;
    uint64_t fetched_total=0;double fetch_ms_total=0;
    std::array<ggml_tensor*,40*3>routed{};std::array<ggml_tensor*,40*3>shared{};
    bool shared_tensor(const ggml_tensor*t)const{return std::find(shared.begin(),shared.end(),t)!=shared.end();}
    // LLAMADECODE B2: decode expert cache. Bank 0 = [embedding | shared experts |
    // gate pool | up pool | down pool]; a slot holds one (layer, expert) triple.
    ggml_context_ptr pool_ctx;std::array<ggml_tensor*,3>pool{};std::array<uint64_t,3>pool_offset{},pool_stride{};
    uint32_t pool_slots=0;
    std::vector<int32_t>slot_key;std::unordered_map<int32_t,int32_t>key_slot;
    std::list<int32_t>lru;std::vector<std::list<int32_t>::iterator>lru_pos;
    uint64_t cache_hits=0,cache_misses=0,cache_bytes=0;double cache_ms=0;
    // LLAMADECODE bind-once: set after a complete decode-layout pass; cleared on any layout rebuild.
    bool resident_bound=false;
    static bool bind_once_enabled(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_BIND_ONCE");return !(e&&e[0]=='0'&&!e[1]);}();return on;}
    bool bound()const{return decode_layout&&resident_bound&&bind_once_enabled();}
    // LLAMADECODE L2 (exclusive victim tier, own VRAM buffer: [gate|up|down] x l2_slots).
    ggml_backend_buffer_ptr l2_buffer;uint32_t l2_slots=0;std::array<char*,3>l2_base{};
    std::vector<int32_t>l2_key;std::unordered_map<int32_t,int32_t>l2_of;
    std::list<int32_t>l2_lru;std::vector<std::list<int32_t>::iterator>l2_pos;
    uint64_t l2_hits=0,l2_stored=0;
    void copy_slot(char*const dst[3],uint64_t dst_slot,const char*const src[3],uint64_t src_slot,void*stream){
#ifdef GGML_USE_CUDA
        for(uint32_t k=0;k<3;++k)require(cudaMemcpyAsync(dst[k]+dst_slot*pool_stride[k],src[k]+src_slot*pool_stride[k],pool_stride[k],
            cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream))==cudaSuccess,"Expert L2 slot copy failed");
#else
        (void)dst;(void)dst_slot;(void)src;(void)src_slot;(void)stream;throw std::runtime_error("Expert L2 requires CUDA");
#endif
    }
    void release_l2(){
        if(l2_buffer&&options.backend)ggml_backend_synchronize(options.backend);
#ifdef GGML_USE_CUDA
        if(l2_buffer)cudaDeviceSynchronize();
#endif
        l2_buffer.reset();l2_slots=0;l2_base={};l2_key.clear();l2_of.clear();l2_lru.clear();l2_pos.clear();
    }
    // LLAMADECODE B2: LRU over (layer, expert). Hits move to the front first,
    // so a miss never evicts an expert this same request needs. Only misses
    // are fetched (explicit DirectStorage extents into bank 0).
    std::unordered_map<int32_t,int32_t> assign_expert_slots(uint32_t layer,const int32_t*ids,size_t count,void*stream){
        std::vector<int32_t>unique;for(size_t i=0;i<count;++i){const auto id=ids[i];require(id>=0&&id<384,"Routed expert id out of range");if(std::find(unique.begin(),unique.end(),id)==unique.end())unique.push_back(id);}
        require(unique.size()<=pool_slots,"Routed experts exceed the cache pool");
        std::vector<int32_t>missing;std::unordered_map<int32_t,int32_t>slot_of;
        for(const auto e:unique){const int32_t key=int32_t(layer)*384+e;auto it=key_slot.find(key);
            if(it!=key_slot.end()){slot_of[e]=it->second;lru.splice(lru.begin(),lru,lru_pos[size_t(it->second)]);++cache_hits;}
            else missing.push_back(e);}
        std::vector<wf_extent>extents;bool copied=false;
        char*const l1[3]={static_cast<char*>(pool[0]->data),static_cast<char*>(pool[1]->data),static_cast<char*>(pool[2]->data)};
        for(const auto e:missing){
            const int32_t slot=lru.back();lru.splice(lru.begin(),lru,lru_pos[size_t(slot)]);
            const int32_t key=int32_t(layer)*384+e;
            int32_t from_l2=-1;
            if(l2_slots){auto it=l2_of.find(key);if(it!=l2_of.end()){from_l2=it->second;l2_lru.splice(l2_lru.begin(),l2_lru,l2_pos[size_t(from_l2)]);}}
            const int32_t victim=slot_key[size_t(slot)];
            if(victim>=0){key_slot.erase(victim);
                if(l2_slots>=2){   // keep the evicted expert in L2 (its least-recent slot; never the one read below)
                    const int32_t t=l2_lru.back();require(t!=from_l2,"Expert L2 victim slot collides with its source");
                    l2_lru.splice(l2_lru.begin(),l2_lru,l2_pos[size_t(t)]);
                    if(l2_key[size_t(t)]>=0)l2_of.erase(l2_key[size_t(t)]);
                    copy_slot(l2_base.data(),uint64_t(t),l1,uint64_t(slot),stream);copied=true;l2_key[size_t(t)]=victim;l2_of[victim]=t;++l2_stored;
                }}
            slot_key[size_t(slot)]=key;key_slot[key]=slot;slot_of[e]=slot;
            if(from_l2>=0){
                copy_slot(l1,uint64_t(slot),l2_base.data(),uint64_t(from_l2),stream);copied=true;
                l2_of.erase(key);l2_key[size_t(from_l2)]=-1;l2_lru.splice(l2_lru.end(),l2_lru,l2_pos[size_t(from_l2)]);++l2_hits;
                continue;
            }
            ++cache_misses;
            for(uint32_t k=0;k<3;++k){
                uint64_t file=0;for(const auto&entry:stages[layer+1].entries)if(entry.t==routed[layer*3+k])file=entry.file_offset;
                require(file,"Routed expert file extent missing");
                extents.push_back({add(file,uint64_t(e)*pool_stride[k]),add(pool_offset[k],uint64_t(slot)*pool_stride[k]),pool_stride[k],0});
            }
        }
#ifdef GGML_USE_CUDA
        // Slot copies complete before DirectStorage overwrites any copied-from L1 slot
        // and before an external decoder (own stream) reads the copied-in slots.
        if(copied)require(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))==cudaSuccess,"Expert L2 copy synchronization failed");
#endif
        if(!extents.empty()){
            uint64_t bytes=0;double fetch_ms=0;
            // A failed or partial fetch poisons the provider (guard); the slot table is then discarded with it.
            require(wf_fetch_extents_direct(fetcher,0,extents.data(),int(extents.size()),stream,&bytes,&fetch_ms)&&!wf_has_failed(fetcher),
                "Routed expert-cache fetch failed");
            cache_bytes+=bytes;cache_ms+=fetch_ms;fetched_total+=bytes;fetch_ms_total+=fetch_ms;
        }
        return slot_of;
    }
    void reset_cache(){slot_key.assign(pool_slots,-1);key_slot.clear();lru.clear();lru_pos.assign(pool_slots,lru.end());
        for(uint32_t s=0;s<pool_slots;++s){lru.push_back(int32_t(s));lru_pos[s]=std::prev(lru.end());}}
    uint32_t stage_count()const{return decode_layout?82u:42u;}
    stage&stage_at(uint32_t id){return decode_layout?decode_stages[id]:stages[id];}
    bool routed_tensor(const ggml_tensor*t)const{return std::find(routed.begin(),routed.end(),t)!=routed.end();}
    static void place(stage&target,const entry&from,ggml_backend_buffer_type_t buft){
        const auto where=align(target.bytes,ggml_backend_buft_get_alignment(buft));
        target.entries.push_back({from.t,from.file_offset,where,from.allocation});target.bytes=add(where,from.allocation);
    }
    // Per-expert spans for a routed projection [ne0,ne1,384]; other tensors are one span.
    void spans_for(const stage&s,std::vector<wf_weight_span>&spans)const{
        for(const auto&e:s.entries){
            require(!(decode_layout&&routed_tensor(e.t)),"Routed experts are not decode stages");
            {wf_weight_span s{};s.tensor_name=e.t->name;s.file_offset=e.file_offset;s.nbytes=ggml_nbytes(e.t);s.alloc_size=e.allocation;s.alloc_offset=e.gpu_offset;s.expert_idx=-1;spans.push_back(s);}
        }
    }
    std::array<llama_dsv41_layer_graph::weights,40>layers{};
    ggml_tensor *embedding=nullptr,*head_norm=nullptr,*head_weight=nullptr;
    std::array<uint64_t,2>engram_offsets{};
    llama_dsv41::engram_layout engram;
    std::shared_ptr<llama_dsv41_engram_prompt_cache>engram_prompt;
    // DS4COMBO producer-warm: the whole-prompt Engram row cache for a request
    // whose tokens are already known, built on its own IoRing while the
    // provider serves another pass (the one-token warmup). Immutable inputs
    // only; adopted by prepare_engram_prompt for exactly the same tokens/history.
    struct early_engram{
        std::vector<int32_t>tokens;llama_dsv41::engram_history initial;
        std::shared_ptr<llama_dsv41_engram_prompt_cache>cache;std::exception_ptr failure;
        std::atomic<bool>cancel{false};double wall_ms=0;
        // Adopted while still loading (overlap): engram_prompt shares cache,
        // this entry and its thread stay until wait_engram_table(1) joins them.
        bool adopted=false;
    };
    std::shared_ptr<early_engram>early;
    bool device_suspended=false;
    void create_pipeline();
    std::thread early_thread;
    void join_early(){if(early)early->cancel=true;if(early_thread.joinable())early_thread.join();early.reset();}
    std::vector<ggml_backend_buffer_ptr>buffers;
    wf_context *fetcher=nullptr;
#ifdef _WIN32
    HANDLE file=INVALID_HANDLE_VALUE;
    HANDLE engram_file=INVALID_HANDLE_VALUE;
    uint8_t *engram_page=nullptr;
    std::vector<HANDLE>parents;
    std::filesystem::path path;
    void identity()const{
        BY_HANDLE_FILE_INFORMATION s{};
        require(file!=INVALID_HANDLE_VALUE&&GetFileInformationByHandle(file,&s),"Cannot inspect verified model lease");
        FILE_ID_INFO id{};require(GetFileInformationByHandleEx(file,FileIdInfo,&id,sizeof(id)),"Cannot bind original model FileIdInfo receipt");
        const uint64_t length=(uint64_t(s.nFileSizeHigh)<<32)|s.nFileSizeLow;
        const uint64_t modified=(uint64_t(s.ftLastWriteTime.dwHighDateTime)<<32)|s.ftLastWriteTime.dwLowDateTime;
        constexpr uint64_t epoch=116444736000000000ull;
        require(modified>=epoch&&(modified-epoch)<=UINT64_MAX/100,"Model timestamp cannot be bound to Unix receipt");
        require(id.VolumeSerialNumber==options.verified_volume&&std::memcmp(id.FileId.Identifier,options.verified_file_id.data(),16)==0&&
            length==options.verified_file_bytes&&(modified-epoch)*100==options.verified_mtime_ns&&s.nNumberOfLinks==1&&
            !(s.dwFileAttributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT|FILE_ATTRIBUTE_SPARSE_FILE|FILE_ATTRIBUTE_COMPRESSED)),
            "Model identity differs from completed SHA receipt; reverify before streaming");
    }
#else
    void identity()const{throw std::runtime_error("Physical V4.1 weight streaming requires Windows");}
#endif
    ~implementation(){
        invalidate();
#ifdef _WIN32
        if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
        if(engram_file!=INVALID_HANDLE_VALUE)CloseHandle(engram_file);
        for(auto handle:parents)CloseHandle(handle);
        _aligned_free(engram_page);
#endif
    }
    void invalidate(){
        poisoned=true;
        join_early();
        engram_prompt.reset();
        if(options.backend&&allocated)ggml_backend_synchronize(options.backend);
        release_l2();
        if(fetcher){wf_stop(fetcher);wf_quiesce(fetcher);wf_destroy(fetcher);fetcher=nullptr;}
        for(auto&s:stages)for(auto&e:s.entries){e.t->data=nullptr;e.t->buffer=nullptr;}
        buffers.clear();allocated=false;leased=false;resident_bound=false;
    }
    template<class F>bool guard(F fn,std::string&error){
        std::lock_guard<std::mutex>lock(mutex);
        try{require(!poisoned,"V4.1 weight provider is invalid");fn();error.clear();return true;}
        catch(const std::exception&e){invalidate();error=e.what();return false;}
    }
    int64_t key(const char *suffix)const{
        const std::string name=std::string("deepseek41.")+suffix;
        const auto id=gguf_find_key(gguf.get(),name.c_str());require(id>=0,"Missing native V4.1 metadata key");return id;
    }
    uint64_t integer(const char *suffix)const{
        const auto id=key(suffix);const auto type=gguf_get_kv_type(gguf.get(),id);
        if(type==GGUF_TYPE_UINT32)return gguf_get_val_u32(gguf.get(),id);
        require(type==GGUF_TYPE_UINT64,"Native integer metadata type changed");return gguf_get_val_u64(gguf.get(),id);
    }
    void number(const char*suffix,float expected)const{
        const auto id=key(suffix);const auto type=gguf_get_kv_type(gguf.get(),id);
        float value;
        if(type==GGUF_TYPE_FLOAT32)value=gguf_get_val_f32(gguf.get(),id);
        else if(type==GGUF_TYPE_FLOAT64)value=float(gguf_get_val_f64(gguf.get(),id));
        else value=float(integer(suffix));
        // Bit equality stays meaningful under release fast-math.
        require(std::memcmp(&value,&expected,sizeof(float))==0,"Native V4.1 numerical metadata differs");
    }
    void text(const char*suffix,const char*expected)const{
        const auto id=key(suffix);require(gguf_get_kv_type(gguf.get(),id)==GGUF_TYPE_STRING,"Native string metadata type changed");
        require(std::strcmp(gguf_get_val_str(gguf.get(),id),expected)==0,"Native V4.1 string metadata differs");
    }
    std::vector<uint64_t>array(const char*suffix,size_t maximum)const{
        const auto id=key(suffix);require(gguf_get_kv_type(gguf.get(),id)==GGUF_TYPE_ARRAY,"Native array metadata type changed");
        const auto type=gguf_get_arr_type(gguf.get(),id);const auto n=gguf_get_arr_n(gguf.get(),id);
        require(n<=maximum&&(type==GGUF_TYPE_UINT32||type==GGUF_TYPE_UINT64),"Native array metadata size/type differs");
        const auto*d=gguf_get_arr_data(gguf.get(),id);std::vector<uint64_t>result(n);
        for(size_t i=0;i<n;++i)result[i]=type==GGUF_TYPE_UINT32?static_cast<const uint32_t*>(d)[i]:static_cast<const uint64_t*>(d)[i];
        return result;
    }
    void metadata(){
        const auto arch=gguf_find_key(gguf.get(),"general.architecture");
        require(arch>=0&&gguf_get_kv_type(gguf.get(),arch)==GGUF_TYPE_STRING&&
            std::strcmp(gguf_get_val_str(gguf.get(),arch),"deepseek41")==0,"Not a native V4.1 GGUF");
        for(auto pair:{std::pair<const char*,uint64_t>{"vocab_size",129280},{"hidden_size",5120},
            {"moe_intermediate_size",2304},{"num_hidden_layers",40},{"num_attention_heads",64},
            {"num_key_value_heads",1},{"head_dim",512},{"qk_rope_head_dim",64},{"q_lora_rank",1280},
            {"o_lora_rank",1024},{"o_groups",8},{"n_routed_experts",384},{"n_shared_experts",1},
            {"num_experts_per_tok",6},{"max_position_embeddings",1048576},{"sliding_window",128},
            {"index_n_heads",32},{"index_head_dim",128},{"index_topk",512},{"candidate_source_layer_id",20},
            {"candidate_topk_blocks",2048},{"candidate_block_size",8},{"hc_mult",4},{"hc_sinkhorn_iters",20},
            {"rope_theta",10000},{"compress_rope_theta",160000}})require(integer(pair.first)==pair.second,"Unsupported native V4.1 geometry");
        for(auto pair:{std::pair<const char*,float>{"rms_norm_eps",1e-20f},{"hc_eps",1e-6f},{"swiglu_limit",10},
            {"routed_scaling_factor",1.5f},{"rope_scaling.factor",16},{"rope_scaling.beta_fast",32},
            {"rope_scaling.beta_slow",1},{"rope_scaling.original_max_position_embeddings",65536}})number(pair.first,pair.second);
        text("scoring_func","sqrtsoftplus");text("hidden_act","silu");text("topk_method","noaux_tc");
        const auto normalized=key("norm_topk_prob");require(gguf_get_kv_type(gguf.get(),normalized)==GGUF_TYPE_BOOL&&gguf_get_val_bool(gguf.get(),normalized),"Unnormalized native routing unsupported");
        auto ratios=array("compress_ratios",40);require(ratios.size()==40,"Missing V4.1 layer ratios");
        for(uint32_t l=0;l<40;++l)require(ratios[l]==(l<2?0u:l<20?2u:1u),"Incorrect native compression ownership");
        require(array("kv_source_layer_ids",4)==std::vector<uint64_t>{2,8,14,20},"Native KV source topology differs");
        require(array("index_source_layer_ids",8)==std::vector<uint64_t>{2,8,14,20,24,28,32,36},"Native index source topology differs");
        text("engram.encoding","e4m3_e8m0_32_row264");
        require(array("engram.layer_ids",2)==std::vector<uint64_t>{1,14},"Native Engram layers differ");
        const auto rows=array("engram.rows",2),mapping=array("engram.token_map",129280),primes=array("engram.primes",48),mult=array("engram.multipliers",8);
        require(rows.size()==2&&mapping.size()==129280&&primes.size()==48&&mult.size()==8,"Native Engram metadata dimensions differ");
        engram.compressed_vocab_size=uint32_t(integer("engram.compressed_vocab_size"));engram.pad_id=uint32_t(integer("engram.pad_id"));
        for(auto v:mapping){require(v<=UINT32_MAX,"Engram mapped ID overflow");engram.token_map.push_back(uint32_t(v));}
        for(size_t e=0;e<2;++e){require(rows[e]<=UINT32_MAX,"Engram rows overflow");engram.rows[e]=uint32_t(rows[e]);
            for(size_t i=0;i<24;++i){require(primes[e*24+i]<=UINT32_MAX,"Engram prime overflow");engram.primes[e][i]=uint32_t(primes[e*24+i]);}
            for(size_t i=0;i<4;++i)engram.multipliers[e][i]=mult[e*4+i];}
        std::string why;require(engram.validate(why),"Invalid original Engram hashing layout");
    }
    ggml_tensor *tensor(const std::string&name,ggml_type type,std::initializer_list<int64_t>shape){
        auto*t=ggml_get_tensor(tensors.get(),name.c_str());require(t&&t->type==type&&!t->data&&!t->buffer,"Missing or converted native weight");
        size_t d=0;for(auto n:shape)require(t->ne[d++]==n,"Native weight shape differs");for(;d<4;++d)require(t->ne[d]==1,"Unexpected native tensor dimension");return t;
    }
    void add_tensor(uint32_t stage_id,ggml_tensor*t){
        const auto id=gguf_find_tensor(gguf.get(),t->name);require(id>=0,"Missing GGUF tensor extent");
        const uint64_t off=add(gguf_get_data_offset(gguf.get()),gguf_get_tensor_offset(gguf.get(),id)),bytes=ggml_nbytes(t);
        require(off<=options.verified_file_bytes&&bytes<=options.verified_file_bytes-off,"GGUF weight extends beyond verified file");
        auto&stage=stages[stage_id];const auto buft=ggml_backend_get_default_buffer_type(options.backend);
        const auto allocation=ggml_backend_buft_get_alloc_size(buft,t);const auto where=align(stage.bytes,ggml_backend_buft_get_alignment(buft));
        require(allocation>=bytes,"Weight backend allocation is undersized");stage.entries.push_back({t,off,where,allocation});stage.bytes=add(where,allocation);
    }
    void plan_weights(){
        embedding=tensor("token_embd.weight",GGML_TYPE_F16,{5120,129280});add_tensor(0,embedding);
        head_norm=tensor("output_norm.weight",GGML_TYPE_F32,{5120});head_weight=tensor("output.weight",GGML_TYPE_Q8_0,{5120,129280});
        add_tensor(41,head_norm);add_tensor(41,head_weight);
        for(uint32_t l=0;l<40;++l){
            const auto prefix="blk."+std::to_string(l)+".";auto&w=layers[l];auto&a=w.attention;auto&f=w.ffn;
            auto get=[&](const char*name,ggml_type type,std::initializer_list<int64_t>shape){auto*t=tensor(prefix+name,type,shape);add_tensor(l+1,t);return t;};
            w.attn_fn=get("hc_attn_fn.weight",GGML_TYPE_F16,{20480,24});w.attn_scale=get("hc_attn_scale.weight",GGML_TYPE_F32,{3});w.attn_base=get("hc_attn_base.weight",GGML_TYPE_F32,{24});w.attn_norm=get("attn_norm.weight",GGML_TYPE_F32,{5120});
            w.ffn_fn=get("hc_ffn_fn.weight",GGML_TYPE_F16,{20480,24});w.ffn_scale=get("hc_ffn_scale.weight",GGML_TYPE_F32,{3});w.ffn_base=get("hc_ffn_base.weight",GGML_TYPE_F32,{24});w.ffn_norm=get("ffn_norm.weight",GGML_TYPE_F32,{5120});
            a.q_a=get("attn_q_a.weight",GGML_TYPE_Q8_0,{5120,1280});a.q_norm=get("attn_q_a_norm.weight",GGML_TYPE_F32,{1280});a.q_b=get("attn_q_b.weight",GGML_TYPE_Q8_0,{1280,32768});
            a.kv=get("attn_kv.weight",GGML_TYPE_Q8_0,{5120,512});a.kv_norm=get("attn_kv_a_norm.weight",GGML_TYPE_F32,{512});
            a.out_a=get("attn_output_a.weight",GGML_TYPE_Q8_0,{4096,8192});a.out_b=get("attn_output_b.weight",GGML_TYPE_Q8_0,{8192,5120});a.attn_sinks=get("attn_sinks.weight",GGML_TYPE_F32,{64});
            if(source(l)){a.compressor=get("attn_compressor_kv.weight",GGML_TYPE_F16,{5120,512});a.compressor_norm=get("attn_compressor_norm.weight",GGML_TYPE_F32,{512});a.index_k=get("indexer.attn_k.weight",GGML_TYPE_F16,{512,128});a.index_norm=get("indexer.k_norm.weight",GGML_TYPE_F32,{128});if(l<20)a.gate=get("attn_compressor_gate.weight",GGML_TYPE_F16,{5120,512});}
            if(query(l)){a.index_q=get("indexer.attn_q_b.weight",GGML_TYPE_F16,{1280,4096});a.index_weights=get("indexer.proj.weight",GGML_TYPE_F16,{5120,32});}
            f.router=get("ffn_gate_inp.weight",GGML_TYPE_F32,{5120,384});f.text_bias=get("exp_probs_b.bias",GGML_TYPE_F32,{384});f.image_bias=get("exp_probs_b_vl.bias",GGML_TYPE_F32,{384});
            f.gate=get("ffn_gate_exps.weight",GGML_TYPE_Q4_K,{5120,2304,384});f.up=get("ffn_up_exps.weight",GGML_TYPE_Q4_K,{5120,2304,384});f.down=get("ffn_down_exps.weight",GGML_TYPE_Q4_K,{2304,5120,384});
            f.shared_gate=get("ffn_gate_shexp.weight",GGML_TYPE_Q8_0,{5120,2304});f.shared_up=get("ffn_up_shexp.weight",GGML_TYPE_Q8_0,{5120,2304});f.shared_down=get("ffn_down_shexp.weight",GGML_TYPE_Q8_0,{2304,5120});
            routed[l*3]=f.gate;routed[l*3+1]=f.up;routed[l*3+2]=f.down;shared[l*3]=f.shared_gate;shared[l*3+1]=f.shared_up;shared[l*3+2]=f.shared_down;
            if(l==1||l==14){w.engram_wkv=get("engram_kv.weight",GGML_TYPE_F16,{6144,25600});w.engram_q_norm=get("engram_q_norm.weight",GGML_TYPE_F32,{5120,4});w.engram_k_norm=get("engram_k_norm.weight",GGML_TYPE_F32,{5120,4});
                const size_t e=l==1?0:1;auto*t=tensor(prefix+"engram_embd.weight",GGML_TYPE_I8,{264,engram.rows[e]});
                const auto id=gguf_find_tensor(gguf.get(),t->name);engram_offsets[e]=add(gguf_get_data_offset(gguf.get()),gguf_get_tensor_offset(gguf.get(),id));
                require(engram_offsets[e]<=options.verified_file_bytes&&ggml_nbytes(t)<=options.verified_file_bytes-engram_offsets[e],"Original Engram table extent is incomplete");}
        }
        for(const auto&s:stages)planned.maximum_stage_bytes=std::max(planned.maximum_stage_bytes,s.bytes);
        {   // LLAMADECODE B1 decode layout: same tensors and file extents, re-packed.
            const auto buft=ggml_backend_get_default_buffer_type(options.backend);
            // LLAMADECODE B2 decode layout: every non-routed stage is resident in
            // its own bank region after one transfer. Even stages land in bank 0:
            // the embedding, then each layer's shared expert. Odd stages land in
            // bank 1: each layer's dense weights, then the head. Routed experts
            // are not stages: they live in bank 0's expert-cache pool.
            uint64_t bank0=0,bank1=0;
            decode_stages[0].bytes=bank0;for(const auto&e:stages[0].entries)place(decode_stages[0],e,buft);bank0=decode_stages[0].bytes;
            for(uint32_t l=0;l<40;++l){
                auto&dense=decode_stages[1+2*l];dense.bytes=bank1;
                for(const auto&e:stages[l+1].entries)if(!routed_tensor(e.t)&&!shared_tensor(e.t))place(dense,e,buft);bank1=dense.bytes;
                auto&experts=decode_stages[2+2*l];experts.bytes=bank0;
                for(const auto&e:stages[l+1].entries)if(shared_tensor(e.t))place(experts,e,buft);bank0=experts.bytes;
            }
            decode_stages[81].bytes=bank1;for(const auto&e:stages[41].entries)place(decode_stages[81],e,buft);bank1=decode_stages[81].bytes;
            resident_dense_bytes=bank1;decode_bank0_resident=bank0;
            for(const auto&s:decode_stages)require(s.bytes&&s.bytes<=planned.maximum_stage_bytes,"Decode stage exceeds the admitted prefill bank size");
        }
        const auto policy=llama_wf_bounded::make(planned.maximum_stage_bytes,options.buffers,
            options.staging_limit_bytes?options.staging_limit_bytes:llama_wf_bounded::baseline_staging_bytes);
        planned.compute_buffer_bytes=policy.compute_buffer_bytes;
        {   // LLAMADECODE B2: expert-cache pool geometry (one slot = gate+up+down).
            const auto buft=ggml_backend_get_default_buffer_type(options.backend);const uint64_t a=ggml_backend_buft_get_alignment(buft);
            for(uint32_t l=0;l<40;++l)for(uint32_t k=0;k<3;++k){const auto*t=routed[l*3+k],*t0=routed[k];
                require(t->type==t0->type&&t->ne[0]==t0->ne[0]&&t->ne[1]==t0->ne[1]&&t->ne[2]==384&&t->nb[2]==t0->nb[2]&&ggml_nbytes(t)==t->nb[2]*384,"Routed expert tensors differ in geometry across layers");}
            for(uint32_t k=0;k<3;++k)pool_stride[k]=routed[k]->nb[2];
            const uint64_t slot_bytes=pool_stride[0]+pool_stride[1]+pool_stride[2];
            const uint64_t base=align(decode_bank0_resident,a),room=planned.compute_buffer_bytes>base?planned.compute_buffer_bytes-base:0;
            pool_slots=uint32_t(std::min<uint64_t>(room/slot_bytes,40ull*384));
            require(pool_slots>=12,"Decode expert-cache pool has fewer than two layers of slots");
            pool_offset[0]=base;pool_offset[1]=add(base,uint64_t(pool_slots)*pool_stride[0]);pool_offset[2]=add(pool_offset[1],uint64_t(pool_slots)*pool_stride[1]);
            require(add(pool_offset[2],uint64_t(pool_slots)*pool_stride[2])<=planned.compute_buffer_bytes,"Decode expert-cache pool exceeds bank 0");
            ggml_context*raw=ggml_init({4*ggml_tensor_overhead(),nullptr,true});require(raw,"Expert pool metadata allocation failed");pool_ctx.reset(raw);
            const char*names[3]={"dsv41_expert_pool_gate","dsv41_expert_pool_up","dsv41_expert_pool_down"};
            for(uint32_t k=0;k<3;++k){pool[k]=ggml_new_tensor_3d(raw,routed[k]->type,routed[k]->ne[0],routed[k]->ne[1],pool_slots);ggml_set_name(pool[k],names[k]);
                require(pool[k]->nb[2]==pool_stride[k],"Expert pool stride differs from the original experts");}
        }
        const auto staging=policy.staging_bytes;
        planned.staging_bytes=staging;
        planned.request_bytes=options.auto_request_size?policy.request_bytes:512*1024;
        // The imported CUDA view aliases its D3D allocation. Count each
        // physical allocation once; the controller monitors selected-GPU NVML.
        planned.device_bytes=add(add(planned.compute_buffer_bytes*options.buffers,staging),factory_allowance+65536);
        if(options.compute_direct){
            planned.device_bytes=add(llama_wf_bounded::shared_heap_physical_bytes(planned.compute_buffer_bytes,options.buffers),factory_allowance+65536);planned.staging_bytes=0;
        }
        // wf_start retains request arrays for every stage. Each wave boundary
        // can add one request beyond ceil(tensor_bytes/io_block). Include
        // vector capacity growth plus temporary old/new allocations; the SDK
        // assertion above pins the conservative request+index byte bound.
        const uint64_t stride=options.ds_gpu_ring_slots?((staging/2)&~uint64_t(65535)):staging;
        require(stride>0,"GPU ring staging stride refused");
        uint64_t requests=0,waves=0;
        for(const auto&s:stages){waves=add(waves,(s.bytes+stride-1)/stride);
            for(const auto&e:s.entries){const uint64_t bytes=ggml_nbytes(e.t);
                const uint64_t cuts=(e.gpu_offset+bytes-1)/stride-e.gpu_offset/stride;
                requests=add(requests,add((bytes+planned.request_bytes-1)/planned.request_bytes,cuts));}}
        {   uint64_t decode_requests=0,decode_waves=0;const bool was=decode_layout;decode_layout=true;
            for(const auto&s:decode_stages){decode_waves=add(decode_waves,(s.bytes+stride-1)/stride);std::vector<wf_weight_span>spans;spans_for(s,spans);
                for(const auto&x:spans){const uint64_t cuts=(x.alloc_offset+x.nbytes-1)/stride-x.alloc_offset/stride;
                    decode_requests=add(decode_requests,add((x.nbytes+planned.request_bytes-1)/planned.request_bytes,cuts));}}
            decode_layout=was;requests=std::max(requests,decode_requests);waves=std::max(waves,decode_waves);
        }
        require(requests<=UINT64_MAX/(4*512)&&waves<=UINT64_MAX/512,"V4.1 DS request metadata budget overflow");
        planned.host_bytes=add(metadata_allowance+factory_allowance,add(requests*(4*512),waves*512));
        if(options.ds_gpu_ring_slots)planned.host_bytes=add(planned.host_bytes,128*mib);
        if(options.zero_padding_only)planned.host_bytes=add(planned.host_bytes,mib); // bounded8192-span complement/request metadata
        if(options.engram_prefetch_host_bytes){
            require(options.engram_prefetch_tokens&&options.engram_prefetch_tokens<=1048576,
                "Whole-prompt Engram requires an explicit admitted token count");
            const uint64_t worst=add(4*mib,uint64_t(options.engram_prefetch_tokens)*48*(4+264));
            require(worst<=options.engram_prefetch_host_bytes,
                "Whole-prompt Engram maximum row/ID capacity exceeds caller admission");
            planned.host_bytes=add(planned.host_bytes,options.engram_prefetch_host_bytes);
        }else require(!options.engram_prefetch_tokens,"Engram token count requires cache host admission");
    }
    void acquire(uint32_t id){
        require(allocated&&!device_suspended&&(!leased||(dense_held>=0&&uint32_t(dense_held)+1==id))&&next==id&&id<stage_count(),"V4.1 physical stage acquired out of order or while device-suspended");identity();
        if(bound()){   // resident in place: no fetcher handshake
            auto*bank=buffers[id%2].get();auto*at=static_cast<char*>(ggml_backend_buffer_get_base(bank));
            const bool prebound=id>=2&&id<=80&&id%2==0&&experts_prebound;
            for(auto&e:stage_at(id).entries){
                if(prebound){require(e.t->buffer==bank&&e.t->data==at+e.gpu_offset,"Shared-expert stage outside its pre-bound bank");continue;}
                require(ggml_backend_tensor_alloc(bank,e.t,at+e.gpu_offset)==GGML_STATUS_SUCCESS,"Cannot bind resident native tensor");
            }
            if(prebound)experts_prebound=false;
            leased=true;return;
        }
        auto*base=wf_wait_layer_gpu(fetcher,int(id));require(base,"Required DirectStorage layer transfer failed");
        wf_cuda_wait_upload(fetcher,int(id));
        require(!wf_has_failed(fetcher),"Required DirectStorage upload synchronization failed");
        ggml_backend_buffer_t buffer=nullptr;
        for(auto&b:buffers)if(ggml_backend_buffer_get_base(b.get())==base)buffer=b.get();
        require(buffer,"DirectStorage returned an unowned compute allocation");
        const bool prebound_routed=decode_layout&&id>=2&&id<=80&&id%2==0&&experts_prebound;
        for(auto&e:stage_at(id).entries){
            if(prebound_routed){require(e.t->buffer==buffer&&e.t->data==static_cast<char*>(base)+e.gpu_offset,"Shared-expert stage landed outside its pre-bound bank");continue;}
            require(ggml_backend_tensor_alloc(buffer,e.t,static_cast<char*>(base)+e.gpu_offset)==GGML_STATUS_SUCCESS,"Cannot bind staged native tensor");
        }
        if(prebound_routed)experts_prebound=false;
        ggml_backend_synchronize(options.backend);leased=true;
        uint64_t fetched=0;double fetch_ms=0;
        require(wf_get_required_layer_fetch(fetcher,int(id),&fetched,&fetch_ms),"Completed strict DirectStorage timing unavailable");
        fetched_total+=fetched;fetch_ms_total+=fetch_ms;
        llama_dsv41_layer_log("[dsv41-fetch] stage=%u bytes=%llu fetch_ms=%.6f decimal_GB_s=%.6f\n",
            id,(unsigned long long)fetched,fetch_ms,double(fetched)/fetch_ms/1e6);
        std::fflush(stderr);
    }
};

llama_dsv41_weight_provider::llama_dsv41_weight_provider():p(new implementation){}
llama_dsv41_weight_provider::~llama_dsv41_weight_provider()=default;
llama_dsv41_weight_provider::llama_dsv41_weight_provider(llama_dsv41_weight_provider&&) noexcept=default;
llama_dsv41_weight_provider&llama_dsv41_weight_provider::operator=(llama_dsv41_weight_provider&&) noexcept=default;

bool llama_dsv41_weight_provider::inspect(const config&options,std::string&error){return p->guard([&]{
    require(!p->inspected&&!p->allocated,"Weight provider cannot reopen");
#if !defined(_WIN32) || !defined(GGML_USE_CUDA)
    (void)options;throw std::runtime_error("V4.1 physical weight provider requires Windows CUDA/DirectStorage");
#else
    require(options.backend&&options.buffers>=1&&options.buffers<=2,"Invalid physical weight backend/buffer admission");
    require((options.staging_limit_bytes==0&&!options.auto_request_size&&!options.async_staging_drain)||
        (options.auto_request_size&&
            (options.staging_limit_bytes==llama_wf_bounded::small_staging_bytes||
             (options.buffers==2&&options.staging_limit_bytes==llama_wf_bounded::profile_staging_bytes))),
        "Unsupported explicit physical fetch profile");
    require(options.admitted_host_bytes>=metadata_allowance,"Bounded native metadata inspection was not admitted");
    require(!options.ds_gpu_ring_slots||(options.ds_gpu_ring_slots==2&&options.buffers==2&&options.auto_request_size&&options.async_staging_drain&&options.staging_limit_bytes==1500*mib),"GPU ring requires explicit native two-buffer profile");
    require(options.ds_submit_threads==0||options.ds_submit_threads==1||options.ds_submit_threads==2||options.ds_submit_threads==4,
        "DirectStorage submit threads must be explicit0/1/2/4");
    require(!options.zero_padding_only||options.compute_direct,"Padding-only zero requires explicit direct compute heaps");
    require(!options.compute_direct||(options.buffers==2&&options.async_staging_drain&&options.auto_request_size&&!options.ds_gpu_ring_slots),
        "Direct compute requires two typed async banks without a ring");
    require(!std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY"),"Unified CUDA memory is outside strict physical weight admission (including env value zero)");
    require(ggml_backend_is_cuda(options.backend),"V4.1 physical provider requires an actual CUDA backend");
    require(options.verified_sha256==original_sha&&options.verified_file_bytes==518596067328ull,"Original 1017 SHA receipt required");
    const char*device=ggml_backend_dev_description(ggml_backend_get_device(options.backend));
    require(device&&std::strstr(device,"RTX 5090"),"V4.1 streaming producer requires the admitted RTX 5090 backend");
    p->options=options;p->path=std::filesystem::u8path(options.model_path);
    require(p->path.is_absolute()&&p->path==p->path.lexically_normal()&&p->path.has_filename()&&p->path.filename().wstring().find(L':')==std::wstring::npos,"Model path must be a physical absolute file");
    require(GetDriveTypeW(p->path.root_path().c_str())==DRIVE_FIXED,"Model must reside on a fixed local volume");
    wchar_t volume[32768]{};require(QueryDosDeviceW(p->path.root_name().c_str(),volume,32768)&&
        std::wstring(volume).find(L"\\Device\\HarddiskVolume")==0,"Model drive must not be substituted or redirected");
    std::vector<std::filesystem::path>chain;for(auto d=p->path.parent_path();;d=d.parent_path()){chain.push_back(d);if(d==d.root_path())break;}
    std::reverse(chain.begin(),chain.end());
    for(const auto&d:chain){HANDLE h=CreateFileW(d.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);require(h!=INVALID_HANDLE_VALUE,"Cannot pin model directory");p->parents.push_back(h);BY_HANDLE_FILE_INFORMATION s{};require(GetFileInformationByHandle(h,&s)&&(s.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)&&!(s.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"Model ancestor is not a physical directory");}
    p->file=CreateFileW(p->path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);require(p->file!=INVALID_HANDLE_VALUE,"Cannot acquire read-only verified model lease");p->identity();
    ggml_context*raw=nullptr;p->gguf.reset(gguf_init_from_file(options.model_path.c_str(),{true,&raw}));p->tensors.reset(raw);
    require(p->gguf&&p->tensors&&gguf_get_data_offset(p->gguf.get())<=8*mib&&gguf_get_n_tensors(p->gguf.get())<=2048,"Native GGUF metadata exceeds inspected original limits");
    p->identity();p->metadata();p->plan_weights();p->inspected=true;
    DWORD sectors=0,sector_bytes=0,free_clusters=0,clusters=0;
    require(GetDiskFreeSpaceW(p->path.root_path().c_str(),&sectors,&sector_bytes,&free_clusters,&clusters)&&
        sector_bytes&&4096%sector_bytes==0,"Unsupported original Engram direct-I/O sector alignment");
    if(!options.engram_prefetch_host_bytes){
    p->engram_page=static_cast<uint8_t*>(_aligned_malloc(8192,4096));require(p->engram_page,"Cannot allocate bounded Engram direct-I/O page");
    p->engram_file=CreateFileW(p->path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_NO_BUFFERING|FILE_FLAG_RANDOM_ACCESS,nullptr);
    require(p->engram_file!=INVALID_HANDLE_VALUE,"Cannot open original Engram tables without OS caching");
    BY_HANDLE_FILE_INFORMATION original{},direct{};require(GetFileInformationByHandle(p->file,&original)&&
        GetFileInformationByHandle(p->engram_file,&direct)&&original.dwVolumeSerialNumber==direct.dwVolumeSerialNumber&&
        original.nFileIndexHigh==direct.nFileIndexHigh&&original.nFileIndexLow==direct.nFileIndexLow,"Engram descriptor differs from verified model");
    }
#endif
},error);}
bool llama_dsv41_weight_provider::allocate(std::string&error){return p->guard([&]{
    require(p->inspected&&!p->allocated,"Inspect/admit native weights before allocating");p->identity();
    require(p->options.admitted_device_bytes>=p->planned.device_bytes&&p->options.admitted_host_bytes>=p->planned.host_bytes,"Physical weight provider exceeds admitted device/host ceiling");
    p->create_pipeline();
},error);}
// DS4COMBO fast switch: release every DS heap/imported buffer of the existing
// pipeline while the provider keeps its model lease, metadata, stage plan and
// Engram state; resume rebuilds the identical pipeline through the same path.
bool llama_dsv41_weight_provider::suspend_device(std::string&error){return p->guard([&]{
    require(p->allocated&&!p->leased&&!p->device_suspended&&(p->next==0||p->next==p->stage_count()),"Device suspend requires an allocated idle provider between passes");
    ggml_backend_synchronize(p->options.backend);
    wf_stop(p->fetcher);wf_quiesce(p->fetcher);require(!wf_has_failed(p->fetcher),"Provider could not quiesce before device suspend");
    wf_destroy(p->fetcher);p->fetcher=nullptr;
    // Same release as the graceful close: the global DS/D3D12 cache goes too.
    require(wf_persist_shutdown("keep-alive suspend")==1,"DirectStorage/D3D12 cache could not be released on suspend");
    for(auto&s:p->stages)for(auto&e:s.entries){e.t->data=nullptr;e.t->buffer=nullptr;}
    p->buffers.clear();p->device_suspended=true;
    llama_dsv41_layer_log("DS4COMBO_PROVIDER_SUSPENDED buffers=%u compute_buffer_bytes=%llu model_lease_kept=1\n",p->options.buffers,(unsigned long long)p->planned.compute_buffer_bytes);
},error);}
bool llama_dsv41_weight_provider::resume_device(std::string&error){return p->guard([&]{
    require(p->allocated&&p->device_suspended&&!p->fetcher&&p->buffers.empty(),"Device resume requires a suspended provider");
    p->identity();p->create_pipeline();p->device_suspended=false;
    llama_dsv41_layer_log("DS4COMBO_PROVIDER_RESUMED buffers=%u compute_buffer_bytes=%llu same_pipeline_builder=1\n",p->options.buffers,(unsigned long long)p->planned.compute_buffer_bytes);
},error);}
bool llama_dsv41_weight_provider::device_suspended()const{std::lock_guard<std::mutex>lock(p->mutex);return p->device_suspended;}
void llama_dsv41_weight_provider::implementation::create_pipeline(){
    auto*p=this;
    wf_params params=wf_params_default();params.model_path=p->options.model_path.c_str();params.gpu_backend=p->options.backend;params.mode=WF_MODE_DSTORAGE;
    params.n_layers=int(p->stage_count());params.n_gpu_buffers=int(p->options.buffers);params.max_layer_bytes=p->planned.maximum_stage_bytes;params.n_ram_buffers=0;
    params.io_block_size=size_t(p->planned.request_bytes);params.ds_staging_limit_bytes=size_t(p->options.staging_limit_bytes);
    params.ds_async_drain=p->options.async_staging_drain;
    params.ds_submit_threads=p->options.ds_submit_threads;
    params.ds_gpu_ring_slots=p->options.ds_gpu_ring_slots;
    params.ds_compute_direct=p->options.compute_direct;
    params.ds_zero_padding_only=p->options.zero_padding_only;
    p->fetcher=wf_create_required_dstorage(&params);require(p->fetcher,"Required original weight DirectStorage initialization failed");
    p->allocated=true; // All subsequent failures must synchronize before free.
    for(uint32_t i=0;i<p->options.buffers;++i){
        if(p->options.compute_direct)require(wf_is_ds_direct(p->fetcher)&&wf_get_n_gpu_bufs(p->fetcher)==int(p->options.buffers)&&wf_get_gpu_buf_ptr(p->fetcher,int(i)),"Required direct bank was not allocated; no relay fallback");
        ggml_backend_buffer_ptr b(p->options.compute_direct?
            ggml_backend_cuda_buffer_from_ptr(wf_get_cuda_device_index(p->fetcher),wf_get_gpu_buf_ptr(p->fetcher,int(i)),p->planned.compute_buffer_bytes):
            ggml_backend_alloc_buffer(p->options.backend,p->planned.compute_buffer_bytes));
        require(bool(b),"Admitted weight buffer allocation/wrapping failed");ggml_backend_buffer_set_usage(b.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(!p->options.compute_direct)wf_set_gpu_buf_ptr(p->fetcher,int(i),ggml_backend_buffer_get_base(b.get()));p->buffers.push_back(std::move(b));
    }
    for(uint32_t i=0;i<p->stage_count();++i){std::vector<wf_weight_span>spans;p->spans_for(p->stage_at(i),spans);wf_register_layer(p->fetcher,int(i),spans.data(),int(spans.size()),p->stage_at(i).bytes);}
    if(p->decode_layout){
        for(uint32_t i=0;i<82;++i)require(wf_set_layer_persistent(p->fetcher,int(i),true),"Resident decode stage refused by the fetcher");
        require(p->buffers.size()==2,"Decode expert cache requires the two direct banks");
        auto*bank=p->buffers[0].get();auto*base=static_cast<char*>(ggml_backend_buffer_get_base(bank));
        for(uint32_t k=0;k<3;++k)require(ggml_backend_tensor_alloc(bank,p->pool[k],base+p->pool_offset[k])==GGML_STATUS_SUCCESS,"Cannot bind the expert-cache pool");
        p->reset_cache();
        std::fprintf(stderr,"LLAMADECODE_DECODE_LAYOUT stages=82 all_resident=1 bank0_resident_bytes=%llu bank1_resident_bytes=%llu expert_cache_slots=%u slot_bytes=%llu pool_bytes=%llu\n",
            (unsigned long long)p->decode_bank0_resident,(unsigned long long)p->resident_dense_bytes,p->pool_slots,
            (unsigned long long)(p->pool_stride[0]+p->pool_stride[1]+p->pool_stride[2]),(unsigned long long)(uint64_t(p->pool_slots)*(p->pool_stride[0]+p->pool_stride[1]+p->pool_stride[2])));
    }else{
        for(auto*t:p->pool)if(t){t->data=nullptr;t->buffer=nullptr;}
        // Owner gate: prefill must stream through BOTH banks again after decode repurposed them
        // (bank 0 expert cache, bank 1 resident dense). Fail loudly rather than run single-banked.
        require(p->buffers.size()==p->options.buffers,"Prefill layout lost a streaming bank");
        if(p->options.compute_direct&&p->options.buffers==2)
            std::fprintf(stderr,"LLAMADECODE_PREFILL_LAYOUT banks=%zu streaming_banks=%u persistent_stages=0 expert_pool_bound=0\n",p->buffers.size(),p->options.buffers);
    }
    require(!wf_has_failed(p->fetcher),"Original layer registration failed");p->identity();if(p->options.ds_gpu_ring_slots)wf_set_layer_major_mode(p->fetcher,true);wf_start(p->fetcher);
    require(!wf_has_failed(p->fetcher),"Original DirectStorage pipeline startup failed");
}
bool llama_dsv41_weight_provider::open(const config&options,std::string&error){return inspect(options,error)&&allocate(error);}
bool llama_dsv41_weight_provider::restart_prompt(const config&options,std::string&error){return p->guard([&]{
    require(p->allocated&&!p->device_suspended&&!p->leased&&p->next==p->stage_count()&&!p->options.ds_gpu_ring_slots,"Only a fully released non-ring resident provider pass can restart");
    auto binding=[](const config&v){return std::tie(v.model_path,v.verified_file_bytes,v.verified_volume,v.verified_mtime_ns,v.verified_file_id,
        v.verified_sha256,v.backend,v.admitted_device_bytes,v.admitted_host_bytes,v.buffers,v.staging_limit_bytes,v.auto_request_size,
        v.async_staging_drain,v.compute_direct,v.zero_padding_only,v.engram_prefetch_host_bytes,v.ds_submit_threads,v.ds_gpu_ring_slots);};
    require(binding(options)==binding(p->options),"Persistent provider cannot change model/backend/geometry/admission");
    require(!options.engram_prefetch_host_bytes||(options.engram_prefetch_tokens&&
        4ull*1024*1024+uint64_t(options.engram_prefetch_tokens)*48*(4+264)<=options.engram_prefetch_host_bytes),"Restarted prompt exceeds admitted immutable row-cache capacity");
    p->identity();wf_stop(p->fetcher);wf_quiesce(p->fetcher);require(!wf_has_failed(p->fetcher),"Completed provider could not quiesce before request reuse");
    if(options.decode_layout!=p->decode_layout){
        // LLAMADECODE B1: rebuild the same admitted pipeline with the other stage
        // layout. Same banks/staging/admission; only stage registration differs.
        require(!options.decode_layout||(p->options.compute_direct&&p->options.buffers==2&&!p->options.zero_padding_only),
            "Decode layout requires the strict two-bank direct fetch profile");
        const auto started=std::chrono::steady_clock::now();
        p->release_l2();   // LLAMADECODE: the decode-only L2 returns its VRAM before the prefill plan is restored
        wf_destroy(p->fetcher);p->fetcher=nullptr;
        for(auto&s:p->stages)for(auto&e:s.entries){e.t->data=nullptr;e.t->buffer=nullptr;}
        p->buffers.clear();p->decode_layout=options.decode_layout;p->experts_prebound=false;p->dense_held=-1;p->resident_bound=false;p->create_pipeline();
        wf_stop(p->fetcher);wf_quiesce(p->fetcher);require(!wf_has_failed(p->fetcher),"Switched provider could not quiesce before request start");
        std::fprintf(stderr,"LLAMADECODE_LAYOUT_SWITCH decode=%d switch_ms=%.3f same_admission=1\n",int(p->decode_layout),
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
    }
    if(p->early&&p->early->adopted)p->join_early();
    p->engram_prompt.reset();p->options.engram_prefetch_tokens=options.engram_prefetch_tokens;p->next=0;
    for(const auto&s:p->stages)for(const auto&e:s.entries)require(!e.t->data&&!e.t->buffer,"Completed pass retained a stale staged tensor");
    llama_dsv41_layer_log("DS4PERF_PROVIDER_REUSED buffers=%u compute_buffer_bytes=%llu model_lease=1 heaps=1 request_arrays=1 starts_in_timed_run=1\n",p->options.buffers,(unsigned long long)p->planned.compute_buffer_bytes);
},error);}
bool llama_dsv41_weight_provider::start_request(std::string&error){return p->guard([&]{
    require(p->allocated&&!p->device_suspended&&!p->leased&&p->next==0,"Request start requires an admitted resident unleased first stage");p->identity();
    p->fetched_total=0;p->fetch_ms_total=0;if(!p->bound())wf_start(p->fetcher);
    require(!wf_has_failed(p->fetcher),"Persistent DirectStorage restart failed");
},error);}
llama_dsv41_weight_provider::capacity llama_dsv41_weight_provider::budget()const{std::lock_guard<std::mutex>lock(p->mutex);return p->inspected&&!p->poisoned?p->planned:capacity{};}
ggml_backend_t llama_dsv41_weight_provider::backend()const{std::lock_guard<std::mutex>lock(p->mutex);return p->allocated&&!p->poisoned?p->options.backend:nullptr;}
bool llama_dsv41_weight_provider::prepare_engram_prompt(const std::vector<int32_t>&tokens,const std::function<bool()>&stop,std::string&error,llama_dsv41::engram_history initial){return p->guard([&]{
    if(!p->options.engram_prefetch_host_bytes)return;
    const char*overlap=std::getenv("LLAMA_DSV41_ENGRAM_PROMPT_OVERLAP");
    const bool during_layer_zero=overlap&&std::strcmp(overlap,"1")==0&&p->leased&&p->next==1;
    require(p->allocated&&((!p->leased&&p->next==0)||during_layer_zero)&&!p->engram_prompt&&tokens.size()==p->options.engram_prefetch_tokens,
        "Engram whole-prompt cache requires its initial or explicitly overlapped layer-zero lease");p->identity();
    if(p->early&&p->early->tokens==tokens){
        require(!p->early->adopted,"Early Engram prompt cache was adopted twice");
        if(overlap&&std::strcmp(overlap,"1")==0){
            // Overlap: adopt the still-loading cache now. Readers wait per table
            // in wait_engram_table, which also checks admission and joins.
            require(p->early->initial.tail==initial.tail,"Early Engram prompt cache was built with a different restored history");
            p->engram_prompt=p->early->cache;p->early->adopted=true;
            llama_dsv41_layer_log("DS4PERF_ENGRAM_EARLY_ADOPTED tokens=%zu deferred=1 per_table_ready=1 same_rows=1\n",tokens.size());
            return;
        }
        const auto waited=std::chrono::steady_clock::now();
        if(p->early_thread.joinable())p->early_thread.join();
        auto adopted=std::move(p->early);
        if(adopted->failure)std::rethrow_exception(adopted->failure);
        require(adopted->initial.tail==initial.tail,"Early Engram prompt cache was built with a different restored history");
        require(adopted->cache&&adopted->cache->budget()<=p->options.engram_prefetch_host_bytes,"Early Engram prompt cache exceeds provider admission");
        p->engram_prompt=std::move(adopted->cache);
        llama_dsv41_layer_log("DS4PERF_ENGRAM_EARLY_ADOPTED tokens=%zu build_wall_ms=%.3f join_wait_ms=%.3f host_plan_bytes=%llu same_rows=1\n",tokens.size(),adopted->wall_ms,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-waited).count(),(unsigned long long)p->engram_prompt->budget());
        return;
    }
#ifdef _WIN32
    p->engram_prompt.reset(new llama_dsv41_engram_prompt_cache);
    p->engram_prompt->prepare(p->engram,tokens.data(),uint32_t(tokens.size()),p->engram_offsets,p->options.engram_prefetch_host_bytes,initial);
    require(p->engram_prompt->budget()<=p->options.engram_prefetch_host_bytes,"Prepared Engram cache exceeds provider admission");
    p->engram_prompt->load(p->path.c_str(),p->file,p->options.verified_file_bytes,stop);p->identity();
#else
    (void)stop;throw std::runtime_error("Whole-prompt Engram IoRing requires Windows");
#endif
},error);}
bool llama_dsv41_weight_provider::start_engram_prompt_early(const std::vector<int32_t>&tokens,llama_dsv41::engram_history initial,std::string&error,bool current_request){return p->guard([&]{
#ifdef _WIN32
    if(current_request&&p->early){
        // A producer-warm build for a request is already queued; prepare adopts
        // it when its tokens match, else takes the ordinary load.
        llama_dsv41_layer_log("DS4PERF_ENGRAM_EARLY_ALREADY_QUEUED tokens=%zu same_tokens=%d\n",tokens.size(),int(p->early->tokens==tokens));return;
    }
    require(p->options.engram_prefetch_host_bytes&&p->allocated&&!p->early&&!tokens.empty()&&tokens.size()<=1048576,"Early Engram prompt cache requires an admitted row cache and one pending request");
    // Both caches may be alive at once: the current pass's own rows and the
    // early request's. Charge both worst cases against the one admission.
    // For the current request itself (sweep start) only its own cache exists.
    const uint64_t worst=add(4*mib,uint64_t(tokens.size())*48*(4+264));
    const uint64_t current=(current_request||p->options.engram_early_is_next_request)?0:add(4*mib,uint64_t(p->options.engram_prefetch_tokens)*48*(4+264));
    if(add(worst,current)>p->options.engram_prefetch_host_bytes){
        // Not admitted alongside the current pass: the request keeps its
        // ordinary layer-zero overlapped load. Visible, never silent.
        llama_dsv41_layer_log("DS4PERF_ENGRAM_EARLY_SKIPPED tokens=%zu worst_bytes=%llu current_bytes=%llu admitted_bytes=%llu reason=host_admission\n",tokens.size(),
            (unsigned long long)worst,(unsigned long long)current,(unsigned long long)p->options.engram_prefetch_host_bytes);
        return;
    }
    p->identity();
    auto e=std::make_shared<implementation::early_engram>();e->tokens=tokens;e->initial=initial;e->cache.reset(new llama_dsv41_engram_prompt_cache);
    const auto layout=p->engram;const auto offsets=p->engram_offsets;const auto path=p->path;const HANDLE file=p->file;
    const uint64_t file_bytes=p->options.verified_file_bytes,admitted=p->options.engram_prefetch_host_bytes;
    p->early=e;
    p->early_thread=std::thread([e,layout,offsets,path,file,file_bytes,admitted]{
        const auto started=std::chrono::steady_clock::now();
        try{
            e->cache->prepare(layout,e->tokens.data(),uint32_t(e->tokens.size()),offsets,admitted,e->initial);
            e->cache->load(path.c_str(),file,file_bytes,[e]{return e->cancel.load();});
        }catch(...){e->failure=std::current_exception();}
        e->wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    });
    llama_dsv41_layer_log("DS4PERF_ENGRAM_EARLY_QUEUED tokens=%zu worst_bytes=%llu current_bytes=%llu admitted_bytes=%llu own_ioring=1\n",tokens.size(),
        (unsigned long long)worst,(unsigned long long)current,(unsigned long long)p->options.engram_prefetch_host_bytes);
#else
    (void)tokens;(void)initial;throw std::runtime_error("Early Engram IoRing requires Windows");
#endif
},error);}
bool llama_dsv41_weight_provider::wait_engram_table(uint32_t index,const std::function<bool()>&stop,double&waited_ms,std::string&error){
    waited_ms=0;std::shared_ptr<llama_dsv41_engram_prompt_cache>cache;
    if(!p->guard([&]{require(index<2&&p->engram_prompt,"Engram table wait has no prepared prompt cache");cache=p->engram_prompt;},error))return false;
    // Outside the provider lock: weight acquire/release and the other
    // table's reads proceed while this table's rows finish loading.
    const auto started=std::chrono::steady_clock::now();std::string failed;
    try{cache->wait_table(index,stop);}catch(const std::exception&e){failed=e.what();}
    waited_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return p->guard([&]{
        require(p->engram_prompt==cache,"Engram prompt cache changed while waiting for a table");
        if((index==1||!failed.empty())&&p->early&&p->early->adopted&&p->early->cache==cache){
            if(!failed.empty())p->early->cancel=true;
            if(p->early_thread.joinable())p->early_thread.join();
            auto adopted=std::move(p->early);
            if(adopted->failure)std::rethrow_exception(adopted->failure);
            llama_dsv41_layer_log("DS4PERF_ENGRAM_EARLY_JOINED tokens=%zu build_wall_ms=%.3f host_plan_bytes=%llu\n",adopted->tokens.size(),adopted->wall_ms,(unsigned long long)cache->budget());
        }
        require(failed.empty(),failed.c_str());
        require(cache->budget()<=p->options.engram_prefetch_host_bytes,"Early Engram prompt cache exceeds provider admission");
    },error);
}
bool llama_dsv41_weight_provider::acquire_embeddings(ggml_tensor*&out,std::string&error){out=nullptr;return p->guard([&]{p->acquire(0);out=p->embedding;},error);}
bool llama_dsv41_weight_provider::acquire_layer(uint32_t layer,llama_dsv41_layer_graph::weights&out,std::string&error){out={};return p->guard([&]{require(layer<40&&!p->decode_layout,"Invalid physical layer");p->acquire(layer+1);out=p->layers[layer];},error);}
bool llama_dsv41_weight_provider::acquire_head(ggml_tensor*&norm,ggml_tensor*&out,std::string&error){norm=nullptr;out=nullptr;return p->guard([&]{p->acquire(p->stage_count()-1);norm=p->head_norm;out=p->head_weight;},error);}
bool llama_dsv41_weight_provider::decode_layout()const{std::lock_guard<std::mutex>lock(p->mutex);return p->decode_layout;}
void llama_dsv41_weight_provider::fetch_totals(uint64_t&bytes,double&milliseconds)const{std::lock_guard<std::mutex>lock(p->mutex);bytes=p->fetched_total;milliseconds=p->fetch_ms_total;}
bool llama_dsv41_weight_provider::decode_banks(uint32_t layer,const void*&dense,const void*&routed)const{
    std::lock_guard<std::mutex>lock(p->mutex);dense=routed=nullptr;
    if(!p->decode_layout||layer>=40||p->buffers.size()!=2)return false;
    dense=p->buffers[(1+2*layer)%2].get();routed=p->buffers[(2+2*layer)%2].get();return true;
}
bool llama_dsv41_weight_provider::acquire_layer_dense(uint32_t layer,llama_dsv41_layer_graph::weights&out,std::string&error){out={};return p->guard([&]{
    require(layer<40&&p->decode_layout&&p->buffers.size()==2&&p->options.compute_direct,"Dense decode stage requires the two-bank decode layout");
    p->acquire(1+2*layer);
    // The shared-expert stage's bank region is fixed (resident, bank 0); bind it
    // now so the layer graph allocates once. Its bytes are present before any
    // post-router node runs (acquire_layer_experts).
    const uint32_t id=2+2*layer;auto*bank=p->buffers[id%2].get();auto*base=static_cast<char*>(ggml_backend_buffer_get_base(bank));
    require(base==wf_get_gpu_buf_ptr(p->fetcher,int(id%2)),"Shared-expert bank pointer differs from its DirectStorage bank");
    for(auto&e:p->decode_stages[id].entries){require(!e.t->buffer&&!e.t->data,"Shared-expert tensor already bound");
        require(ggml_backend_tensor_alloc(bank,e.t,base+e.gpu_offset)==GGML_STATUS_SUCCESS,"Cannot pre-bind shared-expert tensor");}
    p->experts_prebound=true;out=p->layers[layer];
    out.ffn.gate=p->pool[0];out.ffn.up=p->pool[1];out.ffn.down=p->pool[2];
},error);}
bool llama_dsv41_weight_provider::acquire_layer_experts(uint32_t layer,const std::vector<int32_t>&ids,std::vector<int32_t>&slots,std::string&error){slots.clear();return p->guard([&]{
    require(layer<40&&p->decode_layout&&p->leased&&p->next==1+2*layer&&p->experts_prebound,"Routed experts require their leased dense stage");
    require(!ids.empty()&&ids.size()<=6*4096,"Routed expert set size invalid");
    const auto slot_of=p->assign_expert_slots(layer,ids.data(),ids.size(),ggml_backend_cuda_get_stream(p->options.backend));
    // The shared-expert stage (resident) is held with the dense stage through the layer.
    p->dense_held=int(1+2*layer);++p->next;
    p->acquire(2+2*layer);
    slots.reserve(ids.size());for(const auto id:ids)slots.push_back(slot_of.at(id));
},error);}
bool llama_dsv41_weight_provider::prepare_external_decode(const config&options,std::string&error){
    if(!restart_prompt(options,error))return false;   // switches to the decode layout (options.decode_layout)
    return p->guard([&]{
        require(p->decode_layout&&p->buffers.size()==2,"External decode requires the two-bank decode layout");
        require(p->allocated&&!p->device_suspended&&!p->leased&&p->next==0,"External decode requires an idle provider");
        wf_start(p->fetcher);require(!wf_has_failed(p->fetcher),"Resident pass start failed");p->fetched_total=0;p->fetch_ms_total=0;
        const auto started=std::chrono::steady_clock::now();
        // One pass over all stages: every non-routed stage becomes resident in its own bank region.
        for(uint32_t id=0;id<p->stage_count();++id){
            p->acquire(id);ggml_backend_synchronize(p->options.backend);
            for(auto&e:p->stage_at(id).entries){e.t->data=nullptr;e.t->buffer=nullptr;}
            wf_release_layer(p->fetcher,int(id));require(!wf_has_failed(p->fetcher),"Resident pass release failed");p->leased=false;++p->next;
        }
        p->resident_bound=true;
        std::fprintf(stderr,"LLAMADECODE_EXTERNAL_RESIDENT stages=%u bytes=%llu ms=%.3f\n",p->stage_count(),(unsigned long long)p->fetched_total,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
    },error);
}
bool llama_dsv41_weight_provider::decode_resident_tensors(const std::function<bool(uint64_t,uint64_t,void*)>&visit,std::string&error)const{
    std::lock_guard<std::mutex>lock(p->mutex);error.clear();
    try{
        require(p->decode_layout&&p->buffers.size()==2&&!p->leased&&p->next==p->stage_count(),"Resident tensors require a completed decode-layout pass");
        for(uint32_t id=0;id<p->stage_count();++id){auto*base=static_cast<char*>(ggml_backend_buffer_get_base(p->buffers[id%2].get()));
            for(const auto&e:p->decode_stages[id].entries)require(visit(e.file_offset,ggml_nbytes(e.t),base+e.gpu_offset),"External decoder refused a resident tensor");}
        return true;
    }catch(const std::exception&failure){error=failure.what();return false;}
}
bool llama_dsv41_weight_provider::external_expert_slots(uint32_t layer,const int32_t*ids,uint32_t n,int32_t*slots,
        const char**gate,const char**up,const char**down,uint32_t*slot_count,void*stream,std::string&error){return p->guard([&]{
    require(layer<40&&p->decode_layout&&!p->leased&&p->next==p->stage_count()&&ids&&slots&&n&&n<=16,"External expert slots require an idle decode-layout provider");
    const auto slot_of=p->assign_expert_slots(layer,ids,n,stream);
    for(uint32_t i=0;i<n;++i)slots[i]=slot_of.at(ids[i]);
    *gate=static_cast<const char*>(p->pool[0]->data);*up=static_cast<const char*>(p->pool[1]->data);*down=static_cast<const char*>(p->pool[2]->data);
    *slot_count=p->pool_slots;
},error);}
llama_dsv41_weight_provider::expert_cache_stats llama_dsv41_weight_provider::cache_stats()const{std::lock_guard<std::mutex>lock(p->mutex);
    expert_cache_stats s;s.slots=p->pool_slots;s.hits=p->cache_hits;s.misses=p->cache_misses;s.fetched_bytes=p->cache_bytes;s.fetch_ms=p->cache_ms;
    s.l2_slots=p->l2_slots;s.l2_hits=p->l2_hits;s.l2_stored=p->l2_stored;return s;}
bool llama_dsv41_weight_provider::decode_residency(uint32_t layer,uint8_t*tier,uint32_t experts)const{
    std::lock_guard<std::mutex>lock(p->mutex);
    if(!tier||experts!=384||layer>=40||!p->decode_layout)return false;
    std::memset(tier,0,experts);
    for(const auto key:p->slot_key)if(key>=0&&uint32_t(key)/384==layer)tier[key%384]=1;
    for(const auto key:p->l2_key)if(key>=0&&uint32_t(key)/384==layer)tier[key%384]=1;
    return true;
}
uint64_t llama_dsv41_weight_provider::decode_cache_slot_bytes()const{return p->pool_stride[0]+p->pool_stride[1]+p->pool_stride[2];}
bool llama_dsv41_weight_provider::grow_decode_cache(uint64_t bytes,uint32_t*slots,std::string&error){return p->guard([&]{
    require(slots&&p->decode_layout&&p->pool[0]&&p->pool[0]->data&&!p->l2_buffer&&!p->leased,"Expert L2 requires an idle decode-layout provider without an L2");
    const uint64_t slot_bytes=p->pool_stride[0]+p->pool_stride[1]+p->pool_stride[2];
    const uint32_t n=uint32_t(std::min<uint64_t>(bytes/slot_bytes,40ull*384-p->pool_slots));*slots=0;
    if(n<2){std::fprintf(stderr,"LLAMADECODE_L2 slots=0 offered_bytes=%llu\n",(unsigned long long)bytes);return;}
    const auto buft=ggml_backend_get_default_buffer_type(p->options.backend);
    auto*buffer=ggml_backend_buft_alloc_buffer(buft,size_t(uint64_t(n)*slot_bytes));
    require(buffer,"Expert L2 VRAM allocation failed");p->l2_buffer.reset(buffer);
    auto*base=static_cast<char*>(ggml_backend_buffer_get_base(buffer));
    p->l2_base={base,base+uint64_t(n)*p->pool_stride[0],base+uint64_t(n)*(p->pool_stride[0]+p->pool_stride[1])};
    p->l2_slots=n;p->l2_key.assign(n,-1);p->l2_of.clear();p->l2_lru.clear();p->l2_pos.assign(n,p->l2_lru.end());
    for(uint32_t s=0;s<n;++s){p->l2_lru.push_back(int32_t(s));p->l2_pos[s]=std::prev(p->l2_lru.end());}
    *slots=n;
    std::fprintf(stderr,"LLAMADECODE_L2 slots=%u bytes=%llu l1_slots=%u total_vram_slots=%u offered_bytes=%llu\n",n,
        (unsigned long long)(uint64_t(n)*slot_bytes),p->pool_slots,p->pool_slots+n,(unsigned long long)bytes);
},error);}
bool llama_dsv41_weight_provider::release(std::string&error){return p->guard([&]{require(p->leased&&p->next<p->stage_count(),"No physical weight lease to release");ggml_backend_synchronize(p->options.backend);
    if(p->dense_held>=0){ // LLAMADECODE B1: release the held dense stage first (stage order)
        for(auto&e:p->decode_stages[size_t(p->dense_held)].entries){e.t->data=nullptr;e.t->buffer=nullptr;}
        if(!p->bound())wf_release_layer(p->fetcher,p->dense_held);require(!wf_has_failed(p->fetcher),"Dense stage release failed");p->dense_held=-1;}for(auto&e:p->stage_at(p->next).entries){e.t->data=nullptr;e.t->buffer=nullptr;}if(!p->bound())wf_release_layer(p->fetcher,int(p->next));require(!wf_has_failed(p->fetcher),"Required DirectStorage compute release failed");p->leased=false;++p->next;
    if(p->decode_layout&&p->next==p->stage_count()&&!p->resident_bound&&p->bind_once_enabled()){p->resident_bound=true;std::fprintf(stderr,"LLAMADECODE_BIND_ONCE stages=%u resident_in_place=1\n",p->stage_count());}},error);}
void llama_dsv41_weight_provider::invalidate(){if(!p)return;std::lock_guard<std::mutex>lock(p->mutex);p->invalidate();}
bool llama_dsv41_weight_provider::advance_decoded_engram(llama_dsv41::engram_history&history,
        const int32_t*tokens,size_t count,std::string&error){return p->guard([&]{
    require(p->allocated&&!p->leased&&p->next==p->stage_count()&&count&&count<=128,
        "Decoded Engram history requires a released pass and1..128 evaluated tokens");p->identity();
    auto pending=history;std::array<llama_dsv41::engram_rows,128>rows{};std::string why;
    require(llama_dsv41::engram_hash(p->engram,pending,tokens,nullptr,count,rows.data(),why),
        "Decoded Engram token/history validation failed");history=pending;
},error);}
bool llama_dsv41_weight_provider::restored_engram_history(llama_dsv41::engram_history&history,
        const int32_t*tokens,size_t count,std::string&error){return p->guard([&]{
    require(p->allocated&&!p->leased&&p->next==0&&tokens&&count&&count<=3,
        "Restored Engram history requires an idle initial provider and1..3 checkpoint tokens");p->identity();
    llama_dsv41::engram_history pending;std::array<llama_dsv41::engram_rows,3>rows{};std::string why;
    require(llama_dsv41::engram_hash(p->engram,pending,tokens,nullptr,count,rows.data(),why),
        "Restored Engram token/history validation failed");history=pending;
},error);}
bool llama_dsv41_weight_provider::engram_tile(uint32_t index,llama_dsv41::engram_history&history,const int32_t*tokens,const uint8_t*mask,size_t count,std::vector<float>&out,std::string&error){out.clear();return p->guard([&]{
    // A prepared whole-prompt cache owns these immutable encoded rows already.
    // CPU decode may run during the preceding layer without leasing its weights.
    // The unbuffered file fallback retains the original exact layer lease.
    // Engram layers 1 and 14: their (dense) stage id depends on the active layout.
    const uint32_t engram_layer=index==0?1u:14u,engram_stage=p->decode_layout?1u+2u*engram_layer:engram_layer+1u;
    require(p->allocated&&index<2&&count&&count<=4096&&
        ((p->engram_prompt&&p->next<=engram_stage)||
         (p->leased&&p->next==engram_stage)),"Engram read has no prepared prompt cache or physical layer lease");p->identity();
    auto pending=history;std::vector<llama_dsv41::engram_rows>rows(count);std::string why;require(llama_dsv41::engram_hash(p->engram,pending,tokens,mask,count,rows.data(),why),"Native Engram hash failed");
    std::vector<uint32_t>ids(count*24);for(size_t t=0;t<count;++t)std::copy(rows[t][index].begin(),rows[t][index].end(),ids.begin()+t*24);
    std::vector<float>result(count*6144);
    auto read=[&](uint64_t offset,void*dst,size_t bytes){
        if(p->options.engram_prefetch_host_bytes)return p->engram_prompt&&p->engram_prompt->read(index,offset,dst,bytes);
#ifdef _WIN32
        if(!bytes||bytes>264)return false;
        const uint64_t begin=offset&~uint64_t(4095),skip=offset-begin,extent=align(skip+bytes,4096);
        if(extent>8192||begin>p->options.verified_file_bytes||extent>p->options.verified_file_bytes-begin)return false;
        LARGE_INTEGER where;where.QuadPart=LONGLONG(begin);DWORD done=0;
        if(!SetFilePointerEx(p->engram_file,where,nullptr,FILE_BEGIN)||!ReadFile(p->engram_file,p->engram_page,DWORD(extent),&done,nullptr)||done!=extent)return false;
        std::memcpy(dst,p->engram_page+skip,bytes);return true;
#else
        (void)offset;(void)dst;(void)bytes;return false;
#endif
    };
    // DS4COMBO producer-warm: rows come from the immutable prepared cache, so
    // contiguous token ranges decode on separate threads into their own fixed
    // output rows. Same reader, same decode, same positions: identical floats.
    constexpr size_t decode_threads=8;
    if(p->options.engram_prefetch_host_bytes&&count>=1024){
        std::array<std::string,decode_threads>why_part;std::array<bool,decode_threads>ok{};std::vector<std::thread>workers;
        const size_t per=(count+decode_threads-1)/decode_threads;
        const llama_dsv41::engram_read_at reader=read;
        const uint64_t file_bytes=p->options.verified_file_bytes,table_offset=p->engram_offsets[index];const uint32_t table_rows=p->engram.rows[index];
        const uint32_t*id_base=ids.data();float*out_base=result.data();bool*ok_base=ok.data();std::string*why_base=why_part.data();
        for(size_t w=0;w<decode_threads;++w){const size_t first=w*per;if(first>=count)break;const size_t n=std::min(per,count-first);
            workers.emplace_back([&reader,file_bytes,table_offset,table_rows,id_base,out_base,ok_base,why_base,w,first,n]{
                try{ok_base[w]=llama_dsv41::engram_read_batch(reader,file_bytes,table_offset,table_rows,id_base+first*24,n,24,out_base+first*6144,why_base[w]);}
                catch(...){ok_base[w]=false;}});}
        for(auto&t:workers)t.join();
        for(size_t w=0;w<workers.size();++w)require(ok[w],"Original Engram row read/decode failed");
    }else require(llama_dsv41::engram_read_batch(read,p->options.verified_file_bytes,p->engram_offsets[index],p->engram.rows[index],ids.data(),count,24,result.data(),why),"Original Engram row read/decode failed");
    p->identity();out=std::move(result);history=pending;
},error);}
