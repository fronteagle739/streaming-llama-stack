#include "llama-dsv41-layer-log.h"
#include "llama-dsv41-test-fault.h"
#include "llama-dsv41-sweep.h"
#include "llama-dsv41-append-schedule.h"
#include "llama-dsv41-engram-prefetch.h"
#include <cstdlib>

#include "llama-dsv41-cache.h"
#include "llama-dsv41-device-admission.h"
#include "llama-dsv41-host-plan-log.h"
#include "llama-dsv41-executor.h"
#include "boundary-capture.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <future>
#include <stdexcept>
#include <tuple>

namespace {
constexpr uint32_t width=5120,copies=4,vocabulary=129280,layers=40,tile_limit=32;
constexpr uint64_t metadata_bytes=8ull*1024*1024;
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
void success(bool value,const std::string &error){if(!value)throw std::runtime_error(error.empty()?"V4.1 sweep operation failed":error);}
bool finite_bits(const float &value){
    static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"V4.1 requires IEEE F32");
    uint32_t bits;std::memcpy(&bits,&value,4);return (bits&0x7f800000u)!=0x7f800000u;
}
void finite(const std::vector<float>&v){require(std::all_of(v.begin(),v.end(),finite_bits),"Nonfinite completed V4.1 sweep activation");}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,"V4.1 sweep admission overflow");return a+b;}
bool query_source(uint32_t layer){return layer==2||layer==8||layer==14||layer==20||layer==24||layer==28||layer==32||layer==36;}
std::array<int32_t,40> native_owners(){
    std::array<int32_t,40>out{};
    for(uint32_t i=0;i<40;++i)out[i]=i<2?-1:i<8?2:i<14?8:i<20?14:20;
    return out;
}
template<typename T>void upload(ggml_tensor*t,const std::vector<T>&v){
    require(t&&ggml_nbytes(t)==v.size()*sizeof(T),"V4.1 sweep input geometry changed");
    ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(T));
}
std::vector<float> read_f32(ggml_tensor*t,size_t expected){
    require(t&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t)&&size_t(ggml_nelements(t))==expected,
        "V4.1 sweep output geometry changed");
    std::vector<float>out(expected);ggml_backend_tensor_get(t,out.data(),0,expected*sizeof(float));finite(out);return out;
}
void check_graph(ggml_backend_t backend,ggml_cgraph*graph){
    auto tensor=[&](const ggml_tensor*t){if(t&&t->buffer)
        require(ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(t->buffer)),"V4.1 sweep tensor is on another backend");};
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){
        const auto*node=ggml_graph_node(graph,i);tensor(node);for(auto*src:node->src)tensor(src);
        require(ggml_backend_supports_op(backend,node),"V4.1 sweep graph operation is unsupported");
    }
}
ggml_backend_buffer_ptr allocate_graph(ggml_context*ctx,ggml_cgraph*graph,ggml_backend_t backend,uint64_t ceiling){
    check_graph(backend,graph);const auto buft=ggml_backend_get_default_buffer_type(backend);
    require(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,buft)<=ceiling,"V4.1 sweep graph exceeds scratch admission");
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx,buft));
    require(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=ceiling,"V4.1 sweep scratch allocation failed or exceeded admission");
    return buffer;
}
}

struct llama_dsv41_sweep::implementation {
    enum phase {fresh,opened,running,encoder_complete,completed,poisoned};
    mutable std::mutex mutex;
    std::atomic<bool> stop_requested{false},published{false};
    phase state=fresh;
    config options{};
    llama_dsv41_yoco::plan yoco;
    ggml_backend_t backend=nullptr;
    llama_dsv41_weight_provider provider;
    llama_dsv41_carry_store carry;
    std::shared_ptr<llama_dsv41_cache_storage>cache;
    std::unique_ptr<llama_dsv41_tile_executor>executor;
    std::vector<int32_t>tokens;
    uint32_t position_offset=0;
    llama_dsv41::engram_history initial_engram,final_engram;
    // Layer 0's ledger wall of the last run (the auto early-Engram append gap).
    double layer0_wall_ms=0;
    std::vector<float>logits;
    run_timing measured;
    // DS4COMBO fast switch: a completed owner whose device memory, DS pipeline
    // and carry RAM pages were released; only a rebase open may take it over.
    bool device_suspended=false,rebase_ready=false;

    ~implementation(){if(backend)ggml_backend_synchronize(backend);provider.invalidate();}
    void poison(){
        published.store(false);state=poisoned;carry.invalidate();
        // A failed graph may have submitted work before returning an error.
        // Never release its real staged weights or cache buffers while live.
        if(backend)ggml_backend_synchronize(backend);provider.invalidate();
    }
    void check_stop(const cancelled &external={}){
        if(stop_requested.load()||(external&&external())){
            stop_requested.store(true);throw std::runtime_error("V4.1 sweep cancelled; incomplete frontier cannot be exported");
        }
    }
    ggml_context_ptr context(){
        ggml_context_ptr ctx(ggml_init({size_t(metadata_bytes),nullptr,true}));
        require(bool(ctx),"V4.1 sweep graph metadata allocation failed");return ctx;
    }
    llama_dsv41_carry_store::tile seed_tile(ggml_tensor*embedding,uint32_t start,uint32_t count){
        require(embedding&&embedding->data&&embedding->buffer&&embedding->ne[0]==width&&embedding->ne[1]==vocabulary&&
            embedding->ne[2]==1&&embedding->ne[3]==1,"V4.1 staged text embedding geometry changed");
        auto ctx=context();auto*ids=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,count);
        auto*rows=ggml_get_rows(ctx.get(),embedding,ids);
        llama_dsv41_graph_ops ops(ctx.get(),1e-20f,1e-6f,20);
        auto values=ops.seed_embeddings(rows);auto*graph=ggml_new_graph_custom(ctx.get(),8192,false);
        ggml_build_forward_expand(graph,values.residual);ggml_build_forward_expand(graph,values.pre);
        auto buffer=allocate_graph(ctx.get(),graph,backend,options.admitted_graph_bytes);
        upload(ids,std::vector<int32_t>(tokens.begin()+position_offset+start,tokens.begin()+position_offset+start+count));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"V4.1 actual embedding graph failed");
        llama_dsv41_carry_store::tile out;out.start=start;out.count=count;
        out.residual=read_f32(values.residual,size_t(width)*copies*count);
        out.previous_pre=read_f32(values.pre,size_t(copies)*count);return out;
    }
    void head(ggml_tensor*norm,ggml_tensor*weight,const llama_dsv41_carry_store::tile *resident_last=nullptr){
#ifdef GGML_USE_CUDA
        // The bounded one-row Q8/F32 head selects cuBLAS SGEMM, whose first
        // invocation fails under capture on this backend. Keep only the head
        // outside capture; layer/embedding graph policy remains the caller's.
        struct head_graph_scope {
            ggml_backend_t backend;
            explicit head_graph_scope(ggml_backend_t b):backend(ggml_backend_is_cuda(b)?b:nullptr){
                if(backend)ggml_backend_cuda_graphs_disable_push(backend);
            }
            ~head_graph_scope(){if(backend)ggml_backend_cuda_graphs_disable_pop(backend);}
        } head_scope(backend);
#endif
        require(norm&&norm->data&&norm->buffer&&ggml_nelements(norm)==width&&weight&&weight->data&&weight->buffer&&
            weight->type==GGML_TYPE_Q8_0&&weight->ne[0]==width&&weight->ne[1]==vocabulary&&weight->ne[2]==1&&weight->ne[3]==1,
            "V4.1 staged output head geometry changed");
        constexpr uint32_t head_rows=4096;
        llama_dsv41_layer_log("[dsv41-sweep] head begin rows=%u slice_rows=%u precision=F32\n",vocabulary,head_rows);
        std::fflush(stderr);
        std::string error;llama_dsv41_carry_store::tile last;
        if(resident_last)last=*resident_last;
        else success(carry.read_final_tile(options.carry.tokens-1,1,last,error),error);
        logits.clear();logits.reserve(vocabulary);
        for(uint32_t first=0;first<vocabulary;first+=head_rows){
            check_stop();const uint32_t count=std::min(head_rows,vocabulary-first);
            // GGML_PREC_F32 dequantizes this Q8 matrix in the CUDA backend pool.
            // That allocation is invisible to the graph-context allocator. Charge
            // its exact F32 size against the SAME admitted graph ceiling before
            // allocating this slice, rather than concealing a full-head 2.47 GiB
            // temporary behind a small context-size estimate. At most 80 MiB is
            // needed, and each synchronous slice releases its temporary before
            // the next graph is submitted. Provider staging is already admitted.
            const uint64_t temporary_bytes=uint64_t(width)*count*sizeof(float);
            require(temporary_bytes<options.admitted_graph_bytes,
                "V4.1 F32 output-head temporary exceeds graph admission");
            auto ctx=context();
            auto*residual=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,width,copies,1);
            auto*pre=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,copies,1);
            auto*slice=ggml_view_2d(ctx.get(),weight,width,count,weight->nb[1],size_t(first)*weight->nb[1]);
            llama_dsv41_graph_ops ops(ctx.get(),1e-20f,1e-6f,20);
            auto*out=ops.logits(residual,pre,norm,slice);
            auto*graph=ggml_new_graph_custom(ctx.get(),8192,false);
            ggml_build_forward_expand(graph,out);
            auto buffer=allocate_graph(ctx.get(),graph,backend,options.admitted_graph_bytes-temporary_bytes);
            llama_dsv41_layer_log("[dsv41-sweep] head slice first=%u rows=%u graph_bytes=%llu temporary_bytes=%llu\n",
                first,count,(unsigned long long)ggml_backend_buffer_get_size(buffer.get()),
                (unsigned long long)temporary_bytes);
            std::fflush(stderr);
            upload(residual,last.residual);upload(pre,last.previous_pre);
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"V4.1 actual output head slice graph failed");
            const auto values=read_f32(out,count);
            logits.insert(logits.end(),values.begin(),values.end());
        }
        require(logits.size()==vocabulary,"V4.1 sliced output head vocabulary order/size changed");
        llama_dsv41_layer_log("[dsv41-sweep] head complete logits=%zu\n",logits.size());std::fflush(stderr);
    }
};

llama_dsv41_sweep::llama_dsv41_sweep():p(new implementation){}
llama_dsv41_sweep::~llama_dsv41_sweep()=default;

// LLAMADECODE decode-sized fixed plan. One-token steps need a 1-row graph and
// hidden bank; the prefill-sized reservations go back to the device (the
// decode L2 expert cache uses them). LLAMADECODE_DECODE_COMPACT=0 disables.
static bool decode_compact_enabled(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_DECODE_COMPACT");return !(e&&e[0]=='0'&&!e[1]);}();return on;}
static void compact_fixed(llama_dsv41_fixed_storage&f,uint64_t admitted_hidden){
    require(f.ready&&f.graph&&!f.decode_compact,"Decode compaction requires the ready prefill plan");
    ggml_backend_synchronize(f.backend);
    const uint64_t old_graph=ggml_gallocr_get_buffer_size(f.graph.get(),0),old_hidden=f.hidden?ggml_backend_buffer_get_size(f.hidden.get()):0;
    f.full_graph_reserve=f.graph_bytes;f.full_hidden_bytes=f.hidden_bytes;
    f.graph=std::shared_ptr<ggml_gallocr>(ggml_gallocr_new(ggml_backend_get_default_buffer_type(f.backend)),ggml_gallocr_free);
    require(bool(f.graph),"Decode graph allocator creation failed");f.graph_bytes=0;
    if(f.hidden){f.hidden.reset();f.hidden=llama_dsv41_tile_executor::reserve_hidden(f.backend,1,f.tile,f.bf16,f.ids,f.metadata,admitted_hidden);
        f.hidden_bytes=ggml_backend_buffer_get_size(f.hidden.get());}
    f.decode_compact=true;
    std::fprintf(stderr,"LLAMADECODE_DECODE_COMPACT graph_bytes=%llu->dynamic hidden_bytes=%llu->%llu\n",(unsigned long long)old_graph,(unsigned long long)old_hidden,(unsigned long long)f.hidden_bytes);
}
static void restore_fixed(llama_dsv41_fixed_storage&f,uint64_t admitted_hidden){
    require(f.ready&&f.decode_compact,"Prefill restore requires a compacted decode plan");
    ggml_backend_synchronize(f.backend);
    f.graph.reset();f.hidden.reset();
    f.graph=std::shared_ptr<ggml_gallocr>(ggml_gallocr_new(ggml_backend_get_default_buffer_type(f.backend)),ggml_gallocr_free);
    require(bool(f.graph),"Prefill graph allocator re-creation failed");
    if(f.full_graph_reserve){
        ggml_context_ptr metadata(ggml_init({65536,nullptr,true}));require(bool(metadata),"Prefill graph re-reservation metadata failed");
        auto*space=ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I8,int64_t(f.full_graph_reserve));ggml_set_output(space);
        auto*graph=ggml_new_graph_custom(metadata.get(),16,false);ggml_build_forward_expand(graph,space);
        require(ggml_gallocr_reserve(f.graph.get(),graph),"Prefill graph re-reservation failed");
        require(ggml_gallocr_get_buffer_size(f.graph.get(),0)==f.full_graph_reserve,"Prefill graph re-reservation differs");
    }
    f.graph_bytes=f.full_graph_reserve;
    if(f.full_hidden_bytes){f.hidden=llama_dsv41_tile_executor::reserve_hidden(f.backend,f.bank_rows(),f.tile,f.bf16,f.ids,f.metadata,admitted_hidden);
        f.hidden_bytes=ggml_backend_buffer_get_size(f.hidden.get());require(f.hidden_bytes==f.full_hidden_bytes,"Prefill hidden re-reservation differs");}
    f.decode_compact=false;
    std::fprintf(stderr,"LLAMADECODE_PREFILL_RESTORE graph_bytes=%llu hidden_bytes=%llu\n",(unsigned long long)f.graph_bytes,(unsigned long long)f.hidden_bytes);
}
bool llama_dsv41_sweep::open(const llama_dsv41_weight_provider::config &input_weights,const config &input_options,std::string &error,llama_dsv41_sweep *warmup,llama_dsv41_sweep *prefix,const native_base*restored){
    auto options=input_options;auto weights=input_weights;uint32_t yoco_begin=0;
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
    try{
        llama_dsv41_yoco::validate(options.yoco_keep);
        if(options.yoco_keep){
            require(!restored,"YOCO refuses RTX/Mac session restore; re-prefill locally");
            require(options.fixed&&options.carry.resident_ram&&options.carry.direct_ram_tiles&&options.admitted_hidden_bytes&&options.carry.export_hidden_after_head&&options.carry.max_tile_tokens>=140&&(!options.compute_step_tokens||options.compute_step_tokens>=140)&&!options.diagnostic_layers&&!options.diagnostic_carry&&
                !std::getenv("DSV41_BOUNDARY_CAPTURE"),"YOCO requires the single-engine fixed RAM owner without full-row diagnostics");
            if(prefix){
                require(prefix!=this,"YOCO prefix cannot be its own destination");
                std::unique_lock<std::mutex> previous_lock(prefix->p->mutex,std::try_to_lock);
                require(previous_lock.owns_lock(),"YOCO prefix is active");
                yoco_begin=uint32_t(prefix->p->tokens.size());
            }
            const uint32_t begin=yoco_begin;
            const uint32_t end=options.yoco_logical_end?options.yoco_logical_end:begin+options.carry.tokens;
            p->yoco=llama_dsv41_yoco::schedule(options.yoco_keep,options.operation,options.images,
                options.yoco_logical_end?options.yoco_logical_begin:begin,end,begin,begin+options.carry.tokens);
            options.carry.yoco_first=p->yoco.first;
            weights.yoco_encoder_only=p->yoco.encoder_only;
        }
        require(p->state==implementation::fresh,"V4.1 sweep cannot reopen or replace an existing request");
        require(!warmup||!prefix,"Warmup reset and stateful append are distinct operations");
        require(!restored||(!prefix&&options.stateful_chunks&&!options.capture_dspark&&
            restored->frontier.mode==1&&restored->frontier.context==options.context&&
            restored->frontier.position==restored->tokens.size()&&restored->frontier.position&&
            restored->frontier.position<options.context&&options.carry.tokens<options.context-restored->frontier.position&&
            restored->logits.size()==vocabulary&&bool(restored->read)),"Restored base requires a complete same-context mode1 checkpoint");
        if(restored){
            require(std::all_of(restored->tokens.begin(),restored->tokens.end(),[](int32_t t){return t>=0&&uint32_t(t)<vocabulary;}),"Restored base token out of range");
            finite(restored->logits);
            require(dsv41_boundary::sha256(restored->tokens.data(),restored->tokens.size()*4)==restored->tokens_sha256&&
                dsv41_boundary::sha256(restored->logits.data(),restored->logits.size()*4)==restored->logits_sha256,"Restored base token/logit identity mismatch");
        }
        (void)llama_dsv41_plan_append(options.pair_ffn,options.carry.max_tile_tokens,options.compute_step_tokens,0,options.unaligned_append_step_tokens);
        require(!options.unaligned_append_step_tokens||(options.stateful_chunks&&options.fixed&&options.reuse_graph&&!options.reuse_layer_graph),"Unaligned append policy requires the existing stateful fixed owner");
        require(!options.compute_step_tokens||(options.compute_step_tokens<=options.carry.max_tile_tokens&&(!options.pair_ffn||options.compute_step_tokens==options.carry.max_tile_tokens)),"Logical compute step exceeds its admitted tile or changes paired group alignment");
        require(!options.stateful_chunks||(options.carry.resident_ram&&options.reuse_graph&&!options.reuse_layer_graph&&!options.diagnostic_layers&&!std::getenv("DSV41_BOUNDARY_CAPTURE")),"Stateful chunks require explicit RAM and allocator-only graph mode without incomplete diagnostics");
        llama_dsv41_expert_profile::validate(options.expert,options.carry.max_tile_tokens,std::getenv("DSV41_BOUNDARY_CAPTURE"));
        require(!options.admitted_pinned_device_bytes||(options.carry.resident_ram&&
            llama_dsv41_expert_profile::native(options.expert)&&options.admitted_pinned_device_bytes>=64ull*1024*1024),
            "Direct pinned RAM requires native experts, admitted RAM bank and separate64MiB device ceiling");
        require(!options.admitted_prefetch_device_bytes||(options.admitted_pinned_device_bytes&&options.reuse_graph&&!options.reuse_layer_graph&&
            (!options.admitted_index_device_bytes||options.pair_ffn||options.fixed)&&!weights.ds_gpu_ring_slots&&
            uint64_t(options.carry.max_tile_tokens)*81936+65536<=options.admitted_prefetch_device_bytes),
            "Copy overlap requires separately admitted GPU tile, pinned RAM, allocator mode1, original indexer and legacy fetch profile");
        require(options.context&&options.context<=1048576&&options.carry.tokens&&
            options.carry.tokens<options.context,"V4.1 sweep must retain room for the next native decode token");
        require(weights.backend!=nullptr,"V4.1 sweep requires an admitted real compute backend");
        if(options.expert.index_key_windows||options.expert.attention_kv_f16_table){
#ifdef GGML_USE_CUDA
            require(ggml_backend_is_cuda(weights.backend)&&options.reuse_graph&&!options.reuse_layer_graph,"Bounded index windows/direct F16 attention require CUDA and the existing allocator-only graph mode");
#else
            require(false,"Bounded index windows/direct F16 attention require the CUDA implementation");
#endif
        }
        require(!weights.engram_prefetch_host_bytes||weights.engram_prefetch_tokens==options.carry.tokens,
            "Engram whole-prompt admission must bind the sweep token count");
        require(!options.admitted_cast_device_bytes||((options.carry.resident_ram||options.carry.ram_prefix_tokens)&&options.reuse_graph&&!options.reuse_layer_graph&&llama_dsv41_expert_profile::native(options.expert)),
            "Weight cast cache requires native RAM and allocator-only mode1");
        require(!options.audit_casts||options.admitted_cast_device_bytes,"Cast audit requires cache admission");
        auto resident_required=[&](uint32_t n){return options.resident_bf16?
            llama_dsv41_tile_executor::resident_layout_bytes(weights.backend,n,options.carry.max_tile_tokens,options.resident_bf16,options.resident_ids,options.resident_metadata):
            llama_dsv41_tile_executor::resident_hidden_bytes(n);};
        // LLAMADECODE single sweep: a fixed bank smaller than this sweep spills through its last tile.
        const uint32_t bank_rows=options.fixed&&options.fixed->hidden_rows&&options.fixed->hidden_rows<options.carry.tokens?options.fixed->hidden_rows:options.carry.tokens;
        require(bank_rows==options.carry.tokens||(options.resident_bf16&&options.resident_ids&&options.resident_metadata&&options.carry.defer_hidden_readback&&
            options.carry.export_hidden_after_head&&bank_rows%options.carry.max_tile_tokens==0&&bank_rows>=2*options.carry.max_tile_tokens),
            "Hidden-bank spill requires the full BF16/ID/metadata resident plan and a tile-aligned bank of >=2 tiles");
        const bool native_pair=options.pair_ffn&&llama_dsv41_expert_profile::natural_half(options.expert);
        require(!options.pair_ffn||((native_pair?options.carry.max_tile_tokens==4096:
            ((options.dense_mxfp4||options.expert.dense_activation_q8_1)&&llama_dsv41_expert_profile::activation_rounding_q8_1(options.expert)&&
             (options.carry.max_tile_tokens==4096||options.carry.max_tile_tokens==8192||options.carry.max_tile_tokens==16384)))&&
            options.carry.direct_ram_tiles&&((options.carry.defer_hidden_readback&&options.admitted_hidden_bytes>=resident_required(bank_rows))||
            (!options.admitted_hidden_bytes&&options.admitted_prefetch_device_bytes&&!options.carry.defer_hidden_readback&&!options.resident_bf16&&!options.resident_ids&&!options.resident_metadata))&&
            !options.diagnostic_carry),"Fixed grouped FFN requires bounded native4K or explicit original-Q8/hyper-fast selection and admitted resident/RAM owner");
        require(options.carry.max_tile_tokens<=4096||options.pair_ffn,"8192 groups require paired attention/FFN construction");
        require(!options.batch_seed_ram||(options.carry.resident_ram&&options.carry.direct_ram_tiles&&options.admitted_pinned_device_bytes&&options.reuse_graph&&!options.reuse_layer_graph),
            "Batched seed requires the existing direct pinned RAM and allocator-only workspace");
        require(!options.engram_tile_lookahead||weights.engram_prefetch_host_bytes,
            "Engram tile lookahead requires the existing immutable whole-prompt cache");
        require(!options.carry.export_hidden_after_head||options.carry.defer_hidden_readback,"Post-head hidden export requires full-device retention");
        require(!options.carry.defer_hidden_readback||(options.carry.direct_ram_tiles&&options.admitted_pinned_device_bytes&&options.reuse_graph&&!options.reuse_layer_graph&&!options.admitted_prefetch_device_bytes&&!options.diagnostic_carry&&options.admitted_hidden_bytes>=resident_required(bank_rows)),"Final-only hidden readback requires full admitted VRAM residency and forbids intermediate RAM captures");
        require(!options.merge_attention_selected_rows||options.expert.token_tile_attention,"Selected-row union requires explicit token-tile attention");
        require(!options.coalesce_dense||(options.admitted_cast_device_bytes&&options.expert.dense_inputs_f16),"Dense coalescing requires explicit F16 inputs and admitted cast cache");
        require(!options.admitted_index_device_bytes||(options.carry.resident_ram&&options.reuse_graph&&!options.reuse_layer_graph&&
            options.admitted_index_device_bytes>=64ull*1024*1024&&llama_dsv41_expert_profile::native(options.expert)),
            "Bounded GLM index TEST requires native RAM, allocator-only mode and explicit64MiB device admission");
        require(options.admitted_graph_bytes&&options.admitted_device_bytes&&options.admitted_host_bytes,
                "V4.1 sweep requires explicit total device/host/scratch admission");
        require(!options.carry.direct_ram_tiles||(options.carry.resident_ram&&options.admitted_pinned_device_bytes&&llama_dsv41_expert_profile::native(options.expert)),
                "Direct-only carry accounting requires admitted pinned native RAM execution");
        require(llama_dsv41_cache_storage::required_bytes(options.context,options.compressed_cache_bf16,options.capture_dspark)<=options.admitted_cache_bytes,
            "V4.1 sweep canonical cache exceeds admission");
        llama_dsv41_carry_store::budget carry_budget;
        success(options.carry.resident_ram ?
            (options.carry.direct_ram_tiles?llama_dsv41_carry_store::plan_resident_ram_direct(options.carry.tokens,carry_budget,error,options.carry.max_tile_tokens):llama_dsv41_carry_store::plan_resident_ram(options.carry.tokens,carry_budget,error,options.carry.max_tile_tokens)) :
            options.carry.ram_prefix_tokens?llama_dsv41_carry_store::plan_hybrid(options.carry.tokens,options.carry.ram_prefix_tokens,options.carry.max_tile_tokens,
                options.carry.outstanding_bytes,options.carry.disk_reserve_bytes,carry_budget,error):
            llama_dsv41_carry_store::plan(options.carry.tokens,options.carry.outstanding_bytes,carry_budget,error),error);
        require((options.carry.resident_ram||carry_budget.disk_bytes<=options.carry.admitted_file_bytes)&&
            carry_budget.bounded_host_bytes<=options.carry.admitted_host_bytes,
            "V4.1 sweep carry exceeds explicit storage/host admission");
        // Metadata inspection retains the verified original file lease but
        // allocates no DS heaps or GPU weights. Admit the whole job first.
        if(prefix){
            require(prefix!=this&&options.stateful_chunks,"Append requires an explicit distinct stateful completed owner");
            std::unique_lock<std::mutex> previous_lock(prefix->p->mutex,std::try_to_lock);
            require(previous_lock.owns_lock(),"Append prefix is active");
            const bool encoder=prefix->p->state==implementation::encoder_complete;
            require(!encoder||(options.yoco_keep&&prefix->p->options.yoco_keep==options.yoco_keep&&
                options.yoco_logical_end==prefix->p->options.yoco_logical_end&&options.yoco_logical_begin==prefix->p->options.yoco_logical_begin),"YOCO encoder continuation changed logical call");
            require(previous_lock.owns_lock()&&((prefix->p->state==implementation::completed&&prefix->p->published.load()&&prefix->p->carry.frontier_ready())||encoder)&&!prefix->p->stop_requested.load()&&prefix->p->cache,"Append requires a completely certified uncancelled prefix");
            const auto&old=prefix->p->options;
            auto arithmetic=[](const llama_dsv41_expert_profile::policy&e){return std::tie(e.profile,e.dense_inputs_f16,e.index_bf16,e.token_tile_attention,e.expert_f16_cublas,e.admitted_pool_bytes,e.admitted_code_runtime_bytes,e.sorted_expert_routing,e.reuse_expert_routing_map,e.hc_bf16_epilogue,e.expert_mxfp4,e.dense_activation_q8_1,e.index_key_windows,e.attention_kv_f16_table);};
            auto execution=[](const config&c){return std::tie(c.yoco_keep,c.context,c.admitted_cache_bytes,c.admitted_graph_bytes,c.admitted_device_bytes,c.admitted_host_bytes,c.admitted_hidden_bytes,c.admitted_pinned_device_bytes,c.admitted_cast_device_bytes,c.admitted_index_device_bytes,c.admitted_prefetch_device_bytes,c.audit_casts,c.coalesce_dense,c.merge_attention_selected_rows,c.hc_projection_f16_gemm,c.dense_mxfp4,c.batch_seed_ram,c.pair_ffn,c.reuse_graph,c.reuse_layer_graph,c.engram_tile_lookahead,c.resident_bf16,c.async_inputs,c.resident_ids,c.resident_metadata,c.fixed,c.compressed_cache_bf16,c.ctx_cache_grow_step,c.capture_dspark,c.stateful_chunks,c.compute_step_tokens,c.unaligned_append_step_tokens,c.carry.max_tile_tokens,c.carry.resident_ram,c.carry.direct_ram_tiles,c.carry.defer_hidden_readback,c.carry.export_hidden_after_head);};
            require(old.stateful_chunks&&prefix->p->backend==weights.backend&&arithmetic(old.expert)==arithmetic(options.expert)&&execution(old)==execution(options),"Append changed its backend, fixed owner, arithmetic or execution admission");
            require(!prefix->p->tokens.empty()&&prefix->p->tokens.size()<options.context&&options.carry.tokens<options.context-prefix->p->tokens.size(),"Appended context must retain the next native decode position");
            p->position_offset=uint32_t(prefix->p->tokens.size());
            require(!options.yoco_keep||p->position_offset==yoco_begin,"YOCO prefix changed during admission");
            const auto schedule=llama_dsv41_plan_append(options.pair_ffn,options.carry.max_tile_tokens,options.compute_step_tokens,p->position_offset,options.unaligned_append_step_tokens);
            if(schedule.single_attention_append)llama_dsv41_layer_log("DS4PERF_UNALIGNED_APPEND_PLAN prefix_tokens=%u step=%u fixed_group=%u unchanged_budget=1 single_attention_graph=1\n",p->position_offset,schedule.step,options.carry.max_tile_tokens);
            ggml_backend_synchronize(weights.backend);
            prefix->p->published.store(false);prefix->p->state=implementation::poisoned;
            prefix->p->executor.reset();
            if(options.fixed)success(encoder?prefix->p->carry.retire_yoco_encoder(error):prefix->p->carry.retire_shared_ram(error),error);
            p->tokens=std::move(prefix->p->tokens);p->initial_engram=prefix->p->final_engram;
            p->cache=std::move(prefix->p->cache);p->provider=std::move(prefix->p->provider);prefix->p->backend=nullptr;
            success(p->provider.restart_prompt(weights,error),error);
            llama_dsv41_layer_log("DS4PERF_APPEND_OWNER_TRANSFER prefix_tokens=%u chunk_tokens=%u canonical_cache_preserved=1 raw_pool_tails_preserved=1 old_export_revoked=1 Engram_history=%d,%d,%d\n",p->position_offset,options.carry.tokens,p->initial_engram.tail[0],p->initial_engram.tail[1],p->initial_engram.tail[2]);
        }else if(warmup&&warmup->p->rebase_ready){
            // DS4COMBO fast switch: a resumed owner gives its provider, model
            // lease and fixed storage to a fresh request that imports a new base.
            require(warmup!=this&&restored&&options.fixed,"Rebase requires a distinct resumed fixed owner and a restored native base");
            std::unique_lock<std::mutex> owner_lock(warmup->p->mutex,std::try_to_lock);
            require(owner_lock.owns_lock()&&warmup->p->state==implementation::completed&&!warmup->p->device_suspended&&
                warmup->p->backend==weights.backend&&warmup->p->options.fixed==options.fixed,"Rebase owner changed its backend or fixed storage");
            p->provider=std::move(warmup->p->provider);
            warmup->p->state=implementation::poisoned;warmup->p->rebase_ready=false;
            warmup->p->cache.reset();warmup->p->backend=nullptr;
            success(p->provider.restart_prompt(weights,error),error);
            llama_dsv41_layer_log("DS4COMBO_REBASE_OWNER_TRANSFER fixed_storage_kept=1 provider_kept=1 restored_position=%u\n",restored->frontier.position);
        }else if(warmup){
            require(warmup!=this,"Warmup provider cannot transfer to itself");
            std::unique_lock<std::mutex> warm_lock(warmup->p->mutex,std::try_to_lock);
            require(warm_lock.owns_lock()&&warmup->p->state==implementation::completed&&warmup->p->published.load()&&
                warmup->p->tokens.size()==1&&warmup->p->backend==weights.backend&&options.carry.resident_ram&&warmup->p->options.carry.resident_ram,
                "Provider reuse requires a completed one-token RAM warmup on the same backend");
            p->provider=std::move(warmup->p->provider);
            warmup->p->published.store(false);warmup->p->state=implementation::poisoned;
            if(options.fixed){warmup->p->executor.reset();success(warmup->p->carry.retire_shared_ram(error),error);}
            warmup->p->cache.reset();warmup->p->backend=nullptr;
            success(p->provider.restart_prompt(weights,error),error);
        }else success(p->provider.inspect(weights,error),error);
        const auto provider=p->provider.budget();
        require(!options.admitted_hidden_bytes||(options.carry.resident_ram&&
            resident_required(std::min(options.carry.tokens,options.carry.max_tile_tokens))<=options.admitted_hidden_bytes),
            "Resident hidden bank requires admitted RAM mirror and explicit device ceiling");
        // The context cache is charged at what it can hold for its context (<= --cache-bytes).
        const uint64_t cache_charge=llama_dsv41_device_admission::cache_charge(options.admitted_cache_bytes,
            llama_dsv41_cache_storage::required_bytes(options.context,options.compressed_cache_bf16,options.capture_dspark));
        const uint64_t device=add(options.admitted_prefetch_device_bytes,add(add(add(add(add(provider.device_bytes,cache_charge),options.admitted_graph_bytes),options.admitted_hidden_bytes),options.admitted_pinned_device_bytes),add(llama_dsv41_expert_profile::extra_device(options.expert),add(options.admitted_cast_device_bytes,options.admitted_index_device_bytes))));
        // Deliberately conservative: includes both input/output tile vectors,
        // lease copy, graph metadata, bounded Engram and serialization scratch.
        const uint64_t tile_host=options.carry.direct_ram_tiles?
            3ull*tile_limit*(width*copies*4+copies*4+(512+2048)*4)+uint64_t(options.carry.max_tile_tokens)*(6144ull*4+128ull*8)+32ull*1024*1024:
            3ull*options.carry.max_tile_tokens*(width*copies*4+copies*4+(512+2048)*4)+options.carry.max_tile_tokens*6144ull*4;
        const uint64_t pair_host=options.pair_ffn?128ull*1024*1024:0;
        const uint64_t pinned_metadata=(options.admitted_prefetch_device_bytes?16ull*1024*1024:0)+(options.admitted_pinned_device_bytes?16ull*1024*1024:0)+(options.admitted_cast_device_bytes?64ull*1024*1024:0);
        require(!options.diagnostic_layers||(options.carry.resident_ram&&options.diagnostic_layers<40&&bool(options.diagnostic_carry)),
            "Incomplete layer diagnostics require RAM,1..39 layers and an observer");
        const uint64_t allocator_metadata=options.reuse_layer_graph?128ull*1024*1024:options.reuse_graph?16ull*1024*1024:0;
        const uint64_t long_metadata=options.expert.index_key_windows&&options.context>131072?8ull*1024*1024:0;
        uint64_t carry_host=carry_budget.bounded_host_bytes;
        if(options.fixed){llama_dsv41_carry_store::budget fixed_host;std::string why;
            require(llama_dsv41_carry_store::plan_resident_ram_direct(options.fixed->tokens,fixed_host,why,options.fixed->tile),"Fixed carry host plan failed");
            carry_host=std::max(carry_host,fixed_host.bounded_host_bytes);
        }
        // Owner-digest read staging (pinned, llama-dsv41-owner-digest.h): 8 workers x 2 slots x 2 MiB.
        const uint64_t owner_digest_staging=llama_dsv41::owner_digest_memo::staging_bytes();
        const uint64_t base_host=add(owner_digest_staging,add(options.compressed_cache_bf16?1024ull*1024:0,add(options.engram_tile_lookahead?uint64_t(options.carry.max_tile_tokens)*6144*4+8ull*1024*1024:0,add(provider.host_bytes,add(carry_host,
            add(tile_host+pair_host,add(metadata_bytes+pinned_metadata+allocator_metadata+long_metadata+2ull*vocabulary*4+1024*1024,uint64_t(options.stateful_chunks?options.context:options.carry.tokens)*8)))))));
        // LLAMADECODE_ENGRAM_CACHE_MB: the Engram row LRU (early loader) is charged
        // here. Unset: min(4 GiB, this admission's headroom), whole MiB, none below
        // 16 MiB. Explicit: exactly that size or a loud refusal. 0: no cache.
        // Every LLAMADECODE_ENGRAM_EARLY arm (0 included) loads on the early loader.
        const uint64_t row_cache_have=p->provider.engram_row_cache_bytes();uint64_t row_cache_new=0;
        if(!row_cache_have&&weights.engram_prefetch_host_bytes&&options.engram_tile_lookahead){
            const char*e=std::getenv("LLAMADECODE_ENGRAM_CACHE_MB");
            const uint64_t headroom=options.admitted_host_bytes>base_host?options.admitted_host_bytes-base_host:0;
            if(e&&*e){char*end=nullptr;const unsigned long long mb=std::strtoull(e,&end,10);
                require(end&&!*end&&mb<=1048576,"LLAMADECODE_ENGRAM_CACHE_MB must be a whole number of MiB (0 disables)");
                row_cache_new=uint64_t(mb)<<20;
                if(row_cache_new&&row_cache_new>headroom){
                    llama_dsv41_layer_log("LLAMADECODE_ENGRAM_ROWCACHE_REFUSED requested_bytes=%llu headroom_bytes=%llu admitted_host=%llu planned_host=%llu\n",
                        (unsigned long long)row_cache_new,(unsigned long long)headroom,(unsigned long long)options.admitted_host_bytes,(unsigned long long)base_host);
                    require(false,"LLAMADECODE_ENGRAM_CACHE_MB exceeds the producer host admission headroom (no silent shrink)");
                }
            }else{row_cache_new=std::min<uint64_t>(4ull<<30,headroom)&~((1ull<<20)-1);if(row_cache_new<(16ull<<20))row_cache_new=0;}
            if(!e||*e)llama_dsv41_layer_log("LLAMADECODE_ENGRAM_ROWCACHE_ADMISSION bytes=%llu explicit=%d headroom_bytes=%llu\n",(unsigned long long)row_cache_new,int(e&&*e),(unsigned long long)headroom);
        }
        const uint64_t host=add(base_host,row_cache_have?row_cache_have:row_cache_new);
        {   // log only: each planned host component and the row-cache headroom without it
            const uint64_t lookahead=options.engram_tile_lookahead?uint64_t(options.carry.max_tile_tokens)*6144*4+8ull*1024*1024:0;
            const uint64_t metadata=metadata_bytes+pinned_metadata+allocator_metadata+long_metadata+2ull*vocabulary*4+1024*1024;
            const uint64_t token_ids=uint64_t(options.stateful_chunks?options.context:options.carry.tokens)*8;
            llama_dsv41_layer_log("%s\n",llama_dsv41_host_plan_log::line(options.admitted_host_bytes,base_host,{
                {"provider",provider.host_bytes},{"carry",carry_host},{"tile",tile_host},{"pair_ffn",pair_host},
                {"owner_digest_staging",owner_digest_staging},{"engram_lookahead",lookahead},{"bf16_cache_scratch",options.compressed_cache_bf16?1024ull*1024:0},
                {"metadata",metadata},{"token_ids",token_ids}},row_cache_have?row_cache_have:row_cache_new).c_str());
        }
        const auto measured=options.measured_device_free?llama_dsv41_device_admission::admit(options.measured_device_free,options.admitted_device_bytes,options.device_reserve,device)
            :llama_dsv41_device_admission::result{device<=options.admitted_device_bytes,options.admitted_device_bytes,{}};
        llama_dsv41_layer_log("DS4PERF_COMBINED_ADMISSION planned_host=%llu admitted_host=%llu provider_host=%llu carry_host=%llu tile_host=%llu owner_digest_staging=%llu planned_device=%llu admitted_device=%llu"
            " job_device_bound=%llu measured_device_free=%llu device_reserve=%llu cache_charge=%llu cache_admitted=%llu\n",
            (unsigned long long)host,(unsigned long long)options.admitted_host_bytes,(unsigned long long)provider.host_bytes,
            (unsigned long long)carry_budget.bounded_host_bytes,(unsigned long long)tile_host,(unsigned long long)owner_digest_staging,(unsigned long long)device,(unsigned long long)measured.admitted,
            (unsigned long long)options.admitted_device_bytes,(unsigned long long)options.measured_device_free,(unsigned long long)options.device_reserve,
            (unsigned long long)cache_charge,(unsigned long long)options.admitted_cache_bytes);
        if(!measured.ok&&options.measured_device_free)llama_dsv41_layer_log("DS4PERF_DEVICE_ADMISSION_REFUSED %s\n",measured.reason.c_str());
        require(measured.ok&&host<=options.admitted_host_bytes,
            "V4.1 provider/cache/graph/carry combined job exceeds total resource admission");
        if(row_cache_new)success(p->provider.create_engram_row_cache(row_cache_new,error),error);
        // LLAMADECODE_ENGRAM_EARLY: tokens known, file verified, job admitted:
        // the request's Engram rows start now, before RAM/DS/weight-fetch setup.
        if(!warmup&&!prefix)success(p->provider.start_engram_cold(error),error);
        p->check_stop();p->options=options;p->backend=weights.backend;
        if(options.stateful_chunks)p->tokens.reserve(options.context);
        // Before DS/GPU allocation, RAM mode allocates its fully admitted
        // single bank; file mode performs actual-volume/path admission and
        // writes only its owned header, without preallocating the extent.
        if(options.fixed&&!options.fixed->ready){
            auto&f=*options.fixed;
            const bool first_request=f.direct_open&&!warmup&&!prefix&&(options.carry.tokens==f.tokens||(options.stateful_chunks&&options.carry.tokens<=f.tokens));
            require(!warmup&&(first_request||(!options.pair_ffn&&options.carry.tokens==1))&&!f.ram&&!f.ram_registration,"Fixed RAM must be allocated/registered once before imports");
            f.ram=llama_dsv41_carry_store::reserve_ram(f.tokens,options.carry.admitted_host_bytes,f.tile);
            if(f.tile_registration||f.pageable_copies){
                const bool full_resident=f.bf16&&f.ids&&f.metadata&&!options.admitted_prefetch_device_bytes;
                const bool ram_lookahead=!f.bf16&&!f.ids&&!f.metadata&&options.admitted_prefetch_device_bytes;
                require(full_resident||(f.tile_registration&&!f.pageable_copies&&ram_lookahead),"Tile pinning requires its explicit resident or RAM-lookahead plan");
                llama_dsv41_layer_log(f.pageable_copies?"DS4PERF_RAM_ALLOCATED_PAGEABLE_BEFORE_IMPORT bytes=%llu whole_bank_pinned=0\n":"DS4PERF_RAM_ALLOCATED_BEFORE_IMPORT bytes=%llu tile_registration=1 whole_bank_pinned=0\n",(unsigned long long)f.ram->size());
            }else f.ram_registration=llama_dsv41_tile_executor::register_fixed_ram(weights.backend,f.ram);
        }
        auto carry_options=options.carry;
        if(options.fixed){require(options.fixed->ram&&(options.fixed->ram_registration||options.fixed->tile_registration||options.fixed->pageable_copies),"Fixed carry RAM registration missing");carry_options.shared_ram=options.fixed->ram;}
        success(p->carry.create(carry_options,error),error);
        if(!warmup&&!prefix)success(p->provider.allocate(error),error);
        if(options.fixed){
            auto &f=*options.fixed;
            require(options.reuse_graph&&!options.reuse_layer_graph&&((options.resident_bf16&&options.carry.defer_hidden_readback)||(!options.admitted_hidden_bytes&&!options.resident_bf16&&!options.carry.defer_hidden_readback&&options.admitted_prefetch_device_bytes))&&
                options.context==f.context&&options.resident_ids==f.ids&&options.resident_metadata==f.metadata&&options.resident_bf16==f.bf16&&
                ((((options.pair_ffn&&!f.unpaired_resident)||(!options.pair_ffn&&f.unpaired_resident))&&
                  (options.carry.tokens==f.tokens||(options.stateful_chunks&&options.carry.tokens<=f.tokens))&&options.carry.max_tile_tokens==f.tile)||
                 (!options.pair_ffn&&options.carry.tokens==1&&options.carry.max_tile_tokens<=f.tile)),
                "Fixed GPU reservation request differs from its exact warmup/large plan");
            require(!f.unpaired_resident||(!options.pair_ffn&&f.tile<=4096&&f.tile>=32&&f.bf16&&f.ids&&f.metadata&&
                f.tile_registration&&!f.pageable_copies&&options.carry.defer_hidden_readback&&!options.admitted_prefetch_device_bytes&&
                f.backend_temporary_reserve_bytes==1024ull*1024*1024),
                "Unpaired fixed reservation requires its explicit full-resident tile-registration plan");
            if(!f.ready){
                const bool first_request=f.direct_open&&!warmup&&!prefix&&(options.carry.tokens==f.tokens||(options.stateful_chunks&&options.carry.tokens<=f.tokens));
                if(first_request)llama_dsv41_layer_log("DS4COMBO_FIXED_DIRECT_OPEN tokens=%u tile=%u pair=%d warmup_pass=0\n",options.carry.tokens,f.tile,int(options.pair_ffn));
                require(!warmup&&(first_request||(!options.pair_ffn&&options.carry.tokens==1))&&!f.graph&&!f.hidden&&!f.cache&&f.tokens&&f.tile&&options.admitted_graph_bytes>1024ull*1024*1024,"Fixed allocation must follow provider heaps and precede first stage lease during one-token warmup");
                require((f.backend_temporary_reserve_bytes==1024ull*1024*1024||f.backend_temporary_reserve_bytes==768ull*1024*1024)&&f.backend_temporary_reserve_bytes<options.admitted_graph_bytes,"Unsupported fixed backend temporary reserve");
                require(f.backend_temporary_reserve_bytes==1024ull*1024*1024||(options.expert.attention_kv_f16_table&&options.expert.index_key_windows&&options.expert.token_tile_attention&&f.tile<=8192),"Reduced backend reserve requires direct-F16/window profile and<=8K groups");
                f.backend=weights.backend;
                f.graph=std::shared_ptr<ggml_gallocr>(ggml_gallocr_new(ggml_backend_get_default_buffer_type(f.backend)),ggml_gallocr_free);
                require(bool(f.graph),"Fixed graph allocator creation failed");
                ggml_context_ptr metadata(ggml_init({65536,nullptr,true}));require(bool(metadata),"Fixed graph reservation metadata failed");
                const uint64_t reserve=f.tile==16384?0:options.admitted_graph_bytes-f.backend_temporary_reserve_bytes;
                if(reserve){
                auto *space=ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I8,int64_t(reserve));ggml_set_output(space);
                auto *graph=ggml_new_graph_custom(metadata.get(),16,false);ggml_build_forward_expand(graph,space);
                require(ggml_gallocr_reserve(f.graph.get(),graph),"Fixed graph reservation after provider heaps and before first lease failed");
                f.graph_bytes=ggml_gallocr_get_buffer_size(f.graph.get(),0);
                require(f.graph_bytes==reserve,"Fixed graph physical capacity differs from exact reservation");
                }else llama_dsv41_layer_log("DS4PERF_GRAPH_CAPACITY_DYNAMIC group=16384 within_explicit_graph_and_job_cap=1 allocation_in_prompt_clock=1\n");
                if(options.admitted_hidden_bytes){
                    f.hidden=llama_dsv41_tile_executor::reserve_hidden(f.backend,f.bank_rows(),f.tile,f.bf16,f.ids,f.metadata,options.admitted_hidden_bytes);
                    f.hidden_bytes=ggml_backend_buffer_get_size(f.hidden.get());
                }else require(!f.bf16&&!f.ids&&!f.metadata,"Layerwise RAM path cannot claim full-device banks");
                f.cache.reset(new llama_dsv41_cache_storage(native_owners(),f.context,ggml_backend_get_default_buffer_type(f.backend),options.admitted_cache_bytes,options.compressed_cache_bf16,options.capture_dspark,options.ctx_cache_grow_step));
                f.cache_bytes=f.cache->allocated_bytes();f.ready=true;
                ggml_backend_synchronize(f.backend);
                llama_dsv41_layer_log("DS4PERF_FIXED_BEFORE_FIRST_LEASE tokens=%u tile=%u graph_bytes=%llu hidden_bytes=%llu cache_bytes=%llu existing_allocators=1 provider_heaps_created=1\n",f.tokens,f.tile,(unsigned long long)f.graph_bytes,(unsigned long long)f.hidden_bytes,(unsigned long long)f.cache_bytes);
            }else{
                require((warmup||prefix)&&f.backend==weights.backend&&f.graph&&(f.hidden||!options.admitted_hidden_bytes)&&f.cache,"Fixed GPU buffers lost their completed request owner");
                // LLAMADECODE decode-sized plan (the previous owner's executor is already gone).
                if(options.carry.tokens==1&&prefix&&!f.decode_compact&&decode_compact_enabled())compact_fixed(f,options.admitted_hidden_bytes);
                else if(options.carry.tokens>1&&f.decode_compact)restore_fixed(f,options.admitted_hidden_bytes);
            }
        }
        require(p->provider.backend()==p->backend,"V4.1 staged provider/backend identity changed");
        const auto actual=p->provider.budget();
        require(actual.device_bytes<=provider.device_bytes&&actual.host_bytes<=provider.host_bytes,
            "V4.1 provider allocation changed after combined preflight");
        // Fixed request allocation is startup, before the prompt clock. All
        // resource checks above still precede allocation. Prompt graph work,
        // input preparation, Engram IO and every required head stay in run().
        if(options.fixed){require(!prefix||p->cache==options.fixed->cache,"Append lost its exact fixed canonical cache");p->cache=options.fixed->cache;if(!prefix)p->cache->clear();}
        else if(!prefix)p->cache.reset(new llama_dsv41_cache_storage(native_owners(),p->options.context,
            ggml_backend_get_default_buffer_type(p->backend),p->options.admitted_cache_bytes,p->options.compressed_cache_bf16,p->options.capture_dspark,p->options.ctx_cache_grow_step));
        require(p->cache->compact_compressed()==p->options.compressed_cache_bf16,"Fixed cache physical format differs from request admission");
        require(p->cache->captures_dspark()==p->options.capture_dspark,"Fixed cache DSpark capture admission changed");
        p->cache->admit_owner_digest_staging(llama_dsv41::owner_digest_memo::staging_bytes());   // counted in the combined host plan above
        require(!p->options.capture_dspark||!p->options.reuse_layer_graph,"DSpark ring writes require allocator-only graph reuse");
        if(restored){
            // Cache replacement uses the same bounded writer and all-owner
            // target verification as the existing native restore/delta path.
            success(p->cache->replace_native_frontier(restored->frontier,restored->owner_sha256,
                [&](const llama_dsv41::state_span&s,uint64_t off,void*dst,size_t n){p->check_stop();return restored->read(s,off,dst,n);},error),error);
            p->tokens=restored->tokens;p->position_offset=uint32_t(p->tokens.size());
            const size_t count=std::min<size_t>(p->tokens.size(),3);
            success(p->provider.restored_engram_history(p->initial_engram,p->tokens.data()+p->tokens.size()-count,count,error),error);
            (void)llama_dsv41_plan_append(options.pair_ffn,options.carry.max_tile_tokens,options.compute_step_tokens,p->position_offset,options.unaligned_append_step_tokens);
            llama_dsv41_layer_log("DS4COMBO_NATIVE_BASE_IMPORTED position=%u mode=1 all_owner_hashes_verified=1 model_eval=0\n",p->position_offset);
        }
        llama_dsv41_tile_executor::pinned_span direct_span;
        if(p->options.admitted_pinned_device_bytes){
            success(p->carry.ram_storage(direct_span.base,direct_span.bytes,error),error);
            direct_span.admitted_device_bytes=p->options.admitted_pinned_device_bytes;
        }
        p->executor.reset(new llama_dsv41_tile_executor(p->backend,*p->cache,p->options.admitted_graph_bytes,
            1e-20f,1e-6f,20,10.f,1.5f,p->options.carry.max_tile_tokens,
            p->options.admitted_hidden_bytes?p->options.carry.tokens:0,p->options.admitted_hidden_bytes,
            p->options.admitted_pinned_device_bytes?&direct_span:nullptr,p->options.reuse_graph,p->options.reuse_layer_graph,p->options.admitted_cast_device_bytes,p->options.audit_casts,p->options.admitted_index_device_bytes,p->options.admitted_prefetch_device_bytes,p->options.coalesce_dense,p->options.carry.defer_hidden_readback,p->options.merge_attention_selected_rows,p->options.hc_projection_f16_gemm,p->options.dense_mxfp4,p->options.resident_bf16,p->options.carry.export_hidden_after_head,p->options.async_inputs,p->options.resident_ids,p->options.resident_metadata,p->options.pair_ffn,p->options.fixed));
        llama_dsv41_layer_log(prefix?"DS4PERF_APPEND_ALLOCATIONS_OPEN cache_preserved=1 required_owner_admission_counted=1\n":"DS4PERF_FIXED_ALLOCATIONS_OPEN cache=1 resident=1 pinned_owner=1 before_prompt_clock=1\n");
        p->check_stop();p->state=implementation::opened;return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::suspend_device(std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
    if(p->options.yoco_keep){error="YOCO state requires local re-prefill for transport, persistence or rewind";return false;}
    try{
        require(p->state==implementation::completed&&!p->device_suspended&&p->options.fixed&&p->options.fixed->ram&&!p->options.fixed->ram_registration,
            "Device suspend requires a completed fixed owner with tile-registered carry RAM");
        require(!p->options.ctx_cache_grow_step,"Device suspend (switch arena) does not cover the growable context cache");
        auto&f=*p->options.fixed;
        ggml_backend_synchronize(p->backend);
        // The frontier is given up: nothing may export from this owner again.
        p->published.store(false);p->executor.reset();
        if(p->cache)p->cache->forget_owner_digests(); // device contents are dropped
        success(p->carry.retire_shared_ram(error),error);
        f.ram->suspend_pages();
        success(p->provider.suspend_device(error),error);
#ifdef GGML_USE_CUDA
        size_t released=0;int blocks=0;
        require(ggml_backend_cuda_switch_arena_suspend(&released,&blocks),"ggml CUDA switch arena suspend failed or is not enabled");
        size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(p->backend),&free,&total);
        llama_dsv41_layer_log("DS4COMBO_SWEEP_SUSPENDED arena_released_bytes=%zu arena_blocks=%d carry_ram_bytes=%llu device_free=%zu device_total=%zu\n",
            released,blocks,(unsigned long long)f.ram->size(),free,total);
#else
        throw std::runtime_error("Device suspend requires the CUDA switch arena");
#endif
        p->device_suspended=true;p->rebase_ready=false;return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::resume_device(std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
    try{
        require(p->state==implementation::completed&&p->device_suspended,"Device resume requires a suspended completed owner");
#ifdef GGML_USE_CUDA
        size_t mapped=0;int blocks=0;
        require(ggml_backend_cuda_switch_arena_resume(&mapped,&blocks),"ggml CUDA switch arena resume failed");
#else
        size_t mapped=0;int blocks=0;
#endif
        p->options.fixed->ram->resume_pages();
        success(p->provider.resume_device(error),error);
        llama_dsv41_layer_log("DS4COMBO_SWEEP_RESUMED arena_mapped_bytes=%zu arena_blocks=%d carry_ram_bytes=%llu contents=zeroed rebase_required=1\n",
            mapped,blocks,(unsigned long long)p->options.fixed->ram->size());
        p->device_suspended=false;p->rebase_ready=true;return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::start_next_engram(const std::vector<int32_t>&tokens,const std::vector<int32_t>&base_tail,std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
    try{
        require(p->state==implementation::opened,"Early Engram start requires an opened sweep before its run");
        require(base_tail.size()<=3,"Restored Engram history uses at most three tokens");
        llama_dsv41::engram_history initial;
        if(!base_tail.empty())success(p->provider.restored_engram_history(initial,base_tail.data(),base_tail.size(),error),error);
        success(p->provider.start_engram_prompt_early(tokens,initial,error),error);
        return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::early_engram_queued()const{return p->provider.early_engram_queued();}
bool llama_dsv41_sweep::start_append_engram(const std::vector<int32_t>&tokens,std::string&error){
    error.clear();
    if(!llama_dsv41_engram_early_enabled())return true;
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);
    if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
    try{
        // Auto (the default) arms the early start only where =1 could run it.
        if(llama_dsv41_engram_early_auto()&&!(p->options.stateful_chunks&&p->options.engram_tile_lookahead)){
            llama_dsv41_layer_log("LLAMADECODE_ENGRAM_EARLY_DECISION at=append_tokens_arrived tokens=%zu early=0 reason=%s\n",tokens.size(),
                p->options.stateful_chunks?"no_lookahead_worker":"not_stateful");return true;}
        require(p->state==implementation::completed&&p->published.load()&&!p->stop_requested.load()&&p->options.stateful_chunks&&
                !p->device_suspended&&p->options.engram_tile_lookahead,"Append early Engram requires a completed stateful owner with the lookahead worker");
        require(!tokens.empty()&&std::all_of(tokens.begin(),tokens.end(),[](int32_t t){return t>=0&&uint32_t(t)<vocabulary;}),"Append early Engram requires text tokens");
        // Same history the append's open adopts (initial_engram=final_engram).
        // Auto gap: the completed pass's layer-0 wall (the append's open and
        // layer-0 fetch run before layer 1 needs table 0).
        success(p->provider.start_engram_append(tokens,p->final_engram,p->layer0_wall_ms/1000.0,error),error);
        return true;
    }catch(const std::exception&failure){error=failure.what();return false;}
}

bool llama_dsv41_sweep::run(const std::vector<int32_t>&tokens,const cancelled&stop,std::string&error){
    const auto run_started=std::chrono::steady_clock::now();
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
    try{
        require(p->state==implementation::opened,"V4.1 sweep is single-use and requires an opened original checkpoint");
        require(tokens.size()==p->options.carry.tokens&&std::all_of(tokens.begin(),tokens.end(),[](int32_t t){return t>=0&&uint32_t(t)<vocabulary;}),
            "V4.1 sweep requires a nonempty admitted text-token sequence");
        p->check_stop(stop);p->state=implementation::running;
        require(p->tokens.size()==p->position_offset&&tokens.size()<p->options.context-p->position_offset,"Stateful token prefix/extent changed");
        p->tokens.insert(p->tokens.end(),tokens.begin(),tokens.end());
        success(p->cache->ensure_tokens(uint32_t(p->tokens.size()),"prefill",nullptr,error),error);   // growable cache: map before the sweep writes
        success(p->provider.start_request(error),error);
        const auto &owned=tokens;
        const auto schedule=llama_dsv41_plan_append(p->options.pair_ffn,p->options.carry.max_tile_tokens,p->options.compute_step_tokens,p->position_offset,p->options.unaligned_append_step_tokens);
        const uint32_t compute_step=schedule.step;
        const char*engram_overlap_env=std::getenv("LLAMA_DSV41_ENGRAM_PROMPT_OVERLAP");
        const bool engram_overlap=engram_overlap_env&&std::strcmp(engram_overlap_env,"1")==0;
        require(!engram_overlap||p->options.engram_tile_lookahead,"Engram prompt overlap requires the existing lookahead worker");
        auto prepare_engram=[&]{
            const auto started=std::chrono::steady_clock::now();std::string reason;
            success(p->provider.prepare_engram_prompt(owned,[&]{p->check_stop(stop);return false;},reason,p->initial_engram),reason);
            llama_dsv41_layer_log("[dsv41-engram-prefetch-total] wall_ms=%.6f\n",
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());std::fflush(stderr);
        };
        // Overlap: the whole-prompt row cache builds on the provider's early
        // IoRing thread from sweep start (seed and layer 0 run meanwhile);
        // prepare adopts it unfinished. Layer 1's reader waits only for table
        // 0, layer 14's for table 1 (wait_engram_table, outside the lock).
        // The early thread is joined by the table-1 wait or by invalidate.
        if(engram_overlap){
            success(p->provider.start_engram_prompt_early(owned,p->initial_engram,error,true),error);
            llama_dsv41_layer_log("DS4PERF_ENGRAM_PROMPT_QUEUED at=sweep_start consumers=1,14 per_table_ready=1 own_ioring=1\n");
        }
        prepare_engram();
        ggml_tensor*embedding=nullptr;success(p->provider.acquire_embeddings(embedding,error),error);
        const uint32_t seed_rows=p->options.batch_seed_ram?std::min(4096u,p->options.carry.max_tile_tokens):tile_limit;
        const auto seed_started=std::chrono::steady_clock::now();
        for(uint32_t start=0;start<owned.size();start+=seed_rows){
            p->check_stop(stop);const auto count=std::min(seed_rows,uint32_t(owned.size())-start);
            if(p->options.batch_seed_ram){
                llama_dsv41_carry_store::ram_lease lease;success(p->carry.acquire_ram_seed(start,count,lease,error),error);
                std::optional<llama_dsv41_carry_store::ram_completion> done;
                success(p->executor->seed_ram(embedding,std::vector<int32_t>(owned.begin()+start,owned.begin()+start+count),lease,done,error),error);
                p->check_stop(stop);require(bool(done),"Direct seed omitted completed copy proof");
                success(p->carry.complete_ram_seed(lease,*done,error),error);
            }else{
                const auto seeded=p->seed_tile(embedding,start,count);p->check_stop(stop);success(p->carry.seed(seeded,error),error);
            }
        }
        success(p->carry.finish_seed(error),error);success(p->provider.release(error),error);
        llama_dsv41_layer_log("DS4PERF_SEED_TOTAL rows=%u tokens=%zu batched=%u wall_ms=%.6f\n",seed_rows,owned.size(),unsigned(p->options.batch_seed_ram),
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-seed_started).count());
        struct engram_ready {
            uint32_t layer=0,start=0,count=0;
            llama_dsv41::engram_history history;
            std::vector<float> rows;
            double cpu_ms=0,table_wait_ms=0;
        };
            std::future<engram_ready> engram_next;
        auto queue_engram=[&](uint32_t layer,uint32_t start,const llama_dsv41::engram_history &history){
            require(!engram_next.valid(),"Engram lookahead cannot overwrite unconsumed rows");
            engram_next=std::async(std::launch::async,[&,layer,start,history]{
                const auto begun=std::chrono::steady_clock::now();engram_ready out;
                // The first decoder of each Engram layer waits for its own
                // table of the same immutable cache, outside the provider lock.
                if(engram_overlap&&start==0){std::string reason;
                    success(p->provider.wait_engram_table(layer==1?0:1,[&]{p->check_stop(stop);return false;},out.table_wait_ms,reason),reason);}
                out.layer=layer;out.start=start;out.count=std::min(compute_step,uint32_t(owned.size())-start);out.history=history;
                std::vector<uint8_t>mask(out.count,1);std::string reason;
                p->check_stop(stop);out.rows.reserve(uint64_t(out.count)*6144);std::vector<float>part;
                for(uint32_t offset=0;offset<out.count;offset+=4096){const auto rows=std::min(4096u,out.count-offset);
                    p->check_stop(stop);success(p->provider.engram_tile(layer==1?0:1,out.history,owned.data()+start+offset,mask.data()+offset,rows,part,reason),reason);out.rows.insert(out.rows.end(),part.begin(),part.end());}
                out.cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begun).count();return out;
            });
        };
        double token_acquire_ms=0,token_executor_ms=0,token_compute_ms=0,token_prepare_ms=0;
        for(uint32_t layer=0;layer<p->yoco.layers;++layer){
            const auto layer_started=std::chrono::steady_clock::now();
            auto elapsed=[](auto start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
            double upload_ms=0,compute_ms=0,readback_ms=0,engram_ms=0;
            double prepare_ms=0,cleanup_ms=0,carry_read_ms=0,carry_write_ms=0,weight_cache_ms=0;
            double executor_wall_ms=0,executor_phase_gap_ms=0;
            llama_dsv41_layer_log("[dsv41-sweep] layer=%u begin tokens=%zu\n",layer,owned.size());std::fflush(stderr);
            llama_dsv41_test_fault::at_layer(layer,owned.size());   // test-only; off unless LLAMADECODE_TEST_FAULT is set
            p->check_stop(stop);llama_dsv41_layer_graph::weights weights{};
            // LLAMADECODE B1: the decode layout leases the dense stage here; the
            // routed experts arrive through the executor's router hook.
            const bool routed_decode=p->provider.decode_layout();
            if(routed_decode){require(owned.size()==1,"Routed decode layout serves exactly one appended token");
                success(p->provider.acquire_layer_dense(layer,weights,error),error);}
            else success(p->provider.acquire_layer(layer,weights,error),error);
            const double acquire_ms=elapsed(layer_started);
            const uint32_t ratio=layer<2?0:layer<20?2:1;
            // Native ds41_index_batch/attention use floor((start+count)/ratio).
            // Use the complete prompt's fixed floor, not this tile's prefix.
            const uint32_t compressed=ratio?uint32_t(p->tokens.size())/ratio:0;
            const uint32_t selected=std::min(512u,compressed),candidates=layer>=20?std::min(2048u,(compressed+7)/8):0;
            auto ledger_started=std::chrono::steady_clock::now();
            success(p->carry.begin_layer(layer,selected,candidates,error),error);
            const double carry_begin_ms=elapsed(ledger_started);
            llama_dsv41::engram_history history=p->initial_engram;
            if(p->options.engram_tile_lookahead&&(layer==0||layer==13))queue_engram(layer+1,0,p->initial_engram);
            for(uint32_t start=llama_dsv41_yoco::first_row(p->yoco,layer);start<owned.size();start+=compute_step){
                p->check_stop(stop);const uint32_t count=std::min(compute_step,uint32_t(owned.size())-start);
                const auto carry_read_started=std::chrono::steady_clock::now();
                llama_dsv41_carry_store::tile input;llama_dsv41_carry_store::ram_lease direct;
                if(p->options.admitted_pinned_device_bytes)success(p->carry.acquire_ram_tile(start,count,direct,error),error);
                else success(p->carry.read_tile(start,count,input,error),error);
                carry_read_ms+=elapsed(carry_read_started);
                llama_dsv41_tile_executor::request request;request.layer=layer;request.start=start;request.count=count;request.compressed_rows=compressed;
                request.position_offset=p->position_offset;request.single_attention_append=schedule.single_attention_append;
                if(layer>20&&p->yoco.approximate){request.single_attention_append=false;request.yoco_first=p->yoco.first;request.yoco_raw_begin=p->position_offset+p->yoco.first;}
                request.residual=std::move(input.residual);request.previous_pre=std::move(input.previous_pre);
                if(p->options.admitted_pinned_device_bytes)request.direct=direct;
                if(p->options.admitted_prefetch_device_bytes&&uint64_t(start)+count<owned.size()){
                    llama_dsv41_carry_store::ram_lookahead next;
                    success(p->carry.lookahead_ram_tile(direct,std::min(compute_step,uint32_t(owned.size())-start-count),next,error),error);
                    request.lookahead=next;
                }
                if(routed_decode){
                    const void*dense_bank=nullptr,*routed_bank=nullptr;
                    require(p->provider.decode_banks(layer,dense_bank,routed_bank),"Routed decode banks unavailable");
                    request.dense_bank=dense_bank;request.routed_bank=routed_bank;
                    request.route_hook=[this,layer](const std::vector<int32_t>&ids,std::vector<int32_t>&slots,std::string&why){return p->provider.acquire_layer_experts(layer,ids,slots,why);};
                }
                if(!query_source(layer))request.selected=std::move(input.selected);
                if(layer>20)request.candidates=std::move(input.candidates);
                request.token_kinds.assign(count,0);
                if(layer==1||layer==14){
                    const auto engram_started=std::chrono::steady_clock::now();
                    std::vector<uint8_t>mask(count,1);request.text_mask.assign(count,1);
                    if(p->options.engram_tile_lookahead){
                        require(engram_next.valid(),"Engram lookahead omitted the required tile");
                        auto prepared=engram_next.get();
                        require(prepared.layer==layer&&prepared.start==start&&prepared.count==count&&prepared.rows.size()==uint64_t(count)*6144,
                            "Engram lookahead belongs to a different layer/tile");
                        history=prepared.history;request.engram_rows=std::move(prepared.rows);
                        llama_dsv41_layer_log("DS4PERF_ENGRAM_LOOKAHEAD layer=%u start=%u count=%u cpu_ms=%.6f exposed_wait_ms=%.6f prepared_prompt_cache=1\n",layer,start,count,prepared.cpu_ms,elapsed(engram_started));
                        if(engram_overlap&&start==0)llama_dsv41_layer_log("DS4PERF_ENGRAM_TABLE_READY table=%u consumer_layer=%u exposed_wait_ms=%.6f worker_table_wait_ms=%.6f\n",
                            layer==1?0u:1u,layer,elapsed(engram_started),prepared.table_wait_ms);
                        if(start+count<owned.size())queue_engram(layer,start+count,history);
                    }else if(count<=4096)success(p->provider.engram_tile(layer==1?0:1,history,owned.data()+start,mask.data(),count,request.engram_rows,error),error);
                    else{request.engram_rows.reserve(uint64_t(count)*6144);std::vector<float>part;
                        for(uint32_t offset=0;offset<count;offset+=4096){const auto rows=std::min(4096u,count-offset);
                            success(p->provider.engram_tile(layer==1?0:1,history,owned.data()+start+offset,mask.data()+offset,rows,part,error),error);request.engram_rows.insert(request.engram_rows.end(),part.begin(),part.end());}}
                    engram_ms+=elapsed(engram_started);
                }
                p->check_stop(stop);llama_dsv41_tile_executor::result computed;
                ledger_started=std::chrono::steady_clock::now();
                success(p->executor->run(request,weights,computed,error,p->options.expert),error);
                const double executor_ms=elapsed(ledger_started);executor_wall_ms+=executor_ms;
                executor_phase_gap_ms+=executor_ms-computed.prepare_ms-computed.cleanup_ms-computed.upload_ms-computed.compute_ms-computed.readback_ms-computed.weight_cache_ms;
                p->check_stop(stop);
                weight_cache_ms+=computed.weight_cache_ms;prepare_ms+=computed.prepare_ms;cleanup_ms+=computed.cleanup_ms;
                upload_ms+=computed.upload_ms;compute_ms+=computed.compute_ms;readback_ms+=computed.readback_ms;
                require(computed.selected_width==selected&&computed.candidate_width==candidates,"V4.1 executor/carry output widths disagree");
                llama_dsv41_carry_store::tile output;output.start=start;output.count=count;
                output.selected_width=selected;output.candidate_width=candidates;
                output.residual=std::move(computed.residual);output.previous_pre=std::move(computed.next_pre);
                output.selected=std::move(computed.selected);output.candidates=std::move(computed.candidates);
                const auto carry_write_started=std::chrono::steady_clock::now();
                if(p->options.admitted_pinned_device_bytes){
                    require(bool(computed.direct_completion),"Direct pinned executor omitted actual copy/validation completion");
                    success(p->carry.complete_ram_tile(direct,*computed.direct_completion,error),error);
                }else success(p->carry.write_tile(output,error),error);
                carry_write_ms+=elapsed(carry_write_started);
                if(p->options.diagnostic_carry){
                    void *bank=nullptr;uint64_t bytes=0;success(p->carry.ram_storage(bank,bytes,error),error);
                    p->options.diagnostic_carry(layer,start,count,bank,bytes,p->cache->raw(layer));
                }
            }
            if(layer==1)p->final_engram=history;
            if(layer==14)require(history.tail==p->final_engram.tail,"Engram layers disagreed on the completed chunk history");
            ledger_started=std::chrono::steady_clock::now();success(p->carry.commit_layer(error),error);
            const double carry_commit_ms=elapsed(ledger_started);
            ledger_started=std::chrono::steady_clock::now();success(p->provider.release(error),error);
            const double provider_release_ms=elapsed(ledger_started);
            const double ledger_wall_ms=elapsed(layer_started);
            if(layer==0)p->layer0_wall_ms=ledger_wall_ms;
            token_acquire_ms+=acquire_ms;token_executor_ms+=executor_wall_ms;token_compute_ms+=compute_ms;token_prepare_ms+=prepare_ms;
            const double scaffold_ms=ledger_wall_ms-acquire_ms-carry_begin_ms-executor_wall_ms-engram_ms-carry_read_ms-carry_write_ms-carry_commit_ms-provider_release_ms;
            llama_dsv41_layer_log("DS4PERF_LAYER_WALL_LEDGER layer=%u wall_ms=%.6f acquire_ms=%.6f carry_begin_ms=%.6f executor_wall_ms=%.6f executor_phase_gap_ms=%.6f engram_ms=%.6f carry_read_ms=%.6f carry_write_ms=%.6f carry_commit_ms=%.6f provider_release_ms=%.6f scaffold_ms=%.6f\n",layer,ledger_wall_ms,acquire_ms,carry_begin_ms,executor_wall_ms,executor_phase_gap_ms,engram_ms,carry_read_ms,carry_write_ms,carry_commit_ms,provider_release_ms,scaffold_ms);
            require(!p->options.diagnostic_layers||layer+1<p->options.diagnostic_layers,"Intentional incomplete DS4PERF diagnostic stop; no frontier authority");
            llama_dsv41_layer_log("[dsv41-perf] layer=%u wall_ms=%.6f acquire_ms=%.6f upload_ms=%.6f compute_ms=%.6f readback_ms=%.6f engram_ms=%.6f prepare_ms=%.6f cleanup_ms=%.6f carry_read_ms=%.6f carry_write_ms=%.6f weight_cache_ms=%.6f\n",
                layer,elapsed(layer_started),acquire_ms,upload_ms,compute_ms,readback_ms,engram_ms,prepare_ms,cleanup_ms,carry_read_ms,carry_write_ms,weight_cache_ms);std::fflush(stderr);
            llama_dsv41_layer_log("[dsv41-sweep] layer=%u complete\n",layer);std::fflush(stderr);
        }
        require(!engram_next.valid(),"Unconsumed Engram lookahead at completed frontier");
        if(p->yoco.encoder_only){
            success(p->provider.finish_yoco_encoder(error),error);p->check_stop(stop);
            ggml_backend_synchronize(p->backend);
            p->measured.prompt_processing_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-run_started).count();
            p->state=implementation::encoder_complete;
            llama_dsv41_layer_log("ULTRAYOCO_ENCODER_COMPLETE position=%zu streamed_layers=21 decoder_fetches=0 frontier_valid=0\n",p->tokens.size());
            return true;
        }
        std::optional<llama_dsv41_carry_store::tile>resident_last;
        if(p->options.carry.export_hidden_after_head){
            resident_last.emplace();success(p->executor->resident_head_row(*resident_last,error),error);
        }else p->executor.reset();
        p->check_stop(stop);ggml_tensor*norm=nullptr,*weight=nullptr;
        success(p->provider.acquire_head(norm,weight,error),error);p->head(norm,weight,resident_last?&*resident_last:nullptr);
        success(p->provider.release(error),error);p->check_stop(stop);ggml_backend_synchronize(p->backend);
        const auto head_complete=std::chrono::steady_clock::now();
        p->measured.prompt_processing_ms=std::chrono::duration<double,std::milli>(head_complete-run_started).count();
        if(!llama_dsv41_layer_log_on()){ // LLAMADECODE: one line per decode step instead of per-layer logs
            uint64_t ds_bytes=0;double ds_ms=0;p->provider.fetch_totals(ds_bytes,ds_ms);const auto cache=p->provider.cache_stats();
            std::fprintf(stderr,"LLAMADECODE_TOKEN position=%u wall_ms=%.3f ds_bytes=%llu ds_transfer_ms=%.3f stage_wait_ms=%.3f executor_ms=%.3f compute_ms=%.3f prepare_ms=%.3f cache_slots=%u cache_hits_total=%llu cache_misses_total=%llu l2_slots=%u l2_hits_total=%llu\n",
                p->position_offset,p->measured.prompt_processing_ms,(unsigned long long)ds_bytes,ds_ms,token_acquire_ms,token_executor_ms,token_compute_ms,token_prepare_ms,cache.slots,(unsigned long long)cache.hits,(unsigned long long)cache.misses,cache.l2_slots,(unsigned long long)cache.l2_hits);
        }
        llama_dsv41_layer_log("DS4PERF_PROMPT_HEAD_COMPLETE tokens=%zu processing_ms=%.6f bulk_hidden_export_pending=%u\n",owned.size(),p->measured.prompt_processing_ms,unsigned(p->options.carry.export_hidden_after_head));
        if(p->options.carry.export_hidden_after_head){
            for(uint32_t first=p->yoco.first;first<owned.size();first+=p->options.carry.max_tile_tokens){
                p->check_stop(stop);llama_dsv41_carry_store::ram_lease lease;
                success(p->carry.acquire_ram_export(first,std::min(p->options.carry.max_tile_tokens,uint32_t(owned.size())-first),lease,error),error);
                std::optional<llama_dsv41_carry_store::ram_completion>completion;
                success(p->executor->export_hidden_tile(lease,completion,error),error);p->check_stop(stop);
                require(bool(completion),"Final hidden export omitted completed validation/copy proof");
                success(p->carry.complete_ram_export(lease,*completion,error),error);
            }
            p->executor.reset();
        }
        success(p->carry.certify_frontier(error),error);p->check_stop(stop);
        p->measured.export_preparation_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-head_complete).count();
        p->state=implementation::completed;p->published.store(true);return true;
    }catch(const std::bad_alloc&){p->poison();error="std::bad_alloc: a host allocation failed during the V4.1 sweep run";return false;}
    catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::export_native(const consumer&send,std::string&error){
    {
        std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);
        if(!lock.owns_lock()){error="V4.1 sweep is already active";return false;}
        if(p->options.yoco_keep){error="YOCO approximate state cannot be exported to RTX/Mac; re-prefill with YOCO off";return false;}
    }
    return inspect_local(send,error);
}
bool llama_dsv41_sweep::inspect_local(const consumer&send,std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep/export already owns the frontier";return false;}
    try{
        p->check_stop();require(p->state==implementation::completed&&p->published.load()&&p->carry.frontier_ready()&&p->cache&&bool(send),
            "V4.1 native export requires a completely certified private frontier");
        ggml_backend_synchronize(p->backend);
        const auto position=uint32_t(p->tokens.size());
        require(!p->options.capture_dspark||p->cache->dspark_ready(position),"DSpark completed layer frontiers disagree with tokens");
        const llama_dsv41::wire_frontier frontier{p->options.context,position,p->options.capture_dspark?3u:1u,
            p->options.capture_dspark?position:0u,p->options.capture_dspark?std::min(position,128u):0u};
        llama_dsv41::wire_payload_reader reader;
        success(reader.init(frontier,p->tokens.data(),p->tokens.size(),p->logits.data(),p->logits.size(),
            [&](const llama_dsv41::state_span&span,uint64_t offset,void*dst,size_t bytes){
                if(p->stop_requested.load())return false;
                std::string read_error;return p->cache->read_native_span(span,offset,dst,bytes,read_error);
            },error),error);
        success(send(reader,error),error);p->check_stop();return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::owner_digests(std::vector<std::string>&out,const std::vector<std::string>*expected,std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep/export already owns the frontier";return false;}
    try{
        p->check_stop();require(p->state==implementation::completed&&p->published.load()&&p->carry.frontier_ready()&&p->cache,
            "V4.1 owner digests require a completely certified private frontier");
        ggml_backend_synchronize(p->backend);
        const auto position=uint32_t(p->tokens.size());
        require(!p->options.capture_dspark||p->cache->dspark_ready(position),"DSpark completed layer frontiers disagree with tokens");
        const llama_dsv41::wire_frontier frontier{p->options.context,position,p->options.capture_dspark?3u:1u,
            p->options.capture_dspark?position:0u,p->options.capture_dspark?std::min(position,128u):0u};
        success(p->cache->owner_digests(frontier,out,expected,error),error);p->check_stop();return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}

bool llama_dsv41_sweep::apply_decode_delta(const decode_delta&update,
        const llama_dsv41::wire_read_span&read,std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep already owns the frontier";return false;}
    if(p->options.yoco_keep){error="YOCO state requires local re-prefill for transport, persistence or rewind";return false;}
    try{
        p->check_stop();require(p->state==implementation::completed&&p->published.load()&&
            p->options.stateful_chunks&&p->cache&&p->carry.frontier_ready()&&!p->executor,
            "Decode delta requires a completed, idle stateful sweep");
        const auto count=update.evaluated_tokens.size();
        require(count&&count<=llama_dsv41::decode_delta_max_tokens&&update.logits.size()==vocabulary&&p->logits.size()==vocabulary&&bool(read),
            "Decode delta requires 1..maximum_decode_tokens evaluated tokens and complete logits");
        require(std::all_of(update.evaluated_tokens.begin(),update.evaluated_tokens.end(),
            [](int32_t t){return t>=0&&uint32_t(t)<vocabulary;}),"Decode delta token out of range");
        finite(update.logits);
        const uint32_t position=uint32_t(p->tokens.size());
        const llama_dsv41::wire_frontier base{p->options.context,position,p->options.capture_dspark?3u:1u,
            p->options.capture_dspark?position:0u,p->options.capture_dspark?std::min(position,128u):0u};
        llama_dsv41::delta_plan plan;success(llama_dsv41::make_delta_plan(base,update.target,plan,error),error);
        require(plan.token_count==count&&p->tokens.capacity()>=update.target.position,
            "Decode delta positions exceed evaluated tokens or admitted token storage");
        auto valid_sha=[](const std::string&s){return s.size()==64&&std::all_of(s.begin(),s.end(),
            [](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});};
        require(valid_sha(update.base_tokens_sha256)&&valid_sha(update.target_tokens_sha256)&&valid_sha(update.target_logits_sha256),
            "Decode delta requires canonical token/logit SHA256 identities");
        require(dsv41_boundary::sha256(p->tokens.data(),p->tokens.size()*4)==update.base_tokens_sha256,
            "Decode delta base token sequence mismatch");
        const uint64_t old_bytes=uint64_t(p->tokens.size())*4,new_bytes=uint64_t(count)*4;
        const auto token_hash=dsv41_boundary::sha256_stream(old_bytes+new_bytes,[&](uint64_t off,void*dst,size_t bytes){
            auto*out=static_cast<unsigned char*>(dst);
            if(off<old_bytes){const size_t n=size_t(std::min<uint64_t>(bytes,old_bytes-off));
                std::memcpy(out,reinterpret_cast<const unsigned char*>(p->tokens.data())+off,n);off+=n;out+=n;bytes-=n;}
            if(bytes){if(off<old_bytes||off-old_bytes>new_bytes||bytes>new_bytes-(off-old_bytes))return false;
                std::memcpy(out,reinterpret_cast<const unsigned char*>(update.evaluated_tokens.data())+(off-old_bytes),bytes);}
            return true;
        });
        require(token_hash==update.target_tokens_sha256&&
            dsv41_boundary::sha256(update.logits.data(),update.logits.size()*4)==update.target_logits_sha256,
            "Decode delta target token/logit identity mismatch");
        auto history=p->final_engram;
        success(p->provider.advance_decoded_engram(history,update.evaluated_tokens.data(),count,error),error);
        p->published.store(false);p->state=implementation::poisoned;
        success(p->cache->apply_native_delta(base,update.target,update.base_owner_sha256,update.target_owner_sha256,
            [&](const llama_dsv41::state_span&s,uint64_t off,void*dst,size_t bytes){
                return !p->stop_requested.load()&&read(s,off,dst,bytes);
            },error),error);
        p->check_stop();
        p->tokens.insert(p->tokens.end(),update.evaluated_tokens.begin(),update.evaluated_tokens.end());
        std::copy(update.logits.begin(),update.logits.end(),p->logits.begin());p->final_engram=history;
        llama_dsv41_layer_log("DS4PERF_DECODE_DELTA_APPLIED base=%u target=%u tokens=%zu state_bytes=%llu base_and_target_owner_hashes_verified=1 evaluated_on_RTX=0\n",
            position,update.target.position,count,(unsigned long long)plan.state_bytes);
        p->state=implementation::completed;p->published.store(true);return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
    catch(...){p->poison();error="Unknown V4.1 decoded state update failure";return false;}
}

bool llama_dsv41_sweep::capture_rewind_anchor(rewind_anchor&out,std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep already owns the frontier";return false;}
    if(p->options.yoco_keep){error="YOCO state requires local re-prefill for transport, persistence or rewind";return false;}
    try{
        p->check_stop();require(p->state==implementation::completed&&p->published.load()&&
            p->options.stateful_chunks&&p->cache&&!p->executor,
            "Rewind anchor requires a complete idle mode1/3 owner");
        out={};const uint32_t pos=uint32_t(p->tokens.size());
        out.frontier={p->options.context,pos,p->options.capture_dspark?3u:1u,
            p->options.capture_dspark?pos:0u,p->options.capture_dspark?std::min(pos,128u):0u};
        llama_dsv41::wire_plan plan;success(llama_dsv41::make_wire_plan(out.frontier,plan,error),error);
        uint64_t bytes=uint64_t(vocabulary)*4;
        for(const auto&s:plan.spans)if(s.kind!=llama_dsv41::state_kind::compressed&&s.kind!=llama_dsv41::state_kind::index)bytes+=s.bytes;
        require(bytes<=24ull*1024*1024,"Native rewind anchor exceeds explicit24MiB ceiling");
        out.logits=p->logits;out.history=p->final_engram;
        out.tokens_sha256=dsv41_boundary::sha256(p->tokens.data(),p->tokens.size()*4);
        out.logits_sha256=dsv41_boundary::sha256(out.logits.data(),out.logits.size()*4);
        out.mutable_owners.resize(plan.spans.size());
        for(size_t i=0;i<plan.spans.size();++i){const auto&s=plan.spans[i];
            const bool mutable_owner=s.kind!=llama_dsv41::state_kind::compressed&&s.kind!=llama_dsv41::state_kind::index;
            if(mutable_owner)out.mutable_owners[i].resize(size_t(s.bytes));
            out.owner_sha256.push_back(dsv41_boundary::sha256_stream(s.bytes,[&](uint64_t off,void*dst,size_t n){
                p->check_stop();if(!p->cache->read_native_span(s,off,dst,n,error))return false;
                if(mutable_owner)std::memcpy(out.mutable_owners[i].data()+off,dst,n);return true;
            }));
        }
        llama_dsv41_layer_log("DS4PERF_REWIND_ANCHOR position=%u bytes=%llu full_cache_copy=0\n",pos,(unsigned long long)bytes);
        return true;
    }catch(const std::exception&e){out={};p->poison();error=e.what();return false;}
}

bool llama_dsv41_sweep::restore_rewind_anchor(const rewind_anchor&anchor,
        const std::vector<std::string>&current_hashes,std::string&error){
    std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);error.clear();
    if(!lock.owns_lock()){error="V4.1 sweep already owns the frontier";return false;}
    if(p->options.yoco_keep){error="YOCO state requires local re-prefill for transport, persistence or rewind";return false;}
    try{
        p->check_stop();const uint32_t old=uint32_t(p->tokens.size()),target=anchor.frontier.position;
        require(p->state==implementation::completed&&p->published.load()&&p->options.stateful_chunks&&
            p->cache&&!p->executor&&target&&target<old&&anchor.frontier.context==p->options.context&&
            anchor.frontier.mode==(p->options.capture_dspark?3u:1u)&&anchor.logits.size()==vocabulary,
            "Rewind requires a saved earlier mode1/3 frontier in the same owner");
        require(dsv41_boundary::sha256(p->tokens.data(),size_t(target)*4)==anchor.tokens_sha256&&
            dsv41_boundary::sha256(anchor.logits.data(),anchor.logits.size()*4)==anchor.logits_sha256,
            "Rewind anchor token/logit identity differs");
        const llama_dsv41::wire_frontier base{p->options.context,old,p->options.capture_dspark?3u:1u,
            p->options.capture_dspark?old:0u,p->options.capture_dspark?std::min(old,128u):0u};
        llama_dsv41::wire_plan plan;success(llama_dsv41::make_wire_plan(anchor.frontier,plan,error),error);
        require(plan.spans.size()==anchor.mutable_owners.size(),"Rewind anchor owner extent differs");
        p->published.store(false);p->state=implementation::poisoned;
        success(p->cache->rewind_native_frontier(base,anchor.frontier,current_hashes,anchor.owner_sha256,
            [&](const llama_dsv41::state_span&s,uint64_t off,void*dst,size_t n){
                p->check_stop();for(size_t i=0;i<plan.spans.size();++i)if(plan.spans[i].kind==s.kind&&plan.spans[i].index==s.index){
                    const auto&bytes=anchor.mutable_owners[i];if(off>bytes.size()||n>bytes.size()-off)return false;
                    std::memcpy(dst,bytes.data()+off,n);return true;
                }return false;
            },error),error);
        p->tokens.resize(target);std::copy(anchor.logits.begin(),anchor.logits.end(),p->logits.begin());p->final_engram=anchor.history;
        p->state=implementation::completed;p->published.store(true);
        llama_dsv41_layer_log("DS4PERF_REWIND base=%u target=%u token_and_all_owner_hashes_verified=1 evaluated_on_RTX=0\n",old,target);
        return true;
    }catch(const std::exception&e){p->poison();error=e.what();return false;}
}

void llama_dsv41_sweep::cancel(){p->stop_requested.store(true);p->published.store(false);}
llama_dsv41_sweep::run_timing llama_dsv41_sweep::timing()const{
    std::lock_guard<std::mutex>lock(p->mutex);return (p->state==implementation::completed||p->state==implementation::encoder_complete)?p->measured:run_timing{};
}
bool llama_dsv41_sweep::ready()const{return p->published.load()&&!p->stop_requested.load();}
bool llama_dsv41_sweep::device_state_views(device_state&out,std::string&error)const{
    std::lock_guard<std::mutex>lock(p->mutex);out={};error.clear();
    if(p->state!=implementation::completed||!p->published.load()||!p->cache||p->options.capture_dspark){
        error="External decode requires a completed canonical mode1 owner";return false;}
    const bool bf16=p->options.compressed_cache_bf16;const ggml_type owner_type=bf16?GGML_TYPE_BF16:GGML_TYPE_F32;
    if(p->cache->compact_compressed()!=bf16){error="External decode cache precision differs from admission";return false;}out.bf16=bf16;
    for(uint32_t l=0;l<40;++l){auto*t=p->cache->raw(l);if(!t||!t->data||t->type!=GGML_TYPE_F32||ggml_nbytes(t)!=128ull*512*4){error="Raw ring geometry differs";return false;}out.raw[l]=t->data;}
    const uint32_t owners[4]={2,8,14,20};
    for(uint32_t i=0;i<4;++i){auto*c=p->cache->compressed(owners[i]);auto*x=p->cache->index(owners[i]);
        if(!c||!x||!c->data||!x->data||c->type!=owner_type||x->type!=owner_type||c->ne[0]!=512||x->ne[0]!=128||c->ne[1]!=x->ne[1]){error="Compressed/index cache geometry differs";return false;}
        out.compressed[i]=c->data;out.index[i]=x->data;out.rows[i]=uint32_t(c->ne[1]);
        if(i<3){auto*k=p->cache->pool_kv(owners[i]);auto*s=p->cache->pool_score(owners[i]);
            if(!k||!s||!k->data||!s->data||ggml_nbytes(k)!=512*4||ggml_nbytes(s)!=512*4){error="Pool tail geometry differs";return false;}
            out.pool_kv[i]=k->data;out.pool_score[i]=s->data;}}
    return true;
}
bool llama_dsv41_sweep::engram_tail(std::array<int32_t,3>&tail,std::string&error)const{
    std::lock_guard<std::mutex>lock(p->mutex);error.clear();
    if(p->state!=implementation::completed||!p->published.load()){error="Engram history requires a completed owner";return false;}
    tail=p->final_engram.tail;return true;
}
bool llama_dsv41_sweep::commit_external_token(int32_t token,const std::vector<float>&logits,std::string&error){
    std::lock_guard<std::mutex>lock(p->mutex);error.clear();
    try{
        require(p->state==implementation::completed&&p->published.load()&&p->options.stateful_chunks&&!p->executor,
            "External decode commit requires a completed, idle stateful owner");
        require(token>=0&&uint32_t(token)<vocabulary&&logits.size()==vocabulary&&p->tokens.size()+1<p->options.context,"External decode token/logits/context invalid");
        finite(logits);
        auto history=p->final_engram;
        success(p->provider.advance_decoded_engram(history,&token,1,error),error);
        p->tokens.push_back(token);std::copy(logits.begin(),logits.end(),p->logits.begin());p->final_engram=history;
        return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}
bool llama_dsv41_sweep::commit_external_tokens(const int32_t*tokens,size_t count,const std::vector<float>&logits,std::string&error){
    std::lock_guard<std::mutex>lock(p->mutex);error.clear();
    try{
        require(p->state==implementation::completed&&p->published.load()&&p->options.stateful_chunks&&!p->executor,
            "External append commit requires a completed, idle stateful owner");
        require(tokens&&count&&count<=128&&logits.size()==vocabulary&&p->tokens.size()+count<p->options.context,"External append tokens/logits/context invalid");
        for(size_t i=0;i<count;++i)require(tokens[i]>=0&&uint32_t(tokens[i])<vocabulary,"External append token out of vocabulary");
        finite(logits);
        auto history=p->final_engram;
        success(p->provider.advance_decoded_engram(history,tokens,count,error),error);
        p->tokens.insert(p->tokens.end(),tokens,tokens+count);std::copy(logits.begin(),logits.end(),p->logits.begin());p->final_engram=history;
        return true;
    }catch(const std::exception&failure){p->poison();error=failure.what();return false;}
}
llama_dsv41_weight_provider*llama_dsv41_sweep::provider_for_external_decode(){return &p->provider;}
bool llama_dsv41_sweep::ensure_context(uint32_t tokens,const char*phase,bool*grew,std::string&error){
    std::lock_guard<std::mutex>lock(p->mutex);if(grew)*grew=false;
    if(!p->cache){error="Context cache growth requires an allocated owner";return false;}
    return p->cache->ensure_tokens(tokens,phase,grew,error);
}
uint64_t llama_dsv41_sweep::context_next_step_bytes()const{std::lock_guard<std::mutex>lock(p->mutex);return p->cache?p->cache->next_step_bytes():0;}
bool llama_dsv41_sweep::compact_for_external_decode(std::string&error){
    std::lock_guard<std::mutex>lock(p->mutex);error.clear();
    try{
        require(p->state==implementation::completed&&!p->executor&&p->options.fixed,"External decode compaction requires a completed idle fixed owner");
        if(!p->options.fixed->decode_compact&&decode_compact_enabled())compact_fixed(*p->options.fixed,p->options.admitted_hidden_bytes);
        return true;
    }catch(const std::exception&failure){error=failure.what();return false;}
}
bool llama_dsv41_sweep::completed_logits(std::vector<float>&out,std::string&error)const{
    std::lock_guard<std::mutex>lock(p->mutex);out.clear();error.clear();
    if(p->state!=implementation::completed||!p->published.load()||p->stop_requested.load()||p->logits.size()!=vocabulary){
        error="Completed logits require a published completed V4.1 frontier";return false;}
    out=p->logits;return true;
}
