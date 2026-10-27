#include "llama-dsv41-layer-log.h"
#include "llama-dsv41-compact-proof.h"
#include "ggml-dsv41-bf16-inplace.h"
#include "llama-dsv41-weight-casts.h"
#include "ggml-dsv41-hc-half.h"
#include "llama-dsv41-graph-digest.h"
#include "llama-dsv41-graph-bindings.h"
#include <array>
#include "llama-dsv41-executor.h"
#include "llama-dsv41-dense-half-admission.h"
#include "llama-dsv41-route-split.h"
#include "llama-dsv41-hash-dump.h"
#include "ggml-dsv41-expert-routing.h"
#include "ggml-dsv41-topk.h"
#include "ggml-dsv41-stream-topk.h"
#include "ggml-dsv41-attention-g16.h"
#include "ggml-dsv41-rope-inplace.h"
#include "ggml-dsv41-bf16-vector.h"
#include "ggml-dsv41-dense-bf16-store.h"
#include "ggml-dsv41-cast-vector.h"
#include "ggml-dsv41-mul-f16-store.h"
#include "ggml-dsv41-hc-expand.h"
#include "boundary-capture.h"
#include "llama-dsv41-rope.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"
#include "llama-dsv41-direct.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#endif
#include <cstdlib>
#include "ggml-ds4-native-half.h"
#include "ggml-ds4-native-half-stats.h"
#include "ggml-dsv41-index-cublas.h"
#include "llama-dsv41-index-visible-proof.h"
#include "ggml-dsv41-dense-half.h"
#include "ggml-dsv41-shared-sparse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <mutex>
#include <string>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
using clock_type=std::chrono::steady_clock;
double elapsed(clock_type::time_point start){return std::chrono::duration<double,std::milli>(clock_type::now()-start).count();}
// LLAMADECODE_TILE_TIMING=1: one diagnostic line per tile (host phase walls plus GPU
// event time around the tile graph). Off = no events, no extra calls, no output.
bool tile_timing_enabled(){static const bool on=[]{const char*v=std::getenv("LLAMADECODE_TILE_TIMING");return v&&std::string(v)=="1";}();return on;}
// LLAMADECODE: LLAMADECODE_COLLECT_ADMISSION=1 is a diagnostic-only mode. Admission,
// tag-count, reserve-estimate and dispatch-witness checks that do not guard memory
// safety log their violation once per (check, layer, rows) and continue; every other
// check still throws (logged as FATAL). The default (unset) refuses as before.
thread_local uint32_t collect_layer=UINT32_MAX,collect_rows=0;
bool collect_mode(){static const bool on=[]{const char*e=std::getenv("LLAMADECODE_COLLECT_ADMISSION");return e&&!std::strcmp(e,"1");}();return on;}
bool collectable(const char*m){
    if(!std::strncmp(m,"Actual ",7)&&(std::strstr(m,"witness")||std::strstr(m,"coverage")||std::strstr(m,"fixed reserve")))return true;
    static const char*const listed[]={"Dense Q8_1 activation toggle matched no original Q8 projection",
        "Explicit F16 dense toggle matched no admitted projection","F16 dense tag applied at a row count the shape rule excludes",
        "Explicit count/prefix/scatter routing did not match exactly gate/up/down",
        "F16 route-weight store fusion must match exactly one expert roundtrip","No bounded exact tail RoPE reuse candidates",
        "Paired graph has no exact BF16 lifetime reuse candidates","F16 expert cuBLAS requires exactly three expert products",
        "F16 expert cuBLAS shape admission refused","Original Q8 projection graph coverage changed",
        "No single-reader original-Q8 BF16 stores to fuse","Index score exceeds explicit bounded GLM shape admission"};
    for(const char*l:listed)if(!std::strcmp(m,l))return true;
    return false;
}
void require(bool value,const char*message){
    if(value)return;
    if(collect_mode()){
        static std::mutex seen_mutex;static std::set<std::string>seen;
        const bool go_on=collectable(message);
        const std::string key=std::string(message)+"|"+std::to_string(collect_layer)+"|"+std::to_string(collect_rows);
        bool fresh;{std::lock_guard<std::mutex>g(seen_mutex);fresh=seen.insert(key).second;}
        if(fresh||!go_on){std::fprintf(stderr,"LLAMADECODE_COLLECT %s check=\"%s\" layer=%u rows=%u\n",go_on?"CONTINUE":"FATAL",message,collect_layer,collect_rows);std::fflush(stderr);}
        if(go_on)return;
    }
    throw std::runtime_error(message);
}
bool index_f32_cublas_enabled(){
    const char*v=std::getenv("LLAMA_DSV41_INDEX_F32_CUBLAS");
    require(!v||!std::strcmp(v,"0")||!std::strcmp(v,"1"),"Invalid explicit F32 indexer environment");
    return v&&!std::strcmp(v,"1");
}
bool query_source(uint32_t layer){return layer==2||layer==8||layer==14||layer==20||layer==24||layer==28||layer==32||layer==36;}
bool kv_source(uint32_t layer){return layer==2||layer==8||layer==14||layer==20;}
bool engram_layer(uint32_t layer){return layer==1||layer==14;}
ggml_tensor*concat_balanced(ggml_context*ctx,std::vector<ggml_tensor*>parts,int dim){
    require(!parts.empty(),"Empty native projection concatenation");
    // Copy the same ordered slices with a balanced tree. No reduction or
    // projection arithmetic changes; long prefixes are not copied repeatedly.
    while(parts.size()>1){
        std::vector<ggml_tensor*>next;next.reserve((parts.size()+1)/2);
        for(size_t i=0;i<parts.size();i+=2)
            next.push_back(i+1<parts.size()?ggml_concat(ctx,parts[i],parts[i+1],dim):parts[i]);
        parts=std::move(next);
    }
    return parts.front();
}
bool same_compact_ids(const ggml_tensor *value,const ggml_tensor *input){return llama_dsv41_compact_ids_unchanged(value,input);}
bool bf16_rounded_output(const ggml_tensor *t,unsigned depth=0){
    if(!t||depth>8||t->type!=GGML_TYPE_F32)return false;
    if(t->op==GGML_OP_CONT)return bf16_rounded_output(t->src[0],depth+1);
    if(t->op==GGML_OP_CONCAT){const auto *a=t->src[0],*b=t->src[1];
        return t->op_params[0]==2&&a&&b&&a->ne[0]==t->ne[0]&&b->ne[0]==t->ne[0]&&a->ne[1]==t->ne[1]&&b->ne[1]==t->ne[1]&&a->ne[2]+b->ne[2]==t->ne[2]&&bf16_rounded_output(a,depth+1)&&bf16_rounded_output(b,depth+1);
    }
    return ggml_dsv41_hc_bf16_requested(t)||(t->op==GGML_OP_DSV4_ACT_QAT&&t->op_params[0]==GGML_DSV41_QAT_BF16)||(t->op==GGML_OP_CPY&&t->src[0]&&t->src[0]->type==GGML_TYPE_BF16);
}
bool finite_scalar(float value){
    static_assert(sizeof(float)==sizeof(uint32_t)&&std::numeric_limits<float>::is_iec559,"V4.1 requires IEEE F32");
    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));return (bits&0x7f800000u)!=0x7f800000u;
}
template<typename T>void upload(ggml_tensor*t,const std::vector<T>&values,ggml_backend_t async_backend=nullptr){
    require(t&&ggml_nbytes(t)==values.size()*sizeof(T),"V4.1 tile upload geometry changed");
    const auto before=clock_type::now();
    if(async_backend)ggml_backend_tensor_set_async(async_backend,t,values.data(),0,values.size()*sizeof(T));
    else ggml_backend_tensor_set(t,values.data(),0,values.size()*sizeof(T));
    if(std::getenv("DS4PERF_RING_TRACE")&&elapsed(before)>1)llama_dsv41_layer_log("DS4PERF_UPLOAD_API tensor=%s bytes=%zu host_ms=%.6f\n",t->name,values.size()*sizeof(T),elapsed(before));
}
template<typename T>std::vector<T>download(ggml_tensor*t,size_t maximum){
    if(!t)return {};
    require(t->type==(std::is_same<T,float>::value?GGML_TYPE_F32:GGML_TYPE_I32),"V4.1 tile readback type changed");
    require(ggml_nelements(t)>0&&uint64_t(ggml_nelements(t))<=maximum,"V4.1 tile readback exceeded its bounded geometry");
    require(ggml_nbytes(t)==size_t(ggml_nelements(t))*sizeof(T),"V4.1 tile readback element type changed");
    std::vector<T>out(ggml_nelements(t));ggml_backend_tensor_get(t,out.data(),0,out.size()*sizeof(T));return out;
}
void finite(const std::vector<float>&values){
    require(std::all_of(values.begin(),values.end(),finite_scalar),"Nonfinite V4.1 tile activation");
}
void backend_buffers(ggml_backend_t backend,ggml_cgraph*graph){
    auto check=[&](const ggml_tensor*t){if(t&&t->buffer)
        require(ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(t->buffer)),
            "V4.1 layer weight/cache is not staged on the admitted compute backend");};
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){
        const auto*node=ggml_graph_node(graph,i);check(node);
        if(ggml_ds4_native_half_requested(node)) {
            const auto device=ggml_backend_get_device(backend);
            const auto registry=device?ggml_backend_dev_backend_reg(device):nullptr;
            require(registry&&std::strcmp(ggml_backend_reg_name(registry),"CUDA")==0&&
                ggml_ds4_native_half_shape_valid(node),"DS4.1 native-half graph requires validated CUDA expert dispatch");
        }
        for(const auto*source:node->src)check(source);
        require(ggml_backend_supports_op(backend,node),"Admitted V4.1 backend does not support a graph operation");
    }
}
void weights_present(const llama_dsv41_layer_graph::weights&w,uint32_t layer){
    const auto&a=w.attention;const auto&f=w.ffn;
    for(auto*t:{a.q_a,a.q_norm,a.q_b,a.kv,a.kv_norm,a.out_a,a.out_b,a.attn_sinks,
                w.attn_fn,w.attn_scale,w.attn_base,w.attn_norm,w.ffn_fn,w.ffn_scale,w.ffn_base,w.ffn_norm,
                f.router,f.text_bias,f.image_bias,f.gate,f.up,f.down,f.shared_gate,f.shared_up,f.shared_down})
        require(t&&t->data&&t->buffer,"Missing required staged V4.1 layer weight");
    if(kv_source(layer))for(auto*t:{a.compressor,a.compressor_norm,a.index_k,a.index_norm})
        require(t&&t->data&&t->buffer,"Missing staged V4.1 shared-cache producer weight");
    if(layer==2||layer==8||layer==14)require(a.gate&&a.gate->data&&a.gate->buffer,"Missing V4.1 pool2 gate weight");
    if(query_source(layer))for(auto*t:{a.index_q,a.index_weights})require(t&&t->data&&t->buffer,"Missing staged V4.1 index-query weight");
    if(engram_layer(layer))for(auto*t:{w.engram_wkv,w.engram_q_norm,w.engram_k_norm})
        require(t&&t->data&&t->buffer,"Missing staged V4.1 Engram projection weight");
}
}

// Shared with the model-free admission/lifetime fixture. Production uses a
// 96 MiB per-operation backend temporary ceiling; this function allocates only
// graph metadata. Caller preflights all live tensors plus this hidden reserve.
static bool dsv41_original_q8_dense(const ggml_tensor *node) {
    if(!node||node->op!=GGML_OP_MUL_MAT||node->type!=GGML_TYPE_F32)return false;
    const auto *w=node->src[0],*x=node->src[1];
    if(!w||!x||w->type!=GGML_TYPE_Q8_0||x->type!=GGML_TYPE_F32||
       w->ne[0]<=0||w->ne[0]>16384||w->ne[1]<=0||w->ne[1]>32768||
        x->ne[0]!=w->ne[0]||x->ne[1]<=8||x->ne[1]>4096||
       w->ne[2]!=1||w->ne[3]!=1||x->ne[2]!=1||x->ne[3]!=1||
       !ggml_is_contiguous(w)||x->nb[0]!=sizeof(float))return false;
    const auto *root=w;for(unsigned depth=0;depth<32&&root->view_src;++depth)root=root->view_src;
    if(root->op!=GGML_OP_NONE||!root->buffer||!root->data||std::strncmp(root->name,"blk.",4))return false;
    // Same projection allowlist as ggml-dsv41-dense-half.h; HC/router/head absent.
    const char *allowed[]={".attn_q_a.weight",".attn_q_b.weight",".attn_kv.weight",
        ".attn_output_a.weight",".attn_output_b.weight",".attn_compressor_kv.weight",
        ".attn_compressor_gate.weight",".indexer.attn_k.weight",".indexer.attn_q_b.weight",
        ".indexer.proj.weight",".ffn_gate_shexp.weight",".ffn_up_shexp.weight",
        ".ffn_down_shexp.weight",".engram_kv.weight"};
    const size_t len=std::strlen(root->name);
    for(const char *suffix:allowed){const size_t n=std::strlen(suffix);if(len>=n&&!std::strcmp(root->name+len-n,suffix))return true;}
    return false;
}

ggml_cgraph *llama_dsv41_native_batch_graph(ggml_context *ctx,ggml_cgraph *original,
        uint32_t count,uint64_t operation_limit,uint64_t *backend_temporary,
        uint32_t *rounded_matrices,uint32_t *split_matrices) {
    require(ctx&&original&&operation_limit&&backend_temporary&&rounded_matrices&&split_matrices,
        "Invalid native batch graph rewrite");
    *backend_temporary=0;*rounded_matrices=0;*split_matrices=0;
    const int nodes=ggml_graph_n_nodes(original);
    for(int i=0;i<nodes;++i){
        auto*node=ggml_graph_node(original,i);
        if(node->op!=GGML_OP_MUL_MAT)continue;
        auto*weight=node->src[0];auto*input=node->src[1];
        if(node->op_params[6]==0x51384131){
            require(dsv41_original_q8_dense(node)&&node->op_params[0]==GGML_PREC_DEFAULT,"Original Q8 dense MMQ tag lost its explicit projection proof");
            // Actual two-dimensional Q8_1 RHS packing, with the existing 512
            // row padding and a conservative1MiB tail/launch margin. Large
            // projections fit the same96MiB reserve; wider shapes refuse.
            const uint64_t padded=(uint64_t(input->ne[0])+511)/512*512;
            const uint64_t rhs=uint64_t(input->ne[1])*padded*36/32+1024*1024;
            require(rhs<=operation_limit,"Original Q8_1 RHS packing exceeds the explicit per-operation reserve");
            *backend_temporary=std::max(*backend_temporary,operation_limit);continue;
        }
        const bool native_q8=count>8&&input->ne[1]>8&&weight->type==GGML_TYPE_Q8_0;
        // Native batch>8 Q8 matrix kernels stage dequantized weights as half.
        // Existing CPU/CUDA CPY supports Q8->F32, then F32->F16, so bound each
        // row/group piece BEFORE both casts. Never expand a whole layer matrix.
        const uint64_t row_bytes=uint64_t(weight->ne[0])*sizeof(float);
        const uint64_t matrix_bytes=row_bytes*uint64_t(weight->ne[1]);
        // LLAMADECODE B0: a <=8-row step (1-row decode) keeps its small-row
        // product but is sliced by output rows the same way, so a large F16
        // weight (Engram KV, 600 MiB as F32) is never expanded whole. Prefill
        // steps (>8 rows) are unchanged.
        const bool small_step=count<=8;
        const bool split_f16=(small_step||(count>8&&input->ne[1]>8))&&weight->type==GGML_TYPE_F16&&
            node->op_params[0]==GGML_PREC_F32&&matrix_bytes>operation_limit;
        if(!native_q8&&!split_f16)continue;
        require(weight->ne[3]==1&&input->ne[3]==1&&
                (weight->ne[2]==1||weight->ne[2]==input->ne[2]),
            "Native bounded projection has unsupported group/broadcast geometry");
        const uint64_t piece_limit=native_q8?std::min<uint64_t>(operation_limit,32ull*1024*1024):operation_limit;
        const uint64_t rows=std::min<uint64_t>(4096,piece_limit/row_bytes);
        require(rows>0,"One native F32 projection row exceeds backend temporary admission");
        std::vector<ggml_tensor*>group_parts;
        const int64_t groups=std::max(weight->ne[2],input->ne[2]);
        for(int64_t group=0;group<groups;++group){
            auto*group_input=input;
            if(groups>1)group_input=ggml_view_2d(ctx,input,input->ne[0],input->ne[1],
                input->nb[1],size_t(input->ne[2]>1?group:0)*input->nb[2]);
            std::vector<ggml_tensor*>row_parts;
            for(int64_t first=0;first<weight->ne[1];first+=int64_t(rows)){
                const int64_t n=std::min<int64_t>(int64_t(rows),weight->ne[1]-first);
                auto*part=ggml_view_2d(ctx,weight,weight->ne[0],n,weight->nb[1],
                    size_t(first)*weight->nb[1]+size_t(weight->ne[2]>1?group:0)*weight->nb[2]);
                if(native_q8){
                    part=ggml_cast(ctx,ggml_cast(ctx,part,GGML_TYPE_F32),GGML_TYPE_F16);
                    ggml_set_name(part,"dsv41_native_q8_weight_f16");
                }
                auto*product=ggml_mul_mat(ctx,part,group_input);
                ggml_mul_mat_set_prec(product,GGML_PREC_F32);
                if(const char *profile=std::getenv("LLAMA_CUDA_OP_PROFILE_NODES")){if(profile[0]=='1')
                    ggml_format_name(product,"dsv41_mm:%s:g%lld:r%lld",weight->name,(long long)group,(long long)first);}
                row_parts.push_back(product);
            }
            auto*joined=concat_balanced(ctx,std::move(row_parts),0);
            if(groups>1){
                auto*group_result=ggml_reshape_3d(ctx,joined,joined->ne[0],joined->ne[1],1);
                group_parts.push_back(group_result);
            }else group_parts.push_back(joined);
        }
        auto*groups_joined=concat_balanced(ctx,std::move(group_parts),2);
        require(groups_joined&&std::memcmp(node->ne,groups_joined->ne,sizeof(node->ne))==0&&
                std::memcmp(node->nb,groups_joined->nb,sizeof(node->nb))==0,
            "Sliced native projection changed its output geometry");
        // Preserve tensor identity already referenced by factory outputs and
        // downstream nodes. The replacement has identical shape/storage order.
        node->op=groups_joined->op;
        std::memcpy(node->op_params,groups_joined->op_params,sizeof(node->op_params));
        std::memcpy(node->src,groups_joined->src,sizeof(node->src));
        node->view_src=groups_joined->view_src;
        node->view_offs=groups_joined->view_offs;
        if(native_q8)++*rounded_matrices;
        if(uint64_t(weight->ne[1])>rows||groups>1)++*split_matrices;
    }
    auto*graph=ggml_new_graph_custom(ctx,ggml_graph_size(original),false);
    // Preserve the original side-effect ordering (cache publication/raw commit)
    // while recursively introducing each new cast/sliced product dependency.
    for(int i=0;i<nodes;++i)ggml_build_forward_expand(graph,ggml_graph_node(original,i));
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){
        auto*node=ggml_graph_node(graph,i);
        if(node->op!=GGML_OP_MUL_MAT||node->op_params[0]!=GGML_PREC_F32)continue;
        const auto*weight=node->src[0];
        if(weight->type!=GGML_TYPE_F32){
            uint64_t bytes=uint64_t(weight->ne[0])*uint64_t(weight->ne[1])*sizeof(float);
            // Strided grouped RHS is copied to contiguous F32 before GEMM.
            if(!ggml_is_contiguous(node->src[1]))bytes+=uint64_t(ggml_nelements(node->src[1]))*sizeof(float);
            *backend_temporary=std::max(*backend_temporary,bytes);
        }
    }
    return graph;
}

// Preflight is separate from allocation so rejected admission never creates a
// device buffer. Kept externally visible for the model-free allocator fixture.
uint64_t llama_dsv41_allocate_native_graph(ggml_gallocr_t allocation,ggml_cgraph*graph,
        uint64_t backend_temporary,uint64_t admitted_graph_bytes){
    size_t planned=0;
    ggml_gallocr_reserve_n_size(allocation,graph,nullptr,nullptr,&planned);
    if(backend_temporary>admitted_graph_bytes||planned>admitted_graph_bytes-backend_temporary)
        llama_dsv41_layer_log("DS4PERF_GRAPH_ADMISSION_REFUSED planned=%llu temporary=%llu admitted=%llu\n",(unsigned long long)planned,(unsigned long long)backend_temporary,(unsigned long long)admitted_graph_bytes);
    require(backend_temporary<=admitted_graph_bytes&&planned<=admitted_graph_bytes-backend_temporary,
        "V4.1 live graph plus backend F32 temporary exceeds admitted graph ceiling");
    // reserve_n_size records a valid plan but deliberately creates no buffers;
    // explicitly materialize it after admission before initializing tensors.
    require(ggml_gallocr_reserve(allocation,graph),"V4.1 admitted lifetime graph reservation failed");
    require(ggml_gallocr_alloc_graph(allocation,graph),"V4.1 admitted lifetime graph allocation failed");
    const uint64_t actual=ggml_gallocr_get_buffer_size(allocation,0);
    require(actual<=admitted_graph_bytes-backend_temporary,"V4.1 lifetime allocation exceeded preflight admission");
    return actual+backend_temporary;
}

// Same one-bank hidden-state lifetime as decode_layer_major's GPU mode.
// The RAM mirror and all existing readback/finite/ID checks remain authoritative.
// This first bounded step removes repeated H2D only; it does not omit validation.
struct llama_dsv41_tile_executor::resident_storage {
    ggml_context_ptr ctx;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_tensor *hidden=nullptr,*pre=nullptr,*selected=nullptr,*candidates=nullptr;
    ggml_tensor *raw_ids=nullptr,*raw_mask=nullptr,*history_ids=nullptr,*raw_freq=nullptr,*compressed_freq=nullptr,*zero_kinds=nullptr,*text_mask=nullptr;
    uint32_t metadata_rows=0;
    uint32_t selected_source=UINT32_MAX,candidate_source=UINT32_MAX,selected_width=0,candidate_width=0;
    ggml_backend_t backend=nullptr;
    uint32_t tokens=0,total_tokens=0,layer=0,cursor=0,export_cursor=0,yoco_first=0;
    // LLAMADECODE single sweep: tokens = physical bank rows. With spill, logical
    // rows >= spill_start live in their RAM carry records between layers and
    // pass through the bank's last tile (physical rows spill_start..tokens).
    uint32_t spill_start=UINT32_MAX,slot_logical=UINT32_MAX;
    uint64_t spill_h2d_bytes=0,spill_d2h_bytes=0;
    bool spill()const{return spill_start!=UINT32_MAX;}
    bool full()const{return tokens==total_tokens||spill();}
    bool spilled(uint32_t start)const{return spill()&&start>=spill_start;}
    uint32_t physical(uint32_t start)const{return spilled(start)?spill_start:start;}
    std::shared_ptr<ggml_backend_buffer> reserved_owner;
    void layout(uint32_t n,uint32_t tile,bool resident_bf16,bool resident_ids,bool resident_metadata){
        ctx.reset(ggml_init({65536,nullptr,true}));require(bool(ctx),"Resident hidden metadata allocation failed");
        hidden=ggml_new_tensor_3d(ctx.get(),resident_bf16?GGML_TYPE_BF16:GGML_TYPE_F32,5120,4,n);
        pre=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4,n);
        if(resident_ids){
            selected=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,512,n);
            candidates=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,2048,n);
        }
        if(resident_metadata){
            metadata_rows=std::min(n,tile);
            raw_ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,128,metadata_rows);
            raw_mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,128,metadata_rows);
            history_ids=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,127);
            raw_freq=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
            compressed_freq=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
            zero_kinds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,metadata_rows);
            text_mask=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I8,metadata_rows);
        }
    }
    ~resident_storage(){if(buffer){ggml_backend_synchronize(backend);if(!reserved_owner)ggml_backend_buffer_free(buffer);}}
};

struct llama_dsv41_tile_executor::pinned_storage {
    void *base=nullptr;uint64_t bytes=0;
    bool current_bank_copies_forbidden=false;
    ggml_backend_t backend=nullptr;
#ifdef GGML_USE_CUDA
    bool registered=false,tile_registration=false,pageable_copies=false;
    std::vector<std::vector<uint8_t>> pending_host_uploads;
    void *registered_tile=nullptr;uint64_t registered_tile_bytes=0;
    cudaStream_t copies=nullptr;
    cudaEvent_t ready=nullptr;
    uint32_t *flags_device=nullptr,*flags_host=nullptr;
    double last_register_ms=0; // LLAMADECODE_TILE_TIMING: host wall of the last bind_lease unregister+register
#include "llama-dsv41-copy-prefetch.inl"
    static void ok(cudaError_t e,const char *message){if(e!=cudaSuccess)throw std::runtime_error(std::string(message)+": "+cudaGetErrorString(e));}
    static void cleanup(cudaError_t e){if(e!=cudaSuccess){llama_dsv41_layer_log("DSV41_PINNED_COPY_CLEANUP_FAILED %s\n",cudaGetErrorString(e));std::fflush(stderr);std::abort();}}
    ~pinned_storage(){
        if(copies)cleanup(cudaStreamSynchronize(copies));
        if(backend)ggml_backend_synchronize(backend);
        if(registered_tile)cleanup(cudaHostUnregister(registered_tile));
        if(registered)cleanup(cudaHostUnregister(base));
        prefetch_destroy_events();
        if(flags_device)cleanup(cudaFree(flags_device));
        if(flags_host)cleanup(cudaFreeHost(flags_host));
        if(ready)cleanup(cudaEventDestroy(ready));
        if(copies)cleanup(cudaStreamDestroy(copies));
    }
    cudaStream_t compute()const{return static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(backend));}
    void validate(const llama_dsv41_validation &p){
        if(p.residual_n>uint64_t(5120)*4*4096){
            require(p.residual_n%(5120*4)==0,"Grouped validation residual row geometry changed");
            const uint64_t rows=p.residual_n/(5120*4);require(rows<=16384&&p.pre_n==rows*4,"Grouped validation admission changed");
            require(p.mean_n%rows==0&&p.engram_n%rows==0&&p.selected_n%rows==0&&p.candidates_n%rows==0&&
                p.mean_n/rows<=5120&&p.engram_n/rows<=6144&&p.selected_n/rows<=512&&p.candidates_n/rows<=2048,
                "Grouped validation compact widths changed");
            for(uint64_t start=0;start<rows;start+=4096){const uint64_t count=std::min<uint64_t>(4096,rows-start);auto q=p;
                auto part=[&](auto ptr,uint64_t total){return ptr?ptr+start*(total/rows):nullptr;};
                q.residual=part(p.residual,p.residual_n);q.residual_n=count*(p.residual_n/rows);
                q.pre=part(p.pre,p.pre_n);q.pre_n=count*(p.pre_n/rows);
                q.mean=part(p.mean,p.mean_n);q.mean_n=count*(p.mean_n/rows);
                q.engram=part(p.engram,p.engram_n);q.engram_n=count*(p.engram_n/rows);
                q.selected=part(p.selected,p.selected_n);q.selected_n=count*(p.selected_n/rows);
                q.candidates=part(p.candidates,p.candidates_n);q.candidates_n=count*(p.candidates_n/rows);validate(q);
            }
            return;
        }
        const auto before=clock_type::now();
        ok(static_cast<cudaError_t>(llama_dsv41_cuda_validate(p,flags_device,compute())),"GPU carry validator launch failed");
        ok(cudaMemcpyAsync(flags_host,flags_device,4,cudaMemcpyDeviceToHost,compute()),"GPU carry flags readback failed");
        ok(cudaStreamSynchronize(compute()),"GPU carry validation did not complete");
        require(*flags_host==0,"GPU carry contains nonfinite F32 or out-of-range compact IDs");
        if(std::getenv("DS4PERF_RING_TRACE")&&elapsed(before)>1)llama_dsv41_layer_log("DS4PERF_VALIDATION_API host_ms=%.6f\n",elapsed(before));
    }
    void bind_lease(const llama_dsv41_carry_store::ram_lease &lease,const llama_dsv41_carry_store::ram_lookahead *lookahead=nullptr){
        if(!tile_registration)return;
        require(!registered&&lease.base==base&&lease.bytes==bytes&&lease.count(),"Tile pin requires its exact existing RAM lease");
        (void)lease.row(0);(void)lease.row(lease.count()-1);
        uint32_t next_rows=0;const uint8_t*last=lease.row(lease.count()-1);
        if(lookahead){const auto&next=lookahead->lease;
            require(prefetch_buffer&&!lease.seeding&&!next.seeding&&next.base==lease.base&&next.bytes==lease.bytes&&
                next.store_id==lease.store_id&&next.generation==lease.generation&&
                next.first==uint64_t(lease.first)+lease.rows&&next.rows&&next.rows<=prefetch_capacity&&
                next.sw==lease.sw&&next.cw==lease.cw&&next.next_sw==lease.next_sw&&next.next_cw==lease.next_cw,
                "Tile registration lookahead differs from its exact live parent");
            (void)lookahead->row(0);last=lookahead->row(next.rows-1);next_rows=next.rows;
        }
        // row() is authoritative: the existing RAM bank has a4KiB header
        // and3,584B padding after each32-record physical page.
        auto *first=lease.row(0);
        const uint64_t offset=uint64_t(first-static_cast<uint8_t*>(base));
        const uint64_t size=uint64_t(last-first)+llama_dsv41_carry_store::ram_record_bytes;
        require(offset<=bytes&&size<=bytes-offset&&size<=2ull*1024*1024*1024,"Tile pin escaped admitted RAM bank or2GiB bound");
        void *ptr=first;
        if(registered_tile==ptr&&registered_tile_bytes==size)return;
        // Original copy stream completion protects the old registration. No
        // new host buffer or stream; the bank/lease ownership stays unchanged.
        ok(cudaStreamSynchronize(copies),"Previous tile copies did not complete before pin rebinding");
        const auto register_started=clock_type::now();
        if(registered_tile){ok(cudaHostUnregister(registered_tile),"Previous tile RAM unregister failed");registered_tile=nullptr;registered_tile_bytes=0;}
        size_t free_before=0,total=0,free_after=0;ok(cudaMemGetInfo(&free_before,&total),"Cannot inspect GPU memory before tile registration");
        ok(cudaHostRegister(ptr,size_t(size),cudaHostRegisterDefault),"Active tile RAM registration failed");
        registered_tile=ptr;registered_tile_bytes=size;
        last_register_ms=elapsed(register_started); // includes the cudaMemGetInfo between unregister and register
        ok(cudaMemGetInfo(&free_after,&total),"Cannot inspect GPU memory after tile registration");
        llama_dsv41_layer_log("DS4PERF_TILE_HOST_REGISTRATION first=%u count=%u bytes=%llu previous_copy_stream_complete=1 existing_RAM_bank=1\n",lease.start(),lease.count()+next_rows,(unsigned long long)size);
        llama_dsv41_layer_log("DS4PERF_TILE_PIN_WINDOW first=%u current_rows=%u next_rows=%u bytes=%llu cuda_free_before=%llu cuda_free_after=%llu exact_parent_generation=1\n",lease.start(),lease.count(),next_rows,(unsigned long long)size,(unsigned long long)free_before,(unsigned long long)free_after);
    }
    void queue(bool to_device,const llama_dsv41_carry_store::ram_lease &lease,uint64_t offset,uint64_t width,ggml_tensor *tensor){
        require(!current_bank_copies_forbidden,"A resident-only reader attempted an unregistered RAM bank copy");
        if(pageable_copies){
            require(!registered&&!tile_registration&&!prefetch_buffer&&lease.base==base&&lease.bytes==bytes&&width&&offset<=llama_dsv41_carry_store::ram_record_bytes&&width<=llama_dsv41_carry_store::ram_record_bytes-offset&&tensor&&tensor->data&&ggml_is_contiguous(tensor)&&ggml_nbytes(tensor)==width*lease.count(),"Pageable vector copy escaped its exact resident lease/tensor");
            (void)lease.row(0);(void)lease.row(lease.count()-1);
            std::vector<uint8_t> values(size_t(width*lease.count()));
            if(to_device){
                for(uint32_t row=0;row<lease.count();++row)std::memcpy(values.data()+uint64_t(row)*width,lease.row(row)+offset,size_t(width));
                // Reuse the original backend vector upload API and retain its
                // source through the existing validator/copy completion.
                pending_host_uploads.push_back(std::move(values));
                ggml_backend_tensor_set_async(backend,tensor,pending_host_uploads.back().data(),0,size_t(width*lease.count()));
            }else{
                // Original synchronous backend download, then the same CPU
                // RAM-row scatter as the pre-existing non-direct vector path.
                ggml_backend_tensor_get(tensor,values.data(),0,values.size());
                for(uint32_t row=0;row<lease.count();++row)std::memcpy(lease.row(row)+offset,values.data()+uint64_t(row)*width,size_t(width));
            }
            llama_dsv41_layer_log("DS4PERF_RESIDENT_PAGEABLE_COPY to_device=%u first=%u count=%u offset=%llu width=%llu bytes=%llu existing_backend_vector_API=1\n",unsigned(to_device),lease.start(),lease.count(),(unsigned long long)offset,(unsigned long long)width,(unsigned long long)(width*lease.count()));
            return;
        }
        if(tile_registration){
            const uintptr_t begin=reinterpret_cast<uintptr_t>(registered_tile),at=reinterpret_cast<uintptr_t>(lease.row(0));
            const uint64_t extent=uint64_t(lease.row(lease.count()-1)-lease.row(0))+llama_dsv41_carry_store::ram_record_bytes;
            require(lease.base==base&&lease.bytes==bytes&&registered_tile&&at>=begin&&at-begin<=registered_tile_bytes&&extent<=registered_tile_bytes-(at-begin),"Copy uses an unregistered/foreign current or lookahead span");
        }
        const auto before=clock_type::now();
        require(width&&offset<=llama_dsv41_carry_store::ram_record_bytes&&
            width<=llama_dsv41_carry_store::ram_record_bytes-offset&&tensor&&tensor->data&&
            ggml_is_contiguous(tensor)&&ggml_nbytes(tensor)==width*lease.count(),"Direct pinned tensor/record geometry changed");
        for(uint32_t first=0;first<lease.count();first+=4096){
            const ggml_dsv41_ram_slice slice{base,bytes,lease.start()+first,std::min(4096u,lease.count()-first)};
            ok(static_cast<cudaError_t>(llama_dsv41_cuda_copy_ram(to_device,slice,offset,width,
                static_cast<char*>(tensor->data)+uint64_t(first)*width,copies)),"Direct pinned 2D carry copy failed");
        }
        if(std::getenv("DS4PERF_RING_TRACE")&&elapsed(before)>1)llama_dsv41_layer_log("DS4PERF_PINNED_COPY_API to_device=%u bytes=%llu host_ms=%.6f\n",unsigned(to_device),(unsigned long long)(width*lease.count()),elapsed(before));
    }
    void upload_ready(){ok(cudaEventRecord(ready,copies),"Pinned upload fence record failed");ok(cudaStreamWaitEvent(compute(),ready,0),"Compute stream did not accept pinned upload fence");}
    void readback_ready(){ok(cudaEventRecord(ready,compute()),"Pinned readback fence record failed");ok(cudaStreamWaitEvent(copies,ready,0),"Copy stream did not accept compute fence");}
    void finish(){ok(cudaStreamSynchronize(copies),"Direct pinned carry readback did not complete");
        if(pageable_copies&&!pending_host_uploads.empty()){ggml_backend_synchronize(backend);pending_host_uploads.clear();}
    }
#endif
};

struct llama_dsv41_tile_executor::graph_storage {
    ggml_gallocr_t value=nullptr;ggml_backend_t backend=nullptr;void *pending_copies=nullptr;
    struct variant {
        std::array<uint32_t,13> key{};
        ggml_context_ptr ctx;
        llama_dsv41_layer_graph::inputs input{};
        llama_dsv41_layer_graph::cache state{};
        llama_dsv41_layer_graph::result values{};
        ggml_cgraph *graph=nullptr;
        uint64_t backend_temporary=0;
        uint32_t rounded_matrices=0,split_matrices=0;
        llama_dsv41_graph_bindings bindings;
    };
    std::shared_ptr<ggml_gallocr> reserved_owner;
    std::unique_ptr<llama_dsv41_weight_cast_cache> casts;
    bool cache_layouts=false;
    uint32_t layer=UINT32_MAX;
    std::array<unsigned char,sizeof(llama_dsv41_layer_graph::weights)> weight_identity{};
    std::vector<std::unique_ptr<variant>> variants;
    ~graph_storage(){if(value){
#ifdef GGML_USE_CUDA
        if(pending_copies)pinned_storage::cleanup(cudaStreamSynchronize(static_cast<cudaStream_t>(pending_copies)));
#endif
        ggml_backend_synchronize(backend);if(!reserved_owner)ggml_gallocr_free(value);
    }}
};

uint64_t llama_dsv41_tile_executor::resident_hidden_bytes(uint32_t tokens){
    require(tokens&&tokens<=1048576,"Invalid resident hidden token count");
    return uint64_t(tokens)*(5120*4+4)*sizeof(float)+65536;
}

uint64_t llama_dsv41_tile_executor::resident_layout_bytes(ggml_backend_t backend,uint32_t tokens,uint32_t tile,bool bf16,bool ids,bool metadata){
    require(backend&&tokens&&tokens<=1048576&&tile&&tile<=16384,"Invalid resident layout query");
    resident_storage plan;plan.layout(tokens,tile,bf16,ids,metadata);
    const uint64_t bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(plan.ctx.get(),ggml_backend_get_default_buffer_type(backend));
    require(bytes,"Empty resident layout query");return bytes;
}

std::shared_ptr<ggml_backend_buffer> llama_dsv41_tile_executor::reserve_hidden(ggml_backend_t backend,uint32_t tokens,uint32_t tile,bool bf16,bool ids,bool metadata,uint64_t admission){
    require(backend&&tokens&&tile,"Invalid fixed hidden reservation geometry");
    resident_storage plan;plan.layout(tokens,tile,bf16,ids,metadata);
    const auto buft=ggml_backend_get_default_buffer_type(backend);
    const uint64_t bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(plan.ctx.get(),buft);
    require(bytes&&bytes<=admission,"Fixed resident layout exceeds its existing hidden admission");
    auto *buffer=ggml_backend_buft_alloc_buffer(buft,size_t(bytes));
    require(buffer&&ggml_backend_buffer_get_size(buffer)<=admission,"Fixed hidden reservation failed or exceeded admission");
    return std::shared_ptr<ggml_backend_buffer>(buffer,ggml_backend_buffer_free);
}

std::shared_ptr<void> llama_dsv41_tile_executor::register_fixed_ram(ggml_backend_t backend,const std::shared_ptr<llama_dsv41_carry_store::ram_backing>&ram){
#ifdef GGML_USE_CUDA
    require(backend&&ggml_backend_is_cuda(backend)&&ram&&ram->data()&&ram->size()&&ram->size()<=SIZE_MAX,"Fixed RAM registration geometry/backend changed");
    size_t before=0,total=0,after=0;pinned_storage::ok(cudaMemGetInfo(&before,&total),"Cannot inspect GPU memory before fixed RAM registration");
    pinned_storage::ok(cudaHostRegister(ram->data(),size_t(ram->size()),cudaHostRegisterDefault),"Fixed RAM carry registration before provider import failed");
    pinned_storage::ok(cudaMemGetInfo(&after,&total),"Cannot inspect GPU memory after fixed RAM registration");
    llama_dsv41_layer_log("DS4PERF_FIXED_RAM_PIN_CUDA_MEMORY bytes=%llu cuda_free_before=%llu cuda_free_after=%llu\n",(unsigned long long)ram->size(),(unsigned long long)before,(unsigned long long)after);
    llama_dsv41_layer_log("DS4PERF_RAM_REGISTERED_BEFORE_IMPORT bytes=%llu owned_fixed_lifetime=1\n",(unsigned long long)ram->size());
    return std::shared_ptr<void>(ram->data(),[ram](void *base){pinned_storage::cleanup(cudaHostUnregister(base));});
#else
    (void)backend;(void)ram;throw std::runtime_error("Fixed registered RAM requires CUDA");
#endif
}

llama_dsv41_tile_executor::~llama_dsv41_tile_executor()=default;

bool llama_dsv41_tile_executor::seed_ram(ggml_tensor *embedding,const std::vector<int32_t> &ids,
        const llama_dsv41_carry_store::ram_lease &lease,
        std::optional<llama_dsv41_carry_store::ram_completion> &completion,std::string &error,const std::vector<float>*projected){
    completion.reset();error.clear();std::lock_guard<std::mutex>lock(cache.execution_mutex);
    try{
#ifdef GGML_USE_CUDA
        require(!cache.execution_poisoned&&pinned&&workspace&&workspace->value&&!workspace->cache_layouts,
            "Direct seed requires the admitted pinned bank and allocator-only workspace");
        require(lease.seeding&&lease.base==pinned->base&&lease.bytes==pinned->bytes&&lease.store_id&&lease.generation&&
            !lease.sw&&!lease.cw&&!lease.next_sw&&!lease.next_cw&&lease.rows&&lease.rows<=admitted_tile_tokens&&ids.size()==lease.rows,
            "Direct seed differs from the exact initializing RAM span");
        (void)lease.row(0);(void)lease.row(lease.rows-1);
        pinned->bind_lease(lease);
        require(embedding&&embedding->data&&embedding->buffer&&embedding->ne[0]==5120&&embedding->ne[1]==129280&&embedding->ne[2]==1&&embedding->ne[3]==1,
            "Direct seed embedding geometry changed");
        require(std::all_of(ids.begin(),ids.end(),[](int32_t x){return x>=0&&x<129280;}),"Direct seed token is outside vocabulary");
        cache.execution_poisoned=true;
        ggml_context_ptr ctx(ggml_init({8*1024*1024,nullptr,true}));require(bool(ctx),"Direct seed metadata allocation failed");
        auto *tokens=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,lease.rows);
        llama_dsv41_graph_ops ops(ctx.get(),norm_eps,hc_eps,sinkhorn_iters);
        require(!projected||(lease.rows<=128&&projected->size()==size_t(5120)*lease.rows),"Projected image seed exceeds bounded input rows");
        auto*rows=projected?ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,5120,lease.rows):ggml_get_rows(ctx.get(),embedding,tokens);
        auto values=ops.seed_embeddings(rows);
        auto *graph=ggml_new_graph_custom(ctx.get(),8192,false);
        ggml_build_forward_expand(graph,values.residual);ggml_build_forward_expand(graph,values.pre);
        backend_buffers(backend,graph);
        const auto scratch=llama_dsv41_allocate_native_graph(workspace->value,graph,0,admitted_graph_bytes);
        if(projected)ggml_backend_tensor_set(rows,projected->data(),0,projected->size()*4);
        else upload(tokens,ids);
        // The existing GPU validator below synchronizes this same stream
        // before publishing initialization credit or queuing the RAM mirror.
        require(ggml_backend_graph_compute_async(backend,graph)==GGML_STATUS_SUCCESS,"Direct seed graph failed");
        llama_dsv41_validation check;check.residual=static_cast<const float*>(values.residual->data);
        check.residual_n=uint64_t(5120)*4*lease.rows;check.pre=static_cast<const float*>(values.pre->data);check.pre_n=uint64_t(4)*lease.rows;
        pinned->validate(check);pinned->readback_ready();
        pinned->queue(false,lease,0,81920,values.residual);pinned->queue(false,lease,81920,16,values.pre);pinned->finish();
        completion=llama_dsv41_carry_store::ram_completion(lease,true,true,true);
        llama_dsv41_layer_log("DS4PERF_SEED_RAM_COMPLETE start=%u count=%u mirrored_host_bytes=%llu scratch_bytes=%llu validated=1\n",
            lease.first,lease.rows,(unsigned long long)(uint64_t(lease.rows)*81936),(unsigned long long)scratch);
        cache.execution_poisoned=false;return true;
#else
        (void)embedding;(void)ids;(void)lease;throw std::runtime_error("Direct seed requires CUDA");
#endif
    }catch(const std::exception &e){if(backend)ggml_backend_synchronize(backend);error=e.what();return false;}
}

llama_dsv41_tile_executor::llama_dsv41_tile_executor(ggml_backend_t backend,llama_dsv41_cache_storage&cache,
    uint64_t admitted_graph_bytes,float norm_eps,float hc_eps,uint32_t sinkhorn_iters,float swiglu_limit,float expert_scale,uint32_t admitted_tile_tokens,
    uint32_t resident_hidden_tokens,uint64_t admitted_hidden_bytes,const pinned_span *direct,bool reuse_graph,bool reuse_layer_graph,uint64_t admitted_cast_bytes,bool audit_casts,uint64_t admitted_index_bytes,uint64_t admitted_prefetch_bytes,bool coalesce_dense,bool defer_hidden_readback,bool merge_attention_selected_rows,bool hc_projection_f16_gemm,bool dense_mxfp4,bool resident_bf16,bool export_hidden_after_head,bool async_inputs,bool resident_ids,bool resident_metadata,bool pair_ffn,std::shared_ptr<llama_dsv41_fixed_storage> fixed):
    backend(backend),cache(cache),admitted_graph_bytes(admitted_graph_bytes),norm_eps(norm_eps),hc_eps(hc_eps),
    swiglu_limit(swiglu_limit),expert_scale(expert_scale),sinkhorn_iters(sinkhorn_iters),admitted_tile_tokens(admitted_tile_tokens),resident_bf16(resident_bf16),export_hidden_after_head(export_hidden_after_head),async_inputs(async_inputs),resident_ids(resident_ids),resident_metadata(resident_metadata),defer_hidden_readback(defer_hidden_readback),merge_attention_selected_rows(merge_attention_selected_rows),hc_projection_f16_gemm(hc_projection_f16_gemm),dense_mxfp4(dense_mxfp4),pair_ffn(pair_ffn),fixed(fixed){
#ifdef GGML_USE_CUDA
    require(!async_inputs||(direct&&ggml_backend_is_cuda(backend)&&std::getenv("GGML_CUDA_DISABLE_GRAPHS")&&std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))=="1"),"Async inputs require actual pinned CUDA and disabled capture");
#else
    require(!async_inputs,"Async inputs require the CUDA backend");
#endif
    require(!export_hidden_after_head||defer_hidden_readback,"Post-head export requires full-device hidden retention");
    require(!resident_metadata||(resident_bf16&&direct&&defer_hidden_readback),"Resident metadata requires full BF16 hidden and pinned RAM mirror");
    require(!resident_ids||(resident_bf16&&direct&&defer_hidden_readback),"Resident compact IDs require full BF16 hidden and pinned RAM mirror");
    require(!resident_bf16||defer_hidden_readback,"BF16 hidden storage requires the full-device deferred readback contract");
    auto resident_required=[&](uint32_t n){return resident_bf16?
        resident_layout_bytes(backend,n,admitted_tile_tokens,resident_bf16,resident_ids,resident_metadata):resident_hidden_bytes(n);};
    require(!pair_ffn||((admitted_tile_tokens==4096||admitted_tile_tokens==8192||admitted_tile_tokens==16384)&&direct&&reuse_graph&&!reuse_layer_graph&&((defer_hidden_readback&&!admitted_prefetch_bytes)||(!resident_hidden_tokens&&!admitted_hidden_bytes&&admitted_prefetch_bytes&&!resident_bf16&&!resident_ids&&!resident_metadata))),
        "Paired FFN requires explicit4K/8K/16K group and existing admitted resident/RAM allocator path");
    require(!dense_mxfp4||(admitted_cast_bytes&&reuse_graph&&!reuse_layer_graph&&!audit_casts),"Dense FP4 requires the existing admitted cast bank and allocator-only mode");
    // LLAMADECODE single sweep: a fixed bank smaller than the sweep spills its tail rows (exact bytes) to RAM.
    const uint32_t spill_bank=fixed&&fixed->hidden_rows&&resident_hidden_tokens>fixed->hidden_rows&&defer_hidden_readback&&resident_bf16&&resident_ids&&resident_metadata&&
        export_hidden_after_head&&fixed->hidden_rows%admitted_tile_tokens==0&&fixed->hidden_rows>=2*admitted_tile_tokens?fixed->hidden_rows:0;
    require(!defer_hidden_readback||(direct&&reuse_graph&&!reuse_layer_graph&&!admitted_prefetch_bytes&&resident_hidden_tokens&&
        admitted_hidden_bytes>=resident_required(spill_bank?spill_bank:resident_hidden_tokens)),"Deferred hidden readback requires full admitted VRAM bank and direct allocator mode");
    require(backend&&admitted_graph_bytes>0,"V4.1 tile executor requires an admitted backend and scratch ceiling");
    require(!admitted_prefetch_bytes||(direct&&reuse_graph&&!reuse_layer_graph&&(!admitted_index_bytes||pair_ffn||fixed)),"Copy prefetch requires pinned native RAM and allocator-only mode1");
    require(!admitted_index_bytes||(admitted_index_bytes>=64ull*1024*1024&&reuse_graph&&!reuse_layer_graph),
        "Bounded GLM index reuse requires separate64MiB admission and allocator-only mode");index_cublas=admitted_index_bytes!=0;
    require(!index_f32_cublas_enabled()||index_cublas,"F32 indexer needs explicit workspace admission");
    require(finite_scalar(norm_eps)&&norm_eps>0&&finite_scalar(hc_eps)&&hc_eps>0&&sinkhorn_iters>0,
            "Invalid V4.1 tile normalization contract");
    require(finite_scalar(swiglu_limit)&&swiglu_limit>0&&finite_scalar(expert_scale)&&expert_scale>0,
            "Invalid V4.1 tile FFN contract");
    std::lock_guard<std::mutex>lock(cache.execution_mutex);
    require(!cache.execution_backend||cache.execution_backend==backend,"V4.1 shared cache is bound to another compute backend");
    cache.execution_backend=backend;
    if(direct){
#ifdef GGML_USE_CUDA
        require(ggml_backend_is_cuda(backend)&&direct->base&&direct->bytes&&
            direct->bytes<=SIZE_MAX&&direct->admitted_device_bytes>=64ull*1024*1024,"Direct pinned RAM requires separate64MiB device admission");
        pinned.reset(new pinned_storage);pinned->backend=backend;pinned->base=direct->base;pinned->bytes=direct->bytes;
        size_t before=0,total=0,after=0;pinned_storage::ok(cudaMemGetInfo(&before,&total),"Cannot inspect pre-registration GPU memory");
        if(fixed){
            require(fixed->ready&&fixed->ram&&(fixed->ram_registration||fixed->tile_registration||fixed->pageable_copies)&&fixed->ram->data()==direct->base&&direct->bytes<=fixed->ram->size(),"Borrowed RAM registration lost exact fixed owner/extent");
            if(fixed->tile_registration||fixed->pageable_copies){
                const bool full_resident=fixed->bf16&&fixed->ids&&fixed->metadata&&defer_hidden_readback&&!admitted_prefetch_bytes;
                const bool ram_lookahead=!fixed->bf16&&!fixed->ids&&!fixed->metadata&&!defer_hidden_readback&&admitted_prefetch_bytes;
                require(!fixed->ram_registration&&(full_resident||(fixed->tile_registration&&!fixed->pageable_copies&&ram_lookahead)),"Tile registration requires its explicit resident or RAM-lookahead plan");
                pinned->tile_registration=fixed->tile_registration;pinned->pageable_copies=fixed->pageable_copies;
                llama_dsv41_layer_log(fixed->pageable_copies?"DS4PERF_FIXED_RAM_PAGEABLE_VECTORS view_bytes=%llu whole_bank_pinned=0 existing_backend_API=1\n":"DS4PERF_FIXED_RAM_TILE_REGISTRATION view_bytes=%llu whole_bank_pinned=0 no_new_staging=1\n",(unsigned long long)direct->bytes);
            }else{
                unsigned flags=0;pinned_storage::ok(cudaHostGetFlags(&flags,direct->base),"Fixed carry RAM is no longer registered");
                llama_dsv41_layer_log("DS4PERF_FIXED_RAM_REGISTRATION_BORROWED view_bytes=%llu registered_bytes=%llu\n",(unsigned long long)direct->bytes,(unsigned long long)fixed->ram->size());
            }
        }else{pinned_storage::ok(cudaHostRegister(direct->base,size_t(direct->bytes),cudaHostRegisterDefault),"Existing RAM carry bank could not be pinned");pinned->registered=true;}
        pinned_storage::ok(cudaStreamCreateWithFlags(&pinned->copies,cudaStreamNonBlocking),"Pinned carry copy stream creation failed");
        pinned_storage::ok(cudaEventCreateWithFlags(&pinned->ready,cudaEventDisableTiming),"Pinned carry copy fence creation failed");
        pinned_storage::ok(cudaMalloc(reinterpret_cast<void**>(&pinned->flags_device),8),"GPU carry flag allocation failed");
        pinned_storage::ok(cudaMallocHost(reinterpret_cast<void**>(&pinned->flags_host),4),"Pinned carry flag host allocation failed");
        pinned_storage::ok(cudaMemsetAsync(pinned->flags_device,0,8,pinned->compute()),"GPU carry validation warmup initialization failed");
        llama_dsv41_validation warm;warm.residual=reinterpret_cast<float*>(pinned->flags_device+1);warm.residual_n=1;pinned->validate(warm);
        pinned_storage::ok(cudaMemGetInfo(&after,&total),"Cannot inspect post-registration GPU memory");
        const uint64_t delta=before>after?before-after:0;require(delta<=direct->admitted_device_bytes,"Pinned bank/validator global device delta exceeds explicit admission");
        llama_dsv41_layer_log("[dsv41-pinned-carry] bank_bytes=%llu observed_global_device_delta=%llu device_admission=%llu owned_delta_proof=0 direct_page_rows=32 validation_flag_bytes=4 inter_tile_overlap=0\n",
            (unsigned long long)direct->bytes,(unsigned long long)delta,(unsigned long long)direct->admitted_device_bytes);
#else
        throw std::runtime_error("Direct pinned carry requires a CUDA build");
#endif
    }
    require(!reuse_layer_graph||(reuse_graph&&!resident_hidden_tokens),"Cached layer graphs require persistent allocator and no hidden-bank rebinding");
    require(!admitted_cast_bytes||(reuse_graph&&!reuse_layer_graph),"Weight cast cache requires allocator-only mode1");
    require(!audit_casts||admitted_cast_bytes,"Cast audit requires explicit cache admission");
    require(!coalesce_dense||admitted_cast_bytes,"Dense coalescing requires the existing admitted cast cache");
    if(reuse_graph){
        workspace.reset(new graph_storage);workspace->backend=backend;workspace->cache_layouts=reuse_layer_graph;
#ifdef GGML_USE_CUDA
        workspace->pending_copies=pinned?pinned->copies:nullptr;
#endif
        if(fixed){
            require(fixed->ready&&fixed->backend==backend&&fixed->graph&&fixed->graph_bytes<=admitted_graph_bytes,"Fixed graph reservation/backend changed");
            workspace->reserved_owner=fixed->graph;
        }
        workspace->value=workspace->reserved_owner?workspace->reserved_owner.get():ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(workspace->value,"Persistent graph allocator creation failed");
        if(admitted_cast_bytes)workspace->casts.reset(new llama_dsv41_weight_cast_cache(backend,admitted_cast_bytes,audit_casts,coalesce_dense,dense_mxfp4));
    }
    if(resident_hidden_tokens){
#ifdef GGML_USE_CUDA
        require(ggml_backend_is_cuda(backend),"Resident hidden bank requires the admitted CUDA backend");
        const uint64_t minimum=resident_required(std::min(resident_hidden_tokens,admitted_tile_tokens));
        require(minimum<=admitted_hidden_bytes,"Resident hidden bank cannot fit even one admitted tile");
        const uint32_t bank_tokens=spill_bank?spill_bank:resident_required(resident_hidden_tokens)<=admitted_hidden_bytes?resident_hidden_tokens:
            uint32_t((admitted_hidden_bytes-65536)/81936/admitted_tile_tokens)*admitted_tile_tokens;
        resident.reset(new resident_storage);resident->backend=backend;resident->tokens=bank_tokens;resident->total_tokens=resident_hidden_tokens;
        if(spill_bank){
            require(pinned&&!pinned->prefetch_buffer,"Hidden-bank spill requires the pinned RAM carry and no copy prefetch");
            resident->spill_start=bank_tokens-admitted_tile_tokens;
            llama_dsv41_layer_log("LLAMADECODE_PREFILL_SPILL bank_rows=%u resident_rows=%u slot_rows=%u total_rows=%u spilled_rows=%u bytes_per_spilled_tile_layer=%llu storage=own_RAM_carry_records\n",
                bank_tokens,resident->spill_start,admitted_tile_tokens,resident_hidden_tokens,resident_hidden_tokens-resident->spill_start,
                (unsigned long long)(uint64_t(admitted_tile_tokens)*(40960+16)));
        }
        resident->layout(bank_tokens,admitted_tile_tokens,resident_bf16,resident_ids,resident_metadata);
        const auto buft=ggml_backend_get_default_buffer_type(backend);
        const uint64_t planned=ggml_backend_alloc_ctx_tensors_from_buft_size(resident->ctx.get(),buft);
        require(planned<=admitted_hidden_bytes,"Resident hidden physical layout exceeds admission");
        if(fixed){
            require(fixed->ready&&fixed->backend==backend&&fixed->hidden&&fixed->tokens>=resident_hidden_tokens&&fixed->tile>=admitted_tile_tokens&&
                fixed->bf16==resident_bf16&&fixed->ids==resident_ids&&fixed->metadata==resident_metadata&&planned<=fixed->hidden_bytes,
                "Fixed resident reservation geometry/backend changed");
            resident->reserved_owner=fixed->hidden;resident->buffer=resident->reserved_owner.get();
            auto allocation=ggml_tallocr_new(resident->buffer);
            for(auto*t=ggml_get_first_tensor(resident->ctx.get());t;t=ggml_get_next_tensor(resident->ctx.get(),t))ggml_tallocr_alloc(&allocation,t);
        }else resident->buffer=ggml_backend_alloc_ctx_tensors_from_buft(resident->ctx.get(),buft);
        require(resident->buffer&&ggml_backend_buffer_get_size(resident->buffer)<=admitted_hidden_bytes,
            "Resident hidden device allocation failed or exceeded admission");
        if(resident_metadata){
            std::vector<int32_t>raw_ids(uint64_t(resident->metadata_rows)*128),history(127),kinds(resident->metadata_rows,0);
            std::vector<float>mask(raw_ids.size(),0.f);std::vector<int8_t>text(resident->metadata_rows,1);
            for(uint32_t t=0;t<resident->metadata_rows;++t)for(uint32_t row=0;row<128;++row)raw_ids[uint64_t(t)*128+row]=t+row;
            for(uint32_t h=0;h<127;++h)history[h]=(1+h)%128;
            const auto raw=llama_dsv41_rope_frequencies(false),compressed=llama_dsv41_rope_frequencies(true);
            upload(resident->raw_ids,raw_ids);upload(resident->raw_mask,mask);upload(resident->history_ids,history);
            upload(resident->raw_freq,std::vector<float>(raw.begin(),raw.end()));upload(resident->compressed_freq,std::vector<float>(compressed.begin(),compressed.end()));
            upload(resident->zero_kinds,kinds);upload(resident->text_mask,text);ggml_backend_synchronize(backend);
            llama_dsv41_layer_log("DS4PERF_RESIDENT_METADATA_STORAGE rows=%u raw_bytes=%llu initialized_before_prompt=1 existing_hidden_buffer=1\n",resident->metadata_rows,(unsigned long long)(ggml_nbytes(resident->raw_ids)+ggml_nbytes(resident->raw_mask)));
        }
        if(resident_ids)llama_dsv41_layer_log("DS4PERF_RESIDENT_ID_STORAGE tokens=%u selected_capacity_bytes=%llu candidate_capacity_bytes=%llu existing_hidden_buffer=1 conservative_F32_admission=1\n",bank_tokens,(unsigned long long)ggml_nbytes(resident->selected),(unsigned long long)ggml_nbytes(resident->candidates));
        if(resident_bf16)llama_dsv41_layer_log("DS4PERF_RESIDENT_BF16_STORAGE tokens=%u hidden_bytes=%llu pre_F32_bytes=%llu initial_layer_F32=1 existing_cast_kernels=1\n",bank_tokens,(unsigned long long)ggml_nbytes(resident->hidden),(unsigned long long)ggml_nbytes(resident->pre));
        llama_dsv41_layer_log("[dsv41-resident-hidden] tokens=%u total_tokens=%u actual_bytes=%llu admitted_bytes=%llu readbacks=unchanged remainder=RAM\n",
            bank_tokens,resident_hidden_tokens,(unsigned long long)ggml_backend_buffer_get_size(resident->buffer),(unsigned long long)admitted_hidden_bytes);
#else
        throw std::runtime_error("Resident hidden bank requires a CUDA build");
#endif
    }else require(!admitted_hidden_bytes,"Hidden admission supplied without resident tokens");
    if(admitted_prefetch_bytes){
#ifdef GGML_USE_CUDA
        require(bool(pinned),"Copy prefetch requires pinned CUDA storage");pinned->prefetch_initialize(admitted_tile_tokens,admitted_prefetch_bytes);
#else
        throw std::runtime_error("Copy prefetch requires CUDA");
#endif
    }
}

bool llama_dsv41_tile_executor::resident_head_row(llama_dsv41_carry_store::tile &out,std::string &error){
    std::unique_lock<std::mutex>lock(cache.execution_mutex,std::try_to_lock);out={};error.clear();
    if(!lock.owns_lock()){error="Resident head is already computing";return false;}
    try{
        require(export_hidden_after_head&&resident&&pinned&&!cache.execution_poisoned&&resident->layer==40&&
            !resident->cursor&&resident->full(),"Resident head requires every validated device layer");
        // Release only layer workspace/cast caches, retaining the real hidden
        // bank and pinned owner. The head reuses the existing graph ceiling.
        workspace.reset();ggml_backend_synchronize(backend);
        out.start=resident->total_tokens-1;out.count=1;out.residual.resize(5120*4);out.previous_pre.resize(4);
        // A spilled last row is still in the slot from layer 39 (its tile ran last).
        const uint32_t tile_first=resident->spilled(out.start)?(resident->yoco_first?resident->slot_logical:resident->spill_start+(out.start-resident->spill_start)/admitted_tile_tokens*admitted_tile_tokens):out.start;
        require(!resident->spilled(out.start)||resident->slot_logical==tile_first,"Spilled last row left the staging slot before the head");
        const uint32_t row_at=resident->spilled(out.start)?resident->spill_start+(out.start-tile_first):out.start;
        if(resident_bf16){
            std::vector<ggml_bf16_t>row(5120*4);
            ggml_backend_tensor_get(resident->hidden,row.data(),uint64_t(row_at)*row.size()*sizeof(ggml_bf16_t),row.size()*sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(row.data(),out.residual.data(),int64_t(row.size()));
        }else ggml_backend_tensor_get(resident->hidden,out.residual.data(),uint64_t(row_at)*81920,81920);
        ggml_backend_tensor_get(resident->pre,out.previous_pre.data(),uint64_t(row_at)*16,16);
        finite(out.residual);finite(out.previous_pre);
        llama_dsv41_layer_log("DS4PERF_RESIDENT_HEAD_ROW start=%u count=1 F32_values=20484 validated=1 bulk_export_deferred=1\n",out.start);
        return true;
    }catch(const std::exception &e){error=e.what();return false;}
}

bool llama_dsv41_tile_executor::export_hidden_tile(const llama_dsv41_carry_store::ram_lease &lease,
        std::optional<llama_dsv41_carry_store::ram_completion> &completion,std::string &error){
    std::unique_lock<std::mutex>lock(cache.execution_mutex,std::try_to_lock);completion.reset();error.clear();
    if(!lock.owns_lock()){error="Final hidden export is already computing";return false;}
    try{
#ifdef GGML_USE_CUDA
        require(export_hidden_after_head&&resident&&pinned&&!cache.execution_poisoned&&resident->layer==40&&!resident->cursor&&
            resident->full()&&!lease.seeding&&lease.base==pinned->base&&lease.bytes==pinned->bytes&&
            lease.store_id&&lease.generation&&lease.first==resident->export_cursor&&lease.rows&&lease.rows<=admitted_tile_tokens&&
            lease.first<resident->total_tokens&&lease.rows<=resident->total_tokens-lease.first&&
            (resident->spilled(lease.first)?((lease.first-resident->spill_start)%admitted_tile_tokens==0||lease.first==resident->yoco_first):lease.rows<=resident->tokens-lease.first),
            "Final hidden export lost its full device bank or exact contiguous lease");
        (void)lease.row(0);(void)lease.row(lease.rows-1);
        pinned->bind_lease(lease);
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));require(bool(ctx),"Final hidden export metadata allocation failed");
        // This guard outlives every queued copy and precedes metadata teardown.
        struct export_scope {
            ggml_backend_t backend;pinned_storage *pinned;
            export_scope(ggml_backend_t b,pinned_storage *p):backend(b),pinned(p){ggml_backend_cuda_graphs_disable_push(backend);}
            ~export_scope(){pinned_storage::cleanup(cudaStreamSynchronize(pinned->copies));ggml_backend_synchronize(backend);ggml_backend_cuda_graphs_disable_pop(backend);}
        } scope(backend,pinned.get());
        auto *hidden=ggml_new_tensor_3d(ctx.get(),resident->hidden->type,5120,4,lease.rows);
        auto *pre=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4,lease.rows);
        const uint32_t at=resident->physical(lease.first);
        require(ggml_backend_tensor_alloc(resident->buffer,hidden,(char*)resident->hidden->data+
                uint64_t(at)*5120*4*ggml_type_size(resident->hidden->type))==GGML_STATUS_SUCCESS&&
            ggml_backend_tensor_alloc(resident->buffer,pre,(char*)resident->pre->data+uint64_t(at)*16)==GGML_STATUS_SUCCESS,
            "Final hidden export source view escaped its resident allocation");
        cache.execution_poisoned=true;
        if(resident->spilled(lease.first)){
            // Bring the spilled rows' exact layer-39 BF16/pre bytes back into the slot;
            // the unchanged cast/validate/F32 export below then overwrites the record.
            require(resident_bf16,"Spilled export requires the BF16 bank");
            pinned->queue(true,lease,0,40960,hidden);pinned->queue(true,lease,81920,16,pre);pinned->upload_ready();
            resident->slot_logical=lease.first;resident->spill_h2d_bytes+=uint64_t(lease.rows)*40976;
        }
        if(resident_bf16){
            hidden=ggml_cast(ctx.get(),hidden,GGML_TYPE_F32);ggml_set_output(hidden);
            auto *graph=ggml_new_graph_custom(ctx.get(),32,false);ggml_build_forward_expand(graph,hidden);backend_buffers(backend,graph);
            if(!workspace){workspace.reset(new graph_storage);workspace->backend=backend;
                if(fixed)workspace->reserved_owner=fixed->graph;
                workspace->value=workspace->reserved_owner?workspace->reserved_owner.get():ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(workspace->value,"Final hidden export allocator failed");}
            (void)llama_dsv41_allocate_native_graph(workspace->value,graph,0,admitted_graph_bytes);
            require(ggml_backend_graph_compute_async(backend,graph)==GGML_STATUS_SUCCESS,"Final hidden export BF16 conversion failed");
        }
        llama_dsv41_validation check;check.residual=static_cast<const float*>(hidden->data);check.residual_n=uint64_t(5120)*4*lease.rows;
        check.pre=static_cast<const float*>(pre->data);check.pre_n=uint64_t(4)*lease.rows;
        pinned->validate(check);pinned->readback_ready();
        pinned->queue(false,lease,0,81920,hidden);pinned->queue(false,lease,81920,16,pre);pinned->finish();
        resident->export_cursor+=lease.rows;completion=llama_dsv41_carry_store::ram_completion(lease,true,true,true);
        cache.execution_poisoned=false;
        llama_dsv41_layer_log("DS4PERF_EXPORT_HIDDEN_COMPLETE start=%u count=%u copied_host_bytes=%llu after_required_head=1 validated=1\n",
            lease.first,lease.rows,(unsigned long long)(uint64_t(lease.rows)*81936));
        return true;
#else
        (void)lease;throw std::runtime_error("Final hidden export requires the admitted CUDA bank");
#endif
    }catch(const std::exception &e){error=e.what();return false;}
}


void llama_dsv41_tile_executor::clear(){
#ifdef GGML_USE_CUDA
    if(pinned){pinned->finish();pinned->prefetch_valid=false;}
#endif
    cache.clear();
    if(workspace){workspace->variants.clear();workspace->layer=UINT32_MAX;}
    if(resident){resident->layer=0;resident->cursor=0;resident->export_cursor=0;resident->selected_source=UINT32_MAX;resident->candidate_source=UINT32_MAX;resident->selected_width=0;resident->candidate_width=0;}
}

bool llama_dsv41_tile_executor::run(const request&in,const llama_dsv41_layer_graph::weights&w,
                                  result&out,std::string&error){
    return run(in,w,out,error,{});
}
bool llama_dsv41_tile_executor::run(const request&in,const llama_dsv41_layer_graph::weights&w,
        result&out,std::string&error,const llama_dsv41_expert_profile::policy & requested_expert){
    std::unique_lock<std::mutex>lock(cache.execution_mutex,std::try_to_lock);
    out={};error.clear();
    if(!lock.owns_lock()){error="V4.1 tile executor is already computing";return false;}
    if(cache.execution_poisoned){error="V4.1 tile cache requires clear after an incomplete computation";return false;}
    try{
        auto expert=requested_expert;
        expert.generic_weights=w.ffn.gate->type!=GGML_TYPE_Q4_K||w.ffn.up->type!=GGML_TYPE_Q4_K||w.ffn.down->type!=GGML_TYPE_Q4_K;
        if(w.attention.q_a->type!=GGML_TYPE_Q8_0||w.attention.q_b->type!=GGML_TYPE_Q8_0||w.attention.kv->type!=GGML_TYPE_Q8_0||
           w.attention.out_a->type!=GGML_TYPE_Q8_0||w.attention.out_b->type!=GGML_TYPE_Q8_0||
           w.ffn.shared_gate->type!=GGML_TYPE_Q8_0||w.ffn.shared_up->type!=GGML_TYPE_Q8_0||w.ffn.shared_down->type!=GGML_TYPE_Q8_0)
            expert.dense_inputs_f16=false;
        collect_layer=in.layer;collect_rows=in.count;
        const auto prepare_started=clock_type::now();
        const bool merge_rows=merge_attention_selected_rows&&!in.yoco_raw_begin;
        const bool capture_dspark=cache.dspark_ring&&in.layer>=37;
        require(!in.yoco_first || (in.layer>20&&in.start>=in.yoco_first&&in.count<=140),"Invalid YOCO decoder coverage");
        llama_dsv41_dspark_capture::completion::ticket capture_ticket;
        require(!capture_dspark||!workspace||!workspace->cache_layouts,"DSpark ring writes require fresh graph bindings");

        struct wall_probe {
            const request &in;result &out;clock_type::time_point started;
            bool enabled=false,complete=false;double pre_return_ms=0,prepare_flush_ms=0,post_checks_ms=0,boundary_ms=0;
            ~wall_probe(){if(enabled&&complete){const double wall=std::chrono::duration<double,std::milli>(clock_type::now()-started).count();
                const double phases=out.prepare_ms+out.upload_ms+out.compute_ms+out.readback_ms+out.weight_cache_ms+out.cleanup_ms;
                llama_dsv41_layer_log("DS4PERF_EXECUTOR_WALL_PROBE layer=%u start=%u count=%u wall_ms=%.6f pre_return_ms=%.6f teardown_ms=%.6f prepare_flush_ms=%.6f post_checks_ms=%.6f boundary_ms=%.6f unexplained_ms=%.6f\n",in.layer,in.start,in.count,wall,pre_return_ms,wall-pre_return_ms,prepare_flush_ms,post_checks_ms,boundary_ms,pre_return_ms-phases-prepare_flush_ms-post_checks_ms-boundary_ms);}}
        } probe{in,out,prepare_started};
        probe.enabled=std::getenv("DS4PERF_EXECUTOR_WALL_PROBE")&&std::string(std::getenv("DS4PERF_EXECUTOR_WALL_PROBE"))=="1";
        const bool rope_inplace=std::getenv("LLAMA_DSV41_ROPE_INPLACE")&&std::string(std::getenv("LLAMA_DSV41_ROPE_INPLACE"))=="1";
        const bool dense_bf16_store=expert.dense_activation_q8_1&&std::getenv("LLAMA_DSV41_DENSE_BF16_STORE")&&std::string(std::getenv("LLAMA_DSV41_DENSE_BF16_STORE"))=="1";
        const bool mul_f16_store=std::getenv("LLAMA_DSV41_MUL_F16_STORE")&&std::string(std::getenv("LLAMA_DSV41_MUL_F16_STORE"))=="1";
        unsigned mul_f16_fused=0;
        require(!mul_f16_store||!workspace||!workspace->cache_layouts,"F16 route-weight store fusion requires allocator-only graph mode");
        llama_dsv41_expert_profile::validate(expert,in.count,std::getenv("DSV41_BOUNDARY_CAPTURE"));
        require(in.position_offset<=cache.capacity()&&in.start<=cache.capacity()-in.position_offset,"Absolute tile offset exceeds cache capacity");
        const uint32_t absolute_start=in.position_offset+in.start;
        require(!in.position_offset||!workspace||!workspace->cache_layouts,"Offset chunks require allocator-only graph mode");
        require(!in.single_attention_append||(pair_ffn&&in.position_offset%4096&&in.start%2048==0&&in.count<=2048&&admitted_tile_tokens>=4096&&workspace&&!workspace->cache_layouts),
            "Unaligned paired continuation requires explicit bounded single-attention tiles");
        require(!pair_ffn||absolute_start%4096==0||in.single_attention_append||in.yoco_raw_begin,"Paired FFN requires original aligned absolute attention tiles");
        const bool native_pair=pair_ffn&&llama_dsv41_expert_profile::native_attention(expert);
        const uint32_t attention_rows=native_pair?2048u:4096u;
        require(!pair_ffn||(native_pair?(admitted_tile_tokens==4096&&in.count<=4096):
            (llama_dsv41_expert_profile::activation_rounding_q8_1(expert)&&!expert.expert_f16_cublas&&in.count<=16384)),"Paired FFN requires bounded native4K or existing shared Q8_1 experts");
        require(!pair_ffn||native_pair||(dense_mxfp4||expert.dense_activation_q8_1),"Q8 paired FFN requires explicit original-Q8 MMQ or hyper-fast FP4 dense selection");
        require(pair_ffn||in.count<=4096,"Unpaired graph exceeds the original attention bound");
        const uint32_t attention_tiles=pair_ffn?(in.count+attention_rows-1)/attention_rows:1;
        const bool direct_input=bool(in.direct);
        require(direct_input==bool(pinned),"Pinned carry request/configuration mode changed");
        bool skip_tile_pin=false;
        double tile_bind_ms=0,tile_register_ms=0,tile_launch_ms=0,tile_sync_ms=0;float tile_gpu_ms=-1;
#ifdef GGML_USE_CUDA
        struct bank_copy_scope {pinned_storage*p;~bank_copy_scope(){if(p)p->current_bank_copies_forbidden=false;}} bank_scope{pinned.get()};
#endif
        if(direct_input){
            const auto &d=*in.direct;
            require(llama_dsv41_expert_profile::native(expert)&&d.base==pinned->base&&d.bytes==pinned->bytes&&
                d.first==in.start&&d.rows==in.count&&d.store_id&&d.generation,
                "Direct pinned carry requires exact native-expert resident-bank lease");
            (void)d.row(0);(void)d.row(in.count-1);
#ifdef GGML_USE_CUDA
            skip_tile_pin=std::getenv("LLAMA_DSV41_SKIP_UNUSED_TILE_PIN")&&std::string(std::getenv("LLAMA_DSV41_SKIP_UNUSED_TILE_PIN"))=="1"&&
                pinned->tile_registration&&!pinned->prefetch_buffer&&!in.lookahead&&resident&&resident_bf16&&resident_ids&&resident_metadata&&
                resident->full()&&!resident->spilled(in.start)&&defer_hidden_readback&&export_hidden_after_head&&
                in.layer>0&&in.layer<40&&!query_source(in.layer)&&in.layer!=20;
            pinned->current_bank_copies_forbidden=skip_tile_pin;
            pinned->last_register_ms=0;const auto bind_started=clock_type::now();
            if(!skip_tile_pin)pinned->bind_lease(d,in.lookahead?&*in.lookahead:nullptr);
            tile_bind_ms=elapsed(bind_started);tile_register_ms=pinned->last_register_ms;
#endif
        }
        require(in.layer<40,"V4.1 tile layer is outside 0..39");
#ifdef GGML_USE_CUDA
        require(!in.lookahead||(pinned&&pinned->prefetch_buffer&&direct_input),"RAM lookahead supplied without admitted prefetch storage");
        if(in.lookahead){const auto&next=in.lookahead->lease;const auto&current=*in.direct;
            require(next.base==current.base&&next.bytes==current.bytes&&next.store_id==current.store_id&&
                next.generation==current.generation&&next.first==uint64_t(in.start)+in.count&&next.rows&&next.rows<=admitted_tile_tokens&&
                next.sw==current.sw&&next.cw==current.cw&&next.next_sw==current.next_sw&&next.next_cw==current.next_cw,
                "RAM lookahead differs from its exact live parent");(void)in.lookahead->row(0);(void)in.lookahead->row(next.rows-1);}
        if(pinned&&pinned->prefetch_buffer&&in.start==0)require(!pinned->prefetch_valid,"Stale prefetch crossed a layer boundary");
#else
        require(!in.lookahead,"RAM lookahead requires CUDA prefetch");
#endif
        if(resident&&in.yoco_first&&in.layer==resident->layer&&resident->cursor==0){
            require(in.layer>=21&&in.start==in.yoco_first&&in.yoco_first<resident->total_tokens,"YOCO resident layer coverage changed");
            resident->cursor=in.yoco_first;resident->yoco_first=in.yoco_first;
        }
        if(resident)require(in.layer==resident->layer&&in.start==resident->cursor&&
            in.start<resident->total_tokens&&in.count<=resident->total_tokens-in.start,
            "Resident hidden layer/tile lifetime changed");
        // LLAMADECODE single sweep: a spilled tile runs in the bank's last tile (the slot).
        const bool spilled=resident&&resident->spilled(in.start);
        const uint32_t bank_start=resident?resident->physical(in.start):in.start;
        const bool device_hidden=resident&&(in.start<resident->tokens||spilled);
        require(!defer_hidden_readback||(device_hidden&&resident->full()),"Deferred hidden state lost full VRAM residency");
        if(device_hidden)require(spilled?in.count<=admitted_tile_tokens:in.count<=(resident->spill()?resident->spill_start:resident->tokens)-in.start,
            "Resident hidden tile crosses device/RAM split");
        require(!spilled||(direct_input&&!in.lookahead),"Spilled tile requires its direct RAM lease and no lookahead");
        const uint32_t ratio=in.layer<2?0:in.layer<20?2:1;
        const bool zero_kinds=resident_metadata&&std::all_of(in.token_kinds.begin(),in.token_kinds.end(),[](int32_t k){return k==0;});
        const bool all_text=resident_metadata&&engram_layer(in.layer)&&std::all_of(in.text_mask.begin(),in.text_mask.end(),[](int8_t k){return k==1;});
        const bool raw_template=!in.yoco_raw_begin&&resident_metadata&&absolute_start>=128&&absolute_start%128==0;
        require(!resident_metadata||(resident&&in.count<=resident->metadata_rows),"Resident metadata tile exceeds initialized capacity");
        llama_dsv41_tile_plan p(absolute_start,in.count,cache.capacity(),ratio,admitted_tile_tokens,raw_template);
        if(in.yoco_raw_begin){
            require(in.layer>20&&in.yoco_raw_begin<=absolute_start,"YOCO raw coverage is invalid");
            // Build a chronological raw table containing only valid replay
            // rows. The token-tile kernel uses relative raw positions; RoPE,
            // publication and compressed visibility remain absolute.
            p.restrict_raw(in.yoco_raw_begin,admitted_tile_tokens);
        }
        weights_present(w,in.layer);
        require(direct_input?(in.residual.empty()&&in.previous_pre.empty()):
                (in.residual.size()==size_t(5120)*4*in.count&&in.previous_pre.size()==size_t(4)*in.count),
                "V4.1 tile hidden/pre carry does not contain exactly its token rows");
        require(in.token_kinds.size()==in.count&&std::all_of(in.token_kinds.begin(),in.token_kinds.end(),[](int32_t v){return v==0||v==1;}),
                "V4.1 tile requires one text/image routing kind per token");
        const uint32_t maximum_rows=ratio?(cache.capacity()+ratio-1)/ratio:0;
        require(in.compressed_rows<=maximum_rows&&(!ratio?in.compressed_rows==0:in.compressed_rows>=uint32_t(p.compressed_visible.back())),
                "V4.1 tile compressed frontier differs from its admitted cache/causal prefix");
        const uint32_t selected_width=std::min(512u,in.compressed_rows);
        const uint32_t candidate_width=std::min(2048u,(in.compressed_rows+7)/8);
        // Source20's candidates cross reader layers21-23 unchanged before
        // they constrain source24. Carry them even while their mask is unused.
        const bool reuse=in.compressed_rows&&!query_source(in.layer),candidates=in.layer>20&&in.compressed_rows;
        if(resident_ids){
            require(resident&&resident->full()&&resident->selected&&resident->candidates,"Resident ID bank lost full request storage");
            uint32_t previous_source=UINT32_MAX;
            for(uint32_t l=0;l<in.layer;++l)if(query_source(l))previous_source=l;
            require(!reuse||(resident->selected_source==previous_source&&resident->selected_width==selected_width),"Resident selected IDs missing exact completed source layer/width");
            require(!candidates||(resident->candidate_source==20&&resident->candidate_width==candidate_width),"Resident candidates missing completed source20/width");
        }
        require(in.selected.size()==(!direct_input&&reuse?size_t(selected_width)*in.count:0),"V4.1 tile selected-row carry dimensions differ");
        require(in.candidates.size()==(!direct_input&&candidates?size_t(candidate_width)*in.count:0),"V4.1 tile candidate-block carry dimensions differ");
        if(direct_input)require((!reuse||in.direct->input_selected()==selected_width)&&
            (!candidates||in.direct->input_candidates()==candidate_width)&&in.direct->output_selected()==selected_width&&
            in.direct->output_candidates()==(in.layer>=20?candidate_width:0),"Direct RAM compact-width lease changed");
        for(int32_t id:in.selected)require(id>=-1&&id<int64_t(in.compressed_rows),"Invalid V4.1 carried selected-row ID");
        for(int32_t id:in.candidates)require(id>=0&&id<int64_t((in.compressed_rows+7)/8),"Invalid V4.1 carried candidate-block ID");
        const bool engram=engram_layer(in.layer);
        const int64_t row_width=engram?w.engram_wkv->ne[0]:0;
        require(!engram||(row_width>0&&row_width<=6144),"V4.1 gathered Engram row width is invalid");
        require(in.engram_rows.size()==size_t(row_width)*in.count,"V4.1 tile gathered Engram rows differ from staged WKV input");
        require(in.text_mask.size()==(engram?in.count:0)&&std::all_of(in.text_mask.begin(),in.text_mask.end(),[](int8_t v){return v==0||v==1;}),
                "V4.1 tile requires an explicit Engram text mask");
        finite(in.residual);finite(in.previous_pre);finite(in.engram_rows);
        dsv41_boundary::capture boundary(in.layer,in.start,in.count,in.residual,in.previous_pre);
        if(boundary.active())require(std::all_of(in.token_kinds.begin(),in.token_kinds.end(),
            [](int32_t value){return value==0;}),"Boundary capture requires original text rows");
        dsv41_boundary::scope boundary_scope(&boundary);

        const bool pooled=in.layer==2||in.layer==8||in.layer==14;
        graph_storage::variant transient;
        std::unique_ptr<graph_storage::variant> fresh;
        graph_storage::variant *compiled=&transient;
        bool graph_cache_hit=false;
        int routed_split=-1; // LLAMADECODE B1: first node after the router cut
        ggml_tensor*expert_slot_input=nullptr; // LLAMADECODE B2
        ggml_tensor*hash_experts=nullptr; // LLAMADECODE_HASH_DUMP only
        if(workspace&&workspace->cache_layouts){
            require(!boundary.active()&&!resident,"Cached layer graphs refuse capture and hidden-bank rebinding");
            if(workspace->layer!=in.layer){
                workspace->variants.clear();workspace->layer=in.layer;
                std::memcpy(workspace->weight_identity.data(),&w,sizeof(w));
            }else require(std::memcmp(workspace->weight_identity.data(),&w,sizeof(w))==0,"Staged weight identity changed within layer");
            const std::array<uint32_t,13> key{{in.count,p.context,p.ratio,p.history_rows,p.raw_rows,p.published_count,
                uint32_t(p.previous_pool_tail),uint32_t(p.next_pool_tail),in.compressed_rows,uint32_t(expert.profile),uint32_t(row_width),uint32_t(reuse),
                uint32_t(expert.dense_inputs_f16)|(uint32_t(expert.index_bf16)<<1)|(uint32_t(expert.token_tile_attention)<<2)|(uint32_t(expert.expert_f16_cublas)<<3)|(uint32_t(expert.expert_mxfp4)<<4)|(uint32_t(expert.dense_activation_q8_1)<<5)}};
            for(auto &v:workspace->variants)if(v->key==key){compiled=v.get();graph_cache_hit=true;break;}
            if(!graph_cache_hit){
                require(workspace->variants.size()<8,"Layer graph variant metadata limit exceeded before allocation");
                fresh.reset(new graph_storage::variant);compiled=fresh.get();compiled->key=key;
            }
        }
        auto &ctx=compiled->ctx;auto &input=compiled->input;auto &state=compiled->state;
        auto &values=compiled->values;auto *&graph=compiled->graph;
        auto &backend_temporary=compiled->backend_temporary;
        auto &rounded_matrices=compiled->rounded_matrices;auto &split_matrices=compiled->split_matrices;
        ggml_tensor *spill_slot=nullptr;   // LLAMADECODE single sweep: the slot's BF16 view (spilled tiles)
        if(graph_cache_hit)compiled->bindings.restore(in.start,in.count);
        else {
        const bool long_index=expert.index_key_windows&&query_source(in.layer)&&in.compressed_rows>131072;
        ctx.reset(ggml_init({size_t(long_index?16:8)*1024*1024,nullptr,true}));require(bool(ctx),"V4.1 tile metadata allocation failed");
        std::vector<ggml_tensor*>uploaded_inputs;
        auto input_tensor=[&](ggml_tensor*t){ggml_set_input(t);uploaded_inputs.push_back(t);return t;};
        auto f32=[&](int64_t a,int64_t b=1,int64_t c=1){return input_tensor(ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,a,b,c));};
        auto ids=[&](ggml_type type,int64_t a,int64_t b=1){return input_tensor(ggml_new_tensor_2d(ctx.get(),type,a,b));};
        // Later BF16-resident layers bind their actual F32 cast below. Do not
        // keep an unused original F32 leaf alive in uploaded_inputs as well.
        input.residual=(device_hidden&&resident_bf16&&in.layer)?nullptr:f32(5120,4,in.count);
        input.previous_pre=f32(4,in.count);input.positions=ids(GGML_TYPE_I32,in.count);
        ggml_tensor *bf16_destination=nullptr;
        if(device_hidden){
            if(resident_bf16){
                bf16_destination=input_tensor(ggml_new_tensor_3d(ctx.get(),GGML_TYPE_BF16,5120,4,in.count));
                require(ggml_backend_tensor_alloc(resident->buffer,bf16_destination,
                    (char*)resident->hidden->data+uint64_t(bank_start)*5120*4*sizeof(ggml_bf16_t))==GGML_STATUS_SUCCESS,
                    "Resident BF16 hidden tile binding failed");
                // Embeddings are not assumed BF16. Layer0 uses the ordinary
                // F32 graph input, uploaded unchanged from the seed RAM bank.
                if(in.layer)input.residual=ggml_cast(ctx.get(),bf16_destination,GGML_TYPE_F32);
                if(spilled)spill_slot=bf16_destination;
            }else require(ggml_backend_tensor_alloc(resident->buffer,input.residual,
                (char*)resident->hidden->data+uint64_t(bank_start)*5120*4*sizeof(float))==GGML_STATUS_SUCCESS,
                "Resident F32 hidden tile binding failed");
            require(ggml_backend_tensor_alloc(resident->buffer,input.previous_pre,
                (char*)resident->pre->data+uint64_t(bank_start)*4*sizeof(float))==GGML_STATUS_SUCCESS,
                "Resident previous-pre tile binding failed");
        }
        input.raw_frequencies=f32(32);input.compressed_frequencies=f32(32);
        input.history_ids=p.history_rows?ids(GGML_TYPE_I32,p.history_rows):nullptr;
        input.pair_history_ids=pair_ffn&&in.count>attention_rows?ids(GGML_TYPE_I32,127):nullptr;
        input.raw_ids=ids(GGML_TYPE_I32,p.raw_rows,in.count);input.raw_mask=f32(p.raw_rows,in.count);
        input.visible=in.compressed_rows?ids(GGML_TYPE_I32,in.count):nullptr;
        input.publication_ids=p.published_count?ids(GGML_TYPE_I64,p.published_count):nullptr;
        input.publication_positions=p.published_count?ids(GGML_TYPE_I32,p.published_count):nullptr;
        input.previous_selected=reuse?ids(GGML_TYPE_I32,selected_width,in.count):nullptr;
        input.candidate_ids=candidates?ids(GGML_TYPE_I32,candidate_width,in.count):nullptr;
        auto id_slice=[&](ggml_tensor *tensor,ggml_tensor *bank,uint32_t width){
            require(tensor&&bank&&width&&width<=uint64_t(bank->ne[0]),"Resident ID tensor geometry changed");
            const uint64_t offset=uint64_t(bank_start)*width*sizeof(int32_t),bytes=uint64_t(in.count)*width*sizeof(int32_t);
            require(offset<=ggml_nbytes(bank)&&bytes<=ggml_nbytes(bank)-offset&&ggml_nbytes(tensor)==bytes,
                "Resident ID logical-width slice exceeds admitted capacity");
            require(ggml_backend_tensor_alloc(resident->buffer,tensor,(char*)bank->data+offset)==GGML_STATUS_SUCCESS,"Resident ID slice allocation failed");
            return tensor;
        };
        ggml_tensor *selected_destination=nullptr,*candidate_destination=nullptr;
        if(resident_ids){
            if(reuse)id_slice(input.previous_selected,resident->selected,selected_width);
            if(candidates)id_slice(input.candidate_ids,resident->candidates,candidate_width);
            if(query_source(in.layer)&&selected_width)selected_destination=id_slice(ids(GGML_TYPE_I32,selected_width,in.count),resident->selected,selected_width);
            if(in.layer==20&&candidate_width)candidate_destination=id_slice(ids(GGML_TYPE_I32,candidate_width,in.count),resident->candidates,candidate_width);
        }
        input.engram_rows=engram?f32(row_width,in.count):nullptr;input.text_mask=engram?ids(GGML_TYPE_I8,in.count):nullptr;
        input.token_kinds=ids(GGML_TYPE_I32,in.count);
        if(resident_metadata){
            auto bind=[&](ggml_tensor*t,ggml_tensor*bank){require(t&&bank&&t->type==bank->type&&ggml_nbytes(t)<=ggml_nbytes(bank),"Resident metadata type/capacity changed");
                require(ggml_backend_tensor_alloc(resident->buffer,t,bank->data)==GGML_STATUS_SUCCESS,"Resident metadata binding failed");};
            bind(input.raw_frequencies,resident->raw_freq);bind(input.compressed_frequencies,resident->compressed_freq);
            if(input.pair_history_ids)bind(input.pair_history_ids,resident->history_ids);
            if(raw_template){bind(input.raw_ids,resident->raw_ids);bind(input.raw_mask,resident->raw_mask);bind(input.history_ids,resident->history_ids);}
            if(zero_kinds)bind(input.token_kinds,resident->zero_kinds);
            if(all_text)bind(input.text_mask,resident->text_mask);
        }
        state={cache.raw(in.layer),ratio?cache.compressed(in.layer):nullptr,
            ratio?cache.index(in.layer):nullptr,nullptr,nullptr};
        if(pooled){state.previous_pool_kv=cache.pool_kv(in.layer);state.previous_pool_score=cache.pool_score(in.layer);}
        graph=ggml_new_graph_custom(ctx.get(),long_index?16384:8192,false);
        llama_dsv41_graph_ops::observer boundary_observer;
        if(boundary.active())boundary_observer=[&](ggml_tensor*t,const char*name){
            ggml_set_name(t,name);boundary.observe(t,name);
        };
        boundary.add(input.residual,"common_residual");
        boundary.add(input.previous_pre,"previous_pre");
        if(boundary.active()){
            require(in.layer==0&&in.start==0&&in.count==32&&p.ratio==0&&p.history_rows==0&&p.raw_rows==32&&in.compressed_rows==0&&in.selected.empty(),"attention profile requires cold raw-only32 rows");
            boundary.verify_attention_contract(p.raw_ids,p.raw_mask);
            boundary.add(input.raw_ids,"raw_ids");boundary.add(input.raw_mask,"raw_mask_f32");
            boundary.add(w.attention.attn_sinks,"attention_sinks_f32");
        }

        llama_dsv41_graph_ops ops(ctx.get(),norm_eps,hc_eps,sinkhorn_iters,boundary_observer);
        ops.merge_attention_selected_rows(merge_rows,ratio);
        ops.use_hc_projection_f16_gemm(hc_projection_f16_gemm);ops.use_hc_bf16_epilogue(expert.hc_bf16_epilogue);
        require(!merge_attention_selected_rows||expert.token_tile_attention,"Selected-row union requires explicit token-tile attention");
        llama_dsv41_layer_graph factory(ctx.get(),graph,ops,swiglu_limit,expert_scale);
        auto routed_weights=w;expert_slot_input=nullptr;
        if(in.route_hook){expert_slot_input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,6,in.count);ggml_set_name(expert_slot_input,"dsv41_expert_slots");
            ggml_set_input(expert_slot_input);routed_weights.ffn.expert_slots=expert_slot_input;}
        values=pair_ffn?factory.build_pair(in.layer,p,in.compressed_rows,routed_weights,input,state,expert):factory.build(in.layer,p,in.compressed_rows,routed_weights,input,state,expert);
        if(capture_dspark){
            capture_ticket=in.yoco_raw_begin?cache.dspark_completion.begin_replay(in.layer-37,in.position_offset+in.start,in.count):cache.dspark_completion.begin(in.layer-37,in.position_offset+in.start,in.count);
            llama_dsv41_dspark_capture::enqueue(ctx.get(),graph,cache.dspark_ring,values.dspark_mean,
                capture_ticket.slot,capture_ticket.start,capture_ticket.count);
        }

        // Existing same-device ggml CPY publishes exact IDs after graph consumers.
        // Reader generations become valid only after every tile is validated,
        // mirrored to RAM and synchronized at the full layer boundary below.
        if(selected_destination)ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),values.selected,selected_destination));
        if(candidate_destination)ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),values.candidates,candidate_destination));
        if(bf16_destination){
            require(bf16_rounded_output(values.residual),
                "BF16 storage requires an existing explicit BF16-rounded layer output");
            // Existing CUDA CPY converts the already-rounded F32 values. Store
            // only after all graph consumers; keep() pins the F32 result for
            // the unchanged validation and export-copy checks below.
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),values.residual,bf16_destination));
        }
        if(expert.expert_f16_cublas){
            unsigned tagged=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*node=ggml_graph_node(graph,i);
                if(node->op==GGML_OP_MUL_MAT_ID){require(ggml_dsv41_expert_cublas_tag(node),"F16 expert cuBLAS shape admission refused");++tagged;}}
            require(tagged==3,"F16 expert cuBLAS requires exactly three expert products");
        }
        if(boundary.active()){
            // Profile selection and source proof are trusted owner config only.
            boundary.verify_weight("attention_sinks",w.attention.attn_sinks);
            boundary.verify_weight_proof_complete();
        }

        // Output copies participate in the actual gallocr live-byte preflight;
        // no diagnostic allocation can bypass the existing graph ceiling.
        boundary.retain(ctx.get(),graph);
        unsigned original_q8_products=0;
        if(expert.dense_activation_q8_1&&in.count>8){
            require(!dense_mxfp4,"Original Q8 MMQ excludes weight requantization");
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto *node=ggml_graph_node(graph,i);
                if(dsv41_original_q8_dense(node)){node->op_params[0]=GGML_PREC_DEFAULT;node->op_params[6]=0x51384131;++original_q8_products;}}
            // Other admitted GGUFs may contain only Q4 dense projections.
            // The Q8 activation option applies to Q8 weights when present.
        }
        graph=llama_dsv41_native_batch_graph(ctx.get(),graph,in.count,96ull*1024*1024,
            &backend_temporary,&rounded_matrices,&split_matrices);
        if(expert.dense_activation_q8_1&&in.count>8){
            unsigned actual=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto *node=ggml_graph_node(graph,i);
                if(node->op==GGML_OP_MUL_MAT&&node->op_params[6]==0x51384131){require(dsv41_original_q8_dense(node),"Original Q8 projection changed weight owner/type");++actual;}}
            require(actual==original_q8_products,"Original Q8 projection graph coverage changed");
            llama_dsv41_layer_log("DS4PERF_ORIGINAL_Q8_DENSE_GRAPH layer=%u start=%u count=%u products=%u original_Q8_weights=1 activation_Q8_1=1 HC_router_head_excluded=1\n",in.layer,in.start,in.count,actual);
        }
        if(expert.dense_inputs_f16){
            unsigned accepted=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto *node=ggml_graph_node(graph,i);
                if(ggml_dsv41_dense_half_tag(node)){
                    require(node->src[0]->type==GGML_TYPE_F16&&node->op_params[6]!=0x51384131,"F16 activation product overlaps an original-Q8 MMQ tag");++accepted;
                }}
            const char*refused=llama_dsv41_dense_half_admission(in.count,accepted);require(!refused,refused);
            if(int64_t(in.count)<=ggml_dsv41_dense_half_max_untagged_rows)llama_dsv41_layer_log("DS4PERF_DENSE_HALF_SMALL_ROWS layer=%u start=%u count=%u tagged=0 original_F32_products=1\n",in.layer,in.start,in.count);
        }
        if(index_cublas){require(llama_dsv41_expert_profile::native(expert),"GLM index TEST requires preserved native expert products");
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*node=ggml_graph_node(graph,i);
                  if(node->op==GGML_OP_DSV41_INDEX_SCORE){
                      uint32_t score_start=p.start;
                      if(native_pair){
                          uint32_t offset=0;
                          require(llama_dsv41_index_query_offset(node,input.visible,in.count,attention_rows,offset),
                              "Native pair index visibility lost its query owner, partition or exact key-window expression");
                          score_start+=offset;
                      }
                      require(index_f32_cublas_enabled()?ggml_dsv41_index_f32_tag(node):(expert.index_bf16?ggml_dsv41_index_bf16_tag(node):ggml_dsv41_index_cublas_tag(node)),"Index score exceeds explicit bounded GLM shape admission");
                      if(expert.index_bf16)require(ggml_dsv41_index_bf16_causal_tag(node,score_start,ratio),"Index causal slab proof exceeds contiguous position contract");
                       if(index_f32_cublas_enabled())require(ggml_dsv41_index_f32_causal_tag(node,score_start,ratio),"F32 index causal slab proof exceeds contiguous position contract");
                  }}}
        if(workspace&&workspace->casts)graph=workspace->casts->prepare(ctx.get(),graph,in.layer,&backend_temporary);
        if(expert.expert_mxfp4){
            require(std::getenv("GGML_CUDA_DISABLE_GRAPHS")&&std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))=="1","Expert FP4 packing requires the explicit uncaptured producer path");
            require(dense_mxfp4&&workspace&&workspace->casts&&!workspace->cache_layouts&&!pair_ffn,"Expert FP4 requires the existing dense FP4 cast bank, allocator-only graph and<=4K lane");
            if(in.count>4)workspace->casts->prepare_expert_fp4(graph,in.layer,in.count);
            else llama_dsv41_layer_log("DS4PERF_EXPERT_MXFP4_SMALL_BATCH layer=%u count=%u unchanged_original_weights=1 native_MMA_minimum_rows=5\n",in.layer,in.count);
        }
        require(!expert.reuse_expert_routing_map||(std::getenv("GGML_CUDA_DISABLE_GRAPHS")&&std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))=="1"),"Routing-map reuse requires the explicit uncaptured producer path");
        if(expert.sorted_expert_routing){
            unsigned tagged=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i)tagged+=ggml_dsv41_sorted_routing_tag(ggml_graph_node(graph,i),expert.reuse_expert_routing_map)?1:0;
            require(tagged==3,"Explicit count/prefix/scatter routing did not match exactly gate/up/down");
        }
        if(pair_ffn){
            unsigned tagged=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)tagged+=ggml_dsv41_tag_bf16_inplace(ggml_graph_node(graph,i))?1:0;
            require(tagged>0,"Paired graph has no exact BF16 lifetime reuse candidates");
        }
        if(mul_f16_store){
#if defined(GGML_USE_CUDA)
            require(ggml_backend_is_cuda(backend),"F16 route-weight store fusion requires CUDA");
#else
            require(false,"F16 route-weight store fusion requires CUDA");
#endif
            mul_f16_fused=ggml_dsv41_fuse_mul_f16_stores(graph);
            require(mul_f16_fused==1,"F16 route-weight store fusion must match exactly one expert roundtrip");
        }
        if(dense_bf16_store)require(ggml_dsv41_fuse_dense_bf16_stores(graph)>0,"No single-reader original-Q8 BF16 stores to fuse");
        if(rope_inplace){
            unsigned tagged=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i)tagged+=ggml_dsv41_tag_rope_inplace(ggml_graph_node(graph,i))?1:0;
            require(tagged>0,"No bounded exact tail RoPE reuse candidates");
        }
        if(expert.attention_kv_f16_table){
            require(expert.token_tile_attention,"Direct F16 table requires token-tile attention");
            uint64_t raw_mirror=0;uint32_t actual=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto*n=ggml_graph_node(graph,i);if(ggml_dsv41_token_tile_requested(n)){
                require(ggml_dsv41_token_tile_valid(n)&&n->op_params[7]==1,"Direct F16 table lost its allocation/type proof");
                raw_mirror=std::max(raw_mirror,uint64_t(n->src[0]->ne[2]+127)*512*2);++actual;
            }}
            require(actual==attention_tiles&&raw_mirror<=256ull*1024*1024,"Direct F16 raw mirror exceeds existing per-operation ceiling");
            backend_temporary+=raw_mirror;
        }else if(expert.token_tile_attention)backend_temporary+=256ull*1024*1024;
        if(pair_ffn&&!native_pair)backend_temporary+=(in.count>8192?640ull:320ull)*1024*1024; // explicit8192-row MMQ activation/maps/pool temporary
        if(fixed&&backend_temporary>fixed->backend_temporary_reserve_bytes)
            std::fprintf(stderr,"LLAMADECODE_RESERVE layer=%u rows=%u backend_temporary=%llu declared_reserve=%llu rounded=%u split=%u\n",in.layer,in.count,
                (unsigned long long)backend_temporary,(unsigned long long)fixed->backend_temporary_reserve_bytes,rounded_matrices,split_matrices);
        if(fixed)require(backend_temporary<=fixed->backend_temporary_reserve_bytes,"Actual backend temporary exceeds declared fixed reserve");
        // The old context allocator also allocated unused optional inputs.
        // Keep every uploaded leaf allocated and live before graph execution;
        // gallocr may otherwise reuse its slot before the first consumer.
        for(auto*t:uploaded_inputs)ggml_build_forward_expand(graph,t);
        routed_split=-1;
        hash_experts=nullptr;
        if(llama_dsv41_hash_dump_enabled()){ // diagnostic: keep the routed ids readable after compute
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(!std::strcmp(ggml_get_name(ggml_graph_node(graph,i)),"dsv41_ffn_selected_experts")){hash_experts=ggml_graph_node(graph,i);ggml_set_output(hash_experts);}
        }
        if(in.route_hook){
            // LLAMADECODE B1: the graph keeps its original node order (the same
            // allocation plan and kernel sequence as the unrouted step); it is
            // only cut after the router. Nothing before the cut may read experts.
            require(!workspace||!workspace->cache_layouts,"Routed decode split refuses cached layer graphs");
            // LLAMADECODE B2: cut right after the router-id copy into the slot
            // leaf; no expert product may run before it.
            int copy=-1,first_product=-1;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);
                if(!std::strcmp(ggml_get_name(n),"dsv41_expert_slot_copy")){require(copy<0,"Routed decode split found two slot copies");copy=i;}
                if(n->op==GGML_OP_MUL_MAT_ID&&first_product<0)first_product=i;}
            require(copy>=0&&first_product>copy,"Routed decode slot copy must precede every expert product");
            routed_split=copy+1;
        }
        backend_buffers(backend,graph);
        if(llama_dsv41_expert_profile::natural_half(expert)){
            unsigned slots[3]={};
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto * node=ggml_graph_node(graph,i);
                if(ggml_ds4_native_half_requested(node)){require(ggml_ds4_native_half_shape_valid(node),"Invalid native-half graph tag");++slots[node->op_params[3]-1];}}
            require(slots[0]==1&&slots[1]==1&&slots[2]==1,"Native-half graph requires exactly one gate/up/down tag");
        }
        if(workspace&&workspace->cache_layouts){
            std::vector<ggml_tensor*> all;
            for(auto *t=ggml_get_first_tensor(ctx.get());t;t=ggml_get_next_tensor(ctx.get(),t))all.push_back(t);
            compiled->bindings.capture(all,in.start,in.count);
            uint64_t metadata=8ull*1024*1024+compiled->bindings.bytes();for(const auto &v:workspace->variants)metadata+=8ull*1024*1024+v->bindings.bytes();
            require(metadata<=112ull*1024*1024,"Layer graph variants exceeded explicit metadata admission");
            workspace->variants.push_back(std::move(fresh));
        }
        }
        if(const char *diag=std::getenv("DS4PERF_DIAGNOSTIC_LAYERS")){if(*diag){
            llama_dsv41_layer_log("DS4PERF_GRAPH_TRACE layer=%u start=%u count=%u hit=%u semantic_fnv64=%016llx",in.layer,in.start,in.count,unsigned(graph_cache_hit),
                (unsigned long long)llama_dsv41_graph_digest(ctx.get(),graph));std::fputc(10,stderr);std::fflush(stderr);
        }}
        const auto buft=ggml_backend_get_default_buffer_type(backend);
        struct allocator_owner{ggml_gallocr_t value=nullptr;ggml_backend_t backend=nullptr;void *pending_copies=nullptr;bool owned=true;
            void release(){if(value){
#ifdef GGML_USE_CUDA
                if(pending_copies)pinned_storage::cleanup(cudaStreamSynchronize(static_cast<cudaStream_t>(pending_copies)));
#endif
                if(backend)ggml_backend_synchronize(backend);if(owned)ggml_gallocr_free(value);value=nullptr;}}
            ~allocator_owner(){release();}} allocation;
        allocation.backend=backend;
#ifdef GGML_USE_CUDA
        allocation.pending_copies=pinned?pinned->copies:nullptr;
#endif
        allocation.owned=!workspace;
        allocation.value=workspace?workspace->value:ggml_gallocr_new(buft);require(allocation.value,"V4.1 lifetime allocator creation failed");
        const uint64_t scratch=llama_dsv41_allocate_native_graph(allocation.value,graph,
            backend_temporary,admitted_graph_bytes);
        const uint64_t actual=scratch-backend_temporary;
        uint64_t rope_aliases=0,rope_alias_bytes=0,rope_tagged=0;
        if(rope_inplace){
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto*t=ggml_graph_node(graph,i);
                if(!ggml_dsv41_rope_inplace_requested(t))continue;++rope_tagged;
                if(t->data==t->src[0]->data&&t->buffer==t->src[0]->buffer){++rope_aliases;rope_alias_bytes+=ggml_nbytes(t);}
            }
            require(rope_aliases>0,"RoPE allocator did not reuse any last-reader source");
        }
        if(pair_ffn){
            uint64_t tagged=0,aliases=0,bytes=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto*t=ggml_graph_node(graph,i);
                if(!ggml_dsv41_bf16_inplace_requested(t))continue;++tagged;
                if(t->data==t->src[0]->data&&t->buffer==t->src[0]->buffer){++aliases;bytes+=ggml_nbytes(t);}
            }
            require(aliases>0,"Paired BF16 allocator did not actually reuse any last-reader buffer");
            llama_dsv41_layer_log("DS4PERF_BF16_INPLACE_LAYOUT layer=%u start=%u count=%u tagged=%llu actual_aliases=%llu alias_bytes=%llu original_lifetime_checks=1\n",in.layer,in.start,in.count,(unsigned long long)tagged,(unsigned long long)aliases,(unsigned long long)bytes);
        }
        if(const char *diag=std::getenv("DS4PERF_DIAGNOSTIC_LAYERS")){if(*diag){
            llama_dsv41_layer_log("DS4PERF_LAYOUT_TRACE layer=%u start=%u count=%u hit=%u layout_fnv64=%016llx",in.layer,in.start,in.count,unsigned(graph_cache_hit),
                (unsigned long long)llama_dsv41_graph_digest(ctx.get(),graph,true));std::fputc(10,stderr);std::fflush(stderr);
        }}
        require(boundary.bytes()<=actual,"Boundary outputs escaped actual graph admission");
        if(boundary.active())llama_dsv41_layer_log("BOUNDARY_CAPTURE_ADMITTED retained_bytes=%llu live_graph_bytes=%llu host_readback_bytes=65536\n",
            (unsigned long long)boundary.bytes(),(unsigned long long)actual);
        result pending;pending.prepare_ms=elapsed(prepare_started);pending.scratch_bytes=actual+backend_temporary;pending.graph_nodes=ggml_graph_n_nodes(graph);
        const auto flush_started=clock_type::now();
        llama_dsv41_layer_log("[dsv41-executor] layer=%u rows=%u q8_half=%u row_slices=%u graph_cache_hit=%u live_graph_bytes=%llu backend_temporary_bytes=%llu\n",
            in.layer,in.count,rounded_matrices,split_matrices,unsigned(graph_cache_hit),(unsigned long long)actual,(unsigned long long)backend_temporary);
        std::fflush(stderr);
        if(probe.enabled)probe.prepare_flush_ms=elapsed(flush_started);
        const auto upload_start=clock_type::now();
        // These host vectors outlive every queued async transfer. In
        // particular, frequencies and publication IDs are no longer temporaries.
        const auto raw=llama_dsv41_rope_frequencies(false),compressed=llama_dsv41_rope_frequencies(true);
        std::vector<int32_t>pair_history;
        if(input.pair_history_ids&&!resident_metadata){const llama_dsv41_tile_plan second(p.start+attention_rows,std::min(attention_rows,in.count-attention_rows),cache.capacity(),ratio,attention_rows,true);pair_history=second.history_ids;}
        const std::vector<float>raw_values(raw.begin(),raw.end()),compressed_values(compressed.begin(),compressed.end());
        std::vector<int64_t>publication_rows(p.published_count);
        for(uint32_t i=0;i<p.published_count;++i)publication_rows[i]=p.published_first+i;
        struct input_upload_guard {
            ggml_backend_t backend;pinned_storage *pinned;bool live;
            ~input_upload_guard(){if(live){
#ifdef GGML_USE_CUDA
                if(pinned)pinned_storage::cleanup(cudaStreamSynchronize(pinned->copies));
#endif
                ggml_backend_synchronize(backend);
                llama_dsv41_layer_log("DS4PERF_ASYNC_INPUT_GUARD_DRAINED host_vectors_live=1 compute=1 pinned_copies=1\n");
            }}
        } input_guard{backend,pinned.get(),async_inputs};
        uint64_t async_copies=0,async_bytes=0;
        auto upload_input=[&](ggml_tensor*t,const auto &values){
            upload(t,values,async_inputs?backend:nullptr);
            if(async_inputs){++async_copies;async_bytes+=values.size()*sizeof(typename std::decay_t<decltype(values)>::value_type);}
        };
        if(direct_input){
#ifdef GGML_USE_CUDA
            // The admitted GPU prefix already holds the prior layer's exact outputs.
            // Other rows and the first layer still copy the validated RAM lease.
            if(!device_hidden||in.layer==0){
                if(pinned->prefetch_buffer&&in.start)pinned->prefetch_take(*in.direct,in.layer,input.residual,input.previous_pre);
                else{pinned->queue(true,*in.direct,0,81920,input.residual);pinned->queue(true,*in.direct,81920,16,input.previous_pre);}
            }else if(pinned->prefetch_buffer)require(!pinned->prefetch_valid,"Resident input unexpectedly retained a prefetched tile");
#endif
        }else if(!device_hidden||in.layer==0){upload_input(input.residual,in.residual);upload_input(input.previous_pre,in.previous_pre);}
        upload_input(input.positions,p.positions);
        if(input.pair_history_ids&&!resident_metadata)upload_input(input.pair_history_ids,pair_history);
        if(!raw_template){if(input.history_ids)upload_input(input.history_ids,p.history_ids);upload_input(input.raw_ids,p.raw_ids);upload_input(input.raw_mask,p.raw_mask);}
        if(!resident_metadata){upload_input(input.raw_frequencies,raw_values);upload_input(input.compressed_frequencies,compressed_values);}
        if(input.visible)upload_input(input.visible,p.compressed_visible);
        if(p.published_count){upload_input(input.publication_ids,publication_rows);upload_input(input.publication_positions,p.compressed_positions);}
        if(direct_input){
#ifdef GGML_USE_CUDA
            if(reuse&&!resident_ids)pinned->queue(true,*in.direct,81936,uint64_t(selected_width)*4,input.previous_selected);
            if(candidates&&!resident_ids)pinned->queue(true,*in.direct,83984,uint64_t(candidate_width)*4,input.candidate_ids);
#endif
        }else{if(reuse)upload_input(input.previous_selected,in.selected);if(candidates)upload_input(input.candidate_ids,in.candidates);}
#ifdef GGML_USE_CUDA
        double spill_in_ms=0;
        if(spilled&&in.layer){
            // Exact bytes this tile left in its own RAM records after the previous
            // layer (BF16 hidden at 0, pre at 81920) and the mirrored compact IDs.
            const auto spill_started=clock_type::now();
            require(spill_slot&&resident_ids,"Spilled tile requires the BF16 slot and resident IDs");
            pinned->queue(true,*in.direct,0,40960,spill_slot);pinned->queue(true,*in.direct,81920,16,input.previous_pre);
            uint64_t ids=0;
            if(reuse){pinned->queue(true,*in.direct,81936,uint64_t(selected_width)*4,input.previous_selected);ids+=uint64_t(selected_width)*4;}
            if(candidates){pinned->queue(true,*in.direct,83984,uint64_t(candidate_width)*4,input.candidate_ids);ids+=uint64_t(candidate_width)*4;}
            pinned->upload_ready();   // the BF16 unpack below runs on the compute stream
            resident->spill_h2d_bytes+=uint64_t(in.count)*(40976+ids);spill_in_ms=elapsed(spill_started);
        }
#endif
        if(engram){upload_input(input.engram_rows,in.engram_rows);if(!all_text)upload_input(input.text_mask,in.text_mask);}
        if(!zero_kinds)upload_input(input.token_kinds,in.token_kinds);
        boundary.upload_replacements();
        ggml_cgraph compute_view{};ggml_cgraph *compute_graph=graph;
        if(resident_bf16&&in.layer){
            require(ggml_graph_n_nodes(graph)>1&&ggml_graph_node(graph,0)==input.residual&&
                input.residual->op==GGML_OP_CPY&&input.residual->src[0]&&input.residual->src[0]->type==GGML_TYPE_BF16,
                "Resident BF16 unpack must be the first exact graph dependency");
            auto unpack=ggml_graph_view(graph,0,1);
            require(ggml_backend_graph_compute_async(backend,&unpack)==GGML_STATUS_SUCCESS,
                "Resident BF16 input conversion failed");
            // Validate its actual F32 output before any layer compute consumes
            // it, then execute the existing graph suffix exactly once.
            compute_view=ggml_graph_view(graph,1,ggml_graph_n_nodes(graph));compute_graph=&compute_view;
        }
        if(direct_input){
#ifdef GGML_USE_CUDA
            pinned->upload_ready();llama_dsv41_validation check;
            check.residual=static_cast<const float*>(input.residual->data);check.residual_n=uint64_t(5120)*4*in.count;
            check.pre=static_cast<const float*>(input.previous_pre->data);check.pre_n=uint64_t(4)*in.count;
            check.compressed_rows=in.compressed_rows;
            if(reuse){check.selected=static_cast<const int32_t*>(input.previous_selected->data);check.selected_n=uint64_t(selected_width)*in.count;}
            if(candidates){check.candidates=static_cast<const int32_t*>(input.candidate_ids->data);check.candidates_n=uint64_t(candidate_width)*in.count;}
            if(engram){check.engram=static_cast<const float*>(input.engram_rows->data);check.engram_n=uint64_t(row_width)*in.count;}
            pinned->validate(check);
#endif
        }
        // The existing direct-input validator synchronized this compute stream
        // after all metadata copies and its upload_ready wait. Only now may the
        // host lifetime guard disarm; any earlier error drains before unwind.
        input_guard.live=false;
        if(async_inputs)llama_dsv41_layer_log("DS4PERF_ASYNC_INPUTS_COMPLETE layer=%u start=%u count=%u copies=%llu bytes=%llu existing_CUDA_async_API=1 host_vectors_retained=1 validator_complete=1\n",
            in.layer,in.start,in.count,(unsigned long long)async_copies,(unsigned long long)async_bytes);
        if(resident_metadata)llama_dsv41_layer_log("DS4PERF_RESIDENT_METADATA_COMPLETE layer=%u start=%u count=%u raw_template=%u zero_kinds=%u all_text=%u avoided_H2D_bytes=%llu exact_initialized_templates=1\n",in.layer,in.start,in.count,unsigned(raw_template),unsigned(zero_kinds),unsigned(all_text),
            (unsigned long long)(256+(raw_template?uint64_t(in.count)*128*8+127*4:0)+(zero_kinds?uint64_t(in.count)*4:0)+(all_text?in.count:0)));
        pending.upload_ms=elapsed(upload_start);
        // An error after entering the graph can leave partial aliased writes.
        // Poison first, and certify this tile only after all bounded readbacks.
        if(llama_dsv41_hash_dump_enabled()){ggml_backend_synchronize(backend);
            llama_dsv41_hash_phase("in",in.layer,in.position_offset+in.start,in.count,graph,{},{},{});}
        cache.execution_poisoned=true;const auto compute_start=clock_type::now();
#if defined(GGML_USE_CUDA)
        ggml_ds4_native_half_stats expert_before{},expert_after{};
        const auto index_before=index_cublas?ggml_backend_cuda_dsv41_index_cublas_queued():0;
        const auto dense_before=expert.dense_inputs_f16?ggml_backend_cuda_dsv41_dense_half_queued():0;
        const auto dense_fp4_before=dense_mxfp4?ggml_backend_cuda_native_mxfp4_dense_queued():0;
        const bool topk_radix=std::getenv("LLAMA_DSV41_TOPK_RADIX")&&std::string(std::getenv("LLAMA_DSV41_TOPK_RADIX"))=="1";
        const auto topk_radix_before=topk_radix?ggml_backend_cuda_dsv41_topk_radix_queued():0;
        const auto rope_inplace_before=rope_inplace?ggml_backend_cuda_dsv41_rope_inplace_queued():0;
        const auto dense_bf16_store_before=dense_bf16_store?ggml_backend_cuda_dsv41_dense_bf16_store_queued():0;
        const auto mul_f16_store_before=mul_f16_store?ggml_backend_cuda_dsv41_mul_f16_store_queued():0;
        const bool cast_vector=std::getenv("LLAMA_DSV41_CAST_VECTOR")&&std::string(std::getenv("LLAMA_DSV41_CAST_VECTOR"))=="1";
        const auto cast_vector_before=cast_vector?ggml_backend_cuda_dsv41_cast_vector_queued():0;
        const auto cast_elements_before=cast_vector?ggml_backend_cuda_dsv41_cast_vector_elements():0;
        const auto cast_nc_before=cast_vector?ggml_backend_cuda_dsv41_cast_vector_nc_queued():0;
        const bool bf16_vector=std::getenv("LLAMA_DSV41_BF16_VECTOR")&&std::string(std::getenv("LLAMA_DSV41_BF16_VECTOR"))=="1";
        const auto bf16_vector_before=bf16_vector?ggml_backend_cuda_dsv41_bf16_vector_queued():0;
        const auto token_tile_before=expert.token_tile_attention?ggml_backend_cuda_dsv41_token_tile_queued():0;
        const auto stream_topk_before=expert.index_key_windows?ggml_backend_cuda_dsv41_stream_topk_queued():0;
        const auto f16_table_before=expert.attention_kv_f16_table?ggml_backend_cuda_dsv41_token_f16_table_queued():0;
        const bool attention_g16=std::getenv("LLAMA_DSV41_ATTN_G16")&&std::string(std::getenv("LLAMA_DSV41_ATTN_G16"))=="1";
        const auto attention_g16_before=attention_g16?ggml_backend_cuda_dsv41_token_tile_g16_queued():0;
        const auto union_before=merge_rows?ggml_backend_cuda_dsv41_token_union_queued():0;
        const auto expert_cublas_before=expert.expert_f16_cublas?ggml_backend_cuda_dsv41_expert_cublas_queued():0;
        const auto expert_fp4_before=expert.expert_mxfp4?ggml_backend_cuda_native_mxfp4_routed_queued():0;
        const auto hc_expand_before=ggml_backend_cuda_dsv41_hc_expand_queued();
        const auto hc_bf16_before=expert.hc_bf16_epilogue?ggml_backend_cuda_dsv41_hc_bf16_queued():0;
        const auto hc_half_before=hc_projection_f16_gemm?ggml_backend_cuda_dsv41_hc_half_queued():0;
        if(llama_dsv41_expert_profile::natural_half(expert))require(ggml_backend_cuda_ds4_native_half_begin(&expert_before),"Concurrent/incomplete native-half process graph witness");
#endif
        bool prefetch_next=false;
#ifdef GGML_USE_CUDA
        // LLAMADECODE_TILE_TIMING: two timing events, created once per process, recorded on the
        // ggml CUDA backend's compute stream immediately before the tile graph launch and
        // immediately after the last launch (before the existing synchronize).
        static cudaEvent_t tile_events[2]={nullptr,nullptr};
        cudaStream_t tile_stream=nullptr;
        if(tile_timing_enabled()&&ggml_backend_is_cuda(backend)){
            if(!tile_events[0]){pinned_storage::ok(cudaEventCreate(&tile_events[0]),"Tile timing event creation failed");
                pinned_storage::ok(cudaEventCreate(&tile_events[1]),"Tile timing event creation failed");}
            tile_stream=static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(backend));
            pinned_storage::ok(cudaEventRecord(tile_events[0],tile_stream),"Tile timing start event record failed");
        }
#endif
        const auto launch_start=clock_type::now();
#ifdef GGML_USE_CUDA
        // This guard is newer than the local graph/context owners, so any
        // exception after async launch drains readers/compute BEFORE those
        // owners unwind. The parent RAM lease remains live until run returns.
        struct async_prefetch_guard {
            pinned_storage*storage;ggml_backend_t backend;bool live=false;
            ~async_prefetch_guard(){if(live){pinned_storage::cleanup(cudaStreamSynchronize(storage->copies));ggml_backend_synchronize(backend);llama_dsv41_layer_log("DS4PERF_PREFETCH_ASYNC_GUARD_DRAINED copies=1 compute=1 graph_resources_live=1\n");}}
        } asynchronous{pinned.get(),backend,false};
        prefetch_next=pinned&&pinned->prefetch_buffer&&in.lookahead&&(in.layer==0||!resident||in.lookahead->start()>=resident->tokens);
        if(prefetch_next){
            asynchronous.live=true;pinned->prefetch_begin_compute();
            require(ggml_backend_graph_compute_async(backend,compute_graph)==GGML_STATUS_SUCCESS,"V4.1 asynchronous tile backend computation failed");
            pinned->prefetch_queue_next(*in.lookahead,in.layer);
        }else
#endif
        if(in.route_hook){
            require(!prefetch_next&&in.dense_bank&&in.routed_bank&&in.dense_bank!=in.routed_bank,"Routed decode split requires its two banks and no lookahead");
            const int nodes=ggml_graph_n_nodes(compute_graph);const int offset=compute_graph==graph?0:ggml_graph_n_nodes(graph)-nodes;
            const int split=routed_split-1-offset; // the slot copy (last node before the hook)
            require(routed_split>0&&split>=0&&split+1<nodes,"Routed decode split boundary lost");
            ggml_tensor*selected=nullptr;for(int i=0;i<=split;++i){auto*n=ggml_graph_node(compute_graph,i);if(!std::strcmp(ggml_get_name(n),"dsv41_ffn_selected_experts"))selected=n;}
            require(selected,"Routed decode router is not before the split");
            auto reads=[&](const ggml_tensor*node,const void*bank){
                if(node->buffer==bank)return true;
                for(const auto*src:node->src){if(!src)continue;const ggml_tensor*root=src;for(unsigned d=0;d<32&&root->view_src;++d)root=root->view_src;
                    if(src->buffer==bank||root->buffer==bank)return true;}
                return false;};
            for(int i=0;i<=split;++i){auto*n=ggml_graph_node(compute_graph,i);if(reads(n,in.routed_bank)){std::fprintf(stderr,"LLAMADECODE_PREROUTER_BANK0 layer=%u node=%d/%d name=%s op=%s src0=%s src1=%s src2=%s\n",in.layer,i,split,n->name,ggml_op_name(n->op),n->src[0]?n->src[0]->name:"-",n->src[1]?n->src[1]->name:"-",n->src[2]?n->src[2]->name:"-");require(false,"A pre-router node reads the routed-expert bank");}}
            require(selected->type==GGML_TYPE_I32&&selected->ne[0]==6&&ggml_nelements(selected)==6*int64_t(in.count),"Routed decode selected-expert geometry differs");
            auto before=ggml_graph_view(compute_graph,0,split+1);
            require(ggml_backend_graph_compute_async(backend,&before)==GGML_STATUS_SUCCESS,"V4.1 pre-router computation failed");
            ggml_backend_synchronize(backend);
            std::vector<int32_t>ids(size_t(ggml_nelements(selected)));ggml_backend_tensor_get(selected,ids.data(),0,ids.size()*sizeof(int32_t));
            llama_dsv41_layer_log("LLAMADECODE_ROUTE_IDS layer=%u ids=%d,%d,%d,%d,%d,%d\n",in.layer,ids[0],ids[1],ids[2],ids[3],ids[4],ids[5]);std::fflush(stderr);
            const auto routed_started=clock_type::now();std::string why;
            std::vector<int32_t>slots;const bool fetched=in.route_hook(ids,slots,why);if(!fetched)std::fprintf(stderr,"LLAMADECODE_ROUTED_FAILED layer=%u reason=%s\n",in.layer,why.c_str());
            require(fetched,"Routed expert fetch failed");
            require(expert_slot_input&&slots.size()==ids.size()&&ggml_nbytes(expert_slot_input)==slots.size()*4,"Routed expert slot ids differ from the router");
            ggml_backend_tensor_set(expert_slot_input,slots.data(),0,slots.size()*4);
            llama_dsv41_layer_log("LLAMADECODE_ROUTED layer=%u ids=%d,%d,%d,%d,%d,%d fetch_wait_ms=%.3f split_node=%d nodes=%d\n",in.layer,
                ids[0],ids[1],ids[2],ids[3],ids[4],ids[5],elapsed(routed_started),split,nodes);
            auto after=ggml_graph_view(compute_graph,split+1,nodes);
            require(ggml_backend_graph_compute_async(backend,&after)==GGML_STATUS_SUCCESS,"V4.1 post-router computation failed");
        }else
        require(ggml_backend_graph_compute_async(backend,compute_graph)==GGML_STATUS_SUCCESS,"V4.1 tile backend computation failed");
        tile_launch_ms=elapsed(launch_start);
#ifdef GGML_USE_CUDA
        if(tile_stream)pinned_storage::ok(cudaEventRecord(tile_events[1],tile_stream),"Tile timing end event record failed");
#endif
        const auto sync_start=clock_type::now();
        ggml_backend_synchronize(backend);pending.compute_ms=elapsed(compute_start);tile_sync_ms=elapsed(sync_start);
#ifdef GGML_USE_CUDA
        // Both events precede the synchronize on the same stream, so they are complete here.
        if(tile_stream)pinned_storage::ok(cudaEventElapsedTime(&tile_gpu_ms,tile_events[0],tile_events[1]),"Tile timing elapsed read failed");
#endif
        if(llama_dsv41_hash_dump_enabled()){
            std::vector<int32_t>experts;
            if(hash_experts){experts.resize(size_t(ggml_nelements(hash_experts)));ggml_backend_tensor_get(hash_experts,experts.data(),0,experts.size()*4);}
            llama_dsv41_hash_phase("out",in.layer,in.position_offset+in.start,in.count,graph,
                {values.residual,values.next_pre,values.selected,values.selected_mask,values.candidates,values.next_pool_kv,values.next_pool_score,hash_experts},
                {w.ffn.gate,w.ffn.up,w.ffn.down},experts);
        }
        const auto post_started=clock_type::now();
        llama_dsv41_layer_log("[dsv41-compute-timeline] layer=%u start=%u begin_epoch_ms=%.6f end_epoch_ms=%.6f\n",
            in.layer,in.start,std::chrono::duration<double,std::milli>(compute_start.time_since_epoch()).count(),
            std::chrono::duration<double,std::milli>(clock_type::now().time_since_epoch()).count());
#if defined(GGML_USE_CUDA)
        if(topk_radix){
            uint64_t expected=0;
            for(int i=0;i<ggml_graph_n_nodes(compute_graph);++i){const auto *node=ggml_graph_node(compute_graph,i);
                if(ggml_dsv41_stream_topk_requested(node)){require(ggml_dsv41_stream_topk_shape(node),"Private selector lost bounded shape");expected+=node->ne[0]==512;continue;}
                if(node->op!=GGML_OP_TOP_K||node->ne[0]!=512||!node->src[0]||node->src[0]->ne[0]<=8192||ggml_nrows(node->src[0])<=1)continue;
                uint64_t width=(uint64_t(node->src[0]->ne[0])+8191)/8192*512;expected+=2;
                while(width>8192){width=(width+8191)/8192*512;++expected;}
            }
            const auto actual=ggml_backend_cuda_dsv41_topk_radix_queued()-topk_radix_before;
            require(actual==expected,"Actual exact radix top-k first/merge dispatch coverage differs");
            llama_dsv41_layer_log("DS4PERF_TOPK_RADIX_COMPLETE layer=%u start=%u count=%u actual=%llu packed_score_index=1 original_NaN_subnormal_network=1 weight_formats_unchanged=1\n",in.layer,in.start,in.count,(unsigned long long)actual);
        }
        if(expert.index_key_windows){
            uint64_t expected=0;for(int i=0;i<ggml_graph_n_nodes(compute_graph);++i)expected+=ggml_dsv41_stream_topk_requested(ggml_graph_node(compute_graph,i))?1:0;
            require(ggml_backend_cuda_dsv41_stream_topk_queued()==stream_topk_before+expected,"Actual bounded streaming selector coverage differs");
            if(expected)llama_dsv41_layer_log("DS4PERF_STREAM_TOPK_COMPLETE layer=%u start=%u count=%u actual=%llu explicit_bounded_tag=1\n",in.layer,in.start,in.count,(unsigned long long)expected);
        }
        if(rope_inplace){
            require(ggml_backend_cuda_dsv41_rope_inplace_queued()==rope_inplace_before+rope_aliases,"Actual completed in-place RoPE dispatch coverage differs");
            llama_dsv41_layer_log("DS4PERF_ROPE_INPLACE_COMPLETE layer=%u start=%u count=%u tagged=%llu actual_aliases=%llu alias_bytes=%llu original_lifetime_checks=1 unchanged_pair_math=1\n",in.layer,in.start,in.count,(unsigned long long)rope_tagged,(unsigned long long)rope_aliases,(unsigned long long)rope_alias_bytes);
        }
        if(bf16_vector){
            uint64_t expected=0;for(int i=0;i<ggml_graph_n_nodes(compute_graph);++i)expected+=ggml_dsv41_bf16_vector_eligible(ggml_graph_node(compute_graph,i))?1:0;
            require(ggml_backend_cuda_dsv41_bf16_vector_queued()==bf16_vector_before+expected,"Actual exact vector BF16 dispatch coverage differs");
            llama_dsv41_layer_log("DS4PERF_BF16_VECTOR_COMPLETE layer=%u start=%u count=%u actual=%llu exact_helper=1 aligned_vector4=1 scalar_fallback=1\n",in.layer,in.start,in.count,(unsigned long long)expected);
        }
        if(dense_bf16_store){
            uint64_t expected=0;for(int i=0;i<ggml_graph_n_nodes(compute_graph);++i)expected+=ggml_dsv41_dense_bf16_store_requested(ggml_graph_node(compute_graph,i))?1:0;
            require(expected>0&&ggml_backend_cuda_dsv41_dense_bf16_store_queued()==dense_bf16_store_before+expected,"Actual completed original-Q8 BF16 final stores differ");
            llama_dsv41_layer_log("DS4PERF_DENSE_BF16_STORE_COMPLETE layer=%u start=%u count=%u actual=%llu unchanged_StreamK_partials=1 final_fixup_round=1 original_Q8_weights=1\n",in.layer,in.start,in.count,(unsigned long long)expected);
        }
        if(mul_f16_store){
            require(ggml_backend_cuda_dsv41_mul_f16_store_queued()==mul_f16_store_before+mul_f16_fused,"Actual F16 route-weight store coverage differs");
            llama_dsv41_layer_log("DS4PERF_MUL_F16_STORE_COMPLETE layer=%u start=%u count=%u actual=%u original_cast_helpers=1 single_reader_chain=1\n",in.layer,in.start,in.count,mul_f16_fused);
        }
        if(cast_vector){
            const auto calls=ggml_backend_cuda_dsv41_cast_vector_queued()-cast_vector_before;
            const auto elements=ggml_backend_cuda_dsv41_cast_vector_elements()-cast_elements_before;
            const auto nc=ggml_backend_cuda_dsv41_cast_vector_nc_queued()-cast_nc_before;
            // A one-token RAM-only graph can use the existing scalar/MMVQ
            // paths exclusively. Larger graphs must witness vector dispatch.
            require((calls>0||in.count==1)&&elements>=4*calls&&nc<=calls&&
                (calls||(!elements&&!nc)),"Actual completed vector cast counters are empty/inconsistent");
            llama_dsv41_layer_log("DS4PERF_CAST_VECTOR_COMPLETE layer=%u start=%u count=%u calls=%llu elements=%llu strided_calls=%llu original_cast_helper=1 alignment_tail_overlap_guards=1\n",in.layer,in.start,in.count,(unsigned long long)calls,(unsigned long long)elements,(unsigned long long)nc);
        }
        if(expert.expert_mxfp4){const uint64_t expected=in.count>4?3:0;
            require(ggml_backend_cuda_native_mxfp4_routed_queued()==expert_fp4_before+expected,"Actual expert native FP4 activation/MMA witness differs");
            llama_dsv41_layer_log("DS4PERF_EXPERT_MXFP4_COMPLETE layer=%u start=%u count=%u actual=%llu weights_FP4=1 activations_FP4=1 accumulation_F32=1 small_batch_original=%u\n",in.layer,in.start,in.count,(unsigned long long)expected,unsigned(in.count<=4));
        }
        if(index_cublas){uint64_t expected=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)expected+=ggml_dsv41_index_cublas_requested(ggml_graph_node(graph,i))?1:0;
            const auto after=ggml_backend_cuda_dsv41_index_cublas_queued();
            require(after>=index_before&&after-index_before==expected,"Actual bounded GLM index dispatch witness differs");
            if(expected)llama_dsv41_layer_log("[dsv41-index-cublas] layer=%u start=%u rows=%u actual_completed=%llu exact_input_and_dot_admission=%u bf16_operands=%u f32_unrounded_operands=%u native_order_head_accumulation=1\n",in.layer,in.start,in.count,(unsigned long long)expected,unsigned(!expert.index_bf16&&!index_f32_cublas_enabled()),unsigned(expert.index_bf16),unsigned(index_f32_cublas_enabled()));
        }
        if(expert.dense_inputs_f16){
            uint64_t expected=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)expected+=ggml_dsv41_dense_half_requested(ggml_graph_node(graph,i))?1:0;
            const auto after=ggml_backend_cuda_dsv41_dense_half_queued();
            require(after>=dense_before&&after-dense_before==expected,"Actual shared F16/F16/F32 dense dispatch witness differs");
            llama_dsv41_layer_log("DS4PERF_DENSE_HALF_COMPLETE layer=%u start=%u count=%u actual=%llu hc_excluded=1\n",in.layer,in.start,in.count,(unsigned long long)expected);
        }
        if(dense_mxfp4){uint64_t expected=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto*n=ggml_graph_node(graph,i);expected+=n->op==GGML_OP_MUL_MAT&&n->src[0]->type==GGML_TYPE_MXFP4?1:0;}
            require(expected&&ggml_backend_cuda_native_mxfp4_dense_queued()==dense_fp4_before+expected,"Actual shared dense FP4 activation/MMA witness differs");
            llama_dsv41_layer_log("DS4PERF_DENSE_MXFP4_COMPLETE layer=%u start=%u count=%u actual=%llu weights_FP4=1 activations_FP4=1 accumulation_F32=1\n",in.layer,in.start,in.count,(unsigned long long)expected);}
        if(expert.token_tile_attention){
            require(ggml_backend_cuda_dsv41_token_tile_queued()==token_tile_before+attention_tiles,"Actual reused token-tile dispatch witness differs");
            if(expert.attention_kv_f16_table){
                require(ggml_backend_cuda_dsv41_token_f16_table_queued()==f16_table_before+attention_tiles,"Actual direct F16 attention table dispatch witness differs");
                llama_dsv41_layer_log("DS4PERF_ATTN_F16_TABLE_COMPLETE layer=%u start=%u count=%u actual=%u compressed_half_mirror_bytes=0 canonical_cache_unchanged=1 existing_HMMA_body=1\n",in.layer,in.start,in.count,attention_tiles);
            }
            if(attention_g16){
                require(ggml_backend_cuda_dsv41_token_tile_g16_queued()==attention_g16_before+attention_tiles,"Actual G16 attention dispatch coverage differs");
                llama_dsv41_layer_log("DS4PERF_ATTN_G16_COMPLETE layer=%u start=%u count=%u actual=%u R32=1 inactive_KV_score_alias=1 smem_bytes=77824 original_weights=1\n",in.layer,in.start,in.count,attention_tiles);
            }
            if(merge_rows){require(ggml_backend_cuda_dsv41_token_union_queued()==union_before+(selected_width?attention_tiles:0),"Actual shared selected-row union dispatch witness differs");
                llama_dsv41_layer_log("DS4PERF_TOKEN_UNION_COMPLETE layer=%u start=%u count=%u actual=%u packed_mask_bits=4\n",in.layer,in.start,in.count,unsigned(selected_width?attention_tiles:0));}
            llama_dsv41_layer_log("DS4PERF_TOKEN_TILE_COMPLETE layer=%u start=%u count=%u actual=%u half_probabilities=1 original_F32_sinks=1 canonical_caches_unchanged=1\n",in.layer,in.start,in.count,attention_tiles);
        }
        if(expert.expert_f16_cublas){
            require(ggml_backend_cuda_dsv41_expert_cublas_queued()==expert_cublas_before+3,"Actual F16 expert cuBLAS dispatch witness differs");
            llama_dsv41_layer_log("DS4PERF_EXPERT_CUBLAS_COMPLETE layer=%u start=%u count=%u actual=3 operands_F16=1 accumulation_F32=1 shared_gather_scatter=1\n",in.layer,in.start,in.count);
        }
        if(hc_projection_f16_gemm){uint64_t expected=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i)expected+=ggml_dsv41_hc_half_requested(ggml_graph_node(graph,i))?1:0;
            require(expected==(in.count>8?2*attention_tiles:0)&&ggml_backend_cuda_dsv41_hc_half_queued()==hc_half_before+expected,"Actual HC F16/F16/F32 cuBLAS dispatch witness differs");
            llama_dsv41_layer_log("DS4PERF_HC_HALF_COMPLETE layer=%u start=%u count=%u actual=%llu accumulation_F32=1\n",in.layer,in.start,in.count,(unsigned long long)expected);}
        uint64_t hc_expected=0;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)hc_expected+=ggml_dsv41_hc_expand_requested(ggml_graph_node(graph,i))?1:0;
        require(ggml_backend_cuda_dsv41_hc_expand_queued()==hc_expand_before+hc_expected,"Actual reused HC expansion dispatch witness differs");
        if(hc_expected)llama_dsv41_layer_log("DS4PERF_HC_EXPAND_COMPLETE layer=%u start=%u count=%u actual=%llu no_FMA=1 original_BF16_boundaries=1\n",in.layer,in.start,in.count,(unsigned long long)hc_expected);
        if(expert.hc_bf16_epilogue){
            uint64_t expected=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i)expected+=ggml_dsv41_hc_bf16_requested(ggml_graph_node(graph,i))?1:0;
            require(expected==2*attention_tiles&&ggml_backend_cuda_dsv41_hc_bf16_queued()==hc_bf16_before+expected,"Actual HC BF16 epilogue dispatch differs");
            llama_dsv41_layer_log("DS4PERF_HC_BF16_EPILOGUE layer=%u start=%u count=%u actual=%llu removed_rounding_passes=%llu exact_existing_rounding=1 no_FMA=1\n",in.layer,in.start,in.count,(unsigned long long)expected,(unsigned long long)expected);
        }

        if(llama_dsv41_expert_profile::natural_half(expert)){
            require(ggml_backend_cuda_ds4_native_half_complete(&expert_before,&expert_after),"Actual native-half queued/completed three-slot witness mismatch");
            if(expert_after.completed_graphs==1)llama_dsv41_layer_log("DS4_NATIVE_HALF_FIRST_GRAPH_COMPLETE layer=%u rows=%u actual_queued_slots=3 after_existing_graph_sync=1\n",in.layer,in.count);
        }
#endif
        const auto cache_started=clock_type::now();
        if(probe.enabled)probe.post_checks_ms=elapsed(post_started);
        if(workspace&&workspace->casts)workspace->casts->complete();
        pending.weight_cache_ms=elapsed(cache_started);
        const auto boundary_started=clock_type::now();boundary.write(backend,w.attention.kv,w.attn_norm,w.attention.kv_norm);
        if(probe.enabled)probe.boundary_ms=elapsed(boundary_started);
        const auto readback_start=clock_type::now();
        const bool selected_unchanged=!selected_width||(reuse&&same_compact_ids(values.selected,input.previous_selected));
        const bool candidates_unchanged=in.layer<20||!candidate_width||(in.layer>20&&same_compact_ids(values.candidates,input.candidate_ids));
        if(direct_input){
#ifdef GGML_USE_CUDA
            auto geometry=[&](ggml_tensor*t,ggml_type type,uint64_t count){
                require(!count?!t:(t&&t->type==type&&ggml_is_contiguous(t)&&uint64_t(ggml_nelements(t))==count&&ggml_nbytes(t)==count*4),
                    "Direct RAM completed tensor type/geometry changed");
            };
            geometry(values.residual,GGML_TYPE_F32,uint64_t(5120)*4*in.count);
            geometry(values.next_pre,GGML_TYPE_F32,uint64_t(4)*in.count);
            geometry(values.selected,GGML_TYPE_I32,uint64_t(selected_width)*in.count);
            geometry(values.candidates,GGML_TYPE_I32,in.layer>=20?uint64_t(candidate_width)*in.count:0);
            geometry(values.dspark_mean,GGML_TYPE_F32,in.layer>=37?uint64_t(5120)*in.count:0);
            require(query_source(in.layer)||selected_unchanged,"Reader layer changed its carried selected-ID graph source");
            require(in.layer==20||candidates_unchanged,"Reader layer changed its carried candidate-ID graph source");
            llama_dsv41_validation check;
            check.residual=static_cast<const float*>(values.residual->data);check.residual_n=uint64_t(5120)*4*in.count;
            check.pre=static_cast<const float*>(values.next_pre->data);check.pre_n=uint64_t(4)*in.count;
            check.compressed_rows=in.compressed_rows;
            if(values.selected){check.selected=static_cast<const int32_t*>(values.selected->data);check.selected_n=uint64_t(selected_width)*in.count;}
            if(values.candidates){check.candidates=static_cast<const int32_t*>(values.candidates->data);check.candidates_n=uint64_t(candidate_width)*in.count;}
            if(values.dspark_mean){check.mean=static_cast<const float*>(values.dspark_mean->data);check.mean_n=uint64_t(5120)*in.count;}
            pinned->validate(check);pinned->readback_ready();
            if(!defer_hidden_readback||(in.layer==39&&!export_hidden_after_head)){
                pinned->queue(false,*in.direct,0,81920,values.residual);pinned->queue(false,*in.direct,81920,16,values.next_pre);
            }
            // Reader IDs remain physically unchanged. Their graph output must
            // be a pure CONT copy of the exact input, in addition to GPU bounds.
            if(query_source(in.layer)&&selected_width)pinned->queue(false,*in.direct,81936,uint64_t(selected_width)*4,values.selected);
            if(in.layer==20&&candidate_width)pinned->queue(false,*in.direct,83984,uint64_t(candidate_width)*4,values.candidates);
#endif
        }else{
        pending.residual=download<float>(values.residual,size_t(5120)*4*in.count);
        pending.next_pre=download<float>(values.next_pre,size_t(4)*in.count);
        pending.selected=download<int32_t>(values.selected,size_t(512)*in.count);
        pending.candidates=download<int32_t>(values.candidates,size_t(2048)*in.count);
        pending.dspark_mean=download<float>(values.dspark_mean,size_t(5120)*in.count);
        require(pending.residual.size()==size_t(5120)*4*in.count&&pending.next_pre.size()==size_t(4)*in.count,
                "V4.1 completed hidden/pre carry dimensions changed");
        require(pending.selected.size()==size_t(selected_width)*in.count&&
                pending.candidates.size()==(in.layer>=20?size_t(candidate_width)*in.count:0)&&
                pending.dspark_mean.size()==(in.layer>=37?size_t(5120)*in.count:0),
                "V4.1 completed compact/capture carry dimensions changed");
        for(int32_t id:pending.selected)require(id>=-1&&id<int64_t(in.compressed_rows),"Invalid completed V4.1 selected-row ID");
        for(int32_t id:pending.candidates)require(id>=0&&id<int64_t((in.compressed_rows+7)/8),"Invalid completed V4.1 candidate-block ID");
        finite(pending.residual);finite(pending.next_pre);finite(pending.dspark_mean);
        }
        pending.selected_width=selected_width;pending.candidate_width=in.layer>=20?candidate_width:0;
        if(pooled&&p.next_pool_tail){
            require(values.next_pool_kv&&values.next_pool_score,"V4.1 odd pool tail was not produced");
            const auto next_kv=download<float>(values.next_pool_kv,512),next_score=download<float>(values.next_pool_score,512);
            require(next_kv.size()==512&&next_score.size()==512,"V4.1 completed odd pool tail dimensions changed");
            finite(next_kv);finite(next_score);
            upload(state.previous_pool_kv,next_kv);upload(state.previous_pool_score,next_score);
            ggml_backend_synchronize(backend);
        }
        if(device_hidden){
#ifdef GGML_USE_CUDA
            // Use the existing same-device copy helper/compute stream directly:
            // no backend copy fallback can stage this resident data in host RAM.
            auto stream=ggml_backend_cuda_get_stream(backend);
            if(!resident_bf16)ggml_backend_cuda_memcpy_d2d_async(input.residual->data,values.residual->data,
                ggml_nbytes(input.residual),stream);
            ggml_backend_cuda_memcpy_d2d_async(input.previous_pre->data,values.next_pre->data,
                ggml_nbytes(input.previous_pre),stream);
            ggml_backend_synchronize(backend);
            if(spilled){
                // The slot's new BF16 hidden and pre go back to this tile's own RAM
                // records (finished by pinned->finish below); IDs were mirrored above.
                const auto spill_started=clock_type::now();
                pinned->queue(false,*in.direct,0,40960,spill_slot);pinned->queue(false,*in.direct,81920,16,input.previous_pre);
                resident->spill_d2h_bytes+=uint64_t(in.count)*40976;resident->slot_logical=in.start;
                pinned->finish();
                llama_dsv41_layer_log("LLAMADECODE_PREFILL_SPILL_TILE layer=%u start=%u count=%u slot_row=%u h2d_ms=%.3f d2h_ms=%.3f\n",in.layer,in.start,in.count,bank_start,spill_in_ms,elapsed(spill_started));
            }
#endif
        }
        if(direct_input){
#ifdef GGML_USE_CUDA
            pinned->finish();
            if(prefetch_next){pinned->prefetch_measure(in.layer,in.start);asynchronous.live=false;}
            const bool mirrored=!defer_hidden_readback||(in.layer==39&&!export_hidden_after_head);
            if(defer_hidden_readback)llama_dsv41_layer_log("DS4PERF_RESIDENT_HIDDEN_COMPLETE layer=%u start=%u count=%u retained_device_bytes=%llu mirrored_host_bytes=%llu final_restore=%u\n",
                in.layer,in.start,in.count,(unsigned long long)(uint64_t(in.count)*81936),
                (unsigned long long)(mirrored?uint64_t(in.count)*81936:0),unsigned(in.layer==39&&mirrored));
            pending.direct_completion=llama_dsv41_carry_store::ram_completion(*in.direct,selected_unchanged,candidates_unchanged,mirrored);
            if(skip_tile_pin)llama_dsv41_layer_log("DS4PERF_TILE_PIN_NOT_NEEDED layer=%u start=%u count=%u no_RAM_bank_copies=1 resident_hidden_and_IDs=1 final_export_after_head=1 copy_attempts_forbidden=1\n",in.layer,in.start,in.count);
#endif
        }
        if(resident){
            resident->cursor+=in.count;
            if(resident->cursor==resident->total_tokens){
                if(resident_ids){
                    if(query_source(in.layer)){resident->selected_source=in.layer;resident->selected_width=selected_width;}
                    if(in.layer==20){resident->candidate_source=20;resident->candidate_width=candidate_width;}
                }
                resident->cursor=0;++resident->layer;
                if(resident->layer==40)resident->export_cursor=resident->yoco_first;
            }
        }
        if(resident_ids)llama_dsv41_layer_log("DS4PERF_RESIDENT_IDS_COMPLETE layer=%u start=%u count=%u selected_width=%u candidate_width=%u avoided_H2D_bytes=%llu source_copy_bytes=%llu validated=1 RAM_mirror=1 existing_buffer=1\n",
            in.layer,in.start,in.count,selected_width,candidate_width,
            (unsigned long long)(uint64_t(in.count)*4*((reuse?selected_width:0)+(candidates?candidate_width:0))),
            (unsigned long long)(uint64_t(in.count)*4*((query_source(in.layer)?selected_width:0)+(in.layer==20?candidate_width:0))));
        if(in.single_attention_append)llama_dsv41_layer_log("DS4PERF_UNALIGNED_APPEND_COMPLETE layer=%u start=%u absolute_start=%u count=%u attention_tiles=1 fixed_owner_unchanged=1 original_pool_tail_guard=1\n",in.layer,in.start,absolute_start,in.count);
        if(pair_ffn)llama_dsv41_layer_log("DS4PERF_PAIRED_FFN_COMPLETE layer=%u start=%u count=%u attention_tiles=%u expert_products=3 original_attention_max_rows=%u native_pair=%u completed_graph_and_copies=1\n",in.layer,in.start,in.count,attention_tiles,attention_rows,unsigned(native_pair));
        pending.readback_ms=elapsed(readback_start);
        const auto cleanup_started=clock_type::now();allocation.release();pending.cleanup_ms=elapsed(cleanup_started);
        if(capture_dspark)cache.dspark_completion.commit(capture_ticket);
        if(tile_timing_enabled())llama_dsv41_layer_log("LLAMADECODE_TILE_TIMING layer=%u tile=%u rows=%u host_prepare_ms=%.3f host_bind_ms=%.3f host_register_ms=%.3f host_upload_ms=%.3f host_launch_ms=%.3f host_sync_ms=%.3f gpu_ms=%.3f compute_ms=%.3f host_readback_ms=%.3f host_cleanup_ms=%.3f skip_pin=%u prefetch=%u\n",
            in.layer,in.start,in.count,pending.prepare_ms,tile_bind_ms,tile_register_ms,pending.upload_ms,tile_launch_ms,tile_sync_ms,double(tile_gpu_ms),pending.compute_ms,pending.readback_ms,pending.cleanup_ms,unsigned(skip_tile_pin),unsigned(prefetch_next));
        out=std::move(pending);cache.execution_poisoned=false;probe.pre_return_ms=elapsed(prepare_started);probe.complete=true;return true;
    }catch(const std::exception&failure){error=failure.what();return false;}
}
