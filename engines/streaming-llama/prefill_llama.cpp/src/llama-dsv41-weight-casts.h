#pragma once
#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-backend.h"
#include "../ggml/include/ggml-dsv41-dense-half.h"
#include "../ggml/include/ggml-dsv41-native-fp4.h"
#include "llama-dsv41-expert-fp4-pack.h"
#ifdef GGML_USE_CUDA
#include "../ggml/include/ggml-cuda.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// Cache only the bytes produced by the existing Q8->F32->F16 graph casts.
// No conversion, multiplication or accumulation arithmetic is replaced.
class llama_dsv41_weight_cast_cache {
    struct key {
        const void *data=nullptr;
        std::array<int64_t,GGML_MAX_DIMS> ne{};
        std::array<size_t,GGML_MAX_DIMS> nb{};
        bool operator==(const key &o)const{return data==o.data&&ne==o.ne&&nb==o.nb;}
    };
    struct entry {key source;ggml_tensor *half=nullptr;uint64_t offset=0;ggml_tensor original{};};
    struct pending_copy {ggml_tensor *source,*destination;};
    ggml_backend_t backend;
    uint64_t admitted;
    bool audit,coalesce,fp4_weights,ready=false;
    uint32_t layer=UINT32_MAX;
    ggml_context *metadata=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    std::vector<entry> entries;
    std::vector<pending_copy> pending;
    std::vector<std::pair<const void*,ggml_tensor*>> packed_experts;
    uint64_t expert_scratch_offset=0;
    static void need(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
    static uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"Cast-cache byte overflow");return a+b;}
    static key source_key(const ggml_tensor *half){
        need(half&&half->type==GGML_TYPE_F16&&half->op==GGML_OP_CPY&&half->src[1]==half&&
             half->src[0]&&half->src[0]->type==GGML_TYPE_F32&&half->src[0]->op==GGML_OP_CPY&&
             half->src[0]->src[0]&&half->src[0]->src[0]->type==GGML_TYPE_Q8_0&&ggml_is_contiguous(half),
             "Unexpected native Q8 cast chain");
        const auto *part=half->src[0]->src[0];need(part->data,"Unbound Q8 weight part");key result;result.data=part->data;
        std::copy_n(part->ne,GGML_MAX_DIMS,result.ne.begin());std::copy_n(part->nb,GGML_MAX_DIMS,result.nb.begin());return result;
    }
    entry *find(const key &k){for(auto &e:entries)if(e.source==k)return &e;return nullptr;}
    static const ggml_tensor *weight_root(const ggml_tensor *t){
        for(unsigned depth=0;t&&depth<32;++depth){
            if(t->view_src)t=t->view_src;
            else if(t->op==GGML_OP_CPY&&t->src[0])t=t->src[0];
            else return t;
        }
        return nullptr;
    }
    uint32_t join_cached_rows(ggml_context *ctx,ggml_cgraph *graph,
            std::unordered_set<ggml_tensor*> &removed,uint64_t *temporary){
        uint32_t joined=0,products=0;
        const int n=ggml_graph_n_nodes(graph);
        for(int i=n-1;i>=0;--i){
            auto *root=ggml_graph_node(graph,i);
            if(removed.count(root)||root->op!=GGML_OP_CONCAT||root->op_params[0]!=0)continue;
            std::vector<ggml_tensor*> cluster,leaves,stack{root};bool valid=true;
            while(!stack.empty()&&valid){auto *t=stack.back();stack.pop_back();
                if(removed.count(t)){valid=false;break;}cluster.push_back(t);
                if(t->op==GGML_OP_CONCAT&&t->op_params[0]==0){stack.push_back(t->src[1]);stack.push_back(t->src[0]);}
                else if(ggml_dsv41_dense_half_requested(t)&&t->src[0]->op==GGML_OP_VIEW&&
                        t->src[0]->buffer==buffer&&ggml_is_contiguous(t->src[0]))leaves.push_back(t);
                else valid=false;
            }
            if(!valid||leaves.size()<2)continue;
            std::unordered_set<ggml_tensor*> inside(cluster.begin(),cluster.end());
            // A removed result may have no side effect or outside consumer.
            for(auto *t:cluster){if(t==root)continue;
                if(t->flags&GGML_TENSOR_FLAG_OUTPUT){valid=false;break;}
                for(int j=0;j<n&&valid;++j){auto *consumer=ggml_graph_node(graph,j);
                    if(inside.count(consumer))continue;
                    if(consumer->view_src==t)valid=false;
                    for(auto *src:consumer->src)if(src==t)valid=false;
                }
            }
            auto *first=leaves.front()->src[0];auto *rhs=leaves.front()->src[1];
            const auto *origin=weight_root(first);
            if(!origin)continue;
            uintptr_t next=reinterpret_cast<uintptr_t>(first->data);int64_t rows=0;
            for(auto *t:leaves){auto *w=t->src[0];const auto *source=weight_root(w);
                if(t->src[1]!=rhs||!source||std::strcmp(source->name,origin->name)||
                    w->ne[0]!=first->ne[0]||w->ne[2]!=1||w->ne[3]!=1||
                    reinterpret_cast<uintptr_t>(w->data)!=next){valid=false;break;}
                next+=ggml_nbytes(w);rows+=w->ne[1];
            }
            if(!valid||rows!=root->ne[0]||rows>32768)continue;
            auto *whole=ggml_new_tensor_2d(ctx,first->type,first->ne[0],rows);
            ggml_set_name(whole,origin->name);
            const auto begin=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
            need(reinterpret_cast<uintptr_t>(first->data)>=begin&&next-begin<=ggml_backend_buffer_get_size(buffer),
                "Coalesced projection escapes existing cast bank");
            need(ggml_backend_tensor_alloc(buffer,whole,first->data)==GGML_STATUS_SUCCESS,
                "Coalesced cached weight alias binding failed");
            auto *product=ggml_mul_mat(ctx,whole,rhs);ggml_mul_mat_set_prec(product,fp4_weights?GGML_PREC_DEFAULT:GGML_PREC_F32);
            need(fp4_weights?first->type==GGML_TYPE_MXFP4:ggml_dsv41_dense_half_tag(product),"Coalesced dense projection violates explicit shared-path contract");
            need(std::memcmp(root->ne,product->ne,sizeof(root->ne))==0&&
                 std::memcmp(root->nb,product->nb,sizeof(root->nb))==0,"Coalesced dense output layout changed");
            const uint64_t rhs_bytes=uint64_t(ggml_nelements(rhs))*sizeof(ggml_fp16_t);
            need(rhs_bytes<=128ull*1024*1024,"Coalesced dense RHS exceeds bounded cuBLAS temporary");
            *temporary=std::max(*temporary,rhs_bytes);
            root->op=product->op;std::memcpy(root->op_params,product->op_params,sizeof(root->op_params));
            std::memcpy(root->src,product->src,sizeof(root->src));root->view_src=nullptr;root->view_offs=0;
            for(auto *t:cluster)if(t!=root)removed.insert(t);
            ++joined;products+=uint32_t(leaves.size());
        }
        std::fprintf(stderr,"[dsv41-dense-coalesce] layer=%u joined_matrices=%u old_products=%u new_products=%u new_device_buffers=0\n",
            layer,joined,products,joined);return joined;
    }
public:
    llama_dsv41_weight_cast_cache(ggml_backend_t b,uint64_t bytes,bool audit,bool coalesce=false,bool fp4_weights=false):backend(b),admitted(bytes),audit(audit),coalesce(coalesce),fp4_weights(fp4_weights){
        need(!fp4_weights||!audit,"FP4 weights cannot use the F16 bit-copy audit; use the explicit FP4 layout/operand fixtures");
#ifdef GGML_USE_CUDA
        need(backend&&ggml_backend_is_cuda(backend)&&admitted,"Weight cast cache requires explicitly admitted CUDA bytes");
#else
        need(false,"Weight cast cache requires CUDA");
#endif
    }
    ~llama_dsv41_weight_cast_cache(){ggml_backend_synchronize(backend);if(buffer)ggml_backend_buffer_free(buffer);if(metadata)ggml_free(metadata);}
    void prepare_expert_fp4(ggml_cgraph *graph,uint32_t current_layer,uint32_t tokens){
        need(fp4_weights&&ready&&buffer&&metadata&&layer==current_layer&&tokens>4&&tokens<=4096,"Expert FP4 requires ready dense FP4 scratch and4K-or-smaller graph");
        std::vector<ggml_tensor*> products;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto *node=ggml_graph_node(graph,i);if(node->op==GGML_OP_MUL_MAT_ID)products.push_back(node);}
        need(products.size()==3,"Expert FP4 requires exactly gate/up/down products");
        uint64_t total_chunks=0,source_bytes=0,destination_bytes=0;const auto started=std::chrono::steady_clock::now();
        const bool bootstrap=packed_experts.empty();
        for(auto *node:products){auto *source=node->src[0];
            need(source&&source->type==GGML_TYPE_Q4_K&&source->ne[2]==384&&source->ne[3]==1&&
                ((source->ne[0]==5120&&source->ne[1]==2304)||(source->ne[0]==2304&&source->ne[1]==5120))&&node->op_params[2]==0,
                "Expert FP4 source/tag is outside the explicit original Q4_K triplet");
            auto found=std::find_if(packed_experts.begin(),packed_experts.end(),[&](const auto&e){return e.first==source->data;});
            ggml_tensor *packed=nullptr;
            if(found!=packed_experts.end()){packed=found->second;need(!std::memcmp(packed->ne,source->ne,sizeof(source->ne)),"Expert FP4 source geometry changed within a layer");}
            else{
                need(bootstrap,"Incomplete expert FP4 triplet cannot accept a new source");
                uint64_t chunks=0;packed=llama_dsv41_pack_q4k_inplace(metadata,backend,source,buffer,expert_scratch_offset,64ull*1024*1024,&chunks);
                packed_experts.push_back({source->data,packed});total_chunks+=chunks;source_bytes+=ggml_nbytes(source);destination_bytes+=ggml_nbytes(packed);
            }
            node->src[0]=packed;node->op_params[0]=GGML_PREC_DEFAULT;node->op_params[2]=0x46345034;node->op_params[3]=1;
            need(ggml_dsv41_native_fp4_valid(node),"Expert FP4 product did not satisfy the existing native shared MMA contract");
        }
        need(packed_experts.size()==3,"Expert FP4 requires three distinct original weight tensors");
        if(bootstrap)std::fprintf(stderr,"DS4PERF_EXPERT_MXFP4_WEIGHT_PACK layer=%u parts=3 chunks=%llu source_bytes=%llu packed_bytes=%llu conversion_ms=%.6f existing_cast_scratch=67108864 inplace=1 canonical_file_unchanged=1\n",current_layer,(unsigned long long)total_chunks,(unsigned long long)source_bytes,(unsigned long long)destination_bytes,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
    }
    ggml_cgraph *prepare(ggml_context *ctx,ggml_cgraph *graph,uint32_t current_layer,uint64_t *temporary=nullptr){
        need(!coalesce||temporary,"Coalescing requires explicit backend temporary accounting");
        if(layer!=current_layer){
            ggml_backend_synchronize(backend);entries.clear();pending.clear();packed_experts.clear();ready=false;layer=current_layer;
            if(metadata)ggml_free(metadata);metadata=ggml_init({128*1024,nullptr,true});need(metadata,"Cast-cache metadata allocation failed");
            if(buffer)ggml_backend_buffer_reset(buffer);
        }
        std::vector<ggml_tensor*> nodes;
        const int original_nodes=ggml_graph_n_nodes(graph);
        for(int i=0;i<original_nodes;++i){auto *t=ggml_graph_node(graph,i);
            if(std::strcmp(t->name,"dsv41_native_q8_weight_f16")==0)nodes.push_back(t);}
        if(nodes.empty())return graph;
        need(nodes.size()<=128,"Cast-cache part limit exceeded");
        if(!ready){
            need(entries.empty()&&pending.empty(),"Incomplete cast-cache bootstrap cannot be reused");
            const auto buft=ggml_backend_get_default_buffer_type(backend);const auto alignment=ggml_backend_buft_get_alignment(buft);
            need(alignment&&alignment<=65536,"Invalid cast-cache backend alignment");uint64_t required=0;
            for(auto *t:nodes){const auto k=source_key(t);auto *e=find(k);
                if(!e){required=add(required,(alignment-required%alignment)%alignment);
                    auto *half=ggml_new_tensor(metadata,fp4_weights?GGML_TYPE_MXFP4:GGML_TYPE_F16,GGML_MAX_DIMS,t->ne);
                    if(coalesce){const auto *root=weight_root(t);need(root,"Missing original cached projection identity");ggml_set_name(half,root->name);}
                    need(half&&(fp4_weights||std::memcmp(half->nb,t->nb,sizeof(t->nb))==0),"Cast-cache half layout changed");
                    entries.push_back({k,half,required,*t->src[0]->src[0]});e=&entries.back();
                    const auto *source_root=weight_root(t);need(source_root&&source_root->buffer,"Cached weight source lost its original buffer owner");e->original.buffer=source_root->buffer;
                    e->original.op=GGML_OP_NONE;e->original.view_src=nullptr;std::fill(std::begin(e->original.src),std::end(e->original.src),nullptr);
                    required=add(required,fp4_weights?ggml_nbytes(half):ggml_backend_buft_get_alloc_size(buft,half));}
                if(!fp4_weights){ggml_set_output(t);pending.push_back({t,e->half});}
            }
            if(coalesce){
                // Adjacent source rows become adjacent within the SAME cache bank.
                std::sort(entries.begin(),entries.end(),[](const entry &a,const entry &b){return reinterpret_cast<uintptr_t>(a.source.data)<reinterpret_cast<uintptr_t>(b.source.data);});
                required=0;for(auto &e:entries){required=add(required,(alignment-required%alignment)%alignment);e.offset=required;
                    required=add(required,fp4_weights?ggml_nbytes(e.half):ggml_backend_buft_get_alloc_size(buft,e.half));}
            }
            // Same existing cache allocation. Reserve a zero padding guard and
            // bounded decode workspace after its packed weights; no new buffer.
            const uint64_t scratch_offset=fp4_weights?add(add(required,65536),65535)&~uint64_t(65535):0;
            if(fp4_weights){required=add(scratch_offset,64ull*1024*1024);expert_scratch_offset=scratch_offset;}
            need(required<=admitted,"Cast-cache weights exceed separate device admission");
            if(!buffer||ggml_backend_buffer_get_size(buffer)<required){
                if(buffer){ggml_backend_buffer_free(buffer);buffer=nullptr;}
                buffer=ggml_backend_buft_alloc_buffer(buft,size_t(required));need(buffer,"Cast-cache device allocation failed");
            }
            need(ggml_backend_buffer_get_size(buffer)<=admitted,"Cast-cache actual device buffer exceeds admission");
            ggml_backend_buffer_set_usage(buffer,fp4_weights?GGML_BACKEND_BUFFER_USAGE_WEIGHTS:GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            for(auto &e:entries)need(ggml_backend_tensor_alloc(buffer,e.half,static_cast<char*>(ggml_backend_buffer_get_base(buffer))+e.offset)==GGML_STATUS_SUCCESS,
                "Cast-cache half tensor binding failed");
            std::fprintf(stderr,"[dsv41-cast-cache] layer=%u bootstrap_parts=%zu physical_bytes=%llu audit=%u",layer,entries.size(),
                (unsigned long long)ggml_backend_buffer_get_size(buffer),unsigned(audit));std::fputc(10,stderr);
            if(!fp4_weights)return graph;
#ifdef GGML_USE_CUDA
            const auto started=std::chrono::steady_clock::now();auto *stream=ggml_backend_cuda_get_stream(backend);
            // Initialize padding before any pack. Weight views may share the
            // usual quantized tensor tail allocation with adjacent packed rows.
            ggml_backend_buffer_clear(buffer,0);const auto before=ggml_backend_cuda_dsv41_mxfp4_weight_packs_queued();
            auto *scratch=static_cast<char*>(ggml_backend_buffer_get_base(buffer))+scratch_offset;
            uint64_t packed_bytes=0;
            for(auto &e:entries){const int status=ggml_backend_cuda_dsv41_pack_weights_mxfp4(&e.original,e.half,scratch,64ull*1024*1024,stream);
                if(status)std::fprintf(stderr,"DS4PERF_MXFP4_PACK_REFUSED status=%d type=%d inner=%lld rows=%lld source_owner=%u destination_owner=%u\n",status,int(e.original.type),(long long)e.original.ne[0],(long long)ggml_nrows(&e.original),unsigned(e.original.buffer!=nullptr),unsigned(e.half->buffer!=nullptr));
                need(status==0,"Bounded shared FP4 weight conversion refused");packed_bytes+=ggml_nbytes(e.half);}
            ggml_backend_synchronize(backend);need(ggml_backend_cuda_dsv41_mxfp4_weight_packs_queued()==before+entries.size(),"Actual FP4 conversion witness differs");
            std::fprintf(stderr,"DS4PERF_MXFP4_WEIGHTS layer=%u parts=%zu packed_bytes=%llu scratch_bytes=67108864 conversion_ms=%.6f actual_queued=%zu\n",
                layer,entries.size(),(unsigned long long)packed_bytes,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count(),entries.size());ready=true;
#else
            need(false,"FP4 weights require CUDA");
#endif
        }
        std::unordered_set<ggml_tensor*> removed;
        for(auto *t:nodes){auto *e=find(source_key(t));need(e,"Q8 weight binding changed within a layer");
            auto *intermediate=t->src[0];size_t consumers=0;
            for(int i=0;i<original_nodes;++i){auto *n=ggml_graph_node(graph,i);if(n==intermediate)continue;
                for(auto *src:n->src)consumers+=src==intermediate;consumers+=n->view_src==intermediate;}
            need(consumers==1&&!(intermediate->flags&GGML_TENSOR_FLAG_OUTPUT),"Q8 F32 intermediate has an additional live consumer");
            removed.insert(intermediate);t->op=GGML_OP_VIEW;std::memset(t->op_params,0,sizeof(t->op_params));
            if(fp4_weights){t->type=e->half->type;std::copy_n(e->half->nb,GGML_MAX_DIMS,t->nb);}
            std::fill(std::begin(t->src),std::end(t->src),nullptr);t->src[0]=e->half;t->view_src=e->half;t->view_offs=0;
            t->data=e->half->data;t->buffer=e->half->buffer;t->extra=e->half->extra;
        }
        if(coalesce)join_cached_rows(ctx,graph,removed,temporary);
        if(fp4_weights)for(int i=0;i<original_nodes;++i){auto*n=ggml_graph_node(graph,i);
            if(!removed.count(n)&&ggml_dsv41_dense_half_requested(n)&&n->src[0]->type==GGML_TYPE_MXFP4){
                need(n->src[1]->type==GGML_TYPE_F32&&n->src[1]->ne[1]>8,"FP4 dense requires the existing prefill MMQ geometry");
                n->op_params[2]=0;n->op_params[3]=0;ggml_mul_mat_set_prec(n,GGML_PREC_DEFAULT);}}
        auto *replacement=ggml_new_graph_custom(ctx,ggml_graph_size(graph),false);
        for(int i=0;i<original_nodes;++i){auto *n=ggml_graph_node(graph,i);if(!removed.count(n))ggml_build_forward_expand(replacement,n);}
        std::fprintf(stderr,"[dsv41-cast-cache] layer=%u reused_parts=%zu removed_F32_casts=%zu",layer,nodes.size(),removed.size());std::fputc(10,stderr);
        return replacement;
    }
    void complete(){
        if(pending.empty())return;
#ifdef GGML_USE_CUDA
        auto *stream=ggml_backend_cuda_get_stream(backend);
        for(const auto &p:pending){need(p.source->data&&p.destination->data&&ggml_nbytes(p.source)==ggml_nbytes(p.destination),"Completed cast-cache tensor geometry changed");
            ggml_backend_cuda_memcpy_d2d_async(p.destination->data,p.source->data,ggml_nbytes(p.source),stream);}
        ggml_backend_synchronize(backend);
        if(audit)for(const auto &p:pending){const auto bytes=ggml_nbytes(p.source);need(bytes<=16ull*1024*1024,"Cast-cache audit exceeds32MiB host workspace");
            std::vector<uint8_t> a(bytes),b(bytes);ggml_backend_tensor_get(p.source,a.data(),0,bytes);ggml_backend_tensor_get(p.destination,b.data(),0,bytes);
            need(a==b,"Actual GPU cached half-weight bytes differ from original graph cast");}
        std::fprintf(stderr,"[dsv41-cast-cache] layer=%u completed_copy_parts=%zu actual_half_bytes_verified=%u",layer,pending.size(),unsigned(audit));std::fputc(10,stderr);
        ready=true;pending.clear();
#else
        need(false,"Weight cast cache requires CUDA");
#endif
    }
};
