#pragma once

#include "llama-dsv41-cache.h"
#include "ggml-alloc.h"
#include "llama-dsv41-layer.h"
#include "llama-dsv41-carry.h"

#include <string>
#include <memory>
#include <optional>
#include <vector>

// A bounded production tile executor, not a model loader or an inference gate.
// Its owner stages one layer's weights using the existing physical streaming
// fetcher, then visits this layer's token tiles before releasing those weights.
// Host vectors obey the caller's admitted tile bound (default32). Full-context
// carry is a separately admitted RAM bank or an exclusive file.
struct llama_dsv41_fixed_storage {
    uint32_t tokens=0,tile=0,context=0;
    bool bf16=false,ids=false,metadata=false,ready=false,tile_registration=false,pageable_copies=false;
    ggml_backend_t backend=nullptr;
    std::shared_ptr<ggml_gallocr> graph;
    std::shared_ptr<ggml_backend_buffer> hidden;
    std::shared_ptr<llama_dsv41_cache_storage> cache;
    std::shared_ptr<llama_dsv41_carry_store::ram_backing> ram;
    std::shared_ptr<void> ram_registration;
    uint64_t graph_bytes=0,hidden_bytes=0,cache_bytes=0;
    uint64_t backend_temporary_reserve_bytes=1024ull*1024*1024;
    // The same fixed owners can serve an explicitly selected unpaired graph.
    // This changes allocation lifetime only, never expert/attention arithmetic.
    bool unpaired_resident=false;
    // DS4COMBO no-warm-up start: the first real request's open() allocates the
    // fixed RAM bank and GPU reservation itself (same sizes, same order:
    // provider heaps -> fixed reserve -> first stage lease) instead of a
    // throwaway one-token pass that streams every weight once (~20 s).
    bool direct_open=false;
};

class llama_dsv41_tile_executor {
public:
    struct request {
        uint32_t layer=0,start=0,count=0,compressed_rows=0;
        // Carry leases/device banks use start; cache/RoPE positions add this.
        uint32_t position_offset=0;
        bool single_attention_append=false;
        std::vector<float> residual,previous_pre,engram_rows;
        std::vector<int32_t> selected,candidates,token_kinds;
        std::vector<int8_t> text_mask;
        std::optional<llama_dsv41_carry_store::ram_lease> direct;
        std::optional<llama_dsv41_carry_store::ram_lookahead> lookahead;
    };
    struct result {
        std::vector<float> residual,next_pre,dspark_mean;
        std::vector<int32_t> selected,candidates;
        uint64_t scratch_bytes=0;
        uint32_t graph_nodes=0,selected_width=0,candidate_width=0;
        double upload_ms=0,compute_ms=0,readback_ms=0,prepare_ms=0,cleanup_ms=0,weight_cache_ms=0;
        std::optional<llama_dsv41_carry_store::ram_completion> direct_completion;
    };
    struct pinned_span{void *base=nullptr;uint64_t bytes=0,admitted_device_bytes=0;};
    llama_dsv41_tile_executor(ggml_backend_t backend,llama_dsv41_cache_storage &cache,
        uint64_t admitted_graph_bytes,float norm_eps,float hc_eps,uint32_t sinkhorn_iters,
        float swiglu_limit,float expert_scale,uint32_t admitted_tile_tokens=32,
        uint32_t resident_hidden_tokens=0,uint64_t admitted_hidden_bytes=0,const pinned_span *direct=nullptr,bool reuse_graph=false,bool reuse_layer_graph=false,uint64_t admitted_cast_bytes=0,bool audit_casts=false,uint64_t admitted_index_bytes=0,uint64_t admitted_prefetch_bytes=0,bool coalesce_dense=false,bool defer_hidden_readback=false,bool merge_attention_selected_rows=false,bool hc_projection_f16_gemm=false,bool dense_mxfp4=false,bool resident_bf16=false,bool export_hidden_after_head=false,bool async_inputs=false,bool resident_ids=false,bool resident_metadata=false,bool pair_ffn=false,std::shared_ptr<llama_dsv41_fixed_storage> fixed={});
    ~llama_dsv41_tile_executor();
    static uint64_t resident_hidden_bytes(uint32_t tokens);
    // Metadata-only query of the exact existing backend layout, including IDs,
    // metadata and backend alignment. Does not allocate a device buffer.
    static uint64_t resident_layout_bytes(ggml_backend_t,uint32_t tokens,uint32_t tile,bool bf16,bool ids,bool metadata);
    static std::shared_ptr<void> register_fixed_ram(ggml_backend_t,const std::shared_ptr<llama_dsv41_carry_store::ram_backing>&);
    static std::shared_ptr<ggml_backend_buffer> reserve_hidden(ggml_backend_t,uint32_t tokens,uint32_t tile,bool bf16,bool ids,bool metadata,uint64_t admission);
    bool seed_ram(ggml_tensor *embedding,const std::vector<int32_t> &ids,
        const llama_dsv41_carry_store::ram_lease &lease,
        std::optional<llama_dsv41_carry_store::ram_completion> &completion,std::string &error);

    bool resident_head_row(llama_dsv41_carry_store::tile &out,std::string &error);
    bool export_hidden_tile(const llama_dsv41_carry_store::ram_lease &lease,
        std::optional<llama_dsv41_carry_store::ram_completion> &completion,std::string &error);

    // On any compute/readback failure the cache is poisoned until clear(). A
    // partial layer must never be published as a native completed frontier.
    // Invalid requests rejected before computation leave existing state intact.
    bool run(const request &input,const llama_dsv41_layer_graph::weights &weights,
             result &output,std::string &error);
    bool run(const request &,const llama_dsv41_layer_graph::weights &,result &,std::string &,
             const llama_dsv41_expert_profile::policy &);
    void clear();
private:
    ggml_backend_t backend;
    llama_dsv41_cache_storage &cache;
    uint64_t admitted_graph_bytes;
    float norm_eps,hc_eps,swiglu_limit,expert_scale;
    uint32_t sinkhorn_iters;
    uint32_t admitted_tile_tokens;
    bool index_cublas=false;
    bool resident_bf16=false,export_hidden_after_head=false,async_inputs=false,resident_ids=false,resident_metadata=false;
    bool defer_hidden_readback=false,merge_attention_selected_rows=false,hc_projection_f16_gemm=false,dense_mxfp4=false,pair_ffn=false;
    std::shared_ptr<llama_dsv41_fixed_storage> fixed;
    struct resident_storage;
    std::unique_ptr<resident_storage> resident;
    struct pinned_storage;
    std::unique_ptr<pinned_storage> pinned;
    struct graph_storage;
    std::unique_ptr<graph_storage> workspace;
};
