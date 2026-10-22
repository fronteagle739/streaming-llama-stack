#pragma once

#include "llama-dsv41-attention.h"
#include "llama-dsv41-ffn.h"
#include "llama-dsv41-expert-profile.h"

// One layer and one explicitly admitted token tile. A layer-major executor restores compact
// carry inputs, stages this layer's weights, computes, then persists outputs.
// No full-context hidden-state tensor or host activation vector is built.
class llama_dsv41_layer_graph {
public:
    struct weights {
        llama_dsv41_attention_graph::weights attention;
        llama_dsv41_ffn_graph::weights ffn;
        ggml_tensor * attn_fn; ggml_tensor * attn_scale; ggml_tensor * attn_base; ggml_tensor * attn_norm;
        ggml_tensor * ffn_fn; ggml_tensor * ffn_scale; ggml_tensor * ffn_base; ggml_tensor * ffn_norm;
        ggml_tensor * engram_wkv; ggml_tensor * engram_q_norm; ggml_tensor * engram_k_norm;
    };
    struct inputs {
        ggml_tensor * residual; ggml_tensor * previous_pre;
        ggml_tensor * positions; ggml_tensor * raw_frequencies; ggml_tensor * compressed_frequencies;
        ggml_tensor * history_ids; ggml_tensor * raw_ids; ggml_tensor * raw_mask;
        ggml_tensor * visible; ggml_tensor * publication_ids; ggml_tensor * publication_positions;
        ggml_tensor * previous_selected; ggml_tensor * candidate_ids;
        ggml_tensor * engram_rows; ggml_tensor * text_mask; ggml_tensor * token_kinds;
        ggml_tensor *pair_history_ids=nullptr;
    };
    struct cache {
        ggml_tensor * raw; ggml_tensor * compressed; ggml_tensor * index;
        ggml_tensor * previous_pool_kv; ggml_tensor * previous_pool_score;
    };
    struct result {
        ggml_tensor * residual; ggml_tensor * next_pre;
        ggml_tensor * selected; ggml_tensor * selected_mask; ggml_tensor * candidates;
        ggml_tensor * next_pool_kv; ggml_tensor * next_pool_score; ggml_tensor * dspark_mean;
    };
    llama_dsv41_layer_graph(ggml_context * ctx,ggml_cgraph * graph,llama_dsv41_graph_ops & ops,
                            float swiglu_limit,float expert_scale);
    // compressed_rows is the fixed requested sweep frontier / layer ratio,
    // rather than a tiny per-query cap. Future rows remain causally masked.
    result build(uint32_t layer,const llama_dsv41_tile_plan & plan,uint32_t compressed_rows,
                 const weights & w,const inputs & input,const cache & state) const;
    result build(uint32_t,const llama_dsv41_tile_plan &,uint32_t,const weights &,const inputs &,const cache &,
                 const llama_dsv41_expert_profile::policy &) const;
    result build_pair(uint32_t,const llama_dsv41_tile_plan &,uint32_t,const weights &,const inputs &,const cache &,const llama_dsv41_expert_profile::policy &) const;
private:
    struct attention_part {
        llama_dsv41_graph_ops::sublayer ffn;
        ggml_tensor *selected,*mask,*candidates,*next_kv,*next_score,*capture;
        cache after;
    };
    attention_part build_attention(uint32_t,const llama_dsv41_tile_plan &,uint32_t,const weights &,const inputs &,const cache &,const llama_dsv41_expert_profile::policy &)const;
    ggml_context * ctx;
    ggml_cgraph * graph;
    llama_dsv41_graph_ops & ops;
    float swiglu_limit,expert_scale;
    ggml_tensor * publish(ggml_tensor * persistent,ggml_tensor * rows,ggml_tensor * ids) const;
    ggml_tensor * prefix(ggml_tensor * persistent,uint32_t rows,bool preserve_storage=false) const;
    ggml_tensor * keep(ggml_tensor * value) const;
};
