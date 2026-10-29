#pragma once

#include "ggml.h"
#include <functional>

// V4.1 subgraphs deliberately separate producing a mix from consuming the
// previous sublayer's pre coefficients. They do not construct a V4 graph or
// admit inference without the matching shared-cache attention and handoff.
class llama_dsv41_graph_ops {
public:
    using observer = std::function<void(ggml_tensor *, const char *)>;
    struct mix { ggml_tensor * pre; ggml_tensor * post; ggml_tensor * comb; };
    struct sublayer { ggml_tensor * residual; mix next; ggml_tensor * input; };
    struct selection { ggml_tensor * rows; ggml_tensor * mask; ggml_tensor * candidates; };
    struct seed { ggml_tensor * residual; ggml_tensor * pre; };

    llama_dsv41_graph_ops(ggml_context * ctx, float norm_eps, float hc_eps,
                         uint32_t sinkhorn_iters, observer tag = {});
    void merge_attention_selected_rows(bool enabled,uint32_t ratio){merge_selected_rows=enabled;merge_ratio=ratio;}
    void use_hc_projection_f16_gemm(bool enabled){hc_projection_f16_gemm=enabled;}
    void use_hc_bf16_epilogue(bool enabled){hc_bf16_epilogue=enabled;}
    // OPUS22 round 8: the attention sublayer's HC output stays BF16 (exact: the
    // BF16 epilogue already rounds it). Consumers accept BF16 residuals.
    void use_hc_bf16_carry(bool enabled){hc_bf16_carry=enabled;}
    bool hc_bf16_carry_enabled()const{return hc_bf16_carry&&hc_bf16_epilogue;}

    ggml_tensor * bf16(ggml_tensor * input) const;
    ggml_tensor * norm(ggml_tensor * input, ggml_tensor * weight) const;
    // Dequantized text embeddings or host-supplied vision rows [width,tokens].
    // Native input repeats their actual values, without an extra BF16 store.
    seed seed_embeddings(ggml_tensor * embeddings) const;
    // Last FFN pre, rather than a learned V4 head mix. Logits remain F32.
    ggml_tensor * logits(ggml_tensor * residual, ggml_tensor * final_pre,
                         ggml_tensor * output_norm, ggml_tensor * output_weight) const;
    mix mixes(ggml_tensor * residual, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base) const;
    ggml_tensor * collapse(ggml_tensor * residual, ggml_tensor * previous_pre) const;
    // Includes the block's own BF16 boundary before applying post/comb.
    ggml_tensor * expand(ggml_tensor * block, ggml_tensor * residual, const mix & current, bool bf16_output = false) const;
    ggml_tensor * mean_for_capture(ggml_tensor * residual) const;
    ggml_tensor * rope(ggml_tensor * input, ggml_tensor * positions,
                      ggml_tensor * native_frequencies, bool inverse = false) const;
    ggml_tensor * rope_bf16_input(ggml_tensor * input, ggml_tensor * positions,
                      ggml_tensor * native_frequencies, bool inverse = false) const;
    // At most4096 admitted queries /128 raw +512 selected rows per query. Tables remain
      // F32 and only selected rows are gathered. Bounded Q/gathered KV round
      // through native F16; probabilities/sinks remain F32. No context-sized
      // dense mask or F16 cache conversion. Returned heads are BF16-rounded and
    // still need their native inverse RoPE before grouped output projection.
    ggml_tensor * attention_heads(
            ggml_tensor * query, ggml_tensor * raw_table, ggml_tensor * raw_ids,
            ggml_tensor * raw_mask, ggml_tensor * compressed_table,
            ggml_tensor * selected_ids, ggml_tensor * selected_mask, ggml_tensor * sinks,
            bool shared_sparse = false, bool shared_flash_attention = false,
            bool token_tile_attention = false, uint32_t start = 0,
            bool attention_kv_f16_table = false) const;
    // Gather the chronological previous <=127 rows before any ring write.
    // current is [512,tile_tokens], already normalized/rotated/FP8-rounded.
    ggml_tensor * raw_snapshot(ggml_tensor * ring, ggml_tensor * current, ggml_tensor * history_ids) const;
    ggml_tensor * raw_commit(ggml_tensor * ring, ggml_tensor * current,
                            ggml_tensor * attention_complete, uint32_t start) const;
    // Layer20 publishes top2048 blocks but its own top512 remains GLOBAL.
    // Later query sources consume those block IDs from layer-major carry.
    selection index_select(ggml_tensor * q, ggml_tensor * weights, ggml_tensor * keys,
                           ggml_tensor * visible, uint32_t source_layer,
                           ggml_tensor * prior_candidates = nullptr,
                           ggml_cgraph * stream_graph = nullptr) const;
    ggml_tensor * selection_mask(ggml_tensor * selected,ggml_tensor * visible,
                                 ggml_tensor * candidates,int64_t compressed_rows) const;

    // Attention consumes previous FFN pre; FFN consumes current attention pre.
    // The mix computed here belongs to the following sublayer.
    sublayer prepare(ggml_tensor * residual, ggml_tensor * previous_pre,
                     ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
                     ggml_tensor * norm_weight) const;
    ggml_tensor * complete(ggml_tensor * block, const sublayer & layer) const;
    // Attention sublayer: BF16 output when hc_bf16_carry_enabled().
    ggml_tensor * complete_attention(ggml_tensor * block, const sublayer & layer) const;

    ggml_tensor * engram(ggml_tensor * residual, ggml_tensor * gathered_rows,
                         ggml_tensor * wkv, ggml_tensor * q_weight,
                         ggml_tensor * k_weight, ggml_tensor * text_mask) const;

private:
    ggml_context * ctx;
    float norm_eps, hc_eps;
    uint32_t sinkhorn_iters;
    bool merge_selected_rows=false;
    bool hc_projection_f16_gemm=false,hc_bf16_epilogue=false,hc_bf16_carry=false;
    ggml_tensor * widen(ggml_tensor * residual) const;
    uint32_t merge_ratio=0;
    observer observe;
    ggml_tensor * tagged(ggml_tensor * tensor, const char * name) const;
    ggml_tensor * weighted_sum(ggml_tensor * residual, ggml_tensor * weights) const;
    ggml_tensor * sinkhorn(ggml_tensor * comb) const;
    ggml_tensor * gather_per_query(ggml_tensor * table, ggml_tensor * ids) const;
    ggml_tensor * candidate_mask(ggml_tensor * candidates, int64_t blocks) const;
};
