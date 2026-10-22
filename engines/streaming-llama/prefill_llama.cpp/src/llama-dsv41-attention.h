#pragma once

#include "llama-dsv41-graph.h"
#include "llama-dsv41-plan.h"

// V4.1-specific projection/compression graph, separate from old V4 helpers.
// The model loader owns canonical geometry validation and DirectStorage owns
// staging of these weight tensors. No tensor/table payload is allocated here.
class llama_dsv41_attention_graph {
public:
    struct weights {
        ggml_tensor * q_a; ggml_tensor * q_norm; ggml_tensor * q_b;
        ggml_tensor * kv; ggml_tensor * kv_norm;
        ggml_tensor * out_a; ggml_tensor * out_b;
        ggml_tensor * attn_sinks;
        ggml_tensor * compressor; ggml_tensor * gate; ggml_tensor * compressor_norm;
        ggml_tensor * index_k; ggml_tensor * index_norm;
        ggml_tensor * index_q; ggml_tensor * index_weights;
    };
    struct projected { ggml_tensor * qr; ggml_tensor * q; ggml_tensor * raw; };
    struct compressed {
        ggml_tensor * kv; ggml_tensor * index;
        ggml_tensor * next_pool_kv; ggml_tensor * next_pool_score;
    };
    struct index_query { ggml_tensor * q; ggml_tensor * weights; };

    llama_dsv41_attention_graph(ggml_context * ctx, llama_dsv41_graph_ops & ops);
    projected project(const weights & w, ggml_tensor * input, ggml_tensor * positions, ggml_tensor * frequencies) const;
    compressed compress(const weights & w, ggml_tensor * input, const llama_dsv41_tile_plan & plan,
                        ggml_tensor * previous_kv, ggml_tensor * previous_score,
                        ggml_tensor * publication_positions, ggml_tensor * frequencies) const;
    index_query project_index(const weights & w, ggml_tensor * qr, ggml_tensor * input,
                              ggml_tensor * positions, ggml_tensor * frequencies) const;
    ggml_tensor * finish(const weights & w, ggml_tensor * heads,
                         ggml_tensor * positions, ggml_tensor * frequencies) const;

private:
    ggml_context * ctx;
    llama_dsv41_graph_ops & ops;
    ggml_tensor * mm(ggml_tensor * weight, ggml_tensor * input, bool round) const;
    ggml_tensor * qat(ggml_tensor * input, ggml_dsv4_qat_mode mode) const;
};
