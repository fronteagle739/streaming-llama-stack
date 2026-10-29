#pragma once

#include "llama-dsv41-graph.h"

// The original native Q4_K routed path computes F32 experts; its shared expert
// separately rounds gate/up/mid/down to BF16. Preserve these distinct paths.
class llama_dsv41_ffn_graph {
public:
    struct weights {
        ggml_tensor * router; ggml_tensor * text_bias; ggml_tensor * image_bias;
        ggml_tensor * up; ggml_tensor * gate; ggml_tensor * down;
        ggml_tensor * shared_up; ggml_tensor * shared_gate; ggml_tensor * shared_down;
        // LLAMADECODE B2: when set (an I32 [6, tokens] leaf), the expert
        // products read their ids from it. The graph first copies the router
        // ids into it ("dsv41_expert_slot_copy"); the decode hook then
        // rewrites it with expert-cache slots. Route weights keep router ids.
        ggml_tensor * expert_slots = nullptr;
        // Set only by the metadata-validated DirectStorage provider.
        bool native_fp4_layout = false;
    };
    struct result { ggml_tensor * output; ggml_tensor * selected; ggml_tensor * route_weights; };
    llama_dsv41_ffn_graph(ggml_context * ctx,llama_dsv41_graph_ops & ops,float limit,float route_scale);
    // token_kinds: I32[tokens], 0=text,1=image, including mixed batches. Select
    // exact bias rows rather than inferring the entire batch from embd!=null.
    result build(const weights & w,ggml_tensor * input,ggml_tensor * token_kinds) const;
    // Explicit DS4.1-only profile; caller must preflight the complete graph on CUDA.
    // This source-only API is not selected by the existing layer factory.
    result build_native_half(const weights & w,ggml_tensor * input,ggml_tensor * token_kinds) const;
    result build_routed_pair(const weights &,ggml_tensor *,ggml_tensor *,bool native_half=false) const;
    ggml_tensor *build_shared(const weights &,ggml_tensor *) const;
private:
    result build_impl(const weights &,ggml_tensor *,ggml_tensor *,bool) const;
    result routed_impl(const weights &,ggml_tensor *,ggml_tensor *,bool,bool native_router_partitions=false) const;
    ggml_context * ctx;
    llama_dsv41_graph_ops & ops;
    float limit,route_scale;
    ggml_tensor * mm(ggml_tensor * w,ggml_tensor * input,bool round) const;
    ggml_tensor * swiglu(ggml_tensor * gate,ggml_tensor * up) const;
};
