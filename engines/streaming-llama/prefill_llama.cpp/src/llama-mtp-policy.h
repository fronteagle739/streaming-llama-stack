#pragma once

#include <cstdint>

struct llama_mtp_layer_range {
    int32_t base_start = 0;
    int32_t base_end   = 0;
    int32_t mtp_start  = 0;
    int32_t mtp_end    = 0;
};

enum llama_mtp_graph_role {
    LLAMA_MTP_GRAPH_ROLE_TARGET   = 0,
    LLAMA_MTP_GRAPH_ROLE_DRAFT    = 1,
    LLAMA_MTP_GRAPH_ROLE_VALIDATE = 2,
};

struct llama_mtp_decode_plan {
    bool native_requested       = false;
    bool has_nextn              = false;
    bool speculative            = false;
    bool requires_target_graph  = false;
    bool requires_draft_graph   = false;
    bool requires_validate_graph = false;
    int32_t target_graph_role   = LLAMA_MTP_GRAPH_ROLE_TARGET;
    int32_t draft_graph_role    = LLAMA_MTP_GRAPH_ROLE_DRAFT;
    int32_t validate_graph_role = LLAMA_MTP_GRAPH_ROLE_VALIDATE;
    int32_t max_draft           = 0;
    const char * reason         = "MTP off";
};

bool llama_mtp_mode_enabled(int32_t mtp_mode);
bool llama_mtp_graph_uses_nextn(int32_t mtp_mode, int32_t graph_role);
int32_t llama_mtp_default_graph_role(int32_t mtp_mode);
int32_t llama_mtp_clamp_max_draft(int32_t requested_max_draft, int32_t nextn_predict_layers);
bool llama_mtp_should_use_native_drafter(int32_t mtp_mode, int32_t nextn_predict_layers, bool has_external_draft);
bool llama_mtp_should_use_native_draft_context(int32_t mtp_mode, int32_t nextn_predict_layers, bool has_external_draft, bool prefill_streaming);
bool llama_mtp_should_use_stream_local_drafter(int32_t mtp_mode, int32_t nextn_predict_layers, bool has_external_draft, bool prefill_streaming);
bool llama_mtp_should_capture_target_hidden(int32_t mtp_mode, int32_t nextn_predict_layers, int32_t n_tokens, int32_t n_outputs);
llama_mtp_decode_plan llama_mtp_build_decode_plan(int32_t mtp_mode, int32_t nextn_predict_layers, int32_t requested_max_draft);
int32_t llama_mtp_base_layer_count(int32_t n_layer, int32_t nextn_predict_layers);
int32_t llama_mtp_effective_layer_count(int32_t n_layer, int32_t nextn_predict_layers, int32_t mtp_mode, int32_t graph_role);
llama_mtp_layer_range llama_mtp_resolve_layer_range(int32_t n_layer, int32_t nextn_predict_layers);
int32_t llama_mtp_clamp_layer_end(int32_t requested_end, int32_t n_layer, int32_t nextn_predict_layers, int32_t mtp_mode);
int32_t llama_mtp_clamp_layer_end_for_role(int32_t requested_end, int32_t n_layer, int32_t nextn_predict_layers, int32_t mtp_mode, int32_t graph_role);
bool llama_mtp_is_nextn_layer(int32_t layer, int32_t n_layer, int32_t nextn_predict_layers);
