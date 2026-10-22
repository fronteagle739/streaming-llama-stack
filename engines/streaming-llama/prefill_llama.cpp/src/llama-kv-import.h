#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Scatter map entry — one contiguous copy operation.
// ---------------------------------------------------------------------------

struct kv_scatter_entry {
    uint32_t layer_idx;      // KV cache layer index (via map_layer_ids)
    uint32_t tensor_kind;    // 0 = K, 1 = V
    size_t   artifact_offset; // byte offset in flat artifact payload
    size_t   tensor_offset;   // byte offset within the ggml_tensor data
    size_t   bytes;           // contiguous copy length
};

// ---------------------------------------------------------------------------
// Scatter map — pre-computed at model load time.
// ---------------------------------------------------------------------------

struct kv_scatter_map {
    std::vector<kv_scatter_entry> entries;
    size_t total_payload_bytes = 0;
};

// ---------------------------------------------------------------------------
// Model info needed to build the scatter map (extracted from llama_context).
// ---------------------------------------------------------------------------

struct kv_import_layer_info {
    uint32_t cache_idx;      // KV cache layer index
    uint32_t n_embd_k_gqa;  // K embedding dim for this layer
    uint32_t n_embd_v_gqa;  // V embedding dim for this layer
};

struct kv_import_model_info {
    uint32_t n_layers    = 0;
    uint32_t kv_size     = 0;    // total KV cache slots
    bool     v_trans     = false;
    bool     is_mla      = false;
    size_t   type_k_size = 0;    // bytes per K element (e.g., 2 for fp16)
    size_t   type_v_size = 0;    // bytes per V element
    uint64_t model_fingerprint = 0;

    std::vector<kv_import_layer_info> layers;
};

// ---------------------------------------------------------------------------
// Artifact info for validation (extracted from artifact header).
// ---------------------------------------------------------------------------

struct kv_import_artifact_info {
    uint16_t format_major      = 0;
    uint32_t n_layers          = 0;
    uint32_t token_count       = 0;
    bool     v_trans           = false;
    bool     is_mla            = false;
    uint64_t model_fingerprint = 0;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// Build scatter map from model info and artifact token count.
// Returns false if token_count >= kv_size (no room for decode).
bool kv_scatter_map_build(const kv_import_model_info & info,
                          uint32_t token_count,
                          kv_scatter_map & map_out);

// Validate artifact metadata against model info.
// allow_fingerprint_mismatch: if true, skip fingerprint check.
bool kv_import_validate(const kv_import_model_info & model,
                        const kv_import_artifact_info & artifact,
                        bool allow_fingerprint_mismatch,
                        std::string * error);
