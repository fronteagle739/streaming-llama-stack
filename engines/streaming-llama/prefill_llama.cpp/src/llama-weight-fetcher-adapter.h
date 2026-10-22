#pragma once

//
// llama-weight-fetcher-adapter.h — Bridges llama-weight-fetcher with
// llama-prefill-stream's existing tensor_remap_entry / layer_tensor_info.
//
// This adapter keeps the weight fetcher standalone (no llama.h dependency)
// while providing convenient integration with the streaming prefill pipeline.
//

#include "llama-weight-fetcher.h"
#include "llama-prefill-stream.h"

#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend;

// Weight file location — matches llama_model::weight_file_loc
struct wf_adapter_weight_loc {
    uint16_t file_idx    = 0;
    size_t   file_offset = 0;
};

// ---------------------------------------------------------------------------
// Create a weight fetcher from the existing layer_tensor_info collection.
//
// This converts the tensor_remap_entry spans into wf_weight_span registrations
// and creates the fetcher with optimal I/O parameters.
//
// model_path:      GGUF file path (for direct I/O; NULL = mmap fallback only)
// split_paths:     additional split file paths (NULL-terminated array, or NULL)
// gpu_backend:     ggml_backend_t for GPU allocation
// weight_locs:     tensor name → (file_idx, file_offset) map from model loader
// ---------------------------------------------------------------------------
wf_context * wf_adapter_create(
    const std::vector<layer_tensor_info> & layer_infos,
    size_t                                  max_layer_bytes,
    const char *                            model_path,
    const char **                           split_paths,
    int                                     n_split_files,
    void *                                  gpu_backend,
    bool                                    trace,
    const std::unordered_map<std::string, wf_adapter_weight_loc> * weight_locs = nullptr);

// ---------------------------------------------------------------------------
// Remap tensor pointers to the weight fetcher's GPU buffer for a given layer.
//
// After wf_wait_layer_gpu() returns, call this to update each tensor's
// buffer and data pointer so the scheduler dispatches compute to the GPU
// buffer that the fetcher loaded weights into.
//
// layer_entries: the same entries from layer_infos[il].entries
// gpu_buf_ptr:   the pointer returned by wf_wait_layer_gpu()
// ---------------------------------------------------------------------------
void wf_adapter_remap_tensors(
    std::vector<tensor_remap_entry> & layer_entries,
    void *                            gpu_buf_ptr,
    void *                            gpu_buf_handle);  // ggml_backend_buffer_t

// ---------------------------------------------------------------------------
// Restore tensor pointers back to their original CPU state.
// Called after compute is done for a layer (same as prefill_restore_tensor_pointers
// but for a single layer).
// ---------------------------------------------------------------------------
void wf_adapter_restore_tensors(
    std::vector<tensor_remap_entry> & layer_entries);
