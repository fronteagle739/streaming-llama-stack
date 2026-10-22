#pragma once

#include "llama.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend_buffer;
struct ggml_tensor;
struct ggml_cgraph;

//
// RTX Prefill Streaming Mode — Phase 2.1: Pointer-Rerouted 2-Buffer GPU Rotation
//
// This module implements streaming prefill where only 2 layers of model weights
// reside in GPU memory at any time. Instead of snapshotting weights to RAM
// (which caused the Triple-Allocation Trap / SIGKILL on 128GB machines),
// this version uses pointer rerouting:
//
//   1. Model loads with -ngl 0 (all weights in CPU heap RAM)
//   2. Allocate 2 GPU buffers sized for the largest layer
//   3. Temporarily remap all weight tensor->buffer / tensor->data to GPU
//      so the scheduler routes compute to the Metal backend
//   4. Restore original CPU pointers after graph allocation
//   5. Per-layer callbacks upload weights from CPU heap → rotating GPU buffers
//
// Memory budget: ~model_size + 2*max_layer_size + KV + system
//   For 70B Q4_K_M: ~70 GB + 2*1.1 GB + 15 GB ≈ 87 GB (fits in 128 GB)
//

struct llama_prefill_stream_params {
    int    n_buffers          = 2;               // double-buffer by default
    bool   telemetry          = true;            // emit per-layer timing
    bool   enable_overlap     = false;           // Phase 2: load(i+1) while compute(i)
    int    min_streaming_tokens = 32;             // minimum batch tokens before early skip is allowed
    bool   force_streaming    = false;            // bypass small-batch skip for explicit streaming modes
    int    prefetch_distance  = 1;               // layer lookahead distance
    size_t slab_bytes         = 16 * 1024 * 1024; // upload chunk size in bytes (0 = whole tensor)
    bool   remote_weight_relay = false;          // Phase 6: route layer slabs through relay path
    bool   remote_weight_hybrid = true;          // keep local fallback path if relay is unavailable
    int    host_cache_window_layers = 0;         // S3: moving-window layer count (0 = disabled)
    size_t host_cache_window_bytes  = 0;         // S3: moving-window byte budget (0 = disabled)
    int    host_cache_prefetch_distance = 1;     // S3: host prefetch lookahead in layers
    bool   disable_cuda_graphs = false;          // scoped runtime guard for graph-unsafe streaming paths
    bool   route_trace_decode_step = false;      // instrumentation: caller classified this as a decode step
    bool   route_trace_prompt_last = false;      // instrumentation: final marked stream_local prompt position
};

struct llama_layer_metrics {
    int    layer_idx;
    float  load_time_ms;     // weight upload time (CPU→GPU)
    float  compute_time_ms;  // forward pass time
    float  stall_time_ms;    // time waiting for upload
    size_t weight_bytes;
};

struct llama_prefill_stream_result {
    std::vector<llama_layer_metrics> layer_metrics;
    float                            total_time_ms;
    float                            total_tok_s;
    int                              n_tokens_processed;
    bool                             used_streaming;
    bool                             host_cache_enabled = false;
    bool                             host_cache_advice_supported = true;
    int                              host_prefetch_layers = 0;
    size_t                           host_prefetch_bytes = 0;
    int                              host_evict_layers = 0;
    size_t                           host_evict_bytes = 0;
    int                              host_cache_lookups = 0;
    int                              host_cache_hits = 0;
    int                              host_cache_misses = 0;
    float                            host_cache_hit_rate = 0.0f;
    float                            host_storage_wait_ms = 0.0f;
};

//
// Tensor Remap Entry — saves original CPU state, enables GPU rerouting
//
struct tensor_remap_entry {
    ggml_tensor *         tensor;       // pointer to model tensor
    ggml_backend_buffer * orig_buffer;  // original CPU buffer (restored after graph build)
    void *                orig_data;    // original CPU data pointer (source for uploads)
    size_t                nbytes;       // cached ggml_nbytes(tensor)
    size_t                alloc_size = 0;   // GPU alloc size (from get_alloc_size, includes padding)
    size_t                alloc_offset = 0; // byte offset within layer's GPU buffer
    size_t                slice_offset = 0; // offset within tensor for expert slices (0 = full tensor)
};

//
// Per-layer tensor collection
//
struct layer_tensor_info {
    std::vector<tensor_remap_entry> entries;
    size_t                          total_bytes = 0;
};

// Execute streaming prefill on a batch
// Returns 0 on success, negative on error
int llama_prefill_stream(llama_context *                     ctx,
                         const llama_batch &                 batch,
                         const llama_prefill_stream_params & params,
                         llama_prefill_stream_result *       result);

// Query whether streaming prefill is available for this context
bool llama_prefill_stream_available(const llama_context * ctx);

// Calculate total weight bytes for one layer
size_t llama_layer_weight_bytes(const llama_model * model, int layer_idx);

//
// Pointer-Rerouting API — used by decode_streaming() in llama-context.cpp
//

// Collect all per-layer tensor pointers and compute max layer size
// Must be called before remap_tensors_to_gpu()
void prefill_collect_layer_tensors(const llama_model *              model,
                                   std::vector<layer_tensor_info> & out_layers,
                                   size_t &                         out_max_layer_bytes);

// Collect attention-only tensors per layer for split decode.
// Returns total attention weight bytes across all layers.
// Used by split decode: RTX loads only attention weights into VRAM.
size_t prefill_collect_attention_tensors(const llama_model *              model,
                                         std::vector<layer_tensor_info> & out_layers);

// Compute GPU alloc offsets for all layers using the backend's get_alloc_size.
// Must be called after prefill_collect_layer_tensors and after the GPU backend
// is known.  Sets alloc_size, alloc_offset, and re-computes total_bytes for
// each layer to include backend-specific padding (e.g., MATRIX_ROW_PADDING).
void prefill_compute_alloc_offsets(std::vector<layer_tensor_info> & layers,
                                   size_t &                         max_layer_bytes,
                                   ggml_backend_buffer_type_t       gpu_buft);

// Allocate 2 GPU buffers for the largest layer
// Returns the buffer type used (for diagnostics)
bool prefill_allocate_gpu_buffers(ggml_backend_t          backend,
                                  size_t                  max_layer_bytes,
                                  ggml_backend_buffer_t * out_buf_a,
                                  ggml_backend_buffer_t * out_buf_b);

// Temporarily remap all weight tensors to point at a GPU buffer.
// This tricks the backend scheduler into routing all compute to GPU.
// Call before model.build_graph() + ggml_backend_sched_alloc_graph().
void prefill_remap_tensors_to_gpu(std::vector<layer_tensor_info> & layers, ggml_backend_buffer_t gpu_buf);

// Restore original CPU buffer/data pointers on all tensors.
// Call after ggml_backend_sched_alloc_graph() and before per-layer dispatch.
void prefill_restore_tensor_pointers(std::vector<layer_tensor_info> & layers);

// Upload one layer's weights from original CPU heap into the active GPU buffer.
// Sets each tensor's buffer/data to point into gpu_buf at packed offsets.
// Returns upload time in ms.
float prefill_upload_layer(std::vector<tensor_remap_entry> & layer_entries,
                           ggml_backend_buffer_t             gpu_buf,
                           size_t                            slab_bytes);
