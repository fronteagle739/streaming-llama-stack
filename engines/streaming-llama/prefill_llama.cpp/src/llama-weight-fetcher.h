#pragma once

//
// llama-weight-fetcher.h — Optimised NVMe-to-GPU weight streaming pipeline
//
// Standalone library that replaces mmap-based weight loading with a 3-stage
// async pipeline designed to saturate NVMe RAID arrays and PCIe links:
//
//   Stage 0 (NVMe → Pinned RAM): async direct I/O into cudaMallocHost buffers
//   Stage 1 (Pinned → GPU):      cudaMemcpyAsync on dedicated copy stream
//   Stage 2 (GPU compute):       layer execution on compute stream
//
// With GPUDirect Storage (GDS) available, stages 0-2 collapse into a single
// DMA from NVMe controller directly to GPU VRAM via cuFile:
//
//   Stage 0 (NVMe → GPU):  cuFileRead into GPU buffer
//   Stage 1 (GPU compute): layer execution
//
// Pipeline depth (default 4 layers ahead):
//   Layer N:   computing on GPU buf A
//   Layer N+1: H2D in flight to GPU buf B
//   Layer N+2: in pinned RAM, ready to push
//   Layer N+3: NVMe async read in progress
//
// MoE support:
//   FETCH_ALL:    push all expert weights (no stall, max bandwidth)
//   FETCH_ACTIVE: fetch only gating-selected experts (~80% BW saving, half-layer stall)
//
// Post-prefill modes:
//   HANDOFF_MMAP:  release fetcher, fall back to normal mmap for decode
//   KEEP_FETCHER:  keep pipeline active for decode (one token at a time, all layers)
//

#include <cstddef>
#include <cstdint>

// DLL export/import decoration for shared library builds
#if defined(_WIN32) && !defined(WF_STATIC)
#  if defined(LLAMA_BUILD)
#    define WF_API __declspec(dllexport)
#  else
#    define WF_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define WF_API __attribute__((visibility("default")))
#else
#  define WF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
typedef struct wf_context wf_context;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

// Transfer backend — how weights move from storage to GPU
typedef enum {
    WF_MODE_MMAP       = 0,   // legacy: read from mmap'd region (current behaviour)
    WF_MODE_DIRECT_IO  = 1,   // async direct I/O: NVMe → RAM → pinned → GPU
    WF_MODE_GDS        = 2,   // GPUDirect Storage: NVMe → GPU (requires GDS driver, Linux)
    WF_MODE_DSTORAGE   = 3,   // Microsoft DirectStorage: NVMe → GPU via D3D12 (Windows)
    WF_MODE_AUTO       = 4,   // try DSTORAGE/GDS, fall back to DIRECT_IO, then MMAP
} wf_transfer_mode;

// MoE expert fetch strategy
typedef enum {
    WF_MOE_FETCH_ALL    = 0,  // push all expert weights (no stall, max bandwidth)
    WF_MOE_FETCH_ACTIVE = 1,  // fetch only active experts (stall for gating, ~80% BW save)
} wf_moe_mode;

// What happens after prefill completes
typedef enum {
    WF_POST_HANDOFF_MMAP  = 0,  // release fetcher, fall back to mmap for decode
    WF_POST_KEEP_FETCHER  = 1,  // keep pipeline running for decode passes
} wf_post_prefill_mode;

// Pipeline stage a layer is currently in
typedef enum {
    WF_STAGE_IDLE       = 0,  // not requested yet
    WF_STAGE_NVME_READ  = 1,  // async NVMe read in progress
    WF_STAGE_RAM_READY  = 2,  // weight data in RAM staging buffer
    WF_STAGE_H2D_FLIGHT = 3,  // cudaMemcpyAsync to GPU in progress
    WF_STAGE_GPU_READY  = 4,  // on GPU, ready for compute
    WF_STAGE_COMPUTING  = 5,  // GPU kernels executing on this buffer
    WF_STAGE_DONE       = 6,  // compute finished, buffer reclaimable
} wf_layer_stage;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

typedef struct {
    // ---- File access ----
    const char * model_path;              // GGUF file path (primary)
    const char ** split_paths;            // additional split file paths (NULL-terminated, or NULL)
    int           n_split_files;          // number of split files (0 if single file)

    // ---- NVMe I/O tuning ----
    size_t  io_block_size;                // read block size (default: 512 * 1024 = 512KiB)
    int     io_queue_depth;              // async I/O queue depth per thread (default: 4)
    int     io_threads;                  // number of I/O reader threads (default: 8)
    bool    io_direct;                   // use O_DIRECT / FILE_FLAG_NO_BUFFERING (default: true)

    // ---- Pipeline geometry ----
    int     prefetch_depth;              // layers to prefetch ahead (default: 3)
    int     n_gpu_buffers;               // ping-pong GPU buffers (default: 2)
    int     n_ram_buffers;               // pinned RAM staging buffers (default: 2)
    size_t  max_layer_bytes;             // largest layer's (or sub-layer's) weight footprint
    size_t  ds_staging_limit_bytes;        //0 preserves legacy; strict profiles admit250/1500MiB
    bool    ds_async_drain;               // explicit strict profile: existing copy-stream/drain path
    uint32_t ds_submit_threads;           //0 untouched/default; explicit strict TEST1/2/4
    int     n_sublayers;                 // sub-layers per model layer (default: 1, MoE split: 2-4)

    // ---- GPU ----
    void *  gpu_backend;                 // ggml_backend_t for the target GPU
    void *  cuda_device;                 // CUDA device pointer (for GDS registration), or NULL

    // ---- Transfer mode ----
    wf_transfer_mode    mode;            // MMAP / DIRECT_IO / GDS / AUTO

    // ---- MoE ----
    wf_moe_mode         moe_mode;        // FETCH_ALL or FETCH_ACTIVE

    // ---- Post-prefill ----
    wf_post_prefill_mode post_mode;      // HANDOFF_MMAP or KEEP_FETCHER

    // ---- Misc ----
    int     n_layers;                    // total model layers
    bool    trace;                       // emit per-operation timing to stderr
    uint32_t ds_gpu_ring_slots;           //0 off; strict one-shot TEST2, full pass queued upfront
    bool ds_zero_padding_only;           // explicit verified full-span direct fill; default false
    bool ds_compute_direct;              // explicit strict dual-use compute heaps; no relay
} wf_params;

// Default-initialise params with sensible values
static inline wf_params wf_params_default(void) {
    wf_params p;
    p.model_path       = NULL;
    p.split_paths      = NULL;
    p.n_split_files    = 0;
    p.io_block_size    = 512 * 1024;     // 512 KiB
    p.io_queue_depth   = 4;
    p.io_threads       = 8;
    p.io_direct        = true;
    p.prefetch_depth   = 3;
    p.n_gpu_buffers    = 2;
    p.n_ram_buffers    = 2;
    p.max_layer_bytes  = 0;
    p.gpu_backend      = NULL;
    p.cuda_device      = NULL;
    p.mode             = WF_MODE_AUTO;
    p.moe_mode         = WF_MOE_FETCH_ALL;
    p.post_mode        = WF_POST_KEEP_FETCHER;
    p.n_layers         = 0;
    p.trace            = false;
    p.ds_staging_limit_bytes = 0;
    p.ds_async_drain = false;
    p.ds_submit_threads = 0;
    p.ds_gpu_ring_slots = 0;
    p.ds_compute_direct = false;
    p.ds_zero_padding_only = false;
    p.n_sublayers      = 1;
    return p;
}

// ---------------------------------------------------------------------------
// Weight span — describes one tensor's location in the GGUF file
// ---------------------------------------------------------------------------

typedef struct {
    const char * tensor_name;             // ggml tensor name (for diagnostics)
    uint64_t     file_offset;             // absolute byte offset in GGUF file
    size_t       nbytes;                  // tensor size in bytes (actual data from file)
    size_t       alloc_size;              // GPU alloc size (from get_alloc_size, 0 = use nbytes)
    void *       mmap_addr;               // fallback: existing mmap pointer (or NULL)
    int          file_idx;                // split file index (0 for single-file models)

    // MoE metadata
    bool         is_expert;               // true if this tensor belongs to an expert
    int          expert_idx;              // which expert (0..n_experts-1), -1 if not expert

    // GPU buffer offset from ggml allocator (0 = use sequential packing)
    size_t       alloc_offset;            // byte offset within layer's GPU buffer
} wf_weight_span;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create the weight fetcher pipeline.
// Allocates GPU buffers, pinned staging, opens file handles, starts I/O threads.
// Returns NULL on failure.
WF_API wf_context * wf_create(const wf_params * params);

// Explicit opt-in for callers whose host-memory admission cannot tolerate a
// whole-layer pinned fallback. Requires mode DSTORAGE, a CUDA backend, exact
// CUDA/D3D12 adapter identity, and every requested split file. Returns NULL
// before fallback allocations if DS is unavailable or initialization fails.
// Experimental alternate heaps, persistent/global staging, extracted-layer
// files and optional expert caches are rejected, not silently inherited.
// Like wf_create's DS path, compute buffers must subsequently be supplied with
// wf_set_gpu_buf_ptr(). This does not qualify a model or admit VRAM by itself.
WF_API wf_context * wf_create_required_dstorage(const wf_params * params);
// Sticky fatal transfer/registration/synchronization status. Required callers
// must check after void registration, upload waits and release boundaries.
WF_API bool wf_has_failed(const wf_context * wf);

// Destroy the pipeline, free all resources, join threads.
WF_API void wf_destroy(wf_context * wf);

// DS4COMBO: release the process-global DirectStorage/D3D12 cache (staging, files,
// DS factory, D3D12 device). Call only after every wf_context is destroyed. 1 = released.
WF_API int wf_persist_shutdown(const char * why);

// ---------------------------------------------------------------------------
// Layer registration
// ---------------------------------------------------------------------------

// Register the weight spans for a single layer.
// Must be called for each layer [0, n_layers) before starting the pipeline.
// The spans array is copied internally.
WF_API void wf_register_layer(wf_context *         wf,
                        int                  layer_idx,
                        const wf_weight_span * spans,
                        int                  n_spans,
                        size_t               total_bytes);

// ---------------------------------------------------------------------------
// Pipeline control
// ---------------------------------------------------------------------------

// Start the pipeline — begins prefetching from layer 0.
// Call after all layers are registered.
WF_API void wf_start(wf_context * wf);

// reverse-ds: start a BACKWARD pass — streams layers N-1..0 for full-DMA-speed training backward streaming.
// Forward inference uses wf_start (unchanged); reverse must be requested explicitly. See TRAINING_ROADMAP.md.
WF_API void wf_start_reverse(wf_context * wf);

// Request a specific layer be fetched (if not already in pipeline).
// Non-blocking: schedules the fetch and returns immediately.
WF_API void wf_request_layer(wf_context * wf, int layer_idx);

// Block until a layer's weights are on GPU and ready for compute.
// Returns a pointer to the GPU buffer base for this layer's packed weights.
// The caller must NOT free this pointer — it's managed by the ping-pong pool.
WF_API void * wf_wait_layer_gpu(wf_context * wf, int layer_idx);

// Mark a layer's GPU buffer as reclaimable (compute is done).
// The buffer may be reused for a future layer's weights.
WF_API void wf_release_layer(wf_context * wf, int layer_idx);

// Prefetch first half of next layer into prefetch_buf (synchronous).
// Call after compute completes (in post_layer_cb) to pre-load data.
WF_API void wf_prefetch_next_layer(wf_context * wf, int current_layer_idx);

// Allocate prefetch buffer using remaining VRAM after graph_alloc.
WF_API void wf_allocate_prefetch(wf_context * wf);

// Free prefetch buffer (releases VRAM between streaming prefill passes).
WF_API void wf_free_prefetch(wf_context * wf);
// Drain DirectStorage and every CUDA consumer before freeing or reusing any
// WF-owned or WF-targeted GPU allocation. Call after wf_stop() at boundaries.
WF_API void wf_quiesce(wf_context * wf);

// Release current sub-layer and immediately schedule next sub-layer on same buffer.
// Bypasses dma_next_layer counter — used for within-model-layer sub-layer transitions.
WF_API void wf_swap_sublayer(wf_context * wf, int old_layer_idx, int new_layer_idx);

// Signal end of a pass (prefill or decode turn).
// In HANDOFF_MMAP mode, this tears down the pipeline and releases resources.
// In KEEP_FETCHER mode, this resets layer tracking for the next pass.
WF_API void wf_end_pass(wf_context * wf);

// Stop the pipeline, drain all pending I/O, join threads.
// Call before wf_destroy() or to pause between passes.
WF_API void wf_stop(wf_context * wf);

// ---------------------------------------------------------------------------
// MoE: active expert selection
// ---------------------------------------------------------------------------

// Set which experts are active for a given layer.
// Only relevant when moe_mode == WF_MOE_FETCH_ACTIVE.
// Must be called BEFORE wf_wait_layer_gpu() for that layer.
// expert_ids: array of active expert indices, n_active: count.
// LLAMADECODE B1b: a persistent stage occupies its own bank region that no other
// stage of this fetcher writes. After its first complete transfer it is served
// from VRAM (no DMA) for the rest of this fetcher's lifetime. Required DS only.
WF_API bool wf_set_layer_persistent(wf_context * wf, int layer_idx, bool persistent);
// LLAMADECODE B2: synchronously fetch explicit file extents into a direct compute
// bank (required DS, two-bank direct profile). Serialized with the pipeline's DMA
// (one DS submission in flight); the caller's CUDA stream waits on the DS fence
// before any consumer. The caller owns the bank region (it must not overlap any
// pipeline stage written later in the same pass).
typedef struct { uint64_t file_offset; uint64_t bank_offset; uint64_t nbytes; int file_idx; } wf_extent;
WF_API bool wf_fetch_extents_direct(wf_context * wf, int bank, const wf_extent * extents, int n,
                                    void * cuda_stream, uint64_t * bytes, double * milliseconds);
WF_API void wf_set_active_experts(wf_context * wf,
                           int          layer_idx,
                           const int *  expert_ids,
                           int          n_active);

// ---------------------------------------------------------------------------
// GPU buffer management
// ---------------------------------------------------------------------------

// Get the GPU buffer handle for a layer that's ready (after wf_wait_layer_gpu).
// Returns the ggml_backend_buffer_t so callers can remap tensor pointers.
WF_API void * wf_get_gpu_buffer(wf_context * wf, int layer_idx);

// Get the packed offset for a specific span within the layer's GPU buffer.
// span_idx corresponds to the registration order in wf_register_layer().
WF_API size_t wf_get_span_gpu_offset(wf_context * wf, int layer_idx, int span_idx);

// ---------------------------------------------------------------------------
// CUDA event synchronisation (replaces ggml_backend_synchronize)
// ---------------------------------------------------------------------------

// Insert a GPU-side event wait on the compute stream, ensuring the H2D copy
// for this layer has completed.  Non-blocking on the host — only the GPU stalls.
// This replaces the full ggml_backend_synchronize() that was killing overlap.
WF_API void wf_cuda_wait_upload(wf_context * wf, int layer_idx);

// Record a compute-done event on the compute stream for this layer.
// Used internally to track when a GPU buffer becomes reclaimable.
WF_API void wf_cuda_record_compute_done(wf_context * wf, int layer_idx);

// Query the MoE fetch mode.
WF_API wf_moe_mode wf_get_moe_mode(wf_context * wf);

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

typedef struct {
    // Bandwidth
    float    nvme_read_bw_gbps;          // effective NVMe read bandwidth (GB/s)
    float    ram_to_gpu_bw_gbps;         // effective H2D bandwidth (GB/s)
    float    end_to_end_bw_gbps;         // NVMe → GPU effective (GB/s)

    // Timing
    float    total_nvme_read_ms;         // total time in NVMe reads
    float    total_staging_ms;           // total time in RAM→pinned copies
    float    total_h2d_ms;               // total time in H2D transfers
    float    total_stall_ms;             // total time GPU was idle waiting
    float    total_compute_ms;           // total GPU compute time

    // Counters
    size_t   bytes_read_nvme;            // bytes read from NVMe
    size_t   bytes_uploaded_gpu;         // bytes transferred to GPU
    int      layers_fetched;             // layers fully fetched
    int      stall_count;                // number of times GPU had to wait
    int      io_ops;                     // number of NVMe I/O operations

    // MoE
    int      experts_skipped;            // experts not fetched (FETCH_ACTIVE mode)
    size_t   expert_bytes_saved;         // bandwidth saved by skipping experts

    // Three-tier expert cache (VRAM -> pinned RAM -> NVMe)
    uint64_t cache_hits;                 // total VRAM + RAM cache hits
    uint64_t cache_vram_hits;            // expert spans served by the VRAM tier
    uint64_t cache_ram_hits;             // expert spans served by the pinned-RAM tier
    uint64_t cache_misses;               // expert spans falling through to NVMe/mmap
    size_t   cache_bytes_saved;           // NVMe/mmap bytes avoided by cache hits

    // Predictive expert prefetch
    uint64_t expert_prefetch_predicted;   // expert predictions actually issued
    uint64_t expert_prefetch_hits;        // predicted experts selected by routing
    uint64_t expert_prefetch_misses;      // routed experts fetched reactively
    size_t   expert_prefetch_wasted_bytes;// prefetched bytes unused by routing

    // Mode
    bool     gds_active;                 // true if GDS path is in use
    wf_transfer_mode effective_mode;     // actual mode after AUTO resolution
} wf_stats;

// Get cumulative stats since wf_create() or last wf_reset_stats().
WF_API void wf_get_stats(wf_context * wf, wf_stats * stats);
// Completed strict DirectStorage transfer: actual requested file bytes and
// submit-to-visible-compute-buffer wall time (includes enqueue, fences and D2D).
// No filesystem cache or model arithmetic is touched by this snapshot.
WF_API bool wf_get_required_layer_fetch(wf_context * wf, int layer,
                                      uint64_t * bytes, double * milliseconds);

// Reset all counters.
WF_API void wf_reset_stats(wf_context * wf);

// Print a human-readable summary to stderr.
WF_API void wf_print_stats(wf_context * wf);

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

// Check if GPUDirect Storage is available on this system.
WF_API bool wf_gds_available(void);

// Get the current pipeline stage for a layer.
WF_API wf_layer_stage wf_get_layer_stage(wf_context * wf, int layer_idx);

// Get the effective transfer mode (after AUTO resolution).
WF_API wf_transfer_mode wf_get_effective_mode(wf_context * wf);

// Query CUDA VRAM usage. Returns false if CUDA not available.
// free_bytes and total_bytes are set on success.
WF_API bool wf_cuda_get_vram_info(size_t * free_bytes, size_t * total_bytes);

// Query number of GPU buffers and per-buffer capacity.
WF_API int    wf_get_n_gpu_bufs(wf_context * wf);
WF_API size_t wf_get_gpu_buf_capacity(wf_context * wf);
WF_API void * wf_get_gpu_buf_ptr(wf_context * wf, int buf_idx);
WF_API void   wf_set_gpu_buf_ptr(wf_context * wf, int buf_idx, void * ptr);

// R3 DS-direct: CUDA device index the WF resolved (for ggml_backend_cuda_buffer_from_ptr), and whether
// DS-direct mode is ACTUALLY active (false if it was requested but fell back to legacy on heap-alloc failure).
WF_API int    wf_get_cuda_device_index(wf_context * wf);
WF_API bool   wf_is_ds_direct(wf_context * wf);

// ---------------------------------------------------------------------------
// Layer file range query (for parallel RAM prefetch tests)
// ---------------------------------------------------------------------------

// Get the file offset range for a layer's weight data.
// Returns the first span's file_offset and layer total_bytes.
// Useful for parallel NVMe bandwidth tests (read to RAM while DS reads to GPU).
WF_API bool wf_get_layer_file_range(wf_context * wf, int layer_idx,
                                     uint64_t * out_first_offset,
                                     size_t * out_total_bytes);

// Get the model file path.
WF_API const char * wf_get_model_path(wf_context * wf);

// ---------------------------------------------------------------------------
// Layer-major mode
// ---------------------------------------------------------------------------

// Enable/disable layer-major mode. When enabled, the weight fetcher
// skips ubatch boundary detection and reset logic. The caller is
// responsible for managing layer lifecycle via wf_wait_layer_gpu /
// wf_release_layer directly. This prevents Nx weight re-streaming
// when processing multiple ubatches per layer.
WF_API void wf_set_layer_major_mode(wf_context * wf, bool enabled);
WF_API bool wf_get_layer_major_mode(wf_context * wf);

// ---------------------------------------------------------------------------
// DirectStorage interop — expose DS handles for external RAM queues
// ---------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif
