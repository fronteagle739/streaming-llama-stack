#include "llama-dsv41-layer-log.h"
#include "llama-weight-fetcher.h"
#include "llama-wf-bounded-policy.h"
#include "llama-wf-zero-gaps.h"

#include <algorithm>
#include <array>
#include <memory>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Platform-specific async I/O
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  if __has_include(<liburing.h>)
#    define WF_HAS_IO_URING 1
#    include <liburing.h>
#  endif
#endif

// CUDA (optional — only if headers are actually available)
// GGML_USE_CUDA may be defined by the build system, but the CUDA toolkit
// include path is only present for ggml-cuda targets, not the llama library.
// Use __has_include to detect actual availability.
#if defined(GGML_USE_CUDA) && __has_include(<cuda_runtime.h>)
#  define WF_HAS_CUDA 1
#  include <cuda_runtime.h>
#  include "ggml-cuda.h"  // for ggml_backend_cuda_get_stream
// GDS / cuFile (optional)
#  if __has_include(<cufile.h>)
#    define WF_HAS_GDS 1
#    include <cufile.h>
#  endif
#endif

// DirectStorage + D3D12-CUDA interop (Windows only, requires CUDA + DirectStorage SDK)
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE) && defined(WF_HAS_CUDA)
#  define WF_HAS_DSTORAGE 1
#  include <dstorage.h>
#  include <d3d12.h>
#  include <dxgi1_4.h>
#  include <objbase.h>
#endif

static bool wf_verbose = false;  // set to true when trace is enabled
#define WF_ERR(fmt, ...)   fprintf(stderr, "[weight-fetcher] " fmt "\n", ##__VA_ARGS__)
#define WF_LOG(fmt, ...)   do { if (wf_verbose) fprintf(stderr, "[weight-fetcher] " fmt "\n", ##__VA_ARGS__); } while(0)
#define WF_TRACE(wf, fmt, ...) do { if ((wf)->params.trace) fprintf(stderr, "[weight-fetcher] " fmt "\n", ##__VA_ARGS__); } while(0)

#ifdef WF_HAS_CUDA
static int wf_resolve_cuda_device_index(void * gpu_backend) {
    if (!gpu_backend) {
        return -1;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t) gpu_backend);
    if (!dev) {
        return -1;
    }

    const char * dev_name = ggml_backend_dev_name(dev);
    if (dev_name) {
        const size_t prefix_len = strlen(GGML_CUDA_NAME);
        if (strncmp(dev_name, GGML_CUDA_NAME, prefix_len) == 0) {
            char * end = nullptr;
            long idx = strtol(dev_name + prefix_len, &end, 10);
            if (end && *end == '\0' && idx >= 0 && idx < ggml_backend_cuda_get_device_count()) {
                return (int) idx;
            }
        }
    }

    const char * dev_desc = ggml_backend_dev_description(dev);
    if (dev_desc) {
        char cuda_desc[256] = {};
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            ggml_backend_cuda_get_device_description(i, cuda_desc, sizeof(cuda_desc));
            if (strcmp(dev_desc, cuda_desc) == 0) {
                return i;
            }
        }
    }

    return -1;
}

struct wf_cuda_device_scope {
    int  prev_device = -1;
    bool restore     = false;
    bool ok          = false;

    explicit wf_cuda_device_scope(int target_device) {
        if (target_device < 0) {
            return;
        }

        cudaError_t err_prev = cudaGetDevice(&prev_device);
        if (err_prev != cudaSuccess) {
            prev_device = -1;
        }

        if (prev_device == target_device) {
            ok = true;
            return;
        }

        cudaError_t err = cudaSetDevice(target_device);
        if (err == cudaSuccess) {
            ok = true;
            restore = prev_device >= 0;
        } else {
            WF_ERR("cudaSetDevice(%d) failed: %s", target_device, cudaGetErrorString(err));
        }
    }

    ~wf_cuda_device_scope() {
        if (restore) {
            cudaSetDevice(prev_device);
        }
    }
};
#endif

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

// One tensor's worth of fetch metadata
struct wf_span_internal {
    char         tensor_name[128];
    uint64_t     file_offset;
    size_t       nbytes;
    void *       mmap_addr;          // fallback pointer (may be NULL)
    int          file_idx;
    bool         is_expert;
    int          expert_idx;
    size_t       gpu_offset;         // packed offset within layer's GPU buffer
};

struct wf_cache_key {
    int      layer_idx;
    int      expert_idx;
    uint64_t tensor_hash;

    bool operator==(const wf_cache_key & o) const {
        return layer_idx == o.layer_idx && expert_idx == o.expert_idx && tensor_hash == o.tensor_hash;
    }
};

struct wf_cache_key_hash {
    size_t operator()(const wf_cache_key & k) const {
        uint64_t h = k.tensor_hash;
        h ^= (uint64_t)(uint32_t)k.layer_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= (uint64_t)(uint32_t)k.expert_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct wf_cache_entry {
    size_t slot = 0;
    int file_idx = -1;
    uint64_t file_offset = 0;
    size_t nbytes = 0;
    size_t pins = 0;
    std::list<wf_cache_key>::iterator lru_it;
};

struct wf_cache_tier {
    void * arena = nullptr;
    size_t arena_bytes = 0;
    size_t slot_size = 0;
    size_t slot_count = 0;
    std::vector<bool> used;
    std::list<wf_cache_key> lru; // front = MRU, back = LRU
    std::unordered_map<wf_cache_key, wf_cache_entry, wf_cache_key_hash> entries;
    std::mutex mutex;
};

// Per-layer state
struct wf_layer {
    std::vector<wf_span_internal> spans;
    size_t                        total_bytes = 0;

    // Pipeline tracking
    std::atomic<wf_layer_stage>   stage{WF_STAGE_IDLE};
    int                           gpu_buf_idx = -1;    // which ping-pong buffer (-1 = none)
    int                           ram_buf_idx = -1;    // which RAM staging buffer (-1 = none)

    // MoE: active expert mask (empty = fetch all)
    std::vector<int>              active_experts;
    bool                          expert_filter_set = false;
    bool                          persistent = false;          // LLAMADECODE B1b
    std::atomic<bool>             persistent_loaded{false};    // bytes resident in its bank region
    bool                          cache_inserted = false;
    std::vector<bool>             cache_hits;
    std::vector<int>              expert_prefetch_ids;
    std::vector<bool>             expert_prefetch_loaded_spans;
    bool                          expert_prefetch_planned = false;
    bool                          expert_prefetch_loaded = false;

    // Timing
    float                         nvme_read_ms  = 0.0f;
    float                         staging_ms    = 0.0f;
    float                         h2d_ms        = 0.0f;

    wf_layer() = default;
    wf_layer(wf_layer && o) noexcept
        : spans(std::move(o.spans)), total_bytes(o.total_bytes),
          stage(o.stage.load()), gpu_buf_idx(o.gpu_buf_idx), ram_buf_idx(o.ram_buf_idx),
          active_experts(std::move(o.active_experts)), expert_filter_set(o.expert_filter_set),
          persistent(o.persistent), persistent_loaded(o.persistent_loaded.load()),
          cache_inserted(o.cache_inserted), cache_hits(std::move(o.cache_hits)),
          expert_prefetch_ids(std::move(o.expert_prefetch_ids)),
          expert_prefetch_loaded_spans(std::move(o.expert_prefetch_loaded_spans)),
          expert_prefetch_planned(o.expert_prefetch_planned),
          expert_prefetch_loaded(o.expert_prefetch_loaded),
          nvme_read_ms(o.nvme_read_ms), staging_ms(o.staging_ms), h2d_ms(o.h2d_ms) {}
    wf_layer & operator=(wf_layer &&) = delete;
    wf_layer(const wf_layer &) = delete;
    wf_layer & operator=(const wf_layer &) = delete;
};

// Ring buffer slot for GPU or RAM
struct wf_buffer_slot {
    void *  ptr       = nullptr;       // GPU device ptr or pinned host ptr
    size_t  capacity  = 0;
    int     layer_idx = -1;            // which layer occupies this slot (-1 = free)
    bool    owns_ptr  = false;         // true when WF allocated ptr itself
    bool    ext_owned = false;         // R3 DS-direct: ptr is a D3D12-heap import (free via ext_mem/heap, NOT cudaMalloc)
    std::atomic<bool> in_use{false};
#ifdef WF_HAS_CUDA
    cudaEvent_t upload_done  = nullptr; // signalled when H2D is complete
    cudaEvent_t compute_done = nullptr; // signalled when GPU compute finishes
    cudaExternalMemory_t ext_mem = {};  // R3 DS-direct: external memory handle backing ptr (when ext_owned)
#endif
#ifdef WF_HAS_DSTORAGE
    ID3D12Heap *                 heap = nullptr;     // R3 DS-direct: the owning D3D12 heap (when ext_owned)
    std::vector<ID3D12Resource*> quarters;           // R3 DS-direct: placed sub-resources (<4GB each), DS targets
#endif

    wf_buffer_slot() = default;
    wf_buffer_slot(wf_buffer_slot && o) noexcept
        : ptr(o.ptr), capacity(o.capacity), layer_idx(o.layer_idx), owns_ptr(o.owns_ptr), ext_owned(o.ext_owned), in_use(o.in_use.load())
#ifdef WF_HAS_CUDA
        , upload_done(o.upload_done), compute_done(o.compute_done), ext_mem(o.ext_mem)
#endif
#ifdef WF_HAS_DSTORAGE
        , heap(o.heap), quarters(std::move(o.quarters))
#endif
    {}
    wf_buffer_slot(const wf_buffer_slot &) = delete;
    wf_buffer_slot & operator=(const wf_buffer_slot &) = delete;
};

// File handle (platform-abstracted)
struct wf_file_handle {
#ifdef _WIN32
    HANDLE   h = INVALID_HANDLE_VALUE;
#else
    int      fd = -1;
#endif
    size_t   file_size = 0;
};

// Per-chunk overlapped I/O context — one per in-flight I/O operation.
// On Windows this embeds OVERLAPPED so it can be recovered from IOCP.
struct wf_io_chunk {
#ifdef _WIN32
    OVERLAPPED ov;                     // MUST be first member for IOCP casting
#endif
    int        layer_idx;
    int        span_idx;
    int        file_idx;               // which file handle to read from
    uint64_t   file_offset;            // absolute file offset for this chunk
    size_t     nbytes;                 // bytes requested for this chunk
    void *     dest;                   // destination in RAM staging buffer
    bool       pooled;                 // true if from chunk_pool (don't delete)
    int        mt_handle_idx;          // index into mt_files for re-submission (-1 = N/A)
};

// Per-layer I/O completion tracking
struct wf_layer_io_tracker {
    std::atomic<int>   outstanding{0};    // chunks still in flight
    std::atomic<int>   total_chunks{0};   // total chunks submitted
    std::atomic<int>   completed{0};      // chunks finished
    std::atomic<size_t> bytes_done{0};    // bytes completed so far
    std::chrono::high_resolution_clock::time_point t_start;

    wf_layer_io_tracker() = default;
    wf_layer_io_tracker(wf_layer_io_tracker && o) noexcept
        : outstanding(o.outstanding.load()), total_chunks(o.total_chunks.load()),
          completed(o.completed.load()), bytes_done(o.bytes_done.load()),
          t_start(o.t_start) {}
    wf_layer_io_tracker & operator=(wf_layer_io_tracker &&) = delete;
    wf_layer_io_tracker(const wf_layer_io_tracker &) = delete;
    wf_layer_io_tracker & operator=(const wf_layer_io_tracker &) = delete;
};

// ---------------------------------------------------------------------------
// Chunk pool — avoids per-chunk heap allocation on the hot path
// ---------------------------------------------------------------------------

struct wf_chunk_pool {
    std::vector<wf_io_chunk>      storage;   // pre-allocated chunk objects
    std::vector<wf_io_chunk *>    free_list;
    std::mutex                    mu;

    void init(int capacity) {
        storage.resize(capacity);
        free_list.reserve(capacity);
        for (int i = capacity - 1; i >= 0; i--) {
            free_list.push_back(&storage[i]);
        }
    }

    wf_io_chunk * acquire() {
        std::lock_guard<std::mutex> lock(mu);
        if (free_list.empty()) return nullptr;
        auto * c = free_list.back();
        free_list.pop_back();
        return c;
    }

    void release(wf_io_chunk * c) {
        std::lock_guard<std::mutex> lock(mu);
        free_list.push_back(c);
    }
};

// ---------------------------------------------------------------------------
// Main context
// ---------------------------------------------------------------------------

struct wf_context {
    wf_params                     params;
    wf_transfer_mode              effective_mode = WF_MODE_MMAP;
    // Common lifecycle/query code reads this gate on every platform. The
    // DirectStorage resources themselves remain Windows-only, but an inert
    // false field must exist for Mac receiver builds as well.
    bool                          ds_direct_mode = false;
    bool                          required_dstorage = false;
#ifdef WF_HAS_CUDA
    int                           cuda_device_index = -1;
#endif

    // Per-layer metadata
    std::vector<wf_layer>         layers;

    // Per-layer I/O tracking
    std::vector<wf_layer_io_tracker> io_trackers;

    // Buffer pools
    std::vector<wf_buffer_slot>   gpu_bufs;        // ping-pong GPU buffers
    std::vector<wf_buffer_slot>   ram_bufs;        // pinned RAM staging buffers

    // Optional expert cache. Both tiers are default-off and independently sized.
    wf_cache_tier                 expert_vram_cache;
    wf_cache_tier                 expert_ram_cache;
    size_t                        expert_cache_max_span = 0;
    size_t                        expert_cache_vram_request = 0;
    size_t                        expert_cache_ram_request = 0;
    bool                          expert_cache_initialized = false;
    uint64_t                      expert_cache_passes = 0;

    // Optional per-layer EMA expert predictor (default off).
    bool                          expert_prefetch_initialized = false;
    bool                          expert_prefetch_enabled = false;
    int                           expert_prefetch_n = 0;
    int                           expert_prefetch_dist = 2;
    uint64_t                      expert_prefetch_passes = 0;
    std::vector<std::vector<float>> expert_prefetch_ema;
    std::vector<int>              expert_prefetch_last_active;
    uint64_t                      expert_prefetch_log_predicted = 0;
    uint64_t                      expert_prefetch_log_hits = 0;
    uint64_t                      expert_prefetch_log_misses = 0;
    size_t                        expert_prefetch_log_waste = 0;
    std::mutex                    expert_prefetch_mutex;

    // Half-buffer prefetch: DMA first half of next layer while current computes.
    // Avoids CNR complexity — compute stays on full-layer fast path.
    void *                        prefetch_buf = nullptr;   // GPU buffer, half of max_layer
    size_t                        prefetch_buf_size = 0;
    int                           prefetch_layer_idx = -1;  // layer loaded in prefetch_buf (-1 = none)
    std::atomic<bool>             prefetch_ready{false};    // true when first-half DMA is complete
    std::future<void>             prefetch_future;          // async prefetch handle

    // File handles (one per split file, used on Linux / non-RAID)
    std::vector<wf_file_handle>   files;

    // Per-thread file handles for RAID bandwidth (Windows: N handles to same file)
    // DiskSpd/CrystalDiskMark uses per-thread handles with contiguous file regions
    // and steady-state queue depth (QD=4 per handle) to saturate RAID arrays.
    std::vector<wf_file_handle>   mt_files;   // one per I/O thread

    // Per-handle steady-state QD submission queues (DiskSpd model).
    // Each handle reads a contiguous region of the layer, with QD outstanding I/Os.
    // On completion, the IOCP worker re-submits the next chunk from the same handle's
    // queue, maintaining constant pipeline depth — exactly like DiskSpd's main loop.
    // Flat array: submit_queues[layer_idx * n_mt_handles + handle_idx]
    struct wf_submit_queue {
        std::vector<wf_io_chunk *> chunks;   // ordered by file offset (contiguous per handle)
        std::atomic<int>           next{0};  // next chunk index to submit (lock-free)
    };
    wf_submit_queue *             submit_queues = nullptr;
    int                           n_mt_handles = 0;

    // I/O completion threads (IOCP workers on Windows, io_uring/epoll on Linux)
    std::vector<std::thread>      io_threads;
    std::atomic<bool>             io_shutdown{false};

#ifdef _WIN32
    HANDLE                        iocp = NULL;     // I/O completion port
#endif
#ifdef WF_HAS_IO_URING
    struct io_uring               ring;
    bool                          ring_init = false;
#endif

    // Fallback: request queue for platforms without native async I/O
    std::deque<wf_io_chunk *>     io_fallback_queue;
    std::mutex                    io_fallback_mutex;
    std::condition_variable       io_fallback_cv;

    // Pipeline scheduler thread
    std::thread                   pipeline_thread;
    std::mutex                    pipeline_mutex;
    std::condition_variable       pipeline_cv;
    std::atomic<bool>             pipeline_running{false};
    std::atomic<int>              pipeline_next_layer{0};
    // Pass direction (reverse-ds backward streaming). Default +1/0 == forward == current inference behavior;
    // wf_start_reverse() sets dir=-1/start_layer=n_layers-1 for the training backward. See TRAINING_ROADMAP.md.
    int                           dir{+1};
    int                           start_layer{0};

    // DMA control thread — polls compute_done events and fires Submit() instantly.
    // Per-buffer pending slots allow both buffers to be serviced without blocking.
    std::thread                   dma_thread;
    struct dma_buf_slot {
        std::atomic<int>  pending_layer{-1};  // layer to DMA into this buffer (-1 = none)
        // Lock-free compute-done flag (LLAMA_WF_LOCKFREE_READY=1). Set true by a
        // cudaLaunchHostFunc callback enqueued on the compute stream AFTER the
        // layer's compute; the DMA thread polls this atomic (no CUDA driver lock)
        // instead of cudaEventQuery, so the host's kernel-launch burst no longer
        // starves the DMA's readiness check. Starts true (buffer free).
        std::atomic<bool> compute_ready{true};
        dma_buf_slot() = default;
        dma_buf_slot(dma_buf_slot && o) noexcept
            : pending_layer(o.pending_layer.load()), compute_ready(o.compute_ready.load()) {}
        dma_buf_slot(const dma_buf_slot &) = delete;
    };
    std::vector<dma_buf_slot>     dma_buf_pending;    // one per GPU buffer
    std::atomic<int>              dma_next_layer{0};  // next layer to schedule via DMA thread
    std::atomic<int>              dma_in_flight_buf{-1}; // buffer idx with DMA currently in-flight (-1 = idle)
    // Pre-enqueued buffer: DS requests enqueued, waiting for Submit().
    // Pipeline thread calls Submit() instantly on fence completion, then
    // enqueues the NEXT layer (during DMA) for zero inter-transfer gap.
    std::atomic<int>              enqueued_buf{-1};   // buf idx with enqueued requests (-1 = none)
    std::atomic<int>              enqueued_layer{-1}; // layer idx of enqueued requests

    // Chunk pool (avoids heap alloc per I/O chunk)
    wf_chunk_pool                 chunk_pool;

    // Stats — hot-path counters are atomic, cold stats under mutex
    std::atomic<size_t>           stat_bytes_read_nvme{0};
    std::atomic<int>              stat_io_ops{0};
    wf_stats                      stats;
    std::mutex                    stats_mutex;

    // Wall-clock timing for accurate bandwidth
    std::chrono::high_resolution_clock::time_point pass_start;
    std::chrono::high_resolution_clock::time_point pass_end;

    // Per-transfer DMA timing log (submit → fence complete)
    struct dma_timing_entry {
        int      layer_idx;
        int      buf_idx;
        size_t   nbytes;
        std::chrono::high_resolution_clock::time_point t_submit;
        std::chrono::high_resolution_clock::time_point t_fence;
    };
    std::vector<dma_timing_entry>  dma_timing_log;
    std::mutex                     dma_timing_mutex;

#ifdef WF_HAS_CUDA
    cudaStream_t                  copy_stream = nullptr;
    // Staging double-buffer (LLAMA_WF_STAGING_DBLBUF=1): per-half copy_stream events
    // so wave w+2's D3D12 DS write into half (w%2) waits (host-side) for wave w's
    // async D2D (which read that half) to finish — fixes the single-staging-reuse
    // race that corrupts the naive async-D2D path. Enables async D2D -> DMA stays
    // fed without the OP_PROFILE host-pacing crutch -> compute-bound ~685 @20k.
    cudaEvent_t                   ds_stage_evt[2] = { nullptr, nullptr };
    // CUDA compute stream from ggml backend (needed because ggml-cuda uses
    // cudaStreamNonBlocking, so the null stream won't synchronize with it).
    cudaStream_t                  cuda_compute_stream = nullptr;
    // GDS
    bool                          gds_available = false;
#ifdef WF_HAS_GDS
    CUfileHandle_t                cufile_handle = nullptr;
#endif
#endif

    // Sticky fatal state for a DirectStorage queue/device. Kept outside the
    // feature guard so common wait code can reject a failed transfer in every
    // build configuration.
    std::atomic<bool>            ds_failed{false};

#ifdef WF_HAS_DSTORAGE
    // DirectStorage + D3D12-CUDA interop state
    bool                          ds_initialized = false;
    IDStorageFactory *            ds_factory = nullptr;
    IDStorageQueue *              ds_queue = nullptr;
    IDStorageQueue *              ds_queue2 = nullptr;  // second queue to avoid slot contention
    std::mutex                    ds_queue_mutex;       // serializes full and reactive submissions
    std::vector<IDStorageFile *>  ds_files;            // per-shard DS handles
    std::vector<IDStorageFile *>  ds_layer_files;      // per-layer .bin file handles (when using extracted layers)
    bool                          use_layer_files = false;
    ID3D12Device *                d3d_device = nullptr;
    IDXGIFactory4 *               dxgi_factory = nullptr;

    // DS can only target ONE D3D12 resource per queue submission, and
    // Destination.Buffer.Offset must stay < ~4 GB.  We use a single 3.5 GB
    // staging resource.  Layers larger than 3.5 GB are transferred in waves:
    //   wave 0: DS → staging[0..3.5G], fence, D2D → compute[0..3.5G]
    //   wave 1: DS → staging[0..2.1G], fence, D2D → compute[3.5G..5.6G]
    struct ds_staging {
        ID3D12Resource *         d3d_resource = nullptr;
        cudaExternalMemory_t     ext_mem = {};
        HANDLE                   shared_handle = NULL;
        void *                   cuda_ptr = nullptr;
        uint64_t                 size = 0;             // 3.5 GB
        bool                     persistent = false;   // LLAMA_WF_PERSIST_INFRA: cached across requests, skip free
    } ds_stage;
    struct ds_compute_bank {
        void *ptr=nullptr;cudaExternalMemory_t external={};ID3D12Heap *heap=nullptr;
        std::vector<ID3D12Resource*> quarters;uint64_t quarter_bytes=0;
    };
    std::vector<ds_compute_bank> ds_compute_banks;


    // R3 DS-DIRECT (LLAMA_WF_DS_DIRECT): two PERMANENT D3D12-heap-backed CUDA ping-pong buffers —
    // compute = gpu_bufs[0] (ggml-wrapped, non-owning), prefetch = prefetch_buf. DS writes weights
    // DIRECTLY into the prefetch heap's placed quarters (<4GB each, B1-proven), then a single D2D
    // prefetch->compute at the layer boundary. NO staging tier. Default-off == today's exact behavior.
    cudaExternalMemory_t          prefetch_ext_mem = {};    // external memory backing prefetch_buf
    ID3D12Heap *                  prefetch_heap = nullptr;
    std::vector<ID3D12Resource*>  prefetch_quarters;        // placed sub-resources (<4GB each), DS targets
    uint64_t                      ds_quarter_size = 0;      // QSZ: per-quarter byte size (<4GB)
    cudaEvent_t                   prefetch_done = nullptr;   // signalled after DS fills prefetch; D2D waits on it

    // Per-wave prebuild: ds_waves[layer_idx][wave_idx]
    struct ds_wave {
        uint64_t                 ring_sequence = 0;    // global immutable pass wave
        uint64_t                 compute_offset = 0;   // D2D target in compute buffer
        uint64_t                 nbytes = 0;           // bytes transferred this wave
        std::vector<DSTORAGE_REQUEST> reqs;
        std::vector<int>              req_span_idx; // -1 for layer-file requests
    };
    std::vector<std::vector<ds_wave>> ds_waves;        // [layer][wave]

    // D3D12 fence → CUDA external semaphore (GPU-side sync, zero CPU stall)
    ID3D12Fence *                ds_fence = nullptr;
    HANDLE                       ds_fence_handle = NULL;
    cudaExternalSemaphore_t      ds_ext_sem = {};
    std::atomic<uint64_t>        ds_fence_counter{0};
    // Per-layer: fence value signaled when DMA completes
    std::vector<uint64_t>        ds_layer_fence_val;

    struct ds_ring_fence {ID3D12Fence*fence=nullptr;HANDLE handle=nullptr;cudaExternalSemaphore_t semaphore={};};
    struct ds_ring_state {
        ds_ring_fence read[2],drain,compute;
        std::vector<IDStorageQueue3*> queues;
        std::vector<size_t> stage_queue;
        std::vector<HANDLE> ends;
        cudaEvent_t copy_cleanup=nullptr,compute_cleanup=nullptr;
        uint64_t waves=0,requests=0,slot_bytes=0;
        uint32_t transferred=0,released=0;
        bool armed=false,prepared=false;
    };
    std::unique_ptr<ds_ring_state> ds_ring;

    // DS queue capacity
    int                          ds_queue_cap = 0;

    // Staging buffer size constant (set during init)
    uint64_t                     ds_staging_size = 0;

    // Per-layer mmap fallback spans (need cudaMemcpy at submit time)
    struct ds_mmap_span {
        size_t gpu_offset;
        const void * src;
        size_t nbytes;
        int span_idx;
    };
    std::vector<std::vector<ds_mmap_span>> ds_mmap_spans;  // [layer_idx]
    bool                         ds_com_inited = false;
#endif

    // Layer-major mode: when true, skip ubatch boundary reset.
    // The caller manages layer lifecycle directly via wf_wait_layer_gpu /
    // wf_release_layer, processing all ubatches per layer before moving on.
    bool                         layer_major_mode = false;
};

// ---------------------------------------------------------------------------
// Three-tier expert cache helpers
// ---------------------------------------------------------------------------

static uint64_t wf_cache_name_hash(const char * s) {
    uint64_t h = 1469598103934665603ULL;
    for (; s && *s; ++s) {
        h ^= (uint8_t)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static wf_cache_key wf_cache_make_key(int layer_idx, const wf_span_internal & span) {
    return { layer_idx, span.expert_idx, wf_cache_name_hash(span.tensor_name) };
}

static bool wf_cache_content_matches(const wf_cache_entry & e, const wf_span_internal & span) {
    return e.file_idx == span.file_idx && e.file_offset == span.file_offset && e.nbytes == span.nbytes;
}

static bool wf_span_is_active_expert(const wf_context * wf, const wf_layer & layer,
                                     const wf_span_internal & span) {
    if (!span.is_expert) return false;
    if (wf->params.moe_mode != WF_MOE_FETCH_ACTIVE || !layer.expert_filter_set) return true;
    return std::find(layer.active_experts.begin(), layer.active_experts.end(), span.expert_idx)
           != layer.active_experts.end();
}

static size_t wf_cache_env_bytes(const char * name) {
    const char * value = std::getenv(name);
    if (!value || !value[0] || value[0] == '-') return 0;
    char * end = nullptr;
    unsigned long long mb = std::strtoull(value, &end, 10);
    if (end == value || (end && *end != '\0') || mb == 0) return 0;
    const unsigned long long max_mb = (unsigned long long)SIZE_MAX / (1024ULL * 1024ULL);
    return (size_t)std::min(mb, max_mb) * 1024ULL * 1024ULL;
}

static uint64_t wf_physical_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullTotalPhys : 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    return (pages > 0 && page_size > 0) ? (uint64_t)pages * (uint64_t)page_size : 0;
#endif
}

static void wf_cache_touch_locked(wf_cache_tier & tier,
                                  std::unordered_map<wf_cache_key, wf_cache_entry, wf_cache_key_hash>::iterator it) {
    tier.lru.erase(it->second.lru_it);
    tier.lru.push_front(it->first);
    it->second.lru_it = tier.lru.begin();
}

static size_t wf_cache_take_slot_locked(wf_cache_tier & tier, wf_cache_key * evicted_key,
                                        wf_cache_entry * evicted_entry,
                                        const wf_cache_key * exclude = nullptr) {
    for (size_t i = 0; i < tier.used.size(); ++i) {
        if (!tier.used[i]) {
            tier.used[i] = true;
            return i;
        }
    }
    for (auto rit = tier.lru.rbegin(); rit != tier.lru.rend(); ++rit) {
        if (exclude && *rit == *exclude) continue;
        auto found = tier.entries.find(*rit);
        if (found == tier.entries.end() || found->second.pins != 0) continue;
        size_t slot = found->second.slot;
        if (evicted_key) *evicted_key = found->first;
        if (evicted_entry) *evicted_entry = found->second;
        tier.lru.erase(found->second.lru_it);
        tier.entries.erase(found);
        return slot;
    }
    return SIZE_MAX;
}

static void wf_cache_publish_locked(wf_cache_tier & tier, const wf_cache_key & key,
                                    const wf_span_internal & span, size_t slot) {
    auto old = tier.entries.find(key);
    if (old != tier.entries.end()) {
        tier.lru.erase(old->second.lru_it);
        tier.used[old->second.slot] = false;
        tier.entries.erase(old);
    }
    tier.used[slot] = true;
    tier.lru.push_front(key);
    tier.entries.emplace(key, wf_cache_entry{
        slot, span.file_idx, span.file_offset, span.nbytes, 0, tier.lru.begin()
    });
}

static void wf_cache_init(wf_context * wf) {
    if (!wf || wf->expert_cache_initialized) return;
    wf->expert_cache_initialized = true;
    if (wf->required_dstorage) return;
    wf->expert_cache_vram_request = wf_cache_env_bytes("LLAMA_WF_EXPERT_CACHE_MB");
    wf->expert_cache_ram_request = wf_cache_env_bytes("LLAMA_WF_EXPERT_CACHE_RAM_MB");
    if (!wf->expert_cache_vram_request && !wf->expert_cache_ram_request) return;

    for (const auto & layer : wf->layers) {
        for (const auto & span : layer.spans) {
            if (span.is_expert) wf->expert_cache_max_span = std::max(wf->expert_cache_max_span, span.nbytes);
        }
    }
    if (!wf->expert_cache_max_span) {
        WF_ERR("expert cache requested but no expert spans were registered; disabling");
        wf->expert_cache_vram_request = wf->expert_cache_ram_request = 0;
        return;
    }

    const uint64_t physical = wf_physical_ram_bytes();
    const size_t ram_limit = physical ? (size_t)std::min<uint64_t>(physical / 4, SIZE_MAX) : 0;
    if (ram_limit && wf->expert_cache_ram_request > ram_limit) {
        WF_ERR("warning: LLAMA_WF_EXPERT_CACHE_RAM_MB requests %.1f MB; clamping to 25%% of physical RAM (%.1f MB)",
               wf->expert_cache_ram_request / (1024.0 * 1024.0), ram_limit / (1024.0 * 1024.0));
        wf->expert_cache_ram_request = ram_limit;
    }

#ifdef WF_HAS_CUDA
    auto init_tier = [&](wf_cache_tier & tier, size_t requested, bool device) {
        if (!requested) return;
        tier.slot_size = wf->expert_cache_max_span;
        tier.slot_count = requested / tier.slot_size;
        if (!tier.slot_count) {
            WF_ERR("warning: expert-cache %s tier is smaller than one %.2f MB slot; disabling",
                   device ? "VRAM" : "RAM", tier.slot_size / (1024.0 * 1024.0));
            return;
        }
        tier.arena_bytes = tier.slot_count * tier.slot_size;
        cudaError_t err = device ? cudaMalloc(&tier.arena, tier.arena_bytes)
                                 : cudaMallocHost(&tier.arena, tier.arena_bytes);
        if (err != cudaSuccess) {
            WF_ERR("warning: expert-cache %s allocation of %.1f MB failed (%s); tier disabled",
                   device ? "VRAM" : "pinned-RAM", tier.arena_bytes / (1024.0 * 1024.0),
                   cudaGetErrorString(err));
            tier.arena = nullptr;
            tier.arena_bytes = tier.slot_count = 0;
            return;
        }
        tier.used.assign(tier.slot_count, false);
    };
    init_tier(wf->expert_vram_cache, wf->expert_cache_vram_request, true);
    init_tier(wf->expert_ram_cache, wf->expert_cache_ram_request, false);
#else
    WF_ERR("warning: expert cache requested in a build without CUDA; disabling");
#endif

    if (wf->expert_vram_cache.arena || wf->expert_ram_cache.arena) {
        WF_ERR("expert cache enabled: slot=%.2f MB, VRAM=%zu slots/%.1f MB, pinned-RAM=%zu slots/%.1f MB",
               wf->expert_cache_max_span / (1024.0 * 1024.0),
               wf->expert_vram_cache.slot_count, wf->expert_vram_cache.arena_bytes / (1024.0 * 1024.0),
               wf->expert_ram_cache.slot_count, wf->expert_ram_cache.arena_bytes / (1024.0 * 1024.0));
        if (wf->params.moe_mode != WF_MOE_FETCH_ACTIVE) {
            WF_ERR("warning: expert cache is enabled without WF_MOE_FETCH_ACTIVE; FETCH_ALL eviction pressure usually removes its benefit");
        }
    }
}

static bool wf_cache_enabled(const wf_context * wf) {
    return wf && (wf->expert_vram_cache.arena || wf->expert_ram_cache.arena);
}

// Read-only tier probe used by the predictor. An expert is considered cached
// only when every one of its registered weight spans is present in either tier.
static bool wf_cache_contains(wf_context * wf, int layer_idx, const wf_span_internal & span) {
    if (!wf_cache_enabled(wf) || !span.is_expert) return false;
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);
    auto vi = vram.entries.find(key);
    if (vi != vram.entries.end() && wf_cache_content_matches(vi->second, span)) return true;
    auto ri = ram.entries.find(key);
    return ri != ram.entries.end() && wf_cache_content_matches(ri->second, span);
}

#ifdef WF_HAS_CUDA
static void * wf_cache_slot_ptr(const wf_cache_tier & tier, size_t slot) {
    return (char *)tier.arena + slot * tier.slot_size;
}

// Lookup in VRAM -> RAM order and copy directly to the requested destination.
// host_destination is used by the legacy RAM-staged path; CUDA stream ordering
// guarantees its later H2D cannot overtake a VRAM-hit D2H.
static bool wf_cache_lookup(wf_context * wf, int layer_idx, const wf_span_internal & span,
                            void * destination, bool host_destination, bool pin = false,
                            bool defer_copy = false) {
    if (!wf_cache_enabled(wf) || !span.is_expert) return false;
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;

    if (vram.arena) {
        std::lock_guard<std::mutex> lock(vram.mutex);
        auto it = vram.entries.find(key);
        if (it != vram.entries.end() && wf_cache_content_matches(it->second, span)) {
            void * src = wf_cache_slot_ptr(vram, it->second.slot);
            if (!defer_copy) {
                cudaMemcpyAsync(destination, src, span.nbytes,
                                host_destination ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice,
                                wf->copy_stream);
            }
            if (pin) ++it->second.pins;
            wf_cache_touch_locked(vram, it);
            std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
            wf->stats.cache_hits++;
            wf->stats.cache_vram_hits++;
            wf->stats.cache_bytes_saved += span.nbytes;
            return true;
        }
        if (it != vram.entries.end()) {
            vram.used[it->second.slot] = false;
            vram.lru.erase(it->second.lru_it);
            vram.entries.erase(it);
        }
    }

    if (ram.arena) {
        std::scoped_lock lock(vram.mutex, ram.mutex);
        auto it = ram.entries.find(key);
        if (it != ram.entries.end() && wf_cache_content_matches(it->second, span)) {
            void * src = wf_cache_slot_ptr(ram, it->second.slot);
            if (!defer_copy) {
                cudaMemcpyAsync(destination, src, span.nbytes,
                                host_destination ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice,
                                wf->copy_stream);
            }
            if (pin) ++it->second.pins;
            wf_cache_touch_locked(ram, it);

            // Promote RAM hits into VRAM. If VRAM is full, demote its LRU entry
            // into a different RAM slot before reusing the VRAM slot.
            if (vram.arena) {
                wf_cache_key evicted_key{};
                wf_cache_entry evicted{};
                size_t vslot = wf_cache_take_slot_locked(vram, &evicted_key, &evicted);
                if (vslot != SIZE_MAX) {
                    if (evicted.nbytes && ram.slot_count > 1) {
                        wf_cache_key dropped_key{};
                        wf_cache_entry dropped{};
                        size_t rslot = wf_cache_take_slot_locked(ram, &dropped_key, &dropped, &key);
                        if (rslot != SIZE_MAX) {
                            cudaMemcpyAsync(wf_cache_slot_ptr(ram, rslot), wf_cache_slot_ptr(vram, vslot),
                                            evicted.nbytes, cudaMemcpyDeviceToHost, wf->copy_stream);
                            wf_span_internal evicted_span{};
                            evicted_span.file_idx = evicted.file_idx;
                            evicted_span.file_offset = evicted.file_offset;
                            evicted_span.nbytes = evicted.nbytes;
                            wf_cache_publish_locked(ram, evicted_key, evicted_span, rslot);
                        }
                    }
                    cudaMemcpyAsync(wf_cache_slot_ptr(vram, vslot), src, span.nbytes,
                                    cudaMemcpyHostToDevice, wf->copy_stream);
                    wf_cache_publish_locked(vram, key, span, vslot);
                }
            }
            std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
            wf->stats.cache_hits++;
            wf->stats.cache_ram_hits++;
            wf->stats.cache_bytes_saved += span.nbytes;
            return true;
        }
        if (it != ram.entries.end()) {
            ram.used[it->second.slot] = false;
            ram.lru.erase(it->second.lru_it);
            ram.entries.erase(it);
        }
    }

    std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
    wf->stats.cache_misses++;
    return false;
}

static bool wf_cache_copy_existing(wf_context * wf, int layer_idx, const wf_span_internal & span,
                                   void * destination) {
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);
    auto ri = ram.entries.find(key);
    if (ri != ram.entries.end() && ri->second.pins && wf_cache_content_matches(ri->second, span)) {
        cudaMemcpyAsync(destination, wf_cache_slot_ptr(ram, ri->second.slot), span.nbytes,
                        cudaMemcpyHostToDevice, wf->copy_stream);
        return true;
    }
    auto vi = vram.entries.find(key);
    if (vi != vram.entries.end() && wf_cache_content_matches(vi->second, span)) {
        cudaMemcpyAsync(destination, wf_cache_slot_ptr(vram, vi->second.slot), span.nbytes,
                        cudaMemcpyDeviceToDevice, wf->copy_stream);
        return true;
    }
    if (ri != ram.entries.end() && wf_cache_content_matches(ri->second, span)) {
        cudaMemcpyAsync(destination, wf_cache_slot_ptr(ram, ri->second.slot), span.nbytes,
                        cudaMemcpyHostToDevice, wf->copy_stream);
        return true;
    }
    return false;
}

static void wf_cache_unpin(wf_context * wf, int layer_idx, const wf_span_internal & span) {
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);
    auto vi = vram.entries.find(key);
    if (vi != vram.entries.end() && vi->second.pins) --vi->second.pins;
    auto ri = ram.entries.find(key);
    if (ri != ram.entries.end() && ri->second.pins) --ri->second.pins;
}

static void wf_cache_insert_span(wf_context * wf, int layer_idx, const wf_span_internal & span,
                                 const void * gpu_source, bool allow_ram_insert) {
    if (!wf_cache_enabled(wf) || !span.is_expert) return;
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);

    if (vram.arena) {
        auto existing = vram.entries.find(key);
        if (existing != vram.entries.end() && wf_cache_content_matches(existing->second, span)) {
            wf_cache_touch_locked(vram, existing);
        } else {
            wf_cache_key evicted_key{};
            wf_cache_entry evicted{};
            size_t vslot = wf_cache_take_slot_locked(vram, &evicted_key, &evicted);
            if (vslot != SIZE_MAX) {
                if (evicted.nbytes && ram.arena) {
                    wf_cache_key dropped_key{};
                    wf_cache_entry dropped{};
                    size_t rslot = wf_cache_take_slot_locked(ram, &dropped_key, &dropped);
                    if (rslot != SIZE_MAX) {
                        cudaMemcpyAsync(wf_cache_slot_ptr(ram, rslot), wf_cache_slot_ptr(vram, vslot),
                                        evicted.nbytes, cudaMemcpyDeviceToHost, wf->copy_stream);
                        wf_span_internal evicted_span{};
                        evicted_span.file_idx = evicted.file_idx;
                        evicted_span.file_offset = evicted.file_offset;
                        evicted_span.nbytes = evicted.nbytes;
                        wf_cache_publish_locked(ram, evicted_key, evicted_span, rslot);
                    }
                }
                cudaMemcpyAsync(wf_cache_slot_ptr(vram, vslot), gpu_source, span.nbytes,
                                cudaMemcpyDeviceToDevice, wf->copy_stream);
                wf_cache_publish_locked(vram, key, span, vslot);
            }
        }
    }

    if (allow_ram_insert && ram.arena) {
        auto existing = ram.entries.find(key);
        if (existing == ram.entries.end() || !wf_cache_content_matches(existing->second, span)) {
            wf_cache_key dropped_key{};
            wf_cache_entry dropped{};
            size_t rslot = wf_cache_take_slot_locked(ram, &dropped_key, &dropped);
            if (rslot != SIZE_MAX) {
                cudaMemcpyAsync(wf_cache_slot_ptr(ram, rslot), gpu_source, span.nbytes,
                                cudaMemcpyDeviceToHost, wf->copy_stream);
                wf_cache_publish_locked(ram, key, span, rslot);
            }
        } else {
            wf_cache_touch_locked(ram, existing);
        }
    }
}

static void wf_cache_insert_layer(wf_context * wf, int layer_idx) {
    if (!wf_cache_enabled(wf) || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    auto & layer = wf->layers[layer_idx];
    if (layer.cache_inserted || layer.gpu_buf_idx < 0) return;
    layer.cache_inserted = true;
    void * base = wf->gpu_bufs[layer.gpu_buf_idx].ptr;
    const bool ram_idle = !wf->copy_stream || cudaStreamQuery(wf->copy_stream) == cudaSuccess;
    for (const auto & span : layer.spans) {
        if (!wf_span_is_active_expert(wf, layer, span)) continue;
        wf_cache_insert_span(wf, layer_idx, span, (char *)base + span.gpu_offset, ram_idle);
    }
    if (wf->gpu_bufs[layer.gpu_buf_idx].upload_done) {
        cudaEventRecord(wf->gpu_bufs[layer.gpu_buf_idx].upload_done, wf->copy_stream);
    }
}
#endif

// ---------------------------------------------------------------------------
// Platform I/O helpers
// ---------------------------------------------------------------------------

static bool wf_open_file(wf_file_handle * fh, const char * path, bool direct_io) {
#ifdef _WIN32
    // FILE_FLAG_OVERLAPPED is required for IOCP.
    // FILE_FLAG_NO_BUFFERING bypasses the OS page cache for direct NVMe access
    // (required to hit full RAID bandwidth — otherwise the cache manager serialises).
    // FILE_FLAG_SEQUENTIAL_SCAN hints the OS to prefetch aggressively (DiskSpd uses this).
    DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN;
    if (direct_io) {
        flags |= FILE_FLAG_NO_BUFFERING;
    }
    fh->h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                         OPEN_EXISTING, flags, NULL);
    if (fh->h == INVALID_HANDLE_VALUE) {
        WF_ERR("failed to open %s (error %lu)", path, GetLastError());
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(fh->h, &li);
    fh->file_size = (size_t)li.QuadPart;
#else
    int oflags = O_RDONLY;
#ifdef O_DIRECT
    if (direct_io) oflags |= O_DIRECT;
#endif
    fh->fd = open(path, oflags);
    if (fh->fd < 0) {
        WF_ERR("failed to open %s (errno %d)", path, errno);
        return false;
    }
    struct stat st;
    fstat(fh->fd, &st);
    fh->file_size = (size_t)st.st_size;
#endif
    return true;
}

static void wf_close_file(wf_file_handle * fh) {
#ifdef _WIN32
    if (fh->h != INVALID_HANDLE_VALUE) {
        CloseHandle(fh->h);
        fh->h = INVALID_HANDLE_VALUE;
    }
#else
    if (fh->fd >= 0) {
        close(fh->fd);
        fh->fd = -1;
    }
#endif
}

// ---------------------------------------------------------------------------
// IOCP-based async I/O (Windows)
//
// Each ReadFile call submits a 512KiB chunk with an OVERLAPPED struct.
// Multiple chunks are in flight simultaneously (queue depth = io_queue_depth
// per thread × io_threads).  Completions arrive on the IOCP and are
// processed by worker threads that update per-layer atomic counters.
// ---------------------------------------------------------------------------

#ifdef _WIN32

// Submit a single async read chunk via IOCP
static bool wf_submit_read_chunk(wf_context * wf, wf_file_handle * fh,
                                  wf_io_chunk * chunk, uint64_t file_offset) {
    memset(&chunk->ov, 0, sizeof(OVERLAPPED));
    chunk->ov.Offset     = (DWORD)(file_offset & 0xFFFFFFFF);
    chunk->ov.OffsetHigh = (DWORD)(file_offset >> 32);

    DWORD bytes_read = 0;
    BOOL ok = ReadFile(fh->h, chunk->dest, (DWORD)chunk->nbytes, &bytes_read, &chunk->ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            WF_ERR("ReadFile failed: layer %d span %d offset %llu: error %lu",
                   chunk->layer_idx, chunk->span_idx, (unsigned long long)file_offset, err);
            return false;
        }
        // ERROR_IO_PENDING = async I/O submitted successfully
    }
    return true;
}

// Process a single IOCP completion entry
static void wf_process_iocp_entry(wf_context * wf, int thread_id,
                                   LPOVERLAPPED ov, DWORD bytes_transferred, BOOL ok) {
    if (!ov) return;

    wf_io_chunk * chunk = reinterpret_cast<wf_io_chunk *>(ov);

    if (!ok) {
        DWORD err = GetLastError();
        WF_ERR("IOCP completion error: layer %d span %d: error %lu (transferred %lu)",
               chunk->layer_idx, chunk->span_idx, err, bytes_transferred);
    }

    // Update per-layer completion tracking (lock-free)
    int il = chunk->layer_idx;
    if (il >= 0 && il < (int)wf->io_trackers.size()) {
        auto & tracker = wf->io_trackers[il];
        tracker.bytes_done.fetch_add(bytes_transferred, std::memory_order_relaxed);
        int done = tracker.completed.fetch_add(1, std::memory_order_acq_rel) + 1;
        tracker.outstanding.fetch_sub(1, std::memory_order_relaxed);

        if (done == tracker.total_chunks.load(std::memory_order_relaxed)) {
            auto t_done = std::chrono::high_resolution_clock::now();
            float total_ms = std::chrono::duration<float, std::milli>(
                t_done - tracker.t_start).count();
            size_t total_bytes = tracker.bytes_done.load();

            WF_TRACE(wf, "iocp[%d]: layer %d complete: %zu bytes in %.2fms (%.2f GB/s)",
                     thread_id, il, total_bytes, total_ms,
                     total_bytes / (total_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0));

            wf->layers[il].nvme_read_ms = total_ms;
            wf->pipeline_cv.notify_one();
        }
    }

    // Atomic stats — no mutex on hot path
    wf->stat_bytes_read_nvme.fetch_add(bytes_transferred, std::memory_order_relaxed);
    wf->stat_io_ops.fetch_add(1, std::memory_order_relaxed);

    // --- Steady-state QD re-submission (DiskSpd pattern) ---
    // On each completion, immediately submit the next chunk from this handle's
    // queue.  This keeps exactly QD I/Os in flight per handle at all times,
    // which is how DiskSpd/CrystalDiskMark saturates RAID arrays.
    int hi = chunk->mt_handle_idx;
    if (hi >= 0 && il >= 0 && il < (int)wf->io_trackers.size() && wf->submit_queues) {
        auto & sq = wf->submit_queues[il * wf->n_mt_handles + hi];
        int idx = sq.next.fetch_add(1, std::memory_order_relaxed);
        if (idx < (int)sq.chunks.size()) {
            auto * next_chunk = sq.chunks[idx];
            wf_submit_read_chunk(wf, &wf->mt_files[hi], next_chunk, next_chunk->file_offset);
        }
    }

    // Return chunk to pool (or delete if overflow)
    if (chunk->pooled) {
        wf->chunk_pool.release(chunk);
    } else {
        delete chunk;
    }
}

// IOCP completion worker thread — uses batch dequeue for throughput
static void wf_iocp_worker(wf_context * wf, int thread_id) {
    // Batch dequeue: up to 16 completions per syscall
    static constexpr ULONG BATCH_SIZE = 16;
    OVERLAPPED_ENTRY entries[BATCH_SIZE];
    int total_completions = 0;
    int timeout_count = 0;

    while (!wf->io_shutdown.load(std::memory_order_relaxed)) {
        ULONG n_entries = 0;
        BOOL ok = GetQueuedCompletionStatusEx(wf->iocp, entries, BATCH_SIZE,
                                               &n_entries, 100, FALSE);
        if (!ok) {
            DWORD err = GetLastError();
            if (err != WAIT_TIMEOUT) {
                WF_ERR("iocp[%d]: GetQueuedCompletionStatusEx error %lu", thread_id, err);
            } else if (++timeout_count % 50 == 0) {
                // Log every ~5 seconds of pure timeouts (no completions at all)
                WF_LOG("iocp[%d]: %d timeouts, %d completions so far",
                       thread_id, timeout_count, total_completions);
            }
            continue;
        }

        total_completions += n_entries;

        for (ULONG i = 0; i < n_entries; i++) {
            wf_process_iocp_entry(wf, thread_id,
                                  entries[i].lpOverlapped,
                                  entries[i].dwNumberOfBytesTransferred,
                                  TRUE);  // individual error check via GetOverlappedResult if needed
        }
    }
}

#else  // !_WIN32

// Fallback: threaded pread workers for Linux (until io_uring is wired up)
static void wf_fallback_worker(wf_context * wf, int thread_id) {
    while (!wf->io_shutdown.load(std::memory_order_relaxed)) {
        wf_io_chunk * chunk = nullptr;

        {
            std::unique_lock<std::mutex> lock(wf->io_fallback_mutex);
            wf->io_fallback_cv.wait(lock, [&] {
                return !wf->io_fallback_queue.empty() || wf->io_shutdown.load();
            });
            if (wf->io_shutdown.load() && wf->io_fallback_queue.empty()) return;

            chunk = wf->io_fallback_queue.front();
            wf->io_fallback_queue.pop_front();
        }

        if (!chunk) continue;

        // Use stored file_idx and file_offset directly (no reverse-engineering)
        wf_file_handle * fh = &wf->files[chunk->file_idx];
        ssize_t n = pread(fh->fd, chunk->dest, chunk->nbytes, chunk->file_offset);
        size_t bytes_read = (n > 0) ? (size_t)n : 0;

        // Update per-layer tracker (lock-free)
        int il = chunk->layer_idx;
        if (il >= 0 && il < (int)wf->io_trackers.size()) {
            auto & tracker = wf->io_trackers[il];
            tracker.bytes_done.fetch_add(bytes_read, std::memory_order_relaxed);
            int done = tracker.completed.fetch_add(1, std::memory_order_acq_rel) + 1;
            tracker.outstanding.fetch_sub(1, std::memory_order_relaxed);

            if (done == tracker.total_chunks.load(std::memory_order_relaxed)) {
                auto t_done = std::chrono::high_resolution_clock::now();
                wf->layers[il].nvme_read_ms = std::chrono::duration<float, std::milli>(
                    t_done - tracker.t_start).count();
                wf->pipeline_cv.notify_one();
            }
        }

        // Atomic stats — no mutex on hot path
        wf->stat_bytes_read_nvme.fetch_add(bytes_read, std::memory_order_relaxed);
        wf->stat_io_ops.fetch_add(1, std::memory_order_relaxed);

        WF_TRACE(wf, "fallback_worker[%d]: layer %d span %d: %zu bytes at offset %llu",
                 thread_id, il, chunk->span_idx, bytes_read,
                 (unsigned long long)chunk->file_offset);

        // Return chunk to pool (or delete if overflow)
        if (chunk->pooled) {
            wf->chunk_pool.release(chunk);
        } else {
            delete chunk;
        }
    }
}

#endif  // _WIN32

// ---------------------------------------------------------------------------
// Pipeline scheduler
//
// Manages the 4-stage pipeline:
//   NVMe → RAM → Pinned → GPU (or GDS: NVMe → GPU)
//
// Runs on its own thread, advancing layers through the pipeline stages.
// ---------------------------------------------------------------------------

static int wf_find_free_ram_buf(wf_context * wf) {
    for (int i = 0; i < (int)wf->ram_bufs.size(); i++) {
        if (!wf->ram_bufs[i].in_use) return i;
    }
    return -1;
}

static int wf_find_free_gpu_buf(wf_context * wf) {
    for (int i = 0; i < (int)wf->gpu_bufs.size(); i++) {
        if (!wf->gpu_bufs[i].in_use) {
#ifdef WF_HAS_CUDA
            // When skip_eval_sync is active, compute may still be running on this
            // buffer even though in_use was cleared by wf_release_layer().
            // Check the compute_done event to avoid DMA/compute race.
            if (wf->gpu_bufs[i].compute_done) {
                cudaError_t err = cudaEventQuery(wf->gpu_bufs[i].compute_done);
                if (err == cudaErrorNotReady) {
                    continue;  // GPU still using this buffer, try next
                }
                // cudaSuccess or other error — buffer is safe to reuse
            }
#endif
            return i;
        }
    }
    return -1;
}

// Submit NVMe read requests for all spans in a layer.
// Single-pass: handles mmap/skip inline, counts and submits chunks together.
// Returns total number of async chunks submitted (0 = all data was handled via mmap/memcpy).
static int wf_submit_layer_reads(wf_context * wf, int layer_idx, void * dest_buf) {
    auto & layer = wf->layers[layer_idx];
    auto & tracker = wf->io_trackers[layer_idx];
    const size_t block_size = wf->params.io_block_size;

    // Reset tracker
    tracker.outstanding.store(0);
    tracker.total_chunks.store(0);
    tracker.completed.store(0);
    tracker.bytes_done.store(0);
    tracker.t_start = std::chrono::high_resolution_clock::now();

    int total_chunks = 0;
    size_t offset = 0;
#ifdef WF_HAS_CUDA
    if (wf_cache_enabled(wf)) layer.cache_hits.assign(layer.spans.size(), false);
    else layer.cache_hits.clear();
#endif

#ifdef _WIN32
    // Deferred chunk collection for handle-queue distribution (DiskSpd model)
    std::vector<wf_io_chunk *> deferred_chunks;
    bool use_handle_queues = (wf->n_mt_handles > 0);
#else
    // Batch fallback submissions: collect chunks first, notify workers once
    std::vector<wf_io_chunk *> batch_chunks;
#endif

    for (int si = 0; si < (int)layer.spans.size(); si++) {
        auto & span = layer.spans[si];

        // MoE filtering: skip non-active experts
        if (wf->params.moe_mode == WF_MOE_FETCH_ACTIVE && span.is_expert && layer.expert_filter_set) {
            bool active = false;
            for (int eid : layer.active_experts) {
                if (eid == span.expert_idx) { active = true; break; }
            }
            if (!active) {
                memset((char *)dest_buf + offset, 0, span.nbytes);
                {
                    std::lock_guard<std::mutex> lock(wf->stats_mutex);
                    wf->stats.experts_skipped++;
                    wf->stats.expert_bytes_saved += span.nbytes;
                }
                offset += span.nbytes;
                continue;
            }
        }

#ifdef WF_HAS_CUDA
        if (span.is_expert && wf_cache_lookup(wf, layer_idx, span,
                (char *)dest_buf + offset, true, true, true)) {
            layer.cache_hits[si] = true;
            offset += span.nbytes;
            continue;
        }
#endif

        // mmap fallback: synchronous memcpy (no async I/O needed)
        if (wf->effective_mode == WF_MODE_MMAP && span.mmap_addr) {
            memcpy((char *)dest_buf + offset, span.mmap_addr, span.nbytes);
            offset += span.nbytes;
            continue;
        }

        // Submit chunks for this span
        size_t span_offset = 0;
        while (span_offset < span.nbytes) {
            size_t remaining = span.nbytes - span_offset;
            size_t chunk_size = std::min(block_size, remaining);
            // For direct I/O, round up to sector size (4096) so the OS accepts the read.
            // The buffer has padding (10% extra) so the overread is safe.
            if (wf->params.io_direct && (chunk_size & 4095)) {
                chunk_size = (chunk_size + 4095) & ~(size_t)4095;
            }

            // Try pool first, fall back to heap
            wf_io_chunk * chunk = wf->chunk_pool.acquire();
            bool pooled = (chunk != nullptr);
            if (!chunk) chunk = new wf_io_chunk();

            chunk->layer_idx     = layer_idx;
            chunk->span_idx      = si;
            chunk->file_idx      = span.file_idx;
            chunk->file_offset   = span.file_offset + span_offset;
            chunk->nbytes        = chunk_size;
            chunk->dest          = (char *)dest_buf + offset + span_offset;
            chunk->pooled        = pooled;
            chunk->mt_handle_idx = -1;

#ifdef _WIN32
            if (use_handle_queues && span.file_idx == 0) {
                // Defer submission — will be distributed to handle queues below
                deferred_chunks.push_back(chunk);
            } else {
                // Split files or no mt_handles: submit directly to original handle
                chunk->mt_handle_idx = -1;
                wf_file_handle * fh = &wf->files[span.file_idx];
                if (!wf_submit_read_chunk(wf, fh, chunk, chunk->file_offset)) {
                    WF_ERR("submit failed: layer %d span %d chunk at offset %llu",
                           layer_idx, si, (unsigned long long)chunk->file_offset);
                    if (pooled) wf->chunk_pool.release(chunk); else delete chunk;
                    total_chunks++;
                    continue;
                }
            }
#else
            batch_chunks.push_back(chunk);
#endif

            total_chunks++;
            span_offset += chunk_size;
        }

        offset += span.nbytes;
    }

    if (total_chunks == 0) return 0;

    tracker.total_chunks.store(total_chunks);
    tracker.outstanding.store(total_chunks);

#ifdef _WIN32
    // --- DiskSpd-style handle queue distribution ---
    // Assign chunks to handles in CONTIGUOUS ranges.  Each handle reads a
    // sequential region of the layer.  With multiple layers in NVME_READ
    // simultaneously, the overlapping keeps the NVMe busy continuously
    // (steady ~10.5 GB/s) even though individual streams are slower than
    // a single interlocked-sequential stream.
    if (use_handle_queues && !deferred_chunks.empty()) {
        int n_h = wf->n_mt_handles;
        int n_deferred = (int)deferred_chunks.size();
        int base = layer_idx * n_h;
        int chunks_per_handle = (n_deferred + n_h - 1) / n_h;

        // Reset queues for this layer
        for (int hi = 0; hi < n_h; hi++) {
            auto & sq = wf->submit_queues[base + hi];
            sq.chunks.clear();
            sq.next.store(0);
        }

        // Distribute chunks — contiguous ranges per handle
        for (int ci = 0; ci < n_deferred; ci++) {
            int hi = std::min(ci / chunks_per_handle, n_h - 1);
            deferred_chunks[ci]->mt_handle_idx = hi;
            wf->submit_queues[base + hi].chunks.push_back(deferred_chunks[ci]);
        }

        // Submit initial QD chunks per handle, store rest for re-submission.
        // CRITICAL: set sq.next BEFORE submitting ReadFile, so that when a
        // completion fires the IOCP worker re-submits from the correct index
        // (not 0).  Without this, completions arriving before sq.next is set
        // cause double-submission of chunk 0, corrupting the tracker counters
        // and issuing overlapping ReadFile on the same OVERLAPPED (UB).
        int qd = std::max(1, wf->params.io_queue_depth);
        for (int hi = 0; hi < n_h; hi++) {
            auto & sq = wf->submit_queues[base + hi];
            int n_submit = std::min(qd, (int)sq.chunks.size());
            // Point next BEFORE submitting — prevents re-submission race
            sq.next.store(n_submit);
            for (int j = 0; j < n_submit; j++) {
                if (!wf_submit_read_chunk(wf, &wf->mt_files[hi],
                                           sq.chunks[j], sq.chunks[j]->file_offset)) {
                    WF_ERR("submit failed: layer %d handle %d chunk %d", layer_idx, hi, j);
                }
            }
        }

        WF_TRACE(wf, "handle queues: %d chunks across %d handles (QD=%d each, %d/handle)",
                 n_deferred, n_h, qd, chunks_per_handle);
    }
#else
    // Submit all chunks to fallback queue in one lock acquisition
    if (!batch_chunks.empty()) {
        std::lock_guard<std::mutex> lock(wf->io_fallback_mutex);
        for (auto * c : batch_chunks) {
            wf->io_fallback_queue.push_back(c);
        }
        wf->io_fallback_cv.notify_all();  // wake all workers at once
    }
#endif

    WF_TRACE(wf, "submit_layer_reads: layer %d: %d chunks submitted (%zu bytes, block=%zuKiB)",
             layer_idx, total_chunks, layer.total_bytes, block_size / 1024);
    return total_chunks;
}

// Check if all I/O for a layer has completed
static bool wf_layer_io_done(wf_context * wf, int layer_idx) {
    auto & tracker = wf->io_trackers[layer_idx];
    return tracker.completed.load(std::memory_order_acquire) >=
           tracker.total_chunks.load(std::memory_order_relaxed);
}

// H2D transfer: RAM staging → GPU buffer
static void wf_upload_to_gpu(wf_context * wf, int layer_idx) {
    auto & layer = wf->layers[layer_idx];
    int ram_idx = layer.ram_buf_idx;
    int gpu_idx = layer.gpu_buf_idx;

    if (ram_idx < 0 || gpu_idx < 0) {
        WF_ERR("upload_to_gpu: layer %d has no buffer assignment (ram=%d gpu=%d)",
               layer_idx, ram_idx, gpu_idx);
        return;
    }

    void * src = wf->ram_bufs[ram_idx].ptr;
    void * dst = wf->gpu_bufs[gpu_idx].ptr;
    size_t n   = layer.total_bytes;

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef WF_HAS_CUDA
    if (wf->copy_stream) {
        const bool has_cache_hits = layer.cache_hits.size() == layer.spans.size() &&
                std::find(layer.cache_hits.begin(), layer.cache_hits.end(), true) != layer.cache_hits.end();
        if (!has_cache_hits) {
            cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, wf->copy_stream);
        } else {
            size_t span_offset = 0;
            size_t run_start = 0;
            for (int si = 0; si < (int)layer.spans.size(); ++si) {
                const auto & span = layer.spans[si];
                if (layer.cache_hits[si]) {
                    if (span_offset > run_start) {
                        cudaMemcpyAsync((char *)dst + run_start, (char *)src + run_start,
                                        span_offset - run_start, cudaMemcpyHostToDevice, wf->copy_stream);
                    }
                    if (!wf_cache_copy_existing(wf, layer_idx, span,
                            (char *)dst + span.gpu_offset)) {
                        // Defensive fallback: the pinned entry should still exist,
                        // but staging remains a correct source if it does not.
                        cudaMemcpyAsync((char *)dst + span.gpu_offset, (char *)src + span_offset,
                                        span.nbytes, cudaMemcpyHostToDevice, wf->copy_stream);
                    }
                    wf_cache_unpin(wf, layer_idx, span);
                    run_start = span_offset + span.nbytes;
                }
                span_offset += span.nbytes;
            }
            if (n > run_start) {
                cudaMemcpyAsync((char *)dst + run_start, (char *)src + run_start,
                                n - run_start, cudaMemcpyHostToDevice, wf->copy_stream);
            }
        }
        // Record event so compute stream can wait
        if (wf->gpu_bufs[gpu_idx].upload_done) {
            cudaEventRecord(wf->gpu_bufs[gpu_idx].upload_done, wf->copy_stream);
        }
    } else
#endif
    {
        // Fallback: synchronous copy (should not normally happen)
        memcpy(dst, src, n);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    layer.h2d_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // NOTE: RAM buffer is NOT freed here — the async CUDA DMA may still be
    // reading from it.  The pipeline's H2D_FLIGHT → GPU_READY check will
    // free it once cudaEventQuery confirms the copy finished.

    {
        std::lock_guard<std::mutex> lock(wf->stats_mutex);
        wf->stats.bytes_uploaded_gpu += n;
        wf->stats.total_h2d_ms += layer.h2d_ms;
    }

    WF_TRACE(wf, "upload_to_gpu: layer %d: %zu bytes in %.2fms", layer_idx, n, layer.h2d_ms);
}

// Check if H2D async copy has completed (so we can free the RAM buffer safely).
// For synchronous memcpy fallback this always returns true.
static bool wf_h2d_complete(wf_context * wf, int layer_idx) {
    auto & layer = wf->layers[layer_idx];
    int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0) return true;
#ifdef WF_HAS_CUDA
    auto & slot = wf->gpu_bufs[gpu_idx];
    if (slot.upload_done) {
        cudaError_t err = cudaEventQuery(slot.upload_done);
        return (err == cudaSuccess);
    }
#endif
    return true;  // no CUDA or no event → memcpy was synchronous
}

#ifdef WF_HAS_GDS
// GDS path: NVMe → GPU directly
static void wf_gds_read_layer(wf_context * wf, int layer_idx) {
    auto & layer = wf->layers[layer_idx];
    int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0) return;

    void * gpu_ptr = wf->gpu_bufs[gpu_idx].ptr;
    size_t offset = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (auto & span : layer.spans) {
        // MoE filtering
        if (wf->params.moe_mode == WF_MOE_FETCH_ACTIVE && span.is_expert && layer.expert_filter_set) {
            bool active = false;
            for (int eid : layer.active_experts) {
                if (eid == span.expert_idx) { active = true; break; }
            }
            if (!active) {
                offset += span.nbytes;
                continue;
            }
        }

        // cuFileRead: NVMe → GPU VRAM via DMA
        if (span.is_expert && wf_cache_lookup(wf, layer_idx, span,
                (char *)gpu_ptr + offset, false)) {
            offset += span.nbytes;
            continue;
        }

        ssize_t n = cuFileRead(wf->cufile_handle, (char *)gpu_ptr + offset,
                               span.nbytes, span.file_offset, 0);
        if (n < 0 || (size_t)n != span.nbytes) {
            WF_ERR("GDS read failed: layer %d tensor %s: expected %zu got %zd",
                   layer_idx, span.tensor_name, span.nbytes, n);
        }
        offset += span.nbytes;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    layer.nvme_read_ms = ms;

    // Record upload event (GDS writes directly to GPU)
    if (wf->gpu_bufs[gpu_idx].upload_done) {
        cudaEventRecord(wf->gpu_bufs[gpu_idx].upload_done, wf->copy_stream);
    }

    WF_TRACE(wf, "gds_read: layer %d: %zu bytes in %.2fms (%.2f GB/s)",
             layer_idx, offset, ms, offset / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0));
}
#endif

// ---------------------------------------------------------------------------
// DirectStorage backend helpers
// ---------------------------------------------------------------------------

#ifdef WF_HAS_DSTORAGE

// Initialize D3D12 device, DirectStorage factory/queue, create shared D3D12
// buffers for CUDA interop, and set up fence → external semaphore sync.
// Called from wf_create() when mode is WF_MODE_DSTORAGE.
// On success, populates gpu_bufs[].ptr with valid CUDA device pointers.
// Returns true on success, false on failure (caller should fall back).
// 2026-06-01 R3 self-test (LLAMA_WF_TEST_DUAL_BUF=1, runs ONCE): validate the dual-use buffer
// mechanism the USER's design relies on — a D3D12 heap (>4GB) with placed <=4GB sub-resources,
// imported to CUDA as ONE contiguous pointer (cudaExternalMemoryHandleTypeD3D12Heap), writable
// across the 4GB boundary. Proves the foundation before the full re-architecture.
static void wf_selftest_dual_buffer(ID3D12Device * dev) {
#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
    const char * e = std::getenv("LLAMA_WF_TEST_DUAL_BUF");
    if (!(e && e[0] && e[0] != '0')) return;
    static bool done = false; if (done) return; done = true;

    const uint64_t SZ = (uint64_t)5 * 1024 * 1024 * 1024;          // 5 GB -> spans the 4GB boundary
    const uint64_t Q  = ((SZ / 2) + 65535) & ~((uint64_t)65535);   // 2 placed quarters (~2.5GB each, <=4GB)

    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = SZ;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ID3D12Heap * heap = nullptr;
    HRESULT hr = dev->CreateHeap(&hd, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) { fprintf(stderr, "*** DUALBUF_TEST: CreateHeap(5GB) FAILED 0x%08lx\n", (unsigned long)hr); fflush(stderr); return; }

    std::vector<ID3D12Resource*> placed;
    for (int q = 0; q < 2; q++) {
        uint64_t off = (uint64_t)q * Q; if (off >= SZ) break;
        uint64_t qs = (off + Q <= SZ) ? Q : (SZ - off);
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = qs; rd.Height = 1;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ID3D12Resource * res = nullptr;
        hr = dev->CreatePlacedResource(heap, off, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
        if (FAILED(hr)) { fprintf(stderr, "*** DUALBUF_TEST: CreatePlacedResource[%d]@%lluGB FAILED 0x%08lx\n", q, (unsigned long long)(off>>30), (unsigned long)hr); fflush(stderr); heap->Release(); return; }
        placed.push_back(res);
    }

    HANDLE sh = NULL;
    hr = dev->CreateSharedHandle(heap, nullptr, GENERIC_ALL, nullptr, &sh);
    if (FAILED(hr)) { fprintf(stderr, "*** DUALBUF_TEST: CreateSharedHandle(heap) FAILED 0x%08lx\n", (unsigned long)hr); fflush(stderr); for (auto*r:placed)r->Release(); heap->Release(); return; }

    cudaExternalMemoryHandleDesc md = {};
    md.type = cudaExternalMemoryHandleTypeD3D12Heap;
    md.handle.win32.handle = sh;
    md.size = SZ;
    cudaExternalMemory_t em = {};
    cudaError_t ce = cudaImportExternalMemory(&em, &md);
    if (ce != cudaSuccess) { fprintf(stderr, "*** DUALBUF_TEST: cudaImportExternalMemory(D3D12Heap) FAILED %s\n", cudaGetErrorString(ce)); fflush(stderr); CloseHandle(sh); for (auto*r:placed)r->Release(); heap->Release(); return; }

    void * ptr = nullptr;
    cudaExternalMemoryBufferDesc bd = {}; bd.offset = 0; bd.size = SZ; bd.flags = 0;
    ce = cudaExternalMemoryGetMappedBuffer(&ptr, em, &bd);
    if (ce != cudaSuccess) { fprintf(stderr, "*** DUALBUF_TEST: GetMappedBuffer FAILED %s\n", cudaGetErrorString(ce)); fflush(stderr); cudaDestroyExternalMemory(em); CloseHandle(sh); for (auto*r:placed)r->Release(); heap->Release(); return; }

    // Validate the CONTIGUOUS view across the 4GB boundary: memset the whole 5GB + a marker at >4GB.
    cudaError_t m1 = cudaMemset(ptr, 0xAB, SZ);
    cudaError_t s1 = cudaDeviceSynchronize();
    unsigned char host[16] = {0};
    cudaError_t c1 = cudaMemcpy(host, (char*)ptr + (uint64_t)4.5*1024*1024*1024, 16, cudaMemcpyDeviceToHost);
    fprintf(stderr, "*** DUALBUF_TEST: heap=5GB placed=%zu import=OK ptr=%p memset=%s sync=%s read@4.5GB=%s byte0=0x%02x (expect 0xab)\n",
            placed.size(), ptr, cudaGetErrorString(m1), cudaGetErrorString(s1), cudaGetErrorString(c1), host[0]);
    fflush(stderr);

    // STEP 2 validation: wrap the externally-owned dual-use ptr as a ggml CUDA buffer
    // (non-owning) and prove (a) get_base == ptr, (b) ggml_backend_buffer_free does NOT
    // cudaFree the ptr — i.e. the pointer is still live after the ggml buffer is freed.
    {
        ggml_backend_buffer_t gb = ggml_backend_cuda_buffer_from_ptr(0, ptr, SZ);
        void * base = gb ? ggml_backend_buffer_get_base(gb) : nullptr;
        bool base_ok = (base == ptr);
        // free the ggml buffer (non-owning -> must NOT cudaFree ptr)
        if (gb) ggml_backend_buffer_free(gb);
        // after free, the ptr must still be usable: write a distinct marker + read it back
        cudaError_t m2 = cudaMemset(ptr, 0xCD, 4096);
        cudaError_t s2 = cudaDeviceSynchronize();
        unsigned char host2[4] = {0};
        cudaError_t c2 = cudaMemcpy(host2, ptr, 4, cudaMemcpyDeviceToHost);
        fprintf(stderr, "*** DUALBUF_TEST: ggml-wrap base=%p base_ok=%d post-free-write=%s sync=%s byte0=0x%02x (expect 0xcd -> ptr survived non-owning free)\n",
                base, (int)base_ok, cudaGetErrorString(m2), cudaGetErrorString(s2), host2[0]);
        fflush(stderr);
    }

    cudaFree(ptr); cudaDestroyExternalMemory(em); CloseHandle(sh);
    for (auto*r:placed)r->Release(); heap->Release();
#else
    (void)dev;
#endif
}

#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
static bool wf_ds_fence_wait(wf_context * wf, uint64_t val, ID3D12Fence * fence = nullptr);  // defined later; used by the B1 probe
static bool wf_cuda_wait_ds_fence(wf_context * wf, uint64_t val, cudaStream_t stream);
#endif
// 2026-06-01 R3 BLOCKER B1 probe (design-review wf_86c0bf41): the dual-buffer self-test proved
// CUDA writes across the 4GB boundary, but NOT that DirectStorage writes into PLACED-in-heap
// resources at >4GB heap offsets land at the correct heap-absolute address. DS silently writes
// ZEROS for Destination.Buffer.Offset > ~4GB. Open question: when Destination.Buffer.Resource is a
// placed resource at heap_offset=4.25GB/6.375GB and Destination.Buffer.Offset=0, does the data land
// (b) at that placed resource's heap-absolute base [SAFE — DS-direct design is sound], or does DS
// mis-handle it producing zeros / wrapping to heap start [FAIL — need separate committed resources].
// This probe does a REAL DS write into each placed quarter IN ISOLATION (one Submit each per HARD
// LIMIT 2 = cannot write 2 resources/Submit) and reports which quarter index the bytes actually
// landed at, via the single contiguous CUDA view. Gated LLAMA_WF_TEST_DS_PLACED; runs once.
static void wf_selftest_ds_placed(wf_context * wf) {
#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
    const char * e = std::getenv("LLAMA_WF_TEST_DS_PLACED");
    if (!(e && e[0] && e[0] != '0')) return;
    static bool done = false; if (done) return; done = true;
    if (!wf->d3d_device || !wf->ds_queue) { fprintf(stderr, "*** DSPLACED_TEST: no device/queue\n"); fflush(stderr); return; }
    IDStorageFile * src = nullptr;
    if (!wf->ds_files.empty() && wf->ds_files[0]) src = wf->ds_files[0];
    else if (!wf->ds_layer_files.empty() && wf->ds_layer_files[0]) src = wf->ds_layer_files[0];
    if (!src) { fprintf(stderr, "*** DSPLACED_TEST: no source DS file\n"); fflush(stderr); return; }

    const uint64_t SZ = (uint64_t)8704 * 1024 * 1024;             // ~8.5 GB — matches the real ping-pong buffer
    const uint64_t Q  = ((SZ / 4) + 65535) & ~((uint64_t)65535);  // 4 placed quarters (~2.125GB each, <4GB)
    const uint32_t CHUNK = 65536;                                  // 64 KB known (non-zero) file content per quarter

    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = SZ;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ID3D12Heap * heap = nullptr;
    HRESULT hr = wf->d3d_device->CreateHeap(&hd, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) { fprintf(stderr, "*** DSPLACED_TEST: CreateHeap(8.5GB) FAILED 0x%08lx\n", (unsigned long)hr); fflush(stderr); return; }

    std::vector<ID3D12Resource*> placed;
    std::vector<uint64_t> placed_off;
    for (int q = 0; q < 4; q++) {
        uint64_t off = (uint64_t)q * Q; if (off >= SZ) break;
        uint64_t qs = (off + Q <= SZ) ? Q : (SZ - off);
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = qs; rd.Height = 1;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ID3D12Resource * res = nullptr;
        hr = wf->d3d_device->CreatePlacedResource(heap, off, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
        if (FAILED(hr)) { fprintf(stderr, "*** DSPLACED_TEST: CreatePlacedResource[%d]@%.2fGB FAILED 0x%08lx\n", q, off/1e9, (unsigned long)hr); fflush(stderr); for (auto*r:placed)r->Release(); heap->Release(); return; }
        placed.push_back(res); placed_off.push_back(off);
    }

    HANDLE sh = NULL;
    hr = wf->d3d_device->CreateSharedHandle(heap, nullptr, GENERIC_ALL, nullptr, &sh);
    if (FAILED(hr)) { fprintf(stderr, "*** DSPLACED_TEST: CreateSharedHandle FAILED\n"); fflush(stderr); for (auto*r:placed)r->Release(); heap->Release(); return; }
    cudaExternalMemoryHandleDesc md = {}; md.type = cudaExternalMemoryHandleTypeD3D12Heap; md.handle.win32.handle = sh; md.size = SZ;
    cudaExternalMemory_t em = {};
    cudaError_t ce = cudaImportExternalMemory(&em, &md);
    if (ce != cudaSuccess) { fprintf(stderr, "*** DSPLACED_TEST: import FAILED %s\n", cudaGetErrorString(ce)); fflush(stderr); CloseHandle(sh); for (auto*r:placed)r->Release(); heap->Release(); return; }
    void * ptr = nullptr;
    cudaExternalMemoryBufferDesc bd = {}; bd.offset = 0; bd.size = SZ; bd.flags = 0;
    ce = cudaExternalMemoryGetMappedBuffer(&ptr, em, &bd);
    if (ce != cudaSuccess) { fprintf(stderr, "*** DSPLACED_TEST: GetMappedBuffer FAILED %s\n", cudaGetErrorString(ce)); fflush(stderr); cudaDestroyExternalMemory(em); CloseHandle(sh); for (auto*r:placed)r->Release(); heap->Release(); return; }

    int n_ok = 0, n_test = 0;
    for (int q = 0; q < (int)placed.size(); q++) {
        // Isolate: zero the whole heap so any non-zero region after the write is THIS write's data.
        cudaMemset(ptr, 0, SZ); cudaDeviceSynchronize();
        // One DS request -> placed[q], Destination.Buffer.Offset=0 (local to the placed resource).
        DSTORAGE_REQUEST req = {};
        req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
        req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
        req.Source.File.Source        = src;
        req.Source.File.Offset        = 0;
        req.Source.File.Size          = CHUNK;
        req.Destination.Buffer.Resource = placed[q];
        req.Destination.Buffer.Offset   = 0;
        req.Destination.Buffer.Size     = CHUNK;
        req.UncompressedSize            = CHUNK;
        wf->ds_queue->Submit();                       // drain any prior state
        wf->ds_queue->EnqueueRequest(&req);
        uint64_t fv = ++wf->ds_fence_counter;
        wf->ds_queue->EnqueueSignal(wf->ds_fence, fv);
        wf->ds_queue->Submit();
        wf_ds_fence_wait(wf, fv);
        wf_cuda_wait_ds_fence(wf, fv, (cudaStream_t) 0);
        // Scan all quarter-start regions: where did the bytes actually land?
        int landed = -1; unsigned char lb = 0;
        for (int k = 0; k < (int)placed.size(); k++) {
            unsigned char hb[16] = {0};
            cudaMemcpy(hb, (char*)ptr + placed_off[k], 16, cudaMemcpyDeviceToHost);
            bool nz = false; for (int b = 0; b < 16; b++) if (hb[b]) { nz = true; break; }
            if (nz && landed < 0) { landed = k; lb = hb[0]; }
        }
        const char * verdict = (landed == q) ? "OK(heap-local)" : (landed < 0 ? "ZERO-FAIL(silent)" : "WRONG-PLACE(wrap)");
        fprintf(stderr, "*** DSPLACED_TEST: target_q=%d heapoff=%.2fGB -> landed_q=%d byte0=0x%02x %s\n",
                q, placed_off[q]/1e9, landed, lb, verdict);
        fflush(stderr);
        n_test++; if (landed == q) n_ok++;
    }
    fprintf(stderr, "*** DSPLACED_TEST: VERDICT %d/%d quarters heap-position-correct -> DS-direct placed-quarter design is %s\n",
            n_ok, n_test, (n_ok == n_test && n_test > 0) ? "SOUND (proceed STEP 3-6)" : "BROKEN (use separate committed resources per quarter)");
    fflush(stderr);

    cudaFree(ptr); cudaDestroyExternalMemory(em); CloseHandle(sh);
    for (auto*r:placed)r->Release(); heap->Release();
#else
    (void)wf;
#endif
}

// 2026-06-01 R3 DS-direct edit 4: create ONE permanent D3D12-heap-backed CUDA buffer via the B1-proven
// recipe. heap = ALIGN_UP(buf_size,256MB), carved into NQ=3 placed quarters (<4GB each, so DS
// Destination.Buffer.Offset stays <4GB), imported as ONE contiguous CUDA ptr. Out-params receive the ptr
// + the objects required to free it (teardown order matters; see wf_free_dual_use_heap). Frees partial
// state and returns false on any failure.
#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
static bool wf_make_dual_use_heap(ID3D12Device * dev, uint64_t buf_size,
                                  void ** out_ptr, cudaExternalMemory_t * out_em,
                                  ID3D12Heap ** out_heap, std::vector<ID3D12Resource*> * out_quarters,
                                  uint64_t * out_qsz, const char * tag) {
    const uint64_t MB256 = (uint64_t)256 * 1024 * 1024;
    const uint64_t HEAP  = ((buf_size + MB256 - 1) / MB256) * MB256;     // ALIGN_UP 256MB
    const int NQ = 3;
    const uint64_t QSZ = ((HEAP / NQ) + 65535) & ~((uint64_t)65535);     // ~2.92GB for 8.75GB/3, <4GB
    if (QSZ >= (uint64_t)4 * 1024 * 1024 * 1024) { WF_ERR("DS-direct(%s): quarter %.2fGB >= 4GB trap", tag, QSZ/1e9); return false; }

    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = HEAP;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ID3D12Heap * heap = nullptr;
    { size_t fb=0,tb=0; cudaMemGetInfo(&fb,&tb); fprintf(stderr,"*** DSD_VRAM(%s): before CreateHeap free=%.0fMB\n",tag,fb/1048576.0); fflush(stderr); }
    HRESULT hr = dev->CreateHeap(&hd, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) { WF_ERR("DS-direct(%s): CreateHeap(%.2fGB) 0x%08lx", tag, HEAP/1e9, (unsigned long)hr); return false; }
    { size_t fb=0,tb=0; cudaMemGetInfo(&fb,&tb); fprintf(stderr,"*** DSD_VRAM(%s): after CreateHeap(%.2fGB) free=%.0fMB\n",tag,HEAP/1e9,fb/1048576.0); fflush(stderr); }

    std::vector<ID3D12Resource*> q;
    for (int i = 0; i < NQ; i++) {
        uint64_t off = (uint64_t)i * QSZ; if (off >= HEAP) break;
        uint64_t qs = (off + QSZ <= HEAP) ? QSZ : (HEAP - off);
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = qs; rd.Height = 1;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ID3D12Resource * res = nullptr;
        hr = dev->CreatePlacedResource(heap, off, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
        if (FAILED(hr)) { WF_ERR("DS-direct(%s): CreatePlacedResource q%d 0x%08lx", tag, i, (unsigned long)hr); for (auto*r:q)r->Release(); heap->Release(); return false; }
        q.push_back(res);
    }

    HANDLE sh = NULL;
    hr = dev->CreateSharedHandle(heap, nullptr, GENERIC_ALL, nullptr, &sh);
    if (FAILED(hr)) { WF_ERR("DS-direct(%s): CreateSharedHandle 0x%08lx", tag, (unsigned long)hr); for (auto*r:q)r->Release(); heap->Release(); return false; }
    cudaExternalMemoryHandleDesc md = {}; md.type = cudaExternalMemoryHandleTypeD3D12Heap; md.handle.win32.handle = sh; md.size = HEAP;
    cudaExternalMemory_t em = {};
    cudaError_t ce = cudaImportExternalMemory(&em, &md);
    CloseHandle(sh);   // CUDA holds its own reference to the heap; the NT handle is not needed after import
    if (ce != cudaSuccess) { WF_ERR("DS-direct(%s): cudaImportExternalMemory %s", tag, cudaGetErrorString(ce)); for (auto*r:q)r->Release(); heap->Release(); return false; }
    { size_t fb=0,tb=0; cudaMemGetInfo(&fb,&tb); fprintf(stderr,"*** DSD_VRAM(%s): after cudaImportExternalMemory free=%.0fMB\n",tag,fb/1048576.0); fflush(stderr); }
    void * ptr = nullptr;
    cudaExternalMemoryBufferDesc bd = {}; bd.offset = 0; bd.size = HEAP; bd.flags = 0;
    ce = cudaExternalMemoryGetMappedBuffer(&ptr, em, &bd);
    if (ce != cudaSuccess) { WF_ERR("DS-direct(%s): GetMappedBuffer %s", tag, cudaGetErrorString(ce)); cudaDestroyExternalMemory(em); for (auto*r:q)r->Release(); heap->Release(); return false; }

    *out_ptr = ptr; *out_em = em; *out_heap = heap; *out_quarters = std::move(q); if (out_qsz) *out_qsz = QSZ;
    { size_t fb=0, tb=0; cudaMemGetInfo(&fb,&tb);
      fprintf(stderr, "*** DSD_VRAM: after %s heap (%.2fGB) -> VRAM free=%.0fMB/%.0fMB\n",
              tag, HEAP/1e9, fb/1048576.0, tb/1048576.0); fflush(stderr); }
    WF_LOG("DS-direct(%s): heap %.2fGB, %d quarters @ %.2fGB, ptr=%p", tag, HEAP/1e9, NQ, QSZ/1e9, ptr);
    return true;
}

// R3 DS-direct: free a dual-use heap. Order per the CUDA external-memory contract (validated by
// the working staging cleanup): mapped buffer -> external memory -> placed quarters -> heap.
static void wf_free_dual_use_heap(void * ptr, cudaExternalMemory_t em, ID3D12Heap * heap, std::vector<ID3D12Resource*> & quarters) {
    if (ptr) cudaFree(ptr);
    if (em)  cudaDestroyExternalMemory(em);
    for (auto * r : quarters) if (r) r->Release();
    quarters.clear();
    if (heap) heap->Release();
}

// Legacy env-mode prefetch relay. CUDA's free-byte query counts an imported
// view again on this WDDM driver; that was the basis for this historical
// single-compute fallback. It is not a second physical allocation: qualified
// NVML/canary/allocation tests permit the strict two-compute-heap mode below.
// DS writes into the prefetch heap's quarters, then D2D prefetch->compute. buf_size sizes the prefetch heap
// (can be < full layer to halve the doubled cost; the D2D then relays in chunks).
static bool wf_allocate_prefetch_heap(wf_context * wf, uint64_t buf_size) {
    void * pptr = nullptr;
    if (!wf_make_dual_use_heap(wf->d3d_device, buf_size, &pptr, &wf->prefetch_ext_mem,
                               &wf->prefetch_heap, &wf->prefetch_quarters, &wf->ds_quarter_size, "prefetch")) return false;
    const uint64_t MB256 = (uint64_t)256 * 1024 * 1024;
    const uint64_t HEAP  = ((buf_size + MB256 - 1) / MB256) * MB256;
    wf->prefetch_buf = pptr; wf->prefetch_buf_size = HEAP;
    cudaEventCreate(&wf->prefetch_done);
    // Zero the prefetch heap ONCE here (no DS activity yet -> no CUDA-memset-vs-DS race). Per-layer DS then
    // writes only the weight spans; identical per-layer tensor layouts keep the MATRIX_ROW_PADDING gaps zero.
    cudaMemset(pptr, 0, HEAP);
    cudaDeviceSynchronize();
    WF_LOG("DS-direct: prefetch D3D heap ready + zeroed (%.2fGB, QSZ=%.2fGB); compute stays cudaMalloc (no double-accounting)", HEAP/1e9, wf->ds_quarter_size/1e9);
    return true;
}
#endif // WF_HAS_CUDA && WF_HAS_DSTORAGE

// R3 DS-direct edit 4 validation (LLAMA_WF_TEST_DUAL_HEAPS): exercise wf_make_dual_use_heap +
// wf_free_dual_use_heap in ISOLATION — alloc one ~8.58GB heap, prove the contiguous CUDA view spans
// past 8GB, then free it (validates teardown order). Does not touch the live ds_direct path.
static void wf_selftest_dual_heaps(wf_context * wf) {
#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
    const char * e = std::getenv("LLAMA_WF_TEST_DUAL_HEAPS");
    if (!(e && e[0] && e[0] != '0')) return;
    static bool done = false; if (done) return; done = true;
    if (!wf->d3d_device) { fprintf(stderr, "*** DUALHEAP_TEST: no device\n"); fflush(stderr); return; }
    const uint64_t BUF = (uint64_t)8580 * 1024 * 1024;   // ~8.58GB (Kimi max layer)
    void * p = nullptr; cudaExternalMemory_t em = {}; ID3D12Heap * hp = nullptr; std::vector<ID3D12Resource*> q; uint64_t qsz = 0;
    if (!wf_make_dual_use_heap(wf->d3d_device, BUF, &p, &em, &hp, &q, &qsz, "selftest")) { fprintf(stderr, "*** DUALHEAP_TEST: alloc FAILED\n"); fflush(stderr); return; }
    const uint64_t MB256 = (uint64_t)256 * 1024 * 1024; const uint64_t HEAP = ((BUF + MB256 - 1) / MB256) * MB256;
    cudaError_t m = cudaMemset(p, 0xCD, HEAP);
    cudaError_t s = cudaDeviceSynchronize();
    unsigned char h[4] = {0};
    cudaError_t c = cudaMemcpy(h, (char*)p + (uint64_t)8 * 1024 * 1024 * 1024, 4, cudaMemcpyDeviceToHost);  // read past 8GB
    fprintf(stderr, "*** DUALHEAP_TEST: heap=%.2fGB quarters=%zu qsz=%.2fGB ptr=%p memset=%s sync=%s read@8GB=%s byte0=0x%02x (expect 0xcd)\n",
            HEAP/1e9, q.size(), qsz/1e9, p, cudaGetErrorString(m), cudaGetErrorString(s), cudaGetErrorString(c), h[0]);
    fflush(stderr);
    wf_free_dual_use_heap(p, em, hp, q);
    fprintf(stderr, "*** DUALHEAP_TEST: freed (teardown order mapped->ext_mem->quarters->heap)\n");
    fflush(stderr);
#else
    (void)wf;
#endif
}

// 2026-06-01 R3: create the STATEFUL DS objects (fence, CUDA semaphore, queue) — re-created
// PER REQUEST even when the inert infra (device/factory/files/staging) is persisted, because
// reusing the stateful queue/fence stale in-flight state is what wedges the GPU ("GPU is lost").
static bool wf_dstorage_create_stateful(wf_context * wf) {
    HRESULT hr;
    hr = wf->d3d_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&wf->ds_fence));
    if (FAILED(hr)) {
        WF_ERR("DirectStorage: CreateFence failed: 0x%08lx", (unsigned long)hr);
        return false;
    }
    hr = wf->d3d_device->CreateSharedHandle(wf->ds_fence, nullptr, GENERIC_ALL, nullptr, &wf->ds_fence_handle);
    if (FAILED(hr)) {
        WF_ERR("DirectStorage: CreateSharedHandle (fence) failed: 0x%08lx", (unsigned long)hr);
        wf->ds_fence->Release(); wf->ds_fence = nullptr;
        return false;
    }
    cudaExternalSemaphoreHandleDesc semDesc = {};
    semDesc.type                = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semDesc.handle.win32.handle = wf->ds_fence_handle;
    cudaError_t cerr = cudaImportExternalSemaphore(&wf->ds_ext_sem, &semDesc);
    if (cerr != cudaSuccess) {
        WF_ERR("DirectStorage: cudaImportExternalSemaphore failed: %s", cudaGetErrorString(cerr));
        CloseHandle(wf->ds_fence_handle); wf->ds_fence_handle = NULL;
        wf->ds_fence->Release(); wf->ds_fence = nullptr;
        return false;
    }
    wf->ds_fence_counter.store(0);
    wf->ds_layer_fence_val.assign(wf->params.n_layers, 0);
    wf->ds_queue_cap = DSTORAGE_MAX_QUEUE_CAPACITY;
    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity   = (UINT16)wf->ds_queue_cap;
    queueDesc.Priority   = DSTORAGE_PRIORITY_HIGH;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device     = wf->d3d_device;
    hr = wf->ds_factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&wf->ds_queue));
    if (FAILED(hr)) {
        WF_ERR("DirectStorage: CreateQueue failed: 0x%08lx", (unsigned long)hr);
        cudaDestroyExternalSemaphore(wf->ds_ext_sem); wf->ds_ext_sem = {};
        CloseHandle(wf->ds_fence_handle); wf->ds_fence_handle = NULL;
        wf->ds_fence->Release(); wf->ds_fence = nullptr;
        return false;
    }
    wf->ds_queue2 = nullptr;
    return true;
}

// Initialize per-request state that is deliberately not part of the persistent
// DirectStorage cache. gpu_bufs belongs to each new wf_context, so capacities
// and CUDA events must be recreated even when the inert DS objects are reused.
static bool wf_dstorage_init_gpu_slots(wf_context * wf, size_t buf_size) {
    for (int i = 0; i < (int) wf->gpu_bufs.size(); ++i) {
        auto & slot = wf->gpu_bufs[i];
        slot.ptr       = nullptr;  // supplied later by wf_set_gpu_buf_ptr()
        slot.capacity  = buf_size;
        slot.layer_idx = -1;
        slot.owns_ptr  = false;
        slot.in_use.store(false);

        cudaError_t err = cudaEventCreate(&slot.upload_done);
        if (err == cudaSuccess) {
            err = cudaEventCreate(&slot.compute_done);
        }
        if (err != cudaSuccess) {
            WF_ERR("DirectStorage: CUDA event creation failed for GPU buffer %d: %s",
                   i, cudaGetErrorString(err));
            for (auto & cleanup_slot : wf->gpu_bufs) {
                if (cleanup_slot.upload_done) {
                    cudaEventDestroy(cleanup_slot.upload_done);
                    cleanup_slot.upload_done = nullptr;
                }
                if (cleanup_slot.compute_done) {
                    cudaEventDestroy(cleanup_slot.compute_done);
                    cleanup_slot.compute_done = nullptr;
                }
                cleanup_slot.capacity = 0;
            }
            return false;
        }

        WF_LOG("DS GPU buffer %d: %.2f MB (deferred, ggml will provide ptr)",
               i, buf_size / (1024.0 * 1024.0));
    }
    return true;
}

// Persistent inert DS infra (LLAMA_WF_PERSIST_INFRA): device/factory/files/staging allocated
// ONCE and reused across requests (the per-request re-create of these corrupts the 2nd+ prefill).
// Checkpoint identity is separate from persistent GPU staging ownership.
// File metadata is captured from the opened handle, never inferred from size alone.
struct wf_ds_file_identity {
    std::string path;
    BY_HANDLE_FILE_INFORMATION info = {};
};

static bool wf_ds_same_file(const wf_ds_file_identity & a, const wf_ds_file_identity & b) {
    return a.path == b.path &&
        a.info.dwVolumeSerialNumber == b.info.dwVolumeSerialNumber &&
        a.info.nFileIndexHigh == b.info.nFileIndexHigh &&
        a.info.nFileIndexLow == b.info.nFileIndexLow &&
        a.info.nFileSizeHigh == b.info.nFileSizeHigh &&
        a.info.nFileSizeLow == b.info.nFileSizeLow &&
        a.info.ftLastWriteTime.dwHighDateTime == b.info.ftLastWriteTime.dwHighDateTime &&
        a.info.ftLastWriteTime.dwLowDateTime == b.info.ftLastWriteTime.dwLowDateTime;
}

static bool wf_ds_wide_path(const char * path, std::vector<wchar_t> & wide) {
    if (!path || !path[0]) return false;
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
    if (count <= 0) return false;
    wide.resize(count);
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide.data(), count) == count;
}

static bool wf_ds_path_identity(const char * path, wf_ds_file_identity & identity) {
    std::vector<wchar_t> wide;
    if (!wf_ds_wide_path(path, wide)) return false;
    HANDLE file = CreateFileW(wide.data(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    identity.path = path;
    const bool ok = GetFileInformationByHandle(file, &identity.info) != 0;
    CloseHandle(file);
    return ok;
}

struct wf_persist_infra {
    IDXGIFactory4 *      dxgi = nullptr;
    ID3D12Device *       device = nullptr;
    IDStorageFactory *   factory = nullptr;
    std::vector<IDStorageFile*> files;
    ID3D12Resource *     stg_d3d = nullptr;
    cudaExternalMemory_t stg_ext_mem = {};
    HANDLE               stg_shared = NULL;
    void *               stg_ptr = nullptr;
    uint64_t             stg_size = 0;
    bool                 valid = false;
    wf_context *         active_owner = nullptr;
};
static wf_persist_infra g_wf_persist;
static std::vector<wf_ds_file_identity> g_wf_persist_file_ids;

static std::vector<const char *> wf_ds_requested_paths(const wf_context * wf) {
    std::vector<const char *> paths = { wf->params.model_path };
    if (wf->params.n_split_files < 0 || (wf->params.n_split_files && !wf->params.split_paths)) return {};
    for (int i = 0; i < wf->params.n_split_files; ++i) paths.push_back(wf->params.split_paths[i]);
    return paths;
}

// Only called with no active cache owner. Open all replacements first, then swap
// references atomically; a failure preserves the old cache but refuses this load.
static bool wf_persist_refresh_files(wf_context * wf) {
    const auto paths = wf_ds_requested_paths(wf);
    if (paths.empty()) return false;
    std::vector<wf_ds_file_identity> current(paths.size());
    bool same = paths.size() == g_wf_persist_file_ids.size() && paths.size() == g_wf_persist.files.size();
    for (size_t i = 0; i < paths.size(); ++i) {
        if (!wf_ds_path_identity(paths[i], current[i])) {
            WF_ERR("DirectStorage: cannot establish requested file identity at index %zu", i);
            return false;
        }
        if (same && !wf_ds_same_file(current[i], g_wf_persist_file_ids[i])) same = false;
    }
    if (same) return true;
    std::vector<IDStorageFile *> opened;
    bool ok = true;
    for (size_t i = 0; i < paths.size(); ++i) {
        std::vector<wchar_t> wide;
        IDStorageFile * file = nullptr;
        wf_ds_file_identity actual;
        actual.path = paths[i];
        if (!wf_ds_wide_path(paths[i], wide) ||
            FAILED(g_wf_persist.factory->OpenFile(wide.data(), IID_PPV_ARGS(&file)))) {
            ok = false;
        } else {
            opened.push_back(file);
            if (FAILED(file->GetFileInformation(&actual.info)) || !wf_ds_same_file(actual, current[i])) ok = false;
        }
        if (!ok) break;
    }
    if (!ok) {
        for (auto * file : opened) file->Release();
        WF_ERR("DirectStorage: replacement file identity/open failed; refusing cached file reuse");
        return false;
    }
    for (auto * file : g_wf_persist.files) if (file) file->Release();
    g_wf_persist.files = std::move(opened);
    g_wf_persist_file_ids = std::move(current);
    fprintf(stderr, "*** WF_PERSIST_FILES: refreshed %zu checkpoint handles; GPU staging retained\n", paths.size());
    fflush(stderr);
    return true;
}

static bool wf_persist_infra_enabled() {
    const char * e = std::getenv("LLAMA_WF_PERSIST_INFRA");
    // A same-sized D3D12 external-memory re-import can recycle the just-freed
    // CUDA VA while its old driver-side association is still stale. Keep the
    // complete staging quartet alive by default; explicit 0 is diagnostic-only.
    return !(e && e[0] && e[0] == '0');
}

static uint64_t wf_dstorage_requested_staging_size(size_t buf_size, const wf_params &params) {
    if (params.ds_staging_limit_bytes) {
        return llama_wf_bounded::align(std::min<uint64_t>(buf_size,params.ds_staging_limit_bytes),65536);
    }
    uint64_t max_size = (uint64_t) 3500 * 1024 * 1024;
    const char * e = std::getenv("LLAMA_DS_STAGING_MB");
    if (e && e[0]) {
        const int mb = std::atoi(e);
        if (mb > 0) max_size = (uint64_t) mb * 1024 * 1024;
    }
    const uint64_t size = std::min((uint64_t) buf_size, max_size);
    return (size + 65535) & ~((uint64_t) 65535);
}

// EVICT the cached STAGING (the only VRAM the persist cache holds) while keeping the cheap
// device/factory/files alive. Called when a request cannot reuse the cached staging — e.g. MAGI-2
// finishes its 3.42 GB preview pass and the 1080p refiner asks for a 0.59 GB buffer. Without this
// the big buffer sits idle for the rest of the generation and the refiner OOMs beside it. Keeping
// device/factory/files cached preserves the reason the cache exists (re-creating THOSE per request
// corrupted the 2nd+ prefill); only the buffer is dropped.
static void wf_persist_staging_evict(const char * why) {
    if (!g_wf_persist.stg_d3d && !g_wf_persist.stg_ptr) return;
    const double gb = g_wf_persist.stg_size / 1e9;
    // wf_destroy has quiesced the previous owner. DestroyExternalMemory does
    // NOT free GetMappedBuffer's CUDA mapping: forgetting it leaks every
    // differently-sized staging heap across DiT / LoRA / VAE transitions.
    if (g_wf_persist.stg_ptr) {
        cudaError_t err = cudaFree(g_wf_persist.stg_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "*** WF_PERSIST_INFRA: cudaFree(staging): %s\n",
                    cudaGetErrorString(err));
        }
    }
    if (g_wf_persist.stg_ext_mem) cudaDestroyExternalMemory(g_wf_persist.stg_ext_mem);
    if (g_wf_persist.stg_shared) CloseHandle(g_wf_persist.stg_shared);
    if (g_wf_persist.stg_d3d)    g_wf_persist.stg_d3d->Release();
    g_wf_persist.stg_ext_mem = {};
    g_wf_persist.stg_shared  = NULL;
    g_wf_persist.stg_d3d     = nullptr;
    g_wf_persist.stg_ptr     = nullptr;
    g_wf_persist.stg_size    = 0;
    g_wf_persist.valid       = false;   // must be repopulated before the cache is trusted again
    fprintf(stderr, "*** WF_PERSIST_INFRA: EVICTED cached staging (%.2f GB) — %s\n", gb, why);
    fflush(stderr);
}

// DS4COMBO graceful release (close and keep-alive suspend share it): drop the
// process-global cache too — staging heap, DS files, DS factory, D3D12 device,
// DXGI factory — so nothing DirectStorage/D3D12 stays alive. Only valid with no
// live wf_context owning the cache. The next wf_create rebuilds it from scratch.
static int wf_persist_shutdown_impl(const char * why) {
    if (g_wf_persist.active_owner) {
        fprintf(stderr, "*** WF_PERSIST_SHUTDOWN refused: cache still has an active owner (%s)\n", why);
        fflush(stderr);
        return 0;
    }
    wf_persist_staging_evict(why);
    for (auto * file : g_wf_persist.files) if (file) file->Release();
    g_wf_persist.files.clear();
    g_wf_persist_file_ids.clear();
    unsigned long factory_refs = 0, device_refs = 0, dxgi_refs = 0;
    if (g_wf_persist.factory) factory_refs = g_wf_persist.factory->Release();
    if (g_wf_persist.device)  device_refs  = g_wf_persist.device->Release();
    if (g_wf_persist.dxgi)    dxgi_refs    = g_wf_persist.dxgi->Release();
    g_wf_persist.factory = nullptr;
    g_wf_persist.device  = nullptr;
    g_wf_persist.dxgi    = nullptr;
    g_wf_persist.valid   = false;
    fprintf(stderr, "DS4COMBO_WF_PERSIST_SHUTDOWN why=%s ds_factory_refs=%lu d3d12_device_refs=%lu dxgi_refs=%lu\n",
            why, factory_refs, device_refs, dxgi_refs);
    fflush(stderr);
    return 1;
}

static bool wf_persist_infra_matches(wf_context * wf, size_t buf_size) {
    // A live DiT streamer can coexist with a LoRA/VAE streamer. Its mapping
    // cannot be reused or evicted merely because the second asks for a heap.
    if (g_wf_persist.active_owner) return false;
    if (!g_wf_persist.valid || wf->ds_direct_mode) {
        // DS-direct never uses the staging tier; if a stale buffer is cached, free it now rather
        // than letting it hold VRAM for the rest of the process.
        if (wf->ds_direct_mode) wf_persist_staging_evict("ds_direct mode does not use staging");
        return false;
    }

    const uint64_t expected = wf_dstorage_requested_staging_size(buf_size,wf->params);
    const bool objects_valid =
        g_wf_persist.dxgi && g_wf_persist.device && g_wf_persist.factory &&
        g_wf_persist.stg_d3d && g_wf_persist.stg_ext_mem &&
        g_wf_persist.stg_shared && g_wf_persist.stg_ptr &&
        g_wf_persist.files.size() == (size_t) (wf->params.n_split_files + 1);
    const uint64_t resource_size = g_wf_persist.stg_d3d ? g_wf_persist.stg_d3d->GetDesc().Width : 0;
    if (!objects_valid || g_wf_persist.stg_size != expected || resource_size != expected ||
            g_wf_persist.device->GetDeviceRemovedReason() != S_OK) {
        fprintf(stderr,
                "*** WF_PERSIST_INFRA: cached staging rejected "
                "(objects=%d cached=%llu requested=%llu resource=%llu files=%zu/%d)\n",
                (int) objects_valid,
                (unsigned long long) g_wf_persist.stg_size,
                (unsigned long long) expected,
                (unsigned long long) resource_size,
                g_wf_persist.files.size(), wf->params.n_split_files + 1);
        fflush(stderr);
        // The cached buffer cannot serve this request, so it is dead weight from here on: drop it
        // BEFORE the caller allocates its own, so the two never coexist. This is what lets the
        // MAGI-2 1080p refiner have the card after the preview pass.
        wf_persist_staging_evict("size/objects mismatch — cached staging cannot serve this request");
        return false;
    }

    cudaPointerAttributes attrs = {};
    const cudaError_t ptr_err = cudaPointerGetAttributes(&attrs, g_wf_persist.stg_ptr);
    if (ptr_err != cudaSuccess || attrs.type != cudaMemoryTypeDevice ||
            attrs.devicePointer != g_wf_persist.stg_ptr) {
        fprintf(stderr,
                "*** WF_PERSIST_INFRA: cached CUDA staging pointer rejected: %s "
                "(type=%d device_ptr=%p expected=%p)\n",
                cudaGetErrorString(ptr_err), (int) attrs.type,
                attrs.devicePointer, g_wf_persist.stg_ptr);
        fflush(stderr);
        cudaGetLastError();
        return false;
    }
    return true;
}

static bool wf_dstorage_init(wf_context * wf, size_t buf_size) {
    HRESULT hr;
    // 2026-06-01 R3 (USER permanent-buffer design): if the inert infra is cached, REUSE it
    // (device+factory+files+staging) and only re-create the stateful queue/fence. This kills the
    // per-request CreateCommittedResource/cudaImportExternalMemory churn that corrupts 2nd+ prefills.
    if (!wf->required_dstorage && wf_persist_infra_enabled() && wf_persist_infra_matches(wf, buf_size)) {
        hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        wf->ds_com_inited = SUCCEEDED(hr);
        if (!wf_persist_refresh_files(wf)) return false;
        wf->dxgi_factory = g_wf_persist.dxgi;       if (wf->dxgi_factory) wf->dxgi_factory->AddRef();
        wf->d3d_device   = g_wf_persist.device;     if (wf->d3d_device)   wf->d3d_device->AddRef();
        wf->ds_factory   = g_wf_persist.factory;    if (wf->ds_factory)   wf->ds_factory->AddRef();
        wf->ds_files     = g_wf_persist.files;       for (auto * f : wf->ds_files) if (f) f->AddRef();
        wf->ds_stage.d3d_resource = g_wf_persist.stg_d3d;     if (wf->ds_stage.d3d_resource) wf->ds_stage.d3d_resource->AddRef();
        wf->ds_stage.ext_mem      = g_wf_persist.stg_ext_mem;
        wf->ds_stage.shared_handle= g_wf_persist.stg_shared;
        wf->ds_stage.cuda_ptr     = g_wf_persist.stg_ptr;
        wf->ds_stage.size         = g_wf_persist.stg_size;
        wf->ds_stage.persistent   = true;           // skip the CUDA staging free in wf_destroy
        g_wf_persist.active_owner = wf;
        wf->ds_staging_size       = g_wf_persist.stg_size;
        fprintf(stderr, "*** WF_PERSIST_INFRA: REUSED cached device/factory/files/staging (%.2f GB), re-creating queue/fence only\n",
                g_wf_persist.stg_size / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
        {   // DIAG: pin the reuse crash — is the cached CUDA staging mapping still valid? + VRAM
            size_t fr = 0, to = 0; cudaMemGetInfo(&fr, &to);
            cudaGetLastError();
            cudaError_t me = cudaMemset(wf->ds_stage.cuda_ptr, 0, 4096);
            cudaError_t se = cudaDeviceSynchronize();
            fprintf(stderr, "*** REUSE_PROBE: free=%zuMB memset=%s sync=%s stg_ptr=%p d3d=%p files=%zu\n",
                    fr / (1024 * 1024), cudaGetErrorString(me), cudaGetErrorString(se),
                    wf->ds_stage.cuda_ptr, (void*)wf->ds_stage.d3d_resource, wf->ds_files.size());
            fflush(stderr);
        }
        // The persistent cache contains no wf_buffer_slot state. The old reuse
        // path returned before the normal slot setup below, leaving capacity=0
        // and upload_done/compute_done=null on the second request.
        if (!wf_dstorage_init_gpu_slots(wf, buf_size)) return false;
        if (!wf_dstorage_create_stateful(wf)) return false;
        wf->ds_initialized = true;
        return true;
    }

    // --- COM ---
    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    wf->ds_com_inited = SUCCEEDED(hr);

    // --- DXGI factory → find GPU adapter ---
    hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&wf->dxgi_factory));
    if (FAILED(hr)) {
        WF_ERR("DirectStorage: CreateDXGIFactory2 failed: 0x%08lx", (unsigned long)hr);
        return false;
    }

    // Match the DXGI adapter to the CUDA device by LUID so DirectStorage
    // targets the same GPU that CUDA is using (important in multi-GPU systems).
    LUID target_luid = {};
#if WF_HAS_CUDA
    {
        cudaDeviceProp props;
        const int device = wf->required_dstorage ? wf->cuda_device_index : 0;
        const cudaError_t result = cudaGetDeviceProperties(&props, device);
        if (wf->required_dstorage && result != cudaSuccess) {
            WF_ERR("Required DirectStorage: cannot resolve target CUDA adapter");
            return false;
        }
        memcpy(&target_luid, props.luid, sizeof(LUID));
        if (wf->required_dstorage && !target_luid.HighPart && !target_luid.LowPart) {
            WF_ERR("Required DirectStorage: target CUDA adapter has no Windows LUID");
            return false;
        }
        WF_LOG("DirectStorage: CUDA device '%s', LUID=%08x-%08x",
               props.name, target_luid.HighPart, target_luid.LowPart);
    }
#endif

    IDXGIAdapter1 * adapter = nullptr;
    IDXGIAdapter1 * fallback_adapter = nullptr;
    for (UINT i = 0;; i++) {
        const HRESULT enumerated = wf->dxgi_factory->EnumAdapters1(i, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(enumerated) || !adapter) {
            if (fallback_adapter) fallback_adapter->Release();
            return false;
        }
        DXGI_ADAPTER_DESC1 desc {};
        const HRESULT described = adapter->GetDesc1(&desc);
        if (wf->required_dstorage && FAILED(described)) {
            adapter->Release();
            if (fallback_adapter) fallback_adapter->Release();
            return false;
        }
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }
        // Check LUID match
        if (desc.AdapterLuid.HighPart == target_luid.HighPart &&
            desc.AdapterLuid.LowPart == target_luid.LowPart) {
            hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&wf->d3d_device));
            adapter->Release();
            if (SUCCEEDED(hr)) {
                WF_LOG("DirectStorage: matched CUDA device by LUID (adapter %u: %ls)", i, desc.Description);
                break;
            }
        } else {
            // Keep first non-software adapter as fallback
            if (!fallback_adapter) { fallback_adapter = adapter; adapter = nullptr; }
            else { adapter->Release(); }
        }
    }
    // Fallback: use first available adapter if LUID match failed
    if (!wf->required_dstorage && !wf->d3d_device && fallback_adapter) {
        DXGI_ADAPTER_DESC1 desc;
        fallback_adapter->GetDesc1(&desc);
        hr = D3D12CreateDevice(fallback_adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&wf->d3d_device));
        WF_LOG("DirectStorage: LUID match failed, using fallback adapter: %ls", desc.Description);
    }
    if (fallback_adapter) fallback_adapter->Release();
    if (!wf->d3d_device) {
        WF_ERR("DirectStorage: no D3D12 device found");
        wf->dxgi_factory->Release(); wf->dxgi_factory = nullptr;
        return false;
    }

    // --- DirectStorage factory ---
    if(wf->params.ds_submit_threads){
        if(!wf->required_dstorage||(wf->params.ds_submit_threads!=1&&wf->params.ds_submit_threads!=2&&wf->params.ds_submit_threads!=4)){
            WF_ERR("Explicit DirectStorage submit thread profile is outside strict TEST admission");return false;
        }
        DSTORAGE_CONFIGURATION configuration{};configuration.NumSubmitThreads=wf->params.ds_submit_threads;
        hr=DStorageSetConfiguration(&configuration);
        if(FAILED(hr)){WF_ERR("Required DStorageSetConfiguration failed:0x%08lx",(unsigned long)hr);return false;}
        WF_ERR("[dsv41-dstorage-config] submit_threads=%u applied_before_factory=1",wf->params.ds_submit_threads);
    }
    hr = DStorageGetFactory(IID_PPV_ARGS(&wf->ds_factory));
    if (FAILED(hr)) {
        WF_ERR("DirectStorage: DStorageGetFactory failed: 0x%08lx", (unsigned long)hr);
        wf->d3d_device->Release(); wf->d3d_device = nullptr;
        wf->dxgi_factory->Release(); wf->dxgi_factory = nullptr;
        return false;
    }
    // 2026-06-01 R3 coherence fix (LLAMA_WF_PERSIST_FACTORY, default OFF for A/B): DStorageGetFactory
    // returns a process SINGLETON; releasing it per request (wf_destroy:2565) can drive its refcount
    // to 0 and DESTROY+recreate the DS factory (and its GPU upload engine) every request, which is a
    // prime suspect for the 2nd+ consecutive prefill corrupting. Hold ONE permanent reference so the
    // refcount never reaches 0 => the factory persists across requests (per-request Get/Release still
    // oscillates harmlessly above this floor). Verify by behavior-diff (multi-request coherence).
    {
        static IDStorageFactory * g_ds_factory_keepalive = nullptr;
        const char * pf_env = std::getenv("LLAMA_WF_PERSIST_FACTORY");
        const bool persist_factory = pf_env && pf_env[0] && pf_env[0] != '0';
        if (persist_factory && !g_ds_factory_keepalive && wf->ds_factory) {
            wf->ds_factory->AddRef();
            g_ds_factory_keepalive = wf->ds_factory;
            fprintf(stderr, "*** WF_PERSIST_FACTORY: holding permanent DS factory ref %p (kept alive across requests)\n",
                    (void*)wf->ds_factory);
            fflush(stderr);
        }
    }

    wf_selftest_dual_buffer(wf->d3d_device);

    // Cap DirectStorage's internal staging buffer for machines with tighter RAM budgets.
    // Expose this as a tuning knob because small caps can throttle warm sequential reads.
    {
        uint64_t factory_staging_mb = 256;
        const char * factory_stg_env = std::getenv("LLAMA_DS_FACTORY_STAGING_MB");
        if (factory_stg_env && factory_stg_env[0]) {
            int mb = std::atoi(factory_stg_env);
            if (mb > 0) {
                factory_staging_mb = (uint64_t) mb;
            }
        }
        const HRESULT staging_result = wf->ds_factory->SetStagingBufferSize(UINT32(factory_staging_mb * 1024 * 1024));
        if (wf->required_dstorage && FAILED(staging_result)) {
            WF_ERR("Required DirectStorage: factory staging-size request failed: 0x%08lx", (unsigned long)staging_result);
            return false;
        }
        WF_LOG("DirectStorage: factory staging buffer capped to %llu MB",
               (unsigned long long) factory_staging_mb);
    }

    // --- Open DirectStorage file handles ---
    {
        auto open_ds_file = [&](const char * path) -> IDStorageFile * {
            int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
            std::vector<wchar_t> wpath(wlen);
            MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);
            IDStorageFile * f = nullptr;
            hr = wf->ds_factory->OpenFile(wpath.data(), IID_PPV_ARGS(&f));
            if (FAILED(hr)) {
                WF_ERR("DirectStorage: OpenFile failed for %s: 0x%08lx", path, (unsigned long)hr);
                return nullptr;
            }
            return f;
        };

        // Main file
        IDStorageFile * f = open_ds_file(wf->params.model_path);
        if (!f) {
            wf->ds_factory->Release(); wf->ds_factory = nullptr;
            wf->d3d_device->Release(); wf->d3d_device = nullptr;
            wf->dxgi_factory->Release(); wf->dxgi_factory = nullptr;
            return false;
        }
        wf->ds_files.push_back(f);

        // Split files
        for (int i = 0; i < wf->params.n_split_files && wf->params.split_paths; i++) {
            IDStorageFile * sf = open_ds_file(wf->params.split_paths[i]);
            if (sf) wf->ds_files.push_back(sf);
            else if (wf->required_dstorage) return false;
        }

        // Check for per-layer extracted files (LLAMA_PREFILL_LAYER_DIR env var)
        const char * layer_dir_env = std::getenv("LLAMA_PREFILL_LAYER_DIR");
        if (layer_dir_env && layer_dir_env[0]) {
            int n_layers = (int)wf->layers.size();
            wf->ds_layer_files.resize(n_layers, nullptr);
            int opened = 0;
            for (int il = 0; il < n_layers; il++) {
                char path[512];
                snprintf(path, sizeof(path), "%s/layer_%03d.bin", layer_dir_env, il);
                IDStorageFile * lf = open_ds_file(path);
                if (lf) {
                    wf->ds_layer_files[il] = lf;
                    opened++;
                }
            }
            if (opened > 0) {
                wf->use_layer_files = true;
                fprintf(stderr, "[weight-fetcher] Per-layer files: %d/%d layers from %s\n",
                        opened, n_layers, layer_dir_env);
            }
        }
    }

    // --- Compute CUDA buffers: defer allocation ---
    // The ggml compute buffers (gpu_buf_a/b) will be allocated separately by
    // prefill_allocate_gpu_buffers and set via wf_set_gpu_buf_ptr.
    // Don't cudaMalloc here — it wastes VRAM that gets immediately overridden.
    // Only create the CUDA events needed for DMA/compute synchronization.
    if (!wf_dstorage_init_gpu_slots(wf, buf_size)) return false;

    // Strict direct mode owns two imported compute heaps. The older env-mode
    // path below owns a prefetch heap and relays into its normal compute bank.
#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
    if(wf->params.ds_compute_direct){
        size_t free=0,total=0;
        // At this point no imported bank exists, so CUDA's free-byte query
        // still reflects physical availability. Count the two shared heaps
        // once; their permanent CUDA views alias them, not new allocations.
        const uint64_t required=llama_wf_bounded::add(llama_wf_bounded::shared_heap_physical_bytes(buf_size,uint32_t(wf->params.n_gpu_buffers)),512ull*1024*1024);
        if(cudaMemGetInfo(&free,&total)!=cudaSuccess||required>free){
            WF_ERR("DS4PERF_DIRECT_CAPACITY_REFUSED required_bytes=%llu cuda_free_bytes=%zu physical_shared_heaps=2 no_banks_allocated=1",(unsigned long long)required,free);return false;
        }
        wf->ds_compute_banks.resize(wf->params.n_gpu_buffers);
        for(size_t i=0;i<wf->ds_compute_banks.size();++i){auto &bank=wf->ds_compute_banks[i];
            const std::string label="compute-"+std::to_string(i);
            if(!wf_make_dual_use_heap(wf->d3d_device,buf_size,&bank.ptr,&bank.external,&bank.heap,&bank.quarters,&bank.quarter_bytes,label.c_str()))return false;
            wf->gpu_bufs[i].ptr=bank.ptr;wf->gpu_bufs[i].owns_ptr=false;
            if(i&&bank.quarter_bytes!=wf->ds_quarter_size)return false;
            wf->ds_quarter_size=bank.quarter_bytes;
            if(cudaMemset(bank.ptr,0,buf_size)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return false;
        }
        WF_ERR("DS4PERF_DIRECT_HEAPS buffers=%zu compute_capacity=%zu staging_bytes=0 persistent_provider_owned=1",wf->ds_compute_banks.size(),buf_size);
    }else if (wf->ds_direct_mode) {
        // Prefetch heap sized to the full layer (single D2D). If VRAM-tight, LLAMA_WF_DS_PREFETCH_PCT can shrink it.
        uint64_t pf_bytes = buf_size;
        const char * pfp = std::getenv("LLAMA_WF_DS_PREFETCH_PCT");
        if (pfp && pfp[0]) { int pct = std::atoi(pfp); if (pct >= 10 && pct <= 100) pf_bytes = (buf_size * (uint64_t)pct) / 100; }
        if (!wf_allocate_prefetch_heap(wf, pf_bytes)) {
            WF_ERR("DS-direct: prefetch heap allocation FAILED; falling back to legacy staging path");
            wf->ds_direct_mode = false;
        }
    }
#endif

    // --- Create single D3D12 staging resource (≤3.5 GB) ---  [DS-direct edit 7: skipped when ds_direct_mode]
    // DS silently produces zeros for Destination.Buffer.Offset > ~4 GB and
    // cannot write to a second D3D12 resource in the same submission.
    // Solution: one 3.5 GB staging buffer, reused across waves and buffers.
    // Layers larger than 3.5 GB transfer in multiple waves with D2D copy
    // between each wave.
    if (!wf->ds_direct_mode) {
        // LLAMA_DS_STAGING_MB: override staging buffer size (default 3500 MB)
        const char * stg_env = std::getenv("LLAMA_DS_STAGING_MB");
        if (stg_env && stg_env[0]) {
            WF_LOG("DS staging size override: %s MB", stg_env);
        }
        uint64_t stg_size = wf_dstorage_requested_staging_size(buf_size,wf->params);

        auto & stg = wf->ds_stage;
        stg.size = stg_size;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width     = stg_size;
        resDesc.Height = 1; resDesc.DepthOrArraySize = 1; resDesc.MipLevels = 1;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags  = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        hr = wf->d3d_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&stg.d3d_resource));
        if (FAILED(hr)) {
            WF_ERR("DirectStorage: staging CreateCommittedResource failed: 0x%08lx (%.1f GB)",
                    (unsigned long)hr, stg_size / (1024.0 * 1024.0 * 1024.0));
            for (auto * f : wf->ds_files) f->Release(); wf->ds_files.clear();
            wf->ds_factory->Release(); wf->ds_factory = nullptr;
            return false;
        }
        // [2026-08-07 device-hang fix] PIN THE STAGING RESOURCE AGAINST EVICTION. Every DirectStorage
        // write targets this resource; if WDDM demotes/evicts it under VRAM pressure (torch's caching
        // allocator holds its whole peak between denoise passes, and a 6.25GB MoE ring slot sits on top),
        // a DS write into a non-resident resource hangs the device (DXGI_ERROR_DEVICE_HUNG 0x887A0006 —
        // reproducibly at the first submission of pass 2). MAXIMUM priority tells the residency manager
        // to evict anything else first. Best-effort: older runtimes without ID3D12Device1 just skip it.
        {
            ID3D12Device1 * dev1 = nullptr;
            if (SUCCEEDED(wf->d3d_device->QueryInterface(IID_PPV_ARGS(&dev1))) && dev1) {
                ID3D12Pageable * pageable = stg.d3d_resource;
                D3D12_RESIDENCY_PRIORITY prio = D3D12_RESIDENCY_PRIORITY_MAXIMUM;
                HRESULT phr = dev1->SetResidencyPriority(1, &pageable, &prio);
                WF_ERR("DirectStorage: staging residency priority MAXIMUM %s (0x%08lx)",
                       SUCCEEDED(phr) ? "set" : "FAILED", (unsigned long)phr);
                dev1->Release();
            } else {
                WF_ERR("DirectStorage: ID3D12Device1 unavailable — staging residency priority NOT set");
            }
            // PIN IT RESIDENT. Priority alone only says "evict others first" — it does NOT stop
            // WDDM demoting this buffer. That distinction is the whole bug: on Windows cudaMalloc
            // can SUCCEED past dedicated VRAM because the driver evicts someone else, and the
            // someone else is this staging resource. DirectStorage then DMAs into a non-resident
            // allocation and the device hangs (DXGI_ERROR_DEVICE_HUNG) instead of anyone getting a
            // clean out-of-memory error. MakeResident asserts an explicit residency reference, so
            // a greedy torch caching allocator hits an HONEST OOM rather than silently costing us
            // the buffer every DS write targets.
            {
                ID3D12Pageable * pageable = stg.d3d_resource;
                HRESULT mhr = wf->d3d_device->MakeResident(1, &pageable);
                WF_ERR("DirectStorage: staging MakeResident %s (0x%08lx) — %.2f GB pinned",
                       SUCCEEDED(mhr) ? "OK" : "FAILED", (unsigned long)mhr, stg_size / 1e9);
                if (FAILED(mhr)) {
                    WF_ERR("DirectStorage: staging could NOT be pinned resident; the D3D12 heap may "
                           "be evicted under VRAM pressure and DS writes would hang the device. "
                           "Keep MAGI2_TORCH_MEM_FRACTION set to reserve headroom.");
                    if (wf->required_dstorage) return false;
                }
            }
        }
        hr = wf->d3d_device->CreateSharedHandle(stg.d3d_resource, nullptr, GENERIC_ALL, nullptr, &stg.shared_handle);
        if (FAILED(hr)) {
            WF_ERR("DirectStorage: staging CreateSharedHandle failed");
            stg.d3d_resource->Release(); stg.d3d_resource = nullptr;
            for (auto * f : wf->ds_files) f->Release(); wf->ds_files.clear();
            wf->ds_factory->Release(); wf->ds_factory = nullptr;
            return false;
        }
        cudaExternalMemoryHandleDesc memDesc = {};
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = stg.shared_handle;
        memDesc.size = stg_size;
        memDesc.flags = cudaExternalMemoryDedicated;
        cudaError_t cerr = cudaImportExternalMemory(&stg.ext_mem, &memDesc);
        if (cerr != cudaSuccess) {
            WF_ERR("DirectStorage: staging cudaImportExternalMemory failed: %s", cudaGetErrorString(cerr));
            CloseHandle(stg.shared_handle); stg.shared_handle = NULL;
            stg.d3d_resource->Release(); stg.d3d_resource = nullptr;
            for (auto * f : wf->ds_files) f->Release(); wf->ds_files.clear();
            wf->ds_factory->Release(); wf->ds_factory = nullptr;
            return false;
        }
        cudaExternalMemoryBufferDesc bufMapDesc = {};
        bufMapDesc.offset = 0;
        bufMapDesc.size   = stg_size;
        cerr = cudaExternalMemoryGetMappedBuffer(&stg.cuda_ptr, stg.ext_mem, &bufMapDesc);
        if (cerr != cudaSuccess) {
            WF_ERR("DirectStorage: staging CUDA map failed: %s", cudaGetErrorString(cerr));
            cudaDestroyExternalMemory(stg.ext_mem); stg.ext_mem = {};
            CloseHandle(stg.shared_handle); stg.shared_handle = NULL;
            stg.d3d_resource->Release(); stg.d3d_resource = nullptr;
            for (auto * f : wf->ds_files) f->Release(); wf->ds_files.clear();
            wf->ds_factory->Release(); wf->ds_factory = nullptr;
            return false;
        }
        wf->ds_staging_size = stg_size;
        WF_LOG("DS staging: %.2f GB at %p (D3D12 + CUDA), %d compute bufs",
               stg_size / (1024.0 * 1024.0 * 1024.0), stg.cuda_ptr, (int)wf->gpu_bufs.size());
        // Phase 316e (2026-05-04): track DS staging alloc to attribute the
        // ~3.5 GB/cycle VRAM leak. If this 3.5 GB stage IS the leak, allocs
        // per cycle will accumulate without matching frees.
        fprintf(stderr, "*** DS_STAGE_ALLOC: %.2f GB cuda_ptr=%p ext_mem=%p d3d=%p shared=%p\n",
                stg_size / (1024.0 * 1024.0 * 1024.0),
                stg.cuda_ptr, (void*)stg.ext_mem, (void*)stg.d3d_resource, stg.shared_handle);
        fflush(stderr);
    }

    // 2026-06-01 R3: cache the inert infra (device/factory/files/staging) for reuse across
    // requests, THEN create the stateful objects (queue/fence) via the helper.
    if (!wf->required_dstorage && wf_persist_infra_enabled() && !wf->ds_direct_mode && !g_wf_persist.valid &&
            wf->ds_stage.d3d_resource && wf->ds_stage.ext_mem &&
            wf->ds_stage.shared_handle && wf->ds_stage.cuda_ptr) {
        g_wf_persist.dxgi    = wf->dxgi_factory;          if (g_wf_persist.dxgi)    g_wf_persist.dxgi->AddRef();
        g_wf_persist.device  = wf->d3d_device;            if (g_wf_persist.device)  g_wf_persist.device->AddRef();
        g_wf_persist.factory = wf->ds_factory;            if (g_wf_persist.factory) g_wf_persist.factory->AddRef();
        for (auto * file : g_wf_persist.files) if (file) file->Release();
        g_wf_persist.files   = wf->ds_files;               for (auto * f : g_wf_persist.files) if (f) f->AddRef();
        g_wf_persist_file_ids.clear();
        const auto cache_paths = wf_ds_requested_paths(wf);
        for (size_t i = 0; i < g_wf_persist.files.size() && i < cache_paths.size(); ++i) {
            wf_ds_file_identity identity;
            identity.path = cache_paths[i];
            if (FAILED(g_wf_persist.files[i]->GetFileInformation(&identity.info))) {
                g_wf_persist_file_ids.clear();
                break; // Next request must refresh, never trust an unknown identity.
            }
            g_wf_persist_file_ids.push_back(identity);
        }
        g_wf_persist.stg_d3d     = wf->ds_stage.d3d_resource; if (g_wf_persist.stg_d3d) g_wf_persist.stg_d3d->AddRef();
        g_wf_persist.stg_ext_mem = wf->ds_stage.ext_mem;
        g_wf_persist.stg_shared  = wf->ds_stage.shared_handle;
        g_wf_persist.stg_ptr     = wf->ds_stage.cuda_ptr;
        g_wf_persist.stg_size    = wf->ds_staging_size;
        g_wf_persist.valid       = true;
        g_wf_persist.active_owner = wf;
        wf->ds_stage.persistent  = true;
        fprintf(stderr, "*** WF_PERSIST_INFRA: CACHED device/factory/files/staging for reuse (%.2f GB)\n",
                wf->ds_staging_size / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
    }
    if (!wf_dstorage_create_stateful(wf)) return false;
    wf->ds_initialized = true;
    // R3 BLOCKER B1 probe: validate DS writes into placed-in-heap resources at >4GB heap offsets
    // (the make-or-break question for the DS-direct ping-pong design). Runs once, gated env.
    wf_selftest_ds_placed(wf);
    // R3 edit-4 validation: exercise the dual-use heap allocator + teardown in isolation. Runs once, gated env.
    wf_selftest_dual_heaps(wf);
    WF_LOG("DirectStorage initialized: %d GPU bufs (%.2f MB each), staging=%.2f GB, queue cap=%d",
           (int)wf->gpu_bufs.size(), buf_size / (1024.0 * 1024.0),
           wf->ds_staging_size / (1024.0 * 1024.0 * 1024.0), wf->ds_queue_cap);
    return true;
}

// Check if DirectStorage DMA for a layer has completed (fence check).
static bool wf_dstorage_layer_done(wf_context * wf, int layer_idx) {
    uint64_t needed = wf->ds_layer_fence_val[layer_idx];
    if (needed == 0) return true;
    const uint64_t completed = wf->ds_fence->GetCompletedValue();
    if (completed == UINT64_MAX) {
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }
    return completed >= needed;
}

// Helper: CPU-wait for a DS fence value.
static bool wf_ds_fence_wait(wf_context * wf, uint64_t val, ID3D12Fence * fence) {
    if (!fence) fence = wf->ds_fence;
    uint64_t completed = fence->GetCompletedValue();
    // ID3D12Fence::GetCompletedValue returns UINT64_MAX when the device has
    // been removed. It is a failure sentinel, not a fence value satisfying
    // every future wait.
    if (completed == UINT64_MAX) {
        const HRESULT reason = wf->d3d_device
                ? wf->d3d_device->GetDeviceRemovedReason() : E_FAIL;
        WF_ERR("DirectStorage fence %llu failed: D3D12 device removed "
               "(GetDeviceRemovedReason=0x%08lx)",
               (unsigned long long) val, (unsigned long) reason);
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }
    if (completed >= val) return true;

    HANDLE evt = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!evt) {
        WF_ERR("DirectStorage fence %llu wait failed: CreateEvent error %lu",
               (unsigned long long) val, GetLastError());
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }
    const HRESULT set_hr = fence->SetEventOnCompletion(val, evt);
    if (FAILED(set_hr)) {
        WF_ERR("DirectStorage fence %llu wait failed: SetEventOnCompletion=0x%08lx",
               (unsigned long long) val, (unsigned long) set_hr);
        CloseHandle(evt);
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }
    const DWORD wait_result = WaitForSingleObject(evt, INFINITE);
    CloseHandle(evt);

    completed = fence->GetCompletedValue();
    if (wait_result != WAIT_OBJECT_0 || completed == UINT64_MAX || completed < val) {
        const HRESULT reason = completed == UINT64_MAX && wf->d3d_device
                ? wf->d3d_device->GetDeviceRemovedReason() : S_OK;
        WF_ERR("DirectStorage fence %llu wait failed: wait=%lu completed=%llu "
               "device_reason=0x%08lx",
               (unsigned long long) val, (unsigned long) wait_result,
               (unsigned long long) completed, (unsigned long) reason);
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }
    return true;
}

// RetrieveErrorRecord is only valid after the queue's error event is signaled.
// A fence signal can complete even when an earlier request failed, so this must
// run before any CUDA copy consumes the DS destination.
static bool wf_dstorage_wave_succeeded(wf_context * wf, IDStorageQueue * queue,
                                        int layer_idx, int wave_idx) {
    HANDLE error_event = queue->GetErrorEvent();
    if (!error_event) {
        WF_ERR("DS error-event query failed for layer=%d wave=%d",
               layer_idx, wave_idx);
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }

    const DWORD error_state = WaitForSingleObject(error_event, 0);
    if (error_state == WAIT_TIMEOUT) {
        return true;
    }
    if (error_state != WAIT_OBJECT_0) {
        WF_ERR("DS error-event wait failed for layer=%d wave=%d: wait=%lu error=%lu",
               layer_idx, wave_idx, (unsigned long) error_state, GetLastError());
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }

    DSTORAGE_ERROR_RECORD er = {};
    queue->RetrieveErrorRecord(&er);
    if (er.FailureCount == 0) return true;

    const auto & failed = er.FirstFailure;
    if (failed.CommandType == DSTORAGE_COMMAND_TYPE_REQUEST) {
        const auto & req = failed.Request.Request;
        WF_ERR("DS REQUEST FAILURE layer=%d wave=%d: %u command(s) failed, "
               "first hr=0x%08lx (file=%p off=%llu size=%u)",
               layer_idx, wave_idx, (unsigned) er.FailureCount,
               (unsigned long) failed.HResult,
               (void *) req.Source.File.Source,
               (unsigned long long) req.Source.File.Offset,
               (unsigned) req.Source.File.Size);
    } else {
        WF_ERR("DS COMMAND FAILURE layer=%d wave=%d: %u command(s) failed, "
               "first hr=0x%08lx type=%d",
               layer_idx, wave_idx, (unsigned) er.FailureCount,
               (unsigned long) failed.HResult, (int) failed.CommandType);
    }
    wf->ds_failed.store(true, std::memory_order_release);
    return false;
}

// A CPU wait on the D3D12 fence only establishes host-side completion. CUDA
// must wait on the imported fence on the stream that consumes the shared D3D12
// allocation; otherwise the following CUDA read has no cross-API ordering edge.
static bool wf_cuda_wait_ds_fence(wf_context * wf, uint64_t val, cudaStream_t stream) {
    if (!wf || !wf->ds_ext_sem || val == 0) return false;

    cudaExternalSemaphoreWaitParams wait_params = {};
    wait_params.params.fence.value = val;
    cudaError_t err = cudaWaitExternalSemaphoresAsync(
        &wf->ds_ext_sem, &wait_params, 1, stream);
    if (err != cudaSuccess) {
        WF_ERR("DirectStorage CUDA fence wait failed (value=%llu stream=%p): %s",
               (unsigned long long) val, (void *) stream, cudaGetErrorString(err));
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Cold-start interop warm-up (LLAMA_WF_WARM_INTEROP, DEFAULT ON; =0 = exact
// prior behavior for A/B).
//
// The D3D12<->CUDA interop objects — the staging/prefetch external-memory VA
// (cudaExternalMemoryGetMappedBuffer) and the imported D3D12-fence external
// semaphore (cudaImportExternalSemaphore) — are created at load, but the CUDA
// driver wires the copy-engine page tables for the shared VA and the
// semaphore's device/stream association LAZILY, on the first GPU consume. The
// very first real layer fetch therefore RACES that lazy wiring -> intermittent
// "illegal memory access" surfaced at the first cudaWaitExternalSemaphoresAsync
// (the long-standing "WF DirectStorage cold-start AV on the FIRST request after
// a fresh load"). PROMPT-DIRECT never hits it because it never touches this
// interop.
//
// Fix: replay the EXACT first-fetch consume ONCE, serialized, at load — a tiny
// real DS write into the live staging/prefetch resource, an EnqueueSignal on the
// DS fence, the external-semaphore wait on BOTH consumer streams (the null
// default stream used by the SYNC/relay + DS-direct paths, and the ggml compute
// stream used by wf_cuda_wait_upload), then a D2D read of the mapped VA to force
// the copy-engine mapping, then a full device sync so any cold fault aborts HERE
// (deterministically, at load) instead of mid-serve. No concurrent compute runs
// at load, so the serialized sequence completes the wiring with no race; real
// fetches are then warm. Runs once per wf_create (NO process-static guard — a
// fresh wf_context after a model reload must be re-warmed, which also covers the
// PERSIST-INFRA reuse branch that re-creates the per-request fence/semaphore).
// ---------------------------------------------------------------------------
static void wf_warm_ds_interop(wf_context * wf) {
#if defined(WF_HAS_CUDA) && defined(WF_HAS_DSTORAGE)
    const char * e = std::getenv("LLAMA_WF_WARM_INTEROP");
    if (e && e[0] == '0') return;                 // A/B opt-out: byte-identical to prior behavior
    if (!wf || !wf->ds_initialized || !wf->ds_queue || !wf->ds_fence || !wf->ds_ext_sem) return;

    const bool   direct = wf->ds_direct_mode;
    const char * mode   = direct ? "direct" : "relay";
    fprintf(stderr, "*** WF_WARM_INTEROP: warming DS<->CUDA interop at load (mode=%s)\n", mode);
    fflush(stderr);

    // Live staging target + its mapped CUDA view for the configured path.
    ID3D12Resource * dst_res = wf->params.ds_compute_direct
        ? (wf->ds_compute_banks.empty()?nullptr:wf->ds_compute_banks[0].quarters[0]) : direct
        ? (wf->prefetch_quarters.empty() ? nullptr : wf->prefetch_quarters[0])
        : wf->ds_stage.d3d_resource;
    void * stg_ptr = wf->params.ds_compute_direct ? wf->gpu_bufs[0].ptr : direct ? wf->prefetch_buf : wf->ds_stage.cuda_ptr;
    if (!dst_res || !stg_ptr) {
        if (wf->required_dstorage) wf->ds_failed.store(true, std::memory_order_release);
        fprintf(stderr, "*** WF_WARM_INTEROP: SKIP — no live staging target (res=%p ptr=%p)\n",
                (void *) dst_res, stg_ptr);
        fflush(stderr);
        return;
    }

    // Source: any real DS file @0; a few KB is enough to exercise the DMA + fence.
    IDStorageFile * src = nullptr;
    if (!wf->ds_files.empty() && wf->ds_files[0]) src = wf->ds_files[0];
    else if (!wf->ds_layer_files.empty() && wf->ds_layer_files[0]) src = wf->ds_layer_files[0];
    if (!src) {
        if (wf->required_dstorage) wf->ds_failed.store(true, std::memory_order_release);
        fprintf(stderr, "*** WF_WARM_INTEROP: SKIP — no source DS file\n");
        fflush(stderr);
        return;
    }

    const uint32_t CHUNK = 65536;   // 64 KB — one aligned DS request
    if (wf->required_dstorage && (dst_res->GetDesc().Width < CHUNK ||
            wf->files.empty() || wf->files[0].file_size < CHUNK)) {
        WF_ERR("Required DirectStorage: warm request exceeds source/staging extent");
        wf->ds_failed.store(true, std::memory_order_release);
        return;
    }

    // 1) Real DS write into the live staging/prefetch resource, fence-signalled,
    //    then the external-semaphore wait on BOTH (all) consumer streams.
    {
        std::lock_guard<std::mutex> ds_lock(wf->ds_queue_mutex);
        DSTORAGE_REQUEST req = {};
        req.Options.SourceType          = DSTORAGE_REQUEST_SOURCE_FILE;
        req.Options.DestinationType     = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        req.Options.CompressionFormat   = DSTORAGE_COMPRESSION_FORMAT_NONE;
        req.Source.File.Source          = src;
        req.Source.File.Offset          = 0;
        req.Source.File.Size            = CHUNK;
        req.Destination.Buffer.Resource = dst_res;
        req.Destination.Buffer.Offset   = 0;
        req.Destination.Buffer.Size     = CHUNK;
        req.UncompressedSize            = CHUNK;
        wf->ds_queue->Submit();                       // drain any prior state
        wf->ds_queue->EnqueueRequest(&req);
        uint64_t fv = ++wf->ds_fence_counter;
        wf->ds_queue->EnqueueSignal(wf->ds_fence, fv);
        wf->ds_queue->Submit();
        if (!wf_ds_fence_wait(wf, fv)) return;         // host-side DS completion
        if (wf->required_dstorage && !wf_dstorage_wave_succeeded(wf, wf->ds_queue, -1, -1)) return;

        // Wire the fence->semaphore device association the real paths depend on.
        if (!wf_cuda_wait_ds_fence(wf, fv, (cudaStream_t) 0)) return;           // SYNC/relay + DS-direct consumer
        if (wf->cuda_compute_stream && !wf_cuda_wait_ds_fence(wf, fv, wf->cuda_compute_stream)) return;
        if (wf->copy_stream && !wf_cuda_wait_ds_fence(wf, fv, wf->copy_stream)) return;
    }

    // 2) D2D read of the mapped external-memory VA on stream 0 (the SYNC/relay
    //    consumer) to force the copy-engine page-table wiring of the shared VA.
    unsigned char scratch_host[16] = {0};
    void *        scratch = nullptr;
    cudaError_t   alloc_err = cudaMalloc(&scratch, CHUNK);
    bool warm_ok = alloc_err == cudaSuccess && scratch;
    if (alloc_err == cudaSuccess && scratch) {
        const cudaError_t copy_device = cudaMemcpy(scratch, stg_ptr, CHUNK, cudaMemcpyDeviceToDevice);
        const cudaError_t copy_host = cudaMemcpy(scratch_host, scratch, sizeof(scratch_host), cudaMemcpyDeviceToHost);
        const cudaError_t freed = cudaFree(scratch);
        warm_ok = copy_device == cudaSuccess && copy_host == cudaSuccess && freed == cudaSuccess;
    }

    // 3) DS-direct relies on the prefetch heap's inter-span gaps being ZERO
    //    (per-layer DS writes only weight spans). Restore the CHUNK we dirtied.
    if (direct) {
        const cudaError_t zeroed = cudaMemset(stg_ptr, 0, CHUNK);
        warm_ok = warm_ok && zeroed == cudaSuccess;
    }

    // 4) Surface any cold-start fault HERE (at load) rather than mid-serve.
    cudaError_t serr = cudaDeviceSynchronize();
    cudaError_t lerr = cudaGetLastError();
    if (wf->required_dstorage && !warm_ok) {
        WF_ERR("Required DirectStorage: warm allocation/copy/free failed");
        wf->ds_failed.store(true, std::memory_order_release);
    }
    if (serr != cudaSuccess || lerr != cudaSuccess) {
        if (wf->required_dstorage) wf->ds_failed.store(true, std::memory_order_release);
        // Do NOT abort load — fall through to today's behavior. (A true sticky
        // IMA poisons the context, but that is strictly no worse than the
        // pre-fix crash on the first real fetch; logging loudly at load is more
        // diagnostic and we must not regress by refusing to serve.)
        WF_ERR("WF_WARM_INTEROP: warm-up saw CUDA error (sync=%s last=%s) — continuing without warm interop",
               cudaGetErrorString(serr), cudaGetErrorString(lerr));
    } else {
        fprintf(stderr, "*** WF_WARM_INTEROP: OK (mode=%s, byte0=0x%02x) — DS<->CUDA interop wired\n",
                mode, scratch_host[0]);
        fflush(stderr);
    }
#else
    (void) wf;
#endif
}

// ---------------------------------------------------------------------------
// Pre-build wave-based DS request arrays at model load time.
//
// Each layer is split into waves of ≤ staging_size bytes.  All requests in
// a wave target the single staging D3D12 resource at offset 0..staging_size.
// At runtime the transfer loop is: for each wave → enqueue → fence → D2D.
// ---------------------------------------------------------------------------

static void wf_dstorage_prebuild_requests(wf_context * wf) {
    int n_layers = (int)wf->layers.size();
    size_t bs_io = wf->params.io_block_size;
    // R3 DS-direct edit 9: in DS-direct the per-wave stride is the QUARTER size, so each wave maps to
    // exactly ONE prefetch placed-quarter (=> one Submit/quarter, HARD LIMIT 2) and the within-wave
    // offset (stg_off) is exactly the within-quarter Destination.Buffer.Offset (< QSZ < 4GB, B1-proven).
    // Quarter w lives at heap offset w*QSZ, so a request to quarter[w]@(dest-w*QSZ) lands at heap[dest] —
    // i.e. the full layer ends up contiguous in the prefetch buffer, ready for a single prefetch->compute D2D.
    // Legacy path is unchanged (stride = staging size, destination = the single staging resource).
    uint64_t stg_size = wf->ds_direct_mode ? wf->ds_quarter_size : wf->ds_staging_size;
    if(wf->params.ds_gpu_ring_slots)stg_size=(stg_size/2)&~uint64_t(65535);

    // Wave-count and wave-index arithmetic below divides by stg_size. Invalid
    // initialization must produce an empty plan, not a process-killing fault.
    if (stg_size == 0) {
        WF_ERR("DirectStorage: cannot prebuild requests with zero staging size");
        wf->ds_waves.assign(n_layers, {});
        wf->ds_mmap_spans.assign(n_layers, {});
        return;
    }

    wf->ds_waves.resize(n_layers);
    wf->ds_mmap_spans.resize(n_layers);

    size_t total_reqs = 0;
    size_t total_waves = 0;

    for (int il = 0; il < n_layers; il++) {
        auto & layer = wf->layers[il];

        // Collect mmap fallback spans (no DS — just cudaMemcpy)
        for (int span_idx = 0; span_idx < (int) layer.spans.size(); ++span_idx) {
            auto & span = layer.spans[span_idx];
            if (span.file_offset == 0 && span.mmap_addr != nullptr) {
                wf->ds_mmap_spans[il].push_back({ span.gpu_offset, span.mmap_addr, span.nbytes, span_idx });
            }
        }

        // Determine how many waves this layer needs.
        // TASK-A FIX (2026-07-20): capacity must be computed from the PADDED span extent
        // (max gpu_offset+nbytes), not the raw byte sum (total_bytes). Request placement uses
        // alloc-padded gpu_offsets; sizing waves by total_bytes silently DROPPED any request
        // beyond ceil(total_bytes/stg) waves ("if (w >= n_waves) break") and the final wave's
        // staging->compute D2D under-copied the padded tail — the LAST tensors of a layer
        // received stale/garbage bytes (observed: blk.20.attn_output_a/b garbage under
        // keep_mmap=0 stream_local => 100% NaN hidden from layer 20, '<DDD' output).
        size_t layer_bytes = layer.total_bytes;
        for (const auto & span : layer.spans) {
            const uint64_t span_end = span.gpu_offset + span.nbytes;
            if (span_end > layer_bytes) layer_bytes = (size_t) span_end;
        }
        if (layer_bytes > layer.total_bytes) {
            WF_LOG("layer %d: padded span extent %zu > total_bytes %zu (+%zu) — waves sized to extent",
                   il, layer_bytes, layer.total_bytes, layer_bytes - layer.total_bytes);
        }
        int n_waves = layer_bytes > 0 ? (int)((layer_bytes + stg_size - 1) / stg_size) : 0;
        wf->ds_waves[il].resize(n_waves);

        // Set up wave metadata
        for (int w = 0; w < n_waves; w++) {
            auto & wave = wf->ds_waves[il][w];
            wave.ring_sequence=total_waves+uint64_t(w);
            wave.compute_offset = (uint64_t)w * stg_size;
            wave.nbytes = std::min(stg_size, (uint64_t)layer_bytes - wave.compute_offset);
        }

        // Per-layer file mode: sequential read from .bin file
        if (wf->use_layer_files && il < (int)wf->ds_layer_files.size() && wf->ds_layer_files[il]) {
            IDStorageFile * lf = wf->ds_layer_files[il];
            char layer_path[512];
            snprintf(layer_path, sizeof(layer_path), "%s/layer_%03d.bin",
                     std::getenv("LLAMA_PREFILL_LAYER_DIR"), il);
            BY_HANDLE_FILE_INFORMATION fi;
            HANDLE hf = CreateFileA(layer_path, GENERIC_READ, FILE_SHARE_READ,
                                    NULL, OPEN_EXISTING, 0, NULL);
            uint64_t file_size = layer.total_bytes;
            if (hf != INVALID_HANDLE_VALUE) {
                if (GetFileInformationByHandle(hf, &fi))
                    file_size = ((uint64_t)fi.nFileSizeHigh << 32) | fi.nFileSizeLow;
                CloseHandle(hf);
            }

            uint64_t file_off = 0;
            uint64_t dest_off = 0;
            size_t remaining = (size_t)file_size;

            while (remaining > 0) {
                int w = (int)(dest_off / stg_size);
                if (w >= n_waves) break;
                auto & wave = wf->ds_waves[il][w];
                uint64_t stg_off = dest_off - wave.compute_offset;  // offset within staging

                size_t chunk = std::min(remaining, bs_io);
                // Clamp to wave boundary
                uint64_t wave_remaining = stg_size - stg_off;
                if (chunk > wave_remaining) chunk = (size_t)wave_remaining;

                DSTORAGE_REQUEST req = {};
                req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                req.Source.File.Source         = lf;
                req.Source.File.Offset        = file_off;
                req.Source.File.Size          = (UINT32)chunk;
                // DS-direct: target prefetch quarter w (stg_off is the within-quarter offset, <QSZ<4GB).
                ID3D12Resource * dst_res = wf->ds_stage.d3d_resource;
                if(wf->params.ds_compute_direct){
                    const auto &bank=wf->ds_compute_banks[il%wf->ds_compute_banks.size()];
                    if(w>=(int)bank.quarters.size()){wf->ds_failed.store(true);return;}dst_res=bank.quarters[w];
                }else if (wf->ds_direct_mode) {
                    if (w >= (int)wf->prefetch_quarters.size()) break;  // defensive: layer exceeds prefetch heap
                    dst_res = wf->prefetch_quarters[w];
                }
                req.Destination.Buffer.Resource = dst_res;
                req.Destination.Buffer.Offset   = stg_off;
                if(wf->params.ds_gpu_ring_slots){req.Destination.Buffer.Offset+=(wave.ring_sequence%2)*stg_size;req.CancellationTag=0x4453345045524635ull;}
                req.Destination.Buffer.Size     = (UINT32)chunk;
                req.UncompressedSize            = (UINT32)chunk;

                wave.reqs.push_back(req);
                wave.req_span_idx.push_back(-1);
                file_off  += chunk;
                dest_off  += chunk;
                remaining -= chunk;
            }
            for (auto & w : wf->ds_waves[il]) total_reqs += w.reqs.size();
            total_waves += n_waves;
            continue;
        }

        // Standard GGUF shard mode: build requests per span, grouped by wave
        for (int span_idx = 0; span_idx < (int)layer.spans.size(); ++span_idx) {
            auto & span = layer.spans[span_idx];
            if (span.file_offset == 0 && span.mmap_addr != nullptr) continue;
            int file_idx = span.file_idx;
            if (file_idx < 0 || file_idx >= (int)wf->ds_files.size()) continue;

            uint64_t offset = span.file_offset;
            uint64_t dest_offset = span.gpu_offset;
            size_t remaining = span.nbytes;

            while (remaining > 0) {
                int w = (int)(dest_offset / stg_size);
                if (w >= n_waves) break;
                auto & wave = wf->ds_waves[il][w];
                uint64_t stg_off = dest_offset - wave.compute_offset;

                size_t chunk = std::min(remaining, bs_io);
                // Clamp to wave boundary
                uint64_t wave_remaining = stg_size - stg_off;
                if (chunk > wave_remaining) chunk = (size_t)wave_remaining;

                DSTORAGE_REQUEST req = {};
                req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                req.Source.File.Source         = wf->ds_files[file_idx];
                req.Source.File.Offset        = offset;
                req.Source.File.Size          = (UINT32)chunk;
                // DS-direct: target prefetch quarter w (stg_off is the within-quarter offset, <QSZ<4GB).
                ID3D12Resource * dst_res = wf->ds_stage.d3d_resource;
                if(wf->params.ds_compute_direct){
                    const auto &bank=wf->ds_compute_banks[il%wf->ds_compute_banks.size()];
                    if(w>=(int)bank.quarters.size()){wf->ds_failed.store(true);return;}dst_res=bank.quarters[w];
                }else if (wf->ds_direct_mode) {
                    if (w >= (int)wf->prefetch_quarters.size()) break;  // defensive: layer exceeds prefetch heap
                    dst_res = wf->prefetch_quarters[w];
                }
                req.Destination.Buffer.Resource = dst_res;
                req.Destination.Buffer.Offset   = stg_off;
                if(wf->params.ds_gpu_ring_slots){req.Destination.Buffer.Offset+=(wave.ring_sequence%2)*stg_size;req.CancellationTag=0x4453345045524635ull;}
                req.Destination.Buffer.Size     = (UINT32)chunk;
                req.UncompressedSize            = (UINT32)chunk;

                wave.reqs.push_back(req);
                wave.req_span_idx.push_back(span_idx);
                offset      += chunk;
                dest_offset += chunk;
                remaining   -= chunk;
            }
        }
        for (auto & w : wf->ds_waves[il]) total_reqs += w.reqs.size();
        total_waves += n_waves;
    }

    WF_LOG("DS pre-built: %d layers, %zu waves, %zu total requests (%.2f MB)",
           n_layers, total_waves, total_reqs,
           (total_reqs * sizeof(DSTORAGE_REQUEST)) / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
#include "llama-wf-gpu-ring.inl"

// Blocking wave-based layer transfer.
//
// For each wave: enqueue DS requests → fence → D2D staging→compute.
// Returns when the entire layer is in the compute buffer.  Called from the
// DMA thread — it's OK to block because the staging resource is shared and
// can only serve one transfer at a time anyway.
// ---------------------------------------------------------------------------

// Transfer a layer's weights via DirectStorage.
// dest_buf: GPU destination (compute buffer or prefetch buffer)
// dest_offset: byte offset within dest_buf for the first wave's data
// wave_start, wave_end: subset of waves to process (-1 = all)
// If dest_buf differs from gpu_bufs[gpu_idx], the D2D targets dest_buf
// at sequential offsets starting from dest_offset.
static bool wf_dstorage_transfer_layer(wf_context * wf, int layer_idx, int gpu_idx,
                                        void * dest_buf = nullptr, size_t dest_offset = 0,
                                        int wave_start = -1, int wave_end = -1) {
    if (wf->ds_failed.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> ds_lock(wf->ds_queue_mutex);
    if(wf->ds_ring){if(dest_buf||dest_offset||wave_start>=0||wave_end>=0)return wf_ring_error(wf,"Ring forbids partial/relay transfers");return wf_dstorage_ring_transfer(wf,layer_idx,gpu_idx);}
    auto & layer = wf->layers[layer_idx];
    IDStorageQueue * queue = wf->ds_queue;
    void * default_buf = wf->gpu_bufs[gpu_idx].ptr;
    void * target_buf = dest_buf ? dest_buf : default_buf;
    auto & stg = wf->ds_stage;
    auto cuda_ok = [&](cudaError_t status) {
        if (!wf->required_dstorage || status == cudaSuccess) return true;
        WF_ERR("Required DirectStorage: staging/visibility operation failed: %s", cudaGetErrorString(status));
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    };
    if (wf->required_dstorage && !target_buf) {
        wf->ds_failed.store(true, std::memory_order_release);
        return false;
    }

    // Full-layer DS transfers suppress requests for cached expert spans. Pin
    // sources until their replacement copies have been enqueued.
    std::vector<bool> cache_hit(layer.spans.size(), false);
    size_t cache_bytes = 0;
    const bool cache_full_transfer = wf_cache_enabled(wf) && !wf->use_layer_files &&
            wave_start < 0 && wave_end < 0 && dest_offset == 0;
    if (cache_full_transfer) {
        for (int si = 0; si < (int)layer.spans.size(); ++si) {
            const auto & span = layer.spans[si];
            if (!wf_span_is_active_expert(wf, layer, span)) continue;
            cache_hit[si] = wf_cache_lookup(wf, layer_idx, span,
                    (char *)target_buf + span.gpu_offset, false, true);
            if (cache_hit[si]) cache_bytes += span.nbytes;
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Phase 2.2 v11 (2026-05-11): route the WF CUDA ops through wf->copy_stream
    // instead of the default per-thread stream. The default-stream variants of
    // cudaMemset/cudaMemcpy implicitly synchronize with the compute stream,
    // serializing the DStorage prefetch behind compute (overlap efficiency=0%).
    // Async variants on a dedicated copy stream let NVMe + D2D run in parallel
    // with compute, with cudaEventRecord(upload_done, copy_stream) +
    // cudaStreamWaitEvent(compute_stream, upload_done) for the necessary
    // hand-off. The upload_done event is already used by wf_cuda_wait_upload.
    //
    // LLAMA_WF_DS_USE_COPY_STREAM=1 enables (default off until A/B validates).
    // The legacy LLAMA_WF_DS_SUBMIT_BEFORE_ZERO=1 is retained as a no-op
    // alternative; it didn't fix the bottleneck (see commit bb4ae84b).
    static const bool requested_copy_stream = []() {
        const char * e = std::getenv("LLAMA_WF_DS_USE_COPY_STREAM");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    // LLAMA_WF_ASYNC_DRAIN=1: async D2D on copy_stream + gate the NEXT wave's DS
    // read on a drain event (single-staging-safe — fixes the reuse race WITHOUT
    // double-buffering). The async-D2D enqueue is a microsecond lock-grab (vs the
    // sync D2D that holds the DMA thread ~8.7ms/layer lock-contended), so the
    // host's compute-kernel burst stops starving the DMA -> fetch overlaps compute
    // without the OP_PROFILE crutch -> ~685 @20k. Forces copy_stream on.
    static const bool requested_async_drain = []() {
        const char * e = std::getenv("LLAMA_WF_ASYNC_DRAIN");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    const bool use_copy_stream = wf->required_dstorage ? wf->params.ds_async_drain : requested_copy_stream;
    const bool async_drain = wf->required_dstorage ? wf->params.ds_async_drain : requested_async_drain;
    cudaStream_t wf_s = ((use_copy_stream || async_drain) && wf->copy_stream)
                        ? wf->copy_stream : (cudaStream_t)0;
    bool staging_busy = false;  // a prior wave's async D2D is reading the staging buffer

    // Zero the target region before loading (prevents NaN from padding reads).
    // Only zero on first call (wave_start <= 0) to avoid double-zeroing.
    // R3 DS-direct: DS overwrites [0, total_bytes) of prefetch and the prefetch->compute D2D reads
    // exactly total_bytes, so zero only that region (NOT the full 9GB prefetch heap — that would cost
    // ~30-45ms/layer for nothing).
    if (wave_start <= 0 && !wf->ds_direct_mode) {
        size_t zero_bytes = dest_buf ? wf->prefetch_buf_size : layer.total_bytes;
        if (wf_s) {
            if(!cuda_ok(cudaMemsetAsync(target_buf, 0, zero_bytes, wf_s)))return false;
        } else {
            if (!cuda_ok(cudaMemset(target_buf, 0, zero_bytes))) return false;
        }
    } else if(wave_start<=0&&wf->params.ds_compute_direct){
        if(!wf_s||gpu_idx!=layer_idx%int(wf->gpu_bufs.size())||target_buf!=wf->gpu_bufs[gpu_idx].ptr)return false;
        bool routed_stage=false;for(const auto&span:layer.spans)if(span.is_expert){routed_stage=true;break;}
        if(routed_stage&&wf->params.moe_mode==WF_MOE_FETCH_ACTIVE){
            if(!wf->required_dstorage||dest_buf||dest_offset||wave_start>=0||wave_end>=0||!layer.expert_filter_set){
                WF_ERR("Required DirectStorage routed-stage zero refused: unrouted or partial transfer");wf->ds_failed.store(true);return false;}
            // LLAMADECODE_POISON_UNROUTED=1 (debug only): every byte this transfer
            // does not write (unrouted experts and padding) becomes 0xFF, i.e. NaN
            // F16 block scales, so any read of unrouted data turns the logits NaN.
            static const bool poison=[]{const char*e=std::getenv("LLAMADECODE_POISON_UNROUTED");return e&&e[0]=='1'&&!e[1];}();
            std::vector<llama_wf_zero_gaps::interval>owned,gaps;owned.reserve(layer.spans.size());
            for(const auto&span:layer.spans)if(!poison||!span.is_expert||wf_span_is_active_expert(wf,layer,span))owned.push_back({span.gpu_offset,span.nbytes});
            std::string reason;
            if(!llama_wf_zero_gaps::plan(layer.total_bytes,owned,gaps,reason)){WF_ERR("Required DirectStorage routed-stage zero refused: %s",reason.c_str());wf->ds_failed.store(true);return false;}
            uint64_t zeroed=0;
            for(const auto&gap:gaps){if(!cuda_ok(cudaMemsetAsync((char*)target_buf+gap.offset,poison?0xFF:0,size_t(gap.bytes),wf_s)))return false;zeroed+=gap.bytes;}
            if(poison)WF_ERR("LLAMADECODE_POISON_UNROUTED stage=%d poisoned_bytes=%llu fill=0xFF",layer_idx,(unsigned long long)zeroed);
            if(!cuda_ok(cudaStreamSynchronize(wf_s)))return false;
            size_t active=0;for(const auto&span:layer.spans)active+=span.is_expert&&wf_span_is_active_expert(wf,layer,span)?1:0;
            if(llama_dsv41_layer_log_on())WF_ERR("LLAMADECODE_ROUTED_STAGE stage=%d extent=%llu padding_zero_bytes=%llu routed_spans=%zu experts=%zu unrouted_extents_never_read=1",
                layer_idx,(unsigned long long)layer.total_bytes,(unsigned long long)zeroed,active,layer.active_experts.size());
        }else
        // These exact bytes are the DS destination, so complete the existing
        // copy stream's zero BEFORE D3D writes. Other bank's compute may run.
        if(wf->params.ds_zero_padding_only){
            auto refuse=[&](const char*why){WF_ERR("Required DirectStorage padding-only zero refused: %s",why);wf->ds_failed.store(true);return false;};
            if(!wf->required_dstorage||dest_buf||dest_offset||wave_start>=0||wave_end>=0||
               wf->use_layer_files||wf_cache_enabled(wf)||layer.expert_prefetch_planned||
               layer.spans.empty()||layer.spans.size()>8192||
               (layer_idx<(int)wf->ds_mmap_spans.size()&&!wf->ds_mmap_spans[layer_idx].empty()))
                return refuse("requires complete original-file all-expert spans without cache/mmap/partial transfer");
            // Prove the existing prebuilt requests will overwrite every byte of
            // every span once, in source-span order, before omitting its zero.
            std::vector<uint64_t>covered(layer.spans.size(),0);
            for(const auto&wave:wf->ds_waves[layer_idx]){
                if(wave.reqs.size()!=wave.req_span_idx.size())return refuse("missing request/span binding");
                for(size_t ri=0;ri<wave.reqs.size();++ri){
                    const int si=wave.req_span_idx[ri];
                    if(si<0||si>=(int)layer.spans.size())return refuse("request has no original span");
                    const auto&span=layer.spans[si];const auto&req=wave.reqs[ri];
                    const uint64_t off=req.Destination.Buffer.Offset,bytes=req.Destination.Buffer.Size;
                    if(req.Options.SourceType!=DSTORAGE_REQUEST_SOURCE_FILE||
                       req.Options.DestinationType!=DSTORAGE_REQUEST_DESTINATION_BUFFER||
                       !bytes||bytes!=req.Source.File.Size||bytes!=req.UncompressedSize||
                       off>UINT64_MAX-wave.compute_offset||
                       !llama_wf_zero_gaps::cover({span.gpu_offset,span.nbytes},wave.compute_offset+off,bytes,covered[si]))
                        return refuse("noncontiguous or changed request extent");
                }
            }
            std::vector<llama_wf_zero_gaps::interval>writes,gaps;writes.reserve(layer.spans.size());
            for(size_t i=0;i<layer.spans.size();++i){
                const auto&span=layer.spans[i];
                if(covered[i]!=span.nbytes||(span.is_expert&&!wf_span_is_active_expert(wf,layer,span)))return refuse("incomplete or filtered source span");
                writes.push_back({span.gpu_offset,span.nbytes});
            }
            std::string reason;
            if(!llama_wf_zero_gaps::plan(layer.total_bytes,writes,gaps,reason))return refuse(reason.c_str());
            uint64_t zeroed=0;
            for(const auto&gap:gaps){
                if(!cuda_ok(cudaMemsetAsync((char*)target_buf+gap.offset,0,size_t(gap.bytes),wf_s)))return false;
                zeroed+=gap.bytes;
            }
            if(llama_dsv41_layer_log_on())WF_ERR("DS4PERF_DIRECT_ZERO_GAPS stage=%d extent=%llu zero_bytes=%llu gaps=%zu spans=%zu all_request_bytes_verified=1 before_DS_sync_retained=1",
                layer_idx,(unsigned long long)layer.total_bytes,(unsigned long long)zeroed,gaps.size(),writes.size());
        }else{
            size_t begin=layer.total_bytes;for(const auto&span:layer.spans)begin=std::min(begin,span.gpu_offset);
            if(begin&&!wf->required_dstorage)return false;
            if(!cuda_ok(cudaMemsetAsync((char*)target_buf+begin,0,layer.total_bytes-begin,wf_s)))return false;
        }
        if(!cuda_ok(cudaStreamSynchronize(wf_s)))return false;
    } else if (wave_start <= 0 && wf->ds_direct_mode) {
        // R3 DS-direct DEBUG: temporarily SKIP the prefetch zero — testing whether the CUDA memset was
        // racing/clobbering the DirectStorage writes to quarter 0 (heap offset 0). DSPLACED_TEST (no memset)
        // proved quarter-0 DS writes land; the only live-path difference is this zero. If skipping it makes
        // quarter 0 fill, the memset is the culprit and we move it (once at alloc / after DS / DS-side zero).
    }

    // Handle mmap fallback spans (only on first pass / full transfer)
    if (wave_start <= 0 && layer_idx < (int)wf->ds_mmap_spans.size()) {
        for (auto & ms : wf->ds_mmap_spans[layer_idx]) {
            if (ms.span_idx >= 0 && ms.span_idx < (int) layer.spans.size()) {
                const auto & span = layer.spans[ms.span_idx];
                if (span.is_expert && !wf_span_is_active_expert(wf, layer, span)) continue;
            }
            void * dst = dest_buf ? (char *)target_buf + ms.gpu_offset
                                  : (char *)default_buf + ms.gpu_offset;
            if (wf_s) {
                cudaMemcpyAsync(dst, ms.src, ms.nbytes, cudaMemcpyHostToDevice, wf_s);
            } else {
                cudaMemcpy(dst, ms.src, ms.nbytes, cudaMemcpyHostToDevice);
            }
        }
    }

    // Process waves (optionally a subset for split transfers)
    auto & waves = wf->ds_waves[layer_idx];
    int drain_threshold = wf->ds_queue_cap / 2;
    int ws = (wave_start >= 0) ? wave_start : 0;
    int we = (wave_end >= 0) ? std::min(wave_end, (int)waves.size()) : (int)waves.size();
    auto interval_ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
    const double initialize_ms=interval_ms(t0,std::chrono::high_resolution_clock::now());
    double staging_drain_ms=0,enqueue_ms=0,storage_fence_ms=0,cuda_handoff_api_ms=0;
    for (int w = ws; w < we; w++) {
        auto & wave = waves[w];
        if (wave.reqs.empty()) continue;

        // Async-drain mode: before this wave's DS read overwrites the shared staging
        // buffer, wait (host-side) for the PRIOR wave's async D2D — which read that
        // staging — to finish. cudaEventSynchronize doesn't hold the CUDA launch lock,
        // so the host's compute burst keeps running; this only waits for the copy
        // engine (~8.7ms), not a lock window. Single-staging-safe (no double-buffer).
        const auto drain_started=std::chrono::high_resolution_clock::now();
        if (async_drain && wf_s && staging_busy && wf->ds_stage_evt[0]) {
            if(!cuda_ok(cudaEventSynchronize(wf->ds_stage_evt[0])))return false;
            staging_busy = false;
        }
        const auto enqueue_started=std::chrono::high_resolution_clock::now();
        staging_drain_ms+=interval_ms(drain_started,enqueue_started);

        // Drain any prior queue state
        queue->Submit();

        // Enqueue all requests for this wave
        int submitted = 0;
        int wave_enqueued = 0;   // total this wave (submitted resets at drain boundaries)
        for (int ri = 0; ri < (int)wave.reqs.size(); ++ri) {
            auto & req = wave.reqs[ri];
            const int span_idx = ri < (int)wave.req_span_idx.size() ? wave.req_span_idx[ri] : -1;
            if (span_idx >= 0 && span_idx < (int) layer.spans.size()) {
                const auto & span = layer.spans[span_idx];
                if (span.is_expert && !wf_span_is_active_expert(wf, layer, span)) continue;
            }
            if (span_idx >= 0 && span_idx < (int)cache_hit.size() && cache_hit[span_idx]) {
                continue;
            }
            if (submitted >= drain_threshold) {
                uint64_t drain_val = ++wf->ds_fence_counter;
                queue->EnqueueSignal(wf->ds_fence, drain_val);
                queue->Submit();
                if (!wf_ds_fence_wait(wf, drain_val)) return false;
                submitted = 0;
            }
            queue->EnqueueRequest(&req);
            submitted++;
            wave_enqueued++;
        }
        // [2026-08-07 stale-serve hunt] trace-gated: distinguishes "requests skipped at enqueue"
        // (wave_enqueued < reqs) from "requests failed in DS" (error record below).
        WF_LOG("transfer layer %d wave %d: enqueued %d/%d requests (%.1f MB)",
               layer_idx, w, wave_enqueued, (int)wave.reqs.size(), wave.nbytes / 1e6);

        // Signal fence and submit
        uint64_t fence_val = ++wf->ds_fence_counter;
        queue->EnqueueSignal(wf->ds_fence, fence_val);
        queue->Submit();
        const auto storage_started=std::chrono::high_resolution_clock::now();
        enqueue_ms+=interval_ms(enqueue_started,storage_started);
        if (!wf_ds_fence_wait(wf, fence_val)) return false;
        const auto handoff_started=std::chrono::high_resolution_clock::now();
        storage_fence_ms+=interval_ms(storage_started,handoff_started);

        // A failed DS request can still be followed by a completed fence signal.
        // Refuse the staging D2D unless the queue reports that this wave succeeded.
        if (!wf_dstorage_wave_succeeded(wf, queue, layer_idx, w)) return false;

        // Order CUDA after the D3D12 signal before CUDA reads the imported
        // allocation. DS-direct is consumed later on the default stream; the
        // staging path is consumed immediately on wf_s below.
        cudaStream_t ds_consumer_stream = wf->params.ds_compute_direct ? wf_s : wf->ds_direct_mode ? (cudaStream_t)0 : wf_s;
        if (!wf_cuda_wait_ds_fence(wf, fence_val, ds_consumer_stream)) return false;

        // R3 DS-direct edit 10: SKIP the staging->target D2D — DS wrote this wave DIRECTLY into the
        // prefetch quarter (target_buf == prefetch buffer, quarter w at heap offset w*QSZ). The full
        // layer is now contiguous in prefetch; the single prefetch->compute D2D happens at the layer
        // boundary in wf_wait_layer_gpu (edit 13). No staging buffer exists in this mode.
        if (!wf->ds_direct_mode) {
            // D2D copy: staging → destination buffer
            size_t copy_bytes = (size_t)wave.nbytes;
            // In relay mode (dest_buf set), write at dest_offset + relative position
            // within the selected wave range. In normal mode, use wave.compute_offset.
            size_t d2d_dst_off = dest_buf
                ? (dest_offset + wave.compute_offset - waves[ws].compute_offset)
                : wave.compute_offset;
            if (wf_s) {
                if(!cuda_ok(cudaMemcpyAsync((char *)target_buf + d2d_dst_off,
                                stg.cuda_ptr, copy_bytes, cudaMemcpyDeviceToDevice, wf_s)))return false;
            } else {
                if (!cuda_ok(cudaMemcpy((char *)target_buf + d2d_dst_off,
                           stg.cuda_ptr, copy_bytes, cudaMemcpyDeviceToDevice))) return false;
            }
        }

        // Staging copies the whole wave, including stale bytes where cached DS
        // requests were omitted. Re-apply cached spans after that D2D. The same
        // copy is the final fill of the placed prefetch heap in DS-direct mode.
        if (cache_bytes) {
            const uint64_t wave_begin = wave.compute_offset;
            const uint64_t wave_end_offset = wave.compute_offset + wave.nbytes;
            for (int si = 0; si < (int)layer.spans.size(); ++si) {
                if (!cache_hit[si]) continue;
                const auto & span = layer.spans[si];
                const uint64_t span_begin = span.gpu_offset;
                const uint64_t span_end = span.gpu_offset + span.nbytes;
                if (span_begin < wave_end_offset && span_end > wave_begin) {
                    wf_cache_copy_existing(wf, layer_idx, span,
                            (char *)target_buf + span.gpu_offset);
                }
            }
        }

        // Async-drain: record the drain event so the NEXT wave (and the per-layer
        // fence) can wait for this async D2D to finish reading the staging buffer.
        // R3 DS-direct edit 11: no staging buffer exists, so nothing to drain.
        if (!wf->ds_direct_mode && async_drain && wf_s && wf->ds_stage_evt[0]) {
            if(!cuda_ok(cudaEventRecord(wf->ds_stage_evt[0], wf_s)))return false;
            staging_busy = true;
        }

        // Cross-stream barrier: when running on the default stream we use a
        // D2H read to force ordering with the compute stream. With wf_s
        // (a dedicated copy stream) the cudaEventRecord at function exit +
        // wf_cuda_wait_upload's cudaStreamWaitEvent in the consumer make this
        // unnecessary — and the D2H barrier would itself serialize copy_stream
        // with the host, defeating the overlap we're trying to achieve.
        if (!wf_s) {
            uint8_t barrier[16] = {};
            if (!cuda_ok(cudaMemcpy(barrier, target_buf, 16, cudaMemcpyDeviceToHost))) return false;
            (void)barrier;
        }
        cuda_handoff_api_ms+=interval_ms(handoff_started,std::chrono::high_resolution_clock::now());
    }

    // Async-drain per-layer visibility fence: all the layer's async D2Ds must be
    // finished AND globally visible to the compute stream before this transfer
    // returns (the caller's wf_wait_layer_gpu then lets compute read target_buf).
    // cudaStreamSynchronize drains copy_stream; the 16-byte D2H forces the
    // D3D12->CUDA visibility the event-only path lacked (= the all-zero-K fix),
    // now ONCE per layer instead of per wave.
    const auto visibility_started=std::chrono::high_resolution_clock::now();
    if (async_drain && wf_s) {
        if(!cuda_ok(cudaStreamSynchronize(wf_s)))return false;
        uint8_t barrier[16] = {};
        if(wf->params.ds_compute_direct){
            if(!cuda_ok(cudaMemcpyAsync(barrier,target_buf,16,cudaMemcpyDeviceToHost,wf_s))||!cuda_ok(cudaStreamSynchronize(wf_s)))return false;
        }else if(!cuda_ok(cudaMemcpy(barrier, target_buf, 16, cudaMemcpyDeviceToHost)))return false;
        (void)barrier;
        staging_busy = false;
    }

    // For GPU_BUFS=2 with default stream: the 16-byte D2H above forces D2D ordering.
    // For wf_s mode: record upload_done on copy_stream so the compute stream can
    // wait for it via wf_cuda_wait_upload (cudaStreamWaitEvent path at L3199).
    if ((wf_s || cache_bytes) && gpu_idx >= 0 && gpu_idx < (int) wf->gpu_bufs.size()
            && wf->gpu_bufs[gpu_idx].upload_done) {
        if(!cuda_ok(cudaEventRecord(wf->gpu_bufs[gpu_idx].upload_done, cache_bytes ? wf->copy_stream : wf_s)))return false;
    }
    if (cache_bytes && wf->ds_direct_mode) {
        // The caller immediately consumes prefetch_buf on the default stream,
        // which is not ordered with the WF non-blocking copy stream.
        cudaStreamSynchronize(wf->copy_stream);
    }
    for (int si = 0; si < (int)layer.spans.size(); ++si) {
        if (cache_hit[si]) wf_cache_unpin(wf, layer_idx, layer.spans[si]);
    }

    // Snapshot exactly which predicted spans consumed backing-store bandwidth.
    // wf_set_active_experts uses this after routing to account waste precisely.
    {
        std::lock_guard<std::mutex> predictor_lock(wf->expert_prefetch_mutex);
        if (layer.expert_prefetch_planned) {
            layer.expert_prefetch_loaded_spans.assign(layer.spans.size(), false);
            for (int si = 0; si < (int) layer.spans.size(); ++si) {
                const auto & span = layer.spans[si];
                if (span.is_expert && wf_span_is_active_expert(wf, layer, span) && !cache_hit[si]) {
                    layer.expert_prefetch_loaded_spans[si] = true;
                }
            }
            layer.expert_prefetch_loaded = true;
        }
    }

    // NOTE: Small tensors in wave 1+ may fail to load via DS due to the offset
    // mismatch between span registration (sequential packing) and ggml allocator
    // (which uses different alignment/padding). The post-remap repair in
    // wf_adapter_remap_tensors handles this by re-reading from GGUF and uploading
    // directly to the tensor's final data pointer.

    if(wf->params.ds_compute_direct)if(llama_dsv41_layer_log_on())WF_ERR("DS4PERF_DIRECT_FILLED stage=%d bank=%d waves=%d staging_D2D_bytes=0",layer_idx,gpu_idx,we-ws);
    auto t1 = std::chrono::high_resolution_clock::now();
    layer.nvme_read_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    if(wf->required_dstorage)if(llama_dsv41_layer_log_on())WF_ERR("[dsv41-fetch-parts] stage=%d initialize_ms=%.6f staging_drain_ms=%.6f enqueue_ms=%.6f storage_fence_ms=%.6f cuda_handoff_api_ms=%.6f final_visibility_ms=%.6f total_ms=%.6f waves=%d",
        layer_idx,initialize_ms,staging_drain_ms,enqueue_ms,storage_fence_ms,cuda_handoff_api_ms,interval_ms(visibility_started,t1),interval_ms(t0,t1),we-ws);
    wf->stat_bytes_read_nvme.fetch_add(layer.total_bytes - std::min(layer.total_bytes, cache_bytes));
    wf->ds_layer_fence_val[layer_idx] = wf->ds_fence_counter.load();

    WF_TRACE(wf, "dstorage_transfer: layer %d → buf %d: %zu waves, %.1f MB, %.2fms",
             layer_idx, gpu_idx, waves.size(),
             layer.total_bytes / (1024.0 * 1024.0), layer.nvme_read_ms);
    return true;
}

// Transfer a byte range [byte_start, byte_end) of a layer to dest_buf at offset 0.
// Used by the relay pattern: DMA half of layer → prefetch_buf → D2D → compute_buf.
// Processes only waves that overlap the byte range, clamping D2D copies as needed.
static bool wf_dstorage_transfer_range(wf_context * wf, int layer_idx,
                                        void * dest_buf, size_t byte_start, size_t byte_end) {
    if (wf->ds_failed.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> ds_lock(wf->ds_queue_mutex);
    auto & layer = wf->layers[layer_idx];
    IDStorageQueue * queue = wf->ds_queue;
    auto & stg = wf->ds_stage;
    auto & waves = wf->ds_waves[layer_idx];
    int drain_threshold = wf->ds_queue_cap / 2;

    cudaMemset(dest_buf, 0, byte_end - byte_start);

    // Track output position in dest_buf
    size_t dest_written = 0;

    for (int w = 0; w < (int)waves.size(); w++) {
        auto & wave = waves[w];
        size_t wave_start = (size_t)wave.compute_offset;
        size_t wave_end = wave_start + (size_t)wave.nbytes;

        // Skip waves entirely before or after our range
        if (wave_end <= byte_start || wave_start >= byte_end) continue;

        // This wave overlaps our range — transfer it fully via DS
        if (!wave.reqs.empty()) {
            queue->Submit();
            int submitted = 0;
            for (auto & req : wave.reqs) {
                if (submitted >= drain_threshold) {
                    uint64_t dv = ++wf->ds_fence_counter;
                    queue->EnqueueSignal(wf->ds_fence, dv);
                    queue->Submit();
                    if (!wf_ds_fence_wait(wf, dv)) return false;
                    submitted = 0;
                }
                queue->EnqueueRequest(&req);
                submitted++;
            }
            uint64_t fv = ++wf->ds_fence_counter;
            queue->EnqueueSignal(wf->ds_fence, fv);
            queue->Submit();
            if (!wf_ds_fence_wait(wf, fv)) return false;
            if (!wf_dstorage_wave_succeeded(wf, queue, layer_idx, w)) return false;
            if (!wf_cuda_wait_ds_fence(wf, fv, (cudaStream_t) 0)) return false;
        }

        // D2D: copy only the overlapping portion from staging to dest_buf
        size_t overlap_start = std::max(wave_start, byte_start);
        size_t overlap_end   = std::min(wave_end, byte_end);
        size_t stg_offset    = overlap_start - wave_start;  // offset within staging
        size_t copy_bytes    = overlap_end - overlap_start;

        cudaMemcpy((char *)dest_buf + dest_written,
                   (char *)stg.cuda_ptr + stg_offset,
                   copy_bytes, cudaMemcpyDeviceToDevice);
        dest_written += copy_bytes;
    }

    // Serialize the default-stream D2D copies without blocking unrelated CUDA
    // streams. This matches the full-layer DS path and avoids global
    // device-wide stalls during layer-major streaming prefill.
    if (dest_written > 0) {
        uint8_t barrier[16] = {};
        cudaMemcpy(barrier, dest_buf, std::min<size_t>(sizeof(barrier), dest_written), cudaMemcpyDeviceToHost);
        (void) barrier;
    }
    return true;
}

// Drain both sides of the D3D12/CUDA interop contract before destroying any
// queue, fence, semaphore, event, or imported-memory handle.
static void wf_dstorage_quiesce(wf_context * wf) {
    if (!wf || !wf->ds_initialized) return;

    if (wf->ds_queue && wf->ds_fence) {
        uint64_t tail = ++wf->ds_fence_counter;
        wf->ds_queue->EnqueueSignal(wf->ds_fence, tail);
        wf->ds_queue->Submit();
        wf_ds_fence_wait(wf, tail);
    }

    if (wf->copy_stream) {
        cudaError_t err = cudaStreamSynchronize(wf->copy_stream);
        if (err != cudaSuccess) {
            WF_ERR("DirectStorage teardown copy-stream sync failed: %s", cudaGetErrorString(err));
        }
    }
    if (wf->cuda_compute_stream) {
        cudaError_t err = cudaStreamSynchronize(wf->cuda_compute_stream);
        if (err != cudaSuccess) {
            WF_ERR("DirectStorage teardown compute-stream sync failed: %s", cudaGetErrorString(err));
        }
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        WF_ERR("DirectStorage teardown device sync failed: %s", cudaGetErrorString(err));
    }
}

#endif // WF_HAS_DSTORAGE

// ---------------------------------------------------------------------------
// Predictive expert prefetch
// ---------------------------------------------------------------------------

static int wf_env_positive_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return end != value && end && *end == '\0' && parsed > 0 && parsed <= INT_MAX
            ? (int) parsed : fallback;
}

static bool wf_expert_fully_cached(wf_context * wf, int layer_idx, int expert_idx) {
    bool found = false;
    for (const auto & span : wf->layers[layer_idx].spans) {
        if (!span.is_expert || span.expert_idx != expert_idx) continue;
        found = true;
        if (!wf_cache_contains(wf, layer_idx, span)) return false;
    }
    return found;
}

static void wf_expert_prefetch_init(wf_context * wf) {
    if (!wf || wf->expert_prefetch_initialized) return;
    wf->expert_prefetch_initialized = true;
    if (wf->required_dstorage) return;
    const char * gate = std::getenv("LLAMA_WF_EXPERT_PREFETCH");
    if (!(gate && gate[0] && gate[0] != '0')) return;
    if (wf->params.moe_mode != WF_MOE_FETCH_ACTIVE) {
        WF_ERR("warning: LLAMA_WF_EXPERT_PREFETCH requires WF_MOE_FETCH_ACTIVE; disabling");
        return;
    }
#if !defined(WF_HAS_DSTORAGE) || !defined(WF_HAS_CUDA)
    WF_ERR("warning: LLAMA_WF_EXPERT_PREFETCH requires DirectStorage + CUDA; disabling");
    return;
#else
    if (!wf->ds_initialized || wf->use_layer_files || !wf->copy_stream) {
        WF_ERR("warning: LLAMA_WF_EXPERT_PREFETCH requires GGUF-shard DirectStorage and a copy stream; disabling");
        return;
    }
    wf->expert_prefetch_n = wf_env_positive_int("LLAMA_WF_EXPERT_PREFETCH_N", 0);
    wf->expert_prefetch_dist = wf_env_positive_int("LLAMA_WF_EXPERT_PREFETCH_DIST", 2);
    wf->expert_prefetch_ema.resize(wf->layers.size());
    wf->expert_prefetch_last_active.assign(wf->layers.size(), 0);
    for (int il = 0; il < (int) wf->layers.size(); ++il) {
        int max_expert = -1;
        for (const auto & span : wf->layers[il].spans) {
            if (span.is_expert) max_expert = std::max(max_expert, span.expert_idx);
        }
        if (max_expert >= 0) wf->expert_prefetch_ema[il].assign((size_t) max_expert + 1, 0.0f);
    }
    wf->expert_prefetch_enabled = true;
    wf->expert_prefetch_passes = 1;
    WF_ERR("predictive expert prefetch enabled: EMA alpha=0.25, P=%s, DIST=%d",
           wf->expert_prefetch_n ? std::to_string(wf->expert_prefetch_n).c_str() : "1.5x routed",
           wf->expert_prefetch_dist);
#endif
}

static void wf_expert_prefetch_schedule(wf_context * wf, int current_layer_idx) {
    if (!wf || !wf->expert_prefetch_enabled || wf->dir != 1) return;
    const int target = current_layer_idx + wf->expert_prefetch_dist;
    if (target < 0 || target >= (int) wf->layers.size()) return;

    std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
    auto & layer = wf->layers[target];
    auto & scores = wf->expert_prefetch_ema[target];
    if (scores.empty() || layer.stage.load() != WF_STAGE_IDLE || layer.expert_filter_set) return;
    int p = wf->expert_prefetch_n;
    if (p <= 0) p = (wf->expert_prefetch_last_active[target] * 3 + 1) / 2;
    if (p <= 0) return; // no routing history for this layer yet

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(scores.size());
    for (int eid = 0; eid < (int) scores.size(); ++eid) {
        if (scores[eid] > 0.0f && !wf_expert_fully_cached(wf, target, eid)) {
            ranked.emplace_back(scores[eid], eid);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });
    if ((int) ranked.size() > p) ranked.resize(p);
    if (ranked.empty()) return;

    layer.expert_prefetch_ids.clear();
    for (const auto & item : ranked) layer.expert_prefetch_ids.push_back(item.second);
    layer.active_experts = layer.expert_prefetch_ids;
    layer.expert_filter_set = true;
    layer.expert_prefetch_planned = true;
    layer.expert_prefetch_loaded = false;
    layer.expert_prefetch_loaded_spans.clear();
    WF_TRACE(wf, "expert prefetch: layer %d planned %zu experts at distance %d",
             target, layer.expert_prefetch_ids.size(), wf->expert_prefetch_dist);
}

static void wf_expert_prefetch_update_ema(wf_context * wf, int layer_idx,
                                          const int * expert_ids, int n_active) {
    auto & scores = wf->expert_prefetch_ema[layer_idx];
    if (scores.empty()) return;
    for (float & score : scores) score *= 0.75f;
    std::vector<bool> seen(scores.size(), false);
    for (int i = 0; i < n_active; ++i) {
        int eid = expert_ids[i];
        if (eid >= 0 && eid < (int) scores.size() && !seen[eid]) {
            scores[eid] += 0.25f;
            seen[eid] = true;
        }
    }
    wf->expert_prefetch_last_active[layer_idx] = n_active;
}

#if defined(WF_HAS_DSTORAGE) && defined(WF_HAS_CUDA)
static void wf_expert_prefetch_fetch_misses(wf_context * wf, int layer_idx,
                                            const std::vector<int> & misses) {
    if (misses.empty()) return;
    auto & layer = wf->layers[layer_idx];
    const int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0 || gpu_idx >= (int) wf->gpu_bufs.size()) return;
    void * target = wf->gpu_bufs[gpu_idx].ptr;
    auto is_miss = [&](int eid) {
        return std::find(misses.begin(), misses.end(), eid) != misses.end();
    };

    std::vector<bool> backing_miss(layer.spans.size(), false);
    size_t ds_bytes = 0;
    for (int si = 0; si < (int) layer.spans.size(); ++si) {
        const auto & span = layer.spans[si];
        if (!span.is_expert || !is_miss(span.expert_idx)) continue;
        void * dst = (char *) target + span.gpu_offset;
        if (wf_cache_lookup(wf, layer_idx, span, dst, false)) continue;
        if (span.file_offset == 0 && span.mmap_addr) {
            cudaMemcpyAsync(dst, span.mmap_addr, span.nbytes, cudaMemcpyHostToDevice, wf->copy_stream);
        } else {
            backing_miss[si] = true;
        }
    }

    {
        std::lock_guard<std::mutex> ds_lock(wf->ds_queue_mutex);
        const int drain_threshold = std::max(1, wf->ds_queue_cap / 2);
        for (auto & wave : wf->ds_waves[layer_idx]) {
            int submitted = 0;
            for (int ri = 0; ri < (int) wave.reqs.size(); ++ri) {
                int si = ri < (int) wave.req_span_idx.size() ? wave.req_span_idx[ri] : -1;
                if (si < 0 || si >= (int) backing_miss.size() || !backing_miss[si]) continue;
                if (submitted >= drain_threshold) {
                    uint64_t drain = ++wf->ds_fence_counter;
                    wf->ds_queue->EnqueueSignal(wf->ds_fence, drain);
                    wf->ds_queue->Submit();
                    if (!wf_ds_fence_wait(wf, drain)) return;
                    submitted = 0;
                }
                wf->ds_queue->EnqueueRequest(&wave.reqs[ri]);
                ds_bytes += wave.reqs[ri].Source.File.Size;
                ++submitted;
            }
            if (!submitted) continue;
            uint64_t fence = ++wf->ds_fence_counter;
            wf->ds_queue->EnqueueSignal(wf->ds_fence, fence);
            wf->ds_queue->Submit();
            if (!wf_ds_fence_wait(wf, fence)) return;
            if (!wf_dstorage_wave_succeeded(wf, wf->ds_queue, layer_idx, -1)) return;
            if (!wf_cuda_wait_ds_fence(wf, fence, wf->copy_stream)) return;

            const uint64_t wave_begin = wave.compute_offset;
            const uint64_t wave_end = wave.compute_offset + wave.nbytes;
            for (int si = 0; si < (int) layer.spans.size(); ++si) {
                if (!backing_miss[si]) continue;
                const auto & span = layer.spans[si];
                const uint64_t begin = std::max<uint64_t>(span.gpu_offset, wave_begin);
                const uint64_t end = std::min<uint64_t>(span.gpu_offset + span.nbytes, wave_end);
                if (begin >= end) continue;
                const char * src = wf->ds_direct_mode
                        ? (const char *) wf->prefetch_buf + begin
                        : (const char *) wf->ds_stage.cuda_ptr + (begin - wave_begin);
                cudaMemcpyAsync((char *) target + begin, src, end - begin,
                                cudaMemcpyDeviceToDevice, wf->copy_stream);
            }
        }
        // The DS destination is shared with ordinary layer transfers. Drain its
        // consumers before releasing the queue to the DMA thread for reuse.
        if (ds_bytes) cudaStreamSynchronize(wf->copy_stream);
    }
    wf->stat_bytes_read_nvme.fetch_add(ds_bytes, std::memory_order_relaxed);
    if (wf->gpu_bufs[gpu_idx].upload_done) {
        cudaEventRecord(wf->gpu_bufs[gpu_idx].upload_done, wf->copy_stream);
    }
}
#endif

// ---------------------------------------------------------------------------
// DMA control thread
//
// Polls compute_done events on ALL GPU buffers and fires DS Submit() the
// instant each buffer's compute finishes.  Per-buffer pending slots mean
// buffer A's DMA starts immediately even while buffer B's compute is still
// running — no single-thread blocking bottleneck.
//
// Flow:
//   consumer: record compute_done → set dma_buf_pending[gpu_idx] = next_layer
//   DMA thread: poll cudaEventQuery on each pending buf → enqueue + Submit()
//   consumer: continues immediately — zero blocking
// ---------------------------------------------------------------------------

#ifdef WF_HAS_CUDA
static bool wf_lockfree_ready_enabled();  // fwd decl (defined near wf_cuda_record_compute_done)
#endif

#ifdef WF_HAS_DSTORAGE
static void wf_dma_thread_fn(wf_context * wf) {
#ifdef WF_HAS_CUDA
    // Attach this thread to the same CUDA context as the main thread.
    // Without this, cudaEventQuery triggers lazy context init that can
    // corrupt the primary context (cudaErrorSharedObjectInitFailed).
    if (wf->cuda_device_index >= 0) {
        cudaError_t err = cudaSetDevice(wf->cuda_device_index);
        if (err != cudaSuccess) {
            WF_ERR("DMA thread cudaSetDevice(%d) failed: %s",
                   wf->cuda_device_index, cudaGetErrorString(err));
            if (wf->required_dstorage) { wf->ds_failed.store(true,std::memory_order_release); return; }
        }
    }
#endif
    WF_LOG("DMA control thread started (polling %d buffers)", (int)wf->gpu_bufs.size());

    while (wf->pipeline_running.load()) {
        bool did_work = false;

        // Serialize DMA submissions: only one DMA in-flight at a time.
        // Multiple concurrent DS queues fight for PCIe bandwidth, reducing
        // throughput from 20 GB/s to 7 GB/s per transfer.
        int in_flight = wf->dma_in_flight_buf.load();
        if (in_flight >= 0) {
            // A DMA is already in-flight — don't submit another.
            // The pipeline thread clears dma_in_flight_buf when the fence signals.
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }

        for (int bi = 0; bi < (int)wf->gpu_bufs.size(); bi++) {
            int layer_idx = wf->dma_buf_pending[bi].pending_layer.load();
            if (layer_idx < 0) continue;

            // Non-blocking check: has GPU finished computing on this buffer?
#ifdef WF_HAS_CUDA
            if (!wf->required_dstorage && wf_lockfree_ready_enabled()) {
                // Lock-free: poll the host atomic set by the compute-stream callback.
                // No cudaEventQuery -> no CUDA-driver-lock contention with the host.
                if (!wf->dma_buf_pending[bi].compute_ready.load(std::memory_order_acquire)) continue;
            } else if (wf->gpu_bufs[bi].compute_done) {
                cudaError_t err = cudaEventQuery(wf->gpu_bufs[bi].compute_done);
                if (err == cudaErrorNotReady) continue;  // GPU still computing
                if (wf->required_dstorage && err != cudaSuccess) {
                    WF_ERR("Required DirectStorage: compute completion query failed: %s",cudaGetErrorString(err));
                    wf->ds_failed.store(true,std::memory_order_release);return;
                }
                // cudaSuccess or other — safe to overwrite buffer
            }
#endif

            // LLAMADECODE B1: a routed-expert stage is fetched only after the caller
            // publishes this layer's routing; it is never fetched unfiltered.
            if (wf->params.moe_mode == WF_MOE_FETCH_ACTIVE) {
                bool routed = false, ready = true;
                for (const auto & span : wf->layers[layer_idx].spans) if (span.is_expert) { routed = true; break; }
                if (routed) { std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex); ready = wf->layers[layer_idx].expert_filter_set; }
                if (!ready) continue;
            }

            // Compute done — submit DMA immediately
            wf->dma_buf_pending[bi].pending_layer.store(-1);
            if (wf->layers[layer_idx].persistent && wf->layers[layer_idx].persistent_loaded.load(std::memory_order_acquire)) {
                // LLAMADECODE B1b: resident in this bank's own region since its first
                // transfer; nothing else writes it. Publish without a DMA.
                wf->gpu_bufs[bi].in_use = true;
                wf->gpu_bufs[bi].layer_idx = layer_idx;
                wf->layers[layer_idx].gpu_buf_idx = bi;
                {
                    std::lock_guard<std::mutex> lk(wf->dma_timing_mutex);
                    const auto now = std::chrono::high_resolution_clock::now();
                    wf->dma_timing_log.push_back({layer_idx, bi, 0, now, now});
                }
                wf->layers[layer_idx].stage.store(WF_STAGE_GPU_READY);
                wf->pipeline_cv.notify_one();
                did_work = true;
                break;
            }

            // Skip empty sub-layers (dense layer's MoE slot)
            if (wf->layers[layer_idx].total_bytes == 0) {
                wf->layers[layer_idx].stage.store(WF_STAGE_DONE);
                // Schedule next layer on this buffer
                int next = wf->dma_next_layer.fetch_add(wf->dir);   // +dir: forward +1 (unchanged), reverse -1
                if (next >= 0 && next < (int)wf->layers.size()) {
                    wf->dma_buf_pending[bi].pending_layer.store(next);
                }
                did_work = true;
                continue;
            }

            // Blocking wave-based transfer: DS → staging → D2D → compute.
            // When this returns, the entire layer is in the compute buffer.
            wf->gpu_bufs[bi].in_use = true;
            wf->gpu_bufs[bi].layer_idx = layer_idx;
            wf->layers[layer_idx].gpu_buf_idx = bi;
            {   int expected = -1;  // LLAMADECODE B2: an extent fetch may hold the DS queue
                if (!wf->dma_in_flight_buf.compare_exchange_strong(expected, bi)) {
                    wf->dma_buf_pending[bi].pending_layer.store(layer_idx);
                    wf->gpu_bufs[bi].in_use = false; wf->gpu_bufs[bi].layer_idx = -1; wf->layers[layer_idx].gpu_buf_idx = -1;
                    break;
                }
            }
            // Lock-free mode: mark this buffer not-ready until the new layer's
            // compute callback fires (wf_cuda_record_compute_done -> wf_compute_ready_cb).
            wf->dma_buf_pending[bi].compute_ready.store(false, std::memory_order_release);

            {
                std::lock_guard<std::mutex> lk(wf->dma_timing_mutex);
                wf->dma_timing_log.push_back({layer_idx, bi, wf->layers[layer_idx].total_bytes,
                    std::chrono::high_resolution_clock::now(), {}});
            }

            if (!wf_dstorage_transfer_layer(wf, layer_idx, bi)) {
                // The queue/device is no longer usable. Do not publish the old
                // contents of this ring slot as GPU_READY.
                wf->gpu_bufs[bi].in_use = false;
                wf->gpu_bufs[bi].layer_idx = -1;
                wf->layers[layer_idx].gpu_buf_idx = -1;
                wf->dma_in_flight_buf.store(-1);
                wf->pipeline_cv.notify_all();
                break;
            }

            if (wf->layers[layer_idx].persistent) wf->layers[layer_idx].persistent_loaded.store(true, std::memory_order_release);
            // Transfer complete — mark GPU_READY immediately
            {
                std::lock_guard<std::mutex> lk(wf->dma_timing_mutex);
                if (!wf->dma_timing_log.empty()) {
                    auto & last = wf->dma_timing_log.back();
                    if (last.layer_idx == layer_idx)
                        last.t_fence = std::chrono::high_resolution_clock::now();
                    if (wf->required_dstorage && last.layer_idx == layer_idx)
                        if(llama_dsv41_layer_log_on())WF_ERR("[dsv41-fetch-timeline] stage=%d submit_epoch_ms=%.6f fence_epoch_ms=%.6f",
                            layer_idx,
                            std::chrono::duration<double,std::milli>(last.t_submit.time_since_epoch()).count(),
                            std::chrono::duration<double,std::milli>(last.t_fence.time_since_epoch()).count());
                }
            }
            wf->layers[layer_idx].stage.store(WF_STAGE_GPU_READY);
            wf->stats.layers_fetched++;
            wf->dma_in_flight_buf.store(-1);

            wf->pipeline_cv.notify_one();
            did_work = true;
            break;
        }

        if (!did_work) {
            // Brief sleep to avoid burning CPU — 50μs adds negligible latency
            // compared to 250ms DMA/compute times
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    WF_LOG("DMA control thread exiting");
}
#endif

// ---------------------------------------------------------------------------
// Pipeline scheduler
//
// Runs on its own thread, managing the 4-stage pipeline.  The key insight is
// that stages are non-blocking:
//
//   IDLE → NVME_READ:  submit async I/O (returns immediately)
//   NVME_READ → RAM_READY:  poll completion counter (set by IOCP workers)
//   RAM_READY → H2D_FLIGHT: cudaMemcpyAsync (returns immediately)
//   H2D_FLIGHT → GPU_READY: CUDA event (checked by compute stream, not us)
//
// The scheduler loop checks all in-flight layers each iteration and advances
// them when their current stage is done.  This keeps all pipeline stages busy
// concurrently across different layers.
// ---------------------------------------------------------------------------

static void wf_pipeline_thread_fn(wf_context * wf) {
    const int n_layers = wf->params.n_layers;
    const int depth    = wf->params.prefetch_depth;
    int loop_count = 0;
    auto heartbeat_time = std::chrono::high_resolution_clock::now();

    WF_LOG("pipeline thread entered: %d layers, depth=%d, %d RAM bufs, %d GPU bufs",
           n_layers, depth, (int)wf->ram_bufs.size(), (int)wf->gpu_bufs.size());

  try {
    while (wf->pipeline_running.load()) {
        int next = wf->pipeline_next_layer.load();
        bool any_progress = false;

        // --- Phase A: check DMA completion ---
#ifdef WF_HAS_DSTORAGE
        // Wave-based DS transfers: the DMA thread does blocking transfers
        // and sets GPU_READY directly. Nothing for the pipeline thread to do.
        if (wf->ds_initialized) {
            // no-op — DMA thread handles everything
        } else
#endif
        // Non-DirectStorage path: scan all layers for I/O completion
        {
        for (int il = 0; il < n_layers; il++) {
            auto & layer = wf->layers[il];
            wf_layer_stage stage = layer.stage.load();

            if (stage == WF_STAGE_NVME_READ) {
                if (wf_layer_io_done(wf, il)) {
                    layer.stage.store(WF_STAGE_RAM_READY);
                    any_progress = true;
                    WF_TRACE(wf, "pipeline: layer %d NVMe read complete (%.2fms)",
                             il, layer.nvme_read_ms);
                }
            }

            if (layer.stage.load() == WF_STAGE_H2D_FLIGHT) {
                if (wf_h2d_complete(wf, il)) {
                    int ram_idx = layer.ram_buf_idx;
                    if (ram_idx >= 0) {
                        wf->ram_bufs[ram_idx].in_use = false;
                        wf->ram_bufs[ram_idx].layer_idx = -1;
                        layer.ram_buf_idx = -1;
                    }
                    layer.stage.store(WF_STAGE_GPU_READY);
                    wf->stats.layers_fetched++;
                    any_progress = true;

                    WF_TRACE(wf, "pipeline: layer %d H2D complete → GPU_READY", il);
                }
            }
        }
        } // end non-DS path

        // Step 2: Upload RAM_READY layers to GPU — IN CONSUMER ORDER.
        //         The consumer processes layers sequentially (0, 1, 2, ...).
        //         With only N GPU buffers, uploading out of order can deadlock:
        //         layers K+1, K+2 fill both GPU buffers while K is still waiting,
        //         and the consumer can't reach K+1 until K is done.
        //         Fix: find the first non-consumed layer and only upload from there.
        {
            // Find the first layer that the consumer hasn't finished yet
            int first_needed = n_layers;
            for (int il = 0; il < n_layers; il++) {
                wf_layer_stage s = wf->layers[il].stage.load();
                if (s != WF_STAGE_DONE && s != WF_STAGE_COMPUTING &&
                    s != WF_STAGE_GPU_READY) {
                    first_needed = il;
                    break;
                }
            }

            // Upload at most n_gpu_bufs layers starting from first_needed
            int n_gpu = (int)wf->gpu_bufs.size();
            for (int il = first_needed; il < n_layers && il < first_needed + n_gpu; il++) {
                auto & layer = wf->layers[il];
                if (layer.stage.load() != WF_STAGE_RAM_READY) continue;

                int gpu_idx = wf_find_free_gpu_buf(wf);
                if (gpu_idx < 0) break;

                wf->gpu_bufs[gpu_idx].in_use = true;
                wf->gpu_bufs[gpu_idx].layer_idx = il;
                layer.gpu_buf_idx = gpu_idx;
                layer.stage.store(WF_STAGE_H2D_FLIGHT);

                wf_upload_to_gpu(wf, il);
                any_progress = true;

                WF_TRACE(wf, "pipeline: layer %d H2D enqueued (ram_buf=%d)", il, layer.ram_buf_idx);
            }
        }

        // --- Phase B: submit new layers into the pipeline ---
        //
        // With round-robin chunk distribution, each layer's I/O is interlocked
        // sequential.  Use fewer, larger layers to give the RAID controller
        // enough sequential runway to reach full speed.
        {
            int scan_end = std::min(n_layers, next + depth);

#ifdef WF_HAS_DSTORAGE
            // DMA control thread handles all DS submissions after startup.
            // Pipeline thread only does Phase A (fence checks → GPU_READY).
            // Startup layers were submitted directly in wf_start().
            if (wf->ds_initialized) {
                goto phase_b_done;
            }
#endif

            for (int il = 0; il < scan_end; il++) {
                auto & layer = wf->layers[il];
                if (layer.stage.load() != WF_STAGE_IDLE) continue;

#ifdef WF_HAS_GDS
                if (wf->effective_mode == WF_MODE_GDS && wf->gds_available) {
                    int gpu_idx = wf_find_free_gpu_buf(wf);
                    if (gpu_idx < 0) break;

                    wf->gpu_bufs[gpu_idx].in_use = true;
                    wf->gpu_bufs[gpu_idx].layer_idx = il;
                    layer.gpu_buf_idx = gpu_idx;
                    layer.stage.store(WF_STAGE_NVME_READ);

                    wf_gds_read_layer(wf, il);

                    layer.stage.store(WF_STAGE_GPU_READY);
                    wf->stats.layers_fetched++;
                    any_progress = true;
                    continue;
                }
#endif

                // Standard path: submit async NVMe reads → RAM staging buffer
                int ram_idx = wf_find_free_ram_buf(wf);
                if (ram_idx < 0) break;

                wf->ram_bufs[ram_idx].in_use = true;
                wf->ram_bufs[ram_idx].layer_idx = il;
                layer.ram_buf_idx = ram_idx;
                layer.stage.store(WF_STAGE_NVME_READ);

                WF_LOG("pipeline: submitting layer %d reads (ram_buf=%d, buf=%p)",
                       il, ram_idx, wf->ram_bufs[ram_idx].ptr);

                int n_chunks = wf_submit_layer_reads(wf, il, wf->ram_bufs[ram_idx].ptr);

                WF_LOG("pipeline: layer %d submitted %d chunks", il, n_chunks);

                if (n_chunks == 0) {
                    layer.stage.store(WF_STAGE_RAM_READY);
                }

                any_progress = true;
            }
        }

#ifdef WF_HAS_DSTORAGE
        phase_b_done:
#endif

        // --- Heartbeat: log pipeline state every ~1 second (detect stalls AND hangs) ---
        auto now = std::chrono::high_resolution_clock::now();
        float hb_elapsed = std::chrono::duration<float>(now - heartbeat_time).count();
        if (hb_elapsed >= 1.0f) {
            heartbeat_time = now;
            int n_idle = 0, n_nvme = 0, n_ram = 0, n_h2d = 0, n_gpu = 0, n_comp = 0, n_done = 0;
            for (int il = 0; il < n_layers; il++) {
                switch (wf->layers[il].stage.load()) {
                    case WF_STAGE_IDLE:      n_idle++; break;
                    case WF_STAGE_NVME_READ: n_nvme++; break;
                    case WF_STAGE_RAM_READY: n_ram++;  break;
                    case WF_STAGE_H2D_FLIGHT:n_h2d++;  break;
                    case WF_STAGE_GPU_READY: n_gpu++;  break;
                    case WF_STAGE_COMPUTING: n_comp++; break;
                    case WF_STAGE_DONE:      n_done++; break;
                }
            }
            // Count free buffers
            int free_ram = 0, free_gpu = 0;
            for (auto & s : wf->ram_bufs) if (!s.in_use) free_ram++;
            for (auto & s : wf->gpu_bufs) if (!s.in_use) free_gpu++;

            WF_LOG("HEARTBEAT: next=%d idle=%d nvme=%d ram=%d h2d=%d gpu=%d comp=%d done=%d | free_ram=%d free_gpu=%d progress=%s",
                   next, n_idle, n_nvme, n_ram, n_h2d, n_gpu, n_comp, n_done,
                   free_ram, free_gpu, any_progress ? "yes" : "NO");

            // Dump first few non-DONE layers for debugging
            int dumped = 0;
            for (int il = 0; il < n_layers && dumped < 4; il++) {
                wf_layer_stage s = wf->layers[il].stage.load();
                if (s != WF_STAGE_DONE && s != WF_STAGE_IDLE) {
                    auto & trk = wf->io_trackers[il];
                    WF_LOG("  layer[%d]: stage=%d ram=%d gpu=%d io=%d/%d",
                           il, (int)s, wf->layers[il].ram_buf_idx,
                           wf->layers[il].gpu_buf_idx,
                           trk.completed.load(), trk.total_chunks.load());
                    dumped++;
                }
            }
        }

        if (!any_progress) {
            loop_count++;
            // Poll aggressively when DMA is in-flight (fence check is the hot path).
            // Sleep longer when idle (no DMA, no pending work).
            int sleep_us = (wf->dma_in_flight_buf.load() >= 0) ? 5 : 100;
            std::unique_lock<std::mutex> lock(wf->pipeline_mutex);
            wf->pipeline_cv.wait_for(lock, std::chrono::microseconds(sleep_us));
        } else {
            loop_count = 0;
        }
    }
  } catch (const std::exception & e) {
    WF_ERR("PIPELINE THREAD CRASHED: %s", e.what());
  } catch (...) {
    WF_ERR("PIPELINE THREAD CRASHED: unknown exception");
  }
  WF_LOG("pipeline thread exiting");
}

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

static wf_context * wf_create_impl(const wf_params * params, bool required_dstorage) {
    if (!params || !params->model_path || params->n_layers <= 0) {
        WF_ERR("wf_create: invalid params");
        return nullptr;
    }

    if(params->ds_zero_padding_only&&(!required_dstorage||!params->ds_compute_direct||
       !params->ds_async_drain||params->ds_gpu_ring_slots||params->n_gpu_buffers!=2)){
        WF_ERR("Padding-only zero requires the explicit strict two-bank direct profile");
        return nullptr;
    }

    if (required_dstorage) {
#ifndef WF_HAS_DSTORAGE
        WF_ERR("Required DirectStorage is not compiled in; no fallback allocations attempted");
        return nullptr;
#else
        if (params->mode != WF_MODE_DSTORAGE || !params->gpu_backend ||
                !ggml_backend_is_cuda((ggml_backend_t)params->gpu_backend) ||
                !params->max_layer_bytes || params->max_layer_bytes > SIZE_MAX / 2 ||
                params->n_gpu_buffers < 1 || params->n_gpu_buffers > 2 ||
                !params->io_block_size || params->io_block_size > 256ull * 1024 * 1024 ||
                params->n_layers > 128 || params->n_sublayers != 1 ||
                (params->moe_mode != WF_MOE_FETCH_ALL && !(params->moe_mode == WF_MOE_FETCH_ACTIVE &&
                    params->ds_compute_direct && params->n_gpu_buffers == 2 && !params->ds_gpu_ring_slots &&
                    !params->ds_zero_padding_only)) ||
                params->n_split_files < 0 || params->n_split_files > 64 ||
                (params->n_split_files && !params->split_paths)) {
            WF_ERR("Required DirectStorage: invalid explicit backend/geometry");
            return nullptr;
        }
        // These modes can allocate unadmitted heaps/caches or change physical
        // weight-file ownership. A strict producer must qualify them separately.
        for (const char * name : {"LLAMA_WF_DS_DIRECT", "LLAMA_WF_PERSIST_INFRA",
                "LLAMA_WF_PERSIST_FACTORY", "LLAMA_WF_TEST_DUAL_BUF",
                "LLAMA_WF_TEST_DS_PLACED", "LLAMA_WF_TEST_DUAL_HEAPS",
                "LLAMA_WF_EXPERT_CACHE_MB", "LLAMA_WF_EXPERT_CACHE_RAM_MB",
                "LLAMA_WF_EXPERT_PREFETCH", "LLAMA_PREFILL_LAYER_DIR",
                "LLAMA_DS_FACTORY_STAGING_MB", "LLAMA_DS_STAGING_MB",
                "LLAMA_WF_DS_PREFETCH_PCT", "LLAMA_WF_PREFETCH_PCT",
                "LLAMA_WF_DS_USE_COPY_STREAM", "LLAMA_WF_ASYNC_DRAIN",
                "LLAMA_WF_RELAY_LARGE_LAYER", "LLAMA_WF_LOCKFREE_READY",
                "LLAMA_WF_WARM_INTEROP", "LLAMA_DSV4_WF_CKSUM"}) {
            const char * value = std::getenv(name);
            if (value && value[0]) {
                WF_ERR("Required DirectStorage: unqualified environment option %s", name);
                return nullptr;
            }
        }
        for (int i = 0; i < params->n_split_files; ++i) {
            if (!params->split_paths[i] || !params->split_paths[i][0]) return nullptr;
        }
#endif
    }

    // Typed profile admission happens before context/factory/heap allocation.
    // Ambient tuning still refuses in strict mode; the caller uses explicit
    // source-pinned settings rather than inheriting unbounded environment knobs.
    if ((params->ds_staging_limit_bytes || params->ds_async_drain) &&
            (!required_dstorage || !llama_wf_bounded::exact_relay_profile(
                params->max_layer_bytes,uint32_t(params->n_gpu_buffers),
                params->ds_staging_limit_bytes,params->io_block_size))) {
        WF_ERR("Required DirectStorage: unsupported typed relay profile");
        return nullptr;
    }

    if(params->ds_compute_direct&&(!required_dstorage||params->n_gpu_buffers!=2||params->n_ram_buffers||!params->ds_async_drain||params->ds_gpu_ring_slots||params->max_layer_bytes>SIZE_MAX/8)){
        WF_ERR("Required direct compute needs exactly two explicit VRAM banks, async copy ordering and no ring/RAM buffers");return nullptr;
    }
    if(params->ds_gpu_ring_slots && (!required_dstorage || params->ds_gpu_ring_slots!=2 ||
        !params->ds_async_drain || params->n_ram_buffers || params->n_layers<1 || params->n_layers>64 ||
        params->moe_mode!=WF_MOE_FETCH_ALL || !llama_wf_bounded::exact_two_buffer_profile(
            params->max_layer_bytes,uint32_t(params->n_gpu_buffers),params->ds_staging_limit_bytes,params->io_block_size))) {
        WF_ERR("Required DirectStorage: unsupported frontloaded ring profile");return nullptr;
    }
    wf_verbose = params->trace;

    auto * wf = new wf_context();
    wf->params = *params;
    wf->required_dstorage = required_dstorage;
    // R3 DS-DIRECT gate (default-off == today's exact behavior). Set BEFORE any mode-specific init.
    {
        const char * e = std::getenv("LLAMA_WF_DS_DIRECT");
        wf->ds_direct_mode = params->ds_compute_direct||(e && e[0] && e[0] != '0');
        if (wf->ds_direct_mode) WF_LOG("DS-DIRECT mode ENABLED (LLAMA_WF_DS_DIRECT): permanent ping-pong heaps, no staging tier");
    }
    wf->layers.resize(params->n_layers);
    wf->io_trackers.resize(params->n_layers);
    memset(&wf->stats, 0, sizeof(wf->stats));
#ifdef WF_HAS_CUDA
    wf->cuda_device_index = wf_resolve_cuda_device_index(params->gpu_backend);
    if (params->gpu_backend) {
        WF_LOG("resolved CUDA device index: %d", wf->cuda_device_index);
    }
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
    if (required_dstorage && (wf->cuda_device_index < 0 || !cuda_scope.ok)) {
        WF_ERR("Required DirectStorage: target CUDA device cannot be bound");
        delete wf;
        return nullptr;
    }
    if (required_dstorage && !ggml_backend_cuda_get_stream((ggml_backend_t)params->gpu_backend)) {
        WF_ERR("Required DirectStorage: target CUDA compute stream is unavailable");
        delete wf;
        return nullptr;
    }
#endif

    // -- Resolve transfer mode --
    wf->effective_mode = params->mode;
    if (params->mode == WF_MODE_AUTO) {
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE)
        // Prefer DirectStorage on Windows (NVMe → GPU via D3D12)
        wf->effective_mode = WF_MODE_DSTORAGE;
#elif defined(WF_HAS_GDS)
        // TODO: probe for GDS support
        wf->effective_mode = WF_MODE_GDS;
        wf->gds_available = true;
#else
        wf->effective_mode = WF_MODE_DIRECT_IO;
#endif
    }

#ifndef WF_HAS_DSTORAGE
    // DirectStorage requires Windows + LLAMA_DIRECTSTORAGE + CUDA at compile time
    if (wf->effective_mode == WF_MODE_DSTORAGE) {
        WF_LOG("WF_MODE_DSTORAGE not compiled in — falling back to DIRECT_IO");
        wf->effective_mode = WF_MODE_DIRECT_IO;
    }
#endif

    wf->stats.effective_mode = wf->effective_mode;

    WF_LOG("transfer mode: %s",
           wf->effective_mode == WF_MODE_DSTORAGE  ? "DirectStorage (NVMe → GPU via D3D12)" :
           wf->effective_mode == WF_MODE_GDS        ? "GPUDirect Storage" :
           wf->effective_mode == WF_MODE_DIRECT_IO  ? "Direct I/O" :
           wf->effective_mode == WF_MODE_MMAP       ? "mmap (legacy)" : "unknown");

    // -- Open file handles --
    {
        wf_file_handle fh;
        bool use_direct = (wf->effective_mode == WF_MODE_DIRECT_IO) && params->io_direct;
        if (!wf_open_file(&fh, params->model_path, use_direct)) {
            delete wf;
            return nullptr;
        }
        wf->files.push_back(fh);
        WF_LOG("opened %s (%.2f GB)", params->model_path,
               fh.file_size / (1024.0 * 1024.0 * 1024.0));
    }

    // Open split files
    for (int i = 0; i < params->n_split_files && params->split_paths; i++) {
        wf_file_handle fh;
        bool use_direct = (wf->effective_mode == WF_MODE_DIRECT_IO) && params->io_direct;
        if (!wf_open_file(&fh, params->split_paths[i], use_direct)) {
            WF_ERR("warning: failed to open split file %s", params->split_paths[i]);
            if (required_dstorage) { wf_destroy(wf); return nullptr; }
        }
        wf->files.push_back(fh);
    }

    // -- Allocate GPU ping-pong buffers --
    size_t buf_size = params->max_layer_bytes;
    if (buf_size == 0) {
        WF_ERR("warning: max_layer_bytes not set, using 512 MB default");
        buf_size = 512 * 1024 * 1024;
    }
    // Padding: 10% only when raw direct I/O can overread rounded chunks.
    // DirectStorage and buffered RAM-staged reads only need allocator alignment
    // slack, which matters for large Kimi runs that fit VRAM by a narrow margin.
    if (wf->effective_mode == WF_MODE_DSTORAGE ||
        (wf->effective_mode == WF_MODE_DIRECT_IO && !params->io_direct)) {
        buf_size += std::max(buf_size / 100, (size_t)(64 * 1024));
    } else {
        buf_size += buf_size / 10;
    }
    // Round up to 64KB for D3D12 / direct I/O alignment
    buf_size = (buf_size + 65535) & ~(size_t)65535;

    WF_LOG("GPU buffer allocation: %d bufs × %.2f MB = %.2f MB total (max_layer=%.2f MB, mode=%d)",
           params->n_gpu_buffers, buf_size / (1024.0 * 1024.0),
           (buf_size * params->n_gpu_buffers) / (1024.0 * 1024.0),
           params->max_layer_bytes / (1024.0 * 1024.0), (int)wf->effective_mode);

    // The legacy env-mode prefetch relay requires one compute slot. Strict
    // direct mode retains the two requested imported compute banks.
    wf->gpu_bufs.resize(wf->ds_direct_mode&&!params->ds_compute_direct ? 1 : params->n_gpu_buffers);

    bool gpu_bufs_via_dstorage = false;
#ifdef WF_HAS_DSTORAGE
    if (wf->effective_mode == WF_MODE_DSTORAGE) {
        if (wf_dstorage_init(wf, buf_size)) {
            gpu_bufs_via_dstorage = true;
        } else {
            if (required_dstorage) {
                WF_ERR("Required DirectStorage initialization failed; no fallback allocations attempted");
                wf_destroy(wf);
                return nullptr;
            }
            WF_ERR("DirectStorage init failed — falling back to DIRECT_IO");
            wf->effective_mode = WF_MODE_DIRECT_IO;
            wf->stats.effective_mode = WF_MODE_DIRECT_IO;
        }
    }
#endif

    if (!gpu_bufs_via_dstorage) {
        for (int i = 0; i < params->n_gpu_buffers; i++) {
            auto & slot = wf->gpu_bufs[i];
            slot.capacity = buf_size;

#ifdef WF_HAS_CUDA
            cudaError_t err = cudaMalloc(&slot.ptr, buf_size);
            if (err != cudaSuccess) {
                WF_ERR("cudaMalloc failed for GPU buffer %d (%.2f MB): %s",
                       i, buf_size / (1024.0 * 1024.0), cudaGetErrorString(err));
                for (int j = 0; j < i; j++) cudaFree(wf->gpu_bufs[j].ptr);
                for (auto & fh : wf->files) wf_close_file(&fh);
                delete wf;
                return nullptr;
            }
            slot.owns_ptr = true;
            if (!slot.upload_done)  cudaEventCreate(&slot.upload_done);
            if (!slot.compute_done) cudaEventCreate(&slot.compute_done);
#else
#ifdef _WIN32
            slot.ptr = _aligned_malloc(buf_size, 4096);
#else
            slot.ptr = aligned_alloc(4096, buf_size);
#endif
            slot.owns_ptr = true;
#endif
            WF_LOG("GPU buffer %d: %.2f MB at %p", i, buf_size / (1024.0 * 1024.0), slot.ptr);
        }
    }

    // Prefetch buffer: allocated lazily after graph_alloc, using remaining VRAM.
    // Set desired size here; actual allocation happens in wf_allocate_prefetch().
    {
        const char * pf_env = std::getenv("LLAMA_WF_PREFETCH_PCT");
        int pf_pct = (pf_env && pf_env[0]) ? std::atoi(pf_env) : 50;
        if (pf_pct < 0) pf_pct = 0;
        if (pf_pct > 100) pf_pct = 100;
        if (pf_pct > 0 && !wf->ds_direct_mode) {  // relay path for all configs (direct path has MoE corruption)
            wf->prefetch_buf_size = ((buf_size * pf_pct / 100) + 65535) & ~(size_t)65535;
            WF_LOG("prefetch: desired %.2f MB (%d%%), will allocate after graph_alloc",
                   wf->prefetch_buf_size / (1024.0 * 1024.0), pf_pct);
        }
        // R3 DS-direct: prefetch_buf + prefetch_buf_size were already set to the FULL permanent heap
        // (HEAP bytes) by wf_allocate_dual_use_heaps inside wf_dstorage_init; do NOT shrink them here.
        if (wf->ds_direct_mode) {
            WF_LOG("prefetch: DS-direct keeps full permanent heap (%.2f MB), skipping pct-based resize",
                   wf->prefetch_buf_size / (1024.0 * 1024.0));
        }
    }

    // DirectStorage: NVMe → GPU direct — no RAM buffers, chunk pool, or IOCP threads needed
    bool is_dstorage = (wf->effective_mode == WF_MODE_DSTORAGE);

    // -- Allocate pinned RAM staging buffers (not needed for DirectStorage) --
    if (!is_dstorage) {
        wf->ram_bufs.resize(params->n_ram_buffers);
        for (int i = 0; i < params->n_ram_buffers; i++) {
            auto & slot = wf->ram_bufs[i];
            slot.capacity = buf_size;

#ifdef WF_HAS_CUDA
            cudaError_t err = cudaMallocHost(&slot.ptr, buf_size);
            if (err != cudaSuccess) {
                WF_ERR("cudaMallocHost failed for RAM buffer %d (%.2f MB): %s",
                       i, buf_size / (1024.0 * 1024.0), cudaGetErrorString(err));
#ifdef _WIN32
                slot.ptr = _aligned_malloc(buf_size, 4096);
#else
                slot.ptr = aligned_alloc(4096, buf_size);
#endif
                if (!slot.ptr) {
                    WF_ERR("malloc also failed for RAM buffer %d", i);
                }
            } else {
                // Phase 316d (2026-05-04): track WF per-cycle pinned-host allocs.
                // This is the actual allocator that fires in the WF=on path
                // (prefill_pinned_alloc only fires when WF is off).
                fprintf(stderr, "*** WF_PIN_ALLOC: slot=%d ptr=%p +%.1fMB\n",
                        i, slot.ptr, buf_size / (1024.0 * 1024.0));
                fflush(stderr);
            }
#else
#ifdef _WIN32
            slot.ptr = _aligned_malloc(buf_size, 4096);
#else
            slot.ptr = aligned_alloc(4096, buf_size);
#endif
#endif
            if (slot.ptr) {
                WF_LOG("RAM buffer %d: %.2f MB (pinned)", i, buf_size / (1024.0 * 1024.0));
            }
        }
    }

    // -- Create CUDA copy stream --
#ifdef WF_HAS_CUDA
    const cudaError_t stream_created = cudaStreamCreateWithFlags(&wf->copy_stream,
        required_dstorage&&params->ds_async_drain?cudaStreamNonBlocking:cudaStreamDefault);
    const cudaError_t event0_created = cudaEventCreate(&wf->ds_stage_evt[0]);  // staging-drain event (LLAMA_WF_ASYNC_DRAIN)
    const cudaError_t event1_created = cudaEventCreate(&wf->ds_stage_evt[1]);
    if (required_dstorage && (stream_created != cudaSuccess || event0_created != cudaSuccess || event1_created != cudaSuccess)) {
        WF_ERR("Required DirectStorage: copy stream/event initialization failed");
        wf_destroy(wf);
        return nullptr;
    }
#endif

    // -- Initialize chunk pool (not needed for DirectStorage) --
    if (!is_dstorage) {
        size_t chunks_per_layer = (buf_size + params->io_block_size - 1) / params->io_block_size;
        int pool_size = (int)(chunks_per_layer * params->prefetch_depth * 2);
        pool_size = std::max(pool_size, 256);
        wf->chunk_pool.init(pool_size);
        WF_LOG("chunk pool: %d pre-allocated (%zu per layer × %d depth)",
               pool_size, chunks_per_layer, params->prefetch_depth);
    }

    // -- Create I/O completion infrastructure and worker threads (not needed for DirectStorage) --
    if (!is_dstorage) {
        wf->io_shutdown.store(false);
        int n_io = std::max(1, params->io_threads);

#ifdef _WIN32
        wf->iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, n_io);
        if (!wf->iocp) {
            WF_ERR("CreateIoCompletionPort failed: error %lu", GetLastError());
            delete wf;
            return nullptr;
        }

        {
            bool use_direct = (wf->effective_mode == WF_MODE_DIRECT_IO) && params->io_direct;
            wf->mt_files.resize(n_io);
            int mt_ok = 0;
            for (int i = 0; i < n_io; i++) {
                if (wf_open_file(&wf->mt_files[i], params->model_path, use_direct)) {
                    HANDLE h = CreateIoCompletionPort(wf->mt_files[i].h, wf->iocp,
                                                       (ULONG_PTR)(i + 1), 0);
                    if (h != wf->iocp) {
                        WF_ERR("failed to associate mt_file %d with IOCP: error %lu", i, GetLastError());
                    } else {
                        mt_ok++;
                    }
                }
            }
            wf->n_mt_handles = mt_ok;
            WF_LOG("IOCP: %d per-thread file handles (%d ok)", n_io, mt_ok);

            if (mt_ok > 0) {
                wf->submit_queues = new wf_context::wf_submit_queue[params->n_layers * mt_ok];
                WF_LOG("submit queues: %d × %d = %d (QD=%d per handle)",
                       params->n_layers, mt_ok, params->n_layers * mt_ok,
                       params->io_queue_depth);
            }
        }

        for (int fi = 0; fi < (int)wf->files.size(); fi++) {
            HANDLE h = CreateIoCompletionPort(wf->files[fi].h, wf->iocp,
                                               (ULONG_PTR)fi, 0);
            if (h != wf->iocp) {
                WF_ERR("failed to associate file %d with IOCP: error %lu", fi, GetLastError());
            }
        }

        for (int i = 0; i < n_io; i++) {
            wf->io_threads.emplace_back(wf_iocp_worker, wf, i);
        }
        WF_LOG("IOCP: %d completion workers, %d+%d files associated (block=%zuKiB)",
               n_io, (int)wf->mt_files.size(), (int)wf->files.size(),
               params->io_block_size / 1024);
#else
        for (int i = 0; i < n_io; i++) {
            wf->io_threads.emplace_back(wf_fallback_worker, wf, i);
        }
        WF_LOG("started %d I/O fallback workers (block=%zuKiB)",
               n_io, params->io_block_size / 1024);
#endif
    } // !is_dstorage

    // -- GDS setup --
#ifdef WF_HAS_GDS
    if (wf->effective_mode == WF_MODE_GDS) {
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            WF_ERR("GDS driver open failed, falling back to direct I/O");
            wf->effective_mode = WF_MODE_DIRECT_IO;
            wf->gds_available = false;
        } else {
            // Register GPU buffers with GDS
            for (auto & slot : wf->gpu_bufs) {
                CUfileError_t reg_status = cuFileBufRegister(slot.ptr, slot.capacity, 0);
                if (reg_status.err != CU_FILE_SUCCESS) {
                    WF_ERR("GDS buffer registration failed, falling back");
                    wf->effective_mode = WF_MODE_DIRECT_IO;
                    wf->gds_available = false;
                    break;
                }
            }
            if (wf->gds_available) {
                WF_LOG("GDS: GPU buffers registered, DMA path active");
            }
        }
    }
#endif

    // Extract the CUDA compute stream from the GPU backend.
    // ggml-cuda uses cudaStreamNonBlocking, so we MUST wait on the actual
    // compute stream (not the null stream) for D3D12 fence synchronization.
#ifdef WF_HAS_CUDA
    if (params->gpu_backend) {
        wf->cuda_compute_stream = (cudaStream_t)ggml_backend_cuda_get_stream(
            (ggml_backend_t)params->gpu_backend);
        if (wf->cuda_compute_stream) {
            WF_LOG("CUDA compute stream: %p (non-blocking, will use for DS fence sync)",
                   (void *)wf->cuda_compute_stream);
        }
    }
#endif

    // Cold-start: warm the D3D12<->CUDA DirectStorage interop now (at load) so the
    // first real streaming-prefill fetch cannot race the driver's lazy VA/semaphore
    // wiring. Placed here so cuda_compute_stream (captured just above) and copy_stream
    // are both available; covers BOTH the cold and PERSIST-INFRA-reuse init branches
    // (both funnel through wf_create's tail) plus the per-request fence/semaphore that
    // wf_dstorage_create_stateful re-creates. Gated LLAMA_WF_WARM_INTEROP (default on).
    // The DEFINITION of wf_warm_ds_interop lives inside `#ifdef WF_HAS_DSTORAGE` (opened at
    // line 1616), but this CALL was unguarded. On Windows WF_HAS_DSTORAGE is defined so both
    // exist and MSVC is happy; on macOS the definition vanishes and the call does not, so the
    // Mac build fails with 'use of undeclared identifier'. Guard the call to match the
    // definition -- byte-identical on Windows, compilable everywhere else.
#ifdef WF_HAS_DSTORAGE
    wf_warm_ds_interop(wf);
    if (required_dstorage && wf->ds_failed.load(std::memory_order_acquire)) {
        WF_ERR("Required DirectStorage: interop qualification failed during initialization");
        wf_destroy(wf);
        return nullptr;
    }
#endif

    return wf;
}

wf_context * wf_create(const wf_params * params) {
    return wf_create_impl(params, false);
}

wf_context * wf_create_required_dstorage(const wf_params * params) {
    return wf_create_impl(params, true);
}

bool wf_has_failed(const wf_context * wf) {
    return !wf || wf->ds_failed.load(std::memory_order_acquire);
}

void wf_destroy(wf_context * wf) {
    if (!wf) return;

    wf_stop(wf);

#ifdef WF_HAS_CUDA
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
#endif

#ifdef WF_HAS_DSTORAGE
    // wf_stop() joins the host threads; this additionally drains the DS queue
    // and all CUDA consumers before any interop object is destroyed below.
    wf_dstorage_quiesce(wf);
#endif

    // Close files
    for (auto & fh : wf->files) wf_close_file(&fh);
    for (auto & fh : wf->mt_files) wf_close_file(&fh);

    // Free submit queues
    delete[] wf->submit_queues;
    wf->submit_queues = nullptr;

#ifdef _WIN32
    if (wf->iocp) {
        CloseHandle(wf->iocp);
        wf->iocp = NULL;
    }
#endif

    // Free GPU buffers
#ifdef WF_HAS_DSTORAGE
    if (wf->ds_initialized || wf->required_dstorage) {
        // DirectStorage cleanup
        if (wf->ds_queue) wf->ds_queue->Release();
        if (wf->ds_ext_sem) cudaDestroyExternalSemaphore(wf->ds_ext_sem);
        if (wf->ds_fence_handle) CloseHandle(wf->ds_fence_handle);
        if (wf->ds_fence) wf->ds_fence->Release();

        // Release staging resource
        auto & stg = wf->ds_stage;
        // Phase 316e (2026-05-04): log the staging release to confirm/refute
        // ds_stage as the leak source. If the matching DS_STAGE_FREE never
        // logs, this 3.5 GB stays mapped per cycle.
        fprintf(stderr, "*** DS_STAGE_FREE: cuda_ptr=%p ext_mem=%p d3d=%p shared=%p\n",
                stg.cuda_ptr, (void*)stg.ext_mem, (void*)stg.d3d_resource, stg.shared_handle);
        fflush(stderr);
        // Final CUDA-side safety check after wf_dstorage_quiesce() has already
        // drained the DirectStorage queue and CUDA consumers. Without
        // this sync, cudaDestroyExternalMemory only unmaps from the CUDA
        // address space — the D3D12 resource's GPU allocation persists if
        // any unsignalled fence still references it. A CUDA device sync alone
        // is not a substitute for the D3D12 tail-fence wait.
#ifdef WF_HAS_CUDA
        if (stg.cuda_ptr || stg.ext_mem) {
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                fprintf(stderr, "*** DS_STAGE_FREE: cudaDeviceSynchronize: %s\n",
                        cudaGetErrorString(sync_err));
            }
        }
        // Phase 2.3b (2026-05-12) — VRAM-leak fix.
        // Per CUDA docs: cudaExternalMemoryGetMappedBuffer's pointer MUST be
        // freed via cudaFree() before the parent cudaExternalMemory_t is
        // destroyed. Otherwise the GPU mapping stays alive — each request
        // re-imports + maps fresh, accumulating ~staging_size MB per cycle.
        // Pre-fix Phase 2.3b bisect (commit 221b12df) showed perfect 1:1
        // scaling with LLAMA_DS_STAGING_MB.
        // R3 persist: when the staging is the cached persistent buffer, do NOT free its CUDA
        // mapping (cuda_ptr/ext_mem) — it lives in g_wf_persist for reuse. The COM d3d_resource
        // is refcounted (g_wf_persist holds its own AddRef), so Release here just drops THIS wf's ref.
        if (stg.persistent && g_wf_persist.active_owner == wf) {
            g_wf_persist.active_owner = nullptr;
        }
        if (!stg.persistent && stg.cuda_ptr) {
            cudaError_t free_err = cudaFree(stg.cuda_ptr);
            if (free_err != cudaSuccess) {
                fprintf(stderr, "*** DS_STAGE_FREE: cudaFree(cuda_ptr): %s\n",
                        cudaGetErrorString(free_err));
            }
            stg.cuda_ptr = nullptr;
        }
#endif
        if (!stg.persistent && stg.ext_mem) cudaDestroyExternalMemory(stg.ext_mem);
        if (!stg.persistent && stg.shared_handle) CloseHandle(stg.shared_handle);
        if (stg.d3d_resource) stg.d3d_resource->Release();
        // The existing imported-heap helper owns direct pointers. Wrappers and
        // generic slots are non-owning; release only after the queue is quiescent.
        for(auto &bank:wf->ds_compute_banks)wf_free_dual_use_heap(bank.ptr,bank.external,bank.heap,bank.quarters);
        wf->ds_compute_banks.clear();
        // Free CUDA compute buffers
        for (auto & slot : wf->gpu_bufs) {
            if (slot.ptr && slot.owns_ptr) {
                cudaFree(slot.ptr);
            }
            slot.ptr = nullptr;
            slot.owns_ptr = false;
        }
        for (auto * f : wf->ds_files) if (f) f->Release();
        if (wf->ds_factory) wf->ds_factory->Release();
        if (wf->d3d_device) wf->d3d_device->Release();
        if (wf->dxgi_factory) wf->dxgi_factory->Release();
        // R3 persist: do NOT tear down COM while the persisted device/factory are held — that
        // invalidates them and crashes the next (reuse) request. COM stays initialized for the process.
        if (wf->ds_com_inited && !wf->ds_stage.persistent) CoUninitialize();

        // Still destroy CUDA events on the gpu_buf slots
        for (auto & slot : wf->gpu_bufs) {
            if (slot.upload_done)  cudaEventDestroy(slot.upload_done);
            if (slot.compute_done) cudaEventDestroy(slot.compute_done);
            slot.upload_done  = nullptr;
            slot.compute_done = nullptr;
            slot.capacity     = 0;
            slot.layer_idx    = -1;
            slot.in_use.store(false);
        }
    } else
#endif
    {
        for (auto & slot : wf->gpu_bufs) {
#ifdef WF_HAS_CUDA
            if (slot.ptr && slot.owns_ptr) cudaFree(slot.ptr);
            slot.ptr = nullptr;
            slot.owns_ptr = false;
            if (slot.upload_done)  cudaEventDestroy(slot.upload_done);
            if (slot.compute_done) cudaEventDestroy(slot.compute_done);
            slot.upload_done  = nullptr;
            slot.compute_done = nullptr;
            slot.capacity     = 0;
            slot.layer_idx    = -1;
            slot.in_use.store(false);
#else
#ifdef _WIN32
            if (slot.ptr && slot.owns_ptr) _aligned_free(slot.ptr);
#else
            if (slot.ptr && slot.owns_ptr) free(slot.ptr);
#endif
            slot.ptr = nullptr;
            slot.owns_ptr = false;
#endif
        }
    }

    // Free RAM buffers
    {
        int slot_idx = 0;
        for (auto & slot : wf->ram_bufs) {
#ifdef WF_HAS_CUDA
            if (slot.ptr) {
                // Phase 316d (2026-05-04): track WF per-cycle pinned-host frees.
                fprintf(stderr, "*** WF_PIN_FREE: slot=%d ptr=%p -%.1fMB\n",
                        slot_idx, slot.ptr, slot.capacity / (1024.0 * 1024.0));
                fflush(stderr);
                cudaFreeHost(slot.ptr);
            }
#else
#ifdef _WIN32
            if (slot.ptr) _aligned_free(slot.ptr);
#else
            free(slot.ptr);
#endif
#endif
            slot_idx++;
        }
    }

#ifdef WF_HAS_CUDA
    if (wf->copy_stream && wf_cache_enabled(wf)) {
        cudaStreamSynchronize(wf->copy_stream);
    }
    if (wf->expert_vram_cache.arena) {
        cudaFree(wf->expert_vram_cache.arena);
        wf->expert_vram_cache.arena = nullptr;
    }
    if (wf->expert_ram_cache.arena) {
        cudaFreeHost(wf->expert_ram_cache.arena);
        wf->expert_ram_cache.arena = nullptr;
    }
    if (wf->copy_stream) cudaStreamDestroy(wf->copy_stream);
    if (wf->ds_stage_evt[0]) cudaEventDestroy(wf->ds_stage_evt[0]);
    if (wf->ds_stage_evt[1]) cudaEventDestroy(wf->ds_stage_evt[1]);
#endif

#ifdef WF_HAS_GDS
    if (wf->gds_available) {
        for (auto & slot : wf->gpu_bufs) {
            if (slot.ptr) cuFileBufDeregister(slot.ptr);
        }
        cuFileDriverClose();
    }
#endif

    delete wf;
}

void wf_register_layer(wf_context * wf, int layer_idx,
                        const wf_weight_span * spans, int n_spans,
                        size_t total_bytes) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;

    if (wf->required_dstorage) {
        const int span_limit = wf->params.moe_mode == WF_MOE_FETCH_ACTIVE ? 2048 : 128;
        bool valid = !wf->pipeline_running.load() && n_spans > 0 && n_spans <= span_limit && spans &&
            total_bytes > 0 && total_bytes <= wf->params.max_layer_bytes;
        size_t previous = 0;
        for (int i = 0; valid && i < n_spans; ++i) {
            const auto & s = spans[i];
            const size_t offset = s.alloc_offset ? s.alloc_offset : previous;
            const size_t allocation = s.alloc_size ? s.alloc_size : s.nbytes;
            valid = s.tensor_name && s.tensor_name[0] && !s.mmap_addr && s.nbytes > 0 &&
                s.file_idx >= 0 && size_t(s.file_idx) < wf->files.size() &&
                allocation >= s.nbytes && offset >= previous && offset <= total_bytes && allocation <= total_bytes - offset;
            if (valid) {
                const uint64_t file_bytes = wf->files[s.file_idx].file_size;
                valid = s.file_offset <= file_bytes && s.nbytes <= file_bytes - s.file_offset;
                previous = offset + allocation;
            }
        }
        if (!valid) {
            WF_ERR("Required DirectStorage: invalid physical layer registration (no mmap/host fallback permitted)");
            wf->ds_failed.store(true, std::memory_order_release);
            return;
        }
    }

    auto & layer = wf->layers[layer_idx];
    layer.total_bytes = total_bytes;
    layer.spans.resize(n_spans);

    size_t offset = 0;
    for (int i = 0; i < n_spans; i++) {
        auto & dst = layer.spans[i];
        auto & src = spans[i];
        if (src.tensor_name) {
            strncpy(dst.tensor_name, src.tensor_name, sizeof(dst.tensor_name) - 1);
            dst.tensor_name[sizeof(dst.tensor_name) - 1] = '\0';
        } else {
            dst.tensor_name[0] = '\0';
        }
        dst.file_offset = src.file_offset;
        dst.nbytes      = src.nbytes;
        dst.mmap_addr   = src.mmap_addr;
        dst.file_idx    = src.file_idx;
        dst.is_expert   = src.is_expert;
        dst.expert_idx  = src.expert_idx;
        // Use alloc_offset from ggml graph allocator when available.
        // This ensures the DMA destination matches where the adapter remap
        // will point tensors during compute (prevents data/pointer mismatch
        // for small tensors at high offsets in large MoE layers).
        dst.gpu_offset = (src.alloc_offset > 0) ? src.alloc_offset : offset;
        offset = dst.gpu_offset + ((src.alloc_size > 0) ? src.alloc_size : src.nbytes);
    }
    // TASK-A probe (LLAMA_WF_SPAN_DUMP=<il>): dump this layer's span table + the padded extent
    // vs registered total_bytes. The DS wave capacity/copy math uses total_bytes while request
    // placement uses padded gpu_offsets — any overhang is silently dropped/under-copied.
    static const int span_dump_il = []() {
        const char * e = std::getenv("LLAMA_WF_SPAN_DUMP");
        return (e && e[0] != '\0') ? atoi(e) : -1;
    }();
    if (layer_idx == span_dump_il) {
        fprintf(stderr, "[wf] SPAN_DUMP layer %d: n_spans=%d total_bytes=%zu padded_extent=%zu overhang=%lld\n",
                layer_idx, n_spans, total_bytes, offset, (long long) offset - (long long) total_bytes);
        for (int i = 0; i < n_spans; i++) {
            const auto & s = layer.spans[i];
            fprintf(stderr, "[wf]   span[%03d] gpu_off=%12zu nbytes=%11zu file=%d file_off=%12llu mmap=%p '%s'\n",
                    i, s.gpu_offset, s.nbytes, s.file_idx, (unsigned long long) s.file_offset,
                    s.mmap_addr, s.tensor_name);
        }
        fflush(stderr);
    }
}

void wf_start(wf_context * wf) {
    if (!wf || wf->pipeline_running.load()) return;
    if (wf->required_dstorage && wf->ds_failed.load(std::memory_order_acquire)) return;

    wf_cache_init(wf);
    wf_expert_prefetch_init(wf);
    if (wf_cache_enabled(wf) && wf->expert_cache_passes == 0) wf->expert_cache_passes = 1;

    wf->pipeline_running.store(true);
    // FORWARD pass: always reset direction/start to ascending. Critical for training, where each step does
    // forward (wf_start) then backward (wf_start_reverse, dir=-1): without this reset the NEXT step's forward
    // would inherit dir=-1 and the multi-buffer recycle would advance the wrong way (forward stuck@layer 3,
    // dma_next=-1). (Single-buffer masked this — it bypasses the DMA-thread recycle via sync DMA.)
    wf->dir = +1;
    wf->start_layer = 0;
    wf->pipeline_next_layer.store(0);

    // Reset layer stages
    for (auto & layer : wf->layers) {
        layer.stage.store(WF_STAGE_IDLE);
        layer.gpu_buf_idx = -1;
        layer.ram_buf_idx = -1;
        layer.cache_inserted = false;
        layer.expert_filter_set = false;
        layer.expert_prefetch_ids.clear();
        layer.expert_prefetch_loaded_spans.clear();
        layer.expert_prefetch_planned = false;
        layer.expert_prefetch_loaded = false;
    }

    // Reset buffer pool
    for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
    for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }

    // Reset atomic stats (before DS startup so startup bytes are counted)
    wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
    wf->stat_io_ops.store(0, std::memory_order_relaxed);

#ifdef WF_HAS_DSTORAGE
    // Pre-build wave-based DS request arrays on first start
    if (wf->ds_initialized && wf->ds_waves.empty()) {
        wf_dstorage_prebuild_requests(wf);
    }

    if(wf->params.ds_gpu_ring_slots&&!wf_dstorage_ring_prepare(wf)){wf->pipeline_running.store(false);return;}

    // Initialize per-buffer DMA pending slots
    wf->dma_buf_pending.resize(wf->gpu_bufs.size());
    for (auto & slot : wf->dma_buf_pending) { slot.pending_layer.store(-1); }

    // GPU_BUFS > 1: schedule layer 0 for the DMA thread (overlapped pipeline).
    // GPU_BUFS == 1: wf_wait_layer_gpu does sync DMA — don't schedule here
    // to avoid racing with the SYNC_DMA path on the same staging buffer.
    if (wf->ds_initialized && !wf->ds_waves.empty() && wf->gpu_bufs.size() > 1) {
        wf->dma_buf_pending[0].pending_layer.store(0);
        // For ping-pong: also schedule layer 1 on buffer 1
        if (wf->gpu_bufs.size() > 1 && wf->layers.size() > 1) {
            wf->dma_buf_pending[1].pending_layer.store(1);
            wf->dma_next_layer.store(2);
        } else {
            wf->dma_next_layer.store(1);
        }
        wf->dma_timing_log.clear();
        WF_LOG("startup: layers 0,1 scheduled for DMA thread (ping-pong pipeline)");
    }
#endif

    // Record wall-clock start
    wf->pass_start = std::chrono::high_resolution_clock::now();

#ifdef WF_HAS_DSTORAGE
    // Launch DMA control thread — handles all DS submissions via blocking wave transfers.
    // Only for GPU_BUFS > 1 (overlapped mode). For GPU_BUFS == 1, wf_wait_layer_gpu
    // does synchronous DMA — launching the DMA thread would race on the staging buffer.
    if (wf->ds_initialized && !wf->ds_waves.empty() && wf->gpu_bufs.size() > 1) {
        wf->dma_thread = std::thread(wf_dma_thread_fn, wf);
    }
#endif

    wf->pipeline_thread = std::thread(wf_pipeline_thread_fn, wf);
    WF_LOG("pipeline started (prefetch depth=%d, %d GPU bufs, %d RAM bufs, pool=%d chunks)",
           wf->params.prefetch_depth, wf->params.n_gpu_buffers, wf->params.n_ram_buffers,
           (int)wf->chunk_pool.storage.size());
}

// reverse-ds: start a BACKWARD pass that streams layers N-1..0 (training backward). Mirrors wf_start with reverse
// init (start_layer=N-1, dir=-1, ping-pong N-1/N-2, dma_next=N-3). NEW symbol — wf_start (forward/inference) is
// untouched. NOTE: requires the dir-aware DMA + pipeline threads (separate edit) to actually advance descending;
// until those land this sets up correctly but the threads still step +1. See TRAINING_ROADMAP.md.
void wf_start_reverse(wf_context * wf) {
    if(wf&&wf->params.ds_gpu_ring_slots){wf->ds_failed.store(true);WF_ERR("GPU ring is one-shot forward-only");return;}
    if (!wf || wf->pipeline_running.load()) return;
    wf_cache_init(wf);
    const int N = (int)wf->layers.size();

    wf->dir = -1;
    wf->start_layer = N - 1;
    wf->pipeline_running.store(true);
    wf->pipeline_next_layer.store(N - 1);

    for (auto & layer : wf->layers) {
        layer.stage.store(WF_STAGE_IDLE);
        layer.gpu_buf_idx = -1;
        layer.ram_buf_idx = -1;
        layer.cache_inserted = false;
    }
    for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
    for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }
    wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
    wf->stat_io_ops.store(0, std::memory_order_relaxed);

#ifdef WF_HAS_DSTORAGE
    if (wf->ds_initialized && wf->ds_waves.empty()) {
        wf_dstorage_prebuild_requests(wf);
    }
    wf->dma_buf_pending.resize(wf->gpu_bufs.size());
    for (auto & slot : wf->dma_buf_pending) { slot.pending_layer.store(-1); }
    if (wf->ds_initialized && !wf->ds_waves.empty() && wf->gpu_bufs.size() > 1) {
        wf->dma_buf_pending[0].pending_layer.store(N - 1);                 // ping-pong, descending
        if (wf->gpu_bufs.size() > 1 && N > 1) {
            wf->dma_buf_pending[1].pending_layer.store(N - 2);
            wf->dma_next_layer.store(N - 3);
        } else {
            wf->dma_next_layer.store(N - 2);
        }
        wf->dma_timing_log.clear();
        WF_LOG("startup REVERSE: layers %d,%d scheduled for DMA thread (descending pipeline)", N - 1, N - 2);
    }
#endif

    wf->pass_start = std::chrono::high_resolution_clock::now();
#ifdef WF_HAS_DSTORAGE
    if (wf->ds_initialized && !wf->ds_waves.empty() && wf->gpu_bufs.size() > 1) {
        wf->dma_thread = std::thread(wf_dma_thread_fn, wf);
    }
#endif
    wf->pipeline_thread = std::thread(wf_pipeline_thread_fn, wf);
    WF_LOG("pipeline started REVERSE (start_layer=%d, dir=-1)", N - 1);
}

void wf_request_layer(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;

    // Advance the pipeline to cover this layer (direction-aware: forward advances up, reverse advances down)
    int current_next = wf->pipeline_next_layer.load();
    bool ahead = (wf->dir > 0) ? (layer_idx >= current_next) : (layer_idx <= current_next);
    if (ahead) {
        wf->pipeline_next_layer.store(layer_idx + wf->dir);
        wf->pipeline_cv.notify_one();
    }
}

void * wf_wait_layer_gpu(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return nullptr;

#ifdef WF_HAS_DSTORAGE
    if (wf->ds_failed.load(std::memory_order_acquire)) return nullptr;
#endif

    auto & layer = wf->layers[layer_idx];

    // Skip empty sub-layers (e.g., dense layer's MoE slot)
    if (layer.total_bytes == 0 && layer.spans.empty()) {
        layer.stage.store(WF_STAGE_DONE);
        // Return first buffer pointer — nothing will be read from it
        return wf->gpu_bufs.empty() ? nullptr : wf->gpu_bufs[0].ptr;
    }

    // ---- Ubatch boundary detection ----
    // When llama_decode() splits a batch into ubatches, each ubatch runs the
    // full forward pass through all layers.  After ubatch N finishes, all layers
    // are at WF_STAGE_DONE.  When ubatch N+1 starts, pre_layer_cb(0) calls us
    // with layer_idx==0 still at DONE.  We must reset the pipeline inline so
    // the DMA/pipeline threads can service the new pass.
    //
    // Critical ordering:
    //   1. Drain D3D12 fence (all prior DMA truly complete)
    //   2. Reset DMA thread pending slots (prevent stale submissions)
    //   3. Reset all layer/buffer state
    //   4. Set ds_layer_fence_val BEFORE setting stage to NVME_READ
    //      (pipeline thread checks fence when it sees NVME_READ)
    //   5. Submit startup DS layers
    //   6. Set dma_next_layer for the DMA thread to pick up remaining layers
    if (layer_idx == 0 && layer.stage.load() == WF_STAGE_DONE && !wf->layer_major_mode) {
        WF_ERR("ubatch boundary: resetting pipeline for new pass");

        if (wf_cache_enabled(wf)) {
            ++wf->expert_cache_passes;
            if (wf->expert_cache_passes % 100 == 0) {
                std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
                WF_ERR("expert cache @ %llu tokens: VRAM hits=%llu, RAM hits=%llu, misses=%llu, saved=%.2f GB",
                       (unsigned long long)wf->expert_cache_passes,
                       (unsigned long long)wf->stats.cache_vram_hits,
                       (unsigned long long)wf->stats.cache_ram_hits,
                       (unsigned long long)wf->stats.cache_misses,
                       wf->stats.cache_bytes_saved / (1024.0 * 1024.0 * 1024.0));
            }
        }
        if (wf->expert_prefetch_enabled) {
            ++wf->expert_prefetch_passes;
            if (wf->expert_prefetch_passes % 100 == 0) {
                std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
                const uint64_t predicted = wf->stats.expert_prefetch_predicted - wf->expert_prefetch_log_predicted;
                const uint64_t hits = wf->stats.expert_prefetch_hits - wf->expert_prefetch_log_hits;
                const uint64_t misses = wf->stats.expert_prefetch_misses - wf->expert_prefetch_log_misses;
                const size_t waste = wf->stats.expert_prefetch_wasted_bytes - wf->expert_prefetch_log_waste;
                WF_ERR("expert prefetch @ %llu tokens (last 100): predicted=%llu, hits=%llu, misses=%llu, waste=%.2f MB",
                       (unsigned long long) wf->expert_prefetch_passes,
                       (unsigned long long) predicted, (unsigned long long) hits,
                       (unsigned long long) misses, waste / (1024.0 * 1024.0));
                wf->expert_prefetch_log_predicted = wf->stats.expert_prefetch_predicted;
                wf->expert_prefetch_log_hits = wf->stats.expert_prefetch_hits;
                wf->expert_prefetch_log_misses = wf->stats.expert_prefetch_misses;
                wf->expert_prefetch_log_waste = wf->stats.expert_prefetch_wasted_bytes;
            }
        }

#ifdef WF_HAS_DSTORAGE
        // Step 1: Drain D3D12 fence — ensure ALL prior DMA is truly complete
        if (wf->ds_initialized && wf->ds_fence) {
            uint64_t current = wf->ds_fence_counter.load();
            if (current > 0 && !wf_ds_fence_wait(wf, current)) return nullptr;
        }

        // Step 2: Reset DMA thread pending slots (DMA thread is idle at this point
        // since dma_next_layer >= n_layers, but clear to prevent stale pickups)
        for (auto & slot : wf->dma_buf_pending) {
            slot.pending_layer.store(-1);
        }
#endif

        // Step 2.5: Drain the CUDA compute stream — ubatch N's kernels may still
        // be in-flight reading from GPU buffers we're about to recycle.  Without
        // this, the new ubatch's DMA overwrites buffers while old kernels read them.
#ifdef WF_HAS_CUDA
        if (wf->cuda_compute_stream) {
            const cudaError_t synced = cudaStreamSynchronize(wf->cuda_compute_stream);
            if (wf->required_dstorage && synced != cudaSuccess) {
                wf->ds_failed.store(true,std::memory_order_release);return nullptr;
            }
        }
#endif

        // Step 3: Reset all layer and buffer state
        for (auto & l : wf->layers) {
            l.stage.store(WF_STAGE_IDLE);
            l.gpu_buf_idx = -1;
            l.ram_buf_idx = -1;
            l.expert_filter_set = false;
            l.cache_inserted = false;
            l.expert_prefetch_ids.clear();
            l.expert_prefetch_loaded_spans.clear();
            l.expert_prefetch_planned = false;
            l.expert_prefetch_loaded = false;
        }
        for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
        for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }

#ifdef WF_HAS_DSTORAGE
        // Clear fence values to prevent stale completion checks
        for (auto & fv : wf->ds_layer_fence_val) { fv = 0; }

        if (wf->ds_initialized && !wf->ds_waves.empty()) {
            // Schedule the pass's first layer for the DMA thread (start_layer/dir: forward 0/+1, reverse N-1/-1)
            wf->dma_buf_pending[0].pending_layer.store(wf->start_layer);
            wf->dma_next_layer.store(wf->start_layer + wf->dir);
        }
#endif
        // Reset atomic stats for new pass
        wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
        wf->stat_io_ops.store(0, std::memory_order_relaxed);
        wf->pipeline_next_layer.store(wf->start_layer);

        WF_ERR("ubatch boundary reset complete");
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // If layer isn't loaded yet and we have DS waves, do synchronous DMA
    // CRITICAL: wait for ALL GPU work first. The previous layer's CUDA
    // kernels are launched asynchronously — they may still be executing
    // when we overwrite the buffer with the new layer's weights.
#ifdef WF_HAS_DSTORAGE
    // Only use sync path for GPU_BUFS=1. With GPU_BUFS>1 the DMA thread handles
    // all loads — entering the sync path here would race with the DMA thread's
    // use of the shared DS queue and fence.
    if (layer.stage.load() == WF_STAGE_IDLE && wf->ds_initialized && !wf->ds_waves.empty()
        && wf->gpu_bufs.size() <= 1) {
        // One compute destination aliases every layer: establish exclusive ownership
        // before DS/D2D writes any byte of slot 0.
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            WF_ERR("single-buffer compute drain failed before layer %d refill: %s",
                   layer_idx, cudaGetErrorString(err));
            return nullptr;
        }
        // Wait for ALL DMA in-flight to complete before using sync path.
        // Without this, the DMA thread may be writing to the same buffer
        // we're about to use, causing data corruption.
        while (wf->dma_in_flight_buf.load() >= 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        int bi = 0;

        // [2026-07-28 M3 wf=1 DIAG] LLAMA_WF_M3_DIAG=1: localize the single-buffer garble.
        static const bool wf1_diag = []() { const char * e = std::getenv("LLAMA_WF_M3_DIAG"); return e && e[0] && e[0] != '0'; }();
        static std::vector<unsigned long long> g_wf1_hash;   // post-fill hash per layer (sync path is single-threaded)
        static std::vector<char>               g_wf1_have;
        static int                             g_wf1_last_layer = -1;  // last layer this single buffer held (metadata layer_idx gets reset)
        auto wf1_hash_buf = [](void * buf, size_t tb) -> unsigned long long {
            const size_t offs[3] = { 0, tb / 2, tb >= 256 ? tb - 256 : 0 };
            unsigned long long h = 1469598103934665603ULL;
            for (int k = 0; k < 3; k++) {
                unsigned char b[256] = {0};
                size_t n = (tb > offs[k] + 256) ? 256 : (tb > offs[k] ? tb - offs[k] : 0);
                if (n) { cudaMemcpy(b, (char*)buf + offs[k], n, cudaMemcpyDeviceToHost);
                         for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } }
            }
            return h;
        };
        if (wf1_diag && g_wf1_hash.size() < wf->layers.size()) {
            g_wf1_hash.assign(wf->layers.size(), 0); g_wf1_have.assign(wf->layers.size(), 0);
        }
        // ALIAS DETECT: the buffer still holds the PREVIOUS layer's weights. Re-hash and compare
        // to that layer's post-fill hash: CHANGED => something overwrote the weights DURING compute
        // (an aliased KV/activation/output tensor) = wf=1 downstream corruption; MATCH => weights survived.
        if (wf1_diag) {
            int prev = g_wf1_last_layer;
            if (prev >= 0 && prev != layer_idx && prev < (int)g_wf1_have.size() && g_wf1_have[prev]) {
                cudaDeviceSynchronize();
                unsigned long long now = wf1_hash_buf(wf->gpu_bufs[bi].ptr, (size_t)wf->layers[prev].total_bytes);
                fprintf(stderr, "*** WF_M3_ALIAS prev_layer=%d prefill_now=%016llx postfill=%016llx %s\n",
                        prev, now, g_wf1_hash[prev], now == g_wf1_hash[prev] ? "MATCH" : "CHANGED-OVERWRITTEN");
                fflush(stderr);
            }
        }

        wf->gpu_bufs[bi].in_use = true;
        wf->gpu_bufs[bi].layer_idx = layer_idx;
        layer.gpu_buf_idx = bi;
        void * compute_buf = wf->gpu_bufs[bi].ptr;
        int n_waves = (int)wf->ds_waves[layer_idx].size();
        (void)n_waves;

        const char * wf1_path = "?";
        bool transfer_ok = true;

        if (wf->ds_direct_mode) {
            wf1_path = "ds_direct";
            // R3 DS-direct edit 13: DS-fill the WHOLE layer DIRECTLY into the prefetch buffer's placed
            // quarters (no staging tier), then ONE prefetch->compute D2D. compute_buf is the permanent
            // ggml-wrapped buffer that every layer's tensor->data aliases, so compute reads the correct
            // weights once the D2D lands. transfer_layer fence-waits each wave (DS write is CPU-confirmed
            // complete on return); the synchronous D2D + D2H barrier match the legacy relay's proven
            // cross-stream ordering so the compute stream never reads compute_buf before the D2D finishes.
            transfer_ok = wf_dstorage_transfer_layer(
                    wf, layer_idx, bi, /*dest_buf*/wf->prefetch_buf);
            if (!transfer_ok) {
                wf->gpu_bufs[bi].in_use = false;
                wf->gpu_bufs[bi].layer_idx = -1;
                layer.gpu_buf_idx = -1;
                return nullptr;
            }
            static const bool dsd_diag = [](){ const char*e=std::getenv("LLAMA_WF_DS_DIRECT_DIAG"); return e&&e[0]&&e[0]!='0'; }();
            if (dsd_diag && layer_idx <= 1) {
                const uint64_t QSZ = wf->ds_quarter_size;
                uint64_t req0off = (!wf->ds_waves[layer_idx].empty() && !wf->ds_waves[layer_idx][0].reqs.empty())
                                   ? wf->ds_waves[layer_idx][0].reqs[0].Destination.Buffer.Offset : (uint64_t)-1;
                fprintf(stderr, "*** DSD_DIAG il=%d total=%zu QSZ=%.2fGB prefetch_buf=%p compute_buf=%p nquarters=%zu req0.destOff=%llu\n",
                    layer_idx, (size_t)layer.total_bytes, QSZ/1e9, wf->prefetch_buf, compute_buf,
                    wf->prefetch_quarters.size(), (unsigned long long)req0off);
                // Scan prefetch at each quarter boundary + a bit into each.
                const uint64_t offs[6] = { 0, 1ull<<20, QSZ, QSZ+(1ull<<20), 2*QSZ, 2*QSZ+(1ull<<20) };
                char buf[256]; int n=0; n += snprintf(buf+n,sizeof(buf)-n,"*** DSD_DIAG il=%d scan",layer_idx);
                for (int k=0;k<6;k++){ if(offs[k]+4>(uint64_t)wf->prefetch_buf_size){n+=snprintf(buf+n,sizeof(buf)-n," %lluMB=oob",(unsigned long long)(offs[k]>>20));continue;}
                    unsigned char b[4]={0}; cudaMemcpy(b,(char*)wf->prefetch_buf+offs[k],4,cudaMemcpyDeviceToHost);
                    bool nz=b[0]||b[1]||b[2]||b[3]; n += snprintf(buf+n,sizeof(buf)-n," %lluMB=%c(%02x)", (unsigned long long)(offs[k]>>20), nz?'Y':'.', b[0]); }
                fprintf(stderr, "%s\n", buf); fflush(stderr);
#ifdef WF_HAS_DSTORAGE
                HANDLE error_event = wf->ds_queue ? wf->ds_queue->GetErrorEvent() : NULL;
                if (error_event && WaitForSingleObject(error_event, 0) == WAIT_OBJECT_0) {
                    DSTORAGE_ERROR_RECORD er={}; wf->ds_queue->RetrieveErrorRecord(&er);
                    fprintf(stderr, "*** DSD_DIAG il=%d DS_ERR=0x%08lx bytes_read=%llu\n",
                        layer_idx, (unsigned long)er.FirstFailure.HResult, (unsigned long long)wf->stat_bytes_read_nvme.load()); fflush(stderr); }
#endif
            }
            // The DS-direct prefetch allocation was written by D3D12. Enqueue
            // the imported-fence wait on this thread's consuming stream before
            // the prefetch-to-compute CUDA copy.
            if (layer_idx < (int) wf->ds_layer_fence_val.size()) {
                if (!wf_cuda_wait_ds_fence(
                        wf, wf->ds_layer_fence_val[layer_idx], (cudaStream_t) 0)) {
                    wf->gpu_bufs[bi].in_use = false;
                    wf->gpu_bufs[bi].layer_idx = -1;
                    layer.gpu_buf_idx = -1;
                    return nullptr;
                }
            }
            cudaMemcpy(compute_buf, wf->prefetch_buf, layer.total_bytes, cudaMemcpyDeviceToDevice);
            { uint8_t barrier[16] = {}; cudaMemcpy(barrier, compute_buf, 16, cudaMemcpyDeviceToHost); (void)barrier; }
        } else if (wf->prefetch_buf && wf->prefetch_buf_size > 0 && layer.total_bytes > wf->prefetch_buf_size
                   // [2026-07-28 M3 wf=1 GARBLE FIX] This relay (wf_dstorage_transfer_range) is a SEPARATE,
                   // simpler fetch that OMITS mmap-fallback span handling (spans with file_offset==0 &&
                   // mmap_addr!=nullptr, excluded from DS and delivered by H2D in wf_dstorage_transfer_layer),
                   // plus the expert cache-hit reapply, and runs default-stream-only. It leaves every
                   // mmap-fallback span as the cudaMemset(0) zero -> missing/zeroed weight rows -> corrupt
                   // K/V across all layers -> empty/control-char garbage. M3 UNIQUELY hits it: n_sublayers=1
                   // so its ~7.2GB layer exceeds prefetch_buf_size and takes this relay; DSV4/GLM/Kimi split
                   // experts into sub-prefetch-sized sublayers, stay under prefetch_buf_size, and take the
                   // proven wf_dstorage_transfer_layer path (the else below) -> that is exactly why wf=1
                   // works for them but garbles for M3. DEFAULT OFF: fall through to transfer_layer (which
                   // handles >4GB via internal per-wave staging + mmap + cache spans; byte-identical to the
                   // wf=2 DMA path). LLAMA_WF_RELAY_LARGE_LAYER=1 re-enables the faster-but-mmap-broken relay
                   // for A/B. Root-cause: sol-xhigh a0e284dc.
                   && !wf->required_dstorage
                   && ([]{ static const bool relay = []{ const char * v = std::getenv("LLAMA_WF_RELAY_LARGE_LAYER"); return v && v[0] == '1'; }(); return relay; }())) {
            wf1_path = "relay";
            size_t half = std::min(wf->prefetch_buf_size, layer.total_bytes);

            // Zero the compute buffer — needed for MATRIX_ROW_PADDING.
            // The relay pattern bypasses wf_dstorage_transfer_layer's zero.
            cudaMemset(compute_buf, 0, layer.total_bytes);

            // Check if async prefetch has the first half ready
            if (wf->prefetch_ready.load() && wf->prefetch_layer_idx == layer_idx) {
                // Wait for async prefetch to complete (should already be done)
                if (wf->prefetch_future.valid()) wf->prefetch_future.wait();

                // D2D: prefetch_buf (already loaded) → compute_buf[0..half]
                cudaMemcpy(compute_buf, wf->prefetch_buf, half, cudaMemcpyDeviceToDevice);
            } else {
                // No prefetch — load first half synchronously
                if (wf->prefetch_future.valid()) wf->prefetch_future.wait();
                transfer_ok = wf_dstorage_transfer_range(
                        wf, layer_idx, wf->prefetch_buf, 0, half);
                if (transfer_ok) {
                    cudaMemcpy(compute_buf, wf->prefetch_buf, half, cudaMemcpyDeviceToDevice);
                }
            }

            // Second half: always synchronous (relay through prefetch_buf)
            size_t rest = layer.total_bytes - half;
            if (transfer_ok && rest > 0 && rest <= wf->prefetch_buf_size) {
                transfer_ok = wf_dstorage_transfer_range(
                        wf, layer_idx, wf->prefetch_buf, half, layer.total_bytes);
                if (transfer_ok) {
                    cudaMemcpy((char *)compute_buf + half, wf->prefetch_buf, rest, cudaMemcpyDeviceToDevice);
                }
            } else if (rest > 0) {
                transfer_ok = wf_dstorage_transfer_layer(wf, layer_idx, bi);
            }
            wf->prefetch_ready.store(false);
            wf->prefetch_layer_idx = -1;

            // Cross-stream barrier
            {
                uint8_t barrier[16] = {};
                cudaMemcpy(barrier, compute_buf, 16, cudaMemcpyDeviceToHost);
                (void)barrier;
            }
        } else {
            // Direct transfer (small layers or no prefetch)
            wf1_path = "transfer_layer_else";
            transfer_ok = wf_dstorage_transfer_layer(wf, layer_idx, bi);
        }

        if (!transfer_ok || wf->ds_failed.load(std::memory_order_acquire)) {
            // Never make a failed refill visible. In particular, the D3D12
            // device-removed fence sentinel must not turn the previous layer's
            // bytes into a successful wait_layer result.
            wf->gpu_bufs[bi].in_use = false;
            wf->gpu_bufs[bi].layer_idx = -1;
            layer.gpu_buf_idx = -1;
            return nullptr;
        }

        // Store this layer's post-fill hash (for EVERY layer) so the next-layer ALIAS DETECT can compare.
        if (wf1_diag && layer_idx < (int)g_wf1_hash.size()) {
            cudaDeviceSynchronize();
            g_wf1_hash[layer_idx] = wf1_hash_buf(compute_buf, (size_t)layer.total_bytes);
            g_wf1_have[layer_idx] = 1;
            g_wf1_last_layer = layer_idx;
        }

        if (wf1_diag && (layer_idx <= 2 || layer_idx == 30 || layer_idx == (int)wf->layers.size() - 1)) {
            cudaDeviceSynchronize();
            const size_t tb  = (size_t)layer.total_bytes;
            const size_t cap = wf->gpu_bufs[bi].capacity;
            const bool overflow = tb > cap;
            size_t mmap_spans = (layer_idx < (int)wf->ds_mmap_spans.size()) ? wf->ds_mmap_spans[layer_idx].size() : 0;
            size_t mmap_bytes = 0, mmap_maxoff = 0;
            if (layer_idx < (int)wf->ds_mmap_spans.size()) {
                for (auto & ms : wf->ds_mmap_spans[layer_idx]) { mmap_bytes += ms.nbytes; if (ms.gpu_offset + ms.nbytes > mmap_maxoff) mmap_maxoff = ms.gpu_offset + ms.nbytes; }
            }
            // sample within-capacity offsets to avoid OOB reads masking the truth
            const size_t safe_end = (tb <= cap ? tb : cap);
            const size_t offs[3] = { 0, safe_end / 2, safe_end >= 32 ? safe_end - 32 : 0 };
            unsigned long long chk[3] = {0,0,0};
            unsigned char first[3][4];
            for (int k = 0; k < 3; k++) {
                unsigned char b[32] = {0};
                cudaMemcpy(b, (char*)compute_buf + offs[k], 32, cudaMemcpyDeviceToHost);
                unsigned long long s = 0; int nz = 0;
                for (int j = 0; j < 32; j++) { s += b[j]; if (b[j]) nz++; }
                chk[k] = (s << 8) | (unsigned)nz;
                for (int j = 0; j < 4; j++) first[k][j] = b[j];
            }
            fprintf(stderr,
                "*** WF_M3_DIAG L%d path=%s bufs=%zu total=%.1fMB cap=%.1fMB OVERFLOW=%d prefetch_buf=%.1fMB bi=%d "
                "mmap_spans=%zu mmap_MB=%.1f mmap_maxoff=%.1fMB "
                "beg[sum/nz=%llu] mid[sum/nz=%llu] end[sum/nz=%llu]\n",
                layer_idx, wf1_path, wf->gpu_bufs.size(), tb/1e6, cap/1e6, (int)overflow, wf->prefetch_buf_size/1e6, bi,
                mmap_spans, mmap_bytes/1e6, mmap_maxoff/1e6, chk[0], chk[1], chk[2]);
            fflush(stderr);
        }

        layer.stage.store(WF_STAGE_GPU_READY);
    }
#endif

    // Spin-wait for GPU_READY (should be very short if pipeline is ahead)
    int wait_iters = 0;
    while (layer.stage.load() != WF_STAGE_GPU_READY &&
           layer.stage.load() != WF_STAGE_COMPUTING &&
           !wf->ds_failed.load(std::memory_order_acquire)) {
        // Request the layer if not yet in pipeline
        wf_request_layer(wf, layer_idx);
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        // Diagnostic: print status every ~500ms if stuck
        if (++wait_iters % 5000 == 0) {
            wf_layer_stage s = layer.stage.load();
            int ram = layer.ram_buf_idx;
            int gpu = layer.gpu_buf_idx;
            auto & trk = wf->io_trackers[layer_idx];
            WF_ERR("wait_layer_gpu: layer %d stuck (stage=%d, ram=%d, gpu=%d, "
                   "io=%d/%d done, %zu bytes, next=%d)",
                   layer_idx, (int)s, ram, gpu,
                   trk.completed.load(), trk.total_chunks.load(),
                   trk.bytes_done.load(),
                   wf->pipeline_next_layer.load());
            // Also dump buffer status
            for (int bi = 0; bi < (int)wf->ram_bufs.size(); bi++) {
                WF_ERR("  ram_buf[%d]: in_use=%d layer=%d",
                       bi, (int)wf->ram_bufs[bi].in_use.load(), wf->ram_bufs[bi].layer_idx);
            }
            for (int bi = 0; bi < (int)wf->gpu_bufs.size(); bi++) {
                WF_ERR("  gpu_buf[%d]: in_use=%d layer=%d",
                       bi, (int)wf->gpu_bufs[bi].in_use.load(), wf->gpu_bufs[bi].layer_idx);
            }
        }
    }

    if (wf->ds_failed.load(std::memory_order_acquire)) return nullptr;

    auto t1 = std::chrono::high_resolution_clock::now();
    float stall_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (stall_ms > 0.1f) {
        {
            std::lock_guard<std::mutex> lock(wf->stats_mutex);
            wf->stats.total_stall_ms += stall_ms;
            wf->stats.stall_count++;
        }
        WF_TRACE(wf, "wait_layer_gpu: layer %d stalled %.2fms", layer_idx, stall_ms);
    }

    layer.stage.store(WF_STAGE_COMPUTING);

    // Compute for layer L-DIST is the trigger for L's provisional expert set.
    // The ordinary DMA pipeline then resolves the existing tier lookup and
    // writes only remaining predicted spans into L's eventual ring slot.
    wf_expert_prefetch_schedule(wf, layer_idx);

    int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0) return nullptr;

    // Wave-based transfer: data is already in the compute buffer
    // (wf_dstorage_transfer_layer does DS → staging → D2D → compute
    // before setting GPU_READY). No reassembly needed here.

    // Prefetch scheduling moved to wf_prefetch_next_layer (called from post_layer_cb)

    return wf->gpu_bufs[gpu_idx].ptr;
}

// Prefetch first half of the next layer into prefetch_buf.
// Called from post_layer_cb after compute completes — runs synchronously
// but the first-half DMA is "paid for" here so wf_wait_layer_gpu only
// needs the second half, saving ~50% DMA time per layer.
void wf_prefetch_next_layer(wf_context * wf, int current_layer_idx) {
#ifdef WF_HAS_DSTORAGE
    if (!wf || !wf->prefetch_buf || wf->prefetch_buf_size == 0 || !wf->ds_initialized) return;

    int next_il = current_layer_idx + wf->dir;   // +dir: forward next layer, reverse previous layer
    if (next_il < 0 || next_il >= (int)wf->layers.size() || wf->layers[next_il].total_bytes == 0) return;

    size_t half = std::min(wf->prefetch_buf_size, wf->layers[next_il].total_bytes);
    if (!wf_dstorage_transfer_range(wf, next_il, wf->prefetch_buf, 0, half)) {
        wf->prefetch_layer_idx = -1;
        wf->prefetch_ready.store(false);
        return;
    }
    wf->prefetch_layer_idx = next_il;
    wf->prefetch_ready.store(true);
#else
    (void)wf; (void)current_layer_idx;
#endif
}

// Allocate prefetch buffer using remaining VRAM after graph_alloc.
// Call once after the streaming graph is allocated.
void wf_allocate_prefetch(wf_context * wf) {
    if (wf && wf->required_dstorage) return;
#ifdef WF_HAS_CUDA
    if (!wf || wf->prefetch_buf || wf->prefetch_buf_size == 0) return;
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);

    // Query available VRAM
    size_t cuda_free = 0, cuda_total = 0;
    cudaMemGetInfo(&cuda_free, &cuda_total);

    // Use up to 80% of remaining free VRAM, capped at desired size
    size_t max_alloc = (cuda_free * 80) / 100;
    size_t alloc_size = std::min(wf->prefetch_buf_size, max_alloc);
    alloc_size = (alloc_size + 65535) & ~(size_t)65535;

    if (alloc_size < 64 * 1024 * 1024) {
        // Less than 64 MB free — not worth it
        wf->prefetch_buf_size = 0;
        return;
    }

    cudaError_t err = cudaMalloc(&wf->prefetch_buf, alloc_size);
    if (err == cudaSuccess) {
        wf->prefetch_buf_size = alloc_size;
        fprintf(stderr, "*** PREFETCH: allocated %.2f MB (%.0f MB was free)\n",
                alloc_size / (1024.0 * 1024.0), cuda_free / (1024.0 * 1024.0));
    } else {
        wf->prefetch_buf = nullptr;
        wf->prefetch_buf_size = 0;
        fprintf(stderr, "*** PREFETCH: allocation failed (wanted %.2f MB, free=%.0f MB)\n",
                alloc_size / (1024.0 * 1024.0), cuda_free / (1024.0 * 1024.0));
    }
#else
    (void)wf;
#endif
}

void wf_free_prefetch(wf_context * wf) {
#ifdef WF_HAS_CUDA
    if (!wf || !wf->prefetch_buf) return;
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
    cudaFree(wf->prefetch_buf);
    fprintf(stderr, "*** PREFETCH: freed %.2f MB\n", wf->prefetch_buf_size / (1024.0 * 1024.0));
    wf->prefetch_buf = nullptr;
    wf->prefetch_buf_size = 0;
    wf->prefetch_layer_idx = -1;
#else
    (void)wf;
#endif
}

void wf_release_layer(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;

    auto & layer = wf->layers[layer_idx];

    // Skip empty sub-layers — nothing to release
    if (layer.total_bytes == 0 && layer.spans.empty()) {
        layer.stage.store(WF_STAGE_DONE);
        // Don't advance dma_next_layer — the DMA thread handles empty layers
        // via its own skip logic when it encounters them in pending_layer.
        return;
    }

    int gpu_idx = layer.gpu_buf_idx;
#ifdef WF_HAS_DSTORAGE
    if(wf->ds_ring){auto&r=*wf->ds_ring;
        if(!r.prepared||layer_idx!=int(r.released)||gpu_idx!=layer_idx%2){wf_ring_error(wf,"Ring release order changed");return;}
        wf_cuda_record_compute_done(wf,layer_idx);
        if(wf->ds_failed.load(std::memory_order_acquire))return;++r.released;
    }
#endif

    layer.stage.store(WF_STAGE_DONE);
    layer.gpu_buf_idx = -1;
    layer.ram_buf_idx = -1;

#ifdef WF_HAS_DSTORAGE
    // For GPU_BUFS >= 2: schedule next layer for DMA thread (overlapped).
    // For GPU_BUFS == 1: don't schedule — wf_wait_layer_gpu does sync DMA.
    if (gpu_idx >= 0 && wf->ds_initialized && wf->dma_thread.joinable()
        && (int)wf->gpu_bufs.size() > 1) {
        // Recycle this freed buffer onto the next layer IN THE PASS DIRECTION. dir=+1 forward (unchanged),
        // dir=-1 for the training reverse pass (wf_start_reverse) — this was the last hardcoded +1 that made
        // multi-buffer reverse stick (single-buffer skips this path and uses sync DMA). Bound both ends.
        int next = wf->dma_next_layer.fetch_add(wf->dir);
        if (next >= 0 && next < (int)wf->layers.size()) {
            wf->dma_buf_pending[gpu_idx].pending_layer.store(next);
        } else {
            wf->gpu_bufs[gpu_idx].in_use = false;
            wf->gpu_bufs[gpu_idx].layer_idx = -1;
        }
        return;
    }
#endif

    if (gpu_idx >= 0) {
        wf->gpu_bufs[gpu_idx].in_use = false;
        wf->gpu_bufs[gpu_idx].layer_idx = -1;
    }

    // Wake pipeline scheduler — a buffer just freed up
    wf->pipeline_cv.notify_one();
}

void wf_swap_sublayer(wf_context * wf, int old_layer_idx, int new_layer_idx) {
    if (!wf || old_layer_idx < 0 || old_layer_idx >= (int)wf->layers.size()) return;
    if (new_layer_idx < 0 || new_layer_idx >= (int)wf->layers.size()) return;

    auto & old_layer = wf->layers[old_layer_idx];
    int gpu_idx = old_layer.gpu_buf_idx;

    // Mark old layer done
    old_layer.stage.store(WF_STAGE_DONE);
    old_layer.gpu_buf_idx = -1;
    old_layer.ram_buf_idx = -1;

    // Skip if new layer is empty
    if (wf->layers[new_layer_idx].total_bytes == 0) {
        wf->layers[new_layer_idx].stage.store(WF_STAGE_DONE);
        if (gpu_idx >= 0) {
            wf->gpu_bufs[gpu_idx].in_use = false;
            wf->gpu_bufs[gpu_idx].layer_idx = -1;
        }
        return;
    }

#ifdef WF_HAS_DSTORAGE
    // Direct scheduling: tell DMA thread to load new_layer into the same buffer.
    // Does NOT advance dma_next_layer — this is a within-model-layer swap.
    if (gpu_idx >= 0 && wf->ds_initialized && wf->dma_thread.joinable()) {
        wf->dma_buf_pending[gpu_idx].pending_layer.store(new_layer_idx);
        return;
    }
#endif

    if (gpu_idx >= 0) {
        wf->gpu_bufs[gpu_idx].in_use = false;
        wf->gpu_bufs[gpu_idx].layer_idx = -1;
    }
    wf->pipeline_cv.notify_one();
}

void wf_end_pass(wf_context * wf) {
    if (!wf) return;
    if(wf->params.ds_gpu_ring_slots){wf_stop(wf);return;}

    if (wf->params.post_mode == WF_POST_HANDOFF_MMAP) {
        wf_stop(wf);
        WF_LOG("pass ended — handing off to mmap for decode");
    } else {
        // KEEP_FETCHER mode: stop threads cleanly so wf_start() can relaunch.
        // Without this, pipeline_running stays true and wf_start() bails.
        wf->pipeline_running.store(false);
        wf->pipeline_cv.notify_all();

#ifdef WF_HAS_DSTORAGE
        if (wf->dma_thread.joinable()) {
            wf->dma_thread.join();
        }
#endif
        if (wf->pipeline_thread.joinable()) {
            wf->pipeline_thread.join();
        }

        // Reset layer/buffer state for next pass
        wf->pipeline_next_layer.store(0);
        for (auto & layer : wf->layers) {
            layer.stage.store(WF_STAGE_IDLE);
            layer.gpu_buf_idx = -1;
            layer.ram_buf_idx = -1;
            layer.expert_filter_set = false;
            layer.cache_inserted = false;
        }
        for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
        for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }

#ifdef WF_HAS_DSTORAGE
        wf->dma_next_layer.store(0);
        for (auto & slot : wf->dma_buf_pending) { slot.pending_layer.store(-1); }
        for (auto & fv : wf->ds_layer_fence_val) { fv = 0; }
#endif

        WF_LOG("pass ended — pipeline stopped and reset for next pass");
    }
}

void wf_stop(wf_context * wf) {
    if (!wf) return;

    // Stop pipeline thread
    wf->pipeline_running.store(false);
    wf->pipeline_cv.notify_all();

#ifdef WF_HAS_DSTORAGE
    // Join DMA control thread (it polls pipeline_running, exits within ~50μs)
    if (wf->dma_thread.joinable()) {
        wf->dma_thread.join();
    }
#endif

    if (wf->pipeline_thread.joinable()) {
        wf->pipeline_thread.join();
    }
#ifdef WF_HAS_DSTORAGE
    wf_dstorage_ring_stop(wf);
#endif

#ifdef _WIN32
    // Cancel all outstanding I/O on all file handles before shutting down workers.
    // CancelIoEx (not CancelIo) cancels I/O from ANY thread, not just the caller.
    // The ReadFile calls were issued by pipeline + IOCP worker threads, not this thread.
    for (auto & fh : wf->mt_files) {
        if (fh.h != INVALID_HANDLE_VALUE) {
            CancelIoEx(fh.h, NULL);
        }
    }
    for (auto & fh : wf->files) {
        if (fh.h != INVALID_HANDLE_VALUE) {
            CancelIoEx(fh.h, NULL);
        }
    }
#endif

    // Stop I/O threads
    wf->io_shutdown.store(true);

#ifdef _WIN32
    // Post dummy completions to wake IOCP workers
    for (int i = 0; i < (int)wf->io_threads.size(); i++) {
        PostQueuedCompletionStatus(wf->iocp, 0, 0, nullptr);
    }
#else
    wf->io_fallback_cv.notify_all();
#endif

    for (auto & t : wf->io_threads) {
        if (t.joinable()) t.join();
    }
    wf->io_threads.clear();
    wf->io_shutdown.store(false);

    WF_LOG("pipeline stopped");
}

void wf_quiesce(wf_context * wf) {
    if (!wf) return;

#ifdef WF_HAS_CUDA
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
#ifdef WF_HAS_DSTORAGE
    if (wf->ds_initialized) {
        wf_dstorage_quiesce(wf);
        return;
    }
#endif
    if (wf->copy_stream) cudaStreamSynchronize(wf->copy_stream);
    if (wf->cuda_compute_stream) cudaStreamSynchronize(wf->cuda_compute_stream);
    cudaDeviceSynchronize();
#endif
}

bool wf_set_layer_persistent(wf_context * wf, int layer_idx, bool persistent) {
    if (!wf || !wf->required_dstorage || layer_idx < 0 || layer_idx >= (int)wf->layers.size() || wf->pipeline_running.load()) return false;
    auto & layer = wf->layers[layer_idx];
    if (layer.spans.empty()) return false;
    for (const auto & span : layer.spans) if (span.is_expert) return false;  // routed stages are rewritten every token
    layer.persistent = persistent;
    layer.persistent_loaded.store(false, std::memory_order_release);
    return true;
}

void wf_set_active_experts(wf_context * wf, int layer_idx,
                           const int * expert_ids, int n_active) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    if (n_active < 0 || (n_active > 0 && !expert_ids)) return;

    auto & layer = wf->layers[layer_idx];
    if (!wf->expert_prefetch_enabled) {
        // The required DMA thread reads this filter under the same lock.
        std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
        if (n_active > 0) layer.active_experts.assign(expert_ids, expert_ids + n_active);
        else layer.active_experts.clear();
        layer.expert_filter_set = true;
        return;
    }
    auto actual_contains = [&](int eid) {
        for (int i = 0; i < n_active; ++i) if (expert_ids[i] == eid) return true;
        return false;
    };

    bool planned = false;
    {
        std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
        planned = layer.expert_prefetch_planned;
    }
    // If DS has started consuming the provisional filter, let that bounded
    // transfer finish before replacing the vector it reads.
    while (planned) {
        wf_layer_stage stage = layer.stage.load();
        if (stage == WF_STAGE_IDLE || stage == WF_STAGE_GPU_READY ||
                stage == WF_STAGE_COMPUTING || stage == WF_STAGE_DONE) break;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    std::vector<int> misses;
    bool loaded = false;
    size_t wasted_bytes = 0;
    uint64_t hits = 0;
    uint64_t predicted = 0;
    {
        std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
        loaded = layer.expert_prefetch_loaded;
        if (planned && loaded) {
            predicted = layer.expert_prefetch_ids.size();
            for (int eid : layer.expert_prefetch_ids) {
                if (actual_contains(eid)) ++hits;
            }
            for (int i = 0; i < n_active; ++i) {
                if (std::find(layer.expert_prefetch_ids.begin(), layer.expert_prefetch_ids.end(), expert_ids[i]) ==
                        layer.expert_prefetch_ids.end() &&
                        std::find(misses.begin(), misses.end(), expert_ids[i]) == misses.end()) {
                    misses.push_back(expert_ids[i]);
                }
            }
            for (int si = 0; si < (int) layer.spans.size() &&
                    si < (int) layer.expert_prefetch_loaded_spans.size(); ++si) {
                const auto & span = layer.spans[si];
                if (layer.expert_prefetch_loaded_spans[si] &&
                        !actual_contains(span.expert_idx)) {
                    wasted_bytes += span.nbytes;
                }
            }
        }
        if (n_active > 0) layer.active_experts.assign(expert_ids, expert_ids + n_active);
        else layer.active_experts.clear();
        layer.expert_filter_set = true;
        layer.expert_prefetch_planned = false;
        layer.expert_prefetch_loaded = false;
        layer.expert_prefetch_ids.clear();
        layer.expert_prefetch_loaded_spans.clear();
        wf_expert_prefetch_update_ema(wf, layer_idx, expert_ids, n_active);
    }

    if (planned && loaded) {
        std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
        wf->stats.expert_prefetch_predicted += predicted;
        wf->stats.expert_prefetch_hits += hits;
        wf->stats.expert_prefetch_misses += misses.size();
        wf->stats.expert_prefetch_wasted_bytes += wasted_bytes;
    }
#if defined(WF_HAS_DSTORAGE) && defined(WF_HAS_CUDA)
    if (loaded && !misses.empty()) wf_expert_prefetch_fetch_misses(wf, layer_idx, misses);
#endif
}

wf_moe_mode wf_get_moe_mode(wf_context * wf) {
    if (!wf) return WF_MOE_FETCH_ALL;
    return wf->params.moe_mode;
}

void * wf_get_gpu_buffer(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return nullptr;
    int gpu_idx = wf->layers[layer_idx].gpu_buf_idx;
    if (gpu_idx < 0) return nullptr;
    return wf->gpu_bufs[gpu_idx].ptr;
}

size_t wf_get_span_gpu_offset(wf_context * wf, int layer_idx, int span_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return 0;
    auto & layer = wf->layers[layer_idx];
    if (span_idx < 0 || span_idx >= (int)layer.spans.size()) return 0;
    return layer.spans[span_idx].gpu_offset;
}

void wf_cuda_wait_upload(wf_context * wf, int layer_idx) {
#ifdef WF_HAS_CUDA
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    if (wf->ds_failed.load(std::memory_order_acquire)) return;

#ifdef WF_HAS_DSTORAGE
    // DirectStorage path: GPU-side wait on D3D12 fence via CUDA external semaphore.
    // CRITICAL: must wait on the actual compute stream, not the null stream.
    // ggml-cuda creates all streams with cudaStreamNonBlocking, so the null
    // stream does NOT synchronize with them.  Without this fix, the compute
    // kernel reads from the D3D12 buffer before DMA completes → illegal access.
    if (wf->ds_initialized && layer_idx < (int)wf->ds_layer_fence_val.size()) {
        uint64_t fence_val = wf->ds_layer_fence_val[layer_idx];
        if (fence_val > 0) {
            WF_TRACE(wf, "cuda_wait_upload: layer %d -> DS fence %llu (completed=%llu)",
                     layer_idx,
                     (unsigned long long) fence_val,
                     (unsigned long long) (wf->ds_fence ? wf->ds_fence->GetCompletedValue() : 0));
            if (!wf_cuda_wait_ds_fence(wf, fence_val, wf->cuda_compute_stream)) {
                WF_ERR("DirectStorage CUDA fence %llu maps to weight layer %d",
                       (unsigned long long) fence_val, layer_idx);
                return;
            }
        }
    }
#endif

    int gpu_idx = wf->layers[layer_idx].gpu_buf_idx;
    if (gpu_idx < 0) return;

    auto & slot = wf->gpu_bufs[gpu_idx];
    if (slot.upload_done) {
        // Make the compute stream wait for the upload event.
        // Uses the actual ggml-cuda stream (not null — ggml uses NonBlocking).
        const cudaError_t waited = cudaStreamWaitEvent(wf->cuda_compute_stream, slot.upload_done, 0);
        if (wf->required_dstorage && waited != cudaSuccess) { wf->ds_failed.store(true,std::memory_order_release);return; }
    }

    // Insert after upload completion is ordered. All cache transfers use the
    // WF copy stream, and the helper re-records upload_done after its copies.
#ifdef WF_HAS_DSTORAGE
    if (wf_cache_enabled(wf) && wf->ds_initialized && wf->copy_stream &&
            layer_idx < (int)wf->ds_layer_fence_val.size()) {
        if (!wf_cuda_wait_ds_fence(
                wf, wf->ds_layer_fence_val[layer_idx], wf->copy_stream)) return;
    }
#endif
    wf_cache_insert_layer(wf, layer_idx);
    if (slot.upload_done) {
        const cudaError_t waited = cudaStreamWaitEvent(wf->cuda_compute_stream, slot.upload_done, 0);
        if (wf->required_dstorage && waited != cudaSuccess) { wf->ds_failed.store(true,std::memory_order_release);return; }
    }

    // ROUND-10 PROBE (LLAMA_DSV4_WF_CKSUM=<tsv>): post-upload GPU-vs-file checksum of the
    // indexer-related weight spans, per layer, first visit only. The round-9 audit eliminated
    // every other layer-selective mechanism; the surviving gap is upload fidelity of specific
    // layers' spans. Same-run comparison: hash the uploaded GPU bytes AND the mmap source bytes.
    static const char * wf_cksum_path = std::getenv("LLAMA_DSV4_WF_CKSUM");
    if (!wf->required_dstorage && wf_cksum_path && wf_cksum_path[0]) {
        static std::mutex cksum_mutex;
        static std::set<std::pair<int, std::string>> done_set;
        std::lock_guard<std::mutex> lk(cksum_mutex);
        auto & layer = wf->layers[layer_idx];
        if (slot.upload_done) cudaEventSynchronize(slot.upload_done);
        for (const auto & span : layer.spans) {
            if (!strstr(span.tensor_name, "indexer")) continue;
            auto key = std::make_pair(layer_idx, std::string(span.tensor_name));
            if (done_set.count(key)) continue;
            done_set.insert(key);
            std::vector<uint8_t> gpu_bytes(span.nbytes);
            const cudaError_t ce = cudaMemcpy(gpu_bytes.data(),
                    (const uint8_t *) slot.ptr + span.gpu_offset, span.nbytes, cudaMemcpyDeviceToHost);
            auto fnv = [](const uint8_t * p, size_t n) {
                uint64_t h = 1469598103934665603ull;
                for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
                return h;
            };
            const uint64_t gpu_hash  = (ce == cudaSuccess) ? fnv(gpu_bytes.data(), span.nbytes) : 0;
            const uint64_t file_hash = span.mmap_addr ? fnv((const uint8_t *) span.mmap_addr, span.nbytes) : 0;
            if (FILE * fp = fopen(wf_cksum_path, "ab+")) {
                fprintf(fp, "%d\t%s\t%zu\t%016llx\t%016llx\t%s\n",
                        layer_idx, span.tensor_name, span.nbytes,
                        (unsigned long long) gpu_hash, (unsigned long long) file_hash,
                        ce != cudaSuccess ? "COPYERR" : (span.mmap_addr == nullptr ? "NOFILE" :
                        (gpu_hash == file_hash ? "MATCH" : "MISMATCH")));
                fclose(fp);
            }
        }
    }
#else
    (void)wf; (void)layer_idx;
#endif
}

#ifdef WF_HAS_CUDA
// Lock-free readiness (LLAMA_WF_LOCKFREE_READY=1): the DMA thread polls a host
// atomic instead of cudaEventQuery, removing ~40k/s CUDA-driver-lock grabs that
// contend with the host's kernel-launch burst (the reason the per-op OP_PROFILE
// sync crutch is needed to feed the DMA). With this, compute can run ahead without
// host-pacing -> compute ~29.3s + low wf -> ~685 tps @20k target. Default off.
static bool wf_lockfree_ready_enabled() {
    static const bool e = []() {
        const char * v = std::getenv("LLAMA_WF_LOCKFREE_READY");
        return v && v[0] && v[0] != '0';
    }();
    return e;
}
static void CUDART_CB wf_compute_ready_cb(void * userData) {
    auto * flag = static_cast<std::atomic<bool> *>(userData);
    if (flag) flag->store(true, std::memory_order_release);
}
#endif

void wf_cuda_record_compute_done(wf_context * wf, int layer_idx) {
#ifdef WF_HAS_CUDA
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    int gpu_idx = wf->layers[layer_idx].gpu_buf_idx;
    if (gpu_idx < 0) return;

    // Record event on the compute stream. The event fires when all prior
    // compute work completes. The DMA thread checks this event via
    // cudaEventQuery before reusing the buffer — NO CPU-side sync needed.
    //
    // Previously used cudaDeviceSynchronize() here, which blocked the CPU
    // (and DMA thread) until ALL GPU work finished. This created dead time
    // between NVMe bursts (visible as spiky transfer graphs). With events,
    // the DMA thread can start the next layer's NVMe read immediately,
    // and the GPU handles ordering internally via event queries.
    //
    // For GPU_BUFS=1 (sync path): wf_wait_layer_gpu already does
    // cudaStreamSynchronize(compute_stream) before the next D2D copy.
    // For GPU_BUFS=2 (DMA thread): the DMA thread polls cudaEventQuery.

    auto & slot = wf->gpu_bufs[gpu_idx];
    if (slot.compute_done) {
        const cudaError_t recorded = cudaEventRecord(slot.compute_done, wf->cuda_compute_stream);
        if (wf->required_dstorage && recorded != cudaSuccess) { wf->ds_failed.store(true,std::memory_order_release);return; }
    }
    if (!wf->required_dstorage && wf_lockfree_ready_enabled() && gpu_idx < (int)wf->dma_buf_pending.size()) {
        // Fires on the compute stream AFTER this layer's compute completes,
        // marking buffer gpu_idx reusable lock-free (DMA polls compute_ready).
        // cudaLaunchHostFunc is illegal during stream capture (n_ubatches==1 fast
        // path) — guard it; in capture mode the event-query path is used instead.
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(wf->cuda_compute_stream, &cap);
        if (cap == cudaStreamCaptureStatusNone) {
            cudaLaunchHostFunc(wf->cuda_compute_stream, wf_compute_ready_cb,
                               &wf->dma_buf_pending[gpu_idx].compute_ready);
        } else {
            // Capturing: can't enqueue a host func. Mark ready directly so the DMA
            // (which also won't be mid-pipeline during single-ubatch capture) isn't stuck.
            wf->dma_buf_pending[gpu_idx].compute_ready.store(true, std::memory_order_release);
        }
    }
#else
    (void)wf; (void)layer_idx;
#endif
}

bool wf_cuda_get_vram_info(size_t * free_bytes, size_t * total_bytes) {
#ifdef WF_HAS_CUDA
    if (!free_bytes || !total_bytes) return false;
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return err == cudaSuccess;
#else
    (void)free_bytes; (void)total_bytes;
    return false;
#endif
}

int wf_get_n_gpu_bufs(wf_context * wf) {
    return wf ? (int)wf->gpu_bufs.size() : 0;
}

size_t wf_get_gpu_buf_capacity(wf_context * wf) {
    if (!wf || wf->gpu_bufs.empty()) return 0;
    return wf->gpu_bufs[0].capacity;
}

void * wf_get_gpu_buf_ptr(wf_context * wf, int buf_idx) {
    if (!wf || buf_idx < 0 || buf_idx >= (int)wf->gpu_bufs.size()) return nullptr;
    return wf->gpu_bufs[buf_idx].ptr;
}

int wf_get_cuda_device_index(wf_context * wf) {
#ifdef WF_HAS_CUDA
    return wf ? wf->cuda_device_index : 0;
#else
    (void)wf; return 0;
#endif
}

bool wf_is_ds_direct(wf_context * wf) {
    return wf && wf->ds_direct_mode;
}

void wf_set_gpu_buf_ptr(wf_context * wf, int buf_idx, void * ptr) {
    if (!wf || buf_idx < 0 || buf_idx >= (int)wf->gpu_bufs.size()) return;
    // Free the old cudaMalloc'd buffer if present
    if (wf->gpu_bufs[buf_idx].ptr && wf->gpu_bufs[buf_idx].ptr != ptr && wf->gpu_bufs[buf_idx].owns_ptr) {
#ifdef WF_HAS_CUDA
        wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
        cudaFree(wf->gpu_bufs[buf_idx].ptr);
#else
        free(wf->gpu_bufs[buf_idx].ptr);
#endif
    }
    wf->gpu_bufs[buf_idx].ptr = ptr;
    wf->gpu_bufs[buf_idx].owns_ptr = false;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

bool wf_get_required_layer_fetch(wf_context * wf, int layer, uint64_t * bytes, double * milliseconds) {
    if (!wf || !bytes || !milliseconds) return false;
    *bytes = 0; *milliseconds = 0;
#ifdef WF_HAS_DSTORAGE
    if (!wf->required_dstorage || layer < 0 || layer >= int(wf->ds_waves.size())) return false;
    std::lock_guard<std::mutex> lock(wf->dma_timing_mutex);
    for (auto it = wf->dma_timing_log.rbegin(); it != wf->dma_timing_log.rend(); ++it) {
        if (it->layer_idx != layer) continue;
        if (it->nbytes == 0 && wf->layers[layer].persistent) { *bytes = 0; *milliseconds = 0; return true; }
        break;
    }
    for (const auto & wave : wf->ds_waves[layer])
        for (size_t ri = 0; ri < wave.reqs.size(); ++ri) {
            // Routed stages count only the experts actually fetched.
            const int si = ri < wave.req_span_idx.size() ? wave.req_span_idx[ri] : -1;
            if (si >= 0 && si < (int) wf->layers[layer].spans.size()) {
                const auto & span = wf->layers[layer].spans[si];
                if (span.is_expert && !wf_span_is_active_expert(wf, wf->layers[layer], span)) continue;
            }
            *bytes += wave.reqs[ri].Source.File.Size;
        }
    for (auto it = wf->dma_timing_log.rbegin(); it != wf->dma_timing_log.rend(); ++it) {
        if (it->layer_idx != layer) continue;
        if (!it->t_fence.time_since_epoch().count()) return false;
        *milliseconds = std::chrono::duration<double, std::milli>(it->t_fence - it->t_submit).count();
        return *bytes > 0 && *milliseconds > 0;
    }
    // The baseline single-buffer path completes synchronously in acquire and
    // records the same transfer interval directly, without a DMA-thread entry.
    *milliseconds = wf->layers[layer].nvme_read_ms;
    return *bytes > 0 && *milliseconds > 0;
#endif
    return false;
}

void wf_get_stats(wf_context * wf, wf_stats * stats) {
    if (!wf || !stats) return;

    // Merge atomic counters into stats struct
    {
        std::lock_guard<std::mutex> lock(wf->stats_mutex);
        *stats = wf->stats;
    }

    // Overwrite with atomic hot-path counters
    stats->bytes_read_nvme = wf->stat_bytes_read_nvme.load(std::memory_order_relaxed);
    stats->io_ops          = wf->stat_io_ops.load(std::memory_order_relaxed);

    // Sum per-layer NVMe read times for total_nvme_read_ms
    // (each layer's nvme_read_ms is wall-clock from submit to last-chunk-done)
    float sum_nvme_ms = 0.0f;
    for (auto & layer : wf->layers) {
        sum_nvme_ms += layer.nvme_read_ms;
    }
    stats->total_nvme_read_ms = sum_nvme_ms;

    // Sum per-layer H2D times
    float sum_h2d_ms = 0.0f;
    for (auto & layer : wf->layers) {
        sum_h2d_ms += layer.h2d_ms;
    }
    stats->total_h2d_ms = sum_h2d_ms;

    // Wall-clock bandwidth: total bytes / wall time since wf_start()
    auto now = std::chrono::high_resolution_clock::now();
    float wall_ms = std::chrono::duration<float, std::milli>(now - wf->pass_start).count();

    if (wall_ms > 0.0f) {
        stats->nvme_read_bw_gbps = (float)stats->bytes_read_nvme /
            (wall_ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
        stats->end_to_end_bw_gbps = stats->nvme_read_bw_gbps;
    }
    if (sum_h2d_ms > 0) {
        stats->ram_to_gpu_bw_gbps = (float)stats->bytes_uploaded_gpu /
            (sum_h2d_ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
    }

#ifdef WF_HAS_GDS
    stats->gds_active = wf->gds_available;
#else
    stats->gds_active = false;
#endif
}

void wf_reset_stats(wf_context * wf) {
    if (!wf) return;
    std::lock_guard<std::mutex> lock(wf->stats_mutex);
    memset(&wf->stats, 0, sizeof(wf->stats));
    wf->stats.effective_mode = wf->effective_mode;
    wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
    wf->stat_io_ops.store(0, std::memory_order_relaxed);
    wf->expert_prefetch_log_predicted = 0;
    wf->expert_prefetch_log_hits = 0;
    wf->expert_prefetch_log_misses = 0;
    wf->expert_prefetch_log_waste = 0;
    wf->pass_start = std::chrono::high_resolution_clock::now();
}

void wf_print_stats(wf_context * wf) {
    wf_stats s;
    wf_get_stats(wf, &s);

    // wf_print_stats is explicitly called — always print
    WF_ERR("=== Weight Fetcher Stats ===");
    WF_ERR("  mode:           %s%s",
           s.effective_mode == WF_MODE_DSTORAGE  ? "DirectStorage" :
           s.effective_mode == WF_MODE_GDS       ? "GPUDirect Storage" :
           s.effective_mode == WF_MODE_DIRECT_IO ? "Direct I/O" :
           s.effective_mode == WF_MODE_MMAP      ? "mmap" : "unknown",
           s.gds_active ? " (active)" : "");
    WF_ERR("  NVMe read:      %.2f GB in %.2fms (%.2f GB/s)",
           s.bytes_read_nvme / (1024.0 * 1024.0 * 1024.0),
           s.total_nvme_read_ms, s.nvme_read_bw_gbps);
    WF_ERR("  H2D upload:     %.2f GB in %.2fms (%.2f GB/s)",
           s.bytes_uploaded_gpu / (1024.0 * 1024.0 * 1024.0),
           s.total_h2d_ms, s.ram_to_gpu_bw_gbps);
    WF_ERR("  end-to-end:     %.2f GB/s", s.end_to_end_bw_gbps);
    WF_ERR("  GPU stalls:     %d (%.2fms total)", s.stall_count, s.total_stall_ms);
    WF_ERR("  I/O ops:        %d", s.io_ops);
    WF_ERR("  layers fetched: %d", s.layers_fetched);
    if (s.experts_skipped > 0) {
        WF_ERR("  MoE experts skipped: %d (%.2f MB saved)",
               s.experts_skipped, s.expert_bytes_saved / (1024.0 * 1024.0));
    }
    if (wf_cache_enabled(wf)) {
        WF_ERR("  expert cache: VRAM hits=%llu, RAM hits=%llu, misses=%llu (%.2f GB saved)",
               (unsigned long long)s.cache_vram_hits,
               (unsigned long long)s.cache_ram_hits,
               (unsigned long long)s.cache_misses,
               s.cache_bytes_saved / (1024.0 * 1024.0 * 1024.0));
    }
    if (wf->expert_prefetch_enabled) {
        WF_ERR("  expert prefetch: predicted=%llu, hits=%llu, misses=%llu, waste=%.2f MB",
               (unsigned long long) s.expert_prefetch_predicted,
               (unsigned long long) s.expert_prefetch_hits,
               (unsigned long long) s.expert_prefetch_misses,
               s.expert_prefetch_wasted_bytes / (1024.0 * 1024.0));
    }

    // Dump per-transfer DMA timing log
    {
        std::lock_guard<std::mutex> lk(wf->dma_timing_mutex);
        if (!wf->dma_timing_log.empty()) {
            auto t0 = wf->dma_timing_log[0].t_submit;
            double total_gap_ms = 0;
            double total_xfer_ms = 0;
            int n_entries = (int)wf->dma_timing_log.size();
            int n_slow = 0;  // transfers > 1.5× expected time

            for (int i = 0; i < n_entries; i++) {
                auto & e = wf->dma_timing_log[i];
                double submit_ms = std::chrono::duration<double, std::milli>(e.t_submit - t0).count();
                double fence_ms  = e.t_fence.time_since_epoch().count() > 0
                                 ? std::chrono::duration<double, std::milli>(e.t_fence - t0).count()
                                 : -1.0;
                double xfer_ms   = fence_ms >= 0 ? (fence_ms - submit_ms) : -1.0;
                double expected_ms = e.nbytes / (20.0e9 / 1000.0);  // at 20 GB/s
                double gap_ms    = (i > 0 && wf->dma_timing_log[i-1].t_fence.time_since_epoch().count() > 0)
                                 ? std::chrono::duration<double, std::milli>(e.t_submit - wf->dma_timing_log[i-1].t_fence).count()
                                 : 0.0;

                if (xfer_ms > 0) total_xfer_ms += xfer_ms;
                if (i > 0) total_gap_ms += gap_ms;
                if (xfer_ms > expected_ms * 1.5) n_slow++;

                // Print first 10, last 5, and any slow transfers
                if (i < 10 || i >= n_entries - 5 || xfer_ms > expected_ms * 1.5) {
                    WF_ERR("  DMA[%3d] layer=%3d buf=%d %6.1f MB | submit=%8.1fms fence=%8.1fms xfer=%6.1fms (%.1f GB/s) gap=%5.1fms%s",
                           i, e.layer_idx, e.buf_idx, e.nbytes / (1024.0 * 1024.0),
                           submit_ms, fence_ms, xfer_ms,
                           xfer_ms > 0 ? (e.nbytes / (xfer_ms / 1000.0)) / 1e9 : 0.0,
                           gap_ms,
                           xfer_ms > expected_ms * 1.5 ? " *** SLOW" : "");
                } else if (i == 10) {
                    WF_ERR("  ... (%d transfers omitted, showing slow ones) ...", n_entries - 15);
                }
            }

            double avg_xfer = n_entries > 0 ? total_xfer_ms / n_entries : 0;
            double avg_gap  = n_entries > 1 ? total_gap_ms / (n_entries - 1) : 0;
            WF_ERR("  DMA summary: %d transfers, avg xfer=%.1fms, avg gap=%.1fms, %d slow (>1.5x expected)",
                   n_entries, avg_xfer, avg_gap, n_slow);
        }
    }
}

bool wf_gds_available(void) {
#ifdef WF_HAS_GDS
    CUfileError_t status = cuFileDriverOpen();
    if (status.err == CU_FILE_SUCCESS) {
        cuFileDriverClose();
        return true;
    }
#endif
    return false;
}

wf_layer_stage wf_get_layer_stage(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return WF_STAGE_IDLE;
    return wf->layers[layer_idx].stage.load();
}

wf_transfer_mode wf_get_effective_mode(wf_context * wf) {
    if (!wf) return WF_MODE_MMAP;
    return wf->effective_mode;
}

bool wf_get_layer_file_range(wf_context * wf, int layer_idx,
                             uint64_t * out_first_offset,
                             size_t * out_total_bytes) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return false;
    auto & layer = wf->layers[layer_idx];
    if (out_total_bytes) *out_total_bytes = layer.total_bytes;
    // Find first non-zero file offset among spans
    uint64_t first_off = 0;
    for (auto & span : layer.spans) {
        if (span.file_offset > 0) {
            first_off = span.file_offset;
            break;
        }
    }
    if (out_first_offset) *out_first_offset = first_off;
    return true;
}

const char * wf_get_model_path(wf_context * wf) {
    if (!wf) return nullptr;
    return wf->params.model_path;
}

void wf_set_layer_major_mode(wf_context * wf, bool enabled) {
    if (!wf) return;
    wf->layer_major_mode = enabled;
    WF_ERR("layer-major mode %s", enabled ? "ENABLED" : "DISABLED");
}

bool wf_get_layer_major_mode(wf_context * wf) {
    if (!wf) return false;
    return wf->layer_major_mode;
}


int wf_persist_shutdown(const char * why) {
#ifdef WF_HAS_DSTORAGE
    return wf_persist_shutdown_impl(why ? why : "unspecified");
#else
    (void) why;
    return 1;
#endif
}

bool wf_fetch_extents_direct(wf_context * wf, int bank, const wf_extent * extents, int n,
                             void * cuda_stream, uint64_t * bytes, double * milliseconds) {
    if (bytes) *bytes = 0;
    if (milliseconds) *milliseconds = 0;
#if defined(WF_HAS_DSTORAGE) && defined(WF_HAS_CUDA)
    if (!wf || !wf->required_dstorage || !wf->params.ds_compute_direct || !wf->ds_queue || !wf->ds_fence ||
            bank < 0 || bank >= (int) wf->ds_compute_banks.size() || n < 0 || (n && !extents) ||
            wf->ds_failed.load(std::memory_order_acquire)) return false;
    if (n == 0) return true;
    const auto & target = wf->ds_compute_banks[bank];
    const uint64_t quarter = target.quarter_bytes, capacity = quarter * target.quarters.size();
    const size_t block = wf->params.io_block_size;
    if (!quarter || !block) return false;
    auto started = std::chrono::high_resolution_clock::now();
    // Claim the single DS submission slot (the DMA thread uses the same queue).
    for (;;) {
        int expected = -1;
        if (wf->dma_in_flight_buf.compare_exchange_strong(expected, 1000 + bank)) break;
        if (wf->ds_failed.load(std::memory_order_acquire)) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    struct release_slot { wf_context * wf; ~release_slot() { wf->dma_in_flight_buf.store(-1); } } slot{wf};
    uint64_t total = 0; int submitted = 0;
    const int drain_threshold = std::max(1, wf->ds_queue_cap / 2);
    for (int i = 0; i < n; ++i) {
        const auto & e = extents[i];
        if (!e.nbytes || e.file_idx < 0 || e.file_idx >= (int) wf->ds_files.size() ||
                e.bank_offset > capacity || e.nbytes > capacity - e.bank_offset) {
            WF_ERR("Required DirectStorage extent fetch refused: invalid extent %d", i);
            wf->ds_failed.store(true, std::memory_order_release); return false;
        }
        uint64_t source = e.file_offset, destination = e.bank_offset, remaining = e.nbytes;
        while (remaining) {
            const uint64_t q = destination / quarter, within = destination % quarter;
            const uint64_t chunk = std::min<uint64_t>({remaining, (uint64_t) block, quarter - within});
            if (submitted >= drain_threshold) {
                const uint64_t drain = ++wf->ds_fence_counter;
                wf->ds_queue->EnqueueSignal(wf->ds_fence, drain); wf->ds_queue->Submit();
                if (!wf_ds_fence_wait(wf, drain)) return false;
                submitted = 0;
            }
            DSTORAGE_REQUEST req = {};
            req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            req.Source.File.Source = wf->ds_files[e.file_idx];
            req.Source.File.Offset = source;
            req.Source.File.Size = (UINT32) chunk;
            req.Destination.Buffer.Resource = target.quarters[q];
            req.Destination.Buffer.Offset = within;
            req.Destination.Buffer.Size = (UINT32) chunk;
            req.UncompressedSize = (UINT32) chunk;
            wf->ds_queue->EnqueueRequest(&req); ++submitted;
            source += chunk; destination += chunk; remaining -= chunk; total += chunk;
        }
    }
    const uint64_t fence = ++wf->ds_fence_counter;
    wf->ds_queue->EnqueueSignal(wf->ds_fence, fence);
    wf->ds_queue->Submit();
    if (!wf_ds_fence_wait(wf, fence)) return false;
    if (!wf_dstorage_wave_succeeded(wf, wf->ds_queue, -1, -1)) return false;
    if (!wf_cuda_wait_ds_fence(wf, fence, (cudaStream_t) cuda_stream)) return false;
    wf->stat_bytes_read_nvme.fetch_add(total);
    if (bytes) *bytes = total;
    if (milliseconds) *milliseconds = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - started).count();
    return true;
#else
    (void) wf; (void) bank; (void) extents; (void) n; (void) cuda_stream;
    return false;
#endif
}
