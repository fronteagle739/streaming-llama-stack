#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Phase 296 (2026-05-04): release all cached freed buffers from the backend's
// per-stream memory pools back to the CUDA driver. Use between long-running
// graph executions (e.g. between streaming-prefill requests) to prevent the
// legacy pool from accumulating 20+ GB of cached freed buffers across cycles,
// which causes OOM on subsequent prefills. Returns total bytes released.
GGML_BACKEND_API size_t ggml_backend_cuda_pool_clear(ggml_backend_t backend);

// Phase 299 (2026-05-04): per-process CUDA buffer alloc/free tracking.
// Counters increment in alloc_buffer and decrement in the wrapper destructor.
// Use to spot leaks across streaming-prefill cycles: alloc - free should equal
// (live count's worth of) currently-held buffers. If alloc keeps growing
// without matching free, buffers are being leaked at the ggml wrapper level.
extern "C" GGML_BACKEND_API int64_t ggml_backend_cuda_buffer_alloc_total();
extern "C" GGML_BACKEND_API int64_t ggml_backend_cuda_buffer_free_total();
extern "C" GGML_BACKEND_API int64_t ggml_backend_cuda_buffer_live_count();

// Phase 300 (2026-05-04): per-process cudaMalloc accounting via
// ggml_cuda_device_malloc. Catches both wrapper AND pool direct allocations,
// pinpointing whether the leak is in wrapper code or pool code.
extern "C" GGML_BACKEND_API int64_t ggml_cuda_device_malloc_total_bytes();
extern "C" GGML_BACKEND_API int64_t ggml_cuda_device_malloc_total_count();

// Phase 301 (2026-05-04): cuda_graphs map accounting + clear API.
// Each unique graph first_node_ptr creates a ggml_cuda_graph holding device
// resources (instance buffers, kernel arg buffers). Returns current map size.
GGML_BACKEND_API size_t ggml_backend_cuda_graphs_count(ggml_backend_t backend);
GGML_BACKEND_API size_t ggml_backend_cuda_graphs_clear(ggml_backend_t backend);
GGML_BACKEND_API int64_t ggml_backend_cuda_graphs_total_created(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_cuda_graphs_disable_push(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_cuda_graphs_disable_pop(ggml_backend_t backend);
GGML_BACKEND_API bool ggml_backend_cuda_graphs_are_disabled(ggml_backend_t backend);

// Phase 304 (2026-05-04): per-process cudaEventCreate / cudaStreamCreate counts
// from the file-scope Meyers-singleton atomics in common.cuh. Counts only the
// call sites we've explicitly hooked (currently: streaming-prefill event
// creates in llama-context.cpp — the suspected per-cycle leak source).
extern "C" GGML_BACKEND_API int64_t ggml_cuda_event_create_total();
extern "C" GGML_BACKEND_API int64_t ggml_cuda_event_destroy_total();
extern "C" GGML_BACKEND_API int64_t ggml_cuda_stream_create_total();
// Manual increment APIs for callers outside ggml-cuda translation units.
extern "C" GGML_BACKEND_API void ggml_cuda_event_create_count_inc();
extern "C" GGML_BACKEND_API void ggml_cuda_event_destroy_count_inc();
extern "C" GGML_BACKEND_API void ggml_cuda_stream_create_count_inc();
// Phase 306 (2026-05-04): pinned host memory tracking. cudaMallocHost on
// Windows consumes GPU page-table entries that show up in cudaMemGetInfo
// as VRAM "free" decrease, even though physical memory is host RAM. Tracking
// alloc - free totals shows whether pinned memory is the per-cycle leak.
extern "C" GGML_BACKEND_API int64_t ggml_cuda_pinned_alloc_total_bytes();
extern "C" GGML_BACKEND_API int64_t ggml_cuda_pinned_free_total_bytes();
extern "C" GGML_BACKEND_API void    ggml_cuda_pinned_alloc_track(size_t bytes);
extern "C" GGML_BACKEND_API void    ggml_cuda_pinned_free_track(size_t bytes);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Return the CUDA compute stream for the given backend (as void * to avoid
// exposing cudaStream_t in the public header).  Returns nullptr if not CUDA.
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);

// GLM-DSA sparse-gather invalid-slot accounting. The counter is request-scoped:
// begin() resets it before layer-major execution and end() synchronizes, returns
// the accumulated invalid top-k cell count, and deactivates it. Its device
// address remains backend-stable for CUDA graph replay and is freed at teardown.
GGML_BACKEND_API void    ggml_backend_cuda_glm_topk_gather_mask_begin(ggml_backend_t backend);
GGML_BACKEND_API int64_t ggml_backend_cuda_glm_topk_gather_mask_end  (ggml_backend_t backend);

// Thread-local override for the stream used by buffer set_tensor /
// get_tensor / memset_tensor. Normally those use cudaStreamPerThread
// (self-contained, synchronous). When stream capture is active on a
// different stream, those ops fail with cudaErrorInvalidArgument
// because they're on a non-capturing stream. Setting this override to
// the capturing stream routes the copies through it so they're part
// of the capture. Pass NULL to restore default behavior.
GGML_BACKEND_API void ggml_cuda_set_tensor_copy_stream(void * stream);

// Return the base address of the thread-local pinned staging arena used by
// buffer_set_tensor when the tensor-copy-stream override is active. Used by
// split-decode Phase 2d.C1: captured cudaGraph H2D copies replay from stable
// offsets inside this arena, so writing new values at (base + offset) before
// cudaGraphLaunch updates the pos/mask inputs seen by the replay.
// Returns NULL if the arena has not been allocated yet (first call to
// ggml_cuda_set_tensor_copy_stream with a non-null stream allocates 1 MiB).
// The base address is stable across reset cycles — reset() only zeros `tail`.
GGML_BACKEND_API void * ggml_cuda_get_tensor_copy_pinned_base(void);

// Return the arena offset (bytes from base) of the MOST-RECENT staging for
// `tensor_name` during the current capture window. Returns (size_t)-1 if not
// staged yet. The staging log is cleared on ggml_cuda_set_tensor_copy_stream.
// Used by split-decode to find the offset of inp_pos / inp_kq_mask after
// a layer capture.
GGML_BACKEND_API size_t ggml_cuda_get_last_staged_offset(const char * tensor_name);

// Variant that also returns the staged byte count via *out_size.
GGML_BACKEND_API size_t ggml_cuda_get_last_staged_offset_size(const char * tensor_name, size_t * out_size);

// ============================================================================
// Prefill streaming pipeline: dual-stream overlap (copy + compute)
//
// Allows uploading layer N+1 weights on a dedicated copy stream while the
// compute stream executes layer N.  Pinned (page-locked) staging buffers
// make cudaMemcpyAsync truly asynchronous w.r.t. the host.
// ============================================================================

// Create / destroy an opaque pipeline handle (copy stream + event, bound to
// the same device as `backend`).
GGML_BACKEND_API void * ggml_backend_cuda_prefill_pipeline_init(ggml_backend_t backend);
GGML_BACKEND_API void   ggml_backend_cuda_prefill_pipeline_free(void * pipeline);

// Allocate / free page-locked host memory for staging uploads.
GGML_BACKEND_API void * ggml_backend_cuda_prefill_pinned_alloc(size_t size);
GGML_BACKEND_API void   ggml_backend_cuda_prefill_pinned_free(void * ptr);

// Enqueue an H2D copy on the pipeline's copy stream (non-blocking on host).
GGML_BACKEND_API void   ggml_backend_cuda_prefill_upload_async(void * pipeline,
                                                                void * dst_device,
                                                                const void * src_pinned,
                                                                size_t size);

// Insert an event on the copy stream and make the compute stream wait for it.
// Call this in pre_layer_cb BEFORE the layer's kernels are enqueued.
GGML_BACKEND_API void   ggml_backend_cuda_prefill_wait_upload(void * pipeline);

// Host-blocking: wait for all pending copies on the copy stream to finish.
GGML_BACKEND_API void   ggml_backend_cuda_prefill_sync_copy_stream(void * pipeline);

// Raw device-to-device memcpy (layout-agnostic, no tensor metadata needed).
// Both dst and src must be valid GPU pointers on the current device.
GGML_BACKEND_API void   ggml_backend_cuda_memcpy_d2d(void * dst, const void * src, size_t nbytes);

// Raw device-to-host memcpy (GPU → CPU).
// dst must be a valid host pointer, src must be a valid GPU pointer.
GGML_BACKEND_API void   ggml_backend_cuda_memcpy_d2h(void * dst_host, const void * src_device, size_t nbytes);

// Raw host-to-device memcpy (CPU → GPU).
// dst must be a valid GPU pointer, src must be a valid host pointer.
GGML_BACKEND_API void   ggml_backend_cuda_memcpy_h2d(void * dst_device, const void * src_host, size_t nbytes);

// Async D2D memcpy on the given backend's compute stream (stream-ordered, doesn't block CPU).
// Both dst and src must be valid GPU pointers on the current device.
// stream_ptr should come from ggml_backend_cuda_get_stream(). Pass NULL for legacy stream 0.
GGML_BACKEND_API void   ggml_backend_cuda_memcpy_d2d_async(void * dst, const void * src, size_t nbytes, void * stream_ptr);

// Synchronize the given compute stream (blocks CPU until all pending work completes).
// stream_ptr should come from ggml_backend_cuda_get_stream(). Pass NULL for legacy stream 0.
GGML_BACKEND_API void   ggml_backend_cuda_sync_stream(void * stream_ptr);

// Inject a host callback into the CUDA stream. The GPU pauses the stream until
// the callback returns. Fires at the exact GPU execution point — after all
// prior kernels complete, before any subsequent ones start.
typedef void (*ggml_cuda_host_fn)(void * user_data);
GGML_BACKEND_API void   ggml_backend_cuda_launch_host_fn(void * stream_ptr, ggml_cuda_host_fn fn, void * user_data);

// Generate causal attention mask directly on GPU (replaces CPU fill + H2D copy).
// mask_gpu: device pointer to float[n_kv_dim * n_tokens]
// n_kv_dim: physical row stride / allocated width of the mask tensor
// n_kv_visible: logical KV positions visible to this ubatch
// pos_base: absolute position of the first token in this ubatch
GGML_BACKEND_API void   ggml_cuda_generate_causal_mask(void * mask_gpu, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base, void * stream_ptr);
GGML_BACKEND_API void   ggml_cuda_generate_causal_mask_f16(void * mask_gpu, int n_kv_dim, int n_kv_visible, int n_tokens, int pos_base, void * stream_ptr);

// Event-based orchestration: create/destroy a CUDA event, record it on a stream,
// and block the CPU until the event fires.  Provides per-operation completion
// signals without the heavy-handed cudaDeviceSynchronize().
GGML_BACKEND_API void * ggml_backend_cuda_event_create(void);
GGML_BACKEND_API void   ggml_backend_cuda_event_destroy(void * event_ptr);
GGML_BACKEND_API void   ggml_backend_cuda_event_record(void * event_ptr, void * stream_ptr);
GGML_BACKEND_API void   ggml_backend_cuda_event_synchronize(void * event_ptr);
// Make a CUDA stream wait for an event (GPU-side sync, no CPU block).
// The stream pauses at this point until the event is recorded+completed.
GGML_BACKEND_API void   ggml_backend_cuda_stream_wait_event(void * stream_ptr, void * event_ptr);

// Synchronize ALL CUDA streams on current device (wraps cudaDeviceSynchronize).
GGML_BACKEND_API void   ggml_backend_cuda_device_synchronize(void);

// Allocate GPU memory (wraps cudaMalloc). Returns nullptr on failure.
GGML_BACKEND_API void * ggml_backend_cuda_device_malloc(size_t nbytes);

// Free GPU memory (wraps cudaFree).
GGML_BACKEND_API void   ggml_backend_cuda_device_free(void * ptr);
// DS4COMBO fast switch: optional switch arena (see ggml-cuda.cu). Enable before
// the first device allocation; suspend releases all physical memory of
// ggml-owned device blocks and VMM pools while keeping every pointer; resume
// maps and zeroes the same ranges. Contents are not preserved.
GGML_BACKEND_API bool   ggml_backend_cuda_switch_arena_enable(int device, size_t va_bytes);
GGML_BACKEND_API bool   ggml_backend_cuda_switch_arena_suspend(size_t * released_bytes, int * released_blocks);
GGML_BACKEND_API bool   ggml_backend_cuda_switch_arena_resume(size_t * mapped_bytes, int * mapped_blocks);
GGML_BACKEND_API bool   ggml_backend_cuda_switch_arena_status(size_t * live_bytes, int * live_blocks, bool * is_suspended);

// R3 dual-use ping-pong (USER design): wrap an externally-owned CUDA pointer
// (a contiguous view of a D3D12 heap imported via cudaImportExternalMemory +
// cudaExternalMemoryGetMappedBuffer) as a first-class ggml CUDA buffer. The
// buffer does NOT own the pointer — its free_buffer will not cudaFree it; the
// D3D12 heap is the permanent owner. Lets DirectStorage write weights directly
// into the compute buffer with no staging/D2D. Mirrors ggml_backend_cpu_buffer_from_ptr.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_buffer_from_ptr(int device, void * ptr, size_t size);

// Phase 1.3 (2026-05-11): on-demand dump of the CUDA per-op profile counters
// (collected when LLAMA_CUDA_OP_PROFILE=1 is set at server start). Windows
// graceful-shutdown does not reliably fire the std::atexit-registered dump,
// so call this directly (e.g. from decode_streaming on prefill completion
// when LLAMA_CUDA_OP_PROFILE_DUMP_EACH=1).
GGML_BACKEND_API void   ggml_cuda_op_profile_dump_now(void);

// Low-overhead, capture-safe CUDA op-family sampling for the layer-major
// remainder probe. Only LLAMA_DLM_REMAINDER_TIMERS=2 enables CUDA events;
// mode 1 is CPU-only. reset() starts a new chunk epoch. dump_now() never
// synchronizes, reports only completed samples, then destroys the event ring
// before the graph/stream can be torn down at the chunk boundary.
GGML_BACKEND_API void   ggml_cuda_dlm_remainder_profile_reset(void);
GGML_BACKEND_API void   ggml_cuda_dlm_remainder_profile_dump_now(void);

#ifdef  __cplusplus
}
#endif
