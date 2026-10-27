/* DirectStorage-backed parallel expert reads for the CUDA SSD-streaming path (Windows only).
 *
 * WHY THIS EXISTS
 * ---------------
 * The streaming path reads every missed expert through cuda_pread_full(), a blocking pread loop,
 * one expert at a time -- and three reads per expert (gate/up/down). ds4_win32_pread() issues
 * ReadFile with an OVERLAPPED used ONLY to carry the offset, on a handle opened WITHOUT
 * FILE_FLAG_OVERLAPPED; Windows serialises I/O on such a file object, so a thread pool over that
 * one handle cannot raise queue depth at all.
 *
 * Measured on the RTX box (5090, DSV4-Flash 0731 streaming from a 24 GB/s RAID array):
 *     disk duty cycle 93.8%, mean 1.80 GB/s, peak 3.50 GB/s  ==  7.5% of the array
 * Busy essentially all the time, yet delivering 7.5% -- the signature of queue depth 1, not of idle
 * gaps and not of small requests (experts are 6.75 and 13.50 MiB).
 *
 * At each MoE layer the router picks top-k experts simultaneously, so every miss for that layer is
 * known before any read is issued. They should go out as ONE batch.
 *
 * DESIGN
 * ------
 * The production path starts with a null-D3D12-device queue and MEMORY destinations. The optional
 * GPU proof gate creates a second, device-bound queue plus one shareable committed buffer imported
 * into CUDA. A committed resource avoids the placed-resource 4 GB offset trap, and EnqueueSetEvent
 * gives a CPU completion point before CUDA reads the buffer.
 *
 * dstorage.dll is loaded at runtime from the executable's own directory via LoadLibraryW +
 * GetProcAddress, so a missing DLL produces OUR precise diagnostic at streaming init rather than a
 * loader failure at process start -- and cpu/non-streaming runs never need it.
 *
 * NO SILENT FALLBACK: every entry point reports failure explicitly. Callers must decide loudly.
 */
#ifndef DS4_DSTORAGE_H
#define DS4_DSTORAGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up the factory/queue for the model file behind `fd`. Returns 1 on success, 0 on failure
 * (with a detailed reason already printed to stderr, including the HRESULT). Safe to call again;
 * re-initialises if `fd` refers to a different file, which happens when the MTP model swaps in. */
int ds4_ds_init_from_fd(int fd);

/* 1 once a queue is live. */
int ds4_ds_available(void);

/* Begin a new batch. */
void ds4_ds_batch_reset(void);

/* Add one read to the pending batch. `dst_host` must stay valid until submit returns.
 * Returns 1 on success, 0 if the batch is full or the arguments are invalid. */
int ds4_ds_batch_add(uint64_t file_offset, uint32_t bytes, void *dst_host);

/* Development proof gate for DirectStorage BUFFER destinations. The arena is a shareable D3D12
 * committed resource imported into CUDA; `out_cuda_ptr` is its CUDA device address. GPU requests
 * in one batch must not be mixed with MEMORY requests. */
int ds4_ds_gpu_arena_create(uint64_t bytes, void **out_cuda_ptr);
/* Non-zero while the GPU-destination arena resource exists. Any CUDA-side pointer obtained from
 * ds4_ds_gpu_arena_create() is DANGLING once this returns 0 - see the note at its definition. */
int ds4_ds_gpu_arena_alive(void);
void ds4_ds_gpu_arena_release(void);
int ds4_ds_batch_add_gpu(uint64_t file_offset, uint32_t bytes, uint64_t arena_offset);
/* Which of ds4_ds_batch_add_gpu()'s seven refusal branches fired last; 0 = the last call succeeded.
 * 1 !init 2 !gpu_arena 3 bytes==0 4 !queue_count 5 offset/size out of arena 6 batch full
 * 7 GPU add into an already non-GPU batch. */
int ds4_ds_gpu_add_fail_reason(void);

/* R4 prefill pipeline. Requests are built once per model/layer, then replayed
 * into the shared GPU arena. Completion is exported as a D3D12 fence and
 * consumed by CUDA as an external semaphore, so the host never waits for the
 * upload before compute is queued. */
typedef struct ds4_ds_wave_span {
    uint64_t file_offset;
    uint32_t bytes;
    uint32_t arena_slot;
    uint64_t arena_offset;
} ds4_ds_wave_span;
int ds4_ds_prefill_arenas_create(const uint64_t *bytes,
                                 uint32_t count,
                                 void **out_cuda_ptrs);
void ds4_ds_prefill_arenas_release(void);
int ds4_ds_prefill_waves_create(uint32_t count);
int ds4_ds_prefill_wave_build(uint32_t index,
                              const ds4_ds_wave_span *spans,
                              uint32_t count);
int ds4_ds_prefill_wave_submit(uint32_t index, uint64_t *out_fence_value);
int ds4_ds_prefill_wave_cuda_wait(uint64_t fence_value, void *cuda_stream);
int ds4_ds_prefill_waves_check_errors(void);
void ds4_ds_prefill_waves_release(void);

/* How many requests are pending. */
uint32_t ds4_ds_batch_count(void);
/* Pending request bytes and the per-submit request ceiling. */
uint64_t ds4_ds_batch_bytes(void);
uint32_t ds4_ds_batch_capacity(void);

/* Cheap phase/token context for transport bandwidth and slow-batch reporting.
 * set_batch_layer snapshots it onto the next split or synchronous batch. */
void ds4_ds_stats_set_phase(int decode);
void ds4_ds_stats_set_token(uint64_t token_index);
void ds4_ds_stats_set_batch_layer(uint32_t first_layer, uint32_t last_layer);

/* Submit every pending request, wait for completion, and verify no request failed.
 * Returns 1 only if ALL requests completed successfully. All out-parameters are optional;
 * `out_ms` spans submission through observation while `out_wait_ms` measures only the blocking wait.
 *
 * The error check is the one lesson taken verbatim from the weight-fetcher: a completion event can
 * fire even when a request FAILED, so the queue's error record must be examined before any consumer
 * is allowed to touch the destination memory. */
int ds4_ds_batch_submit_wait(uint64_t *out_bytes, double *out_ms, double *out_wait_ms);

/* Split submit/wait API. ds4_ds_batch_submit_wait() above remains exactly
 * submit(); wait() for legacy callers. The transport owns an eight-slot FIFO
 * ring; batch_wait drains the oldest submitted slot. */
int ds4_ds_batch_submit(void);
uint32_t ds4_ds_batch_inflight(void);
uint32_t ds4_ds_pipeline_capacity(void);
/* 1 = oldest submitted batch is complete, 0 = it is still in flight,
 * -1 = no submitted batch. This does not consume the completion. */
int ds4_ds_batch_oldest_ready(void);
int ds4_ds_batch_wait(uint64_t *out_bytes, double *out_ms, double *out_wait_ms);

/* Self-test: read `span_bytes` of the model twice -- once as sequential single-request submits
 * (queue depth 1) and once in batches of `batch` -- comparing both against pread for byte equality
 * and reporting both throughputs. This proves whether queue depth actually buys anything on this
 * array BEFORE the decode path depends on it. Returns 1 if both transports matched pread. */
int ds4_ds_selftest(int fd, uint64_t start_offset, uint64_t span_bytes, uint32_t chunk_bytes,
                    uint32_t batch);

void ds4_ds_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_DSTORAGE_H */
