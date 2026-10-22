// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

// ============================================================================
// Split-decode Phase 3: MTLSharedEvent-based event API for event-gated
// single-graph decode on Mac. Mirrors the Windows-side CUDA event API.
// See docs/split_decode_phase3_metal_design.md for architecture details.
// ============================================================================

typedef struct ggml_metal_event * ggml_metal_event_t;

// Create a new MTLSharedEvent bound to the given Metal backend's device.
// Returns NULL if backend is not Metal or allocation fails.
GGML_BACKEND_API ggml_metal_event_t ggml_backend_metal_event_init(ggml_backend_t backend);

// Free an event created via ggml_backend_metal_event_init.
GGML_BACKEND_API void ggml_backend_metal_event_free(ggml_backend_t backend, ggml_metal_event_t ev);

// Host-side signal: atomically increment the event's expected value and
// set the underlying MTLSharedEvent's signaledValue. Unblocks any GPU
// encodeWaitForEvent or host wait waiting on this event.
// Used by the network RX thread to release GPU work waiting for input.
GGML_BACKEND_API void ggml_backend_metal_event_signal_host(ggml_backend_t backend, ggml_metal_event_t ev);

// Host-side wait: spin-wait until the underlying MTLSharedEvent's
// signaledValue reaches or exceeds the current expected value. Used by
// the TX thread to wait for per-layer compute output to land.
GGML_BACKEND_API void ggml_backend_metal_event_wait_host(ggml_backend_t backend, ggml_metal_event_t ev);

// 2026-04-20 (split-decode mega gate fix): race-free wait/signal pair
// where the caller provides the target value (usually token_idx + 1).
// Both encoder and signaler derive the same target from the same input,
// so there's no shared counter that both sides race to fetch_add. The
// plain signal_host/wait_host variants use an atomic counter that can
// be read by the signaler BEFORE the encoder advances it, causing GPU
// hangs or premature wakeups; use these _value variants whenever the
// encoder thread and signaler thread are independent and may interleave.
GGML_BACKEND_API void ggml_backend_metal_event_signal_host_value(ggml_backend_t backend, ggml_metal_event_t ev, uint64_t target);
GGML_BACKEND_API void ggml_backend_metal_event_wait_host_value  (ggml_backend_t backend, ggml_metal_event_t ev, uint64_t target);

// GPU-side signal: submit a command buffer that encodes signalEvent on
// the Metal queue. The cmd_buf executes AFTER any previously-submitted
// work on the queue (including the graph_compute cmd_buf), so this
// event's signaledValue transitions after all prior GPU work is done.
// Equivalent to cudaEventRecord on the default stream. Use this instead
// of ggml_backend_synchronize to enable per-layer non-blocking sync in
// Phase 3 split-decode: the host then calls event_wait_host to block
// only until GPU finishes the relevant layer, without a full-queue sync.
GGML_BACKEND_API void ggml_backend_metal_event_record(ggml_backend_t backend, ggml_metal_event_t ev);

// GPU-side wait: submit a command buffer that encodes waitForEvent on
// the Metal queue. Subsequently-submitted cmd_bufs do not execute until
// ev's signaledValue reaches the encoded value. The wait value captured
// is the event's current expected value — the caller typically bumps it
// via event_signal_host (from network RX thread) or a matching cmd_buf's
// encodeSignalEvent. Equivalent to cudaStreamWaitEvent. Used by the
// Phase 3 Option B single-dispatch path to gate per-layer compute on
// input-data arrival without the CPU-GPU barrier cost.
GGML_BACKEND_API void ggml_backend_metal_event_submit_wait(ggml_backend_t backend, ggml_metal_event_t ev);

// Enqueue a blit copy from src tensor to dst tensor on the Metal queue.
// Both tensors must have ->buffer set to Metal backend buffers and
// identical byte size. Runs after any previously-submitted work on the
// queue (including graph_compute). Does NOT wait. Used by Phase 3
// Option B to snapshot a layer's compute-buffer output into a stable
// per-layer shared MTLBuffer so the next layer's compute can reuse the
// compute buffer without clobbering the previous layer's result.
GGML_BACKEND_API bool ggml_backend_metal_cpy_tensor_async(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst);

// Path A Phase D.2 (2026-04-20) — segmented graph_compute with per-segment
// event gates. Serially encodes each segment's node range into its own
// cmd_buf, optionally preceded by encodeWaitForEvent(wait_event) and
// followed by encodeSignalEvent(signal_event). Commits all segments to
// the same Metal queue preserving FIFO execution order. Used by
// split-decode mega-graph dispatch to inject per-layer event gates —
// the Mac analog of CUDA's cudaStreamWaitEvent/cudaEventRecord chain
// around cudaGraphLaunch per layer.
//
// - segments: array of n_segments entries, ordered by start_idx, non-
//   overlapping; together they should cover all nodes the caller wants
//   computed (unspecified ranges are skipped).
// - For pure-Metal graphs only; no CPU-backend splits. Caller is
//   responsible for calling ggml_backend_sched_alloc_graph beforehand.
struct ggml_metal_gated_segment {
    int start_idx;                     // first node index (inclusive)
    int end_idx;                       // one past last node index (exclusive)
    ggml_metal_event_t wait_event;     // non-null: encodeWaitForEvent before compute
    ggml_metal_event_t signal_event;   // non-null: encodeSignalEvent after compute
    // 2026-04-20 (race fix): if non-zero, encode wait/signal with these
    // explicit target values rather than the shared ev->value fetch_add
    // counter. MUST be used when wait_event is host-signaled (via
    // event_signal_host_value with the same target) AND the encoder runs
    // on a different thread from the signaler. Caller guarantees the
    // target strictly increases across successive calls on the same event.
    uint64_t wait_target;
    uint64_t signal_target;
    // 2026-04-21 (iter 9 / inp_out_ids fix): if fill_buf != NULL, issue a
    // Metal blit fillBuffer:range:value:0 for fill_len bytes at fill_offs in
    // fill_buf BEFORE this segment's compute encoder. Used to reset the
    // inp_out_ids slot (offset 0 of the compute buffer) to int32=0 before
    // the last-layer GET_ROWS that uses it — gallocr aliases intermediate
    // float tensors to that slot across layers 0-76, leaving garbage there
    // by the time segment 77 runs.
    void * fill_buf;   // id<MTLBuffer> cast to void*; NULL = no fill
    size_t fill_offs;  // byte offset within fill_buf
    size_t fill_len;   // number of bytes to fill (typically 4 for one I32)
};

GGML_BACKEND_API bool ggml_backend_metal_graph_compute_gated(
        ggml_backend_t backend,
        struct ggml_cgraph * gf,
        int n_segments,
        const struct ggml_metal_gated_segment * segments);

// 2026-04-21 (iter 9): return the id<MTLBuffer> (as void*) and byte offset
// within that buffer for a tensor that lives in a Metal backend buffer.
// Used by llama-context.cpp to populate fill_buf/fill_offs for gated-segment
// inp_out_ids fix. Returns NULL/0 if tensor is not in a Metal buffer.
GGML_BACKEND_API void * ggml_backend_metal_tensor_get_mtl_buf(
        struct ggml_tensor * tensor,
        size_t             * out_offs);

#ifdef __cplusplus
}
#endif
