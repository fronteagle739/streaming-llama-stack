#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

// Path A Phase D.2 (2026-04-20) — segmented graph_compute with per-segment
// event gates. Serially encodes each segment's node range into its own
// cmd_buf, optionally preceded by encodeWaitForEvent(wait_event) and
// followed by encodeSignalEvent(signal_event). All segments are committed
// to the same Metal queue, preserving FIFO execution order. Used by
// split-decode mega-graph dispatch to inject per-layer event waits/signals
// into a single graph_compute call — the Mac analog of CUDA's
// cudaStreamWaitEvent/cudaEventRecord chain around cudaGraphLaunch per layer.
//
// Caller must:
//   - segments ordered by increasing start_idx, non-overlapping
//   - together they should cover the full node range the caller wants
//     computed (unspecified ranges are skipped)
//   - per-segment wait_event is signaled externally (RX thread
//     event_signal_host) before GPU advances past it
//   - per-segment signal_event is checked by external TX thread via
//     event_wait_host (or a subsequent cmd_buf's encodeWaitForEvent)
//
// Bypasses the parallel n_cb encode_async machinery — serial encoding for
// event-injection correctness. For pure-Metal graphs; no CPU-backend splits.
// (struct ggml_metal_gated_segment declared in ggml-metal.h — public API header.)
struct ggml_metal_gated_segment;  // forward decl

enum ggml_status ggml_metal_graph_compute_gated(
    ggml_metal_t ctx,
    struct ggml_cgraph * gf,
    int n_segments,
    const struct ggml_metal_gated_segment * segments);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

// Accessor for the underlying device handle — used by Phase 3 public
// ggml_backend_metal_event_init wrapper.
ggml_metal_device_t ggml_metal_get_device(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
