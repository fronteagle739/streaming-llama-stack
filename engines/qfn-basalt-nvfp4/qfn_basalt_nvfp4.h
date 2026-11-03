// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>

#if defined(_WIN32) && defined(QFN_NVFP4_BUILD)
#define QFN_NVFP4_API __declspec(dllexport)
#elif defined(_WIN32)
#define QFN_NVFP4_API __declspec(dllimport)
#else
#define QFN_NVFP4_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum qfn_basalt_nvfp4_mode {
    QFN_NVFP4_AUTO = 0,
    QFN_NVFP4_W4A4 = 1,
    QFN_NVFP4_NATIVE_Q8_1 = 2,
    QFN_NVFP4_MMQ_Q8 = 3,
    QFN_NVFP4_CALIBRATED_A4 = 4,
    QFN_NVFP4_CALIBRATED_DECODE = 5
};

QFN_NVFP4_API int qfn_basalt_nvfp4_abi(void);
// Initializes CUDA modules/attributes with zero routes, before any graph capture.
// A zero activation row warms only the quantizer; no model weights are accessed.
QFN_NVFP4_API int qfn_basalt_nvfp4_prepare(void * stream);
// Zero means invalid dimensions/mode or arithmetic overflow. Workspace is 256-byte aligned.
QFN_NVFP4_API uint64_t qfn_basalt_nvfp4_workspace_bytes(
    int nt, int top_k, uint32_t physical_slots, int mode);
// Fixed geometry E=2560, FF=640, K=10, blob_stride=2764816 including all FP32 tails.
// x[t*E+d], ids[t*K+k] (physical blob index), out[(t*K+k)*E+d] (UNWEIGHTED).
// AUTO chooses W4A4 at nt>=1024 (512-slot bank); otherwise native q8_1 (arbitrary slot domain).
// W4A4 requires distinct IDs within each token; native q8_1 accepts repeated physical slots.
// Explicit mode3 uses published MMQ-D4/F32 q8 on a512-slot bank, distinct IDs per token.
// Mode3 reserves256SM fixup workspace before prepare; larger-device mode3 fails loudly
// while modes0/1/2 availability is preserved. Its final short window retains MMQ.
// Forced modes are parity/measurement variants. W4A4 subwindows keep the caller's phase.
// W4A4 workspace covers min(nt,16384) tokens; wider invocations retain W4A4 in all subwindows.
// Native nt<=85899345: upstream SwiGLU grid.x=25*nt must fit CUDA's INT_MAX block bound.
// run allocates nothing, reads no device data on the host, and never synchronizes.
// Returns CUDA status. Invalid device IDs/tails trap asynchronously, surfaced by caller sync.
// prepare must succeed first; all input/output/workspace addresses must survive graph replay.
// Caller spans must be disjoint. The current CUDA context must remain alive after prepare;
// cudaDeviceReset/context replacement requires unloading/reloading this DLL before prepare.
QFN_NVFP4_API int qfn_basalt_nvfp4_run(
    const float * x, const int32_t * ids, const uint8_t * raw,
    uint64_t blob_stride, uint32_t physical_slots, int nt, int top_k,
    float * out, void * workspace, uint64_t workspace_bytes, void * stream, int mode);

// Separate calibrated capability probe; does not change NVFP4 ABI1 or modes0..3.
QFN_NVFP4_API int qfn_basalt_nvfp4_calibrated_abi(void);
// Query workspace through workspace_bytes(nt,10,512,QFN_NVFP4_CALIBRATED_A4).
// The old run() rejects mode4 because it has no calibration pointer.
// Current-layer input_scale[expert*3+role], role0gate/1up/2down, raw exported
// g=calibrated_amax/(6*448), NOT its inverse. Exactly1536 contiguous F32 values,
// aligned4, finitepositive, disjoint and persistent through graph replay.
// Logical512 bank, unique IDs per token, allT>0 inclfinal short16Kwindow.
// Separate gate/up globals are fully consumed; H does not contain s_down until
// after the down product. Same unweighted output/input/ownership contracts above.
QFN_NVFP4_API int qfn_basalt_nvfp4_run_calibrated(
    const float * x, const int32_t * ids, const uint8_t * raw, const float * input_scale,
    uint64_t blob_stride, uint32_t physical_slots, int nt, int top_k,
    float * out, void * workspace, uint64_t workspace_bytes, void * stream);


// Separate slot-aware calibrated decode capability. Old ABI1/modes0..4 are unchanged.
// One to four target rows (ordinary decode and existing MTP verification capacity), K10.
// physical_ids index the arbitrary cache slab; logical_ids index input_scale[512][3].
// All selected payload bytes, including16-byte tails, are gathered in the queried
// persistent workspace on this stream before static arithmetic. No cache/policy writes,
// allocations, host reads, synchronization or worker CUDA calls occur during run/replay.
QFN_NVFP4_API int qfn_basalt_nvfp4_calibrated_decode_abi(void);
QFN_NVFP4_API int qfn_basalt_nvfp4_run_calibrated_decode(
    const float * x,const int32_t * physical_ids,const int32_t * logical_ids,
    const uint8_t * raw,const float * input_scale,uint64_t blob_stride,uint32_t physical_slots,
    int nt,int top_k,float * out,void * workspace,uint64_t workspace_bytes,void * stream);

#ifdef __cplusplus
}
#endif
