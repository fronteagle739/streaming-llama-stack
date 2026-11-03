// SPDX-License-Identifier: MIT
#pragma once
#include "qfn_basalt_nvfp4.h"

#ifdef __cplusplus
extern "C" {
#endif

enum qfn_basalt_qsa_key_type { QFN_QSA_F16 = 0, QFN_QSA_F32 = 1 };
enum qfn_basalt_qsa_mode { QFN_QSA_TC = 0, QFN_QSA_WARP_ORACLE = 1 };

QFN_NVFP4_API int qfn_basalt_qsa_abi(void); // 1, independent of NVFP4 ABI
QFN_NVFP4_API int qfn_basalt_qsa_prepare(void * stream);
// Fixed D=128,H=4. Full matrix (all pools, no causal pruning), F32 output.
// Logical shapes: q[stream,token,head,D], k[stream,pool,D], w[stream,token,head],
// out[stream,token,pool]. D/head-weight/output-pool elements are contiguous.
// All strides are bytes. INPUTS support zero/broadcast and transposed outer strides.
// Distinct OUTPUT rows/streams must not overlap; broadcast output writes are invalid.
// Admission rejects zero/short output strides on repeated axes; caller guarantees
// cross-axis non-aliasing (the GGML bridge allocates a separate contiguous output).
// Mode0 adapts Basalt3xTF32; mode1 retains current QFN scalar-warp arithmetic for parity.
// Score = scale_heads * sum_h relu(scale_embd * dot(q_h,k_pool)) * w_h.
// No allocation, scratch, CPU readback, synchronization, selection, tail or cache policy.
// prepare must succeed before capture/run. Caller spans are valid/disjoint and persist
// through replay; context replacement requires DLL unload/reload then prepare.
// Returns CUDA status; nt<=1048560, n_stream<=65535; n_pool positive I32.
QFN_NVFP4_API int qfn_basalt_qsa_run(
    const float * q, const void * k, const float * w, float * out,
    int nt, int n_pool, int n_stream, int key_type,
    uint64_t q_head_stride, uint64_t q_token_stride, uint64_t q_stream_stride,
    uint64_t k_pool_stride, uint64_t k_stream_stride,
    uint64_t w_token_stride, uint64_t w_stream_stride,
    uint64_t out_token_stride, uint64_t out_stream_stride,
    float scale_embd, float scale_heads, void * stream, int mode);

#ifdef __cplusplus
}
#endif
