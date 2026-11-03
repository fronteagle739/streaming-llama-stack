// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#if defined(_WIN32) && defined(QFN_NVFP4_BUILD)
#define QFN_ATTN_API __declspec(dllexport)
#elif defined(_WIN32)
#define QFN_ATTN_API __declspec(dllimport)
#else
#define QFN_ATTN_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
enum qfn_basalt_attn_mode { QFN_ATTN_PROMPT=0,QFN_ATTN_DECODE64=1 };
// Native contract: D256, F32Q/optionalgate/output, F16KV, H=12*Hkv.
// All strides are bytes; D is contiguous. Read-only input axes may broadcast.
// Output head/token spans must be disjoint. Selected IDs retain order/repeats;
// Output and scratch never alias any input or each other. Caller keeps every
// device allocation alive in the prepared CUDA context until replay completes;
// descriptor metadata is copied into launch parameters and need not persist.
// each ID is independently masked unless 0<=ID<n_kv and ID<=position[query].
// cap0 (or n_kv0) writes zero before optional sigmoid gate. No cache conversion.
#pragma pack(push,8)
typedef struct qfn_basalt_attn_desc {
    int32_t mode,nt,n_heads,n_kv_heads,n_kv,cap;
    float scale;
    uint64_t q_head_stride,q_token_stride;
    uint64_t gate_head_stride,gate_token_stride;
    uint64_t k_head_stride,k_cell_stride,v_head_stride,v_cell_stride;
    uint64_t ids_token_stride,out_head_stride,out_token_stride;
} qfn_basalt_attn_desc;
#pragma pack(pop)
QFN_ATTN_API int qfn_basalt_attn_abi(void);
// All module/attribute/PDL initialization happens before capture; the CUDA context
// and buffers survive replay. Caller inputs must be ready on this CUDA stream.
QFN_ATTN_API int qfn_basalt_attn_prepare(void * stream);
// Pure host query: UINT64_MAX invalid; prompt/cap0 require0 bytes. Decode scratch
// is caller-owned,256-byte aligned, disjoint, persistent, no allocation in run.
QFN_ATTN_API uint64_t qfn_basalt_attn_workspace_bytes(const qfn_basalt_attn_desc * desc);
QFN_ATTN_API int qfn_basalt_attn_run(const float * q,const float * gate,
    const void * k,const void * v,const int32_t * ids,const int32_t * position,
    float * out,const qfn_basalt_attn_desc * desc,void * workspace,
    uint64_t workspace_bytes,void * stream);
#ifdef __cplusplus
}
#endif
