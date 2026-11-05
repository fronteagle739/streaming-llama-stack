// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "native_ffn_workspace.h"

// Independent experimental ABI. The canonical ABI4 adapter remains unchanged.
// Weights: all adjacent-nibble E2M1 bytes [384,N,K/2], then E8M0 bytes
// [384,K/128,N,4]. No conversion, allocation or weight copy inside this API.
// Routes: stable expert-sorted scatter indices (token*6+slot), plus 385 bounds.
// Input/output: contiguous F32 [tokens,5120]; route weights F32 [tokens,6].
// The only routed intermediate stores are BF16 and MXFP8, in caller workspace.
using slstack_fi_native_ffn_fn=int (*)(const void*,const void*,const void*,const float*,const float*,
    const int32_t*,const int32_t*,float*,int,float,void*,uint64_t,cudaStream_t);
// Optional ABI2 epilogue: no change to ABI1's unsteered packed GEMMs.
// A=[384,rank,2304], vectors=[rank,5121] (strength then unit vector),
// caller-owned dots=[tokens*6,rank]. Original down output retains its native
// BF16 store; the F32 writer delta is added per expert before the usual sum.
struct slstack_fi_writer_delta {uint32_t version,rank;const float*a,*vectors;float*dots;};
using slstack_fi_native_ffn_rank_fn=int (*)(const void*,const void*,const void*,const float*,const float*,
    const int32_t*,const int32_t*,float*,int,float,void*,uint64_t,cudaStream_t,const slstack_fi_writer_delta*);
// Packed resident append: each of weight_slots cache slots holds [N,K/2] values
// then [K/128,N,4] scales. Bounds still have 385 entries; slots at or above
// weight_slots must be empty, and every weight allocation must span the pool.
using slstack_fi_native_ffn_slots_fn=int (*)(const void*,const void*,const void*,const float*,const float*,
    const int32_t*,const int32_t*,float*,int,float,void*,uint64_t,cudaStream_t,const slstack_fi_writer_delta*,int);
