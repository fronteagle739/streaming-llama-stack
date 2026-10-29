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
