// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
// Q8_0 -> E4M3/E8M0 speed proxy: weights use one exponent per32x32 tile;
// activations use one exponent per32 values. Caller owns512MiB aligned arena.
// Native source format1: E4M3 [N,K], then E8M0 [N/32,K/32], lossless values.
using slstack_dense_fp8_fn=int (*)(const void*,int,const float*,float*,int,int,int,int64_t,void*,uint64_t,cudaStream_t);
