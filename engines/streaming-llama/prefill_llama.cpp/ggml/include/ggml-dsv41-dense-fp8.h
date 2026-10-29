// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-dsv41-dense-bf16-store.h"
#include <cstdlib>
#include <cstring>
inline bool ggml_dsv41_dense_fp8_requested(){const char *v=std::getenv("LLAMA_DSV41_DENSE_FP8");return v&&!std::strcmp(v,"1");}
constexpr int32_t ggml_dsv41_dense_fp8_magic=0x46384431;
inline bool ggml_dsv41_dense_fp8_native_weight(const ggml_tensor *w){
    return w&&w->op==GGML_OP_NONE&&w->type==GGML_TYPE_Q8_0&&w->op_params[14]==ggml_dsv41_dense_fp8_magic&&w->op_params[15]==1;
}
inline bool ggml_dsv41_dense_fp8_eligible(const ggml_tensor *t){
    if(!ggml_dsv41_dense_bf16_store_requested(t))return false;
    const auto n=t->src[0]->ne[1],k=t->src[0]->ne[0];
    return (n==1280&&k==5120)||(n==32768&&k==1280)||(n==512&&k==5120)||(n==128&&k==512)||
           (n==5120&&k==8192)||(n==2304&&k==5120)||(n==5120&&k==2304);
}
extern "C" GGML_API uint64_t ggml_backend_cuda_dsv41_dense_fp8_queued(void);
