#pragma once

#include "common.cuh"

// MLA-aware Flash Attention kernel (ik_llama.cpp port, iter 15).
// Only dispatched when env LLAMA_FA_NEW_MMA=1.
void ggml_cuda_flash_attn_ext_mma_new_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
