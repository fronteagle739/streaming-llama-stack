#pragma once

#include "common.cuh"

void ggml_cuda_op_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// V6: routes through the patched dense MMA kernel with indices-gather loads.
// Called when LLAMA_MLA_SPARSE_FA_V6=1.
void ggml_cuda_op_flash_attn_ext_sparse_v6(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const int32_t * prepared_indices,
        const char * prepared_mask);
