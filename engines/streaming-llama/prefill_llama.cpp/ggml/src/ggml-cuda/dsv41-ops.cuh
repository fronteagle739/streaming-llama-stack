#pragma once
#include "common.cuh"

void ggml_cuda_op_dsv41_engram_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_pool2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_raw_commit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_index_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_candidate_blocks(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_selected_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_masked_half(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
