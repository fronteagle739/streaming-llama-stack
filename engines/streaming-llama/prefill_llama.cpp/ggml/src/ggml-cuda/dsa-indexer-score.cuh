#pragma once

#include "common.cuh"

void ggml_cuda_op_dsa_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsa_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// Typed DS4 TEST wrapper, fixed256-row/query slabs and original masks.
void ggml_cuda_op_dsv41_index_score_cublas(ggml_backend_cuda_context & ctx,ggml_tensor *dst);
