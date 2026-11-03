#pragma once
// QFNSTREAM: Qwen sparse attention (QSA) selection for Qwen3.8 Flash Next. See ggml_qfn_qsa_select in ggml.h.
#include "common.cuh"

void ggml_cuda_op_qfn_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qfn_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qfn_slot_map(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qfn_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qfn_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
