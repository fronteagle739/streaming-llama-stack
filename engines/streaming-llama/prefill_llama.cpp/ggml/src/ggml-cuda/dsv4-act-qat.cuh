#pragma once

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_act_qat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
