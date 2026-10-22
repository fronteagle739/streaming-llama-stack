#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_topk_radix_queued(void);
#ifdef __cplusplus
}
#endif
