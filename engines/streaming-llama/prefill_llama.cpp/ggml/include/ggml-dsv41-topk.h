#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_topk_radix_queued(void);
// LLAMADECODE fast exact top-512 select (default off; producer --topk-kernel fast).
GGML_BACKEND_API void ggml_backend_cuda_dsv41_topk_fast_set(int on);
GGML_BACKEND_API int ggml_backend_cuda_dsv41_topk_fast_enabled(void);
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_topk_fast_queued(void);
// Rows routed to the unchanged current pipeline (NaN/subnormal); synchronous read.
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_topk_fast_dirty_rows(void);
#ifdef __cplusplus
}
#endif
