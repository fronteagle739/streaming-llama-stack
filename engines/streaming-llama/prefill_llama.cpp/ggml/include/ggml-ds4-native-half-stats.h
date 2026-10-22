#pragma once
#include "ggml-backend.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct ggml_ds4_native_half_stats { uint64_t queued[3],completed[3],completed_graphs,graph_active; };
GGML_BACKEND_API bool ggml_backend_cuda_ds4_native_half_begin(struct ggml_ds4_native_half_stats *);
GGML_BACKEND_API bool ggml_backend_cuda_ds4_native_half_complete(const struct ggml_ds4_native_half_stats *,struct ggml_ds4_native_half_stats *);
GGML_BACKEND_API void ggml_backend_cuda_ds4_native_half_snapshot(struct ggml_ds4_native_half_stats *);
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_rhs_pack_queued(void);
#ifdef __cplusplus
}
static_assert(sizeof(ggml_ds4_native_half_stats)==64,"fixed C/NVCC/MSVC stats ABI");
#endif
