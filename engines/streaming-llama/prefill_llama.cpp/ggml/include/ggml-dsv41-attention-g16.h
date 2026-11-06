#pragma once
#include "ggml-backend.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_token_tile_g16_queued(void);
#ifdef __cplusplus
}
#endif
