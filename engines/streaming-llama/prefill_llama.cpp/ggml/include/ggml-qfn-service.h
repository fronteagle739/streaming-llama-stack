#pragma once

#include "ggml-backend.h"
#include <stdint.h>

// Stable mapped records for ONE captured decode epoch. Limits match the
// existing slot-map's <=64 selected IDs, without truncating a supported batch.
#define GGML_QFN_SERVICE_MAX 64
#define GGML_QFN_SERVICE_MAGIC UINT64_C(0x51464e5352563031)

struct ggml_qfn_service_copy {
    uint64_t source, destination, bytes;
};
struct ggml_qfn_service_table {
    uint64_t index;
    int32_t slot;
    int32_t reserved;
};
struct ggml_qfn_service_control {
    volatile int32_t aborted;
    volatile int32_t error;
    uint64_t first_ticket, last_ticket;
};
struct ggml_qfn_service_record {
    uint64_t magic, ticket;
    uint64_t table, bell, control;
    int32_t request, layer;
    volatile int32_t active;
    uint32_t demotions, fills, updates;
    uint64_t table_entries;
    ggml_qfn_service_copy demote[GGML_QFN_SERVICE_MAX];
    ggml_qfn_service_copy fill[GGML_QFN_SERVICE_MAX];
    ggml_qfn_service_table update[2 * GGML_QFN_SERVICE_MAX];
};

#ifdef __cplusplus
static_assert(sizeof(ggml_qfn_service_copy) == 24, "service copy ABI");
static_assert(sizeof(ggml_qfn_service_table) == 16, "service table ABI");
extern "C" {
#endif
// Warm the three kernels before any external wait; return their addresses for
// explicit CUDA graph nodes. No model arithmetic is executed by preparation.
GGML_BACKEND_API int ggml_backend_cuda_qfn_service_prepare(void ** demote, void ** fill, void ** publish);
#ifdef __cplusplus
}
#endif
