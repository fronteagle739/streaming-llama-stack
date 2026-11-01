/* Device side of the expert doorbell (see ds4_expert_doorbell.h). Self-contained:
 * include from any .cu file. `sh` is the device alias of the mapped host block. */
#ifndef DS4_EXPERT_DOORBELL_CUH
#define DS4_EXPERT_DOORBELL_CUH
#include "ds4_expert_doorbell.h"

/* One warp. ids: n routed expert ids of one decode token (distinct). Writes the
 * hit slots to slots_dev, posts the record, rings the bell on all hits. */
__global__ static void ds4db_resolve_kernel(const int32_t *ids, uint32_t n, uint32_t layer, uint32_t seq,
                                            ds4db_shared *sh, int32_t *slots_dev) {
    const uint32_t i = threadIdx.x;
    const uint32_t experts = sh->experts;
    int32_t id = -1, s = -1;
    if (i < n) {
        id = ids[i];
        if (id >= 0 && (uint32_t)id < experts) s = sh->map[layer * experts + (uint32_t)id];
    }
    const uint32_t miss = __ballot_sync(0xffffffffu, i < n && s < 0);
    ds4db_record *r = &sh->rec[seq % DS4DB_RING];
    if (i < n) { r->ids[i] = id; r->slots[i] = s; slots_dev[i] = s; }
    __threadfence_system();
    __syncwarp();
    if (i == 0) {
        r->layer = layer; r->n = n; r->miss = miss;
        __threadfence_system();
        r->posted = seq;
        __threadfence_system();
        if (!miss) sh->bell = seq;
    }
}

/* After the bell: copy the pager's slots for the missed ids. */
__global__ static void ds4db_fixup_kernel(ds4db_shared *sh, uint32_t seq, int32_t *slots_dev, uint32_t n) {
    const uint32_t i = threadIdx.x;
    const ds4db_record *r = &sh->rec[seq % DS4DB_RING];
    if (i < n && (r->miss >> i) & 1u) slots_dev[i] = r->slots[i];
}

/* VRAM residency (1) of every expert of `layer` from the live map, for the
 * cache-affinity reranker; read at execution time, never a host snapshot. */
__global__ static void ds4db_tier_kernel(const ds4db_shared *sh, uint32_t layer, uint8_t *tier, uint32_t experts) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < experts) tier[e] = sh->map[layer * sh->experts + e] >= 0 ? 1u : 0u;
}
#endif
