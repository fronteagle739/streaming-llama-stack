#pragma once
#include <stdint.h>

/* Versioned, synchronous bridge to ds4's existing pinned expert cache. The
 * caller owns physical/RSS admission and serializes all calls with routing.
 * Slot sources are consumed before return; no deferred pointer can be evicted.
 * A model/layout change must clear the cache before reusing expert keys. */
struct ds4_external_host_cache_v1 {
    uint32_t (*resize)(uint64_t bytes, uint64_t gate_bytes, uint64_t down_bytes);
    int (*find)(uint32_t layer, uint32_t expert);
    int (*upload)(uint32_t slot, char *const *dst, const uint64_t *sizes);
    int (*store)(uint32_t layer, uint32_t expert, const char *const *src, const uint64_t *sizes);
    void (*clear)(void);
};
