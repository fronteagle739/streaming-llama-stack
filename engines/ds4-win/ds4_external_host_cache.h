#pragma once
#include <stdint.h>
#include <stddef.h>

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

/* ggml's dispatch-selected CPU dot kernel (including AVX2/AVX-512 builds).
 * All weights retain their original quantization; no expanded weight copy. */
typedef void (*ds4_external_dot)(int, float *, size_t, const void *, size_t,
                               const void *, size_t, int);
#ifdef __cplusplus
extern "C" {
#endif
int ds4_external_cpu_q4(float *out, const float *x, const char *const *gate,
    const char *const *up, const char *const *down, const float *weights,
    uint32_t n, uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim, float clamp);
#ifdef __cplusplus
}
#endif
