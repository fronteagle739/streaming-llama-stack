/* In-stream expert hit resolve with a mapped host "doorbell" (Betelgeuse-style).
 *
 * Shared, self-contained layout (no ds4/ggml dependencies) so any streaming
 * engine can include it: a device kernel resolves the routed experts of one
 * decode layer against a residency map in mapped host memory. On all hits it
 * rings the bell itself and the stream never waits for the host. On a miss it
 * posts the record and the stream waits (cuStreamWaitValue32 on the bell, no
 * spinning kernel, so no WDDM timeout risk) until a host pager thread fills the
 * missing slots and rings the bell.
 *
 * Ordering contract (single decode stream, records in sequence order):
 *   - seq increases by one per resolved layer (never reused while in flight);
 *   - the device writes ids/slots/miss, fences, then publishes `posted`;
 *     on all hits it then writes bell = seq;
 *   - the pager processes records strictly in seq order; on a miss it may
 *     change the map and slots (the stream is blocked behind the bell, so no
 *     later resolve has read the map), fences, then writes bell = seq;
 *   - victims never include an expert this record resolved as a hit (the
 *     pager's cache touches every hit of the record before choosing victims).
 */
#ifndef DS4_EXPERT_DOORBELL_H
#define DS4_EXPERT_DOORBELL_H
#include <stdint.h>

#define DS4DB_VERSION 1u
#define DS4DB_MAX_LAYERS 64u
#define DS4DB_MAX_EXPERTS 512u
#define DS4DB_MAX_K 16u
#define DS4DB_RING 256u

typedef struct ds4db_record {
    volatile uint32_t posted;      /* seq once ids/slots/miss are valid */
    volatile uint32_t miss;        /* bit i set: ids[i] missed */
    volatile uint32_t layer;
    volatile uint32_t n;
    volatile int32_t ids[DS4DB_MAX_K];
    volatile int32_t slots[DS4DB_MAX_K];   /* hits by the device, misses by the pager */
    uint32_t pad[12];
} ds4db_record;

typedef struct ds4db_shared {
    uint32_t version, layers, experts, ring;
    volatile uint32_t bell;        /* the stream waits bell >= seq */
    volatile uint32_t failed;      /* pager failure: nonzero poisons the session */
    uint32_t pad0[26];
    volatile int32_t map[DS4DB_MAX_LAYERS * DS4DB_MAX_EXPERTS];   /* slot or -1 */
    ds4db_record rec[DS4DB_RING];
} ds4db_shared;

#endif
