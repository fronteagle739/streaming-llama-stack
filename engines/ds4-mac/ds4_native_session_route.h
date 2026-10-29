/* Studio DECODEFIX/DECODEFIX2 length rule: strict less-than on NEW rows.
 * Ported from ds4perffix-source-76cc0cee08a0, transport SHA256
 * c21d89edb2390c89cc1304ccc2ed1559fc137572520bc6fad4f1df67528d1d1b. */
#ifndef DS4_NATIVE_SESSION_ROUTE_H
#define DS4_NATIVE_SESSION_ROUTE_H
#include <stdbool.h>
#include <stdint.h>
/* A disk seed may initialize an idle/lazy session. Once the RTX owns a
 * checkpoint, replacement must use the verified rewind/import protocol. */
static inline bool ds4_ns_disk_seed_allowed(bool configured, bool started) {
    return !configured || !started;
}
static inline bool ds4_ns_local_work(uint32_t threshold, uint32_t live,
                                     uint32_t total, bool extends) {
    return threshold && extends && total >= live && total-live < threshold;
}
#endif
