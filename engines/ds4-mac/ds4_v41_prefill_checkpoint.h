#ifndef DS4_V41_PREFILL_CHECKPOINT_H
#define DS4_V41_PREFILL_CHECKPOINT_H

/* ES1: when a V4.1 prefill chunk may defer its decoder.
 *
 * A deferred ("encoder-only") sweep runs layers 0-19 for the chunk and leaves
 * the decoder invalid until a later sweep rebuilds its dependency suffix, so a
 * chunk boundary reached by a deferred sweep has no continuable frontier and
 * the KV store refuses to stage it ("session has no valid checkpoint to
 * stage").  Before ES1 every chunk of a long streamed prefill deferred except
 * the last two, so a 435,563-token cold prefill cancelled at 360,448 left
 * nothing on disk.  `checkpoint_tokens` bounds how far a prefill may run on
 * deferred sweeps: once a chunk would end that many tokens past the last
 * completed boundary, that chunk completes its decoder and the boundary can be
 * stored.  0 restores the pre-ES1 schedule. */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Two 32k sweeps per completed boundary: the completing sweep measured about
 * +7 s over a ~70 s deep chunk (Mac log, 2026-09-24 11:34:46), so roughly +5 %
 * on a long cold prefill in exchange for losing at most ~64k tokens on a
 * cancel.  DS4_V41_PREFILL_CHECKPOINT_TOKENS overrides it (0 = off). */
#define DS4_V41_PREFILL_CHECKPOINT_DEFAULT_TOKENS 65536u

static inline uint32_t ds41_prefill_checkpoint_tokens_parse(const char *v) {
    if (!v || !v[0]) return DS4_V41_PREFILL_CHECKPOINT_DEFAULT_TOKENS;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v) return DS4_V41_PREFILL_CHECKPOINT_DEFAULT_TOKENS;
    if (n <= 0) return 0u;
    if (n > 0x7fffffffL) n = 0x7fffffffL;
    return (uint32_t)n;
}

/* `end_since_complete`: tokens from the last completed (continuable) boundary
 * -- the prefill's resume point to begin with -- to this chunk's end. */
static inline bool ds41_prefill_should_defer_decoder(bool encoder_resident,
                                                     uint32_t count,
                                                     uint32_t remaining,
                                                     bool decoder_suffix_lever,
                                                     bool defer_disabled,
                                                     uint32_t end_since_complete,
                                                     uint32_t checkpoint_tokens) {
    if (encoder_resident || count < 16384u || remaining < count ||
        remaining - count < 8192u || !decoder_suffix_lever || defer_disabled)
        return false;
    if (checkpoint_tokens && end_since_complete >= checkpoint_tokens)
        return false;
    return true;
}

#endif
