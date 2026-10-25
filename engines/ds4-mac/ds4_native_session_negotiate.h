#ifndef DS4_NATIVE_SESSION_NEGOTIATE_H
#define DS4_NATIVE_SESSION_NEGOTIATE_H
/* Decode-delta maximum negotiation with the RTX producer (pure; CPU-tested).
   The peer's accepted maximum is maximum_decode_delta_tokens when it sends one,
   else its maximum_decode_tokens (an old producer sends 128; rtxmac-producer-2
   sent 49152 there), else 128. effective = min(that, DS4_NS_DECODE_DELTA_MAX).
   A peer below one 128-token chunk is refused. */
#include <stdbool.h>
#include <stdint.h>
#define DS4_NS_DECODE_DELTA_CHUNK 128u
#define DS4_NS_DECODE_DELTA_MAX 49152u
static bool ds4_ns_negotiate_decode_values(bool have_delta_max,uint64_t delta_max,bool have_legacy,uint64_t legacy,uint32_t*effective){
    uint64_t advertised=DS4_NS_DECODE_DELTA_CHUNK;
    if(have_delta_max)advertised=delta_max;
    else if(have_legacy)advertised=legacy;
    if(advertised<DS4_NS_DECODE_DELTA_CHUNK)return false;
    *effective=(uint32_t)(advertised<DS4_NS_DECODE_DELTA_MAX?advertised:DS4_NS_DECODE_DELTA_MAX);
    return true;
}
#endif
