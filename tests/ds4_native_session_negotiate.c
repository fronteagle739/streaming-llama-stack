/* Mac native transport: decode-delta maximum negotiation with the RTX producer,
   one case per producer/client pairing. Build: cl /nologo /TC /I engines\ds4-mac tests\ds4_native_session_negotiate.c */
#include "ds4_native_session_negotiate.h"
#include <stdio.h>

static int failures = 0;
static void expect(int ok, const char *what) { printf("%s %s\n", ok ? "PASS" : "FAIL", what); if (!ok) ++failures; }
/* The pre-negotiation (rtxmac-producer-1 era) Mac readiness test. */
static int old_client_accepts(int have_legacy, unsigned long long legacy) { return have_legacy && legacy == 128; }

int main(void) {
    expect(ds4_ns_negotiate_yoco_values(2541,2541,2), "exact cone v2");
    expect(ds4_ns_negotiate_yoco_values(140,140,1), "approximate140 v1 compatibility");
    expect(ds4_ns_negotiate_yoco_values(0,0,0), "full-state peer compatibility");
    expect(!ds4_ns_negotiate_yoco_values(2541,2541,1), "v1 cannot label exact cone");
    expect(!ds4_ns_negotiate_yoco_values(140,140,2), "v2 cannot label approximate140");
    expect(!ds4_ns_negotiate_yoco_values(2541,140,1), "peer window must match");
    expect(!ds4_ns_negotiate_yoco_values(139,139,0), "unsupported window refused");
    uint32_t e = 0;
    /* new producer (rtxmac-producer-2b): maximum_decode_tokens=128, maximum_decode_delta_tokens=49152 */
    expect(ds4_ns_negotiate_decode_values(1, 49152, 1, 128, &e) && e == 49152, "new producer + new client: 49152");
    expect(old_client_accepts(1, 128), "new producer + old client: the old ==128 readiness check still passes (it sends <=128)");
    /* old producer: maximum_decode_tokens=128, no delta field */
    expect(ds4_ns_negotiate_decode_values(0, 0, 1, 128, &e) && e == 128, "old producer + new client: 128");
    /* no field at all */
    expect(ds4_ns_negotiate_decode_values(0, 0, 0, 0, &e) && e == 128, "peer advertises nothing: 128");
    /* rtxmac-producer-2 (lockstep build) advertised 49152 in maximum_decode_tokens */
    expect(ds4_ns_negotiate_decode_values(0, 0, 1, 49152, &e) && e == 49152, "rtxmac-producer-2 + new client: 49152");
    expect(ds4_ns_negotiate_decode_values(1, 4096, 1, 128, &e) && e == 4096, "smaller peer maximum wins");
    expect(ds4_ns_negotiate_decode_values(1, 1u << 20, 1, 128, &e) && e == DS4_NS_DECODE_DELTA_MAX, "own maximum caps a larger peer");
    expect(!ds4_ns_negotiate_decode_values(1, 64, 1, 128, &e), "peer below one 128-token chunk refused");
    printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
