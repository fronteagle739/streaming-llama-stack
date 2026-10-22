#ifndef DS4_SPEC_FETCH_POLICY_H
#define DS4_SPEC_FETCH_POLICY_H
/* Engine-independent speculation admission. No model, CUDA, allocator, cache,
 * clocks, side effects or mutable policy state. The adapter owns the existing
 * target's serial EWMA and the preceding real verify's byte/time counters. */
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
typedef struct {
    double predicted_miss_bytes;
    double measured_miss_bytes, measured_read_seconds;
    double propose_seconds, verify_nonread_seconds;
    double plain_tokens_per_second;
    uint32_t useful_tokens;
} ds4_spec_fetch_input;
typedef struct {
    bool observe_serial, priced, admit;
    double predicted_miss_bytes, bytes_per_second;
    double expected_seconds, plain_seconds;
} ds4_spec_fetch_decision;
/* false = invalid evidence; caller must report a loud error, never accept it.
 * No serial observation yet: obtain one with the existing target path.
 * Serial observed but no verify yet: allow a real proposal/verify to price it.
 * Then admit only if predicted full wall <= useful target tokens / serial rate. */
static inline bool ds4_spec_fetch_decide(const ds4_spec_fetch_input *in,
                                        ds4_spec_fetch_decision *out) {
    if(!in || !out)return false;
    *out=(ds4_spec_fetch_decision){0};
    const double values[]={in->predicted_miss_bytes,in->measured_miss_bytes,
        in->measured_read_seconds,in->propose_seconds,in->verify_nonread_seconds,
        in->plain_tokens_per_second};
    for(unsigned i=0;i<sizeof(values)/sizeof(values[0]);i++)
        if(!isfinite(values[i]) || values[i]<0)return false;
    out->predicted_miss_bytes=in->predicted_miss_bytes;
    if(in->plain_tokens_per_second<=0){out->observe_serial=true;return true;}
    out->admit=true;
    if(in->measured_miss_bytes<=0 || in->measured_read_seconds<=0 || !in->useful_tokens)return true;
    out->priced=true;out->bytes_per_second=in->measured_miss_bytes/in->measured_read_seconds;
    out->expected_seconds=in->propose_seconds+in->verify_nonread_seconds+
        in->predicted_miss_bytes/out->bytes_per_second;
    out->plain_seconds=in->useful_tokens/in->plain_tokens_per_second;
    if(!isfinite(out->expected_seconds) || !isfinite(out->plain_seconds) ||
        !isfinite(out->bytes_per_second) || out->bytes_per_second<=0)return false;
    out->admit=out->expected_seconds<=out->plain_seconds;return true;
}
#endif
