#include "../ds4_spec_fetch_policy.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    ds4_spec_fetch_input in={0};ds4_spec_fetch_decision out;
    assert(ds4_spec_fetch_decide(&in,&out) && out.observe_serial && !out.admit);
    in.plain_tokens_per_second=5;
    assert(ds4_spec_fetch_decide(&in,&out) && !out.observe_serial && !out.priced && out.admit);
    uint32_t state=1234567;
    for(int i=0;i<10000;i++) {
        state=1664525u*state+1013904223u;in.measured_miss_bytes=(state%10000000u)+1;
        state=1664525u*state+1013904223u;in.measured_read_seconds=(state%10000u+1)*.0001;
        state=1664525u*state+1013904223u;in.propose_seconds=(state%1000u)*.00001;
        state=1664525u*state+1013904223u;in.verify_nonread_seconds=(state%1000u)*.0001;
        state=1664525u*state+1013904223u;in.plain_tokens_per_second=(state%2000u+1)*.01;
        in.useful_tokens=1+state%6;in.predicted_miss_bytes=in.measured_miss_bytes;
        double bandwidth=in.measured_miss_bytes/in.measured_read_seconds;
        double expected=in.propose_seconds+in.verify_nonread_seconds+in.measured_miss_bytes/bandwidth;
        double plain=in.useful_tokens/in.plain_tokens_per_second;
        assert(ds4_spec_fetch_decide(&in,&out) && out.priced && out.admit==(expected<=plain));
        assert(out.expected_seconds==expected && out.plain_seconds==plain);
    }
    in.propose_seconds=NAN;assert(!ds4_spec_fetch_decide(&in,&out));
    in.propose_seconds=-1;assert(!ds4_spec_fetch_decide(&in,&out));
    in.propose_seconds=INFINITY;assert(!ds4_spec_fetch_decide(&in,&out));
    assert(!ds4_spec_fetch_decide(NULL,&out));
    puts("Shared fetch policy: 10000 exact original-formula decisions, bootstrap and invalid evidence PASS");return 0;
}
