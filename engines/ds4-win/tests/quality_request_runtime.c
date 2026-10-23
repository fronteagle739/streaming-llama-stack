/* Reused from private provider e1e45374ddc307c75bd7fc0e4b8dccce6b0af6e2, tests/quality_request_runtime.c. */
/* Link this instead of ds4.o for test_quality_request only. It uses the real
 * setters/readback, with model capability enabled solely in the test binary. */
#include "../ds4.c"
void quality_test_enable(void) {
    g_quality.supported=true;
    g_quality.phases_supported=true;
}
