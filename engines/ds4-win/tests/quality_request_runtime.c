/* Link this instead of ds4.o for test_quality_request only. It uses the real
 * setters/readback, with model capability enabled solely in the test binary. */
#include "../ds4.c"
void quality_test_enable(void) {
    g_quality.supported=true;
    g_quality.phases_supported=true;
}
