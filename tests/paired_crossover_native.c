#include <assert.h>
#include "../engines/ds4-mac/ds4_native_session_route.h"
int main(void) {
    assert(ds4_ns_disk_seed_allowed(false, false));
    assert(ds4_ns_disk_seed_allowed(false, true));
    assert(ds4_ns_disk_seed_allowed(true, false));
    assert(!ds4_ns_disk_seed_allowed(true, true));
    assert(ds4_ns_local_work(11776, 0, 12, true));
    assert(ds4_ns_local_work(11776, 120000, 120023, true));
    assert(ds4_ns_local_work(11776, 0, 11775, true));
    assert(!ds4_ns_local_work(11776, 0, 11776, true));
    assert(!ds4_ns_local_work(11776, 0, 200000, true));
    assert(!ds4_ns_local_work(11776, 100, 120, false));
    assert(!ds4_ns_local_work(11776, 100, 99, true));
    assert(!ds4_ns_local_work(0, 0, 12, true));
    return 0;
}
