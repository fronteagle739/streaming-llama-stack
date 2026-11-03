// include/basalt/kernels/dp4a.hpp - the byte-wise dot product and the doorbell pause the kernels share.
#pragma once

#include <cstdint>

#define DP4A(a, b, c) __dp4a((a), (b), (c))

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract.
__device__ __forceinline__ void spin_pause() {
    __nanosleep(100);
}
