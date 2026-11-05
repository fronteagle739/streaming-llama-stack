#pragma once
#include <stdint.h>
#if defined(__CUDACC__)
#define DS4_DIRECTION_HOST_DEVICE __host__ __device__
#else
#define DS4_DIRECTION_HOST_DEVICE
#endif

// -2 is the existing packed CPU-lane assignment; -1 is an absent/invalid slot.
DS4_DIRECTION_HOST_DEVICE constexpr bool ds4_direction_rank_slot_active(int32_t slot) {
    return slot >= 0 || slot == -2;
}
// Largest fixed temporary set in direction_fused (all eight rank slots),
// excluding the separately charged factor closure and dense-weight cache.
constexpr uint64_t ds4_direction_packed_scratch_bytes() {
    return 5120ull*2304ull/32ull*17ull +
        (6ull*8ull*2304ull+6ull*8ull+5120ull*8ull+1ull+6ull*2304ull+6ull*5120ull)*4ull+
        6ull*sizeof(int32_t);
}
#undef DS4_DIRECTION_HOST_DEVICE
