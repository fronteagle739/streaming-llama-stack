#pragma once
#include <stdint.h>
namespace ds4_direction_cache_memory {
constexpr uint64_t dense_capacity_bytes() {return 40ull*5120ull*(8192ull+2304ull)*sizeof(float);}
constexpr uint64_t prefill_factor_floats() {return 40ull*(8192ull+2304ull+384ull*2304ull)+80ull*5121ull;}
constexpr uint64_t native_factor_floats() {return 40ull*(8192ull+2304ull+384ull*2304ull)+80ull*5121ull;}
constexpr bool fits(uint64_t resident,uint64_t bytes,uint64_t budget,uint64_t available,uint64_t floor) {
    return resident<=budget && bytes<=budget-resident && available>=floor && bytes<=available-floor;
}
}
