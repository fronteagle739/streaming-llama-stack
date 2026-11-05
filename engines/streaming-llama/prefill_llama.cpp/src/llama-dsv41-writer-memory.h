#pragma once
#include <cstdint>
#include <stdexcept>
#include "../../../ds4-win/ds4_direction_cache_memory.h"

namespace dsv41_writer_memory {
inline uint64_t prefill_device_bytes(uint32_t count) {
    if (count > 8) throw std::runtime_error("Direction bank count must be0..8");
    if (!count) return 0;
    const uint64_t factor_floats = ds4_direction_cache_memory::prefill_factor_floats();
    return count * factor_floats * sizeof(float) + (8ull << 20);
}
// Both engines retain the full target-factor closure; decode also retains
// the complete dense writer working set. This is separate from kernel workspace.
inline uint64_t resident_device_bytes(uint32_t count) {
    if (count > 8) throw std::runtime_error("Direction bank count must be0..8");
    if (!count) return 0;
    const uint64_t factor_floats = ds4_direction_cache_memory::prefill_factor_floats()+ds4_direction_cache_memory::native_factor_floats();
    return count * factor_floats * sizeof(float) + ds4_direction_cache_memory::dense_capacity_bytes() + (8ull << 20);
}
inline uint64_t pending_device_bytes(uint32_t count,uint64_t allocated) {
    const auto total=resident_device_bytes(count);
    if(allocated>total)throw std::runtime_error("Direction allocations exceed the admitted writer plan");
    return total-allocated;
}
}
