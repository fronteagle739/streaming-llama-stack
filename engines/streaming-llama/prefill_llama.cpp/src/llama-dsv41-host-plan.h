#pragma once
// Staging, carry and graph minima are charged before allocating the optional
// Engram LRU. A reused warmup cache is a preference, never an extra admission.
#include <algorithm>
#include <cstdint>
#include <string>
namespace llama_dsv41_host_plan {
struct result { bool ok; uint64_t row_cache_bytes, planned_host; std::string reason; };
inline result plan(uint64_t admitted, uint64_t minimum, uint64_t desired_cache) {
    if (minimum > admitted) return {false, 0, minimum,
        "V4.1 non-elastic host minimum=" + std::to_string(minimum) +
        " exceeds admitted_host=" + std::to_string(admitted) +
        " (provider staging, carry bank/spill, tile and graph metadata)"};
    constexpr uint64_t mib = 1ull << 20;
    uint64_t cache = std::min(desired_cache, admitted - minimum) / mib * mib;
    if (cache < 16 * mib) cache = 0;
    return {true, cache, minimum + cache, {}};
}
}
