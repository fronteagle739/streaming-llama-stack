#pragma once
// LLAMADECODE host-plan cost log (log only; tested in tests/dsv41_host_plan_log.cpp).
// The default Engram row cache takes the host headroom left by every other planned
// component (min(4 GiB, headroom), whole MiB). This line shows each component's
// bytes and the row-cache headroom there would be without it, so the cost of a new
// component is visible. It changes no plan or allocation.
#include <cstdint>
#include <string>
#include <vector>

namespace llama_dsv41_host_plan_log {

struct component { const char *name; uint64_t bytes; };

inline uint64_t headroom(uint64_t admitted, uint64_t planned) { return admitted > planned ? admitted - planned : 0; }

// LLAMADECODE_HOST_PLAN admitted_host=A planned_base=B sum_matches=0|1 row_cache_headroom=H row_cache_bytes=R
//   components=name:bytes,... row_cache_headroom_without=name:bytes,...
inline std::string line(uint64_t admitted, uint64_t planned_base, const std::vector<component> &parts, uint64_t row_cache_bytes) {
    uint64_t sum = 0;
    for (const auto &p : parts) sum += p.bytes;
    std::string s = "LLAMADECODE_HOST_PLAN admitted_host=" + std::to_string(admitted) + " planned_base=" + std::to_string(planned_base) +
        " sum_matches=" + (sum == planned_base ? "1" : "0") + " row_cache_headroom=" + std::to_string(headroom(admitted, planned_base)) +
        " row_cache_bytes=" + std::to_string(row_cache_bytes) + " components=";
    for (size_t i = 0; i < parts.size(); ++i) s += (i ? "," : "") + std::string(parts[i].name) + ":" + std::to_string(parts[i].bytes);
    s += " row_cache_headroom_without=";
    for (size_t i = 0; i < parts.size(); ++i)
        s += (i ? "," : "") + std::string(parts[i].name) + ":" +
             std::to_string(headroom(admitted, planned_base > parts[i].bytes ? planned_base - parts[i].bytes : 0));
    return s;
}

} // namespace llama_dsv41_host_plan_log
