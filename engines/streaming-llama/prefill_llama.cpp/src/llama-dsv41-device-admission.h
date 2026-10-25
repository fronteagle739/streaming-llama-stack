#pragma once
// LLAMADECODE measured device plan (pure, tested in tests/dsv41_device_admission.cpp).
//
// The producer loads its CUDA modules eagerly (CUDA_MODULE_LOADING=EAGER), so
// the module code is resident before admission and the free VRAM measured after
// CUDA init already excludes it. The job is admitted at
//   admitted = min(--job-device-bytes, measured_free - reserve)
// and the combined plan (sweep) must fit under it; --job-device-bytes stays the
// caller's upper bound. No component is shrunk to make a plan fit.
//
// The context cache is charged at what it can physically hold for its context
// (fixed buffer or growable full-context mapping), never above --cache-bytes:
// the cache never allocates more than that, so the plan stays exact.
#include <algorithm>
#include <cstdint>
#include <string>

namespace llama_dsv41_device_admission {

struct result {
    bool ok = false;
    uint64_t admitted = 0;   // min(bound, measured_free - reserve); 0 when refused before
    std::string reason;      // empty when ok
};

// measured_free: device free bytes after CUDA init + eager module load.
// bound: --job-device-bytes. reserve: --gpu-reserve-bytes. planned: the plan
// that must fit (the sweep's combined device total).
inline result admit(uint64_t measured_free, uint64_t bound, uint64_t reserve, uint64_t planned) {
    result r;
    if (!bound) { r.reason = "device admission needs an explicit job bound"; return r; }
    if (measured_free <= reserve) {
        r.reason = "measured free device memory " + std::to_string(measured_free) + " does not exceed the reserve " + std::to_string(reserve);
        return r;
    }
    r.admitted = std::min(bound, measured_free - reserve);
    if (planned > r.admitted) {
        r.reason = "device plan " + std::to_string(planned) + " exceeds the measured admission " + std::to_string(r.admitted) +
            " (min(job " + std::to_string(bound) + ", free " + std::to_string(measured_free) + " - reserve " + std::to_string(reserve) + "))";
        return r;
    }
    r.ok = true;
    return r;
}

// Device bytes the context cache can occupy: `required` = its full-context
// logical bytes (llama_dsv41_cache_storage::required_bytes), plus per-owner VMM
// rounding (8 owner tensors x granularity) and 1 MiB tensor-alignment slack.
inline uint64_t cache_charge(uint64_t admitted_cache, uint64_t required, uint64_t granularity = 2ull << 20) {
    return std::min(admitted_cache, required + 8 * granularity + (1ull << 20));
}

} // namespace llama_dsv41_device_admission
