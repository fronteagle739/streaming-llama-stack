#pragma once
#include <cstdint>
#include <stdexcept>

// Scheduling only: fixed storage, arithmetic, and expert policy are unchanged.
// A paired owner's unaligned continuation uses one <=2K attention graph, so
// build_pair never joins across the native odd pool-tail boundary.
struct llama_dsv41_append_schedule {
    uint32_t step;
    bool single_attention_append;
};
inline llama_dsv41_append_schedule llama_dsv41_plan_append(
        bool paired, uint32_t maximum, uint32_t requested_step,
        uint32_t position, uint32_t unaligned_step) {
    if (!maximum || (unaligned_step != 0 && unaligned_step != 2048))
        throw std::runtime_error("Append requires an explicit0/2048 unaligned step");
    const uint32_t step=requested_step ? requested_step : maximum;
    if (!step || step>maximum || (paired && step!=maximum))
        throw std::runtime_error("Append logical step differs from admitted group");
    if (paired && position%4096) {
        if (unaligned_step!=2048 || maximum<4096)
            throw std::runtime_error("Paired append requires aligned positions or explicit2K continuation");
        return {unaligned_step,true};
    }
    return {step,false};
}
