#pragma once

#include "llama-dsv41-wire.h"

namespace llama_dsv41 {

// A bounded range of an existing canonical target span. No separate storage
// or wire format: transport adapters use the existing artifact DELTA metadata.
struct state_slice {
    state_span span;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct delta_plan {
    wire_frontier base;
    wire_frontier target;
    uint32_t first_token = 0;
    uint32_t token_count = 0;
    uint64_t state_bytes = 0;
    std::vector<state_slice> slices;
};

// Geometry only. The caller must independently authenticate the model/session,
// verify the base tokens and owner hashes, hold both frontiers immutable during
// transfer, and invalidate any partially applied target. This plan cannot
// certify state equality, perform fallback prefill, or publish a cache.
// Logits are the complete existing129280-F32 vector; token IDs are appended
// from first_token. Pool tails are needed only at an odd target position.
bool make_delta_plan(const wire_frontier & base, const wire_frontier & target,
                     delta_plan & result, std::string & error);

} // namespace llama_dsv41
