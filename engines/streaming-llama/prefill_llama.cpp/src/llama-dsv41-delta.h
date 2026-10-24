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

// apply_decode_delta: peers stream decoded rows as <=128-token deltas (the
// normal chunk). The accepted maximum, advertised as maximum_decode_tokens, is
// one whole local append (the 49152-token prefill chunk): full raw rings plus
// every appended compressed/index row, target position only. Same plan, same
// artifact and the same base/target owner, token and logit SHA checks.
constexpr uint32_t decode_delta_chunk_tokens = 128;
constexpr uint32_t decode_delta_max_tokens = 49152;
// Sanity cap on a decode artifact's declared bytes before streaming (the
// receiver then requires the exact extent): 64 MiB covers any <=128-token
// delta; a larger one adds at most 6404 bytes per token (3x2048/2 + 2048
// compressed, 3x512/2 + 512 index, 4 token) over that.
constexpr uint64_t decode_delta_byte_cap(uint32_t tokens) {
    return (64ull << 20) + (tokens > decode_delta_chunk_tokens ? uint64_t(tokens) * 6404u : 0u);
}

} // namespace llama_dsv41
