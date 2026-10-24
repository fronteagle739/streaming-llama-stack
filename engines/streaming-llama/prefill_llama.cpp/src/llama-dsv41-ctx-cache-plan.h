#pragma once

// LLAMADECODE producer context cache: precision flag and the growable-cache
// plan arithmetic (pure, unit-tested in tests/test-dsv41-ctx-cache-plan.cpp).
//
// The four compressed/index owners (layers 2, 8, 14: ratio 2; layer 20:
// ratio 1) are the only context-sized producer caches. A growable cache
// reserves their full-context virtual range and maps physical memory in
// `step`-token steps (native ds4-win ctxgrow: 2048), each owner tensor on its
// own granularity-aligned range. Admission still counts the full context;
// the unmapped part is decode-time L2 headroom.

#include <algorithm>
#include <cstdint>
#include <string>

namespace llama_dsv41_ctx_cache {

enum class precision { f32, bf16 };

inline bool parse_precision(const std::string & text, precision & out) {
    if (text == "f32") { out = precision::f32; return true; }
    if (text == "bf16") { out = precision::bf16; return true; }
    return false;
}
inline const char * name(precision p) { return p == precision::bf16 ? "bf16" : "f32"; }
inline uint64_t element_bytes(bool bf16) { return bf16 ? 2 : 4; }

constexpr uint32_t default_step = 2048;
constexpr uint32_t max_context = 1048576;

// The raw rings and pool tails stay F32 and fixed-size (not growable).
inline uint64_t fixed_bytes() { return 40ull*128*512*4 + 3ull*2*512*4; }

inline uint64_t round_up(uint64_t value, uint64_t granularity) {
    return granularity ? (value + granularity - 1) / granularity * granularity : value;
}
// Rows of owner i (0..3) covering `tokens` positions.
inline uint64_t owner_rows(uint32_t owner, uint32_t tokens) {
    return owner < 3 ? (uint64_t(tokens) + 1) / 2 : uint64_t(tokens);
}
// Mapped context after a request for `need` positions: the next step multiple, capped at the context.
inline uint32_t round_tokens(uint32_t context, uint32_t step, uint32_t need) {
    const uint64_t t = (uint64_t(need ? need : 1) + step - 1) / step * step;
    return t > context ? context : uint32_t(t);
}
// Logical bytes of the compressed+index rows covering `tokens` (no granularity).
inline uint64_t logical_bytes(uint32_t tokens, bool bf16) {
    uint64_t rows = 0;
    for (uint32_t i = 0; i < 4; ++i) rows += owner_rows(i, tokens);
    return rows * (512 + 128) * element_bytes(bf16);
}
// Physical bytes mapped to cover `tokens`: each of the 8 owner tensors rounded
// up to the VMM granularity and capped at its full-context range.
inline uint64_t mapped_bytes(uint32_t context, uint32_t tokens, bool bf16, uint64_t granularity) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        const uint64_t rows = std::min(owner_rows(i, tokens), owner_rows(i, context));
        for (const uint64_t width : {uint64_t(512), uint64_t(128)}) {
            const uint64_t full = round_up(owner_rows(i, context) * width * element_bytes(bf16), granularity);
            sum += std::min(round_up(rows * width * element_bytes(bf16), granularity), full);
        }
    }
    return sum;
}
// Physical bytes the next step maps once `mapped_tokens` are covered (0 at full context).
inline uint64_t next_step_bytes(uint32_t context, uint32_t mapped_tokens, uint32_t step, bool bf16, uint64_t granularity) {
    if (mapped_tokens >= context) return 0;
    const uint32_t next = round_tokens(context, step, mapped_tokens + 1);
    return mapped_bytes(context, next, bf16, granularity) - mapped_bytes(context, mapped_tokens, bf16, granularity);
}
// Largest step-rounded position the growable cache can map within `admitted` bytes
// (fixed tensors included). 0 = not even the first step fits.
inline uint32_t reachable_tokens(uint32_t context, uint32_t step, bool bf16, uint64_t granularity, uint64_t admitted) {
    uint32_t best = 0;
    for (uint64_t t = step; ; t += step) {
        const uint32_t tokens = t > context ? context : uint32_t(t);
        if (fixed_bytes() + mapped_bytes(context, tokens, bf16, granularity) > admitted) break;
        best = tokens;
        if (tokens == context) break;
    }
    return best;
}
// Decode L2 offer with a growable cache: the free memory less the floor and
// margin, less the next step's bytes (the step that may map mid-decode).
inline uint64_t l2_offer(uint64_t free_bytes, uint64_t floor, uint64_t margin, uint64_t next_step) {
    const uint64_t keep = floor + margin + next_step;
    return free_bytes > keep ? free_bytes - keep : 0;
}
inline uint32_t l2_slots(uint64_t offer, uint64_t slot_bytes, uint32_t l1_slots) {
    const uint64_t most = l1_slots < 40u*384 ? 40u*384 - l1_slots : 0;
    const uint64_t n = slot_bytes ? std::min<uint64_t>(offer / slot_bytes, most) : 0;
    return n < 2 ? 0 : uint32_t(n);
}
// Producer flags: --ctx-cache-precision f32|bf16 (absent: the existing
// --compressed-cache-bf16 0/1, default f32) and --ctx-cache-growable 0|1
// (default 0). nullptr = flag absent. Returns the refusal, empty when valid.
struct options { precision cache = precision::f32; bool growable = false; };
inline std::string parse_options(const char * precision_arg, const char * growable_arg, const char * legacy_bf16_arg, options & out) {
    out = {};
    const std::string legacy = legacy_bf16_arg ? legacy_bf16_arg : "";
    if (legacy_bf16_arg && legacy != "0" && legacy != "1") return "compressed-cache-bf16 requires0/1";
    if (precision_arg) {
        if (!parse_precision(precision_arg, out.cache)) return "ctx-cache-precision requires f32 or bf16";
        if (legacy_bf16_arg && (legacy == "1") != (out.cache == precision::bf16))
            return "ctx-cache-precision conflicts with compressed-cache-bf16";
    } else if (legacy == "1") out.cache = precision::bf16;
    if (growable_arg) {
        const std::string g = growable_arg;
        if (g != "0" && g != "1") return "ctx-cache-growable requires0/1";
        out.growable = g == "1";
    }
    return {};
}

// Startup refusal of impossible configurations; empty string = admissible.
inline std::string refusal(uint32_t context, bool bf16, bool growable, uint32_t step, uint64_t granularity, uint64_t admitted) {
    if (!context || context > max_context) return "context cache: context must be in 1..1048576";
    if (growable && (!step || step % 2048)) return "context cache: growth step must be a positive multiple of 2048";
    if (fixed_bytes() + logical_bytes(context, bf16) > admitted)
        return std::string("context cache: ") + (bf16 ? "bf16" : "f32") + " cache for context " + std::to_string(context) +
            " needs " + std::to_string(fixed_bytes() + logical_bytes(context, bf16)) + " bytes, over the admitted " + std::to_string(admitted);
    if (growable && reachable_tokens(context, step, bf16, granularity, admitted) < context)
        return "context cache: growable cache reaches only " + std::to_string(reachable_tokens(context, step, bf16, granularity, admitted)) +
            " of context " + std::to_string(context) + " within the admitted " + std::to_string(admitted) + " bytes (VMM granularity " +
            std::to_string(granularity) + "); lower the context";
    return {};
}

} // namespace llama_dsv41_ctx_cache
