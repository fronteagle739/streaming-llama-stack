// LLAMADECODE --ctx-cache-precision / growable producer context cache: flag
// parsing, the VRAM plan arithmetic (grow step, next-step reserve, L2 offer and
// slots) and the refusal of impossible configurations. Byte figures are the
// real producer geometry: 2 MiB VMM granularity, the 3,225,419,776-byte
// admitted cache of the 1M device bill, 19,906,560-byte expert slots.
#include "llama-dsv41-ctx-cache-plan.h"
#include <cstdio>
#include <string>

namespace cc = llama_dsv41_ctx_cache;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const uint64_t G = 2ull << 20, admitted = 3225419776ull, slot = 19906560ull;
    // Flags.
    cc::options o;
    expect(cc::parse_options(nullptr, nullptr, nullptr, o).empty() && o.cache == cc::precision::f32 && !o.growable, "default f32 fixed");
    expect(cc::parse_options("bf16", "1", nullptr, o).empty() && o.cache == cc::precision::bf16 && o.growable, "bf16 growable");
    expect(cc::parse_options("f32", "0", "0", o).empty() && o.cache == cc::precision::f32, "f32 with legacy 0");
    expect(cc::parse_options("bf16", nullptr, "1", o).empty() && o.cache == cc::precision::bf16, "bf16 with legacy 1");
    expect(cc::parse_options(nullptr, nullptr, "1", o).empty() && o.cache == cc::precision::bf16, "legacy 1 alone = bf16");
    expect(!cc::parse_options("bf16", nullptr, "0", o).empty(), "conflict bf16 vs legacy 0 refused");
    expect(!cc::parse_options("f32", nullptr, "1", o).empty(), "conflict f32 vs legacy 1 refused");
    expect(!cc::parse_options("f16", nullptr, nullptr, o).empty(), "unknown precision refused");
    expect(!cc::parse_options("f32", "yes", nullptr, o).empty(), "growable not 0/1 refused");
    expect(!cc::parse_options(nullptr, nullptr, "2", o).empty(), "legacy not 0/1 refused");

    // Full-context bytes equal the producer's existing required_bytes formula (6400 / 3200 B per token).
    expect(cc::fixed_bytes() + cc::logical_bytes(262144, false) == 1688219648ull, "262K f32 full bytes");
    expect(cc::fixed_bytes() + cc::logical_bytes(1000001, true) == 3210503168ull, "1000001 bf16 full bytes");
    expect(cc::logical_bytes(2048, false) == 2048ull * 6400, "one f32 step = 2048 x 6400 B logical");

    // Step rounding.
    expect(cc::round_tokens(262144, 2048, 1) == 2048 && cc::round_tokens(262144, 2048, 2048) == 2048 &&
           cc::round_tokens(262144, 2048, 2049) == 4096 && cc::round_tokens(999424, 2048, 999424) == 999424 &&
           cc::round_tokens(1000, 2048, 999) == 1000, "round to the step, capped at the context");
    // Granular mapping: first f32 step maps 3 x (2 MiB + 2 MiB) + (4 MiB + 2 MiB) (each index range rounds up to 2 MiB).
    expect(cc::mapped_bytes(262144, 2048, false, G) == 3 * (2 * G) + (2 * G + G), "first f32 step physical bytes");
    expect(cc::mapped_bytes(262144, 262144, false, G) == cc::logical_bytes(262144, false), "262K f32 full map has no rounding");
    // Next-step reserve.
    expect(cc::next_step_bytes(262144, 262144, 2048, false, G) == 0, "no next step at full context");
    expect(cc::next_step_bytes(262144, 2048, 2048, false, G) == cc::mapped_bytes(262144, 4096, false, G) - cc::mapped_bytes(262144, 2048, false, G),
           "next step = mapped(next) - mapped(now)");
    uint64_t worst = 0;
    for (uint32_t t = 2048; t < 999424; t += 2048) worst = std::max(worst, cc::next_step_bytes(999424, t, 2048, true, G));
    expect(worst <= 8 * G, "bf16 step never maps more than 8 granules");

    // Reachable context and refusals.
    expect(cc::refusal(262144, false, false, 2048, G, admitted).empty(), "262K f32 fixed admissible");
    expect(cc::refusal(262144, false, true, 2048, G, admitted).empty(), "262K f32 growable admissible");
    expect(cc::refusal(1000001, true, false, 2048, G, admitted).empty(), "1000001 bf16 fixed admissible");
    expect(!cc::refusal(1000001, true, true, 2048, G, admitted).empty(), "1000001 bf16 growable refused (granule rounding)");
    expect(cc::reachable_tokens(1000001, 2048, true, G, admitted) == 999424, "bf16 growable reaches 999424 within the bill");
    expect(cc::refusal(999424, true, true, 2048, G, admitted).empty(), "999424 bf16 growable admissible");
    expect(!cc::refusal(1000001, false, false, 2048, G, admitted).empty(), "1M f32 refused (6.4 GB over 3.2 GB)");
    expect(!cc::refusal(524288, false, true, 2048, G, admitted).empty(), "512K f32 refused");
    expect(cc::refusal(502000, false, false, 2048, G, admitted).empty() && !cc::refusal(503000, false, false, 2048, G, admitted).empty(),
           "f32 fixed cap between 502000 and 503000");
    expect(!cc::refusal(0, false, false, 2048, G, admitted).empty() && !cc::refusal(1048577, true, false, 2048, G, admitted).empty(),
           "context out of range refused");
    expect(!cc::refusal(262144, false, true, 1000, G, admitted).empty(), "step not a multiple of 2048 refused");

    // L2 offer (shrink L2 first: the next step is kept free) and slots.
    expect(cc::l2_offer(13125935104ull, 512ull << 20, 768ull << 20, 0) == 13125935104ull - (1280ull << 20), "fixed offer unchanged");
    expect(cc::l2_offer(13125935104ull, 512ull << 20, 768ull << 20, 6 * G) == 13125935104ull - (1280ull << 20) - 6 * G, "growable offer keeps the next step");
    expect(cc::l2_offer(100, 512ull << 20, 768ull << 20, G) == 0, "offer floors at 0");
    expect(cc::l2_slots(11783757824ull, slot, 272) == 591, "measured 262K offer = 591 L2 slots");
    expect(cc::l2_slots(slot, slot, 272) == 0, "under two slots = no L2");
    expect(cc::l2_slots(~0ull, slot, 272) == 40 * 384 - 272, "L2 capped at every expert");
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
