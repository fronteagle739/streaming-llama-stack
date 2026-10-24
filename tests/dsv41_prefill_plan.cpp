// LLAMADECODE single-sweep prefill plan: near-equal split and hidden-bank strategy.
#include "llama-dsv41-prefill-plan.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace p = llama_dsv41_prefill;
static int passed = 0, failed = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    ok ? ++passed : ++failed;
}

int main() {
    // One sweep whenever the prompt fits the capacity (ragged tail included).
    auto s = p::split(103319, 107868);
    expect(s.count == 1 && s.chunk == 103319, "103,319 tokens, capacity 107,868: one sweep");
    s = p::split(49152, 49152);
    expect(s.count == 1 && s.chunk == 49152, "49,152 at the legacy capacity: one sweep");
    s = p::split(14859, 49152);
    expect(s.count == 1 && s.chunk == 14859, "16K: one sweep, chunk = prompt");
    // Last resort: near-equal, 4096-aligned sweeps, never a tiny tail.
    s = p::split(103319, 49152);
    expect(s.count == 3 && s.chunk == 36864 && 103319 - 2 * 36864 == 29591, "103,319 at 49,152: 36,864 + 36,864 + 29,591 (old: + 5,015 tail)");
    s = p::split(224588, 107868);
    expect(s.count == 3 && s.chunk == 77824, "224,588 at 107,868: three sweeps of <= 77,824");
    s = p::split(224588, 114688);
    expect(s.count == 2 && s.chunk == 114688 && 224588 - 114688 == 109900, "224,588 at 114,688: 114,688 + 109,900");
    s = p::split(98305, 49152);
    expect(s.count == 3 && s.chunk == 36864, "98,305 at 49,152: three near-equal sweeps, not 49,152 + 49,152 + 1");
    s = p::split(1000, 4095);
    expect(s.count == 1, "small prompt below one aligned tile: one sweep");
    s = p::split(5000, 4095);
    expect(s.count == 0, "capacity below one aligned tile cannot split: refused");
    for (uint32_t n = 1; n < 400000; n += 977) {
        for (uint32_t cap : {49152u, 61440u, 107868u, 127000u}) {
            const auto t = p::split(n, cap);
            if (!(t.count && t.chunk <= cap && uint64_t(t.chunk) * t.count >= n && uint64_t(t.chunk) * (t.count - 1) < n &&
                  (t.count == 1 || t.chunk % 4096 == 0))) { expect(false, "split invariants"); return 1; }
            // Near-equal: no more sweeps than ceil(n / aligned cap), and the remainder
            // misses a full chunk by less than one 4096 tile per sweep.
            if (t.count > 1 && (t.count > (n + cap / 4096 * 4096 - 1) / (cap / 4096 * 4096) ||
                                uint64_t(t.chunk) * t.count - n >= 4096ull * t.count)) {
                expect(false, "near-equal invariant"); return 1;
            }
        }
    }
    expect(true, "split invariants over 410 x 4 prompts (cover, <= cap, aligned, no slack sweep)");
    // Host bytes that scale with capacity (match the producer's admissions).
    expect(p::carry_bank_bytes(49152) == 4536143872ull, "carry bank at 49,152 = 4,536,143,872 B (DS4PERF_RAM_ALLOCATED_BEFORE_IMPORT)");
    expect(p::engram_prompt_bytes(49152) == 636485632ull, "Engram prompt cache at 49,152 = 636,485,632 B (--engram-prefetch-bytes)");
    expect(p::carry_bank_bytes(49184) - p::carry_bank_bytes(49152) == 2953216ull, "one more 32-row page = 2,953,216 B");
    // Hidden bank strategy (layout ~51,216 B/row + 64 KiB, like the BF16+ID resident bank).
    const uint64_t admitted = 2521649916ull;
    auto layout = [](uint32_t rows) { return uint64_t(rows) * 51216ull + 65536ull; };
    auto b = p::plan_bank(49152, 4096, admitted, layout);
    expect(b.fits && !b.spill && b.rows == 49152 && b.resident_rows == 49152, "capacity 49,152: resident bank, no spill (today's plan)");
    b = p::plan_bank(30000, 4096, admitted, layout);
    expect(b.fits && !b.spill && b.rows == 30000, "smaller capacity: resident");
    b = p::plan_bank(107868, 4096, admitted, layout);
    expect(b.fits && b.spill && b.rows == 49152 && b.resident_rows == 45056 && b.slot_rows == 4096, "capacity 107,868: bank 49,152 = 45,056 resident + one 4,096 slot");
    b = p::plan_bank(107868, 4096, 4096ull * 51216 + 65536, layout);
    expect(!b.fits, "bank of one tile cannot spill (needs a resident tile and a slot)");
    std::printf("dsv41_prefill_plan: %d PASS, %d FAIL\n", passed, failed);
    return failed ? 1 : 0;
}
