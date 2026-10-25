// LLAMADECODE single-sweep prefill plan: near-equal split and hidden-bank strategy.
#include "llama-dsv41-prefill-plan.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

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
    // Owner fold policy (--prefill-sweep-tokens S): S-token sweeps, remainder folded into the last.
    auto counts = [](const std::vector<p::extent> &v) { std::vector<uint32_t> c; for (const auto &e : v) c.push_back(e.count); return c; };
    auto starts_ok = [](const std::vector<p::extent> &v, uint32_t s) {
        for (size_t i = 0; i < v.size(); ++i) if (v[i].first != uint32_t(i) * s) return false;
        return true;
    };
    auto f = p::fold(110000, 49152, 1048576);
    expect(f.count == 2 && f.sweep == 49152 && f.last == 60848 && !f.tail, "110,000 at S 49,152: 49,152 + 60,848 (folded)");
    auto v = p::plan_sweeps(110000, 60848, 49152);
    expect(counts(v) == std::vector<uint32_t>{49152, 60848} && starts_ok(v, 49152), "110,000, S 49,152, C 60,848: sweeps [0:49,152] [49,152:60,848]");
    v = p::plan_sweeps(93668, 93668, 65536);
    expect(counts(v) == std::vector<uint32_t>{93668}, "93,668 at S 65,536: one sweep of 93,668");
    v = p::plan_sweeps(109000, 59848, 49152);
    expect(counts(v) == std::vector<uint32_t>{49152, 59848}, "109,000 at S 49,152, C 59,848: 49,152 + 59,848");
    v = p::plan_sweeps(30000, 49152, 49152);
    expect(counts(v) == std::vector<uint32_t>{30000}, "N < S: one sweep");
    v = p::plan_sweeps(49152, 49152, 49152);
    expect(counts(v) == std::vector<uint32_t>{49152}, "N = S: one sweep");
    v = p::plan_sweeps(49153, 49153, 49152);
    expect(counts(v) == std::vector<uint32_t>{49153}, "N = S + 1: one folded sweep of 49,153");
    v = p::plan_sweeps(98304, 49152, 49152);
    expect(counts(v) == std::vector<uint32_t>{49152, 49152}, "N = 2S: two S sweeps, nothing to fold");
    // RAM-limited (S + r > C, no fold cap hit): rebalance the final passes, never a tiny tail.
    f = p::fold(230000, 49152, 81000);
    expect(f.rebalance && !f.tail && f.count == 5 && f.pass == 45056 && f.last == 37488, "230,000 at S 49,152, C 81,000: 82,544 over C -> 3 x 49,152 + 45,056 + 37,488");
    v = p::plan_sweeps(110000, 60847, 49152);
    expect(counts(v) == std::vector<uint32_t>{49152, 32768, 28080}, "110,000 one row over the capacity: 49,152 + 32,768 + 28,080 (was + 11,696 tail)");
    expect(!p::fold(110000, 49000, 1048576).count, "S not a 4096 multiple: refused");
    expect(!p::fold(110000, 65536, 49152).count && p::plan_sweeps(110000, 49152, 65536).empty(), "S above the capacity: refused");
    expect(!p::fold(110000, 0, 1048576).count, "S = 0 is not fold");
    v = p::plan_sweeps(30000, 49152, 65536);
    expect(counts(v) == std::vector<uint32_t>{30000}, "N < S with S above the capacity: one sweep of N");
    expect(p::plan_sweeps(60000, 49152, 65536).empty(), "N <= S but above the capacity: refused");
    // Invariants: cover N contiguously, no pass > C, every start 4096-aligned, no pass < 4096
    // unless N is, the primary sweeps first; over N x C x (S, F).
    uint64_t checked = 0;
    for (uint32_t n = 1; n < 400000; n += 977) {
        for (const auto sf : {std::pair<uint32_t, uint32_t>{49152, 0}, {49152, 61440}, {65536, 0}}) {
            for (uint32_t cap = sf.first; cap <= 2 * sf.first + 4096; cap += 1517) {
                const auto w = p::plan_sweeps(n, cap, sf.first, sf.second);
                uint64_t next = 0; bool ok = !w.empty();
                for (size_t i = 0; ok && i < w.size(); ++i) {
                    ok = w[i].first == next && w[i].first % 4096 == 0 && w[i].count <= cap && (w[i].count >= 4096 || n < 4096 || w.size() == 1);
                    next += w[i].count;
                }
                ok = ok && next == n;
                for (size_t i = 0; ok && i + 2 < w.size(); ++i) ok = w[i].count == sf.first || w[i + 1].count != sf.first;   // S sweeps lead
                if (!ok) { std::printf("n=%u cap=%u S=%u F=%u\n", n, cap, sf.first, sf.second); expect(false, "fold invariants"); return 1; }
                ++checked;
            }
        }
    }
    std::printf("fold invariant cases: %llu\n", (unsigned long long)checked);
    expect(true, "fold invariants (cover, <= C, aligned starts, no pass < 4096, S first) over N x C x (S, F)");
    // Fold cap F = 61,440 (owner, measured): fold only while S + r <= F.
    auto capped = [&](uint32_t n) { return counts(p::plan_sweeps(n, 1048576, 49152, 61440)); };
    expect(capped(103189) == std::vector<uint32_t>{49152, 54037}, "fold cap: 103,189 -> 49,152 + 54,037");
    expect(capped(93668) == std::vector<uint32_t>{49152, 44516}, "fold cap: 93,668 -> 49,152 + 44,516 (93,668 > 61,440)");
    expect(capped(60000) == std::vector<uint32_t>{60000}, "fold cap: 60,000 -> one sweep");
    expect(capped(70000) == std::vector<uint32_t>{49152, 20848}, "fold cap: 70,000 -> 49,152 + 20,848");
    expect(capped(110000) == std::vector<uint32_t>{49152, 60848}, "fold cap: 110,000 -> 49,152 + 60,848");
    expect(capped(120000) == std::vector<uint32_t>{49152, 49152, 21696}, "fold cap: 120,000 -> 49,152 + 49,152 + 21,696");
    expect(capped(30000) == std::vector<uint32_t>{30000} && capped(98304) == std::vector<uint32_t>{49152, 49152}, "fold cap: N < S and N = 2S unchanged");
    expect(p::fold(93668, 49152, 1048576, 61440).tail && !p::fold(93668, 49152, 1048576).tail, "fold cap absent (0): today's fold");
    expect(counts(p::plan_sweeps(110000, 60000, 49152, 61440)) == std::vector<uint32_t>{49152, 32768, 28080}, "capacity below the fold cap: rebalanced, no tiny tail");
    // Owner examples (fold cap 61,440).
    expect(counts(p::plan_sweeps(103189, 53000, 49152, 61440)) == std::vector<uint32_t>{49152, 28672, 25365}, "103,189 at C 53,000 -> 49,152 + 28,672 + 25,365 (was 49,152 + 49,152 + 4,885)");
    expect(counts(p::plan_sweeps(103189, 54037, 49152, 61440)) == std::vector<uint32_t>{49152, 54037} && counts(p::plan_sweeps(103189, 70000, 49152, 61440)) == std::vector<uint32_t>{49152, 54037},
           "103,189 at C >= 54,037 -> 49,152 + 54,037");
    expect(counts(p::plan_sweeps(93668, 53000, 49152, 61440)) == std::vector<uint32_t>{49152, 44516}, "93,668 -> 49,152 + 44,516 (fold cap)");
    expect(counts(p::plan_sweeps(150000, 50000, 49152, 61440)) == std::vector<uint32_t>{49152, 49152, 28672, 23024}, "150,000 at C 50,000 -> 2 x 49,152 + 28,672 + 23,024");
    f = p::fold(103189, 49152, 53000, 61440);
    expect(f.rebalance && !f.tail && f.passes == 2, "rebalance flagged, not a tail");
    // Absent option: the sweep list is exactly today's split().
    for (uint32_t n = 1; n < 400000; n += 977) {
        for (uint32_t cap : {49152u, 61440u, 107868u, 127000u}) {
            const auto t = p::split(n, cap);
            const auto w = p::plan_sweeps(n, cap);
            bool ok = w.size() == t.count;
            for (size_t i = 0; ok && i < w.size(); ++i)
                ok = w[i].first == uint32_t(i) * t.chunk && w[i].count == std::min(t.chunk, n - uint32_t(i) * t.chunk);
            if (!ok) { expect(false, "absent fold == split"); return 1; }
        }
    }
    expect(p::plan_sweeps(5000, 4095).empty(), "absent fold: split refusal is an empty list");
    expect(true, "absent fold == split() over 410 x 4 prompts");
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
