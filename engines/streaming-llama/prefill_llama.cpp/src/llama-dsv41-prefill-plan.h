#pragma once
// LLAMADECODE single-sweep prefill plan (owner: one prompt = one weight sweep).
//
// Per layer, while its weights are resident, the sweep visits ALL of the
// prompt's tiles (ragged tail included). The producer's fixed capacity C
// (--prefill-chunk-tokens) is the largest prompt one sweep can take; it is
// sized by the caller from the actual prompt and bounded by host RAM (the
// locked carry bank and the whole-prompt Engram cache scale with C).
// The CUDA hidden bank (--hidden-device-bytes) does NOT scale: when it cannot
// hold all C rows, its last tile is a staging slot and the rows past the
// resident prefix are spilled, per tile, to their own RAM carry records
// (exact BF16/F32/ID bytes) between layers. Only a prompt larger than C is
// split, into near-equal sweeps (never a tiny tail).
// Pure arithmetic only (plus the small sweep list); no CUDA or model access.
#include <algorithm>
#include <cstdint>
#include <vector>

namespace llama_dsv41_prefill {

// The pre-single-sweep contract value: still advertised as
// maximum_prefill_tokens so old clients (which require exactly 49152) keep
// their handshake; maximum_prefill_sweep_tokens advertises C.
constexpr uint32_t legacy_sweep_tokens = 49152;
constexpr uint32_t max_sweep_tokens = 1048576;
constexpr uint32_t sweep_align = 4096;   // paired attention tiles: absolute 4096 alignment

// Host bytes that scale with the capacity (same formulas as
// llama_dsv41_carry_store::plan_resident_ram_direct and the provider's
// whole-prompt Engram admission). Per row beyond 49152: 92288 + 12864 B.
constexpr uint64_t carry_bank_bytes(uint32_t rows) { return 4096ull + uint64_t((rows + 31) / 32) * 2953216ull; }
constexpr uint64_t engram_prompt_bytes(uint32_t rows) { return (4ull << 20) + uint64_t(rows) * 48ull * (4 + 264); }

struct sweeps { uint32_t count = 0, chunk = 0; };
// One sweep when n <= cap; otherwise ceil(n / aligned cap) sweeps of a near-equal,
// align-rounded chunk (the last one is the remainder, never a tiny tail unless n is).
inline sweeps split(uint32_t n, uint32_t cap, uint32_t align = sweep_align) {
    sweeps out;
    if (!n || !cap || !align) return out;
    if (n <= cap) { out.count = 1; out.chunk = n; return out; }
    const uint32_t aligned = cap / align * align;
    if (!aligned) return out;                      // cap below one aligned tile: refuse (count 0)
    const uint32_t k = (n + aligned - 1) / aligned;
    const uint32_t even = (n + k - 1) / k;
    out.chunk = (even + align - 1) / align * align;
    if (out.chunk > aligned) out.chunk = aligned;
    out.count = (n + out.chunk - 1) / out.chunk;
    return out;
}

// Owner hide plan (--prefill-hide-tokens H, --prefill-bank-tokens B): "to avoid the
// RAM issue we split into smaller ones larger than the stream-behind-compute size".
// H = the smallest pass whose compute hides the weight stream (measured 32768 here);
// B = the resident hidden-bank rows; C = the RAM-limited capacity; B' = min(B, C).
//   k = ceil(N / B') near-equal passes. If k > 1 and N/k < H: k-1 passes (folded,
//   each <= C) when they fit, else keep k (fold_refused). A pass is never < H
//   unless N < H or the fold did not fit.
//   Every non-last pass is one 4096-aligned size a: a = floor4096(N/k) (the last
//   pass is the largest) unless the last would exceed the limit (B', or C when
//   folded), then a = ceil4096(N/k); passes must be <= the limit and, where N/k
//   allows, >= H. If neither fits, k grows.
struct hide_plan {
    std::vector<uint32_t> passes;
    uint32_t k = 0, bank = 0;
    bool folded = false, fold_refused = false;
};
inline bool hide_try(uint32_t n, uint32_t k, uint64_t limit, uint32_t h, bool need_h, std::vector<uint32_t> &out, uint32_t align = sweep_align) {
    if (!k) return false;
    if (k == 1) { if (n > limit) return false; out.assign(1, n); return true; }
    const uint64_t down = uint64_t(n) / k / align * align, up = ((uint64_t(n) + k - 1) / k + align - 1) / align * align;
    for (const uint64_t a : {down, up}) {
        if (!a || a > limit || a * (k - 1) >= n) continue;
        const uint64_t last = n - a * (k - 1);
        if (last > limit || (need_h && (a < h || last < h))) continue;
        out.assign(k - 1, uint32_t(a)); out.push_back(uint32_t(last));
        return true;
    }
    return false;
}
inline hide_plan plan_hide(uint32_t n, uint32_t cap, uint32_t h, uint32_t bank, uint32_t align = sweep_align) {
    hide_plan out;
    if (!n || !cap || !h || !bank || !align) return out;
    out.bank = bank;
    const uint32_t bp = std::min(bank, cap);
    const uint32_t k0 = uint32_t((uint64_t(n) + bp - 1) / bp);
    if (k0 > 1 && uint64_t(n) < uint64_t(h) * k0) {   // N/k < H: fold to k-1 passes if each fits C
        if (hide_try(n, k0 - 1, cap, h, n >= h, out.passes, align)) { out.k = k0 - 1; out.folded = true; return out; }
        out.fold_refused = true;
    }
    for (uint32_t k = k0; k <= std::max<uint32_t>(k0, n / align + 1); ++k) {
        const bool need_h = uint64_t(n) >= uint64_t(h) * k;   // passes >= H wherever N/k allows
        if (hide_try(n, k, bp, h, need_h, out.passes, align)) { out.k = k; return out; }
    }
    out.passes.clear();
    return out;
}

// The (first, count) sweep list the producer runs: plan_hide(n, cap, hide, bank)
// when hide is set, else today's split(n, cap). Empty = refused.
struct extent { uint32_t first = 0, count = 0; };
inline std::vector<extent> plan_sweeps(uint32_t n, uint32_t cap, uint32_t hide = 0, uint32_t bank = 0) {
    std::vector<extent> out;
    if (hide) {
        uint32_t first = 0;
        for (const uint32_t count : plan_hide(n, cap, hide, bank ? bank : legacy_sweep_tokens).passes) { out.push_back({first, count}); first += count; }
    } else {
        const auto t = split(n, cap);
        if (t.count) for (uint32_t first = 0; first < n; first += t.chunk) out.push_back({first, std::min(t.chunk, n - first)});
    }
    return out;
}

// Hidden bank strategy for capacity C. bytes(rows) is the exact backend layout
// size of a rows-row bank (monotone in rows). resident: the bank holds all C
// rows (today's plan). spill: bank = the largest tile multiple that fits,
// resident prefix = bank - one tile, the last tile is the staging slot.
struct bank {
    uint32_t rows = 0, resident_rows = 0, slot_rows = 0;
    bool spill = false, fits = false;
};
template <typename Bytes>
bank plan_bank(uint32_t capacity, uint32_t tile, uint64_t admitted, Bytes bytes) {
    bank out;
    if (!capacity || !tile) return out;
    if (bytes(capacity) <= admitted) { out.rows = out.resident_rows = capacity; out.fits = true; return out; }
    uint32_t rows = 0;
    for (uint32_t n = tile; n < capacity && bytes(n) <= admitted; n += tile) rows = n;
    if (rows < 2 * tile) return out;               // a slot needs at least one resident tile beside it
    out.rows = rows; out.slot_rows = tile; out.resident_rows = rows - tile; out.spill = true; out.fits = true;
    return out;
}

} // namespace llama_dsv41_prefill
