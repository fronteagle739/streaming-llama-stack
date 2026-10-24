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
// Pure arithmetic only; no allocation, CUDA or model access.
#include <cstdint>

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
