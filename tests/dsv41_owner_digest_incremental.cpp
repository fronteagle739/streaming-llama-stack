// LLAMADECODE incremental owner verify: the producer's owner_sha256 values stay
// the plain SHA-256 of each canonical span (wire unchanged); compressed/index
// owners are extended over appended rows from a saved SHA-256 running state.
// Checks: (1) resumable SHA-256 == one-shot SHA-256 at arbitrary splits;
// (2) on synthetic owners across a decode-delta ladder (mode1 and mode3, odd and
// even positions, sub-128 and >128 deltas) the incremental digests equal a full
// re-hash and an independent SHA-256 of every span, while reading only the
// appended rows plus the bounded ring/pool owners; (3) a corrupted appended row
// is detected; (4) a stale memo (rewind without invalidate) is wrong and
// invalidate() fixes it; (5) through llama_dsv41_cache_storage on the CPU
// backend: restore + apply_native_delta ladder verifies incrementally, the
// inspect digests equal the canonical full hashes, a corrupted appended row is
// refused, and an `expected` mismatch falls back to a full re-hash.
// Run twice: default (incremental) and LLAMADECODE_OWNER_VERIFY=full.
#include "llama-dsv41-owner-digest.h"
#include "llama-dsv41-delta.h"
#include "llama-dsv41-cache.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

namespace ld = llama_dsv41;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++failures;
}
static constexpr uint32_t vocab = 129280;
static uint64_t row_bytes(ld::state_kind k) {
    switch (k) {
        case ld::state_kind::raw_window: return 512u * 4u;
        case ld::state_kind::dspark_hidden: return 3u * 5120u * 4u;
        case ld::state_kind::compressed: return 512u * 4u;
        case ld::state_kind::index: return 128u * 4u;
        default: return 0;
    }
}
static uint8_t mix(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ c * 0x94D049BB133111EBull ^ d;
    x ^= x >> 31; x *= 0xD6E8FEB86659FD93ull; x ^= x >> 29;
    return uint8_t(x);
}
// Canonical synthetic state at position p (same model as dsv41_decode_delta_big):
// ring slot r = latest q < p with q%128==r; compressed/index rows fixed once
// written; pool tails depend on the position. `salt` makes a different history.
static void canonical(uint32_t p, const ld::state_span &s, uint64_t off, void *dst, size_t n, uint64_t salt = 0) {
    auto *o = static_cast<uint8_t *>(dst);
    const uint64_t rb = row_bytes(s.kind);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t at = off + i;
        uint64_t key = 0;
        if (s.kind == ld::state_kind::raw_window || s.kind == ld::state_kind::dspark_hidden) {
            const uint64_t slot = at / rb;
            key = (p - 1) - ((p - 1 + 128 - slot) % 128);
        } else if (s.kind == ld::state_kind::compressed || s.kind == ld::state_kind::index) {
            key = at / rb;
        } else {
            key = p;
        }
        o[i] = mix(uint64_t(s.kind) * 64 + s.index, key + salt, at % (rb ? rb : 1u << 20), rb ? 0 : at);
    }
}
static ld::wire_frontier frontier(uint32_t ctx, uint32_t p, uint32_t mode) { return {ctx, p, mode, mode == 3 ? p : 0, mode == 3 ? std::min(p, 128u) : 0}; }
static ld::wire_plan plan_of(const ld::wire_frontier &f) {
    ld::wire_plan p; std::string e; if (!ld::make_wire_plan(f, p, e)) throw std::runtime_error(e); return p;
}
using key = std::pair<int, uint32_t>;
struct store {
    std::map<key, std::vector<uint8_t>> spans;
    ld::owner_digest_memo::reader reader() {
        return [this](const ld::state_span &s, uint64_t off, void *dst, size_t n) {
            auto it = spans.find({int(s.kind), s.index});
            if (it == spans.end() || off > it->second.size() || n > it->second.size() - off || s.bytes != it->second.size()) return false;
            std::memcpy(dst, it->second.data() + off, n); return true;
        };
    }
};
// Device state at p: compressed/index prefix rows are kept (append-only),
// the rest rebuilt; so a salted history below the old live rows survives.
static void advance(store &st, const ld::wire_plan &plan, uint64_t salt = 0) {
    std::map<key, std::vector<uint8_t>> next;
    for (const auto &sp : plan.spans) {
        auto &v = next[{int(sp.kind), sp.index}];
        v.resize(size_t(sp.bytes));
        canonical(plan.frontier.position, sp, 0, v.data(), v.size(), salt);
        auto old = st.spans.find({int(sp.kind), sp.index});
        if ((sp.kind == ld::state_kind::compressed || sp.kind == ld::state_kind::index) && old != st.spans.end())
            std::memcpy(v.data(), old->second.data(), std::min(old->second.size(), v.size()));
    }
    st.spans = std::move(next);
}
static std::vector<std::string> independent(const store &st, const ld::wire_plan &plan) {
    std::vector<std::string> out;
    for (const auto &sp : plan.spans) { const auto &v = st.spans.at({int(sp.kind), sp.index}); out.push_back(dsv41_boundary::sha256(v.data(), v.size())); }
    return out;
}
static std::vector<std::string> canonical_hashes(const ld::wire_plan &plan) {
    std::vector<std::string> out;
    for (const auto &sp : plan.spans) {
        std::vector<uint8_t> v(size_t(sp.bytes)); canonical(plan.frontier.position, sp, 0, v.data(), v.size());
        out.push_back(dsv41_boundary::sha256(v.data(), v.size()));
    }
    return out;
}
static uint64_t bounded_bytes(const ld::wire_plan &plan) {
    uint64_t b = 0;
    for (const auto &s : plan.spans) if (s.kind != ld::state_kind::compressed && s.kind != ld::state_kind::index) b += s.bytes;
    return b;
}

static void test_resumable_sha() {
    std::mt19937_64 rng(7);
    std::vector<uint8_t> data(300000);
    for (auto &b : data) b = uint8_t(rng());
    bool ok = true;
    for (size_t split : {size_t(0), size_t(1), size_t(55), size_t(63), size_t(64), size_t(65), size_t(2048), size_t(100003), data.size()}) {
        auto st = dsv41_boundary::sha256_begin();
        dsv41_boundary::sha256_update(st, data.data(), split);
        const auto mid = dsv41_boundary::sha256_hex(st); // finalizing a copy must not disturb the state
        dsv41_boundary::sha256_update(st, data.data() + split, data.size() - split);
        ok = ok && dsv41_boundary::sha256_hex(st) == dsv41_boundary::sha256(data.data(), data.size())
                && mid == dsv41_boundary::sha256(data.data(), split);
    }
    expect(ok, "resumable SHA-256 equals one-shot SHA-256 at every split (and mid-finalize is non-destructive)");
    auto st = dsv41_boundary::sha256_begin(); dsv41_boundary::sha256_update(st, "abc", 3);
    expect(dsv41_boundary::sha256_hex(st) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "resumable SHA-256 known vector");
}

static void test_memo_ladder(uint32_t mode) {
    const uint32_t ctx = 262144;
    store st; ld::owner_digest_memo memo, fresh;
    uint32_t p = 25001;
    auto plan = plan_of(frontier(ctx, p, mode)); advance(st, plan);
    uint64_t hashed = 0;
    auto first = memo.digests(plan, st.reader(), true, &hashed);
    bool ok = first == independent(st, plan);
    bool small = true, equal = true;
    for (uint32_t step : {1u, 1u, 2u, 31u, 92u, 127u, 128u, 129u, 300u, 4096u, 3u}) {
        const auto base = plan;
        p += step; plan = plan_of(frontier(ctx, p, mode));
        ld::delta_plan dp; std::string e;
        if (!ld::make_delta_plan(base.frontier, plan.frontier, dp, e)) { expect(false, e.c_str()); return; }
        advance(st, plan);
        uint64_t got_bytes = 0;
        const auto inc = memo.digests(plan, st.reader(), true, &got_bytes);
        const auto full = fresh.digests(plan, st.reader(), false);
        equal = equal && inc == full && inc == independent(st, plan) && inc == canonical_hashes(plan);
        // appended compressed/index bytes + bounded owners only
        uint64_t appended = 0;
        for (const auto &s : dp.slices) if (s.span.kind == ld::state_kind::compressed || s.span.kind == ld::state_kind::index) appended += s.bytes;
        small = small && got_bytes == appended + bounded_bytes(plan);
    }
    char what[160];
    std::snprintf(what, sizeof what, "mode%u: incremental digests == full re-hash == independent SHA-256 over the delta ladder", mode); expect(ok && equal, what);
    std::snprintf(what, sizeof what, "mode%u: incremental reads exactly the appended compressed/index rows + bounded ring/pool owners", mode); expect(small, what);
}

// The realistic chain: one memo, verify base then target on every step.
static void test_memo_chain() {
    const uint32_t ctx = 262144;
    store st; ld::owner_digest_memo memo;
    uint32_t p = 24000;
    auto plan = plan_of(frontier(ctx, p, 1)); advance(st, plan);
    bool ok = memo.digests(plan, st.reader(), true) == canonical_hashes(plan);
    uint64_t base_compressed_reads = 0;
    for (uint32_t step : {32u, 63u, 84u, 92u, 67u, 128u, 128u, 1u, 7u}) {
        const auto base = plan; const auto base_hashes = canonical_hashes(base);
        // base verify on the unchanged store (before the delta is written)
        uint64_t n = 0; ok = ok && memo.digests(base, st.reader(), true, &n) == base_hashes;
        base_compressed_reads += n - bounded_bytes(base);
        p += step; plan = plan_of(frontier(ctx, p, 1)); advance(st, plan);
        ok = ok && memo.digests(plan, st.reader(), true) == canonical_hashes(plan);
    }
    expect(ok, "chained base+target verification on one memo equals canonical full hashes");
    expect(base_compressed_reads == 0, "base verification re-reads no compressed/index rows");
}

static void test_corruption() {
    const uint32_t ctx = 262144;
    store st; ld::owner_digest_memo memo;
    uint32_t p = 25000;
    auto plan = plan_of(frontier(ctx, p, 1)); advance(st, plan);
    memo.digests(plan, st.reader(), true);
    p += 92; plan = plan_of(frontier(ctx, p, 1)); advance(st, plan);
    const auto want = canonical_hashes(plan);
    // flip one byte in the LAST appended compressed row of owner 3 and one in an index row of owner 0
    auto &c3 = st.spans.at({int(ld::state_kind::compressed), 3u}); c3[c3.size() - 100] ^= 0x01;
    auto got = memo.digests(plan, st.reader(), true);
    size_t mism = 0; for (size_t i = 0; i < got.size(); ++i) mism += got[i] != want[i];
    expect(mism == 1, "a corrupted appended compressed row is detected (exactly that owner differs)");
    c3[c3.size() - 100] ^= 0x01;
    // re-arm at the previous position, corrupt the first appended index row
    memo.invalidate();
    const auto prev = plan_of(frontier(ctx, p - 92, 1));
    memo.digests(prev, [&](const ld::state_span &s, uint64_t off, void *dst, size_t n) { canonical(prev.frontier.position, s, off, dst, n); return true; }, true);
    auto &i0 = st.spans.at({int(ld::state_kind::index), 0u}); const size_t first_new = size_t((p - 92) / 2) * 512u;
    i0[first_new + 3] ^= 0x80;
    got = memo.digests(plan, st.reader(), true);
    mism = 0; for (size_t i = 0; i < got.size(); ++i) mism += got[i] != want[i];
    expect(mism == 1, "a corrupted first appended index row is detected");
}

static void test_stale_memo() {
    const uint32_t ctx = 262144;
    store st; ld::owner_digest_memo memo;
    uint32_t p = 20000;
    auto plan = plan_of(frontier(ctx, p, 1)); advance(st, plan);
    memo.digests(plan, st.reader(), true);
    // rewind to 19900 (compressed/index truncate) then a DIFFERENT history to 20050
    store other; advance(other, plan_of(frontier(ctx, 19900, 1)));
    for (auto &kv : other.spans) { auto it = st.spans.find(kv.first); if (it != st.spans.end() && (kv.first.first == int(ld::state_kind::compressed) || kv.first.first == int(ld::state_kind::index))) kv.second.assign(it->second.begin(), it->second.begin() + kv.second.size()); }
    plan = plan_of(frontier(ctx, 20050, 1)); advance(other, plan, 0x5151);
    const auto truth = independent(other, plan);
    const auto stale = memo.digests(plan, other.reader(), true);
    expect(stale != truth, "a memo kept across a rewind is stale (why rewind/restore/clear invalidate it)");
    memo.invalidate();
    expect(memo.digests(plan, other.reader(), true) == truth, "after invalidate() the digests equal a full re-hash");
}

// ---- llama_dsv41_cache_storage on the CPU backend -------------------------
static std::array<int32_t, 40> native_owners() {
    std::array<int32_t, 40> out{};
    for (uint32_t i = 0; i < 40; ++i) out[i] = i < 2 ? -1 : i < 8 ? 2 : i < 14 ? 8 : i < 20 ? 14 : 20;
    return out;
}
static void test_cache() {
    const uint32_t ctx = 8192;
    llama_dsv41_cache_storage cache(native_owners(), ctx, ggml_backend_cpu_buffer_type(), 2 * llama_dsv41_cache_storage::required_bytes(ctx));
    std::string error;
    uint32_t p = 3001;
    auto source = [&](uint32_t pos, int corrupt_kind = -1) {
        return [pos, corrupt_kind](const ld::state_span &s, uint64_t off, void *dst, size_t n) {
            canonical(pos, s, off, dst, n);
            if (corrupt_kind >= 0 && int(s.kind) == corrupt_kind && s.index == 2) {
                const uint64_t at = s.bytes - 8; // last appended row
                if (at >= off && at < off + n) static_cast<uint8_t *>(dst)[at - off] ^= 0x10;
            }
            return true;
        };
    };
    auto f = frontier(ctx, p, 1);
    bool ok = cache.restore_native_frontier(f, canonical_hashes(plan_of(f)), source(p), error);
    expect(ok, ("cache restore verifies all owners: " + error).c_str());
    std::vector<std::string> got;
    ok = cache.owner_digests(f, got, nullptr, error) && got == canonical_hashes(plan_of(f));
    expect(ok, "cache inspect digests after restore == canonical full hashes");
    bool ladder = true;
    for (uint32_t step : {1u, 32u, 92u, 128u, 129u, 2000u, 3u}) {
        const auto base = frontier(ctx, p, 1); p += step; const auto target = frontier(ctx, p, 1);
        ladder = ladder && cache.apply_native_delta(base, target, canonical_hashes(plan_of(base)), canonical_hashes(plan_of(target)), source(p), error);
        ladder = ladder && cache.owner_digests(target, got, nullptr, error) && got == canonical_hashes(plan_of(target));
    }
    expect(ladder, ("cache apply_native_delta ladder verifies base+target; inspect == canonical: " + error).c_str());
    {
        const auto base = frontier(ctx, p, 1); const auto target = frontier(ctx, p + 40, 1);
        const bool refused = !cache.apply_native_delta(base, target, canonical_hashes(plan_of(base)), canonical_hashes(plan_of(target)),
            source(p + 40, int(ld::state_kind::compressed)), error);
        expect(refused && error.find("SHA256 mismatch") != std::string::npos, ("cache refuses a corrupted appended compressed row: " + error).c_str());
        std::vector<std::string> none;
        expect(!cache.owner_digests(target, none, nullptr, error), "poisoned cache refuses owner digests");
    }
    // fresh state for the fallback check
    cache.clear(); p = 3001; f = frontier(ctx, p, 1);
    ok = cache.restore_native_frontier(f, canonical_hashes(plan_of(f)), source(p), error);
    const auto base = f; p += 50; f = frontier(ctx, p, 1);
    ok = ok && cache.apply_native_delta(base, f, canonical_hashes(plan_of(base)), canonical_hashes(plan_of(f)), source(p), error);
    // rewrite one PREFIX row behind the memo (owner 0 compressed row 10)
    ggml_tensor *t = cache.compressed(2);
    std::vector<float> row(512, 0.0f); row[5] = 1.25f;
    ggml_backend_tensor_set(t, row.data(), 10 * 512 * 4, 512 * 4);
    auto expected = canonical_hashes(plan_of(f));
    {
        const auto pl = plan_of(f);
        for (size_t i = 0; i < pl.spans.size(); ++i) if (pl.spans[i].kind == ld::state_kind::compressed && pl.spans[i].index == 0) {
            std::vector<uint8_t> v(size_t(pl.spans[i].bytes)); canonical(p, pl.spans[i], 0, v.data(), v.size());
            std::memcpy(v.data() + 10 * 2048, row.data(), 2048); expected[i] = dsv41_boundary::sha256(v.data(), v.size());
        }
    }
    ok = ok && cache.owner_digests(f, got, &expected, error);
    expect(ok && got == expected, "an `expected` mismatch falls back to a full re-hash that sees the real rows");
    ok = cache.owner_digests(f, got, nullptr, error) && got == expected;
    expect(ok, "after the fallback the memo carries the re-hashed digests");
}

int main() {
    std::printf("LLAMADECODE_OWNER_VERIFY=%s\n", ld::owner_verify_incremental() ? "incremental" : "full");
    try {
        test_resumable_sha();
        test_memo_ladder(1);
        test_memo_ladder(3);
        test_memo_chain();
        test_corruption();
        test_stale_memo();
        test_cache();
    } catch (const std::exception &e) {
        std::printf("FAIL exception %s\n", e.what()); ++failures;
    }
    std::printf(failures ? "FAILURES %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
