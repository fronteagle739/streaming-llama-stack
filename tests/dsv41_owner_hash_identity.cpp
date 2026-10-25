// LLAMADECODE fast owner hash: the native SHA-256 (Windows CNG), the parallel
// double-buffered owner_digest_memo and the in-place BF16->F32 widening give
// byte-identical digests to the portable SHA-256 over the bounded 1 MiB reads.
// Checks: (1) sha256_native == portable sha256 at odd sizes and splits, resumed
// extend + finalize-copy (hex() twice, then extend again), move; (2) the memo on
// hand-made spans of odd sizes, append-only growth by odd amounts, with pinned-
// style caller staging, a too-small staging and no staging: digests == portable
// full re-hash and only appended + bounded bytes are read; (3) a BF16 (and an F32)
// llama_dsv41_cache_storage on the CPU backend filled with random bits: owner
// digests (new path: 2 MiB in-place-widened reads, threads, CNG) == the portable
// SHA-256 of read_native_span's bounded scratch-widened bytes (old path), over a
// rising position ladder (memo extend) in mode 1. Run with the default
// (incremental) and LLAMADECODE_OWNER_VERIFY=full.
#include "llama-dsv41-owner-digest.h"
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
static std::string portable(const void *p, size_t n) { return dsv41_boundary::sha256(p, n); }

static void test_native_sha() {
    std::mt19937_64 rng(11);
    std::vector<uint8_t> data((5u << 20) + 3);
    for (auto &b : data) b = uint8_t(rng());
    bool ok = true;
    for (size_t size : {size_t(0), size_t(1), size_t(3), size_t(63), size_t(64), size_t(65), size_t(1000003),
                        size_t(2u << 20) - 1, size_t(2u << 20), size_t(2u << 20) + 1, data.size()}) {
        dsv41_boundary::sha256_native h; h.update(data.data(), size);
        ok = ok && h.hex() == portable(data.data(), size);
    }
    expect(ok, "native SHA-256 == portable at odd sizes and around the 2 MiB slot");
    ok = true;
    for (size_t split : {size_t(0), size_t(1), size_t(55), size_t(64), size_t(100003), size_t(2u << 20) + 7, data.size()}) {
        dsv41_boundary::sha256_native h; h.update(data.data(), split);
        const auto mid = h.hex(), again = h.hex();
        h.update(data.data() + split, data.size() - split);
        ok = ok && mid == again && mid == portable(data.data(), split) && h.hex() == portable(data.data(), data.size());
    }
    expect(ok, "native SHA-256 resumed: finalize-copy is non-destructive and repeatable, extend == one-shot");
    dsv41_boundary::sha256_native a; a.update("abc", 3);
    dsv41_boundary::sha256_native b(std::move(a)); dsv41_boundary::sha256_native c; c = std::move(b);
    expect(c.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "native SHA-256 known vector after moves");
}

static void test_memo_spans() {
    using sk = ld::state_kind;
    std::mt19937_64 rng(5);
    // Owners: compressed 0..3 and index 0..3 (append-only, memo slots), two bounded ones.
    std::vector<std::pair<sk, uint32_t>> ids = {{sk::compressed, 0}, {sk::compressed, 1}, {sk::compressed, 2}, {sk::compressed, 3},
                                                {sk::index, 0}, {sk::index, 1}, {sk::index, 2}, {sk::index, 3},
                                                {sk::raw_window, 0}, {sk::pool_kv, 1}};
    std::vector<std::vector<uint8_t>> full(ids.size());
    for (auto &v : full) { v.resize((7u << 20) + 13); for (auto &x : v) x = uint8_t(rng()); }
    std::vector<uint64_t> sizes = {1, 0, 3, (2u << 20) + 5, 77, 2u << 20, 1000001, 64, 4099, 3};
    auto plan_at = [&](const std::vector<uint64_t> &sz) {
        ld::wire_plan p; p.frontier = {1u << 20, 0, 1, 0, 0};
        for (size_t i = 0; i < ids.size(); ++i) p.spans.push_back({ids[i].first, ids[i].second, sz[i]});
        return p;
    };
    auto reader = [&](const ld::state_span &s, uint64_t off, void *dst, size_t n) {
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i].first == s.kind && ids[i].second == s.index) {
            if (off + n > s.bytes) return false;
            std::memcpy(dst, full[i].data() + off, n); return true;
        }
        return false;
    };
    for (int arm = 0; arm < 3; ++arm) {
        ld::owner_digest_memo memo;
        std::vector<uint8_t> staging;
        if (arm == 0) { staging.resize(ld::owner_digest_memo::staging_bytes(3)); memo.set_staging(staging.data(), staging.size()); }
        if (arm == 1) { staging.resize(ld::owner_digest_memo::chunk); memo.set_staging(staging.data(), staging.size()); } // too small: own buffer
        auto sz = sizes; bool ok = true, bytes_ok = true; uint32_t pos = 1;
        for (int step = 0; step < 6; ++step) {
            const auto prev = sz;
            if (step) for (size_t i = 0; i < 8; ++i) sz[i] = std::min<uint64_t>(sz[i] + (step * 1000003ull + i * 7) % (3u << 20) + 1, full[i].size());
            auto plan = plan_at(sz); plan.frontier.position = pos += 5;
            uint64_t hashed = 0;
            const auto got = memo.digests(plan, reader, true, &hashed);
            uint64_t want = 0;
            for (size_t i = 0; i < ids.size(); ++i) {
                ok = ok && got[i] == portable(full[i].data(), size_t(sz[i]));
                want += (step && i < 8) ? sz[i] - prev[i] : sz[i];
            }
            bytes_ok = bytes_ok && hashed == want;
        }
        const char *name[] = {"caller staging (3 workers)", "too-small staging (own buffer)", "no staging"};
        expect(ok, (std::string("memo on odd-size spans with append-only growth == portable full hash: ") + name[arm]).c_str());
        expect(bytes_ok, (std::string("memo reads only appended + bounded bytes: ") + name[arm]).c_str());
    }
    {   // a failing read throws and leaves the memo un-armed; the next pass re-hashes in full
        ld::owner_digest_memo memo;
        auto plan = plan_at(sizes);
        memo.digests(plan, reader, true);
        bool threw = false;
        try { memo.digests(plan, [](const ld::state_span &, uint64_t, void *, size_t) { return false; }, true); } catch (const std::exception &) { threw = true; }
        uint64_t hashed = 0; const auto got = memo.digests(plan, reader, true, &hashed);
        uint64_t all = 0; bool ok = true;
        for (size_t i = 0; i < ids.size(); ++i) { all += sizes[i]; ok = ok && got[i] == portable(full[i].data(), size_t(sizes[i])); }
        expect(threw && memo.ready() && ok && hashed == all, "failed read throws; the next pass re-hashes everything and is correct");
    }
}

static std::array<int32_t, 40> native_owners() {
    std::array<int32_t, 40> out{};
    for (uint32_t i = 0; i < 40; ++i) out[i] = i < 2 ? -1 : i < 8 ? 2 : i < 14 ? 8 : i < 20 ? 14 : 20;
    return out;
}
static void fill(ggml_tensor *t, std::mt19937_64 &rng) {
    if (!t) return;
    std::vector<uint8_t> v(ggml_nbytes(t));
    for (auto &x : v) x = uint8_t(rng());
    ggml_backend_tensor_set(t, v.data(), 0, v.size());
}
static void test_cache(bool bf16) {
    const uint32_t ctx = 8192;
    llama_dsv41_cache_storage cache(native_owners(), ctx, ggml_backend_cpu_buffer_type(), 2 * llama_dsv41_cache_storage::required_bytes(ctx), bf16);
    std::mt19937_64 rng(bf16 ? 3 : 4);
    for (uint32_t l = 0; l < 40; ++l) {
        fill(cache.raw(l), rng);
        for (auto get : {&llama_dsv41_cache_storage::compressed, &llama_dsv41_cache_storage::index,
                         &llama_dsv41_cache_storage::pool_kv, &llama_dsv41_cache_storage::pool_score}) {
            try { fill((cache.*get)(l), rng); } catch (const std::exception &) {}
        }
    }
    bool ok = true; std::string error;
    for (uint32_t p : {1u, 3u, 129u, 1001u, 2049u, 4097u, 6001u, 8191u}) {
        ld::wire_frontier f{ctx, p, 1, 0, 0};
        ld::wire_plan plan; ok = ok && ld::make_wire_plan(f, plan, error);
        std::vector<std::string> old;
        for (const auto &s : plan.spans) {   // old path: bounded scratch widening, odd 1 MiB-ish slices
            auto st = dsv41_boundary::sha256_begin(); std::vector<uint8_t> page(1000004);
            for (uint64_t off = 0; off < s.bytes;) {
                const size_t n = size_t(std::min<uint64_t>(page.size(), s.bytes - off));
                ok = ok && cache.read_native_span(s, off, page.data(), n, error);
                dsv41_boundary::sha256_update(st, page.data(), n); off += n;
            }
            old.push_back(dsv41_boundary::sha256_hex(st));
        }
        std::vector<std::string> got;
        ok = ok && cache.owner_digests(f, got, nullptr, error) && got == old;
    }
    expect(ok, (std::string(bf16 ? "BF16" : "F32") + " cache owner digests (new path) == portable SHA of the old bounded reads, position ladder: " + error).c_str());
}

int main() {
    std::printf("LLAMADECODE_OWNER_VERIFY=%s\n", ld::owner_verify_incremental() ? "incremental" : "full");
    try {
        test_native_sha();
        test_memo_spans();
        test_cache(true);
        test_cache(false);
    } catch (const std::exception &e) {
        std::printf("FAIL exception %s\n", e.what()); ++failures;
    }
    std::printf(failures ? "FAILURES %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
