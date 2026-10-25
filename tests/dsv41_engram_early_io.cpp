// LLAMADECODE_ENGRAM_EARLY: the early multi-IoRing Engram loader against a
// synthetic table file (CPU + disk only; no model, no GPU). Checks that every
// prompt row read() back from the early cache equals the file bytes and the
// original single-ring loader's cache, across ring/depth settings, a tight
// admission (shrunken arena), per-table readiness and previous-request reuse.
// Usage: dsv41_engram_early_io.exe <synthetic-file-path>   (created if absent)
#include "llama-dsv41-engram-prefetch.h"
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++failures;
}
static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31);
}
static uint8_t byte_at(uint64_t offset) { return uint8_t(mix(offset >> 3) >> (8 * (offset & 7))); }

int main(int argc, char **argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <synthetic-file>\n", argv[0]); return 2; }
    llama_dsv41::engram_layout layout;
    layout.compressed_vocab_size = 129280; layout.pad_id = 0;
    layout.token_map.resize(129280);
    for (uint32_t i = 0; i < 129280; ++i) layout.token_map[i] = i;
    for (size_t l = 0; l < 2; ++l) {
        uint64_t total = 0;
        for (size_t c = 0; c < 24; ++c) { layout.primes[l][c] = 16381 + uint32_t(c) * 2 + uint32_t(l) * 64; total += layout.primes[l][c]; }
        layout.rows[l] = uint32_t(total);
        for (size_t j = 0; j < 4; ++j) layout.multipliers[l][j] = (0x51ed27ull + j * 7919 + l * 104729) | 1;
    }
    std::string why;
    expect(layout.validate(why), "synthetic Engram layout validates");
    const uint64_t base0 = 3 * 4096 + 123, base1 = base0 + uint64_t(layout.rows[0]) * 264 + 777;
    const uint64_t file_bytes = ((base1 + uint64_t(layout.rows[1]) * 264 + 4095) / 4096 + 1) * 4096;
    const std::wstring path(argv[1], argv[1] + std::strlen(argv[1]));
    {   // Create (or reuse) the deterministic file.
        WIN32_FILE_ATTRIBUTE_DATA info{};
        const bool reuse = GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &info) &&
            ((uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow) == file_bytes;
        if (!reuse) {
            HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (f == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "cannot create file\n"); return 2; }
            std::vector<uint8_t> chunk(8 << 20);
            for (uint64_t at = 0; at < file_bytes; at += chunk.size()) {
                const size_t n = size_t(std::min<uint64_t>(chunk.size(), file_bytes - at));
                for (size_t i = 0; i < n; ++i) chunk[i] = byte_at(at + i);
                DWORD wrote = 0;
                if (!WriteFile(f, chunk.data(), DWORD(n), &wrote, nullptr) || wrote != n) { std::fprintf(stderr, "write failed\n"); return 2; }
            }
            CloseHandle(f);
        }
        std::printf("synthetic file bytes=%llu rows=%u+%u reused=%d\n", (unsigned long long)file_bytes, layout.rows[0], layout.rows[1], int(reuse));
    }
    HANDLE verified = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (verified == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "cannot open file\n"); return 2; }
    const std::array<uint64_t, 2> offsets{base0, base1};

    std::mt19937 rng(77);
    auto make_tokens = [&](size_t n) { std::vector<int32_t> t(n); for (auto &x : t) x = int32_t(rng() % 129280); return t; };
    // Every prompt row: compare early cache vs file bytes and vs a reference cache.
    auto check = [&](const llama_dsv41_engram_prompt_cache &c, const llama_dsv41_engram_prompt_cache *ref,
                     const std::vector<int32_t> &tokens, llama_dsv41::engram_history h) {
        std::vector<llama_dsv41::engram_rows> rows(tokens.size());
        if (!llama_dsv41::engram_hash(layout, h, tokens.data(), nullptr, tokens.size(), rows.data(), why)) return false;
        uint8_t got[264], want[264], other[264];
        for (const auto &r : rows) for (uint32_t t = 0; t < 2; ++t) for (uint32_t id : r[t]) {
            const uint64_t off = offsets[t] + uint64_t(id) * 264;
            if (!c.read(t, off, got, 264)) return false;
            for (int i = 0; i < 264; ++i) want[i] = byte_at(off + i);
            if (std::memcmp(got, want, 264)) return false;
            if (ref && (!ref->read(t, off, other, 264) || std::memcmp(got, other, 264))) return false;
        }
        return true;
    };
    const uint64_t admitted = 636485632; // the Studio profile's --engram-prefetch-bytes
    const auto tokens = make_tokens(3000);

    // Reference: today's single-ring loader.
    auto reference = std::make_shared<llama_dsv41_engram_prompt_cache>();
    reference->prepare(layout, tokens.data(), uint32_t(tokens.size()), offsets, admitted);
    reference->load(path.c_str(), verified, file_bytes, [] { return false; });
    expect(check(*reference, nullptr, tokens, {}), "reference loader rows equal file bytes");

    struct setting { const char *rings, *depth, *batch; };
    for (const setting s : {setting{"4", "1024", "256"}, setting{"1", "32", "1"}, setting{"8", "4096", "4096"}, setting{"3", "100", "7"}}) {
        _putenv_s("LLAMADECODE_ENGRAM_RINGS", s.rings); _putenv_s("LLAMADECODE_ENGRAM_DEPTH", s.depth); _putenv_s("LLAMADECODE_ENGRAM_BATCH", s.batch);
        llama_dsv41_engram_prompt_cache c;
        c.prepare_early(layout, tokens.data(), uint32_t(tokens.size()), offsets, admitted, {}, nullptr);
        bool t0 = false, t1 = false;
        std::thread loader([&] { c.load_early(path.c_str(), verified, file_bytes, [] { return false; }); });
        try { c.wait_table(0, [] { return false; }); t0 = true; c.wait_table(1, [] { return false; }); t1 = true; } catch (...) {}
        loader.join();
        char what[160];
        std::snprintf(what, sizeof what, "early loader rings=%s depth=%s batch=%s: tables ready in order, every row equals file and reference",
                      s.rings, s.depth, s.batch);
        expect(t0 && t1 && check(c, reference.get(), tokens, {}) && c.budget() <= admitted, what);
    }
    {   // Paced (overlaps the weight stream): 1 ring x depth 256, unbatched, whatever
        // the full-IOPS env says; same rows as the reference loader.
        llama_dsv41_engram_prompt_cache c;
        c.set_weight_overlap(true);
        c.prepare_early(layout, tokens.data(), uint32_t(tokens.size()), offsets, admitted, {}, nullptr);
        bool t0 = false, t1 = false;
        std::thread loader([&] { c.load_early(path.c_str(), verified, file_bytes, [] { return false; }); });
        try { c.wait_table(0, [] { return false; }); t0 = true; c.wait_table(1, [] { return false; }); t1 = true; } catch (...) {}
        loader.join();
        expect(t0 && t1 && check(c, reference.get(), tokens, {}) && c.budget() <= admitted,
               "paced loader (weight overlap, env 3x100x7 ignored): every row equals file and reference (see rings=1 slots=256 paced=1 in log)");
    }
    _putenv_s("LLAMADECODE_ENGRAM_RINGS", ""); _putenv_s("LLAMADECODE_ENGRAM_DEPTH", ""); _putenv_s("LLAMADECODE_ENGRAM_BATCH", "");
    {   // Sparse prompt (~0.9% density, mostly unmerged 4/8 KiB extents).
        const auto sparse = make_tokens(150);
        llama_dsv41_engram_prompt_cache ref, c;
        ref.prepare(layout, sparse.data(), uint32_t(sparse.size()), offsets, admitted);
        ref.load(path.c_str(), verified, file_bytes, [] { return false; });
        c.prepare_early(layout, sparse.data(), uint32_t(sparse.size()), offsets, admitted, {}, nullptr);
        c.load_early(path.c_str(), verified, file_bytes, [] { return false; });
        expect(check(c, &ref, sparse, {}), "sparse prompt: early rows equal file and reference");
    }
    {   // Tight admission: worst case for this prompt only; the arena shrinks.
        const uint64_t tight = 4ull * 1024 * 1024 + uint64_t(tokens.size()) * 48 * (4 + 264);
        llama_dsv41_engram_prompt_cache c;
        c.prepare_early(layout, tokens.data(), uint32_t(tokens.size()), offsets, tight, {}, nullptr);
        c.load_early(path.c_str(), verified, file_bytes, [] { return false; });
        expect(c.budget() <= tight && check(c, reference.get(), tokens, {}), "tight admission: arena shrinks inside admission, rows exact");
    }
    {   // Restored history tail.
        llama_dsv41::engram_history h; const int32_t tail[3]{11, 22, 33};
        llama_dsv41::engram_rows scratch[3];
        expect(llama_dsv41::engram_hash(layout, h, tail, nullptr, 3, scratch, why), "history tail hashes");
        llama_dsv41_engram_prompt_cache c;
        c.prepare_early(layout, tokens.data(), uint32_t(tokens.size()), offsets, admitted, h, nullptr);
        c.load_early(path.c_str(), verified, file_bytes, [] { return false; });
        expect(check(c, nullptr, tokens, h), "restored history: rows exact");
    }
    {   // Previous-request reuse: an append sharing half its tokens.
        auto first = std::make_shared<llama_dsv41_engram_prompt_cache>();
        first->prepare_early(layout, tokens.data(), uint32_t(tokens.size()), offsets, admitted, {}, nullptr);
        first->load_early(path.c_str(), verified, file_bytes, [] { return false; });
        std::vector<int32_t> next(tokens.begin(), tokens.begin() + 1500);
        const auto fresh = make_tokens(1500); next.insert(next.end(), fresh.begin(), fresh.end());
        llama_dsv41_engram_prompt_cache c;
        c.prepare_early(layout, next.data(), uint32_t(next.size()), offsets, admitted - first->budget(), {}, first);
        c.load_early(path.c_str(), verified, file_bytes, [] { return false; });
        expect(check(c, nullptr, next, {}), "previous-request reuse: copied + read rows exact (see hit_rows in log)");
    }
    {   // Cancellation drains every submitted read and fails wait_table.
        llama_dsv41_engram_prompt_cache c;
        const auto big = make_tokens(20000);
        c.prepare_early(layout, big.data(), uint32_t(big.size()), offsets, admitted, {}, nullptr);
        std::atomic<int> polls{0};
        bool threw = false, wait_threw = false;
        try { c.load_early(path.c_str(), verified, file_bytes, [&] { return ++polls > 3; }); } catch (const std::exception &) { threw = true; }
        try { c.wait_table(1, [] { return false; }); } catch (const std::exception &) { wait_threw = true; }
        expect(threw && wait_threw, "cancel: load throws after draining, table wait reports failure");
    }
    CloseHandle(verified);
    std::printf(failures ? "FAILURES %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
