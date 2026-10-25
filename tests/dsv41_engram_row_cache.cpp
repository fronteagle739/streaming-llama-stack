// LLAMADECODE_ENGRAM_CACHE_MB: the host-RAM LRU of 264-byte Engram rows.
// Exact LRU order, byte recovery, eviction, the charge bound and a random
// stress run against a reference list+map model.
#include "llama-dsv41-engram-row-cache.h"
#include <cstdio>
#include <list>
#include <random>
#include <unordered_map>

using cache = llama_dsv41_engram_row_cache;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static void row(uint32_t t, uint32_t id, uint8_t *out) { for (uint32_t i = 0; i < cache::row_bytes; ++i) out[i] = uint8_t(t * 131 + id * 7 + i); }

int main() {
    expect(cache::capacity_rows(0) == 0 && cache::capacity_rows(cache::fixed_bytes) == 0, "no rows below the fixed charge");
    expect(cache::capacity_rows(cache::fixed_bytes + 3 * cache::bytes_per_row) == 3, "capacity from bytes");
    for (uint64_t mb : {1ull, 7ull, 64ull, 333ull}) {
        cache c(mb << 20);
        char what[96]; std::snprintf(what, sizeof what, "%llu MiB: footprint %llu <= charge", (unsigned long long)mb, (unsigned long long)c.footprint());
        expect(c.footprint() <= c.charge() && c.charge() == (mb << 20), what);
    }
    {
        cache c(cache::fixed_bytes + 3 * cache::bytes_per_row);
        uint8_t a[264], b[264];
        row(0, 10, a); c.insert(0, 10, a);
        row(1, 10, a); c.insert(1, 10, a);
        row(0, 20, a); c.insert(0, 20, a);
        expect(c.rows() == 3 && c.contains(0, 10) && c.contains(1, 10) && !c.contains(1, 20), "table is part of the key");
        expect(c.lookup(0, 10, b) && (row(0, 10, a), !std::memcmp(a, b, 264)), "lookup returns the exact row");
        // LRU now: (1,10) < (0,20) < (0,10)
        row(1, 30, a); c.insert(1, 30, a);
        expect(!c.contains(1, 10) && c.contains(0, 10) && c.contains(0, 20) && c.evictions() == 1, "least recently used row evicted");
        expect(c.order() == std::vector<uint64_t>{cache::key_of(0, 20), cache::key_of(0, 10), cache::key_of(1, 30)}, "exact LRU order");
        row(0, 20, a); c.insert(0, 20, a);
        expect(c.rows() == 3 && c.evictions() == 1 && c.order().back() == cache::key_of(0, 20), "re-insert only touches");
    }
    {
        // Random stress vs a reference LRU (list + map), with heavy eviction and probe-chain deletes.
        const uint32_t cap = 5000;
        cache c(cache::fixed_bytes + cap * cache::bytes_per_row);
        std::list<uint64_t> ref; std::unordered_map<uint64_t, std::list<uint64_t>::iterator> where;
        std::mt19937_64 rng(7); bool ok = true; uint64_t hits = 0;
        uint8_t a[264], b[264];
        for (int step = 0; step < 400000 && ok; ++step) {
            const uint32_t t = uint32_t(rng() & 1), id = uint32_t(rng() % 12000);
            const uint64_t k = cache::key_of(t, id);
            if (rng() % 3) {
                const bool got = c.lookup(t, id, b);
                ok = got == bool(where.count(k));
                if (got) { ++hits; row(t, id, a); ok = ok && !std::memcmp(a, b, 264); ref.splice(ref.begin(), ref, where[k]); }
            } else {
                row(t, id, a); c.insert(t, id, a);
                if (where.count(k)) ref.splice(ref.begin(), ref, where[k]);
                else { ref.push_front(k); where[k] = ref.begin(); if (ref.size() > cap) { where.erase(ref.back()); ref.pop_back(); } }
            }
            if (step % 50000 == 0) { std::vector<uint64_t> lru(ref.rbegin(), ref.rend()); ok = ok && c.order() == lru; }
        }
        std::vector<uint64_t> lru(ref.rbegin(), ref.rend());
        char what[96]; std::snprintf(what, sizeof what, "400K random ops match the reference LRU (hits=%llu evictions=%llu)", (unsigned long long)hits, (unsigned long long)c.evictions());
        expect(ok && c.order() == lru && c.rows() == cap, what);
    }
    bool threw = false;
    try { cache c(cache::fixed_bytes); } catch (const std::invalid_argument &) { threw = true; }
    expect(threw, "a cache below one row is refused");

    // Default size (CACHE_MB unset): headroom - 256 MiB margin, <= 4 GiB, whole MiB, off below 16 MiB.
    constexpr uint64_t MiB = 1ull << 20;
    expect(cache::default_margin_bytes == 256 * MiB, "default margin is 256 MiB");
    // G3 v1 death (cfg-cp-1m, 6.25 GiB): headroom 221,756,416 took 221,249,536 (0.5 MB slack).
    expect(cache::default_bytes(221756416) == 0, "6.25 GiB producer (headroom 221.8 MB < margin): default cache off");
    // cfg-eg4c (+2 GiB): headroom 2,376,186,880 -> 2,107,751,424 -> 2010 MiB.
    expect(cache::default_bytes(2376186880ull) == 2010 * MiB, "8.25 GiB producer (headroom 2.38 GB): 2010 MiB, 256 MiB left");
    expect(2376186880ull - cache::default_bytes(2376186880ull) >= cache::default_margin_bytes, "slack >= margin");
    // Measured peak RSS over planned_host: max 180,603,904 (91K multi-sweep) < margin.
    expect(180603904ull < cache::default_margin_bytes, "margin covers the worst measured unplanned RSS (180.6 MB)");
    expect(cache::default_bytes(0) == 0 && cache::default_bytes(256 * MiB) == 0, "no headroom / headroom == margin: off");
    expect(cache::default_bytes(256 * MiB + 16 * MiB - 1) == 0, "below 16 MiB after the margin: off");
    expect(cache::default_bytes(256 * MiB + 16 * MiB) == 16 * MiB, "exactly 16 MiB after the margin: 16 MiB");
    expect(cache::default_bytes(256 * MiB + 100 * MiB + 12345) == 100 * MiB, "rounded down to whole MiB");
    expect(cache::default_bytes(64ull << 30) == 4ull << 30, "capped at 4 GiB");
    for (uint64_t h : {0ull, 1ull, 300 * MiB, 777777777ull, 5ull << 30, 100ull << 30}) {
        const uint64_t b = cache::default_bytes(h);
        char what[128]; std::snprintf(what, sizeof what, "headroom %llu: default %llu leaves >= margin, whole MiB", (unsigned long long)h, (unsigned long long)b);
        expect(b == 0 || (b <= h - cache::default_margin_bytes && b % MiB == 0 && b >= 16 * MiB && b <= (4ull << 30)), what);
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
