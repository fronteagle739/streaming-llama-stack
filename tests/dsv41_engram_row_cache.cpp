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
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
