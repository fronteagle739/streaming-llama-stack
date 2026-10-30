// LLAMADECODE_ENGRAM_EARLY: the pure Engram read planner (sorted row ids ->
// 4 KiB-aligned extents, same/adjacent pages merged up to 64 KiB). Exact byte
// recovery is checked against a synthetic file buffer.
#include "llama-dsv41-engram-plan.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ep = llama_dsv41_engram_plan;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static constexpr uint32_t rb = 264;

// Simulate the loader: read every extent from the file, copy each planned row.
static bool recover(const std::vector<uint8_t> &file, const std::vector<uint32_t> &ids, const uint8_t *hits,
                    uint64_t base, uint32_t cap, size_t &extents, uint64_t &bytes, bool &geometry_ok) {
    const auto p = ep::build(ids.data(), ids.size(), hits, base, rb, cap);
    extents = p.extents.size(); bytes = p.bytes; geometry_ok = true;
    std::vector<uint8_t> out(ids.size() * rb, 0), seen(ids.size(), 0), buffer(cap);
    size_t r = 0;
    uint64_t previous_end = 0;
    for (const auto &e : p.extents) {
        if (e.file_offset % 4096 || e.bytes % 4096 || !e.bytes || e.bytes > cap || e.file_offset + e.bytes > file.size() ||
            e.file_offset + 4096 < previous_end) geometry_ok = false; // a capped extent may share its last page with the next
        previous_end = e.file_offset + e.bytes;
        if (!geometry_ok) return false;
        std::memcpy(buffer.data(), file.data() + e.file_offset, e.bytes);
        for (uint32_t k = 0; k < e.rows; ++k, ++r) {
            const auto &row = p.rows[r];
            if (row.index < e.first || row.index > e.last || row.offset + rb > e.bytes) { geometry_ok = false; return false; }
            std::memcpy(out.data() + size_t(row.index) * rb, buffer.data() + row.offset, rb);
            ++seen[row.index];
        }
    }
    if (r != p.rows.size()) { geometry_ok = false; return false; }
    for (size_t i = 0; i < ids.size(); ++i) {
        const bool hit = ep::is_hit(hits, i);
        if (seen[i] != (hit ? 0 : 1)) return false;
        if (!hit && std::memcmp(out.data() + i * rb, file.data() + base + uint64_t(ids[i]) * rb, rb)) return false;
    }
    return true;
}

int main() {
    // A synthetic "model file": an unaligned table base and random bytes.
    const uint64_t base = 1000003;
    const uint32_t table_rows = 60000;
    std::vector<uint8_t> file(((base + uint64_t(table_rows) * rb + 8191) / 4096 + 2) * 4096);
    std::mt19937_64 rng(1234);
    for (auto &b : file) b = uint8_t(rng());
    size_t extents = 0; uint64_t bytes = 0; bool geometry = false;

    {   // Empty table.
        const std::vector<uint32_t> none;
        expect(ep::build(none.data(), 0, nullptr, base, rb).extents.empty(), "empty table: no extents");
    }
    {   // Page-crossing row: base 0, id 15 spans 3960..4224 -> one 8 KiB extent.
        const std::vector<uint32_t> ids{15};
        const auto p = ep::build(ids.data(), ids.size(), nullptr, 0, rb);
        expect(p.extents.size() == 1 && p.extents[0].file_offset == 0 && p.extents[0].bytes == 8192 && p.rows[0].offset == 3960,
               "page-crossing row reads both pages in one 8 KiB extent");
        expect(recover(file, ids, nullptr, base, 65536, extents, bytes, geometry) && geometry, "page-crossing row bytes recovered");
    }
    {   // Duplicates across pages: id 15 twice (crosses a page) and 16.
        const std::vector<uint32_t> ids{15, 15, 16};
        const auto p = ep::build(ids.data(), ids.size(), nullptr, 0, rb);
        expect(p.extents.size() == 1 && p.rows.size() == 3 && p.rows[0].offset == p.rows[1].offset && p.extents[0].bytes == 8192,
               "duplicate ids across a page boundary share one extent and offset");
        expect(recover(file, ids, nullptr, base, 65536, extents, bytes, geometry) && geometry, "duplicate rows recovered exactly");
    }
    {   // PAIRFIX scan geometry: a gap tolerance merges rows across unread pages.
        const std::vector<uint32_t> ids{0, 100, 2000, 3000};   // ~pages 0, 6, 129, 193 past the base
        const auto strict = ep::build(ids.data(), ids.size(), nullptr, base, rb, 1 << 20);
        const auto merged = ep::build(ids.data(), ids.size(), nullptr, base, rb, 1 << 20, ep::page_bytes, 64 * 1024);
        bool exact = merged.rows.size() == ids.size();
        size_t r = 0;
        for (const auto &e : merged.extents) {
            exact = exact && e.file_offset % 4096 == 0 && e.bytes % 4096 == 0 && e.bytes <= (1u << 20);
            for (uint32_t k = 0; k < e.rows; ++k, ++r) {
                const auto &row = merged.rows[r];
                exact = exact && std::memcmp(file.data() + e.file_offset + row.offset, file.data() + base + uint64_t(ids[row.index]) * rb, rb) == 0;
            }
        }
        expect(strict.extents.size() == 4, "gap 0 keeps one extent per separated row");
        expect(merged.extents.size() == 3 && exact, "64 KiB gap merges nearby rows; far rows stay separate; bytes exact");
        const auto through = ep::build(ids.data(), ids.size(), nullptr, base, rb, 1 << 20, ep::page_bytes, 1 << 20);
        expect(through.extents.size() == 1 && through.rows.size() == ids.size(), "1 MiB gap reads straight through up to the cap");
    }
    {   // Adjacent pages merge; a gap does not.
        const std::vector<uint32_t> adjacent{0, 16};   // 0..264 page 0, 4224..4488 page 1
        const auto a = ep::build(adjacent.data(), adjacent.size(), nullptr, 0, rb);
        expect(a.extents.size() == 1 && a.extents[0].bytes == 8192, "adjacent pages merge into one extent");
        const std::vector<uint32_t> gap{0, 100};       // 26400 is page 6
        const auto g = ep::build(gap.data(), gap.size(), nullptr, 0, rb);
        expect(g.extents.size() == 2 && g.extents[0].bytes == 4096 && g.extents[1].file_offset == 24576, "a page gap splits extents");
    }
    {   // Merge cap: 2,000 consecutive rows (528 KB) split into <=64 KiB extents.
        std::vector<uint32_t> ids(2000);
        for (uint32_t i = 0; i < 2000; ++i) ids[i] = 777 + i;
        const auto p = ep::build(ids.data(), ids.size(), nullptr, base, rb, 65536);
        bool capped = true;
        for (const auto &e : p.extents) capped = capped && e.bytes <= 65536;
        expect(capped && p.extents.size() >= 9 && p.extents.size() <= 10, "dense run respects the 64 KiB merge cap");
        expect(recover(file, ids, nullptr, base, 65536, extents, bytes, geometry) && geometry, "dense run bytes recovered exactly");
        expect(recover(file, ids, nullptr, base, 8192, extents, bytes, geometry) && geometry, "dense run recovered at the minimum 8 KiB cap");
    }
    {   // All hits: nothing to read.
        const std::vector<uint32_t> ids{1, 2, 3};
        const uint8_t bits[1]{0x07};
        expect(ep::build(ids.data(), ids.size(), bits, base, rb).extents.empty(), "all rows resident: no extents");
    }
    {   // Unsorted input is refused.
        const std::vector<uint32_t> ids{5, 4};
        bool threw = false;
        try { ep::build(ids.data(), ids.size(), nullptr, base, rb); } catch (const std::invalid_argument &) { threw = true; }
        expect(threw, "unsorted ids are refused");
    }
    // Randomised: sparse (0.2%, Engram 43K density) and dense (30%), with and
    // without random hits, against the synthetic file.
    for (const double density : {0.002, 0.05, 0.3}) {
        for (const bool with_hits : {false, true}) {
            std::vector<uint32_t> ids;
            std::uniform_real_distribution<double> u(0, 1);
            for (uint32_t id = 0; id < table_rows; ++id) if (u(rng) < density) ids.push_back(id);
            if (density > 0.1) ids.insert(ids.begin() + ids.size() / 2, ids[ids.size() / 2]); // a duplicate too
            std::vector<uint8_t> bits((ids.size() + 7) / 8, 0);
            if (with_hits) for (size_t i = 0; i < ids.size(); ++i) if (u(rng) < 0.25) bits[i >> 3] |= uint8_t(1u << (i & 7));
            const bool ok = recover(file, ids, with_hits ? bits.data() : nullptr, base, 65536, extents, bytes, geometry);
            char what[160];
            std::snprintf(what, sizeof what, "random density=%.3f hits=%d rows=%zu extents=%zu bytes=%llu exact recovery", density, int(with_hits),
                          ids.size(), extents, (unsigned long long)bytes);
            expect(ok && geometry, what);
        }
    }
    std::printf(failures ? "FAILURES %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
