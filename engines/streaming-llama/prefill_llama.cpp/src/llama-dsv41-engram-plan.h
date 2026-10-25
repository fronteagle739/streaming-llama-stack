#pragma once
// LLAMADECODE_ENGRAM_EARLY: pure read planner for the whole-prompt Engram row
// cache. Sorted row ids of one table -> 4 KiB-aligned file extents. Rows on the
// same or the next 4 KiB page merge into one extent up to a byte cap (64 KiB).
// Rows marked in an optional hit bitmap are already resident and are skipped.
// No IO, no allocation in next_extent; build() materialises a plan for tests.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llama_dsv41_engram_plan {

constexpr uint32_t page_bytes = 4096;
constexpr uint32_t default_merge_cap = 65536;

struct extent {
    uint64_t file_offset = 0; // 4 KiB aligned
    uint32_t bytes = 0;       // multiple of 4 KiB, <= cap
    uint32_t first = 0;       // first id index in the extent (not a hit)
    uint32_t last = 0;        // last id index in the extent (not a hit), inclusive
    uint32_t rows = 0;        // non-hit rows in [first, last]
};

inline bool is_hit(const uint8_t *bits, size_t i) {
    return bits && ((bits[i >> 3] >> (i & 7)) & 1u);
}

// Byte offset of id index i inside extent e.
inline uint32_t row_offset(const extent &e, const uint32_t *ids, size_t i, uint64_t base, uint32_t row_bytes) {
    return uint32_t(base + uint64_t(ids[i]) * row_bytes - e.file_offset);
}

// Next extent at or after *pos. ids must be non-decreasing (duplicates allowed:
// they map to the same bytes). Returns false when every remaining row is a hit.
inline bool next_extent(const uint32_t *ids, size_t n, const uint8_t *hits, size_t &pos,
                        uint64_t base, uint32_t row_bytes, uint32_t cap, extent &out) {
    if (row_bytes == 0 || row_bytes > page_bytes || cap < 2 * page_bytes || cap % page_bytes)
        throw std::invalid_argument("Engram plan geometry");
    while (pos < n && is_hit(hits, pos)) ++pos;
    if (pos >= n) return false;
    auto span = [&](size_t i, uint64_t &b, uint64_t &e) {
        if (i && ids[i] < ids[i - 1]) throw std::invalid_argument("Engram plan ids must be sorted");
        const uint64_t off = base + uint64_t(ids[i]) * row_bytes;
        b = off & ~uint64_t(page_bytes - 1);
        e = (off + row_bytes + page_bytes - 1) & ~uint64_t(page_bytes - 1);
    };
    uint64_t begin = 0, end = 0;
    span(pos, begin, end);
    out = {};
    out.file_offset = begin;
    out.first = out.last = uint32_t(pos);
    out.rows = 1;
    ++pos;
    while (pos < n) {
        if (is_hit(hits, pos)) {
            if (ids[pos] < ids[pos - 1]) throw std::invalid_argument("Engram plan ids must be sorted");
            ++pos;
            continue;
        }
        uint64_t b = 0, e = 0;
        span(pos, b, e);
        if (b > end) break;                       // a gap: not the same or adjacent page
        const uint64_t grown = std::max(end, e);
        if (grown - begin > cap) break;           // merge cap
        end = grown;
        out.last = uint32_t(pos);
        ++out.rows;
        ++pos;
    }
    out.bytes = uint32_t(end - begin);
    return true;
}

struct row { uint32_t index = 0, offset = 0; };
struct plan {
    std::vector<extent> extents;
    std::vector<row> rows; // per extent, rows in order; extent k owns rows[sum(rows of 0..k-1) ...]
    uint64_t bytes = 0;
};

inline plan build(const uint32_t *ids, size_t n, const uint8_t *hits, uint64_t base, uint32_t row_bytes,
                  uint32_t cap = default_merge_cap) {
    plan p;
    size_t pos = 0;
    extent e;
    while (next_extent(ids, n, hits, pos, base, row_bytes, cap, e)) {
        p.extents.push_back(e);
        p.bytes += e.bytes;
        for (size_t i = e.first; i <= e.last; ++i)
            if (!is_hit(hits, i)) p.rows.push_back({uint32_t(i), row_offset(e, ids, i, base, row_bytes)});
    }
    return p;
}

} // namespace llama_dsv41_engram_plan
