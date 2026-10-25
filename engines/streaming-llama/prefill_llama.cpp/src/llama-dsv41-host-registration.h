#pragma once
// LLAMADECODE host registration of the fixed RAM carry bank (no CUDA here; pure geometry).
//  tile       : today's path, the active tile span is (re)registered per bound tile.
//  whole      : the whole bank registered once per sweep in carry-page-aligned segments.
//  tile-async : at most two page-rounded tile spans; the next one registers on a helper thread.
// Registration never changes a copied byte: the same 2D copies run in every mode.
#include "../ggml/include/ggml-dsv41-carry-copy.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llama_dsv41_host_registration {
// Whole mode only (Windows): raise the process working-set minimum/maximum by `extra` bytes before
// registering, so CUDA's page locking of the (already resident, VirtualLock'ed) carry bank has quota.
// No new physical RAM is consumed: the same pages stay resident. Returns false (and the Win32 error)
// when Windows refuses; restore_working_set() undoes the raise. Implemented in llama-dsv41-carry.cpp.
bool raise_working_set(uint64_t extra, uint64_t &min_before, uint64_t &min_after, unsigned long &error);
void restore_working_set();
enum class mode : uint8_t { tile = 0, whole = 1, tile_async = 2 };
inline bool parse(const std::string &s, mode &m) {
    if (s == "tile") { m = mode::tile; return true; }
    if (s == "whole") { m = mode::whole; return true; }
    if (s == "tile-async") { m = mode::tile_async; return true; }
    return false;
}
inline const char *name(mode m) { return m == mode::whole ? "whole" : m == mode::tile_async ? "tile-async" : "tile"; }

struct span {
    uint64_t offset = 0, bytes = 0;
    uint64_t end() const { return offset + bytes; }
    bool operator==(const span &o) const { return offset == o.offset && bytes == o.bytes; }
};
constexpr uint64_t segment_limit = 2ull * 1024 * 1024 * 1024;

inline uint64_t row_address(uint32_t row) {
    using namespace ggml_dsv41_copy;
    return header_bytes + uint64_t(row / page_rows) * page_bytes + uint64_t(row % page_rows) * record_bytes;
}
// Rows the bank geometry can address (whole 32-row pages after the header).
inline uint64_t bank_rows(uint64_t bank_bytes) {
    using namespace ggml_dsv41_copy;
    return bank_bytes > header_bytes ? (bank_bytes - header_bytes) / page_bytes * page_rows : 0;
}
// Whole-bank segments: each <= limit and every boundary on a carry page boundary
// (header + k*page_bytes, a 4 KiB multiple), so no single page copy straddles two.
inline std::vector<span> whole_segments(uint64_t bank_bytes, uint64_t limit = segment_limit) {
    using namespace ggml_dsv41_copy;
    std::vector<span> out;
    if (bank_bytes <= header_bytes || limit < header_bytes + page_bytes) return out;
    const uint64_t pages = (limit - header_bytes) / page_bytes;
    uint64_t at = 0, end = header_bytes + pages * page_bytes;
    while (at < bank_bytes) {
        const uint64_t stop = end < bank_bytes ? end : bank_bytes;
        out.push_back({at, stop - at});
        at = stop; end = at + pages * page_bytes;
    }
    return out;
}
// Page-rounded span of rows [start, start+count): starts at its first carry page,
// ends at the next page boundary (clipped to the bank). Empty when outside the bank.
inline span rows_span(uint64_t bank_bytes, uint32_t start, uint32_t count) {
    using namespace ggml_dsv41_copy;
    if (!count || uint64_t(start) + count > bank_rows(bank_bytes)) return {};
    const uint64_t first = header_bytes + uint64_t(start / page_rows) * page_bytes;
    const uint64_t last_page = (uint64_t(start) + count - 1) / page_rows;
    uint64_t end = header_bytes + (last_page + 1) * page_bytes;
    if (end > bank_bytes) end = bank_bytes;
    return {first, end - first};
}
inline bool overlaps(const span &a, const span &b) { return a.bytes && b.bytes && a.offset < b.end() && b.offset < a.end(); }
inline bool inside(const span &outer, uint64_t begin, uint64_t end) { return outer.bytes && begin >= outer.offset && end <= outer.end() && begin <= end; }
// Every per-page 2D copy that ggml_dsv41_cuda_copy_ram issues for rows [start,start+count)
// (full record extent) must lie wholly inside ONE registered span of the bank.
inline bool copies_covered(const std::vector<span> &registered, uint64_t bank_bytes, uint32_t start, uint32_t count) {
    using namespace ggml_dsv41_copy;
    if (!count || uint64_t(start) + count > maximum_rows) return false;
    for (uint32_t done = 0; done < count;) {
        const uint32_t row = start + done;
        uint32_t rows = page_rows - row % page_rows;
        if (rows > count - done) rows = count - done;
        const uint64_t begin = row_address(row), end = row_address(row + rows - 1) + record_bytes;
        if (end > bank_bytes) return false;
        bool ok = false;
        for (const auto &s : registered) if (inside(s, begin, end)) { ok = true; break; }
        if (!ok) return false;
        done += rows;
    }
    return true;
}
} // namespace llama_dsv41_host_registration
