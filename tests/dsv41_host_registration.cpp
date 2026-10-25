// LLAMADECODE --host-registration: mode parsing, whole-bank segment geometry
// (<=2 GiB, carry-page aligned, exact cover) and the copy-coverage check that
// every per-page 2D carry copy must pass before it is queued.
#include "llama-dsv41-host-registration.h"
#include <cstdio>

namespace hr = llama_dsv41_host_registration;
using namespace ggml_dsv41_copy;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static uint64_t bank(uint32_t rows) { return header_bytes + uint64_t((rows + page_rows - 1) / page_rows) * page_bytes; }

int main() {
    hr::mode m = hr::mode::whole;
    expect(hr::parse("tile", m) && m == hr::mode::tile, "parse tile");
    expect(hr::parse("whole", m) && m == hr::mode::whole, "parse whole");
    expect(hr::parse("tile-async", m) && m == hr::mode::tile_async, "parse tile-async");
    expect(!hr::parse("async", m) && !hr::parse("", m) && !hr::parse("Whole", m), "unknown modes refused");
    expect(std::string(hr::name(hr::mode::tile_async)) == "tile-async" && std::string(hr::name(hr::mode::tile)) == "tile", "names round-trip");

    // 49,152-row bank (today's default sweep): 4,536,143,872 B -> 3 segments.
    const uint64_t b49 = bank(49152);
    expect(b49 == 4536143872ull, "49K bank bytes");
    auto seg = hr::whole_segments(b49);
    expect(seg.size() == 3, "49K bank = 3 segments");
    bool limits = true, aligned = true, contiguous = true;
    for (size_t i = 0; i < seg.size(); ++i) {
        limits &= seg[i].bytes <= hr::segment_limit && seg[i].bytes > 0;
        aligned &= seg[i].offset % 4096 == 0 && (seg[i].offset == 0 || (seg[i].offset - header_bytes) % page_bytes == 0);
        contiguous &= i == 0 ? seg[i].offset == 0 : seg[i].offset == seg[i - 1].end();
    }
    expect(limits, "every segment <= 2 GiB");
    expect(aligned, "segment boundaries on 4 KiB-aligned carry pages");
    expect(contiguous && seg.back().end() == b49, "segments exactly cover the bank");
    expect(hr::copies_covered(seg, b49, 0, 49152), "whole bank: every row copy covered");
    bool all_tiles = true;
    for (uint32_t s = 0; s < 49152; s += 4096) all_tiles &= hr::copies_covered(seg, b49, s, 4096);
    expect(all_tiles, "whole bank: every 4096-row tile covered");
    // A 107,840-row (262K single-sweep) bank: segments still bounded and covering.
    const uint64_t b107 = bank(107840);
    auto seg107 = hr::whole_segments(b107);
    expect(seg107.size() == 5 && seg107.back().end() == b107 && hr::copies_covered(seg107, b107, 0, 107840), "107,840-row bank: 5 segments cover all rows");
    expect(hr::whole_segments(header_bytes).empty(), "empty bank has no segments");

    // Coverage refuses copies outside the registered spans.
    std::vector<hr::span> partial{seg[0], seg[2]};
    const uint32_t seg1_first_row = uint32_t((seg[1].offset - header_bytes) / page_bytes * page_rows);
    expect(!hr::copies_covered(partial, b49, seg1_first_row, 32), "rows in an unregistered segment refused");
    expect(!hr::copies_covered(partial, b49, seg1_first_row - 32, 64), "copy reaching into an unregistered segment refused");
    expect(hr::copies_covered(partial, b49, seg1_first_row - 32, 32), "last page before the gap still covered");
    expect(!hr::copies_covered(seg, b49, 49152 - 16, 32), "rows past the bank refused");
    expect(!hr::copies_covered(seg, b49, 0, 0), "empty copy refused");
    // A span that ends inside a page does not cover that page's records.
    std::vector<hr::span> cut{{0, hr::row_address(31)}};
    expect(!hr::copies_covered(cut, b49, 0, 32) && hr::copies_covered({{0, hr::row_address(31) + record_bytes}}, b49, 0, 32), "record extent checked to its last byte");

    // tile-async units: page-rounded tile spans.
    const auto t0 = hr::rows_span(b49, 0, 4096), t1 = hr::rows_span(b49, 4096, 4096);
    expect(t0.offset == header_bytes && t0.bytes == 128 * page_bytes && t1.offset == t0.end(), "4096-row tiles are adjacent whole pages");
    expect(!hr::overlaps(t0, t1) && hr::overlaps(t0, hr::rows_span(b49, 4080, 32)), "adjacent tiles disjoint; mid-page tile overlaps");
    expect(t0.bytes * 2 == 756023296ull, "two tile spans ~756 MB (proven size)");
    expect(hr::copies_covered({t0}, b49, 0, 4096) && !hr::copies_covered({t0}, b49, 4096, 1), "tile span covers only its own tile");
    expect(hr::rows_span(b49, 49152 - 4096, 4096).end() == b49 && !hr::rows_span(b49, 49152, 1).bytes, "last tile clipped to bank; past-end empty");
    const auto odd = hr::rows_span(b49, 100, 10);
    expect(odd.offset == header_bytes + 3 * page_bytes && odd.bytes == page_bytes && hr::copies_covered({odd}, b49, 100, 10), "unaligned rows round to their page");

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
