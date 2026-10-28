#pragma once
#include <cstdint>

// ABI 3 workspace offsets. One caller-owned allocation is reused across all
// products; no CUDA allocation or default memory-pool policy changes in the DLL.
namespace ggml_dsv41_flashinfer {
constexpr uint64_t align256(uint64_t bytes) { return (bytes+255)/256*256; }
struct workspace_layout {
    uint64_t weight, weight_scale, activation, activation_scale, output, bytes;
};
constexpr workspace_layout layout(uint64_t rows, uint64_t n, uint64_t k) {
    const uint64_t sf_rows=(rows+384*3+127)/128*128;
    const uint64_t sb=align256(384*n*k/2);
    const uint64_t ab=sb+align256(384*n*k/32);
    const uint64_t sa=ab+align256(rows*k);
    const uint64_t ob=sa+align256(sf_rows*k/32);
    return {0,sb,ab,sa,ob,ob+align256(rows*n*2)};
}
// Fixed maximum prevents differently sized products accumulating cached large
// allocations in GGML's legacy pool. Its 5% allocation rounding is admitted too.
constexpr uint64_t workspace_bytes=layout(4096*6,5120,2304).bytes;
static_assert(workspace_bytes>=layout(4096*6,2304,5120).bytes,"Both projection shapes fit");
}
