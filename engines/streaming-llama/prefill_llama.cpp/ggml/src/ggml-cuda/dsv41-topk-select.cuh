#pragma once
// LLAMADECODE fast exact top-512 select (default OFF; producer --topk-kernel fast).
//
// Replaces ONLY the multi-pass large-input selector (ggml_cuda_op_top_k, nrows>1,
// k==512, ncols>8192: 8192-column tile sorts + merge passes). For every row whose
// scores contain no NaN and no subnormal, that selector returns the top 512 under
// the total order (value descending, index ascending), with -0 == +0, sorted in
// that order (packed key = ordered(value)<<32 | ~index, see dsv41_topk_pack_key).
// The fast kernel computes the same set and order with a 4 x 8-bit radix select
// on the ordered value bits plus one index-ordered collection pass, then sorts the
// 512 winners by the same packed key. Rows with any NaN or subnormal (where the
// current selector falls back to its original comparison network under FTZ) are
// flagged and recomputed by the unchanged current pipeline, row-filtered.
//
// Shared by the CUDA kernel and the CPU test (tests/dsv41_topk_fast_cpu.cpp):
// integer-only helpers, so -use_fast_math/FTZ cannot change them.
#include <stdint.h>
#ifdef __CUDACC__
#define DSV41_TOPK_HD __host__ __device__ __forceinline__
#else
#define DSV41_TOPK_HD inline
#endif

namespace dsv41_topk_select {
constexpr int k = 512;
constexpr int min_cols = 8193;  // the selector it replaces runs only above one 8192 tile

// 0 clean, 1 NaN, 2 subnormal: the current radix kernel's original-network triggers.
DSV41_TOPK_HD unsigned classify(uint32_t bits) {
    const uint32_t m = bits & 0x7fffffffu;
    return m > 0x7f800000u ? 1u : ((m && m < 0x00800000u) ? 2u : 0u);
}
// The value half of dsv41_topk_pack_key: +-0 canonicalized, order-preserving.
DSV41_TOPK_HD uint32_t ordered(uint32_t bits) {
    if (!(bits & 0x7fffffffu)) bits = 0;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
DSV41_TOPK_HD uint64_t pack(uint32_t ord, int id) {
    return (uint64_t(ord) << 32) | uint64_t(0xffffffffu - uint32_t(id));
}
DSV41_TOPK_HD int unpack_id(uint64_t key) { return int(0xffffffffu - uint32_t(key)); }

// Radix-select state: the k-th largest ordered value has bits `prefix` under
// `mask`; `need` winners remain to be taken from that region (index order).
struct state { uint32_t prefix, mask, need; };
// Region: ordered values whose resolved bits equal the prefix. Above: strictly greater.
DSV41_TOPK_HD bool in_region(const state &s, uint32_t o) { return (o & s.mask) == s.prefix; }
DSV41_TOPK_HD bool above(const state &s, uint32_t o) { return (o & s.mask) > s.prefix; }
// After the 256-bin digit histogram of the region at `shift`, the chosen bin
// holds `in_bin` elements and bins above it hold `above_bin` (above_bin < need
// <= above_bin + in_bin). Returns true when the selection is fully determined:
// every element of the new region is a winner, or all 32 bits are resolved
// (one value left; ties then go to the lowest indices).
DSV41_TOPK_HD bool advance(state &s, int shift, uint32_t bin, uint32_t above_bin, uint32_t in_bin) {
    s.need -= above_bin;
    s.prefix |= bin << shift;
    s.mask |= 0xffu << shift;
    return in_bin == s.need || shift == 0;
}
}  // namespace dsv41_topk_select
