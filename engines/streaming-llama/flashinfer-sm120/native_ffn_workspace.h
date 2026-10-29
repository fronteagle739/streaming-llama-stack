// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
namespace slstack_fi_native {
constexpr uint64_t aligned(uint64_t n){return (n+255)/256*256;}
struct workspace_layout {uint64_t activation,scale,gate,up,down,bytes;};
constexpr workspace_layout layout(uint64_t tokens){
    const uint64_t rows=tokens*6,sf_rows=(rows+384*3+127)/128*128;
    const uint64_t scale=aligned(rows*5120);
    const uint64_t gate=scale+aligned(sf_rows*5120/32);
    const uint64_t up=gate+aligned(rows*2304*2);
    const uint64_t end=gate+aligned(rows*5120*2);
    return {0,scale,gate,up,gate,end};
}
constexpr uint64_t scale_offset(int n,int k){return uint64_t(384)*n*k/2;}
static_assert(layout(16384).bytes<(1536ull<<20),"16K routed workspace stays below1.5GiB");
}
