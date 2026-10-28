#pragma once
#include <stdint.h>
#if defined(__CUDACC__)
#define DSV41_TL_HD __host__ __device__
#else
#define DSV41_TL_HD
#endif
struct ggml_dsv41_tilelang_record { int32_t row, mask; };
static_assert(sizeof(ggml_dsv41_tilelang_record)==8,"Native sparse record ABI");
// Native records contain one selected row and a four-query membership mask.
// Build one official TileLang list:128 causal raw positions, then <=512 selected
// compressed positions. No context-sized host buffer or masked dummy key leaks.
DSV41_TL_HD inline bool ggml_dsv41_tilelang_ids(int32_t *out,uint32_t query,uint32_t queries,
        uint32_t raw_rows,uint32_t raw_min,uint32_t comp_rows,
        const ggml_dsv41_tilelang_record *records,const uint32_t *counts,uint32_t stride){
    if(query>=queries||raw_rows!=queries+127||raw_min>127||!comp_rows)return false;
    for(uint32_t j=0;j<640;++j)out[j]=-1;
    for(uint32_t j=0;j<128;++j)if(query+j>=raw_min)out[j]=int32_t(query+j);
    const uint32_t tile=query/4,bit=1u<<(query%4),n=counts[tile];
    if(n>stride)return false;
    uint32_t written=128;
    for(uint32_t j=0;j<n;++j){const auto r=records[uint64_t(tile)*stride+j];
        if(!(uint32_t(r.mask)&bit))continue;
        if(r.row<0||uint32_t(r.row)>=comp_rows||written==640)return false;
        out[written++]=int32_t(raw_rows+uint32_t(r.row));
    }
    return true;
}
#undef DSV41_TL_HD
