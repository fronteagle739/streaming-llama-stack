#pragma once
#include <algorithm>
#include <cstdint>

namespace llama_dsv41_singlepass {
inline uint64_t hidden_budget(uint64_t base,uint64_t full_cache,uint64_t active_cache,uint64_t required){
    if(required<=base)return required;
    // Subtract before adding so even hostile uint64 inputs cannot wrap.
    return base+std::min(required-base,full_cache>active_cache?full_cache-active_cache:0);
}
// Keep N=1's index/selection shapes while several legacy sweeps share one
// weight pass. The final <=12K tail belongs to the preceding 49K sweep.
// `end` is the absolute end of this grouped pass, not the cache capacity.
inline uint32_t frontier(uint32_t start,uint32_t end) {
    constexpr uint32_t primary=49152,tail=12288;
    if(start>=end)return 0;
    const uint64_t next=(uint64_t(start)/primary+1)*primary;
    return next>=end||uint64_t(end)-next<=tail?end:uint32_t(next);
}
}
