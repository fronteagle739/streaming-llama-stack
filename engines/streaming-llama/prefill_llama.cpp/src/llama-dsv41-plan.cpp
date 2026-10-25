#include "llama-dsv41-plan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

llama_dsv41_tile_plan::llama_dsv41_tile_plan(uint32_t start, uint32_t count, uint32_t context, uint32_t ratio, uint32_t admitted_tile_tokens) :
    llama_dsv41_tile_plan(start,count,context,ratio,admitted_tile_tokens,false) {}

llama_dsv41_tile_plan::llama_dsv41_tile_plan(uint32_t start,uint32_t count,uint32_t context,uint32_t ratio,uint32_t admitted_tile_tokens,bool device_raw_template) :
    start(start), count(count), context(context), ratio(ratio), raw_start(start) {
    if ((admitted_tile_tokens!=32&&admitted_tile_tokens!=128&&admitted_tile_tokens!=512&&admitted_tile_tokens!=2048&&admitted_tile_tokens!=4096&&admitted_tile_tokens!=8192&&admitted_tile_tokens!=16384)||
        !context || context > 1048576 || !count || count > admitted_tile_tokens || start >= context || count > context - start || ratio > 2)
        throw std::runtime_error("V4.1 requires a contiguous tile within its explicit count/context admission, ratio 0/1/2");
    history_rows = std::min(start, 127u);
    raw_rows = std::min(start + count, 128u);
    positions.resize(count); history_ids.resize(history_rows);
    if(device_raw_template && !(start>=128&&start%128==0&&raw_rows==128&&history_rows==127))
        throw std::runtime_error("Device raw template requires exact steady aligned128-row window");
    if(!device_raw_template)raw_ids.assign(size_t(raw_rows) * count, 0);
    if(!device_raw_template)raw_mask.assign(size_t(raw_rows) * count, -std::numeric_limits<float>::infinity());
    const uint32_t first_history = start - history_rows;
    for (uint32_t h = 0; h < history_rows; ++h) history_ids[h] = int32_t((first_history + h) % 128);
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t pos = start + t, first = pos >= 127 ? pos - 127 : 0;
        positions[t] = int32_t(pos);
        if(!device_raw_template)for (uint32_t p = first; p <= pos; ++p) {
            const size_t at = size_t(t) * raw_rows + p - first;
            raw_ids[at] = int32_t(p - first_history); raw_mask[at] = 0.f;
        }
    }
    previous_pool_tail = ratio == 2 && start % 2;
    next_pool_tail = ratio == 2 && (start + count) % 2;
    published_first = ratio ? start / ratio : 0;
    published_count = ratio ? (start + count) / ratio - published_first : 0;
    compressed_positions.resize(published_count);
    compressed_visible.resize(count);
    for (uint32_t row = 0; row < published_count; ++row) compressed_positions[row] = int32_t((published_first + row) * ratio);
    for (uint32_t t = 0; t < count; ++t) compressed_visible[t] = ratio ? int32_t((start + t + 1) / ratio) : 0;
}

void llama_dsv41_tile_plan::restrict_raw(uint32_t begin,uint32_t admission){
    if(begin>start)throw std::runtime_error("YOCO raw lower bound exceeds query start");
    llama_dsv41_tile_plan bounded(start-begin,count,context,ratio,admission);
    raw_start=start-begin;history_rows=bounded.history_rows;raw_rows=bounded.raw_rows;
    history_ids=std::move(bounded.history_ids);raw_ids=std::move(bounded.raw_ids);raw_mask=std::move(bounded.raw_mask);
    for(uint32_t h=0;h<history_rows;++h)history_ids[h]=int32_t((start-history_rows+h)%128);
}
