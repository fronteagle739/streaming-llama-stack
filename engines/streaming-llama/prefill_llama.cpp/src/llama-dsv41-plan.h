#pragma once

#include <cstdint>
#include <vector>

// One contiguous, single-sequence tile. The planner is bounded independently
// of context length; the full layer-major scheduler pages the other carries.
struct llama_dsv41_tile_plan {
    uint32_t start, count, context, ratio;
    uint32_t history_rows, raw_rows, published_first, published_count;
    bool previous_pool_tail, next_pool_tail;
    std::vector<int32_t> positions, history_ids, raw_ids, compressed_positions;
    std::vector<float> raw_mask;
    std::vector<int32_t> compressed_visible;

    llama_dsv41_tile_plan(uint32_t start, uint32_t count, uint32_t context, uint32_t ratio, uint32_t admitted_tile_tokens=32);
    // Only the admitted executor may use the exact device template. Keep the
    // original ABI/layout and ordinary host vectors for all existing callers.
    llama_dsv41_tile_plan(uint32_t start, uint32_t count, uint32_t context, uint32_t ratio, uint32_t admitted_tile_tokens,bool device_raw_template);
};
