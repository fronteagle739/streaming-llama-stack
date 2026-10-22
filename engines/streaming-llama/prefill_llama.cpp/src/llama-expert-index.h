#pragma once

#include <cstdint>

struct llama_model;

struct expert_slab_loc {
    int      file_idx;
    uint64_t file_offset;
    uint64_t span_bytes;
};

// Pure offset arithmetic used by the model index and CPU fixtures.
bool dsv4_expert_slab_loc_from_geometry(
        int               file_idx,
        uint64_t          tensor_offset,
        uint64_t          expert_stride,
        uint64_t          tensor_bytes,
        int               expert,
        uint64_t          expected_span,
        expert_slab_loc * out);

// Returns the packed routed-expert locations in gate, up, down order.
bool dsv4_expert_slab_index(
        const llama_model * model,
        int                 layer,
        int                 expert,
        expert_slab_loc     out[3]);
