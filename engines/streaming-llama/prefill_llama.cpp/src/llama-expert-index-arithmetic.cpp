#include "llama-expert-index.h"

#include <limits>

namespace {

constexpr int DSV4_EXPERTS = 256;

} // namespace

bool dsv4_expert_slab_loc_from_geometry(
        int               file_idx,
        uint64_t          tensor_offset,
        uint64_t          expert_stride,
        uint64_t          tensor_bytes,
        int               expert,
        uint64_t          expected_span,
        expert_slab_loc * out) {
    if (out == nullptr || file_idx < 0 || expert < 0 || expert >= DSV4_EXPERTS) {
        return false;
    }
    if (expert_stride != expected_span || tensor_bytes != expert_stride * DSV4_EXPERTS) {
        return false;
    }

    const uint64_t delta = static_cast<uint64_t>(expert) * expert_stride;
    if (tensor_offset > std::numeric_limits<uint64_t>::max() - delta ||
        tensor_offset + delta > std::numeric_limits<uint64_t>::max() - expert_stride) {
        return false;
    }

    const uint64_t slab_begin = tensor_offset + delta;
    const uint64_t slab_end   = slab_begin + expert_stride;
    if (tensor_offset > std::numeric_limits<uint64_t>::max() - tensor_bytes ||
        slab_end > tensor_offset + tensor_bytes) {
        return false;
    }

    *out = { file_idx, slab_begin, expert_stride };
    return true;
}
