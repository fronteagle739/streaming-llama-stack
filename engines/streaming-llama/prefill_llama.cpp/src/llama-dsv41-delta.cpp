#include "llama-dsv41-delta.h"

#include <algorithm>
#include <utility>

namespace llama_dsv41 {

bool make_delta_plan(const wire_frontier & base, const wire_frontier & target,
                     delta_plan & result, std::string & error) {
    // Clear stale ranges even on rejection: callers cannot accidentally reuse
    // a successful plan after an invalid transition.
    result = {};
    wire_plan before, after;
    if (!make_wire_plan(base, before, error) || !make_wire_plan(target, after, error)) return false;
    if (base.context != target.context || base.mode != target.mode ||
        (base.mode != 1 && base.mode != 3) || base.position >= target.position ||
        (base.mode == 3 && (base.capture_length != std::min(base.position, 128u) ||
                           target.capture_length != std::min(target.position, 128u)))) {
        error = "native V4.1 delta requires forward, same-context/mode, complete frontiers";
        return false;
    }
    delta_plan plan;
    plan.base = base;
    plan.target = target;
    plan.first_token = base.position;
    plan.token_count = target.position - base.position;
    auto append = [&](const state_span & span, uint64_t offset, uint64_t bytes) {
        if (!bytes) return;
        plan.slices.push_back({span, offset, bytes});
        plan.state_bytes += bytes;
    };
    for (const state_span & span : after.spans) {
        switch (span.kind) {
            case state_kind::raw_window:
            case state_kind::dspark_hidden: {
                const uint64_t row_bytes = span.kind == state_kind::raw_window ? 512u*4u : 3u*5120u*4u;
                if (plan.token_count >= 128) {
                    append(span, 0, span.bytes);
                } else {
                    const uint32_t first = base.position % 128;
                    const uint32_t count = std::min(plan.token_count, 128u - first);
                    append(span, uint64_t(first)*row_bytes, uint64_t(count)*row_bytes);
                    append(span, 0, uint64_t(plan.token_count-count)*row_bytes);
                }
                break;
            }
            case state_kind::compressed:
            case state_kind::index: {
                const uint32_t ratio = span.index < 3 ? 2 : 1;
                const uint64_t row_bytes = span.kind == state_kind::compressed ? 512u*4u : 128u*4u;
                const uint64_t offset = uint64_t(base.position/ratio)*row_bytes;
                append(span, offset, span.bytes-offset);
                break;
            }
            case state_kind::pool_kv:
            case state_kind::pool_score:
                append(span, 0, span.bytes);
                break;
        }
    }
    // Bounded independently of context. Two pieces per physical ring,
    // eight compressed/index pieces and at most six pool tails.
    if (plan.slices.size() > 96) {
        error = "native V4.1 delta slice count exceeds canonical bound";
        return false;
    }
    for (const state_slice & slice : plan.slices) {
        if (slice.offset > slice.span.bytes || slice.bytes > slice.span.bytes-slice.offset ||
            (slice.offset & 3) || (slice.bytes & 3)) {
            error = "native V4.1 delta slice lies outside its canonical span";
            return false;
        }
    }
    result = std::move(plan);
    error.clear();
    return true;
}

} // namespace llama_dsv41
