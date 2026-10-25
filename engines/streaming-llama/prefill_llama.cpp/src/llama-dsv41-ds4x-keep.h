#pragma once
// LLAMADECODE keep-open (LLAMADECODE_DS4X_KEEP_OPEN): after a prefill weight sweep
// rebuilt the provider banks, the ds4 decoder stays open and only the resident
// weights whose device copy moved are re-registered. Pure plan, tested in
// tests/dsv41_ds4x_keep.cpp.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_dsv41_ds4x_keep {
struct resident {
    uint64_t offset = 0, bytes = 0;
    void *device = nullptr;
};
struct plan {
    bool delta = false;            // same file ranges in the same order: move pointers in place
    std::vector<size_t> moved;     // indices into `now` to (re)register
    size_t unchanged = 0;
};
// delta: every (offset, bytes) matches in order, so only changed device pointers
// are replaced. Otherwise (layout changed) the caller drops every range and
// registers all of `now` (moved = all indices).
inline plan reregister_plan(const std::vector<resident> &held, const std::vector<resident> &now) {
    plan p;
    p.delta = held.size() == now.size();
    for (size_t i = 0; p.delta && i < now.size(); ++i)
        p.delta = held[i].offset == now[i].offset && held[i].bytes == now[i].bytes;
    for (size_t i = 0; i < now.size(); ++i) {
        if (p.delta && held[i].device == now[i].device) { ++p.unchanged; continue; }
        p.moved.push_back(i);
    }
    return p;
}
}
