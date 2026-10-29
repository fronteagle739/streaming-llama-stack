#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

// Bounded replay. Positions refer to the logical call, never a tile.
namespace llama_dsv41_yoco {
// Layers21..39 introduce only private128-row windows; compressed KV and
// candidate/index keys come from layer20 over EVERY row. A rectangular replay
// loses at most127 valid outputs per layer at its left edge. After19 layers,
// 128+19*127 rows retain every final window and the final logits. The triangular
// cone can skip that warm-up FFN work later; this deliberately reuses the
// existing contiguous carry/DSpark ownership and keeps the larger rectangle.
constexpr uint32_t exact_keep = 128 + 19 * 127;
enum class operation { prefill, decode, verify, append_decode, all_logits };
inline void validate(uint32_t keep) {
    if(keep!=0 && keep!=140 && keep!=exact_keep) throw std::runtime_error("yoco_keep must be 0, 140 or 2541");
}
// Provenance is negotiated separately from the unchanged complete mode3 wire
// layout. v1 is approximate140; v2 certifies the exact dependency cone. Never
// interpret one version as the other, including for a short full-row prompt.
inline uint32_t remote_version(uint32_t keep) {
    validate(keep);return keep==exact_keep?2u:keep==140?1u:0u;
}
inline bool remote_compatible(uint32_t keep,uint32_t version) {
    return remote_version(keep)==version;
}
struct plan {
    uint32_t first=0, count=0, layers=40;
    bool approximate=false, encoder_only=false;
    bool exact_cone=false;
};
inline plan schedule(uint32_t keep,operation op,bool /*images*/,uint32_t logical_begin,
                     uint32_t logical_end,uint32_t sweep_begin,uint32_t sweep_end) {
    validate(keep);
    if(logical_begin>=logical_end || sweep_begin<logical_begin || sweep_end>logical_end || sweep_begin>=sweep_end)
        throw std::runtime_error("YOCO logical/sweep bounds are invalid");
    plan p{0,sweep_end-sweep_begin,40,false,false};
    // Projected vision rows follow the same layer coverage as text. Seeding
    // and Engram masks use absolute image spans in the sweep, before narrowing
    // decoder coverage. Layer20 must publish EVERY row; an image intersecting
    // the replay tail remains a projected image row in layers21..39 as well.
    if(!keep || op!=operation::prefill || logical_end-logical_begin<=keep) return p;
    const uint32_t replay=logical_end-keep;
    if(sweep_end<logical_end) {
        // Internal encoder endpoints are not continuable frontiers. Keep the
        // entire replay interval in the final sweep, including at forced turns.
        if(sweep_end>replay) throw std::runtime_error("YOCO tail crosses a sweep boundary; re-plan the logical prefill");
        p.layers=21;p.encoder_only=true;
    } else p.first=std::max(sweep_begin,replay)-sweep_begin;
    p.approximate=keep==140;p.exact_cone=keep==exact_keep;
    return p;
}
inline uint32_t first_row(const plan&p,uint32_t layer) { return layer>20?p.first:0; }
// Keep the full prefill's GEMM row shapes. A ragged 2541-row rectangle is
// structurally sufficient but changes shape-dependent floating-point kernels.
// Padding its left edge to an existing tile only adds redundant warm-up work.
inline void align_exact(plan&p,uint32_t tile,bool padding=true){
    if(!tile)throw std::runtime_error("Exact cone tile cannot be zero");
    if(padding&&p.exact_cone&&!p.encoder_only)p.first-=p.first%tile;
}
inline bool may_export(uint32_t keep) {validate(keep);return keep==0;}
inline bool may_rewind(uint32_t keep) {return may_export(keep);}
}
