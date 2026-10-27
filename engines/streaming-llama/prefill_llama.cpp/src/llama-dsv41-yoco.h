#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

// Approximate bounded replay. Positions refer to the logical call, never a tile.
namespace llama_dsv41_yoco {
enum class operation { prefill, decode, verify, append_decode, all_logits };
inline void validate(uint32_t keep) {
    if(keep!=0 && keep!=140) throw std::runtime_error("yoco_keep must be 0 or 140");
}
struct plan {
    uint32_t first=0, count=0, layers=40;
    bool approximate=false, encoder_only=false;
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
    p.approximate=true;
    return p;
}
inline uint32_t first_row(const plan&p,uint32_t layer) { return layer>20?p.first:0; }
inline bool may_export(uint32_t keep) {validate(keep);return keep==0;}
inline bool may_rewind(uint32_t keep) {return may_export(keep);}
}
