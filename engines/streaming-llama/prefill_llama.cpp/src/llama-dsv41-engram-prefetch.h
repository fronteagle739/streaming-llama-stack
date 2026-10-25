#pragma once
#include "llama-dsv41-engram.h"
#include "llama-dsv41-engram-row-cache.h"
#include <functional>
#include <memory>
// LLAMADECODE_ENGRAM_EARLY=1|auto (default auto, read once): the early starts
// are armed. =0: no early start; the load starts at sweep start. Auto: each
// start is decided by llama-dsv41-engram-early-decision.h (requests /
// LLAMADECODE_ENGRAM_IOPS <= gap; else sweep start). Every arm loads on the
// early multi-ring loader (prepare_early/load_early), so the row cache is
// consulted on every load; the single-ring load() is no longer called.
bool llama_dsv41_engram_early_enabled();
bool llama_dsv41_engram_early_auto();
// LLAMADECODE_ENGRAM_EARLY=pause (forced pause arm).
bool llama_dsv41_engram_early_pause();
double llama_dsv41_engram_early_iops();
// The weight provider marks its DirectStorage pass active/idle; the early
// loader reports its time-weighted overlap with it.
void llama_dsv41_engram_weight_stream(bool active);
class llama_dsv41_engram_prompt_cache {
    struct implementation;
    std::unique_ptr<implementation> p;
public:
    llama_dsv41_engram_prompt_cache();
    ~llama_dsv41_engram_prompt_cache();
    llama_dsv41_engram_prompt_cache(const llama_dsv41_engram_prompt_cache&)=delete;
    llama_dsv41_engram_prompt_cache&operator=(const llama_dsv41_engram_prompt_cache&)=delete;
    void prepare(const llama_dsv41::engram_layout&,const int32_t*,uint32_t,const std::array<uint64_t,2>&,uint64_t,llama_dsv41::engram_history initial={});
    uint64_t budget()const;
    void load(const wchar_t*,void *verified_file,uint64_t,const std::function<bool()>&);
    bool read(uint32_t,uint64_t,void*,size_t)const;
    // Table 0's rows complete and publish before table 1's, so a reader of
    // table 0 (layer 1) need not wait for table 1 (layer 14). Throws when the
    // load failed or stop() reports cancellation.
    void wait_table(uint32_t,const std::function<bool()>&)const;
    // LLAMADECODE_ENGRAM_EARLY only: the same rows/ids as prepare(), an IO
    // arena sized from the admission headroom, and a multi-IoRing extent load
    // (planner extents, batched submits, table 0 published before table 1).
    // prior: the provider's last completed cache; its rows are copied, not read.
    void prepare_early(const llama_dsv41::engram_layout&,const int32_t*,uint32_t,const std::array<uint64_t,2>&,uint64_t,
                       llama_dsv41::engram_history,std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior);
    // Requests the early loader would issue for the prepared rows (rows held by
    // the row cache, when set, are excluded; prior-cache hits are not).
    uint64_t planned_requests()const;
    // LLAMADECODE_ENGRAM_CACHE_MB: set before prepare_early; load_early copies
    // the cached rows, reads only the missing ones, then inserts them.
    void set_row_cache(std::shared_ptr<llama_dsv41_engram_row_cache>);
    // Set before prepare_early: true when the load overlaps the weight stream
    // (sweep start, producer-warm, synchronous fallback); its misses then use
    // the paced 1-ring/depth-256 geometry (llama_dsv41_engram_early::choose_pacing).
    void set_weight_overlap(bool);
    void load_early(const wchar_t*,void *verified_file,uint64_t,const std::function<bool()>&);
};
