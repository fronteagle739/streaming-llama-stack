#pragma once

#include "llama-dsv41-layer.h"
#include "llama-dsv41-engram.h"
#include "ggml-cpp.h"
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "../../../ds4-win/ds4_external_host_cache.h"

// Original native-1017 GGUF -> bounded DirectStorage layer staging. Metadata
// tensors never mmap or allocate their complete file payload. Engram tables
// remain opaque, original row264 bytes on disk and are excluded from stages.
// This provider does not enable the main model inference gate.
class llama_dsv41_weight_provider {
public:
    struct config {
        std::string model_path;
        // Caller must first verify this complete original checkpoint's SHA.
        // Physical file size must equal the verified receipt, not a partial.
        uint64_t verified_file_bytes=0;
        // Identity bound to that completed SHA verification on Windows:
        // volume serial, 128-bit FileIdInfo identifier, Unix last-write ns.
        // A stale same-sized file must be reverified, never trusted by size.
        uint64_t verified_volume=0,verified_mtime_ns=0;
        std::array<uint8_t,16> verified_file_id{};
        std::string verified_sha256;
        ggml_backend_t backend=nullptr;
        // Includes compute buffers plus DS GPU staging and factory host stage.
        uint64_t admitted_device_bytes=0,admitted_host_bytes=0;
        uint64_t job_host_bytes=0; // complete process RSS ceiling, including decode RAM
        uint32_t buffers=1;
        // Source-pinned candidate profile, independently admitted before allocate.
        // Zero/default retains the measured baseline, not an environment override.
        uint64_t staging_limit_bytes=0;
        bool auto_request_size=false;
        bool async_staging_drain=false;
        bool compute_direct=false;
        // Whole-prompt original row264 cache on its own IoRing. Zero is off.
        uint64_t engram_prefetch_host_bytes=0;
        uint32_t engram_prefetch_tokens=0;
        // No-warm-up start: the early Engram cache is built for the very request
        // that runs next on this provider, which adopts it instead of building
        // its own, so only one prompt cache is ever alive.
        bool engram_early_is_next_request=false;
        // LLAMADECODE_ENGRAM_EARLY=1, cold launch: the request's tokens (and a
        // restored base's last <=3 tokens). The early cache starts as soon as
        // the open's combined admission passes, before the first weight fetch.
        std::vector<int32_t> engram_early_tokens,engram_early_tail;
        bool zero_padding_only=false;
        uint32_t ds_submit_threads=0;
        uint32_t ds_gpu_ring_slots=0;
        // LLAMADECODE B1: 82-stage decode layout (embedding, then per layer a
        // dense stage and a routed-expert stage, then the head). Only the
        // routed experts' bytes are fetched. Switched at a pass boundary.
        bool decode_layout=false;
        bool yoco_encoder_only=false; // omit stages 22..41 before any prefetch
    };
    struct capacity {
        uint64_t maximum_stage_bytes=0,compute_buffer_bytes=0;
        uint64_t device_bytes=0,host_bytes=0;
        uint64_t staging_bytes=0,request_bytes=0;
    };

    llama_dsv41_weight_provider();
    ~llama_dsv41_weight_provider();
    llama_dsv41_weight_provider(const llama_dsv41_weight_provider&)=delete;
    llama_dsv41_weight_provider&operator=(const llama_dsv41_weight_provider&)=delete;
    llama_dsv41_weight_provider(llama_dsv41_weight_provider&&) noexcept;
    llama_dsv41_weight_provider&operator=(llama_dsv41_weight_provider&&) noexcept;
    // Windows+CUDA+DS only. Caller owns backend for the full provider lifetime.
    // inspect retains the read-only physical file lease and metadata only;
    // no DS factory or GPU buffer allocation. Admit combined job budgets from
    // budget() BEFORE allocate(). open() is the standalone inspect+allocate.
    bool inspect(const config &options,std::string &error);
    bool allocate(std::string &error);
    bool open(const config &options,std::string &error);
    // Consume only a fully completed pass, retaining the original model lease,
    // heaps, requests and imported pointers. Start is deferred to the timed run.
    bool restart_prompt(const config &options,std::string &error);
    bool start_request(std::string &error);
    bool finish_yoco_encoder(std::string &error);
    // DS4COMBO fast switch: release / rebuild the DS heaps and imported compute
    // buffers between passes; model lease, plan and Engram state are kept.
    bool suspend_device(std::string &error);
    bool resume_device(std::string &error);
    bool device_suspended() const;
    capacity budget() const;
    ggml_backend_t backend() const;
    bool prepare_engram_prompt(const std::vector<int32_t> &tokens,
         const std::function<bool()> &cancelled,std::string &error,llama_dsv41::engram_history initial={},const uint8_t* text_mask=nullptr);
    // DS4COMBO producer-warm: start the whole-prompt Engram row cache for a
    // known later request on its own IoRing; prepare_engram_prompt adopts it
    // for exactly these tokens/history. Both caches are charged to admission.
    // current_request: the cache is for the request about to run (sweep start),
    // so no second cache is alive and an already-queued early cache is kept.
    bool start_engram_prompt_early(const std::vector<int32_t> &tokens,llama_dsv41::engram_history initial,std::string &error,bool current_request=false,const uint8_t* text_mask=nullptr);
    // Under LLAMA_DSV41_ENGRAM_PROMPT_OVERLAP=1 prepare_engram_prompt adopts a
    // still-loading early cache; every Engram reader first waits here, outside
    // the provider lock, for its table (0: layer 1, 1: layer 14). The table-1
    // wait joins the early build. waited_ms is the time spent blocked.
    bool wait_engram_table(uint32_t index,const std::function<bool()> &cancelled,double &waited_ms,std::string &error);
    // LLAMADECODE_ENGRAM_EARLY=1 or auto (auto: started only when the planned
    // requests / LLAMADECODE_ENGRAM_IOPS fit the gap; else the ordinary path).
    // Cold: after inspect/admission, before allocate's wf_start, for
    // config.engram_early_tokens. No-op when that list is empty.
    bool start_engram_cold(std::string &error);
    // Whether an early cache is queued (not yet adopted/joined).
    bool early_engram_queued() const;
    // LLAMADECODE_ENGRAM_CACHE_MB: the process-wide Engram row LRU, created
    // once (fully allocated) and kept across requests. Its bytes are charged to
    // every request's combined host admission.
    uint64_t engram_row_cache_bytes() const;
    bool create_engram_row_cache(uint64_t bytes,std::string &error);
    // Warm: on a completed released pass, the moment an append's tokens
    // arrive (before the next request's open/start_request/layer-0 fetch).
    // Charges the live previous cache; reuses its rows (copied, not read).
    // gap_s: the auto decision's gap (the previous pass's layer-0 wall).
    bool start_engram_append(const std::vector<int32_t> &tokens,llama_dsv41::engram_history initial,double gap_s,std::string &error);
    bool engram_tile(uint32_t index,llama_dsv41::engram_history &history,
        const int32_t *tokens,const uint8_t *text_mask,size_t count,
        std::vector<float> &output,std::string &error);
    // Advance decoded text history with the existing hash/token-map logic.
    // No table fetch or model computation. Only a completed released pass may
    // accept at most128 already evaluated tokens per synchronization batch.
    bool advance_decoded_engram(llama_dsv41::engram_history &history,
        const int32_t *tokens,size_t count,std::string &error);
    // Before the first stage lease, map a restored checkpoint's last three
    // IDs using the existing original model token map. No table/model eval.
    bool restored_engram_history(llama_dsv41::engram_history &history,
        const int32_t *tokens,size_t count,std::string &error,const uint8_t*mask=nullptr);
    // Strict order: embedding stage, layers0..39, output head. One lease at a
    // time; release synchronizes the actual backend before any buffer reuse.
    bool acquire_embeddings(ggml_tensor *&embedding,std::string &error);
    bool acquire_layer(uint32_t layer,llama_dsv41_layer_graph::weights &weights,std::string &error);
    bool acquire_head(ggml_tensor *&norm,ggml_tensor *&output,std::string &error);
    // LLAMADECODE B1 (decode layout only). acquire_layer_dense binds the dense
    // stage and pre-binds the routed-expert tensors at their fixed bank; the
    // experts' bytes arrive only through acquire_layer_experts, which publishes
    // the routing, releases the dense stage and waits for the routed stage.
    bool acquire_layer_dense(uint32_t layer,llama_dsv41_layer_graph::weights &weights,std::string &error);
    // LLAMADECODE B2: returns the cache-pool slot of each routed id; only
    // misses are fetched (DirectStorage, explicit extents into bank 0).
    bool acquire_layer_experts(uint32_t layer,const std::vector<int32_t> &ids,std::vector<int32_t> &slots,std::string &error);
    // LLAMADECODE ds4-decode: put the provider in the decode layout and make
    // every non-routed stage resident (one pass), so an external decoder can
    // read the weights in place; list them as (file offset, bytes, device).
    bool prepare_external_decode(const config &options,std::string &error);
    bool decode_resident_tensors(const std::function<bool(uint64_t,uint64_t,void*)> &visit,std::string &error) const;
    // The expert-cache slots of one token's routed ids (only misses are
    // fetched), the pool bases, and the slot count. cuda_stream receives the
    // DirectStorage fence wait (the external decoder's stream).
    bool external_expert_slots(uint32_t layer,const int32_t *ids,uint32_t n,int32_t *slots,
        const char **gate,const char **up,const char **down,uint32_t *slot_count,void *cuda_stream,std::string &error);
    struct expert_cache_stats{uint32_t slots=0;uint64_t hits=0,misses=0,fetched_bytes=0;double fetch_ms=0;
        uint32_t l2_slots=0;uint64_t l2_hits=0,l2_stored=0;
        uint32_t host_slots=0;uint64_t host_hits=0,host_stored=0,host_bytes=0;
        double host_store_ms=0,host_sync_ms=0,host_upload_ms=0,hook_ms=0,residency_ms=0;};
    bool configure_host_cache(const ds4_external_host_cache_v1 &api,uint64_t host_limit,uint64_t floor,std::string &error);
    bool release_host_cache(std::string &error);
    bool release_prefill_rows(std::string &error);
    // LLAMADECODE cache growth: an exclusive VRAM L2 behind the bank-0 pool
    // (decode layout only). L1 victims are copied into it; an L1 miss found
    // in L2 is copied back instead of fetched. Released when the provider
    // leaves the decode layout. Returns the slot count in *slots.
    bool grow_decode_cache(uint64_t bytes,uint32_t *slots,std::string &error);
    // LLAMADECODE growable context cache: return the L2's VRAM (contents dropped)
    // so a smaller L2 can be planned after a context growth step.
    bool release_decode_cache(std::string &error);
    uint64_t decode_cache_slot_bytes() const;
    // B5: tier[e]=1 for each of the layer's experts resident in L1 or L2.
    bool decode_residency(uint32_t layer,uint8_t *tier,uint32_t experts) const;
    expert_cache_stats cache_stats() const;
    bool decode_layout() const;
    // DirectStorage bytes and transfer milliseconds of the stages acquired since start_request.
    void fetch_totals(uint64_t &bytes,double &milliseconds) const;
    // The ggml buffers holding a decode layer's dense and routed stages.
    bool decode_banks(uint32_t layer,const void *&dense,const void *&routed) const;
    bool release(std::string &error);
    void invalidate();
private:
    struct implementation;
    std::unique_ptr<implementation>p;
};
