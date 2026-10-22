#pragma once

#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "llama-adapter.h"
#include "llama-cparams.h"
#include "llama-decode-handoff.h"
#include "llama-graph.h"
#include "llama-kv-artifact.h"
#include "llama-kv-tier.h"
#include "llama-kv-tier-export.h"
#include "llama.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>


// Per-layer callback type: (layer_index, total_layers)
using layer_callback_fn = std::function<void(int, int)>;

// Hidden state storage mode for layer-major processing
enum llama_hidden_mode {
    LLAMA_HIDDEN_CPU    = 0,  // CPU RAM — synchronous PCIe round-trips (default)
    LLAMA_HIDDEN_GPU    = 1,  // GPU buffer — eliminates PCIe, GPU→GPU copies
    LLAMA_HIDDEN_DBLBUF = 2,  // Double-buffer — two GPU staging tensors, pipelined upload/compute
};

// KV cache location mode for layer-major processing
enum llama_kv_mode {
    LLAMA_KV_GPU = 0,  // KV cache on GPU (default, all layers resident)
    LLAMA_KV_RAM = 1,  // KV cache in RAM with double-buffer prefetch to GPU
};

struct llama_prefill_kv_mode_config {
    llama_kv_mode mode        = LLAMA_KV_GPU;
    bool          forced      = false;
    bool          explicit_env = false;
    bool          recognized  = true;
    const char *  source      = "default";
    const char *  value       = nullptr;
};

static inline llama_prefill_kv_mode_config llama_prefill_resolve_kv_mode_from_env() {
    llama_prefill_kv_mode_config cfg;

    const char * mode_env = std::getenv("LLAMA_PREFILL_KV_MODE");
    const char * legacy_env = std::getenv("LLAMA_PREFILL_KV_RAM");
    const char * env = nullptr;

    if (mode_env && mode_env[0] != '\0') {
        cfg.source = "LLAMA_PREFILL_KV_MODE";
        cfg.value = mode_env;
        cfg.explicit_env = true;
        env = mode_env;
    } else if (legacy_env && legacy_env[0] != '\0') {
        cfg.source = "LLAMA_PREFILL_KV_RAM";
        cfg.value = legacy_env;
        cfg.explicit_env = true;
        env = legacy_env;
    }

    if (!env) {
        return cfg;
    }

    if (std::strcmp(env, "ram") == 0 || std::strcmp(env, "cpu") == 0 || std::strcmp(env, "1") == 0) {
        cfg.mode = LLAMA_KV_RAM;
        cfg.forced = true;
    } else if (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "0") == 0) {
        cfg.mode = LLAMA_KV_GPU;
        cfg.forced = true;
    } else if (std::strcmp(env, "auto") == 0) {
        cfg.mode = LLAMA_KV_GPU;
        cfg.forced = false;
    } else {
        cfg.mode = LLAMA_KV_GPU;
        cfg.forced = false;
        cfg.recognized = false;
    }

    return cfg;
}

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0;  // memory allocated for the model
    size_t context = 0;  // memory allocated for the context
    size_t compute = 0;  // memory allocated for temporary compute buffers

    size_t total() const { return model + context + compute; }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(const llama_model & model, llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model &   get_model() const;
    const llama_cparams & get_cparams() const;
    bool is_mla_model() const;

    ggml_backend_sched_t get_sched() const;
    llm_graph_result * get_graph_result() const { return gf_res_prev.get(); }

    uint32_t n_ctx() const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch() const;
    uint32_t n_ubatch() const;
    uint32_t n_seq_max() const;

    uint32_t n_threads() const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    void remap_single_output(int32_t batch_tokens, int32_t output_token) {
        output_ids.assign((size_t) batch_tokens, -1);
        if (output_token >= 0 && output_token < batch_tokens) {
            output_ids[(size_t) output_token] = 0;
        }
    }

    bool get_mtp_last_hidden(std::vector<float> & out, llama_pos * pos = nullptr, llama_seq_id * seq_id = nullptr) const;

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t              get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings(bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapter_lora(llama_adapter_lora * adapter, float scale);

    bool rm_adapter_lora(llama_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(const llama_ubatch &     ubatch,
                                      llm_graph_type           gtype,
                                      llama_memory_context_i * mctx,
                                      ggml_status &            ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    // Streaming prefill mode - process layers one-by-one with weight streaming
    // Returns 0 on success, negative on error
    // If streaming not implemented, falls back to regular decode with warning
    int decode_streaming(const llama_batch & batch_inp, int n_layer_buffers);

    // RTX streaming prefill mode accessors
    bool is_prefill_streaming() const { return prefill_streaming; }
    // STREAMING CONTINUATION: reserve cells [0, n) for seq_id (see llama_kv_cache::reserve_prefix_cells).
    // Exported: the server (a separate binary) calls it.
    LLAMA_API bool reserve_prefill_prefix_cells(llama_seq_id seq_id, uint32_t n);
    // NVMe KV tier, restore after a relaunch (docs/KV_TIER_DESIGN.md): the server hands every
    // streaming request's tokens over (they are recorded next to the rows), asks for the longest
    // stored prefix (attaches that conversation on a hit), and routes a continuation whose prefix
    // the tier holds through the RTX streaming pass.
    LLAMA_API bool     kv_tier_enabled() const;
    LLAMA_API void     set_kv_tier_prompt_tokens(const std::vector<llama_token> & tokens);
    LLAMA_API uint32_t kv_tier_restore_prefix(const std::vector<llama_token> & req_tokens, uint32_t lcp_min);
    LLAMA_API bool     kv_tier_prefix_ready(uint32_t n_past) const;
    // Delta routing (2026-09-10): how many leading tokens of `tokens` the decode receiver holds
    // resident (0 when it cannot be asked), and the rows the tier holds for the open conversation.
    LLAMA_API uint32_t kv_decoder_resident_prefix(const llama_token * tokens, size_t n);
    LLAMA_API uint32_t kv_tier_held_rows() const;
    // KV-reuse part B: true when the DSV4 KV lives in a CPU-backed module (post
    // stream_local migration) — the prompt-cache save is safe in that state.
    // Inline over a tracked flag (set by migrate_kv_cache_to_cpu, cleared by the
    // new-prompt/reverse GPU rebuilds) so the server exe links without an export.
    bool dsv4_kv_cpu_resident() const { return dsv4_kv_cpu_resident_flag; }
    bool dsv4_kv_cpu_resident_flag = false;
    // RETAIN-SLOT-KV (LLAMA_DSV4_RETAIN_SLOT_KV=1, default OFF, DSV4-only):
    // after a stream_local (no-Mac) prefill, keep the GPU-resident DSV4 KV module
    // alive across the prefill->decode boundary and across turns instead of the
    // migrate-to-CPU + reverse-migrate round trip. kv_slot_resident() is valid
    // only while the SAME memory module that was marked still exists (module-
    // epoch guard): every module swap (CPU migrate, new-prompt rebuild, reverse-
    // migrate, artifact-import roundtrip) bumps kv_module_epoch and thereby
    // invalidates the claim — structural staleness protection, no dangling reuse.
    // Inline over tracked flags (same linkage rationale as above).
    LLAMA_API bool retain_slot_kv_enabled() const;
    void mark_kv_slot_resident() {
        kv_slot_resident_flag  = true;
        kv_slot_resident_epoch = kv_module_epoch;
    }
    bool kv_slot_resident() const {
        return kv_slot_resident_flag && kv_slot_resident_epoch == kv_module_epoch;
    }
    bool     kv_slot_resident_flag  = false;
    uint64_t kv_slot_resident_epoch = 0;
    uint64_t kv_module_epoch        = 0;  // bumped on every `memory` module swap
    LLAMA_API bool retained_route_capable() const;
    void set_route_reuse_enabled(bool enabled) { route_reuse_enabled = enabled; }
    bool is_route_reuse_enabled() const { return route_reuse_enabled; }
    int32_t get_prefill_decode_remote_nodes() const { return prefill_decode_remote_nodes; }
    uint64_t route_reuse_blob_generation() const { return prefill_kv_state_cache_generation; }
    bool route_reuse_candidate_ready() const {
        return route_reuse_enabled && dsv4_kv_cpu_resident_flag &&
               prefill_kv_state_cache_generation != 0 &&
               route_reuse_published_generation == prefill_kv_state_cache_generation &&
               route_reuse_migrated_generation == prefill_kv_state_cache_generation &&
               !prefill_kv_state_cache.empty() &&
               !prefill_kv_state_cache_prefix_tokens.empty() &&
               !prefill_kv_artifact_bytes.empty();
    }
    LLAMA_API void invalidate_route_reuse_state();
    bool is_prefill_only() const { return prefill_only; }
    bool is_prefill_overlap() const { return prefill_overlap; }
    int32_t get_prefill_decode_mode() const { return prefill_decode_mode; }

    void set_prefill_streaming(bool v) { prefill_streaming = v; }
    void set_n_sublayers(int n) { n_sublayers = n; }
    int  get_n_sublayers() const { return n_sublayers; }
    void set_inside_streaming_decode(bool v) { inside_streaming_decode = v; }
    bool is_inside_streaming_decode() const { return inside_streaming_decode; }
    bool migrate_stream_local_kv_to_cpu() { return migrate_kv_cache_to_cpu(); }
    void set_flash_attn_enabled(bool v) { cparams.flash_attn = v; }
    bool is_flash_attn_enabled() const { return cparams.flash_attn; }
    // Runtime override for n_ubatch / n_batch.  Used by the streaming-prefill
    // handoff path so a single prefill pass can size its ubatch up to cover
    // the entire prompt — leaves zero tail for Mac to re-prefill, which
    // avoids the cross-stack divergence that compounds with re-prefill depth.
    // Caller is responsible for restoring the original values after the pass
    // and for triggering sched_need_reserve so compute buffers re-allocate.
    void set_n_ubatch(uint32_t v) { cparams.n_ubatch = v; }
    void set_n_batch(uint32_t v) { cparams.n_batch = v; }
    uint32_t get_n_ubatch_cached() const { return cparams.n_ubatch; }
    uint32_t get_n_batch_cached() const { return cparams.n_batch; }
    void set_wf_prefill_done(bool v) { wf_prefill_done = v; }
    bool is_wf_prefill_done() const { return wf_prefill_done; }
    void update_n_p_eval(int n) { n_p_eval += n; }
    bool has_decode_handoff_pending() const { return decode_handoff_pending; }

    // Reset streaming/handoff state for a fresh streaming pass.
    // Call this after memory_clear() before re-entering llama_decode() so the
    // streaming prefill path doesn't see stale handoff/position state from a
    // previous pass.  Used by WIR refinement loops.
    void set_sched_need_reserve(bool v) { sched_need_reserve = v; }

    // WIR soft mask control: set alpha and tau for soft causal mask relaxation
    void set_wir_soft_mask(float alpha, float tau) {
        cparams.wir_soft_mask_alpha = alpha;
        cparams.wir_soft_mask_tau   = tau;
    }

    void reset_streaming_state() {
        wf_prefill_done        = false;
        dsv4_prefill_fingerprint_done = false;
        decode_handoff_pending = false;
        decode_handoff_pending_suppressed = false;
        decode_handoff_active  = false;
        inside_streaming_decode = false;
        // Note: do NOT clear sched_need_reserve here. The streaming
        // path inside llama_prefill_stream handles scheduler re-reservation
        // at the actual batch size via the graph build + alloc path.
        // Caller should call llama_synchronize(ctx) before this to clear
        // stale CUDA errors from the previous streaming pass.
        // Clearing it would prevent the scheduler from allocating compute
        // buffers for large batches (they'd stay at the 1-token minimum).
    }
    // Reset KV-RAM per-layer buffers between streaming passes.
    // Frees pinned host memory allocated by kv_ram_layer::alloc_pinned().
    // The staging buffer is kept (needed by prompt_save for KV serialization).
    // Next pass re-initializes kv_ram_layers from scratch.
    // Out-of-line + exported so server-context.cpp can call it via llama.dll.
    LLAMA_API void clear_kv_ram_state();

    // TASK A (streamed-KV visibility, 2026-07-20): re-upload the KV-RAM host mirrors
    // (K / V / sparse-indexer K_IDX per layer) back into the GPU cache tensors at the
    // streaming-prefill -> resident-decode boundary. The layer-major pass D2H-downloads
    // every layer's cache into the mirrors and leaves the GPU cells stale, so the
    // resident decode attends garbage (the streamed prompt is INVISIBLE). Returns bytes
    // uploaded. No-op without KV-RAM state.
    size_t reupload_kv_ram_to_gpu();
    // TASK-A probe (LLAMA_DSV4_AUX_DUMP=1): norm/NaN/extent stats of every DSV4 attention
    // state tensor (raw K/V, CSA/HCA/LID compressed caches, compressor states) at a named
    // boundary. Diff a streamed-prompt run vs a resident-prompt run. No-op without the env.
    void dsv4_aux_norm_dump(const char * tag);
    // Worker thread function for async pool→pageable copyout (pageable mode only).
    void kv_copyout_thread_fn();
    // True when KV-RAM contains serialized cells from a prior streaming
    // prefill. The CACHED-PREFIX HANDOFF path uses this to know it can
    // re-export those cells for a cache-reuse retry without running any
    // new prefill on Windows.
    bool has_kv_ram_state() const { return !kv_ram_layers.empty(); }
    bool trigger_decode_handoff() { return maybe_apply_decode_handoff(); }
    int32_t get_prefill_transport_mode() const { return prefill_transport_mode; }
    void set_prefill_kv_layout_flash_attn(bool v) { prefill_kv_layout_flash_attn = v; }
    bool get_prefill_kv_layout_flash_attn() const { return prefill_kv_layout_flash_attn; }

    // Persistent KV store (Engine A): compute the model fingerprint EXACTLY as
    // the write hook in export_kv_artifact does (model desc + GGUF file identity
    // + KV layout bits: type_k/v, v_trans, is_mla, flash). The server read hook
    // calls this so its lookup namespace matches what export stored. Returns 0
    // when the model has no usable description.
    LLAMA_API uint64_t compute_kv_store_model_fp() const;

    // Persistent KV store read hook (server-facing, plain types so the store's
    // internal types never cross the llama.dll boundary). When the disk store is
    // enabled, does a longest-prefix lookup for `request_tokens`; on a verified
    // hit (chain prefix + token-exact + CRC) it INSTALLS the restored artifact
    // bytes as the retained KV blob + prefix tokens (so the existing CACHED-
    // PREFIX HANDOFF export reuses them) and fills `out_prefix_tokens` with the
    // reused leading tokens. Returns the reuse length (>= min_reuse), or 0 on a
    // disabled store / miss / any verify failure (caller falls open to
    // streaming). Safe to call when the store is disabled (returns 0, no-op).
    LLAMA_API uint32_t kv_store_install_prefix(const std::vector<llama_token> & request_tokens,
                                               uint32_t min_reuse,
                                               std::vector<llama_token> & out_prefix_tokens);
    // True iff the persistent KV store is enabled in this process.
    LLAMA_API bool kv_store_enabled() const;

    // Set a callback for token relay during decode handoff.
    // When set, relay_tokens_cb() is used instead of relay_tokens(stdout).
    // Used by llama-server to route Mac-generated tokens to HTTP SSE.
    using token_relay_cb_t = std::function<bool(int32_t token_id, const std::string & text, bool done)>;
    void set_token_relay_callback(token_relay_cb_t cb) { token_relay_callback = std::move(cb); }
    bool has_kv_artifact_path() const { return !prefill_kv_artifact_path.empty(); }
    void set_prefill_kv_artifact_path(const std::string & path) { prefill_kv_artifact_path = path; }
    void set_prefill_sampling_profile(const llama_kv_artifact_metadata::sampling_profile_t & profile) {
        prefill_sampling_profile = profile;
        has_prefill_sampling_profile = true;
    }
    void clear_prefill_sampling_profile() { has_prefill_sampling_profile = false; }
    bool has_prefill_sampling_profile_data() const { return has_prefill_sampling_profile; }
    void set_prefill_requested_n_predict(int32_t value) { prefill_requested_n_predict = value; }
    void clear_prefill_requested_n_predict() { prefill_requested_n_predict = -1; }
    int32_t get_prefill_requested_n_predict() const { return prefill_requested_n_predict; }
    void set_decode_stop_words(const std::vector<std::string> & words) { decode_stop_words = words; }
    void clear_decode_stop_words() { decode_stop_words.clear(); }
    int32_t get_last_relay_stop_reason() const { return last_relay_stop_reason; }
    const std::string & get_last_relay_stopping_word() const { return last_relay_stopping_word; }
    double get_last_relay_decode_elapsed_ms() const { return last_relay_decode_elapsed_ms; }
    int32_t get_last_relay_decode_intervals() const { return last_relay_decode_intervals; }
    double get_last_relay_decode_tok_s() const { return last_relay_decode_tok_s; }
    double get_last_relay_decode_tok_s_early() const { return last_relay_decode_tok_s_early; }
    double get_last_relay_decode_tok_s_middle() const { return last_relay_decode_tok_s_middle; }
    double get_last_relay_decode_tok_s_late() const { return last_relay_decode_tok_s_late; }
    // Mac-retained-prefix handle from the most recent relay (Option B cross-turn retained-KV).
    // 0 = the Mac reported no handle (legacy receiver) => the server keeps today's behavior.
    uint32_t get_last_relay_retained_token_count() const { return last_relay_retained_token_count; }
    uint64_t get_last_relay_retained_prefix_hash() const { return last_relay_retained_prefix_hash; }
    bool get_last_relay_retained_prefix_miss() const { return last_relay_retained_prefix_miss; }
    const std::string & get_last_relay_status() const { return last_relay_status; }
    // Option B cross-turn retained-KV store accessors (LLAMA_DSV4_MAC_RETAIN_HANDLE).
    void set_mac_retained_handle(uint32_t count, uint64_t hash,
                                 const llama_token * prefix, size_t prefix_len) {
        mac_retained_count = count;
        mac_retained_hash  = hash & 0x7fffffffffffffffULL;
        if (prefix && prefix_len > 0) {
            mac_retained_prefix_tokens.assign(prefix, prefix + prefix_len);
        } else {
            mac_retained_prefix_tokens.clear();
        }
    }
    void clear_mac_retained_handle() {
        mac_retained_count = 0;
        mac_retained_hash  = 0;
        mac_retained_prefix_tokens.clear();
    }
    uint32_t get_mac_retained_count() const { return mac_retained_count; }
    uint64_t get_mac_retained_hash()  const { return mac_retained_hash; }
    const std::vector<llama_token> & get_mac_retained_prefix_tokens() const { return mac_retained_prefix_tokens; }
    void     set_mac_retained_reuse_count(uint32_t n) { mac_retained_reuse_count = n; }
    uint32_t get_mac_retained_reuse_count() const { return mac_retained_reuse_count; }
    void     set_mac_retained_reuse_hash(uint64_t hash) { mac_retained_reuse_hash = hash & 0x7fffffffffffffffULL; }
    uint64_t get_mac_retained_reuse_hash() const { return mac_retained_reuse_hash; }
    void set_prefill_tail_tokens(const llama_token * tokens, size_t count) {
        if (tokens && count > 0) {
            prefill_tail_tokens.assign(tokens, tokens + count);
        } else {
            prefill_tail_tokens.clear();
        }
    }
    void clear_prefill_tail_tokens() { prefill_tail_tokens.clear(); }
    bool has_prefill_tail_tokens() const { return !prefill_tail_tokens.empty(); }
    const std::vector<llama_token> & get_prefill_tail_tokens_vec() const { return prefill_tail_tokens; }
    void set_prefill_handoff_prompt(const llama_token * tokens,
                                    size_t              count,
                                    const std::string & text,
                                    uint32_t            streamed_prefix_count) {
        if (tokens && count > 0) {
            prefill_handoff_prompt_tokens.assign(tokens, tokens + count);
        } else {
            prefill_handoff_prompt_tokens.clear();
        }
        prefill_handoff_prompt_text = text;
        prefill_handoff_streamed_prefix_count = streamed_prefix_count;
    }
    void clear_prefill_handoff_prompt() {
        prefill_handoff_prompt_tokens.clear();
        prefill_handoff_prompt_text.clear();
        prefill_handoff_streamed_prefix_count = 0;
        // Default false; only the gated splice route sets it. Thus all existing
        // callers retain the prior prompt-text serialization byte-for-byte.
        prompt_direct_tokens_only = false;
        clear_prefill_handoff_image_embd();
        clear_prefill_handoff_raw_image();
    }
    void set_prompt_direct_tokens_only(bool value) { prompt_direct_tokens_only = value; }
    void set_prefill_handoff_streamed_prefix_count(uint32_t value) {
        prefill_handoff_streamed_prefix_count = value;
    }
    // v1.8 ship-embeddings: stash the image embeddings (copied) to ship in the handoff
    // artifact so the receiver can re-prefill the image cells itself (correct Metal F32
    // FA) instead of using the RTX's degraded F16-VKQ image K/V. See GROUND_TRUTH §11.
    void set_prefill_handoff_image_embd(const float * embd, uint32_t n_cells, uint32_t n_embd, uint32_t pos_start) {
        if (embd && n_cells > 0 && n_embd > 0) {
            prefill_handoff_image_embd.assign(embd, embd + (size_t) n_cells * (size_t) n_embd);
            prefill_handoff_img_cell_count = n_cells;
            prefill_handoff_img_n_embd     = n_embd;
            prefill_handoff_img_pos_start  = pos_start;
        } else {
            clear_prefill_handoff_image_embd();
        }
    }
    void clear_prefill_handoff_image_embd() {
        prefill_handoff_image_embd.clear();
        prefill_handoff_img_cell_count = 0;
        prefill_handoff_img_n_embd     = 0;
        prefill_handoff_img_pos_start  = 0;
    }
    const std::vector<float> & get_prefill_handoff_image_embd() const { return prefill_handoff_image_embd; }
    uint32_t get_prefill_handoff_img_cell_count() const { return prefill_handoff_img_cell_count; }
    uint32_t get_prefill_handoff_img_n_embd()     const { return prefill_handoff_img_n_embd; }
    uint32_t get_prefill_handoff_img_pos_start()  const { return prefill_handoff_img_pos_start; }
    // v1.9 small-image-forward: stash the RAW image bytes (PNG/JPEG) so the receiver runs
    // mlx_vlm generate() on them locally. prompt_text/tokens (set_prefill_handoff_prompt)
    // carry the question. See GROUND_TRUTH §11 (USER 2026-06-01).
    void set_prefill_handoff_raw_image(const uint8_t * data, size_t n, const std::string & question) {
        if (data && n > 0) {
            prefill_handoff_raw_image.assign(data, data + n);
            prefill_handoff_raw_image_question = question;  // bare question -> KVA prompt_text for mlx_vlm
        } else {
            clear_prefill_handoff_raw_image();
        }
    }
    void clear_prefill_handoff_raw_image() { prefill_handoff_raw_image.clear(); prefill_handoff_raw_image_question.clear(); }
    const std::vector<uint8_t> & get_prefill_handoff_raw_image() const { return prefill_handoff_raw_image; }
    const std::string & get_prefill_handoff_raw_image_question() const { return prefill_handoff_raw_image_question; }
    void set_prefill_handoff_prompt_token_range(const llama_token * tokens, size_t count, uint32_t pos0) {
        if (!tokens || count == 0) {
            return;
        }
        const size_t end = (size_t) pos0 + count;
        if (prefill_handoff_prompt_tokens.size() < end) {
            prefill_handoff_prompt_tokens.resize(end, LLAMA_TOKEN_NULL);
        }
        for (size_t i = 0; i < count; ++i) {
            prefill_handoff_prompt_tokens[(size_t) pos0 + i] = tokens[i];
        }
    }
    const std::vector<llama_token> & get_prefill_handoff_prompt_tokens() const { return prefill_handoff_prompt_tokens; }
    const std::string & get_prefill_handoff_prompt_text() const { return prefill_handoff_prompt_text; }
    uint32_t get_prefill_handoff_streamed_prefix_count() const { return prefill_handoff_streamed_prefix_count; }
    void set_prefill_handoff_capture_seq_id(llama_seq_id seq_id) { prefill_handoff_capture_seq_id = seq_id; }
    void clear_prefill_handoff_capture_seq_id() { prefill_handoff_capture_seq_id = -1; }
    llama_seq_id get_prefill_handoff_capture_seq_id() const { return prefill_handoff_capture_seq_id; }
    int32_t get_prefill_crossover_tokens() const { return prefill_crossover_tokens; }
    bool has_transport_plan() const { return decode_handoff_plan.use_transport; }
    const std::string & get_prefill_kv_host() const { return prefill_kv_host; }
    int32_t get_prefill_kv_port() const { return prefill_kv_port; }
    const std::string & get_token_stream_host() const { return token_stream_host; }
    int32_t get_token_stream_port() const { return token_stream_port; }

    llama_decode_executor_i * get_decode_handoff_executor(bool refresh = false);
    const llama_decode_handoff_plan & get_decode_handoff_plan() const { return decode_handoff_plan; }

    // Map RTX slot to Mac decode session: slot N -> kv_port_base+(N%sessions)*2
    // Wraps around when more RTX slots than Mac sessions.
    void set_handoff_slot_id(int32_t slot_id) {
        const int32_t sessions = std::max<int32_t>(1, prefill_decode_remote_nodes);
        const int32_t session = slot_id % sessions;
        const int32_t port_offset = session * 2;
        decode_handoff_plan.kv_port          = prefill_kv_port + port_offset;
        decode_handoff_plan.token_stream_port = token_stream_port + port_offset;
    }

    // Arm handoff for prompt-direct mode (sets pending flag without going through prefill)
    void arm_decode_handoff_for_prompt_direct() { decode_handoff_pending = true; }
    // Arm handoff when the server already has the desired local KV state for
    // this request (for example after prompt-cache reuse plus Mac tail tokens)
    // and just needs the normal post-prefill export/relay path to run.
    void arm_decode_handoff_for_cached_prefill() {
        decode_handoff_active = false;
        decode_handoff_pending = true;
    }
    // Arm handoff for PROMPT-DIRECT KV cache-hit path (Gate #2):
    // Mac already holds the KV for this prefix (session_id from the cache reply).
    // We skip sending the token stream and instead send a zero-byte PROMPT-DIRECT
    // trigger that the Mac receiver interprets as "use your cached KV".
    // session_id is stored so the export path can embed it in the artifact header
    // so Mac can match against its LRU entry.
    void arm_decode_handoff_for_prompt_direct_cache_hit(uint64_t cached_session_id) {
        prompt_direct_cache_hit     = true;
        prompt_direct_cached_sid    = cached_session_id;
        decode_handoff_pending      = true;
    }
    bool     prompt_direct_cache_hit  = false;
    uint64_t prompt_direct_cached_sid = 0;
    // Pre-populate the KV state cache with raw data (for prompt-direct: token IDs)
    void set_prefill_kv_state_cache(const std::vector<uint8_t> & data) {
        prefill_kv_state_cache = data;
        if (++prefill_kv_state_cache_generation == 0) {
            ++prefill_kv_state_cache_generation;
        }
        route_reuse_published_generation = 0;
        route_reuse_migrated_generation = 0;
    }
    void clear_prefill_kv_state_cache() {
        prefill_kv_state_cache.clear();
        prefill_kv_state_cache_prefix_tokens.clear();
        prefill_kv_state_cache_generation = 0;
        route_reuse_published_generation = 0;
        route_reuse_migrated_generation = 0;
        // Do NOT clear prefill_kv_artifact_bytes here — it's populated per
        // export and consumed by the in-memory publish that happens AFTER
        // this clear in the prompt-direct path. It is cleared/overwritten
        // at the start of the next export_kv_artifact call.
    }
    // Token IDs that the prefill_kv_state_cache blob was captured for. The
    // CACHED-PREFIX HANDOFF path validates the current request's prefix
    // tokens match these before re-exporting the blob — otherwise a prior
    // request's bytes would ship to Mac for an unrelated prompt (@@@@ bug).
    void set_prefill_kv_state_cache_prefix_tokens(const std::vector<llama_token> & tokens) {
        prefill_kv_state_cache_prefix_tokens = tokens;
    }
    const std::vector<llama_token> & get_prefill_kv_state_cache_prefix_tokens() const {
        return prefill_kv_state_cache_prefix_tokens;
    }
    bool has_kv_blob_cache() const { return !prefill_kv_state_cache.empty(); }
    // When true, export_kv_artifact writes raw data without KVARTIF1 header
    bool prompt_direct_export = false;
    bool prompt_direct_tokens_only = false;
    void set_cached_wf(void * wf) { cached_wf = wf; }
    void * get_cached_wf() const { return cached_wf; }
    // Non-owning pointer to the active WF during a streaming prefill, set by the prefill driver so
    // decode_layer_major can reset+re-drive the WF per ubatch in the C1 fast-path-multiub mode.
    void set_prefill_active_wf(void * wf) { prefill_active_wf = wf; }

    // Phase 328 (2026-05-05): persistent gpu_buf_a/b for WF cache reuse.
    // The original per-request alloc/free at llama-prefill-stream.cpp:1316
    // changes VAs across requests, which Phase 250 found triggers GSP
    // wedge when LLAMA_PREFILL_WF_REUSE_CACHED_DANGEROUS=1. Persisting the
    // buffers (allocate once, reuse across all decode_streaming calls,
    // free only at context destroy) keeps VAs stable so WF descriptors
    // remain valid. Combined with cache reuse, this drops per-decode-step
    // cost from ~50s (full NVMe pass) to ~1s (compute only).
    void set_persist_gpu_buf_a(ggml_backend_buffer_t b) { persist_gpu_buf_a = b; }
    ggml_backend_buffer_t get_persist_gpu_buf_a() const { return persist_gpu_buf_a; }
    void set_persist_gpu_buf_b(ggml_backend_buffer_t b) { persist_gpu_buf_b = b; }
    ggml_backend_buffer_t get_persist_gpu_buf_b() const { return persist_gpu_buf_b; }
    void set_persist_gpu_buf_size(size_t s) { persist_gpu_buf_size = s; }
    size_t get_persist_gpu_buf_size() const { return persist_gpu_buf_size; }
    void set_force_safe_mla_multiubatch_inputs(bool v) { force_safe_mla_multiubatch_inputs = v; }
    bool get_force_safe_mla_multiubatch_inputs() const { return force_safe_mla_multiubatch_inputs; }

    int32_t get_prefill_layer_buffers() const { return prefill_layer_buffers; }
    int32_t get_prefill_prefetch_distance() const { return prefill_prefetch_distance; }
    int32_t get_prefill_slab_bytes() const { return prefill_slab_bytes; }
    const std::vector<float> & get_last_layer_compute_times_ms() const { return last_layer_compute_times_ms; }
    bool should_use_streaming_for_batch(const llama_batch & batch) const;

    // GOAL#1 L1 (env LLAMA_WF_DECODE_RESIDENT_MB, default off; DSV4 + STREAM_LOCAL + keep_mmap
    // only): one-time upload of every NON-routed-expert weight (attention/norms/router/shared
    // experts/embd/lm_head) into a PERSISTENT GPU buffer and re-point the tensors at it, so the
    // resident decode runs the dense majority of per-token weight bytes all-GPU with zero mmap
    // faulting. Routed experts (*_exps) stay on the existing path (L2 caches them later).
    // Subsequent calls are cheap: they only re-apply the device pointers if a layer-major
    // streaming pass restored the original mmap pointers (wf_adapter_restore_tensors does that
    // after every pass). Returns true when the working set is resident.
    bool ensure_dense_resident_for_decode();

    // GOAL#1 L2 (env LLAMA_WF_EXPERT_SLAB=1 + LLAMA_WF_EXPERT_CACHE_MB, default off; DSV4 +
    // STREAM_LOCAL only): give the keep_mmap=0 resident decode an EXPERT byte source. One
    // rotating VRAM slab (max per-layer stacked *_exps set, ~2.3 GB) that ALL MoE layers'
    // ffn_{up,gate,down}_exps tensors alias; a sched eval-callback observes ffn_moe_topk-<il>
    // mid-graph and fetches ONLY the routed experts' byte ranges into the slab (file-read miss /
    // VRAM-LRU hit) before that layer's MUL_MAT_ID executes. MUL_MAT_ID reads only routed
    // experts, so stale bytes of unrouted experts are never touched. Returns true when armed.
    bool ensure_expert_slab_for_decode();
    // reads the routed ids from the just-computed topk tensor (synchronizes first — the
    // streaming sched runs with eval_skip_sync=true, so the callback fires with the node's
    // compute still in flight) and fetches those experts into the slab.
    void expert_slab_fetch_layer(int il, ggml_tensor * topk);
    // number of MoE (has-_exps) layers registered; 0 = slab off
    int  expert_slab_active() const { return expert_slab_buf ? (int) expert_slab_layers.size() : 0; }

    // Guard against sched_reserve() and decode handoff running during the
    // inner llama_decode() call of a streaming prefill pass.  Save and clear
    // the flags before the call, restore after return.
    struct streaming_guard_state {
        bool sched_need_reserve;
        bool decode_handoff_pending;
        bool decode_handoff_pending_suppressed;
    };
    streaming_guard_state suppress_sched_and_handoff() {
        streaming_guard_state s {
            sched_need_reserve,
            decode_handoff_pending,
            decode_handoff_pending_suppressed,
        };
        sched_need_reserve = false;
        decode_handoff_pending_suppressed =
            decode_handoff_pending_suppressed || decode_handoff_pending;
        decode_handoff_pending = false;
        return s;
    }
    void restore_sched_and_handoff(const streaming_guard_state & s) {
        sched_need_reserve = s.sched_need_reserve;
        decode_handoff_pending = s.decode_handoff_pending;
        decode_handoff_pending_suppressed = s.decode_handoff_pending_suppressed;
    }

    // Per-layer dispatch callbacks for streaming prefill.
    // When set, graph_compute() routes to graph_compute_per_layer() automatically.
    void set_per_layer_callbacks(layer_callback_fn pre_cb, layer_callback_fn post_cb,
                                  layer_callback_fn mid_cb = nullptr) {
        per_layer_pre_cb  = std::move(pre_cb);
        per_layer_post_cb = std::move(post_cb);
        per_layer_mid_cb  = std::move(mid_cb);
    }

    bool has_per_layer_callbacks() const { return per_layer_pre_cb || per_layer_post_cb; }

    // Set only the mid-layer callback (for selective expert streaming in AR decode)
    // without disturbing pre/post callbacks left by prefill.
    void set_mid_layer_callback(layer_callback_fn mid_cb) {
        per_layer_mid_cb = std::move(mid_cb);
    }

    // Trace-only observer for named nodes that must be read immediately after execution.
    void set_lid_topk_callback(std::function<void(int, ggml_tensor *)> cb) {
        per_layer_lid_topk_cb = std::move(cb);
    }

    // Reset scheduler and graph state after streaming prefill cleanup.
    // The scheduler retains stale buffer/split assignments from the previous
    // streaming pass.  Without reset, a subsequent decode (streaming or not)
    // may hit cuBLAS errors because the scheduler's internal tensor-to-buffer
    // map references the freed streaming GPU buffers.
    // Follows the same reset pattern as graph_reserve() at line 3233.
    // Does NOT set sched_need_reserve (no full scheduler rebuild needed).
    void invalidate_graph() {
        ggml_backend_sched_reset(sched.get());
        gf_res_prev->reset();
    }

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(const char *  filepath,
                         llama_token * tokens_out,
                         size_t        n_token_capacity,
                         size_t *      n_token_count_out);

    bool state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count);

    size_t state_seq_load_file(llama_seq_id  seq_id,
                               const char *  filepath,
                               llama_token * tokens_out,
                               size_t        n_token_capacity,
                               size_t *      n_token_count_out);

    size_t state_seq_save_file(llama_seq_id        seq_id,
                               const char *        filepath,
                               const llama_token * tokens,
                               size_t              n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void                    perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(ggml_opt_dataset_t      dataset,
                   ggml_opt_result_t       result_train,
                   ggml_opt_result_t       result_eval,
                   int64_t                 idata_split,
                   ggml_opt_epoch_callback callback_train,
                   ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(ggml_opt_dataset_t               dataset,
                        ggml_opt_result_t                result,
                        const std::vector<llama_token> & tokens,
                        const std::vector<llama_token> & labels_sparse,
                        llama_batch &                    batch,
                        ggml_opt_epoch_callback          callback,
                        bool                             train,
                        int64_t                          idata_in_loop,
                        int64_t                          ndata_in_loop,
                        int64_t                          t_loop_start);

  private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

  public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // per-layer graph compute for streaming prefill
    // runs a single scheduler pass while forcing synchronization at layer boundaries
    // via eval callbacks, so pre/post layer hooks can upload and prefetch weights.
    struct layer_node_range {
        int start;      // first node index (inclusive)
        int end;        // one past last node (exclusive)
        int layer_idx;  // -1 = pre-layer, -2 = post-layer, >=0 = transformer layer
    };

    using layer_callback_fn = std::function<void(int layer_idx, int n_layer)>;

    ggml_status graph_compute_per_layer(ggml_cgraph *     gf,
                                        bool              batched,
                                        layer_callback_fn pre_layer_cb,    // called before each layer dispatches
                                        layer_callback_fn post_layer_cb);  // called after each layer syncs

    // Layer-major ubatch processing: execute a node range [node_start, node_end)
    // from an already-allocated graph using ggml_graph_view + backend compute.
    // The graph must have been previously allocated via ggml_backend_sched_alloc_graph.
    // Returns GGML_STATUS_SUCCESS on success.
    ggml_status compute_node_range(ggml_cgraph * gf, int node_start, int node_end);

    // Layer-major decode: process all ubatches layer-by-layer instead of
    // ubatch-by-ubatch. Streams each layer's weights once via WF, processes
    // all ubatches through that layer, then moves to the next layer.
    // Hidden states spill to RAM between layer transitions.
    // Activated by LLAMA_PREFILL_LAYER_MAJOR=1 env var.
    //
    // hidden_mode controls where hidden states are stored between layers:
    //   LLAMA_HIDDEN_CPU:    CPU RAM, synchronous PCIe transfers (default)
    //   LLAMA_HIDDEN_GPU:    GPU buffer (~128 MB), GPU→GPU copies (~900 GB/s)
    //   LLAMA_HIDDEN_DBLBUF: Double-buffer staging, pipelined upload/compute
    int decode_layer_major(llama_batch & batch, llama_hidden_mode hidden_mode = LLAMA_HIDDEN_CPU,
                           llama_kv_mode kv_mode = LLAMA_KV_GPU);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(uint32_t                       n_tokens,
                                uint32_t                       n_seqs,
                                uint32_t                       n_outputs,
                                const llama_memory_context_i * mctx,
                                bool                           split_only = false,
                                size_t *                       sizes      = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

  public:
    // Phase 1.3 — split-decode per-layer graph execution.
    // Mirrors the per-layer pattern in decode_streaming() but for a single layer
    // with split-mode flags. Called by the public C API llama_build_compute_split_layer.
    // Returns 0 on success, negative on error (see enum in llama.h).
    int32_t build_compute_split_layer(const struct llama_split_layer_params * params,
                                       float * out_buf,
                                       bool    defer_read = false);
    // Phase 3 Step 4: finalize a deferred-read submit (event_wait + tensor_get).
    int32_t wait_split_layer_async();

    // Path A Phase D (2026-04-20) — mega-graph split-decode launch.
    // Builds ONE cgraph for all n_layer FFN_ONLY subgraphs using the per-layer
    // external tensor arrays on `sg` (attn_out_buf[], hidden_buf[]). ONE call
    // to model.build_graph + ONE scheduler compute. Per-layer outputs land in
    // sg->metal_hidden_buf[il+1] via ggml_cpy nodes emitted by the builder.
    //
    // First-iteration semantics: all per-layer attn_out buffers must already
    // be populated before this is called (e.g. caller either has pre-filled
    // them synchronously, or on Apple unified memory the RX thread and GPU
    // happen to serialize). Per-layer event-gated cmd_buf splitting is a
    // follow-on iteration (requires a new ggml-metal compute-with-events API).
    //
    // Requires Metal backend + shared per-layer MTLBuffers already allocated
    // on the split_graph (via llama_split_graph_create with FFN_ALL_LAYERS).
    // Returns 0 on success, negative on error.
    int32_t build_compute_split_mega(struct llama_split_graph * sg,
                                       int32_t token_pos,
                                       int32_t seq_id);

    // Apply output norm + LM head to a hidden state, write logits to out_logits
    // (size n_vocab fp32). Used for bookend-last and real decode-loop semantics.
    int32_t apply_output_head(const float * hidden,
                               int32_t token_pos, int32_t seq_id,
                               float * out_logits);

    // Run the GLM/DeepSeek NextN MTP block from a captured target hidden state
    // and current token, writing full draft logits to out_logits.
    int32_t mtp_draft_logits(const float * hidden,
                              int32_t token_id,
                              int32_t token_pos,
                              int32_t seq_id,
                              float * out_logits);

    int32_t mtp_draft_step(const float * hidden,
                            int32_t token_id,
                            int32_t token_pos,
                            int32_t seq_id,
                            float * out_logits,
                            float * out_hidden,
                            int32_t n_embd_out);

    // Look up the embedding for a token. Returns the initial hidden state
    // (pre layer 0) in out_hidden (size n_embd fp32).
    int32_t embed_token(int32_t token_id,
                         int32_t token_pos, int32_t seq_id,
                         float * out_hidden);

    // Helper used by llama_split_graph_capture_layer: return the first
    // CUDA backend on this context (or nullptr if none). Public because
    // llama_split_graph_* are free functions and need to reach into the
    // context's backend list.
    ggml_backend_t get_cuda_backend() const;

    // Metal backend on this context (or nullptr if none). Phase 3 Mac
    // split_graph mirror needs this analogous to get_cuda_backend.
    ggml_backend_t get_metal_backend() const;

    // Reusable per-call GPU staging buffers for build_compute_split_layer.
    // Allocated once on first use, keyed by buffer type. Both buffers are sized
    // for one token's worth of embeddings (n_embd fp32). Saves 250 us/call of
    // buffer alloc+free overhead across the per-layer split loop.
    ggml_backend_buffer_t split_in_buf = nullptr;
    ggml_backend_buffer_t split_attn_buf = nullptr;
    ggml_backend_buffer_type_t split_buf_buft = nullptr;
    size_t split_buf_bytes = 0;

    // Per-(layer_id, mode) graph result cache for build_compute_split_layer.
    // Key is (layer_id << 1) | (mode - 1); mode is FFN_ONLY=1 or ATTN_ONLY=2 in
    // llama_split_layer_mode, so we map to 0..(2*n_layer-1). Each entry owns a
    // fresh llm_graph_result so the graph built for this (layer_id, mode) is
    // stable across calls — same gf pointer, same node pointers, allowing
    // llm_graph_result::can_reuse() to return true after the first call and
    // skip model.build_graph() entirely. This is also what enables ggml-cuda's
    // CUDA graph cache to hit (keyed on cgraph->nodes[0] pointer) so that
    // capture-replay actually engages across calls.
    std::vector<std::unique_ptr<llm_graph_result>> split_layer_graph_results;

    // When non-null, an active cudaStreamBeginCapture was started on the
    // gpu_backend's stream by llama_split_graph_capture_layer(). The
    // build_compute_split_layer path will skip its ggml_backend_synchronize
    // in this case so the sync doesn't land inside the capture (CUDA
    // disallows waits on the capturing stream).
    // Pointer to cudaStream_t (void* so the header is usable without CUDA types).
    void *                                           split_capture_stream = nullptr;

    // Phase 3 Mac event-based sync: opaque MTLSharedEvent allocated on
    // first use by build_compute_split_layer when SPLIT_DECODE_EVENT_SYNC=1.
    // Replaces per-layer ggml_backend_synchronize with encode_signal +
    // event_wait_host — saves ~45 ms/token on Mac FFN loop.
    void *                                           split_sync_event = nullptr;

    // Phase 3 Step 4 async submit state. Set by
    // llama_submit_split_layer_async; consumed by llama_wait_split_layer_async.
    // Only one layer submit in flight at a time. wrap_ctx and batch are
    // freed immediately in the submit path — once graph_compute returns,
    // the cmd_buf already encodes the relevant buffer/tensor addresses.
    bool                                             async_in_flight = false;
    float *                                          async_pending_out_buf = nullptr;
    struct ggml_tensor *                             async_pending_t_embd = nullptr;
    size_t                                           async_pending_out_bytes = 0;

    // Phase 2d.C1 — during capture, build_compute_split_layer normally skips
    // the final ggml_backend_tensor_get (D2H) because that copy would land in
    // the captured graph with an invalid host destination. When this device
    // pointer is non-null, the capture path instead enqueues a D2D copy from
    // res->t_embd to this destination so the replay writes output into a
    // stable device buffer the caller can read via get_output afterwards.
    // Set to llama_split_graph::device_out[il] at capture-layer time.
    void *                                           split_capture_output_device = nullptr;
    size_t                                           split_capture_output_bytes  = 0;

    // Phase 2d.C2 — during capture, build_compute_split_layer records the
    // host pointer for the kq_mask tensor so the replay path can update it
    // per token. Set to the self_kq_mask->data pointer during set_inputs.
    // kq_mask is a HOST buffer (is_host) so the captured graph's H2D copy
    // reads from this host address — updating it before replay fixes the
    // stale mask bug where all tokens used the capture-time mask.
    void *                                           split_capture_kq_mask_ptr = nullptr;
    int32_t                                          split_capture_kq_mask_n_kv = 0;
    size_t                                           split_capture_kq_mask_bytes_per_entry = 0;

  private:
    llm_graph_params graph_params(llm_graph_result *             res,
                                  const llama_ubatch &           ubatch,
                                  const llama_memory_context_i * mctx,
                                  llm_graph_type                 gtype) const;

    llm_graph_cb graph_get_cb() const;
    void refresh_decode_handoff_plan(bool log_plan = false);
    void arm_decode_handoff_after_prefill();
    bool maybe_apply_decode_handoff();
    bool migrate_kv_cache_to_cpu(const std::vector<uint8_t> * src_blob = nullptr);
    bool capture_kv_state_blob(std::vector<uint8_t> & out, std::string & err);
    bool reload_kv_state_into_fresh_memory(const std::vector<uint8_t> & state,
                                           bool                         keep_imported_memory,
                                           std::vector<uint8_t> *       echoed_state,
                                           std::string &                err);
    bool export_kv_artifact(std::string * out_path, llama_kv_artifact_summary * summary, std::string & err);
    bool validate_kv_artifact_roundtrip(const std::string & path, std::string & err);
    bool should_capture_mtp_last_hidden(const llama_ubatch & ubatch, int32_t n_outputs_cur) const;
    bool capture_mtp_last_hidden(const llm_graph_result & res, const llama_ubatch & ubatch);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data(llama_io_read_i & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;

    // RTX streaming prefill mode
    bool    prefill_streaming     = false;
    int     n_sublayers           = 1;  // sub-layers per model layer (MoE split)
    bool    inside_streaming_decode = false;
    bool    wf_prefill_done        = false;
    bool    dsv4_prefill_fingerprint_done = false;
    void *  cached_wf              = nullptr;  // wf_context * cached across batches
    void *  prefill_active_wf      = nullptr;  // non-owning; active WF during prefill (C1 per-ubatch reset)
    // Phase 328 (2026-05-05): persistent compute buffers for WF cache reuse.
    // Allocated lazily on first prefill_stream call, freed at context destroy.
    ggml_backend_buffer_t persist_gpu_buf_a = nullptr;
    ggml_backend_buffer_t persist_gpu_buf_b = nullptr;
    size_t persist_gpu_buf_size = 0;

    // KV-RAM state (persists across decode_layer_major calls)
    struct kv_ram_layer {
        void * orig_k_data = nullptr;
        void * orig_v_data = nullptr;
        void * orig_k_idx_data = nullptr;
        // Storage mode: "pinned" = per-layer cudaMallocHost (iter-16 path, gated by
        // LLAMA_PREFILL_KV_RAM_STORAGE=pinned); "pageable" = plain std::vector +
        // rotating pinned staging pool (new default for large ctx).
        // k_ram / v_ram point to either the pinned alloc (pinned mode) or the
        // pageable_k / pageable_v data (pageable mode — do NOT free via cudaFreeHost).
        uint8_t * k_ram = nullptr;  // pinned host buffer for K (pinned mode only)
        uint8_t * v_ram = nullptr;  // pinned host buffer for V (pinned mode only)
        uint8_t * k_idx_ram = nullptr;  // pinned/pageable host buffer for sparse indexer K
        std::vector<uint8_t> pageable_k;  // pageable bulk storage for K (pageable mode)
        std::vector<uint8_t> pageable_v;  // pageable bulk storage for V (pageable mode)
        std::vector<uint8_t> pageable_k_idx;  // pageable bulk storage for sparse indexer K
        size_t k_bytes = 0;
        size_t v_bytes = 0;
        size_t k_idx_bytes = 0;

        // Alloc / free helpers — defined in llama-context.cpp (needs cuda_runtime.h).
        // Must be exported from llama.dll so server-context.cpp can call them via
        // clear_kv_ram_state() which is also out-of-line.
        LLAMA_API void alloc_pinned();
        LLAMA_API void free_pinned();
        // Allocate pageable bulk storage (plain std::vector, no CUDA pinning).
        void alloc_pageable();
        // Free pageable storage and clear k_ram/v_ram aliases.
        void free_pageable();
    };
    std::vector<kv_ram_layer> kv_ram_layers;
    ggml_backend_buffer_t     kv_staging_buf  = nullptr;
    uint8_t *                 kv_staging_base = nullptr;
    size_t                    kv_max_k_bytes  = 0;
    size_t                    kv_max_v_bytes  = 0;
    size_t                    kv_max_k_idx_bytes = 0;

    // Rotating pinned staging pool (pageable mode).
    // N pinned buffers, each sized max(k_max + v_max) bytes.
    // Pool slots are acquired round-robin by kv_upload; kv_download reuses
    // the same slot it uploaded into (slot index = il % kv_pool_n).
    int                        kv_pool_n     = 0;  // 0 = pool not allocated
    size_t                     kv_pool_slot_bytes = 0;  // bytes per slot
    std::vector<uint8_t *>     kv_pool_bufs;       // [n] pinned pointers
    bool                       kv_storage_pageable = false;  // true = pageable+pool mode

    // NVMe KV tier (docs/KV_TIER_DESIGN.md): with LLAMA_KV_TIER_DIR set, every layer's new rows are
    // also written once to the conversation's row files by the copyout worker (pageable mode is
    // forced). Phase 1 shadows the pageable mirror and verifies against it; later phases read the
    // continuation, the export and the post-relaunch restore from the files instead of host RAM.
    bool                       kv_storage_tier = false;
    std::shared_ptr<llama_kv_tier> kv_tier;
    int                        kv_tier_verify_ok  = 0;   // per-pass shadow-verify counters (kv_upload)
    int                        kv_tier_verify_bad = 0;
    // Phase-1 proof: read every tier file back and compare it with the pageable mirror it shadows.
    // Called with the copyout worker drained and the mirror still allocated.
    void kv_tier_verify_mirror(const char * where);
    // Record the covered prompt tokens next to the rows (tokens.bin), if the server handed them over.
    void kv_tier_write_tokens_if_any();
    std::vector<llama_token> kv_tier_prompt_tokens;
    // STREAMED EXPORT (P2b): the artifact planned by export_kv_artifact when the tier holds every
    // cell of the sequence; the publish pulls its chunks straight from the row files and clears it.
    std::unique_ptr<llama_kv_tier_export> kv_tier_export;
    // MIRROR-FREE tier mode (default with the tier; LLAMA_KV_TIER_MIRROR=1 keeps the pageable mirror):
    // the row files are the only host copy of the KV, so the per-layer host mirror sized for the
    // whole context (5.4 GB at 49k for GLM-5.3, 27 GB at 262k) is never allocated. k_ram stays null;
    // every consumer checks it. Counts the layers whose prefix rows nothing could supply in a pass.
    bool                       kv_tier_mirror_free = false;
    int                        kv_tier_prefix_missing = 0;
    // Tier I/O attribution per pass (printed on the TIMING line): the prefix reads in kv_upload and
    // the row writes on the copyout worker.
    double                     kv_tier_read_ms = 0.0;
    uint64_t                   kv_tier_read_bytes = 0;
    std::atomic<uint64_t>      kv_tier_write_us{0};
    std::atomic<uint64_t>      kv_tier_write_bytes{0};

    // Async copyout worker (pageable mode): performs D2H staging→pageable memcpy
    // in background so kv_download returns immediately.
    // Job struct: describes one slot-to-pageable copy to do after GPU D2H completes.
    struct CopyoutJob {
        int        slot_idx;       // pool slot index (il % kv_pool_n)
        uint8_t *  pool_src;       // pinned pool slot (k, v, k_idx at fixed offsets)
        uint8_t *  pageable_k_dst; // destination in pageable_k
        uint8_t *  pageable_v_dst; // destination in pageable_v (nullptr if MLA/v_bytes==0)
        uint8_t *  pageable_k_idx_dst; // destination in pageable_k_idx
        size_t     k_bytes;
        size_t     v_bytes;
        size_t     k_idx_bytes;
        void *     done_event;     // cudaEvent_t — signals when D2H is complete
        // NVMe tier: the rows this pass produced for layer il (the slot holds the whole layer;
        // the new rows are the slice [tier_row0, tier_row0 + tier_n_rows) at each kind's row size).
        int        il           = -1;
        int64_t    tier_row0    = 0;
        int64_t    tier_n_rows  = 0;
        size_t     tier_rbk     = 0;
        size_t     tier_rbv     = 0;
        size_t     tier_rbi     = 0;
    };

    // Per-slot inflight state: 0=free, 2=memcpy in flight (enqueued in worker).
    // kv_upload must wait for slot to reach 0 before writing pageable→pool.
    // Stored as a raw dynamic array (std::atomic is non-movable, can't go in std::vector).
    std::unique_ptr<std::atomic<int>[]> kv_pool_inflight;   // [kv_pool_n], each 0 or 2

    // Per-slot download-complete CUDA events (reused across layers).
    std::vector<void *>           kv_pool_dl_events;  // [kv_pool_n], cudaEvent_t

    // Worker thread machinery.
    std::thread              kv_copyout_worker;
    std::mutex               kv_copyout_mu;
    std::condition_variable  kv_copyout_cv;
    std::queue<CopyoutJob>   kv_copyout_queue;
    std::atomic<bool>        kv_copyout_stop{false};
    // Count of jobs that have been enqueued but not yet fully completed by the worker.
    // Incremented before push, decremented after memcpy in worker.
    // cleanup_kv_ram waits for this to reach 0 before accessing KV data.
    std::atomic<int>         kv_copyout_pending{0};
    bool    prefill_only          = false;
    bool    prefill_overlap       = false;
    ggml_type prefill_kv_type_k   = GGML_TYPE_F16;
    ggml_type prefill_kv_type_v   = GGML_TYPE_F16;
    bool      prefill_kv_swa_full = false;
    bool      prefill_kv_layout_flash_attn = false;
    int32_t prefill_min_stream_batch_tokens = 32;
    int32_t prefill_layer_buffers = 2;
    int32_t prefill_prefetch_distance = 1;
    int32_t prefill_slab_bytes        = 16 * 1024 * 1024;
    int32_t prefill_host_cache_window_layers = 0;
    int32_t prefill_host_cache_window_bytes  = 0;
    int32_t prefill_host_cache_prefetch_distance = 1;
    int32_t prefill_decode_mode       = LLAMA_PREFILL_DECODE_MODE_AUTO;
    int32_t prefill_decode_gpu_layers = -1;
    int32_t prefill_decode_remote_layers = 0;
    bool    prefill_decode_strict_partition = false;
    bool    prefill_decode_require_no_fallback = false;
    bool    prefill_decode_transport_required = false;
    int32_t prefill_decode_remote_nodes = 1;
    std::string prefill_decode_remote_ranges;
    std::string prefill_decode_remote_failover = "reroute";
    int32_t prefill_transport_mode = LLAMA_PREFILL_TRANSPORT_MODE_DISABLED;
    int32_t prefill_execution_mode = LLAMA_PREFILL_EXECUTION_MODE_COUPLED;
    int32_t prefill_tb_chunk_bytes = 4 * 1024 * 1024;
    std::string prefill_kv_transport;
    std::string prefill_tb_direct_endpoint;
    std::string prefill_kv_host;
    int32_t prefill_kv_port = 0;
    int32_t prefill_kv_streams = 0;
    int32_t prefill_kv_stream_chunk_bytes = 0;
    int32_t prefill_kv_max_inflight_bytes = 0;
    int32_t prefill_kv_socket_send_buf = 0;
    int32_t prefill_kv_socket_recv_buf = 0;
    std::string prefill_kv_bind_addrs;
    std::string prefill_kv_peer_addrs;
    std::string prefill_kv_balance;
    bool    prefill_kv_transport_fallback = false;
    bool    prefill_loopback_remote_gpu = false; // test-only: map tb_remote_* decode layers to local GPU
    bool    prefill_kv_validate_import = false;
    bool    prefill_remote_weights = false;
    bool    prefill_remote_weights_hybrid = true;
    bool    prefill_shadow_catchup = false;
    bool    force_safe_mla_multiubatch_inputs = false;
    int32_t prefill_shadow_catchup_threshold_tokens = -1;
    int32_t prefill_shadow_catchup_max_batch = -1;
    std::string prefill_kv_artifact_path;
    std::string prefill_tb_session_dir;
    llama_kv_artifact_metadata::sampling_profile_t prefill_sampling_profile = {};
    bool has_prefill_sampling_profile = false;
    int32_t prefill_requested_n_predict = -1;
    std::vector<std::string> decode_stop_words;
    int32_t last_relay_stop_reason = 0;
    std::string last_relay_stopping_word;
    double last_relay_decode_elapsed_ms = 0.0;
    int32_t last_relay_decode_intervals = 0;
    double last_relay_decode_tok_s = 0.0;
    double last_relay_decode_tok_s_early = 0.0;
    double last_relay_decode_tok_s_middle = 0.0;
    double last_relay_decode_tok_s_late = 0.0;
    // Mac-retained-prefix handle captured from the most recent token relay (Option B).
    uint32_t last_relay_retained_token_count = 0;
    uint64_t last_relay_retained_prefix_hash = 0;
    bool last_relay_retained_prefix_miss = false;
    std::string last_relay_status;
    // Option B cross-turn retained-KV STORE (LLAMA_DSV4_MAC_RETAIN_HANDLE). Recorded at a
    // plain PROMPT-DIRECT completion from the Mac's reported handle; consumed on the NEXT
    // turn to route a cached-prefix continuation (ship only the delta). Empty/0 => today's
    // behavior. Survives prompt_clear (the KV lives on the Mac; this is only the routing
    // handle). mac_retained_reuse_count is the count armed for a delta-only send this turn
    // (read by the handoff runtime builder), 0 otherwise.
    uint32_t mac_retained_count = 0;
    uint64_t mac_retained_hash  = 0;
    std::vector<llama_token> mac_retained_prefix_tokens;
    uint32_t mac_retained_reuse_count = 0;
    uint64_t mac_retained_reuse_hash = 0;
    std::vector<llama_token> prefill_tail_tokens;
    std::vector<llama_token> prefill_handoff_prompt_tokens;
    std::string prefill_handoff_prompt_text;
    uint32_t prefill_handoff_streamed_prefix_count = 0;
    llama_seq_id prefill_handoff_capture_seq_id = -1;
    // v1.8 ship-embeddings: image embeddings stashed for the handoff artifact.
    std::vector<float> prefill_handoff_image_embd;
    uint32_t prefill_handoff_img_cell_count = 0;
    uint32_t prefill_handoff_img_n_embd     = 0;
    uint32_t prefill_handoff_img_pos_start  = 0;
    // v1.9 small-image-forward: raw image bytes (PNG/JPEG) + the bare question stashed for the
    // handoff artifact (the question becomes the KVA prompt_text for the receiver's mlx_vlm generate).
    std::vector<uint8_t> prefill_handoff_raw_image;
    std::string          prefill_handoff_raw_image_question;
    // Phase 5.3 (2026-05-12): seq_id we actually exported from in the last
    // capture_kv_state_blob call. Differs from prefill_handoff_capture_seq_id
    // when streaming-prefill TB wrote cells with seq_id=0 and the multi-slot
    // direct export fell back to legacy seq_cp(0). Used to set
    // meta.source_slot_id correctly so the receiver routes the import.
    llama_seq_id prefill_handoff_exported_seq_id = -1;

    // Prefill strategy + token relay (Phase 3/4)
    int32_t     prefill_strategy = 0;
    int32_t     prefill_crossover_tokens = 0;
    std::string token_stream_host;
    int32_t     token_stream_port = 0;
    int32_t     kv_sync_batch_size = 512;

    llama_decode_handoff_plan                decode_handoff_plan;
    std::unique_ptr<llama_decode_executor_i> decode_handoff_executor;
    bool                                     decode_handoff_pending            = false;
    bool                                     decode_handoff_pending_suppressed = false;
    bool                                     decode_handoff_active             = false;
    token_relay_cb_t                         token_relay_callback;  // server SSE hook
    uint64_t                                 prefill_stream_start_unix_us = 0;
    uint64_t                                 prefill_stream_compute_done_unix_us = 0;
    uint64_t                                 prefill_stream_complete_unix_us = 0;
    uint64_t                                 last_transport_first_chunk_send_unix_us = 0;
    uint64_t                                 last_transport_last_chunk_send_unix_us = 0;

    llama_cross cross;  // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0;  // capacity (of floats) for embeddings
    float * embd      = nullptr;

    std::vector<float> mtp_last_hidden;
    llama_pos          mtp_last_hidden_pos = -1;
    llama_seq_id       mtp_last_hidden_seq_id = -1;
    bool               mtp_last_hidden_valid = false;

    // [TAG_MTP_SHIP_HIDDEN_2026_06_24] Full per-token LAST-LAYER hidden states for the STREAMED prefix, captured during
    // decode_layer_major (env LLAMA_PREFILL_SHIP_MTP_HIDDEN=1) so the Mac MTP sidecar (layer-78 NextN) can warm its own
    // KV over a KV-inject session — which otherwise lacks the prompt's hidden states and so disables MTP (receiver
    // mtp_can_run gate). fp16, row-major [n_tokens][n_embd], indexed by absolute token position. Serialized into the KV
    // artifact and consumed by the receiver as full_hidden. Phase A of MTP-on-KV-inject.
    std::vector<uint16_t> mtp_streamed_hidden;        // fp16, mtp_streamed_hidden_n_tokens * n_embd
    int64_t               mtp_streamed_hidden_n_tokens = 0;
    int64_t               mtp_streamed_hidden_n_embd   = 0;
    bool                  mtp_streamed_hidden_valid    = false;

    // TODO: simplify
    struct sampling_info {
        std::map<llama_seq_id, llama_sampler *> samplers;

        float * logits      = nullptr;
        size_t  logits_size = 0;

        llama_token * sampled      = nullptr;
        size_t        sampled_size = 0;

        float * probs      = nullptr;
        size_t  probs_size = 0;

        llama_token * candidates      = nullptr;
        size_t        candidates_size = 0;

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0;           // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids;  // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // L1 dense-residency state (ensure_dense_resident_for_decode)
    struct dense_resident_entry {
        ggml_tensor * tensor   = nullptr;
        void *        dev_data = nullptr;
    };
    ggml_backend_buffer_t              dense_resident_buf = nullptr;
    std::vector<dense_resident_entry>  dense_resident_entries;
    bool                               dense_resident_failed = false;

    // L2 expert-slab state (ensure_expert_slab_for_decode)
    struct expert_slab_tensor {
        ggml_tensor * t           = nullptr;  // the *_exps tensor, pre-pointed at the slab
        uint64_t      file_offset = 0;        // whole-tensor offset within its GGUF file
        uint16_t      file_idx    = 0;
        bool          has_file    = false;    // false -> host-data fallback only (keep_mmap=1)
        int           kind        = 0;        // 0=up 1=gate 2=down
        size_t        stride      = 0;        // nb[2] = bytes per expert
        size_t        slab_off    = 0;        // this kind's base offset within the slab
        void *        dev_data    = nullptr;  // slab pointer to re-apply after WF restore
        void *        host_data   = nullptr;  // original data ptr (mmap view) for fallback reads
    };
    // indexed by graph il (empty vector = dense layer)
    std::vector<std::vector<expert_slab_tensor>> expert_slab_layers;
    ggml_backend_buffer_t              expert_slab_buf = nullptr;
    std::map<uint16_t, FILE *>         expert_slab_files;      // cached GGUF handles for expert reads
    std::vector<uint8_t>               expert_slab_stage;      // reusable read staging (max stride)
    ggml_backend_buffer_t              expert_lru_buf  = nullptr;
    size_t                             expert_lru_slot_stride = 0;
    int                                expert_lru_n_slots     = 0;
    // slab ownership: per (kind slot 0..2, expert) which graph-il's bytes currently sit in the slab
    std::vector<int>                   expert_slab_owner;      // [kind*n_expert + e] -> il or -1
    // LRU: key (il<<20 | kind<<18 | e) -> slot; recency list of keys
    std::unordered_map<uint32_t, int>  expert_lru_map;
    std::list<uint32_t>                expert_lru_order;       // front = most recent
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> expert_lru_pos;
    std::vector<uint32_t>              expert_lru_slot_key;    // [slot] -> key (for eviction bookkeeping)
    bool                               expert_slab_failed = false;
    // tiny stub tensors driving ggml_backend_tensor_copy for raw D2D ranges
    ggml_context_ptr                   expert_stub_ctx;
    ggml_tensor *                      expert_stub_src = nullptr;
    ggml_tensor *                      expert_stub_dst = nullptr;
    // stats (per process lifetime; logged periodically)
    uint64_t expert_slab_bytes_file = 0, expert_slab_bytes_d2d = 0, expert_slab_hits = 0, expert_slab_misses = 0;

    ggml_backend_t                backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // Expose backends for streaming prefill
    const std::vector<ggml_backend_ptr> & get_backends() const { return backends; }

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // Per-layer dispatch callbacks (set by streaming prefill)
    layer_callback_fn per_layer_pre_cb;
    layer_callback_fn per_layer_post_cb;
    layer_callback_fn per_layer_mid_cb; // fires at ffn_moe_topk for selective expert streaming
    std::function<void(int, ggml_tensor *)> per_layer_lid_topk_cb;
    std::vector<float> last_layer_compute_times_ms;

    // Default-off resident-decode DSV4 LID selection trace.  Unlike the
    // streaming per-layer observer above, this wraps the ordinary scheduler
    // eval callback used by ctx->decode().
    bool                                  resident_lid_trace_enabled = false;
    bool                                  resident_lid_trace_armed   = false;
    std::string                           resident_lid_trace_path;
    uint64_t                              resident_lid_trace_step    = 0;
    uint64_t                              resident_lid_trace_cur_step = 0;
    llama_pos                             resident_lid_trace_pos     = -1;
    int32_t                               resident_lid_trace_needle_row = 1400;
    int                                   resident_lid_trace_score_il = -1;
    std::vector<float>                    resident_lid_trace_scores;
    // lid_k (needle KEY) capture: distinguish query-vs-key for bug#2. Needle row key norm vs a reference row
    // (row 700) key norm + NaN count. If needle norm ~0/NaN/wildly-off vs ref -> the resident LID cache's
    // needle key is garbage (key bug); if similar -> the query is the issue. Additive; env-gated only.
    float                                 resident_lid_trace_needle_key_norm = -1.0f;
    float                                 resident_lid_trace_ref_key_norm    = -1.0f;
    int32_t                               resident_lid_trace_needle_key_nan  = 0;
    int                                   resident_lid_trace_key_il          = -1;
    float                                 resident_lid_trace_query_norm      = -1.0f;  // decode-token query norm
    int32_t                               resident_lid_trace_query_nan       = 0;
    ggml_backend_sched_eval_callback      resident_lid_trace_user_cb = nullptr;
    void *                                resident_lid_trace_user_data = nullptr;
    static bool resident_lid_eval_callback(ggml_tensor * t, bool ask, void * user_data);

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size;  // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0;  // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t prefill_last_token_id = -1;  // last token ID from the prefill batch (for KV artifact header)
    std::vector<uint8_t> prefill_kv_state_cache;  // cached KV state blob captured during streaming prefill
    // Prefix token IDs that prefill_kv_state_cache represents. Used by
    // CACHED-PREFIX HANDOFF to verify the blob applies to the current
    // request (prefix tokens must match) before re-exporting.
    std::vector<llama_token> prefill_kv_state_cache_prefix_tokens;
    uint64_t prefill_kv_state_cache_generation = 0;
    uint64_t route_reuse_published_generation = 0;
    uint64_t route_reuse_migrated_generation = 0;
    bool route_reuse_enabled = false;
    // Fully-serialized KV artifact (header + extensions + payload) as
    // produced by llama_kv_artifact_serialize_to_buffer. Populated by
    // export_kv_artifact alongside the on-disk file; consumed by the
    // in-memory transport path so the SSD round-trip can be skipped.
    // Cleared on the same lifecycle as prefill_kv_state_cache.
    std::vector<uint8_t> prefill_kv_artifact_bytes;
    mutable int32_t n_eval   = 0;  // number of eval calls

    mutable int32_t n_reused = 0;  // number of times the previous graph was reused
};
