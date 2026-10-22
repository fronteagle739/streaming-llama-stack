#pragma once

#include "llama.h"

#include <cstdint>

// Phase 326 (2026-05-05): bumped from 256 to 1024 to support NP=1000+ for
// shared-prefix-via-seq_cp architecture. Used for fixed-size arrays in
// llama_batch_allocr (seq_pos, seq_cpl, seq_idx) and llama_kv_cache cell
// per-seq tracking. 4x increase = ~4x memory in those arrays but they are
// small (<1KB each) so impact is negligible.
#define LLAMA_MAX_SEQ 1024

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool offload_recurrent;
    bool flash_attn;
    bool auto_fa;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    int32_t n_expert_splits;  // split MoE experts into N groups for sub-layer pipelining (default: 1)

    bool sparse_mla = false;                  // GLM/DeepSeek sparse MLA / DSA routing
    bool sparse_mla_trace = false;
    bool sparse_mla_prefill = true;
    bool sparse_mla_decode = true;
    int32_t sparse_mla_top_k = 0;
    int32_t sparse_mla_prefill_top_k = 0;
    int32_t sparse_mla_prefill_selector = LLAMA_SPARSE_MLA_SELECTOR_SCORE;
    bool sparse_mla_prefill_indexer = true;
    int32_t sparse_mla_decode_baseline = LLAMA_SPARSE_MLA_SELECTOR_SCORE;
    int32_t sparse_mla_fa_mode = LLAMA_SPARSE_MLA_FA_OFF;
    bool sparse_mla_fused_kernel = false;
    bool sparse_mla_fused_topk = false;
    int32_t moe_prefill_expert_cap = 0;
    int32_t mtp_mode = LLAMA_MTP_MODE_OFF;
    int32_t mtp_max_draft = LLAMA_MTP_DEFAULT_MAX_DRAFT;

    // WIR soft causal mask relaxation (Experiment 5 from wir-soft-mask-experiments.md)
    // alpha=0.0: standard causal mask. alpha=1.0: fully bidirectional.
    // tau: spatial decay distance in tokens (future token at distance d gets
    //      alpha * exp(-d/tau) relaxation). 0 = no spatial decay (uniform alpha).
    float wir_soft_mask_alpha = 0.0f;
    float wir_soft_mask_tau   = 0.0f;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
