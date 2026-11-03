#pragma once
// QFNSTREAM: Qwen3.8 Flash Next streaming engine (single RTX, weights streamed from NVMe).
//
// Residency: every dense tensor (attention, delta-net, hyper-connection, router, shared expert, output head) is
// loaded once into VRAM. The routed experts (31.6 GiB at Q2_0) stream per layer through two DirectStorage banks
// (the stack's weight fetcher); the n-gram (PLE) table (26.8 GiB) is read row by row through the stack's IoRing
// Engram loader. Prefill is layer-major over the whole prompt (one weight pass), in row tiles, with the hidden
// streams, KV/index caches and recurrent states resident in VRAM.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "qfn-pack.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>

struct qfn_layer_w {
    ggml_tensor *hc_attn_norm = nullptr, *hc_attn_down = nullptr, *hc_attn_up = nullptr, *hc_attn_inject = nullptr;
    ggml_tensor *hc_ffn_norm = nullptr, *hc_ffn_down = nullptr, *hc_ffn_up = nullptr, *hc_ffn_inject = nullptr;
    // gated delta net
    ggml_tensor *attn_qkv = nullptr, *attn_gate = nullptr, *ssm_a = nullptr, *ssm_alpha = nullptr, *ssm_beta = nullptr;
    ggml_tensor *ssm_conv1d = nullptr, *ssm_dt = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
    // Qwen sparse attention
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *idx_k_proj = nullptr, *idx_q_proj = nullptr, *idx_k_norm = nullptr, *idx_q_norm = nullptr;
    // PLE n-gram embedding
    ggml_tensor *ple_key = nullptr, *ple_value = nullptr, *ple_norm_key = nullptr, *ple_norm_query = nullptr;
    ggml_tensor *ple_norm_conv = nullptr, *ple_conv1d = nullptr;
    // FFN
    ggml_tensor *ffn_gate_inp = nullptr, *ffn_gate_inp_shexp = nullptr;
    ggml_tensor *ffn_gate_shexp = nullptr, *ffn_up_shexp = nullptr, *ffn_down_shexp = nullptr;
};

struct qfn_options {
    std::string model;
    // optional: a second copy of the pack (same size and offsets, e.g. a sparse file on another drive) that serves
    // the routed experts; everything else is read from `model`
    std::string experts_path;
    // Optional exactness artifact: raw bytes read back from the uploaded PLE tensor.
    std::string ple_row_dump;
    // Main-expert PREFILL arithmetic only: 0 auto, 1 W4A4, 2 native-q8_1.
    // Native-packed v7 chooses qualified DSV grouped prefill at admission unless explicitly overridden.
    // Other formats retain native-q8; decode and scalar indexing retain their qualified references.
    // Decode/v3 target arithmetic is independently selected; copied MTP drafter remains Q2_0.
    // basalt-tc keeps TC scoring for every query size. The explicit
    // basalt-tc-prefill policy uses TC for >=64 queries and scalar below64.
    // Auto selects qualified refined TC prefill on SM120, scalar decode/reference elsewhere.
    std::string qsa_indexer = "auto";
    int         nvfp4_mode = 2;
    bool        nvfp4_mode_explicit = false;
    int         nvfp4_decode_mode = 2; // 2 native-q8(default), 5 slot-aware calibrated-w4a4, 6 packed DSV grouped
    bool        nvfp4_decode_mode_explicit = false;
    uint32_t    n_ctx       = 4096;   // capacity of caches and the hidden carry (tokens)
    uint32_t    tile        = 2048;   // prefill row tile
    uint32_t    head_batch  = 256;    // rows per output-head matmul
    uint64_t    host_budget = 6ull << 30;  // host bytes the engine may use (PLE row set, staging)
    // decode expert cache: slots of one expert each (gate|up|down); 0 = size from free VRAM after everything else
    uint32_t    cache_slots = 0;
    uint64_t    vram_margin = 192ull << 20;  // VRAM left free when sizing the cache (DSV4.1 decode: 128 + 64 MiB)
    int         qsa_attention = 0; // 0 native (default), 1 explicit published Basalt prompt/decode attention
    float       lfu_half_life = 64.0f;       // recency-LFU half-life in decode steps (DSV4.1 E8)
    bool        verbose     = true;
};

// Decode sampling and the quality/speed dial. bias = 0 is Max quality (the full model); a positive bias adds that
// many nats to the router logits of experts already in the VRAM cache before the top-k choice (Max speed 1.36125,
// the DSV4.1 dial range); the expert weights always come from the unbiased probabilities.
struct qfn_sampling {
    bool     do_sample = true;    // false is greedy without discarding the stored temperature/filters
    float    temperature = 1.0f;   // legacy sampling defaults; 0 = greedy
    bool greedy() const { return !do_sample || temperature <= 0.0f; }
    float    top_p = 0.95f, min_p = 0.0f, typical_p = 1.0f;
    int      top_k = 20;
    float    repeat_penalty = 1.0f, presence_penalty = 0.0f, frequency_penalty = 0.0f;
    int      repeat_last_n = 64;    // -1 = whole history, 0 = no penalties
    std::map<int32_t, float> logit_bias;
    uint64_t seed = 1;
    float    residency_bias = 0.0f;
};

// Field-wise precedence also normalizes existing repetition aliases within each explicit layer.
std::string qfn_merge_sampling_defaults(const std::string & lower, const std::string & upper);
// Shared CLI and serve request validation. String logit-bias keys use the pack tokenizer.
qfn_sampling qfn_parse_sampling(const std::string & body,
    const std::function<std::vector<int32_t>(const std::string &)> & tokenize);

struct qfn_decode_stats {
    uint64_t tokens = 0, expert_hits = 0, expert_misses = 0, miss_bytes = 0;
    uint64_t attempts = 0, replays = 0;  // v3 graph attempts / discarded attempts; expert_misses counts fetched misses
    uint64_t drafted = 0, accepted = 0;    // MTP drafts proposed / kept
    uint64_t policy_batches = 0, policy_conflicts = 0, policy_replays = 0;
    uint64_t policy_restore_bytes = 0, policy_restore_slots = 0;
    // Actual RAM promotions in this decode/feed interval, including discarded target work.
    uint64_t ram_tier_hits = 0;
    // Subset of total policy restoration directed to pinned RAM; actual refetched bytes.
    uint64_t policy_restore_ram_slots = 0, policy_restore_ram_bytes = 0;
    double   policy_restore_ms = 0;
    double   build_ms = 0, gpu_ms = 0, route_ms = 0, stage_ms = 0, mtp_ms = 0;   // decode time ledger
    double   seconds = 0, miss_ms = 0;
};

// Logits requested from a prefill: full rows for positions >= full_from, argmax + logsumexp for every position
// when all_top1 is set.
struct qfn_logits_request {
    int64_t full_from = -1;
    bool    all_top1  = false;
    std::function<void(int64_t pos, const float * logits, int64_t n_vocab)> on_full;
    std::function<void(int64_t pos, int32_t top1, float lse)> on_top1;
};

class qfn_engine {
public:
    qfn_engine();
    ~qfn_engine();
    // Optional cold prompt allows the shared early loader to overlap engine setup.
    bool open(const qfn_options & opts, std::string & err, const std::vector<int32_t> & cold_prompt = {});
    // Layer-major prefill of tokens[0..n) at positions 0..n-1 (a fresh sequence). Seeds the decode expert cache
    // with the experts the prompt routed to most. QFN_MOE_PASS=1 separates attention tiles from larger routed/
    // shared FFN batches and releases the decode slab during prefill; its identical prompt-ranked entries are
    // then reloaded through DirectStorage. QFN_PREFILL_RELEASE_CACHE=0/1 independently controls the phase edge.
    bool prefill(const std::vector<int32_t> & tokens, const qfn_logits_request & req, std::string & err);
    // Layer-major prefill of tokens appended to the live sequence (positions n_past()..).
    bool append(const std::vector<int32_t> & tokens, const qfn_logits_request & req, std::string & err);
    // A failed mode1 graph cannot be resumed; fresh prefill performs the required complete reset.
    bool reset_required() const;
    int64_t n_past() const;
    // Token-by-token decode after prefill. `first` is the token sampled from the prefill's last logits (it is
    // decoded first). on_token returns false to stop. Stops at n_max tokens or an end-of-generation token.
    bool decode(int32_t first, int n_max, const qfn_sampling & s,
                const std::function<bool(int32_t token, const float * logits)> & on_token, qfn_decode_stats & st, std::string & err);
    // Teacher-forced: run tokens through the decode path (expert cache) without sampling; logits after the last.
    bool feed(const std::vector<int32_t> & tokens, std::vector<float> & last_logits, qfn_decode_stats & st, std::string & err, float bias = 0.0f);
    // Start a request (including an EOG-only request) with a reproducible seed.
    void begin_sampling(const qfn_sampling & s);
    // Repeated draws advance the RNG. A fresh sequence/decode completion starts a new request.
    int32_t sample(const float * logits, const qfn_sampling & s);
    // Model-free checks of the same target distribution, including speculative history.
    static bool sampling_self_test(std::string & err);
    static bool cache_policy_self_test(std::string & err);
    bool is_eog(int32_t token) const;
    const std::vector<std::string> & vocab() const;
    const qfn_hparams & hparams() const;
    struct impl;
private:
    std::unique_ptr<impl> p;
};
