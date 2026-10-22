#pragma once

// llama-wir.h — Wavefront Iterative Refinement (WIR) library
//
// Extracted from test-wavefront-convergence.cpp into a reusable module.
// Provides block-based Gauss-Seidel refinement for use in the decode
// receiver, decode handoff, or standalone testing.
//
// The core algorithm:
//   1. Start with a draft token sequence (from small model or speculative decode)
//   2. Refine blocks of tokens left-to-right using the big model
//   3. Each block sees converged left context + current block tokens
//   4. KV cache is preserved for left context between passes (optional)
//   5. Block converges when 0 tokens change in a pass

#include "llama.h"

#include <functional>
#include <string>
#include <vector>

// Result of a single block refinement
struct wir_block_result {
    int block_start;        // start offset in candidate
    int block_len;          // tokens in this block
    int passes_used;        // number of passes before convergence
    float initial_agreement; // % agreement with ground truth at start
    float final_agreement;   // % agreement with ground truth at end
    double total_ms;         // wall-clock time for this block
    double total_prefill_tokens; // total tokens prefilled across all passes
};

// Result of a full wavefront refinement
struct wir_result {
    int total_tokens;           // total tokens in candidate
    int total_passes;           // sum of passes across all blocks
    int total_blocks;           // number of blocks processed
    int tokens_changed;         // total tokens changed from draft
    double total_ms;            // wall-clock time for all blocks
    double total_prefill_tokens; // total tokens prefilled
    std::vector<wir_block_result> blocks; // per-block results
};

// Configuration for WIR refinement
struct wir_params {
    int block_size         = 64;    // tokens per block in wavefront mode
    int max_passes         = 10;    // max refinement passes per block
    float confidence       = 0.95f; // per-token freeze threshold
    int topk_accept        = 3;     // accept if draft token in top-K
    bool reuse_kv          = true;  // preserve left-context KV between passes
    std::string refine_mode = "all"; // "all", "n-worst"
    int refine_n           = 1;     // for n-worst: tokens to fix per pass

    // Token relay callback: called each time a block converges with its tokens.
    // (block_start, tokens, count) -> bool (return false to abort)
    using token_cb_t = std::function<bool(int block_start, const llama_token * tokens, int count)>;
    token_cb_t token_callback = nullptr;
};

// Result of manual refinement (refine-last-n or refine-mask)
struct wir_manual_result {
    int total_tokens;       // total tokens in the sequence
    int refinable_tokens;   // tokens eligible for refinement
    int tokens_changed;     // tokens actually changed
    int passes_used;        // refinement passes completed
    double total_ms;        // wall-clock time
};

// Per-pass statistics for N-least convergence tracking
struct wir_pass_stats {
    int pass;                   // pass number (1-based)
    int tokens_changed;         // tokens swapped this pass
    int tokens_matching;        // tokens matching ground truth after this pass
    int total_tokens;           // total candidate tokens
    float agreement_pct;        // tokens_matching / total_tokens * 100
    double pass_ms;             // wall-clock time for this pass
};

// Result of N-least refinement
struct wir_nleast_result {
    int total_tokens;
    int final_changed;          // total tokens changed from original draft
    int passes_used;
    float final_agreement_pct;  // vs ground truth
    double total_ms;
    std::vector<wir_pass_stats> per_pass; // convergence trace
};

// Compute softmax entropy-based confidence in [0,1].
// 1.0 = peaked distribution (high confidence), 0.0 = uniform.
float wir_confidence_from_logits(const float * logits, int n_vocab);

// Refinement strategy configuration
struct wir_refine_config {
    // Strategy: "replace-all" replaces ALL disagreements per pass.
    //           "n-least" replaces only the N worst per pass.
    //           "hybrid" does replace_all_passes of replace-all, then n-least.
    std::string strategy = "n-least";

    int n_worst         = 8;     // tokens to replace per pass in n-least mode
    int max_passes      = 100;   // total passes (across hybrid phases)
    int replace_all_passes = 0;  // for hybrid: how many replace-all passes first
    int topk_accept     = 1;     // accept current token if it's in top-K of model's distribution
                                 // 1 = exact match only (default). 3-5 = accept near-optimal tokens.

    // Confidence gate: tokens with confidence >= this threshold are NOT replaced.
    // Prevents replacing tokens the model is already confident about.
    float confidence_threshold = 0.70f;

    // Change gate: only replace tokens whose current confidence is BELOW this.
    // This is separate from confidence_threshold (which gates on the NEW token's
    // confidence). change_conf gates on how confident the model is about the
    // EXISTING token — low confidence = the model thinks it's wrong = replace it.
    // 0.0 = disabled (use confidence_threshold only). Range: 0.0-1.0.
    float change_conf = 0.0f;

    // Lock gate: tokens with confidence >= this are permanently locked from
    // left-to-right (wavefront). Unlike confidence_threshold which just skips
    // replacement, lock_conf + wavefront_lock_window PERMANENTLY freezes tokens.
    // Set this high (0.90-0.99) to only lock very confident tokens.
    // 0.0 = use confidence_threshold for locking (backward compat).
    float lock_conf = 0.0f;

    // Adjacency constraint: selected tokens must have at least this many
    // positions between them. Prevents cascading instability from changing
    // neighbouring tokens in the same pass.
    int min_token_gap   = 1;     // 1 = no two adjacent tokens in same pass

    // Repetition penalty: applied to the model's logits before selecting replacements.
    // Tokens that appear more than once in the candidate sequence get their logits
    // divided by this factor, making the model prefer novel tokens over repeated ones.
    // 1.0 = no penalty, 1.1-1.3 = mild, 1.5+ = strong.
    float repetition_penalty = 1.0f;

    // N-gram repetition penalty: penalizes tokens that would continue a repeated n-gram.
    // Scans the candidate for repeated sequences of `ngram_penalty_len` tokens.
    // If a token at position K would form an n-gram that already exists earlier,
    // it gets a log-prob penalty of `ngram_penalty_weight` (negative = penalize).
    int   ngram_penalty_len    = 0;     // 0 = disabled, 3-5 = typical
    float ngram_penalty_weight = -5.0f; // log-prob penalty for repeated n-grams

    // Frequency-based n-gram penalty: counts how many times each M-gram appears
    // in the candidate and multiplies the penalty by the count. This targets
    // specifically the sequences that repeat most. When an M-gram appears K times,
    // the penalty applied is: ngram_freq_penalty_weight * (K - 1).
    // Set ngram_freq_min_len to the minimum sequence length to track (e.g., 3).
    int   ngram_freq_min_len   = 0;     // 0 = disabled, 3+ = typical
    float ngram_freq_penalty   = -2.0f; // per-extra-occurrence penalty

    // Refine temperature: instead of always picking top-1 (greedy) for replacement,
    // apply temperature to the refine model's logits before selecting. This allows
    // the big model to pick slightly less probable but more natural tokens, helping
    // break cross-tokenizer artifacts. 0.0 = greedy (default), 0.1-0.3 = mild.
    float refine_temp = 0.0f;

    // Soft lock: a token can only be REPLACED if the proposed replacement has
    // probability at least `soft_lock_margin` higher than the current token's
    // probability. This prevents oscillation between similar-probability alternatives.
    // E.g., soft_lock_margin=0.10 means the new token must be 10% more probable.
    // 0.0 = disabled (any improvement replaces). Range: 0.0 - 1.0.
    float soft_lock_margin = 0.0f;

    // Wavefront locking: tokens that reach >= confidence_threshold can be "locked"
    // (never changed again) but ONLY if they are within `wavefront_lock_window`
    // positions of the last locked token. This pins the answer from left to right,
    // preventing distant tokens from locking prematurely and creating islands of
    // frozen text that block convergence between them.
    //
    // 0 = disabled (no locking). N > 0 = a token at position P can only lock if
    // P <= (last_locked_position + wavefront_lock_window).
    // Initially nothing is locked (last_locked_position = -1), so only tokens
    // in [0, wavefront_lock_window) can lock on the first pass.
    int wavefront_lock_window = 0;

    // Early stop: stops when ALL unlocked tokens are above confidence_threshold
    // (model is satisfied with every position). Also stops if average confidence
    // across all tokens hasn't improved for `early_stop_patience` passes.
    // Default patience = 0 (disabled). Set to batch_size/2 for auto-scaling,
    // or a fixed value. Works without ground truth.
    int early_stop_patience = 0; // 0 = disabled, -1 = auto (batch_size/2), N = fixed

    // Logging: if non-empty, write per-pass snapshots every snapshot_interval passes
    std::string log_path;
    int snapshot_interval = 5;   // write snapshot every N passes
};

// Full-sequence refinement with configurable strategy.
// Supports replace-all, n-least, and hybrid modes.
// See wir_refine_config for parameters.
wir_nleast_result wir_refine_sequence(
    llama_model  * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,
    const std::vector<llama_token> & ground_truth,
    const wir_refine_config & config);

// Legacy wrapper for backward compatibility
inline wir_nleast_result wir_refine_n_least(
    llama_model  * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,
    const std::vector<llama_token> & ground_truth,
    int n_worst,
    int max_passes)
{
    wir_refine_config cfg;
    cfg.strategy = "n-least";
    cfg.n_worst = n_worst;
    cfg.max_passes = max_passes;
    return wir_refine_sequence(model, ctx, prompt_tokens, candidate, ground_truth, cfg);
}

// Refine the last M tokens of an already-prefilled sequence.
// The context must have the full sequence in its KV cache (all positions).
// `tokens` is the complete token sequence (prompt + generated).
// `n_refine` is the number of tokens from the end to refine.
// `n_passes` is the number of refinement passes.
// Modifies `tokens` in-place for the last n_refine positions.
// Returns the refinement result.
wir_manual_result wir_refine_last_n(
    llama_model  * model,
    llama_context * ctx,
    std::vector<llama_token> & tokens,  // complete sequence, modified in-place
    int n_refine,                       // refine the last N tokens
    int n_passes,                       // number of refinement passes
    int topk_accept = 1);               // accept if in top-K (1 = exact match only)

// Refine specific token positions in an already-prefilled sequence using a mask.
// `tokens` is the complete token sequence (all positions in KV cache).
// `mask` has the same length as `tokens`. mask[i] != 0 means position i is
// eligible for refinement; mask[i] == 0 means position i is frozen.
// This enables surgical editing: fix specific tokens while keeping others locked.
// Modifies `tokens` in-place at masked positions.
wir_manual_result wir_refine_masked(
    llama_model  * model,
    llama_context * ctx,
    std::vector<llama_token> & tokens,      // complete sequence, modified in-place
    const std::vector<int8_t> & mask,       // mask[i] != 0 → refinable
    int n_passes,
    int topk_accept = 1);

// Refine a single block of tokens using the big model.
// Modifies `candidate` in-place for the block range [block_start, block_start+block_len).
// `prompt_tokens` is the prompt prefix (not part of candidate).
// `ground_truth` is optional (can be empty) for agreement tracking.
// `ctx` must have enough context for prompt + candidate up to block_start + block_len.
wir_block_result wir_refine_block(
    llama_model  * model,
    llama_context * ctx,                              // shared context (KV managed by caller or reused)
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,       // modified in-place
    const std::vector<llama_token> & ground_truth,    // optional, for agreement tracking
    int block_start,
    int block_len,
    const wir_params & params);

// Run full wavefront refinement over entire candidate.
// Splits candidate into blocks of `params.block_size` and refines left-to-right.
// Returns aggregate result with per-block details.
wir_result wir_refine_wavefront(
    llama_model  * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,       // modified in-place
    const std::vector<llama_token> & ground_truth,    // optional
    const wir_params & params);

// Run WIR decode: given an imported KV cache (prompt already prefilled),
// generate `n_tokens` new tokens using wavefront refinement with a draft model.
// This is the main entry point for decode receiver integration.
//
// `draft_model` and `draft_ctx` are the small/fast model for initial generation.
// `refine_model` and `refine_ctx` are the big model for refinement.
// `prompt_tokens` is the prompt already in the KV cache of refine_ctx.
// `n_tokens` is how many tokens to generate.
// Returns the generated tokens.
std::vector<llama_token> wir_decode(
    llama_model   * draft_model,
    llama_context * draft_ctx,
    llama_model   * refine_model,
    llama_context * refine_ctx,
    const std::vector<llama_token> & prompt_tokens,
    int n_tokens,
    const wir_params & params);

// Simplified single-model WIR decode: no separate draft model.
// Uses the refine model for both draft (with temperature) and refinement.
// Slower but requires only one model.
std::vector<llama_token> wir_decode_single_model(
    llama_model   * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    int n_tokens,
    float draft_temp,   // temperature for draft generation (e.g. 0.8)
    const wir_params & params);

// Post-KV-import WIR decode: the KV cache already contains the prompt state.
// Generates draft tokens from the current KV state (without clearing it),
// then refines using wavefront block refinement that preserves the imported
// prompt KV and only manipulates the generated token positions.
//
// This is the correct entry point for the decode receiver after KV import.
//
// `prompt_token_count` is the number of tokens in the imported KV state.
// `last_token` is the last token ID from the imported state.
// `n_tokens` is how many new tokens to generate.
// Returns the generated (and refined) tokens.
std::vector<llama_token> wir_decode_from_kv(
    llama_model   * model,
    llama_context * ctx,
    int prompt_token_count,   // number of tokens already in KV cache
    llama_token last_token,   // last token from the imported state
    int n_tokens,
    float draft_temp,
    const wir_params & params,
    const float * prefill_logits = nullptr,  // optional: pre-cached logits predicting first token
    int prefill_logits_n_vocab = 0);         // vocab size for prefill_logits

// Cross-tokenizer WIR decode: uses a separate draft model with a DIFFERENT
// tokenizer. The flow is:
//   1. Draft model generates N tokens autoregressively (its own tokenizer)
//   2. Draft tokens are detokenized to text
//   3. Text is re-tokenized with the refine model's tokenizer
//   4. Refine model runs wavefront block refinement on re-tokenized candidate
//
// This bridges tokenizer boundaries via text space. The re-tokenized candidate
// may have a different token count than N (tokenizers segment differently).
//
// `draft_model/ctx`: small fast model (loads its own vocab, can be different arch)
// `refine_model/ctx`: big model for refinement (KV already contains prompt)
// `prompt_text`: the original prompt string (needed to prefill the draft model)
// `prompt_token_count`: tokens in the refine model's KV cache
// `last_token`: last token ID in the refine model's KV
// `n_tokens`: approximate number of tokens to generate
// `draft_temp`: sampling temperature for draft generation
//
// Returns tokens in the refine model's vocabulary.
std::vector<llama_token> wir_decode_cross_tokenizer(
    llama_model   * draft_model,
    llama_context * draft_ctx,
    llama_model   * refine_model,
    llama_context * refine_ctx,
    const std::string & prompt_text,  // original prompt for draft model prefill
    int prompt_token_count,           // tokens in refine KV cache
    llama_token last_token,           // last token in refine KV
    int n_tokens,
    float draft_temp,
    const wir_params & params);

// ─── Continuous draft→refine loop for long generation ───────────────────────
//
// Generates an unlimited-length response by repeating:
//   1. Draft model generates `chunk_size` tokens
//   2. Re-tokenize draft output for the refine model (cross-tokenizer bridge)
//   3. Big model refines the chunk (replace-all + n-least using wir_refine_config)
//   4. Emit the converged chunk via `emit_callback`
//   5. Feed converged tokens as left-context for the next draft chunk
//   6. Stop when the big model's refinement produces an EOG token, OR max_total reached
//
// Draft model EOG tokens are IGNORED — only the refine model decides when to stop.
//
// `emit_callback`: called for each converged chunk. Return false to abort.
// `max_total`: maximum total tokens to generate across all chunks (0 = unlimited)

struct wir_loop_config {
    int chunk_size           = 128;   // tokens per draft→refine cycle
    int max_total            = 0;     // max total tokens (0 = no limit)
    int min_chunks_before_eog = 2;   // ignore EOG in first N chunks (thinking models)
    wir_refine_config refine;        // refinement settings for each chunk
};

struct wir_loop_result {
    int total_tokens;                 // total tokens generated
    int total_chunks;                 // number of chunks processed
    int total_refine_passes;          // sum of refinement passes across all chunks
    double total_ms;
    bool stopped_by_eog;              // true if refine model produced EOG
};

using wir_emit_cb = std::function<bool(const llama_token * tokens, int count)>;

wir_loop_result wir_continuous_loop(
    llama_model   * draft_model,
    llama_context * draft_ctx,
    llama_model   * refine_model,
    llama_context * refine_ctx,
    const std::string & prompt_text,
    const std::vector<llama_token> & refine_prompt_tokens, // prompt in refine vocab
    const wir_loop_config & config,
    wir_emit_cb emit_callback);
