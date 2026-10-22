// llama-wir.cpp — Wavefront Iterative Refinement (WIR) implementation
//
// Extracted from test-wavefront-convergence.cpp into reusable library.
// See llama-wir.h for API documentation.

#include "llama-wir.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ─── confidence from logits ─────────────────────────────────────────────────

float wir_confidence_from_logits(const float * logits, int n_vocab) {
    float max_val = *std::max_element(logits, logits + n_vocab);

    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum_exp += exp((double)(logits[i] - max_val));
    }
    double log_sum = log(sum_exp);

    double weighted = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        double x = (double)(logits[i] - max_val);
        weighted += x * exp(x);
    }
    double entropy = log_sum - weighted / sum_exp;

    double max_entropy = log((double)n_vocab);
    if (max_entropy < 1e-9) return 1.0f;

    double normalized = entropy / max_entropy;
    return (float)(1.0 - normalized);
}

// ─── Sequence refinement (replace-all, n-least, hybrid) ─────────────────────

wir_nleast_result wir_refine_sequence(
    llama_model  * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,
    const std::vector<llama_token> & ground_truth,
    const wir_refine_config & config)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_cand = (int)candidate.size();
    const int n_total = n_prompt + n_cand;

    // For streaming models: the scheduler was reserved for n_tokens=1 in WF mode.
    // Force a re-reserve so compute buffers are sized for the actual batch.
    if (model->hparams.wf_active) {
        ctx->set_sched_need_reserve(true);
    }

    wir_nleast_result result = {};
    result.total_tokens = n_cand;

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<llama_token> original = candidate;

    auto count_agreement = [&]() -> int {
        int m = 0;
        int n = std::min(n_cand, (int)ground_truth.size());
        for (int i = 0; i < n; i++) {
            if (candidate[i] == ground_truth[i]) m++;
        }
        return m;
    };

    auto detokenize_candidate = [&]() -> std::string {
        std::string out;
        char buf[256];
        for (int i = 0; i < n_cand; i++) {
            int n = llama_token_to_piece(vocab, candidate[i], buf, sizeof(buf), 0, true);
            if (n > 0) out.append(buf, n);
        }
        return out;
    };

    // Wavefront locking state: locked[j] = true means candidate[j] is frozen
    std::vector<bool> locked(n_cand, false);
    int last_locked_pos = -1; // position of the rightmost locked token

    // Open log file if configured
    FILE * log_fp = nullptr;
    if (!config.log_path.empty()) {
        log_fp = fopen(config.log_path.c_str(), "w");
        if (log_fp) {
            fprintf(log_fp, "# WIR refinement log: strategy=%s n_worst=%d confidence_threshold=%.2f min_gap=%d\n",
                    config.strategy.c_str(), config.n_worst, config.confidence_threshold, config.min_token_gap);
            fprintf(log_fp, "# Candidate tokens: %d, Ground truth tokens: %d\n\n", n_cand, (int)ground_truth.size());
        }
    }

    // Per-position info collected each pass
    struct pos_info {
        int   cand_idx;
        float log_prob;      // log-probability of current candidate token
        float confidence;    // softmax probability of current token (0-1)
        float top1_prob;     // softmax probability of top-1 prediction
        llama_token top1;
    };

    // Track whether incremental KV decode works for this model.
    // Hybrid/recurrent models may not support partial seq_rm.
    bool incremental_kv_ok = true;

    for (int pass = 1; pass <= config.max_passes; pass++) {
        auto t_pass_start = std::chrono::high_resolution_clock::now();

        // Determine strategy for this pass
        bool use_replace_all;
        if (config.strategy == "replace-all") {
            use_replace_all = true;
        } else if (config.strategy == "hybrid") {
            use_replace_all = (pass <= config.replace_all_passes);
        } else {
            use_replace_all = false; // n-least
        }

        // Incremental KV optimization: on pass 1 decode prompt+candidate.
        // On subsequent passes, keep the prompt KV intact and only re-decode
        // the candidate region. For streaming 397B models this avoids re-streaming
        // hundreds of GB of weights per refinement pass.
        //
        // We also need the logits at position (n_prompt - 1) which predicts candidate[0].
        // On pass 1 this comes from the full decode. On subsequent passes we
        // re-decode from (n_prompt - 1) so the prompt's last token produces
        // logits predicting candidate[0] with the correct left-context.

        // For streaming models: use ubatch size for chunking so each llama_decode
        // call fits in a single ubatch. This ensures the graph_compute_per_layer
        // fast path fires (which requires n_ubatches==1) and the per-layer
        // weight fetcher callbacks work correctly.
        // For non-streaming: use the full batch size for fewer decode calls.
        const bool wf_streaming = model ? model->hparams.wf_active : false;
        const int batch_max = wf_streaming ? (int)llama_n_ubatch(ctx) : (int)llama_n_batch(ctx);

        // Helper lambda: full decode (clear all + decode prompt+candidate)
        auto full_decode = [&]() -> bool {
            llama_memory_clear(llama_get_memory(ctx), true);

            // Sync GPU to clear any pending CUDA errors from the previous
            // streaming pass before resetting state.
            llama_synchronize(ctx);

            // Reset streaming/handoff state so the streaming prefill path can
            // re-enter cleanly.
            ctx->reset_streaming_state();

            std::vector<llama_token> seq;
            seq.reserve(n_total);
            seq.insert(seq.end(), prompt_tokens.begin(), prompt_tokens.end());
            seq.insert(seq.end(), candidate.begin(), candidate.end());

            for (int off = 0; off < n_total; off += batch_max) {
                // For streaming: reset state before each chunk so the streaming
                // path can re-enter cleanly (handoff/prefill_done from previous chunk)
                if (wf_streaming && off > 0) {
                    ctx->reset_streaming_state();
                }
                int blen = std::min(batch_max, n_total - off);
                llama_batch batch = llama_batch_init(blen, 0, 1);
                batch.n_tokens = blen;
                for (int i = 0; i < blen; i++) {
                    batch.token[i]     = seq[off + i];
                    batch.pos[i]       = off + i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    int pos = off + i;
                    batch.logits[i] = (pos >= n_prompt - 1 && pos < n_total - 1) ? 1 : 0;
                }
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "wir_refine_sequence: full decode failed pass %d offset %d\n", pass, off);
                    llama_batch_free(batch);
                    return false;
                }
                llama_batch_free(batch);
            }
            return true;
        };

        // Helper lambda: incremental decode (keep prompt KV, re-decode candidate only)
        auto incremental_decode = [&]() -> bool {
            int rm_from = std::max(0, n_prompt - 1);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, rm_from, -1);

            // Reset streaming state so the streaming prefill path re-streams
            // all layers from NVMe.  Without this, the WF reuses stale GPU
            // buffer data from the previous pass.
            ctx->reset_streaming_state();

            int redecode_start = rm_from;
            int redecode_len = n_total - redecode_start;

            std::vector<llama_token> redecode_seq;
            redecode_seq.reserve(redecode_len);
            if (n_prompt > 0 && rm_from == n_prompt - 1) {
                redecode_seq.push_back(prompt_tokens[n_prompt - 1]);
            }
            redecode_seq.insert(redecode_seq.end(), candidate.begin(), candidate.end());

            for (int off = 0; off < redecode_len; off += batch_max) {
                int blen = std::min(batch_max, redecode_len - off);
                llama_batch batch = llama_batch_init(blen, 0, 1);
                batch.n_tokens = blen;
                for (int i = 0; i < blen; i++) {
                    batch.token[i]     = redecode_seq[off + i];
                    batch.pos[i]       = redecode_start + off + i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    int pos = redecode_start + off + i;
                    batch.logits[i] = (pos >= n_prompt - 1 && pos < n_total - 1) ? 1 : 0;
                }
                if (llama_decode(ctx, batch) != 0) {
                    llama_batch_free(batch);
                    return false; // signal caller to fall back to full decode
                }
                llama_batch_free(batch);
            }
            return true;
        };

        // Incremental KV optimization: on pass 1 always do full decode.
        // On subsequent passes, try incremental (keep prompt KV). If it fails
        // (e.g., hybrid/recurrent models that don't support partial seq_rm),
        // fall back to full decode and disable incremental for future passes.
        //
        // For streaming models (wf_active), always use full_decode. Incremental
        // KV saves compute but not I/O — every pass must stream all weights from
        // NVMe regardless. And the streaming path needs reset_streaming_state()
        // + memory_clear() to re-enter cleanly (seq_rm leaves stale recurrent
        // state that breaks M-RoPE position checks).
        const bool force_full = model ? model->hparams.wf_active : false;

        if (pass == 1 || !incremental_kv_ok || force_full) {
            if (!full_decode()) goto done;
        } else {
            if (!incremental_decode()) {
                fprintf(stderr, "[wir] incremental KV decode failed on pass %d, falling back to full decode\n", pass);
                incremental_kv_ok = false;
                if (!full_decode()) goto done;
            }
        }

        // Collect per-position info
        {
            std::vector<pos_info> positions(n_cand);

            // Pre-compute token frequency map once per pass (used by repetition penalty)
            std::unordered_map<llama_token, int> freq;
            // Pre-compute escalated penalty per token (penalty^count)
            std::unordered_map<llama_token, float> escalated_penalties;
            if (config.repetition_penalty > 1.0f) {
                for (int k = 0; k < n_cand; k++) {
                    freq[candidate[k]]++;
                }
                for (auto & [tok, count] : freq) {
                    if (count <= 1) continue;
                    float escalated = 1.0f;
                    for (int c = 0; c < count; c++) {
                        escalated *= config.repetition_penalty;
                    }
                    escalated_penalties[tok] = escalated;
                }
            }

            // Pre-compute n-gram data structures once per pass.
            // For n-gram penalty: build set of all n-grams (hashed) to detect repeats.
            // For frequency penalty: build count map of all m-grams.
            // Uses FNV-1a hash of token sequences as keys.
            auto hash_ngram = [&](int start, int len) -> uint64_t {
                uint64_t h = 14695981039346656037ULL; // FNV offset basis
                for (int g = 0; g < len; g++) {
                    h ^= (uint64_t)(uint32_t)candidate[start + g];
                    h *= 1099511628211ULL; // FNV prime
                }
                return h;
            };

            // N-gram penalty: for each position j, check if the n-gram ending at j-1
            // (i.e. candidate[j-ngl..j-1]) appeared earlier. We pre-index all n-grams.
            std::unordered_set<uint64_t> ngram_set;
            // ngram_seen_at[j] = true if the n-gram ending at j duplicates an earlier one
            std::vector<bool> ngram_is_repeat(n_cand, false);
            if (config.ngram_penalty_len > 0) {
                int ngl = config.ngram_penalty_len;
                for (int j = ngl - 1; j < n_cand; j++) {
                    uint64_t h = hash_ngram(j - ngl + 1, ngl);
                    if (ngram_set.count(h)) {
                        ngram_is_repeat[j] = true;
                    }
                    ngram_set.insert(h);
                }
            }

            // Frequency n-gram penalty: count occurrences of each m-gram
            std::unordered_map<uint64_t, int> ngram_freq_counts;
            // ngram_freq_at[j] = count of the m-gram ending at position j
            std::vector<int> ngram_freq_at(n_cand, 0);
            if (config.ngram_freq_min_len > 0) {
                int mlen = config.ngram_freq_min_len;
                for (int j = mlen - 1; j < n_cand; j++) {
                    uint64_t h = hash_ngram(j - mlen + 1, mlen);
                    ngram_freq_counts[h]++;
                    ngram_freq_at[j] = ngram_freq_counts[h];
                }
            }

            // Top-K acceptance: filled per-position after penalties are applied
            std::vector<bool> in_topk(n_cand, false);

            for (int j = 0; j < n_cand; j++) {
                int logit_pos = n_prompt - 1 + j;
                const float * logits = llama_get_logits_ith(ctx, logit_pos);

                if (!logits) {
                    positions[j] = {j, 0.0f, 1.0f, 1.0f, candidate[j]};
                    continue;
                }

                // Apply repetition penalty: copy logits and penalize repeated tokens
                std::vector<float> adj_logits(logits, logits + n_vocab);

                if (!escalated_penalties.empty()) {
                    // Apply pre-computed escalated penalties
                    for (auto & [tok, escalated] : escalated_penalties) {
                        if (adj_logits[tok] > 0) {
                            adj_logits[tok] /= escalated;
                        } else {
                            adj_logits[tok] *= escalated;
                        }
                    }
                }

                // N-gram penalty: use pre-computed repeat detection
                if (config.ngram_penalty_len > 0 && j >= config.ngram_penalty_len && ngram_is_repeat[j]) {
                    adj_logits[candidate[j]] += config.ngram_penalty_weight;
                }

                // Frequency-based n-gram penalty: use pre-computed counts
                if (config.ngram_freq_min_len > 0 && j >= config.ngram_freq_min_len) {
                    int count = ngram_freq_at[j];
                    if (count > 1) {
                        adj_logits[candidate[j]] += config.ngram_freq_penalty * (count - 1);
                    }
                }

                const float * eff_logits = adj_logits.data();

                // Top-K acceptance check using PENALTY-ADJUSTED logits.
                // This ensures repeated tokens get penalized before the topk check,
                // so "fetchingfetching" won't be accepted just because "fetching"
                // is in the raw top-10.
                if (config.topk_accept > 1) {
                    std::vector<std::pair<float, llama_token>> topk_buf(config.topk_accept, {-1e30f, 0});
                    for (int v = 0; v < n_vocab; v++) {
                        if (eff_logits[v] > topk_buf.back().first) {
                            topk_buf.back() = {eff_logits[v], (llama_token)v};
                            for (int t = config.topk_accept - 1; t > 0 && topk_buf[t].first > topk_buf[t-1].first; t--)
                                std::swap(topk_buf[t], topk_buf[t-1]);
                        }
                    }
                    for (auto & [s, id] : topk_buf) {
                        if (id == candidate[j]) { in_topk[j] = true; break; }
                    }
                }

                // Select replacement token (using penalty-adjusted logits)
                llama_token top1 = 0;
                float top1_val = eff_logits[0];
                if (config.refine_temp > 0.0f) {
                    // Temperature sampling: sample from softmax(logits/temp)
                    std::vector<float> probs(n_vocab);
                    float mx = eff_logits[0];
                    for (int v = 1; v < n_vocab; v++) mx = std::max(mx, eff_logits[v]);
                    double sum = 0;
                    for (int v = 0; v < n_vocab; v++) {
                        probs[v] = expf((eff_logits[v] - mx) / config.refine_temp);
                        sum += probs[v];
                    }
                    double r = ((double)rand() / RAND_MAX) * sum;
                    double cum = 0;
                    for (int v = 0; v < n_vocab; v++) {
                        cum += probs[v];
                        if (cum >= r) { top1 = v; top1_val = eff_logits[v]; break; }
                    }
                } else {
                    // Greedy: pick top-1
                    for (int v = 1; v < n_vocab; v++) {
                        if (eff_logits[v] > top1_val) { top1_val = eff_logits[v]; top1 = v; }
                    }
                }

                // Log-probability and confidence of current token (using original logits)
                float max_val = logits[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (logits[v] > max_val) max_val = logits[v];
                }
                double sum_exp = 0.0;
                for (int v = 0; v < n_vocab; v++) {
                    sum_exp += exp((double)(logits[v] - max_val));
                }
                float log_sum = (float)log(sum_exp);
                float log_prob = logits[candidate[j]] - max_val - log_sum;
                float prob = expf(log_prob);
                float top1_log_prob = eff_logits[top1] - max_val - log_sum;
                float top1_p = expf(top1_log_prob);

                positions[j] = {j, log_prob, prob, top1_p, top1};
            }

            // (in_topk was filled per-position inside the loop above using
            // penalty-adjusted logits, so rep/ngram penalties override topk)

            int changed = 0;

            // Determine the effective lock and change thresholds
            const float eff_lock_conf = config.lock_conf > 0.0f ? config.lock_conf : config.confidence_threshold;
            const float eff_change_conf = config.change_conf > 0.0f ? config.change_conf : config.confidence_threshold;

            // LOCK BEFORE REPLACE: scan from the left and lock contiguous tokens
            // that the model agrees with. These won't be touched by replacement.
            if (config.wavefront_lock_window > 0) {
                // Unlock strongly-disagreed tokens first
                if (config.soft_lock_margin > 0.0f) {
                    int unlocked = 0;
                    for (int j = 0; j < n_cand; j++) {
                        if (!locked[j]) continue;
                        if (positions[j].top1 == candidate[j]) continue;
                        float margin = positions[j].top1_prob - positions[j].confidence;
                        if (margin > config.soft_lock_margin * 2.0f) {
                            locked[j] = false;
                            unlocked++;
                        }
                    }
                    if (unlocked > 0) {
                        last_locked_pos = -1;
                        for (int j = n_cand - 1; j >= 0; j--) {
                            if (locked[j]) { last_locked_pos = j; break; }
                        }
                        fprintf(stderr, "[wavefront] pass %d: UNLOCKED %d tokens, frontier at %d/%d\n",
                                pass, unlocked, last_locked_pos + 1, n_cand);
                    }
                }

                // Lock contiguous agreeing tokens from the left
                int newly_locked = 0;
                for (int j = last_locked_pos + 1; j < n_cand; j++) {
                    if (locked[j]) continue;
                    if (positions[j].top1 == candidate[j] &&
                        positions[j].top1_prob >= eff_lock_conf) {
                        locked[j] = true;
                        last_locked_pos = j;
                        newly_locked++;
                    } else {
                        break;
                    }
                }
                if (newly_locked > 0) {
                    fprintf(stderr, "[wavefront] pass %d: locked %d tokens, frontier at %d/%d\n",
                            pass, newly_locked, last_locked_pos + 1, n_cand);
                }
            }

            if (use_replace_all) {
                for (int j = 0; j < n_cand; j++) {
                    if (locked[j]) continue;
                    // Skip if in top-K OR if model is confident about current token
                    if (config.topk_accept > 1 ? in_topk[j] : (positions[j].top1 == candidate[j])) continue;
                    if (positions[j].confidence >= eff_change_conf) continue;
                    // Soft lock: only replace if new token is significantly more probable
                    if (config.soft_lock_margin > 0.0f &&
                        (positions[j].top1_prob - positions[j].confidence) < config.soft_lock_margin) continue;
                    candidate[j] = positions[j].top1;
                    changed++;
                }
            } else {
                // N-least: sort by log_prob, pick N worst with adjacency + confidence + lock constraints
                std::vector<int> sorted_indices(n_cand);
                std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
                std::sort(sorted_indices.begin(), sorted_indices.end(),
                          [&](int a, int b) { return positions[a].log_prob < positions[b].log_prob; });

                std::vector<bool> blocked(n_cand, false);

                for (int k = 0; k < n_cand && changed < config.n_worst; k++) {
                    int idx = sorted_indices[k];

                    if (locked[idx]) continue;                                    // wavefront-locked
                    if (config.topk_accept > 1 ? in_topk[idx] : (positions[idx].top1 == candidate[idx])) continue;
                    if (positions[idx].confidence >= eff_change_conf) continue;
                    if (blocked[idx]) continue;                                   // adjacency blocked
                    // Soft lock: only replace if new token is significantly more probable
                    if (config.soft_lock_margin > 0.0f &&
                        (positions[idx].top1_prob - positions[idx].confidence) < config.soft_lock_margin) continue;

                    candidate[idx] = positions[idx].top1;
                    changed++;

                    for (int g = 1; g <= config.min_token_gap; g++) {
                        if (idx - g >= 0)    blocked[idx - g] = true;
                        if (idx + g < n_cand) blocked[idx + g] = true;
                    }
                }
            }

            // (Locking now happens BEFORE replacement — see above)

            // Detokenize-retokenize: fix token boundary artifacts.
            // After corrections, the concatenated text may have merged words
            // ("partTransport") that only exist because token N and N+1 were
            // independently correct but form a garbled word together.
            // Re-tokenizing re-aligns boundaries correctly.
            // Debug: track frontier token
            {
                int fp = last_locked_pos + 1;
                if (fp < n_cand && (pass <= 5 || pass % 20 == 0 || changed <= 3)) {
                    char buf_cur[64], buf_top[64];
                    int nc = llama_token_to_piece(vocab, candidate[fp], buf_cur, sizeof(buf_cur), 0, true);
                    int nt = llama_token_to_piece(vocab, positions[fp].top1, buf_top, sizeof(buf_top), 0, true);
                    if (nc > 0) buf_cur[nc] = 0; else buf_cur[0] = 0;
                    if (nt > 0) buf_top[nt] = 0; else buf_top[0] = 0;
                    fprintf(stderr, "[debug] pass %d: frontier=%d cur='%s'(%d) top1='%s'(%d) match=%d prob=%.3f changed=%d\n",
                            pass, fp, buf_cur, candidate[fp], buf_top, positions[fp].top1,
                            (int)(positions[fp].top1 == candidate[fp]),
                            positions[fp].top1_prob, changed);
                }
            }

            // NOTE: retokenize removed from per-pass loop. It belongs in the
            // cross-tokenizer bridge (when draft and refine have different tokenizers),
            // not in the refinement loop where it causes oscillation.

            auto t_pass_end = std::chrono::high_resolution_clock::now();
            double pass_ms = std::chrono::duration<double, std::milli>(t_pass_end - t_pass_start).count();

            int matching = count_agreement();
            float agree_pct = n_cand > 0 ? 100.0f * matching / n_cand : 0.0f;

            wir_pass_stats ps = {};
            ps.pass = pass;
            ps.tokens_changed = changed;
            ps.tokens_matching = matching;
            ps.total_tokens = n_cand;
            ps.agreement_pct = agree_pct;
            ps.pass_ms = pass_ms;
            result.per_pass.push_back(ps);

            const char * mode_str = use_replace_all ? "replace-all" : "n-least";
            fprintf(stderr, "[%s] pass %d: changed=%d matching=%d/%d (%.1f%%) %.0fms\n",
                    mode_str, pass, changed, matching, n_cand, agree_pct, pass_ms);

            // Snapshot to log file
            if (log_fp && (pass % config.snapshot_interval == 0 || pass == 1 || agree_pct >= 100.0f)) {
                fprintf(log_fp, "=== Pass %d (%s) — changed=%d matching=%d/%d (%.1f%%) %.0fms ===\n",
                        pass, mode_str, changed, matching, n_cand, agree_pct, pass_ms);
                std::string text = detokenize_candidate();
                fprintf(log_fp, "%s\n\n", text.c_str());
                fflush(log_fp);
            }

            if (changed == 0 || agree_pct >= 100.0f) break;

            // Early stop: check if ALL unlocked tokens are above confidence threshold.
            {
                int patience = config.early_stop_patience;
                if (patience == -1) patience = n_cand / 2; // auto-scale to batch_size/2

                // Compute average confidence across all tokens
                float avg_conf = 0.0f;
                int n_unlocked = 0;
                for (int j = 0; j < n_cand; j++) {
                    if (!locked[j]) {
                        avg_conf += positions[j].confidence;
                        n_unlocked++;
                    }
                }
                if (n_unlocked > 0) avg_conf /= n_unlocked;

                // All tokens above threshold → stop
                bool all_above = true;
                for (int j = 0; j < n_cand; j++) {
                    if (!locked[j] && positions[j].confidence < config.confidence_threshold) {
                        all_above = false;
                        break;
                    }
                }
                if (all_above) {
                    fprintf(stderr, "[early-stop] all tokens above %.0f%% confidence (avg=%.1f%%), stopping\n",
                            config.confidence_threshold * 100.0f, avg_conf * 100.0f);
                    break;
                }

                // Stall: average confidence hasn't improved for `patience` passes
                // Store avg_conf in per_pass for tracking (overload agreement_pct)
                if (patience > 0 && !result.per_pass.empty()) {
                    // The per_pass already stores agreement_pct; we'll check if changes stalled
                    if ((int)result.per_pass.size() >= patience) {
                        // Check if tokens_changed has been the same for the last `patience` passes
                        bool changes_stalled = true;
                        int ws = (int)result.per_pass.size() - patience;
                        for (int ep = ws + 1; ep < (int)result.per_pass.size(); ep++) {
                            if (result.per_pass[ep].tokens_changed != result.per_pass[ws].tokens_changed) {
                                changes_stalled = false;
                                break;
                            }
                        }
                        if (changes_stalled) {
                            fprintf(stderr, "[early-stop] refinement stalled for %d passes (avg_conf=%.1f%%), stopping\n",
                                    patience, avg_conf * 100.0f);
                            break;
                        }
                    }
                }
            }
        }
    }

done:
    if (log_fp) {
        fprintf(log_fp, "=== FINAL — %d passes ===\n", (int)result.per_pass.size());
        std::string text = detokenize_candidate();
        fprintf(log_fp, "%s\n", text.c_str());
        fclose(log_fp);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.passes_used = (int)result.per_pass.size();

    int final_matching = count_agreement();
    result.final_agreement_pct = n_cand > 0 ? 100.0f * final_matching / n_cand : 0.0f;

    result.final_changed = 0;
    for (int i = 0; i < n_cand; i++) {
        if (candidate[i] != original[i]) result.final_changed++;
    }

    return result;
}

// ─── manual refinement: refine last N tokens ────────────────────────────────

wir_manual_result wir_refine_last_n(
    llama_model  * model,
    llama_context * ctx,
    std::vector<llama_token> & tokens,
    int n_refine,
    int n_passes,
    int topk_accept)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_total = (int)tokens.size();

    if (n_refine <= 0 || n_refine > n_total || n_passes <= 0) {
        return {n_total, 0, 0, 0, 0.0};
    }

    const int refine_start = n_total - n_refine;
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> original(tokens.begin() + refine_start, tokens.end());
    int total_changed = 0;

    for (int pass = 0; pass < n_passes; pass++) {
        // Remove KV from refine_start - 1 onwards (need left-context logits)
        llama_memory_t mem = llama_get_memory(ctx);
        int rm_from = std::max(0, refine_start - 1);
        llama_memory_seq_rm(mem, 0, rm_from, -1);

        // Build batch: left-context token + refinable tokens
        int left_extra = (refine_start > 0) ? 1 : 0;
        int batch_len = n_refine + left_extra;
        llama_batch batch = llama_batch_init(batch_len, 0, 1);
        batch.n_tokens = batch_len;

        int bi = 0;
        if (left_extra) {
            batch.token[bi]     = tokens[refine_start - 1];
            batch.pos[bi]       = refine_start - 1;
            batch.n_seq_id[bi]  = 1;
            batch.seq_id[bi][0] = 0;
            batch.logits[bi]    = 1;
            bi++;
        }
        for (int j = 0; j < n_refine; j++, bi++) {
            batch.token[bi]     = tokens[refine_start + j];
            batch.pos[bi]       = refine_start + j;
            batch.n_seq_id[bi]  = 1;
            batch.seq_id[bi][0] = 0;
            batch.logits[bi]    = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "wir_refine_last_n: decode failed at pass %d\n", pass);
            llama_batch_free(batch);
            break;
        }

        int changed = 0;
        // Logits at batch[left_extra + j - 1] predict tokens[refine_start + j]
        // For j == 0 with left_extra: batch[0] logits predict tokens[refine_start]
        for (int j = 0; j < n_refine; j++) {
            int logit_idx = (j == 0 && left_extra) ? 0 : (left_extra + j - 1);
            if (j == 0 && !left_extra) continue; // can't predict first without left context
            const float * logits = llama_get_logits_ith(ctx, logit_idx);
            if (!logits) continue;

            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }

            if (topk_accept > 1) {
                // Check if current token is in top-K
                bool in_topk = false;
                std::vector<std::pair<float, int>> top(topk_accept, {-1e30f, 0});
                for (int v = 0; v < n_vocab; v++) {
                    if (logits[v] > top.back().first) {
                        top.back() = {logits[v], v};
                        std::sort(top.begin(), top.end(), [](auto & a, auto & b) { return a.first > b.first; });
                    }
                }
                for (auto & [s, id] : top) {
                    if (id == tokens[refine_start + j]) { in_topk = true; break; }
                }
                if (!in_topk) {
                    tokens[refine_start + j] = best;
                    changed++;
                }
            } else if (best != tokens[refine_start + j]) {
                tokens[refine_start + j] = best;
                changed++;
            }
        }

        total_changed += changed;
        llama_batch_free(batch);
        if (changed == 0) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    int final_changed = 0;
    for (int j = 0; j < n_refine; j++) {
        if (tokens[refine_start + j] != original[j]) final_changed++;
    }

    return {n_total, n_refine, final_changed, n_passes, ms};
}

// ─── manual refinement: refine masked positions ─────────────────────────────

wir_manual_result wir_refine_masked(
    llama_model  * model,
    llama_context * ctx,
    std::vector<llama_token> & tokens,
    const std::vector<int8_t> & mask,
    int n_passes,
    int topk_accept)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_total = (int)tokens.size();

    if (n_total == 0 || (int)mask.size() != n_total || n_passes <= 0) {
        return {n_total, 0, 0, 0, 0.0};
    }

    // Find first and last refinable positions
    int first_refinable = -1, last_refinable = -1;
    int n_refinable = 0;
    for (int i = 0; i < n_total; i++) {
        if (mask[i]) {
            if (first_refinable < 0) first_refinable = i;
            last_refinable = i;
            n_refinable++;
        }
    }
    if (n_refinable == 0) {
        return {n_total, 0, 0, 0, 0.0};
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<llama_token> original = tokens;

    for (int pass = 0; pass < n_passes; pass++) {
        // Remove KV from (first_refinable - 1) onwards to re-decode
        llama_memory_t mem = llama_get_memory(ctx);
        int rm_from = std::max(0, first_refinable - 1);
        llama_memory_seq_rm(mem, 0, rm_from, -1);

        // Re-decode from rm_from to last_refinable (inclusive)
        int span_start = rm_from;
        int span_end = last_refinable + 1; // exclusive
        int batch_len = span_end - span_start;

        llama_batch batch = llama_batch_init(batch_len, 0, 1);
        batch.n_tokens = batch_len;
        for (int i = 0; i < batch_len; i++) {
            batch.token[i]     = tokens[span_start + i];
            batch.pos[i]       = span_start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "wir_refine_masked: decode failed at pass %d\n", pass);
            llama_batch_free(batch);
            break;
        }

        int changed = 0;
        // Logits at batch position (pos - span_start) predict the token at pos+1
        // So to predict tokens[t], we need logits at batch position (t-1 - span_start)
        for (int t = span_start + 1; t <= last_refinable; t++) {
            if (!mask[t]) continue; // frozen position

            int logit_batch_idx = t - 1 - span_start;
            if (logit_batch_idx < 0 || logit_batch_idx >= batch_len) continue;

            const float * logits = llama_get_logits_ith(ctx, logit_batch_idx);
            if (!logits) continue;

            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }

            if (topk_accept > 1) {
                bool in_topk = false;
                std::vector<std::pair<float, int>> top(topk_accept, {-1e30f, 0});
                for (int v = 0; v < n_vocab; v++) {
                    if (logits[v] > top.back().first) {
                        top.back() = {logits[v], v};
                        std::sort(top.begin(), top.end(), [](auto & a, auto & b) { return a.first > b.first; });
                    }
                }
                for (auto & [s, id] : top) {
                    if (id == tokens[t]) { in_topk = true; break; }
                }
                if (!in_topk) {
                    tokens[t] = best;
                    changed++;
                }
            } else if (best != tokens[t]) {
                tokens[t] = best;
                changed++;
            }
        }

        llama_batch_free(batch);
        if (changed == 0) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    int final_changed = 0;
    for (int i = 0; i < n_total; i++) {
        if (tokens[i] != original[i]) final_changed++;
    }

    return {n_total, n_refinable, final_changed, n_passes, ms};
}

// ─── helper: find top-K tokens from logits ──────────────────────────────────

struct scored_token {
    llama_token id;
    float score;
};

static void topk_from_logits(const float * logits, int n_vocab, int k,
                              std::vector<scored_token> & out) {
    out.clear();
    out.reserve(k);
    for (int v = 0; v < n_vocab; v++) {
        if ((int)out.size() < k) {
            out.push_back({v, logits[v]});
            if ((int)out.size() == k) {
                std::sort(out.begin(), out.end(),
                          [](const scored_token & a, const scored_token & b) {
                              return a.score > b.score;
                          });
            }
        } else if (logits[v] > out.back().score) {
            out.back() = {v, logits[v]};
            // re-sort to keep descending order
            for (int i = k - 1; i > 0 && out[i].score > out[i-1].score; i--) {
                std::swap(out[i], out[i-1]);
            }
        }
    }
}

static bool token_in_topk(const std::vector<scored_token> & topk, llama_token id) {
    for (const auto & t : topk) {
        if (t.id == id) return true;
    }
    return false;
}

// ─── block refinement ───────────────────────────────────────────────────────

wir_block_result wir_refine_block(
    llama_model  * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,
    const std::vector<llama_token> & ground_truth,
    int block_start,
    int block_len,
    const wir_params & params)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_ctx_tokens = n_prompt + block_start + block_len;
    const int left_ctx_end_pos = n_prompt + block_start - 1;

    auto t_block_start = std::chrono::high_resolution_clock::now();
    double total_prefill_toks = 0.0;

    // measure initial block agreement
    const bool has_gt = !ground_truth.empty();
    int n_compare = has_gt ? std::min(block_len, (int)ground_truth.size() - block_start) : block_len;
    auto count_block_matches = [&]() -> int {
        if (!has_gt) return 0;
        int m = 0;
        for (int j = 0; j < n_compare; j++) {
            if (candidate[block_start + j] == ground_truth[block_start + j]) m++;
        }
        return m;
    };
    float init_agree = has_gt && n_compare > 0
        ? (float)count_block_matches() / n_compare * 100.0f : 0.0f;

    bool kv_cached = false;
    std::vector<float> cached_left_logits;
    std::vector<scored_token> topk_buf;

    int pass;
    for (pass = 1; pass <= params.max_passes; pass++) {
        // Try partial decode with KV reuse
        bool did_partial = false;

        if (params.reuse_kv && kv_cached) {
            llama_memory_t mem = llama_get_memory(ctx);
            bool need_left_logits = cached_left_logits.empty() && left_ctx_end_pos >= 0;
            int rm_from = need_left_logits ? left_ctx_end_pos : (n_prompt + block_start);
            bool ok = llama_memory_seq_rm(mem, 0, rm_from, -1);

            if (ok) {
                int extra = need_left_logits ? 1 : 0;
                int batch_len = block_len + extra;
                llama_batch batch = llama_batch_init(batch_len, 0, 1);
                batch.n_tokens = batch_len;

                int bi = 0;
                if (need_left_logits) {
                    if (left_ctx_end_pos < n_prompt) {
                        batch.token[bi] = prompt_tokens[left_ctx_end_pos];
                    } else {
                        batch.token[bi] = candidate[left_ctx_end_pos - n_prompt];
                    }
                    batch.pos[bi]       = left_ctx_end_pos;
                    batch.n_seq_id[bi]  = 1;
                    batch.seq_id[bi][0] = 0;
                    batch.logits[bi]    = 1;
                    bi++;
                }
                for (int j = 0; j < block_len; j++, bi++) {
                    batch.token[bi]     = candidate[block_start + j];
                    batch.pos[bi]       = n_prompt + block_start + j;
                    batch.n_seq_id[bi]  = 1;
                    batch.seq_id[bi][0] = 0;
                    batch.logits[bi]    = 1;
                }

                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "wir_refine_block: partial decode failed\n");
                    llama_batch_free(batch);
                    break;
                }
                total_prefill_toks += batch_len;

                int changed = 0;

                // Process logits for block tokens
                if (need_left_logits) {
                    const float * logits = llama_get_logits_ith(ctx, 0);
                    if (logits) {
                        cached_left_logits.assign(logits, logits + n_vocab);
                        llama_token best = 0;
                        float best_val = logits[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
                        }
                        if (params.topk_accept > 1) {
                            topk_from_logits(logits, n_vocab, params.topk_accept, topk_buf);
                            if (!token_in_topk(topk_buf, candidate[block_start])) {
                                candidate[block_start] = best;
                                changed++;
                            }
                        } else if (best != candidate[block_start]) {
                            candidate[block_start] = best;
                            changed++;
                        }
                    }
                    for (int j = 1; j < block_len; j++) {
                        const float * logits2 = llama_get_logits_ith(ctx, extra + j - 1);
                        if (!logits2) continue;
                        llama_token best = 0;
                        float best_val = logits2[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (logits2[v] > best_val) { best_val = logits2[v]; best = v; }
                        }
                        if (params.topk_accept > 1) {
                            topk_from_logits(logits2, n_vocab, params.topk_accept, topk_buf);
                            if (!token_in_topk(topk_buf, candidate[block_start + j])) {
                                candidate[block_start + j] = best;
                                changed++;
                            }
                        } else if (best != candidate[block_start + j]) {
                            candidate[block_start + j] = best;
                            changed++;
                        }
                    }
                } else if (!cached_left_logits.empty()) {
                    // Use cached logits for block[0]
                    llama_token best = 0;
                    float best_val = cached_left_logits[0];
                    for (int v = 1; v < n_vocab; v++) {
                        if (cached_left_logits[v] > best_val) { best_val = cached_left_logits[v]; best = v; }
                    }
                    if (params.topk_accept > 1) {
                        topk_from_logits(cached_left_logits.data(), n_vocab, params.topk_accept, topk_buf);
                        if (!token_in_topk(topk_buf, candidate[block_start])) {
                            candidate[block_start] = best;
                            changed++;
                        }
                    } else if (best != candidate[block_start]) {
                        candidate[block_start] = best;
                        changed++;
                    }
                    for (int j = 1; j < block_len; j++) {
                        const float * logits2 = llama_get_logits_ith(ctx, j - 1);
                        if (!logits2) continue;
                        llama_token best2 = 0;
                        float best_val2 = logits2[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (logits2[v] > best_val2) { best_val2 = logits2[v]; best2 = v; }
                        }
                        if (params.topk_accept > 1) {
                            topk_from_logits(logits2, n_vocab, params.topk_accept, topk_buf);
                            if (!token_in_topk(topk_buf, candidate[block_start + j])) {
                                candidate[block_start + j] = best2;
                                changed++;
                            }
                        } else if (best2 != candidate[block_start + j]) {
                            candidate[block_start + j] = best2;
                            changed++;
                        }
                    }
                }

                llama_batch_free(batch);
                did_partial = true;

                if (changed == 0) break;
                continue;
            }
            // partial removal failed — fall through to full prefill
        }

        if (!did_partial) {
            // Full prefill: clear all memory, decode entire sequence
            llama_memory_clear(llama_get_memory(ctx), true);

            std::vector<llama_token> seq;
            seq.reserve(n_ctx_tokens);
            seq.insert(seq.end(), prompt_tokens.begin(), prompt_tokens.end());
            seq.insert(seq.end(), candidate.begin(), candidate.begin() + block_start + block_len);

            llama_batch batch = llama_batch_init(n_ctx_tokens, 0, 1);
            batch.n_tokens = n_ctx_tokens;
            for (int i = 0; i < n_ctx_tokens; i++) {
                batch.token[i]     = seq[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                int first_logit_pos = n_prompt + block_start - 1;
                int last_logit_pos  = n_prompt + block_start + block_len - 1;
                batch.logits[i] = (i >= first_logit_pos && i <= last_logit_pos) ? 1 : 0;
            }

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "wir_refine_block: full decode failed\n");
                llama_batch_free(batch);
                break;
            }
            total_prefill_toks += n_ctx_tokens;

            // Cache left-context logits for subsequent passes
            if (params.reuse_kv && left_ctx_end_pos >= 0) {
                const float * ll = llama_get_logits_ith(ctx, left_ctx_end_pos);
                if (ll) {
                    cached_left_logits.assign(ll, ll + n_vocab);
                }
                kv_cached = true;
            }

            // Replace tokens in this block
            int changed = 0;
            for (int j = 0; j < block_len; j++) {
                int logit_pos = n_prompt + block_start + j - 1;
                const float * logits = llama_get_logits_ith(ctx, logit_pos);
                if (!logits) continue;

                llama_token best = 0;
                float best_val = logits[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (logits[v] > best_val) { best_val = logits[v]; best = v; }
                }

                if (params.topk_accept > 1) {
                    topk_from_logits(logits, n_vocab, params.topk_accept, topk_buf);
                    if (!token_in_topk(topk_buf, candidate[block_start + j])) {
                        candidate[block_start + j] = best;
                        changed++;
                    }
                } else if (best != candidate[block_start + j]) {
                    candidate[block_start + j] = best;
                    changed++;
                }
            }

            llama_batch_free(batch);

            if (changed == 0) break;
        }
    }

    auto t_block_end = std::chrono::high_resolution_clock::now();
    double block_ms = std::chrono::duration<double, std::milli>(t_block_end - t_block_start).count();

    float final_agree = has_gt && n_compare > 0
        ? (float)count_block_matches() / n_compare * 100.0f : 0.0f;

    return {block_start, block_len, pass, init_agree, final_agree, block_ms, total_prefill_toks};
}

// ─── wavefront refinement ───────────────────────────────────────────────────

wir_result wir_refine_wavefront(
    llama_model  * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token>       & candidate,
    const std::vector<llama_token> & ground_truth,
    const wir_params & params)
{
    const int n_candidate = (int)candidate.size();
    const int block_size = params.block_size;

    auto t_start = std::chrono::high_resolution_clock::now();

    wir_result result = {};
    result.total_tokens = n_candidate;

    // Save original candidate for change counting
    std::vector<llama_token> original = candidate;

    bool left_ctx_ready = false;

    for (int bs = 0; bs < n_candidate; bs += block_size) {
        int blen = std::min(block_size, n_candidate - bs);

        wir_block_result br = wir_refine_block(
            model, ctx, prompt_tokens, candidate, ground_truth,
            bs, blen, params);

        // Update left_ctx_ready for next block (KV from this block is now cached)
        br.block_start = bs; // ensure correct in result
        result.blocks.push_back(br);
        result.total_passes += br.passes_used;
        result.total_blocks++;
        result.total_prefill_tokens += br.total_prefill_tokens;

        // Notify via callback if set
        if (params.token_callback) {
            if (!params.token_callback(bs, candidate.data() + bs, blen)) {
                break; // caller requested abort
            }
        }

        left_ctx_ready = true;
    }

    // Count total changes
    for (int i = 0; i < n_candidate; i++) {
        if (candidate[i] != original[i]) result.tokens_changed++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

// ─── autoregressive draft generation ────────────────────────────────────────

static std::vector<llama_token> generate_draft(
    llama_model   * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    int n_tokens,
    float temp)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_memory_clear(llama_get_memory(ctx), true);

    const int n_prompt = (int)prompt_tokens.size();

    // Prefill prompt
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = n_prompt;
    for (int i = 0; i < n_prompt; i++) {
        batch.token[i]     = prompt_tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_prompt - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "wir: draft prefill failed\n");
        llama_batch_free(batch);
        return {};
    }
    llama_batch_free(batch);

    // Generate tokens autoregressively
    std::vector<llama_token> output;
    output.reserve(n_tokens);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_tokens; i++) {
        const float * logits = llama_get_logits(ctx);
        if (!logits) break;

        llama_token next;
        if (temp <= 0.0f) {
            // Greedy
            next = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; next = v; }
            }
        } else {
            // Temperature sampling — simple top-1 after temperature division
            // For a proper draft, greedy or low-temp is fine
            std::vector<float> scaled(n_vocab);
            for (int v = 0; v < n_vocab; v++) scaled[v] = logits[v] / temp;
            float max_val = *std::max_element(scaled.begin(), scaled.end());
            double sum = 0.0;
            for (int v = 0; v < n_vocab; v++) {
                scaled[v] = expf(scaled[v] - max_val);
                sum += scaled[v];
            }
            // Sample from distribution
            double r = ((double)rand() / RAND_MAX) * sum;
            double cumul = 0.0;
            next = 0;
            for (int v = 0; v < n_vocab; v++) {
                cumul += scaled[v];
                if (cumul >= r) { next = v; break; }
            }
        }

        if (llama_vocab_is_eog(vocab, next) && (int)output.size() >= 2) break;

        output.push_back(next);

        // Decode next token
        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens = 1;
        step.token[0]     = next;
        step.pos[0]       = n_prompt + i;
        step.n_seq_id[0]  = 1;
        step.seq_id[0][0] = 0;
        step.logits[0]    = 1;

        if (llama_decode(ctx, step) != 0) {
            llama_batch_free(step);
            break;
        }
        llama_batch_free(step);
    }

    return output;
}

// ─── WIR decode with draft model ────────────────────────────────────────────

std::vector<llama_token> wir_decode(
    llama_model   * draft_model,
    llama_context * draft_ctx,
    llama_model   * refine_model,
    llama_context * refine_ctx,
    const std::vector<llama_token> & prompt_tokens,
    int n_tokens,
    const wir_params & params)
{
    // Step 1: Generate draft with small model (greedy)
    auto draft = generate_draft(draft_model, draft_ctx, prompt_tokens, n_tokens, 0.0f);

    if (draft.empty()) {
        fprintf(stderr, "wir_decode: draft generation produced no tokens\n");
        return {};
    }

    fprintf(stderr, "[wir] draft: %d tokens generated\n", (int)draft.size());

    // Step 2: Refine with big model using wavefront
    std::vector<llama_token> empty_gt; // no ground truth in production
    wir_result result = wir_refine_wavefront(
        refine_model, refine_ctx, prompt_tokens, draft, empty_gt, params);

    fprintf(stderr, "[wir] refinement: %d blocks, %d total passes, %d tokens changed, %.1f ms\n",
            result.total_blocks, result.total_passes, result.tokens_changed, result.total_ms);

    return draft; // modified in-place by refinement
}

// ─── Single-model WIR decode ────────────────────────────────────────────────

std::vector<llama_token> wir_decode_single_model(
    llama_model   * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    int n_tokens,
    float draft_temp,
    const wir_params & params)
{
    // Step 1: Generate draft with temperature (same model)
    auto draft = generate_draft(model, ctx, prompt_tokens, n_tokens, draft_temp);

    if (draft.empty()) {
        fprintf(stderr, "wir_decode_single_model: draft generation produced no tokens\n");
        return {};
    }

    fprintf(stderr, "[wir-single] draft: %d tokens at temp=%.2f\n", (int)draft.size(), draft_temp);

    // Step 2: Refine with same model (greedy) using wavefront
    std::vector<llama_token> empty_gt;
    wir_result result = wir_refine_wavefront(
        model, ctx, prompt_tokens, draft, empty_gt, params);

    fprintf(stderr, "[wir-single] refinement: %d blocks, %d total passes, %d tokens changed, %.1f ms\n",
            result.total_blocks, result.total_passes, result.tokens_changed, result.total_ms);

    return draft;
}

// ─── Draft generation from existing KV state (NO clear) ─────────────────────

static std::vector<llama_token> generate_draft_from_kv(
    llama_model   * model,
    llama_context * ctx,
    int prompt_token_count,   // tokens already in KV
    llama_token seed_token,   // last token from KV
    int n_tokens,
    float temp,
    int * out_gen_pos_start = nullptr, // output: position where output[0] sits in KV
    const float * forced_initial_logits = nullptr, // if set, use these instead of llama_get_logits
    int forced_logits_n_vocab = 0)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Do NOT clear KV — the prompt state is already imported.
    // The KV has positions [0, prompt_token_count). Logits from the last
    // prefill position (prompt_token_count - 1) predict the first generated token.
    //
    // Check if logits are already available (from a preceding prefill or decode).
    // If so, use them directly. Otherwise, decode seed_token at prompt_token_count
    // to prime the logits (Strategy B, used by decode receiver after KV import).

    std::vector<llama_token> output;
    output.reserve(n_tokens);

    // Try to use existing logits first (prefer forced/cached over context logits)
    bool have_initial_logits = false;
    if (forced_initial_logits && forced_logits_n_vocab > 0) {
        have_initial_logits = true;
    } else {
        const float * existing_logits = llama_get_logits(ctx);
        if (existing_logits) {
            float sum = 0.0f;
            for (int v = 0; v < std::min(100, n_vocab); v++) sum += fabsf(existing_logits[v]);
            have_initial_logits = (sum > 0.0f);
        }
    }

    llama_token current = seed_token;
    int gen_pos_start = prompt_token_count; // position where first generated token goes

    if (!have_initial_logits) {
        // No logits available — decode seed_token at prompt_token_count to get them.
        // This adds seed_token to KV at position prompt_token_count.
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0]     = seed_token;
        batch.pos[0]       = prompt_token_count;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "wir: draft-from-kv seed decode failed\n");
            llama_batch_free(batch);
            return {};
        }
        llama_batch_free(batch);
        gen_pos_start = prompt_token_count + 1; // first generated token goes after seed
    }

    for (int i = 0; i < n_tokens; i++) {
        const float * logits;
        if (i == 0 && have_initial_logits && forced_initial_logits) {
            logits = forced_initial_logits;
        } else {
            logits = llama_get_logits(ctx);
        }
        if (!logits) break;

        llama_token next;
        if (temp <= 0.0f) {
            // Greedy
            next = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; next = v; }
            }
        } else {
            // Temperature sampling
            std::vector<float> scaled(n_vocab);
            for (int v = 0; v < n_vocab; v++) scaled[v] = logits[v] / temp;
            float max_val = *std::max_element(scaled.begin(), scaled.end());
            double sum = 0.0;
            for (int v = 0; v < n_vocab; v++) {
                scaled[v] = expf(scaled[v] - max_val);
                sum += scaled[v];
            }
            double r = ((double)rand() / RAND_MAX) * sum;
            double cumul = 0.0;
            next = 0;
            for (int v = 0; v < n_vocab; v++) {
                cumul += scaled[v];
                if (cumul >= r) { next = v; break; }
            }
        }

        // Skip EOG check for first few tokens: thinking models emit <think>
        // which is marked as EOG but is actually a preamble, not a stop signal.
        // After 2 tokens, honour EOG normally.
        if (llama_vocab_is_eog(vocab, next) && (int)output.size() >= 2) break;

        output.push_back(next);
        current = next;

        // Decode the new token to get logits for the next iteration
        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens = 1;
        step.token[0]     = next;
        step.pos[0]       = gen_pos_start + (int)output.size() - 1;
        step.n_seq_id[0]  = 1;
        step.seq_id[0][0] = 0;
        step.logits[0]    = 1;

        if (llama_decode(ctx, step) != 0) {
            llama_batch_free(step);
            break;
        }
        llama_batch_free(step);
    }

    if (out_gen_pos_start) *out_gen_pos_start = gen_pos_start;
    return output;
}

// ─── KV-preserving block refinement ─────────────────────────────────────────
//
// After KV import, the prompt is already in the cache at positions [0, prompt_n).
// The draft tokens are at positions [prompt_n, prompt_n + draft_n).
// To refine a block [block_start, block_start + block_len) within the draft:
//   1. Remove KV for positions >= prompt_n + block_start (the block and everything after)
//   2. Re-decode the block tokens at their positions
//   3. Compare logits with candidate, update disagreements

static wir_block_result wir_refine_block_from_kv(
    llama_model  * model,
    llama_context * ctx,
    int prompt_n,                              // positions [0, prompt_n) are imported prompt KV
    llama_token left_seed_token,               // last token of prompt (for block_start==0)
    int gen_pos_start,                         // KV position of candidate[0]
    std::vector<llama_token> & candidate,      // draft tokens, modified in-place
    int block_start,                           // offset within candidate
    int block_len,
    const wir_params & params,
    const std::vector<float> * cached_initial_logits = nullptr) // logits predicting candidate[0]
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto t_block_start = std::chrono::high_resolution_clock::now();
    double total_prefill_toks = 0.0;
    std::vector<scored_token> topk_buf;

    int pass;
    for (pass = 1; pass <= params.max_passes; pass++) {
        // Remove KV for positions that need to be re-decoded.
        //
        // candidate[j] sits at KV position gen_pos_start + j.
        //
        // For block_start > 0: the left-context token is candidate[block_start-1]
        //   at position gen_pos_start + block_start - 1. We remove from that position
        //   onwards and re-decode it + block tokens.
        //
        // For block_start == 0 with cached_initial_logits: we have the logits from
        //   the prefill that predict candidate[0]. Don't touch prompt KV at all.
        //   Just remove from gen_pos_start onwards and re-decode only block tokens.
        //   Use cached logits for candidate[0] prediction.
        //
        // For block_start == 0 without cached logits: we must re-decode the seed
        //   token. The seed sits at gen_pos_start - 1 if gen_pos_start > prompt_n
        //   (Strategy B), or we need to fall back to removing prompt's last pos.

        llama_memory_t mem = llama_get_memory(ctx);

        bool use_cached_for_block0 = (block_start == 0 && cached_initial_logits &&
                                       !cached_initial_logits->empty());

        int rm_from;
        bool include_left_in_batch;
        int left_pos;
        llama_token left_tok = left_seed_token;

        if (block_start == 0 && use_cached_for_block0) {
            // Don't remove any prompt KV. Only remove generated tokens.
            rm_from = gen_pos_start;
            include_left_in_batch = false;
            left_pos = -1; // unused
        } else if (block_start == 0) {
            // No cached logits. Must re-decode left context.
            // If gen_pos_start > prompt_n, seed is at gen_pos_start - 1.
            // If gen_pos_start == prompt_n, fall back to removing prompt's last pos.
            if (gen_pos_start > prompt_n) {
                left_pos = gen_pos_start - 1; // seed position
                rm_from = left_pos;
            } else {
                left_pos = prompt_n - 1; // last prompt position
                rm_from = left_pos;
            }
            include_left_in_batch = true;
        } else {
            // block_start > 0: left is candidate[block_start-1]
            left_pos = gen_pos_start + block_start - 1;
            left_tok = candidate[block_start - 1];
            rm_from = left_pos;
            include_left_in_batch = true;
        }

        bool ok = llama_memory_seq_rm(mem, 0, rm_from, -1);
        if (!ok) {
            fprintf(stderr, "wir_refine_block_from_kv: seq_rm(%d, -1) failed\n", rm_from);
            break;
        }

        // Build batch
        const int extra = include_left_in_batch ? 1 : 0;
        int batch_len = block_len + extra;

        llama_batch batch = llama_batch_init(batch_len, 0, 1);
        batch.n_tokens = batch_len;

        int bi = 0;
        if (include_left_in_batch) {
            batch.token[bi]     = left_tok;
            batch.pos[bi]       = left_pos;
            batch.n_seq_id[bi]  = 1;
            batch.seq_id[bi][0] = 0;
            batch.logits[bi]    = 1;
            bi++;
        }
        for (int j = 0; j < block_len; j++, bi++) {
            batch.token[bi]     = candidate[block_start + j];
            batch.pos[bi]       = gen_pos_start + block_start + j;
            batch.n_seq_id[bi]  = 1;
            batch.seq_id[bi][0] = 0;
            batch.logits[bi]    = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "wir_refine_block_from_kv: decode failed\n");
            llama_batch_free(batch);
            break;
        }
        total_prefill_toks += batch_len;

        // Process logits and update candidate
        int changed = 0;

        auto update_from_logits = [&](const float * logits, int cand_idx) {
            if (!logits) return;
            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            if (params.topk_accept > 1) {
                topk_from_logits(logits, n_vocab, params.topk_accept, topk_buf);
                if (!token_in_topk(topk_buf, candidate[cand_idx])) {
                    candidate[cand_idx] = best;
                    changed++;
                }
            } else if (best != candidate[cand_idx]) {
                candidate[cand_idx] = best;
                changed++;
            }
        };

        auto update_token = [&](int logit_batch_idx, int cand_idx) {
            update_from_logits(llama_get_logits_ith(ctx, logit_batch_idx), cand_idx);
        };

        auto update_token_from_logits = [&](const float * logits, int cand_idx) {
            update_from_logits(logits, cand_idx);
        };

        if (use_cached_for_block0) {
            // Use cached logits for candidate[0], batch logits for rest
            // batch[0] = candidate[0], batch[1] = candidate[1], etc. (no left token in batch)
            // cached logits predict candidate[0]
            update_token_from_logits(cached_initial_logits->data(), block_start);
            // batch[j] logits (at candidate[j] position) predict candidate[j+1]
            for (int j = 0; j < block_len - 1; j++) {
                update_token(j, block_start + j + 1);
            }
        } else if (include_left_in_batch) {
            // batch[0] = left context token, its logits predict candidate[block_start]
            update_token(0, block_start);
            // batch[extra + j - 1] logits predict candidate[block_start + j] for j >= 1
            for (int j = 1; j < block_len; j++) {
                update_token(extra + j - 1, block_start + j);
            }
        } else {
            // No left context at all (shouldn't happen)
            for (int j = 0; j < block_len - 1; j++) {
                update_token(j, block_start + j + 1);
            }
        }

        llama_batch_free(batch);

        if (changed == 0) break;
    }

    auto t_block_end = std::chrono::high_resolution_clock::now();
    double block_ms = std::chrono::duration<double, std::milli>(t_block_end - t_block_start).count();

    return {block_start, block_len, pass, 0.0f, 0.0f, block_ms, total_prefill_toks};
}

// ─── Post-KV-import WIR decode ──────────────────────────────────────────────

std::vector<llama_token> wir_decode_from_kv(
    llama_model   * model,
    llama_context * ctx,
    int prompt_token_count,
    llama_token last_token,
    int n_tokens,
    float draft_temp,
    const wir_params & params,
    const float * prefill_logits,
    int prefill_logits_n_vocab)
{
    // Capture initial logits before draft generation (for block 0 refinement).
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> initial_logits;

    // Prefer caller-provided logits (they're from before any AR/draft modified the context)
    if (prefill_logits && prefill_logits_n_vocab > 0) {
        initial_logits.assign(prefill_logits, prefill_logits + prefill_logits_n_vocab);
    } else {
        const float * lg = llama_get_logits(ctx);
        if (lg) {
            float sum = 0.0f;
            for (int v = 0; v < std::min(100, n_vocab); v++) sum += fabsf(lg[v]);
            if (sum > 0.0f) {
                initial_logits.assign(lg, lg + n_vocab);
            }
        }
    }

    // Step 1: Generate draft from current KV state (preserves imported prompt KV)
    int gen_pos_start = prompt_token_count;
    auto draft = generate_draft_from_kv(model, ctx, prompt_token_count, last_token,
                                         n_tokens, draft_temp, &gen_pos_start,
                                         initial_logits.empty() ? nullptr : initial_logits.data(),
                                         initial_logits.empty() ? 0 : (int)initial_logits.size());

    if (draft.empty()) {
        fprintf(stderr, "wir_decode_from_kv: draft generation produced no tokens\n");
        return {};
    }

    fprintf(stderr, "[wir-kv] draft: %d tokens at temp=%.2f, gen_pos_start=%d\n",
            (int)draft.size(), draft_temp, gen_pos_start);

    // Step 2: Refine using KV-preserving wavefront block refinement.
    // candidate[j] sits at KV position gen_pos_start + j.

    const int n_draft = (int)draft.size();
    const int block_size = params.block_size;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> original = draft;
    wir_result result = {};
    result.total_tokens = n_draft;

    for (int bs = 0; bs < n_draft; bs += block_size) {
        int blen = std::min(block_size, n_draft - bs);

        // Pass cached initial logits for block 0 so it doesn't need to re-decode
        // the prompt's last position (which would corrupt the prompt KV).
        const std::vector<float> * block_cached_logits =
            (bs == 0 && !initial_logits.empty()) ? &initial_logits : nullptr;

        wir_block_result br = wir_refine_block_from_kv(
            model, ctx, prompt_token_count, last_token, gen_pos_start,
            draft, bs, blen, params, block_cached_logits);

        result.blocks.push_back(br);
        result.total_passes += br.passes_used;
        result.total_blocks++;
        result.total_prefill_tokens += br.total_prefill_tokens;

        // Notify via callback
        if (params.token_callback) {
            if (!params.token_callback(bs, draft.data() + bs, blen)) {
                break;
            }
        }
    }

    // Count changes
    for (int i = 0; i < n_draft; i++) {
        if (draft[i] != original[i]) result.tokens_changed++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    fprintf(stderr, "[wir-kv] refinement: %d blocks, %d total passes, %d/%d tokens changed, %.1f ms\n",
            result.total_blocks, result.total_passes, result.tokens_changed, n_draft, result.total_ms);

    return draft;
}

// ─── helpers for cross-tokenizer bridging ────────────────────────────────────

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab,
                                               const std::string & text, bool add_bos) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    if (n <= 0) return {};
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
    return tokens;
}

static std::string detokenize(const llama_vocab * vocab,
                               const std::vector<llama_token> & tokens) {
    std::string out;
    char buf[256];
    for (auto id : tokens) {
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
    }
    return out;
}

// ─── Cross-tokenizer WIR decode ─────────────────────────────────────────────

std::vector<llama_token> wir_decode_cross_tokenizer(
    llama_model   * draft_model,
    llama_context * draft_ctx,
    llama_model   * refine_model,
    llama_context * refine_ctx,
    const std::string & prompt_text,
    int prompt_token_count,
    llama_token last_token,
    int n_tokens,
    float draft_temp,
    const wir_params & params)
{
    const llama_vocab * draft_vocab  = llama_model_get_vocab(draft_model);
    const llama_vocab * refine_vocab = llama_model_get_vocab(refine_model);

    // Step 1: Tokenize prompt with draft model's tokenizer and prefill
    auto draft_prompt = tokenize_text(draft_vocab, prompt_text, true);
    if (draft_prompt.empty()) {
        fprintf(stderr, "wir_cross: draft prompt tokenization failed\n");
        return {};
    }

    fprintf(stderr, "[wir-cross] draft prompt: %d tokens (refine prompt: %d tokens)\n",
            (int)draft_prompt.size(), prompt_token_count);

    // Step 2: Generate draft tokens with draft model (clear + prefill + AR)
    auto draft_tokens = generate_draft(draft_model, draft_ctx, draft_prompt, n_tokens, draft_temp);
    if (draft_tokens.empty()) {
        fprintf(stderr, "wir_cross: draft generation produced no tokens\n");
        return {};
    }

    fprintf(stderr, "[wir-cross] draft: %d tokens generated with draft model\n", (int)draft_tokens.size());

    // Step 3: Detokenize draft output to text using draft model's vocab
    std::string draft_text = detokenize(draft_vocab, draft_tokens);

    fprintf(stderr, "[wir-cross] draft text (%d chars): %.80s%s\n",
            (int)draft_text.size(), draft_text.c_str(),
            draft_text.size() > 80 ? "..." : "");

    // Step 4: Re-tokenize with refine model's tokenizer (no BOS — continuation)
    auto candidate = tokenize_text(refine_vocab, draft_text, false);
    if (candidate.empty()) {
        fprintf(stderr, "wir_cross: re-tokenization produced no tokens\n");
        return {};
    }

    fprintf(stderr, "[wir-cross] re-tokenized: %d draft tokens -> %d refine tokens (%.1f%% token count change)\n",
            (int)draft_tokens.size(), (int)candidate.size(),
            100.0f * ((float)candidate.size() - (float)draft_tokens.size()) / (float)draft_tokens.size());

    // Step 5: Refine the re-tokenized candidate using the big model.
    // The refine model's KV cache has the prompt at positions [0, prompt_token_count).
    // We need to:
    //   a) Remove any KV from the draft generation (the refine ctx shouldn't have
    //      any — it only has the imported prompt KV)
    //   b) Decode the candidate tokens at positions [prompt_token_count, prompt_token_count + n_candidate)
    //   c) Run wavefront block refinement

    const int n_candidate = (int)candidate.size();
    const int block_size = params.block_size;

    // First, decode all candidate tokens into the refine model's KV cache
    // so the refinement has a starting point.
    {
        // Remove any state beyond prompt (in case the refine ctx was used before)
        llama_memory_t mem = llama_get_memory(refine_ctx);
        llama_memory_seq_rm(mem, 0, prompt_token_count, -1);

        // Decode candidate in batches
        const int batch_max = 512;
        for (int i = 0; i < n_candidate; i += batch_max) {
            int blen = std::min(batch_max, n_candidate - i);
            llama_batch batch = llama_batch_init(blen, 0, 1);
            batch.n_tokens = blen;
            for (int j = 0; j < blen; j++) {
                batch.token[j]     = candidate[i + j];
                batch.pos[j]       = prompt_token_count + i + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1; // need logits for refinement
            }
            if (llama_decode(refine_ctx, batch) != 0) {
                fprintf(stderr, "wir_cross: candidate prefill failed at offset %d\n", i);
                llama_batch_free(batch);
                return {};
            }
            llama_batch_free(batch);
        }
    }

    fprintf(stderr, "[wir-cross] candidate prefilled into refine model KV\n");

    // Step 6: Wavefront block refinement on the candidate
    // candidate[j] was decoded at position prompt_token_count + j
    const int cross_gen_pos = prompt_token_count;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> original = candidate;
    wir_result result = {};
    result.total_tokens = n_candidate;

    for (int bs = 0; bs < n_candidate; bs += block_size) {
        int blen = std::min(block_size, n_candidate - bs);

        wir_block_result br = wir_refine_block_from_kv(
            refine_model, refine_ctx, prompt_token_count, last_token,
            cross_gen_pos, candidate, bs, blen, params);

        result.blocks.push_back(br);
        result.total_passes += br.passes_used;
        result.total_blocks++;
        result.total_prefill_tokens += br.total_prefill_tokens;

        if (params.token_callback) {
            if (!params.token_callback(bs, candidate.data() + bs, blen)) {
                break;
            }
        }
    }

    for (int i = 0; i < n_candidate; i++) {
        if (candidate[i] != original[i]) result.tokens_changed++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    fprintf(stderr, "[wir-cross] refinement: %d blocks, %d total passes, %d/%d tokens changed, %.1f ms\n",
            result.total_blocks, result.total_passes, result.tokens_changed, n_candidate, result.total_ms);

    return candidate;
}

// ─── Continuous draft→refine loop ───────────────────────────────────────────

static std::vector<llama_token> tokenize_text_internal(const llama_vocab * vocab,
                                                        const std::string & text, bool add_bos) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    if (n <= 0) return {};
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
    return tokens;
}

static std::string detokenize_internal(const llama_vocab * vocab,
                                        const llama_token * tokens, int count) {
    std::string out;
    char buf[256];
    for (int i = 0; i < count; i++) {
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
    }
    return out;
}

wir_loop_result wir_continuous_loop(
    llama_model   * draft_model,
    llama_context * draft_ctx,
    llama_model   * refine_model,
    llama_context * refine_ctx,
    const std::string & prompt_text,
    const std::vector<llama_token> & refine_prompt_tokens,
    const wir_loop_config & config,
    wir_emit_cb emit_callback)
{
    const llama_vocab * draft_vocab  = llama_model_get_vocab(draft_model);
    const llama_vocab * refine_vocab = llama_model_get_vocab(refine_model);
    const int draft_n_vocab  = llama_vocab_n_tokens(draft_vocab);

    wir_loop_result loop_result = {};
    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill draft model with prompt
    auto draft_prompt = tokenize_text_internal(draft_vocab, prompt_text, true);
    if (draft_prompt.empty()) {
        fprintf(stderr, "[wir-loop] draft prompt tokenization failed\n");
        return loop_result;
    }

    {
        llama_batch batch = llama_batch_init((int)draft_prompt.size(), 0, 1);
        batch.n_tokens = (int)draft_prompt.size();
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.token[i] = draft_prompt[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
        }
        llama_decode(draft_ctx, batch);
        llama_batch_free(batch);
    }

    // Prefill refine model with prompt.
    // For streaming models (wf_active), skip this — wir_refine_sequence
    // will do memory_clear + full re-decode for each chunk, which handles
    // the streaming path correctly. Prefilling here would crash because
    // the streaming buffers aren't allocated until the first decode call.
    if (!refine_model->hparams.wf_active) {
        int n_rp = (int)refine_prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_rp, 0, 1);
        batch.n_tokens = n_rp;
        for (int i = 0; i < n_rp; i++) {
            batch.token[i] = refine_prompt_tokens[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_rp - 1) ? 1 : 0;
        }
        llama_decode(refine_ctx, batch);
        llama_batch_free(batch);
    }

    int draft_pos = (int)draft_prompt.size();   // next position for draft model
    int refine_pos = (int)refine_prompt_tokens.size(); // next position for refine model

    // Track all emitted refine tokens as rolling context for subsequent chunks
    std::vector<llama_token> all_refine_tokens = refine_prompt_tokens;

    // Main loop: draft chunk → refine → emit → repeat
    for (;;) {
        if (config.max_total > 0 && loop_result.total_tokens >= config.max_total) break;

        int chunk_size = config.chunk_size;
        if (config.max_total > 0) {
            chunk_size = std::min(chunk_size, config.max_total - loop_result.total_tokens);
        }

        // Step 1: Draft model generates chunk_size tokens (greedy, ignore EOG)
        std::vector<llama_token> draft_chunk;
        draft_chunk.reserve(chunk_size);
        for (int i = 0; i < chunk_size; i++) {
            const float * logits = llama_get_logits(draft_ctx);
            if (!logits) break;
            llama_token best = 0; float bv = logits[0];
            for (int v = 1; v < draft_n_vocab; v++) { if (logits[v] > bv) { bv = logits[v]; best = v; } }
            // Ignore draft EOG — big model decides when to stop
            draft_chunk.push_back(best);
            llama_batch step = llama_batch_init(1, 0, 1);
            step.n_tokens = 1; step.token[0] = best; step.pos[0] = draft_pos + i;
            step.n_seq_id[0] = 1; step.seq_id[0][0] = 0; step.logits[0] = 1;
            llama_decode(draft_ctx, step);
            llama_batch_free(step);
        }
        draft_pos += (int)draft_chunk.size();

        if (draft_chunk.empty()) break;

        // Step 2: Token bridge — use direct token IDs if same tokenizer,
        // otherwise detokenize→retokenize through text space
        std::vector<llama_token> refine_chunk;
        // Check if tokenizers are compatible (same vocab size + same BOS/EOS tokens)
        bool same_tokenizer = (llama_vocab_n_tokens(draft_vocab) == llama_vocab_n_tokens(refine_vocab) &&
                                llama_vocab_bos(draft_vocab) == llama_vocab_bos(refine_vocab) &&
                                llama_vocab_eos(draft_vocab) == llama_vocab_eos(refine_vocab));
        if (same_tokenizer) {
            // Same tokenizer: pass token IDs directly, no text bridge
            refine_chunk = draft_chunk;
        } else {
            // Cross-tokenizer: detokenize draft → retokenize for refine model
            std::string chunk_text = detokenize_internal(draft_vocab, draft_chunk.data(), (int)draft_chunk.size());
            refine_chunk = tokenize_text_internal(refine_vocab, chunk_text, false);
            if (refine_chunk.empty()) break;
        }

        fprintf(stderr, "[wir-loop] chunk %d: draft %d tokens -> refine %d tokens\n",
                loop_result.total_chunks + 1, (int)draft_chunk.size(), (int)refine_chunk.size());

        // For streaming models: free the draft context to make VRAM space
        // for the refine context. The draft context will be recreated after
        // refinement. The draft MODEL stays loaded (only context freed).
        const bool swap_ctx = refine_model->hparams.wf_active;
        if (swap_ctx) {
            llama_free(draft_ctx);
            draft_ctx = nullptr;
            fprintf(stderr, "[wir-loop] freed draft context for streaming refine\n");
        }

        // Step 3: Refine the chunk using the big model
        // For streaming models, we need a fresh refine context per chunk
        // (the previous one may not exist yet or has stale state).
        if (swap_ctx && refine_ctx == nullptr) {
            // First chunk or after swap: create refine context
            llama_context_params rcp = llama_context_default_params();
            rcp.n_ctx = (int)all_refine_tokens.size() + (int)refine_chunk.size() + 256;
            rcp.n_batch = rcp.n_ctx;
            rcp.n_ubatch = std::min(rcp.n_batch, (uint32_t)1536);
            rcp.no_perf = true;
            rcp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            rcp.prefill_streaming = true; // Enable NVMe streaming for 397B
            refine_ctx = llama_init_from_model(refine_model, rcp);
            if (!refine_ctx) {
                fprintf(stderr, "[wir-loop] failed to create refine context\n");
                break;
            }
            // Force scheduler re-reserve at actual batch size
            refine_ctx->set_sched_need_reserve(true);
            fprintf(stderr, "[wir-loop] created refine context (streaming=%d)\n",
                    (int)rcp.prefill_streaming);
        }

        // Rebuild context each chunk
        llama_memory_clear(llama_get_memory(refine_ctx), true);
        llama_synchronize(refine_ctx);
        refine_ctx->reset_streaming_state();

        // Build complete sequence: all previous tokens + this chunk
        // Empty ground truth — we don't have AR baseline in continuous mode
        std::vector<llama_token> empty_gt;

        auto refine_result = wir_refine_sequence(
            refine_model, refine_ctx, all_refine_tokens, refine_chunk, empty_gt, config.refine);

        loop_result.total_refine_passes += refine_result.passes_used;

        // Step 4: Check for EOG in refined chunk
        // Skip EOG check for the first N chunks (thinking models emit <think>...</think>
        // which contains EOG-like tokens that shouldn't stop generation)
        bool found_eog = false;
        int eog_pos = -1;
        if (loop_result.total_chunks >= config.min_chunks_before_eog) {
            for (int i = 0; i < (int)refine_chunk.size(); i++) {
                if (llama_vocab_is_eog(refine_vocab, refine_chunk[i])) {
                    found_eog = true;
                    eog_pos = i;
                    break;
                }
            }
        }

        int emit_count = found_eog ? eog_pos : (int)refine_chunk.size();

        // Step 5: Emit converged tokens
        if (emit_count > 0) {
            if (emit_callback && !emit_callback(refine_chunk.data(), emit_count)) {
                break; // caller asked to stop
            }
            // Append emitted tokens to our running context
            all_refine_tokens.insert(all_refine_tokens.end(),
                                      refine_chunk.begin(), refine_chunk.begin() + emit_count);
            loop_result.total_tokens += emit_count;
        }

        // For streaming models: free refine context and recreate draft context
        if (swap_ctx) {
            llama_free(refine_ctx);
            refine_ctx = nullptr;

            // Recreate draft context with full context (prompt + all emitted tokens)
            llama_context_params dcp = llama_context_default_params();
            dcp.n_ctx = draft_pos + 256;
            dcp.n_batch = dcp.n_ctx;
            dcp.no_perf = true;
            draft_ctx = llama_init_from_model(draft_model, dcp);
            if (!draft_ctx) {
                fprintf(stderr, "[wir-loop] failed to recreate draft context\n");
                break;
            }
            // Re-prefill draft with all tokens so far
            std::vector<llama_token> draft_full;
            draft_full.insert(draft_full.end(), draft_prompt.begin(), draft_prompt.end());
            // Add corrected tokens from all previous chunks
            if (same_tokenizer) {
                // Use refine tokens directly (same vocab)
                for (size_t t = refine_prompt_tokens.size(); t < all_refine_tokens.size(); t++) {
                    draft_full.push_back(all_refine_tokens[t]);
                }
            }
            if (!draft_full.empty()) {
                llama_batch pb = llama_batch_init((int)draft_full.size(), 0, 1);
                pb.n_tokens = (int)draft_full.size();
                for (int i = 0; i < pb.n_tokens; i++) {
                    pb.token[i] = draft_full[i]; pb.pos[i] = i;
                    pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0;
                    pb.logits[i] = (i == pb.n_tokens - 1) ? 1 : 0;
                }
                llama_decode(draft_ctx, pb);
                llama_batch_free(pb);
            }
            draft_pos = (int)draft_full.size();
            fprintf(stderr, "[wir-loop] recreated draft context with %d tokens\n", draft_pos);
        }

        // Step 6: Sync draft model context with refined output.
        // The draft model continued from its own greedy tokens, but the refine model
        // may have changed them. Clear the draft's KV for the chunk range and re-decode
        // with the corrected tokens so the next chunk continues from the right context.
        if (emit_count > 0 && !swap_ctx) {
            // Get corrected tokens for the draft model
            std::vector<llama_token> draft_corrected;
            if (same_tokenizer) {
                draft_corrected.assign(refine_chunk.begin(), refine_chunk.begin() + emit_count);
            } else {
                std::string refined_text = detokenize_internal(refine_vocab, refine_chunk.data(), emit_count);
                draft_corrected = tokenize_text_internal(draft_vocab, refined_text, false);
            }

            // Remove draft KV from chunk start onwards and re-decode corrected tokens
            int draft_chunk_start = draft_pos - (int)draft_chunk.size();
            llama_memory_t draft_mem = llama_get_memory(draft_ctx);
            llama_memory_seq_rm(draft_mem, 0, draft_chunk_start, -1);

            if (!draft_corrected.empty()) {
                const int batch_max = 512;
                for (int off = 0; off < (int)draft_corrected.size(); off += batch_max) {
                    int blen = std::min(batch_max, (int)draft_corrected.size() - off);
                    llama_batch db = llama_batch_init(blen, 0, 1);
                    db.n_tokens = blen;
                    for (int i = 0; i < blen; i++) {
                        db.token[i] = draft_corrected[off + i];
                        db.pos[i] = draft_chunk_start + off + i;
                        db.n_seq_id[i] = 1; db.seq_id[i][0] = 0;
                        db.logits[i] = (off + i == (int)draft_corrected.size() - 1) ? 1 : 0;
                    }
                    llama_decode(draft_ctx, db);
                    llama_batch_free(db);
                }
                draft_pos = draft_chunk_start + (int)draft_corrected.size();
            }
        }

        loop_result.total_chunks++;

        if (found_eog) {
            loop_result.stopped_by_eog = true;
            fprintf(stderr, "[wir-loop] refine model produced EOG at chunk %d pos %d\n",
                    loop_result.total_chunks, eog_pos);
            break;
        }

        refine_pos = (int)all_refine_tokens.size();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    loop_result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    fprintf(stderr, "[wir-loop] done: %d tokens, %d chunks, %d refine passes, %.0fms (%.1f tok/s)\n",
            loop_result.total_tokens, loop_result.total_chunks, loop_result.total_refine_passes,
            loop_result.total_ms,
            loop_result.total_tokens > 0 ? loop_result.total_tokens / (loop_result.total_ms / 1000.0) : 0);

    return loop_result;
}
