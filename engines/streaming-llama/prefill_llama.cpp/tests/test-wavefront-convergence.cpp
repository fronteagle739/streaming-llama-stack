// test-wavefront-convergence.cpp
//
// Proof-of-principle convergence test for Wavefront Iterative Refinement.
//
// Two modes:
//   Jacobi  (--mode jacobi)  — all tokens refined globally each pass
//   Wavefront (--mode wavefront) — block-by-block Gauss-Seidel refinement
//
// 1. Load a small model and generate a "rough" answer to a prompt
// 2. Load a bigger model and generate the AR ground truth answer (temp=0)
// 3. Iteratively refine the rough answer using the big model
// 4. Measure token agreement with ground truth per pass
//
// Usage:
//   ./test-wavefront-convergence \
//       --small-model /path/to/3b.gguf \
//       --big-model   /path/to/12b.gguf \
//       --prompt "Explain how a transistor works in 3 sentences." \
//       -n 128 --max-passes 20 --mode wavefront --block-size 16
//
//   Sweep mode (runs wavefront across multiple block sizes):
//       ./test-wavefront-convergence \
//           --small-model /path/to/3b.gguf --big-model /path/to/12b.gguf \
//           --prompt "..." -n 128 --mode sweep

#include "llama.h"

// internal headers for embedding matrix access
#include "../src/llama-model.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

// ─── helpers ────────────────────────────────────────────────────────────────

static std::string token_to_str(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) return "<?>";
    return std::string(buf, n);
}

static std::string tokens_to_str(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string out;
    for (auto id : tokens) out += token_to_str(vocab, id);
    return out;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_bos) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
    return tokens;
}

// ─── confidence from logits ─────────────────────────────────────────────────

// softmax entropy → confidence in [0,1].  1 = maximally confident.
static float confidence_from_logits(const float * logits, int n_vocab) {
    // numerical stability: subtract max
    float max_val = *std::max_element(logits, logits + n_vocab);

    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum_exp += exp((double)(logits[i] - max_val));
    }
    double log_sum = log(sum_exp);

    // H = -sum(p * log(p)) = log(sum_exp) - (1/sum_exp) * sum(x_i * exp(x_i))
    double weighted = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        double x = (double)(logits[i] - max_val);
        weighted += x * exp(x);
    }
    double entropy = log_sum - weighted / sum_exp;

    // normalize to [0,1]
    double max_entropy = log((double)n_vocab);
    if (max_entropy < 1e-9) return 1.0f;

    double normalized = entropy / max_entropy;  // 0 = peaked, 1 = uniform
    return (float)(1.0 - normalized);           // invert: 1 = confident
}

// ─── full-sequence confidence measurement ───────────────────────────────────
//
// Decode the full candidate via batch.token and measure self-consistency
// and mean confidence. Returns {self_consistency%, mean_confidence, gt_agreement%}.

struct confidence_result {
    float self_consistency_pct;
    float mean_confidence;
    float gt_agreement_pct;
};

static confidence_result measure_sequence_confidence(
    llama_model * model,
    llama_context * ctx,
    const std::vector<llama_token> & prompt_tokens,
    const std::vector<llama_token> & candidate,
    const std::vector<llama_token> & ground_truth)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_gen = (int)candidate.size();
    const int n_total = n_prompt + n_gen;

    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<llama_token> full_seq;
    full_seq.reserve(n_total);
    full_seq.insert(full_seq.end(), prompt_tokens.begin(), prompt_tokens.end());
    full_seq.insert(full_seq.end(), candidate.begin(), candidate.end());

    llama_batch batch = llama_batch_init(n_total, 0, 1);
    batch.n_tokens = n_total;
    for (int i = 0; i < n_total; i++) {
        batch.token[i]     = full_seq[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i >= n_prompt - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return {0.0f, 0.0f, 0.0f};
    }

    int n_self = 0, n_gt = 0;
    double sum_conf = 0.0;

    for (int j = 0; j < n_gen; j++) {
        int logit_pos = n_prompt - 1 + j;
        const float * logits = llama_get_logits_ith(ctx, logit_pos);
        if (!logits) continue;

        llama_token argmax = 0;
        float best_val = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
        }

        float conf = confidence_from_logits(logits, n_vocab);
        sum_conf += conf;

        if (argmax == candidate[j]) n_self++;
        if (j < (int)ground_truth.size() && candidate[j] == ground_truth[j]) n_gt++;
    }

    llama_batch_free(batch);

    int n_compare = std::min(n_gen, (int)ground_truth.size());
    return {
        (float)n_self / n_gen * 100.0f,
        (float)(sum_conf / n_gen),
        n_compare > 0 ? (float)n_gt / n_compare * 100.0f : 0.0f
    };
}

// ─── embedding matrix cache ─────────────────────────────────────────────────
//
// Dequantize the tok_embd matrix to float32 for computing expected embeddings.
// This is a one-time cost; for 27B with 151936×3584 it's ~2.2GB of float32.
// We use top-k to avoid needing the full matrix in hot paths.

struct embedding_cache {
    std::vector<float> data;   // [n_vocab × n_embd] float32
    int n_vocab = 0;
    int n_embd  = 0;

    void init(llama_model * model) {
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        n_embd  = llama_model_n_embd(model);

        const ggml_tensor * tok_embd = model->tok_embd;
        if (!tok_embd) {
            fprintf(stderr, "embedding_cache: tok_embd not found\n");
            return;
        }

        data.resize((size_t)n_vocab * n_embd);

        if (tok_embd->type == GGML_TYPE_F32) {
            // direct copy
            ggml_backend_tensor_get(tok_embd, data.data(), 0, data.size() * sizeof(float));
        } else {
            // dequantize: read raw quantized data, then convert to float
            size_t raw_size = ggml_nbytes(tok_embd);
            std::vector<uint8_t> raw(raw_size);
            ggml_backend_tensor_get(tok_embd, raw.data(), 0, raw_size);

            auto traits = ggml_get_type_traits(tok_embd->type);
            if (traits->to_float) {
                // to_float expects full rows; tok_embd is [n_embd, n_vocab] in ggml (row-major in n_embd)
                // each "row" in ggml is n_embd elements = one token's embedding
                traits->to_float(raw.data(), data.data(), (int64_t)n_vocab * n_embd);
            } else {
                fprintf(stderr, "embedding_cache: unsupported type %d\n", tok_embd->type);
            }
        }

        fprintf(stderr, "embedding_cache: loaded %d × %d (%.1f MB), tok_embd type=%d\n",
                n_vocab, n_embd, data.size() * sizeof(float) / (1024.0 * 1024.0),
                tok_embd->type);

        // verify: use ggml_get_rows to dequantize a few tokens and compare
        // this uses the exact same code path as the model's internal embedding lookup
        {
            size_t row_size = ggml_row_size(tok_embd->type, n_embd);
            std::vector<float> ref_embd(n_embd);
            std::vector<uint8_t> row_raw(row_size);
            auto row_traits = ggml_get_type_traits(tok_embd->type);

            int test_tokens[] = {0, 1, 100, 1000, n_vocab - 1};
            int n_test = 5;
            float max_diff = 0.0f;
            for (int t = 0; t < n_test; t++) {
                int tok = test_tokens[t];
                if (tok >= n_vocab) continue;
                // read one row from backend
                ggml_backend_tensor_get(tok_embd, row_raw.data(), (size_t)tok * row_size, row_size);
                row_traits->to_float(row_raw.data(), ref_embd.data(), n_embd);
                // compare with our cached version
                const float * cached = get(tok);
                for (int d = 0; d < n_embd; d++) {
                    float diff = fabsf(cached[d] - ref_embd[d]);
                    if (diff > max_diff) max_diff = diff;
                }
            }
            fprintf(stderr, "embedding_cache: verification max_diff=%.9e (0 = exact match)\n", max_diff);
        }
    }

    // get embedding for token id: returns pointer to n_embd floats
    const float * get(llama_token id) const {
        return data.data() + (size_t)id * n_embd;
    }
};

// ─── soft-embedding refinement pass ─────────────────────────────────────────
//
// Instead of replacing tokens with argmax, compute expected embeddings from
// the softmax distribution and feed them back via llama_batch.embd.
// Uses temperature annealing and damped updates for convergence.

struct soft_pass_result {
    int   n_tokens;
    float agreement_pct;    // argmax agreement with ground truth
    float mean_confidence;
    float temperature;
    double elapsed_ms;
};

static soft_pass_result run_soft_refinement_pass(
    llama_model * model,
    llama_context * ctx,
    const embedding_cache & emb_cache,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<float> & soft_embd,             // [n_gen × n_embd] soft embeddings, modified in-place
    const std::vector<llama_token> & ground_truth,
    int n_gen,
    float temperature,
    float alpha_base,                            // base damping coefficient
    int top_k)                                   // top-k tokens for expected embedding
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_embd  = emb_cache.n_embd;
    const int n_total = n_prompt + n_gen;

    llama_memory_clear(llama_get_memory(ctx), true);

    auto t0 = std::chrono::high_resolution_clock::now();

    // build batch: prompt tokens (discrete) + generated tokens (soft embeddings)
    // We need a hybrid batch: discrete tokens for prompt, continuous embeddings for generated.
    // llama_batch only supports one mode (token OR embd), so we use embd for everything:
    // convert prompt tokens to their embeddings too.
    llama_batch batch = llama_batch_init(n_total, n_embd, 1);
    batch.n_tokens = n_total;

    // fill prompt positions with discrete token embeddings
    for (int i = 0; i < n_prompt; i++) {
        const float * tok_emb = emb_cache.get(prompt_tokens[i]);
        memcpy(batch.embd + (size_t)i * n_embd, tok_emb, n_embd * sizeof(float));
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_prompt - 1) ? 1 : 0; // logits for last prompt pos (predicts gen[0])
    }

    // fill generated positions with soft embeddings
    for (int j = 0; j < n_gen; j++) {
        memcpy(batch.embd + (size_t)(n_prompt + j) * n_embd,
               soft_embd.data() + (size_t)j * n_embd,
               n_embd * sizeof(float));
        batch.pos[n_prompt + j]       = n_prompt + j;
        batch.n_seq_id[n_prompt + j]  = 1;
        batch.seq_id[n_prompt + j][0] = 0;
        batch.logits[n_prompt + j]    = 1; // need logits at all generated positions
    }

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "soft_refine: decode failed\n");
        llama_batch_free(batch);
        return {n_gen, 0.0f, 0.0f, temperature, 0.0};
    }

    // process logits: compute expected embeddings with temperature and damping
    int n_match = 0;
    double sum_conf = 0.0;

    for (int j = 0; j < n_gen; j++) {
        // logits at position (n_prompt - 1 + j) predict generated token j
        // logits at position (n_prompt - 1 + j) predict generated token j
        int logit_pos = n_prompt - 1 + j;
        const float * logits = llama_get_logits_ith(ctx, logit_pos);
        if (!logits) continue;

        // compute softmax with temperature
        float max_logit = *std::max_element(logits, logits + n_vocab);
        std::vector<float> probs(n_vocab);
        double sum_exp = 0.0;
        for (int v = 0; v < n_vocab; v++) {
            probs[v] = expf((logits[v] - max_logit) / temperature);
            sum_exp += probs[v];
        }
        for (int v = 0; v < n_vocab; v++) {
            probs[v] /= (float)sum_exp;
        }

        // find argmax for measurement
        llama_token argmax = 0;
        float max_prob = probs[0];
        for (int v = 1; v < n_vocab; v++) {
            if (probs[v] > max_prob) { max_prob = probs[v]; argmax = v; }
        }
        if (j < (int)ground_truth.size() && argmax == ground_truth[j]) n_match++;
        sum_conf += max_prob;

        // compute expected embedding using top-k tokens
        // find top-k indices
        std::vector<std::pair<float, int>> prob_idx(n_vocab);
        for (int v = 0; v < n_vocab; v++) {
            prob_idx[v] = {probs[v], v};
        }
        std::partial_sort(prob_idx.begin(), prob_idx.begin() + top_k, prob_idx.end(),
                         [](const auto & a, const auto & b) { return a.first > b.first; });

        // renormalize top-k
        float topk_sum = 0.0f;
        for (int k = 0; k < top_k; k++) topk_sum += prob_idx[k].first;

        // compute expected embedding: weighted sum of top-k token embeddings
        std::vector<float> expected_embd(n_embd, 0.0f);
        for (int k = 0; k < top_k; k++) {
            float w = prob_idx[k].first / topk_sum;
            int tok_id = prob_idx[k].second;
            const float * tok_emb = emb_cache.get(tok_id);
            for (int d = 0; d < n_embd; d++) {
                expected_embd[d] += w * tok_emb[d];
            }
        }

        // adaptive damping: high confidence → α closer to 1
        float confidence = max_prob; // use max prob as confidence
        float alpha = alpha_base + (1.0f - alpha_base) * confidence;

        // damped update
        float * cur_embd = soft_embd.data() + (size_t)j * n_embd;
        for (int d = 0; d < n_embd; d++) {
            cur_embd[d] = (1.0f - alpha) * cur_embd[d] + alpha * expected_embd[d];
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int n_compare = std::min(n_gen, (int)ground_truth.size());
    float agreement = n_compare > 0 ? (float)n_match / n_compare * 100.0f : 0.0f;

    llama_batch_free(batch);

    return {n_gen, agreement, (float)(sum_conf / n_gen), temperature, ms};
}

// ─── position-weighted block refinement (wavefront) ─────────────────────────
//
// Refine one block using discrete tokens with position-weighted acceptance.
// Left tokens in the block are conservative (need high confidence to change),
// right tokens always accept the argmax. This stabilizes left context within
// the block while letting right tokens explore freely.
// Uses batch.token throughout (no batch.embd — avoids hybrid SSM issues).

struct soft_block_result {
    int   block_start;
    int   block_len;
    int   passes_used;
    float initial_agreement;
    float final_agreement;
    double total_ms;
};

static soft_block_result run_soft_block_refinement(
    llama_model * model,
    llama_context * ctx,
    const embedding_cache & emb_cache,
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token> & candidate,           // full candidate, block tokens updated on exit
    const std::vector<llama_token> & ground_truth,
    int block_start,
    int block_len,
    int max_passes,
    float init_temperature,
    float gamma,
    float alpha_base,
    int top_k)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_embd  = emb_cache.n_embd;
    const int n_left  = n_prompt + block_start;  // prompt + converged left context

    auto t0 = std::chrono::high_resolution_clock::now();

    // measure initial block agreement
    int n_compare = std::min(block_len, (int)ground_truth.size() - block_start);
    auto count_matches = [&]() {
        int m = 0;
        for (int j = 0; j < n_compare; j++) {
            if (candidate[block_start + j] == ground_truth[block_start + j]) m++;
        }
        return m;
    };
    float init_agree = n_compare > 0 ? (float)count_matches() / n_compare * 100.0f : 0.0f;

    (void)init_temperature; (void)gamma; (void)top_k; // unused — kept for API compat

    int pass;
    for (pass = 1; pass <= max_passes; pass++) {
        llama_memory_clear(llama_get_memory(ctx), true);

        // build full sequence: prompt + left context + block tokens (all discrete via batch.token)
        std::vector<llama_token> full_seq;
        full_seq.reserve(n_left + block_len);
        full_seq.insert(full_seq.end(), prompt_tokens.begin(), prompt_tokens.end());
        full_seq.insert(full_seq.end(), candidate.begin(), candidate.begin() + block_start + block_len);

        int n_total = n_left + block_len;
        llama_batch batch = llama_batch_init(n_total, 0, 1);
        batch.n_tokens = n_total;

        for (int i = 0; i < n_total; i++) {
            batch.token[i]     = full_seq[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i >= n_left - 1) ? 1 : 0;
        }

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "soft_block: decode failed\n");
            llama_batch_free(batch);
            break;
        }

        // collect logits and update tokens
        // block[j] predicted by logits at batch position n_left - 1 + j
        //
        // Pass-ramping acceptance threshold (simulated annealing):
        // Early passes: threshold ~0 (accept all argmax, free exploration)
        // Later passes: threshold ramps up to alpha_base (only accept confident changes)
        // This lets early passes do bulk correction, late passes stabilize.
        float pass_frac = (max_passes > 1) ? (float)(pass - 1) / (max_passes - 1) : 1.0f;
        float accept_thresh = alpha_base * pass_frac;

        int changed = 0;
        int blocked = 0;  // tokens that wanted to change but were blocked by threshold

        for (int j = 0; j < block_len; j++) {
            int logit_pos = n_left - 1 + j;
            const float * logits = llama_get_logits_ith(ctx, logit_pos);
            if (!logits) continue;

            // find argmax and its confidence
            llama_token argmax = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
            }

            if (argmax == candidate[block_start + j]) continue; // already correct

            float conf = confidence_from_logits(logits, n_vocab);

            if (conf >= accept_thresh) {
                candidate[block_start + j] = argmax;
                changed++;
            } else {
                blocked++;
            }
        }

        llama_batch_free(batch);

        // Converged when nothing wants to change (not just blocked by threshold)
        if (changed == 0 && blocked == 0) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float final_agree = n_compare > 0 ? (float)count_matches() / n_compare * 100.0f : 0.0f;

    return {block_start, block_len, pass, init_agree, final_agree, ms};
}

// ─── AR generation (greedy, temp=0) ─────────────────────────────────────────

struct generation_result {
    std::vector<llama_token> tokens;
    double elapsed_ms;
};

struct sampling_config {
    float rep_penalty = 1.0f;  // 1.0 = disabled
    float min_p       = 0.0f;  // 0.0 = disabled
    int   top_k       = 0;     // 0 = disabled
    float top_p       = 1.0f;  // 1.0 = disabled
    float temp        = 0.0f;  // 0.0 = greedy
};

// generate_ar: if ext_ctx is provided, reuse it (clear KV + decode); otherwise create/destroy one.
static generation_result generate_ar(
    llama_model * model,
    const std::vector<llama_token> & prompt_tokens,
    int n_generate,
    int /*n_gpu_layers*/,
    sampling_config sconf = {},
    llama_context * ext_ctx = nullptr)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context * ctx = ext_ctx;
    bool owns_ctx = false;
    if (!ctx) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx    = prompt_tokens.size() + n_generate + 16;
        ctx_params.n_batch  = prompt_tokens.size();
        ctx_params.no_perf  = true;
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) { fprintf(stderr, "generate_ar: failed to create context\n"); exit(1); }
        owns_ctx = true;
    } else {
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (sconf.rep_penalty != 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, sconf.rep_penalty, 0.0f, 0.0f));
    }
    if (sconf.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(sconf.top_k));
    }
    if (sconf.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(sconf.top_p, 1));
    }
    if (sconf.min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(sconf.min_p, 1));
    }
    if (sconf.temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(sconf.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // prefill the prompt
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt_tokens.data()), prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "generate_ar: prefill failed\n"); exit(1);
    }

    std::vector<llama_token> out;
    out.reserve(n_generate);

    for (int i = 0; i < n_generate; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        out.push_back(id);
        batch = llama_batch_get_one(&out.back(), 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "generate_ar: decode step %d failed\n", i); exit(1);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    llama_sampler_free(smpl);
    if (owns_ctx) llama_free(ctx);
    return {out, ms};
}

// ─── single refinement pass ─────────────────────────────────────────────────
//
// Given prompt + candidate tokens, run a full prefill, get logits at every
// generated position, and replace low-confidence tokens with argmax(logits).

struct pass_result {
    int   tokens_changed;
    int   tokens_frozen;       // above threshold
    float mean_confidence;
    float agreement_pct;       // vs ground truth
    double elapsed_ms;
};

static pass_result run_refinement_pass(
    llama_model * model,
    llama_context * shared_ctx,                     // if non-null, reuse this context
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token> & candidate,          // modified in-place
    std::vector<bool> & frozen,                     // frozen mask, modified in-place
    const std::vector<llama_token> & ground_truth,  // for measurement
    float confidence_threshold)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_gen    = (int)candidate.size();
    const int n_total  = n_prompt + n_gen;

    // build full sequence: prompt + candidate
    std::vector<llama_token> full_seq;
    full_seq.reserve(n_total);
    full_seq.insert(full_seq.end(), prompt_tokens.begin(), prompt_tokens.end());
    full_seq.insert(full_seq.end(), candidate.begin(), candidate.end());

    llama_context * ctx = shared_ctx;
    bool owns_ctx = false;

    if (!ctx) {
        // create context per-pass (original behavior)
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx    = n_total + 16;
        ctx_params.n_batch  = n_total;
        ctx_params.no_perf  = true;

        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) { fprintf(stderr, "refinement: failed to create context\n"); exit(1); }
        owns_ctx = true;
    } else {
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // build batch with logits requested at positions [n_prompt-1 .. n_total-1]
    llama_batch batch = llama_batch_init(n_total, 0, 1);
    batch.n_tokens = n_total;
    for (int i = 0; i < n_total; i++) {
        batch.token[i]      = full_seq[i];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i] = (i >= n_prompt - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "refinement: decode failed\n"); exit(1);
    }

    // now collect logits and refine
    int changed = 0;
    int n_frozen = 0;
    double sum_conf = 0.0;

    for (int j = 0; j < n_gen; j++) {
        int logit_pos = n_prompt - 1 + j;
        const float * logits = llama_get_logits_ith(ctx, logit_pos);
        if (!logits) {
            fprintf(stderr, "refinement: no logits at position %d\n", logit_pos);
            continue;
        }

        float conf = confidence_from_logits(logits, n_vocab);
        sum_conf += conf;

        llama_token best = 0;
        float best_val = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best = v;
            }
        }

        if (best != candidate[j]) {
            candidate[j] = best;
            changed++;
        }

        if (conf >= confidence_threshold) {
            n_frozen++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // measure agreement with ground truth
    int n_compare = std::min(n_gen, (int)ground_truth.size());
    int n_match = 0;
    for (int j = 0; j < n_compare; j++) {
        if (candidate[j] == ground_truth[j]) n_match++;
    }

    llama_batch_free(batch);
    if (owns_ctx) {
        llama_free(ctx);
    }

    return {
        changed,
        n_frozen,
        (float)(sum_conf / n_gen),
        n_compare > 0 ? (float)n_match / n_compare * 100.0f : 0.0f,
        ms,
    };
}

// ─── wavefront (Gauss-Seidel) block refinement ──────────────────────────────
//
// Refine one block at a time, left to right.
// Each block sees: prompt + all converged blocks to its left + current block.
// Only tokens within the active block are replaced.
// A block converges when 0 tokens change or max_passes_per_block is reached.

struct block_result {
    int   block_idx;
    int   block_start;       // offset into candidate
    int   block_len;
    int   passes_used;
    float initial_agreement; // block-local agreement before refinement
    float final_agreement;   // block-local agreement after refinement
    double total_ms;
    double total_prefill_tokens; // sum of tokens prefilled across all passes for this block
};

static block_result run_block_refinement(
    llama_model * model,
    llama_context * shared_ctx,                     // if non-null, reuse this context
    const std::vector<llama_token> & prompt_tokens,
    std::vector<llama_token> & candidate,          // full candidate, modified in-place for this block
    const std::vector<llama_token> & ground_truth,
    int block_start,                                // start offset in candidate
    int block_len,                                  // tokens in this block
    int max_passes_per_block,
    float confidence_threshold,
    bool reuse_kv = false,                          // if true, preserve left-context KV cache between passes
    bool left_ctx_ready = false)                    // if true, left context is already in KV cache (skip first full prefill)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_prompt = (int)prompt_tokens.size();
    const int n_ctx_tokens = n_prompt + block_start + block_len;

    // position where left context ends (its logits predict block[0])
    const int left_ctx_end_pos = n_prompt + block_start - 1;

    auto t_block_start = std::chrono::high_resolution_clock::now();
    double total_prefill_toks = 0.0;

    // measure initial block-local agreement
    int n_compare = std::min(block_len, (int)ground_truth.size() - block_start);
    auto count_block_matches = [&]() {
        int m = 0;
        for (int j = 0; j < n_compare; j++) {
            if (candidate[block_start + j] == ground_truth[block_start + j]) m++;
        }
        return m;
    };
    float init_agree = n_compare > 0 ? (float)count_block_matches() / n_compare * 100.0f : 0.0f;

    // for KV reuse: track whether left context is already cached
    bool kv_cached = left_ctx_ready;
    // cached logits from the left-context end position (predicts block[0], doesn't change)
    std::vector<float> cached_left_logits;

    int pass;
    for (pass = 1; pass <= max_passes_per_block; pass++) {
        llama_context * ctx = shared_ctx;
        bool owns_ctx = false;

        if (!ctx) {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx   = n_ctx_tokens + 16;
            ctx_params.n_batch = n_ctx_tokens;
            ctx_params.no_perf = true;
            ctx = llama_init_from_model(model, ctx_params);
            if (!ctx) { fprintf(stderr, "block_refine: ctx failed\n"); exit(1); }
            owns_ctx = true;
        }

        bool did_partial = false;

        if (reuse_kv && kv_cached && ctx == shared_ctx) {
            // try to remove block positions from memory, preserving left context
            llama_memory_t mem = llama_get_memory(ctx);

            // if we don't have cached left logits yet (first pass of new block with
            // left_ctx_ready), remove from left_ctx_end_pos to also re-decode that
            // position and get its logits
            bool need_left_logits = cached_left_logits.empty() && left_ctx_end_pos >= 0;
            int rm_from = need_left_logits ? left_ctx_end_pos : (n_prompt + block_start);
            bool ok = llama_memory_seq_rm(mem, 0, rm_from, -1);

            if (ok) {
                // partial removal succeeded — decode block tokens (+ possibly last left-ctx token)
                int extra = need_left_logits ? 1 : 0;
                int batch_len = block_len + extra;
                llama_batch batch = llama_batch_init(batch_len, 0, 1);
                batch.n_tokens = batch_len;

                int bi = 0;
                if (need_left_logits) {
                    // re-decode the last left-context token to get its logits
                    batch.token[bi]     = candidate[block_start - 1 < 0 ?
                                            0 : block_start - 1]; // shouldn't happen, but safe
                    // actually need the correct token at left_ctx_end_pos
                    // that's either a prompt token or a candidate token
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

                if (llama_decode(ctx, batch)) {
                    fprintf(stderr, "block_refine: partial decode failed\n"); exit(1);
                }
                total_prefill_toks += batch_len;

                int changed = 0;

                // block[0]: get logits from either the re-decoded left-ctx position or cached
                if (need_left_logits) {
                    // logits at batch position 0 = left_ctx_end_pos, predicts block[0]
                    const float * logits = llama_get_logits_ith(ctx, 0);
                    if (logits) {
                        cached_left_logits.assign(logits, logits + n_vocab);
                        llama_token best = 0;
                        float best_val = logits[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
                        }
                        if (best != candidate[block_start]) {
                            candidate[block_start] = best;
                            changed++;
                        }
                    }
                    // block[1..]: logits at batch[extra + j - 1] predict block[j]
                    for (int j = 1; j < block_len; j++) {
                        const float * logits2 = llama_get_logits_ith(ctx, extra + j - 1);
                        if (!logits2) continue;
                        llama_token best = 0;
                        float best_val = logits2[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (logits2[v] > best_val) { best_val = logits2[v]; best = v; }
                        }
                        if (best != candidate[block_start + j]) {
                            candidate[block_start + j] = best;
                            changed++;
                        }
                    }
                } else if (!cached_left_logits.empty()) {
                    // use cached logits for block[0]
                    llama_token best = 0;
                    float best_val = cached_left_logits[0];
                    for (int v = 1; v < n_vocab; v++) {
                        if (cached_left_logits[v] > best_val) { best_val = cached_left_logits[v]; best = v; }
                    }
                    if (best != candidate[block_start]) {
                        candidate[block_start] = best;
                        changed++;
                    }
                    // block[1..]: logits at batch[j-1] predict block[j]
                    for (int j = 1; j < block_len; j++) {
                        const float * logits2 = llama_get_logits_ith(ctx, j - 1);
                        if (!logits2) continue;
                        llama_token best2 = 0;
                        float best_val2 = logits2[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (logits2[v] > best_val2) { best_val2 = logits2[v]; best2 = v; }
                        }
                        if (best2 != candidate[block_start + j]) {
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
            // full prefill: clear all memory, decode entire sequence
            if (ctx == shared_ctx) {
                llama_memory_clear(llama_get_memory(ctx), true);
            }

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

            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "block_refine: decode failed\n"); exit(1);
            }
            total_prefill_toks += n_ctx_tokens;

            // cache left-context logits for KV reuse on subsequent passes
            if (reuse_kv && ctx == shared_ctx && left_ctx_end_pos >= 0) {
                const float * ll = llama_get_logits_ith(ctx, left_ctx_end_pos);
                if (ll) {
                    cached_left_logits.assign(ll, ll + n_vocab);
                }
                kv_cached = true;
            }

            // replace tokens in this block
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
                if (best != candidate[block_start + j]) {
                    candidate[block_start + j] = best;
                    changed++;
                }
            }

            llama_batch_free(batch);
            if (owns_ctx) {
                llama_free(ctx);
            }

            if (changed == 0) break;
        }
    }

    auto t_block_end = std::chrono::high_resolution_clock::now();
    double block_ms = std::chrono::duration<double, std::milli>(t_block_end - t_block_start).count();

    float final_agree = n_compare > 0 ? (float)count_block_matches() / n_compare * 100.0f : 0.0f;

    return {0, block_start, block_len, pass, init_agree, final_agree, block_ms, total_prefill_toks};
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    std::string small_model_path;
    std::string big_model_path;
    std::string prompt = "Explain how a transistor works in 3 sentences.";
    int n_generate    = 128;
    int max_passes    = 1;
    float conf_thresh = 0.95f;
    int ngl           = 99;
    std::string mode  = "jacobi";   // "jacobi", "wavefront", "sweep", "soft", or "soft-wavefront"
    int block_size    = 16;          // tokens per block in wavefront mode
    bool reuse_ctx    = false;       // reuse llama_context across passes (avoid creation overhead)
    bool reuse_kv     = false;       // preserve left-context KV cache between passes within a block
    float soft_temp   = 2.0f;        // initial temperature for soft mode
    float soft_gamma  = 0.75f;       // temperature decay per pass
    float soft_alpha  = 0.3f;        // base damping coefficient for soft mode
    int   soft_topk   = 32;          // top-k tokens for expected embedding computation
    float convergence_thresh = 0.0f; // 0 = disabled; e.g. 0.90 = stop when 90% of tokens stable
    // Draft sampling params (small model)
    float draft_rep_penalty = 1.0f;
    float draft_min_p       = 0.0f;
    int   draft_top_k       = 0;
    float draft_temp        = 0.0f;
    int   output_jacobi     = 0;     // finisher passes over whole output (0=disabled)
    float jacobi_temp       = 0.0f;  // Jacobi annealing: start temp, decays to 0 (0=greedy)
    float conf_min          = 0.0f;  // V-schedule min (0=disabled flat schedule). e.g. 0.85
    int   topk_accept       = 1;     // accept candidate if in top-K predictions (1=exact match only)
    bool  self_con_check    = false; // run self-consistency check (expensive diagnostic)
    std::string refine_mode = "all"; // "all", "n-worst", "longest-run"
    int   refine_n          = 1;     // for n-worst: how many tokens to fix per pass

    // parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--small-model") == 0 && i+1 < argc) { small_model_path = argv[++i]; }
        else if (strcmp(argv[i], "--big-model") == 0 && i+1 < argc) { big_model_path = argv[++i]; }
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) { prompt = argv[++i]; }
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) { n_generate = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max-passes") == 0 && i+1 < argc) { max_passes = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--confidence") == 0 && i+1 < argc) { conf_thresh = atof(argv[++i]); }
        else if (strcmp(argv[i], "-ngl") == 0 && i+1 < argc) { ngl = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) { mode = argv[++i]; }
        else if (strcmp(argv[i], "--block-size") == 0 && i+1 < argc) { block_size = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--reuse-context") == 0) { reuse_ctx = true; }
        else if (strcmp(argv[i], "--reuse-kv") == 0) { reuse_kv = true; reuse_ctx = true; }
        else if (strcmp(argv[i], "--soft-temp") == 0 && i+1 < argc) { soft_temp = atof(argv[++i]); }
        else if (strcmp(argv[i], "--soft-gamma") == 0 && i+1 < argc) { soft_gamma = atof(argv[++i]); }
        else if (strcmp(argv[i], "--soft-alpha") == 0 && i+1 < argc) { soft_alpha = atof(argv[++i]); }
        else if (strcmp(argv[i], "--soft-topk") == 0 && i+1 < argc) { soft_topk = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--convergence-threshold") == 0 && i+1 < argc) { convergence_thresh = atof(argv[++i]); }
        else if (strcmp(argv[i], "--draft-rep-penalty") == 0 && i+1 < argc) { draft_rep_penalty = atof(argv[++i]); }
        else if (strcmp(argv[i], "--draft-min-p") == 0 && i+1 < argc) { draft_min_p = atof(argv[++i]); }
        else if (strcmp(argv[i], "--draft-top-k") == 0 && i+1 < argc) { draft_top_k = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--draft-temp") == 0 && i+1 < argc) { draft_temp = atof(argv[++i]); }
        else if (strcmp(argv[i], "--output-jacobi") == 0 && i+1 < argc) { output_jacobi = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--jacobi-temp") == 0 && i+1 < argc) { jacobi_temp = atof(argv[++i]); }
        else if (strcmp(argv[i], "--confidence-min") == 0 && i+1 < argc) { conf_min = atof(argv[++i]); }
        else if (strcmp(argv[i], "--topk-accept") == 0 && i+1 < argc) { topk_accept = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--self-con") == 0) { self_con_check = true; }
        else if (strcmp(argv[i], "--refine-mode") == 0 && i+1 < argc) { refine_mode = argv[++i]; }
        else if (strcmp(argv[i], "--refine-n") == 0 && i+1 < argc) { refine_n = atoi(argv[++i]); }
        else {
            fprintf(stderr, "Usage: %s --small-model <path> --big-model <path> [options]\n", argv[0]);
            fprintf(stderr, "  --prompt <text>         Prompt text\n");
            fprintf(stderr, "  -n <int>                Tokens to generate (default: 128)\n");
            fprintf(stderr, "  --max-passes <int>      Max refinement passes (default: 20)\n");
            fprintf(stderr, "  --confidence <float>    Freeze threshold (default: 0.80)\n");
            fprintf(stderr, "  -ngl <int>              GPU layers (default: 99)\n");
            fprintf(stderr, "  --mode <jacobi|wavefront|sweep|soft|soft-wavefront|confidence-trace>  Refinement mode\n");
            fprintf(stderr, "  --block-size <int>      Tokens per block in wavefront mode (default: 16)\n");
            fprintf(stderr, "  --reuse-context         Reuse llama_context across passes (faster)\n");
            fprintf(stderr, "  --reuse-kv              Preserve left-context KV cache between passes (implies --reuse-context)\n");
            fprintf(stderr, "  --soft-temp <float>     Initial temperature for soft mode (default: 2.0)\n");
            fprintf(stderr, "  --soft-gamma <float>    Temperature decay per pass (default: 0.75)\n");
            fprintf(stderr, "  --soft-alpha <float>    Base damping coefficient (default: 0.3)\n");
            fprintf(stderr, "  --soft-topk <int>       Top-k for expected embedding (default: 32)\n");
            fprintf(stderr, "  --convergence-threshold <float>  Stop Jacobi when stability >= threshold (0=disabled, e.g. 0.90)\n");
            fprintf(stderr, "  --draft-rep-penalty <float>      Draft model repetition penalty (default: 1.0=off)\n");
            fprintf(stderr, "  --draft-min-p <float>            Draft model min-p sampling (default: 0.0=off)\n");
            fprintf(stderr, "  --draft-top-k <int>              Draft model top-k sampling (default: 0=off)\n");
            fprintf(stderr, "  --draft-temp <float>             Draft model temperature (default: 0.0=greedy)\n");
            fprintf(stderr, "  --output-jacobi <int>            Finisher Jacobi passes over whole output (default: 0=off)\n");
            fprintf(stderr, "  --confidence-min <float>         V-schedule minimum confidence (default: 0=flat). e.g. 0.85\n");
            fprintf(stderr, "  --topk-accept <int>              Accept candidate if in top-K predictions (default: 1=exact match)\n");
            fprintf(stderr, "  --self-con                       Enable self-consistency check (expensive diagnostic, off by default)\n");
            fprintf(stderr, "  --refine-mode <all|n-worst|n-least-conf|longest-run>  Refinement strategy (default: all)\n");
            fprintf(stderr, "  --refine-n <int>                 For n-worst: fix N tokens per pass (default: 1)\n");
            return 1;
        }
    }

    if (small_model_path.empty() || big_model_path.empty()) {
        fprintf(stderr, "Error: both --small-model and --big-model are required\n");
        return 1;
    }

    ggml_backend_load_all();

    // ════════════════════════════════════════════════════════════════════════
    // GENERATE MODE: functional end-to-end Jacobi decoding
    // ════════════════════════════════════════════════════════════════════════
    if (mode == "generate") {
        fprintf(stderr, "\n=== JACOBI GENERATE MODE ===\n");
        fprintf(stderr, "Batch size: %d tokens, Jacobi passes: %d, Max output: %d, Refine: %s",
                block_size, max_passes, n_generate, refine_mode.c_str());
        if (refine_mode == "n-worst" || refine_mode == "n-least-conf") fprintf(stderr, " (N=%d)", refine_n);
        if (convergence_thresh > 0.0f)
            fprintf(stderr, ", convergence threshold: %.0f%%", convergence_thresh * 100);
        fprintf(stderr, "\n");

        // Load both models
        llama_model_params small_mp = llama_model_default_params();
        small_mp.n_gpu_layers = ngl;
        llama_model * sm = llama_model_load_from_file(small_model_path.c_str(), small_mp);
        if (!sm) { fprintf(stderr, "Failed to load small model\n"); return 1; }

        llama_model_params big_mp = llama_model_default_params();
        big_mp.n_gpu_layers = ngl;
        llama_model * bm = llama_model_load_from_file(big_model_path.c_str(), big_mp);
        if (!bm) { fprintf(stderr, "Failed to load big model\n"); return 1; }

        const llama_vocab * sv = llama_model_get_vocab(sm);
        const llama_vocab * bv = llama_model_get_vocab(bm);
        const int bv_n_vocab = llama_vocab_n_tokens(bv);

        // Create persistent contexts (avoid per-batch creation overhead)
        int max_ctx = n_generate + 1024; // generous
        llama_context_params bcp = llama_context_default_params();
        bcp.n_ctx = max_ctx; bcp.n_batch = max_ctx; bcp.no_perf = true;
        llama_context * bctx = llama_init_from_model(bm, bcp);
        if (!bctx) { fprintf(stderr, "Failed to create big context\n"); return 1; }

        // Persistent small model context
        llama_context_params scp = llama_context_default_params();
        scp.n_ctx = max_ctx; scp.n_batch = 1024; scp.no_perf = true;
        llama_context * sctx = llama_init_from_model(sm, scp);
        if (!sctx) { fprintf(stderr, "Failed to create small context\n"); return 1; }

        // Tokenize prompt for big model
        std::vector<llama_token> big_prompt = tokenize(bv, prompt, true);

        // Accumulated refined output (big model tokens)
        std::vector<llama_token> refined_all;
        // Accumulated text (for re-tokenizing with small model)
        std::string output_text;

        // Start timing AFTER model loading (model load is not part of throughput)
        auto t_total_start = std::chrono::high_resolution_clock::now();
        int total_draft_ms = 0;
        int total_refine_ms = 0;
        int total_batches = 0;
        bool done = false;

        // Track KV cache validity across batches (for models supporting partial KV ops)
        int big_kv_valid_upto = 0;  // positions [0, big_kv_valid_upto) have valid KV in bctx
        bool supports_partial_kv = true;  // set to false if llama_memory_seq_rm fails

        fprintf(stderr, "\n--- Output ---\n");

        while (!done && (int)refined_all.size() < n_generate) {
            int remaining = n_generate - (int)refined_all.size();
            int batch_n = std::min(block_size, remaining);

            // ── Draft: small model generates next batch_n tokens ──
            auto t_draft_start = std::chrono::high_resolution_clock::now();

            // Build small model input: original prompt + refined output so far
            std::string full_text_for_small = prompt;
            if (!output_text.empty()) {
                full_text_for_small += output_text;
            }
            std::vector<llama_token> small_input = tokenize(sv, full_text_for_small, true);

            sampling_config draft_sconf;
            draft_sconf.rep_penalty = draft_rep_penalty;
            draft_sconf.min_p       = draft_min_p;
            draft_sconf.top_k       = draft_top_k;
            draft_sconf.temp        = draft_temp;
            auto draft = generate_ar(sm, small_input, batch_n, ngl, draft_sconf, sctx);

            auto t_draft_end = std::chrono::high_resolution_clock::now();
            total_draft_ms += (int)std::chrono::duration<double, std::milli>(t_draft_end - t_draft_start).count();

            if (draft.tokens.empty()) {
                fprintf(stderr, "\n[Small model produced no tokens — stopping]\n");
                break;
            }

            // Detokenize draft and re-tokenize with big model vocab
            std::string draft_text;
            for (auto id : draft.tokens) {
                char buf[256];
                int n = llama_token_to_piece(sv, id, buf, sizeof(buf), 0, true);
                if (n > 0) draft_text.append(buf, n);
            }

            std::vector<llama_token> candidate;
            {
                int cn = -llama_tokenize(bv, draft_text.c_str(), draft_text.size(), nullptr, 0, false, true);
                candidate.resize(cn);
                llama_tokenize(bv, draft_text.c_str(), draft_text.size(), candidate.data(), candidate.size(), false, true);
            }

            // Trim to batch_n
            if ((int)candidate.size() > batch_n) candidate.resize(batch_n);
            int n_cand = (int)candidate.size();

            // ── Refine: Jacobi passes with prefix acceptance ──
            //
            // Key idea: track how many tokens from the LEFT have converged
            // (model's argmax == candidate). Once accepted into the prefix,
            // those tokens don't need re-evaluation. We also gate updates
            // on confidence: only change a token if the model is more
            // confident in the new token than the current one.
            auto t_refine_start = std::chrono::high_resolution_clock::now();

            // Build prefix: big_prompt + all refined tokens so far
            std::vector<llama_token> prefix;
            prefix.insert(prefix.end(), big_prompt.begin(), big_prompt.end());
            prefix.insert(prefix.end(), refined_all.begin(), refined_all.end());
            int n_prefix_base = (int)prefix.size();

            // Track which candidate tokens are "frozen" (accepted into prefix)
            int accepted_prefix_len = 0;  // how many candidate tokens from left are frozen

            int passes_used = 0;
            float final_stability = 0.0f;

            // V-shaped confidence schedule: descend from conf_thresh to conf_min,
            // then ascend back. Step = 0.01 per pass.
            // If conf_min == 0, use flat conf_thresh for all passes.
            const float conf_step = 0.01f;
            // How many passes to reach the valley
            int valley_pass = (conf_min > 0.0f)
                ? (int)((conf_thresh - conf_min) / conf_step)
                : 0;

            // Use cross-batch KV validity tracking
            int kv_valid_upto = big_kv_valid_upto;

            // Track changed candidate indices for cascade-only checking
            std::vector<int> prev_changed_indices;  // candidate indices that changed last pass

            for (int pass = 1; pass <= max_passes; pass++) {
                // Compute per-pass confidence threshold
                float pass_conf = conf_thresh;  // default: flat
                if (conf_min > 0.0f && conf_min < conf_thresh) {
                    if (pass <= valley_pass + 1) {
                        pass_conf = conf_thresh - (pass - 1) * conf_step;
                        if (pass_conf < conf_min) pass_conf = conf_min;
                    } else {
                        pass_conf = conf_min + (pass - valley_pass - 1) * conf_step;
                        if (pass_conf > conf_thresh) pass_conf = conf_thresh;
                    }
                }

                // Current prefix = base prefix + accepted candidate prefix
                int n_prefix = n_prefix_base + accepted_prefix_len;
                int n_remaining = n_cand - accepted_prefix_len;
                if (n_remaining <= 0) break;  // all accepted

                // Determine what to submit based on KV cache state
                int submit_from;  // absolute position to start submitting tokens
                if (pass == 1 && kv_valid_upto >= n_prefix_base && supports_partial_kv) {
                    // First pass with valid KV from previous batch!
                    // KV covers [0, kv_valid_upto) which includes the entire prefix.
                    // We only need to submit the new candidate tokens.
                    // Remove any stale KV beyond the prefix (shouldn't exist, but be safe)
                    llama_memory_seq_rm(llama_get_memory(bctx), 0, n_prefix_base, -1);
                    submit_from = n_prefix_base;
                } else if (pass == 1) {
                    // First pass, no usable KV: clear and submit full sequence
                    llama_memory_clear(llama_get_memory(bctx), true);
                    submit_from = 0;
                } else {
                    // If all KV is valid (nothing changed last pass), we're done
                    if (kv_valid_upto >= n_prefix + n_remaining) break;

                    // Subsequent passes: try to keep KV for [0, kv_valid_upto),
                    // and only remove stale KV from kv_valid_upto onward
                    bool partial_ok = true;
                    if (kv_valid_upto > 0) {
                        partial_ok = llama_memory_seq_rm(llama_get_memory(bctx), 0,
                                                         kv_valid_upto, -1);
                    }
                    if (partial_ok && kv_valid_upto > 0) {
                        submit_from = kv_valid_upto;
                    } else {
                        // Partial removal not supported — full clear fallback
                        supports_partial_kv = false;
                        llama_memory_clear(llama_get_memory(bctx), true);
                        submit_from = 0;
                    }
                }

                // Build batch: only tokens from submit_from onward
                int n_total_seq = n_prefix + n_remaining;
                int n_submit = n_total_seq - submit_from;
                llama_batch batch = llama_batch_init(n_submit, 0, 1);
                batch.n_tokens = n_submit;

                // Build set of candidate indices to check on this pass
                // Pass 1: check all. Pass 2+: only near previous changes (±2).
                // For n-worst and longest-run modes, always check all (need full picture).
                const int cascade_radius = 2;
                bool use_cascade = (refine_mode == "all");
                std::unordered_set<int> check_set;
                if (pass == 1 || prev_changed_indices.empty() || !use_cascade) {
                    // Check everything
                    for (int j = 0; j < n_remaining; j++) {
                        check_set.insert(accepted_prefix_len + j);
                    }
                } else {
                    // Only check positions near previous changes
                    for (int ci : prev_changed_indices) {
                        for (int d = -cascade_radius; d <= cascade_radius; d++) {
                            int idx = ci + d;
                            if (idx >= accepted_prefix_len && idx < n_cand) {
                                check_set.insert(idx);
                            }
                        }
                    }
                    // Also check the prefix boundary (first non-accepted position)
                    if (accepted_prefix_len < n_cand) {
                        check_set.insert(accepted_prefix_len);
                    }
                }

                // Set logits flags: only request logits at positions we'll check
                for (int i = 0; i < n_submit; i++) {
                    int abs_pos = submit_from + i;
                    llama_token tok;
                    if (abs_pos < n_prefix_base) {
                        tok = prefix[abs_pos];
                    } else {
                        tok = candidate[abs_pos - n_prefix_base];
                    }
                    batch.token[i]     = tok;
                    batch.pos[i]       = abs_pos;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    // Request logits only for positions we'll check
                    // The logit at position P predicts token at position P+1,
                    // so to check candidate[ci] we need logits at abs_pos = n_prefix_base + ci - 1
                    int cand_idx_predicted = abs_pos - n_prefix_base + 1;
                    batch.logits[i] = (cand_idx_predicted >= 0 &&
                                       check_set.count(cand_idx_predicted) > 0) ? 1 : 0;
                }

                if (llama_decode(bctx, batch)) {
                    fprintf(stderr, "\n[Decode failed at batch %d pass %d]\n", total_batches, pass);
                    llama_batch_free(batch);
                    done = true;
                    break;
                }

                // Collect all disagreements, then selectively apply based on refine mode
                struct disagree_info {
                    int cand_idx;
                    llama_token argmax;
                    float conf;       // big model's confidence in its preferred token
                    float cur_prob;   // big model's probability of the CURRENT draft token
                };
                std::vector<disagree_info> disagreements;
                int new_prefix_accepted = 0;
                bool prefix_broken = false;

                for (int j = 0; j < n_remaining; j++) {
                    int cand_idx = accepted_prefix_len + j;
                    bool should_check = (check_set.count(cand_idx) > 0);

                    int abs_logit_pos = n_prefix - 1 + j;
                    int batch_idx = abs_logit_pos - submit_from;
                    if (batch_idx < 0) { prefix_broken = true; continue; }

                    if (!should_check) {
                        prefix_broken = true;
                        continue;
                    }

                    const float * logits = llama_get_logits_ith(bctx, batch_idx);
                    if (!logits) { prefix_broken = true; continue; }

                    llama_token argmax = 0;
                    float best_val = logits[0];
                    for (int v = 1; v < bv_n_vocab; v++) {
                        if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
                    }

                    bool in_topk = (argmax == candidate[cand_idx]);
                    if (!in_topk && topk_accept > 1) {
                        float thresh = best_val;
                        for (int k = 1; k < topk_accept && !in_topk; k++) {
                            float next_best = -1e30f;
                            for (int v = 0; v < bv_n_vocab; v++) {
                                if (logits[v] < thresh && logits[v] > next_best) {
                                    next_best = logits[v];
                                }
                            }
                            thresh = next_best;
                            if (logits[candidate[cand_idx]] >= thresh) {
                                in_topk = true;
                            }
                        }
                    }

                    if (in_topk) {
                        if (!prefix_broken) new_prefix_accepted++;
                    } else {
                        prefix_broken = true;
                        float conf_new = confidence_from_logits(logits, bv_n_vocab);
                        if (conf_new >= pass_conf) {
                            // Compute softmax probability of the current draft token
                            // (how much the big model believes the existing token)
                            float cur_logit = logits[candidate[cand_idx]];
                            double sum_exp = 0.0;
                            for (int v = 0; v < bv_n_vocab; v++) {
                                sum_exp += exp((double)(logits[v] - best_val));
                            }
                            float cur_p = (float)(exp((double)(cur_logit - best_val)) / sum_exp);
                            disagreements.push_back({cand_idx, argmax, conf_new, cur_p});
                        }
                    }
                }

                // Select which disagreements to actually apply based on refine mode
                std::vector<disagree_info> to_apply;
                if (refine_mode == "n-worst") {
                    // Sort by confidence descending (most confident corrections first)
                    std::sort(disagreements.begin(), disagreements.end(),
                        [](const disagree_info & a, const disagree_info & b) {
                            return a.conf > b.conf;
                        });
                    int n_fix = std::min(refine_n, (int)disagreements.size());
                    to_apply.assign(disagreements.begin(), disagreements.begin() + n_fix);
                } else if (refine_mode == "n-least-conf") {
                    // Sort by current token probability ascending (least believed tokens first)
                    // These are the tokens the big model thinks are most damaged
                    std::sort(disagreements.begin(), disagreements.end(),
                        [](const disagree_info & a, const disagree_info & b) {
                            return a.cur_prob < b.cur_prob;
                        });
                    int n_fix = std::min(refine_n, (int)disagreements.size());
                    to_apply.assign(disagreements.begin(), disagreements.begin() + n_fix);
                } else if (refine_mode == "longest-run") {
                    // Find the longest contiguous run of disagreements
                    if (!disagreements.empty()) {
                        // Sort by position
                        std::sort(disagreements.begin(), disagreements.end(),
                            [](const disagree_info & a, const disagree_info & b) {
                                return a.cand_idx < b.cand_idx;
                            });
                        int best_start = 0, best_len = 1;
                        int cur_start = 0, cur_len = 1;
                        for (int i = 1; i < (int)disagreements.size(); i++) {
                            if (disagreements[i].cand_idx == disagreements[i-1].cand_idx + 1) {
                                cur_len++;
                            } else {
                                if (cur_len > best_len) {
                                    best_start = cur_start;
                                    best_len = cur_len;
                                }
                                cur_start = i;
                                cur_len = 1;
                            }
                        }
                        if (cur_len > best_len) {
                            best_start = cur_start;
                            best_len = cur_len;
                        }
                        to_apply.assign(disagreements.begin() + best_start,
                                       disagreements.begin() + best_start + best_len);
                    }
                } else {
                    // "all" — apply everything (original behavior)
                    to_apply = disagreements;
                }

                // Apply selected corrections
                int changed = 0;
                int earliest_change = n_prefix + n_remaining;
                std::vector<int> cur_changed_indices;
                for (auto & d : to_apply) {
                    candidate[d.cand_idx] = d.argmax;
                    changed++;
                    cur_changed_indices.push_back(d.cand_idx);
                    int change_pos = n_prefix_base + d.cand_idx;
                    if (change_pos < earliest_change) earliest_change = change_pos;
                }

                // Save changed indices for next pass cascade checking
                prev_changed_indices = cur_changed_indices;

                // Advance accepted prefix
                accepted_prefix_len += new_prefix_accepted;

                // Update KV validity boundary
                if (changed == 0) {
                    kv_valid_upto = n_prefix + n_remaining;  // all KV valid
                } else {
                    kv_valid_upto = earliest_change;  // invalidate from first change
                }

                llama_batch_free(batch);
                passes_used = pass;
                final_stability = (float)(n_cand - changed) / n_cand;

                if (conf_min > 0.0f || max_passes > 1) {
                    fprintf(stderr, "    pass %2d: conf=%.2f disagree=%d changed=%d prefix_accepted=%d/%d submitted=%d\n",
                            pass, pass_conf, (int)disagreements.size(), changed, accepted_prefix_len, n_cand, n_submit);
                    if ((refine_mode == "n-least-conf" || refine_mode == "n-worst") && !to_apply.empty()) {
                        for (auto & d : to_apply) {
                            fprintf(stderr, "      %s pos %d: cur_prob=%.4f repl_conf=%.4f\n",
                                    refine_mode.c_str(), d.cand_idx, d.cur_prob, d.conf);
                        }
                    }
                }

                if (changed == 0 && new_prefix_accepted == 0) break;  // true convergence

                if (convergence_thresh > 0.0f) {
                    float conv_pct = (float)accepted_prefix_len / n_cand;
                    if (conv_pct >= convergence_thresh) break;
                }
            }

            auto t_refine_end = std::chrono::high_resolution_clock::now();
            total_refine_ms += (int)std::chrono::duration<double, std::milli>(t_refine_end - t_refine_start).count();

            // ── Optional self-consistency check (expensive: doubles refine cost) ──
            // Off by default; enable with --self-con flag
            float self_con_pct = 0.0f;
            float mean_conf = 0.0f;
            bool did_selfcon_decode = false;
            if (self_con_check) {
                llama_memory_clear(llama_get_memory(bctx), true);
                int np = n_prefix_base;
                int n_total = np + n_cand;
                llama_batch vbatch = llama_batch_init(n_total, 0, 1);
                vbatch.n_tokens = n_total;
                for (int i = 0; i < np; i++) {
                    vbatch.token[i]     = prefix[i];
                    vbatch.pos[i]       = i;
                    vbatch.n_seq_id[i]  = 1;
                    vbatch.seq_id[i][0] = 0;
                    vbatch.logits[i]    = (i == np - 1) ? 1 : 0;
                }
                for (int j = 0; j < n_cand; j++) {
                    vbatch.token[np + j]     = candidate[j];
                    vbatch.pos[np + j]       = np + j;
                    vbatch.n_seq_id[np + j]  = 1;
                    vbatch.seq_id[np + j][0] = 0;
                    vbatch.logits[np + j]    = 1;
                }
                if (!llama_decode(bctx, vbatch)) {
                    did_selfcon_decode = true;
                    int n_match = 0;
                    float sum_conf = 0.0f;
                    struct disagreement { int pos; float conf; std::string have; std::string want; };
                    std::vector<disagreement> disagrees;

                    for (int j = 0; j < n_cand; j++) {
                        int logit_pos = np - 1 + j;
                        const float * logits = llama_get_logits_ith(bctx, logit_pos);
                        if (!logits) continue;
                        llama_token argmax = 0;
                        float best_val = logits[0];
                        for (int v = 1; v < bv_n_vocab; v++) {
                            if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
                        }
                        float conf = confidence_from_logits(logits, bv_n_vocab);
                        if (argmax == candidate[j]) {
                            n_match++;
                        } else {
                            disagrees.push_back({j, conf,
                                token_to_str(bv, candidate[j]),
                                token_to_str(bv, argmax)});
                        }
                        sum_conf += conf;
                    }
                    self_con_pct = 100.0f * n_match / n_cand;
                    mean_conf = sum_conf / n_cand;

                    if (!disagrees.empty()) {
                        std::sort(disagrees.begin(), disagrees.end(),
                            [](const disagreement & a, const disagreement & b) { return a.conf > b.conf; });
                        int show = std::min((int)disagrees.size(), 15);
                        fprintf(stderr, "  Top %d disagreements (big model confident token is wrong):\n", show);
                        for (int d = 0; d < show; d++) {
                            auto & dg = disagrees[d];
                            fprintf(stderr, "    pos %3d conf=%.3f  have=%-20s  want=%-20s\n",
                                    dg.pos, dg.conf, dg.have.c_str(), dg.want.c_str());
                        }
                    }
                }
                llama_batch_free(vbatch);
            }

            // ── Check for stop token and collect output ──
            int emit_len = n_cand;
            for (int j = 0; j < n_cand; j++) {
                if (llama_vocab_is_eog(bv, candidate[j])) {
                    emit_len = j;
                    done = true;
                    break;
                }
            }

            // Update cross-batch KV validity
            if (supports_partial_kv && !done) {
                if (did_selfcon_decode) {
                    // Self-con check rebuilt KV for [0, n_prefix_base + n_cand)
                    if (emit_len < n_cand) {
                        llama_memory_seq_rm(llama_get_memory(bctx), 0,
                                            n_prefix_base + emit_len, -1);
                    }
                    big_kv_valid_upto = n_prefix_base + emit_len;
                } else {
                    // No self-con check: KV from refine pass is valid up to kv_valid_upto
                    // Trim anything beyond emitted tokens
                    int desired_end = n_prefix_base + emit_len;
                    if (kv_valid_upto > desired_end) {
                        llama_memory_seq_rm(llama_get_memory(bctx), 0,
                                            desired_end, -1);
                    }
                    big_kv_valid_upto = std::min(kv_valid_upto, desired_end);
                }
            } else {
                big_kv_valid_upto = 0;
            }

            // Emit refined tokens
            std::string batch_text;
            for (int j = 0; j < emit_len; j++) {
                char buf[256];
                int n = llama_token_to_piece(bv, candidate[j], buf, sizeof(buf), 0, true);
                if (n > 0) batch_text.append(buf, n);
            }

            // Print output as it's produced
            fprintf(stdout, "%s", batch_text.c_str());
            fflush(stdout);

            // Accumulate
            refined_all.insert(refined_all.end(), candidate.begin(), candidate.begin() + emit_len);
            output_text += batch_text;
            total_batches++;

            fprintf(stderr, "[batch %d: drafted %d → refined %d tokens, %d passes, prefix-accepted %d/%d, self-con %.1f%%, mean-conf %.3f, %s]\n",
                    total_batches, (int)draft.tokens.size(), emit_len, passes_used,
                    accepted_prefix_len, n_cand, self_con_pct, mean_conf,
                    done ? "STOP" : "continue");
        }

        fprintf(stdout, "\n");

        // ── Finisher: full-output Jacobi passes ──
        if (output_jacobi > 0 && !refined_all.empty()) {
            fprintf(stderr, "\n--- Finisher: %d Jacobi passes over %d output tokens ---\n",
                    output_jacobi, (int)refined_all.size());

            // Need a context big enough for prompt + full output
            int fin_total = (int)big_prompt.size() + (int)refined_all.size();
            llama_free(bctx);
            llama_context_params fcp = llama_context_default_params();
            fcp.n_ctx = fin_total + 16; fcp.n_batch = fin_total + 16; fcp.no_perf = true;
            bctx = llama_init_from_model(bm, fcp);

            int n_out = (int)refined_all.size();
            int n_pre = (int)big_prompt.size();

            // Measure pre-finisher self-consistency
            auto measure_selfcon = [&]() -> std::pair<float,float> {
                llama_memory_clear(llama_get_memory(bctx), true);
                llama_batch fb = llama_batch_init(fin_total, 0, 1);
                fb.n_tokens = fin_total;
                for (int i = 0; i < n_pre; i++) {
                    fb.token[i] = big_prompt[i]; fb.pos[i] = i;
                    fb.n_seq_id[i] = 1; fb.seq_id[i][0] = 0;
                    fb.logits[i] = (i == n_pre - 1) ? 1 : 0;
                }
                for (int j = 0; j < n_out; j++) {
                    fb.token[n_pre + j] = refined_all[j]; fb.pos[n_pre + j] = n_pre + j;
                    fb.n_seq_id[n_pre + j] = 1; fb.seq_id[n_pre + j][0] = 0;
                    fb.logits[n_pre + j] = 1;
                }
                float sc = 0, mc = 0;
                if (!llama_decode(bctx, fb)) {
                    int nm = 0; float sm = 0;
                    for (int j = 0; j < n_out; j++) {
                        const float * lg = llama_get_logits_ith(bctx, n_pre - 1 + j);
                        if (!lg) continue;
                        llama_token am = 0; float bv2 = lg[0];
                        for (int v = 1; v < bv_n_vocab; v++)
                            if (lg[v] > bv2) { bv2 = lg[v]; am = v; }
                        if (am == refined_all[j]) nm++;
                        sm += confidence_from_logits(lg, bv_n_vocab);
                    }
                    sc = 100.0f * nm / n_out;
                    mc = sm / n_out;
                }
                llama_batch_free(fb);
                return {sc, mc};
            };

            auto [pre_sc, pre_mc] = measure_selfcon();
            fprintf(stderr, "  Pre-finisher:  self-con %.1f%%, mean-conf %.3f\n", pre_sc, pre_mc);

            auto t_fin_start = std::chrono::high_resolution_clock::now();
            for (int fp = 1; fp <= output_jacobi; fp++) {
                llama_memory_clear(llama_get_memory(bctx), true);
                llama_batch fb = llama_batch_init(fin_total, 0, 1);
                fb.n_tokens = fin_total;
                for (int i = 0; i < n_pre; i++) {
                    fb.token[i] = big_prompt[i]; fb.pos[i] = i;
                    fb.n_seq_id[i] = 1; fb.seq_id[i][0] = 0;
                    fb.logits[i] = (i == n_pre - 1) ? 1 : 0;
                }
                for (int j = 0; j < n_out; j++) {
                    fb.token[n_pre + j] = refined_all[j]; fb.pos[n_pre + j] = n_pre + j;
                    fb.n_seq_id[n_pre + j] = 1; fb.seq_id[n_pre + j][0] = 0;
                    fb.logits[n_pre + j] = 1;
                }
                if (llama_decode(bctx, fb)) {
                    fprintf(stderr, "  Finisher pass %d decode failed\n", fp);
                    llama_batch_free(fb);
                    break;
                }
                int changed = 0;
                for (int j = 0; j < n_out; j++) {
                    const float * lg = llama_get_logits_ith(bctx, n_pre - 1 + j);
                    if (!lg) continue;
                    llama_token am = 0; float bv2 = lg[0];
                    for (int v = 1; v < bv_n_vocab; v++)
                        if (lg[v] > bv2) { bv2 = lg[v]; am = v; }
                    if (am != refined_all[j]) { refined_all[j] = am; changed++; }
                }
                llama_batch_free(fb);
                fprintf(stderr, "  Finisher pass %d: %d/%d changed\n", fp, changed, n_out);
                if (changed == 0) break;
            }

            auto [post_sc, post_mc] = measure_selfcon();
            auto t_fin_end = std::chrono::high_resolution_clock::now();
            double fin_ms = std::chrono::duration<double, std::milli>(t_fin_end - t_fin_start).count();
            fprintf(stderr, "  Post-finisher: self-con %.1f%%, mean-conf %.3f (%.0f ms)\n",
                    post_sc, post_mc, fin_ms);

            // Reprint the finisher output
            output_text.clear();
            for (auto id : refined_all) {
                char buf[256];
                int n = llama_token_to_piece(bv, id, buf, sizeof(buf), 0, true);
                if (n > 0) output_text.append(buf, n);
            }
            fprintf(stderr, "\n--- Finisher output ---\n");
            fprintf(stdout, "%s\n", output_text.c_str());
            fflush(stdout);
        }

        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        double total_tps = refined_all.size() / (total_ms / 1000.0);

        fprintf(stderr, "\n=== GENERATE SUMMARY ===\n");
        fprintf(stderr, "Total tokens:         %d\n", (int)refined_all.size());
        fprintf(stderr, "Batches:              %d (batch_size=%d)\n", total_batches, block_size);
        fprintf(stderr, "Draft time:           %d ms\n", total_draft_ms);
        fprintf(stderr, "Refine time:          %d ms\n", total_refine_ms);
        fprintf(stderr, "Total time:           %.0f ms\n", total_ms);
        fprintf(stderr, "Effective tok/s:      %.1f\n", total_tps);

        llama_free(sctx);
        llama_free(bctx);
        llama_model_free(bm);
        llama_model_free(sm);
        return 0;
    }

    // ════════════════════════════════════════════════════════════════════════
    // TEST MODES: need ground truth comparison
    // ════════════════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────────────────
    // Step 1: Load small model, generate rough answer
    // ────────────────────────────────────────────────────────────────────────

    fprintf(stderr, "\n=== STEP 1: Generate rough answer (small model) ===\n");
    fprintf(stderr, "Loading: %s\n", small_model_path.c_str());

    llama_model_params small_mparams = llama_model_default_params();
    small_mparams.n_gpu_layers = ngl;
    llama_model * small_model = llama_model_load_from_file(small_model_path.c_str(), small_mparams);
    if (!small_model) { fprintf(stderr, "Failed to load small model\n"); return 1; }

    const llama_vocab * small_vocab = llama_model_get_vocab(small_model);
    std::vector<llama_token> small_prompt_tokens = tokenize(small_vocab, prompt, true);
    fprintf(stderr, "Prompt (%d tokens): %s\n", (int)small_prompt_tokens.size(), prompt.c_str());

    auto rough = generate_ar(small_model, small_prompt_tokens, n_generate, ngl);
    double rough_tps = rough.tokens.size() / (rough.elapsed_ms / 1000.0);
    fprintf(stderr, "Rough answer (%d tokens, %.0f ms, %.1f tok/s):\n",
            (int)rough.tokens.size(), rough.elapsed_ms, rough_tps);
    fprintf(stderr, ">>> %s\n", tokens_to_str(small_vocab, rough.tokens).c_str());

    llama_model_free(small_model);

    // ────────────────────────────────────────────────────────────────────────
    // Step 2: Load big model, generate ground truth (greedy, temp=0)
    // ────────────────────────────────────────────────────────────────────────

    fprintf(stderr, "\n=== STEP 2: Generate ground truth (big model, greedy) ===\n");
    fprintf(stderr, "Loading: %s\n", big_model_path.c_str());

    llama_model_params big_mparams = llama_model_default_params();
    big_mparams.n_gpu_layers = ngl;
    llama_model * big_model = llama_model_load_from_file(big_model_path.c_str(), big_mparams);
    if (!big_model) { fprintf(stderr, "Failed to load big model\n"); return 1; }

    const llama_vocab * big_vocab = llama_model_get_vocab(big_model);
    std::vector<llama_token> big_prompt_tokens = tokenize(big_vocab, prompt, true);

    auto truth = generate_ar(big_model, big_prompt_tokens, n_generate, ngl);
    double truth_tps = truth.tokens.size() / (truth.elapsed_ms / 1000.0);
    fprintf(stderr, "Ground truth (%d tokens, %.0f ms, %.1f tok/s):\n",
            (int)truth.tokens.size(), truth.elapsed_ms, truth_tps);
    fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, truth.tokens).c_str());

    // ────────────────────────────────────────────────────────────────────────
    // Step 3: Re-tokenize the rough answer using the big model's vocab
    //
    // The small and big models may use different tokenizers.
    // Safest approach: detokenize the rough answer with the small vocab,
    // then re-tokenize with the big vocab.
    // ────────────────────────────────────────────────────────────────────────

    fprintf(stderr, "\n=== STEP 3: Prepare rough answer for refinement ===\n");

    // Detokenize rough answer to text, then re-tokenize with big model vocab
    // We need the raw text of the rough output (no BOS)
    std::string rough_text;
    for (auto id : rough.tokens) {
        char buf[256];
        // Use the small model's vocab to detokenize
        int n = llama_token_to_piece(small_vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) rough_text.append(buf, n);
    }

    // Re-tokenize the rough text with big model vocab (no BOS — this is continuation)
    int rough_retok_n = -llama_tokenize(big_vocab, rough_text.c_str(), rough_text.size(), nullptr, 0, false, true);
    std::vector<llama_token> candidate(rough_retok_n);
    llama_tokenize(big_vocab, rough_text.c_str(), rough_text.size(), candidate.data(), candidate.size(), false, true);

    // Trim or pad to match ground truth length
    int n_gen = std::min((int)candidate.size(), (int)truth.tokens.size());
    candidate.resize(n_gen);

    // Measure initial agreement
    int initial_match = 0;
    for (int j = 0; j < n_gen; j++) {
        if (candidate[j] == truth.tokens[j]) initial_match++;
    }
    float initial_agreement = (float)initial_match / n_gen * 100.0f;

    fprintf(stderr, "Rough answer re-tokenized: %d tokens (big model vocab)\n", n_gen);
    fprintf(stderr, "Initial agreement with ground truth: %.1f%% (%d/%d tokens)\n",
            initial_agreement, initial_match, n_gen);

    // ────────────────────────────────────────────────────────────────────────
    // Step 4: Iterative refinement
    // ────────────────────────────────────────────────────────────────────────

    bool use_wavefront = (mode == "wavefront");
    bool use_sweep     = (mode == "sweep");

    // optionally create a shared context for reuse across all refinement passes
    llama_context * shared_ctx = nullptr;
    if (reuse_ctx) {
        // size the context for the largest possible batch: prompt + all candidate tokens
        int max_ctx = (int)big_prompt_tokens.size() + n_gen + 16;
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx   = max_ctx;
        ctx_params.n_batch = max_ctx;
        ctx_params.no_perf = true;

        shared_ctx = llama_init_from_model(big_model, ctx_params);
        if (!shared_ctx) { fprintf(stderr, "Failed to create shared context\n"); return 1; }
        fprintf(stderr, "Reusing shared context (n_ctx=%d)\n", max_ctx);
    }

    if (use_sweep) {
        // ── Sweep mode ──────────────────────────────────────────────────
        // Run wavefront refinement for each block size, track milestones.
        // Models are already loaded — just re-run with different block sizes.

        fprintf(stderr, "\n=== STEP 4: BLOCK SIZE SWEEP ===\n");
        fprintf(stderr, "Tokens to refine: %d\n", n_gen);
        fprintf(stderr, "Max passes per block: %d\n\n", max_passes);

        const int sweep_block_sizes[] = {16, 32, 64, 128, 256, 512};
        const int n_sweep = 6;
        const float milestones[] = {70.0f, 80.0f, 90.0f, 95.0f, 99.0f, 100.0f};
        const int n_milestones = 6;

        // results[block_size_idx][milestone_idx] = total passes to reach milestone (-1 if not reached)
        int results[6][6];
        double sweep_time[6];       // total wall-clock per block size
        double sweep_eff_decode[6]; // effective decode tok/s per block size

        for (int si = 0; si < n_sweep; si++) {
            int bs = sweep_block_sizes[si];

            // skip block sizes larger than the sequence
            if (bs > n_gen) {
                for (int mi = 0; mi < n_milestones; mi++) results[si][mi] = -1;
                sweep_time[si] = 0;
                sweep_eff_decode[si] = 0;
                continue;
            }

            // fresh copy of candidate for this block size
            std::vector<llama_token> cand_copy(candidate.begin(), candidate.end());

            // track cumulative passes across all blocks
            int cum_passes = 0;
            double cum_ms = 0.0;
            bool milestone_hit[6] = {false};
            for (int mi = 0; mi < n_milestones; mi++) results[si][mi] = -1;

            // check initial agreement against milestones
            {
                int m = 0;
                for (int j = 0; j < n_gen; j++) {
                    if (cand_copy[j] == truth.tokens[j]) m++;
                }
                float agree = (float)m / n_gen * 100.0f;
                for (int mi = 0; mi < n_milestones; mi++) {
                    if (!milestone_hit[mi] && agree >= milestones[mi]) {
                        milestone_hit[mi] = true;
                        results[si][mi] = 0; // reached at 0 passes
                    }
                }
            }

            int n_blocks = (n_gen + bs - 1) / bs;

            fprintf(stderr, "--- Block size %d (%d blocks) ---\n", bs, n_blocks);
            fprintf(stderr, "  %-6s  %-8s  %-8s  %-10s  %-10s  %-8s  %-10s  %-10s\n",
                    "Block", "Start", "Len", "InitAgr%", "FinalAgr%", "Passes", "Time(ms)", "Overall%");
            fprintf(stderr, "  ------  --------  --------  ----------  ----------  --------  ----------  ----------\n");

            // for KV reuse between blocks: clear memory before first block of each sweep iteration
            if (reuse_kv && shared_ctx) {
                llama_memory_clear(llama_get_memory(shared_ctx), true);
            }

            for (int b = 0; b < n_blocks; b++) {
                int bstart = b * bs;
                int blen   = std::min(bs, n_gen - bstart);

                // after block 0 converges, its KV is in the cache — subsequent blocks
                // can skip re-prefilling the left context
                bool left_ready = reuse_kv && (b > 0);

                block_result br = run_block_refinement(
                    big_model, shared_ctx, big_prompt_tokens, cand_copy, truth.tokens,
                    bstart, blen, max_passes, conf_thresh, reuse_kv, left_ready);

                cum_passes += br.passes_used;
                cum_ms += br.total_ms;

                // measure overall agreement after this block
                int m = 0;
                for (int j = 0; j < n_gen; j++) {
                    if (cand_copy[j] == truth.tokens[j]) m++;
                }
                float agree = (float)m / n_gen * 100.0f;

                // log each block as it completes
                fprintf(stderr, "  %-6d  %-8d  %-8d  %-10.1f  %-10.1f  %-8d  %-10.0f  %-10.1f\n",
                        b, bstart, blen, br.initial_agreement, br.final_agreement,
                        br.passes_used, br.total_ms, agree);

                // check milestones
                for (int mi = 0; mi < n_milestones; mi++) {
                    if (!milestone_hit[mi] && agree >= milestones[mi]) {
                        milestone_hit[mi] = true;
                        results[si][mi] = cum_passes;
                    }
                }
            }

            sweep_time[si] = cum_ms;
            sweep_eff_decode[si] = n_gen / (cum_ms / 1000.0);

            // final agreement
            int fm = 0;
            for (int j = 0; j < n_gen; j++) {
                if (cand_copy[j] == truth.tokens[j]) fm++;
            }
            fprintf(stderr, "  Final: %.1f%% (%d passes, %.0f ms, eff decode %.1f tok/s)\n\n",
                    (float)fm / n_gen * 100.0f, cum_passes, cum_ms, sweep_eff_decode[si]);
        }

        // ── Print results matrix ────────────────────────────────────────
        fprintf(stderr, "\n=== SWEEP RESULTS: Passes to reach milestone ===\n");
        fprintf(stderr, "%-12s", "BlockSize");
        for (int mi = 0; mi < n_milestones; mi++) {
            char hdr[16];
            if (milestones[mi] == 100.0f) snprintf(hdr, sizeof(hdr), "100%%");
            else snprintf(hdr, sizeof(hdr), "%.0f%%", milestones[mi]);
            fprintf(stderr, "  %-8s", hdr);
        }
        fprintf(stderr, "  %-10s  %-10s\n", "Time(ms)", "Eff tok/s");
        fprintf(stderr, "------------");
        for (int mi = 0; mi < n_milestones; mi++) fprintf(stderr, "  --------");
        fprintf(stderr, "  ----------  ----------\n");

        for (int si = 0; si < n_sweep; si++) {
            if (sweep_block_sizes[si] > n_gen) continue;
            fprintf(stderr, "%-12d", sweep_block_sizes[si]);
            for (int mi = 0; mi < n_milestones; mi++) {
                if (results[si][mi] >= 0)
                    fprintf(stderr, "  %-8d", results[si][mi]);
                else
                    fprintf(stderr, "  %-8s", "---");
            }
            fprintf(stderr, "  %-10.0f  %-10.1f\n", sweep_time[si], sweep_eff_decode[si]);
        }

        fprintf(stderr, "\n--- Ground truth ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, truth.tokens).c_str());

    } else if (mode == "confidence-trace") {
        // ── Confidence trace mode ────────────────────────────────────────
        // Wavefront refinement with per-pass confidence measurement.
        // After EVERY decode pass within each block, measures full-sequence
        // confidence. Shows how quickly the model becomes confident.
        // Run at different --block-size values to compare.

        fprintf(stderr, "\n=== STEP 4: Confidence trace (wavefront, BS=%d) ===\n", block_size);

        // ensure we have a context
        llama_context * ct_ctx = shared_ctx;
        bool owns_ct_ctx = false;
        if (!ct_ctx) {
            int max_ctx = (int)big_prompt_tokens.size() + n_gen + 16;
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = max_ctx; cp.n_batch = max_ctx; cp.no_perf = true;
            ct_ctx = llama_init_from_model(big_model, cp);
            if (!ct_ctx) { fprintf(stderr, "confidence-trace: ctx failed\n"); return 1; }
            owns_ct_ctx = true;
        }

        const int n_vocab_ct = llama_vocab_n_tokens(big_vocab);
        int n_blocks = (n_gen + block_size - 1) / block_size;

        // measure initial confidence
        confidence_result cr0 = measure_sequence_confidence(
            big_model, ct_ctx, big_prompt_tokens, candidate, truth.tokens);

        fprintf(stderr, "\n%-8s  %-6s  %-6s  %-10s  %-10s  %-10s  %-10s\n",
                "CumPass", "Block", "Pass", "SelfCon%", "MeanConf", "GTAgree%", "Changed");
        fprintf(stderr, "--------  ------  ------  ----------  ----------  ----------  ----------\n");
        fprintf(stderr, "%-8d  %-6s  %-6s  %-10.1f  %-10.4f  %-10.1f  %-10s\n",
                0, "-", "-", cr0.self_consistency_pct, cr0.mean_confidence,
                cr0.gt_agreement_pct, "-");

        int cum_passes = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < n_blocks; b++) {
            int bstart = b * block_size;
            int blen   = std::min(block_size, n_gen - bstart);
            int n_prompt_ct = (int)big_prompt_tokens.size();
            int n_left = n_prompt_ct + bstart;

            for (int pass = 1; pass <= max_passes; pass++) {
                llama_memory_clear(llama_get_memory(ct_ctx), true);

                // build full seq: prompt + candidate[0..bstart+blen]
                std::vector<llama_token> full_seq;
                full_seq.reserve(n_left + blen);
                full_seq.insert(full_seq.end(), big_prompt_tokens.begin(), big_prompt_tokens.end());
                full_seq.insert(full_seq.end(), candidate.begin(), candidate.begin() + bstart + blen);

                int n_total = n_left + blen;
                llama_batch batch = llama_batch_init(n_total, 0, 1);
                batch.n_tokens = n_total;
                for (int i = 0; i < n_total; i++) {
                    batch.token[i]     = full_seq[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = (i >= n_left - 1) ? 1 : 0;
                }

                if (llama_decode(ct_ctx, batch)) {
                    fprintf(stderr, "confidence-trace: decode failed\n");
                    llama_batch_free(batch);
                    break;
                }

                // update block tokens (standard argmax)
                int changed = 0;
                for (int j = 0; j < blen; j++) {
                    int logit_pos = n_left - 1 + j;
                    const float * logits = llama_get_logits_ith(ct_ctx, logit_pos);
                    if (!logits) continue;

                    llama_token argmax = 0;
                    float best_val = logits[0];
                    for (int v = 1; v < n_vocab_ct; v++) {
                        if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
                    }
                    if (argmax != candidate[bstart + j]) {
                        candidate[bstart + j] = argmax;
                        changed++;
                    }
                }
                llama_batch_free(batch);

                cum_passes++;

                // measure full-sequence confidence only at checkpoints
                // (full measurement is expensive — doubles the compute per pass)
                bool at_checkpoint = (changed == 0) ||
                    (cum_passes == 1) || (cum_passes == 5) || (cum_passes == 10) ||
                    (cum_passes == 15) || (cum_passes == 20) || (cum_passes == 25) ||
                    (cum_passes == 30) || (cum_passes == 40) || (cum_passes == 50) ||
                    (cum_passes % 50 == 0);

                if (at_checkpoint) {
                    confidence_result cr = measure_sequence_confidence(
                        big_model, ct_ctx, big_prompt_tokens, candidate, truth.tokens);

                    fprintf(stderr, "%-8d  %-6d  %-6d  %-10.1f  %-10.4f  %-10.1f  %-10d\n",
                            cum_passes, b, pass, cr.self_consistency_pct, cr.mean_confidence,
                            cr.gt_agreement_pct, changed);
                }

                if (changed == 0) break;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // final measurement
        confidence_result cr_final = measure_sequence_confidence(
            big_model, ct_ctx, big_prompt_tokens, candidate, truth.tokens);

        fprintf(stderr, "\n=== CONFIDENCE TRACE SUMMARY (BS=%d) ===\n", block_size);
        fprintf(stderr, "Total passes:         %d\n", cum_passes);
        fprintf(stderr, "Total time:           %.0f ms\n", total_ms);
        fprintf(stderr, "Self-consistency:     %.1f%%\n", cr_final.self_consistency_pct);
        fprintf(stderr, "Mean confidence:      %.4f\n", cr_final.mean_confidence);
        fprintf(stderr, "GT agreement:         %.1f%%\n", cr_final.gt_agreement_pct);
        fprintf(stderr, "\n--- Refined output ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, candidate).c_str());
        fprintf(stderr, "\n--- Ground truth ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, truth.tokens).c_str());

        if (owns_ct_ctx) llama_free(ct_ctx);

    } else if (mode == "soft") {
        // ── Soft-embedding mode ──────────────────────────────────────────
        // Work in continuous embedding space instead of discrete tokens.
        // Uses temperature annealing and damped updates for convergence.

        fprintf(stderr, "\n=== STEP 4: Soft-embedding refinement ===\n");
        fprintf(stderr, "Temperature: %.2f, Gamma: %.3f, Alpha: %.2f, Top-k: %d\n",
                soft_temp, soft_gamma, soft_alpha, soft_topk);

        // 1. Build embedding cache from big model
        embedding_cache emb_cache;
        emb_cache.init(big_model);
        if (emb_cache.data.empty()) {
            fprintf(stderr, "ERROR: embedding cache init failed\n");
            if (shared_ctx) llama_free(shared_ctx);
            llama_model_free(big_model);
            return 1;
        }

        const int n_embd = emb_cache.n_embd;

        // 2. Initialize soft embeddings from the draft model's tokens
        std::vector<float> soft_embd((size_t)n_gen * n_embd);
        for (int j = 0; j < n_gen; j++) {
            const float * tok_emb = emb_cache.get(candidate[j]);
            memcpy(soft_embd.data() + (size_t)j * n_embd, tok_emb, n_embd * sizeof(float));
        }

        fprintf(stderr, "Initialized %d soft embeddings from draft tokens (n_embd=%d)\n\n", n_gen, n_embd);

        // 3. Create shared context if not already created
        llama_context * soft_ctx = shared_ctx;
        bool owns_soft_ctx = false;
        if (!soft_ctx) {
            int max_ctx = (int)big_prompt_tokens.size() + n_gen + 16;
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx   = max_ctx;
            ctx_params.n_batch = max_ctx;
            ctx_params.no_perf = true;
            soft_ctx = llama_init_from_model(big_model, ctx_params);
            if (!soft_ctx) { fprintf(stderr, "soft: failed to create context\n"); return 1; }
            owns_soft_ctx = true;
        }

        // 4. Run iterative soft refinement passes with temperature annealing
        fprintf(stderr, "%-6s  %-10s  %-12s  %-10s  %-10s\n",
                "Pass", "Agree%", "MeanConf", "Temp", "Time(ms)");
        fprintf(stderr, "------  ----------  ------------  ----------  ----------\n");

        float temperature = soft_temp;
        double total_soft_ms = 0.0;

        for (int pass = 1; pass <= max_passes; pass++) {
            soft_pass_result r = run_soft_refinement_pass(
                big_model, soft_ctx, emb_cache,
                big_prompt_tokens, soft_embd, truth.tokens,
                n_gen, temperature, soft_alpha, soft_topk);

            total_soft_ms += r.elapsed_ms;

            fprintf(stderr, "%-6d  %-10.1f  %-12.4f  %-10.3f  %-10.0f\n",
                    pass, r.agreement_pct, r.mean_confidence, temperature, r.elapsed_ms);

            // anneal temperature
            temperature *= soft_gamma;
            if (temperature < 0.01f) temperature = 0.01f;

            // check for convergence (high agreement + high confidence)
            if (r.agreement_pct >= 100.0f) {
                fprintf(stderr, "\nFull convergence at pass %d!\n", pass);
                break;
            }
        }

        // 5. Final measurement: decode the final soft embeddings one more time at T→0
        //    to get clean argmax tokens
        soft_pass_result final_r = run_soft_refinement_pass(
            big_model, soft_ctx, emb_cache,
            big_prompt_tokens, soft_embd, truth.tokens,
            n_gen, 0.01f, 1.0f, 1);  // very low temp, full alpha, top-1
        total_soft_ms += final_r.elapsed_ms;

        double eff_decode = n_gen / (total_soft_ms / 1000.0);

        // Convert final soft embeddings to discrete tokens (argmax from last pass)
        // and evaluate confidence via batch.token
        std::vector<llama_token> soft_candidate(n_gen);
        {
            // Decode soft embeddings one last time to get argmax tokens
            llama_memory_clear(llama_get_memory(soft_ctx), true);
            int n_total = (int)big_prompt_tokens.size() + n_gen;
            int n_embd_s = emb_cache.n_embd;
            llama_batch eb = llama_batch_init(n_total, n_embd_s, 1);
            eb.n_tokens = n_total;
            for (int i = 0; i < (int)big_prompt_tokens.size(); i++) {
                const float * tok_emb = emb_cache.get(big_prompt_tokens[i]);
                memcpy(eb.embd + (size_t)i * n_embd_s, tok_emb, n_embd_s * sizeof(float));
                eb.pos[i] = i; eb.n_seq_id[i] = 1; eb.seq_id[i][0] = 0;
                eb.logits[i] = (i == (int)big_prompt_tokens.size() - 1) ? 1 : 0;
            }
            for (int j = 0; j < n_gen; j++) {
                memcpy(eb.embd + (size_t)((int)big_prompt_tokens.size() + j) * n_embd_s,
                       soft_embd.data() + (size_t)j * n_embd_s, n_embd_s * sizeof(float));
                eb.pos[(int)big_prompt_tokens.size() + j] = (int)big_prompt_tokens.size() + j;
                eb.n_seq_id[(int)big_prompt_tokens.size() + j] = 1;
                eb.seq_id[(int)big_prompt_tokens.size() + j][0] = 0;
                eb.logits[(int)big_prompt_tokens.size() + j] = 1;
            }
            llama_decode(soft_ctx, eb);
            const int n_v = llama_vocab_n_tokens(big_vocab);
            for (int j = 0; j < n_gen; j++) {
                int lp = (int)big_prompt_tokens.size() - 1 + j;
                const float * logits = llama_get_logits_ith(soft_ctx, lp);
                llama_token best = 0; float bv = logits[0];
                for (int v = 1; v < n_v; v++) { if (logits[v] > bv) { bv = logits[v]; best = v; } }
                soft_candidate[j] = best;
            }
            llama_batch_free(eb);
        }

        // Now evaluate confidence using batch.token (ground-truth code path)
        {
            llama_memory_clear(llama_get_memory(soft_ctx), true);
            std::vector<llama_token> full_seq;
            full_seq.insert(full_seq.end(), big_prompt_tokens.begin(), big_prompt_tokens.end());
            full_seq.insert(full_seq.end(), soft_candidate.begin(), soft_candidate.end());

            int n_total = (int)full_seq.size();
            int n_prompt_len = (int)big_prompt_tokens.size();
            llama_batch batch = llama_batch_init(n_total, 0, 1);
            batch.n_tokens = n_total;
            for (int i = 0; i < n_total; i++) {
                batch.token[i] = full_seq[i]; batch.pos[i] = i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = (i >= n_prompt_len - 1) ? 1 : 0;
            }
            llama_decode(soft_ctx, batch);

            int n_self = 0, n_gt = 0;
            double sum_c = 0.0; float min_c = 1.0f; int n_low = 0;
            const int n_v = llama_vocab_n_tokens(big_vocab);

            fprintf(stderr, "\n=== FINAL CONFIDENCE EVALUATION (via batch.token) ===\n");
            fprintf(stderr, "%-6s  %-20s  %-20s  %-10s  %-6s\n",
                    "Pos", "SoftArgmax", "TokenArgmax", "Conf", "Match");
            fprintf(stderr, "------  --------------------  --------------------  ----------  ------\n");

            for (int j = 0; j < n_gen; j++) {
                int lp = n_prompt_len - 1 + j;
                const float * logits = llama_get_logits_ith(soft_ctx, lp);
                if (!logits) continue;
                llama_token argmax = 0; float bv = logits[0];
                for (int v = 1; v < n_v; v++) { if (logits[v] > bv) { bv = logits[v]; argmax = v; } }
                float conf = confidence_from_logits(logits, n_v);
                sum_c += conf; if (conf < min_c) min_c = conf; if (conf < 0.5f) n_low++;
                bool sm = (argmax == soft_candidate[j]);
                if (sm) n_self++;
                if (j < (int)truth.tokens.size() && soft_candidate[j] == truth.tokens[j]) n_gt++;
                if (j < 10 || !sm) {
                    fprintf(stderr, "%-6d  %-20s  %-20s  %-10.4f  %s\n",
                            j, token_to_str(big_vocab, soft_candidate[j]).c_str(),
                            token_to_str(big_vocab, argmax).c_str(), conf, sm ? "yes" : "NO");
                }
            }
            fprintf(stderr, "\nSelf-consistency:     %.1f%% (%d/%d)\n",
                    (float)n_self / n_gen * 100.0f, n_self, n_gen);
            fprintf(stderr, "GT agreement:         %.1f%% (%d/%d)\n",
                    final_r.agreement_pct, n_gt, n_gen);
            fprintf(stderr, "Mean confidence:      %.4f\n", (float)(sum_c / n_gen));
            fprintf(stderr, "Min confidence:       %.4f\n", min_c);
            fprintf(stderr, "Low confidence (<0.5): %d/%d tokens\n", n_low, n_gen);

            llama_batch_free(batch);
        }

        fprintf(stderr, "\n=== SOFT REFINEMENT SUMMARY ===\n");
        fprintf(stderr, "Prompt:               %s\n", prompt.c_str());
        fprintf(stderr, "Tokens refined:       %d\n", n_gen);
        fprintf(stderr, "Initial agreement:    %.1f%%\n", initial_agreement);
        fprintf(stderr, "Final agreement:      %.1f%%\n", final_r.agreement_pct);
        fprintf(stderr, "Final confidence:     %.4f\n", final_r.mean_confidence);
        fprintf(stderr, "Total time:           %.0f ms\n", total_soft_ms);
        fprintf(stderr, "Effective decode:     %.1f tok/s\n", eff_decode);
        fprintf(stderr, "AR decode baseline:   %.1f tok/s\n", truth_tps);

        fprintf(stderr, "\n--- Soft output ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, soft_candidate).c_str());
        fprintf(stderr, "\n--- Ground truth ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, truth.tokens).c_str());

        if (owns_soft_ctx) {
            llama_free(soft_ctx);
            soft_ctx = nullptr;
        }

    } else if (mode == "soft-wavefront") {
        // ── Soft-wavefront mode ──────────────────────────────────────────
        // Block-by-block soft-embedding refinement (Gauss-Seidel scheduling).
        // Prompt + converged left blocks use discrete embeddings (stable).
        // Only the active block uses soft embeddings.

        fprintf(stderr, "\n=== STEP 4: Soft-wavefront refinement ===\n");
        fprintf(stderr, "Block size: %d, Temp: %.2f, Gamma: %.3f, Alpha: %.2f, Top-k: %d\n",
                block_size, soft_temp, soft_gamma, soft_alpha, soft_topk);

        embedding_cache emb_cache;
        emb_cache.init(big_model);
        if (emb_cache.data.empty()) {
            fprintf(stderr, "ERROR: embedding cache init failed\n");
            if (shared_ctx) llama_free(shared_ctx);
            llama_model_free(big_model);
            return 1;
        }

        // create context if not already shared
        llama_context * sw_ctx = shared_ctx;
        bool owns_sw_ctx = false;
        if (!sw_ctx) {
            int max_ctx = (int)big_prompt_tokens.size() + n_gen + 16;
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx   = max_ctx;
            ctx_params.n_batch = max_ctx;
            ctx_params.no_perf = true;
            sw_ctx = llama_init_from_model(big_model, ctx_params);
            if (!sw_ctx) { fprintf(stderr, "soft-wavefront: ctx failed\n"); return 1; }
            owns_sw_ctx = true;
        }

        int n_blocks = (n_gen + block_size - 1) / block_size;
        fprintf(stderr, "Blocks: %d (last block: %d tokens)\n\n", n_blocks,
                n_gen - (n_blocks - 1) * block_size);

        fprintf(stderr, "%-6s  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
                "Block", "Start", "Len", "InitAgr%", "FinalAgr%", "Passes", "Time(ms)");
        fprintf(stderr, "------  --------  --------  ----------  ----------  ----------  ----------\n");

        double total_sw_ms = 0.0;
        int total_sw_passes = 0;

        for (int b = 0; b < n_blocks; b++) {
            int bstart = b * block_size;
            int blen   = std::min(block_size, n_gen - bstart);

            soft_block_result br = run_soft_block_refinement(
                big_model, sw_ctx, emb_cache,
                big_prompt_tokens, candidate, truth.tokens,
                bstart, blen, max_passes,
                soft_temp, soft_gamma, soft_alpha, soft_topk);

            total_sw_ms += br.total_ms;
            total_sw_passes += br.passes_used;

            fprintf(stderr, "%-6d  %-8d  %-8d  %-10.1f  %-10.1f  %-10d  %-10.0f\n",
                    b, bstart, blen, br.initial_agreement, br.final_agreement,
                    br.passes_used, br.total_ms);
        }

        // final overall agreement
        int sw_match = 0;
        for (int j = 0; j < std::min(n_gen, (int)truth.tokens.size()); j++) {
            if (candidate[j] == truth.tokens[j]) sw_match++;
        }
        float sw_agree = (float)sw_match / std::min(n_gen, (int)truth.tokens.size()) * 100.0f;
        double sw_eff = n_gen / (total_sw_ms / 1000.0);

        // ── Final confidence evaluation ──────────────────────────────────
        // Run one more decode of the converged output and measure per-token
        // confidence. A high-confidence fixed point is valid even if it
        // differs from AR ground truth.
        {
            llama_memory_clear(llama_get_memory(sw_ctx), true);

            std::vector<llama_token> full_seq;
            full_seq.insert(full_seq.end(), big_prompt_tokens.begin(), big_prompt_tokens.end());
            full_seq.insert(full_seq.end(), candidate.begin(), candidate.end());

            int n_total = (int)full_seq.size();
            int n_prompt_len = (int)big_prompt_tokens.size();
            llama_batch batch = llama_batch_init(n_total, 0, 1);
            batch.n_tokens = n_total;
            for (int i = 0; i < n_total; i++) {
                batch.token[i]     = full_seq[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (i >= n_prompt_len - 1) ? 1 : 0;
            }

            llama_decode(sw_ctx, batch);

            int n_self_match = 0;
            int n_gt_match = 0;
            double sum_conf = 0.0;
            float min_conf = 1.0f;
            int n_low_conf = 0;  // tokens with conf < 0.5
            const int n_v = llama_vocab_n_tokens(big_vocab);

            fprintf(stderr, "\n=== FINAL CONFIDENCE EVALUATION ===\n");
            fprintf(stderr, "%-6s  %-20s  %-20s  %-10s  %-6s\n",
                    "Pos", "Converged", "ModelArgmax", "Conf", "Match");
            fprintf(stderr, "------  --------------------  --------------------  ----------  ------\n");

            for (int j = 0; j < n_gen; j++) {
                int logit_pos = n_prompt_len - 1 + j;
                const float * logits = llama_get_logits_ith(sw_ctx, logit_pos);
                if (!logits) continue;

                llama_token argmax = 0;
                float best_val = logits[0];
                for (int v = 1; v < n_v; v++) {
                    if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
                }

                float conf = confidence_from_logits(logits, n_v);
                sum_conf += conf;
                if (conf < min_conf) min_conf = conf;
                if (conf < 0.5f) n_low_conf++;

                bool self_consistent = (argmax == candidate[j]);
                if (self_consistent) n_self_match++;
                if (j < (int)truth.tokens.size() && candidate[j] == truth.tokens[j]) n_gt_match++;

                // print first 10 + any mismatches
                if (j < 10 || !self_consistent) {
                    std::string conv_str = token_to_str(big_vocab, candidate[j]);
                    std::string argmax_str = token_to_str(big_vocab, argmax);
                    fprintf(stderr, "%-6d  %-20s  %-20s  %-10.4f  %s\n",
                            j, conv_str.c_str(), argmax_str.c_str(), conf,
                            self_consistent ? "yes" : "NO");
                }
            }

            float self_consist_pct = (float)n_self_match / n_gen * 100.0f;
            float mean_conf = (float)(sum_conf / n_gen);

            fprintf(stderr, "\nSelf-consistency:     %.1f%% (%d/%d tokens match model's own argmax)\n",
                    self_consist_pct, n_self_match, n_gen);
            fprintf(stderr, "GT agreement:         %.1f%% (%d/%d)\n",
                    sw_agree, sw_match, n_gen);
            fprintf(stderr, "Mean confidence:      %.4f\n", mean_conf);
            fprintf(stderr, "Min confidence:       %.4f\n", min_conf);
            fprintf(stderr, "Low confidence (<0.5): %d/%d tokens\n", n_low_conf, n_gen);

            llama_batch_free(batch);
        }

        fprintf(stderr, "\n=== SOFT-WAVEFRONT SUMMARY ===\n");
        fprintf(stderr, "Block size:           %d\n", block_size);
        fprintf(stderr, "Tokens refined:       %d\n", n_gen);
        fprintf(stderr, "Initial agreement:    %.1f%%\n", initial_agreement);
        fprintf(stderr, "Final agreement:      %.1f%% (%d/%d)\n", sw_agree, sw_match, n_gen);
        fprintf(stderr, "Total passes:         %d\n", total_sw_passes);
        fprintf(stderr, "Total time:           %.0f ms\n", total_sw_ms);
        fprintf(stderr, "Effective decode:     %.1f tok/s\n", sw_eff);
        fprintf(stderr, "AR decode baseline:   %.1f tok/s\n", truth_tps);

        fprintf(stderr, "\n--- Refined output ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, candidate).c_str());
        fprintf(stderr, "\n--- Ground truth ---\n");
        fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, truth.tokens).c_str());

        if (owns_sw_ctx) {
            llama_free(sw_ctx);
        }

    } else {
        // ── Single-mode (jacobi or wavefront) ───────────────────────────

    fprintf(stderr, "\n=== STEP 4: Iterative refinement [%s] ===\n",
            use_wavefront ? "WAVEFRONT / Gauss-Seidel" : "JACOBI / global");
    if (use_wavefront) {
        fprintf(stderr, "Block size: %d tokens\n", block_size);
    }

    std::vector<float> agreement_curve;
    agreement_curve.push_back(initial_agreement);
    double total_refine_ms = 0.0;
    double total_prefill_tokens_all = 0.0;
    int    total_passes_all = 0;

    if (!use_wavefront) {
        // ── Jacobi mode (existing) ──────────────────────────────────────
        fprintf(stderr, "%-6s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
                "Pass", "Agree%", "Changed", "Frozen", "MeanConf", "Time(ms)", "Tok/s");
        fprintf(stderr, "------  ----------  ----------  ----------  ----------  ----------  ----------\n");

        std::vector<bool> frozen(n_gen, false);

        for (int pass = 1; pass <= max_passes; pass++) {
            pass_result r = run_refinement_pass(
                big_model, shared_ctx, big_prompt_tokens, candidate, frozen,
                truth.tokens, conf_thresh);

            agreement_curve.push_back(r.agreement_pct);
            total_refine_ms += r.elapsed_ms;
            total_prefill_tokens_all += (big_prompt_tokens.size() + n_gen);
            total_passes_all++;

            double pass_tps = (double)(big_prompt_tokens.size() + n_gen) / (r.elapsed_ms / 1000.0);
            fprintf(stderr, "%-6d  %-10.1f  %-10d  %-10d  %-10.3f  %-10.0f  %-10.0f\n",
                    pass, r.agreement_pct, r.tokens_changed, r.tokens_frozen,
                    r.mean_confidence, r.elapsed_ms, pass_tps);

            if (r.tokens_changed == 0) {
                fprintf(stderr, "\nNo tokens changed in pass %d — converged.\n", pass);
                break;
            }
        }
    } else {
        // ── Wavefront / Gauss-Seidel mode ───────────────────────────────
        // Split candidate into blocks, refine left-to-right.
        // Each block sees converged left context from all prior blocks.

        int n_blocks = (n_gen + block_size - 1) / block_size;
        fprintf(stderr, "Blocks: %d (last block: %d tokens)\n\n", n_blocks,
                n_gen - (n_blocks - 1) * block_size);

        fprintf(stderr, "%-6s  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-12s\n",
                "Block", "Start", "Len", "InitAgr%", "FinalAgr%", "Passes", "Time(ms)", "PrefillTok/s");
        fprintf(stderr, "------  --------  --------  ----------  ----------  ----------  ----------  ------------\n");

        // for KV reuse: clear memory before first block
        if (reuse_kv && shared_ctx) {
            llama_memory_clear(llama_get_memory(shared_ctx), true);
        }

        for (int b = 0; b < n_blocks; b++) {
            int bstart = b * block_size;
            int blen   = std::min(block_size, n_gen - bstart);
            bool left_ready = reuse_kv && (b > 0);

            block_result br = run_block_refinement(
                big_model, shared_ctx, big_prompt_tokens, candidate, truth.tokens,
                bstart, blen, max_passes, conf_thresh, reuse_kv, left_ready);

            total_refine_ms += br.total_ms;
            total_prefill_tokens_all += br.total_prefill_tokens;
            total_passes_all += br.passes_used;

            double eff_tps = br.total_prefill_tokens / (br.total_ms / 1000.0);

            fprintf(stderr, "%-6d  %-8d  %-8d  %-10.1f  %-10.1f  %-10d  %-10.0f  %-12.0f\n",
                    b, bstart, blen, br.initial_agreement, br.final_agreement,
                    br.passes_used, br.total_ms, eff_tps);

            // measure overall agreement after this block
            int total_match = 0;
            int total_compare = std::min(n_gen, (int)truth.tokens.size());
            for (int j = 0; j < total_compare; j++) {
                if (candidate[j] == truth.tokens[j]) total_match++;
            }
            float overall = (float)total_match / total_compare * 100.0f;
            agreement_curve.push_back(overall);
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Step 5: Summary
    // ────────────────────────────────────────────────────────────────────────

    // final overall agreement
    int final_match = 0;
    int final_compare = std::min(n_gen, (int)truth.tokens.size());
    for (int j = 0; j < final_compare; j++) {
        if (candidate[j] == truth.tokens[j]) final_match++;
    }
    float final_agree = (float)final_match / final_compare * 100.0f;

    double overall_tps = total_prefill_tokens_all / (total_refine_ms / 1000.0);

    // compute effective decode tok/s: output tokens / wall-clock refinement time
    double effective_decode_tps = n_gen / (total_refine_ms / 1000.0);

    fprintf(stderr, "\n=== CONVERGENCE SUMMARY ===\n");
    fprintf(stderr, "Mode:                 %s\n", use_wavefront ? "wavefront (Gauss-Seidel)" : "jacobi (global)");
    if (use_wavefront) fprintf(stderr, "Block size:           %d\n", block_size);
    fprintf(stderr, "Prompt:               %s\n", prompt.c_str());
    fprintf(stderr, "Tokens refined:       %d\n", n_gen);
    fprintf(stderr, "Initial agreement:    %.1f%%\n", initial_agreement);
    fprintf(stderr, "Final agreement:      %.1f%% (%d/%d)\n", final_agree, final_match, final_compare);
    fprintf(stderr, "Total passes:         %d\n", total_passes_all);
    fprintf(stderr, "Total refine time:    %.0f ms\n", total_refine_ms);
    fprintf(stderr, "Effective prefill:    %.0f tok/s (total tokens through model / time)\n", overall_tps);
    fprintf(stderr, "Effective decode:     %.1f tok/s (output tokens / refine time)\n", effective_decode_tps);
    fprintf(stderr, "AR decode baseline:   %.1f tok/s (big model, greedy)\n", truth_tps);
    fprintf(stderr, "Speedup vs AR:        %.2fx\n", effective_decode_tps / truth_tps);

    fprintf(stderr, "\nAgreement curve: ");
    for (size_t i = 0; i < agreement_curve.size(); i++) {
        fprintf(stderr, "%.1f%%", agreement_curve[i]);
        if (i + 1 < agreement_curve.size()) fprintf(stderr, " -> ");
    }
    fprintf(stderr, "\n");

    // final refined text
    fprintf(stderr, "\n--- Refined output ---\n");
    fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, candidate).c_str());
    fprintf(stderr, "\n--- Ground truth ---\n");
    fprintf(stderr, ">>> %s\n", tokens_to_str(big_vocab, truth.tokens).c_str());

    // ── Final confidence evaluation ──────────────────────────────────
    {
        llama_context * eval_ctx = shared_ctx;
        bool owns_eval = false;
        if (!eval_ctx) {
            int max_ctx = (int)big_prompt_tokens.size() + n_gen + 16;
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = max_ctx; cp.n_batch = max_ctx; cp.no_perf = true;
            eval_ctx = llama_init_from_model(big_model, cp);
            owns_eval = true;
        } else {
            llama_memory_clear(llama_get_memory(eval_ctx), true);
        }

        std::vector<llama_token> full_seq;
        full_seq.insert(full_seq.end(), big_prompt_tokens.begin(), big_prompt_tokens.end());
        full_seq.insert(full_seq.end(), candidate.begin(), candidate.end());

        int n_total = (int)full_seq.size();
        int n_prompt_len = (int)big_prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_total, 0, 1);
        batch.n_tokens = n_total;
        for (int i = 0; i < n_total; i++) {
            batch.token[i]     = full_seq[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i >= n_prompt_len - 1) ? 1 : 0;
        }

        llama_decode(eval_ctx, batch);

        int n_self_match = 0;
        int n_gt_match = 0;
        double sum_conf = 0.0;
        float min_conf = 1.0f;
        int n_low_conf = 0;
        const int n_v = llama_vocab_n_tokens(big_vocab);

        fprintf(stderr, "\n=== FINAL CONFIDENCE EVALUATION ===\n");
        fprintf(stderr, "%-6s  %-20s  %-20s  %-10s  %-6s\n",
                "Pos", "Converged", "ModelArgmax", "Conf", "Match");
        fprintf(stderr, "------  --------------------  --------------------  ----------  ------\n");

        for (int j = 0; j < n_gen; j++) {
            int logit_pos = n_prompt_len - 1 + j;
            const float * logits = llama_get_logits_ith(eval_ctx, logit_pos);
            if (!logits) continue;

            llama_token argmax = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_v; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; argmax = v; }
            }

            float conf = confidence_from_logits(logits, n_v);
            sum_conf += conf;
            if (conf < min_conf) min_conf = conf;
            if (conf < 0.5f) n_low_conf++;

            bool self_consistent = (argmax == candidate[j]);
            if (self_consistent) n_self_match++;
            if (j < (int)truth.tokens.size() && candidate[j] == truth.tokens[j]) n_gt_match++;

            if (j < 10 || !self_consistent) {
                std::string conv_str = token_to_str(big_vocab, candidate[j]);
                std::string argmax_str = token_to_str(big_vocab, argmax);
                fprintf(stderr, "%-6d  %-20s  %-20s  %-10.4f  %s\n",
                        j, conv_str.c_str(), argmax_str.c_str(), conf,
                        self_consistent ? "yes" : "NO");
            }
        }

        float self_consist_pct = (float)n_self_match / n_gen * 100.0f;
        float mean_conf_eval = (float)(sum_conf / n_gen);

        fprintf(stderr, "\nSelf-consistency:     %.1f%% (%d/%d tokens match model's own argmax)\n",
                self_consist_pct, n_self_match, n_gen);
        fprintf(stderr, "GT agreement:         %.1f%% (%d/%d)\n", final_agree, final_match, final_compare);
        fprintf(stderr, "Mean confidence:      %.4f\n", mean_conf_eval);
        fprintf(stderr, "Min confidence:       %.4f\n", min_conf);
        fprintf(stderr, "Low confidence (<0.5): %d/%d tokens\n", n_low_conf, n_gen);

        llama_batch_free(batch);
        if (owns_eval) llama_free(eval_ctx);
    }

    // check success criteria
    if (final_agree >= 85.0f) {
        fprintf(stderr, "\n*** STRONG/MODERATE SUCCESS: %.1f%% agreement reached ***\n", final_agree);
    } else if (final_agree >= 70.0f) {
        fprintf(stderr, "\n*** PARTIAL SUCCESS: %.1f%% agreement (below 85%% target) ***\n", final_agree);
    } else {
        fprintf(stderr, "\n*** LOW CONVERGENCE: %.1f%% — needs investigation ***\n", final_agree);
    }

    } // end single-mode

    if (shared_ctx) {
        llama_free(shared_ctx);
    }
    llama_model_free(big_model);

    return 0;
}
