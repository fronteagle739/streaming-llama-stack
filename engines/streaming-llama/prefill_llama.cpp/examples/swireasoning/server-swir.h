#pragma once
// SwiReasoning (explicit<->latent) decode support for the server.
// Shares the math in examples/swireasoning/swireasoning.cpp;
// --swir-selftest checks the embedding round trip.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Per-request parameters (mirrors the harness CLI flags).
struct swir_params {
    bool    enabled      = false;
    float   alpha_0      = 0.5f;   // soft-embedding interpolation toward <think>
    float   beta_0       = 0.7f;   // explicit-embedding interpolation toward </think>
    int32_t window_size  = 256;    // min normal-mode steps before allowing a switch to soft
    int32_t max_switches = 2;      // soft->normal switches before convergence injection
};

// Per-slot runtime state. Reset each request.
struct swir_runtime {
    int32_t mode            = 0;       // 0 = soft (latent), 1 = normal (explicit); starts soft
    int32_t mode_stay_steps = 0;
    float   ref_entropy     = 0.0f;
    int32_t switch_count    = 0;
    bool    locked_normal   = false;
    bool    first_step      = true;
    bool    last_to_soft    = false;   // transition fired this step
    bool    last_to_normal  = false;

    std::vector<llama_token> inject_queue;   // forced convergence/termination tokens
    int32_t answer_budget = -1;              // >=0: counting down to stop after termination

    std::vector<float> last_emb;             // embedding to feed next step
    bool pending = false;                    // last_emb is ready to feed

    void reset() { *this = swir_runtime(); }

    // entropy of the (untempered) next-token distribution
    float compute_entropy(const float * probs, int32_t n_vocab) const {
        float entropy = 0.0f;
        for (int32_t i = 0; i < n_vocab; i++) {
            float p = std::max(probs[i], 1e-12f);
            entropy -= p * std::logf(p);
        }
        return entropy;
    }

    // entropy-trend switch state machine (ref generation_utils.py:317-331)
    void update_mode(float cur_entropy, int32_t window_size, int32_t max_switches) {
        last_to_soft = last_to_normal = false;
        if (first_step) {
            ref_entropy = cur_entropy;
            first_step = false;
            return;
        }
        mode_stay_steps++;
        const bool allow_switch = (mode_stay_steps >= window_size);
        const bool to_normal = (mode == 0) && (cur_entropy < ref_entropy);
        const bool to_soft   = (mode == 1) && (cur_entropy > ref_entropy) && allow_switch && !locked_normal;
        if (to_normal) {
            mode = 1; last_to_normal = true;
            if (max_switches > 0) switch_count++;
        }
        if (to_soft) {
            mode = 0; last_to_soft = true;
        }
        if (to_normal || to_soft) {
            mode_stay_steps = 0;
            ref_entropy = cur_entropy;
        }
    }

    bool should_use_soft() const { return mode == 0; }
};

// Model-level, read-only, shared across slots: the token-embedding matrix E (host float,
// row-major [n_vocab][n_embd]) + cached <think>/</think>/newline embeddings. Populated once
// by swir_load_model_data() (defined in server-context.cpp, which has the internal model header).
struct swir_model_data {
    int32_t n_vocab = 0;
    int32_t n_embd  = 0;
    std::vector<float> E;
    std::vector<float> emb_think, emb_endthink, emb_nl;
    std::vector<llama_token> convergence_ids;  // "</think>"
    std::vector<llama_token> termination_ids;  // "</think>\n\nThe final answer is"
    std::vector<llama_token> endthink_ids;     // tokens of "</think>" (for explicit-lock detection)
    bool has_specials = false;
    bool loaded       = false;

    // soft_emb = probs_original @ E  (full vocab, prune p<1e-9 for speed; bit-exact-validated)
    void compute_soft_embedding(const float * probs, float * out) const {
        std::memset(out, 0, (size_t) n_embd * sizeof(float));
        for (int32_t j = 0; j < n_vocab; j++) {
            const float p = probs[j];
            if (p < 1e-9f) continue;
            const float * e = &E[(size_t) j * n_embd];
            for (int32_t i = 0; i < n_embd; i++) out[i] += p * e[i];
        }
    }
};

// Fill rt.last_emb for the NEXT step from this step's distribution + sampled token, with the
// alpha/beta interpolation toward <think>/</think>/newline (ref generation_utils.py:341-351).
inline void swir_build_last_emb(swir_runtime & rt, const swir_model_data & m,
                                const swir_params & p, const float * probs,
                                llama_token tok, int step, int n_predict) {
    rt.last_emb.resize(m.n_embd);
    const float prog = (float) step / (float) std::max(1, n_predict);
    const bool is_soft    = rt.should_use_soft() && !rt.locked_normal;
    const bool beta_blend = m.has_specials && rt.last_to_normal && step > 0 && !is_soft;

    std::vector<float> soft;
    if (is_soft || beta_blend) {
        soft.resize(m.n_embd);
        m.compute_soft_embedding(probs, soft.data());
        if (m.has_specials) {
            if (step == 0) {
                for (int i = 0; i < m.n_embd; i++) soft[i] = 0.9f * soft[i] + 0.1f * m.emb_nl[i];
            } else if (rt.last_to_soft) {
                const float a = p.alpha_0 + (1.f - p.alpha_0) * prog;
                for (int i = 0; i < m.n_embd; i++) soft[i] = a * soft[i] + (1.f - a) * m.emb_think[i];
            }
        }
    }

    if (is_soft) {
        std::memcpy(rt.last_emb.data(), soft.data(), (size_t) m.n_embd * sizeof(float));
    } else if (beta_blend) {
        const float b = p.beta_0 + (1.f - p.beta_0) * prog;
        for (int i = 0; i < m.n_embd; i++) rt.last_emb[i] = b * soft[i] + (1.f - b) * m.emb_endthink[i];
    } else {
        std::memcpy(rt.last_emb.data(), &m.E[(size_t) tok * m.n_embd], (size_t) m.n_embd * sizeof(float));
    }
    rt.pending = true;
}

// NaN/Inf guard for a freshly built embedding.
inline bool swir_emb_ok(const std::vector<float> & e) {
    for (float v : e) { if (v != v || std::fabs(v) > 1e6f) return false; }
    return true;
}

// True if `id` is (part of) the </think> token sequence — locks the slot to explicit mode.
inline bool swir_is_endthink(const swir_model_data & m, llama_token id) {
    for (llama_token t : m.endthink_ids) { if (t == id) return true; }
    return false;
}

// Overthinking control: on a soft->normal switch at the cap, queue convergence/termination
// tokens to inject (ref generation_utils.py:353-384).
inline void swir_inject_triggers(swir_runtime & rt, const swir_model_data & m,
                                 const swir_params & p, int step) {
    if (p.max_switches <= 0 || step <= 0 || !rt.last_to_normal) return;
    if (rt.switch_count > 2 * p.max_switches) {
        rt.inject_queue = m.termination_ids;
        rt.answer_budget = 32;
    } else if (rt.switch_count >= p.max_switches && rt.answer_budget < 0) {
        rt.inject_queue = m.convergence_ids;
    }
}
