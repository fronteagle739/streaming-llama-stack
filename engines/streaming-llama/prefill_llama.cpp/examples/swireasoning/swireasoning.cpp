// SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs
// Training-free inference framework that dynamically switches between
// explicit (token-based) and latent (soft embedding) reasoning modes.
//
// Based on: https://github.com/sdc17/SwiReasoning (ICLR 2026)
//
// Usage:
//   ./llama-swir -m model.gguf -n 256 -p "prompt"
//   ./llama-swir -m model.gguf -n 256 -p "prompt" --swir-soft-embd
//   ./llama-swir -m model.gguf -n 256 -p "prompt" --swir-benchmark
//   ./llama-swir -m model.gguf -n 256 -p "prompt" --swir-alpha 1.0 --swir-beta 0.7

#include "llama.h"
#include "../../src/llama-model.h"  // for internal model access

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// SwiReasoning parameters
struct swir_params {
    float   alpha_0      = 1.0f;    // soft embedding interpolation
    float   beta_0       = 0.7f;    // end-thinking interpolation
    int32_t window_size  = 512;     // min steps before mode switch
    int32_t max_switches = -1;      // max soft->normal switches (-1 = unlimited)
    bool    verbose      = false;   // print mode switches
    bool    soft_embd    = true;    // soft embedding injection (default on; --swir-off for vanilla)
    bool    selftest     = false;   // run embedding round-trip self-test
};

// SwiReasoning state
struct swir_state {
    int32_t   mode            = 0;  // 0 = soft (latent), 1 = normal (explicit); ref starts soft
    int32_t   mode_stay_steps = 0;
    float     ref_entropy     = 0.0f;
    int32_t   switch_count    = 0;
    bool      locked_normal   = false;
    bool      first_step      = true;
    bool      last_to_soft    = false;  // transition fired this step (for alpha/beta in A3)
    bool      last_to_normal  = false;

    int32_t n_embd  = 0;
    int32_t n_vocab = 0;
    bool    verbose = false;

    // token embeddings matrix E: [n_vocab * n_embd], row-major
    std::vector<float> token_embeddings;
    bool    embeddings_loaded = false;

    // special-token embeddings for alpha/beta interpolation (A3)
    std::vector<float> emb_think, emb_endthink, emb_nl;
    bool has_specials = false;

    // overthinking control (A4)
    std::vector<llama_token> inject_queue;
    int32_t answer_budget = -1;

    // entropy of the (untempered) next-token distribution
    float compute_entropy(const float * probs, int32_t n_vocab) {
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
        bool allow_switch = (mode_stay_steps >= window_size);
        bool to_normal = (mode == 0) && (cur_entropy < ref_entropy);
        bool to_soft   = (mode == 1) && (cur_entropy > ref_entropy) && allow_switch && !locked_normal;
        if (to_normal) {
            mode = 1; last_to_normal = true;
            if (max_switches > 0) {
                switch_count++;
                if (verbose) fprintf(stderr, "\nswir: soft->normal (count=%d/%d)\n", switch_count, max_switches);
            }
        }
        if (to_soft) {
            mode = 0; last_to_soft = true;
            if (verbose) fprintf(stderr, "\nswir: normal->soft\n");
        }
        if (to_normal || to_soft) {
            mode_stay_steps = 0;
            ref_entropy = cur_entropy;
        }
    }

    bool should_use_soft() const { return mode == 0; }

    // Cache <think>/</think>/newline embeddings for alpha/beta interpolation (ref L248-253).
    void load_specials(const llama_vocab * vocab) {
        auto row = [&](const char * s, std::vector<float> & dst) -> bool {
            llama_token ids[16];
            int n = llama_tokenize(vocab, s, (int) std::strlen(s), ids, 16, false, true);
            if (n <= 0) return false;
            llama_token id = ids[n - 1];  // last piece
            if (id < 0 || id >= n_vocab) return false;
            dst.assign(token_embeddings.begin() + (size_t) id * n_embd,
                       token_embeddings.begin() + (size_t) (id + 1) * n_embd);
            return true;
        };
        has_specials = row("<think>", emb_think) & row("</think>", emb_endthink) & row("\n", emb_nl);
        if (verbose) fprintf(stderr, "swir: specials %s\n", has_specials ? "loaded" : "MISSING (alpha/beta off)");
    }

    // soft_emb = probs_original @ E  (full vocab, no top-K, no normalization) — ref L339.
    // tokens with negligible prob are pruned (mass error < n_vocab*1e-9 ~ 2.5e-4) for speed.
    void compute_soft_embedding(const float * probs, int32_t n_vocab, int32_t n_embd, float * out) {
        std::memset(out, 0, n_embd * sizeof(float));
        for (int32_t j = 0; j < n_vocab; j++) {
            const float p = probs[j];
            if (p < 1e-9f) continue;
            const float * e = &token_embeddings[(size_t) j * n_embd];
            for (int32_t i = 0; i < n_embd; i++) out[i] += p * e[i];
        }
    }
};

// Load token_embd.weight as a host float matrix E, row-major [n_vocab][n_embd].
// ggml stores it ne=[n_embd, n_vocab] => each vocab row is n_embd contiguous values,
// so after row-wise dequant the layout is ALREADY [n_vocab][n_embd]. No transpose, no fallback.
static bool load_token_embeddings(llama_model * model, int32_t n_vocab, int32_t n_embd,
                                  std::vector<float> & E, bool verbose) {
    const struct llama_model * m = model;
    const ggml_tensor * t = m->get_tensor("token_embd.weight");
    if (!t) { fprintf(stderr, "swir: token_embd.weight not found\n"); return false; }
    if (verbose) {
        fprintf(stderr, "swir: token_embd.weight type=%s ne=[%lld, %lld]\n",
                ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1]);
    }
    GGML_ASSERT(t->ne[0] == n_embd && t->ne[1] == n_vocab);

    // copy (possibly GPU-resident, quantized) bytes to host
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    E.resize((size_t) n_vocab * n_embd);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(E.data(), raw.data(), E.size() * sizeof(float));
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            fprintf(stderr, "swir: no to_float for %s\n", ggml_type_name(t->type));
            return false;
        }
        const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
        for (int64_t row = 0; row < n_vocab; row++) {
            tr->to_float(raw.data() + row * row_bytes, E.data() + (size_t) row * n_embd, n_embd);
        }
    }
    if (verbose) fprintf(stderr, "swir: loaded E [%d x %d]\n", n_vocab, n_embd);
    return true;
}

// Prove embedding injection is faithful: logits from decoding token t must match
// logits from decoding E[t] as an embd batch at the same position.
static int run_selftest(llama_context * ctx, const llama_vocab * vocab,
                        const std::vector<float> & E, int32_t n_vocab, int32_t n_embd) {
    (void) vocab;
    const llama_token test_tokens[] = {10, 100, 1000, 5000};
    const float eps = 2e-2f;
    int failures = 0;
    for (llama_token t : test_tokens) {
        if (t >= n_vocab) continue;
        // path A: decode the token id
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_token tt = t;
        llama_batch tb = llama_batch_get_one(&tt, 1);
        if (llama_decode(ctx, tb)) { fprintf(stderr, "selftest: token decode failed\n"); return 1; }
        const float * lt = llama_get_logits(ctx);
        std::vector<float> L_tok(lt, lt + n_vocab);
        // path B: decode E[t] as an embedding at the same position
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch eb = llama_batch_init(1, n_embd, 1);
        eb.n_tokens = 1;
        std::memcpy(eb.embd, &E[(size_t) t * n_embd], n_embd * sizeof(float));
        eb.pos[0] = 0; eb.n_seq_id[0] = 1; eb.seq_id[0][0] = 0; eb.logits[0] = 1;
        int rc = llama_decode(ctx, eb);
        std::vector<float> L_emb;
        if (!rc) { const float * le = llama_get_logits(ctx); L_emb.assign(le, le + n_vocab); }
        llama_batch_free(eb);
        if (rc) { fprintf(stderr, "selftest: tok %d embd decode rc=%d\n", t, rc); failures++; continue; }
        float max_abs = 0.f;
        for (int j = 0; j < n_vocab; j++) max_abs = std::max(max_abs, std::fabs(L_tok[j] - L_emb[j]));
        fprintf(stderr, "selftest: tok %5d  max|dLogit|=%.5f  %s\n", t, max_abs, max_abs < eps ? "PASS" : "FAIL");
        if (max_abs >= eps) failures++;
    }
    fprintf(stderr, "selftest: %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

static void print_usage(int argc, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-p prompt] [--swir-off] [--swir-selftest]\n", argv[0]);
    printf("\n    %s -m model.gguf -n 256 -p \"prompt\" --swir-alpha 1.0 --swir-beta 0.7 --swir-window 512\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello, my name is";
    int n_predict = 256;
    int ngl = 99;
    uint32_t seed = 1234;   // fixed by default for reproducibility (--seed N)
    swir_params swir;

    // parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0) {
                n_predict = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "-p") == 0) {
                prompt = argv[++i];
            } else if (strcmp(argv[i], "-ngl") == 0) {
                ngl = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "--swir-alpha") == 0) {
                swir.alpha_0 = std::stof(argv[++i]);
            } else if (strcmp(argv[i], "--swir-beta") == 0) {
                swir.beta_0 = std::stof(argv[++i]);
            } else if (strcmp(argv[i], "--swir-window") == 0) {
                swir.window_size = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "--swir-max-switches") == 0) {
                swir.max_switches = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "--swir-verbose") == 0) {
                swir.verbose = true;
            } else if (strcmp(argv[i], "--swir-soft-embd") == 0) {
                swir.soft_embd = true;
            } else if (strcmp(argv[i], "--swir-off") == 0) {
                swir.soft_embd = false;
            } else if (strcmp(argv[i], "--swir-selftest") == 0) {
                swir.selftest = true;
            } else if (strcmp(argv[i], "--seed") == 0) {
                seed = (uint32_t) std::stoul(argv[++i]);
            } else {
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
    }

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t n_embd  = llama_model_n_embd(model);

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_predict + 1024;
    ctx_params.n_batch = n_predict;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // sampling: temp 0.6 + top_p + top_k, fixed-seed dist (greedy degenerates — see port plan)
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.6f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    // init SwiReasoning state
    swir_state state;
    state.n_embd  = n_embd;
    state.n_vocab = n_vocab;
    state.verbose = swir.verbose;

    // embedding round-trip self-test: prove embd injection is faithful, then exit
    if (swir.selftest) {
        if (!load_token_embeddings(model, n_vocab, n_embd, state.token_embeddings, swir.verbose)) {
            llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model); return 1;
        }
        int rc = run_selftest(ctx, vocab, state.token_embeddings, n_vocab, n_embd);
        llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
        return rc;
    }

    // load token embeddings if soft embedding mode is enabled
    if (swir.soft_embd) {
        fprintf(stderr, "swir: loading token embeddings for soft injection (%d tokens, %d dim)...\n",
                n_vocab, n_embd);

        auto t_start = ggml_time_us();
        load_token_embeddings(model, n_vocab, n_embd, state.token_embeddings, swir.verbose);
        state.embeddings_loaded = true;
        state.load_specials(vocab);   // cache <think>/</think>/newline for alpha/beta
        auto t_end = ggml_time_us();

        fprintf(stderr, "swir: embeddings loaded in %.2f s\n",
                (t_end - t_start) / 1000000.0f);
    }

    // tokenize the prompt
    int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

    // print prompt
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string s(buf, n);
            printf("%s", s.c_str());
        }
    }

    // prepare batch for prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // encode prompt
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "error: failed to encode prompt\n");
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // pre-tokenize </think> for the explicit-lock check (no leading space, parse specials)
    llama_token endthink_ids[8];
    int n_endthink = llama_tokenize(vocab, "</think>", 8, endthink_ids, 8, false, true);

    // overthinking control: token sequences injected at switch-count thresholds (ref L264-265)
    auto tok_seq = [&](const char * s) {
        llama_token ids[64];
        int n = llama_tokenize(vocab, s, (int) std::strlen(s), ids, 64, false, true);
        return std::vector<llama_token>(ids, ids + std::max(0, n));
    };
    const std::vector<llama_token> convergence_ids = tok_seq("</think>");
    const std::vector<llama_token> termination_ids = tok_seq("</think>\n\nThe final answer is");

    // main generation loop with SwiReasoning
    const int prompt_len = (int) prompt_tokens.size();
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    int n_soft_steps = 0;
    int n_normal_steps = 0;

    std::vector<float> probs(n_vocab), last_emb(n_embd), soft_emb(n_embd);
    for (int n_pos = prompt_len; n_pos < prompt_len + n_predict; n_pos++) {
        const float * logits = llama_get_logits(ctx);

        // probs_original = softmax(logits)  (untempered: drives entropy + soft_emb)
        float mx = -1e30f;
        for (int32_t j = 0; j < n_vocab; j++) mx = std::max(mx, logits[j]);
        float sum = 0.f;
        for (int32_t j = 0; j < n_vocab; j++) { probs[j] = std::expf(logits[j] - mx); sum += probs[j]; }
        const float inv = 1.f / sum;
        for (int32_t j = 0; j < n_vocab; j++) probs[j] *= inv;

        const int step = n_pos - prompt_len;

        // sample, then override with any queued (injected) convergence/termination token
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (!state.inject_queue.empty()) {
            tok = state.inject_queue.front();
            state.inject_queue.erase(state.inject_queue.begin());
        }
        for (int j = 0; j < n_endthink; j++) if (tok == endthink_ids[j]) state.locked_normal = true;

        // entropy-trend switch (after lock so this token's </think> suppresses to_soft)
        const float entropy = state.compute_entropy(probs.data(), n_vocab);
        state.update_mode(entropy, swir.window_size, swir.max_switches);

        // on a soft->normal switch, inject convergence/termination to curb overthinking (ref L353-384)
        if (swir.max_switches > 0 && step > 0 && state.last_to_normal) {
            if (state.switch_count > 2 * swir.max_switches) {
                state.inject_queue = termination_ids;
                state.answer_budget = 32;
            } else if (state.switch_count >= swir.max_switches && state.answer_budget < 0) {
                state.inject_queue = convergence_ids;
            }
        }

        char buf[256];
        int bn = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (bn > 0) { fwrite(buf, 1, (size_t) bn, stdout); fflush(stdout); }
        if (llama_vocab_is_eog(vocab, tok)) break;

        // answer-budget countdown after a termination injection
        if (state.answer_budget >= 0 && --state.answer_budget == 0) break;

        // pure-vanilla path (--swir-off): feed the sampled token id
        if (!state.embeddings_loaded) {
            llama_batch tb = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, tb)) { fprintf(stderr, "\nswir: vanilla decode failed at pos %d\n", n_pos); break; }
            n_normal_steps++; n_decode++;
            continue;
        }

        // swir path: feed an embedding every step (soft = probs@E, else E[tok]),
        // with alpha/beta interpolation toward <think>/</think>/newline (ref L341-351).
        const float prog = (float) step / (float) std::max(1, n_predict);
        const bool  is_soft    = state.should_use_soft() && !state.locked_normal;
        const bool  beta_blend = state.has_specials && state.last_to_normal && step > 0 && !is_soft;
        if (is_soft || beta_blend) {
            state.compute_soft_embedding(probs.data(), n_vocab, n_embd, soft_emb.data());
            if (state.has_specials) {
                if (step == 0) {
                    for (int i = 0; i < n_embd; i++) soft_emb[i] = 0.9f * soft_emb[i] + 0.1f * state.emb_nl[i];
                } else if (state.last_to_soft) {
                    const float a = swir.alpha_0 + (1.f - swir.alpha_0) * prog;
                    for (int i = 0; i < n_embd; i++) soft_emb[i] = a * soft_emb[i] + (1.f - a) * state.emb_think[i];
                }
            }
        }
        if (is_soft) {
            std::memcpy(last_emb.data(), soft_emb.data(), n_embd * sizeof(float));
            n_soft_steps++;
        } else if (beta_blend) {
            const float b = swir.beta_0 + (1.f - swir.beta_0) * prog;
            for (int i = 0; i < n_embd; i++) last_emb[i] = b * soft_emb[i] + (1.f - b) * state.emb_endthink[i];
            n_normal_steps++;
        } else {
            std::memcpy(last_emb.data(), &state.token_embeddings[(size_t) tok * n_embd], n_embd * sizeof(float));
            n_normal_steps++;
        }

        // NaN/Inf guard
        bool bad = false;
        for (int i = 0; i < n_embd; i++) { float v = last_emb[i]; if (v != v || std::fabs(v) > 1e6f) { bad = true; break; } }
        if (bad) { fprintf(stderr, "\nswir: bad embedding at step %d, stopping\n", n_decode); break; }

        llama_batch eb = llama_batch_init(1, n_embd, 1);
        eb.n_tokens = 1;
        std::memcpy(eb.embd, last_emb.data(), n_embd * sizeof(float));
        eb.pos[0] = n_pos; eb.n_seq_id[0] = 1; eb.seq_id[0][0] = 0; eb.logits[0] = 1;
        int rc = llama_decode(ctx, eb);
        llama_batch_free(eb);
        if (rc) { fprintf(stderr, "\nswir: embd decode rc=%d at pos %d\n", rc, n_pos); break; }

        n_decode++;
    }

    printf("\n\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "swir: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            n_decode, (t_main_end - t_main_start) / 1000000.0f,
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));
    fprintf(stderr, "swir: final mode = %s\n", state.mode == 0 ? "soft (latent)" : "normal (explicit)");
    fprintf(stderr, "swir: soft steps = %d, normal steps = %d\n", n_soft_steps, n_normal_steps);
    if (swir.max_switches > 0) {
        fprintf(stderr, "swir: switches = %d/%d\n", state.switch_count, swir.max_switches);
    }

    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
