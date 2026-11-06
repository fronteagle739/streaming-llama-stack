#include "diffusion-core.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// Entropy-bound denoiser (PROVEN). Bytes identical to the original in diffusion-cli.cpp; relocated here
// so llama-diffusion-cli and llama-diffusion-server share one copy. See diffusion-core.h for the contract.
void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    llama_model * model   = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t C       = params.max_length - n_input;          // canvas length

    // Env-tunable sampler config (benchmark/experiment knobs; UNSET => the proven defaults, bit-identical).
    // LLAMA_DG_ENTROPY_BOUND / _STABILITY / _CONFIDENCE / _TMIN / _TMAX / _STEPS override the struct fields
    // so configs can be swept without a rebuild. LLAMA_DG_LOCK_STABLE (K>0) enables SureLock-style position
    // locking: once a canvas position's argmax holds for K steps it is frozen (never renoised) — a guided/
    // decoder upgrade that speeds convergence and stops renoise from churning already-converged tokens.
    auto env_f = [](const char * k, float d) { const char * v = getenv(k); return (v && *v) ? (float) atof(v) : d; };
    auto env_i = [](const char * k, int   d) { const char * v = getenv(k); return (v && *v) ? atoi(v) : d; };
    const float   eb_bound       = env_f("LLAMA_DG_ENTROPY_BOUND", params.entropy_bound);
    const int32_t stab           = env_i("LLAMA_DG_STABILITY",     params.stability_threshold);
    const float   conf           = env_f("LLAMA_DG_CONFIDENCE",    params.confidence_threshold);
    const float   tmin           = env_f("LLAMA_DG_TMIN",          params.t_min);
    const float   tmax           = env_f("LLAMA_DG_TMAX",          params.t_max);
    const int32_t lock_threshold = env_i("LLAMA_DG_LOCK_STABLE",   0);
    // Re-denoise depth (only when resuming from a prior draft): the fraction of canvas positions to
    // RE-NOISE (random-replace) before refining; the rest keep the draft token (refine-in-place). 1.0 =
    // re-noise everything (≈ from scratch); ~0.2-0.4 = light refinement of the draft (the guided-loop use).
    const float   renoise_depth  = std::min(1.0f, std::max(0.0f, env_f("LLAMA_DG_RENOISE_DEPTH", 1.0f)));
    const bool    resume         = params.resume_from_canvas;
    const int32_t S       = std::max(1, env_i("LLAMA_DG_STEPS", params.max_denoising_steps));
    // Per-position re-noise SCOPE (the wavefront critic's surgical fix): LLAMA_DG_RENOISE_POSITIONS="i,j,k"
    // re-noises ONLY those canvas positions on a resume (the critic-flagged span), keeping every other token
    // EXACT — so a span correction doesn't disturb the rest of the block. Empty => the renoise_depth fraction.
    std::vector<char> renoise_pos;
    if (const char * rp = getenv("LLAMA_DG_RENOISE_POSITIONS"); resume && rp && *rp) {
        const int32_t C0 = params.max_length - n_input;
        renoise_pos.assign(C0, 0);
        const char * p = rp;
        while (*p) {
            char * e = nullptr; long v = strtol(p, &e, 10);
            if (e == p) { break; }
            if (v >= 0 && v < C0) { renoise_pos[(size_t) v] = 1; }
            p = (*e) ? e + 1 : e;
        }
    }
    // PKV (opt-in, LLAMA_DG_PKV=1): prefill the prompt's per-layer K/V ONCE, then each denoise step decodes a
    // CANVAS-ONLY batch (the cached prompt K/V is concatenated in-graph). Amortises the prompt re-attention +
    // (256-aligned KV) enables flash attention. Canvas logits then start at row 0, not n_input.
    const bool pkv = (env_i("LLAMA_DG_PKV", 0) != 0) && n_input > 0 && C > 0;
    int32_t logit_off = pkv ? 0 : n_input;                        // canvas rows: row 0 (pkv) or n_input (unified)

    // Guided diffusion token bias (negative token/span masks
    // + per-token guidance weights). LLAMA_DG_BAN_TOKENS="id,id,..." hard-bans those vocab ids (never
    // argmax/sampled anywhere on the canvas); LLAMA_DG_BIAS="id:delta,id:delta,..." adds delta to a token's
    // post-temperature logit (positive=prefer, negative=soft-dislike). Applied to EVERY canvas position —
    // a steering signal a critic (or the user) shapes the draft with. nullptr bias => zero overhead.
    std::vector<float> logit_bias;
    int32_t            n_biased = 0;
    auto ensure_bias = [&]() { if (logit_bias.empty()) { logit_bias.assign(n_vocab, 0.0f); } };
    if (const char * ban = getenv("LLAMA_DG_BAN_TOKENS"); ban && *ban) {
        ensure_bias();
        const char * p = ban;
        while (*p) {
            char * end = nullptr;
            long id = strtol(p, &end, 10);
            if (end == p) { break; }
            if (id >= 0 && id < n_vocab) { logit_bias[id] = -1e30f; n_biased++; }
            p = (*end) ? end + 1 : end;                            // skip the separator
        }
    }
    if (const char * bz = getenv("LLAMA_DG_BIAS"); bz && *bz) {
        ensure_bias();
        const char * p = bz;
        while (*p) {
            char * end = nullptr;
            long id = strtol(p, &end, 10);
            if (end == p) { break; }
            float delta = 0.0f;
            if (*end == ':') { delta = strtof(end + 1, &end); }
            if (id >= 0 && id < n_vocab) { logit_bias[id] += delta; n_biased++; }
            while (*end && *end != ',' && *end != ';' && *end != ' ') { end++; }   // to next separator
            p = (*end) ? end + 1 : end;
        }
    }
    const float * bias = logit_bias.empty() ? nullptr : logit_bias.data();

    // Per-position guided diffusion map: the global LLAMA_DG_BIAS/BAN above steer the
    // WHOLE canvas; a critic instead flags SPECIFIC canvas positions. LLAMA_DG_GUIDANCE_FILE points at a text
    // file, one directive per line (# comments / blank lines ignored):
    //   "<pos> pin  <tok>"          freeze canvas position <pos> to token <tok> (always accepted, never renoised)
    //   "<pos> ban  <tok>"          hard-ban <tok> at <pos> ONLY (never argmax/sampled there; fine elsewhere)
    //   "<pos> bias <tok> <delta>"  add <delta> to <tok>'s post-temperature logit at <pos> ONLY
    // This is a sparse critic-to-canvas guidance map: only flagged positions get a
    // dense [n_vocab] bias row (global bias + per-pos deltas) built ONCE here; every other position uses `bias`.
    struct dg_pos_guide { bool pinned = false; llama_token pin_tok = -1; std::vector<std::pair<int32_t,float>> deltas; };
    std::vector<dg_pos_guide>       pos_guide;       // size C, or empty when no guidance file
    std::vector<std::vector<float>> pos_bias_rows;   // size C; row[pos] non-empty => use instead of `bias`
    int32_t n_pos_biased = 0, n_pinned = 0;
    if (const char * gf = getenv("LLAMA_DG_GUIDANCE_FILE"); gf && *gf) {
        FILE * f = fopen(gf, "r");
        if (f) {
            pos_guide.assign(C, dg_pos_guide{});
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                char * s = line; while (*s == ' ' || *s == '\t') { s++; }
                if (*s == '#' || *s == '\n' || *s == '\r' || *s == '\0') { continue; }
                char    op[16] = {0};
                int32_t pos = -1; long tok = -1; float delta = 0.0f;
                const int nf = sscanf(s, "%d %15s %ld %f", &pos, op, &tok, &delta);
                if (nf < 3 || pos < 0 || pos >= C || tok < 0 || tok >= n_vocab) { continue; }
                if (strcmp(op, "pin") == 0) {
                    pos_guide[pos].pinned = true; pos_guide[pos].pin_tok = (llama_token) tok; n_pinned++;
                } else if (strcmp(op, "ban") == 0) {
                    pos_guide[pos].deltas.emplace_back((int32_t) tok, -1e30f);
                } else if (strcmp(op, "bias") == 0 && nf >= 4) {
                    pos_guide[pos].deltas.emplace_back((int32_t) tok, delta);
                }
            }
            fclose(f);
            pos_bias_rows.assign(C, std::vector<float>{});
            for (int32_t pos = 0; pos < C; pos++) {
                if (pos_guide[pos].deltas.empty()) { continue; }
                std::vector<float> row(n_vocab, 0.0f);
                if (bias) { std::copy(bias, bias + n_vocab, row.begin()); }
                for (const auto & d : pos_guide[pos].deltas) {
                    if (d.first >= 0 && d.first < n_vocab) { row[d.first] += d.second; }
                }
                pos_bias_rows[pos] = std::move(row);
                n_pos_biased++;
            }
            LOG_INF("%s: LLAMA_DG_GUIDANCE_FILE -> %d positions biased, %d pinned\n", __func__, n_pos_biased, n_pinned);
        } else {
            LOG_WRN("%s: LLAMA_DG_GUIDANCE_FILE could not open '%s'\n", __func__, gf);
        }
    }
    // Per-position guidance disables the device-sample fast path (it has no per-position bias term), so the
    // host worker (which honors pos_bias_rows) runs — same rule as the global `bias` below.
    const bool has_pos_guide = (n_pos_biased > 0 || n_pinned > 0);

    // Top-k candidate dump for guided critique: LLAMA_DG_DUMP_TOPK_FILE=path
    // writes, per FINAL-step canvas position, the diffusion's OWN top-K (token_id:logprob) candidates — so a
    // big-model critic can SCORE those exact ids (no re-tokenization mismatch) and soft-bias AMONG them.
    // LLAMA_DG_TOPK sets K (default 8). Needs the host worker (it has the per-position logits row), so it
    // forces gpu_sample off below — same rule as bias/guidance.
    const char *  dump_topk_file = getenv("LLAMA_DG_DUMP_TOPK_FILE");
    const bool    dump_topk      = dump_topk_file && *dump_topk_file;
    const int32_t topk_k         = std::max(1, std::min(64, env_i("LLAMA_DG_TOPK", 8)));
    std::vector<std::vector<std::pair<int32_t, float>>> topk_canvas;  // [C][<=K] (id, logprob) of the LAST step
    if (dump_topk) { topk_canvas.assign(C, {}); }

    // device-side sampling (PR #24423): per-position argmax/entropy/multinomial read straight from the device
    // SC buffer (dg_sc_logits_gpu = this step's post-softcap canvas logits), skipping the host full-canvas
    // vocab scan (~48ms/step on the 5090). DEFAULT ON (validated 1.58x/step, coherence held); needs SC (the
    // device buffer source) and no token bias (the kernel has no bias term). The kernel declines (-> host
    // worker, byte-identical argmax) on non-CUDA / multi-GPU / no SC, so default-on is safe on any backend.
    // Force off with LLAMA_DG_GPU_SAMPLE=0 (e.g. to A/B against the host path).
    const bool gpu_sample = (env_i("LLAMA_DG_GPU_SAMPLE", 1) != 0) && params.self_conditioning
                            && bias == nullptr && !has_pos_guide && !dump_topk;

    LOG_INF("%s: eb=%.3f stab=%d conf=%.4f temp=[%.2f,%.2f] steps=%d lock=%d sc=%d bias_tokens=%d "
            "resume=%d renoise=%.2f\n", __func__, eb_bound, stab, conf, tmin, tmax, S, lock_threshold,
            params.self_conditioning ? 1 : 0, n_biased, resume ? 1 : 0, resume ? renoise_depth : 1.0f);

    llama_set_causal_attn(ctx, false);
    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<llama_token> current_canvas(C);                   // working (renoised) canvas, fed to the forward
    std::vector<llama_token> argmax_canvas(C, 0);                 // model's best prediction = the output
    std::vector<llama_token> prev_argmax(C, -1);                 // stability history (-1 -> step 0 unstable)
    std::vector<int32_t>     pos_held(C, 0);                     // per-position argmax-stable step count (locking)
    for (int32_t i = 0; i < C; i++) {
        if (resume) {
            // resume: seed from the prior draft (output_tokens canvas region, caller-provided). Re-noise the
            // flagged SCOPE (LLAMA_DG_RENOISE_POSITIONS) or, if none, a renoise_depth fraction; keep the rest
            // as the draft token (refine-in-place). Kept positions start as their own argmax → immediately
            // stable/lockable (guidance steers the rest).
            const llama_token seed = output_tokens[n_input + i];
            const bool rn = renoise_pos.empty() ? (uni01(rng) < renoise_depth) : (renoise_pos[(size_t) i] != 0);
            current_canvas[i] = rn ? vocab_dist(rng) : seed;
            argmax_canvas[i]  = seed;
            prev_argmax[i]    = seed;
        } else {
            current_canvas[i] = vocab_dist(rng);                  // random init (not mask) — default path
        }
    }
    std::vector<llama_token> denoiser(C);                        // per-position multinomial sample
    std::vector<llama_token> renoise(C);                         // fresh random tokens for rejected positions
    std::vector<float>       entropy(C);
    std::vector<float>       u(C);                               // pre-drawn multinomial draws (determinism)
    std::vector<int32_t>     order(C);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    // self-conditioning = softmax(previous step's canvas logits / previous t), gated off on the first step.
    std::vector<float> sc_buffer;
    if (params.self_conditioning) {
        sc_buffer.assign((size_t) C * n_vocab, 0.0f);
    }

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);

    float prev_temp_inv = 1.0f;
    int   held          = 0;
    bool  finished      = false;
    int32_t steps_run   = 0;
    const auto t_start  = std::chrono::steady_clock::now();
    // per-phase timing probe (LLAMA_DG_TIMING=1): separates the forward (decode) from the sampling so the
    // long-context falloff can be attributed. Adds a synchronize after decode (perturbs timing) — diagnostic only.
    const bool dg_timing = env_i("LLAMA_DG_TIMING", 0) != 0;
    double t_decode_ms = 0.0, t_sample_ms = 0.0;
    auto ms_since = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    if (pkv) {
        // PKV PREFILL: cache the prompt's per-layer K/V once (set_phase PREFILL → the forward ggml_cpy's each
        // layer's prompt K/V into the store), then switch to DECODE for the per-step canvas decodes.
        llama_diffusion_set_phase(model, /*PREFILL*/ 1, n_input);
        llama_batch pb = llama_batch_init(n_input, 0, 1);
        pb.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            pb.token[i] = input_tokens[i]; pb.pos[i] = i;
            pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0; pb.logits[i] = (i == n_input - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, pb) != 0) { LOG_ERR("%s: PKV prefill failed\n", __func__); }
        llama_batch_free(pb);
        llama_diffusion_set_phase(model, /*DECODE*/ 2, n_input);
    }

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        steps_run++;
        const int32_t step_idx  = S - cur_step;                  // 0-based
        const float   t         = tmin + (tmax - tmin) * ((float) cur_step / (float) S);
        const float   temp_inv  = 1.0f / t;

        if (params.self_conditioning) {
            llama_diffusion_set_sc(model, sc_buffer.data(),
                                   step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
        }

        if (pkv) {
            // CANVAS-ONLY decode batch: the cached prompt K/V is concatenated in-graph. Canvas positions are
            // n_input + i so RoPE aligns with the cached prompt (positions 0..n_input).
            batch.n_tokens = C;
            for (int32_t i = 0; i < C; i++) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_input + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        } else {
            batch.n_tokens = params.max_length;
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = (i < n_input) ? input_tokens[i] : current_canvas[i - n_input];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        }

        auto t_dec0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode at step %d\n", __func__, step_idx);
            break;
        }
        if (dg_timing) { llama_synchronize(ctx); t_decode_ms += ms_since(t_dec0); }
        auto t_smp0 = std::chrono::steady_clock::now();
        // pre-draw the step's randomness single-threaded so the output is seed-reproducible (both sampling
        // paths consume the SAME u — the device kernel takes it verbatim, so greedy decode is bit-identical
        // and stochastic draws match the host modulo FP reduction order).
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // Sample per canvas position: argmax, entropy of softmax(raw/t), one multinomial draw. The device path
        // (PR #24423) reads this step's canvas logits on-GPU from the SC buffer and returns the small
        // per-position arrays — no 268 MB host vocab scan. On decline/disabled we fall back to the host worker
        // (also the only path that honors a token bias and refreshes the host SC stash).
        bool sampled_on_device = false;
        if (gpu_sample) {
            llama_synchronize(ctx);                              // land the in-graph SC cpy into dg_sc_logits_gpu
            sampled_on_device = llama_diffusion_device_sample(model, u.data(), argmax_canvas.data(),
                                                              entropy.data(), denoiser.data(), C, temp_inv);
        }
        if (!sampled_on_device) {
            const float * logits = llama_get_logits(ctx);        // canvas rows packed: [max_length, n_vocab]

            // stash this step's canvas logits for the next step's self-conditioning (canvas rows at logit_off).
            // The device path doesn't need it (dg_sc_logits_gpu is the SC source); kept here for host parity.
            if (params.self_conditioning) {
                std::memcpy(sc_buffer.data(), logits + (size_t) logit_off * n_vocab,
                            (size_t) C * n_vocab * sizeof(float));
            }

            // per position: argmax, entropy of softmax(raw/t), and a multinomial sample
            auto worker = [&](int32_t p0, int32_t p1) {
                for (int32_t pos = p0; pos < p1; pos++) {
                    // per-position guidance row (global bias + this position's deltas) if flagged, else the
                    // global `bias`. pos_bias_rows[pos] is empty for unflagged positions.
                    const float * pbias = (!pos_bias_rows.empty() && !pos_bias_rows[pos].empty())
                                          ? pos_bias_rows[pos].data() : bias;
                    const float * row = logits + (size_t) (logit_off + pos) * n_vocab;
                    float m = -INFINITY; int32_t amax = 0;
                    for (int32_t v = 0; v < n_vocab; v++) {
                        const float z = row[v] * temp_inv + (pbias ? pbias[v] : 0.0f);
                        if (z > m) { m = z; amax = v; }
                    }
                    float Z = 0.0f;
                    for (int32_t v = 0; v < n_vocab; v++) {
                        Z += expf(row[v] * temp_inv + (pbias ? pbias[v] : 0.0f) - m);
                    }
                    const float target = u[pos] * Z;
                    float   cum = 0.0f, H = 0.0f;
                    int32_t sampled = n_vocab - 1; bool picked = false;
                    for (int32_t v = 0; v < n_vocab; v++) {
                        const float e = expf(row[v] * temp_inv + (pbias ? pbias[v] : 0.0f) - m);
                        const float p = e / Z;
                        if (p > 0.0f) { H -= p * logf(p); }
                        cum += e;
                        if (!picked && cum >= target) { sampled = v; picked = true; }
                    }
                    entropy[pos]       = H;
                    argmax_canvas[pos] = amax;
                    denoiser[pos]      = sampled;
                    if (dump_topk) {
                        // record this position's top-K candidates (by post-bias/temp logit) as (id, logprob).
                        // K-bounded insertion sort, descending — cheap for small K over the row we already scan.
                        const float logZ = logf(Z);
                        auto & tk = topk_canvas[pos]; tk.clear();
                        for (int32_t v = 0; v < n_vocab; v++) {
                            const float z = row[v] * temp_inv + (pbias ? pbias[v] : 0.0f);
                            if ((int32_t) tk.size() < topk_k) {
                                tk.emplace_back(v, z);
                                for (int32_t j = (int32_t) tk.size() - 1; j > 0 && tk[j].second > tk[j - 1].second; --j) {
                                    std::swap(tk[j], tk[j - 1]);
                                }
                            } else if (z > tk.back().second) {
                                tk.back() = { v, z };
                                for (int32_t j = topk_k - 1; j > 0 && tk[j].second > tk[j - 1].second; --j) {
                                    std::swap(tk[j], tk[j - 1]);
                                }
                            }
                        }
                        for (auto & pr : tk) { pr.second = pr.second - m - logZ; }  // logit -> logprob
                    }
                }
            };
            {
                std::vector<std::thread> pool;
                const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
                for (unsigned ti = 0; ti < nth; ti++) {
                    const int32_t p0 = (int32_t) ti * chunk;
                    const int32_t p1 = std::min(p0 + chunk, C);
                    if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
                }
                for (auto & th : pool) { th.join(); }
            }
        }

        if (dg_timing) t_sample_ms += ms_since(t_smp0);

        // accept the lowest-entropy positions within the MI bound (sum of strictly-earlier entropies <= bound)
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= eb_bound) { accepted[pos] = 1; }
        }
        // A pinned (critic-approved) position is ALWAYS accepted — the MI bound never renoises it.
        if (n_pinned > 0) {
            for (int32_t pos = 0; pos < C; pos++) { if (pos_guide[pos].pinned) { accepted[pos] = 1; } }
        }

        // renoise: accepted -> sampled token, rest -> fresh random; the output canvas is the argmax.
        // SureLock guidance (lock_threshold>0): freeze a position to its argmax once that argmax has held
        // for lock_threshold consecutive steps, so renoise never disturbs a converged token.
        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            pos_held[pos] = (argmax_canvas[pos] == prev_argmax[pos]) ? pos_held[pos] + 1 : 0;
            // A pinned position is frozen to the critic-approved token — never renoised, and
            // the OUTPUT at that position IS the pin token (so the surrounding canvas conditions on it).
            const bool pinned = n_pinned > 0 && pos_guide[pos].pinned;
            if (pinned) {
                current_canvas[pos]          = pos_guide[pos].pin_tok;
                output_tokens[n_input + pos] = pos_guide[pos].pin_tok;
                entropy_sum += entropy[pos];
                continue;
            }
            const bool locked = lock_threshold > 0 && pos_held[pos] >= lock_threshold;
            if (locked) {
                current_canvas[pos] = argmax_canvas[pos];
            } else {
                current_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
            }
            output_tokens[n_input + pos] = argmax_canvas[pos];
            entropy_sum += entropy[pos];
        }

        // adaptive stop: argmax stable for stability_threshold steps AND confident (low mean entropy)
        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < conf;
        if (held >= stab && confident) { finished = true; }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        if (params.step_callback &&
            !params.step_callback(step_idx, S, output_tokens, params.max_length, params.step_callback_user_data)) {
            break;
        }
    }

    if (pkv) {
        llama_diffusion_set_phase(model, /*UNIFIED*/ 0, 0);       // restore the default phase for the next gen
    }
    llama_batch_free(batch);
    n_generated = params.max_length;

    // Dump the FINAL-step per-position top-K candidates (id:logprob, comma-sep, one line/position) so a
    // big-model critic can score the diffusion's OWN candidates (no re-tokenization) and soft-bias among
    // them (top-k guided critique). Positions align 1:1 with the canvas / LLAMA_DG_DUMP_FILE ids.
    if (dump_topk && !topk_canvas.empty()) {
        if (FILE * tf = fopen(dump_topk_file, "w")) {
            for (int32_t pos = 0; pos < C; pos++) {
                for (size_t k = 0; k < topk_canvas[pos].size(); k++) {
                    fprintf(tf, "%s%d:%.5f", k ? "," : "", topk_canvas[pos][k].first, topk_canvas[pos][k].second);
                }
                fputc('\n', tf);
            }
            fclose(tf);
            LOG_INF("%s: LLAMA_DG_DUMP_TOPK_FILE -> %d positions x top-%d\n", __func__, C, topk_k);
        } else {
            LOG_WRN("%s: LLAMA_DG_DUMP_TOPK_FILE could not open '%s'\n", __func__, dump_topk_file);
        }
    }

    const auto   t_end   = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // steps_run/S = adaptive-stop efficiency; ms/step = per-forward cost. THE diffusion speed metrics.
    LOG_INF("%s: GEN steps_run=%d/%d wall=%.0fms (%.1fms/step) canvas=%d\n", __func__,
            steps_run, S, wall_ms, steps_run > 0 ? wall_ms / steps_run : 0.0, C);
    if (dg_timing && steps_run > 0) {
        LOG_INF("%s: TIMING decode=%.1fms/step sample=%.1fms/step (other=%.1fms/step) pkv=%d gpu_sample=%d\n",
                __func__, t_decode_ms / steps_run, t_sample_ms / steps_run,
                (wall_ms - t_decode_ms - t_sample_ms) / steps_run, pkv ? 1 : 0, gpu_sample ? 1 : 0);
    }
}
