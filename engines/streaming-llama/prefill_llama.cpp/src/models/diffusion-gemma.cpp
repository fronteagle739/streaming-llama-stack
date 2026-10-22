#include "models.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// DiffusionGemma: block text-diffusion MoE on a Gemma-4 backbone. A single no-cache bidirectional forward
// over [prompt | canvas] reproduces the reference two-pass (causal encoder prefill + bidirectional decoder
// denoise) result when self-conditioning is off. Three things are region-aware, split at
// P = n_tokens - canvas_length (canvas = the last canvas_length positions):
//   1. input embeddings: prompt = embed*sqrt(n_embd); canvas = rmsnorm_noscale(same)   [zero-SC]
//   2. per-layer scalar:  prompt = encoder scalar (enc_out_scale); canvas = decoder scalar (out_scale)
//   3. attention mask:    prompt queries causal (SWA-clipped in sliding layers); canvas queries bidirectional
// The Gemma-4 backbone (attn + dense-MLP + 128-expert MoE) is identical to gemma4.cpp (proven on gemma4-A4B).
// Milestone A: zero self-conditioning (the exactness forward). Milestone B adds the SC soft-embedding.

// Region-aware additive mask for the unified [prompt | canvas] forward. Prompt queries are causal
// (SWA-clipped in sliding layers); canvas queries are bidirectional. Canvas->prompt reach: global
// layers see all prompt, sliding layers only the last (n_swa-1) prompt positions. Square [n_tokens,n_tokens].
class llm_graph_input_attn_diffusion : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion(const llama_hparams & hparams, const llama_cparams & cparams,
                                   int64_t n_prompt) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt) {}
    ~llm_graph_input_attn_diffusion() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;
        const int64_t P        = n_prompt;

        // swa clips keys outside the sliding window, but only for prompt (causal) queries.
        const auto fill = [&](float * data, bool swa) {
            std::fill(data, data + n_tokens * n_tokens, -INFINITY);
            for (int64_t q = 0; q < n_tokens; ++q) {
                const bool q_is_canvas = q >= P;
                const uint64_t row = q * n_tokens;
                // canvas->prompt sliding bound: last (n_swa-1) prompt positions (<= 0 for short prompts)
                const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;
                for (int64_t k = 0; k < n_tokens; ++k) {
                    const bool k_is_canvas = k >= P;
                    bool allow;
                    if (q_is_canvas) {
                        if (swa) {
                            allow = k_is_canvas || (k >= canvas_prompt_lo); // sliding: last (n_swa-1) prompt + all canvas
                        } else {
                            allow = true;                                   // global: all prompt + canvas
                        }
                    } else {
                        allow = (!k_is_canvas) && (k <= q);                 // prompt query: causal over earlier prompt
                    }
                    if (allow && swa && !q_is_canvas &&
                        llama_hparams::is_masked_swa(hparams.n_swa, hparams.swa_type, k, q)) {
                        allow = false;
                    }
                    if (allow) {
                        data[row + k] = 0.0f;
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        fill((float *) self_kq_mask->data, false);

        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            fill((float *) self_kq_mask_swa->data, true);
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    int64_t n_prompt;
};

// PKV DECODE-phase mask (prompt-KV caching): the C canvas queries (positions P..P+C) attend
// [cached prompt (first P keys) | fresh canvas (last C keys)] — rectangular [P+C, C]. Global layers see all
// prompt; SWA layers see the last (n_swa-1) prompt + all canvas. F16 when flash_attn (so FA-512/256 engages),
// else F32. (PR #24423-style; used only when llama_diffusion_set_phase selects DG_PKV_DECODE.)
class llm_graph_input_attn_diffusion_decode : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion_decode(const llama_hparams & hparams, const llama_cparams & cparams,
                                          int64_t n_prompt, int64_t n_canvas) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt), n_canvas(n_canvas) {}
    ~llm_graph_input_attn_diffusion_decode() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        const int64_t P = n_prompt, C = n_canvas, n_kv = P + C;
        const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;
        auto allow_key = [&](int64_t k, bool swa) -> bool {
            return (k < P) ? (swa ? (k >= canvas_prompt_lo) : true) : true; // bidirectional over the canvas
        };
        auto fill_f32 = [&](float * data, bool swa) {
            std::fill(data, data + n_kv * C, -INFINITY);
            for (int64_t q = 0; q < C; ++q) { const uint64_t row = q * n_kv;
                for (int64_t k = 0; k < n_kv; ++k) if (allow_key(k, swa)) data[row + k] = 0.0f; }
        };
        auto fill_f16 = [&](ggml_fp16_t * data, bool swa) {
            const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY), zero = ggml_fp32_to_fp16(0.0f);
            std::fill(data, data + n_kv * C, ninf);
            for (int64_t q = 0; q < C; ++q) { const uint64_t row = q * n_kv;
                for (int64_t k = 0; k < n_kv; ++k) if (allow_key(k, swa)) data[row + k] = zero; }
        };
        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) fill_f16((ggml_fp16_t *) self_kq_mask->data, false);
        else                                      fill_f32((float *) self_kq_mask->data, false);
        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            if (self_kq_mask_swa->type == GGML_TYPE_F16) fill_f16((ggml_fp16_t *) self_kq_mask_swa->data, true);
            else                                          fill_f32((float *) self_kq_mask_swa->data, true);
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    int64_t n_prompt, n_canvas;
};

// Self-conditioning input: uploads the previous denoise step's raw canvas logits [n_vocab, C] (host F32),
// which the graph turns into a soft token embedding fed back into the canvas embedding.
class llm_graph_input_sc : public llm_graph_input_i {
public:
    llm_graph_input_sc(const float * src, int64_t n_vocab, int64_t C) :
        src(src), n_vocab(n_vocab), C(C) {}
    ~llm_graph_input_sc() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        if (sc_logits && src) {
            GGML_ASSERT(ggml_nelements(sc_logits) == n_vocab * C);
            ggml_backend_tensor_set(sc_logits, src, 0, (size_t) n_vocab * C * sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    ggml_tensor * sc_logits = nullptr;
    const float * src;
    int64_t       n_vocab;
    int64_t       C;
};

static constexpr const char * DG_MULTIGPU_LAYER_SPLIT_MARKER = "DG_MULTIGPU_LAYER_SPLIT_2026_06_16";

static bool dg_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void dg_trace_alloc(const char * what, const char * where) {
    if (dg_env_enabled("LLAMA_DG_ALLOC_TRACE")) {
        std::fprintf(stderr, "diffusion-gemma: %s %s %s\n", DG_MULTIGPU_LAYER_SPLIT_MARKER, what, where);
    }
}

static bool dg_layer_split_active(const llama_model & m) {
    return m.split_mode() == LLAMA_SPLIT_MODE_LAYER && m.n_devices() > 1;
}

static ggml_backend_buffer_type_t dg_buft_for_dev(ggml_backend_dev_t dev) {
    return dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
}

static ggml_backend_dev_t dg_sc_device(const llama_model & m) {
    return dg_layer_split_active(m) ? m.dev_output() : m.dev_layer(0);
}

static void dg_free_pkv_store(const llama_model & m) {
    if (m.dg_pkv_buf) {
        ggml_backend_buffer_free(m.dg_pkv_buf);
        m.dg_pkv_buf = nullptr;
    }
    if (m.dg_pkv_ctx) {
        ggml_free(m.dg_pkv_ctx);
        m.dg_pkv_ctx = nullptr;
    }
    for (auto * buf : m.dg_pkv_bufs) {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
    }
    for (auto * ctx : m.dg_pkv_ctxs) {
        if (ctx) {
            ggml_free(ctx);
        }
    }
    m.dg_pkv_bufs.clear();
    m.dg_pkv_ctxs.clear();
    m.dg_pkv_k.clear();
    m.dg_pkv_v.clear();
    m.dg_pkv_cap = 0;
}

static void dg_new_pkv_layer_tensors(const llama_model & m, ggml_context * ctx, int il, int64_t cap) {
    const int64_t hd  = (m.hparams.is_swa(il) && m.hparams.n_embd_head_k_swa > 0)
                        ? m.hparams.n_embd_head_k_swa : m.hparams.n_embd_head_k;
    const int64_t nkv = m.hparams.n_head_kv(il);
    m.dg_pkv_k[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nkv, cap);
    m.dg_pkv_v[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nkv, cap);
    ggml_format_name(m.dg_pkv_k[il], "dg_pkv_k_l%d", il);
    ggml_format_name(m.dg_pkv_v[il], "dg_pkv_v_l%d", il);
}

struct dg_pkv_buft_group {
    ggml_backend_buffer_type_t buft = nullptr;
    std::vector<int> layers;
};

// Build the SC soft-embedding weight once: tok_embd dequantized + transposed to [n_vocab, n_embd] F16 in a
// device weights buffer, so the per-step SC matmul runs on-device instead of dequantizing a weight per step.
static void dg_ensure_sc_embT(const llama_model & m) {
    if (m.dg_sc_embT != nullptr) {
        return;
    }
    ggml_tensor * src = m.tok_embd;
    GGML_ASSERT(src != nullptr);
    const int64_t n_embd  = src->ne[0];
    const int64_t n_vocab = src->ne[1];

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    m.dg_sc_embT_ctx = ggml_init(ip);
    GGML_ASSERT(m.dg_sc_embT_ctx != nullptr);
    m.dg_sc_embT = ggml_new_tensor_2d(m.dg_sc_embT_ctx, GGML_TYPE_F16, n_vocab, n_embd);
    ggml_set_name(m.dg_sc_embT, "dg_sc_embT");

    ggml_backend_dev_t dev = dg_sc_device(m);
    ggml_backend_buffer_type_t buft = dg_buft_for_dev(dev);
    m.dg_sc_embT_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.dg_sc_embT_ctx, buft);
    GGML_ASSERT(m.dg_sc_embT_buf != nullptr);
    ggml_backend_buffer_set_usage(m.dg_sc_embT_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    dg_trace_alloc("dg_sc_embT", dev ? ggml_backend_dev_name(dev) : "CPU");

    // dequantize each row on the host and scatter the transpose into F16
    const ggml_type st       = src->type;
    const size_t    row_size = ggml_row_size(st, n_embd);
    std::vector<char> host_src((size_t) row_size * n_vocab);
    ggml_backend_tensor_get(src, host_src.data(), 0, host_src.size());

    std::vector<ggml_fp16_t> dstT((size_t) n_vocab * n_embd);
    const ggml_type_traits * tr = ggml_get_type_traits(st);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));
    auto worker = [&](int64_t v0, int64_t v1) {
        std::vector<float> tmp(n_embd);
        for (int64_t v = v0; v < v1; ++v) {
            const char * row = host_src.data() + (size_t) v * row_size;
            if (st == GGML_TYPE_F32) {
                std::memcpy(tmp.data(), row, (size_t) n_embd * sizeof(float));
            } else {
                tr->to_float(row, tmp.data(), n_embd);
            }
            for (int64_t e = 0; e < n_embd; ++e) {
                dstT[(size_t) e * n_vocab + v] = ggml_fp32_to_fp16(tmp[e]);
            }
        }
    };
    std::vector<std::thread> pool;
    const int64_t chunk = (n_vocab + nth - 1) / nth;
    for (unsigned t = 0; t < nth; ++t) {
        const int64_t v0 = (int64_t) t * chunk;
        const int64_t v1 = std::min(v0 + chunk, n_vocab);
        if (v0 < v1) {
            pool.emplace_back(worker, v0, v1);
        }
    }
    for (auto & th : pool) {
        th.join();
    }

    ggml_backend_tensor_set(m.dg_sc_embT, dstT.data(), 0, dstT.size() * sizeof(ggml_fp16_t));
}

// Allocate the GPU-resident SC logits buffer once: [n_vocab, canvas] F32. Layer split places it on the
// output/lm_head device; non-layer-split keeps the legacy layer-0 placement. The graph
// VIEWS this (no host ggml_set_input), and llama_diffusion_set_sc uploads the prev-step logits into it — so
// the SC softmax/embed run on-GPU instead of a 262 MB host round-trip + CPU split (PERF_ROOTCAUSE). C is
// clamped to canvas_length in dg_canvas_embed, so [n_vocab, canvas] always covers the live view.
static void dg_ensure_sc_logits_gpu(const llama_model & m, int64_t n_vocab, int64_t canvas) {
    if (m.dg_sc_logits_gpu != nullptr) {
        return;
    }
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
    m.dg_sc_logits_ctx = ggml_init(ip);
    GGML_ASSERT(m.dg_sc_logits_ctx != nullptr);
    m.dg_sc_logits_gpu = ggml_new_tensor_2d(m.dg_sc_logits_ctx, GGML_TYPE_F32, n_vocab, canvas);
    ggml_set_name(m.dg_sc_logits_gpu, "dg_sc_logits_gpu");

    ggml_backend_dev_t dev = dg_sc_device(m);
    ggml_backend_buffer_type_t buft = dg_buft_for_dev(dev);
    m.dg_sc_logits_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.dg_sc_logits_ctx, buft);
    GGML_ASSERT(m.dg_sc_logits_buf != nullptr);
    ggml_backend_buffer_set_usage(m.dg_sc_logits_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    dg_trace_alloc("dg_sc_logits_gpu", dev ? ggml_backend_dev_name(dev) : "CPU");
    // zero-init: step 0 reads this BEFORE any cpy-back has run; the dg_sc_use=0 gate zeroes the SC signal, but
    // softmax(uninitialised garbage) could be NaN and NaN*0 = NaN → corrupt the canvas. Zeros are safe.
    ggml_backend_buffer_clear(m.dg_sc_logits_buf, 0);
}

// Lazily (re)allocate the device-resident F32 prompt-KV store (per-layer K,V, grow-only) for a prompt of
// length P. Layer split groups layers by their actual buffer type; non-layer-split keeps the legacy single
// layer-0 buffer. PKV_PREFILL writes it; PKV_DECODE reads the first P. (The
// PROVEN no-cache UNIFIED path never calls this — PKV is opt-in.)
static void dg_ensure_pkv_store(const llama_model & m, int64_t P) {
    const bool layer_split = dg_layer_split_active(m);
    const bool has_store = layer_split ? !m.dg_pkv_bufs.empty() : (m.dg_pkv_buf != nullptr);
    if (has_store && m.dg_pkv_cap >= P) {
        return;
    }
    dg_free_pkv_store(m);

    const int     n_layer = (int) m.hparams.n_layer;
    const int64_t cap     = P;
    m.dg_pkv_k.resize(n_layer);
    m.dg_pkv_v.resize(n_layer);

    if (layer_split) {
        std::vector<dg_pkv_buft_group> groups;
        for (int il = 0; il < n_layer; ++il) {
            ggml_backend_buffer_type_t buft = dg_buft_for_dev(m.dev_layer(il));
            auto it = std::find_if(groups.begin(), groups.end(), [&](const dg_pkv_buft_group & g) {
                return g.buft == buft;
            });
            if (it == groups.end()) {
                groups.push_back({ buft, {} });
                it = groups.end() - 1;
            }
            it->layers.push_back(il);
        }

        for (const dg_pkv_buft_group & group : groups) {
            ggml_init_params ip = {
                ggml_tensor_overhead() * (size_t) (2 * group.layers.size() + 4), nullptr, /*.no_alloc=*/ true
            };
            ggml_context * ctx = ggml_init(ip);
            GGML_ASSERT(ctx != nullptr);
            for (int il : group.layers) {
                dg_new_pkv_layer_tensors(m, ctx, il, cap);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, group.buft);
            GGML_ASSERT(buf != nullptr);
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            m.dg_pkv_ctxs.push_back(ctx);
            m.dg_pkv_bufs.push_back(buf);
            dg_trace_alloc("dg_pkv_group", ggml_backend_buft_name(group.buft));
        }
    } else {
        ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (2 * n_layer + 4), nullptr, /*.no_alloc=*/ true };
        m.dg_pkv_ctx = ggml_init(ip);
        GGML_ASSERT(m.dg_pkv_ctx != nullptr);
        for (int il = 0; il < n_layer; ++il) {
            dg_new_pkv_layer_tensors(m, m.dg_pkv_ctx, il, cap);
        }
        ggml_backend_dev_t dev = m.dev_layer(0);
        ggml_backend_buffer_type_t buft = dg_buft_for_dev(dev);
        m.dg_pkv_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.dg_pkv_ctx, buft);
        GGML_ASSERT(m.dg_pkv_buf != nullptr);
        ggml_backend_buffer_set_usage(m.dg_pkv_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    m.dg_pkv_cap = cap;
}

llm_build_diffusion_gemma::llm_build_diffusion_gemma(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model) {
    ggml_tensor * cur;

    const int64_t canvas_length = (int64_t) hparams.diffusion_canvas_length;
    // PKV phases (opt-in via llama_diffusion_set_phase; UNIFIED = the proven no-cache path, default):
    //   PREFILL: the whole batch is prompt (P=n_tokens, C=0) — cache its K/V for the block's decode steps.
    //   DECODE:  the whole batch is the fresh canvas (C=n_tokens); P = the cached prompt length (NOT in the
    //            batch) — each layer concats the cached prompt K/V before attending.
    //   UNIFIED: region split P|C (prompt = first n_tokens-canvas_length, canvas = last canvas_length).
    const int dg_phase = (int) model.dg_pkv_phase;
    int64_t P, C;
    if (dg_phase == llama_model::DG_PKV_PREFILL) {
        P = n_tokens; C = 0;
    } else if (dg_phase == llama_model::DG_PKV_DECODE) {
        P = model.dg_pkv_P; C = n_tokens;
    } else {
        P = (canvas_length > 0 && n_tokens > canvas_length) ? (n_tokens - canvas_length) : 0;
        C = n_tokens - P;
    }

    // input embeddings: prompt = embed*sqrt(n_embd); canvas = rms_norm_noscale(embed*sqrt(n_embd) [+SC])
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_scale(ctx0, inpL, sqrtf((float) n_embd)); // embed_scale = sqrt(n_embd) (ScaledWordEmbedding)
    cb(inpL, "inp_scaled", -1);

    // Canvas input embedding = rms_norm_noscale(embed*sqrt(n_embd) [+ self-conditioning]). With SC enabled
    // (the reference scheme), the previous step's canvas logits are turned into a soft token embedding and
    // added before the norm; the {0,1} gate (dg_sc_use) zeros it on the first step. Zero-SC -> plain norm.
    auto dg_canvas_embed = [&](ggml_tensor * canvas) -> ggml_tensor * {
        if (model.dg_sc_enabled) {
            const int64_t Cc      = canvas->ne[1];
            const int64_t n_vocab = model.tok_embd->ne[1];
            const int64_t canvas_len = (int64_t) hparams.diffusion_canvas_length;
            canvas = ggml_cont(ctx0, canvas);

            // SC GPU-RESIDENT (PR #24423-style): instead of a 262 MB host ggml_set_input uploaded+softmaxed on
            // the CPU split every step, VIEW the persistent on-device logits buffer (filled in set_sc). The
            // softmax/embed then run on GPU. Buffer is [n_vocab, canvas_length]; Cc <= canvas_length, so view
            // the first Cc columns (contiguous). dg_sc_use=0 on step 0 gates a possibly-uninit buffer to zero.
            dg_ensure_sc_logits_gpu(model, n_vocab, canvas_len > 0 ? canvas_len : Cc);
            ggml_tensor * sc_logits = ggml_view_2d(ctx0, model.dg_sc_logits_gpu, n_vocab, Cc,
                                                   model.dg_sc_logits_gpu->nb[1], 0);

            // raw/temperature, then softmax over vocab; soft_emb = (softmax @ embed) * embed_scale
            ggml_tensor * probs = ggml_soft_max(ctx0, ggml_scale(ctx0, sc_logits, model.dg_sc_temp_inv));
            dg_ensure_sc_embT(model);
            ggml_tensor * soft = ggml_mul_mat(ctx0, model.dg_sc_embT, probs);             // [n_embd, Cc]
            soft = ggml_scale(ctx0, soft, sqrtf((float) n_embd));
            // SC gated MLP: pre_norm -> down( gelu(gate) * up )
            ggml_tensor * normed = build_norm(soft, model.sc_pre_norm, nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * g  = ggml_gelu(ctx0, ggml_mul_mat(ctx0, model.sc_gate, normed)); // [n_ff, Cc]
            ggml_tensor * u  = ggml_mul_mat(ctx0, model.sc_up, normed);                    // [n_ff, Cc]
            ggml_tensor * sc_sig = ggml_mul_mat(ctx0, model.sc_down, ggml_mul(ctx0, g, u)); // [n_embd, Cc]
            sc_sig = ggml_scale(ctx0, sc_sig, model.dg_sc_use); // runtime {0,1} gate (0 == first step)
            canvas = ggml_add(ctx0, canvas, sc_sig);
            canvas = ggml_rms_norm(ctx0, canvas, hparams.f_norm_rms_eps); // post_norm, no scale
        } else {
            canvas = ggml_rms_norm(ctx0, ggml_cont(ctx0, canvas), hparams.f_norm_rms_eps); // no scale (zero-SC)
        }
        return canvas;
    };

    if (dg_phase == llama_model::DG_PKV_DECODE) {
        inpL = dg_canvas_embed(inpL);                 // whole batch is the fresh canvas
    } else if (dg_phase == llama_model::DG_PKV_PREFILL) {
        // prompt-only (encoder): leave inpL as the scaled embedding (no canvas embed / no SC)
    } else if (P > 0 && P < n_tokens) {
        ggml_tensor * prompt = ggml_view_2d(ctx0, inpL, n_embd, P, inpL->nb[1], 0);
        ggml_tensor * canvas = ggml_view_2d(ctx0, inpL, n_embd, C, inpL->nb[1], P * inpL->nb[1]);
        canvas = dg_canvas_embed(canvas);
        inpL = ggml_concat(ctx0, ggml_cont(ctx0, prompt), canvas, 1);
    } else if (P == 0) {
        inpL = dg_canvas_embed(inpL); // pure-canvas
    }
    // else (P == n_tokens, UNIFIED pure-prompt): leave as embed*sqrt
    cb(inpL, "inp_region", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    // Region-aware no-cache mask. DECODE: rectangular [n_kv=P+C keys, C queries] over the cached prompt +
    // fresh canvas. UNIFIED/PREFILL: square [n_tokens, n_tokens]. fills self_kq_mask (+swa) per layer kind.
    llm_graph_input_attn_no_cache * inp_attn = nullptr;
    if (dg_phase == llama_model::DG_PKV_DECODE) {
        const int64_t n_kv = P + C;
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion_decode>(hparams, cparams, P, C);
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, C, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        ggml_set_name(uptr->self_kq_mask, "self_kq_mask");
        uptr->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, uptr->self_kq_mask, GGML_TYPE_F16)
                                                     : uptr->self_kq_mask;
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            uptr->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, C, 1, 1);
            ggml_set_input(uptr->self_kq_mask_swa);
            ggml_set_name(uptr->self_kq_mask_swa, "self_kq_mask_swa");
            uptr->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, uptr->self_kq_mask_swa, GGML_TYPE_F16)
                                                            : uptr->self_kq_mask_swa;
        }
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    } else {
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion>(hparams, cparams, P);
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        ggml_set_name(uptr->self_kq_mask, "self_kq_mask");
        uptr->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, uptr->self_kq_mask, GGML_TYPE_F16)
                                                     : uptr->self_kq_mask;
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            uptr->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
            ggml_set_input(uptr->self_kq_mask_swa);
            ggml_set_name(uptr->self_kq_mask_swa, "self_kq_mask_swa");
            uptr->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, uptr->self_kq_mask_swa, GGML_TYPE_F16)
                                                            : uptr->self_kq_mask_swa;
        }
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.is_swa(il) && hparams.n_embd_head_k_swa > 0 ? hparams.n_embd_head_k_swa : hparams.n_embd_head_k;
        const int64_t n_embd_head_v = hparams.is_swa(il) && hparams.n_embd_head_v_swa > 0 ? hparams.n_embd_head_v_swa : hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == n_embd_head_v);

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const float   freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float   freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int     n_rot_l      = hparams.n_rot_layer(il);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        // Q projection + per-head q-norm + partial/proportional rope
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        // K/V projection + k-norm + V no-scale rms-norm + K rope (V == K when v_proj absent, global layers)
        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        ggml_tensor * Vcur = model.layers[il].wv ? build_lora_mm(model.layers[il].wv, cur) : Kcur;
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps); // v-norm: no scale
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur_pos", il);

        if (dg_phase == llama_model::DG_PKV_PREFILL) {
            // PKV PREFILL: cache this layer's prompt K/V (post-rope) for the block's decode steps, then attend.
            ggml_tensor * sk = ggml_view_3d(ctx0, model.dg_pkv_k[il], n_embd_head, n_head_kv, n_tokens,
                                            model.dg_pkv_k[il]->nb[1], model.dg_pkv_k[il]->nb[2], 0);
            ggml_tensor * sv = ggml_view_3d(ctx0, model.dg_pkv_v[il], n_embd_head, n_head_kv, n_tokens,
                                            model.dg_pkv_v[il]->nb[1], model.dg_pkv_v[il]->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, sk));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, sv));
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        } else if (dg_phase == llama_model::DG_PKV_DECODE) {
            // PKV DECODE: prepend the cached prompt K/V (first P) to the fresh canvas K/V, attend [P+C].
            ggml_tensor * pk = ggml_view_3d(ctx0, model.dg_pkv_k[il], n_embd_head, n_head_kv, P,
                                            model.dg_pkv_k[il]->nb[1], model.dg_pkv_k[il]->nb[2], 0);
            ggml_tensor * pv = ggml_view_3d(ctx0, model.dg_pkv_v[il], n_embd_head, n_head_kv, P,
                                            model.dg_pkv_v[il]->nb[1], model.dg_pkv_v[il]->nb[2], 0);
            ggml_tensor * Kfull = ggml_concat(ctx0, pk, Kcur, 2);
            ggml_tensor * Vfull = ggml_concat(ctx0, pv, Vcur, 2);
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kfull, Vfull,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        } else {
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);
        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // dense-MLP (shared expert) + 128-expert MoE + ffn_post_norm + residual (gemma4 backbone)
        if (model.layers[il].ffn_gate_inp != nullptr) {
            ggml_tensor * cur_mlp = build_norm(attn_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp, model.layers[il].ffn_post_norm_1, NULL, LLM_NORM_RMS, il);

            ggml_tensor * cur_moe = build_norm(attn_out, model.layers[il].ffn_pre_norm_2, NULL, LLM_NORM_RMS, il);
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp);
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true, 1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    /* up_exps_s   */ nullptr,
                    /* gate_exps_s */ nullptr,
                    /* down_exps_s */ model.layers[il].ffn_down_exps_s); // per-expert down scale (router.per_expert_scale)
            cur_moe = build_norm(cur_moe, model.layers[il].ffn_post_norm_2, NULL, LLM_NORM_RMS, il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
        } else {
            cur = build_norm(attn_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_GELU, LLM_FFN_PAR, il);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cur = ggml_add(ctx0, cur, attn_out);

        // region-aware per-layer scalar: prompt * encoder scalar, canvas * decoder scalar.
        // PHASE-AWARE: in DECODE the batch is pure canvas (P is the CACHED prompt length, NOT in the batch) and
        // in PREFILL it is pure prompt — so the UNIFIED region-split (which assumes P+C == n_tokens) must NOT
        // run there or its canvas view overflows the canvas-only tensor.
        if (dg_phase == llama_model::DG_PKV_DECODE) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);      // whole batch is canvas (decoder)
        } else if (dg_phase == llama_model::DG_PKV_PREFILL) {
            cur = ggml_mul(ctx0, cur, model.layers[il].enc_out_scale);  // whole batch is prompt (encoder)
        } else if (P > 0 && P < n_tokens) {
            ggml_tensor * prompt = ggml_view_2d(ctx0, cur, n_embd, P, cur->nb[1], 0);
            ggml_tensor * canvas = ggml_view_2d(ctx0, cur, n_embd, C, cur->nb[1], P * cur->nb[1]);
            prompt = ggml_mul(ctx0, ggml_cont(ctx0, prompt), model.layers[il].enc_out_scale);
            canvas = ggml_mul(ctx0, ggml_cont(ctx0, canvas), model.layers[il].out_scale);
            cur = ggml_concat(ctx0, prompt, canvas, 1);
        } else if (P == 0) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);      // pure-canvas
        } else {
            cur = ggml_mul(ctx0, cur, model.layers[il].enc_out_scale);  // pure-prompt
        }
        cb(cur, "out_scaled", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    // GPU-RESIDENT SC (PR #24423-style): persist THIS step's canvas logits (the last C output rows) into
    // dg_sc_logits_gpu for the NEXT step's SC read — entirely on-device, no host round-trip. The cpy's source
    // is `cur` (the lm_head output), which is downstream of the SC read in dg_canvas_embed, so topo order runs
    // read-before-write on the shared buffer (no hazard — same trick the PR uses). F32→F32, bit-identical to
    // what the old host upload pushed. C==canvas_length at decode; the buffer is [n_vocab, canvas_length].
    if (model.dg_sc_enabled && C > 0 && model.dg_sc_logits_gpu) {
        const int64_t n_out   = cur->ne[1];
        const int64_t n_vocab = cur->ne[0];
        if (n_out >= C) {
            ggml_tensor * canvas_logits = ggml_view_2d(ctx0, cur, n_vocab, C,
                                                       cur->nb[1], (n_out - C) * cur->nb[1]);
            ggml_tensor * sc_dst = ggml_view_2d(ctx0, model.dg_sc_logits_gpu, n_vocab, C,
                                                model.dg_sc_logits_gpu->nb[1], 0);
            ggml_tensor * sc_cpy = ggml_cpy(ctx0, canvas_logits, sc_dst);
            ggml_build_forward_expand(gf, sc_cpy);
        }
    }
}

// Public API: set per-request self-conditioning (no-op on other models). sc_logits is borrowed and must
// stay valid through the next llama_decode (when the graph uploads it). Keep enabled=true across the whole
// run (use_sc=0 on the first step) so the SC subgraph shape stays stable between sched_reserve and decode.
void llama_diffusion_set_sc(struct llama_model * model, const float * sc_logits,
                            float use_sc, float temp_inv, bool enabled) {
    if (!model || model->arch != LLM_ARCH_DIFFUSION_GEMMA) {
        return;
    }
    if (enabled && use_sc != 0.0f && sc_logits && model->dg_sc_logits_gpu &&
        (dg_env_enabled("DG_SC_CHECK") || dg_env_enabled("LLAMA_DG_SC_CHECK"))) {
        const size_t n = (size_t) ggml_nelements(model->dg_sc_logits_gpu);
        std::vector<float> dev(n);
        ggml_backend_tensor_get(model->dg_sc_logits_gpu, dev.data(), 0, n * sizeof(float));
        double maxabs = 0.0;
        double sumabs = 0.0;
        size_t nmiss = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs((double) dev[i] - (double) sc_logits[i]);
            maxabs = std::max(maxabs, d);
            sumabs += d;
            nmiss += d != 0.0;
        }
        std::fprintf(stderr, "DG_SC_CHECK: maxabs=%.9g sumabs=%.9g nmiss=%zu/%zu\n",
                     maxabs, sumabs, nmiss, n);
    }
    model->dg_sc_logits_ptr = sc_logits;   // vestigial (kept for compat); the graph reads the GPU buffer now
    model->dg_sc_use        = use_sc;
    model->dg_sc_temp_inv   = temp_inv;
    model->dg_sc_enabled    = enabled;
    // GPU-RESIDENT SC (PR #24423-style): the prev-step canvas logits live in dg_sc_logits_gpu, written by the
    // PREVIOUS forward's lm_head via an in-graph ggml_cpy (see build) — no host upload here. set_sc only
    // carries the per-step gate/temp; the host `sc_logits` ptr is unused for the device path.
}

// Public API: select the prompt-KV-caching phase for the next llama_decode (no-op on other models). UNIFIED
// (default) = the proven no-cache forward; PREFILL writes the prompt K/V store (P = prompt length); DECODE
// reads the first P. Opt-in: the diffusion runner calls this only when LLAMA_DG_PKV is enabled.
void llama_diffusion_set_phase(struct llama_model * model, int phase, int32_t P) {
    if (!model || model->arch != LLM_ARCH_DIFFUSION_GEMMA) {
        return;
    }
    model->dg_pkv_phase = (llama_model::dg_pkv_phase_t) phase;
    model->dg_pkv_P     = P;
    if (phase != llama_model::DG_PKV_UNIFIED && P > 0) {
        dg_ensure_pkv_store(*model, P);
    }
}

// Public API (PR #24423-style): Stage-1 device sampling — argmax / entropy / one multinomial draw per canvas
// position read straight from the device SC buffer (dg_sc_logits_gpu, the post-softcap canvas logits the last
// forward wrote in-graph), so the per-step host full-canvas vocab scan goes away. u is [n_tokens] host
// pre-drawn uniforms (the host RNG stream, for reproducibility); argmax/entropy/sampled are [n_tokens] host
// outputs. The CUDA kernel is reached via the backend-reg proc address (keeps the llama<->ggml-cuda link at
// the existing boundary). Caller MUST llama_synchronize(ctx) first. Returns false (caller falls back to the
// host path) on non-DiffusionGemma / no SC buffer / non-CUDA / unsupported tensor. The CUDA side resolves the
// device that owns the logits buffer, so layer-split output-device sampling is supported. argmax matches the
// host bit-for-bit; entropy/sampled differ only by FP reduction order. Only valid with NO token bias.
typedef bool (*dg_cuda_sample_fn)(struct ggml_tensor *, const float *, int *, float *, int *, int, float);

bool llama_diffusion_device_sample(const struct llama_model * model, const float * u, int * argmax,
                                   float * entropy, int * sampled, int n_tokens, float inv_temp) {
    if (!model || model->arch != LLM_ARCH_DIFFUSION_GEMMA || model->dg_sc_logits_gpu == nullptr ||
        !u || !argmax || !entropy || !sampled || n_tokens <= 0) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        return false;
    }
    static dg_cuda_sample_fn fn =
        (dg_cuda_sample_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_diffusion_sample");
    if (!fn) {
        return false;
    }
    return fn(model->dg_sc_logits_gpu, u, argmax, entropy, sampled, n_tokens, inv_temp);
}
