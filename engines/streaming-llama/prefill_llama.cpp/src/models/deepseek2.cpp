#include "models.h"

#include "../llama-mtp-policy.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

llm_build_deepseek2::llm_build_deepseek2(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();

    // Phase 1 (iter 29): sparse MLA gate for GLM-5.1 / DeepSeek V3.2 sparse-attention (DSA).
    // Decode is wired for n_tokens == 1; prefill gathers a sparse K/V + causal mask.
    // Explicit runtime flags win; legacy env values are resolved once into cparams.
    const bool sparse_mla_enabled = cparams.sparse_mla;
    const bool sparse_mla_trace = cparams.sparse_mla_trace;
    const bool sparse_mla_prefill_enabled = cparams.sparse_mla_prefill;
    const bool sparse_mla_decode_enabled = cparams.sparse_mla_decode;
    const int sparse_mla_top_k_cap = cparams.sparse_mla_top_k;
    const int sparse_mla_prefill_top_k_cap = cparams.sparse_mla_prefill_top_k;
    const bool sparse_mla_prefill_recent =
        cparams.sparse_mla_prefill_selector == LLAMA_SPARSE_MLA_SELECTOR_RECENT;
    const bool sparse_mla_prefill_prefix =
        cparams.sparse_mla_prefill_selector == LLAMA_SPARSE_MLA_SELECTOR_PREFIX;
    const bool sparse_mla_prefill_skip_indexer = !cparams.sparse_mla_prefill_indexer;
    const bool sparse_mla_fa_with_indices = cparams.sparse_mla_fa_mode == LLAMA_SPARSE_MLA_FA_INDICES;
    const bool sparse_mla_fa_v5 = cparams.sparse_mla_fa_mode == LLAMA_SPARSE_MLA_FA_V5;
    const bool sparse_mla_fa_v6 = cparams.sparse_mla_fa_mode == LLAMA_SPARSE_MLA_FA_V6;
    const int moe_prefill_expert_cap = cparams.moe_prefill_expert_cap;
    // [GLM_DSA_DETERMINISTIC_SPARSE_FA_V1]
    // The v6 NCOLS32 launcher normally partitions the continuous K work over
    // the occupancy-sized grid. Most ubatch shapes put some block boundaries
    // inside a query/head tile, sending those tiles through the global
    // stream-K seam/fixup reduction. Fixed recent rows then depend on which
    // tiles hit that path, while early rows that stay in whole tiles are stable.
    //
    // Mark only GLM-DSA sparse ops for the tile-aligned launcher. Other
    // architectures remain byte-identical. =0 is the exact legacy/fixup A/B.
    static const bool glm_dsa_deterministic_sparse_fa_gate = []() {
        const char * env =
            std::getenv("LLAMA_MLA_SPARSE_FA_V6_TILE_ALIGNED_GRID");
        return !(env && env[0] == '0');
    }();
    const bool glm_dsa_deterministic_sparse_fa =
        model.arch == LLM_ARCH_GLM_DSA &&
        sparse_mla_fa_v6 &&
        glm_dsa_deterministic_sparse_fa_gate;
    const bool sparse_mla_indexer_available = hparams.indexer_n_head > 0 &&
                                              hparams.indexer_head_size > 0 &&
                                              hparams.indexer_top_k > 0;
    static bool sparse_mla_logged = false;
    if (sparse_mla_enabled && !sparse_mla_logged) {
        if (sparse_mla_indexer_available) {
            fprintf(stderr,
                    "[sparse-mla] sparse MLA gate active; indexer_n_head=%u, indexer_head_size=%u, indexer_top_k=%u (prefill=%d decode=%d)\n",
                    hparams.indexer_n_head, hparams.indexer_head_size, hparams.indexer_top_k,
                    sparse_mla_prefill_enabled ? 1 : 0, sparse_mla_decode_enabled ? 1 : 0);
            if (sparse_mla_top_k_cap > 0 && sparse_mla_top_k_cap < (int) hparams.indexer_top_k) {
                fprintf(stderr,
                        "[sparse-mla] sparse top_k=%d caps sparse gather below model top_k=%u\n",
                        sparse_mla_top_k_cap, hparams.indexer_top_k);
            }
            if (sparse_mla_prefill_top_k_cap > 0 && sparse_mla_prefill_top_k_cap < (int) hparams.indexer_top_k) {
                fprintf(stderr,
                        "[sparse-mla] sparse prefill top_k=%d caps prefill gather below model top_k=%u\n",
                        sparse_mla_prefill_top_k_cap, hparams.indexer_top_k);
            }
            if (sparse_mla_prefill_recent || sparse_mla_prefill_prefix) {
                fprintf(stderr,
                        "[sparse-mla] sparse prefill positional selector active: %s\n",
                        sparse_mla_prefill_prefix ? "prefix" : "recent");
            }
            if (sparse_mla_prefill_skip_indexer) {
                fprintf(stderr,
                        "[sparse-mla] sparse prefill indexer disabled: positional prefill will not populate DSA k_idx cache\n");
            }
            if (moe_prefill_expert_cap > 0) {
                fprintf(stderr,
                        "[sparse-mla] moe prefill expert cap=%d: approximate prefill caps routed experts\n",
                        moe_prefill_expert_cap);
            }
            if (sparse_mla_fa_with_indices) {
                fprintf(stderr,
                        "[sparse-mla] sparse FA indices mode: sparse FA will read full K/V through top-k indices when eligible\n");
            }
            if (sparse_mla_fa_v5) {
                fprintf(stderr,
                        "[sparse-mla] sparse FA v5: sparse FA will use shmem-tile K/V gather when eligible\n");
            }
            if (sparse_mla_fa_v6) {
                fprintf(stderr,
                        "[sparse-mla] sparse FA v6: sparse FA will use dense MMA with indexed K/V loads when eligible\n");
            }
        } else {
            fprintf(stderr,
                    "[sparse-mla] sparse MLA requested but this model has no indexer tensors (indexer_n_head=%u); sparse gate inactive, falling back to dense MLA\n",
                    hparams.indexer_n_head);
        }
        sparse_mla_logged = true;
    }
    // Effective sparse routing flag; used by the branch point added in phase 4.
    const bool use_sparse_mla = sparse_mla_enabled && sparse_mla_indexer_available;

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    const int32_t total_n_layers = (int32_t) hparams.n_layer;
    const int32_t nextn_layers   = (int32_t) hparams.nextn_predict_layers;
    const bool mtp_enabled       = llama_mtp_graph_uses_nextn(cparams.mtp_mode, params.mtp_graph_role) && nextn_layers > 0;
    const int effective_n_layers = llama_mtp_effective_layer_count(
        total_n_layers, nextn_layers, cparams.mtp_mode, params.mtp_graph_role);

    // {n_embd, n_tokens}
    // For per-layer graphs with external_input: use the intermediate buffer
    // instead of embedding lookup. This allows feeding previous layer output
    // as the next layer's input.
    ggml_tensor * inp_token_embd = nullptr;
    if (params.external_input) {
        if (mtp_enabled) {
            inp_token_embd = build_inp_embd(model.tok_embd);
        }
        inpL = params.external_input;
    } else {
        inp_token_embd = build_inp_embd(model.tok_embd);
        inpL = inp_token_embd;
    }

    // (optional) temperature tuning - used by mistral-large
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_inp_attn_scale();
    }

    // Per-layer streaming: only build nodes for [layer_start, layer_end)
    const int il_start = (params.layer_start >= 0) ? params.layer_start : 0;
    const int il_end   = llama_mtp_clamp_layer_end_for_role(
        params.layer_end, total_n_layers, nextn_layers, cparams.mtp_mode, params.mtp_graph_role);

    // Skip attention-input setup when no layers will run (embed-only mode) or when
    // ALL layers in the range use external_attn_out (FFN-only — attention skipped
    // via the goto post_attn path). These inputs reference KV-cache tensors that
    // would otherwise become orphaned graph nodes without buffers.
    // Mega-graph path (external_attn_out_per_layer) also skips attention inputs.
    const bool skip_attn_inputs =
        (il_start >= il_end) ||
        (params.external_attn_out != nullptr && (il_end - il_start) == 1) ||
        (params.external_attn_out_per_layer != nullptr);

    // inp_pos is only used inside the attention block (for RoPE). Skip when
    // attention won't run.
    ggml_tensor * inp_pos = skip_attn_inputs ? nullptr : build_inp_pos();

    auto * inp_attn_kv = (!is_mla && !skip_attn_inputs) ? build_attn_inp_kv() : nullptr;
    auto * inp_attn_k  = (is_mla  && !skip_attn_inputs) ? build_attn_inp_k()  : nullptr;

    // Only build out_ids for the last layer (it's used by get_rows at output).
    // Also skip when stop_after_attn is set (we don't reach the get_rows block),
    // and skip when no layer iterations run (apply_output_head path: il_start == il_end).
    // Also skip when n_outputs == 0 (intermediate ubatches in layer-major mode) —
    // a zero-element I32 tensor gets allocated at offset 0 of its compute buffer,
    // which gallocr aliases with other tensors; set_inputs then writes OOB garbage
    // indices that corrupt MoE routing get_rows calls later in the graph.
    ggml_tensor * inp_out_ids = (il_end >= effective_n_layers && !params.stop_after_attn && il_start < il_end && n_outputs > 0)
        ? build_inp_out_ids() : nullptr;

    // [OPTION A / DSA_SHARED_SPARSE_REUSE 2026-07-07] Persist the previous FULL cadence layer's top-k selection so
    // the "shared" layers can REUSE it and run SPARSE attention (skipping their own indexer score-GEMM) instead of
    // the dense O(n^2) fallback. env-gated LLAMA_DSA_SHARED_SPARSE_REUSE (default OFF -> byte-identical to the
    // existing dense/sparse paths). prev_full_topk resets to null each forward pass (function re-entered per ubatch).
    ggml_tensor * prev_full_topk = nullptr;
    static const bool dsa_shared_sparse_reuse = []() {
        const char * e = std::getenv("LLAMA_DSA_SHARED_SPARSE_REUSE");
        return e && e[0] && e[0] != '0';
    }();

    for (int il = 0; il < effective_n_layers; ++il) {
        // Skip layers outside the requested range
        if (il < il_start || il >= il_end) {
            // Still advance inpL for skipped layers by creating a dummy
            // identity — the graph won't compute these nodes (they have
            // no consumers). For layers BEFORE the range, inpL will be
            // overwritten when we set the intermediate buffer input.
            continue;
        }

        ggml_tensor * inpSA = inpL;

        if (mtp_enabled && llama_mtp_is_nextn_layer(il, total_n_layers, nextn_layers)) {
            const auto & nextn = model.layers[il].nextn;
            if (nextn.eh_proj == nullptr || nextn.enorm == nullptr || nextn.hnorm == nullptr ||
                nextn.shared_head_norm == nullptr) {
                throw std::runtime_error("MTP mode requested but GLM_DSA NextN tensors are missing from the model");
            }
            if (inp_token_embd == nullptr) {
                throw std::runtime_error("MTP mode requested but token embeddings are unavailable for the NextN layer");
            }

            ggml_tensor * emb_norm = build_norm(inp_token_embd, nextn.enorm, nullptr, LLM_NORM_RMS, il);
            cb(emb_norm, "mtp_enorm", il);
            ggml_tensor * hidden_norm = build_norm(inpL, nextn.hnorm, nullptr, LLM_NORM_RMS, il);
            cb(hidden_norm, "mtp_hnorm", il);

            cur = ggml_concat(ctx0, emb_norm, hidden_norm, 0);
            cb(cur, "mtp_eh_concat", il);
            cur = ggml_mul_mat(ctx0, nextn.eh_proj, cur);
            cb(cur, "mtp_eh_proj", il);

            inpL  = cur;
            inpSA = cur;
        }

        // Mega-graph split-decode (Path A): per-layer external attention output.
        // When external_attn_out_per_layer is set, the array indexed by
        // (il - il_start) provides each layer's attention output.
        if (params.external_attn_out_per_layer != nullptr) {
            cur = params.external_attn_out_per_layer[il - il_start];
            if (params.external_input_per_layer != nullptr) {
                // Override inpL from the per-layer input buffer (breaks residual
                // propagation — used when caller drives layer inputs externally).
                inpL  = params.external_input_per_layer[il - il_start];
                inpSA = inpL;
            }
            cb(cur, "attn_out_external_pl", il);
            goto post_attn;
        }

        // Split-decode ffn-only path: skip the entire attention block (norm + compute)
        // and use the externally-provided attention output instead. This is the Mac-side
        // path — attention was computed on RTX and returned over the wire.
        if (params.external_attn_out != nullptr) {
            cur = params.external_attn_out;
            cb(cur, "attn_out_external", il);
            goto post_attn;
        }

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        // 2026-05-31 PIN_QKV_INPUT (env, default off): at chunk5 (n_kv>=163840) gallocr
        // aliases the attn_norm-output / kv_cmpr_pe compute slot with an n_kv-scaled FA/mask
        // transient at the largest working set, clobbering the shared QKV input → Kcur+Vcur=0
        // for layers>=1 (input is healthy; 3 input-delivery fixes failed → it's the OUTPUT slot).
        // ggml_set_output marks the buffer live-to-graph-end so gallocr can't reuse/alias it
        // (same anti-reuse pattern as l_out). Marker PIN_QKV_INPUT for build-grep.
        {
            static const bool pin_qkv = std::getenv("LLAMA_PREFILL_PIN_QKV_INPUT") != nullptr;  // PIN_QKV_INPUT
            if (pin_qkv) ggml_set_output(cur);
        }

        // self_attention
        {
            ggml_tensor * q = NULL;

            const bool is_lite = model.layers[il].wq;

            // Phase 3 (iter 29): capture post-norm q_lora reference for the sparse-MLA
            // indexer path. wq_b overwrites `q` below with [n_head * n_embd_head_k, n_tokens];
            // the sparse indexer needs the pre-wq_b q_lora ([q_lora_rank, n_tokens]).
            ggml_tensor * q_lora_for_indexer = nullptr;
            if (!is_lite) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                // Phase 76: env-gated eps override for MLA q_a / kv_a layernorms.
                // mlx_lm/models/deepseek_v3.py hardcodes eps=1e-6 for these two
                // RMSNorms (independent of config.rms_norm_eps), while llama.cpp
                // uses the GGUF-declared global value (1e-5 for Kimi K2.6). When
                // streaming KV inject from RTX → Mac MLX, this mismatch produces
                // bit-different kv_latent values that cascade into degenerate
                // attention scores. Setting LLAMA_DEEPSEEK2_MLA_NORM_EPS=1e-6 makes
                // RTX's q_a / kv_a outputs match what MLX would compute.
                {
                    static const float mla_norm_eps = []() {
                        const char * e = std::getenv("LLAMA_DEEPSEEK2_MLA_NORM_EPS");
                        return e ? std::strtof(e, nullptr) : 0.0f;
                    }();
                    if (mla_norm_eps > 0.0f && mla_norm_eps < 1.0f) {
                        q = ggml_rms_norm(ctx0, q, mla_norm_eps);
                        q = ggml_mul(ctx0, q, model.layers[il].attn_q_a_norm);
                    } else {
                        q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                    }
                }
                cb(q, "q", il);
                q_lora_for_indexer = q;

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
                // is_lite path has no q_lora split; sparse-MLA branch falls through to dense.
            }
            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);
            {
                static const bool pin_qkv = std::getenv("LLAMA_PREFILL_PIN_QKV_INPUT") != nullptr;  // PIN_QKV_INPUT
                if (pin_qkv) ggml_set_output(kv_cmpr_pe);  // pin the wkv_a_mqa output (Kcur/Vcur source)
            }

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            // Phase 76: env-gated eps override (same as q_a above).
            {
                static const float mla_norm_eps = []() {
                    const char * e = std::getenv("LLAMA_DEEPSEEK2_MLA_NORM_EPS");
                    return e ? std::strtof(e, nullptr) : 0.0f;
                }();
                if (mla_norm_eps > 0.0f && mla_norm_eps < 1.0f) {
                    kv_cmpr = ggml_rms_norm(ctx0, kv_cmpr, mla_norm_eps);
                    kv_cmpr = ggml_mul(ctx0, kv_cmpr, model.layers[il].attn_kv_a_norm);
                } else {
                    kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
                }
            }
            cb(kv_cmpr, "kv_cmpr", il);

            if (is_mla) {
                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                if (cparams.flash_attn) {
                    const char * force_cont = std::getenv("LLAMA_FA_MLA_KCUR_CONT");
                    if (force_cont && force_cont[0] != '\0' && force_cont[0] != '0') {
                        Kcur = ggml_cont(ctx0, Kcur);
                    }
                }
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                if (cparams.flash_attn) {
                    const char * force_v_cont = std::getenv("LLAMA_FA_MLA_VCUR_CONT");
                    if (force_v_cont && force_v_cont[0] != '\0' && force_v_cont[0] != '0') {
                        Vcur = ggml_cont(ctx0, Vcur);
                    }
                }
                cb(Vcur, "Vcur", il);

                // DIAG (2026-05-28): persist Kcur/Vcur (MLA cache inputs) at the dump layer so
                // they can be checksummed — splits multi-ubatch L1 attention divergence into
                // cache-content-wrong (K/V differ at token 0) vs FA-read-wrong (K/V identical).
                // Gated by LLAMA_DUMP_LAYER; diagnostic only.
                {
                    const char * dl_env = std::getenv("LLAMA_DUMP_LAYER");
                    if (dl_env && dl_env[0]) {
                        const int dl = atoi(dl_env);
                        if (il == dl || il == dl - 1) { ggml_set_output(Kcur); ggml_set_output(Vcur); }
                    }
                }

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                ggml_tensor * sparse_attn_out = nullptr;

                // Phase 3 (iter 29): sparse-MLA indexer projections + RoPE (scores not yet consumed).
                // Adds indexer K/Q projections, norm, and RoPE on the rope_dim prefix per head
                // (matching sglang nsa_indexer.py:346). Hadamard + head-gate + score matmul
                // still land in later phases (3c/3d). Decode and prefill now consume
                // top-k gathers; prefill also gathers the matching causal mask rows.
                if (use_sparse_mla && q_lora_for_indexer != nullptr && inp_pos != nullptr &&
                    model.layers[il].indexer_attn_k != nullptr &&
                    model.layers[il].indexer_attn_q_b != nullptr) {
                    const int64_t idx_head_size = (int64_t) hparams.indexer_head_size;
                    const int64_t n_idx_head    = (int64_t) hparams.indexer_n_head;
                    const int64_t rope_dim_idx  = (int64_t) n_embd_head_qk_rope; // same n_rot as main MLA

                    // GLM DSA reference (HF/vLLM) applies indexer-specific NON-interleaved (NeoX / half-split)
                    // RoPE to the lightning indexer, UNLIKE the main MLA attention which is interleaved
                    // (GLM_DSA -> LLAMA_ROPE_TYPE_NORM = 0). When the indexer reuses the main NORM rope its
                    // query/key are position-phased incorrectly, so the full-layer top-k can omit the
                    // relevant (needle) key. SEMI hides this (dense shared layers); FULL (sparse-reuse shared)
                    // exposes it as needle-drop (single-needle recall 0/N). Env-gated A/B, DEFAULT OFF so
                    // SEMI/default/production behavior is byte-for-byte unchanged. Ref: vLLM sets
                    // is_neox_style = not config.indexer_rope_interleave.
                    static const bool dsa_indexer_rope_neox = []() {
                        const char * e = std::getenv("LLAMA_DSA_INDEXER_ROPE_NEOX");
                        const bool on = e && e[0] && e[0] != '0';
                        // One-time propagation marker (verify the env var actually reached this binary by
                        // BEHAVIOR/log, never by assumption — see the NCOLS-launcher lesson). Also a build-grep
                        // marker string so we can confirm the new object linked into the DLL/exe.
                        fprintf(stderr, "[dsa-indexer] LLAMA_DSA_INDEXER_ROPE_NEOX=%s -> indexer RoPE=%s\n",
                                e ? e : "(unset)", on ? "NeoX/non-interleaved" : "NORM/interleaved(default)");
                        return on;
                    }();
                    const int indexer_rope_type = dsa_indexer_rope_neox
                        ? (rope_type | GGML_ROPE_TYPE_NEOX)
                        : rope_type;

                    const bool skip_prefill_indexer =
                        sparse_mla_prefill_skip_indexer &&
                        n_tokens > 1 &&
                        (sparse_mla_prefill_recent || sparse_mla_prefill_prefix);

                    ggml_tensor * idx_k_full = nullptr;
                    if (skip_prefill_indexer) {
                        idx_k_full = ggml_view_3d(ctx0, Kcur,
                            idx_head_size, 1, n_tokens,
                            Kcur->nb[1], Kcur->nb[2], 0);
                        cb(idx_k_full, "indexer_k_dummy_from_kcur", il);
                    } else {
                        // ==== K path (single-head, shape [idx_head_size, n_tokens]) ====
                        ggml_tensor * idx_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                        ggml_mul_mat_set_prec(idx_k, GGML_PREC_F32);
                        cb(idx_k, "indexer_k_proj", il);

                        // NOTE: sglang uses nn.LayerNorm (mean-variance normalize + affine), NOT RMSNorm.
                        // Using LLM_NORM_RMS here silently discards the mean-centering step, which
                        // causes the indexer scoring to pick the wrong top-k tokens (verified: top-2048
                        // decode was degenerate "the every every every inventor..." until this fix).
                        // See sglang/srt/layers/attention/nsa/nsa_indexer.py:228 — LayerNorm(head_dim).
                        idx_k = build_norm(idx_k, model.layers[il].indexer_k_norm,
                                                  model.layers[il].indexer_k_norm_b,
                                                  LLM_NORM, il);
                        cb(idx_k, "indexer_k_norm", il);

                        // Reshape to 3D [idx_head_size, 1, n_tokens] for rope_ext + concat semantics
                        ggml_tensor * idx_k_3d = ggml_reshape_3d(ctx0, idx_k, idx_head_size, 1, n_tokens);

                        // View rope prefix [rope_dim, 1, n_tokens] then apply RoPE
                        ggml_tensor * idx_k_rope_view = ggml_view_3d(ctx0, idx_k_3d,
                            rope_dim_idx, 1, n_tokens,
                            ggml_row_size(idx_k_3d->type, idx_head_size),
                            ggml_row_size(idx_k_3d->type, idx_head_size),
                            0);
                        ggml_tensor * idx_k_rope = ggml_rope_ext(ctx0, ggml_cont(ctx0, idx_k_rope_view),
                            inp_pos, nullptr, rope_dim_idx, indexer_rope_type, n_ctx_orig,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                        cb(idx_k_rope, "indexer_k_rope", il);

                        // View non-rope tail [idx_head_size - rope_dim, 1, n_tokens]
                        ggml_tensor * idx_k_nope_view = ggml_view_3d(ctx0, idx_k_3d,
                            idx_head_size - rope_dim_idx, 1, n_tokens,
                            ggml_row_size(idx_k_3d->type, idx_head_size),
                            ggml_row_size(idx_k_3d->type, idx_head_size),
                            ggml_row_size(idx_k_3d->type, rope_dim_idx));

                        // Concat rope + nope → [idx_head_size, 1, n_tokens]
                        idx_k_full = ggml_concat(ctx0, idx_k_rope, ggml_cont(ctx0, idx_k_nope_view), 0);
                        cb(idx_k_full, "indexer_k_full", il);
                    }

                    // [2026-07-28 DSA STORE-DEPS FIX, codex-v4] SET_ROWS writes the indexer/main K
                    // caches through an ALIAS: the cache views consumed by the DSA score / sparse FA
                    // are constructed independently, so the GGML DAG sees NO producer->consumer edge
                    // from the store to its readers. Any scheduler freedom (node-range dispatch,
                    // reachability pruning, multi-stream graph opt) can then run the reader before /
                    // without the store -> stale or UNINITIALIZED top-k indices -> wild-address
                    // gather -> the GLM-DSA partial-ubatch IMA. Fix: attach the store result as an
                    // extra ordering-only src on each consumer (below). Default ON;
                    // LLAMA_DSA_STORE_DEPS=0 is the negative A/B.
                    static const bool dsa_store_deps_enabled = []() {
                        const char * e = std::getenv("LLAMA_DSA_STORE_DEPS");
                        return e == nullptr || e[0] != '0';
                    }();
                    ggml_tensor * idx_k_store_dep = nullptr;

                    // Phase 3.5-lite step 3 (iter 29): persist current ubatch's indexer K into the
                    // dedicated indexer K cache (layer.k_idx), using the SAME slot positions as
                    // the main K cache (inp_attn_k->get_k_idxs()). The store op is added to the
                    // forward graph even though no node reads the cache yet — step 4/5 will
                    // consume it for score computation.
                    if (idx_k_full != nullptr && inp_attn_k != nullptr && inp_attn_k->get_k_idxs() != nullptr) {
                        idx_k_store_dep = build_sparse_store_indexer_k(idx_k_full, inp_attn_k->get_k_idxs(), il);
                        if (idx_k_store_dep != nullptr) {
                            ggml_build_forward_expand(gf, idx_k_store_dep);
                            cb(idx_k_store_dep, "indexer_k_store", il);
                        }
                    }

                    const bool sparse_decode = n_tokens == 1;
                    const bool sparse_prefill = n_tokens > 1;
                    // [DSA_SHARED_CADENCE 2026-07-06] ROOT-CAUSE FIX for GLM-5.2 DSA multi-ubatch incoherence.
                    // GLM-5.2's indexer runs a per-layer CADENCE (HF indexer_types): only "full" layers compute
                    // their own top-k; "shared" layers reuse the previous full layer's selection. The RTX ran an
                    // INDEPENDENT ggml_top_k on ALL 78 layers -> the 57 "shared" layers mis-selected keys the model
                    // was not trained for -> hidden drift accumulates -> the needle is dropped (only when
                    // n_kv>top_k; else top-k is identity so shared==fresh, which is why dense/single-ubatch/short-
                    // ctx all "worked"). Derive the cadence from freq/offset (GLM-5.2: index_topk_freq=4,
                    // index_skip_topk_offset=3 => 21 full / 57 shared over 78 layers; matches the Mac receiver's
                    // patched_shared=57) since the GGUF omits the cadence keys. FIX (Option B, env-gated
                    // LLAMA_DSA_SHARED_CADENCE default OFF): route a "shared" layer's attention to the DENSE
                    // build_attn fallback (sparse_route_enabled=false -> sparse_attn_out stays null (L402) -> dense
                    // MLA over the full causal window = a SUPERSET of the full layer's top-k, so the needle is
                    // always attended). Memory-safe (uses the KV cache directly, no gather). Follow-up Option A =
                    // shared layers REUSE the persisted full-layer top-k (keeps bounded top_k=2048 attention AND
                    // cuts the O(N^2) indexer from 78 layers to 21). Prefill only; decode + dense byte-unaffected.
                    bool dsa_shared_layer = false;
                    {
                        static const bool cad_on = []() {
                            const char * e = std::getenv("LLAMA_DSA_SHARED_CADENCE");
                            return e && e[0] && e[0] != '0';
                        }();
                        if (cad_on && sparse_prefill) {
                            static const int cad_freq = []() {
                                const char * e = std::getenv("LLAMA_DSA_TOPK_FREQ");
                                const int v = (e && e[0]) ? atoi(e) : 4; return v > 0 ? v : 4;
                            }();
                            static const int cad_offset = []() {
                                const char * e = std::getenv("LLAMA_DSA_SKIP_OFFSET");
                                return (e && e[0]) ? atoi(e) : 3;
                            }();
                            dsa_shared_layer = (std::max<int>((int) il - cad_offset + 1, 0) % cad_freq) != 0;
                            if (dsa_shared_layer) {
                                static bool cad_logged = false;
                                if (!cad_logged) { cad_logged = true;
                                    fprintf(stderr, "[DSA_SHARED_CADENCE] ACTIVE freq=%d offset=%d (il=%d is_shared=1 -> dense attn)\n",
                                            cad_freq, cad_offset, (int) il); fflush(stderr); }
                            }
                        }
                    }
                    // [OPTION A] A shared layer with a persisted full-layer top-k reuses it + runs sparse (no indexer).
                    const bool dsa_reuse_topk = dsa_shared_layer && dsa_shared_sparse_reuse &&
                        (prev_full_topk != nullptr) && sparse_prefill;
                    const bool sparse_route_enabled =
                        ((sparse_decode && sparse_mla_decode_enabled) ||
                         (sparse_prefill && sparse_mla_prefill_enabled)) && (!dsa_shared_layer || dsa_reuse_topk);

                    if (sparse_route_enabled) {
                    ggml_tensor * idx_scores = nullptr;
                    int64_t n_kv_scoring = 0;

                    ggml_tensor * idx_k_cache = nullptr;
                    if (!skip_prefill_indexer) {
                        // Iter 34k: revert to data-dependent n_kv view. The kq_mask gather
                        // requires kq_mask.ne[0] >= n_kv_scoring; n_kv_max=kv_size is too big.
                        // CPY mismatch at top_k=2048 is now prevented by the explicit F16 cast
                        // on gathered_k BEFORE reshape/permute (line ~700).
                        idx_k_cache = build_inp_indexer_k(il);
                    }
                    if (idx_k_cache != nullptr) {
                        if (idx_k_cache->ne[2] == 1) {
                            idx_k_cache = ggml_reshape_2d(ctx0, idx_k_cache, idx_k_cache->ne[0], idx_k_cache->ne[1]);
                        }
                        cb(idx_k_cache, "indexer_k_cache", il);
                        n_kv_scoring = idx_k_cache->ne[1];
                    } else if (skip_prefill_indexer && inp_attn_k != nullptr && inp_attn_k->get_kq_mask() != nullptr) {
                        n_kv_scoring = inp_attn_k->get_kq_mask()->ne[0];
                    }

                    const bool sparse_prefill_positional =
                        sparse_prefill &&
                        (sparse_mla_prefill_recent || sparse_mla_prefill_prefix) &&
                        (idx_k_cache != nullptr || (skip_prefill_indexer && n_kv_scoring > 0)) &&
                        inp_pos != nullptr &&
                        inp_attn_k != nullptr &&
                        inp_attn_k->get_kq_mask() != nullptr;
                    const bool sparse_prefill_direct_indices = sparse_prefill_positional;
                    uint32_t sparse_top_k_limit = hparams.indexer_top_k;
                    const int sparse_top_k_cap =
                        (sparse_prefill && sparse_mla_prefill_top_k_cap > 0) ?
                            sparse_mla_prefill_top_k_cap : sparse_mla_top_k_cap;
                    if (sparse_top_k_cap > 0) {
                        sparse_top_k_limit = std::min<uint32_t>(sparse_top_k_limit, (uint32_t) sparse_top_k_cap);
                    }
                    // [VARIABLE TOP-K 2026-07-07] LLAMA_DSA_TOPK_OVERRIDE sets the top-k directly and can INCREASE it
                    // beyond the model's indexer_top_k (2048) — LLAMA_MLA_SPARSE_TOP_K is a cap (only reduces). A larger
                    // k covers more of a long context so full-sparse recall improves (top-2048 misses mid/late needles
                    // at 20k -> 1/3; 4096 attends to 2x the keys, still far cheaper than dense O(n_kv)). The lambda below
                    // clamps to n_kv, so at short ctx it is a no-op. Env-gated (default 0 -> unchanged). Tunable
                    // speed/recall knob per the user's request.
                    static const int topk_override = []() {
                        const char * e = std::getenv("LLAMA_DSA_TOPK_OVERRIDE");
                        return (e && e[0]) ? atoi(e) : 0;
                    }();
                    if (topk_override > 0) {
                        sparse_top_k_limit = (uint32_t) topk_override;
                    }
                    auto sparse_top_k_for_scores = [&](int64_t n_kv_scores) -> int {
                        GGML_ASSERT(n_kv_scores > 0);
                        return (int) std::min<uint32_t>(sparse_top_k_limit, (uint32_t) n_kv_scores);
                    };
                    ggml_tensor * topk_indices_fused = nullptr;

                    if (!dsa_reuse_topk && !sparse_prefill_direct_indices && idx_scores == nullptr && idx_k_cache != nullptr) {
                    // ==== Q path (multi-head, shape [n_idx_head * idx_head_size, n_tokens]) ====
                    ggml_tensor * idx_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, q_lora_for_indexer);
                    ggml_mul_mat_set_prec(idx_q, GGML_PREC_F32);
                    cb(idx_q, "indexer_q_proj", il);

                    // Reshape-view to 3D [idx_head_size, n_idx_head, n_tokens]
                    ggml_tensor * idx_q_3d = ggml_reshape_3d(ctx0, idx_q, idx_head_size, n_idx_head, n_tokens);

                    // View rope prefix [rope_dim, n_idx_head, n_tokens] then apply RoPE
                    ggml_tensor * idx_q_rope_view = ggml_view_3d(ctx0, idx_q_3d,
                        rope_dim_idx, n_idx_head, n_tokens,
                        ggml_row_size(idx_q_3d->type, idx_head_size),
                        ggml_row_size(idx_q_3d->type, idx_head_size) * n_idx_head,
                        0);
                    ggml_tensor * idx_q_rope = ggml_rope_ext(ctx0, ggml_cont(ctx0, idx_q_rope_view),
                        inp_pos, nullptr, rope_dim_idx, indexer_rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                    cb(idx_q_rope, "indexer_q_rope", il);

                    // View non-rope tail [idx_head_size - rope_dim, n_idx_head, n_tokens]
                    ggml_tensor * idx_q_nope_view = ggml_view_3d(ctx0, idx_q_3d,
                        idx_head_size - rope_dim_idx, n_idx_head, n_tokens,
                        ggml_row_size(idx_q_3d->type, idx_head_size),
                        ggml_row_size(idx_q_3d->type, idx_head_size) * n_idx_head,
                        ggml_row_size(idx_q_3d->type, rope_dim_idx));

                    // Concat rope + nope → [idx_head_size, n_idx_head, n_tokens]
                    ggml_tensor * idx_q_full = ggml_concat(ctx0, idx_q_rope, ggml_cont(ctx0, idx_q_nope_view), 0);
                    cb(idx_q_full, "indexer_q_full", il);

                        // Phase 3.5-lite step 5 (iter 29, REVISED post-audit):
                        //   head_gates[h, q_t] = indexer_proj · cur                       [n_idx_head, n_tokens]
                        //   per_head[k_t, q_t, h] = sum_d(idx_k_cache[d, k_t] * idx_q[d, h, q_t])
                        //   per_head *= softmax_scale       (scale BEFORE relu — matters once relu is present)
                        //   per_head = relu(per_head)       (required — HF modeling_glm_moe_dsa.py
                        //                                    and sglang's fp8_index kernel both do this)
                        //   weighted = per_head * head_gates * (n_idx_head^-0.5)
                        //   final[k_t, q_t] = sum_h(weighted[k_t, q_t, h])
                        //
                        // Without the relu, negative head contributions cancel positive ones
                        // during the weighted sum → aggregated scores collapse to near-zero noise →
                        // top-k picks arbitrary positions → attractor collapse. Symptom observed:
                        // decode output repeats "the", "every", "computing" regardless of context.
                        //
                        // The old pre-aggregation trick (q_weighted = sum_h(q * gate) then
                        // scores = k · q_weighted) is only valid when the aggregation is linear.
                        // ReLU makes it non-linear, so we compute per-head scores first.

                        ggml_tensor * head_gates = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                        ggml_mul_mat_set_prec(head_gates, GGML_PREC_F32);
                        cb(head_gates, "indexer_head_gates", il);
                        head_gates = ggml_scale(ctx0, head_gates, 1.0f / sqrtf((float) n_idx_head));
                        cb(head_gates, "indexer_head_gates_scaled", il);

                        const float softmax_scale = 1.0f / sqrtf((float) idx_head_size);

                        // Iter 37: env-gated fused indexer-scoring kernel
                        //   sparse_mla_fused_kernel routes through GGML_OP_DSA_INDEXER_SCORE
                        //   which fuses (mul_mat → scale → relu → mul → sum_rows) per chunk into one
                        //   CUDA launch. Closes ~280 s of the 645 s sparse-40k op profile.
                        // Constraints (per ggml_cuda_op_dsa_indexer_score support_op): D=128,
                        // contiguous F16 k_cache + F32 q + F32 gates.
                        const bool sparse_fused_kernel = cparams.sparse_mla_fused_kernel;
                        bool sparse_fused_topk = cparams.sparse_mla_fused_topk;

                        // Phase 42: gate fused kernel on n_tokens > 1 (prefill only).
                        // Phase 37c measured ~80× faster suffix prefill (9487 vs 119 tok/s)
                        // but ~2.4× slower decode (3.5 vs 8.3 tok/s) when fused is on for
                        // n_tokens=1. The fused kernel amortizes setup over many queries;
                        // for single-query decode it has higher per-call cost than the
                        // chunked HEAD_CHUNK=16 path. This gate gives best of both.
                        const bool fused_eligible =
                            sparse_fused_kernel
                            && n_tokens > 1
                            && idx_head_size == 128
                            && idx_k_cache->type == GGML_TYPE_F16
                            && head_gates->type  == GGML_TYPE_F32;

                        // [DSA_LIGHTNING_INDEXER 2026-07-06] env-gated (default OFF) drop-in for the
                        // incoherent fused score kernel. The validated deepseek-lid-cuda
                        // GGML_OP_LIGHTNING_INDEXER computes the same
                        //   idx_scores[k,q] = sum_h relu(scale_embd * q_h . k) * w_h
                        // in ONE fused kernel (memory-safe: fuses sum_h so it never materializes the
                        // [n_kv, n_head, n_batch] score intermediate that our cublas path blows 29 GB on),
                        // and it needle-passes upstream. It produces idx_scores for the legacy (coherent)
                        // ggml_top_k path. head_gates are already pre-scaled by 1/sqrt(n_idx_head) above,
                        // so scale_heads = 1.0 and scale_embd = softmax_scale. Enable with
                        // LLAMA_MLA_SPARSE_LID=1 (prefill only; decode keeps its validated path).
                        static const bool use_lid = []() {
                            const char * e = std::getenv("LLAMA_MLA_SPARSE_LID");
                            return e != nullptr && e[0] != '0';
                        }();
                        const bool lid_eligible = use_lid
                            && n_tokens > 1
                            && idx_head_size <= 256
                            && (idx_k_cache->type == GGML_TYPE_F16 || idx_k_cache->type == GGML_TYPE_F32)
                            && head_gates->type == GGML_TYPE_F32;

                        // [LID-DIAG 2026-07-06] one-time (first DSA layer, ANY il) proof of
                        // env-propagation + eligibility, printed regardless of lid_eligible so a
                        // false gate (env not seen, wrong k-type, etc.) is visible. Resolves the
                        // "both A/B passes 2/2 HIT + no marker" ambiguity: use_lid shows whether
                        // LLAMA_MLA_SPARSE_LID reached the process; lid_eligible shows the branch taken.
                        {
                            // Skip the n_tokens==1 sched_reserve/decode calls so this shows the
                            // real streaming-prefill values (a short prompt goes PROMPT-DIRECT to the
                            // Mac and never runs this block with n_tokens>1 at all).
                            static bool lid_diag_done = false;
                            if (!lid_diag_done && n_tokens > 1 && use_lid) {
                                lid_diag_done = true;
                                fprintf(stderr, "[LID-DIAG] il=%d use_lid=%d lid_eligible=%d n_tokens=%d "
                                    "idx_head_size=%d n_idx_head=%d idx_k_type=%d head_gates_type=%d\n",
                                    (int) il, (int) use_lid, (int) lid_eligible, (int) n_tokens,
                                    (int) idx_head_size, (int) n_idx_head,
                                    (int) idx_k_cache->type, (int) head_gates->type);
                            }
                        }

                        // ==== DSA_SCORE_BUF_BOUND_TOPK_2026_06_24 (OOM fix) ====
                        // ROOT CAUSE of the ctx>=32768 OOM: ggml_dsa_indexer_score allocates a
                        // PERSISTENT graph tensor idx_scores = [n_kv, n_q]. n_kv is the padded KV
                        // fill count, which at graph-RESERVATION time is the WORST CASE (full kv_size
                        // = n_ctx), so the score buffer is sized to n_ctx regardless of the actual
                        // prompt length. At ctx=32768, ub=1024 this reserves ~7.4 GB and OOMs the
                        // 5090. LLAMA_DSA_SCORE_CUBLAS_K_SLAB only tiles the cublas GEMM *workspace*
                        // INSIDE the kernel — it does NOT shrink this graph output tensor.
                        //
                        // FIX: route the prefill fused path through ggml_dsa_indexer_topk, whose
                        // graph output is [top_k, n_q] (top_k is capped by indexer_top_k, ~2048).
                        // The full [n_kv, n_q] score matrix is still computed correctly, but only in
                        // a TRANSIENT cuda-pool scratch inside the op (reused, not in the persistent
                        // graph-reservation buffer), so the reservation no longer scales with n_ctx.
                        // top-k semantics are identical (radix top-k over the full key set == the
                        // ggml_dsa_indexer_score → ggml_top_k path). Auto-enabled only for the
                        // prefill fused path (fused_eligible already implies n_tokens>1); opt out
                        // with LLAMA_MLA_SPARSE_FUSED_TOPK_AUTO=0 to restore the legacy score buffer.
                        if (fused_eligible && !sparse_fused_topk) {
                            static const bool topk_auto = []() {
                                const char * e = std::getenv("LLAMA_MLA_SPARSE_FUSED_TOPK_AUTO");
                                return e == nullptr || (e[0] != '0');
                            }();
                            if (topk_auto) {
                                sparse_fused_topk = true;
                                if (sparse_mla_trace && il == 0) {
                                    fprintf(stderr,
                                        "[DSA_SCORE_BOUND_TOPK] prefill score buffer bounded to top_k "
                                        "(was [n_kv=%lld, n_q=%d]); set LLAMA_MLA_SPARSE_FUSED_TOPK_AUTO=0 to disable\n",
                                        (long long) n_kv_scoring, (int) n_tokens);
                                }
                            }
                        }

                        if (lid_eligible) {
                            // One-time route confirmation (env-propagation proof per CLAUDE.md:
                            // "confirm an env var took effect by runtime behavior, not launcher source").
                            // Fires on the FIRST lid_eligible layer (any il) — NOT gated on il==0,
                            // since GLM-5.2 may not run DSA on layer 0.
                            static bool lid_announced = false;
                            if (!lid_announced) {
                                lid_announced = true;
                                fprintf(stderr, "[LID] lightning-indexer path ACTIVE "
                                    "(LLAMA_MLA_SPARSE_LID=1, il=%d, idx_head_size=%d, n_idx_head=%d, n_tokens=%d)\n",
                                    (int) il, (int) idx_head_size, (int) n_idx_head, (int) n_tokens);
                            }
                            // Lightning-indexer path (validated GGML_OP_LIGHTNING_INDEXER).
                            // idx_k_cache is [D, n_kv] (may be a reshape_2d view) -> ensure contiguous,
                            // then present as [D, 1, n_kv, 1] (op expects k->ne[1]==1).
                            ggml_tensor * lid_k = idx_k_cache;
                            if (!ggml_is_contiguous(lid_k)) {
                                lid_k = ggml_cont(ctx0, lid_k);
                            }
                            lid_k = ggml_reshape_4d(ctx0, lid_k, idx_head_size, 1, n_kv_scoring, 1);
                            // idx_q_full [D, H, n_tokens] (ne[3]=1) -> cont for the op's strided reads.
                            ggml_tensor * lid_q = ggml_cont(ctx0, idx_q_full);
                            // head_gates [H, n_tokens] (already 1/sqrt(n_idx_head)-scaled) -> [H, n_tokens, 1, 1].
                            ggml_tensor * lid_w = head_gates;
                            if (!ggml_is_contiguous(lid_w)) {
                                lid_w = ggml_cont(ctx0, lid_w);
                            }
                            lid_w = ggml_reshape_4d(ctx0, lid_w, n_idx_head, n_tokens, 1, 1);
                            // scale_embd = softmax_scale; scale_heads = 1.0 (gates pre-scaled).
                            ggml_tensor * lid_score = ggml_lightning_indexer(ctx0,
                                lid_q, lid_k, lid_w, softmax_scale, 1.0f);
                            // [n_kv, n_tokens, 1, 1] -> [n_kv, n_tokens] for the downstream top-k.
                            idx_scores = ggml_reshape_2d(ctx0, lid_score, n_kv_scoring, n_tokens);
                            cb(idx_scores, "indexer_scores_lid", il);
                        } else if (fused_eligible) {
                            // idx_k_cache may have been reshape_2d'd above to [D, n_kv]; ensure contiguous.
                            ggml_tensor * idx_k_for_kernel = idx_k_cache;
                            if (!ggml_is_contiguous(idx_k_for_kernel)) {
                                idx_k_for_kernel = ggml_cont(ctx0, idx_k_for_kernel);
                            }
                            // idx_q_full: [D, H, n_tokens] but may carry stride from concat / view → cont.
                            ggml_tensor * idx_q_for_kernel = ggml_cont(ctx0, idx_q_full);
                            ggml_tensor * gates_for_kernel = head_gates;
                            if (!ggml_is_contiguous(gates_for_kernel)) {
                                gates_for_kernel = ggml_cont(ctx0, gates_for_kernel);
                            }
                            if (sparse_fused_topk) {
                                const int top_k = sparse_top_k_for_scores(n_kv_scoring);
                                // [DSA_TOPK_CAUSAL 2026-07-03] ROOT-CAUSE FIX for the DSA
                                // multi-ubatch prefill content corruption: pass the query
                                // positions so the kernel masks scores at cache cells
                                // k > pos[q] to -FLT_MAX BEFORE top-k selection. Without
                                // this the top-k budget was shared with unwritten/stale
                                // cells (zeros at layer 0; the PREVIOUS layer's K in the
                                // shared layer-major staging slab at layers 1+), which
                                // CROWDED OUT causally-valid keys — the pos-causal
                                // attention mask masked the invalid picks but could not
                                // recover the dropped valid keys (worst for early tokens;
                                // "first64 cells corrupt" signature). Prefill only:
                                // decode (n_tokens==1) keeps the legacy full-range
                                // selection (its cells beyond fill are genuinely unwritten
                                // zeros in the DEDICATED decode cache — behavior kept
                                // identical to the validated decode path). Disable for
                                // A/B with LLAMA_DSA_TOPK_CAUSAL=0.
                                static const bool topk_causal = []() {
                                    const char * e = std::getenv("LLAMA_DSA_TOPK_CAUSAL");
                                    return e == nullptr || (e[0] != '0');  // default ON
                                }();
                                ggml_tensor * topk_pos = nullptr;
                                if (topk_causal && sparse_prefill && inp_pos != nullptr &&
                                    inp_pos->type == GGML_TYPE_I32 && inp_pos->ne[0] >= n_tokens) {
                                    topk_pos = inp_pos;
                                }
                                topk_indices_fused = ggml_dsa_indexer_topk(ctx0,
                                    idx_k_for_kernel, idx_q_for_kernel, gates_for_kernel,
                                    topk_pos, softmax_scale, top_k);
                                if (dsa_store_deps_enabled && idx_k_store_dep != nullptr) {
                                    // [DSA STORE-DEPS] Ordering-only source. CUDA DSA_TOPK consumes
                                    // src[0..3]; src[4] makes the indexer-K cache write visible to
                                    // graph scheduling / node-range dispatch / stream joins.
                                    GGML_ASSERT(topk_indices_fused->src[4] == nullptr);
                                    topk_indices_fused->src[4] = idx_k_store_dep;
                                }
                                cb(topk_indices_fused, "indexer_topk_indices_fused", il);
                            } else {
                                idx_scores = ggml_dsa_indexer_score(ctx0,
                                    idx_k_for_kernel, idx_q_for_kernel, gates_for_kernel,
                                    softmax_scale);
                                if (dsa_store_deps_enabled && idx_k_store_dep != nullptr) {
                                    // [DSA STORE-DEPS] CUDA DSA_SCORE consumes src[0..2].
                                    GGML_ASSERT(idx_scores->src[4] == nullptr);
                                    idx_scores->src[4] = idx_k_store_dep;
                                }
                                cb(idx_scores, "indexer_scores_fused", il);
                            }
                        } else {
                            // HEAD-CHUNKED scoring (iter 30 fix): the all-at-once per_head materialization
                            // OOMs at 32k+ context (e.g. 32k × 32 heads × 4096 tokens × 4B = 16 GB > 32 GB VRAM
                            // alongside other tensors). Process heads in groups of HEAD_CHUNK so each chunk's
                            // peak intermediate is [n_kv, HEAD_CHUNK, n_tokens] = 2 GB per chunk at 32k ctx.
                            // Accumulate the head-weighted sum incrementally.
                            ggml_tensor * idx_q_cont = ggml_cont(ctx0, idx_q_full);
                            // idx_q_cont shape: [idx_head_size, n_idx_head, n_tokens]

                            const int64_t HEAD_CHUNK = 16;  // 2 chunks for n_idx_head=32
                            GGML_ASSERT(n_idx_head % HEAD_CHUNK == 0 && "indexer_n_head must be divisible by HEAD_CHUNK");
                            const int64_t n_chunks = n_idx_head / HEAD_CHUNK;

                            // Iter 35c: revert to HEAD_CHUNK=16 (iter 30 baseline) after per-head
                            // experiment showed no speedup. The 14x gap to 900 tok/s target is
                            // graph-overhead dominated, not from cont/permute or matmul precision.
                            ggml_tensor * scored_accum = nullptr;  // running [n_kv, n_tokens]

                            for (int64_t gidx = 0; gidx < n_chunks; ++gidx) {
                                // View the chunk's heads from idx_q_cont
                                ggml_tensor * idx_q_chunk_view = ggml_view_3d(ctx0, idx_q_cont,
                                    idx_head_size, HEAD_CHUNK, n_tokens,
                                    idx_q_cont->nb[1],           // nb1 — stride between heads
                                    idx_q_cont->nb[2],           // nb2 — stride between tokens
                                    gidx * HEAD_CHUNK * idx_q_cont->nb[1]);  // offset for this chunk's first head
                                ggml_tensor * idx_q_chunk = ggml_cont(ctx0, idx_q_chunk_view);
                                ggml_tensor * idx_q_chunk_flat = ggml_reshape_2d(ctx0, idx_q_chunk,
                                    idx_head_size, HEAD_CHUNK * n_tokens);

                                // Per-chunk per-head scores: [n_kv, HEAD_CHUNK * n_tokens]
                                ggml_tensor * chunk_scores_flat = ggml_mul_mat(ctx0, idx_k_cache, idx_q_chunk_flat);
                                ggml_mul_mat_set_prec(chunk_scores_flat, GGML_PREC_F32);
                                ggml_tensor * chunk_scores = ggml_reshape_3d(ctx0, chunk_scores_flat,
                                    n_kv_scoring, HEAD_CHUNK, n_tokens);
                                chunk_scores = ggml_scale(ctx0, chunk_scores, softmax_scale);
                                chunk_scores = ggml_relu(ctx0, chunk_scores);

                                // Slice head_gates for this chunk: [HEAD_CHUNK, n_tokens]
                                ggml_tensor * gate_chunk = ggml_view_2d(ctx0, head_gates,
                                    HEAD_CHUNK, n_tokens,
                                    head_gates->nb[1],
                                    gidx * HEAD_CHUNK * head_gates->nb[0]);
                                ggml_tensor * gate_chunk_bcast = ggml_reshape_3d(ctx0,
                                    ggml_cont(ctx0, gate_chunk),
                                    1, HEAD_CHUNK, n_tokens);

                                // weighted: [n_kv, HEAD_CHUNK, n_tokens]
                                ggml_tensor * weighted_chunk = ggml_mul(ctx0, chunk_scores, gate_chunk_bcast);
                                // Sum over HEAD_CHUNK: permute heads to ne[0], sum_rows, reshape back
                                ggml_tensor * wp = ggml_cont(ctx0,
                                    ggml_permute(ctx0, weighted_chunk, 1, 0, 2, 3));
                                ggml_tensor * chunk_sum_3d = ggml_sum_rows(ctx0, wp);
                                ggml_tensor * chunk_sum = ggml_reshape_2d(ctx0, chunk_sum_3d,
                                    n_kv_scoring, n_tokens);

                                if (scored_accum == nullptr) {
                                    scored_accum = chunk_sum;
                                } else {
                                    scored_accum = ggml_add(ctx0, scored_accum, chunk_sum);
                                }
                            }
                            // scored_accum IS the final sum_h(relu(q·k*scale)*gate) — shape [n_kv, n_tokens].
                            // The head-loop already integrates gates and sum; no further aggregation needed.
                            idx_scores = scored_accum;
                            cb(idx_scores, "indexer_scores", il);
                        }

                    }

                    if (idx_scores != nullptr || sparse_prefill_direct_indices || topk_indices_fused != nullptr || dsa_reuse_topk) {

                        // DIAGNOSTIC (iter 29): position-ramp baselines that bypass the indexer.
                        // recent baseline selects MOST RECENT positions.
                        // prefix baseline selects FIRST positions (incl. BOS).
                        // If either produces coherent output, infrastructure is correct and
                        // indexer is the bug. If neither, infra has a hidden issue. Decode only.
                        const bool sparse_recent_baseline =
                            cparams.sparse_mla_decode_baseline == LLAMA_SPARSE_MLA_SELECTOR_RECENT;
                        const bool sparse_prefix_baseline =
                            cparams.sparse_mla_decode_baseline == LLAMA_SPARSE_MLA_SELECTOR_PREFIX;
                        if (topk_indices_fused == nullptr && sparse_recent_baseline && n_tokens == 1) {
                            // score ∝ position: top_k picks tail
                            ggml_tensor * ramp = ggml_arange(ctx0, 0.0f, (float) n_kv_scoring, 1.0f);
                            idx_scores = ggml_reshape_2d(ctx0, ramp, n_kv_scoring, 1);
                            cb(idx_scores, "indexer_scores_recent_baseline", il);
                        } else if (topk_indices_fused == nullptr && sparse_prefix_baseline && n_tokens == 1) {
                            // score ∝ -position: top_k picks head (includes BOS + system prompt)
                            ggml_tensor * ramp = ggml_arange(ctx0, 0.0f, (float) n_kv_scoring, 1.0f);
                            ramp = ggml_scale(ctx0, ramp, -1.0f);
                            idx_scores = ggml_reshape_2d(ctx0, ramp, n_kv_scoring, 1);
                            cb(idx_scores, "indexer_scores_prefix_baseline", il);
                        }

                        // Phase 3.5-lite step 6a (iter 29): apply causal mask + top_k.
                        // Reuse the main-attn causal mask (inp_attn_k->get_kq_mask) — it's
                        // -inf at future positions and 0 at causally-valid ones. Shape
                        // is [n_kv_mask, n_tokens, 1, ns] (possibly padded to 256 for FA).
                        // View the first n_kv x n_tokens slice and cast to f32 to match scores.
                        const int64_t n_kv_scores = sparse_prefill_direct_indices ? n_kv_scoring :
                            (idx_scores != nullptr ? idx_scores->ne[0] : n_kv_scoring);
                        const int top_k = sparse_top_k_for_scores(n_kv_scores);
                        ggml_tensor * topk_indices = topk_indices_fused;

                        if (dsa_reuse_topk) {
                            // [OPTION A] Reuse the previous FULL cadence layer's top-k selection (positions), so this
                            // shared layer gathers ITS OWN K/V at those positions and runs sparse attention without
                            // paying for its own indexer score-GEMM + top_k. top_k above was computed from the same
                            // n_kv_scoring the full layer used (consecutive layers share the KV), so it matches.
                            topk_indices = prev_full_topk;
                        } else if (sparse_prefill_direct_indices) {
                            ggml_tensor * pos_i = inp_pos;
                            if (pos_i->ne[0] != n_tokens) {
                                pos_i = ggml_view_1d(ctx0, pos_i, n_tokens, 0);
                            }
                            ggml_tensor * pos_f = ggml_cast(ctx0, pos_i, GGML_TYPE_F32);
                            pos_f = ggml_reshape_2d(ctx0, pos_f, 1, n_tokens);
                            ggml_tensor * pos_grid = ggml_repeat_4d(ctx0, pos_f, top_k, n_tokens, 1, 1);

                            ggml_tensor * idx_f = ggml_arange(ctx0, 0.0f, (float) top_k, 1.0f);
                            idx_f = ggml_reshape_2d(ctx0, idx_f, top_k, 1);
                            idx_f = ggml_repeat_4d(ctx0, idx_f, top_k, n_tokens, 1, 1);
                            if (sparse_mla_prefill_recent) {
                                idx_f = ggml_sub(ctx0, pos_grid, idx_f);
                            }
                            idx_f = ggml_clamp(ctx0, idx_f, 0.0f, (float) (n_kv_scores - 1));
                            topk_indices = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);
                            cb(topk_indices, sparse_mla_prefill_prefix ?
                                "indexer_topk_indices_prefill_prefix" : "indexer_topk_indices_prefill_recent", il);
                        } else if (topk_indices == nullptr) {
                            // Iter 34: scoring against full kv_size with no causal mask.
                            // Cache positions beyond actual fill are zero-initialized (memset
                            // at allocation), so idx_q · 0 = 0 for those — they rank low
                            // naturally. For positions within fill but ahead of current query
                            // position, this MAY produce non-causal attention. Tradeoff for
                            // shape-stability across sched_reserve and runtime; acceptable
                            // for first iteration of full-DSA validation. Refine with a
                            // proper position-based mask after the build loop is unblocked.
                            //
                            // [DSA_TOPK_CAUSAL 2026-07-03] That refinement is now REQUIRED for
                            // prefill: the iter-34 assumptions do NOT hold in the layer-major
                            // multi-ubatch path — (a) valid keys can score <= 0 (gates can be
                            // negative) and rank BELOW thousands of zero cells; (b) at layers
                            // 1+ the shared staging slab holds the PREVIOUS layer's K at cells
                            // beyond the current write point (real, high-scoring stale data).
                            // Both crowd causally-valid keys out of the fixed top-k budget ->
                            // content corruption. Mask scores at cells > pos[q] to a huge
                            // negative before top_k (mirrors the fused-kernel fix; same
                            // LLAMA_DSA_TOPK_CAUSAL=0 A/B gate). Decode (n_tokens==1) kept
                            // unchanged, matching the validated decode path.
                            static const bool topk_causal_decomp = []() {
                                const char * e = std::getenv("LLAMA_DSA_TOPK_CAUSAL");
                                return e == nullptr || (e[0] != '0');  // default ON
                            }();
                            if (topk_causal_decomp && sparse_prefill && inp_pos != nullptr &&
                                n_tokens > 1) {
                                const int64_t n_kv_sc = idx_scores->ne[0];
                                ggml_tensor * pos_i = inp_pos;
                                if (pos_i->ne[0] != n_tokens) {
                                    pos_i = ggml_view_1d(ctx0, pos_i, n_tokens, 0);
                                }
                                ggml_tensor * pos_f = ggml_cast(ctx0, pos_i, GGML_TYPE_F32);
                                pos_f = ggml_reshape_2d(ctx0, pos_f, 1, n_tokens);
                                ggml_tensor * pos_b = ggml_repeat_4d(ctx0, pos_f, n_kv_sc, n_tokens, 1, 1);
                                ggml_tensor * ramp = ggml_arange(ctx0, 0.0f, (float) n_kv_sc, 1.0f);
                                ramp = ggml_reshape_2d(ctx0, ramp, n_kv_sc, 1);
                                ramp = ggml_repeat_4d(ctx0, ramp, n_kv_sc, n_tokens, 1, 1);
                                // valid = 1 where cell idx <= pos (diff >= 0), else 0
                                ggml_tensor * diff  = ggml_sub(ctx0, pos_b, ramp);
                                ggml_tensor * valid = ggml_step(ctx0,
                                    ggml_scale_bias(ctx0, diff, 1.0f, 0.5f));
                                // bias: 0 where valid, -3e37 (below any real score) where not
                                ggml_tensor * bias = ggml_scale_bias(ctx0, valid, 3.0e37f, -3.0e37f);
                                idx_scores = ggml_add(ctx0, idx_scores, bias);
                                cb(idx_scores, "indexer_scores_causal", il);
                            }
                            topk_indices = ggml_top_k(ctx0, idx_scores, top_k);
                            cb(topk_indices, "indexer_topk_indices", il);
                        }
                        ggml_build_forward_expand(gf, topk_indices);

                        // [TOPK_STABLE 2026-07-02] ROOT-CAUSE FIX for the DSA-on multi-ubatch
                        // q8_0 k_get_rows illegal-memory-access (7.5 TB OOB, sanitizer.log).
                        //
                        // The pos-causal-mask fix (below) discards the gathered-dense-mask chain
                        // (whose only live consumer was ggml_get_rows(mask_gather_src, topk_indices))
                        // and instead adds a NEW consumer of topk_indices via ggml_cast(reshape_2d).
                        // This changes the galloc lifetime of topk_indices: with the old mask-gather
                        // consumer pruned, topk_indices is freed EARLIER and its buffer is reused by a
                        // later tensor. In the multi-ubatch layer-major staging path, compute is split
                        // into per-layer node ranges and resolve_view_pointers() re-points every VIEW
                        // node at view_src->data at run time. The K/V gather feeds topk_indices through
                        // ggml_reshape_1d (a VIEW), so its index buffer was re-pointed at the recycled
                        // topk_indices->data -> garbage indices -> k_get_rows on the q8_0 main-K read a
                        // wild pointer. Single-ubatch uses the whole-graph fast path (no per-range
                        // re-resolution, correct galloc ordering) so it never crashed; dense has no
                        // sparse gather so it is unaffected.
                        //
                        // FIX: materialize topk_indices into an INDEPENDENT contiguous copy once, and
                        // route BOTH the gather and the pos-causal mask through it. A ggml_cont(CONT/DUP)
                        // node is NOT a view (view_src==NULL) so resolve_view_pointers leaves its data
                        // alone, and its galloc buffer is self-contained -> immune to topk_indices being
                        // freed/reused. Prefill-only (multi-stream); decode keeps the direct tensor.
                        // Identity semantics, so single-ubatch and dense are byte-unaffected.
                        ggml_tensor * topk_indices_stable = topk_indices;
                        if (sparse_prefill && topk_indices->type == GGML_TYPE_I32) {
                            topk_indices_stable = ggml_cont(ctx0, topk_indices);
                            cb(topk_indices_stable, "indexer_topk_indices_stable", il);
                            ggml_build_forward_expand(gf, topk_indices_stable);
                        }
                        // [OPTION A] Persist a FULL cadence layer's selection so the following shared layers reuse it.
                        // (Shared layers keep dsa_shared_layer==true -> they don't overwrite it; the next full layer does.)
                        if (dsa_shared_sparse_reuse && !dsa_shared_layer && sparse_prefill) {
                            prev_full_topk = topk_indices_stable;
                        }

                        // [2026-07-28 DSA TOPK-STABLE PIN, codex-v5] Layer-major repeatedly executes
                        // graph SLICES against gallocr's frozen single-pass lifetime plan; that plan
                        // recycles this CONT's allocation, so ubatch>=1's sparse FA reads recycled/
                        // unwritten pool memory through src[3] (GPU census: launches 1..88 valid,
                        // 89+ = 98.6% of the indices tensor raw float garbage -> the >3072 IMA, now
                        // sanitized to zero-attention). Same class + same fix as the April OPTION-A
                        // FIX above (compute-sanitizer-proven, gated to SHARED_SPARSE_REUSE=1 which
                        // shared-CADENCE configs never set) and the M3 wf=1 Q pin. Hold the small
                        // indices allocation for the graph lifetime whenever layer-major sparse
                        // prefill runs. Escape hatch: LLAMA_PREFILL_DSA_PIN_TOPK=0 for A/B.
                        static const bool pin_dlm_topk_stable = []() {
                            const char * lm   = std::getenv("LLAMA_PREFILL_LAYER_MAJOR");
                            const char * gate = std::getenv("LLAMA_PREFILL_DSA_PIN_TOPK");
                            return lm && lm[0] != '\0' && lm[0] != '0' &&
                                   !(gate && gate[0] == '0');
                        }();
                        const bool pin_topk_stable =
                            pin_dlm_topk_stable ||
                            (dsa_shared_sparse_reuse && !dsa_shared_layer);
                        if (sparse_prefill && pin_topk_stable) {
                            ggml_set_output(topk_indices_stable);
                        }

                        // Phase 3.5-lite step 6b/6d (iter 29): gather K/V and the matching
                        // causal mask rows by topk_indices. Decode uses one sparse stream.
                        // Prefill flattens [top_k, n_tokens] indices, gathers all rows once,
                        // then reshapes to [dim, 1, top_k, n_tokens], i.e. n_tokens
                        // independent one-query streams.
                        const int64_t sparse_pos0 = ubatch.pos ? (int64_t) ubatch.pos[0] : 0;
                        // [DSA_POS_DIAG 2026-07-06] per-ubatch check: is inp_pos the CURRENT ubatch's
                        // positions (ne0==n_tokens, offset-0 view safe) or the FULL prompt (ne0>n_tokens ->
                        // the pos-causal-mask/topk-causal ggml_view_1d(inp_pos,n_tokens,0) reads the WRONG
                        // slice for ubatch u>0 -> misaligned causal mask -> multi-ubatch garble). Prints
                        // once per ubatch at il==0 when LLAMA_DSA_POS_DIAG is set.
                        if (std::getenv("LLAMA_DSA_POS_DIAG") && il == 0) {
                            fprintf(stderr, "[DSA_POS_DIAG] ubatch il=0 n_tokens=%d inp_pos_ne0=%d sparse_pos0=%lld "
                                "offset0_view_taken=%d n_kv_scoring=%lld\n",
                                (int) n_tokens, (int) (inp_pos ? inp_pos->ne[0] : -1), (long long) sparse_pos0,
                                (int) (inp_pos && inp_pos->ne[0] != n_tokens), (long long) n_kv_scoring);
                        }
                        if (sparse_route_enabled && inp_attn_k != nullptr) {
                            ggml_build_forward_expand(gf, Qcur);
                            ggml_build_forward_expand(gf, Vcur);
                            ggml_build_forward_expand(gf, Kcur);

                            // Dense build_attn() normally stores K before reading the cache.
                            // The sparse branch bypasses dense build_attn(), so store main K
                            // explicitly before gathering top-k rows.
                            ggml_tensor * k_main_store = build_sparse_store_main_k(Kcur, inp_attn_k->get_k_idxs(), il);
                            if (k_main_store == nullptr) {
                                if (sparse_mla_trace && il < 4) {
                                    fprintf(stderr, "[sparse-mla] layer %d sparse gather skipped: main K store unavailable\n", il);
                                }
                                goto skip_gather;
                            }
                            ggml_build_forward_expand(gf, k_main_store);
                            cb(k_main_store, "indexer_main_k_store", il);

                            // Fetch main K cache as a 4D view via graph helper (keeps
                            // llama_kv_cache_context out of this TU's include set).
                            ggml_tensor * k_main_4d = build_sparse_main_k_view(il);
                            if (k_main_4d == nullptr) {
                                if (sparse_mla_trace && il < 4) {
                                    fprintf(stderr, "[sparse-mla] layer %d sparse gather skipped: main K view unavailable\n", il);
                                }
                                goto skip_gather;
                            }
                            // k_main_4d shape: [n_embd_head_k, n_head_kv=1, n_kv, ns]
                            if (k_main_4d->ne[1] != 1 || k_main_4d->ne[3] != 1) {
                                if (sparse_mla_trace && il < 4) {
                                    fprintf(stderr,
                                            "[sparse-mla] layer %d sparse gather skipped: unsupported main K shape [%lld,%lld,%lld,%lld]\n",
                                            il,
                                            (long long) k_main_4d->ne[0], (long long) k_main_4d->ne[1],
                                            (long long) k_main_4d->ne[2], (long long) k_main_4d->ne[3]);
                                }
                                goto skip_gather;
                            }

                            const int64_t sparse_streams = sparse_prefill ? n_tokens : 1;

                            // Gather only the sparse causal mask. This is small
                            // (top_k x n_tokens) and is also used by the no-K/V-gather path.
                            ggml_tensor * sparse_kq_mask = nullptr;
                            if (inp_attn_k->get_kq_mask() != nullptr) {
                                ggml_tensor * kq_mask_sparse_src = inp_attn_k->get_kq_mask();
                                const int64_t mask_stream_tokens =
                                    kq_mask_sparse_src->ne[1] *
                                    kq_mask_sparse_src->ne[2] *
                                    kq_mask_sparse_src->ne[3];
                                if (mask_stream_tokens == n_tokens && kq_mask_sparse_src->ne[0] >= n_kv_scores) {
                                    ggml_tensor * mask_gather_src = ggml_reshape_3d(ctx0, kq_mask_sparse_src,
                                        1, kq_mask_sparse_src->ne[0], n_tokens);
                                    cb(mask_gather_src, "indexer_sparse_mask_src", il);

                                    ggml_tensor * mask_gathered = ggml_get_rows(ctx0, mask_gather_src, topk_indices);
                                    cb(mask_gathered, "indexer_sparse_mask_gathered", il);

                                    sparse_kq_mask = ggml_cont(ctx0,
                                        ggml_permute(ctx0, mask_gathered, 1, 0, 3, 2));
                                    if (sparse_kq_mask->type != GGML_TYPE_F16) {
                                        sparse_kq_mask = ggml_cast(ctx0, sparse_kq_mask, GGML_TYPE_F16);
                                    }
                                    cb(sparse_kq_mask, "indexer_sparse_mask", il);

                                    // [DSA_POS_CAUSAL_MASK 2026-06-25] ROOT-CAUSE FIX for GLM-5.2 DSA-on
                                    // multi-ubatch garble. The gathered causal mask above reads the main
                                    // kq_mask at the (non-causal) top_k-selected key indices. In the
                                    // multi-ubatch KV-inject path the whole prompt K is pre-stored, so the
                                    // indexer top_k can select keys at FUTURE positions (idx > query pos);
                                    // the gathered mask values for those came back as finite garbage
                                    // (+/-6e4 as f16), not -inf -> (a) earliest tokens whose top_k held NO
                                    // causal key got an all-non-(-inf)/all-(-inf) row -> softmax NaN; (b)
                                    // valid query rows had a future key with a large POSITIVE garbage mask
                                    // -> that wrong key dominated the softmax -> retrieval degraded (needle
                                    // missed) even where it didn't NaN. Rebuild the sparse mask directly
                                    // from positions: a selected key is valid iff its cache index <= the
                                    // query position. This is exact causal masking, immune to the gather
                                    // garbage. Prefill only (n_tokens>1, streams = query tokens); decode
                                    // (n_tokens==1) keeps the gathered mask. Disable with
                                    // LLAMA_DSA_POS_CAUSAL_MASK=0.
                                    static const bool pos_causal_mask = []() {
                                        const char * e = std::getenv("LLAMA_DSA_POS_CAUSAL_MASK");
                                        return e == nullptr || (e[0] != '0');  // default ON
                                    }();
                                    if (pos_causal_mask && sparse_prefill && inp_pos != nullptr) {
                                        // pos[t] for each query token -> [1, n_tokens]
                                        ggml_tensor * pos_i = inp_pos;
                                        if (pos_i->ne[0] != n_tokens) {
                                            pos_i = ggml_view_1d(ctx0, pos_i, n_tokens, 0);
                                        }
                                        ggml_tensor * pos_f = ggml_cast(ctx0, pos_i, GGML_TYPE_F32);
                                        pos_f = ggml_reshape_2d(ctx0, pos_f, 1, n_tokens);
                                        ggml_tensor * pos_bcast = ggml_repeat_4d(ctx0, pos_f, top_k, n_tokens, 1, 1);
                                        // selected key cache indices as f32 -> [top_k, n_tokens]
                                        // [TOPK_STABLE] read the independent copy, not a view of
                                        // topk_indices (which may be freed/reused; see note at build).
                                        ggml_tensor * idx_f = ggml_cast(ctx0,
                                            ggml_reshape_2d(ctx0, topk_indices_stable, top_k, n_tokens), GGML_TYPE_F32);
                                        // diff = pos - idx ; valid (key <= query pos) iff diff >= 0
                                        ggml_tensor * diff = ggml_sub(ctx0, pos_bcast, idx_f);
                                        // step(diff + 0.5): 1 when diff >= 0 (idx <= pos), else 0
                                        // (diff is integer-valued; +0.5 puts the 0/-1 boundary mid-step)
                                        ggml_tensor * valid = ggml_step(ctx0,
                                            ggml_scale_bias(ctx0, diff, 1.0f, 0.5f));
                                        // [POSCAUSAL_MIN gate 2026-07-02] The WIP min-combine (9f03b832)
                                        // was motivated ONLY by "keep the mask_gather chain live so the
                                        // graph/scheduling doesn't shift" — a workaround for the multi-ubatch
                                        // illegal access. That crash is now root-caused elsewhere (partial-
                                        // ubatch inp_tokens tail; see llama-context.cpp
                                        // [PARTIAL_UB_TOKEN_TAIL]), so the workaround rationale is gone.
                                        // DEFAULT = REPLACE (the 2026-06-25 validated design): exact causal
                                        // mask by construction, immune to ALL gather garbage (min-combine is
                                        // only immune to future-slot garbage; garbage at a VALID slot would
                                        // wrongly suppress a real key). LLAMA_DSA_POSCAUSAL_MIN=1 re-enables
                                        // the min-combine for A/B on the KV-inject path.
                                        static const bool poscausal_min = []() {
                                            const char * e = std::getenv("LLAMA_DSA_POSCAUSAL_MIN");
                                            return e && e[0] != '\0' && e[0] != '0';  // default OFF (replace)
                                        }();
                                        if (poscausal_min) {
                                            // 0 where valid, -3e4 (~-inf for softmax, finite in f16) where
                                            // the selected key is in the FUTURE (idx > query pos).
                                            ggml_tensor * causal = ggml_scale_bias(ctx0, valid, 3.0e4f, -3.0e4f);
                                            causal = ggml_reshape_4d(ctx0, causal, top_k, 1, 1, n_tokens);
                                            if (causal->type != GGML_TYPE_F32) {
                                                causal = ggml_cast(ctx0, causal, GGML_TYPE_F32);
                                            }
                                            // min(a,b) = a - relu(a-b); ggml has no elementwise binary min.
                                            ggml_tensor * gathered_f32 = sparse_kq_mask;
                                            if (gathered_f32->type != GGML_TYPE_F32) {
                                                gathered_f32 = ggml_cast(ctx0, gathered_f32, GGML_TYPE_F32);
                                            }
                                            ggml_tensor * amb = ggml_sub(ctx0, gathered_f32, causal);   // a - b
                                            ggml_tensor * mn  = ggml_sub(ctx0, gathered_f32,
                                                                         ggml_relu(ctx0, amb));         // a - relu(a-b)
                                            if (mn->type != GGML_TYPE_F16) {
                                                mn = ggml_cast(ctx0, mn, GGML_TYPE_F16);
                                            }
                                            sparse_kq_mask = mn;
                                        } else {
                                            // REPLACE: mask = valid*BIG - BIG : 0 where valid, -BIG (~-inf)
                                            // where not (original 2026-06-25 constants; -1e30 saturates to
                                            // -inf in the f16 cast, matching the validated single-ubatch path).
                                            ggml_tensor * causal = ggml_scale_bias(ctx0, valid, 1e30f, -1e30f);
                                            causal = ggml_reshape_4d(ctx0, causal, top_k, 1, 1, n_tokens);
                                            if (causal->type != GGML_TYPE_F16) {
                                                causal = ggml_cast(ctx0, causal, GGML_TYPE_F16);
                                            }
                                            sparse_kq_mask = causal;
                                        }
                                        cb(sparse_kq_mask, "indexer_sparse_mask_poscausal", il);
                                    }

                                    // [DIAG_SPARSE_NAN 2026-06-25] env-gated: name the sparse mask +
                                    // the topk indices so the cb_eval LLAMA_LAYER_CMP_DUMP hook can capture them.
                                    {
                                        static const bool diag_sn = []() {
                                            const char * e = std::getenv("LLAMA_DIAG_SPARSE_NAN");
                                            return e && e[0] != '\0' && e[0] != '0';
                                        }();
                                        if (diag_sn && n_tokens > 1) {
                                            char dn[64];
                                            snprintf(dn, sizeof(dn), "diagmask-%d", il);  ggml_set_name(sparse_kq_mask, dn);
                                            snprintf(dn, sizeof(dn), "diagtopk-%d", il);  ggml_set_name(topk_indices, dn);
                                            // [2026-07-06] also dump the top_k INPUT (causal-biased idx_scores) to
                                            // disambiguate: garbage idx_scores for ub>0 => scoring/galloc bug;
                                            // valid idx_scores + garbage topk => ggml_top_k output buffer reuse.
                                            if (idx_scores) { snprintf(dn, sizeof(dn), "diagscore-%d", il); ggml_set_name(idx_scores, dn); }
                                        }
                                    }
                                }
                            }

                            if (sparse_prefill && sparse_kq_mask == nullptr) {
                                if (sparse_mla_trace && il < 4) {
                                    fprintf(stderr,
                                            "[sparse-mla] layer %d prefill gather skipped: sparse mask unavailable (n_tokens=%lld top_k=%d)\n",
                                            il, (long long) n_tokens, top_k);
                                }
                                goto skip_gather;
                            }

                            // [FA_DECOUPLE 2026-07-07] The sparse fused-FA path (ggml_flash_attn_ext_sparse)
                            // is a DISTINCT GGML op (GGML_OP_FLASH_ATTN_EXT_SPARSE) with its own
                            // always-compiled CUDA kernel + supports_op (ggml-cuda.cu:6098 = tensor
                            // types/shapes ONLY, no cparams.flash_attn dependency). It builds k_main_4d
                            // fresh from the indexer and its own sparse_kq_mask, and attends over exactly
                            // top_k(=2048, 256-aligned) gathered keys — so it needs NONE of the dense-FA
                            // KV-cache machinery that cparams.flash_attn (=-fa on) sets up.
                            //
                            // The server runs -fa OFF for the DENSE MLA path (the Phase22 multi-ubatch dense
                            // FA bug), and that ALSO killed THIS sparse path: cparams.flash_attn==false ->
                            // sparse_fa_indices_eligible==false -> fell through to the unfused
                            // gather+build_attn_mha (materialized QK/softmax/AV — the 53% MUL_MAT + SOFT_MAX
                            // in the op profile, and the reason sparse loses <32k). Note the gate also
                            // requires Qcur F32, which is the -fa-OFF state (FA-on casts Q to F16) — so the
                            // cparams.flash_attn clause was self-contradictory here.
                            //
                            // Decouple: gate the sparse fused-FA on its OWN opt-in, independent of -fa.
                            // Env-gated (LLAMA_MLA_SPARSE_FA_DECOUPLE=1) so it A/Bs against the current
                            // unfused path in ONE build. If any real tensor condition fails, it still falls
                            // through to the unfused path (no regression).
                            static const bool sparse_fa_decouple_cparams = []() {
                                const char * e = std::getenv("LLAMA_MLA_SPARSE_FA_DECOUPLE");
                                return e && e[0] != '\0' && e[0] != '0';
                            }();
                            const bool sparse_fa_flash_gate = cparams.flash_attn || sparse_fa_decouple_cparams;

                            const bool sparse_fa_indices_eligible =
                                (sparse_mla_fa_with_indices || sparse_mla_fa_v5 || sparse_mla_fa_v6) &&
                                sparse_fa_flash_gate &&
                                Qcur->type == GGML_TYPE_F32 &&
                                (k_main_4d->type == GGML_TYPE_F16 || k_main_4d->type == GGML_TYPE_Q8_0) &&
                                topk_indices->type == GGML_TYPE_I32 &&
                                (sparse_kq_mask == nullptr || sparse_kq_mask->type == GGML_TYPE_F16) &&
                                (!sparse_mla_fa_v6 || sparse_kq_mask != nullptr) &&
                                Qcur->ne[0] == k_main_4d->ne[0] &&
                                Vcur->ne[0] <= k_main_4d->ne[0] &&
                                topk_indices->ne[0] == top_k &&
                                topk_indices->ne[1] >= sparse_streams;

                            // [FA_DECOUPLE_DIAG] One-time (first n_tokens>1 sparse layer) breakdown of the
                            // eligibility gate so the A/B is conclusive in one run: shows WHICH sub-condition
                            // blocks the fused path when it does not fire. Gated on sparse_mla_trace.
                            if (sparse_mla_trace && n_tokens > 1) {
                                // Z1b needs this per CADENCE CLASS, not once: the whole question is whether the fused
                                // path engages on SHARED layers (57 of 78) as well as FULL ones (21). A single static
                                // flag only ever showed one layer, which cannot answer it either way.
                                static bool fa_elig_diag_full = false, fa_elig_diag_shared = false;
                                bool * fa_elig_diag_flag = dsa_shared_layer ? &fa_elig_diag_shared : &fa_elig_diag_full;
                                if (!*fa_elig_diag_flag) {
                                    *fa_elig_diag_flag = true;
                                    fprintf(stderr,
                                        "[FA_DECOUPLE_DIAG] il=%d shared=%d eligible=%d | fa_v6=%d fa_v5=%d fa_idx=%d "
                                        "cparams_fa=%d decouple=%d gate=%d | Qf32=%d ktype_ok=%d topk_i32=%d "
                                        "mask_ok=%d v6_needs_mask=%d Qd_eq_Kd=%d Vd_le_Kd=%d topk0_eq=%d topk1_ge=%d "
                                        "| top_k=%d n_kv=%lld streams=%lld\n",
                                        (int) il, (int) dsa_shared_layer, (int) sparse_fa_indices_eligible,
                                        (int) sparse_mla_fa_v6, (int) sparse_mla_fa_v5, (int) sparse_mla_fa_with_indices,
                                        (int) cparams.flash_attn, (int) sparse_fa_decouple_cparams, (int) sparse_fa_flash_gate,
                                        (int) (Qcur->type == GGML_TYPE_F32),
                                        (int) (k_main_4d->type == GGML_TYPE_F16 || k_main_4d->type == GGML_TYPE_Q8_0),
                                        (int) (topk_indices->type == GGML_TYPE_I32),
                                        (int) (sparse_kq_mask == nullptr || sparse_kq_mask->type == GGML_TYPE_F16),
                                        (int) (!sparse_mla_fa_v6 || sparse_kq_mask != nullptr),
                                        (int) (Qcur->ne[0] == k_main_4d->ne[0]),
                                        (int) (Vcur->ne[0] <= k_main_4d->ne[0]),
                                        (int) (topk_indices->ne[0] == top_k),
                                        (int) (topk_indices->ne[1] >= sparse_streams),
                                        (int) top_k, (long long) k_main_4d->ne[2], (long long) sparse_streams);
                                }
                            }

                            if (sparse_fa_indices_eligible) {
                                ggml_tensor * v_main_4d = ggml_view_4d(ctx0, k_main_4d,
                                    Vcur->ne[0], 1, k_main_4d->ne[2], 1,
                                    k_main_4d->nb[1],
                                    k_main_4d->nb[2],
                                    k_main_4d->nb[3],
                                    0);
                                cb(v_main_4d, "indexer_sparse_v_full", il);

                                sparse_attn_out = ggml_flash_attn_ext_sparse(ctx0,
                                    Qcur, k_main_4d, v_main_4d, topk_indices_stable, sparse_kq_mask,
                                    kq_scale, top_k, 0.0f);
                                if (glm_dsa_deterministic_sparse_fa) {
                                    ggml_flash_attn_ext_sparse_set_deterministic(
                                        sparse_attn_out, true);
                                }
                                if (dsa_store_deps_enabled) {
                                    // [DSA STORE-DEPS] Ordering-only source. Sparse FA consumes
                                    // src[0..4]; src[5] forces the main-K SET_ROWS store to complete
                                    // before the independently-constructed cache view is gathered.
                                    GGML_ASSERT(sparse_attn_out->src[5] == nullptr);
                                    sparse_attn_out->src[5] = k_main_store;
                                }
                                // [GLM_TOPK_WRITE_COVERAGE_V1]
                                //
                                // Refuted diagnostic only: retaining the stable CONT's direct
                                // source and re-materializing it at the consumer made the census
                                // complete but corrupted every tested prompt size. Keep src[6]
                                // absent by default; =1 explicitly restores that old experiment.
                                static const bool glm_topk_write_coverage = []() {
                                    const char * e =
                                        std::getenv("LLAMA_PREFILL_GLM_TOPK_WRITE_COVERAGE");
                                    return e && e[0] != '\0' && e[0] != '0';
                                }();
                                if (glm_topk_write_coverage &&
                                        topk_indices_stable->src[0] != nullptr) {
                                    GGML_ASSERT(sparse_attn_out->src[6] == nullptr);
                                    sparse_attn_out->src[6] =
                                        topk_indices_stable->src[0];
                                }
                                cb(sparse_attn_out, sparse_mla_fa_v6 ? "kqv_out_sparse_indices_v6" :
                                    (sparse_mla_fa_v5 ? "kqv_out_sparse_indices_v5" : "kqv_out_sparse_indices"), il);

                                if (model.layers[il].wv_b) {
                                    sparse_attn_out = ggml_permute(ctx0, sparse_attn_out, 0, 2, 1, 3);
                                    sparse_attn_out = ggml_mul_mat(ctx0, model.layers[il].wv_b, sparse_attn_out);
                                    cb(sparse_attn_out, "kqv_out_sparse_indices_mla", il);
                                    sparse_attn_out = ggml_permute(ctx0, sparse_attn_out, 0, 2, 1, 3);
                                    sparse_attn_out = ggml_cont(ctx0, sparse_attn_out);
                                }

                                sparse_attn_out = ggml_reshape_2d(ctx0, sparse_attn_out,
                                    sparse_attn_out->ne[0]*sparse_attn_out->ne[1],
                                    sparse_attn_out->ne[2]*sparse_attn_out->ne[3]);

                                if (model.layers[il].wo) {
                                    sparse_attn_out = build_lora_mm(model.layers[il].wo, sparse_attn_out);
                                }

                                if (sparse_mla_trace && il < 4) {
                                    fprintf(stderr,
                                            "[sparse-mla] layer %d %s sparse-fa-indices wired: n_kv=%lld top_k=%d streams=%lld pos0=%lld k_dim=%lld v_dim=%lld\n",
                                            il, sparse_prefill ? "prefill" : "decode",
                                            (long long) k_main_4d->ne[2], top_k,
                                            (long long) sparse_streams, (long long) sparse_pos0,
                                            (long long) k_main_4d->ne[0], (long long) v_main_4d->ne[0]);
                                }
                                goto skip_gather;
                            }

                            // Reshape to [n_embd_head_k, n_kv]
                            ggml_tensor * k_main_2d = ggml_reshape_2d(ctx0, k_main_4d,
                                k_main_4d->ne[0], k_main_4d->ne[2]);
                            cb(k_main_2d, "indexer_gather_k_src", il);

                            // topk_indices is [top_k, n_tokens]; flatten to gather all sparse
                            // rows in query-major order, then restore query streams below.
                            // [TOPK_STABLE] flatten the independent copy (prefill) so the get_rows
                            // index buffer cannot be corrupted by topk_indices being freed/reused.
                            ggml_tensor * topk_1d = ggml_reshape_1d(ctx0, topk_indices_stable, top_k * sparse_streams);

                            ggml_tensor * gathered_k = ggml_get_rows(ctx0, k_main_2d, topk_1d);
                            // gathered_k shape: [n_embd_head_k, top_k * sparse_streams]
                            cb(gathered_k, "indexer_gathered_k", il);

                            // Iter 34: cast F32→F16 BEFORE reshape/permute. ggml_get_rows
                            // returns F32 (ggml.c:3801 TODO). build_attn_mha would otherwise
                            // cast on the permuted view, which produced the original CPY
                            // mismatch (cpy.cu:384 src_ne[1]=top_k vs dst_ne[1]=ubatch).
                            if (gathered_k->type == GGML_TYPE_F32) {
                                gathered_k = ggml_cast(ctx0, gathered_k, GGML_TYPE_F16);
                                cb(gathered_k, "indexer_gathered_k_f16", il);
                            }

                            // V cache for MLA is a view into K (first n_embd_head_v dims) — reuse
                            // same gather. See llama-graph.cpp build_attn line 2347:
                            //   ggml_view_4d(k, v_cur->ne[0], k->ne[1..3], ...) with offset 0.
                            // Apply same semantics to gathered_k.
                            ggml_tensor * gathered_v = ggml_view_2d(ctx0, gathered_k,
                                Vcur->ne[0], top_k * sparse_streams,
                                gathered_k->nb[1], 0);
                            cb(gathered_v, "indexer_gathered_v", il);

                            // Layout mirrors dense MLA K cache per llama-kv-cache.cpp:1319-1324:
                            // [n_embd_head_k, n_head_kv, n_kv, ns] — ne[1]=H_KV (heads), ne[2]=L_K.
                            // build_attn_mha's permute(0,2,1,3) swaps dim 1 and 2 before FA, so FA
                            // sees ne[1]=L_K, ne[2]=H_KV. ggml_can_mul_mat check then requires
                            // q.ne[2] % k.ne[2] == 0, i.e. H_Q % H_KV == 0 — only satisfiable when
                            // H_KV divides H_Q (here H_KV=1 for MLA).
                            //
                            // Iter 31 tried dim-swap ([D, top_k, 1, streams] pre-permute) which
                            // broke this check because post-permute ne[2]=top_k instead of 1.
                            // Reverted: pre-permute keeps H_KV=1 in dim 1, top_k in dim 2.
                            ggml_tensor * sparse_k = ggml_reshape_4d(ctx0, gathered_k,
                                gathered_k->ne[0], 1, top_k, sparse_streams);
                            cb(sparse_k, "indexer_sparse_k", il);

                            ggml_tensor * sparse_v = ggml_view_4d(ctx0, gathered_k,
                                gathered_v->ne[0], 1, top_k, sparse_streams,
                                ggml_row_size(gathered_k->type, gathered_v->ne[0]),
                                gathered_k->nb[1],
                                gathered_k->nb[1] * top_k,
                                0);
                            cb(sparse_v, "indexer_sparse_v", il);

                            sparse_attn_out = build_attn_mha(Qcur, sparse_k, sparse_v,
                                nullptr, sparse_kq_mask, nullptr, model.layers[il].wv_b, kq_scale, il);
                            cb(sparse_attn_out, "kqv_out_sparse", il);

                            if (model.layers[il].wo) {
                                sparse_attn_out = build_lora_mm(model.layers[il].wo, sparse_attn_out);
                            }

                            if (sparse_mla_trace && il < 4) {
                                fprintf(stderr,
                                        "[sparse-mla] layer %d %s gather wired: n_kv=%lld top_k=%d streams=%lld pos0=%lld k_dim=%lld v_dim=%lld\n",
                                        il, sparse_prefill ? "prefill" : "decode",
                                        (long long) k_main_4d->ne[2], top_k,
                                        (long long) sparse_streams, (long long) sparse_pos0,
                                        (long long) sparse_k->ne[0], (long long) sparse_v->ne[0]);
                            }
                        } else if (sparse_mla_trace && il < 4 && n_tokens > 1) {
                            fprintf(stderr,
                                    "[sparse-mla] layer %d prefill gather skipped: n_tokens=%lld n_seqs=%u n_seq_tokens=%u pos0=%lld top_k=%d\n",
                                    il, (long long) n_tokens, ubatch.n_seqs, ubatch.n_seq_tokens,
                                    (long long) sparse_pos0, top_k);
                        }
                        skip_gather:;
                    }
                    }

                    // Keep explicit references for sparse indexer intermediates that are consumed
                    // indirectly by the top-k/gather branch above.
                    (void) idx_k_full;
                }

                // note: MLA with the absorption optimzation converts into MQA (ie: GQA with 1 group)
                cur = sparse_attn_out != nullptr ? sparse_attn_out : build_attn(inp_attn_k,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
                // Semantic diagnostic frontier: the final attention output
                // projection immediately before the attention residual add.
                // Naming is behavior-neutral; readback remains gated by
                // LLAMA_DLM_SUBLAYER_FNV.
                cb(cur, "attn_out_proj", il);
            } else {
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                cb(kv, "kv", il);

                // split into {n_embd_head_qk_nope, n_head, n_tokens}
                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
                cb(k_nope, "k_nope_view", il);

                // and {n_embd_head_v, n_head, n_tokens}
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope));
                cb(Vcur, "Vcur_view", il);

                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "Vcur_cont", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(Kcur, "Kcur", il);

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // note: MLA without the absorption optimization converts into MHA (ie: GQA with full n_head groups)
                cur = build_attn(inp_attn_kv,
                            model.layers[il].wo, NULL,
                            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }
        }
post_attn:
        // Split-decode attn-only path: stop here and expose cur as the per-layer result.
        // This is the RTX-side path — compute attention, return attn_out, skip FFN.
        if (params.stop_after_attn) {
            cb(cur, "attn_out", il);
            inpL = cur;
            continue;  // next layer — the FFN path is skipped for all layers in this graph
        }
        if (il == effective_n_layers - 1 && inp_out_ids) {
            // 2026-04-21 (iter 10): In the mega-dispatch path, cur and inpSA
            // are already external [n_embd, 1] tensors (attn_wraps[il] and
            // in_wraps[il]) containing exactly one row each.  get_rows with
            // inp_out_ids=[0] is a mathematical no-op but it introduces two
            // extra graph nodes that gallocr aliases to the inp_out_ids slot
            // at offset 0 of the compute buffer.  The aliasing chain caused:
            //   iter 7: ffn_norm src0 → offset 0 (reads I32 index as float)
            //   iter 8: ggml_cont workaround shifted the issue (still wrong)
            //   iter 9: fillBuffer fix zeroed offset 0 → mhb[78] non-zero
            //           but l_out-77 magnitude wrong (missing inpSA residual)
            // Root fix: skip get_rows entirely for external per-layer tensors.
            // The cont workaround is also removed (no get_rows, no aliasing).
            if (params.external_attn_out_per_layer == nullptr) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }
            // (no ggml_cont needed — we either skipped get_rows above, or
            // the per-call single-layer graph has no gallocr slot pressure)
        }
        // DIAG (2026-05-29): dump the L1 attention output (cur, BEFORE the residual
        // add) to split the multi-ubatch L1 divergence into attention-compute (cur
        // differs) vs residual/aliasing (cur matches single-ub but ffn_inp differs).
        // Gated by LLAMA_DUMP_LAYER; diagnostic only. NOTE: set_output(cur) keeps the
        // tensor — if this alone fixes the bug, the cause is buffer-reuse of cur.
        {
            const char * dl_env = std::getenv("LLAMA_DUMP_LAYER");
            if (dl_env && dl_env[0]) {
                const int dl = atoi(dl_env);
                if (il == dl || il == dl - 1) {
                    ggml_format_name(cur, "attn_out-%d", il);
                    ggml_set_output(cur);
                }
            }
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // DIAG (2026-05-28): persist ffn_inp (post-attention residual) at the dump
        // layer so it survives allocation and can be checksummed. Lets us split the
        // multi-ubatch L1 divergence into attention (ffn_inp differs) vs MoE-FFN
        // (ffn_inp identical, l_out differs). Gated by LLAMA_DUMP_LAYER; diagnostic only.
        {
            const char * dl_env = std::getenv("LLAMA_DUMP_LAYER");
            if (dl_env && dl_env[0]) {
                const int dl = atoi(dl_env);
                if (il == dl || il == dl - 1) {
                    ggml_set_output(ffn_inp);
                }
            }
        }

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            const int64_t n_expert_used_eff =
                (n_tokens > 1 && moe_prefill_expert_cap > 0) ?
                    std::max<int64_t>(1, std::min<int64_t>((int64_t) n_expert_used, (int64_t) moe_prefill_expert_cap)) :
                    (int64_t) n_expert_used;
            // MoE branch — build_moe_ffn handles per-group dispatch internally
            // when cparams.n_expert_splits > 1. Shared expert must be computed
            // BEFORE MoE when splitting (its weights are in sub-layer 0).
            if (cparams.n_expert_splits > 1) {
                // Compute shared expert first (uses shexp weights in slice 0)
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used_eff,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    model.layers[il].ffn_gate_up_exps);
                cb(moe_out, "ffn_moe_out", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used_eff,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    model.layers[il].ffn_gate_up_exps);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp =
                        build_ffn(cur,
                            model.layers[il].ffn_up_shexp, NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }
        }
        if (ggml_tensor * sidecars = build_token_sidecars(ffn_inp, il)) {
            // Additive routed experts do not consume a base-router slot: GLM
            // still selects its normal top-8 MoE experts, while each sidecar's
            // learned sigmoid gate controls its own strength per token.
            cur = ggml_add(ctx0, cur, sidecars);
            cb(cur, "ffn_token_sidecars", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Mega-graph path: copy this layer's output into the caller's
        // shared per-layer output buffer so the network TX thread can
        // read it once the GPU reaches this point.
        if (params.external_output_per_layer != nullptr) {
            ggml_tensor * out_dst = params.external_output_per_layer[il - il_start];
            if (out_dst != nullptr) {
                // 2026-04-20: Mark layer 77's `cur` as a graph OUTPUT so
                // ggml_backend_sched doesn't reuse its compute-buffer slot
                // for downstream post-loop nodes (result_norm, result_output)
                // BEFORE the cpy has read from it. Diagnostics showed that
                // without this, layer 77's compute slot gets overwritten by
                // result_norm's intermediate and the cpy reads zeros.
                // Marking ALL layers' cur as output is also fine — it just
                // prevents compute-buffer reuse for those slots, at the cost
                // of slightly larger working set.
                ggml_set_output(cur);
                ggml_tensor * copied = ggml_cpy(ctx0, cur, out_dst);
                cb(copied, "l_out_external_pl", il);
                ggml_build_forward_expand(gf, copied);
                // Use the (unchanged value of) cur for next layer's input;
                // the copy is a side-effect write, not a data-flow dependency.
            }
        }

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // For per-layer graphs that don't include the last layer,
    // skip output processing (norm + lm_head). Just expand the
    // last layer's output as the graph result.
    // Also skip when stop_after_attn is set — in that case the last layer's
    // attention output is what we want to return, not the post-norm + LM head.
    // Also skip when external_attn_out is set (FFN_ONLY split mode) — the
    // caller will invoke apply_output_head separately and would double-apply
    // the norm otherwise.
    if (il_end >= effective_n_layers && !params.stop_after_attn && params.external_attn_out == nullptr) {
        const bool output_from_mtp =
            mtp_enabled &&
            effective_n_layers > 0 &&
            llama_mtp_is_nextn_layer(effective_n_layers - 1, total_n_layers, nextn_layers);
        const auto & output_nextn = output_from_mtp ?
            model.layers[effective_n_layers - 1].nextn :
            model.layers[0].nextn;
        ggml_tensor * output_norm = output_from_mtp && output_nextn.shared_head_norm != nullptr ?
            output_nextn.shared_head_norm :
            model.output_norm;
        ggml_tensor * output_head = output_from_mtp && output_nextn.shared_head_head != nullptr ?
            output_nextn.shared_head_head :
            model.output;

        cur = build_norm(cur, output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = ggml_mul_mat(ctx0, output_head, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    } else {
        // Per-layer: the "result" is the layer output (l_out)
        res->t_embd = cur;
        res->t_logits = nullptr;
    }

    ggml_build_forward_expand(gf, cur);
}
