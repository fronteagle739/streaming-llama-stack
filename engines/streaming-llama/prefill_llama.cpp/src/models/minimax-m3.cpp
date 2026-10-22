#include "models.h"

#include "../llama-mtp-policy.h"

// MiniMax-M3, text-only: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling) and
// swigluoai activation. Vision tower and MTP heads are dropped (text-only conversion).

llm_build_minimax_m3::llm_build_minimax_m3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

    // swigluoai params, shared by dense and shared-expert FFNs
    const float swiglu_alpha = 1.702f;
    const float swiglu_limit = 7.0f;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // NextN / MTP draft graph (DeepSeek-V3-style). ALL MTP handling below is gated on mtp_enabled, so the
    // normal forward AND the streaming-prefill path are byte-identical (il runs 0..n_layer, inpL = token
    // embeddings, output = the main head). Only the DRAFT role (set by llama_context::mtp_draft_*) branches.
    const int32_t total_n_layers = (int32_t) n_layer;
    const int32_t nextn_layers   = (int32_t) hparams.nextn_predict_layers;
    const bool    mtp_enabled    = llama_mtp_graph_uses_nextn(cparams.mtp_mode, params.mtp_graph_role) && nextn_layers > 0;

    ggml_tensor * inp_token_embd = build_inp_embd(model.tok_embd);
    inpL = (mtp_enabled && params.external_input) ? params.external_input : inp_token_embd;

    ggml_tensor * inp_pos = build_inp_pos();
    auto inp_attn = build_attn_inp_kv();
    // Only create inp_out_ids when n_outputs > 0.  When n_outputs == 0
    // (intermediate ubatches in layer-major mode), build_inp_out_ids()
    // returns a zero-element I32 tensor that gallocr aliases at offset 0
    // of the compute buffer.  set_inputs() then writes an OOB index into
    // that aliased memory, corrupting MoE routing tensors.
    ggml_tensor * inp_out_ids = (n_outputs > 0) ? build_inp_out_ids() : nullptr;

    // effective_n_layers EXCLUDES the nextn/MTP layers from the MAIN graph (they have no main-model KV:
    // n_layer_kv_from_start = n_layer - nextn). For a normal GGUF (nextn=0) it == n_layer, so the normal and
    // streaming paths are unchanged. The DRAFT role additionally clamps to [il_start,il_end) (its own layer).
    const int effective_n_layers = llama_mtp_effective_layer_count(total_n_layers, nextn_layers, cparams.mtp_mode, params.mtp_graph_role);
    const int il_start = mtp_enabled ? (params.layer_start >= 0 ? params.layer_start : 0) : 0;
    const int il_end   = mtp_enabled
        ? llama_mtp_clamp_layer_end_for_role(params.layer_end, total_n_layers, nextn_layers, cparams.mtp_mode, params.mtp_graph_role)
        : effective_n_layers;
    for (int il = 0; il < effective_n_layers; ++il) {
        if (mtp_enabled && (il < il_start || il >= il_end)) continue; // DRAFT builds only its own layer(s)
        ggml_tensor * inpSA = inpL;

        // MTP draft prelude (gated -> the normal/streaming path skips it): combine the normalized NEXT-token
        // embedding with the normalized PREVIOUS hidden state via eh_proj (mirrors deepseek2.cpp:192-204 +
        // vllm MiniMaxM3MultiTokenPredictorLayer), then the regular M3 decoder layer runs on the result.
        if (mtp_enabled && llama_mtp_is_nextn_layer(il, total_n_layers, nextn_layers)) {
            const auto & nextn = model.layers[il].nextn;
            // vllm/MLX mask the position-0 token embedding (no previous token to condition on there): zero
            // inp_token_embd where pos==0 before enorm. pos_keep[t] = (pos[t] >= 1).
            ggml_tensor * pos_keep = ggml_step(ctx0, ggml_add(ctx0,
                ggml_cast(ctx0, inp_pos, GGML_TYPE_F32),
                ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens), -0.5f)));
            ggml_tensor * emb_in      = ggml_mul(ctx0, inp_token_embd, ggml_reshape_2d(ctx0, pos_keep, 1, n_tokens));
            ggml_tensor * emb_norm    = build_norm(emb_in, nextn.enorm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * hidden_norm = build_norm(inpL,   nextn.hnorm, nullptr, LLM_NORM_RMS, il);
            cur = ggml_mul_mat(ctx0, nextn.eh_proj, ggml_concat(ctx0, emb_norm, hidden_norm, 0));
            cb(cur, "mtp_eh_proj", il);
            inpL  = cur;
            inpSA = cur;
        }

        // self-attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // per-head QK RMSNorm (a single head_dim vector applied to every head)
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // partial rotary: only the first n_rot dims are rotated
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);

            // [2026-07-28 M3 wf=1 TARGETED FIX — gallocr reuse-collision on L1 Qcur]
            // Layer-major executes graph slices repeatedly using a gallocr lifetime plan derived
            // from one full, in-order graph evaluation. With ONE WF slab (LLAMA_WF_GPU_BUFS=1) that
            // allocation layout lets layer 1's final Qcur slot be recycled; after a partial
            // 256-token pass its upper query half can remain unwritten. The MMA F16 FA kernel then
            // reads that logical Q region as float2 (fattn-mma-f16.cuh:1322; compute-sanitizer
            // initcheck-verified) and propagates NaN through tokens 256..511 -> garbage decode.
            // Pin the proven first-fault tensor against gallocr reuse (allocation-only; one extra
            // L1 Q allocation, no memset/sync/per-token cost — same mechanism as PIN_MASK).
            // Cheap replacement for global GGML_ALLOC_NO_FREE_REUSE (which also fixes it but costs
            // full-graph no-reuse VRAM). LLAMA_PREFILL_M3_WF1_PIN_Q=0 restores the old layout (A/B).
            {
                static const bool pin_wf1_l1_q = [] {
                    const char * lm   = std::getenv("LLAMA_PREFILL_LAYER_MAJOR");
                    const char * wf   = std::getenv("LLAMA_WF_GPU_BUFS");
                    const char * gate = std::getenv("LLAMA_PREFILL_M3_WF1_PIN_Q");
                    return lm && lm[0] != '\0' && lm[0] != '0' &&
                           wf && std::atoi(wf) == 1 &&
                           !(gate && gate[0] == '0');
                }();
                if (pin_wf1_l1_q && cparams.flash_attn && n_tokens > 256 && il == 1) {
                    ggml_set_output(Qcur);
                }
            }

            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // §14a block-sparse (MSA) routing: sparse layers (il >= leading-dense, index tensors
            // present) go through build_attn_m3_sparse when LLAMA_M3_SPARSE is set. STEP 1 passes
            // block_mask = nullptr, which is numerically identical to dense build_attn (validates the
            // routing + the new method's KV-cache plumbing via the coherence gate). STEP 2 computes
            // block_mask from the index branch (index_q/k proj + GemmaRMS norm + partial RoPE ->
            // block-max-pool -> top-k + init/local -> additive 0/-inf mask).
            static const bool m3_sparse_env = [] {
                const char * e = std::getenv("LLAMA_M3_SPARSE");
                return e && e[0] != '\0' && e[0] != '0';
            }();
            const bool m3_sparse_layer = m3_sparse_env
                && il >= (int) hparams.n_layer_dense_lead
                && model.layers[il].attn_index_q != nullptr;
            if (m3_sparse_layer) {
                // === §14a STEP 2: build the block-sparse mask from the index branch ===
                // Produces an additive mask [n_kv, n_q] (0 = selected block, -1e30 = pruned) that
                // build_attn_m3_sparse folds into the causal kq_mask. At full block coverage
                // (topk*block_size >= seqlen) every block is selected -> mask all-zero -> == dense.
                const int64_t idx_dim   = (int64_t) hparams.indexer_head_size;
                const int64_t idx_heads = (int64_t) hparams.indexer_n_head;
                const int64_t bs        = (int64_t) hparams.sparse_block_size;
                // testing overrides (LLAMA_M3_{TOPK,INIT_BLOCKS,LOCAL_BLOCKS}) let the synthetic gate
                // exercise the init/local-block FORCE without regenerating the gguf; <0 => use the hparam.
                static const int64_t topk_ov  = [] { const char * e = std::getenv("LLAMA_M3_TOPK");         return e ? (int64_t) atoll(e) : (int64_t) -1; }();
                static const int64_t init_ov  = [] { const char * e = std::getenv("LLAMA_M3_INIT_BLOCKS");  return e ? (int64_t) atoll(e) : (int64_t) -1; }();
                static const int64_t local_ov = [] { const char * e = std::getenv("LLAMA_M3_LOCAL_BLOCKS"); return e ? (int64_t) atoll(e) : (int64_t) -1; }();
                const int64_t topk      = topk_ov  >= 0 ? topk_ov  : (int64_t) hparams.indexer_top_k;
                const int64_t init_blk  = init_ov  >= 0 ? init_ov  : (int64_t) hparams.sparse_init_blocks;
                const int64_t local_blk = local_ov >= 0 ? local_ov : (int64_t) hparams.sparse_local_blocks;

                // index Q/K branch: proj -> GemmaRMS norm (folded +1 in weights) -> partial RoPE
                ggml_tensor * iq = build_lora_mm(model.layers[il].attn_index_q, cur);
                iq = ggml_reshape_3d(ctx0, iq, idx_dim, idx_heads, n_tokens);
                iq = build_norm(iq, model.layers[il].attn_index_q_norm, NULL, LLM_NORM_RMS, il);
                iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                ggml_tensor * ik = build_lora_mm(model.layers[il].attn_index_k, cur);
                ik = ggml_reshape_3d(ctx0, ik, idx_dim, 1, n_tokens);
                ik = build_norm(ik, model.layers[il].attn_index_k_norm, NULL, LLM_NORM_RMS, il);
                ik = ggml_rope_ext(ctx0, ik, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                // PER-(GQA-group) block masks: M3 has indexer_n_head == n_head_kv, so each kv-group g
                // gets its OWN selection driven by index head g. We build idx_heads masks [n_kv, n_q] and
                // expand to a FULL per-head mask [n_kv, n_q, n_head]: soft_max broadcasts the mask ne[2]
                // INTERLEAVED (i02 % ne12) while GQA groups are CONTIGUOUS (head/gs), so a [.,.,n_head_kv]
                // mask would mis-map — the full per-head mask (each group repeated gs times) lines up.
                // Store this ubatch's index K into the indexer-K cache, then score against the FULL cache
                // view (all real prior keys). THIS makes DECODE correct (n_tokens=1 must attend every prior
                // key) and keeps prefill consistent (both score the same F32 cache -> bit-exact selection).
                ggml_tensor * idxk_store = build_sparse_store_indexer_k(ik, inp_attn->get_k_idxs(), il);
                if (idxk_store != nullptr) {
                    ggml_build_forward_expand(gf, idxk_store);
                }
                ggml_tensor * idxk_cache = build_inp_indexer_k(il); // [idx_dim, n_kv, ns] or null
                ggml_tensor * ik_2d = (idxk_cache != nullptr && idxk_cache->ne[2] == 1)
                    ? ggml_reshape_2d(ctx0, idxk_cache, idxk_cache->ne[0], idxk_cache->ne[1]) // [idx_dim, n_kv]
                    : ggml_reshape_2d(ctx0, ik, idx_dim, n_tokens);                          // direct fallback

                // Size the KEY axis to the cache n_kv (= causal-mask leaf ne[0]); pad to a block multiple
                // for the pool. With the cache view scores cover all n_kv real keys; the causal mask uses
                // n_past so the current tokens attend all prior. NOTE: n_past = n_kv - n_tokens assumes an
                // unpadded contiguous cache (n_kv == pos0 + n_tokens), which holds for kv_unified; a padded
                // cache would need the true pos0 (from inp_pos) — a real-model refinement.
                const int64_t n_kv     = inp_attn->get_kq_mask()->ne[0];
                const int64_t n_blocks = (n_kv + bs - 1) / bs;
                const int64_t n_kv_pad = n_blocks * bs;
                // Causal via the MAIN attention's kq_mask (0 visible / -inf masked) — correct for prefill
                // AND decode at RUNTIME (it encodes real key/query positions), unlike a build-time n_past
                // which wrongly assumes the queries are the last n_kv rows (breaks for a padded/decode
                // cache). Reshape to [n_kv, n_q] (kv_unified -> n_stream==1); added to scores pre-pool.
                // cast to F32: with FA ON get_kq_mask() is F16, and adding it to the F32 scores would
                // hit ggml's unsupported f32+f16 binary op (FA OFF: already F32, the cast is a no-op).
                ggml_tensor * kqm_2d = ggml_cast(ctx0,
                    ggml_reshape_2d(ctx0, inp_attn->get_kq_mask(), n_kv, n_tokens), GGML_TYPE_F32);

                // PER-GROUP is opt-in (LLAMA_M3_SPARSE_PERGROUP). It builds a FULL per-head mask which
                // (a) needs FA off and (b) currently disables CUDA graphs (a per-head-mask capture issue)
                // => a perf hit on the RTX. DEFAULT = the shared index-head-0 mask (1 group), which is
                // CUDA-graph + FA compatible and is the originally-validated path. Per-group LOGIC is
                // CPU-validated (coherence gate green); the CUDA-graph fix + accuracy tuning come with weights.
                static const bool m3_per_group = [] {
                    const char * e = std::getenv("LLAMA_M3_SPARSE_PERGROUP");
                    return e && e[0] != '\0' && e[0] != '0';
                }();
                const int64_t n_groups = m3_per_group ? idx_heads : 1;
                ggml_tensor * grp_stack = nullptr; // [n_kv, n_q, n_groups]
                for (int64_t g = 0; g < n_groups; ++g) {
                    ggml_tensor * iq_g = ggml_cont(ctx0,
                        ggml_view_2d(ctx0, iq, idx_dim, n_tokens, iq->nb[2], g * iq->nb[1])); // index head g
                    ggml_tensor * scores = ggml_mul_mat(ctx0, ik_2d, iq_g); // [n_kv, n_q] (cache view)
                    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                    scores = ggml_add(ctx0, scores, kqm_2d); // causal: + kq_mask (-inf on masked keys)
                    const int64_t key_pad = n_kv_pad - scores->ne[0];
                    if (key_pad > 0) {
                        ggml_tensor * neg = ggml_fill(ctx0,
                            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, key_pad, n_tokens), -1e30f);
                        scores = ggml_concat(ctx0, scores, neg, 0); // [n_kv_pad, n_q]
                    }
                    // block-max-pool over keys: [n_kv_pad, n_q] -> [n_blocks, n_q]
                    ggml_tensor * sc4 = ggml_reshape_4d(ctx0, scores, n_kv_pad, n_tokens, 1, 1);
                    ggml_tensor * bsc = ggml_pool_2d(ctx0, sc4, GGML_OP_POOL_MAX, (int) bs, 1, (int) bs, 1, 0, 0);
                    bsc = ggml_reshape_2d(ctx0, bsc, n_blocks, n_tokens);
                    auto K2 = [&](float v) { return ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blocks, n_tokens), v); };
                    // init/local-block FORCE via BIG-SCORE INJECTION *before* the top-k, so the sink (first
                    // init_blk) + recent window (last local_blk up to the query's block) outrank naturals and
                    // are selected WITHIN the topk budget. Mirrors scripts/m3_msa_reference.py m3_index_topk
                    // ("topk budget INCLUDES them"; where(local,1e29) then where(init,1e30) so init wins on
                    // overlap) and the MLX _msa_general_mask. The OLD path added force ON TOP of the natural
                    // top-topk (rank<topk OR forced) -> selected topk+#forced blocks, diverging from the
                    // oracle/MLX (which select topk total) at partial coverage. Cross-check 2026-06-24.
                    if (init_blk > 0 || local_blk > 0) {
                        ggml_tensor * bidx = ggml_repeat(ctx0,
                            ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_blocks, 1.0f), n_blocks, 1), K2(0.0f));
                        if (local_blk > 0) { // qblk - local_blk < b <= qblk, qblk = floor(pos(query)/bs)
                            ggml_tensor * qblk = ggml_repeat(ctx0, ggml_reshape_2d(ctx0,
                                ggml_floor(ctx0, ggml_scale(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f / (float) bs)),
                                1, n_tokens), K2(0.0f));
                            ggml_tensor * diff  = ggml_sub(ctx0, bidx, qblk);                                  // b - qblk
                            ggml_tensor * upper = ggml_step(ctx0, ggml_sub(ctx0, K2(0.5f), diff));             // b <= qblk
                            ggml_tensor * lower = ggml_step(ctx0, ggml_add(ctx0, diff, K2((float) local_blk - 0.5f))); // b > qblk-local
                            ggml_tensor * lmask = ggml_mul(ctx0, upper, lower);                                // 1 if local
                            bsc = ggml_add(ctx0, ggml_mul(ctx0, bsc, ggml_sub(ctx0, K2(1.0f), lmask)),
                                                 ggml_scale(ctx0, lmask, 1e29f));                              // where(local,1e29,bsc)
                        }
                        if (init_blk > 0) { // b < init_blk (global sink); applied AFTER local so init wins on overlap
                            ggml_tensor * imask = ggml_step(ctx0, ggml_sub(ctx0, K2((float) init_blk - 0.5f), bidx)); // 1 if b<init
                            bsc = ggml_add(ctx0, ggml_mul(ctx0, bsc, ggml_sub(ctx0, K2(1.0f), imask)),
                                                 ggml_scale(ctx0, imask, 1e30f));                              // where(init,1e30,bsc)
                        }
                    }
                    // selection via pairwise rank on the (force-injected) scores: rank[b,q]=#blocks scoring
                    // higher; selected iff rank<topk. Forced blocks (huge score) rank at the top, filling the
                    // topk budget first -> total selected == topk (oracle/MLX semantics).
                    ggml_tensor * ref = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_blocks, n_blocks, n_tokens);
                    ggml_tensor * A = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, bsc, n_blocks, 1, n_tokens), ref);
                    ggml_tensor * B = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, bsc, 1, n_blocks, n_tokens), ref);
                    ggml_tensor * rank = ggml_reshape_2d(ctx0,
                        ggml_sum_rows(ctx0, ggml_step(ctx0, ggml_sub(ctx0, A, B))), n_blocks, n_tokens);
                    ggml_tensor * not_sel = ggml_step(ctx0, ggml_sub(ctx0, rank, K2((float) topk - 0.5f))); // 1 if rank>=topk
                    ggml_tensor * mask_blk = ggml_scale(ctx0, not_sel, -1e30f); // -1e30 pruned / 0 selected
                    // expand blocks -> keys -> view [n_kv, n_q]
                    ggml_tensor * mref = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, bs, n_blocks, n_tokens);
                    ggml_tensor * mexp = ggml_reshape_2d(ctx0,
                        ggml_repeat(ctx0, ggml_reshape_3d(ctx0, mask_blk, 1, n_blocks, n_tokens), mref),
                        n_kv_pad, n_tokens);
                    ggml_tensor * mg = (n_kv_pad > n_kv)
                        ? ggml_cont(ctx0, ggml_view_2d(ctx0, mexp, n_kv, n_tokens, mexp->nb[1], 0))
                        : mexp;
                    mg = ggml_reshape_3d(ctx0, mg, n_kv, n_tokens, 1);
                    grp_stack = (grp_stack == nullptr) ? mg : ggml_concat(ctx0, grp_stack, mg, 2);
                }
                ggml_tensor * block_mask;
                if (m3_per_group) {
                    // expand groups -> heads (CONTIGUOUS): [n_kv,n_q,idx_heads] -> [n_kv,n_q,gs,idx_heads]
                    //   -> [n_kv,n_q,n_head] so head h uses group h/gs.
                    const int64_t gs = n_head / idx_heads;
                    ggml_tensor * href = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, gs, idx_heads);
                    block_mask = ggml_reshape_3d(ctx0,
                        ggml_repeat(ctx0, ggml_reshape_4d(ctx0, grp_stack, n_kv, n_tokens, 1, idx_heads), href),
                        n_kv, n_tokens, n_head);                                      // [n_kv, n_q, n_head]
                } else {
                    block_mask = ggml_reshape_2d(ctx0, grp_stack, n_kv, n_tokens);    // shared [n_kv, n_q]
                }

                cur = build_attn_m3_sparse(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, block_mask,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }
        }

        if (il == il_end - 1 && inp_out_ids) {        // last BUILT layer (effective last, or the DRAFT layer)
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            // leading dense FFN (swigluoai)
            ggml_tensor * g = build_lora_mm(model.layers[il].ffn_gate, cur);
            ggml_tensor * u = build_lora_mm(model.layers[il].ffn_up,   cur);
            g   = ggml_swiglu_oai(ctx0, g, u, swiglu_alpha, swiglu_limit);
            cur = build_lora_mm(model.layers[il].ffn_down, g);
            cb(cur, "ffn_out", il);
        } else {
            // routed experts (swigluoai MoE)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (swigluoai)
            ggml_tensor * sg = build_lora_mm(model.layers[il].ffn_gate_shexp, cur);
            ggml_tensor * su = build_lora_mm(model.layers[il].ffn_up_shexp,   cur);
            sg = ggml_swiglu_oai(ctx0, sg, su, swiglu_alpha, swiglu_limit);
            ggml_tensor * ffn_shexp = build_lora_mm(model.layers[il].ffn_down_shexp, sg);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // MTP draft uses the NextN shared head (its own final norm + lm_head, the latter optionally shared with
    // the main output); the normal path uses the main output head (UNCHANGED).
    if (mtp_enabled) {
        const auto & nextn = model.layers[total_n_layers - 1].nextn;
        cur = build_norm(cur, nextn.shared_head_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;
        cur = build_lora_mm(nextn.shared_head_head != nullptr ? nextn.shared_head_head : model.output, cur);
    } else {
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
