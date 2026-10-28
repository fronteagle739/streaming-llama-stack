#include "models.h"
#include "../llama-memory-hybrid-idx.h"

// GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a k-pool DSA indexer,
// mHC residual streams, DeepSeek-style MoE.
//
// Ported from upstream llama.cpp src/models/glm5-next.cpp (0f701c882), trunk only (no MTP graph). Uses the
// fused DSV4_HC_PRE/COMB/POST and GATED_DELTA_NET ops (upstream's defaults); the indexer scoring stays on the
// unfused path (the fork's ggml_lightning_indexer predates upstream's masked variant).
// Fork differences: no recurrent rollback (one conv/ssm state slot), the indexer selection mask is built
// from the raw F32 kq_mask leaf inside the layer (the layer-major executor never evaluates the head-of-graph
// F16 mask cast for layers > 0) and cast to F16 there when flash attention is on, and the fork's MLA mask may be
// padded to 256 columns under flash attention, so the dump rows of the selection scatter start at the padded width.

// glm5-next k-pool indexer inputs
class llm_graph_input_glm5_kpool : public llm_graph_input_i {
public:
    llm_graph_input_glm5_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_glm5_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == std::max(mctx->get_n_kpool_new(), 1u);

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64 [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32 [n_pool]
    ggml_tensor * pool_idxs     = nullptr; // I32 [kpool, n_pool]
    ggml_tensor * pool_mask     = nullptr; // F32 [n_pool, n_tokens]
    ggml_tensor * tail_idxs     = nullptr; // I32 [kpool - 1, n_tokens]
    ggml_tensor * sel_mask      = nullptr; // F32 [n_sel, 1, 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr; // I32 [kpool, n_new]
    ggml_tensor * new_pool_rep  = nullptr; // I64 [n_new]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0;
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
};

// x / sqrt(sum(x^2) + eps), FLA's l2 norm (upstream build_gdn_l2_norm)
// The trained ds4 GGUF keeps several norm/affine vectors in BF16. Binary
// broadcast kernels consume these operands as F32; preserve the stored values
// with an explicit conversion inside the consuming layer's graph range.
static ggml_tensor * glm5_f32(ggml_context * ctx, ggml_tensor * t) {
    return t && t->type != GGML_TYPE_F32 ? ggml_cast(ctx, t, GGML_TYPE_F32) : t;
}

static ggml_tensor * glm5_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

// Causal conv1d over one of Q/K/V
static ggml_tensor * glm5_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                 ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                 int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                 int64_t d_conv, int64_t head_dim, int64_t n_head,
                                 int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // the last d_conv - 1 columns are the new conv state
    ggml_tensor * conv_x_last = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, conv_x_last,
            ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1)   * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

static ggml_tensor * glm5_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * glm5_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * glm5_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, glm5_f32(ctx, scale));
    x = ggml_add(ctx, x, glm5_f32(ctx, base));
    return x;
}

llm_graph_input_glm5_kpool * llm_build_glm5_next::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();
    const uint32_t n_kv   = mctx_idx->get_n_kv();
    // a ubatch that completes no pool still builds one dummy entry, so the graph does not
    // change shape every kpool tokens
    const uint32_t n_new = std::max(mctx_hyb->get_n_kpool_new(), 1u);

    auto inp = std::make_unique<llm_graph_input_glm5_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pool, n_tokens);
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);

    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    ggml_build_forward_expand(gf, inp->pool_mask);
    ggml_build_forward_expand(gf, inp->tail_idxs);

    inp->n_kv = n_kv;

    // selection width: the top pools plus the optional tail, also the sparse attention bound
    {
        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        const int64_t n_sel      = kpool*n_top_pool + (hparams.indexer_kpool_select_tail ? kpool - 1 : 0);
        inp->n_sel = (uint32_t) n_sel;

        inp->sel_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_sel, 1, 1, n_tokens);
        ggml_set_input(inp->sel_mask);
        ggml_build_forward_expand(gf, inp->sel_mask);
    }

    inp->n_new = n_new;

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_new);
    ggml_set_input(inp->new_pool_idxs);
    inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new);
    ggml_set_input(inp->new_pool_rep);

    return (llm_graph_input_glm5_kpool *) res->add_input(std::move(inp));
}

ggml_tensor * llm_build_glm5_next::build_hc_mean(ggml_tensor * x) const {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx0, acc, 1.0f/hc);
}

ggml_tensor * llm_build_glm5_next::build_hc_pre(
        ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
        ggml_tensor ** post, ggml_tensor ** comb, int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    // Name the first arithmetic node so layer-major ranges include this
    // normalization after restoring each microbatch boundary.
    cb(flat_norm, "hc_input_norm", il);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = glm5_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = glm5_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = glm5_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = glm5_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = glm5_view_2d(ctx0, mixes, hc, nt, 0);
    pre = glm5_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = glm5_view_2d(ctx0, mixes, hc, nt, hc);
    *post = glm5_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps, (int32_t) hparams.dsv4_hc_sinkhorn_iters);
    cb(*comb, "hc_comb", il);

    return ggml_dsv4_hc_pre(ctx0, x, pre);
}

ggml_tensor * llm_build_glm5_next::build_hc_post(
        ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb, int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);
    GGML_UNUSED(il);

    return ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
}

llm_build_glm5_next::llm_build_glm5_next(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // recurrent state + K-only MLA cache through the generic hybrid input, plus the indexer cache
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_rs    = inp_hyb->get_recr();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head_kda   = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head_kda * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int     n_trunk      = n_layer - (int) hparams.nextn_predict_layers;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * prev_sel = nullptr;

    for (int il = 0; il < n_trunk; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, glm5_f32(ctx0, layer.attn_norm), nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda,
                                  d_inner, n_seq_tokens, n_seqs, il);
        } else {
            cur = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, &prev_sel, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, glm5_f32(ctx0, layer.ffn_norm), nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    glm5_f32(ctx0, layer.ffn_exp_probs_b),
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // narrow to the output tokens, then collapse the streams
    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_mean(inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, glm5_f32(ctx0, model.output_norm), nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// KDA layer, g_a/g_b output gate

ggml_tensor * llm_build_glm5_next::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
        int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il) {

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Kcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Vcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // Decay gate, ssm_a holds -exp(A_log) (the ds4 GGUF stores A_log itself)
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, glm5_f32(ctx0, layer.ssm_dt_b));

    ggml_tensor * A = glm5_f32(ctx0, layer.ssm_a);
    if (hparams.kda_a_is_log) {
        A = ggml_neg(ctx0, ggml_exp(ctx0, A));
    }
    A = ggml_reshape_3d(ctx0, A, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    // Match FLA l2 norm
    constexpr float l2_eps = 1e-6f;
    Qcur = glm5_l2_norm(ctx0, Qcur, l2_eps);
    Kcur = glm5_l2_norm(ctx0, Kcur, l2_eps);

    // fused gated delta net (upstream build_delta_net_fused), K = 1: the output carries the final state only
    ggml_tensor * gdn = ggml_gated_delta_net(ctx0, Qcur, Kcur, Vcur, g1, beta, state, 1);
    cb(gdn, "kda_gdn", il);

    ggml_tensor * output = ggml_view_4d(ctx0, gdn, head_dim, n_head_kda, n_seq_tokens, n_seqs,
            ggml_row_size(gdn->type, head_dim),
            ggml_row_size(gdn->type, head_dim * n_head_kda),
            ggml_row_size(gdn->type, head_dim * n_head_kda * n_seq_tokens), 0);
    ggml_tensor * new_state = ggml_view_4d(ctx0, gdn, head_dim, head_dim, n_head_kda, n_seqs,
            ggml_row_size(gdn->type, head_dim),
            ggml_row_size(gdn->type, head_dim * head_dim),
            ggml_row_size(gdn->type, head_dim * head_dim * n_head_kda),
            ggml_row_size(gdn->type, head_dim * n_head_kda * n_seq_tokens * n_seqs));
    output = ggml_cont(ctx0, output);
    cb(output, "kda_scan_out", il);

    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, new_state,
            ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                         kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // output gate, then RMSNorm(o) * Sigmoid(g2)
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
    ggml_tensor * g2  = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_tokens);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_tokens);
    ggml_tensor * normed = build_norm(o, glm5_f32(ctx0, layer.ssm_o_norm), nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

// Scores pools of kpool consecutive tokens, expands the selected pools and the incomplete tail into an additive mask

ggml_tensor * llm_build_glm5_next::build_kpool_select(
        ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * kq_mask, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_glm5_kpool * inp_kpool, int il) {

    const auto * mctx_lid = mctx_hyb->get_idx();

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;
    const int64_t n_pool         = inp_kpool->pool_cells->ne[0];
    const int64_t n_new          = inp_kpool->n_new;

    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);
    cb(iq, "indexer_q", il);

    // Per-token key and pool gate scores, cached together
    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    ik = build_norm(ik, glm5_f32(ctx0, layer.indexer_k_norm), glm5_f32(ctx0, layer.indexer_k_norm_b), LLM_NORM, il);
    cb(ik, "indexer_k", il);

    ggml_tensor * ig = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, cur);
    cb(ig, "indexer_gate", il);

    // Cache rows store key | gate | pooled
    ggml_tensor * pzero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_indexer, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_concat(ctx0, ggml_concat(ctx0, ik, ig, 0), pzero, 0);
    packed = ggml_reshape_3d(ctx0, packed, 3*n_embd_indexer, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, n_embd_indexer);

    // Pool the entries completed by this ubatch. The last one is a dummy when the ubatch completes none.
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    rows = ggml_reshape_3d(ctx0, rows, 2*n_embd_indexer, kpool, n_new);

    ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], 0);
    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_embd_indexer));

    ggml_tensor * logits = ggml_add(ctx0, pg, glm5_f32(ctx0, layer.indexer_kpool_ape));
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3)); // [kpool, head_dim, n_new]
    ggml_tensor * probs = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, logits, kpool, n_embd_indexer*n_new));
    probs = ggml_reshape_3d(ctx0, probs, kpool, n_embd_indexer, n_new);

    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    ggml_tensor * pooled_new = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pk)); // [1, head_dim, n_new]
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, n_embd_indexer, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    // scatter the fresh pooled keys into their rep rows, then gather all n_pool of them by cell
    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
    ggml_tensor * pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    pooled = ggml_reshape_3d(ctx0, pooled, n_embd_indexer, 1, n_pool);
    cb(pooled, "indexer_pool_k", il);

    ggml_tensor * sel_idx = nullptr;
    {
        ggml_tensor * weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
        weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));
        cb(weights, "indexer_weights", il);

        ggml_tensor * q_p = ggml_permute(ctx0, iq, 0, 2, 1, 3);     // [head_dim, n_tokens, n_head]
        ggml_tensor * k_p = ggml_permute(ctx0, pooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);            // [n_pool, n_tokens, n_head]
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));   // [n_head, n_tokens, n_pool]
        ggml_tensor * score = ggml_relu(ctx0, kq);
        score = ggml_mul(ctx0, score, weights);
        score = ggml_sum_rows(ctx0, score);                          // [1, n_tokens, n_pool]
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
        score = ggml_add(ctx0, score, inp_kpool->pool_mask);
        cb(score, "indexer_score", il);

        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], UNORDERED

        // The selection mask marks the first min(nv, n_top_pool) slots as the visible pools, so order the set by descending score.
        ggml_tensor * sel_score = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
        ggml_tensor * sel_order = ggml_argsort(ctx0,
                ggml_reshape_2d(ctx0, sel_score, n_top_pool, n_tokens), GGML_SORT_ORDER_DESC);
        top_k = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, ggml_cast(ctx0, top_k, GGML_TYPE_F32), 1, n_top_pool, n_tokens), sel_order);
        top_k = ggml_cast(ctx0, ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, n_top_pool, n_tokens)), GGML_TYPE_I32);
        cb(top_k, "indexer_top_k", il);

        sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
                ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens));  // [kpool, n_top_pool*n_tokens]
        sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);

        if (hparams.indexer_kpool_select_tail) {
            // Append the incomplete tail with n_kv for missing cells.
            sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
        }
    }
    const int64_t n_sel = sel_idx->ne[0];

    ggml_build_forward_expand(gf, sel_idx);

    // fork: the mask may be padded past n_kv (flash attention + MLA), dump rows start after the padding
    const int64_t n_kv_mask = kq_mask->ne[0];
    GGML_ASSERT(kq_mask->type == GGML_TYPE_F32);

    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_mask + n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv_mask + n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv_mask + n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // Live slots (visible pools, real tail cells) address disjoint cells. Each dead slot writes its own dump row
    // n_kv_mask + slot, so the scatter indices of a token are unique: idx = dump + live*(idx - dump), live = exp(mask).
    GGML_ASSERT(inp_kpool->sel_mask->ne[0] == n_sel && inp_kpool->sel_mask->ne[3] == n_tokens);
    ggml_tensor * live  = ggml_exp(ctx0, ggml_reshape_2d(ctx0, inp_kpool->sel_mask, n_sel, n_tokens));
    ggml_tensor * dump  = ggml_arange(ctx0, (float) n_kv_mask, (float) (n_kv_mask + n_sel), 1.0f);
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));
    sel = ggml_view_2d(ctx0, sel, n_kv_mask, n_tokens, sel->nb[2], 0);

    // Fold causal visibility before shared-indexer reuse.
    GGML_ASSERT(kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
    sel = ggml_add(ctx0, sel, ggml_reshape_2d(ctx0, kq_mask, n_kv_mask, n_tokens));
    cb(sel, "indexer_sel", il);

    return sel;
}

// Nope MLA layer with sparse attention over the indexer selection

ggml_tensor * llm_build_glm5_next::build_dsa_layer(
        ggml_tensor * cur, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
        llm_graph_input_glm5_kpool * inp_kpool, ggml_tensor ** prev_sel, int il) {

    const auto * mctx_mla = mctx_hyb->get_attn();

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla;
    const float   kq_scale = 1.0f / sqrtf((float) n_embd_head_k_mla);

    GGML_ASSERT(hparams.n_rot == 0 && "GLM5-Next MLA is nope-only");

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, glm5_f32(ctx0, layer.attn_q_a_norm), nullptr, LLM_NORM_RMS, il);
    cb(qr, "q_resid", il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens);

    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = build_norm(kv_cmpr, glm5_f32(ctx0, layer.attn_kv_a_norm), nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr", il);

    // absorb wk_b so the cache holds only the latent
    ggml_tensor * q_absorbed = ggml_permute(ctx0, q, 0, 2, 1, 3);
    q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_absorbed);
    q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);
    cb(q_absorbed, "q_absorbed", il);

    // the raw F32 mask leaf, consumed inside the layer
    ggml_tensor * kq_mask = inp_attn->self_kq_mask;

    ggml_tensor * sel = nullptr;
    if (hparams.is_indexer_full(il)) {
        sel = build_kpool_select(cur, qr, kq_mask, layer, mctx_hyb, inp_kpool, il);
        *prev_sel = sel;
    } else {
        GGML_ASSERT(*prev_sel != nullptr && "shared indexer layer must follow a full indexer layer");
        sel = *prev_sel;
    }

    ggml_build_forward_expand(gf, q_absorbed);
    ggml_build_forward_expand(gf, kv_cmpr);
    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv_cmpr, inp_attn->get_k_idxs(), il));

    // The scatter selection already includes the causal mask.
    ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
    if (cparams.flash_attn) {
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
    }
    cb(mask, "kq_mask_dsa", il);

    ggml_tensor * k = mctx_mla->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * out = build_attn_mha(q_absorbed, k, v, nullptr, mask, nullptr, layer.wv_b, kq_scale, il);
    cb(out, "kqv_out", il);

    out = ggml_mul_mat(ctx0, layer.wo, out);
    cb(out, "attn_out", il);

    return out;
}
