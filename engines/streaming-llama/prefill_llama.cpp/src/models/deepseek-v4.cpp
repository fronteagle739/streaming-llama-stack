#include "models.h"

#include "../llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// DeepSeek-V4 QAT activation-simulation round trips.
//
// DSV4 is trained WITH its deployment quantisation in the graph. The official graph rotates the
// indexer activations with a 128-wide Hadamard and immediately runs an FP4 round trip on them, and
// runs an FP8 round trip on the attention compressor's KV. This fork shipped the Hadamard (upstream
// PR 21038) but NEITHER round trip, so the compressed rows it stores are not the representation the
// weights were trained against.
//
// Measured 2026-08-27, same 153 GB GGUF, temperature 0, exact match: ds4-server (which does apply
// them) repeats a code planted at the start of a 596-token prompt; this fork returns filler. Forcing
// the indexer dense (LLAMA_DSV4_INDEXER_TOP_K=60000) does NOT fix it, which is why the attention
// compressor's FP8 matters as much as the indexer's FP4 -- the content is wrong, not just the
// ranking.
//
// ON by default: this is the model's graph, not an option. LLAMA_DSV4_QAT=0 disables it, for A/B
// against the old behaviour only.
static bool dsv4_qat_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("LLAMA_DSV4_QAT");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return enabled;
}

// Which layer LLAMA_DSV4_COMP_DUMP pins its tensors on. Default 2 -- but EVERY dump taken so far
// was layer 2, so a divergence that only starts deeper was structurally invisible. This makes the
// layer a knob (LLAMA_DSV4_COMP_DUMP_LAYER) instead of a constant.
static int dsv4_comp_dump_layer() {
    static const int layer = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP_LAYER");
        return (e && e[0]) ? std::atoi(e) : 2;
    }();
    return layer;
}

static bool dsv4_indexer_qat_enabled() {
    return dsv4_qat_enabled();
}

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

static ggml_tensor * dsv4_with_zero_dep(ggml_context * ctx, ggml_tensor * t, ggml_tensor * dep) {
    if (dep == nullptr) {
        return t;
    }

    ggml_tensor * zero = ggml_scale(ctx, ggml_sum(ctx, dep), 0.0f);
    return ggml_add(ctx, t, zero);
}

static constexpr int64_t DSV4_CSA_RATIO  = 4;
static constexpr int64_t DSV4_HCA_RATIO  = 128;
static constexpr int64_t DSV4_SWA_WINDOW = 128;

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llm_build_deepseek4::build_hc_weighted_sum(
        ggml_tensor * x,
        ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);

        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }

    return acc;
}

ggml_tensor * llm_build_deepseek4::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llm_build_deepseek4::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    // Layer-major range discovery keys off the first layer-tagged node.  The
    // HCA reshape and norm used to be anonymous, making hc_mixes the apparent
    // layer start.  On layers > 0 that skipped both operations and caused the
    // generic hidden restore to select hc_mixes->src[0] (the hc_fn weight)
    // instead of x, the full [n_embd, hc, n_tokens] residual stream.
    cb(flat, "hc_flat", il);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    cb(flat_norm, "hc_flat_norm", il);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);
    pre = ggml_add(ctx0, pre, eps);
    cb(pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb, il);
    cb(*comb, "hc_comb", il);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llm_build_deepseek4::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llm_build_deepseek4::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);
    pre = ggml_add(ctx0, pre, eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llm_build_deepseek4::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(score, name, il);

    // HCA HAS NEVER BEEN VERIFIED (2026-08-28). Every dump, oracle and numpy comparison in the
    // needle investigation landed on CSA layers (2, 30, 42 -- all clean). The compress-ratio array
    // is [0,0,4,128,4,128,...]: even layers >=2 are CSA, odd layers 3..41 are HCA. That is 20
    // layers, ~45% of the model's attention, on the LONG-RANGE path -- the regime that fails --
    // with no coverage at all.
    //
    // Worth pinning specifically because this softmax reduces over DSV4_HCA_RATIO = 128 rows,
    // where CSA's reduces over 2*ratio = 8. A CUDA soft_max defect at width 128 but not at width 8
    // would attenuate HCA only, stay invisible to every CSA check, and still differ from Metal --
    // and unlike a graph-DESIGN bug it survives the E3 filter (Metal runs this same graph and
    // recalls, so nothing expressible purely in the graph can be the cause).
    //
    // Distinct names from the CSA pins so a name can never resolve to the wrong path, and
    // ggml_set_output + forward_expand rather than cb() alone -- a name by itself leaves the buffer
    // free to be recycled before readback, which is what produced the bogus rel=133 "divergence".
    static const bool hca_dump_pin = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    const bool hca_pin_here = hca_dump_pin && il == dsv4_comp_dump_layer();

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    // Pin the ggml_cont OUTPUTS, never `kv`/`score` themselves. Those are reshape VIEWS, and
    // ggml_set_output on a view does not stop the allocator reusing the base tensor it points into:
    // the first attempt captured `kv` that way and read back 32767/131072 values above 1e30 while
    // the pooled result was a sane 1.114 -- i.e. the dump was a recycled buffer, not what the
    // kernel consumed. That made all four comparisons void, including the soft_max one.
    // `values`/`scores` are materialised tensors, so they survive to the readback.
    if (hca_pin_here) {
        cb(values, "hca_values", il);
        cb(scores, "hca_scores", il);
        ggml_set_output(values);
        ggml_set_output(scores);
        ggml_build_forward_expand(gf, values);
        ggml_build_forward_expand(gf, scores);
    }

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    // The pooled tensor gets its own name for the same reason the CSA one does: everything dumped
    // downstream is post norm+rope+qat, so the pooling itself cannot be checked against a
    // recomputation from the state unless it is captured here.
    if (hca_pin_here) {
        cb(weights, "hca_state_compress_weights", il);
        cb(comp,    "hca_state_compress_pooled",  il);
        ggml_set_output(weights);
        ggml_set_output(comp);
        ggml_build_forward_expand(gf, weights);
        ggml_build_forward_expand(gf, comp);
    }

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llm_build_deepseek4::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    ggml_tensor * prev_idxs = dsv4_view_1d(ctx0, state_read_idxs, ratio*n_blocks, 0);
    ggml_tensor * cur_idxs  = dsv4_view_1d(ctx0, state_read_idxs, ratio*n_blocks, ratio*n_blocks);

    ggml_tensor * kv_prev = ggml_get_rows(ctx0, kv_state, prev_idxs);
    kv_prev = ggml_cont(ctx0, ggml_view_2d(ctx0, kv_prev, n_embd_head, ratio*n_blocks, kv_prev->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_get_rows(ctx0, score_state, prev_idxs);
    score_prev = ggml_cont(ctx0, ggml_view_2d(ctx0, score_prev, n_embd_head, ratio*n_blocks, score_prev->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_get_rows(ctx0, kv_state, cur_idxs);
    kv_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, kv_cur, n_embd_head, ratio*n_blocks, kv_cur->nb[1],
            ggml_row_size(kv_cur->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_get_rows(ctx0, score_state, cur_idxs);
    score_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, score_cur, n_embd_head, ratio*n_blocks, score_cur->nb[1],
            ggml_row_size(score_cur->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    // Give the POOLED tensor a name of its own so LLAMA_DSV4_COMP_DUMP can capture it. Every
    // other dumped compressed tensor is post norm+rope+qat, which makes it impossible to check
    // the pooling itself against a recomputation from the dumped state_kv/state_score -- the
    // norm weights and the RoPE stand in the way. With its own name the pooling gets a
    // self-contained numpy oracle and needs no second implementation to compare against.
    // Naming + an output pin only; no graph change when the dump env is unset.
    {
        char pooled_name[GGML_MAX_NAME];
        snprintf(pooled_name, sizeof(pooled_name), "%s_pooled", name);
        cb(comp, pooled_name, il);
    }
    static const bool comp_dump_pin_pooled = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    if (comp_dump_pin_pooled && il == dsv4_comp_dump_layer()) {
        ggml_set_output(comp);
    }

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llm_build_deepseek4::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot;
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            0);
    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(indexer_q_pe, "lid_q_pe", il);

    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    indexer_q = ggml_mul_mat(ctx0, inp_lid.k_rot, indexer_q);
    cb(indexer_q, "lid_q_rot", il);

    // QAT: FP4 activation round trip on the indexer query, immediately after the Hadamard
    // (ds4.c:13674 -- dsv4_indexer_qat_row_inplace_cpu = hadamard THEN fp4). Q and the indexer keys
    // must go through the SAME round trip or the scores are computed between two different
    // representations, which is worse than skipping it on both.
    if (dsv4_indexer_qat_enabled()) {
        indexer_q = ggml_dsv4_act_qat(ctx0, indexer_q, GGML_DSV4_QAT_FP4_E2M1, 0);
        cb(indexer_q, "lid_q_qat", il);
    }

    // round-6 probe: pin the post-Hadamard decode query (the actual computed node,
    // BEFORE the later permute view) + its qr input, for offline score cross-check.
    static const bool lid_probe_pin_q = []() {
        const char * e = std::getenv("LLAMA_DSV4_LID_PROBE");
        return e && e[0] != '\0';
    }();
    if (lid_probe_pin_q && nt == 1) {
        ggml_set_output(indexer_q);
        ggml_set_output(qr);
    }
    // round-7: PREFILL-time query dump (quant-vs-decode-path split) — pin the prefill
    // graph's lid_q_rot under COMP_DUMP so post_layer_cb can dump the last ubatch's queries.
    static const bool comp_dump_pin_q = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != '\0';
    }();
    if (comp_dump_pin_q && il == 2 && nt > 1) {
        ggml_set_output(indexer_q);
    }

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    cb(indexer_weights, "lid_weights", il);
    if (lid_probe_pin_q && nt == 1) {
        ggml_set_output(indexer_weights);  // round-6: per-head score weights for the offline cross-check
    }

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    // The CUDA LID score-reduction matcher requires the portable score tail to
    // be consecutive in post-order graph traversal.  indexer_weights is a
    // non-leaf projection consumed by the second branch of the score MUL; if
    // left unvisited, its MUL_MAT (and view) are emitted between RELU and
    // SCALE.  Expand the final weights view first so the later score traversal
    // emits the exact seven-node tail.
    ggml_build_forward_expand(gf, indexer_weights);

// Keep the portable pure-ggml-op decomposition as the CPU/non-CUDA reference.
// CUDA recognizes this exact private chain and fuses the two transposes plus
// relu/scale/mul/sum_rows into one DSV4-specific score-reduction kernel, while
// preserving this fallback on unmatched layouts and other backends.
#if 0
    ggml_tensor * indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, 1.0f / sqrtf(float(n_embd_indexer_head)), 1.0f / sqrtf(float(n_indexer_head)));
    cb(indexer_score, "indexer_score", il);
#else
    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    cb(indexer_q, "lid_q", il);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
    cb(indexer_k, "lid_k", il);

    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
    cb(indexer_kq, "lid_kq", il);

    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
    cb(indexer_kq, "lid_kq", il);

    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
    indexer_score = ggml_sum_rows(ctx0, indexer_score);
    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
    cb(indexer_score, "lid_score", il);
#endif
    indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
    cb(indexer_score, "lid_score_masked", il);

    // BUG#2 test lever (2026-07-16, default-off): the resident stream_local decode marginally demotes the
    // needle (its LID score sits ~0.05 below the K=512 cutoff at the layers where it is a strong match ->
    // selected at too few layers -> wrong recall). LLAMA_DSV4_INDEXER_TOP_K=<N> raises the top-k so the
    // borderline needle is included; if recall returns with a modest bump (e.g. 768), the marginal-demotion
    // diagnosis is CONFIRMED and this is a working (band-aid) fix pending the numeric root-cause match.
    static const uint32_t indexer_top_k_override = []() -> uint32_t {
        const char * e = std::getenv("LLAMA_DSV4_INDEXER_TOP_K");
        return (e && e[0] != '\0') ? (uint32_t) std::max(0, std::atoi(e)) : 0u;
    }();
    // LLAMA_DSV4_INDEXER_TOP_K_DECODE=<N>: same override but applied ONLY to single-token (decode)
    // graphs, leaving prefill selection untouched. Sparse prefill KV is proven recall-sufficient
    // (Mac decode of it recalls); this splits decode-LID-ranking from the rest of the decode
    // pipeline: dense decode (N>=n_lid) recalls => only LID ranking is broken at decode (and this
    // is a deployable band-aid: stream_local decode is weight-bound, extra CSA FLOPs negligible);
    // still fails => LID exonerated, bug upstream in the decode hidden state.
    static const uint32_t indexer_top_k_decode_override = []() -> uint32_t {
        const char * e = std::getenv("LLAMA_DSV4_INDEXER_TOP_K_DECODE");
        return (e && e[0] != '\0') ? (uint32_t) std::max(0, std::atoi(e)) : 0u;
    }();
    const uint32_t eff_top_k =
            (indexer_top_k_decode_override != 0 && nt == 1) ? indexer_top_k_decode_override :
            indexer_top_k_override != 0 ? indexer_top_k_override : hparams.indexer_top_k;
    const uint32_t n_top_k = indexer_score->ne[0] < eff_top_k ? (uint32_t) indexer_score->ne[0] : eff_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    // LLAMA_DSV4_LID_PROBE: pin the masked score vector and the top-k output as graph
    // outputs so their buffers survive graph compute for a post-decode readback on the
    // ordinary (non-streaming/resident) decode path. Zero effect when the env is unset.
    static const bool lid_probe_pin = []() {
        const char * e = std::getenv("LLAMA_DSV4_LID_PROBE");
        return e && e[0] != '\0';
    }();
    if (lid_probe_pin && nt == 1) {
        ggml_set_output(indexer_score);
        ggml_set_output(top_k);
    }

    return top_k;
}

ggml_tensor * llm_build_deepseek4::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llm_build_deepseek4::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);
    GGML_ASSERT(inp_attn->self_k_rot_swa == nullptr);
    GGML_ASSERT(inp_attn->self_v_rot_swa == nullptr);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_context * mctx_swa = inp_attn->mctx->get_swa();

    ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

    ggml_tensor * raw_k = mctx_swa->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    // Pin what ATTENTION ACTUALLY READS. Every compressed tensor verified so far was the value as
    // COMPUTED; this is the view read back out of the CSA cache after cpy_k wrote it at
    // state_write_idxs. A correct row written to, or read from, the wrong index would leave every
    // earlier check passing and still feed attention the wrong content -- which is exactly the
    // symptom (fluent output, wrong facts, only for material outside the raw window).
    static const bool comp_dump_pin_comp_k = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    if (comp_dump_pin_comp_k && il == dsv4_comp_dump_layer()) {
        ggml_set_output(csa_k);
    }

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    // EXPERIMENT (LLAMA_DSV4_RAW_MASK_LEAF, default off): take the mask from the LEAF that
    // set_input_kq_mask actually writes, instead of the _cnv copy that get_kq_mask_swa() returns.
    //
    // Measured 2026-08-28 on the failing prompt, same run, layer 42:
    //     leaf self_kq_mask_swa      -inf=40896   q0=1   q255=128   == ds4's rule, 256/256 exact
    //     _cnv copy (what we read)   -inf=0       q0=256 q255=256   ==  0/256
    // i.e. the raw half of the mask attention sees carries no causal/window masking at all, which
    // would make prefill attend the whole chunk instead of the 128-token window.
    //
    // NOT yet established as the cause: the streaming-prefill path patches FA mask tensors at
    // runtime (arm_fa_patch, llama-context.cpp:18285) and may supply the real mask to the FA kernel
    // by a route that bypasses this cast, in which case the empty copy is harmless. This switch is
    // the discriminator -- if recall RETURNS with it on, the copy was the defect.
    static const bool raw_mask_from_leaf = []() {
        const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_LEAF");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    ggml_tensor * raw_mask = inp_attn->get_kq_mask_swa();

    // REFUTED EXPERIMENT (LLAMA_DSV4_RAW_MASK_OWN, **default OFF**, opt-in only): consume DSV4's
    // own raw-window mask instead of
    // the generic ISWA `_cnv`. `_cnv` is a ggml_cast built at the graph HEAD and the streaming
    // executor computes per-layer node ranges, so it is never refreshed -- measured as 65536 zeros
    // while its source leaf is correct (-inf=40896, == ds4's rule 256/256). Consuming the leaf
    // directly is NOT an option: set_input_kq_mask writes it through a host pointer and asserts
    // `ggml_backend_buffer_is_host`, so pulling it into the graph moves it device-side and crashes
    // (llama-kv-cache.cpp:2294 -- observed). `inp_dsv4->get_raw_kq_mask()` is a separate DSV4-owned
    // input, filled by the SAME set_input_kq_mask, consumed HERE inside the layer range -- exactly
    // how inp_csa.kq_mask already works, which is the half of the mask that arrives correct.
    // This header said "THE FIX ... default ON" until 2026-08-28. It was wrong on both counts and
    // contradicted the code ten lines below it. The shipping fix is LLAMA_DSV4_RAW_MASK_LEAF (the
    // ggml_cont route); THIS knob is a refuted alternative. Corrected because the stale header cost
    // real time twice -- once when written, once when re-read during the GRAPH_ALLOC gate.
    //
    // DEFAULT OFF as of the first test: supplying the CORRECT causal+windowed mask here produces
    // DEGENERATE output ('<<<<<...', finish=length, reply_sha1 86517851cc0bda85) -- and that is the
    // SAME hash attempt 0 produced by substituting the leaf directly. Two independent routes that
    // both deliver content-correct masking break the model identically, which argues the empty raw
    // half is NOT a defect on this path. See the retraction note. Opt-in only.
    static const bool raw_mask_own = []() {
        const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_OWN");
        return e && e[0] != '\0' && e[0] != '0';      // default OFF
    }();
    if (raw_mask_own && inp_dsv4->get_raw_kq_mask()) {
        raw_mask = inp_dsv4->get_raw_kq_mask();
        cb(raw_mask, "csa_raw_mask_own", il);
    }

    if (raw_mask_from_leaf && inp_attn->self_kq_mask_swa) {
        // Bisected 2026-08-28: the LEAF is correct on device (-inf=40896, == ds4's rule 256/256)
        // while `_cnv = ggml_cast(leaf, F16)` reads back as 65536 zeros -- a node whose output was
        // never written, because it is built once at the graph HEAD (llama-graph.cpp:3450) and the
        // streaming executor computes per-layer node RANGES. Same structure as the M3 defect at
        // llama-context.cpp:19024. Rebuild the cast HERE, inside the range that consumes it.
        //
        // The first attempt at this substituted the leaf directly and produced degenerate output.
        // Likely cause, from llm_set_attn_mask_shape's own comment: "the streaming mask leaf is
        // sized at constant-max (LLAMA_PREFILL_STREAMING_MASK_MAXKV) but its cached cross-backend
        // copy carries a per-ubatch (smaller) ne". Concatenating a constant-max leaf against a
        // per-ubatch compressed mask misaligns the two halves -- which is exactly what garbage
        // output looks like. So take the LIVE region explicitly rather than the whole leaf.
        // [LEAFCONT 2026-08-28] ALWAYS take a ggml_cont, not only when the shapes differ.
        //
        // This one line is why every attempt at the leaf route died on
        // `llama-kv-cache.cpp:2294 GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))`. When the
        // shapes MATCH -- which is the normal case, both are [256,256] on the streaming pass -- the
        // branch below was skipped and `raw_mask = leaf` handed the RAW LEAF straight to the concat
        // and thence to a GPU op. That forces the scheduler to place the leaf device-side, and
        // set_input_kq_mask then writes it through a raw host pointer and asserts. The shape guard
        // made the cont look like a shape fix when it is really what keeps the leaf off the GPU.
        //
        // Evidence this works: `csa_raw_mask_leaf` a few lines below is exactly `ggml_cont(leaf)`
        // with no view, it lives in this same layer graph, and NODE_SCAN reads it back at
        // inf=40896 -- the correct 128-window mask -- with no assert. The cont is scheduled on the
        // host backend (its result is what crosses to the device), so the leaf itself never moves.
        //
        // Keep the view for the genuine shape-mismatch case; just never skip the cont.
        ggml_tensor * leaf = inp_attn->self_kq_mask_swa;
        if (raw_mask && (leaf->ne[0] != raw_mask->ne[0] || leaf->ne[1] != raw_mask->ne[1])) {
            leaf = ggml_view_2d(ctx0, leaf, raw_mask->ne[0], raw_mask->ne[1], leaf->nb[1], 0);
        }
        raw_mask = ggml_cont(ctx0, leaf);
        cb(raw_mask, "csa_raw_mask_fixed", il);
    }
    ggml_tensor * csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    // BISECT (2026-08-28): the leaf self_kq_mask_swa is CORRECT on device (-inf=40896, == ds4's
    // rule 256/256) yet what the FA kernel reads is missing exactly those 40896 maskings. With
    // fa=on there are TWO cast nodes between them:
    //     leaf(F32) -> self_kq_mask_swa_cnv = ggml_cast(leaf, F16)   [llama-graph.cpp:3450]
    //               -> cast BACK to csa_mask->type (F32)             [immediately below]
    // -inf survives both conversions mathematically, so one of the NODES is not producing what it
    // should. Pin the F16 _cnv here, BEFORE the back-cast, to say which:
    //     _cnv correct -> the F16->F32 back-cast is at fault
    //     _cnv empty   -> the F32->F16 cast (or its scheduling) is
    static const bool comp_dump_pin_cnv = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    if (comp_dump_pin_cnv && il == dsv4_comp_dump_layer()) {
        ggml_tensor * cnv_dump = ggml_cont(ctx0, raw_mask);   // still the _cnv, pre-back-cast
        cb(cnv_dump, "csa_raw_mask_cnv", il);
        ggml_set_output(cnv_dump);
        ggml_build_forward_expand(gf, cnv_dump);
    }

    if (raw_mask->type != csa_mask->type) {
        raw_mask = ggml_cast(ctx0, raw_mask, csa_mask->type);
    }

    // 2026-08-28: the dumped CONCAT mask carries 57408 -inf entries, ALL of them in the compressed
    // half -- the raw half is 0.0 everywhere, i.e. every query token may attend every raw cell.
    // That contradicts the loaded config (`n_swa = 128`, `causal attn = 1`) and it contradicts the
    // model's own behaviour (a needle 104 tokens back is recalled exactly, output is fluent), so it
    // is either a real masking defect or an artefact of what gets dumped. Pin the raw mask ALONE,
    // before the concat, to tell those apart.
    //
    // A ggml_cont copy rather than the tensor itself: `raw_mask` is a graph INPUT, and today's HCA
    // run proved that pinning a non-materialised tensor reads back recycled memory (32767 values
    // above 1e30 next to a sane pooled result). A cont is a real node with its own buffer.
    static const bool comp_dump_pin_raw_mask = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    if (comp_dump_pin_raw_mask && il == dsv4_comp_dump_layer()) {
        ggml_tensor * raw_mask_dump = ggml_cont(ctx0, raw_mask);
        cb(raw_mask_dump, "csa_raw_mask", il);
        ggml_set_output(raw_mask_dump);
        ggml_build_forward_expand(gf, raw_mask_dump);

        // ALSO the LEAF. `get_kq_mask_swa()` returns self_kq_mask_swa_CNV -- a converted COPY of the
        // leaf that set_input_kq_mask actually writes (llama-graph.cpp:~783 calls it with
        // cparams.causal_attn, which the server log confirms is 1). The copy came back with zero
        // -inf entries, so the question is whether the leaf is masked and the copy loses it, or
        // neither is masked. Dumping both in ONE run answers it without a second build.
        if (inp_attn->self_kq_mask_swa) {
            ggml_tensor * leaf_dump = ggml_cont(ctx0, inp_attn->self_kq_mask_swa);
            cb(leaf_dump, "csa_raw_mask_leaf", il);
            ggml_set_output(leaf_dump);
            ggml_build_forward_expand(gf, leaf_dump);
        }
    }

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    cb(kq_mask, "csa_lid_kq_mask", il);

    // ALIGNMENT CHECK. k_all is concat(raw_k, csa_k) along dim 2 and kq_mask is
    // concat(raw_mask, csa_mask) along dim 0, so the mask's dim 0 must equal raw_k->ne[2] +
    // csa_k->ne[2] EXACTLY and in that order. If the raw mask is padded wider than the raw window
    // actually is, the surplus entries land on the first compressed rows and silently mask (or
    // unmask) the wrong content -- which would look exactly like this bug: the raw window keeps
    // answering correctly while anything that must come back through a compressed row does not.
    // Logged rather than reasoned about, and gated on the dump env so it costs nothing normally.
    static const bool comp_dump_pin_attn = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    if (comp_dump_pin_attn && il == dsv4_comp_dump_layer()) {
        fprintf(stderr, "DSV4_ALIGN il=%d: raw_k ne2=%lld csa_k ne2=%lld k_all ne2=%lld | "
                        "raw_mask ne0=%lld csa_mask ne0=%lld kq_mask ne0=%lld | %s\n",
                        il, (long long) raw_k->ne[2], (long long) csa_k->ne[2],
                        (long long) k_all->ne[2], (long long) raw_mask->ne[0],
                        (long long) csa_mask->ne[0], (long long) kq_mask->ne[0],
                        (kq_mask->ne[0] == k_all->ne[2]) ? "ALIGNED" : "*** MISALIGNED ***");
        fflush(stderr);
        // Pin them on the PREFILL path too. The decode-side LID_PROBE dumps of these same tensors
        // come back as recycled buffers (mask with 1024 distinct small floats, output full of NaN
        // and +-3.4e38), so nothing can be concluded from them. The COMP_DUMP path is the one that
        // reproduces a numpy recomputation to float32 epsilon, so ask the question there instead:
        // how many all-zero (never-written) cache rows does the mask leave ALLOWED?
        ggml_set_output(k_all);
        ggml_set_output(kq_mask);
    }

    // Keep a handle on the PRE-CAST mask. The decode dump shows the mask attention consumes holding
    // NaN where -inf belongs (485 NaN, 0 -inf, finite values all 0.0) and the attention output then
    // entirely NaN. That is either produced by this f32->f16 cast or already present upstream, and
    // the two have completely different fixes -- so dump both and let the numbers say which.
    ggml_tensor * kq_mask_pre_cast = kq_mask;

    // [NO_F16_MASK 2026-08-28] LLAMA_DSV4_NO_F16_MASK (default off) leaves the concatenated mask in
    // F32 while flash attention stays ON.
    //
    // One of only two things `fa=off` changes that remain untested. fa=off RECALLS the needle (n=2,
    // byte-identical replies) and fa=on does not (n=6), with the mask now verified correct
    // (inf=40896 on both compressed paths, zero NaN) and the attention sink excluded. This cast is
    // the difference in the TYPE reaching the kernel: fa=off never performs it.
    //
    // NOT about precision loss in the mask itself -- every entry is exactly {0, -inf} and both
    // survive f16 exactly (build_top_k_mask fills -INFINITY, writes 0 at the top-k rows, then ADDs
    // the base mask). It is about an f16 mask meeting an f16 VKQ accumulator, the regime
    // fattn-mma-f16.cuh:15-34 documents as having produced needle loss before.
    //
    // Caveat to check at runtime: ggml_flash_attn_ext may REQUIRE an f16 mask. If this asserts or
    // silently falls back, the variable cannot be isolated this way -- cast to f16 and back instead,
    // which exercises the round-trip without changing the type the kernel sees.
    static const bool no_f16_mask = []() {
        const char * e = std::getenv("LLAMA_DSV4_NO_F16_MASK");
        return e != nullptr && e[0] != 0 && e[0] != '0';
    }();
    // Gate to PREFILL (nt > 1). Measured 2026-08-28: without this the knob also strips the cast on
    // DECODE graphs, where LLAMA_DSV4_PREFILL_DECOMP does NOT apply (it requires n_tokens > 1) -- so
    // the FA kernel still runs and asserts `mask->type == GGML_TYPE_F16` (fattn-common.cuh:826).
    // The two knobs must cover the SAME graphs or they fight each other; this makes the mask stay
    // F32 exactly where the decomposed path replaces the kernel.
    if (cparams.flash_attn && no_f16_mask && kq_mask->ne[1] > 1) {
        static int nf16_dbg = 0;
        if (nf16_dbg < 3) {
            nf16_dbg++;
            fprintf(stderr, "*** NO_F16_MASK: kept kq_mask as %s at il=%d (fa still ON)\n",
                    ggml_type_name(kq_mask->type), il);
            fflush(stderr);
        }
    } else if (cparams.flash_attn && kq_mask->type != GGML_TYPE_F16) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
    }

    // CANDIDATE-A EXPERIMENT (LLAMA_DSV4_ZERO_DECODE_SINK, default off): drop the attention sink on the
    // CSA path at DECODE (nt==1). If far-recall RETURNS, the sink over-suppresses the needle weight vs
    // the correct-recalling Mac (whose mlx-vlm attn_sink may stay at its zeros init) => sink is bug#2's
    // consumption mechanism. Diagnostic only, env-gated.
    static const bool zero_decode_sink = []() {
        const char * e = std::getenv("LLAMA_DSV4_ZERO_DECODE_SINK");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    // [ZERO_PREFILL_SINK 2026-08-28] The PREFILL-side counterpart, which did not exist.
    //
    // `LLAMA_DSV4_ZERO_DECODE_SINK` above is gated to nt==1, and on the shipped preset
    // (382f322e21c2, RTX prefill + Mac decode) DECODE RUNS ON THE MAC -- so that lever cannot touch
    // anything the RTX computes. Every needle result on this preset is decided by PREFILL.
    //
    // Why sinks are now the live suspect: with the raw mask FIXED and verified (inf=40896 at layer
    // 42, CSA and HCA both, zero NaN, fluent output) the needle is STILL lost with fa=on while
    // fa=off recalls. Mask plumbing is therefore excluded, and what remains among the things fa=off
    // changes is the FA kernel, the SINK path, and accumulation precision. Sinks reach the FA kernel
    // by a completely different route (`ggml_flash_attn_ext_add_sinks`) than the non-FA path, so a
    // sink handled wrongly under FA would reproduce exactly this signature.
    //
    // This is the first time the sink can be tested with the mask CONTROLLED rather than confounded:
    // every previous FA-side experiment ran against an all-visible mask.
    static const bool zero_prefill_sink = []() {
        const char * e = std::getenv("LLAMA_DSV4_ZERO_PREFILL_SINK");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    ggml_tensor * eff_sinks = ((zero_decode_sink  && cur->ne[1] == 1) ||
                               (zero_prefill_sink && cur->ne[1] >  1)) ? nullptr : sinks;
    if (zero_prefill_sink && cur->ne[1] > 1) {
        static int zps_dbg = 0;
        if (zps_dbg < 3) {
            zps_dbg++;
            fprintf(stderr, "*** ZERO_PREFILL_SINK: dropped CSA sink at il=%d nt=%lld\n",
                    il, (long long) cur->ne[1]);
            fflush(stderr);
        }
    }

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, eff_sinks, nullptr, kq_scale, il);
    cb(out, "attn_csa_lid", il);

    // PREFILL-side attention pins. The decode-side probe is a dead end on this route (its readback
    // returns recycled buffers; a control capture of the logits reads 100% NaN while sampling works),
    // but the prefill COMP_DUMP path is trustworthy -- it reproduces numpy to float32 epsilon. And
    // the attention over compressed rows is one of the only two things left: every stage of the
    // compressor is now verified correct with respect to its inputs, yet recall still fails.
    //
    // Distinct names from the decode pins so a name can never resolve to the wrong graph, and
    // ggml_set_output + forward_expand because a cb() NAME alone leaves the buffer free to be
    // recycled before the post-layer readback (that mistake produced a bogus rel=133 "divergence"
    // on the projection check).
    static const bool comp_dump_pin_prefill_attn = []() {
        const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
        return e && e[0] != 0;
    }();
    if (comp_dump_pin_prefill_attn && il == dsv4_comp_dump_layer() && cur->ne[1] > 1) {
        cb(q,   "csa_pre_q",   il);
        cb(out, "csa_pre_out", il);
        ggml_set_output(q);
        ggml_set_output(out);
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, out);
    }

    // LLAMA_DSV4_LID_PROBE (bug#2, 2026-07-18): pin the CSA attention CONSUMPTION tensors at DECODE (nt==1)
    // so the clean LID_PROBE post-decode dump (llama_dsv4_lid_probe_dump) can raw-byte them. Offline the
    // oracle recomputes q@k_all -> (masked joint) softmax -> @k_all and compares to attn_csa_lid: this is the
    // decode attention MATH - the last un-verified bug#2 suspect after compression+selection+query were all
    // verified. k_all is already named "csa_k_all", out "attn_csa_lid"; name the main decode q here.
    static const bool lid_probe_pin_csa = []() {
        const char * e = std::getenv("LLAMA_DSV4_LID_PROBE");
        return e && e[0] != '\0';
    }();
    if (lid_probe_pin_csa && cur->ne[1] == 1) {
        // NAME THE TENSOR THAT IS ACTUALLY PINNED. This block used to pin `kq_mask` while the name
        // "csa_lid_kq_mask" had been attached earlier to the PRE-CAST tensor -- so the probe resolved
        // the name to an UNPINNED tensor whose buffer was recycled, and dumped 1024 distinct floats
        // in +-0.08 where a mask can only be 0 or -inf. Every conclusion from that dump was void.
        //
        // Decode-only names, distinct from the prefill ones, so `ggml_graph_get_tensor` cannot
        // resolve to a same-named tensor from a graph that never computed these.
        cb(q,       "csa_dec_q",    il);
        cb(k_all,   "csa_dec_k",    il);
        cb(kq_mask, "csa_dec_mask", il);   // post-cast: the mask attention actually consumes
        cb(kq_mask_pre_cast, "csa_dec_mask_pre", il);   // pre-cast f32, to locate where NaN enters
        cb(out,     "csa_dec_out",  il);
        ggml_set_output(kq_mask_pre_cast);
        ggml_build_forward_expand(gf, kq_mask_pre_cast);
        ggml_set_output(q);
        ggml_set_output(k_all);
        ggml_set_output(kq_mask);
        ggml_set_output(out);
        // ggml_set_output alone marks intent; it does not guarantee the node is scheduled or that
        // its buffer survives. `csa_dec_out` came back 32768/32768 non-finite -- allocated, never
        // written. Expand each pinned tensor into the graph explicitly so it is a real computed
        // node of THIS graph, which is what the dump reads back.
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, k_all);
        ggml_build_forward_expand(gf, kq_mask);
        ggml_build_forward_expand(gf, out);
    }

    return out;
}

ggml_tensor * llm_build_deepseek4::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_hca = inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);
    GGML_ASSERT(inp_attn->self_k_rot_swa == nullptr);
    GGML_ASSERT(inp_attn->self_v_rot_swa == nullptr);

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_context * mctx_swa = inp_attn->mctx->get_swa();

    ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

    ggml_tensor * raw_k = mctx_swa->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = inp_dsv4->mctx->get_hca()->get_k(ctx0, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask_swa();

    // [LEAFCONT_HCA 2026-08-28] Same fix as the CSA path, applied here for CONSISTENCY.
    //
    // `get_kq_mask_swa()` returns `self_kq_mask_swa_cnv`, a ggml_cast built at the GRAPH HEAD that
    // the layer-major executor never evaluates -- measured at inf=0 (no causal masking, no window)
    // while its source leaf reads inf=40896. The CSA path now takes the leaf via ggml_cont; without
    // the same treatment here, EVEN (CSA) layers would run with a correct mask while ODD (HCA)
    // layers ran with an empty one, which is a state the model was never designed for and is a
    // candidate for the NaN that appears in layers 23..41 once the CSA mask is corrected.
    //
    // ggml_cont, not the leaf itself: handing the raw leaf to a GPU op forces it device-side and
    // trips GGML_ASSERT(ggml_backend_buffer_is_host) in set_input_kq_mask. The cont is scheduled on
    // the host backend and only its RESULT crosses to the device, so the leaf never moves. That one
    // distinction is what made the CSA fix work after four failed attempts.
    //
    // Shares the CSA knob so both compressed paths move together -- masking half the layers and not
    // the other half is exactly the inconsistency this is meant to remove.
    {
        static const bool raw_mask_from_leaf_hca = []() {
            const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_LEAF");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        if (raw_mask_from_leaf_hca && inp_attn->self_kq_mask_swa) {
            ggml_tensor * leaf = inp_attn->self_kq_mask_swa;
            if (raw_mask && (leaf->ne[0] != raw_mask->ne[0] || leaf->ne[1] != raw_mask->ne[1])) {
                leaf = ggml_view_2d(ctx0, leaf, raw_mask->ne[0], raw_mask->ne[1], leaf->nb[1], 0);
            }
            raw_mask = ggml_cont(ctx0, leaf);
            cb(raw_mask, "hca_raw_mask_fixed", il);
        }
    }

    ggml_tensor * hca_mask = inp_hca.kq_mask;
    if (raw_mask->type != hca_mask->type) {
        raw_mask = ggml_cast(ctx0, raw_mask, hca_mask->type);
    }

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    // [NO_F16_MASK 2026-08-28] LLAMA_DSV4_NO_F16_MASK (default off) leaves the concatenated mask in
    // F32 while flash attention stays ON.
    //
    // One of only two things `fa=off` changes that remain untested. fa=off RECALLS the needle (n=2,
    // byte-identical replies) and fa=on does not (n=6), with the mask now verified correct
    // (inf=40896 on both compressed paths, zero NaN) and the attention sink excluded. This cast is
    // the difference in the TYPE reaching the kernel: fa=off never performs it.
    //
    // NOT about precision loss in the mask itself -- every entry is exactly {0, -inf} and both
    // survive f16 exactly (build_top_k_mask fills -INFINITY, writes 0 at the top-k rows, then ADDs
    // the base mask). It is about an f16 mask meeting an f16 VKQ accumulator, the regime
    // fattn-mma-f16.cuh:15-34 documents as having produced needle loss before.
    //
    // Caveat to check at runtime: ggml_flash_attn_ext may REQUIRE an f16 mask. If this asserts or
    // silently falls back, the variable cannot be isolated this way -- cast to f16 and back instead,
    // which exercises the round-trip without changing the type the kernel sees.
    static const bool no_f16_mask = []() {
        const char * e = std::getenv("LLAMA_DSV4_NO_F16_MASK");
        return e != nullptr && e[0] != 0 && e[0] != '0';
    }();
    // Gate to PREFILL (nt > 1). Measured 2026-08-28: without this the knob also strips the cast on
    // DECODE graphs, where LLAMA_DSV4_PREFILL_DECOMP does NOT apply (it requires n_tokens > 1) -- so
    // the FA kernel still runs and asserts `mask->type == GGML_TYPE_F16` (fattn-common.cuh:826).
    // The two knobs must cover the SAME graphs or they fight each other; this makes the mask stay
    // F32 exactly where the decomposed path replaces the kernel.
    if (cparams.flash_attn && no_f16_mask && kq_mask->ne[1] > 1) {
        static int nf16_dbg = 0;
        if (nf16_dbg < 3) {
            nf16_dbg++;
            fprintf(stderr, "*** NO_F16_MASK: kept kq_mask as %s at il=%d (fa still ON)\n",
                    ggml_type_name(kq_mask->type), il);
            fflush(stderr);
        }
    } else if (cparams.flash_attn && kq_mask->type != GGML_TYPE_F16) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
    }

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llm_build_deepseek4::build_raw_attention(
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const bool is_swa = hparams.is_swa(il);

    ggml_tensor * k_rot = is_swa ? inp_attn->self_k_rot_swa : inp_attn->self_k_rot;
    ggml_tensor * v_rot = is_swa ? inp_attn->self_v_rot_swa : inp_attn->self_v_rot;
    GGML_ASSERT(v_rot == nullptr);

    if (k_rot) {
        q  = ggml_mul_mat(ctx0, k_rot, q);
        kv = ggml_mul_mat(ctx0, k_rot, kv);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_context * mctx_cur = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
    const auto & k_idxs = is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, k_idxs, il));

    const auto & kq_mask = is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask();
    // Mutable alias: the LEAFCONT/RAWMASK_F32 blocks below substitute a cont of the leaf, and the
    // binding above is a const reference.
    ggml_tensor * kq_mask_eff = kq_mask;

    // [LEAFCONT_RAW 2026-08-28] The RAW layers need the same mask fix as CSA and HCA.
    //
    // **Every DSV4 layer is SWA.** llama-model.cpp calls `hparams.set_swa_pattern(0)`, and
    // set_swa_pattern's body is `swa_layers[il] = n_pattern == 0 || ...` -- with n_pattern 0 the
    // first clause is unconditionally true. So `is_swa(il)` holds for layers 0/1/43 too, and this
    // path reads `get_kq_mask_swa()` -> `self_kq_mask_swa_cnv`, the head-built ggml_cast the
    // layer-major executor never evaluates. Measured on 2026-08-28: `raw_kq_mask_obs-1` reads
    // **inf=0** across 9 firings -- no causal masking and no window on the raw layers either.
    //
    // LEAFCONT was applied to build_csa_lid_attention and build_hca_attention but NOT here, so the
    // raw layers kept consuming the empty mask. That is the third partial-coverage instance in this
    // session (after CSA-without-HCA, and two knobs with mismatched graph guards), and it matches
    // the localisation exactly: giving the raw layers a cont-of-the-leaf mask (LLAMA_DSV4_RAW_MASK_F32
    // + decomp) restored recall at n=2, byte-identical to global fa=off. That knob's real effect was
    // never the dtype -- it was delivering a CORRECT mask.
    //
    // Cast back to the consumed mask's type afterwards so flash attention still gets f16
    // (fattn-common.cuh:826 asserts `mask->type == GGML_TYPE_F16`). ggml_cont of the leaf rather
    // than the leaf itself: handing the raw leaf to a GPU op forces it device-side and trips
    // GGML_ASSERT(ggml_backend_buffer_is_host) in set_input_kq_mask.
    //
    // Shares LLAMA_DSV4_RAW_MASK_LEAF with the compressed paths so all three move together --
    // fixing a subset is what caused the NaN cascade earlier today.
    {
        static const bool raw_leafcont = []() {
            const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_LEAF");
            return e != nullptr && e[0] != 0 && e[0] != '0';
        }();
        ggml_tensor * leaf_src = is_swa ? inp_attn->self_kq_mask_swa : inp_attn->self_kq_mask;
        if (raw_leafcont && leaf_src && kq_mask_eff && kq_mask_eff->ne[1] > 1) {
            ggml_tensor * lm = leaf_src;
            if (lm->ne[0] != kq_mask_eff->ne[0] || lm->ne[1] != kq_mask_eff->ne[1]) {
                lm = ggml_view_2d(ctx0, lm, kq_mask_eff->ne[0], kq_mask_eff->ne[1], lm->nb[1], 0);
            }
            ggml_tensor * fixed = ggml_cont(ctx0, lm);
            if (fixed->type != kq_mask_eff->type) {
                fixed = ggml_cast(ctx0, fixed, kq_mask_eff->type);
            }
            kq_mask_eff = fixed;
            cb(kq_mask_eff, "raw_kq_mask_leafcont", il);
            static int rlc_dbg = 0;
            if (rlc_dbg < 3) {
                rlc_dbg++;
                fprintf(stderr, "*** LEAFCONT_RAW: raw mask from leaf as %s at il=%d nt=%lld\n",
                        ggml_type_name(kq_mask_eff->type), il, (long long) kq_mask_eff->ne[1]);
                fflush(stderr);
            }
        }
    }

    // [RAWMASK_OBS 2026-08-28] Make the RAW layers' consumed mask observable.
    //
    // Why this is needed: LLAMA_DLM_NODE_NAN_SCAN matches nodes by their per-layer "-<il>" name
    // suffix, so it can ONLY see cb()-named per-layer nodes. The `_cnv` masks are built ONCE AT THE
    // GRAPH HEAD with no layer suffix and are therefore STRUCTURALLY INVISIBLE to it. Measured: a
    // layer-0 scan returned 55 nodes and ZERO with "mask" in the name -- which says nothing about
    // the base mask, though it looks like it does.
    //
    // The CSA path is observable only because it happens to build a per-layer ggml_cont COPY for its
    // dump. Do the same here so the RAW half can be checked. Then a scan reports:
    //     raw_kq_mask_obs-<il>  inf == 0        -> the base/SWA cast is dead on this path too, and
    //                                             the defect is not confined to the compressed path
    //     inf == the causal count               -> the cast DOES execute here, so only the CSA
    //                                             route is broken
    // That distinction decides whether this is a DSV4-CSA bug or a whole-fork layer-major+FA bug,
    // which is the open question after the owner reported that ISWA models have never worked.
    //
    // ggml_cont (not the tensor itself): pinning a view/input does not stop the allocator reusing
    // the base, and consuming a mask LEAF directly moves it device-side and trips the host-buffer
    // assert. A cont copy consumed by nothing does neither. Env-gated so it is inert by default.
    {
        static const bool raw_mask_obs = []() {
            const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_OBS");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        if (raw_mask_obs && kq_mask) {
            ggml_tensor * obs = ggml_cont(ctx0, kq_mask);
            cb(obs, "raw_kq_mask_obs", il);
            ggml_set_output(obs);
            ggml_build_forward_expand(gf, obs);
        }
    }

    // [RAWMASK_F32 2026-08-28] LLAMA_DSV4_RAW_MASK_F32 (default off): give the RAW layers an F32
    // mask, the same treatment the compressed paths get from LEAFCONT.
    //
    // The raw path reads `_cnv`, which is F16 under fa=on. That produces an asymmetry the compressed
    // path does not have:
    //     fa=off        raw layers: non-FA kernel + F32 mask  -> RECALLS, coherent
    //     DECOMP_RAW=1  raw layers: non-FA kernel + F16 mask  -> DEGENERATE
    // whereas on the COMPRESSED path the dtype was measured IRRELEVANT (decomp with f16 and with f32
    // gave byte-identical replies, 251a936d3c1a8934).
    //
    // Combined with LLAMA_DSV4_PREFILL_DECOMP_RAW this reproduces fa=off semantics on exactly the
    // raw layers, which is the last untested combination and the only way left to test the FA kernel
    // there -- decomposing them with an F16 mask degenerates, so the kernel question cannot be
    // reached without this.
    //
    // ggml_cont of the leaf, not the leaf itself: handing the raw leaf to a GPU op forces it
    // device-side and trips GGML_ASSERT(ggml_backend_buffer_is_host) in set_input_kq_mask. Prefill
    // only (kq_mask->ne[1] > 1) so it covers the same graphs as the decomp knob -- mismatched guards
    // already cost one assert today.
    {
        static const bool raw_mask_f32 = []() {
            const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_F32");
            return e != nullptr && e[0] != 0 && e[0] != '0';
        }();
        ggml_tensor * leaf_src = is_swa ? inp_attn->self_kq_mask_swa : inp_attn->self_kq_mask;
        if (raw_mask_f32 && leaf_src && kq_mask && kq_mask->ne[1] > 1) {
            ggml_tensor * f32m = leaf_src;
            if (f32m->ne[0] != kq_mask->ne[0] || f32m->ne[1] != kq_mask->ne[1]) {
                f32m = ggml_view_2d(ctx0, f32m, kq_mask->ne[0], kq_mask->ne[1], f32m->nb[1], 0);
            }
            kq_mask_eff = ggml_cont(ctx0, f32m);
            cb(kq_mask_eff, "raw_kq_mask_f32", il);
            static int rmf_dbg = 0;
            if (rmf_dbg < 3) {
                rmf_dbg++;
                fprintf(stderr, "*** RAWMASK_F32: raw mask as %s at il=%d nt=%lld\n",
                        ggml_type_name(kq_mask_eff->type), il, (long long) kq_mask_eff->ne[1]);
                fflush(stderr);
            }
        }
    }

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask_eff, sinks, nullptr, kq_scale, il);
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llm_build_deepseek4::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_attn_kv_iswa * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k;
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(q_pe, "q_pe", il);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_idxs) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape = layer.attn_comp_ape;
        if (ape->type != GGML_TYPE_F32) {
            ape = ggml_cast(ctx0, ape, GGML_TYPE_F32);
        }

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, ape, inp_dsv4->get_hca().state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);

    }

    if (ratio == DSV4_CSA_RATIO && inp_dsv4->get_csa().state_idxs) {
        // Pin the compressor's INPUT. The pooling was verified exact against numpy, but only GIVEN
        // its inputs; the projection that produces those inputs -- this matmul -- has never been
        // checked. With the input dumped alongside the already-dumped output and the weights (which
        // are byte-identical to the GGUF), the whole step becomes a self-contained numpy check on
        // the RTX alone: csa_state_kv ?= attn_comp_wkv @ cur. No Mac, no cross-machine dumping.
        cb(cur, "csa_comp_input", il);
        // PIN IT. A cb() name alone is not enough: the dump is a POST-LAYER readback, and by then
        // ggml has recycled the buffer of any tensor that is not a graph output. Naming without
        // pinning produced a "divergence" of 133x relative on the projection check -- which is a
        // recycled buffer, not a defect, and I nearly reported it as one. Every other dumped tensor
        // here is pinned; this one was the exception.
        {
            static const bool pin_comp_input = []() {
                const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
                return e && e[0] != 0;
            }();
            if (pin_comp_input && il == dsv4_comp_dump_layer()) {
                ggml_set_output(cur);
                ggml_build_forward_expand(gf, cur);
            }
        }
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;
        if (csa_ape->type != GGML_TYPE_F32) {
            csa_ape = ggml_cast(ctx0, csa_ape, GGML_TYPE_F32);
        }

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        // Pin the POSITION INDICES and the pre-APE score. The scores decide the pooling weights, and
        // this term is indexed by position -- so wrong indices make each compressed row pool the
        // wrong tokens, which is precisely a position-dependent retrieval failure. And the measured
        // boundary IS positional: a fact inside the 128-token raw window is recalled, one just
        // outside it is not.
        //
        // state_pos can be sanity-checked with NO reference at all (monotonic? in range? distinct?),
        // which beats another cross-backend diff -- an input-side check cannot be defeated by a
        // recycled buffer or a misaligned batch.
        static const bool pin_ape = []() {
            const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
            return e && e[0] != 0;
        }();
        if (pin_ape && il == dsv4_comp_dump_layer()) {
            cb(inp_dsv4->get_csa().state_pos, "csa_state_pos", il);
            cb(csa_state_score, "csa_state_score_pre_ape", il);
            cb(csa_ape_rows,    "csa_ape_rows", il);
            ggml_set_output(inp_dsv4->get_csa().state_pos);
            ggml_set_output(csa_state_score);
            ggml_set_output(csa_ape_rows);
            ggml_build_forward_expand(gf, csa_state_score);
            ggml_build_forward_expand(gf, csa_ape_rows);
        }
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0,
                inp_dsv4->mctx->get_csa_state()->get_kv(ctx0, il), csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0,
                inp_dsv4->mctx->get_csa_state()->get_score(ctx0, il), csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);
        // QAT: FP8 round trip on the ATTENTION compressed KV (ds4.c:13288). This is the content
        // attention reads back for pooled rows, so it -- not the indexer's ranking -- is what
        // decides whether a fact survives compression. Applied over the NOPE part only; the RoPE
        // tail that build_overlap_compressed_kv_from_state just wrote is left alone, matching the
        // reference's `dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT)`.
        if (dsv4_qat_enabled()) {
            kv_comp_csa_state = ggml_dsv4_act_qat(ctx0, kv_comp_csa_state,
                    GGML_DSV4_QAT_FP8_E4M3, (int) hparams.n_rot);
            cb(kv_comp_csa_state, "csa_state_compress_qat", il);
        }
        cb(kv_comp_csa_state, "csa_state_compress_final", il);  // unique name for the round-11 oracle

        // ROUND-11 (LLAMA_DSV4_COMP_DUMP): pin the CSA-attention compression chain at il==2 —
        // the LID chain was exonerated in round-5 but THIS chain (what attention actually reads
        // for pooled rows) was never numerically verified.
        static const bool comp_dump_pin_csa = []() {
            const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
            return e && e[0] != '\0';
        }();
        if (comp_dump_pin_csa && il == dsv4_comp_dump_layer()) {
            ggml_set_output(csa_state_kv);
            ggml_set_output(csa_state_score);
            ggml_set_output(kv_comp_csa_state);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        csa_state_kv    = dsv4_with_zero_dep(ctx0, csa_state_kv,    kv_comp_csa_state);
        csa_state_score = dsv4_with_zero_dep(ctx0, csa_state_score, kv_comp_csa_state);

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;
        if (lid_ape->type != GGML_TYPE_F32) {
            lid_ape = ggml_cast(ctx0, lid_ape, GGML_TYPE_F32);
        }

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0,
                inp_dsv4->mctx->get_lid_state()->get_kv(ctx0, il), lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0,
                inp_dsv4->mctx->get_lid_state()->get_score(ctx0, il), lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = ggml_mul_mat(ctx0, inp_dsv4->get_lid().k_rot, kv_comp_lid_state);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
            // QAT: the Hadamard is only HALF of the model's graph -- the official DSV4 graph runs the
            // FP4 activation round trip immediately after it, on the indexer compressor KV as well as
            // on the indexer Q (ds4.c:13290). Without it the stored indexer keys are not the ones the
            // weights were trained against. See dsv4_indexer_qat_enabled() for the gate.
            if (dsv4_indexer_qat_enabled()) {
                kv_comp_lid_state = ggml_dsv4_act_qat(ctx0, kv_comp_lid_state, GGML_DSV4_QAT_FP4_E2M1, 0);
                cb(kv_comp_lid_state, "lid_state_compress_qat", il);
            }
        }

        // LLAMA_DSV4_COMP_DUMP: pin the compression-chain tensors as graph outputs so the
        // post-layer dump reads live values, not recycled intermediate buffers (round-5
        // oracle; first dump attempt read garbage precisely because these are transient).
        static const bool comp_dump_pin = []() {
            const char * e = std::getenv("LLAMA_DSV4_COMP_DUMP");
            return e && e[0] != '\0';
        }();
        if (comp_dump_pin && il == dsv4_comp_dump_layer()) {
            ggml_set_output(lid_state_kv);
            ggml_set_output(lid_state_score);
            ggml_set_output(kv_comp_lid_state);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));

        lid_state_kv    = dsv4_with_zero_dep(ctx0, lid_state_kv,    kv_comp_lid_state);
        lid_state_score = dsv4_with_zero_dep(ctx0, lid_state_score, kv_comp_lid_state);

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    ggml_tensor * hca_state_dep = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        ggml_tensor * hca_source_kv = ggml_concat(ctx0,
                inp_dsv4->mctx->get_hca_state()->get_kv(ctx0, il), hca_state_kv, 1);
        ggml_tensor * hca_source_score = ggml_concat(ctx0,
                inp_dsv4->mctx->get_hca_state()->get_score(ctx0, il), hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_dsv4->get_hca().state_read_idxs,
                inp_dsv4->get_hca().state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        // Same FP8 round trip as the CSA path above: HCA rows are the same attention-side
        // compressed KV at a different ratio, and ds4 selects the transform by HEAD WIDTH
        // (ds4.c:13287), not by which compressor produced the row.
        if (dsv4_qat_enabled()) {
            kv_comp_hca = ggml_dsv4_act_qat(ctx0, kv_comp_hca,
                    GGML_DSV4_QAT_FP8_E4M3, (int) hparams.n_rot);
            cb(kv_comp_hca, "hca_state_compress_qat", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_hca()->cpy_k(ctx0,
                    kv_comp_hca, inp_dsv4->get_hca().state_write_idxs, il));
        hca_state_dep = kv_comp_hca;
    }

    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state_kv    = dsv4_with_zero_dep(ctx0, hca_state_kv,    hca_state_dep);
        hca_state_score = dsv4_with_zero_dep(ctx0, hca_state_score, hca_state_dep);

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, inp_dsv4->get_hca().state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_dsv4->get_hca().state_persist_src_idxs);

        hca_state_kv = inp_dsv4->mctx->get_hca_state()->cpy_kv(ctx0,
                hca_persist_kv, inp_dsv4->get_hca().state_persist_dst_idxs, il);
        hca_state_score = inp_dsv4->mctx->get_hca_state()->cpy_score(ctx0,
                hca_persist_score, inp_dsv4->get_hca().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    ggml_tensor * out = nullptr;
    if (ratio == DSV4_CSA_RATIO &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot &&
            inp_attn->self_k_rot_swa == nullptr &&
            inp_attn->self_v_rot_swa == nullptr) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio == DSV4_HCA_RATIO &&
            inp_dsv4->get_hca().kq_mask &&
            inp_attn->self_k_rot_swa == nullptr &&
            inp_attn->self_v_rot_swa == nullptr) {
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));
    out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    // layer.wo_a is loaded 2D {o_group_dim, o_lora_rank*o_groups} to match mainline GGUFs (teamblobfish);
    // reshape back to the 3D grouped form {o_group_dim, o_lora_rank, o_groups} the batched matmul expects.
    ggml_tensor * wo_a_3d = ggml_reshape_3d(ctx0, layer.wo_a, o_group_dim, o_lora_rank, n_groups);
    ggml_tensor * oa = ggml_mul_mat(ctx0, wo_a_3d, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_attn_kv_iswa * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);
    if (inp_attn->self_kq_mask_swa) {
        ggml_build_forward_expand(gf, inp_attn->self_kq_mask_swa);
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
