#include "models.h"

ggml_cgraph * clip_graph_kimivl::build() {
    // (2D positions are applied below via explicit interleaved-RoPE cos/sin tables, so the old
    //  pos_h/pos_w I32 inputs + half-split build_rope_2d are no longer used.)
    // kimi_k25 interpolates the learned 64x64 pos-embed to the patch grid with BICUBIC (no
    // antialias), per Learnable2DInterpPosEmb; clip.cpp's default is BILINEAR+ANTIALIAS which
    // blurs the downsample. Match the reference.
    ggml_tensor * learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BICUBIC);

    // 2026-05-31: Kimi-K2.6 (model_type kimi_k25) vision uses INTERLEAVED 2D-RoPE: adjacent complex
    // pair k = (q[2k], q[2k+1]) is rotated by the WIDTH position when k is even and the HEIGHT
    // position when k is odd (freq θ^(-4*(k/2)/d_head)), i.e. head-dim layout [W0,H0,W1,H1,...]
    // (exactly mlx_vlm kimi_k25 Rope2DPosEmb/apply_rope). clip.cpp's build_rope_2d is HALF-SPLIT and
    // mismatches everywhere except pair 0 -> near-orthogonal features the LM can't OCR. Apply the
    // rope explicitly with host-precomputed cos/sin (filled in clip.cpp set_input), which is
    // unambiguous re: the ggml rope-mode pairing convention.
    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head/2, n_patches);
    ggml_set_name(rope_cos, "kimi_rope_cos");
    ggml_set_input(rope_cos);
    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head/2, n_patches);
    ggml_set_name(rope_sin, "kimi_rope_sin");
    ggml_set_input(rope_sin);

    // build ViT with 2D position embeddings
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        const int64_t dh = cur->ne[0]; // head_dim
        const int64_t nh = cur->ne[1];
        const int64_t np = cur->ne[2];
        cur = ggml_cont(ctx0, cur);
        // adjacent complex pairs (q[2k], q[2k+1]) -> real / imag halves
        ggml_tensor * p4 = ggml_reshape_4d(ctx0, cur, 2, dh/2, nh, np); // [(re,im), pair, head, pos]
        ggml_tensor * re = ggml_cont(ctx0, ggml_view_4d(ctx0, p4, 1, dh/2, nh, np, p4->nb[1], p4->nb[2], p4->nb[3], 0));
        ggml_tensor * im = ggml_cont(ctx0, ggml_view_4d(ctx0, p4, 1, dh/2, nh, np, p4->nb[1], p4->nb[2], p4->nb[3], p4->nb[0]));
        re = ggml_reshape_3d(ctx0, re, dh/2, nh, np);
        im = ggml_reshape_3d(ctx0, im, dh/2, nh, np);
        ggml_tensor * c = ggml_reshape_3d(ctx0, rope_cos, dh/2, 1, np); // broadcast over heads
        ggml_tensor * s = ggml_reshape_3d(ctx0, rope_sin, dh/2, 1, np);
        ggml_tensor * o_re = ggml_sub(ctx0, ggml_mul(ctx0, re, c), ggml_mul(ctx0, im, s));
        ggml_tensor * o_im = ggml_add(ctx0, ggml_mul(ctx0, re, s), ggml_mul(ctx0, im, c));
        // re-interleave -> [head_dim, head, pos]
        o_re = ggml_reshape_4d(ctx0, ggml_cont(ctx0, o_re), 1, dh/2, nh, np);
        o_im = ggml_reshape_4d(ctx0, ggml_cont(ctx0, o_im), 1, dh/2, nh, np);
        ggml_tensor * o = ggml_concat(ctx0, o_re, o_im, 0); // [(re,im), pair, head, pos]
        return ggml_reshape_3d(ctx0, ggml_cont(ctx0, o), dh, nh, np);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // patch_merger
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection norm
        int proj_inp_dim = cur->ne[0];
        cur = ggml_view_2d(ctx0, cur,
            n_embd, cur->ne[1] * scale_factor * scale_factor,
            ggml_row_size(cur->type, n_embd), 0);
        cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = ggml_view_2d(ctx0, cur,
            proj_inp_dim, cur->ne[1] / scale_factor / scale_factor,
            ggml_row_size(cur->type, proj_inp_dim), 0);
        cb(cur, "proj_inp_normed", -1);

        // projection mlp
        if (std::getenv("LLAMA_VISION_DUMP_VITOUT")) {
            fprintf(stderr, "*** PROJ_WEIGHTS ne: mm_1_w=[%lld,%lld] mm_2_w=[%lld,%lld] inp_norm=[%lld] ffn_in=[%lld,%lld]\n",
                (long long) model.mm_1_w->ne[0], (long long) model.mm_1_w->ne[1],
                (long long) model.mm_2_w->ne[0], (long long) model.mm_2_w->ne[1],
                (long long) model.mm_input_norm_w->ne[0],
                (long long) cur->ne[0], (long long) cur->ne[1]);
            fflush(stderr);
        }
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF, // kimi_k25 projector uses exact (erf) GELU, not the tanh approx
            -1);
        cb(cur, "proj_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
