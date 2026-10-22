#include "models.h"

llm_build_gemma4::llm_build_gemma4(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.tok_embd_per_layer) {
        inp_per_layer = project_per_layer_inputs(inpL, get_per_layer_inputs());
        ggml_build_forward_expand(gf, inp_per_layer);
    }

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

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = model.layers[il].wv ? build_lora_mm(model.layers[il].wv, cur) : Kcur;
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                                 freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        } else {
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, nullptr, nullptr,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);
        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

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
                    model.layers[il].ffn_gate_up_exps);
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

        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur);
            cur = ggml_gelu(ctx0, cur);
            ggml_tensor * inp_this_layer = view_2d_slice(inp_per_layer, il);
            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur);
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, NULL, LLM_NORM_RMS, il);
            cur = ggml_add(ctx0, pe_in, cur);
        }

        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        }

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
}

ggml_tensor * llm_build_gemma4::view_2d_slice(ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

ggml_tensor * llm_build_gemma4::get_per_layer_inputs() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    ggml_tensor * inp_per_layer;
    const float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;
        inp_per_layer = ggml_get_rows(ctx0, model.tok_embd_per_layer, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);
        res->add_input(std::move(inp));
    } else {
        const int64_t embd_size = model.tok_embd_per_layer->ne[0];
        ggml_tensor * padding = ggml_view_1d(ctx0, model.tok_embd_per_layer, embd_size, 0);
        inp_per_layer = ggml_cast(ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
    }
    return inp_per_layer;
}

ggml_tensor * llm_build_gemma4::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    ggml_tensor * per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_scale(ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);
    per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm, NULL, LLM_NORM_RMS, -1);

    inp_per_layer = ggml_add(ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    return ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
}
