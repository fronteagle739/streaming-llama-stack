#include "models.h"

llm_build_gemma4_assistant::llm_build_gemma4_assistant(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);

    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.is_swa(il) && hparams.n_embd_head_k_swa > 0 ? hparams.n_embd_head_k_swa : hparams.n_embd_head_k;
        const int64_t n_head      = hparams.n_head(il);
        const int64_t n_head_kv   = hparams.n_head_kv(il);
        const int     n_rot_l     = hparams.n_rot_layer(il);

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base,
                             freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = model.layers[il].wv ? build_lora_mm(model.layers[il].wv, cur) : Kcur;
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base,
                                 freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        } else {
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, nullptr, nullptr,
                             nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cur = ggml_add(ctx0, cur, inpL);
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        }
        inpL = build_cvec(cur, il);
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
