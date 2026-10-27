#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

// TODO: remove in follow-up PR - move to .cpp files
#include "../llama-memory-recurrent.h"
#include <cmath>

struct llm_graph_context_mamba : public llm_graph_context {
    llm_graph_context_mamba(const llm_graph_params & params);

    virtual ~llm_graph_context_mamba() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;

};

// Base class for RWKV-related models
struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);
};

struct llm_build_afmoe : public llm_graph_context {
    llm_build_afmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_apertus : public llm_graph_context {
    llm_build_apertus(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_arcee : public llm_graph_context {
    llm_build_arcee(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_arctic : public llm_graph_context {
    llm_build_arctic(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_arwkv7 : public llm_build_rwkv7_base {
    llm_build_arwkv7(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_baichuan : public llm_graph_context {
    llm_build_baichuan(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bailingmoe2 : public llm_graph_context {
    llm_build_bailingmoe2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bailingmoe : public llm_graph_context {
    llm_build_bailingmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bert : public llm_graph_context {
    llm_build_bert(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bitnet : public llm_graph_context {
    llm_build_bitnet(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bloom : public llm_graph_context {
    llm_build_bloom(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_chameleon : public llm_graph_context {
    llm_build_chameleon(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_chatglm : public llm_graph_context {
    llm_build_chatglm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_codeshell : public llm_graph_context {
    llm_build_codeshell(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_cogvlm : public llm_graph_context {
    llm_build_cogvlm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_cohere2_iswa : public llm_graph_context {
    llm_build_cohere2_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_command_r : public llm_graph_context {
    llm_build_command_r(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dbrx : public llm_graph_context {
    llm_build_dbrx(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deci : public llm_graph_context {
    llm_build_deci(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deepseek2 : public llm_graph_context {
    llm_build_deepseek2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deepseek : public llm_graph_context {
    llm_build_deepseek(const llama_model & model, const llm_graph_params & params);
};

// DeepSeek-V4 (LLM_ARCH_DEEPSEEK4) — ported from upstream llama.cpp origin/dsv4
// and adapted to this fork's flat llm_build_* / llm_graph_context idiom.
struct llm_build_deepseek4 : public llm_graph_context {
    llm_build_deepseek4(const llama_model & model, const llm_graph_params & params);

    ggml_tensor * build_hc_pre(
            ggml_tensor * x,
            ggml_tensor * hc_fn,
            ggml_tensor * hc_scale,
            ggml_tensor * hc_base,
            ggml_tensor ** post,
            ggml_tensor ** comb,
            int il) const;

    ggml_tensor * build_hc_post(
            ggml_tensor * x,
            ggml_tensor * residual,
            ggml_tensor * post,
            ggml_tensor * comb,
            int il) const;

    ggml_tensor * build_hc_head(
            ggml_tensor * x,
            ggml_tensor * hc_fn,
            ggml_tensor * hc_scale,
            ggml_tensor * hc_base) const;

    ggml_tensor * build_attention(
            const llama_model & model,
            llm_graph_input_dsv4 * inp_dsv4,
            ggml_tensor * cur,
            ggml_tensor * inp_pos,
            int il) const;

    ggml_tensor * build_hca_compressed_kv_from_state(
            ggml_tensor * kv_state,
            ggml_tensor * score_state,
            ggml_tensor * state_read_idxs,
            ggml_tensor * comp_pos,
            ggml_tensor * norm,
            int64_t n_embd_head,
            const char * name,
            int il) const;

    ggml_tensor * build_overlap_compressed_kv_from_state(
            ggml_tensor * kv_state,
            ggml_tensor * score_state,
            ggml_tensor * state_read_idxs,
            ggml_tensor * comp_pos,
            ggml_tensor * norm,
            int64_t ratio,
            int64_t n_embd_head,
            const char * name,
            int il) const;

    ggml_tensor * build_lid_top_k(
            const llama_model & model,
            llm_graph_input_dsv4 * inp_dsv4,
            ggml_tensor * qr,
            ggml_tensor * cur,
            ggml_tensor * inp_pos,
            int il) const;

    ggml_tensor * build_top_k_mask(
            ggml_tensor * kq_mask,
            ggml_tensor * top_k,
            const char * name,
            int il) const;

    ggml_tensor * build_csa_lid_attention(
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
            int il) const;

    ggml_tensor * build_hca_attention(
            llm_graph_input_dsv4 * inp_dsv4,
            llm_graph_input_attn_kv_iswa * inp_attn,
            ggml_tensor * q,
            ggml_tensor * kv,
            ggml_tensor * sinks,
            float kq_scale,
            int il) const;

    ggml_tensor * build_raw_attention(
            llm_graph_input_attn_kv_iswa * inp_attn,
            ggml_tensor * q,
            ggml_tensor * kv,
            ggml_tensor * sinks,
            float kq_scale,
            int il) const;

    ggml_tensor * build_hc_weighted_sum(
            ggml_tensor * x,
            ggml_tensor * weights) const;

    ggml_tensor * build_hc_sinkhorn(
            ggml_tensor * comb,
            int il) const;
};

struct llm_build_dots1 : public llm_graph_context {
    llm_build_dots1(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dream : public llm_graph_context {
    llm_build_dream(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_ernie4_5 : public llm_graph_context {
    llm_build_ernie4_5(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_ernie4_5_moe : public llm_graph_context {
    llm_build_ernie4_5_moe(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_exaone4 : public llm_graph_context {
    llm_build_exaone4(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_exaone : public llm_graph_context {
    llm_build_exaone(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_exaone_moe : public llm_graph_context {
    llm_build_exaone_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_falcon : public llm_graph_context {
    llm_build_falcon(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_falcon_h1 : public llm_graph_context_mamba {
    llm_build_falcon_h1(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma2_iswa : public llm_graph_context {
    llm_build_gemma2_iswa(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_gemma3 : public llm_graph_context {
    llm_build_gemma3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma3n_iswa : public llm_graph_context {
    const llama_model & model;

    const int64_t n_embd_head;
    const int64_t n_embd_altup;
    const int64_t n_altup;
    const int     i_altup_act;
    const int     n_layer_sparsity = 10; // number of layers using activation sparsity
    const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

    llm_build_gemma3n_iswa(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * calc_magnitude(ggml_tensor * x);
    ggml_tensor * view_2d_slice(ggml_tensor * x, int idx);
    ggml_tensor * get_per_layer_inputs();
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inputs_embeds, ggml_tensor * inp_per_layer);
    ggml_tensor * gaussian_topk(ggml_tensor * x);
    ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
    ggml_tensor * altup_predict(ggml_tensor * cur, int il);
    ggml_tensor * laurel(ggml_tensor * cur, int il);
    ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
};

struct llm_build_gemma4 : public llm_graph_context {
    const llama_model & model;
    const int64_t n_embd_per_layer;

    llm_build_gemma4(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * view_2d_slice(ggml_tensor * x, int idx);
    ggml_tensor * get_per_layer_inputs();
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
};

struct llm_build_gemma4_assistant : public llm_graph_context {
    llm_build_gemma4_assistant(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_diffusion_gemma : public llm_graph_context {
    const llama_model & model;
    llm_build_diffusion_gemma(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma_embedding : public llm_graph_context {
    llm_build_gemma_embedding(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma : public llm_graph_context {
    llm_build_gemma(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_glm4 : public llm_graph_context {
    llm_build_glm4(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_glm4_moe : public llm_graph_context {
    llm_build_glm4_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gpt2 : public llm_graph_context {
    llm_build_gpt2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gptneox : public llm_graph_context {
    llm_build_gptneox(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_granite : public llm_graph_context {
    llm_build_granite(const llama_model & model, const llm_graph_params & params);

private:
    ggml_tensor * build_attention_layer(
              ggml_tensor             * cur,
              ggml_tensor             * inp_pos,
              llm_graph_input_attn_kv * inp_attn,
        const llama_model             & model,
        const int64_t                 n_embd_head,
        const int                     il);

    ggml_tensor * build_layer_ffn(
              ggml_tensor       * cur,
              ggml_tensor       * inpSA,
        const llama_model       & model,
        const int                 il);
};

struct llm_build_granite_hybrid : public llm_graph_context_mamba {
    llm_build_granite_hybrid(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_layer_ffn(ggml_tensor * cur, ggml_tensor * inpSA, const llama_model & model, const int il);
    ggml_tensor * build_attention_layer(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn,
        const llama_model & model,const int64_t n_embd_head, const int il);
};

struct llm_build_grok : public llm_graph_context {
    llm_build_grok(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_grovemoe : public llm_graph_context {
    llm_build_grovemoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_hunyuan_dense : public llm_graph_context {
    llm_build_hunyuan_dense(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_hunyuan_moe : public llm_graph_context {
    llm_build_hunyuan_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_internlm2 : public llm_graph_context {
    llm_build_internlm2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_jais : public llm_graph_context {
    llm_build_jais(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_jamba : public llm_graph_context_mamba {
    llm_build_jamba(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_kimi_linear : public llm_graph_context_mamba {
    llm_build_kimi_linear(const llama_model & model, const llm_graph_params & params);

    std::pair<ggml_tensor *, ggml_tensor *> build_kda_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * gk,
                ggml_tensor * beta,
                ggml_tensor * state,
                        int   il);

    std::pair<ggml_tensor *, ggml_tensor *> build_kda_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * gk,
                ggml_tensor * beta,
                ggml_tensor * state,
                ggml_tensor * causal_mask,
                ggml_tensor * identity,
                ggml_tensor * diag_mask,
                        int   il);

    const llama_model & model;
};

struct llm_build_kimi_k3 : public llm_build_delta_net_base {
    llm_build_kimi_k3(const llama_model & model, const llm_graph_params & params);

    const llama_model & model;
    std::vector<ggml_tensor *> resi;
    ggml_tensor * resi_stack = nullptr;
    int resi_stack_n = -1;

    void res_push(ggml_tensor * cur, int64_t n_embd, int64_t n_tokens);
    ggml_tensor * res_stack(int64_t n_embd, int64_t n_tokens);
    ggml_tensor * res_mix(ggml_tensor * cur, ggml_tensor * score_w,
                          int64_t n_embd, int64_t n_tokens, int il);
    ggml_tensor * build_kda_layer(ggml_tensor * cur, const llama_layer & layer,
                                  llm_graph_input_rs * inp_rs,
                                  int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                  int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);
    ggml_tensor * build_mla_layer(ggml_tensor * cur, const llama_layer & layer,
                                  llm_graph_input_attn_k * inp_attn_k,
                                  llm_graph_input_attn_kv * inp_attn_kv,
                                  int64_t n_embd_head_k_mla, int64_t n_embd_head_v_mla,
                                  int64_t kv_lora_rank, int64_t n_embd_head_qk_rope,
                                  int64_t n_embd_head_qk_nope, float kq_scale, int il);
    ggml_tensor * build_latent_moe(ggml_tensor * cur, const llama_layer & layer,
                                   int64_t n_embd_latent, int il);
};

class llm_graph_input_glm5_kpool;
class llama_memory_hybrid_idx_context;

struct llm_build_glm5_next : public llm_build_delta_net_base {
    llm_build_glm5_next(const llama_model & model, const llm_graph_params & params);

    const llama_model & model;

    llm_graph_input_glm5_kpool * build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb);

    ggml_tensor * build_hc_mean(ggml_tensor * x) const;
    ggml_tensor * build_hc_pre(ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
                               ggml_tensor ** post, ggml_tensor ** comb, int il) const;
    ggml_tensor * build_hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb, int il) const;

    ggml_tensor * build_kda_layer(ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
                                  int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                  int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);
    ggml_tensor * build_kpool_select(ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * kq_mask, const llama_layer & layer,
                                     const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_glm5_kpool * inp_kpool, int il);
    ggml_tensor * build_dsa_layer(ggml_tensor * cur, const llama_layer & layer,
                                  const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
                                  llm_graph_input_glm5_kpool * inp_kpool, ggml_tensor ** prev_sel, int il);
};

struct llm_build_lfm2 : public llm_graph_context {
    const llama_model & model;

    llm_build_lfm2(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_moe_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_dense_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_attn_block(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn, int il) const;
    ggml_tensor * build_shortconv_block(ggml_tensor * cur, llm_graph_input_rs * inp_recr, int il);

};

struct llm_build_llada : public llm_graph_context {
    llm_build_llada(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_llada_moe : public llm_graph_context {
    llm_build_llada_moe(const llama_model & model, const llm_graph_params & params);
};

template <bool embed>
struct llm_build_llama : public llm_graph_context {
    llm_build_llama(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_llama_iswa : public llm_graph_context {
    llm_build_llama_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_maincoder : public llm_graph_context {
    llm_build_maincoder(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mamba : public llm_graph_context_mamba {
    llm_build_mamba(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mimo2_iswa : public llm_graph_context {
    llm_build_mimo2_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_minicpm3 : public llm_graph_context {
    llm_build_minicpm3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_minimax_m2 : public llm_graph_context {
    llm_build_minimax_m2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_minimax_m3 : public llm_graph_context {
    llm_build_minimax_m3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mistral3 : public llm_graph_context {
    llm_build_mistral3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_modern_bert : public llm_graph_context {
    llm_build_modern_bert(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mpt : public llm_graph_context {
    llm_build_mpt(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_nemotron : public llm_graph_context {
    llm_build_nemotron(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_nemotron_h : public llm_graph_context_mamba {
    llm_build_nemotron_h(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_ffn_layer(ggml_tensor * cur, const llama_model & model, const int il);
    ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
        const llama_model & model, const int64_t n_embd_head, const int il);
};

struct llm_build_neo_bert : public llm_graph_context {
    llm_build_neo_bert(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_olmo2 : public llm_graph_context {
    llm_build_olmo2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_olmoe : public llm_graph_context {
    llm_build_olmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_olmo : public llm_graph_context {
    llm_build_olmo(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_openai_moe_iswa : public llm_graph_context {
    llm_build_openai_moe_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_openelm : public llm_graph_context {
    llm_build_openelm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_orion : public llm_graph_context {
    llm_build_orion(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_pangu_embedded : public llm_graph_context {
    llm_build_pangu_embedded(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_phi2 : public llm_graph_context {
    llm_build_phi2(const llama_model & model, const llm_graph_params & params);
};

template<bool iswa>
struct llm_build_phi3 : public llm_graph_context {
    llm_build_phi3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_plamo2 : public llm_graph_context_mamba {
    llm_build_plamo2(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
        ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                const llama_model & model, int il);
};

struct llm_build_plamo : public llm_graph_context {
    llm_build_plamo(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_plamo3 : public llm_graph_context {
    llm_build_plamo3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_plm : public llm_graph_context {
    llm_build_plm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen2 : public llm_graph_context {
    llm_build_qwen2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen2moe : public llm_graph_context {
    llm_build_qwen2moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen2vl : public llm_graph_context {
    llm_build_qwen2vl(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3 : public llm_graph_context {
    llm_build_qwen3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3moe : public llm_graph_context {
    llm_build_qwen3moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3vl : public llm_graph_context {
    llm_build_qwen3vl(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3vlmoe : public llm_graph_context {
    llm_build_qwen3vlmoe(const llama_model & model, const llm_graph_params & params);
};
struct llm_build_qwen3next : public llm_graph_context_mamba {
    llm_build_qwen3next(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                ggml_tensor * causal_mask,
                ggml_tensor * identity,
                ggml_tensor * diag_mask,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * beta,
                ggml_tensor * state,
                ggml_tensor * causal_mask,
                ggml_tensor * identity,
                ggml_tensor * diag_mask,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * beta,
                ggml_tensor * state,
                int           il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};

struct llm_build_qwen35 : public llm_build_delta_net_base {
    llm_build_qwen35(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};

struct llm_build_qwen35moe : public llm_build_delta_net_base {
    llm_build_qwen35moe(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};

struct llm_build_qwen : public llm_graph_context {
    llm_build_qwen(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_refact : public llm_graph_context {
    llm_build_refact(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rnd1 : public llm_graph_context {
    llm_build_rnd1(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rwkv6 : public llm_build_rwkv6_base {
    llm_build_rwkv6(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rwkv6qwen2 : public llm_build_rwkv6_base {
    llm_build_rwkv6qwen2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rwkv7 : public llm_build_rwkv7_base {
    llm_build_rwkv7(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_seed_oss : public llm_graph_context {
    llm_build_seed_oss(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_smallthinker : public llm_graph_context {
    llm_build_smallthinker(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_smollm3 : public llm_graph_context {
    llm_build_smollm3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_stablelm : public llm_graph_context {
    llm_build_stablelm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_starcoder2 : public llm_graph_context {
    llm_build_starcoder2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_starcoder : public llm_graph_context {
    llm_build_starcoder(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_step35_iswa : public llm_graph_context {
    llm_build_step35_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_t5_dec : public llm_graph_context {
    llm_build_t5_dec(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_t5_enc : public llm_graph_context {
    llm_build_t5_enc(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_wavtokenizer_dec : public llm_graph_context {
    llm_build_wavtokenizer_dec(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_xverse : public llm_graph_context {
    llm_build_xverse(const llama_model & model, const llm_graph_params & params);
};
