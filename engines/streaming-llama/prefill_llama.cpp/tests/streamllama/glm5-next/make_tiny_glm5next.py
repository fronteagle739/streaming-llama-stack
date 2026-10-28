#!/usr/bin/env python3
# Write tiny random-weight glm5-next GGUFs for the fork-vs-upstream acceptance test.
#
#   up-shared.gguf : upstream converter layout, layer 3 reuses the layer-1 indexer selection (indexer.types)
#   up-full.gguf   : upstream converter layout, every MLA layer has its own indexer
#   ds4-full.gguf  : the same weights as up-full.gguf in the ds4 (antirez GLM-5.3-Flash) layout
#
# usage: PYTHONPATH=<llama.cpp>/gguf-py python3 make_tiny_glm5next.py <out_dir> [seed] [kda_head_dim=16]
import sys
import os
import numpy as np
import gguf

out_dir = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
os.makedirs(out_dir, exist_ok=True)

ARCH = "glm5-next"
n_vocab, n_embd, n_head = 256, 64, 4
head_dim_kda, d_conv = int(sys.argv[3]) if len(sys.argv)>3 else 16, 4
if head_dim_kda not in (16,32,64,128):
    raise ValueError('KDA head dimension must be 16, 32, 64 or 128')
d_inner = n_head * head_dim_kda
q_lora, kv_lora, qk_nope, v_head = 32, 32, 16, 16
n_ihead, ihead, kpool, top_k = 4, 16, 4, 8
hc, hc_iters, hc_eps = 4, 20, 1e-6
n_ff, n_ff_exp, n_expert, n_used, n_shared = 96, 32, 8, 2, 1
# trunk: 0 KDA (dense ffn), 1 MLA, 2 KDA, 3 MLA, 4 KDA, 5 MLA; layer 6 is the NextN (MTP) block
n_trunk = 6
is_mla = [0, 1, 0, 1, 0, 1, 1]
n_layer = len(is_mla)
swiglu_limit = 10.0

rng = np.random.default_rng(seed)


def rnd(*shape, std=0.1):
    return (rng.standard_normal(shape) * std).astype(np.float32)


def norm_w(n):
    return (1.0 + rnd(n, std=0.05)).astype(np.float32)


# numpy shapes are reversed gguf ne
T = {}
A_LOG = {}
T["token_embd.weight"] = rnd(n_vocab, n_embd, std=0.5)
T["output_norm.weight"] = norm_w(n_embd)
T["output.weight"] = rnd(n_vocab, n_embd, std=0.2)
for i in range(n_layer):
    b = f"blk.{i}."
    T[b + "attn_norm.weight"] = norm_w(n_embd)
    T[b + "ffn_norm.weight"] = norm_w(n_embd)
    if i < n_trunk:
        for k in ("attn", "ffn"):
            T[b + f"hc_{k}_fn.weight"] = rnd((2 + hc) * hc, hc * n_embd, std=0.05)
            T[b + f"hc_{k}_base.weight"] = rnd((2 + hc) * hc, std=0.2)
            T[b + f"hc_{k}_scale.weight"] = (0.5 + rng.random(3)).astype(np.float32)
    if not is_mla[i]:
        for q in ("q", "k", "v"):
            T[b + f"attn_{q}.weight"] = rnd(d_inner, n_embd, std=0.15)
            T[b + f"ssm_conv1d_{q}.weight"] = rnd(1, d_inner, 1, d_conv, std=0.3)
        T[b + "ssm_f_a.weight"] = rnd(head_dim_kda, n_embd, std=0.15)
        T[b + "ssm_f_b.weight"] = rnd(d_inner, head_dim_kda, std=0.15)
        T[b + "ssm_beta.weight"] = rnd(n_head, n_embd, std=0.15)
        # upstream stores -exp(A_log), ds4 stores A_log: keep A_log in float32 so both layouts carry the same value
        A_LOG[i] = np.log(rng.uniform(1.0, 16.0, n_head)).astype(np.float32)
        T[b + "ssm_a"] = -np.exp(A_LOG[i])
        T[b + "ssm_dt.bias"] = rnd(d_inner, std=0.5)
        T[b + "ssm_g_a.weight"] = rnd(head_dim_kda, n_embd, std=0.15)
        T[b + "ssm_g_b.weight"] = rnd(d_inner, head_dim_kda, std=0.15)
        T[b + "ssm_norm.weight"] = norm_w(head_dim_kda)
        T[b + "attn_output.weight"] = rnd(n_embd, d_inner, std=0.15)
    else:
        T[b + "attn_q_a_norm.weight"] = norm_w(q_lora)
        T[b + "attn_kv_a_norm.weight"] = norm_w(kv_lora)
        T[b + "attn_q_a.weight"] = rnd(q_lora, n_embd, std=0.15)
        T[b + "attn_q_b.weight"] = rnd(n_head * qk_nope, q_lora, std=0.2)
        T[b + "attn_kv_a_mqa.weight"] = rnd(kv_lora, n_embd, std=0.15)
        T[b + "attn_k_b.weight"] = rnd(n_head, kv_lora, qk_nope, std=0.2)
        T[b + "attn_v_b.weight"] = rnd(n_head, v_head, kv_lora, std=0.2)
        T[b + "attn_output.weight"] = rnd(n_embd, n_head * v_head, std=0.15)
        T[b + "indexer.k_norm.weight"] = norm_w(ihead)
        T[b + "indexer.k_norm.bias"] = rnd(ihead, std=0.05)
        T[b + "indexer.proj.weight"] = rnd(n_ihead, n_embd, std=0.3)
        T[b + "indexer.attn_k.weight"] = rnd(ihead, n_embd, std=0.3)
        T[b + "indexer.attn_q_b.weight"] = rnd(n_ihead * ihead, q_lora, std=0.3)
        T[b + "indexer_compressor_gate.weight"] = rnd(ihead, n_embd, std=0.3)
        T[b + "indexer_compressor_ape.weight"] = rnd(kpool, ihead, std=0.3)
    if i == 0:
        T[b + "ffn_gate.weight"] = rnd(n_ff, n_embd, std=0.15)
        T[b + "ffn_up.weight"] = rnd(n_ff, n_embd, std=0.15)
        T[b + "ffn_down.weight"] = rnd(n_embd, n_ff, std=0.15)
    else:
        T[b + "ffn_gate_inp.weight"] = rnd(n_expert, n_embd, std=0.3)
        T[b + "exp_probs_b.bias"] = rnd(n_expert, std=0.1)
        T[b + "ffn_gate_exps.weight"] = rnd(n_expert, n_ff_exp, n_embd, std=0.15)
        T[b + "ffn_up_exps.weight"] = rnd(n_expert, n_ff_exp, n_embd, std=0.15)
        T[b + "ffn_down_exps.weight"] = rnd(n_expert, n_embd, n_ff_exp, std=0.15)
        T[b + "ffn_gate_shexp.weight"] = rnd(n_ff_exp * n_shared, n_embd, std=0.15)
        T[b + "ffn_up_shexp.weight"] = rnd(n_ff_exp * n_shared, n_embd, std=0.15)
        T[b + "ffn_down_shexp.weight"] = rnd(n_embd, n_ff_exp * n_shared, std=0.15)
    if i >= n_trunk:
        T[b + "nextn.eh_proj.weight"] = rnd(n_embd, 2 * n_embd, std=0.1)
        T[b + "nextn.enorm.weight"] = norm_w(n_embd)
        T[b + "nextn.hnorm.weight"] = norm_w(n_embd)
        T[b + "nextn.shared_head_norm.weight"] = norm_w(n_embd)


def common_kv(w):
    p = ARCH + "."
    w.add_uint32(p + "context_length", 4096)
    w.add_uint32(p + "embedding_length", n_embd)
    w.add_uint32(p + "block_count", n_layer)
    w.add_uint32(p + "nextn_predict_layers", n_layer - n_trunk)
    w.add_uint32(p + "vocab_size", n_vocab)
    w.add_uint32(p + "feed_forward_length", n_ff)
    w.add_uint32(p + "expert_feed_forward_length", n_ff_exp)
    w.add_uint32(p + "expert_count", n_expert)
    w.add_uint32(p + "expert_used_count", n_used)
    w.add_uint32(p + "expert_shared_count", n_shared)
    w.add_uint32(p + "leading_dense_block_count", 1)
    w.add_float32(p + "expert_weights_scale", 2.5)
    w.add_bool(p + "expert_weights_norm", True)
    w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32(p + "attention.head_count", n_head)
    w.add_uint32(p + "attention.q_lora_rank", q_lora)
    w.add_uint32(p + "attention.kv_lora_rank", kv_lora)
    w.add_uint32(p + "attention.indexer.head_count", n_ihead)
    w.add_uint32(p + "attention.indexer.key_length", ihead)
    w.add_uint32(p + "attention.indexer.top_k", top_k)
    w.add_uint32(p + "hyper_connection.count", hc)
    w.add_uint32(p + "hyper_connection.sinkhorn_iterations", hc_iters)
    w.add_float32(p + "hyper_connection.epsilon", hc_eps)
    w.add_string("tokenizer.ggml.model", "none")


def write_up(path, shared):
    w = gguf.GGUFWriter(path, ARCH)
    common_kv(w)
    p = ARCH + "."
    w.add_array(p + "attention.head_count_kv", [1 if m else 0 for m in is_mla])
    w.add_uint32(p + "attention.key_length", kv_lora)
    w.add_uint32(p + "attention.value_length", kv_lora)
    w.add_uint32(p + "attention.key_length_mla", qk_nope)
    w.add_uint32(p + "attention.value_length_mla", v_head)
    w.add_uint32(p + "rope.dimension_count", 0)
    w.add_uint32(p + "ssm.conv_kernel", d_conv)
    w.add_uint32(p + "kda.head_dim", head_dim_kda)
    w.add_float32(p + "kda.gate_lower_bound", -5.0)
    w.add_uint32(p + "attention.indexer.kpool", kpool)
    w.add_bool(p + "attention.indexer.kpool_select_tail", True)
    w.add_uint32(p + "expert_gating_func", 2)
    w.add_array(p + "swiglu_clamp_exp", [swiglu_limit] * n_layer)
    w.add_array(p + "swiglu_clamp_shexp", [swiglu_limit] * n_layer)
    if shared:
        w.add_array(p + "attention.indexer.types", [i != 3 for i in range(n_trunk)])
    for name, t in T.items():
        if shared and name.startswith("blk.3.indexer"):
            continue
        w.add_tensor(name, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


DS4_NAMES = {
    "attn_q.weight": "kda_q.weight", "attn_k.weight": "kda_k.weight", "attn_v.weight": "kda_v.weight",
    "ssm_conv1d_q.weight": "kda_q_conv.weight", "ssm_conv1d_k.weight": "kda_k_conv.weight",
    "ssm_conv1d_v.weight": "kda_v_conv.weight", "ssm_f_a.weight": "kda_f_a.weight", "ssm_f_b.weight": "kda_f_b.weight",
    "ssm_beta.weight": "kda_beta.weight", "ssm_g_a.weight": "kda_g_a.weight", "ssm_g_b.weight": "kda_g_b.weight",
    "ssm_norm.weight": "kda_o_norm.weight", "ssm_dt.bias": "kda_dt_bias.weight", "ssm_a": "kda_a_log.weight",
    "indexer_compressor_ape.weight": "indexer.pool_ape.weight",
    "indexer_compressor_gate.weight": "indexer.pool_gate.weight",
}


def write_ds4(path):
    w = gguf.GGUFWriter(path, ARCH)
    common_kv(w)
    p = ARCH + "."
    w.add_uint32(p + "trunk_block_count", n_trunk)
    w.add_array(p + "layer_types", is_mla)
    w.add_uint32(p + "attention.key_length", qk_nope)
    w.add_uint32(p + "attention.value_length", v_head)
    w.add_uint32(p + "attention.rope_dimension_count", 0)
    w.add_uint32(p + "linear_attention.head_count", n_head)
    w.add_uint32(p + "linear_attention.head_dimension", head_dim_kda)
    w.add_uint32(p + "linear_attention.conv_kernel", d_conv)
    w.add_float32(p + "linear_attention.gate_lower_bound", -5.0)
    w.add_uint32(p + "attention.indexer.pool_size", kpool)
    w.add_float32(p + "swiglu_limit", swiglu_limit)
    for name, t in T.items():
        parts = name.split(".", 2)
        if parts[0] == "blk":
            i = int(parts[1])
            suffix = parts[2]
            if not is_mla[i] and suffix == "attn_output.weight":
                suffix = "kda_output.weight"
            elif suffix in DS4_NAMES:
                suffix = DS4_NAMES[suffix]
            if suffix == "kda_a_log.weight":
                t = A_LOG[i]
            if suffix.endswith("_conv.weight"):
                t = t.reshape(d_inner, 1, d_conv)
            name = f"blk.{i}.{suffix}"
        w.add_tensor(name, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


write_up(os.path.join(out_dir, "up-shared.gguf"), shared=True)
write_up(os.path.join(out_dir, "up-full.gguf"), shared=False)
write_ds4(os.path.join(out_dir, "ds4-full.gguf"))
print("wrote", out_dir)
