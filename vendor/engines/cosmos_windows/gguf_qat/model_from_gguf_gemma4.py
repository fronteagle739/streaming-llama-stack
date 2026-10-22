# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/model_from_gguf_gemma4.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch

def gguf_to_gemma4_text_config(reader, dtype=torch.bfloat16):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    A = 'gemma4'
    n_layer = int(reader.meta(f'{A}.block_count'))
    hidden = int(reader.meta(f'{A}.embedding_length'))
    ffn = int(reader.meta(f'{A}.feed_forward_length'))
    n_head = int(reader.meta(f'{A}.attention.head_count'))
    eps = float(reader.meta(f'{A}.attention.layer_norm_rms_epsilon', 1e-06))
    hckv = list(reader.meta(f'{A}.attention.head_count_kv'))
    swp = list(reader.meta(f'{A}.attention.sliding_window_pattern'))
    assert len(hckv) == n_layer and len(swp) == n_layer, (len(hckv), len(swp), n_layer)
    layer_types = ['sliding_attention' if bool(s) else 'full_attention' for s in swp]
    sliding_kv = [hckv[i] for i in range(n_layer) if layer_types[i] == 'sliding_attention']
    full_kv = [hckv[i] for i in range(n_layer) if layer_types[i] == 'full_attention']
    num_kv = sliding_kv[0] if sliding_kv else hckv[0]
    num_global_kv = full_kv[0] if full_kv else None
    assert all((k == num_kv for k in sliding_kv)), ('non-uniform sliding kv', sliding_kv)
    if full_kv:
        assert all((k == num_global_kv for k in full_kv)), ('non-uniform full kv', full_kv)
    head_dim = int(reader.meta(f'{A}.attention.key_length_swa', 256))
    global_head_dim = int(reader.meta(f'{A}.attention.key_length', 512))
    k_eq_v = False
    for i in range(n_layer):
        if layer_types[i] == 'full_attention' and (not reader.has(f'blk.{i}.attn_v.weight')):
            k_eq_v = True
            break
    sliding_window = int(reader.meta(f'{A}.attention.sliding_window', 1024))
    rope_theta = float(reader.meta(f'{A}.rope.freq_base', 1000000.0))
    rope_theta_swa = float(reader.meta(f'{A}.rope.freq_base_swa', 10000.0))
    rope_dim_full = int(reader.meta(f'{A}.rope.dimension_count', global_head_dim))
    partial_full = float(rope_dim_full) / float(global_head_dim) if global_head_dim else 1.0
    softcap = reader.meta(f'{A}.final_logit_softcapping', None)
    ctx = int(reader.meta(f'{A}.context_length', 262144))
    emb = reader.info('token_embd.weight')
    vocab = int(emb.shape[1])
    hidden_chk = int(emb.shape[0])
    assert hidden_chk == hidden, (hidden_chk, hidden)
    n_experts = int(reader.meta(f'{A}.expert_count'))
    n_used = int(reader.meta(f'{A}.expert_used_count'))
    moe_inter = int(reader.meta(f'{A}.expert_feed_forward_length'))
    ple = int(reader.meta(f'{A}.embedding_length_per_layer_input', 0))
    rope_parameters = {'sliding_attention': {'rope_type': 'default', 'rope_theta': rope_theta_swa}, 'full_attention': {'rope_type': 'proportional', 'partial_rotary_factor': partial_full, 'rope_theta': rope_theta}}
    cfg = Gemma4TextConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=ffn, num_hidden_layers=n_layer, num_attention_heads=n_head, num_key_value_heads=num_kv, num_global_key_value_heads=num_global_kv, head_dim=head_dim, global_head_dim=global_head_dim, attention_k_eq_v=k_eq_v, sliding_window=sliding_window, layer_types=layer_types, max_position_embeddings=ctx, rms_norm_eps=eps, rope_parameters=rope_parameters, attention_bias=False, tie_word_embeddings=True, final_logit_softcapping=float(softcap) if softcap is not None else None, hidden_size_per_layer_input=ple, enable_moe_block=True, num_experts=n_experts, top_k_experts=n_used, moe_intermediate_size=moe_inter, use_double_wide_mlp=False, num_kv_shared_layers=0, pad_token_id=int(reader.meta('tokenizer.ggml.padding_token_id', 0)), bos_token_id=int(reader.meta('tokenizer.ggml.bos_token_id', 2)), eos_token_id=int(reader.meta('tokenizer.ggml.eos_token_id', 1)), dtype=dtype)
    return cfg

class QuantizedGemma4Experts(torch.nn.Module):

    def __init__(self, num_experts: int, hidden: int, intermediate: int, gate_up_raw: torch.Tensor, down_raw: torch.Tensor, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.act_fn = act_fn
        self.register_buffer('gate_up_raw', gate_up_raw, persistent=False)
        self.register_buffer('down_raw', down_raw, persistent=False)
        self._gu_blocks = gate_up_raw.shape[1] // 18
        self._dn_blocks = down_raw.shape[1] // 18

    def _dequant_expert(self, raw_e: torch.Tensor, out_rows: int, in_cols: int, dtype: torch.dtype) -> torch.Tensor:
        from cosmos_windows.gguf_qat.dequant import dequantize_q4_0
        flat = dequantize_q4_0(raw_e.reshape(-1)).to(dtype)
        return flat.reshape(out_rows, in_cols)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        dt = hidden_states.dtype
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gu = self._dequant_expert(self.gate_up_raw[expert_idx], 2 * self.intermediate_dim, self.hidden_dim, dt)
            dn = self._dequant_expert(self.down_raw[expert_idx], self.hidden_dim, self.intermediate_dim, dt)
            gate, up = torch.nn.functional.linear(current_state, gu).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = torch.nn.functional.linear(cur, dn)
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, cur.to(final_hidden_states.dtype))
            del gu, dn
        return final_hidden_states

def _gguf_name_for_hf(hf_name: str) -> Optional[Dict[str, object]]:
    if hf_name == 'embed_tokens.weight':
        return {'gguf': 'token_embd.weight', 'transpose': True}
    if hf_name == 'norm.weight':
        return {'gguf': 'output_norm.weight', 'transpose': False}
    if hf_name.startswith('layers.'):
        rest = hf_name[len('layers.'):]
        i, _, tail = rest.partition('.')
        mp = {'self_attn.q_proj.weight': ('attn_q.weight', True), 'self_attn.k_proj.weight': ('attn_k.weight', True), 'self_attn.v_proj.weight': ('attn_v.weight', True), 'self_attn.o_proj.weight': ('attn_output.weight', True), 'self_attn.q_norm.weight': ('attn_q_norm.weight', False), 'self_attn.k_norm.weight': ('attn_k_norm.weight', False), 'mlp.gate_proj.weight': ('ffn_gate.weight', True), 'mlp.up_proj.weight': ('ffn_up.weight', True), 'mlp.down_proj.weight': ('ffn_down.weight', True), 'input_layernorm.weight': ('attn_norm.weight', False), 'post_attention_layernorm.weight': ('post_attention_norm.weight', False), 'pre_feedforward_layernorm.weight': ('ffn_norm.weight', False), 'post_feedforward_layernorm.weight': ('post_ffw_norm.weight', False), 'pre_feedforward_layernorm_2.weight': ('pre_ffw_norm_2.weight', False), 'post_feedforward_layernorm_1.weight': ('post_ffw_norm_1.weight', False), 'post_feedforward_layernorm_2.weight': ('post_ffw_norm_2.weight', False), 'router.proj.weight': ('ffn_gate_inp.weight', True), 'router.scale': ('ffn_gate_inp.scale', False), 'router.per_expert_scale': ('ffn_down_exps.scale', False)}
        if tail in mp:
            g, tr = mp[tail]
            return {'gguf': f'blk.{i}.{g}', 'transpose': tr}
    return None

def _materialize_text_model(model, reader, cfg, device, dtype, prefix=''):
    from transformers.models.gemma4.modeling_gemma4 import ACT2FN
    from cosmos_windows.gguf_qat.dequant import dequantize
    materialized = 0
    skipped: List[str] = []
    for hf_name, p in model.named_parameters():
        m = _gguf_name_for_hf(hf_name)
        if m is None:
            skipped.append(hf_name)
            continue
        gname = m['gguf']
        if not reader.has(gname):
            skipped.append(f'{hf_name} (no gguf {gname})')
            continue
        info = reader.info(gname)
        raw = reader.raw_bytes_pread(gname)
        flat = dequantize(raw, info.quant, info.shape).reshape(-1)
        if len(info.shape) == 2 and m['transpose']:
            w = flat.reshape(int(info.shape[1]), int(info.shape[0]))
        else:
            w = flat.reshape(tuple((int(s) for s in p.shape)))
        w = w.to(dtype)
        if tuple(w.shape) != tuple(p.shape):
            raise ValueError(f"shape mismatch {hf_name}: got {tuple(w.shape)} want {tuple(p.shape)} (gguf {gname} {info.shape} transpose={m['transpose']})")
        mod_path, _, leaf = hf_name.rpartition('.')
        submod = model.get_submodule(mod_path)
        setattr(submod, leaf, torch.nn.Parameter(w.to(device), requires_grad=False))
        materialized += 1
        del raw, flat, w
    act_fn = ACT2FN[cfg.hidden_activation]
    for i, layer in enumerate(model.layers):
        gu = reader.raw_bytes_pread(f'blk.{i}.ffn_gate_up_exps.weight')
        dn = reader.raw_bytes_pread(f'blk.{i}.ffn_down_exps.weight')
        E = cfg.num_experts
        gu_per = gu.numel() // E
        dn_per = dn.numel() // E
        gu_re = gu.reshape(E, gu_per).contiguous()
        dn_re = dn.reshape(E, dn_per).contiguous()
        qexp = QuantizedGemma4Experts(num_experts=E, hidden=cfg.hidden_size, intermediate=cfg.moe_intermediate_size, gate_up_raw=gu_re, down_raw=dn_re, act_fn=act_fn)
        layer.experts = qexp
        del gu, dn
    for i, layer in enumerate(model.layers):
        gname = f'blk.{i}.layer_output_scale.weight'
        if reader.has(gname):
            info = reader.info(gname)
            raw = reader.raw_bytes_pread(gname)
            v = dequantize(raw, info.quant, info.shape).reshape(-1).to(dtype).to(device)
            layer.register_buffer('layer_scalar', v, persistent=False)
            del raw
    import transformers.models.gemma4.modeling_gemma4 as gm
    fresh_rotary = gm.Gemma4TextRotaryEmbedding(config=cfg, device=device)
    if isinstance(model.rotary_emb, gm.Gemma4TextRotaryEmbedding):
        for lt in fresh_rotary.layer_types:
            for suf in ('inv_freq', 'original_inv_freq'):
                bn = f'{lt}_{suf}'
                if hasattr(fresh_rotary, bn):
                    model.rotary_emb.register_buffer(bn, getattr(fresh_rotary, bn).clone().to(device), persistent=False)
            sn = f'{lt}_attention_scaling'
            if hasattr(fresh_rotary, sn):
                setattr(model.rotary_emb, sn, getattr(fresh_rotary, sn))
        model.rotary_emb.rope_init_fns = fresh_rotary.rope_init_fns
        model.rotary_emb.rope_type = fresh_rotary.rope_type
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            mod_path, _, leaf = name.rpartition('.')
            submod = model.get_submodule(mod_path) if mod_path else model
            if leaf == 'embed_scale':
                val = torch.tensor(cfg.hidden_size ** 0.5, device=device)
            elif leaf == 'layer_scalar':
                val = torch.ones_like(buf, device=device)
            else:
                val = torch.zeros_like(buf, device=device)
            submod.register_buffer(leaf, val, persistent=False)
    return (materialized, skipped)

def build_gemma4_from_gguf(reader, device='cuda', dtype=torch.bfloat16, verbose=True):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    cfg = gguf_to_gemma4_text_config(reader, dtype=dtype)
    if verbose:
        print(f'[gemma4] cfg: L={cfg.num_hidden_layers} h={cfg.hidden_size} heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads}/global_kv={cfg.num_global_key_value_heads} hd={cfg.head_dim}/global_hd={cfg.global_head_dim} k_eq_v={cfg.attention_k_eq_v} experts={cfg.num_experts}/top{cfg.top_k_experts} moe_inter={cfg.moe_intermediate_size} softcap={cfg.final_logit_softcapping} ple={cfg.hidden_size_per_layer_input}', flush=True)
    with torch.device('meta'):
        model = Gemma4TextModel(cfg)
    materialized, skipped = _materialize_text_model(model, reader, cfg, device, dtype)
    leftover_meta = [n for n, p in model.named_parameters() if p.is_meta]
    model.eval()
    return (model, cfg, materialized, skipped, leftover_meta)

def build_gemma4_causal_lm_from_gguf(reader, device='cuda', dtype=torch.bfloat16, verbose=True):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = gguf_to_gemma4_text_config(reader, dtype=dtype)
    if verbose:
        print(f'[gemma4] cfg: L={cfg.num_hidden_layers} h={cfg.hidden_size} heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads}/global_kv={cfg.num_global_key_value_heads} hd={cfg.head_dim}/global_hd={cfg.global_head_dim} k_eq_v={cfg.attention_k_eq_v} experts={cfg.num_experts}/top{cfg.top_k_experts} softcap={cfg.final_logit_softcapping}', flush=True)
    with torch.device('meta'):
        model = Gemma4ForCausalLM(cfg)
    materialized, skipped = _materialize_text_model(model.model, reader, cfg, device, dtype, prefix='model.')
    model.lm_head.weight = model.model.embed_tokens.weight
    leftover_meta = [n for n, p in model.named_parameters() if p.is_meta]
    model.eval()
    return (model, cfg, materialized, skipped, leftover_meta)
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '12.0')), label='model_from_gguf_gemma4')
    path = sys.argv[1] if len(sys.argv) > 1 else ''
    r = GGUFTensorReader(path, device='cuda')
    model, cfg, nmat, skipped, leftover = build_gemma4_from_gguf(r, device='cuda', dtype=torch.bfloat16)
    print(f'materialized {nmat} non-expert params; skipped {len(skipped)}; leftover-meta {len(leftover)}')
    print('  skipped sample:', skipped[:8])
    if leftover:
        print('  LEFTOVER META PARAMS:', leftover[:12])
    r.close()
    print(f'VRAM allocated: {torch.cuda.memory_allocated() / 1000000000.0:.2f} GB, reserved {torch.cuda.memory_reserved() / 1000000000.0:.2f} GB')
    from transformers import AutoTokenizer
    tok_dir = sys.argv[2] if len(sys.argv) > 2 else None
    if tok_dir:
        tok = AutoTokenizer.from_pretrained(tok_dir)
        msgs = [{'role': 'user', 'content': 'who are you?'}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt')
        if isinstance(ids, dict):
            ids = ids['input_ids']
        ids = ids.to('cuda')
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        with torch.no_grad():
            out = model(input_ids=ids)
            print('logits shape:', tuple(out.last_hidden_state.shape))
    print(f'RSS now {current_rss_gb():.3f} GB (peak {peak_rss_gb():.3f} GB)')
