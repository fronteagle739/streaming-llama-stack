# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/model_from_gguf.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Dict, Optional
import torch

def gguf_to_qwen2_config(reader, dtype=torch.bfloat16):
    from transformers import Qwen2Config
    A = 'qwen2'
    n_layer = int(reader.meta(f'{A}.block_count'))
    hidden = int(reader.meta(f'{A}.embedding_length'))
    ffn = int(reader.meta(f'{A}.feed_forward_length'))
    n_head = int(reader.meta(f'{A}.attention.head_count'))
    n_kv = int(reader.meta(f'{A}.attention.head_count_kv'))
    eps = float(reader.meta(f'{A}.attention.layer_norm_rms_epsilon', 1e-06))
    rope = float(reader.meta(f'{A}.rope.freq_base', 1000000.0))
    ctx = int(reader.meta(f'{A}.context_length', 32768))
    emb = reader.info('token_embd.weight')
    vocab = int(emb.shape[1])
    hidden_chk = int(emb.shape[0])
    assert hidden_chk == hidden, (hidden_chk, hidden)
    head_dim = int(reader.info('blk.0.attn_k.weight').shape[1]) // n_kv
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=ffn, num_hidden_layers=n_layer, num_attention_heads=n_head, num_key_value_heads=n_kv, max_position_embeddings=ctx, rope_theta=rope, rms_norm_eps=eps, tie_word_embeddings=True, attention_bias=True, head_dim=head_dim, bos_token_id=int(reader.meta('tokenizer.ggml.bos_token_id', 151643)), eos_token_id=int(reader.meta('tokenizer.ggml.eos_token_id', 151645)), dtype=dtype)
    return cfg

def _gguf_name_for_hf(hf_name: str) -> Optional[Dict[str, object]]:
    if hf_name == 'model.embed_tokens.weight':
        return {'gguf': 'token_embd.weight', 'transpose': True}
    if hf_name == 'lm_head.weight':
        return {'gguf': 'token_embd.weight', 'transpose': True}
    if hf_name == 'model.norm.weight':
        return {'gguf': 'output_norm.weight', 'transpose': False}
    if hf_name.startswith('model.layers.'):
        rest = hf_name[len('model.layers.'):]
        i, _, tail = rest.partition('.')
        mp = {'self_attn.q_proj.weight': ('attn_q.weight', True), 'self_attn.q_proj.bias': ('attn_q.bias', False), 'self_attn.k_proj.weight': ('attn_k.weight', True), 'self_attn.k_proj.bias': ('attn_k.bias', False), 'self_attn.v_proj.weight': ('attn_v.weight', True), 'self_attn.v_proj.bias': ('attn_v.bias', False), 'self_attn.o_proj.weight': ('attn_output.weight', True), 'mlp.gate_proj.weight': ('ffn_gate.weight', True), 'mlp.up_proj.weight': ('ffn_up.weight', True), 'mlp.down_proj.weight': ('ffn_down.weight', True), 'input_layernorm.weight': ('attn_norm.weight', False), 'post_attention_layernorm.weight': ('ffn_norm.weight', False)}
        if tail in mp:
            g, tr = mp[tail]
            return {'gguf': f'blk.{i}.{g}', 'transpose': tr}
    return None

def build_qwen2_from_gguf(reader, device='cuda', dtype=torch.bfloat16):
    from transformers import Qwen2ForCausalLM
    from cosmos_windows.gguf_qat.dequant import dequantize
    cfg = gguf_to_qwen2_config(reader, dtype=dtype)
    with torch.device('meta'):
        model = Qwen2ForCausalLM(cfg)
    materialized = 0
    skipped = []
    state_names = dict(model.named_parameters())
    for hf_name, p in state_names.items():
        m = _gguf_name_for_hf(hf_name)
        if m is None:
            skipped.append(hf_name)
            continue
        gname = m['gguf']
        if not reader.has(gname):
            skipped.append(f'{hf_name} (no gguf {gname})')
            continue
        info = reader.info(gname)
        raw = reader.raw_bytes(gname)
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
    if hasattr(model, 'lm_head') and model.lm_head.weight.is_meta:
        model.lm_head.weight = model.model.embed_tokens.weight
    import transformers.models.qwen2.modeling_qwen2 as qm
    fresh_rotary = qm.Qwen2RotaryEmbedding(config=cfg, device=device)
    for mod in model.modules():
        if isinstance(mod, qm.Qwen2RotaryEmbedding):
            mod.register_buffer('inv_freq', fresh_rotary.inv_freq.clone().to(device), persistent=False)
            mod.original_inv_freq = mod.inv_freq
            mod.attention_scaling = fresh_rotary.attention_scaling
            mod.rope_type = fresh_rotary.rope_type
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            mod_path, _, leaf = name.rpartition('.')
            submod = model.get_submodule(mod_path) if mod_path else model
            submod.register_buffer(leaf, torch.zeros_like(buf, device=device), persistent=leaf in getattr(submod, '_persistent_buffers', {}))
    model.eval()
    return (model, cfg, materialized, skipped)
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    start_rss_guard(limit_gb=6.0, label='model_from_gguf')
    path = sys.argv[1] if len(sys.argv) > 1 else ''
    r = GGUFTensorReader(path, device='cuda')
    model, cfg, nmat, skipped = build_qwen2_from_gguf(r, device='cuda', dtype=torch.bfloat16)
    print(f"materialized {nmat} params; skipped {len(skipped)}: {skipped[:6]}{('...' if len(skipped) > 6 else '')}")
    print(f'any meta params left: {[n for n, p in model.named_parameters() if p.is_meta]}')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('')
    msgs = [{'role': 'user', 'content': 'who are you?'}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=False)
    if not torch.is_tensor(ids):
        ids = ids['input_ids'] if isinstance(ids, dict) else torch.tensor(ids)
    ids = ids.to('cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    print('GEN:', tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    print(f'RSS now {current_rss_gb():.3f} GB (peak {peak_rss_gb():.3f} GB)')
    r.close()
