# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/streamed_model_deepseek2.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.streamed_forward import StreamedLoRA, rms_norm
from cosmos_windows.gguf_qat.streamed_model import _ds_sweep, _ds_layer

class Deepseek2Config:

    def __init__(self, g):
        A = g.arch()
        if A not in {'deepseek2', 'glm-dsa'}:
            raise ValueError(f'DeepSeek/GLM MLA trainer does not support architecture {A!r}')
        self.total_n_layer = int(g.meta(f'{A}.block_count'))
        self.arch = A
        self.nextn_layers = int(g.meta(f'{A}.nextn_predict_layers', 0) or 0)
        if self.nextn_layers < 0 or self.nextn_layers >= self.total_n_layer:
            raise ValueError('invalid nextn_predict_layers in GGUF metadata')
        self.n_layer = self.total_n_layer - self.nextn_layers
        self.hidden = int(g.meta(f'{A}.embedding_length'))
        self.n_head = int(g.meta(f'{A}.attention.head_count'))
        self.q_lora = int(g.meta(f'{A}.attention.q_lora_rank'))
        self.kv_lora = int(g.meta(f'{A}.attention.kv_lora_rank'))
        self.qk_head = int(g.meta(f'{A}.attention.key_length_mla', 192))
        self.v_head = int(g.meta(f'{A}.attention.value_length_mla', 128))
        self.qk_rope = int(g.meta(f'{A}.rope.dimension_count', 64))
        self.qk_nope = self.qk_head - self.qk_rope
        self.eps = float(g.meta(f'{A}.attention.layer_norm_rms_epsilon', 1e-06))
        self.rope_base = float(g.meta(f'{A}.rope.freq_base', 50000.0))
        self.yarn_factor = float(g.meta(f'{A}.rope.scaling.factor', 1.0))
        self.yarn_orig_ctx = int(g.meta(f'{A}.rope.scaling.original_context_length', 4096))
        self.yarn_beta_fast = float(g.meta(f'{A}.rope.scaling.yarn_beta_fast', 32.0))
        self.yarn_beta_slow = float(g.meta(f'{A}.rope.scaling.yarn_beta_slow', 1.0))
        self.yarn_log_mul = float(g.meta(f'{A}.rope.scaling.yarn_log_multiplier', 0.0))
        self.n_dense = int(g.meta(f'{A}.leading_dense_block_count', 1))
        self.n_expert = int(g.meta(f'{A}.expert_count', 0))
        self.n_expert_used = int(g.meta(f'{A}.expert_used_count', 0))
        self.expert_w_norm = bool(g.meta(f'{A}.expert_weights_norm', True))
        self.expert_w_scale = float(g.meta(f'{A}.expert_weights_scale', 1.0))
        self.expert_gating = int(g.meta(f'{A}.expert_gating_func', 2))
        emb = g.info('token_embd.weight')
        self.vocab = int(emb.shape[1])
        self.eos_id = int(g.meta('tokenizer.ggml.eos_token_id', 0))
        freq_scale = 1.0 / self.yarn_factor
        attn_factor = 1.0
        attn_factor_org = attn_factor
        self.mscale = attn_factor_org * (1.0 + 0.1 * self.yarn_log_mul * math.log(1.0 / freq_scale))
        self.kq_scale = self.mscale * self.mscale / math.sqrt(self.qk_head)
        self.freq_scale = freq_scale
        self.rope_interleave = A == 'glm-dsa'
        if A == 'glm-dsa' and hasattr(g, 'has') and g.has('blk.0.indexer.attn_q_b.weight'):
            self.index_head = int(g.info('blk.0.indexer.attn_k.weight').shape[1])
            self.index_heads = int(g.info('blk.0.indexer.attn_q_b.weight').shape[1]) // self.index_head
        else:
            self.index_head = 0
            self.index_heads = 0

def yarn_rope_cache(cfg: Deepseek2Config, seqlen, device, dtype=torch.float32):
    dim = cfg.qk_rope
    base = cfg.rope_base
    half = dim // 2
    freqs = base ** (-torch.arange(0, half, device=device, dtype=torch.float64) / half * 2.0 / dim * dim)
    inv_freq = 1.0 / base ** (torch.arange(0, half, device=device, dtype=torch.float64) * 2.0 / dim)

    def correction_dim(num_rot):
        return dim * math.log(cfg.yarn_orig_ctx / (num_rot * 2 * math.pi)) / (2 * math.log(base))
    low = math.floor(correction_dim(cfg.yarn_beta_fast))
    high = math.ceil(correction_dim(cfg.yarn_beta_slow))
    low = max(low, 0)
    high = min(high, half - 1)
    if low == high:
        high += 0.001
    ramp = (torch.arange(half, device=device, dtype=torch.float64) - low) / (high - low)
    ramp = torch.clamp(ramp, 0.0, 1.0)
    inv_freq_mask = 1.0 - ramp
    inv_freq_interp = inv_freq / cfg.yarn_factor
    inv_freq = inv_freq_interp * (1 - inv_freq_mask) + inv_freq * inv_freq_mask
    t = torch.arange(seqlen, device=device, dtype=torch.float64)
    f = torch.outer(t, inv_freq)
    emb = torch.cat([f, f], dim=-1)
    return (emb.cos().to(dtype), emb.sin().to(dtype))

def apply_rope_neox(x, cos, sin):
    n_rot = cos.shape[-1]
    half = n_rot // 2
    x1, x2 = (x[..., :half], x[..., half:])
    rot = torch.cat([-x2, x1], dim=-1)
    c = cos.view(*[1] * (x.dim() - 2), cos.shape[0], n_rot)
    s = sin.view(*[1] * (x.dim() - 2), sin.shape[0], n_rot)
    return x * c + rot * s

def apply_rope_interleaved(x, cos, sin):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos_half = cos[..., :cos.shape[-1] // 2]
    sin_half = sin[..., :sin.shape[-1] // 2]
    shape = (*[1] * (x.dim() - 2), cos.shape[0], cos_half.shape[-1])
    c = cos_half.view(shape)
    s = sin_half.view(shape)
    out = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1)
    return out.flatten(-2)

def apply_main_rope(cfg, x, cos, sin):
    return apply_rope_interleaved(x, cos, sin) if cfg.rope_interleave else apply_rope_neox(x, cos, sin)

def deepseek2_layer_forward(hidden, il, g, cfg: Deepseek2Config, lora, rope, mask, weight_overrides=None, sidecar_bank=None, stop_after_attention=False):
    dev = hidden.device
    B, T, H = hidden.shape
    eps = cfg.eps

    def dq(name):
        if weight_overrides is not None and name in weight_overrides:
            return weight_overrides[name].to(torch.bfloat16)
        return g.dequant_2d_hf(name, dtype=torch.bfloat16).detach()

    def dq1(name):
        if weight_overrides is not None and name in weight_overrides:
            return weight_overrides[name].to(torch.float32).reshape(-1)
        return g.dequant(name, dtype=torch.float32).detach().reshape(-1)
    x = rms_norm(hidden, dq1(f'blk.{il}.attn_norm.weight'), eps)
    Wq_a = dq(f'blk.{il}.attn_q_a.weight')
    q = F.linear(x, Wq_a)
    if lora is not None and lora.has(il):
        d = lora.delta(il, 'q_a', x.dtype)
        if d is not None:
            q = q + F.linear(x, d)
    q = rms_norm(q, dq1(f'blk.{il}.attn_q_a_norm.weight'), eps)
    q_in = q
    Wq_b = dq(f'blk.{il}.attn_q_b.weight')
    q = F.linear(q_in, Wq_b)
    if lora is not None and lora.has(il):
        d = lora.delta(il, 'q_b', q_in.dtype)
        if d is not None:
            q = q + F.linear(q_in, d)
    q = q.view(B, T, cfg.n_head, cfg.qk_head)
    q_nope = q[..., :cfg.qk_nope]
    q_pe = q[..., cfg.qk_nope:]
    Wkv_a = dq(f'blk.{il}.attn_kv_a_mqa.weight')
    c = F.linear(x, Wkv_a)
    if lora is not None and lora.has(il):
        d = lora.delta(il, 'kv_a', x.dtype)
        if d is not None:
            c = c + F.linear(x, d)
    kv_cmpr = c[..., :cfg.kv_lora]
    k_pe = c[..., cfg.kv_lora:]
    kv_cmpr = rms_norm(kv_cmpr, dq1(f'blk.{il}.attn_kv_a_norm.weight'), eps)
    k_b_name = f'blk.{il}.attn_k_b.weight'
    v_b_name = f'blk.{il}.attn_v_b.weight'
    ik = g.info(k_b_name)
    iv = g.info(v_b_name)
    from cosmos_windows.gguf_qat.dequant import dequantize
    if weight_overrides is not None and k_b_name in weight_overrides:
        kb = weight_overrides[k_b_name]
    else:
        with torch.no_grad():
            kb = dequantize(g.raw_bytes_pread(k_b_name), ik.quant, ik.shape)
    if weight_overrides is not None and v_b_name in weight_overrides:
        vb = weight_overrides[v_b_name]
    else:
        with torch.no_grad():
            vb = dequantize(g.raw_bytes_pread(v_b_name), iv.quant, iv.shape)
    kb = kb.reshape(cfg.n_head, cfg.kv_lora, cfg.qk_nope).to(torch.bfloat16)
    vb = vb.reshape(cfg.n_head, cfg.v_head, cfg.kv_lora).to(torch.bfloat16)
    kvc = kv_cmpr.to(torch.bfloat16)
    k_nope = torch.einsum('btc,hcd->bthd', kvc, kb)
    v = torch.einsum('btc,hdc->bthd', kvc, vb)
    cos, sin = rope
    q_pe = apply_main_rope(cfg, q_pe.transpose(1, 2), cos, sin).transpose(1, 2)
    k_pe_r = apply_main_rope(cfg, k_pe.unsqueeze(2).transpose(1, 2), cos, sin).transpose(1, 2)
    k_pe_r = k_pe_r.expand(B, T, cfg.n_head, cfg.qk_rope)
    Q = torch.cat([q_nope, q_pe], dim=-1).to(torch.float32)
    K = torch.cat([k_nope, k_pe_r], dim=-1).to(torch.float32)
    Vv = v.to(torch.float32)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    Vv = Vv.transpose(1, 2)
    attn = F.scaled_dot_product_attention(Q, K, Vv, attn_mask=mask, scale=cfg.kq_scale)
    attn = attn.transpose(1, 2).reshape(B, T, cfg.n_head * cfg.v_head).to(hidden.dtype)
    Wo = dq(f'blk.{il}.attn_output.weight')
    o = F.linear(attn, Wo)
    if lora is not None and lora.has(il):
        d = lora.delta(il, 'o', attn.dtype)
        if d is not None:
            o = o + F.linear(attn, d)
    hidden = hidden + o
    del Wq_a, Wq_b, Wkv_a, Wo, kb, vb, Q, K, Vv, attn, x
    if stop_after_attention:
        return hidden
    ffn_input = hidden
    y = rms_norm(hidden, dq1(f'blk.{il}.ffn_norm.weight'), eps)
    if not g.has(f'blk.{il}.ffn_gate_inp.weight'):
        Wg = dq(f'blk.{il}.ffn_gate.weight')
        Wu = dq(f'blk.{il}.ffn_up.weight')
        Wd = dq(f'blk.{il}.ffn_down.weight')
        hidden = hidden + F.linear(F.silu(F.linear(y, Wg)) * F.linear(y, Wu), Wd)
        del Wg, Wu, Wd
    else:
        moe = deepseek2_moe(y, il, g, cfg, weight_overrides=weight_overrides)
        Wgs = dq(f'blk.{il}.ffn_gate_shexp.weight')
        Wus = dq(f'blk.{il}.ffn_up_shexp.weight')
        Wds = dq(f'blk.{il}.ffn_down_shexp.weight')
        sh = F.linear(F.silu(F.linear(y, Wgs)) * F.linear(y, Wus), Wds)
        if os.environ.get('GGUF_QAT_FINITE_CHECK') == '1':
            if not torch.isfinite(moe).all():
                raise RuntimeError(f'non-finite routed MoE output at layer {il}')
            if not torch.isfinite(sh).all():
                raise RuntimeError(f'non-finite shared-expert output at layer {il}')
        hidden = hidden + moe + sh
        del Wgs, Wus, Wds, moe
    if sidecar_bank is not None:
        hidden = hidden + sidecar_bank(ffn_input)
    return hidden

def deepseek2_moe(y, il, g, cfg: Deepseek2Config, weight_overrides=None):
    B, T, H = y.shape
    yf = y.reshape(B * T, H)
    n_exp, top_k = (cfg.n_expert, cfg.n_expert_used)
    router_name = f'blk.{il}.ffn_gate_inp.weight'
    Wr = weight_overrides[router_name].to(torch.float32) if weight_overrides is not None and router_name in weight_overrides else g.dequant_2d_hf(router_name, dtype=torch.float32).detach()
    logits = F.linear(yf.float(), Wr)
    scores = torch.sigmoid(logits)
    sel = scores
    bias_name = f'blk.{il}.exp_probs_b.bias'
    if g.has(bias_name):
        bias = weight_overrides[bias_name].to(torch.float32).reshape(-1) if weight_overrides is not None and bias_name in weight_overrides else g.dequant(bias_name, dtype=torch.float32).detach().reshape(-1)
        sel = scores + bias
    topv, topi = torch.topk(sel, top_k, dim=-1)
    weights = torch.gather(scores, -1, topi)
    if cfg.expert_w_norm:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * cfg.expert_w_scale
    from cosmos_windows.gguf_qat.dequant import dequantize
    gi = g.info(f'blk.{il}.ffn_gate_exps.weight')
    ui = g.info(f'blk.{il}.ffn_up_exps.weight')
    di = g.info(f'blk.{il}.ffn_down_exps.weight')
    inter = int(gi.shape[1])
    expert_bytes = {gi.name: gi.n_bytes // n_exp, ui.name: ui.n_bytes // n_exp, di.name: di.n_bytes // n_exp}
    out = torch.zeros_like(yf)
    yb = yf.to(torch.bfloat16)
    for e in torch.unique(topi.reshape(-1)).tolist():
        tok_idx, slot = torch.where(topi == e)
        if tok_idx.numel() == 0:
            continue
        gkey = f'{gi.name}::expert:{e}'
        ukey = f'{ui.name}::expert:{e}'
        dkey = f'{di.name}::expert:{e}'
        if weight_overrides is not None and gkey in weight_overrides:
            ge = weight_overrides[gkey].to(torch.bfloat16)
        else:
            with torch.no_grad():
                raw = g.raw_bytes_slice(gi.name, e * expert_bytes[gi.name], expert_bytes[gi.name])
                ge = dequantize(raw, gi.quant, (inter, H)).to(torch.bfloat16)
                del raw
        if weight_overrides is not None and ukey in weight_overrides:
            ue = weight_overrides[ukey].to(torch.bfloat16)
        else:
            with torch.no_grad():
                raw = g.raw_bytes_slice(ui.name, e * expert_bytes[ui.name], expert_bytes[ui.name])
                ue = dequantize(raw, ui.quant, (inter, H)).to(torch.bfloat16)
                del raw
        if weight_overrides is not None and dkey in weight_overrides:
            de = weight_overrides[dkey].to(torch.bfloat16)
        else:
            with torch.no_grad():
                raw = g.raw_bytes_slice(di.name, e * expert_bytes[di.name], expert_bytes[di.name])
                de = dequantize(raw, di.quant, (H, inter)).to(torch.bfloat16)
                del raw
        xin = yb[tok_idx]
        gate = F.linear(xin, ge)
        up = F.linear(xin, ue)
        product = F.silu(gate) * up
        cur = F.linear(product, de)
        if os.environ.get('GGUF_QAT_FINITE_CHECK') == '1' and (not torch.isfinite(cur).all()):
            raise RuntimeError(f'non-finite routed expert at layer {il} expert {e}; gate_max={float(gate.float().abs().max())} up_max={float(up.float().abs().max())} product_max={float(product.float().abs().max())}')
        w = weights[tok_idx, slot].to(torch.bfloat16).unsqueeze(-1)
        out.index_add_(0, tok_idx, (cur * w).to(out.dtype))
        del ge, ue, de, gate, up, product
    return out.reshape(B, T, H).to(y.dtype)

def _causal_mask(T, device):
    i = torch.arange(T, device=device)
    keep = i[None, :] <= i[:, None]
    m = torch.zeros(T, T, device=device, dtype=torch.float32).masked_fill(~keep, float('-inf'))
    return m.view(1, 1, T, T)

class StreamedDeepseek2(nn.Module):

    def __init__(self, g, cfg: Deepseek2Config, rank=16, alpha=None, target_layers=None, device='cuda:0'):
        super().__init__()
        self.g = g
        self.cfg = cfg
        self.device = device
        if target_layers is None:
            target_layers = list(range(cfg.n_layer))
        alpha = alpha if alpha is not None else rank * 2
        StreamedLoRA.PROJ = ('q_a', 'q_b', 'kv_a', 'o')

        def dims(il, proj):
            if proj == 'q_a':
                return (cfg.q_lora, cfg.hidden)
            if proj == 'q_b':
                return (cfg.n_head * cfg.qk_head, cfg.q_lora)
            if proj == 'kv_a':
                return (cfg.kv_lora + cfg.qk_rope, cfg.hidden)
            if proj == 'o':
                return (cfg.hidden, cfg.n_head * cfg.v_head)
        self.lora = StreamedLoRA(cfg, rank, alpha, target_layers, dims, device)
        self.embd = g.dequant_2d_hf('token_embd.weight', dtype=torch.bfloat16).detach()
        self.out_norm = g.dequant('output_norm.weight', dtype=torch.float32).detach().reshape(-1)
        self.lm_head = g.dequant_2d_hf('output.weight', dtype=torch.bfloat16).detach()
        self._rope = {}
        self.weight_overrides = {}
        self.sidecar_banks = nn.ModuleDict()
        self.sidecars_enabled = True

    def set_weight_override(self, name, value):
        self.weight_overrides.clear()
        if name is not None:
            self.weight_overrides[str(name)] = value

    def attach_token_sidecars(self, layer_indices, expert_widths, *, gate_init=0.0, dtype=torch.float32):
        from cosmos_windows.gguf_qat.sidecar import SidecarExpertBank
        for layer in layer_indices:
            layer = int(layer)
            if not 0 <= layer < self.cfg.n_layer:
                raise ValueError(f'sidecar layer {layer} outside [0,{self.cfg.n_layer})')
            bank = SidecarExpertBank(self.cfg.hidden, list(expert_widths), eps=self.cfg.eps, act='silu', gate_init=gate_init)
            self.sidecar_banks[str(layer)] = bank.to(device=self.device, dtype=dtype)
        return self.sidecar_banks

    def _ropes(self, T):
        if T not in self._rope:
            self._rope[T] = yarn_rope_cache(self.cfg, T, self.device)
        return self._rope[T]

    def _global_weight(self, name, base, dtype):
        value = self.weight_overrides.get(name)
        return value.to(dtype) if value is not None else base

    def _layer(self, hidden, il, rope, mask):
        key = str(il)
        sidecar_bank = self.sidecar_banks[key] if self.sidecars_enabled and key in self.sidecar_banks else None
        return deepseek2_layer_forward(hidden, il, self.g, self.cfg, self.lora, rope, mask, weight_overrides=self.weight_overrides, sidecar_bank=sidecar_bank)

    def _weight_2d(self, name, dtype=torch.bfloat16):
        value = self.weight_overrides.get(name)
        return value.to(dtype) if value is not None else self.g.dequant_2d_hf(name, dtype=dtype).detach()

    def _weight_1d(self, name, dtype=torch.float32):
        value = self.weight_overrides.get(name)
        return value.to(dtype).reshape(-1) if value is not None else self.g.dequant(name, dtype=dtype).detach().reshape(-1)

    def post_attention_hidden(self, hidden, il, seqlen):
        return deepseek2_layer_forward(hidden, il, self.g, self.cfg, self.lora, self._ropes(seqlen), _causal_mask(seqlen, self.device), weight_overrides=self.weight_overrides, stop_after_attention=True)

    def router_bias_loss(self, hidden, il, *, temperature=0.05):
        ffn_input = self.post_attention_hidden(hidden, il, hidden.shape[1])
        normalized = rms_norm(ffn_input, self._weight_1d(f'blk.{il}.ffn_norm.weight'), self.cfg.eps)
        logits = F.linear(normalized.float(), self._weight_2d(f'blk.{il}.ffn_gate_inp.weight', torch.float32))
        scores = torch.sigmoid(logits)
        bias = self._weight_1d(f'blk.{il}.exp_probs_b.bias', torch.float32)
        soft_selection = torch.softmax((scores + bias) / max(float(temperature), 0.0001), dim=-1)
        utilization = soft_selection.reshape(-1, soft_selection.shape[-1]).mean(0)
        uniform = torch.full_like(utilization, 1.0 / utilization.numel())
        balance = utilization.numel() * F.mse_loss(utilization, uniform, reduction='sum')
        entropy = -(soft_selection * torch.log_softmax((scores + bias) / max(float(temperature), 0.0001), dim=-1)).sum(-1).mean()
        return balance + 0.0001 * entropy

    def indexer_distillation_loss(self, hidden, il):
        if self.cfg.arch != 'glm-dsa' or self.cfg.index_heads < 1:
            raise ValueError('indexer distillation requires a GLM-DSA indexer')
        B, T, _ = hidden.shape
        x = rms_norm(hidden, self._weight_1d(f'blk.{il}.attn_norm.weight'), self.cfg.eps)
        q_lora = F.linear(x.to(torch.bfloat16), self._weight_2d(f'blk.{il}.attn_q_a.weight'))
        q_lora = rms_norm(q_lora, self._weight_1d(f'blk.{il}.attn_q_a_norm.weight'), self.cfg.eps)
        q = F.linear(q_lora, self._weight_2d(f'blk.{il}.attn_q_b.weight'))
        q = q.view(B, T, self.cfg.n_head, self.cfg.qk_head)
        q_nope, q_pe = (q[..., :self.cfg.qk_nope], q[..., self.cfg.qk_nope:])
        compressed = F.linear(x.to(torch.bfloat16), self._weight_2d(f'blk.{il}.attn_kv_a_mqa.weight'))
        kv = rms_norm(compressed[..., :self.cfg.kv_lora], self._weight_1d(f'blk.{il}.attn_kv_a_norm.weight'), self.cfg.eps).to(torch.bfloat16)
        k_pe = compressed[..., self.cfg.kv_lora:]
        k_name = f'blk.{il}.attn_k_b.weight'
        k_override = self.weight_overrides.get(k_name)
        if k_override is None:
            from cosmos_windows.gguf_qat.dequant import dequantize
            info = self.g.info(k_name)
            k_raw = dequantize(self.g.raw_bytes_pread(k_name), info.quant, info.shape)
        else:
            k_raw = k_override
        k_b = k_raw.reshape(self.cfg.n_head, self.cfg.kv_lora, self.cfg.qk_nope).to(torch.bfloat16)
        k_nope = torch.einsum('btc,hcd->bthd', kv, k_b)
        cos, sin = self._ropes(T)
        q_pe = apply_main_rope(self.cfg, q_pe.transpose(1, 2), cos, sin).transpose(1, 2)
        k_pe = apply_main_rope(self.cfg, k_pe.unsqueeze(2).transpose(1, 2), cos, sin).transpose(1, 2)
        teacher_q = torch.cat((q_nope, q_pe), dim=-1).float().transpose(1, 2)
        teacher_k = torch.cat((k_nope, k_pe.expand(B, T, self.cfg.n_head, self.cfg.qk_rope)), dim=-1).float().transpose(1, 2)
        dense_scores = torch.matmul(teacher_q, teacher_k.transpose(-1, -2)) * self.cfg.kq_scale
        teacher = torch.logsumexp(dense_scores, dim=1) - math.log(self.cfg.n_head)
        idx_q = F.linear(q_lora.float(), self._weight_2d(f'blk.{il}.indexer.attn_q_b.weight', torch.float32))
        idx_q = idx_q.view(B, T, self.cfg.index_heads, self.cfg.index_head).transpose(1, 2)
        idx_k = F.linear(x.float(), self._weight_2d(f'blk.{il}.indexer.attn_k.weight', torch.float32))
        idx_k = F.layer_norm(idx_k, (self.cfg.index_head,), self._weight_1d(f'blk.{il}.indexer.k_norm.weight'), self._weight_1d(f'blk.{il}.indexer.k_norm.bias'), 1e-06)
        rope = self.cfg.qk_rope
        idx_q = torch.cat((apply_rope_neox(idx_q[..., :rope], cos, sin), idx_q[..., rope:]), -1)
        idx_k_rope = apply_rope_neox(idx_k[..., :rope].unsqueeze(1), cos, sin).squeeze(1)
        idx_k = torch.cat((idx_k_rope, idx_k[..., rope:]), -1)
        per_head = torch.einsum('bhtd,bkd->bhtk', idx_q, idx_k)
        per_head = F.relu(per_head * self.cfg.index_head ** (-0.5))
        head_gate = F.linear(x.float(), self._weight_2d(f'blk.{il}.indexer.proj.weight', torch.float32)).transpose(1, 2).unsqueeze(-1)
        student = (per_head * head_gate * self.cfg.index_heads ** (-0.5)).sum(1)
        causal = torch.triu(torch.ones(T, T, device=hidden.device, dtype=torch.bool), diagonal=1)
        teacher = teacher.masked_fill(causal, -1000000000.0).detach()
        student = student.masked_fill(causal, -1000000000.0)
        return F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction='batchmean') / T

    def nextn_input(self, input_ids):
        if self.cfg.nextn_layers != 1 or input_ids.shape[1] < 3:
            raise ValueError('NextN training requires GLM-DSA with one draft layer and sequence length >= 3')
        with torch.no_grad():
            trunk_hidden = self.prefix_hidden(input_ids, self.cfg.n_layer)
        il = self.cfg.n_layer
        token_embd = F.embedding(input_ids[:, 1:], self.embd)
        hidden = rms_norm(trunk_hidden[:, :-1], self._weight_1d(f'blk.{il}.nextn.hnorm.weight'), self.cfg.eps)
        token_embd = rms_norm(token_embd, self._weight_1d(f'blk.{il}.nextn.enorm.weight'), self.cfg.eps)
        eh_weight = self._weight_2d(f'blk.{il}.nextn.eh_proj.weight')
        return F.linear(torch.cat((token_embd, hidden), -1).to(eh_weight.dtype), eh_weight)

    def nextn_loss(self, input_ids, labels):
        il = self.cfg.n_layer
        fused = self.nextn_input(input_ids)
        fused = deepseek2_layer_forward(fused, il, self.g, self.cfg, self.lora, self._ropes(fused.shape[1]), _causal_mask(fused.shape[1], self.device), weight_overrides=self.weight_overrides)
        fused = rms_norm(fused, self._weight_1d(f'blk.{il}.nextn.shared_head_norm.weight'), self.cfg.eps)
        logits = F.linear(fused[:, :-1].to(self.lm_head.dtype), self.lm_head)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 2:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def prefix_hidden(self, input_ids, prefix_end):
        B, T = input_ids.shape
        hidden = F.embedding(input_ids, self._global_weight('token_embd.weight', self.embd, torch.bfloat16))
        rope = self._ropes(T)
        mask = _causal_mask(T, self.device)
        with _ds_sweep(self.g) as ds_on:
            for il in range(prefix_end):
                with _ds_layer(self.g, il, ds_on):
                    hidden = self._layer(hidden, il, rope, mask)
                if os.environ.get('GGUF_QAT_FINITE_CHECK') == '1' and (not torch.isfinite(hidden).all()):
                    finite = hidden.float()[torch.isfinite(hidden)]
                    maximum = float(finite.abs().max()) if finite.numel() else float('nan')
                    raise RuntimeError(f'non-finite streamed prefix activation after layer {il}; largest_finite_abs={maximum}')
                if il % 2 == 1:
                    torch.cuda.empty_cache()
        return hidden.detach()

    def suffix_loss(self, hidden, labels, prefix_end, T, use_checkpoint=True, use_suffix_ds=False):
        rope = self._ropes(T)
        mask = _causal_mask(T, self.device)
        for il in range(prefix_end, self.cfg.n_layer):

            def fn(h, _il=il):
                return self._layer(h, _il, rope, mask)
            hidden = ckpt.checkpoint(fn, hidden, use_reentrant=False) if use_checkpoint and self.training else fn(hidden)
            torch.cuda.empty_cache()
        hidden = rms_norm(hidden, self._global_weight('output_norm.weight', self.out_norm, torch.float32), self.cfg.eps)
        logits = F.linear(hidden, self._global_weight('output.weight', self.lm_head, torch.bfloat16))
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        return F.cross_entropy(sl, labels[:, 1:].reshape(-1), ignore_index=-100)

    def forward(self, input_ids, labels=None, use_checkpoint=True):
        B, T = input_ids.shape
        hidden = F.embedding(input_ids, self._global_weight('token_embd.weight', self.embd, torch.bfloat16))
        rope = self._ropes(T)
        mask = _causal_mask(T, self.device)
        with _ds_sweep(self.g) as ds_on:
            for il in range(self.cfg.n_layer):
                with _ds_layer(self.g, il, ds_on):

                    def fn(h, _il=il):
                        return self._layer(h, _il, rope, mask)
                    hidden = ckpt.checkpoint(fn, hidden, use_reentrant=False) if use_checkpoint and self.training else fn(hidden)
                torch.cuda.empty_cache()
        hidden = rms_norm(hidden, self._global_weight('output_norm.weight', self.out_norm, torch.float32), self.cfg.eps)
        logits = F.linear(hidden, self._global_weight('output.weight', self.lm_head, torch.bfloat16))
        if labels is None:
            return logits
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        return F.cross_entropy(sl, labels[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def suffix_logits(self, hidden, prefix_end, T):
        rope = self._ropes(T)
        mask = _causal_mask(T, self.device)
        for il in range(prefix_end, self.cfg.n_layer):
            hidden = self._layer(hidden, il, rope, mask)
            torch.cuda.empty_cache()
        hidden = rms_norm(hidden, self._global_weight('output_norm.weight', self.out_norm, torch.float32), self.cfg.eps)
        return F.linear(hidden, self._global_weight('output.weight', self.lm_head, torch.bfloat16))

    @torch.no_grad()
    def generate_greedy(self, input_ids, max_new_tokens=24, eos_id=None):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -min(ids.shape[1], 128):], labels=None, use_checkpoint=False)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
        return ids

def build_streamed_deepseek2(first_shard, device='cuda:0', rank=16, alpha=None, target_layers=None, ds=True, n_gpu_buffers=3):
    if ds:
        from cosmos_windows.gguf_qat.ds_streamed_reader import DSStreamedGGUF
        g = DSStreamedGGUF(first_shard, device=device, n_gpu_buffers=n_gpu_buffers)
    else:
        from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
        g = StreamedGGUF(first_shard, device=device)
    if g.arch() not in {'deepseek2', 'glm-dsa'}:
        raise ValueError(f'expected deepseek2 or glm-dsa GGUF, got {g.arch()!r}')
    cfg = Deepseek2Config(g)
    model = StreamedDeepseek2(g, cfg, rank=rank, alpha=alpha, target_layers=target_layers, device=device)
    return (model, g, cfg)
if __name__ == '__main__':
    import os, time
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
    start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '10.0')), label='streamed_deepseek2')
    first = ''
    dev = 'cuda:0'
    print(f'device {dev} = {torch.cuda.get_device_name(0)}', flush=True)
    model, g, cfg = build_streamed_deepseek2(first, device=dev, rank=8)
    print(f'deepseek2 cfg: L={cfg.n_layer} H={cfg.hidden} nh={cfg.n_head} qk={cfg.qk_head}(nope{cfg.qk_nope}+rope{cfg.qk_rope}) v={cfg.v_head} qlora={cfg.q_lora} kvlora={cfg.kv_lora} experts={cfg.n_expert}/top{cfg.n_expert_used} escale={cfg.expert_w_scale:.3f} kq_scale={cfg.kq_scale:.4f} vocab={cfg.vocab}', flush=True)
    n_layers = int(os.environ.get('STREAM_NLAYERS', '0'))
    if n_layers:
        cfg.n_layer = n_layers
        print(f'[smoke] limiting to first {n_layers} layers', flush=True)
    from cosmos_windows.gguf_qat.bpe_tokenizer_from_gguf import build_bpe_tokenizer_from_gguf
    tok = build_bpe_tokenizer_from_gguf(g)
    prompt = tok.apply_chat_template([{'role': 'user', 'content': 'What is the capital of France?'}], add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(dev)
    print(f'[smoke] prompt ids {ids.shape}', flush=True)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model.forward(ids, labels=None, use_checkpoint=False)
    print(f'[smoke] forward {tuple(logits.shape)} in {time.time() - t0:.1f}s VRAM {torch.cuda.max_memory_allocated(dev) / 1000000000.0:.1f}GB', flush=True)
    nxt = logits[0, -1].topk(5)
    print('[smoke] top-5:', [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(nxt.values, nxt.indices)], flush=True)
    g.close()
    print(f'[smoke] peak RSS {peak_rss_gb():.3f} GB', flush=True)
