# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/streamed_forward.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class Step35Config:

    def __init__(self, g):
        A = 'step35'
        self.n_layer = int(g.meta(f'{A}.block_count'))
        self.hidden = int(g.meta(f'{A}.embedding_length'))
        self.head_count = [int(x) for x in g.meta(f'{A}.attention.head_count')]
        self.head_count_kv = [int(x) for x in g.meta(f'{A}.attention.head_count_kv')]
        self.head_dim = int(g.meta(f'{A}.attention.key_length'))
        self.v_head_dim = int(g.meta(f'{A}.attention.value_length', self.head_dim))
        self.eps = float(g.meta(f'{A}.attention.layer_norm_rms_epsilon', 1e-05))
        self.sliding_window = int(g.meta(f'{A}.attention.sliding_window', 512))
        swp = g.meta(f'{A}.attention.sliding_window_pattern')
        self.is_swa = [bool(x) for x in swp]
        self.rope_base = float(g.meta(f'{A}.rope.freq_base', 5000000.0))
        self.rope_base_swa = float(g.meta(f'{A}.rope.freq_base_swa', 10000.0))
        self.n_dense = int(g.meta(f'{A}.leading_dense_block_count', 3))
        self.n_expert = int(g.meta(f'{A}.expert_count', 0))
        self.n_expert_used = int(g.meta(f'{A}.expert_used_count', 0))
        self.expert_w_norm = bool(g.meta(f'{A}.expert_weights_norm', True))
        self.expert_w_scale = float(g.meta(f'{A}.expert_weights_scale', 1.0))
        self.expert_gating = int(g.meta(f'{A}.expert_gating_func', 2))
        emb = g.info('token_embd.weight')
        self.vocab = int(emb.shape[1])
        self.n_rot = self.head_dim

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w.float()).to(dt)

def build_rope_cache(n_rot: int, base: float, seqlen: int, device, dtype=torch.float32):
    half = n_rot // 2
    inv_freq = 1.0 / base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    t = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos().to(dtype), emb.sin().to(dtype))

def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    n_rot = cos.shape[-1]
    if n_rot <= 0:
        return x
    xr, xp = (x[..., :n_rot], x[..., n_rot:])
    half = n_rot // 2
    x1, x2 = (xr[..., :half], xr[..., half:])
    rot = torch.cat([-x2, x1], dim=-1)
    c = cos.view(1, 1, cos.shape[0], n_rot)
    s = sin.view(1, 1, sin.shape[0], n_rot)
    xr = xr * c + rot * s
    return torch.cat([xr, xp], dim=-1)

class StreamedLoRA(nn.Module):
    PROJ = ('q', 'k', 'v', 'o')

    def __init__(self, cfg, rank: int, alpha: float, target_layers, dims, device):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.target_layers = set(target_layers)
        for il in target_layers:
            for proj in self.PROJ:
                out_f, in_f = dims(il, proj)
                a = torch.zeros(rank, in_f, device=device, dtype=torch.float32)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                b = torch.zeros(out_f, rank, device=device, dtype=torch.float32)
                self.A[f'l{il}_{proj}'] = nn.Parameter(a)
                self.B[f'l{il}_{proj}'] = nn.Parameter(b)

    def delta(self, il: int, proj: str, dtype):
        key = f'l{il}_{proj}'
        if key not in self.A:
            return None
        return (self.B[key] @ self.A[key]).to(dtype) * self.scaling

    def has(self, il):
        return il in self.target_layers

def linear_with_lora(x, W, lora, il, proj):
    out = F.linear(x, W)
    if lora is not None:
        d = lora.delta(il, proj, x.dtype)
        if d is not None:
            out = out + F.linear(x, d)
    return out

def step35_layer_forward(hidden, il, g, cfg: Step35Config, lora, rope_full, rope_swa, attn_mask_full, attn_mask_swa):
    dev = hidden.device
    B, T, H = hidden.shape
    n_head = cfg.head_count[il]
    n_kv = cfg.head_count_kv[il]
    hd = cfg.head_dim
    is_swa = cfg.is_swa[il]
    eps = cfg.eps

    def dq(name):
        return g.dequant_2d_hf(name, dtype=torch.bfloat16).detach()

    def dq1(name):
        return g.dequant(name, dtype=torch.float32).detach().reshape(-1)
    attn_norm_w = dq1(f'blk.{il}.attn_norm.weight')
    x = rms_norm(hidden, attn_norm_w, eps)
    Wq = dq(f'blk.{il}.attn_q.weight')
    Wk = dq(f'blk.{il}.attn_k.weight')
    Wv = dq(f'blk.{il}.attn_v.weight')
    Wo = dq(f'blk.{il}.attn_output.weight')
    q = linear_with_lora(x, Wq, lora, il, 'q')
    k = linear_with_lora(x, Wk, lora, il, 'k')
    v = linear_with_lora(x, Wv, lora, il, 'v')
    q = q.view(B, T, n_head, hd).transpose(1, 2)
    k = k.view(B, T, n_kv, hd).transpose(1, 2)
    v = v.view(B, T, n_kv, cfg.v_head_dim).transpose(1, 2)
    if g.has(f'blk.{il}.attn_q_norm.weight'):
        qn = dq1(f'blk.{il}.attn_q_norm.weight')
        q = rms_norm(q, qn, eps)
    if g.has(f'blk.{il}.attn_k_norm.weight'):
        kn = dq1(f'blk.{il}.attn_k_norm.weight')
        k = rms_norm(k, kn, eps)
    cos, sin = rope_swa if is_swa else rope_full
    q = apply_rope_neox(q, cos, sin)
    k = apply_rope_neox(k, cos, sin)
    if n_kv != n_head:
        rep = n_head // n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    mask = attn_mask_swa if is_swa else attn_mask_full
    scale = 1.0 / math.sqrt(hd)
    attn = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask, scale=scale)
    attn = attn.to(hidden.dtype)
    if g.has(f'blk.{il}.attn_gate.weight'):
        Wg = dq(f'blk.{il}.attn_gate.weight')
        gate = torch.sigmoid(F.linear(x, Wg))
        gate = gate.transpose(1, 2).unsqueeze(-1)
        attn = attn * gate
    attn = attn.transpose(1, 2).reshape(B, T, n_head * cfg.v_head_dim)
    attn_out = linear_with_lora(attn, Wo, lora, il, 'o')
    hidden = hidden + attn_out
    del Wq, Wk, Wv, Wo, x, q, k, v, attn
    ffn_norm_w = dq1(f'blk.{il}.ffn_norm.weight')
    y = rms_norm(hidden, ffn_norm_w, eps)
    if not g.has(f'blk.{il}.ffn_gate_inp.weight'):
        Wg_ = dq(f'blk.{il}.ffn_gate.weight')
        Wu = dq(f'blk.{il}.ffn_up.weight')
        Wd = dq(f'blk.{il}.ffn_down.weight')
        ff = F.linear(F.silu(F.linear(y, Wg_)) * F.linear(y, Wu), Wd)
        hidden = hidden + ff
        del Wg_, Wu, Wd
    else:
        moe = step35_moe(y, il, g, cfg)
        Wgs = dq(f'blk.{il}.ffn_gate_shexp.weight')
        Wus = dq(f'blk.{il}.ffn_up_shexp.weight')
        Wds = dq(f'blk.{il}.ffn_down_shexp.weight')
        sh = F.linear(F.silu(F.linear(y, Wgs)) * F.linear(y, Wus), Wds)
        hidden = hidden + moe + sh
        del Wgs, Wus, Wds, moe
    return hidden

def step35_moe(y, il, g, cfg: Step35Config):
    B, T, H = y.shape
    yf = y.reshape(B * T, H)
    n_exp, top_k = (cfg.n_expert, cfg.n_expert_used)
    Wr = g.dequant_2d_hf(f'blk.{il}.ffn_gate_inp.weight', dtype=torch.float32).detach()
    logits = F.linear(yf.float(), Wr)
    if g.has(f'blk.{il}.exp_probs_b.bias'):
        bias = g.dequant(f'blk.{il}.exp_probs_b.bias', dtype=torch.float32).detach().reshape(-1)
    else:
        bias = None
    scores = torch.sigmoid(logits)
    sel = scores + bias if bias is not None else scores
    topv, topi = torch.topk(sel, top_k, dim=-1)
    weights = torch.gather(scores, -1, topi)
    if cfg.expert_w_norm:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * cfg.expert_w_scale
    raw_g = g.raw_bytes_pread(f'blk.{il}.ffn_gate_exps.weight')
    raw_u = g.raw_bytes_pread(f'blk.{il}.ffn_up_exps.weight')
    raw_d = g.raw_bytes_pread(f'blk.{il}.ffn_down_exps.weight')
    gi = g.info(f'blk.{il}.ffn_gate_exps.weight')
    ui = g.info(f'blk.{il}.ffn_up_exps.weight')
    di = g.info(f'blk.{il}.ffn_down_exps.weight')
    inter = int(gi.shape[1])
    from cosmos_windows.gguf_qat.dequant import dequantize
    gper = raw_g.numel() // n_exp
    uper = raw_u.numel() // n_exp
    dper = raw_d.numel() // n_exp
    raw_g = raw_g.view(n_exp, gper)
    raw_u = raw_u.view(n_exp, uper)
    raw_d = raw_d.view(n_exp, dper)
    out = torch.zeros_like(yf)
    hit = torch.unique(topi.reshape(-1))
    yb = yf.to(torch.bfloat16)
    for e in hit.tolist():
        tok_idx, slot = torch.where(topi == e)
        if tok_idx.numel() == 0:
            continue
        with torch.no_grad():
            ge = dequantize(raw_g[e], gi.quant, (inter, H)).to(torch.bfloat16)
            ue = dequantize(raw_u[e], ui.quant, (inter, H)).to(torch.bfloat16)
            de = dequantize(raw_d[e], di.quant, (H, inter)).to(torch.bfloat16)
        xin = yb[tok_idx]
        cur = F.linear(F.silu(F.linear(xin, ge)) * F.linear(xin, ue), de)
        w = weights[tok_idx, slot].to(torch.bfloat16).unsqueeze(-1)
        out.index_add_(0, tok_idx, (cur * w).to(out.dtype))
        del ge, ue, de
    del raw_g, raw_u, raw_d
    return out.reshape(B, T, H).to(y.dtype)
