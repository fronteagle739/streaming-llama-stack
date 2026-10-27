# MiniMax-M3 (text-only) for MLX — block-sparse (MSA) GQA + DeepSeek-V3-style MoE.
#
# Vendored in mlx-vlm (the synced source) so it reaches the Macs; imports only mlx-lm primitives that
# are present on the Macs (SwitchGLU, RMSNorm, SDPA, masks). Adapted from mlx_lm/models/minimax.py with
# the M3 deltas: PER-HEAD QK-norm, partial RoPE, the MSA block-sparse attention (the validated selection
# from scripts/m3_msa_mlx.py in the RTX repo), DeepSeek-V3 MoE (sigmoid gating + routing bias + SHARED
# experts), 3 leading DENSE layers, and gemma-style norm.
#
# Parity target = the RTX gguf engine (prefill_llama.cpp/src/models/minimax-m3.cpp), validated there via
# the coherence gate (sparse == dense at full block coverage). FIRST version: prefill MSA (no cache);
# decode (cache) falls back to dense until the MLX indexer-K cache lands. swiglu_oai + real-weight
# correctness are weight-gated refinements (the coherence gate is activation-independent).

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU

NEG = -1e30


class M3Cache(KVCache):
    """KVCache that ALSO holds the per-token MSA index-K, so a decode query can select blocks over the
    cached prior index-K (the index-K is a separate projection of the hidden states, not derivable from
    the main K). Grows alongside K/V via update_idx."""

    def __init__(self):
        super().__init__()
        self.idx_keys = None  # [n_kv, idx_dim]

    def update_idx(self, idx_k):
        # idx_k: [B, n_new, d] (or [n_new, d]); grow along the n_kv axis (-2) -> handles B>=1.
        self.idx_keys = idx_k if self.idx_keys is None else mx.concatenate([self.idx_keys, idx_k], axis=-2)
        return self.idx_keys


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "minimax_m3"
    hidden_size: int = 4096
    intermediate_size: int = 11008          # dense-layer FFN
    moe_intermediate_size: int = 1408       # per-expert FFN
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_hidden_layers: int = 60
    first_k_dense_replace: int = 3          # leading DENSE layers; the rest are MoE
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    vocab_size: int = 200064
    rms_norm_eps: float = 1e-5
    rope_theta: float = 5_000_000.0
    rotary_dim: int = 64                    # partial RoPE (n_rot of head_dim)
    tie_word_embeddings: bool = False
    use_gemma_norm: bool = True
    swiglu_alpha: float = 1.702      # swiglu_oai (matches the RTX GGML op: minimax-m3.cpp)
    swiglu_limit: float = 7.0
    # MSA (block-sparse) indexer config
    indexer_n_head: int = 4                 # == num_key_value_heads
    indexer_head_size: int = 128
    indexer_top_k: int = 16                 # in BLOCKS
    sparse_block_size: int = 128
    sparse_init_blocks: int = 0
    sparse_local_blocks: int = 1
    num_mtp_modules: int = 0                 # DeepSeek-V3-style NextN/MTP draft heads (0 = none)


# ---------------------------------------------------------------------------
# MSA block selection (vendored from the validated scripts/m3_msa_mlx.py)
# ---------------------------------------------------------------------------
def _index_score(iq, ik, block_size):
    """BlockMaxPool index score. iq:[n,H,d] ik:[n,d] -> [H,n,n_blocks]."""
    n, H, d = iq.shape
    n_blocks = (n + block_size - 1) // block_size
    pad = n_blocks * block_size - n
    qk = mx.matmul(iq.transpose(1, 0, 2), ik.T[None])      # [H, n(q), n(k)]
    tri = mx.tril(mx.ones((n, n)))                          # causal: j<=t
    qk = mx.where(tri[None] > 0, qk, NEG)
    if pad:
        qk = mx.concatenate([qk, mx.full((H, n, pad), NEG)], axis=2)
    qk = qk.reshape(H, n, n_blocks, block_size)
    return mx.max(qk, axis=3)                               # [H, n, n_blocks]


def _select_blocks(iq, ik, topk, init_blocks, local_blocks, block_size):
    """Per-(head, token) selected block ids (sorted asc, -1 pad), len topk. -> [H,n,topk]."""
    score = _index_score(iq, ik, block_size)               # [H,n,n_blocks]
    H, n, n_blocks = score.shape
    bidx = mx.arange(n_blocks)
    tpos = mx.arange(n)
    valid = (tpos // block_size + 1)[:, None]               # [n,1] blocks holding j<=t
    causal = bidx[None, :] < valid                          # [n,n_blocks]
    local_mask = (bidx[None, :] >= mx.maximum(0, valid - local_blocks)) & causal
    init_mask = (bidx[None, :] < init_blocks) & causal
    s = mx.where(causal, score, NEG)
    s = mx.where(local_mask[None], 1e29, s)                 # force recent (local)
    s = mx.where(init_mask[None], 1e30, s)                  # force sink (init) — wins on overlap
    k = min(topk, n_blocks)
    part = mx.argpartition(-s, kth=k - 1, axis=2)[:, :, :k]
    part = mx.sort(part, axis=2).astype(mx.int32)
    part = mx.where(part < valid[None], part, -1)
    if k < topk:
        part = mx.concatenate([part, mx.full((H, n, topk - k), -1, dtype=part.dtype)], axis=2)
    return part


def _allowed_mask(sel_g, n, block_size):
    """[n,n] bool: key j allowed for query t iff block(j) in sel_g[t] (and causal). sel_g:[n,topk]."""
    blk_of_key = mx.arange(n) // block_size
    eq = sel_g[:, :, None] == blk_of_key[None, None, :]     # [n,topk,n]
    allowed = mx.any(eq, axis=1)                            # [n,n]
    causal = mx.tril(mx.ones((n, n))) > 0
    return allowed & causal


def _msa_general_mask(iq, idx_keys, q_offset, args):
    """Per-HEAD additive MSA mask [H, L, n_kv] (0 selected / NEG pruned). GENERAL: L query tokens at
    positions q_offset..q_offset+L-1 select blocks over n_kv cached keys — subsumes prefill (L==n_kv,
    offset 0) AND decode (L==1, offset=n_kv-1). Each kv-group's selection expands to its gs query heads."""
    Hkv = args.indexer_n_head
    H = args.num_attention_heads
    gs = H // Hkv
    bs = args.sparse_block_size
    L = iq.shape[0]
    n_kv = idx_keys.shape[0]
    n_blocks = (n_kv + bs - 1) // bs
    pad = n_blocks * bs - n_kv

    qk = mx.matmul(iq.transpose(1, 0, 2), idx_keys.T[None])      # [Hkv, L, n_kv]
    qpos = (q_offset + mx.arange(L))[:, None]                    # [L,1]
    kpos = mx.arange(n_kv)[None, :]                              # [1,n_kv]
    causal = kpos <= qpos                                        # [L,n_kv]
    qk = mx.where(causal[None], qk, NEG)
    if pad:
        qk = mx.concatenate([qk, mx.full((Hkv, L, pad), NEG)], axis=2)
    bsc = mx.max(qk.reshape(Hkv, L, n_blocks, bs), axis=3)       # [Hkv,L,n_blocks] block-max-pool

    bidx = mx.arange(n_blocks)
    valid = (qpos // bs) + 1                                     # [L,1] # causal blocks
    causal_b = bidx[None, :] < valid                            # [L,n_blocks]
    local_m = (bidx[None, :] >= mx.maximum(0, valid - args.sparse_local_blocks)) & causal_b
    init_m = (bidx[None, :] < args.sparse_init_blocks) & causal_b
    s = mx.where(causal_b[None], bsc, NEG)
    s = mx.where(local_m[None], 1e29, s)                        # force recent (local)
    s = mx.where(init_m[None], 1e30, s)                        # force sink (init) — wins on overlap
    k = min(args.indexer_top_k, n_blocks)
    part = mx.argpartition(-s, kth=k - 1, axis=2)[:, :, :k]      # [Hkv,L,k] selected block ids

    blk_of_key = mx.arange(n_kv) // bs                          # [n_kv]
    eq = part[:, :, :, None] == blk_of_key[None, None, None, :]  # [Hkv,L,k,n_kv]
    allowed = mx.any(eq, axis=2) & causal[None]                # [Hkv,L,n_kv]
    grp = mx.where(allowed, 0.0, NEG)
    return mx.stack([grp[h // gs] for h in range(H)])          # [H,L,n_kv]


# ---------------------------------------------------------------------------
class M3Attention(nn.Module):
    def __init__(self, args: ModelArgs, sparse: bool):
        super().__init__()
        self.args = args
        self.sparse = sparse
        H, Hkv, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.H, self.Hkv, self.hd = H, Hkv, hd
        self.scale = hd**-0.5

        self.q_proj = nn.Linear(args.hidden_size, H * hd, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, Hkv * hd, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, Hkv * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, args.hidden_size, bias=False)

        # PER-HEAD QK-norm (a single head_dim vector applied to every head)
        self.q_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

        if sparse:
            idx_h, idx_d = args.indexer_n_head, args.indexer_head_size
            self.index_q_proj = nn.Linear(args.hidden_size, idx_h * idx_d, bias=False)
            self.index_k_proj = nn.Linear(args.hidden_size, idx_d, bias=False)
            self.index_q_norm = nn.RMSNorm(idx_d, eps=args.rms_norm_eps)
            self.index_k_norm = nn.RMSNorm(idx_d, eps=args.rms_norm_eps)
            self.index_rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        H, Hkv, hd = self.H, self.Hkv, self.hd

        q = self.q_proj(x).reshape(B, L, H, hd)
        k = self.k_proj(x).reshape(B, L, Hkv, hd)
        v = self.v_proj(x).reshape(B, L, Hkv, hd)
        q = self.q_norm(q)                                  # per-head (over hd)
        k = self.k_norm(k)
        q = q.transpose(0, 2, 1, 3)                         # [B,H,L,hd]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        attn_mask = mask
        # MSA block-sparse (any B): per-head additive mask over the n_kv cache keys. Works for PREFILL
        # (cache None or M3Cache, q_offset 0) AND DECODE (M3Cache holds the prior index-K), B>=1. The
        # index Q/K are RoPE'd at the query positions (position-aware selection, matches RTX); ik is roped
        # BEFORE caching so cached prior index-K stays roped at its own positions (decode-consistency holds).
        if self.sparse and os.getenv("M3_MLX_SPARSE", "1") != "0":
            a = self.args
            iq = self.index_q_norm(self.index_q_proj(x).reshape(B, L, a.indexer_n_head, a.indexer_head_size))
            ik = self.index_k_norm(self.index_k_proj(x).reshape(B, L, a.indexer_head_size))
            q_offset = (cache.offset - L) if isinstance(cache, M3Cache) else 0  # position of query 0
            iq = self.index_rope(iq.transpose(0, 2, 1, 3), offset=q_offset).transpose(0, 2, 1, 3)  # [B,L,Hidx,d]
            ik = self.index_rope(ik, offset=q_offset)                                               # [B,L,d]
            idx_keys = cache.update_idx(ik) if isinstance(cache, M3Cache) else ik                    # [B,n_kv,d]
            attn_mask = mx.stack(
                [_msa_general_mask(iq[b], idx_keys[b], q_offset, a) for b in range(B)])  # [B,H,L,n_kv]

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


def _swiglu_oai(x_up, x_gate, alpha, limit):
    # GGML ggml_swiglu_oai parity (minimax-m3.cpp): x=min(gate,limit), y=clamp(up,+-limit),
    # out = (x * sigmoid(alpha*x)) * (y + 1).
    g = mx.minimum(x_gate, limit)
    u = mx.clip(x_up, -limit, limit)
    return (g * mx.sigmoid(alpha * g)) * (u + 1.0)


class SwiGLUOAI(nn.Module):
    """swiglu_oai activation for SwitchGLU (called as activation(x_up, x_gate))."""
    def __init__(self, alpha: float, limit: float):
        super().__init__()
        self.alpha, self.limit = alpha, limit

    def __call__(self, x_up, x_gate):
        return _swiglu_oai(x_up, x_gate, self.alpha, self.limit)


class M3MLP(nn.Module):
    """Dense swiglu_oai FFN (the leading first_k_dense_replace layers + the shared expert)."""
    def __init__(self, hidden_size: int, intermediate_size: int, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.alpha, self.limit = alpha, limit

    def __call__(self, x):
        return self.down_proj(_swiglu_oai(self.up_proj(x), self.gate_proj(x), self.alpha, self.limit))


class M3MoE(nn.Module):
    """DeepSeek-V3-style MoE: sigmoid gate + routing-bias top-k + routed experts + shared experts."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
                                    activation=SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))
        self.n_shared = args.n_shared_experts
        if self.n_shared > 0:
            self.shared_experts = M3MLP(args.hidden_size, args.moe_intermediate_size * self.n_shared,
                                        args.swiglu_alpha, args.swiglu_limit)

    def __call__(self, x):
        gates = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig = scores
        scores = scores + self.e_score_correction_bias
        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        w = mx.take_along_axis(orig, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w.astype(x.dtype)
        y = self.switch_mlp(x, inds)
        y = (y * w[..., None]).sum(axis=-2)
        if self.n_shared > 0:
            y = y + self.shared_experts(x)
        return y


class M3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        sparse = layer_idx >= args.first_k_dense_replace
        self.self_attn = M3Attention(args, sparse=sparse)
        self.mlp = (M3MoE(args) if sparse
                    else M3MLP(args.hidden_size, args.intermediate_size, args.swiglu_alpha, args.swiglu_limit))
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return r + self.mlp(self.post_attention_layernorm(r))


class M3MTPLayer(nn.Module):
    """DeepSeek-V3-style NextN / MTP draft head (mirrors vllm MiniMaxM3MultiTokenPredictorLayer): combine the
    normalized NEXT-token embedding with the normalized PREVIOUS hidden state via eh_proj, run ONE full M3
    decoder layer (forced sparse-attn + MoE — `is_mtp_block` in vllm only swaps the config source, so
    structurally it's a standard M3 sparse+MoE layer), then `final_layernorm` before the shared lm_head.
    Token at position 0 is masked out (no previous token to condition on there)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        # a sparse layer index forces sparse-attn + MoE (matches force_sparse_attn/force_moe=True)
        self.transformer_layer = M3DecoderLayer(args, args.first_k_dense_replace)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_embeds, prev_hidden, positions, cache=None):
        p = positions if positions.ndim == input_embeds.ndim - 1 else positions[None]
        input_embeds = mx.where(p[..., None] == 0, 0.0, input_embeds)  # MTP ignores the position-0 token
        h = self.eh_proj(mx.concatenate([self.enorm(input_embeds), self.hnorm(prev_hidden)], axis=-1))
        mask = create_attention_mask(h, cache)
        return self.transformer_layer(h, mask, cache)                  # post-residual (== vllm hidden+=residual)

    def norm_for_logits(self, h):
        return self.final_layernorm(h)


class M3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [M3DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # dense layers want a normal causal mask; the sparse layers build their own MSA mask internally.
        causal = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, causal, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = M3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        out = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # M3Cache (KVCache + per-token index-K) so the MSA selection works during DECODE.
        return [M3Cache() for _ in self.model.layers]

    def sanitize(self, weights):
        # gemma norm: the HF M3 checkpoint stores RAW norm weights; the RTX gguf converter folds w->w+1 so
        # standard RMS works (prefill_llama.cpp convert_hf_to_gguf MiniMaxM3, all *norm.weight). nn.RMSNorm
        # here is standard, so apply the +1 to every *.norm.weight on load. The synthetic test never loads
        # a checkpoint, so this is a no-op there.
        if self.args.use_gemma_norm:
            weights = {k: (v + 1.0 if k.endswith("norm.weight") else v) for k, v in weights.items()}

        # HF M3 MoE layout (block_sparse_moe.*, per-expert w1=gate / w2=down / w3=up; key names + the
        # w1/w2/w3 convention confirmed vs the authoritative vLLM loader vllm/models/minimax_m3/*/model.py:
        # 704 block_sparse_moe naming, 817-822 ckpt w1=gate/w2=down/w3=up, 273 e_score_correction_bias) ->
        # the MLX layout: mlp.* with the routed experts STACKED into SwitchGLU switch_mlp.{gate,up,down}_proj
        # ([n_expert, out, in], cf. mlx-lm minimax.py). The MTP head (model.mtp.layers.*) is loaded by the
        # SEPARATE drafter, so drop it here. The cross-check's cc_mlx.npz already uses MLX names, so NONE of
        # these patterns fire there (non-disruptive). FP8 dequant stays weight-gated (added when applicable).
        import re
        exp_re = re.compile(r"^(model\.layers\.\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$")
        w2proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        experts: dict = {}
        out: dict = {}
        for k, v in weights.items():
            if k.startswith("model.mtp.") or ".mtp." in k:
                continue  # NextN/MTP draft head -> handled by the MTP drafter, not the main model
            m = exp_re.match(k)
            if m is not None:
                experts.setdefault((m.group(1), w2proj[m.group(3)]), {})[int(m.group(2))] = v
                continue
            if ".block_sparse_moe." in k:
                k = (k.replace(".block_sparse_moe.gate.e_score_correction_bias",
                               ".mlp.e_score_correction_bias")
                      .replace(".block_sparse_moe.", ".mlp."))
            out[k] = v
        for (prefix, proj), xs in experts.items():
            out[f"{prefix}.mlp.switch_mlp.{proj}.weight"] = mx.stack(
                [xs[i] for i in range(len(xs))], axis=0)
        return out
