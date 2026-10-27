"""Numpy oracle for the non-trivial DeepSeek-V4-Flash / DSpark math.

Faithful fp32 ports of the official reference implementation
(``DeepSeek-V4-Flash-DSpark/inference/model.py`` + ``kernel.py``) and of the
DeepSpec DSpark reference (``deepspec/modeling/dspark``): expert routing
(sqrtsoftplus + noaux_tc, hash tid2eid), limited SwiGLU, hyper-connection
sinkhorn split / pre / post / head, the compress-ratio KV pooling ladder
(simple + overlapping windows), YaRN frequency scaling, and the DSpark
markov / confidence / draft-block-layout math.

The MLX modules are tested against these oracles on the Mac
(``test_structural_mlx.py``); the oracles themselves are validated on any
box with numpy by ``test_reference_numpy.py`` (hand-computed cases +
formulation-equivalence proofs).

numpy only — no torch, no mlx.
"""

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Routing (MoEGate)
# ---------------------------------------------------------------------------

def score_activation(logits: np.ndarray, func: str) -> np.ndarray:
    logits = logits.astype(np.float64)
    if func == "softmax":
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    if func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-logits))
    if func == "sqrtsoftplus":
        return np.sqrt(np.log1p(np.exp(logits)))
    raise ValueError(func)


def noaux_tc_select(
    logits: np.ndarray,
    bias: np.ndarray,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[np.ndarray, np.ndarray]:
    """noaux_tc routing: the correction bias shifts scores for SELECTION only;
    routing WEIGHTS are gathered from the unbiased scores. Non-softmax scores
    are renormalized over the selected experts, then scaled."""
    scores = score_activation(logits, scoring_func)
    biased = scores + bias.astype(np.float64)
    inds = np.argsort(-biased, axis=-1, kind="stable")[..., :top_k]
    weights = np.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


def hash_select(
    logits: np.ndarray,
    tid2eid: np.ndarray,
    input_ids: np.ndarray,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[np.ndarray, np.ndarray]:
    """Hash routing (first num_hash_layers): expert indices come from the
    per-token-id lookup table; weights still come from the learned scores."""
    scores = score_activation(logits, scoring_func)
    inds = tid2eid[input_ids]
    weights = np.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


# ---------------------------------------------------------------------------
# Limited SwiGLU
# ---------------------------------------------------------------------------

def limited_swiglu(gate: np.ndarray, up: np.ndarray, limit: float) -> np.ndarray:
    gate = gate.astype(np.float64)
    up = up.astype(np.float64)
    if limit and limit > 0:
        gate = np.minimum(gate, limit)      # gate clamped from above only
        up = np.clip(up, -limit, limit)     # up clamped both sides
    silu = gate / (1.0 + np.exp(-gate))
    return silu * up


# ---------------------------------------------------------------------------
# Hyper-connections (sinkhorn split / pre / post / head)
# ---------------------------------------------------------------------------

def hc_split_sinkhorn(
    mixes: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the mix vector into (pre, post, comb); comb is softmaxed then
    sinkhorn-normalized toward doubly-stochastic. Mirrors kernel.py
    hc_split_sinkhorn / mlx _hc_split_sinkhorn_ops exactly."""
    mixes = mixes.astype(np.float64)
    scale = scale.astype(np.float64)
    base = base.astype(np.float64)

    pre = _sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2.0 * _sigmoid(mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])
    comb = mixes[..., 2 * hc_mult :].reshape(*mixes.shape[:-1], hc_mult, hc_mult) * scale[2]
    comb = comb + base[2 * hc_mult :].reshape(hc_mult, hc_mult)

    m = comb.max(-1, keepdims=True)
    e = np.exp(comb - m)
    comb = e / e.sum(-1, keepdims=True) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    return pre, post, comb


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)


def hc_pre(
    x: np.ndarray,
    fn: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x: [..., hc, d] -> collapsed [..., d], plus (post, comb) for hc_post."""
    x = x.astype(np.float64)
    flat = x.reshape(*x.shape[:-2], -1)
    mixes = _rms(flat, norm_eps) @ fn.astype(np.float64).T
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
    y = (pre[..., None] * x).sum(-2)
    return y, post, comb


def hc_post(x: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    """Expand [..., d] back to [..., hc, d]: post-weighted broadcast of the
    sublayer output plus comb^T-mixed residual streams."""
    x = x.astype(np.float64)
    residual = residual.astype(np.float64)
    y = post[..., None] * x[..., None, :]
    y = y + np.swapaxes(comb, -1, -2) @ residual
    return y


def hc_head_collapse(
    x: np.ndarray,
    fn: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
    eps: float,
    norm_eps: float,
) -> np.ndarray:
    """Final collapse [..., hc, d] -> [..., d] (model tail and DSpark tail)."""
    x = x.astype(np.float64)
    flat = x.reshape(*x.shape[:-2], -1)
    mixes = _rms(flat, norm_eps) @ fn.astype(np.float64).T
    pre = _sigmoid(mixes * scale.reshape(-1)[0] + base) + eps
    return (pre[..., None] * x).sum(-2)


def mean_lane_capture(h: np.ndarray) -> np.ndarray:
    """DSpark context feature per target layer: mean over hyper-connection
    lanes ([..., hc, d] -> [..., d]); reference Transformer.forward:
    ``main_hiddens.append(h.mean(dim=2))``."""
    return h.astype(np.float64).mean(-2)


# ---------------------------------------------------------------------------
# Compress-ratio KV pooling (prefill formulation)
# ---------------------------------------------------------------------------

def compress_prefill_simple(kv: np.ndarray, gate: np.ndarray, ape: np.ndarray, ratio: int) -> np.ndarray:
    """ratio-128 (HCA-style) pooling: softmax(gate + ape) over each window.
    kv/gate: [T, d] with T a multiple of ratio; ape: [ratio, d]. -> [T/r, d]."""
    T, d = kv.shape
    kv = kv.reshape(T // ratio, ratio, d).astype(np.float64)
    gate = gate.reshape(T // ratio, ratio, d).astype(np.float64) + ape.astype(np.float64)
    w = np.exp(gate - gate.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    return (kv * w).sum(1)


def compress_prefill_overlap(kv: np.ndarray, gate: np.ndarray, ape: np.ndarray, ratio: int) -> np.ndarray:
    """ratio-4 (CSA-style) OVERLAPPING pooling, mlx `_overlap_compress_kv`
    formulation: kv/gate [T, 2d]; the first-half dims of the PREVIOUS window
    join the second-half dims of the current window (2*ratio rows softmaxed
    together; missing previous window enters as -inf gate). -> [T/r, d]."""
    T, dd = kv.shape
    d = dd // 2
    n = T // ratio
    kv = kv.reshape(n, ratio, dd).astype(np.float64)
    gate = gate.reshape(n, ratio, dd).astype(np.float64) + ape.astype(np.float64)

    kv_a, kv_b = kv[..., :d], kv[..., d:]
    g_a, g_b = gate[..., :d], gate[..., d:]
    kv_prev = np.concatenate([np.zeros((1, ratio, d)), kv_a[:-1]], axis=0)
    g_prev = np.concatenate([np.full((1, ratio, d), -np.inf), g_a[:-1]], axis=0)

    kv_cat = np.concatenate([kv_prev, kv_b], axis=1)   # [n, 2r, d]
    g_cat = np.concatenate([g_prev, g_b], axis=1)
    w = np.exp(g_cat - g_cat.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    return (kv_cat * w).sum(1)


def compress_prefill_overlap_torch_style(
    kv: np.ndarray, gate: np.ndarray, ape: np.ndarray, ratio: int
) -> np.ndarray:
    """The official inference/model.py `overlap_transform` formulation of the
    SAME ratio-4 pooling (rows :ratio <- previous window's first-half dims,
    rows ratio: <- current window's second-half dims). Used to prove the two
    formulations equivalent in test_reference_numpy.py."""
    T, dd = kv.shape
    d = dd // 2
    n = T // ratio
    kv = kv.reshape(n, ratio, dd).astype(np.float64)
    gate = gate.reshape(n, ratio, dd).astype(np.float64) + ape.astype(np.float64)

    new_kv = np.zeros((n, 2 * ratio, d))
    new_g = np.full((n, 2 * ratio, d), -np.inf)
    new_kv[:, ratio:] = kv[..., d:]
    new_g[:, ratio:] = gate[..., d:]
    new_kv[1:, :ratio] = kv[:-1, :, :d]
    new_g[1:, :ratio] = gate[:-1, :, :d]

    w = np.exp(new_g - new_g.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    return (new_kv * w).sum(1)


# ---------------------------------------------------------------------------
# YaRN
# ---------------------------------------------------------------------------

def yarn_inv_freqs(
    dims: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> np.ndarray:
    """Interpolated inverse frequencies, mirroring precompute_freqs_cis /
    DeepseekV4RoPE. Returns inv_freq[dims//2]."""
    import math

    inv_freq = 1.0 / (base ** (np.arange(0, dims, 2, dtype=np.float64) / dims))

    def correction_dim(num_rotations):
        return (
            dims
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
    if low == high:
        high += 0.001
    ramp = (np.arange(dims // 2, dtype=np.float64) - low) / (high - low)
    smooth = 1.0 - np.clip(ramp, 0.0, 1.0)
    return inv_freq / factor * (1.0 - smooth) + inv_freq * smooth


# ---------------------------------------------------------------------------
# DSpark: markov head, confidence head, draft-block layout, walk
# ---------------------------------------------------------------------------

def markov_bias(w1: np.ndarray, w2: np.ndarray, prev_ids: np.ndarray) -> np.ndarray:
    """First-order transition bias B(x_prev, .) = W1[x_prev] @ W2^T.
    w1: [V, r] embedding table; w2: [V, r] (Linear(r->V) weight)."""
    emb = w1[prev_ids].astype(np.float64)
    return emb @ w2.astype(np.float64).T


def markov_sample_block(
    base_logits: np.ndarray,
    anchor_id: int,
    w1: np.ndarray,
    w2: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """Greedy semi-autoregressive block sampling: position k adds the markov
    bias of the previously sampled token (x_0 = anchor), then argmax.
    base_logits: [block, V]. Returns (tokens, corrected_logits)."""
    tokens: List[int] = []
    corrected = np.empty_like(base_logits, dtype=np.float64)
    prev = int(anchor_id)
    for k in range(base_logits.shape[0]):
        step = base_logits[k].astype(np.float64) + markov_bias(w1, w2, np.array(prev))
        corrected[k] = step
        prev = int(step.argmax())
        tokens.append(prev)
    return tokens, corrected


def confidence_scores(
    hidden: np.ndarray,
    prev_ids: np.ndarray,
    w1: np.ndarray,
    proj: np.ndarray,
) -> np.ndarray:
    """c_k = sigmoid(w . [h_k ; W1[x_{k-1}]]). hidden: [block, d] (the
    hc-head-collapsed, PRE-final-norm draft hidden); proj: [1, d + r]."""
    feats = np.concatenate([hidden.astype(np.float64), w1[prev_ids].astype(np.float64)], axis=-1)
    return _sigmoid(feats @ proj.astype(np.float64).reshape(-1))


def confident_prefix_length(scores: np.ndarray, threshold: float) -> int:
    """Static-threshold truncation (official eval draft_ops): keep the prefix
    up to (excluding) the first position whose confidence < threshold."""
    if threshold <= 0.0:
        return int(scores.shape[0])
    below = scores < threshold
    if not below.any():
        return int(scores.shape[0])
    return int(np.argmax(below))


def cumulative_survival(scores: np.ndarray) -> np.ndarray:
    """Prefix survival probabilities a_j = prod_{i<=j} c_i (paper Sec 3.2)."""
    return np.cumprod(scores.astype(np.float64))


def dspark_draft_input_ids(anchor_id: int, noise_token_id: int, block_size: int) -> List[int]:
    """gamma inputs = anchor + (gamma-1) noise tokens -> gamma draft logits
    (the anchor is the first prediction position)."""
    return [int(anchor_id)] + [int(noise_token_id)] * (block_size - 1)


def dspark_draft_positions(ctx_len: int, block_size: int) -> List[int]:
    """Draft block occupies absolute positions [ctx_len, ctx_len+block)."""
    return list(range(ctx_len, ctx_len + block_size))


def speculative_walk_greedy(draft_tokens: List[int], target_tokens: List[int]) -> Tuple[int, List[int]]:
    """Exact-greedy walk: accept drafts until first mismatch with the target's
    choice, then append the target's token at that position (bonus)."""
    accepted = len(draft_tokens)
    for i, (d, t) in enumerate(zip(draft_tokens, target_tokens)):
        if d != t:
            accepted = i
            break
    return accepted, list(draft_tokens[:accepted]) + list(target_tokens[accepted : accepted + 1])
