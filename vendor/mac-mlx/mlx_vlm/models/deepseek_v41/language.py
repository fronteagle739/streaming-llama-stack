"""DeepSeek-V4.1 Flash text stack for mlx-vlm.

Ported from PipeNetwork/deepseek-v41-mlx (Apache-2.0), commit 8bdd543.
Sources: attention.py, compressor.py, indexer.py, sparse_attention.py,
fakequant.py, layers.py, moe.py, hyper_connections.py, cache.py and model.py.
DeepSeek's reference model.py/kernel.py define the numerical semantics.

One conversation owns one growing, exactly trimmable cache; no decode state is
stored on modules. Window KV retains the activation dtype (bf16 for the release,
fp32 for parity). Compressor projections are retained in fp32 so trimming can
rebuild the pooled tail and recover an unfinished group. SharedState is fresh
per forward, and index-key owners always read their own keys. Hyper-connections
use staggered pre-mixes, with an initial one-hot collapse and no hyper-head.
QAT is enabled by default; DSV41_QAT=0 disables activation fake quantization.
All RoPE paths use V4's adjacent-pair YaRN implementation; window-only layers
omit YaRN. Text-only, batch one; masks and larger batches fail explicitly.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import LanguageModelOutput
from ..deepseek_v4.language import DeepseekV4RoPE, _limited_swiglu
from ..deepseek_v4.hyper_connection import _hc_split_sinkhorn_ops
from .config import TextConfig
from .checkpoint_layout import sanitize_weights_v41, make_quantization_config_v41
from .engram import (
    Engram, EngramLayout, EngramHashCache, NgramHashState,
    load_compressed_token_map,
    LEDGER as _ENGRAM_LEDGER,
)
from .prefill_ledger import (
    _PREFILL_LEDGER_ON, _PREFILL_LEDGER_CUTS, _PREFILL_LEDGER_CUTS_LEVEL,
    PREFILL_LEDGER, PREFILL_LAYER_MS,
    _prefill_ledger_add, _prefill_ledger_chunk, _prefill_layer_add,
)
from .prefill_linear import _PREFILL_MXFP8_STEEL, prefill_linear
from ...custom_kernels.deepseek_v41.v41_dispatch import (
    enabled as _v41_prefill_enabled, verification as _v41_verification,
    _VERIFY as _v41_verify_context,
)

# FFN experiments latch on backbone MoEs at attach (never DSpark stages).
_FFN_DOWN_BF16_ENGAGED = False
_FFN_EXPERT_PAD16_ENGAGED = False

__all__ = ["LanguageModel", "DeepseekV41Model", "DeepseekV41Block",
           "DeepseekV41Cache", "SharedState"]

DSV41_CACHE_STEP = 2048

FP8_MAX = 448.0     # e4m3fn

import time

# Phase 0 of the v41 port (2026-09-15): the step ledger. DSV41_STEP_LEDGER=1 times the n-gram hashing on
# the host; DSV41_STEP_LEDGER_CUTS=1 additionally evaluates after every layer of a decode step and records
# per-layer GPU ms (attribution only: it serialises the N+1 pipeline).
_LEDGER_ON = os.environ.get("DSV41_STEP_LEDGER", "0") == "1"
_LEDGER_CUTS = _LEDGER_ON and os.environ.get("DSV41_STEP_LEDGER_CUTS", "0") == "1"
# v41 port lever 1 (2026-09-15): intra-step overlap for decode steps. The ledger showed the DSV4.1
# step fully serial -- the host builds all 40 layers (with two 3-6 ms Engram memmap fetches inline)
# and only then does the GPU start. With DSV41_STEP_OVERLAP=1: the rows of every Engram layer after
# the first are fetched on a worker thread as soon as the step's hashes exist, and an async_eval is
# issued right before the second Engram layer so the GPU runs layers 0..13 while the host finishes
# the fetch and builds 14..39. Numerics are untouched (the same ops in the same order).
_STEP_OVERLAP = os.environ.get("DSV41_STEP_OVERLAP", "0") == "1"
# Prefill too (2026-09-15 evening): a 512-token chunk fetches 12,288 Engram rows per Engram layer on the
# host; with this switch layer 14's fetch runs on the worker while the GPU has layers 0..13.
_STEP_OVERLAP_PREFILL = _STEP_OVERLAP and os.environ.get("DSV41_STEP_OVERLAP_PREFILL", "0") == "1"
# v41 port lever 2a (2026-09-15): the promoted fp32 decode weights are half of what a token reads
# (wo_a 134 MB x 40 layers + lm_head 2.65 GB of ~16 GB). DSV41_WO_A_BF16=1 contracts the attention
# output in bf16 (fp32 accumulation inside the GEMM; wo_b consumes bf16 anyway) from a bf16 weight;
# DSV41_HEAD_BF16=1 keeps the head bf16 and needs DSV41_OMLX_HEAD=1 for fp32 logits (checked at
# attach time, loudly). Both off: byte-for-byte the previous path.
_WO_A_BF16 = os.environ.get("DSV41_WO_A_BF16", "0") == "1"
_HEAD_BF16 = os.environ.get("DSV41_HEAD_BF16", "0") == "1"
_PREFETCH_POOL = None


def _prefetch_pool():
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _PREFETCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsv41-engram-prefetch")
    return _PREFETCH_POOL
LEDGER = {"hash_host_ms": 0.0, "hash_calls": 0, "cut_steps": 0}
LEDGER_LAYER_MS: dict = {}
FP4_MAX = 6.0       # e2m1

# Test hook: parity tests compare the continuous math with the QAT simulation
# switched off on BOTH sides (quantization grids amplify ~1e-7 implementation
# noise to one full code step whenever a value lands on a rounding midpoint,
# so the fake-quant paths are held to a looser threshold separately).

FP8_MAX_INV = mx.array(1.0 / 448.0, dtype=mx.float32)
FP4_MAX_INV = mx.array(1.0 / 6.0, dtype=mx.float32)

_E2M1_LUT = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)
# comparison per midpoint chosen so exact ties land on the even-mantissa code
_E2M1_GT = mx.array([0.25, 1.25, 2.5, 5.0], dtype=mx.float32)     # tie rounds DOWN
_E2M1_GE = mx.array([0.75, 1.75, 3.5], dtype=mx.float32)          # tie rounds UP


def _log2_ceil_bits(x: mx.array) -> mx.array:
    """ceil(log2(x)) for positive normal fp32 x, via IEEE 754 bits (fast_log2_ceil)."""
    bits = x.astype(mx.float32).view(mx.uint32)
    exp = ((bits >> 23) & 0xFF).astype(mx.int32) - 127
    man = (bits & 0x7FFFFF).astype(mx.int32)
    return exp + (man != 0).astype(mx.int32)


def _pow2(e: mx.array) -> mx.array:
    """2**e for integer e, via IEEE 754 bits (fast_pow2)."""
    return ((e + 127) << 23).astype(mx.uint32).view(mx.float32)


def _ue8m0_scale(amax: mx.array, max_inv: mx.array) -> mx.array:
    """The kernel's fast_round_scale: 2**ceil(log2(amax * (1/max)))."""
    return _pow2(_log2_ceil_bits(amax * max_inv))


def _e2m1_round(v: mx.array) -> mx.array:
    """Round |v| <= 6 to the nearest e2m1 value, ties to even, keeping the sign."""
    mag = mx.abs(v)
    # Binary search the seven midpoints. The first two levels round ties UP;
    # all four final-level midpoints round ties DOWN (even mantissa codes).
    upper = mag >= 1.75
    middle = mag >= mx.where(upper, 3.5, 0.75)
    last = mx.where(upper, mx.where(middle, 5.0, 2.5),
                    mx.where(middle, 1.25, 0.25))
    idx = (4 * upper.astype(mx.int32) + 2 * middle.astype(mx.int32)
           + (mag > last).astype(mx.int32))
    return mx.sign(v) * _E2M1_LUT[idx]


def _blockify(x: mx.array, block: int):
    n = x.shape[-1]
    if n % block:
        raise ValueError(f"last dim {n} not divisible by block {block}")
    return x.reshape(*x.shape[:-1], n // block, block)


def fake_quant_fp8_ue8m0(x: mx.array, block: int = 32) -> mx.array:
    """act_quant(..., scale_fmt='ue8m0', inplace=True): fp8 round-trip, pow2 scales."""
    if os.environ.get("DSV41_QAT", "1") == "0":
        return x
    if os.environ.get("DSV41_METAL_QUANT", "0") == "1":
        from .quant_kernels import fake_quant
        return fake_quant(x, block, "fp8_ue8m0")
    dtype = x.dtype
    shape = x.shape
    xb = _blockify(x.astype(mx.float32), block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 1e-4)
    s = _ue8m0_scale(amax, FP8_MAX_INV)
    q = mx.from_fp8(mx.to_fp8(mx.clip(xb / s, -FP8_MAX, FP8_MAX)), mx.float32) * s
    return q.reshape(shape).astype(dtype)


def fake_quant_fp4_ue8m0(x: mx.array, block: int = 32) -> mx.array:
    """fp4_act_quant(..., scale_dtype=e8m0, inplace=True): the indexer's q/k path."""
    if os.environ.get("DSV41_QAT", "1") == "0":
        return x
    if os.environ.get("DSV41_METAL_QUANT", "0") == "1":
        from .quant_kernels import fake_quant
        return fake_quant(x, block, "fp4_ue8m0")
    dtype = x.dtype
    shape = x.shape
    xb = _blockify(x.astype(mx.float32), block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 6.0 * 2.0 ** -126)
    s = _ue8m0_scale(amax, FP4_MAX_INV)
    q = _e2m1_round(mx.clip(xb / s, -FP4_MAX, FP4_MAX)) * s
    return q.reshape(shape).astype(dtype)


def fake_quant_fp4_e4m3(x: mx.array, block: int = 16) -> mx.array:
    """fp4_act_quant(..., 16, scale_dtype=e4m3, inplace=True): the compressed-KV path."""
    if os.environ.get("DSV41_QAT", "1") == "0":
        return x
    if os.environ.get("DSV41_METAL_QUANT", "0") == "1":
        from .quant_kernels import fake_quant
        return fake_quant(x, block, "fp4_e4m3")
    dtype = x.dtype
    shape = x.shape
    xb = _blockify(x.astype(mx.float32), block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 6.0 * 2.0 ** -9)
    s = mx.from_fp8(mx.to_fp8(amax / FP4_MAX), mx.float32)
    q = _e2m1_round(mx.clip(xb / s, -FP4_MAX, FP4_MAX)) * s
    return q.reshape(shape).astype(dtype)


@partial(mx.compile, shapeless=True)
def _v41_rms_norm(x, weight, eps):
    xf = x.astype(mx.float32)
    var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return (weight * xf).astype(x.dtype)


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, returning the input dtype (matches reference)."""

    def __init__(self, dim: int, eps: float = 1e-20):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # Opt-in until the compiled/Metal reductions pass hardware parity. Pass
        # fp32 to fast RMSNorm: bf16 inputs AND weights otherwise round the
        # normalized activation before multiplying the learned weight.
        if os.environ.get("DSV41_FAST_RMS_NORM", "0") == "1":
            return mx.fast.rms_norm(
                x.astype(mx.float32), self.weight.astype(mx.float32), self.eps
            ).astype(x.dtype)
        if os.environ.get("DSV41_COMPILE_PRIMITIVES", "0") == "1":
            return _v41_rms_norm(x, self.weight, self.eps)
        dtype = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(var + self.eps)
        return (self.weight * xf).astype(dtype)



def _gather_kv(kv: mx.array, idx: mx.array) -> mx.array:
    """kv [b, n, d], idx [b, m, k] -> [b, m, k, d]; row 0 for negative idx."""
    if isinstance(kv, tuple):
        window, pooled = kv
        boundary = window.shape[1]
        win = _gather_kv(window, mx.minimum(idx, boundary - 1))
        comp = _gather_kv(pooled, idx - boundary)
        return mx.where((idx >= boundary)[..., None], comp, win)
    b, n, d = kv.shape
    flat = kv.reshape(b * n, d)
    safe = mx.maximum(idx, 0).astype(mx.int32)
    base = (mx.arange(b, dtype=mx.int32) * n).reshape(b, 1, 1)
    return flat[(safe + base).reshape(-1)].reshape(*idx.shape, d)


def _sparse_attn_chunk(q, kv, sink, ic, softmax_scale, *, sdpa=False):
    """Reference fp32 arithmetic, or opt-in gathered SDPA with fp32 sinks."""
    if sdpa:
        b, c, h, d = q.shape
        # Every query has its OWN selected K/V rows. Fold queries into batch,
        # retaining one KV head; broadcasting K/V over heads would waste memory.
        rows = _gather_kv(kv, ic).astype(mx.float32).reshape(b * c, 1, ic.shape[-1], d)
        queries = q.astype(mx.float32).reshape(b * c, h, 1, d)
        # Selection already enforces window visibility and compressed
        # group < floor((position+1)/ratio). Only -1 padding remains to mask;
        # a causal mask here would incorrectly interpret selected-slot order.
        valid = (ic >= 0).reshape(b * c, 1, 1, ic.shape[-1])
        out = mx.fast.scaled_dot_product_attention(
            queries, rows, rows, scale=softmax_scale, mask=valid,
            # MLX requires sinks to promote to the Q/K/V output dtype. Keep
            # fp32 Q/K/V: rounding learned sinks to bf16 can change norms by 14%.
            sinks=sink.reshape(h).astype(mx.float32))
        return out.reshape(b, c, h, d).astype(q.dtype)

    qc = q.astype(mx.float32)
    kvc = _gather_kv(kv, ic).astype(mx.float32)
    logits = mx.einsum("bchd,bckd->bchk", qc, kvc) * softmax_scale
    valid = (ic >= 0)[:, :, None, :]
    logits = mx.where(valid, logits, -1e30)
    mmax = mx.max(logits, axis=-1, keepdims=True)
    mmax = mx.maximum(mx.maximum(mmax, sink), -1e30)
    w = mx.exp(logits - mmax)
    w = mx.where(valid, w, 0.0)
    denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sink - mmax)
    o = mx.einsum("bchk,bckd->bchd", w, kvc) / denom
    return o.astype(q.dtype)


def sparse_attn(q: mx.array, kv: mx.array, attn_sink: mx.array, topk_idxs: mx.array,
                softmax_scale: float, chunk: int = 256) -> mx.array:
    """q [b,m,h,d], kv [b,n,d], attn_sink [h], topk_idxs [b,m,k] (-1 = masked).

    GLM's sparse MLA kernel has no sink and separate 512+64 key / 512 value
    dimensions; V4.1 uses the same 512 channels for keys AND values. It cannot
    replace this operation. DSV41_METAL_SPARSE_ATTN=1 enables V4.1's own
    experimental kernel; it fails loudly on unsupported inputs or Metal errors.
    DSV41_SDPA_SPARSE_ATTN=1 enables gathered SDPA with sinks (MLX 0.32.2).
    IMPORTANT: MLX 0.32.2 falls back to unfused ops at D=512. Accepting sinks
    does not imply fusion; validate_sdpa.py probes force_fused=True explicitly.
    Both flags also accept 'decode' or 'prefill' to restrict their regime;
    SDPA takes precedence over Metal in that regime. Both are off by default.
    The default retains O1's pure-MLX path, including its optional compilation.
    DSV41_PREFILL_FLASH_ATTN=1 independently selects shared BF16 gathering plus
    online MMA prefill at n>=DSV41_PREFILL_FLASH_MIN_N (default 64). Batches <=8
    always retain their existing route. The master switch defaults off.
    DSV41_PREFILL_V41_ATTN=1 selects the imported fp16-staged rb16 kernel
    at N>=64, independently of other switches; DSpark verify is excluded.
    """
    if _v41_prefill_enabled("ATTN", q.shape[1]):
        from ...custom_kernels.deepseek_v41.v41_attention import sparse_attn_v41

        return sparse_attn_v41(q, kv, attn_sink, topk_idxs, softmax_scale)
    if os.environ.get("DSV41_PREFILL_FLASH_ATTN", "0") == "1":
        from ...custom_kernels.deepseek_v41.sparse_attention import (
            prefill_flash_enabled, sparse_attention,
        )

        if prefill_flash_enabled(q.shape[1]):
            return sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)
    sdpa = os.environ.get("DSV41_SDPA_SPARSE_ATTN", "0")
    if sdpa not in ("0", "1", "decode", "prefill"):
        raise ValueError("DSV41_SDPA_SPARSE_ATTN must be 0, 1, decode, or prefill")
    metal = os.environ.get("DSV41_METAL_SPARSE_ATTN", "0")
    if metal not in ("0", "1", "decode", "prefill"):
        raise ValueError("DSV41_METAL_SPARSE_ATTN must be 0, 1, decode, or prefill")
    regime = "decode" if q.shape[1] == 1 else "prefill"
    use_sdpa = sdpa in ("1", regime)
    if not use_sdpa and metal in ("1", regime):
        from ...custom_kernels.deepseek_v41.sparse_attention import sparse_attention
        # Bound the tiled fp32 score/weight workspace just as the MLX path bounds
        # its gathered KV. A long prefill must not allocate [B,M,H,K] all at once.
        outs = [sparse_attention(q[:, start:start + chunk], kv, attn_sink,
                                 topk_idxs[:, start:start + chunk], softmax_scale)
                for start in range(0, q.shape[1], chunk)]
        return mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]

    b, m, h, d = q.shape
    sink = attn_sink.astype(mx.float32).reshape(1, 1, h, 1)

    # Convert reused source rows once, instead of every occurrence in [b,c,k,d].
    # Half -> float is exact. Keep decode/short-query gathers local: converting a
    # long cache for one query would increase both traffic and temporary storage.
    sources = kv if isinstance(kv, tuple) else (kv,)
    if m > chunk and m * topk_idxs.shape[-1] > sum(s.shape[1] for s in sources):
        sources = tuple(s.astype(mx.float32) for s in sources)
        kv = sources if isinstance(kv, tuple) else sources[0]

    attend = partial(_sparse_attn_chunk, sdpa=True) if use_sdpa else _sparse_attn_chunk
    if not use_sdpa and os.environ.get("DSV41_COMPILE_SPARSE_ATTN", "0") == "1":
        # Reuse the callable across chunks AND layers; MLX caches by input shape.
        if not hasattr(sparse_attn, "_compiled_chunk"):
            sparse_attn._compiled_chunk = mx.compile(_sparse_attn_chunk)
        attend = sparse_attn._compiled_chunk

    outs = []
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        outs.append(attend(q[:, start:stop], kv, sink,
                           topk_idxs[:, start:stop], softmax_scale))

    return mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]


NEG_INF = float("-inf")
POS_INF = float("inf")


_SPARSE_TABLES = {}
_S4_COUNTS = {}
_S4_COUNT_ENABLED = os.environ.get("DSV41_S4_COUNT", "0") == "1"


def _s4_count(name, layer=None, amount=1):
    if _S4_COUNT_ENABLED:
        key = name if layer is None else f"{name}.{layer}"
        _S4_COUNTS[key] = _S4_COUNTS.get(key, 0) + amount


def _sparse_position_tables(end_pos, ratio):
    """Immutable integer geometry, grown geometrically; no conversation state.

    Retain only the largest table for each compression ratio. Slices remain
    valid across growth, cache trim, interleaved conversations and restoration.
    """
    tables = _SPARSE_TABLES.get(ratio)
    if tables is None or tables[0].shape[0] < end_pos:
        _s4_count("table_build")
        capacity = max(2048, 1 << (max(end_pos, 1) - 1).bit_length())
        positions = np.arange(capacity, dtype=np.int32)
        tables = (mx.array(positions), mx.array((positions + 1) // ratio))
        _SPARSE_TABLES[ratio] = tables
    return tables


def _index_geometry(shared, start_pos, n, nb, ratio):
    """Share prefill visibility across index sources; decode needs no mask."""
    if not hasattr(shared, "index_geometry"):
        shared.index_geometry = {}
    key = (start_pos, n, nb, ratio)
    if key not in shared.index_geometry:
        _s4_count("geometry_build")
        positions, lengths = _sparse_position_tables(start_pos + n, ratio)
        lens = lengths[start_pos:start_pos + n, None]
        vis = None if n == 1 else positions[None, :nb] < lens
        shared.index_geometry[key] = (lens, vis)
    return shared.index_geometry[key]


def _index_scores(x, projection, q, index_k, scale, matmul=False):
    # Preserve the reference's fp32 projection and head reduction. Compilation
    # is opt-in because fused arithmetic can change ties in the subsequent top-k.
    w = x.astype(mx.float32) @ projection.astype(mx.float32).T
    w = w * scale
    if matmul:
        # einsum's Python lowering captures the key length in a reshape when
        # compiled shapelessly. Broadcast matmul keeps growing decode lengths
        # symbolic. Keep this alternative confined to the opt-in compiler path.
        scores = q.astype(mx.float32) @ index_k.astype(mx.float32)[:, None].swapaxes(-1, -2)
    else:
        scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32),
                           index_k.astype(mx.float32))
    return mx.sum(mx.maximum(scores, 0.0) * w[..., None], axis=2)


def _mask_candidate_blocks(scores, keep, block_size):
    """Broadcast one bool per block, never repeat it into a position mask."""
    width = scores.shape[-1]
    _s4_count("block_mask")
    pad = (-width) % block_size
    if pad:
        scores = mx.concatenate([scores, mx.full(
            (*scores.shape[:-1], pad), NEG_INF, dtype=scores.dtype)], axis=-1)
    shape = scores.shape
    scores = scores.reshape(*shape[:-1], -1, block_size)
    return mx.where(keep[..., None], scores, NEG_INF).reshape(shape)[..., :width]


def _index_skip_min_nb():
    """Smallest compressed-bank width nb at which DSV41_INDEX_SKIP pays.

    Measured on the Mac (whole-model prefill tok/s, fused indexer with the
    tile skip against the same kernel without it): 34,298 tokens 418.4 vs
    439.3 (-4.8%), 47,235 424.1 vs 430.2 (-1.4%), 68,639 416.2 vs 398.5
    (+4.4%), 97,932 395.1 vs 383.2 (+3.1%). The admitted width is capped at
    candidate_topk_blocks * candidate_block_size = 16,384 positions, so the
    skippable share of the 32-key tiles is about (1 - 16384/nb) ** 4 -- 0.18,
    0.33, 0.48 at the three largest sizes -- and below roughly 60,000 the
    per-tile vote and the mask reads cost more than the skipped tiles save.

    0 means "always skip when the switch is on", which is what the A/B
    measurement arms use. Anything that is not a non-negative integer is a
    loud failure here, never a silent fallback to the default.
    """
    raw = os.environ.get("DSV41_INDEX_SKIP_MIN_NB", "60000")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = -1
    if value < 0:
        raise ValueError("DSV41_INDEX_SKIP_MIN_NB must be a non-negative integer, "
                         f"got {raw!r}")
    return value


def _index_skip_decision(nb, layer_id):
    """Per call, never latched: nb grows as the prompt is prefilled and differs
    per layer (compression ratio 2 layers hold half the context).

    Returns (engage, min_nb). Prints the first time each outcome occurs -- once
    for the first call that engages skipping and once for the first that
    declines it -- because a single global line reports whichever outcome the
    first tiny chunk happened to take and then never speaks again.
    """
    min_nb = _index_skip_min_nb()
    engage = nb >= min_nb
    seen = _index_skip_decision.__dict__.setdefault("printed", set())
    if engage not in seen:
        seen.add(engage)
        print(f"[DSV41] INDEX_SKIP {'engaged' if engage else 'declined'} "
              f"layer={layer_id} nb={nb} min_nb={min_nb}", flush=True)
    return engage, min_nb


def _dense_decode_attn(q, kv, attn_sink, window_size, scale):
    """All visible compressed rows plus the window; no gather or index mask.

    Only valid for a single query after the second top-k retains every row.
    Opt-in pending hardware parity: contractions may use a different reduction.
    """
    window, compressed = kv
    _s4_count("dense_decode")
    rows = mx.concatenate([window[:, -window_size:], compressed], axis=1)
    rows = rows.astype(mx.float32)[:, None]
    logits = mx.einsum("bchd,bckd->bchk", q.astype(mx.float32), rows) * scale
    sink = attn_sink.astype(mx.float32).reshape(1, 1, -1, 1)
    mmax = mx.maximum(mx.maximum(mx.max(logits, axis=-1, keepdims=True), sink), -1e30)
    weights = mx.exp(logits - mmax)
    denom = mx.sum(weights, axis=-1, keepdims=True) + mx.exp(sink - mmax)
    return (mx.einsum("bchk,bckd->bchd", weights, rows) / denom).astype(q.dtype)


def select_candidate_blocks(scores: mx.array, lens: mx.array, topk_blocks: int,
                            block_size: int, block_mask: bool = False) -> mx.array:
    """Level one of the two-level top-k. scores [b, n, nb] with unreachable
    positions already at -inf; lens [n, 1] (positions visible per query).
    Returns a block mask when block_mask=True; legacy callers get positions.
    The indexer skips this operation entirely when every block is retained."""
    width = scores.shape[-1]
    _s4_count("candidate_select")
    pad = (-width) % block_size
    if pad:
        scores_p = mx.concatenate(
            [scores, mx.full((*scores.shape[:-1], pad), NEG_INF, dtype=scores.dtype)], axis=-1)
    else:
        scores_p = scores
    blocks = scores_p.reshape(*scores.shape[:-1], -1, block_size).max(axis=-1)  # [b, n, NB]
    nb_blocks = blocks.shape[-1]

    # pin the block holding each query's newest position: it is only partly
    # filled and could otherwise be outscored by an older, full block
    last = (lens - 1) // block_size                                # [n, 1]
    block_ids = (_sparse_position_tables(nb_blocks, 1)[0][:nb_blocks]
                 if block_mask else mx.arange(nb_blocks))
    pin = block_ids[None, :] == last                              # [n, NB]
    blocks = mx.where(pin[None], POS_INF, blocks)

    k = min(topk_blocks, nb_blocks)
    if k == nb_blocks:
        # Every block is retained: partitioning cannot change this boolean mask.
        keep = blocks > NEG_INF
        return keep if block_mask else mx.repeat(keep, block_size, axis=-1)[..., :width]
    top_idx = mx.argpartition(-blocks, k - 1, axis=-1)[..., :k]
    top_val = mx.take_along_axis(blocks, top_idx, axis=-1)
    keep = mx.zeros(blocks.shape, dtype=mx.bool_)
    keep = mx.put_along_axis(keep, top_idx, top_val > NEG_INF, axis=-1)
    return keep if block_mask else mx.repeat(keep, block_size, axis=-1)[..., :width]


def _shared_rows(value, published_start, start_pos):
    """Align a per-row SharedState tensor to the rows a layer is about to compute.

    `topk_idxs` and `candidates` are [b, n, ...]: one entry per QUERY row. On the
    default path every layer of a chunk runs the same rows, so `published_start`
    equals `start_pos` and this returns the tensor untouched. Under
    DSV41_PREFILL_CONE each decoder layer runs a strictly SHORTER, suffix-aligned
    row range than the layer that published the selection, so the reader takes the
    matching suffix. A reader that starts BEFORE the publisher, or past its end, is
    a wiring error and fails loudly rather than broadcasting the wrong rows.
    """
    if value is None or published_start is None or published_start == start_pos:
        return value
    offset = start_pos - published_start
    if offset < 0 or offset >= value.shape[1]:
        raise ValueError(
            f"DSV41_PREFILL_CONE: shared per-row tensor published at {published_start} "
            f"({value.shape[1]} rows) cannot serve the rows starting at {start_pos}")
    return value[:, offset:]


class Indexer(nn.Module):
    def __init__(self, args: TextConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layer_ids
        self.ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.uses_candidates = 0 <= args.candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.index_fused = False  # Published only after attach-time compilation.
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(args.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)

    def publish_keys(self, latents, start_pos, rope, cache):
        k = self.k_norm(self.wk(latents))
        k = _rope(rope, k, start_pos - start_pos % self.ratio)
        cache.append("index_k", fake_quant_fp4_ue8m0(k, 32))

    def __call__(self, x: mx.array, qr: mx.array, start_pos: int, offset: int,
                 rope, index_k: mx.array, shared) -> mx.array:
        """Score and pick top-k compressed positions for each query.

        x [b, n, dim] (post-attn-norm), qr [b, n, q_lora_rank],
        index_k [b, nb, head_dim] — the owner's key cache, already sliced to the
        nb = (start_pos+n)//ratio complete groups. Returns [b, n, k] int32
        indices into the concatenated window+compressed KV, -1 = masked.
        """
        bsz, n, _ = x.shape
        fused_index = (self.index_fused and n >= 64 and not _v41_verify_context.get())
        skip_enabled = fused_index and os.environ.get("DSV41_INDEX_SKIP", "0") == "1"
        if self.index_fused:
            for name in ("DSV41_PREFILL_TILED_INDEX", "DSV41_PREFILL_V41_INDEX",
                         "DSV41_METAL_INDEXER"):
                if os.environ.get(name, "0") == "1":
                    raise ValueError(f"DSV41_INDEX_FUSED cannot combine with {name}")
        _s4_count("index_call", self.layer_id)
        ratio = self.ratio
        nb = index_k.shape[1]
        precompute = os.environ.get("DSV41_S4_PRECOMPUTE", "1") == "1"
        # This is a compressed-position threshold, NOT an attention threshold:
        # even when all candidate blocks fit, index_topk still prunes positions.
        candidate_limit = self.candidate_topk_blocks * self.candidate_block_size
        all_candidates = nb <= candidate_limit
        if precompute:
            lens, vis = _index_geometry(shared, start_pos, n, nb, ratio)
        else:
            lens = ((start_pos + mx.arange(n) + 1) // ratio)[:, None]
            vis = mx.arange(nb)[None, :] < lens
        # Candidates are one row per QUERY. Identity on the default path; under
        # DSV41_PREFILL_CONE this layer runs a suffix of the source layer's rows.
        shared_candidates = _shared_rows(
            shared.candidates, getattr(shared, "candidates_start", None), start_pos)

        if (os.environ.get("DSV41_S4_BYPASS", "1") == "1"
                and nb <= self.index_topk and all_candidates
                and (not self.uses_candidates or shared_candidates is None)):
            # Sorted top-k of the whole row is the identity, including the
            # causal -1 lanes in prefill. Key publication still happens above us.
            if self.is_candidate_source:
                shared.candidates = None
                shared.candidates_start = start_pos
            _s4_count("index_bypass", self.layer_id)
            positions = _sparse_position_tables(max(nb, 1), 1)[0][:nb]
            idx = mx.broadcast_to(positions[None, None], (bsz, n, nb))
            return mx.where(idx < lens[None], idx + offset, mx.array(-1, mx.int32))

        q = (prefill_linear(self.wq_b, qr)
             if _PREFILL_MXFP8_STEEL and x.shape[1] > 1 else self.wq_b(qr)
             ).reshape(bsz, n, self.n_heads, self.head_dim)
        q = _rope(rope, q, start_pos)
        q = fake_quant_fp4_ue8m0(q, 32)

        # Prefill chunks only: DSpark's verify forward is n = 1 + drafts and the one-query-per-threadgroup
        # kernel is far slower than the eager path at tiny n (2026-09-15 arm: decode 20 -> 7-14 tok/s).
        v41_index = _v41_prefill_enabled("INDEX", n)
        if (not v41_index and n >= int(os.environ.get("DSV41_PREFILL_TILED_MIN_N", "64"))
                and os.environ.get("DSV41_PREFILL_TILED_INDEX", "0") == "1"):
            from ...custom_kernels.deepseek_v41.index_prefill import index_scores_topk

            _s4_count("index_score", self.layer_id)
            need_blocks = self.is_candidate_source and not (precompute and all_candidates)
            candidates = (shared_candidates if self.uses_candidates
                          and not self.is_candidate_source else None)
            result = index_scores_topk(
                x, self.weights_proj.weight, q, index_k, lens, offset,
                min(self.index_topk, nb),
                scale=self.softmax_scale * self.n_heads ** -0.5,
                visibility=vis, candidates=candidates,
                candidate_block_size=self.candidate_block_size,
                candidates_are_blocks=precompute, return_block_scores=need_blocks,
            )
            if self.is_candidate_source:
                if need_blocks:
                    result, block_scores = result
                    # Treat each emitted maximum as one position. Transform
                    # Reuse the reference's exact pin arithmetic, then add one
                    # for its block_size=1 consumer. MLX Metal integer division
                    # truncates (-1)//8 to 0 at lens=0; ceil(lens/8) loses that pin.
                    block_lens = (lens - 1) // self.candidate_block_size + 1
                    keep = select_candidate_blocks(
                        block_scores, block_lens, self.candidate_topk_blocks,
                        1, block_mask=True,
                    )
                    shared.candidates = (keep if precompute else
                        mx.repeat(keep, self.candidate_block_size, axis=-1)[..., :nb])
                else:
                    shared.candidates = None
                shared.candidates_start = start_pos
            return result

        # Keep the mainline FP32 scoring contract and bounded query workspace,
        # while sharing S4 geometry and allowing its optional compiled scorer.
        score_fn = _index_scores
        metal_indexer = os.environ.get("DSV41_METAL_INDEXER", "0") == "1"
        _s4_count("index_score", self.layer_id)
        if fused_index:
            from ...custom_kernels.deepseek_v41.index_fused import index_scores_fused
        elif v41_index:
            from ...custom_kernels.deepseek_v41.v41_index import (
                index_scores_v41, index_topk_v41,
            )
        elif metal_indexer:
            from ...custom_kernels.deepseek_v41.indexer import index_scores, index_topk
            score_fn = index_scores
        elif os.environ.get("DSV41_S4_COMPILE_INDEXER", "0") == "1":
            if not hasattr(_index_scores, "compiled"):
                _index_scores.compiled = mx.compile(partial(_index_scores, matmul=True), shapeless=True)
            score_fn = _index_scores.compiled
        workspace_mb = int(os.environ.get("DSV41_INDEX_WORKSPACE_MB", "0"))
        query_chunk = (max(1, workspace_mb * 1024**2 // (bsz * self.n_heads * nb * 4))
                       if workspace_mb > 0 else n)
        masked_by_kernel = False
        if fused_index:
            # The kernel writes NEG_INF for every unadmitted key and skips the
            # 32-key tiles no admitted block touches. Only the consumer layers
            # have a mask; the source layer builds it below, as before.
            keep = (shared_candidates if skip_enabled and self.uses_candidates
                    and not self.is_candidate_source else None)
            if keep is not None and not _index_skip_decision(nb, self.layer_id)[0]:
                # Below DSV41_INDEX_SKIP_MIN_NB too few tiles are skippable to
                # pay for the per-tile vote and the mask reads, so this call
                # takes the plain fused path with no mask input -- exactly as
                # if the switch were off -- and _mask_candidate_blocks runs
                # eagerly below. The decision is per call: the same layer
                # engages later in the prefill once its nb has grown.
                keep = None
            # Both candidate layouts reach the kernel. DSV41_S4_PRECOMPUTE=0
            # leaves the per-position form, which IS the block mask repeated
            # candidate_block_size times and trimmed to nb, so the kernel reads
            # it at stride candidate_block_size: the same four reads per row per
            # BN=32 tile as the block form. A mask matching neither layout is
            # still refused, loudly, by the wrapper's shape check.
            scores = index_scores_fused(
                x, self.weights_proj.weight, q, index_k,
                self.softmax_scale * self.n_heads ** -0.5,
                candidates=keep, block_size=self.candidate_block_size,
                block_mask=precompute,
            )
            # masked_by_kernel is the one variable that carries "the kernel
            # already applied the candidate mask for THIS call": it is true if
            # and only if a mask was handed to the kernel, so the eager
            # _mask_candidate_blocks below runs on every call that did not.
            masked_by_kernel = keep is not None
        elif v41_index:
            scores = index_scores_v41(
                x, self.weights_proj.weight, q, index_k,
                self.softmax_scale * self.n_heads ** -0.5, lens,
            )
        else:
            parts = [score_fn(x[:, begin:begin + query_chunk], self.weights_proj.weight,
                             q[:, begin:begin + query_chunk], index_k,
                             self.softmax_scale * self.n_heads ** -0.5)
                     for begin in range(0, n, query_chunk)]
            scores = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if vis is not None:
            scores = mx.where(vis[None], scores, NEG_INF)

        if self.is_candidate_source:
            shared.candidates = (None if precompute and all_candidates else
                select_candidate_blocks(scores, lens, self.candidate_topk_blocks,
                                        self.candidate_block_size, block_mask=precompute))
            shared.candidates_start = start_pos
        elif self.uses_candidates and shared_candidates is not None and not masked_by_kernel:
            # masked_by_kernel: re-masking would be idempotent but costs a full
            # read and write of the reduced [b, n, nb] tensor.
            scores = (_mask_candidate_blocks(scores, shared_candidates, self.candidate_block_size)
                      if precompute else mx.where(shared_candidates, scores, NEG_INF))

        k = min(self.index_topk, nb)
        if v41_index:
            return index_topk_v41(scores, k, lens, offset)
        if metal_indexer:
            return index_topk(scores, k, lens, offset)
        idx = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        idx = mx.sort(idx, axis=-1)                              # position order
        # Never turn an unreachable top-k lane into a real KV gather. Partition
        # has no padded lanes, but -inf candidates can tie with masked positions.
        visible = ((idx < lens.astype(mx.int32)[None])
                   & (mx.take_along_axis(scores, idx, axis=-1) > NEG_INF))
        return mx.where(visible, idx + offset, mx.array(-1, mx.int32))


class ClampedSwiGLU(nn.Module):
    """Called by SwitchGLU as activation(x_up, x_gate)."""

    def __init__(self, limit: float = 0.0):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        x = x.astype(mx.float32)
        gate = gate.astype(mx.float32)
        return _limited_swiglu(gate, x, self.limit)


@mx.compile
def _v41_expert_select(x, weight, bias, gate_temp, score_func, topk,
                       norm_topk_prob, route_scale):
    scores = (x.astype(mx.float32) @ weight.astype(mx.float32).T) / gate_temp
    if score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:
        scores = mx.sqrt(nn.softplus(scores))
    # V4 skips top-k normalization for softmax; V4.1 does not. Keep the
    # temperature, topk=1 exception, and unmodified scores for the weights.
    indices = mx.argpartition(-(scores + bias), topk - 1, axis=-1)[..., :topk]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if norm_topk_prob and topk > 1:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return weights * route_scale, indices


class Gate(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.topk = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.routed_scaling_factor
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)
        self.bias_vl = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array, image_mask=None):
        if os.environ.get("DSV41_GATE_FP32", "0") == "1" and self.weight.dtype != mx.float32:
            # Promote the parameter itself once, rather than keeping a second
            # cached array that becomes stale after load_weights/update/casting.
            # Inference opt-in: this changes stored parameter dtype/memory.
            self.weight = self.weight.astype(mx.float32)
        if os.environ.get("DSV41_COMPILE_PRIMITIVES", "0") == "1":
            bias = self.e_score_correction_bias
            if image_mask is not None:
                bias = mx.where(image_mask[..., None], self.bias_vl, bias)
            return _v41_expert_select(
                x, self.weight, bias, self.gate_temp, self.score_func, self.topk,
                self.norm_topk_prob, self.route_scale)
        scores = (x.astype(mx.float32) @ self.weight.astype(mx.float32).T) / self.gate_temp
        if self.score_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.score_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:  # sqrtsoftplus
            scores = mx.sqrt(nn.softplus(scores))

        # the bias picks experts but does not scale them
        bias = self.e_score_correction_bias
        if image_mask is not None:
            bias = mx.where(image_mask[..., None], self.bias_vl, bias)
        biased = scores + bias
        indices = mx.argpartition(-biased, self.topk - 1, axis=-1)[..., :self.topk]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class SharedExpert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.limit = limit

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        gate = (prefill_linear(self.gate_proj, x)
                if _PREFILL_MXFP8_STEEL and x.shape[-2] > 1 else self.gate_proj(x)
                ).astype(mx.float32)
        up = (prefill_linear(self.up_proj, x)
              if _PREFILL_MXFP8_STEEL and x.shape[-2] > 1 else self.up_proj(x)
              ).astype(mx.float32)
        if self.limit > 0:
            up = mx.clip(up, -self.limit, self.limit)
            gate = mx.minimum(gate, self.limit)
        h = nn.silu(gate) * up
        return (prefill_linear(self.down_proj, h.astype(dtype))
                if _PREFILL_MXFP8_STEEL and x.shape[-2] > 1
                else self.down_proj(h.astype(dtype)))


@partial(mx.compile, shapeless=True)
def _v41_moe_combine(y, weights):
    # Keep both the product and reduction in fp32, and the shared-expert add
    # outside this helper so there is no activation cast before that add.
    return mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)


def _ffn_down_bf16(mlp, x, indices):
    """Prefill-only SwitchGLU path with BF16 hidden/down; caller combines FP32."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    global _FFN_DOWN_BF16_ENGAGED
    if not _FFN_DOWN_BF16_ENGAGED:
        print(f"[DSV41] FFN_DOWN_BF16 engaged n={x.shape[0]}", flush=True)
        _FFN_DOWN_BF16_ENGAGED = True
    # n >= 64 implies indices.size >= 64: exactly SwitchGLU's sorted path.
    x, idx, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    if mlp.training:
        idx = mx.stop_gradient(idx)
    up = mlp.up_proj(x, idx, sorted_indices=True)
    gate = mlp.gate_proj(x, idx, sorted_indices=True)
    hidden = mlp.activation(up, gate).astype(mx.bfloat16)
    down = mlp.down_proj(hidden, idx, sorted_indices=True)
    return _scatter_unsort(down, inverse, indices.shape).squeeze(-2)


def _ffn_expert_pad16_layout(idx, experts):
    """GPU layout with exactly one scalar host read for the dynamic row count."""
    counts = mx.zeros((experts,), dtype=mx.int32).at[idx].add(1)
    lengths = ((counts + 15) // 16) * 16
    starts = mx.cumsum(counts) - counts
    end = mx.cumsum(lengths)
    padded_starts = end - lengths
    r_pad = int(end[-1].item())  # The one permitted host read per routed FFN.
    positions = mx.arange(r_pad, dtype=mx.int32)
    ids = mx.searchsorted(end, positions, side="right")
    local = positions - padded_starts[ids]
    gather = mx.where(local < counts[ids], starts[ids] + local, idx.size)
    keep = padded_starts[idx] + mx.arange(idx.size, dtype=mx.int32) - starts[idx]
    return counts, ids, gather, keep


def _ffn_expert_pad16(mlp, x, indices, down_bf16):
    """Pad each sorted expert to 16 rows, then discard padding before unsort."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    global _FFN_EXPERT_PAD16_ENGAGED, _FFN_DOWN_BF16_ENGAGED
    n = x.shape[0]
    x, idx, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    counts, ids, gather, keep = _ffn_expert_pad16_layout(idx, mlp.up_proj.num_experts)
    if not _FFN_EXPERT_PAD16_ENGAGED:
        occupied = int(mx.sum(counts > 0).item())  # One extra read, first call only.
        print(f"[DSV41] FFN_EXPERT_PAD16 engaged R={idx.size} R_pad={ids.size} "
              f"occupied_experts={occupied}", flush=True)
        _FFN_EXPERT_PAD16_ENGAGED = True
    x_ext = mx.concatenate((x, mx.zeros((1, 1, x.shape[-1]), dtype=x.dtype)))
    x = x_ext[gather]
    if mlp.training:
        ids = mx.stop_gradient(ids)
    up = mlp.up_proj(x, ids, sorted_indices=True)
    gate = mlp.gate_proj(x, ids, sorted_indices=True)
    hidden = mlp.activation(up, gate)
    if down_bf16:
        if not _FFN_DOWN_BF16_ENGAGED:
            print(f"[DSV41] FFN_DOWN_BF16 engaged n={n}", flush=True)
            _FFN_DOWN_BF16_ENGAGED = True
        hidden = hidden.astype(mx.bfloat16)
    down = mlp.down_proj(hidden, ids, sorted_indices=True)
    return _scatter_unsort(down[keep], inverse, indices.shape).squeeze(-2)


class MoE(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.ffn_down_bf16 = False  # Only backbone attach enables this, never DSpark stages.
        self.ffn_expert_pad16 = False
        self.ffn_expert_tiles = False
        self.ffn_fused = False
        self.gate = Gate(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
                                 activation=ClampedSwiGLU(args.swiglu_limit), bias=False)
        self.shared_experts = SharedExpert(args.hidden_size, args.moe_intermediate_size * args.n_shared_experts, args.swiglu_limit)

    def __call__(self, x: mx.array, image_mask=None) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf, None if image_mask is None else image_mask.reshape(-1))
        down_bf16 = (self.ffn_down_bf16 and xf.shape[0] >= 64
                     and not _v41_verify_context.get())
        pad16 = (self.ffn_expert_pad16 and xf.shape[0] >= 64
                 and not _v41_verify_context.get())
        expert_tiles = (getattr(self, "ffn_expert_tiles", False) and xf.shape[0] >= 64
                        and not _v41_verify_context.get())
        fused = (getattr(self, "ffn_fused", False) and xf.shape[0] >= 64
                 and not _v41_verify_context.get())
        if (os.environ.get("DSV41_PREFILL_GROUPED_MXFP4", "0") == "1"
                and not down_bf16  # Precision trial uses the stock gather_qmm kernels.
                and not pad16
                and not expert_tiles
                and not fused
                and xf.shape[0] > 1
                and xf.shape[0] >= int(os.environ.get("DSV41_PREFILL_GROUPED_MIN_ROWS", "256"))):
            from .moe_prefill import moe_prefill_grouped

            y = moe_prefill_grouped(self, xf, indices, weights)
        else:
            if fused:
                from ...custom_kernels.deepseek_v41.moe_prefill_fused_ffn import moe_prefill_fused_ffn

                y = moe_prefill_fused_ffn(self.switch_mlp, xf, indices)
            elif expert_tiles:
                from ...custom_kernels.deepseek_v41.moe_prefill_expert_tiles import moe_prefill_expert_tiles

                y = moe_prefill_expert_tiles(self.switch_mlp, xf, indices)
            elif pad16:
                y = _ffn_expert_pad16(self.switch_mlp, xf, indices, down_bf16)
            else:
                y = (_ffn_down_bf16(self.switch_mlp, xf, indices) if down_bf16
                     else self.switch_mlp(xf, indices))            # [tokens, topk, dim]
            if os.environ.get("DSV41_COMPILE_PRIMITIVES", "0") == "1":
                y = _v41_moe_combine(y, weights)
            else:
                y = mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)
        y = y + self.shared_experts(xf).astype(mx.float32)
        return y.reshape(shape).astype(x.dtype)


def _rope(rope, x, start_pos, inverse=False):
    """V4 RoPE takes sequence on the penultimate axis, including headed inputs."""
    if x.ndim == 4:
        return rope(x.swapaxes(1, 2), start_pos, inverse).swapaxes(1, 2)
    return rope(x, start_pos, inverse)


class _LayerCache:
    """Capacity buffers with logical-length views under the original field names.

    Each field starts at `step` rows and doubles independently. Only growth
    copies retained rows; appends overwrite a slice and trim retains capacity.
    """

    fields = ("win_kv", "kv", "score", "comp_kv", "index_k")

    def __init__(self, step=DSV41_CACHE_STEP, bounded=False):
        self.step = step
        self.bounded = bounded
        self.buffers = {name: None for name in self.fields}
        self.lengths = dict.fromkeys(self.fields, 0)
        self.bases = dict.fromkeys(self.fields, 0)

    def __getattr__(self, name):
        if name in self.fields:
            buf = self.buffers[name]
            return None if buf is None else buf[:, :self.lengths[name]]
        raise AttributeError(name)

    def append(self, name, rows):
        if self.bounded and name in ("win_kv", "kv", "score"):
            previous = getattr(self, name)
            self.buffers[name] = (rows if previous is None or not previous.shape[1]
                                  else mx.concatenate([previous, rows], axis=1))
            self.lengths[name] = self.buffers[name].shape[1]
            return
        off = self.lengths[name]
        end = off + rows.shape[1]
        buf = self.buffers[name]
        capacity = self.step if buf is None else buf.shape[1]
        while capacity < end:
            capacity *= 2
        if buf is None or capacity != buf.shape[1]:
            grown = mx.zeros((rows.shape[0], capacity, *rows.shape[2:]), dtype=rows.dtype)
            if off:
                grown[:, :off] = buf[:, :off]
            buf = grown
        buf[:, off:end] = rows
        self.buffers[name] = buf
        self.lengths[name] = end

    def truncate(self, name, length):
        self.lengths[name] = min(self.lengths[name], length)

    def absolute_slice(self, name, start, stop):
        base = self.bases[name]
        if start < base:
            raise ValueError(f"{name} history before {base} is no longer retained")
        return getattr(self, name)[:, start - base:stop - base]

    def discard_before(self, name, position):
        value = getattr(self, name)
        if value is None:
            return
        count = position - self.bases[name]
        if not 0 <= count <= self.lengths[name]:
            raise ValueError(f"Invalid {name} retention boundary {position}")
        self.buffers[name] = value[:, count:]
        self.lengths[name] -= count
        self.bases[name] = position


class DeepseekV41Cache:
    """Growing conversation state, including replayable compressor projections.

    Construct through LanguageModel.make_cache(). State is a flat array list:
    version/offset/per-field-length header, five logical views per layer, then
    compressed token history. Version-one states remain readable. Restoring
    allocates independent capacity buffers; nbytes counts allocated storage.
    Empty rank-one arrays represent absent slots. Restoring into a fresh cache
    from the same model preserves everything needed for continuation and trim.
    """

    # Only live models may supply compressor weights / Engram hash parameters.
    # Disk files never construct a model or import a class named by the file.
    _restore_models = {}

    # End position of the last DSV41_PREFILL_CONE prefill, or None, and the lowest
    # offset trim() will rewind to. A cone prefill leaves reserved (never-computed)
    # window-KV rows below each decoder layer's cone. Nothing reads them while the
    # conversation runs FORWARD -- the sliding window never reaches further back
    # than window_size - 1 rows and every decoder layer holds the last window_size
    # positions for real -- and a rewind to `_cone_floor` or above is equally safe
    # for the same reason. Only a rewind BELOW the floor would attend into the gap,
    # and that is the one thing trim() refuses. See trim_floor().
    _cone_end = None
    _cone_floor = 0

    def __init__(self, model, token_map=None):
        self._attentions = [layer.attn for layer in model.layers]
        try:
            self.step = int(os.environ.get("DSV41_CACHE_STEP", DSV41_CACHE_STEP))
        except ValueError:
            self.step = DSV41_CACHE_STEP
        if self.step < 1:
            raise ValueError("DSV41_CACHE_STEP must be positive")
        self.bounded = os.environ.get("DSV41_BOUNDED_CACHE", "0") == "1"
        self.layers = [_LayerCache(self.step, self.bounded) for _ in model.layers]
        self._offset = 0
        self.engram = None
        if model.layout is not None:
            if token_map is None:
                raise ValueError("Engram requires set_token_map() or a model_dir/tokenizer.json")
            cfg = model.config
            state = NgramHashState(model.layout, token_map, cfg.engram_pad_token_id,
                                   cfg.engram_compressed_vocab_size)
            self.engram = EngramHashCache(state)
        self._bind_model(model, token_map)

    def _bind_model(self, model, token_map):
        import hashlib
        import json
        import weakref
        from mlx_lm.models import cache as lm_cache

        qat = os.environ.get("DSV41_QAT", "1") != "0"
        key = (id(model), qat)
        entry = self._restore_models.get(key)
        if entry is None or entry[0]() is not model or entry[2] is not token_map:
            digest = hashlib.sha256(json.dumps(
                model.config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode())
            digest.update(str(qat).encode())
            if token_map is not None:
                digest.update(np.asarray(token_map, dtype="<i8").tobytes())
            signature = digest.hexdigest()
            registry = self._restore_models
            ref = weakref.ref(model, lambda _: registry.pop(key, None))
            self._restore_models[key] = (ref, signature, token_map)
        self._restore_signature = self._restore_models[key][1]
        self._qat = qat
        lm_cache.DeepseekV41Cache = type(self)

    @property
    def meta_state(self):
        """mlx-lm's string-only metadata; model weights stay outside the cache."""
        if self.bounded:
            raise ValueError("Bounded V4.1 cache cannot be serialized or used for prefix rollback")
        if self._cone_end is not None:
            raise ValueError(
                "A DSV41_PREFILL_CONE cache cannot be serialized for prefix rollback: "
                "the state carries no record of the reserved window-KV rows nor of the "
                "rewind floor trim() enforces, so a restored copy would believe it can "
                "be trimmed into them. In-memory reuse is fine -- see trim_floor().")
        return ("deepseek_v41_v1", self._restore_signature)

    @classmethod
    def from_state(cls, state, meta_state):
        if len(meta_state) != 2 or meta_state[0] != "deepseek_v41_v1":
            raise ValueError("Unsupported V4.1 cache metadata")
        qat = os.environ.get("DSV41_QAT", "1") != "0"
        matches = [(ref(), token_map) for (_, bound_qat), (ref, sig, token_map)
                   in list(cls._restore_models.items())
                   if sig == meta_state[1] and bound_qat == qat and ref() is not None]
        if len(matches) != 1:
            raise ValueError("V4.1 restore requires one matching live model; call make_cache() first")
        out = cls(*matches[0])
        out.state = state
        return out

    def clone(self):
        """Independent conversation buffers, sharing only immutable model data.

        In-memory APC must not serialize the CPU Engram history to MLX and back,
        nor synchronize a device header to recover lengths already held in Python.
        """
        import copy

        if self._qat != (os.environ.get("DSV41_QAT", "1") != "0"):
            raise ValueError("V4.1 cache QAT mode changed")
        out = copy.copy(self)
        out.layers = []
        for lc in self.layers:
            copied = _LayerCache(self.step, self.bounded)
            for name in lc.fields:
                value = getattr(lc, name)
                if value is not None:
                    copied.append(name, value)
            copied.bases = dict(lc.bases)
            out.layers.append(copied)
        if self.engram is not None:
            hs = copy.copy(self.engram.hash_state)
            if hs.history is not None:
                hs.history = hs.history[:, :self.offset].copy()
            out.engram = EngramHashCache(hs)
        return out

    @property
    def offset(self):
        return self._offset

    @property
    def capacity(self):
        return max((buf.shape[1] for lc in self.layers for buf in lc.buffers.values()
                    if buf is not None), default=0)

    def note_cone_prefill(self, end_pos, floor):
        """Record a DSV41_PREFILL_CONE prefill and the rewind floor it leaves.

        `floor` is the lowest offset this prefill still leaves rewindable: the
        HIGHEST first-published row in the decoder half (the top layer's, which
        publishes exactly `window_size` rows) plus window_size - 1, because a
        forward step at offset p reads that layer's window [p - window + 1, p] and
        every row below its first published row is reserved, never computed. The
        driver derives it from the loop that did the reserving, not from algebra.
        """
        self._cone_end = max(int(end_pos), self._cone_end or 0)
        self._cone_floor = max(self._cone_floor, int(floor))

    def trim_floor(self):
        """Lowest offset trim() accepts; 0 unless a cone prefill has run.

        Public so a caller that must PREFLIGHT a rollback (the receiver's
        `_can_trim_cache_to_offset`) can refuse before it commits, instead of
        discovering the refusal from trim() after it has promised a prefix.
        """
        return 0 if self._cone_end is None else self._cone_floor

    def is_trimmable(self):
        # A cone cache IS rewindable, down to trim_floor(). Above that floor every
        # decoder layer still holds a full window of real rows, which is the only
        # thing a forward step from the new offset reads; below it the rows are the
        # ones the cone deliberately never computed, and trim() refuses.
        return not self.bounded

    def trim(self, n):
        n = min(max(int(n), 0), self.offset)
        if not n:
            return 0
        if self.bounded:
            raise ValueError("Bounded V4.1 cache requires a fresh prefill for rollback")
        kept = self.offset - n
        if self._cone_end is not None and kept < self._cone_floor:
            raise ValueError(
                "This cache was filled by a DSV41_PREFILL_CONE prefill ending at "
                f"{self._cone_end}, so the decoder half holds real window KV only from "
                f"row {self._cone_floor - self._attentions[-1].window_size + 1} upwards; "
                f"rewinding to {kept} would attend into rows the cone never computed. "
                f"The lowest offset this cache can be rewound to is {self._cone_floor} "
                "(DeepseekV41Cache.trim_floor). Re-prefill from scratch instead, or "
                "switch DSV41_PREFILL_CONE off for conversations that rewind further.")
        for attn, lc in zip(self._attentions, self.layers):
            for name in ("win_kv", "kv", "score"):
                value = getattr(lc, name)
                if value is not None:
                    lc.truncate(name, kept)
            if attn.is_kv_source:
                attn.rebuild_tail(lc, kept)
        if self.engram is not None:
            self.engram.trim(n)
            hs = self.engram.hash_state
            hs.history = hs.history[:, :kept].copy()
        self._offset = kept
        return n

    @property
    def state(self):
        # EVALUATION-ONLY for anything it marks unrestorable, and the decode path depends
        # on that: the receiver calls mx.eval(cache[0].state) every step to materialise
        # lazy buffers (mlx_token_stream_server.py:3958). Refusing HERE broke decode
        # outright after a cone prefill -- the request returned empty. A bounded cache
        # already solves this by emitting a version the SETTER rejects; a cone cache does
        # the same, so a round-trip stays impossible while evaluation keeps working.
        empty = mx.zeros((0,), dtype=mx.int32)
        lengths = [lc.lengths[k] for lc in self.layers for k in lc.fields]
        # Versions 3 (bounded) and 4 (cone) are evaluation-only; the setter refuses both
        # and meta_state rejects persistence outright.
        version = 4 if self._cone_end is not None else (3 if self.bounded else 2)
        arrays = [mx.array([version, self.offset, *lengths], dtype=mx.int32)]
        for lc in self.layers:
            arrays.extend(empty if getattr(lc, k) is None else getattr(lc, k)
                          for k in lc.fields)
        hs = None if self.engram is None else self.engram.hash_state
        arrays.append(empty if hs is None or hs.history is None else
                      mx.array(hs.history[:, :self.offset].astype(np.int32)))
        return arrays

    @state.setter
    def state(self, arrays):
        if self.bounded:
            raise ValueError("Bounded V4.1 cache cannot restore serialized history")
        if len(arrays) != 2 + 5 * len(self.layers):
            raise ValueError("V4.1 cache state has the wrong number of arrays")
        if arrays[0].ndim != 1 or arrays[0].dtype != mx.int32:
            raise ValueError("Invalid V4.1 cache state header")
        # Only this small control header crosses to CPU (202 ints for 40 layers).
        header = np.array(arrays[0]).tolist()
        if len(header) >= 1 and header[0] == 4:
            raise ValueError(
                "A DSV41_PREFILL_CONE cache cannot be restored from state: the array "
                "list carries no record of the reserved window-KV rows nor of the "
                "rewind floor trim() enforces, so the restored copy would believe it "
                "can be trimmed into them. In-memory reuse is fine -- see trim_floor().")
        if (len(header) < 2 or header[0] not in (1, 2) or header[1] < 0
                or len(header) != (2 if header[0] == 1 else 2 + 5 * len(self.layers))):
            raise ValueError("Invalid V4.1 cache state header")
        offset = int(header[1])
        restored = []
        for i, attn in enumerate(self._attentions):
            lc = _LayerCache(self.step)
            for j, (name, value) in enumerate(zip(lc.fields, arrays[1 + 5 * i:6 + 5 * i])):
                if value.ndim != 3 and not (value.ndim == 1 and value.size == 0):
                    raise ValueError(f"Invalid cache rank for layer {i}.{name}")
                if header[0] == 2 and header[2 + 5 * i + j] != (0 if value.ndim == 1 else value.shape[1]):
                    raise ValueError("Cache logical length disagrees with state array")
                if value.ndim == 1 and value.size == 0:
                    continue
                if name != "win_kv" and not attn.is_kv_source:
                    raise ValueError("A non-owner cannot hold compressor state")
                if name == "index_k" and not attn.is_index_source:
                    raise ValueError("A non-index-owner cannot hold index keys")
                if name in ("comp_kv", "index_k"):
                    expected = offset // attn.ratio
                else:
                    expected = offset
                if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != expected:
                    raise ValueError(f"Invalid cache shape for layer {i}.{name}")
                width = attn.indexer.head_dim if name == "index_k" else attn.head_dim
                if value.shape[2] != width:
                    raise ValueError(f"Invalid cache width for layer {i}.{name}")
                lc.append(name, value)
            if offset and lc.win_kv is None:
                raise ValueError("Missing window cache")
            if offset and attn.is_kv_source and lc.kv is None:
                raise ValueError("Missing compressor projections")
            if offset and attn.is_kv_source and attn.ratio > 1 and lc.score is None:
                raise ValueError("Missing compressor score projections")
            if attn.is_kv_source and offset // attn.ratio:
                if lc.comp_kv is None or (attn.is_index_source and lc.index_k is None):
                    raise ValueError("Missing completed compressed groups")
            restored.append(lc)
        if self.engram is not None:
            # Hashing currently runs on CPU; this one token-history transfer is
            # necessary for disk restore. KV/projection tensors stay on device.
            history = np.array(arrays[-1], dtype=np.int64)
            if history.shape != (1, offset) and not (offset == 0 and history.size == 0):
                raise ValueError("Invalid Engram cache history")
            self.engram.hash_state.history = history.reshape(1, offset).copy()
            self.engram.hash_state.length = offset
        elif arrays[-1].size:
            raise ValueError("Unexpected Engram cache history")
        self.layers, self._offset = restored, offset

    @property
    def nbytes(self):
        total = sum(buf.nbytes for lc in self.layers for buf in lc.buffers.values()
                    if buf is not None)
        return total + (self.engram.nbytes if self.engram is not None else 0)

    def finish(self, end_pos):
        if self.bounded:
            for attn, lc in zip(self._attentions, self.layers):
                lc.discard_before("win_kv", max(0, end_pos - attn.window_size))
                if attn.is_kv_source:
                    complete = end_pos - end_pos % attn.ratio
                    for name in ("kv", "score"):
                        lc.discard_before(name, complete)
        self._offset = end_pos


class SharedState:
    """Fresh per forward: latest KV owner, key owner, selection and candidates."""

    def __init__(self):
        self.kv_src_cache = None
        self.index_src_cache = None
        self.topk_idxs = None
        self.candidates = None
        # Absolute prompt position of row 0 of topk_idxs / candidates. Identical to
        # every reader's start_pos on the default path; under DSV41_PREFILL_CONE the
        # reader runs a suffix of the publisher's rows. See _shared_rows.
        self.topk_start = None
        self.candidates_start = None


@partial(mx.compile, shapeless=True)
def _v41_compress_pool(kv, score):
    # Inputs are already grouped fp32 projections. V4's helper adds APE and
    # casts softmax weights to the KV dtype; V4.1 has neither APE nor overlap.
    return mx.sum(kv * mx.softmax(score, axis=2), axis=2)


class Compressor(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ratio = config.compress_ratios[layer_id]
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        if self.ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)

    def pool(self, lc, first, stop, dtype):
        """Complete groups [first, stop); partial projections remain in lc."""
        if stop <= first:
            return None
        _s4_count("pooled_groups", amount=stop - first)
        ratio = self.ratio
        kv = lc.absolute_slice("kv", first * ratio, stop * ratio)
        if ratio > 1:
            shape = (kv.shape[0], stop - first, ratio, kv.shape[-1])
            score = lc.absolute_slice("score", first * ratio, stop * ratio).reshape(shape)
            if os.environ.get("DSV41_COMPILE_PRIMITIVES", "0") == "1":
                kv = _v41_compress_pool(kv.reshape(shape), score)
            else:
                kv = mx.sum(kv.reshape(shape) * mx.softmax(score, axis=2), axis=2)
        return self.norm(kv.astype(dtype))

    def __call__(self, x, start_pos, lc):
        if self.ratio == 1:
            # Match the reference's activation-dtype matmul at ratio one.
            kv = self.wkv(x).astype(mx.float32)
        else:
            xf = x.astype(mx.float32)
            kv = xf @ self.wkv.weight.astype(mx.float32).T
            score = xf @ self.wgate.weight.astype(mx.float32).T
            lc.append("score", score)
        lc.append("kv", kv)
        return self.pool(lc, start_pos // self.ratio,
                         (start_pos + x.shape[1]) // self.ratio, x.dtype)


def window_idx_matrix(start_pos, n, window):
    width = min(window, start_pos + n)
    end = start_pos + mx.arange(n, dtype=mx.int32)
    first = mx.maximum(end - window + 1, 0)
    idx = first[:, None] + mx.arange(width, dtype=mx.int32)[None]
    return mx.where(idx <= end[:, None], idx, -1)


class Attention(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.ratio = config.compress_ratios[layer_id]
        self.is_kv_source = layer_id in config.kv_source_layer_ids
        self.is_index_source = layer_id in config.index_source_layer_ids
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.wo_a = MultiLinear(self.n_heads * self.head_dim // self.n_groups,
                                self.o_lora_rank, self.n_groups)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, config.hidden_size, bias=False)
        if self.is_kv_source:
            self.compressor = Compressor(config, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(config, layer_id)
        scaling = dict(config.rope_scaling) if self.ratio and config.rope_scaling else None
        if scaling is not None:
            scaling.setdefault("rope_type", scaling.get("type", "yarn"))
        theta = config.compress_rope_theta if self.ratio else config.rope_theta
        self.rope = DeepseekV4RoPE(config.qk_rope_head_dim, theta, scaling)
        self.group_rope = DeepseekV4RoPE(config.qk_rope_head_dim, theta, scaling,
                                       freq_scale=max(self.ratio, 1))

    def _publish(self, latents, first_group, lc):
        pos = first_group * self.ratio
        if self.is_index_source:
            self.indexer.publish_keys(latents, pos, self.group_rope, lc)
        rotated = _rope(self.group_rope, latents, pos)
        lc.append("comp_kv", fake_quant_fp4_e4m3(rotated, 16))

    def rebuild_tail(self, lc, kept):
        """Drop invalid groups and recompute the last complete group from projections."""
        groups = kept // self.ratio
        first = max(groups - 1, 0)
        for name in ("comp_kv", "index_k"):
            value = getattr(lc, name)
            if value is not None:
                lc.truncate(name, first)
        if groups:
            latents = self.compressor.pool(lc, first, groups, lc.win_kv.dtype)
            self._publish(latents, first, lc)

    def _compressed_indices(self, x, qr, start_pos, end_pos, lc, shared,
                            kv_already_prepared=False):
        if self.is_kv_source:
            if not kv_already_prepared:
                latents = self.compressor(x, start_pos, lc)
                if latents is not None:
                    self._publish(latents, start_pos // self.ratio, lc)
            # Republished either way: under DSV41_PREFILL_CONE the compute half runs
            # with a SharedState that never saw prepare_kv, and every layer above the
            # boundary reads this pointer for the compressed bank.
            shared.kv_src_cache = lc
        if self.is_index_source:
            if self.indexer.owns_k:
                # Even an incomplete group MUST replace the last owner's pointer.
                shared.index_src_cache = lc
            if end_pos // self.ratio:
                keys = shared.index_src_cache.index_k
                indices = self.indexer(x, qr, start_pos, end_pos, self.rope, keys, shared)
            else:
                indices = mx.zeros((*x.shape[:2], 0), dtype=mx.int32)
            shared.topk_idxs = indices
            shared.topk_start = start_pos
            shared.dense_compressed = (
                end_pos // self.ratio <= min(self.indexer.index_topk,
                    self.indexer.candidate_topk_blocks * self.indexer.candidate_block_size)
                and shared.candidates is None)
        else:
            _s4_count("index_reuse", self.layer_id)
        return _shared_rows(shared.topk_idxs, getattr(shared, "topk_start", None), start_pos)

    def prepare_kv(self, x, start_pos, cache, shared):
        """Publish this layer's KV -- the half of __call__ that WRITES cache state.

        Separated from the compute half so a layer-major or cone prefill can run it
        over a different row range (DSV41_PREFILL_LAYER_MAJOR / DSV41_PREFILL_CONE).
        This is the window KV append plus, for a KV-source layer, the compressor and
        its bank publication; `x` is the post-attn-norm hidden, exactly as __call__
        passes it. Returns this layer's cache.
        """
        lc = cache.layers[self.layer_id]
        kv = _rope(self.rope, self.kv_norm(
            prefill_linear(self.wkv, x)
            if _PREFILL_MXFP8_STEEL and x.shape[1] > 1 else self.wkv(x)), start_pos)
        lc.append("win_kv", fake_quant_fp8_ue8m0(kv, 32))
        if self.ratio and self.is_kv_source:
            latents = self.compressor(x, start_pos, lc)
            if latents is not None:
                self._publish(latents, start_pos // self.ratio, lc)
            shared.kv_src_cache = lc
        return lc

    def reserve_kv_rows(self, cache, count, batch, dtype):
        """Placeholder window-KV rows for positions DSV41_PREFILL_CONE never computes.

        Absolute position indexes win_kv (`window_idx_matrix` emits absolute indices,
        rebased only by `lc.bases`), so rows below a layer's cone must still OCCUPY
        their slots for the rows above them to be addressable. Nothing reads them: the
        sliding window never reaches further back than `window_size - 1` rows and the
        cone always covers at least the last `window_size` positions of every decoder
        layer, at this prefill, at every later decode step, and after a rewind to any
        offset at or above the floor. The cache records that floor
        (DeepseekV41Cache.note_cone_prefill) and trim() refuses to go below it, so a
        later turn can never attend into the gap.
        """
        if count <= 0:
            return
        cache.layers[self.layer_id].append(
            "win_kv", mx.zeros((batch, count, self.head_dim), dtype=dtype))

    def __call__(self, x, start_pos, cache, shared, kv_already_prepared=False):
        return self.block(x, start_pos, cache, shared, kv_already_prepared)

    def block(self, x, start_pos, cache, shared, kv_already_prepared=False):
        """The compute half: q, RoPE, index selection, sparse attention, wo_a/wo_b.

        With kv_already_prepared=True it neither appends to win_kv nor republishes the
        compressed bank -- it reads what prepare_kv stored -- and computes only the
        rows it was given. Default False composes the two halves in the original
        order, so the default path is unchanged.
        """
        b, n, _ = x.shape
        _attn_cuts = _PREFILL_LEDGER_CUTS_LEVEL >= 2 and n > 1
        end_pos = start_pos + n
        lc = cache.layers[self.layer_id]
        qr = self.q_norm(prefill_linear(self.wq_a, x)
                         if _PREFILL_MXFP8_STEEL and x.shape[1] > 1 else self.wq_a(x))
        q = (prefill_linear(self.wq_b, qr)
             if _PREFILL_MXFP8_STEEL and x.shape[1] > 1 else self.wq_b(qr)
             ).reshape(b, n, self.n_heads, self.head_dim)
        q = _rope(self.rope, q, start_pos)
        if _attn_cuts:
            _t0 = time.perf_counter()
            mx.eval(q)
            mx.synchronize()
            _prefill_layer_add(self.layer_id, "attn_proj_q_ms", (time.perf_counter() - _t0) * 1000.0)
        if not kv_already_prepared:
            self.prepare_kv(x, start_pos, cache, shared)
        kv_all = lc.win_kv
        if _attn_cuts:
            _t0 = time.perf_counter()
            mx.eval(kv_all)
            mx.synchronize()
            _prefill_layer_add(self.layer_id, "attn_proj_kv_ms", (time.perf_counter() - _t0) * 1000.0)
        idxs = window_idx_matrix(start_pos, n, self.window_size)
        base = lc.bases["win_kv"]
        if base:
            idxs = mx.where(idxs >= 0, idxs - base, -1)
        idxs = mx.broadcast_to(idxs[None], (b, *idxs.shape))
        if _attn_cuts:
            _index_ms = 0.0
        if self.ratio:
            cidx = self._compressed_indices(x, qr, start_pos, end_pos, lc, shared,
                                            kv_already_prepared)
            if _attn_cuts and (self.is_index_source or self.is_kv_source):
                # Top-k alone does not force KV/key publication on bypass or
                # KV-only layers, nor the projections of an incomplete group.
                _index_outputs = [cidx] if self.is_index_source else []
                if self.is_index_source and self.indexer.is_candidate_source:
                    _index_outputs.append(shared.candidates)
                if self.is_kv_source:
                    _index_outputs.extend(getattr(lc, name) for name in
                                          ("kv", "score", "comp_kv", "index_k"))
                _index_outputs = [value for value in _index_outputs if value is not None]
                _t0 = time.perf_counter()
                mx.eval(*_index_outputs)
                mx.synchronize()
                _index_ms = (time.perf_counter() - _t0) * 1000.0
            if base:
                cidx = mx.where(cidx >= 0, cidx - base, -1)
            if end_pos // self.ratio:
                kv_all = (kv_all, shared.kv_src_cache.comp_kv)
                idxs = mx.concatenate([idxs, cidx], axis=-1)
        if _attn_cuts:
            # Window-only and shared-index/KV reuse layers do no index work.
            _prefill_layer_add(self.layer_id, "attn_index_ms", _index_ms)
        if (os.environ.get("DSV41_S4_DENSE", "0") == "1" and n == 1
                and self.ratio and end_pos // self.ratio and shared.dense_compressed):
            out = _dense_decode_attn(q, kv_all, self.attn_sink,
                                     self.window_size, self.head_dim ** -0.5)
        else:
            out = sparse_attn(q, kv_all, self.attn_sink, idxs, self.head_dim ** -0.5)
        if _attn_cuts:
            _t0 = time.perf_counter()
            mx.eval(out)
            mx.synchronize()
            _prefill_layer_add(self.layer_id, "attn_sparse_ms", (time.perf_counter() - _t0) * 1000.0)
        out = _rope(self.rope, out, start_pos, inverse=True)
        out = out.reshape(b, n, self.n_groups, -1)
        # MultiLinear provides the checkpoint layout; preserve PN's fp32 contraction.
        if self.wo_a.weight.shape[-1] != out.shape[-1] or self.wo_a.weight.dtype == mx.uint32:
            raise ValueError("wo_a must be dequantized to floating [o_groups, o_lora_rank, input_dim]")
        if _WO_A_BF16 and self.wo_a.weight.dtype == mx.bfloat16:
            out = mx.einsum("bsgd,grd->bsgr", out.astype(mx.bfloat16), self.wo_a.weight)
        else:
            out = mx.einsum("bsgd,grd->bsgr", out.astype(mx.float32),
                            self.wo_a.weight.astype(mx.float32))
        out = (prefill_linear(self.wo_b, out.reshape(b, n, -1).astype(x.dtype))
               if _PREFILL_MXFP8_STEEL and x.shape[1] > 1
               else self.wo_b(out.reshape(b, n, -1).astype(x.dtype)))
        if _attn_cuts:
            _t0 = time.perf_counter()
            mx.eval(out)
            mx.synchronize()
            _prefill_layer_add(self.layer_id, "attn_out_ms", (time.perf_counter() - _t0) * 1000.0)
        return out


class HyperConnection(nn.Module):
    """Compute the *next* sub-block's pre-mix and this sub-block's post/comb.

    Collapse uses the incoming pre-mix, never the pre returned here. See
    hyper_connection.py for the fusion contract and numerical constraints.
    """

    def __init__(self, config):
        super().__init__()
        self.hc = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        # Moving the normalization before GEMM changes FP32 rounding. Keep the
        # reference's projection-first order unless explicitly opted in.
        self.fast_norm = os.environ.get("DSV41_HC_FAST_NORM", "0") == "1"
        size = (2 + self.hc) * self.hc
        self.fn = mx.zeros((size, self.hc * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((size,), dtype=mx.float32)
        self.scale = mx.zeros((3,), dtype=mx.float32)

    def __call__(self, x, incoming_pre=None):
        if os.environ.get("DSV41_HC_FUSED", "0") == "1":
            from .hyper_connection import fused_staggered_hc

            if incoming_pre is None:
                raise ValueError("DSV41_HC_FUSED needs incoming_pre; block_graph integration is required")
            xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)
            projection = xf @ self.fn.astype(mx.float32).T
            return fused_staggered_hc(x, incoming_pre, projection, self.scale, self.base,
                                      self.iters, self.eps, self.norm_eps)
        xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)
        if self.fast_norm:
            mixes = mx.fast.rms_norm(xf, None, self.norm_eps) @ self.fn.astype(mx.float32).T
        else:
            rsqrt = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + self.norm_eps)
            mixes = (xf @ self.fn.astype(mx.float32).T) * rsqrt
        split = _hc_split_sinkhorn_ops
        if os.environ.get("DSV41_METAL_SINKHORN", "0") == "1":
            from .hyper_connection import metal_sinkhorn
            split = metal_sinkhorn
        return split(mixes, self.scale, self.base, self.hc, self.iters, self.eps)


def hc_pre(x, pre_mix):
    return mx.sum(pre_mix[..., None].astype(mx.float32) * x.astype(mx.float32),
                  axis=2).astype(x.dtype)


def hc_post(x, residual, post, comb):
    from .hyper_connection import COMPILE_POST, compiled_hc_post

    if os.environ.get("DSV41_HC_FUSED_POST", "0") == "1":
        from .hyper_connection import metal_hc_post

        return metal_hc_post(x, residual, post, comb)
    if COMPILE_POST:
        return compiled_hc_post(x, residual, post, comb)
    prod = comb.astype(mx.float32)[..., None] * residual.astype(mx.float32)[..., :, None, :]
    out = post.astype(mx.float32)[..., None] * x.astype(mx.float32)[..., None, :]
    return (out + mx.sum(prod, axis=2)).astype(x.dtype)


def _prefill_block_prepared(layer, h, pre_mix, start_pos, cache, shared, image_mask=None):
    """megakernel.prefill_block, with attention told its KV is already published.

    DSV41_PREFILL_MEGAKERNEL is ON in the production preset, so the cone MUST run the
    same kernels as the path it is measured against; `prefill_block` cannot take the
    kv_already_prepared flag and megakernel.py is out of scope for this change, so
    this mirrors it line for line and calls exactly the same helpers.
    KEEP IN SYNC WITH megakernel.prefill_block.
    """
    from .megakernel import (_check_rows, _prefill_hc, collapse_norm_rows,
                             expand_collapse_norm_rows, expand_rows)

    # prefill_block's own "more than one token" guard is about a DECODE step
    # reaching it, not about the row kernels: those index grid.y by [batch, token]
    # and are correct at one row. The cone's top layer computes exactly one row and
    # must still run these kernels, or that layer alone would silently take the
    # eager collapse and the parity gate would be measuring two different paths.
    _check_rows(h)
    attn_pre, post, comb = _prefill_hc(layer.attn_hc, h)
    x = collapse_norm_rows(h, pre_mix, layer.attn_norm)
    attention = layer.attn(x, start_pos, cache, shared, kv_already_prepared=True)
    h, x = expand_collapse_norm_rows(attention, h, post, comb, attn_pre, layer.ffn_norm)
    if _PREFILL_LEDGER_CUTS:
        t0 = time.perf_counter()
        mx.eval(h, x, attn_pre)
        mx.synchronize()
        _prefill_layer_add(layer.attn.layer_id, "attn_ms", (time.perf_counter() - t0) * 1000.0)
    ffn_pre, post, comb = _prefill_hc(layer.ffn_hc, h)
    h = expand_rows(layer.ffn(x, image_mask), h, post, comb)
    if _PREFILL_LEDGER_CUTS:
        t0 = time.perf_counter()
        mx.eval(h, ffn_pre)
        mx.synchronize()
        _prefill_layer_add(layer.attn.layer_id, "ffn_ms", (time.perf_counter() - t0) * 1000.0)
    return h, ffn_pre


class DeepseekV41Block(nn.Module):
    def __init__(self, layer_id, config, layout):
        super().__init__()
        self.attn = Attention(layer_id, config)
        self.ffn = MoE(config)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.engram = Engram(config, layer_id, layout) if layer_id in config.engram_layer_ids else None

    def _attn_input(self, h, pre_mix):
        """The attention input for this block: HC collapse then attn_norm.

        Single definition so prepare_attn_kv and the compute half agree, including
        under DSV41_PREFILL_MEGAKERNEL, whose fused row kernel is not bit-identical
        to the eager collapse and must therefore be used by BOTH halves.
        """
        if os.environ.get("DSV41_PREFILL_MEGAKERNEL", "0") == "1" and h.shape[1] > 1:
            from .megakernel import collapse_norm_rows
            return collapse_norm_rows(h, pre_mix, self.attn_norm)
        if os.environ.get("DSV41_HC_FUSED", "0") == "1":
            x = self.attn_hc(h, pre_mix)[0]
        else:
            x = hc_pre(h, pre_mix)
        return self.attn_norm(x)

    def prepare_attn_kv(self, h, pre_mix, start_pos, cache, shared):
        """Block-level half of Attention.prepare_kv: build x from (h, pre_mix), publish KV.

        Used by DSV41_PREFILL_CONE to publish a layer's KV over rows it will not go on
        to compute. Returns the layer cache that was written.
        """
        return self.attn.prepare_kv(self._attn_input(h, pre_mix), start_pos, cache, shared)

    def __call__(self, h, pre_mix, start_pos, cache, shared, image_mask=None,
                 kv_already_prepared=False):
        if kv_already_prepared:
            # DSV41_PREFILL_CONE only. It must run the SAME kernels the path it is
            # measured against runs, at every row count including the top layer's
            # single row -- the two single-token DECODE specialisations below are
            # not that path and are never reachable from the cone.
            if os.environ.get("DSV41_PREFILL_MEGAKERNEL", "0") == "1":
                return _prefill_block_prepared(self, h, pre_mix, start_pos, cache,
                                               shared, image_mask)
            # Otherwise fall through to the eager path, which is exactly what the
            # default prefill runs when DSV41_PREFILL_MEGAKERNEL is off.
        else:
            if os.environ.get("DSV41_MEGAKERNEL", "0") == "1" and h.shape[:2] == (1, 1):
                from .megakernel import decode_block
                return decode_block(self, h, pre_mix, start_pos, cache, shared, image_mask)
            if os.environ.get("DSV41_COMPILE_BLOCK", "0") == "1" and h.shape[:2] == (1, 1):
                from .block_graph import BlockGraphs
                graph = getattr(self, "_block_graph", None)
                if graph is None:
                    graph = BlockGraphs()
                    object.__setattr__(self, "_block_graph", graph)
                return graph(self, h, pre_mix, start_pos, cache, shared, image_mask)
            if os.environ.get("DSV41_PREFILL_MEGAKERNEL", "0") == "1" and h.shape[1] > 1:
                from .megakernel import prefill_block
                return prefill_block(self, h, pre_mix, start_pos, cache, shared, image_mask)
        if os.environ.get("DSV41_HC_FUSED", "0") == "1":
            x, attn_pre, post, comb = self.attn_hc(h, pre_mix)
        else:
            attn_pre, post, comb = self.attn_hc(h)
            x = hc_pre(h, pre_mix)
        x = self.attn_norm(x)
        h = hc_post(self.attn(x, start_pos, cache, shared, kv_already_prepared),
                    h, post, comb)
        if _PREFILL_LEDGER_CUTS and h.shape[1] > 1:
            _t0 = time.perf_counter()
            mx.eval(h, attn_pre)
            mx.synchronize()
            _prefill_layer_add(self.attn.layer_id, "attn_ms", (time.perf_counter() - _t0) * 1000.0)
        if os.environ.get("DSV41_HC_FUSED", "0") == "1":
            x, ffn_pre, post, comb = self.ffn_hc(h, attn_pre)
        else:
            ffn_pre, post, comb = self.ffn_hc(h)
            x = hc_pre(h, attn_pre)
        x = self.ffn_norm(x)
        if _PREFILL_LEDGER_CUTS and h.shape[1] > 1:
            h = hc_post(self.ffn(x, image_mask), h, post, comb)
            _t0 = time.perf_counter()
            mx.eval(h, ffn_pre)
            mx.synchronize()
            _prefill_layer_add(self.attn.layer_id, "ffn_ms", (time.perf_counter() - _t0) * 1000.0)
            return h, ffn_pre
        return hc_post(self.ffn(x, image_mask), h, post, comb), ffn_pre


# --------------------------------------------------------------- prefill order
# Two OFF-BY-DEFAULT prefill switches. Contract: ./tmp/chatfix/cone/CONE_DESIGN.md.
# Neither implies the other and neither changes the default path.
#
#   DSV41_PREFILL_LAYER_MAJOR=1
#       Reorder a multi-token prefill into two phases: every DSV41_PREFILL_CHUNK-wide
#       chunk through layers [0, boundary) first, carrying (h, pre_mix); then every
#       chunk through [boundary, n_layers). Decoder layers still run at FULL width, so
#       this is pure loop reordering and is expected to be bit-identical: chunk k's
#       layer L sees exactly the cache contents it sees today, and the per-chunk
#       SharedState is reset at the boundary layer, which republishes every field the
#       layers above it read (see prefill_boundary_layer).
#
#   DSV41_PREFILL_CONE=1   (requires DSV41_PREFILL_LAYER_MAJOR=1)
#       Replaces phase two with the causal cone of the LAST prompt position. Decoder
#       layers read a `sliding_window`-wide window plus a compressed bank the boundary
#       layer publishes once over the whole prompt, so the cone widens by exactly
#       `window - 1` rows per layer. Exact for next-token prediction, not an
#       approximation -- but it produces hidden states only for the cone rows.
#
#   DSV41_PREFILL_CONE_TAIL=<W>   (default 1; only meaningful with the cone)
#       How many TOP-layer rows the cone targets. W = 1 is next-token prediction and
#       is exactly the formula above. W > 1 moves every first(il) earlier by W - 1,
#       costing (n_layers - boundary) * (W - 1) extra rows in total and still CONSTANT
#       in prompt length, and it cannot change the final row's result because that
#       row's inputs are untouched. It exists so a consumer that reads more than the
#       last row -- the DSpark drafter warms from the last `sliding_window` capture
#       rows -- is served exactly instead of refused. Setting it lifts the DSpark and
#       capture_layer_ids refusals and NOTHING else: it never switches DSpark on or
#       off, and DSpark never sets it. Contract: ./tmp/chatfix/cone/DSPARK_AND_THE_CONE.md.
#
# DSV41_PREFILL_CHUNK is read here with the receiver's own 1..4096 validation
# (mlx_token_stream_server.py:2734) so the two cannot drift apart silently.


def _prefill_chunk_size():
    """DSV41_PREFILL_CHUNK, validated exactly as the receiver validates it."""
    raw = os.environ.get("DSV41_PREFILL_CHUNK", "256")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"DSV41_PREFILL_CHUNK must be in 1..4096, got {raw!r}") from None
    if not 1 <= value <= 4096:
        raise ValueError(f"DSV41_PREFILL_CHUNK must be in 1..4096, got {value}")
    return value


def prefill_boundary_layer(config):
    """The first full-resolution decoder layer, DERIVED from the loaded config.

    It is the first layer whose `compress_ratios` entry is 1. The two-phase split
    resets SharedState at this layer, which is sound ONLY because this one layer
    republishes every per-forward field the layers above it read: the compressed KV
    owner, the index-key owner, the top-k selection and the candidate mask. Each of
    those is checked here and named in the failure, because a config where one of
    them is owned BELOW the boundary would make a fresh phase-two SharedState read a
    None or a stale pointer, and the bit-identity claim would be false. There is no
    workaround: carrying phase one's SharedState into phase two would reintroduce the
    cross-chunk state the split exists to avoid, so this raises instead.
    """
    n_layers = config.num_hidden_layers
    ratios = list(config.compress_ratios)[:n_layers]
    boundary = next((i for i, ratio in enumerate(ratios) if ratio == 1), None)
    if boundary is None:
        raise ValueError(
            "DSV41_PREFILL_LAYER_MAJOR: compress_ratios has no entry equal to 1 among "
            f"the {n_layers} backbone layers, so there is no decoder boundary to split at")
    if any(ratio != 1 for ratio in ratios[boundary:]):
        raise ValueError(
            f"DSV41_PREFILL_LAYER_MAJOR: compress_ratios is not 1 for every layer from "
            f"the derived boundary {boundary} upwards: {ratios[boundary:]}")
    kv_sources = list(config.kv_source_layer_ids)
    if not kv_sources or boundary != max(kv_sources):
        raise ValueError(
            f"DSV41_PREFILL_LAYER_MAJOR: kv_source_layer_ids -- the derived boundary is "
            f"{boundary} but the last compressed-KV source is {max(kv_sources) if kv_sources else None}; "
            "phase two would start with no KV owner in its SharedState")
    if boundary not in list(config.index_source_layer_ids):
        raise ValueError(
            f"DSV41_PREFILL_LAYER_MAJOR: index_source_layer_ids -- the derived boundary "
            f"{boundary} is not an index source, so phase two would start with no "
            "index-key owner and no top-k selection in its SharedState")
    if boundary != config.candidate_source_layer_id:
        raise ValueError(
            f"DSV41_PREFILL_LAYER_MAJOR: candidate_source_layer_id is "
            f"{config.candidate_source_layer_id}, not the derived boundary {boundary}; "
            "phase two would start with no candidate mask in its SharedState")
    return boundary


def cone_first_row(n_tokens, layer_id, boundary, n_layers, window, tail=1):
    """First prompt row layer `layer_id` must COMPUTE for the last `tail` positions.

        first(il) = n_tokens - (tail + (n_layers - 1 - il) * (window - 1)),  min 0

    `tail` is DSV41_PREFILL_CONE_TAIL: the number of TOP-layer rows the cone is
    asked to produce. tail=1 is next-token prediction and reproduces the original
    formula exactly. A wider tail only moves first(il) EARLIER, so the final row's
    inputs are untouched and its result cannot change.

    Exactness at every one of the `tail` rows: row r of layer il reads its own KV
    over [r - (window - 1), r], and this layer's KV is published over
    [first(il - 1), n) = [first(il) - (window - 1), n), which covers it for every
    r >= first(il). The recurrence bottoms out at the boundary layer, whose KV the
    cone's own phase publishes over the WHOLE prefix [0, n).

    Layers below the boundary always start at row 0; first(boundary - 1) = 0 is the
    base case the recurrence is built on.
    """
    if layer_id < boundary:
        return 0
    return max(n_tokens - (int(tail) + (n_layers - 1 - layer_id) * (window - 1)), 0)


def cone_total_rows(n_layers, boundary, window, tail=1):
    """Decoder rows a cone of `tail` top rows costs, CONSTANT in prompt length."""
    n_decoder = n_layers - boundary
    return int(tail) * n_decoder + (window - 1) * n_decoder * (n_decoder - 1) // 2


def prefill_cone_tail():
    """(tail, explicitly_set) for DSV41_PREFILL_CONE_TAIL.

    Default 1 -- the cone targets the one row next-token prediction needs. A wider
    tail exists so a consumer that reads MORE than the last row (the DSpark drafter
    warms from the target's hidden/capture rows) can be served exactly instead of
    refused. `explicitly_set` is what lifts those refusals: an unset tail must never
    be read as consent to truncate somebody's history.
    """
    raw = os.environ.get("DSV41_PREFILL_CONE_TAIL")
    if raw is None:
        return 1, False
    try:
        tail = int(raw)
    except ValueError:
        raise ValueError(
            f"DSV41_PREFILL_CONE_TAIL must be an integer >= 1, got {raw!r}") from None
    if tail < 1:
        raise ValueError(
            f"DSV41_PREFILL_CONE_TAIL must be >= 1, got {tail}: the cone must always "
            "produce at least the one row next-token prediction needs")
    return tail, True


def _prefill_switches(n_tokens, cache, capture_layer_ids):
    """Resolve the two prefill-ordering switches, or refuse, naming what and why.

    Returns (layer_major, cone). Both are False unless explicitly set to 1, and
    neither ever sets the other: DSV41_PREFILL_CONE without
    DSV41_PREFILL_LAYER_MAJOR is refused, never promoted.
    """
    layer_major = os.environ.get("DSV41_PREFILL_LAYER_MAJOR", "0") == "1"
    cone = os.environ.get("DSV41_PREFILL_CONE", "0") == "1"
    if cone and not layer_major:
        raise ValueError(
            "DSV41_PREFILL_CONE=1 requires DSV41_PREFILL_LAYER_MAJOR=1: the cone "
            "replaces phase two of the layer-major split and there is no phase two "
            "without DSV41_PREFILL_LAYER_MAJOR. Set both, or neither. Setting "
            "DSV41_PREFILL_CONE will not switch DSV41_PREFILL_LAYER_MAJOR on for you.")
    tail, tail_set = prefill_cone_tail()
    if cone and os.environ.get("DSV41_DSPARK", "0") == "1" and not tail_set:
        raise ValueError(
            "DSV41_PREFILL_CONE=1 with DSV41_DSPARK=1 needs DSV41_PREFILL_CONE_TAIL=<W> "
            "set explicitly. The drafter is warmed from the target's rows at the END of "
            "the prompt -- DSV4.1 DSpark from the capture taps of layers "
            "dspark_target_layer_ids, keeping the last sliding_window of them; a "
            "hidden-state drafter from prefill_from_target_hidden -- and a bare cone "
            "produces ONE row. Set DSV41_PREFILL_CONE_TAIL to the number of rows the "
            "drafter should be warmed with (sliding_window or more for DSpark); the "
            "receiver logs the count it actually used. The cone will not truncate the "
            "drafter's history behind your back and will not disable DSpark for you, "
            "and setting the tail does not switch DSpark on or off.")
    if not layer_major or n_tokens <= 1:
        return False, False
    if getattr(cache, "bounded", False):
        raise ValueError(
            "DSV41_PREFILL_LAYER_MAJOR=1 cannot run with DSV41_BOUNDED_CACHE=1: the "
            "layer-major order takes the whole prompt in ONE forward, so the bounded "
            "cache retires nothing until the end and would not do what it says; the "
            "cone additionally reserves window-KV rows it never computes, which "
            "position-keyed retirement must not be asked to interpret. Switch one off.")
    if cone and capture_layer_ids and not tail_set:
        raise ValueError(
            "DSV41_PREFILL_CONE=1 cannot serve capture_layer_ids="
            f"{list(capture_layer_ids)} without DSV41_PREFILL_CONE_TAIL=<W>: a capture "
            "tap reads the attention input at EVERY prompt position and a bare cone "
            "computes one row at the top layer. Set the tail to the number of rows the "
            "tap consumer needs; every tap is then delivered over exactly those last W "
            "positions.")
    return layer_major, cone


class DeepseekV41Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layout = EngramLayout.from_config(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV41Block(i, config, self.layout)
                       for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _run_layer_span(self, layer_ids, h, pre_mix, start_pos, cache, shared,
                        hashes, token_mask, image_mask, capture_set, captures,
                        kv_already_prepared=False):
        """The layer loop, over an arbitrary contiguous span of layers.

        The default path passes the whole stack and one SharedState, so it runs the
        original loop verbatim; the layer-major and cone drivers reuse it so the two
        can never drift apart. `captures` collects the DSpark taps per layer id
        instead of appending straight to the sink, so a chunked run can concatenate
        them back into one entry per layer.
        """
        n = h.shape[1]
        layer_ids = list(layer_ids)
        prefetch: dict = {}
        cut_at = None
        if _STEP_OVERLAP and (n == 1 or _STEP_OVERLAP_PREFILL) and hashes is not None:
            engram_layers = [(i, self.layers[i]) for i in layer_ids
                             if self.layers[i].engram is not None]
            if len(engram_layers) > 1:
                cut_at = engram_layers[1][0]
                pool = _prefetch_pool()
                for i, layer in engram_layers[1:]:
                    if layer.engram.source != "ram":
                        prefetch[i] = pool.submit(layer.engram.rows_prepare,
                                                  hashes[:, :, layer.engram.layer_hash_index, :])
        for i in layer_ids:
            layer = self.layers[i]
            if layer.engram is not None:
                if i == cut_at:
                    # The GPU starts on layers 0..i-1 now; the host joins the prefetch and builds the rest.
                    mx.async_eval(h, pre_mix)
                fut = prefetch.pop(i, None)
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], token_mask,
                                 prepared=None if fut is None else fut.result())
                if _PREFILL_LEDGER_CUTS and n > 1:
                    _t0 = time.perf_counter()
                    mx.eval(h, pre_mix)
                    mx.synchronize()
                    _prefill_layer_add(i, "engram_ms", (time.perf_counter() - _t0) * 1000.0)
            # The DSpark (MTP) head reads the ATTENTION INPUT of its target
            # layers, not their output: the reference taps h here, after any
            # engram contribution and before the block runs. The lanes are kept
            # ([b, n, hc, d]); the drafter's context_from_captures means them.
            if i in capture_set:
                captures.setdefault(i, []).append(h)
            h, pre_mix = layer(h, pre_mix, start_pos, cache, shared, image_mask,
                               kv_already_prepared)
            if _LEDGER_CUTS and n == 1:
                _t0 = time.perf_counter()
                mx.eval(h, pre_mix)
                mx.synchronize()
                LEDGER_LAYER_MS[i] = LEDGER_LAYER_MS.get(i, 0.0) + (time.perf_counter() - _t0) * 1000.0
        return h, pre_mix

    @staticmethod
    def _slice_rows(value, start, stop):
        """Row slice of a [b, n, ...] mx array or an [B, L, ...] numpy array, or None."""
        return None if value is None else value[:, start:stop]

    def _prefill_layer_major(self, h, pre_mix, start_pos, cache, hashes, token_mask,
                             image_mask, capture_set, captures, cone):
        """DSV41_PREFILL_LAYER_MAJOR, and DSV41_PREFILL_CONE on top of it."""
        n = h.shape[1]
        n_layers = len(self.layers)
        boundary = prefill_boundary_layer(self.config)
        window = self.config.sliding_window
        tail, _ = prefill_cone_tail()
        chunk = _prefill_chunk_size()
        bounds = [(begin, min(begin + chunk, n)) for begin in range(0, n, chunk)]
        # One chunk needs no boundary eval: it would only insert a synchronisation a
        # short forward (a DSpark verify step, a tiny prompt) does not need.
        cut = len(bounds) > 1

        def span(layer_ids, hx, px, row0, shared, kv_already_prepared=False):
            stop = row0 + hx.shape[1]
            return self._run_layer_span(
                layer_ids, hx, px, start_pos + row0, cache, shared,
                self._slice_rows(hashes, row0, stop),
                self._slice_rows(token_mask, row0, stop),
                self._slice_rows(image_mask, row0, stop),
                capture_set, captures, kv_already_prepared)

        # Phase one: every chunk through the compressed half, one fresh SharedState
        # per chunk exactly as today. mx.eval at each boundary keeps the graph bounded.
        carry_h, carry_pre = [], []
        for begin, end in bounds:
            chunk_h, chunk_pre = span(range(boundary), h[:, begin:end],
                                      pre_mix[:, begin:end], begin, SharedState())
            if cut:
                mx.eval(chunk_h, chunk_pre)
            carry_h.append(chunk_h)
            carry_pre.append(chunk_pre)
        # The carry is [b, N, hc, D] in the ACTIVATION dtype. The design doc asks for
        # fp16; casting here would change the numbers and break the bit-identity gate
        # that is this switch's whole purpose, so the carry keeps the dtype the default
        # path produces. Both are two bytes an element for the release checkpoint.
        h = carry_h[0] if len(carry_h) == 1 else mx.concatenate(carry_h, axis=1)
        pre_mix = carry_pre[0] if len(carry_pre) == 1 else mx.concatenate(carry_pre, axis=1)
        del carry_h, carry_pre

        if cone and cone_first_row(n, boundary, boundary, n_layers, window, tail) <= 0:
            # NOT a silent fallback: at this length the cone already covers every row,
            # so the full phase two IS the cone and the result is identical. Say so.
            reach = tail + (n_layers - 1 - boundary) * (window - 1)
            print(f"[DSV41] PREFILL_CONE: prompt of {n} rows is no longer than the "
                  f"boundary layer's cone ({reach} rows at DSV41_PREFILL_CONE_TAIL="
                  f"{tail}); running the full decoder half over all {n} rows -- "
                  f"identical result, no rows skipped", flush=True)
            cone = False

        if not cone:
            # Phase two at FULL width: chunk-major again, one fresh SharedState per
            # chunk, which the boundary layer repopulates before anything reads it.
            out_h, out_pre = [], []
            for begin, end in bounds:
                chunk_h, chunk_pre = span(range(boundary, n_layers), h[:, begin:end],
                                          pre_mix[:, begin:end], begin, SharedState())
                if cut:
                    mx.eval(chunk_h, chunk_pre)
                out_h.append(chunk_h)
                out_pre.append(chunk_pre)
            h = out_h[0] if len(out_h) == 1 else mx.concatenate(out_h, axis=1)
            pre_mix = out_pre[0] if len(out_pre) == 1 else mx.concatenate(out_pre, axis=1)
            return h, pre_mix

        # Phase two as the cone. The boundary layer publishes its window KV, the
        # compressor bank and the index keys over the WHOLE prompt, chunk by chunk,
        # with a throwaway SharedState per chunk exactly as phase one uses.
        boundary_cache = cache.layers[boundary]
        for begin, end in bounds:
            self.layers[boundary].prepare_attn_kv(
                h[:, begin:end], pre_mix[:, begin:end], start_pos + begin, cache,
                SharedState())
            mx.eval(*[value for value in (boundary_cache.win_kv, boundary_cache.kv,
                                          boundary_cache.score, boundary_cache.comp_kv,
                                          boundary_cache.index_k) if value is not None])

        shared = SharedState()
        rows = cone_first_row(n, boundary, boundary, n_layers, window, tail)
        x, x_pre = h[:, rows:], pre_mix[:, rows:]
        x, x_pre = span([boundary], x, x_pre, rows, shared, kv_already_prepared=True)
        mx.eval(x, x_pre)
        # The HIGHEST row any decoder layer starts its real KV at. The top layer's,
        # since first(il) rises with il, and it is what decides how far back this
        # cache may later be rewound: see DeepseekV41Cache.note_cone_prefill.
        highest_publish = None
        for il in range(boundary + 1, n_layers):
            previous, rows = rows, cone_first_row(n, il, boundary, n_layers, window, tail)
            highest_publish = previous
            # This layer publishes KV over [first(il-1), N) -- its own cone plus the
            # window the next layer and every later decode step read. `x` is exactly
            # those rows. The rows below them never existed as activations above the
            # boundary, so their cache slots are reserved, never computed.
            self.layers[il].attn.reserve_kv_rows(cache, previous, x.shape[0], x.dtype)
            self.layers[il].prepare_attn_kv(x, x_pre, start_pos + previous, cache, shared)
            x, x_pre = x[:, rows - previous:], x_pre[:, rows - previous:]
            x, x_pre = span([il], x, x_pre, rows, shared, kv_already_prepared=True)
            mx.eval(x, x_pre)
        # With no layer above the boundary nothing was reserved, so nothing is out of
        # bounds for a later rewind and the floor stays where it was.
        floor = 0 if highest_publish is None else start_pos + highest_publish + window - 1
        cache.note_cone_prefill(start_pos + n, floor)
        # Every tap the cone collected covers its own layer's cone, so the taps have
        # DIFFERENT row counts and could not be concatenated. Cut all of them to the
        # last `tail` positions -- the rows the switch promised and the only rows a
        # tap consumer is entitled to -- so every tap means the same positions.
        for layer_id, rows_list in list(captures.items()):
            block = rows_list[0] if len(rows_list) == 1 else mx.concatenate(rows_list, axis=1)
            captures[layer_id] = [block[:, -tail:]]
        return x, x_pre

    def __call__(self, inputs, cache, inputs_embeds=None, hidden_sink=None, skip_final_norm=False,
                 capture_layer_ids=None, capture_sink=None, image_mask=None):
        layer_major, cone = _prefill_switches(
            inputs.shape[1] if inputs is not None else inputs_embeds.shape[1],
            cache, capture_layer_ids)
        start_pos = cache.offset
        hashes = None
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if _PREFILL_LEDGER_ON and h.shape[1] > 1:
            _prefill_hash_ms = 0.0
            _prefill_hash0 = LEDGER["hash_host_ms"]
            _prefill_fetch0 = _ENGRAM_LEDGER["engram_host_ms"]
        # Launch the embedding while the CPU hash/table path prepares its rows.
        # Kept independently switchable so its contribution can be measured.
        if os.environ.get("DSV41_ASYNC_PREPARE", "0") == "1":
            mx.async_eval(h)
        token_mask = None
        if cache.engram is not None:
            if image_mask is not None and os.environ.get("DSV41_COALESCE_HASH_INPUTS", "0") == "1":
                # One device-to-host boundary instead of separate ids/mask waits.
                packed = np.array(mx.concatenate([inputs, image_mask.astype(inputs.dtype)], axis=0))
                ids, token_mask = packed[:inputs.shape[0]], ~packed[inputs.shape[0]:].astype(bool)
            else:
                ids = np.array(inputs)
                token_mask = None if image_mask is None else ~np.array(image_mask, dtype=bool)
            if _LEDGER_ON:
                _t0 = time.perf_counter()
                hashes = cache.engram.hash_state.hashes(ids, start_pos, token_mask)
                LEDGER["hash_host_ms"] += (time.perf_counter() - _t0) * 1000.0
                LEDGER["hash_calls"] += 1
            elif _PREFILL_LEDGER_ON and h.shape[1] > 1:
                _t0 = time.perf_counter()
                hashes = cache.engram.hash_state.hashes(ids, start_pos, token_mask)
                _prefill_hash_ms = (time.perf_counter() - _t0) * 1000.0
            else:
                hashes = cache.engram.hash_state.hashes(ids, start_pos, token_mask)
        b, n, dim = h.shape
        hc = self.config.hc_mult
        h = mx.broadcast_to(h[:, :, None, :], (b, n, hc, dim))
        pre_mix = mx.broadcast_to((mx.arange(hc) == 0).astype(mx.float32), (b, n, hc))
        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        if os.environ.get("DSV41_COMPILE_BLOCK", "0") == "1":
            from .block_graph import BlockGraphs
            if getattr(self, "_block_graph", None) is None:
                object.__setattr__(self, "_block_graph", BlockGraphs())
                for layer in self.layers:
                    object.__setattr__(layer, "_block_graph", self._block_graph)
            self._block_graph.configure()
        captures: dict = {}
        if layer_major:
            h, pre_mix = self._prefill_layer_major(
                h, pre_mix, start_pos, cache, hashes, token_mask, image_mask,
                capture_set, captures, cone)
        else:
            h, pre_mix = self._run_layer_span(
                range(len(self.layers)), h, pre_mix, start_pos, cache, SharedState(),
                hashes, token_mask, image_mask, capture_set, captures)
        if capture_sink is not None:
            # One entry per capture layer, in layer order, whatever the chunking was.
            capture_sink.extend(rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=1)
                                for _, rows in sorted(captures.items()))
        if _LEDGER_CUTS and n == 1:
            LEDGER["cut_steps"] += 1
        graph = getattr(self, "_block_graph", None)
        if graph is not None and n == 1 and os.environ.get("DSV41_GRAPH_REPORT", "0") == "1":
            graph.report(len(self.layers))
        h = hc_pre(h, pre_mix)
        if hidden_sink is not None:
            hidden_sink.append(h)
        if not skip_final_norm:
            h = self.norm(h)
        # Retire bounded histories eagerly, outside compiled block capture.
        cache.finish(start_pos + n)
        if _PREFILL_LEDGER_ON and n > 1:
            if _LEDGER_ON:
                _prefill_hash_ms = LEDGER["hash_host_ms"] - _prefill_hash0
            _prefill_ledger_chunk(n, _prefill_hash_ms,
                                  _ENGRAM_LEDGER["engram_host_ms"] - _prefill_fetch0)
        return h


class LanguageModel(nn.Module):
    # Partial compressor groups are retained, but shape-dependent matmul rounding
    # can flip fake-quant codes (see test_capacity_growth_300_decode_tokens).
    # Short chunk tests do not establish 32k sparse-selection/recall parity.
    no_chunked_prefill = True
    from .native_guard import NATIVE_STATUS as native_kernel_status

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = self.args = config
        self.model = DeepseekV41Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Local import avoids the stage module's shared-block import cycle.
        from .dspark import DSparkStack, n_dspark_stages

        if n_dspark_stages(config):
            self.dspark = DSparkStack(config)
        self._dspark_verify_calls = 0
        self._dspark_verify_shapes = {}
        self._token_map = None
        if config.model_dir and self.model.layout is not None:
            self._load_token_map(config.model_dir)

    def set_token_map(self, token_map):
        token_map = np.asarray(token_map, dtype=np.int64)
        if token_map.shape != (self.config.vocab_size,):
            raise ValueError("Engram token map must cover the vocabulary")
        if token_map.min() < 0 or token_map.max() + 1 != self.config.engram_compressed_vocab_size:
            raise ValueError("Engram token map disagrees with compressed vocabulary size")
        self._token_map = token_map.copy()

    def _load_token_map(self, model_dir):
        token_map, size = load_compressed_token_map(
            Path(model_dir) / "tokenizer.json", self.config.engram_compressed_vocab_size)
        if size != self.config.engram_compressed_vocab_size:
            raise ValueError("Cached Engram token map has the wrong vocabulary size")
        self.set_token_map(token_map)

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        if inputs is None and self.model.layout is not None:
            raise ValueError("Engram hashing requires token ids in inputs, even with inputs_embeds")
        if inputs is None and inputs_embeds is None:
            raise ValueError("Provide inputs token ids or inputs_embeds")
        shape = inputs.shape if inputs is not None else inputs_embeds.shape[:2]
        if len(shape) != 2 or shape[1] == 0:
            raise ValueError("inputs must have nonempty shape [batch, sequence]")
        if shape[0] != 1:
            raise NotImplementedError("DeepSeek-V4.1 currently supports batch one")
        if kwargs.get("mask") is not None:
            raise NotImplementedError("DeepSeek-V4.1 attention masks are not implemented")
        image_mask = kwargs.get("image_mask")
        if image_mask is None and inputs is not None and hasattr(self, "image_token_id"):
            # Derive from this chunk's real ids, including during direct LM
            # calls by the generic prefill/decode loop; no stale prompt mask.
            image_mask = inputs == self.image_token_id
        if image_mask is not None and (image_mask.shape != shape or image_mask.dtype != mx.bool_):
            raise ValueError("image_mask must be boolean and match inputs")
        if inputs_embeds is not None and inputs_embeds.shape != (*shape, self.config.hidden_size):
            raise ValueError("inputs_embeds must match inputs and hidden_size")
        if cache is None:
            cache = self.make_cache()
        if not isinstance(cache, (list, tuple)) or len(cache) != 1 or not isinstance(cache[0], DeepseekV41Cache):
            raise ValueError("Expected the single-element V4.1 cache list")
        hidden_sink = kwargs.get("hidden_sink")
        if kwargs.get("return_hidden", False) and hidden_sink is None:
            hidden_sink = []
        capture_layer_ids = kwargs.get("capture_layer_ids")
        capture_sink = [] if capture_layer_ids is not None else None
        hidden = self.model(inputs, cache[0], inputs_embeds, hidden_sink,
                            kwargs.get("skip_final_norm", False),
                            capture_layer_ids=capture_layer_ids, capture_sink=capture_sink,
                            image_mask=image_mask)
        logits = None
        if not kwargs.get("skip_logits", False):
            head_in = hidden
            if 1 < head_in.shape[1] < shape[1]:
                # A DSV41_PREFILL_CONE_TAIL > 1 prefill: the model returned the last W
                # rows because a drafter wants their HIDDEN states, not their logits.
                # Only the final row is ever sampled, and every caller reads
                # logits[0, -1, :], so keep the head off the other W - 1 rows rather
                # than buy a W x vocab fp32 GEMM nobody reads. (At W = 1 there is
                # nothing to cut, and a full-width prefill has shape[1] rows here.)
                head_in = head_in[:, -1:]
            if (os.environ.get("DSV41_OMLX_HEAD", "0") == "1"
                    and type(self.lm_head) is nn.Linear and "bias" not in self.lm_head):
                from .omlx_head import project_logits

                logits = project_logits(head_in, self.lm_head.weight)
            else:
                logits = self.lm_head(head_in.astype(mx.float32))
        hidden_states = None
        if capture_sink is not None or hidden_sink is not None:
            # DSpark captures first, then any final-hidden sink, as in V4.
            hidden_states = list(capture_sink or []) + list(hidden_sink or [])
        return LanguageModelOutput(logits=logits,
                                   hidden_states=hidden_states,
                                   shared_kv_states=None)

    # ------------------------------------------------------- speculative hooks
    def make_dspark_drafter(self):
        """Bind the resident stages without allocating another set of weights.

        Receivers opt in by passing this object as draft_model, draft_kind='dspark',
        draft_block_size=config.dspark_block_size. Omitting it disables speculation.
        """
        from ...speculative.drafters.deepseek_v41_dspark import Model, ModelConfig

        if not hasattr(self, "dspark"):
            raise ValueError("The loaded checkpoint does not contain DSpark stages")
        return Model(ModelConfig.from_dict({"text_config": self.config}),
                     stack=self.dspark).bind(self)

    def speculative_verify_captures(self, inputs, cache, capture_layer_ids, sampler=None):
        """Verify pass for the DSpark block drafter.

        ONE target forward that both advances the conversation cache and taps the
        ``capture_layer_ids`` attention inputs the drafter needs for its next
        round's ``main_hidden``. Returns ``(captures, logits, rollback_state,
        target_tokens)``, the tuple ``speculative/dspark.py`` expects.

        ``rollback_state`` is always ``None``: unlike V4, whose pooled
        compressed-KV caches can need a replay snapshot,
        :class:`DeepseekV41Cache` trims exactly -- it drops invalid groups and
        recomputes the last complete one from the retained fp32 compressor
        projections -- so a plain trim restores the committed state.
        """
        self._dspark_verify_calls += 1
        shape = tuple(inputs.shape)
        self._dspark_verify_shapes[shape] = self._dspark_verify_shapes.get(shape, 0) + 1
        with _v41_verification():
            out = self(inputs, cache=cache, capture_layer_ids=list(capture_layer_ids))
        target_tokens = sampler(out.logits) if sampler is not None else None
        return out.hidden_states, out.logits, None, target_tokens

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size) -> int:
        """Drop the rejected tail of a verify pass from the conversation cache.

        ``block_size`` is the verify length (``1 + proposal``) and ``accepted``
        the number of accepted drafts, so ``accepted + 1`` cells stay committed
        (the anchor plus the accepted drafts) and the rest are trimmed.
        """
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        values = [int(a) for a in accepted.tolist()]
        if len(set(values)) != 1:
            raise ValueError("DeepSeek-V4.1 speculative rollback requires uniform acceptance")
        max_a = values[0]
        trim = int(block_size) - (max_a + 1)
        if trim < 0:
            raise ValueError("Accepted more tokens than the verify pass wrote")
        if trim:
            for cache in caches:
                if cache is not None:
                    cache.trim(trim)
        return max_a

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [DeepseekV41Cache(self.model, self._token_map)]

    def attach_engram_storage(self, model_dir):
        if _HEAD_BF16 and os.environ.get("DSV41_OMLX_HEAD", "0") != "1":
            raise RuntimeError("DSV41_HEAD_BF16=1 needs DSV41_OMLX_HEAD=1 (bf16 head storage with "
                               "fp32 logits); refusing to serve bf16 logits silently")
        if not hasattr(self, "_index_fused_attached"):
            index_flag = os.environ.get("DSV41_INDEX_FUSED", "0")
            if index_flag not in ("0", "1"):
                raise ValueError("DSV41_INDEX_FUSED must be 0 or 1")
            skip_flag = os.environ.get("DSV41_INDEX_SKIP", "0")
            if skip_flag not in ("0", "1"):
                raise ValueError("DSV41_INDEX_SKIP must be 0 or 1")
            if skip_flag == "1" and index_flag != "1":
                raise ValueError("DSV41_INDEX_SKIP=1 requires DSV41_INDEX_FUSED=1")
            # Loud at attach, not on the first long call: the per-call gate
            # below reads the same env var through the same validator.
            skip_min_nb = _index_skip_min_nb()
            if skip_flag == "1":
                print(f"[DSV41] INDEX_SKIP armed, engages from nb={skip_min_nb}",
                      flush=True)
            if index_flag == "1":
                for name in ("DSV41_PREFILL_TILED_INDEX", "DSV41_PREFILL_V41_INDEX",
                             "DSV41_METAL_INDEXER"):
                    if os.environ.get(name, "0") == "1":
                        raise ValueError(f"DSV41_INDEX_FUSED cannot combine with {name}")
                from ...custom_kernels.deepseek_v41.index_fused import prepare_index_fused

                # Compile failure is fatal before latching.
                prepare_index_fused(self.layers, skip=skip_flag == "1")
            self._index_fused_attached = index_flag == "1"
            for layer in self.layers:
                if hasattr(getattr(layer, "attn", None), "indexer"):
                    layer.attn.indexer.index_fused = self._index_fused_attached
        if not hasattr(self, "_ffn_fused_attached"):
            fused_flag = os.environ.get("DSV41_FFN_FUSED", "0")
            if fused_flag not in ("0", "1"):
                raise ValueError("DSV41_FFN_FUSED must be 0 or 1")
            if fused_flag == "1" and (
                    getattr(self, "_ffn_down_bf16_attached",
                            os.environ.get("DSV41_FFN_DOWN_BF16", "0") == "1")
                    or getattr(self, "_ffn_expert_pad16_attached",
                               os.environ.get("DSV41_FFN_EXPERT_PAD16", "0") == "1")
                    or getattr(self, "_ffn_expert_tiles_attached",
                               os.environ.get("DSV41_FFN_EXPERT_TILES", "0") == "1")
                    or os.environ.get("DSV41_PREFILL_GROUPED_MXFP4", "0") == "1"):
                raise ValueError("DSV41_FFN_FUSED requires DOWN_BF16, EXPERT_PAD16, "
                                 "EXPERT_TILES and PREFILL_GROUPED_MXFP4 to be off")
        if not hasattr(self, "_ffn_down_bf16_attached"):
            self._ffn_down_bf16_attached = os.environ.get("DSV41_FFN_DOWN_BF16", "0") == "1"
            for layer in self.layers:
                layer.ffn.ffn_down_bf16 = self._ffn_down_bf16_attached
        if not hasattr(self, "_ffn_expert_pad16_attached"):
            self._ffn_expert_pad16_attached = os.environ.get("DSV41_FFN_EXPERT_PAD16", "0") == "1"
            for layer in self.layers:
                layer.ffn.ffn_expert_pad16 = self._ffn_expert_pad16_attached
        if not hasattr(self, "_ffn_expert_tiles_attached"):
            flag = os.environ.get("DSV41_FFN_EXPERT_TILES", "0")
            if flag not in ("0", "1"):
                raise ValueError("DSV41_FFN_EXPERT_TILES must be 0 or 1")
            if flag == "1":
                if (self._ffn_down_bf16_attached or self._ffn_expert_pad16_attached
                        or os.environ.get("DSV41_PREFILL_GROUPED_MXFP4", "0") == "1"):
                    raise ValueError("DSV41_FFN_EXPERT_TILES requires DOWN_BF16, EXPERT_PAD16 "
                                     "and PREFILL_GROUPED_MXFP4 to be off")
                from ...custom_kernels.deepseek_v41.moe_prefill_expert_tiles import prepare_expert_tiles

                prepare_expert_tiles(self.layers)  # Compile failure is fatal before latching.
            self._ffn_expert_tiles_attached = flag == "1"
            for layer in self.layers:
                layer.ffn.ffn_expert_tiles = self._ffn_expert_tiles_attached
        if not hasattr(self, "_ffn_fused_attached"):
            if fused_flag == "1":
                from ...custom_kernels.deepseek_v41.moe_prefill_fused_ffn import prepare_fused_ffn

                prepare_fused_ffn(self.layers)  # Compile failure is fatal before latching.
            self._ffn_fused_attached = fused_flag == "1"
            for layer in self.layers:
                layer.ffn.ffn_fused = self._ffn_fused_attached
        if os.environ.get("DSV41_PROMOTE_DECODE_WEIGHTS", "0") == "1":
            # These contractions already execute in FP32. Retaining that exact
            # representation removes full weight conversions on every token.
            # This deliberately spends 3.73 GiB for the release checkpoint.
            # DSV41_WO_A_BF16 / DSV41_HEAD_BF16 leave the weight bf16 and read half the bytes.
            with mx.stream(mx.cpu):
                if not _WO_A_BF16:
                    for layer in self.layers:
                        layer.attn.wo_a.weight = layer.attn.wo_a.weight.astype(mx.float32)
                        mx.eval(layer.attn.wo_a.weight)
                if not _HEAD_BF16:
                    self.lm_head.weight = self.lm_head.weight.astype(mx.float32)
                    mx.eval(self.lm_head.weight)
        if self.model.layout is not None and self._token_map is None:
            self._load_token_map(model_dir)
        for layer in self.layers:
            if layer.engram is not None:
                layer.engram.attach_storage(model_dir)
                layer.engram.warm()

    def engram_stats(self):
        return {i: layer.engram.stats() for i, layer in enumerate(self.layers)
                if layer.engram is not None}

    def save_engram_hot(self):
        return {i: layer.engram.save_hot() for i, layer in enumerate(self.layers)
                if layer.engram is not None}

    def sanitize(self, weights):
        return sanitize_weights_v41(weights, self.config, self.config.engram_source)

    @property
    def quant_predicate(self):
        quantization_config = make_quantization_config_v41(self)

        def predicate(path, _):
            path = path.removeprefix("language_model.")
            protected = (".wo_a", ".engram.embed", ".compressor", ".indexer.wk",
                         ".weights_proj", ".attn_hc", ".ffn_hc", ".ffn.gate")
            if any(name in path for name in protected):
                return False
            return quantization_config.get(path, True)

        return predicate

    @property
    def cast_predicate(self):
        def predicate(key):
            protected = ("attn_sink", "e_score_correction_bias", "bias_vl", ".attn_hc.",
                         ".ffn_hc.", "q_weight", "k_weight", ".compressor.", "weights_proj")
            return not any(name in key for name in protected)

        return predicate

    def pipeline(self, group):
        """Identity for one rank; reject unsupported staging before mutation."""
        from .distribution import require_single_rank

        require_single_rank(group, "pipeline")
        return self

    def shard(self, group=None):
        """Do not slice weights until the attention/MoE reductions are present."""
        from .distribution import require_single_rank

        if group is None:
            group = mx.distributed.init()
        require_single_rank(group, "shard")
        return self
