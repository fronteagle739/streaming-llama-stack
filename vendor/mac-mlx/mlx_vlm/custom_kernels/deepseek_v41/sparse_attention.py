"""Sink-aware sparse MQA, built lazily by MLX's Metal JIT.

Indices are the visibility contract: -1 is hidden, [0,NW) addresses window
rows, [NW,NW+NC) addresses compressed rows. The caller's real Indexer applies
group < floor((position+1)/ratio); do not impose single-timeline causality here.
All D channels (including RoPE) contribute to both K and V. No native fallback.
"""
from functools import lru_cache
import os
from pathlib import Path

import mlx.core as mx

_COUNTERS = {"calls": 0, "launches": 0, "fused": 0, "decode_streaming": 0, "prefill_flash": 0,
             "qk": 0, "softmax": 0, "pv": 0}


def status():
    return {"metal_available": mx.metal.is_available(), **_COUNTERS}


def prefill_flash_enabled(n):
    """Opt-in prefill threshold; decode/DSpark batches <=8 are always excluded."""
    return (n > 8 and os.environ.get("DSV41_PREFILL_FLASH_ATTN", "0") == "1"
            and n >= max(9, int(os.environ.get("DSV41_PREFILL_FLASH_MIN_N", "64"))))


@lru_cache(maxsize=1)
def _kernel():
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_METAL_SPARSE_ATTN=1 requires Metal; no fallback")
    return mx.fast.metal_kernel(
        name="dsv41_sparse_mqa_sink",
        input_names=["Q", "Window", "Compressed", "Indices", "Sink", "Scale", "K", "NW", "NC"],
        output_names=["Output"],
        source=Path(__file__).with_name("sparse_attention.metal").read_text(),
    )


@lru_cache(maxsize=1)
def _decode_kernel():
    return mx.fast.metal_kernel(
        name="dsv41_sparse_decode_streaming",
        input_names=["Q", "Window", "Compressed", "Indices", "Sink", "Scale", "K", "NW", "NC"],
        output_names=["Output"],
        source=Path(__file__).with_name("sparse_decode_simd.metal").read_text(),
    )


@lru_cache(maxsize=3)
def _stage(stage):
    inputs, outputs = {
        "qk": (["Q", "Window", "Compressed", "Indices", "Scale", "K", "NW", "NC"], ["Scores"]),
        "softmax": (["Scores", "Indices", "Sink", "K", "NW", "NC"], ["Weights", "Denom"]),
        "pv": (["Weights", "Denom", "Window", "Compressed", "Indices", "K", "NW", "NC"], ["Output"]),
    }[stage]
    return mx.fast.metal_kernel(
        name="dsv41_sparse_" + stage, input_names=inputs, output_names=outputs,
        source=Path(__file__).with_name("sparse_" + stage + ".metal").read_text(),
        header="#include <metal_simdgroup_matrix>\n",
    )


def sparse_attention(q, kv, sink, indices, scale):
    """[B,M,H,D] x shared [B,N,D]; production sparse H=64,D=512,K<=640.

    Reduced heads/widths are supported for synthetic differential gates.
    FP32 QK, exponentials, denominator and PV; final cast only. SIMD reduction
    order differs from MLX GEMM, so this experimental path remains opt-in.
    DSV41_METAL_SPARSE_DECODE=1 selects the single-launch streaming D=512
    decode kernel and permits K<=32768. It defaults off.
    DSV41_PREFILL_FLASH_ATTN=1 selects online prefill for BF16 banks, D=512,
    at N>=DSV41_PREFILL_FLASH_MIN_N (default 64, hard minimum 9).
    Both switches default off and apply only to their respective regimes.
    """
    from .v41_dispatch import enabled

    if q.ndim == 4 and enabled("ATTN", q.shape[1]):
        from .v41_attention import sparse_attn_v41

        return sparse_attn_v41(q, kv, sink, indices, scale)
    if q.ndim == 4 and prefill_flash_enabled(q.shape[1]):
        from .sparse_attention_prefill import sparse_attn_prefill

        out = sparse_attn_prefill(q, kv, sink, indices, scale)
        for name in ("calls", "fused", "prefill_flash"):
            _COUNTERS[name] += 1
        _COUNTERS["launches"] += 2  # Shared gather followed by online MMA attention.
        return out
    kernel = _kernel()  # Availability assertion occurs before any dispatch.
    if q.ndim != 4 or indices.ndim != 3:
        raise ValueError("expected Q[B,M,H,D] and indices[B,M,K]")
    b, m, h, d = q.shape
    k = indices.shape[-1]
    streaming = (os.environ.get("DSV41_METAL_SPARSE_DECODE", "0") == "1"
                 and m == 1 and d == 512)
    max_k = 32768 if streaming else 640
    if d % 32 or not 32 <= d <= 512 or not 1 <= k <= max_k or m < 1:
        raise ValueError(f"V4.1 Metal requires D=32..512 aligned to 32, 1<=K<={max_k}, M>=1")
    if indices.shape[:2] != (b, m) or sink.shape != (h,):
        raise ValueError("V4.1 Metal indices/sink shape mismatch")
    if q.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError("V4.1 Metal requires fp32/fp16/bf16 Q")
    if indices.dtype != mx.int32:
        raise ValueError("V4.1 Metal requires signed int32 indices (-1 is masked)")
    window, compressed = kv if isinstance(kv, tuple) else (kv, None)
    for bank in (window, compressed):
        if bank is not None and (bank.ndim != 3 or bank.shape[0] != b or
                                 bank.shape[2] != d or bank.dtype not in
                                 (mx.float32, mx.float16, mx.bfloat16)):
            raise ValueError("V4.1 Metal banks must be floating [B,N,D]")
    nc = 0 if compressed is None else compressed.shape[1]
    if compressed is None:
        compressed = mx.zeros((b, 1, d), dtype=window.dtype)
    # Cache lengths and selected width are runtime scalars: growing a cache must
    # not compile a new Metal library on every decode token.
    sizes = [mx.array(v, dtype=mx.int32) for v in (k, window.shape[1], nc)]
    if streaming:
        threads = 512 if k < 4096 else 1024
        out = _decode_kernel()(
            inputs=[q, window, compressed, indices, sink.astype(mx.float32),
                    mx.array(scale, dtype=mx.float32), *sizes],
            template=[("T", q.dtype), ("M", m), ("H", h), ("D", d),
                      ("THREADS", threads), ("SLICES", 1)],
            grid=(threads, b*m*h, 1), threadgroup=(threads, 1, 1),
            output_shapes=[q.shape], output_dtypes=[q.dtype])[0]
        _COUNTERS["calls"] += 1
        _COUNTERS["launches"] += 1
        _COUNTERS["fused"] += 1
        _COUNTERS["decode_streaming"] += 1
        return out
    if k > 32 and m > 1:
        template = [("T", q.dtype), ("M", m), ("H", h), ("D", d)]
        scores = _stage("qk")(
            inputs=[q, window, compressed, indices, mx.array(scale, dtype=mx.float32), *sizes],
            template=template, grid=(128, b*m*((h+7)//8)*((k+31)//32), 1),
            threadgroup=(128, 1, 1), output_shapes=[(b,m,h,k)],
            output_dtypes=[mx.float32])[0]
        weights, denom = _stage("softmax")(
            inputs=[scores, indices, sink.astype(mx.float32), *sizes], template=template,
            grid=(256, b*m*h, 1), threadgroup=(256, 1, 1),
            output_shapes=[(b,m,h,k), (b,m,h)], output_dtypes=[mx.float32, mx.float32])
        out = _stage("pv")(
            inputs=[weights, denom, window, compressed, indices, *sizes], template=template,
            grid=(128, b*m*((h+7)//8)*(d//32), 1), threadgroup=(128, 1, 1),
            output_shapes=[q.shape], output_dtypes=[q.dtype])[0]
        for name in ("qk", "softmax", "pv"):
            _COUNTERS[name] += 1
        _COUNTERS["calls"] += 1
        _COUNTERS["launches"] += 3
        return out
    out = kernel(
        inputs=[q, window, compressed, indices, sink.astype(mx.float32),
                mx.array(scale, dtype=mx.float32), *sizes],
        template=[("T", q.dtype), ("M", m), ("H", h), ("D", d)],
        grid=(256, b * m * h, 1), threadgroup=(256, 1, 1),
        output_shapes=[q.shape], output_dtypes=[q.dtype],
    )[0]
    _COUNTERS["calls"] += 1
    _COUNTERS["launches"] += 1
    _COUNTERS["fused"] += 1
    return out
