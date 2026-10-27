"""Opt-in query-parallel online prefill attention for BF16 KV banks.

Four SIMD groups own four independent queries, eight heads each. The default
two-launch path gathers shared K=V once for all head groups, then runs dense
MMA attention. ``gathered=False`` measures the single-launch repeated-gather
variant. Neither materializes global scores or probabilities.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_PREFILL_FLASH_ATTN=1 requires Metal; no fallback")
    return mx.fast.metal_kernel(
        name="dsv41_sparse_attention_prefill",
        input_names=["Q", "Window", "Compressed", "Indices", "Sink", "Scale",
                     "K", "NW", "NC"],
        output_names=["Output"],
        source=Path(__file__).with_suffix(".metal").read_text(),
        header="#include <metal_simdgroup_matrix>\n",
    )


@lru_cache(maxsize=1)
def _gather_kernel():
    return mx.fast.metal_kernel(
        name="dsv41_sparse_attention_prefill_gather",
        input_names=["Window", "Compressed", "Indices", "K", "NW", "NC"],
        output_names=["Gathered"],
        source=Path(__file__).with_name("sparse_attention_prefill_gather.metal").read_text(),
    )


def sparse_attn_prefill(q, kv_all, sink, idxs, scale, *, gathered=True):
    """[B,N,H,512] queries, BF16 shared banks, int32 [B,N,K] IDs, FP32 sink.

    IDs address window then compressed rows; -1 is masked. Visibility is fully
    determined by IDs, with no additional causality or sorted-ID assumption.
    FP32 arithmetic throughout; output has q.dtype (BF16 in production). This
    low-level entry point permits small parity fixtures; production dispatch
    enforces the minimum N. Gathered workspace is B*N*K*512*2 bytes: 1.25 GiB
    at B=1/N=2048/K=640. K=V uses one allocation, not separate K and V buffers.
    """
    if q.ndim != 4 or idxs.ndim != 3:
        raise ValueError("expected Q[B,N,H,512] and indices[B,N,K]")
    b, n, h, d = q.shape
    k = idxs.shape[-1]
    if b < 1 or n <= 1 or h < 1 or d != 512 or not 1 <= k <= 640:
        raise ValueError("prefill requires B,H>=1, N>1, D=512, 1<=K<=640")
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError("prefill Q must be bf16/fp16/fp32")
    if idxs.shape[:2] != (b, n) or idxs.dtype != mx.int32:
        raise ValueError("prefill indices must be int32 [B,N,K]")
    if sink.shape != (h,) or sink.dtype != mx.float32:
        raise ValueError("prefill sink must be fp32 [H]")
    window, compressed = kv_all if isinstance(kv_all, tuple) else (kv_all, None)
    for bank in (window, compressed):
        if bank is not None and (bank.ndim != 3 or bank.shape[0] != b or
                                 bank.shape[2] != d or bank.dtype != mx.bfloat16):
            raise ValueError("prefill KV banks must be bf16 [B,N,512]")
    if window is None:
        raise ValueError("prefill requires a window bank (possibly empty)")
    nw = window.shape[1]
    nc = 0 if compressed is None else compressed.shape[1]
    # JIT inputs need an allocation even for an unused bank; NW/NC remain zero.
    if nw == 0:
        window = mx.zeros((b, 1, d), dtype=mx.bfloat16)
    if nc == 0:
        compressed = mx.zeros((b, 1, d), dtype=mx.bfloat16)
    kernel = _kernel()  # Check Metal availability before creating either JIT.
    sizes = [mx.array(v, dtype=mx.int32) for v in (k, nw, nc)]
    if gathered:
        window = _gather_kernel()(
            inputs=[window, compressed, idxs, *sizes],
            template=[("KV_T", mx.bfloat16), ("M", n), ("D", d)],
            grid=(k * d, b * n, 1), threadgroup=(128, 1, 1),
            output_shapes=[(b, n, k, d)], output_dtypes=[mx.bfloat16],
        )[0]
    return kernel(
        inputs=[q, window, compressed, idxs, sink,
                mx.array(scale, dtype=mx.float32),
                *sizes],
        template=[("T", q.dtype), ("KV_T", mx.bfloat16), ("M", n),
                  ("H", h), ("D", d), ("GATHERED", gathered)],
        grid=(128, b * ((n + 3) // 4) * ((h + 7) // 8), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape], output_dtypes=[q.dtype],
    )[0]


def sparse_attn_prefill_fused(q, kv_all, sink, idxs, scale):
    """Single-launch comparison: repeated bank gathers across head groups."""
    return sparse_attn_prefill(q, kv_all, sink, idxs, scale, gathered=False)
