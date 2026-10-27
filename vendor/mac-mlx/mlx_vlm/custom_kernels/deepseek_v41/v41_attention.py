# SPDX-License-Identifier: MIT
"""Fork rb16 attention, opt-in through DSV41_PREFILL_V41_ATTN=1.

Source: IngeniousIdiocy/ds4-v41-m3ultra c8aa547,
metal/dsv4_misc.metal:5784-5882,6116-6286. See V41_KERNELS_NOTICE.md.
The online-softmax helpers are verbatim. Explicit IDs replace ring addressing;
Q and both banks are converted to fp16 once per call, output fp32 to Q dtype.
No speed or numerical-parity claim until the Mac gates have run.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .v41_dispatch import attn_geometry


@lru_cache(maxsize=1)
def _kernel():
    header, source = Path(__file__).with_suffix(".metal").read_text().split(
        "// MLX_HEADER_END", 1)
    return mx.fast.metal_kernel(
        name="dsv41_fork_attention_rb16",
        input_names=["Q", "Window", "Compressed", "Indices", "Sink", "Scale"],
        output_names=["Output"],
        header="#include <metal_stdlib>\nusing namespace metal;\n" + header, source=source,
        ensure_row_contiguous=True,
    )


def sparse_attn_v41(q, kv_all, attn_sink, idxs, scale, *, cohort=None,
                    block=None):
    """Q[B,N,64,512], banks[B,rows,512], IDs[B,N,K]; all conversions included.

    IDs <0 or >= NW+NC are skipped, preserving order, duplicates and holes.
    Two-dimensional banks are accepted for B=1. No inferred causal mask.
    DSV41_V41_ATTN_COHORT (default 8, one of 4/8/16/32) is how many heads share
    one threadgroup and therefore one gather; DSV41_V41_ATTN_BLOCK (default 16,
    one of 4/8/16) is how many selected IDs are staged per pass. The keyword
    arguments override the environment for sweeps. Neither changes the
    arithmetic or the order of the online softmax, only the work split, so
    every setting is bit-identical to the default. An invalid setting raises.
    """
    if not mx.metal.is_available():
        raise RuntimeError("V41 attention port requires Apple Metal")
    if q.ndim != 4 or q.shape[2:] != (64, 512) or q.shape[1] < 64:
        raise ValueError("V41 attention requires Q[B,N>=64,64,512]")
    b, n, h, d = q.shape
    if (b < 1 or idxs.ndim != 3 or idxs.shape[:2] != (b, n)
            or not 1 <= idxs.shape[-1] <= 640 or idxs.dtype != mx.int32
            or attn_sink.shape != (h,)):
        raise ValueError("expected int32 IDs[B,N,1..640] and sink[64]")
    floating = (mx.bfloat16, mx.float16, mx.float32)
    if q.dtype not in floating or attn_sink.dtype not in floating:
        raise ValueError("Q and sink must be floating point")
    window, compressed = kv_all if isinstance(kv_all, tuple) else (kv_all, None)
    banks = []
    lengths = []
    for bank in (window, compressed):
        if bank is None:
            lengths.append(0)
            banks.append(mx.zeros((b, 1, d), dtype=mx.float16))
            continue
        if bank.ndim == 2 and b == 1:
            bank = bank[None]
        if (bank.ndim != 3 or bank.shape[0] != b or bank.shape[2] != d
                or bank.dtype not in floating):
            raise ValueError("banks must be floating [B,rows,512]")
        lengths.append(bank.shape[1])
        banks.append(bank.astype(mx.float16) if bank.shape[1]
                     else mx.zeros((b, 1, d), dtype=mx.float16))
    geometry = attn_geometry(n, h, b, cohort=cohort, block=block)
    out = _kernel()(
        inputs=[q.astype(mx.float16), *banks, idxs,
                attn_sink.astype(mx.float32), mx.array([scale], mx.float32)],
        template=[("NT", n), ("H", h), ("K", idxs.shape[-1]),
                  ("NW", lengths[0]), ("NC", lengths[1]),
                  ("COHORT", geometry.cohort), ("BLOCK", geometry.block),
                  ("LEAN", True)],
        grid=geometry.grid, threadgroup=geometry.threadgroup,
        output_shapes=[q.shape], output_dtypes=[mx.float32],
    )[0]
    return out.astype(q.dtype)
