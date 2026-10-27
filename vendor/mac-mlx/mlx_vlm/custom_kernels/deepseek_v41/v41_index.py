# SPDX-License-Identifier: MIT
"""Fork tiled index scores and block-sort/merge top-k; see V41_KERNELS_NOTICE.md.

Source: IngeniousIdiocy/ds4-v41-m3ultra c8aa547, metal/dsv4_misc.metal:
2226-2357 and metal/argsort.metal:51-315. The V4.1 wrapper uses the GLM-named
FP32 tile (ds4_metal.m:38455-38463); Q/K staging and accumulation stay FP32.
Candidate masking remains at the language-layer call site before selection.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=3)
def _kernel(stage):
    filename, inputs, outputs = {
        "score": ("v41_index.metal", ["Q", "Weights", "Keys", "Lens", "Scale"], ["Scores"]),
        "sort": ("v41_topk_sort.metal", ["Scores"], ["Output"]),
        "merge": ("v41_topk_merge.metal", ["Scores", "Work"], ["Output"]),
    }[stage]
    return mx.fast.metal_kernel(
        name="dsv41_fork_index_" + stage, input_names=inputs, output_names=outputs,
        source=Path(__file__).with_name(filename).read_text(),
        header="#include <metal_simdgroup_matrix>\n", ensure_row_contiguous=True,
    )


def index_scores_v41(x, projection, q, index_k, scale, lens):
    """Q[B,M,H,128], keys[B,N,128], projection[H,X], lens[M,1].

    Lens is the nondecreasing compressed visibility count shared across batch.
    Projection and input promotion run once per layer call; the fork stages
    Q/K as float inside each 8-query x 32-key tile, accumulating in fp32.
    """
    if not mx.metal.is_available():
        raise RuntimeError("V41 index port requires Apple Metal")
    if q.ndim != 4 or q.shape[-1] != 128 or q.shape[1] < 64:
        raise ValueError("V41 index requires Q[B,M>=64,H,128]")
    b, m, h, d = q.shape
    if (index_k.ndim != 3 or index_k.shape[0] != b or index_k.shape[2] != d
            or min(b, h, index_k.shape[1]) < 1 or x.shape[:2] != (b, m)
            or projection.shape != (h, x.shape[-1]) or lens.size != m):
        raise ValueError("invalid V41 index shapes")
    if any(a.dtype not in (mx.bfloat16, mx.float16, mx.float32)
           for a in (x, projection, q, index_k)):
        raise ValueError("index inputs must be floating point")
    n = index_k.shape[1]
    weights = x.astype(mx.float32) @ projection.astype(mx.float32).T
    return _kernel("score")(
        inputs=[q.astype(mx.float32), weights, index_k.astype(mx.float32),
                mx.clip(lens, 0, n).astype(mx.int32), mx.array([scale], mx.float32)],
        template=[("M", m), ("N", n), ("H", h)],
        grid=(((n + 31) // 32) * 128, (m + 7) // 8, b),
        threadgroup=(128, 1, 1), output_shapes=[(b, m, n)],
        output_dtypes=[mx.float32],
    )[0]


def index_topk_v41(scores, k, lens, offset):
    """Original descending bitonic/merge ranking, then local ID formatting.

    Finite scores/-inf only. Ties retain the fork's ordering (not MLX's
    unspecified argpartition tie choice). No cache allocation/layout dependency.
    """
    if not mx.metal.is_available():
        raise RuntimeError("V41 top-k port requires Apple Metal")
    if scores.ndim != 3 or scores.dtype != mx.float32:
        raise ValueError("scores must be fp32 [B,M,N]")
    b, m, n = scores.shape
    if min(b, m) < 1 or not 1 <= k <= n or lens.size != m:
        raise ValueError("expected 1<=k<=N and M visibility lengths")
    # The fork chooses the device's maximum; 256 is a portable fixed bound.
    threads = min(256, 1 << (n - 1).bit_length())
    blocks = (n + threads - 1) // threads
    block_k = min(k, threads)
    width = ((blocks - 1) * block_k + min(n - (blocks - 1) * threads, block_k)
             if blocks > 1 else k)
    common = [("N", n), ("ROWS", b*m), ("WORK", width), ("BLOCK_TOPK", block_k)]
    work = _kernel("sort")(
        inputs=[scores], template=common + [("TOPK", block_k), ("THREADS", threads)],
        grid=(blocks*b*m*threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(b, m, width)], output_dtypes=[mx.int32],
    )[0]
    run = block_k
    while run < width:
        groups = (width + 2*run - 1) // (2*run)
        out_width = k if groups == 1 else width
        merge_threads = min(512, run)
        work = _kernel("merge")(
            inputs=[scores, work],
            template=common + [("RUN", run), ("TOPK", out_width), ("THREADS", merge_threads)],
            grid=(groups*b*m*merge_threads, 1, 1), threadgroup=(merge_threads, 1, 1),
            output_shapes=[(b, m, out_width)], output_dtypes=[mx.int32],
        )[0]
        run *= 2
    idx = mx.sort(work, axis=-1)
    visible = ((idx < lens.reshape(1, m, 1))
               & (mx.take_along_axis(scores, idx, axis=-1) > -float("inf")))
    return mx.where(visible, idx + offset, mx.array(-1, mx.int32))
