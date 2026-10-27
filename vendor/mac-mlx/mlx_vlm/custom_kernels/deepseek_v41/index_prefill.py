"""Opt-in bounded-workspace FP32 prefill index scoring and selection.

One query per threadgroup, 1024 keys per selection tile, 32 keys per FP32
SIMD-matrix scoring subtile. Only the best 512 score/id pairs survive a tile.
No [B,S,H,N] or [B,S,N] score buffer is allocated. Optional block maxima
support the candidate-source layer without reconstructing either buffer.

Inputs are already fake-quantised by the model (currently FP4 UE8M0 for both
index queries and keys). This module does not change their quantisation.
FP32 matrix/reduction arithmetic can round differently from MLX einsum;
hardware parity and performance must be measured before enabling by default.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="dsv41_index_prefill",
        input_names=["Q", "Keys", "Weights", "Lens", "Visibility", "Candidates", "Offset"],
        output_names=["Indices", "BlockMax"],
        source=Path(__file__).with_suffix(".metal").read_text(),
        header="#include <metal_simdgroup_matrix>\n",
        ensure_row_contiguous=True,
    )


def index_scores_topk(
    x, weights_proj, q, index_k, lens, offset, k, *, scale=None,
    visibility=None, candidates=None, candidate_block_size=8,
    candidates_are_blocks=True, return_block_scores=False,
):
    """Return position-sorted int32 [B,S,k] ids (or [S,k] without batch).

    q is [B,S,32,128], index_k [B,N,128], x [B,S,X], projection [32,X].
    lens has S entries; visible local keys satisfy key < lens[query].
    visibility optionally supplies the existing [S,N] boolean mask.
    candidates is the existing [B,S,ceil(N/block_size)] block mask, or
    [B,S,N] when candidates_are_blocks=False. Masked/-inf selections emit -1
    AFTER sorting local positions and adding offset to reachable positions.
    Equal scores prefer smaller local positions; MLX's tie choice is unspecified.
    With return_block_scores=True, also return causal, masked block maxima.
    """
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_PREFILL_TILED_INDEX=1 requires Metal")
    unbatched = q.ndim == 3
    if unbatched:
        if x.ndim != 2 or index_k.ndim != 2:
            raise ValueError("unbatched x and index_k must be rank two")
        x, q, index_k = x[None], q[None], index_k[None]
        if candidates is not None and candidates.ndim == 2:
            candidates = candidates[None]
    if q.ndim != 4 or x.ndim != 3 or index_k.ndim != 3:
        raise ValueError("expected x[B,S,X], q[B,S,32,128], index_k[B,N,128]")
    b, s, h, d = q.shape
    n = index_k.shape[1]
    if (min(b, s, n, x.shape[-1]) < 1 or (h, d) != (32, 128)
            or x.shape[:2] != (b, s) or weights_proj.shape != (h, x.shape[-1])
            or index_k.shape != (b, n, d) or lens.size != s):
        raise ValueError("invalid V4.1 prefill index shapes (requires H=32, D=128)")
    if not isinstance(k, int) or not 1 <= k <= min(512, n):
        raise ValueError("expected 1 <= k <= min(512, N)")
    if not isinstance(offset, int) or not 0 <= offset <= 2**31 - 1 - n:
        raise ValueError("offset plus key positions must fit nonnegative int32")
    if not isinstance(candidate_block_size, int) or candidate_block_size < 1:
        raise ValueError("candidate_block_size must be a positive integer")
    if return_block_scores and 1024 % candidate_block_size:
        raise ValueError("block-max output requires a block size dividing 1024")
    allowed = (mx.float32, mx.float16, mx.bfloat16)
    if any(a.dtype not in allowed for a in (x, weights_proj, q, index_k)):
        raise ValueError("index inputs must be fp32, fp16 or bf16")
    nb = (n + candidate_block_size - 1) // candidate_block_size
    if visibility is not None and (visibility.shape != (s, n) or visibility.dtype != mx.bool_):
        raise ValueError("visibility must be bool [S,N]")
    nc = nb if candidates_are_blocks else n
    if candidates is not None and (candidates.shape != (b, s, nc) or candidates.dtype != mx.bool_):
        raise ValueError("candidate mask has the wrong shape or dtype")

    # Preserve the reference projection and scaling, once for the entire query
    # batch rather than once per key tile. Dense weights remain FP32 here.
    if scale is None:
        scale = d ** -0.5 * h ** -0.5
    weights = (x.astype(mx.float32) @ weights_proj.astype(mx.float32).T) * scale
    dummy = mx.array([True], dtype=mx.bool_)
    out, blocks = _kernel()(
        inputs=[q, index_k, weights, lens.reshape(s).astype(mx.int32),
                visibility if visibility is not None else dummy,
                candidates if candidates is not None else dummy,
                mx.array([offset], dtype=mx.int32)],
        template=[("S", s), ("N", n), ("TOPK", k), ("BLOCK", candidate_block_size),
                  ("NB", nb), ("NC", nc), ("HAS_VIS", visibility is not None),
                  ("HAS_CAND", candidates is not None), ("CAND_BLOCKS", candidates_are_blocks),
                  ("EMIT_BLOCKS", return_block_scores)],
        grid=(128, b * s, 1), threadgroup=(128, 1, 1),
        output_shapes=[(b, s, k), (b, s, nb) if return_block_scores else (1,)],
        output_dtypes=[mx.int32, mx.float32],
    )
    if unbatched:
        out = out[0]
        if return_block_scores:
            blocks = blocks[0]
    return (out, blocks) if return_block_scores else out
