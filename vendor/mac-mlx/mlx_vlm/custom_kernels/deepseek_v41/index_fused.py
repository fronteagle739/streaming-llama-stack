"""Attach-latched BF16-storage/FP32-accumulation index scoring (MLX 0.32.2).

Only the reduced [B,S,N] tensor leaves the kernel. The caller retains the
existing visibility, candidate-block masking and eager argpartition semantics.
Steel loads BF16 operands into FP32 MMA fragments, exactly as expert_tiles_qmm;
this is not a claim of native BF16 MMA throughput on Apple hardware.
"""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

UPSTREAM_COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
# BEGIN SOURCE HASHES
SOURCE_HASHES = {'LICENSE': 'ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01',
 'mlx/backend/metal/kernels/bf16.h': 'abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb',
 'mlx/backend/metal/kernels/bf16_math.h': '1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496',
 'mlx/backend/metal/kernels/complex.h': '16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1',
 'mlx/backend/metal/kernels/defines.h': 'a2930dbd644c69c4b66a511a094034217f3c03f48e29a1613f601532150f9163',
 'mlx/backend/metal/kernels/steel/defines.h': 'b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a',
 'mlx/backend/metal/kernels/steel/gemm/mma.h': '9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7',
 'mlx/backend/metal/kernels/steel/gemm/transforms.h': '73487e29beab5af9478bc931c49b5e8d66937c2fa2651499867f47ac5529afc2',
 'mlx/backend/metal/kernels/steel/utils/integral_constant.h': '886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e',
 'mlx/backend/metal/kernels/steel/utils/type_traits.h': '4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c',
 'mlx/backend/metal/kernels/utils.h': '5e1568e9edde9d05dbf86f68fa0d6c6240f2c32b973c7c6a76166b9c0d91543d'}
# END SOURCE HASHES

_ENGAGED = set()          # one line per distinct skip outcome, not one globally
BM, BN, THREADS = 16, 32, 64


def _source():
    """One body for both variants; the define selects the candidate skip."""
    return Path(__file__).with_suffix(".metal").read_text(
        encoding="utf-8").split("// --- index fused body ---", 1)


def _require_runtime():
    if version("mlx") != "0.32.2":
        raise RuntimeError("index fused requires exactly MLX 0.32.2")
    if not mx.metal.is_available():
        raise RuntimeError("index fused requires the Metal GPU backend")


@lru_cache(maxsize=1)
def _kernel():
    _require_runtime()
    header, body = _source()
    return mx.fast.metal_kernel(
        name="dsv41_index_fused_v0322", input_names=["Q", "Keys", "Weights"],
        output_names=["Output"], header=header,
        source="#define DSV41_INDEX_SKIP 0\n" + body,
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"})


@lru_cache(maxsize=1)
def _skip_kernel():
    """Same body, plus the [B,S,NB] block mask: masked write and tile skip."""
    _require_runtime()
    header, body = _source()
    return mx.fast.metal_kernel(
        name="dsv41_index_skip_v0322",
        input_names=["Q", "Keys", "Weights", "Blocks"],
        output_names=["Output"], header=header,
        source="#define DSV41_INDEX_SKIP 1\n" + body,
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"})


def fused_scores(q, index_k, weights, candidates=None, block_size=8, block_mask=True):
    """One tiled launch; no host query chunking or headwise device allocation.

    ``candidates`` is the candidate mask from the candidate-source layer, in
    either of the two layouts that layer produces. ``block_mask=True`` is the
    precomputed [B,S,ceil(N/block_size)] block form (DSV41_S4_PRECOMPUTE=1);
    ``block_mask=False`` is the per-position [B,S,N] form, which is that same
    block mask repeated ``block_size`` times and trimmed, so the kernel reads
    it at stride ``block_size`` -- one read per block in both layouts. When it
    is given the kernel writes NEG_INF for every unadmitted key -- exactly what
    ``_mask_candidate_blocks`` leaves -- and skips the scoring of every 32-key
    tile no admitted block touches.
    """
    if q.ndim != 4 or index_k.ndim != 3:
        raise ValueError("index fused expects Q[B,S,32,128], K[B,N,128]")
    b, s, h, d = q.shape
    n = index_k.shape[1]
    if (h != 32 or d != 128 or min(b, s, n) <= 0
            or index_k.shape != (b, n, d) or weights.shape != (b, s, h)):
        raise ValueError("index fused requires positive B,S,N, H=32, D=128 and matching weights")
    allowed = (mx.bfloat16, mx.float16, mx.float32)
    if any(a.dtype not in allowed for a in (q, index_k, weights)):
        raise ValueError("index fused requires floating point inputs")
    inputs = [q.astype(mx.bfloat16), index_k.astype(mx.bfloat16),
              weights.astype(mx.float32)]
    launch = dict(grid=(THREADS * ((n + BN - 1) // BN), (s + BM - 1) // BM, b),
                  threadgroup=(THREADS, 1, 1), output_shapes=[(b, s, n)],
                  output_dtypes=[mx.float32])
    if candidates is None:
        return _kernel()(inputs=inputs, **launch)[0]
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("index fused candidate block_size must be a positive integer")
    blocks = (n + block_size - 1) // block_size
    width = blocks if block_mask else n
    if candidates.dtype != mx.bool_ or candidates.shape != (b, s, width):
        raise ValueError(
            f"index fused expects a bool {'block' if block_mask else 'position'} "
            f"mask [{b},{s},{width}], got {candidates.dtype} "
            f"{tuple(candidates.shape)}")
    return _skip_kernel()(
        inputs=inputs + [candidates],
        template=[("BS", block_size), ("POSMASK", int(not block_mask))], **launch)[0]


def index_scores_fused(x, projection, q, index_k, scale, candidates=None,
                       block_size=8, block_mask=True):
    """Preserve the eager FP32 weights projection, including its rounding."""
    skip = candidates is not None
    if skip not in _ENGAGED:
        # The call site decides the skip per call, so the first chunk's outcome
        # is not the run's outcome: announce each outcome the first time it
        # happens instead of latching on whichever came first.
        _ENGAGED.add(skip)
        print(f"[DSV41] INDEX_FUSED engaged n={q.shape[1]} nb={index_k.shape[1]} "
              f"skip={skip}", flush=True)
    weights = x.astype(mx.float32) @ projection.astype(mx.float32).T
    weights = weights * scale
    return fused_scores(q, index_k, weights, candidates, block_size, block_mask)


@lru_cache(maxsize=1)
def prewarm_index_fused():
    # Runtime B/S/N: this covers the production signature and both tile tails.
    # Every input is device-addressed (>7 elements), even for the probe.
    out = fused_scores(mx.ones((1, 17, 32, 128), mx.bfloat16),
                       mx.ones((1, 33, 128), mx.bfloat16),
                       mx.ones((1, 17, 32), mx.float32))
    mx.async_eval(out)
    mx.synchronize()


@lru_cache(maxsize=8)
def prewarm_index_skip(block_size=8, block_mask=True):
    # At the production block_size N=33 is five blocks of eight, so the second
    # key tile holds block four alone: this probe compiles and runs the skipped
    # tile, a partly admitted tile and both tile tails. 33 is a multiple of
    # neither 32 nor 8, so the position form also exercises the short final
    # block and the stride clamp. The flags stay device-addressed (>7 elements)
    # like every other input.
    blocks = (33 + block_size - 1) // block_size
    keep = mx.arange(blocks)[None, None] < blocks - 1
    if not block_mask:
        keep = mx.repeat(keep, block_size, axis=-1)[..., :33]
    out = fused_scores(mx.ones((1, 17, 32, 128), mx.bfloat16),
                       mx.ones((1, 33, 128), mx.bfloat16),
                       mx.ones((1, 17, 32), mx.float32),
                       mx.broadcast_to(keep, (1, 17, keep.shape[-1])),
                       block_size, block_mask)
    mx.async_eval(out)
    mx.synchronize()


def prepare_index_fused(layers, skip=False):
    """Validate backbone indexers and compile before publishing any latch."""
    try:
        indexers = [layer.attn.indexer for layer in layers
                    if hasattr(layer.attn, "indexer")]
        if not indexers:
            raise ValueError("index fused found no backbone indexers")
        for indexer in indexers:
            if (indexer.n_heads != 32 or indexer.head_dim != 128
                    or indexer.candidate_block_size <= 0):
                raise ValueError("index fused requires H=32, D=128 and positive block_size")
        prewarm_index_fused()
        if skip:
            # Both layouts: DSV41_S4_PRECOMPUTE decides which one the candidate
            # source produces, and it is read per call, not latched here.
            for size in sorted({i.candidate_block_size for i in indexers}):
                for block_mask in (True, False):
                    prewarm_index_skip(size, block_mask)
    except Exception as exc:
        raise RuntimeError("DSV41_INDEX_FUSED=1 attach failed; refusing to enable the custom kernel") from exc
