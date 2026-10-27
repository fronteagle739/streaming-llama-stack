"""Experimental MLX 0.32.2 sorted MXFP4 QMM with expert-aligned BM16 tiles.

RoutingPlan is lazy GPU metadata, reused by all three FFN projections. Sorted,
in-range uint32 labels are a caller precondition (as for stock sorted QMM).
No values or dynamic tile counts cross to the host. Shapes are CPU metadata.
"""
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

UPSTREAM_COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
EXPERTS = 384
# Hashes of unmodified tag blobs, including MLX's implicit utils() preamble.
# BEGIN SOURCE HASHES
SOURCE_HASHES = {'LICENSE': 'ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01',
 'mlx/backend/metal/kernels/bf16.h': 'abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb',
 'mlx/backend/metal/kernels/bf16_math.h': '1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496',
 'mlx/backend/metal/kernels/complex.h': '16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1',
 'mlx/backend/metal/kernels/defines.h': 'a2930dbd644c69c4b66a511a094034217f3c03f48e29a1613f601532150f9163',
 'mlx/backend/metal/kernels/fp4.h': 'e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a',
 'mlx/backend/metal/kernels/fp8.h': '3ee75c58a1e125ee9eaed3961ef89719d47ce4c8446bb930390d7a4579f08439',
 'mlx/backend/metal/kernels/fp_quantized.h': 'bdf6d419b623cdfd7493ff58fbf9c70b4a2c2c39511330d6a61646093148225e',
 'mlx/backend/metal/kernels/logging.h': '197d9d81944146bea274ebcbc17b2b97b39c7478c0316a3166a08b3fb2467e4b',
 'mlx/backend/metal/kernels/quantized_utils.h': 'a12841b57d505f6cca81631901b845dbf76bc90dfc099e7f40c763b2cc838c51',
 'mlx/backend/metal/kernels/steel/defines.h': 'b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a',
 'mlx/backend/metal/kernels/steel/gemm/gemm.h': 'c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0',
 'mlx/backend/metal/kernels/steel/gemm/loader.h': 'c1b82153670b1e18371a6e88fdbcf440af692957775c2f837700315ab8a5112e',
 'mlx/backend/metal/kernels/steel/gemm/mma.h': '9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7',
 'mlx/backend/metal/kernels/steel/gemm/params.h': '6711e72e32310eafb088895d5a6d20fcfca814158f5b38eb72b6e69d9d773e43',
 'mlx/backend/metal/kernels/steel/gemm/transforms.h': '73487e29beab5af9478bc931c49b5e8d66937c2fa2651499867f47ac5529afc2',
 'mlx/backend/metal/kernels/steel/utils.h': 'd4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a',
 'mlx/backend/metal/kernels/steel/utils/integral_constant.h': '886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e',
 'mlx/backend/metal/kernels/steel/utils/type_traits.h': '4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c',
 'mlx/backend/metal/kernels/utils.h': '5e1568e9edde9d05dbf86f68fa0d6c6240f2c32b973c7c6a76166b9c0d91543d'}
# END SOURCE HASHES


@dataclass(frozen=True)
class RoutingPlan:
    offsets: mx.array
    prefix: mx.array
    rows: int
    experts: int = EXPERTS
    tile_rows: int = 16

    @property
    def candidates(self):
        # sum ceil(n[e]/BM) <= ceil(R/BM) + E - 1; one extra is harmless.
        return (self.rows + self.tile_rows - 1) // self.tile_rows + self.experts


@lru_cache(maxsize=1)
def _sources():
    text = Path(__file__).with_suffix(".metal").read_text(encoding="utf-8")
    header, bodies = text.split("// --- metadata body ---", 1)
    metadata, gemm = bodies.split("// --- gemm body ---", 1)
    return header, metadata, gemm


@lru_cache(maxsize=1)
def _kernels():
    if version("mlx") != "0.32.2":
        raise RuntimeError("expert tiles requires exactly MLX 0.32.2")
    if not mx.metal.is_available():
        raise RuntimeError("expert tiles requires the Metal GPU backend")
    header, metadata, gemm = _sources()
    options = dict(ensure_row_contiguous=True, compile_options={"math_mode": "safe"})
    return (
        mx.fast.metal_kernel(name="dsv41_expert_tiles_metadata_v0322",
            input_names=["Ids"], output_names=["Offsets", "Prefix"],
            source=metadata, **options),
        mx.fast.metal_kernel(name="dsv41_expert_tiles_qmm_v0322",
            input_names=["X", "W", "Scales", "Offsets", "Prefix"],
            output_names=["Output"], header=header, source=gemm, **options),
    )


def expert_tiles_metadata(sorted_ids, experts=EXPERTS, *, tile_rows=16):
    """Compute O[0:E+1] lower bounds and P[0:E+1] tile prefixes on GPU."""
    if experts != EXPERTS:
        raise ValueError("expert tiles metadata is specialised to E=384")
    if tile_rows not in (16, 32):
        raise ValueError("expert tiles metadata tile_rows must be 16 or 32")
    if sorted_ids.ndim != 1 or sorted_ids.dtype != mx.uint32:
        raise ValueError("expert tiles requires sorted uint32 IDs with shape [R]")
    if sorted_ids.size > 2**31 - 32:
        raise ValueError("expert tiles R exceeds the signed 32-bit row limit")
    offsets, prefix = _kernels()[0](inputs=[sorted_ids],
        template=[("BM", tile_rows)],
        grid=(512, 1, 1), threadgroup=(512, 1, 1),
        output_shapes=[(EXPERTS + 1,), (EXPERTS + 1,)],
        output_dtypes=[mx.uint32, mx.uint32])
    return RoutingPlan(offsets, prefix, sorted_ids.size, tile_rows=tile_rows)


def validate_weight(weight, scales):
    if weight.ndim != 3 or weight.shape[0] != EXPERTS or weight.dtype != mx.uint32:
        raise ValueError("expert tiles W must be uint32 [384, N, K/8]")
    _, n, packed_k = weight.shape
    k = packed_k * 8
    if k == 0 or n == 0 or k % 32 or n % 32:
        raise ValueError("expert tiles requires positive K and N multiples of 32")
    if scales.shape != (EXPERTS, n, k // 32) or scales.dtype != mx.uint8:
        raise ValueError("expert tiles scales must be uint8 [384, N, K/32]")
    if n * (k // 2) > 2**31 - 1:
        raise ValueError("expert tiles weight stride exceeds MLX's int32 limit")
    return k, n


def expert_tiles_qmm(x, weight, scales, routing):
    """Return [R,1,N], in sorted order, with X's BF16 or FP32 dtype."""
    k, n = validate_weight(weight, scales)
    if x.dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("expert tiles X must be bfloat16 or float32")
    if x.shape != (routing.rows, 1, k):
        raise ValueError("expert tiles X must have shape [routing.rows, 1, K]")
    if (routing.tile_rows != 16 or routing.experts != EXPERTS
            or routing.offsets.shape != (EXPERTS + 1,)
            or routing.prefix.shape != (EXPERTS + 1,)
            or routing.offsets.dtype != mx.uint32 or routing.prefix.dtype != mx.uint32):
        raise ValueError("expert tiles requires a 384-expert uint32 RoutingPlan")
    if routing.rows == 0:
        return mx.zeros((0, 1, n), dtype=x.dtype)
    return _kernels()[1](inputs=[x, weight, scales, routing.offsets, routing.prefix],
        template=[("T", x.dtype)],
        grid=(64 * (n // 32), routing.candidates, 1), threadgroup=(64, 1, 1),
        output_shapes=[(routing.rows, 1, n)], output_dtypes=[x.dtype])[0]


@lru_cache(maxsize=1)
def prewarm_expert_tiles():
    """Attach only: force compilation before publishing the enabled latch.

    K, N and R are runtime shape arguments, so these are the same compiled
    kernels used for real model shapes. Synchronisation here has no readback;
    it is never called by the routed FFN or an individual projection.
    """
    ids = mx.concatenate((mx.zeros((17,), mx.uint32), mx.full((16,), 383, mx.uint32)))
    routing = expert_tiles_metadata(ids)
    w = mx.zeros((EXPERTS, 32, 4), mx.uint32)
    s = mx.full((EXPERTS, 32, 1), 127, mx.uint8)
    outputs = [expert_tiles_qmm(mx.zeros((33, 1, 32), dtype), w, s, routing)
               for dtype in (mx.bfloat16, mx.float32)]
    # A second metadata signature covers constant-address-space (<8) ID arrays.
    tiny = expert_tiles_metadata(mx.array([383], dtype=mx.uint32))
    mx.async_eval(*outputs, tiny.offsets, tiny.prefix)
    mx.synchronize()
