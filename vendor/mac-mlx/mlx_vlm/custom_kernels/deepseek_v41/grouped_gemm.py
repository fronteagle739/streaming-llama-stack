"""Experimental expert-aligned MXFP4 GEMM, compiled lazily on Metal.

Offsets must be nondecreasing int32 [E+1], start at zero and end at R.
Rows are sorted by expert. No host readback or full weight dequantization.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

TILES = (32, 64, 32)


@lru_cache(maxsize=1)
def _source():
    return Path(__file__).with_suffix(".metal").read_text().split("// --- kernel body ---", 1)


@lru_cache(maxsize=1)
def _kernel():
    if not mx.metal.is_available():
        raise RuntimeError("grouped_mxfp4 requires Metal")
    return mx.fast.metal_kernel(
        name="dsv41_grouped_mxfp4",
        input_names=["X", "W", "Scales", "Offsets", "Work"],
        output_names=["Output"],
        source=_source()[1], header=_source()[0],
    )


@lru_cache(maxsize=1)
def _tuning_source():
    parts = Path(__file__).with_name("grouped_gemm_tuning.metal").read_text().split(
        "// --- kernel body ---", 1)
    return _source()[0] + parts[0], parts[1]


@lru_cache(maxsize=3)
def _tuning_kernel(unroll=1):
    if not mx.metal.is_available():
        raise RuntimeError("grouped_mxfp4 requires Metal")
    header, source = _tuning_source()
    if unroll == 1:
        # Let the compiler make exactly the original loop-unrolling decision.
        source = source.replace("#pragma clang loop unroll_count(UNROLL)", "")
    return mx.fast.metal_kernel(
        name=f"dsv41_grouped_mxfp4_tuning_u{unroll}",
        input_names=["X", "W", "Scales", "Offsets", "Work"],
        output_names=["Output"], source=source, header=header,
    )


@lru_cache(maxsize=1)
def _offset_kernel():
    return mx.fast.metal_kernel(
        name="dsv41_expert_offsets", input_names=["Ids"],
        output_names=["Offsets"], source="""
        uint e = thread_position_in_grid.x;
        if (e > E) return;
        int lo = 0, hi = R;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (Ids[mid] < e) lo = mid + 1;
            else hi = mid;
        }
        Offsets[e] = lo;
        """,
    )


def expert_offsets(sorted_ids, experts):
    """GPU lower bounds of sorted, valid expert IDs; no token-sized CPU copy."""
    return _offset_kernel()(
        inputs=[sorted_ids], template=[("E", experts), ("R", sorted_ids.size)],
        grid=(experts + 1, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(experts + 1,)], output_dtypes=[mx.int32])[0]


@lru_cache(maxsize=1)
def _work_kernel():
    return mx.fast.metal_kernel(
        name="dsv41_expert_worklist", input_names=["Offsets"],
        output_names=["Work"], source="""
        uint e = thread_position_in_grid.x;
        if (e >= E) return;
        int first = 0;
        // E=384: a small O(E^2) scan avoids a host sync/dynamic dispatch.
        for (uint j = 0; j < e; ++j)
            first += (Offsets[j + 1] - Offsets[j] + BM - 1) / BM;
        int count = Offsets[e + 1] - Offsets[e];
        for (int row = 0; row < count; row += BM) {
            Work[2 * first] = int(e);
            Work[2 * first + 1] = row;
            ++first;
        }
        """,
    )


def prepare_worklist(rows, offsets, bm=TILES[0]):
    """Compact valid tasks followed by -1 sentinels; reuse for all projections.

    sum(ceil(count/BM)) <= floor(R/BM)+E, so the grid needs no CPU count.
    """
    experts = offsets.size - 1
    capacity = rows // bm + experts
    return _work_kernel()(
        inputs=[offsets], template=[("E", experts), ("BM", bm)],
        grid=(experts, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(capacity, 2)], output_dtypes=[mx.int32], init_value=-1)[0]


def _grouped_mxfp4(x_sorted, w_packed, scales, offsets, *, out_dtype=mx.bfloat16,
                   worklist=None, tiles=TILES, guard=0, config=None, mode="full",
                   verbose=False):
    """Internal sweep entry; config/mode select benchmark-only kernels.

    Production uses the original source when config=None and mode='full'.
    Decode mode returns uint32 [capacity, column_bands, 128] checksums.
    Reused worklists MUST have been built with the selected config's BM.
    """
    if x_sorted.ndim != 2 or w_packed.ndim != 3:
        raise ValueError("expected X[R,K] and packed W[E,N,K/8]")
    rows, k = x_sorted.shape
    e, n, kp = w_packed.shape
    if not e or not n or not k or k % 32 or kp * 8 != k:
        raise ValueError("MXFP4 requires positive E,N,K and K divisible by 32")
    if scales.shape != (e, n, k // 32) or offsets.shape != (e + 1,):
        raise ValueError("scales/offsets shape mismatch")
    if (w_packed.dtype != mx.uint32 or scales.dtype != mx.uint8
            or offsets.dtype != mx.int32):
        raise ValueError("expected uint32 weights, uint8 scales, int32 offsets")
    if x_sorted.dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("X must be BF16 or FP32 (SwiGLU down input)")
    if out_dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("output must be BF16 or FP32")
    if mode not in ("full", "decode", "mma"):
        raise ValueError("mode must be full, decode or mma")
    if mode != "full" and config is None:
        from .grouped_gemm_tuning import GemmConfig
        config = GemmConfig(tiles=tiles)
    if config is not None:
        if tiles != TILES and tiles != config.tiles:
            raise ValueError("tiles and config.tiles disagree")
        config.validate(2 if x_sorted.dtype == mx.bfloat16 else 4)
        tiles = config.tiles
    bm, bn, bk = tiles
    if config is None and (bm not in (16, 32) or bn not in (32, 64) or bk not in (32, 64)):
        raise ValueError("supported sweep: BM16/32, BN32/64, BK32/64")
    if guard < 0:
        raise ValueError("guard must be nonnegative")
    if not rows and mode == "full":
        return mx.full((2 * guard, n), -12345, dtype=out_dtype)
    if not rows:
        raise ValueError("attribution requires nonempty input")
    if worklist is None:
        worklist = prepare_worklist(rows, offsets, bm)
    if worklist.ndim != 2 or worklist.shape[1] != 2 or worklist.dtype != mx.int32:
        raise ValueError("worklist must be int32 [capacity,2]")
    template = [("T", x_sorted.dtype), ("O", out_dtype), ("K", k), ("N", n),
                ("BM", bm), ("BN", bn), ("BK", bk), ("GUARD", guard)]
    if config is not None:
        template += [("MODE", {"full": 0, "decode": 1, "mma": 2}[mode]),
                     ("VECTOR_DECODE", config.vector_decode),
                     ("NATIVE_LOADS", config.native_loads),
                     ("CONTIGUOUS", config.contiguous), ("PAD", config.padding),
                     ("UNROLL", config.unroll)]
    output_shape = (rows + 2 * guard, n)
    if mode == "decode":
        output_shape = (worklist.shape[0], (n + bn - 1) // bn, 128)
        out_dtype = mx.uint32
    return (_kernel() if config is None else _tuning_kernel(config.unroll))(
        inputs=[x_sorted, w_packed, scales, offsets, worklist],
        template=template,
        grid=(128, worklist.shape[0], (n + bn - 1) // bn),
        threadgroup=(128, 1, 1), output_shapes=[output_shape], verbose=verbose,
        output_dtypes=[out_dtype], init_value=-12345 if guard and mode != "decode" else None)[0]


def grouped_mxfp4(x_sorted, w_packed, scales, expert_offsets, *, out_dtype=mx.bfloat16):
    """Compute expert-segmented X @ W.T, with FP32 accumulation.

    BF16 X uses BF16 tile storage; FP32 X preserves FP32 X/W tiles for down.
    The caller owns offset validity. Output is [R,N], BF16 by default.
    """
    return _grouped_mxfp4(x_sorted, w_packed, scales, expert_offsets,
                          out_dtype=out_dtype)
