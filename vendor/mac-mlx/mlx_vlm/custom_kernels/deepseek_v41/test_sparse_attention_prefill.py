"""Mac gates; run pytest -s on this file, or python -m <this module> --timing.

The main matrix uses production H=64/D=512. Reference evaluation is chunked to
avoid a prompt-wide gathered KV allocation. Timing excludes data creation/JIT,
includes dispatch/evaluation, and compares the production 256-query three-stage
chunks with query-parallel fused and gather+MMA paths for one complete
layer-equivalent attention, including the gather allocation and both launches.
"""

import argparse
from contextlib import contextmanager
import os
from statistics import median
from time import perf_counter

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires Apple Metal", allow_module_level=True)

from mlx_vlm.custom_kernels.deepseek_v41 import sparse_attention as dispatch
from mlx_vlm.custom_kernels.deepseek_v41.sparse_attention_prefill import (
    sparse_attn_prefill, sparse_attn_prefill_fused,
)
from mlx_vlm.models.deepseek_v41.language import (
    _sparse_attn_chunk,
    sparse_attn,
    window_idx_matrix,
)


@contextmanager
def _flash_switch(value):
    name = "DSV41_PREFILL_FLASH_ATTN"
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _case(n, context, mode, *, b=1, h=64, k=640, seed=41):
    rng = np.random.default_rng(seed)
    nw, nc = context, context // 2

    def bf16(shape):
        return mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(mx.bfloat16)

    q = bf16((b, n, h, 512))
    banks = (bf16((b, nw, 512)), bf16((b, nc, 512)))
    sink = mx.array(np.linspace(-8, 8, h, dtype=np.float32))
    # Real initial-window padding, shifted into the end of the raw bank. For
    # n<128 the actual window function returns fewer columns: pad to W=128.
    win = np.asarray(window_idx_matrix(0, n, 128)).copy()
    win = np.where(win >= 0, win + context - n, -1)
    win = np.pad(win, ((0, 0), (0, 128 - win.shape[1])), constant_values=-1)
    mask = np.tile(win >= 0, (1, 5))
    if mode == "window":
        ids = rng.integers(0, nw, (b, n, 640), dtype=np.int32)
    elif mode == "compressed":
        ids = nw + rng.integers(0, nc, (b, n, 640), dtype=np.int32)
    else:
        ids = nw + rng.integers(0, nc, (b, n, 640), dtype=np.int32)
        ids[:, :, :128] = win
    ids = np.where(mask[None], ids, -1).astype(np.int32)
    ids[:, :, 151:167] = -1  # Holes within/across selected blocks.
    ids[:, 0, :] = -1       # Sink alone must give exact zero.
    ids[:, 1, 1:] = -1      # Singleton visible row.
    if mode == "mixed":
        ids[:, -1, :4] = [nw - 1, nw, nw + nc - 1, nw]  # Boundary + duplicate.
    return q, banks, sink, mx.array(ids[:, :, :k]), 512 ** -0.5


def _three_stage(q, banks, sink, ids, scale):
    # Force the existing path independent of the caller's environment. K>32
    # and every chunk N>1 in these tests guarantee QK -> softmax -> PV.
    with _flash_switch("0"):
        parts = [dispatch.sparse_attention(q[:, i:i + 256], banks, sink,
                                          ids[:, i:i + 256], scale)
                 for i in range(0, q.shape[1], 256)]
    return mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def _check(case, label):
    q, banks, sink, ids, scale = case
    before = dispatch.status()["qk"]
    old = _three_stage(*case)
    assert dispatch.status()["qk"] - before == (q.shape[1] + 255) // 256
    new = sparse_attn_prefill(*case)
    fused = sparse_attn_prefill_fused(*case)
    mx.eval(old, new, fused)
    assert new.shape == q.shape and new.dtype == q.dtype
    assert fused.shape == q.shape and fused.dtype == q.dtype
    # Reference is the existing three-operation FP32 math (QK, sink softmax,
    # PV), using pure MLX contractions rather than either custom kernel.
    maxima = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    reference_squared = 0.0
    direct_max = 0.0
    for start in range(0, q.shape[1], 16):
        stop = start + 16
        ref = _sparse_attn_chunk(q[:, start:stop].astype(mx.float32), banks,
                                 sink.reshape(1, 1, -1, 1), ids[:, start:stop], scale)
        mx.eval(ref)
        reference = np.asarray(ref).astype(np.float64)
        outputs = [np.asarray(x[:, start:stop].astype(mx.float32)).astype(np.float64)
                   for x in (old, new, fused)]
        assert np.isfinite(reference).all()
        for j, output in enumerate(outputs):
            assert np.isfinite(output).all()
            delta = output - reference
            maxima[j] = max(maxima[j], float(np.max(np.abs(delta))))
            squared[j] += np.sum(delta * delta)
        reference_squared += np.sum(reference * reference)
        direct_max = max(direct_max, float(np.max(np.abs(outputs[0] - outputs[1]))))
    relative = np.sqrt(squared / max(reference_squared, np.finfo(np.float64).tiny))
    print(f"{label}: old max_abs={maxima[0]:.8g} rms_relative={relative[0]:.8g}; "
          f"gather_mma max_abs={maxima[1]:.8g} rms_relative={relative[1]:.8g}; "
          f"fused max_abs={maxima[2]:.8g} rms_relative={relative[2]:.8g}; "
          f"old/flash max_abs={direct_max:.8g}")
    # Same reference and exactly twice the baseline's own error, with no loose
    # absolute floor. A zero-error baseline requires zero flash error too.
    assert np.all(maxima <= 2 * maxima[0])
    assert np.all(relative <= 2 * relative[0])
    assert np.all(np.asarray(new[:, 0].astype(mx.float32)) == 0)
    assert np.all(np.asarray(fused[:, 0].astype(mx.float32)) == 0)


@pytest.mark.parametrize("n", [16, 512, 2048])
@pytest.mark.parametrize("context", [6000, 30000])
@pytest.mark.parametrize("mode", ["window", "compressed", "mixed"])
def test_production_parity(n, context, mode):
    _check(_case(n, context, mode), f"n={n} context={context} {mode}")


def test_partial_tiles_and_batch():
    _check(_case(19, 256, "mixed", b=2, h=11, k=37), "batch=2 N=19 H=11 K=37")


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_query_dtypes(dtype):
    q, banks, sink, ids, scale = _case(9, 128, "mixed", h=11, k=37)
    _check((q.astype(dtype), banks, sink, ids, scale), str(dtype))


def test_invalid_ids_are_masked():
    q, banks, sink, _, scale = _case(9, 128, "mixed", h=11, k=37)
    bad = mx.array([-1, -7, 192, 199], dtype=mx.int32)
    ids = mx.broadcast_to(bad[None, None], (1, 9, 4))
    for path in (sparse_attn_prefill, sparse_attn_prefill_fused):
        actual = path(q, banks, sink, ids, scale)
        mx.eval(actual)
        assert np.all(np.asarray(actual.astype(mx.float32)) == 0)


@pytest.mark.parametrize("empty_bank", ["window", "compressed"])
@pytest.mark.parametrize("gathered", [False, True])
def test_empty_bank(empty_bank, gathered):
    q, banks, sink, ids, scale = _case(16, 128, empty_bank)
    if empty_bank == "window":
        banks = (mx.zeros((1, 0, 512), dtype=mx.bfloat16), banks[1])
    else:
        banks = (banks[0], mx.zeros((1, 0, 512), dtype=mx.bfloat16))
    # All valid IDs target row zero of the nonempty bank.
    ids = mx.where(ids >= 0, 0, -1).astype(mx.int32)
    actual = sparse_attn_prefill(q, banks, sink, ids, scale, gathered=gathered)
    reference_bank = banks[1] if empty_bank == "window" else banks[0]
    expected = _sparse_attn_chunk(q.astype(mx.float32), reference_bank,
                                  sink.reshape(1, 1, -1, 1), ids, scale).astype(q.dtype)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                               np.asarray(expected.astype(mx.float32)), atol=0, rtol=0.008)


@pytest.mark.parametrize("sink_value", [-100.0, 100.0])
def test_extreme_sinks_and_masked_blocks(sink_value):
    q, banks, _, ids, scale = _case(16, 128, "mixed")
    _check((q, banks, mx.full((64,), sink_value, dtype=mx.float32), ids, scale),
           f"sink={sink_value}")


def test_dispatch_is_opt_in_and_decode_unchanged(monkeypatch):
    case = _case(64, 128, "mixed")
    marker = object()
    from mlx_vlm.custom_kernels.deepseek_v41 import sparse_attention_prefill as module

    monkeypatch.setattr(module, "sparse_attn_prefill", lambda *args: marker)
    monkeypatch.delenv("DSV41_PREFILL_FLASH_MIN_N", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_FLASH_ATTN", raising=False)
    before = dispatch.status()
    assert dispatch.sparse_attention(*case) is not marker
    assert dispatch.status()["prefill_flash"] == before["prefill_flash"]
    monkeypatch.setenv("DSV41_PREFILL_FLASH_ATTN", "1")
    assert dispatch.sparse_attention(*case) is marker
    # The new flag alone reaches the dispatcher through language.sparse_attn.
    monkeypatch.delenv("DSV41_METAL_SPARSE_ATTN", raising=False)
    monkeypatch.delenv("DSV41_SDPA_SPARSE_ATTN", raising=False)
    assert sparse_attn(*case) is marker
    q, banks, sink, ids, scale = case
    decode = (q[:, :1], banks, sink, ids[:, :1], scale)
    with _flash_switch("0"):
        expected = dispatch.sparse_attention(*decode)
    actual = dispatch.sparse_attention(*decode)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)),
                                  np.asarray(expected.astype(mx.float32)))


@pytest.mark.parametrize("n", [1, 2, 6, 8, 9, 63, 64, 128])
@pytest.mark.parametrize("threshold", [None, "1", "128"])
def test_prefill_threshold(n, threshold, monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_FLASH_ATTN", "1")
    if threshold is None:
        monkeypatch.delenv("DSV41_PREFILL_FLASH_MIN_N", raising=False)
    else:
        monkeypatch.setenv("DSV41_PREFILL_FLASH_MIN_N", threshold)
    expected = n > 8 and n >= int(threshold or "64")
    assert dispatch.prefill_flash_enabled(n) is expected
    monkeypatch.setenv("DSV41_PREFILL_FLASH_ATTN", "0")
    assert not dispatch.prefill_flash_enabled(n)


@pytest.mark.parametrize("n", [6, 8, 63])
@pytest.mark.parametrize("metal", ["0", "1"])
@pytest.mark.parametrize("threshold", [None, "1", "128"])
def test_short_batches_keep_existing_route(n, metal, threshold, monkeypatch):
    if threshold == "1" and n > 8:
        pytest.skip("N=63 is intentionally eligible when the threshold is lowered")
    from mlx_vlm.custom_kernels.deepseek_v41 import sparse_attention_prefill as module

    def forbidden(*args):
        pytest.fail("short batch entered the prefill kernel")

    monkeypatch.setattr(module, "sparse_attn_prefill", forbidden)
    monkeypatch.setenv("DSV41_METAL_SPARSE_ATTN", metal)
    monkeypatch.setenv("DSV41_SDPA_SPARSE_ATTN", "0")
    if threshold is None:
        monkeypatch.delenv("DSV41_PREFILL_FLASH_MIN_N", raising=False)
    else:
        monkeypatch.setenv("DSV41_PREFILL_FLASH_MIN_N", threshold)
    case = _case(n, 128, "mixed", h=8)
    for path in (dispatch.sparse_attention, sparse_attn):
        with _flash_switch("0"):
            expected = path(*case)
        with _flash_switch("1"):
            actual = path(*case)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)),
                                      np.asarray(expected.astype(mx.float32)))


def timing(repeats=5):
    """Warm, interleaved per-layer attention timing; no model or speed claims."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for context in (6000, 30000):
        case = _case(2048, context, "mixed")
        mx.eval(case)
        paths = (_three_stage, sparse_attn_prefill_fused, sparse_attn_prefill)
        names = ("three_stage", "fused", "gather_mma")
        for path in paths:
            for _ in range(2):
                mx.eval(path(*case))
        samples = [[], [], []]
        for rep in range(repeats):
            for index in ((rep + j) % len(paths) for j in range(len(paths))):
                mx.synchronize()
                start = perf_counter()
                mx.eval(paths[index](*case))
                mx.synchronize()
                samples[index].append((perf_counter() - start) * 1000)
        flops = 4 * 2048 * 640 * 64 * 512
        summary = "; ".join(
            f"{name}={median(values):.3f} ms "
            f"({flops / (median(values) * 1e9):.3f} nominal TFLOP/s)"
            for name, values in zip(names, samples)
        )
        print(f"n=2048 context={context}, attention ms/layer-equivalent: "
              f"{summary}; gather buffer=1.25 GiB, read+write=2.684 GB; "
              f"samples={samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    timing(args.repeats)
