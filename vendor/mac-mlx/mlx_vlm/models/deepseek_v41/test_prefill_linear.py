"""Synthetic Mac checks and timings; no checkpoint required.

Run this file with --timing for the interleaved qmm/dequant+Steel sweep.
"""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform
import runpy
from time import perf_counter

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn


DENSE_SHAPES = (
    (32768, 1280),  # attention wq_b
    (5120, 8192),   # attention wo_b
    (1280, 5120),   # attention wq_a
    (512, 5120),    # attention wkv
    (2304, 5120),   # shared gate/up
    (5120, 2304),   # shared down
)


def _load(monkeypatch, enabled="1", min_m=None):
    for key, value in (("DSV41_PREFILL_MXFP8_STEEL", enabled),
                       ("DSV41_PREFILL_STEEL_MIN_M", min_m)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    # Isolate switches and counters without importing the full model package.
    namespace = runpy.run_path(str(Path(__file__).with_name("prefill_linear.py")))
    return namespace["prefill_linear"]


def _layer(n, k, bias=False):
    layer = nn.QuantizedLinear(
        k, n, bias=bias, group_size=32, bits=8, mode="mxfp8",
    )
    weights = (mx.random.normal((n, k)) * k ** -0.5).astype(mx.bfloat16)
    layer.weight, layer.scales = mx.quantize(
        weights, group_size=32, bits=8, mode="mxfp8",
    )
    if bias:
        layer.bias = mx.random.normal((n,)).astype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer


def _qmm(layer, x):
    out = mx.quantized_matmul(
        x, layer.weight, layer.scales, transpose=True,
        group_size=32, bits=8, mode="mxfp8",
    )
    bias = getattr(layer, "bias", None)
    return out if bias is None else out + bias


def _errors(actual, baseline):
    actual, baseline = actual.astype(mx.float32), baseline.astype(mx.float32)
    error = actual - baseline
    rms = mx.sqrt(mx.mean(mx.square(baseline)))
    # Near-zero entries use 0.1% of the baseline RMS as denominator floor.
    floor = mx.maximum(rms * 1e-3, 1e-12)
    values = (
        mx.max(mx.abs(error)),
        mx.sqrt(mx.mean(mx.square(error))) / mx.maximum(rms, 1e-12),
        mx.max(mx.abs(error) / mx.maximum(mx.abs(baseline), floor)),
        mx.all(mx.isfinite(actual)) & mx.all(mx.isfinite(baseline)),
    )
    mx.eval(*values)
    return dict(zip(("max_abs", "rms_relative", "near_zero_aware", "finite"),
                    (value.item() for value in values)))


@pytest.mark.parametrize("n,k", DENSE_SHAPES)
def test_dense_numerics(monkeypatch, n, k):
    linear = _load(monkeypatch)
    mx.random.seed(41)
    layer = _layer(n, k)
    x = mx.random.normal((2048, k)).astype(mx.bfloat16)
    mx.eval(x)
    baseline = _qmm(layer, x)
    actual = linear(layer, x)
    oracle = x.astype(mx.float32) @ mx.dequantize(
        layer.weight, layer.scales, getattr(layer, "biases", None),
        group_size=32, bits=8, mode="mxfp8", dtype=mx.float32,
    ).T
    candidate_error = _errors(actual, baseline)
    oracle_error = _errors(oracle, baseline)
    print(f"M=2048 N={n} K={k}: Steel={candidate_error}; FP32 oracle={oracle_error}")
    assert actual.dtype == baseline.dtype == mx.bfloat16
    assert linear.__globals__["STEEL_CALLS"] == 1
    assert candidate_error["finite"] and oracle_error["finite"]
    for metric in ("max_abs", "rms_relative"):
        assert candidate_error[metric] <= 2 * oracle_error[metric], (
            metric, candidate_error, oracle_error,
        )


@pytest.mark.parametrize("enabled", [None, "0"])
def test_off_calls_layer_identically(monkeypatch, enabled):
    linear = _load(monkeypatch, enabled, min_m="ignored-while-off")
    x, result = object(), object()
    calls = []

    class Layer:
        def __call__(self, value):
            calls.append(value)
            return result

    # x deliberately has no shape/dtype: OFF must not even inspect it.
    assert linear(Layer(), x) is result
    assert len(calls) == 1 and calls[0] is x
    assert linear.__globals__["STEEL_CALLS"] == 0


@pytest.mark.parametrize("case", ["small_m", "vector", "fp32", "fp16",
                                  "plain", "mode", "group_size", "bits"])
def test_ineligible_calls_layer_identically(monkeypatch, case):
    linear = _load(monkeypatch)
    calls, result = [], object()

    class Layer(nn.QuantizedLinear):
        def __call__(self, value):
            calls.append(value)
            return result

    layer = Layer(64, 32, bias=False, group_size=32, bits=8, mode="mxfp8")
    shape = (16, 64) if case == "small_m" else (64,) if case == "vector" else (256, 64)
    dtype = {"fp32": mx.float32, "fp16": mx.float16}.get(case, mx.bfloat16)
    x = mx.zeros(shape, dtype=dtype)
    if case in ("mode", "group_size", "bits"):
        setattr(layer, case, {"mode": "affine", "group_size": 64, "bits": 4}[case])
    if case == "plain":
        def layer(value):
            calls.append(value)
            return result
    assert linear(layer, x) is result
    assert len(calls) == 1 and calls[0] is x
    assert linear.__globals__["STEEL_CALLS"] == 0


@pytest.mark.parametrize("bias", [False, True])
def test_threshold_bias_and_per_call_dequantization(monkeypatch, bias):
    linear = _load(monkeypatch, min_m="16")
    mx.random.seed(42)
    layer = _layer(32, 64, bias=bias)
    x = mx.random.normal((1, 16, 64)).astype(mx.bfloat16)
    original = mx.dequantize
    calls = []

    def dequantize(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "dequantize", dequantize)
    before = {name: id(value) for name, value in layer.items()}
    for _ in range(2):
        actual = linear(layer, x)
        expected = x @ original(
            layer.weight, layer.scales, getattr(layer, "biases", None),
            group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16,
        ).T
        if bias:
            expected = expected + layer.bias
        assert mx.array_equal(actual, expected).item()
    assert len(calls) == linear.__globals__["STEEL_CALLS"] == 2
    assert all(call["dtype"] == mx.bfloat16 for call in calls)
    assert {name: id(value) for name, value in layer.items()} == before


def _timed_ms(fn):
    mx.synchronize()
    start = perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - start) * 1000


def timing_helper():
    """Include a fresh dequantization in every candidate sample; no test asserts."""
    print(f"MLX {version('mlx')}; {platform.platform()}; {platform.machine()}")
    print("Record the installed MLX source commit separately for source builds.")
    print("Forced MIN_M=16 for the sweep; production default remains 256.")
    with pytest.MonkeyPatch.context() as monkeypatch:
        linear = _load(monkeypatch, min_m="16")
    for n, k in DENSE_SHAPES:
        mx.random.seed(41)
        layer = _layer(n, k)
        for m in (16, 32, 128, 256, 512, 2048):
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            mx.eval(x)
            functions = {"qmm": lambda: _qmm(layer, x),
                         "dequant+Steel": lambda: linear(layer, x)}
            first = {name: _timed_ms(fn) for name, fn in functions.items()}
            for _ in range(5):
                for fn in functions.values():
                    _timed_ms(fn)
            samples = {name: [] for name in functions}
            names = list(functions)
            for repeat in range(20):
                # Alternate which kernel goes first to reduce ordering bias.
                for name in names[::1 if repeat % 2 == 0 else -1]:
                    samples[name].append(_timed_ms(functions[name]))
            print(f"M={m} N={n} K={k}; first-call ms (may compile)={first}")
            for name, values in samples.items():
                p10, median, p90 = np.percentile(values, [10, 50, 90])
                print(f"  {name}: median={median:.3f} p10={p10:.3f} p90={p90:.3f} ms")
        # Release each weight bank before creating the next one.
        del functions, layer, x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.timing:
        timing_helper()
    else:
        parser.error("use --timing, or run this file with pytest")
