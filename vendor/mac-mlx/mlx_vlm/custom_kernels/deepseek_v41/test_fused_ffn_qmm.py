"""Mac acceptance: uint32 identity is mandatory, then 5 warm + 10 timed.

DSV41_FFN_FUSED_TEST_RELEASE=1 uses D5120/I2304; default D128/I96.
Run with -s to retain mismatch counts, source hashes and interleaved timings.
"""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pytest

if platform.system() != "Darwin":
    pytest.skip("Mac Metal acceptance; use test_fused_ffn_cpu.py on Windows", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

from mlx_vlm.custom_kernels.deepseek_v41 import fused_ffn_qmm as kernel
from mlx_vlm.custom_kernels.deepseek_v41 import moe_prefill_fused_ffn as adapter
from mlx_vlm.custom_kernels.deepseek_v41.expert_tiles_fixtures import E, SEED, routes, metadata_numpy
from mlx_vlm.custom_kernels.deepseek_v41.test_expert_tiles_qmm import (
    assert_bits, make_bank, measure, stock_qmm,
)
from mlx_vlm.models.deepseek_v41.language import ClampedSwiGLU

D, I = (5120, 2304) if os.environ.get("DSV41_FFN_FUSED_TEST_RELEASE", "0") == "1" else (128, 96)
CASES = ["probe_a", "balanced32", "8x1536", "singleton_empty", "fused_boundaries"]


@pytest.fixture(scope="module", autouse=True)
def runtime():
    assert version("mlx") == "0.32.2"
    assert mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    start = perf_counter()
    kernel.prewarm_fused_ffn()
    print(json.dumps(dict(mlx=version("mlx"), mlx_lm=version("mlx-lm"),
        device=mx.metal.device_info(), D=D, I=I, E=E,
        cold_compile_ms=(perf_counter() - start) * 1000,
        fused_sha256=sha256(Path(kernel.__file__).with_suffix(".metal").read_bytes()).hexdigest(),
        base_header_sha256=sha256(kernel._base_sources()[0].encode()).hexdigest())))


@pytest.fixture(scope="module")
def mlp():
    model = SwitchGLU(32, 32, 1, activation=ClampedSwiGLU(10.0), bias=False)
    for i, name in enumerate(("gate_proj", "up_proj", "down_proj")):
        n, k = (D, I) if name == "down_proj" else (I, D)
        layer = QuantizedSwitchLinear(32, 32, 1, bias=False,
                                      group_size=32, bits=4, mode="mxfp4")
        layer.weight, layer.scales = make_bank(n, k, SEED + i)
        setattr(model, name, layer)
    model.eval()
    adapter.validate_mlp(model)
    return model


def fused(x, mlp, plan):
    g, u = mlp.gate_proj, mlp.up_proj
    return kernel.expert_gate_up_swiglu(x, g.weight, g.scales, u.weight, u.scales,
                                       plan, mlp.activation.limit)


@pytest.mark.parametrize("case", CASES + ["empty"])
def test_metadata(case):
    ids = np.sort(routes(case).ravel())
    plan = kernel.fused_ffn_metadata(mx.array(ids))
    mx.eval(plan.offsets, plan.prefix)
    offsets, prefix = metadata_numpy(ids, 32)
    np.testing.assert_array_equal(np.asarray(plan.offsets), offsets)
    np.testing.assert_array_equal(np.asarray(plan.prefix), prefix)
    assert plan.candidates == (ids.size + 31) // 32 + E and plan.tile_rows == 32


@pytest.mark.parametrize("case", CASES)
def test_hidden_bits(mlp, case):
    ids = mx.array(np.sort(routes(case).ravel()))
    assert ids.size >= 4 * E  # Stock sorted GEMM dispatch; small QMV tested below.
    plan = kernel.fused_ffn_metadata(ids)
    mx.random.seed(SEED)
    x = mx.random.normal((ids.size, 1, D)).astype(mx.bfloat16)
    gate, up = stock_qmm(x, mlp.gate_proj, ids), stock_qmm(x, mlp.up_proj, ids)
    assert gate.dtype == up.dtype == mx.bfloat16
    stock = mlp.activation(up, gate)
    assert stock.dtype == mx.float32
    assert_bits(case + "/hidden-uint32", fused(x, mlp, plan), stock)


@pytest.mark.parametrize("limit", [0.0, -1.0, 0.5, 10.0, 10.03125])
def test_clamp_boundaries_and_bf16_rounding(limit):
    # gate = x0 + x2, up = x1 + x3; 0.03125 creates BF16 halfway sums
    # near +/-10. Each channel uses the same exact E2M1=1, E8M0=127 weights.
    ids = mx.array(np.sort(routes("fused_boundaries")))
    plan = kernel.fused_ffn_metadata(ids)
    gate_w = np.zeros((E, 32, 4), np.uint32)
    up_w = gate_w.copy()
    gate_w[:, :, 0], up_w[:, :, 0] = 0x0202, 0x2020
    g, u = mx.array(gate_w), mx.array(up_w)
    s = mx.full((E, 32, 1), 127, mx.uint8)
    values = np.array([-128, -10.0625, -10, -9.9375, -.5, -0., 0., .5,
                       9.9375, 10, 10.0625, 128], np.float32)
    x_np = np.zeros((ids.size, 1, 32), np.float32)
    rows = np.arange(ids.size)
    x_np[:, 0, 0] = values[rows % len(values)]
    x_np[:, 0, 1] = values[(rows // len(values)) % len(values)]
    x_np[:, 0, 2] = np.array([0., .03125, -.03125])[rows % 3]
    x_np[:, 0, 3] = np.array([0., -.03125, .03125])[(rows // 3) % 3]
    x = mx.array(x_np).astype(mx.bfloat16)
    from types import SimpleNamespace
    gate = stock_qmm(x, SimpleNamespace(weight=g, scales=s), ids)
    up = stock_qmm(x, SimpleNamespace(weight=u, scales=s), ids)
    stock = ClampedSwiGLU(limit)(up, gate)
    actual = kernel.expert_gate_up_swiglu(x, g, s, u, s, plan, limit)
    assert_bits(f"boundary/limit={limit}", actual, stock)


def test_decode_magnitudes_and_cancellation():
    rng = np.random.default_rng(SEED)
    ids = mx.array(np.sort(routes("fused_boundaries")))
    plan = kernel.fused_ffn_metadata(ids)
    banks = [mx.array(rng.integers(0, 2**32, (E, 32, 8), dtype=np.uint32)) for _ in range(2)]
    scales = [mx.array(rng.choice(np.array([0, 64, 110, 120, 127, 135], np.uint8),
                                  (E, 32, 2))) for _ in range(2)]
    x = mx.array(rng.choice(np.array([0., -0., 2**-16, -2**-16, .5, -.5, 1., -1., 128., -128.],
                                    np.float32), (ids.size, 1, 64))).astype(mx.bfloat16)
    from types import SimpleNamespace
    g, u = [stock_qmm(x, SimpleNamespace(weight=w, scales=s), ids) for w, s in zip(banks, scales)]
    actual = kernel.expert_gate_up_swiglu(x, banks[0], scales[0], banks[1], scales[1], plan, 10.)
    assert_bits("decode/cancellation", actual, ClampedSwiGLU(10.)(u, g))


@pytest.mark.parametrize("case", ["probe_a", "balanced32", "8x1536"])
def test_gate_up_activation_timing(mlp, case):
    ids = mx.array(np.sort(routes(case).ravel()))
    plan = kernel.fused_ffn_metadata(ids)
    mx.random.seed(SEED)
    x = mx.random.normal((ids.size, 1, D)).astype(mx.bfloat16)
    mx.eval(x, ids, plan.offsets, plan.prefix)
    stock = lambda: mlp.activation(stock_qmm(x, mlp.up_proj, ids), stock_qmm(x, mlp.gate_proj, ids))
    custom = lambda: fused(x, mlp, plan)
    assert_bits(case + "/timing-gate", custom(), stock())
    measure(case + "/gate+up+activation", dict(stock=stock, fused=custom))


@pytest.mark.parametrize("case,n", [("probe_a", 2048), ("balanced32", 2048),
                                    ("8x1536", 2048), ("probe_a", 64), ("probe_a", 257)])
def test_whole_routed_ffn(mlp, case, n, monkeypatch, capsys):
    # n=64 also gates exactness against stock's QMV dispatch; never xfail it.
    ids = mx.array(routes(case)[:n])
    mx.random.seed(SEED)
    x = mx.random.normal((n, D)).astype(mx.bfloat16)
    weights = mx.random.uniform(shape=(n, 6))
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    mx.eval(x, ids, weights)
    monkeypatch.setattr(adapter, "_ENGAGED", False)
    assert_bits(f"{case}/{n}/unsorted-down", adapter.moe_prefill_fused_ffn(mlp, x, ids), mlp(x, ids))
    adapter.moe_prefill_fused_ffn(mlp, x, ids)
    assert capsys.readouterr().out.count("FFN_FUSED engaged") == 1

    def combine(y):
        return mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)

    stock = lambda: combine(mlp(x, ids))
    custom = lambda: combine(adapter.moe_prefill_fused_ffn(mlp, x, ids))
    assert_bits(f"{case}/{n}/FP32-combine", custom(), stock())
    measure(f"{case}/{n}/whole-routed-FFN", dict(stock=stock, fused=custom))


def test_empty_and_rejections():
    ids = mx.zeros((0,), mx.uint32)
    plan = kernel.fused_ffn_metadata(ids)
    w, s = mx.zeros((E, 32, 4), mx.uint32), mx.full((E, 32, 1), 127, mx.uint8)
    x = mx.zeros((0, 1, 32), mx.bfloat16)
    assert kernel.expert_gate_up_swiglu(x, w, s, w, s, plan, 10.).shape == (0, 1, 32)
    with pytest.raises(ValueError, match="BF16"):
        kernel.expert_gate_up_swiglu(x.astype(mx.float32), w, s, w, s, plan, 10.)
    with pytest.raises(ValueError, match="M32"):
        kernel.expert_gate_up_swiglu(x, w, s, w, s, adapter.expert_tiles_metadata(ids), 10.)
    with pytest.raises(ValueError, match="finite"):
        kernel.expert_gate_up_swiglu(x, w, s, w, s, plan, float("nan"))
