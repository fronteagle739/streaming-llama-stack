"""Mac/Metal acceptance: zero bit mismatches, then 5 warm + 10 timed samples.

Default feature widths are D=128/I=96, retaining exact E=384 and routing rows.
DSV41_EXPERT_TILES_TEST_RELEASE=1 selects checkpoint widths D=5120/I=2304.
No tolerance gates, BF16 casts of FP32 down, or expected-failure bit tests.
"""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import pytest

if platform.system() != "Darwin":
    pytest.skip("Mac Metal tests; run test_expert_tiles_cpu.py on Windows", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

from mlx_vlm.custom_kernels.deepseek_v41 import expert_tiles_qmm as kernel
from mlx_vlm.custom_kernels.deepseek_v41 import moe_prefill_expert_tiles as adapter
from mlx_vlm.custom_kernels.deepseek_v41.expert_tiles_fixtures import (
    E, SEED, metadata_numpy, routes,
)
from mlx_vlm.models.deepseek_v41.language import ClampedSwiGLU

RELEASE = os.environ.get("DSV41_EXPERT_TILES_TEST_RELEASE", "0") == "1"
D, I = (5120, 2304) if RELEASE else (128, 96)
CASES = ["probe_a", "balanced32", "8x1536", "singleton_empty", "boundaries"]
CASES += [f"remainder_{r}" for r in range(16)]


def assert_bits(label, actual, stock):
    assert actual.shape == stock.shape and actual.dtype == stock.dtype, label
    assert actual.dtype in (mx.bfloat16, mx.float32)
    bit_dtype = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    # These are test boundaries, the only place values are read back.
    mx.eval(actual, stock)
    a, b = np.asarray(actual.view(bit_dtype)), np.asarray(stock.view(bit_dtype))
    bad = np.flatnonzero(a.ravel() != b.ravel())
    first = None if not bad.size else dict(
        index=np.unravel_index(int(bad[0]), a.shape),
        actual=hex(int(a.ravel()[bad[0]])), stock=hex(int(b.ravel()[bad[0]])))
    print(f"{label}: mismatches={bad.size}/{a.size} first={first}")
    assert bad.size == 0, f"{label}: {bad.size} bit mismatches; first={first}"


def stock_qmm(x, layer, ids):
    return mx.gather_qmm(x, layer.weight, layer.scales, rhs_indices=ids,
                         transpose=True, group_size=32, bits=4, mode="mxfp4",
                         sorted_indices=True)


def measure(label, arms):
    samples = {name: [] for name in arms}
    for step in range(15):
        order = list(arms)
        if step % 2:
            order.reverse()
        for name in order:
            mx.synchronize()
            start = perf_counter()
            y = arms[name]()  # Construct a fresh lazy graph for every sample.
            mx.eval(y)
            mx.synchronize()
            if step >= 5:
                samples[name].append((perf_counter() - start) * 1000)
            del y
    for name, values in samples.items():
        assert len(values) == 10
        print(json.dumps(dict(timing=label, arm=name, warmups=5, samples=10,
                              median_ms=median(values), samples_ms=values)))


@pytest.fixture(scope="module", autouse=True)
def runtime():
    assert version("mlx") == "0.32.2", "The bit oracle must be unmodified MLX 0.32.2"
    assert mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    start = perf_counter()
    kernel.prewarm_expert_tiles()  # Compile before collecting any timings.
    print(json.dumps(dict(mlx=version("mlx"), mlx_lm=version("mlx-lm"),
        device=mx.metal.device_info(), D=D, I=I, E=E,
        cold_compile_ms=(perf_counter() - start) * 1000,
        source_sha256=sha256(Path(kernel.__file__).with_suffix(".metal").read_bytes()).hexdigest(),
        source_hashes=kernel.SOURCE_HASHES)))


def make_bank(n, k, seed):
    """Build one dense expert at a time to bound release-width peak memory."""
    mx.random.seed(seed)
    packed, scales = [], []
    for _ in range(E):
        w = (mx.random.normal((1, n, k)) / k**0.5).astype(mx.bfloat16)
        q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
        mx.eval(q, s)
        packed.append(q)
        scales.append(s)
    q, s = mx.concatenate(packed), mx.concatenate(scales)
    mx.eval(q, s)
    return q, s


@pytest.fixture(scope="module")
def mlp():
    # Tiny placeholders avoid constructing unused checkpoint-sized dense banks.
    mlp = SwitchGLU(32, 32, 1, activation=ClampedSwiGLU(10.0), bias=False)
    for i, name in enumerate(("gate_proj", "up_proj", "down_proj")):
        n, k = (D, I) if name == "down_proj" else (I, D)
        layer = QuantizedSwitchLinear(32, 32, 1, bias=False,
                                      group_size=32, bits=4, mode="mxfp4")
        layer.weight, layer.scales = make_bank(n, k, SEED + i)
        setattr(mlp, name, layer)
    mlp.eval()
    adapter.validate_mlp(mlp)
    return mlp


@pytest.mark.parametrize("case", [*CASES, "empty"])
def test_gpu_metadata(case):
    ids_np = np.sort(routes(case).ravel())
    routing = kernel.expert_tiles_metadata(mx.array(ids_np))
    mx.eval(routing.offsets, routing.prefix)
    offsets, prefix = metadata_numpy(ids_np)
    np.testing.assert_array_equal(np.asarray(routing.offsets), offsets)
    np.testing.assert_array_equal(np.asarray(routing.prefix), prefix)
    assert routing.candidates == (len(ids_np) + 15) // 16 + E


@pytest.mark.parametrize("case", CASES)
def test_projection_bits(mlp, case):
    ids = mx.array(np.sort(routes(case).ravel()))
    # Ensure the stock oracle dispatches fp_gather_qmm_rhs, not QMV.
    assert ids.size >= 4 * E
    routing = kernel.expert_tiles_metadata(ids)
    mx.random.seed(SEED)
    x = mx.random.normal((ids.size, 1, D)).astype(mx.bfloat16)
    gate = stock_qmm(x, mlp.gate_proj, ids)
    up = stock_qmm(x, mlp.up_proj, ids)
    custom_gate = kernel.expert_tiles_qmm(x, mlp.gate_proj.weight, mlp.gate_proj.scales, routing)
    custom_up = kernel.expert_tiles_qmm(x, mlp.up_proj.weight, mlp.up_proj.scales, routing)
    assert_bits(case + "/gate", custom_gate, gate)
    assert_bits(case + "/up", custom_up, up)
    hidden = mlp.activation(up, gate)
    custom_hidden = mlp.activation(custom_up, custom_gate)
    assert hidden.dtype == mx.float32
    assert_bits(case + "/activation", custom_hidden, hidden)
    down = stock_qmm(hidden, mlp.down_proj, ids)
    custom_down = kernel.expert_tiles_qmm(custom_hidden, mlp.down_proj.weight,
                                         mlp.down_proj.scales, routing)
    assert_bits(case + "/down-FP32", custom_down, down)


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_adversarial_magnitudes_signed_zeros(dtype):
    rng = np.random.default_rng(SEED)
    ids = mx.array(np.sort(routes("boundaries")))
    routing = kernel.expert_tiles_metadata(ids)
    # Every E2M1 code, including positive and negative zero, at varied scales.
    q = mx.array(rng.integers(0, 2**32, (E, 32, 8), dtype=np.uint32))
    s = mx.array(rng.choice(np.array([0, 64, 110, 120, 127, 135], np.uint8), (E, 32, 2)))
    values = np.array([0.0, -0.0, 2**-16, -(2**-16), .5, -.5, 1, -1, 128, -128], np.float32)
    x = mx.array(rng.choice(values, (ids.size, 1, 64)))
    x = x.astype(mx.bfloat16 if dtype == "bf16" else mx.float32)
    actual = kernel.expert_tiles_qmm(x, q, s, routing)
    stock = mx.gather_qmm(x, q, s, rhs_indices=ids, transpose=True,
                          group_size=32, bits=4, mode="mxfp4", sorted_indices=True)
    assert_bits("adversarial/" + dtype, actual, stock)


@pytest.mark.parametrize("case", ["probe_a", "balanced32", "8x1536"])
def test_projection_timing(mlp, case):
    ids = mx.array(np.sort(routes(case).ravel()))
    routing = kernel.expert_tiles_metadata(ids)
    mx.random.seed(SEED)
    x = mx.random.normal((ids.size, 1, D)).astype(mx.bfloat16)
    hidden = mlp.activation(stock_qmm(x, mlp.up_proj, ids), stock_qmm(x, mlp.gate_proj, ids))
    mx.eval(x, hidden, ids, routing.offsets, routing.prefix)
    for name, value in (("gate", x), ("up", x), ("down", hidden)):
        layer = getattr(mlp, name + "_proj")
        stock = lambda: stock_qmm(value, layer, ids)
        custom = lambda: kernel.expert_tiles_qmm(value, layer.weight, layer.scales, routing)
        assert_bits(case + "/timing-gate/" + name, custom(), stock())
        measure(case + "/projection/" + name, dict(stock=stock, custom=custom))


@pytest.mark.parametrize("case,n", [("probe_a", 2048), ("balanced32", 2048),
                                    ("8x1536", 2048), ("probe_a", 64), ("probe_a", 257)])
def test_whole_routed_ffn(mlp, case, n, monkeypatch, capsys):
    # n=64 deliberately tests stock's different QMV dispatch. It is a strict
    # extra acceptance gate, not an assumption of RHS-QMM bit equivalence.
    ids = mx.array(routes(case)[:n])
    mx.random.seed(SEED)
    x = mx.random.normal((n, D)).astype(mx.bfloat16)
    weights = mx.random.uniform(shape=(n, 6))
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    mx.eval(x, ids, weights)
    seen = []
    original_meta, original_qmm = adapter.expert_tiles_metadata, adapter.expert_tiles_qmm

    def metadata(*args):
        plan = original_meta(*args)
        seen.append(("metadata", plan))
        return plan

    def project(value, w, s, plan):
        seen.append(("qmm", plan))
        return original_qmm(value, w, s, plan)

    monkeypatch.setattr(adapter, "_ENGAGED", False)
    monkeypatch.setattr(adapter, "expert_tiles_metadata", metadata)
    monkeypatch.setattr(adapter, "expert_tiles_qmm", project)
    stock_routed = mlp(x, ids)
    custom_routed = adapter.moe_prefill_expert_tiles(mlp, x, ids)
    assert_bits(f"{case}/{n}/unsorted", custom_routed, stock_routed)
    assert [name for name, _ in seen] == ["metadata", "qmm", "qmm", "qmm"]
    assert all(plan is seen[0][1] for _, plan in seen)
    adapter.moe_prefill_expert_tiles(mlp, x, ids)
    assert capsys.readouterr().out.count("FFN_EXPERT_TILES engaged") == 1
    monkeypatch.setattr(adapter, "expert_tiles_metadata", original_meta)
    monkeypatch.setattr(adapter, "expert_tiles_qmm", original_qmm)

    def combine(y):
        return mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)

    stock = lambda: combine(mlp(x, ids))
    custom = lambda: combine(adapter.moe_prefill_expert_tiles(mlp, x, ids))
    assert_bits(f"{case}/{n}/FP32-combine", custom(), stock())
    # Includes sort, one metadata kernel, all projections, activation, unsort,
    # and FP32 weighted combine. Routing indices/weights are fixed inputs.
    measure(f"{case}/{n}/whole-routed-FFN", dict(stock=stock, custom=custom))


def test_rejects_unsupported_inputs():
    ids = mx.zeros((1536,), mx.uint32)
    with pytest.raises(ValueError, match="E=384"):
        kernel.expert_tiles_metadata(ids, 8)
    with pytest.raises(ValueError, match="uint32"):
        kernel.expert_tiles_metadata(ids.astype(mx.int32))
    routing = kernel.expert_tiles_metadata(ids)
    for k, n in ((40, 32), (32, 33)):
        with pytest.raises(ValueError, match="multiples of 32"):
            kernel.expert_tiles_qmm(mx.zeros((1536, 1, k), mx.bfloat16),
                mx.zeros((E, n, k // 8), mx.uint32),
                mx.zeros((E, n, k // 32), mx.uint8), routing)
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        kernel.expert_tiles_qmm(mx.zeros((1536, 1, 32), mx.float16),
            mx.zeros((E, 32, 4), mx.uint32), mx.zeros((E, 32, 1), mx.uint8), routing)
