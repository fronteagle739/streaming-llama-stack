"""Mac differential gates and timing (no checkpoint required).

Default tests retain E=384/top6/token counts with reduced feature widths.
DSV41_GROUPED_TEST_RELEASE=1 selects D=5120/I=2304 for the same cases.
Run as a module with --timing for isolated and composed layer timings.
"""
import gc
import json
import os
import platform
from types import SimpleNamespace
import sys

# Dispatch the controller before importing MLX/test dependencies. The standalone
# grouped_gemm_harness.py entry point also avoids mlx_vlm's package __init__.
from mlx_vlm.custom_kernels.deepseek_v41.grouped_gemm_harness import (
    LaunchHung, Variant, bounded_call, controller, evaluate, parse_args, report, shapes_for,
)

if __name__ == "__main__":
    _cli_args = parse_args()
    if not _cli_args.worker:
        sys.exit(controller(_cli_args))

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires Apple Metal", allow_module_level=True)

from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

from mlx_vlm.custom_kernels.deepseek_v41.grouped_gemm import (
    _grouped_mxfp4, _source, _tuning_source, expert_offsets, grouped_mxfp4, prepare_worklist,
)
from mlx_vlm.custom_kernels.deepseek_v41.grouped_gemm_tuning import CONFIGS, launch_info
from mlx_vlm.models.deepseek_v41.language import ClampedSwiGLU, MoE
from mlx_vlm.models.deepseek_v41.moe_prefill import moe_prefill_grouped

RELEASE = os.environ.get("DSV41_GROUPED_TEST_RELEASE", "0") == "1"
E, TOPK = 384, 6
D, I = (5120, 2304) if RELEASE else (128, 96)
CASES = ("balanced32", "random2048", "random8192", "skewed", "empty", "tails")
_selected = os.environ.get("DSV41_GROUPED_TEST_CONFIGS", "fast")
TEST_CONFIGS = [None] + list(CONFIGS if _selected == "all" else _selected.split(","))


def assert_bits(label, actual, reference):
    """Require identical storage, including signed zero and FP32 down low bits."""
    assert actual.dtype == reference.dtype, (label, actual.dtype, reference.dtype)
    dtype = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    a, b = np.array(actual.view(dtype)), np.array(reference.view(dtype))
    mismatch = int(np.count_nonzero(a != b))
    print(json.dumps(dict(bit_check=label, mismatched_elements=mismatch, elements=a.size)))
    assert mismatch == 0, f"{label}: {mismatch}/{a.size} elements differ in storage bits"


@pytest.fixture(scope="module", autouse=True)
def record_runtime():
    print(json.dumps(dict(mlx=mx.__version__, mlx_path=mx.__file__, release=RELEASE,
                          mlx_commit=os.environ.get("DSV41_MLX_COMMIT", "unrecorded"),
                          device=mx.metal.device_info())))


def routes(case, experts=E):
    rng = np.random.default_rng(4101)
    if case == "balanced32":
        ids = (np.arange(2048 * TOPK) % experts).reshape(2048, TOPK)
    elif case == "tails":
        lengths = np.zeros(experts, dtype=np.int32)
        lengths[1:9] = [1, 15, 16, 17, 31, 32, 33, 65]
        return np.repeat(np.arange(experts), lengths).astype(np.uint32), lengths
    else:
        tokens = 8192 if case == "random8192" else 2048
        pool = experts // 4 if case == "empty" else experts
        p = None
        if case == "skewed":
            p = np.ones(pool)
            p[:8] = 200
            p /= p.sum()
        ids = np.stack([rng.choice(pool, TOPK, replace=False, p=p) for _ in range(tokens)])
    counts = np.bincount(ids.ravel(), minlength=experts).astype(np.int32)
    assert counts.sum() == ids.size
    assert np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0)
    return ids.astype(np.uint32), counts


def bank(experts, n, k, seed=41):
    """Quantize/evaluate one expert at a time; never allocate FP32 [E,N,K]."""
    mx.random.seed(seed)
    packed, scales = [], []
    for _ in range(experts):
        w = (mx.random.normal((1, n, k)) / k**0.5).astype(mx.bfloat16)
        q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
        mx.eval(q, s)
        packed.append(q)
        scales.append(s)
    q, s = mx.concatenate(packed), mx.concatenate(scales)
    mx.eval(q, s)
    return q, s


def baseline(x, q, s, ids):
    return mx.gather_qmm(x[:, None, :], q, s, rhs_indices=ids,
                         transpose=True, group_size=32, bits=4,
                         mode="mxfp4", sorted_indices=True).squeeze(-2)


def oracle(x, q, s, counts):
    """Explicit FP32 dequantized matmul, one expert's dense weights at a time."""
    pieces, start = [], 0
    for e, count in enumerate(counts):
        if count:
            w = mx.dequantize(q[e], s[e], group_size=32, bits=4,
                              mode="mxfp4", dtype=mx.float32)
            y = x[start:start + int(count)].astype(mx.float32) @ w.T
            mx.eval(y)
            pieces.append(y)
            start += int(count)
    result = mx.concatenate(pieces)
    mx.eval(result)
    return result


def metrics(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    err = a - b
    rms = mx.sqrt(mx.mean(b * b))
    floor = mx.maximum(rms * 0.01, 1e-12)
    values = [mx.max(mx.abs(err)), mx.sqrt(mx.mean(err * err)) / mx.maximum(rms, 1e-12),
              mx.max(mx.abs(err) / mx.maximum(mx.abs(b), floor)),
              mx.all(mx.isfinite(a)) & mx.all(mx.isfinite(b)), rms]
    mx.eval(*values)
    return dict(zip(("max_abs", "rms_relative", "near_zero_aware", "finite", "reference_rms"),
                    [v.item() for v in values]))


def accept(label, actual, reference, truth):
    diff, base, candidate = metrics(actual, reference), metrics(reference, truth), metrics(actual, truth)
    print(json.dumps(dict(case=label, vs_gather=diff, gather_vs_oracle=base,
                          grouped_vs_oracle=candidate)))
    assert diff["finite"] and base["finite"] and candidate["finite"]
    # Calibrate against baseline's own oracle error, with a small FP32
    # reduction-order floor when the baseline happens to be exact.
    floor = 8e-6
    assert candidate["rms_relative"] <= 1.25 * base["rms_relative"] + floor
    assert candidate["max_abs"] <= 1.5 * base["max_abs"] + floor * base["reference_rms"]
    assert candidate["near_zero_aware"] <= 1.5 * base["near_zero_aware"] + floor / 0.01


@pytest.fixture(scope="module", params=("gate", "up", "down"))
def projection(request):
    name = request.param
    n, k = (D, I) if name == "down" else (I, D)
    q, s = bank(E, n, k, seed={"gate": 41, "up": 42, "down": 43}[name])
    yield name, q, s, k
    del q, s
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("config_name", TEST_CONFIGS)
def test_projection(projection, case, config_name):
    name, q, s, k = projection
    ids, counts = routes(case)
    ids = mx.array(np.sort(ids.ravel()))
    offsets = expert_offsets(ids, E)
    np.testing.assert_array_equal(np.array(offsets), np.r_[0, np.cumsum(counts)])
    mx.random.seed(141)
    x = mx.random.normal((ids.size, k)).astype(mx.bfloat16)
    if name == "down":
        # Real ClampedSwiGLU arithmetic, including FP32 values not representable
        # in BF16; the composed test supplies actual gate/up GEMM results.
        gate = (mx.random.normal(x.shape) * 4).astype(mx.bfloat16)
        x = ClampedSwiGLU(3.0)(x * 4, gate)
        assert x.dtype == mx.float32
        assert mx.any(x != x.astype(mx.bfloat16).astype(mx.float32)).item()
    dtype = mx.float32 if name == "down" else mx.bfloat16
    actual = (grouped_mxfp4(x, q, s, offsets, out_dtype=dtype) if config_name is None
              else _grouped_mxfp4(x, q, s, offsets, out_dtype=dtype, config=CONFIGS[config_name]))
    reference = baseline(x, q, s, ids)
    assert_bits(f"{config_name}/{name}/{case}", actual, reference)
    truth = oracle(x, q, s, counts)
    accept(f"{name}/{case}/K{k}", actual, reference, truth)


def test_rounded_down_experiment(projection):
    name, q, s, k = projection
    if name != "down":
        pytest.skip("down-only numerical experiment")
    ids, counts = routes("tails")
    ids = mx.array(ids)
    offsets = expert_offsets(ids, E)
    mx.random.seed(441)
    up = mx.random.normal((ids.size, k)).astype(mx.bfloat16)
    gate = mx.random.normal(up.shape).astype(mx.bfloat16)
    full = ClampedSwiGLU(1.0)(up, gate)
    rounded = full.astype(mx.bfloat16)
    actual = grouped_mxfp4(rounded, q, s, offsets, out_dtype=mx.float32)
    # gather_qmm's output follows its input dtype. Converting the explicitly
    # rounded values back to FP32 compares the same operands/output boundary.
    ref = baseline(rounded.astype(mx.float32), q, s, ids)
    truth = oracle(rounded, q, s, counts)
    accept("experimental-rounded-down/FP32-output", actual, ref, truth)
    print(json.dumps(dict(experimental_rounding_drift=metrics(
        actual, baseline(full, q, s, ids)))))


@pytest.mark.parametrize("tiles", [(16, 32, 32), (16, 64, 32), (32, 32, 32), (32, 64, 32), (32, 64, 64)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_tails_canaries_and_worklist(tiles, dtype):
    ids, counts = routes("tails", experts=12)
    offsets = mx.array(np.r_[0, np.cumsum(counts)], dtype=mx.int32)
    bm = tiles[0]
    work = prepare_worklist(ids.size, offsets, bm)
    expected = [(e, row) for e, count in enumerate(counts) for row in range(0, count, bm)]
    np.testing.assert_array_equal(np.array(work)[:len(expected)], expected)
    assert np.all(np.array(work)[len(expected):] == -1)
    # Unique exact power-of-two expert values and non-tile-aligned N/K.
    q = mx.full((12, 70, 12), 0x22222222, dtype=mx.uint32)  # q=1, K=96
    s = mx.broadcast_to(mx.arange(120, 132, dtype=mx.uint8)[:, None, None], (12, 70, 3))
    x = mx.ones((ids.size, 96), dtype=dtype)
    y = _grouped_mxfp4(x, q, s, offsets, out_dtype=dtype,
                       worklist=work, tiles=tiles, guard=1)
    host = np.array(y.astype(mx.float32))
    canary = mx.array(-12345, dtype=dtype).astype(mx.float32).item()
    assert np.all(host[[0, -1]] == canary)
    expected_y = np.broadcast_to((96 * np.exp2(ids.astype(np.int32) - 7))[:, None], (ids.size, 70))
    np.testing.assert_array_equal(host[1:-1], expected_y)


@pytest.mark.parametrize("config_name", list(CONFIGS))
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_tuning_tails_canaries(config_name, dtype):
    config = CONFIGS[config_name]
    ids, counts = routes("tails", experts=12)
    # Include full 128-row tiles and a 129-row task boundary.
    counts[9:12] = [127, 128, 129]
    ids = np.repeat(np.arange(12, dtype=np.uint32), counts)
    offsets = mx.array(np.r_[0, np.cumsum(counts)], dtype=mx.int32)
    work = prepare_worklist(ids.size, offsets, config.tiles[0])
    expected = [(e, row) for e, count in enumerate(counts)
                for row in range(0, int(count), config.tiles[0])]
    np.testing.assert_array_equal(np.array(work)[:len(expected)], expected)
    assert np.all(np.array(work)[len(expected):] == -1)
    q, s = bank(12, 70, 96, seed=945)
    mx.random.seed(946)
    x = mx.random.normal((ids.size, 96)).astype(dtype)
    y = _grouped_mxfp4(x, q, s, offsets, out_dtype=dtype,
                       worklist=work, config=config, guard=1)
    assert_bits(config_name + "/tails", y[1:-1], baseline(x, q, s, mx.array(ids)))
    canary = mx.full((2, 70), -12345, dtype=dtype)
    assert_bits(config_name + "/canary", mx.concatenate([y[:1], y[-1:]]), canary)


def check_probe(config_name, dtype, mode):
    """Worker-only small correctness gate; exactly one unsafe MODE per process."""
    def probe(x, q, s, offsets, **kwargs):
        assert kwargs.get("mode") == mode
        result, _ = evaluate(mx, f"{config_name}/{mode}/small-launch",
                             dict(X=x.shape, W=q.shape, scales=s.shape),
                             lambda: _grouped_mxfp4(x, q, s, offsets, **kwargs))
        return result

    config = CONFIGS[config_name]
    counts = np.array([0, 1, 15, 16, 17, 31, 32, 33, 65, 127, 128, 129])
    ids = np.repeat(np.arange(12, dtype=np.uint32), counts)
    offsets = mx.array(np.r_[0, np.cumsum(counts)], dtype=mx.int32)
    work = prepare_worklist(ids.size, offsets, config.tiles[0])
    expected = [(e, row) for e, count in enumerate(counts)
                for row in range(0, int(count), config.tiles[0])]
    q, s = bank(12, 70, 96, seed=945)
    mx.random.seed(946)
    x = mx.random.normal((ids.size, 96)).astype(dtype)
    if mode == "mma":
        ones = mx.ones_like(x)
        mma = probe(ones, q, s, offsets, out_dtype=dtype, config=config, mode="mma")
        mma_other = probe(ones, mx.zeros_like(q), mx.zeros_like(s), offsets,
                                  out_dtype=dtype, config=config, mode="mma")
        k_padded = ((96 + config.tiles[2] - 1) // config.tiles[2]) * config.tiles[2]
        assert_bits(config_name + "/mma-dummy", mma, mx.full(mma.shape, k_padded, dtype=dtype))
        assert_bits(config_name + "/mma-weight-independent", mma, mma_other)
        return
    checksum = probe(x, q, s, offsets, out_dtype=dtype,
                              worklist=work, config=config, mode="decode")
    # CPU oracle consumes the same staged values, with no second probe variant.
    decoded = mx.dequantize(q, s, group_size=32, bits=4, mode="mxfp4", dtype=dtype)
    host_x, host_w = np.array(x.astype(mx.float32)), np.array(decoded.astype(mx.float32))
    bm, bn, bk = config.tiles
    checksum_ref = np.zeros(checksum.shape, dtype=np.uint32)
    starts = np.r_[0, np.cumsum(counts)]
    readers = (np.arange(128) + 32) % 128
    for task, (expert, row) in enumerate(expected):
        for band in range((70 + bn - 1) // bn):
            h = np.zeros(128, dtype=np.uint32)
            for kb in range(0, 96, bk):
                xs, ws = np.zeros((bm, bk), np.float32), np.zeros((bn, bk), np.float32)
                nr, nc, nk = min(bm, counts[expert] - row), min(bn, 70 - band * bn), min(bk, 96 - kb)
                xs[:nr, :nk] = host_x[starts[expert] + row:starts[expert] + row + nr, kb:kb + nk]
                ws[:nc, :nk] = host_w[expert, band * bn:band * bn + nc, kb:kb + nk]
                for tile in (xs, ws):
                    bits = tile.ravel().view(np.uint32)
                    for base in range(0, bits.size, 128):
                        h = ((h << 5) | (h >> 27)) ^ bits[base + readers]
            checksum_ref[task, band] = h
    np.testing.assert_array_equal(np.array(checksum), checksum_ref)
    assert np.all(np.array(checksum)[len(expected):] == 0)
    # Change only the last scale group: the observable hash must depend on it.
    changed = mx.concatenate([mx.zeros_like(x[:, :-32]), mx.ones_like(x[:, -32:])], axis=1)
    probe0 = probe(mx.zeros_like(x), q, s, offsets, config=config, mode="decode")
    probe1 = probe(changed, q, s, offsets, config=config, mode="decode")
    assert mx.any(probe0 != probe1).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("vector", [False, True])
def test_all_decode_codes_against_mlx(dtype, vector):
    # Exercise exactly the helper compiled into the GEMM, without MMA changing
    # nonfinite/signed-zero behavior. All 16 nibbles x all 256 scale bytes.
    kernel = mx.fast.metal_kernel(
        name="dsv41_decode_audit_vector" if vector else "dsv41_decode_audit",
        input_names=["Q", "S"], output_names=["Y"],
        header=_tuning_source()[0] if vector else _source()[0], source="""
        alignas(16) threadgroup T scratch[128 * 8];
        uint i = thread_position_in_grid.x;
        uint t = thread_index_in_threadgroup;
        dsv41_decode8<T>(Q[i] * 0x11111111u, dsv41_scale<T>(S[i]), scratch + t * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Rotate readers to make every vector store observable.
        uint reader = (t + 1) % 128;
        uint out = (i / 128) * 128 + reader;
        for (uint j = 0; j < 8; ++j) Y[out * 8 + j] = scratch[reader * 8 + j];
        """ if vector else """
        uint i = thread_position_in_grid.x;
        Y[i] = dsv41_decode<T>(Q[i], dsv41_scale<T>(S[i]));
        """)
    codes = np.tile(np.arange(16, dtype=np.uint32), 256)
    scales = np.repeat(np.arange(256, dtype=np.uint8), 16)
    actual = kernel(inputs=[mx.array(codes), mx.array(scales)], template=[("T", dtype)],
                    grid=(4096, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(4096, 8) if vector else (4096,)], output_dtypes=[dtype])[0]
    packed = mx.array((codes * np.uint32(0x11111111))[:, None])
    packed = mx.broadcast_to(packed, (4096, 4))
    reference = mx.dequantize(packed, mx.array(scales[:, None]), group_size=32,
                              bits=4, mode="mxfp4", dtype=dtype)[:, 0]
    a, b = np.array(actual.astype(mx.float32)), np.array(reference.astype(mx.float32))
    if vector:
        b = np.broadcast_to(b[:, None], a.shape).copy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    good = ~np.isnan(b)
    np.testing.assert_array_equal(a[good].view(np.uint32), b[good].view(np.uint32))


def make_moe(experts, d, inter):
    # Allocate only small placeholder modules, then attach quantized banks.
    # This avoids SwitchGLU allocating all release FP32 expert parameters.
    mlp = SwitchGLU(32, 32, 1, activation=ClampedSwiGLU(1.0))
    for seed, (name, n, k) in enumerate((("gate_proj", inter, d), ("up_proj", inter, d),
                                        ("down_proj", d, inter)), 241):
        layer = QuantizedSwitchLinear(32, 32, 1, bias=False, group_size=32, bits=4, mode="mxfp4")
        layer.weight, layer.scales = bank(experts, n, k, seed)
        setattr(mlp, name, layer)
    mlp.eval()
    return SimpleNamespace(switch_mlp=mlp)


def composed_oracle(moe, x, indices, weights):
    order = mx.argsort(indices.flatten())
    inverse = mx.argsort(order)
    counts = np.bincount(np.array(indices).ravel(), minlength=E)
    sx = x[order // TOPK]
    mlp = moe.switch_mlp
    gate = oracle(sx, mlp.gate_proj.weight, mlp.gate_proj.scales, counts).astype(mx.bfloat16)
    up = oracle(sx, mlp.up_proj.weight, mlp.up_proj.scales, counts).astype(mx.bfloat16)
    down = oracle(mlp.activation(up, gate), mlp.down_proj.weight, mlp.down_proj.scales, counts)
    y = down[inverse].reshape(*indices.shape, x.shape[-1])
    return mx.sum(y * weights[..., None], axis=-2)


@pytest.mark.parametrize("compile_combine", ["0", "1"])
@pytest.mark.parametrize("case", ["skewed", "random8192"])
@pytest.mark.parametrize("config_name", TEST_CONFIGS)
def test_composed_moe(monkeypatch, compile_combine, case, config_name):
    monkeypatch.setenv("DSV41_COMPILE_PRIMITIVES", compile_combine)
    moe = make_moe(E, D, I)
    ids, _ = routes(case)
    indices = mx.array(ids)
    mx.random.seed(341)
    x = (mx.random.normal((ids.shape[0], D)) * 4).astype(mx.bfloat16)
    weights = mx.softmax(mx.random.normal(indices.shape), axis=-1) * 1.5
    actual = moe_prefill_grouped(moe, x, indices, weights,
                                 gemm_config=CONFIGS[config_name] if config_name else None)
    reference = mx.sum(moe.switch_mlp(x, indices).astype(mx.float32) * weights[..., None], axis=-2)
    truth = composed_oracle(moe, x, indices, weights)
    assert actual.dtype == mx.float32
    assert_bits(f"{config_name}/composed/{case}", actual, reference)
    accept(f"composed/{case}/FP32-down", actual, reference, truth)


def test_clamp_boundaries():
    v = mx.array([-4, -1.0078125, -1, -0.99609375, 0, 0.99609375, 1, 1.0078125, 4], mx.bfloat16)
    import mlx.nn as nn
    actual = ClampedSwiGLU(1.0)(v[:, None], v[None, :])
    expected = nn.silu(mx.minimum(v.astype(mx.float32)[None, :], 1.0)) * mx.clip(v.astype(mx.float32)[:, None], -1.0, 1.0)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_switch_off_threshold_and_decode(monkeypatch):
    # Exercise the real MoE hook without requiring checkpoint-sized weights.
    import mlx_vlm.models.deepseek_v41.moe_prefill as adapter
    calls = []
    def original(x, indices):
        calls.append("original")
        return mx.ones((*indices.shape, 32), dtype=mx.float32)
    def grouped(moe, x, indices, weights):
        calls.append("grouped")
        return mx.ones(x.shape, dtype=mx.float32)
    fake = SimpleNamespace(dim=32, switch_mlp=original,
        gate=lambda x, mask: (mx.ones((x.shape[0], 1)), mx.zeros((x.shape[0], 1), dtype=mx.uint32)),
        shared_experts=lambda x: mx.zeros_like(x))
    monkeypatch.setattr(adapter, "moe_prefill_grouped", grouped)
    monkeypatch.delenv("DSV41_PREFILL_GROUPED_MXFP4", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_GROUPED_MIN_ROWS", raising=False)
    MoE.__call__(fake, mx.zeros((1, 256, 32), dtype=mx.bfloat16))
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MXFP4", "1")
    for rows in (1, 255, 256):
        MoE.__call__(fake, mx.zeros((1, rows, 32), dtype=mx.bfloat16))
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MIN_ROWS", "1")
    MoE.__call__(fake, mx.zeros((1, 1, 32), dtype=mx.bfloat16))
    assert calls == ["original", "original", "original", "grouped", "original"]


def timing_worker(args):
    """One projection/config/MODE for this process; never reuse after a timeout."""
    job = Variant(*args.worker)
    shapes = shapes_for(job, args.tokens, args.release)
    timeout = min(args.launch_timeout, 120.0)
    label = job.label
    d, inter = (5120, 2304) if args.release else (128, 96)
    report(worker=label, shapes=shapes, mlx=mx.__version__, mlx_path=mx.__file__,
           platform=platform.platform(),
           mlx_commit=os.environ.get("DSV41_MLX_COMMIT", "unrecorded"))

    # This small check runs only after explicit --attribution-unsafe, and only in
    # the selected MODE. All its GPU work/readback shares a bounded host scope.
    if job.mode != "full":
        bounded_call(label + "/small-correctness", dict(X=[694, 96], W=[12, 70, 12]),
                     lambda: check_probe(job.config,
                         mx.float32 if job.projection == "down" else mx.bfloat16, job.mode), timeout)

    def setup():
        report(device=mx.metal.device_info())
        ids, counts = routes(("random8192" if args.tokens == 8192 else "random2048")
                             if args.case == "random" else args.case)
        indices = mx.array(ids)
        order = mx.argsort(indices.flatten())
        sorted_ids = indices.flatten()[order]
        offsets = expert_offsets(sorted_ids, E)
        config = CONFIGS.get(job.config)
        work = prepare_worklist(indices.size, offsets, 32 if config is None else config.tiles[0])
        mx.random.seed(1441)
        x = mx.random.normal((args.tokens, d)).astype(mx.bfloat16)
        sx = x[order // TOPK]
        weights = mx.softmax(mx.random.normal(indices.shape), axis=-1) * 1.5
        mx.eval(x, sx, weights, indices, offsets, work)
        if job.projection in ("down", "routed_layer"):
            moe = make_moe(E, d, inter)
            mlp = moe.switch_mlp
        if job.projection == "routed_layer":
            def reference_fn():
                return mx.sum(mlp(x, indices).astype(mx.float32) * weights[..., None], axis=-2)
            fn = reference_fn if job.config == "gather_qmm" else lambda: moe_prefill_grouped(
                moe, x, indices, weights, gemm_config=config)
            return fn, reference_fn, None
        if job.projection == "down":
            gate = baseline(sx, mlp.gate_proj.weight, mlp.gate_proj.scales, sorted_ids)
            up = baseline(sx, mlp.up_proj.weight, mlp.up_proj.scales, sorted_ids)
            operand = mlp.activation(up, gate)
            q, scales = mlp.down_proj.weight, mlp.down_proj.scales
            mx.eval(operand)
        else:
            operand = sx
            q, scales = bank(E, inter, d, seed=241 if job.projection == "gate" else 242)
        dtype = mx.float32 if job.projection == "down" else mx.bfloat16
        def reference_fn():
            return baseline(operand, q, scales, sorted_ids)
        fn = reference_fn if job.config == "gather_qmm" else lambda: _grouped_mxfp4(
            operand, q, scales, offsets, out_dtype=dtype, worklist=work,
            config=config, mode=job.mode, verbose=args.verbose)
        if config:
            report(occupancy=label, **launch_info(config, counts, q.shape[1], operand.shape[1],
                                                 4 if dtype == mx.float32 else 2))
        return fn, reference_fn, 2 * indices.size * q.shape[1] * operand.shape[1]

    (fn, reference_fn, useful_flops), _ = bounded_call(label + "/setup", shapes, setup, timeout)
    reference = None
    if job.mode == "full":
        reference, _ = evaluate(mx, label + "/reference", shapes, reference_fn, timeout)
    actual, cold_ms = evaluate(mx, label + "/first-evaluation", shapes, fn, timeout)
    report(first_evaluation=label, compile_plus_execution_ms=cold_ms)
    if reference is not None:
        bounded_call(label + "/bit-check", shapes,
                     lambda: assert_bits(label, actual, reference), timeout)
    for i in range(5):
        evaluate(mx, label + f"/warmup-{i}", shapes, fn, timeout)
    bounded_call(label + "/reset-peak-memory", shapes, mx.reset_peak_memory, timeout)
    samples = [evaluate(mx, label + f"/sample-{i}", shapes, fn, timeout)[1]
               for i in range(args.repetitions)]
    result = dict(zip(("p10_ms", "median_ms", "p90_ms"), np.percentile(samples, [10, 50, 90]).tolist()))
    if useful_flops is not None and job.mode == "full":
        result["useful_tflops"] = useful_flops / (result["median_ms"] * 1e9)
    peak, _ = bounded_call(label + "/peak-memory", shapes, mx.get_peak_memory, timeout)
    target = 45.0 if job.projection == "routed_layer" else (14.0 if job.projection != "down" else None)
    report(timing=label, first_timed_execution_ms=cold_ms, results=result, peak_bytes=peak,
           target_ms=target, target_met=job.mode == "full" and target is not None and result["median_ms"] <= target)
    if args.capture:
        def capture():
            mx.metal.start_capture(f"{args.capture}-{job.projection}-{job.config}-{job.mode}.gputrace")
            try:
                mx.eval(fn())
                mx.synchronize()
            finally:
                mx.metal.stop_capture()
        bounded_call(label + "/capture", shapes, capture, timeout)


if __name__ == "__main__":
    # os._exit avoids Python/MLX destructors joining a hung Metal command buffer.
    # The controller also has a process deadline if a native call holds the GIL.
    try:
        timing_worker(_cli_args)
    except LaunchHung:
        report(worker="/".join(_cli_args.worker), status="HUNG", exit_code=124)
        os._exit(124)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        report(worker="/".join(_cli_args.worker), status="FAILED", error=str(exc), exit_code=1)
        os._exit(1)
    report(worker="/".join(_cli_args.worker), status="OK", exit_code=0)
    os._exit(0)
