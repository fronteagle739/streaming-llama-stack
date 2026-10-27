"""Run this file directly on Windows: no MLX imports, GPU calls or git commands."""
import ast
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("expert_tiles_cpu_helpers", HERE / "test_expert_tiles_cpu.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


@pytest.mark.parametrize("case", ["probe_a", "balanced32", "8x1536", "singleton_empty", "fused_boundaries", "empty"])
def test_m32_metadata_and_tile_coverage(case):
    ids = np.sort(base.fixtures.routes(case).ravel())
    offsets, prefix = base.fixtures.metadata_numpy(ids, 32)
    counts = np.bincount(ids, minlength=384)
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(counts)])
    np.testing.assert_array_equal(prefix, np.r_[0, np.cumsum((counts + 31) // 32)])
    candidates = (ids.size + 31) // 32 + 384
    assert prefix[-1] < candidates
    coverage = np.zeros(ids.size, np.int32)
    for q in range(candidates):
        if q >= prefix[-1]:
            continue
        expert = np.searchsorted(prefix, q, side="right") - 1
        row = int(offsets[expert]) + 32 * (q - int(prefix[expert]))
        m = min(32, int(offsets[expert + 1]) - row)
        assert 1 <= m <= 32 and np.all(ids[row:row + m] == expert)
        for row_half in (0, 16):
            if row_half < m:
                coverage[row + row_half:row + min(m, row_half + 16)] += 1
    np.testing.assert_array_equal(coverage, np.ones(ids.size))
    if case in ("balanced32", "8x1536"):
        assert prefix[-1] == 384


def test_metadata_wrapper_launches_actual_m32_template():
    tree = ast.parse((HERE / "expert_tiles_qmm.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ("RoutingPlan", "expert_tiles_metadata")]
    calls = []

    def launch(**kw):
        calls.append(kw)
        return base.fixtures.metadata_numpy(kw["inputs"][0], dict(kw["template"])["BM"])

    ns = dict(dataclass=dataclass, EXPERTS=384, mx=SimpleNamespace(array=np.ndarray, uint32=np.uint32),
              _kernels=lambda: (launch,))
    exec(compile(ast.Module(nodes, []), "metadata_cpu", "exec"), ns)
    ids = np.sort(base.fixtures.routes("fused_boundaries"))
    plan = ns["expert_tiles_metadata"](ids, tile_rows=32)
    assert calls[0]["template"] == [("BM", 32)]
    assert calls[0]["grid"] == calls[0]["threadgroup"] == (512, 1, 1)
    assert plan.candidates == (ids.size + 31) // 32 + 384 and plan.tile_rows == 32
    assert ns["expert_tiles_metadata"](ids).tile_rows == 16
    with pytest.raises(ValueError, match="16 or 32"):
        ns["expert_tiles_metadata"](ids, tile_rows=64)
    metal = (HERE / "expert_tiles_qmm.metal").read_text(encoding="utf-8")
    assert "(bounds[lane + 1] - bounds[lane] + BM - 1) / BM" in metal


@pytest.fixture
def dispatch(monkeypatch):
    ns, calls, moe, model, dspark = base.dispatch.__wrapped__(monkeypatch)
    moe.ffn_fused = dspark.ffn_fused = False
    ns["prepare_fused_ffn"] = lambda layers: calls.append("compile-fused")
    ns["moe_prefill_fused_ffn"] = lambda mlp, x, ids: (
        calls.append("fused") or np.full((len(x), 6, 4), .75, np.float32))
    return ns, calls, moe, model, dspark


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("n,verify", [(1, False), (63, False), (64, False), (2048, False), (64, True)])
def test_latch_prefill_and_dspark(dispatch, monkeypatch, flag, n, verify):
    ns, calls, moe, model, dspark = dispatch
    if flag is None:
        monkeypatch.delenv("DSV41_FFN_FUSED", raising=False)
    else:
        monkeypatch.setenv("DSV41_FFN_FUSED", flag)
    ns["attach"](model, ".")
    assert calls == (["compile-fused"] if flag == "1" else [])
    assert model._ffn_fused_attached == moe.ffn_fused == (flag == "1")
    assert not dspark.ffn_fused
    # Later environment changes cannot change a published latch.
    monkeypatch.setenv("DSV41_FFN_FUSED", "0" if flag == "1" else "1")
    ns["attach"](model, ".")
    calls.clear()
    token = ns["_v41_verify_context"].set(verify)
    try:
        actual = ns["forward"](moe, np.ones((n, 4), np.float32))
    finally:
        ns["_v41_verify_context"].reset(token)
    enabled = flag == "1" and n >= 64 and not verify
    assert calls == (["fused"] if enabled else ["stock"])
    np.testing.assert_allclose(actual, .75 if enabled else .5, rtol=1e-6)


@pytest.mark.parametrize("other", ["DSV41_FFN_DOWN_BF16", "DSV41_FFN_EXPERT_PAD16",
                                    "DSV41_FFN_EXPERT_TILES", "DSV41_PREFILL_GROUPED_MXFP4"])
def test_conflicts_fail_before_compile(dispatch, monkeypatch, other):
    ns, calls, _, model, _ = dispatch
    monkeypatch.setenv("DSV41_FFN_FUSED", "1")
    monkeypatch.setenv(other, "1")
    with pytest.raises(ValueError, match="FFN_FUSED requires"):
        ns["attach"](model, ".")
    assert not calls and not hasattr(model, "_ffn_fused_attached")


def test_invalid_flag_and_compile_failure_retry(dispatch, monkeypatch):
    ns, calls, moe, model, _ = dispatch
    monkeypatch.setenv("DSV41_FFN_FUSED", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ns["attach"](model, ".")
    monkeypatch.setenv("DSV41_FFN_FUSED", "1")

    def fail(layers):
        raise RuntimeError("shader failed")

    ns["prepare_fused_ffn"] = fail
    with pytest.raises(RuntimeError, match="shader failed"):
        ns["attach"](model, ".")
    assert not hasattr(model, "_ffn_fused_attached") and not moe.ffn_fused
    ns["prepare_fused_ffn"] = lambda layers: calls.append("retry")
    ns["attach"](model, ".")
    assert calls == ["retry"] and moe.ffn_fused


def test_adapter_plans_unsort_and_engage_once(monkeypatch, capsys):
    switch = ModuleType("mlx_lm.models.switch_layers")

    def gather(x, indices):
        order = np.argsort(indices.flatten())
        return x.reshape(-1, 1, x.shape[-1])[order // 6], indices.flatten()[order], np.argsort(order)

    switch._gather_sort = gather
    switch._scatter_unsort = lambda x, inverse, shape: x[inverse].reshape(*shape, 1, x.shape[-1])
    monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch)
    tree = ast.parse((HERE / "moe_prefill_fused_ffn.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "moe_prefill_fused_ffn")
    calls = []

    def metadata(ids, experts, bm):
        assert experts == 384
        calls.append(bm)
        return SimpleNamespace(ids=ids, tile_rows=bm)

    def gate_up(x, gw, gs, uw, us, plan, limit):
        assert plan.tile_rows == 32 and limit == 10.
        calls.append("fused")
        return (x * gw[plan.ids, None, :] + gs) * (x * uw[plan.ids, None, :] + us)

    def down(x, w, s, plan):
        assert plan.tile_rows == 16
        calls.append("down")
        return x * w[plan.ids, None, :] + s

    ns = dict(mx=SimpleNamespace(bfloat16=np.float32, uint32=np.uint32, expand_dims=np.expand_dims),
              _ENGAGED=False, fused_ffn_metadata=lambda ids, e: metadata(ids, e, 32),
              expert_tiles_metadata=lambda ids, e: metadata(ids, e, 16),
              expert_gate_up_swiglu=gate_up, expert_tiles_qmm=down)
    exec(compile(ast.Module([function], []), "adapter_cpu", "exec"), ns)
    bank = np.arange(384, dtype=np.float32)[:, None] / 384 + .5
    layer = SimpleNamespace(weight=bank, scales=np.float32(.25))
    mlp = SimpleNamespace(gate_proj=layer, up_proj=layer, down_proj=layer,
                          activation=SimpleNamespace(limit=10.), training=False)
    x = np.arange(64 * 4, dtype=np.float32).reshape(64, 4) / 256
    ids = (np.arange(64 * 6, dtype=np.uint32).reshape(64, 6) * 7) % 384
    actual = ns["moe_prefill_fused_ffn"](mlp, x, ids)
    projected = x[:, None, :] * bank[ids] + .25
    np.testing.assert_array_equal(actual, (projected * projected) * bank[ids] + .25)
    assert calls == [32, 16, "fused", "down"]
    ns["moe_prefill_fused_ffn"](mlp, x, ids)
    assert capsys.readouterr().out == "[DSV41] FFN_FUSED engaged n=64\n"


def test_no_runtime_readbacks():
    for name in ("fused_ffn_qmm.py", "moe_prefill_fused_ffn.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        for node in tree.body:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    assert call.func.attr not in ("item", "tolist", "eval")
                    if call.func.attr in ("async_eval", "synchronize"):
                        assert isinstance(node, ast.FunctionDef) and node.name == "prewarm_fused_ffn"


def test_projection_wrapper_grid_dtype_and_plan_validation():
    tree = ast.parse((HERE / "fused_ffn_qmm.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expert_gate_up_swiglu")
    calls = []

    def launch(**kw):
        calls.append(kw)
        return [np.zeros(kw["output_shapes"][0], kw["output_dtypes"][0])]

    ns = dict(mx=SimpleNamespace(bfloat16="bf16", uint32=np.uint32, float32=np.float32,
                                array=np.array, zeros=np.zeros), math=math, EXPERTS=384,
              validate_weight=lambda w, s: w, _kernel=lambda: launch)
    exec(compile(ast.Module([function], []), "projection_cpu", "exec"), ns)
    plan = SimpleNamespace(rows=33, experts=384, tile_rows=32, candidates=386,
                           offsets=np.zeros(385, np.uint32), prefix=np.zeros(385, np.uint32))
    x = SimpleNamespace(dtype="bf16", shape=(33, 1, 64))
    result = ns["expert_gate_up_swiglu"](x, (64, 96), None, (64, 96), None, plan, 10.)
    assert result.shape == (33, 1, 96) and result.dtype == np.float32
    assert calls[0]["grid"] == (384, 386, 1) and calls[0]["threadgroup"] == (128, 1, 1)
    assert calls[0]["template"] == [("T", "bf16")]
    assert calls[0]["inputs"][-1].dtype == np.float32
    plan.tile_rows = 16
    with pytest.raises(ValueError, match="M32"):
        ns["expert_gate_up_swiglu"](x, (64, 96), None, (64, 96), None, plan, 10.)


def test_prepare_wraps_compile_failure_and_rejects_activation():
    tree = ast.parse((HERE / "moe_prefill_fused_ffn.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_fused_ffn")
    function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]

    class Activation:
        limit = 10.

    def fail():
        raise RuntimeError("shader failed")

    ns = dict(ClampedSwiGLU=Activation, math=math, validate_mlp=lambda mlp: None,
              prewarm_fused_ffn=fail)
    exec(compile(ast.Module([function], []), "prepare_cpu", "exec"), ns)
    mlp = SimpleNamespace(activation=Activation())
    layers = [SimpleNamespace(ffn=SimpleNamespace(switch_mlp=mlp))]
    with pytest.raises(RuntimeError, match="attach failed") as error:
        ns["prepare_fused_ffn"](layers)
    assert str(error.value.__cause__) == "shader failed"
    mlp.activation = SimpleNamespace(limit=10.)
    with pytest.raises(RuntimeError, match="attach failed") as error:
        ns["prepare_fused_ffn"](layers)
    assert isinstance(error.value.__cause__, ValueError)


if __name__ == "__main__":
    class StandaloneModule(pytest.Module):
        def _getobj(self):
            return sys.modules[__name__]

    class Collector:
        def pytest_pycollect_makemodule(self, module_path, parent):
            return StandaloneModule.from_parent(parent.session, path=module_path)

    raise SystemExit(pytest.main(["--noconftest", "-q", str(Path(__file__).resolve()),
                                 *sys.argv[1:]], plugins=[Collector()]))
