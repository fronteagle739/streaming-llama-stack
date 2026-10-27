"""Windows CPU metadata, provenance and attach/dispatch tests; never import MLX."""
import ast
from contextvars import ContextVar
import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parent
LANGUAGE = HERE.parents[1] / "models" / "deepseek_v41" / "language.py"


def _file_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _file_module("expert_tiles_fixtures")
CASES = ["probe_a", "balanced32", "8x1536", "singleton_empty", "boundaries", "empty"]
CASES += [f"remainder_{r}" for r in range(16)]


@pytest.mark.parametrize("case", CASES)
def test_metadata_and_candidate_coverage(case):
    ids = np.sort(fixtures.routes(case).ravel())
    offsets, prefix = fixtures.metadata_numpy(ids)
    counts = np.bincount(ids, minlength=384)
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(counts)])
    np.testing.assert_array_equal(prefix, np.r_[0, np.cumsum((counts + 15) // 16)])
    assert offsets[-1] == ids.size and prefix.shape == offsets.shape == (385,)
    candidates = (ids.size + 15) // 16 + 384
    assert prefix[-1] < candidates
    coverage = np.zeros(ids.size, np.int32)
    for q in range(candidates):
        if q >= prefix[-1]:
            continue
        expert = np.searchsorted(prefix, q, side="right") - 1
        row = int(offsets[expert]) + 16 * (q - int(prefix[expert]))
        m = min(16, int(offsets[expert + 1]) - row)
        assert 1 <= m <= 16 and np.all(ids[row:row + m] == expert)
        coverage[row:row + m] += 1
    np.testing.assert_array_equal(coverage, np.ones(ids.size))
    if case == "probe_a":
        old = sum((int(offsets[e + 1]) - 1) // 16 - int(offsets[e]) // 16 + 1
                  for e in range(384) if counts[e])
        assert (prefix[-1], old, candidates) == (989, 1120, 1152)
    if case in ("balanced32", "8x1536"):
        assert prefix[-1] == 768


def _language_methods():
    tree = ast.parse(LANGUAGE.read_text(encoding="utf-8"))
    # attach validates DSV41_INDEX_SKIP_MIN_NB through this module-level helper,
    # so it has to come along or attach cannot be executed in isolation.
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_index_skip_min_nb"]
    assert len(selected) == 1
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in ("MoE", "LanguageModel"):
            continue
        method = "__call__" if cls.name == "MoE" else "attach_engram_storage"
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
        node.name = "forward" if cls.name == "MoE" else "attach"
        selected.append(node)

    class InjectHelpers(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module.endswith(("moe_prefill_expert_tiles", "moe_prefill_fused_ffn")):
                return ast.Pass()
            return node

    module = ast.fix_missing_locations(InjectHelpers().visit(ast.Module(selected, [])))
    ns = dict(os=os, mx=np, _HEAD_BF16=False,
              _v41_verify_context=ContextVar("expert_tiles_verify", default=False))
    exec("from __future__ import annotations\n" + ast.unparse(module), ns)
    return ns


@pytest.fixture
def dispatch(monkeypatch):
    for name in ("DSV41_FFN_FUSED", "DSV41_FFN_EXPERT_TILES", "DSV41_FFN_DOWN_BF16", "DSV41_FFN_EXPERT_PAD16",
                 "DSV41_PREFILL_GROUPED_MXFP4", "DSV41_COMPILE_PRIMITIVES",
                 "DSV41_PROMOTE_DECODE_WEIGHTS"):
        monkeypatch.setenv(name, "0")
    ns = _language_methods()
    calls = []
    ns["prepare_expert_tiles"] = lambda layers: calls.append("compile")
    ns["moe_prefill_expert_tiles"] = lambda mlp, x, ids: (
        calls.append("custom") or np.full((len(x), 6, 4), 0.75, np.float32))
    moe = SimpleNamespace(dim=4, ffn_down_bf16=False, ffn_expert_pad16=False,
        ffn_expert_tiles=False, shared_experts=lambda x: np.zeros_like(x),
        gate=lambda x, mask: (np.full((len(x), 6), 1 / 6, np.float32),
                              np.zeros((len(x), 6), np.uint32)),
        switch_mlp=lambda x, ids: (calls.append("stock") or np.full((len(x), 6, 4), 0.5, np.float32)))
    dspark = SimpleNamespace(ffn_expert_tiles=False)
    model = SimpleNamespace(layers=[SimpleNamespace(ffn=moe, engram=None)],
                            dspark_stages=[dspark], model=SimpleNamespace(layout=None))
    return ns, calls, moe, model, dspark


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("n,verify", [(1, False), (63, False), (64, False),
                                       (2048, False), (64, True), (2048, True)])
def test_attach_latch_dispatch(dispatch, monkeypatch, flag, n, verify):
    ns, calls, moe, model, dspark = dispatch
    if flag is None:
        monkeypatch.delenv("DSV41_FFN_EXPERT_TILES")
    else:
        monkeypatch.setenv("DSV41_FFN_EXPERT_TILES", flag)
    ns["attach"](model, ".")
    monkeypatch.setenv("DSV41_FFN_EXPERT_TILES", "0" if flag == "1" else "1")
    ns["attach"](model, ".")
    ns["_v41_verify_context"].set(verify)
    result = ns["forward"](moe, np.ones((1, n, 4), np.float32))
    custom = flag == "1" and n >= 64 and not verify
    assert calls == (["compile"] if flag == "1" else []) + ["custom" if custom else "stock"]
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.75 if custom else 0.5)
    assert dspark.ffn_expert_tiles is False


def test_attach_compile_failure_is_fatal_and_retryable(dispatch, monkeypatch):
    ns, calls, moe, model, _ = dispatch
    monkeypatch.setenv("DSV41_FFN_EXPERT_TILES", "1")
    def fail(layers):
        raise RuntimeError("Metal compiler rejected shader")
    ns["prepare_expert_tiles"] = fail
    with pytest.raises(RuntimeError, match="Metal compiler"):
        ns["attach"](model, ".")
    assert not hasattr(model, "_ffn_expert_tiles_attached") and not moe.ffn_expert_tiles
    ns["prepare_expert_tiles"] = lambda layers: calls.append("retry")
    ns["attach"](model, ".")
    assert calls == ["retry"] and moe.ffn_expert_tiles


@pytest.mark.parametrize("conflict", ["DSV41_FFN_DOWN_BF16", "DSV41_FFN_EXPERT_PAD16",
                                     "DSV41_PREFILL_GROUPED_MXFP4"])
def test_attach_rejects_conflicting_experiments(dispatch, monkeypatch, conflict):
    ns, calls, _, model, _ = dispatch
    monkeypatch.setenv("DSV41_FFN_EXPERT_TILES", "1")
    monkeypatch.setenv(conflict, "1")
    with pytest.raises(ValueError, match="requires DOWN_BF16"):
        ns["attach"](model, ".")
    assert not calls and not hasattr(model, "_ffn_expert_tiles_attached")


def test_attach_rejects_unknown_flag(dispatch, monkeypatch):
    ns, _, _, model, _ = dispatch
    monkeypatch.setenv("DSV41_FFN_EXPERT_TILES", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ns["attach"](model, ".")


def test_no_host_reads_and_sync_only_at_attach():
    for name in ("expert_tiles_qmm.py", "moe_prefill_expert_tiles.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        for node in tree.body:
            for call in ast.walk(node):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                assert call.func.attr not in ("item", "tolist", "eval")
                if call.func.attr in ("async_eval", "synchronize"):
                    assert isinstance(node, ast.FunctionDef) and node.name == "prewarm_expert_tiles"


def test_adapter_reuses_one_plan_and_unsorts(monkeypatch, capsys):
    import sys
    switch = ModuleType("mlx_lm.models.switch_layers")

    def gather(x, indices):
        order = np.argsort(indices.flatten())
        return x.reshape(-1, 1, x.shape[-1])[order // indices.shape[-1]], indices.flatten()[order], np.argsort(order)

    switch._gather_sort = gather
    switch._scatter_unsort = lambda x, inverse, shape: x[inverse].reshape(*shape, 1, x.shape[-1])
    monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch)
    tree = ast.parse((HERE / "moe_prefill_expert_tiles.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "moe_prefill_expert_tiles")
    calls = []

    def metadata(ids, experts):
        assert experts == 384
        plan = SimpleNamespace(ids=ids)
        calls.append(("metadata", plan))
        return plan

    def project(value, weight, scales, plan):
        calls.append(("qmm", plan))
        return value * weight[plan.ids, None, :] + scales

    ns = dict(mx=SimpleNamespace(bfloat16=np.float32, float32=np.float32,
                                 uint32=np.uint32, expand_dims=np.expand_dims),
              _ENGAGED=False, expert_tiles_metadata=metadata, expert_tiles_qmm=project)
    exec(compile(ast.Module([function], []), "adapter_cpu", "exec"), ns)
    bank = np.arange(384, dtype=np.float32)[:, None] / 384 + 0.5
    layer = SimpleNamespace(weight=bank, scales=np.float32(0.25))
    activation = lambda up, gate: up * gate
    mlp = SimpleNamespace(up_proj=layer, gate_proj=layer, down_proj=layer,
                          training=False, activation=activation)
    x = np.arange(64 * 4, dtype=np.float32).reshape(64, 4) / 256
    ids = (np.arange(64 * 6, dtype=np.uint32).reshape(64, 6) * 7) % 384
    actual = ns["moe_prefill_expert_tiles"](mlp, x, ids)
    assert [name for name, _ in calls] == ["metadata", "qmm", "qmm", "qmm"]
    assert all(plan is calls[0][1] for _, plan in calls)
    up = x[:, None, :] * bank[ids] + .25
    reference = (up * up) * bank[ids] + .25
    np.testing.assert_array_equal(actual, reference)
    ns["moe_prefill_expert_tiles"](mlp, x, ids)
    assert capsys.readouterr().out == "[DSV41] FFN_EXPERT_TILES engaged n=64\n"


def test_prepare_wraps_compile_failure():
    tree = ast.parse((HERE / "moe_prefill_expert_tiles.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_expert_tiles")
    def fail():
        raise RuntimeError("shader failed")
    ns = dict(validate_mlp=lambda mlp: None, prewarm_expert_tiles=fail)
    exec(compile(ast.Module([node], []), "prepare_cpu", "exec"), ns)
    with pytest.raises(RuntimeError, match="attach failed") as error:
        ns["prepare_expert_tiles"]([])
    assert str(error.value.__cause__) == "shader failed"


def test_vendored_source_matches_pinned_objects():
    upstream = Path("./GitHub/mlx")
    if not (upstream / ".git").exists():
        pytest.skip("Pinned local MLX object store is not present")
    vendor = _file_module("expert_tiles_vendor")
    header, hashes = vendor.bundle(vendor.TagSource(upstream))
    metal = (HERE / "expert_tiles_qmm.metal").read_text(encoding="utf-8")
    assert header == metal.split("// --- metadata body ---", 1)[0]
    tree = ast.parse((HERE / "expert_tiles_qmm.py").read_text(encoding="utf-8"))
    manifest = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SOURCE_HASHES" for t in n.targets))
    assert hashes == ast.literal_eval(manifest)
    assert '#include "' not in header and "[[kernel]]" not in header


if __name__ == "__main__":
    # Run this file directly on Windows: package-based pytest discovery imports
    # mlx_vlm/__init__.py (and MLX) before it reaches this CPU-only module.
    import sys

    class StandaloneModule(pytest.Module):
        def _getobj(self):
            return sys.modules[__name__]

    class Collector:
        def pytest_pycollect_makemodule(self, module_path, parent):
            return StandaloneModule.from_parent(parent.session, path=module_path)

    raise SystemExit(pytest.main(["--noconftest", "-q", str(Path(__file__).resolve()),
                                 *sys.argv[1:]], plugins=[Collector()]))
