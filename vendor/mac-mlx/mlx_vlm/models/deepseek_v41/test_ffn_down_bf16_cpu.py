"""CPU dispatch checks and a BF16 rounding model; no Metal accuracy claims."""

import ast
from contextvars import ContextVar
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load():
    tree = ast.parse(Path(__file__).with_name("language.py").read_text())
    selected = []
    for node in tree.body:
        # attach validates DSV41_INDEX_SKIP_MIN_NB through this helper.
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_ffn_down_bf16", "_index_skip_min_nb"):
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name in ("MoE", "LanguageModel"):
            name = "__call__" if node.name == "MoE" else "attach_engram_storage"
            method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
            method.name = "forward" if node.name == "MoE" else "attach"
            selected.append(method)
    ns = dict(os=os, mx=np, _FFN_DOWN_BF16_ENGAGED=False,
              _v41_verify_context=ContextVar("verify", default=False), _HEAD_BF16=False)
    exec("from __future__ import annotations\n" + ast.unparse(ast.Module(selected, [])), ns)
    return ns


def _attach(ns, moe, monkeypatch, flag):
    monkeypatch.setenv("DSV41_FFN_EXPERT_PAD16", "0")
    monkeypatch.delenv("DSV41_PROMOTE_DECODE_WEIGHTS", raising=False)
    if flag is None:
        monkeypatch.delenv("DSV41_FFN_DOWN_BF16", raising=False)
    else:
        monkeypatch.setenv("DSV41_FFN_DOWN_BF16", flag)
    model = SimpleNamespace(layers=[SimpleNamespace(ffn=moe, engram=None)],
                            model=SimpleNamespace(layout=None))
    ns["attach"](model, ".")
    return model


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("n,verify", [(1, False), (63, False), (64, False),
                                       (2048, False), (64, True), (2048, True)])
def test_dispatch_and_attach_latch(monkeypatch, flag, n, verify):
    ns = _load()
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MXFP4", "0")
    monkeypatch.setenv("DSV41_COMPILE_PRIMITIVES", "0")
    calls = []
    routed = np.arange(n * 6 * 4, dtype=np.float32).reshape(n, 6, 4) / 100
    weights = np.full((n, 6), 0.25, np.float32)
    moe = SimpleNamespace(dim=4, ffn_down_bf16=False,
                          gate=lambda *a: (weights, np.zeros((n, 6), np.int32)),
                          switch_mlp=lambda *a: (calls.append("fp32") or routed),
                          shared_experts=lambda x: np.zeros_like(x))
    ns["_ffn_down_bf16"] = lambda *a: (calls.append("bf16") or routed)
    model = _attach(ns, moe, monkeypatch, flag)
    monkeypatch.setenv("DSV41_FFN_DOWN_BF16", "0" if flag == "1" else "1")
    ns["attach"](model, ".")  # Environment changes after attach have no effect.
    ns["_v41_verify_context"].set(verify)
    actual = ns["forward"](moe, np.ones((1, n, 4), np.float32))
    assert calls == ["bf16" if flag == "1" and n >= 64 and not verify else "fp32"]
    np.testing.assert_array_equal(actual[0], (routed * weights[..., None]).sum(axis=-2))
    assert actual.dtype == np.float32


def _bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def test_cpu_numerics_and_single_engagement(monkeypatch, capsys):
    ns = _load()
    bf16 = object()
    casts = []

    class Hidden(np.ndarray):
        def astype(self, dtype, *args, **kwargs):
            assert dtype is bf16
            casts.append(dtype)
            return _bf16(self)

    # Exercise the production helper with NumPy equivalents of MLX sort/unsort.
    switch = ModuleType("mlx_lm.models.switch_layers")

    def gather(x, indices):
        order = np.argsort(indices.flatten())
        return x.reshape(-1, 1, x.shape[-1])[order // 6], indices.flatten()[order], np.argsort(order)

    switch._gather_sort = gather
    switch._scatter_unsort = lambda x, inverse, shape: x[inverse].reshape(*shape, 1, x.shape[-1])
    monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch)
    ns["mx"] = SimpleNamespace(expand_dims=np.expand_dims, bfloat16=bf16)
    rng = np.random.default_rng(41)
    n, experts, dim, hidden = 64, 8, 32, 64
    x = _bf16(rng.normal(size=(n, dim)))
    indices = np.argsort(rng.random((n, experts)), axis=-1)[:, :6]
    # Random MXFP4 E2M1 values with power-of-two scales, at fan-in magnitudes.
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)

    def weight(shape, scale):
        return lut[rng.integers(0, 8, shape)] * rng.choice(np.array([-1, 1], np.float32), shape) * scale

    wg, wu = weight((experts, hidden, dim), 1 / 16), weight((experts, hidden, dim), 1 / 16)
    wd = weight((experts, dim, hidden), 1 / 32)

    def project(value, w, ids):
        return np.einsum("bi,boi->bo", value.reshape(len(ids), -1), w[ids])[:, None, :]

    def activation(up, gate):
        gate = np.minimum(gate, 10)
        return (gate / (1 + np.exp(-gate)) * np.clip(up, -10, 10)).view(Hidden)

    mlp = SimpleNamespace(training=False,
        up_proj=lambda v, ids, **kw: _bf16(project(v, wu, ids)),
        gate_proj=lambda v, ids, **kw: _bf16(project(v, wg, ids)),
        activation=activation,
        down_proj=lambda v, ids, **kw: _bf16(project(v, _bf16(wd), ids)))
    sorted_x, ids, inverse = gather(np.expand_dims(x, (-2, -3)), indices)
    h = activation(mlp.up_proj(sorted_x, ids), mlp.gate_proj(sorted_x, ids))
    ref = switch._scatter_unsort(project(h, wd, ids), inverse, indices.shape).squeeze(-2)
    got = ns["_ffn_down_bf16"](mlp, x, indices)
    ns["_ffn_down_bf16"](mlp, x, indices)
    assert len(casts) == 2
    assert capsys.readouterr().out == "[DSV41] FFN_DOWN_BF16 engaged n=64\n"
    routes = rng.uniform(.05, .4, (n, 6)).astype(np.float32)
    routes *= 1.5 / routes.sum(axis=-1, keepdims=True)
    ref, got = ((y.astype(np.float32) * routes[..., None]).sum(axis=-2) for y in (ref, got))
    error = got - ref
    max_abs = float(np.max(np.abs(error)))
    rms_rel = float(np.sqrt(np.mean(error ** 2) / np.mean(ref ** 2)))
    print(f"CPU BF16 rounding model: max_abs={max_abs:.8g} rms_rel={rms_rel:.8g}")
    assert np.isfinite(got).all() and 0 < rms_rel < .02
