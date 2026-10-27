"""NumPy layout specification; no MLX import or Metal numerical claims."""

import numpy as np
import pytest


def _pad_layout(indices, experts):
    """Reference for mx scatter-add, cumsum, searchsorted and gather operations.

    The conversion of ends[-1] to a shape is cheap in NumPy, but would require
    a GPU-to-host read in MLX. This is not a sync-free production implementation.
    """
    idx = np.sort(indices.reshape(-1))
    counts = np.zeros(experts, dtype=np.int32)
    np.add.at(counts, idx, 1)
    lengths = ((counts + 15) // 16) * 16
    starts = np.cumsum(counts, dtype=np.int32) - counts
    ends = np.cumsum(lengths, dtype=np.int32)
    padded_starts = ends - lengths
    positions = np.arange(int(ends[-1]), dtype=np.int32)
    padded_ids = np.searchsorted(ends, positions, side="right")
    local = positions - padded_starts[padded_ids]
    # Sentinel R addresses the appended zero row (equivalent to -1 in X_ext).
    gather = np.where(local < counts[padded_ids],
                      starts[padded_ids] + local, idx.size)
    discard = padded_starts[idx] + np.arange(idx.size) - starts[idx]
    return idx, counts, lengths, padded_ids, gather, discard


def _check_layout(indices, experts):
    idx, counts, lengths, padded_ids, gather, discard = _pad_layout(indices, experts)
    rows = indices.size
    np.testing.assert_array_equal(counts, np.bincount(indices.ravel(), minlength=experts))
    assert gather.size - rows == int(np.sum((16 - counts % 16) % 16))
    assert gather.size % 16 == 0
    assert np.all(np.diff(padded_ids) >= 0)
    tiles = padded_ids.reshape(-1, 16)
    assert np.all(tiles == tiles[:, :1])
    np.testing.assert_array_equal(np.bincount(padded_ids, minlength=experts), lengths)
    np.testing.assert_array_equal(gather[discard], np.arange(rows))
    np.testing.assert_array_equal(padded_ids[discard], idx)
    # Tagged rows expose duplication, loss, sentinel leakage and inverse errors.
    x = np.arange(1, rows * 3 + 1, dtype=np.float32).reshape(rows, 1, 3)
    order = np.argsort(indices.ravel())
    sx = x[order]
    padded = np.concatenate((sx, np.zeros((1, 1, 3), np.float32)))[gather]
    assert np.all(padded[gather == rows] == 0)
    # A row-local expert-specific map must commute with pad/discard/unsort.
    padded_y = padded * (padded_ids[:, None, None] + 1)
    actual = padded_y[discard][np.argsort(order)]
    reference = x * (indices.ravel()[:, None, None] + 1)
    np.testing.assert_array_equal(actual, reference)
    return counts, lengths


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 31, 32, 33])
def test_boundary_counts_and_empty_experts(count):
    _check_layout(np.full((count, 1), 3, np.int32), 8)


def test_mixed_boundary_counts():
    counts = np.array([0, 1, 15, 16, 17, 31, 32, 33, 0], np.int32)
    ids = np.repeat(np.arange(counts.size), counts)
    np.random.default_rng(41).shuffle(ids)
    _check_layout(ids, counts.size)


@pytest.mark.parametrize("seed", range(12))
def test_random_routings(seed):
    rng = np.random.default_rng(seed)
    experts = 384
    # Zipf with replacement exercises skew and unoccupied experts in the layout.
    probabilities = np.arange(1, experts + 1, dtype=np.float64) ** -1.1
    probabilities /= probabilities.sum()
    ids = rng.choice(experts, (2048, 6), p=probabilities)
    _check_layout(ids, experts)
    # Distinct experts per token matches top-k routing's additional constraint.
    scores = rng.gumbel(size=(64, experts)) + np.log(probabilities)
    ids = np.argpartition(-scores, 5, axis=-1)[:, :6]
    _check_layout(ids, experts)


def test_fixed_capacity_cost():
    # A no-sync shape must cover this valid top-6 routing, too. All 384 counts
    # are 1 modulo 16, making all experts require 15 additional rows.
    ids = np.tile(np.arange(6), (2048, 1))
    ids[:63] = np.arange(6, 384).reshape(63, 6)
    counts, lengths = _check_layout(ids, 384)
    assert np.all(counts % 16 == 1)
    assert lengths.sum() == 18048
    assert lengths.sum() / ids.size == 1.46875


def test_host_read_once_per_forward(monkeypatch, capsys):
    """Run production attach/dispatch/layout with mocked MLX; count every item()."""
    import ast
    from contextvars import ContextVar
    import os
    from pathlib import Path
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    host_read = Mock(side_effect=lambda value: np.ndarray.item(value))

    class Array(np.ndarray):
        def __getitem__(self, key):
            return np.asarray(super().__getitem__(key)).view(Array)

        def item(self):
            return host_read(self)

        @property
        def at(self):
            value = self

            class At:
                def __getitem__(self, idx):
                    def add(amount):
                        result = value.copy()
                        np.add.at(result, idx, amount)
                        return result
                    return SimpleNamespace(add=add)

            return At()

    tree = ast.parse(Path(__file__).with_name("language.py").read_text())
    selected = []
    for node in tree.body:
        # attach validates DSV41_INDEX_SKIP_MIN_NB through this helper.
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_ffn_expert_pad16_layout", "_ffn_expert_pad16", "_index_skip_min_nb"):
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name in ("MoE", "LanguageModel"):
            name = "__call__" if node.name == "MoE" else "attach_engram_storage"
            method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
            method.name = "forward" if node.name == "MoE" else "attach"
            selected.append(method)
    mock_mx = SimpleNamespace(
        zeros=lambda *a, **kw: np.zeros(*a, **kw).view(Array), int32=np.int32,
        float32=np.float32, cumsum=np.cumsum, arange=np.arange,
        searchsorted=np.searchsorted, where=np.where, sum=np.sum,
        concatenate=np.concatenate, expand_dims=np.expand_dims)
    ns = dict(mx=mock_mx, os=os, _HEAD_BF16=False,
              _FFN_EXPERT_PAD16_ENGAGED=False, _FFN_DOWN_BF16_ENGAGED=False,
              _v41_verify_context=ContextVar("pad_verify", default=False))
    exec("from __future__ import annotations\n" + ast.unparse(ast.Module(selected, [])), ns)
    layout = ns["_ffn_expert_pad16_layout"] = Mock(wraps=ns["_ffn_expert_pad16_layout"])
    switch = ModuleType("mlx_lm.models.switch_layers")

    def gather(x, indices):
        order = np.argsort(indices.ravel())
        return (x.reshape(-1, 1, x.shape[-1])[order // indices.shape[-1]],
                indices.ravel()[order], np.argsort(order))

    switch._gather_sort = gather
    switch._scatter_unsort = lambda y, inv, shape: y[inv].reshape(*shape, 1, y.shape[-1])
    monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch)
    monkeypatch.setenv("DSV41_FFN_EXPERT_PAD16", "1")
    monkeypatch.setenv("DSV41_FFN_DOWN_BF16", "0")
    monkeypatch.setenv("DSV41_PROMOTE_DECODE_WEIGHTS", "0")
    # Padding must also take precedence over grouped prefill.
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MXFP4", "1")
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MIN_ROWS", "64")
    monkeypatch.setenv("DSV41_COMPILE_PRIMITIVES", "0")
    up = Mock(side_effect=lambda x, ids, **kw: x * (ids[:, None, None] + 1))
    up.num_experts = 8
    mlp = SimpleNamespace(training=False, up_proj=up,
                          gate_proj=lambda x, ids, **kw: np.ones_like(x),
                          activation=lambda up, gate: up * gate,
                          down_proj=lambda h, ids, **kw: h + ids[:, None, None])
    moe = SimpleNamespace(dim=2, ffn_expert_pad16=False, ffn_down_bf16=False,
                          switch_mlp=mlp, shared_experts=np.zeros_like)
    holder = SimpleNamespace(layers=[SimpleNamespace(ffn=moe, engram=None)],
                             model=SimpleNamespace(layout=None))
    ns["attach"](holder, ".")
    monkeypatch.setenv("DSV41_FFN_EXPERT_PAD16", "0")
    ns["attach"](holder, ".")
    assert moe.ffn_expert_pad16  # Attached value survives environment changes.
    rng = np.random.default_rng(41)
    for forward, n in enumerate((64, 65, 64)):
        indices = np.argsort(rng.random((n, 8)), axis=-1)[:, :6]
        moe.gate = lambda *a: (np.ones((n, 6), np.float32), indices)
        x = np.arange(n * 2, dtype=np.float32).reshape(1, n, 2)
        host_read.reset_mock()
        layout.reset_mock()
        actual = ns["forward"](moe, x)
        layout.assert_called_once()
        # R_pad each forward; occupied-expert scalar on the first forward only.
        assert host_read.call_count == (2 if forward == 0 else 1)
        counts = np.bincount(indices.ravel(), minlength=8)
        r_pad = int((((counts + 15) // 16) * 16).sum())
        assert np.ndarray.item(host_read.call_args_list[0].args[0]) == r_pad
        reference = (x[0, :, None, :] * (indices[..., None] + 1) + indices[..., None]).sum(axis=1)
        np.testing.assert_array_equal(actual[0], reference)
        output = capsys.readouterr().out
        assert output == (f"[DSV41] FFN_EXPERT_PAD16 engaged R={n * 6} R_pad={r_pad} "
                          "occupied_experts=8\n" if forward == 0 else "")
