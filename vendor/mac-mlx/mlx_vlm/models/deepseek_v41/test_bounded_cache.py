"""Bounded cache must preserve the unbounded model's exact attention coordinates."""

import importlib.util
import os

import pytest

mx = pytest.importorskip("mlx.core")
from . import test_language_dsv41_mac as helper


@pytest.fixture
def model(monkeypatch):
    candidate = os.environ.get("DSV41_CANDIDATE_LANGUAGE")
    if candidate:
        spec = importlib.util.spec_from_file_location(
            "mlx_vlm.models.deepseek_v41._candidate_language", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(helper, "LanguageModel", module.LanguageModel)
    monkeypatch.setenv("DSV41_QAT", "1")
    monkeypatch.setenv("DSV41_METAL_SPARSE_ATTN", "0")
    return helper.random_model()


@pytest.mark.parametrize("chunk", [1, 5, 17])
def test_bounded_matches_history(model, monkeypatch, chunk):
    ids = helper.token_ids()
    monkeypatch.setenv("DSV41_BOUNDED_CACHE", "0")
    full = model.make_cache()
    monkeypatch.setenv("DSV41_BOUNDED_CACHE", "1")
    bounded = model.make_cache()
    for start in range(0, ids.shape[1], chunk):
        x = ids[:, start:start + chunk]
        expected = model(x, cache=full).logits
        actual = model(x, cache=bounded).logits
        mx.eval(actual, expected)
        assert bool(mx.array_equal(actual, expected).item())
        for attn, lc in zip(model.layers, bounded[0].layers):
            assert lc.win_kv.shape[1] <= attn.attn.window_size
            for name in ("kv", "score"):
                value = getattr(lc, name)
                if value is not None:
                    assert value.shape[1] < attn.attn.ratio
    assert bounded[0].nbytes < full[0].nbytes
    assert not bounded[0].is_trimmable()
    with pytest.raises(ValueError, match="rollback"):
        bounded[0].trim(1)
    with pytest.raises(ValueError, match="serialized"):
        _ = bounded[0].meta_state
    cloned = bounded[0].clone()
    a = model(ids[:, :1], cache=bounded).logits
    b = model(ids[:, :1], cache=[cloned]).logits
    assert bool(mx.array_equal(a, b).item())


def test_compressor_missing_history_fails(model, monkeypatch):
    monkeypatch.setenv("DSV41_BOUNDED_CACHE", "1")
    cache = model.make_cache()
    mx.eval(model(helper.token_ids()[:, :17], cache=cache).logits)
    lc = cache[0].layers[2]
    with pytest.raises(ValueError, match="no longer retained"):
        lc.absolute_slice("kv", 0, 2)


def test_index_workspace_tiles_execute_and_preserve_selection(model, monkeypatch):
    from types import SimpleNamespace

    attn = model.layers[2].attn
    x = mx.random.normal((1, 17, model.config.hidden_size))
    qr = mx.random.normal((1, 17, model.config.q_lora_rank))
    keys = mx.random.normal((1, 4096, attn.indexer.head_dim))
    original = mx.einsum
    calls = []

    def counted(expression, *args):
        if expression == "bshd,btd->bsht":
            calls.append(args[0].shape[1])
        return original(expression, *args)

    monkeypatch.setattr(mx, "einsum", counted)
    monkeypatch.setenv("DSV41_INDEX_WORKSPACE_MB", "0")
    a = attn.indexer(x, qr, 8192, 8209, attn.rope, keys, SimpleNamespace(candidates=None))
    mx.eval(a)
    assert calls == [17]
    calls.clear()
    monkeypatch.setenv("DSV41_INDEX_WORKSPACE_MB", "1")
    b = attn.indexer(x, qr, 8192, 8209, attn.rope, keys, SimpleNamespace(candidates=None))
    mx.eval(b)
    assert len(calls) > 1 and sum(calls) == 17
    assert bool(mx.array_equal(a, b).item())

