"""Native regression for the S4 / bounded-index-workspace merge boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.deepseek_v41 import language
from mlx_vlm.models.deepseek_v41.test_language_dsv41_mac import tiny_config


@pytest.mark.parametrize("compiled", ["0", "1"])
def test_s4_scoring_keeps_workspace_bound_and_selected_positions(monkeypatch, compiled):
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", "1")
    monkeypatch.setenv("DSV41_S4_BYPASS", "0")
    monkeypatch.setenv("DSV41_S4_COMPILE_INDEXER", compiled)
    monkeypatch.setenv("DSV41_QAT", "1")
    mx.random.seed(41)
    cfg = tiny_config()
    indexer = language.Indexer(cfg, 5)
    rope = language.Attention(5, cfg).rope
    n, width = 17, 4096
    x = mx.random.normal((1, n, cfg.hidden_size))
    qr = mx.random.normal((1, n, cfg.q_lora_rank))
    keys = mx.random.normal((1, width, cfg.index_head_dim))
    mx.eval(indexer.parameters(), x, qr, keys)

    monkeypatch.setenv("DSV41_INDEX_WORKSPACE_MB", "0")
    expected = indexer(x, qr, width - n, 0, rope, keys,
                       SimpleNamespace(candidates=None))
    mx.eval(expected)

    calls = []
    original = language._index_scores
    def counted(x, *args, **kwargs):
        calls.append(x.shape[1])
        return original(x, *args, **kwargs)
    if compiled == "1":
        original_compiled = original.compiled
        def counted_compiled(x, *args, **kwargs):
            calls.append(x.shape[1])
            return original_compiled(x, *args, **kwargs)
        counted.compiled = counted_compiled
    monkeypatch.setattr(language, "_index_scores", counted)
    monkeypatch.setenv("DSV41_INDEX_WORKSPACE_MB", "1")
    actual = indexer(x, qr, width - n, 0, rope, keys,
                     SimpleNamespace(candidates=None))
    mx.eval(actual)
    assert calls == [8, 8, 1], "S4 scoring must retain mainline's query tiles"
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
