"""KV-selection widths, tested WITHOUT MLX so they run on the RTX box too.

V4.1 keeps only `sliding_window` tokens dense; everything older is POOLED by the Compressor, and at
attention time the Indexer attends only `index_topk` of those pooled entries. That is where the
fourfold KV saving comes from, and equally where its inconsistency comes from: anything outside the
selected set is reconstructed from a pooled group rather than read back exactly.

`config.py` deliberately imports without MLX (it carries a file-import fallback for CPU hosts), so
these load it by path rather than through the package, which pulls in mlx.core.
"""
import importlib.util
import pathlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "_dsv41_config_under_test", pathlib.Path(__file__).with_name("config.py"))


def _config_module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def _text_config(**over):
    return _config_module().TextConfig(**over)


def test_kv_widths_default_to_the_checkpoints_own_values(monkeypatch):
    monkeypatch.delenv("DSV41_INDEX_TOPK", raising=False)
    monkeypatch.delenv("DSV41_SLIDING_WINDOW", raising=False)
    cfg = _text_config()
    assert cfg.index_topk == 512
    assert cfg.sliding_window == 128


def test_kv_widths_can_be_widened_by_environment(monkeypatch):
    monkeypatch.setenv("DSV41_INDEX_TOPK", "4096")
    monkeypatch.setenv("DSV41_SLIDING_WINDOW", "1024")
    cfg = _text_config()
    assert cfg.index_topk == 4096
    assert cfg.sliding_window == 1024


def test_an_unset_or_blank_override_is_not_a_zero(monkeypatch):
    """A preset writing an empty string must not collapse the dense window to nothing."""
    monkeypatch.setenv("DSV41_INDEX_TOPK", "")
    monkeypatch.setenv("DSV41_SLIDING_WINDOW", "   ")
    cfg = _text_config()
    assert cfg.index_topk == 512
    assert cfg.sliding_window == 128


@pytest.mark.parametrize("bad", ["0", "-1", "many"])
def test_a_nonsense_width_is_refused_rather_than_coerced(monkeypatch, bad):
    monkeypatch.setenv("DSV41_INDEX_TOPK", bad)
    monkeypatch.delenv("DSV41_SLIDING_WINDOW", raising=False)
    with pytest.raises(ValueError, match="DSV41_INDEX_TOPK"):
        _text_config()


def test_compress_ratios_are_NOT_overridable(monkeypatch):
    """Deliberate: the pooling cannot be switched off from the environment.

    `checkpoint_layout` lays out tensors per ratio -- compressor and indexer weights exist only where
    the ratio is non-zero -- and a zero ratio also drops the indexer and uses plain `rope_theta` with
    YaRN off, which is exactly wrong for long context. An override here would not load the checkpoint
    the model was trained as, so the widths above are the supported way to trade the approximation
    back.
    """
    monkeypatch.setenv("DSV41_COMPRESS_RATIOS", "0,0,0,0")
    cfg = _text_config()
    assert cfg.compress_ratios[2] == 2
