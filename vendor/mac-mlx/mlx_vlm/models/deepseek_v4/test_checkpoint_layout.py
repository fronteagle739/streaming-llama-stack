"""Weight-free structural validation of the DeepSeek-V4-Flash-DSpark layout.

Compares the pure-python expected layout (built ONLY from config.json values,
mirroring the module structure) bit-for-bit against the vendored manifest of
the REAL HF release (names + dtypes + shapes of all 72,317 tensors, captured
from the safetensors index + shard headers). Any divergence between what the
MLX modules expect and what the checkpoint ships shows up here — with no
weights, no mlx, no numpy required (runs on the Windows dev box).

  pytest mlx_vlm/models/deepseek_v4/test_checkpoint_layout.py    (Mac)
  python mlx_vlm/models/deepseek_v4/test_checkpoint_layout.py    (any box)
"""

try:
    from mlx_vlm.models.deepseek_v4.checkpoint_layout import (
        dspark_checkpoint_layout,
        expand_manifest,
        full_checkpoint_layout,
        load_flash_config,
        load_manifest,
        num_draft_layers,
        target_checkpoint_layout,
    )
except ImportError:  # no mlx on this box: import the (mlx-free) module by path
    import importlib.util
    import os

    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_layout.py")
    _spec = importlib.util.spec_from_file_location("_dsv4_checkpoint_layout", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    dspark_checkpoint_layout = _m.dspark_checkpoint_layout
    expand_manifest = _m.expand_manifest
    full_checkpoint_layout = _m.full_checkpoint_layout
    load_flash_config = _m.load_flash_config
    load_manifest = _m.load_manifest
    num_draft_layers = _m.num_draft_layers
    target_checkpoint_layout = _m.target_checkpoint_layout


def _diff(expected, actual, what):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    assert not missing, f"{what}: expected keys absent from checkpoint: {missing[:10]}"
    assert not extra, f"{what}: checkpoint keys the layout does not expect: {extra[:10]}"
    mism = [
        (k, expected[k], actual[k])
        for k in expected
        if expected[k] != actual[k]
    ]
    assert not mism, f"{what}: dtype/shape mismatches: {mism[:10]}"


def test_manifest_covers_all_tensors():
    manifest = load_manifest()
    real = expand_manifest(manifest)
    assert len(real) == manifest["total_tensors"] == 72317


def test_config_layer_accounting():
    cfg = load_flash_config()
    ratios = cfg["compress_ratios"]
    n = cfg["num_hidden_layers"]
    assert n == 43 and len(ratios) == 46
    assert num_draft_layers(cfg) == 3, "3 DSpark draft layers per official inference config"
    assert all(r == 0 for r in ratios[n:]), "draft layers use raw sliding-window attention"
    target = ratios[:n]
    assert target.count(4) == 21 and target.count(128) == 20 and target.count(0) == 2
    assert cfg["dspark_block_size"] == 5
    assert cfg["dspark_target_layer_ids"] == [40, 41, 42]
    assert cfg["dspark_noise_token_id"] == 128799
    assert cfg["dspark_markov_rank"] == 256


def test_target_layout_matches_release():
    cfg = load_flash_config()
    real = expand_manifest(load_manifest())
    real_target = {k: v for k, v in real.items() if not k.startswith("mtp.")}
    _diff(target_checkpoint_layout(cfg), real_target, "target")


def test_dspark_layout_matches_release():
    cfg = load_flash_config()
    real = expand_manifest(load_manifest())
    real_mtp = {k: v for k, v in real.items() if k.startswith("mtp.")}
    _diff(dspark_checkpoint_layout(cfg), real_mtp, "dspark")


def test_full_layout_exact():
    cfg = load_flash_config()
    real = expand_manifest(load_manifest())
    _diff(full_checkpoint_layout(cfg), real, "full")


def test_structural_facts_used_by_modules():
    """Facts the MLX modules rely on, pinned against the real release."""
    cfg = load_flash_config()
    real = expand_manifest(load_manifest())

    # Indexer only on ratio-4 layers; compressor on every compressed layer.
    for i, r in enumerate(cfg["compress_ratios"][: cfg["num_hidden_layers"]]):
        assert (f"layers.{i}.attn.compressor.ape" in real) == (r != 0)
        assert (f"layers.{i}.attn.indexer.wq_b.weight" in real) == (r == 4)
        assert (f"layers.{i}.ffn.gate.tid2eid" in real) == (i < cfg["num_hash_layers"])
        assert (f"layers.{i}.ffn.gate.bias" in real) == (i >= cfg["num_hash_layers"])

    # DSpark head: context proj on stage 0 only; heads on the last stage only.
    assert "mtp.0.main_proj.weight" in real and "mtp.1.main_proj.weight" not in real
    assert "mtp.2.markov_head.markov_w1.weight" in real
    assert "mtp.2.confidence_head.proj.weight" in real
    assert "mtp.0.markov_head.markov_w1.weight" not in real
    # Draft blocks are bias-routed (never hash) with no compressor/indexer.
    for s in range(3):
        assert f"mtp.{s}.ffn.gate.bias" in real
        assert f"mtp.{s}.ffn.gate.tid2eid" not in real
        assert f"mtp.{s}.attn.compressor.ape" not in real

    # Confidence head input = hidden + markov rank (no bias term shipped).
    assert real["mtp.2.confidence_head.proj.weight"][1] == (
        1,
        cfg["hidden_size"] + cfg["dspark_markov_rank"],
    )
    assert "mtp.2.confidence_head.proj.bias" not in real

    # main_proj consumes len(target_layer_ids) x hidden concatenated features.
    assert real["mtp.0.main_proj.weight"][1] == (
        cfg["hidden_size"],
        cfg["hidden_size"] * len(cfg["dspark_target_layer_ids"]),
    )

    # The drafter shares the target's embedding + lm head (none under mtp.*).
    assert not any(k.startswith("mtp.") and "embed" in k for k in real)
    assert not any(k.startswith("mtp.") and k.endswith("head.weight") and "markov" not in k and "confidence" not in k for k in real)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"CHECKPOINT_LAYOUT_TEST ALL PASS ({len(fns)} tests)")


if __name__ == "__main__":
    main()
