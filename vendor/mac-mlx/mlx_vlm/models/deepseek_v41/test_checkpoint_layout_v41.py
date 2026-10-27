"""CPU-only release layout checks; no mlx_vlm package imports."""

import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

import pytest


def _load(name):
    spec = importlib.util.spec_from_file_location(
        "v41_layout_test_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config = _load("config")
layout = _load("checkpoint_layout")


@pytest.fixture(scope="module")
def release():
    candidates = [os.environ["DSV41_META_DIR"]] if "DSV41_META_DIR" in os.environ else [
        "~/tmp/dsv41", "./tmp/upstream_dsv41"
    ]
    roots = [Path(path).expanduser() for path in candidates]
    root = next((path for path in roots if all(
        (path / name).is_file() for name in ("v41_config.json", "v41_tensors.json")
    )), roots[0])
    paths = [root / "v41_config.json", root / "v41_tensors.json"]
    if not all(p.exists() for p in paths):
        pytest.skip("Original DeepSeek-V4.1 release metadata is absent; set DSV41_META_DIR")
    raw, meta = [json.loads(p.read_text()) for p in paths]
    return raw, meta, config.TextConfig.from_dict(raw["text_config"])


@pytest.mark.parametrize("source", ["ram", "hybrid"])
def test_release_layout(release, source):
    _, meta, c = release
    plan = layout.plan_v41(meta, c, source)
    assert plan.keys() == meta.keys()
    actual = {}
    experts = defaultdict(list)
    fp8_pairs = fp4_experts = 0
    for key, entry in plan.items():
        dtype, shape = meta[key][:2]
        if entry["op"] == "drop":
            assert entry["dst"] is None
            continue
        for out in entry.get("outputs", [entry]):
            signature = (out["expect_dtype"], out["expect_shape"])
            if out["dst"] in actual:
                assert entry["op"] in ("expert_stack", "dequant_bf16_reshape")
                assert actual[out["dst"]] == signature
            actual[out["dst"]] = signature
        if entry["op"] == "dequant_bf16_reshape":
            assert ".attn.wo_a." in key
            root = ("dspark.stages." + key[4:] if key.startswith("mtp.") else "model." + key)
            assert entry["dst"] == root.rsplit(".", 1)[0] + ".weight"
            assert entry["expect_shape"] == [8, 1024, 4096]
            assert entry["expect_dtype"] == "bfloat16"
        if entry["op"] == "view_u32_scale32" and key.endswith(".weight"):
            fp8_pairs += 1
            out, inp = shape
            scale = key[:-6] + "scale"
            assert dtype == "F8_E4M3"
            assert meta[scale][:2] == ["F8_E8M0", [out // 32, inp // 32]]
            expected_scale = [out, inp // 32]
            expected_weight = [out, inp // 4]
            assert plan[scale]["expect_shape"] == expected_scale
            assert plan[scale]["expect_dtype"] == "uint8"
            assert entry["expect_shape"] == expected_weight
            assert entry["expect_dtype"] == "uint32"
        if entry["op"] == "expert_stack":
            experts[entry["dst"]].append(entry["index"])
            if key.endswith(".weight"):
                fp4_experts += 1
                out, half_in = shape
                scale = key[:-6] + "scale"
                assert dtype == "I8"
                assert meta[scale][:2] == ["F8_E8M0", [out, half_in // 16]]
                experts_count = c.dspark_n_routed_experts if key.startswith("mtp.") else c.n_routed_experts
                assert plan[scale]["expect_shape"] == [experts_count, out, half_in // 16]
                assert entry["expect_shape"] == [experts_count, out, half_in // 4]
        if entry["op"] == "engram_chunk":
            chunks = entry["outputs"]
            assert len(chunks) == c.engram_ram_chunks
            assert sum(x["expect_shape"][0] for x in chunks) == shape[0]
            assert chunks[0]["row_start"] == 0
            assert chunks[-1]["row_stop"] == shape[0]
            for a, b in zip(chunks, chunks[1:]):
                assert a["row_stop"] == b["row_start"]
            for j, out in enumerate(chunks):
                rows = min(math.ceil(shape[0] / c.engram_ram_chunks),
                           shape[0] - j * math.ceil(shape[0] / c.engram_ram_chunks))
                assert out["expect_shape"] == [rows, 64 if key.endswith(".weight") else 8]
    assert all(sorted(indices) == list(range(c.dspark_n_routed_experts if key.startswith("dspark.") else c.n_routed_experts))
               for key, indices in experts.items())
    assert actual == layout.expected_parameters(c, source)
    assert fp8_pairs == 312
    assert fp4_experts == (c.num_hidden_layers * c.n_routed_experts + 3 * c.dspark_n_routed_experts) * 3
    for key in meta:
        if key.startswith(("vision.", "aligner.")) or key.startswith("image_"):
            assert plan[key]["op"] == "drop"
        if key.startswith("mtp."):
            assert plan[key]["op"] != "drop"
            assert plan[key]["dst"].startswith("dspark.stages.")
        if ".engram.embed." in key:
            assert plan[key]["op"] == ("engram_chunk" if source == "ram" else "drop")
    assert actual["model.layers.0.ffn.gate.bias_vl"] == ("float32", [384])
    assert actual["model.layers.0.ffn.gate.e_score_correction_bias"] == ("float32", [384])
    assert not any("hc_head" in k for k in actual)
    assert not any(".wo_a.scales" in k for k in actual)
    counts = Counter(e["op"] for e in plan.values())
    expected_counts = dict(dequant_bf16_reshape=86, drop=266 if source == "ram" else 270,
                           expert_stack=94464, rename=641, view_u32_scale32=624)
    if source == "ram":
        expected_counts["engram_chunk"] = 4
    assert counts == expected_counts
    dtype_bytes = {"bfloat16": 2, "float32": 4, "uint32": 4, "uint8": 1}
    resident_bytes = sum(dtype_bytes[dtype] * math.prod(shape) for dtype, shape in actual.values())
    assert resident_bytes == (502889816528 if source == "ram" else 300131784128) + 8047486344
    dspark_bytes = sum(dtype_bytes[dtype] * math.prod(shape)
                       for key, (dtype, shape) in actual.items() if key.startswith("dspark."))
    # The constructed DSpark stack owns the restored checkpoint tensors.
    assert dspark_bytes == 8047486344
    print(f"{source}: fp8 pairs={fp8_pairs}, fp4 expert projections={fp4_experts}, dropped={counts['drop']}; ops={dict(sorted(counts.items()))}")


def test_config_roundtrip(release):
    raw, _, _ = release
    parsed = config.ModelConfig.from_dict(raw)
    assert {f.name for f in fields(parsed.text_config)} == set(raw["text_config"]) | {
        "engram_source", "engram_ram_chunks", "model_dir", "gate_temp", "sliding_window"
    }
    assert {f.name: getattr(parsed.text_config, f.name) for f in fields(parsed.text_config)
            if f.name in raw["text_config"]} == raw["text_config"]
    for key, value in raw["text_config"].items():
        assert getattr(config.TextConfig(), key) == value
    assert vars(parsed.vision_config) == raw["vision_config"]
    assert parsed.quantization_config == raw["quantization_config"]
    enriched = dict(raw, model_dir="./checkpoint", engram_source="ram")
    parsed = config.ModelConfig.from_dict(enriched)
    assert parsed.text_config.model_dir == parsed.model_dir == "./checkpoint"
    assert parsed.text_config.engram_source == "ram"
    assert "model_dir" not in raw
    assert "engram_source" not in raw["text_config"]


@pytest.mark.parametrize("changes", [
    {"compress_ratios": [0]},
    {"compress_ratios": [4] * 40},
    {"compress_ratios": [128] * 40},
    {"kv_source_layer_ids": [0]},
    {"kv_source_layer_ids": [40]},
    {"engram_source": "invalid"},
    {"engram_ram_chunks": 0},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        config.TextConfig(**changes)


def test_backbone_ratios():
    c = config.TextConfig()
    short = c.compress_ratios[:c.num_hidden_layers]
    assert config.TextConfig(compress_ratios=short).compress_ratios == short


@pytest.mark.parametrize("mutation", ["unknown", "missing", "scale128", "expert_scale"])
def test_bad_manifest(release, mutation):
    _, meta, c = release
    changed = dict(meta)
    if mutation == "unknown":
        changed["hc_head_fn"] = ["F32", [4, 20480]]
    elif mutation == "missing":
        changed.pop("layers.0.attn.wq_a.scale")
    elif mutation == "scale128":
        changed["layers.0.attn.wq_a.scale"] = ["F8_E8M0", [10, 40]]
    else:
        changed["layers.0.ffn.experts.0.w1.scale"] = ["F8_E8M0", [72, 160]]
    with pytest.raises(ValueError):
        layout.plan_v41(changed, c, "hybrid")


def test_nvme_parameters(release):
    _, meta, c = release
    assert layout.expected_parameters(c, "nvme") == layout.expected_parameters(c, "hybrid")
    assert layout.plan_v41(meta, c, "nvme") == layout.plan_v41(meta, c, "hybrid")


@pytest.mark.parametrize("source", ["ram", "hybrid"])
def test_sanitizer_byte_operations(monkeypatch, source):
    # This adapter checks byte manipulation without claiming to test MLX kernels.
    import types

    import numpy as np

    mx = types.ModuleType("mlx.core")
    for name in ("uint32", "uint8", "int8", "float32", "repeat", "stack"):
        setattr(mx, name, getattr(np, name))
    mx.bfloat16 = np.uint16  # Opaque two-byte payload, never used in arithmetic.
    dequant_calls, evaluated = [], []

    def dequantize(weight, scale, **kwargs):
        # Check the MLX call contract; sentinel output tests the grouped reshape.
        assert len(dequant_calls) == len(evaluated)
        assert kwargs == dict(group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)
        assert weight.dtype == np.uint32
        assert scale.dtype == np.uint8
        output = np.arange(weight.size * 4, dtype=np.uint16).reshape(weight.shape[0], -1)
        dequant_calls.append((weight, scale, output))
        return output

    mx.dequantize = dequantize
    mx.eval = lambda value: evaluated.append(value)
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    c = config.TextConfig(
        hidden_size=32, head_dim=32, q_lora_rank=32, num_attention_heads=2,
        o_groups=2, o_lora_rank=32, moe_intermediate_size=32, n_routed_experts=3,
        num_hidden_layers=3, compress_ratios=[0, 2, 1], kv_source_layer_ids=[1],
        index_source_layer_ids=[1, 2], index_n_heads=1, index_head_dim=32,
        engram_layer_ids=[1], engram_num_embeddings=[65], engram_head_dim=32,
        vocab_size=32,
    )
    dtypes = {"BF16": np.uint16, "F32": np.float32, "I8": np.int8,
              "F8_E4M3": np.uint8, "F8_E8M0": np.uint8}
    weights = {
        key: np.arange(math.prod(shape)).astype(dtypes[dtype]).reshape(shape)
        for key, dtype, shape in layout._source_parameters(c)
    }
    # Scale entries precede weights to exercise order-independent pair consumption.
    weights = dict(reversed(list(weights.items())))
    weights["vision.unused"] = np.zeros(1, dtype=np.float32)
    original_keys = set(weights)
    result = layout.sanitize_weights_v41(weights, c, source)
    assert set(weights) == original_keys
    assert layout.sanitize_weights_v41(result, c, source) is result
    for name in ("wq_a", "wo_b"):
        src = f"layers.0.attn.{name}."
        dst = "model." + src
        expected_weight = weights[src + "weight"].view(np.uint32)
        expected_scale = np.repeat(weights[src + "scale"], 32, axis=0)
        np.testing.assert_array_equal(result[dst + "weight"].reshape(expected_weight.shape), expected_weight)
        np.testing.assert_array_equal(result[dst + "scales"].reshape(expected_scale.shape), expected_scale)
    assert len(dequant_calls) == len(evaluated) == c.num_hidden_layers
    for i, (packed, scale, output), value in zip(
        reversed(range(c.num_hidden_layers)), dequant_calls, evaluated
    ):
        src = f"layers.{i}.attn.wo_a."
        weight = result["model." + src + "weight"]
        assert weight is value
        assert np.shares_memory(packed, weights[src + "weight"])
        np.testing.assert_array_equal(packed, weights[src + "weight"].view(np.uint32))
        np.testing.assert_array_equal(scale, np.repeat(weights[src + "scale"], 32, axis=0))
        np.testing.assert_array_equal(weight, output.reshape(c.o_groups, c.o_lora_rank, -1))
        assert weight.dtype == mx.bfloat16
    assert not any(".wo_a.scales" in key for key in result)
    for proj, name in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
        for expert in range(c.n_routed_experts):
            src = f"layers.0.ffn.experts.{expert}.{name}."
            dst = f"model.layers.0.ffn.switch_mlp.{proj}."
            np.testing.assert_array_equal(result[dst + "weight"][expert], weights[src + "weight"].view(np.uint32))
            np.testing.assert_array_equal(result[dst + "scales"][expert], weights[src + "scale"])
    if source == "ram":
        for src_kind, dst_kind, dtype in (("weight", "weight", np.uint32), ("scale", "scales", np.uint8)):
            chunks = [result[f"model.layers.1.engram.embed.{dst_kind}.{i}"] for i in range(c.engram_ram_chunks)]
            np.testing.assert_array_equal(np.concatenate(chunks), weights[f"layers.1.engram.embed.{src_kind}"].view(dtype))
    else:
        assert not any(".engram.embed." in key for key in result)


def test_quantization_selection(monkeypatch):
    import types

    # Exercise routing independently of MLX and without constructing huge layers.
    nn = types.ModuleType("mlx.nn")
    nn.Module = types.SimpleNamespace(is_module=lambda _: True)
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda modules, **kwargs: [(name, None) for name in modules]
    mlx = types.ModuleType("mlx")
    mlx.nn = nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    prefix = "language_model.model.layers.0."
    kinds = {
        "ffn.switch_mlp.gate_proj": "mxfp4",
        "ffn.shared_experts.up_proj": "mxfp8",
        "attn.wq_a": "mxfp8",
        "attn.wo_b": "mxfp8",
        "attn.indexer.wq_b": "mxfp8",
        "engram.wkv": "mxfp8",
    }
    excluded = ["attn.compressor.wkv", "attn.indexer.wk", "attn.indexer.weights_proj",
                "ffn.gate", "engram.embed", "attn.wo_a"]
    model = types.SimpleNamespace(leaf_modules=lambda: [prefix + name for name in [*kinds, *excluded]])
    quant = layout.make_quantization_config_v41(model)
    assert {k: quant[k] for k in ("group_size", "bits", "mode")} == {
        "group_size": 64, "bits": 8, "mode": "affine"
    }
    for name, mode in kinds.items():
        assert quant[prefix + name] == {
            "group_size": 32, "bits": 4 if mode == "mxfp4" else 8, "mode": mode
        }
    assert all(prefix + name not in quant for name in excluded)
    assert not any(".wo_a" in key for key in quant)
