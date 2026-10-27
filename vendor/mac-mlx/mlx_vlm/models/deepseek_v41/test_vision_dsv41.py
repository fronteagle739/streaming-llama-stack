"""Windows CPU tests: real header inventory and actual port vs unedited Torch.

The NumPy adapter executes vision.py itself; it is not a second implementation
of the tower. FP32 parity tests validate equations, not Metal/BF16 kernels.
"""

import ast
from functools import partial
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest
from scipy.special import erf, expit, softmax
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from dsv41_source import source_tree


def load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


config = load("config")
layout = load("vision_layout")
enc = load("encoding_dsv41")
processor = load("image_processing")
oracle = load("reference_cpu_dsv41")
RELEASE = json.loads((HERE / "vision_checkpoint_metadata.json").read_text())


class Module:
    @staticmethod
    def is_module(x):
        return isinstance(x, Module)

    def parameters(self):
        return {k: v for k, v in flatten(self)}

    def leaf_modules(self):
        return list(leaves(self))


class Linear(Module):
    def __init__(self, inp, out, bias=True):
        self.weight = np.zeros((out, inp), np.float32)
        if bias:
            self.bias = np.zeros(out, np.float32)

    def __call__(self, x):
        return x @ self.weight.T + getattr(self, "bias", 0)


def flatten(obj, prefix=""):
    entries = enumerate(obj) if isinstance(obj, list) else vars(obj).items()
    for k, v in entries:
        name = prefix + str(k)
        if isinstance(v, np.ndarray):
            yield name, v
        elif isinstance(v, (Module, list)):
            yield from flatten(v, name + ".")


def leaves(obj, prefix=""):
    children = [(k, v) for k, v in vars(obj).items() if isinstance(v, (Module, list))]
    if not children:
        yield prefix.rstrip("."), obj
    for k, child in children:
        if isinstance(child, list):
            for i, entry in enumerate(child):
                yield from leaves(entry, prefix + k + f".{i}.")
        else:
            yield from leaves(child, prefix + k + ".")


@pytest.fixture
def runtime(monkeypatch):
    mx = types.ModuleType("mlx.core")
    for name in ("array", "arange", "ones", "zeros", "mean", "sum", "sqrt", "pad", "split",
                 "cos", "sin", "concatenate", "stack", "broadcast_to", "where", "float32",
                 "int32", "uint32", "bool_", "argpartition", "take_along_axis"):
        setattr(mx, name, getattr(np, name))
    # No BF16 arithmetic claim: source dtype validation is tested against headers.
    mx.bfloat16 = np.float32
    mx.compile = lambda fn, **kwargs: fn  # eager CPU source only
    mx.rsqrt = lambda x: 1 / np.sqrt(x)
    mx.softmax, mx.sigmoid = softmax, expit
    mx.fast = types.SimpleNamespace(scaled_dot_product_attention=lambda q, k, v, scale:
                                   softmax(q @ k.swapaxes(-1, -2) * scale, axis=-1) @ v)
    nn = types.ModuleType("mlx.nn")
    nn.Module, nn.Linear = Module, Linear
    nn.silu = lambda x: x * expit(x)
    nn.gelu = lambda x: x * .5 * (1 + erf(x / np.sqrt(2)))
    nn.softplus = lambda x: np.logaddexp(x, 0)
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda x, **kw: x if isinstance(x, list) else list(x.items())
    mlx = types.ModuleType("mlx")
    mlx.core, mlx.nn = mx, nn
    for name, module in {"mlx": mlx, "mlx.core": mx, "mlx.nn": nn, "mlx.utils": utils}.items():
        monkeypatch.setitem(sys.modules, name, module)
    # A private package gives actual runtime modules their relative imports.
    pkg = types.ModuleType("vision_test_pkg")
    pkg.__path__ = [str(HERE)]
    monkeypatch.setitem(sys.modules, pkg.__name__, pkg)
    monkeypatch.setitem(sys.modules, pkg.__name__ + ".encoding_dsv41", enc)
    monkeypatch.setitem(sys.modules, pkg.__name__ + ".vision_layout", layout)
    monkeypatch.setitem(sys.modules, pkg.__name__ + ".config", config)
    port = load(pkg.__name__ + ".vision", HERE / "vision.py")
    return port, mx, nn


def tiny():
    return config.VisionConfig(num_hidden_layers=2, hidden_size=16,
                               num_attention_heads=2, intermediate_size=12,
                               patch_size=2, min_pixels=36, max_image_tokens=64)


def ref_args(c, dim=24):
    return types.SimpleNamespace(
        vision_n_layers=c.num_hidden_layers, vision_dim=c.hidden_size,
        vision_n_heads=c.num_attention_heads, vision_inter_dim=c.intermediate_size,
        vision_patch_size=c.patch_size, vision_rope_theta=c.rope_theta,
        vision_downsample_ratio=c.downsample_ratio, vision_min_pixels=c.min_pixels,
        vision_max_wh_ratio=c.max_wh_ratio, vision_max_n_token=c.max_image_tokens,
        dim=dim, vision_enabled=True, image_token_id=129264)


@pytest.fixture
def full_graph(runtime, monkeypatch):
    port, mx, nn = runtime
    # Zero-stride views build the FULL release graph without allocating GBs.
    def virtual(shape, dtype=np.float32):
        return np.broadcast_to(np.zeros((), dtype), shape)
    class ShapeLinear(Module):
        def __init__(self, inp, out, bias=True):
            self.weight = virtual((out, inp))
            if bias:
                self.bias = virtual((out,))
    monkeypatch.setattr(nn, "Linear", ShapeLinear)
    monkeypatch.setattr(mx, "ones", virtual)
    c = config.ModelConfig.from_dict(RELEASE["config"])
    model = Module()
    model.vision = port.VisionModel(c.vision_config)
    model.aligner = port.Aligner(c.vision_config, c.text_config.hidden_size)
    for marker in layout.MARKERS:
        setattr(model, marker, virtual((c.text_config.hidden_size,)))
    graph = dict(flatten(model))
    assert graph.keys() == RELEASE["tensors"].keys()
    return graph


@pytest.mark.parametrize("key", sorted(RELEASE["tensors"]))
def test_each_real_index_tensor(key, full_graph):
    c = config.ModelConfig.from_dict(RELEASE["config"])
    expected = layout.expected_vision_parameters(c.vision_config, c.text_config.hidden_size)
    actual = RELEASE["tensors"][key]
    assert (actual["dtype"], actual["shape"]) == expected[key]
    assert list(full_graph[key].shape) == actual["shape"]
    assert actual["data_offsets"][1] - actual["data_offsets"][0] == 2 * np.prod(actual["shape"])
    assert actual["shard"] in ("model-00001-of-00048.safetensors", "model-00002-of-00048.safetensors")


def test_inventory_is_complete_and_strict():
    c = config.ModelConfig.from_dict(RELEASE["config"])
    metadata = RELEASE["tensors"]
    assert len(metadata) == 266
    assert sum(k.startswith("vision.") for k in metadata) == 259
    assert sum(k.startswith("aligner.") for k in metadata) == 4
    layout.validate_vision_metadata(metadata, c.vision_config, c.text_config.hidden_size)
    for mutated in ({k: v for k, v in metadata.items() if k != "image_start"},
                    {**metadata, "vision.unknown": metadata["image_start"]},
                    {**metadata, "image_start": {"dtype": "F32", "shape": [5120]}},
                    {**metadata, "aligner.w1.weight": {"dtype": "BF16", "shape": [9216, 5120]}}):
        with pytest.raises(ValueError):
            layout.validate_vision_metadata(mutated, c.vision_config, c.text_config.hidden_size)


def test_constructed_tree_and_per_tensor_quantization(runtime):
    port, _, _ = runtime
    c = tiny()
    model = Module()
    model.vision, model.aligner = port.VisionModel(c), port.Aligner(c, 24)
    for marker in layout.MARKERS:
        setattr(model, marker, np.zeros(24, np.float32))
    expected = layout.expected_vision_parameters(c, 24)
    actual = dict(flatten(model))
    assert actual.keys() == expected.keys()
    assert {k: list(v.shape) for k, v in actual.items()} == {k: s for k, (_, s) in expected.items()}
    assert layout.sanitize_vision_weights(actual, c, 24) is actual
    checkpoint = load("checkpoint_layout")
    graph = model.leaf_modules() + [
        ("language_model.model.layers.0.ffn.switch_mlp.gate_proj", Linear(32, 32)),
        ("language_model.model.layers.0.attn.wq_a", Linear(32, 32)),
    ]
    model.leaf_modules = lambda: graph
    quant = checkpoint.make_quantization_config_v41(model)
    for key, _ in graph:
        if key.startswith(("vision.", "aligner.")):
            assert quant[key] is False
    assert quant[graph[-2][0]] == dict(bits=4, group_size=32, mode="mxfp4")
    assert quant[graph[-1][0]] == dict(bits=8, group_size=32, mode="mxfp8")
    bad = dict(actual)
    bad["vision.blocks.0.attn.wo.weight"] = np.zeros((16, 16), np.uint32)
    with pytest.raises(ValueError):
        layout.sanitize_vision_weights(bad, c, 24)


@pytest.mark.parametrize("grid", [(3, 3), (4, 7), (1, 5)])
def test_actual_port_reference_cpu_parity(runtime, grid):
    port, _, _ = runtime
    ref = oracle._load(oracle.REF_DIR / "vision.py", "_vision_unedited_reference")
    c, (h, w) = tiny(), grid
    with torch.random.fork_rng():
        torch.manual_seed(41)
        rv, ra = ref.ViT(ref_args(c)).float(), ref.Aligner(ref_args(c)).float()
    pv, pa = port.VisionModel(c), port.Aligner(c, 24)
    for ours, reference in ((pv, rv), (pa, ra)):
        dest = dict(flatten(ours))
        assert dest.keys() == reference.state_dict().keys()
        for key, tensor in reference.state_dict().items():
            dest[key][...] = tensor.numpy()
    patches = np.random.default_rng(41).normal(size=(h * w, 3, 2, 2)).astype(np.float32)
    with torch.no_grad():
        expected_v = rv(torch.from_numpy(patches), h, w)
        expected_a = ra(expected_v, h, w).numpy()
    actual_v = pv(patches, h, w)
    np.testing.assert_allclose(actual_v, expected_v.numpy(), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(pa(actual_v, h, w), expected_a, atol=2e-6, rtol=2e-5)
    assert actual_v.dtype == np.float32
    assert pa(actual_v, h, w).shape == (((h + 2) // 3) * ((w + 2) // 3), 24)
    with pytest.raises(ValueError):
        pv(patches[:-1], h, w)


@pytest.mark.parametrize("size", [(1, 1), (547, 349), (4000, 3000), (10, 6000), (6000, 10)])
@pytest.mark.parametrize("clamp", [None, 2])
def test_preprocessing_reference_exact(size, clamp):
    ref = oracle._load(oracle.REF_DIR / "image_processor.py", "_image_unedited_reference")
    c = config.VisionConfig(max_wh_ratio=clamp)
    assert processor.plan_image_grid(*size, c) == ref.plan_image_grid(*size, ref_args(c))
    # Nonuniform channels/pixels catch resampling, normalization and CHW ordering.
    pixels = np.random.default_rng(3).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    record = {"data": buffer.getvalue()}
    actual, *ag = processor.load_image(record, c)
    expected, *eg = ref.load_image(record, ref_args(c))
    assert ag == eg
    assert actual.dtype == np.float32
    assert torch.equal(torch.from_numpy(actual).to(torch.bfloat16), expected)
    assert ag[2] * (ag[3] + 1) + 2 <= c.max_image_tokens


class Tokenizer:
    """Character-level text tokens, with one atomic configured image token."""
    def encode(self, text, add_special_tokens=False):
        out = []
        for i, part in enumerate(text.split(enc.IMAGE_PLACEHOLDER)):
            if i:
                out.append(129264)
            out.extend(ord(x) for x in part)
        return out

    def convert_tokens_to_ids(self, text):
        return 129264

    unk_token_id = None


def test_one_image_exact_chat_tokens_and_embeddings(runtime, monkeypatch):
    port, _, _ = runtime
    messages = [{"role": "user", "content": [{"type": "text", "text": "Describe "},
                 {"type": "image_url", "image_url": {"url": "one.png"}},
                 {"type": "text", "text": " please."}]}]
    prompt, media = enc.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
    c = config.ModelConfig(vision_config=tiny())
    patches = np.zeros((4 * 7, 3, 2, 2), np.float32)
    loader = lambda *args: (patches, 4, 7, 2, 3)
    ids, types_, images = enc.prepare_vision_inputs(prompt, media["images"], Tokenizer(), c, loader)
    before, after = prompt.split(enc.IMAGE_PLACEHOLDER)
    expected_types = [0, 1, 1, 1, 2, 1, 1, 1, 2, 3]
    assert ids == [*map(ord, before), *([129264] * 10), *map(ord, after)]
    assert types_ == [-1] * len(before) + expected_types + [-1] * len(after)
    assert images[0].start == len(before)
    # Cross-check the unedited reference's complete prompt expansion.
    ref = oracle._load(oracle.REF_DIR / "image_processor.py", "_image_encoding_reference")
    monkeypatch.setitem(sys.modules, "encoding", enc)
    monkeypatch.setattr(ref, "load_image", loader)
    rid, rtypes, _ = ref.prepare_vl_inputs(prompt, media["images"], Tokenizer(), ref_args(tiny()))
    assert (ids, types_) == (rid, rtypes)
    model = types.SimpleNamespace(
        config=c, vision=lambda *args: None,
        aligner=lambda *args: np.arange(12, dtype=np.float32).reshape(6, 2),
        image_start=np.array([-10, -11], np.float32),
        image_newline=np.array([-20, -21], np.float32),
        image_end=np.array([-30, -31], np.float32))
    embeds = np.full((1, len(ids), 2), 100, np.float32)
    actual = port.merge_image_embeddings(model, np.array([ids]), embeds, images)
    expected = np.array([[-10, -11], [0, 1], [2, 3], [4, 5], [-20, -21],
                         [6, 7], [8, 9], [10, 11], [-20, -21], [-30, -31]], np.float32)
    np.testing.assert_array_equal(actual[0, len(before):len(before) + 10], expected)
    np.testing.assert_array_equal(actual[:, :len(before)], embeds[:, :len(before)])
    np.testing.assert_array_equal(actual[:, len(before) + 10:], embeds[:, len(before) + 10:])
    assert actual.dtype == embeds.dtype
    with pytest.raises(ValueError, match="placeholder count"):
        enc.prepare_vision_inputs(prompt, [], Tokenizer(), c, loader)
    with pytest.raises(ValueError, match="configured vision"):
        enc.prepare_vision_inputs(prompt, media["images"], Tokenizer(), config.ModelConfig(), loader)
    images[0].types = [3] * 10
    with pytest.raises(ValueError, match="span layout"):
        port.merge_image_embeddings(model, np.array([ids]), embeds, images)


def extracted_language_classes(runtime, names, extra=None):
    # Same structural CPU strategy used by the existing checkpoint/DSpark tests,
    # but execute the actual changed class methods, including masks and calls.
    _, mx, nn = runtime
    tree = ast.parse((HERE / "language.py").read_text())
    env = dict(mx=mx, nn=nn, np=np, os=os, partial=partial, TextConfig=config.TextConfig,
               __name__="vision_test_pkg.extracted_language", __package__="vision_test_pkg",
               **(extra or {}))
    exec(compile(source_tree(tree, names, env), "language.py", "exec"), env)
    return env


def test_image_routing_selects_bias_vl(runtime):
    gate = extracted_language_classes(runtime, ["Gate"])["Gate"](
        config.TextConfig(n_routed_experts=3, num_experts_per_tok=1, hidden_size=2))
    gate.e_score_correction_bias[:] = [10, 0, 0]
    gate.bias_vl[:] = [0, 10, 0]
    values = np.zeros((3, 2), np.float32)
    _, indices = gate(values, np.array([False, True, False]))
    np.testing.assert_array_equal(indices[:, 0], [0, 1, 0])
    _, indices = gate(values)
    np.testing.assert_array_equal(indices[:, 0], [0, 0, 0])


def test_image_masks_reach_hash_contribution_and_blocks(runtime):
    seen = []
    def hashes(ids, offset, mask):
        seen.append(("hash", mask.copy()))
        return np.zeros((*ids.shape, 1, 1), np.int64)
    class Engram:
        layer_hash_index = 0
        def __call__(self, h, ids, mask):
            seen.append(("engram", mask.copy()))
            return h
    class Block:
        engram = Engram()
        def __call__(self, h, pre, start, cache, shared, image_mask):
            seen.append(("block", image_mask.copy()))
            return h, pre
    env = extracted_language_classes(runtime, ["DeepseekV41Model"],
        dict(SharedState=lambda: None, hc_pre=lambda h, pre: h[:, :, 0]))
    model = env["DeepseekV41Model"].__new__(env["DeepseekV41Model"])
    model.config = types.SimpleNamespace(hc_mult=2)
    model.layers, model.norm = [Block()], lambda x: x
    model.embed_tokens = lambda ids: np.zeros((*ids.shape, 2), np.float32)
    cache = types.SimpleNamespace(offset=0, engram=types.SimpleNamespace(
        hash_state=types.SimpleNamespace(hashes=hashes)))
    cache.finish = lambda end: setattr(cache, "_offset", end)
    mask = np.array([[False, True, True, False]])
    model(np.array([[9, 129264, 129264, 10]]), cache, image_mask=mask)
    assert [name for name, _ in seen] == ["hash", "engram", "block"]
    for name, value in seen:
        np.testing.assert_array_equal(value, mask if name == "block" else ~mask)


def test_mask_reaches_gate_through_moe_and_block(runtime):
    seen = []
    env = extracted_language_classes(runtime, ["MoE", "DeepseekV41Block"],
        dict(hc_pre=lambda x, p: x, hc_post=lambda x, *args: x))
    moe = env["MoE"].__new__(env["MoE"])
    moe.dim = 2
    def gate(x, mask):
        seen.append(mask.copy())
        return np.ones((x.shape[0], 1), np.float32), np.zeros((x.shape[0], 1), np.int32)
    moe.gate = gate
    moe.switch_mlp = lambda x, indices: x[:, None, :]
    moe.shared_experts = lambda x: x * 0
    block = env["DeepseekV41Block"].__new__(env["DeepseekV41Block"])
    block.ffn, block.attn = moe, lambda x, *args: x
    block.attn_hc = block.ffn_hc = lambda h: (None, None, None)
    block.attn_norm = block.ffn_norm = lambda x: x
    mask = np.array([[False, True, False]])
    block(np.ones((1, 3, 2), np.float32), None, 0, None, None, mask)
    np.testing.assert_array_equal(seen[0], mask.reshape(-1))


def test_language_entry_derives_mask_per_chunk(runtime):
    class Cache:
        pass
    seen = []
    class Backbone:
        layout = None
        def __call__(self, ids, cache, embeds, sink, skip, **kwargs):
            seen.append(kwargs["image_mask"])
            return np.zeros((*ids.shape, 2), np.float32)
    env = extracted_language_classes(runtime, ["LanguageModel"],
        dict(DeepseekV41Cache=Cache, LanguageModelOutput=types.SimpleNamespace))
    lm = env["LanguageModel"].__new__(env["LanguageModel"])
    lm.config = types.SimpleNamespace(hidden_size=2)
    lm.model, lm.lm_head, lm.image_token_id = Backbone(), lambda x: x, 129264
    cache = [Cache()]
    for ids in ([[4, 129264, 129264]], [[129264, 5]], [[6]]):
        values = np.array(ids, np.int32)
        lm(values, cache=cache, inputs_embeds=np.zeros((*values.shape, 2), np.float32))
        np.testing.assert_array_equal(seen[-1], values == 129264)
    with pytest.raises(ValueError, match="image_mask"):
        lm(np.array([[6]]), cache=cache, image_mask=np.ones((1, 2), bool))


def test_actual_top_level_vision_hooks(runtime, monkeypatch):
    port, _, _ = runtime
    child = types.ModuleType("vision_test_pkg.child")
    child.__path__ = [str(HERE)]
    monkeypatch.setitem(sys.modules, child.__name__, child)
    base = types.ModuleType("vision_test_pkg.base")
    base.InputEmbeddingsFeatures = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, base.__name__, base)
    for name, mod in {"vision": port, "config": config, "vision_layout": layout}.items():
        monkeypatch.setitem(sys.modules, child.__name__ + "." + name, mod)
    class Language(Module):
        def __init__(self, c):
            self.model = types.SimpleNamespace(embed_tokens=lambda ids:
                np.zeros((*ids.shape, c.hidden_size), np.float32))
        def __call__(self, ids, cache=None, **kw):
            return kw
        def sanitize(self, weights):
            assert weights.keys() == {"model.fake"}
            return weights
        quant_predicate = staticmethod(lambda path, module: True)
    lang = types.ModuleType(child.__name__ + ".language")
    lang.LanguageModel = Language
    monkeypatch.setitem(sys.modules, lang.__name__, lang)
    wrapper = load(child.__name__ + ".deepseek_v41", HERE / "deepseek_v41.py")
    c = config.ModelConfig(text_config=config.TextConfig(hidden_size=24), vision_config=tiny())
    model = wrapper.Model(c)
    assert model.language_model.image_token_id == c.image_token_id
    weights = {k: v.copy() for k, v in flatten(model)}
    assert weights.keys() == layout.expected_vision_parameters(c.vision_config, 24).keys()
    weights["model.fake"] = np.zeros(1, np.float32)
    sanitized = model.sanitize(weights)
    assert sanitized.keys() == (weights.keys() - {"model.fake"}) | {"language_model.model.fake"}
    assert model.sanitize(sanitized).keys() == sanitized.keys()
    for name in ("vision.blocks.0.attn.wqkv", "aligner.w1"):
        assert model.quant_predicate(name, None) is False
    assert model.quant_predicate("language_model.model.fake", None) is True
    model.image_start[:] = 7
    _, _, images = enc.prepare_vision_inputs(enc.IMAGE_PLACEHOLDER, [{}], Tokenizer(), c,
        lambda *args: (np.zeros((9, 3, 2, 2), np.float32), 3, 3, 1, 1))
    ids = np.full((1, 4), 129264, np.int32)
    result = model(ids, pixel_values=images)
    assert result["inputs_embeds"].shape == (1, 4, 24)
    np.testing.assert_array_equal(result["inputs_embeds"][0, 0], np.full(24, 7))
    np.testing.assert_array_equal(result["image_mask"], np.ones((1, 4), bool))
    with pytest.raises(ValueError, match="missing"):
        model.sanitize({"model.fake": np.zeros(1, np.float32)})


def test_real_tokenizer_image_placeholder():
    from tokenizers import Tokenizer as RealTokenizer

    path = oracle.TOKENIZER_DIR / "tokenizer.json"
    if not path.is_file():
        pytest.skip(f"Set DSV41_TOKENIZER_DIR for real tokenizer test: {path}")
    tokenizer = RealTokenizer.from_file(str(path))
    c = config.ModelConfig.from_dict(RELEASE["config"])
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": "one.png"},
                 {"type": "text", "text": "Describe this image."}]}]
    prompt, media = enc.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
    raw = tokenizer.encode(prompt, add_special_tokens=False).ids
    assert raw.count(c.image_token_id) == 1
    ids, types_, images = enc.prepare_vision_inputs(prompt, media["images"], tokenizer, c,
        lambda *args: (None, 3, 3, 1, 1))
    at = raw.index(c.image_token_id)
    assert ids == raw[:at] + [c.image_token_id] * 4 + raw[at + 1:]
    assert types_[at:at + 4] == [0, 1, 2, 3]
    assert images[0].start == at


def test_smoke_quantization_derives_each_packed_shape(runtime, monkeypatch):
    _, _, nn = runtime
    checkpoint = load("checkpoint_layout")
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.deepseek_v41.checkpoint_layout", checkpoint)
    smoke = load("_vision_smoke", HERE.parents[2] / "scripts" / "dsv41_vision_smoke.py")
    dense = "language_model.model.layers.0.attn.wq_a"
    expert = "language_model.model.layers.0.ffn.switch_mlp.gate_proj"
    vision = "vision.patch_embed.proj"
    graph = {dense + ".weight": np.zeros((32, 64), np.float32),
             expert + ".weight": np.zeros((2, 32, 64), np.float32),
             vision + ".weight": np.zeros((16, 588), np.float32)}
    model = types.SimpleNamespace(parameters=lambda: graph,
        leaf_modules=lambda: [(p, Linear(64, 32)) for p in (dense, expert, vision)])
    weights = {dense + ".weight": np.zeros((32, 16), np.uint32),
               dense + ".scales": np.zeros((32, 2), np.uint8),
               expert + ".weight": np.zeros((2, 32, 8), np.uint32),
               expert + ".scales": np.zeros((2, 32, 2), np.uint8),
               vision + ".weight": graph[vision + ".weight"]}
    applied = {}
    def quantize(model, bits, group_size, mode, class_predicate):
        for p in (dense, expert, vision):
            if class_predicate(p, None):
                assert p not in applied
                applied[p] = (bits, group_size, mode)
    monkeypatch.setattr(nn, "quantize", quantize, raising=False)
    smoke.quantize_loaded_model(model, weights)
    assert applied == {dense: (8, 32, "mxfp8"), expert: (4, 32, "mxfp4")}
    weights[dense + ".scales"] = np.zeros((32, 1), np.uint8)
    with pytest.raises(ValueError, match="disagrees with graph"):
        smoke.quantize_loaded_model(model, weights)
