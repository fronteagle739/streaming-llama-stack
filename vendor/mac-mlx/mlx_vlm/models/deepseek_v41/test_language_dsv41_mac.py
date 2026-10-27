"""Mac-only tests for the V4.1 language port.

Tiny config and transplant adapted from PipeNetwork/deepseek-v41-mlx
(Apache-2.0), commit 8bdd543, tests/test_parity.py. Batch reduced from 2 to 1
because this slice explicitly supports one conversation. Set DSV41_PN_DIR to
the upstream checkout to enable the optional Torch CPU reference comparison.
The numerical reference test substitutes fp32 dequantized Engram embeddings,
as upstream does; ordinary cache tests additionally cover native mxfp8 RAM rows.

Run: python -m pytest mlx_vlm/models/deepseek_v41/test_language_dsv41_mac.py -v
"""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="DeepSeek-V4.1 language tests require MLX on Mac")
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.models.deepseek_v41.config import TextConfig
from mlx_vlm.models.deepseek_v41.language import LanguageModel

VOCAB, S, DECODE, MAXSEQ = 96, 48, 8, 64

CFG = dict(
    vocab_size=VOCAB, dim=64, n_layers=8, moe_inter_dim=32,
    n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=3,
    score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=1.0,
    q_lora_rank=32, head_dim=64, rope_head_dim=16, norm_eps=1e-20,
    o_groups=4, o_lora_rank=16, window_size=8,
    compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1),
    kv_source_layers=(2, 5), index_source_layers=(2, 5, 7),
    candidate_source_layer=5, candidate_topk_blocks=3, candidate_block_size=2,
    original_seq_len=64, rope_theta=10000.0, rope_factor=4.0, beta_fast=32, beta_slow=1,
    compress_rope_theta=40000.0,
    index_n_heads=8, index_head_dim=32, index_topk=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 2), engram_num_embeddings=(384, 552),
    engram_max_ngram_size=4, engram_vocab_size=50,
    engram_n_heads=2, engram_head_dim=32,
    engram_compressed_vocab_size=VOCAB,
)


def transplant(rm, mm, rng, torch):
    D = CFG["dim"]

    def w(*shape, s=None):
        fan = shape[-1] if len(shape) > 1 else shape[0]
        s = s if s is not None else fan ** -0.5
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def setp(param, arr):
        with torch.no_grad():
            param.copy_(torch.from_numpy(np.asarray(arr)))

    def setl(rl, ml, W):
        setp(rl.weight, W)
        ml.weight = mx.array(W)

    def setnorm(rn, mn, dim):
        g = (1.0 + 0.1 * rng.standard_normal(dim)).astype(np.float32)
        setp(rn.weight, g)
        mn.weight = mx.array(g)

    setl(rm.embed, mm.model.embed_tokens, w(VOCAB, D, s=0.5))
    setl(rm.head, mm.lm_head, w(VOCAB, D, s=0.3))
    setnorm(rm.norm, mm.model.norm, D)

    for li in range(CFG["n_layers"]):
        rb, mb = rm.layers[li], mm.layers[li]
        for nm in ("attn", "ffn"):
            fn = w((2 + 4) * 4, 4 * D, s=0.05)
            ba = (0.2 * rng.standard_normal((2 + 4) * 4)).astype(np.float32)
            sc = (0.5 + 0.2 * rng.standard_normal(3)).astype(np.float32)
            setp(getattr(rb, f"hc_{nm}_fn"), fn); setattr(getattr(mb, f"{nm}_hc"), "fn", mx.array(fn))
            setp(getattr(rb, f"hc_{nm}_base"), ba); setattr(getattr(mb, f"{nm}_hc"), "base", mx.array(ba))
            setp(getattr(rb, f"hc_{nm}_scale"), sc); setattr(getattr(mb, f"{nm}_hc"), "scale", mx.array(sc))
        setnorm(rb.attn_norm, mb.attn_norm, D)
        setnorm(rb.ffn_norm, mb.ffn_norm, D)

        ra, ma = rb.attn, mb.attn
        NH, HD = CFG["n_heads"], CFG["head_dim"]
        QLR, OLR, G = CFG["q_lora_rank"], CFG["o_lora_rank"], CFG["o_groups"]
        setl(ra.wq_a, ma.wq_a, w(QLR, D))
        setl(ra.wq_b, ma.wq_b, w(NH * HD, QLR))
        setl(ra.wkv, ma.wkv, w(HD, D))
        wo = w(G * OLR, NH * HD // G)
        setp(ra.wo_a.weight, wo)
        ma.wo_a.weight = mx.array(wo.reshape(G, OLR, -1))
        setl(ra.wo_b, ma.wo_b, w(D, G * OLR))
        setnorm(ra.q_norm, ma.q_norm, QLR)
        setnorm(ra.kv_norm, ma.kv_norm, HD)
        sk = w(NH, s=0.5)
        setp(ra.attn_sink, sk); ma.attn_sink = mx.array(sk)

        if ra.compressor is not None:
            setl(ra.compressor.wkv, ma.compressor.wkv, w(HD, D))
            if CFG["compress_ratios"][li] > 1:
                setl(ra.compressor.wgate, ma.compressor.wgate, w(HD, D))
            setnorm(ra.compressor.norm, ma.compressor.norm, HD)
        if ra.indexer is not None:
            INH, IHD = CFG["index_n_heads"], CFG["index_head_dim"]
            setl(ra.indexer.wq_b, ma.indexer.wq_b, w(INH * IHD, QLR))
            setl(ra.indexer.weights_proj, ma.indexer.weights_proj, w(INH, D))
            if ra.indexer.owns_k:
                setl(ra.indexer.wk, ma.indexer.wk, w(IHD, HD))
                setnorm(ra.indexer.k_norm, ma.indexer.k_norm, IHD)

        NE, INTER = CFG["n_routed_experts"], CFG["moe_inter_dim"]
        gw = w(NE, D)
        setp(rb.ffn.gate.weight, gw); mb.ffn.gate.weight = mx.array(gw)
        gb = (0.3 * rng.standard_normal(NE)).astype(np.float32)
        setp(rb.ffn.gate.bias, gb); mb.ffn.gate.e_score_correction_bias = mx.array(gb)
        mb.ffn.gate.bias_vl = mx.zeros((NE,), dtype=mx.float32)  # unused, text-only

        e1, e2, e3 = w(NE, INTER, D), w(NE, D, INTER), w(NE, INTER, D)
        for i in range(NE):
            setp(rb.ffn.experts[i].w1.weight, e1[i])
            setp(rb.ffn.experts[i].w2.weight, e2[i])
            setp(rb.ffn.experts[i].w3.weight, e3[i])
        mb.ffn.switch_mlp.gate_proj.weight = mx.array(e1)
        mb.ffn.switch_mlp.down_proj.weight = mx.array(e2)
        mb.ffn.switch_mlp.up_proj.weight = mx.array(e3)
        setl(rb.ffn.shared_experts.w1, mb.ffn.shared_experts.gate_proj, w(INTER, D))
        setl(rb.ffn.shared_experts.w2, mb.ffn.shared_experts.down_proj, w(D, INTER))
        setl(rb.ffn.shared_experts.w3, mb.ffn.shared_experts.up_proj, w(INTER, D))

        if rb.engram is not None:
            re, me = rb.engram, mb.engram
            rows = CFG["engram_num_embeddings"][re.layer_hash_index]
            EHD = CFG["engram_head_dim"]
            vals = torch.from_numpy(w(rows, EHD, s=0.5)).to(torch.float8_e4m3fn).float().numpy()
            scales = (2.0 ** rng.integers(-2, 3, size=(rows, EHD // 32))).astype(np.float32)
            setp(re.embed.weight, vals)
            setp(re.embed.scale, scales)
            # As in PN's CPU harness, avoid the storage adapter's bf16 return:
            # these are exact dequantized fp8 rows, retained in fp32 for matmul.
            me.embed = nn.Embedding(rows, EHD)
            me.embed.weight = mx.array((vals.reshape(rows, -1, 32) * scales[..., None]).reshape(rows, EHD))
            n_cols = (CFG["engram_max_ngram_size"] - 1) * CFG["engram_n_heads"]
            setl(re.wkv, me.wkv, w(D * 5, n_cols * EHD))
            qw = (1.0 + 0.3 * rng.standard_normal((4, D))).astype(np.float32)
            kw = (1.0 + 0.3 * rng.standard_normal((4, D))).astype(np.float32)
            setp(re.q_weight, qw); me.q_weight = mx.array(qw)
            setp(re.k_weight, kw); me.k_weight = mx.array(kw)


def tiny_config():
    names = {
        "dim": "hidden_size", "n_layers": "num_hidden_layers",
        "moe_inter_dim": "moe_intermediate_size", "n_heads": "num_attention_heads",
        "n_activated_experts": "num_experts_per_tok", "score_func": "scoring_func",
        "route_scale": "routed_scaling_factor", "rope_head_dim": "qk_rope_head_dim",
        "norm_eps": "rms_norm_eps", "window_size": "sliding_window",
        "kv_source_layers": "kv_source_layer_ids", "index_source_layers": "index_source_layer_ids",
        "candidate_source_layer": "candidate_source_layer_id",
    }
    rope_keys = ("original_seq_len", "rope_factor", "beta_fast", "beta_slow")
    values = {names.get(k, k): list(v) if isinstance(v, tuple) else v
              for k, v in CFG.items() if k not in rope_keys}
    values.update(
        rope_scaling={"rope_type": "yarn", "factor": CFG["rope_factor"],
                      "original_max_position_embeddings": CFG["original_seq_len"],
                      "beta_fast": CFG["beta_fast"], "beta_slow": CFG["beta_slow"]},
        engram_source="ram", engram_ram_chunks=2, engram_pad_token_id=2,
        num_nextn_predict_layers=0, max_position_embeddings=MAXSEQ,
    )
    return TextConfig(**values)


def random_model():
    """Exercise nonzero HC/gates/sinks and real, packed mxfp8 Engram RAM rows."""
    mx.random.seed(7)
    model = LanguageModel(tiny_config())
    model.set_token_map(np.arange(VOCAB))
    rng = np.random.default_rng(7)
    parameters = []
    for key, value in tree_flatten(model.parameters()):
        if ".engram.embed." in key:
            continue
        shape = value.shape
        scale = shape[-1] ** -0.5 if len(shape) > 1 else 0.2
        arr = (rng.standard_normal(shape) * scale).astype(np.float32)
        if "norm.weight" in key or key.endswith(("q_weight", "k_weight")):
            arr = 1.0 + 0.1 * arr
        elif key.endswith(".scale"):
            arr = 0.5 + arr
        parameters.append((key, mx.array(arr)))
    model.load_weights(parameters, strict=False)
    for layer in model.layers:
        if layer.engram is None:
            continue
        embed = layer.engram.embed
        for i, weight in enumerate(embed.weight):
            values = mx.array(rng.normal(0, 0.5, (weight.shape[0], CFG["engram_head_dim"])).astype(np.float32))
            packed, scales = mx.quantize(values, group_size=32, bits=8, mode="mxfp8")
            embed.weight[i], embed.scales[i] = packed, scales
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("source", ["ram", "nvme", "hybrid"])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_target_checkpoint_plan_equals_constructed_parameters(source, mtp, vision):
    """Compare the actual quantized target tree, including a release-style MTP tail.

    This must not compare two planner-derived sets: the receiver constructs and
    quantizes a Model, whose complete parameter names are the load contract.
    """
    from mlx_vlm.models.deepseek_v41.config import ModelConfig, VisionConfig
    from mlx_vlm.models.deepseek_v41.deepseek_v41 import Model
    from mlx_vlm.models.deepseek_v41.checkpoint_layout import (
        _source_parameters, _dspark_source_parameters, _outputs,
        make_quantization_config_v41, plan_v41,
    )
    from mlx_vlm.models.deepseek_v41.vision_layout import expected_vision_parameters

    c = tiny_config()
    c.engram_source = source
    if mtp:
        c.num_nextn_predict_layers = 3
        c.compress_ratios += [0, 0, 0]
        c.dspark_target_layer_ids = [5, 6, 7]
        c.dspark_n_routed_experts = 8
    vc = VisionConfig(hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4) if vision else None
    model = Model(ModelConfig(text_config=c, vision_config=vc))
    metadata = {k: (d, s) for k, d, s in _source_parameters(c)}
    metadata.update({k: (d, s) for k, d, s in _dspark_source_parameters(c)})
    vp = expected_vision_parameters(vc, c.hidden_size) if vision else {}
    metadata.update(vp)
    plan = plan_v41(metadata, c, source)
    planned = {"language_model." + out["dst"]
               for entry in plan.values() for out in _outputs(entry)} | vp.keys()
    quantization = make_quantization_config_v41(model)
    nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                mode=quantization["mode"],
                class_predicate=lambda path, module: quantization.get(path, False))
    built = dict(tree_flatten(model.parameters()))
    assert planned == built.keys(), {
        "unbuilt": sorted(planned - built.keys()),
        "unplanned": sorted(built.keys() - planned),
    }
    assert any("dspark." in key for key in planned) == mtp
    # Exercise the wrapper's actual sanitize path and strict load, including
    # retained vision payloads and deliberately present raw MTP payloads.
    dtypes = {"BF16": mx.bfloat16, "F32": mx.float32, "I8": mx.int8,
              "F8_E4M3": mx.uint8, "F8_E8M0": mx.uint8}
    raw = {k: mx.full(s, 127 if d == "F8_E8M0" else 0, dtypes[d])
           for k, (d, s) in metadata.items()}
    sanitized = model.sanitize(raw)
    assert sanitized.keys() == built.keys()
    assert model.sanitize(sanitized).keys() == built.keys()
    model.load_weights(list(sanitized.items()), strict=True)


def token_ids():
    ids = np.random.default_rng(19).integers(3, VOCAB, size=(1, S + DECODE)).astype(np.int32)
    ids[0, [0, 5, 17]] = 2
    return mx.array(ids)


def assert_close(actual, expected, label, atol=1e-5, rtol=1e-5):
    mx.eval(actual, expected)
    diff = float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item())
    print(f"{label}: max abs diff={diff:.9g}")
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item()), f"{label}: max abs diff={diff:.9g}"


@pytest.fixture(params=["0", "1"], ids=["qat-off", "qat-on"])
def model_and_ids(request, monkeypatch):
    monkeypatch.setenv("DSV41_QAT", request.param)
    return random_model(), token_ids()


def test_prefill_decode_and_chunks(model_and_ids):
    model, ids = model_and_ids
    full = model(ids).logits
    assert full.shape == (1, S + DECODE, VOCAB)
    assert bool(mx.all(mx.isfinite(full)).item())
    cache = model.make_cache()
    pieces = [model(ids[:, :S], cache=cache).logits]
    for i in range(S, S + DECODE):
        pieces.append(model(ids[:, i:i + 1], cache=cache).logits)
    assert_close(mx.concatenate(pieces, axis=1), full, "prefill then decode")
    cache = model.make_cache()
    chunks = [model(ids[:, i:i + 5], cache=cache).logits for i in range(0, ids.shape[1], 5)]
    assert_close(mx.concatenate(chunks, axis=1), full, "chunks of five")
    assert cache[0].offset == S + DECODE


@pytest.mark.parametrize("trim", [1, 2, 9, 23, S + DECODE])
def test_trim_and_refeed(model_and_ids, trim):
    model, ids = model_and_ids
    cache = model.make_cache()
    full = model(ids, cache=cache).logits
    mx.eval(full, cache[0].state)
    assert cache[0].is_trimmable()
    assert cache[0].trim(trim) == trim
    kept = ids.shape[1] - trim
    assert cache[0].offset == kept
    for i, layer in enumerate(model.layers):
        lc = cache[0].layers[i]
        assert lc.win_kv.shape[1] == kept
        if layer.attn.is_kv_source:
            assert lc.kv.shape[1] == kept
            assert lc.comp_kv.shape[1] == kept // layer.attn.ratio
            assert lc.index_k.shape[1] == kept // layer.attn.ratio
    # Refill through odd boundaries, including a partial ratio-two group.
    pieces = [model(ids[:, i:i + 5], cache=cache).logits
              for i in range(kept, ids.shape[1], 5)]
    assert_close(mx.concatenate(pieces, axis=1), full[:, kept:], f"trim({trim}) replay")


def test_cache_state_roundtrip_and_isolation(model_and_ids):
    model, ids = model_and_ids
    cache = model.make_cache()
    first = model(ids[:, :33], cache=cache).logits
    mx.eval(first, cache[0].state)
    restored = model.make_cache()
    restored[0].state = cache[0].state
    assert restored[0].offset == 33
    assert restored[0].nbytes > 0
    # A separate conversation must not change the original cache.
    model(ids[:, :7], cache=model.make_cache())
    expected = model(ids[:, 33:], cache=cache).logits
    actual = model(ids[:, 33:], cache=restored).logits
    assert_close(actual, expected, "restored continuation")
    restored[0].trim(24)
    actual = model(ids[:, 32:], cache=restored).logits
    assert_close(actual, model(ids).logits[:, 32:], "restored cache trim")


def test_batch_and_masks_fail_explicitly():
    model = random_model()
    with pytest.raises(NotImplementedError, match="batch one"):
        model(mx.zeros((2, 1), dtype=mx.int32))
    with pytest.raises(NotImplementedError, match="attention masks"):
        model(mx.zeros((1, 1), dtype=mx.int32), mask=mx.ones((1, 1), dtype=mx.bool_))
    with pytest.raises(ValueError, match="image_mask"):
        model(mx.zeros((1, 1), dtype=mx.int32), image_mask=mx.ones((1, 2), dtype=mx.bool_))


def test_capacity_growth_300_decode_tokens(model_and_ids, monkeypatch):
    """FP8/FP4 fake-quant is discontinuous: ~1e-6 prefill/decode matmul noise flips codes.
    A win_kv row-142 FP8 flip (-0.25 vs -0.28125, same amax) moves logits by O(1).
    This also occurs without growth; QAT uses 5e-3 before the first flip, like PN.
    """
    # Force real doublings in a short test; production defaults to 2048 rows.
    monkeypatch.setenv("DSV41_CACHE_STEP", "16")
    model, _ = model_and_ids
    ids = mx.array(np.random.default_rng(29).integers(3, VOCAB, (1, 301)).astype(np.int32))
    cache = model.make_cache()
    pieces = [model(ids[:, :1], cache=cache).logits]
    mx.eval(pieces[-1], cache[0].state)
    previous = cache[0].capacity
    doublings = 0
    growth = {}
    for pos in range(1, ids.shape[1]):
        pieces.append(model(ids[:, pos:pos + 1], cache=cache).logits)
        mx.eval(pieces[-1], cache[0].state)
        capacity = cache[0].capacity
        if capacity != previous:
            assert capacity == previous * 2
            doublings += 1
        previous = capacity
        assert cache[0].offset <= capacity
        for i, lc in enumerate(cache[0].layers):
            for name, buf in lc.buffers.items():
                if buf is None:
                    continue
                key = (i, name)
                old, count = growth.get(key, (buf.shape[1], 0))
                if old != buf.shape[1]:
                    assert buf.shape[1] == 2 * old
                    count += 1
                growth[key] = (buf.shape[1], count)
                assert getattr(lc, name).shape[1] == lc.lengths[name]
                assert lc.lengths[name] <= buf.shape[1]
    limit = int(np.ceil(np.log2(ids.shape[1] / 16)))
    assert 0 < doublings <= limit
    assert all(count <= limit for _, count in growth.values())
    actual = mx.concatenate(pieces, axis=1)
    if os.environ.get("DSV41_QAT") == "1":
        reference_cache = model.make_cache()
        expected = model(ids, cache=reference_cache).logits
        mx.eval(actual, expected, reference_cache[0].state)
        actual_np, expected_np = np.asarray(actual), np.asarray(expected)
        assert np.isfinite(actual_np).all() and np.isfinite(expected_np).all()
        agreement = np.mean(actual_np.argmax(axis=-1) == expected_np.argmax(axis=-1))
        assert agreement >= 0.98, f"300 decode steps: argmax agreement={agreement:.6f}"
        first_flip = ids.shape[1]
        for decoded, prefilled in zip(cache[0].layers, reference_cache[0].layers):
            # These are dequantized FP8 grid values, so exact inequality detects a code flip.
            changed = np.any(np.asarray(decoded.win_kv) != np.asarray(prefilled.win_kv),
                             axis=(0, 2))
            positions = np.flatnonzero(changed)
            if positions.size:
                first_flip = min(first_flip, int(positions[0]))
        error = np.max(np.abs(actual_np - expected_np), axis=-1)
        scale = np.max(np.abs(expected_np), axis=-1)
        assert first_flip > 0, "FP8 code flip at position zero leaves no pre-flip parity coverage"
        assert np.all(error[:, :first_flip] <= 5e-3 * scale[:, :first_flip]), (
            f"300 decode steps: per-position relative error exceeds 5e-3 before flip {first_flip}")
    else:
        assert_close(actual, model(ids).logits, "300 decode steps")
    # Snapshot and trim must remain independent across growth and overwrites.
    snapshot = cache[0].state
    restored = model.make_cache()
    restored[0].state = snapshot
    capacity = cache[0].capacity
    cache[0].trim(150)
    assert cache[0].capacity == capacity
    replay = model(ids[:, 151:], cache=cache).logits
    assert_close(replay, model(ids).logits[:, 151:], "trim across growth boundaries")
    assert restored[0].offset == 301
    assert_close(restored[0].layers[0].win_kv, snapshot[1], "restored storage isolation")


def test_output_flags(model_and_ids):
    model, ids = model_and_ids
    ids = ids[:, :7]
    normal = model(ids)
    assert normal.hidden_states is None
    assert normal.shared_kv_states is None
    captured = model(ids, skip_logits=True, return_hidden=True)
    assert captured.logits is None
    assert len(captured.hidden_states) == 1
    hidden = captured.hidden_states[0]
    assert hidden.shape == (1, 7, model.config.hidden_size)
    assert_close(model.lm_head(model.model.norm(hidden).astype(mx.float32)),
                 normal.logits, "captured collapsed pre-norm hidden")
    sink = [mx.zeros_like(hidden)]
    unnormalized = model(ids, hidden_sink=sink, skip_final_norm=True)
    assert len(sink) == len(unnormalized.hidden_states) == 2
    assert_close(sink[-1], hidden, "external hidden sink")
    assert_close(unnormalized.logits, model.lm_head(hidden.astype(mx.float32)),
                 "skip final norm")
    skipped = model(ids, skip_logits=True, skip_final_norm=True)
    assert skipped.logits is None and skipped.hidden_states is None


def test_embeddings_without_token_ids():
    config = tiny_config()
    with_engram = LanguageModel(config)
    embeds = mx.zeros((1, 3, config.hidden_size))
    with pytest.raises(ValueError, match="Engram.*token ids.*inputs"):
        with_engram(inputs=None, inputs_embeds=embeds)
    config.engram_layer_ids = []
    config.engram_num_embeddings = []
    model = LanguageModel(config)
    ids = mx.array([[3, 4, 5]], dtype=mx.int32)
    embeds = model.model.embed_tokens(ids)
    assert_close(model(inputs_embeds=embeds).logits, model(ids).logits,
                 "embeddings without Engram")


def _load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def load_reference(root, monkeypatch, torch):
    """Use PN's independent CPU kernels and its three declared reference patches."""
    ref_dir = root / "docs" / "reference"
    kernel = _load_module("kernel", root / "tests" / "kernel_stub.py", monkeypatch)
    kernel.DISABLE_FAKE_QUANT = True
    engram = _load_module("engram", ref_dir / "engram.py", monkeypatch)
    monkeypatch.setattr(engram, "build_compressed_token_map", lambda _: (list(range(VOCAB)), VOCAB))
    _load_module("image_processor", ref_dir / "image_processor.py", monkeypatch)
    _load_module("vision", ref_dir / "vision.py", monkeypatch)
    ref = _load_module("_dsv41_cpu_reference", ref_dir / "model.py", monkeypatch)

    def embedding_fp32(self, indices):
        vals = torch.nn.functional.embedding(indices, self.weight).float()
        scales = torch.nn.functional.embedding(indices, self.scale).float()
        return (vals.unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)).flatten(-2)

    monkeypatch.setattr(ref.ParallelEngramEmbedding, "forward", embedding_fp32)
    head_forward = ref.ParallelHead.forward
    monkeypatch.setattr(ref.ParallelHead, "forward",
                        lambda self, x, full_logits=False: head_forward(self, x, True))
    index_forward = ref.Indexer.forward

    def own_keys(self, x, qr, latent, start_pos, offset):
        if self.owns_k:
            ref.shared_attn.index_k = self.k_cache
        return index_forward(self, x, qr, latent, start_pos, offset)

    monkeypatch.setattr(ref.Indexer, "forward", own_keys)
    return ref


def test_torch_reference_parity(monkeypatch):
    torch = pytest.importorskip("torch")
    root = Path(os.environ.get("DSV41_PN_DIR", "./tmp/upstream_dsv41/pipenetwork"))
    if not root.is_dir():
        pytest.skip("Set DSV41_PN_DIR to the PipeNetwork checkout for reference parity")
    monkeypatch.setenv("DSV41_QAT", "0")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        ref = load_reference(root, monkeypatch, torch)
        args = ref.ModelArgs(max_batch_size=1, max_seq_len=MAXSEQ, dtype="bf16",
                             expert_dtype=None, n_mtp_layers=0, vision_n_layers=0,
                             engram_pad_id=2, **CFG)
        with torch.no_grad():
            reference = ref.Transformer(args, tokenizer=None).float()
        model = LanguageModel(tiny_config())
        model.set_token_map(np.arange(VOCAB))
        rng = np.random.default_rng(7)
        transplant(reference, model, rng, torch)
        # Consume both original batches from upstream's RNG, then use its first.
        ids = rng.integers(3, VOCAB, size=(2, S + DECODE)).astype(np.int64)[:1]
        ids[0, [0, 5]] = 2
        for length in (7, 33, 48, 56):
            with torch.no_grad():
                _, logits, _ = reference(torch.from_numpy(ids[:, :length]), 0)
            expected = mx.array(logits.float().numpy())
            actual = model(mx.array(ids[:, :length].astype(np.int32))).logits
            # Upstream measures max error relative to the largest reference logit.
            scale = max(float(mx.max(mx.abs(expected)).item()), 1e-9)
            assert_close(actual, expected, f"Torch prefill {length}", atol=1e-4 * scale, rtol=0)
        cache = model.make_cache()
        model(mx.array(ids[:, :S].astype(np.int32)), cache=cache)
        with torch.no_grad():
            reference(torch.from_numpy(ids[:, :S]), 0)
            for pos in range(S, S + DECODE):
                _, logits, _ = reference(torch.from_numpy(ids[:, pos:pos + 1]), pos)
                expected = mx.array(logits.float().numpy())
                actual = model(mx.array(ids[:, pos:pos + 1].astype(np.int32)), cache=cache).logits
                scale = max(float(mx.max(mx.abs(expected)).item()), 1e-9)
                assert_close(actual, expected, f"Torch decode {pos}", atol=1e-4 * scale, rtol=0)
    finally:
        torch.set_default_dtype(old_dtype)


