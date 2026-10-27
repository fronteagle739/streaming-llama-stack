"""Mac/Metal routed FFN precision trial: 2048 tokens, top-6, random MXFP4.

DSV41_TEST_FFN_DIM/INTER/EXPERTS default to 256/256/32 for a small fixture;
set 5120/2304/384 for checkpoint-sized projections (substantial unified memory).
Down timing excludes gate/up, activation casts, sorting and routing combine.
"""

import os
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx_lm.models.switch_layers import _gather_sort

from mlx_vlm.models.deepseek_v41 import language as lang
from mlx_vlm.models.deepseek_v41.config import TextConfig


def _errors(label, actual, reference):
    actual, reference = (np.asarray(v.astype(mx.float32)) for v in (actual, reference))
    delta = actual - reference
    max_abs = float(np.max(np.abs(delta)))
    rms_rel = float(np.sqrt(np.mean(delta ** 2) / max(np.mean(reference ** 2), 1e-30)))
    print(f"{label}: max_abs={max_abs:.8g} rms_rel={rms_rel:.8g}")
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    assert 0 < rms_rel < .02  # Synthetic smoke tolerance, not model-quality acceptance.


def test_routed_ffn_and_down_timing(monkeypatch, capsys):
    if not mx.metal.is_available():
        pytest.skip("Requires Mac Metal")
    for key in ("DSV41_PREFILL_GROUPED_MXFP4", "DSV41_COMPILE_PRIMITIVES",
                "DSV41_PROMOTE_DECODE_WEIGHTS", "DSV41_GATE_FP32",
                "DSV41_FFN_EXPERT_PAD16"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(lang, "_HEAD_BF16", False)
    monkeypatch.setattr(lang, "_PREFILL_MXFP8_STEEL", False)
    monkeypatch.setattr(lang, "_FFN_DOWN_BF16_ENGAGED", False)
    dim = int(os.environ.get("DSV41_TEST_FFN_DIM", "256"))
    inter = int(os.environ.get("DSV41_TEST_FFN_INTER", "256"))
    experts = int(os.environ.get("DSV41_TEST_FFN_EXPERTS", "32"))
    args = TextConfig(hidden_size=dim, moe_intermediate_size=inter,
                      n_routed_experts=experts, num_experts_per_tok=6)
    mx.random.seed(41)
    moe = lang.MoE(args)
    moe.eval()
    moe.gate.weight = mx.random.normal((experts, dim)) * dim ** -.5
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(moe.switch_mlp, name)
        layer.weight = (mx.random.normal(layer.weight.shape) * layer.input_dims ** -.5).astype(mx.bfloat16)
        setattr(moe.switch_mlp, name, layer.to_quantized(group_size=32, bits=4, mode="mxfp4"))
    moe.eval()
    x = mx.random.normal((1, 2048, dim)).astype(mx.bfloat16)
    mx.eval(moe.parameters(), x)
    reference = moe(x)
    mx.eval(reference)
    # Exercise the real attach method without needing an Engram checkpoint.
    monkeypatch.setenv("DSV41_FFN_DOWN_BF16", "1")
    holder = SimpleNamespace(layers=[SimpleNamespace(ffn=moe, engram=None)],
                             model=SimpleNamespace(layout=None))
    lang.LanguageModel.attach_engram_storage(holder, ".")
    monkeypatch.setenv("DSV41_FFN_DOWN_BF16", "0")  # Latched at attach.
    actual = moe(x)
    mx.eval(actual)
    # DSpark verification is excluded even at n=2048.
    with lang._v41_verification():
        verified = moe(x)
        mx.eval(verified)
    np.testing.assert_array_equal(np.asarray(verified.astype(mx.float32)),
                                  np.asarray(reference.astype(mx.float32)))
    for n in (1, 63):
        moe.ffn_down_bf16 = False
        small_ref = moe(x[:, :n])
        moe.ffn_down_bf16 = True
        small_got = moe(x[:, :n])
        mx.eval(small_ref, small_got)
        np.testing.assert_array_equal(np.asarray(small_ref.astype(mx.float32)),
                                      np.asarray(small_got.astype(mx.float32)))
    xf = x.reshape(2048, dim)
    weights, indices = moe.gate(xf)
    routed_ref = moe.switch_mlp(xf, indices)
    routed_got = lang._ffn_down_bf16(moe.switch_mlp, xf, indices)
    combine_ref = lang._v41_moe_combine(routed_ref, weights)
    combine_got = lang._v41_moe_combine(routed_got, weights)
    mx.eval(combine_ref, combine_got)
    assert combine_got.dtype == combine_ref.dtype == mx.float32
    assert capsys.readouterr().out == "[DSV41] FFN_DOWN_BF16 engaged n=2048\n"
    _errors("MoE incl shared/final BF16", actual, reference)
    _errors("routed FP32 combine", combine_got, combine_ref)

    mlp = moe.switch_mlp
    sx, ids, _ = _gather_sort(mx.expand_dims(xf, (-2, -3)), indices)
    hidden = mlp.activation(mlp.up_proj(sx, ids, sorted_indices=True),
                            mlp.gate_proj(sx, ids, sorted_indices=True))
    low = hidden.astype(mx.bfloat16)
    mx.eval(hidden, low, ids)
    assert hidden.dtype == mx.float32 and low.dtype == mx.bfloat16
    samples = {"fp32": [], "bf16": []}
    outputs = {}
    for step in range(15):  # Five warm-up calls per dtype; ten interleaved measurements.
        order = [("fp32", hidden), ("bf16", low)]
        if step % 2:
            order.reverse()
        for name, value in order:
            start = perf_counter()
            out = mlp.down_proj(value, ids, sorted_indices=True)
            mx.eval(out)
            elapsed = (perf_counter() - start) * 1000
            if step >= 5:
                samples[name].append(elapsed)
            outputs[name] = out
    _errors("down projection", outputs["bf16"], outputs["fp32"])
    assert outputs["bf16"].dtype == mx.bfloat16
    print(f"n=2048 topk=6 E={experts} D={dim} I={inter}; 5 warm + 10 timed, host+eval ms")
    for name, values in samples.items():
        print(f"down X={name}: median_ms={median(values):.4f} mean_ms={np.mean(values):.4f}")
