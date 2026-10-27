"""Mac/Metal pad16 parity and timing: 2048 tokens, top-6, Zipf arm-A routing.

DSV41_TEST_FFN_DIM/INTER/EXPERTS default to 256/256/384; 5120/2304/384
uses checkpoint-sized projections and substantial unified memory. Timings include
host dispatch and completion; layout timing includes the dynamic R_pad read.
"""

import heapq
import math
import os
import random
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx_lm.models.switch_layers import _gather_sort

from mlx_vlm.models.deepseek_v41 import language as lang
from mlx_vlm.models.deepseek_v41.config import TextConfig


def _zipf_routing(experts):
    # Exactly factorial probe arm A: Gumbel top-6 without replacement.
    rng = random.Random(20260916)
    logits = [-1.1 * math.log(e + 1) for e in range(experts)]
    rows = []
    for _ in range(2048):
        scores = [v - math.log(-math.log(max(rng.random(), 1e-300))) for v in logits]
        rows.append(heapq.nlargest(6, range(experts), key=scores.__getitem__))
    return np.array(rows, dtype=np.uint32)


@pytest.fixture(scope="module")
def routed_fixture():
    if not mx.metal.is_available():
        pytest.skip("Requires Mac Metal")
    dim = int(os.environ.get("DSV41_TEST_FFN_DIM", "256"))
    inter = int(os.environ.get("DSV41_TEST_FFN_INTER", "256"))
    experts = int(os.environ.get("DSV41_TEST_FFN_EXPERTS", "384"))
    assert experts >= 6
    routes = _zipf_routing(experts)
    assert all(len(set(row)) == 6 for row in routes)
    counts = np.bincount(routes.ravel(), minlength=experts)
    r_pad = int((((counts + 15) // 16) * 16).sum())
    if experts == 384:
        assert 14500 <= r_pad <= 16000  # Skewed ~15k, not 18,048 worst case.
    mx.random.seed(41)
    args = TextConfig(hidden_size=dim, moe_intermediate_size=inter,
                      n_routed_experts=experts, num_experts_per_tok=6)
    moe = lang.MoE(args)
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(moe.switch_mlp, name)
        layer.weight = (mx.random.normal(layer.weight.shape) * layer.input_dims ** -.5).astype(mx.bfloat16)
        setattr(moe.switch_mlp, name, layer.to_quantized(group_size=32, bits=4, mode="mxfp4"))
    indices = mx.array(routes)

    class FixedRouting(nn.Module):
        def __call__(self, x, image_mask=None):
            return mx.full((x.shape[0], 6), 1 / 6, dtype=mx.float32), indices[:x.shape[0]]

    moe.gate = FixedRouting()
    moe.eval()
    x = mx.random.normal((2048, dim)).astype(mx.bfloat16)
    mx.eval(moe.parameters(), x, indices)
    return SimpleNamespace(moe=moe, mlp=moe.switch_mlp, x=x, indices=indices,
                           counts=counts, r_pad=r_pad, experts=experts, dim=dim, inter=inter)


@pytest.fixture(autouse=True)
def isolate_flags(monkeypatch):
    for key in ("DSV41_PREFILL_GROUPED_MXFP4", "DSV41_COMPILE_PRIMITIVES",
                "DSV41_PROMOTE_DECODE_WEIGHTS", "DSV41_GATE_FP32",
                "DSV41_FFN_DOWN_BF16", "DSV41_FFN_EXPERT_PAD16"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(lang, "_HEAD_BF16", False)
    monkeypatch.setattr(lang, "_PREFILL_MXFP8_STEEL", False)
    monkeypatch.setattr(lang, "_FFN_DOWN_BF16_ENGAGED", False)
    monkeypatch.setattr(lang, "_FFN_EXPERT_PAD16_ENGAGED", False)


def _assert_bits(actual, reference):
    mx.eval(actual, reference)
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    bits = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    np.testing.assert_array_equal(np.asarray(actual.view(bits)), np.asarray(reference.view(bits)))


def _measure(arms):
    samples = {name: [] for name in arms}
    for step in range(15):  # Five warm-up calls and ten timed calls per arm.
        order = list(arms)
        if step % 2:
            order.reverse()
        for name in order:
            mx.synchronize()
            start = perf_counter()
            result = arms[name]()  # Rebuild the graph every call.
            mx.eval(result)
            mx.synchronize()
            elapsed = (perf_counter() - start) * 1000
            if step >= 5:
                samples[name].append(elapsed)
    for name, values in samples.items():
        assert len(values) == 10
        print(f"{name}: median_ms={median(values):.4f} mean_ms={np.mean(values):.4f}")


@pytest.mark.parametrize("down_bf16", [False, True], ids=["fp32-down", "bf16-down"])
def test_routed_bit_equality_and_timing(routed_fixture, down_bf16, capsys):
    f = routed_fixture
    unpadded = (lambda: lang._ffn_down_bf16(f.mlp, f.x, f.indices)) if down_bf16 else (
        lambda: f.mlp(f.x, f.indices))
    padded = lambda: lang._ffn_expert_pad16(f.mlp, f.x, f.indices, down_bf16)
    reference, actual = unpadded(), padded()
    _assert_bits(actual, reference)
    assert actual.dtype == (mx.bfloat16 if down_bf16 else mx.float32)
    _assert_bits(padded(), reference)
    lines = capsys.readouterr().out.splitlines()
    engage = (f"[DSV41] FFN_EXPERT_PAD16 engaged R=12288 R_pad={f.r_pad} "
              f"occupied_experts={np.count_nonzero(f.counts)}")
    assert lines.count(engage) == 1
    assert len(lines) == (2 if down_bf16 else 1)
    print(f"n=2048 topk=6 E={f.experts} D={f.dim} I={f.inter} R=12288 R_pad={f.r_pad}; "
          f"down={'bf16' if down_bf16 else 'fp32'}; 5 warm + 10 timed, host+eval+sync ms")
    _measure({"whole_routed_pad_OFF": unpadded, "whole_routed_pad_ON": padded})
    # Exclude sorting from this arm; include counts..map/keep and R_pad.item().
    _, idx, _ = _gather_sort(mx.expand_dims(f.x, (-2, -3)), f.indices)
    mx.eval(idx)
    _measure({"layout_counts_to_map_keep_including_host_read":
              lambda: lang._ffn_expert_pad16_layout(idx, f.experts)})
    output = capsys.readouterr().out
    assert "engaged" not in output
    print(output, end="")


@pytest.mark.parametrize("down_bf16", [False, True], ids=["fp32-down", "bf16-down"])
def test_attach_dispatch_and_engage_once(routed_fixture, monkeypatch, capsys, down_bf16):
    f = routed_fixture
    moe = f.moe
    monkeypatch.setattr(moe, "ffn_expert_pad16", False)
    monkeypatch.setattr(moe, "ffn_down_bf16", down_bf16)
    x = f.x[None]
    reference = moe(x)
    mx.eval(reference)
    monkeypatch.setenv("DSV41_FFN_EXPERT_PAD16", "1")
    monkeypatch.setenv("DSV41_FFN_DOWN_BF16", "1" if down_bf16 else "0")
    holder = SimpleNamespace(layers=[SimpleNamespace(ffn=moe, engram=None)],
                             model=SimpleNamespace(layout=None))
    lang.LanguageModel.attach_engram_storage(holder, ".")
    monkeypatch.setenv("DSV41_FFN_EXPERT_PAD16", "0")
    monkeypatch.setenv("DSV41_FFN_DOWN_BF16", "0" if down_bf16 else "1")
    lang.LanguageModel.attach_engram_storage(holder, ".")
    assert moe.ffn_expert_pad16 and moe.ffn_down_bf16 == down_bf16
    # The pad helper must win over grouped prefill, with the down dtype composed.
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MXFP4", "1")
    pad = Mock(wraps=lang._ffn_expert_pad16)
    monkeypatch.setattr(lang, "_ffn_expert_pad16", pad)
    _assert_bits(moe(x), reference)
    _assert_bits(moe(x), reference)
    assert pad.call_count == 2
    assert all(call.args[3] == down_bf16 for call in pad.call_args_list)
    assert capsys.readouterr().out.count("FFN_EXPERT_PAD16 engaged") == 1
    monkeypatch.setenv("DSV41_PREFILL_GROUPED_MXFP4", "0")
    for n in (1, 63, 64):
        for verify in (False, True):
            pad.reset_mock()
            token = lang._v41_verify_context.set(verify)
            try:
                mx.eval(moe(x[:, :n]))
            finally:
                lang._v41_verify_context.reset(token)
            assert pad.call_count == int(n >= 64 and not verify)
    assert "FFN_EXPERT_PAD16 engaged" not in capsys.readouterr().out
