"""CPU reference tests, loaded by path so pytest never needs MLX.

The long split regression deliberately records the ORIGINAL Indexer pointer bug.
It is not repaired or concealed by loosening the allclose tolerance.
"""
import importlib.util
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dsv41_cpu_oracle", HERE / "reference_cpu_dsv41.py")
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)
GOLDEN = HERE / "reference_cpu_dsv41_fingerprint.json"


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tokenizer():
    if oracle.reference is None or not all(
        (oracle.TOKENIZER_DIR / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        pytest.skip("CPU reference inputs absent; set DSV41_REF_DIR and DSV41_TOKENIZER_DIR")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(oracle.TOKENIZER_DIR, local_files_only=True)


@pytest.fixture(scope="module")
def model(tokenizer):
    return oracle.build_reference(oracle.tiny_args(tokenizer), tokenizer)


def assert_logits_close(actual, expected):
    diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), f"max abs diff = {diff:.9g}"


def test_cpu_forward_and_original_engram(model, tokenizer):
    # Batch two also exercises per-batch sparse gathers and n-gram state.
    ids = torch.tensor(oracle.FINGERPRINT_IDS).repeat(2, 1)
    ids[1] = ids[1].flip(0)
    logits, tokens = oracle.forward_logits(model, ids, 0)
    assert logits.shape == (2, 16, len(tokenizer))
    assert logits.dtype == torch.float32 and logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    assert torch.equal(tokens, logits.argmax(-1))
    assert all(p.device.type == "cpu" for p in model.parameters())
    for layer in (1, 4):
        embed = model.layers[layer].engram.embed
        assert type(embed) is oracle.reference.ParallelEngramEmbedding
        assert embed.weight.dtype == torch.float8_e4m3fn
        assert embed.scale.dtype == torch.float8_e8m0fnu
        assert set(embed.scale.view(torch.uint8).unique().tolist()) == {126, 127, 128}


@pytest.mark.parametrize("cut", [8])
def test_split_all_positions(model, cut):
    # Only complete ratio-2 groups (8 tokens) are portable for this 12-token case.
    ids = torch.tensor(oracle.FINGERPRINT_IDS)[:, :12]
    whole, tokens = oracle.forward_logits(model, ids, 0)
    split, split_tokens = oracle.run_split(model, ids, cut)
    assert_logits_close(split, whole)
    assert torch.equal(split_tokens, tokens)


def test_split_group_of_eight_boundary_effect(model):
    # Incomplete groups share a split error; its magnitude depends on torch/OS.
    ids = torch.tensor(oracle.FINGERPRINT_IDS)[:, :12]
    whole, _ = oracle.forward_logits(model, ids, 0)
    diffs = [
        (oracle.run_split(model, ids, cut)[0] - whole).abs().max().item()
        for cut in range(1, 7)
    ]
    np.testing.assert_allclose(diffs, [diffs[0]] * 6, rtol=1e-5, atol=1e-6)


@pytest.fixture
def recorded_fingerprint():
    expected = json.loads(GOLDEN.read_text())
    # The allowlisted files exclude this legacy golden, recorded on Windows.
    # Newly generated fingerprints carry platform.system() explicitly.
    expected.setdefault("platform", "Windows")
    if (torch.__version__, platform.system()) != (expected["torch_version"], expected["platform"]):
        pytest.skip(
            f"Fingerprint requires torch {expected['torch_version']} on {expected['platform']}; "
            f"running torch {torch.__version__} on {platform.system()}"
        )
    return expected


def test_long_split_reference_index_pointer_quirk(recorded_fingerprint, model):
    ids = torch.tensor(oracle.FINGERPRINT_IDS)
    whole, _ = oracle.forward_logits(model, ids, 0)
    idx = model.layers[1].attn.indexer
    records = []

    def capture(module, inputs, output):
        # model.py:537-548 only publishes self.k_cache if latent is not None.
        # On even zero-based start_pos (odd token count), ratio-2 has no
        # latent and reads the ratio-1 layer-3 pointer from the previous call.
        latent, pos = inputs[2], inputs[3]
        records.append((pos, latent is None,
                        oracle.reference.shared_attn.index_k is idx.k_cache,
                        oracle.reference.shared_attn.index_k is model.layers[3].attn.indexer.k_cache))

    hook = idx.register_forward_hook(capture)
    try:
        split, _ = oracle.run_split(model, ids, 8)
    finally:
        hook.remove()
    assert records[0] == (0, False, True, False)
    for pos, absent, own, stale in records[1:]:
        assert (absent, own, stale) == (pos % 2 == 0, pos % 2 == 1, pos % 2 == 0)
    diff = (split - whole).abs().amax(-1)[0]
    assert_logits_close(split[:, :12], whole[:, :12])
    # This is an explicit known-divergence regression, NOT an allclose pass.
    assert not torch.allclose(split, whole, rtol=1e-5, atol=1e-6)
    expected = recorded_fingerprint["known_split_quirk"]
    np.testing.assert_allclose(diff.numpy(), expected["per_position_max_abs_diff"],
                               rtol=1e-4, atol=1e-6,
                               err_msg=f"reference split max abs diff = {diff.max().item():.9g}")
    # A final single decode completing a group uses the correct key owner.
    split_last, _ = oracle.run_split(model, ids, 15)
    assert_logits_close(split_last, whole)


def test_fixed_seed_fingerprint(recorded_fingerprint, model, tokenizer):
    expected = recorded_fingerprint
    actual = oracle.make_fingerprint(model)
    for key in ("args", "seed", "source_sha256", "tokenizer_sha256", "input_ids",
                "shape", "probe_indices", "output_token_ids"):
        assert actual[key] == expected[key], key
    for key in ("sum_float64", "sum_squares_float64", "per_position_sum_float64", "probe_values"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6, err_msg=key)
    # Rebuild proves random initialization, rather than reusing the same weights.
    other = oracle.build_reference(oracle.tiny_args(tokenizer), tokenizer)
    again, _ = oracle.forward_logits(other, oracle.FINGERPRINT_IDS, 0)
    original, _ = oracle.forward_logits(model, oracle.FINGERPRINT_IDS, 0)
    assert torch.equal(again, original)


def test_quantization_identity_and_gemms():
    x = torch.arange(128, dtype=torch.bfloat16).view(2, 64)[:, ::2]
    original = x.clone()
    for quant in (oracle.act_quant, oracle.fp4_act_quant):
        assert quant(x, 32, inplace=True) is x
        y, scales = quant(x, 32)
        assert y is x and torch.equal(x, original)
        assert torch.equal(scales.float(), torch.ones(2, 1))
    # Two K blocks and two N blocks distinguish all weight/activation scales.
    a = torch.ones(2, 64, dtype=torch.bfloat16)
    b = torch.ones(33, 64, dtype=torch.float8_e4m3fn)
    a_s = torch.tensor([[1., 2.], [4., 8.]])
    b_s = torch.tensor([[2., 3.], [5., 7.]])
    y = oracle.fp8_gemm(a, a_s, b, b_s, block_size=32)
    assert torch.equal(y[:, 0], torch.tensor([256., 1024.]))
    assert torch.equal(y[:, 32], torch.tensor([608., 2432.]))
    # Packed nibbles 2=+1, A=-1: alternate +1/-1. Scales differ by K block.
    packed = torch.full((1, 32), 0xA2, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    a[:, 1::2] = 0
    y = oracle.fp4_gemm(a, a_s, packed, torch.tensor([[2., 3.]]), act_block_size=32)
    assert torch.equal(y[:, 0], torch.tensor([128., 512.]))


@pytest.mark.parametrize("iters", [1, 3, 20])
def test_sinkhorn_order_and_epsilon(iters):
    # Independent NumPy scalar loops with a deliberately visible epsilon.
    mixes = torch.arange(8, dtype=torch.float32).view(1, 1, 8) / 3
    scale = torch.tensor([.3, -.4, .7])
    base = torch.arange(8, dtype=torch.float32) / 7
    pre, post, comb = oracle.hc_split_sinkhorn(mixes, scale, base, 2, iters, .01)
    m, a, b = mixes.numpy().ravel(), scale.numpy(), base.numpy()
    np.testing.assert_allclose(pre.numpy().ravel(), 1 / (1 + np.exp(-(m[:2]*a[0]+b[:2]))) + .01, rtol=1e-6)
    np.testing.assert_allclose(post.numpy().ravel(), 2 / (1 + np.exp(-(m[2:4]*a[1]+b[2:4]))), rtol=1e-6)
    c = (m[4:]*a[2]+b[4:]).reshape(2, 2).copy()
    for row in c:
        row[:] = np.exp(row - max(row))
        row[:] = row / sum(row) + .01
    for col in range(2):
        c[:, col] /= sum(c[:, col]) + .01
    for _ in range(iters - 1):
        for row in c:
            row /= sum(row) + .01
        for col in range(2):
            c[:, col] /= sum(c[:, col]) + .01
    np.testing.assert_allclose(comb.numpy()[0, 0], c, rtol=1e-6, atol=1e-7)


def test_sparse_attention_sink_absence_duplicates_and_bf16_rounding():
    q = torch.tensor([[[[1., 0.]], [[0., 1.]]]], dtype=torch.bfloat16)
    kv = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.bfloat16)
    sink = torch.tensor([.3])
    idx = torch.tensor([[[0, -1, 1, 0], [-1, -1, -1, -1]]], dtype=torch.int32)
    y = oracle.sparse_attn(q, kv, sink, idx, .7)
    weights = torch.exp(torch.tensor([.7, 2.1, .7]) - 2.1)
    expected = (weights.bfloat16().float() @ kv.float()[0, [0, 1, 0]]) / (weights.sum() + torch.exp(sink - 2.1))
    assert torch.equal(y[0, 0, 0], expected.bfloat16())
    assert torch.equal(y[0, 1], torch.zeros_like(y[0, 1]))
    # Force an online-max change in block 2 and an entirely absent first block.
    idx = torch.tensor([[[0]*64 + [1], [-1]*64 + [1]]], dtype=torch.int32)
    y = oracle.sparse_attn(q, kv, sink, idx, .7)
    correction = torch.exp(torch.tensor(.7 - 2.1))
    expected = (64 * correction * kv.float()[0, 0] + kv.float()[0, 1]) / (64 * correction + 1 + torch.exp(sink - 2.1))
    assert torch.equal(y[0, 0, 0], expected.bfloat16())
    expected = kv.float()[0, 1] / (1 + torch.exp(sink - 2.8))
    assert torch.equal(y[0, 1, 0], expected.bfloat16())
