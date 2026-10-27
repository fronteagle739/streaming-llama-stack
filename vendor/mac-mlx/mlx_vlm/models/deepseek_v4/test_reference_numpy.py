"""Validate the numpy DSV4/DSpark oracle with hand-computed cases and
formulation-equivalence proofs. numpy-only: runs on any box (including the
Windows dev box, no mlx required).

  pytest mlx_vlm/models/deepseek_v4/test_reference_numpy.py     (Mac)
  python mlx_vlm/models/deepseek_v4/test_reference_numpy.py     (any box w/ numpy)
"""

import math

import numpy as np

try:
    from mlx_vlm.models.deepseek_v4 import reference_dsv4 as ref
except ImportError:  # no mlx on this box: import the (mlx-free) module by path
    import importlib.util
    import os

    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_dsv4.py")
    _spec = importlib.util.spec_from_file_location("_dsv4_reference", _p)
    ref = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ref)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_sqrtsoftplus_values():
    # softplus(0) = ln 2; large x -> x; very negative -> ~0
    s = ref.score_activation(np.array([0.0, 20.0, -20.0]), "sqrtsoftplus")
    assert abs(s[0] - math.sqrt(math.log(2.0))) < 1e-12
    assert abs(s[1] - math.sqrt(20.0)) < 1e-6
    assert s[2] < 1e-4


def test_noaux_tc_bias_affects_selection_not_weights():
    logits = np.array([[2.0, 1.0, 0.5, -1.0]])
    bias = np.array([0.0, 0.0, 5.0, 0.0])  # forces expert 2 into the top-2
    inds, weights = ref.noaux_tc_select(
        logits, bias, top_k=2, routed_scaling_factor=1.5, norm_topk_prob=True
    )
    assert set(inds[0].tolist()) == {0, 2}, "bias must steer selection"
    scores = ref.score_activation(logits, "sqrtsoftplus")[0]
    expect = np.array([scores[0], scores[2]])
    expect = expect / expect.sum() * 1.5  # weights from UNBIASED scores
    got = weights[0][np.argsort(inds[0])]
    np.testing.assert_allclose(np.sort(got), np.sort(expect), rtol=1e-12)
    assert abs(weights.sum() - 1.5) < 1e-12, "norm_topk_prob -> weights sum to scale"


def test_hash_select_uses_table_indices():
    logits = RNG.normal(size=(3, 8))
    tid2eid = np.array([[1, 5], [0, 7], [3, 4], [2, 6]])
    ids = np.array([2, 0, 3])
    inds, weights = ref.hash_select(logits, tid2eid, ids, 1.5, True)
    np.testing.assert_array_equal(inds, tid2eid[ids])
    scores = ref.score_activation(logits, "sqrtsoftplus")
    for t in range(3):
        w = scores[t, inds[t]]
        w = w / w.sum() * 1.5
        np.testing.assert_allclose(weights[t], w, rtol=1e-12)


# ---------------------------------------------------------------------------
# Limited SwiGLU
# ---------------------------------------------------------------------------

def test_limited_swiglu_clamps():
    limit = 10.0
    gate = np.array([-30.0, 0.0, 30.0])
    up = np.array([-30.0, 2.0, 30.0])
    y = ref.limited_swiglu(gate, up, limit)
    # gate: clamped from ABOVE only; up: clamped both sides
    silu = lambda v: v / (1 + math.exp(-v))
    np.testing.assert_allclose(
        y, [silu(-30.0) * -10.0, silu(0.0) * 2.0, silu(10.0) * 10.0], rtol=1e-9
    )
    # limit=0 disables clamping
    y0 = ref.limited_swiglu(gate, up, 0.0)
    np.testing.assert_allclose(y0[2], silu(30.0) * 30.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------

def test_sinkhorn_doubly_stochastic():
    hc, iters, eps = 4, 20, 1e-6
    mixes = RNG.normal(size=(5, (2 + hc) * hc))
    scale = np.array([1.0, 1.0, 1.0])
    base = RNG.normal(size=((2 + hc) * hc,))
    pre, post, comb = ref.hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
    assert pre.shape == (5, hc) and post.shape == (5, hc) and comb.shape == (5, hc, hc)
    assert np.all(pre > 0) and np.all(pre < 1 + 2 * eps)
    assert np.all(post > 0) and np.all(post < 2)
    # after the final column normalization, columns sum to ~1;
    # rows are near 1 (doubly-stochastic fixed point of sinkhorn)
    np.testing.assert_allclose(comb.sum(-2), 1.0, atol=1e-4)
    np.testing.assert_allclose(comb.sum(-1), 1.0, atol=1e-3)


def test_hc_pre_post_algebra():
    hc, d = 4, 8
    x = RNG.normal(size=(3, hc, d))
    fn = RNG.normal(size=((2 + hc) * hc, hc * d)) * 0.1
    scale = np.ones(3)
    base = np.zeros((2 + hc) * hc)
    y, post, comb = ref.hc_pre(x, fn, scale, base, hc, 20, 1e-6, 1e-6)
    assert y.shape == (3, d)
    # hc_post: y_out[j] = post[j]*x_new + sum_i comb[i,j]*residual[i]
    x_new = RNG.normal(size=(3, d))
    out = ref.hc_post(x_new, x, post, comb)
    j = 2
    manual = post[:, j, None] * x_new + np.einsum("ti,tid->td", comb[:, :, j], x)
    np.testing.assert_allclose(out[:, j], manual, rtol=1e-10)


def test_hc_head_collapse_is_sigmoid_weighted_sum():
    hc, d = 4, 8
    x = RNG.normal(size=(2, hc, d))
    fn = np.zeros((hc, hc * d))
    base = np.array([10.0, 10.0, -10.0, -10.0])  # lanes 0,1 ~1; lanes 2,3 ~0
    y = ref.hc_head_collapse(x, fn, np.ones(1), base, 0.0, 1e-6)
    np.testing.assert_allclose(y, x[:, 0] + x[:, 1], atol=1e-3)


def test_mean_lane_capture():
    h = RNG.normal(size=(2, 5, 4, 8))
    np.testing.assert_allclose(ref.mean_lane_capture(h), h.mean(2), rtol=1e-12)


# ---------------------------------------------------------------------------
# Compressors
# ---------------------------------------------------------------------------

def test_compress_simple_hand_case():
    ratio, d = 2, 3
    kv = np.arange(4 * d, dtype=np.float64).reshape(4, d)
    gate = np.zeros((4, d))
    ape = np.zeros((ratio, d))
    out = ref.compress_prefill_simple(kv, gate, ape, ratio)
    # uniform gates -> plain window means
    np.testing.assert_allclose(out, [(kv[0] + kv[1]) / 2, (kv[2] + kv[3]) / 2])
    # one-hot-ish gate picks the dominant row
    gate2 = np.zeros((4, d))
    gate2[1] = 50.0
    out2 = ref.compress_prefill_simple(kv, gate2, ape, ratio)
    np.testing.assert_allclose(out2[0], kv[1], atol=1e-6)


def test_compress_overlap_formulations_equivalent():
    """The mlx `_overlap_compress_kv` shifted-split formulation equals the
    official inference/model.py `overlap_transform` formulation."""
    ratio, d = 4, 6
    for T in (4, 8, 16):
        kv = RNG.normal(size=(T, 2 * d))
        gate = RNG.normal(size=(T, 2 * d))
        ape = RNG.normal(size=(ratio, 2 * d))
        a = ref.compress_prefill_overlap(kv, gate, ape, ratio)
        b = ref.compress_prefill_overlap_torch_style(kv, gate, ape, ratio)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_compress_overlap_first_window_uses_only_current():
    """Window 0 has no previous window: its pooled output must depend only on
    the second-half (current-window) dims."""
    ratio, d = 4, 5
    kv = RNG.normal(size=(4, 2 * d))
    gate = RNG.normal(size=(4, 2 * d))
    ape = np.zeros((ratio, 2 * d))
    out1 = ref.compress_prefill_overlap(kv, gate, ape, ratio)
    kv2 = kv.copy()
    kv2[:, :d] = 999.0  # first-half dims of window 0 must be ignored
    out2 = ref.compress_prefill_overlap(kv2, gate, ape, ratio)
    np.testing.assert_allclose(out1, out2, rtol=1e-12)


# ---------------------------------------------------------------------------
# YaRN
# ---------------------------------------------------------------------------

def test_yarn_endpoints():
    import math

    # Flash params: factor 16, original_max_position_embeddings 65536
    dims, base, factor, orig = 64, 10000.0, 16.0, 65536
    inv = ref.yarn_inv_freqs(dims, base, factor, orig)
    plain = 1.0 / (base ** (np.arange(0, dims, 2) / dims))
    low = math.floor(dims * math.log(orig / (32 * 2 * math.pi)) / (2 * math.log(base)))
    # dims below the beta_fast correction boundary: smooth = 1 -> unchanged
    np.testing.assert_allclose(inv[: max(low, 1)], plain[: max(low, 1)], rtol=1e-12)
    # interpolation ratio stays within [1/factor, 1] and is monotone
    r = inv / plain
    assert np.all(r <= 1.0 + 1e-12) and np.all(r >= 1.0 / factor - 1e-12)
    assert np.all(np.diff(r) <= 1e-12)

    # a parameterization whose beta_slow boundary falls INSIDE dims//2:
    # dims past it must be fully interpolated (divided by factor)
    inv2 = ref.yarn_inv_freqs(dims, base, factor, 10000)
    plain2 = plain
    high2 = math.ceil(dims * math.log(10000 / (2 * math.pi)) / (2 * math.log(base)))
    assert high2 < dims // 2, "test parameterization must exercise the slow end"
    np.testing.assert_allclose(inv2[high2 + 1 :], plain2[high2 + 1 :] / factor, rtol=1e-12)


# ---------------------------------------------------------------------------
# DSpark heads
# ---------------------------------------------------------------------------

def test_markov_bias_and_block_sampling():
    V, r = 12, 4
    w1 = RNG.normal(size=(V, r))
    w2 = RNG.normal(size=(V, r))
    b = ref.markov_bias(w1, w2, np.array(3))
    np.testing.assert_allclose(b, w1[3] @ w2.T, rtol=1e-12)

    base = np.zeros((3, V))
    anchor = 5
    toks, corrected = ref.markov_sample_block(base, anchor, w1, w2)
    # with zero base logits the chain is fully determined by the bias
    t0 = int((w1[anchor] @ w2.T).argmax())
    t1 = int((w1[t0] @ w2.T).argmax())
    t2 = int((w1[t1] @ w2.T).argmax())
    assert toks == [t0, t1, t2]
    np.testing.assert_allclose(corrected[1], w1[t0] @ w2.T, rtol=1e-12)


def test_confidence_and_truncation():
    d, r, block = 6, 3, 5
    hidden = RNG.normal(size=(block, d))
    w1 = RNG.normal(size=(9, r))
    proj = RNG.normal(size=(1, d + r))
    prev = np.array([1, 2, 3, 4, 5])
    c = ref.confidence_scores(hidden, prev, w1, proj)
    assert c.shape == (block,) and np.all((c > 0) & (c < 1))
    manual0 = 1 / (1 + np.exp(-(np.concatenate([hidden[0], w1[1]]) @ proj.reshape(-1))))
    np.testing.assert_allclose(c[0], manual0, rtol=1e-10)

    assert ref.confident_prefix_length(np.array([0.9, 0.8, 0.3, 0.9]), 0.5) == 2
    assert ref.confident_prefix_length(np.array([0.9, 0.8]), 0.5) == 2
    assert ref.confident_prefix_length(np.array([0.2, 0.9]), 0.5) == 0
    assert ref.confident_prefix_length(np.array([0.2, 0.9]), 0.0) == 2
    np.testing.assert_allclose(
        ref.cumulative_survival(np.array([0.5, 0.5, 1.0])), [0.5, 0.25, 0.25]
    )


def test_draft_block_layout():
    ids = ref.dspark_draft_input_ids(anchor_id=42, noise_token_id=128799, block_size=5)
    assert ids == [42, 128799, 128799, 128799, 128799]
    assert ref.dspark_draft_positions(ctx_len=100, block_size=5) == [100, 101, 102, 103, 104]


def test_speculative_walk():
    assert ref.speculative_walk_greedy([1, 2, 3], [1, 2, 3, 9]) == (3, [1, 2, 3, 9])
    assert ref.speculative_walk_greedy([1, 5, 3], [1, 2, 3, 9]) == (1, [1, 2])
    assert ref.speculative_walk_greedy([7], [1]) == (0, [1])


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"REFERENCE_NUMPY_TEST ALL PASS ({len(fns)} tests)")


if __name__ == "__main__":
    main()
