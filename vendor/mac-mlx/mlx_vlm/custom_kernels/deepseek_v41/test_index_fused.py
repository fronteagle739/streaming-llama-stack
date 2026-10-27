"""Mac acceptance gate (strict selected sets) and whole-Indexer 5+10 timing.

pytest -q -s mlx_vlm/custom_kernels/deepseek_v41/test_index_fused.py
python -m mlx_vlm.custom_kernels.deepseek_v41.test_index_fused --bench
No tolerance or alternative tied indices are accepted by the selection gate.
"""
import argparse
from contextlib import contextmanager
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Mac Metal tests; do not import MLX on Windows", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires Apple Metal", allow_module_level=True)

from mlx_vlm.custom_kernels.deepseek_v41 import index_fused as index_fused_module
from mlx_vlm.custom_kernels.deepseek_v41.index_fused import (
    index_scores_fused, prepare_index_fused, prewarm_index_fused,
    prewarm_index_skip,
)
from mlx_vlm.models.deepseek_v41 import language as lang

SCALE = 128 ** -0.5 * 32 ** -0.5
SETTINGS = dict(DSV41_INDEX_FUSED="0", DSV41_INDEX_SKIP="0", DSV41_PREFILL_TILED_INDEX="0",
    DSV41_PREFILL_V41_INDEX="0", DSV41_METAL_INDEXER="0", DSV41_S4_COMPILE_INDEXER="0",
    DSV41_QAT="1", DSV41_METAL_QUANT="0", DSV41_INDEX_TOPK="512",
    DSV41_INDEX_WORKSPACE_MB="512", DSV41_S4_PRECOMPUTE="1", DSV41_S4_BYPASS="0",
    # 0 = skip on every call the switch allows. The shipped default is 60000
    # (below it the skip is a measured prefill regression); the threshold gate
    # itself is exercised by test_the_skip_threshold_gates_the_kernel_per_call.
    DSV41_INDEX_SKIP_MIN_NB="0")


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)
    prewarm_index_fused()


@contextmanager
def _environment():
    previous = {k: os.environ.get(k) for k in SETTINGS}
    os.environ.update(SETTINGS)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _inputs(n, nb, batch=1):
    mx.random.seed(117)
    x = mx.random.normal((batch, n, 64)).astype(mx.bfloat16)
    p = (mx.random.normal((32, 64)) / 8).astype(mx.float32)
    q = lang.fake_quant_fp4_ue8m0(mx.random.normal((batch, n, 32, 128)))
    keys = lang.fake_quant_fp4_ue8m0(mx.random.normal((batch, nb, 128)))
    mx.eval(x, p, q, keys)
    return x, p, q, keys


def _eager_scores(x, p, q, keys):
    b, n, h, _ = q.shape
    chunk = max(1, 512 * 1024**2 // (b * h * keys.shape[1] * 4))
    parts = [lang._index_scores(x[:, i:i + chunk], p, q[:, i:i + chunk], keys, SCALE)
             for i in range(0, n, chunk)]
    return mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def _positions(keep, nb, block_size=8):
    """The DSV41_S4_PRECOMPUTE=0 layout of the same mask: the block flags
    repeated block_size times and trimmed to nb. The kernel reads this at
    stride block_size, so both layouts must give bit-identical output."""
    return mx.repeat(keep, block_size, axis=-1)[..., :nb]


def _both_layouts(x, p, q, keys, keep, block_size, label):
    """Score once per layout and prove the two are indistinguishable: same
    values, same NEG_INF geometry, therefore the same selected sets and the
    same set of tiles the kernel took the skip exit on."""
    nb = keys.shape[1]
    prewarm_index_skip(block_size, True)
    prewarm_index_skip(block_size, False)
    block = index_scores_fused(x, p, q, keys, SCALE, candidates=keep,
                               block_size=block_size, block_mask=True)
    position = index_scores_fused(x, p, q, keys, SCALE,
                                  candidates=_positions(keep, nb, block_size),
                                  block_size=block_size, block_mask=False)
    mx.eval(block, position)
    np.testing.assert_array_equal(np.array(block), np.array(position))
    np.testing.assert_array_equal(np.array(mx.isinf(block)), np.array(mx.isinf(position)))
    # The skip exit writes NEG_INF over a whole 16x32 tile. Equal NEG_INF
    # geometry plus equal values means the two layouts skipped the same tiles;
    # count them from the flags, the arithmetic the kernel's vote performs.
    skipped, tiles = _skipped_tiles(keep, nb, block_size)
    print(f"{label} layouts_identical tiles={tiles} skipped={skipped}", flush=True)
    return block, position, skipped


def _masked(scores, lens, keep=None, block_size=8):
    """Visibility first, then the candidate blocks: the production order."""
    nb = scores.shape[-1]
    scores = mx.where((mx.arange(nb)[None] < lens)[None], scores, -float("inf"))
    if keep is not None:
        scores = lang._mask_candidate_blocks(scores, keep, block_size)
    return scores


def _select_masked(scores, lens, k=512, offset=128):
    nb = scores.shape[-1]
    k = min(k, nb)
    idx = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
    idx = mx.sort(idx, axis=-1)
    visible = ((idx < lens.astype(mx.int32)[None])
               & (mx.take_along_axis(scores, idx, axis=-1) > -float("inf")))
    return mx.where(visible, idx + offset, mx.array(-1, mx.int32))


def _select(scores, lens, keep=None, block_size=8, k=512, offset=128):
    return _select_masked(_masked(scores, lens, keep, block_size), lens, k, offset)


# FP32 has u = 2^-24. A length-m summation carries at most gamma(m) relative
# error against the exactly rounded result, whatever the order, so two orders
# differ by at most the sum of their two bounds (Higham, ASNA 2nd ed, 3.1).
U = 2.0 ** -24


def _gamma(m):
    return m * U / (1.0 - m * U)


def _score_bound(q_row, keys_cols, w_row):
    """Two-sided bound on the reduced score at `keys_cols` for one query row.

    The 128-term dot and the 32-term head sum are the only reductions; the
    weights projection is computed identically on both paths, so it cancels.
    """
    q64 = np.asarray(q_row, np.float64)                 # [h, d]
    w64 = np.abs(np.asarray(w_row, np.float64))[:, None]  # [h, 1]
    k64 = np.asarray(keys_cols, np.float64)             # [c, d]
    dots = q64 @ k64.T                                  # [h, c]
    magnitude = np.abs(q64) @ np.abs(k64).T             # [h, c]
    heads = np.maximum(dots, 0.0) * w64
    return 2.0 * (_gamma(128) * (magnitude * w64).sum(0)
                  + _gamma(32) * np.abs(heads).sum(0))


def _evidence(masked_reference, q, keys, weights, k=512, offset=128):
    return SimpleNamespace(scores=masked_reference, q=q, keys=keys,
                           weights=weights, k=k, offset=offset)


def _prove_boundary_ties(actual, expected, evidence, label):
    """Every differing ID must sit within the fp32 reordering bound of the
    k-th largest reference score. Anything else is a real selection error."""
    rows = np.argwhere(np.any(actual != expected, axis=-1))
    slots = int((actual != expected).sum())
    keys_np, worst_gap, worst_ratio, differing = {}, 0.0, 0.0, 0
    for bi, ri in rows:
        bi, ri = int(bi), int(ri)
        reference = np.asarray(evidence.scores[bi, ri], np.float64)
        picked = set(int(v) for v in actual[bi, ri]) - {-1}
        wanted = set(int(v) for v in expected[bi, ri]) - {-1}
        symmetric = sorted(picked ^ wanted)
        differing += len(symmetric)
        k = min(evidence.k, reference.size)
        top = np.argpartition(-reference, k - 1)[:k]
        boundary = int(top[np.argmin(reference[top])])
        threshold = reference[boundary]
        if bi not in keys_np:
            keys_np[bi] = np.array(evidence.keys[bi].astype(mx.float32))
        columns = np.array([v - evidence.offset for v in symmetric], np.int64)
        assert np.all((columns >= 0) & (columns < reference.size)), f"{label} bad ID"
        query = np.array(evidence.q[bi, ri].astype(mx.float32))
        weight = np.array(evidence.weights[bi, ri].astype(mx.float32))
        bound = (_score_bound(query, keys_np[bi][columns], weight)
                 + _score_bound(query, keys_np[bi][[boundary]], weight)[0])
        gaps = np.abs(reference[columns] - threshold)
        for column, gap, tol in zip(columns, gaps, bound):
            assert gap <= tol, (f"{label} selection error at row {bi},{ri} key "
                                f"{column}: score is {gap:.6g} from the k-th "
                                f"threshold, bound {tol:.6g}")
        worst_gap = max(worst_gap, float(gaps.max()))
        worst_ratio = max(worst_ratio,
                          float(np.where(bound > 0, gaps / np.maximum(bound, 1e-300), 0).max()))
    print(f"{label} rows_differing={len(rows)}/{actual.shape[0] * actual.shape[1]} "
          f"({len(rows) / (actual.shape[0] * actual.shape[1]):.6f}) "
          f"slots_differing={slots}/{actual.size} ({slots / actual.size:.8f}) "
          f"ids_differing={differing} max_gap_to_kth={worst_gap:.6g} "
          f"max_gap_over_bound={worst_ratio:.4f}", flush=True)


def _same_sets(actual, expected, evidence=None, label=""):
    # -1 placement is irrelevant to the selected set; multiplicity is checked.
    # Exact set equality is the gate; a row may only differ when every ID in
    # the symmetric difference is a boundary tie within the reordering bound.
    assert actual.dtype == expected.dtype == mx.int32
    actual, expected = np.sort(np.array(actual), -1), np.sort(np.array(expected), -1)
    if evidence is None or np.array_equal(actual, expected):
        np.testing.assert_array_equal(actual, expected)
        return
    _prove_boundary_ties(actual, expected, evidence, label)


def _metrics(actual, expected, label):
    error = actual - expected
    maximum = mx.max(mx.abs(error))
    relative = mx.sqrt(mx.mean(error * error) / mx.maximum(mx.mean(expected * expected), 1e-30))
    mx.eval(maximum, relative)
    print(f"{label} max_abs={maximum.item():.9g} rms_rel={relative.item():.9g}", flush=True)
    assert bool(mx.all(mx.isfinite(actual)).item())


@pytest.mark.parametrize("nb", [3000, 6000, 15000, 31000])
@pytest.mark.parametrize("n", [64, 512, 2048])
@pytest.mark.parametrize("candidates", [False, True])
def test_selected_sets_matrix(n, nb, candidates):
    x, p, q, keys = _inputs(n, nb)
    reference = _eager_scores(x, p, q, keys)
    fused = index_scores_fused(x, p, q, keys, SCALE)
    mx.eval(reference, fused)
    label = f"n={n} nb={nb} candidates={candidates}"
    _metrics(fused, reference, label)
    weights = (x.astype(mx.float32) @ p.astype(mx.float32).T) * SCALE
    keep = (mx.random.uniform(shape=(1, n, (nb + 7) // 8)) > 0.4) if candidates else None
    kernels = []
    if candidates:
        # The same mask through the kernel, in BOTH layouts: NEG_INF written
        # there, not here, and the two layouts must be indistinguishable.
        mx.eval(keep, weights)
        block, position, _ = _both_layouts(x, p, q, keys, keep, 8, label)
        kernels = [("block", block), ("position", position)]
    for mode, lens in (
        ("empty", mx.zeros((n, 1), mx.int32)),
        ("full", mx.full((n, 1), nb, mx.int32)),
        ("causal", ((mx.arange(n) + nb - n + 1)[:, None]).astype(mx.int32)),
    ):
        masked_reference = _masked(reference, lens, keep)
        expected = _select_masked(masked_reference, lens)
        actual = _select(fused, lens, keep)
        mx.eval(actual, expected, masked_reference)
        evidence = _evidence(masked_reference, q, keys, weights)
        _same_sets(actual, expected, evidence, f"{label} mode={mode}")
        for layout, scores in kernels:
            # Visibility is applied after the kernel, exactly as in production.
            kernel = _select_masked(_masked(scores, lens), lens)
            mx.eval(kernel)
            _same_sets(kernel, expected, evidence,
                       f"{label} mode={mode} kernel_mask={layout}")
        if mode == "empty":
            assert bool(mx.all(actual == -1).item())


@pytest.mark.parametrize("kind", ["zero", "nonzero_boundary", "sparse_candidates"])
@pytest.mark.parametrize("block_size", [3, 8, 16])
def test_ties_and_query_key_tails(kind, block_size):
    n, nb = 65, 1031  # Partial query AND key tiles, batch addressing too.
    x = mx.ones((2, n, 64), mx.float32)
    p = mx.ones((32, 64), mx.float32) / 64
    q = mx.full((2, n, 32, 128), 0 if kind == "zero" else 1, mx.bfloat16)
    # First 511 rows strictly win; the 512th ties across the remaining keys.
    keys = mx.broadcast_to(mx.where(mx.arange(nb) < 511, 2.0, 1.0)[None, :, None], (2, nb, 128))
    ref = _eager_scores(x, p, q, keys)
    fused = index_scores_fused(x, p, q, keys, SCALE)
    _metrics(fused, ref, f"ties={kind} block_size={block_size}")
    lens = mx.full((n, 1), nb, mx.int32)
    keep = None
    if kind == "sparse_candidates":
        keep = mx.broadcast_to((mx.arange((nb + block_size - 1) // block_size) < 2)[None, None],
                               (2, n, (nb + block_size - 1) // block_size))
    _same_sets(_select(fused, lens, keep, block_size), _select(ref, lens, keep, block_size))
    if keep is None:
        return
    # Block sizes 3 and 16 do not divide the BN=32 key tile: the kernel must
    # still mask every key and may only skip a tile no admitted block reaches.
    # nb=1031 is a multiple of neither 32 nor any of the block sizes, so this
    # also covers the position layout's short final block.
    expected = _masked(ref, lens, keep, block_size)
    mx.eval(expected)
    both = _both_layouts(x, p, q, keys, keep, block_size,
                         f"ties={kind} block_size={block_size}")
    for kernel in both[:2]:
        _same_sets(_select_masked(_masked(kernel, lens), lens), _select_masked(expected, lens))
        np.testing.assert_array_equal(np.array(mx.isinf(kernel)), np.array(mx.isinf(expected)))


def _admitted_mask(n, nb, blocks=2048, block_size=8, jitter=0.02, seed=31):
    """A realistic block mask: one shared key-bank ranking plus per-query
    jitter. Adjacent queries admit nearly the same blocks, which is what makes
    a 16-row threadgroup skippable at all (see index_skip_impl.md)."""
    rng = np.random.default_rng(seed)
    nblocks = (nb + block_size - 1) // block_size
    scores = (rng.standard_normal(nblocks)[None]
              + jitter * rng.standard_normal((n, nblocks)))
    k = min(blocks, nblocks)
    keep = np.zeros((n, nblocks), bool)
    np.put_along_axis(keep, np.argpartition(-scores, k - 1, axis=-1)[:, :k], True, axis=-1)
    return mx.array(keep[None])


def _skipped_tiles(keep, nb, block_size=8, rows=16, width=32):
    """Kernel arithmetic: a tile is skipped when no admitted block of any of
    its `rows` query rows reaches its `width` keys."""
    keep = np.array(keep)
    b, n, nblocks = keep.shape
    pad = -n % rows
    grouped = np.pad(keep, ((0, 0), (0, pad), (0, 0))).reshape(
        b, (n + pad) // rows, rows, nblocks).any(2)                # [b, RT, NB]
    tiles = np.zeros((b, grouped.shape[1], -(-nb // width)), bool)
    for tile, col0 in enumerate(range(0, nb, width)):
        first = col0 // block_size
        last = min((col0 + width - 1) // block_size, nblocks - 1)
        tiles[:, :, tile] = grouped[:, :, first:last + 1].any(-1)
    return int((~tiles).sum()), tiles.size


def _skipped_tiles_by_position(keep, nb, block_size=8, rows=16, width=32):
    """Same count taken through the expanded position mask, independently."""
    positions = np.repeat(np.array(keep), block_size, axis=-1)[..., :nb]
    b, n, _ = positions.shape
    pad_r, pad_c = -n % rows, -nb % width
    padded = np.pad(positions, ((0, 0), (0, pad_r), (0, pad_c)))
    tiles = padded.reshape(b, (n + pad_r) // rows, rows,
                           (nb + pad_c) // width, width).any((2, 4))
    return int((~tiles).sum()), tiles.size


@pytest.mark.parametrize("nb", [25700, 62000, 131072])
def test_candidate_skip_selected_sets_and_tiles(nb):
    """Production consumer geometry: n=2048 against a 16,384-position bank."""
    n, block_size = 2048, 8
    x, p, q, keys = _inputs(n, nb)
    keep = _admitted_mask(n, nb, block_size=block_size)
    weights = (x.astype(mx.float32) @ p.astype(mx.float32).T) * SCALE
    lens = mx.full((n, 1), nb, mx.int32)
    reference = _masked(_eager_scores(x, p, q, keys), lens, keep, block_size)
    mx.eval(reference, weights, keep)
    block, position, _ = _both_layouts(x, p, q, keys, keep, block_size, f"skip nb={nb}")

    # (a) identical NEG_INF geometry and identical selected index set, from
    #     the block layout and from the per-position layout alike.
    evidence = _evidence(reference, q, keys, weights)
    expected = _select_masked(reference, lens)
    mx.eval(expected)
    for layout, kernel in (("block", block), ("position", position)):
        assert bool(mx.all(mx.isinf(kernel) == mx.isinf(reference)).item())
        actual = _select_masked(_masked(kernel, lens), lens)
        mx.eval(actual)
        _same_sets(actual, expected, evidence, f"skip n={n} nb={nb} layout={layout}")

    # (b) the skipped-tile count, by block flags and again by position.
    skipped, tiles = _skipped_tiles(keep, nb, block_size)
    by_position = _skipped_tiles_by_position(keep, nb, block_size)
    assert (skipped, tiles) == by_position, "block and position arithmetic disagree"
    admitted = int(np.array(keep).sum()) / np.array(keep).size
    print(f"skip nb={nb} tiles={tiles} skipped={skipped} "
          f"fraction={skipped / tiles:.4f} admitted_blocks={admitted:.4f} "
          f"ideal_waste={1 - min(2048 * block_size, nb) / nb:.4f}", flush=True)

    # (c) the same launch with and without the mask, five warm, ten timed.
    positions = _positions(keep, nb, block_size)
    mx.eval(positions)

    def time_path(masked, block_mask=True):
        mask = (keep if block_mask else positions) if masked else None
        def once():
            return index_scores_fused(x, p, q, keys, SCALE, candidates=mask,
                                      block_size=block_size, block_mask=block_mask)
        for _ in range(5):
            mx.eval(once())
        samples = []
        for _ in range(10):
            mx.synchronize()
            start = time.perf_counter()
            out = once()
            if not masked:
                out = lang._mask_candidate_blocks(out, keep, block_size)
            mx.eval(out)
            mx.synchronize()
            samples.append(1000 * (time.perf_counter() - start))
        return float(np.median(samples))
    plain, skipping = time_path(False), time_path(True)
    by_position = time_path(True, block_mask=False)
    print(f"skip nb={nb} eager_mask_ms={plain:.3f} kernel_skip_ms={skipping:.3f} "
          f"kernel_skip_position_ms={by_position:.3f} "
          f"speedup={plain / skipping:.3f}x "
          f"speedup_position={plain / by_position:.3f}x", flush=True)


def _model_paths(n, nb, *, candidates=False, source=False, production_width=False):
    config = lang.TextConfig(**({} if production_width else dict(hidden_size=64, q_lora_rank=64)),
                             candidate_topk_blocks=4 if source else 2048)
    indexer = lang.Indexer(config, 20 if source else 24)
    prepare_index_fused([SimpleNamespace(attn=SimpleNamespace(indexer=indexer))])
    x = mx.random.normal((1, n, config.hidden_size)).astype(mx.bfloat16)
    qr = mx.random.normal((1, n, config.q_lora_rank)).astype(mx.bfloat16)
    keys = lang.fake_quant_fp4_ue8m0(mx.random.normal((1, nb, 128)).astype(mx.bfloat16))
    keep = (mx.random.uniform(shape=(1, n, (nb + 7) // 8)) > 0.4) if candidates else None
    if keep is not None and os.environ["DSV41_S4_PRECOMPUTE"] == "0":
        keep = mx.repeat(keep, 8, axis=-1)[..., :nb]
    rope = lang.DeepseekV4RoPE(config.qk_rope_head_dim, config.compress_rope_theta,
                             dict(config.rope_scaling))
    mx.eval(x, qr, keys, indexer.parameters(), rope.parameters())
    if keep is not None:
        mx.eval(keep)

    def run(enabled):
        # A single indexer/identical weights; latch was compiled above.
        indexer.index_fused = enabled
        shared = SimpleNamespace(candidates=keep)
        out = indexer(x, qr, nb - n, 128, rope, keys, shared)
        return (out, shared.candidates) if source and shared.candidates is not None else (out,)
    return lambda: run(False), lambda: run(True)


@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("source,candidates", [(False, False), (False, True), (True, False)])
def test_whole_indexer_and_candidate_source(monkeypatch, precompute, source, candidates):
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", str(int(precompute)))
    eager, fused = _model_paths(64, 1031, candidates=candidates, source=source)
    expected, actual = eager(), fused()
    mx.eval(*expected, *actual)
    _same_sets(actual[0], expected[0])
    if source:
        np.testing.assert_array_equal(np.array(actual[1]), np.array(expected[1]))


@pytest.mark.parametrize("precompute", [False, True])
def test_whole_indexer_with_the_skip_switch(monkeypatch, precompute):
    """DSV41_INDEX_SKIP=1 changes nothing that leaves Indexer.__call__, in
    EITHER candidate layout: DSV41_S4_PRECOMPUTE=0 is the production setting
    and its per-position mask now reaches the kernel instead of being refused.
    """
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", str(int(precompute)))
    eager, fused = _model_paths(64, 1031, candidates=True)
    expected = eager()
    monkeypatch.setenv("DSV41_INDEX_SKIP", "1")
    prewarm_index_skip(8, precompute)
    actual = fused()
    mx.eval(*expected, *actual)
    _same_sets(actual[0], expected[0])


@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("min_nb,engages", [(1032, False), (1031, True), (0, True)])
def test_the_skip_threshold_gates_the_kernel_per_call(monkeypatch, precompute,
                                                      min_nb, engages):
    """DSV41_INDEX_SKIP_MIN_NB decides per call, and neither side changes the
    selected set.

    Below the threshold the call must reach the kernel with NO mask input --
    exactly as if DSV41_INDEX_SKIP were 0 -- and _mask_candidate_blocks must
    run at the call site instead; at or above it the kernel masks and skips
    nb = 1031 here, so the thresholds 1032, 1031 and 0 put the call
    below, exactly at, and above the gate.
    """
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", str(int(precompute)))
    monkeypatch.setenv("DSV41_INDEX_SKIP", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", str(min_nb))
    prewarm_index_skip(8, precompute)
    masked = []
    real = index_fused_module.index_scores_fused

    def watched(*args, **kwargs):
        masked.append(kwargs.get("candidates") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(index_fused_module, "index_scores_fused", watched)
    eager, fused = _model_paths(64, 1031, candidates=True)
    expected = eager()            # eager path: never calls the fused scorer
    actual = fused()
    mx.eval(*expected, *actual)
    assert masked == [engages]
    _same_sets(actual[0], expected[0])



def benchmark(n, nb, candidates=False, source=False):
    """Whole production-width Indexer.__call__; fresh graphs and evaluated outputs."""
    with _environment():
        paths = _model_paths(n, nb, candidates=candidates, source=source, production_width=True)
        expected, actual = paths[0](), paths[1]()
        mx.eval(*expected, *actual)
        _same_sets(actual[0], expected[0])
        if source:
            np.testing.assert_array_equal(np.array(actual[1]), np.array(expected[1]))
        for _ in range(5):
            for path in paths:
                mx.eval(*path())
        samples = [[], []]
        for repeat in range(10):
            for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                mx.synchronize()
                start = time.perf_counter()
                mx.eval(*paths[i]())
                mx.synchronize()
                samples[i].append(1000 * (time.perf_counter() - start))
        print(f"whole_indexer n={n} nb={nb} candidates={candidates} source={source} "
              f"warm=5 timed=10 eager_ms={np.median(samples[0]):.3f} "
              f"fused_ms={np.median(samples[1]):.3f} samples_ms={samples}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--nb", type=int, nargs="+", default=[3000, 6000, 15000, 31000])
    parser.add_argument("--source", action="store_true", help="also benchmark candidate-source side output")
    args = parser.parse_args()
    if min(args.n) < 64:
        parser.error("fused prefill requires n >= 64")
    for n in args.n:
        for nb in args.nb:
            for candidates in (False, True):
                benchmark(n, nb, candidates)
            if args.source:
                benchmark(n, nb, source=True)
