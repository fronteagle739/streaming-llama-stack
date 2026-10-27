"""Windows checks for actual dispatch/masks and a NumPy model of Metal tiles.

Run with python (not package pytest discovery); never imports MLX.
"""
import ast
from contextvars import ContextVar
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parent
LANGUAGE = HERE.parents[1] / "models/deepseek_v41/language.py"


def _production():
    tree = ast.parse(LANGUAGE.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_mask_candidate_blocks", "_index_scores",
                            "_index_skip_min_nb", "_index_skip_decision")]
    for cls, name, alias in (("Indexer", "__call__", "forward"),
                             ("LanguageModel", "attach_engram_storage", "attach")):
        c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        node = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.name = alias
        nodes.append(node)

    class Inject(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return ast.Pass()

    ns = dict(os=os, mx=np, NEG_INF=-float("inf"), _HEAD_BF16=False,
              _s4_count=lambda *a: None, _PREFILL_MXFP8_STEEL=False,
              _v41_verify_context=ContextVar("index_fused_test", default=False),
              _v41_prefill_enabled=lambda *a: False,
              _rope=lambda rope, q, pos: q, fake_quant_fp4_ue8m0=lambda q, block: q)
    module = ast.fix_missing_locations(Inject().visit(ast.Module(nodes, [])))
    exec("from __future__ import annotations\n" + ast.unparse(module), ns)
    return ns


def _kernel_block_flags(keep, block_size=8, block_mask=True):
    """Exactly what the kernel reads: one flag per block, from either layout.

    Block-indexed (DSV41_S4_PRECOMPUTE=1) is the mask itself. Position-indexed
    (=0) is mx.repeat(keep, block_size, -1)[..., :nb], so block b's flag sits
    at column b*block_size -- the stride read -- clamped to the last valid
    column so a short final block or a nb that is not a multiple of block_size
    can never address past the mask.
    """
    if block_mask:
        return keep
    width = keep.shape[-1]
    cols = np.minimum(np.arange(-(-width // block_size)) * block_size, width - 1)
    return keep[..., cols]


def _tile_scores(q, keys, weights, keep=None, block_size=8, block_mask=True):
    """Emulate the kernel's fragment ownership, zero padding and head loop.

    With ``keep`` this also emulates the candidate skip: the threadgroup ORs
    one flag per (row, block) over its BM=16 rows and the blocks its BN=32
    keys touch, and on an all-false verdict writes NEG_INF and returns.
    ``block_mask`` picks the mask layout; the flags are resolved once, through
    the same strided/clamped read the kernel performs.
    Returns the reduced scores and the number of tiles that took that exit.
    """
    b, s, h, d = q.shape
    n = keys.shape[1]
    if keep is not None:
        keep = _kernel_block_flags(keep, block_size, block_mask)
    nblocks = 0 if keep is None else keep.shape[-1]
    out = np.empty((b, s, n), np.float32)
    visits = np.zeros_like(out, np.uint8)
    skipped = 0
    for batch in range(b):
        for row0 in range(0, s, 16):
            for col0 in range(0, n, 32):
                mr, nc = min(16, s - row0), min(32, n - col0)
                if keep is not None:
                    first = col0 // block_size
                    last = min((col0 + 31) // block_size, nblocks - 1)
                    if not keep[batch, row0:row0 + mr, first:last + 1].any():
                        out[batch, row0:row0 + mr, col0:col0 + nc] = -np.inf
                        visits[batch, row0:row0 + mr, col0:col0 + nc] += 1
                        skipped += 1
                        continue
                reduced = np.zeros((16, 32), np.float32)
                for head in range(h):
                    scores = (q[batch, row0:row0 + mr, head]
                              @ keys[batch, col0:col0 + nc].T)
                    reduced[:mr, :nc] += np.maximum(scores, 0) * weights[
                        batch, row0:row0 + mr, head, None]
                for sg in range(2):
                    for lane in range(32):
                        qid = lane // 4
                        sm = (qid & 4) + ((lane // 2) % 4)
                        sn = (qid & 2) * 2 + (lane % 2) * 2 + sg * 8
                        for m in range(2):
                            for nn in range(2):
                                for j in range(2):
                                    r, c = sm + m * 8, sn + nn * 16 + j
                                    if r < mr and c < nc:
                                        value = reduced[r, c]
                                        if keep is not None and not keep[
                                                batch, row0 + r, (col0 + c) // block_size]:
                                            value = -np.inf
                                        out[batch, row0 + r, col0 + c] = value
                                        visits[batch, row0 + r, col0 + c] += 1
    np.testing.assert_array_equal(visits, 1)
    return out, skipped


def _skipped_tiles(keep, nb, block_size=8, rows=16, width=32):
    """The same count from the block flags alone, without the kernel model."""
    b, n, nblocks = keep.shape
    pad = -n % rows
    grouped = np.pad(keep, ((0, 0), (0, pad), (0, 0))).reshape(
        b, (n + pad) // rows, rows, nblocks).any(2)
    tiles = np.zeros((b, grouped.shape[1], -(-nb // width)), bool)
    for tile, col0 in enumerate(range(0, nb, width)):
        first = col0 // block_size
        last = min((col0 + width - 1) // block_size, nblocks - 1)
        tiles[:, :, tile] = grouped[:, :, first:last + 1].any(-1)
    return int((~tiles).sum()), tiles.size


def _skipped_tiles_by_position(keep, nb, block_size=8, rows=16, width=32):
    """And again through the expanded position mask, an independent route."""
    positions = np.repeat(keep, block_size, axis=-1)[..., :nb]
    b, n, _ = positions.shape
    pad_r, pad_c = -n % rows, -nb % width
    padded = np.pad(positions, ((0, 0), (0, pad_r), (0, pad_c)))
    tiles = padded.reshape(b, (n + pad_r) // rows, rows,
                           (nb + pad_c) // width, width).any((2, 4))
    return int((~tiles).sum()), tiles.size


def _admitted_mask(n, nb, blocks=2048, block_size=8, jitter=0.02, seed=31):
    """A shared key-bank ranking plus per-query jitter, as on the Mac side."""
    rng = np.random.default_rng(seed)
    nblocks = (nb + block_size - 1) // block_size
    scores = (rng.standard_normal(nblocks)[None]
              + jitter * rng.standard_normal((n, nblocks)))
    k = min(blocks, nblocks)
    keep = np.zeros((n, nblocks), bool)
    np.put_along_axis(keep, np.argpartition(-scores, k - 1, axis=-1)[:, :k], True, axis=-1)
    return keep[None]


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 17, 33), (2, 31, 65), (1, 64, 97)])
@pytest.mark.parametrize("block_size", [3, 8, 16])
def test_tiles_visibility_candidates_and_ties(shape, block_size):
    b, s, n = shape
    rng = np.random.default_rng(9)
    # Exactly representable FP4 values/products isolate geometry from rounding.
    q = rng.integers(-2, 3, (b, s, 32, 128)).astype(np.float32) / 2
    keys = rng.integers(-2, 3, (b, n, 128)).astype(np.float32) / 2
    w = rng.integers(-2, 3, (b, s, 32)).astype(np.float32) / 4
    eager = np.sum(np.maximum(np.einsum("bshd,btd->bsht", q, keys), 0) * w[..., None], axis=2)
    tiled, skipped = _tile_scores(q, keys, w)
    assert skipped == 0
    np.testing.assert_array_equal(tiled, eager)
    # Include empty/full visibility and a tied kth score, including signed w.
    for lens in (np.zeros((s, 1), np.int32), np.full((s, 1), n, np.int32),
                 (np.arange(s) * 2 % (n + 1))[:, None]):
        for mask in (False, True):
            keep = rng.random((b, s, (n + block_size - 1) // block_size)) > 0.4
            ref = np.where(np.arange(n) < lens, eager, -np.inf)
            actual = np.where(np.arange(n) < lens, tiled, -np.inf)
            if mask:
                actual = _production()["_mask_candidate_blocks"](actual, keep, block_size)
                ref = np.where(np.repeat(keep, block_size, axis=-1)[..., :n], ref, -np.inf)
            np.testing.assert_array_equal(actual, ref)
            for scores in (actual, np.where(np.isfinite(actual), 0, -np.inf)):
                k = min(7, n)
                idx = np.sort(np.argpartition(-scores, k - 1, axis=-1)[..., :k])
                result = np.where((idx < lens) & np.isfinite(np.take_along_axis(scores, idx, -1)),
                                  idx + 128, -1)
                assert np.all((result == -1) | ((result >= 128) & (result < 128 + lens)))
                if not lens.any():
                    np.testing.assert_array_equal(result, -1)


@pytest.mark.parametrize("shape", [(1, 17, 100), (2, 33, 131), (1, 48, 1031)])
@pytest.mark.parametrize("block_size", [3, 8, 16])
def test_candidate_tile_skip_matches_the_eager_mask(shape, block_size):
    """N is never a multiple of 32 here and 100/8 = 13 blocks is not a
    multiple of 4, so the last tile of a row holds a partial block range."""
    b, s, n = shape
    rng = np.random.default_rng(5)
    nblocks = (n + block_size - 1) // block_size
    q = rng.integers(-2, 3, (b, s, 32, 128)).astype(np.float32) / 2
    keys = rng.integers(-2, 3, (b, n, 128)).astype(np.float32) / 2
    w = rng.integers(-2, 3, (b, s, 32)).astype(np.float32) / 4
    eager = np.sum(np.maximum(np.einsum("bshd,btd->bsht", q, keys), 0) * w[..., None], axis=2)
    for admitted in (0.0, 0.08, 0.5, 1.0):
        keep = rng.random((b, s, nblocks)) < admitted
        tiled, skipped = _tile_scores(q, keys, w, keep, block_size)
        # (a) exactly what score-then-_mask_candidate_blocks leaves.
        reference = np.where(np.repeat(keep, block_size, axis=-1)[..., :n], eager, -np.inf)
        np.testing.assert_array_equal(tiled, reference)
        production = _production()["_mask_candidate_blocks"](eager, keep, block_size)
        np.testing.assert_array_equal(tiled, production)
        # (a2) and the per-position layout of the very same mask: identical
        # output, identical NEG_INF geometry, identical skipped-tile count.
        positions = np.repeat(keep, block_size, axis=-1)[..., :n]
        tiled_p, skipped_p = _tile_scores(q, keys, w, positions, block_size, False)
        np.testing.assert_array_equal(tiled_p, tiled)
        np.testing.assert_array_equal(np.isinf(tiled_p), np.isinf(tiled))
        assert skipped_p == skipped
        # (b) the skipped-tile count, from the flags alone.
        expected, tiles = _skipped_tiles(keep, n, block_size)
        assert skipped == expected
        by_position, tiles_p = _skipped_tiles_by_position(keep, n, block_size)
        assert tiles == tiles_p == -(-s // 16) * -(-n // 32) * b
        if 32 % block_size == 0:
            # Every block lies inside one tile: the two routes must agree.
            assert skipped == by_position
        else:
            # A straddling block can keep a tile alive with no admitted key
            # inside it. Skipping fewer tiles is safe; skipping more is not.
            assert skipped <= by_position
        assert (skipped == tiles) == (admitted == 0.0)


@pytest.mark.parametrize("n", [33, 100, 131, 160, 1031])
@pytest.mark.parametrize("block_size", [3, 7, 8, 16, 64])
def test_stride_read_recovers_the_block_mask_exactly(n, block_size):
    """The position layout is read at stride block_size, one flag per block.

    n covers nb not a multiple of 32 (33, 100, 131, 1031), nb not a multiple
    of 8 (33, 100, 131, 1031 at block_size 8; every n at 3/7), and block
    counts that are not a multiple of 4 (100/8 = 13, 33/8 = 5, 160/64 = 3).
    """
    rng = np.random.default_rng(17)
    nblocks = -(-n // block_size)
    keep = rng.random((2, 9, nblocks)) < 0.4
    positions = np.repeat(keep, block_size, axis=-1)[..., :n]
    assert positions.shape[-1] == n
    # The reads themselves: in range, strictly increasing, never clamped for a
    # well-formed mask, because block NB-1 starts at (NB-1)*block_size < n.
    cols = np.minimum(np.arange(nblocks) * block_size, n - 1)
    assert cols[-1] == (nblocks - 1) * block_size < n
    assert np.all(np.diff(cols) > 0) and np.all(cols < n)
    # ... and one read per block a tile touches, never one per key: at
    # block_size 8 that is the same four reads per row as the block layout.
    for col0 in range(0, n, 32):
        blk0 = col0 // block_size
        blk1 = min((col0 + 31) // block_size, nblocks - 1)
        reads = blk1 - blk0 + 1
        assert 1 <= reads <= -(-32 // block_size) + 1 and reads <= 32
        assert np.all(cols[blk0:blk1 + 1] < n)
    np.testing.assert_array_equal(
        _kernel_block_flags(positions, block_size, False), keep)


@pytest.mark.parametrize("n,block_size", [(33, 8), (33, 32), (40, 8), (65, 16),
                                          (100, 8), (160, 64), (37, 3), (37, 7)])
def test_tail_tile_with_only_a_short_final_block_admitted(n, block_size):
    """A tile whose only admitted block is the short final one, so most of the
    tile's 32 columns are past nb. Both layouts must leave exactly what
    _mask_candidate_blocks leaves and take the same skip decisions."""
    rng = np.random.default_rng(23)
    nblocks = -(-n // block_size)
    q = rng.integers(-2, 3, (1, 17, 32, 128)).astype(np.float32) / 2
    keys = rng.integers(-2, 3, (1, n, 128)).astype(np.float32) / 2
    w = rng.integers(-2, 3, (1, 17, 32)).astype(np.float32) / 4
    eager = np.sum(np.maximum(np.einsum("bshd,btd->bsht", q, keys), 0) * w[..., None], axis=2)
    masked = _production()["_mask_candidate_blocks"]
    for last_only in (True, False):
        keep = np.zeros((1, 17, nblocks), bool)
        keep[..., -1 if last_only else 0] = True
        positions = np.repeat(keep, block_size, axis=-1)[..., :n]
        block_out, block_skipped = _tile_scores(q, keys, w, keep, block_size)
        pos_out, pos_skipped = _tile_scores(q, keys, w, positions, block_size, False)
        reference = masked(eager, keep, block_size)
        np.testing.assert_array_equal(block_out, reference)
        np.testing.assert_array_equal(pos_out, reference)
        assert block_skipped == pos_skipped
        # The admitted block really does sit in the final, partly valid tile.
        live = np.isfinite(reference[0, 0])
        assert live.any() and live.sum() <= block_size
        if last_only:
            assert np.flatnonzero(live)[-1] == n - 1


@pytest.mark.parametrize("nb", [25700, 62000, 131072])
def test_consumer_layer_skip_fraction(nb):
    """The production consumer geometry: 2048 admitted blocks of 8 out of
    ceil(nb/8), n=2048 queries, so 16,384 admitted key positions."""
    n, block_size = 2048, 8
    keep = _admitted_mask(n, nb, block_size=block_size)
    skipped, tiles = _skipped_tiles(keep, nb, block_size)
    assert (skipped, tiles) == _skipped_tiles_by_position(keep, nb, block_size)
    assert tiles == (n // 16) * -(-nb // 32)
    ideal = 1 - min(2048 * block_size, nb) / nb
    print(f"nb={nb} tiles={tiles} skipped={skipped} fraction={skipped / tiles:.4f} "
          f"ideal_waste={ideal:.4f}", flush=True)
    # The skip can never exceed the fraction of unadmitted keys, and with a
    # row-correlated mask it recovers most of it at the long contexts.
    assert skipped / tiles <= ideal + 1e-9
    assert skipped / tiles >= {25700: 0.012, 62000: 0.24, 131072: 0.50}[nb]
    # A row-independent mask ORs 16 rows into almost every tile: the skip is
    # worth having only because adjacent queries admit nearly the same blocks.
    loose = _admitted_mask(n, nb, block_size=block_size, jitter=50.0)
    assert _skipped_tiles(loose, nb, block_size)[0] / tiles < 0.01


@pytest.mark.parametrize("nb", [3000, 6000, 15000, 31000])
@pytest.mark.parametrize("n", [64, 512, 2048])
def test_production_grid_and_causal_lengths(nb, n):
    cols = np.concatenate([np.arange(c, min(c + 32, nb)) for c in range(0, nb, 32)])
    rows = np.concatenate([np.arange(r, min(r + 16, n)) for r in range(0, n, 16)])
    np.testing.assert_array_equal(cols, np.arange(nb))
    np.testing.assert_array_equal(rows, np.arange(n))
    for ratio in (1, 2):
        start = nb * ratio - n
        lens = (start + rows + 1) // ratio
        assert lens[-1] == nb and np.all(lens[:-1] <= lens[1:])


def _selection_gate():
    """Lift the Mac gate out of test_index_fused.py; it never imports MLX."""
    tree = ast.parse((HERE / "test_index_fused.py").read_text(encoding="utf-8"))
    wanted = ("_gamma", "_score_bound", "_evidence", "_prove_boundary_ties", "_same_sets")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == len(wanted)
    ns = dict(np=np, mx=SimpleNamespace(int32=np.int32, float32=np.float32),
              U=2.0 ** -24, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.Module(nodes, []), "index_fused_gate_cpu", "exec"), ns)
    return ns


def _numpy_select(scores, k=512, offset=128):
    idx = np.sort(np.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(np.int32), -1)
    return np.where(np.isfinite(np.take_along_axis(scores, idx, -1)), idx + offset, -1)


def test_boundary_tie_gate_accepts_reordering_and_rejects_a_real_error():
    """The selection gate: exact sets where they hold, and where a reordered
    FP32 reduction flips a boundary tie, every differing ID must sit within
    the documented bound of the k-th score. A far-off ID must still fail."""
    gate = _selection_gate()
    rng = np.random.default_rng(3)
    b, n, nb, k, h, d = 1, 24, 4096, 64, 32, 128
    q = rng.standard_normal((b, n, h, d)).astype(np.float32) / 8
    keys = rng.standard_normal((b, nb, d)).astype(np.float32) / 8
    weights = rng.standard_normal((b, n, h)).astype(np.float32) / 32
    reference = np.einsum("bshd,btd->bst", q, keys, optimize=True) * 0  # shape only
    for bi in range(b):
        dots = np.maximum(np.einsum("shd,td->sht", q[bi], keys[bi]), 0)
        reference[bi] = np.einsum("sht,sh->st", dots, weights[bi])
    bound = np.zeros_like(reference)
    for bi in range(b):
        for ri in range(n):
            bound[bi, ri] = gate["_score_bound"](q[bi, ri], keys[bi], weights[bi, ri])
    # A reordered reduction can move each score anywhere inside its own bound.
    perturbed = reference + (rng.uniform(-0.5, 0.5, reference.shape) * bound).astype(np.float32)
    evidence = gate["_evidence"](reference, q, keys, weights, k=k)
    expected = _numpy_select(reference, k)
    actual = _numpy_select(perturbed, k)
    differing = int((np.sort(actual, -1) != np.sort(expected, -1)).sum())
    assert differing, "no boundary tie flipped: the fixture proves nothing"
    gate["_same_sets"](actual, expected, evidence, "cpu_gate")
    # Exact equality still holds, and is still required, without evidence.
    gate["_same_sets"](expected, expected)
    with pytest.raises(AssertionError):
        gate["_same_sets"](actual, expected)
    # A genuine error: swap the lowest selected ID for the strongest rejected
    # one, which is nowhere near the k-th score.
    broken = actual.copy()
    rejected = np.argsort(-reference[0, 0])[k:]
    broken[0, 0, 0] = int(rejected[0]) + 128
    with pytest.raises(AssertionError, match="selection error"):
        gate["_same_sets"](broken, expected, evidence, "cpu_gate_broken")


def test_fp4_power_of_two_values_are_exact_bf16():
    codes = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
    values = np.ldexp(codes[:, None], np.arange(-126, 125)[None]).astype(np.float32)
    # BF16 discards the low 16 FP32 bits: every nonzero code retains all bits,
    # including the quantizer's minimum scale, assuming no denormal flushing.
    assert np.isfinite(values).all()
    assert np.all((values.view(np.uint32) & 0xFFFF) == 0)
    assert np.all(((-values).view(np.uint32) & 0xFFFF) == 0)


def test_prepare_validates_before_compile_and_wraps_failure():
    tree = ast.parse((HERE / "index_fused.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "prepare_index_fused")
    calls = []
    ns = dict(prewarm_index_fused=lambda: calls.append("compile"),
              prewarm_index_skip=lambda size, block_mask: calls.append(
                  f"skip{size}{'' if block_mask else 'pos'}"))
    exec(compile(ast.Module([node], []), "prepare_index_fused_cpu", "exec"), ns)
    indexer = SimpleNamespace(n_heads=32, head_dim=128, candidate_block_size=8)
    layers = [SimpleNamespace(attn=SimpleNamespace(indexer=indexer))]
    ns["prepare_index_fused"](layers)
    assert calls == ["compile"]
    indexer.head_dim = 64
    with pytest.raises(RuntimeError, match="attach failed"):
        ns["prepare_index_fused"](layers)
    assert calls == ["compile"]
    indexer.head_dim = 128
    def fail():
        raise RuntimeError("shader rejected")
    ns["prewarm_index_fused"] = fail
    with pytest.raises(RuntimeError, match="attach failed") as error:
        ns["prepare_index_fused"](layers)
    assert str(error.value.__cause__) == "shader rejected"
    # skip=True compiles the masked variant as well, once per block size and
    # once per mask layout: DSV41_S4_PRECOMPUTE is read per call, not latched.
    ns["prewarm_index_fused"] = lambda: calls.append("compile")
    ns["prepare_index_fused"](layers, skip=True)
    assert calls == ["compile", "compile", "skip8", "skip8pos"]


@pytest.fixture
def dispatch(monkeypatch):
    for name in ("INDEX_FUSED", "PREFILL_TILED_INDEX", "PREFILL_V41_INDEX", "METAL_INDEXER",
                 "FFN_FUSED", "FFN_DOWN_BF16", "FFN_EXPERT_PAD16", "FFN_EXPERT_TILES",
                 "PROMOTE_DECODE_WEIGHTS", "S4_COMPILE_INDEXER", "S4_PRECOMPUTE", "S4_BYPASS"):
        monkeypatch.setenv("DSV41_" + name, "0")
    monkeypatch.setenv("DSV41_INDEX_WORKSPACE_MB", "512")
    ns = _production()
    calls = []
    ns["prepare_index_fused"] = lambda layers, skip=False: calls.append(
        "compile+skip" if skip else "compile")
    ns["_index_geometry"] = lambda shared, start_pos, n, nb, ratio: (
        ((start_pos + np.arange(n) + 1) // ratio)[:, None],
        None if n == 1 else np.arange(nb)[None, :]
        < ((start_pos + np.arange(n) + 1) // ratio)[:, None])
    def score(label, x, p, q, keys, scale, candidates=None, block_size=8,
              block_mask=True):
        calls.append(label if candidates is None else
                     f"{label}+{'mask' if block_mask else 'posmask'}{block_size}")
        nb = keys.shape[1]
        scores = np.broadcast_to(np.arange(nb, dtype=np.float32),
                                 (*q.shape[:2], nb)).copy()
        if candidates is not None:
            # The real kernel writes NEG_INF for every unadmitted key. Model
            # that, so a test can tell from the returned selection whether the
            # mask reached the scores at all, whoever applied it.
            keep = (np.repeat(candidates, block_size, axis=-1)[..., :nb]
                    if block_mask else candidates)
            scores = np.where(keep, scores, -np.inf)
        return scores
    ns["index_scores_fused"] = lambda *a, **kw: score("fused", *a, **kw)
    ns["_index_scores"] = lambda *a, **kw: score("eager", *a, **kw)
    masked = ns["_mask_candidate_blocks"]
    def counted(scores, keep, block_size):
        calls.append("eager_mask")
        return masked(scores, keep, block_size)
    ns["_mask_candidate_blocks"] = counted
    indexer = SimpleNamespace(index_fused=False, layer_id=24, ratio=1, index_topk=4,
        candidate_topk_blocks=8, candidate_block_size=8, n_heads=32, head_dim=128,
        softmax_scale=128 ** -0.5, is_candidate_source=False, uses_candidates=False,
        wq_b=lambda qr: np.zeros((*qr.shape[:2], 4096), np.float32),
        weights_proj=SimpleNamespace(weight=np.zeros((32, 4), np.float32)))
    model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(indexer=indexer),
                            ffn=SimpleNamespace(), engram=None)], model=SimpleNamespace(layout=None))
    return ns, calls, indexer, model


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("n,verify", [(1, False), (63, False), (64, False),
                                       (2048, False), (64, True), (2048, True)])
def test_attach_latch_and_dispatch(dispatch, monkeypatch, flag, n, verify):
    ns, calls, indexer, model = dispatch
    if flag is None:
        monkeypatch.delenv("DSV41_INDEX_FUSED")
    else:
        monkeypatch.setenv("DSV41_INDEX_FUSED", flag)
    ns["attach"](model, ".")
    monkeypatch.setenv("DSV41_INDEX_FUSED", "0" if flag == "1" else "1")
    ns["attach"](model, ".")
    ns["_v41_verify_context"].set(verify)
    x = np.ones((1, n, 4), np.float32)
    out = ns["forward"](indexer, x, x, 0, 128, None,
                        np.zeros((1, 17, 128), np.float32), SimpleNamespace(candidates=None))
    assert calls == (["compile"] if flag == "1" else []) + [
        "fused" if flag == "1" and n >= 64 and not verify else "eager"]
    assert out.shape == (1, n, 4)


@pytest.mark.parametrize("precompute", ["0", "1"])
@pytest.mark.parametrize("skip", ["0", "1"])
def test_call_site_masks_in_the_kernel_and_drops_the_eager_mask(
        dispatch, monkeypatch, skip, precompute):
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP", skip)
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", precompute)
    # 0 = always skip when the switch is on; this nb is far below the shipped
    # DSV41_INDEX_SKIP_MIN_NB, which is gated on its own below.
    monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", "0")
    indexer.uses_candidates = True
    ns["attach"](model, ".")
    assert calls == ["compile+skip" if skip == "1" else "compile"]
    n, nb = 64, 136  # 17 blocks of eight: not a multiple of four either.
    keep = np.broadcast_to(np.arange(17)[None, None] % 3 == 0, (1, n, 17))
    if precompute == "0":
        keep = np.repeat(keep, 8, axis=-1)[..., :nb]
    x = np.ones((1, n, 4), np.float32)
    out = ns["forward"](indexer, x, x, 0, 128, None, np.zeros((1, nb, 128), np.float32),
                        SimpleNamespace(candidates=keep))
    assert out.shape == (1, n, indexer.index_topk)
    # The mask is applied once: in the kernel -- in whichever layout the
    # candidate source produced -- or eagerly afterwards.
    if skip == "1":
        expected = ["fused+mask8" if precompute == "1" else "fused+posmask8"]
    else:
        expected = ["fused", "eager_mask"] if precompute == "1" else ["fused"]
    assert calls[1:] == expected


def test_call_site_passes_the_position_layout_through_instead_of_refusing(
        dispatch, monkeypatch):
    """The DSV41_S4_PRECOMPUTE=1 refusal is gone: the per-position mask is the
    block mask at stride block_size, so the kernel takes it as-is."""
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP", "1")
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", "0")
    monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", "0")
    indexer.uses_candidates = True
    ns["attach"](model, ".")
    n, nb = 64, 136
    x = np.ones((1, n, 4), np.float32)
    out = ns["forward"](indexer, x, x, 0, 128, None, np.zeros((1, nb, 128), np.float32),
                        SimpleNamespace(candidates=np.ones((1, n, nb), bool)))
    assert out.shape == (1, n, indexer.index_topk)
    assert calls == ["compile+skip", "fused+posmask8"]
    # And nothing anywhere still refuses that combination.
    assert "requires DSV41_S4_PRECOMPUTE=1" not in LANGUAGE.read_text(encoding="utf-8")


# --- DSV41_INDEX_SKIP_MIN_NB: the per-call threshold gate -------------------
# Measured on the Mac (whole-model prefill tok/s, with skip vs without):
# nb 34,298 -4.8%, 47,235 -1.4%, 68,639 +4.4%, 97,932 +3.1%. The admitted width
# is capped at 2048 * 8 = 16,384 positions, so the skippable tile fraction is
# about (1 - 16384/nb) ** 4 and only pays above roughly 60,000.

SKIP_MIN_NB_DEFAULT = 60000
ADMITTED_BLOCKS = [0, 1]        # positions 0..15 -- the only finite scores


def _assert_admitted_selection(out):
    """Every row selects the top of the admitted range, not the top of the
    bank: proof that the candidate mask reached the scores exactly once."""
    expected = np.broadcast_to(np.array([12, 13, 14, 15]) + 128, out.shape)
    np.testing.assert_array_equal(out, expected)


def _threshold_case(dispatch, monkeypatch, nb, min_nb, precompute, skip="1"):
    """One dispatch through the real call site at a given nb and threshold.

    The fake kernel scores are ``arange(nb)``, so an unmasked selection is the
    top of the bank and a masked one is the top of block 1 -- positions
    12..15. Every query row sees the whole admitted range (start_pos = nb - n),
    so the returned indices say whether the mask was applied AT ALL, whichever
    side applied it, and ``calls`` says which side that was.
    """
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP", skip)
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", precompute)
    if min_nb is None:
        monkeypatch.delenv("DSV41_INDEX_SKIP_MIN_NB", raising=False)
    else:
        monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", str(min_nb))
    indexer.uses_candidates = True
    ns["attach"](model, ".")
    n = 64
    keep = np.zeros((1, n, -(-nb // 8)), bool)
    keep[..., ADMITTED_BLOCKS] = True
    if precompute == "0":
        keep = np.repeat(keep, 8, axis=-1)[..., :nb]
    x = np.ones((1, n, 4), np.float32)
    out = ns["forward"](indexer, x, x, nb - n, 128, None,
                        np.zeros((1, nb, 128), np.float32),
                        SimpleNamespace(candidates=keep))
    return out, calls[1:]


@pytest.mark.parametrize("precompute", ["0", "1"])
@pytest.mark.parametrize("nb,engages", [(120, False), (136, True), (152, True)])
def test_threshold_decides_per_call_and_the_eager_mask_runs_iff_the_kernel_did_not(
        dispatch, monkeypatch, nb, engages, precompute):
    """nb below, exactly at and above the threshold. The mask must be applied
    exactly once either way: in the kernel when the skip engages, eagerly at
    the call site when it does not."""
    out, calls = _threshold_case(dispatch, monkeypatch, nb, 136, precompute)
    if engages:
        assert calls == ["fused+mask8" if precompute == "1" else "fused+posmask8"]
    else:
        # No mask input: the plain fused path, exactly as if the switch were
        # off. masked_by_kernel is False, so the eager pass must run -- and in
        # the block layout that is the counted _mask_candidate_blocks.
        assert calls == (["fused", "eager_mask"] if precompute == "1" else ["fused"])
    # Applied once (or, in the position layout, at least once and idempotently):
    # the selection is the top of the admitted range, never the top of the bank.
    _assert_admitted_selection(out)


@pytest.mark.parametrize("precompute", ["0", "1"])
def test_threshold_zero_always_skips_when_the_switch_is_on(dispatch, monkeypatch, precompute):
    """0 is the measurement arm: skip every call the switch allows."""
    out, calls = _threshold_case(dispatch, monkeypatch, 136, 0, precompute)
    assert calls == ["fused+mask8" if precompute == "1" else "fused+posmask8"]
    _assert_admitted_selection(out)


def test_the_shipped_default_declines_a_small_call(dispatch, monkeypatch):
    """Unset means 60000, so nothing changes for a short prompt."""
    out, calls = _threshold_case(dispatch, monkeypatch, 136, None, "1")
    assert calls == ["fused", "eager_mask"]
    _assert_admitted_selection(out)
    assert _production()["_index_skip_min_nb"]() == SKIP_MIN_NB_DEFAULT


def test_the_switch_being_off_still_takes_the_unmasked_fused_path(dispatch, monkeypatch):
    """A threshold of 0 cannot turn the skip on by itself."""
    out, calls = _threshold_case(dispatch, monkeypatch, 136, 0, "1", skip="0")
    assert calls == ["fused", "eager_mask"]
    _assert_admitted_selection(out)


def test_the_decision_is_per_call_not_latched_per_model(dispatch, monkeypatch, capsys):
    """nb grows through the prefill and differs per layer: one indexer must
    decline early and engage later, and each outcome prints once."""
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP", "1")
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", "136")
    indexer.uses_candidates = True
    ns["attach"](model, ".")
    capsys.readouterr()
    n = 64
    for nb in (120, 120, 136, 152):
        keep = np.zeros((1, n, -(-nb // 8)), bool)
        keep[..., ADMITTED_BLOCKS] = True
        x = np.ones((1, n, 4), np.float32)
        ns["forward"](indexer, x, x, nb - n, 128, None,
                      np.zeros((1, nb, 128), np.float32),
                      SimpleNamespace(candidates=keep))
    assert calls[1:] == ["fused", "eager_mask", "fused", "eager_mask",
                         "fused+mask8", "fused+mask8"]
    # One line per distinct outcome, the first time it happens -- not one line
    # globally, which reports only whatever the first tiny chunk did.
    lines = [l for l in capsys.readouterr().out.splitlines() if "INDEX_SKIP" in l]
    assert len(lines) == 2
    assert "declined" in lines[0] and "nb=120" in lines[0] and "min_nb=136" in lines[0]
    assert "engaged" in lines[1] and "nb=136" in lines[1] and "min_nb=136" in lines[1]
    assert "layer=24" in lines[0] and "layer=24" in lines[1]


@pytest.mark.parametrize("skip", ["0", "1"])
@pytest.mark.parametrize("value", ["", " ", "abc", "-1", "-60000", "12.5", "1e5",
                                   "0x10", "None", "60000.0", "6e4"])
def test_an_invalid_threshold_is_loud_at_attach(dispatch, monkeypatch, value, skip):
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP", skip)
    monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", value)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        ns["attach"](model, ".")
    # Nothing compiled, nothing latched: no silent fallback to the default.
    assert not calls and not indexer.index_fused
    assert not hasattr(model, "_index_fused_attached")


@pytest.mark.parametrize("value,expected", [("0", 0), ("1", 1), ("60000", 60000),
                                            ("100000", 100000)])
def test_a_valid_threshold_is_accepted_at_attach(dispatch, monkeypatch, value, expected):
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP", "1")
    monkeypatch.setenv("DSV41_INDEX_SKIP_MIN_NB", value)
    ns["attach"](model, ".")
    assert calls == ["compile+skip"] and indexer.index_fused
    assert ns["_index_skip_min_nb"]() == expected
    assert ns["_index_skip_decision"](expected, 24)[0]
    assert (expected == 0) or not ns["_index_skip_decision"](expected - 1, 24)[0]


@pytest.mark.parametrize("fused,skip,message", [("1", "yes", "must be 0 or 1"),
                                                ("0", "1", "requires DSV41_INDEX_FUSED=1")])
def test_skip_switch_is_validated_at_attach(dispatch, monkeypatch, fused, skip, message):
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", fused)
    monkeypatch.setenv("DSV41_INDEX_SKIP", skip)
    with pytest.raises(ValueError, match=message):
        ns["attach"](model, ".")
    assert not calls and not indexer.index_fused


def test_failed_compile_does_not_latch(dispatch, monkeypatch):
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    def fail(layers, skip=False):
        raise RuntimeError("compile failed")
    ns["prepare_index_fused"] = fail
    with pytest.raises(RuntimeError, match="compile failed"):
        ns["attach"](model, ".")
    assert not hasattr(model, "_index_fused_attached") and not indexer.index_fused
    ns["prepare_index_fused"] = lambda layers, skip=False: calls.append("retry")
    ns["attach"](model, ".")
    assert indexer.index_fused and calls == ["retry"]


@pytest.mark.parametrize("conflict", ["PREFILL_TILED_INDEX", "PREFILL_V41_INDEX", "METAL_INDEXER"])
def test_conflicts_at_attach_and_after_attach(dispatch, monkeypatch, conflict):
    ns, calls, indexer, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "1")
    monkeypatch.setenv("DSV41_" + conflict, "1")
    with pytest.raises(ValueError, match="cannot combine"):
        ns["attach"](model, ".")
    assert not calls and not indexer.index_fused
    monkeypatch.setenv("DSV41_" + conflict, "0")
    ns["attach"](model, ".")
    monkeypatch.setenv("DSV41_" + conflict, "1")
    x = np.ones((1, 64, 4), np.float32)
    with pytest.raises(ValueError, match="cannot combine"):
        ns["forward"](indexer, x, x, 0, 128, None, None, None)


def test_invalid_switch(dispatch, monkeypatch):
    ns, _, _, model = dispatch
    monkeypatch.setenv("DSV41_INDEX_FUSED", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ns["attach"](model, ".")


def test_bundle_and_no_headwise_output():
    source = (HERE / "index_fused.metal").read_text(encoding="utf-8")
    upstream = (HERE / "expert_tiles_qmm.metal").read_text(encoding="utf-8")
    for name, body in re.findall(r'// BEGIN UPSTREAM ([^\n]+)\n(.*?)// END UPSTREAM \1', source, re.S):
        assert f"// BEGIN UPSTREAM {name}\n{body}// END UPSTREAM {name}" in upstream
    def manifest(file):
        tree = ast.parse((HERE / file).read_text())
        return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SOURCE_HASHES" for t in n.targets))
    recorded, original = manifest("index_fused.py"), manifest("expert_tiles_qmm.py")
    assert len(recorded) == 11 and all(original[k] == v for k, v in recorded.items())
    wrapper = (HERE / "index_fused.py").read_text()
    assert 'compile_options={"math_mode": "safe"}' in wrapper
    assert 'output_shapes=[(b, s, n)]' in wrapper
    body = source.split("// --- index fused body ---")[1]
    # Still only the reduced [b, s, n] row: the skip adds a NEG_INF write to
    # the same addresses, never a headwise buffer.
    assert body.count("Output[") == 3 and "threadgroup bfloat16_t ks[BN * LD]" in body
    assert body.count("Output[(size_t(batch) * S + row) * N + col") == 3
    # Both variants come from this one body; the define picks the skip.
    assert body.count("#if DSV41_INDEX_SKIP") == 2 and body.count("#endif") == 2
    for value in ("0", "1"):
        assert f'source="#define DSV41_INDEX_SKIP {value}' in wrapper.replace("\\n", "")
    # The layout is a template flag on the one masked variant, not a third
    # kernel, and every mask read goes through the single strided accessor.
    assert '("BS", block_size), ("POSMASK", int(not block_mask))' in wrapper
    assert body.count("#define DSV41_BLOCK_FLAG") == 1
    assert body.count("DSV41_BLOCK_FLAG(") == 3          # define + two reads
    assert "metal::min((blk) * uint(BS), MW - 1u) : (blk)" in body
    assert "const uint NB = POSMASK ? (MW + uint(BS) - 1) / uint(BS) : MW;" in body
    # No mask is read by any other route.
    assert body.count("Blocks[") == 1 and "Blocks_shape[2]" in body
    tree = ast.parse(wrapper)
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name not in ("prewarm_index_fused",
                                                               "prewarm_index_skip"):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in ("eval", "async_eval", "synchronize", "item", "tolist")


if __name__ == "__main__":
    # Match the sibling CPU harness: pytest package discovery imports MLX.
    import sys

    class StandaloneModule(pytest.Module):
        def _getobj(self):
            return sys.modules[__name__]

    class Collector:
        def pytest_pycollect_makemodule(self, module_path, parent):
            return StandaloneModule.from_parent(parent.session, path=module_path)

    raise SystemExit(pytest.main(["--noconftest", "-q", str(Path(__file__).resolve()),
                                 *sys.argv[1:]], plugins=[Collector()]))
