"""Mac differential gates and timing helper for the opt-in prefill indexer.

Run the full shape matrix with pytest. --bench reports the isolated score/top-k
pair plus a pair through the real Indexer hook (including query projection,
RoPE, quantisation, geometry, candidate selection and the workspace chunk loop).
--model-matrix runs n=2048 at N=6000/30000 with candidates off/on. N counts
compressed keys, not prompt tokens. Compilation/input creation are excluded.
"""

import argparse
import importlib
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires Apple Metal", allow_module_level=True)

from mlx_vlm.custom_kernels.deepseek_v41.index_prefill import index_scores_topk
from mlx_vlm.models.deepseek_v41.language import (
    Indexer,
    TextConfig,
    _index_scores,
    _mask_candidate_blocks,
    fake_quant_fp4_ue8m0,
    select_candidate_blocks,
)

SCALE = 128 ** -0.5 * 32 ** -0.5


@pytest.fixture(autouse=True)
def _reference_environment(monkeypatch):
    monkeypatch.setenv("DSV41_QAT", "1")
    monkeypatch.setenv("DSV41_METAL_QUANT", "0")
    monkeypatch.setenv("DSV41_INDEX_TOPK", "512")


def _inputs(n, context, mask, *, batch=1, dtype=mx.float32, seed=71):
    mx.random.seed(seed)
    x = mx.random.normal((batch, n, 64)).astype(dtype)
    p = (mx.random.normal((32, 64)) / 8).astype(dtype)
    q = fake_quant_fp4_ue8m0(mx.random.normal((batch, n, 32, 128)).astype(dtype))
    keys = fake_quant_fp4_ue8m0(mx.random.normal((batch, context, 128)).astype(dtype))
    lens = (mx.arange(n, dtype=mx.int32) + context - n + 1)[:, None]
    visibility = mx.arange(context)[None] < lens
    keep = None
    if mask:
        keep = mx.random.uniform(shape=(batch, n, (context + 7) // 8)) > 0.35
    mx.eval(x, p, q, keys, lens, visibility)
    if keep is not None:
        mx.eval(keep)
    return x, p, q, keys, lens, visibility, keep


def _reference(x, p, q, keys, lens, offset, k, *, visibility=None,
               candidates=None, candidates_are_blocks=True, workspace_mb=512):
    """The existing Indexer score/mask/partition/sort/gather/where path."""
    b, n, h, _ = q.shape
    context = keys.shape[1]
    chunk = max(1, workspace_mb * 1024**2 // (b * h * context * 4)) if workspace_mb else n
    parts = [_index_scores(x[:, i:i + chunk], p, q[:, i:i + chunk], keys, SCALE)
             for i in range(0, n, chunk)]
    scores = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    if visibility is not None:
        scores = mx.where(visibility[None], scores, -float("inf"))
    if candidates is not None:
        scores = (_mask_candidate_blocks(scores, candidates, 8) if candidates_are_blocks
                  else mx.where(candidates, scores, -float("inf")))
    idx = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
    idx = mx.sort(idx, axis=-1)
    visible = ((idx < lens.astype(mx.int32)[None])
               & (mx.take_along_axis(scores, idx, axis=-1) > -float("inf")))
    return mx.where(visible, idx + offset, mx.array(-1, mx.int32)), scores


def _assert_selection(actual, expected, scores, offset):
    """Exact sets and position order, permitting only EXACT boundary-score ties.

    -inf ties can move the -1 slots after sorting raw ids; compare reachable
    positions in order and require the same count, never turn a -1 into a gather.
    A tolerance on scores must not conceal a changed near-tie ranking.
    """
    assert actual.dtype == mx.int32
    a, e = np.array(actual), np.array(expected)
    assert a.shape == e.shape
    assert np.all((a == -1) | (a >= offset))
    for bi in range(a.shape[0]):
        for qi in range(a.shape[1]):
            av, ev = a[bi, qi], e[bi, qi]
            av, ev = av[av >= 0] - offset, ev[ev >= 0] - offset
            assert len(av) == len(ev), (bi, qi, "reachable count")
            assert np.all(av[1:] > av[:-1]), (bi, qi, "position order/duplicates")
            if np.array_equal(av, ev):
                continue
            removed = np.setdiff1d(ev, av)
            added = np.setdiff1d(av, ev)
            removed_scores = np.array(scores[bi, qi, mx.array(removed, dtype=mx.int32)])
            added_scores = np.array(scores[bi, qi, mx.array(added, dtype=mx.int32)])
            np.testing.assert_array_equal(np.sort(removed_scores), np.sort(added_scores),
                                          err_msg=f"non-tie selection mismatch at {bi=}, {qi=}")


@pytest.mark.parametrize("n", [16, 256, 2048])
@pytest.mark.parametrize("context", [6000, 30000])
@pytest.mark.parametrize("mask", [False, True])
def test_prefill_shape_matrix(n, context, mask):
    x, p, q, keys, lens, visibility, keep = _inputs(n, context, mask)
    expected, scores = _reference(x, p, q, keys, lens, 128, 512,
                                  visibility=visibility, candidates=keep)
    actual = index_scores_topk(x, p, q, keys, lens, 128, 512,
                               visibility=visibility, candidates=keep)
    mx.eval(expected, actual, scores)
    _assert_selection(actual, expected, scores, 128)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("k", [1, 37, 512])
def test_batch_dtype_tail_and_position_masks(dtype, k):
    x, p, q, keys, lens, visibility, keep = _inputs(16, 1031, True, batch=2, dtype=dtype)
    # Exercise the legacy S4_PRECOMPUTE=0 per-position mask, strided inputs,
    # and a partial final key tile/block. Exclude extra keys through visibility.
    keep = mx.repeat(keep, 8, axis=-1)[..., :1031]
    visibility = visibility & (mx.arange(1031)[None] % 11 != 0)
    x = mx.stack([x, x], axis=-1)[..., 0]
    q = mx.stack([q, q], axis=-1)[..., 0]
    expected, scores = _reference(x, p, q, keys, lens, 31, k, visibility=visibility,
                                  candidates=keep, candidates_are_blocks=False)
    actual = index_scores_topk(x, p, q, keys, lens, 31, k, visibility=visibility,
                               candidates=keep, candidates_are_blocks=False)
    mx.eval(expected, actual, scores)
    _assert_selection(actual, expected, scores, 31)


@pytest.mark.parametrize("tie", ["zero", "duplicate_keys", "negative", "masked"])
def test_ties_and_unreachable(tie):
    x, p, q, keys, _, _, keep = _inputs(16, 1031, True)
    lens = mx.array([0, 1, 7, 8, 9, 37, 511, 512, 513, 600, 700, 800, 900, 1024, 1030, 1031])[:, None]
    visibility = mx.arange(1031)[None] < lens
    if tie == "zero":
        q = mx.zeros_like(q)
    elif tie == "duplicate_keys":
        keys = mx.broadcast_to(keys[:, :1], keys.shape)
    elif tie == "negative":
        x, p, q, keys = mx.ones_like(x), -mx.ones_like(p), mx.ones_like(q), mx.ones_like(keys)
    else:
        keep = mx.zeros_like(keep)
    expected, scores = _reference(x, p, q, keys, lens, 128, 512,
                                  visibility=visibility, candidates=keep)
    # No visibility buffer: lens alone must impose exactly the same causality.
    actual = index_scores_topk(x, p, q, keys, lens, 128, 512, candidates=keep)
    mx.eval(expected, actual, scores)
    _assert_selection(actual, expected, scores, 128)
    if tie == "masked":
        np.testing.assert_array_equal(np.array(actual), -np.ones((1, 16, 512), dtype=np.int32))


def test_unbatched_small_context():
    x, p, q, keys, lens, visibility, _ = _inputs(16, 37, False)
    expected, scores = _reference(x, p, q, keys, lens, 0, 37, visibility=visibility)
    actual = index_scores_topk(x[0], p, q[0], keys[0], lens, 0, 37)
    assert actual.shape == (16, 37)
    _assert_selection(actual[None], expected, scores, 0)


@pytest.mark.parametrize("context", [1023, 1024, 1025, 2048, 3079])
@pytest.mark.parametrize("k", [1, 37, 512])
def test_tile_boundaries_and_rejected_tile_maxima(context, k):
    # Unique, exactly representable scores: tile 1 loses to tile 0, while tile
    # 2 wins. Threshold rejection must preserve tile 1's candidate maxima.
    rank = np.arange(context, dtype=np.float32) + 1
    rank[:1024] += 10000
    rank[2048:] += 20000
    x, p = mx.ones((1, 2, 64)), mx.ones((32, 64)) / 64
    q = mx.ones((1, 2, 32, 128))
    keys = mx.broadcast_to(mx.array(rank)[None, :, None], (1, context, 128))
    lens = mx.array([context, context - 7], dtype=mx.int32)[:, None]
    visibility = mx.arange(context)[None] < lens
    actual, blocks = index_scores_topk(x, p, q, keys, lens, 31, k,
                                       return_block_scores=True)
    expected, scores = _reference(x, p, q, keys, lens, 31, k, visibility=visibility)
    mx.eval(actual, blocks, expected, scores)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    padded = np.pad(np.array(scores), ((0, 0), (0, 0), (0, (-context) % 8)),
                    constant_values=-np.inf)
    np.testing.assert_array_equal(np.array(blocks), padded.reshape(1, 2, -1, 8).max(-1))


@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("context", [37, 1031, 30000])
def test_candidate_source_maxima_and_pinned_partial_block(precompute, context):
    x, p, q, keys, _, _, _ = _inputs(16, context, False)
    lens = mx.array([0, 1, 7, 8, 9, 15, 16, 17, 25, 31, 32, 33,
                     context - 3, context - 2, context - 1, context])[:, None]
    visibility = mx.arange(context)[None] < lens
    k = min(512, context)
    actual, blocks = index_scores_topk(x, p, q, keys, lens, 128, k,
                                       visibility=visibility, return_block_scores=True)
    expected, scores = _reference(x, p, q, keys, lens, 128, k, visibility=visibility)
    expected_keep = select_candidate_blocks(scores, lens, 5, 8, block_mask=precompute)
    block_lens = (lens - 1) // 8 + 1
    keep = select_candidate_blocks(blocks, block_lens, 5, 1, block_mask=True)
    if not precompute:
        keep = mx.repeat(keep, 8, axis=-1)[..., :context]
    mx.eval(actual, expected, scores, keep, expected_keep)
    _assert_selection(actual, expected, scores, 128)
    np.testing.assert_array_equal(np.array(keep), np.array(expected_keep))
    # At lens=0 all maxima are correctly -inf; pinning alone retains block 0
    # on MLX 0.32.2 (integer // truncates). Derive the expected pin on-device
    # so this remains valid if a later MLX version adopts floor division.
    last = np.array((lens - 1) // 8).reshape(-1)
    for qi in range(5):  # explicit regression gates: lens={0,1,7,8,9}
        if last[qi] >= 0:
            pin = last[qi] if precompute else last[qi] * 8
            assert np.array(keep)[0, qi, pin]
    assert np.isneginf(np.array(blocks)[0, 0]).all()
    assert (np.array(actual)[0, 0] == -1).all()
    pad = (-context) % 8
    padded = mx.concatenate([scores, mx.full((1, 16, pad), -float("inf"))], axis=-1) if pad else scores
    np.testing.assert_allclose(np.array(blocks), np.array(padded.reshape(1, 16, -1, 8).max(-1)),
                               rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("all_candidates", [False, True])
def test_indexer_source_and_consumer_hook(monkeypatch, precompute, all_candidates):
    lang = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    monkeypatch.setenv("DSV41_S4_PRECOMPUTE", str(int(precompute)))
    monkeypatch.setenv("DSV41_S4_BYPASS", "0")
    monkeypatch.setenv("DSV41_METAL_INDEXER", "0")
    monkeypatch.setenv("DSV41_S4_COMPILE_INDEXER", "0")
    # Exercise the production n>=64 gate, including zero-length geometry.
    monkeypatch.setenv("DSV41_PREFILL_TILED_MIN_N", "64")
    monkeypatch.setattr(lang, "_rope", lambda rope, q, pos: q)
    monkeypatch.setattr(lang, "_PREFILL_MXFP8_STEEL", False)
    mx.random.seed(71)
    args = TextConfig(hidden_size=64, q_lora_rank=64, candidate_topk_blocks=256 if all_candidates else 5)
    # Compression by two gives a genuine lens=0 at start_pos=0.
    args.compress_ratios[20] = args.compress_ratios[24] = 2
    source, consumer = Indexer(args, 20), Indexer(args, 24)
    n = 64
    x, _, _, keys, _, _, _ = _inputs(n, 1031, False)
    qr = mx.random.normal((1, n, 64))
    reference, fused = SimpleNamespace(candidates=None), SimpleNamespace(candidates=None)
    monkeypatch.setenv("DSV41_PREFILL_TILED_INDEX", "0")
    e0 = source(x, qr, 0, 128, None, keys, reference)
    e1 = consumer(x, qr, 0, 128, None, keys, reference)
    monkeypatch.setenv("DSV41_PREFILL_TILED_INDEX", "1")
    tiled = importlib.import_module("mlx_vlm.custom_kernels.deepseek_v41.index_prefill")
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs["return_block_scores"])
        return index_scores_topk(*args, **kwargs)

    monkeypatch.setattr(tiled, "index_scores_topk", tracked)
    a0 = source(x, qr, 0, 128, None, keys, fused)
    a1 = consumer(x, qr, 0, 128, None, keys, fused)
    assert calls == [not (precompute and all_candidates), False]
    mx.eval(a0, a1, e0, e1)
    lens = ((mx.arange(n, dtype=mx.int32) + 1) // 2)[:, None]
    visibility = mx.arange(1031)[None] < lens
    for indexer, actual, expected, mask in [(source, a0, e0, None),
                                           (consumer, a1, e1, reference.candidates)]:
        q = fake_quant_fp4_ue8m0(indexer.wq_b(qr).reshape(1, n, 32, 128))
        _, scores = _reference(x, indexer.weights_proj.weight, q, keys, lens, 128, 512,
                               visibility=visibility, candidates=mask,
                               candidates_are_blocks=precompute)
        _assert_selection(actual, expected, scores, 128)
    if precompute and all_candidates:
        assert fused.candidates is reference.candidates is None
    else:
        np.testing.assert_array_equal(np.array(fused.candidates), np.array(reference.candidates))


def test_switch_default_off_and_decode_untouched(monkeypatch):
    lang = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    tiled = importlib.import_module("mlx_vlm.custom_kernels.deepseek_v41.index_prefill")

    def forbidden(*args, **kwargs):
        pytest.fail("tiled kernel called when disabled or for decode")

    monkeypatch.setattr(tiled, "index_scores_topk", forbidden)
    monkeypatch.setattr(lang, "_rope", lambda rope, q, pos: q)
    monkeypatch.setattr(lang, "_PREFILL_MXFP8_STEEL", False)
    monkeypatch.setenv("DSV41_S4_BYPASS", "0")
    monkeypatch.setenv("DSV41_METAL_INDEXER", "0")
    monkeypatch.setenv("DSV41_S4_COMPILE_INDEXER", "0")
    indexer = Indexer(TextConfig(hidden_size=64, q_lora_rank=64), 2)
    monkeypatch.setenv("DSV41_PREFILL_TILED_MIN_N", "64")
    for n, enabled in [(64, None), (1, "1"), (16, "1"), (63, "1")]:
        if enabled is None:
            monkeypatch.delenv("DSV41_PREFILL_TILED_INDEX", raising=False)
        else:
            monkeypatch.setenv("DSV41_PREFILL_TILED_INDEX", enabled)
        x, _, _, keys, _, _, _ = _inputs(n, 1031, False)
        mx.eval(indexer(x, x, 2062 - n, 128, None, keys, SimpleNamespace(candidates=None)))


@contextmanager
def _environment(**settings):
    old = {key: os.environ.get(key) for key in settings}
    os.environ.update({key: str(value) for key, value in settings.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _model_timing_paths(n, context, mask, keys, keep):
    """Call production Indexer.__call__, not a second copy of its chunk loop.

    Production projection widths; synthetic weights/inputs and consumer masks.
    Layer 24 and source layer 20 both use ratio=1, so N and lens match the
    isolated pair. This is an indexer benchmark, not a loaded-model benchmark.
    """
    lang = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    config = TextConfig()
    indexer = Indexer(config, 24)
    source = Indexer(config, 20) if mask else None
    x = mx.random.normal((1, n, config.hidden_size))
    qr = mx.random.normal((1, n, config.q_lora_rank))
    rope = lang.DeepseekV4RoPE(config.qk_rope_head_dim, config.compress_rope_theta,
                             dict(config.rope_scaling))
    mx.eval(x, qr, indexer.parameters(), rope.parameters())
    if source is not None:
        mx.eval(source.parameters())

    def run(module, enabled):
        shared = SimpleNamespace(candidates=keep if module.uses_candidates else None)
        with _environment(DSV41_PREFILL_TILED_INDEX=int(enabled)):
            out = module(x, qr, context - n, 128, rope, keys, shared)
        # Candidate-source side output must be evaluated as the model pays it.
        return ((out, shared.candidates) if module.is_candidate_source
                and shared.candidates is not None else (out,))

    paths = {
        "model_reference": lambda: run(indexer, False),
        "model_tiled": lambda: run(indexer, True),
    }
    if source is not None:
        paths.update({
            "source_reference": lambda: run(source, False),
            "source_tiled": lambda: run(source, True),
        })
    return paths


def timing_helper(n=256, context=30000, mask=True, repeats=5, workspace_mb=512):
    if repeats < 1 or workspace_mb < 0:
        raise ValueError("repeats must be positive and workspace_mb nonnegative")
    # Explicitly measure the eager reference from the report, restoring the
    # caller's environment afterwards. Each path uses the same quantisation.
    with _environment(DSV41_INDEX_WORKSPACE_MB=workspace_mb, DSV41_INDEX_TOPK=512,
                      DSV41_METAL_INDEXER=0, DSV41_S4_COMPILE_INDEXER=0,
                      DSV41_S4_PRECOMPUTE=1, DSV41_S4_BYPASS=0,
                      DSV41_PREFILL_TILED_MIN_N=64, DSV41_QAT=1,
                      DSV41_METAL_QUANT=0):
        return _timing_helper(n, context, mask, repeats, workspace_mb)


def _timing_helper(n, context, mask, repeats, workspace_mb):
    """Fresh graphs, warmed inputs/JIT, rotating path order to limit drift."""
    x, p, q, keys, lens, visibility, keep = _inputs(n, context, mask)
    paths = {
        "isolated_reference": lambda: (_reference(x, p, q, keys, lens, 128, 512,
            visibility=visibility, candidates=keep, workspace_mb=workspace_mb)[0],),
        "isolated_tiled": lambda: (index_scores_topk(x, p, q, keys, lens, 128, 512,
            visibility=visibility, candidates=keep),),
    }
    # The default hook gate excludes n<64; never label an eager call 'tiled'.
    if n >= 64:
        paths.update(_model_timing_paths(n, context, mask, keys, keep))
    for fn in paths.values():
        for _ in range(2):
            mx.eval(*fn())
    samples = {name: [] for name in paths}
    names = list(paths)
    for rep in range(repeats):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for name in (order if rep % 2 == 0 else list(reversed(order))):
            start = time.perf_counter()
            mx.eval(*paths[name]())
            samples[name].append((time.perf_counter() - start) * 1000)
    result = {name: float(np.median(ms)) for name, ms in samples.items()}
    chunk = min(n, max(1, workspace_mb * 1024**2 // (32 * context * 4))) if workspace_mb else n
    print(f"n={n} compressed_keys={context} candidates={mask} workspace_mb={workspace_mb} "
          f"query_chunk={chunk} chunks={(n + chunk - 1) // chunk} "
          f"headwise_chunk_MiB={32 * chunk * context * 4 / 1024**2:.1f} "
          f"reduced_rows_MiB={n * context * 4 / 1024**2:.1f}")
    print("Eager reference; mask/top-k once after score concatenation; no per-chunk eval. "
          "model_* includes production-width query preparation; source_* also evaluates candidates. "
          "Synthetic isolated indexer: excludes other layers and their memory pressure.")
    print(f"Median ms: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", required=True)
    parser.add_argument("--n", type=int, default=256, choices=[16, 256, 2048])
    parser.add_argument("--context", type=int, default=30000, choices=[6000, 30000])
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workspace-mb", type=int, default=512)
    parser.add_argument("--candidates", action="store_true", help="explicit alias for the default")
    parser.add_argument("--model-matrix", action="store_true",
                        help="run n=2048, N=6000/30000, candidates off/on")
    args = parser.parse_args()
    if args.candidates and args.no_candidates:
        parser.error("--candidates and --no-candidates are mutually exclusive")
    if args.model_matrix:
        for context in (6000, 30000):
            for mask in (False, True):
                timing_helper(2048, context, mask, args.repeats, args.workspace_mb)
    else:
        timing_helper(args.n, args.context, not args.no_candidates, args.repeats, args.workspace_mb)
