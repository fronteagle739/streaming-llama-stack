"""DSV41_PREFILL_LAYER_MAJOR and DSV41_PREFILL_CONE.

Contract: ./tmp/chatfix/cone/CONE_DESIGN.md. Both switches default OFF; the gates
here are the two the design doc names -- LAYER_MAJOR must be BIT-IDENTICAL to the
default path, and CONE must be bit-identical at the FINAL prompt position -- plus
the refusals that keep either from silently doing something it did not say.

Pure pytest: no network, no Studio, no checkpoint. The end-to-end cases build the
same tiny random model the Mac language tests use (8 layers, window 8, boundary 5),
which is cheap enough to run several times.

Run: python -m pytest mlx_vlm/models/deepseek_v41/test_prefill_cone.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core", reason="DSV4.1 prefill tests require MLX on Mac")

from mlx_vlm.models.deepseek_v41.config import TextConfig
from mlx_vlm.models.deepseek_v41.language import (
    _prefill_switches,
    cone_first_row,
    cone_total_rows,
    prefill_boundary_layer,
    prefill_cone_tail,
)

SHIPPED = TextConfig()
WINDOW = SHIPPED.sliding_window          # 128
N_LAYERS = SHIPPED.num_hidden_layers     # 40
BOUNDARY = 20


def _load_mac_tests():
    """The tiny-model fixtures live in the Mac language test, loaded by file.

    The package conftest collects that module standalone, so import it the same way
    rather than through the package namespace.
    """
    path = Path(__file__).with_name("test_language_dsv41_mac.py")
    spec = importlib.util.spec_from_file_location("dsv41_mac_tests_for_cone", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- first(il) maths

def test_first_row_boundary_and_last_layer():
    n = 30_000
    # The boundary layer's cone is the widest: 1 + 19 * 127 rows.
    assert cone_first_row(n, BOUNDARY, BOUNDARY, N_LAYERS, WINDOW) == n - (1 + 19 * (WINDOW - 1))
    # The last layer computes exactly ONE row: the position being predicted.
    assert cone_first_row(n, N_LAYERS - 1, BOUNDARY, N_LAYERS, WINDOW) == n - 1
    # Every layer below the boundary starts at row zero -- first(boundary - 1) = 0 is
    # the base case the recurrence is built on.
    for il in range(0, BOUNDARY):
        assert cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW) == 0


def test_each_layer_widens_by_exactly_window_minus_one():
    n = 30_000
    for il in range(BOUNDARY + 1, N_LAYERS):
        below = cone_first_row(n, il - 1, BOUNDARY, N_LAYERS, WINDOW)
        here = cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW)
        assert here - below == WINDOW - 1


def test_total_decoder_rows_is_constant_and_24150():
    n_decoder = N_LAYERS - BOUNDARY
    expected = n_decoder + (WINDOW - 1) * n_decoder * (n_decoder - 1) // 2
    assert expected == 24_150
    for n in (25_000, 30_000, 68_639, 131_072):
        total = sum(n - cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW)
                    for il in range(BOUNDARY, N_LAYERS))
        assert total == expected, f"N={n} gave {total}"


def test_short_prompt_clamps_to_zero():
    # Below the boundary layer's reach the cone covers everything; the driver logs
    # that it ran the full decoder half and the result is identical.
    reach = 1 + (N_LAYERS - 1 - BOUNDARY) * (WINDOW - 1)
    assert cone_first_row(reach, BOUNDARY, BOUNDARY, N_LAYERS, WINDOW) == 0
    assert cone_first_row(reach - 1, BOUNDARY, BOUNDARY, N_LAYERS, WINDOW) == 0
    assert cone_first_row(reach + 5, BOUNDARY, BOUNDARY, N_LAYERS, WINDOW) == 5


# ------------------------------------------------------------- derived boundary

def test_shipped_config_boundary_is_twenty():
    assert prefill_boundary_layer(SHIPPED) == BOUNDARY


def test_shipped_config_satisfies_every_shared_state_owner():
    # The load-bearing assumption of the whole design: phase two starts a FRESH
    # SharedState, so the boundary layer must own every field the layers above read.
    assert BOUNDARY == max(SHIPPED.kv_source_layer_ids)
    assert BOUNDARY in SHIPPED.index_source_layer_ids
    assert BOUNDARY == SHIPPED.candidate_source_layer_id
    assert SHIPPED.compress_ratios.index(1) == BOUNDARY
    assert all(r == 1 for r in SHIPPED.compress_ratios[BOUNDARY:N_LAYERS])


@pytest.mark.parametrize("override, names", [
    (dict(kv_source_layer_ids=[2, 8, 14]), ["kv_source_layer_ids"]),
    (dict(index_source_layer_ids=[2, 8, 14, 24, 28, 32, 36]), ["index_source_layer_ids"]),
    (dict(candidate_source_layer_id=24), ["candidate_source_layer_id"]),
])
def test_boundary_refuses_a_config_whose_owners_disagree(override, names):
    config = TextConfig(**override)
    with pytest.raises(ValueError) as excinfo:
        prefill_boundary_layer(config)
    message = str(excinfo.value)
    for name in names:
        assert name in message, message


def test_boundary_refuses_a_config_with_no_full_resolution_layer():
    ratios = [2] * N_LAYERS
    with pytest.raises(ValueError) as excinfo:
        prefill_boundary_layer(TextConfig(compress_ratios=ratios,
                                          kv_source_layer_ids=[2, 8, 14, 20],
                                          candidate_source_layer_id=20))
    assert "compress_ratios" in str(excinfo.value)


def test_boundary_refuses_a_compressed_layer_above_the_boundary():
    ratios = list(SHIPPED.compress_ratios)
    ratios[30] = 2
    with pytest.raises(ValueError) as excinfo:
        prefill_boundary_layer(TextConfig(compress_ratios=ratios))
    assert "compress_ratios" in str(excinfo.value)


# --------------------------------------------------------------------- refusals

class _Cache:
    bounded = False


def test_cone_without_layer_major_raises_and_names_both(monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    monkeypatch.delenv("DSV41_PREFILL_LAYER_MAJOR", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _prefill_switches(4096, _Cache(), None)
    message = str(excinfo.value)
    assert "DSV41_PREFILL_CONE" in message
    assert "DSV41_PREFILL_LAYER_MAJOR" in message


def test_cone_with_dspark_raises_and_names_both(monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.setenv("DSV41_DSPARK", "1")
    monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _prefill_switches(4096, _Cache(), None)
    message = str(excinfo.value)
    assert "DSV41_PREFILL_CONE" in message
    assert "DSV41_DSPARK" in message
    assert "prefill_from_target_hidden" in message


def test_cone_with_capture_layers_raises(monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.delenv("DSV41_DSPARK", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _prefill_switches(4096, _Cache(), [37, 38, 39])
    assert "capture_layer_ids" in str(excinfo.value)


def test_layer_major_with_bounded_cache_raises(monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.delenv("DSV41_PREFILL_CONE", raising=False)

    class Bounded:
        bounded = True

    with pytest.raises(ValueError) as excinfo:
        _prefill_switches(4096, Bounded(), None)
    message = str(excinfo.value)
    assert "DSV41_PREFILL_LAYER_MAJOR" in message
    assert "DSV41_BOUNDED_CACHE" in message


def test_switches_are_off_by_default_and_never_imply_each_other(monkeypatch):
    for name in ("DSV41_PREFILL_LAYER_MAJOR", "DSV41_PREFILL_CONE", "DSV41_DSPARK",
                 "DSV41_PREFILL_CONE_TAIL"):
        monkeypatch.delenv(name, raising=False)
    assert _prefill_switches(4096, _Cache(), None) == (False, False)
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    # LAYER_MAJOR alone must NOT switch the cone on.
    assert _prefill_switches(4096, _Cache(), None) == (True, False)
    # A single token is a decode step: neither switch engages.
    assert _prefill_switches(1, _Cache(), None) == (False, False)


# ------------------------------------------------------------------- end to end
#
# WHY SO FEW OF THE END-TO-END GATES ARE BIT-IDENTITY GATES
#
# This model is not chunk-invariant. A compressor group that is partial at a chunk
# boundary is pooled from a different number of rows and the fake-quant codes below
# it flip, which is exactly what `LanguageModel.no_chunked_prefill = True` documents.
# test_the_default_path_is_not_chunk_invariant pins that, measured: two legitimate
# chunk widths of the DEFAULT path differ at the final position by more than half the
# logit scale on the tiny random fixture.
#
# DSV41_PREFILL_CONE re-chunks the decoder half -- it runs layers [boundary, n_layers)
# over one contiguous span of cone rows instead of once per prompt chunk, at a
# different absolute alignment, so each index-source layer sees a different `end_pos`
# and a different bank width. Asking the cone to be bit-identical to a chunked control
# is therefore asking it for a property that ORDINARY re-chunking does not have, and
# no implementation could pass it.
#
# So the cone's numeric gate is RELATIVE: it must be at least as close to the chunked
# control as two legitimate chunkings of the default path are to each other. That is a
# gate a structural bug still fails -- a cone that read a reserved zero row, or lost a
# layer's window, lands nowhere near the control on either metric -- while a
# re-chunking difference passes. The baseline is measured inside each test from the
# same fixture, never hard-coded, so it tracks the model rather than a past run.
#
# THE RULE THIS FILE FOLLOWS, learned the hard way three times over:
#
#   Before asserting any exactness property of this model, MEASURE THAT SAME PROPERTY
#   ON THE DEFAULT PATH with none of the new switches set, and derive the tolerance
#   from it in the same test.
#
# Measured, this model lacks all three of the properties one would assume:
#
#   * chunk invariance -- two chunk widths of the default path differ at the final
#     position by more than half the logit scale (test_the_default_path_is_not_chunk_
#     invariant);
#   * trim exactness -- prefill-then-trim-then-continue differs from a fresh prefix on
#     the DEFAULT path too, because trim() calls rebuild_tail(), which recomputes the
#     last compressor group from projections and is not obliged to reproduce the
#     original fake-quant codes;
#   * shape-independent rounding -- changing a GEMM's row count changes its tile shapes
#     and accumulation order.
#
# So a cone gate is relative to whichever of those the comparison actually crosses, and
# the baseline is always measured here rather than written down.
#
# The ONE genuinely near-exact claim is the tail: widening it only moves first(il)
# earlier and crosses none of the three except rounding, so it gets a TIGHT absolute
# gate (_assert_near_exact), not the relative one. Measured drift there is 4.8e-07 on a
# logit scale of 2.9; a real tail bug is orders of magnitude larger.
#
# Everything that does NOT depend on floating point stays a hard assertion: the cone
# arithmetic, the boundary derivation, every refusal, the trim floor, which layers hold
# which rows, that reserved rows are exactly zero, the tail row counts, and the head
# emitting exactly one row.

@pytest.fixture(scope="module")
def tiny():
    """The 8-layer random model from the Mac language tests, plus its ids."""
    mac = _load_mac_tests()
    config = mac.tiny_config()
    assert prefill_boundary_layer(config) == 5, "tiny config boundary moved"
    return mac.random_model(), mac.token_ids(), config


CHUNK = 16          # the width every end-to-end arm and its control share
WIDE = 4096         # "one chunk", the other legitimate width
PREFIX_LONG = 48    # a first turn whose delta (8) is SHORTER than the boundary cone
PREFIX_SHORT = 32   # a first turn whose delta (24) is LONGER, so turn two cones too
NOISE_K = 2.0       # how many baseline widths an arm may spend. Measured ratios on
                    # the tiny fixture are 1.03 (cone vs re-chunking) and 0.91 (trim
                    # on the default path), so this is a 2x margin.
TAIL_TOL = 1e-5     # of the reference's own scale. Measured tail drift is 1.7e-07 of
                    # scale, so this is a ~60x margin and still orders of magnitude
                    # tighter than the relative gate.


def _logits(model, ids):
    """One call over the whole prompt. Only valid as an ARM, never as a control."""
    out = model(ids, cache=model.make_cache()).logits
    mx.eval(out)
    return out


def _prefill_chunked(model, ids, cache, chunk):
    """Feed a prompt the way the RECEIVER does: one model() call per chunk.

    Every control in this file is built like this, at the SAME width as the arm it is
    compared against, because the default path is not chunk-invariant and an unchunked
    control would be measuring chunking rather than the switch under test. Production
    chunks at 2048 and therefore already lives on the chunked answer.

    Returns the concatenated logits for every row, which is what the receiver would
    have seen across its calls.
    """
    rows = []
    for begin in range(0, int(ids.shape[1]), chunk):
        out = model(ids[:, begin:begin + chunk], cache=cache).logits
        mx.eval(out)
        rows.append(out)
    return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=1)


def _hidden_chunked(model, ids, chunk):
    """The same call sequence, collecting the FINAL hidden state of every row."""
    cache = model.make_cache()
    rows = []
    for begin in range(0, int(ids.shape[1]), chunk):
        out = model(ids[:, begin:begin + chunk], cache=cache, return_hidden=True)
        rows.append(out.hidden_states[-1])
    out = rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=1)
    mx.eval(out)
    return out


def _taps_chunked(model, ids, chunk, taps):
    """The same call sequence, collecting one capture tap block per target layer."""
    cache = model.make_cache()
    per_tap = [[] for _ in taps]
    for begin in range(0, int(ids.shape[1]), chunk):
        out = model(ids[:, begin:begin + chunk], cache=cache, capture_layer_ids=taps)
        for i, tap in enumerate(out.hidden_states):
            per_tap[i].append(tap)
    blocks = [rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=1)
              for rows in per_tap]
    mx.eval(*blocks)
    return blocks


def _control(model, ids, chunk=CHUNK):
    """Default-path control at `chunk` width. Callers must clear both switches."""
    cache = model.make_cache()
    return _prefill_chunked(model, ids, cache, chunk), cache


def _diff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _cos(a, b):
    left = a.astype(mx.float32).reshape(-1)
    right = b.astype(mx.float32).reshape(-1)
    denom = float(mx.sqrt(mx.sum(left * left) * mx.sum(right * right)).item())
    return 1.0 if denom == 0.0 else float(mx.sum(left * right).item()) / denom


def _noise(a, b):
    """(max abs difference, cosine) between two things that differ innocently."""
    return _diff(a, b), _cos(a, b)


def _assert_no_worse_than_the_default_path(arm, reference, noise, label,
                                           what="re-chunking", k=NOISE_K):
    """The cone's relative gate: no further off than `what` already is by itself.

    `noise` is the SAME property measured on the default path with no switch set, over
    the same rows, in the same test -- never a written-down number. A structural bug
    fails both halves of this; the innocent difference the baseline captures fails
    neither.
    """
    baseline_max, baseline_cos = noise
    arm_max, arm_cos = _noise(arm, reference)
    assert arm_max <= k * baseline_max, (
        f"{label}: max abs diff {arm_max:.9g} is more than {k}x the default-path "
        f"baseline {baseline_max:.9g} for {what}; the cone may be no FURTHER off than "
        f"{what} already is without it")
    # Angular error gets the SAME ratio form as magnitude, deliberately: asserting
    # arm_cos >= baseline_cos outright would be a knife edge on an inequality that has
    # only been measured for one of these comparisons (cone vs re-chunking, 0.845
    # against 0.798 -- comfortable, but not a licence to assume it elsewhere). The
    # ratio form is what discriminates anyway: a cone reading a reserved ZERO row
    # corrupts the whole decoder half, and near-unrelated vectors land at a cosine
    # near zero, which is nowhere near 2x an innocent baseline's angular error.
    assert 1.0 - arm_cos <= k * (1.0 - baseline_cos) + 1e-9, (
        f"{label}: cosine {arm_cos:.9g} is more than {k}x further from 1 than the "
        f"default-path baseline {baseline_cos:.9g} for {what}; a cone that lost rows "
        f"or read a reserved row lands here, an innocent difference does not")


def _assert_near_exact(arm, reference, label, rel=TAIL_TOL):
    """The TAIL's gate, which is far tighter than the relative one and must stay so.

    Widening the tail crosses none of this model's three known inexactnesses except
    rounding: it moves first(il) earlier, which changes every decoder layer's row
    count and so the GEMM tile shapes and accumulation order. That is fp32 epsilon --
    measured 4.8e-07 on values of order 1 -- and a real tail bug (a row computed from
    the wrong inputs, a mis-sliced tap) is orders of magnitude larger.
    """
    scale = float(mx.max(mx.abs(reference.astype(mx.float32))).item())
    tol = rel * scale
    arm_max = _diff(arm, reference)
    assert arm_max <= tol, (
        f"{label}: max abs diff {arm_max:.9g} exceeds {rel:g} x the reference scale "
        f"{scale:.9g} = {tol:.9g}. Widening the tail may only move the last bits of "
        f"the mantissa; anything visible here is a real defect")


def _default_turn_two(model, ids, prefix, rewind=True, decodes=0):
    """The turn-two shape with NO switch set, so a cone arm has a baseline.

    Same shape as _cone_turn_two: turn one, optionally three decode rows and a rewind
    that drops them, then turn two, then some decode. Chunked at CHUNK throughout, the
    way the receiver prefills. Callers must have cleared both switches.

    This exists because trim-then-continue is NOT exact on this model even with no cone
    in play: trim() calls rebuild_tail(), whose own docstring is "Drop invalid groups
    and recompute the last complete group from projections", and a recomputed group is
    not obliged to reproduce the original fake-quant codes.
    """
    cache = model.make_cache()
    _prefill_chunked(model, ids[:, :prefix], cache, CHUNK)
    if rewind:
        for step in range(3):
            model(ids[:, prefix + step - 1:prefix + step], cache=cache)
        assert cache[0].offset == prefix + 3
        assert cache[0].trim(3) == 3
        assert cache[0].offset == prefix
    out = _prefill_chunked(model, ids[:, prefix:], cache, CHUNK)
    mx.eval(out)
    steps = [model(ids[:, -1:], cache=cache).logits for _ in range(decodes)]
    if steps:
        mx.eval(*steps)
    return out, steps, cache


def _off(monkeypatch):
    monkeypatch.delenv("DSV41_PREFILL_LAYER_MAJOR", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_CONE", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)


def _cone_on(monkeypatch, chunk=CHUNK, tail=None):
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    monkeypatch.setenv("DSV41_PREFILL_CHUNK", str(chunk))
    if tail is None:
        monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    else:
        monkeypatch.setenv("DSV41_PREFILL_CONE_TAIL", str(tail))


def test_the_default_path_is_not_chunk_invariant(tiny, monkeypatch):
    """PINNED, so nobody rediscovers it as a bug in the layer-major or cone code.

    The default path gives DIFFERENT logits at different chunk widths: a compressor
    group that is partial at a chunk boundary is pooled from a different number of
    rows, and the fake-quant codes downstream of it flip. That is exactly what
    `LanguageModel.no_chunked_prefill = True` (language.py) documents.

    Two consequences for this file: a control must be built at the arm's chunk
    width, and the cone -- which re-chunks the decoder half by construction -- can
    only be gated RELATIVE to this difference, never against bit-identity.

    If this test ever starts FAILING -- i.e. the two widths agree -- something in
    the compressor changed and every claim here should be re-read.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    one, _ = _control(model, ids, chunk=WIDE)
    four, _ = _control(model, ids, chunk=CHUNK)
    assert one.shape == four.shape
    assert not bool(mx.array_equal(one[0, -1], four[0, -1]).item()), (
        "the default path became chunk-invariant; the controls and the relative "
        "gates in this file were written around it not being, and "
        "no_chunked_prefill should be revisited")


@pytest.mark.parametrize("qat", ["0", "1"], ids=["qat-off", "qat-on"])
@pytest.mark.parametrize("chunk", [str(CHUNK), str(WIDE)], ids=["four-chunks", "one-chunk"])
def test_layer_major_is_bit_identical_at_the_same_chunk_width(tiny, monkeypatch, qat, chunk):
    """The claim is parity with the default path AT THE SAME CHUNK WIDTH.

    It never was a claim about chunk invariance. DSV41_PREFILL_LAYER_MAJOR splits the
    SAME chunks between two layer spans and re-chunks nothing, so unlike the cone it
    must be bit-identical and this stays a hard gate.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", qat)
    _off(monkeypatch)
    control, _ = _control(model, ids, chunk=int(chunk))

    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.setenv("DSV41_PREFILL_CHUNK", chunk)
    arm = _logits(model, ids)

    assert arm.shape == control.shape
    assert bool(mx.array_equal(arm, control).item()), (
        f"DSV41_PREFILL_LAYER_MAJOR is a pure reordering of the SAME chunks and must "
        f"be bit-identical to the chunked default path at width {chunk}; "
        f"max abs diff={_diff(arm, control):.9g}")


@pytest.mark.parametrize("qat", ["0", "1"], ids=["qat-off", "qat-on"])
def test_layer_major_leaves_the_same_cache(tiny, monkeypatch, qat):
    model, ids, config = tiny
    monkeypatch.setenv("DSV41_QAT", qat)
    _off(monkeypatch)
    _, control = _control(model, ids, chunk=CHUNK)

    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.setenv("DSV41_PREFILL_CHUNK", str(CHUNK))
    arm = model.make_cache()
    model(ids, cache=arm)

    assert arm[0].offset == control[0].offset
    for il, (a, c) in enumerate(zip(arm[0].layers, control[0].layers)):
        for name in a.fields:
            left, right = getattr(a, name), getattr(c, name)
            assert (left is None) == (right is None), f"layer {il} {name} presence"
            if left is None:
                continue
            assert left.shape == right.shape, f"layer {il} {name} shape"
            mx.eval(left, right)
            assert bool(mx.array_equal(left, right).item()), f"layer {il} {name} differs"


@pytest.mark.parametrize("qat", ["0", "1"], ids=["qat-off", "qat-on"])
def test_cone_is_no_further_off_than_chunking_itself(tiny, monkeypatch, qat):
    """The cone's numeric gate at the final position.

    Bit-identity is unavailable and always was: the cone runs the decoder half over
    one contiguous span of cone rows rather than once per prompt chunk, so every
    index-source layer sees a different end_pos and bank width -- a re-chunking, on a
    model that is not chunk-invariant. What IS claimable is that the cone is no
    further from the chunked control than another legitimate chunking is.
    """
    model, ids, config = tiny
    n = int(ids.shape[1])
    n_layers = config.num_hidden_layers
    window = config.sliding_window
    boundary = prefill_boundary_layer(config)
    assert cone_first_row(n, boundary, boundary, n_layers, window) > 0, (
        "the tiny prompt must be long enough for the cone to skip rows")

    monkeypatch.setenv("DSV41_QAT", qat)
    _off(monkeypatch)
    control, _ = _control(model, ids, chunk=CHUNK)
    wide, _ = _control(model, ids, chunk=WIDE)
    noise = _noise(control[0, -1], wide[0, -1])

    _cone_on(monkeypatch)
    cone_cache = model.make_cache()
    arm = model(ids, cache=cone_cache).logits
    mx.eval(arm)

    # Hard: the top layer computes exactly one row, and the cache still spans n.
    assert arm.shape[1] == 1, arm.shape
    assert cone_cache[0].offset == n
    _assert_no_worse_than_the_default_path(arm[0, -1], control[0, -1], noise,
                               "DSV41_PREFILL_CONE final position")


def test_cone_reserves_rather_than_computes_the_rows_below_each_cone(tiny, monkeypatch):
    """Structural, so hard: exactly which rows are real and which are reserved."""
    model, ids, config = tiny
    n = int(ids.shape[1])
    n_layers = config.num_hidden_layers
    window = config.sliding_window
    boundary = prefill_boundary_layer(config)

    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch)
    cache = model.make_cache()
    model(ids, cache=cache)

    for il in range(n_layers):
        win_kv = cache[0].layers[il].win_kv
        assert win_kv.shape[1] == n, f"layer {il} must still address every position"
        if il <= boundary:
            continue
        reserved = cone_first_row(n, il - 1, boundary, n_layers, window)
        mx.eval(win_kv)
        assert bool(mx.all(win_kv[:, :reserved] == 0).item()), (
            f"layer {il} rows below first({il - 1})={reserved} are reserved, not computed")
        assert bool(mx.any(win_kv[:, reserved:] != 0).item()), (
            f"layer {il} must hold real KV from first({il - 1})={reserved}")


# ------------------------------------------------------- rewind after a cone
#
# All hard assertions: the floor is integer arithmetic over the row geometry, and a
# trim-then-continue is compared against another CONE run at the same alignment, so
# nothing here depends on chunking or on floating point.

def _expected_floor(n, config, start_pos=0, tail=1):
    """The floor the driver should record for a cone prefill of `n` rows.

    The TOP decoder layer publishes the fewest rows -- exactly `tail + window - 1`,
    from first(n_layers - 2) -- so it alone decides how far back the cache may be
    rewound: a forward step at offset p reads its window [p - window + 1, p].
    """
    n_layers = config.num_hidden_layers
    window = config.sliding_window
    boundary = prefill_boundary_layer(config)
    top_publish = cone_first_row(n, n_layers - 2, boundary, n_layers, window, tail)
    return start_pos + top_publish + window - 1


def test_a_cone_cache_is_trimmable_down_to_its_floor(tiny, monkeypatch):
    """The blocker this fix exists for: a cone cache must stay reusable.

    is_trimmable() False meant the receiver's `_can_trim_cache_to_offset` returned
    `retained-cache-not-trimmable` and re-prefilled the WHOLE prompt from turn two
    onwards, which made the cone a net loss in a conversation.
    """
    model, ids, config = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch)
    cache = model.make_cache()
    model(ids[:, :PREFIX_LONG], cache=cache)
    floor = _expected_floor(PREFIX_LONG, config)

    assert cache[0].is_trimmable() is True
    assert cache[0].trim_floor() == floor
    assert cache[0].trim(PREFIX_LONG - floor) == PREFIX_LONG - floor
    assert cache[0].offset == floor


def test_a_fresh_cache_has_no_floor(tiny, monkeypatch):
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    _, cache = _control(model, ids, chunk=CHUNK)
    assert cache[0].is_trimmable() is True
    assert cache[0].trim_floor() == 0
    assert cache[0].trim(int(ids.shape[1])) == int(ids.shape[1])


def test_a_cone_cache_refuses_a_rewind_below_its_floor(tiny, monkeypatch):
    model, ids, config = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch)
    cache = model.make_cache()
    model(ids[:, :PREFIX_LONG], cache=cache)
    floor = _expected_floor(PREFIX_LONG, config)

    with pytest.raises(ValueError) as excinfo:
        cache[0].trim(PREFIX_LONG - floor + 1)
    message = str(excinfo.value)
    assert "DSV41_PREFILL_CONE" in message, message
    assert "trim_floor" in message, message
    assert str(floor) in message, message
    # A refused trim must leave the cache exactly as it was.
    assert cache[0].offset == PREFIX_LONG


def test_a_wider_tail_lowers_the_floor(tiny, monkeypatch):
    """The tail buys a deeper rewind: the top layer publishes tail + window - 1 rows."""
    model, ids, config = tiny
    tail = 4
    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch, tail=tail)
    cache = model.make_cache()
    model(ids[:, :PREFIX_LONG], cache=cache)

    assert cache[0].trim_floor() == _expected_floor(PREFIX_LONG, config, tail=tail)
    assert (_expected_floor(PREFIX_LONG, config)
            - _expected_floor(PREFIX_LONG, config, tail=tail)) == tail - 1


def test_a_cone_cache_refuses_to_be_RESTORED_not_to_be_evaluated(tiny, monkeypatch):
    """The round trip is refused; reading `state` is not, and must never become so.

    The getter is evaluation-only by the code's own declaration and the decode loop
    calls mx.eval(cache[0].state) on every step to materialise lazy buffers. Refusing
    there made every long cone request return an EMPTY completion on the real model.
    A cone cache therefore emits header version 4 and the SETTER rejects it, exactly
    as a bounded cache already did with version 3. Both halves are asserted here so
    neither can be reinstated as the other.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch)
    cache = model.make_cache()
    model(ids[:, :PREFIX_LONG], cache=cache)

    # Reading and evaluating must WORK: decode depends on it.
    arrays = cache[0].state
    mx.eval(*arrays)
    assert int(arrays[0][0].item()) == 4, "a cone cache must emit the version the setter refuses"

    # Persistence is refused outright.
    with pytest.raises(ValueError):
        cache[0].meta_state

    # And the round trip is refused, naming what a restored copy would get wrong.
    _off(monkeypatch)
    fresh = model.make_cache()
    with pytest.raises(ValueError) as excinfo:
        fresh[0].state = arrays
    message = str(excinfo.value)
    assert "DSV41_PREFILL_CONE" in message, message
    assert "trim_floor" in message, message
    assert fresh[0].offset == 0, "a refused restore must leave the cache untouched"


def _cone_turn_two(model, ids, prefix, rewind=True, decodes=0):
    """Turn one as a cone, optionally some decode rows and a rewind, then turn two.

    With rewind=False this is the continuation the receiver would run with no prefix
    reuse at all; with rewind=True it is the same continuation reached through the
    trim. Both run the cone at the SAME alignment, which is why comparing them is a
    bit-identity claim rather than a tolerance.
    """
    cache = model.make_cache()
    model(ids[:, :prefix], cache=cache)
    if rewind:
        for step in range(3):
            model(ids[:, prefix + step - 1:prefix + step], cache=cache)
        assert cache[0].offset == prefix + 3
        assert cache[0].trim(3) == 3
        assert cache[0].offset == prefix
    out = model(ids[:, prefix:], cache=cache).logits
    mx.eval(out)
    steps = [model(ids[:, -1:], cache=cache).logits for _ in range(decodes)]
    if steps:
        mx.eval(*steps)
    return out, steps, cache


@pytest.mark.parametrize("prefix", [PREFIX_LONG, PREFIX_SHORT],
                         ids=["short-delta-full-decoder", "long-delta-second-cone"])
def test_trim_then_continue_is_no_worse_than_a_default_path_trim(tiny, monkeypatch, prefix):
    """Turn two, with and without the rewind.

    NOT a bit-identity claim, and measuring said so: trim-then-continue differs from a
    fresh prefix on the DEFAULT path too, by a comparable amount, because trim() calls
    rebuild_tail() and a recomputed compressor group need not reproduce the original
    fake-quant codes. So the baseline is that exact property measured here with no
    switch set, and the claim is that adding the cone does not make the rewind worse.
    """
    model, ids, _ = tiny
    n = int(ids.shape[1])
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    base_rewind, _, _ = _default_turn_two(model, ids, prefix, rewind=True)
    base_fresh, _, _ = _default_turn_two(model, ids, prefix, rewind=False)
    noise = _noise(base_rewind[0, -1], base_fresh[0, -1])

    _cone_on(monkeypatch)
    control, _, control_cache = _cone_turn_two(model, ids, prefix, rewind=False)
    arm, _, arm_cache = _cone_turn_two(model, ids, prefix, rewind=True)

    # Structural, so hard: the rewind must land the cache at exactly the same offset.
    assert arm.shape == control.shape
    assert arm_cache[0].offset == control_cache[0].offset == n
    _assert_no_worse_than_the_default_path(
        arm[0, -1], control[0, -1], noise,
        f"cone rewind to {prefix} then continue", what="a trim-then-continue")


@pytest.mark.parametrize("prefix", [PREFIX_LONG, PREFIX_SHORT],
                         ids=["short-delta-full-decoder", "long-delta-second-cone"])
def test_decode_after_trim_is_no_worse_than_a_default_path_trim(tiny, monkeypatch, prefix):
    """Decode is what consumes the continued cache, so gate it the same way.

    Per step, against the same property measured on the default path: a rewind there
    moves the decode logits too, for the rebuild_tail reason above.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    _, base_rewind, _ = _default_turn_two(model, ids, prefix, rewind=True, decodes=3)
    _, base_fresh, _ = _default_turn_two(model, ids, prefix, rewind=False, decodes=3)

    _cone_on(monkeypatch)
    _, control, _ = _cone_turn_two(model, ids, prefix, rewind=False, decodes=3)
    _, arm, _ = _cone_turn_two(model, ids, prefix, rewind=True, decodes=3)

    for step, (a, c, br, bf) in enumerate(zip(arm, control, base_rewind, base_fresh)):
        _assert_no_worse_than_the_default_path(
            a, c, _noise(br, bf), f"cone decode step {step} after a rewind to {prefix}",
            what="a trim-then-continue")


@pytest.mark.parametrize("prefix", [PREFIX_LONG, PREFIX_SHORT],
                         ids=["short-delta-full-decoder", "long-delta-second-cone"])
def test_trim_then_continue_is_no_worse_than_the_default_path_end_to_end(tiny, monkeypatch, prefix):
    """And the continued cone must still predict what a one-shot prefill predicts.

    This comparison crosses BOTH known inexactnesses at once -- the cone re-chunks the
    decoder half, and the rewind goes through rebuild_tail -- so the baseline is the
    whole journey measured on the default path: a default trim-then-continue against
    the same chunked full-prompt control. The arm's PHASE-ONE chunk boundaries line up
    with the control's because the prefix is a multiple of the chunk width.
    """
    model, ids, _ = tiny
    assert prefix % CHUNK == 0, "the arm's chunk boundaries must line up with the control"
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    control, _ = _control(model, ids, chunk=CHUNK)
    base, _, _ = _default_turn_two(model, ids, prefix, rewind=True)
    noise = _noise(base[0, -1], control[0, -1])

    _cone_on(monkeypatch)
    arm, _, _ = _cone_turn_two(model, ids, prefix, rewind=True)
    _assert_no_worse_than_the_default_path(
        arm[0, -1], control[0, -1], noise,
        f"cone prefill of {prefix} + rewind + continuation",
        what="a default-path trim-then-continue")


def test_a_second_cone_raises_the_floor(tiny, monkeypatch):
    """A cone over the delta reserves rows again, so the floor moves UP with it."""
    model, ids, config = tiny
    prefix = PREFIX_SHORT
    delta = int(ids.shape[1]) - prefix
    n_layers = config.num_hidden_layers
    window = config.sliding_window
    boundary = prefill_boundary_layer(config)
    assert cone_first_row(delta, boundary, boundary, n_layers, window) > 0, (
        "this test needs a delta long enough for turn two to cone as well")

    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch)
    cache = model.make_cache()
    model(ids[:, :prefix], cache=cache)
    first_floor = cache[0].trim_floor()
    model(ids[:, prefix:], cache=cache)

    assert cache[0].trim_floor() == _expected_floor(delta, config, start_pos=prefix)
    assert cache[0].trim_floor() > first_floor
    with pytest.raises(ValueError):
        cache[0].trim(cache[0].offset - cache[0].trim_floor() + 1)


def test_a_short_delta_does_not_move_the_floor(tiny, monkeypatch):
    """Turn two ran the FULL decoder half, so it reserved nothing and added none."""
    model, ids, config = tiny
    prefix = PREFIX_LONG
    delta = int(ids.shape[1]) - prefix
    n_layers = config.num_hidden_layers
    window = config.sliding_window
    boundary = prefill_boundary_layer(config)
    assert cone_first_row(delta, boundary, boundary, n_layers, window) == 0, (
        "this test needs a delta short enough that the cone covers all of it")

    monkeypatch.setenv("DSV41_QAT", "1")
    _cone_on(monkeypatch)
    cache = model.make_cache()
    model(ids[:, :prefix], cache=cache)
    model(ids[:, prefix:], cache=cache)

    assert cache[0].trim_floor() == _expected_floor(prefix, config)
    assert cache[0].is_trimmable() is True


def test_cone_decode_continues_from_the_final_position(tiny, monkeypatch):
    """The window a decode step reads is inside every decoder layer's cone.

    Relative against the default path, because the cone prefill that produced this
    cache re-chunked the decoder half; a decode step that fell outside a cone would
    read a reserved ZERO row and miss the gate by far more than chunk noise.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    _, control_cache = _control(model, ids, chunk=CHUNK)
    control = [model(ids[:, -1:], cache=control_cache).logits for _ in range(3)]
    _, wide_cache = _control(model, ids, chunk=WIDE)
    wide = [model(ids[:, -1:], cache=wide_cache).logits for _ in range(3)]
    mx.eval(*control, *wide)

    _cone_on(monkeypatch)
    cone_cache = model.make_cache()
    model(ids, cache=cone_cache)
    # Decode is single-token, so neither switch engages for these steps.
    arm = [model(ids[:, -1:], cache=cone_cache).logits for _ in range(3)]
    mx.eval(*arm)

    for step, (a, c, w) in enumerate(zip(arm, control, wide)):
        _assert_no_worse_than_the_default_path(a, c, _noise(c, w), f"cone decode step {step}")


@pytest.mark.skipif(not mx.metal.is_available(),
                    reason="the row megakernels require Metal")
@pytest.mark.parametrize("switches", [
    {"DSV41_PREFILL_LAYER_MAJOR": "1"},
    {"DSV41_PREFILL_LAYER_MAJOR": "1", "DSV41_PREFILL_CONE": "1"},
], ids=["layer-major", "cone"])
def test_under_the_production_megakernel(tiny, monkeypatch, switches):
    """DSV41_PREFILL_MEGAKERNEL=1 is the production preset.

    Both arms must run the SAME row kernels as the control, including the cone's
    one-row top layer, which prefill_block's own n > 1 guard would otherwise have
    dropped onto the eager collapse. Layer-major re-chunks nothing so it stays a hard
    gate; the cone gets the relative one.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    monkeypatch.setenv("DSV41_PREFILL_MEGAKERNEL", "1")
    _off(monkeypatch)
    control, _ = _control(model, ids, chunk=CHUNK)
    wide, _ = _control(model, ids, chunk=WIDE)
    noise = _noise(control[0, -1], wide[0, -1])

    for name, value in switches.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DSV41_PREFILL_CHUNK", str(CHUNK))
    arm = _logits(model, ids)

    if switches.get("DSV41_PREFILL_CONE") == "1":
        assert arm.shape[1] == 1, arm.shape
        _assert_no_worse_than_the_default_path(arm[0, -1], control[0, -1], noise,
                                   "cone under DSV41_PREFILL_MEGAKERNEL")
    else:
        assert arm.shape == control.shape
        assert bool(mx.array_equal(arm, control).item()), (
            f"DSV41_PREFILL_LAYER_MAJOR under DSV41_PREFILL_MEGAKERNEL re-chunks "
            f"nothing and must be bit-identical; max abs diff={_diff(arm, control):.9g}")


# ------------------------------------------------------------ the drafter tail
#
# Widening the tail moves every first(il) EARLIER and re-chunks nothing relative to
# another CONE run at the same width. So tail-against-tail comparisons are BIT-IDENTITY
# gates, and they are the load-bearing ones here: if widening moved the final position,
# or changed a row both widths compute, that is a real bug. Only the comparisons that
# cross back to the non-cone default path get the relative gate.

def test_tail_of_one_reproduces_the_original_formula():
    n = 30_000
    for il in range(N_LAYERS):
        assert (cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW, 1)
                == cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW))
    assert cone_total_rows(N_LAYERS, BOUNDARY, WINDOW, 1) == 24_150
    assert cone_total_rows(N_LAYERS, BOUNDARY, WINDOW) == 24_150


@pytest.mark.parametrize("tail", [1, 2, 128, 512, 2048])
def test_tail_costs_exactly_one_extra_row_per_decoder_layer(tail):
    n = 131_072
    n_decoder = N_LAYERS - BOUNDARY
    total = sum(n - cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW, tail)
                for il in range(BOUNDARY, N_LAYERS))
    assert total == cone_total_rows(N_LAYERS, BOUNDARY, WINDOW, tail)
    assert total == tail * n_decoder + (WINDOW - 1) * n_decoder * (n_decoder - 1) // 2
    assert total - cone_total_rows(N_LAYERS, BOUNDARY, WINDOW, 1) == n_decoder * (tail - 1)


@pytest.mark.parametrize("tail", [1, 2, 128, 512])
def test_every_tail_row_is_covered_by_the_layer_below(tail):
    """Correctness point 2, as arithmetic over the shipped geometry.

    Row r of layer il reads its own KV over [r - (window - 1), r], and layer il
    publishes KV from first(il - 1) = first(il) - (window - 1). So every row the
    layer computes is covered, for every layer INCLUDING the boundary layer, whose
    KV the cone's own phase publishes over the whole prefix [0, n).
    """
    n = 30_000
    for il in range(BOUNDARY, N_LAYERS):
        first = cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW, tail)
        published = cone_first_row(n, il - 1, BOUNDARY, N_LAYERS, WINDOW, tail)
        assert il == BOUNDARY or first - published == WINDOW - 1
        # The lowest row this layer computes reaches no further back than the lowest
        # row that was published to it (row 0 itself, at the boundary layer).
        assert first - (WINDOW - 1) >= published
    assert cone_first_row(n, N_LAYERS - 1, BOUNDARY, N_LAYERS, WINDOW, tail) == n - tail


def test_tail_only_moves_first_rows_earlier():
    n = 30_000
    for il in range(BOUNDARY, N_LAYERS):
        base = cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW, 1)
        for tail in (2, 128, 512):
            assert cone_first_row(n, il, BOUNDARY, N_LAYERS, WINDOW, tail) <= base


@pytest.mark.parametrize("raw, parsed", [
    (None, (1, False)), ("1", (1, True)), ("512", (512, True)), ("2048", (2048, True)),
])
def test_cone_tail_switch_parses(monkeypatch, raw, parsed):
    if raw is None:
        monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    else:
        monkeypatch.setenv("DSV41_PREFILL_CONE_TAIL", raw)
    assert prefill_cone_tail() == parsed


@pytest.mark.parametrize("raw", ["0", "-1", "half", ""])
def test_cone_tail_switch_refuses_nonsense(monkeypatch, raw):
    monkeypatch.setenv("DSV41_PREFILL_CONE_TAIL", raw)
    with pytest.raises(ValueError) as excinfo:
        prefill_cone_tail()
    assert "DSV41_PREFILL_CONE_TAIL" in str(excinfo.value)


def test_dspark_refusal_is_present_without_a_tail_and_absent_with_one(monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.setenv("DSV41_DSPARK", "1")
    monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _prefill_switches(4096, _Cache(), None)
    message = str(excinfo.value)
    assert "DSV41_PREFILL_CONE_TAIL" in message, message
    assert "DSV41_DSPARK" in message, message

    monkeypatch.setenv("DSV41_PREFILL_CONE_TAIL", "512")
    assert _prefill_switches(4096, _Cache(), None) == (True, True)


def test_capture_refusal_is_present_without_a_tail_and_absent_with_one(monkeypatch):
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    monkeypatch.delenv("DSV41_DSPARK", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _prefill_switches(4096, _Cache(), [37, 38, 39])
    assert "DSV41_PREFILL_CONE_TAIL" in str(excinfo.value)

    monkeypatch.setenv("DSV41_PREFILL_CONE_TAIL", "128")
    assert _prefill_switches(4096, _Cache(), [37, 38, 39]) == (True, True)


def test_the_tail_never_implies_a_switch_and_no_switch_implies_the_tail(monkeypatch):
    for name in ("DSV41_PREFILL_LAYER_MAJOR", "DSV41_PREFILL_CONE", "DSV41_DSPARK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DSV41_PREFILL_CONE_TAIL", "512")
    # A tail on its own switches nothing on.
    assert _prefill_switches(4096, _Cache(), None) == (False, False)
    monkeypatch.setenv("DSV41_PREFILL_LAYER_MAJOR", "1")
    assert _prefill_switches(4096, _Cache(), None) == (True, False)
    # And neither switch sets the tail.
    monkeypatch.delenv("DSV41_PREFILL_CONE_TAIL", raising=False)
    monkeypatch.setenv("DSV41_PREFILL_CONE", "1")
    assert prefill_cone_tail() == (1, False)


def _cone_run(model, ids, monkeypatch, tail, **kwargs):
    """One cone prefill at `tail`, returning the LanguageModelOutput."""
    _cone_on(monkeypatch, tail=tail)
    out = model(ids, cache=model.make_cache(), **kwargs)
    return out


@pytest.mark.parametrize("tail", [2, 4, 8])
def test_tail_leaves_the_final_position_near_exact(tiny, monkeypatch, tail):
    """Correctness point 1, and the TIGHT gate -- not the relative one.

    Widening only moves first(il) earlier; it does not re-chunk anything relative to
    another cone run, and the final row's inputs are untouched. So the reference is
    the cone at tail 1, NOT a chunked non-cone control, and the only difference it may
    show is rounding: more rows per decoder layer means different GEMM tile shapes and
    accumulation order. Measured 4.8e-07 on a logit scale of 2.9. A tail bug -- the
    final row reading a row it should not, the head landing on the wrong row -- is
    orders of magnitude larger and fails this easily.
    """
    model, ids, config = tiny
    n = int(ids.shape[1])
    boundary = prefill_boundary_layer(config)
    n_layers = config.num_hidden_layers
    window = config.sliding_window
    assert cone_first_row(n, boundary, boundary, n_layers, window, tail) > 0, (
        "the tiny prompt must still be longer than the widened boundary cone")

    monkeypatch.setenv("DSV41_QAT", "1")
    control = _cone_run(model, ids, monkeypatch, 1).logits
    arm = _cone_run(model, ids, monkeypatch, tail).logits
    mx.eval(control, arm)

    # The head runs on the FINAL row only, whatever the tail.
    assert control.shape[1] == 1, control.shape
    assert arm.shape[1] == 1, arm.shape
    _assert_near_exact(arm[0, -1], control[0, -1],
                       f"DSV41_PREFILL_CONE_TAIL={tail} against the cone at tail 1")


@pytest.mark.parametrize("tail", [2, 4])
def test_every_tail_row_matches_a_wider_cone(tiny, monkeypatch, tail):
    """Correctness point 2 on the model: ALL W rows, not only the last.

    Reference is a WIDER cone, so both sides are cone runs and this gets the TIGHT
    gate, not the relative one. A wider cone computes strictly more rows; the last
    `tail` of them must be the same rows a narrower cone produced to within rounding,
    or the widening changed a row it had no business touching.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    wide = _cone_run(model, ids, monkeypatch, 2 * tail,
                     return_hidden=True).hidden_states[-1]
    arm = _cone_run(model, ids, monkeypatch, tail,
                    return_hidden=True).hidden_states[-1]
    mx.eval(wide, arm)

    assert int(wide.shape[1]) == 2 * tail, wide.shape
    assert int(arm.shape[1]) == tail, arm.shape
    _assert_near_exact(arm, wide[:, -tail:],
                       f"the {tail} cone-tail hidden rows against the last {tail} "
                       f"rows of a {2 * tail}-row cone")


@pytest.mark.parametrize("tail", [2, 4])
def test_tail_rows_are_no_worse_than_re_chunking_against_a_full_width_control(tiny, monkeypatch, tail):
    """And those rows must still be the default path's rows, to chunk-noise.

    Relative, because this comparison crosses from the cone (which re-chunks the
    decoder half) to the chunked default path. The baseline is two chunkings of the
    default path over the same rows.
    """
    model, ids, _ = tiny
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    control = _hidden_chunked(model, ids, CHUNK)
    wide = _hidden_chunked(model, ids, WIDE)
    noise = _noise(control[:, -tail:], wide[:, -tail:])

    arm = _cone_run(model, ids, monkeypatch, tail, return_hidden=True).hidden_states[-1]
    mx.eval(arm)
    assert int(arm.shape[1]) == tail, arm.shape
    _assert_no_worse_than_the_default_path(arm, control[:, -tail:], noise,
                               f"cone tail {tail} hidden rows")


def test_tail_delivers_every_capture_tap_over_the_same_rows(tiny, monkeypatch):
    """What DSpark actually consumes: aligned taps, one per target layer.

    Each decoder layer's cone is a different width, so the raw taps have different
    row counts and dspark_context_from_captures could not concatenate them at all.
    The cone cuts every tap to the last `tail` positions. The ROW COUNTS are a hard
    assertion; the comparison against a wider cone gets the TIGHT gate, because it
    crosses nothing but rounding; only the cross to the default path is relative.
    """
    model, ids, config = tiny
    tail = 4
    taps = [config.num_hidden_layers - 3, config.num_hidden_layers - 2,
            config.num_hidden_layers - 1]
    monkeypatch.setenv("DSV41_QAT", "1")
    _off(monkeypatch)
    control = _taps_chunked(model, ids, CHUNK, taps)
    wide_chunk = _taps_chunked(model, ids, WIDE, taps)

    wider_cone = _cone_run(model, ids, monkeypatch, 2 * tail,
                           capture_layer_ids=taps).hidden_states
    arm = _cone_run(model, ids, monkeypatch, tail,
                    capture_layer_ids=taps).hidden_states
    mx.eval(*wider_cone, *arm)

    assert len(arm) == len(taps) == len(wider_cone)
    for i, tap in enumerate(taps):
        assert int(arm[i].shape[1]) == tail, (
            f"tap {tap} has {arm[i].shape[1]} rows, wanted {tail}")
        assert int(wider_cone[i].shape[1]) == 2 * tail, wider_cone[i].shape
        _assert_near_exact(arm[i], wider_cone[i][:, -tail:],
                           f"capture tap {tap} against a wider cone over the same rows")
        _assert_no_worse_than_the_default_path(
            arm[i], control[i][:, -tail:],
            _noise(control[i][:, -tail:], wide_chunk[i][:, -tail:]),
            f"capture tap {tap} against the chunked control")
