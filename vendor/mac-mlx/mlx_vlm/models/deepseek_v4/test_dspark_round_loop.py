"""Weight-free, mlx-free coverage for the DSpark speculative round-loop bookkeeping.

Exercises the REAL pure helpers in ``mlx_vlm/speculative/dspark_bookkeeping.py``
(the ones the ``dspark.py`` round loop uses for verify length, rollback trim,
context-commit length and next-round position) and simulates a full multi-round
speculative loop to assert the load-bearing invariant that ties the target cache
back to the drafter's ``position``:

    after a round that verifies ``[anchor, *drafts]`` (writes ``1+proposal`` cells)
    and rolls back the rejected tail (``proposal-accepted`` cells), the target
    cache offset equals the next anchor position == the committed context length.

No mlx / numpy / weights, so it runs on the Windows dev box:

  pytest mlx_vlm/models/deepseek_v4/test_dspark_round_loop.py    (Mac)
  python mlx_vlm/models/deepseek_v4/test_dspark_round_loop.py    (any box)

The dspark.py round loop and its ``draft_block``/verify/rollback wiring are also
covered (mlx-gated, mocked target) by the ``_dspark_rounds`` tests in
``mlx_vlm/tests/test_speculative.py``.
"""

import os


def _load_bookkeeping():
    """Import the (mlx-free) bookkeeping module by path so this runs where mlx
    (and therefore ``import mlx_vlm``) is unavailable."""
    try:
        from mlx_vlm.speculative import dspark_bookkeeping as m  # type: ignore

        return m
    except Exception:  # pragma: no cover - exercised only off-Mac
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(
            os.path.join(here, "..", "..", "speculative", "dspark_bookkeeping.py")
        )
        spec = importlib.util.spec_from_file_location("_dspark_bookkeeping", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_BK = _load_bookkeeping()

# The release configuration: block_size 5, noise token 128799, taps 40/41/42.
BLOCK_SIZE = 5
NOISE_TOKEN_ID = 128799


def test_verify_len_is_anchor_plus_drafts():
    for proposal in range(0, BLOCK_SIZE + 1):
        assert _BK.dspark_verify_len(proposal) == 1 + proposal


def test_rollback_trim_is_rejected_tail():
    # full accept -> nothing to trim.
    assert _BK.dspark_rollback_trim(5, 5) == 0
    # one rejected -> trim one.
    assert _BK.dspark_rollback_trim(5, 4) == 1
    # everything rejected -> trim the whole drafted block.
    assert _BK.dspark_rollback_trim(5, 0) == 5
    # confidence-truncated proposal (<block) rejects its own tail correctly.
    assert _BK.dspark_rollback_trim(3, 1) == 2


def test_context_commit_is_anchor_plus_accepted():
    for accepted in range(0, BLOCK_SIZE + 1):
        assert _BK.dspark_context_commit_len(accepted) == accepted + 1


def test_next_position_advances_by_committed_context():
    assert _BK.dspark_next_position(100, 0) == 101  # only the anchor commits
    assert _BK.dspark_next_position(100, 3) == 104
    assert _BK.dspark_next_position(100, 5) == 106  # anchor + full block


def _simulate_round(cache_offset, position, proposal, accepted):
    """Mirror one round loop's cache/position/context bookkeeping.

    Returns the post-round ``(cache_offset, position, committed)`` the same way
    ``_dspark_rounds`` advances them: the verify writes ``1+proposal`` cells; on
    rejection the cache is trimmed by ``proposal-accepted``; the next anchor sits
    at ``position + accepted + 1``.
    """
    assert position == cache_offset  # anchor is NOT in the cache at round start
    cache_offset += _BK.dspark_verify_len(proposal)  # verify wrote anchor + drafts
    trim = _BK.dspark_rollback_trim(proposal, accepted)
    if trim > 0:
        cache_offset -= trim
    committed = _BK.dspark_context_commit_len(accepted)
    next_position = _BK.dspark_next_position(position, accepted)
    # The core invariant: committed context length == cache offset == next anchor
    # position (the anchor for the next round is not yet in the cache).
    assert cache_offset == next_position
    return cache_offset, next_position, committed


def test_multiround_cache_and_position_stay_aligned():
    # Start: prompt of length P processed, first bonus sampled (anchor) but NOT
    # yet written to the cache -> offset == position == P.
    P = 12
    cache_offset = position = P
    total_committed = 0
    # A mix of full-accept, partial-accept, and full-reject rounds.
    rounds = [
        (5, 5),  # all drafts accepted -> no rollback
        (5, 2),  # partial accept -> trim 3
        (5, 0),  # all rejected -> trim 5, only anchor commits
        (3, 1),  # confidence-truncated proposal, partial accept -> trim 2
        (5, 4),  # partial accept -> trim 1
    ]
    for proposal, accepted in rounds:
        cache_offset, position, committed = _simulate_round(
            cache_offset, position, proposal, accepted
        )
        total_committed += committed
    # Every committed token is exactly one advance of the absolute position.
    assert position == P + total_committed
    # 5 rounds committed anchor(=1) + accepted each: (5+1)+(2+1)+(0+1)+(1+1)+(4+1)
    assert total_committed == (5 + 1) + (2 + 1) + (0 + 1) + (1 + 1) + (4 + 1)


def test_full_accept_round_needs_no_rollback():
    # A round where every drafted token matches the target: trim is 0, so the
    # loop's ``if rollback_trim > 0`` guard skips ``rollback_speculative_cache``.
    proposal = accepted = BLOCK_SIZE
    assert _BK.dspark_rollback_trim(proposal, accepted) == 0
    off, pos, committed = _simulate_round(50, 50, proposal, accepted)
    assert off == pos == 50 + BLOCK_SIZE + 1
    assert committed == BLOCK_SIZE + 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"DSPARK_ROUND_LOOP_BOOKKEEPING_TEST ALL PASS ({len(fns)}/{len(fns)})")


if __name__ == "__main__":
    _run_all()
