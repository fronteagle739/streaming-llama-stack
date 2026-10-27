"""Pure (mlx-free) round-loop bookkeeping for the DSpark speculative loop.

Factored out of ``dspark.py`` so the accept/commit/position/rollback invariants
are unit-testable on a box without Metal/MLX (the ``dspark.py`` round loop imports
mlx). The invariants, per round, for a block that DRAFTS ``proposal`` tokens of
which ``accepted`` are accepted by the target (``0 <= accepted <= proposal``):

* verify input = ``[anchor, *drafts]`` -> length ``1 + proposal``.
* committed context this round = anchor + accepted drafts = ``accepted + 1``
  positions (the bonus token is the NEXT anchor; its capture is produced by the
  next verify, so it is NOT committed to the context window this round).
* rejected tail = ``proposal - accepted`` tokens; when > 0 the target cache is
  rolled back (trimmed, or replayed to the committed prefix when a compressed-KV
  window would otherwise overwrite committed state).
* next anchor position = ``position + accepted + 1`` == the committed context
  length. ``position`` feeds only the RoPE offsets and we take it == the target
  cache offset at round start (anchor NOT yet in the cache); commit 5bd134b flags
  this off-by-one as the one runtime alignment a Mac needle run must confirm.
"""


def dspark_verify_len(proposal: int) -> int:
    """Length of the target verify input = anchor + the proposed drafts."""
    return 1 + int(proposal)


def dspark_rollback_trim(proposal: int, accepted: int) -> int:
    """Number of tail draft tokens rejected (trimmed/replayed off the cache)."""
    return int(proposal) - int(accepted)


def dspark_context_commit_len(accepted: int) -> int:
    """Verify positions committed to the sliding context this round: anchor plus
    every accepted draft."""
    return int(accepted) + 1


def dspark_next_position(position: int, accepted: int) -> int:
    """Anchor absolute index for the next round (== committed context length)."""
    return int(position) + int(accepted) + 1
