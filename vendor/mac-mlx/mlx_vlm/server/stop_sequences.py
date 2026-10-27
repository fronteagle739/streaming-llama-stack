"""Incremental stop-sequence matching for the OpenAI-compatible chat server.

This module is deliberately **pure Python** — it imports nothing from ``mlx`` at
module level — so it can be unit-tested on any platform (e.g. a Windows dev box
with no Metal / MLX). Keep it that way.

Background / why this exists
----------------------------
The plain ``mlx_vlm.server`` (``/v1/chat/completions``) previously had no OpenAI
``stop`` support: its request schema silently dropped the field, so a client that
sent ``stop`` still decoded all the way to ``max_tokens``. Streaming Studio's
agent loop relies on ``stop`` to end a GLM-5.2 decode at a tool-call block
(``</invoke>``, ``</function_calls>``, ``</tool_call>``, ``<|tool_call_end|>``);
without it the backend kept generating to ``max_tokens=65536`` after the client
had already disconnected — a *zombie decode* that blocks the next agent round
(streaming-studio commit 50570bb). See :class:`StopSequenceMatcher`.

Semantics: INCLUDE (deviation from OpenAI)
------------------------------------------
OpenAI trims the matched stop sequence from the returned text. We instead use
**include** semantics: the text up to *and including* the matched stop sequence
is emitted, then generation halts. This is intentional:

* Streaming needs no hold-back buffering — every token can be emitted as soon as
  it is produced, and we only need to look at the accumulated tail to decide
  when to stop.
* The Studio daemon's tool-call parsers want the *complete* block (a trailing
  ``</invoke>`` / ``</function_calls>`` etc. must be present), so trimming the
  terminator would leave the block malformed for those parsers.

The daemon already tolerates both include and exclude backends, so include is
safe. The distinct ``_apply_stop_sequences`` helper in ``anthropic.py`` keeps
Anthropic's spec-mandated *exclude/trim* semantics; this module is the OpenAI
chat path's include-semantics counterpart.
"""

from typing import List, Optional, Tuple, Union

# Defensive cap on the number of stop sequences we honor for a single request.
# OpenAI's own API caps ``stop`` at 4 entries; we allow a little more headroom
# but bound it so a pathological request can't force us to scan an unbounded
# list on every generated token.
MAX_STOP_SEQUENCES = 8


def normalize_stop_sequences(
    stop: Optional[Union[str, List[str]]],
) -> List[str]:
    """Normalize an OpenAI ``stop`` field into a clean list of strings.

    Accepts either a single string or a list of strings (the two shapes the
    OpenAI API allows). Returns a list with:

    * ``None`` -> ``[]``
    * a bare string -> ``[string]`` (if non-empty)
    * empty / non-string entries dropped
    * duplicates removed (preserving first-seen order)
    * capped at :data:`MAX_STOP_SEQUENCES` entries

    Never raises on odd input — anything unusable is simply filtered out — so it
    is safe to call on request data that has already passed pydantic validation.
    """
    if stop is None:
        return []
    if isinstance(stop, str):
        candidates: List[str] = [stop]
    else:
        try:
            candidates = list(stop)
        except TypeError:
            return []

    seen = set()
    result: List[str] = []
    for item in candidates:
        if not isinstance(item, str) or item == "":
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= MAX_STOP_SEQUENCES:
            break
    return result


def find_stop(
    text: str, stop_sequences: List[str]
) -> Tuple[Optional[int], Optional[str]]:
    """Find the earliest completed stop sequence in ``text`` (include semantics).

    Returns ``(end_index, sequence)`` where ``end_index`` is the offset *just
    past* the matched sequence (so ``text[:end_index]`` includes the terminator),
    and ``sequence`` is the matched stop string. Returns ``(None, None)`` if no
    stop sequence is present.

    "Earliest" is decided by the start position of the match; ties (two
    sequences starting at the same offset) are broken by the longer match, so
    the fuller terminator wins.
    """
    if not text or not stop_sequences:
        return None, None
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_seq: Optional[str] = None
    for seq in stop_sequences:
        if not seq:
            continue
        idx = text.find(seq)
        if idx < 0:
            continue
        end = idx + len(seq)
        if (
            best_start is None
            or idx < best_start
            or (idx == best_start and end > (best_end or -1))
        ):
            best_start = idx
            best_end = end
            best_seq = seq
    if best_start is None:
        return None, None
    return best_end, best_seq


class StopSequenceMatcher:
    """Incremental, token-boundary-aware stop-sequence matcher.

    Feed generated text chunks in via :meth:`feed`. The matcher accumulates the
    full text so a stop sequence that straddles a token boundary (e.g. one token
    ends ``"...</inv"`` and the next begins ``"oke>"``) is still detected — the
    match is evaluated against the accumulated tail, not the per-chunk delta.

    Usage (streaming, include semantics)::

        matcher = StopSequenceMatcher(["</invoke>", "</function_calls>"])
        for token in decode():
            visible, matched = matcher.feed(token.text)
            emit(visible)              # text up to & incl. terminator on the
                                       # matching chunk, else the whole chunk
            if matched is not None:
                finish_reason = "stop"
                break                  # halt decode promptly

    Because we use include semantics, ``feed`` never has to hold anything back:
    on a non-matching chunk it returns the chunk unchanged; on the matching
    chunk it returns exactly the portion of that chunk up to and including the
    terminator, and ``matched()`` becomes true. Any text after the terminator
    within the same chunk is discarded (generation stops there).
    """

    def __init__(self, stop_sequences: Union[List[str], None, str] = None):
        if isinstance(stop_sequences, str) or stop_sequences is None:
            self.stop_sequences = normalize_stop_sequences(stop_sequences)
        else:
            # Already a list — still normalize (dedupe / drop empties / cap).
            self.stop_sequences = normalize_stop_sequences(list(stop_sequences))
        # Longest stop sequence, used to bound how far back a straddling match
        # can begin. We only need to re-scan the tail of the accumulated text.
        self._max_len = max((len(s) for s in self.stop_sequences), default=0)
        self._accumulated = ""
        self._emitted_len = 0
        self._matched_sequence: Optional[str] = None

    @property
    def active(self) -> bool:
        """True when at least one usable stop sequence is configured."""
        return bool(self.stop_sequences)

    @property
    def matched_sequence(self) -> Optional[str]:
        """The stop sequence that halted generation, or ``None``."""
        return self._matched_sequence

    def matched(self) -> bool:
        return self._matched_sequence is not None

    def feed(self, chunk: str) -> Tuple[str, Optional[str]]:
        """Consume a text chunk; return ``(visible_text, matched_sequence)``.

        ``visible_text`` is the portion of this chunk that should be emitted
        (the whole chunk when no stop is hit; the slice up to and including the
        terminator on the chunk that completes a stop sequence). Once a stop has
        matched, subsequent ``feed`` calls return ``("", <sequence>)`` — callers
        should have stopped decoding, but this keeps the matcher idempotent.

        A straddling match is handled by re-scanning a bounded tail of the
        accumulated text: everything not yet emitted, plus up to ``max_len - 1``
        bytes before it (so a sequence whose start landed in a previous chunk is
        still found).
        """
        if self._matched_sequence is not None:
            return "", self._matched_sequence
        if not chunk:
            return "", None

        prev_len = len(self._accumulated)
        self._accumulated += chunk

        if not self.stop_sequences:
            # No stop configured: emit everything, never match.
            self._emitted_len = len(self._accumulated)
            return chunk, None

        # Re-scan from a point far enough back that a stop sequence which began
        # in an earlier chunk (straddling the boundary) is still visible. We
        # never need to look further back than (max_len - 1) before the start of
        # the un-emitted region.
        scan_start = max(0, prev_len - (self._max_len - 1))
        end_index, seq = find_stop(self._accumulated[scan_start:], self.stop_sequences)

        if end_index is None:
            # No stop yet — emit the whole new chunk.
            self._emitted_len = len(self._accumulated)
            return chunk, None

        abs_end = scan_start + end_index
        # Emit only the not-yet-emitted text up to & including the terminator.
        visible_start = self._emitted_len
        visible = self._accumulated[visible_start:abs_end]
        self._emitted_len = abs_end
        self._matched_sequence = seq
        return visible, seq

    @property
    def text(self) -> str:
        """The full accumulated text fed so far (including any post-terminator
        bytes from the matching chunk, which were *not* emitted)."""
        return self._accumulated

    @property
    def emitted_text(self) -> str:
        """The accumulated text up to (and including) the stop terminator — i.e.
        everything :meth:`feed` has returned as visible so far."""
        return self._accumulated[: self._emitted_len]
