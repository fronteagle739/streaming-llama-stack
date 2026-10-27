"""DeepSeek-V4-Flash DSpark speculative-decoding **round loop**.

The DSpark drafter (``speculative/drafters/deepseek_v4_dspark``) is a *block*
draft head: given the block anchor (the target's just-sampled token) and a window
of the target's layer-40/41/42 hidden states (``main_hidden``), it produces a
whole block of ``block_size`` (=5) candidate tokens in ONE backbone pass, sharpens
them with a first-order Markov head, and truncates the block at the first
low-confidence position (its confidence head). This module drives that drafter
through the standard draft -> verify -> walk -> rollback speculative loop, in the
same shape as ``dflash.py``/``mtp.py`` but with the DSpark specifics:

  * Context is the target-layer hidden capture (``capture_layer_ids`` plumbing in
    ``models/deepseek_v4/language.py``) reduced to ``main_hidden`` (mean the
    hyper-connection lanes, concat the taps). Each round appends the freshly
    committed positions' captures and slides a ``sliding_window`` window — the
    faithful equivalent of the reference's persistent draft KV cache (see the
    DESIGN NOTE in ``deepseek_v4_dspark.py``).
  * The block itself carries the noise token (id 128799) internally: the drafter's
    ``_draft_input_ids`` builds ``[anchor, noise, ...]`` — the round loop never
    injects noise, it only supplies ``anchor``/``main_hidden``/``position`` and
    the block size.
  * ``position`` (the anchor's absolute index) feeds only the RoPE offsets. We use
    ``position == target cache.offset`` at the start of the round, i.e. the number
    of already-committed context tokens with the anchor NOT yet written to the
    cache (the verify pass writes it). Commit 5bd134b flags this exact off-by-one
    as the one runtime alignment a Mac needle/coherence run must confirm; shapes
    are unaffected either way.
  * Verify runs the target on ``[anchor, *draft_tokens]`` (``speculative_verify_captures``
    also snapshots the compressed-KV pooling caches so rejection can be rolled
    back correctly), accepts the longest prefix whose tokens match the target's
    own greedy/sampled tokens (``_speculative_walk``), takes the target's token at
    the first mismatch as the bonus, then trims/replays the target cache back to
    the committed length.

WEIGHT-FREE / MAC-GATED: the drafter modules + this loop's bookkeeping are
structurally tested weight-free (``tests/test_speculative.py`` DSpark round-loop
mocks + ``models/deepseek_v4/test_dspark_round_loop.py`` pure-python bookkeeping).
End-to-end numeric coherence is gated on MLX + the release weights on the Mac.
"""

import json
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models.cache import CacheList
from .common import (
    _SpeculativeSamplerRNG,
    _dflash_block_total,
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_uniform_acceptance,
    generation_stream,
)
from .dspark_bookkeeping import (
    dspark_context_commit_len as _dspark_context_commit_len,
)
from .dspark_bookkeeping import (
    dspark_next_position as _dspark_next_position,  # noqa: F401 (documents invariant)
)
from .dspark_bookkeeping import (
    dspark_rollback_trim as _dspark_rollback_trim,
)
from .dspark_bookkeeping import (
    dspark_verify_len as _dspark_verify_len,
)

# --------------------------------------------------------------------------- #
# mlx helpers                                                                  #
# --------------------------------------------------------------------------- #


def _dspark_draft_count(draft_model, draft_block_size: Optional[int]) -> int:
    """Number of block tokens to draft this round (== DSpark config block_size,
    5 for the release). The drafter itself may return fewer after its confidence
    truncation; this is the ceiling handed to ``draft_block``."""
    return _dflash_block_total(draft_model, draft_block_size)


def dspark_context_from_captures(captures: List[mx.array]) -> mx.array:
    """Reduce the raw target-layer captures to ``main_hidden``.

    Mirrors ``DeepseekV4DSparkDraftModel.context_from_captures``: mean each tap
    over the hyper-connection lanes ([B, L, hc, d] -> [B, L, d]) then concat the
    taps -> [B, L, d * n_taps]. Kept standalone so ``speculative_hidden_state``
    (which does not hold the drafter) can produce the same tensor the round loop
    feeds ``draft_block``.
    """
    lanes_meaned = []
    for h in captures:
        if h.ndim == 4:
            h = h.mean(axis=2)
        lanes_meaned.append(h)
    return mx.concatenate(lanes_meaned, axis=-1)


def _dspark_cache_offset(prompt_cache: List[Any]) -> int:
    """Absolute committed-token count from the target prompt cache (scalar).

    DeepSeek-V4 ratio-0 layers hold a bare ``RotatingKVCache`` (``.offset``);
    compressed layers hold a ``CacheList`` whose first child is the local
    ``RotatingKVCache``. Return the max offset as a plain int.
    """
    for entry in prompt_cache:
        off = getattr(entry, "offset", None)
        if off is None and isinstance(entry, CacheList):
            children = getattr(entry, "caches", None) or []
            if children:
                off = getattr(children[0], "offset", None)
        if off is not None:
            return int(off.max().item()) if isinstance(off, mx.array) else int(off)
    return 0


def _dspark_trim_window(context_hidden: mx.array, window: int) -> mx.array:
    if window and context_hidden.shape[1] > window:
        return context_hidden[:, -window:, :]
    return context_hidden


def _dspark_verify(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
    target_layer_ids: List[int],
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[List[mx.array], mx.array, Any]:
    """Run one target verify pass. Returns ``(captures, target_tokens, rollback_state)``.

    Prefers the model's ``speculative_verify_captures`` (which also snapshots the
    compressed-KV pooling caches for a correct rejection rollback); falls back to
    a plain capture forward for models/mocks that do not implement it.
    """
    verify_captures = getattr(lm, "speculative_verify_captures", None)
    if callable(verify_captures):
        captures, logits, rollback_state, target_tokens = verify_captures(
            verify_input, prompt_cache, target_layer_ids, sampler
        )
        if target_tokens is None:
            target_tokens = sampler(logits)
        return captures, target_tokens, rollback_state
    out = lm(verify_input, cache=prompt_cache, capture_layer_ids=target_layer_ids)
    return out.hidden_states, sampler(out.logits), None


# --------------------------------------------------------------------------- #
# Round loops                                                                  #
# --------------------------------------------------------------------------- #


# Every NUMERIC candidate for the DSpark acceptance gap is closed (bf16, sampling policy,
# context tap, block size, quantisation asymmetry, Markov bias, noise ids, final tap), and
# acceptance is still 0.41 GREEDY -- the drafter disagrees with the target on 59% of
# DETERMINISTIC tokens. Another numeric check will not say why. This records WHAT it proposed
# at each rejected position, so the question becomes "which tokens does it get wrong, and do
# they rhyme?" -- which no dtype or config check can answer.
#
# Off unless DSV41_DSPARK_REJECT_LOG names a file. One short JSON line per round.
_DSPARK_REJECT_LOG = os.environ.get("DSV41_DSPARK_REJECT_LOG", "")


def _dspark_record_round(draft_tokens, target_tokens, accepted):
    """Append one round: what was drafted, what the target wanted, how many survived."""
    try:
        d = [int(t) for t in draft_tokens.reshape(-1).tolist()]
        t = [int(x) for x in target_tokens.reshape(-1).tolist()]
        with open(_DSPARK_REJECT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"draft": d, "target": t,
                                 "accepted": int(accepted)}) + chr(10))
    except Exception:
        # Diagnostics must never take generation down with them.
        pass


def _dspark_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    greedy_sampling: bool = False,
) -> Generator[Tuple[int, None], None, None]:
    """DSpark speculative round loop (batch size 1).

    ``hidden`` is the reduced ``main_hidden`` [1, L, d*taps] from prefill (see
    ``dspark_context_from_captures``). The caller has already yielded the first
    bonus token.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "Speculative decoding with a DSpark drafter currently only supports "
            "mlx_vlm.models.deepseek_v4."
        )

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dspark_draft_count(draft_model, draft_block_size)
    window = int(getattr(draft_model, "window_size", 0) or 0)
    draft_model.reset(model)

    preserve_seed = bool(getattr(draft_model, "preserves_target_seed", False))
    if preserve_seed and any(callable(getattr(c, "is_trimmable", None))
                             and not c.is_trimmable() for c in prompt_cache):
        raise ValueError("V4.1 DSpark requires rollback-capable caches (DSV41_BOUNDED_CACHE=0)")
    sampler_rng = _SpeculativeSamplerRNG(draft_model, enabled=False)
    verify_sampler = sampler
    if preserve_seed and not greedy_sampling:
        from ..models.deepseek_v41.dspark_sampling import DSparkSamplerRNG, DSparkTargetSampler

        sampler_rng = DSparkSamplerRNG()
        verify_sampler = DSparkTargetSampler(sampler)

    context_hidden = _dspark_trim_window(hidden, window)
    b = int(first_bonus)
    emitted = 1  # first bonus already yielded by the caller
    timing = getattr(draft_model, "phase_timing", None)
    # v41 port lever 3 (2026-09-15): cost-feedback admission control. A serial step is a zero-draft
    # round through the same verify/commit/rollback code below; the controller sees wall-clock ms
    # per round (measured at the walk, which forces the target tokens) and committed counts.
    controller = None
    if os.environ.get("DSV41_DSPARK_ADAPTIVE", "0") == "1":
        from .dspark_controller import DSparkCostController
        controller = DSparkCostController.from_env()
        draft_model.cost_controller = controller
    from time import perf_counter as _clock

    while emitted < max_tokens:
        budget = max_tokens - emitted
        n_draft = min(block_total, budget)
        if n_draft < 1:
            break

        position = _dspark_cache_offset(prompt_cache)
        if timing is not None:
            mx.eval(context_hidden)
            mx.synchronize()
            mx.synchronize(generation_stream)
            timing.start()
        t_round = _clock()
        draft_ms = 0.0
        declined = False
        drafting = controller is None or controller.should_draft()
        if drafting:
            result = sampler_rng.draft_tokens(
                draft_model.draft_block,
                mx.array([b], dtype=token_dtype),
                context_hidden,
                position,
                sampler=sampler,
                greedy=greedy_sampling,
                block_size=n_draft,
                **({"confidence_threshold": controller.p_min} if controller is not None else {}),
            )
            draft_tokens = result["draft_tokens"]
            if controller is not None and not controller.admits(int(draft_tokens.shape[1])):
                # M21/M22: the confident prefix is too short -- decline before the verify. The
                # drafter's cost is paid (its confident-prefix length is already on the host).
                declined = True
                draft_ms = (_clock() - t_round) * 1000.0
                draft_tokens = mx.zeros((1, 0), dtype=token_dtype)
        else:
            draft_tokens = mx.zeros((1, 0), dtype=token_dtype)
        proposal = int(draft_tokens.shape[1])
        mx.async_eval(draft_tokens)

        if timing is not None:
            mx.eval(draft_tokens)
            mx.synchronize()
            timing.phase("draft")

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
            )
            captures, target_tokens, rollback_state = _dspark_verify(
                lm, verify_input, prompt_cache, target_layer_ids, verify_sampler
            )
        mx.async_eval(target_tokens)

        if timing is not None:
            mx.eval(target_tokens, captures)
            mx.synchronize(generation_stream)
            mx.synchronize()
            timing.phase("verify")

        accepted, new_tokens = _speculative_walk(draft_tokens, target_tokens, budget)
        if controller is not None:
            round_ms = (_clock() - t_round) * 1000.0
            if proposal > 0:
                controller.record_attempt("verify", len(new_tokens), round_ms)
            else:
                if declined:
                    controller.record_attempt("decline", 0, draft_ms)
                controller.record_serial(max(0.0, round_ms - draft_ms), len(new_tokens))
        if _DSPARK_REJECT_LOG and proposal > 0:
            _dspark_record_round(draft_tokens, target_tokens, accepted)
        # The last emitted token is the next uncached anchor, including when the
        # token budget cuts off a fully accepted block before its bonus token.
        if preserve_seed:
            accepted = min(accepted, len(new_tokens) - 1)
        if preserve_seed and not greedy_sampling:
            verify_sampler.commit(len(new_tokens))
            sampler_rng.target_sampled()
        draft_model.accept_lens.append(accepted)

        # Roll back before yielding: a consumer may stop at the token budget,
        # EOS, or cancellation and then reuse this conversation cache.
        b = new_tokens[-1] if new_tokens else b

        # Commit the anchor + accepted drafts into the sliding context window.
        commit = _dspark_context_commit_len(accepted)
        ctx_reduced = dspark_context_from_captures(captures)[:, :commit, :]
        context_hidden = _dspark_trim_window(
            mx.concatenate([context_hidden, ctx_reduced], axis=1), window
        )

        # Roll the target cache back to the committed length on rejection.
        if _dspark_rollback_trim(proposal, accepted) > 0:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, rollback_state, accepted, _dspark_verify_len(proposal)
                )

        if timing is not None:
            mx.eval(context_hidden)
            mx.synchronize(generation_stream)
            mx.synchronize()
            timing.finish(accepted, len(new_tokens))

        sent = 0
        try:
            for tok in new_tokens:
                sent += 1
                emitted += 1
                yield tok, None
        finally:
            # A streaming consumer can close on EOS/cancellation mid-block.
            # Keep only rows that predict tokens actually handed to it.
            if preserve_seed and sent < commit:
                draft_model.accept_lens[-1] = max(0, sent - 1)
                for entry in prompt_cache:
                    if entry is not None:
                        entry.trim(commit - sent)
                if not greedy_sampling and sent:
                    verify_sampler.commit_prefix(sent, len(new_tokens))

        if emitted >= max_tokens:
            return

        if emitted % 256 == 0:
            mx.clear_cache()


def _dspark_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    eos_token_ids: Optional[set] = None,
    greedy_sampling: bool = False,
    row_ids: Optional[List[int]] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batched DSpark round loop (B >= 1).

    DSpark requires uniform batch acceptance (``requires_uniform_batch_acceptance``):
    every row commits the same number of tokens per round, so all rows keep a
    shared block ``position`` and a shared sliding context window, and the
    confidence truncation is disabled for the batch (a uniform ``block_size``
    block is drafted; the walk + uniform clamp decide acceptance). Continuous
    batching filters finished rows when the target caches support ``.filter()``.
    """
    if getattr(draft_model, "preserves_target_seed", False):
        if first_bonus.shape[0] != 1 or row_ids not in (None, [0]):
            raise NotImplementedError("V4.1 DSpark supports one conversation with row id zero")
        rounds = _dspark_rounds(model, draft_model, prompt_cache, hidden,
            first_bonus=int(first_bonus.item()), max_tokens=max_tokens, sampler=sampler,
            draft_block_size=draft_block_size, token_dtype=token_dtype,
            greedy_sampling=greedy_sampling)
        try:
            for token, logprobs in rounds:
                yield [token], logprobs
                if ((eos_token_ids is not None and token in eos_token_ids)
                        or (stop_check is not None and stop_check(0, token))):
                    return
        finally:
            rounds.close()
        return

    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache."
        )

    B = int(first_bonus.shape[0])
    row_ids = list(range(B)) if row_ids is None else list(row_ids)
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dspark_draft_count(draft_model, draft_block_size)
    window = int(getattr(draft_model, "window_size", 0) or 0)
    draft_model.reset(model)

    context_hidden = _dspark_trim_window(hidden, window)  # [B, L, d*taps]
    b = first_bonus.tolist()
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))

    while len(active_idx) > 0:
        n_active = len(active_idx)
        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        n_draft = min(block_total, max(0, min(budgets)))
        if n_draft < 1:
            break

        position = _dspark_cache_offset(prompt_cache)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        result = draft_model.draft_block(
            b_arr,
            context_hidden,
            position,
            sampler=sampler,
            greedy=greedy_sampling,
            block_size=n_draft,
            confidence_threshold=0.0,  # uniform block for batched acceptance
        )
        draft_tokens = result["draft_tokens"]
        proposal = int(draft_tokens.shape[1])
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            captures, target_tokens, rollback_state = _dspark_verify(
                lm, verify_input, prompt_cache, target_layer_ids, sampler
            )
        mx.async_eval(target_tokens)

        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, target_tokens, budgets
        )
        if n_active > 1 and len(set(accepted_list)) > 1:
            accepted_list, new_tokens_list = _speculative_walk_batch_uniform_acceptance(
                draft_tokens, target_tokens, accepted_list, budgets
            )
        accepted = accepted_list[0]
        for a in accepted_list:
            draft_model.accept_lens.append(a)

        # Commit anchor + accepted drafts into the shared sliding window.
        commit = _dspark_context_commit_len(accepted)
        ctx_reduced = dspark_context_from_captures(captures)[:, :commit, :]
        context_hidden = _dspark_trim_window(
            mx.concatenate([context_hidden, ctx_reduced], axis=1), window
        )

        # Emit (map active slots back to original indices).
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if eos_token_ids is not None and tok in eos_token_ids:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if _dspark_rollback_trim(proposal, accepted) > 0:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache,
                    rollback_state,
                    mx.array(accepted_list),
                    _dspark_verify_len(proposal),
                )

        if all(finished[active_idx[j]] for j in range(n_active)):
            break
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active and all(
            hasattr(c, "filter") for c in prompt_cache
        ):
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                c.filter(keep_mx)
            context_hidden = context_hidden[keep_mx]
            active_idx = [active_idx[j] for j in keep_slots]

        if sum(emitted) % 256 == 0:
            mx.clear_cache()
