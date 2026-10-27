#!/usr/bin/env python3
"""SwiReasoning MLX decode loop for the mlx-vlm generate path.

Implements the entropy-gated latent/explicit switch-thinking decode used by
``generate_step`` when a request passes ``swi_reasoning={...}``. The pure-python
state machine lives in ``swi_state.py``; this module holds only the MLX pieces:
building the token-embedding matrix E once, and the single-sequence decode
generator that yields ``(token_id, logprobs)`` for EXPLICIT steps only (latent
steps feed a soft embedding and emit nothing).

Faithful to sdc17/SwiReasoning (generation_utils.py); cross-checked against the
llama.cpp port Antonbe1b/swireasoning-llamacpp. See swi_state.py for the
no-emit-on-latent divergence note.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Generator, Optional, Tuple

import mlx.core as mx

from .swi_state import SwiConfig, SwiState, swi_alpha, swi_beta


def _resolve_embed_tokens(lm):
    for obj in (lm, getattr(lm, "model", None),
                getattr(getattr(lm, "model", None), "model", None)):
        if obj is None:
            continue
        emb = getattr(obj, "embed_tokens", None)
        if emb is not None:
            return emb
    return None


def _encode_marker(tokenizer, text: str):
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(x) for x in ids]
    except Exception:
        try:
            ids = tokenizer.encode(text)
            return [int(x) for x in ids]
        except Exception:
            return []


def build_swi_model_data(model, tokenizer) -> Optional[dict]:
    """Build + return the dequantized E matrix (bf16) + marker embeddings/ids.

    Returns None when E cannot be built (caller then decodes normally). A one-time
    ~1-2 GB bf16 dequant of the vocab x hidden embedding matrix; the paper's
    soft_emb = probs_original @ E needs the full-vocab matrix.
    """
    try:
        lm = model.language_model if hasattr(model, "language_model") else model
        emb_tok = _resolve_embed_tokens(lm)
        if emb_tok is None:
            raise RuntimeError("cannot resolve embed_tokens")
        w = getattr(emb_tok, "weight", None)
        if w is None:
            raise RuntimeError("embed_tokens has no .weight")
        scales = getattr(emb_tok, "scales", None)
        if scales is None:
            E = w.astype(mx.bfloat16)
        else:
            biases = getattr(emb_tok, "biases", None)
            group_size = int(getattr(emb_tok, "group_size", 64))
            bits = int(getattr(emb_tok, "bits", 4))
            mode = getattr(emb_tok, "mode", None)
            E = None
            for kwargs in (
                {"group_size": group_size, "bits": bits,
                 **({"mode": mode} if mode is not None else {})},
                {"group_size": group_size, "bits": bits},
            ):
                try:
                    E = mx.dequantize(w, scales, biases, **kwargs)
                    break
                except TypeError:
                    continue
            if E is None:
                raise RuntimeError("mx.dequantize signature mismatch")
            E = E.astype(mx.bfloat16)
        mx.eval(E)
        vocab, hidden = int(E.shape[0]), int(E.shape[1])

        def _row(text):
            ids = _encode_marker(tokenizer, text)
            if not ids:
                return None
            last = ids[-1]
            if last < 0 or last >= vocab:
                return None
            return E[last]

        endthink_ids = _encode_marker(tokenizer, "</think>")
        data = {
            "E": E,
            "vocab": vocab,
            "hidden": hidden,
            "think": _row("<think>"),
            "endthink": _row("</think>"),
            "nl": _row("\n"),
            "endthink_ids": endthink_ids,
            "convergence_ids": endthink_ids,
            "termination_ids": _encode_marker(tokenizer, "</think>\n\nThe final answer is"),
        }
        data["has_specials"] = data["endthink"] is not None and data["nl"] is not None
        sys.stderr.write(
            f"[mlx-vlm swi] built E [{vocab} x {hidden}] bf16 "
            f"(~{vocab * hidden * 2 / (1024 ** 3):.2f} GB); "
            f"has_specials={data['has_specials']} endthink_ids={endthink_ids}\n"
        )
        sys.stderr.flush()
        return data
    except Exception as e:
        sys.stderr.write(f"[mlx-vlm swi] E-matrix build FAILED ({e}); swi disabled\n")
        sys.stderr.flush()
        return None


def swi_decode_loop(
    model,
    prompt_cache,
    kwargs: dict,
    generation_stream,
    sampler,
    first_logprobs: mx.array,
    swi_config: SwiConfig,
    model_data: dict,
    max_tokens: int,
    stats_sink: Optional[dict] = None,
) -> Generator[Tuple[int, mx.array], None, None]:
    """Single-sequence SwiReasoning decode from a warm prompt cache.

    Args:
      first_logprobs: log-softmax of the prompt's final logits (raw, from prefill)
        — used for the step-0 distribution.  probs_original = exp(logprobs).
      sampler: Callable[[logprobs_2d], token_array] (the request's tempered sampler).
      model_data: output of build_swi_model_data().
    Yields (token_id, logprobs[vocab]) for EXPLICIT steps only.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    E = model_data["E"]
    think = model_data["think"]
    endthink = model_data["endthink"]
    nl = model_data["nl"]
    has_specials = model_data["has_specials"]

    swi = SwiState(
        cfg=swi_config,
        endthink_ids=list(model_data["endthink_ids"]),
        convergence_ids=list(model_data["convergence_ids"]),
        termination_ids=list(model_data["termination_ids"]),
    )

    n_predict = max(1, int(max_tokens))
    step_cap = int(os.environ.get(
        "MLX_VLM_SWI_MAX_STEPS", str(max(4 * n_predict, n_predict + 2048))
    ))

    def _forward_ids(tok_id: int) -> mx.array:
        with mx.stream(generation_stream):
            out = lm(mx.array([[int(tok_id)]]), inputs_embeds=None,
                     cache=prompt_cache, **kwargs)
            return out.logits[:, -1, :]

    def _forward_emb(emb_row: mx.array) -> mx.array:
        embd = emb_row.reshape(1, 1, -1).astype(E.dtype)
        with mx.stream(generation_stream):
            out = lm(None, inputs_embeds=embd, cache=prompt_cache, **kwargs)
            return out.logits[:, -1, :]

    def _logprobs_of(logits_row: mx.array) -> mx.array:
        return logits_row - mx.logsumexp(logits_row, axis=-1, keepdims=True)

    # step-0 distribution from the prefill's logprobs.
    lp2d = first_logprobs.reshape(1, -1)
    emitted = 0
    step = 0
    while emitted < max_tokens and step < step_cap:
        probs = mx.exp(lp2d)                                   # [1, vocab] = softmax(raw logits)
        _p = mx.clip(probs, 1e-12, None)
        entropy = float((-mx.sum(probs * mx.log(_p), axis=-1)).item())

        swi.update_mode(entropy)
        swi.inject_triggers(step)
        latent = swi.should_use_soft()
        swi.count_step(latent)

        if latent:
            soft = (probs.astype(E.dtype) @ E).reshape(-1)     # [hidden]
            if has_specials:
                if step == 0 and nl is not None:
                    soft = 0.9 * soft + 0.1 * nl.astype(soft.dtype)
                elif swi.last_to_soft and think is not None:
                    a = swi_alpha(swi_config.alpha_0, step, n_predict)
                    soft = a * soft + (1.0 - a) * think.astype(soft.dtype)
            lp2d = _logprobs_of(_forward_emb(soft))
            step += 1
            continue

        forced = swi.pop_forced()
        if forced is not None:
            tok = int(forced)
        else:
            tok = int(sampler(lp2d).reshape(-1)[-1].item())

        yield tok, lp2d.reshape(-1)
        emitted += 1
        swi.note_emitted(tok)

        if swi.budget_tick():
            break

        beta_blend = has_specials and swi.last_to_normal and step > 0 and endthink is not None
        if beta_blend:
            soft = (probs.astype(E.dtype) @ E).reshape(-1)
            b = swi_beta(swi_config.beta_0, step, n_predict)
            emb = b * soft + (1.0 - b) * endthink.astype(soft.dtype)
            lp2d = _logprobs_of(_forward_emb(emb))
        else:
            lp2d = _logprobs_of(_forward_ids(tok))
        step += 1

    if stats_sink is not None:
        stats_sink.update(swi.stats())
        stats_sink["steps"] = step
        stats_sink["emitted"] = emitted
    sys.stderr.write(
        f"[mlx-vlm swi] done: {swi.soft_steps} soft / {swi.normal_steps} normal steps, "
        f"switches={swi.switch_count}, final_mode={swi.stats()['final_mode']}, emitted={emitted}\n"
    )
    sys.stderr.flush()
