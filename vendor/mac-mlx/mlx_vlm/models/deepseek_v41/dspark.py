"""DeepSeek-V4.1 Flash DSpark draft stages, in MLX.

Port of the release's ``mtp.*`` namespace from DeepSeek's reference
``inference/model.py``: ``get_dspark_topk_idxs`` / ``DSparkAttention`` /
``DSparkMarkovHead`` / ``DSparkConfidenceHead`` / ``DSparkBlock``
(``forward_embed`` + ``forward_head``) at lines 1021-1158, the three stages
appended after the backbone sharing the target's ``embed`` and ``head``
(1207-1213), the target-layer tap (1264-1266) and ``Transformer.forward_spec``
(1274-1282).

Three stages, each a full V4.1 block: ratio-0 (sliding-window only) MLA
attention, a 128-expert / top-3 MoE, two RMSNorms and the two staggered
hyper-connection mixers. Stage 0 additionally owns ``main_proj``/``main_norm``
(the target-hidden mixer); the last stage owns the final ``norm``, the
``markov_head`` and the ``confidence_head``. Everything else is imported from
``language.py`` so the DSpark stages and the backbone share one implementation
of attention, MoE, hyper-connections, RoPE, sparse attention and QAT.

RESOLVED (PR #2214's open question, COMPARISON.md A.13): the checkpoint ships no
``mtp.*.hc_head*`` tensors because the reference's ``forward_head`` collapses the
hyper-connection lanes with ``Block.hc_pre`` -- the PARAMETER-FREE weighted sum
over the previous sub-block's ``pre_mix`` ("Collapse the hc copies into one,
weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]") -- not with a learned
head. V4.1 has no ``HyperHead`` anywhere, backbone or MTP. So ``forward_head``
here calls ``language.hc_pre`` and needs no weights.

No decode state lives on these modules: the draft window KV is recomputed from
the supplied ``main_x`` window every round (see ``DSparkAttention.draft_forward``).

Verified on this box by ``python -m py_compile`` and by the structural tests in
``test_dspark_dsv41.py``; the MLX numerics are Mac-gated.
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .checkpoint_layout import DSPARK_PREFIX as DSPARK_TREE_PREFIX
from .checkpoint_layout import dspark_stage_count
from .config import TextConfig
from .language import (
    Attention,
    HyperConnection,
    MoE,
    RMSNorm,
    _rope,
    fake_quant_fp8_ue8m0,
    hc_post,
    hc_pre,
    sparse_attn,
)

__all__ = [
    "DSPARK_TREE_PREFIX",
    "DSparkAttention",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DSparkStack",
    "DSparkStage",
    "dspark_forward_embed",
    "dspark_forward_head",
    "dspark_noise_draft_ids",
    "dspark_run_stages",
    "dspark_stage_config",
    "get_dspark_topk_idxs",
    "make_identity_pre_mix",
    "n_dspark_stages",
]

# ``DSPARK_TREE_PREFIX`` ("dspark.stages.") is where the release's ``mtp.*``
# tensors land in the LanguageModel namespace; ``DSparkStack`` reproduces that
# tree, so ``LanguageModel.dspark = DSparkStack(config)`` is all the wiring a
# single-checkpoint load needs.


def n_dspark_stages(config: TextConfig) -> int:
    """Number of DSpark stages the config describes (0 when it describes none).

    One implementation, in ``checkpoint_layout`` -- which stays MLX-free so the
    CPU-only layout tests can import it by file -- so the runtime tree and the
    planned checkpoint tree can never disagree about how many stages exist.
    """
    return dspark_stage_count(config)


def dspark_stage_config(config: TextConfig) -> TextConfig:
    """The stage-local view of the text config.

    The MTP MoE is narrower than the backbone's: ``dspark_n_routed_experts`` (128)
    experts with ``dspark_num_experts_per_tok`` (3) active, against the backbone's
    384/6. Everything else -- hidden size, head geometry, rope, hc_mult, the
    swiglu limit, the gate scoring -- is shared, so the stage builds
    ``language.MoE`` / ``language.Attention`` unchanged against this view.
    """
    return replace(
        config,
        n_routed_experts=config.dspark_n_routed_experts,
        num_experts_per_tok=config.dspark_num_experts_per_tok,
    )


def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int,
                         start_pos: int) -> mx.array:
    """Reference ``get_dspark_topk_idxs`` (model.py 1020-1029), in MLX.

    Every block query sees the same key set: the ``min(window_size, start_pos+1)``
    valid window slots followed by all ``block_size`` draft keys (which live at
    offsets ``window_size ...`` in the concatenated KV). The reference's
    ``sparse_attn`` applies no mask of its own, so the draft block is
    BIDIRECTIONAL over itself -- the whole block is produced in one pass from the
    anchor plus noise, not autoregressively.

    The reference asserts ``start_pos > 0`` because its ``DSparkBlock.forward``
    returns before the block attention while ``start_pos == 0`` (prefill only
    seeds the ring). Our loop never drafts during prefill, and it counts
    ``start_pos`` as the last COMMITTED index, so a one-token prompt legitimately
    reaches ``start_pos == 0`` with one context row; the bound is ``>= 0`` here.
    """
    if start_pos < 0:
        raise ValueError("DSpark drafting starts after at least one committed token")
    matrix = mx.concatenate([
        mx.arange(min(window_size, start_pos + 1), dtype=mx.int32),
        window_size + mx.arange(block_size, dtype=mx.int32),
    ])
    return mx.broadcast_to(matrix.reshape(1, 1, -1),
                           (bsz, block_size, matrix.shape[0]))


def make_identity_pre_mix(b: int, n: int, hc_mult: int) -> mx.array:
    """Reference ``make_identity_pre_mix``: "initial one-hot mix"."""
    return mx.broadcast_to((mx.arange(hc_mult) == 0).astype(mx.float32), (b, n, hc_mult))


def dspark_noise_draft_ids(anchor: mx.array, block_size: int,
                           noise_token_id: int) -> mx.array:
    """Reference ``forward_embed``: ``draft_input_ids`` is a full block of the
    noise token with the anchor written into position zero
    (``input_ids.new_full([B, block_size], noise_token_id)``;
    ``draft_input_ids[:, 0] = input_ids``)."""
    anchor = anchor.reshape(-1, 1).astype(mx.int32)
    if block_size == 1:
        return anchor
    noise = mx.full((anchor.shape[0], block_size - 1), noise_token_id, dtype=mx.int32)
    return mx.concatenate([anchor, noise], axis=1)


# Run the DSpark draft block in fp32 instead of the ambient bf16. A MEASUREMENT ARM: it exists to
# test whether the drafter is being quietly degraded by bf16 on Metal, which would surface only
# as acceptance rate. Default off so the production path is unchanged.
_DSPARK_BLOCK_FP32 = os.environ.get("MLX_DSPARK_BLOCK_FP32", "") not in ("", "0")
# COUNT INVOCATIONS, do not announce installations. An env var that reaches the process proves
# nothing about whether the branch ran -- and the first fp32 arm came out FASTER than bf16,
# which is the wrong direction for real fp32 matmuls and is exactly what a silently-inert
# flag looks like. This says, once, that the cast actually executed.
_DSPARK_FP32_CALLS = 0
if _DSPARK_BLOCK_FP32:
    print("[dspark] MLX_DSPARK_BLOCK_FP32=1: draft block will run fp32", file=sys.stderr, flush=True)


class DSparkMarkovHead(nn.Module):
    """Reference ``DSparkMarkovHead``: a rank-``dspark_markov_rank`` embedding of
    the previously chosen token and a full-vocab projection back out. Returns
    ``(logit_bias, embed)``; the bias sharpens the parallel block into a
    first-order-Markov-consistent sequence, the embed feeds the confidence head.

    Checkpoint: ``mtp.2.markov_head.embed.weight`` and
    ``mtp.2.markov_head.head.weight``, both bf16 [vocab, rank]. The reference's
    ``ParallelHead`` is "bf16 in the checkpoint, kept as fp32 here so the logits
    come out in fp32 directly" -- we keep the stored bf16 and cast the activation
    instead, which gives the same fp32 logits without a second copy of a 66 MB
    table.
    """

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, markov_rank)
        self.head = nn.Linear(markov_rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> Tuple[mx.array, mx.array]:
        embed = self.embed(token_ids)
        return self.head(embed.astype(mx.float32)), embed


class DSparkConfidenceHead(nn.Module):
    """Reference ``DSparkConfidenceHead``: one bias-free linear over
    ``[hidden ; markov_embed]``.

    Reference comment, quoted because the dtype is load-bearing: "proj in the
    checkpoint is stored in bf16, while the parameter here is stored in fp32 for
    fp32 confidence score." We keep the checkpoint's bf16 parameter and run the
    contraction in fp32 (``hidden.float()``), which is the same arithmetic.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        feats = mx.concatenate(
            [hidden.astype(mx.float32), markov_embed.astype(mx.float32)], axis=-1)
        return self.proj(feats).squeeze(-1)


class DSparkAttention(Attention):
    """Reference ``DSparkAttention``: ratio-0 (sliding-window) MLA whose CONTEXT
    keys/values are projected from the target's hidden ``main_x`` rather than from
    re-embedded context tokens, concatenated with the draft block's own KV.

    Inherits the whole V4.1 parameter surface from ``language.Attention``
    (``wq_a``/``q_norm``/``wq_b``/``wkv``/``kv_norm``/``wo_a``/``wo_b``/
    ``attn_sink``/``rope``) -- the MTP layer ids sit past the backbone, so
    ``compress_ratios[layer_id] == 0`` gives no compressor, no indexer, and the
    plain ``rope_theta`` with YaRN off, exactly as the reference's
    ``if self.compress_ratio: ... else: original_seq_len, rope_theta = 0,
    args.rope_theta``.

    DESIGN NOTE (documented simplification, correctness-preserving; the same one
    our V4 drafter carries). The reference keeps a persistent ``window_kv_cache``
    ring, seeded from ``main_x`` during target prefill and written one column per
    decode step at slot ``start_pos % win``. Here each round recomputes the
    context KV from the supplied ``main_x`` window. The ring's content is a
    deterministic function of ``main_x`` and the absolute position
    (``kv_norm(wkv(main_x))`` + rope at that position + the fp8 round-trip, all
    per-position), so this is numerically identical; it trades a small recompute
    for not maintaining draft-side cache bookkeeping. Attention is a sum over
    keys, so the ring's rotated order versus our chronological order cannot
    matter either.
    """

    def draft_forward(self, x: mx.array, main_x: mx.array, ctx_offset: int,
                      block_offset: int) -> mx.array:
        if self.ratio != 0:
            raise ValueError("DSpark attention requires compress ratio 0")
        b, block, _ = x.shape
        ctx_len = main_x.shape[1]

        # Context KV from the projected target hidden, roped at its absolute
        # positions and fp8-fake-quantized exactly like the backbone window KV
        # (reference: kv_norm(wkv(main_x)); apply_rotary_emb; act_quant).
        ckv = fake_quant_fp8_ue8m0(
            _rope(self.rope, self.kv_norm(self.wkv(main_x)), ctx_offset), 32)

        # Block query / KV from the draft-token stream at the block's positions
        # (reference: freqs_cis[start_pos + seqlen : start_pos + seqlen + block]).
        q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(b, block, self.n_heads, self.head_dim)
        q = _rope(self.rope, q, block_offset)
        bkv = fake_quant_fp8_ue8m0(
            _rope(self.rope, self.kv_norm(self.wkv(x)), block_offset), 32)

        kv = mx.concatenate([ckv, bkv], axis=1)
        idxs = get_dspark_topk_idxs(ctx_len, b, block, block_offset - 1)
        out = sparse_attn(q, kv, self.attn_sink, idxs, self.head_dim ** -0.5)
        out = _rope(self.rope, out, block_offset, inverse=True)

        # Same grouped output projection as language.Attention.__call__.
        out = out.reshape(b, block, self.n_groups, -1)
        if self.wo_a.weight.shape[-1] != out.shape[-1] or self.wo_a.weight.dtype == mx.uint32:
            raise ValueError("wo_a must be dequantized to floating [o_groups, o_lora_rank, input_dim]")
        out = mx.einsum("bsgd,grd->bsgr", out.astype(mx.float32),
                        self.wo_a.weight.astype(mx.float32))
        return self.wo_b(out.reshape(b, block, -1).astype(x.dtype))


class DSparkStage(nn.Module):
    """Reference ``DSparkBlock``: "DSpark stage stored under the mtp.*
    checkpoint namespace."

    Structurally ``language.DeepseekV41Block`` with ``DSparkAttention`` and the
    MTP MoE, no engram, plus the stage-specific heads. The staggered
    hyper-connection wiring is the backbone's verbatim: attention consumes the
    previous sub-block's ``pre_mix`` and the FFN consumes the one this
    attention produced.
    """

    def __init__(self, config: TextConfig, stage_idx: int, n_stages: Optional[int] = None):
        super().__init__()
        stage_config = dspark_stage_config(config)
        n_stages = int(n_stages or config.num_nextn_predict_layers)
        self.stage_idx = stage_idx
        self.attn = DSparkAttention(config.num_hidden_layers + stage_idx, stage_config)
        self.ffn = MoE(stage_config)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        if stage_idx == 0:
            taps = len(config.dspark_target_layer_ids)
            if taps == 0:
                raise ValueError("DSpark needs target layers")
            self.main_proj = nn.Linear(config.hidden_size * taps, config.hidden_size, bias=False)
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if stage_idx == n_stages - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(config.vocab_size, config.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank)

    def __call__(self, h: mx.array, pre_mix: mx.array, main_x: mx.array,
                 ctx_offset: int, block_offset: int) -> Tuple[mx.array, mx.array]:
        # MEASUREMENT ARM, default OFF. The draft block runs in the model ambient bf16, same as
        # the target. But the target is checked by OUTPUT QUALITY and the drafter has exactly one
        # observable -- acceptance -- so a bf16 weakness that is invisible in the target shows up
        # ONLY as depressed acceptance. That is the shape of the standing ~2x, and it is the last
        # candidate the structural audit did not close.
        #
        # fp32 and NOT fp16: these weights are bf16 ON DISK, so bf16->fp32 is lossless while
        # bf16->fp16 narrows the exponent from 8 bits to 5 (measured: 0.34 maxabs drift on the
        # confidence head, against 0.0 for fp32). The Mac-is-bad-at-bf16 rule is about what we
        # COMPUTE there, not about re-casting a stored bf16 weight downward.
        #
        # Activations only. MLX promotes a bf16 weight against an fp32 activation and returns
        # fp32 -- measured bit-identical to casting the weight, so no weight copy is needed.
        if _DSPARK_BLOCK_FP32:
            global _DSPARK_FP32_CALLS
            _DSPARK_FP32_CALLS += 1
            if _DSPARK_FP32_CALLS == 1:
                print("[dspark] fp32 draft block FIRED (h dtype was %s)" % h.dtype,
                      file=sys.stderr, flush=True)
            h = h.astype(mx.float32)
            pre_mix = pre_mix.astype(mx.float32)
            main_x = main_x.astype(mx.float32)
        attn_pre, post, comb = self.attn_hc(h)
        x = self.attn_norm(hc_pre(h, pre_mix))
        # Reference: `super().forward(x, start_pos, pre_mix, None, main_x)`
        # -- "drafts are text: no VL bias", so the MoE gate never sees image_mask.
        h = hc_post(self.attn.draft_forward(x, main_x, ctx_offset, block_offset), h, post, comb)
        ffn_pre, post, comb = self.ffn_hc(h)
        x = self.ffn_norm(hc_pre(h, attn_pre))
        return hc_post(self.ffn(x), h, post, comb), ffn_pre


def dspark_forward_embed(stage0: DSparkStage, main_hidden: mx.array,
                         draft_ids: mx.array, embed_fn: Callable[[mx.array], mx.array],
                         hc_mult: int) -> Tuple[mx.array, mx.array]:
    """Reference ``DSparkBlock.forward_embed``: mix the concatenated target taps
    down to one hidden, embed the ``[anchor, noise, ...]`` block with the
    TARGET's embedding, and expand to the ``hc_mult`` hyper-connection lanes."""
    main_x = stage0.main_norm(stage0.main_proj(main_hidden))
    x = embed_fn(draft_ids)
    b, n, d = x.shape
    return mx.contiguous(mx.broadcast_to(x[:, :, None, :], (b, n, hc_mult, d))), main_x


def dspark_run_stages(stages: Sequence[DSparkStage], x: mx.array, main_x: mx.array,
                      ctx_offset: int, block_offset: int,
                      hc_mult: int) -> Tuple[mx.array, mx.array]:
    """Reference ``Transformer.forward_spec``'s stage loop: one identity pre-mix,
    then every stage in order, carrying the staggered mix forward."""
    pre_mix = make_identity_pre_mix(x.shape[0], x.shape[1], hc_mult)
    for stage in stages:
        x, pre_mix = stage(x, pre_mix, main_x, ctx_offset, block_offset)
    return x, pre_mix


def dspark_forward_head(
    last: DSparkStage,
    x: mx.array,
    pre_mix: mx.array,
    anchor: mx.array,
    head_fn: Callable[[mx.array], mx.array],
    block_size: int,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    greedy: bool = True,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Reference ``DSparkBlock.forward_head``.

    Collapse the lanes with the parameter-free ``hc_pre`` (see the module
    docstring on the absent ``hc_head``), project the stage norm through the
    TARGET's head for the whole block at once, then walk the block SEQUENTIALLY:
    position i's logits take the Markov bias of the token chosen at i-1
    (``output_ids[:, 0]`` is the anchor), a token is sampled, and its Markov
    embed is stacked for the confidence head.

    Returns ``(output_ids [B, block+1], logits [B, block, V], confidence
    [B, block], hidden [B, block, d])``; ``hidden`` is the collapsed pre-norm
    stream the confidence head scored, returned so callers need not recompute it.
    """
    hidden = hc_pre(x, pre_mix)
    # The reference ParallelHead contracts in fp32, as does the target LM.
    logits = head_fn(last.norm(hidden).astype(mx.float32))
    if block_size != logits.shape[1]:
        raise ValueError("DSpark head block size must match the stage output")
    # Cache one trace per sampler/mode; the loop is unrolled while tracing. Its
    # Markov dependency is required by reference model.py, not five target passes.
    key = (getattr(sampler, "dspark_compile_key", sampler), greedy)
    if getattr(last, "_walk_key", None) != key or not hasattr(last, "_walk"):
        def walk(scores, seed, embed_weight, head_weight):
            return dspark_markov_walk(scores, seed, embed_weight, head_weight,
                                      sampler, greedy)

        last._walk_key = key
        last._walk = (mx.compile(walk, inputs=mx.random.state, outputs=mx.random.state)
                      if os.environ.get("DSV41_DSPARK_COMPILE_HEAD", "0") == "1"
                      else walk)
    ids, corrected, embeds = last._walk(
        logits, anchor, last.markov_head.embed.weight, last.markov_head.head.weight)
    confidence = last.confidence_head(hidden, embeds)
    return ids, corrected, confidence, hidden


def dspark_markov_walk(logits, anchor, embed_weight, head_weight, sampler, greedy):
    """Fixed-size reference Markov recurrence, with no per-position host reads.

    All heavy stage and vocabulary projections preceding this recurrence are
    batched. Weight arguments are explicit so compiled traces cannot retain
    stale weights. RNG state is an input AND output of the compiled wrapper.
    """
    prev = anchor.reshape(-1).astype(mx.int32)
    ids, embeds, corrected = [prev], [], []
    for position in range(logits.shape[1]):
        embed = embed_weight[prev]
        bias = embed.astype(mx.float32) @ head_weight.astype(mx.float32).T
        step = logits[:, position].astype(mx.float32) + bias
        corrected.append(step)
        embeds.append(embed)
        prev = (mx.argmax(step, axis=-1) if greedy or sampler is None
                else sampler(step - mx.logsumexp(step, axis=-1, keepdims=True)).reshape(-1))
        prev = prev.astype(mx.int32)
        ids.append(prev)
    return mx.stack(ids, axis=1), mx.stack(corrected, axis=1), mx.stack(embeds, axis=1)


class DSparkStack(nn.Module):
    """The three DSpark stages as one module, so a target ``LanguageModel`` can
    hold them at ``self.dspark`` and the release's ``mtp.*`` tensors land at
    ``dspark.stages.<i>.*`` (the tree ``checkpoint_layout.py`` plans).

    The embedding and the LM head are NOT owned here: the reference shares the
    target's (``self.mtp[-1].embed = self.embed``; ``self.mtp[-1].head =
    self.head``), and so does the drafter, which binds them at runtime.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        n_stages = n_dspark_stages(config)
        if n_stages == 0:
            raise ValueError("This config does not describe DSpark stages")
        self.config = config
        self.stages: List[DSparkStage] = [DSparkStage(config, s, n_stages)
                                          for s in range(n_stages)]

    def forward_spec(self, main_hidden: mx.array, anchor: mx.array, position: int,
                     embed_fn, head_fn, *, block_size: Optional[int] = None,
                     sampler=None, greedy: bool = True):
        """Reference ``Transformer.forward_spec`` for one decode round.

        ``position`` is the anchor's absolute index (the reference's
        ``start_pos + seqlen``); ``main_hidden`` is the concatenated target taps
        for the committed context, already trimmed to the sliding window.
        """
        config = self.config
        block = int(block_size or config.dspark_block_size)
        draft_ids = dspark_noise_draft_ids(anchor, block, config.dspark_noise_token_id)
        x, main_x = dspark_forward_embed(
            self.stages[0], main_hidden, draft_ids, embed_fn, config.hc_mult)
        ctx_offset = int(position) - main_x.shape[1]
        x, pre_mix = dspark_run_stages(
            self.stages, x, main_x, ctx_offset, int(position), config.hc_mult)
        return dspark_forward_head(self.stages[-1], x, pre_mix, anchor, head_fn,
                                   block, sampler, greedy)
