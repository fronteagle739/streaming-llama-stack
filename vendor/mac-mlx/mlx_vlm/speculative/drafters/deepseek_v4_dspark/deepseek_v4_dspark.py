"""DeepSeek-V4-Flash-DSpark speculative draft head (MLX).

Faithful MLX port of the DSpark draft head shipped under the ``mtp.*`` namespace
of ``deepseek-ai/DeepSeek-V4-Flash-DSpark``. The reference is the release's own
``inference/model.py`` (classes ``DSparkBlock`` / ``DSparkAttention`` /
``DSparkMarkovHead`` / ``DSparkConfidenceHead`` and ``Transformer.forward_spec``
/ ``forward_head``); the runtime acceptance/truncation semantics match DeepSeek's
``DeepSpec/deepspec/eval/dspark/draft_ops.py``.

Algorithm (one speculative round, mirroring ``forward_spec`` + ``forward_head``):

  1. The target model runs and, at each of ``target_layer_ids`` (40/41/42), taps
     its hyper-connection hidden state, means it over the ``hc_mult`` lanes and
     concatenates the taps -> ``main_hidden`` [B, L, hidden*3]. (This capture is
     wired in ``deepseek_v4/language.py`` via ``capture_layer_ids``; the target's
     sampled next token is the block ``anchor``.)
  2. ``main_x = stage0.main_norm(stage0.main_proj(main_hidden))``  -> [B, L, d].
  3. Build the draft input ids ``[anchor, noise, ..., noise]`` (length block),
     embed with the *target* embedding, expand to ``hc_mult`` lanes, and run the
     ``n_draft_layers`` DSpark blocks. Every block attends over the sliding
     window of ``main_x``-derived KV (the projected target context) plus its own
     block-causal KV -> the whole block is produced in ONE pass.
  4. The last stage collapses lanes (its own ``hc_head``), projects to the shared
     ``lm_head`` -> ``base_logits`` [B, block, V]. A first-order Markov head then
     sharpens the block SEQUENTIALLY: position i's logits get the transition bias
     of the previously chosen token, then a token is sampled; ``x_0`` is the
     anchor. The confidence head scores each position from
     ``[hidden ; markov_embed]``; the drafted prefix is truncated at the first
     position whose ``sigmoid(confidence) < confidence_threshold``.

WEIGHT-FREE / MAC-GATED STATUS: the parameter surface is pinned bit-for-bit
against the real 72,317-tensor release manifest by
``mlx_vlm/models/deepseek_v4/test_checkpoint_layout.py`` (``mtp.*`` keys), and
the Markov/confidence/walk math is pinned by the numpy oracle
(``reference_dsv4.py`` + ``test_reference_numpy.py``). The MLX modules here are
structurally exercised weight-free by
``mlx_vlm/tests/test_speculative.py`` (shapes/dtypes/param-surface). End-to-end
NUMERIC validation and generation-loop wiring require MLX + the ~167 GB release
weights on the Mac (see the module README / final handoff for the exact remaining
integration + run commands).

DESIGN NOTE (documented simplification, correctness-preserving): the reference
seeds a persistent sliding KV cache from ``main_x`` during target prefill and
appends one column per decode step. Here each draft round recomputes the context
KV from the supplied ``main_x`` window (``wkv``/``kv_norm`` on <=window_size
positions) instead of persisting a draft KV cache. The context KV is a
deterministic function of ``main_x``, so this is numerically identical; it trades
a small recompute for not maintaining draft-side cache/rope bookkeeping. The
reference's block ``sparse_attn`` top-k selects the FULL window + FULL block (see
``get_dspark_topk_idxs``), so the dense window+block attention used here is
exactly faithful, not an approximation.
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ....models.base import scaled_dot_product_attention
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.hyper_connection import (
    HyperConnection,
    HyperHead,
    hc_expand,
)
from ....models.deepseek_v4.language import (
    DeepseekV4MoE,
    LocalAttention,
)
from .config import DeepseekV4DSparkConfig


def make_quantization_config(model):
    """Same fp4-expert / fp8-attention scheme as the target + MTP drafter."""
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    mxfp8_modules = {
        k: mxfp8
        for k, _ in flat_modules
        if ".ffn.shared_experts." in k
        or ".attn.w" in k
        or k.endswith(".main_proj")
    }
    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **mxfp8_modules,
    }


class DSparkMarkovHead(nn.Module):
    """First-order transition head. ``markov_w1`` embeds the previous token to a
    rank-``markov_rank`` code; ``markov_w2`` projects it back to a full-vocab
    logit bias. Mirrors reference ``DSparkMarkovHead``: bias = W2(W1[t]),
    embed = W1[t]."""

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> Tuple[mx.array, mx.array]:
        embed = self.markov_w1(token_ids)
        return self.markov_w2(embed), embed


class DSparkConfidenceHead(nn.Module):
    """c_i = sigmoid(w . [hidden_i ; markov_embed_i]). Reference
    ``DSparkConfidenceHead`` (Linear(dim+rank -> 1, no bias), fp32)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        feats = mx.concatenate(
            [hidden.astype(mx.float32), markov_embed.astype(mx.float32)], axis=-1
        )
        return self.proj(feats).squeeze(-1)


class DSparkAttention(LocalAttention):
    """Sliding-window (ratio-0) MLA attention whose CONTEXT keys/values come from
    the projected target hidden ``main_x`` rather than from re-embedded context
    tokens. Reuses ``LocalAttention``'s exact parameter surface (wq_a/q_norm/
    wq_b/wkv/kv_norm/wo_a[o_groups x o_lora]/wo_b/attn_sink/rope); only the
    forward differs. Mirrors reference ``DSparkAttention.forward``."""

    def draft_forward(
        self,
        x: mx.array,
        main_x: mx.array,
        ctx_offset: int,
        block_offset: int,
    ) -> mx.array:
        B, block, _ = x.shape
        rd = self.config.qk_rope_head_dim

        # Context KV from the projected target hidden (main_x): kv_norm(wkv(.)),
        # rope only the rope tail at the context's absolute positions.
        ckv = self.kv_norm(self.wkv(main_x)).reshape(B, 1, main_x.shape[1], self.head_dim)
        ckv = self._rope_tail(ckv, ctx_offset, rd)

        # Block query / KV from the draft-token stream.
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, block, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self._rope_tail(q, block_offset, rd)

        bkv = self.kv_norm(self.wkv(x)).reshape(B, 1, block, self.head_dim)
        bkv = self._rope_tail(bkv, block_offset, rd)

        kv = mx.concatenate([ckv, bkv], axis=2)

        # Additive mask [1,1,block,ctx+block]: block queries see ALL context
        # (the sliding window) and are causal over the block itself.
        ctx_len = main_x.shape[1]
        block_causal = mx.triu(
            mx.full((block, block), mx.finfo(mx.float32).min, dtype=mx.float32),
            k=1,
        )
        ctx_zeros = mx.zeros((block, ctx_len), dtype=mx.float32)
        mask = mx.concatenate([ctx_zeros, block_causal], axis=-1)[None, None]

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self._rope_tail(out, block_offset, rd, inverse=True)

        out = out.reshape(B, self.o_groups, -1, block, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out

    def _rope_tail(self, x, offset, rope_dim, inverse: bool = False):
        # DSpark ratio-0 attention ropes only the last ``rope_dim`` dims of the
        # head (the NoPE prefix passes through); DeepseekV4RoPE already inserts
        # +inf freqs for the NoPE prefix, so applying it to the full head_dim is
        # equivalent to roping only the tail.
        return self.rope(x, offset, inverse=inverse)


class DSparkStage(nn.Module):
    """One DSpark draft block == a DeepseekV4 hyper-connection block with ratio-0
    ``DSparkAttention`` + bias-routed MoE, plus the stage-specific heads (stage 0
    owns main_proj/main_norm; the last stage owns hc_head/norm/markov/confidence).
    Mirrors reference ``DSparkBlock``."""

    def __init__(self, dspark_config: DeepseekV4DSparkConfig, stage_idx: int):
        super().__init__()
        config = dspark_config.text_config
        n_layers = config.num_hidden_layers
        # layer_idx >= num_hash_layers -> the MoE gate is BIAS-routed (never hash),
        # exactly as the draft stages ship (mtp.{s}.ffn.gate.bias, no tid2eid).
        layer_idx = n_layers + stage_idx
        self.stage_idx = stage_idx

        self.attn = DSparkAttention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

        if stage_idx == 0:
            n_taps = len(dspark_config.target_layer_ids)
            self.main_proj = nn.Linear(
                config.hidden_size * n_taps, config.hidden_size, bias=False
            )
            self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if stage_idx == dspark_config.n_draft_layers - 1:
            self.hc_head = HyperHead(config)
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                config.vocab_size, dspark_config.markov_rank
            )
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + dspark_config.markov_rank
            )

    def __call__(
        self,
        x: mx.array,
        main_x: mx.array,
        ctx_offset: int,
        block_offset: int,
        draft_ids: mx.array,
    ) -> mx.array:
        residual = x
        y, post, comb = self.attn_hc(x)
        y = self.attn.draft_forward(
            self.attn_norm(y), main_x, ctx_offset, block_offset
        )
        x = hc_expand(y, residual, post, comb)

        residual = x
        y, post, comb = self.ffn_hc(x)
        y = self.ffn(self.ffn_norm(y), draft_ids)
        return hc_expand(y, residual, post, comb)


class DeepseekV4DSparkDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: DeepseekV4DSparkConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("DeepseekV4DSparkConfig.text_config must be set")
        self.args = text_config
        self.window_size = text_config.sliding_window
        self.block_size = int(config.block_size)
        self.noise_token_id = int(config.noise_token_id)
        self.target_layer_ids = list(config.target_layer_ids)
        self.confidence_threshold = float(config.confidence_threshold)

        self.stages = [
            DSparkStage(config, s) for s in range(config.n_draft_layers)
        ]

        # Shared with the target (bound at runtime; no mtp.* embed/head ship).
        self._input_embed = None
        self._lm_head_fn = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    # ------------------------------------------------------------------ binding
    def bind(self, target_model) -> "DeepseekV4DSparkDraftModel":
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        if inner is None:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self._input_embed = inner.embed_tokens

        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self) -> List[RotatingKVCache]:
        # Draft-side rotating caches are provided for a future persistent-KV
        # optimisation; the default draft_block recomputes context KV from the
        # supplied main_x window, so this cache is not required for correctness.
        return [
            RotatingKVCache(max_size=self.window_size)
            for _ in range(len(self.stages))
        ]

    def reset(self, target_model) -> List[RotatingKVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    # ---------------------------------------------------------------- capture
    @property
    def capture_layer_ids(self) -> List[int]:
        """Target layer indices to tap for ``main_hidden`` (see language.py
        ``capture_layer_ids``)."""
        return list(self.target_layer_ids)

    def context_from_captures(self, captures: List[mx.array]) -> mx.array:
        """Reduce the target-layer hidden taps to ``main_hidden``: mean over the
        hyper-connection lanes ([B,L,hc,d] -> [B,L,d]) then concat over taps ->
        [B, L, d*len(target_layer_ids)] (reference ``h.mean(dim=2)`` +
        ``torch.cat(main_hiddens, dim=-1)``)."""
        lanes_meaned = []
        for h in captures:
            if h.ndim == 4:
                h = h.mean(axis=2)
            lanes_meaned.append(h)
        return mx.concatenate(lanes_meaned, axis=-1)

    # ----------------------------------------------------------------- drafting
    def _draft_input_ids(self, anchor: mx.array, block: int) -> mx.array:
        anchor = anchor.reshape(-1, 1).astype(mx.int32)
        B = anchor.shape[0]
        if block == 1:
            return anchor
        noise = mx.full((B, block - 1), self.noise_token_id, dtype=mx.int32)
        return mx.concatenate([anchor, noise], axis=1)

    def _backbone(
        self,
        anchor: mx.array,
        main_hidden: mx.array,
        position: int,
        block: int,
    ) -> Tuple[mx.array, mx.array]:
        """Run stage0 embed + the DSpark block stack. Returns (pre-head hidden
        [B, block, hc, d], draft_ids [B, block])."""
        if self._input_embed is None:
            raise RuntimeError("bind(target_model) must be called before drafting.")

        stage0 = self.stages[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))
        main_x = main_x[:, -self.window_size :]

        draft_ids = self._draft_input_ids(anchor, block)
        x = self._input_embed(draft_ids)
        x = mx.broadcast_to(
            x[:, :, None, :],
            (x.shape[0], x.shape[1], self.args.hc_mult, x.shape[2]),
        )
        x = mx.contiguous(x)

        ctx_offset = int(position) - main_x.shape[1]
        block_offset = int(position)
        for stage in self.stages:
            x = stage(x, main_x, ctx_offset, block_offset, draft_ids)
        return x, draft_ids

    def _markov_confidence(
        self,
        hidden: mx.array,
        base_logits: mx.array,
        anchor: mx.array,
        sampler,
        greedy: bool,
        block: int,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Sequential Markov sharpening + confidence, mirroring reference
        ``forward_head`` and the numpy oracle ``markov_sample_block`` /
        ``confidence_scores``."""
        last = self.stages[-1]
        prev = anchor.reshape(-1).astype(mx.int32)
        ids = [prev]
        markov_embeds = []
        corrected = []
        for i in range(block):
            bias, embed = last.markov_head(prev)
            step = base_logits[:, i].astype(mx.float32) + bias.astype(mx.float32)
            corrected.append(step)
            markov_embeds.append(embed)
            if greedy or sampler is None:
                nxt = mx.argmax(step, axis=-1).astype(mx.int32)
            else:
                nxt = sampler(step).reshape(-1).astype(mx.int32)
            ids.append(nxt)
            prev = nxt
        output_ids = mx.stack(ids, axis=1)
        base_logits = mx.stack(corrected, axis=1)
        markov_embed = mx.stack(markov_embeds, axis=1)
        confidence = last.confidence_head(hidden, markov_embed)
        return output_ids, base_logits, confidence

    def _confident_prefix_length(self, confidence: mx.array, threshold: float) -> int:
        if threshold <= 0.0:
            return int(confidence.shape[1])
        below = (mx.sigmoid(confidence[0].astype(mx.float32)) < threshold).astype(
            mx.int32
        )
        if int(below.sum().item()) == 0:
            return int(confidence.shape[1])
        return int(mx.argmax(below).item())

    def draft_block(
        self,
        anchor: mx.array,
        main_hidden: mx.array,
        position: int = 0,
        *,
        sampler=None,
        greedy: bool = True,
        block_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Draft one DSpark block. Returns a dict with:
        ``draft_tokens`` [B, n] (n = confident prefix length), ``output_ids``
        [B, block+1] (anchor + full block), ``confidence`` [B, block],
        ``logits`` [B, block, V] (Markov-corrected), and ``draft_token_count``.

        ``position`` is the absolute token position of the block anchor
        (== the number of confirmed context tokens). It only feeds the RoPE
        offsets: the ``main_x`` sliding window is roped at ``position - len(window)``
        and the block at ``position`` (reference: context freqs at
        ``start_pos:start_pos+seqlen``, block freqs at ``start_pos+seqlen:+block``).
        DOCUMENTED AMBIGUITY (Mac+weight-gated to pin down): the exact off-by-one
        convention between the target's ``cache.offset`` and this ``position``
        is the one runtime alignment a needle/coherence test on the Mac must
        confirm; shapes are unaffected either way.
        """
        block = int(block_size or self.block_size)
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        hidden4d, _ = self._backbone(anchor, main_hidden, position, block)

        last = self.stages[-1]
        hidden = last.hc_head(hidden4d)
        base_logits = self._lm_head_fn(last.norm(hidden))

        output_ids, logits, confidence = self._markov_confidence(
            hidden, base_logits, anchor, sampler, greedy, block
        )
        proposal = self._confident_prefix_length(confidence, threshold)
        draft_tokens = output_ids[:, 1 : 1 + proposal]
        self.draft_lens.append(proposal)
        return {
            "draft_tokens": draft_tokens,
            "output_ids": output_ids,
            "confidence": confidence,
            "logits": logits,
            "draft_token_count": proposal,
        }

    # ----------------------------------------------------------------- weights
    @property
    def quant_predicate(self):
        quantization_config = make_quantization_config(self)

        def predicate(path, _):
            return quantization_config.get(path, True)

        return predicate

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map the release ``mtp.*`` checkpoint surface onto this module tree.

        ``mtp.{s}.<...>`` -> ``stages.{s}.<...>`` with the same field remaps the
        target uses: hc_{attn,ffn}_{fn,base,scale} -> {attn,ffn}_hc.{...};
        ffn.gate.bias -> ffn.gate.e_score_correction_bias; shared-expert w1/w2/w3
        -> gate/down/up_proj; routed experts stacked into switch_mlp; wo_a folded
        to [o_groups, o_lora, -1]; hc_head_{fn,base,scale} -> hc_head.{fn,base,
        scale}. fp8 (uint8 + block-128 ue8m0 scale) and fp4 (packed 2/byte)
        weights are kept native-quantized (view uint32 + expand scales), matching
        the target ``LanguageModel.sanitize``.
        """
        n_stages = self.config.n_draft_layers
        n_experts = self.args.n_routed_experts

        # 1) strip the mtp. prefix (keep the stage index).
        stripped = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                stripped[k[len("mtp.") :]] = v
            else:
                stripped[k] = v
        weights = stripped

        # 2) quantized scale handling (fp4 experts / fp8 linears), identical to
        #    the target model's sanitize.
        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)
            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue
            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        # 3) key remap: {s}.<field> -> stages.{s}.<field>.
        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            parts = k.split(".", 1)
            if not (parts[0].isdigit() and len(parts) == 2):
                remapped[k] = v
                continue
            s, rest = parts[0], parts[1]
            nk = rest
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            if nk.startswith("ffn.gate.bias"):
                nk = nk.replace("ffn.gate.bias", "ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f"hc_{sub}_{param}", f"{sub}_hc.{param}")
            nk = nk.replace("hc_head_fn", "hc_head.fn")
            nk = nk.replace("hc_head_base", "hc_head.base")
            nk = nk.replace("hc_head_scale", "hc_head.scale")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[f"stages.{s}.{nk}"] = v
        weights = remapped

        # 4) stack routed experts into switch_mlp; fold wo_a to grouped shape.
        for s in range(n_stages):
            eprefix = f"stages.{s}.ffn.experts"
            for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                for suffix in ("weight", "scales"):
                    key0 = f"{eprefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{eprefix}.{e}.{src}.{suffix}")
                            for e in range(n_experts)
                        ]
                        weights[
                            f"stages.{s}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)
            woa = f"stages.{s}.attn.wo_a"
            for key in (f"{woa}.weight", f"{woa}.scales", f"{woa}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )
        return weights
