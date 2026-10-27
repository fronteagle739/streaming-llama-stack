"""DeepSeek-V4.1-Flash DSpark speculative draft head (MLX).

The three ``mtp.*`` stages of the release, driven as a drafter. The stage
modules themselves live in ``mlx_vlm/models/deepseek_v41/dspark.py`` (which
reuses the backbone's attention / MoE / hyper-connection / RoPE / sparse-attention
/ QAT implementations); this file is the round interface
``mlx_vlm/speculative/dspark.py`` drives, identical in shape to our V4 DSpark
drafter so that loop needs no change.

One speculative round (reference ``Transformer.forward_spec`` + ``forward_head``):

  1. The target runs and taps the ATTENTION INPUT of layers 37/38/39 -- the
     reference comment is explicit, "the MTP head reads the attention input of
     its target layers, not their output" -- means each tap over the hc lanes
     and concatenates them: ``main_hidden`` [B, L, hidden*3]. That capture is
     wired in ``models/deepseek_v41/language.py`` (``capture_layer_ids`` /
     ``capture_sink``); the target's sampled token is the block ``anchor``.
  2. ``main_x = stage0.main_norm(stage0.main_proj(main_hidden))`` over the last
     ``sliding_window`` positions.
  3. The draft ids are ``[anchor, noise, ..., noise]`` (noise id 128799), embedded
     with the TARGET embedding, expanded to the ``hc_mult`` lanes and pushed
     through the three stages in ONE pass: every block query attends over the
     whole ``main_x`` window plus all block keys (``get_dspark_topk_idxs``
     selects exactly that, and the reference's sparse kernel adds no mask).
  4. The last stage collapses the lanes with the parameter-free ``hc_pre``,
     projects through the TARGET ``lm_head`` and sharpens the block sequentially
     with the Markov head; the confidence head scores each position from
     ``[hidden ; markov_embed]`` and truncates the proposal at the first
     ``sigmoid(confidence) < confidence_threshold``.

ACCEPTANCE (``speculative/dspark.py``, pure helpers in
``speculative/dspark_bookkeeping.py``): the verify input is ``[anchor, *drafts]``
(``1 + proposal`` cells); the target's greedy/sampled tokens are walked against
the drafts and the longest matching prefix is accepted, the target's token at the
first mismatch becoming the bonus and the next anchor; ``proposal - accepted``
rejected cells are trimmed off the target cache (``DeepseekV41Cache.trim`` is
exact -- it rebuilds the pooled compressor tail -- so no replay snapshot is
needed); the next anchor sits at ``position + accepted + 1``, which equals the
committed context length and the target cache offset.

WEIGHT-FREE / MAC-GATED: the parameter surface is pinned against the release's
2,401 ``mtp.*`` tensors by ``models/deepseek_v41/test_checkpoint_layout_v41.py``
and ``test_dspark_dsv41.py``; the MLX numerics need the Mac and the weights.
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ....models.deepseek_v41.checkpoint_layout import sanitize_dspark_weights
from ....models.deepseek_v41.dspark import (
    DSparkStage,
    dspark_forward_embed,
    dspark_forward_head,
    dspark_noise_draft_ids,
    dspark_run_stages,
)
from .config import DeepseekV41DSparkConfig

DRAFTER_TREE_PREFIX = "stages."


def make_quantization_config(model):
    """The release's own scheme, restricted to the drafter tree: fp4 routed
    experts, fp8 shared experts / attention projections / ``main_proj``, and the
    bf16 Markov and confidence heads left alone."""
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    config = {"group_size": 64, "bits": 8, "mode": "affine"}
    for key, _ in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if ".ffn.switch_mlp." in key and key.endswith("_proj"):
            config[key] = mxfp4
        elif (".ffn.shared_experts." in key
              or (".attn.w" in key and not key.endswith(".attn.wo_a"))
              or key.endswith(".main_proj")):
            config[key] = mxfp8
    return config


class DeepseekV41DSparkDraftModel(nn.Module):
    preserves_target_seed = True
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: DeepseekV41DSparkConfig, *, stack=None):
        super().__init__()
        text_config = config.text_config
        if text_config is None:
            raise ValueError("DeepseekV41DSparkConfig.text_config must be set")
        self.config = config
        self.args = text_config
        self.window_size = int(text_config.sliding_window)
        self.block_size = int(config.block_size)
        self.noise_token_id = int(config.noise_token_id)
        self.target_layer_ids = list(config.target_layer_ids)
        self.confidence_threshold = float(config.confidence_threshold)
        n_stages = int(config.n_draft_layers)
        self.stages: List[DSparkStage] = list(stack.stages) if stack is not None else [
            DSparkStage(text_config, s, n_stages) for s in range(n_stages)
        ]
        # Shared with the target and bound at runtime: the release ships no
        # mtp.*.embed / mtp.*.head (the reference assigns the Transformer's own).
        self._input_embed = None
        self._lm_head_fn = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []
        self.draft_calls = 0

    # ------------------------------------------------------------------ binding
    def bind(self, target_model) -> "DeepseekV41DSparkDraftModel":
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (hasattr(target_model, "language_model")
              and hasattr(target_model.language_model, "model")
              and hasattr(target_model.language_model.model, "embed_tokens")):
            inner = target_model.language_model.model
        if inner is None:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self._input_embed = inner.embed_tokens

        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (getattr(target_model, "lm_head", None)
                            or getattr(lm, "lm_head", None)
                            or self._input_embed.as_linear)
        return self

    def adopt(self, stack) -> "DeepseekV41DSparkDraftModel":
        """Reuse a target-resident ``DSparkStack`` (the ``dspark.stages.*`` tree
        planned by ``checkpoint_layout``) instead of this drafter's own stages,
        so a single-checkpoint load needs no second copy of 7.5 GiB."""
        self.stages = list(stack.stages)
        return self

    def make_cache(self) -> List[Any]:
        """No draft-side cache: the window KV is recomputed from ``main_x`` every
        round (see ``DSparkAttention``'s DESIGN NOTE)."""
        return []

    def reset(self, target_model) -> List[Any]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self.draft_calls = 0
        return self.make_cache()

    # ---------------------------------------------------------------- capture
    @property
    def capture_layer_ids(self) -> List[int]:
        """Target layers whose attention input feeds ``main_hidden``."""
        return list(self.target_layer_ids)

    def context_from_captures(self, captures: List[mx.array]) -> mx.array:
        """Reference ``h.mean(dim=2)`` per tap then ``torch.cat(main_hiddens,
        dim=-1)``: [B, L, hc, d] x n_taps -> [B, L, d * n_taps]."""
        lanes = [h.mean(axis=2) if h.ndim == 4 else h for h in captures]
        return mx.concatenate(lanes, axis=-1)

    # ----------------------------------------------------------------- drafting
    def _confident_prefix_length(self, confidence: mx.array, threshold: float) -> int:
        if threshold <= 0.0:
            return int(confidence.shape[1])
        below = mx.sigmoid(confidence.astype(mx.float32)) < threshold
        # One host boundary only when adaptive verification is explicitly enabled.
        # Appending a sentinel handles full confidence without a second reduction.
        stop = mx.concatenate([below, mx.ones((below.shape[0], 1), dtype=mx.bool_)], axis=1)
        return int(mx.min(mx.argmax(stop.astype(mx.int32), axis=1)).item())

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
        """Draft one DSpark block.

        ``position`` is the absolute index of the anchor == the number of
        committed context tokens (the anchor is not in the target cache yet; the
        verify pass writes it). It feeds only the RoPE offsets: the ``main_x``
        window is roped from ``position - len(window)`` and the block from
        ``position``, mirroring the reference's context freqs at
        ``start_pos : start_pos + seqlen`` and block freqs at
        ``start_pos + seqlen : + block_size``.

        Returns ``draft_tokens`` [B, n] (n = the confident prefix length),
        ``output_ids`` [B, block+1] (anchor + the full block), ``confidence``
        [B, block], Markov-corrected ``logits`` [B, block, V] and
        ``draft_token_count``.
        """
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError("bind(target_model) must be called before drafting.")
        self.draft_calls += 1
        position = int(position)
        if position < 1:
            raise ValueError("DSpark needs at least one committed context token")
        block = int(block_size or self.block_size)
        threshold = (self.confidence_threshold if confidence_threshold is None
                     else float(confidence_threshold))

        main_hidden = main_hidden[:, -self.window_size:]
        draft_ids = dspark_noise_draft_ids(anchor, block, self.noise_token_id)
        x, main_x = dspark_forward_embed(self.stages[0], main_hidden, draft_ids,
                                         self._input_embed, self.args.hc_mult)
        ctx_offset = position - main_x.shape[1]
        x, pre_mix = dspark_run_stages(self.stages, x, main_x, ctx_offset, position,
                                       self.args.hc_mult)
        output_ids, logits, confidence, _hidden = dspark_forward_head(
            self.stages[-1], x, pre_mix, anchor, self._lm_head_fn, block, sampler, greedy)

        proposal = self._confident_prefix_length(confidence, threshold)
        self.draft_lens.append(proposal)
        return {
            "draft_tokens": output_ids[:, 1:1 + proposal],
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
            protected = (".wo_a", ".attn_hc", ".ffn_hc", ".ffn.gate",
                         ".markov_head.", ".confidence_head.")
            if any(name in path for name in protected):
                return False
            return quantization_config.get(path, True)

        return predicate

    @property
    def cast_predicate(self):
        def predicate(key):
            protected = ("attn_sink", "e_score_correction_bias", "bias_vl",
                         ".attn_hc.", ".ffn_hc.")
            return not any(name in key for name in protected)

        return predicate

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Release ``mtp.<stage>.*`` (or a target-tree ``dspark.stages.<i>.*``)
        onto this module's ``stages.<i>.*``, with the same fp8 pair / fp4
        expert-stack / ``wo_a`` dequantization the backbone gets."""
        return sanitize_dspark_weights(weights, self.args, DRAFTER_TREE_PREFIX)


Model = DeepseekV41DSparkDraftModel
