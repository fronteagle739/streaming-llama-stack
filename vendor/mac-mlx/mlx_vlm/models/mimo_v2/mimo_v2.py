"""Text bridge for MiMo V2.5/V2.6, with explicit pipeline layer ownership.

Architecture source: mlx-lm PR 1219 (pinned in _arch_v26.py). Original media
is encoded separately by multimodal.py and injected before the first layer.
"""

import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache, QuantizedKVCache
from ._arch_v26 import Model as _MimoLMBase
from ._arch_v26 import ModelArgs as _LMArgs

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig


class _MimoServeModel(_MimoLMBase):
    """Adapt the pinned architecture's embedding and logits contract for MLX VLM."""

    def __init__(self, config):
        super().__init__(config)
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = len(self.model.layers)
        self.model.pipeline = self.pipeline

    @property
    def pipeline_layers(self):
        return self.model.layers[self.start_idx : self.end_idx]

    def pipeline(self, group):
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        n_layers = len(self.model.layers)

        raw_counts = os.environ.get("MLX_VLM_PIPELINE_LAYERS") or os.environ.get(
            "MLX_PIPELINE_LAYERS", ""
        )
        if raw_counts:
            counts = [int(part.strip()) for part in raw_counts.split(",") if part.strip()]
            if len(counts) != self.pipeline_size:
                raise ValueError(
                    "MLX_VLM_PIPELINE_LAYERS must provide one layer count per rank "
                    f"(got {len(counts)} for world size {self.pipeline_size})"
                )
            if sum(counts) != n_layers:
                raise ValueError(
                    "MLX_VLM_PIPELINE_LAYERS must sum to the model layer count "
                    f"(got {sum(counts)} for {n_layers} layers)"
                )
        else:
            base = n_layers // self.pipeline_size
            extra = n_layers % self.pipeline_size
            counts = [
                base + (1 if rank < extra else 0)
                for rank in range(self.pipeline_size)
            ]

        end_idx = n_layers - sum(counts[: self.pipeline_rank])
        start_idx = end_idx - counts[self.pipeline_rank]
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.model.layers = self.model.layers[: self.end_idx]
        self.model.layers[: self.start_idx] = [None] * self.start_idx
        self.model.start_idx = self.start_idx
        self.model.end_idx = self.end_idx
        self.model.pipeline_rank = self.pipeline_rank
        self.model.pipeline_size = self.pipeline_size

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        skip_logits = kwargs.pop("skip_logits", False)
        logits = super().__call__(inputs, cache, input_embeddings=inputs_embeds)
        return LanguageModelOutput(logits=None if skip_logits else logits)

    def make_cache(self):
        caches = []
        for layer in self.pipeline_layers:
            if layer.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size))
            else:
                caches.append(QuantizedKVCache(group_size=64, bits=8)
                              if os.environ.get("MLX_MIMO_KV_BITS") == "8" else KVCache())
        return caches


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        # from_dict filters to the pinned architecture's ModelArgs fields.
        args = config.to_dict()
        args.setdefault("n_shared_experts", None)
        args.setdefault("routed_scaling_factor", None)
        self.language_model = _MimoServeModel(_LMArgs.from_dict(args))

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        out = self.language_model(input_ids, cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        # The inferencerlabs "-INF" checkpoints arrive ALREADY in mlx post-sanitize layout: experts
        # pre-stacked into `switch_mlp`, no per-expert (`mlp.experts.N`) tensors, no fp8 `weight_scale_inv`.
        # mlx-lm's mimo_v2_flash `sanitize` is built to transform a RAW HF checkpoint (fp8 dequant + stack
        # experts) — re-running it on already-processed weights corrupts them. So only run it on a genuinely
        # RAW checkpoint; skip when the post-sanitize keys are present (matches the glm_moe_dsa bridge).
        already_sanitized = any(".switch_mlp." in k for k in weights)
        if not already_sanitized:
            weights = self.language_model.sanitize(weights)

        def remap(key: str) -> str:
            if key.startswith("language_model."):
                return key
            if key.startswith("model.") or key.startswith("lm_head."):
                return f"language_model.{key}"
            return key

        return {remap(k): v for k, v in weights.items()}

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()
