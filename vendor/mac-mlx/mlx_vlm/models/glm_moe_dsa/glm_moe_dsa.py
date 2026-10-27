"""GLM-5.1 (`glm_moe_dsa`) bridge for mlx-vlm.

The architecture (MLA q_a/q_b + lightning Indexer + MoE + `unembed_out`) already lives, complete, in
mlx-lm's `deepseek_v32` — and `glm_moe_dsa.Model(DSV32Model)` reuses it verbatim, including a `sanitize`
that builds `unembed_out` and splits the MLA `kv_b`. The ONLY reason stock mlx-lm can't serve the
inferencerlabs GLM-5.1-4.8bit-INF checkpoint is the per-tensor mixed quant (config `quantization_config={}`,
plain keys → no quantization applied). Our mlx-vlm `load_model` HAS that mixed-quant path
(`inf_decode.quant_groups_from_model`). So this module is a THIN bridge: instantiate mlx-lm's arch as the
`language_model`, expose the mlx-vlm Model contract, and let mlx-vlm's loader quantize + load.
"""

import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

# ModelArgs is version-stable (glm_moe_dsa.py is identical across 0.30.7/0.31.3); only the deepseek_v32
# FORWARD (the lightning indexer's RoPE) drifted, so we pick the Model class by checkpoint version below.
from mlx_lm.models.glm_moe_dsa import Model as _CurrentLMModel
from mlx_lm.models.glm_moe_dsa import ModelArgs as _LMArgs
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from . import native_ops as _native_ops
from .config import ModelConfig


def _iter_leaf_caches(caches):
    if caches is None:
        return
    if isinstance(caches, (list, tuple)):
        for item in caches:
            yield from _iter_leaf_caches(item)
        return
    yield caches


def _accepted_to_list(accepted) -> list[int]:
    if isinstance(accepted, int):
        return [accepted]
    try:
        return [int(v) for v in accepted.tolist()]
    except Exception:
        try:
            return [int(v) for v in accepted]
        except TypeError:
            return [int(accepted)]


def _lm_args_from_config(config: ModelConfig) -> _LMArgs:
    params = config.to_dict()
    for key in ("n_shared_experts", "n_routed_experts"):
        if hasattr(config, key):
            params.setdefault(key, getattr(config, key))
    if getattr(config, "rope_parameters", None) is None:
        params["rope_parameters"] = {
            "rope_theta": getattr(config, "rope_theta", None) or 10000.0
        }
    else:
        params.setdefault("rope_parameters", config.rope_parameters)
    return _LMArgs.from_dict(params)


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _env_off(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "0", "false", "no", "off",
    }


def _use_legacy_indexer(mlx_sanitized: Optional[str]) -> bool:
    """The lightning-indexer RoPE application changed AFTER mlx-lm 0.30.7: 0.31.x applies RoPE to the FULL
    q/k in the indexer, 0.30.x only to the rope-split (q_pe/k_pe). A checkpoint SANITIZED with 0.30.x needs
    the legacy partial-rope indexer (else the sparse-attention scores are wrong → garbage). ONLY 0.30.x (and
    older) get the legacy arch; 0.31+ and unknown/absent default to the CURRENT mlx-lm deepseek_v32 — so
    newer quants are never wrongly downgraded."""
    if not mlx_sanitized:
        return False
    try:
        major, minor = (int(x) for x in str(mlx_sanitized).split(".")[:2])
    except Exception:  # noqa: BLE001 — unparseable version -> current arch (safe default)
        return False
    return (major, minor) <= (0, 30)


class _NoOpIndexer(nn.Module):
    def __init__(self, wrapped=None):
        super().__init__()
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        # deepseek_v32 ties the attention cache to the indexer cache with mx.depends.
        # Preserve that cache update while forcing dense attention by dropping top-k.
        if self.wrapped is not None:
            self.wrapped(*args, **kwargs)
        return None


class _GlmIndexer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        rope_scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None)
        self.rope = initialize_rope(
            dims=self.rope_head_dim,
            base=_rope_theta(config),
            traditional=bool(getattr(config, "indexer_rope_interleave", True)),
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        b, s, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        q_pe, q_nope = mx.split(q, [self.rope_head_dim], axis=-1)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset=offset)
        q = mx.concatenate([q_pe, q_nope], axis=-1)

        k = self.wk(x)
        k = self.k_norm(k)
        k = mx.reshape(k, (b, 1, s, self.head_dim))
        k_pe, k_nope = mx.split(k, [self.rope_head_dim], axis=-1)
        k_pe = self.rope(k_pe, offset=offset)
        k = mx.concatenate([k_pe, k_nope], axis=-1)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
        if _env_flag("MLX_GLM_DSA_DECODE_ONLY") and s > 1:
            return None
        if k.shape[2] <= self.index_topk:
            return None
        if _env_flag("MLX_GLM_DSA_RECENT_TOPK"):
            q_pos = mx.arange(offset, offset + s) if not isinstance(offset, mx.array) else offset + mx.arange(s)
            start = mx.maximum(q_pos - self.index_topk + 1, 0)
            idx = start[:, None] + mx.arange(self.index_topk)[None, :]
            idx = mx.minimum(idx, k.shape[2] - 1)
            return idx[None, None, :, :]
        weights_raw = self.weights_proj(x) * (self.n_heads**-0.5)  # [B, L, H]

        # --- optional native fused indexer path (oMLX GLM DSA Steel kernel) ---
        # Computes sum_h relu((q*scale)_h @ k.T) * weight_h -> [B, 1, L, K] in a
        # single kernel. Falls back to the pure-MLX path below on any mismatch,
        # yielding identical scores (same math, same order of operations).
        scores = _native_ops.fused_indexer_scores(
            q * self.softmax_scale, k, weights_raw
        )
        if scores is None:
            scores = (q @ k.swapaxes(-1, -2)) * self.softmax_scale
            scores = mx.maximum(scores, 0)
            weights = weights_raw.swapaxes(-1, -2)[..., None]  # [B, H, L, 1]
            scores = (scores * weights).sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        native_topk = _native_ops.dsa_topk_indices(scores, self.index_topk)
        if native_topk is not None:
            return native_topk
        return mx.argpartition(-scores, kth=self.index_topk - 1, axis=-1)[
            ..., : self.index_topk
        ]


def _rope_theta(config: ModelConfig):
    if getattr(config, "rope_theta", None) is not None:
        return config.rope_theta
    params = getattr(config, "rope_parameters", None) or {}
    return params.get("rope_theta", 10000.0)


def _replace_rope_modules(attn, config: ModelConfig) -> tuple[int, int]:
    patched_attention = 0
    patched_indexer = 0
    rope_scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None)
    if getattr(config, "rope_interleave", None) is not None and hasattr(attn, "rope"):
        attn.rope = initialize_rope(
            dims=config.qk_rope_head_dim,
            base=_rope_theta(config),
            traditional=bool(config.rope_interleave),
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=rope_scaling,
        )
        patched_attention = 1
    indexer = getattr(attn, "indexer", None)
    if (
        getattr(config, "indexer_rope_interleave", None) is not None
        and indexer is not None
        and hasattr(indexer, "rope")
    ):
        indexer.rope = initialize_rope(
            dims=config.qk_rope_head_dim,
            base=_rope_theta(config),
            traditional=bool(config.indexer_rope_interleave),
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=rope_scaling,
        )
        patched_indexer = 1
    return patched_attention, patched_indexer


def _glm_attention_forward(attn, x, mask=None, cache=None, prev_topk_indices=None):
    B, L, _ = x.shape

    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    q = attn.q_b_proj(qr)

    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)
    compressed_kv = attn.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(compressed_kv, [attn.kv_lora_rank], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent = attn.kv_a_layernorm(compressed_kv)

    offset = cache[0].offset if cache is not None else 0
    q_pe = attn.rope(q_pe, offset)
    k_pe = attn.rope(k_pe, offset)

    kv_latent = mx.expand_dims(kv_latent, axis=1)

    if cache is not None:
        kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
    else:
        cache = [None] * 2

    if getattr(attn, "skip_topk", False):
        topk_indices = prev_topk_indices
        if getattr(attn, "indexer", None) is not None:
            attn.indexer(x, qr, mask, cache=cache[1])
    else:
        topk_indices = attn.indexer(x, qr, mask, cache=cache[1]) if getattr(attn, "indexer", None) is not None else None
    if topk_indices is not None:
        if L == 1:
            idx = topk_indices[:, :, 0, :, None]
            kv_latent = mx.take_along_axis(
                kv_latent,
                mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                axis=2,
            )
            k_pe = mx.take_along_axis(
                k_pe,
                mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                axis=2,
            )
            if mask is not None:
                mask = mx.take_along_axis(mask, topk_indices, axis=-1)
        else:
            shape = list(topk_indices.shape)
            shape[-1] = kv_latent.shape[2]
            sparse_mask = mx.zeros(shape, dtype=mx.bool_)
            sparse_mask = mx.put_along_axis(
                sparse_mask, topk_indices, mx.array(True), axis=-1
            )
            if mask is not None:
                sparse_mask = sparse_mask & mask
            mask = sparse_mask

    try:
        indexer_cache = cache[1] if cache is not None else None
    except Exception:
        indexer_cache = None
    if (
        cache is not None
        and cache[0] is not None
        and indexer_cache is not None
        and getattr(indexer_cache, "keys", None) is not None
    ):
        cache[0].keys = mx.depends(cache[0].keys, (indexer_cache.keys, indexer_cache.values))

    pe_scores = (q_pe * attn.scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        pe_scores = mx.where(
            mask,
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )

    if L == 1:
        q_nope = attn.embed_q(q_nope)
        k = v = kv_latent
    else:
        k = attn.embed_q(kv_latent, transpose=False)
        v = attn.unembed_out(kv_latent)

    output = scaled_dot_product_attention(
        q_nope, k, v, cache=cache, scale=attn.scale, mask=pe_scores
    )
    if L == 1:
        output = attn.unembed_out(output)

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output), topk_indices


def _glm_decoder_layer_forward(layer, x, mask=None, cache=None, prev_topk_indices=None):
    r, topk_indices = _glm_attention_forward(
        layer.self_attn,
        layer.input_layernorm(x),
        mask,
        cache,
        prev_topk_indices=prev_topk_indices,
    )
    h = x + r
    r = layer.mlp(layer.post_attention_layernorm(h))
    return h + r, topk_indices


def _patch_dsa_debug_switches(language_model, config: ModelConfig) -> None:
    """Runtime A/B switches for GLM-5.2 attention/indexer debugging.

    These are deliberate ifdefs, not quality workarounds:
      MLX_VLM_GLM_DSA_DISABLE=1          -> dense MLA attention; no sparse top-k mask.
      MLX_GLM_DSA_SPARSE=0               -> same dense MLA fallback, using the SSOT sparse-toggle name.
      MLX_VLM_GLM_DSA_DISABLE_SHARED=1   -> do not reuse shared top-k layers when an upstream
                                            implementation exposes skip_topk/indexer_types.
      MLX_GLM_DSA_SHARE_TOPK=0            -> same no-shared-topk fallback, using the SSOT toggle name.
    """
    disable_dsa = (
        _env_flag("MLX_VLM_GLM_DSA_DISABLE")
        or _env_off("MLX_GLM_DSA_SPARSE")
    )
    disable_shared = (
        _env_flag("MLX_VLM_GLM_DSA_DISABLE_SHARED")
        or _env_off("MLX_GLM_DSA_SHARE_TOPK")
    )
    layers = getattr(getattr(language_model, "model", None), "layers", []) or []
    patched_dsa = 0
    patched_shared = 0
    patched_rope = 0
    patched_indexer_rope = 0
    indexer_types = getattr(config, "indexer_types", None) or []
    shared_topk_supported = bool(indexer_types)
    enable_shared = shared_topk_supported and not disable_dsa and not disable_shared
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "indexer", None) is not None:
            attn.indexer = _GlmIndexer(config)
        rope_count, indexer_rope_count = _replace_rope_modules(attn, config)
        patched_rope += rope_count
        patched_indexer_rope += indexer_rope_count

        skip_topk = (
            enable_shared
            and idx < len(indexer_types)
            and str(indexer_types[idx]).lower() == "shared"
        )
        attn.skip_topk = bool(skip_topk)
        if skip_topk:
            patched_shared += 1
        if disable_dsa and hasattr(attn, "indexer"):
            attn.indexer = _NoOpIndexer(attn.indexer)
            patched_dsa += 1
    runtime = {
        "available": True,
        "sparse_enabled": not disable_dsa,
        "share_topk_enabled": enable_shared,
        "disable_dsa": disable_dsa,
        "disable_shared": disable_shared,
        "layers": len(layers),
        "patched_dsa": patched_dsa,
        "patched_shared": patched_shared,
        "patched_rope": patched_rope,
        "patched_indexer_rope": patched_indexer_rope,
        "indexer_types": bool(getattr(config, "indexer_types", None)),
        "shared_topk_supported": shared_topk_supported,
        "rope_interleave": getattr(config, "rope_interleave", None),
        "indexer_rope_interleave": getattr(config, "indexer_rope_interleave", None),
        "recent_topk": _env_flag("MLX_GLM_DSA_RECENT_TOPK"),
        "decode_only": _env_flag("MLX_GLM_DSA_DECODE_ONLY"),
    }
    try:
        setattr(config, "_glm_dsa_runtime", runtime)
        setattr(language_model, "_glm_dsa_runtime", runtime)
        setattr(language_model, "_glm_dsa_use_helper", True)
    except Exception:
        pass
    print(
        "[glm_moe_dsa] debug switches:"
        f" disable_dsa={disable_dsa} patched_dsa={patched_dsa};"
        f" disable_shared={disable_shared} patched_shared={patched_shared};"
        f" patched_rope={patched_rope} patched_indexer_rope={patched_indexer_rope};"
        f" indexer_types={bool(getattr(config, 'indexer_types', None))}",
        flush=True,
    )


class _CurrentServeModel(_CurrentLMModel):
    """Adapt current mlx-lm GLM to the mlx_vlm.server contract without vendoring it.

    GLM-5.2 mxfp4 should keep the current mlx-lm implementation because that path carries the newer DSA /
    shared-indexer fixes. The server still calls every bridged text model with ``inputs_embeds=...`` and
    expects ``out.logits``; mirror the GLM-5.1 bridge by feeding those precomputed embeddings into the
    current transformer forward instead of re-embedding chunked token ids.
    """

    def _model_forward(
        self,
        inputs=None,
        cache=None,
        inputs_embeds=None,
        return_topk: bool = False,
    ):
        model = self.model
        h = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        pipeline_rank = model.pipeline_rank
        pipeline_size = model.pipeline_size

        if cache is None:
            cache = [None] * model.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        topk_indices = None
        for i in range(model.num_layers):
            layer = model.layers[model.start_idx + i]
            if getattr(self, "_glm_dsa_use_helper", False):
                h, topk_indices = _glm_decoder_layer_forward(
                    layer,
                    h,
                    mask,
                    cache[i],
                    prev_topk_indices=topk_indices,
                )
            else:
                h = layer(h, mask, cache[i])

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        hidden = model.norm(h)
        if return_topk:
            return hidden, topk_indices
        return hidden

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        skip_logits = kwargs.pop("skip_logits", False)
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        if return_shared_kv:
            hidden, _ = self._model_forward(
                inputs,
                cache,
                inputs_embeds=inputs_embeds,
                return_topk=True,
            )
        else:
            hidden = self._model_forward(inputs, cache, inputs_embeds=inputs_embeds)
        logits = None if skip_logits else self.lm_head(hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden] if return_hidden else None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def _speculative_verify(self, inputs: mx.array, cache, sampler=None):
        sample_logits = sampler is not None
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=not sample_logits,
        )
        hidden = out.hidden_states[-1]
        shared_kv_states = out.shared_kv_states or {}
        if not sample_logits:
            return hidden, shared_kv_states, None
        return hidden, shared_kv_states, None, sampler(out.logits)

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        return self._speculative_verify(inputs, cache, sampler)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        return self._speculative_verify(inputs, cache)

    def rollback_speculative_cache(
        self,
        caches,
        gdn_states: Any,
        accepted,
        block_size: int,
    ) -> int:
        del gdn_states
        accepted_list = _accepted_to_list(accepted)
        max_a = max(accepted_list) if accepted_list else 0
        trim = int(block_size) - (max_a + 1)
        if trim > 0:
            for cache in _iter_leaf_caches(caches):
                if hasattr(cache, "trim"):
                    cache.trim(trim)
        return max_a


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if _use_legacy_indexer(getattr(config, "mlx_sanitized", None)):
            # vendored mlx-lm 0.30.7 deepseek_v32 (partial-rope indexer) for 0.30.x-sanitized checkpoints
            from ._arch_v32_0307 import Model as _LMModel
        else:
            _LMModel = _CurrentServeModel
        # Build the arch (reuses deepseek_v32). from_dict filters to the ModelArgs fields.
        self.language_model = _LMModel(_lm_args_from_config(config))
        _patch_dsa_debug_switches(self.language_model, config)

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
        out = self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=kwargs.pop("inputs_embeds", None),
            **kwargs,
        )
        # legacy vendored arch now returns a LanguageModelOutput; the current mlx-lm arch returns bare logits.
        logits = out.logits if hasattr(out, "logits") else out
        return LanguageModelOutput(
            logits=logits,
            hidden_states=getattr(out, "hidden_states", None),
            shared_kv_states=getattr(out, "shared_kv_states", None),
            gdn_states=getattr(out, "gdn_states", None),
        )

    def sanitize(self, weights):
        # The inferencerlabs "-INF" checkpoints arrive ALREADY in mlx post-sanitize layout: experts
        # pre-stacked into `switch_mlp`, `unembed_out` pre-built, and NO per-expert (`mlp.experts.N`) or
        # `kv_b_proj` tensors. mlx-lm's deepseek_v32 `sanitize` is built to transform a RAW HF checkpoint
        # (stack experts, build unembed_out by splitting kv_b) — re-running it on already-processed weights
        # corrupts them (keys still match, so it loads, but values are wrong -> garbage generation). So only
        # run mlx-lm's sanitize on a genuinely RAW checkpoint; skip it when the post-sanitize keys are present.
        already_sanitized = any(
            (".switch_mlp." in k) or (".unembed_out." in k) or (".embed_q." in k) for k in weights
        )
        if not already_sanitized:
            # mlx-lm's sanitize expects the bare `model.*` / `lm_head.*` namespace.
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
