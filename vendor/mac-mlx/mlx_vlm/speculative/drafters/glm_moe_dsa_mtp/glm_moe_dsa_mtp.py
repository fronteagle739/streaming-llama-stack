from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer

from ....models.base import create_attention_mask
from ....models.cache import BatchKVCache, CacheList, KVCache
from ....models.glm_moe_dsa.glm_moe_dsa import (
    _NoOpIndexer,
    _glm_decoder_layer_forward,
    _lm_args_from_config,
    _patch_dsa_debug_switches,
)
from .config import GlmMoeDsaMTPConfig


class _SharedHead(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)


class GlmMoeDsaMTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: GlmMoeDsaMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("GlmMoeDsaMTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        self.source_layer = int(config.source_layer or text_config.num_hidden_layers)
        self.recurrent_state = str(
            getattr(config, "recurrent_state", "raw") or "raw"
        ).lower().replace("-", "_")
        self.compute_dtype_name, self._compute_dtype = self._resolve_compute_dtype(
            getattr(config, "compute_dtype", "float16")
        )
        self.share_topk_source = self._resolve_share_topk_source(
            getattr(config, "share_topk_source", "internal")
        )
        self.dsa_mode = self._resolve_dsa_mode(getattr(config, "dsa_mode", "sparse"))
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.decoder = DeepseekV32DecoderLayer(
            _lm_args_from_config(text_config),
            layer_idx=self.source_layer,
        )
        self.shared_head = _SharedHead(hidden_size, text_config.rms_norm_eps)

        shim = SimpleNamespace(model=SimpleNamespace(layers=[self.decoder]))
        _patch_dsa_debug_switches(shim, text_config)
        object.__setattr__(
            self,
            "_decoder_sparse_indexer",
            getattr(self.decoder.self_attn, "indexer", None),
        )
        self._decoder_sparse_skip_topk = bool(
            getattr(self.decoder.self_attn, "skip_topk", False)
        )
        self._share_topk_enabled = self._compute_share_topk_enabled()
        self._apply_dsa_mode()

        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None
        self._cache: List[CacheList] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._seed_logits: Optional[mx.array] = None
        self._next_position: Any = 0
        self._round_appended = 0
        self._round_prefix_appended = 0
        self._kv_valid_len: Any = 0
        self._position: Any = 0
        self._draft_round = 0
        self._shared_topk = None

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def _recurrent_from_raw_mtp_hidden(self, hidden: mx.array) -> mx.array:
        hidden = self._cast_compute(hidden)
        if self.recurrent_state == "raw":
            return hidden
        return self.shared_head.norm(hidden)

    def _logits_from_mtp_hidden(self, hidden: mx.array) -> mx.array:
        sidecar_head = getattr(self.shared_head, "head", None)
        if self._lm_head_fn is None and sidecar_head is None:
            raise RuntimeError("bind(target_model) must be called before sampling.")
        hidden = self._cast_compute(hidden)
        if self.recurrent_state == "raw":
            hidden = self.shared_head.norm(hidden)
        if sidecar_head is not None:
            return sidecar_head(hidden)
        return self._lm_head_fn(hidden)

    @staticmethod
    def _resolve_compute_dtype(compute_dtype: str):
        normalized = str(compute_dtype or "float16").strip().lower()
        normalized = normalized.replace("-", "_")
        if normalized in ("", "native", "auto", "none"):
            return "native", None
        if normalized in ("fp16", "float16", "f16"):
            return "float16", mx.float16
        if normalized in ("bf16", "bfloat16"):
            return "bfloat16", mx.bfloat16
        raise ValueError(
            f"unsupported GLM MTP compute dtype {compute_dtype!r}; "
            "expected native, float16, or bfloat16"
        )

    @staticmethod
    def _resolve_share_topk_source(share_topk_source: str) -> str:
        normalized = str(share_topk_source or "internal").strip().lower()
        normalized = normalized.replace("-", "_")
        if normalized in ("", "default", "internal", "mtp", "mtp_internal"):
            return "internal"
        if normalized in ("target", "target_prev", "target_previous"):
            return "target"
        if normalized in ("off", "none", "disabled", "false", "0"):
            return "off"
        raise ValueError(
            f"unsupported GLM MTP share-topk source {share_topk_source!r}; "
            "expected target, internal, or off"
        )

    @staticmethod
    def _resolve_dsa_mode(dsa_mode: str) -> str:
        normalized = str(dsa_mode or "sparse").strip().lower()
        normalized = normalized.replace("-", "_")
        aliases = {
            "default": "sparse",
            "enabled": "sparse",
            "true": "sparse",
            "1": "sparse",
            "off": "dense",
            "disabled": "dense",
            "false": "dense",
            "0": "dense",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in ("sparse", "dense"):
            return normalized
        raise ValueError(
            f"unsupported GLM MTP DSA mode {dsa_mode!r}; expected sparse or dense"
        )

    def _compute_share_topk_enabled(self) -> bool:
        return (
            getattr(self, "dsa_mode", "sparse") != "dense"
            and self.share_topk_source != "off"
            and bool(getattr(self.args, "index_share_for_mtp_iteration", False))
        )

    def _apply_dsa_mode(self) -> None:
        attn = self.decoder.self_attn
        if getattr(self, "dsa_mode", "sparse") == "dense":
            if self._decoder_sparse_indexer is not None:
                attn.indexer = _NoOpIndexer(self._decoder_sparse_indexer)
            setattr(attn, "skip_topk", False)
            self._shared_topk = None
        else:
            if self._decoder_sparse_indexer is not None:
                attn.indexer = self._decoder_sparse_indexer
            setattr(attn, "skip_topk", bool(self._decoder_sparse_skip_topk))
        self._share_topk_enabled = self._compute_share_topk_enabled()

    def set_dsa_mode(self, dsa_mode: str) -> "GlmMoeDsaMTPDraftModel":
        self.dsa_mode = self._resolve_dsa_mode(dsa_mode)
        self._apply_dsa_mode()
        if hasattr(self, "config") and hasattr(self.config, "dsa_mode"):
            self.config.dsa_mode = self.dsa_mode
        return self

    def set_compute_dtype(self, compute_dtype: str) -> "GlmMoeDsaMTPDraftModel":
        self.compute_dtype_name, self._compute_dtype = self._resolve_compute_dtype(
            compute_dtype
        )
        if hasattr(self, "config") and hasattr(self.config, "compute_dtype"):
            self.config.compute_dtype = self.compute_dtype_name
        return self

    def set_share_topk_source(
        self, share_topk_source: str
    ) -> "GlmMoeDsaMTPDraftModel":
        self.share_topk_source = self._resolve_share_topk_source(share_topk_source)
        self._share_topk_enabled = self._compute_share_topk_enabled()
        if not self._share_topk_enabled:
            self._shared_topk = None
        if hasattr(self, "config") and hasattr(self.config, "share_topk_source"):
            self.config.share_topk_source = self.share_topk_source
        return self

    def _cast_compute(self, value):
        dtype = getattr(self, "_compute_dtype", None)
        if dtype is None or value is None:
            return value
        try:
            if getattr(value, "dtype", None) in (mx.float16, mx.bfloat16, mx.float32):
                return value.astype(dtype)
        except Exception:
            pass
        return value

    def bind(self, target_model) -> "GlmMoeDsaMTPDraftModel":
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
        self._input_embed_scale = float(getattr(inner, "embed_scale", 1.0))

        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[CacheList]:
        if left_padding is not None:
            return [CacheList(BatchKVCache(left_padding), BatchKVCache(left_padding))]
        return [CacheList(KVCache(), KVCache())]

    def reset(
        self, target_model, left_padding: Optional[List[int]] = None
    ) -> List[CacheList]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._draft_round = 0
        self._cache = self.make_cache(left_padding)
        self._seed_token = None
        self._seed_hidden = None
        self._seed_logits = None
        self._next_position = 0
        self._round_appended = 0
        self._round_prefix_appended = 0
        self._shared_topk = None
        return self._cache

    def draft_eval_state(self):
        state = [self._seed_token, self._seed_hidden, self._seed_logits]
        for cache in self._cache:
            state.append(cache.state)
        if self._shared_topk is not None:
            state.append(self._shared_topk)
        return state

    @property
    def has_seed(self) -> bool:
        return self._seed_token is not None and self._seed_hidden is not None

    @staticmethod
    def _topk_tail(topk):
        if topk is None:
            return None
        try:
            shape = tuple(int(dim) for dim in topk.shape)
        except Exception:
            return topk
        if len(shape) >= 4 and shape[2] > 1:
            return topk[:, :, -1:, :]
        return topk

    @staticmethod
    def _topk_matches_seq_len(topk, seq_len: int) -> bool:
        if topk is None:
            return False
        try:
            return int(topk.shape[2]) == int(seq_len)
        except Exception:
            return False

    def _reusable_shared_topk(self, seq_len: int):
        if (
            not bool(getattr(self, "_share_topk_enabled", False))
            or self._shared_topk is None
        ):
            return None
        if self._topk_matches_seq_len(self._shared_topk, seq_len):
            return self._shared_topk
        tail = self._topk_tail(self._shared_topk)
        if self._topk_matches_seq_len(tail, seq_len):
            return tail
        return None

    @staticmethod
    def _shared_topk_from_state(shared_kv_states: dict):
        if not isinstance(shared_kv_states, dict):
            return None
        for key in ("topk_indices", "shared_topk", "glm_dsa_topk"):
            topk = shared_kv_states.get(key)
            if topk is not None:
                return topk
        return None

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ) -> None:
        del left_padding
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if position is None:
            position = kv_valid_len
        self._kv_valid_len = kv_valid_len
        self._position = position
        cache_empty = not self._cache or all(
            cache.empty() for cache in self._cache
        )
        if bool(getattr(self, "_share_topk_enabled", False)):
            shared_topk = self._shared_topk_from_state(shared_kv_states)
            if (
                shared_topk is not None
                and getattr(self, "share_topk_source", "internal") == "target"
                and not self.has_seed
            ):
                self._shared_topk = self._topk_tail(shared_topk)
            elif cache_empty or not self.has_seed:
                self._shared_topk = None
        if cache_empty:
            self._next_position = kv_valid_len

    def _layer_cache(self, cache):
        if cache is None:
            return None
        if isinstance(cache, (list, tuple)):
            return cache[0] if cache else None
        return cache

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        cache: Optional[List[CacheList]],
        *,
        zero_first_token: bool = False,
    ) -> mx.array:
        token_embed = self._cast_compute(token_embed)
        hidden = self._cast_compute(hidden)
        if zero_first_token and token_embed.shape[1] > 0:
            token_embed = mx.concatenate(
                [mx.zeros_like(token_embed[:, :1, :]), token_embed[:, 1:, :]],
                axis=1,
            )
        h = mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        h = self.eh_proj(h)
        layer_cache = self._layer_cache(cache)
        attn_cache = layer_cache[0] if layer_cache is not None else None
        mask = create_attention_mask(h, attn_cache, return_array=True)

        attn = self.decoder.self_attn
        old_skip_topk = bool(getattr(attn, "skip_topk", False))
        prev_topk = self._reusable_shared_topk(int(token_embed.shape[1]))
        attn.skip_topk = bool(prev_topk is not None)
        try:
            h, topk = _glm_decoder_layer_forward(
                self.decoder,
                h,
                mask,
                layer_cache,
                prev_topk_indices=prev_topk,
            )
        finally:
            attn.skip_topk = old_skip_topk

        if self._share_topk_enabled:
            self._shared_topk = topk if topk is not None else prev_topk
        else:
            self._shared_topk = None
        return self._recurrent_from_raw_mtp_hidden(h)

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
        *,
        zero_first_token: bool = False,
    ) -> mx.array:
        token_embed = (
            self._input_embed(tokens.astype(token_dtype)) * self._input_embed_scale
        )
        h = self._forward_hidden(
            token_embed,
            hidden[:, : tokens.shape[1], :],
            self._cache,
            zero_first_token=zero_first_token,
        )
        steps = int(tokens.shape[1])
        self._next_position = (
            self._next_position + steps
            if isinstance(self._next_position, int)
            else self._next_position + steps
        )
        return h

    def _forward_token(
        self,
        tok: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
        *,
        zero_first_token: bool = False,
    ) -> mx.array:
        return self._forward_tokens(
            tok,
            hidden,
            token_dtype,
            zero_first_token=zero_first_token,
        )

    @staticmethod
    def _previous_hidden_for_positions(hidden: mx.array) -> mx.array:
        if hidden.shape[1] == 0:
            return hidden
        zero = mx.zeros_like(hidden[:, :1, :])
        if hidden.shape[1] == 1:
            return zero
        return mx.concatenate([zero, hidden[:, :-1, :]], axis=1)

    def _set_seed_from_hidden(self, hidden: mx.array, sampler, greedy: bool) -> None:
        hidden = self._cast_compute(hidden)
        logits = self._logits_from_mtp_hidden(hidden)
        self._seed_token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        self._seed_hidden = hidden
        self._seed_logits = logits

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        shared_kv_states: Optional[dict] = None,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)

        tokens = mx.concatenate([input_ids.astype(token_dtype), bonus], axis=1)
        previous_hidden = mx.concatenate(
            [
                self._previous_hidden_for_positions(hidden[:, : input_ids.shape[1], :]),
                hidden[:, input_ids.shape[1] - 1 : input_ids.shape[1], :],
            ],
            axis=1,
        )
        if bool(getattr(self, "_share_topk_enabled", False)):
            shared_topk = self._shared_topk_from_state(shared_kv_states)
            self._shared_topk = (
                shared_topk
                if getattr(self, "share_topk_source", "internal") == "target"
                else None
            )
        self._next_position = 0
        h = self._forward_tokens(
            tokens,
            previous_hidden,
            token_dtype,
            zero_first_token=True,
        )
        self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        if self._share_topk_enabled:
            self._shared_topk = self._topk_tail(self._shared_topk)
        else:
            self._shared_topk = None

    def prefill_committed_prompt_history(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype = mx.int32,
        shared_kv_states: Optional[dict] = None,
    ) -> None:
        if input_ids.shape[1] == 0:
            return

        if bool(getattr(self, "_share_topk_enabled", False)):
            shared_topk = self._shared_topk_from_state(shared_kv_states)
            self._shared_topk = (
                shared_topk
                if getattr(self, "share_topk_source", "internal") == "target"
                else None
            )

        self._next_position = 0
        warmed = self._forward_tokens(
            input_ids.astype(token_dtype),
            self._previous_hidden_for_positions(hidden[:, : input_ids.shape[1], :]),
            token_dtype,
            zero_first_token=True,
        )
        self._seed_token = None
        self._seed_hidden = None
        self._seed_logits = None
        self._round_appended = 0
        self._round_prefix_appended = 0
        self._shared_topk = None
        mx.eval(warmed)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        prefix_appended = int(getattr(self, "_round_prefix_appended", 0) or 0)
        draft_appended = int(getattr(self, "_round_appended", 0) or 0)
        keep_appended = min(int(accepted), draft_appended)
        trim = (prefix_appended + draft_appended) - (
            prefix_appended + keep_appended
        )
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )
            self._shared_topk = None

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])

        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        elif self._seed_token is None or self._seed_hidden is None:
            self._shared_topk = None
        self._round_appended = 0
        self._round_prefix_appended = 0

    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: List[int],
        new_tokens: List[List[int]],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if len(accepted) <= 1:
            self.accept_verified_tokens(
                verify_hidden,
                draft_tokens,
                int(accepted[0]),
                new_tokens[0],
                sampler,
                token_dtype,
                greedy,
            )
            return

        accepted_set = {int(a) for a in accepted}
        if len(accepted_set) != 1:
            raise ValueError(
                "GLM MTP batched cache update requires uniform acceptance."
            )
        accepted_i = accepted_set.pop()

        prefix_appended = int(getattr(self, "_round_prefix_appended", 0) or 0)
        draft_appended = int(getattr(self, "_round_appended", 0) or 0)
        keep_appended = min(accepted_i, draft_appended)
        trim = (prefix_appended + draft_appended) - (
            prefix_appended + keep_appended
        )
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )
            self._shared_topk = None

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, accepted_i):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])

        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )
            token_chunks.append(bonus)
            hidden_chunks.append(verify_hidden[:, accepted_i : accepted_i + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        elif self._seed_token is None or self._seed_hidden is None:
            self._shared_topk = None
        self._round_appended = 0
        self._round_prefix_appended = 0

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)

        for cache in self._cache:
            cache_filter = getattr(cache, "filter", None)
            if callable(cache_filter):
                cache_filter(keep)
                continue
            for item in getattr(cache, "caches", (cache,)):
                if item.keys is not None:
                    item.keys = item.keys[keep]
                    item.values = item.values[keep]

        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]
        if self._seed_logits is not None:
            self._seed_logits = self._seed_logits[keep]

        for attr in ("_next_position", "_kv_valid_len", "_position"):
            value = getattr(self, attr)
            if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
                setattr(self, attr, value[keep])
        self._shared_topk = None

    def _draft_block_impl(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        *,
        return_logits: bool = False,
    ):
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target embeddings and LM head."
            )

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus[:, None].astype(token_dtype)

        h_prev = hidden
        tokens: List[mx.array] = []
        logits_rows: List[mx.array] = []
        self._round_appended = 0
        self._round_prefix_appended = 0
        used_seed = self._seed_token is not None and self._seed_hidden is not None
        if (
            bool(getattr(self, "_share_topk_enabled", False))
            and not used_seed
            and getattr(self, "share_topk_source", "internal") != "target"
        ):
            self._shared_topk = None
        zero_first_decode = bool(
            getattr(self, "debug_zero_first_decode_token", False)
        )

        if self._seed_token is not None and self._seed_hidden is not None:
            tok = self._seed_token.astype(token_dtype)
            h_prev = self._seed_hidden
            tokens.append(tok)
            logits_rows.append(
                self._seed_logits
                if self._seed_logits is not None
                else self._logits_from_mtp_hidden(h_prev)
            )
            self._seed_token = None
            self._seed_hidden = None
            self._seed_logits = None

        while len(tokens) < block_size - 1:
            is_prefix_row = (
                not used_seed
                and not tokens
                and self._round_prefix_appended == 0
                and self._round_appended == 0
            )
            h_prev = self._forward_token(
                tok,
                h_prev,
                token_dtype,
                zero_first_token=zero_first_decode and is_prefix_row,
            )
            if is_prefix_row:
                self._round_prefix_appended = 1
            else:
                self._round_appended += 1
            logits = self._logits_from_mtp_hidden(h_prev)
            logits_rows.append(logits)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(tok)

        self._draft_round += 1
        draft_tokens = mx.concatenate(tokens, axis=1)
        if not return_logits:
            return draft_tokens
        if logits_rows:
            draft_logits = mx.concatenate(logits_rows, axis=1)
        else:
            draft_logits = mx.zeros((draft_tokens.shape[0], 0, 0))
        return draft_tokens, draft_logits

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        return self._draft_block_impl(
            last_bonus,
            hidden,
            cache,
            block_size,
            sampler,
            token_dtype,
            greedy,
            return_logits=False,
        )

    def draft_block_with_logits(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        return self._draft_block_impl(
            last_bonus,
            hidden,
            cache,
            block_size,
            sampler,
            token_dtype,
            greedy,
            return_logits=True,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = GlmMoeDsaMTPDraftModel._remap_raw_weights(self, dict(weights))
        ensure_sidecar_head = getattr(self, "_ensure_sidecar_head", None)
        if callable(ensure_sidecar_head):
            ensure_sidecar_head(weights)
        weights = GlmMoeDsaMTPDraftModel._dequant_fp8(weights)
        weights = GlmMoeDsaMTPDraftModel._stack_experts(self, weights)
        weights = GlmMoeDsaMTPDraftModel._split_kv_b_proj(self, weights)
        weights = GlmMoeDsaMTPDraftModel._cast_bf16_to_fp16(weights)
        return weights

    def _ensure_sidecar_head(self, weights: Dict[str, mx.array]) -> None:
        if "shared_head.head.weight" not in weights:
            return
        if hasattr(self.shared_head, "head"):
            return
        weight = weights["shared_head.head.weight"]
        out_dim = int(weight.shape[0])
        in_dim = int(getattr(self.args, "hidden_size", 0) or weight.shape[-1])
        self.shared_head.head = nn.Linear(in_dim, out_dim, bias=False)

    def _remap_raw_weights(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        source_prefix = f"model.layers.{self.source_layer}."
        remapped = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            if key.startswith(source_prefix):
                key = key[len(source_prefix) :]
            elif key.startswith("mtp_block."):
                key = "decoder." + key[len("mtp_block.") :]
            elif key.startswith("shared_head_norm."):
                key = "shared_head.norm." + key[len("shared_head_norm.") :]
            elif key.startswith("model.decoder."):
                key = key[len("model.") :]
            elif key.startswith("model."):
                key = key[len("model.") :]

            if key.startswith("embed_tokens."):
                continue
            if key.startswith(("eh_proj.", "enorm.", "hnorm.", "shared_head.")):
                out_key = key
            elif key.startswith("decoder."):
                out_key = key
            else:
                out_key = f"decoder.{key}"
            remapped[out_key] = value
        return remapped

    @staticmethod
    def _dequant_fp8(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        def dequant(weight, scale_inv):
            dtype = mx.float16
            weight = mx.from_fp8(weight, dtype=mx.float16)
            block_size = 128
            m, n = weight.shape
            pad_bottom = (-m) % block_size
            pad_side = (-n) % block_size
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                (
                    (m + pad_bottom) // block_size,
                    block_size,
                    (n + pad_side) // block_size,
                    block_size,
                )
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        new_weights = {}
        for key, value in weights.items():
            if "weight_scale_inv" in key:
                scale_inv = value
                weight_key = key.replace("_scale_inv", "")
                if weight_key in weights:
                    new_weights[weight_key] = dequant(weights[weight_key], scale_inv)
            elif key not in new_weights:
                new_weights[key] = value
        return new_weights

    @staticmethod
    def _cast_bf16_to_fp16(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return {
            key: value.astype(mx.float16)
            if getattr(value, "dtype", None) == mx.bfloat16
            else value
            for key, value in weights.items()
        }

    def _stack_experts(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefix = "decoder.mlp.experts"
        for src, dst in (
            ("gate_proj", "gate_proj"),
            ("down_proj", "down_proj"),
            ("up_proj", "up_proj"),
        ):
            for suffix in ("weight", "scales", "biases"):
                key0 = f"{prefix}.0.{src}.{suffix}"
                if key0 not in weights:
                    continue
                stacked = [
                    weights.pop(f"{prefix}.{expert}.{src}.{suffix}")
                    for expert in range(self.args.n_routed_experts)
                ]
                weights[f"decoder.mlp.switch_mlp.{dst}.{suffix}"] = mx.stack(stacked)
        return weights

    def _split_kv_b_proj(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefix = "decoder.self_attn"
        weight_key = f"{prefix}.kv_b_proj.weight"
        if weight_key not in weights:
            return weights

        quantized = f"{prefix}.kv_b_proj.scales" in weights
        value = weights.pop(weight_key)
        head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim

        if quantized:
            dims = self.args.kv_lora_rank
            scales = weights.pop(f"{prefix}.kv_b_proj.scales")
            biases = weights.pop(f"{prefix}.kv_b_proj.biases")
            bits = (value.shape[-1] * 32) // dims
            group_size = dims // scales.shape[-1]
            value = mx.dequantize(
                value,
                scales,
                biases,
                bits=bits,
                group_size=group_size,
            )

        num_heads = self.args.num_attention_heads
        value = value.reshape(num_heads, head_dim, -1)
        wk = mx.contiguous(
            value[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
        )
        wv = mx.contiguous(value[:, self.args.qk_nope_head_dim :, :])

        if quantized:
            wk, wk_scales, wk_biases = mx.quantize(
                wk, bits=bits, group_size=group_size
            )
            wv, wv_scales, wv_biases = mx.quantize(
                wv, bits=bits, group_size=group_size
            )
            weights[f"{prefix}.embed_q.scales"] = wk_scales
            weights[f"{prefix}.unembed_out.scales"] = wv_scales
            weights[f"{prefix}.embed_q.biases"] = wk_biases
            weights[f"{prefix}.unembed_out.biases"] = wv_biases

        weights[f"{prefix}.embed_q.weight"] = wk
        weights[f"{prefix}.unembed_out.weight"] = wv
        return weights
