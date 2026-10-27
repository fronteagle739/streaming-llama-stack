from __future__ import annotations
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from glm52_mtp import MtpSidecarInfo, mlx_runtime_direct_tensor_map, mlx_runtime_quantizable_bases

@dataclass(frozen=True)
class MlxMtpWeightPlan:
    direct: dict[str, str]
    kv_b_source: str
    expert_sources: dict[str, list[str]]
    expert_targets: dict[str, str]

def build_mlx_mtp_weight_plan(info: MtpSidecarInfo) -> MlxMtpWeightPlan:
    layer = info.source_layer
    p = f'model.layers.{layer}'
    direct = {f'{p}.eh_proj.weight': 'eh_proj.weight', f'{p}.enorm.weight': 'enorm.weight', f'{p}.hnorm.weight': 'hnorm.weight', f'{p}.shared_head.norm.weight': 'shared_head_norm.weight', f'{p}.input_layernorm.weight': 'mtp_block.input_layernorm.weight', f'{p}.post_attention_layernorm.weight': 'mtp_block.post_attention_layernorm.weight', f'{p}.self_attn.q_a_proj.weight': 'mtp_block.self_attn.q_a_proj.weight', f'{p}.self_attn.q_a_layernorm.weight': 'mtp_block.self_attn.q_a_layernorm.weight', f'{p}.self_attn.q_b_proj.weight': 'mtp_block.self_attn.q_b_proj.weight', f'{p}.self_attn.kv_a_proj_with_mqa.weight': 'mtp_block.self_attn.kv_a_proj_with_mqa.weight', f'{p}.self_attn.kv_a_layernorm.weight': 'mtp_block.self_attn.kv_a_layernorm.weight', f'{p}.self_attn.o_proj.weight': 'mtp_block.self_attn.o_proj.weight', f'{p}.self_attn.indexer.k_norm.bias': 'mtp_block.self_attn.indexer.k_norm.bias', f'{p}.self_attn.indexer.k_norm.weight': 'mtp_block.self_attn.indexer.k_norm.weight', f'{p}.self_attn.indexer.weights_proj.weight': 'mtp_block.self_attn.indexer.weights_proj.weight', f'{p}.self_attn.indexer.wk.weight': 'mtp_block.self_attn.indexer.wk.weight', f'{p}.self_attn.indexer.wq_b.weight': 'mtp_block.self_attn.indexer.wq_b.weight', f'{p}.mlp.gate.e_score_correction_bias': 'mtp_block.mlp.gate.e_score_correction_bias', f'{p}.mlp.gate.weight': 'mtp_block.mlp.gate.weight', f'{p}.mlp.shared_experts.down_proj.weight': 'mtp_block.mlp.shared_experts.down_proj.weight', f'{p}.mlp.shared_experts.gate_proj.weight': 'mtp_block.mlp.shared_experts.gate_proj.weight', f'{p}.mlp.shared_experts.up_proj.weight': 'mtp_block.mlp.shared_experts.up_proj.weight'}
    expert_sources = {'gate_proj': [f'{p}.mlp.experts.{idx}.gate_proj.weight' for idx in range(info.n_routed_experts)], 'up_proj': [f'{p}.mlp.experts.{idx}.up_proj.weight' for idx in range(info.n_routed_experts)], 'down_proj': [f'{p}.mlp.experts.{idx}.down_proj.weight' for idx in range(info.n_routed_experts)]}
    expert_targets = {'gate_proj': 'mtp_block.mlp.switch_mlp.gate_proj.weight', 'up_proj': 'mtp_block.mlp.switch_mlp.up_proj.weight', 'down_proj': 'mtp_block.mlp.switch_mlp.down_proj.weight'}
    return MlxMtpWeightPlan(direct=dict(sorted(direct.items())), kv_b_source=f'{p}.self_attn.kv_b_proj.weight', expert_sources=expert_sources, expert_targets=expert_targets)

def _require_attr(obj: Any, *paths: str) -> Any:
    for path in paths:
        cur = obj
        ok = True
        for part in path.split('.'):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok:
            return cur
    raise RuntimeError(f"could not resolve any of: {', '.join(paths)}")

def _resolve_embed_binding(lang: Any) -> tuple[Any, float]:
    owners = []
    model = getattr(lang, 'model', None)
    if model is not None:
        owners.append(model)
    owners.append(lang)
    for owner in owners:
        embed = getattr(owner, 'embed_tokens', None)
        if embed is None:
            continue
        return (embed, float(getattr(owner, 'embed_scale', 1.0)))
    raise RuntimeError('could not resolve target embed_tokens')

def _sidecar_text_config_namespace(info: MtpSidecarInfo) -> Any | None:
    try:
        with info.config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    text = config.get('text_config')
    if not isinstance(text, dict):
        text = config
    values = dict(text)
    for key in ('model_type', 'rope_parameters', 'quantization', 'quantization_config'):
        if key in config and key not in values:
            values[key] = config[key]
    return SimpleNamespace(**values)

def target_logits_and_hidden(lang, ids, cache):
    try:
        out = lang(ids, cache=cache, return_hidden=True)
        logits = out.logits if hasattr(out, 'logits') else None
        hidden_states = getattr(out, 'hidden_states', None)
        if hidden_states:
            return (logits, hidden_states[-1])
        if isinstance(out, tuple) and len(out) >= 2:
            return (out[0], out[1])
    except TypeError:
        pass
    except ValueError:
        pass
    model = getattr(lang, 'model', None)
    lm_head = getattr(lang, 'lm_head', None)
    if model is None or lm_head is None:
        out = lang(ids, cache=cache)
        logits = out.logits if hasattr(out, 'logits') else out
        return (logits, None)
    hidden = model(ids, cache=cache)
    return (lm_head(hidden), hidden)

def target_logits_hidden_topk(lang, ids, cache):
    model_forward = getattr(lang, '_model_forward', None)
    lm_head = getattr(lang, 'lm_head', None)
    if callable(model_forward) and lm_head is not None:
        try:
            result = model_forward(ids, cache=cache, return_topk=True)
        except TypeError:
            pass
        else:
            if isinstance(result, tuple) and len(result) >= 2:
                hidden, topk = (result[0], result[1])
                return (lm_head(hidden), hidden, topk)
    logits, hidden = target_logits_and_hidden(lang, ids, cache)
    return (logits, hidden, None)

class Glm52MtpDrafter:

    def __init__(self, target_model, sidecar_info: MtpSidecarInfo, sidecar_weights, compute_dtype: str='native', recurrent_state: str='shared_head_norm', history_materialization: str='eager', share_topk_source: str='internal', match_trunk_precision: str='auto'):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import CacheList, KVCache
        from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer
        try:
            from mlx_vlm.models.glm_moe_dsa.glm_moe_dsa import _glm_decoder_layer_forward, _patch_dsa_debug_switches
        except Exception:
            _glm_decoder_layer_forward = None
            _patch_dsa_debug_switches = None
        self.mx = mx
        self.nn = nn
        self._create_attention_mask = create_attention_mask
        self._glm_decoder_layer_forward = _glm_decoder_layer_forward
        self._cache_list_type = CacheList
        self._kv_cache_type = KVCache
        self.info = sidecar_info
        self.compute_dtype_name, self._compute_dtype = self._resolve_compute_dtype(mx, compute_dtype)
        self.recurrent_state = self._resolve_recurrent_state(recurrent_state)
        self.history_materialization = self._resolve_history_materialization(history_materialization)
        self.share_topk_source = self._resolve_share_topk_source(share_topk_source)
        self.match_trunk_precision = self._resolve_match_trunk_precision(match_trunk_precision)
        lang = getattr(target_model, 'language_model', target_model)
        self.lang = lang
        self.args = _require_attr(lang, 'args', 'model.args', 'config')
        self.dsa_config = _sidecar_text_config_namespace(sidecar_info) or getattr(target_model, 'config', None) or getattr(lang, 'config', None) or self.args
        self.embed_tokens, self.embed_scale = _resolve_embed_binding(lang)
        self.lm_head = _require_attr(lang, 'lm_head', 'model.lm_head')
        self.enorm = nn.RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.eh_proj = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size, bias=False)
        self.shared_head_norm = nn.RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.mtp_block = DeepseekV32DecoderLayer(self.args, int(sidecar_info.source_layer))
        if _patch_dsa_debug_switches is not None:
            shim = SimpleNamespace(model=SimpleNamespace(layers=[self.mtp_block]))
            _patch_dsa_debug_switches(shim, self.dsa_config)
        self._share_topk_enabled = self.share_topk_source != 'off' and bool(getattr(self.dsa_config, 'index_share_for_mtp_iteration', sidecar_info.index_share_for_mtp_iteration))
        self._load_sidecar_weights(sidecar_weights)
        self._mtp_cache = None
        self._round_appended = 0
        self._round_prefix_appended = 0
        self._seed_token = None
        self._seed_hidden = None
        self._shared_topk = None
        self.accepted = 0
        self.checked = 0

    @staticmethod
    def _resolve_compute_dtype(mx, compute_dtype: str):
        normalized = str(compute_dtype or 'native').strip().lower()
        if normalized in ('', 'native', 'auto', 'none'):
            return ('native', None)
        if normalized in ('fp16', 'float16', 'f16'):
            return ('float16', mx.float16)
        if normalized in ('bf16', 'bfloat16'):
            return ('bfloat16', mx.bfloat16)
        raise ValueError(f'unsupported MTP compute dtype {compute_dtype!r}; expected native, float16, or bfloat16')

    def _cast_weight(self, value):
        dtype = self._compute_dtype
        if dtype is None:
            return value
        try:
            if value.dtype in (self.mx.float16, self.mx.bfloat16, self.mx.float32):
                return value.astype(dtype)
        except Exception:
            pass
        return value

    @staticmethod
    def _resolve_recurrent_state(recurrent_state: str) -> str:
        normalized = str(recurrent_state or 'shared_head_norm').strip().lower()
        normalized = normalized.replace('-', '_')
        if normalized in ('', 'default', 'norm', 'normalized', 'shared_head_norm'):
            return 'shared_head_norm'
        if normalized in ('raw', 'pre_norm'):
            return 'raw'
        raise ValueError(f'unsupported MTP recurrent state {recurrent_state!r}; expected shared_head_norm or raw')

    @staticmethod
    def _resolve_history_materialization(history_materialization: str) -> str:
        normalized = str(history_materialization or 'eager').strip().lower()
        normalized = normalized.replace('-', '_')
        if normalized in ('', 'default', 'eager', 'eval'):
            return 'eager'
        if normalized in ('lazy', 'defer', 'deferred'):
            return 'lazy'
        raise ValueError(f'unsupported MTP history materialization {history_materialization!r}; expected eager or lazy')

    @staticmethod
    def _resolve_share_topk_source(share_topk_source: str) -> str:
        normalized = str(share_topk_source or 'internal').strip().lower()
        normalized = normalized.replace('-', '_')
        if normalized in ('', 'default', 'internal', 'mtp', 'mtp_internal'):
            return 'internal'
        if normalized in ('target', 'target_prev', 'target_previous'):
            return 'target'
        if normalized in ('off', 'none', 'disabled', 'false', '0'):
            return 'off'
        raise ValueError(f'unsupported MTP share-topk source {share_topk_source!r}; expected target, internal, or off')

    @staticmethod
    def _resolve_match_trunk_precision(match_trunk_precision: str) -> str:
        normalized = str(match_trunk_precision or 'auto').strip().lower()
        normalized = normalized.replace('-', '_')
        if normalized in ('', 'auto', 'default'):
            return 'auto'
        if normalized in ('on', 'force', 'yes', 'true', '1', 'match'):
            return 'on'
        if normalized in ('off', 'no', 'false', '0', 'none', 'disabled'):
            return 'off'
        raise ValueError(f'unsupported MTP match-trunk-precision {match_trunk_precision!r}; expected auto, on, or off')

    @staticmethod
    def _topk_tail(topk):
        if topk is None:
            return None
        try:
            shape = tuple((int(dim) for dim in topk.shape))
        except Exception:
            return topk
        if len(shape) >= 4 and shape[2] > 1:
            return topk[:, :, -1:, :]
        return topk

    @staticmethod
    def _topk_matches_input(topk, input_ids) -> bool:
        if topk is None:
            return False
        try:
            return int(topk.shape[2]) == int(input_ids.shape[1])
        except Exception:
            return False

    def shared_topk_snapshot(self):
        return self._shared_topk

    def restore_shared_topk(self, snapshot) -> None:
        self._shared_topk = snapshot if self._share_topk_enabled else None

    def seed_shared_topk(self, topk, *, tail: bool=True) -> bool:
        if not bool(getattr(self, '_share_topk_enabled', False)) or getattr(self, 'share_topk_source', 'target') != 'target' or topk is None:
            return False
        self._shared_topk = self._topk_tail(topk) if tail else topk
        return self._shared_topk is not None

    def reset_iteration_topk(self) -> None:
        self._shared_topk = None

    def _reusable_shared_topk(self, input_ids):
        if not bool(getattr(self, '_share_topk_enabled', False)) or self._shared_topk is None:
            return None
        if self._topk_matches_input(self._shared_topk, input_ids):
            return self._shared_topk
        tail = self._topk_tail(self._shared_topk)
        if self._topk_matches_input(tail, input_ids):
            return tail
        return None

    def _load_sidecar_weights(self, sidecar_weights) -> None:
        if self.info.tensor_name_format == 'mlx_mtp_runtime':
            self._load_mlx_runtime_weights(sidecar_weights)
            self._maybe_requantize_to_trunk(sidecar_weights)
            return
        mx = self.mx
        plan = build_mlx_mtp_weight_plan(self.info)
        weights = []
        missing = [name for name in plan.direct if name not in sidecar_weights]
        if missing:
            raise RuntimeError(f'MTP sidecar missing direct weights: {missing[:3]}')
        for src, dst in plan.direct.items():
            weights.append((dst, self._cast_weight(sidecar_weights[src])))
        kv_b = self._cast_weight(sidecar_weights[plan.kv_b_source])
        head_dim = int(self.args.qk_nope_head_dim) + int(self.args.v_head_dim)
        n_heads = int(self.args.num_attention_heads)
        kv_b = kv_b.reshape(n_heads, head_dim, -1)
        wk = mx.contiguous(kv_b[:, :int(self.args.qk_nope_head_dim), :].swapaxes(-1, -2))
        wv = mx.contiguous(kv_b[:, int(self.args.qk_nope_head_dim):, :])
        weights.append(('mtp_block.self_attn.embed_q.weight', wk))
        weights.append(('mtp_block.self_attn.unembed_out.weight', wv))
        for proj_name, srcs in plan.expert_sources.items():
            missing = [name for name in srcs if name not in sidecar_weights]
            if missing:
                raise RuntimeError(f'MTP sidecar missing {proj_name} experts: {missing[:3]}')
            weights.append((plan.expert_targets[proj_name], mx.stack([self._cast_weight(sidecar_weights[s]) for s in srcs])))
        self.load_weights(weights, strict=False)
        mx.eval(self.parameters())

    def _sidecar_is_prequantized(self, sidecar_weights) -> bool:
        try:
            return any((str(name).endswith('.scales') for name in sidecar_weights))
        except Exception:
            return False

    def _trunk_quant_spec(self) -> dict | None:
        qcfg = dict(getattr(self.info, 'quantization', {}) or {})
        if not qcfg:
            cfg = self.dsa_config
            qcfg = dict(getattr(cfg, 'quantization_config', None) or getattr(cfg, 'quantization', None) or {})
        if not qcfg:
            return None
        return {'group_size': int(qcfg.get('group_size', 32) or 32), 'bits': int(qcfg.get('bits', 4) or 4), 'mode': str(qcfg.get('mode', 'mxfp4') or 'mxfp4')}

    def _maybe_requantize_to_trunk(self, sidecar_weights) -> None:
        policy = getattr(self, 'match_trunk_precision', 'auto')
        if policy != 'on':
            return
        if self._sidecar_is_prequantized(sidecar_weights):
            return
        spec = self._trunk_quant_spec() or {'group_size': 32, 'bits': 4, 'mode': 'mxfp4'}
        self._requantize_to_trunk(spec)

    @staticmethod
    def _trunk_quant_decision(path, has_to_quantized: bool, last_dim, spec: dict):
        if str(path).endswith('.mlp.gate'):
            return False
        if not has_to_quantized:
            return False
        try:
            group_size = int(spec['group_size'])
            if last_dim is None or int(last_dim) % group_size != 0:
                return False
            return {'group_size': group_size, 'bits': int(spec['bits']), 'mode': str(spec['mode'])}
        except Exception:
            return False

    def _requantize_to_trunk(self, spec: dict) -> None:
        module = self._as_module()

        def _predicate(path, module_obj, _spec=spec):
            weight = getattr(module_obj, 'weight', None)
            last_dim = None
            if weight is not None:
                try:
                    last_dim = int(weight.shape[-1])
                except Exception:
                    last_dim = None
            return self._trunk_quant_decision(path, hasattr(module_obj, 'to_quantized'), last_dim, _spec)
        self.nn.quantize(module, class_predicate=_predicate)
        self._refresh_refs_from_module(module)
        self.mx.eval(self.parameters())

    def _quantize_for_weight_keys(self, sidecar_weights) -> None:
        quantized_bases = {name[:-len('.scales')] for name in sidecar_weights if name.endswith('.scales')}
        if not quantized_bases:
            return
        qcfg = dict(getattr(self.info, 'quantization', {}) or {})
        group_size = int(qcfg.get('group_size', 32) or 32)
        bits = int(qcfg.get('bits', 4) or 4)
        mode = str(qcfg.get('mode', 'mxfp4') or 'mxfp4')
        wanted = mlx_runtime_quantizable_bases().intersection(quantized_bases)
        if not wanted:
            return
        module = self._as_module()

        def _predicate(path, module_obj, _wanted=wanted):
            if path in _wanted and hasattr(module_obj, 'to_quantized'):
                return {'group_size': group_size, 'bits': bits, 'mode': mode}
            return False
        self.nn.quantize(module, class_predicate=_predicate)
        self._refresh_refs_from_module(module)

    def _refresh_refs_from_module(self, module=None) -> None:
        module = module or self._as_module()
        self.enorm = module.enorm
        self.hnorm = module.hnorm
        self.eh_proj = module.eh_proj
        self.shared_head_norm = module.shared_head_norm
        self.mtp_block = module.mtp_block

    def _load_mlx_runtime_weights(self, sidecar_weights) -> None:
        expected = set(mlx_runtime_direct_tensor_map(self.info.source_layer).values())
        expected.update({'mtp_block.self_attn.embed_q.weight', 'mtp_block.self_attn.unembed_out.weight', 'mtp_block.mlp.switch_mlp.down_proj.weight', 'mtp_block.mlp.switch_mlp.gate_proj.weight', 'mtp_block.mlp.switch_mlp.up_proj.weight'})
        missing = [name for name in sorted(expected) if name not in sidecar_weights]
        if missing:
            raise RuntimeError(f'MTP runtime sidecar missing weights: {missing[:3]}')
        self._quantize_for_weight_keys(sidecar_weights)
        weights = [(name, self._cast_weight(value)) for name, value in sorted(sidecar_weights.items()) if not name.startswith('__')]
        self.load_weights(weights, strict=False)
        self.mx.eval(self.parameters())

    def make_cache(self):
        return self._cache_list_type(self._kv_cache_type(), self._kv_cache_type())

    @property
    def has_seed(self) -> bool:
        return self._seed_token is not None and self._seed_hidden is not None

    def reset_cache(self) -> None:
        self._mtp_cache = None
        self._round_appended = 0
        self._round_prefix_appended = 0
        self._seed_token = None
        self._seed_hidden = None
        self._shared_topk = None

    def reset_request(self) -> None:
        self.reset_cache()
        self.accepted = 0
        self.checked = 0

    def _cache(self):
        if self._mtp_cache is None:
            self._mtp_cache = self.make_cache()
        return self._mtp_cache

    def cache_offset(self) -> int:
        cache = self._cache()
        if not cache:
            return 0
        try:
            return int(getattr(cache[0], 'offset', 0) or 0)
        except Exception:
            return 0

    def rollback_cache(self, offset: int) -> None:
        target = max(0, int(offset))
        for entry in self._cache():
            current = int(getattr(entry, 'offset', target) or 0)
            delta = current - target
            if delta > 0 and hasattr(entry, 'trim'):
                entry.trim(delta)

    def load_weights(self, weights, strict=False):
        module = self._as_module()
        module.load_weights(weights, strict=strict)

    def parameters(self):
        return self._as_module().parameters()

    def _as_module(self):
        if hasattr(self, '_module'):
            return self._module
        nn = self.nn

        class _Module(nn.Module):
            pass
        module = _Module()
        module.enorm = self.enorm
        module.hnorm = self.hnorm
        module.eh_proj = self.eh_proj
        module.shared_head_norm = self.shared_head_norm
        module.mtp_block = self.mtp_block
        self._module = module
        return module

    @staticmethod
    def _first_scalar(value) -> int:
        try:
            return int(value.item())
        except Exception:
            return int(value)

    def _sample_token_array(self, logits, sampler=None):
        if callable(sampler):
            token = sampler(logits)
        else:
            token = self.mx.argmax(logits, axis=-1)
        try:
            if int(token.ndim) == 1:
                token = token[:, None]
        except Exception:
            pass
        return token

    def _forward_hidden(self, input_ids, previous_hidden_states, *, zero_first_token: bool=False):
        mx = self.mx
        emb = self.embed_tokens(input_ids) * self.embed_scale
        if self._compute_dtype is not None:
            emb = emb.astype(self._compute_dtype)
            previous_hidden_states = previous_hidden_states.astype(self._compute_dtype)
        if zero_first_token and emb.shape[1] > 0:
            emb = mx.concatenate([mx.zeros_like(emb[:, :1, :]), emb[:, 1:, :]], axis=1)
        hidden = self.eh_proj(mx.concatenate([self.enorm(emb), self.hnorm(previous_hidden_states)], axis=-1))
        cache = self._cache()
        mask = self._create_attention_mask(hidden, cache[0], return_array=True)
        if self._glm_decoder_layer_forward is not None:
            attn = self.mtp_block.self_attn
            old_skip_topk = bool(getattr(attn, 'skip_topk', False))
            prev_topk = self._reusable_shared_topk(input_ids)
            attn.skip_topk = bool(prev_topk is not None)
            try:
                hidden, topk = self._glm_decoder_layer_forward(self.mtp_block, hidden, mask, cache, prev_topk_indices=prev_topk)
            finally:
                attn.skip_topk = old_skip_topk
            if self._share_topk_enabled:
                self._shared_topk = topk if topk is not None else prev_topk
            else:
                self._shared_topk = None
        else:
            hidden = self.mtp_block(hidden, mask, cache)
        return self._recurrent_from_raw_mtp_hidden(hidden)

    def _forward(self, input_ids, previous_hidden_states, *, zero_first_token: bool=False):
        hidden = self._forward_hidden(input_ids, previous_hidden_states, zero_first_token=zero_first_token)
        return self._logits_from_mtp_hidden(hidden)

    def _logits_from_mtp_hidden(self, hidden):
        if self.recurrent_state == 'raw':
            hidden = self.shared_head_norm(hidden)
        return self.lm_head(hidden)

    def _recurrent_from_raw_mtp_hidden(self, hidden):
        if self.recurrent_state == 'raw':
            return hidden
        return self.shared_head_norm(hidden)

    def prefill(self, input_ids, previous_hidden_states):
        self.reset_iteration_topk()
        zero_first = bool(getattr(self._cache()[0], 'offset', 0) == 0)
        hidden = self._forward_hidden(input_ids, previous_hidden_states, zero_first_token=zero_first)
        self.mx.eval(hidden)
        self.reset_iteration_topk()
        return hidden

    def prefill_from_target_hidden(self, input_ids, previous_hidden_states, bonus_token: int, shared_topk=None, sampler=None):
        mx = self.mx
        if int(input_ids.shape[1]) <= 0:
            return None
        self.reset_iteration_topk()
        self.seed_shared_topk(shared_topk, tail=False)
        bonus = mx.array([[self._first_scalar(bonus_token)]], dtype=input_ids.dtype)
        shifted = mx.concatenate([input_ids[:, 1:], bonus], axis=1)
        hidden = self._forward_hidden(shifted, previous_hidden_states[:, :int(shifted.shape[1]), :], zero_first_token=False)
        logits = self._logits_from_mtp_hidden(hidden[:, -1:, :])
        self._seed_token = self._sample_token_array(logits, sampler)
        self._seed_hidden = hidden[:, -1:, :]
        mx.eval(self._seed_token, self._seed_hidden)
        return hidden

    def prefill_committed_prompt_history(self, input_ids, previous_hidden_states, shared_topk=None):
        if int(input_ids.shape[1]) <= 1:
            return None
        return self.append_committed_from_previous_hidden(input_ids[:, 1:], previous_hidden_states[:, :-1, :], shared_topk=shared_topk, update_seed=False)

    def append_committed_from_previous_hidden(self, input_ids, previous_hidden_states, *, preserve_topk: bool=False, update_seed: bool=True, shared_topk=None, reuse_existing_topk: bool=False, sampler=None):
        mx = self.mx
        if int(input_ids.shape[1]) <= 0:
            return None
        if not reuse_existing_topk:
            self.reset_iteration_topk()
        self.seed_shared_topk(shared_topk, tail=False)
        zero_first = bool(getattr(self._cache()[0], 'offset', 0) == 0)
        hidden = self._forward_hidden(input_ids, previous_hidden_states, zero_first_token=zero_first)
        if update_seed:
            logits = self._logits_from_mtp_hidden(hidden[:, -1:, :])
            self._seed_token = self._sample_token_array(logits, sampler)
            self._seed_hidden = hidden[:, -1:, :]
            mx.eval(self._seed_token, self._seed_hidden)
        else:
            self._seed_token = None
            self._seed_hidden = None
            if self.history_materialization == 'eager':
                mx.eval(hidden)
        if not preserve_topk and (not update_seed):
            self.reset_iteration_topk()
        return hidden

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted: int, new_tokens, sampler=None) -> None:
        mx = self.mx
        accepted = max(0, int(accepted))
        prefix_appended = int(getattr(self, '_round_prefix_appended', 0) or 0)
        draft_appended = int(getattr(self, '_round_appended', 0) or 0)
        keep_drafts = min(accepted, draft_appended)
        total_appended = prefix_appended + draft_appended
        keep_total = prefix_appended + keep_drafts
        trim = total_appended - keep_total
        if trim > 0:
            for entry in self._cache():
                if hasattr(entry, 'trim'):
                    entry.trim(trim)
            self._shared_topk = None
        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_drafts, accepted):
            token_chunks.append(draft_tokens[:, draft_idx:draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx:draft_idx + 1, :])
        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=draft_tokens.dtype))
            hidden_chunks.append(verify_hidden[:, accepted:accepted + 1, :])
        if token_chunks:
            tokens = token_chunks[0] if len(token_chunks) == 1 else mx.concatenate(token_chunks, axis=1)
            hiddens = hidden_chunks[0] if len(hidden_chunks) == 1 else mx.concatenate(hidden_chunks, axis=1)
            hidden = self._forward_hidden(tokens, hiddens, zero_first_token=False)
            logits = self._logits_from_mtp_hidden(hidden[:, -1:, :])
            self._seed_token = self._sample_token_array(logits, sampler)
            self._seed_hidden = hidden[:, -1:, :]
            mx.eval(self._seed_token, self._seed_hidden)
        else:
            self._seed_token = None
            self._seed_hidden = None
        self._round_appended = 0
        self._round_prefix_appended = 0
        if self._seed_token is None or self._seed_hidden is None:
            self.reset_iteration_topk()

    def propose(self, input_token_id: int, previous_hidden_states, sampler=None):
        used_seed = self._seed_token is not None and self._seed_hidden is not None
        if not used_seed:
            self.reset_iteration_topk()
        if self._seed_token is not None and self._seed_hidden is not None:
            seed = self._seed_token
            logits = self._logits_from_mtp_hidden(self._seed_hidden)
            draft_id = self._first_scalar(seed[:, -1])
            self._seed_token = None
            self._seed_hidden = None
            return (draft_id, logits)
        token = self.mx.array([[int(input_token_id)]])
        logits = self._forward(token, previous_hidden_states)
        draft_id = self._first_scalar(self._sample_token_array(logits[:, -1:, :], sampler))
        self.reset_iteration_topk()
        return (draft_id, logits)

    def propose_many(self, input_token_id: int, previous_hidden_states, max_draft: int, *, reuse_existing_topk: bool=False, sampler=None, return_tensors: bool=True):
        mx = self.mx
        collect_tensors = bool(return_tensors)
        self._round_appended = 0
        self._round_prefix_appended = 0
        drafts: list[int] = []
        logits_rows = []
        hidden_rows = []
        max_draft = max(0, int(max_draft))
        if max_draft <= 0:
            return (drafts, logits_rows, hidden_rows)
        token = int(input_token_id)
        hidden_base = previous_hidden_states
        used_seed = self._seed_token is not None and self._seed_hidden is not None
        if not reuse_existing_topk and (not used_seed):
            self.reset_iteration_topk()
        if self._seed_token is not None and self._seed_hidden is not None:
            seed = self._seed_token
            hidden = self._seed_hidden
            draft_id = self._first_scalar(seed[:, -1])
            self._seed_token = None
            self._seed_hidden = None
            drafts.append(draft_id)
            if collect_tensors:
                logits = self._logits_from_mtp_hidden(hidden)
                logits_rows.append(logits)
                hidden_rows.append(hidden)
            token = draft_id
            hidden_base = hidden[:, -1:, :]
        while len(drafts) < max_draft:
            if hidden_base is None:
                break
            hidden = self._forward_hidden(mx.array([[int(token)]]), hidden_base)
            if not used_seed and (not drafts):
                self._round_prefix_appended = 1
            else:
                self._round_appended += 1
            logits = self._logits_from_mtp_hidden(hidden)
            draft_id = self._first_scalar(self._sample_token_array(logits[:, -1:, :], sampler))
            drafts.append(draft_id)
            if collect_tensors:
                logits_rows.append(logits)
                hidden_rows.append(hidden)
            token = draft_id
            hidden_base = hidden[:, -1:, :]
        if collect_tensors and (logits_rows or hidden_rows):
            mx.eval(*logits_rows, *hidden_rows)
        return (drafts, logits_rows, hidden_rows)

    def record_verification(self, draft_id: int | None, target_id: int, *, accepted: bool | None=None) -> None:
        if draft_id is None:
            return
        self.checked += 1
        is_accepted = bool(accepted) if accepted is not None else int(draft_id) == int(target_id)
        if is_accepted:
            self.accepted += 1

    @property
    def acceptance_rate(self) -> float:
        if self.checked <= 0:
            return 0.0
        return self.accepted / self.checked
