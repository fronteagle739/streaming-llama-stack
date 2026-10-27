from __future__ import annotations
import argparse
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from kv_handoff import _hash_token_ids, inject_standard_rows, _sampling_profile_from_kva_header, _parse_kva_prompt_payload, _parse_kva_prompt_payload_bytes, _describe_kva_ref, _release_kv_views, _keep_warm_enabled, _keep_warm_period_s, _start_keep_warm_thread, _keep_warm_tick, _cache_entry_offset, _cache_offset, _cache_leaf_offsets, _can_trim_cache_to_offset, _trim_cache_to_offset, _validate_kv_injection_cache, _tbp1_send_frame, _recv_into, _recv_all, _receive_kva_via_tbp1, _tbp1_listener_thread, prefix_query, inject_mla_rows, select_delta_base, TBP_MAGIC, TBP_HEADER_SIZE, TBP_MSG_HELLO, TBP_MSG_SESSION_START, TBP_MSG_KV_SEG_BEGIN, TBP_MSG_KV_CHUNK, TBP_MSG_KV_SEG_END, TBP_MSG_KV_DONE, TBP_MSG_QUERY_PREFIX_CACHE, TBP_MSG_PREFIX_CACHE_REPLY
from glm52_mtp import is_quantized_mtp_sidecar, resolve_sidecar_dir, validate_sidecar
from glm52_mtp_drafter import target_logits_and_hidden, target_logits_hidden_topk
from mtp_adaptive_depth import make_adaptive_depth_controller
from mlx_adapters import attach_mlx_loras, attach_token_sidecars
from swi_reasoning import SwiConfig, SwiState, swi_alpha, swi_beta
try:
    import transformers.dynamic_module_utils as _dmu
    _orig_resolve_trc = _dmu.resolve_trust_remote_code

    def _auto_trust(trust_remote_code, *a, **kw):
        return _orig_resolve_trc(True, *a, **kw)
    _dmu.resolve_trust_remote_code = _auto_trust
except Exception:
    pass
STOP_NONE = 0
STOP_EOS = 1
STOP_LIMIT = 2
STOP_WORD = 3
STOP_CANCELLED = 4
_SAMPLING_PROFILE_DEFAULTS = {'seed': 0, 'min_keep': 1, 'top_k': 40, 'top_p': 0.95, 'min_p': 0.05, 'typ_p': 1.0, 'temp': 0.8, 'dynatemp_range': 0.0, 'dynatemp_exponent': 1.0, 'penalty_last_n': 64, 'penalty_repeat': 1.0, 'penalty_freq': 0.0, 'penalty_present': 0.0, 'xtc_probability': 0.0, 'xtc_threshold': 0.1, 'top_n_sigma': -1.0, 'mirostat': 0, 'mirostat_tau': 5.0, 'mirostat_eta': 0.1, 'user_sampling_config': 0}

def _is_dsv41(model_or_config) -> bool:
    config = getattr(model_or_config, 'config', model_or_config)
    kind = config.get('model_type') if isinstance(config, dict) else getattr(config, 'model_type', None)
    return kind == 'deepseek_v41'

def _dsv41_available_breakdown(vm_stat: str) -> dict:
    import re
    page_size = int(re.search('page size of (\\d+) bytes', vm_stat).group(1))
    counts = {key.strip(): int(value.strip().rstrip('.')) for line in vm_stat.splitlines()[1:] if ':' in line for key, value in [line.split(':', 1)]}
    out = {key: page_size * counts[key] for key in ('Pages free', 'Pages inactive', 'Pages speculative', 'Pages purgeable')}
    file_backed = counts.get('File-backed pages')
    out['file_extra'] = 0 if file_backed is None else page_size * max(0, file_backed - counts['Pages inactive'] - counts['Pages speculative'])
    out['available'] = sum(out.values())
    out['wired'] = page_size * counts.get('Pages wired down', 0)
    return out

def _dsv41_available_bytes(vm_stat: str) -> int:
    return _dsv41_available_breakdown(vm_stat)['available']

def _dsv41_breakdown_text(breakdown: dict) -> str:
    return ' '.join(('%s=%.1f' % (k.replace('Pages ', ''), v / (1 << 30)) for k, v in breakdown.items() if k != 'available')) + ' GiB'

def _dsv41_wait_for_memory(source: str, required_bytes, read_vm_stat, *, timeout_s: float, poll_s: float=2.0, say=None, clock=None, sleep=None):
    import time as _time
    clock = clock or _time.monotonic
    sleep = sleep or _time.sleep
    say = say or (lambda msg: print(msg, file=sys.stderr, flush=True))
    t0 = clock()
    last_err = None
    while True:
        breakdown = _dsv41_available_breakdown(read_vm_stat())
        available = breakdown['available']
        try:
            _dsv41_memory_guard(available, source, required_bytes)
            return (available, breakdown)
        except RuntimeError as exc:
            last_err = exc
        waited = clock() - t0
        say(f'[DSV41] memory guard waiting ({waited:.0f}/{timeout_s:.0f} s): {_dsv41_breakdown_text(breakdown)}')
        if waited >= timeout_s:
            raise RuntimeError(f'{last_err} after waiting {waited:.0f} s [{_dsv41_breakdown_text(breakdown)}]')
        sleep(poll_s)

def _dsv41_memory_guard(available: int, source: str, required_bytes=None) -> int:
    if source not in ('hybrid', 'nvme', 'ram'):
        raise ValueError(f'Invalid V4.1 Engram source: {source!r}')
    required = (468 if source == 'ram' else 279) * (1 << 30) if required_bytes is None else int(required_bytes)
    if available < required + 20 * (1 << 30):
        raise RuntimeError(f'[DSV41] Refusing to load: host-wide available memory {available / (1 << 30):.2f} GiB; need {required / (1 << 30):.0f} GiB resident + 20 GiB headroom ({source})')
    return required

def _dsv41_encoding():
    from mlx_vlm.models.deepseek_v41 import encoding_dsv41
    return encoding_dsv41

def _dsv41_render_prompt(messages, thinking_mode='chat', reasoning_effort=None) -> str:
    if not messages or not isinstance(messages, list):
        raise ValueError('[DSV41] Expected a nonempty messages list')
    for message in messages:
        content = message.get('content')
        if isinstance(content, list) and any((block.get('type') != 'text' for block in content)):
            raise ValueError('[DSV41] Only text messages are supported')
    return _dsv41_encoding().encode_messages(messages, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)

def _dsv41_stop_id(tokenizer) -> int:
    token_id = tokenizer.token_to_id('<｜end▁of▁sentence｜>')
    if token_id is None:
        raise ValueError('[DSV41] tokenizer.json is missing the end-of-sentence token')
    return int(token_id)

class _Dsv41Tokenizer:

    def __init__(self, model_dir):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(str(Path(model_dir) / 'tokenizer.json'))
        self.eos_token_id = _dsv41_stop_id(self.backend_tokenizer)

    def encode(self, text, add_special_tokens=False):
        return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)

def _dsv41_parse_completion(text, thinking_mode):
    encoding = _dsv41_encoding()
    candidate = text
    if thinking_mode == 'thinking' and encoding.thinking_end_token not in candidate:
        candidate += encoding.thinking_end_token
    if not candidate.endswith(encoding.eos_token):
        candidate += encoding.eos_token
    try:
        return encoding.parse_message_from_completion_text(candidate, thinking_mode)
    except (AssertionError, ValueError, IndexError) as exc:
        return {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': [], 'raw_completion': text, 'parse_error': str(exc)}

def _dsv41_materialize(model, mx):
    modules = [('embedding', model.language_model.model.embed_tokens)]
    modules.extend(((f'layer {i}', layer) for i, layer in enumerate(model.layers)))
    modules.extend([('final norm', model.language_model.model.norm), ('lm_head', model.language_model.lm_head)])
    if hasattr(model, 'vision'):
        modules.extend([('vision', model.vision), ('aligner', model.aligner)])
    for name, module in modules:
        with mx.stream(mx.cpu):
            mx.eval(module.parameters())
        mx.clear_cache()
        print(f'[DSV41] materialized {name} resident={mx.get_active_memory() / (1 << 30):.2f} GiB', file=sys.stderr, flush=True)

def _load_dsv41(model_dir, raw):
    import subprocess
    import mlx.core as mx
    import mlx.nn as nn
    source = (os.environ.get('LLAMA_MLX_ENGRAM_SOURCE') or raw.get('engram_source', raw.get('text_config', {}).get('engram_source', 'hybrid'))).strip().lower()

    def _read_vm_stat() -> str:
        try:
            return subprocess.check_output(['vm_stat'], text=True)
        except Exception as exc:
            raise RuntimeError('[DSV41] Refusing to load: cannot read host-wide vm_stat memory') from exc
    from math import prod
    from mlx_vlm.models.deepseek_v41 import Model, ModelConfig
    from mlx_vlm.models.deepseek_v41.checkpoint_layout import make_quantization_config_v41, expected_parameters
    from mlx_vlm.models.deepseek_v41.vision_layout import expected_vision_parameters
    raw = dict(raw, model_dir=str(model_dir), engram_source=source)
    config = ModelConfig.from_dict(raw)
    sizes = {'bfloat16': 2, 'float32': 4, 'uint32': 4, 'uint8': 1}
    specs = expected_parameters(config.text_config, source)
    resident_specs = {name: spec for name, spec in specs.items() if not (name.startswith('dspark.') and os.environ.get('DSV41_DSPARK', '0') != '1')}
    parameter_bytes = sum((sizes[dtype] * prod(shape) for dtype, shape in resident_specs.values()))
    if config.vision_config is not None:
        parameter_bytes += sum((2 * prod(shape) for _, shape in expected_vision_parameters(config.vision_config, config.text_config.hidden_size).values()))
    if os.environ.get('DSV41_PROMOTE_DECODE_WEIGHTS', '0') == '1':
        parameter_bytes += sum(((4 - sizes[dtype]) * prod(shape) for name, (dtype, shape) in specs.items() if name.endswith('.wo_a.weight') and os.environ.get('DSV41_WO_A_BF16', '0') != '1' or (name == 'lm_head.weight' and os.environ.get('DSV41_HEAD_BF16', '0') != '1')))
    available, breakdown = _dsv41_wait_for_memory(source, parameter_bytes, _read_vm_stat, timeout_s=float(os.environ.get('DSV41_MEMORY_GUARD_WAIT_S', '90')))
    required = _dsv41_memory_guard(available, source, parameter_bytes)
    print(f"[DSV41] memory available={available / (1 << 30):.2f} GiB resident_required={required / (1 << 30):.0f} GiB headroom=20 GiB source={source} QAT={os.environ.get('DSV41_QAT', '1')} [{_dsv41_breakdown_text(breakdown)}]", file=sys.stderr, flush=True)
    model_dir = Path(model_dir)
    model = Model(config)
    weight_map = json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    shards = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    weights = {}
    for shard, keys in sorted(shards.items()):
        loaded = mx.load(str(model_dir / shard))
        weights.update(((key, loaded[key]) for key in keys))
        del loaded
    with mx.stream(mx.cpu):
        weights = model.sanitize(weights)
    quantization = make_quantization_config_v41(model)

    def predicate(path, module):
        if path in quantization:
            return quantization[path]
        if not hasattr(module, 'to_quantized'):
            return False
        if hasattr(module, 'weight') and module.weight.size % 64 != 0:
            return False
        return f'{path}.scales' in weights
    nn.quantize(model, group_size=quantization['group_size'], bits=quantization['bits'], mode=quantization.get('mode', 'affine'), class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    del weights
    model.eval()
    _dsv41_materialize(model, mx)
    model.attach_engram_storage(model_dir)
    if source == 'ram':
        from mlx.utils import tree_flatten
        mx.eval([getattr(model, n) for n in ('image_start', 'image_end', 'image_newline') if hasattr(model, n)])
        table_bytes = sum((a.nbytes for name, a in tree_flatten(model.parameters()) if '.engram.embed.' in name))
        if any((layer.engram is not None and (not layer.engram.resident or layer.engram.shard is not None or layer.engram.rows_cache is not None) for layer in model.layers)):
            raise RuntimeError('[DSV41] Resident mode unexpectedly attached disk-backed Engram storage')
        print(f'[DSV41] fully resident: Engram={table_bytes} bytes MLX active={mx.get_active_memory()} bytes; no row cache or NVMe mapping', file=sys.stderr, flush=True)
    print(f'[DSV41] Engram attached stats={json.dumps(model.language_model.engram_stats())}', file=sys.stderr, flush=True)
    return (model, _Dsv41Tokenizer(model_dir))

def _dsv41_save_hot(session):
    if not _is_dsv41(session.model):
        return
    try:
        counts = session.model.language_model.save_engram_hot()
        print(f'[DSV41] saved Engram hot rows={sum(counts.values())} tables={counts}', file=sys.stderr, flush=True)
    except Exception as exc:
        print(f'[DSV41] save_engram_hot failed: {exc}', file=sys.stderr, flush=True)

def _default_sampling_profile() -> Dict[str, Any]:
    return dict(_SAMPLING_PROFILE_DEFAULTS)

def _sampling_profile_from_config(cfg: Any) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    src: Dict[str, Any] = {}
    nested = cfg.get('sampling')
    if isinstance(nested, dict):
        src.update(nested)
    src.update(cfg)
    aliases = {'temperature': 'temp', 'temp': 'temp', 'top_p': 'top_p', 'top_k': 'top_k', 'min_p': 'min_p', 'typical_p': 'typ_p', 'typ_p': 'typ_p', 'min_keep': 'min_keep', 'seed': 'seed', 'mirostat': 'mirostat', 'mirostat_tau': 'mirostat_tau', 'mirostat_eta': 'mirostat_eta'}
    out: Dict[str, Any] = {}
    for src_key, dst_key in aliases.items():
        if src_key in src and src[src_key] is not None:
            out[dst_key] = src[src_key]
    return out

def _merge_sampling_profiles(*profiles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _default_sampling_profile()
    for profile in profiles:
        if not profile:
            continue
        for key, value in profile.items():
            if key in merged and value is not None:
                merged[key] = value
    for key in ('seed', 'min_keep', 'top_k', 'penalty_last_n', 'mirostat'):
        try:
            merged[key] = int(merged[key])
        except Exception:
            merged[key] = int(_SAMPLING_PROFILE_DEFAULTS[key])
    for key in ('top_p', 'min_p', 'typ_p', 'temp', 'dynatemp_range', 'dynatemp_exponent', 'penalty_repeat', 'penalty_freq', 'penalty_present', 'xtc_probability', 'xtc_threshold', 'top_n_sigma', 'mirostat_tau', 'mirostat_eta'):
        try:
            merged[key] = float(merged[key])
        except Exception:
            merged[key] = float(_SAMPLING_PROFILE_DEFAULTS[key])
    merged['min_keep'] = max(1, int(merged['min_keep']))
    merged['top_k'] = max(0, int(merged['top_k']))
    merged['top_p'] = min(1.0, max(0.0, float(merged['top_p'])))
    merged['min_p'] = max(0.0, float(merged['min_p']))
    merged['temp'] = max(0.0, float(merged['temp']))
    return merged

def _overlay_sampling_profile_overrides(*profiles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for profile in profiles:
        if not profile:
            continue
        for key, value in profile.items():
            if key in _SAMPLING_PROFILE_DEFAULTS and value is not None:
                merged[key] = value
    return merged

def _strip_language_model_prefix(weights: dict) -> dict:
    prefix = 'language_model.'
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

def _strip_language_model_quant_config(qcfg: dict) -> dict:
    if not isinstance(qcfg, dict):
        return {}
    stripped = {}
    prefix = 'language_model.'
    passthrough = {'group_size', 'quant_method', 'bits', 'mode'}
    for k, v in qcfg.items():
        if k in passthrough:
            stripped[k] = v
        elif k.startswith(prefix):
            stripped[k[len(prefix):]] = v
    return stripped

def _apply_mixed_quant_from_weights(model, weights: dict, group_size: int, quantization_config: Optional[dict]=None) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from collections import defaultdict
    qcfg = quantization_config if isinstance(quantization_config, dict) else {}
    global_mode = str(qcfg.get('mode') or 'affine')
    param_in_dim = {}
    for path, arr in tree_flatten(model.parameters()):
        if path.endswith('.weight') and getattr(arr, 'ndim', 0) >= 2:
            param_in_dim[path[:-len('.weight')]] = int(arr.shape[-1])
    groups = defaultdict(set)
    n_fallback = 0
    for wk, w_arr in weights.items():
        if not wk.endswith('.weight'):
            continue
        layer_path = wk[:-len('.weight')]
        sk = layer_path + '.scales'
        if sk not in weights:
            continue
        s_arr = weights[sk]
        if w_arr.dtype != mx.uint32 or w_arr.ndim not in (2, 3):
            continue
        n_groups = int(s_arr.shape[-1])
        if n_groups <= 0:
            continue
        input_dim = param_in_dim.get(layer_path, 0)
        if input_dim > 0 and input_dim % n_groups == 0 and (input_dim // n_groups in (32, 64, 128)):
            gs = input_dim // n_groups
        else:
            if not group_size:
                continue
            gs = int(group_size)
            input_dim = n_groups * gs
            n_fallback += 1
        bits = round(w_arr.shape[-1] * 32 / input_dim)
        if bits <= 0:
            continue
        override = qcfg.get(layer_path) or qcfg.get('language_model.' + layer_path)
        mode = str(override.get('mode') or global_mode) if isinstance(override, dict) else global_mode
        groups[bits, gs, mode].add(layer_path)
    if groups:
        sys.stderr.write(f"[mlx-server] per-tensor mixed quant: {sum((len(v) for v in groups.values()))} modules across {len(groups)} (bits, group_size, mode) combos {sorted(((k, len(v)) for k, v in groups.items()))}{(f' ({n_fallback} on the config group size)' if n_fallback else '')}\n")
        sys.stderr.flush()
    for (bits, gs, mode), layer_set in sorted(groups.items()):

        def _predicate(path, module, _layer_set=layer_set):
            return path in _layer_set and hasattr(module, 'to_quantized')
        nn.quantize(model, group_size=gs, bits=bits, mode=mode, class_predicate=_predicate)

def _load_kimi_text_lm_from_vlm_checkpoint(model_path: str, lazy: bool=True, return_vision_weights: bool=False):
    import glob
    import json
    import mlx.core as mx
    from mlx_lm.utils import _get_classes, load_tokenizer
    model_dir = Path(model_path)
    with open(model_dir / 'config.json', 'r', encoding='utf-8') as f:
        root_config = json.load(f)
    text_config = dict(root_config.get('text_config') or {})
    if not text_config:
        raise ValueError('checkpoint has no text_config for MLX-LM text load')
    model_type = str(text_config.get('model_type', '')).lower()
    if model_type not in ('kimi_k2', 'deepseek_v3'):
        raise ValueError(f'text_config model_type {model_type!r} is not a Kimi/DeepSeek MLA model')
    qcfg = root_config.get('quantization_config') or text_config.get('quantization_config') or {}
    text_config['quantization_config'] = _strip_language_model_quant_config(qcfg)
    text_config.pop('auto_map', None)
    weight_files = sorted(glob.glob(str(model_dir / 'model*.safetensors')))
    if not weight_files:
        weight_files = sorted(glob.glob(str(model_dir / '*.safetensors')))
    if not weight_files:
        raise FileNotFoundError(f'No safetensors found in {model_dir}')
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    try:
        from mlx_vlm.inf_decode import build_name_map, decode_keystream, index_keystream, is_inf_checkpoint
        if is_inf_checkpoint(model_dir):
            ks = index_keystream(model_dir) or decode_keystream(model_dir, root_config)
            name_map = build_name_map(model_dir, ks)
            renamed = {name_map.get(k, k): v for k, v in weights.items()}
            missing = sum((1 for k in weights if k not in name_map))
            sys.stderr.write(f'[mlx-server] INF checkpoint: renamed {len(weights) - missing}/{len(weights)} tensors via the decoded keystream\n')
            sys.stderr.flush()
            weights = renamed
    except ImportError:
        pass
    vision_weights = {k: v for k, v in weights.items() if k.startswith(('vision_tower.', 'mm_projector.'))}
    weights = _strip_language_model_prefix(weights)
    if not weights:
        raise ValueError('no language_model.* weights found in checkpoint')
    model_class, model_args_class = _get_classes(text_config)
    model = model_class(model_args_class.from_dict(text_config))
    if hasattr(model, 'sanitize'):
        weights = model.sanitize(weights)
    _qcfg = text_config.get('quantization_config') or {}
    group_size = int(_qcfg.get('group_size', 64) or 64)
    _apply_mixed_quant_from_weights(model, weights, group_size, quantization_config=_qcfg)
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    if not lazy:
        mx.eval(model.parameters())
    tokenizer = load_tokenizer(model_dir, eos_token_ids=text_config.get('eos_token_id', None))
    if return_vision_weights:
        return (model, tokenizer, text_config, vision_weights)
    return (model, tokenizer, text_config)

def _has_vlm_text_config(model_path: str) -> bool:
    try:
        with open(Path(model_path) / 'config.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return False
    return isinstance(cfg.get('text_config'), dict)
_KV_INJECTED = b'\x01KV-INJECTED'
_KIMI_PSEUDO_BASE = 1 << 30

def _first_fwd_probe_enabled() -> bool:
    if os.environ.get('LLAMA_MLX_FIRST_FWD_PROBE', '') not in ('', '0'):
        return True
    try:
        return os.path.exists(os.path.expanduser('~/.streaming-studio/first_fwd_probe'))
    except Exception:
        return False

def _flatten_mx_arrays(obj, out=None, depth=0):
    import mlx.core as mx
    if out is None:
        out = []
    if depth > 6 or obj is None:
        return out
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _flatten_mx_arrays(x, out, depth + 1)
        return out
    for attr in ('keys', 'values'):
        a = getattr(obj, attr, None)
        if isinstance(a, mx.array):
            out.append(a)
    children = getattr(obj, 'caches', None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _flatten_mx_arrays(c, out, depth + 1)
    return out

def _first_fwd_probe_layers(lang):
    seen = []
    roots = [lang]
    for name in ('model', 'language_model'):
        sub = getattr(lang, name, None)
        if sub is not None:
            roots.append(sub)
            subsub = getattr(sub, 'model', None)
            if subsub is not None:
                roots.append(subsub)
    for root in roots:
        for attr in ('layers', 'blocks', 'h'):
            obj = getattr(root, attr, None)
            if isinstance(obj, list) and obj and all((callable(x) for x in obj)):
                for x in obj:
                    if all((x is not y for y in seen)):
                        seen.append(x)
    return seen or None

def _first_fwd_probe_wrap(lang):
    import mlx.core as mx
    layers = _first_fwd_probe_layers(lang)
    if not layers:
        sys.stderr.write('[mlx-server] first-forward probe: no decoder layer list found\n')
        sys.stderr.flush()
        return None
    times = []
    swapped = []
    cache = {}

    def _timed_call(self, *a, **kw):
        t0 = time.time()
        out = super(type(self), self).__call__(*a, **kw)
        try:
            if isinstance(out, mx.array):
                mx.eval(out)
            elif isinstance(out, (list, tuple)):
                mx.eval(*[x for x in out if isinstance(x, mx.array)])
        except Exception:
            pass
        times.append(time.time() - t0)
        return out
    for layer in layers:
        cls = type(layer)
        if cls not in cache:
            try:
                cache[cls] = type('Timed' + cls.__name__, (cls,), {'__call__': _timed_call})
            except TypeError:
                continue
        try:
            layer.__class__ = cache[cls]
            swapped.append((layer, cls))
        except TypeError:
            pass
    return (swapped, times)

def _first_fwd_probe_report(probe, total_s: float) -> None:
    swapped, times = probe
    for layer, cls in swapped:
        try:
            layer.__class__ = cls
        except TypeError:
            pass
    if not times:
        sys.stderr.write('[mlx-server] first-forward probe: no layer was called\n')
        sys.stderr.flush()
        return
    order = sorted(range(len(times)), key=lambda i: -times[i])
    top = ', '.join((f'L{i}={times[i]:.3f}s' for i in order[:6]))
    srt = sorted(times)
    sys.stderr.write(f'[mlx-server] first-forward probe: {len(times)} layers sum={sum(times):.2f}s of {total_s:.2f}s (min {srt[0]:.3f}s median {srt[len(srt) // 2]:.3f}s max {srt[-1]:.3f}s); slowest: {top} [FIRST_FWD_PROBE]\n')
    sys.stderr.flush()

def _clear_cache_per_request() -> bool:
    return os.environ.get('LLAMA_MLX_CLEAR_CACHE_PER_REQUEST', '') not in ('', '0')

def _kva_prompt_count_hash(kva_ref) -> Tuple[int, int]:
    close_after = False
    if isinstance(kva_ref, (bytes, bytearray, memoryview)):
        data = memoryview(kva_ref)

        def read_at(off: int, n: int) -> bytes:
            return data[off:off + n]
        size = len(data)
    else:
        f = open(kva_ref, 'rb')
        close_after = True

        def read_at(off: int, n: int) -> bytes:
            f.seek(off)
            return f.read(n)
        size = os.path.getsize(kva_ref)
    try:
        if size < 44:
            raise ValueError(f'kva too short: {size} bytes')
        if bytes(read_at(0, 8)) != b'KVARTIF1':
            raise ValueError('bad KVA magic')
        header = bytes(read_at(8, 36))
        _format_major, format_minor, _n_layers, _n_ctx, _token_count, _type_k, _type_v, _flags = struct.unpack_from('<HH III HH I', header, 0)
        payload_bytes, _payload_crc32 = struct.unpack_from('<QI', header, 24)
        off = 44
        if format_minor >= 1:
            off += 16
        if format_minor >= 3:
            off += 84
        if format_minor >= 4:
            off += 4
        prompt_token_count = 0
        prompt_text_bytes = 0
        if format_minor >= 5:
            prompt_header = bytes(read_at(off, 12))
            prompt_token_count, prompt_text_bytes, _streamed_prefix = struct.unpack_from('<III', prompt_header, 0)
            off += 12
        if format_minor >= 6:
            off += 8
        if format_minor >= 7:
            off += 4
        img_cell_count = 0
        img_n_embd = 0
        img_raw_bytes = 0
        if format_minor >= 8:
            _img_hdr = bytes(read_at(off, 16))
            img_cell_count, img_n_embd, _ps, img_raw_bytes = struct.unpack_from('<IIII', _img_hdr, 0)
            off += 16
        if not _flags & 1 << 5:
            img_raw_bytes = 0
        img_embd_bytes = img_cell_count * img_n_embd * 4
        mtp_hidden_bytes = 0
        if format_minor >= 9:
            _mtp_hdr = bytes(read_at(off, 8))
            _mtp_n_tok, _mtp_n_embd = struct.unpack_from('<II', _mtp_hdr, 0)
            off += 8
            mtp_hidden_bytes = _mtp_n_tok * _mtp_n_embd * 2
        if format_minor >= 10:
            off += 8
        if prompt_token_count <= 0:
            return (0, 0, img_cell_count)
        payload_offset = off
        payload_end = payload_offset + payload_bytes
        if payload_end > size:
            raise ValueError(f'kva payload truncated: header says {payload_bytes} bytes from {payload_offset}, object is {size}')
        prompt_token_off = payload_end - mtp_hidden_bytes - img_raw_bytes - img_embd_bytes - prompt_token_count * 4
        if prompt_token_off < payload_offset or prompt_token_off + prompt_token_count * 4 > size:
            raise ValueError('KVA prompt token payload is out of range')
        h = 14695981039346656037
        for i in range(prompt_token_count):
            tok_bytes = bytes(read_at(prompt_token_off + i * 4, 4))
            v = struct.unpack_from('<I', tok_bytes, 0)[0]
            for shift in (0, 8, 16, 24):
                h ^= v >> shift & 255
                h = h * 1099511628211 & 18446744073709551615
        return (prompt_token_count, h, img_cell_count)
    finally:
        if close_after:
            f.close()

class LineSocket:

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b''
        self.rid = '-'
        self.started = time.monotonic()
        self._accepted_logged = False
        self._first_token = False
        self.local = self.peer = None
        for attr, getter in (('local', 'getsockname'), ('peer', 'getpeername')):
            try:
                setattr(self, attr, getattr(sock, getter)())
            except OSError:
                pass
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def _log(self, stage, error=None, event='ok', elapsed=None):
        if not self._accepted_logged and stage != 'accept':
            self._accepted_logged = True
            self._log('accept', elapsed=0.0)
        t = time.monotonic() - self.started if elapsed is None else elapsed
        sys.stderr.write(f"[mlx-transport] rid={self.rid} stage={stage} t={t:.3f}s event={event} local={self.local!r} peer={self.peer!r} errno={getattr(error, 'errno', None)} winerror={getattr(error, 'winerror', None)} exception={(type(error).__name__ if error else '-')}\n")
        sys.stderr.flush()

    def configure_diagnostics(self, cfg):
        rid = cfg.get('rid')
        if isinstance(rid, str) and len(rid) == 8 and all((c in '0123456789abcdef' for c in rid)):
            self.rid = rid
        self._log('config-read')

    def read_line(self, timeout_s: float=30.0) -> Optional[str]:
        deadline = time.monotonic() + timeout_s
        while b'\n' not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._log('config-recv', TimeoutError(), event='timeout')
                return None
            try:
                self.sock.settimeout(remaining)
                chunk = self.sock.recv(4096)
            except (socket.timeout, TimeoutError) as exc:
                self._log('config-recv', exc, event='timeout')
                return None
            except OSError as exc:
                self._log('config-recv', exc, event='error')
                return None
            if not chunk:
                self._log('config-recv', event='eof')
                return None
            self._buf += chunk
        nl = self._buf.index(b'\n')
        line = self._buf[:nl].decode('utf-8', errors='replace')
        self._buf = self._buf[nl + 1:]
        return line

    def write_line(self, line: str) -> bool:
        try:
            msg = json.loads(line)
        except ValueError:
            msg = {}
        stage = 'done-send' if msg.get('done') else 'ready-send' if msg.get('status') == 'ready' else 'token-send'
        try:
            self.sock.sendall((line + '\n').encode('utf-8'))
        except OSError as exc:
            self._log(stage, exc, event='error')
            return False
        if msg.get('done'):
            self._log('done')
        elif msg.get('status') == 'ready':
            self._log('ready-sent')
        elif 'text' in msg and (not self._first_token):
            self._first_token = True
            self._log('first-token')
        return True

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError as exc:
            self._log('shutdown', exc, event='error')
        try:
            self.sock.close()
        except OSError as exc:
            self._log('close', exc, event='error')
        self._log('close')

def _send_handshake(ls: LineSocket, status: str, model_fp: int, token_count: int, extra: Optional[dict]=None) -> bool:
    msg = {'version': 1, 'status': status, 'model_fingerprint': int(model_fp) & 18446744073709551615, 'token_count': int(token_count)}
    if extra:
        msg.update(extra)
    return ls.write_line(json.dumps(msg, separators=(',', ':')))

def _send_token(ls: LineSocket, token_id: int, text: str, pos: int) -> bool:
    msg = {'token_id': int(token_id), 'text': text, 'pos': int(pos), 'timestamp_us': int(time.time() * 1000000), 'dropped_count': 0}
    return ls.write_line(json.dumps(msg, separators=(',', ':')))

def _send_done(ls: LineSocket, tokens_generated: int, decode_tok_s: float, stop_reason: int, stopping_word: str='', extra: Optional[Dict[str, Any]]=None, retained_token_count: int=0, retained_prefix_hash: int=0, completion: Optional[Dict[str, Any]]=None) -> bool:
    msg = {'done': True, 'tokens_generated': int(tokens_generated), 'decode_tok_s': float(decode_tok_s), 'stop_reason': int(stop_reason)}
    if completion is not None:
        msg.update(completion)
    if stopping_word:
        msg['stopping_word'] = stopping_word
    if retained_token_count and int(retained_token_count) > 0:
        msg['retained_token_count'] = int(retained_token_count)
        msg['retained_prefix_hash'] = int(retained_prefix_hash) & 9223372036854775807
    if isinstance(extra, dict) and extra:
        msg['swi_reasoning'] = extra
    return ls.write_line(json.dumps(msg, separators=(',', ':')))

def _dsv41_cone_tail_rows():
    raw = os.environ.get('DSV41_PREFILL_CONE_TAIL')
    if raw is None:
        return None
    tail = int(raw)
    if tail < 1:
        raise ValueError(f'DSV41_PREFILL_CONE_TAIL must be >= 1, got {tail}')
    return tail

def _preflight_retained_prefix(session, expected_n: int, expected_hash: int):
    expected_n = int(expected_n)
    expected_hash = int(expected_hash) & 9223372036854775807
    retained_tokens = getattr(session, '_retained_tokens', None) or []
    retained_cache = getattr(session, '_retained_cache', None)
    retained_n = len(retained_tokens)
    actual_hash = _hash_token_ids(retained_tokens[:expected_n]) & 9223372036854775807 if expected_n > 0 and retained_n >= expected_n else 0
    if expected_n <= 0 or expected_hash == 0:
        return (False, retained_n, actual_hash, 'invalid-expected-handle')
    if not retained_cache or retained_n < expected_n:
        return (False, retained_n, actual_hash, 'missing-or-short-retained-cache')
    offsets: List[int] = []

    def _collect_offsets(entry) -> bool:
        children = getattr(entry, 'caches', None)
        if children:
            return all((_collect_offsets(child) for child in children))
        offset = _cache_entry_offset(entry, -1)
        if offset < 0:
            return False
        offsets.append(offset)
        return True
    if not all((_collect_offsets(entry) for entry in retained_cache)) or not offsets:
        return (False, retained_n, actual_hash, 'missing-cache-offset')
    if any((offset != retained_n for offset in offsets)):
        return (False, retained_n, actual_hash, 'misaligned-cache-offset')
    if not _can_trim_cache_to_offset(retained_cache, expected_n):
        return (False, retained_n, actual_hash, 'retained-cache-not-trimmable')
    if actual_hash != expected_hash:
        return (False, retained_n, actual_hash, 'retained-prefix-hash-mismatch')
    return (True, retained_n, actual_hash, 'ok')

def _retained_prefix_ready_or_nack(session, ls, model_fp: int, report_n: int, expected_n: int, expected_hash: int, has_kv_payload: bool=False) -> bool:
    ok, retained_n, actual_hash, reason = _preflight_retained_prefix(session, expected_n, expected_hash)
    if has_kv_payload:
        ok = False
        reason = 'reuse-count-with-kv-payload'
    if not ok:
        sys.stderr.write(f'[mlx-server] reuse-count REJECTED: expected_n={int(expected_n)} retained_n={retained_n} expected_hash=0x{int(expected_hash) & 9223372036854775807:016x} actual_hash=0x{actual_hash:016x} reason={reason}\n')
        sys.stderr.flush()
        _send_handshake(ls, 'retained-prefix-miss', model_fp, report_n)
        return False
    sys.stderr.write(f'[mlx-server] retained-prefix READY: n={int(expected_n)} hash=0x{actual_hash:016x}\n')
    sys.stderr.flush()
    return True

def _should_retain_injected_state(gate_enabled: bool, stop_reason: int, inject_succeeded: bool, ship_image: bool, swi_active: bool) -> bool:
    return bool(gate_enabled and stop_reason != STOP_CANCELLED and inject_succeeded and (not ship_image) and (not swi_active))

def _mtp_accept_prefix(draft_tokens, target_tokens) -> int:
    accepted = 0
    for draft_id, target_id in zip(draft_tokens, target_tokens):
        if int(draft_id) != int(target_id):
            break
        accepted += 1
    return accepted

def _mtp_update_depth_stats(draft_tokens, target_tokens, depth_attempted, depth_matched, first_mismatch_counts) -> int:
    draft_list = list(draft_tokens)
    target_list = list(target_tokens)
    accepted = 0
    first_mismatch = None
    for depth_idx, (draft_id, target_id) in enumerate(zip(draft_list, target_list)):
        if depth_idx < len(depth_attempted):
            depth_attempted[depth_idx] += 1
            if int(draft_id) == int(target_id):
                depth_matched[depth_idx] += 1
        if first_mismatch is None:
            if int(draft_id) == int(target_id):
                accepted += 1
            else:
                first_mismatch = depth_idx
    if first_mismatch is None:
        first_mismatch = len(draft_list)
    if first_mismatch < len(first_mismatch_counts):
        first_mismatch_counts[first_mismatch] += 1
    return accepted

def _mtp_update_lag_stats(draft_tokens, target_tokens, lag_values, lag_attempted, lag_matched) -> None:
    draft_list = [int(token) for token in list(draft_tokens)]
    target_list = [int(token) for token in list(target_tokens)]
    for lag_idx, lag in enumerate(lag_values):
        if lag_idx >= len(lag_attempted) or lag_idx >= len(lag_matched):
            break
        lag_i = int(lag)
        for draft_idx, draft_id in enumerate(draft_list):
            target_idx = draft_idx + lag_i
            if target_idx < 0 or target_idx >= len(target_list):
                continue
            lag_attempted[lag_idx] += 1
            if draft_id == target_list[target_idx]:
                lag_matched[lag_idx] += 1

def _mtp_token_rank_from_logits(mx, logits_row, token_id: int) -> int | None:
    try:
        token = int(token_id)
        row = logits_row
        while int(getattr(row, 'ndim', 1) or 1) > 1:
            row = row[0]
        vocab = int(row.shape[-1])
        if token < 0 or token >= vocab:
            return None
        token_score = row[token]
        rank = mx.sum(row > token_score) + 1
        mx.eval(rank)
        return int(rank.item())
    except Exception:
        return None

def _mtp_debug_rank_stats(mx, draft_logits, verify_logits, draft_tokens, target_tokens):
    q_target_ranks = []
    p_draft_ranks = []
    for idx, (draft_id, target_id) in enumerate(zip(draft_tokens, target_tokens)):
        q_rank = None
        if idx < len(draft_logits):
            try:
                q_rank = _mtp_token_rank_from_logits(mx, draft_logits[idx][0, -1, :], int(target_id))
            except Exception:
                q_rank = None
        p_rank = _mtp_token_rank_from_logits(mx, verify_logits[0, idx, :], int(draft_id))
        q_target_ranks.append(q_rank)
        p_draft_ranks.append(p_rank)
    return (q_target_ranks, p_draft_ranks)

def _mtp_reference_history_tokens(current_id: int, draft_tokens, accepted_count: int, *, current_already_seeded: bool=False):
    accepted = max(0, int(accepted_count))
    if current_already_seeded:
        return [int(t) for t in list(draft_tokens)[:accepted]]
    return [int(current_id), *[int(t) for t in list(draft_tokens)[:accepted]]]

def _mtp_target_tokens_from_logits(mx, verify_logits, count: int, mode: str='batched'):
    n = max(0, int(count))
    if n <= 0:
        return []
    normalized = str(mode or 'batched').lower().replace('-', '_')
    if normalized == 'rowwise':
        return [int(verify_logits[0, idx, :].argmax(-1).item()) for idx in range(n)]
    if normalized != 'batched':
        raise ValueError('MTP target token eval must be batched or rowwise')
    try:
        target_arr = mx.argmax(verify_logits[0, :n, :], axis=-1)
        mx.eval(target_arr)
        if hasattr(target_arr, 'tolist'):
            return [int(token) for token in target_arr.tolist()]
        return [int(target_arr[idx].item()) for idx in range(n)]
    except Exception:
        return [int(verify_logits[0, idx, :].argmax(-1).item()) for idx in range(n)]

class _MlxRequestSampler:

    def __init__(self, mx, profile: Dict[str, Any]):
        self.mx = mx
        self.profile = _merge_sampling_profiles(profile)
        self.temperature = float(self.profile['temp'])
        self.top_k = int(self.profile['top_k'])
        self.top_p = float(self.profile['top_p'])
        self.min_p = float(self.profile['min_p'])
        self.min_keep = max(1, int(self.profile['min_keep']))
        self.seed = int(self.profile.get('seed', 0) or 0)
        if self.seed > 0:
            try:
                mx.random.seed(self.seed)
            except Exception:
                pass

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def _as_2d(self, logits):
        if getattr(logits, 'ndim', 0) == 3:
            return logits[:, -1, :]
        if getattr(logits, 'ndim', 0) == 1:
            return logits[None, :]
        return logits

    def filtered_logits(self, logits):
        mx = self.mx
        logits_2d = self._as_2d(logits)
        if self.greedy:
            return logits_2d
        scaled = logits_2d / self.temperature
        vocab = int(scaled.shape[-1])
        if vocab <= 0:
            return scaled
        sorted_idx = mx.argsort(-scaled, axis=-1)
        sorted_logits = mx.take_along_axis(scaled, sorted_idx, axis=-1)
        rank = mx.arange(vocab)[None, :]
        mask = rank < 0
        keep = min(vocab, max(1, self.min_keep))
        if 0 < self.top_k < vocab:
            mask = mx.logical_or(mask, rank >= max(keep, self.top_k))
        if self.min_p > 0.0:
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            min_threshold = sorted_probs[:, :1] * self.min_p
            min_p_mask = mx.logical_and(sorted_probs < min_threshold, rank >= keep)
            mask = mx.logical_or(mask, min_p_mask)
        if 0.0 < self.top_p < 1.0:
            topk_logits = mx.where(mask, -float('inf'), sorted_logits)
            sorted_probs = mx.softmax(topk_logits, axis=-1)
            cdf = mx.cumsum(sorted_probs, axis=-1)
            prev_cdf = mx.concatenate([mx.zeros_like(cdf[:, :1]), cdf[:, :-1]], axis=-1)
            top_p_mask = mx.logical_and(prev_cdf >= self.top_p, rank >= keep)
            mask = mx.logical_or(mask, top_p_mask)
        sorted_filtered = mx.where(mask, -float('inf'), sorted_logits)
        filtered = mx.full(scaled.shape, -float('inf'))
        return mx.put_along_axis(filtered, sorted_idx, sorted_filtered, axis=-1)

    def probs(self, logits):
        return self.mx.softmax(self.filtered_logits(logits), axis=-1)

    def sample_array(self, logits):
        mx = self.mx
        logits_2d = self._as_2d(logits)
        if self.greedy:
            return mx.argmax(logits_2d, axis=-1).astype(mx.int32)
        return mx.random.categorical(self.filtered_logits(logits_2d), axis=-1).astype(mx.int32)

    def __call__(self, logits):
        return self.sample_array(logits)

    def sample_token(self, logits) -> int:
        token = self.sample_array(logits)
        self.mx.eval(token)
        return int(token.reshape(-1)[-1].item())

    def sample_from_probs(self, probs, fallback_probs=None) -> int:
        mx = self.mx
        mass = mx.sum(probs, axis=-1, keepdims=True)
        mx.eval(mass)
        if float(mass.reshape(-1)[-1].item()) <= 1e-20:
            if fallback_probs is not None:
                return self.sample_from_probs(fallback_probs)
            token = mx.argmax(probs, axis=-1).astype(mx.int32)
            mx.eval(token)
            return int(token.reshape(-1)[-1].item())
        normalized = probs / mass
        logits = mx.where(normalized > 0, mx.log(normalized), -float('inf'))
        token = mx.random.categorical(logits, axis=-1).astype(mx.int32)
        mx.eval(token)
        return int(token.reshape(-1)[-1].item())

def _mtp_speculative_sampling_decision(mx, sampler: _MlxRequestSampler, draft_tokens, draft_logits, verify_logits, target_token_eval: str) -> Dict[str, Any]:
    draft_list = [int(t) for t in list(draft_tokens)]
    target_greedy = _mtp_target_tokens_from_logits(mx, verify_logits, len(draft_list) + 1, target_token_eval)
    if sampler.greedy:
        accepted = _mtp_accept_prefix(draft_list, target_greedy[:len(draft_list)])
        return {'accepted_count': accepted, 'next_token': int(target_greedy[accepted]), 'target_greedy': target_greedy, 'accepted_flags': [idx < accepted for idx in range(len(draft_list))], 'accept_probs': [1.0 if idx < accepted else 0.0 for idx in range(len(draft_list))], 'residual_sampled': accepted < len(draft_list)}
    accepted = 0
    accepted_flags: List[bool] = []
    accept_probs: List[float] = []
    for idx, draft_id in enumerate(draft_list):
        p_probs = sampler.probs(verify_logits[0, idx, :])
        q_logits = draft_logits[idx] if idx < len(draft_logits) else verify_logits[0, idx, :]
        q_probs = sampler.probs(q_logits)
        mx.eval(p_probs, q_probs)
        p_tok = float(p_probs[0, int(draft_id)].item())
        q_tok = float(q_probs[0, int(draft_id)].item())
        accept_prob = 1.0 if q_tok <= 1e-20 else min(1.0, max(0.0, p_tok / q_tok))
        accept_probs.append(accept_prob)
        draw = float(mx.random.uniform().item())
        if draw <= accept_prob:
            accepted += 1
            accepted_flags.append(True)
            continue
        residual = mx.maximum(p_probs - q_probs, 0.0)
        next_token = sampler.sample_from_probs(residual, fallback_probs=p_probs)
        accepted_flags.append(False)
        accepted_flags.extend((False for _ in draft_list[idx + 1:]))
        accept_probs.extend((0.0 for _ in draft_list[idx + 1:]))
        return {'accepted_count': accepted, 'next_token': int(next_token), 'target_greedy': target_greedy, 'accepted_flags': accepted_flags, 'accept_probs': accept_probs, 'residual_sampled': True}
    bonus_probs = sampler.probs(verify_logits[0, len(draft_list), :])
    next_token = sampler.sample_from_probs(bonus_probs)
    return {'accepted_count': accepted, 'next_token': int(next_token), 'target_greedy': target_greedy, 'accepted_flags': accepted_flags, 'accept_probs': accept_probs, 'residual_sampled': False}

def _mtp_forward_verify_token_ids(mx, ids_arr, forward_token_ids, mode: str):
    logits, hidden, _topk = _mtp_forward_verify_token_ids_with_topk(mx, ids_arr, forward_token_ids, mode)
    return (logits, hidden)

def _target_forward_result_with_topk(result):
    if isinstance(result, tuple):
        if len(result) >= 3:
            return (result[0], result[1], result[2])
        if len(result) >= 2:
            return (result[0], result[1], None)
    raise ValueError('target forward must return (logits, hidden[, shared_topk])')

def _mtp_forward_verify_token_ids_with_topk(mx, ids_arr, forward_token_ids, mode: str):
    if str(mode or 'block').lower().replace('-', '_') != 'rowwise':
        return _target_forward_result_with_topk(forward_token_ids(ids_arr))
    total = int(ids_arr.shape[1])
    logits_rows = []
    hidden_rows = []
    topk_rows = []
    for idx in range(total):
        logits, hidden, topk = _target_forward_result_with_topk(forward_token_ids(ids_arr[:, idx:idx + 1]))
        logits_rows.append(logits)
        if hidden is not None:
            hidden_rows.append(hidden)
        if topk is not None:
            topk_rows.append(topk)
    logits = mx.concatenate(logits_rows, axis=1)
    hidden = mx.concatenate(hidden_rows, axis=1) if hidden_rows else None
    topk = mx.concatenate(topk_rows, axis=2) if topk_rows else None
    return (logits, hidden, topk)

def _mtp_topk_slice(topk, start: int | None=None, stop: int | None=None):
    if topk is None:
        return None
    return topk[:, :, start:stop, :]

def _mtp_topk_concat(mx, pieces):
    usable = [piece for piece in pieces if piece is not None]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    return mx.concatenate(usable, axis=2)

def _mtp_prepare_reference_topk(mtp_drafter, current_prev_topk) -> bool:
    if bool(getattr(mtp_drafter, 'has_seed', False)):
        return True
    return bool(mtp_drafter.seed_shared_topk(current_prev_topk))

class MLXSession:
    _LCP_REUSE_MIN = int(os.environ.get('LLAMA_MLX_LCP_REUSE_MIN', '64'))

    def __init__(self, model_path: str, mtp_mode: str='off', mtp_sidecar: str='auto', mtp_max_draft: int=1, mtp_quantized_sidecar_policy: str='warn', mtp_debug_windows: int=0, mtp_compute_dtype: str='float16', mtp_recurrent_state: str='raw', mtp_schedule: str='reference', mtp_prompt_history: str='committed', mtp_cache_commit: str='replay', mtp_verify_profile: str='safe', mtp_target_token_eval: str='auto', mtp_verify_forward: str='auto', mtp_verify_hidden_eval: str='auto', mtp_history_materialization: str='auto', mtp_share_topk_source: str='internal', mtp_match_trunk_precision: str='auto', mtp_adaptive_depth: str='off', mtp_token_budget: int=0, adapters: Optional[List[str]]=None, token_sidecars: Optional[List[str]]=None, token_sidecar_scale: float=1.0, sampling_profile: Optional[Dict[str, Any]]=None):
        self.model_path = model_path
        self._sampling_profile = _merge_sampling_profiles(sampling_profile)
        self._mtp_mode = str(mtp_mode or 'off').lower()
        self._mtp_sidecar_spec = str(mtp_sidecar or 'auto')
        self._mtp_max_draft = int(mtp_max_draft)
        self._mtp_quantized_sidecar_policy = str(mtp_quantized_sidecar_policy or 'warn').lower()
        self._mtp_debug_windows = max(0, int(mtp_debug_windows))
        self._mtp_compute_dtype = str(mtp_compute_dtype or 'float16').lower()
        self._mtp_recurrent_state = str(mtp_recurrent_state or 'raw').lower()
        self._mtp_schedule = str(mtp_schedule or 'reference').lower().replace('-', '_')
        self._mtp_prompt_history = str(mtp_prompt_history or 'committed').lower().replace('-', '_')
        self._mtp_cache_commit = str(mtp_cache_commit or 'replay').lower().replace('-', '_')
        self._mtp_verify_profile = str(mtp_verify_profile or 'safe').lower().replace('-', '_')
        self._mtp_target_token_eval = self._resolve_mtp_profile_value(mtp_target_token_eval, safe_value='batched', mtplx_value='batched')
        self._mtp_verify_forward = self._resolve_mtp_profile_value(mtp_verify_forward, safe_value='block', mtplx_value='block')
        self._mtp_verify_hidden_eval = self._resolve_mtp_profile_value(mtp_verify_hidden_eval, safe_value='eager', mtplx_value='defer')
        self._mtp_history_materialization = self._resolve_mtp_profile_value(mtp_history_materialization, safe_value='eager', mtplx_value='lazy')
        self._mtp_share_topk_source = str(mtp_share_topk_source or 'internal').lower().replace('-', '_')
        self._mtp_match_trunk_precision = str(mtp_match_trunk_precision or 'auto').lower().replace('-', '_')
        self._mtp_adaptive_depth = str(mtp_adaptive_depth or 'off').lower()
        self._mtp_token_budget = max(0, int(mtp_token_budget or 0))
        self._adapter_paths = [str(path) for path in adapters or [] if str(path).strip()]
        self._token_sidecar_paths = [str(path) for path in token_sidecars or [] if str(path).strip()]
        self._token_sidecar_scale = float(token_sidecar_scale)
        self._mtp_info = None
        self._mtp_weights = None
        self._mtp_drafter = None
        self.model = None
        self.tokenizer = None
        self._use_vlm = False
        self._retained_cache = None
        self._retained_tokens: List[int] = []
        self._prompt_store = None
        self._swi_embed = None
        self._swi_last_stats = None

    def _resolve_mtp_profile_value(self, value, *, safe_value: str, mtplx_value: str) -> str:
        normalized = str(value or 'auto').lower().replace('-', '_')
        if normalized == 'auto':
            return mtplx_value if self._mtp_verify_profile == 'mtplx' else safe_value
        return normalized

    def load(self) -> Tuple[float, int]:
        import mlx.core as mx
        if mx.metal.is_available():
            max_rec = mx.device_info().get('max_recommended_working_set_size', 0)
            if max_rec <= 0:
                _env_gb = os.environ.get('LLAMA_MLX_WIRED_LIMIT_GB', '').strip()
                if _env_gb:
                    max_rec = int(float(_env_gb) * 1024 ** 3)
                else:
                    try:
                        import subprocess as _sp
                        _mb = int(_sp.check_output(['sysctl', '-n', 'iogpu.wired_limit_mb'], text=True).strip() or '0')
                        if _mb > 0:
                            max_rec = max(0, (_mb - 16 * 1024) * 1024 ** 2)
                    except Exception as _we:
                        sys.stderr.write(f'[mlx-server] wired limit fallback unavailable: {_we}' + chr(10))
            if max_rec > 0:
                mx.set_wired_limit(max_rec)
                sys.stderr.write(f'[mlx-server] wired_limit set to {max_rec // 1024 ** 3} GB\n')
                sys.stderr.flush()
        t0 = time.time()
        config_path = Path(self.model_path) / 'config.json'
        raw_config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        is_dsv41 = _is_dsv41(raw_config)
        if is_dsv41:
            self.model, self.tokenizer = _load_dsv41(self.model_path, raw_config)
            self._use_vlm = True
            self._dsv41_thinking_mode = 'chat'
            self._dsv41_requests = 0
            self._dsv41_completion = None
            import atexit
            import signal
            atexit.register(_dsv41_save_hot, self)

            def terminate(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminate)
        prefer_text_lm_requested = os.environ.get('LLAMA_MLX_TEXT_LM_LOAD', '1') not in ('', '0')
        prefer_text_lm = not is_dsv41 and prefer_text_lm_requested and _has_vlm_text_config(self.model_path)
        if prefer_text_lm_requested and (not prefer_text_lm):
            sys.stderr.write('[mlx-server] text MLX-LM loader skipped: checkpoint has no text_config\n')
            sys.stderr.flush()
        if prefer_text_lm:
            try:
                self.model, self.tokenizer, _text_config, _vision_weights = _load_kimi_text_lm_from_vlm_checkpoint(self.model_path, lazy=True, return_vision_weights=True)
                mx.eval(self.model.parameters())
                self._use_vlm = False
                self._load_kimi_vision(raw_config, _vision_weights)
                del _vision_weights
                sys.stderr.write('[mlx-server] loaded Kimi text model via mlx_lm.deepseek_v3 (latent MLA cache)\n')
                sys.stderr.flush()
            except Exception as e_text:
                strict_text = os.environ.get('LLAMA_MLX_TEXT_LM_STRICT', '1') not in ('', '0')
                if strict_text:
                    raise RuntimeError(f'Failed Kimi text MLX-LM load: {e_text}')
                sys.stderr.write(f'[mlx-server] Kimi text MLX-LM load failed: {e_text}; falling back to mlx_vlm full-KV cache path\n')
                sys.stderr.flush()
                prefer_text_lm = False
        if not prefer_text_lm and (not is_dsv41):
            try:
                from mlx_vlm.utils import load as vlm_load
                self.model, self.processor = vlm_load(self.model_path, lazy=True, trust_remote_code=True)
                mx.eval(self.model.parameters())
                self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
                self._use_vlm = True
            except Exception as e_vlm:
                try:
                    from mlx_lm import load as lm_load
                    self.model, self.tokenizer = lm_load(self.model_path, lazy=True)
                    mx.eval(self.model.parameters())
                    self._use_vlm = False
                except Exception as e_lm:
                    raise RuntimeError(f'Failed to load model via mlx_vlm ({e_vlm}) and mlx_lm ({e_lm})')
        if self.tokenizer is None or not hasattr(self.tokenizer, 'encode'):
            tok_json = os.path.join(self.model_path, 'tokenizer.json')
            if os.path.exists(tok_json):
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json, clean_up_tokenization_spaces=False)
        lora_branches = attach_mlx_loras(self.model, self._adapter_paths)
        token_experts = attach_token_sidecars(self.model, self._token_sidecar_paths, scale=self._token_sidecar_scale)
        if lora_branches or token_experts:
            if is_dsv41:
                _dsv41_materialize(self.model, mx)
            else:
                mx.eval(self.model.parameters())
            sys.stderr.write(f'[mlx-server] attached adapters: lora_branches={lora_branches} token_experts={token_experts} token_sidecar_scale={self._token_sidecar_scale:g}\n')
            sys.stderr.flush()
        name = os.path.basename(os.path.normpath(self.model_path)).encode('utf-8')
        fp = 14695981039346656037
        for b in name:
            fp = (fp ^ b) * 1099511628211
            fp &= 18446744073709551615
        self._init_prompt_store()
        self._init_mtp_sidecar()
        return (time.time() - t0, fp)

    def _load_kimi_vision(self, raw_config, vision_weights):
        self._kimi_vision = None
        self._kimi_vision_error = 'checkpoint has no vision tower'
        self._kimi_vision_prompt = None
        if not raw_config.get('vision_config') or not vision_weights:
            return
        if os.environ.get('LLAMA_MLX_KIMI_VISION', '1') in ('', '0'):
            self._kimi_vision_error = 'image input disabled (LLAMA_MLX_KIMI_VISION=0)'
            return
        try:
            from kimi_vision import KimiVision
            t0 = time.time()
            self._kimi_vision = KimiVision(raw_config, vision_weights)
            self._kimi_vision_error = None
            sys.stderr.write(f'[mlx-server] Kimi vision tower loaded in {time.time() - t0:.1f}s ({len(vision_weights)} tensors); images prefill on the cached prompt\n')
        except Exception as e:
            self._kimi_vision_error = f'vision tower failed to load: {e}'
            sys.stderr.write(f'[mlx-server] {self._kimi_vision_error}\n')
        sys.stderr.flush()

    def kimi_vision_prompt(self, prompt_text, raw_image):
        from kimi_vision import build_vision_prompt, split_images
        if getattr(self, '_kimi_vision', None) is None:
            raise ValueError('this Kimi receiver cannot take images: ' + str(getattr(self, '_kimi_vision_error', None) or 'no vision tower'))
        text_ids = list(self.tokenizer.encode(prompt_text))
        vp = build_vision_prompt(self._kimi_vision, text_ids, split_images(raw_image))
        self._kimi_vision_prompt = vp
        sys.stderr.write(f'[mlx-server] image prompt: {len(vp.images)} image(s), {sum((c for _, c, _, _ in vp.images))} image cells, {len(vp.token_ids)} positions\n')
        sys.stderr.flush()
        return vp.token_ids

    def _init_mtp_sidecar(self):
        if _is_dsv41(self.model):
            self._mtp_mode = 'off'
            return
        if self._mtp_mode in ('', 'off', 'none', 'disabled'):
            return
        if self._mtp_mode not in ('validate', 'load', 'draft', 'speculative'):
            raise ValueError(f'unsupported --mtp-mode {self._mtp_mode!r}; expected off, validate, load, draft, or speculative')
        if self._mtp_schedule not in ('reference', 'target_known_next'):
            raise ValueError(f'unsupported --mtp-schedule {self._mtp_schedule!r}; expected reference or target_known_next')
        if self._mtp_verify_profile not in ('safe', 'mtplx'):
            raise ValueError(f'unsupported --mtp-verify-profile {self._mtp_verify_profile!r}; expected safe or mtplx')
        if self._mtp_target_token_eval not in ('batched', 'rowwise'):
            raise ValueError(f'unsupported --mtp-target-token-eval {self._mtp_target_token_eval!r}; expected batched or rowwise')
        if self._mtp_verify_forward not in ('block', 'rowwise'):
            raise ValueError(f'unsupported --mtp-verify-forward {self._mtp_verify_forward!r}; expected block or rowwise')
        if self._mtp_verify_hidden_eval not in ('eager', 'defer'):
            raise ValueError(f'unsupported --mtp-verify-hidden-eval {self._mtp_verify_hidden_eval!r}; expected eager or defer')
        if self._mtp_history_materialization not in ('eager', 'lazy'):
            raise ValueError(f'unsupported --mtp-history-materialization {self._mtp_history_materialization!r}; expected eager or lazy')
        if self._mtp_prompt_history not in ('committed', 'shifted_bonus'):
            raise ValueError(f'unsupported --mtp-prompt-history {self._mtp_prompt_history!r}; expected committed or shifted_bonus')
        if self._mtp_cache_commit not in ('replay', 'native'):
            raise ValueError(f'unsupported --mtp-cache-commit {self._mtp_cache_commit!r}; expected replay or native')
        if self._mtp_share_topk_source not in ('target', 'internal', 'off'):
            raise ValueError(f'unsupported --mtp-share-topk-source {self._mtp_share_topk_source!r}; expected target, internal, or off')
        if self._mtp_schedule == 'target_known_next' and self._mtp_prompt_history == 'shifted_bonus':
            raise ValueError('--mtp-prompt-history shifted_bonus is the native reference scheduler path; use --mtp-schedule reference')
        if self._mtp_max_draft < 1:
            raise ValueError('--mtp-max-draft must be >= 1')
        if self._mtp_quantized_sidecar_policy not in ('allow', 'warn', 'reject'):
            raise ValueError('--mtp-quantized-sidecar-policy must be allow, warn, or reject')
        sidecar_dir = resolve_sidecar_dir(self.model_path, self._mtp_sidecar_spec)
        if sidecar_dir is None:
            raise ValueError('--mtp-mode requires --mtp-sidecar auto|PATH')
        info = validate_sidecar(sidecar_dir)
        self._mtp_info = info
        sys.stderr.write(f'[mlx-server] MTP sidecar validated ({self._mtp_mode}): {info.summary()}\n')
        for warning in info.warnings:
            sys.stderr.write(f'[mlx-server] MTP sidecar warning: {warning}\n')
        if is_quantized_mtp_sidecar(info):
            msg = 'MTP sidecar is quantized; GLM-5.2 MXFP4 MTP acceptance is a known-degraded experimental path. Use the non-quantized FP16-runtime sidecar for correctness/performance baselines.'
            if self._mtp_quantized_sidecar_policy == 'reject':
                raise ValueError(f'{msg} Override with --mtp-quantized-sidecar-policy allow only for explicit quantized-MTP experiments.')
            if self._mtp_quantized_sidecar_policy == 'warn':
                sys.stderr.write(f'[mlx-server] MTP sidecar warning: {msg}\n')
        sys.stderr.flush()
        if self._mtp_max_draft > max(1, info.num_nextn_predict_layers) and (not info.index_share_for_mtp_iteration):
            raise ValueError(f'--mtp-max-draft {self._mtp_max_draft} exceeds sidecar num_nextn_predict_layers={info.num_nextn_predict_layers}')
        if self._mtp_max_draft > max(1, info.num_nextn_predict_layers):
            sys.stderr.write(f'[mlx-server] MTP iterative draft window enabled: max_draft={self._mtp_max_draft} via shared GLM MTP layer (sidecar layers={info.num_nextn_predict_layers})\n')
            sys.stderr.flush()
        if self._mtp_mode in ('load', 'draft', 'speculative'):
            import mlx.core as mx
            self._mtp_weights = mx.load(str(info.model_path))
            sys.stderr.write(f'[mlx-server] MTP sidecar arrays loaded lazily: {len(self._mtp_weights)} tensors from {info.model_path}\n')
            sys.stderr.flush()
        if self._mtp_mode in ('draft', 'speculative'):
            from glm52_mtp_drafter import Glm52MtpDrafter
            self._mtp_drafter = Glm52MtpDrafter(self.model, info, self._mtp_weights, compute_dtype=self._mtp_compute_dtype, recurrent_state=self._mtp_recurrent_state, history_materialization=self._mtp_history_materialization, share_topk_source=self._mtp_share_topk_source, match_trunk_precision=self._mtp_match_trunk_precision)
            mode_msg = 'target-verified diagnostics enabled; emitted tokens remain target-model tokens' if self._mtp_mode == 'draft' else 'speculative block verification enabled'
            sys.stderr.write(f'[mlx-server] MTP drafter ready: {mode_msg}\n')
            sys.stderr.write(f'[mlx-server] MTP compute dtype: {self._mtp_drafter.compute_dtype_name}\n')
            sys.stderr.write(f'[mlx-server] MTP recurrent state: {self._mtp_drafter.recurrent_state}\n')
            sys.stderr.write(f'[mlx-server] MTP schedule: {self._mtp_schedule}\n')
            sys.stderr.write(f'[mlx-server] MTP prompt history: {self._mtp_prompt_history}\n')
            sys.stderr.write(f'[mlx-server] MTP cache commit: {self._mtp_cache_commit}\n')
            sys.stderr.write(f'[mlx-server] MTP share top-k source: {self._mtp_share_topk_source}\n')
            sys.stderr.write(f'[mlx-server] MTP verify profile: {self._mtp_verify_profile} (target_token_eval={self._mtp_target_token_eval}, verify_forward={self._mtp_verify_forward}, verify_hidden_eval={self._mtp_verify_hidden_eval}, history_materialization={self._mtp_history_materialization})\n')
            sys.stderr.write(f"[mlx-server] MTP token budget: {(self._mtp_token_budget if self._mtp_token_budget > 0 else 'unlimited')}\n")
            sys.stderr.write(f'[mlx-server] MTP embed scale: {self._mtp_drafter.embed_scale:g}\n')
            sys.stderr.flush()

    def _init_prompt_store(self):
        cache_dir = os.environ.get('LLAMA_MLX_PROMPT_CACHE_DIR', '').strip()
        if not cache_dir:
            self._prompt_store = None
            return
        try:
            if _is_dsv41(self.model):
                self._make_cache()
            from prompt_cache_store import PromptCacheStore, model_key_for
            quant_cfg = self._mlx_quant_config()
            try:
                import mlx_lm as _mlxlm
                mlx_lm_ver = getattr(_mlxlm, '__version__', '')
            except Exception:
                mlx_lm_ver = ''
            mkey = model_key_for(self.model_path, quant_cfg, mlx_lm_ver)
            hot = int(os.environ.get('LLAMA_MLX_PROMPT_CACHE_HOT', '8'))
            max_gb = float(os.environ.get('LLAMA_MLX_PROMPT_CACHE_MAX_GB', '500'))
            self._prompt_store = PromptCacheStore(cache_dir, mkey, hot_size=hot, max_gb=max_gb, lcp_reuse_min=self._LCP_REUSE_MIN)
            sys.stderr.write(f'[mlx-server] prompt-cache store enabled: dir={cache_dir} model_key={mkey} hot={hot} max_gb={max_gb}\n')
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f'[mlx-server] prompt-cache store disabled (init error): {e}\n')
            sys.stderr.flush()
            self._prompt_store = None

    def _mlx_quant_config(self):
        try:
            cfg_path = os.path.join(self.model_path, 'config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                return cfg.get('quantization', {}) or {}
        except Exception:
            pass
        return {}

    def _make_cache(self):
        if _is_dsv41(self.model):
            return self.model.make_cache()
        if self._use_vlm:
            from mlx_vlm.models.cache import make_prompt_cache
        else:
            from mlx_lm.models.cache import make_prompt_cache
        lang = getattr(self.model, 'language_model', self.model)
        return make_prompt_cache(lang)

    def _fresh_or_store(self, prompt_token_ids, text_only, image_hashes=None):
        if self._prompt_store is None or not self._prompt_store.enabled:
            return (self._make_cache(), 0)
        if not text_only and (not image_hashes):
            return (self._make_cache(), 0)
        try:
            res = self._prompt_store.lookup(list(prompt_token_ids), image_hashes=image_hashes)
        except Exception as e:
            sys.stderr.write(f'[mlx-server] prompt-cache lookup failed: {e}\n')
            sys.stderr.flush()
            res = None
        if res is None:
            return (self._make_cache(), 0)
        cache, lcp, _entry = res
        image_cells = sum((int(p[1]) for p in getattr(_entry, 'image_positions', ()) or ()))
        target = int(lcp) + (image_cells if image_hashes else 0)
        if not 0 < lcp < len(prompt_token_ids) or _cache_offset(cache) != target:
            sys.stderr.write('[mlx-server] prompt-cache DISK MISS: cache offsets disagree with prefix\n')
            return (self._make_cache(), 0)
        if not image_hashes:
            self._retained_cache = cache
            self._retained_tokens = list(prompt_token_ids[:lcp])
        sys.stderr.write(f"[mlx-server] prompt-cache DISK HIT{(' (image)' if image_hashes else '')}: reusing {lcp} tok from disk (skipping that much prefill), suffix={len(prompt_token_ids) - lcp} tok\n")
        sys.stderr.flush()
        return (cache, lcp)

    def prefix_query(self, tokens: List[int]) -> int:
        return prefix_query(self, tokens)

    def _cache_with_reuse(self, prompt_token_ids, kv_state_bytes, text_only=True, image_hashes=None, reuse_retained_count=0, delta_base=0):
        retained_n = len(self._retained_tokens) if self._retained_tokens else 0
        new_n = len(prompt_token_ids)
        if kv_state_bytes and delta_base > 0:
            return select_delta_base(self, prompt_token_ids, delta_base)
        if kv_state_bytes:
            if self._retained_cache is not None:
                sys.stderr.write(f'[mlx-server] cache-reset (KV-inject): dropping retained cache of {retained_n} tok before injecting {len(kv_state_bytes)} bytes of RTX KV\n')
                sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []
            return (self._make_cache(), 0)
        if image_hashes:
            return self._fresh_or_store(prompt_token_ids, text_only, image_hashes)
        if reuse_retained_count and int(reuse_retained_count) > 0:
            n = int(reuse_retained_count)
            if self._retained_cache and self._retained_tokens and (n <= retained_n):
                extra = _cache_offset(self._retained_cache) - n
                if not _trim_cache_to_offset(self._retained_cache, n):
                    self._retained_cache = None
                    self._retained_tokens = []
                    raise RuntimeError('retained-prefix-miss after preflight: cache did not trim exactly')
                self._retained_tokens = list(self._retained_tokens[:n])
                sys.stderr.write(f'[mlx-server] reuse-count HONORED: reusing {n} retained tok as prefix (trimmed {extra}), prefilling {new_n} delta tok as continuation (LCP bypass)\n')
                sys.stderr.flush()
                return (self._retained_cache, 0)
            raise RuntimeError(f'retained-prefix-miss after preflight (N={n} retained={retained_n})')
        if not self._retained_cache or not self._retained_tokens:
            sys.stderr.write(f'[mlx-server] cache-fresh: no retained tokens; consulting disk store then prefilling remainder of {new_n} tok\n')
            sys.stderr.flush()
            return self._fresh_or_store(prompt_token_ids, text_only)
        cache_n = _cache_offset(self._retained_cache)
        lim = min(new_n, retained_n, cache_n)
        lcp = 0
        ret = self._retained_tokens
        new = prompt_token_ids
        for i in range(lim):
            if ret[i] != new[i]:
                break
            lcp = i + 1
        diverge_at = lcp if lcp < min(retained_n, new_n) else -1
        retained_drop = retained_n - lcp
        if lcp < self._LCP_REUSE_MIN:
            sys.stderr.write(f'[mlx-server] cache-reset: LCP={lcp} < min={self._LCP_REUSE_MIN} (new_n={new_n} retained_n={retained_n} diverge_at={diverge_at}); dropping retained cache, fresh prefill of all {new_n} tok\n')
            sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []
            return self._fresh_or_store(prompt_token_ids, text_only)
        if lcp >= new_n:
            sys.stderr.write(f'[mlx-server] cache-reuse: LCP={lcp} covers the whole new prompt (new_n={new_n}); rolling back one token and re-prefilling it [KV_DELTA_REPEAT]\n')
            sys.stderr.flush()
            lcp = new_n - 1
            diverge_at = -1
            retained_drop = retained_n - lcp
        if not _trim_cache_to_offset(self._retained_cache, lcp):
            sys.stderr.write('[mlx-server] cache-reuse: cache did not trim exactly; bailing\n')
            sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []
            return (self._make_cache(), 0)
        if retained_drop > 0:
            sys.stderr.write(f'[mlx-server] cache-rollback: trimmed retained KV from {retained_n} to {lcp} tok (diverge_at={diverge_at}); will re-prefill {new_n - lcp} new tok\n')
        else:
            sys.stderr.write(f'[mlx-server] cache-reuse: LCP={lcp}/{new_n} tok (saving {lcp} tok of re-prefill)\n')
        sys.stderr.flush()
        self._retained_tokens = list(self._retained_tokens[:lcp])
        return (self._retained_cache, lcp)

    def _save_retained_state(self, cache, prompt_token_ids, generated_ids):
        try:
            toks = list(prompt_token_ids) + list(generated_ids)
            offsets = _cache_leaf_offsets(cache)
            if not offsets or min(offsets) < 0 or len(set(offsets)) != 1:
                raise RuntimeError(f'cache child offsets are missing/misaligned: {offsets}')
            off = _cache_offset(cache)
            if off < len(toks):
                toks = toks[:off]
            if not _trim_cache_to_offset(cache, len(toks)):
                raise RuntimeError(f'cache offset {off} cannot normalize to {len(toks)} committed tokens')
            normalized_offsets = _cache_leaf_offsets(cache)
            if not normalized_offsets or len(set(normalized_offsets)) != 1 or normalized_offsets[0] != len(toks):
                raise RuntimeError(f'cache/token normalization failed: offsets={normalized_offsets} tokens={len(toks)}')
            self._retained_cache = cache
            self._retained_tokens = toks
        except Exception as e:
            sys.stderr.write(f'[mlx-server] cache retention save failed: {e}\n')
            sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []

    def _eos_ids(self) -> set:
        if _is_dsv41(self.model):
            return {self.tokenizer.eos_token_id}
        eos: set = set()
        cfg = getattr(self.model, 'config', getattr(self.model, 'args', None))
        text_cfg = getattr(cfg, 'text_config', cfg)
        eid = getattr(text_cfg, 'eos_token_id', None)
        if isinstance(eid, list):
            eos.update((int(x) for x in eid))
        elif eid is not None:
            eos.add(int(eid))
        tok_eos = getattr(self.tokenizer, 'eos_token_id', None)
        if tok_eos is not None:
            eos.add(int(tok_eos))
        for _marker in ('<|user|>', '<|observation|>'):
            try:
                _ids = self.tokenizer.encode(_marker, add_special_tokens=False)
                if isinstance(_ids, (list, tuple)) and len(_ids) == 1:
                    eos.add(int(_ids[0]))
            except Exception:
                pass
        return eos

    def _swi_resolve_embed_tokens(self):
        lang = getattr(self.model, 'language_model', self.model)
        for obj in (lang, getattr(lang, 'model', None), getattr(getattr(lang, 'model', None), 'model', None)):
            if obj is None:
                continue
            emb = getattr(obj, 'embed_tokens', None)
            if emb is not None:
                return emb
        return None

    def _swi_encode_marker_ids(self, text: str) -> List[int]:
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return [int(x) for x in ids]
        except Exception:
            return []

    def _swi_get_embed(self, mx):
        if self._swi_embed == 'FAILED':
            return None
        if self._swi_embed is not None:
            return self._swi_embed
        try:
            emb_tok = self._swi_resolve_embed_tokens()
            if emb_tok is None:
                raise RuntimeError('cannot resolve embed_tokens on the language model')
            w = getattr(emb_tok, 'weight', None)
            if w is None:
                raise RuntimeError('embed_tokens has no .weight')
            scales = getattr(emb_tok, 'scales', None)
            if scales is None:
                E = w.astype(mx.bfloat16)
            else:
                biases = getattr(emb_tok, 'biases', None)
                group_size = int(getattr(emb_tok, 'group_size', 64))
                bits = int(getattr(emb_tok, 'bits', 4))
                mode = getattr(emb_tok, 'mode', None)
                E = None
                for kwargs in ({'group_size': group_size, 'bits': bits, **({'mode': mode} if mode is not None else {})}, {'group_size': group_size, 'bits': bits}):
                    try:
                        E = mx.dequantize(w, scales, biases, **kwargs)
                        break
                    except TypeError:
                        continue
                if E is None:
                    raise RuntimeError('mx.dequantize signature mismatch for embed_tokens')
                E = E.astype(mx.bfloat16)
            mx.eval(E)
            vocab, hidden = (int(E.shape[0]), int(E.shape[1]))

            def _row(text: str):
                ids = self._swi_encode_marker_ids(text)
                if not ids:
                    return None
                last = ids[-1]
                if last < 0 or last >= vocab:
                    return None
                return E[last]
            think = _row('<think>')
            endthink = _row('</think>')
            nl = _row('\n')
            endthink_ids = self._swi_encode_marker_ids('</think>')
            convergence_ids = self._swi_encode_marker_ids('</think>')
            termination_ids = self._swi_encode_marker_ids('</think>\n\nThe final answer is')
            has_specials = endthink is not None and nl is not None
            self._swi_embed = {'E': E, 'vocab': vocab, 'hidden': hidden, 'think': think, 'endthink': endthink, 'nl': nl, 'endthink_ids': endthink_ids, 'convergence_ids': convergence_ids, 'termination_ids': termination_ids, 'has_specials': has_specials}
            sys.stderr.write(f'[mlx-server] SwiReasoning: built E [{vocab} x {hidden}] bf16 (~{vocab * hidden * 2 / 1024 ** 3:.2f} GB); markers think={think is not None} endthink={endthink is not None} nl={nl is not None} endthink_ids={endthink_ids}\n')
            sys.stderr.flush()
            return self._swi_embed
        except Exception as e:
            import traceback
            sys.stderr.write(f'[mlx-server] SwiReasoning: E-matrix build FAILED ({e}); falling back to normal decode (swi disabled this session)\n')
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self._swi_embed = 'FAILED'
            return None

    def _dsv41_encode_messages(self, messages, reasoning_effort=None):
        encoding = _dsv41_encoding()
        rendered, media = encoding.encode_messages(messages, thinking_mode=self._dsv41_thinking_mode, reasoning_effort=reasoning_effort, return_multi_modal_data=True)
        images = media.get('images', [])
        self._dsv41_vision_prompt = None
        if not images:
            return self.tokenizer.encode(rendered)
        ids, _, pixels = encoding.prepare_vision_inputs(rendered, images, self.tokenizer.backend_tokenizer, self.model.config)
        self._dsv41_vision_prompt = (list(ids), pixels)
        sys.stderr.write(f'[DSV41] vision prepared: images={len(images)} positions={len(ids)}\n')
        return ids

    def encode_prompt(self, prompt_text: str, fallback_token_ids: List[int]) -> List[int]:
        if _is_dsv41(self.model) and prompt_text:
            try:
                messages = json.loads(prompt_text)
            except (ValueError, TypeError):
                messages = None
            if isinstance(messages, dict):
                messages = messages.get('messages')
            if isinstance(messages, list):
                return self._dsv41_encode_messages(messages)
            elif '<｜Assistant｜>' not in prompt_text:
                prompt_text = _dsv41_render_prompt([{'role': 'user', 'content': prompt_text}], self._dsv41_thinking_mode)
            return self.tokenizer.encode(prompt_text)
        if fallback_token_ids:
            return list(fallback_token_ids)
        if not prompt_text:
            return []
        return list(self.tokenizer.encode(prompt_text))

    def vision_generate(self, raw_image_bytes, prompt_text, max_tokens, stop_words, ls):
        if _is_dsv41(self.model):
            raise ValueError('[DSV41] Send images as image_url message blocks through the remote prompt endpoint')
        import tempfile, os
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        if not self._use_vlm:
            raise RuntimeError('vision_generate requires mlx_vlm mode (LLAMA_MLX_TEXT_LM_LOAD=0)')
        import struct, json as _json
        if isinstance(raw_image_bytes, (bytes, bytearray)) and raw_image_bytes[:4] == b'MIMG':
            n_img = struct.unpack_from('<I', raw_image_bytes, 4)[0]
            sizes = [struct.unpack_from('<I', raw_image_bytes, 8 + 4 * i)[0] for i in range(n_img)]
            off = 8 + 4 * n_img
            image_blobs = []
            for sz in sizes:
                image_blobs.append(bytes(raw_image_bytes[off:off + sz]))
                off += sz
        else:
            image_blobs = [raw_image_bytes]
        chat_prompt = prompt_text
        try:
            if prompt_text and prompt_text.lstrip()[:1] == '[':
                _conv = _json.loads(prompt_text)
                if isinstance(_conv, list) and _conv:
                    chat_prompt = _conv
        except Exception:
            pass
        img_paths = []
        for _blob in image_blobs:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(_blob)
                img_paths.append(f.name)
        img_path = img_paths[0] if img_paths else None
        t0 = time.time()
        n = 0
        pos = 0
        decoded = ''
        stop_reason = STOP_EOS
        stopping_word = ''
        try:
            formatted = apply_chat_template(self.processor, self.model.config, chat_prompt, num_images=len(img_paths))
            sys.stderr.write(f'[mlx-server] VISION_GEN: images={len(img_paths)} turns={(len(chat_prompt) if isinstance(chat_prompt, list) else 1)} prompt={str(chat_prompt)[:90]!r} max_tokens={max_tokens}\n')
            sys.stderr.flush()
            for chunk in stream_generate(self.model, self.processor, formatted, image=img_paths, max_tokens=max_tokens, temperature=0.0):
                text = getattr(chunk, 'text', None)
                if text is None:
                    text = chunk if isinstance(chunk, str) else ''
                tid = int(getattr(chunk, 'token', -1) or -1)
                if not text:
                    continue
                if not _send_token(ls, tid, text, pos):
                    stop_reason = STOP_CANCELLED
                    break
                pos += 1
                n += 1
                decoded += text
                hw = ''
                for sw in stop_words:
                    if sw and decoded.endswith(sw):
                        hw = sw
                        break
                if hw:
                    stop_reason = STOP_WORD
                    stopping_word = hw
                    break
        finally:
            for _p in img_paths:
                try:
                    os.unlink(_p)
                except Exception:
                    pass
        tps = n / max(time.time() - t0, 1e-09)
        sys.stderr.write(f'[mlx-server] VISION_GEN done: {n} tok, {tps:.2f} tok/s, stop={stop_reason}\n')
        sys.stderr.flush()
        return (n, tps, stop_reason, stopping_word)

    def prefill_and_decode(self, prompt_token_ids: List[int], max_tokens: int, stop_words: List[str], ls: LineSocket, chunk_size: int=2048, kv_state_bytes: bytes=b'', kv_is_mla: bool=False, last_token_id: int=-1, image_embd=None, img_pos_start: int=0, sampling_profile: Optional[Dict[str, Any]]=None, mtp_hidden=None, swi_config: Optional['SwiConfig']=None, kv_delta_base: int=0, reuse_retained_count: int=0) -> Tuple[int, float, int, str]:
        import mlx.core as mx
        import mlx.nn as nn
        import contextlib
        if (Path.home() / '.streaming-studio' / 'runtime-profile.json').is_file():
            import importlib
            from mlx_vlm import runtime_profile
            importlib.reload(runtime_profile).run_requested(self.model, prompt_token_ids)
        is_dsv41 = _is_dsv41(self.model)
        if is_dsv41 and os.environ.get('DSV41_DSPARK_COLD_BENCH', '0') == '1':
            self._retained_cache = None
            self._retained_tokens = []
        vision_prompt = getattr(self, '_dsv41_vision_prompt', None) if is_dsv41 else None
        self._dsv41_vision_prompt = None
        kimi_vp = getattr(self, '_kimi_vision_prompt', None)
        self._kimi_vision_prompt = None
        if kimi_vp is not None:
            if kimi_vp.token_ids != list(prompt_token_ids):
                raise ValueError('Kimi image prompt tokens do not match this request')
            if kv_state_bytes or image_embd is not None:
                raise ValueError('Kimi image requests are prompt-direct only')
            swi_config = None
        vision_embeds = None
        if vision_prompt is not None and vision_prompt[0] != list(prompt_token_ids):
            raise ValueError('[DSV41] Vision prompt tokens do not match this request')
        if is_dsv41:
            self._dsv41_completion = None
            if image_embd is not None:
                raise ValueError('[DSV41] Image embeddings are unsupported')
            if kv_delta_base:
                raise ValueError('[DSV41] RTX KV deltas are unsupported; send the full prompt')
            kv_state_bytes = b''
            mtp_hidden = None
            swi_config = None
            chunk_size = int(os.environ.get('DSV41_PREFILL_CHUNK', '256'))
            if not 1 <= chunk_size <= 4096:
                raise ValueError('DSV41_PREFILL_CHUNK must be in 1..4096')
            if not prompt_token_ids:
                raise ValueError('[DSV41] Local prefill requires prompt tokens')
        request_sampling = _merge_sampling_profiles(self._sampling_profile, sampling_profile)
        token_sampler = _MlxRequestSampler(mx, request_sampling)
        sys.stderr.write(f"[mlx-server] sampling profile: temp={request_sampling['temp']:.3f} top_k={request_sampling['top_k']} top_p={request_sampling['top_p']:.3f} min_p={request_sampling['min_p']:.3f} seed={request_sampling['seed']}\n")
        sys.stderr.flush()
        self._swi_last_stats = None
        _swi_active = bool(swi_config is not None and getattr(swi_config, 'enabled', False) and (image_embd is None))
        if _swi_active:
            sys.stderr.write(f'[mlx-server] SwiReasoning ENABLED for this request (alpha_0={swi_config.alpha_0} beta_0={swi_config.beta_0} window_size={swi_config.window_size} max_switch_count={swi_config.max_switch_count} termination_max_tokens={swi_config.termination_max_tokens}); MTP forced OFF\n')
            sys.stderr.flush()
        _ship_image = image_embd is not None and len(image_embd) > 0
        if _ship_image:
            kv_state_bytes = b''
        _img_hashes: List[str] = []
        _img_positions: List[List[int]] = []
        if _ship_image and self._prompt_store is not None:
            try:
                from prompt_cache_store import image_hash as _image_hash
                _cells = int(len(image_embd))
                _raw = image_embd.tobytes() if hasattr(image_embd, 'tobytes') else bytes(image_embd)
                _img_hashes = [_image_hash(_raw)]
                _img_positions = [[int(img_pos_start), _cells]]
            except Exception as _ih_err:
                sys.stderr.write(f'[mlx-server] image-hash failed (skipping store keying): {_ih_err}\n')
                sys.stderr.flush()
                _img_hashes = []
                _img_positions = []
        if kv_state_bytes:
            _enable_streaming = os.environ.get('LLAMA_MLX_ENABLE_STREAMING_INJECT', '1') not in ('', '0')
            _safe_inject_cells = 1000000
            _max_safe_tail = int(os.environ.get('LLAMA_MLX_MAX_SAFE_TAIL', '2048'))
            _allow_no_tail = os.environ.get('LLAMA_MLX_ALLOW_NO_TAIL', '0') not in ('', '0')
            _np = len(prompt_token_ids)
            _bypass_reason = None
            if not _enable_streaming:
                _bypass_reason = 'default-disabled (set LLAMA_MLX_ENABLE_STREAMING_INJECT=1 to opt in)'
            elif _np <= _safe_inject_cells and (not _allow_no_tail):
                _bypass_reason = f'no tail re-prefill possible (n_prompt={_np} <= inject={_safe_inject_cells}); set LLAMA_MLX_ALLOW_NO_TAIL=1 to attempt direct decode from injected cells'
            elif _np - _safe_inject_cells > _max_safe_tail:
                _bypass_reason = f'tail too large ({_np - _safe_inject_cells} > {_max_safe_tail})'
            if _bypass_reason:
                sys.stderr.write(f'[mlx-server] streaming-inject bypassed: {_bypass_reason}; falling back to PROMPT-DIRECT (cache retention minimizes cost on subsequent turns)\n')
                sys.stderr.flush()
                kv_state_bytes = b''
        mtp_drafter = self._mtp_drafter
        _mtp_kv_inject = bool(kv_state_bytes and mtp_hidden is not None)
        mtp_can_run = bool(mtp_drafter is not None and (not _ship_image) and (kimi_vp is None) and (not kv_state_bytes or _mtp_kv_inject))
        if _swi_active and mtp_can_run:
            mtp_can_run = False
        if mtp_drafter is not None and (not mtp_can_run):
            sys.stderr.write('[mlx-server] MTP draft diagnostics skipped for this request (requires full local text-token prefill)\n')
            sys.stderr.flush()
        _reuse_prefix_candidate = None
        _reuse_prefix_for_save = None
        if reuse_retained_count and int(reuse_retained_count) > 0 and (not kv_state_bytes) and (not _ship_image) and self._retained_cache and self._retained_tokens and (int(reuse_retained_count) <= len(self._retained_tokens)):
            _reuse_prefix_candidate = list(self._retained_tokens[:int(reuse_retained_count)])
        if mtp_can_run:
            cache, lcp_reused = self._cache_with_reuse(prompt_token_ids, kv_state_bytes, text_only=not _ship_image, image_hashes=_img_hashes, reuse_retained_count=reuse_retained_count, delta_base=int(kv_delta_base or 0))
            _reuse_count_honored = bool(_reuse_prefix_candidate is not None and cache is self._retained_cache and (len(self._retained_tokens) == int(reuse_retained_count)))
            if _reuse_count_honored:
                _reuse_prefix_for_save = _reuse_prefix_candidate
            if lcp_reused > 0 or _reuse_count_honored:
                mtp_can_run = False
                sys.stderr.write(f'[mlx-server] prefix reuse hit ({(int(reuse_retained_count) if _reuse_count_honored else lcp_reused)} cached tok) -> MTP off this turn (target-only decode)\n')
                sys.stderr.flush()
            else:
                mtp_drafter.reset_request()
                sys.stderr.write('[mlx-server] MTP draft diagnostics: using fresh local prefill to warm the MTP sidecar cache\n')
                sys.stderr.flush()
        else:
            cache, lcp_reused = self._cache_with_reuse(prompt_token_ids, kv_state_bytes, text_only=not _ship_image, image_hashes=_img_hashes, reuse_retained_count=reuse_retained_count, delta_base=int(kv_delta_base or 0))
            if _reuse_prefix_candidate is not None and cache is self._retained_cache and (len(self._retained_tokens) == int(reuse_retained_count)):
                _reuse_prefix_for_save = _reuse_prefix_candidate
        _ship_image_store_hit = bool(_ship_image and _img_hashes and (lcp_reused > 0))
        self._retained_cache = None
        self._retained_tokens = []
        lang = getattr(self.model, 'language_model', self.model)
        dspark_runtime = None
        if is_dsv41 and (not _swi_active) and (vision_prompt is None) and (not kv_state_bytes):
            from mlx_vlm.models.deepseek_v41.dspark_receiver import DSparkReceiver, enabled
            if enabled():
                dspark_runtime = DSparkReceiver(lang, token_sampler)
        eos_ids = self._eos_ids()
        mtp_reference_prev_hidden = None
        mtp_reference_prev_topk = None
        gen_stream = mx.new_thread_local_stream(mx.default_device())

        @contextlib.contextmanager
        def _wired(model):
            if not mx.metal.is_available():
                yield
                return
            max_rec = mx.device_info().get('max_recommended_working_set_size', 0)
            if max_rec > 0:
                old = mx.set_wired_limit(max_rec)
                try:
                    yield
                finally:
                    mx.synchronize(gen_stream)
                    mx.set_wired_limit(old)
            else:
                yield

        def _forward_token_ids_with_topk(ids_arr, final=True):
            nonlocal vision_embeds
            skip_logits = not final and is_dsv41 and (os.environ.get('DSV41_PREFILL_LAST_LOGITS', '0') == '1')
            with mx.stream(gen_stream):
                if dspark_runtime is not None:
                    out = dspark_runtime.capture(ids_arr, cache, skip_logits=skip_logits)
                    return (out.logits, None, None)
                if vision_prompt is not None and cache[0].offset < len(prompt_token_ids):
                    if vision_embeds is None:
                        vision_embeds = self.model.get_input_embeddings(mx.array([prompt_token_ids], dtype=mx.int32), vision_prompt[1]).inputs_embeds
                        mx.eval(vision_embeds)
                        sys.stderr.write('[DSV41] vision embedding invocation complete\n')
                    start = cache[0].offset
                    out = lang(ids_arr, cache=cache, inputs_embeds=vision_embeds[:, start:start + ids_arr.shape[1]])
                    return (out.logits, None, None)
                if kimi_vp is not None:
                    start = _cache_offset(cache)
                    if 0 <= start < len(prompt_token_ids):
                        ids = list(prompt_token_ids[start:start + int(ids_arr.shape[1])])
                        if any((t >= _KIMI_PSEUDO_BASE for t in ids)):
                            if ids != ids_arr.reshape(-1).tolist():
                                raise RuntimeError('image prefill chunk is not at the cache offset')
                            from kimi_vision import embed_rows, forward_embeddings
                            rows = embed_rows(self._kimi_vision, kimi_vp, lang.model.embed_tokens, ids)
                            return (forward_embeddings(lang, rows, cache), None, None)
                if mtp_can_run:
                    logits, hidden, topk = target_logits_hidden_topk(lang, ids_arr, cache)
                elif skip_logits:
                    out = lang(ids_arr, cache=cache, skip_logits=True)
                    logits = None
                    hidden = None
                    topk = None
                else:
                    out = lang(ids_arr, cache=cache)
                    logits = out.logits if hasattr(out, 'logits') else out
                    hidden = None
                    topk = None
                return (logits, hidden, topk)

        def _forward_token_ids(ids_arr):
            logits, hidden, _topk = _forward_token_ids_with_topk(ids_arr)
            return (logits, hidden)

        def _forward_verify_token_ids(ids_arr):
            return _mtp_forward_verify_token_ids_with_topk(mx, ids_arr, _forward_token_ids_with_topk, self._mtp_verify_forward)

        def _warm_mtp(ids_arr, hidden) -> None:
            if mtp_can_run and hidden is not None:
                mtp_drafter.prefill(ids_arr, hidden)
        n_prompt = len(prompt_token_ids)
        t_prefill0 = time.time()
        try:
            _t_warm0 = time.time()
            mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)
            _t_warm = time.time() - _t_warm0
            sys.stderr.write(f'[mlx-server] request warm-up eval: {_t_warm:.3f}s [FIRST_EVAL]\n')
            sys.stderr.flush()
        except Exception as _we:
            if is_dsv41:
                raise
            sys.stderr.write(f'[mlx-server] request warm-up eval failed: {_we}\n')
        swi_last_logits_2d = None
        _mac_retain_handle_enabled = os.environ.get('LLAMA_DSV4_MAC_RETAIN_HANDLE', '') not in ('', '0')
        _kv_inject_was_requested = bool(kv_state_bytes)
        _kv_inject_succeeded = False
        _kv_outer_view = None
        if kv_state_bytes:
            sys.stderr.write(f'[mlx-server] KV inject: {len(kv_state_bytes)} bytes, is_mla={kv_is_mla}, n_prompt={n_prompt}\n')
            sys.stderr.flush()
            try:
                if len(kv_state_bytes) >= 8 and kv_state_bytes[:8] == b'KVARTIF1':
                    _kv_outer_view = kv_state_bytes
                    _inner = _parse_kva_prompt_payload_bytes(kv_state_bytes)
                    kv_state_bytes = _inner.get('kv_state_bytes') or b''
                    if _inner.get('is_mla') is not None:
                        kv_is_mla = bool(_inner['is_mla'])
                    inner_base = int(_inner.get('base_prefix_cells', 0) or 0)
                    if inner_base != int(kv_delta_base or 0):
                        raise RuntimeError('nested KV artifact base disagrees with outer base')
                    _inner = None
                    if not kv_state_bytes:
                        raise RuntimeError('nested KV artifact contains no KV payload')
                _delta_base = int(kv_delta_base or 0)
                if not 0 <= _delta_base <= n_prompt:
                    raise RuntimeError(f'invalid delta base {_delta_base} for {n_prompt} prompt tokens')
                _validate_kv_injection_cache(cache, _delta_base)
                from kv_converter import parse_kv_payload_mla, inject_kv_mla, inject_kv_idx, parse_kv_payload, inject_kv_into_cache
                if kv_is_mla:
                    _lm = getattr(self.model, 'language_model', self.model)
                    cfg = getattr(_lm, 'args', None) or getattr(_lm, 'config', None)
                    kv_lora_rank = getattr(cfg, 'kv_lora_rank', 512)
                    rope_dim = getattr(cfg, 'qk_rope_head_dim', 64)
                    layers_mla, mla_meta = parse_kv_payload_mla(kv_state_bytes, kv_lora_rank, rope_dim)
                    cell_count = mla_meta['cell_count']
                    _delta_base = int(kv_delta_base or 0)
                    _incoming_cells = cell_count
                    if cell_count <= 0 or _delta_base + cell_count > n_prompt:
                        raise RuntimeError('KV artifact cells exceed the supplied prompt')
                    cell_count = _delta_base + _incoming_cells
                    _no_yarn_trunc = os.environ.get('LLAMA_MLX_NO_YARN_TRUNC', '1') not in ('', '0')
                    _orig_max = cell_count
                    if _no_yarn_trunc:
                        safe_cells = cell_count
                    else:
                        _rope_scaling = getattr(cfg, 'rope_scaling', None) or {}
                        if isinstance(_rope_scaling, dict):
                            _orig_max = _rope_scaling.get('original_max_position_embeddings', cell_count)
                        else:
                            _orig_max = getattr(_rope_scaling, 'original_max_position_embeddings', cell_count)
                        safe_cells = min(cell_count, int(_orig_max))
                    _force_logit_tail = os.environ.get('LLAMA_MLX_FULL_KV_LOGIT_TAIL', '1') not in ('', '0')
                    if _force_logit_tail and safe_cells >= n_prompt and (n_prompt > 0):
                        safe_cells = max(0, n_prompt - max(1, int(os.environ.get('LLAMA_MLX_TAIL_REPREFILL_N', 1))))
                        sys.stderr.write('[mlx-server] full-KV inject: retaining 1-token MLX tail to compute first generated token logits\n')
                        sys.stderr.flush()
                    if safe_cells < cell_count:
                        sys.stderr.write(f'[mlx-server] truncating injection to {safe_cells} cells (YaRN boundary at {_orig_max}); {cell_count - safe_cells} cells will be re-prefilled\n')
                        sys.stderr.flush()
                        layers_mla = [(kv[:, :, :safe_cells, :], kp[:, :, :safe_cells, :]) for kv, kp in layers_mla]
                    inject_mla_rows(cache, layers_mla, mla_meta, safe_cells, _delta_base, _incoming_cells, kv_delta_base)
                    if self.model is not None:
                        _force_dense = os.path.exists('/tmp/force_dense_indexer')
                        try:
                            _lys = getattr(getattr(self.model, 'model', None), 'layers', None) or []
                            for _ly in _lys:
                                if _ly is None:
                                    continue
                                _at = getattr(_ly, 'self_attn', None)
                                _ix = getattr(_at, 'indexer', None) if _at is not None else None
                                if _ix is None or not hasattr(_ix, 'index_topk'):
                                    continue
                                if not hasattr(_ix, '_orig_index_topk'):
                                    _ix._orig_index_topk = _ix.index_topk
                                _ix.index_topk = 1000000000 if _force_dense else _ix._orig_index_topk
                        except Exception as _efd:
                            sys.stderr.write(f'[mlx-server] force-dense toggle failed: {_efd}\n')
                        if _force_dense:
                            sys.stderr.write('[mlx-server] FORCE_DENSE_INDEXER(file): indexer DENSE this request\n')
                            sys.stderr.flush()
                    _validate_kv_injection_cache(cache, safe_cells)
                    cell_count = safe_cells
                    if os.environ.get('LLAMA_MLX_INJECT_DIAG', '') not in ('', '0'):
                        try:
                            _c0m = cache[0].caches[0] if hasattr(cache[0], 'caches') else cache[0]
                            _kl = _c0m.keys
                            _kp = _c0m.values
                            sys.stderr.write(f'[mlx-server] INJECTED-cache0[L0]: kv_latent shape={tuple(_kl.shape)} mean={float(mx.mean(_kl)):.4f} std={float(mx.std(_kl)):.4f} absmax={float(mx.max(mx.abs(_kl))):.4f} nan={int(mx.sum(mx.isnan(_kl)))} | k_pe shape={tuple(_kp.shape)} mean={float(mx.mean(_kp)):.4f} std={float(mx.std(_kp)):.4f} absmax={float(mx.max(mx.abs(_kp))):.4f}\n')
                            sys.stderr.flush()
                        except Exception as _edc:
                            sys.stderr.write(f'[mlx-server] cache0 dump failed: {_edc}\n')
                        try:
                            _firstnan = -1
                            for _L in range(len(cache)):
                                _icL = cache[_L].caches[0] if hasattr(cache[_L], 'caches') else cache[_L]
                                if int(mx.sum(mx.isnan(_icL.keys)).item()) > 0:
                                    _firstnan = _L
                                    break
                            sys.stderr.write(f'[mlx-server] INJECTED first-NaN-layer (RTX export) = {_firstnan} of {len(cache)}\n')
                            sys.stderr.flush()
                        except Exception as _efn:
                            sys.stderr.write(f'[mlx-server] first-NaN scan failed: {_efn}\n')
                        try:
                            _N = min(64, int(cell_count))
                            _refc = self._make_cache()
                            _ = lang(mx.array(prompt_token_ids[:_N])[None], cache=_refc)
                            _nL = len(cache)
                            for _L in (0, _nL // 2, _nL - 1):
                                _rcL = _refc[_L].caches[0] if hasattr(_refc[_L], 'caches') else _refc[_L]
                                _icL = cache[_L].caches[0] if hasattr(cache[_L], 'caches') else cache[_L]
                                _rkl = _rcL.keys[:, :, :_N, :].astype(mx.float32)
                                _ikl = _icL.keys[:, :, :_N, :].astype(mx.float32)
                                _rkp = _rcL.values[:, :, :_N, :].astype(mx.float32)
                                _ikp = _icL.values[:, :, :_N, :].astype(mx.float32)
                                mx.eval(_rkl, _ikl, _rkp, _ikp)
                                sys.stderr.write(f'[mlx-server] VALUE-DIFF L{_L} first{_N} kv_latent max={float(mx.max(mx.abs(_rkl - _ikl))):.5f} mean={float(mx.mean(mx.abs(_rkl - _ikl))):.5f} (ref-std={float(mx.std(_rkl)):.5f}) || k_pe max={float(mx.max(mx.abs(_rkp - _ikp))):.5f} mean={float(mx.mean(mx.abs(_rkp - _ikp))):.5f} (ref-std={float(mx.std(_rkp)):.5f})\n')
                            sys.stderr.flush()
                        except Exception as _evd:
                            sys.stderr.write(f'[mlx-server] VALUE-DIFF failed: {_evd}\n')
                        try:
                            if os.path.exists('/tmp/value_diff_deep'):
                                _NC = int(cell_count)
                                _refc2 = self._make_cache()
                                _CH = 1024
                                for _o in range(0, _NC, _CH):
                                    _ = lang(mx.array(prompt_token_ids[_o:min(_o + _CH, _NC)])[None], cache=_refc2)
                                _nL = len(cache)
                                for _L in (0, _nL // 2, _nL - 1):
                                    _rcL = _refc2[_L].caches[0] if hasattr(_refc2[_L], 'caches') else _refc2[_L]
                                    _icL = cache[_L].caches[0] if hasattr(cache[_L], 'caches') else cache[_L]
                                    _rk = _rcL.keys[:, :, :_NC, :].astype(mx.float32)
                                    _ik = _icL.keys[:, :, :_NC, :].astype(mx.float32)
                                    _dprof = mx.mean(mx.abs(_rk - _ik), axis=(0, 1, 3))
                                    _rstd = mx.std(_rk)
                                    mx.eval(_dprof, _rstd)
                                    _CHK = 256
                                    _chunks = [float(mx.mean(_dprof[_i:_i + _CHK]).item()) for _i in range(0, _NC, _CHK)]
                                    _order = sorted(range(len(_chunks)), key=lambda _j: -_chunks[_j])[:6]
                                    sys.stderr.write(f'[mlx-server] DEEP-DIFF L{_L} cells={_NC} ref-std={float(_rstd.item()):.5f} chunk256=[' + ','.join((f'{_c:.4f}' for _c in _chunks)) + ']\n')
                                    sys.stderr.write(f'[mlx-server] DEEP-DIFF L{_L} worst=' + ','.join((f'[{_j * _CHK}:{min((_j + 1) * _CHK, _NC)}]={_chunks[_j]:.4f}' for _j in _order)) + '\n')
                                    try:
                                        _rix = _refc2[_L].caches[1] if hasattr(_refc2[_L], 'caches') and len(_refc2[_L].caches) > 1 else None
                                        _iix = cache[_L].caches[1] if hasattr(cache[_L], 'caches') and len(cache[_L].caches) > 1 else None
                                        _rk2 = getattr(_rix, 'keys', None) if _rix is not None else None
                                        _ik2 = getattr(_iix, 'keys', None) if _iix is not None else None
                                        if _rk2 is None and _rix is not None and hasattr(_rix, 'state'):
                                            _rk2 = _rix.state[0]
                                        if _ik2 is None and _iix is not None and hasattr(_iix, 'state'):
                                            _ik2 = _iix.state[0]
                                        if _rk2 is not None and _ik2 is not None:
                                            _NI = min(_NC, int(_rk2.shape[-2]), int(_ik2.shape[-2]))
                                            _r3 = _rk2[..., :_NI, :].astype(mx.float32)
                                            _i3 = _ik2[..., :_NI, :].astype(mx.float32)
                                            _dp2 = mx.mean(mx.abs(_r3 - _i3), axis=tuple(range(_r3.ndim - 2)) + (_r3.ndim - 1,))
                                            _rs2 = mx.std(_r3)
                                            mx.eval(_dp2, _rs2)
                                            _ch2 = [float(mx.mean(_dp2[_i:_i + _CHK]).item()) for _i in range(0, _NI, _CHK)]
                                            sys.stderr.write(f'[mlx-server] DEEP-DIFF-IDX L{_L} cells={_NI} ref-std={float(_rs2.item()):.5f} chunk256=[' + ','.join((f'{_c:.4f}' for _c in _ch2)) + ']\n')
                                            if _L != 0:
                                                _c00 = cache[0].caches[1] if hasattr(cache[0], 'caches') and len(cache[0].caches) > 1 else None
                                                _ik0 = getattr(_c00, 'keys', None) if _c00 is not None else None
                                                if _ik0 is None and _c00 is not None and hasattr(_c00, 'state'):
                                                    _ik0 = _c00.state[0]
                                                _rc0 = _refc2[0].caches[1] if hasattr(_refc2[0], 'caches') and len(_refc2[0].caches) > 1 else None
                                                _rk0 = getattr(_rc0, 'keys', None) if _rc0 is not None else None
                                                if _rk0 is None and _rc0 is not None and hasattr(_rc0, 'state'):
                                                    _rk0 = _rc0.state[0]
                                                if _ik0 is not None:
                                                    _i03 = _ik0[..., :_NI, :].astype(mx.float32)
                                                    _dii = float(mx.mean(mx.abs(_i3 - _i03)).item())
                                                    _dri = float(mx.mean(mx.abs(_i3 - _rk0[..., :_NI, :].astype(mx.float32))).item()) if _rk0 is not None else -1.0
                                                    sys.stderr.write(f'[mlx-server] DEEP-DIFF-IDX-CROSS L{_L}: inj[L]-vs-inj[L0]={_dii:.5f} inj[L]-vs-ref[L0]={_dri:.5f}\n')
                                        else:
                                            sys.stderr.write(f'[mlx-server] DEEP-DIFF-IDX L{_L} skipped: no indexer cache pair\n')
                                    except Exception as _eix:
                                        sys.stderr.write(f'[mlx-server] DEEP-DIFF-IDX L{_L} failed: {_eix}\n')
                                sys.stderr.flush()
                        except Exception as _evd2:
                            sys.stderr.write(f'[mlx-server] DEEP-DIFF failed: {_evd2}\n')
                else:
                    cfg = getattr(getattr(self.model, 'language_model', self.model), 'args', None) or getattr(self.model, 'config', None)
                    n_kv_heads = getattr(cfg, 'num_key_value_heads', getattr(cfg, 'n_kv_heads', 8))
                    head_dim = getattr(cfg, 'head_dim', 128)
                    if int(kv_delta_base or 0) > 0:
                        raise RuntimeError('delta artifacts are only supported for MLA caches')
                    layers_std, std_meta = parse_kv_payload(kv_state_bytes, n_kv_heads, head_dim)
                    cell_count = std_meta['cell_count']
                    if not 0 < cell_count <= n_prompt:
                        raise RuntimeError('KV artifact cells exceed the supplied prompt')
                    inject_standard_rows(cache, layers_std, cell_count)
                    _validate_kv_injection_cache(cache, cell_count)
                inject_dt = time.time() - t_prefill0
                sys.stderr.write(f'[mlx-server] KV inject done in {inject_dt:.2f}s, cell_count={cell_count}\n')
                sys.stderr.flush()
                _skip_tail = os.environ.get('LLAMA_MLX_SKIP_TAIL', '0') not in ('', '0')
                tail_ids = [] if _skip_tail else prompt_token_ids[cell_count:]
                if tail_ids:
                    tail_arr = mx.array(tail_ids)
                    n_tail = len(tail_ids)
                    sys.stderr.write(f'[mlx-server] re-prefill tail: {n_tail} tokens\n')
                    sys.stderr.flush()
                    last_logits = None
                    for start in range(0, n_tail, chunk_size):
                        end = min(start + chunk_size, n_tail)
                        chunk = tail_arr[start:end][None]
                        last_logits, last_hidden = _forward_token_ids(chunk)
                        mx.eval(last_logits)
                        _warm_mtp(chunk, last_hidden)
                    first_id = token_sampler.sample_token(last_logits[0, -1, :])
                    n_prompt = cell_count + n_tail
                    for _c in cache:
                        if hasattr(_c, 'keys') and _c.keys is not None:
                            mx.eval(_c.keys, _c.values)
                            break
                    try:
                        _c0c = cache[0].caches[0] if hasattr(cache[0], 'caches') else cache[0]
                        _klc = _c0c.keys
                        _injr = _klc[:, :, :cell_count, :]
                        _tailr = _klc[:, :, cell_count:, :]
                        sys.stderr.write(f'[mlx-server] KV-COMPARE kv_latent INJECTED(RTX,0:{cell_count}) std={float(mx.std(_injr)):.5f} absmax={float(mx.max(mx.abs(_injr))):.4f} || TAIL(MLX,{cell_count}:) std={float(mx.std(_tailr)):.5f} absmax={float(mx.max(mx.abs(_tailr))):.4f}\n')
                        _vc = _c0c.values
                        _injv = _vc[:, :, :cell_count, :]
                        _tailv = _vc[:, :, cell_count:, :]
                        sys.stderr.write(f'[mlx-server] KV-COMPARE k_pe INJECTED(RTX) std={float(mx.std(_injv)):.5f} absmax={float(mx.max(mx.abs(_injv))):.4f} || TAIL(MLX) std={float(mx.std(_tailv)):.5f} absmax={float(mx.max(mx.abs(_tailv))):.4f}\n')
                        sys.stderr.flush()
                    except Exception as _ekc:
                        sys.stderr.write(f'[mlx-server] KV-COMPARE failed: {_ekc}\n')
                    sys.stderr.write(f'[mlx-server] tail done, first_id={first_id}\n')
                else:
                    first_id = last_token_id if last_token_id >= 0 else prompt_token_ids[-1]
                    n_prompt = cell_count
                if _mac_retain_handle_enabled:
                    _materialize = []
                    for _outer_cache in cache:
                        _cache_children = getattr(_outer_cache, 'caches', None) or [_outer_cache]
                        for _cache_child in _cache_children:
                            _state = getattr(_cache_child, 'state', None)
                            if _state is not None:
                                _materialize.extend((x for x in _state if x is not None))
                    if _materialize:
                        mx.eval(*_materialize)
                    _kv_inject_succeeded = True
                _released = _release_kv_views(kv_state_bytes, _kv_outer_view)
                _kv_outer_view = None
                kv_state_bytes = _KV_INJECTED
                if _released:
                    sys.stderr.write(f'[mlx-server] artifact buffer released after inject: {_released} [KV_RELEASE]\n')
                    sys.stderr.flush()
            except Exception as _kv_err:
                import traceback as _kv_tb
                sys.stderr.write(f'[mlx-server] KV inject failed: {_kv_err}\n[mlx-server] KV inject traceback:\n{_kv_tb.format_exc()}')
                sys.stderr.flush()
                self._retained_cache = None
                self._retained_tokens = []
                if not prompt_token_ids or reuse_retained_count:
                    raise RuntimeError('KV artifact injection failed; full prompt unavailable') from _kv_err
                sys.stderr.write(f'[mlx-server] KV inject failed -> full local prefill of {len(prompt_token_ids)} tokens [KV_INJECT_FALLBACK]\n')
                sys.stderr.flush()
                _release_kv_views(kv_state_bytes, _kv_outer_view)
                kv_state_bytes = b''
                _kv_outer_view = None
                _inner = None
                layers_mla = layers_std = mla_meta = std_meta = None
                _materialize = []
                cache = self._make_cache()
                lcp_reused = 0
                n_prompt = len(prompt_token_ids)
                kv_delta_base = 0
                _reuse_prefix_for_save = None
                _kv_inject_was_requested = _kv_inject_succeeded = False
                _mtp_kv_inject = False
                mtp_hidden = None
                mtp_reference_prev_hidden = mtp_reference_prev_topk = None
                mtp_can_run = bool(mtp_drafter is not None and (not _ship_image) and (not _swi_active))
                if mtp_drafter is not None:
                    mtp_drafter.reset_request()
                token_sampler = _MlxRequestSampler(mx, request_sampling)
                t_prefill0 = time.time()
        if not kv_state_bytes and _ship_image and _ship_image_store_hit:
            emb_tok = getattr(lang, 'embed_tokens', None) or getattr(getattr(lang, 'model', None), 'embed_tokens', None) or getattr(getattr(getattr(lang, 'model', None), 'model', None), 'embed_tokens', None)
            if emb_tok is None:
                raise RuntimeError('ship-embeddings: cannot resolve embed_tokens on the language model')
            if not self._use_vlm:
                raise RuntimeError('ship-image re-prefill requires mlx_vlm mode; relaunch with LLAMA_MLX_TEXT_LM_LOAD=0')
            suffix_ids = list(prompt_token_ids[lcp_reused:])
            _cells = int(len(image_embd))
            n_prompt = len(prompt_token_ids) + _cells
            sys.stderr.write(f'[mlx-server] SHIP_EMBD store-hit: reusing {lcp_reused} text tok + {_cells} image cells from disk; re-prefilling {len(suffix_ids)} suffix tok\n')
            sys.stderr.flush()
            if suffix_ids:
                suffix_emb = emb_tok(mx.array([suffix_ids]))
                with mx.stream(gen_stream):
                    out = lang(None, inputs_embeds=suffix_emb, cache=cache)
                    last_logits = out.logits if hasattr(out, 'logits') else out
                mx.eval(last_logits)
                first_id = token_sampler.sample_token(last_logits[0, -1, :])
            else:
                first_id = last_token_id if last_token_id >= 0 else prompt_token_ids[-1]
        elif not kv_state_bytes and _ship_image:
            cache = self._make_cache()
            emb_tok = getattr(lang, 'embed_tokens', None) or getattr(getattr(lang, 'model', None), 'embed_tokens', None) or getattr(getattr(getattr(lang, 'model', None), 'model', None), 'embed_tokens', None)
            if emb_tok is None:
                raise RuntimeError('ship-embeddings: cannot resolve embed_tokens on the language model')
            n_prefix = int(img_pos_start)
            prefix_ids = list(prompt_token_ids[:n_prefix])
            suffix_ids = list(prompt_token_ids[n_prefix:])
            parts = []
            if prefix_ids:
                parts.append(emb_tok(mx.array([prefix_ids])))
            img_mx = mx.array(image_embd)
            img_mx = img_mx.astype(parts[0].dtype if parts else mx.bfloat16)
            parts.append(img_mx[None])
            if suffix_ids:
                parts.append(emb_tok(mx.array([suffix_ids])).astype(img_mx.dtype))
            spliced = mx.concatenate(parts, axis=1)
            total_pos = int(spliced.shape[1])
            sys.stderr.write(f'[mlx-server] SHIP_EMBD: spliced inputs_embeds prefix={len(prefix_ids)} image={len(image_embd)} suffix={len(suffix_ids)} total={total_pos} n_embd={int(spliced.shape[-1])}\n')
            sys.stderr.flush()
            if not self._use_vlm:
                raise RuntimeError('ship-image re-prefill requires mlx_vlm mode; relaunch with LLAMA_MLX_TEXT_LM_LOAD=0')
            with mx.stream(gen_stream):
                out = lang(None, inputs_embeds=spliced, cache=cache)
                last_logits = out.logits if hasattr(out, 'logits') else out
            mx.eval(last_logits)
            first_id = token_sampler.sample_token(last_logits[0, -1, :])
            n_prompt = total_pos
            sys.stderr.write(f'[mlx-server] SHIP_EMBD: prefill done first_id={first_id} n_prompt={n_prompt}\n')
            sys.stderr.flush()
        elif not kv_state_bytes:
            n_prompt = len(prompt_token_ids)
            prefill_start = lcp_reused
            n_to_prefill = n_prompt - prefill_start
            if lcp_reused:
                sys.stderr.write(f'[mlx-server] prefill {n_to_prefill} delta tokens (skipping {lcp_reused} cached), chunk={chunk_size}\n')
            else:
                sys.stderr.write(f'[mlx-server] prefill {n_prompt} tokens, chunk={chunk_size}\n')
            sys.stderr.flush()
            prompt_arr = mx.array(prompt_token_ids[prefill_start:])
            last_logits = None
            mtp_hidden_chunks = [] if mtp_can_run else None
            mtp_topk_chunks = [] if mtp_can_run else None
            _t_cache0 = time.time()
            _cache_arrays = _flatten_mx_arrays(cache)
            if _cache_arrays:
                mx.eval(*_cache_arrays)
            _t_cache = time.time() - _t_cache0
            del _cache_arrays
            _probe = _first_fwd_probe_wrap(lang) if _first_fwd_probe_enabled() else None
            _t_first_fwd = time.time()
            _prefill_ledger_on = is_dsv41 and os.environ.get('DSV41_PREFILL_LEDGER', '0') == '1'
            if _prefill_ledger_on:
                _pl = {'tokens': 0, 'chunks': 0, 'build_ms': 0.0, 'eval_ms': 0.0}
                _pl_model = None
                try:
                    from mlx_vlm.models.deepseek_v41 import language as _pl_model
                    _pl_before = dict(_pl_model.PREFILL_LEDGER)
                    _pl_layers_before = dict(_pl_model.PREFILL_LAYER_MS)
                except Exception as _plexc:
                    _pl_model = None
                    _pl_error = type(_plexc).__name__
            _layer_major = is_dsv41 and os.environ.get('DSV41_PREFILL_LAYER_MAJOR', '0') == '1' and (n_to_prefill > 1)
            _prefill_bounds = [(0, n_to_prefill)] if _layer_major else [(start, min(start + chunk_size, n_to_prefill)) for start in range(0, n_to_prefill, chunk_size)]
            if _layer_major:
                sys.stderr.write(f'[mlx-server] DSV41_PREFILL_LAYER_MAJOR=1: one forward of {n_to_prefill} tok (the model chunks at DSV41_PREFILL_CHUNK)\n')
                sys.stderr.flush()
            for start, end in _prefill_bounds:
                if _prefill_ledger_on and end - start > 1:
                    _pl_t0 = time.perf_counter()
                chunk = prompt_arr[start:end][None]
                _t_build0 = time.time()
                last_logits, last_hidden, last_topk = _forward_token_ids_with_topk(chunk, final=end >= n_to_prefill)
                _t_build = time.time() - _t_build0
                if _prefill_ledger_on and end - start > 1:
                    _pl_t1 = time.perf_counter()
                if last_logits is not None:
                    mx.eval(last_logits)
                if is_dsv41:
                    mx.eval(cache[0].state)
                if _prefill_ledger_on and end - start > 1:
                    _pl_t2 = time.perf_counter()
                    _pl['tokens'] += end - start
                    _pl['chunks'] += 1
                    _pl['build_ms'] += (_pl_t1 - _pl_t0) * 1000.0
                    _pl['eval_ms'] += (_pl_t2 - _pl_t1) * 1000.0
                if start == 0:
                    _t_fwd = time.time() - _t_first_fwd
                    sys.stderr.write(f'[mlx-server] delta prefill: {_t_first_fwd - t_prefill0:.2f}s before the first forward (cache reuse/trim), first chunk of {end} tok in {_t_fwd:.2f}s [DELTA_TIMING] cache_eval={_t_cache:.2f}s build={_t_build:.2f}s eval={_t_fwd - _t_build:.2f}s\n')
                    sys.stderr.flush()
                    if _probe is not None:
                        _first_fwd_probe_report(_probe, _t_fwd)
                        _probe = None
                if mtp_hidden_chunks is not None and last_hidden is not None:
                    mx.eval(last_hidden)
                    if int(last_hidden.shape[1]) != end - start:
                        _cone_tail = _dsv41_cone_tail_rows()
                        _want = None if _cone_tail is None else min(_cone_tail, end - start)
                        if _want is None or int(last_hidden.shape[1]) != _want:
                            raise RuntimeError(f'[DSV41] target hidden has {int(last_hidden.shape[1])} rows for {end - start} prompt tokens: a drafter warmed from the target hidden state needs one row per position, or exactly the DSV41_PREFILL_CONE_TAIL rows it asked for (currently {_cone_tail}). Switch DSV41_PREFILL_CONE off, set the tail, or run without the drafter.')
                    mtp_hidden_chunks.append(last_hidden)
                if mtp_topk_chunks is not None and last_topk is not None:
                    mx.eval(last_topk)
                    mtp_topk_chunks.append(last_topk)
            prefill_dt = time.time() - t_prefill0
            tok_s_label = n_to_prefill / max(prefill_dt, 1e-09)
            if _prefill_ledger_on:
                try:
                    _pl_n = max(1, _pl['chunks'])
                    _pl_parts = ['tokens=%d chunks=%d chunk_ms=%.2f' % (_pl['tokens'], _pl['chunks'], (_pl['build_ms'] + _pl['eval_ms']) / _pl_n)]
                    for _pl_k in ('build_ms', 'eval_ms'):
                        _pl_parts.append('%s/chunk=%.2f' % (_pl_k, _pl[_pl_k] / _pl_n))
                    if _pl_model is not None:
                        for _pl_k in ('hash_ms', 'engram_fetch_ms'):
                            _pl_parts.append('%s=%.3f' % (_pl_k, _pl_model.PREFILL_LEDGER[_pl_k] - _pl_before[_pl_k]))
                        if _pl_model._PREFILL_LEDGER_CUTS:
                            _pl_rows = [(key, value - _pl_layers_before.get(key, 0.0)) for key, value in _pl_model.PREFILL_LAYER_MS.items()]
                            _pl_totals = {}
                            for (_, phase), value in _pl_rows:
                                _pl_totals[phase] = _pl_totals.get(phase, 0.0) + value
                            _pl_rows = sorted(((key, value) for key, value in _pl_rows if value > 0), key=lambda row: (-row[1], row[0]))
                            _pl_parts.append('cuts: layer_phase_ms/chunk=[%s] totals_ms/chunk=[%s]' % (' '.join(('%d:%s=%.2f' % (layer, phase, value / _pl_n) for (layer, phase), value in _pl_rows[:8])), ' '.join(('%s=%.2f' % (phase, value / _pl_n) for phase, value in sorted(_pl_totals.items())))))
                    else:
                        _pl_parts.append('model_ledger_unavailable=%s' % _pl_error)
                    sys.stderr.write('[prefill-ledger] ' + ' '.join(_pl_parts) + '\n')
                    sys.stderr.flush()
                except Exception:
                    pass
            sys.stderr.write(f'[mlx-server] prefill done in {prefill_dt:.2f}s = {tok_s_label:.1f} tok/s (delta)\n')
            sys.stderr.flush()
            first_id = token_sampler.sample_token(last_logits[0, -1, :])
            if _swi_active:
                swi_last_logits_2d = last_logits[:, -1, :]
            if mtp_hidden_chunks:
                full_hidden = mtp_hidden_chunks[0] if len(mtp_hidden_chunks) == 1 else mx.concatenate(mtp_hidden_chunks, axis=1)
                full_topk = _mtp_topk_concat(mx, mtp_topk_chunks) if mtp_topk_chunks else None
                mtp_reference_prev_hidden = full_hidden[:, -1:, :]
                mtp_reference_prev_topk = _mtp_topk_slice(full_topk, -1, None)
                _warm_rows = int(full_hidden.shape[1])
                _warm_ids = prompt_arr[None]
                if _warm_rows != int(_warm_ids.shape[1]):
                    _warm_ids = _warm_ids[:, -_warm_rows:]
                _warm_tail = _dsv41_cone_tail_rows()
                _warm_note = '' if _warm_tail is None else ' (DSV41_PREFILL_CONE_TAIL=%d)' % _warm_tail
                sys.stderr.write(f'[mlx-server] MTP warm: {_warm_rows} target hidden row(s) for {int(prompt_arr.shape[0])} prompt token(s){_warm_note}\n')
                sys.stderr.flush()
                if self._mtp_prompt_history == 'committed':
                    mtp_drafter.prefill_committed_prompt_history(_warm_ids, full_hidden, shared_topk=_mtp_topk_slice(full_topk, None, -1))
                    warm_msg = '[mlx-server] MTP draft diagnostics: warmed committed prompt history (no-bonus diagnostic)\n'
                else:
                    mtp_drafter.prefill_from_target_hidden(_warm_ids, full_hidden, first_id, shared_topk=full_topk, sampler=token_sampler)
                    warm_msg = f'[mlx-server] MTP draft diagnostics: warmed native shifted prompt state with bonus first_id={first_id}\n'
                sys.stderr.write(warm_msg)
                sys.stderr.flush()
        if mtp_can_run and _mtp_kv_inject and (mtp_hidden is not None):
            _n_h = int(mtp_hidden.shape[0])
            full_hidden = mx.array(mtp_hidden)[None]
            mtp_reference_prev_hidden = full_hidden[:, -1:, :]
            mtp_reference_prev_topk = None
            _ids = mx.array(prompt_token_ids[:_n_h])[None]
            if self._mtp_prompt_history == 'committed':
                mtp_drafter.prefill_committed_prompt_history(_ids, full_hidden, shared_topk=None)
            else:
                mtp_drafter.prefill_from_target_hidden(_ids, full_hidden, first_id, shared_topk=None, sampler=token_sampler)
            sys.stderr.write(f'[mlx-server] MTP warmed from shipped hidden ({_n_h}x{int(mtp_hidden.shape[1])}) first_id={first_id} mode={self._mtp_prompt_history} - KV-inject speculative decode enabled\n')
            sys.stderr.flush()

        def _step(tok_id_arr):
            with mx.stream(gen_stream):
                ids = tok_id_arr[None]
                if mtp_can_run:
                    logits, hidden, topk = target_logits_hidden_topk(lang, ids, cache)
                else:
                    out = lang(ids, cache=cache)
                    logits = out.logits if hasattr(out, 'logits') else out
                    hidden = None
                    topk = None
                logits = logits[:, -1, :]
                y = token_sampler.sample_array(logits)
                if mtp_can_run:
                    return (y, hidden[:, -1:, :] if hidden is not None else None, _mtp_topk_slice(topk, -1, None))
                return y
        decoded_text = ''
        tokens_generated = 0
        stop_reason = STOP_LIMIT
        stopping_word = ''
        decode_t0 = time.time()
        pos_base = n_prompt + len(_reuse_prefix_for_save or [])
        all_generated_ids = []
        mtp_token_budget = max(0, int(getattr(self, '_mtp_token_budget', 0)))

        def _emit_current(current_id: int) -> bool:
            nonlocal decoded_text, tokens_generated, stop_reason, stopping_word
            all_generated_ids.append(current_id)
            try:
                full_text = self.tokenizer.decode(all_generated_ids)
                tok_text = full_text[len(decoded_text):]
            except Exception:
                tok_text = ''
            if current_id in eos_ids:
                stop_reason = STOP_EOS
                return False
            if not _send_token(ls, current_id, tok_text, pos_base + tokens_generated):
                stop_reason = STOP_CANCELLED
                return False
            tokens_generated += 1
            decoded_text += tok_text
            hit_word = ''
            for sw in stop_words:
                if sw and decoded_text.endswith(sw):
                    hit_word = sw
                    break
            if hit_word:
                stop_reason = STOP_WORD
                stopping_word = hit_word
                return False
            if tokens_generated % 256 == 0 and _clear_cache_per_request():
                mx.clear_cache()
            return True

        def _mtp_remaining_after_current() -> int:
            remaining = max_tokens - tokens_generated - 1
            if mtp_token_budget > 0:
                remaining = min(remaining, mtp_token_budget - tokens_generated - 1)
            return max(0, int(remaining))

        def _mtp_budget_allows_step() -> bool:
            return mtp_token_budget <= 0 or tokens_generated < mtp_token_budget
        swi_embed = None
        if _swi_active:
            swi_embed = self._swi_get_embed(mx)
            if swi_embed is None or swi_last_logits_2d is None:
                if swi_last_logits_2d is None and swi_embed is not None:
                    sys.stderr.write('[mlx-server] SwiReasoning: no prompt-final logits captured (non local-prefill route); decoding normally\n')
                    sys.stderr.flush()
                _swi_active = False
        with _wired(self.model):
            _ledger_on = os.environ.get('DSV41_STEP_LEDGER', '0') == '1'
            _ledger = {'build_ms': 0.0, 'wait_ms': 0.0, 'emit_ms': 0.0, 'iter_ms': 0.0, 'n': 0}
            if dspark_runtime is not None:
                dspark_tokens = dspark_runtime.tokens(cache, first_id, max_tokens)
                try:
                    for token in dspark_tokens:
                        if not _emit_current(token):
                            break
                finally:
                    dspark_tokens.close()
                    mx.synchronize()
                    sys.stderr.write('[DSV41_DSPARK] ' + json.dumps(dspark_runtime.stats()) + chr(10))
                    sys.stderr.flush()
            elif is_dsv41:
                current_id = first_id
                while tokens_generated < max_tokens:
                    _it0 = time.perf_counter() if _ledger_on else 0.0
                    if not _emit_current(current_id):
                        break
                    _it1 = time.perf_counter() if _ledger_on else 0.0
                    if tokens_generated < max_tokens:
                        y = _step(mx.array([current_id]))
                        _it2 = time.perf_counter() if _ledger_on else 0.0
                        mx.eval(y, cache[0].state)
                        current_id = int(y.item())
                        if _ledger_on:
                            _it3 = time.perf_counter()
                            _ledger['emit_ms'] += (_it1 - _it0) * 1000.0
                            _ledger['build_ms'] += (_it2 - _it1) * 1000.0
                            _ledger['wait_ms'] += (_it3 - _it2) * 1000.0
                            _ledger['iter_ms'] += (_it3 - _it0) * 1000.0
                            _ledger['n'] += 1
            elif _swi_active:
                E = swi_embed['E']
                _think_emb = swi_embed['think']
                _endthink_emb = swi_embed['endthink']
                _nl_emb = swi_embed['nl']
                _has_specials = swi_embed['has_specials']
                lang_swi = getattr(self.model, 'language_model', self.model)
                swi = SwiState(cfg=swi_config, endthink_ids=list(swi_embed['endthink_ids']), convergence_ids=list(swi_embed['convergence_ids']), termination_ids=list(swi_embed['termination_ids']))
                _swi_n_predict = max(1, int(max_tokens))
                _swi_step_cap = int(os.environ.get('LLAMA_MLX_SWI_MAX_STEPS', str(max(4 * _swi_n_predict, _swi_n_predict + 2048))))

                def _swi_forward_ids(tok_id: int):
                    with mx.stream(gen_stream):
                        out = lang_swi(mx.array([[int(tok_id)]]), cache=cache)
                        lg = out.logits if hasattr(out, 'logits') else out
                    return lg[:, -1, :]

                def _swi_forward_emb(emb_row):
                    embd = emb_row.reshape(1, 1, -1).astype(E.dtype)
                    with mx.stream(gen_stream):
                        out = lang_swi(None, inputs_embeds=embd, cache=cache)
                        lg = out.logits if hasattr(out, 'logits') else out
                    return lg[:, -1, :]

                def _swi_soft_emb(probs_row):
                    return (probs_row.astype(E.dtype) @ E).reshape(-1)
                logits_row = swi_last_logits_2d
                step = 0
                swi_stop = False
                while tokens_generated < max_tokens and step < _swi_step_cap and (not swi_stop):
                    probs_row = mx.softmax(logits_row.astype(mx.float32), axis=-1)
                    _p = mx.clip(probs_row, 1e-12, None)
                    entropy = float((-mx.sum(probs_row * mx.log(_p), axis=-1)).item())
                    swi.update_mode(entropy)
                    swi.inject_triggers(step)
                    latent = swi.should_use_soft()
                    swi.count_step(latent)
                    if latent:
                        soft = _swi_soft_emb(probs_row)
                        if _has_specials:
                            if step == 0 and _nl_emb is not None:
                                soft = 0.9 * soft + 0.1 * _nl_emb.astype(soft.dtype)
                            elif swi.last_to_soft and _think_emb is not None:
                                a = swi_alpha(swi_config.alpha_0, step, _swi_n_predict)
                                soft = a * soft + (1.0 - a) * _think_emb.astype(soft.dtype)
                        logits_row = _swi_forward_emb(soft)
                        step += 1
                        continue
                    forced = swi.pop_forced()
                    if forced is not None:
                        tok = int(forced)
                    else:
                        tok = token_sampler.sample_token(logits_row)
                    if not _emit_current(tok):
                        _trim_cache_to_offset(cache, pos_base + tokens_generated)
                        break
                    swi.note_emitted(tok)
                    if not swi.locked_normal and '</think>' in decoded_text:
                        swi.force_lock()
                    if swi.budget_tick():
                        swi_stop = True
                        break
                    beta_blend = _has_specials and swi.last_to_normal and (step > 0) and (_endthink_emb is not None)
                    if beta_blend:
                        soft = _swi_soft_emb(probs_row)
                        b = swi_beta(swi_config.beta_0, step, _swi_n_predict)
                        emb = b * soft + (1.0 - b) * _endthink_emb.astype(soft.dtype)
                        logits_row = _swi_forward_emb(emb)
                    else:
                        logits_row = _swi_forward_ids(tok)
                    step += 1
                self._swi_last_stats = swi.stats()
                self._swi_last_stats['steps'] = step
                sys.stderr.write(f"[mlx-server] SwiReasoning done: {self._swi_last_stats['soft_steps']} soft / {self._swi_last_stats['normal_steps']} normal steps, switches={self._swi_last_stats['switch_count']}, final_mode={self._swi_last_stats['final_mode']}, emitted={tokens_generated}\n")
                sys.stderr.flush()
            elif mtp_can_run:
                current_id = first_id
                mtp_error_logged = False
                mtp_speculative = self._mtp_mode == 'speculative'
                mtp_verified_windows = 0
                mtp_verified_attempted = 0
                mtp_verified_match_sum = 0
                mtp_verified_prefix_sum = 0
                mtp_full_windows = 0
                mtp_full_window_match_sum = 0
                mtp_full_window_prefix_sum = 0
                mtp_depth_attempted = [0 for _ in range(max(1, self._mtp_max_draft))]
                mtp_depth_accepted = [0 for _ in range(max(1, self._mtp_max_draft))]
                mtp_depth_matched = [0 for _ in range(max(1, self._mtp_max_draft))]
                mtp_first_mismatch = [0 for _ in range(max(1, self._mtp_max_draft + 1))]
                mtp_lag_values = [-3, -2, -1, 0, 1, 2, 3]
                mtp_lag_attempted = [0 for _ in mtp_lag_values]
                mtp_lag_matched = [0 for _ in mtp_lag_values]
                mtp_topk_prev_available = 0
                mtp_topk_seeded_rounds = 0
                mtp_adaptive_ctrl = make_adaptive_depth_controller(self._mtp_adaptive_depth, self._mtp_max_draft)
                while tokens_generated < max_tokens:
                    if mtp_speculative and self._mtp_schedule == 'reference' and _mtp_budget_allows_step():
                        accepted_count = 0
                        verified_accept_count = 0
                        draft_tokens = []
                        emit_after_current = []
                        next_current_id = current_id
                        remaining_after_current = _mtp_remaining_after_current()
                        current_prev_hidden = mtp_reference_prev_hidden
                        current_prev_topk = mtp_reference_prev_topk
                        if remaining_after_current > 0 and current_prev_hidden is not None and (not mtp_error_logged):
                            verify_offset = _cache_offset(cache)
                            mtp_offset = mtp_drafter.cache_offset()
                            mtp_topk_snapshot = mtp_drafter.shared_topk_snapshot()
                            had_seed = bool(getattr(mtp_drafter, 'has_seed', False))
                            try:
                                draft_limit = min(mtp_adaptive_ctrl.next_depth() if mtp_adaptive_ctrl is not None else self._mtp_max_draft, remaining_after_current)
                                if current_prev_topk is not None:
                                    mtp_topk_prev_available += 1
                                seeded_topk = _mtp_prepare_reference_topk(mtp_drafter, current_prev_topk)
                                if seeded_topk:
                                    mtp_topk_seeded_rounds += 1
                                draft_tokens, draft_logits, _draft_hidden = mtp_drafter.propose_many(current_id, current_prev_hidden, draft_limit, reuse_existing_topk=seeded_topk, sampler=token_sampler, return_tensors=True)
                                if draft_tokens:
                                    verify_input_tokens = [current_id, *draft_tokens]
                                    verify_logits, verify_hidden, verify_topk = _forward_verify_token_ids(mx.array(verify_input_tokens)[None])
                                    decision = _mtp_speculative_sampling_decision(mx, token_sampler, draft_tokens, draft_logits, verify_logits, self._mtp_target_token_eval)
                                    target_tokens = decision['target_greedy']
                                    compare_tokens = target_tokens[:len(draft_tokens)]
                                    if self._mtp_verify_hidden_eval == 'eager':
                                        eval_values = [verify_hidden]
                                        if verify_topk is not None:
                                            eval_values.append(verify_topk)
                                        mx.eval(*eval_values)
                                    match_count = sum((1 for draft_id, target_id in zip(draft_tokens, compare_tokens) if int(draft_id) == int(target_id)))
                                    accepted_count = int(decision['accepted_count'])
                                    if mtp_adaptive_ctrl is not None:
                                        mtp_adaptive_ctrl.observe(accepted_count, len(draft_tokens))
                                    accepted_flags = list(decision.get('accepted_flags', []))
                                    for idx, (draft_id, target_id) in enumerate(zip(draft_tokens, compare_tokens)):
                                        mtp_drafter.record_verification(draft_id, target_id, accepted=accepted_flags[idx] if idx < len(accepted_flags) else False)
                                    _mtp_update_depth_stats(draft_tokens, compare_tokens, mtp_depth_attempted, mtp_depth_matched, mtp_first_mismatch)
                                    _mtp_update_lag_stats(draft_tokens, compare_tokens, mtp_lag_values, mtp_lag_attempted, mtp_lag_matched)
                                    next_current_id = int(decision['next_token'])
                                    emit_after_current = [int(t) for t in draft_tokens[:accepted_count]]
                                    committed_after_current = 1 + accepted_count
                                    _trim_cache_to_offset(cache, verify_offset + committed_after_current)
                                    mtp_reference_prev_hidden = verify_hidden[:, accepted_count:accepted_count + 1, :]
                                    mtp_reference_prev_topk = _mtp_topk_slice(verify_topk, accepted_count, accepted_count + 1)
                                    if self._mtp_debug_windows and mtp_verified_windows < self._mtp_debug_windows:
                                        try:
                                            draft_text = self.tokenizer.decode(list(draft_tokens))
                                        except Exception:
                                            draft_text = ''
                                        try:
                                            target_text = self.tokenizer.decode(list(compare_tokens))
                                        except Exception:
                                            target_text = ''
                                        q_target_ranks, p_draft_ranks = _mtp_debug_rank_stats(mx, draft_logits, verify_logits, draft_tokens, compare_tokens)
                                        sys.stderr.write(f"[mlx-server] MTP debug window {mtp_verified_windows + 1}: schedule=reference had_seed={had_seed} prev_topk={current_prev_topk is not None} seeded_topk={seeded_topk} verify_offset={verify_offset} mtp_offset={mtp_offset} current={current_id} target_next={target_tokens[0]} accepted={accepted_count}/{len(draft_tokens)} canonical={not token_sampler.greedy} residual={decision.get('residual_sampled', False)} draft={draft_tokens} target={compare_tokens} bonus={next_current_id} q_target_rank={q_target_ranks} p_draft_rank={p_draft_ranks} draft_text={draft_text!r} target_text={target_text!r}\n")
                                        sys.stderr.flush()
                                    verified_accept_count = accepted_count
                                    mtp_verified_windows += 1
                                    mtp_verified_attempted += len(draft_tokens)
                                    mtp_verified_match_sum += match_count
                                    mtp_verified_prefix_sum += verified_accept_count
                                    if len(draft_tokens) == self._mtp_max_draft:
                                        mtp_full_windows += 1
                                        mtp_full_window_match_sum += match_count
                                        mtp_full_window_prefix_sum += verified_accept_count
                                    for depth_idx in range(min(verified_accept_count, len(mtp_depth_accepted))):
                                        mtp_depth_accepted[depth_idx] += 1
                                    if self._mtp_cache_commit == 'native':
                                        mtp_new_tokens = [int(t) for t in draft_tokens[:accepted_count]]
                                        if accepted_count < remaining_after_current:
                                            mtp_new_tokens.append(int(next_current_id))
                                        mtp_drafter.accept_verified_tokens(verify_hidden, mx.array(draft_tokens)[None], accepted_count, mtp_new_tokens, sampler=token_sampler)
                                    else:
                                        mtp_drafter.rollback_cache(mtp_offset)
                                        mtp_drafter.restore_shared_topk(mtp_topk_snapshot)
                                        history_tokens = _mtp_reference_history_tokens(current_id, draft_tokens, accepted_count, current_already_seeded=had_seed)
                                        if had_seed:
                                            history_hidden = verify_hidden[:, :accepted_count, :] if accepted_count else None
                                            history_topk = _mtp_topk_slice(verify_topk, 0, accepted_count)
                                        else:
                                            history_hidden = current_prev_hidden
                                            history_topk = current_prev_topk
                                        if accepted_count and (not had_seed):
                                            history_hidden = mx.concatenate([current_prev_hidden, verify_hidden[:, :accepted_count, :]], axis=1)
                                            history_topk = _mtp_topk_concat(mx, [current_prev_topk, _mtp_topk_slice(verify_topk, 0, accepted_count)])
                                        if history_tokens and history_hidden is not None:
                                            mtp_drafter.append_committed_from_previous_hidden(mx.array(history_tokens)[None], history_hidden, shared_topk=history_topk, update_seed=False)
                                    if mtp_drafter.checked and mtp_drafter.checked % 16 == 0:
                                        sys.stderr.write(f'[mlx-server] MTP speculative: {mtp_drafter.accepted}/{mtp_drafter.checked} accepted ({mtp_drafter.acceptance_rate:.1%}), last_window={verified_accept_count}/{len(draft_tokens)}\n')
                                        sys.stderr.flush()
                                else:
                                    y, current_hidden, current_topk = _step(mx.array([current_id]))
                                    if current_hidden is not None:
                                        eval_values = [y, current_hidden]
                                        if current_topk is not None:
                                            eval_values.append(current_topk)
                                        mx.eval(*eval_values)
                                    else:
                                        mx.eval(y)
                                    next_current_id = int(y.item())
                                    mtp_reference_prev_hidden = current_hidden
                                    mtp_reference_prev_topk = current_topk
                                    mtp_drafter.rollback_cache(mtp_offset)
                                    mtp_drafter.restore_shared_topk(mtp_topk_snapshot)
                                    mtp_drafter.append_committed_from_previous_hidden(mx.array([[int(current_id)]]), current_prev_hidden, shared_topk=current_prev_topk, update_seed=False)
                            except Exception as mtp_err:
                                draft_tokens = []
                                emit_after_current = []
                                next_current_id = current_id
                                _trim_cache_to_offset(cache, verify_offset)
                                try:
                                    mtp_drafter.rollback_cache(mtp_offset)
                                    mtp_drafter.restore_shared_topk(mtp_topk_snapshot)
                                except Exception:
                                    pass
                                mtp_error_logged = True
                                sys.stderr.write(f'[mlx-server] MTP disabled after error: {mtp_err}\n')
                                sys.stderr.flush()
                        if (remaining_after_current <= 0 or current_prev_hidden is None or mtp_error_logged) and stop_reason == STOP_LIMIT and (not draft_tokens):
                            if remaining_after_current > 0:
                                y, current_hidden, current_topk = _step(mx.array([current_id]))
                                if current_hidden is not None:
                                    eval_values = [y, current_hidden]
                                    if current_topk is not None:
                                        eval_values.append(current_topk)
                                    mx.eval(*eval_values)
                                else:
                                    mx.eval(y)
                                next_current_id = int(y.item())
                                mtp_reference_prev_hidden = current_hidden
                                mtp_reference_prev_topk = current_topk
                        if not _emit_current(current_id):
                            _trim_cache_to_offset(cache, pos_base + tokens_generated)
                            break
                        for draft_id in emit_after_current:
                            if tokens_generated >= max_tokens:
                                break
                            if not _emit_current(int(draft_id)):
                                _trim_cache_to_offset(cache, pos_base + tokens_generated)
                                break
                        if stop_reason != STOP_LIMIT:
                            break
                        current_id = next_current_id
                        continue
                    y, current_hidden, current_topk = _step(mx.array([current_id]))
                    if current_hidden is not None:
                        eval_values = [y, current_hidden]
                        if current_topk is not None:
                            eval_values.append(current_topk)
                        mx.eval(*eval_values)
                    else:
                        mx.eval(y)
                    next_id_from_y = int(y.item())
                    next_current_id = next_id_from_y
                    accepted_count = 0
                    verified_accept_count = 0
                    draft_tokens = []
                    emit_after_current = []
                    remaining_after_current = max_tokens - tokens_generated - 1
                    if mtp_token_budget > 0:
                        remaining_after_current = min(remaining_after_current, mtp_token_budget - tokens_generated - 1)
                    if remaining_after_current > 0 and current_hidden is not None and (not mtp_error_logged) and _mtp_budget_allows_step():
                        verify_offset = _cache_offset(cache)
                        mtp_offset = mtp_drafter.cache_offset()
                        mtp_topk_snapshot = mtp_drafter.shared_topk_snapshot()
                        had_seed = bool(getattr(mtp_drafter, 'has_seed', False))
                        try:
                            proposal_current_id = current_id
                            proposal_hidden = current_hidden
                            verify_input_tokens = None
                            target_tokens = []
                            if mtp_speculative and remaining_after_current > 1:
                                if current_topk is not None:
                                    mtp_topk_prev_available += 1
                                mtp_drafter.append_committed_from_previous_hidden(mx.array([[next_id_from_y]]), current_hidden, shared_topk=current_topk, preserve_topk=True, sampler=token_sampler)
                                if mtp_drafter.shared_topk_snapshot() is not None:
                                    mtp_topk_seeded_rounds += 1
                                proposal_current_id = next_id_from_y
                                proposal_hidden = current_hidden
                                draft_limit = min(self._mtp_max_draft, remaining_after_current - 1)
                            else:
                                draft_limit = min(self._mtp_max_draft, remaining_after_current)
                            draft_tokens, draft_logits, _draft_hidden = mtp_drafter.propose_many(proposal_current_id, proposal_hidden, draft_limit, reuse_existing_topk=mtp_speculative and remaining_after_current > 1, sampler=token_sampler, return_tensors=True)
                            if draft_tokens:
                                if mtp_speculative and remaining_after_current > 1:
                                    verify_input_tokens = [next_id_from_y, *draft_tokens]
                                else:
                                    verify_input_tokens = list(draft_tokens)
                                verify_logits, verify_hidden, verify_topk = _forward_verify_token_ids(mx.array(verify_input_tokens)[None])
                                if mtp_speculative and remaining_after_current > 1:
                                    decision = _mtp_speculative_sampling_decision(mx, token_sampler, draft_tokens, draft_logits, verify_logits, self._mtp_target_token_eval)
                                    target_tokens = decision['target_greedy']
                                    compare_tokens = target_tokens[:len(draft_tokens)]
                                else:
                                    decision = None
                                    target_tokens = [next_id_from_y]
                                    target_tokens.extend(_mtp_target_tokens_from_logits(mx, verify_logits, len(draft_tokens) - 1, self._mtp_target_token_eval))
                                    compare_tokens = target_tokens
                                if self._mtp_verify_hidden_eval == 'eager':
                                    eval_values = [verify_hidden]
                                    if verify_topk is not None:
                                        eval_values.append(verify_topk)
                                    mx.eval(*eval_values)
                                match_count = sum((1 for draft_id, target_id in zip(draft_tokens, compare_tokens) if int(draft_id) == int(target_id)))
                                if decision is not None:
                                    accepted_count = int(decision['accepted_count'])
                                    accepted_flags = list(decision.get('accepted_flags', []))
                                    for idx, (draft_id, target_id) in enumerate(zip(draft_tokens, compare_tokens)):
                                        mtp_drafter.record_verification(draft_id, target_id, accepted=accepted_flags[idx] if idx < len(accepted_flags) else False)
                                else:
                                    for draft_id, target_id in zip(draft_tokens, compare_tokens):
                                        mtp_drafter.record_verification(draft_id, target_id)
                                    accepted_count = _mtp_accept_prefix(draft_tokens, compare_tokens)
                                _mtp_update_depth_stats(draft_tokens, compare_tokens, mtp_depth_attempted, mtp_depth_matched, mtp_first_mismatch)
                                _mtp_update_lag_stats(draft_tokens, compare_tokens, mtp_lag_values, mtp_lag_attempted, mtp_lag_matched)
                                if mtp_speculative and remaining_after_current > 1:
                                    next_current_id = int(decision['next_token'])
                                    emit_after_current = [int(next_id_from_y), *[int(t) for t in draft_tokens[:accepted_count]]]
                                else:
                                    bonus_idx = accepted_count
                                    if bonus_idx == 0:
                                        next_current_id = next_id_from_y
                                    else:
                                        next_current_id = token_sampler.sample_token(verify_logits[0, bonus_idx - 1, :])
                                if mtp_speculative:
                                    committed_after_current = 1 + accepted_count if remaining_after_current > 1 else accepted_count
                                    _trim_cache_to_offset(cache, verify_offset + committed_after_current)
                                else:
                                    _trim_cache_to_offset(cache, verify_offset)
                                    accepted_count = 0
                                    next_current_id = next_id_from_y
                                target_debug_tokens = compare_tokens if mtp_speculative and remaining_after_current > 1 else target_tokens
                                bonus_debug = target_tokens[accepted_count] if mtp_speculative and remaining_after_current > 1 and target_tokens else next_current_id
                                if self._mtp_debug_windows and mtp_verified_windows < self._mtp_debug_windows:
                                    try:
                                        draft_text = self.tokenizer.decode(list(draft_tokens))
                                    except Exception:
                                        draft_text = ''
                                    try:
                                        target_text = self.tokenizer.decode(list(target_debug_tokens))
                                    except Exception:
                                        target_text = ''
                                    q_target_ranks, p_draft_ranks = _mtp_debug_rank_stats(mx, draft_logits, verify_logits, draft_tokens, target_debug_tokens)
                                    sys.stderr.write(f"[mlx-server] MTP debug window {mtp_verified_windows + 1}: had_seed={had_seed} prev_topk={current_topk is not None} seeded_topk={mtp_drafter.shared_topk_snapshot() is not None} verify_offset={verify_offset} mtp_offset={mtp_offset} current={current_id} target_next={next_id_from_y} accepted={accepted_count}/{len(draft_tokens)} canonical={bool(decision is not None and (not token_sampler.greedy))} residual={bool(decision and decision.get('residual_sampled', False))} draft={draft_tokens} target={target_debug_tokens} bonus={bonus_debug} q_target_rank={q_target_ranks} p_draft_rank={p_draft_ranks} draft_text={draft_text!r} target_text={target_text!r}\n")
                                    sys.stderr.flush()
                                verified_accept_count = accepted_count
                                mtp_verified_windows += 1
                                mtp_verified_attempted += len(draft_tokens)
                                mtp_verified_match_sum += match_count
                                mtp_verified_prefix_sum += verified_accept_count
                                if len(draft_tokens) == self._mtp_max_draft:
                                    mtp_full_windows += 1
                                    mtp_full_window_match_sum += match_count
                                    mtp_full_window_prefix_sum += verified_accept_count
                                for depth_idx in range(min(verified_accept_count, len(mtp_depth_accepted))):
                                    mtp_depth_accepted[depth_idx] += 1
                                mtp_drafter.rollback_cache(mtp_offset)
                                mtp_drafter.restore_shared_topk(mtp_topk_snapshot)
                                history_tokens = []
                                history_hidden = []
                                history_topk = None
                                if mtp_speculative and remaining_after_current > 1:
                                    history_tokens.append(int(next_id_from_y))
                                    history_hidden.append(current_hidden)
                                    history_topk_pieces = [current_topk]
                                    if accepted_count:
                                        history_tokens.extend((int(t) for t in draft_tokens[:accepted_count]))
                                        history_hidden.append(verify_hidden[:, :accepted_count, :])
                                        history_topk_pieces.append(_mtp_topk_slice(verify_topk, 0, accepted_count))
                                    history_tokens.append(int(next_current_id))
                                    history_hidden.append(verify_hidden[:, accepted_count:accepted_count + 1, :])
                                    history_topk_pieces.append(_mtp_topk_slice(verify_topk, accepted_count, accepted_count + 1))
                                    history_topk = _mtp_topk_concat(mx, history_topk_pieces)
                                else:
                                    if not had_seed:
                                        history_tokens.append(int(current_id))
                                        history_hidden.append(current_hidden)
                                        history_topk = current_topk
                                    if mtp_speculative and accepted_count:
                                        history_tokens.extend((int(t) for t in draft_tokens[:accepted_count]))
                                        accepted_hidden = [current_hidden]
                                        if accepted_count > 1:
                                            accepted_hidden.append(verify_hidden[:, :accepted_count - 1, :])
                                        history_hidden.append(accepted_hidden[0] if len(accepted_hidden) == 1 else mx.concatenate(accepted_hidden, axis=1))
                                    bonus_prev_hidden = current_hidden if accepted_count <= 0 else verify_hidden[:, accepted_count - 1:accepted_count, :]
                                    history_tokens.append(int(next_current_id))
                                    history_hidden.append(bonus_prev_hidden)
                                    bonus_prev_topk = current_topk if accepted_count <= 0 else _mtp_topk_slice(verify_topk, accepted_count - 1, accepted_count)
                                    history_topk = _mtp_topk_concat(mx, [history_topk, bonus_prev_topk])
                                if history_tokens and history_hidden:
                                    hist_hidden = history_hidden[0] if len(history_hidden) == 1 else mx.concatenate(history_hidden, axis=1)
                                    mtp_drafter.append_committed_from_previous_hidden(mx.array(history_tokens)[None], hist_hidden, shared_topk=history_topk, sampler=token_sampler)
                                if mtp_drafter.checked and mtp_drafter.checked % 16 == 0:
                                    label = 'MTP speculative' if mtp_speculative else 'MTP diagnostic'
                                    sys.stderr.write(f'[mlx-server] {label}: {mtp_drafter.accepted}/{mtp_drafter.checked} accepted ({mtp_drafter.acceptance_rate:.1%}), last_window={verified_accept_count}/{len(draft_tokens)}\n')
                                    sys.stderr.flush()
                            if not draft_tokens and mtp_speculative and (remaining_after_current > 1):
                                mtp_drafter.rollback_cache(mtp_offset)
                                mtp_drafter.restore_shared_topk(mtp_topk_snapshot)
                        except Exception as mtp_err:
                            _trim_cache_to_offset(cache, verify_offset)
                            try:
                                mtp_drafter.rollback_cache(mtp_offset)
                                mtp_drafter.restore_shared_topk(mtp_topk_snapshot)
                            except Exception:
                                pass
                            mtp_error_logged = True
                            sys.stderr.write(f'[mlx-server] MTP disabled after error: {mtp_err}\n')
                            sys.stderr.flush()
                    if not _emit_current(current_id):
                        _trim_cache_to_offset(cache, pos_base + tokens_generated)
                        break
                    for draft_id in emit_after_current:
                        if tokens_generated >= max_tokens:
                            break
                        if not _emit_current(int(draft_id)):
                            _trim_cache_to_offset(cache, pos_base + tokens_generated)
                            break
                    if stop_reason != STOP_LIMIT:
                        break
                    current_id = next_current_id
            elif max_tokens > 0:
                y_in = mx.array([first_id])
                y = _step(y_in)
                mx.async_eval(y)
                n = 0
                current_id = first_id
                while n < max_tokens:
                    _it0 = time.perf_counter() if _ledger_on else 0.0
                    if n + 1 < max_tokens:
                        next_y = _step(y)
                        mx.async_eval(next_y)
                    _it1 = time.perf_counter() if _ledger_on else 0.0
                    next_id_from_y = int(y.item())
                    _it2 = time.perf_counter() if _ledger_on else 0.0
                    _emitted_ok = _emit_current(current_id)
                    if _ledger_on:
                        _it3 = time.perf_counter()
                        _ledger['build_ms'] += (_it1 - _it0) * 1000.0
                        _ledger['wait_ms'] += (_it2 - _it1) * 1000.0
                        _ledger['emit_ms'] += (_it3 - _it2) * 1000.0
                        _ledger['iter_ms'] += (_it3 - _it0) * 1000.0
                        _ledger['n'] += 1
                    if not _emitted_ok:
                        break
                    current_id = next_id_from_y
                    y = next_y if n + 1 < max_tokens else y
                    n += 1
        decode_dt = time.time() - decode_t0
        decode_tok_s = tokens_generated / max(decode_dt, 1e-09)
        if os.environ.get('DSV41_STEP_LEDGER', '0') == '1':
            try:
                _lg = locals().get('_ledger') or {}
                _n = max(1, int(_lg.get('n', 0)))
                _parts = ['tokens=%d' % _lg.get('n', 0)]
                for _k in ('iter_ms', 'build_ms', 'wait_ms', 'emit_ms'):
                    _parts.append('%s/tok=%.2f' % (_k, float(_lg.get(_k, 0.0)) / _n))
                try:
                    from mlx_vlm.models.deepseek_v41 import engram as _dsv41_engram, language as _dsv41_language
                    _el = _dsv41_engram.LEDGER
                    _ll = _dsv41_language.LEDGER
                    _parts.append('engram_host_ms/call=%.3f calls=%d' % (_el['engram_host_ms'] / max(1, _el['engram_calls']), _el['engram_calls']))
                    _parts.append('hash_host_ms/call=%.3f calls=%d' % (_ll['hash_host_ms'] / max(1, _ll['hash_calls']), _ll['hash_calls']))
                    if _ll.get('cut_steps'):
                        _steps = _ll['cut_steps']
                        _rows = sorted(_dsv41_language.LEDGER_LAYER_MS.items())
                        _parts.append('cut_steps=%d layer_ms/step=[%s]' % (_steps, ' '.join(('%d:%.2f' % (_i, _v / _steps) for _i, _v in _rows))))
                    for _d in (_el, _ll):
                        for _k in list(_d):
                            _d[_k] = 0 if isinstance(_d[_k], int) else 0.0
                    _dsv41_language.LEDGER_LAYER_MS.clear()
                except Exception as _lexc:
                    _parts.append('model_ledger_unavailable=%s' % type(_lexc).__name__)
                sys.stderr.write('[ledger] ' + ' '.join(_parts) + '\n')
                sys.stderr.flush()
            except Exception:
                pass
        if mtp_can_run and mtp_drafter.checked:
            label = 'MTP speculative' if self._mtp_mode == 'speculative' else 'MTP diagnostic'
            match_avg = mtp_verified_match_sum / max(mtp_verified_windows, 1)
            prefix_avg = mtp_verified_prefix_sum / max(mtp_verified_windows, 1)
            full_window_match_avg = mtp_full_window_match_sum / max(mtp_full_windows, 1)
            full_window_avg = mtp_full_window_prefix_sum / max(mtp_full_windows, 1)
            depth_rates = []
            for accepted_n, attempted_n in zip(mtp_depth_accepted, mtp_depth_attempted):
                depth_rates.append('-' if attempted_n <= 0 else f'{accepted_n}/{attempted_n}')
            depth_match_rates = []
            for matched_n, attempted_n in zip(mtp_depth_matched, mtp_depth_attempted):
                depth_match_rates.append('-' if attempted_n <= 0 else f'{matched_n}/{attempted_n}')
            mismatch_rates = [str(count) for count in mtp_first_mismatch]
            lag_rates = []
            for lag, matched_n, attempted_n in zip(mtp_lag_values, mtp_lag_matched, mtp_lag_attempted):
                rate = '-' if attempted_n <= 0 else f'{matched_n}/{attempted_n}'
                lag_rates.append(f'{lag}:{rate}')
            sys.stderr.write(f"[mlx-server] {label} final: {mtp_drafter.accepted}/{mtp_drafter.checked} accepted ({mtp_drafter.acceptance_rate:.1%}), match_avg={match_avg:.2f}/{self._mtp_max_draft}, prefix_avg={prefix_avg:.2f}/{self._mtp_max_draft}, full_match_avg={full_window_match_avg:.2f}/{self._mtp_max_draft} full_window_avg={full_window_avg:.2f}/{self._mtp_max_draft} windows={mtp_verified_windows} full_windows={mtp_full_windows} topk_prev={mtp_topk_prev_available}/{mtp_verified_windows} topk_seeded={mtp_topk_seeded_rounds}/{mtp_verified_windows} depth_accept=[{','.join(depth_rates)}] depth_match=[{','.join(depth_match_rates)}] first_mismatch=[{','.join(mismatch_rates)}] lag_match=[{','.join(lag_rates)}]\n")
            sys.stderr.flush()
        _committed_generated_ids = list(all_generated_ids[:tokens_generated])
        _ship_image_persistable = _ship_image and bool(_img_hashes) and (self._prompt_store is not None) and self._prompt_store.enabled
        if _swi_active:
            sys.stderr.write('[mlx-server] cache-no-retain: SwiReasoning session (latent cells), skipping retention\n')
            sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []
        elif _should_retain_injected_state(_mac_retain_handle_enabled, stop_reason, _kv_inject_succeeded, _ship_image, _swi_active):
            self._save_retained_state(cache, prompt_token_ids, _committed_generated_ids)
            sys.stderr.write(f'[mlx-server] cache-retain (KV-inject): prompt={len(prompt_token_ids)} generated={len(all_generated_ids)} retained={len(self._retained_tokens)} offset={_cache_offset(self._retained_cache)}\n')
            sys.stderr.flush()
        elif _mac_retain_handle_enabled and _kv_inject_was_requested and (not _kv_inject_succeeded):
            self._retained_cache = None
            self._retained_tokens = []
        elif stop_reason != STOP_CANCELLED and (not kv_state_bytes) and (not _ship_image):
            _effective_prompt = _reuse_prefix_for_save + list(prompt_token_ids) if _reuse_prefix_for_save is not None else prompt_token_ids
            persisted = False
            if self._prompt_store is not None and self._prompt_store.enabled:
                try:
                    if _trim_cache_to_offset(cache, len(_effective_prompt)):
                        persisted = self._prompt_store.save(cache, _effective_prompt, _committed_generated_ids)
                except Exception as e:
                    sys.stderr.write(f'[mlx-server] prompt-cache save failed: {e}\n')
                    sys.stderr.flush()
                    persisted = False
            if persisted:
                self._save_retained_state(cache, _effective_prompt, [])
            else:
                self._save_retained_state(cache, _effective_prompt, _committed_generated_ids)
        elif stop_reason != STOP_CANCELLED and (not kv_state_bytes) and _ship_image_persistable:
            try:
                boundary = len(prompt_token_ids) + sum((p[1] for p in _img_positions))
                ok = _trim_cache_to_offset(cache, boundary) and self._prompt_store.save(cache, prompt_token_ids, _committed_generated_ids, image_hashes=_img_hashes, image_positions=_img_positions)
            except Exception as e:
                sys.stderr.write(f'[mlx-server] prompt-cache image save failed: {e}\n')
                sys.stderr.flush()
                ok = False
            sys.stderr.write(f"[mlx-server] ship-embd disk persist {('OK' if ok else 'skipped')} (image-keyed); in-RAM retention skipped (text-space path)\n")
            sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []
        elif kv_state_bytes or _ship_image:
            sys.stderr.write('[mlx-server] cache-no-retain: injected/ship-embd session, skipping retention\n')
            sys.stderr.flush()
            self._retained_cache = None
            self._retained_tokens = []
        try:
            mx.eval(*(c.state for c in cache if hasattr(c, 'state')))
        except Exception:
            pass
        if _clear_cache_per_request():
            mx.clear_cache()
        if is_dsv41:
            if stop_reason == STOP_CANCELLED:
                self._retained_cache = None
                self._retained_tokens = []
            self._dsv41_completion = _dsv41_parse_completion(decoded_text, self._dsv41_thinking_mode)
            self._dsv41_requests += 1
            print(f'[DSV41] prefill={tok_s_label:.3f} tok/s decode={decode_tok_s:.3f} tok/s resident={mx.get_active_memory() / (1 << 30):.2f} GiB engram={json.dumps(lang.engram_stats())}', file=sys.stderr, flush=True)
            if self._dsv41_requests % 50 == 0:
                _dsv41_save_hot(self)
        return (tokens_generated, decode_tok_s, stop_reason, stopping_word)

def _serve_one_session(listen_sock: socket.socket, session: MLXSession, model_fp: int, kva_path: str, max_tokens: int, chunk_size: int, accept_timeout_s: float, kva_path_shared: Optional[list]=None) -> bool:
    listen_sock.settimeout(accept_timeout_s)
    try:
        client, addr = listen_sock.accept()
    except (socket.timeout, TimeoutError):
        sys.stderr.write('[mlx-server] no client connected, retrying...\n')
        sys.stderr.flush()
        return True
    if kva_path_shared is not None:
        latest = kva_path_shared[0]
        if latest is not None and latest is not kva_path:
            sys.stderr.write(f'[mlx-server] fresh KVA already present at accept: {_describe_kva_ref(latest)}\n')
            sys.stderr.flush()
            kva_path = latest
    ls = LineSocket(client)
    try:
        cfg_line = ls.read_line(timeout_s=10.0)
        if not cfg_line:
            return True
        client.settimeout(600.0)
        stop_words: List[str] = []
        config_max_tokens = -1
        config_prompt_token_count = 0
        config_prompt_hash = 0
        config_reuse_retained_count = 0
        config_reuse_retained_prefix_hash = 0
        config_sampling_profile: Dict[str, Any] = {}
        config_prompt_token_ids = None
        config_swi_config = None
        cfg = {}
        if cfg_line:
            try:
                cfg = json.loads(cfg_line)
                ls.configure_diagnostics(cfg)
                stop_words = list(cfg.get('stop_words', []) or [])
                config_max_tokens = int(cfg.get('max_tokens', -1))
                config_prompt_token_count = int(cfg.get('prompt_token_count', 0) or 0)
                config_prompt_hash = int(cfg.get('prompt_hash', 0) or 0)
                if os.environ.get('LLAMA_DSV4_MAC_RETAIN_HANDLE', '') not in ('', '0'):
                    config_reuse_retained_count = int(cfg.get('reuse_retained_count', 0) or 0)
                    config_reuse_retained_prefix_hash = int(cfg.get('reuse_retained_prefix_hash', 0) or 0) & 9223372036854775807
                config_sampling_profile = _sampling_profile_from_config(cfg)
                config_prompt_token_ids = cfg.get('prompt_token_ids')
                _swi_raw = cfg.get('swi_reasoning')
                if _swi_raw is None and isinstance(cfg.get('extra_body'), dict):
                    _swi_raw = cfg['extra_body'].get('swi_reasoning')
                if _swi_raw is not None:
                    _parsed = SwiConfig.from_dict(_swi_raw)
                    if _parsed.enabled:
                        config_swi_config = _parsed
            except json.JSONDecodeError as e:
                sys.stderr.write(f'[mlx-server] bad decode_config: {e}\n')
        sys.stderr.write(f'[mlx-server] stop_words={stop_words} config_max_tokens={config_max_tokens} config_prompt_tokens={config_prompt_token_count} config_prompt_hash=0x{config_prompt_hash:016x}\n')
        sys.stderr.flush()
        if _is_dsv41(session.model):
            extra_body = cfg.get('extra_body') or {}
            thinking = cfg.get('enable_thinking', extra_body.get('enable_thinking', False))
            budget = cfg.get('thinking_budget', extra_body.get('thinking_budget'))
            session._dsv41_thinking_mode = 'thinking' if thinking or (budget is not None and int(budget) > 0) else 'chat'
            session._dsv41_completion = None
            session._dsv41_vision_prompt = None
            if cfg.get('prompt') and (not cfg.get('messages')):
                cfg['messages'] = [{'role': 'user', 'content': cfg['prompt']}]
            if cfg.get('messages'):
                messages = [dict(message) for message in cfg['messages']]
                if cfg.get('tools'):
                    messages[0]['tools'] = cfg['tools']
                config_prompt_token_ids = session._dsv41_encode_messages(messages, cfg.get('reasoning_effort', extra_body.get('reasoning_effort')))
                config_prompt_token_count = 0
                config_prompt_hash = 0
            if config_prompt_token_ids:
                kva_path_shared = None
        if config_prompt_token_ids:
            from make_prompt_direct_kva import build_prompt_direct_kva
            kva_path = build_prompt_direct_kva('', token_ids=[int(t) for t in config_prompt_token_ids], requested_n_predict=config_max_tokens)
            sys.stderr.write(f'[mlx-server] per-request prompt: {len(config_prompt_token_ids)} tokens -> prompt-direct (shim mode)\n')
            sys.stderr.flush()
        if kva_path_shared is not None and (config_prompt_hash or config_prompt_token_count):
            import time as _time
            _wait_s = float(os.environ.get('LLAMA_MLX_KVA_WAIT_S', '600'))
            _poll_s = 0.05
            _deadline = _time.monotonic() + _wait_s
            _last_logged_ref = id(kva_path)
            sys.stderr.write(f'[mlx-server] waiting up to {_wait_s:.0f}s for KVA matching config_hash=0x{config_prompt_hash:016x} count={config_prompt_token_count}\n')
            sys.stderr.flush()
            while _time.monotonic() < _deadline:
                candidate = kva_path_shared[0]
                if candidate is not None:
                    try:
                        cand_n, cand_hash, cand_img = _kva_prompt_count_hash(candidate)
                        if cand_img > 0:
                            ok_count = config_prompt_token_count == 0 or cand_n + cand_img == config_prompt_token_count
                            ok_hash = True
                        else:
                            ok_count = config_prompt_token_count == 0 or cand_n == config_prompt_token_count
                            ok_hash = config_prompt_hash == 0 or cand_hash == config_prompt_hash
                        if ok_count and ok_hash:
                            if id(candidate) != _last_logged_ref:
                                sys.stderr.write(f'[mlx-server] KVA matched config: {_describe_kva_ref(candidate)} count={cand_n} hash=0x{cand_hash:016x}\n')
                                sys.stderr.flush()
                            kva_path = candidate
                            break
                        if id(candidate) != _last_logged_ref:
                            sys.stderr.write(f"[mlx-server] KVA candidate count={cand_n} hash=0x{cand_hash:016x} doesn't match yet, polling…\n")
                            sys.stderr.flush()
                            _last_logged_ref = id(candidate)
                    except Exception as _e:
                        if id(candidate) != _last_logged_ref:
                            sys.stderr.write(f'[mlx-server] KVA parse retry: {_e}\n')
                            sys.stderr.flush()
                            _last_logged_ref = id(candidate)
                _time.sleep(_poll_s)
            else:
                sys.stderr.write(f"[mlx-server] KVA wait timed out after {_wait_s:.0f}s; proceeding with whatever's loaded — hash check will reject if stale\n")
                sys.stderr.flush()
        current_kva_ref = kva_path
        if _is_dsv41(session.model) and kva_path is None:
            _send_handshake(ls, 'error', model_fp, 0)
            _send_done(ls, 0, 0.0, STOP_NONE, 'DSV41 requires messages, prompt, or prompt_token_ids')
            return True
        if isinstance(kva_path, (bytes, bytearray, memoryview)):
            kva = _parse_kva_prompt_payload_bytes(kva_path)
        else:
            kva = _parse_kva_prompt_payload(kva_path)
        session._kimi_vision_prompt = None
        if kva.get('raw_image') and (not getattr(session, '_use_vlm', False)) and (not _is_dsv41(session.model)):
            try:
                prompt_token_ids = session.kimi_vision_prompt(kva['prompt_text'], kva['raw_image'])
            except Exception as e:
                sys.stderr.write(f'[mlx-server] image request refused: {e}\n')
                sys.stderr.flush()
                _send_handshake(ls, 'error', model_fp, 0, extra={'error': str(e)})
                _send_done(ls, 0, 0.0, STOP_NONE, '')
                return True
        else:
            prompt_token_ids = session.encode_prompt(kva['prompt_text'], kva['token_ids'])
        n_prompt = len(prompt_token_ids)
        kva_prompt_hash = _hash_token_ids(prompt_token_ids) if prompt_token_ids else 0
        _kva_img_cells = int(kva.get('img_cell_count', 0))
        _effective_count = n_prompt + _kva_img_cells
        if config_prompt_token_count and config_prompt_token_count != n_prompt and (config_prompt_token_count != _effective_count):
            sys.stderr.write(f'[mlx-server] KVA prompt token-count mismatch: config={config_prompt_token_count} kva={n_prompt} (img_cells={_kva_img_cells}); rejecting stale KVA\n')
            sys.stderr.flush()
            _send_handshake(ls, 'error', model_fp, 0)
            _send_done(ls, 0, 0.0, STOP_NONE, 'stale_kva_prompt_count')
            return True
        if config_prompt_hash and _kva_img_cells == 0 and (config_prompt_hash != kva_prompt_hash):
            sys.stderr.write(f'[mlx-server] KVA prompt hash mismatch: config=0x{config_prompt_hash:016x} kva=0x{kva_prompt_hash:016x}; rejecting stale KVA\n')
            sys.stderr.flush()
            _send_handshake(ls, 'error', model_fp, 0)
            _send_done(ls, 0, 0.0, STOP_NONE, 'stale_kva_prompt_hash')
            return True
        kv_bytes = kva.get('kv_state_bytes', b'')
        if _is_dsv41(session.model):
            kv_bytes = b''
        kv_is_mla = bool(kva.get('is_mla', 0))
        requested_n_predict = int(kva.get('requested_n_predict', -1))
        session_sampling_overrides = _overlay_sampling_profile_overrides(kva.get('sampling_profile'), config_sampling_profile)
        session_sampling_profile = _merge_sampling_profiles(session._sampling_profile, session_sampling_overrides)
        session_max_tokens = max_tokens
        if config_max_tokens >= 0:
            session_max_tokens = config_max_tokens
        if requested_n_predict >= 0:
            session_max_tokens = requested_n_predict
        if n_prompt == 0 and (not kv_bytes):
            sys.stderr.write(f"[mlx-server] kva has no prompt payload (format_minor={kva['format_minor']})\n")
            _send_handshake(ls, 'error', model_fp, 0)
            _send_done(ls, 0, 0.0, STOP_NONE, 'no_prompt_payload')
            return True
        report_n = n_prompt if n_prompt > 0 else kva.get('token_count', 0)
        route_hint = 'KV-inject' if kv_bytes else 'local-prefill'
        sys.stderr.write(f"[mlx-server] kva format_minor={kva['format_minor']} prompt_token_count={kva['prompt_token_count']} kv_bytes={len(kv_bytes)} is_mla={kv_is_mla} prompt_ids={n_prompt} route={route_hint} prompt_hash=0x{kva_prompt_hash:016x} requested_n_predict={requested_n_predict} max_tokens={session_max_tokens} temp={session_sampling_profile['temp']:.3f} top_p={session_sampling_profile['top_p']:.3f} top_k={session_sampling_profile['top_k']}\n")
        sys.stderr.flush()
        if config_reuse_retained_count > 0 and (not _retained_prefix_ready_or_nack(session, ls, model_fp, report_n, config_reuse_retained_count, config_reuse_retained_prefix_hash, has_kv_payload=bool(kv_bytes))):
            return True
        _hs_extra = {'thinking_mode': getattr(session, '_dsv41_thinking_mode', None)} if _is_dsv41(session.model) else None
        if not _send_handshake(ls, 'ready', model_fp, report_n, extra=_hs_extra):
            return True
        try:
            _raw_image = kva.get('raw_image')
            if _raw_image and session._kimi_vision_prompt is None:
                n_gen, tps, stop_reason, stopping_word = session.vision_generate(_raw_image, kva.get('prompt_text', ''), session_max_tokens, stop_words, ls)
            else:
                kva['kv_state_bytes'] = b''
                if kva_path_shared is not None and kva_path_shared[0] is current_kva_ref:
                    kva_path_shared[0] = None
                kva_path = None
                latest = None
                current_kva_ref = None
                n_gen, tps, stop_reason, stopping_word = session.prefill_and_decode(prompt_token_ids, max_tokens=session_max_tokens, stop_words=stop_words, ls=ls, chunk_size=chunk_size, kv_state_bytes=kv_bytes, kv_is_mla=kv_is_mla, last_token_id=kva.get('last_token_id', -1), image_embd=kva.get('image_embd'), img_pos_start=int(kva.get('img_pos_start', 0)), sampling_profile=session_sampling_overrides, mtp_hidden=kva.get('mtp_hidden'), swi_config=config_swi_config, reuse_retained_count=config_reuse_retained_count, kv_delta_base=int(kva.get('base_prefix_cells', 0)))
            sys.stderr.write(f'[mlx-server] decode done: {n_gen} tok in {n_gen / max(tps, 1e-09):.2f}s = {tps:.2f} tok/s, stop_reason={stop_reason}\n')
            kv_bytes = b''
            sys.stderr.flush()
            _rtok = getattr(session, '_retained_tokens', None) or []
            _rt_count = len(_rtok)
            _rt_hash = _hash_token_ids(_rtok) if _rt_count > 0 else 0
            _send_done(ls, n_gen, tps, stop_reason, stopping_word, extra=getattr(session, '_swi_last_stats', None), retained_token_count=_rt_count, retained_prefix_hash=_rt_hash, completion=getattr(session, '_dsv41_completion', None))
        except Exception as e:
            import traceback
            if _is_dsv41(session.model):
                session._retained_cache = None
                session._retained_tokens = []
            sys.stderr.write(f'[mlx-server] decode error: {e}\n')
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            _send_done(ls, 0, 0.0, STOP_NONE, '')
    finally:
        ls.close()
    return True

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='127.0.0.1', help='Local interface for both token and KV listeners')
    p.add_argument('--port', type=int, default=9301, help='bind port for token stream (default 9301)')
    p.add_argument('--model', required=True, help='path to MLX model directory')
    p.add_argument('--adapter', action='append', default=[], help='Studio MLX LoRA adapter directory; repeat to stack multiple adapters.')
    p.add_argument('--token-sidecar', action='append', default=[], help='studio.token_gated_sidecar.v1 safetensors artifact; repeat for multiple experts.')
    p.add_argument('--token-sidecar-scale', type=float, default=1.0, help='global ceiling multiplied by each learned per-token sidecar gate (default 1.0).')
    p.add_argument('--kv-artifact', default=None, help='path to KVARTIF1 file (KV bytes are injected when present; prompt payload supplies tail/fallback). Required unless --kv-listen-port is set.')
    p.add_argument('--kv-listen-port', type=int, default=None, help="if set, listen on this port for TBP1 KVA frames from the RTX Windows server (port 9100 in production). Each received KVA becomes the next session's prompt.")
    p.add_argument('--max-tokens', type=int, default=256, help='decode cap per session')
    p.add_argument('--chunk-size', type=int, default=2048, help='prefill chunk size to limit peak memory on long prompts')
    p.add_argument('--one-shot', action='store_true', help='serve a single session and exit (default: loop forever)')
    p.add_argument('--accept-timeout-s', type=float, default=600.0, help='how long to wait for a client to connect (default 600s)')
    p.add_argument('--temperature', '--temp', dest='temperature', type=float, default=None, help='decode sampling temperature override; KVA/GUI sampling metadata wins when set per request.')
    p.add_argument('--top-p', type=float, default=None, help='decode nucleus sampling override.')
    p.add_argument('--top-k', type=int, default=None, help='decode top-k sampling override.')
    p.add_argument('--min-p', type=float, default=None, help='decode min-p sampling override.')
    p.add_argument('--sampling-seed', type=int, default=None, help='decode sampler seed override; <=0 leaves MLX RNG unseeded.')
    p.add_argument('--mtp-mode', choices=('off', 'validate', 'load', 'draft', 'speculative'), default='off', help='GLM-5.2 MTP sidecar mode: off, validate header only, load arrays, draft with target-verified diagnostic checking, or speculative with target-verified draft commits.')
    p.add_argument('--mtp-sidecar', default='auto', help='GLM-5.2 MTP sidecar directory, or auto for sibling <model>-mtp-layer78.')
    p.add_argument('--mtp-max-draft', type=int, default=7, help='maximum iterative GLM-5.2 MTP draft tokens to propose per target step.')
    p.add_argument('--mtp-quantized-sidecar-policy', choices=('allow', 'warn', 'reject'), default='warn', help='policy for quantized GLM-5.2 MTP sidecars: allow silently, warn (default), or reject so the non-quantized FP16-runtime MTP sidecar is required.')
    p.add_argument('--mtp-debug-windows', type=int, default=0, help='log draft/target token IDs for the first N MTP verification windows.')
    p.add_argument('--mtp-compute-dtype', choices=('native', 'float16', 'fp16', 'bfloat16', 'bf16'), default='native', help='MTP sidecar compute dtype. Defaults to native (= the bf16 sidecar weights, no cast): float16 casting collapses speculative acceptance with LENGTH (67%@64tok -> 23%@128tok) because the fp16-cast draft logits diverge from the bf16-precision target; native is length-stable (46%@128tok). float16/bfloat16 remain selectable for diagnosis.')
    p.add_argument('--mtp-recurrent-state', choices=('shared_head_norm', 'shared-head-norm', 'raw'), default='shared_head_norm', help='Hidden state carried between MTP draft steps. raw matches the GLM/vLLM NextN recurrence; shared_head_norm is a diagnostic mode.')
    p.add_argument('--mtp-schedule', choices=('reference', 'target-known-next', 'target_known_next'), default='reference', help='Speculative scheduling contract. reference drafts from the current target token and verifies [current,drafts...] like MTPLX; target_known_next keeps the earlier diagnostic path that target-steps once before drafting.')
    p.add_argument('--mtp-prompt-history', choices=('committed', 'shifted-bonus', 'shifted_bonus'), default='committed', help='Prompt MTP warmup. committed keeps the current best replay path; shifted_bonus appends prompt_ids[1:] plus the first target bonus like native mlx-vlm/MTPLX.')
    p.add_argument('--mtp-cache-commit', choices=('replay', 'native'), default='replay', help='How reference scheduling commits MTP cache history after target verification. replay rolls back speculative rows and replays the verified committed prefix; native trims only rejected draft rows and appends the target bonus MTPLX-style.')
    p.add_argument('--mtp-verify-profile', choices=('safe', 'mtplx'), default='safe', help='MTP verify optimisation bundle. safe batches target-id extraction but keeps hidden/history materialisation eager. mtplx also defers verify hidden eval and MTP history materialisation.')
    p.add_argument('--mtp-target-token-eval', choices=('auto', 'batched', 'rowwise'), default='auto', help='How to extract greedy target ids from verify logits. batched uses one MLX eval boundary; rowwise preserves the older per-row sync path.')
    p.add_argument('--mtp-verify-forward', choices=('auto', 'block', 'rowwise'), default='auto', help='How to run the target verifier forward pass. block verifies the whole speculative window in one target call; rowwise verifies one token at a time for qmv/cache diagnostics.')
    p.add_argument('--mtp-verify-hidden-eval', choices=('auto', 'eager', 'defer'), default='auto', help='Whether to materialise verify hidden states immediately. auto is eager for safe and defer for mtplx.')
    p.add_argument('--mtp-history-materialization', choices=('auto', 'eager', 'lazy'), default='auto', help='Whether committed MTP history appends force mx.eval immediately. auto is eager for safe and lazy for mtplx.')
    p.add_argument('--mtp-share-topk-source', choices=('target', 'internal', 'off'), default='internal', help='GLM DSA top-k sharing source for MTP. internal matches SGLang NextN by reusing only top-k computed by the MTP layer, target seeds from target-layer verifier top-k, and off disables MTP shared-index reuse.')
    p.add_argument('--mtp-match-trunk-precision', choices=('auto', 'on', 'off'), default='auto', help="MTPLX fix #2 (EXPERIMENTAL): requantize a bf16 MTP sidecar to the quantized trunk's mxfp4 spec. EMPIRICAL (Mac A/B 2026-06-22): a NAIVE requant COLLAPSES GLM-5.2 acceptance (67%->16% @128) without MTPLX-style head calibration, so auto and off both KEEP the bf16 sidecar (the better config); only on requants (opt-in for a future calibrated port).")
    p.add_argument('--mtp-adaptive-depth', choices=('off', 'streak', 'streak-floor'), default='off', help='Dynamically adapt the MTP draft depth from rolling acceptance: go deeper while drafts are fully accepted (predictable spans -> ~free speed), shallower when rejected early (deep drafts lose to target-only). streak adapts in [1, max-draft]; streak-floor can drop to target-only for hopeless spans. off uses the fixed --mtp-max-draft.')
    p.add_argument('--mtp-token-budget', type=int, default=0, help='Maximum generated tokens that may use MTP before falling back to target-only decode. 0 means unlimited.')
    args = p.parse_args()
    config_path = Path(args.model) / 'config.json'
    local_dsv41 = config_path.is_file() and _is_dsv41(json.loads(config_path.read_text(encoding='utf-8')))
    if args.kv_artifact is None and args.kv_listen_port is None and (not local_dsv41):
        sys.stderr.write('[mlx-server] either --kv-artifact or --kv-listen-port is required\n')
        return 2
    if args.kv_artifact is not None and (not Path(args.kv_artifact).is_file()):
        sys.stderr.write(f'[mlx-server] --kv-artifact not found: {args.kv_artifact}\n')
        return 2
    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listen.bind((args.host, args.port))
    except OSError as e:
        sys.stderr.write(f'[mlx-server] bind {args.host}:{args.port} failed: {e}\n')
        return 3
    listen.listen(1)
    sys.stderr.write(f'[mlx-server] listening on {args.host}:{args.port}\n')
    sys.stderr.flush()
    cli_sampling_profile: Dict[str, Any] = {}
    if args.temperature is not None:
        cli_sampling_profile['temp'] = args.temperature
    if args.top_p is not None:
        cli_sampling_profile['top_p'] = args.top_p
    if args.top_k is not None:
        cli_sampling_profile['top_k'] = args.top_k
    if args.min_p is not None:
        cli_sampling_profile['min_p'] = args.min_p
    if args.sampling_seed is not None:
        cli_sampling_profile['seed'] = args.sampling_seed
    session = MLXSession(args.model, mtp_mode=args.mtp_mode, mtp_sidecar=args.mtp_sidecar, mtp_max_draft=args.mtp_max_draft, mtp_quantized_sidecar_policy=args.mtp_quantized_sidecar_policy, mtp_debug_windows=args.mtp_debug_windows, mtp_compute_dtype=args.mtp_compute_dtype, mtp_recurrent_state=args.mtp_recurrent_state, mtp_schedule=args.mtp_schedule, mtp_prompt_history=args.mtp_prompt_history, mtp_cache_commit=args.mtp_cache_commit, mtp_verify_profile=args.mtp_verify_profile, mtp_target_token_eval=args.mtp_target_token_eval, mtp_verify_forward=args.mtp_verify_forward, mtp_verify_hidden_eval=args.mtp_verify_hidden_eval, mtp_history_materialization=args.mtp_history_materialization, mtp_share_topk_source=args.mtp_share_topk_source, mtp_match_trunk_precision=args.mtp_match_trunk_precision, mtp_adaptive_depth=args.mtp_adaptive_depth, mtp_token_budget=args.mtp_token_budget, adapters=args.adapter, token_sidecars=args.token_sidecar, token_sidecar_scale=args.token_sidecar_scale, sampling_profile=cli_sampling_profile)
    sys.stderr.write(f'[mlx-server] loading model {args.model}\n')
    sys.stderr.flush()
    try:
        load_dt, model_fp = session.load()
    except Exception as e:
        import traceback
        sys.stderr.write(f'[mlx-server] model load failed: {e}\n')
        traceback.print_exc(file=sys.stderr)
        return 4
    sys.stderr.write(f'[mlx-server] model loaded in {load_dt:.1f}s, fingerprint=0x{model_fp:016x}\n')
    sys.stderr.flush()
    try:
        import mlx.core as mx
        sys.stderr.write('[mlx-server] warming up Metal JIT (1-token prefill)...\n')
        sys.stderr.flush()
        _warmup_cache = session._make_cache()
        _warmup_lang = getattr(session.model, 'language_model', session.model)
        _warmup_bos = session.tokenizer.backend_tokenizer.token_to_id(_dsv41_encoding().bos_token) if _is_dsv41(session.model) else 1
        _warmup_ids = mx.array([[_warmup_bos]])
        _warmup_out = _warmup_lang(_warmup_ids, cache=_warmup_cache)
        _warmup_logits = _warmup_out.logits if hasattr(_warmup_out, 'logits') else _warmup_out
        mx.eval(_warmup_logits)
        del _warmup_cache, _warmup_lang, _warmup_ids, _warmup_out, _warmup_logits
        mx.clear_cache()
        _lim_gb = float(os.environ.get('LLAMA_MLX_CACHE_LIMIT_GB', '24') or 24)
        try:
            mx.set_cache_limit(int(_lim_gb * (1 << 30)))
            sys.stderr.write(f"[mlx-server] Metal buffer pool bounded at {_lim_gb:g} GB (per-request clear {('ON' if _clear_cache_per_request() else 'off')})\n")
            sys.stderr.flush()
        except Exception as _lim_e:
            sys.stderr.write(f'[mlx-server] set_cache_limit failed (non-fatal): {_lim_e}\n')
            sys.stderr.flush()
        sys.stderr.write('[mlx-server] Metal JIT warmup complete\n')
        sys.stderr.flush()
        _start_keep_warm_thread()
    except Exception as _warmup_e:
        if _is_dsv41(session.model):
            raise
        sys.stderr.write(f'[mlx-server] warmup failed (non-fatal): {_warmup_e}\n')
        sys.stderr.flush()
    import threading
    kva_path_shared = [args.kv_artifact]
    event_new_kva = threading.Event()
    shutdown_event = threading.Event()
    if args.kv_listen_port is not None:
        t = threading.Thread(target=_tbp1_listener_thread, args=(args.kv_listen_port, kva_path_shared, event_new_kva, shutdown_event, session.prefix_query, args.host), daemon=True)
        t.start()
        sys.stderr.write(f'[mlx-server] TBP1 KVA listener on port {args.kv_listen_port}\n')
        sys.stderr.flush()
        if args.kv_artifact is None and (not _is_dsv41(session.model)):
            sys.stderr.write('[mlx-server] waiting for first KVA from Windows...\n')
            sys.stderr.flush()
            event_new_kva.wait()
            event_new_kva.clear()
            sys.stderr.write(f'[mlx-server] first KVA received: {_describe_kva_ref(kva_path_shared[0])}\n')
            sys.stderr.flush()
    try:
        while True:
            current_kva = kva_path_shared[0]
            if current_kva is None and (not _is_dsv41(session.model)):
                sys.stderr.write('[mlx-server] no KVA available yet, waiting...\n')
                sys.stderr.flush()
                event_new_kva.wait(timeout=5.0)
                event_new_kva.clear()
                continue
            keep_going = _serve_one_session(listen, session, model_fp, kva_path=current_kva, max_tokens=args.max_tokens, chunk_size=args.chunk_size, accept_timeout_s=args.accept_timeout_s, kva_path_shared=kva_path_shared if args.kv_listen_port else None)
            if event_new_kva.is_set():
                event_new_kva.clear()
                sys.stderr.write(f'[mlx-server] new KVA available: {_describe_kva_ref(kva_path_shared[0])}\n')
                sys.stderr.flush()
            if args.one_shot:
                break
            if not keep_going:
                break
    except KeyboardInterrupt:
        sys.stderr.write('[mlx-server] interrupted\n')
    finally:
        shutdown_event.set()
        if _is_dsv41(session.model):
            _dsv41_save_hot(session)
            import atexit
            atexit.unregister(_dsv41_save_hot)
        try:
            listen.close()
        except OSError:
            pass
    return 0
if __name__ == '__main__':
    sys.exit(main())
