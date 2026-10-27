from __future__ import annotations
import os, posixpath, json as _json
from pathlib import Path
from typing import Optional
from . import preset_samplers
OMLX_STUDIO_BASE_PATH = 'runtime/omlx'
DEFAULT_MAC_MFERENCE_SERVER = 'MferenceServer'

def mlx_vlm_argv(model_path: str, *, host: str, port: int, max_kv_size: Optional[int]=None, kv_bits: Optional[int]=None, python: str='python', draft_model: Optional[str]=None, draft_kind: Optional[str]=None, draft_block_size: Optional[int]=None) -> list:
    argv = [python, '-m', 'mlx_vlm.server', '--model', str(model_path), '--host', host, '--port', str(port)]
    if max_kv_size:
        argv += ['--max-kv-size', str(int(max_kv_size))]
    if kv_bits:
        argv += ['--kv-bits', str(int(kv_bits))]
    if draft_model:
        argv += ['--draft-model', str(draft_model)]
        if draft_kind:
            argv += ['--draft-kind', str(draft_kind)]
        if draft_block_size is not None:
            argv += ['--draft-block-size', str(int(draft_block_size))]
    return argv

def mference_server_argv(model_path: str, *, port: int, binary: str=DEFAULT_MAC_MFERENCE_SERVER, bind: str='loopback', bind_address: Optional[str]=None, model_id: Optional[str]=None, max_context: Optional[int]=None, queue_limit: Optional[int]=None, prompt_cache_mode: Optional[str]=None, expert_cache_slots: Optional[int]=None, expert_cache_gb: Optional[float]=None, expert_cache_headroom_gb: Optional[float]=None) -> list[str]:
    bind_mode = str(bind or 'loopback').strip().lower()
    if bind_mode not in {'loopback', 'tailnet'}:
        raise ValueError("Mference bind must be 'loopback' or 'tailnet'")
    cache_mode = str(prompt_cache_mode or 'single-prefix').strip().lower()
    if cache_mode not in {'off', 'single-prefix'}:
        raise ValueError("Mference prompt_cache_mode must be 'off' or 'single-prefix'")
    if expert_cache_slots is not None and expert_cache_gb is not None:
        raise ValueError('Mference expert_cache_slots and expert_cache_gb are mutually exclusive')
    argv = [str(binary or DEFAULT_MAC_MFERENCE_SERVER), '--model', str(model_path), '--port', str(int(port))]
    exact_address = str(bind_address or '').strip()
    if exact_address:
        argv += ['--bind-address', exact_address]
    else:
        argv += ['--bind', bind_mode]
    if model_id:
        argv += ['--model-id', str(model_id)]
    if max_context is not None:
        argv += ['--max-context', str(int(max_context))]
    if queue_limit is not None:
        argv += ['--queue-limit', str(int(queue_limit))]
    argv += ['--prompt-cache-mode', cache_mode]
    if expert_cache_slots is not None:
        argv += ['--expert-cache-slots', str(int(expert_cache_slots))]
    if expert_cache_gb is not None:
        argv += ['--expert-cache-gb', str(float(expert_cache_gb))]
    if expert_cache_headroom_gb is not None:
        argv += ['--expert-cache-headroom-gb', str(float(expert_cache_headroom_gb))]
    return argv

def omlx_argv(model_path: str, *, host: str, port: int, python: str='python', omlx_binary: str | None=None, log_level: str | None=None, memory_guard: str | None=None, api_key: str | None=None, max_concurrent_requests=None, base_path: str | None=None, ctx=None, deepseek_v41_engram_ssd_offload: bool | None=None) -> list:
    binary = str(omlx_binary or '').strip()
    if not binary:
        base = str(python or '')
        binary = posixpath.join(posixpath.dirname(base), 'omlx') if base and base != 'python' else 'omlx'
    directory = str(model_path).rstrip('/')
    parent = posixpath.dirname(directory) or directory
    argv = [binary, 'serve', '--model-dir', parent, '--host', host, '--port', str(port)]
    resolved_base = str(base_path or OMLX_STUDIO_BASE_PATH)
    settings_path = omlx_write_settings(resolved_base, ctx, model_id=omlx_model_id(model_path), deepseek_v41_engram_ssd_offload=deepseek_v41_engram_ssd_offload)
    if settings_path is None and (base_path is not None or deepseek_v41_engram_ssd_offload is not None):
        raise RuntimeError('Could not write oMLX settings for the requested base-path/Engram mode')
    argv += ['--base-path', resolved_base]
    if log_level:
        argv += ['--log-level', str(log_level)]
    if memory_guard:
        argv += ['--memory-guard', str(memory_guard)]
    if max_concurrent_requests not in (None, ''):
        argv += ['--max-concurrent-requests', str(int(max_concurrent_requests))]
    if api_key:
        argv += ['--api-key', str(api_key)]
    return argv

def omlx_write_settings(base_path: str, ctx, *, model_id: str | None=None, deepseek_v41_engram_ssd_offload: bool | None=None) -> Optional[str]:
    try:
        directory = Path(str(base_path))
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'settings.json'
        current = {}
        if target.is_file():
            current = _json.loads(target.read_text(encoding='utf-8')) or {}
            if not isinstance(current, dict):
                raise ValueError('Existing settings must be a JSON object; preserved without overwriting')
        current.setdefault('auth', {})['skip_api_key_verification'] = True
        window = int(ctx) if ctx else 0
        if window > 0:
            current.setdefault('sampling', {})['max_context_window_policy'] = window
        target.write_text(_json.dumps(current, indent=2), encoding='utf-8')
        if deepseek_v41_engram_ssd_offload is not None:
            if not model_id or not isinstance(deepseek_v41_engram_ssd_offload, bool):
                raise ValueError('Engram mode requires a model id and a boolean')
            model_target = directory / 'model_settings.json'
            model_settings = _json.loads(model_target.read_text(encoding='utf-8')) if model_target.exists() else {'version': 1, 'models': {}}
            model_settings.setdefault('models', {}).setdefault(model_id, {})['deepseek_v41_engram_ssd_offload'] = deepseek_v41_engram_ssd_offload
            model_target.write_text(_json.dumps(model_settings, indent=2), encoding='utf-8')
        return str(target)
    except Exception as exc:
        action = 'launch blocked' if deepseek_v41_engram_ssd_offload is not None else 'launching uncapped'
        print(f'[omlx] could not write settings at {base_path}: {exc!r}; {action}', flush=True)
        return None

def omlx_model_id(model_path: str) -> str:
    directory = str(model_path).rstrip('/')
    return posixpath.basename(directory) or posixpath.basename(posixpath.dirname(directory))

def mlx_lm_argv(model_path: str, *, host: str, port: int, python: str='python', trust_remote_code: bool=False, temperature=None, min_p=None, top_p=None, top_k=None, max_tokens=None, context=None, decode_concurrency=None, prompt_concurrency=None) -> list:
    argv = [python, '-m', 'mlx_lm.server', '--model', str(model_path), '--host', host, '--port', str(port)]
    if trust_remote_code:
        argv.append('--trust-remote-code')

    def _num(v):
        if v is None or v == '':
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    argv += preset_samplers.launch_args(dict(temperature=temperature, min_p=min_p, top_p=top_p, top_k=top_k), 'mlx-lm')
    _mt = _num(max_tokens)
    if _mt is None or _mt <= 0:
        _mt = _num(context)
    if _mt is not None and _mt > 0:
        argv += ['--max-tokens', str(int(_mt))]
    _dc = _num(decode_concurrency)
    if _dc is not None and _dc > 0:
        argv += ['--decode-concurrency', str(int(_dc))]
    _pc = _num(prompt_concurrency)
    if _pc is not None and _pc > 0:
        argv += ['--prompt-concurrency', str(int(_pc))]
    return argv
