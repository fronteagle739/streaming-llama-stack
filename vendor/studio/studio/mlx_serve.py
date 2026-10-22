# Derived from Studio backend/studio/mlx_serve.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Optional
_MLX_SERVE_BIN_CANDIDATES = ['mlx-serve', '/opt/homebrew/bin/mlx-serve', '/opt/homebrew/opt/mlx-serve/mlx-serve', os.path.expanduser('~/.local/mlx-serve/mlx-serve'), '/Applications/MLX Core.app/Contents/Resources/mlx-serve', '/Applications/MLX Core.app/Contents/MacOS/mlx-serve']

def find_mlx_serve() -> Optional[Path]:
    env = os.environ.get('STUDIO_MLX_SERVE_BIN')
    if env:
        p = Path(os.path.expanduser(env))
        if p.is_file():
            return p
    for c in _MLX_SERVE_BIN_CANDIDATES:
        p = Path(os.path.expanduser(c))
        if p.is_file():
            return p
    which = shutil.which('mlx-serve')
    if which:
        return Path(which)
    return None
MLX_SERVE_GENERIC_KEYS: tuple[str, ...] = ('model_path', 'model', 'model_alias', 'ctx', 'max_tokens', 'temperature', 'top_p', 'top_k')
MLX_SERVE_FLAG_KEYS: tuple[str, ...] = ('mlx_serve_pld', 'mlx_serve_pld_draft_len', 'mlx_serve_pld_key_len', 'mlx_serve_mtp', 'mlx_serve_no_mtp', 'mlx_serve_mtp_depth', 'mlx_serve_mtp_history_window', 'mlx_serve_mtp_head_kv_quant', 'mlx_serve_dspark', 'mlx_serve_no_ds4_mtp', 'mlx_serve_drafter', 'mlx_serve_draft_block_size', 'mlx_serve_reasoning_budget', 'mlx_serve_no_vision', 'mlx_serve_skip_mem_preflight', 'mlx_serve_timeout', 'mlx_serve_config_overrides', 'mlx_serve_prefix_cache_entries', 'mlx_serve_prefix_cache_mem', 'mlx_serve_prefix_cache_disk', 'mlx_serve_ssm_checkpoint_max', 'mlx_serve_tokenize_cache_entries', 'mlx_serve_kv_quant', 'mlx_serve_kv_attn_mode', 'mlx_serve_wired_margin_gib', 'mlx_serve_max_concurrent', 'mlx_serve_bind_address', 'mlx_serve_extra_flags', 'mlx_serve_env')
MLX_SERVE_PRESET_KEYS: tuple[str, ...] = MLX_SERVE_GENERIC_KEYS + MLX_SERVE_FLAG_KEYS
DEFAULT_MLX_SERVE_BIND = '127.0.0.1'

def _flag_on(cfg: dict, key: str, default: str='0') -> bool:
    return str(cfg.get(key) or default).strip().lower() not in ('0', 'false', 'no', 'off', '')

def _flag_off(cfg: dict, key: str) -> bool:
    if key not in (cfg or {}):
        return False
    value = cfg.get(key)
    if value is None or str(value).strip() == '':
        return False
    return str(value).strip().lower() in ('0', 'false', 'no', 'off')

def mlx_serve_config_block(preset: dict) -> tuple[dict, str]:
    p = preset or {}
    mode = str(p.get('mode') or (p.get('rtx') or {}).get('mode') or '').strip().lower()
    block = 'mac' if mode == 'remote' else 'rtx'
    cfg = p.get(block)
    return (cfg if isinstance(cfg, dict) else {}, block)

def _resolve_config(preset: dict, config: Optional[dict], config_name: str) -> tuple[dict, str]:
    if config is None:
        return mlx_serve_config_block(preset)
    return (config or {}, config_name or 'config')

def _require_model_path(cfg: dict, cfg_name: str, preset: dict) -> str:
    model = str((cfg or {}).get('model_path') or '').strip()
    if model:
        return model
    mode = str((preset or {}).get('mode') or 'local').strip().lower() or 'local'
    raise ValueError(f'mlx-serve: the {cfg_name!r} block has no model_path (preset mode={mode!r}; keys present: {sorted(cfg or {})}). A {mode!r} preset keeps its mlx-serve config in {cfg_name!r}, so set model_path there. NOT falling back to the other block -- that would launch a different checkpoint than the preset names.')

def mlx_serve_argv(preset: dict, *, port: int=8080, config: Optional[dict]=None, config_name: str='', bind_address: Optional[str]=None) -> Optional[list[str]]:
    cfg, cfg_name = _resolve_config(preset, config, config_name)
    model = _require_model_path(cfg, cfg_name, preset)
    binary = find_mlx_serve()
    if binary is None:
        return None
    bind = str(cfg.get('mlx_serve_bind_address') or '').strip() or str(bind_address or '').strip() or DEFAULT_MLX_SERVE_BIND
    argv = [str(binary), '--serve', '--model', model, '--host', bind, '--port', str(port)]
    if cfg.get('ctx'):
        argv += ['--ctx-size', str(int(cfg['ctx']))]
    if cfg.get('max_tokens'):
        argv += ['--max-tokens', str(int(cfg['max_tokens']))]
    for key, flag in (('temperature', '--temp'), ('top_p', '--top-p'), ('top_k', '--top-k')):
        if cfg.get(key) is not None:
            argv += [flag, str(cfg[key])]
    if _flag_off(cfg, 'mlx_serve_pld'):
        argv.append('--no-pld')
    if cfg.get('mlx_serve_pld_draft_len'):
        argv += ['--pld-draft-len', str(int(cfg['mlx_serve_pld_draft_len']))]
    if cfg.get('mlx_serve_pld_key_len'):
        argv += ['--pld-key-len', str(int(cfg['mlx_serve_pld_key_len']))]
    if _flag_on(cfg, 'mlx_serve_no_mtp'):
        argv.append('--no-mtp')
    elif _flag_on(cfg, 'mlx_serve_mtp'):
        argv.append('--mtp')
    if cfg.get('mlx_serve_mtp_depth'):
        argv += ['--mtp-depth', str(int(cfg['mlx_serve_mtp_depth']))]
    if cfg.get('mlx_serve_mtp_history_window'):
        argv += ['--mtp-history-window', str(int(cfg['mlx_serve_mtp_history_window']))]
    if _flag_on(cfg, 'mlx_serve_mtp_head_kv_quant'):
        argv.append('--mtp-head-kv-quant')
    if _flag_on(cfg, 'mlx_serve_dspark'):
        argv.append('--dspark')
    if _flag_on(cfg, 'mlx_serve_no_ds4_mtp'):
        argv.append('--no-ds4-mtp')
    drafter = str(cfg.get('mlx_serve_drafter') or '').strip()
    if drafter:
        argv += ['--drafter', drafter]
    if cfg.get('mlx_serve_draft_block_size'):
        argv += ['--draft-block-size', str(int(cfg['mlx_serve_draft_block_size']))]
    if cfg.get('mlx_serve_reasoning_budget'):
        argv += ['--reasoning-budget', str(int(cfg['mlx_serve_reasoning_budget']))]
    if _flag_on(cfg, 'mlx_serve_no_vision'):
        argv.append('--no-vision')
    if _flag_on(cfg, 'mlx_serve_skip_mem_preflight'):
        argv.append('--skip-mem-preflight')
    if cfg.get('mlx_serve_timeout') is not None:
        argv += ['--timeout', str(int(cfg['mlx_serve_timeout']))]
    if cfg.get('mlx_serve_prefix_cache_entries') is not None:
        argv += ['--prefix-cache-entries', str(int(cfg['mlx_serve_prefix_cache_entries']))]
    pc_mem = str(cfg.get('mlx_serve_prefix_cache_mem') or '').strip()
    if pc_mem:
        argv += ['--prefix-cache-mem', pc_mem]
    pc_disk = str(cfg.get('mlx_serve_prefix_cache_disk') or '').strip()
    if pc_disk:
        argv += ['--prefix-cache-disk', pc_disk]
    if cfg.get('mlx_serve_ssm_checkpoint_max') is not None:
        argv += ['--ssm-checkpoint-max', str(int(cfg['mlx_serve_ssm_checkpoint_max']))]
    if cfg.get('mlx_serve_tokenize_cache_entries') is not None:
        argv += ['--tokenize-cache-entries', str(int(cfg['mlx_serve_tokenize_cache_entries']))]
    kvq = str(cfg.get('mlx_serve_kv_quant') or '').strip().lower()
    if kvq and kvq != 'off':
        argv += ['--kv-quant', kvq]
    kvam = str(cfg.get('mlx_serve_kv_attn_mode') or '').strip().lower()
    if kvam and kvam != 'auto':
        argv += ['--kv-attn-mode', kvam]
    if cfg.get('mlx_serve_wired_margin_gib') is not None:
        argv += ['--wired-margin-gib', str(int(cfg['mlx_serve_wired_margin_gib']))]
    if cfg.get('mlx_serve_max_concurrent') is not None:
        argv += ['--max-concurrent', str(int(cfg['mlx_serve_max_concurrent']))]
    overrides = str(cfg.get('mlx_serve_config_overrides') or '').strip()
    if overrides:
        argv += ['--config-overrides', overrides]
    extra = str(cfg.get('mlx_serve_extra_flags') or '').strip()
    if extra:
        argv += extra.split()
    return argv

def mlx_serve_env_for(preset: dict, *, config: Optional[dict]=None, config_name: str='') -> dict:
    cfg, _ = _resolve_config(preset, config, config_name)
    raw_env = cfg.get('mlx_serve_env')
    if not raw_env:
        raw_env = (preset or {}).get('mlx_serve_env')
    out: dict = {}
    if isinstance(raw_env, dict) and raw_env:
        for k, v in raw_env.items():
            out[str(k)] = None if v is None else str(v)
    elif isinstance(raw_env, str) and raw_env.strip():
        for tok in raw_env.split():
            if '=' in tok:
                k, _, v = tok.partition('=')
                out[k.strip()] = v.strip()
    return out
