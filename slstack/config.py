from __future__ import annotations
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path=None, overrides=None):
    cfg = json.loads((ROOT / 'config.example.json').read_text(encoding='utf-8'))
    if path:
        cfg.update(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ('host_budget_gib', 'min_available_gib', 'job_commit_gib', 'request_timeout_s', 'lease_minutes'):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    wait = cfg.get('gpu_wait_seconds', 0)
    if type(wait) not in (int, float) or not math.isfinite(wait) or not 0 <= wait <= 7200:
        raise ValueError('gpu_wait_seconds must be finite and in [0,7200]')
    if cfg['lease_minutes'] > 20:
        raise ValueError('GPU lease must be at most 20 minutes')
    if cfg['host'] not in ('127.0.0.1', '::1'):
        raise ValueError('This local stack binds to loopback only')
    command_text = ' '.join(cfg['ds4_command']).lower()
    low_kv = any(key in ('--cache-type-k', '--cache-type-v', '-ctk', '-ctv', '--kv-type') and value.lower().startswith(('q4', 'q5', 'q6', 'iq'))
                 for key, value in zip(cfg['ds4_command'], cfg['ds4_command'][1:]))
    if low_kv or any(flag in command_text for flag in ('--kv-disk', '--kv-swap', '--ram-streaming')):
        raise ValueError('Disk state spill, host weight residency and KV below 8 bit are not admitted')
    for kind in ('image', 'video'):
        if cfg[f'{kind}_mode'] not in ('streaming', 'resident'):
            raise ValueError(f'{kind}_mode must be streaming or resident')
    root = Path(cfg['runtime_root']).expanduser()
    root = (root if root.is_absolute() else ROOT / root).resolve()
    cfg['runtime_root'] = str(root)
    if not cfg.get('gpu_uuid') and os.name == 'nt':
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.total', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10, check=True)
            devices = [(int(line.split(',')[1]), line.split(',')[0].strip()) for line in result.stdout.splitlines()]
            cfg['gpu_uuid'] = max(devices)[1]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    selection = root / 'state' / 'model_selection.json'
    if selection.exists() and not (overrides or {}).get('model'):
        saved = json.loads(selection.read_text())
        if saved.get('lora_applied') is not False:
            raise ValueError('Unsafe model selection state')
        cfg['model'] = saved['model']
    from .models import Models
    selected = Models(cfg).resolve()
    cfg.update(model=selected['path'], model_name=selected['id'])
    return cfg


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def environment(cfg, scope='studio'):
    root = Path(cfg['runtime_root'])
    root.mkdir(parents=True, exist_ok=True)
    from .envvars import apply
    env = apply(dict(os.environ), cfg, scope)
    for key in ('TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
                'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR',
                'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'UV_CACHE_DIR', 'PIP_CACHE_DIR',
                'DIFFUSERS_CACHE', 'TRANSFORMERS_CACHE',
                'KERNELS_CACHE', 'HF_KERNELS_CACHE', 'TORCHINDUCTOR_CACHE_DIR',
                'ZIG_GLOBAL_CACHE_DIR', 'ZIG_LOCAL_CACHE_DIR', 'CLANG_MODULE_CACHE_PATH', 'SWIFT_MODULECACHE_PATH'):
        directory = root / 'cache' / key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    for key, suffix in (('STUDIO_DATA_ROOT', 'studio'), ('STUDIO_OUTPUT_DIR', 'outputs'),
                        ('STUDIO_DATASETS_DIR', 'datasets'), ('STUDIO_LORAS_DIR', 'loras')):
        env[key] = str(root / suffix)
    env['STUDIO_RUNTIME_ROOT'] = str(root)
    env['STUDIO_HF_CACHE'] = str(root / 'cache' / 'hf_home')
    env['STUDIO_WF_DLL'] = str(path(cfg.get('weight_fetcher', 'build/llama/bin/slstack-weight-fetcher.dll')))
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               STUDIO_COSMOS_WINDOWS_PATH=str(path(cfg['streaming_root'])),
               STUDIO_COSMOS_PYTHON=cfg['studio_python'])
    if cfg.get('comfyui_models'):
        env['STUDIO_COMFYUI_PATH'] = str(path(cfg['comfyui_models']))
    if cfg.get('gpu_uuid'):
        env['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid']
        env['STUDIO_GPU_INDEX'] = cfg['gpu_uuid']
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(path(cfg['studio_root'])),
                                       str(path(cfg['streaming_root']))))
    return env
