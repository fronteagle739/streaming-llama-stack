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
    if cfg['lease_minutes'] > 20:
        raise ValueError('GPU lease must be at most 20 minutes')
    if cfg['host'] not in ('127.0.0.1', '::1'):
        raise ValueError('This local stack binds to loopback only')
    for kind in ('image', 'video'):
        if cfg[f'{kind}_mode'] not in ('streaming', 'resident'):
            raise ValueError(f'{kind}_mode must be streaming or resident')
    root = Path(cfg['runtime_root']).resolve()
    if os.name == 'nt' and not root.is_relative_to(Path('runtime/ds4stack').resolve()):
        raise ValueError('All runtime writes must be under runtime/ds4stack')
    cfg['runtime_root'] = str(root)
    selection = root / 'state' / 'model_selection.json'
    if selection.exists() and not (overrides or {}).get('model'):
        saved = json.loads(selection.read_text())
        if saved.get('lora_applied') is not False:
            raise ValueError('Unsafe model selection state')
        cfg['model'] = saved['model']
    return cfg


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def environment(cfg):
    root = Path(cfg['runtime_root'])
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ('TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
                'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR',
                'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'UV_CACHE_DIR', 'PIP_CACHE_DIR'):
        directory = root / 'cache' / key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    for key, suffix in (('STUDIO_DATA_ROOT', 'studio'), ('STUDIO_OUTPUT_DIR', 'outputs'),
                        ('STUDIO_DATASETS_DIR', 'datasets'), ('STUDIO_LORAS_DIR', 'loras')):
        env[key] = str(root / suffix)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               STUDIO_COSMOS_WINDOWS_PATH=str(path(cfg['streaming_root'])),
               STUDIO_COSMOS_PYTHON=cfg['studio_python'])
    if cfg.get('comfyui_models'):
        env['STUDIO_COMFYUI_PATH'] = str(path(cfg['comfyui_models']))
    if cfg.get('gpu_uuid'):
        env['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid']
        env['STUDIO_GPU_INDEX'] = cfg['gpu_uuid']
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(path(cfg['studio_root']) / 'backend'),
                                       str(path(cfg['streaming_root']))))
    return env
