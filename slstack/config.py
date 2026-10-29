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
    from .multichunk import selection
    selection(cfg.get('producer_chunks_per_pass', 1))
    from .singlepass import registration
    registration(cfg.get('producer_carry_registration', 'tile'))
    from .combo_ram import runtime_min_free_gib
    cfg['runtime_min_free_gib'] = runtime_min_free_gib(cfg)
    if cfg.get('mac'):
        from .pairing import endpoint
        cfg['mac'] = endpoint(cfg['mac'])
    if 'paired_local_prefill_max_tokens' in cfg:
        from .combo_crossover import paired_threshold
        paired_threshold(cfg)
    cfg.setdefault('yoco_keep',2541 if cfg.get('mac') or cfg.get('serving','single_engine')=='single_engine' else 0)
    keep=cfg['yoco_keep']
    if type(keep) is not int or keep not in (0,140,2541):
        raise ValueError('yoco_keep must be 0, 140 or 2541')
    if keep and not cfg.get('mac') and cfg.get('serving','single_engine')!='single_engine':
        raise ValueError('YOCO needs single_engine or an explicitly paired Mac receiver')
    if cfg.get('native_tier', 'accurate') not in ('accurate', 'fast', 'ultra', 'hyper'):
        raise ValueError('mode must be accurate, fast, ultra or hyper')
    if cfg.get('serving', 'single_engine') not in ('single_engine', 'native'):
        raise ValueError('serving must be single_engine or native')
    for key in ('host_budget_gib', 'min_available_gib', 'producer_desktop_reserve_gib', 'job_commit_gib', 'request_timeout_s', 'lease_minutes'):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if type(cfg.get('gpu_min_free_mib', 0)) is not int or cfg.get('gpu_min_free_mib', 0) < 0:
        raise ValueError('gpu_min_free_mib must be a nonnegative integer')
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
    if cfg.get('h3_attention_backend','sdpa') not in ('sdpa','sageattention','flash_attn2'):
        raise ValueError('h3_attention_backend must be sdpa, sageattention or flash_attn2')
    root = Path(cfg['runtime_root']).expanduser()
    root = (root if root.is_absolute() else ROOT / root).resolve()
    cfg['runtime_root'] = str(root)
    cfg.setdefault('generation_control_file', str(root / 'generation' / 'live.json'))
    cfg.setdefault('stack_preferences_file', str(root / 'settings' / 'stack.local.json'))
    if not cfg.get('studio_python'):
        cfg['studio_python'] = str(root / 'media-venv' /
                                   ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
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
    configure_vision(cfg)
    return cfg


def configure_vision(cfg):
    """Resolve the encoder for a vision-capable launch after model selection."""
    if cfg.get('vision_bridge_version') != 1:return cfg
    encoder=cfg.get('vision_encoder') or cfg.get('ds4_vision')
    argv=cfg.get('ds4_command') or []
    if not encoder and '--vision' in argv:
        index=argv.index('--vision')
        if index+1>=len(argv):raise ValueError('--vision needs an encoder GGUF path')
        encoder=argv[index+1]
    if encoder:cfg['vision_encoder']=str(path(encoder))
    elif cfg.get('model'):
        candidates=list(Path(cfg['model']).parent.glob('*-Vision.gguf'))
        if len(candidates)==1:cfg['vision_encoder']=str(candidates[0].resolve())
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
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env
