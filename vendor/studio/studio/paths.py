# Derived from Studio backend/studio/paths.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence
APP_DIR_NAME = 'front-eagle-stack'

def is_windows() -> bool:
    return sys.platform.startswith('win')

def is_macos() -> bool:
    return sys.platform == 'darwin'

def platform_name() -> str:
    if is_windows():
        return 'windows'
    if is_macos():
        return 'macos'
    return 'linux'

def is_packaged() -> bool:
    if _env_bool('STUDIO_PACKAGED'):
        return True
    return bool(os.environ.get('STUDIO_RESOURCE_ROOT'))

def _env_bool(name: str, default: bool=False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'on', 'yes', 'y')

def app_data_root() -> Path:
    env = os.environ.get('STUDIO_DATA_ROOT')
    if env:
        return Path(env).expanduser()
    if is_windows():
        base = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        return Path(base) / APP_DIR_NAME
    if is_macos():
        return Path.home() / 'Library' / 'Application Support' / APP_DIR_NAME
    return Path.home() / f'.{APP_DIR_NAME}'

def resource_root() -> Optional[Path]:
    env = os.environ.get('STUDIO_RESOURCE_ROOT')
    return Path(env).expanduser() if env else None

def runtime_root() -> Path:
    env = os.environ.get('STUDIO_RUNTIME_ROOT')
    if env:
        return Path(env).expanduser()
    res = resource_root()
    if res is not None:
        return res / 'runtime'
    return app_data_root() / 'runtime'

def installer_python(name: str) -> Path:
    root = runtime_root() / name
    if is_windows():
        candidates = [root / 'Scripts' / 'python.exe', root / 'python.exe']
    else:
        candidates = [root / 'bin' / 'python', root / 'bin' / 'python3', root / 'python']
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return candidates[0]

def _first_existing(candidates: Sequence[Path], *, want: str) -> Optional[Path]:
    for c in candidates:
        try:
            if want == 'file' and c.is_file():
                return c
            if want == 'dir' and c.is_dir():
                return c
            if want == 'any' and c.exists():
                return c
        except OSError:
            continue
    return None

def resolve(*, env: Optional[str]=None, settings_value: Optional[str]=None, installer_default: Optional[Path]=None, dev_candidates: Sequence[str | Path]=(), want: str='any', settings_getter: Optional[Callable[[], Optional[str]]]=None) -> Path:
    if env:
        v = os.environ.get(env)
        if v:
            return Path(v).expanduser()
    sv = settings_value
    if sv is None and settings_getter is not None:
        try:
            sv = settings_getter()
        except Exception:
            sv = None
    if sv:
        return Path(str(sv)).expanduser()
    devs = [Path(str(d)).expanduser() for d in dev_candidates]
    if installer_default is not None:
        try:
            inst_exists = want == 'file' and installer_default.is_file() or (want == 'dir' and installer_default.is_dir()) or (want == 'any' and installer_default.exists())
        except OSError:
            inst_exists = False
        if inst_exists or is_packaged():
            return installer_default
    hit = _first_existing(devs, want=want)
    if hit is not None:
        return hit
    if devs:
        return devs[0]
    if installer_default is not None:
        return installer_default
    return app_data_root()

def data_subdir(name: str, env: Optional[str]=None) -> Path:
    if env:
        v = os.environ.get(env)
        if v:
            return Path(v).expanduser()
    return app_data_root() / name

def hf_cache_dir() -> Path:
    return resolve(env='STUDIO_HF_CACHE', installer_default=app_data_root() / 'hf_cache', dev_candidates=[''] if is_windows() else [], want='dir')

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def vendored_mlx_vlm_dir() -> Optional[Path]:
    env = os.environ.get('STUDIO_VENDORED_MLX_VLM')
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(repo_root() / 'backend' / 'vendored' / 'mlx_vlm')
    res = resource_root()
    if res is not None:
        candidates.append(res / 'backend' / 'vendored' / 'mlx_vlm')
        candidates.append(res / 'runtime' / 'mlx_vlm')
    candidates.append(runtime_root() / 'mlx_vlm')
    for c in candidates:
        try:
            if (c / 'pyproject.toml').is_file() and (c / 'mlx_vlm').is_dir():
                return c
        except OSError:
            continue
    return None
