"""Portable family-to-environment routing, matching Studio's isolated workers."""
from pathlib import Path
import os

PROFILES = {
    'studio': dict(directory='media-venv', lock='media-win-py312.txt'),
    'qwen21': dict(directory='media-qwen21-venv', lock='media-qwen21-win-py312.txt'),
}


def profile_for(arch):
    return 'qwen21' if (arch or '').startswith(('qwen-image-2.1','qwenimage2.1')) else 'studio'


def python_for(cfg, arch):
    profile=profile_for(arch)
    if profile=='studio' and cfg.get('studio_python'):return cfg['studio_python']
    root=Path(cfg['runtime_root'])/PROFILES[profile]['directory']
    return str(root/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
