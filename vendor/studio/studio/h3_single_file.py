# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional
H3_COMPONENTS = ('text_encoder', 'tokenizer', 'video_vae', 'audio_vae', 'processor')
DEFAULT_BASE_CANDIDATES = ('', '', '')

def looks_like_single_file_checkpoint(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == '.safetensors'

def find_component_base(candidates: Iterable[str]=DEFAULT_BASE_CANDIDATES) -> Optional[Path]:
    for candidate in candidates:
        base = Path(candidate)
        if base.is_dir() and all(((base / name).exists() for name in H3_COMPONENTS)):
            return base
    return None

def _link_dir(link: Path, target: Path) -> None:
    if link.exists():
        return
    if sys.platform == 'win32':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True, text=True)
    else:
        link.symlink_to(target, target_is_directory=True)

def _link_file(link: Path, target: Path) -> None:
    if link.exists():
        return
    try:
        os.link(target, link)
    except OSError:
        import shutil
        shutil.copy2(target, link)

def compose_single_file_model_dir(checkpoint: Path, *, base: Optional[Path]=None, cache_root: Optional[Path]=None) -> Path:
    checkpoint = Path(checkpoint)
    if not looks_like_single_file_checkpoint(checkpoint):
        raise ValueError(f'not a single-file H3 checkpoint: {checkpoint}')
    base = Path(base) if base is not None else find_component_base()
    if base is None:
        raise RuntimeError('no base MiniMax-H3 install was found to borrow components from. A single-file H3 checkpoint carries only the transformer; the text encoder, tokenizer, video VAE, audio VAE and processor come from a full install. Register or download one, then retry.')
    missing = [name for name in H3_COMPONENTS if not (base / name).exists()]
    if missing:
        raise RuntimeError(f"base H3 install {base} is missing {', '.join(missing)}")
    root = Path(cache_root) if cache_root else checkpoint.parent / '.h3-composed'
    model_dir = root / checkpoint.stem
    transformer = model_dir / 'transformer'
    transformer.mkdir(parents=True, exist_ok=True)
    _link_file(transformer / checkpoint.name, checkpoint)
    manifest = transformer / 'model.safetensors.index.json'
    if not manifest.exists():
        with checkpoint.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('H3 checkpoint has a truncated safetensors header')
            header_size = struct.unpack('<Q', raw)[0]
            if not 0 < header_size <= 16 * 1024 * 1024:
                raise ValueError('H3 checkpoint has an invalid safetensors header size')
            header = json.loads(stream.read(header_size))
        tensors = {name: info for name, info in header.items() if name != '__metadata__'}
        if not tensors:
            raise ValueError('H3 checkpoint contains no tensor records')
        payload_size = checkpoint.stat().st_size - 8 - header_size
        total_size = 0
        for name, info in tensors.items():
            first, last = info['data_offsets']
            if not 0 <= first <= last <= payload_size:
                raise ValueError(f'H3 checkpoint tensor {name} exceeds its file payload')
            total_size += last - first
        manifest.write_text(json.dumps({'metadata': {'total_size': total_size}, 'weight_map': {name: checkpoint.name for name in tensors}}, indent=2), encoding='utf-8')
    base_config = base / 'transformer' / 'config.json'
    target_config = transformer / 'config.json'
    if base_config.is_file() and (not target_config.exists()):
        target_config.write_text(base_config.read_text(encoding='utf-8'), encoding='utf-8')
    for name in H3_COMPONENTS:
        _link_dir(model_dir / name, base / name)
    return model_dir

def describe(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    transformer = model_dir / 'transformer'
    weights = sorted((p.name for p in transformer.glob('*.safetensors'))) if transformer.is_dir() else []
    return {'model_dir': str(model_dir), 'transformer_weights': weights, 'components': [n for n in H3_COMPONENTS if (model_dir / n).exists()], 'config': (transformer / 'config.json').is_file()}
