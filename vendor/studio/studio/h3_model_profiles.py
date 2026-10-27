# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
H3_REFERENCE_SHORT_EDGE = 2048

@dataclass(frozen=True)
class H3ModelProfile:
    name: str
    lora: Optional[Path]
    text_embed: Optional[Path]
    sigma_shift: float
    steps: int
    reference_at_clip_short_edge: bool
    searched: tuple = ()
    worker_env: tuple = ()
_VIGGLE_LORA_NAMES = ('viggle_animate_dmd_lora_r64.safetensors', 'viggle_animate_dmd_lora.safetensors', 'pytorch_lora_weights.safetensors')
_VIGGLE_EMBED_NAMES = ('text_cond/fixed_embed_fwd_anyframe.safetensors',)

def _roots(selected: Path) -> tuple:
    base = selected.parent if selected.suffix else selected
    return tuple(dict.fromkeys((base, base.parent)))

def _first(roots, subdirs, names) -> Optional[Path]:
    for root in roots:
        for sub in subdirs:
            for name in names:
                candidate = root / sub / name if sub else root / name
                if candidate.is_file():
                    return candidate
    return None

def is_viggle_animate(model_dir) -> bool:
    return 'viggle' in str(model_dir).replace('\\', '/').lower()

def h3_model_profile(model_dir) -> Optional[H3ModelProfile]:
    if not model_dir or not is_viggle_animate(model_dir):
        return None
    roots = _roots(Path(str(model_dir)))
    lora = _first(roots, ('loras', 'lora', ''), _VIGGLE_LORA_NAMES)
    embed = _first(roots, ('',), _VIGGLE_EMBED_NAMES)
    return H3ModelProfile(name='viggle-animate', lora=lora, text_embed=embed, sigma_shift=3.0, steps=4, reference_at_clip_short_edge=True, searched=tuple((str(r) for r in roots)), worker_env=h3_checkpoint_worker_env(model_dir))
BASE_H3_STEP_ADAPTERS = frozenset((name.lower() for name in ('taomate_h3_3step_comfy.safetensors', 'minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors', 'minimax_h3_fl2v_turbo_4step_v1.1.safetensors', 'minimax_h3_ref2va_turbo_4step_v1.1.safetensors')))

def is_base_distill_adapter(path) -> bool:
    name = Path(str(path).replace('\\', '/')).name.lower()
    extra = {n.strip().lower() for n in os.environ.get('STUDIO_H3_STEP_ADAPTERS', '').split(';') if n.strip()}
    return name in BASE_H3_STEP_ADAPTERS or name in extra

def profile_adapters(profile: H3ModelProfile, requested) -> tuple:
    kept, removed = ([], [])
    for lora in requested or ():
        path = getattr(lora, 'path', lora)
        if is_base_distill_adapter(path):
            removed.append(Path(str(path)).name)
        else:
            kept.append(lora)
    has_own = any((is_viggle_animate(getattr(lora, 'path', lora)) for lora in kept))
    return (kept if has_own else [None, *kept], removed)

def clip_reference_scale(width: int, height: int) -> float:
    return min(1.0, min(int(width), int(height)) / float(H3_REFERENCE_SHORT_EDGE))
_SECTIONED_QKV = re.compile('viggle|(?<![a-z])(?:h3)?eros|minimax_h3_(?:ref2va|fl2va)_pruned_int8_convrot|minimax_h3_(?:fused_refdelta|turbo_adalnfix)', re.I)
QKV_SECTIONED_ENV = 'STUDIO_H3_COMFY_QKV_SECTIONED'

def qkv_sectioned_without_prefix(model_dir) -> bool:
    if not model_dir:
        return False
    root = Path(str(model_dir).replace('\\', '/'))
    names = [root.name]
    if root.is_dir():
        names.extend(p.name for p in (root/'transformer').glob('*.safetensors'))
    return any(_SECTIONED_QKV.search(name) for name in names) or is_viggle_animate(model_dir)

def h3_checkpoint_worker_env(model_dir) -> tuple:
    root=Path(str(model_dir))
    config=root/'transformer'/'config.json' if root.is_dir() else root.parent/'config.json'
    if config.is_file() and config.stat().st_size<=1024**2:
        layout=json.loads(config.read_text()).get('qkv_layout')
        if layout in ('sectioned','interleaved'):
            return ((QKV_SECTIONED_ENV,'1' if layout=='sectioned' else '0'),)
    return ((QKV_SECTIONED_ENV, '1'),) if qkv_sectioned_without_prefix(model_dir) else ()

def apply_h3_worker_env(env: dict, model_dir) -> dict:
    env.pop(QKV_SECTIONED_ENV, None)
    for name, value in h3_checkpoint_worker_env(model_dir):
        env[name] = value
    return env
START_FRAME_REFERENCE = 'reference'
START_FRAME_KEYFRAME = 'keyframe'

def start_frame_role(model_dir) -> str:
    return START_FRAME_REFERENCE if is_viggle_animate(model_dir or '') else START_FRAME_KEYFRAME

def start_frame_required(model_dir) -> bool:
    return is_viggle_animate(model_dir or '')

def _ref_path(ref) -> str:
    return str(getattr(ref, 'path', ref) or '')

def refs_with_start_frame(refs, start_frame):
    if refs is None or not start_frame:
        return refs
    own = next((r for r in refs if _ref_path(r) == start_frame), start_frame)
    return [own, *[r for r in refs if _ref_path(r) != start_frame]]

def keyframes_with_start_frame(keyframes, start_frame) -> list:
    out = [dict(k) for k in keyframes or ()]
    if not start_frame:
        return out
    for entry in out:
        if float(entry.get('seconds') or 0) == 0.0:
            entry['first'] = start_frame
            return out
    return [{'seconds': 0.0, 'first': start_frame, 'last': ''}, *out]
