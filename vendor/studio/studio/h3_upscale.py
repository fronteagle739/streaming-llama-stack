# Derived from Studio backend/studio/h3_upscale.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from dataclasses import dataclass
import math
from pathlib import Path, PureWindowsPath
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class H3UpscaleSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    mode: Literal['two_pass', 'video'] = 'two_pass'
    low_width: int = Field(608, ge=32, le=8192)
    low_height: int = Field(352, ge=32, le=8192)
    width: int = Field(1376, ge=32, le=8192)
    height: int = Field(768, ge=32, le=8192)
    first_steps: int = Field(8, ge=2, le=200)
    split: int = Field(4, ge=1, le=199)
    second_steps: int = Field(3, ge=1, le=200)
    denoise: float = Field(1, ge=0, le=1)
    schedule: Literal['workflow', 'continuation'] = 'workflow'
    model: str = ''
    source_video: str = ''
    turbo: bool = True

def validate_upscale(cfg: H3UpscaleSettings):
    for value in (cfg.low_width, cfg.low_height, cfg.width, cfg.height):
        if value % 32:
            raise ValueError('H3 pass dimensions must be multiples of 32')
    if cfg.split >= cfg.first_steps:
        raise ValueError('Sigma split must be smaller than first-pass schedule steps')
    if cfg.mode == 'two_pass':
        if not all((1 <= high / low <= 4 for low, high in ((cfg.low_width, cfg.width), (cfg.low_height, cfg.height)))):
            raise ValueError('H3 latent upscale must be between 1x and 4x on each axis')
    elif not cfg.source_video.strip():
        raise ValueError('H3 video upscaling requires a source video')

@dataclass(frozen=True)
class SigmaPlan:
    low: list[float]
    high: list[float]

def _resample(values, steps):
    result = []
    for i in range(steps + 1):
        x = i * (len(values) - 1) / steps
        left = min(int(x), len(values) - 2)
        result.append(values[left] + (values[left + 1] - values[left]) * (x - left))
    return result

def sigma_plan(cfg: H3UpscaleSettings) -> SigmaPlan:
    validate_upscale(cfg)
    raw = [1 - i / cfg.first_steps for i in range(cfg.first_steps + 1)]
    full = [12 * s / (1 + 11 * s) for s in raw]
    low = full[:cfg.split + 1]
    tail = [0.9035, 0.6316, 0.3158, 0] if cfg.schedule == 'workflow' else full[cfg.split:]
    high = [s * cfg.denoise for s in _resample(tail, cfg.second_steps)]
    return SigmaPlan(low, high)

def require_fast_weights(path, *, directory=False) -> Path:
    p = Path(path)
    for name in (str(p), str(p.resolve())):
        if PureWindowsPath(name).drive.upper() not in ('D:', 'L:'):
            raise ValueError(f'H3 weights must stream only from D: or L: ({p})')
    if directory:
        if not p.is_dir():
            raise FileNotFoundError(p)
    else:
        if p.suffix.lower() != '.safetensors':
            raise ValueError('H3 weights require complete .safetensors files')
        if not p.is_file() or Path(str(p) + '.part').exists():
            raise FileNotFoundError(f'H3 weights missing or still downloading: {p}')
        import json, struct
        with p.open('rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(f.read(header_size))
        end = max((v['data_offsets'][1] for k, v in header.items() if k != '__metadata__'))
        if p.stat().st_size != 8 + header_size + end:
            raise ValueError(f'H3 checkpoint incomplete (still downloading): {p}')
    return p

def turbo_path(component):
    filename = 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors' if component.lower() == 'ref2va' else 'minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors'
    return Path('') / filename
