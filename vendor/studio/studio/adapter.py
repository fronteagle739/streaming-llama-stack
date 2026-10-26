# Derived from Studio backend/studio/adapter.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import importlib
import json
import math
import os
import re
import shutil
import subprocess
from . import worker_registry
import sys
import time
import uuid
from types import SimpleNamespace
import wave
from pathlib import Path
from typing import Optional, Protocol
from .config import Settings
from .chat_artifacts import output_dir_for_params
from .models import GenResult, ModelInfo

def failure_tail(lines, keep: int=20, limit: int=1600) -> str:
    rows = [str(x).rstrip() for x in lines or [] if str(x).strip()]

    def is_progress_noise(r: str) -> bool:
        if not (r.lstrip().startswith('{') and '"event"' in r):
            return False
        return not any((k in r for k in ('"failed"', '"error"', 'error_type', 'traceback')))

    def looks_like_failure(r: str) -> bool:
        return any((k in r for k in ('"failed"', 'error_type', 'Traceback', 'Error:', 'ERROR', 'Exception')))
    human = [r for r in rows if not is_progress_noise(r)]
    chosen = human if human else rows
    failures = [r for r in chosen if looks_like_failure(r)]
    context = [r for r in chosen if r not in failures][-keep:]
    ordered = context + failures[-3:] if failures else chosen[-keep:]
    return '\n'.join(ordered)[-limit:]

def _clean_caption_text(value, fallback: str='') -> str:
    text = str(value or '').strip()
    return text or fallback

def _clamp_ideogram_coord(value, lo: int=0, hi: int=1000) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return lo

def _ideogram_bbox(box) -> list[int]:
    x0 = _clamp_ideogram_coord(getattr(box, 'x', 0), 0, 999)
    y0 = _clamp_ideogram_coord(getattr(box, 'y', 0), 0, 999)
    x1 = _clamp_ideogram_coord(float(getattr(box, 'x', 0)) + float(getattr(box, 'width', 1)), x0 + 1, 1000)
    y1 = _clamp_ideogram_coord(float(getattr(box, 'y', 0)) + float(getattr(box, 'height', 1)), y0 + 1, 1000)
    return [y0, x0, y1, x1]

def _normalize_hex_color(value) -> str | None:
    s = str(value or '').strip()
    if not s:
        return None
    if re.fullmatch('#[0-9a-fA-F]{6}', s):
        return s.upper()
    return None

def _palette(values, limit: int) -> list[str]:
    out: list[str] = []
    for value in values or []:
        color = _normalize_hex_color(value)
        if color and color not in out:
            out.append(color)
        if len(out) >= limit:
            break
    return out

def _resolve_glm_emit_bin(repo: str | None=None) -> str:
    override = os.environ.get('STUDIO_GLM_EMIT_BIN')
    if override:
        return override
    roots: list[Path] = []
    if repo:
        roots.append(Path(repo) / 'prefill_llama.cpp')
    roots.append(Path(''))
    seen: set[str] = set()
    builds = ('build', 'build_kvfix_clone', 'build_kvfix', 'build4', 'build3', 'build2')
    for root in roots:
        key = os.path.normcase(os.path.abspath(str(root)))
        if key in seen:
            continue
        seen.add(key)
        for build in builds:
            cand = root / build / 'bin' / 'llama-glm-emit.exe'
            if cand.is_file():
                return str(cand)
    return str(roots[0] / 'build' / 'bin' / 'llama-glm-emit.exe')

def _composition_guide_to_ideogram_prompt(guide, prompt: str) -> str:
    ideogram = getattr(guide, 'ideogram', None)
    high_level = _clean_caption_text(getattr(ideogram, 'high_level_description', None) if ideogram else None, _clean_caption_text(prompt, 'Generated image'))
    caption: dict[str, object] = {'high_level_description': high_level}
    aesthetics = _clean_caption_text(getattr(ideogram, 'aesthetics', None) if ideogram else None)
    lighting = _clean_caption_text(getattr(ideogram, 'lighting', None) if ideogram else None)
    medium = _clean_caption_text(getattr(ideogram, 'medium', None) if ideogram else None)
    art_style = _clean_caption_text(getattr(ideogram, 'art_style', None) if ideogram else None)
    colors = _palette(getattr(ideogram, 'color_palette', []) if ideogram else [], 16)
    if any((aesthetics, lighting, medium, art_style, colors)):
        medium_l = medium.lower()
        style: dict[str, object] = {'aesthetics': aesthetics or 'coherent, high quality, visually balanced', 'lighting': lighting or 'natural, well-shaped lighting'}
        if 'photo' in medium_l or 'photograph' in medium_l:
            style['photo'] = art_style or 'natural lens perspective'
            style['medium'] = medium or 'photograph'
        else:
            style['medium'] = medium or 'illustration'
            style['art_style'] = art_style or aesthetics or 'detailed digital illustration'
        if colors:
            style['color_palette'] = colors
        caption['style_description'] = style
    background = _clean_caption_text(getattr(guide, 'background', ''), f'background and environment implied by: {high_level}')
    elements = []
    for i, box in enumerate(getattr(guide, 'boxes', []) or [], start=1):
        box_type = 'text' if str(getattr(box, 'type', 'obj')).lower() == 'text' else 'obj'
        desc = _clean_caption_text(getattr(box, 'desc', None), _clean_caption_text(getattr(box, 'label', None), f'element {i}'))
        element: dict[str, object] = {'type': box_type, 'bbox': _ideogram_bbox(box)}
        if box_type == 'text':
            text = _clean_caption_text(getattr(box, 'text', None), desc)
            element['text'] = text
        element['desc'] = desc
        elem_colors = _palette([*(getattr(box, 'colors', []) or []), getattr(box, 'color', None)], 5)
        if elem_colors:
            element['color_palette'] = elem_colors
        elements.append(element)
    if not elements:
        elements.append({'type': 'obj', 'desc': high_level})
    caption['compositional_deconstruction'] = {'background': background, 'elements': elements}
    return json.dumps(caption, ensure_ascii=False, separators=(',', ':'))

def mux_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path | None=None, *, should_cancel=None, duration: float | None=None, ffmpeg_path: str | None=None) -> Path:
    video = Path(video_path)
    audio = Path(audio_path)
    if not video.exists():
        raise FileNotFoundError(f'video file not found: {video}')
    if not audio.exists():
        raise FileNotFoundError(f'audio file not found: {audio}')
    ffmpeg = ffmpeg_path or shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required for audio_mode=mux but was not found on PATH')
    out = Path(output_path) if output_path is not None else video.with_name(f'{video.stem}_with_audio{video.suffix}')
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, '-n' if should_cancel is not None else '-y', '-i', str(video), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(out)]
    if duration is not None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Audio mux duration must be finite and positive')
        cmd[-1:-1] = ['-af', 'apad', '-t', str(duration)]
    if should_cancel is None:
        res = subprocess.run(cmd, capture_output=True, text=True)
    else:
        if should_cancel():
            raise RuntimeError('Audio mux cancelled before encoding')
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 120
        try:
            while True:
                if should_cancel():
                    raise RuntimeError('Audio mux cancelled')
                if time.monotonic() > deadline:
                    raise TimeoutError('Audio mux exceeded its 120 second encoding budget')
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            res = SimpleNamespace(returncode=process.returncode, stdout=stdout, stderr=stderr)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or '').strip()[-500:]
        raise RuntimeError(f'ffmpeg audio mux failed: {detail}')
    return out

def _detect_ideo_quant_from_weights(model_path: str) -> str | None:
    try:
        root = Path(model_path or '')
        if not root.is_dir():
            return None
        cands = sorted(root.glob('transformer/*.safetensors')) or sorted(root.glob('**/*.safetensors'))
        if not cands:
            return None
        with open(cands[0], 'rb') as fh:
            import struct as _struct
            n = _struct.unpack('<Q', fh.read(8))[0]
            if n <= 0 or n > 50000000:
                return None
            hdr = json.loads(fh.read(n).decode('utf-8', 'ignore'))
        for k, v in hdr.items():
            if k == '__metadata__':
                continue
            if isinstance(v, dict) and str(v.get('dtype', '')).upper().startswith('F8'):
                return 'fp8'
            if any((t in k for t in ('absmax', 'quant_state', 'quant_map', 'bitsandbytes'))):
                return 'nf4'
    except Exception:
        return None
    return None

def _infer_ideo_quant(params) -> str:
    hay = f"{params.model or ''} {getattr(params, 'model_arch', '') or ''}".lower()
    if 'fp8' in hay:
        return 'fp8'
    if 'nf4' in hay:
        return 'nf4'
    detected = _detect_ideo_quant_from_weights(str(params.model or ''))
    if detected:
        return detected
    return str(getattr(params, 'ideo_quant', 'nf4') or 'nf4')

def _sdedit_i2i_enabled(backend: str) -> bool:
    off = ('0', 'false', 'off', 'no')
    if os.environ.get('STUDIO_SDEDIT_I2I', '1').strip().lower() in off:
        return False
    if os.environ.get(f'STUDIO_SDEDIT_I2I_{backend.upper()}', '1').strip().lower() in off:
        return False
    return True

def _resolve_denoise_strength(params, default: float=0.6) -> float:
    d = getattr(params, 'denoise', None)
    try:
        return max(0.0, min(1.0, float(d))) if d is not None else float(default)
    except (TypeError, ValueError):
        return float(default)
LTX_360_DEFAULT_TRIGGER = '360-equirectangular'
LTX_360_DEFAULT_PROMPT_PREFIX = '360-equirectangular, 360 degree equirectangular panorama, '

def _is_ltx_360_mode(params) -> bool:
    return str(getattr(params, 'ltx_mode', '') or '').strip().lower() == '360'

def _snap_multiple(value, mult: int=32) -> int:
    try:
        v = int(round(float(value) / mult)) * mult
    except (TypeError, ValueError):
        v = mult
    return max(mult, v)

def _apply_ltx_360_recipe(params, support_files, loras, prompt, width, height):
    if not _is_ltx_360_mode(params):
        return (prompt, width, height, None)
    sf = dict(support_files or {})
    lora_path = str(sf.get('lora_360') or '').strip()
    if lora_path and Path(lora_path).is_file():
        try:
            strength = float(params.ltx_360_strength) if getattr(params, 'ltx_360_strength', None) is not None else float(sf.get('lora_360_strength') or 1.0)
        except (TypeError, ValueError):
            strength = 1.0
        name = re.sub('[^0-9A-Za-z_]+', '_', Path(lora_path).stem[:48]).strip('_') or 'ltx_360'
        if not any((str((l or {}).get('path') or '') == lora_path for l in loras)):
            loras.insert(0, {'path': lora_path, 'weight': strength, 'name': name})
    trigger = str(sf.get('lora_360_trigger') or LTX_360_DEFAULT_TRIGGER).strip()
    prefix = str(sf.get('lora_360_prompt_prefix') or LTX_360_DEFAULT_PROMPT_PREFIX)
    base = prompt or ''
    if trigger and trigger.lower() not in base.lower():
        prompt = f'{prefix}{base}'.strip()
    w = int(width or 0)
    h = int(height or 0)
    if w <= 0 and h <= 0:
        w = 1920
    elif w <= 0:
        w = h * 2
    w = _snap_multiple(w)
    h = _snap_multiple(w // 2)
    return (prompt, w, h, 'equirectangular')

def _save_waveform_wav(audio, path: str | Path, sample_rate: int=24000) -> None:
    path = Path(path)
    sample_rate = int(sample_rate or 24000)
    try:
        import torchaudio
        tensor = audio
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach().float().cpu()
        while getattr(tensor, 'ndim', 0) > 2 and int(tensor.shape[0]) == 1:
            tensor = tensor.squeeze(0)
        if getattr(tensor, 'ndim', 0) == 1:
            tensor = tensor.unsqueeze(0)
        elif getattr(tensor, 'ndim', 0) == 2 and int(tensor.shape[0]) > 8 and (int(tensor.shape[1]) <= 8):
            tensor = tensor.transpose(0, 1)
        torchaudio.save(str(path), tensor.float(), sample_rate=sample_rate)
        return
    except ModuleNotFoundError:
        pass
    except Exception:
        pass
    tensor = audio
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().float().cpu()
    while getattr(tensor, 'ndim', 0) > 2 and int(tensor.shape[0]) == 1:
        tensor = tensor.squeeze(0)
    data = tensor.tolist() if hasattr(tensor, 'tolist') else tensor
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list) or not data:
        raise RuntimeError('generated audio waveform is empty')
    if isinstance(data[0], (int, float)):
        channels = [data]
    elif isinstance(data[0], list) and len(data) <= 8:
        channels = data
    elif isinstance(data[0], list) and data[0] and (len(data[0]) <= 8):
        channels = [list(ch) for ch in zip(*data)]
    else:
        raise RuntimeError('generated audio waveform has an unsupported shape')
    channels = [list(ch) for ch in channels if ch]
    if not channels:
        raise RuntimeError('generated audio waveform has no samples')
    n_samples = min((len(ch) for ch in channels))
    frames = bytearray()
    for i in range(n_samples):
        for ch in channels:
            sample = float(ch[i])
            if not math.isfinite(sample):
                sample = 0.0
            sample = max(-1.0, min(1.0, sample))
            scale = 32767 if sample >= 0 else 32768
            frames.extend(int(round(sample * scale)).to_bytes(2, 'little', signed=True))
    with wave.open(str(path), 'wb') as fh:
        fh.setnchannels(len(channels))
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(frames)

class EngineBackend(Protocol):
    name: str

    def info(self) -> dict:
        ...

    def list_models(self) -> list[ModelInfo]:
        ...

    def select_model(self, name: str) -> None:
        ...

    def generate(self, params, on_step=None, should_cancel=None) -> GenResult:
        ...

class UnavailableBackend:
    name = 'unavailable'

    def __init__(self, settings: Settings, detail: str) -> None:
        self.settings = settings
        self.detail = detail

    def info(self) -> dict:
        return {'engine': self.name, 'model': None, 'device': None, 'detail': self.detail}

    def list_models(self) -> list[ModelInfo]:
        return []

    def select_model(self, name: str) -> None:
        raise RuntimeError(f'real backend unavailable; cannot select {name!r}. {self.detail}')

    def generate(self, params, *args, **kwargs) -> GenResult:
        raise RuntimeError(f'real backend unavailable; generation cannot start. {self.detail}')
_CONTROL_CAPABLE_ARCHES: frozenset = frozenset({'sdxl'})
H3_SPATIAL_STRIDE = 32
H3_DEFAULT_MAX_SHORT_EDGE = 768

def _snap_to_stride(v: int) -> int:
    return max(H3_SPATIAL_STRIDE, int(round(v / H3_SPATIAL_STRIDE)) * H3_SPATIAL_STRIDE)

def h3_clamp_dimensions(width, height, *, max_short_edge=None):
    try:
        w, h = (int(width or 0), int(height or 0))
    except (TypeError, ValueError):
        return (width, height, None)
    if w <= 0 or h <= 0:
        return (width, height, None)
    cap = max_short_edge
    if cap is None:
        try:
            cap = int(os.environ.get('STUDIO_H3_MAX_SHORT_EDGE', H3_DEFAULT_MAX_SHORT_EDGE))
        except (TypeError, ValueError):
            cap = H3_DEFAULT_MAX_SHORT_EDGE
    cap = max(H3_SPATIAL_STRIDE, cap // H3_SPATIAL_STRIDE * H3_SPATIAL_STRIDE)
    if min(w, h) <= cap:
        aw, ah = (_snap_to_stride(w), _snap_to_stride(h))
        if (aw, ah) == (w, h):
            return (w, h, None)
        return (aw, ah, f"{w}x{h} is not on H3's {H3_SPATIAL_STRIDE}px grid; rendering at {aw}x{ah}")
    scale = cap / float(min(w, h))

    def _snap(v):
        return max(H3_SPATIAL_STRIDE, int(v * scale) // H3_SPATIAL_STRIDE * H3_SPATIAL_STRIDE)
    nw, nh = (_snap(w), _snap(h))
    if min(nw, nh) > cap:
        if nw <= nh:
            nw = cap
        else:
            nh = cap
    return (nw, nh, f"{w}x{h} exceeds H3's {cap}px short edge; rendering at {nw}x{nh}")

def _cosmos_model_request(model: str, builtins) -> bool:
    if model in builtins:
        return True
    try:
        return Path(model).exists()
    except OSError:
        return False

class RealCosmosBackend:
    name = 'real'

    def __init__(self, settings: Settings, cosmos_cli) -> None:
        self.settings = settings
        self._cc = cosmos_cli
        self._cli = cosmos_cli.Cli(dict(cosmos_cli.DEFAULTS))
        self._cli.st['gpu_index'] = settings.gpu_index
        self._cli.st['out'] = str(settings.output_dir)
        self._resident_aux_pipe = None
        self._resident_aux_key = None
        self._resident_aux_label = None
        self._glm_worker = None
        self._vram_mgr = None
        settings.ensure_dirs()

    def _cuda_device(self) -> str:
        selected = str(self._cli.st.get('gpu_index', self.settings.gpu_index) or '0')
        torch_idx: int | None = None
        try:
            import torch
            if torch.cuda.is_available():
                if selected.isdigit():
                    try:
                        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL).decode().strip().splitlines()
                        target = None
                        for line in rows:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 2 and parts[0] == selected:
                                target = {'uuid': parts[1].replace('GPU-', '').lower(), 'name': parts[2] if len(parts) >= 3 else '', 'memory_mib': int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else 0}
                                break
                        if target:
                            for i in range(torch.cuda.device_count()):
                                props = torch.cuda.get_device_properties(i)
                                uuid = str(getattr(props, 'uuid', '') or '').replace('GPU-', '').lower()
                                if uuid and uuid == target['uuid']:
                                    torch_idx = i
                                    break
                            if torch_idx is None:
                                for i in range(torch.cuda.device_count()):
                                    props = torch.cuda.get_device_properties(i)
                                    total_mib = int(getattr(props, 'total_memory', 0) / (1024 * 1024))
                                    if props.name == target['name'] and abs(total_mib - target['memory_mib']) < 512:
                                        torch_idx = i
                                        break
                    except Exception:
                        torch_idx = None
                    if torch_idx is None:
                        torch_idx = int(selected)
                if torch_idx is not None:
                    torch.cuda.set_device(torch_idx)
        except Exception:
            pass
        return f'cuda:{torch_idx}' if torch_idx is not None else 'cuda'

    def _apply_gpu_assignment(self, params, arch: str) -> None:
        try:
            from . import settings_store
        except Exception:
            return
        cli = getattr(self, '_cli', None)
        if cli is None or not hasattr(cli, 'st'):
            return
        model_name = str(getattr(params, 'model', '') or '')
        arch_bucket = (arch or '').split('-')[0] if arch else ''
        idx = settings_store.gpu_for(model_name, arch or '', arch_bucket)
        if idx is None or idx == 'cpu':
            cli.st['gpu_index'] = str(getattr(self.settings, 'gpu_index', '0'))
            return
        cli.st['gpu_index'] = idx

    def _get_vram_mgr(self):
        if self._vram_mgr is None:
            from .vram_residency import VramResidencyManager
            self._vram_mgr = VramResidencyManager(device=self._cuda_device())
        return self._vram_mgr

    def info(self) -> dict:
        mgr_resident = bool(self._vram_mgr is not None and self._vram_mgr.resident_keys())
        resident = bool(getattr(self._cli, '_pipe', None) is not None or getattr(self._cli, '_streamer', None) is not None or self._resident_aux_pipe is not None or mgr_resident)
        current = self._resident_aux_label or (self._cli.current if resident else None)
        return {'engine': 'real', 'model': current, 'configured_model': self._cli.st.get('model'), 'current': current, 'device': 'cuda' if resident else None}

    def list_models(self) -> list[ModelInfo]:
        from .stream_capability import annotate_note
        from . import model_overrides
        out: list[ModelInfo] = []
        for name, m in self._cc.MODELS.items():
            p = self._builtin_model_path(name)
            arch = m.get('arch') or m.get('arch_hint')
            if name == 'ideogram-4' and (not arch):
                arch = 'ideogram-4'
            out.append(model_overrides.apply(ModelInfo(name=name, label=m['label'], status=self._cc.Cli._model_status(p), modalities=m['modalities'], current=name == self._cli.current, path=str(p), note=annotate_note(arch, m.get('note')), arch=arch)))
        return out

    def _builtin_model_path(self, name: str) -> Path:
        m = self._cc.MODELS[name]
        default = Path(self._cc.MODELS_ROOT) / m['dir']
        try:
            from . import model_overrides
            return Path(model_overrides.resolve_path(name, default))
        except Exception:
            return default

    def _selected_builtin_path_matches(self, name: str) -> bool:
        try:
            current = os.path.normcase(os.path.abspath(str(self._cli.st.get('model') or '')))
            wanted = os.path.normcase(os.path.abspath(str(self._builtin_model_path(name))))
            return bool(current) and current == wanted
        except Exception:
            return False

    def select_model(self, name: str) -> None:
        if name not in self._cc.MODELS:
            raise ValueError(f'unknown model {name!r}')
        self._clear_resident_aux_pipe()
        m = self._cc.MODELS[name]
        p = self._builtin_model_path(name)
        status = self._cc.Cli._model_status(p)
        if status == 'ABSENT':
            print(f'  {name} not downloaded yet ({p}). /models shows status.', flush=True)
            return
        if status == 'downloading':
            print(f'  {name} is still downloading (incomplete shards in {p}). Wait for it to finish.', flush=True)
            return
        self._cli.current = name
        self._cli.st['model'] = str(p)
        self._cli.st['offload'] = m['offload']
        self._cli._pipe = None
        print(f"  model = {m['label']}", flush=True)
        print(f"    path={p}  offload={m['offload']}  modalities={', '.join(m['modalities'])}", flush=True)
        if m.get('note'):
            print(f"    note: {m['note']}", flush=True)

    def apply_settings(self, defaults: dict) -> None:
        st = self._cli.st
        reload_keys = ('weight_stream', 'weight_backend', 'prefetch')
        changed = any((k in defaults and st.get(k) != defaults[k] for k in reload_keys))
        for k, v in defaults.items():
            if k in st:
                st[k] = v
        if defaults.get('gpu_index') is not None:
            st['gpu_index'] = str(defaults['gpu_index'])
        if changed:
            self._cli._pipe = None
            self._clear_resident_aux_pipe()

    def _clear_resident_aux_pipe(self, keep_manager: bool=False) -> None:
        pipe = self._resident_aux_pipe
        self._resident_aux_pipe = None
        self._resident_aux_key = None
        self._resident_aux_label = None
        if not keep_manager and self._vram_mgr is not None:
            try:
                self._vram_mgr.clear()
            except Exception:
                pass
        if pipe is not None:
            try:
                tf = getattr(pipe, 'transformer', None)
                streamer = getattr(pipe, '_zimage_streamer', None) or getattr(tf, '_zimage_streamer', None) or getattr(tf, '_glm_streamer', None) or getattr(tf, '_krea2_streamer', None) or getattr(tf, '_krea2_ds_streamer', None) or getattr(tf, '_krea2_int8_resident', None) or getattr(tf, '_boogu_streamer', None)
                close = getattr(streamer, 'close', None)
                if callable(close):
                    close()
            except Exception:
                pass
            try:
                del pipe
            except Exception:
                pass
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self, free_scail: bool=True) -> None:
        worker = getattr(self, '_glm_worker', None)
        if worker is not None:
            worker.close()
            self._glm_worker = None
        free = getattr(self._cli, 'free', None)
        if callable(free):
            free()
        else:
            self._cli._pipe = None
            try:
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._clear_resident_aux_pipe()
        if free_scail:
            try:
                import sys as _sys
                _sb = _sys.modules.get('cosmos_windows.weight_streaming.scail_backend')
                if _sb is not None and hasattr(_sb, 'unload_scail2_pipeline'):
                    _sb.unload_scail2_pipeline()
            except Exception:
                pass
        self._kill_qwen_worker()
        self._kill_moss_worker()
        if hasattr(self, '_kill_ideogram_worker'):
            self._kill_ideogram_worker()
        try:
            from .gpu_reclaim import free_aux_caches
            free_aux_caches()
        except Exception:
            pass

    def upscale_argv(self, *, image: str, out: str, scale: int, backbone: str, steps: int, ckpt_type: str, prompt: str='') -> list:
        import os
        pidpy = os.environ.get('PID_VENV_PYTHON') or self.settings.pid_python
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        script = str(Path(repo) / 'cosmos_windows' / 'pid_upscale.py')
        argv = [pidpy, script, '--image', image, '--out', out, '--scale', str(scale), '--backbone', backbone, '--ckpt-type', ckpt_type, '--steps', str(steps), '--gpu', str(self._cli.st.get('gpu_index', self.settings.gpu_index)), '--pid-repo', self.settings.pid_repo, '--pid-model-dir', self.settings.pid_model_dir]
        if prompt:
            argv += ['--prompt', prompt]
        return argv

    def _default_reasoner_assets(self, arch: str) -> tuple[str | None, str | None]:
        a = (arch or '').lower()
        if 'cosmos' not in a and a not in ('nano', 'super', 'super-i2v', 'super-t2i', 'droid'):
            return (None, None)
        is_super = 'super' in a
        gguf = (os.environ.get('STUDIO_COSMOS_SUPER_REASONER_GGUF') if is_super else None) or os.environ.get('STUDIO_COSMOS_REASONER_GGUF')
        mmproj = (os.environ.get('STUDIO_COSMOS_SUPER_REASONER_MMPROJ') if is_super else None) or os.environ.get('STUDIO_COSMOS_REASONER_MMPROJ')
        root = os.environ.get('COSMOS_MODELS_ROOT', '')
        for cand_g, cand_m in ((f'{root}/Cosmos3-Super/cosmos3-super-reasoner-q8_0.gguf', f'{root}/Cosmos3-Super/cosmos3-super-mmproj.gguf') if is_super else (None, None), (f'{root}/Cosmos3-Nano/cosmos3-nano-reasoner-q8_0.gguf', f'{root}/Cosmos3-Nano/cosmos3-nano-mmproj.gguf')):
            if not gguf and cand_g and Path(cand_g).is_file():
                gguf = cand_g
                if not mmproj and cand_m and Path(cand_m).is_file():
                    mmproj = cand_m
        gguf = gguf if gguf and Path(gguf).is_file() else None
        mmproj = mmproj if mmproj and Path(mmproj).is_file() else None
        return (gguf, mmproj)

    def _maybe_reason(self, params) -> None:
        if not getattr(params, 'reason', False):
            return
        from .llama_reasoner import reason_then_prompt
        arch = (params.model_arch or '').lower()
        reasoner_gguf = getattr(params, 'reasoner_model', None)
        sf = dict(getattr(params, 'model_support_files', None) or {})
        mmproj = sf.get('reasoner_mmproj') or None
        if not reasoner_gguf:
            reasoner_gguf, default_mmproj = self._default_reasoner_assets(arch)
            mmproj = mmproj or default_mmproj
        if not reasoner_gguf:
            return
        modality = getattr(params, 'modality', None)
        image = getattr(params, 'image_path', None)
        rkw = {}
        if getattr(params, 'reason_temperature', None) is not None:
            rkw['temperature'] = float(params.reason_temperature)
        if getattr(params, 'reason_top_p', None) is not None:
            rkw['top_p'] = float(params.reason_top_p)
        params.prompt = reason_then_prompt(params.prompt, reasoner_gguf, mmproj=mmproj, image=image, modality=modality, system=getattr(params, 'reason_system', None) or None, max_tokens=int(getattr(params, 'reason_max_tokens', 200) or 200), **rkw)

    def _h3_enhancer_assets(self) -> tuple[str | None, str | None, str | None]:
        gguf = os.environ.get('STUDIO_H3_ENHANCER_GGUF')
        mmproj = os.environ.get('STUDIO_H3_ENHANCER_MMPROJ')
        device = os.environ.get('STUDIO_H3_ENHANCER_DEVICE') or None
        gguf = gguf if gguf and Path(gguf).is_file() else None
        mmproj = mmproj if mmproj and Path(mmproj).is_file() else None
        if gguf:
            return (gguf, mmproj, device)
        default_gguf, default_mmproj = self._default_reasoner_assets('cosmos')
        return (default_gguf, mmproj or default_mmproj, device)

    def _maybe_enhance_h3(self, params, on_step=None) -> None:
        if not bool(getattr(params, 'h3_prompt_enhance', True)):
            return
        if bool(getattr(params, 'prompt_already_enhanced', False)):
            return
        from . import enhance_presets, h3_prompt_enhancer as h3, llm_launch, settings_store
        raw = str(params.prompt or '')
        if h3.has_labelled_sections(raw):
            return
        reference_block, body = h3.split_reference_block(raw)
        modality = str(getattr(params, 'modality', None) or 't2v')
        preset = enhance_presets.auto_select_for(modality, family='h3')
        if preset is None:
            raise RuntimeError(f'H3 Prompt Enhance is enabled but no enhancer preset matches modality {modality!r}. Turn off the h3_prompt_enhance toggle to send the raw prompt.')
        duration = (params.num_frames or 121) / float(params.fps or 24)
        rendered = enhance_presets.render(preset, body, video_duration=duration)
        system, user_text = (rendered['system_prompt'], rendered['user_text'])
        temperature = getattr(params, 'reason_temperature', None)
        temp = float(temperature) if temperature is not None else 0.3
        image_paths: list[str] = []
        if rendered['needs_images']:
            for candidate in [*(getattr(params, 'reference_images', None) or []), getattr(params, 'image_path', None), getattr(params, 'image2_path', None)]:
                cand = str(candidate) if candidate else ''
                if cand and Path(cand).is_file() and (cand not in image_paths):
                    image_paths.append(cand)
        video_path = None
        if rendered['needs_video']:
            source = getattr(params, 'video_path', None) or getattr(params, 'reference_video', None) or getattr(params, 'source_video', None)
            if source and Path(str(source)).is_file():
                video_path = str(source)
        has_visual = bool(image_paths) or bool(video_path)
        enhanced, route, errors = ('', '', [])
        selection = getattr(params, 'h3_enhancer_backend', None) or settings_store.load_h3_enhancer_backend()

        def _visual_data_paths() -> list[str]:
            paths = list(image_paths)
            if video_path:
                try:
                    for fp in llm_launch.reason_local_image_paths(video_path=video_path, out_dir=self.settings.output_dir):
                        if fp not in paths:
                            paths.append(fp)
                except Exception:
                    pass
            return paths

        def _run_endpoint() -> tuple[str, str]:
            endpoint = (os.environ.get('STUDIO_H3_ENHANCER_ENDPOINT') or '').strip()
            if not endpoint:
                raise RuntimeError("H3 Prompt Enhance backend is set to 'endpoint' but STUDIO_H3_ENHANCER_ENDPOINT is not configured. Point it at the OpenAI-compatible enhancer (e.g. the Mac Qwen3-VL sidecar), pick a different H3 enhancer backend, or turn off the h3_prompt_enhance toggle.")
            try:
                r = llm_launch.enhance_via_endpoint(endpoint, user_text, system=system, images=_visual_data_paths() or None, temperature=temp, max_new_tokens=1600)
                out = (r.get('answer') or '').strip()
            except Exception as exc:
                raise RuntimeError(f'H3 Prompt Enhance is enabled and STUDIO_H3_ENHANCER_ENDPOINT={endpoint} is configured, but the enhancer endpoint failed ({type(exc).__name__}: {exc}). Fix or unset STUDIO_H3_ENHANCER_ENDPOINT (a configured endpoint is not silently bypassed), or turn off the h3_prompt_enhance toggle.') from exc
            if not out:
                raise RuntimeError(f'H3 Prompt Enhance is enabled and STUDIO_H3_ENHANCER_ENDPOINT={endpoint} is configured, but the endpoint returned an empty rewrite. Fix or unset STUDIO_H3_ENHANCER_ENDPOINT, or turn off the h3_prompt_enhance toggle.')
            if not h3.has_labelled_sections(out):
                raise RuntimeError(f"H3 Prompt Enhance endpoint {endpoint} (model {r.get('model')!r}) returned a rewrite WITHOUT the H3 labelled sections — the endpoint is likely serving the wrong model. Pin STUDIO_H3_ENHANCER_MODEL to the served checkpoint id, or fix the endpoint. Rewrite head: {out[:160]!r}")
            return (out, 'h3-endpoint')

        def _run_live_preset() -> tuple[str, str]:
            if llm_launch.active_server() is None:
                raise RuntimeError("H3 Prompt Enhance backend is set to 'live-preset' but no LLM preset is active in Connect. Start an LLM preset, pick a different H3 enhancer backend, or turn off the h3_prompt_enhance toggle.")
            r = llm_launch.enhance_via_server(user_text, recaption=True, style='caption', max_new_tokens=1600, temperature=temp, system=system, images=_visual_data_paths() or None if has_visual else None)
            return ((r.get('answer') or '').strip(), 'live-preset')

        def _resolve_local_mode() -> str:
            local_mode = (os.environ.get('STUDIO_H3_ENHANCER_LOCAL_MODE') or 'quant').strip().lower()
            if local_mode not in ('quant', 'local'):
                raise RuntimeError(f"STUDIO_H3_ENHANCER_LOCAL_MODE={local_mode!r} is invalid; allowed values are 'quant' (4-bit disk-cached resident reasoner, the default) or 'local' (full-precision ~17.5 GB resident reasoner). Fix or unset the env var.")
            return local_mode

        def _run_local_reasoner(local_mode: str) -> tuple[str, str]:
            paths: list[str] = []
            if has_visual:
                paths = llm_launch.reason_local_image_paths(image_path=image_paths[0] if image_paths else None, video_path=video_path, out_dir=self.settings.output_dir)
                for extra in image_paths[1:]:
                    if extra not in paths:
                        paths.append(extra)
            res = llm_launch.run_local_reason(local_mode, user_text, image_paths=paths or None, system=system, temperature=temp, max_new_tokens=1600)
            return ((res.get('answer') or '').strip(), f'local-reasoner:{local_mode}')

        def _run_gguf() -> tuple[str, str]:
            reasoner_gguf, mmproj, device = self._h3_enhancer_assets()
            if not reasoner_gguf:
                raise RuntimeError('no STUDIO_H3_ENHANCER_GGUF / Cosmos reasoner configured')
            image = image_paths[0] if image_paths else None
            from .llama_reasoner import LlamaReasoner
            out = LlamaReasoner(reasoner_gguf, mmproj=mmproj, device=device).expand_prompt(user_text, system=system, image=image, max_tokens=1600, temperature=temp, collapse=False).strip()
            return (out, 'gguf-fallback')
        if selection == 'auto':
            if (os.environ.get('STUDIO_H3_ENHANCER_ENDPOINT') or '').strip():
                enhanced, route = _run_endpoint()
            if not enhanced and (not has_visual) and (llm_launch.active_server() is not None):
                try:
                    enhanced, route = _run_live_preset()
                except Exception as exc:
                    errors.append(f'live preset: {exc}')
            if not enhanced and has_visual:
                local_mode = _resolve_local_mode()
                try:
                    enhanced, route = _run_local_reasoner(local_mode)
                except Exception as exc:
                    errors.append(f'local vision reasoner ({local_mode}): {exc}')
            if not enhanced:
                try:
                    enhanced, route = _run_gguf()
                except Exception as exc:
                    errors.append(f'fallback gguf: {exc}')
            if not enhanced:
                raise RuntimeError('H3 Prompt Enhance is enabled but no enhancer backend produced a rewrite [' + '; '.join(errors) + ']. Start an LLM preset in Connect, configure the enhancer model (STUDIO_H3_ENHANCER_GGUF, +_DEVICE if the default device crashes), or turn off the h3_prompt_enhance toggle.')
        elif selection == 'endpoint':
            enhanced, route = _run_endpoint()
        elif selection in ('live-preset', 'local-reasoner', 'gguf'):
            if selection == 'live-preset':
                enhanced, route = _run_live_preset()
            elif selection == 'local-reasoner':
                enhanced, route = _run_local_reasoner(_resolve_local_mode())
            else:
                enhanced, route = _run_gguf()
            if not enhanced or not h3.has_labelled_sections(enhanced):
                raise RuntimeError(f'H3 Prompt Enhance backend {selection!r} did not produce a labelled H3 rewrite (got {enhanced[:160]!r}). The pinned backend cannot enhance this prompt — pick a different H3 enhancer backend in the Generate/Settings screen, or turn off the h3_prompt_enhance toggle.')
        else:
            raise RuntimeError(f'H3 Prompt Enhance backend {selection!r} is not recognised; allowed values are auto, endpoint, live-preset, local-reasoner, gguf. Fix it in the Generate/Settings screen.')
        params.prompt = h3.join_reference_block(reference_block, enhanced)
        if on_step:
            try:
                on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_prompt_enhance', 'text': f'H3 Prompt Enhance applied ({route})', 'original_prompt': raw, 'enhanced_prompt': params.prompt, 'backend': selection, 'backend_route': route})
            except TypeError:
                pass

    def generate(self, params, on_step=None, should_cancel=None) -> GenResult:
        self._resolve_registered_model(params)
        from .generation_contract import validate_request
        validate_request(params, sys.platform, getattr(self.settings, 'cosmos_windows_path', None))
        from . import audio, settings_store
        arch = str(getattr(params, 'model_arch', '') or '')
        worker = getattr(self, '_glm_worker', None)
        if worker is not None and (not arch.startswith('glm')):
            worker.close()
            self._glm_worker = None
        gpu = settings_store.gpu_for(str(getattr(params, 'model', '') or ''), arch, arch.split('-')[0])
        if gpu in (None, 'cpu'):
            gpu = str(getattr(self.settings, 'gpu_index', '0'))
        with audio.gpu_workload(gpu):
            return self._generate_with_gpu_slot(params, on_step, should_cancel)

    def _generate_with_gpu_slot(self, params, on_step=None, should_cancel=None) -> GenResult:
        try:
            from . import triposplat_backend
            triposplat_backend.free_gpu_aux()
        except Exception:
            pass
        if os.environ.get('STUDIO_ATTENTION_PREFER_CUDNN', '1').strip().lower() not in ('0', 'false', 'no', 'off'):
            try:
                from cosmos_windows.attention_memeff import enable_memeff_attention
                enable_memeff_attention(allow_math_fallback=True)
            except Exception:
                pass
        from .cosmos_conditioning import validate as validate_cosmos_conditioning
        validate_cosmos_conditioning(params)
        self._maybe_reason(params)
        arch = (params.model_arch or '').lower()
        self._apply_gpu_assignment(params, arch)
        if not arch.startswith(('qwen-image-2.1', 'qwenimage2.1')):
            self._kill_qwen_worker()
        if not arch.startswith('moss-image'):
            self._kill_moss_worker()
        if not arch.startswith('ideogram') and getattr(self, '_ideogram_worker', None) is not None:
            self._kill_ideogram_worker()
        if params.control_image or params.control_kind or (params.control_guides or []):
            from .generation_contract import family
            if family(arch) not in _CONTROL_CAPABLE_ARCHES:
                raise RuntimeError(f"ControlNet/structural conditioning is not supported for '{arch or params.model_arch}' in this build — the control image/guide would be silently ignored. Remove the control guide, or use a model+pipeline that consumes it.")
        if bool(getattr(params, 'weight_stream', True)):
            from .stream_capability import DIRECT, NATIVE, OFFLOAD, UNSUPPORTED, stream_capability
            _cap = stream_capability(arch)
            _allow_fallback = os.environ.get('STUDIO_ALLOW_STREAMING_FALLBACK', '0') in ('1', 'true', 'True')
            if _cap not in (NATIVE, DIRECT) and (not _allow_fallback):
                detail = 'CPU-offload/staged loading is not streaming; it is the slow fallback path.' if _cap == OFFLOAD else 'This model has no streaming path in this build.'
                raise RuntimeError(f"'{arch or 'cosmos'}' cannot run with weight_stream=true because it does not have a native streaming implementation yet (capability={_cap}). {detail} Select a native streaming model, turn weight streaming off for resident mode, or set STUDIO_ALLOW_STREAMING_FALLBACK=1 only for diagnostic fallback runs.")
            if _cap == UNSUPPORTED:
                print(f"[stream] '{arch or 'cosmos'}' has no weight-streaming path in this build — running resident (streaming is declared-unsupported for this model).", flush=True)
            elif _cap == OFFLOAD:
                print(f"[stream] '{arch or 'cosmos'}' has no native streamer yet; using bounded-memory CPU-offload/staged fallback (migration debt).", flush=True)
            elif _cap == DIRECT:
                print(f"[stream] '{arch or 'cosmos'}' direct-resident path active (no CPU-offload/staged fallback).", flush=True)
            else:
                from .attention_contract import require_flash_attention_for_arch
                _attn = require_flash_attention_for_arch(arch)
                if _attn.get('backend') not in ('not-required', 'none'):
                    print(f"[attention] '{arch or 'cosmos'}' FlashAttention backend: {_attn.get('backend')} ({_attn.get('detail')})", flush=True)
                print(f"[stream] '{arch or 'cosmos'}' native weight streaming active.", flush=True)
        if arch.startswith('bernini'):
            return self._generate_bernini(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('scail'):
            return self._generate_scail(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('domainshuttle') or arch.startswith('domain-shuttle'):
            return self._generate_domainshuttle(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('minimax-h3') or arch == 'h3':
            return self._generate_minimax_h3(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('magi'):
            return self._generate_magi2(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('wan'):
            return self._generate_wan(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('hunyuan'):
            return self._generate_hunyuan(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('ideogram'):
            return self._generate_ideogram(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('joyai'):
            return self._generate_joyai(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('ltx'):
            return self._generate_ltx(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('qwen-image-2.1') or arch.startswith('qwenimage2.1'):
            return self._generate_qwen_image21(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('moss-image'):
            return self._generate_moss_image(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('z-image') or arch.startswith('zimage'):
            return self._generate_zimage(params, on_step=on_step, should_cancel=should_cancel)
        if arch == 'ultratex':
            return self._generate_ultratex(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('flux2') or arch.startswith('flux-2') or arch.startswith('klein'):
            return self._generate_flux2klein(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('anima'):
            return self._generate_anima(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('sdxl') or arch.startswith('sd-xl') or arch.startswith('sd_xl') or (arch in ('illustrious', 'noobai', 'pony', 'animagine')):
            return self._generate_sdxl(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('glm') or str(getattr(params, 'model', '')) in ('glm-image', 'glm-image-decoder'):
            return self._generate_glm(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('krea2-native'):
            return self._generate_krea2_native(params, on_step=on_step, should_cancel=should_cancel)
        if arch in {'krea2-raw', 'krea2-turbo'} and params.model and Path(params.model).is_file():
            from .krea_hf_stream import preflight_single_file
            preflight_single_file(params.model)
            files = {**dict(params.model_support_files or {}), 'dit': str(params.model)}
            bound = params.model_copy(update={'model_support_files': files})
            return self._generate_krea2_native(bound, on_step=on_step, should_cancel=should_cancel)
        if arch in {'krea2-raw', 'krea2-turbo'} and params.model and Path(params.model).is_dir() and (sys.platform == 'win32'):
            from .models import resolve_load_mode
            if not resolve_load_mode(params)[1]:
                singles = sorted(Path(params.model).glob('*.safetensors'))
                if len(singles) != 1:
                    raise RuntimeError(f'KREA_STREAMING_NEEDS_NATIVE_FILE: streaming Krea needs exactly one native single-file checkpoint in {params.model} (found {len(singles)}); register that file as the model, or use a resident mode')
                from .krea_hf_stream import preflight_single_file
                preflight_single_file(str(singles[0]))
                files = {**dict(params.model_support_files or {}), 'dit': str(singles[0])}
                bound = params.model_copy(update={'model': str(singles[0]), 'model_support_files': files})
                return self._generate_krea2_native(bound, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('boogu') or arch.startswith('krea'):
            return self._generate_boogu_krea(params, on_step=on_step, should_cancel=should_cancel)
        if arch.startswith('triposplat') or arch.startswith('tripo') or arch in ('3dgs', 'splat', 'image-to-3d', '3d'):
            return self._generate_triposplat(params, on_step=on_step, should_cancel=should_cancel)
        if params.control_image or params.control_kind or (params.control_guides or []):
            raise RuntimeError(f"ControlNet conditioning is not supported by the Cosmos pipeline (model {params.model_arch or params.model or 'cosmos'!r}). The structural map cannot be applied here — remove it, or generate with a ControlNet-capable backend.")
        requested_builtin = str(getattr(params, 'model', '') or '')
        if requested_builtin in getattr(self._cc, 'MODELS', {}) and (requested_builtin != getattr(self._cli, 'current', None) or not self._selected_builtin_path_matches(requested_builtin)):
            self.select_model(requested_builtin)
        st = self._cli.st
        st.update(width=params.width, height=params.height, seed=params.seed, negative=params.negative, num_frames=params.num_frames, fps=params.fps, sampler=params.sampler, sigmas=params.sigmas, cfg_batch=params.cfg_batch, weight_stream=params.weight_stream, out=str(output_dir_for_params(params)), denoise=params.denoise, mask_path=params.mask_path, loras=[l.model_dump() for l in params.loras], control_image=params.control_image, control_kind=params.control_kind, control_weight=params.control_weight, control_guides=[c.model_dump() for c in params.control_guides or []], composition_guide=params.composition_guide.model_dump() if params.composition_guide else None, composition_guide_path=params.composition_guide_path, audio_path=params.audio_path, audio_mode=params.audio_mode, enable_sound=params.audio_mode in ('generate', 'condition'), model_arch=params.model_arch, model_support_files=params.model_support_files, image2_path=params.image2_path, keyframes=[k.model_dump() for k in params.keyframes or []])
        if params.steps is not None:
            st['steps'] = params.steps
            self._cli._user_set.add('steps')
        if params.guidance is not None:
            st['guidance'] = params.guidance
            self._cli._user_set.add('guidance')
        t0 = time.time()
        gen_kw = dict(on_step=on_step, should_cancel=should_cancel)
        if params.modality == 'v2v' or getattr(params, 'video_path', None):
            gen_kw['video_path'] = getattr(params, 'video_path', None)
            gen_kw['num_conditional_frames'] = getattr(params, 'num_conditional_frames', None)
        result_path = self._cli.generate(params.modality, params.prompt, params.image_path, **gen_kw)
        if result_path is None:
            raise RuntimeError(f'generate({params.modality!r}) produced no output — unsupported modality for the current model, or a generation error (see daemon log)')
        seconds = round(time.time() - t0, 2)
        kind = 'video' if str(result_path).lower().endswith(('.mp4', '.webm', '.gif')) else 'image'
        if params.audio_path and params.audio_mode == 'mux':
            if kind != 'video':
                raise RuntimeError('audio_mode=mux requires a generated video output')
            result_path = mux_audio(result_path, params.audio_path)
            kind = 'video'
        return GenResult(path=str(result_path), seconds=seconds, modality=params.modality, width=params.width, height=params.height, kind=kind, seed=params.seed, engine=self.name)

    def _resolve_registered_model(self, params) -> None:
        model = getattr(params, 'model', None)
        if not model:
            return
        try:
            from . import model_overrides
            override = model_overrides.get(str(model)) or {}
        except Exception:
            override = {}
        override_path = str(override.get('path') or '').strip()
        override_support = {str(k): str(v) for k, v in (override.get('support_files') or {}).items() if v}
        if str(model) == 'hunyuan-image-3':
            params.model = override_path or self.settings.hunyuan_model
            if not getattr(params, 'model_arch', None):
                params.model_arch = 'hunyuan-3'
            if override_support:
                merged = dict(override_support)
                merged.update(getattr(params, 'model_support_files', None) or {})
                params.model_support_files = merged
            return
        if str(model) == 'ideogram-4':
            try:
                from . import ideogram_registry
                repo_dir, quant, _desc, present = ideogram_registry.resolve(self.settings)
                if override_path:
                    params.model = override_path
                elif present:
                    params.model = str(repo_dir)
                if not override_path and present and (quant != 'auto'):
                    params.ideo_quant = quant
            except Exception:
                if override_path:
                    params.model = override_path
                pass
            if not getattr(params, 'model_arch', None):
                params.model_arch = 'ideogram-4'
            if override_support:
                merged = dict(override_support)
                merged.update(getattr(params, 'model_support_files', None) or {})
                params.model_support_files = merged
            return
        if str(model) == 'flux2-klein':
            if override_path:
                try:
                    if Path(override_path).is_dir():
                        override_support.setdefault('components_repo', override_path)
                        params.model = ''
                    else:
                        params.model = override_path
                except OSError:
                    params.model = override_path
            if not getattr(params, 'model_arch', None):
                params.model_arch = 'flux2-klein'
            if override_support:
                merged = dict(override_support)
                merged.update(getattr(params, 'model_support_files', None) or {})
                params.model_support_files = merged
            return
        if str(model) == 'qwen-image-2.1':
            params.model = override_path or getattr(self.settings, 'qwen_image21_model', params.model)
            if not getattr(params, 'model_arch', None):
                params.model_arch = 'qwen-image-2.1'
            if override_support:
                merged = dict(override_support)
                merged.update(getattr(params, 'model_support_files', None) or {})
                params.model_support_files = merged
            return
        try:
            p = Path(model)
            if p.exists():
                return
        except OSError:
            pass
        try:
            from . import custom_models
            info = custom_models.get(str(model))
        except Exception:
            info = None
        if not info:
            if override_path and str(model) not in getattr(self._cc, 'MODELS', {}):
                params.model = override_path
                if override_support:
                    merged = dict(override_support)
                    merged.update(getattr(params, 'model_support_files', None) or {})
                    params.model_support_files = merged
            elif not getattr(params, 'model_arch', None) and (not _cosmos_model_request(str(model), getattr(self._cc, 'MODELS', {}))):
                raise RuntimeError(f'Model {str(model)!r} is not installed or registered on this Studio: it is not a built-in Cosmos model, a checkpoint path or a custom model entry (studio_custom_models.json).')
            return
        params.model = override_path or info.get('path') or params.model
        if info.get('_runtime_migrated') or not getattr(params, 'model_arch', None):
            params.model_arch = info.get('arch')
        merged = dict(info.get('support_files') or {})
        merged.update(override_support)
        merged.update(getattr(params, 'model_support_files', None) or {})
        params.model_support_files = merged

    def _run_guarded_subprocess(self, argv: list[str], kind: str, *, on_step=None, should_cancel=None, env_extra: Optional[dict]=None, isolate_python: bool=False) -> tuple[int, list[str]]:
        import threading
        from .jobs import kill_process_tree, managed_popen
        from .progress import parse_line
        if should_cancel and should_cancel():
            return (-1, [])
        launch_options = {'isolate_python': True} if isolate_python else {}
        proc = managed_popen(argv, env_extra=env_extra, **launch_options)
        lines: list[str] = []

        def watch_cancel() -> None:
            while proc.poll() is None:
                if should_cancel and should_cancel():
                    kill_process_tree(proc)
                    return
                time.sleep(0.25)
        threading.Thread(target=watch_cancel, daemon=True).start()
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip('\n')
            lines.append(line)
            ev = parse_line(line, kind)
            if ev and ev.get('type') == 'gen_progress' and on_step:
                try:
                    on_step(int(ev.get('step') or 0), int(ev.get('total') or 0), None, ev)
                except TypeError:
                    on_step(int(ev.get('step') or 0), int(ev.get('total') or 0), None)
            elif ev and ev.get('type') in ('stage_timing', 'gen_status') and on_step:
                try:
                    on_step(0, 0, None, ev)
                except TypeError:
                    pass
        return (proc.wait(), lines)

    def _image_out_path(self, prefix: str, prompt: str, seed: int, params=None) -> Path:
        out_dir = output_dir_for_params(params)
        stem = re.sub('[^A-Za-z0-9]+', '_', (prompt or prefix).strip())[:40].strip('_') or prefix
        return out_dir / f'{prefix}_{stem}_{uuid.uuid4().hex}_{seed}.png'

    @staticmethod
    def _first_image(result):
        images = getattr(result, 'images', result)
        if isinstance(images, dict):
            for key in ('images', 'image', 'sample', 'output'):
                if key in images:
                    return RealCosmosBackend._first_image(images[key])
        if isinstance(images, (list, tuple)):
            if not images:
                raise RuntimeError('pipeline returned an empty image list')
            return RealCosmosBackend._first_image(images[0])
        if hasattr(images, 'save'):
            return images
        raise RuntimeError(f'pipeline returned unsupported image type {type(images).__name__}')

    @staticmethod
    def _lora_cache_key(loras) -> tuple:
        out = []
        for lora in loras or []:
            out.append((str(getattr(lora, 'path', '') or ''), float(getattr(lora, 'weight', 1.0) or 1.0), str(getattr(lora, 'name', '') or '')))
        return tuple(out)

    def _cleanup_torch(self) -> None:
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _stream_pipeline_cache_enabled() -> bool:
        return os.environ.get('STUDIO_STREAM_PIPELINE_CACHE', '1').strip().lower() not in ('0', 'false', 'off', 'no', 'n')

    def _generate_hunyuan(self, params, on_step=None, should_cancel=None) -> GenResult:
        self.unload()
        self._cleanup_torch()
        from . import safety
        ok, free = safety.enough_ram(8.0)
        if not ok:
            raise RuntimeError(f'Not enough free RAM to load HunyuanImage-3 (80B): {free:.1f} GB free, need >= 8 GB.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        script = Path(repo) / 'cosmos_windows' / 'hunyuan_gen.py'
        if not script.is_file():
            raise RuntimeError('Hunyuan helper is missing; set STUDIO_COSMOS_WINDOWS_PATH to the RTX repo that contains cosmos_windows/hunyuan_gen.py.')
        model_dir = str(params.model or self.settings.hunyuan_model)
        if not Path(model_dir).exists():
            model_dir = self.settings.hunyuan_model
        out = self._image_out_path('hunyuan', params.prompt, params.seed, params)
        argv = [self.settings.hunyuan_python, str(script), '--prompt', params.prompt, '--out', str(out), '--steps', str(params.steps or 50), '--size', f'{params.width}x{params.height}', '--seed', str(params.seed), '--gpu', self.settings.gpu_index, '--model-dir', model_dir, '--ar-mode', 'skip']
        if getattr(params, 'image_path', None):
            argv += ['--image', str(params.image_path)]
            argv += ['--denoise', str(params.denoise if params.denoise is not None else 1.0)]
        if getattr(params, 'hunyuan_native_edit', False):
            if not params.image_path:
                raise ValueError('Native Hunyuan edit requires a source image')
            argv += ['--native-edit']
            for reference in params.reference_images:
                argv += ['--reference-image', str(reference)]
        if getattr(params, 'mask_path', None):
            argv += ['--mask', str(params.mask_path)]
        t0 = time.time()
        rc, lines = self._run_guarded_subprocess(safety.guard_argv(argv, min_gb=3.0, repo=repo, python=self.settings.hunyuan_python), 'hunyuan_gen', on_step=on_step, should_cancel=should_cancel)
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        if rc != 0:
            tail = failure_tail(lines, limit=1000)
            raise RuntimeError(f'Hunyuan generation failed (exit {rc}): {tail}')
        if not out.is_file():
            for line in reversed(lines):
                m = re.search('RESULT DONE\\s+(.+\\.(?:png|jpg|jpeg|webp))', line, re.I)
                if m and Path(m.group(1).strip()).is_file():
                    out = Path(m.group(1).strip())
                    break
        if not out.is_file():
            raise RuntimeError('Hunyuan generation completed without an output image')
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='t2i', width=params.width, height=params.height, kind='image', seed=params.seed, engine=self.name)

    def _kill_qwen_worker(self):
        worker = getattr(self, '_qwen_worker', None)
        self._qwen_worker = None
        self._qwen_worker_key = None
        if worker is not None:
            worker.close()

    def _generate_qwen_image21(self, params, on_step=None, should_cancel=None) -> GenResult:
        from . import qwen_image21
        python = str(getattr(self.settings, 'qwen_image21_python', '') or '')
        model_dir = str(params.model or getattr(self.settings, 'qwen_image21_model', '') or '')
        if not Path(python).is_file():
            raise RuntimeError(f'Qwen-Image-2.1 interpreter not found: {python!r}. Set STUDIO_QWEN_IMAGE21_PYTHON.')
        from .qwen_comfy_load import is_single_file_checkpoint
        model_path = Path(model_dir)
        if not is_single_file_checkpoint(model_path) and (not (model_path / 'model_index.json').is_file()):
            raise RuntimeError(f'Qwen-Image-2.1 model directory not found or incomplete: {model_dir!r}. Set STUDIO_QWEN_IMAGE21_MODEL to a diffusers repo, or a ComfyUI single-file .safetensors (its text encoder/VAE come from STUDIO_QWEN_IMAGE21_BASE).')
        out = self._image_out_path('qwen-image-2.1', params.prompt, params.seed, params)
        request_path = out.with_suffix('.request.json')
        regions = [region.model_dump() if hasattr(region, 'model_dump') else dict(region) for region in getattr(params, 'qwen_regions', None) or []]
        request = {'prompt': params.prompt, 'negative': params.negative, 'width': int(params.width or 1024), 'height': int(params.height or 1024), 'steps': int(params.steps or 40), 'true_cfg_scale': float(params.guidance if params.guidance is not None else 1.0), 'seed': int(params.seed), 'image_path': str(params.image_path or ''), 'reference_images': [str(path) for path in params.reference_images or []], 'source_width': getattr(params, 'qwen_source_width', None), 'source_height': getattr(params, 'qwen_source_height', None), 'regions': regions, 'output_rgba': bool(getattr(params, 'qwen_output_rgba', False)), 'use_kv_cache': bool(getattr(params, 'qwen_use_kv_cache', True)), 'request_encoder_staging': bool(getattr(params, 'qwen_request_encoder_staging', False)), 'output_path': str(out)}
        request['two_pass'] = params.two_pass.model_dump() if params.two_pass else None
        request['loras'] = qwen_image21.validate_loras([{'path': l.path, 'weight': l.weight} for l in params.loras or []])
        request['sampler'], request['sigmas'] = qwen_image21.scheduler_choice(getattr(params, 'sampler', None), getattr(params, 'sigmas', None))
        qwen_image21.validate_request(request)
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(request, indent=2), encoding='utf-8')
        from .models import resolve_load_mode
        from .qwen_worker_client import QwenWorker
        _, resident, _ = resolve_load_mode(params)
        env = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(self.settings.gpu_index)}
        base_repo = str(getattr(self.settings, 'qwen_image21_base', '') or '')
        if base_repo:
            env['STUDIO_QWEN_IMAGE21_BASE'] = base_repo
        precision = params.qwen_weight_precision
        key = (python, model_dir, resident, precision, request['request_encoder_staging'], tuple(sorted(env.items())))
        worker = getattr(self, '_qwen_worker', None)
        if not worker or not worker.alive() or getattr(self, '_qwen_worker_key', None) != key:
            self.unload()
            self._cleanup_torch()
            argv = [python, '-u', '-m', 'studio.qwen_image21', '--serve', '--model-dir', model_dir, '--precision', precision]
            if not resident:
                argv.append('--stream-weights')
            worker = QwenWorker(argv, env, log_path=out.with_suffix('.worker.jsonl'))
            self._qwen_worker, self._qwen_worker_key = (worker, key)
        t0 = time.time()
        try:
            worker_result = worker.generate(request, on_step=on_step, should_cancel=should_cancel) or {}
        except BaseException:
            self._kill_qwen_worker()
            raise
        finally:
            if not resident:
                self._kill_qwen_worker()
        if not out.is_file():
            raise RuntimeError('Qwen-Image-2.1 generation completed without an output image')
        metrics = {}
        metrics_path = out.with_suffix(out.suffix + '.metrics.json')
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                metrics = {}
        return GenResult(path=str(out), seconds=float(metrics.get('seconds') or round(time.time() - t0, 2)), modality='i2i' if request['image_path'] or request['reference_images'] else 't2i', width=int(metrics.get('width') or params.width), height=int(metrics.get('height') or params.height), kind='image', seed=params.seed, engine=self.name, timings={'wall_seconds': metrics.get('seconds'), 'gpu': metrics.get('gpu'), 'condition_images': metrics.get('condition_images'), 'mode': metrics.get('mode'), 'worker_log': worker_result.get('worker_log')})

    def _kill_moss_worker(self):
        worker = getattr(self, '_moss_worker', None)
        if worker is not None:
            self._moss_worker_key = None
            worker.close()
            if worker.alive():
                raise RuntimeError('MOSS worker did not exit; refusing an overlapping model load')
        self._moss_worker = self._moss_worker_key = None

    def _generate_moss_image(self, params, on_step=None, should_cancel=None) -> GenResult:
        from . import moss_image
        from .models import resolve_load_mode
        from .qwen_worker_client import QwenWorker
        python = str((params.model_support_files or {}).get('python') or os.environ.get('STUDIO_MOSS_IMAGE_PYTHON') or self.settings.qwen_image21_python)
        if not Path(python).is_file():
            raise RuntimeError('MOSS-Image interpreter missing; configure STUDIO_MOSS_IMAGE_PYTHON with diffusers>=0.38 and transformers>=5.3')
        root = Path(params.model)
        missing = moss_image.missing_components(root)
        if missing:
            raise RuntimeError('MOSS-Image local checkpoint incomplete: ' + ', '.join(missing))
        out = self._image_out_path('moss-image', params.prompt, params.seed, params)
        request = moss_image.prepare_request(params, out)
        out.with_suffix('.request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
        mode, resident, _ = resolve_load_mode(params)
        gpu = self._cli.st.get('gpu_index', self.settings.gpu_index) if self._cli is not None else self.settings.gpu_index
        cache_root = Path((params.model_support_files or {}).get('runtime_cache') or Path(getattr(self.settings, 'output_dir', out.parent / 'outputs')).parent / 'model_caches' / 'moss-image')
        cache_root.mkdir(parents=True, exist_ok=True)
        env = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(gpu), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        for key, relative in {'HF_HOME': 'hf', 'STUDIO_HF_CACHE': 'hf', 'HF_HUB_CACHE': 'hf/hub', 'HUGGINGFACE_HUB_CACHE': 'hf/hub', 'TRANSFORMERS_CACHE': 'hf/hub', 'DIFFUSERS_CACHE': 'hf/hub', 'KERNELS_CACHE': 'kernels', 'HF_KERNELS_CACHE': 'kernels', 'TORCH_HOME': 'torch', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor', 'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'XDG_CACHE_HOME': 'xdg', 'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp'}.items():
            directory = cache_root / relative
            directory.mkdir(parents=True, exist_ok=True)
            env[key] = str(directory)
        argv = [python, '-u', '-m', 'studio.moss_image', '--model-dir', str(root), '--precision', params.qwen_weight_precision, '--pack-dir', str(cache_root / 'int8')]
        if not resident:
            argv.append('--stream-weights')
        cached = mode == 'resident_cached'
        if cached:
            argv.append('--cache-pipeline')
        if mode == 'streaming_nonlazy':
            argv.append('--eager-components')
        key = (tuple(argv), tuple(sorted(env.items())))
        if cached:
            key += ((request['prompt'], request['negative'], request['guidance'] != 1),)
        worker = getattr(self, '_moss_worker', None)
        if not worker or not worker.alive() or getattr(self, '_moss_worker_key', None) != key:
            self.unload()
            self._cleanup_torch()
            worker = QwenWorker(argv, env, log_path=out.with_suffix('.worker.jsonl'), completion_prefix='MOSS_JOB_DONE ')
            self._moss_worker, self._moss_worker_key = (worker, key)
        try:
            result = worker.generate(request, on_step=on_step, should_cancel=should_cancel)
        except BaseException:
            self._kill_moss_worker()
            raise
        finally:
            if not cached:
                self._kill_moss_worker()
        if not out.is_file():
            raise RuntimeError('MOSS-Image worker finished without an output image')
        return GenResult(path=str(out), seconds=result['seconds'], modality=params.modality, kind='image', width=result['width'], height=result['height'], seed=params.seed, engine=self.name, timings={**result, 'worker_log': result.get('worker_log')})

    def _generate_minimax_h3(self, params, on_step=None, should_cancel=None) -> GenResult:
        if params.h3_seamless or params.h3_extend_from:
            from .h3_live import require_h3_extension_runtime
            require_h3_extension_runtime(getattr(self.settings, 'cosmos_windows_path', '') or '')
        self.unload()
        self._cleanup_torch()
        from . import safety
        repo = Path(str(getattr(self.settings, 'cosmos_windows_path', '') or ''))
        runtime = repo / 'cosmos_windows' / 'weight_streaming' / 'minimax_h3_backend.py'
        if not runtime.is_file():
            raise RuntimeError('MiniMax H3 runtime is missing; set STUDIO_COSMOS_WINDOWS_PATH to the RTX repo containing cosmos_windows/weight_streaming/minimax_h3_backend.py.')
        modality = str(params.modality or 't2v')
        upscale = getattr(params, 'h3_upscale', None)
        if upscale is not None:
            from .h3_upscale import validate_upscale
            validate_upscale(upscale)
        from . import h3_image_mode as h3img
        image_mode = modality in h3img.IMAGE_MODALITIES
        audio_mode = modality in h3img.AUDIO_MODALITIES
        if upscale is not None:
            image_mode = audio_mode = False
        if image_mode or audio_mode:
            params.prompt = h3img.build_prompt_for(modality, params.prompt or '')
            params.prompt_already_enhanced = True
            frames = int(params.num_frames or 0)
            if frames < 5 or frames > 49:
                params.num_frames = h3img.DEFAULT_IMAGE_FRAMES
            params.fps = int(params.fps or 0) or h3img.DEFAULT_IMAGE_FPS
        if audio_mode:
            params.width = params.height = h3img.DEFAULT_AUDIO_SIZE
            if not params.steps or int(params.steps) < h3img.DEFAULT_AUDIO_STEPS:
                params.steps = h3img.DEFAULT_AUDIO_STEPS
        if (image_mode or audio_mode) and on_step:
            on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_derived_mode', 'text': f"{modality}: {params.num_frames}-frame clip at {params.width}x{params.height}, {params.steps or 'default'} steps; deliverable = {('AUDIO track' if audio_mode else 'LAST frame')}"})
        from . import h3_single_file
        official_mode = {'t2v': 't2v', 'i2v': 'i2v', 'l2v': 'l2v', 'fflf': 'flf2v', 'r2v': 'ref2v', 'ref2v': 'ref2v', 'v2v': 'v2v', 'edit': 'v2v', **h3img.IMAGE_MODALITIES, **h3img.AUDIO_MODALITIES}.get(modality)
        if upscale is not None and upscale.mode == 'video':
            official_mode = 't2v'
        if not official_mode:
            raise RuntimeError(f'MiniMax H3 v1 does not support modality {modality!r}')
        h3_start_frame = str(params.image_path or '').strip() if upscale is None and modality in ('t2v', 'r2v', 'ref2v') else ''
        if h3_start_frame and official_mode == 't2v':
            official_mode = 'i2v'
        from .h3_model_profiles import h3_checkpoint_worker_env, h3_model_profile, profile_adapters
        h3_layout_env = dict(h3_checkpoint_worker_env(params.model))
        h3_profile = h3_model_profile(params.model) if upscale is None else None
        h3_profile_loras = None
        if h3_profile is not None:
            _adapters, _removed = (list(params.loras or []),[]) if (params.model_support_files or {}).get('_stack_explicit_loras_only') else profile_adapters(h3_profile, list(params.loras or []))
            if _adapters and _adapters[0] is None:
                if h3_profile.lora is None:
                    raise RuntimeError(f"{Path(str(params.model)).name} is a Viggle-Animate checkpoint and needs its own DMD adapter (viggle_animate_dmd_lora_r64.safetensors); it was not found under {', '.join(h3_profile.searched)}. The base H3 turbo adapter is not a substitute.")
                _adapters[0] = SimpleNamespace(path=str(h3_profile.lora), weight=1.0)
            h3_profile_loras = _adapters
            params.steps = h3_profile.steps
            h3_layout_env['STUDIO_H3_SIGMA_SHIFT_VIDEO'] = f'{h3_profile.sigma_shift:g}'
            if h3_profile.text_embed is not None:
                h3_layout_env['STUDIO_H3_FIXED_TEXT_EMBED'] = str(h3_profile.text_embed)
            if on_step:
                on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_profile', 'text': f'{h3_profile.name} recipe: its DMD adapter + {max(0, len(_adapters) - 1)} of your LoRA(s), {h3_profile.steps} steps, shift {h3_profile.sigma_shift:g}' + (f"; removed base step adapter(s) {', '.join(_removed)}" if _removed else '')})
        registered = Path(str(params.model or ''))
        if h3_single_file.looks_like_single_file_checkpoint(registered):
            composed = h3_single_file.compose_single_file_model_dir(registered)
            if on_step:
                on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_single_file', 'text': f'composed {registered.name} into a model dir at {composed} (transformer = the finetune, components linked from the base install)'})
            registered = composed
        components = {'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}
        selected = components.get(registered.name.casefold())
        if (registered / 'transformer').is_dir():
            component, model_dir = (registered.name, registered)
        elif selected:
            component, model_dir = (selected, registered)
        else:
            wants_refs = any((str(r).strip() for r in list(params.reference_images or []) + list(params.reference_videos or []) + list(params.reference_audios or [])))
            component = 'Ref2VA' if official_mode in ('ref2v', 'v2v') or wants_refs else 'FL2VA'
            model_dir = registered / component
        if not model_dir.is_dir():
            raise RuntimeError(f'MiniMax H3 {component} checkpoint is not installed at {model_dir} (registered path: {registered}). Download it, or select the other H3 model -- FL2VA and Ref2VA can both serve every modality.')
        transformer_index = model_dir / 'transformer' / 'model.safetensors.index.json'
        partial_weight = next(model_dir.rglob('*.part'), None)
        if (upscale is None or upscale.mode != 'video') and (not transformer_index.is_file() or partial_weight is not None):
            raise RuntimeError(f'MiniMax H3 {component} checkpoint is incomplete at {model_dir}; finish its transformer manifest and partial weight files before generation.')
        if upscale is not None:
            from .h3_upscale_runner import preflight
            from .h3_upscale import turbo_path
            checked_loras = [(Path(lora.path), lora.weight) for lora in params.loras or []]
            if upscale.mode == 'two_pass' and upscale.turbo:
                checked_loras.append((turbo_path(component), 1.0))
            preflight(model_dir, upscale, checked_loras)
        if upscale is None or upscale.mode != 'video':
            self._maybe_enhance_h3(params, on_step=on_step)
        out_dir = output_dir_for_params(params)
        stem = re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'h3').strip())[:40].strip('_') or 'h3'
        steps = int(params.steps or 50)
        if upscale is not None:
            params.width, params.height = (upscale.width, upscale.height)
        _cw, _ch, _clamp_note = (params.width, params.height, None) if upscale is not None else h3_clamp_dimensions(params.width, params.height, max_short_edge=params.h3_max_short_edge)
        if _clamp_note:
            print(f'WARNING: MiniMax H3 {_clamp_note}', file=sys.stderr, flush=True)
            if on_step:
                try:
                    on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_dimension_clamp', 'text': _clamp_note, 'requested': [int(params.width or 0), int(params.height or 0)], 'rendered': [_cw, _ch]})
                except Exception:
                    pass
            params.width, params.height = (_cw, _ch)
        out = out_dir / f'minimax_h3_{stem}_{params.width}x{params.height}_s{steps}_seed{params.seed}.mp4'
        if upscale is not None:
            import uuid
            out = out.with_stem(out.stem + '_' + upscale.mode + '_' + uuid.uuid4().hex[:8])
        runner = str(getattr(self.settings, 'cosmos_python', '') or sys.executable)
        cuda_device = self._cuda_device()
        torch_gpu = cuda_device.split(':', 1)[1] if ':' in cuda_device else '0'
        argv = [runner, '-m', 'cosmos_windows.weight_streaming.minimax_h3_backend', 'generate', '--model-dir', str(model_dir), '--prompt', params.prompt, '--mode', official_mode, '--out', str(out), '--duration', str((params.num_frames or 121) / float(params.fps or 24)), '--width', str(params.width), '--height', str(params.height), '--steps', str(steps), '--seed', str(params.seed), '--gpu', str(torch_gpu), '--attention-backend', str(params.h3_attention_backend), '--sparse-attention', str(getattr(params, 'h3_sparse_attention', 'auto') or 'auto'), '--sparse-method', str(getattr(params, 'h3_sparse_method', 'sol-attn') or 'sol-attn'), '--json-events']
        if params.h3_reference_resolution is not None:
            argv += ['--reference-resolution', str(int(params.h3_reference_resolution))]
        elif h3_profile is not None and h3_profile.reference_at_clip_short_edge:
            argv += ['--reference-resolution', str(min(int(params.width), int(params.height)))]
        if params.h3_seamless:
            argv += ['--seamless']
        if params.h3_extend_from:
            argv += ['--extend-from', params.h3_extend_from]
        if (official_mode in ('i2v', 'flf2v') or h3_start_frame) and params.image_path:
            argv += ['--image', str(params.image_path)]
        if official_mode in ('l2v', 'flf2v') and params.image2_path:
            argv += ['--last-frame', str(params.image2_path)]
        for frame_index in {'i2v': [0], 'l2v': [-1], 'flf2v': [0, -1]}.get(official_mode, [0] if h3_start_frame else []):
            argv += ['--frame-index', str(frame_index)]
        images = [str(r).strip() for r in params.reference_images or [] if str(r).strip()]
        videos: list[str] = []
        for candidate in [params.video_path or params.reference_video or params.source_video, *(params.reference_videos or [])]:
            v = str(candidate).strip() if candidate else ''
            if v and v not in videos:
                videos.append(v)
        audios: list[str] = []
        for candidate in [params.audio_path, *(params.reference_audios or [])]:
            a = str(candidate).strip() if candidate else ''
            if a and a not in audios:
                audios.append(a)
        if len(images) > 9:
            raise RuntimeError(f'MiniMax H3 Ref2VA accepts at most 9 reference images; got {len(images)}.')
        if len(videos) > 3:
            raise RuntimeError(f'MiniMax H3 Ref2VA accepts at most 3 reference videos; got {len(videos)}.')
        if len(audios) > 3:
            raise RuntimeError(f'MiniMax H3 Ref2VA accepts at most 3 reference audios; got {len(audios)}.')
        for ref in images:
            argv += ['--reference-image', ref]
        for video in videos:
            argv += ['--source-video', video]
        for audio in audios:
            argv += ['--reference-audio', audio]
        for lora in h3_profile_loras if h3_profile_loras is not None else params.loras or []:
            if str(lora.path).strip():
                argv += ['--lora', f'{lora.path}:{float(lora.weight):g}']
        t0 = time.time()
        extra = {}
        from .character_rotation_native import ROTATION_RECEIPT
        if upscale is not None:
            argv[2:4] = ['studio.h3_upscale_runner']
            argv += ['--upscale-json', upscale.model_dump_json()]
            extra['env_extra'] = {'PYTHONPATH': os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(repo), os.environ.get('PYTHONPATH', '')])}
        elif h3_profile is not None or ROTATION_RECEIPT.get():
            argv[2] = 'studio.h3_generate_runner'
            if ROTATION_RECEIPT.get():
                argv += ['--rotation-receipt']
            extra['env_extra'] = {'PYTHONPATH': os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(repo), os.environ.get('PYTHONPATH', '')])}
        if h3_layout_env:
            extra['env_extra'] = {**extra.get('env_extra', {}), **h3_layout_env}
        guarded = safety.guard_argv(argv, min_gb=6.0, repo=str(repo), python=runner)
        rc, lines = self._run_guarded_subprocess(guarded, 'minimax_h3', on_step=on_step, should_cancel=should_cancel, **extra)
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        if rc != 0:
            tail = failure_tail(lines, limit=1600)
            raise RuntimeError(f'MiniMax H3 generation failed (exit {rc}): {tail}')
        if not out.is_file():
            for line in reversed(lines):
                ev = __import__('studio.progress', fromlist=['parse_line']).parse_line(line, 'minimax_h3') or {}
                candidate = ev.get('path') if ev.get('type') in ('gen_done', 'result') else None
                if candidate and Path(str(candidate)).is_file():
                    out = Path(str(candidate))
                    break
        if not out.is_file():
            raise RuntimeError('MiniMax H3 completed without an output video')
        if image_mode:
            still = out.with_name(out.stem + '_still.png')
            h3img.extract_last_frame(out, still)
            if on_step:
                on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_derived_mode', 'text': f'extracted final frame -> {still.name}'})
            return GenResult(path=str(still), seconds=round(time.time() - t0, 2), modality=modality, width=params.width, height=params.height, kind='image', seed=params.seed, engine=self.name)
        if audio_mode:
            wav = out.with_name(out.stem + '_audio.wav')
            h3img.extract_audio(out, wav)
            if on_step:
                on_step(0, 0, None, {'type': 'gen_status', 'stage': 'h3_derived_mode', 'text': f'extracted audio track -> {wav.name}'})
            return GenResult(path=str(wav), seconds=round(time.time() - t0, 2), modality=modality, width=params.width, height=params.height, kind='audio', seed=params.seed, engine=self.name)
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality=modality, width=params.width, height=params.height, kind='video', seed=params.seed, engine=self.name)

    def _maybe_enhance_magi2(self, params, on_step=None) -> None:
        from . import llm_launch, magi2_prompt_enhancer as m2
        prompt = (getattr(params, 'prompt', '') or '').strip()
        if not prompt:
            return
        if m2.is_structured_caption(prompt):
            return
        if not bool(getattr(params, 'magi2_prompt_enhance', True)):
            if on_step:
                on_step({'event': 'stage_log', 'stage': 'magi2_prompt', 'level': 'warn', 'message': 'magi2_prompt_enhance is OFF and the prompt is not a structured caption — MAGI-2 will very likely render a black clip.'})
            return
        mode = 'i2v' if getattr(params, 'image_paths', None) else 't2v'
        system, source = m2.system_prompt(mode)
        endpoint = (os.environ.get('STUDIO_H3_ENHANCER_ENDPOINT') or '').strip()
        try:
            if endpoint:
                r = llm_launch.enhance_via_endpoint(endpoint, prompt, system=system, temperature=0.3, max_new_tokens=8000)
                raw, route = (r.get('answer') or '', f'endpoint:{endpoint}')
            elif llm_launch.active_server() is not None:
                r = llm_launch.enhance_via_server(prompt, system=system, temperature=0.3, max_new_tokens=8000)
                raw, route = (r.get('answer') or '', 'live-preset')
            else:
                raise RuntimeError('no enhancer backend available: set STUDIO_H3_ENHANCER_ENDPOINT or start an LLM preset in Connect')
        except Exception as exc:
            raise RuntimeError(f'MAGI-2 prompt compilation failed ({type(exc).__name__}: {exc}). MAGI-2 needs a structured JSON caption; sending the raw prompt would render a black clip. Fix the enhancer backend, or supply an already-compiled caption as the prompt.') from exc
        obj, err = m2.parse_caption(raw)
        if obj is None:
            raise RuntimeError(f"MAGI-2 prompt compilation produced an INVALID caption via {route} (spec {source}): {err}. Refusing to generate — an unstructured prompt renders black. Reply head: {(raw or '')[:200]!r}")
        params.prompt = m2.to_prompt_string(obj)
        if on_step:
            on_step({'event': 'stage_log', 'stage': 'magi2_prompt', 'message': f'prompt compiled to MAGI-2 structured caption via {route}', 'spec_source': source, 'mode': mode, **m2.describe(obj)})

    def _generate_magi2(self, params, on_step=None, should_cancel=None) -> GenResult:
        self._maybe_enhance_magi2(params, on_step=on_step)
        self.unload()
        self._cleanup_torch()
        from . import safety
        repo = Path(str(getattr(self.settings, 'cosmos_windows_path', '') or ''))
        runtime = repo / 'cosmos_windows' / 'weight_streaming' / 'magi2_backend.py'
        if not runtime.is_file():
            raise RuntimeError('MAGI-2 runtime is missing; set STUDIO_COSMOS_WINDOWS_PATH to the RTX repo containing cosmos_windows/weight_streaming/magi2_backend.py.')
        modality = str(params.modality or 't2v')
        mode = {'t2v': 't2v', 'i2v': 'i2v'}.get(modality)
        if not mode:
            raise RuntimeError(f'MAGI-2-preview supports t2v and i2v only; got modality {modality!r}')
        if mode == 'i2v' and (not params.image_path):
            raise RuntimeError('MAGI-2 i2v requires a first-frame image.')
        model_dir = Path(str(params.model or ''))
        if not model_dir.is_dir():
            raise RuntimeError(f'MAGI-2 checkpoint root is required, but it was not found at {model_dir}. Download the sand-ai/MAGI-2-preview snapshot (~307GB) and register its path.')
        _wh = (int(params.width or 0), int(params.height or 0))
        _exact = {(800, 448): '800x448', (640, 640): '640sq'}
        short_edge = min(_wh) if all(_wh) else 0
        resolution = _exact.get(_wh) or ('272p' if 0 < short_edge <= 256 else '540p' if 0 < short_edge <= 512 else '1080p')
        out_dir = output_dir_for_params(params)
        stem = re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'magi2').strip())[:40].strip('_') or 'magi2'
        out = out_dir / f'magi2_{stem}_{params.width}x{params.height}_seed{params.seed}.mp4'
        runner = str(getattr(self.settings, 'cosmos_python', '') or sys.executable)
        sf = dict(getattr(params, 'model_support_files', None) or {})
        argv = [runner, '-m', 'cosmos_windows.weight_streaming.magi2_backend', 'generate', '--model-dir', str(model_dir), '--prompt', params.prompt, '--mode', mode, '--out', str(out), '--resolution', resolution, '--seconds', '10', '--seed', str(params.seed), '--json-events']
        if params.width and params.height:
            argv += ['--output-width', str(int(params.width)), '--output-height', str(int(params.height))]
        if params.steps:
            argv += ['--steps', str(int(params.steps))]
        if mode == 'i2v' and params.image_path:
            argv += ['--image', str(params.image_path)]
        repo_root = str(sf.get('repo_root') or '').strip()
        if repo_root:
            argv += ['--repo-root', repo_root]
        world_size = str(sf.get('world_size') or '').strip()
        if world_size:
            argv += ['--world-size', world_size]
        engine = str(sf.get('engine') or '').strip()
        if engine:
            argv += ['--engine', engine]
        neg = str(getattr(params, 'negative_prompt', '') or sf.get('negative_prompt') or '').strip()
        if neg:
            argv += ['--negative-prompt', neg]
        batch_size = str(sf.get('batch_size') or '').strip()
        if batch_size:
            argv += ['--batch-size', batch_size]
        expert_dtype = str(sf.get('expert_dtype') or '').strip()
        if expert_dtype:
            argv += ['--expert-dtype', expert_dtype]
        compute_dtype = str(getattr(params, 'compute_dtype', '') or sf.get('compute_dtype') or '').strip()
        if compute_dtype:
            argv += ['--compute-dtype', compute_dtype]
        ring_budget = str(sf.get('ring_budget_gb') or '').strip()
        if ring_budget:
            argv += ['--ring-budget-gb', ring_budget]
        refiner_steps = str(sf.get('refiner_steps') or '').strip()
        if refiner_steps:
            argv += ['--refiner-steps', refiner_steps]
        magi_env = {'MAGI2_ATTN_TIER': str(sf.get('attn_tier') or 'fa2'), 'MAGI2_CFG_MODE': str(sf.get('cfg_mode') or ('batched' if resolution in ('272p', '640sq', '800x448') else 'auto')), 'MAGI2_RING_DEPTH': str(sf.get('ring_depth') or '1'), 'MAGI2_DS_GPU_BUFFERS': str(sf.get('ds_gpu_buffers') or '1'), 'MAGI2_TORCH_MEM_FRACTION': str(sf.get('torch_mem_fraction') or '0.90'), 'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,garbage_collection_threshold:0.7')}
        if sf.get('mlp_chunk_tokens'):
            magi_env['MAGI2_MLP_CHUNK_TOKENS'] = str(sf['mlp_chunk_tokens'])
        if on_step:
            on_step({'event': 'stage_log', 'stage': 'magi2_runtime', 'message': f'MAGI-2 dials (child env only): resolution={resolution}', **{k: v for k, v in magi_env.items() if k.startswith('MAGI2_')}})
        t0 = time.time()
        guarded = safety.guard_argv(argv, min_gb=6.0, repo=str(repo), python=runner)
        rc, lines = self._run_guarded_subprocess(guarded, 'magi2', on_step=on_step, should_cancel=should_cancel, env_extra=magi_env)
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        if rc != 0:
            tail = failure_tail(lines, limit=1600)
            raise RuntimeError(f'MAGI-2 generation failed (exit {rc}): {tail}')
        if not out.is_file():
            for line in reversed(lines):
                ev = __import__('studio.progress', fromlist=['parse_line']).parse_line(line, 'magi2') or {}
                candidate = ev.get('path') if ev.get('type') in ('gen_done', 'result') else None
                if candidate and Path(str(candidate)).is_file():
                    out = Path(str(candidate))
                    break
        if not out.is_file():
            raise RuntimeError('MAGI-2 completed without an output video')
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality=modality, width=params.width, height=params.height, kind='video', seed=params.seed, engine=self.name)

    def _generate_ideogram(self, params, on_step=None, should_cancel=None) -> GenResult:
        if os.environ.get('STUDIO_IDEOGRAM_INPROC', '').strip().lower() in ('1', 'true', 'on', 'yes'):
            try:
                return self._generate_ideogram_inproc(params, on_step=on_step, should_cancel=should_cancel)
            except Exception as exc:
                print(f'[ideo] in-process path failed ({type(exc).__name__}: {exc})', flush=True)
        if os.environ.get('STUDIO_IDEOGRAM_WORKER', '').strip().lower() in ('1', 'true', 'on', 'yes'):
            try:
                return self._generate_ideogram_worker(params, on_step=on_step, should_cancel=should_cancel)
            except Exception as exc:
                print(f'[ideo] persistent worker failed ({type(exc).__name__}: {exc}); falling back to the per-gen subprocess', flush=True)
        return self._generate_ideogram_subprocess(params, on_step=on_step, should_cancel=should_cancel)

    def _ensure_ideogram_worker(self, quant: str, weights_repo: str | None):
        import subprocess
        key = (quant, weights_repo or '')
        proc = getattr(self, '_ideogram_worker', None)
        if proc is not None and getattr(self, '_ideogram_worker_key', None) == key and (proc.poll() is None):
            return proc
        self._kill_ideogram_worker()
        repo_dir = str(getattr(self.settings, 'cosmos_windows_path', '') or '') or ''
        script = str(Path(repo_dir) / 'cosmos_windows' / 'ideogram_worker.py')
        if not Path(script).is_file():
            raise RuntimeError(f'ideogram_worker.py not found at {script}')
        ideo_src = os.environ.get('STUDIO_IDEOGRAM4_SRC', '')
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([p for p in (repo_dir, ideo_src, os.environ.get('PYTHONPATH', '')) if p])}
        argv = [self.settings.ideogram_python, '-u', script, '--quant', quant]
        if weights_repo:
            argv += ['--weights-repo', weights_repo]
        print(f'[ideo] spawning persistent worker (ideogram-venv): {self.settings.ideogram_python}', flush=True)
        proc = worker_registry.popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', env=env, bufsize=1)
        import json as _json
        t0 = time.time()
        while time.time() - t0 < 120:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError('ideogram worker exited before ready')
            try:
                msg = _json.loads(line.strip())
            except Exception:
                continue
            if msg.get('ready'):
                break
        else:
            raise RuntimeError('ideogram worker did not become ready in 120s')
        self._ideogram_worker = proc
        self._ideogram_worker_key = key
        return proc

    def _kill_ideogram_worker(self) -> None:
        proc = getattr(self, '_ideogram_worker', None)
        self._ideogram_worker = None
        self._ideogram_worker_key = None
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write('{"cmd":"quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _generate_ideogram_worker(self, params, on_step=None, should_cancel=None) -> GenResult:
        import json as _json
        quant = _infer_ideo_quant(params)
        weights_repo = str(params.model) if params.model and Path(str(params.model)).exists() else None
        seed = int(params.seed)
        prompt = params.prompt
        if getattr(params, 'composition_guide', None):
            prompt = _composition_guide_to_ideogram_prompt(params.composition_guide, params.prompt)
        out = self._image_out_path('ideogram', params.prompt, seed, params)
        key = (quant, weights_repo or '')
        cached = getattr(self, '_ideogram_worker', None) is not None and getattr(self, '_ideogram_worker_key', None) == key and (self._ideogram_worker.poll() is None)
        if not cached:
            self.unload()
            self._cleanup_torch()
        proc = self._ensure_ideogram_worker(quant, weights_repo)
        req = {'prompt': prompt, 'out': str(out), 'width': params.width, 'height': params.height, 'steps': params.steps, 'seed': seed, 'preset': str(getattr(params, 'ideo_preset', 'V4_QUALITY_48') or 'V4_QUALITY_48'), 'image': getattr(params, 'image_path', None), 'denoise': params.denoise, 'mask': getattr(params, 'mask_path', None)}
        t0 = time.time()
        proc.stdin.write(_json.dumps(req) + '\n')
        proc.stdin.flush()
        result = None
        while True:
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError('ideogram worker died during generation')
            try:
                msg = _json.loads(line.strip())
            except Exception:
                continue
            if msg.get('loaded'):
                print(f"[ideo] worker loaded model in {msg.get('seconds')}s", flush=True)
                continue
            if 'ok' in msg:
                result = msg
                break
        if not result.get('ok'):
            raise RuntimeError(f"ideogram worker error: {result.get('error')}")
        path = result.get('path') or str(out)
        if not Path(path).is_file():
            raise RuntimeError('ideogram worker reported ok but no output file')
        return GenResult(path=str(path), seconds=round(time.time() - t0, 2), modality='i2i' if getattr(params, 'image_path', None) else 't2i', width=params.width, height=params.height, kind='image', seed=seed, engine=self.name)

    def _generate_ideogram_inproc(self, params, on_step=None, should_cancel=None) -> GenResult:
        from . import ideogram_pipeline as ip
        quant = _infer_ideo_quant(params)
        weights_repo = str(params.model) if params.model and Path(str(params.model)).exists() else None
        device = self._cuda_device()
        preset = str(getattr(params, 'ideo_preset', 'V4_QUALITY_48') or 'V4_QUALITY_48')
        seed = int(params.seed)
        prompt = params.prompt
        if getattr(params, 'composition_guide', None):
            prompt = _composition_guide_to_ideogram_prompt(params.composition_guide, params.prompt)
        out = self._image_out_path('ideogram', params.prompt, seed, params)
        pipe_key = ('ideogram', weights_repo or quant, quant, device)
        t0 = time.time()
        if self._resident_aux_key == pipe_key and self._resident_aux_pipe is not None:
            pipe, ig = self._resident_aux_pipe
            print(f'[ideo] resident cache hit on {device}', flush=True)
        else:
            print(f'[ideo] building resident Ideogram4 {quant} on {device}', flush=True)
            self.unload()
            self._cleanup_torch()
            bt0 = time.time()
            pipe, ig = ip.build_ideogram_pipeline(quant=quant, weights_repo=weights_repo, device=device, cosmos_windows_repo=str(getattr(self.settings, 'cosmos_windows_path', '') or '') or None)
            print(f'[ideo] built in {time.time() - bt0:.0f}s', flush=True)
            self._resident_aux_pipe = (pipe, ig)
            self._resident_aux_key = pipe_key
            self._resident_aux_label = 'ideogram'

        def _cb(step, total):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                try:
                    on_step(int(step), int(total), None)
                except TypeError:
                    on_step(int(step), int(total))
        ip.ideogram_generate(pipe, ig, prompt=prompt, out_path=str(out), preset=preset, width=params.width, height=params.height, steps=params.steps, seed=seed, image=getattr(params, 'image_path', None), denoise=params.denoise, mask=getattr(params, 'mask_path', None), on_step=_cb if on_step or should_cancel else None)
        if not out.is_file():
            raise RuntimeError('Ideogram in-process generation produced no output')
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if getattr(params, 'image_path', None) else 't2i', width=params.width, height=params.height, kind='image', seed=seed, engine=self.name)

    def _generate_ideogram_subprocess(self, params, on_step=None, should_cancel=None) -> GenResult:
        self.unload()
        self._cleanup_torch()
        from . import safety
        ok, free = safety.enough_ram(6.0)
        if not ok:
            raise RuntimeError(f'Not enough free RAM for Ideogram 4: {free:.1f} GB free, need >= 6 GB.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        script = Path(repo) / 'cosmos_windows' / 'ideogram_gen.py'
        if not script.is_file():
            raise RuntimeError('Ideogram helper is missing; set STUDIO_COSMOS_WINDOWS_PATH to the RTX repo that contains cosmos_windows/ideogram_gen.py.')
        prompt = params.prompt
        if getattr(params, 'composition_guide', None):
            prompt = _composition_guide_to_ideogram_prompt(params.composition_guide, params.prompt)
        out = self._image_out_path('ideogram', params.prompt, params.seed, params)
        argv = [self.settings.ideogram_python, str(script), '--prompt', prompt, '--out', str(out), '--width', str(params.width), '--height', str(params.height), '--preset', str(getattr(params, 'ideo_preset', 'V4_QUALITY_48') or 'V4_QUALITY_48'), '--seed', str(params.seed), '--quant', _infer_ideo_quant(params), '--gpu', self.settings.gpu_index]
        if getattr(params, 'image_path', None):
            argv += ['--image', str(params.image_path)]
            argv += ['--denoise', str(params.denoise if params.denoise is not None else 1.0)]
        if getattr(params, 'mask_path', None):
            argv += ['--mask', str(params.mask_path)]
        model_path = str(params.model or '')
        if model_path and Path(model_path).exists():
            argv += ['--weights-repo', model_path]
        t0 = time.time()
        rc, lines = self._run_guarded_subprocess(safety.guard_argv(argv, min_gb=4.0, repo=repo, python=self.settings.ideogram_python), 'ideogram_gen', on_step=on_step, should_cancel=should_cancel)
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        if rc != 0:
            tail = failure_tail(lines, limit=1000)
            raise RuntimeError(f'Ideogram generation failed (exit {rc}): {tail}')
        if not out.is_file():
            for line in reversed(lines):
                m = re.search('RESULT DONE\\s+(.+\\.(?:png|jpg|jpeg|webp))', line, re.I)
                if m and Path(m.group(1).strip()).is_file():
                    out = Path(m.group(1).strip())
                    break
        if not out.is_file():
            raise RuntimeError('Ideogram generation completed without an output image')
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='t2i', width=params.width, height=params.height, kind='image', seed=params.seed, engine=self.name)

    @staticmethod
    def _is_lora_only_safetensors(path: str | Path | None) -> bool:
        if not path:
            return False
        p = Path(path)
        if p.suffix.lower() != '.safetensors' or not p.is_file():
            return False
        try:
            from safetensors import safe_open
            with safe_open(str(p), framework='pt', device='cpu') as fh:
                keys = list(fh.keys())
        except Exception:
            try:
                import json
                import struct
                with p.open('rb') as fh:
                    raw_len = fh.read(8)
                    if len(raw_len) != 8:
                        return False
                    header_len = struct.unpack('<Q', raw_len)[0]
                    header = json.loads(fh.read(header_len).decode('utf-8').strip())
                keys = [k for k in header.keys() if k != '__metadata__']
            except Exception:
                return False
        return bool(keys) and all(('.lora_' in k for k in keys))

    def _generate_zimage(self, params, on_step=None, should_cancel=None) -> GenResult:
        is_i2i = (params.modality or 't2i').lower() == 'i2i' and bool(params.image_path)
        if is_i2i and (not _sdedit_i2i_enabled('zimage')):
            raise RuntimeError('Z-Image i2i (SDEdit) is disabled via STUDIO_SDEDIT_I2I=0. Re-enable it, or use t2i / an i2i-capable model. (Never silently degrades an i2i request to t2i.)')
        from . import safety
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        zimage = importlib.import_module('cosmos_windows.weight_streaming.zimage_backend')
        import inspect
        import torch
        model_file = params.model
        if not model_file or not Path(model_file).is_file():
            raise RuntimeError(f'Z-Image model path not found: {model_file!r}')
        sf = dict(params.model_support_files or {})
        transformer_file = sf.get('transformer') or model_file
        t0 = time.time()
        from .models import resolve_load_mode
        _load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        resident_mode = is_resident
        stream_weights = not is_resident
        device = self._cuda_device()
        pipe_key = ('zimage', transformer_file, sf.get('config_dir'), sf.get('text_encoder'), sf.get('vae'), sf.get('tokenizer'), sf.get('scheduler'), device, self._lora_cache_key(params.loras), is_i2i)
        if resident_mode and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None):
            pipe = self._resident_aux_pipe
        else:
            min_free = float(os.environ.get('STUDIO_ZIMAGE_MIN_FREE_RAM_GB', '6'))
            ok, free = safety.enough_ram(min_free)
            if not ok:
                mode = 'resident' if resident_mode else 'streamed'
                raise RuntimeError(f'Not enough system RAM to build {mode} Z-Image: {free:.1f} GB free, need >= {min_free:.0f} GB. Close other apps and retry.')
            self.unload()
            self._cleanup_torch()
            pipe = zimage.build_zimage_pipeline(transformer_file=transformer_file, config_dir=sf.get('config_dir'), text_encoder=sf.get('text_encoder'), vae=sf.get('vae'), tokenizer_dir=sf.get('tokenizer'), scheduler_dir=sf.get('scheduler'), device=device, offload=False, stream_weights=stream_weights, n_gpu_buffers=int(os.environ.get('STUDIO_ZIMAGE_GPU_BUFFERS', '1')), loras=[l.model_dump() for l in params.loras or []] or None, i2i=is_i2i)
            if resident_mode:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'z-image'
        try:
            vae = getattr(pipe, 'vae', None)
            if vae is not None and hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
            call_params = set(inspect.signature(pipe.__call__).parameters)
            custom_sigmas = None
            configure_sampling = getattr(zimage, 'configure_zimage_sampling', None)
            if callable(configure_sampling):
                custom_sigmas = configure_sampling(pipe, sampler=params.sampler, sigmas=params.sigmas, num_inference_steps=int(params.steps or 8))
            kwargs = {'prompt': params.prompt, 'height': params.height, 'width': params.width, 'num_inference_steps': int(params.steps or 8)}
            if custom_sigmas is not None and 'sigmas' in call_params:
                kwargs['sigmas'] = custom_sigmas
            if 'guidance_scale' in call_params:
                kwargs['guidance_scale'] = params.guidance if params.guidance is not None else 1.0
            if 'generator' in call_params:
                kwargs['generator'] = torch.Generator('cpu').manual_seed(int(params.seed))
            if is_i2i and 'image' in call_params:
                from PIL import Image as _PILImage
                kwargs['image'] = _PILImage.open(params.image_path).convert('RGB')
                if 'strength' in call_params:
                    kwargs['strength'] = _resolve_denoise_strength(params)
            if 'callback_on_step_end' in call_params and (on_step or should_cancel):
                _zi_total = int(params.steps or 8)

                def _zi_step_cb(_pipe, _step, _t, _cbk):
                    if should_cancel and should_cancel():
                        raise RuntimeError('generation cancelled')
                    if on_step:
                        on_step(int(_step) + 1, _zi_total, None)
                    return _cbk
                kwargs['callback_on_step_end'] = _zi_step_cb
            result = pipe(**kwargs)
            img = self._first_image(result)
            out = self._image_out_path('zimage', params.prompt, params.seed, params)
            img.save(str(out))
        finally:
            if not resident_mode:
                try:
                    streamer = getattr(pipe, '_zimage_streamer', None) or getattr(getattr(pipe, 'transformer', None), '_zimage_streamer', None)
                    close = getattr(streamer, 'close', None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if is_i2i else 't2i', width=params.width, height=params.height, kind='image', seed=params.seed, engine=self.name)

    def _generate_ultratex(self, params, on_step=None, should_cancel=None) -> GenResult:
        from .ultratex import generate_views
        from .models import resolve_load_mode
        files = dict(params.model_support_files or {})
        if not files.get('normal_views') or not params.image_path:
            raise ValueError('UltraTex requires six world-normal views and a reference image; use Character Builder Texture')
        if params.loras:
            raise ValueError('UltraTex already applies its trained adapter; extra LoRA stacks are unsupported')
        mode, resident, lazy = resolve_load_mode(params)
        args = {'ultratex_model_dir': str(params.model), 'ultratex_device': self._cuda_device(), 'ultratex_load_mode': 'resident' if resident else 'streaming', 'ultratex_steps': params.steps, 'ultratex_seed': params.seed}
        args.update({'ultratex_' + key: value for key, value in files.items() if key != 'normal_views'})
        out = output_dir_for_params(params) / f'ultratex_{int(time.time())}'
        t0 = time.time()
        views, metadata = generate_views(json.loads(Path(files['normal_views']).read_text(encoding='utf-8')), params.image_path, out, args, on_step=on_step, should_cancel=should_cancel)
        from PIL import Image
        with Image.open(out / 'albedo_grid.png') as grid:
            width, height = grid.size
        return GenResult(path=str(out / 'albedo_grid.png'), seconds=round(time.time() - t0, 2), modality=params.modality, width=width, height=height, kind='image', seed=params.seed, engine=self.name)

    def _generate_flux2klein(self, params, on_step=None, should_cancel=None) -> GenResult:
        import torch
        from .models import resolve_load_mode
        t0 = time.time()
        device = self._cuda_device()
        model_path = str(params.model or '')
        sf = dict(params.model_support_files or {})
        _default_klein = ''
        if not os.path.isdir(_default_klein):
            _default_klein = ''
        repo_dir = model_path if Path(model_path).is_dir() else sf.get('components_repo') or os.environ.get('STUDIO_FLUX2_KLEIN_DIR', _default_klein)
        _load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        keep_resident = is_resident
        from diffusers import Flux2KleinKVPipeline, Flux2Transformer2DModel
        _klein_single_file = bool(model_path and Path(model_path).is_file() and (Path(model_path) != Path(repo_dir)))

        def _build_klein_pipe():
            if _klein_single_file:
                p = Flux2KleinKVPipeline.from_pretrained(repo_dir, torch_dtype=torch.bfloat16)
                p.transformer = Flux2Transformer2DModel.from_single_file(model_path, torch_dtype=torch.bfloat16)
                return p.to(device)
            from transformers import Qwen3ForCausalLM, AutoTokenizer
            from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
            transformer = Flux2Transformer2DModel.from_pretrained(repo_dir, subfolder='transformer', torch_dtype=torch.bfloat16, device_map=device)
            text_encoder = Qwen3ForCausalLM.from_pretrained(repo_dir, subfolder='text_encoder', torch_dtype=torch.bfloat16, device_map=device)
            vae = AutoencoderKLFlux2.from_pretrained(repo_dir, subfolder='vae', torch_dtype=torch.bfloat16).to(device)
            tokenizer = AutoTokenizer.from_pretrained(repo_dir, subfolder='tokenizer')
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo_dir, subfolder='scheduler')
            return Flux2KleinKVPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, is_distilled=True)
        mgr = self._get_vram_mgr()
        klein_key = f'flux2-klein::{repo_dir}::{model_path}'
        component_resident = keep_resident and (not _klein_single_file) and (str(os.environ.get('STUDIO_KLEIN_COMPONENT_RESIDENT', '')).strip() not in ('', '0'))
        _release_keys = []
        if component_resident:
            from transformers import Qwen3ForCausalLM, AutoTokenizer
            from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
            tkey = f'flux2-klein:transformer::{repo_dir}'
            ekey = f'flux2-klein:text_encoder::{repo_dir}'
            vkey = f'flux2-klein:vae::{repo_dir}'
            warm = mgr.is_resident(tkey) and mgr.is_resident(ekey) and mgr.is_resident(vkey)
            if not warm and self._resident_aux_pipe is not None:
                self._clear_resident_aux_pipe(keep_manager=True)
            print(f"[flux2-klein] {('WARM component-resident reuse' if warm else 'building COMPONENT-resident direct-to-VRAM pipeline')} on {device} (transformer/text_encoder/vae as separate manager keys)", flush=True)
            transformer = mgr.acquire(tkey, lambda: Flux2Transformer2DModel.from_pretrained(repo_dir, subfolder='transformer', torch_dtype=torch.bfloat16, device_map=device), est_gb=14.0, pin=True)
            text_encoder = mgr.acquire(ekey, lambda: Qwen3ForCausalLM.from_pretrained(repo_dir, subfolder='text_encoder', torch_dtype=torch.bfloat16, device_map=device), est_gb=2.0, pin=True)
            vae = mgr.acquire(vkey, lambda: AutoencoderKLFlux2.from_pretrained(repo_dir, subfolder='vae', torch_dtype=torch.bfloat16).to(device), est_gb=0.16, pin=True)
            tokenizer = AutoTokenizer.from_pretrained(repo_dir, subfolder='tokenizer')
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo_dir, subfolder='scheduler')
            pipe = Flux2KleinKVPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, is_distilled=True)
            _release_keys = [tkey, ekey, vkey]
            self._resident_aux_label = 'flux2-klein'
        elif keep_resident and mgr.is_resident(klein_key):
            pipe = mgr.acquire(klein_key, _build_klein_pipe, pin=True)
        else:
            print(f"[flux2-klein] building {('resident' if keep_resident else 'direct-disposable')} {('single-file' if _klein_single_file else 'direct-to-VRAM')} pipeline on {device}", flush=True)
            self.unload()
            self._cleanup_torch()
            if _klein_single_file:
                from . import ram_preflight
                ram_preflight.preflight_resident_ram(24, 'FLUX.2-Klein single-file load — staged through host RAM', env_prefix='FLUX2_KLEIN', advice='Free host RAM (unload other models) before a single-file FLUX.2-Klein load.')
            if keep_resident:
                pipe = mgr.acquire(klein_key, _build_klein_pipe, est_gb=16.0, pin=True)
            else:
                pipe = _build_klein_pipe()
        if keep_resident:
            self._resident_aux_label = 'flux2-klein'
        gen = torch.Generator(device=device).manual_seed(int(params.seed))
        steps = int(params.steps or 4)
        kwargs = dict(prompt=params.prompt, num_inference_steps=steps, width=params.width, height=params.height, generator=gen)
        import inspect as _inspect
        call_parameters = _inspect.signature(pipe.__call__).parameters
        if params.guidance is not None and 'guidance_scale' in call_parameters:
            kwargs['guidance_scale'] = float(params.guidance)
        if params.image_path:
            from PIL import Image as _Img
            kwargs['image'] = _Img.open(params.image_path).convert('RGB')
            if 'strength' in call_parameters:
                kwargs['strength'] = _resolve_denoise_strength(params)
        cb_holder = {'i': 0}

        def _cb(pipe_, step_idx, t, cbk):
            cb_holder['i'] = step_idx + 1
            if on_step:
                on_step(step_idx + 1, steps)
            if should_cancel and should_cancel():
                raise RuntimeError('FLUX.2-Klein generation cancelled')
            return cbk

        def _run_klein():
            from .two_pass import run_pipeline
            from .generation_loras import phase_loras
            specs = [row.model_dump() for row in params.loras]
            option = getattr(params, 'two_pass', None)
            if 'callback_on_step_end' in call_parameters:
                kwargs['callback_on_step_end'] = _cb
            elif on_step is not None or should_cancel is not None:
                raise ValueError('Installed FLUX.2-Klein pipeline has no bounded cancellation callback')
            with phase_loras(pipe.transformer, specs, option, device=device, receipt=lora_receipt) as apply:
                return run_pipeline(pipe, kwargs, option, apply, first_pass=specs).images[0]
        try:
            lora_receipt = {}
            image = _run_klein()
            out_dir = output_dir_for_params(params)
            from .sim import _safe
            import uuid
            out = out_dir / f'klein_{_safe(params.prompt)[:28]}_{int(t0)}_{uuid.uuid4().hex[:12]}.png'
            image.save(out)
            if lora_receipt:
                Path(str(out) + '.loras.json').write_text(json.dumps(lora_receipt, indent=2), encoding='utf-8')
            return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality=params.modality, width=params.width, height=params.height, kind='image', seed=params.seed, engine=self.name)
        finally:
            if component_resident:
                for _k in _release_keys:
                    try:
                        self._get_vram_mgr().release(_k)
                    except Exception:
                        pass
            elif keep_resident:
                try:
                    self._get_vram_mgr().release(klein_key)
                except Exception:
                    pass
            else:
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()

    def _generate_anima(self, params, on_step=None, should_cancel=None) -> GenResult:
        anima_i2i = (params.modality or 't2i').lower() == 'i2i' and bool(params.image_path)
        if anima_i2i and (not _sdedit_i2i_enabled('anima')):
            raise RuntimeError('Anima i2i (SDEdit) is disabled via STUDIO_SDEDIT_I2I=0. Re-enable it, or use t2i / an i2i-capable model. (Never silently degrades an i2i request to t2i.)')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        anima = importlib.import_module('cosmos_windows.weight_streaming.anima_backend')
        import inspect
        import torch
        model_file = params.model
        if not model_file or not Path(model_file).is_file():
            raise RuntimeError(f'Anima model path not found: {model_file!r}')
        sf = dict(params.model_support_files or {})
        qwen = sf.get('text_encoder') or getattr(anima, 'DEFAULT_QWEN', '')
        if '06b' not in Path(qwen).name.lower() and Path(getattr(anima, 'DEFAULT_QWEN', '')).is_file():
            qwen = getattr(anima, 'DEFAULT_QWEN', '')
        anima_dir = sf.get('components_repo') or sf.get('anima_dir') or getattr(anima, 'DEFAULT_DIR', '')
        t0 = time.time()
        from .models import resolve_load_mode
        force_oneshot = os.environ.get('STUDIO_ANIMA_ONESHOT', '').strip().lower() in ('1', 'true', 'on', 'yes')
        _load_mode, _anima_is_resident, _is_lazy = resolve_load_mode(params)
        resident_mode = _anima_is_resident and (not force_oneshot)
        stream_weights = not resident_mode
        device = self._cuda_device()
        pipe_key = ('anima', model_file, qwen, anima_dir, device, stream_weights, self._lora_cache_key(params.loras))
        total_steps = int(params.steps or 24)
        if resident_mode and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None):
            pipe = self._resident_aux_pipe
            print(f'[ANIMA] resident cache hit on {device}', flush=True)
        else:
            print(f"[ANIMA] building {('resident' if resident_mode else 'streamed-block')} pipeline on {device} (weight_stream={bool(params.weight_stream)})", flush=True)
            self.unload()
            self._cleanup_torch()
            build_t0 = time.time()
            pipe = anima.build_anima_pipeline(transformer_file=model_file, qwen_file=qwen, anima_dir=anima_dir, device=device, loras=[l.model_dump() for l in params.loras or []] or None, stream_weights=stream_weights)
            print(f'[ANIMA] pipeline built in {time.time() - build_t0:.2f}s', flush=True)
            if resident_mode:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'anima'
        try:
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(0, total_steps, None, {'text': 'Anima pipeline ready; preparing prompt'})
            vae = None
            if hasattr(pipe, 'get_component'):
                try:
                    vae = pipe.get_component('vae')
                except Exception:
                    vae = None
            vae = vae or getattr(pipe, 'vae', None)
            if vae is not None and hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
            call_params = set(inspect.signature(pipe.__call__).parameters)
            custom_sigmas = None
            configure_sampling = getattr(anima, 'configure_anima_sampling', None)
            if callable(configure_sampling):
                custom_sigmas = configure_sampling(pipe, sampler=params.sampler, sigmas=params.sigmas, num_inference_steps=total_steps)
            kwargs = {'prompt': params.prompt, 'height': params.height, 'width': params.width, 'num_inference_steps': total_steps, 'generator': torch.Generator(device).manual_seed(int(params.seed)), 'output': 'images'}
            if 'callback_on_step_end' in call_params:

                def _callback_on_step_end(pipe_obj, step, timestep, callback_kwargs):
                    if should_cancel and should_cancel():
                        try:
                            setattr(pipe_obj, '_interrupt', True)
                        except Exception:
                            pass
                        raise RuntimeError('generation cancelled')
                    if on_step:
                        on_step(int(step) + 1, total_steps, None)
                    return callback_kwargs
                kwargs['callback_on_step_end'] = _callback_on_step_end
                if 'callback_on_step_end_tensor_inputs' in call_params:
                    kwargs['callback_on_step_end_tensor_inputs'] = []
            elif 'callback' in call_params:

                def _callback(step, timestep, latents):
                    if should_cancel and should_cancel():
                        raise RuntimeError('generation cancelled')
                    if on_step:
                        on_step(int(step) + 1, total_steps, None)
                kwargs['callback'] = _callback
                if 'callback_steps' in call_params:
                    kwargs['callback_steps'] = 1
            if custom_sigmas is not None and 'sigmas' in call_params:
                kwargs['sigmas'] = custom_sigmas
            denoise_t0 = time.time()
            print(f"[ANIMA] denoise start: {total_steps} steps {params.width}x{params.height}{(' (SDEdit i2i)' if anima_i2i else '')}", flush=True)
            _anima_sched = None
            _anima_orig_step = None
            if on_step:
                try:
                    _anima_sched = pipe.get_component('scheduler') if hasattr(pipe, 'get_component') else getattr(pipe, 'scheduler', None)
                except Exception:
                    _anima_sched = None
                if _anima_sched is not None and hasattr(_anima_sched, 'step'):
                    _anima_orig_step = _anima_sched.step
                    _anima_ctr = {'n': 0}

                    def _anima_sched_step(*_a, _orig=_anima_orig_step, _ctr=_anima_ctr, **_k):
                        _ctr['n'] += 1
                        try:
                            if should_cancel and should_cancel():
                                raise RuntimeError('generation cancelled')
                            on_step(min(_ctr['n'], total_steps), total_steps, None)
                        except RuntimeError:
                            raise
                        except Exception:
                            pass
                        return _orig(*_a, **_k)
                    _anima_sched.step = _anima_sched_step
            try:
                if anima_i2i and hasattr(anima, 'anima_generate'):
                    cb_kw = {k: v for k, v in kwargs.items() if k in ('callback_on_step_end', 'callback_on_step_end_tensor_inputs', 'callback', 'callback_steps')}
                    result = anima.anima_generate(pipe, prompt=params.prompt, image_path=params.image_path, strength=_resolve_denoise_strength(params), height=params.height, width=params.width, num_inference_steps=total_steps, sampler=params.sampler, sigmas=params.sigmas, seed=int(params.seed), output='images', **cb_kw)
                else:
                    result = pipe(**kwargs)
            finally:
                if _anima_sched is not None and _anima_orig_step is not None:
                    _anima_sched.step = _anima_orig_step
            print(f'[ANIMA] denoise/decode finished in {time.time() - denoise_t0:.2f}s', flush=True)
            img = self._first_image(result)
            out = self._image_out_path('anima', params.prompt, params.seed, params)
            img.save(str(out))
            if on_step:
                on_step(total_steps, total_steps, None)
        finally:
            if not resident_mode:
                try:
                    transformer = pipe.get_component('transformer') if hasattr(pipe, 'get_component') else None
                except Exception:
                    transformer = None
                transformer = transformer or getattr(pipe, 'transformer', None)
                streamer = getattr(transformer, '_anima_streamer', None)
                close = getattr(streamer, 'close', None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if anima_i2i else 't2i', width=params.width, height=params.height, kind='image', seed=params.seed, engine=self.name)

    def _generate_glm(self, params, on_step=None, should_cancel=None) -> GenResult:
        import torch
        model_dir = os.environ.get('STUDIO_GLM_MODEL') or str(getattr(self.settings, 'glm_model', '') or '')
        if getattr(params, 'model', None) and Path(str(params.model)).is_dir():
            model_dir = str(params.model)
        if not Path(model_dir).is_dir():
            raise RuntimeError(f'GLM-Image model dir not found ({model_dir!r}')
        w, h = (int(params.width or 1024), int(params.height or 1024))
        steps = int(params.steps or 28)
        guidance = float(params.guidance if params.guidance is not None else 3.5)
        seed = int(params.seed)
        out = self._image_out_path('glm', params.prompt, seed, params)
        is_i2i = bool(getattr(params, 'image_path', None))
        emit_bin = _resolve_glm_emit_bin(str(getattr(self.settings, 'cosmos_windows_path', '') or ''))
        ar_gguf = os.environ.get('STUDIO_GLM_AR_GGUF', '')
        fast_ar = not is_i2i and os.environ.get('STUDIO_GLM_FAST_AR', '1').strip().lower() not in ('0', 'false', 'off', 'no') and Path(emit_bin).is_file() and Path(ar_gguf).is_file()
        if os.environ.get('STUDIO_GLM_SUBPROCESS', '').strip().lower() in ('1', 'true', 'on', 'yes'):
            return self._generate_glm_subprocess(params, model_dir, w, h, steps, guidance, seed, out, fast_ar, emit_bin, ar_gguf, should_cancel=should_cancel, on_step=on_step)
        from . import glm_pipeline
        device = self._cuda_device()
        from .models import resolve_load_mode
        load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        resident_mode = is_resident
        offload = 'none' if is_resident else os.environ.get('STUDIO_GLM_OFFLOAD', 'stream')
        skip_ar = fast_ar
        pipe_key = ('glm', model_dir, device, offload, skip_ar, load_mode)
        t0 = time.time()
        cache_hit = resident_mode and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None)
        if cache_hit:
            pipe = self._resident_aux_pipe
            print(f'[GLM] resident cache hit on {device}', flush=True)
        else:
            print(f"[GLM] building {('resident' if resident_mode else f'streaming({offload})')} pipeline on {device} (fast_ar={fast_ar})", flush=True)
            self.unload()
            self._cleanup_torch()
            build_t0 = time.time()
            pipe, _has_macos, _dev, _dtype = glm_pipeline.build_glm_pipeline(model_dir, diffusion_device=device, offload=offload, skip_ar=skip_ar, load_mode=load_mode, should_cancel=should_cancel)
            print(f'[GLM] pipeline built in {time.time() - build_t0:.2f}s', flush=True)
            if resident_mode:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'glm'
        pipe._studio_glm_loading['cache_hit'] = cache_hit
        try:
            ref_image = None
            if is_i2i:
                from PIL import Image, ImageOps
                ref_image = [ImageOps.fit(Image.open(params.image_path).convert('RGB'), (w, h), method=Image.LANCZOS)]
            prior = None
            if fast_ar:
                sf = dict(params.model_support_files or {})

                def _f(key, env, default):
                    try:
                        return float(sf.get(key) or os.environ.get(env, default))
                    except (TypeError, ValueError):
                        return float(default)
                pg_pen = getattr(params, 'glm_rep_penalty', None)
                pg_win = getattr(params, 'glm_rep_window', None)
                rep_penalty = float(pg_pen) if pg_pen is not None else _f('rep_penalty', 'STUDIO_GLM_REP_PENALTY', 1.1)
                rep_window = int(pg_win) if pg_win is not None else int(_f('rep_window', 'STUDIO_GLM_REP_WINDOW', 64))
                prior = glm_pipeline.emit_fast_ar_prior(pipe, prompt=params.prompt, width=w, height=h, seed=seed, emit_bin=emit_bin, ar_gguf=ar_gguf, rep_penalty=rep_penalty, rep_window=rep_window)
            lora_receipt = {}
            image = glm_pipeline.glm_diffuse(pipe, prompt=params.prompt, prior_token_ids=prior, width=w, height=h, steps=steps, guidance=guidance, seed=seed, ref_image=ref_image, has_macos=False, diffusion_device=device, on_step=on_step, should_cancel=should_cancel, free_te_after_encode=load_mode == 'resident_lazy', model_dir=model_dir, loras=params.loras, receipt=lora_receipt)
            image.save(str(out))
            loading = getattr(pipe, '_studio_glm_loading', None)
            if lora_receipt.get('loras'):
                Path(str(out) + '.loras.json').write_text(json.dumps(lora_receipt, indent=2), encoding='utf-8')
        finally:
            if not resident_mode:
                streamer = getattr(getattr(pipe, 'transformer', None), '_glm_streamer', None)
                if streamer is not None:
                    try:
                        streamer.close()
                    except Exception:
                        pass
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
        if not out.is_file():
            raise RuntimeError('GLM-Image generation produced no output file')
        if loading is not None:
            Path(str(out) + '.loading.json').write_text(json.dumps(loading, indent=2), encoding='utf-8')
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if is_i2i else 't2i', width=w, height=h, kind='image', seed=seed, engine=self.name, timings={'load_mode': load_mode, 'cache_hit': cache_hit, 'generation': loading.get('generation') if loading else None})

    def _generate_triposplat(self, params, on_step=None, should_cancel=None) -> GenResult:
        from . import triposplat_backend as tb
        img = getattr(params, 'image_path', None)
        if not img or not Path(str(img)).is_file():
            raise RuntimeError("TripoSplat is image->3D — set image_path to the source image (the GUI 'image to 3D' input).")
        sf = dict(params.model_support_files or {})

        def _i(key, default):
            try:
                return int(float(sf.get(key) or default))
            except (TypeError, ValueError):
                return int(default)

        def _f(key, default):
            try:
                return float(sf.get(key) or default)
            except (TypeError, ValueError):
                return float(default)
        num_gaussians = _i('num_gaussians', 262144)
        steps = int(params.steps or _i('steps', 20))
        guidance = float(params.guidance if params.guidance is not None else _f('guidance', 3.0))
        shift = _f('shift', 3.0)
        seed = int(params.seed)
        device = self._cuda_device()
        pipe_key = ('triposplat', device)
        t0 = time.time()
        if self._resident_aux_key == pipe_key and self._resident_aux_pipe is not None:
            pipe = self._resident_aux_pipe
            print(f'[TripoSplat] resident cache hit on {device}', flush=True)
        else:
            print(f'[TripoSplat] building pipeline on {device}', flush=True)
            self.unload()
            self._cleanup_torch()
            bt0 = time.time()
            pipe, _dev = tb.build_triposplat_pipeline(device=device)
            print(f'[TripoSplat] pipeline built in {time.time() - bt0:.2f}s', flush=True)
            self._resident_aux_pipe = pipe
            self._resident_aux_key = pipe_key
            self._resident_aux_label = 'triposplat'
        base = self._image_out_path('tripo', params.prompt or Path(str(img)).stem, seed, params).stem
        out_dir = output_dir_for_params(params)
        res = tb.generate_splat(pipe, image=str(img), out_dir=str(out_dir), basename=base, num_gaussians=num_gaussians, steps=steps, guidance=guidance, shift=shift, seed=seed, on_step=on_step, should_cancel=should_cancel)
        gaussian = res['gaussian']
        preview = res.get('prepared')
        try:
            azimuths = None
            csv = (sf.get('render_views') or '').strip()
            if csv:
                azimuths = [float(x) for x in csv.replace(';', ',').split(',') if x.strip()]
            views = tb.render_splat_views(gaussian, out_dir=str(out_dir), basename=f'{base}_view', azimuths=azimuths)
            if views:
                preview = views[0]
        except Exception as e:
            print(f'[TripoSplat] view render skipped: {type(e).__name__}: {e}', flush=True)
        mesh = None
        if str(sf.get('export_mesh', '')).strip().lower() in ('1', 'true', 'on', 'yes'):
            try:
                mesh = tb.export_splat_mesh(gaussian, out_path=str(out_dir / f'{base}.obj'))
            except Exception as e:
                print(f'[TripoSplat] mesh export skipped: {type(e).__name__}: {e}', flush=True)
        if not Path(res['ply']).is_file():
            raise RuntimeError('TripoSplat produced no .ply output')
        return GenResult(path=res['ply'], seconds=round(time.time() - t0, 2), modality='image-to-3d', width=int(params.width or 1024), height=int(params.height or 1024), kind='splat', seed=seed, engine=self.name, splat=res.get('splat'), mesh=mesh, preview=preview)

    def _generate_glm_subprocess(self, params, model_dir, w, h, steps, guidance, seed, out, fast_ar, emit_bin, ar_gguf, should_cancel=None, on_step=None) -> GenResult:
        import subprocess
        glm_py = os.environ.get('STUDIO_GLM_PYTHON') or ''
        if not Path(glm_py).is_file():
            raise RuntimeError(f'GLM python not found ({glm_py!r}')
        from .models import resolve_load_mode
        from .glm_worker_client import GlmWorker
        load_mode, _, _ = resolve_load_mode(params)
        request = dict(model=model_dir, prompt=params.prompt, out=str(out), width=w, height=h, steps=steps, guidance=guidance, seed=seed, device=self._cuda_device(), load_mode=load_mode, ar_device='cpu', loras=[l.model_dump() for l in params.loras], image=getattr(params, 'image_path', None), fast_ar=fast_ar, emit_bin=emit_bin, ar_gguf=ar_gguf)
        t0 = time.time()
        runtime = str(self.settings.cosmos_windows_path)
        worker = getattr(self, '_glm_worker', None)
        if worker is not None and (worker.python != glm_py or worker.runtime != runtime or worker.process.poll() is not None):
            worker.close()
            worker = self._glm_worker = None
        if worker is None:
            self.unload()
            worker = self._glm_worker = GlmWorker(glm_py, runtime)
        try:
            receipt = worker.generate(request, timeout=int(os.environ.get('STUDIO_GLM_TIMEOUT', '1800')), on_step=on_step, should_cancel=should_cancel)
        except BaseException:
            self._glm_worker = None
            raise
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if request['image'] else 't2i', width=w, height=h, kind='image', seed=seed, engine=self.name, timings={'worker_pid': receipt['pid'], 'platform': sys.platform, 'load_mode': load_mode, 'cache_hit': receipt['cache_hit']})

    def _generate_sdxl(self, params, on_step=None, should_cancel=None) -> GenResult:
        from . import sdxl_diffusion
        import torch
        from .sdxl_control import preflight
        controls = preflight(params)
        ckpt = sdxl_diffusion.resolve_ckpt(params.model)
        loras = [l.model_dump() for l in params.loras or []]
        t0 = time.time()
        device = self._cuda_device()
        from .models import resolve_load_mode
        load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        direct_one_shot = bool(getattr(params, 'weight_stream', True))
        resident_mode = is_resident and (not direct_one_shot)
        offload = 'none'
        pipe_key = ('sdxl', ckpt, device, offload, load_mode, direct_one_shot, self._lora_cache_key(params.loras))
        if resident_mode and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None):
            pipe = self._resident_aux_pipe
            print(f'[SDXL] resident cache hit on {device}', flush=True)
        else:
            mode_label = 'resident' if resident_mode else 'direct-resident one-shot'
            print(f'[SDXL] building {mode_label} pipeline on {device}' + (f' with {len(loras)} LoRA(s)' if loras else ''), flush=True)
            self.unload()
            self._cleanup_torch()
            build_t0 = time.time()
            pipe = sdxl_diffusion.build_sdxl_pipeline(ckpt, device=device, dtype=torch.float16, offload=offload, loras=loras or None)
            print(f'[SDXL] pipeline built in {time.time() - build_t0:.2f}s', flush=True)
            if resident_mode:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'sdxl'
        try:
            w, h = (int(params.width or 1024), int(params.height or 1024))
            steps = int(params.steps or 28)
            guidance = float(params.guidance if params.guidance is not None else 6.0)
            seed = int(params.seed)
            clip_skip = params.sdxl_clip_skip if params.sdxl_clip_skip is not None else sdxl_diffusion.resolve_clip_skip(params.model_support_files)
            out = self._image_out_path('sdxl', params.prompt, seed, params)
            is_i2i = (params.modality or 't2i').lower() == 'i2i'
            if is_i2i and (not params.image_path):
                raise RuntimeError('SDXL i2i requires a source image (image_path).')
            sdxl_diffusion.generate_sdxl(pipe, prompt=params.prompt, negative=params.negative or '', width=w, height=h, steps=steps, guidance=guidance, seed=seed, clip_skip=clip_skip, prediction_type=params.sdxl_prediction_type or params.model_support_files.get('prediction_type'), sampler=params.sampler or '', karras=(params.sigmas or '').lower() == 'karras', out_path=str(out), image_path=params.image_path if is_i2i else None, strength=float(params.denoise) if params.denoise is not None else 0.6, two_pass=params.two_pass, first_pass_loras=loras, on_step=on_step, should_cancel=should_cancel, controls=controls)
        finally:
            if not resident_mode:
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if is_i2i else 't2i', width=w, height=h, kind='image', seed=seed, engine=self.name)

    @staticmethod
    def _krea2_native_loras(params, sf: dict, dit_override: str | None) -> list[dict]:
        loras = [l.model_dump() if hasattr(l, 'model_dump') else l for l in params.loras or []]
        if sf.get('_stack_explicit_loras_only'):return loras
        default_lora = str(sf.get('lora') or '').strip()
        variant = str(sf.get('variant') or '').strip().lower()
        if default_lora and (not dit_override or variant in ('turbo', 'oss_turbo')):
            default_lora = ''
        if default_lora:
            try:
                default_strength = float(sf.get('lora_strength', 0.6))
            except (TypeError, ValueError):
                default_strength = 0.6

            def _norm_lora(p):
                try:
                    return os.path.normcase(os.path.normpath(os.path.abspath(str(p).replace('\\', os.sep))))
                except Exception:
                    return str(p)

            def _path_of(item) -> str:
                if isinstance(item, dict):
                    return str(item.get('path') or '')
                return str(getattr(item, 'path', '') or '')
            if not any((_norm_lora(_path_of(d)) == _norm_lora(default_lora) for d in loras)):
                loras = [{'path': default_lora, 'weight': default_strength, 'name': 'krea2-turbo-lora'}] + loras
        return loras

    def _generate_krea2_native(self, params, on_step=None, should_cancel=None) -> GenResult:
        if params.krea_request_process:
            from .krea_process import generate_owned_krea
            return generate_owned_krea(self, params, on_step=on_step, should_cancel=should_cancel)
        import torch
        from . import krea2_diffusion
        from .models import resolve_load_mode
        if params.control_image or params.control_kind or (params.control_guides or []):
            raise RuntimeError('ControlNet conditioning is not supported by the native Krea2 pipeline.')
        if not krea2_diffusion.krea2_available():
            f = krea2_diffusion.krea2_files()
            raise RuntimeError(f"Krea2 int8 source files not found (dit={f['dit']!r}, text_encoder={f['text_encoder']!r}")
        t0 = time.time()
        device = self._cuda_device()
        _load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        offload = 'int8_resident' if is_resident else 'model'
        raw_variant = str(params.model_arch or '').lower() == 'krea2-raw'
        sf = dict(getattr(params, 'model_support_files', None) or {})
        dit_override = str(sf.get('dit') or '').strip() or None
        loras = self._krea2_native_loras(params, sf, dit_override)
        if params.krea2_edit:
            if not params.image_path:
                raise ValueError('Krea identity edit requires a primary image')
            if params.mask_path:
                raise ValueError('Krea identity edit does not support an inpaint mask')
            loras = krea2_diffusion.ensure_krea2_edit_lora(loras, explicit_path=params.krea2_edit_lora_path, dit_path=dit_override)
        cacheable = is_resident or self._stream_pipeline_cache_enabled()
        pipe_key = ('krea2-native', device, offload, dit_override or '') + (('raw',) if raw_variant else ())
        if cacheable and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None):
            pipe = self._resident_aux_pipe
            print(f"[krea2-native] {('resident' if is_resident else 'streamed setup')} cache hit on {device}", flush=True)
        else:
            print(f"[krea2-native] building {('resident-direct' if is_resident else 'streamed-block')} pipeline on {device}", flush=True)
            self.unload()
            self._cleanup_torch()
            build_kwargs = {'device': device, 'dtype': torch.bfloat16, 'offload': offload, 'dit_path': dit_override}
            if raw_variant:
                build_kwargs['distilled'] = False
            pipe = krea2_diffusion.build_krea2_pipeline(**build_kwargs)
            if cacheable:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'krea2-native' if is_resident else 'krea2-native-stream-cache'
        w, h = (int(params.width or 1024), int(params.height or 1024))
        steps = int(params.steps or 8)
        guidance = float(params.guidance if params.guidance is not None else 0.0)
        seed = int(params.seed)
        out = self._image_out_path(params.model_arch or 'krea2-native', params.prompt, seed, params)
        ok = False
        try:
            if params.krea2_edit:
                krea2_diffusion.generate_krea2_edit(pipe, prompt=params.prompt, image_path=str(params.image_path), reference_path=params.krea2_edit_reference_path, recipe=params.krea2_edit_recipe, width=w, height=h, steps=steps, guidance=guidance, seed=seed, grounding_px=params.krea2_edit_grounding_px, loras=loras, out_path=str(out), on_step=on_step, should_cancel=should_cancel)
                modality = 'i2i'
            elif params.image_path:
                strength = float(params.denoise) if params.denoise is not None else 0.6
                krea2_diffusion.generate_krea2_i2i(pipe, prompt=params.prompt, image_path=str(params.image_path), mask_path=getattr(params, 'mask_path', None) or None, negative=params.negative or '', width=w, height=h, steps=steps, guidance=guidance, strength=strength, seed=seed, loras=loras, tile_size=params.krea2_i2i_tile_size, tile_overlap=params.krea2_i2i_tile_overlap, out_path=str(out), on_step=on_step, should_cancel=should_cancel)
                modality = 'i2i'
            else:
                krea2_diffusion.generate_krea2(pipe, prompt=params.prompt, negative=params.negative or '', width=w, height=h, steps=steps, guidance=guidance, seed=seed, loras=[l.model_dump() for l in params.loras] if params.two_pass and params.two_pass.enabled else loras, two_pass=params.two_pass, baseline_loras=self._krea2_native_loras(params.model_copy(update={'loras': []}), sf, dit_override), out_path=str(out), on_step=on_step, should_cancel=should_cancel)
                modality = 't2i'
            ok = True
        finally:
            if not cacheable or not ok:
                try:
                    streamer = getattr(getattr(pipe, 'transformer', None), '_krea2_streamer', None)
                    close = getattr(streamer, 'close', None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                if self._resident_aux_pipe is pipe:
                    self._resident_aux_pipe = None
                    self._resident_aux_key = None
                    self._resident_aux_label = None
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality=modality, width=w, height=h, kind='image', seed=seed, engine=self.name)

    def generate_krea2_batch(self, params, prompts: list[str], on_event=None, should_cancel=None) -> dict:
        import torch
        from . import krea2_diffusion
        from .models import resolve_load_mode
        if params.control_image or params.control_kind or (params.control_guides or []):
            raise RuntimeError('ControlNet conditioning is not supported by the native Krea2 pipeline.')
        if getattr(params, 'image_path', None):
            raise RuntimeError('Krea2 batch currently supports t2i only; use single generation for i2i/inpaint.')
        if not krea2_diffusion.krea2_available():
            f = krea2_diffusion.krea2_files()
            raise RuntimeError(f"Krea2 int8 source files not found (dit={f['dit']!r}, text_encoder={f['text_encoder']!r}")
        clean_prompts = [str(p or '').strip() for p in prompts if str(p or '').strip()]
        if not clean_prompts:
            raise RuntimeError('Krea2 batch requires at least one prompt.')
        t0 = time.time()
        device = self._cuda_device()
        _load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        offload = 'int8_resident' if is_resident else 'model'
        sf = dict(getattr(params, 'model_support_files', None) or {})
        dit_override = str(sf.get('dit') or '').strip() or None
        cacheable = is_resident or self._stream_pipeline_cache_enabled()
        pipe_key = ('krea2-native', device, offload, dit_override or '')
        if cacheable and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None):
            pipe = self._resident_aux_pipe
            print(f"[krea2-native] {('resident' if is_resident else 'streamed setup')} batch cache hit on {device}", flush=True)
        else:
            print(f"[krea2-native] building {('resident-direct' if is_resident else 'streamed-block')} batch pipeline on {device} ({len(clean_prompts)} image(s))", flush=True)
            self.unload()
            self._cleanup_torch()
            pipe = krea2_diffusion.build_krea2_pipeline(device=device, dtype=torch.bfloat16, offload=offload, dit_path=dit_override)
            if cacheable:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'krea2-native' if is_resident else 'krea2-native-stream-cache'
        loras = self._krea2_native_loras(params, sf, dit_override)
        w, h = (int(params.width or 1024), int(params.height or 1024))
        steps = int(params.steps or 8)
        guidance = float(params.guidance if params.guidance is not None else 0.0)
        seed = int(params.seed)
        out_paths = [str(self._image_out_path('krea2-native-batch', prompt, seed + i, params)) for i, prompt in enumerate(clean_prompts)]

        def _step(step: int, total: int) -> None:
            if on_event:
                elapsed = max(0.0, time.time() - t0)
                per_step = elapsed / max(1, int(step))
                on_event({'type': 'step', 'step': int(step), 'total': int(total), 'seconds': round(per_step, 2), 'n': len(clean_prompts)})
        ok = False
        try:
            if on_event:
                on_event({'type': 'gen_status', 'text': f'starting Krea2 native batch ({len(clean_prompts)} image(s))'})
            paths = krea2_diffusion.generate_krea2_batch(pipe, prompts=clean_prompts, negative=params.negative or '', width=w, height=h, steps=steps, guidance=guidance, seed=seed, out_paths=out_paths, loras=loras, on_step=_step, should_cancel=should_cancel)
            for path in paths:
                if on_event:
                    on_event({'type': 'saved', 'path': path})
            ok = True
        finally:
            if not cacheable or not ok:
                try:
                    streamer = getattr(getattr(pipe, 'transformer', None), '_krea2_streamer', None)
                    close = getattr(streamer, 'close', None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                if self._resident_aux_pipe is pipe:
                    self._resident_aux_pipe = None
                    self._resident_aux_key = None
                    self._resident_aux_label = None
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
        seconds = round(time.time() - t0, 2)
        step_s = seconds / max(1, steps)
        throughput = f'{step_s:.1f}s/step for {len(paths)} imgs = {step_s / max(1, len(paths)):.2f}s/img/step'
        if on_event:
            on_event({'type': 'throughput', 'text': throughput})
        return {'path': paths[-1] if paths else '', 'paths': paths, 'seconds': seconds, 'throughput': throughput, 'modality': 't2i', 'width': w, 'height': h, 'seed': seed, 'engine': self.name}

    def _generate_boogu_krea(self, params, on_step=None, should_cancel=None) -> GenResult:
        import torch
        from . import boogu_krea
        from .models import resolve_load_mode
        arch = (params.model_arch or 'krea2').lower()
        if params.control_image or params.control_kind or (params.control_guides or []):
            raise RuntimeError(f'ControlNet conditioning is not supported by the {arch} pipeline.')
        ckpt = str(params.model or '')
        if not ckpt or not Path(ckpt).is_dir():
            raise RuntimeError(f'{arch} model directory not found: {ckpt!r}')
        t0 = time.time()
        device = self._cuda_device()
        _load_mode, is_resident, _is_lazy = resolve_load_mode(params)
        if not arch.startswith('boogu') and (not is_resident):
            mode = 'native_streaming'
        elif arch == 'boogu-turbo' and (not is_resident):
            mode = 'native_streaming'
        elif not arch.startswith('boogu') and os.environ.get('STUDIO_KREA2_DISABLE_STAGED', '0') not in ('1', 'true', 'True'):
            mode = 'staged' if is_resident else 'staged_disposable'
        else:
            mode = 'resident' if is_resident else os.environ.get('STUDIO_BOOGU_KREA_OFFLOAD', 'streaming')
        cacheable = is_resident or (mode in ('streaming', 'stream', 'streamed', 'native_streaming') and self._stream_pipeline_cache_enabled() and (not (arch.startswith('boogu') or arch.startswith('krea'))))
        pipe_key = ('boogu_krea', arch, ckpt, device, mode)
        if cacheable and self._resident_aux_key == pipe_key and (self._resident_aux_pipe is not None):
            pipe = self._resident_aux_pipe
            print(f"[boogu-krea] {('resident' if is_resident else 'streamed setup')} cache hit ({arch}) on {device}", flush=True)
        else:
            print(f'[boogu-krea] building {mode} {arch} pipeline on {device}', flush=True)
            self.unload()
            self._cleanup_torch()
            pipe = boogu_krea.build_pipeline(arch, ckpt, device=device, mode=mode, dtype=torch.bfloat16)
            if cacheable:
                self._resident_aux_pipe = pipe
                self._resident_aux_key = pipe_key
                self._resident_aux_label = 'boogu_krea' if is_resident else 'boogu_krea-stream-cache'
        w, h = (int(params.width or 1024), int(params.height or 1024))
        seed = int(params.seed)
        refs = None
        ref_paths = [str(p) for p in params.reference_images or []]
        if params.image_path:
            ref_paths = [str(params.image_path)] + ref_paths
        if ref_paths:
            seen_refs: set[str] = set()
            ref_paths = [p for p in ref_paths if not (os.path.normcase(os.path.abspath(p)) in seen_refs or seen_refs.add(os.path.normcase(os.path.abspath(p))))]
        if ref_paths:
            if not arch.startswith('boogu'):
                raise RuntimeError("the Krea-2 diffusers checkpoint is T2I-only here — use arch 'krea2-native' for i2i/mask, or a boogu-edit checkpoint for reference-image editing.")
            from PIL import Image
            refs = [Image.open(p).convert('RGB') for p in ref_paths]
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        out = self._image_out_path(arch, params.prompt, seed, params)
        ok = False
        try:
            lora_receipt = {}
            img = boogu_krea.generate(pipe, arch, prompt=params.prompt, negative=params.negative or '', width=w, height=h, steps=params.steps, guidance=params.guidance, seed=seed, device=device, reference_images=refs, on_step=on_step, loras=params.loras, receipt=lora_receipt, should_cancel=should_cancel, two_pass=getattr(params, 'two_pass', None))
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            img.save(str(out))
            krea_streamer = getattr(getattr(pipe, 'transformer', None), '_krea_hf_streamer', None)
            if krea_streamer:
                stream_receipt = {**krea_streamer.receipt(), 'requested_arch': arch, 'checkpoint': ckpt}
                Path(str(out) + '.streaming.json').write_text(json.dumps(stream_receipt, indent=2), encoding='utf-8')
            if lora_receipt.get('loras'):
                Path(str(out) + '.loras.json').write_text(json.dumps(lora_receipt, indent=2), encoding='utf-8')
            ok = True
        finally:
            if not cacheable or not ok:
                krea_cleanup_error = None
                transformer = None
                streamer = None
                try:
                    transformer = getattr(pipe, 'transformer', None)
                    streamer = getattr(transformer, '_boogu_streamer', None) or getattr(transformer, '_krea_hf_streamer', None)
                    close = getattr(streamer, 'close', None)
                    if callable(close):
                        close()
                except Exception as cleanup_error:
                    if streamer is not None and getattr(transformer, '_krea_hf_streamer', None) is streamer:
                        krea_cleanup_error = cleanup_error
                if self._resident_aux_pipe is pipe:
                    self._resident_aux_pipe = None
                    self._resident_aux_key = None
                    self._resident_aux_label = None
                try:
                    del pipe
                except Exception:
                    pass
                self._cleanup_torch()
                if krea_cleanup_error:
                    raise RuntimeError('Krea native streaming did not close its owned weights') from krea_cleanup_error
                if ok and krea_streamer:
                    Path(str(out) + '.streaming.json').write_text(json.dumps({**krea_streamer.receipt(), 'requested_arch': arch, 'checkpoint': ckpt}, indent=2), encoding='utf-8')
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='i2i' if refs else 't2i', width=w, height=h, kind='image', seed=seed, engine=self.name)

    def _generate_temporal_pipeline(self, params, on_step=None, should_cancel=None) -> GenResult:
        from . import temporal_video
        from .chat_artifacts import output_dir_for_params
        temporal_video.preflight(params)
        if params.weight_stream:
            raise ValueError('Complete temporal pipelines use resident loading; turn weight streaming off')
        self.unload()
        started = time.perf_counter()
        device = self._cuda_device()
        output, frames, receipt = temporal_video.generate(params, output_dir_for_params(params), device=device, on_step=on_step, should_cancel=should_cancel)
        return GenResult(path=str(output), seconds=round(time.perf_counter() - started, 3), modality='t2v', kind='video', width=params.width, height=params.height, seed=params.seed, engine=self.name, model=params.model, model_arch=params.model_arch, timings={'device': device, 'frames': frames, 'fps': params.fps or 16, 'loras': receipt['loras']})

    def _generate_wan(self, params, on_step=None, should_cancel=None) -> GenResult:
        import importlib
        import re
        if Path(str(params.model)).is_dir():
            return self._generate_temporal_pipeline(params, on_step, should_cancel)
        sf = dict(params.model_support_files or {})
        high = params.model
        if not high or not Path(high).is_file():
            raise RuntimeError(f"Wan model path not found: {high!r}. Register the Wan high-noise denoiser as the model path (and the low-noise denoiser under the 'low_denoiser' support-file slot).")
        modality = params.modality or 't2i'
        keyframes = sorted(params.keyframes or [], key=lambda k: k.frame if k.frame is not None else 10 ** 9)
        first_frame = params.image_path
        last_frame = params.image2_path
        for k in keyframes:
            if (k.role == 'first' or k.frame == 0) and (not first_frame):
                first_frame = k.image_path
            if (k.role == 'last' or k.frame == (params.num_frames or 1) - 1) and (not last_frame):
                last_frame = k.image_path
        if modality in ('i2v', 'fflf', 'i2i') and (not first_frame):
            raise RuntimeError(f'Wan {modality} generation requires a first frame image.')
        self.unload()
        self._cleanup_torch()
        from . import safety
        min_free = float(os.environ.get('STUDIO_WAN_MIN_FREE_RAM_GB', '6'))
        ok, free = safety.enough_ram(min_free)
        if not ok:
            raise RuntimeError(f'Not enough system RAM for streamed Wan: {free:.1f} GB free, need >= {min_free:.0f} GB. The streamed path does not need a large host-RAM staging buffer, but it still refuses to start below this safety floor.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        wan = importlib.import_module('cosmos_windows.weight_streaming.wan_backend')
        image_mode = modality in ('t2i', 'i2i')
        is_video = not image_mode and bool(params.num_frames and params.num_frames > 1) and (modality in ('t2v', 'i2v', 'fflf', 'multikeyframe', 'av'))
        nf = params.num_frames if is_video else 1
        wan_sdedit = modality == 'i2i' and bool(first_frame) and _sdedit_i2i_enabled('wan')
        t0 = time.time()
        scheduler_name = params.sampler or 'flowmatch'

        def progress(step, total, _latents=None):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(int(step), int(total), None)
        try:
            frames = wan.streamed_wan_generate(high_file=high, prompt=params.prompt, height=params.height, width=params.width, num_frames=nf, steps=params.steps or 30, low_file=sf.get('low_denoiser') or None, vae_path=sf.get('vae') or '', umt5_file=sf.get('text_encoder') or '', umt5_tokenizer_dir=sf.get('tokenizer') or None, negative_prompt=params.negative or '', guidance_scale=params.guidance if params.guidance is not None else 5.0, crossover_step=params.wan_crossover_step, boundary_ratio=params.wan_boundary_ratio, shift=params.wan_shift, scheduler_name=scheduler_name, sigmas_name=params.sigmas, seed=params.seed, lazy_denoisers=True, image_path=first_frame if first_frame and (modality in ('i2v', 'fflf', 'multikeyframe') or (modality == 'i2i' and (not wan_sdedit))) else None, init_image_path=first_frame if wan_sdedit else None, denoise_strength=_resolve_denoise_strength(params) if wan_sdedit else 1.0, last_image_path=last_frame if last_frame else None, keyframes=[k.model_dump() for k in keyframes], loras=[l.model_dump() for l in params.loras] or None if params.loras else None, progress=progress, should_cancel=should_cancel)
            seconds = round(time.time() - t0, 2)
            out_dir = output_dir_for_params(params)
            stem = 'wan_' + (re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'wan').strip())[:40].strip('_') or 'gen')
            stem = f'{stem}_{params.seed}'
            if is_video:
                from diffusers.utils import export_to_video
                out = out_dir / f'{stem}.mp4'
                export_to_video(frames, str(out), fps=params.fps or 24)
                kind = 'video'
            else:
                out = out_dir / f'{stem}.png'
                frames[0].save(str(out))
                kind = 'image'
            return GenResult(path=str(out), seconds=seconds, modality=modality if is_video or image_mode else 't2i', width=params.width, height=params.height, kind=kind, seed=params.seed, engine=self.name)
        finally:
            try:
                del frames
            except Exception:
                pass
            self._cleanup_torch()

    def _generate_bernini(self, params, on_step=None, should_cancel=None) -> GenResult:
        import importlib
        import re
        sf = dict(params.model_support_files or {})
        tf_dir = params.model
        if not tf_dir or not (Path(tf_dir).is_dir() or Path(tf_dir).is_file()):
            raise RuntimeError(f"Bernini model path not found: {tf_dir!r}. Register the Bernini transformer directory (diffusers 1.3B snapshot 'transformer/' dir) as the model path.")
        modality = params.modality or 't2i'
        refs = [p for p in getattr(params, 'reference_images', None) or [] if str(p).strip()]
        source_video = getattr(params, 'source_video', None) or None
        reference_video = getattr(params, 'reference_video', None) or None
        task = getattr(params, 'bernini_task', None) or modality
        if modality == 'i2i' and (not refs):
            _img = getattr(params, 'image_path', None)
            if _img and str(_img).strip():
                refs = [str(_img)]
        if source_video is None and modality in ('v2v', 'mv2v', 'rv2v'):
            _vid = getattr(params, 'video_path', None)
            if _vid and str(_vid).strip():
                source_video = str(_vid)
        self.unload()
        self._cleanup_torch()
        from . import safety
        ok, free = safety.enough_ram(float(os.environ.get('STUDIO_WAN_MIN_FREE_RAM_GB', '6')))
        if not ok:
            raise RuntimeError(f'Not enough system RAM for Bernini: {free:.1f} GB free.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        bb = importlib.import_module('cosmos_windows.weight_streaming.bernini_backend')
        is_video = bool(params.num_frames and params.num_frames > 1) and modality in ('t2v', 'i2v', 'v2v', 'r2v', 'rv2v', 'mv2v')
        nf = params.num_frames if is_video else 1
        t0 = time.time()
        umt5 = sf.get('text_encoder') or ''
        vae = sf.get('vae') or ''

        def _progress(step, total, _l=None):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(int(step), int(total), None)
        try:
            if Path(tf_dir).is_dir():
                frames = bb.bernini_generate_resident(str(tf_dir), params.prompt or '', task=task, reference_image_paths=refs or None, source_video_path=source_video, reference_video_path=reference_video, height=params.height, width=params.width, num_frames=nf, steps=params.steps or 40, guidance_scale=params.guidance if params.guidance is not None else 5.0, seed=params.seed, umt5_file=umt5, vae_path=vae, progress=_progress, should_cancel=should_cancel)
            else:
                wan = importlib.import_module('cosmos_windows.weight_streaming.wan_backend')

                def _progress(step, total, _l=None):
                    if should_cancel and should_cancel():
                        raise RuntimeError('generation cancelled')
                    if on_step:
                        on_step(int(step), int(total), None)
                user_loras = [l.model_dump() for l in params.loras] or None if params.loras else None
                distill = os.environ.get('STUDIO_BERNINI_DISTILL_LORA', '')
                use_distill = user_loras is None and Path(distill).is_file() and not sf.get('_stack_explicit_loras_only')
                loras = user_loras if user_loras is not None else [{'path': distill, 'weight': 3.0, 'expert': 'high'}, {'path': distill, 'weight': 1.5, 'expert': 'low'}] if use_distill else None
                has_distill = bool(user_loras) or use_distill
                frames = wan.streamed_wan_generate(high_file=str(tf_dir), prompt=params.prompt or '', height=params.height, width=params.width, num_frames=nf, steps=params.steps or (20 if has_distill else 40), low_file=sf.get('low_denoiser') or None, vae_path=vae, umt5_file=umt5, umt5_config_dir=sf.get('text_encoder_config') or None, umt5_tokenizer_dir=sf.get('tokenizer') or None, negative_prompt=params.negative or '', guidance_scale=params.guidance if params.guidance is not None else 1.0 if has_distill else 5.0, scheduler_name=params.sampler or ('lcm' if has_distill else 'unipc'), sigmas_name=params.sigmas if getattr(params, 'sigmas', None) not in (None, '', 'karras') else 'simple' if has_distill else None, seed=params.seed, crossover_step=params.wan_crossover_step if params.wan_crossover_step is not None else 3 if has_distill else None, shift=params.wan_shift if params.wan_shift is not None else 8.0, bernini_reference_images=refs or None, bernini_source_video=source_video, bernini_reference_video=reference_video, bernini_task=task, loras=loras, progress=_progress, should_cancel=should_cancel)
            seconds = round(time.time() - t0, 2)
            out_dir = output_dir_for_params(params)
            stem = 'bernini_' + (re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'gen').strip())[:40].strip('_') or 'gen')
            stem = f'{stem}_{params.seed}'
            if is_video:
                from diffusers.utils import export_to_video
                out = out_dir / f'{stem}.mp4'
                export_to_video(frames, str(out), fps=params.fps or 16)
                kind = 'video'
            else:
                out = out_dir / f'{stem}.png'
                frames[0].save(str(out))
                kind = 'image'
            return GenResult(path=str(out), seconds=seconds, modality=modality, width=params.width, height=params.height, kind=kind, seed=params.seed, engine=self.name)
        finally:
            try:
                del frames
            except Exception:
                pass
            self._cleanup_torch()

    def _generate_scail(self, params, on_step=None, should_cancel=None) -> GenResult:
        import importlib
        import re
        ref_image = params.image_path
        driving_video = getattr(params, 'video_path', None)
        if not ref_image:
            raise RuntimeError('SCAIL-2 requires a reference character image (image_path).')
        if not driving_video:
            raise RuntimeError('SCAIL-2 requires a driving / pose video (video_path).')
        scail_loras = [l for l in getattr(params, 'loras', None) or [] if getattr(l, 'path', '').strip()]
        if len(scail_loras) > 1:
            raise RuntimeError('SCAIL-2 currently supports one LoRA adapter on its native path; use Bernini RV2V for stacked CausVid/LightX2V-style LoRAs.')
        scail_lora_path = scail_loras[0].path if scail_loras else None
        scail_lora_alpha = scail_loras[0].weight if scail_loras else None
        modality = (params.modality or 'animate').lower()
        replace_flag = modality in ('replace', 'replacement', 'char-replace', 'rv2v')
        self.unload(free_scail=False)
        self._cleanup_torch()
        from . import safety
        ok, free = safety.enough_ram(float(os.environ.get('STUDIO_WAN_MIN_FREE_RAM_GB', '6')))
        if not ok:
            raise RuntimeError(f'Not enough system RAM for SCAIL-2: {free:.1f} GB free.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        sb = importlib.import_module('cosmos_windows.weight_streaming.scail_backend')

        def _progress(step, total, _l=None):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(int(step), int(total), None)
        t0 = time.time()
        frames = None
        try:
            _auto_shift = 3.0 if params.height and params.width and (max(params.height, params.width) <= 720) else 5.0
            frames = sb.scail2_animate(prompt=params.prompt or '', image_path=ref_image, pose_video_path=driving_video, replace_flag=replace_flag, target_h=params.height, target_w=params.width, segment_len=81, shift=params.wan_shift if getattr(params, 'wan_shift', None) is not None else _auto_shift, sample_solver=params.sampler or 'unipc', sampling_steps=params.steps or 40, guide_scale=params.guidance if params.guidance is not None else 5.0, negative_prompt=params.negative or None, seed=params.seed if params.seed is not None else -1, ckpt_dir=os.environ.get('SCAIL2_CKPT_DIR', ''), scail_safetensors=os.environ.get('SCAIL2_SAFETENSORS', ''), lora_path=scail_lora_path, lora_alpha=scail_lora_alpha, stream_weights=True, n_gpu_buffers=int(os.environ.get('STUDIO_SCAIL_GPU_BUFFERS', '3')), device_id=int(os.environ.get('STUDIO_SCAIL_DEVICE_ID', '0')), progress=_progress)
            seconds = round(time.time() - t0, 2)
            out_dir = output_dir_for_params(params)
            stem = 'scail_' + (re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'gen').strip())[:40].strip('_') or 'gen')
            out = out_dir / f'{stem}_{params.seed}.mp4'
            from diffusers.utils import export_to_video
            export_to_video(frames, str(out), fps=params.fps or 16)
            return GenResult(path=str(out), seconds=seconds, modality='replace' if replace_flag else 'animate', width=params.width, height=params.height, kind='video', seed=params.seed, engine=self.name)
        finally:
            try:
                del frames
            except Exception:
                pass
            self._cleanup_torch()

    def _generate_domainshuttle(self, params, on_step=None, should_cancel=None) -> GenResult:
        import importlib
        import re
        sf = dict(params.model_support_files or {})
        model_dir = params.model
        if not model_dir or not Path(model_dir).is_dir():
            raise RuntimeError(f'DomainShuttle model path not found: {model_dir!r}')
        refs = [str(p) for p in getattr(params, 'reference_images', None) or [] if str(p).strip()]
        if not refs and getattr(params, 'image_path', None):
            refs = [str(params.image_path)]
        if not refs:
            raise RuntimeError('DomainShuttle r2v requires at least one reference image (params.reference_images or params.image_path).')
        refs = refs[:5]
        domain_codes = list(getattr(params, 'reference_domains', None) or [])
        self.unload()
        self._cleanup_torch()
        from . import safety
        min_free = float(os.environ.get('STUDIO_DOMAINSHUTTLE_MIN_FREE_RAM_GB_GATE', '6'))
        ok, free = safety.enough_ram(min_free)
        if not ok:
            raise RuntimeError(f'Not enough system RAM for DomainShuttle: {free:.1f} GB free.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        db = importlib.import_module('cosmos_windows.weight_streaming.domainshuttle_backend')
        vae = sf.get('vae') or ''
        umt5 = sf.get('text_encoder') or ''
        tokenizer = sf.get('tokenizer') or ''
        num_frames = int(params.num_frames or 81)
        is_video = num_frames > 1
        _auto_shift = 5.0 if params.height and params.width and (max(params.height, params.width) <= 832) else 12.0
        guidance = (float(params.guidance), float(params.guidance)) if params.guidance is not None else (4.0, 3.0)

        def _progress(step, total, _l=None):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(int(step), int(total), None)
        t0 = time.time()
        frames = None
        try:
            frames = db.domainshuttle_r2v_generate(domain_model_dir=str(model_dir), reference_image_paths=refs, domain_codes=domain_codes or None, prompt=params.prompt or '', negative_prompt=params.negative or None, height=int(params.height or 480), width=int(params.width or 832), num_frames=num_frames, steps=int(params.steps or 40), guidance_scale=guidance, shift=params.wan_shift if getattr(params, 'wan_shift', None) is not None else _auto_shift, boundary=0.875, seed=params.seed if params.seed is not None else 42, vae_path=vae, umt5_path=umt5, tokenizer_dir=tokenizer, memory_mode=os.environ.get('STUDIO_DOMAINSHUTTLE_MEMORY_MODE', 'streaming'), device_id=0, progress=_progress, should_cancel=should_cancel)
            seconds = round(time.time() - t0, 2)
            out_dir = output_dir_for_params(params)
            stem = 'domainshuttle_' + (re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'gen').strip())[:40].strip('_') or 'gen')
            stem = f'{stem}_{params.seed}'
            if is_video:
                from diffusers.utils import export_to_video
                out = out_dir / f'{stem}.mp4'
                export_to_video(frames, str(out), fps=params.fps or 24)
                kind = 'video'
            else:
                out = out_dir / f'{stem}.png'
                frames[0].save(str(out))
                kind = 'image'
            return GenResult(path=str(out), seconds=seconds, modality='r2v', width=params.width, height=params.height, kind=kind, seed=params.seed, engine=self.name)
        finally:
            try:
                del frames
            except Exception:
                pass
            self._cleanup_torch()

    @staticmethod
    def _joyai_checkpoint_path(path: str | Path | None) -> str:
        p = Path(path or '')
        if p.is_file():
            return str(p)
        if p.is_dir():
            for name in ('JoyAI-Echo-release.safetensors', 'echo-longvideo-release.safetensors'):
                candidate = p / name
                if candidate.is_file():
                    return str(candidate)
            hits = sorted([x for x in p.glob('*.safetensors') if 'joy' in x.name.lower() or 'echo' in x.name.lower()], key=lambda x: x.stat().st_size, reverse=True)
            if hits:
                return str(hits[0])
        raise RuntimeError(f'JoyAI-Echo checkpoint not found at {path!r}. Register the .safetensors file directly, or a directory containing JoyAI-Echo-release.safetensors.')

    def _generate_joyai(self, params, on_step=None, should_cancel=None) -> GenResult:
        from .story_writer import normalize_story_writer_response, parse_prompts
        model_file = self._joyai_checkpoint_path(params.model)
        prompt_json = normalize_story_writer_response(params.prompt, fallback_idea=params.prompt)
        prompts = parse_prompts(prompt_json)
        shot_prompt = prompts[0] if prompts else params.prompt
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        ltx = importlib.import_module('cosmos_windows.weight_streaming.ltx_backend')
        proxy = params.model_copy(deep=True)
        proxy.model = model_file
        proxy.model_arch = 'ltx-2.3'
        proxy.modality = 'av' if params.modality == 'av' else 't2v'
        proxy.prompt = shot_prompt
        sf = dict(params.model_support_files or {})
        for key in ('text_encoder', 'tokenizer'):
            value = sf.get(key)
            if value:
                candidate = Path(value)
                if not (candidate.is_dir() and 'gemma' in str(candidate).lower()):
                    sf.pop(key, None)
        sf.setdefault('components_repo', getattr(ltx, 'LTX23_DEFAULT_COMPONENTS_REPO', ''))
        sf.setdefault('text_encoder', getattr(ltx, 'LTX23_DEFAULT_TEXT_ENCODER_DIR', ''))
        sf.setdefault('tokenizer', getattr(ltx, 'LTX23_DEFAULT_TEXT_ENCODER_DIR', ''))
        proxy.model_support_files = sf
        return self._generate_ltx(proxy, on_step=on_step, should_cancel=should_cancel)

    def _generate_ltx(self, params, on_step=None, should_cancel=None) -> GenResult:
        import importlib
        import re
        if Path(str(params.model)).is_dir():
            return self._generate_temporal_pipeline(params, on_step, should_cancel)
        model_file = params.model
        if not model_file or not Path(model_file).is_file():
            raise RuntimeError(f'LTX model path not found: {model_file!r}. Register the LTX 2.3 transformer safetensors as the model path.')
        self.unload()
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        ltx = importlib.import_module('cosmos_windows.weight_streaming.ltx_backend')
        sf = dict(params.model_support_files or {})
        components_repo = sf.get('components_repo') or getattr(ltx, 'LTX23_DEFAULT_COMPONENTS_REPO', '')
        text_encoder_dir = sf.get('text_encoder') or getattr(ltx, 'LTX23_DEFAULT_TEXT_ENCODER_DIR', None)
        tokenizer_dir = sf.get('tokenizer') or None
        loras = [l.model_dump() for l in params.loras or []]
        if self._is_lora_only_safetensors(model_file):
            lora_name = re.sub('[^0-9A-Za-z_]+', '_', Path(model_file).stem[:48]).strip('_') or 'ltx_adapter'
            loras.insert(0, {'path': str(model_file), 'weight': 1.0, 'name': lora_name})
            model_file = sf.get('base_model') or getattr(ltx, 'LTX23_DEFAULT_MODEL_FILE', '')
            if not model_file or not Path(model_file).is_file():
                raise RuntimeError('This LTX registration is a LoRA adapter, but no LTX base_model is available. Set the base_model support file to the LTX 2.3 transformer safetensors.')
        total_steps = int(params.steps or 30)
        modality = params.modality or 't2v'
        keyframes = sorted(params.keyframes or [], key=lambda k: k.frame if k.frame is not None else 10 ** 9)
        first_frame = params.image_path
        last_frame = params.image2_path
        for k in keyframes:
            if (k.role == 'first' or k.frame == 0) and (not first_frame):
                first_frame = k.image_path
            if (k.role == 'last' or k.frame == (params.num_frames or 1) - 1) and (not last_frame):
                last_frame = k.image_path
        if modality in ('i2v', 'fflf', 'i2i') and (not first_frame):
            raise RuntimeError(f'LTX {modality} generation requires a first frame image.')
        image_mode = modality in ('t2i', 'i2i')
        ltx_sdedit = modality == 'i2i' and bool(first_frame) and _sdedit_i2i_enabled('ltx')
        ltx_prompt, ltx_width, ltx_height, projection = _apply_ltx_360_recipe(params, sf, loras, params.prompt, params.width, params.height)

        def progress(step, total, _latents=None):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(int(step), int(total), None)
        t0 = time.time()
        out_obj = None
        try:
            out_obj = ltx.streamed_ltx_generate(model_file, ltx_prompt, components_repo=components_repo, negative_prompt=params.negative or 'worst quality, blurry, jittery, distorted, watermark, text', num_frames=1 if image_mode else params.num_frames or 97, height=ltx_height, width=ltx_width, num_inference_steps=total_steps, guidance_scale=params.guidance if params.guidance is not None else 4.0, audio_guidance_scale=params.guidance if params.guidance is not None else 4.0, frame_rate=float(params.fps or 24), seed=params.seed, n_gpu_buffers=3, decode_audio=params.modality == 'av' or params.audio_mode == 'condition', text_encoder_dir=text_encoder_dir, tokenizer_dir=tokenizer_dir, loras=loras or None, sampler_name=params.sampler, sigmas_name=params.sigmas, image_path=first_frame if first_frame and (modality in ('i2v', 'fflf', 'multikeyframe', 'av') or (modality == 'i2i' and (not ltx_sdedit))) else None, init_image_path=first_frame if ltx_sdedit else None, denoise_strength=_resolve_denoise_strength(params) if ltx_sdedit else None, last_image_path=last_frame if last_frame else None, keyframes=[k.model_dump() for k in keyframes], progress=progress)
            seconds = round(time.time() - t0, 2)
            frames = out_obj['video']
            if isinstance(frames, list) and frames and isinstance(frames[0], list):
                frames = frames[0]
            out_dir = output_dir_for_params(params)
            stem = 'ltx_' + (re.sub('[^A-Za-z0-9]+', '_', (params.prompt or 'ltx').strip())[:40].strip('_') or 'gen')
            if image_mode:
                out = out_dir / f'{stem}_{params.seed}.png'
                frames[0].save(str(out))
                kind = 'image'
            else:
                out = out_dir / f'{stem}_{params.seed}.mp4'
                from diffusers.utils import export_to_video
                export_to_video(frames, str(out), fps=params.fps or 24)
                kind = 'video'
            if not image_mode and out_obj.get('audio') is not None:
                audio_out = out.with_suffix('.wav')
                try:
                    _save_waveform_wav(out_obj['audio'], audio_out, int(out_obj.get('audio_sample_rate') or 24000))
                    out = mux_audio(out, audio_out)
                except Exception as exc:
                    print(f'  [studio] LTX audio mux skipped: {exc}', flush=True)
            if not image_mode and params.audio_path and (params.audio_mode == 'mux'):
                out = mux_audio(out, params.audio_path)
            return GenResult(path=str(out), seconds=seconds, modality='av' if modality == 'av' else modality, width=ltx_width, height=ltx_height, kind=kind, seed=params.seed, engine=self.name, projection=projection)
        finally:
            try:
                if isinstance(out_obj, dict):
                    out_obj.clear()
            except Exception:
                pass
            self._cleanup_torch()

    def foley(self, params, video_path: str, on_step=None, should_cancel=None) -> GenResult:
        import importlib
        import re
        self._resolve_registered_model(params)
        if not (params.model_arch or '').lower().startswith('ltx'):
            raise RuntimeError('Foley generation currently requires an LTX 2.3 model registration.')
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f'foley source video not found: {video_path}')
        self.unload()
        self._cleanup_torch()
        from . import safety
        min_free = float(os.environ.get('STUDIO_FOLEY_MIN_FREE_RAM_GB', '12'))
        ok, free = safety.enough_ram(min_free)
        if not ok:
            raise RuntimeError(f'Not enough free RAM for LTX foley: {free:.1f} GB free, need >= {min_free:.0f} GB. Close other apps and retry.')
        repo = str(getattr(self.settings, 'cosmos_windows_path', '') or '')
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        ltx = importlib.import_module('cosmos_windows.weight_streaming.ltx_backend')
        model_file = params.model
        sf = dict(params.model_support_files or {})
        components_repo = sf.get('components_repo') or getattr(ltx, 'LTX23_DEFAULT_COMPONENTS_REPO', '')
        text_encoder_dir = sf.get('text_encoder') or getattr(ltx, 'LTX23_DEFAULT_TEXT_ENCODER_DIR', None)
        tokenizer_dir = sf.get('tokenizer') or None
        loras = [l.model_dump() for l in params.loras or []]
        if self._is_lora_only_safetensors(model_file):
            lora_name = re.sub('[^0-9A-Za-z_]+', '_', Path(model_file).stem[:48]).strip('_') or 'ltx_adapter'
            loras.insert(0, {'path': str(model_file), 'weight': 1.0, 'name': lora_name})
            model_file = sf.get('base_model') or getattr(ltx, 'LTX23_DEFAULT_MODEL_FILE', '')
        if not model_file or not Path(model_file).is_file():
            raise RuntimeError(f'LTX model path not found for foley: {model_file!r}')

        def progress(step, total, _latents=None):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            if on_step:
                on_step(int(step), int(total), None)
        t0 = time.time()
        out_obj = ltx.streamed_ltx_generate(model_file, params.prompt, components_repo=components_repo, negative_prompt=params.negative or 'distorted, noisy, harsh, clipping, unrelated sounds', num_frames=max(1, int(params.num_frames or 17)), height=params.height, width=params.width, num_inference_steps=int(params.steps or 8), guidance_scale=params.guidance if params.guidance is not None else 4.0, audio_guidance_scale=params.guidance if params.guidance is not None else 4.0, frame_rate=float(params.fps or 24), seed=params.seed, n_gpu_buffers=3, decode_audio=True, decode_video=False, video_path=str(source), video_condition_strength=1.0, text_encoder_dir=text_encoder_dir, tokenizer_dir=tokenizer_dir, loras=loras or None, sampler_name=params.sampler, sigmas_name=params.sigmas, progress=progress)
        audio = out_obj.get('audio') if isinstance(out_obj, dict) else None
        if audio is None:
            raise RuntimeError('LTX foley completed without an audio waveform.')
        out_dir = output_dir_for_params(params)
        stem = 'foley_' + (re.sub('[^A-Za-z0-9]+', '_', (params.prompt or source.stem).strip())[:40].strip('_') or 'audio')
        stem = f'{stem}_{params.seed}'
        wav = out_dir / f'{stem}.wav'
        out = out_dir / f'{stem}.mp4'
        _save_waveform_wav(audio, wav, int(out_obj.get('audio_sample_rate') or 24000))
        out = mux_audio(source, wav, output_path=out)
        try:
            if isinstance(out_obj, dict):
                out_obj.clear()
        except Exception:
            pass
        self._cleanup_torch()
        return GenResult(path=str(out), seconds=round(time.time() - t0, 2), modality='foley', width=params.width, height=params.height, kind='video', seed=params.seed, engine=self.name)

def _vendor_root() -> Optional[str]:
    vendor = Path(__file__).resolve().parent.parent / 'vendor'
    return str(vendor) if (vendor / 'cosmos_windows').is_dir() else None

def _backend_roots(settings: Settings) -> list:
    roots = []
    if settings.cosmos_windows_path:
        roots.append((str(settings.cosmos_windows_path), 'STUDIO_COSMOS_WINDOWS_PATH'))
    v = _vendor_root()
    if v:
        roots.append((v, 'vendored backend/vendor'))
    return roots

def load_backend(settings: Settings) -> tuple[EngineBackend, bool, str]:
    if settings.force_mock:
        from .mock_backend import MockBackend
        return (MockBackend(settings), True, 'forced mock (STUDIO_FORCE_MOCK=1)')
    if sys.platform == 'darwin':
        try:
            import torch
            if not torch.backends.mps.is_available():
                raise RuntimeError('torch.backends.mps not available on this Mac')
            from .mac_backend import MacLocalBackend
            return (MacLocalBackend(settings), False, 'mac-local MPS backend (Apple Silicon)')
        except Exception as exc:
            detail = f'mac MPS backend unavailable: {type(exc).__name__}: {exc}'
            return (UnavailableBackend(settings, detail), False, detail)
    roots = _backend_roots(settings)
    source = 'site-packages'
    for path, label in reversed(roots):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    if roots:
        source = roots[0][1]
    try:
        importlib.import_module('torch')
    except Exception as exc:
        detail = f'real backend unavailable: torch is not importable in this daemon Python ({sys.executable}). Start the daemon with STUDIO_DAEMON_PYTHON/STUDIO_COSMOS_PYTHON pointing at the cosmos venv. {type(exc).__name__}: {exc}'
        return (UnavailableBackend(settings, detail), False, detail)
    try:
        cosmos_cli = importlib.import_module('cosmos_windows.cosmos_cli')
    except Exception as exc:
        hint = '' if roots else ' (set STUDIO_COSMOS_WINDOWS_PATH or vendor the backend)'
        detail = f'cosmos_windows unavailable{hint}: {type(exc).__name__}: {exc}'
        return (UnavailableBackend(settings, detail), False, detail)
    try:
        return (RealCosmosBackend(settings, cosmos_cli), False, f'cosmos_windows imported ({source})')
    except Exception as exc:
        detail = f'real backend init failed: {type(exc).__name__}: {exc}'
        return (UnavailableBackend(settings, detail), False, detail)
