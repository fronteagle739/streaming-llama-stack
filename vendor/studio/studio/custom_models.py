# Derived from Studio backend/studio/custom_models.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Optional
from .config import settings
_lock = threading.Lock()
ARCHITECTURES: dict = {'cosmos3-t2i': ['t2i', 'i2i'], 'cosmos3-i2v': ['i2v', 'v2v', 'fflf', 'multikeyframe'], 'cosmos3-t2v': ['t2v', 'i2v', 'v2v', 'fflf', 'multikeyframe'], 'cosmos3-av': ['av'], 'ltx-2.3': ['t2i', 'i2i', 't2v', 'i2v', 'fflf', 'multikeyframe', 'av'], 'wan-2.2': ['t2i', 'i2i', 't2v', 'i2v', 'fflf', 'multikeyframe'], 'bernini': ['t2i', 'i2i', 't2v', 'v2v', 'r2v', 'rv2v', 'mv2v'], 'scail-2': ['animate', 'replace'], 'domainshuttle': ['r2v'], 'minimax-h3': ['t2v', 'i2v', 'l2v', 'fflf', 'r2v', 'v2v', 'edit', 't2i', 'i2i', 't2a', 'ref2a'], 'magi-2': ['t2v', 'i2v'], 'anima-1.0': ['t2i', 'i2i'], 'z-image': ['t2i', 'i2i'], 'flux2-klein': ['t2i', 'i2i'], 'ultratex': ['i2i'], 'qwen-image-2.1': ['t2i', 'i2i'], 'moss-image': ['t2i', 'i2i'], 'sdxl': ['t2i', 'i2i'], 'krea2': ['t2i'], 'krea2-turbo': ['t2i'], 'krea2-raw': ['t2i'], 'krea2-native': ['t2i', 'i2i'], 'boogu': ['t2i', 'i2i'], 'boogu-turbo': ['t2i'], 'boogu-edit': ['i2i', 't2i'], 'glm-image': ['t2i', 'i2i'], 'hunyuan-3': ['t2i'], 'ideogram-4': ['t2i'], 'joyai-echo': ['t2v', 'i2v', 'fflf', 'multikeyframe', 'av'], 'triposplat': ['image-to-3d'], 'acestep': ['music'], 'yue2': ['music']}

def _comfy_models_root() -> Optional[Path]:
    candidates: list[Path] = []
    env = os.environ.get('STUDIO_COMFYUI_PATH') or os.environ.get('COMFYUI_PATH')
    if env:
        p = Path(env)
        candidates.append(p if p.name.lower() == 'models' else p / 'models')
    candidates += [Path(''), Path(''), Path('')]
    for p in candidates:
        try:
            if p.is_dir():
                return p
        except OSError:
            pass
    return None

def _first_file(root: Path, subdir: str, patterns: list[str]) -> Optional[str]:
    base = root / subdir
    if not base.is_dir():
        return None
    files = [p for p in base.rglob('*') if p.is_file()]
    for pat in patterns:
        pl = pat.lower()
        hit = next((p for p in files if pl in p.name.lower()), None)
        if hit:
            return str(hit)
    return None

def _first_dir(root: Path, subdir: str, patterns: list[str]) -> Optional[str]:
    base = root / subdir
    if not base.is_dir():
        return None
    dirs = [p for p in base.rglob('*') if p.is_dir()]
    for pat in patterns:
        pl = pat.lower()
        hit = next((p for p in dirs if pl in p.name.lower()), None)
        if hit:
            return str(hit)
    return None

def _first_file_any(root: Path, subdirs: list[str], patterns: list[str]) -> Optional[str]:
    for subdir in subdirs:
        hit = _first_file(root, subdir, patterns)
        if hit:
            return hit
    return None

def _official_wan_denoiser(role: str) -> Optional[str]:
    env = os.environ.get(f'STUDIO_WAN_{role.upper()}_DENOISER') or os.environ.get(f'WAN_FP8_REFERENCE_{role.upper()}')
    if env and Path(env).is_file():
        return env
    filename = f'wan2.2_t2v_{role}_noise_14B_fp16.safetensors'
    roots = [os.environ.get('STUDIO_WAN_OFFICIAL_ROOT'), '']
    for raw in roots:
        if not raw:
            continue
        p = Path(raw) / filename
        try:
            if p.is_file():
                return str(p)
        except OSError:
            pass
    return None

def comfy_support_defaults() -> dict[str, dict[str, str]]:
    root = _comfy_models_root()
    return _cached_scan(('comfy-defaults', str(root)), _path_sig(str(root)) if root else (), _comfy_support_defaults_uncached)

def _comfy_support_defaults_uncached() -> dict[str, dict[str, str]]:
    root = _comfy_models_root()
    if root is None:
        return {}
    umt5 = _first_file(root, 'text_encoders', ['umt5_xxl', 'umt5', 't5'])
    qwen = _first_file(root, 'text_encoders', ['qwen_3_8b', 'qwen3_8b', 'qwen_3_4b_fp8', 'qwen_3_4b', 'qwen'])
    zimage_qwen = _first_file(root, 'text_encoders', ['qwen_3_4b.safetensors', 'qwen_3_4b', 'qwen3_4b'])
    gemma_dir = _first_dir(root, 'text_encoders', ['gemma-3', 'gemma_3'])
    gemma = gemma_dir or _first_file(root, 'text_encoders', ['gemma_3', 'gemma-3', 'gemma'])
    ltx_base = _first_file(root, 'diffusion_models', ['ltx-2.3-22b', 'ltx-2.3', 'ltx23', 'ltx'])
    ltx_projection = _first_file(root, 'clip', ['ltx-2.3_text_projection', 'ltx_text_projection'])
    ltx_vae = _first_file(root, 'vae', ['ltx23_video_vae', 'ltx2_3', 'taeltx'])
    ltx_360_lora = _first_file(root, 'loras', ['360-equirectangular', '360_equirectangular'])
    wan_vae = _first_file(root, 'vae', ['wan_2.1_vae', 'wan'])
    wan_low = _official_wan_denoiser('low') or _first_file_any(root, ['checkpoints', 'diffusion_models', 'models'], ['wan2.2_t2v_low_noise_14b_fp16', 't2vlow', 'lowv', 'low'])
    qwen_vae = _first_file(root, 'vae', ['qwen_image_vae', 'qwen'])
    zimage_config = next((str(p) for p in (Path(os.environ.get('STUDIO_ZIMAGE_CONFIG_DIR', '')) if os.environ.get('STUDIO_ZIMAGE_CONFIG_DIR') else None, Path(''), Path('')) if p and p.is_dir()), None)
    zimage_vae = None
    if zimage_config:
        for candidate in (Path(zimage_config) / 'vae' / 'diffusion_pytorch_model.safetensors', Path(zimage_config) / 'vae'):
            if candidate.exists():
                zimage_vae = str(candidate)
                break
    hunyuan_vae = _first_file(root, 'vae', ['hunyuan'])
    cosmos_vae = _first_file(root, 'vae', ['cosmos'])
    anima_vae = _first_file(root, 'vae', ['anima'])
    joy_vae = _first_file(root, 'vae', ['joy', 'echo'])
    image_encoder = _first_file(root, 'clip_vision', ['clip_vision_h', 'clip_vision', 'siglip'])
    ltx_audio_vae = _first_file_any(root, ['checkpoints', 'vae', 'audio', 'audio_encoders'], ['ltx23_audio_vae', 'ltx_audio_vae'])
    wan_audio_vae = _first_file_any(root, ['checkpoints', 'vae', 'audio', 'audio_encoders'], ['wan_audio_vae', 'wan_audio'])
    joy_audio_vae = _first_file_any(root, ['checkpoints', 'vae', 'audio', 'audio_encoders'], ['joy_audio_vae', 'echo_audio_vae', 'joyai_audio'])
    cosmos_audio_vae = _first_file_any(root, ['checkpoints', 'vae', 'audio', 'audio_encoders'], ['cosmos_audio_vae', 'cosmos_audio'])
    audio_encoder = _first_file_any(root, ['audio', 'audio_encoders', 'checkpoints', 'LLM', 'text_encoders'], ['whisper', 'wav2vec', 'wavlm', 'clap', 'audio_encoder', 'speech_encoder', 'infinitetalk', 's2v'])
    audio_tokenizer = _first_file_any(root, ['audio', 'audio_tokenizers', 'checkpoints'], ['encodec', 'dac', 'snac', 'audio_tokenizer', 'codec'])
    vocoder = _first_file_any(root, ['audio', 'vocoders', 'checkpoints'], ['vocoder', 'bigvgan', 'hifigan', 'vocos'])
    anima_base = _first_file(root, 'diffusion_models', ['anima-base', 'anima-preview3-base', 'anima'])
    sdxl_vae = _first_file(root, 'vae', ['sdxl_vae', 'sdxl-vae', 'sdxl_fp16', 'sdxlvae'])
    glm_ar = next((str(p) for p in (Path(os.environ.get('STUDIO_GLM_AR_GGUF', '')) if os.environ.get('STUDIO_GLM_AR_GGUF') else None, Path(''), Path('')) if p and p.is_file()), None)
    scail_root = next((p for p in (Path(os.environ.get('STUDIO_DOMAINSHUTTLE_BASE_ROOT', '')) if os.environ.get('STUDIO_DOMAINSHUTTLE_BASE_ROOT') else None, Path('')) if p and p.is_dir()), None)
    scail_tokenizer = str(scail_root / 'umt5-xxl') if scail_root and (scail_root / 'umt5-xxl').is_dir() else None
    defaults: dict[str, dict[str, str]] = {}

    def add(arch: str, **paths: Optional[str]) -> None:
        clean = {k: v for k, v in paths.items() if v}
        if clean:
            defaults[arch] = clean
    add('cosmos3-t2i', text_encoder=qwen, vae=cosmos_vae)
    add('cosmos3-i2v', vision_encoder=image_encoder, clip_vision=image_encoder, vae=cosmos_vae)
    add('cosmos3-t2v', text_encoder=qwen, vae=cosmos_vae)
    add('cosmos3-av', text_encoder=qwen, vision_encoder=image_encoder, clip_vision=image_encoder, audio_encoder=audio_encoder, audio_tokenizer=audio_tokenizer, audio_vae=cosmos_audio_vae, vocoder=vocoder, vae=cosmos_vae)
    add('ltx-2.3', base_model=ltx_base, text_encoder=gemma or umt5, tokenizer=gemma_dir, vae=ltx_vae, text_projection=ltx_projection, clip_vision=image_encoder, audio_vae=ltx_audio_vae, lora_360=ltx_360_lora)
    add('wan-2.2', low_denoiser=wan_low, text_encoder=umt5, vae=wan_vae, tokenizer=scail_tokenizer, image_encoder=image_encoder, clip_vision=image_encoder)
    _bernini_low = _first_file_any(root, ['checkpoints', 'diffusion_models'], ['wan22_bernini_low', 'bernini_low'])
    _bernini_snap = next((str(p) for p in (Path(os.environ.get('STUDIO_BERNINI_SNAPSHOT', '')) if os.environ.get('STUDIO_BERNINI_SNAPSHOT') else None, Path('')) if p and p.is_dir()), None)
    add('bernini', text_encoder=umt5, vae=wan_vae, low_denoiser=_bernini_low, text_encoder_config=str(Path(_bernini_snap) / 'text_encoder') if _bernini_snap else None, tokenizer=str(Path(_bernini_snap) / 'tokenizer') if _bernini_snap else None)
    _ds_root = next((p for p in (Path(os.environ.get('STUDIO_DOMAINSHUTTLE_BASE_ROOT', '')) if os.environ.get('STUDIO_DOMAINSHUTTLE_BASE_ROOT') else None, scail_root) if p and p.is_dir()), None)
    if _ds_root is not None:
        _ds_vae = _ds_root / 'Wan2.1_VAE.pth'
        _ds_t5 = _ds_root / 'umt5-xxl' / 'models_t5_umt5-xxl-enc-bf16.pth'
        _ds_tok = _ds_root / 'umt5-xxl'
        add('domainshuttle', vae=str(_ds_vae) if _ds_vae.is_file() else None, text_encoder=str(_ds_t5) if _ds_t5.is_file() else None, tokenizer=str(_ds_tok) if _ds_tok.is_dir() else None)
    add('anima-1.0', base_model=anima_base, text_encoder=qwen, vae=anima_vae, clip_vision=image_encoder)
    add('z-image', config_dir=zimage_config, text_encoder=zimage_qwen, vae=zimage_vae, clip_vision=image_encoder)
    add('sdxl', vae=sdxl_vae)
    add('glm-image', ar_gguf=glm_ar)
    add('hunyuan-3', text_encoder=qwen, vae=hunyuan_vae, vision_encoder=image_encoder, clip_vision=image_encoder)
    add('joyai-echo', text_encoder=qwen or gemma, vae=joy_vae, clip_vision=image_encoder, audio_encoder=audio_encoder, audio_tokenizer=audio_tokenizer, audio_vae=joy_audio_vae, vocoder=vocoder)
    return defaults

def _path() -> Path:
    return settings.state_dir / 'studio_custom_models.json'

def _load() -> list:
    p = _path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError):
            return []
    return []

def _save(items: list) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2), encoding='utf-8')
    from . import settings_history
    settings_history.after_write('custom-models')

def _normalize_support_files(support_files: Optional[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (support_files or {}).items():
        if value is None or value == '':
            continue
        out[str(key)] = str(value)
    return out

def add(name: str, path: str, arch: str, label: Optional[str]=None, modalities: Optional[list]=None, note: Optional[str]=None, support_files: Optional[dict]=None) -> dict:
    with _lock:
        items = [m for m in _load() if m['name'] != name]
        entry = {'name': name, 'label': label or name, 'path': path, 'arch': arch, 'modalities': modalities or _modalities_for_model(arch, path), 'note': note, 'support_files': _normalize_support_files(support_files)}
        items.append(entry)
        _save(items)
        return entry

def _safetensors_header(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        hits = sorted(p.rglob('*.safetensors'), key=lambda x: x.stat().st_size, reverse=True)
        p = hits[0] if hits else p
    if not p.is_file() or p.suffix.lower() != '.safetensors':
        return {}
    try:
        with p.open('rb') as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return {}
            header_len = int.from_bytes(raw_len, 'little', signed=False)
            if header_len <= 0 or header_len > 32 * 1024 * 1024:
                return {}
            return json.loads(f.read(header_len).decode('utf-8'))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}

def _component_for_key(key: str) -> Optional[str]:
    if key.startswith(('model.diffusion_model.', 'diffusion_model.', 'net.')):
        return 'transformer'
    if key.startswith(('vae.', 'first_stage_model.')):
        return 'vae'
    if key.startswith(('text_encoders.', 'text_encoder.', 'conditioner.', 'cond_stage_model.')):
        return 'text_encoder'
    if key.startswith(('clip_vision.', 'image_encoder.', 'vision_encoder.')):
        return 'vision_encoder'
    if key.startswith(('audio_encoder.', 'audio_encoders.')):
        return 'audio_encoder'
    if key.startswith(('audio_vae.', 'vocoder.')):
        return key.split('.', 1)[0]
    return None

def _single_safetensors(path: str) -> Optional[Path]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == '.safetensors':
        return p
    if not p.is_dir():
        return None
    try:
        hits = sorted(p.rglob('*.safetensors'), key=lambda x: x.stat().st_size, reverse=True)
    except OSError:
        return None
    return hits[0] if hits else None

def _shape(header: dict, *keys: str) -> Optional[list]:
    for key in keys:
        value = header.get(key, {})
        shape = value.get('shape') if isinstance(value, dict) else None
        if isinstance(shape, list):
            return shape
    return None

def _arch_hint_from_safetensors(header: dict, keys: list[str]) -> Optional[str]:
    patch = _shape(header, 'patch_embedding.weight', 'model.diffusion_model.patch_embedding.weight', 'diffusion_model.patch_embedding.weight')
    if patch and len(patch) >= 2:
        try:
            out_channels = int(patch[0])
            in_channels = int(patch[1])
        except (TypeError, ValueError):
            out_channels = in_channels = 0
        if out_channels == 5120 and in_channels in (16, 36):
            return 'wan-2.2'
    if any((k.startswith(('model.diffusion_model.transformer_blocks.', 'diffusion_model.transformer_blocks.', 'transformer_blocks.')) for k in keys)):
        return 'ltx-2.3'
    if any(('.layers.' in f'.{k}' and ('.attention.qkv.' in k or '.x_embedder.' in k or 'all_x_embedder' in k) for k in keys)):
        return 'z-image'
    if any((k.startswith(('net.blocks.', 'model.diffusion_model.blocks.', 'diffusion_model.blocks.')) for k in keys)):
        joined = '\n'.join(keys[:2000])
        if 'adaln_modulation_cross_attn' in joined or 'llm_adapter' in joined:
            return 'anima-1.0'
    if any((k.startswith(('model.diffusion_model.input_blocks.', 'model.diffusion_model.label_emb.', 'conditioner.embedders.1.')) for k in keys)):
        if any(('label_emb' in k for k in keys)) or any((k.startswith('conditioner.embedders.1.') for k in keys)):
            return 'sdxl'
    return None
_SCAN_TTL_S = 30.0
_SCAN_CACHE: dict[tuple, tuple[float, tuple, object]] = {}
_SCAN_LOCK = threading.Lock()

def _path_sig(path: str) -> tuple:
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except (OSError, ValueError, TypeError):
        return ()

def _cached_scan(key: tuple, sig: tuple, compute):
    import copy
    import time
    now = time.monotonic()
    with _SCAN_LOCK:
        hit = _SCAN_CACHE.get(key)
    if hit is not None and hit[1] == sig and (now - hit[0] < _SCAN_TTL_S):
        return copy.deepcopy(hit[2])
    value = compute()
    with _SCAN_LOCK:
        _SCAN_CACHE[key] = (now, sig, value)
    return copy.deepcopy(value)

def checkpoint_metadata(path: str) -> dict:
    return _cached_scan(('ckpt', str(path)), _path_sig(path), lambda: _checkpoint_metadata_uncached(path))

def _checkpoint_metadata_uncached(path: str) -> dict:
    p = _single_safetensors(path)
    if p is None:
        return {}
    header = _safetensors_header(str(p))
    keys = [k for k in header.keys() if k != '__metadata__']
    if not keys:
        return {}
    components = sorted({c for k in keys if (c := _component_for_key(k))})
    comfyish = bool(components) or any((k.startswith(('model.', 'diffusion_model.', 'net.')) for k in keys))
    if not comfyish:
        return {'tensor_count': len(keys), 'checkpoint_file': str(p)}
    return {'checkpoint_format': 'comfy-single-file', 'checkpoint_components': components, 'arch_hint': _arch_hint_from_safetensors(header, keys), 'tensor_count': len(keys), 'checkpoint_file': str(p)}

def _wan_in_channels(path: str) -> Optional[int]:
    header = _safetensors_header(path)
    for key in ('patch_embedding.weight', 'model.diffusion_model.patch_embedding.weight', 'diffusion_model.patch_embedding.weight'):
        shape = header.get(key, {}).get('shape')
        if isinstance(shape, list) and len(shape) >= 2:
            try:
                return int(shape[1])
            except (TypeError, ValueError):
                return None
    return None

def _modalities_for_model(arch: Optional[str], path: str, saved: Optional[list]=None) -> list:
    if arch == 'wan-2.2':
        in_channels = _wan_in_channels(path)
        if in_channels == 16:
            return ['t2i', 't2v']
        if in_channels == 36:
            return ARCHITECTURES['wan-2.2']
    if arch in ARCHITECTURES:
        return ARCHITECTURES[arch]
    return saved or ['t2i']
