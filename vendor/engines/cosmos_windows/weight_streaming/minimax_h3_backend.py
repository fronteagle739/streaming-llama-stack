# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import math
import os as _os
import re
import sys
import time as _time
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
H3_TRANSFORMER_BLOCKS = 50
H3_DEFAULT_FPS = 24

def repack_h3_qkv_tensor(path, tensor, *, heads, head_dim):
    from .minimax_h3_fused import sectioned_qkv_to_interleaved
    if path.endswith('attn.qkv_proj.weight') or (path.endswith('attn.qkv_proj.weight_scale') and tensor.numel() > 1):
        return sectioned_qkv_to_interleaved(tensor, heads=heads, head_dim=head_dim)
    return tensor
_SOURCE_BLOCK_RE = re.compile('(?:^|/)blocks\\.(\\d+)\\.')
_COMFY_STATE_DICT_PREFIX = 'model.diffusion_model.'

def _strip_comfy_prefix(name: str) -> str:
    return name[len(_COMFY_STATE_DICT_PREFIX):] if name.startswith(_COMFY_STATE_DICT_PREFIX) else name
_DIFFUSERS_BLOCK_RE = re.compile('(?:^|/)transformer_blocks\\.(\\d+)\\.')

@dataclass(frozen=True)
class H3StreamingLayout:
    block_count: int
    block_ids: tuple[int, ...]
    missing_block_ids: tuple[int, ...]
    incomplete_block_ids: tuple[int, ...]
    block_bytes: tuple[int, ...]
    global_tensor_count: int
    global_bytes: int
    transformer_bytes_seen: int
    complete: bool
    checkpoint_naming: str

    @property
    def largest_block_bytes(self) -> int:
        return max(self.block_bytes, default=0)

@dataclass(frozen=True)
class H3HostCachePlan:
    budget_bytes: int
    cached_block_ids: tuple[int, ...]
    cached_bytes: int
    uncached_block_ids: tuple[int, ...]

def remap_uncached_h3_blocks(blocks: dict[int, list], cached_block_ids: Iterable[int]):
    cached = {int(block_id) for block_id in cached_block_ids}
    uncached = [block_id for block_id in sorted(blocks) if block_id not in cached]
    return ({logical: transport for transport, logical in enumerate(uncached)}, {transport: blocks[logical] for transport, logical in enumerate(uncached)})

class H3TieredBlockStreamer:

    def __init__(self, index, blocks: dict[int, list], plan: H3HostCachePlan, *, device='cuda', n_gpu_buffers=3, host_staging_buffers=2, ram_reserve_gb: float=4.0):
        import psutil
        import torch
        from .ds_backend import DirectStorageStreamer, _aln, _torch_dtype_for, enforce_ds_mode
        if plan.budget_bytes > 20000000000:
            raise ValueError('H3 host cache is limited to 20GB on the RTX backend')
        available = int(psutil.virtual_memory().available)
        reserve = int(ram_reserve_gb * 1000000000)
        if plan.cached_bytes > max(0, available - reserve):
            raise MemoryError(f'H3 host cache needs {plan.cached_bytes / 1000000000.0:.2f}GB but only {available / 1000000000.0:.2f}GB is available with {ram_reserve_gb:.2f}GB reserve')
        self.torch = torch
        self.device = device
        self.blocks = blocks
        self.plan = plan
        self.logical_to_transport, transport_groups = remap_uncached_h3_blocks(blocks, plan.cached_block_ids)
        if not transport_groups:
            raise ValueError('H3 cache cannot contain every block; leave at least one DirectStorage block')
        self.direct = DirectStorageStreamer(index, component='transformer', n_gpu_buffers=n_gpu_buffers, device=device, layer_groups=transport_groups, host_staging_buffers=host_staging_buffers)
        enforce_ds_mode(self.direct, 'MiniMax H3 uncached tier', 'MINIMAX_H3')
        packed: dict[int, tuple[list[tuple[object, int]], int]] = {}
        arena_bytes = 0
        largest = 0
        for block_id in plan.cached_block_ids:
            entries = []
            block_start = arena_bytes
            for ref in blocks[block_id]:
                entries.append((ref, arena_bytes))
                arena_bytes = _aln(arena_bytes + ref.nbytes)
            packed[block_id] = (entries, block_start)
            largest = max(largest, arena_bytes - block_start)
        self.host_arena = torch.empty(arena_bytes, dtype=torch.uint8, pin_memory=True)
        arena_view = memoryview(self.host_arena.numpy()).cast('B')
        handles = {}
        try:
            for entries, _ in packed.values():
                for ref, offset in entries:
                    handle = handles.setdefault(ref.file, open(ref.file, 'rb', buffering=0))
                    handle.seek(ref.offset)
                    got = handle.readinto(arena_view[offset:offset + ref.nbytes])
                    if got != ref.nbytes:
                        raise RuntimeError(f'short host-cache read for {ref.name}: {got} != {ref.nbytes}')
        finally:
            for handle in handles.values():
                handle.close()
        self.cached_layout = packed
        self.gpu_cache_buffer = torch.empty(largest, dtype=torch.uint8, device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.copy_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self._cached_in_flight = False
        self._dtype = _torch_dtype_for

    def new_pass(self, reverse: bool=False):
        self.direct.new_pass(reverse=reverse)

    def wait_layer(self, logical_block_id: int):
        transport_id = self.logical_to_transport.get(logical_block_id)
        if transport_id is not None:
            return self.direct.wait_layer(transport_id)
        entries, block_start = self.cached_layout[logical_block_id]
        block_bytes = max((offset + ref.nbytes for ref, offset in entries), default=block_start) - block_start
        if self._cached_in_flight:
            self.copy_stream.wait_event(self.compute_done)
        with self.torch.cuda.stream(self.copy_stream):
            self.gpu_cache_buffer[:block_bytes].copy_(self.host_arena[block_start:block_start + block_bytes], non_blocking=True)
            self.copy_done.record(self.copy_stream)
        self.torch.cuda.current_stream(self.device).wait_event(self.copy_done)
        self._cached_in_flight = True
        return {ref.name: self.gpu_cache_buffer[offset - block_start:offset - block_start + ref.nbytes].view(self._dtype(ref.dtype)).reshape(ref.shape) for ref, offset in entries}

    def release_layer(self, logical_block_id: int):
        transport_id = self.logical_to_transport.get(logical_block_id)
        if transport_id is not None:
            self.direct.release_layer(transport_id)
        else:
            self.compute_done.record(self.torch.cuda.current_stream(self.device))

    def close(self):
        self.direct.close()

@dataclass(frozen=True)
class H3GenerationRequest:
    model_dir: Path
    prompt: str
    mode: str
    out: Path
    image: Path | None = None
    last_frame: Path | None = None
    reference_images: tuple[Path, ...] = ()
    source_videos: tuple[Path, ...] = ()
    reference_audio: tuple[Path, ...] = ()
    frame_indices: tuple[int, ...] = ()
    loras: tuple[tuple[Path, float], ...] = ()
    duration: float = 5.0
    width: int = 1344
    height: int = 768
    steps: int = 50
    seed: int = 42
    gpu: int = 0
    host_cache_gb: float = 0.0
    resident: bool = False
    attention_backend: str = 'sdpa'
    reference_resolution: int | None = None
    seamless: bool = False
    extend_from: Path | None = None
    sparse_attention: str = 'auto'
    sparse_method: str = 'sol-attn'
    sparse_tau: float = 1.3
    sparse_keep_percent: float = 10.0
    sparse_start_percent: float = 0.2
    sparse_end_percent: float = 1.0
    sparse_min_tokens: int = 12288
    sparse_extra_tokens: int = 256
    sparse_sink_conditioning: str = 'exact_kv_and_rows'

    def sparse_config(self):
        from .minimax_h3_sparse import H3SparseAttentionConfig
        return H3SparseAttentionConfig(mode=str(self.sparse_attention), method=str(self.sparse_method), tau=float(self.sparse_tau), keep_percent=float(self.sparse_keep_percent), start_percent=float(self.sparse_start_percent), end_percent=float(self.sparse_end_percent), min_tokens=int(self.sparse_min_tokens), extra_tokens=int(self.sparse_extra_tokens), sink_conditioning=str(self.sparse_sink_conditioning))

def h3_dense_attention_bound(*, duration: float, width: int, height: int, steps: int, fps: int=H3_DEFAULT_FPS, inner_dim: int=7168) -> dict:
    if duration <= 0 or width <= 0 or height <= 0 or (steps <= 0) or (fps <= 0):
        raise ValueError('duration, dimensions, steps, and fps must be positive')
    requested_frames = max(1, round(duration * fps))
    frames = requested_frames + (5 - requested_frames) % 17
    latent_frames = 2 if frames <= 5 else (frames - 5) // 17 * 5 + 2
    token_width = math.ceil(width / 32)
    token_height = math.ceil(height / 32)
    video_tokens = latent_frames * token_width * token_height
    attention_flops_per_block = 4 * video_tokens * video_tokens * inner_dim
    total = attention_flops_per_block * H3_TRANSFORMER_BLOCKS * steps
    return {'kind': 'dense_attention_only_lower_bound', 'fps': fps, 'frames': frames, 'latent_frames': latent_frames, 'spatial_tokens_per_frame': token_width * token_height, 'video_tokens': video_tokens, 'attention_flops_per_block': attention_flops_per_block, 'attention_flops_total': total, 'excludes': ['text_tokens', 'audio_tokens', 'projections', 'mlp', 'vae', 'text_encoder', 'io']}

def _parse_lora(value: str) -> tuple[Path, float]:
    try:
        path, raw_strength = value.rsplit(':', 1)
        strength = float(raw_strength)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('LoRA must be PATH:STRENGTH') from exc
    if not path or not math.isfinite(strength):
        raise argparse.ArgumentTypeError('LoRA path must be non-empty and strength finite')
    return (Path(path), strength)
from .minimax_h3_attention import SUPPORTED_H3_ATTENTION_BACKENDS

def validate_h3_generation_request(request: H3GenerationRequest, *, check_files: bool=True) -> list[str]:
    errors: list[str] = []
    if request.attention_backend not in SUPPORTED_H3_ATTENTION_BACKENDS:
        errors.append('attention backend must be one of ' + ', '.join(SUPPORTED_H3_ATTENTION_BACKENDS))
    errors.extend(request.sparse_config().validate())
    if request.mode == 't2v':
        if request.image or request.last_frame or request.frame_indices:
            errors.append('t2v does not accept keyframes; use i2v, l2v or fflf')
    elif request.mode == 'i2v':
        if request.image is None:
            errors.append('i2v requires --image')
        if request.last_frame:
            errors.append('i2v does not accept --last-frame; use fflf')
        if request.frame_indices != (0,):
            errors.append('i2v requires exactly --frame-index 0')
    elif request.mode == 'l2v':
        if request.last_frame is None:
            errors.append('l2v requires --last-frame')
        if request.image:
            errors.append('l2v does not accept --image')
        if request.frame_indices != (-1,):
            errors.append('l2v requires exactly --frame-index -1')
    elif request.mode in ('flf2v', 'fflf'):
        if request.image is None or request.last_frame is None:
            errors.append('fflf requires both --image and --last-frame')
        if request.frame_indices != (0, -1):
            errors.append('fflf requires --frame-index 0 --frame-index -1')
    elif request.mode == 'ref2v':
        if not request.reference_images and (not request.reference_audio) and (not request.source_videos):
            errors.append('ref2v requires reference image, video, and/or audio media')
        if request.frame_indices and request.frame_indices not in ((0,), (-1,), (0, -1)):
            errors.append('keyframe indices must be 0, -1, or 0 and -1')
        if request.frame_indices and (0 in request.frame_indices and request.image is None or (-1 in request.frame_indices and request.last_frame is None)):
            errors.append('keyframe indices need the matching --image / --last-frame')
    elif request.mode == 'v2v':
        if not request.source_videos:
            errors.append('v2v requires --source-video')
        if request.image or request.last_frame or request.frame_indices:
            errors.append('v2v does not accept keyframes')
    else:
        errors.append(f'unsupported H3 mode: {request.mode}')
    if len(request.reference_images) > 9:
        errors.append(f'at most 9 reference images are supported; got {len(request.reference_images)}')
    if len(request.source_videos) > 3:
        errors.append(f'at most 3 reference videos are supported; got {len(request.source_videos)}')
    if len(request.reference_audio) > 3:
        errors.append(f'at most 3 reference audios are supported; got {len(request.reference_audio)}')
    _max_dur = float(os.environ.get('H3_MAX_DURATION', '600'))
    if request.duration <= 0:
        errors.append(f'duration must be positive (got {request.duration:.3f}s)')
    elif request.duration > _max_dur:
        errors.append(f'duration {request.duration:.1f}s exceeds H3_MAX_DURATION={_max_dur:.0f}s; raise that env var if the run is intended')
    if request.width <= 0 or request.height <= 0 or request.width % 32 or request.height % 32:
        errors.append('dimensions must be positive multiples of 32')
    max_pixels = int(os.environ.get('H3_MAX_PIXELS', str(8192 * 8192)))
    if request.width * request.height > max_pixels:
        errors.append(f'canvas exceeds H3_MAX_PIXELS={max_pixels}; raise that env var for an intended experiment')
    if request.steps < 1:
        errors.append('steps must be at least 1')
    if request.seed < 0:
        errors.append('seed must be non-negative')
    if request.gpu < 0:
        errors.append('gpu must be non-negative')
    if not 0 <= request.host_cache_gb <= 20:
        errors.append('host cache must be between 0 and 20GB')
    if request.reference_resolution is not None and (not 256 <= request.reference_resolution <= 4096):
        errors.append('reference resolution must be between 256 and 4096 (short edge, px)')
    if not request.prompt.strip():
        errors.append('prompt must not be empty')
    if check_files:
        candidates = [request.model_dir, request.image, request.last_frame, *request.reference_images, *request.source_videos, *request.reference_audio, *(entry[0] for entry in request.loras)]
        for path in candidates:
            if path is not None and (not path.exists()):
                errors.append(f'path does not exist: {path}')
    return errors

def _emit_json_event(enabled: bool, event: str, **payload) -> None:
    if enabled:
        print(json.dumps({'event': event, 'time': time.time(), **payload}), flush=True)

def _encode_h3_conditioning(request: H3GenerationRequest, geometry, *, root: Path, device: str, event_callback, frame_tensor=None, quantize_frame: bool=False, projection_video: bool=False, projection_alpha: float=1.0, carried_latent=None, source_video_overrides=None) -> dict:
    from .minimax_h3_vae_mux import encode_h3_keyframes, encode_h3_reference_audio, encode_h3_reference_visuals
    out: dict = {}
    if frame_tensor is None and carried_latent is not None:
        if request.image is not None:
            raise ValueError('Latent continuity cannot also specify a first-frame keyframe')
        from .minimax_h3_live import _latent_continuity_rows
        carried_rows = _latent_continuity_rows(carried_latent)
        if request.last_frame is None:
            out.update(condition_video_rows=carried_rows, frame_indices=(0,))
        else:
            import torch
            destination = encode_h3_keyframes(root, (request.last_frame,), geometry, frame_indices=(-1,), seed=42, device=device, event_callback=event_callback).rows
            carried_rows = carried_rows.to(device=destination.device, dtype=destination.dtype)
            if carried_rows.shape[1] != destination.shape[1]:
                raise ValueError(f'carried rows are {tuple(carried_rows.shape)} and the pinned last frame is {tuple(destination.shape)}; their feature widths must match')
            out.update(condition_video_rows=torch.cat([carried_rows, destination], dim=0), frame_indices=(0, -1))
    if frame_tensor is not None:
        if not 0 < projection_alpha <= 1:
            raise ValueError('projection_alpha must be in (0, 1]')
        if projection_alpha != 1 and carried_latent is None:
            raise ValueError('Blended projection requires a sampled carry')
        if request.image is not None or request.last_frame is not None:
            raise ValueError('Tensor continuity cannot also specify file keyframes')
        from .minimax_h3_vae_mux import encode_h3_frame_tensor
        out.update(condition_video_rows=encode_h3_frame_tensor(root, frame_tensor, geometry, quantize=quantize_frame, video=projection_video, device=device, event_callback=event_callback), frame_indices=(0,))
        if projection_alpha != 1:
            from .minimax_h3_live import _latent_continuity_rows
            carried = _latent_continuity_rows(carried_latent)
            projected = out['condition_video_rows']
            carried = carried.to(device=projected.device, dtype=projected.dtype)
            if carried.shape != projected.shape:
                raise ValueError('Projected and carried conditioning shapes differ')
            out['condition_video_rows'] = projection_alpha * projected + (1 - projection_alpha) * carried
    kf_images, frame_indices = ([], [])
    if request.image is not None:
        kf_images.append(request.image)
        frame_indices.append(0)
    if request.last_frame is not None:
        kf_images.append(request.last_frame)
        frame_indices.append(-1)
    if kf_images:
        condition = encode_h3_keyframes(root, tuple(kf_images), geometry, frame_indices=tuple(frame_indices), seed=42, device=device, event_callback=event_callback)
        out['condition_video_rows'] = condition.rows
        out['frame_indices'] = tuple(frame_indices)
    has_references = bool(request.reference_images or request.reference_audio or request.source_videos)
    if not has_references:
        return out
    video_media = tuple(({'path': video, 'kind': 'video', 'condition_index': index, 'material_chain': 'video.reference_preserve', **((source_video_overrides or {}).get(str(video)) or {})} for index, video in enumerate(request.source_videos)))
    visual_media = video_media + tuple(request.reference_images)
    visual = encode_h3_reference_visuals(root, visual_media, geometry, seed=42, device=device, reference_resolution=request.reference_resolution, event_callback=event_callback) if visual_media else None
    audio = encode_h3_reference_audio(root, request.reference_audio, max_duration_seconds=max(15.0, float(request.duration) + 1.0), device=device, event_callback=event_callback) if request.reference_audio else None
    ref_blocks = [] if visual is None else list(visual.ref_blocks)
    if audio is not None:
        ref_blocks.extend(({'kind': 'audio', 'ref_audio_t': item.ref_audio_t, 'condition_index': item.condition_index, 'material_chain': item.material_chain} for item in audio.items))
    out.update({'reference_video_rows': None if visual is None else visual.rows, 'reference_audio_rows': None if audio is None else audio.rows, 'ref_blocks': ref_blocks})
    return out

def h3_torch_memory_fraction(total_bytes: int, free_bytes: int, staging_bytes: int, requested: float | None=None) -> float:
    if total_bytes <= 0 or free_bytes <= staging_bytes:
        raise RuntimeError('H3 has insufficient free dedicated VRAM for DirectStorage staging')
    headroom = min(0.97, (free_bytes - staging_bytes) / total_bytes)
    if requested is not None and (not 0 < requested <= 1):
        raise ValueError('H3_TORCH_MEM_FRACTION must be positive and at most one')
    return min(headroom, requested) if requested is not None else headroom

def h3_dedicated_free_bytes(torch_module, device: int=0) -> int | None:
    if sys.platform != 'win32':
        return None
    import subprocess
    from uuid import UUID
    identity = getattr(torch_module.cuda.get_device_properties(device), 'uuid', None)
    try:
        raw_bytes = getattr(identity, 'bytes', None)
        if raw_bytes is not None:
            gpu_uuid = 'GPU-' + str(UUID(bytes=bytes(raw_bytes)))
        else:
            text = str(identity or '').strip()
            if text.lower().startswith('gpu-'):
                text = text[4:]
            gpu_uuid = 'GPU-' + str(UUID(text))
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise RuntimeError('H3 cannot identify its CUDA GPU for dedicated VRAM admission') from exc
    try:
        output = subprocess.check_output(['nvidia-smi', '--id=' + gpu_uuid, '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5)
        fields = output.strip().split(',')
        if len(fields) != 2 or fields[0].strip().lower() != gpu_uuid.lower():
            raise ValueError('NVIDIA returned a different GPU')
        free = int(fields[1].strip()) * 1024 ** 2
        if free <= 0:
            raise ValueError('no dedicated VRAM available')
        return free
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError('H3 dedicated VRAM query failed; no model loaded') from exc

def generate_h3(request: H3GenerationRequest, *, json_events: bool=False) -> dict:
    errors = validate_h3_generation_request(request)
    if errors:
        raise ValueError('; '.join(errors))
    _emit_json_event(json_events, 'request_validated', mode=request.mode, output=str(request.out), lora_count=len(request.loras), attention_backend=request.attention_backend)
    if min(request.width, request.height) > 768:
        _emit_json_event(json_events, 'warning', width=request.width, height=request.height, message='Requested canvas exceeds the released 768px training short edge; quality and available memory require measurement. No downscale applied.')
    try:
        import os as _os
        import torch as _torch
        if _torch.cuda.is_available():
            _free, _total = _torch.cuda.mem_get_info(0)
            _cuda_free = _free
            _dedicated_free = h3_dedicated_free_bytes(_torch, 0)
            if _dedicated_free is not None:
                _free = min(_free, _dedicated_free)
            _stage = int(float(_os.environ.get('LLAMA_DS_STAGING_MB', '1300')) * 1024 ** 2) + 1024 ** 3
            _requested = _os.environ.get('H3_TORCH_MEM_FRACTION')
            _frac = h3_torch_memory_fraction(_total, _free, _stage, float(_requested) if _requested is not None else None)
            _torch.cuda.set_per_process_memory_fraction(_frac, 0)
            _emit_json_event(json_events, 'vram_reserve', fraction=_frac, torch_cap_gb=round(_torch.cuda.get_device_properties(0).total_memory / 1000000000.0 * _frac, 2), free_bytes=_free, total_bytes=_total, staging_reserve_bytes=_stage, cuda_free_bytes=_cuda_free, dedicated_free_bytes=_dedicated_free, message='bounded torch VRAM by CUDA and dedicated headroom, with DirectStorage staging and slack reserved')
    except ImportError:
        pass
    _warn_at = float(os.environ.get('H3_WARN_DURATION', '15'))
    if request.duration > _warn_at:
        warning = f'H3 duration {request.duration:.1f}s is beyond the released {_warn_at:.0f}s spec — quality past this point is unverified here, not blocked'
        _emit_json_event(json_events, 'warning', message=warning, duration=request.duration)
        if not json_events:
            print(f'WARNING: {warning}', file=sys.stderr, flush=True)
    from .minimax_h3_attention import require_h3_attention_backend
    attention_capability = require_h3_attention_backend(request.attention_backend)
    bound = h3_dense_attention_bound(duration=request.duration, width=request.width, height=request.height, steps=request.steps)
    _emit_json_event(json_events, 'performance_bound', **bound)
    _emit_json_event(json_events, 'attention_backend_selected', **attention_capability.as_dict())
    from .minimax_h3_sparse import H3SparseAttentionUnavailable, detect_h3_sparse_attention
    sparse_config = request.sparse_config()
    sparse_capability = detect_h3_sparse_attention(f'cuda:{request.gpu}')
    if sparse_config.mode == 'on' and (not sparse_capability.available):
        raise H3SparseAttentionUnavailable(sparse_capability)
    _emit_json_event(json_events, 'sparse_attention_capability', mode=sparse_config.mode, method=sparse_config.method, **sparse_capability.as_dict())
    if sparse_config.mode == 'auto' and (not sparse_capability.available) and (not json_events):
        print(f'WARNING: H3 sparse attention unavailable, running dense: {sparse_capability.reason}', file=sys.stderr, flush=True)
    layout = inspect_h3_transformer(request.model_dir)
    _emit_json_event(json_events, 'checkpoint_inspected', **asdict(layout))
    if not layout.complete:
        raise RuntimeError(f'H3 checkpoint incomplete: {layout.block_count}/50 blocks; missing={list(layout.missing_block_ids)} partial={list(layout.incomplete_block_ids)}')
    device = f'cuda:{request.gpu}'
    root = request.model_dir
    text_encoder_dir = root / 'text_encoder'
    if not text_encoder_dir.is_dir():
        raise RuntimeError(f'H3 text encoder directory is missing: {text_encoder_dir}')
    video_vae_dir = root / 'video_vae'
    if not video_vae_dir.is_dir():
        raise RuntimeError(f'H3 video VAE directory is missing: {video_vae_dir}')
    required_vae_weights = (video_vae_dir / 'source' / 'model.safetensors', root / 'audio_vae' / 'model.safetensors')
    incomplete_vae = [str(path) for path in required_vae_weights if not path.is_file() or path.stat().st_size == 0]
    if incomplete_vae:
        raise RuntimeError('H3 VAE weights are incomplete: ' + ', '.join(incomplete_vae))
    from .minimax_h3_pipeline import denoise_h3_t2v, h3_latent_geometry
    from .minimax_h3_text_encoder import encode_h3_text_prompt
    from .minimax_h3_vae_mux import decode_and_mux_h3
    callback = lambda event, payload: _emit_json_event(json_events, event, **payload)
    _emit_json_event(json_events, 'text_encoding_started', device=device)
    prompt_embeddings = encode_h3_text_prompt(text_encoder_dir, request.prompt, device=device, event_callback=callback)
    geometry = h3_latent_geometry(request.duration, request.width, request.height)
    conditioning = _encode_h3_conditioning(request, geometry, root=root, device=device, event_callback=callback)
    extension = None
    if request.extend_from is not None:
        from .minimax_h3_extension import load_context, extension_prefix
        if request.image is not None:
            raise ValueError('H3 extension cannot also pin a first frame')
        extension = extension_prefix(*load_context(request.extend_from, width=request.width, height=request.height, device=device))
        callback('extension_context', {'method': 'masked_latent', 'frames': 39, 'source': str(request.extend_from)})
    _emit_json_event(json_events, 'denoise_started', device=device)
    video_latent, audio_latent, geometry = denoise_h3_t2v(root, prompt_embeddings, duration=request.duration, width=request.width, height=request.height, steps=request.steps, seed=request.seed, device=device, host_cache_gb=request.host_cache_gb, resident=request.resident, loras=request.loras, attention_backend=request.attention_backend, event_callback=callback, sparse_attention=sparse_config, extension_context=extension, **conditioning)
    result = decode_and_mux_h3(root, video_latent, audio_latent, geometry, request.out, event_callback=callback, **{'trim_head_frames': 39} if extension is not None else {})
    if request.seamless or extension is not None:
        from .minimax_h3_extension import save_context
        save_context(request.out, video_latent, audio_latent, geometry)
    if not request.out.is_file() or request.out.stat().st_size == 0:
        raise RuntimeError(f'H3 decode/mux did not create a non-empty output: {request.out}')
    _emit_json_event(json_events, 'completed', output=str(request.out), bytes=request.out.stat().st_size)
    return result

def _component_filter(names: Iterable[str], component: str) -> tuple[set[str], str]:
    names = set(names)
    prefix = component.rstrip('/') + '/'
    if any((name.startswith(prefix) for name in names)):
        return ({name for name in names if name.startswith(prefix)}, prefix)
    return (names, '')

def group_h3_transformer_blocks(index, component: str='transformer'):
    selected, _ = _component_filter(index.refs, component)
    source_hits = any((_SOURCE_BLOCK_RE.search(_strip_comfy_prefix(name)) for name in selected))
    diffusers_hits = any((_DIFFUSERS_BLOCK_RE.search(_strip_comfy_prefix(name)) for name in selected))
    if source_hits and diffusers_hits:
        raise ValueError('checkpoint mixes source blocks.N and diffusers transformer_blocks.N naming')
    rx = _SOURCE_BLOCK_RE if source_hits else _DIFFUSERS_BLOCK_RE
    naming = 'source' if source_hits else 'diffusers' if diffusers_hits else 'unknown'
    blocks: dict[int, list] = {}
    globals_: list = []
    for name in sorted(selected):
        ref = index.refs[name]
        match = rx.search(_strip_comfy_prefix(name))
        if match:
            blocks.setdefault(int(match.group(1)), []).append(ref)
        else:
            globals_.append(ref)
    return (blocks, globals_, naming)

def inspect_h3_transformer_index(index, component: str='transformer') -> H3StreamingLayout:
    blocks, globals_, naming = group_h3_transformer_blocks(index, component)
    ids = tuple(sorted(blocks))
    missing = tuple((i for i in range(H3_TRANSFORMER_BLOCKS) if i not in blocks))
    incomplete: tuple[int, ...] = ()
    if naming == 'source':
        required_suffixes = {'adaln_proj.linear.bias', 'adaln_proj.linear.weight', 'attn.out_proj.weight', 'attn.k_norm.weight', 'attn.q_norm.weight', 'attn.qkv_proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight', 'norm1.weight', 'norm2.weight'}
        bad = []
        for block_id, refs in blocks.items():
            marker = f'blocks.{block_id}.'
            suffixes = {ref.name.split(marker, 1)[1] for ref in refs if marker in ref.name}
            if not required_suffixes.issubset(suffixes):
                bad.append(block_id)
        incomplete = tuple(sorted(bad))
    block_bytes = tuple((sum((ref.nbytes for ref in blocks[i])) for i in ids))
    global_bytes = sum((ref.nbytes for ref in globals_))
    return H3StreamingLayout(block_count=len(blocks), block_ids=ids, missing_block_ids=missing, incomplete_block_ids=incomplete, block_bytes=block_bytes, global_tensor_count=len(globals_), global_bytes=global_bytes, transformer_bytes_seen=sum(block_bytes) + global_bytes, complete=ids == tuple(range(H3_TRANSFORMER_BLOCKS)) and (not incomplete), checkpoint_naming=naming)

def inspect_h3_transformer(model_dir: str | Path) -> H3StreamingLayout:
    from .safetensors_index import build_index
    path = Path(model_dir)
    transformer_dir = path / 'transformer'
    index = build_index(transformer_dir if transformer_dir.is_dir() else path,recursive=False)
    return inspect_h3_transformer_index(index, component='transformer')

def plan_h3_host_cache(index, cache_gb: float, component: str='transformer', hot_block_ids: Iterable[int]=()) -> H3HostCachePlan:
    if cache_gb < 0:
        raise ValueError('cache_gb must be non-negative')
    blocks, _, _ = group_h3_transformer_blocks(index, component)
    budget = int(cache_gb * 1000000000)
    priority: list[int] = []
    for raw in hot_block_ids:
        block_id = int(raw)
        if block_id not in blocks:
            raise ValueError(f'hot block {block_id} is absent from the checkpoint')
        if block_id not in priority:
            priority.append(block_id)
    priority.extend((i for i in sorted(blocks) if i not in priority))
    cached: list[int] = []
    used = 0
    for block_id in priority:
        size = sum((ref.nbytes for ref in blocks[block_id]))
        if used + size <= budget:
            cached.append(block_id)
            used += size
    cached_set = set(cached)
    return H3HostCachePlan(budget_bytes=budget, cached_block_ids=tuple(cached), cached_bytes=used, uncached_block_ids=tuple((i for i in sorted(blocks) if i not in cached_set)))
_STREAM_STATS = {'binds': 0, 'wait_s': 0.0, 'bytes': 0}
_STREAMER = None
_DLL_BYTES_AT_RESET = 0
_DLL_STATS_UNAVAILABLE_REPORTED = False

def _dll_uploaded_bytes() -> int:
    global _DLL_STATS_UNAVAILABLE_REPORTED, _DLL_STATS_REASON
    reason = ''
    if _STREAMER is None:
        import sys as _sys
        dupes = [k for k in _sys.modules if k.endswith('minimax_h3_backend')]
        _DLL_STATS_REASON = f"no streamer published in {__name__} (binds={_STREAM_STATS['binds']}); modules: {dupes}"
        if not _DLL_STATS_UNAVAILABLE_REPORTED:
            _DLL_STATS_UNAVAILABLE_REPORTED = True
            print(f"[h3] fetcher upload counter unavailable: no streamer published in {__name__} (binds={_STREAM_STATS['binds']}); modules named minimax_h3_backend: {dupes}", file=sys.stderr, flush=True)
        return 0
    value = 0
    if not hasattr(_STREAMER, 'fetch_stats'):
        reason = 'streamer has no fetch_stats (stale ds_backend)'
    else:
        try:
            stats = _STREAMER.fetch_stats()
            if not stats:
                reason = 'wf_get_stats not exported by this DLL'
            elif stats.get('layout_suspect'):
                reason = 'wf_stats ctypes layout failed its sanity check'
            else:
                value = int(stats.get('bytes_uploaded_gpu', 0) or 0)
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
    if reason:
        _DLL_STATS_REASON = reason
    if reason and (not _DLL_STATS_UNAVAILABLE_REPORTED):
        _DLL_STATS_UNAVAILABLE_REPORTED = True
        print(f'[h3] fetcher upload counter unavailable: {reason}', file=sys.stderr, flush=True)
    return value
_DLL_STATS_REASON = ''

def stream_stats() -> dict:
    out = dict(_STREAM_STATS)
    uploaded = _dll_uploaded_bytes()
    if _DLL_STATS_REASON:
        out['dll_counter_unavailable'] = _DLL_STATS_REASON
    if uploaded:
        out['dll_uploaded_bytes'] = max(0, uploaded - _DLL_BYTES_AT_RESET)
    return out

def reset_stream_stats() -> None:
    global _DLL_BYTES_AT_RESET
    _STREAM_STATS.update(binds=0, wait_s=0.0, bytes=0)
    _DLL_BYTES_AT_RESET = _dll_uploaded_bytes()

def build_streamed_h3_transformer(transformer, index, *, device: str='cuda', n_gpu_buffers: int=3, host_staging_buffers: int=2, host_cache_gb: float=0.0, resident: bool=False):
    if not 0 <= host_cache_gb <= 20:
        raise ValueError('host_cache_gb must be between 0 and 20')
    from accelerate.utils import set_module_tensor_to_device
    from .ds_backend import DirectStorageStreamer, enforce_ds_mode
    blocks, global_refs, naming = group_h3_transformer_blocks(index)
    layout = inspect_h3_transformer_index(index)
    if not layout.complete:
        raise ValueError(f'H3 transformer is incomplete: found {layout.block_count}/50 blocks; missing {list(layout.missing_block_ids)}; partial {list(layout.incomplete_block_ids)}')
    module_blocks = getattr(transformer, 'blocks', None)
    if module_blocks is None:
        module_blocks = getattr(transformer, 'transformer_blocks', None)
    if module_blocks is None or len(module_blocks) != H3_TRANSFORMER_BLOCKS:
        raise ValueError('H3 transformer must expose 50 blocks or transformer_blocks')

    def param_path(name: str) -> str:
        bare = name.split('/', 1)[1] if name.startswith('transformer/') else name
        return _strip_comfy_prefix(bare)
    from .minimax_h3_fused import dequantize_streamed_batch, is_quant_sidecar, sectioned_qkv_to_interleaved
    state_keys = set(transformer.state_dict().keys())
    resident_views = {}
    if resident:
        import torch
        state = transformer.state_dict()
        offsets, arena_bytes = ({}, 0)
        for refs in blocks.values():
            for ref in refs:
                path = param_path(ref.name)
                if path in state:
                    arena_bytes = (arena_bytes + 255) // 256 * 256
                    size = state[path].numel() * state[path].element_size()
                    offsets[path] = (arena_bytes, size)
                    arena_bytes += size
        arena = torch.empty(arena_bytes, dtype=torch.uint8, device=device)
        resident_views = {path: arena[offset:offset + size].view(state[path].dtype).view(state[path].shape) for path, (offset, size) in offsets.items()}
    indexed_paths = {param_path(name) for name in index.refs if (name.startswith('transformer/') or '/' not in name) and (not is_quant_sidecar(name))}
    missing_model_keys = sorted((path for path in indexed_paths if path not in state_keys and path != 'rope.inv_freq'))
    if missing_model_keys:
        raise ValueError('H3 checkpoint/model key mismatch before streaming (likely source-vs-diffusers conversion): ' + ', '.join(missing_model_keys[:8]))
    _cfg = getattr(transformer, 'config', None)
    _forced = _os.environ.get('STUDIO_H3_COMFY_QKV_SECTIONED', '')
    _repack_qkv = _forced == '1' if _forced in ('0', '1') else any((str(n).startswith(_COMFY_STATE_DICT_PREFIX) for n in index.refs))
    _heads = getattr(_cfg, 'num_attention_heads', 0) or 0
    _head_dim = getattr(_cfg, 'attention_head_dim', 0) or 0
    if _repack_qkv and (not (_heads and _head_dim)):
        raise ValueError('cannot re-pack ComfyUI qkv_proj without head geometry on the model config')
    global_streamer = DirectStorageStreamer(index, component='transformer', n_gpu_buffers=1, device=device, layer_groups={0: global_refs}, host_staging_buffers=1)
    enforce_ds_mode(global_streamer, 'MiniMax H3 globals', 'MINIMAX_H3')
    _g_bound: list[str] = []
    _g_skipped: list[str] = []
    try:
        globals_fetched = global_streamer.wait_layer(0)
        for ref in global_refs:
            path = param_path(ref.name)
            if path in state_keys:
                value = globals_fetched[ref.name].clone()
                if _repack_qkv:
                    value = repack_h3_qkv_tensor(path, value, heads=_heads, head_dim=_head_dim)
                set_module_tensor_to_device(transformer, path, device, value=value)
                _g_bound.append(path)
            else:
                _g_skipped.append(ref.name)
    finally:
        import torch
        torch.cuda.synchronize(device)
        global_streamer.close()
    del globals_fetched
    if _os.environ.get('STUDIO_H3_DEBUG_BIND') == '1':
        block_paths_all = {param_path(r.name) for refs in blocks.values() for r in refs}
        covered = set(_g_bound) | {p for p in block_paths_all if p in state_keys}
        never = sorted((k for k in state_keys if k not in covered))
        print(f'[bind] globals bound={len(_g_bound)} skipped={len(_g_skipped)} state_keys={len(state_keys)} NEVER_BOUND={len(never)}', file=sys.stderr, flush=True)
        if _g_skipped:
            print(f'[bind] skipped globals (first 5): {_g_skipped[:5]}', file=sys.stderr, flush=True)
        if never:
            print(f'[bind] never bound (first 10): {never[:10]}', file=sys.stderr, flush=True)
    if host_cache_gb:
        cache_plan = plan_h3_host_cache(index, host_cache_gb)
        streamer = H3TieredBlockStreamer(index, blocks, cache_plan, device=device, n_gpu_buffers=n_gpu_buffers, host_staging_buffers=host_staging_buffers)
    else:
        streamer = DirectStorageStreamer(index, component='transformer', n_gpu_buffers=n_gpu_buffers, device=device, layer_groups=blocks, host_staging_buffers=host_staging_buffers, record_compute_done=True if resident else None)
        enforce_ds_mode(streamer, 'MiniMax H3', 'MINIMAX_H3')
    global _STREAMER, _DLL_BYTES_AT_RESET
    _STREAMER = streamer
    _DLL_BYTES_AT_RESET = _dll_uploaded_bytes()
    block_paths = {block_id: [param_path(ref.name) for ref in refs if param_path(ref.name) in state_keys] for block_id, refs in blocks.items()}
    from .minimax_h3_int8 import H3ConvRotLinear, H3Int8ConvRotLinear
    int8_resident = any((isinstance(m, H3Int8ConvRotLinear) for m in transformer.modules()))
    convrot_resident = any((isinstance(m, H3ConvRotLinear) for m in transformer.modules()))
    if _os.environ.get('STUDIO_H3_DEBUG_BIND') == '1':
        n_convrot = sum((isinstance(m, H3ConvRotLinear) for m in transformer.modules()))
        n_int8 = sum((isinstance(m, H3Int8ConvRotLinear) for m in transformer.modules()))
        print(f'[bind] int8_resident={int8_resident} convrot_resident={convrot_resident} n_convrot={n_convrot} n_int8={n_int8} undo_rotation={not convrot_resident}', file=sys.stderr, flush=True)

    def evict_buffers_only(block_id):
        streamer.release_layer(block_id)

    def make_bind(block_id):

        def bind(own: bool=False):
            _t0 = _time.perf_counter()
            fetched = streamer.wait_layer(block_id)
            _STREAM_STATS['wait_s'] += _time.perf_counter() - _t0
            _STREAM_STATS['binds'] += 1
            _STREAM_STATS['bytes'] += sum((int(t.numel()) * t.element_size() for t in fetched.values()))
            if not int8_resident:
                fetched = dequantize_streamed_batch(fetched, undo_rotation=not convrot_resident)
            bound = skipped = 0
            for name, tensor in fetched.items():
                path = param_path(name)
                if _repack_qkv:
                    tensor = repack_h3_qkv_tensor(path, tensor, heads=_heads, head_dim=_head_dim)
                if path in state_keys:
                    if own:
                        owned = resident_views[path]
                        owned.copy_(tensor)
                        tensor = owned
                    set_module_tensor_to_device(transformer, path, device, value=tensor)
                    bound += 1
                else:
                    skipped += 1
            if _os.environ.get('STUDIO_H3_DEBUG_BIND') == '1':
                names = list(fetched)
                sample = next((t for n, t in fetched.items() if n.endswith('.weight')), None)
                stat = 'none' if sample is None else f'mean|w|={sample.float().abs().mean():.5f} dtype={sample.dtype}'
                print(f"[bind] block={block_id} fetched={len(names)} bound={bound} skipped={skipped} first={(names[0] if names else '-')} {stat}", file=sys.stderr, flush=True)
        return bind

    def make_evict(block_id):

        def evict():
            for path in block_paths[block_id]:
                set_module_tensor_to_device(transformer, path, 'meta')
            streamer.release_layer(block_id)
        return evict
    for block_id, block in enumerate(module_blocks):
        raw_forward = type(block).forward
        bind, evict = (make_bind(block_id), make_evict(block_id))
        block._h3_stream_bind = bind
        block._h3_stream_evict = evict
        block._h3_raw_forward = raw_forward
        if resident:
            bind(own=True)
            torch.cuda.synchronize(device)
            evict_buffers_only(block_id)
            continue

        def wrapped(*args, _block=block, _forward=raw_forward, _bind=bind, _evict=evict, **kwargs):
            _bind()
            try:
                return _forward(_block, *args, **kwargs)
            finally:
                _evict()
        block.forward = wrapped
    if not resident:
        transformer.register_forward_pre_hook(lambda _module, _args, _kwargs=None: streamer.new_pass())
    else:
        import torch
        torch.cuda.synchronize(device)
        streamer.close()
    transformer.eval()
    return (transformer, streamer)

def build_streamed_backward_h3_transformer(transformer, index, *, device='cuda'):
    from ..streamed_backward import apply_streamed_multi
    transformer, streamer = build_streamed_h3_transformer(transformer, index, device=device, n_gpu_buffers=1, host_staging_buffers=2, host_cache_gb=0)
    transformer.requires_grad_(False)
    for block in transformer.blocks:
        raw_forward = block._h3_raw_forward
        bind, evict = (block._h3_stream_bind, block._h3_stream_evict)

        def backward_wrapped(x, *, adaln_input, combined_indices, rope_cache=None, cu_seqlens=None, _block=block, _raw=raw_forward, _bind=bind, _evict=evict):
            call = lambda layer, hidden: _raw(layer, hidden, adaln_input=adaln_input, combined_indices=combined_indices, rope_cache=rope_cache, cu_seqlens=cu_seqlens)
            return apply_streamed_multi(_block, call, x, bind=_bind, evict=_evict)
        block.forward = backward_wrapped
        block._h3_backward_streamed = True
    return (transformer, streamer)

def smoke_available_h3_blocks(model_dir: str | Path, *, max_blocks: int=1, tokens: int=2, device: str='cuda') -> dict:
    if max_blocks < 1:
        raise ValueError('max_blocks must be at least 1')
    if tokens < 1:
        raise ValueError('tokens must be at least 1')
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .ds_backend import DirectStorageStreamer, enforce_ds_mode
    from .minimax_h3_native import build_meta_h3_native_model
    from .safetensors_index import build_index
    root = Path(model_dir)
    transformer_dir = root / 'transformer' if (root / 'transformer').is_dir() else root
    index = build_index(transformer_dir,recursive=False)
    blocks, _, naming = group_h3_transformer_blocks(index)
    layout = inspect_h3_transformer_index(index)
    if naming != 'source':
        raise ValueError('incremental smoke currently requires released source blocks.N checkpoint naming')
    unusable = set(layout.incomplete_block_ids)
    usable: list[int] = []
    for block_id in range(H3_TRANSFORMER_BLOCKS):
        if block_id not in blocks or block_id in unusable:
            break
        usable.append(block_id)
    selected = usable[:max_blocks]
    if not selected:
        raise ValueError('checkpoint has no complete leading H3 block to smoke-test')
    model = build_meta_h3_native_model(transformer_dir / 'config.json')
    state_keys = set(model.state_dict())
    for block_id in selected:
        missing = [ref.name for ref in blocks[block_id] if ref.name not in state_keys]
        if missing:
            raise ValueError(f'native H3 block {block_id} key mismatch: {missing[:8]}')
    selected_groups = {block_id: blocks[block_id] for block_id in selected}
    streamer = DirectStorageStreamer(index, component='transformer', n_gpu_buffers=1, device=device, layer_groups=selected_groups, host_staging_buffers=2)
    enforce_ds_mode(streamer, 'MiniMax H3 smoke', 'MINIMAX_H3')
    generator = torch.Generator(device=device).manual_seed(3)
    hidden = torch.randn((tokens, model.config.hidden_size), generator=generator, dtype=torch.bfloat16, device=device)
    adaln_input = torch.randn((1, model.config.time_embed_dim), generator=generator, dtype=torch.bfloat16, device=device)
    combined_indices = torch.zeros(tokens, dtype=torch.long, device=device)
    reports = []
    try:
        with torch.inference_mode():
            for block_id in selected:
                transfer_start = time.perf_counter()
                loaded = streamer.wait_layer(block_id)
                torch.cuda.synchronize()
                transfer_seconds = time.perf_counter() - transfer_start
                for name, tensor in loaded.items():
                    set_module_tensor_to_device(model, name, device, value=tensor)
                compute_start = time.perf_counter()
                hidden = model.blocks[block_id](hidden, adaln_input=adaln_input, combined_indices=combined_indices)
                torch.cuda.synchronize()
                compute_seconds = time.perf_counter() - compute_start
                finite = bool(torch.isfinite(hidden).all().item())
                norm = float(hidden.float().norm().item())
                block_nbytes = sum((ref.nbytes for ref in blocks[block_id]))
                reports.append({'block': block_id, 'bytes': block_nbytes, 'transfer_seconds': transfer_seconds, 'directstorage_gbps': block_nbytes / max(transfer_seconds, 1e-09) / 1000000000.0, 'compute_seconds': compute_seconds, 'output_finite': finite, 'output_l2': norm})
                for ref in blocks[block_id]:
                    set_module_tensor_to_device(model, ref.name, 'meta')
                streamer.release_layer(block_id)
                if not finite:
                    raise RuntimeError(f'H3 block {block_id} produced non-finite output')
    finally:
        streamer.close()
    return {'scope': 'incremental_dit_block_smoke_not_generation', 'checkpoint_complete': layout.complete, 'complete_leading_blocks_available': len(usable), 'blocks_executed': len(reports), 'tokens': tokens, 'device': device, 'reports': reports}

def _generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate video with native streamed MiniMax H3')
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--mode', required=True, choices=('t2v', 'i2v', 'l2v', 'flf2v', 'fflf', 'ref2v', 'v2v'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seamless', action='store_true')
    parser.add_argument('--extend-from', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--last-frame', type=Path)
    parser.add_argument('--reference-image', action='append', type=Path, default=[])
    parser.add_argument('--reference-audio', action='append', type=Path, default=[])
    parser.add_argument('--source-video', action='append', type=Path, default=[])
    parser.add_argument('--frame-index', action='append', type=int, default=[])
    parser.add_argument('--lora', action='append', type=_parse_lora, default=[], metavar='PATH:STRENGTH')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--width', type=int, default=1344)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--host-cache-gb', type=float, default=0.0)
    parser.add_argument('--resident', action='store_true', help='hold all 50 DiT blocks in VRAM instead of re-streaming them every denoise step (needs the transformer to fit)')
    from .minimax_h3_attention import SUPPORTED_H3_ATTENTION_BACKENDS
    parser.add_argument('--attention-backend', choices=SUPPORTED_H3_ATTENTION_BACKENDS, default='sdpa')
    parser.add_argument('--reference-resolution', type=int, default=None, help='Ref2VA reference-image sampling short edge (px); omit for the released 2048.')
    from .minimax_h3_sparse import SPARSE_ATTENTION_METHODS, SPARSE_ATTENTION_MODES, SPARSE_SINK_CONDITIONING
    parser.add_argument('--sparse-attention', choices=SPARSE_ATTENTION_MODES, default='auto', help='Block-sparse DiT attention: auto (on when the kernel is available, loudly dense otherwise), on (required), off.')
    parser.add_argument('--sparse-method', choices=SPARSE_ATTENTION_METHODS, default='sol-attn', help='sol-attn: adaptive threshold (--sparse-tau); sla: fixed --sparse-keep-percent.')
    parser.add_argument('--sparse-tau', type=float, default=1.3)
    parser.add_argument('--sparse-keep-percent', type=float, default=10.0)
    parser.add_argument('--sparse-start-percent', type=float, default=0.2, help='Schedule fraction at which sparse attention begins (sigma <= 1 - value).')
    parser.add_argument('--sparse-end-percent', type=float, default=1.0)
    parser.add_argument('--sparse-min-tokens', type=int, default=12288)
    parser.add_argument('--sparse-extra-tokens', type=int, default=256)
    parser.add_argument('--sparse-sink-conditioning', choices=SPARSE_SINK_CONDITIONING, default='exact_kv_and_rows')
    parser.add_argument('--json-events', action='store_true')
    return parser

def _generate_main(argv: list[str]) -> int:
    args = _generation_parser().parse_args(argv)
    request = H3GenerationRequest(model_dir=args.model_dir, prompt=args.prompt, mode=args.mode, out=args.out, image=args.image, last_frame=args.last_frame, seamless=args.seamless, extend_from=args.extend_from, reference_images=tuple(args.reference_image), source_videos=tuple(args.source_video), reference_audio=tuple(args.reference_audio), frame_indices=tuple(args.frame_index), loras=tuple(args.lora), duration=args.duration, width=args.width, height=args.height, steps=args.steps, seed=args.seed, gpu=args.gpu, host_cache_gb=args.host_cache_gb, resident=args.resident, attention_backend=args.attention_backend, reference_resolution=args.reference_resolution, sparse_attention=args.sparse_attention, sparse_method=args.sparse_method, sparse_tau=args.sparse_tau, sparse_keep_percent=args.sparse_keep_percent, sparse_start_percent=args.sparse_start_percent, sparse_end_percent=args.sparse_end_percent, sparse_min_tokens=args.sparse_min_tokens, sparse_extra_tokens=args.sparse_extra_tokens, sparse_sink_conditioning=args.sparse_sink_conditioning)
    highres_chunks = max(request.width, request.height) > 768 and 'STUDIO_H3_MLP_CHUNK' not in os.environ
    if highres_chunks:
        os.environ['STUDIO_H3_MLP_CHUNK'] = '16384'
        _emit_json_event(args.json_events, 'highres_mlp_chunks', rows=16384, message='bounded MLP/final-head temporaries; GEMM rounding may differ')
    highres_queries = max(request.width, request.height) > 768 and 'STUDIO_H3_ATTN_QUERY_CHUNK' not in os.environ
    if highres_queries:
        os.environ['STUDIO_H3_ATTN_QUERY_CHUNK'] = '8192'
        _emit_json_event(args.json_events, 'highres_attention_chunks', rows=8192, message='bounded dense FA2 launch duration; full key/value context retained')
    try:
        generate_h3(request, json_events=args.json_events)
    except Exception as exc:
        details = exc.as_dict() if hasattr(exc, 'as_dict') else None
        payload = {'error_type': type(exc).__name__, 'message': str(exc)}
        if details is not None:
            payload['details'] = details
        _emit_json_event(args.json_events, 'failed', **payload)
        if not args.json_events:
            print(f'MiniMax H3 generation failed: {exc}', file=sys.stderr)
        return 2
    finally:
        if highres_chunks:
            os.environ.pop('STUDIO_H3_MLP_CHUNK', None)
        if highres_queries:
            os.environ.pop('STUDIO_H3_ATTN_QUERY_CHUNK', None)
    return 0

def _main(argv: list[str] | None=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'generate':
        return _generate_main(argv[1:])
    parser = argparse.ArgumentParser(description='Inspect an H3 transformer for Studio block streaming')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--host-cache-gb', type=float, default=0.0)
    parser.add_argument('--smoke-blocks', type=int, default=0)
    parser.add_argument('--smoke-tokens', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    from .safetensors_index import build_index
    root = Path(args.model_dir)
    transformer_dir = root / 'transformer'
    index = build_index(transformer_dir if transformer_dir.is_dir() else root,recursive=False)
    layout = inspect_h3_transformer_index(index)
    result = {'layout': asdict(layout)}
    if args.host_cache_gb:
        result['host_cache_plan'] = asdict(plan_h3_host_cache(index, args.host_cache_gb))
        result['host_cache_runtime_active'] = True
    if args.smoke_blocks:
        result['gpu_smoke'] = smoke_available_h3_blocks(root, max_blocks=args.smoke_blocks, tokens=args.smoke_tokens, device=args.device)
    print(json.dumps(result, indent=2))
    return 0 if layout.complete else 2
if __name__ == '__main__':
    raise SystemExit(_main())
