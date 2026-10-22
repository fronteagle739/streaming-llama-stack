# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_vae_mux.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import dataclass
import contextlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import types
from typing import Any, Callable, Mapping
H3_FPS = 24
H3_AUDIO_RATE = 32000
H3_AUDIO_CHANNELS = 2
H3_AV_DRIFT_TOLERANCE_S = 0.25
H3_REFERENCE_SHORT_EDGE = 2048
H3_REFERENCE_SHORT_EDGE_MIN = 256
H3_REFERENCE_SHORT_EDGE_MAX = 4096

@dataclass(frozen=True)
class H3DecodeResult:
    output_path: Path
    width: int
    height: int
    frames: int
    fps: int
    audio_samples: int
    audio_sample_rate: int
    av_drift_seconds: float
    decode_seconds: float
    mux_seconds: float
    peak_decoder_weight_bytes: int
    transport: str = 'DirectStorage'

@dataclass(frozen=True)
class H3VisualConditionItem:
    rows: Any
    kind: str
    latent_t: int
    latent_h: int
    latent_w: int
    condition_index: int
    frame_index: int | None = None
    material_chain: str | None = None
    timestamps: tuple[float, ...] = ()

@dataclass(frozen=True)
class H3VisualCondition:
    rows: Any
    items: tuple[H3VisualConditionItem, ...]
    latent_t_total: int
    seed: int = 42

    @property
    def ref_blocks(self) -> list[dict[str, Any]]:
        return [{'kind': item.kind, 'latent_t': item.latent_t, 'latent_h': item.latent_h, 'latent_w': item.latent_w, 'condition_index': item.condition_index, 'material_chain': item.material_chain} for item in self.items]

@dataclass(frozen=True)
class H3AudioConditionItem:
    rows: Any
    ref_audio_t: int
    duration_seconds: float
    condition_index: int
    material_chain: str

@dataclass(frozen=True)
class H3AudioCondition:
    rows: Any
    items: tuple[H3AudioConditionItem, ...]

def _emit(callback, event: str, **payload) -> None:
    if callback is not None:
        callback(event, payload)

def _component_root(model_dir: str | Path) -> Path:
    root = Path(model_dir)
    candidates = (root, root / 'FL2VA', root / 'Ref2VA')
    for candidate in candidates:
        if (candidate / 'audio_vae').is_dir():
            return candidate
    raise FileNotFoundError(f'MiniMax H3 component root not found under {root}')

def _ordered_media(media, *, default_kind: str) -> list[dict[str, Any]]:
    if isinstance(media, (str, Path, Mapping)) or hasattr(media, 'size'):
        media = [media]
    if not isinstance(media, (list, tuple)) or not media:
        raise ValueError('H3 conditioned media must be a non-empty ordered sequence')
    result = []
    for index, value in enumerate(media):
        if isinstance(value, Mapping):
            item = dict(value)
            if 'path' not in item and 'image' not in item and ('frames' not in item):
                raise ValueError(f'conditioned media item {index} has no path/image/frames')
        elif hasattr(value, 'size') and (not isinstance(value, (str, Path))):
            item = {'image': value}
        else:
            item = {'path': value}
        item.setdefault('kind', default_kind)
        item.setdefault('condition_index', index)
        result.append(item)
    return result

def _load_pil_image(item: Mapping[str, Any]):
    from PIL import Image, ImageOps
    source = item.get('image')
    if source is not None:
        return source.convert('RGB')
    path = Path(item['path'])
    if not path.is_file():
        raise FileNotFoundError(f'H3 condition image does not exist: {path}')
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')

def _prepare_keyframe_images(media, geometry):
    from PIL import Image
    width = int(_geometry_value(geometry, 'width'))
    height = int(_geometry_value(geometry, 'height'))
    prepared = []
    for index, item in enumerate(_ordered_media(media, default_kind='image')):
        image = _load_pil_image(item)
        if image.size == (width, height):
            prepared_image = image
        elif index == 0:
            prepared_image = image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            scale = max(width / image.width, height / image.height)
            resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))), Image.Resampling.LANCZOS)
            left, top = ((resized.width - width) // 2, (resized.height - height) // 2)
            prepared_image = resized.crop((left, top, left + width, top + height))
        prepared.append((item, prepared_image))
    return prepared

def _reference_image_shape(width: int, height: int, *, short_edge: int=H3_REFERENCE_SHORT_EDGE) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError('reference image dimensions must be positive')
    if int(short_edge) <= 0:
        raise ValueError('reference sampling short edge must be positive')
    scale = float(int(short_edge)) / min(width, height)
    nearest32 = lambda value: max(32, int(round(value / 32.0)) * 32)
    return (nearest32(width * scale), nearest32(height * scale))

def _decode_reference_video_frames(path, *, width: int, height: int, frames: int, start: float=0.0):
    import numpy as np
    command = [shutil.which('ffmpeg') or 'ffmpeg', '-v', 'error']
    if start > 0:
        command += ['-ss', f'{start:.9g}']
    command += ['-i', str(path), '-map', '0:v:0', '-an', '-vf', f'fps={H3_FPS},scale={width}:{height}:flags=lanczos,setsar=1', '-frames:v', str(frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    decoded = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height * 3
    if not decoded.stdout or len(decoded.stdout) % frame_bytes:
        raise ValueError('ffmpeg returned invalid H3 reference-video RGB frames')
    return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, height, width, 3)

@contextlib.contextmanager
def _seeded_vae_sample(seed: int, device):
    import torch
    device = torch.device(device)
    devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(int(seed))
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        yield

def _geometry_value(geometry: Any, name: str, default=None):
    if isinstance(geometry, Mapping):
        value = geometry.get(name, default)
    else:
        value = getattr(geometry, name, default)
    if value is None:
        raise ValueError(f'H3 geometry is missing {name}')
    return value

def _normalise_audio_latent(audio_latent):
    import torch
    if not isinstance(audio_latent, torch.Tensor):
        raise TypeError('audio_latent must be a torch.Tensor')
    if audio_latent.ndim == 2 and int(audio_latent.shape[1]) == 32:
        if int(audio_latent.shape[0]) % 2:
            raise ValueError('packed H3 audio rows must be divisible by two')
        audio_latent = audio_latent.reshape(2, -1, 32).permute(0, 2, 1)
    elif audio_latent.ndim == 3 and tuple(audio_latent.shape[:2]) == (2, 32):
        pass
    elif audio_latent.ndim == 3 and tuple(audio_latent.shape[1:])[:1] == (32,):
        if int(audio_latent.shape[0]) != 1:
            raise ValueError('H3 audio latent must be [2,32,T] or packed [2*T,32]')
        audio_latent = audio_latent.expand(2, -1, -1)
    else:
        raise ValueError('H3 audio latent must be [2,32,T] or packed [2*T,32]')
    return audio_latent.contiguous()

def _reverse_normalize(latent, config: Mapping[str, Any], channels: int, label: str):
    import torch
    mean_values, std_values = (config.get('latents_mean'), config.get('latents_std'))
    if not isinstance(mean_values, list) or not isinstance(std_values, list):
        raise ValueError(f'{label} config is missing latent normalization statistics')
    if len(mean_values) != channels or len(std_values) != channels:
        raise ValueError(f'{label} latent statistics must contain {channels} channels')
    mean = torch.as_tensor(mean_values, device=latent.device, dtype=latent.dtype)
    std = torch.as_tensor(std_values, device=latent.device, dtype=latent.dtype)
    shape = [1] * latent.ndim
    shape[1] = channels
    return latent.mul(std.view(shape)).add(mean.view(shape))

def _meta_dtype(dtype: str):
    import torch
    table = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16, 'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool, 'F8_E4M3': torch.float8_e4m3fn}
    try:
        return table[dtype]
    except KeyError as exc:
        raise ValueError(f'unsupported VAE checkpoint dtype {dtype}') from exc

def _import_component_entry(component_dir: Path, config: Mapping[str, Any]):
    auto_map = config.get('auto_map') or {}
    target = auto_map.get('AutoModel') or auto_map.get('Model')
    if not isinstance(target, str) or '.' not in target:
        raise ValueError(f'{component_dir}/config.json has no supported auto_map entry')
    module_name, class_name = target.rsplit('.', 1)
    entry = component_dir / f'{module_name}.py'
    if not entry.is_file():
        raise FileNotFoundError(f'H3 VAE remote-code entry is missing: {entry}')
    package_name = f'_h3_vae_{abs(hash(str(component_dir.resolve()))):x}'
    package = types.ModuleType(package_name)
    package.__path__ = [str(component_dir)]
    sys.modules[package_name] = package
    qualified = f'{package_name}.{module_name}'
    spec = importlib.util.spec_from_file_location(qualified, entry)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import H3 VAE entry {entry}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return (module, getattr(module, class_name))

def _build_meta_component(component_dir: Path):
    import torch
    from .safetensors_index import build_index
    config = json.loads((component_dir / 'config.json').read_text(encoding='utf-8'))
    index = build_index(component_dir)
    module, component_class = _import_component_entry(component_dir, config)

    def metadata_load_file(filename, *args, **kwargs):
        selected = {name: ref for name, ref in index.refs.items() if Path(ref.file).resolve() == Path(filename).resolve()}
        if not selected:
            raise FileNotFoundError(f'checkpoint shard is absent from VAE index: {filename}')
        return {_checkpoint_name(name): torch.empty(ref.shape, dtype=_meta_dtype(ref.dtype), device='meta') for name, ref in selected.items()}
    import safetensors.torch
    original_global_load_file = safetensors.torch.load_file
    original_module_load_file = getattr(module, 'load_file', None)
    original_load_state = torch.nn.Module.load_state_dict

    def metadata_load_state(model, state, *args, **kwargs):
        from .minimax_h3_w4a8 import logical_w4a8_meta_state
        return original_load_state(model, logical_w4a8_meta_state(state, model.state_dict()), *args, **kwargs)
    torch.nn.Module.load_state_dict = metadata_load_state
    safetensors.torch.load_file = metadata_load_file
    if original_module_load_file is not None:
        module.load_file = metadata_load_file
    try:
        with torch.device('meta'):
            wrapper = component_class.from_pretrained(str(component_dir))
    finally:
        torch.nn.Module.load_state_dict = original_load_state
        safetensors.torch.load_file = original_global_load_file
        if original_module_load_file is not None:
            module.load_file = original_module_load_file
    model = getattr(wrapper, 'model', wrapper)
    return (wrapper, model, index, config)

def _checkpoint_name(index_name: str) -> str:
    return index_name.split('/', 1)[1] if '/' in index_name else index_name

def _model_name(checkpoint_name: str) -> str:
    checkpoint_name = _checkpoint_name(checkpoint_name)
    if checkpoint_name.endswith('.weight_g'):
        return checkpoint_name[:-9] + '.parametrizations.weight.original0'
    if checkpoint_name.endswith('.weight_v'):
        return checkpoint_name[:-9] + '.parametrizations.weight.original1'
    return checkpoint_name

def _bind_resident_directstorage(model, index, refs, *, device: str, label: str):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .ds_backend import DirectStorageStreamer, enforce_ds_mode
    refs = list(refs)
    if not refs:
        raise ValueError(f'{label} has no decoder weights')
    state_keys = set(model.state_dict())
    missing = [_model_name(ref.name) for ref in refs if _model_name(ref.name) not in state_keys]
    if missing:
        raise ValueError(f'{label} checkpoint/model mismatch: {missing[:8]}')
    streamer = DirectStorageStreamer(index, component='', n_gpu_buffers=1, device=device, layer_groups={0: refs}, host_staging_buffers=1)
    enforce_ds_mode(streamer, label, 'MINIMAX_H3_VAE')
    paths = []
    try:
        loaded = streamer.wait_layer(0)
        from .minimax_h3_vae_decode_config import decode_cache_weights
        if decode_cache_weights() and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        for checkpoint_name, tensor in loaded.items():
            path = _model_name(checkpoint_name)
            set_module_tensor_to_device(model, path, device, value=tensor.clone())
            paths.append(path)
        torch.cuda.synchronize(device)
    finally:
        streamer.close()
    return (paths, sum((ref.nbytes for ref in refs)))

def _video_block_groups(index):
    groups: dict = {}
    prefix = 'decoder.transformer_blocks.'
    for ref in index.refs.values():
        name = _checkpoint_name(ref.name)
        if not name.startswith(prefix):
            continue
        block_text, sep, _ = name[len(prefix):].partition('.')
        if sep and block_text.isdigit():
            groups.setdefault(int(block_text), []).append(ref)
    return list(groups.values())
VAE_ACTIVATION_HEADROOM_BYTES = int(2.5 * 1000000000.0)
_COMPUTE_LINEAR_MARKERS = ('.attn.to_qkv.', '.attn.to_out.', '.ff.w1.', '.ff.w2.')

def _is_compute_linear(path: str) -> bool:
    return any((marker in path for marker in _COMPUTE_LINEAR_MARKERS))

def _resident_block_bytes(model) -> int:
    total = 0
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        if name.startswith('decoder.transformer_blocks.'):
            packed = name.rpartition('.')[0] in getattr(model, '_h3_packed_w4a8_names', ())
            width = 2 if _is_compute_linear(name) and (not packed) else tensor.element_size()
            total += int(tensor.numel()) * width
    return total

def _vae_blocks_fit_in_vram(block_bytes: int, device: str='cuda') -> tuple:
    override = os.environ.get('MINIMAX_H3_VAE_RESIDENT_BUDGET_GB', '').strip()
    if override:
        allowed = int(float(override) * 1000000000.0)
        return (block_bytes <= allowed, allowed, f'override {override} GB')
    try:
        import torch
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
        fraction = torch.cuda.get_per_process_memory_fraction(torch.device(device))
        allocator_free = int(total_bytes * fraction) - torch.cuda.memory_allocated(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    except Exception as exc:
        return (False, 0, f'free VRAM unreadable ({type(exc).__name__})')
    from .minimax_h3_vae_decode_config import decode_tile_extra_bytes
    headroom = VAE_ACTIVATION_HEADROOM_BYTES + decode_tile_extra_bytes()
    allowed = min(int(free_bytes), int(allocator_free)) - headroom
    reason = f'{block_bytes / 1000000000.0:.2f} GB of blocks against {free_bytes / 1000000000.0:.2f} GB free ({allocator_free / 1000000000.0:.2f} GB under the allocator cap) minus {headroom / 1000000000.0:.1f} GB activation headroom; allocated={allocated / 1000000000.0:.2f}, reserved={reserved / 1000000000.0:.2f} GB'
    return (block_bytes <= allowed, allowed, reason)

def _install_resident_video_blocks(model, index, groups, streamer, *, device: str, event_callback=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .minimax_h3_w4a8 import video_block_tensors
    started = time.perf_counter()
    bound_paths = []
    held = 0
    streamer.new_pass()
    for block_id in sorted(groups):
        fetched = streamer.wait_layer(block_id)
        from .minimax_h3_vae_decode_config import decode_cache_weights
        if decode_cache_weights() and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        for checkpoint_name, tensor in video_block_tensors(fetched, model).items():
            path = _model_name(checkpoint_name)
            packed = path.rpartition('.')[0] in getattr(model, '_h3_packed_w4a8_names', ())
            target = torch.bfloat16 if _is_compute_linear(path) and (not packed) else tensor.dtype
            value = tensor.to(dtype=target, copy=True)
            set_module_tensor_to_device(model, path, device, value=value, dtype=value.dtype)
            bound_paths.append(path)
            held += int(value.numel()) * value.element_size()
        del fetched
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        streamer.release_layer(block_id)
    _emit(event_callback, 'vae_blocks_resident', blocks=len(groups), bytes=held, seconds=time.perf_counter() - started)
    _emit(event_callback, 'vae_blocks_bound_once', tensors=len(bound_paths), bytes=held)
    return bound_paths

def _install_streamed_video_blocks(model, index, *, device: str, event_callback=None, resident: bool=False):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .ds_backend import DirectStorageStreamer, enforce_ds_mode
    try:
        blocks = model.decoder.transformer_blocks
    except AttributeError as exc:
        raise ValueError('H3 video VAE has no decoder.transformer_blocks') from exc
    groups = {}
    for ref in index.refs.values():
        prefix = 'decoder.transformer_blocks.'
        name = _checkpoint_name(ref.name)
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        block_text, sep, _ = tail.partition('.')
        if not sep or not block_text.isdigit():
            continue
        groups.setdefault(int(block_text), []).append(ref)
    if sorted(groups) != list(range(len(blocks))):
        raise ValueError(f'H3 video VAE block index is incomplete: checkpoint={sorted(groups)} model={len(blocks)}')
    from .minimax_h3_w4a8 import video_block_tensors, is_w4a8_sidecar
    state_keys = set(model.state_dict())
    for block_id, refs in groups.items():
        missing = [_model_name(ref.name) for ref in refs if not is_w4a8_sidecar(ref.name) and _model_name(ref.name) not in state_keys]
        if missing:
            raise ValueError(f'H3 video VAE block {block_id} mismatch: {missing[:8]}')
    streamer = DirectStorageStreamer(index, component='', n_gpu_buffers=2, device=device, layer_groups=groups, host_staging_buffers=1)
    enforce_ds_mode(streamer, 'MiniMax H3 video VAE blocks', 'MINIMAX_H3_VAE')
    for block_id, block in enumerate([] if resident else blocks):
        raw_forward = type(block).forward
        refs = groups[block_id]

        def wrapped(*args, _id=block_id, _block=block, _raw=raw_forward, _refs=refs, **kwargs):
            started = time.perf_counter()
            loaded = streamer.wait_layer(_id)
            loaded = video_block_tensors(loaded, model)
            for checkpoint_name, tensor in loaded.items():
                path = _model_name(checkpoint_name)
                is_compute_linear = any((marker in path for marker in ('.attn.to_qkv.', '.attn.to_out.', '.ff.w1.', '.ff.w2.')))
                packed = path.rpartition('.')[0] in getattr(model, '_h3_packed_w4a8_names', ())
                value = tensor.to(torch.bfloat16) if is_compute_linear and (not packed) else tensor
                if packed:
                    set_module_tensor_to_device(model, path, device, value=value, dtype=value.dtype)
                else:
                    set_module_tensor_to_device(model, path, device, value=value)
            _emit(event_callback, 'vae_video_block_bound', block=_id, bytes=sum((ref.nbytes for ref in _refs)), seconds=time.perf_counter() - started)
            try:
                return _raw(_block, *args, **kwargs)
            finally:
                for ref in _refs:
                    if is_w4a8_sidecar(ref.name) and _model_name(ref.name) not in state_keys:
                        continue
                    set_module_tensor_to_device(model, _model_name(ref.name), 'meta')
                streamer.release_layer(_id)
        block.forward = wrapped
    if resident:
        return (streamer, groups)
    model.decoder.register_forward_pre_hook(lambda _module, _args, _kwargs=None: streamer.new_pass(), with_kwargs=True)
    return (streamer, groups)

def _materialize_video_derived_buffers(model, *, device: str) -> None:
    import torch
    pos_embed = model.decoder.pos_embed
    inv_freq = getattr(pos_embed, 'inv_freq', None)
    if not isinstance(inv_freq, torch.Tensor):
        raise RuntimeError('H3 video VAE decoder has no rotary inv_freq buffer')
    if inv_freq.device.type == 'meta':
        rope_theta = float(getattr(model.decoder.config, 'rope_theta', 100.0))
        step = 2 * int(pos_embed.n_dim) / int(pos_embed.dim)
        rebuilt = 1 / rope_theta ** torch.arange(0, 1, step, dtype=torch.float32, device=device)
        pos_embed.inv_freq = rebuilt

def _bounded_decode_temporal_streaming(self, z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype):
    import torch
    if num_chunks == 0 and z_head is None and (z_tail is None):
        context_tokens = int(self.tokens_chunk_size)
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, context_tokens, 1, 1)], dim=2)
        pad_tokens += context_tokens
        num_chunks = 1
    total_frames, pad_frames, output_frames = self._decode_temporal_output_frame_plan(z, z_head, z_tail, num_chunks, pad_tokens)
    if output_frames <= 0:
        raise ValueError(f'decode_temporal streaming planned non-positive output_frames={output_frames} total_frames={total_frames} pad_frames={pad_frames}')
    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    dec = None
    dec_overlap = None
    write_pos = logical_frames = dropped_frames = 0
    host_sink = getattr(self, '_h3_host_output', None)
    encoder = getattr(self, '_h3_progressive_encoder', None)
    from .minimax_h3_vae_decode_config import decode_reuse_workspace
    reuse_workspace = decode_reuse_workspace()

    def write_part(part):
        nonlocal dec, write_pos, logical_frames, dropped_frames
        part_frames = int(part.shape[2])
        if part_frames <= 0:
            return
        logical_frames += part_frames
        capture = getattr(self, '_h3_capture_final_frame', None)
        retained = min(part_frames, max(0, output_frames - write_pos))
        capture_count = getattr(self, '_h3_capture_frame_count', 1)
        capture_start = max(0, output_frames - capture_count - write_pos)
        if capture is not None and retained > capture_start:
            capture(part[:, :, capture_start:retained], write_pos + retained - 1)
        if host_sink is not None:
            part = host_sink['finish'](part)
        if dec is None:
            out_shape = list(part.shape)
            out_shape[2] = output_frames
            if host_sink is not None:
                dec = host_sink['allocate'](out_shape, part.dtype)
            else:
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
        remaining = int(dec.shape[2]) - write_pos
        copy_frames = min(part_frames, max(0, remaining))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames].copy_(part[:, :, :copy_frames], non_blocking=host_sink is not None)
            if encoder is not None and (not encoder.failed):
                encoder.write(part[:, :, :copy_frames])
            write_pos += copy_frames
        dropped_frames += part_frames - copy_frames
    for index in range(num_chunks):
        start = index * self.tokens_chunk_size
        stop = start + self.tokens_chunk_size + self.token_overlap
        clip_z = z[:, :, start:stop]
        if index == 0 and z_head is not None:
            clip_z = torch.cat([z_head, clip_z], dim=2)
        if index == num_chunks - 1 and z_tail is not None:
            clip_z = torch.cat([clip_z, z_tail], dim=2)
        trace = getattr(self, '_h3_decode_trace', None)
        before = len(trace['traversals']) if trace is not None else 0
        clip_dec = self._adaptive_decode(clip_z)
        if trace is not None:
            trace['temporal_chunks'].append(dict(index=index, start=start, stop=stop, latent_shape=list(clip_z.shape), traversals=len(trace['traversals']) - before, output_start=write_pos))
        if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
            clip_dec = clip_dec.to(temporal_cat_dtype)
        if clip_dec.device != z.device:
            clip_dec = clip_dec.to(z.device)
        dec_tail = None
        if index == 0 and z_head is not None:
            head_view = clip_dec[:, :, self.vae_ratio_t - 1:self.vae_ratio_t]
            write_part(head_view)
            del head_view
            clip_dec = clip_dec[:, :, self.vae_ratio_t:]
        if index == num_chunks - 1 and z_tail is not None:
            dec_tail = clip_dec[:, :, -1:]
            clip_dec = clip_dec[:, :, :-self.vae_ratio_t]
        for split in range(split_count):
            frame_start = split * chunk_dec
            frame_stop = min(frame_start + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, frame_start:frame_stop]
            clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:]
            if split == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = self.blend(dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3)
                    del dec_overlap
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = clip_dec_chunk.clone(memory_format=torch.contiguous_format)
            del clip_dec_chunk
        if index == num_chunks - 1:
            if dec_overlap is not None:
                write_part(dec_overlap)
                del dec_overlap
                dec_overlap = None
            if dec_tail is not None:
                write_part(dec_tail)
                del dec_tail
                dec_tail = None
        del clip_dec, clip_z
        if z.is_cuda and index + 1 < num_chunks and (not reuse_workspace):
            torch.cuda.synchronize(z.device)
            torch.cuda.empty_cache()
    if dec is None:
        raise RuntimeError('decode_temporal streaming produced no output tensor')
    if host_sink is not None and z.is_cuda:
        torch.cuda.synchronize(z.device)
    if logical_frames != total_frames or dropped_frames != pad_frames or write_pos != output_frames:
        raise RuntimeError(f'decode_temporal streaming frame plan mismatch: logical_frames={logical_frames} total_frames={total_frames} dropped_frames={dropped_frames} pad_frames={pad_frames} write_pos={write_pos} output_frames={output_frames}')
    return dec

def _vae_host_output_enabled() -> bool:
    return os.environ.get('MINIMAX_H3_VAE_HOST_OUTPUT', '1').strip().lower() not in ('0', 'false', 'no', 'off')

def _progressive_video_enabled() -> bool:
    return os.environ.get('MINIMAX_H3_PROGRESSIVE_VIDEO', '0').strip().lower() in ('1', 'true', 'yes', 'on')

def _progressive_encoder_args() -> list:
    if os.environ.get('MINIMAX_H3_PROGRESSIVE_NVENC', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
        return ['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq', '-rc', 'vbr', '-cq', '18', '-b:v', '0']
    return ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18']

class _ProgressiveVideoEncoder:

    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        self.path = Path(path)
        self.frames_written = 0
        self.failed: str | None = None
        self._proc = None
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            self.failed = 'ffmpeg not found'
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._proc = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an', *_progressive_encoder_args(), '-pix_fmt', 'yuv420p', '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, ValueError) as exc:
            self.failed = f'could not start ffmpeg: {exc}'
            self._proc = None

    def write(self, part) -> None:
        if self._proc is None or self.failed:
            return
        import torch
        try:
            block = part[0]
            if block.dtype != torch.uint8:
                block = block.clamp(0, 1).mul(255).to(torch.uint8)
            packed = block.permute(1, 2, 3, 0).contiguous().cpu().numpy()
            self._proc.stdin.write(memoryview(packed))
            self.frames_written += int(part.shape[2])
        except (OSError, ValueError, BrokenPipeError) as exc:
            self.failed = f'progressive encode failed after {self.frames_written} frames: {exc}'

    def close(self, *, expected_frames: int | None=None) -> bool:
        if self._proc is None:
            return False
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self.failed = self.failed or 'progressive encode timed out'
        stderr = b''
        try:
            stderr = self._proc.stderr.read() or b''
            self._proc.stderr.close()
        except (OSError, ValueError):
            pass
        if self._proc.returncode not in (0, None):
            self.failed = self.failed or f"ffmpeg exited {self._proc.returncode}: {stderr.decode('utf-8', 'replace')[-300:]}"
        if expected_frames is not None and self.frames_written != int(expected_frames):
            self.failed = self.failed or f'progressive encode wrote {self.frames_written} frames, expected {expected_frames}'
        if self.failed or not self.path.is_file() or self.path.stat().st_size == 0:
            return False
        return True

def _host_output_sink(processor, *, height: int, width: int) -> dict:
    import torch

    def finish(part):
        pixels = processor.revert_tensor(part)
        if pixels.ndim == 4:
            pixels = pixels.reshape(1, -1, *pixels.shape[1:]).transpose(1, 2)
        pixels = pixels[..., :height, :width].float()
        return pixels.clamp_(0, 1).mul_(255).to(torch.uint8).contiguous()

    def allocate(shape, dtype):
        try:
            return torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
        except (RuntimeError, OSError):
            return torch.empty(shape, dtype=dtype, device='cpu')
    return {'finish': finish, 'allocate': allocate}
_VIDEO_DECODE_CACHE = None
_VIDEO_DECODE_PARKED = None

def decode_offload_enabled() -> bool:
    return os.environ.get('MINIMAX_H3_VAE_DECODE_OFFLOAD', '0').strip() not in ('', '0', 'false', 'no', 'off')

def park_video_decode_cache(event_callback=None) -> int:
    global _VIDEO_DECODE_PARKED
    import torch
    cache = _VIDEO_DECODE_CACHE
    if cache is None or _VIDEO_DECODE_PARKED is not None:
        return 0
    model = cache['model']
    state = model.state_dict()
    parked, total = ({}, 0)
    for path in list(cache['paths']) + list(cache['resident_paths']):
        tensor = state.get(path)
        if tensor is None or not tensor.is_cuda:
            continue
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        parked[path] = host
        total += tensor.numel() * tensor.element_size()
    if not parked:
        return 0
    torch.cuda.synchronize()
    from accelerate.utils import set_module_tensor_to_device
    for path, host in parked.items():
        set_module_tensor_to_device(model, path, 'cpu', value=host)
    _VIDEO_DECODE_PARKED = parked
    _emit(event_callback, 'vae_decoder_parked', bytes=total, tensors=len(parked))
    return total

def unpark_video_decode_cache(device: str, event_callback=None) -> int:
    global _VIDEO_DECODE_PARKED
    import torch
    cache, parked = (_VIDEO_DECODE_CACHE, _VIDEO_DECODE_PARKED)
    if cache is None or not parked:
        return 0
    from accelerate.utils import set_module_tensor_to_device
    model, total = (cache['model'], 0)
    for path, host in parked.items():
        set_module_tensor_to_device(model, path, device, value=host.to(device, non_blocking=True))
        total += host.numel() * host.element_size()
    torch.cuda.synchronize()
    _VIDEO_DECODE_PARKED = None
    _emit(event_callback, 'vae_decoder_unparked', bytes=total, tensors=len(parked))
    return total

def _video_decoder_cache_key(component_dir, device):
    files = sorted(component_dir.rglob('*.safetensors')) + sorted(component_dir.rglob('config.json'))
    return (str(component_dir.resolve()), str(device), tuple(((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)), tuple(((k, v) for k, v in sorted(os.environ.items()) if k.startswith(('MINIMAX_H3_VAE_', 'STUDIO_H3_FUSED_W4A8')))))

def clear_video_decode_cache():
    global _VIDEO_DECODE_CACHE, _VIDEO_DECODE_PARKED
    cached, _VIDEO_DECODE_CACHE = (_VIDEO_DECODE_CACHE, None)
    _VIDEO_DECODE_PARKED = None
    if cached is not None:
        from accelerate.utils import set_module_tensor_to_device
        for path in cached['resident_paths'] + cached['paths']:
            set_module_tensor_to_device(cached['model'], path, 'meta')

@contextlib.contextmanager
def _projection_stage(callback, name, device):
    if callback is None:
        yield
        return
    import torch

    def sync():
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    started = time.perf_counter()
    sync()
    ready = time.perf_counter()
    try:
        yield
    finally:
        sync()
        callback(dict(event='vae_projection_profile', stage=name, pending_s=ready - started, seconds=time.perf_counter() - ready))

def _decode_video(component_dir: Path, latent, geometry, *, device: str, event_callback=None, progressive_path: Path | None=None, decoded_frame_callback=None, decoded_frame_count=1, profile_callback=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    global _VIDEO_DECODE_CACHE
    from .minimax_h3_vae_decode_config import decode_cache_weights
    cache_requested = decode_cache_weights()
    cache_key = _video_decoder_cache_key(component_dir, device) if cache_requested else None
    if _VIDEO_DECODE_CACHE is not None and (not cache_requested or _VIDEO_DECODE_CACHE['key'] != cache_key):
        clear_video_decode_cache()
    cached = _VIDEO_DECODE_CACHE
    if cached is not None and decode_offload_enabled():
        unpark_video_decode_cache(device, event_callback)
    if cached is None:
        wrapper, model, index, config = _build_meta_component(component_dir)
    else:
        wrapper, model, index, config = (cached[k] for k in ('wrapper', 'model', 'index', 'config'))
    from .minimax_h3_w4a8 import prepare_packed_w4a8_blocks
    from .minimax_h3_vae_decode_config import configure_video_decode
    from .minimax_h3_vae_fast import decoder_mode
    configure_video_decode(model, width=int(_geometry_value(geometry, 'width')), height=int(_geometry_value(geometry, 'height')))
    int8_mode = decoder_mode(component_dir) == 'int8_convrot'
    if int8_mode:
        packed_blocks = ()
        if cached is None:
            from .minimax_h3_vae_fast import replace_heavy_linears, install_int8_decoder_forward
            replace_heavy_linears(model)
            install_int8_decoder_forward(model)
    else:
        packed_blocks = prepare_packed_w4a8_blocks(model, index) if cached is None else cached['packed_blocks']
    if cache_requested and (not int8_mode) and (len(packed_blocks) != 4 * len(model.decoder.transformer_blocks)):
        raise ValueError('H3 decoder cache requires all heavy linears packed with fused W4A8 enabled')
    if packed_blocks:
        _emit(event_callback, 'vae_w4a8_fused', modules=len(packed_blocks))
    trace = dict(temporal_chunks=[], traversals=[]) if os.environ.get('MINIMAX_H3_VAE_DECODE_TRACE', '0') == '1' else None
    model._h3_decode_trace = trace

    def count_traversal(_module, args):
        trace['traversals'].append(list(args[0].shape))
    trace_hook = model.decoder.register_forward_pre_hook(count_traversal) if trace is not None else None
    block_prefix = 'decoder.transformer_blocks.'
    globals_ = [ref for ref in index.refs.values() if (_checkpoint_name(ref.name).startswith('decoder.') or _checkpoint_name(ref.name).startswith('post_quant_conv.')) and (not _checkpoint_name(ref.name).startswith(block_prefix))]
    if cached is None:
        paths, global_bytes = _bind_resident_directstorage(model, index, globals_, device=device, label='MiniMax H3 video VAE globals')
    else:
        paths, global_bytes = (cached['paths'], cached['global_bytes'])
    _materialize_video_derived_buffers(model, device=device)
    model._decode_temporal_streaming = types.MethodType(_bounded_decode_temporal_streaming, model)
    block_bytes = _resident_block_bytes(model)
    vae_resident, _allowed, _fit_reason = (True, 0, 'cached packed decoder') if cached is not None else (True, 0, 'int8 convrot decoder') if int8_mode else _vae_blocks_fit_in_vram(block_bytes, device)
    if not vae_resident:
        _emit(event_callback, 'vae_residency_declined', reason=_fit_reason, bytes=block_bytes)
    if cached is None and int8_mode:
        from .minimax_h3_vae_fast import bind_int8_convrot_blocks, default_int8_convrot_file
        int8_file = default_int8_convrot_file(component_dir)
        bind_started = time.perf_counter()
        resident_paths, block_bytes = bind_int8_convrot_blocks(model, int8_file, device=device)
        streamer, groups = (None, {})
        _emit(event_callback, 'vae_int8_convrot_bound', file=str(int8_file), bytes=block_bytes, tensors=len(resident_paths), seconds=time.perf_counter() - bind_started)
    elif cached is None:
        streamer, groups = _install_streamed_video_blocks(model, index, device=device, event_callback=event_callback, resident=vae_resident)
        resident_paths: list = []
    else:
        streamer, groups, resident_paths = (None, cached['groups'], cached['resident_paths'])
        _emit(event_callback, 'vae_decoder_cache_hit', bytes=block_bytes + global_bytes)
    if vae_resident and cached is None and (not int8_mode):
        resident_paths = _install_resident_video_blocks(model, index, groups, streamer, device=device, event_callback=event_callback)
        with contextlib.suppress(Exception):
            streamer.close()
        block_bytes = _resident_block_bytes(model)
    peak_block_bytes = max((sum((ref.nbytes for ref in refs)) for refs in groups.values()), default=0)
    if int8_mode:
        block_bytes = cached['block_bytes'] if cached is not None else block_bytes
    height, width = (int(_geometry_value(geometry, 'height')), int(_geometry_value(geometry, 'width')))
    processor = getattr(wrapper, 'processor', getattr(model, 'processor', None))
    if processor is None:
        raise RuntimeError('H3 video VAE exposes no output processor')
    host_output = _vae_host_output_enabled()
    captured_parts = []

    def capture_final_frame(part, index):
        with _projection_stage(profile_callback, 'capture_revert', device):
            pixels = processor.revert_tensor(part)
        with _projection_stage(profile_callback, 'capture_slice', device):
            if pixels.ndim == 4:
                pixels = pixels.reshape(1, -1, *pixels.shape[1:]).transpose(1, 2)
            frame = pixels[0, :, :, :height, :width]
        with _projection_stage(profile_callback, 'capture_copy', device):
            frame = frame.float().to('cpu', copy=True).contiguous()
        captured_parts.append(frame)
        tail = torch.cat(captured_parts, dim=1)
        decoded_frame_callback(tail[:, 0] if decoded_frame_count == 1 else tail, index)
    model._h3_capture_final_frame = capture_final_frame if decoded_frame_callback else None
    model._h3_capture_frame_count = int(decoded_frame_count)
    encoder = None
    if host_output:
        model._h3_host_output = _host_output_sink(processor, height=height, width=width)
        if progressive_path is not None and _progressive_video_enabled():
            encoder = _ProgressiveVideoEncoder(progressive_path, width=width, height=height, fps=H3_FPS)
            if encoder.failed:
                _emit(event_callback, 'vae_progressive_unavailable', reason=encoder.failed)
                encoder = None
            else:
                model._h3_progressive_encoder = encoder
    progressive_video = None
    succeeded = False
    try:
        if cache_requested and cached is None:
            from .minimax_h3_vae_integrity import verify_decoder_upload
            integrity_started = time.perf_counter()
            integrity = verify_decoder_upload(model, index, paths if int8_mode else paths + resident_paths)
            _emit(event_callback, 'vae_decoder_integrity', **integrity, seconds=time.perf_counter() - integrity_started)
            if integrity['mismatches'] or integrity['unverified']:
                raise RuntimeError(f"VAE upload verification failed: {integrity['byte_mismatch_count']} of {integrity['checked_tensors']} TENSORS differ (not bytes) {integrity['mismatches'][:8]}; {integrity['unverified_count']} bound-but-not-in-index {integrity['unverified'][:8]}")
        model.eval()
        decoded_latent = _reverse_normalize(latent.to(device), config, 24, 'video_vae')
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=False):
            frames = model.decode_base(decoded_latent)
            if trace is not None:
                _emit(event_callback, 'vae_decode_traversals', **trace, traversal_count=len(trace['traversals']), temporal_chunk_count=len(trace['temporal_chunks']), tokens_chunk_size=model.tokens_chunk_size, token_overlap=model.token_overlap, token_drop=model.token_drop, frame_pre_padding=model.frame_pre_padding, frame_overlap=model.frame_overlap, vae_ratio_t=model.vae_ratio_t, decoder_tile_size=model.decoder_tile_size, stack_tiling=model.stack_tiling)
            if not host_output:
                frames = processor.revert_tensor(frames)
        if frames.ndim == 4:
            frames = frames.reshape(1, -1, *frames.shape[1:]).transpose(1, 2)
        if frames.ndim != 5 or int(frames.shape[0]) != 1 or int(frames.shape[1]) != 3:
            raise ValueError(f'H3 video VAE returned unsupported shape {tuple(frames.shape)}')
        if int(frames.shape[-2]) < height or int(frames.shape[-1]) < width:
            raise ValueError('H3 video VAE decoded canvas is smaller than requested geometry')
        if host_output:
            if frames.is_cuda or frames.dtype != torch.uint8:
                raise RuntimeError('H3 host-output decode returned a non-host or non-uint8 buffer')
            frames = frames[..., :height, :width]
        else:
            frames = frames[..., :height, :width].float().contiguous()
        peak_weight_bytes = max(2 * global_bytes, global_bytes + 3 * peak_block_bytes)
        if encoder is not None:
            if encoder.close(expected_frames=int(frames.shape[2])):
                progressive_video = encoder.path
                _emit(event_callback, 'vae_progressive_video_ready', path=str(encoder.path), frames=encoder.frames_written)
            else:
                _emit(event_callback, 'vae_progressive_discarded', reason=encoder.failed)
                Path(encoder.path).unlink(missing_ok=True)
        succeeded = True
        return (frames, peak_weight_bytes, progressive_video)
    finally:
        if streamer is not None:
            streamer.close()
        from .minimax_h3_vae_decode_config import decode_cache_limit_bytes
        keep = succeeded and cache_requested and vae_resident and (int8_mode or len(packed_blocks) == 144) and (block_bytes + global_bytes <= decode_cache_limit_bytes())
        if keep:
            _VIDEO_DECODE_CACHE = dict(key=cache_key, wrapper=wrapper, model=model, index=index, config=config, packed_blocks=packed_blocks, paths=paths, global_bytes=global_bytes, resident_paths=resident_paths, groups=groups, block_bytes=block_bytes)
            _emit(event_callback, 'vae_decoder_cached', bytes=block_bytes + global_bytes)
            if decode_offload_enabled():
                park_video_decode_cache(event_callback)
        else:
            _VIDEO_DECODE_CACHE = None
            if succeeded and cache_requested:
                _emit(event_callback, 'vae_decoder_cache_declined', bytes=block_bytes + global_bytes, reason=f'decoder must be resident, fully packed (W4A8 or INT8 ConvRot), and at most {decode_cache_limit_bytes() / 1000000000.0:.2f} GB')
        for path in [] if keep else resident_paths:
            set_module_tensor_to_device(model, path, 'meta')
        for path in [] if keep else paths:
            set_module_tensor_to_device(model, path, 'meta')
        model._decode_temporal_streaming = None
        if trace_hook is not None:
            trace_hook.remove()
        model._h3_host_output = None
        model._h3_progressive_encoder = None
        model._h3_capture_final_frame = None
        model._h3_capture_frame_count = 1
        del wrapper, model
        torch.cuda.empty_cache()
_AUDIO_COMPONENT_CACHE: dict = {}
_AUDIO_COMPONENT_LOCK = threading.Lock()

def _audio_component(component_dir: Path):
    if os.environ.get('MINIMAX_H3_VAE_AUDIO_CACHE', '0') != '1':
        return _build_meta_component(component_dir)
    key = str(Path(component_dir).resolve())
    with _AUDIO_COMPONENT_LOCK:
        hit = _AUDIO_COMPONENT_CACHE.get(key)
        if hit is None:
            hit = _build_meta_component(component_dir)
            _AUDIO_COMPONENT_CACHE[key] = hit
    return hit

def clear_audio_component_cache() -> None:
    with _AUDIO_COMPONENT_LOCK:
        _AUDIO_COMPONENT_CACHE.clear()

def _decode_audio(component_dir: Path, latent, *, device: str, event_callback=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    _t0 = time.perf_counter()
    wrapper, model, index, config = _audio_component(component_dir)
    _t_construct = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    refs = [ref for ref in index.refs.values() if not ref.name.startswith('encoder.')]
    paths, resident_bytes = _bind_resident_directstorage(model, index, refs, device=device, label='MiniMax H3 audio VAE decoder')
    _t_bind = time.perf_counter() - _t0
    _t_decode_start = time.perf_counter()
    try:
        model.eval()
        decoded_latent = _reverse_normalize(latent.to(device), config, 32, 'audio_vae')
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            waveform = model.decode(decoded_latent)
        if waveform.ndim != 3 or tuple(waveform.shape[:2]) != (2, 1):
            raise ValueError(f'H3 audio VAE returned unsupported shape {tuple(waveform.shape)}')
        _emit(event_callback, 'vae_audio_decode_timing', construct_s=round(_t_construct, 4), bind_s=round(_t_bind, 4), decode_s=round(time.perf_counter() - _t_decode_start, 4))
        return (waveform[:, 0].float().clamp(-1, 1).contiguous(), 2 * resident_bytes)
    finally:
        for path in paths:
            set_module_tensor_to_device(model, path, 'meta')
        del wrapper, model
        torch.cuda.empty_cache()

def _release_bound_component(wrapper, model, paths, *, device: str) -> None:
    import torch
    from accelerate.utils import set_module_tensor_to_device
    for path in paths:
        set_module_tensor_to_device(model, path, 'meta')
    del wrapper, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _bind_resident_safetensors(model, index, refs, *, device: str, label: str):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    refs = list(refs)
    if not refs:
        raise ValueError(f'{label} has no weights')
    state_keys = set(model.state_dict())
    missing = [_model_name(ref.name) for ref in refs if _model_name(ref.name) not in state_keys]
    if missing:
        raise ValueError(f'{label} checkpoint/model mismatch: {missing[:8]}')
    by_file: dict = {}
    for ref in refs:
        by_file.setdefault(ref.file, []).append(ref)
    paths = []
    for file, group in by_file.items():
        with safe_open(file, framework='pt', device=str(device)) as handle:
            keys = set(handle.keys())
            for ref in group:
                key = ref.name if ref.name in keys else _checkpoint_name(ref.name)
                if key not in keys:
                    raise KeyError(f'{label}: {ref.name} not found in {file}')
                path = _model_name(ref.name)
                set_module_tensor_to_device(model, path, device, value=handle.get_tensor(key))
                paths.append(path)
    torch.cuda.synchronize(device)
    return (paths, sum((ref.nbytes for ref in refs)))

def _build_video_encoder(component_dir: Path, *, device: str, profile_callback=None, transport: str='directstorage'):
    with _projection_stage(profile_callback, 'encoder_construct', device):
        wrapper, model, index, config = _build_meta_component(component_dir)
    refs = [ref for ref in index.refs.values() if _checkpoint_name(ref.name).startswith('encoder.') or _checkpoint_name(ref.name).startswith('quant_conv.')]
    with _projection_stage(profile_callback, 'encoder_bind', device):
        if transport == 'safetensors':
            paths, resident_bytes = _bind_resident_safetensors(model, index, refs, device=device, label='MiniMax H3 video VAE encoder')
        else:
            paths, resident_bytes = _bind_resident_directstorage(model, index, refs, device=device, label='MiniMax H3 video VAE encoder')
    model.eval()
    from .minimax_h3_vae_fast import install_fast_encoder
    with _projection_stage(profile_callback, 'encoder_fast_install', device):
        fast = install_fast_encoder(model)
    if fast is not None:
        from accelerate.utils import set_module_tensor_to_device
        anchor = next(iter(model.named_parameters()))[0]
        for path in paths:
            if path != anchor:
                set_module_tensor_to_device(model, path, 'meta')
        resident_bytes = fast.resident_bytes
    return (wrapper, model, config, paths, resident_bytes)

class H3PersistentVisualEncoder:

    def __init__(self, model_dir, *, device: str, transport: str='safetensors'):
        started = time.perf_counter()
        self.device = str(device)
        self.transport = transport
        import torch
        root = _component_root(model_dir)
        with torch.cuda.device(torch.device(self.device)):
            self.wrapper, self.model, self.config, self.paths, self.resident_bytes = _build_video_encoder(root / 'video_vae', device=self.device, transport=transport)
        self.lock = threading.Lock()
        self.build_seconds = time.perf_counter() - started

    def close(self) -> None:
        with self.lock:
            if self.model is not None:
                _release_bound_component(self.wrapper, self.model, self.paths, device=self.device)
                self.wrapper = self.model = None

def _visual_rows_from_latent(latent, config):
    from .minimax_h3_pipeline import prepare_h3_visual_condition_rows
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    return prepare_h3_visual_condition_rows(latent, latents_mean=config['latents_mean'], latents_std=config['latents_std'])

def h3_motion_projection_frames(model_dir):
    root = _component_root(model_dir) / 'video_vae'
    config = json.loads((root / 'config.json').read_text())
    source = json.loads((root / 'source' / 'config.json').read_text())
    ratio = math.prod(source['time_down'])
    boundary = int(config['vae_clip_length']) % ratio
    if boundary != 1 or ratio != 4:
        raise ValueError('Motion projection requires the verified H3 17-frame / stride-4 layout')
    return boundary + ratio

def encode_h3_frame_tensor(model_dir, frame, geometry, *, quantize=False, video=False, device='cuda', event_callback=None, profile_callback=None):
    import torch
    expected = (3, int(_geometry_value(geometry, 'height')), int(_geometry_value(geometry, 'width')))
    if video:
        expected = (3, h3_motion_projection_frames(model_dir), *expected[1:])
    if not isinstance(frame, torch.Tensor) or tuple(frame.shape) != expected or (not frame.is_floating_point()):
        raise ValueError(f'H3 decoded frame must be float RGB {expected}')
    wrapper, model, config, paths, resident = _build_video_encoder(_component_root(model_dir) / 'video_vae', device=device, **{'profile_callback': profile_callback} if profile_callback else {})
    started = time.perf_counter()
    try:
        with _projection_stage(profile_callback, 'encoder_input', device):
            image = frame.to(device=device, dtype=torch.float32)
            if quantize:
                image = image.clamp(0, 1).mul(255).round().to(torch.uint8).float().div(255)
        with _projection_stage(profile_callback, 'encoder_forward', device):
            with torch.inference_mode(), _seeded_vae_sample(42, next(model.parameters()).device):
                if video:
                    latents, leading = model.encode_videos(image, transform_input=True, use_fp16_latent=True, encode_prefix=True)
                    latent = latents[0]
                    if leading != [0] or latent.ndim != 4 or latent.shape[1] != 2:
                        raise ValueError(f'Unexpected H3 prefix layout: {leading}, {tuple(latent.shape)}')
                    latent = latent[:, -1:].contiguous()
                else:
                    latent = model.encode_images(image, transform_input=True, use_fp16_latent=True)[0]
        with _projection_stage(profile_callback, 'encoder_rows', device):
            rows = _visual_rows_from_latent(latent, config)
    finally:
        with _projection_stage(profile_callback, 'encoder_release', device):
            _release_bound_component(wrapper, model, paths, device=device)
    _emit(event_callback, 'vae_frame_tensor_encoded', rows=int(rows.shape[0]), video=bool(video), input_frames=int(frame.shape[1]) if video else 1, encoded_latent_t=2 if video else 1, retained_latent_t=1, quantize=bool(quantize), seconds=time.perf_counter() - started, resident_bytes=resident)
    return rows

def encode_h3_keyframes(model_dir, images, geometry, *, frame_indices, seed: int=42, device: str='cuda', event_callback=None) -> H3VisualCondition:
    import torch
    frame_indices = tuple((int(value) for value in frame_indices))
    if frame_indices not in ((0,), (-1,), (0, -1)):
        raise ValueError('H3 keyframe signature must be (0,), (-1,), or (0,-1)')
    if int(seed) != 42:
        raise ValueError('released H3 visual VAE posterior sampling is pinned to seed 42')
    prepared = _prepare_keyframe_images(images, geometry)
    if len(prepared) != len(frame_indices):
        raise ValueError('H3 keyframe image count must match frame_indices')
    root = _component_root(model_dir)
    wrapper, model, config, paths, resident = _build_video_encoder(root / 'video_vae', device=device)
    output = []
    started = time.perf_counter()
    try:
        parameter = next(model.parameters())
        for (media_item, image), frame_index in zip(prepared, frame_indices):
            with torch.inference_mode(), _seeded_vae_sample(42, parameter.device):
                latent = model.encode_images(image, use_fp16_latent=True)[0]
            rows = _visual_rows_from_latent(latent, config)
            output.append(H3VisualConditionItem(rows=rows, kind='image', latent_t=1, latent_h=int(_geometry_value(geometry, 'height')) // 16, latent_w=int(_geometry_value(geometry, 'width')) // 16, condition_index=int(media_item['condition_index']), frame_index=frame_index, material_chain='image.target_canvas'))
    finally:
        _release_bound_component(wrapper, model, paths, device=device)
    rows = torch.cat([item.rows for item in output], dim=0).contiguous()
    _emit(event_callback, 'vae_keyframes_encoded', count=len(output), rows=int(rows.shape[0]), seconds=time.perf_counter() - started, resident_bytes=resident)
    return H3VisualCondition(rows=rows, items=tuple(output), latent_t_total=len(output), seed=42)

def _video_timestamps(frame_count: int) -> tuple[float, ...]:
    sample_count = max(1, math.ceil(frame_count / 12))
    values = [index / 2.0 for index in range(sample_count)]
    if len(values) % 2:
        values.append(values[-1])
    return tuple(((values[index] + values[index + 1]) / 2 for index in range(0, len(values), 2)))

def encode_h3_reference_visuals(model_dir, media, geometry, *, seed: int=42, device: str='cuda', reference_resolution: int | None=None, event_callback=None, encoder: 'H3PersistentVisualEncoder | None'=None) -> H3VisualCondition:
    import numpy as np
    import torch
    from PIL import Image
    if int(seed) != 42:
        raise ValueError('released H3 visual VAE posterior sampling is pinned to seed 42')
    image_short_edge = H3_REFERENCE_SHORT_EDGE if reference_resolution is None else int(reference_resolution)
    if image_short_edge <= 0:
        raise ValueError('reference sampling resolution must be positive')
    ordered = _ordered_media(media, default_kind='image')
    started = time.perf_counter()

    def _reuse(item):
        done = item['encoded']
        return H3VisualConditionItem(rows=done.rows.to(device), kind=done.kind, latent_t=done.latent_t, latent_h=done.latent_h, latent_w=done.latent_w, condition_index=int(item['condition_index']), material_chain=done.material_chain, timestamps=done.timestamps)
    if all((item.get('encoded') is not None for item in ordered)):
        output = [_reuse(item) for item in ordered]
        rows = torch.cat([item.rows for item in output], dim=0).contiguous()
        total_t = sum((item.latent_t for item in output))
        _emit(event_callback, 'vae_reference_visuals_encoded', count=len(output), rows=int(rows.shape[0]), latent_t_total=total_t, seconds=time.perf_counter() - started, resident_bytes=0, prefetched=len(output), encoder_device=None)
        return H3VisualCondition(rows=rows, items=tuple(output), latent_t_total=total_t, seed=42)
    if encoder is not None:
        encoder.lock.acquire()
        model, config, resident = (encoder.model, encoder.config, 0)
        wrapper = paths = None
    else:
        root = _component_root(model_dir)
        wrapper, model, config, paths, resident = _build_video_encoder(root / 'video_vae', device=device)
    output = []
    started = time.perf_counter()
    prefetched = 0
    try:
        parameter = next(model.parameters())
        for item in ordered:
            kind = str(item['kind'])
            if item.get('encoded') is not None:
                output.append(_reuse(item))
                prefetched += 1
                continue
            if kind == 'image':
                image = _load_pil_image(item)
                if 'width' in item and 'height' in item:
                    width, height = (int(item['width']), int(item['height']))
                else:
                    width, height = _reference_image_shape(image.width, image.height, short_edge=image_short_edge)
                image = image.resize((width, height), Image.Resampling.LANCZOS)
                with torch.inference_mode(), _seeded_vae_sample(42, parameter.device):
                    latent = model.encode_images(image, use_fp16_latent=True)[0]
                timestamps = ()
                material_chain = str(item.get('material_chain', 'image.reference_preserve'))
            elif kind in ('video', 'video_audio'):
                width = int(item.get('width', _geometry_value(geometry, 'width')))
                height = int(item.get('height', _geometry_value(geometry, 'height')))
                frame_count = int(_geometry_value(geometry, 'frames'))
                frames = item.get('frames')
                if frames is None:
                    frames = _decode_reference_video_frames(item['path'], width=width, height=height, frames=frame_count, start=float(item.get('start_time_seconds', 0.0)))
                frames = np.asarray(frames)
                if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                    raise ValueError('H3 reference video frames must be [T,H,W,3] uint8')
                with torch.inference_mode(), _seeded_vae_sample(42, parameter.device):
                    latent = model.encode_videos(frames, use_fp16_latent=True)[0]
                timestamps = _video_timestamps(int(frames.shape[0]))
                material_chain = str(item.get('material_chain', f'{kind}.reference_preserve'))
            else:
                raise ValueError(f'unsupported H3 visual reference kind {kind!r}')
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            rows = _visual_rows_from_latent(latent, config).to(device)
            output.append(H3VisualConditionItem(rows=rows, kind=kind, latent_t=int(latent.shape[2]), latent_h=int(latent.shape[3]), latent_w=int(latent.shape[4]), condition_index=int(item['condition_index']), material_chain=material_chain, timestamps=timestamps))
    finally:
        if encoder is not None:
            encoder.lock.release()
        else:
            _release_bound_component(wrapper, model, paths, device=device)
    rows = torch.cat([item.rows for item in output], dim=0).contiguous()
    total_t = sum((item.latent_t for item in output))
    _emit(event_callback, 'vae_reference_visuals_encoded', count=len(output), rows=int(rows.shape[0]), latent_t_total=total_t, seconds=time.perf_counter() - started, resident_bytes=resident, prefetched=prefetched, encoder_device=None if encoder is None else encoder.device)
    return H3VisualCondition(rows=rows, items=tuple(output), latent_t_total=total_t, seed=42)

def encode_h3_source_video_item(encoder: 'H3PersistentVisualEncoder', path, *, width: int, height: int, frames: int, start: float=0.0):
    import numpy as np
    import torch
    started = time.perf_counter()
    decoded = _decode_reference_video_frames(path, width=int(width), height=int(height), frames=int(frames), start=float(start))
    decoded = np.asarray(decoded)
    with encoder.lock:
        model = encoder.model
        if model is None:
            raise RuntimeError('the session source-video encoder was closed')
        with torch.cuda.device(torch.device(encoder.device)):
            with torch.inference_mode(), _seeded_vae_sample(42, next(model.parameters()).device):
                latent = model.encode_videos(decoded, use_fp16_latent=True)[0]
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            rows = _visual_rows_from_latent(latent, encoder.config)
            torch.cuda.synchronize(encoder.device)
    item = H3VisualConditionItem(rows=rows, kind='video', latent_t=int(latent.shape[2]), latent_h=int(latent.shape[3]), latent_w=int(latent.shape[4]), condition_index=0, material_chain='video.reference_preserve', timestamps=_video_timestamps(int(decoded.shape[0])))
    return (item, time.perf_counter() - started)

def _probe_audio_rate(path) -> int:
    probe = subprocess.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate', '-of', 'default=nw=1:nk=1', str(path)], check=True, capture_output=True, text=True)
    return int(probe.stdout.strip())

def _load_reference_waveform(item, *, max_duration_seconds: float):
    import numpy as np
    import torch
    chain = str(item.get('material_chain', 'audio'))
    source_rate = int(item['source_sample_rate']) if item.get('source_sample_rate') else _probe_audio_rate(item['path'])
    if chain != 'audio':
        source_rate = 44100
    command = [shutil.which('ffmpeg') or 'ffmpeg', '-v', 'error']
    start = float(item.get('start_time_seconds', 0.0))
    if start > 0:
        command += ['-ss', f'{start:.9g}']
    command += ['-i', str(item['path']), '-map', '0:a:0', '-vn', '-ac', '2']
    if chain != 'audio':
        command += ['-ar', str(source_rate)]
    command += ['-t', f'{max_duration_seconds:.9g}', '-f', 'f32le', 'pipe:1']
    decoded = subprocess.run(command, check=True, capture_output=True)
    if not decoded.stdout or len(decoded.stdout) % 8:
        raise ValueError('ffmpeg returned invalid H3 reference-audio PCM')
    waveform = torch.from_numpy(np.frombuffer(decoded.stdout, dtype=np.float32).reshape(-1, 2).T.copy())
    return (waveform, source_rate)

@contextlib.contextmanager
def _deterministic_audio_encode():
    import torch
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.enabled, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.enabled, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old[:5]
        torch.use_deterministic_algorithms(old[5])

def encode_h3_reference_audio(model_dir, media, *, max_duration_seconds: float, device: str='cuda', event_callback=None) -> H3AudioCondition:
    import torch
    import torchaudio
    if not math.isfinite(float(max_duration_seconds)) or max_duration_seconds <= 0:
        raise ValueError('H3 reference audio duration bound must be positive')
    ordered = _ordered_media(media, default_kind='audio')
    root = _component_root(model_dir)
    wrapper, model, index, config = _build_meta_component(root / 'audio_vae')
    refs = [ref for ref in index.refs.values() if ref.name.startswith(('encoder.', 'pre_block.', 'mean_proj.'))]
    paths, resident = _bind_resident_directstorage(model, index, refs, device=device, label='MiniMax H3 audio VAE encoder')
    output = []
    started = time.perf_counter()
    try:
        model.eval()
        for item in ordered:
            kind = str(item.get('kind', 'audio'))
            default_chain = 'audio' if kind == 'audio' else f'{kind}.reference_preserve'
            chain = str(item.get('material_chain', default_chain))
            item = {**item, 'material_chain': chain}
            if item.get('has_audio') is False:
                rows = torch.empty((0, 32), dtype=torch.float32)
                output.append(H3AudioConditionItem(rows=rows, ref_audio_t=0, duration_seconds=0.0, condition_index=int(item['condition_index']), material_chain=chain))
                continue
            waveform, source_rate = _load_reference_waveform(item, max_duration_seconds=float(max_duration_seconds))
            if source_rate != H3_AUDIO_RATE:
                waveform = torchaudio.transforms.Resample(source_rate, H3_AUDIO_RATE)(waveform)
            waveform = waveform.to(device)
            with torch.inference_mode(), _deterministic_audio_encode():
                audio_data = model.preprocess(waveform.unsqueeze(1), H3_AUDIO_RATE)
                encoded = model.encoder(audio_data)
                if bool(getattr(model, 'attn_proj', False)):
                    encoded = model.pre_block(encoded.transpose(1, 2)).transpose(1, 2)
                latent = model.mean_proj(encoded).float()
            if latent.ndim != 3:
                raise ValueError('H3 audio posterior mode must be a 3-D tensor')
            if int(latent.shape[-1]) != 32:
                if int(latent.shape[1]) != 32:
                    raise ValueError(f'cannot canonicalize H3 audio latent {tuple(latent.shape)}')
                latent = latent.transpose(1, 2).contiguous()
            latent = latent.cpu()
            mean = torch.tensor(config['latents_mean']).view(1, 1, 32)
            std = torch.tensor(config['latents_std']).view(1, 1, 32)
            rows = ((latent - mean) / std).reshape(-1, 32).float().contiguous()
            output.append(H3AudioConditionItem(rows=rows, ref_audio_t=int(latent.shape[1]), duration_seconds=float(waveform.shape[-1]) / H3_AUDIO_RATE, condition_index=int(item['condition_index']), material_chain=chain))
    finally:
        _release_bound_component(wrapper, model, paths, device=device)
    rows = torch.cat([item.rows for item in output], dim=0).contiguous()
    _emit(event_callback, 'vae_reference_audio_encoded', count=len(output), rows=int(rows.shape[0]), seconds=time.perf_counter() - started, resident_bytes=resident)
    return H3AudioCondition(rows=rows, items=tuple(output))

def _run(command, *, stdin_writer=None) -> None:
    process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin_writer else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    captured = {}
    readers = []
    for stream, key in ((process.stdout, 'out'), (process.stderr, 'err')):
        thread = threading.Thread(target=lambda s=stream, k=key: captured.__setitem__(k, s.read()), daemon=True)
        thread.start()
        readers.append(thread)
    write_error = None
    try:
        if stdin_writer:
            assert process.stdin is not None
            try:
                stdin_writer(process.stdin)
            except (BrokenPipeError, OSError) as exc:
                write_error = exc
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                write_error = write_error or exc
            process.stdin = None
        process.wait()
        for thread in readers:
            thread.join()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.returncode or write_error is not None:
        write_detail = f'; stdin: {write_error}' if write_error is not None else ''
        stderr = captured.get('err', b'')
        raise RuntimeError(f"media command failed ({process.returncode}){write_detail}: {stderr.decode('utf-8', 'replace')[-2000:]}")

def _sanitize_audio_waveform(waveform):
    import torch
    source = waveform.float()
    finite = torch.isfinite(source)
    nonfinite_samples = int((~finite).sum().item())
    finite_source = torch.where(finite, source, torch.zeros_like(source))
    clipped_samples = int((finite_source.abs() > 1.0).sum().item())
    safe = torch.nan_to_num(source, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
    return (safe, {'nonfinite_samples': nonfinite_samples, 'clipped_samples': clipped_samples, 'total_samples': int(source.numel())})

def _create_mux_workdir(parent: Path, *, attempts: int=32) -> Path:
    parent = Path(parent)
    base = f'h3_mux_{os.getpid()}_{time.time_ns()}'
    last_error: OSError | None = None
    for attempt in range(max(1, int(attempts))):
        candidate = parent / (base if attempt == 0 else f'{base}_{attempt}')
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError as exc:
            last_error = exc
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    raise OSError(f'could not create a unique H3 mux work directory after {attempts} attempts under {parent}') from last_error

def _mux_h3(frames, waveform, out: Path, *, fps: int=H3_FPS, ready_video: Path | None=None) -> tuple[int, float]:
    import numpy as np
    import torch
    ffmpeg, ffprobe = (shutil.which('ffmpeg'), shutil.which('ffprobe'))
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError('ffmpeg and ffprobe are required for H3 MP4 delivery')
    if frames.ndim != 5 or tuple(frames.shape[:2]) != (1, 3):
        raise ValueError('decoded H3 frames must be [1,3,T,H,W]')
    if waveform.ndim != 2 or int(waveform.shape[0]) != H3_AUDIO_CHANNELS:
        raise ValueError('decoded H3 audio must be stereo [2,samples]')
    frame_count, height, width = (int(frames.shape[2]), int(frames.shape[3]), int(frames.shape[4]))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing H3 output: {out}')
    workdir = _create_mux_workdir(out.parent)
    video_path, audio_path = (workdir / 'video.mp4', workdir / 'audio.m4a')
    try:

        def write_frames(pipe):
            for index in range(frame_count):
                frame = frames[0, :, index]
                if frame.dtype == torch.uint8:
                    packed = frame.permute(1, 2, 0).contiguous().cpu().numpy()
                else:
                    if float(frame.amin()) < 0:
                        frame = frame.add(1).mul(0.5)
                    packed = frame.clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                pipe.write(memoryview(packed))
        encoder_probe = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True, check=True)
        if 'h264_nvenc' in encoder_probe.stdout:
            encoder_args = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq', '-rc', 'vbr', '-cq', '18', '-b:v', '0']
        else:
            encoder_args = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']

        def encode_video(args):
            _run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an', *args, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path)], stdin_writer=write_frames)
        if ready_video is not None and Path(ready_video).is_file():
            shutil.move(str(ready_video), str(video_path))
        else:
            try:
                encode_video(encoder_args)
            except RuntimeError:
                if encoder_args[1] != 'h264_nvenc':
                    raise
                video_path.unlink(missing_ok=True)
                encode_video(['-c:v', 'libx264', '-preset', 'fast', '-crf', '18'])

        def write_audio(pipe):
            chunk = 1048576
            total = int(waveform.shape[1])
            for start in range(0, total, chunk):
                interleaved = waveform[:, start:start + chunk].transpose(0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)
                pipe.write(memoryview(interleaved))
        _run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', str(H3_AUDIO_RATE), '-ac', str(H3_AUDIO_CHANNELS), '-i', 'pipe:0', '-vn', '-c:a', 'aac', '-ar', str(H3_AUDIO_RATE), '-ac', '2', str(audio_path)], stdin_writer=write_audio)
        _run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(video_path), '-i', str(audio_path), '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', '-movflags', '+faststart', str(out)])
        probe = subprocess.run([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(out)], check=True, capture_output=True, text=True)
        doc = json.loads(probe.stdout)
        video = next((s for s in doc['streams'] if s.get('codec_type') == 'video'), None)
        audio = next((s for s in doc['streams'] if s.get('codec_type') == 'audio'), None)
        if not video or not audio:
            raise ValueError('H3 output must contain one video and one audio stream')
        if video.get('codec_name') != 'h264' or video.get('pix_fmt') != 'yuv420p':
            raise ValueError('H3 video output is not H264 yuv420p')
        if audio.get('codec_name') != 'aac' or int(audio.get('sample_rate', 0)) != H3_AUDIO_RATE:
            raise ValueError('H3 audio output is not AAC 32kHz')
        if int(audio.get('channels', 0)) != H3_AUDIO_CHANNELS:
            raise ValueError('H3 audio output is not stereo')

        def stream_duration(stream, fallback):
            raw = stream.get('duration')
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return fallback
            return value if math.isfinite(value) and value > 0 else fallback
        video_duration = stream_duration(video, frame_count / fps)
        audio_duration = stream_duration(audio, int(waveform.shape[1]) / H3_AUDIO_RATE)
        drift = abs(video_duration - audio_duration)
        if drift > H3_AV_DRIFT_TOLERANCE_S:
            raise ValueError(f'H3 A/V drift {drift:.3f}s exceeds 0.25s')
        return (frame_count, drift)
    finally:
        for path in (video_path, audio_path):
            path.unlink(missing_ok=True)
        try:
            workdir.rmdir()
        except OSError:
            pass

def decode_av_h3(video_dir, audio_dir, video_latent, audio_latent, geometry, *, device, event_callback=None, progressive_path=None, decoded_frame_callback=None, decoded_frame_count=1):
    import torch
    decode_started = time.perf_counter()
    frames, video_peak, ready_video = _decode_video(video_dir, video_latent, geometry, device=device, event_callback=event_callback, progressive_path=progressive_path, **{'decoded_frame_callback': decoded_frame_callback, 'decoded_frame_count': decoded_frame_count} if decoded_frame_callback else {})
    waveform, audio_peak = _decode_audio(audio_dir, audio_latent, device=device, event_callback=event_callback)
    waveform, audio_repair = _sanitize_audio_waveform(waveform)
    if audio_repair['nonfinite_samples'] or audio_repair['clipped_samples']:
        _emit(event_callback, 'vae_audio_sanitized', **audio_repair)
    decode_seconds = time.perf_counter() - decode_started
    peaks = {}
    try:
        peaks = {'peak_alloc_gb': round(torch.cuda.max_memory_allocated(device) / 1000000000.0, 2), 'peak_reserved_gb': round(torch.cuda.max_memory_reserved(device) / 1000000000.0, 2), 'host_output': _vae_host_output_enabled(), 'frames_device': str(frames.device)}
    except Exception:
        pass
    _emit(event_callback, 'vae_decode_completed', seconds=decode_seconds, video_frames=int(frames.shape[2]), audio_samples=int(waveform.shape[1]), **peaks)
    return (frames, waveform, ready_video, video_peak, audio_peak, decode_seconds)

def decode_and_mux_h3(model_dir, video_latent, audio_latent, geometry, out, event_callback: Callable[[str, dict], None] | None=None, *, decoded_frame_callback=None, decoded_frame_count=1, trim_tail_frames: int=0, frame_validator=None, trim_head_frames: int=0) -> H3DecodeResult:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('MiniMax H3 VAE decode requires an NVIDIA CUDA device')
    if not isinstance(video_latent, torch.Tensor) or video_latent.ndim != 5:
        raise ValueError('video_latent must be a torch.Tensor [1,24,T,H,W]')
    if tuple(video_latent.shape[:2]) != (1, 24):
        raise ValueError('video_latent must have shape [1,24,T,H,W]')
    audio_latent = _normalise_audio_latent(audio_latent)
    root = _component_root(model_dir)
    video_dir, audio_dir = (root / 'video_vae', root / 'audio_vae')
    override = os.environ.get('MINIMAX_H3_VIDEO_VAE_DIR', '').strip()
    if override:
        candidate = Path(override)
        if not candidate.is_dir():
            raise FileNotFoundError(f'MINIMAX_H3_VIDEO_VAE_DIR points at {candidate}, which is not a directory')
        video_dir = candidate
    if not video_dir.is_dir():
        raise FileNotFoundError(f'MiniMax H3 video VAE is not complete: {video_dir}')
    if not audio_dir.is_dir():
        raise FileNotFoundError(f'MiniMax H3 audio VAE is not complete: {audio_dir}')
    output = Path(out)
    device = str(video_latent.device) if video_latent.is_cuda else 'cuda'
    _emit(event_callback, 'vae_decode_started', device=device)
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass
    progressive_path = output.with_name(output.stem + '.progressive.mp4')
    progressive_path.unlink(missing_ok=True)
    frames, waveform, ready_video, video_peak, audio_peak, decode_seconds = decode_av_h3(video_dir, audio_dir, video_latent, audio_latent, geometry, device=device, event_callback=event_callback, progressive_path=progressive_path, decoded_frame_callback=decoded_frame_callback, decoded_frame_count=decoded_frame_count)
    if frame_validator is not None:
        frame_validator(frames)
    mux_started = time.perf_counter()
    if trim_head_frames:
        if ready_video is not None:
            raise RuntimeError('H3 prefix trimming requires non-progressive decode')
        if int(trim_head_frames) >= int(frames.shape[2]):
            raise ValueError('H3 prefix trimming would leave no frames')
        frames = frames[:, :, int(trim_head_frames):].contiguous()
        audio_start = int(round(int(trim_head_frames) * H3_AUDIO_RATE / H3_FPS))
        waveform = waveform[:, audio_start:].contiguous()
    if trim_tail_frames:
        if ready_video is not None:
            raise RuntimeError('clipback cannot trim a progressively-encoded clip: the video is already written. Turn off MINIMAX_H3_PROGRESSIVE_VIDEO to use it.')
        keep = int(frames.shape[2]) - int(trim_tail_frames)
        if keep < 1:
            raise ValueError(f'clipback would leave {keep} frames; the clip is too short to trim')
        frames = frames[:, :, :keep].contiguous()
        audio_keep = int(round(keep * H3_AUDIO_RATE / H3_FPS))
        if waveform.shape[1] > audio_keep:
            waveform = waveform[:, :audio_keep].contiguous()
        _emit(event_callback, 'clipback_trimmed', frames=keep, dropped=int(trim_tail_frames), audio_samples=int(waveform.shape[1]))
    frame_count, drift = _mux_h3(frames, waveform, output, fps=H3_FPS, ready_video=ready_video)
    mux_seconds = time.perf_counter() - mux_started
    _emit(event_callback, 'vae_mux_completed', output=str(output), seconds=mux_seconds, bytes=output.stat().st_size, av_drift_seconds=drift)
    return H3DecodeResult(output_path=output, width=int(_geometry_value(geometry, 'width')), height=int(_geometry_value(geometry, 'height')), frames=frame_count, fps=H3_FPS, audio_samples=int(waveform.shape[1]), audio_sample_rate=H3_AUDIO_RATE, av_drift_seconds=drift, decode_seconds=decode_seconds, mux_seconds=mux_seconds, peak_decoder_weight_bytes=max(video_peak, audio_peak))
__all__ = ['H3AudioCondition', 'H3AudioConditionItem', 'H3DecodeResult', 'H3VisualCondition', 'H3VisualConditionItem', 'decode_and_mux_h3', 'clear_audio_component_cache', 'decode_av_h3', 'encode_h3_keyframes', 'encode_h3_reference_audio', 'encode_h3_reference_visuals']
