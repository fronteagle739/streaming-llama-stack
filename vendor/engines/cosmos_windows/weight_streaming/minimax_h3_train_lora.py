# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_train_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import torch
import torch.nn.functional as F
from cosmos_windows.h3_concept_dataset.assembler import validate_manifest
from .minimax_h3_lora import H3_LORA_FORMAT, H3LoRAConfig
TRAINER_FORMAT = 'minimax-h3-streamed-lora-trainer-v1'
MAX_VRAM_GB = 32.0
MAX_HOST_RAM_GB = 32.0

@dataclass(frozen=True)
class CachedRealSample:
    sample: dict[str, Any]
    cache_path: Path

@dataclass(frozen=True)
class TrainerConfig:
    steps: int = 100
    learning_rate: float = 0.0001
    checkpoint_every: int = 10
    max_vram_gb: float = 32.0
    max_host_ram_gb: float = 32.0
    seed: int = 3102026

    def validate(self) -> None:
        if not 1 <= self.steps <= 100000:
            raise ValueError('steps must be in [1, 100000]')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.01:
            raise ValueError('learning-rate must be finite and in (0, 0.01]')
        if not 1 <= self.checkpoint_every <= self.steps:
            raise ValueError('checkpoint-every must be in [1, steps]')
        if not 1 <= self.max_vram_gb <= MAX_VRAM_GB:
            raise ValueError('max-vram-gb must be in [1, 32]')
        if not 1 <= self.max_host_ram_gb <= MAX_HOST_RAM_GB:
            raise ValueError('max-host-ram-gb must be in [1, 32]')

def emit_json(event: str, callback: Callable[[dict[str, Any]], None] | None=None, **payload: Any) -> dict[str, Any]:
    record = {'event': event, 'time': time.time(), **payload}
    if callback is None:
        print(json.dumps(record, sort_keys=True), flush=True)
    else:
        callback(record)
    return record

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def validate_training_manifest(path: str | Path, *, split: str='train', require_h3_geometry: bool=True) -> list[dict[str, Any]]:
    manifest = validate_manifest(Path(path))
    samples = [sample for sample in manifest['samples'] if sample['split'] == split]
    if not samples:
        raise ValueError(f'manifest has no {split} samples')
    if require_h3_geometry:
        from .minimax_h3_pipeline import h3_latent_geometry
        for sample in samples:
            media = sample['media']
            try:
                h3_latent_geometry(float(media['duration_seconds']), int(media['width']), int(media['height']))
            except ValueError as exc:
                raise ValueError(f"sample {sample['id']} is not H3-trainable: {exc}") from exc
    return samples

def only_lora_parameters(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [name for name, _ in named if not name.endswith(('.lora_A', '.lora_B'))]
    if not named or invalid:
        raise RuntimeError(f'only LoRA parameters may train; invalid={invalid[:8]}')
    return tuple((parameter for _, parameter in named))

def rectified_flow_pair(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.shape != noise.shape:
        raise ValueError('clean/noise shapes differ')
    if sigma.numel() != 1 or not 0 < float(sigma) < 1:
        raise ValueError('sigma must be scalar in (0, 1)')
    sigma = sigma.to(device=clean.device, dtype=clean.dtype)
    return (clean * (1 - sigma) + noise * sigma, clean - noise)

def arm_reverse_directstorage(streamer: Any) -> None:
    if not hasattr(streamer, 'new_pass'):
        raise TypeError('streamer has no new_pass(reverse=...) contract')
    streamer.new_pass(reverse=True)

def train_step(model: torch.nn.Module, streamer: Any, optimizer: torch.optim.Optimizer, inputs: torch.Tensor, target: torch.Tensor, *, max_grad_norm: float=1.0) -> dict[str, float]:
    parameters = only_lora_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = F.mse_loss(prediction.float(), target.float())
    arm_reverse_directstorage(streamer)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad_norm)):
        raise RuntimeError('non-finite streamed LoRA training state')
    optimizer.step()
    return {'loss': float(loss.detach()), 'grad_norm': float(grad_norm)}

def build_real_streamed_model(model_dir: str | Path, lora_config: H3LoRAConfig, *, device='cuda:0'):
    from .minimax_h3_backend import build_streamed_backward_h3_transformer
    from .minimax_h3_lora import inject_h3_lora
    from .minimax_h3_native import build_meta_h3_native_model
    from .safetensors_index import build_index
    root = Path(model_dir)
    transformer = root / 'transformer' if (root / 'transformer').is_dir() else root
    model = build_meta_h3_native_model(transformer / 'config.json')
    model, streamer = build_streamed_backward_h3_transformer(model, build_index(transformer), device=device)
    replaced = inject_h3_lora(model, lora_config, device=device)
    only_lora_parameters(model)
    return (model, streamer, replaced)

def export_inference_safetensors(path: str | Path, model: torch.nn.Module, *, lora_config: H3LoRAConfig, dataset_sha256: str, step: int) -> Path:
    from safetensors.torch import save_file
    state = {name: parameter.detach().float().cpu().contiguous() for name, parameter in model.named_parameters() if name.endswith(('.lora_A', '.lora_B'))}
    if not state:
        raise ValueError('model has no LoRA parameters')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, destination, metadata={'format': H3_LORA_FORMAT, 'config': json.dumps(asdict(lora_config), sort_keys=True), 'dataset_sha256': dataset_sha256, 'step': str(int(step))})
    return destination

def save_checkpoint(directory: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, lora_config: H3LoRAConfig, dataset_sha256: str, step: int, generator: torch.Generator | None=None) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter = export_inference_safetensors(directory / f'adapter-step-{step:06d}.safetensors', model, lora_config=lora_config, dataset_sha256=dataset_sha256, step=step)
    trainer = directory / f'trainer-step-{step:06d}.pt'
    torch.save({'format': TRAINER_FORMAT, 'step': int(step), 'dataset_sha256': dataset_sha256, 'adapter': adapter.name, 'optimizer': optimizer.state_dict(), 'generator_state': None if generator is None else generator.get_state()}, trainer)
    (directory / 'latest.json').write_text(json.dumps({'format': TRAINER_FORMAT, 'step': int(step), 'dataset_sha256': dataset_sha256, 'adapter': adapter.name, 'trainer': trainer.name}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return (adapter, trainer)

def _load_adapter(path: Path, model: torch.nn.Module, *, dataset_sha256: str) -> int:
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
    if metadata.get('format') != H3_LORA_FORMAT:
        raise ValueError('unsupported H3 LoRA safetensors format')
    if metadata.get('dataset_sha256') != dataset_sha256:
        raise ValueError('dataset manifest changed since checkpoint')
    state = load_file(path, device='cpu')
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError('LoRA checkpoint keys do not match model')
    modules = dict(model.named_modules())
    for name, value in state.items():
        module_name, leaf = name.rsplit('.', 1)
        parameter = getattr(modules[module_name], leaf)
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return int(metadata['step'])

def resume_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, dataset_sha256: str, generator: torch.Generator | None=None) -> int:
    path = Path(path)
    if path.is_dir():
        latest = json.loads((path / 'latest.json').read_text(encoding='utf-8'))
        path = path / latest['trainer']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != TRAINER_FORMAT:
        raise ValueError('unsupported H3 trainer checkpoint')
    if payload.get('dataset_sha256') != dataset_sha256:
        raise ValueError('dataset manifest changed since checkpoint')
    step = _load_adapter(path.parent / payload['adapter'], model, dataset_sha256=dataset_sha256)
    optimizer.load_state_dict(payload['optimizer'])
    device = only_lora_parameters(model)[0].device
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    if generator is not None and payload.get('generator_state') is not None:
        generator.set_state(payload['generator_state'])
    return step

def _process_rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except Exception:
        return 0.0

def run_streamed_training(model: torch.nn.Module, streamer: Any, batch_factory: Callable[[int, torch.Generator], tuple[torch.Tensor, torch.Tensor]], optimizer: torch.optim.Optimizer, *, config: TrainerConfig, lora_config: H3LoRAConfig, dataset_sha256: str, output_dir: str | Path, resume: str | Path | None=None, callback: Callable[[dict[str, Any]], None] | None=None) -> Path:
    config.validate()
    parameters = only_lora_parameters(model)
    device = parameters[0].device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    start = 0
    if resume is not None:
        start = resume_checkpoint(resume, model, optimizer, dataset_sha256=dataset_sha256, generator=generator)
    if start >= config.steps:
        raise ValueError('resume checkpoint has already reached requested steps')
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        fraction = min(1.0, config.max_vram_gb * 1024 ** 3 / properties.total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    emit_json('training_start', callback, start_step=start, steps=config.steps, max_vram_gb=config.max_vram_gb, max_host_ram_gb=config.max_host_ram_gb)
    adapter = None
    for step in range(start + 1, config.steps + 1):
        inputs, target = batch_factory(step, generator)
        started = time.perf_counter()
        metrics = train_step(model, streamer, optimizer, inputs, target)
        rss = _process_rss_gb()
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0
        if rss > config.max_host_ram_gb or peak > config.max_vram_gb:
            raise RuntimeError(f'training memory bound exceeded: host={rss:.2f}GiB cuda={peak:.2f}GiB')
        emit_json('train_step', callback, step=step, steps=config.steps, seconds=time.perf_counter() - started, host_rss_gb=rss, peak_vram_gb=peak, **metrics)
        if step % config.checkpoint_every == 0 or step == config.steps:
            adapter, trainer = save_checkpoint(output_dir, model, optimizer, lora_config=lora_config, dataset_sha256=dataset_sha256, step=step, generator=generator)
            emit_json('checkpoint_saved', callback, step=step, adapter=str(adapter), trainer=str(trainer))
    assert adapter is not None
    emit_json('training_complete', callback, steps=config.steps, adapter=str(adapter))
    return adapter

def _tiny_model(seed: int=7):
    from ..streamed_backward import apply_streamed_multi
    generator = torch.Generator().manual_seed(seed)
    residency = {'current': 0, 'peak': 0, 'binds': [], 'evicts': []}

    class Block(torch.nn.Module):

        def __init__(self, index: int):
            super().__init__()
            self.index = index
            self.register_buffer('base_cpu', torch.randn(8, 8, generator=generator))
            self.lora_A = torch.nn.Parameter(torch.randn(2, 8, generator=generator) * 0.1)
            self.lora_B = torch.nn.Parameter(torch.randn(8, 2, generator=generator) * 0.1)
            self.base = None

        def bind(self):
            if self.base is not None:
                raise RuntimeError('block already resident')
            self.base = self.base_cpu
            residency['current'] += 1
            residency['peak'] = max(residency['peak'], residency['current'])
            residency['binds'].append(self.index)

        def evict(self):
            self.base = None
            residency['current'] -= 1
            residency['evicts'].append(self.index)

        def raw(self, value):
            return value + F.linear(value, self.base) + F.linear(F.linear(value, self.lora_A), self.lora_B)

        def forward(self, value):
            return apply_streamed_multi(self, lambda layer, hidden: layer.raw(hidden), value, bind=self.bind, evict=self.evict)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList((Block(0), Block(1)))

        def forward(self, value):
            for block in self.blocks:
                value = block(value)
            return value

    class Streamer:

        def __init__(self):
            self.reverse_calls = 0

        def new_pass(self, reverse=False):
            if reverse:
                self.reverse_calls += 1
    return (Model(), Streamer(), residency, generator)

def tiny_gradient_checkpoint_test(output: str | Path | None=None) -> dict[str, Any]:
    model, streamer, residency, generator = _tiny_model()
    optimizer = torch.optim.AdamW(only_lora_parameters(model), lr=0.01)
    inputs = torch.randn(4, 8, generator=generator)
    target = torch.randn(4, 8, generator=generator)
    metrics = train_step(model, streamer, optimizer, inputs, target)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    resumed = False
    if output is not None:
        lora_config = H3LoRAConfig(rank=2, alpha=2.0, targets=('tiny',))
        save_checkpoint(output, model, optimizer, lora_config=lora_config, dataset_sha256='tiny-dataset', step=1, generator=generator)
        restored, _, _, restored_generator = _tiny_model()
        restored_optimizer = torch.optim.AdamW(only_lora_parameters(restored), lr=0.01)
        step = resume_checkpoint(output, restored, restored_optimizer, dataset_sha256='tiny-dataset', generator=restored_generator)
        resumed = step == 1 and all((torch.equal(before[name], value) for name, value in restored.named_parameters()))
    return {**metrics, 'reverse_calls': streamer.reverse_calls, 'peak_base_blocks': residency['peak'], 'base_blocks_remaining': residency['current'], 'forward_bind_order': residency['binds'][:2], 'backward_bind_order': residency['binds'][2:], 'checkpoint_resumed': resumed}

def _decode_rgb_video(path: Path, *, width: int, height: int, frames: int) -> torch.Tensor:
    import subprocess
    command = ['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'fps=24,scale={width}:{height}:flags=lanczos', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    result = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height * 3
    count, remainder = divmod(len(result.stdout), frame_bytes)
    if count < 1 or remainder or count > frames:
        raise RuntimeError(f'invalid decoded frame payload for {path}: {count} frames')
    values = torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8)
    video = values.view(count, height, width, 3)
    if count < frames:
        video = torch.cat((video, video[-1:].expand(frames - count, -1, -1, -1)), dim=0)
    return video.permute(3, 0, 1, 2).unsqueeze(0).float().div_(255)

def _encode_one_video(model_dir: Path, video_path: Path, *, width: int, height: int, frames: int, device: str) -> torch.Tensor:
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .minimax_h3_pipeline import patchify_h3_video
    from .minimax_h3_vae_mux import _build_meta_component, _model_name
    _wrapper, vae, index, config = _build_meta_component(model_dir / 'video_vae')
    refs = [ref for ref in index.refs.values() if ref.name.startswith('encoder.') or ref.name.startswith('quant_conv.')]
    keys = set(vae.state_dict())
    backend = PortableBackend(pin_memory=True)
    try:
        for ref in refs:
            name = _model_name(ref.name)
            if name not in keys:
                raise ValueError(f'VAE encoder key mismatch: {name}')
            set_module_tensor_to_device(vae, name, device, value=backend.read_tensor(ref, device=device, direct=True))
    finally:
        backend.close()
    video = _decode_rgb_video(video_path, width=width, height=height, frames=frames)
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1)
    latent_mean = torch.tensor(config['latents_mean'], device=device).view(1, 24, 1, 1, 1)
    latent_std = torch.tensor(config['latents_std'], device=device).view(1, 24, 1, 1, 1)
    with torch.inference_mode():
        moments = vae.encode_temporal((video.to(device) - pixel_mean) / pixel_std)
        latent = moments.float().chunk(2, dim=1)[0]
        rows = patchify_h3_video((latent - latent_mean) / latent_std).float().cpu()
    del vae
    torch.cuda.empty_cache()
    return rows

def select_training_samples(samples: Sequence[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if not 1 <= count <= len(samples):
        raise ValueError(f'samples must be in [1, {len(samples)}]')
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    return [samples[index] for index in order[:count]]

def sample_for_step(samples: Sequence[Any], step: int) -> Any:
    if not samples or step < 1:
        raise ValueError('step cycling requires samples and a positive step')
    return samples[(step - 1) % len(samples)]

def prepare_one_real_sample(model_dir: str | Path, manifest_path: str | Path, *, device: str, sample: dict[str, Any] | None=None, callback=None) -> dict[str, Any]:
    from .minimax_h3_pipeline import h3_latent_geometry
    from .minimax_h3_text_encoder import encode_h3_text_prompt
    manifest_path = Path(manifest_path).resolve()
    sample = sample or validate_training_manifest(manifest_path)[0]
    media = sample['media']
    geometry = h3_latent_geometry(media['duration_seconds'], media['width'], media['height'])
    emit_json('sample_prepare_start', callback, sample_id=sample['id'])
    prompt = encode_h3_text_prompt(Path(model_dir) / 'text_encoder', sample['caption'], device=device, event_callback=lambda event, payload: emit_json(event, callback, **payload)).cpu()
    torch.cuda.empty_cache()
    emit_json('prompt_cache_ready', callback, tokens=prompt.shape[0], host_rss_gb=_process_rss_gb())
    rows = _encode_one_video(Path(model_dir), manifest_path.parent / sample['video'], width=media['width'], height=media['height'], frames=geometry.frames, device=device)
    if tuple(rows.shape) != (geometry.video_rows, 96):
        raise RuntimeError(f'VAE produced {tuple(rows.shape)}, expected {(geometry.video_rows, 96)}')
    emit_json('video_cache_ready', callback, rows=rows.shape[0], host_rss_gb=_process_rss_gb())
    return {'sample': sample, 'geometry': geometry, 'prompt': prompt, 'video_rows': rows}

def write_real_sample_cache(path: str | Path, *, prompt: torch.Tensor, video_rows: torch.Tensor, manifest_sha256: str, sample_id: str) -> Path:
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({'prompt': prompt.detach().to(device='cpu', dtype=torch.bfloat16).contiguous(), 'video_rows': video_rows.detach().to(device='cpu', dtype=torch.float16).contiguous()}, path, metadata={'format': TRAINER_FORMAT, 'manifest_sha256': manifest_sha256, 'sample_id': sample_id})
    return path

def real_sample_cache_valid(path: str | Path, *, manifest_sha256: str, sample_id: str) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        from safetensors import safe_open
        with safe_open(path, framework='pt', device='cpu') as handle:
            metadata = handle.metadata()
            keys = set(handle.keys())
        return metadata.get('format') == TRAINER_FORMAT and metadata.get('manifest_sha256') == manifest_sha256 and (metadata.get('sample_id') == sample_id) and (keys == {'prompt', 'video_rows'})
    except Exception:
        return False

def prepare_real_sample_cache(model_dir: str | Path, manifest_path: str | Path, *, cache_dir: str | Path, device: str, count: int, seed: int, max_host_ram_gb: float, callback=None) -> list[CachedRealSample]:
    manifest_path = Path(manifest_path).resolve()
    selected = select_training_samples(validate_training_manifest(manifest_path), count=count, seed=seed)
    digest = file_sha256(manifest_path)
    cache_dir = Path(cache_dir)
    cached: list[CachedRealSample] = []
    for sample in selected:
        path = cache_dir / f"{sample['id']}.safetensors"
        if real_sample_cache_valid(path, manifest_sha256=digest, sample_id=sample['id']):
            emit_json('sample_cache_hit', callback, sample_id=sample['id'], path=str(path))
        else:
            prepared = prepare_one_real_sample(model_dir, manifest_path, device=device, sample=sample, callback=callback)
            write_real_sample_cache(path, prompt=prepared['prompt'], video_rows=prepared['video_rows'], manifest_sha256=digest, sample_id=sample['id'])
            emit_json('sample_cache_written', callback, sample_id=sample['id'], path=str(path), bytes=path.stat().st_size)
        rss = _process_rss_gb()
        if rss > max_host_ram_gb:
            raise RuntimeError(f'sample cache exceeded host RAM bound: {rss:.2f} GiB')
        cached.append(CachedRealSample(sample, path))
    emit_json('dataset_cache_ready', callback, samples=len(cached), bytes=sum((item.cache_path.stat().st_size for item in cached)))
    return cached

def run_real_training(*, model_dir: str | Path, manifest: str | Path, output_dir: str | Path, device: str='cuda:1', rank: int=4, samples: int=6, steps: int=30, learning_rate: float=0.0001, checkpoint_every: int=5, seed: int=3102026, cache_dir: str | Path | None=None, max_vram_gb: float=32.0, max_host_ram_gb: float=32.0, resume: str | Path | None=None, callback=None) -> Path:
    from .minimax_h3_lora import _load_h3_lora_artifact
    from .minimax_h3_pipeline import build_h3_t2v_packed_layout, h3_latent_geometry
    from safetensors.torch import load_file
    config = TrainerConfig(steps=steps, learning_rate=learning_rate, checkpoint_every=checkpoint_every, max_vram_gb=max_vram_gb, max_host_ram_gb=max_host_ram_gb, seed=seed)
    config.validate()
    if not 1 <= rank <= 128:
        raise ValueError('rank must be in [1, 128]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for real H3 training')
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    total_memory = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gb * 1024 ** 3 / total_memory), dev)
    torch.cuda.reset_peak_memory_stats(dev)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / 'sample-cache'
    cached = prepare_real_sample_cache(model_dir, manifest, cache_dir=cache_dir, device=device, count=samples, seed=seed, max_host_ram_gb=max_host_ram_gb, callback=callback)
    lora_config = H3LoRAConfig(rank=rank, alpha=float(rank))
    emit_json('dit_build_start', callback, rank=rank)
    model, streamer, replaced = build_real_streamed_model(model_dir, lora_config, device=device)
    parameters = only_lora_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    generator = torch.Generator(device=dev).manual_seed(seed)
    dataset_digest = file_sha256(manifest)
    start_step = 0
    if resume is not None:
        start_step = resume_checkpoint(resume, model, optimizer, dataset_sha256=dataset_digest, generator=generator)
        if start_step >= steps:
            raise ValueError('resume artifact already reached requested training step')
        emit_json('training_resumed', callback, step=start_step, resume=str(resume))
    emit_json('training_start', callback, start_step=start_step, steps=steps, samples=len(cached), seed=seed, checkpoint_every=checkpoint_every, trainable_tensors=len(parameters), injected_modules=len(replaced))
    adapter = None
    try:
        for step in range(start_step + 1, steps + 1):
            cached_sample = sample_for_step(cached, step)
            tensors = load_file(cached_sample.cache_path, device='cpu')
            sample = cached_sample.sample
            media = sample['media']
            geometry = h3_latent_geometry(media['duration_seconds'], media['width'], media['height'])
            clean_video = tensors['video_rows'].to(dev, dtype=torch.float32)
            prompt = tensors['prompt'].to(dev, dtype=torch.bfloat16)
            clean_audio = torch.zeros((geometry.audio_rows, 32), device=dev)
            sigma = torch.rand((), generator=generator, device=dev).mul_(0.9).add_(0.05)
            video_state, video_target = rectified_flow_pair(clean_video, torch.randn(clean_video.shape, generator=generator, device=dev), sigma)
            audio_state, _ = rectified_flow_pair(clean_audio, torch.randn(clean_audio.shape, generator=generator, device=dev), sigma)
            layout = build_h3_t2v_packed_layout(geometry, text_len=prompt.shape[0])
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            emit_json('train_step_start', callback, step=step, sample_id=sample['id'])
            prediction, _audio_prediction = model(video_rows=video_state, audio_rows=audio_state, prompt_embeds=prompt, video_timestep=1 - float(sigma), audio_timestep=1 - float(sigma), packed_layout=layout)
            loss = F.mse_loss(prediction.float(), video_target.float())
            arm_reverse_directstorage(streamer)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f'non-finite H3 training state at step {step}')
            optimizer.step()
            peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
            rss = _process_rss_gb()
            if peak > max_vram_gb or rss > max_host_ram_gb:
                raise RuntimeError(f'training memory bound exceeded: VRAM={peak:.2f} RSS={rss:.2f}')
            emit_json('train_step', callback, step=step, steps=steps, sample_id=sample['id'], sigma=float(sigma), loss=float(loss.detach()), grad_norm=float(grad_norm), seconds=time.perf_counter() - started, peak_vram_gb=peak, host_rss_gb=rss)
            if step % checkpoint_every == 0 or step == steps:
                adapter, trainer = save_checkpoint(output_dir, model, optimizer, lora_config=lora_config, dataset_sha256=dataset_digest, step=step, generator=generator)
                emit_json('checkpoint_saved', callback, step=step, adapter=str(adapter), trainer=str(trainer))
        assert adapter is not None
        payload = _load_h3_lora_artifact(adapter)
        expected = {name for name, _ in model.named_parameters() if name.endswith(('.lora_A', '.lora_B'))}
        if set(payload['lora']) != expected:
            raise RuntimeError('inference loader roundtrip key mismatch')
        emit_json('training_complete', callback, adapter=str(adapter), steps=steps, inference_loader_roundtrip=True, peak_vram_gb=torch.cuda.max_memory_allocated(dev) / 1024 ** 3, host_rss_gb=_process_rss_gb())
        return adapter
    finally:
        streamer.close()
        del model
        torch.cuda.empty_cache()

def run_real_one_step_smoke(**kwargs) -> Path:
    kwargs.setdefault('samples', 1)
    kwargs.setdefault('steps', 1)
    kwargs.setdefault('rank', 1)
    kwargs.setdefault('checkpoint_every', 1)
    return run_real_training(**kwargs)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    tiny = commands.add_parser('tiny-test', help='CPU gradient/checkpoint proof')
    tiny.add_argument('--output', type=Path)
    validate = commands.add_parser('validate', help='validate an H3 training manifest')
    validate.add_argument('manifest', type=Path)
    validate.add_argument('--allow-non-h3-geometry', action='store_true')
    preflight = commands.add_parser('preflight', help='validate real model/data without allocating CUDA')
    preflight.add_argument('--model-dir', required=True, type=Path)
    preflight.add_argument('--manifest', required=True, type=Path)
    train = commands.add_parser('train', help='real packed H3 reverse-streamed training')
    train.add_argument('--model-dir', required=True, type=Path)
    train.add_argument('--manifest', required=True, type=Path)
    train.add_argument('--output', required=True, type=Path)
    train.add_argument('--device', default='cuda:1')
    train.add_argument('--samples', type=int, default=6)
    train.add_argument('--steps', type=int, default=30)
    train.add_argument('--rank', type=int, default=4)
    train.add_argument('--learning-rate', type=float, default=0.0001)
    train.add_argument('--checkpoint-every', type=int, default=5)
    train.add_argument('--seed', type=int, default=3102026)
    train.add_argument('--cache-dir', type=Path)
    train.add_argument('--max-vram-gb', type=float, default=32.0)
    train.add_argument('--max-host-ram-gb', type=float, default=32.0)
    train.add_argument('--resume', type=Path)
    return parser

def main(argv: Sequence[str] | None=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == 'tiny-test':
        emit_json('tiny_test', **tiny_gradient_checkpoint_test(args.output))
    elif args.command == 'validate':
        samples = validate_training_manifest(args.manifest, require_h3_geometry=not args.allow_non_h3_geometry)
        emit_json('manifest_valid', samples=len(samples), manifest_sha256=file_sha256(args.manifest))
    elif args.command == 'preflight':
        from .minimax_h3_backend import inspect_h3_transformer
        samples = validate_training_manifest(args.manifest)
        layout = inspect_h3_transformer(args.model_dir)
        if not layout.complete:
            raise RuntimeError('H3 transformer checkpoint is incomplete')
        emit_json('preflight_ready', samples=len(samples), blocks=layout.block_count, one_block_bytes=max(layout.block_bytes), max_vram_gb=MAX_VRAM_GB, max_host_ram_gb=MAX_HOST_RAM_GB)
    else:
        run_real_training(model_dir=args.model_dir, manifest=args.manifest, output_dir=args.output, device=args.device, samples=args.samples, steps=args.steps, rank=args.rank, learning_rate=args.learning_rate, checkpoint_every=args.checkpoint_every, seed=args.seed, cache_dir=args.cache_dir, max_vram_gb=args.max_vram_gb, max_host_ram_gb=args.max_host_ram_gb, resume=args.resume)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
