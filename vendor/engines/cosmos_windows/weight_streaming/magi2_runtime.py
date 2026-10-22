# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_runtime.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from cosmos_windows.weight_streaming.magi2_prompt import MAGI2_CLIP_SECONDS, mode_config_defaults, resolution_tier
EmitSink = Callable[[str], None]

class Magi2EventEmitter:

    def __init__(self, *, enabled: bool=True, debug: bool=False, sink: Optional[EmitSink]=None):
        self.enabled = enabled
        self.debug = debug
        self._sink: EmitSink = sink if sink is not None else lambda line: print(line, flush=True)
        self.events: List[dict] = []

    def emit(self, event: str, **payload) -> dict:
        obj = {'event': event, 'time': round(time.time(), 6), **payload}
        self.events.append(obj)
        if self.enabled:
            self._sink(json.dumps(obj))
        return obj

    def debug_emit(self, event: str, **payload) -> Optional[dict]:
        if not self.debug:
            return None
        return self.emit(event, level='debug', **payload)

    def stage_started(self, stage: str, **payload) -> dict:
        return self.emit('stage_log', stage=stage, message=f'{stage} started', **payload)

    def stage_timing(self, stage: str, seconds: float, **payload) -> dict:
        return self.emit('stage_timing', stage=stage, seconds=round(seconds, 3), **payload)

    def fallback(self, what: str, reason: str, **payload) -> dict:
        return self.emit('attention_backend_selected', fallback=True, what=what, message=f'FALLBACK: {what} -> {reason}', reason=reason, **payload)

@dataclass(frozen=True)
class StageSpec:
    name: str
    residency: str
    enabled: bool
    note: str = ''

def plan_stages(mode: str, resolution: str) -> List[StageSpec]:
    tier = resolution_tier(resolution)
    stages: List[StageSpec] = [StageSpec('text_encode', 'resident-evict', True, 'Qwen3.5-27B: load->encode once->evict before preview (per-layer stream if it will not fit 32GB; Mac sidecar is the documented alternative).')]
    if mode == 'i2v':
        stages.append(StageSpec('ref_image_encode', 'resident-evict', True, 'Wan2.2 VAE encodes the first frame; evicts before preview.'))
    stages.append(StageSpec('preview_denoise', 'stream', True, 'MoE preview DiT streamed as a double-buffered layer ring (~12GB ring + resident globals); every step re-streams the experts.'))
    if tier.uses_refiner:
        stages.append(StageSpec('refiner_denoise', 'resident', True, 'Refiner is dense 14GB — resident after preview evicts; windowed local attention bounds 1080p activations.'))
    stages += [StageSpec('vae_decode', 'resident-evict', True, 'Turbo VAE (video) + Stable-Audio VAE (audio), each loaded/decoded/evicted.'), StageSpec('mux', 'none', True, 'ffmpeg muxes generated stereo audio into the mp4.')]
    return stages

def stage_names(mode: str, resolution: str) -> List[str]:
    return [s.name for s in plan_stages(mode, resolution) if s.enabled]

@dataclass(frozen=True)
class BatchItem:
    prompt: str
    image_path: Optional[str]
    seed: int
    index: int

def build_batches(prompts: Sequence[str], images: Optional[Sequence[Optional[str]]], *, base_seed: int, batch_size: int) -> List[List[BatchItem]]:
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    imgs = list(images) if images is not None else [None] * len(prompts)
    if len(imgs) == 1 and len(prompts) > 1:
        imgs = imgs * len(prompts)
    if len(imgs) != len(prompts):
        raise ValueError(f'images ({len(imgs)}) must match prompts ({len(prompts)})')
    items = [BatchItem(prompt=p, image_path=img, seed=base_seed + i, index=i) for i, (p, img) in enumerate(zip(prompts, imgs))]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def free_cuda_and_gc(*objs) -> None:
    import gc
    for o in objs:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

class StageMemory:

    def __init__(self, name: str, emit: Optional['Magi2EventEmitter']=None):
        self.name = name
        self.emit = emit
        self._resident = None

    def hold(self, obj):
        self._resident = obj
        return obj

    def __enter__(self):
        if self.emit:
            self.emit.emit('stage_log', stage=self.name, message=f'{self.name}: enter (free-before-alloc)')
        return self

    def __exit__(self, *exc):
        obj = self._resident
        for meth in ('evict', 'close'):
            fn = getattr(obj, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        self._resident = None
        free_cuda_and_gc()
        if self.emit:
            self.emit.emit('stage_timing', stage=self.name, seconds=0.0, freed=True)
        return False

@dataclass(frozen=True)
class StageBudget:
    stage: str
    mechanism: str
    peak_vram_gb: float
    peak_rss_gb: float

def stage_budget_table(mode: str, resolution: str, *, ring_budget_gb: float=12.0, transport: str='directstorage', vae_budget_gb: float=8.0) -> List[StageBudget]:
    tier = resolution_tier(resolution)
    weight_rss = 0.05 if transport == 'directstorage' else 0.25
    mech_weight = 'TRUE DirectStorage NVMe->GPU DMA (D3D12 heap imported as CUDA ext-mem; ~0 pageable host RAM)' if transport == 'directstorage' else 'bounded-pinned host staging fallback (256MB reused)'
    rows: List[StageBudget] = [StageBudget('text_encode', 'Qwen3.5-27B meta-build + per-layer NVMe stream (build_streamed_text_encoder)', peak_vram_gb=2.5, peak_rss_gb=weight_rss + 0.3)]
    if mode == 'i2v':
        rows.append(StageBudget('ref_image_encode', 'Wan2.2 VAE meta-build+byte-index bind; tiled encode', peak_vram_gb=3.5, peak_rss_gb=weight_rss + 0.2))
    rows.append(StageBudget('preview_denoise', f'whole-model double-buffered layer ring via {mech_weight} + resident globals', peak_vram_gb=round(ring_budget_gb + 4.0 + 5.0 + 3.0, 1), peak_rss_gb=weight_rss + 0.5))
    if tier.uses_refiner:
        rows.append(StageBudget('refiner_denoise', f'dense 14GB streamed ({mech_weight}) / resident after preview evict', peak_vram_gb=14.0 + 6.0, peak_rss_gb=weight_rss + 0.3))
    rows.append(StageBudget('vae_decode', 'turbo/Wan VAE meta-build+bind; tiled spatial + temporal-chunk decode', peak_vram_gb=round(3.0 + vae_budget_gb, 1), peak_rss_gb=weight_rss + 0.4))
    rows.append(StageBudget('mux', 'ffmpeg subprocess (audio+video)', peak_vram_gb=0.0, peak_rss_gb=0.2))
    return rows

def assert_budgets_fit(table: Sequence[StageBudget], *, vram_total_gb: float=32.0, host_rss_cap_gb: float=7.75) -> None:
    for b in table:
        if b.peak_vram_gb > vram_total_gb:
            raise MemoryError(f'stage {b.stage!r} peak VRAM {b.peak_vram_gb:.1f}GB > {vram_total_gb:.0f}GB card (mechanism: {b.mechanism})')
        if b.peak_rss_gb > host_rss_cap_gb:
            raise MemoryError(f'stage {b.stage!r} peak RSS {b.peak_rss_gb:.2f}GB > {host_rss_cap_gb:.2f}GB host cap (mechanism: {b.mechanism}) — a load is staging pageable host RAM, forbidden on this box')

class Magi2Executors:

    def encode_text(self, item: 'BatchItem', emit: Magi2EventEmitter):
        raise NotImplementedError

    def denoise_preview(self, batch: List['BatchItem'], contexts, steps: int, emit: Magi2EventEmitter):
        raise NotImplementedError

    def denoise_refiner(self, batch, latents, contexts, steps: int, emit: Magi2EventEmitter):
        raise NotImplementedError

    def decode_and_mux(self, item: 'BatchItem', latent, out_path: Path, emit: Magi2EventEmitter) -> Path:
        raise NotImplementedError

@dataclass
class EngineResult:
    outputs: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    seconds: float = 0.0

def run_stream_sdpa_engine(*, mode: str, resolution: str, batches: List[List[BatchItem]], steps: int, refiner_steps: Optional[int], executors: Magi2Executors, emit: Magi2EventEmitter, out_paths: List[Path]) -> EngineResult:
    t0 = time.time()
    tier = resolution_tier(resolution)
    plan = plan_stages(mode, resolution)
    result = EngineResult(stages=[s.name for s in plan if s.enabled])
    emit.emit('denoise_started', device='cuda', stage='preview', engine='stream-sdpa', resolution=resolution, mode=mode, batches=len(batches), stages=result.stages)
    out_idx = 0
    for bi, batch in enumerate(batches):
        emit.emit('text_encoding_started', device='cuda', batch=bi, items=len(batch))
        ts = time.time()
        contexts = [executors.encode_text(item, emit) for item in batch]
        emit.emit('text_encoding_complete', batch=bi)
        emit.stage_timing('text_encode', time.time() - ts, batch=bi)
        ts = time.time()
        latents = executors.denoise_preview(batch, contexts, steps, emit)
        emit.stage_timing('preview_denoise', time.time() - ts, batch=bi, steps=steps)
        if tier.uses_refiner and refiner_steps:
            ts = time.time()
            latents = executors.denoise_refiner(batch, latents, contexts, refiner_steps, emit)
            emit.stage_timing('refiner_denoise', time.time() - ts, batch=bi, steps=refiner_steps)
        emit.emit('vae_decode_started', batch=bi)
        for k, item in enumerate(batch):
            latent = latents[k] if isinstance(latents, (list, tuple)) else latents
            path = executors.decode_and_mux(item, latent, out_paths[out_idx], emit)
            emit.emit('vae_mux_completed', output=str(path), index=out_idx)
            result.outputs.append(str(path))
            out_idx += 1
    result.seconds = round(time.time() - t0, 2)
    emit.emit('completed', outputs=result.outputs, count=len(result.outputs), seconds=result.seconds)
    return result

@dataclass
class DrySmokeReport:
    ok: bool
    device: Optional[str] = None
    ring_depth: int = 0
    layers_built: int = 0
    attn_out_shape: Optional[tuple] = None
    peak_vram_bytes: int = 0
    pinned_bytes: int = 0
    rss_delta_mib: float = 0.0
    rss_flat: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {'ok': self.ok, 'device': self.device, 'ring_depth': self.ring_depth, 'layers_built': self.layers_built, 'attn_out_shape': self.attn_out_shape, 'peak_vram_bytes': self.peak_vram_bytes, 'pinned_bytes': self.pinned_bytes, 'rss_delta_mib': round(self.rss_delta_mib, 1), 'rss_flat': self.rss_flat, 'notes': list(self.notes)}

def gpu_dry_smoke(*, num_layers: int=6, layer_bytes: int=64 * 1024 * 1024, seq_len: int=512, num_heads: int=24, head_dim: int=128, ring_budget_bytes: int=256 * 1024 * 1024, emit: Optional[Magi2EventEmitter]=None) -> DrySmokeReport:
    from cosmos_windows.weight_streaming.magi2_layer_ring import FakeLayerSource, LayerPlan, Magi2LayerRingPlan, Magi2LayerRingStreamer, dry_run_pass, _aln
    from cosmos_windows.weight_streaming.safetensors_index import TensorRef
    report = DrySmokeReport(ok=False)
    layers = []
    for i in range(num_layers):
        ref = TensorRef(name=f'block.layers.{i}.fake.weight', file='<fake>', offset=0, nbytes=layer_bytes, dtype='BF16', shape=(layer_bytes // 2,))
        layers.append(LayerPlan(layer_idx=i, is_moe=2 <= i < num_layers - 2, nbytes=layer_bytes, packed_nbytes=_aln(layer_bytes), tensor_count=1, refs=(ref,)))
    depth = max(2, ring_budget_bytes // _aln(layer_bytes))
    depth = min(depth, num_layers)
    plan = Magi2LayerRingPlan(layers=tuple(layers), global_bytes=0, global_tensor_count=0, vram_budget_bytes=ring_budget_bytes, ring_depth=depth, ring_bytes=depth * _aln(layer_bytes))
    sched = dry_run_pass(plan, steps=1)
    report.ring_depth = depth
    report.notes.append(f'schedule: {sched.layers_streamed} layer-streams, {sched.bytes_streamed} bytes, max_resident={sched.max_resident_layers}')
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        report.notes.append(f'torch unavailable: {exc}; ran schedule-only dry run')
        if emit:
            emit.fallback('gpu_dry_smoke', f'torch unavailable: {exc}')
        return report
    if not torch.cuda.is_available():
        report.notes.append('CUDA unavailable; ran schedule-only dry run')
        if emit:
            emit.fallback('gpu_dry_smoke', 'CUDA unavailable')
        return report
    device = 'cuda'
    report.device = device
    torch.cuda.reset_peak_memory_stats()
    try:
        from cosmos_windows.weight_streaming.streamed_text_encoder import RssProbe
        streamer = Magi2LayerRingStreamer(plan, device=device, source=FakeLayerSource())
        report.pinned_bytes = streamer.pinned_bytes
        abort_mib = streamer.pinned_bytes / (1024 * 1024) + 512
        streamer.new_pass()
        d = plan.ring_depth
        built = 0
        with RssProbe('magi2 layer-ring smoke', abort_mib=abort_mib) as probe:
            for pos in range(min(d, num_layers)):
                streamer.prefetch(pos)
            for i in range(num_layers):
                buf = streamer.wait_layer_buffer(i)
                assert buf.numel() == streamer.slot_bytes
                built += 1
                streamer.evict(i)
                ahead = i + d
                if ahead < num_layers:
                    streamer.prefetch(ahead)
        report.layers_built = built
        report.rss_delta_mib = probe.delta_mib
        model_mib = plan.total_layer_bytes / (1024 * 1024)
        report.rss_flat = not probe.aborted and (probe.delta_mib <= abort_mib or probe.baseline <= 0)
        report.notes.append(f'RSS delta {probe.delta_mib:.0f} MiB vs {model_mib:.0f} MiB model, pinned {report.pinned_bytes / 1000000.0:.0f} MB (flat={report.rss_flat})')
        from cosmos_windows.weight_streaming.magi2_attention_adapter import preview_bidirectional_sdpa
        q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = [0, seq_len // 2, seq_len]
        sink = torch.zeros(num_heads, device=device, dtype=torch.float32)
        out = preview_bidirectional_sdpa(q, k, v, cu, sink=sink)
        report.attn_out_shape = tuple(out.shape)
        report.peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        report.ok = tuple(out.shape) == (seq_len, num_heads, head_dim) and built == num_layers and (report.rss_flat is not False)
        streamer.close()
        if emit:
            emit.emit('stage_log', stage='gpu_dry_smoke', message='dry smoke ok', attn_out_shape=list(report.attn_out_shape), peak_vram_bytes=report.peak_vram_bytes)
    except Exception as exc:
        report.notes.append(f'gpu dry smoke failed: {type(exc).__name__}: {exc}')
        if emit:
            emit.fallback('gpu_dry_smoke', f'{type(exc).__name__}: {exc}')
    return report
