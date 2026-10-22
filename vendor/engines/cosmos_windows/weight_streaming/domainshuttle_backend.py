# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/domainshuttle_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
_EXPERT_SUBDIRS = ('low_noise_model', 'high_noise_model')
_EXPERT_WEIGHT_NAME = 'diffusion_pytorch_model.safetensors'
_DS_BLOCK_RE = re.compile('(?:^|[/.])blocks\\.(\\d+)\\.')
_DS_NUM_BLOCKS = 40
_DS_TENSORS_PER_BLOCK = 34
_DS_BLOCK_BYTES = 860167168
_DS_GLOBALS_BYTES = 571013248
DOMAIN_CODE: dict[str, int] = {'Human': 0, 'Man': 0, 'Woman': 0, 'object': 1, 'Object': 1, 'Fantasy Domain': 2, 'Background': 3, 'others': 4, 'Other': 4}
DEFAULT_NEGATIVE_PROMPT = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
_BOUNDARY = 0.875
_VAE_KWARGS = {'vae_type': 'AutoencoderKLWan', 'temporal_compression_ratio': 4, 'spatial_compression_ratio': 8}
_TEXT_ENCODER_KWARGS = {'text_length': 512, 'vocab': 256384, 'dim': 4096, 'dim_attn': 4096, 'dim_ffn': 10240, 'num_heads': 64, 'num_layers': 24, 'num_buckets': 32, 'shared_pos': False, 'dropout': 0.0}
_SCHEDULER_KWARGS = {'num_train_timesteps': 1000, 'shift': 12.0, 'use_dynamic_shifting': False, 'base_shift': 0.5, 'max_shift': 1.15, 'base_image_seq_len': 256, 'max_image_seq_len': 4096}

def _ensure_vendor_on_path() -> None:
    vendor_root = str(Path(__file__).resolve().parent / 'domainshuttle_vendor')
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

def resolve_domain_codes(domain_codes, n: int) -> list[int]:
    if not domain_codes:
        return [0] * n
    out: list[int] = []
    for v in domain_codes[:n]:
        if isinstance(v, bool):
            out.append(0)
        elif isinstance(v, int):
            out.append(int(v) if 0 <= int(v) <= 4 else 4)
        else:
            out.append(DOMAIN_CODE.get(str(v).strip(), DOMAIN_CODE.get(str(v).strip().title(), 4)))
    while len(out) < n:
        out.append(0)
    return out[:n]

def preflight_weights(domain_model_dir: str) -> None:
    root = Path(domain_model_dir)
    if not root.is_dir():
        raise RuntimeError(f'DomainShuttle model dir not found: {domain_model_dir!r}. Register the directory that contains {_EXPERT_SUBDIRS[0]}/ and {_EXPERT_SUBDIRS[1]}')
    missing: list[str] = []
    for sub in _EXPERT_SUBDIRS:
        d = root / sub
        cfg = d / 'config.json'
        wt = d / _EXPERT_WEIGHT_NAME
        if not d.is_dir():
            missing.append(f'{sub}/ (directory absent)')
            continue
        if not cfg.is_file():
            missing.append(f'{sub}/config.json')
        try:
            ok_wt = wt.is_file() and wt.stat().st_size > 1 << 30
        except OSError:
            ok_wt = False
        if not ok_wt:
            got = wt.stat().st_size if wt.is_file() else 0
            missing.append(f'{sub}/{_EXPERT_WEIGHT_NAME} (have {got / 1000000000.0:.1f} GB, expect ~35 GB)')
    if missing:
        raise RuntimeError(f"DomainShuttle weights are incomplete under {domain_model_dir!r}: {', '.join(missing)}. Finish the HF download of CNcreator0331/DomainShuttle_weight (both experts, ~70 GB total) before generating.")

def preflight_base_components(vae_path: str, umt5_path: str, tokenizer_dir: str) -> None:
    problems: list[str] = []
    if not vae_path or not Path(vae_path).is_file():
        problems.append(f'VAE not found: {vae_path!r} (expect Wan2.1_VAE.pth)')
    if not umt5_path or not Path(umt5_path).is_file():
        problems.append(f'T5 encoder not found: {umt5_path!r} (expect models_t5_umt5-xxl-enc-bf16.pth)')
    if not tokenizer_dir or not Path(tokenizer_dir).is_dir():
        problems.append(f'tokenizer dir not found: {tokenizer_dir!r} (expect a google/umt5-xxl dir)')
    if problems:
        raise RuntimeError('DomainShuttle base components missing: ' + '; '.join(problems) + '')

def _free_host_ram_gb() -> float:
    from . import ram_guard
    return ram_guard.free_host_ram_gb()

def _preflight_host_ram(memory_mode: str) -> None:
    if memory_mode != 'sequential_cpu_offload':
        return
    from . import ram_guard
    need = float(os.environ.get('STUDIO_DOMAINSHUTTLE_MIN_FREE_RAM_GB', '72'))
    ram_guard.preflight_host_ram(need, 'DomainShuttle v1 (sequential_cpu_offload) — both A14B experts (~35 GB each) staged on CPU', force_env='STUDIO_DOMAINSHUTTLE_FORCE', advice="The RAM-bounded path is the DomainShuttle-aware NVMe block streamer (memory_mode='streaming'/'streamed'); prefer it over a resident load.")

def _free_vram_gb(device_id: int=0) -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info(int(device_id))
        return free / 1024 ** 3
    except Exception:
        return 0.0

def estimate_streaming_vram_gb(n_gpu_buffers: int=3, *, streamed_experts: int=1) -> float:
    globals_gb = _DS_GLOBALS_BYTES / 1024 ** 3
    buffers_gb = max(1, int(n_gpu_buffers)) * (_DS_BLOCK_BYTES / 1024 ** 3)
    return int(streamed_experts) * (globals_gb + buffers_gb)

def _preflight_streaming(domain_model_dir: str, *, n_gpu_buffers: int=3, device_id: int=0) -> None:
    preflight_weights(domain_model_dir)
    if os.environ.get('STUDIO_DOMAINSHUTTLE_FORCE', '').strip() in ('1', 'true', 'yes'):
        return
    est = estimate_streaming_vram_gb(n_gpu_buffers, streamed_experts=1)
    margin = float(os.environ.get('STUDIO_DOMAINSHUTTLE_VRAM_MARGIN_GB', '6'))
    need = float(os.environ.get('STUDIO_DOMAINSHUTTLE_MIN_FREE_VRAM_GB', f'{est + margin:.2f}'))
    free = _free_vram_gb(device_id)
    if free <= 0.0:
        return
    if free < need:
        raise RuntimeError(f'DomainShuttle streaming needs ~{need:.1f} GB free VRAM (streamed working set ~{est:.1f} GB for one active A14B expert: {_DS_GLOBALS_BYTES / 1000000000.0:.2f} GB globals + {n_gpu_buffers} x {_DS_BLOCK_BYTES / 1000000000.0:.2f} GB block buffers, + ~{margin:.0f} GB VAE/activation margin), but only {free:.1f} GB is free on cuda:{device_id}. Free VRAM (unload other models) or lower n_gpu_buffers / STUDIO_DOMAINSHUTTLE_MIN_FREE_VRAM_GB. STUDIO_DOMAINSHUTTLE_FORCE=1 overrides.')

def _videos_tensor_to_pil(video):
    import numpy as np
    from PIL import Image
    v = video[0] if video.ndim == 5 else video
    frames = []
    n_t = v.shape[1]
    for f in range(n_t):
        img = v[:, f].permute(1, 2, 0).contiguous().float().clamp(0, 1)
        arr = (img.cpu().numpy() * 255.0).round().astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames

def ds_group_by_layer(idx):
    layers: dict[int, list] = {}
    globs: list = []
    for name, ref in idx.refs.items():
        m = _DS_BLOCK_RE.search(name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    return (layers, globs)

class _DSGroupedIndex:

    def __init__(self, base_idx):
        layers, _globs = ds_group_by_layer(base_idx)
        self._layers = layers
        self.refs = {r.name: r for refs in layers.values() for r in refs}

    def group_by_layer(self, component: str=''):
        return (self._layers, [])

    def names(self):
        return list(self.refs.keys())

def _load_meta_domainshuttle(expert_dir: str):
    import json
    from accelerate import init_empty_weights
    _ensure_vendor_on_path()
    from videox_fun.models import DomainShuttleWanTransformer3DModel
    cfg_path = Path(expert_dir) / 'config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    ctor_cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with init_empty_weights():
        model = DomainShuttleWanTransformer3DModel.from_config(ctor_cfg)
    return (model, ctor_cfg)

def build_streamed_domainshuttle(expert_dir: str, device: str='cuda', n_gpu_buffers: int=3, dtype=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    weight_file = str(Path(expert_dir) / _EXPERT_WEIGHT_NAME)
    dtype = dtype or torch.bfloat16
    idx = build_index(weight_file)
    model, cfg = _load_meta_domainshuttle(expert_dir)
    model = model.to(dtype)
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = ds_group_by_layer(idx)
    n_blocks = len(layer_refs)
    if n_blocks == 0:
        raise RuntimeError(f'ds_group_by_layer found 0 blocks in {weight_file} — the `blocks.N.` regex matched nothing. Check the tensor names: {list(idx.names())[:4]}')
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        if ref.name in sd_keys:
            set_module_tensor_to_device(model, ref.name, device, value=pb.read_tensor(ref, device))
    for name, buf in list(model.named_buffers()):
        if buf.device.type != 'meta' and str(buf.device) != str(device):
            set_module_tensor_to_device(model, name, device, value=buf.to(device))
    ds_idx = _DSGroupedIndex(idx)
    streamer = DirectStorageStreamer(ds_idx, component='', n_gpu_buffers=n_gpu_buffers, device=device)
    from .ds_backend import enforce_ds_mode
    enforce_ds_mode(streamer, 'DomainShuttle', 'DOMAINSHUTTLE')
    print(f'[DOMAINSHUTTLE-STREAM] {n_blocks} blocks streamed, {len(global_refs)} resident globals, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s), {Path(expert_dir).name}', flush=True)
    layer_paths = {il: [r.name for r in refs if r.name in sd_keys] for il, refs in layer_refs.items()}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                if name in sd_keys:
                    set_module_tensor_to_device(model, name, device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for name in layer_paths[il]:
                set_module_tensor_to_device(model, name, 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(model.blocks):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(x, *a, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict, **kw):
            _b()
            try:
                out = _fwd(_layer, x, *a, **kw)
            finally:
                _e()
            return out
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    model.eval()
    model.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f'[DOMAINSHUTTLE-STREAM] wrapped {n_blocks} DomainShuttleWanAttentionBlocks (bind/evict per block; single residual stream; incl. self_attn.*_ref + modulation_ref).', flush=True)
    return (model, streamer)

def _close_ds_streamer(streamer, label: str) -> None:
    if streamer is None:
        return
    close = getattr(streamer, 'close', None)
    if not callable(close):
        return
    try:
        close()
        print(f'[DOMAINSHUTTLE-STREAM] closed {label} DirectStorage streamer', flush=True)
    except Exception as e:
        print(f'[DOMAINSHUTTLE-STREAM] {label} streamer close skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)

def build_streaming_domainshuttle_pipeline(domain_model_dir: str, *, vae, tokenizer, text_encoder, scheduler, device: str='cuda', n_gpu_buffers: int=3, dtype=None, lazy_experts: bool=True):
    _ensure_vendor_on_path()
    from videox_fun.pipeline.pipeline_wan2_2_domainshuttle import Wan2_2DomainShuttlePipeline
    if not lazy_experts:
        transformer_low, streamer_low = build_streamed_domainshuttle(str(Path(domain_model_dir) / 'low_noise_model'), device=device, n_gpu_buffers=n_gpu_buffers, dtype=dtype)
        transformer_high, streamer_high = build_streamed_domainshuttle(str(Path(domain_model_dir) / 'high_noise_model'), device=device, n_gpu_buffers=n_gpu_buffers, dtype=dtype)
        pipe = Wan2_2DomainShuttlePipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer_low, transformer_2=transformer_high, scheduler=scheduler)
        pipe._streamer_low_expert = streamer_low
        pipe._streamer_high_expert = streamer_high
        pipe._ds_lazy = False
        return pipe
    transformer_high, streamer_high2 = build_streamed_domainshuttle(str(Path(domain_model_dir) / 'high_noise_model'), device=device, n_gpu_buffers=n_gpu_buffers, dtype=dtype)
    StreamingCls = _make_streaming_pipeline_class()
    pipe = StreamingCls(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer_high, transformer_2=transformer_high, scheduler=scheduler)
    pipe._ds_domain_model_dir = str(domain_model_dir)
    pipe._ds_device = device
    pipe._ds_n_gpu_buffers = int(n_gpu_buffers)
    pipe._ds_dtype = dtype
    pipe._ds_high = transformer_high
    pipe._ds_high_streamer = streamer_high2
    pipe._ds_low = None
    pipe._ds_low_streamer = None
    pipe._ds_lazy = True
    return pipe

def _make_streaming_pipeline_class():
    _ensure_vendor_on_path()
    from videox_fun.pipeline.pipeline_wan2_2_domainshuttle import Wan2_2DomainShuttlePipeline

    class _StreamingDomainShuttlePipeline(Wan2_2DomainShuttlePipeline):

        def _ds_ensure_low(self):
            if self._ds_low is not None:
                return
            import gc
            _close_ds_streamer(self._ds_high_streamer, 'HIGH before LOW build')
            self._ds_high = None
            self._ds_high_streamer = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            low, low_streamer = build_streamed_domainshuttle(str(Path(self._ds_domain_model_dir) / 'low_noise_model'), device=self._ds_device, n_gpu_buffers=self._ds_n_gpu_buffers, dtype=self._ds_dtype)
            self._ds_low = low
            self._ds_low_streamer = low_streamer
            print('[DOMAINSHUTTLE-STREAM] crossover: evicted HIGH, streaming LOW (lazy one-at-a-time)', flush=True)

        def __call__(self, *args, boundary: float=0.875, **kwargs):

            class _LazyExpertProxy:

                def __init__(self, owner):
                    object.__setattr__(self, '_owner', owner)

                def __call__(self, *a, **k):
                    owner = object.__getattribute__(self, '_owner')
                    owner._ds_ensure_low()
                    return owner._ds_low(*a, **k)

                def __getattr__(self, name):
                    owner = object.__getattribute__(self, '_owner')
                    tgt = owner._ds_low if owner._ds_low is not None else owner._ds_high
                    return getattr(tgt, name)

                def __setattr__(self, name, value):
                    owner = object.__getattribute__(self, '_owner')
                    tgt = owner._ds_low if owner._ds_low is not None else owner._ds_high
                    setattr(tgt, name, value)
            object.__setattr__(self, 'transformer_2', self._ds_high)
            object.__setattr__(self, 'transformer', _LazyExpertProxy(self))
            try:
                return super().__call__(*args, boundary=boundary, **kwargs)
            finally:
                _close_ds_streamer(getattr(self, '_ds_high_streamer', None), 'HIGH teardown')
                _close_ds_streamer(getattr(self, '_ds_low_streamer', None), 'LOW teardown')
    return _StreamingDomainShuttlePipeline
_StreamingDomainShuttlePipeline = None

def domainshuttle_r2v_generate(*, domain_model_dir: str, reference_image_paths: list[str], domain_codes=None, prompt: str, negative_prompt: str | None=None, height: int=480, width: int=832, num_frames: int=81, steps: int=40, guidance_scale=(4.0, 3.0), shift: float | None=None, boundary: float=_BOUNDARY, seed: int=42, vae_path: str='', umt5_path: str='', tokenizer_dir: str='', memory_mode: str='sequential_cpu_offload', max_reference_num: int=5, device_id: int=0, progress=None, should_cancel=None):
    import torch
    refs = [str(p) for p in reference_image_paths or [] if str(p).strip()]
    if not refs:
        raise RuntimeError('DomainShuttle r2v requires at least one reference image.')
    refs = refs[:max_reference_num]
    for p in refs:
        if not Path(p).is_file():
            raise RuntimeError(f'DomainShuttle reference image not found: {p!r}')
    streaming = str(memory_mode).strip() in ('streaming', 'streamed')
    n_gpu_buffers = int(os.environ.get('STUDIO_DOMAINSHUTTLE_N_GPU_BUFFERS', '3'))
    preflight_base_components(vae_path, umt5_path, tokenizer_dir)
    if streaming:
        _preflight_streaming(domain_model_dir, n_gpu_buffers=n_gpu_buffers, device_id=device_id)
    else:
        preflight_weights(domain_model_dir)
        _preflight_host_ram(memory_mode)
    _ensure_vendor_on_path()
    from videox_fun.models import AutoencoderKLWan, AutoTokenizer, DomainShuttleWanTransformer3DModel, WanT5EncoderModel
    from videox_fun.pipeline.pipeline_wan2_2_domainshuttle import Wan2_2DomainShuttlePipeline
    from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from videox_fun.utils.fp8_optimization import replace_parameters_by_name
    from videox_fun.utils.utils import filter_kwargs, get_image_latent
    device = f'cuda:{int(device_id)}' if torch.cuda.is_available() else 'cpu'
    weight_dtype = torch.bfloat16
    codes = resolve_domain_codes(domain_codes, len(refs))
    if shift is None:
        shift = 5.0 if max(int(height), int(width)) <= 832 else 12.0
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
    if isinstance(guidance_scale, (int, float)):
        guidance_scale = (float(guidance_scale), float(guidance_scale))

    def _emit(step: int, total: int) -> None:
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        if progress:
            try:
                progress(int(step), int(total))
            except TypeError:
                progress(int(step), int(total), None)
    vae = AutoencoderKLWan.from_pretrained(vae_path, additional_kwargs=dict(_VAE_KWARGS)).to(weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    text_encoder = WanT5EncoderModel.from_pretrained(umt5_path, additional_kwargs=dict(_TEXT_ENCODER_KWARGS), low_cpu_mem_usage=True, torch_dtype=weight_dtype).eval()
    scheduler = FlowUniPCMultistepScheduler(**filter_kwargs(FlowUniPCMultistepScheduler, dict(_SCHEDULER_KWARGS)))
    transformer = transformer_2 = None
    if streaming:
        vae = vae.to(device=device)
        text_encoder = text_encoder.to(device=device)
        print(f'[DOMAINSHUTTLE] STREAMING build ({Path(domain_model_dir).name}: HIGH first, LOW at crossover; n_gpu_buffers={n_gpu_buffers})', flush=True)
        pipeline = build_streaming_domainshuttle_pipeline(domain_model_dir, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, device=device, n_gpu_buffers=n_gpu_buffers, dtype=weight_dtype, lazy_experts=True)
    else:
        print(f'[DOMAINSHUTTLE] loading LOW expert ({Path(domain_model_dir).name}/low_noise_model)', flush=True)
        transformer = DomainShuttleWanTransformer3DModel.from_pretrained(str(Path(domain_model_dir) / 'low_noise_model'), low_cpu_mem_usage=True, torch_dtype=weight_dtype)
        print('[DOMAINSHUTTLE] loading HIGH expert (high_noise_model)', flush=True)
        transformer_2 = DomainShuttleWanTransformer3DModel.from_pretrained(str(Path(domain_model_dir) / 'high_noise_model'), low_cpu_mem_usage=True, torch_dtype=weight_dtype)
        pipeline = Wan2_2DomainShuttlePipeline(transformer=transformer, transformer_2=transformer_2, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler)
        if memory_mode == 'sequential_cpu_offload':
            replace_parameters_by_name(transformer, ['modulation', 'modulation_ref'], device=device)
            replace_parameters_by_name(transformer_2, ['modulation', 'modulation_ref'], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        else:
            pipeline.to(device=device)
    ref_latents = [get_image_latent(p, sample_size=[int(height), int(width)], padding=True) for p in refs]
    reference_images = torch.cat(ref_latents, dim=2)
    domain_code = torch.tensor([codes], dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    ratio = int(getattr(vae.config, 'temporal_compression_ratio', 4))
    nf = int(num_frames)
    video_length = 1 if nf <= 1 else int((nf - 1) // ratio * ratio) + 1
    total_steps = int(steps)

    def _cb(_pipe, i, _t, kw):
        _emit(i + 1, total_steps)
        return kw
    print(f'[DOMAINSHUTTLE] r2v: {len(refs)} ref(s) domains={codes} {width}x{height} frames={video_length} steps={total_steps} shift={shift} guidance={guidance_scale}', flush=True)
    try:
        with torch.no_grad():
            out = pipeline(prompt, negative_prompt=negative_prompt, height=int(height), width=int(width), num_frames=video_length, num_inference_steps=total_steps, guidance_scale=guidance_scale if len(tuple(guidance_scale)) > 1 else guidance_scale[0], generator=generator, reference_images=reference_images, domain_code=domain_code, boundary=boundary, shift=shift, callback_on_step_end=_cb)
        return _videos_tensor_to_pil(out.videos)
    finally:
        for _attr in ('_ds_high_streamer', '_ds_low_streamer', '_streamer_high_expert', '_streamer_low_expert'):
            _close_ds_streamer(getattr(pipeline, _attr, None), _attr)
        try:
            del pipeline, transformer, transformer_2, vae, text_encoder
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
