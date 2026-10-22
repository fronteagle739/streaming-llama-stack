# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/zimage_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import os
from pathlib import Path
DEFAULT_FILE = ''
DEFAULT_CONFIG_DIR = os.environ.get('ZIMAGE_CONFIG_DIR', '')
DEFAULT_TEXT_ENCODER = os.environ.get('ZIMAGE_TEXT_ENCODER', '')
DEFAULT_VAE = os.environ.get('ZIMAGE_VAE', '')
_RAM_GUARD_ARMED = False
_ZIMAGE_REMAP = (('final_layer.', 'all_final_layer.2-1.'), ('x_embedder.', 'all_x_embedder.2-1.'), ('.attention.out.bias', '.attention.to_out.0.bias'), ('.attention.k_norm.weight', '.attention.norm_k.weight'), ('.attention.q_norm.weight', '.attention.norm_q.weight'), ('.attention.out.weight', '.attention.to_out.0.weight'), ('model.diffusion_model.', ''))

def configure_zimage_sampling(pipe, *, sampler: str | None=None, sigmas: str | None=None, num_inference_steps: int=8) -> list[float] | None:
    from cosmos_windows.res4lyf_sampling import configure_diffusers_flow_scheduler
    return configure_diffusers_flow_scheduler(pipe, sampler=sampler, sigmas=sigmas, num_inference_steps=int(num_inference_steps or 8), log_prefix='[ZIMAGE]')

def _require_path(path: str | os.PathLike | None, label: str, *, file: bool | None=None) -> Path:
    if not path:
        raise FileNotFoundError(f"Z-Image {label} is required. Register it in the model's support files.")
    p = Path(path)
    ok = p.exists() if file is None else p.is_file() if file else p.is_dir()
    if not ok:
        kind = 'file' if file else 'directory' if file is False else 'path'
        raise FileNotFoundError(f'Z-Image {label} {kind} not found: {p}')
    return p

def _component_dir(config_dir: Path, subfolder: str) -> Path:
    p = config_dir / subfolder
    if not (p / 'config.json').is_file():
        raise FileNotFoundError(f'Z-Image config bundle is missing {subfolder}/config.json under {config_dir}')
    return p

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def _guard_ram(label: str, *, min_free_gb: float | None=None) -> None:
    global _RAM_GUARD_ARMED
    from . import ram_guard
    floor = float(os.environ.get('ZIMAGE_MIN_FREE_RAM_GB', min_free_gb or 10.0))
    ram_guard.preflight_host_ram(floor, label, skip_env='ZIMAGE_SKIP_RAM_PREFLIGHT', force_env='ZIMAGE_FORCE_RESIDENT', advice='Use the Z-Image per-block streamer (stream_weights=True) on a RAM-constrained box.')
    max_proc = float(os.environ.get('ZIMAGE_MAX_PROC_RAM_GB', 14.0))
    min_free = float(os.environ.get('ZIMAGE_WATCHDOG_MIN_FREE_GB', 4.0))
    try:
        from cosmos_windows import mem_guard
    except Exception:
        mem_guard = None
    if mem_guard is None:
        return
    if not _RAM_GUARD_ARMED:
        mem_guard.start_watchdog(max_process_gb=max_proc, min_free_gb=min_free, poll_seconds=0.5, label=label)
        _RAM_GUARD_ARMED = True

def _assert_cuda_resident(pipe, device: str | None=None) -> None:
    import torch
    if not torch.cuda.is_available():
        return
    expected = torch.device(device or 'cuda')
    expected_index = expected.index
    if expected.type == 'cuda' and expected_index is None:
        expected_index = torch.cuda.current_device()
    modules = [getattr(pipe, name, None) for name in ('transformer', 'text_encoder', 'vae')]
    bad = []
    for name, module in zip(('transformer', 'text_encoder', 'vae'), modules):
        if module is None:
            continue
        param = next(module.parameters(), None)
        if param is None:
            continue
        if param.device.type == 'meta':
            continue
        wrong_type = param.device.type != 'cuda'
        wrong_index = expected_index is not None and param.device.index not in (None, expected_index)
        if wrong_type or wrong_index:
            bad.append(f'{name}={param.device}')
    if bad:
        raise RuntimeError(f"Z-Image expected CUDA-resident modules on {device or 'cuda'}, got " + ', '.join(bad))

def _load_qwen3_text_encoder(text_encoder: str, config_dir: Path, *, dtype, device: str):
    import torch
    from transformers import Qwen3Config, Qwen3Model
    p = _require_path(text_encoder, 'text_encoder', file=None)
    if p.is_dir():
        model = Qwen3Model.from_pretrained(str(p), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True, device_map={'': device})
        return model.eval()
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    cfg_path = _component_dir(config_dir, 'text_encoder') / 'config.json'
    cfg = Qwen3Config(**{k: v for k, v in _load_json(cfg_path).items() if not k.startswith('_')})
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights():
            model = Qwen3Model(cfg)
    finally:
        torch.set_default_dtype(old_default_dtype)
    expected_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    missing: list[str] = []
    loaded = 0
    from .direct_gpu_load import DirectSafetensorsReader
    with DirectSafetensorsReader(str(p), device) as sf:
        keys = set(sf.keys())
        for dst, shape in expected_shapes.items():
            src = f'model.{dst}' if f'model.{dst}' in keys else dst
            if src not in keys:
                missing.append(dst)
                continue
            raw = sf.get_tensor(src)
            if tuple(raw.shape) != shape:
                raise RuntimeError(f'Z-Image text_encoder tensor {src!r} has shape {tuple(raw.shape)}, expected {shape}. The selected text encoder does not match the Z-Image config bundle.')
            if raw.dtype == torch.uint8:
                raise RuntimeError(f'Z-Image text_encoder tensor {src!r} uses packed uint8 Comfy quantization. That quant format needs an architecture-loader plugin; use a bf16/fp16 single-file or local HF-style Qwen3 folder for now.')
            scale_key = src[:-len('.weight')] + '.weight_scale' if src.endswith('.weight') else ''
            if 'float8' in str(raw.dtype) and scale_key in keys:
                scale = sf.get_tensor(scale_key)
                if tuple(scale.shape) not in ((), (1,)):
                    raise RuntimeError(f'Z-Image text_encoder tensor {src!r} uses grouped fp8 scales {tuple(scale.shape)}. Use a bf16/fp16 single-file or local HF-style Qwen3 folder until grouped fp8 support lands.')
                value = raw.to(dtype) * scale.to(dtype)
            else:
                value = raw.to(dtype)
            set_module_tensor_to_device(model, dst, device, value=value)
            loaded += 1
            del raw, value
    if missing:
        raise RuntimeError(f'Z-Image text_encoder load left {len(missing)} tensors unfilled, e.g. {missing[:4]}. The selected text encoder file is not compatible with the local config bundle.')
    print(f'[ZIMAGE] text encoder loaded {loaded}/{len(expected_shapes)} tensors to {device}', flush=True)
    return model.eval()

def _load_zimage_vae(vae: str | None, config_dir: Path, *, dtype, device: str):
    from diffusers import AutoencoderKL
    p = Path(vae) if vae else config_dir / 'vae'
    p = _require_path(p, 'vae', file=None)
    if p.is_dir():
        model = AutoencoderKL.from_pretrained(str(p), torch_dtype=dtype, local_files_only=True, device_map={'': device})
    else:
        from safetensors.torch import load_file
        try:
            from accelerate import init_empty_weights
            with init_empty_weights(include_buffers=False):
                model = AutoencoderKL.from_config(str(_component_dir(config_dir, 'vae')))
            _assign = True
        except Exception:
            model = AutoencoderKL.from_config(str(_component_dir(config_dir, 'vae')))
            _assign = False
        from .direct_gpu_load import load_safetensors_direct
        sd = load_safetensors_direct(str(p), device)
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=_assign)
        except RuntimeError as exc:
            raise RuntimeError(f"Z-Image VAE file {p} does not match the AutoencoderKL config in {config_dir / 'vae'}. Point the z-image VAE support file at a Z-Image/Tongyi AutoencoderKL VAE, not a Qwen-image VAE.") from exc
        if missing:
            raise RuntimeError(f'Z-Image VAE load left {len(missing)} tensors unfilled, e.g. {missing[:4]}')
        if unexpected:
            print(f'[ZIMAGE] ignored {len(unexpected)} unexpected VAE tensors, e.g. {unexpected[:4]}', flush=True)
    if hasattr(model, 'enable_tiling'):
        model.enable_tiling()
    return model.to(device=device, dtype=dtype).eval()

def _remap_zimage_key(name: str) -> str:
    for old, new in _ZIMAGE_REMAP:
        name = name.replace(old, new)
    return name

def _load_meta_zimage(config_dir: Path, *, dtype):
    import torch
    from accelerate import init_empty_weights
    from diffusers import ZImageTransformer2DModel
    cfg = ZImageTransformer2DModel.load_config(str(_component_dir(config_dir, 'transformer')))
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights():
            model = ZImageTransformer2DModel.from_config(cfg)
    finally:
        torch.set_default_dtype(old_default_dtype)
    return model

def _split_qkv_ref(dst_name: str, ref):
    from .safetensors_index import TensorRef
    if len(ref.shape) < 1 or ref.shape[0] % 3 != 0 or ref.nbytes % 3 != 0:
        raise RuntimeError(f'Z-Image fused QKV tensor {ref.name!r} has unsupported shape {ref.shape}')
    rows = ref.shape[0] // 3
    part_nbytes = ref.nbytes // 3
    shape = (rows, *ref.shape[1:])
    for idx, proj in enumerate(('to_q', 'to_k', 'to_v')):
        yield TensorRef(name=dst_name.replace('.attention.qkv.weight', f'.attention.{proj}.weight'), file=ref.file, offset=ref.offset + idx * part_nbytes, nbytes=part_nbytes, dtype=ref.dtype, shape=shape)

def _build_zimage_stream_index(transformer_file: str):
    from .safetensors_index import SafetensorsIndex, TensorRef, build_index
    src_idx = build_index(transformer_file)
    refs: dict[str, TensorRef] = {}
    for src_name, src_ref in src_idx.refs.items():
        dst_name = _remap_zimage_key(src_name)
        if dst_name == 'norm_final.weight':
            continue
        if '.attention.qkv.weight' in dst_name:
            for out_ref in _split_qkv_ref(dst_name, src_ref):
                refs[out_ref.name] = out_ref
            continue
        refs[dst_name] = TensorRef(name=dst_name, file=src_ref.file, offset=src_ref.offset, nbytes=src_ref.nbytes, dtype=src_ref.dtype, shape=src_ref.shape)
    return SafetensorsIndex(refs)

def build_streamed_zimage_transformer(transformer_file: str, config_dir: str | Path, *, device: str='cuda', dtype=None, n_gpu_buffers: int=3):
    import os as _os
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    dtype = dtype or torch.bfloat16
    _guard_ram('Z-Image streamed transformer load', min_free_gb=10.0)
    config_root = _require_path(config_dir, 'config_dir', file=False)
    transformer_file = str(_require_path(transformer_file, 'transformer_file', file=True))
    model = _load_meta_zimage(config_root, dtype=dtype)
    idx = _build_zimage_stream_index(transformer_file)
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = idx.group_by_layer('')
    if len(layer_refs) != len(model.layers):
        raise RuntimeError(f'Z-Image streaming keyset mismatch: checkpoint has {len(layer_refs)} layer groups, module has {len(model.layers)} layers.')
    missing = sorted({r.name for refs in layer_refs.values() for r in refs} - sd_keys)
    if missing:
        raise RuntimeError(f'Z-Image streamed layer tensors have no module home, e.g. {missing[:8]}')
    pb = PortableBackend(pin_memory=True)
    n_glob_resident = 0
    for ref in global_refs:
        if ref.name not in sd_keys:
            continue
        value = pb.read_tensor(ref, device)
        if value.is_floating_point():
            value = value.to(dtype)
        set_module_tensor_to_device(model, ref.name, device, value=value, dtype=dtype if value.is_floating_point() else None)
        n_glob_resident += 1
    streamer = DirectStorageStreamer(idx, component='', n_gpu_buffers=n_gpu_buffers, device=device)
    if getattr(streamer, 'mode', None) != 3 and _os.environ.get('ZIMAGE_ALLOW_NON_DIRECTSTORAGE') != '1':
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"Z-Image streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3). Refusing to run this production path without real NVMe->GPU streaming. Set ZIMAGE_ALLOW_NON_DIRECTSTORAGE=1 only for diagnostics.")

    def _resolve(path: str) -> str:
        parent, _, leaf = path.rpartition('.')
        try:
            module = model.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(module, 'base_layer') else path
    layer_paths = {il: [r.name for r in refs if r.name in sd_keys] for il, refs in layer_refs.items()}

    def make_bind(il: int):

        def bind():
            for name, tensor in streamer.wait_layer(il).items():
                if name not in sd_keys:
                    continue
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                set_module_tensor_to_device(model, _resolve(name), device, value=tensor, dtype=dtype if tensor.is_floating_point() else None)
        return bind

    def make_evict(il: int):

        def evict():
            for path in layer_paths[il]:
                set_module_tensor_to_device(model, _resolve(path), 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(model.layers):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(*args, _layer=layer, _forward=cls_forward, _bind=bind, _evict=evict, **kwargs):
            _bind()
            try:
                return _forward(_layer, *args, **kwargs)
            finally:
                _evict()
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    model.register_forward_pre_hook(lambda _m, _a: streamer.new_pass())
    model._zimage_streamer = streamer
    model.eval()
    print(f'[ZIMAGE-STREAM] {streamer.n} main layers streamed, {n_glob_resident} globals/refiner tensors resident, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)
    return (model, streamer)

def build_resident_zimage_transformer(transformer_file: str, config_dir: str | Path, *, device: str='cuda', dtype=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    dtype = dtype or torch.bfloat16
    _guard_ram('Z-Image resident transformer load', min_free_gb=10.0)
    config_root = _require_path(config_dir, 'config_dir', file=False)
    transformer_file = str(_require_path(transformer_file, 'transformer_file', file=True))
    model = _load_meta_zimage(config_root, dtype=dtype)
    idx = _build_zimage_stream_index(transformer_file)
    sd_keys = set(model.state_dict().keys())
    idx_keys = set(idx.refs.keys())
    missing = sorted(sd_keys - idx_keys)
    unexpected = sorted(idx_keys - sd_keys)
    if missing:
        raise RuntimeError(f'Z-Image resident load left module tensors unmapped, e.g. {missing[:8]}')
    if unexpected:
        raise RuntimeError(f'Z-Image resident load has unexpected checkpoint tensors, e.g. {unexpected[:8]}')
    pb = PortableBackend(pin_memory=True)
    loaded = 0
    for name in sorted(sd_keys):
        ref = idx.refs[name]
        value = pb.read_tensor(ref, device)
        if value.is_floating_point():
            value = value.to(dtype)
        set_module_tensor_to_device(model, name, device, value=value, dtype=dtype if value.is_floating_point() else None)
        loaded += 1
        del value
    model.eval()
    print(f'[ZIMAGE-RESIDENT] loaded {loaded}/{len(sd_keys)} tensors directly to {device} ({dtype})', flush=True)
    return model

def build_zimage_pipeline(transformer_file: str=DEFAULT_FILE, config_dir: str | None=None, text_encoder: str | None=None, vae: str | None=None, tokenizer_dir: str | None=None, scheduler_dir: str | None=None, device: str='cuda', dtype=None, offload: bool=False, token: str | None=None, loras=None, repo: str | None=None, stream_weights: bool=True, n_gpu_buffers: int=3, i2i: bool=False):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
    try:
        from diffusers import ZImageImg2ImgPipeline
    except Exception:
        ZImageImg2ImgPipeline = None
    from transformers import Qwen2Tokenizer
    dtype = dtype or torch.bfloat16
    _guard_ram('Z-Image pipeline load', min_free_gb=10.0)
    if repo and (not config_dir) and Path(repo).is_dir():
        config_dir = repo
    config_root = _require_path(config_dir or DEFAULT_CONFIG_DIR, 'config_dir', file=False)
    transformer_file = str(_require_path(transformer_file, 'transformer_file', file=True))
    text_encoder = text_encoder or DEFAULT_TEXT_ENCODER
    vae = vae or DEFAULT_VAE or str(config_root / 'vae')
    tokenizer_root = Path(tokenizer_dir) if tokenizer_dir else config_root / 'tokenizer'
    scheduler_root = Path(scheduler_dir) if scheduler_dir else config_root / 'scheduler'
    _require_path(tokenizer_root, 'tokenizer', file=False)
    _require_path(scheduler_root, 'scheduler', file=False)
    streamer = None
    if stream_weights:
        transformer, streamer = build_streamed_zimage_transformer(transformer_file, config_root, device=device, dtype=dtype, n_gpu_buffers=n_gpu_buffers)
    else:
        transformer = build_resident_zimage_transformer(transformer_file, config_root, device=device, dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_root), local_files_only=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
    text_model = _load_qwen3_text_encoder(text_encoder, config_root, dtype=dtype, device=device)
    vae_model = _load_zimage_vae(vae, config_root, dtype=dtype, device=device)
    if i2i:
        if ZImageImg2ImgPipeline is None:
            raise RuntimeError("Z-Image i2i needs diffusers' ZImageImg2ImgPipeline (not present in this diffusers build). Upgrade diffusers or use t2i.")
        PipeCls = ZImageImg2ImgPipeline
    else:
        PipeCls = ZImagePipeline
    pipe = PipeCls(scheduler=scheduler, vae=vae_model, text_encoder=text_model, tokenizer=tokenizer, transformer=transformer)
    if loras:
        from .lora_stack import apply_lora_stack
        apply_lora_stack(pipe, loras)
    if offload:
        if streamer is not None:
            raise RuntimeError('Z-Image CPU offload is incompatible with the streamed transformer path.')
        pipe.enable_model_cpu_offload()
    else:
        if streamer is None:
            pipe.to(device)
        else:
            pipe._zimage_streamer = streamer
        if str(device).startswith('cuda'):
            _assert_cuda_resident(pipe, device)
    return pipe
DEFAULT_CONTROLNET = ''

def build_zimage_controlnet_pipeline(transformer_file: str=DEFAULT_FILE, controlnet=DEFAULT_CONTROLNET, config_dir: str | None=None, text_encoder: str | None=None, vae: str | None=None, tokenizer_dir: str | None=None, scheduler_dir: str | None=None, device: str='cuda', dtype=None, offload: bool=False, token: str | None=None, loras=None, repo: str | None=None, stream_weights: bool=True, n_gpu_buffers: int=3):
    import os as _os, glob as _glob, torch
    from diffusers import ZImageControlNetPipeline, ZImageControlNetModel
    dtype = dtype or torch.bfloat16
    base = build_zimage_pipeline(transformer_file=transformer_file, config_dir=config_dir, text_encoder=text_encoder, vae=vae, tokenizer_dir=tokenizer_dir, scheduler_dir=scheduler_dir, device=device, dtype=dtype, offload=False, token=token, loras=None, repo=repo, stream_weights=stream_weights, n_gpu_buffers=n_gpu_buffers)
    cn = controlnet
    if isinstance(controlnet, str) and _os.path.isdir(controlnet):
        sf = _glob.glob(_os.path.join(controlnet, '*.safetensors'))
        cn = sf[0] if sf else controlnet
    cnet = ZImageControlNetModel.from_single_file(cn, torch_dtype=dtype, token=token) if isinstance(cn, str) and cn.endswith('.safetensors') else ZImageControlNetModel.from_pretrained(cn, torch_dtype=dtype, token=token)
    pipe = ZImageControlNetPipeline(scheduler=base.scheduler, vae=base.vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer, transformer=base.transformer, controlnet=cnet)
    if loras:
        from .lora_stack import apply_lora_stack
        apply_lora_stack(pipe, loras)
    if offload:
        if getattr(getattr(pipe, 'transformer', None), '_zimage_streamer', None) is not None:
            raise RuntimeError('Z-Image ControlNet CPU offload is incompatible with the streamed transformer path.')
        pipe.enable_model_cpu_offload()
    else:
        cnet.to(device=device, dtype=dtype)
        if getattr(getattr(pipe, 'transformer', None), '_zimage_streamer', None) is None:
            pipe.to(device)
        if str(device).startswith('cuda'):
            _assert_cuda_resident(pipe, device)
    return pipe
if __name__ == '__main__':
    import os, sys, time, torch
    from .gpu_select import pick_free_gpu, assert_gpu_idle
    assert_gpu_idle()
    pick_free_gpu()
    tok = None
    try:
        tok = open('').read().strip()
    except Exception:
        pass
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'a highly detailed cinematic portrait, neon rim light'
    pipe = build_zimage_pipeline(token=tok)
    t = time.time()
    img = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=8, generator=torch.Generator('cpu').manual_seed(42)).images[0]
    os.makedirs('', exist_ok=True)
    img.save('')
    print(f'RESULT ZIMAGE_OK {time.time() - t:.0f}s peak {torch.cuda.max_memory_allocated() / 1000000000.0:.1f}GB')
