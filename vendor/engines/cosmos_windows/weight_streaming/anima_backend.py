# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/anima_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import os
import re
DEFAULT_TF = ''
DEFAULT_QWEN = ''
DEFAULT_DIR = ''
ANIMA_RESIDENT_HOST_RAM_GB = 2.0
ANIMA_VRAM_HEADROOM_MIB = 3072

def _preflight_resident_ram() -> None:
    from . import ram_guard
    need = float(os.environ.get('ANIMA_MIN_FREE_RAM_GB', str(ANIMA_RESIDENT_HOST_RAM_GB)))
    ram_guard.preflight_host_ram(need, 'Anima resident load (stream_weights=False) — DiT + Qwen3 + VAE read direct to GPU through 64 MB chunk buffers', skip_env='ANIMA_SKIP_RAM_PREFLIGHT', force_env='ANIMA_FORCE_RESIDENT', advice='Use stream_weights=True to bind/evict Anima transformer blocks per-forward instead of a resident load.')

def _preflight_resident_vram(device: str, files) -> int:
    from . import vram_guard
    weight_bytes = 0
    for f in files:
        try:
            weight_bytes += os.path.getsize(f)
        except (OSError, TypeError):
            pass
    try:
        headroom = int(os.environ.get('ANIMA_VRAM_HEADROOM_MIB', ANIMA_VRAM_HEADROOM_MIB))
    except ValueError:
        headroom = ANIMA_VRAM_HEADROOM_MIB
    need_mib = int(weight_bytes / (1024 * 1024)) + headroom
    if str(device).startswith('cuda'):
        vram_guard.preflight_vram(str(device), need_mib, 'Anima resident load', skip_env='ANIMA_SKIP_VRAM_PREFLIGHT')
    return need_mib
_ANIMA_TRANSFORMER_PREFIXES = ('net.', 'model.diffusion_model.', 'diffusion_model.')

def _anima_transformer_prefix(keys) -> str | None:
    if not keys:
        return None
    for prefix in _ANIMA_TRANSFORMER_PREFIXES:
        if all((key.startswith(prefix) for key in keys)):
            return prefix
    return None

def _normalize_anima_transformer_state_dict(state_dict: dict) -> tuple[dict, str]:
    prefix = _anima_transformer_prefix(list(state_dict))
    if prefix in (None, 'net.'):
        return (state_dict, prefix or '')
    return ({'net.' + key[len(prefix):]: value for key, value in state_dict.items()}, prefix)

def _anima_source_to_net_key(key: str, source_prefix: str | None) -> str:
    if not source_prefix or source_prefix == 'net.':
        return key
    return 'net.' + key[len(source_prefix):]
_ANIMA_COSMOS2_RENAMES = {'t_embedder.1': 'time_embed.t_embedder', 't_embedding_norm': 'time_embed.norm', 'blocks': 'transformer_blocks', 'adaln_modulation_self_attn.1': 'norm1.linear_1', 'adaln_modulation_self_attn.2': 'norm1.linear_2', 'adaln_modulation_cross_attn.1': 'norm2.linear_1', 'adaln_modulation_cross_attn.2': 'norm2.linear_2', 'adaln_modulation_mlp.1': 'norm3.linear_1', 'adaln_modulation_mlp.2': 'norm3.linear_2', 'self_attn': 'attn1', 'cross_attn': 'attn2', 'q_proj': 'to_q', 'k_proj': 'to_k', 'v_proj': 'to_v', 'output_proj': 'to_out.0', 'q_norm': 'norm_q', 'k_norm': 'norm_k', 'mlp.layer1': 'ff.net.0.proj', 'mlp.layer2': 'ff.net.2', 'x_embedder.proj.1': 'patch_embed.proj', 'final_layer.adaln_modulation.1': 'norm_out.linear_1', 'final_layer.adaln_modulation.2': 'norm_out.linear_2', 'final_layer.linear': 'proj_out'}
_ANIMA_COSMOS2_REMOVES = ('accum_video_sample_counter', 'accum_image_sample_counter', 'accum_iteration', 'accum_train_in_hours', 'pos_embedder.seq', 'pos_embedder.dim_spatial_range', 'pos_embedder.dim_temporal_range', '_extra_state')

def _convert_anima_cosmos2_key(key: str) -> str | None:
    new_key = key[4:] if key.startswith('net.') else key
    for old, new in _ANIMA_COSMOS2_RENAMES.items():
        new_key = new_key.replace(old, new)
    if any((marker in new_key for marker in _ANIMA_COSMOS2_REMOVES)):
        return None
    return new_key

def _child_module(module, name: str):
    if name.isdigit() and hasattr(module, '__getitem__'):
        try:
            return module[int(name)]
        except Exception:
            pass
    return getattr(module, name)

def _module_parent(root, key: str):
    parts = key.split('.')
    module = root
    for part in parts[:-1]:
        module = _child_module(module, part)
    return (module, parts[-1])

def _set_module_tensor_to_meta(root, key: str) -> bool:
    try:
        module, name = _module_parent(root, key)
        param = getattr(module, '_parameters', {}).get(name)
        if param is not None:
            import torch
            module._parameters[name] = torch.nn.Parameter(torch.empty(tuple(param.shape), device='meta', dtype=param.dtype), requires_grad=param.requires_grad)
            return True
        buf = getattr(module, '_buffers', {}).get(name)
        if buf is not None:
            import torch
            module._buffers[name] = torch.empty(tuple(buf.shape), device='meta', dtype=buf.dtype)
            return True
    except Exception:
        return False
    return False

class _AnimaTransformerStreamer:

    def __init__(self, model, transformer_file: str, *, dtype, device: str):
        from safetensors import safe_open
        self.model = model
        self.transformer_file = transformer_file
        self.dtype = dtype
        self.device = str(device)
        self._cm = safe_open(transformer_file, framework='pt', device='cpu')
        self._handle = self._cm.__enter__()
        source_keys = list(self._handle.keys())
        source_prefix = _anima_transformer_prefix(source_keys)
        if source_prefix is None:
            raise RuntimeError(f"Anima native streaming expected a Cosmos2 transformer key layout ({', '.join(_ANIMA_TRANSFORMER_PREFIXES)}), got unknown keys")
        self.source_prefix = source_prefix
        expected = set(model.state_dict().keys())
        self.target_to_source: dict[str, str] = {}
        for source in source_keys:
            target = _convert_anima_cosmos2_key(_anima_source_to_net_key(source, source_prefix))
            if target in expected:
                self.target_to_source[target] = source
        missing = sorted(expected - set(self.target_to_source))
        if missing:
            raise RuntimeError(f'Anima streamed transformer key mismatch: missing={missing[:8]}')
        self.groups: dict[int, list[str]] = {}
        self.globals: list[str] = []
        for target in sorted(self.target_to_source):
            m = re.match('^transformer_blocks\\.(\\d+)\\.', target)
            if m:
                self.groups.setdefault(int(m.group(1)), []).append(target)
            else:
                self.globals.append(target)

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass

    def _read_target(self, target: str):
        value = self._handle.get_tensor(self.target_to_source[target])
        if value.is_floating_point():
            return value.to(device=self.device, dtype=self.dtype)
        return value.to(device=self.device)

    def _bind_targets(self, targets: list[str]) -> None:
        from accelerate.utils import set_module_tensor_to_device
        state = self.model.state_dict()
        for target in targets:
            value = self._read_target(target)
            want = tuple(state[target].shape)
            if tuple(value.shape) != want:
                raise RuntimeError(f'Anima tensor shape mismatch for {target}: {tuple(value.shape)} != {want}')
            assign_dtype = self.dtype if value.is_floating_point() else None
            set_module_tensor_to_device(self.model, target, self.device, value=value, dtype=assign_dtype)

    def _evict_targets(self, targets: list[str]) -> None:
        from accelerate.utils import set_module_tensor_to_device
        for target in targets:
            if not _set_module_tensor_to_meta(self.model, target):
                set_module_tensor_to_device(self.model, target, 'meta')

    def materialize_globals(self) -> int:
        self._bind_targets(self.globals)
        return len(self.globals)

    def register(self) -> None:

        def _wrap(module, targets: list[str]):
            raw_forward = type(module).forward
            streamer = self

            def wrapped(*args, _module=module, _raw_forward=raw_forward, _targets=targets, **kwargs):
                streamer._bind_targets(_targets)
                try:
                    return _raw_forward(_module, *args, **kwargs)
                finally:
                    streamer._evict_targets(_targets)
            module.forward = wrapped
            module._anima_raw_forward = raw_forward
        for i, block in enumerate(self.model.transformer_blocks):
            targets = sorted(self.groups.get(i, []))
            if targets:
                _wrap(block, targets)
        self.model._anima_streamer = self
_ANIMA_BLOCK_RE = re.compile('^transformer_blocks\\.(\\d+)\\.')

def _anima_ds_target(source_key: str, source_prefix):
    return _convert_anima_cosmos2_key(_anima_source_to_net_key(source_key, source_prefix))

def _anima_group_blocks(idx, source_prefix, expected):
    blocks, globs = ({}, [])
    for name, ref in idx.refs.items():
        target = _anima_ds_target(name, source_prefix)
        if target is None or target not in expected:
            continue
        m = _ANIMA_BLOCK_RE.match(target)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    ordered = {i: blocks[k] for i, k in enumerate(sorted(blocks))}
    return (ordered, globs)

class _AnimaBlockIndex:

    def __init__(self, idx, source_prefix, expected):
        self._idx = idx
        self._source_prefix = source_prefix
        self._expected = expected

    def group_by_layer(self, component=''):
        return _anima_group_blocks(self._idx, self._source_prefix, self._expected)

    def __getattr__(self, k):
        return getattr(self._idx, k)

def _attach_ds_anima_blocks(transformer, transformer_file, dtype, device, source_prefix, n_gpu_buffers=3):
    import os as _os
    from accelerate.utils import set_module_tensor_to_device
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    expected = set(transformer.state_dict().keys())
    idx = build_index(transformer_file)
    streamer = DirectStorageStreamer(_AnimaBlockIndex(idx, source_prefix, expected), component='', n_gpu_buffers=n_gpu_buffers, device=device)
    if getattr(streamer, 'mode', None) != 3 and _os.environ.get('ANIMA_ALLOW_NON_DIRECTSTORAGE') != '1':
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"Anima streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3) — R1 requires the optimised NVMe->GPU path. Set ANIMA_ALLOW_NON_DIRECTSTORAGE=1 for diagnostics.")
    if streamer.n != len(transformer.transformer_blocks):
        streamer.close()
        raise RuntimeError(f'Anima DS block count {streamer.n} != {len(transformer.transformer_blocks)} transformer_blocks (Cosmos2->diffusers block grouping mismatch)')
    block_paths = {il: [t for t in (_anima_ds_target(r.name, source_prefix) for r, _ao in streamer.plan[il][0]) if t in expected] for il in range(streamer.n)}

    def make_pre(il):

        def hook(_mod, _args):
            for name, t in streamer.wait_layer(il).items():
                target = _anima_ds_target(name, source_prefix)
                if target in expected:
                    if t.is_floating_point():
                        t = t.to(dtype)
                    set_module_tensor_to_device(transformer, target, device, value=t, dtype=dtype if t.is_floating_point() else None)
        return hook

    def make_post(il):

        def hook(_mod, _args, output):
            for target in block_paths[il]:
                if not _set_module_tensor_to_meta(transformer, target):
                    set_module_tensor_to_device(transformer, target, 'meta')
            streamer.release_layer(il)
            return output
        return hook
    transformer.register_forward_pre_hook(lambda m, a: streamer.new_pass())
    for il, block in enumerate(transformer.transformer_blocks):
        block.register_forward_pre_hook(make_pre(il))
        block.register_forward_hook(make_post(il))
    transformer._anima_ds_streamer = streamer
    return streamer

def _load_anima_streamed_transformer(transformer_file: str, anima_dir: str, dtype, device: str='cuda'):
    from accelerate import init_empty_weights
    from diffusers import CosmosTransformer3DModel
    config = os.path.join(anima_dir, 'transformer')
    cfg = CosmosTransformer3DModel.load_config(config)
    with init_empty_weights(include_buffers=False):
        transformer = CosmosTransformer3DModel.from_config(cfg)
    streamer = _AnimaTransformerStreamer(transformer, transformer_file, dtype=dtype, device=device)
    n_globals = streamer.materialize_globals()
    source_prefix = streamer.source_prefix
    streamer.close()
    ds = _attach_ds_anima_blocks(transformer, transformer_file, dtype, device, source_prefix)
    transformer._anima_streamer = ds
    print(f'[ANIMA-STREAM] DS-streamed transformer: {n_globals} globals resident, {ds.n} blocks via DirectStorage (mode={ds.mode})', flush=True)
    return transformer.eval()

def _load_anima_transformer(transformer_file: str, anima_dir: str, dtype, device: str='cuda', stream_weights: bool=False):
    from diffusers import CosmosTransformer3DModel
    from .direct_gpu_load import load_safetensors_direct, safetensors_keys
    if stream_weights:
        return _load_anima_streamed_transformer(transformer_file, anima_dir, dtype, device=device)
    config = os.path.join(anima_dir, 'transformer')
    state_dict = None
    if os.path.isfile(transformer_file) and transformer_file.lower().endswith('.safetensors'):
        prefix = _anima_transformer_prefix(safetensors_keys(transformer_file))
        state_dict = load_safetensors_direct(transformer_file, device)
        if prefix and prefix != 'net.':
            state_dict, source_prefix = _normalize_anima_transformer_state_dict(state_dict)
            print(f"[ANIMA] normalized transformer keys from {source_prefix!r} to 'net.' for {transformer_file}", flush=True)
    transformer = None
    if state_dict is not None:
        try:
            import torch as _torch
            from accelerate import init_empty_weights
            from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers as _convert
            converted = _convert(dict(state_dict))
            cfg = CosmosTransformer3DModel.load_config(config)
            with init_empty_weights(include_buffers=False):
                transformer = CosmosTransformer3DModel.from_config(cfg)
            missing, _unexpected = transformer.load_state_dict(converted, strict=False, assign=True)
            if missing:
                raise RuntimeError(f'manual converter load left {len(missing)} params on meta, e.g. {missing[:5]}')
            transformer = transformer.to(device=device, dtype=dtype)
        except Exception as exc:
            print(f'[ANIMA] manual converter load failed ({type(exc).__name__}: {exc}); falling back to from_single_file', flush=True)
            transformer = None
    if transformer is None:
        transformer = CosmosTransformer3DModel.from_single_file(transformer_file, config=config, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={'': device})
    _assert_no_meta_params(transformer, transformer_file)
    return transformer

def _assert_no_meta_params(module, source: str) -> None:
    meta_params = [name for name, param in module.named_parameters() if getattr(param, 'is_meta', False)]
    if meta_params:
        raise RuntimeError(f'Anima transformer load from {source!r} left {len(meta_params)} parameters on the meta device (first: {meta_params[:5]}). The checkpoint key layout likely does not match the Anima/Cosmos transformer converter.')

def _load_qwen3_text_encoder(qwen_file: str, anima_dir: str, dtype, device: str='cuda'):
    from transformers import Qwen3Model, Qwen3Config
    from .direct_gpu_load import load_safetensors_direct
    cfg = json.load(open(os.path.join(anima_dir, 'text_encoder', 'config.json'), encoding='utf-8'))
    qcfg = Qwen3Config(**{k: v for k, v in cfg.items() if not k.startswith('_')})
    try:
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=False):
            te = Qwen3Model(qcfg)
        assign = True
    except Exception:
        te = Qwen3Model(qcfg)
        assign = False
    sd = load_safetensors_direct(qwen_file, device)
    sd = {k[6:] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing, unexpected = te.load_state_dict(sd, strict=False, assign=assign)
    if missing:
        raise RuntimeError(f'qwen text-encoder load left {len(missing)} params unfilled (e.g. {missing[:3]}) — the comfy file may be a different qwen variant')
    return te.to(dtype)

def build_anima_pipeline(transformer_file: str=DEFAULT_TF, qwen_file: str=DEFAULT_QWEN, anima_dir: str=DEFAULT_DIR, device: str='cuda', dtype=None, loras=None, stream_weights: bool=False):
    import torch
    from diffusers import AnimaAutoBlocks, AutoencoderKLQwenImage, AnimaTextConditioner, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer, T5Tokenizer
    dtype = dtype or torch.bfloat16

    def _uss(tag):
        if os.environ.get('STUDIO_USS_TRACE'):
            try:
                import psutil
                print(f'[USS] anima {tag}: {psutil.Process().memory_full_info().uss / 1024 ** 3:.2f}GB', flush=True)
            except Exception:
                pass
    _uss('start')
    if not stream_weights:
        _preflight_resident_ram()
        _preflight_resident_vram(device, [transformer_file, qwen_file] + [os.path.join(anima_dir, sub, fn) for sub in ('vae', 'text_conditioner') if os.path.isdir(os.path.join(anima_dir, sub)) for fn in os.listdir(os.path.join(anima_dir, sub)) if fn.endswith('.safetensors')])
    transformer = _load_anima_transformer(transformer_file, anima_dir, dtype, device=device, stream_weights=stream_weights)
    _uss('after transformer')
    if loras:
        from .lora_stack import apply_lora_stack
        apply_lora_stack(transformer, loras)
    text_encoder = _load_qwen3_text_encoder(qwen_file, anima_dir, dtype, device=device)
    _uss('after qwen text_encoder')
    vae = AutoencoderKLQwenImage.from_pretrained(anima_dir, subfolder='vae', torch_dtype=dtype, device_map=str(device))
    _uss('after vae')
    text_conditioner = AnimaTextConditioner.from_pretrained(anima_dir, subfolder='text_conditioner', torch_dtype=dtype, device_map=str(device))
    _uss('after text_conditioner')
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(anima_dir, subfolder='scheduler')
    tokenizer = Qwen2Tokenizer.from_pretrained(anima_dir, subfolder='tokenizer')
    t5_tokenizer = T5Tokenizer.from_pretrained(anima_dir, subfolder='t5_tokenizer')
    pipe = AnimaAutoBlocks().init_pipeline(anima_dir)
    pipe.update_components(transformer=transformer, text_encoder=text_encoder, vae=vae, text_conditioner=text_conditioner, scheduler=scheduler, tokenizer=tokenizer, t5_tokenizer=t5_tokenizer)
    streamed_transformer = getattr(transformer, '_anima_streamer', None) is not None
    for comp in (text_encoder,) if streamed_transformer else (transformer, text_encoder):
        try:
            comp.to(device)
        except Exception:
            pass
    return pipe

def configure_anima_sampling(pipe, *, sampler: str | None=None, sigmas: str | None=None, num_inference_steps: int=24) -> list[float] | None:
    from cosmos_windows.res4lyf_sampling import configure_diffusers_flow_scheduler
    return configure_diffusers_flow_scheduler(pipe, sampler=sampler, sigmas=sigmas, num_inference_steps=int(num_inference_steps or 24), log_prefix='[ANIMA]')

def _anima_encode_init_latent(pipe, image_path: str, height: int, width: int, device: str='cuda'):
    import numpy as np
    import torch
    from PIL import Image
    vae = pipe.vae
    img = Image.open(image_path).convert('RGB').resize((int(width), int(height)), Image.LANCZOS)
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(img)).float().div(127.5).sub(1.0)
        x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, vae.dtype)
        z = vae.encode(x).latent_dist.mode()
        zdim = int(vae.config.z_dim)
        mean = torch.tensor(vae.config.latents_mean).view(1, zdim, 1, 1, 1).to(z)
        std = torch.tensor(vae.config.latents_std).view(1, zdim, 1, 1, 1).to(z)
        z_norm = (z - mean) / std
        return z_norm.to(torch.float32).contiguous()

def anima_generate(pipe, prompt, *, image_path: str | None=None, strength: float=0.6, height: int=768, width: int=768, num_inference_steps: int=24, sampler: str | None=None, sigmas: str | None=None, seed: int=42, output: str='images', **kw):
    import numpy as np
    import torch
    from .sdedit import flow_start_index, flow_noise_latent, flow_shift_sigma
    custom = None
    if sampler or sigmas:
        custom = configure_anima_sampling(pipe, sampler=sampler, sigmas=sigmas, num_inference_steps=num_inference_steps)
    gen = torch.Generator(device='cuda').manual_seed(int(seed))
    if not image_path:
        return pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, sigmas=custom, generator=gen, output=output, **kw)
    full = list(custom) if custom is not None else list(np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps))
    start = flow_start_index(len(full), strength)
    raw_sigma = float(full[start])
    sched = getattr(pipe, 'scheduler', None)
    try:
        shift = float(getattr(getattr(sched, 'config', None), 'shift', 1.0) or 1.0)
        if bool(getattr(getattr(sched, 'config', None), 'use_dynamic_shifting', False)):
            shift = 1.0
    except Exception:
        shift = 1.0
    start_sigma = flow_shift_sigma(raw_sigma, shift)
    with torch.no_grad():
        z0 = _anima_encode_init_latent(pipe, image_path, height, width)
        noise = torch.randn(z0.shape, generator=gen, device=z0.device, dtype=z0.dtype)
        x_t = flow_noise_latent(z0, noise, start_sigma)
    print(f'[ANIMA] SDEdit i2i: strength={strength} -> start step {start}/{len(full)} raw_sigma={raw_sigma:.4f} shift={shift} sigma0={start_sigma:.4f}', flush=True)
    return pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, sigmas=full[start:], latents=x_t, generator=gen, output=output, **kw)
DEFAULT_CONTROLNET_DIR = ''
_LLLITE_KINDS = {'depth': 'anima-lllite-depth-1.safetensors', 'lineart': 'anima-lllite-lineart-1.safetensors', 'scribble': 'anima-lllite-scribble-1.safetensors', 'pose': 'anima-lllite-pose-1.safetensors', 'inpainting': 'anima-lllite-inpainting-v2.safetensors', 'any': 'anima-lllite-any-test-like-v2.safetensors'}

def _resolve_lllite_path(kind_or_path: str, controlnet_dir: str=DEFAULT_CONTROLNET_DIR) -> str:
    import glob
    if os.path.isfile(kind_or_path):
        return kind_or_path
    fname = _LLLITE_KINDS.get(kind_or_path.lower())
    if fname:
        p = os.path.join(controlnet_dir, fname)
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(os.path.join(controlnet_dir, f'anima-lllite-{kind_or_path.lower()}*.safetensors')))
    if hits:
        return hits[0]
    raise FileNotFoundError(f'LLLite weights not found for {kind_or_path!r} in {controlnet_dir}. Known kinds: {sorted(_LLLITE_KINDS)}; or pass a full .safetensors path.')

def build_anima_controlnet(control_kind: str, transformer_file: str=DEFAULT_TF, qwen_file: str=DEFAULT_QWEN, anima_dir: str=DEFAULT_DIR, controlnet_dir: str=DEFAULT_CONTROLNET_DIR, device: str='cuda', dtype=None, strength: float=1.0, loras=None):
    import torch
    from .anima_lllite import attach_lllite
    dtype = dtype or torch.bfloat16
    pipe = build_anima_pipeline(transformer_file=transformer_file, qwen_file=qwen_file, anima_dir=anima_dir, device=device, dtype=dtype, loras=loras)
    transformer = pipe.get_component('transformer') if hasattr(pipe, 'get_component') else getattr(pipe, 'transformer', None)
    if transformer is None:
        transformer = pipe.components['transformer'] if hasattr(pipe, 'components') else None
    if transformer is None:
        raise RuntimeError('could not reach the transformer component on the Anima pipeline to attach LLLite')
    weights_path = _resolve_lllite_path(control_kind, controlnet_dir)
    lllite = attach_lllite(transformer, weights_path, strength=strength, device=device, dtype=dtype)
    return (pipe, lllite)
if __name__ == '__main__':
    import sys, time, torch
    from .gpu_select import pick_free_gpu, assert_gpu_idle
    assert_gpu_idle()
    pick_free_gpu()
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'a highly detailed anime portrait, cinematic lighting'
    pipe = build_anima_pipeline()
    t = time.time()
    out = pipe(prompt=prompt, height=768, width=768, num_inference_steps=24, generator=torch.Generator('cuda').manual_seed(42), output='images')
    imgs = out if isinstance(out, list) else getattr(out, 'images', out)
    img = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    os.makedirs('', exist_ok=True)
    img.save('')
    print(f'RESULT ANIMA_OK {time.time() - t:.0f}s peak {torch.cuda.max_memory_allocated() / 1000000000.0:.1f}')
