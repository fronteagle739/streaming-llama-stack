# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/hunyuan_gen.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import inspect
import math
import os
import re
import sys
import time
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _image_candidates(result) -> list:
    out: list = []

    def _walk(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            for key in ('samples', 'image', 'images', 'output', 'outputs', 'path'):
                if key in obj:
                    _walk(obj[key])
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
            return
        if isinstance(obj, str):
            if os.path.isfile(obj):
                out.append(obj)
            return
        out.append(obj)
    if hasattr(result, 'samples'):
        _walk(getattr(result, 'samples'))
    else:
        _walk(result)
    return out

def _save_one_hunyuan_image(img, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    if img is None:
        raise RuntimeError('generate_image returned no image/path')
    if isinstance(img, str):
        if not os.path.isfile(img):
            raise RuntimeError(f'generate_image returned a missing path: {img}')
        if os.path.abspath(img) != os.path.abspath(out_path):
            import shutil
            shutil.copyfile(img, out_path)
    elif hasattr(img, 'save'):
        img.save(out_path)
    else:
        try:
            from cosmos_windows import mem_guard as _mg
            if _mg.free_gb() < 4.0:
                raise RuntimeError(f'only {_mg.free_gb():.1f}GB free RAM before tensor image materialization')
        except RuntimeError:
            raise
        except Exception:
            pass
        try:
            import numpy as _np
            from PIL import Image as _Image
            try:
                import torch as _torch
                if isinstance(img, _torch.Tensor):
                    img = img.detach().float().cpu()
                    while img.ndim > 3:
                        img = img[0]
                    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                        img = img.permute(1, 2, 0)
                    img = img.numpy()
            except Exception:
                pass
            arr = _np.asarray(img)
            if arr.dtype != _np.uint8:
                arr = arr.astype(_np.float32)
                if arr.min() < 0:
                    arr = (arr.clip(-1, 1) + 1) / 2
                if arr.max() <= 1.5:
                    arr = (arr.clip(0, 1) * 255).astype(_np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(_np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            _Image.fromarray(arr).save(out_path)
        except Exception as exc:
            raise RuntimeError(f'unsupported generate_image return type {type(img).__name__}: {exc}') from exc
    if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 0:
        raise RuntimeError(f'save reported success but output is missing/empty: {out_path}')

def _save_hunyuan_result(result, out_path: str) -> None:
    images = _image_candidates(result)
    if not images:
        raise RuntimeError('generate_image returned no image/path')
    _save_one_hunyuan_image(images[0], out_path)

def _safe_name(text: str, max_len: int=40) -> str:
    s = re.sub('[^A-Za-z0-9._-]+', '_', (text or 'image').strip()).strip('._-')
    return (s or 'image')[:max_len]

def _clamp_denoise(value: float | None) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value)))

def _resize_cover(image, width: int, height: int):
    from PIL import Image
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))

def _load_cover_image(path: str, width: int, height: int):
    from PIL import Image
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f'init image not found: {path}')
    with Image.open(path) as im:
        return _resize_cover(im.convert('RGB'), width, height)

def _native_conditioning_images(source: str, references: list[str]):
    from PIL import Image
    images = []
    for path in [source, *references]:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return images

def _save_resized_source(image_path: str, out_path: str, width: int, height: int) -> None:
    img = _load_cover_image(image_path, width, height)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    img.save(out_path)

def _final_mask_composite(out_path: str, image_path: str | None, mask_path: str | None) -> None:
    if not image_path or not mask_path or (not os.path.isfile(out_path)) or (not os.path.isfile(mask_path)):
        return
    from PIL import Image
    with Image.open(out_path) as generated:
        with Image.open(mask_path) as mask_src:
            with Image.open(image_path) as init_src:
                mask = _resize_cover(mask_src.convert('L'), generated.width, generated.height)
                init = _resize_cover(init_src.convert('RGB'), generated.width, generated.height)
                Image.composite(generated.convert('RGB'), init, mask).save(out_path)

def _hunyuan_effective_steps(num_steps: int, denoise: float) -> int:
    if denoise <= 0.0:
        return 0
    return max(1, min(int(num_steps), int(math.ceil(denoise * int(num_steps)))))

def _hunyuan_skip_steps(num_steps: int, denoise: float) -> int:
    return max(0, min(int(num_steps) - 1, int(num_steps) - _hunyuan_effective_steps(num_steps, denoise)))

def _mask_to_hunyuan_latents(mask_path: str | None, *, width: int, height: int, latent_hw: tuple[int, int], device):
    if not mask_path:
        return None
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f'mask image not found: {mask_path}')
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    with _Image.open(mask_path) as im:
        mask = _resize_cover(im.convert('L'), width, height)
    arr = _np.asarray(mask, dtype=_np.float32) / 255.0
    t = _torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device)
    t = _torch.nn.functional.interpolate(t, size=latent_hw, mode='area')
    return t.clamp(0.0, 1.0)

def _install_hunyuan_img2img_call(pipe, model, kw: dict, spec: dict):
    import torch
    denoise = _clamp_denoise(spec.get('denoise'))
    num_steps = int(kw.get('num_inference_steps') or spec.get('steps') or 50)
    skip = _hunyuan_skip_steps(num_steps, denoise)
    batch_size = int(kw.get('batch_size') or 1)
    image_size = list(kw.get('image_size') or spec.get('image_size') or [])
    if len(image_size) != 2:
        raise RuntimeError(f'Hunyuan img2img expected image_size [height,width], got {image_size!r}')
    height, width = (int(image_size[0]), int(image_size[1]))
    device = getattr(pipe, '_execution_device', None) or getattr(model, 'device', None) or torch.device('cuda')
    if not isinstance(device, torch.device):
        device = torch.device(device)
    generators = kw.get('generator')
    seed = int(spec.get('seed') or 0)
    if not isinstance(generators, list) or not generators:
        generators = [torch.Generator(device).manual_seed(seed + i) for i in range(batch_size)]
    init_image = _load_cover_image(str(spec['image_path']), width, height)
    init_tensor = model.image_processor.pil_image_to_tensor(init_image).unsqueeze(0).to(device=device)
    _t, clean = model.vae_encode(init_tensor, generator=generators[0])
    clean = clean.float()
    if clean.shape[0] != batch_size:
        clean = clean.repeat(batch_size, 1, 1, 1)[:batch_size]
    orig_set_timesteps = pipe.scheduler.set_timesteps
    orig_set_timesteps(num_steps, device=device)
    full_sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
    sigma_start = full_sigmas[skip]
    noise_chunks = [torch.randn(clean[0:1].shape, generator=generators[i % len(generators)], device=device, dtype=torch.float32) for i in range(batch_size)]
    noise = torch.cat(noise_chunks, dim=0)
    latents = ((1.0 - sigma_start) * clean + sigma_start * noise).to(torch.bfloat16)
    mask = _mask_to_hunyuan_latents(spec.get('mask_path'), width=width, height=height, latent_hw=(clean.shape[-2], clean.shape[-1]), device=device)
    if mask is not None:
        latents = (mask * latents.float() + (1.0 - mask) * ((1.0 - sigma_start) * clean + sigma_start * noise)).to(torch.bfloat16)

    def _sliced_set_timesteps(num_inference_steps: int, device=None, **kwargs):
        orig_set_timesteps(num_inference_steps, device=device, **kwargs)
        sched = pipe.scheduler
        sched.sigmas = sched.sigmas[skip:].contiguous()
        sched.timesteps = sched.timesteps[skip:].contiguous()
        sched.timesteps_full = sched.timesteps_full[skip:].contiguous()
        sched.num_inference_steps = int(sched.timesteps.shape[0])
        sched._step_index = None
        sched._begin_index = None
    pipe.scheduler.set_timesteps = _sliced_set_timesteps
    kw['latents'] = latents
    spec['state'] = {'clean': clean, 'noise': noise, 'mask': mask}
    print(f'[hunyuan] img2img denoise={denoise:.2f}; running {num_steps - skip}/{num_steps} streamed steps', flush=True)

    def _restore():
        pipe.scheduler.set_timesteps = orig_set_timesteps
    return _restore

def _preserve_hunyuan_mask_latents(pipe, step: int, callback_kwargs: dict, spec: dict):
    state = spec.get('state') or {}
    mask = state.get('mask')
    if mask is None:
        return callback_kwargs
    latents = callback_kwargs.get('latents')
    if latents is None:
        return callback_kwargs
    import torch
    sigmas = getattr(pipe.scheduler, 'sigmas', None)
    if sigmas is None or len(sigmas) == 0:
        return callback_kwargs
    idx = min(int(step) + 1, int(sigmas.shape[0]) - 1)
    sigma_next = sigmas[idx].to(device=latents.device, dtype=torch.float32)
    clean = state['clean'].to(device=latents.device, dtype=torch.float32)
    noise = state['noise'].to(device=latents.device, dtype=torch.float32)
    mask = mask.to(device=latents.device, dtype=torch.float32)
    original_at_sigma = (1.0 - sigma_next) * clean + sigma_next * noise
    callback_kwargs['latents'] = (mask * latents.float() + (1.0 - mask) * original_at_sigma).to(latents.dtype)
    return callback_kwargs

def _batch_output_paths(out_dir: str, prompts: list[str], seeds: list[int], same_prompt: bool) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        tag = f'seed{seed}' if same_prompt else f'p{i:02d}_seed{seed}_{_safe_name(prompt, 24)}'
        paths.append(os.path.join(out_dir, f'hunyuan_{tag}.png'))
    return paths

def _patch_batch_text_scalars(model, batch_size: int) -> None:
    if batch_size <= 1 or getattr(model, '_studio_batch_text_patch', False):
        return
    raw = model.prepare_model_inputs

    def _wrapped(self, *args, **kwargs):
        prompt = kwargs.get('prompt')
        if prompt is None and args:
            prompt = args[0]
        if isinstance(prompt, list):
            n = len(prompt)
            for key in ('system_prompt', 'cot_text'):
                value = kwargs.get(key)
                if isinstance(value, str):
                    kwargs[key] = [value] * n
                elif isinstance(value, list) and len(value) == 1 and (n > 1):
                    kwargs[key] = value * n
        return raw(*args, **kwargs)
    model.prepare_model_inputs = types.MethodType(_wrapped, model)
    model._studio_batch_text_patch = True
    print(f'[hunyuan] patched scalar system/cot text for native batch size={batch_size}', flush=True)

class _BatchSafeRoPE:

    def __init__(self, raw):
        object.__setattr__(self, '_raw', raw)
        object.__setattr__(self, '_studio_batch_rope_proxy', True)

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __setattr__(self, name, value):
        return setattr(self._raw, name, value)

    def __call__(self, seq_len, device, rope_image_info=None, position_ids=None):
        if position_ids is None:
            return self._raw(seq_len, device, rope_image_info=rope_image_info, position_ids=None)
        cos_cache, sin_cache = self._raw(seq_len, device, rope_image_info=rope_image_info, position_ids=None)
        want = int(position_ids.shape[0])
        have = int(cos_cache.shape[0])
        if have != want:
            if have <= 0 or want % have != 0:
                raise RuntimeError(f'Hunyuan RoPE cache batch {have} cannot match position batch {want}')
            repeats = want // have
            cos_cache = cos_cache.repeat(repeats, 1, 1)
            sin_cache = sin_cache.repeat(repeats, 1, 1)
        head_size = cos_cache.size(-1)
        idx = position_ids.unsqueeze(-1).expand(-1, -1, head_size)
        return (cos_cache.gather(dim=1, index=idx), sin_cache.gather(dim=1, index=idx))

def _align_tensor_batch(t, target: int):
    have = int(t.shape[0])
    want = int(target)
    if want <= 0 or have == want:
        return t
    if have > want:
        return t[:want]
    repeats = (want + have - 1) // have
    shape = [1] * t.ndim
    shape[0] = repeats
    return t.repeat(*shape)[:want]

def _align_rotary_batch(q, cos, sin):
    want = int(q.shape[0])
    return (_align_tensor_batch(cos, want), _align_tensor_batch(sin, want))

def _patch_batch_rotary_alignment(model, batch_size: int) -> None:
    if batch_size <= 1:
        return
    mod = sys.modules.get(type(model).__module__)
    raw = getattr(mod, 'apply_rotary_pos_emb', None) if mod is not None else None
    if raw is None or getattr(raw, '_studio_batch_rotary_align', False):
        return

    def _wrapped(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if position_ids is None:
            try:
                cos, sin = _align_rotary_batch(q, cos, sin)
            except Exception:
                pass
        return raw(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    _wrapped._studio_batch_rotary_align = True
    mod.apply_rotary_pos_emb = _wrapped
    print(f'[hunyuan] patched rotary application for native batch size={batch_size}', flush=True)

def _patch_batch_modulation_alignment(model, batch_size: int) -> None:
    if batch_size <= 1:
        return
    mod = sys.modules.get(type(model).__module__)
    if mod is None:
        return
    patched = 0

    def _wrap_class_forward(cls):
        raw = getattr(cls, 'forward', None)
        if raw is None or getattr(raw, '_studio_batch_mod_align', False):
            return False
        try:
            params = list(inspect.signature(raw).parameters)
            if len(params) < 3:
                return False
        except Exception:
            return False

        def _wrapped(self, x, t=None, *args, **kwargs):
            if t is not None and hasattr(x, 'shape') and hasattr(t, 'shape'):
                try:
                    if int(t.shape[0]) != int(x.shape[0]):
                        t = _align_tensor_batch(t, int(x.shape[0]))
                except Exception:
                    pass
            for key in ('temb', 'timestep_emb', 'time_emb', 'conditioning'):
                val = kwargs.get(key)
                if val is not None and hasattr(x, 'shape') and hasattr(val, 'shape'):
                    try:
                        if int(val.shape[0]) != int(x.shape[0]):
                            kwargs[key] = _align_tensor_batch(val, int(x.shape[0]))
                    except Exception:
                        pass
            return raw(self, x, t, *args, **kwargs)
        _wrapped._studio_batch_mod_align = True
        setattr(cls, 'forward', _wrapped)
        return True
    for name, obj in vars(mod).items():
        if not isinstance(obj, type):
            continue
        lname = name.lower()
        if any((token in lname for token in ('modulation', 'adaln', 'timestep', 'timeembed', 'time_embed', 'resblock', 'unetdown', 'unetup'))):
            if _wrap_class_forward(obj):
                patched += 1
    try:
        import torch.nn as _nn
        for module in model.modules():
            if not isinstance(module, _nn.Module):
                continue
            cls = type(module)
            lname = cls.__name__.lower()
            if not any((token in lname for token in ('modulation', 'adaln', 'timestep', 'timeembed', 'time_embed', 'resblock', 'unetdown', 'unetup'))):
                continue
            if _wrap_class_forward(cls):
                patched += 1
    except Exception:
        pass
    if patched:
        print(f'[hunyuan] patched timestep/modulation batch alignment on {patched} class(es)', flush=True)
    else:
        print('[hunyuan] timestep/modulation batch patch skipped: no matching classes found', flush=True)

def _patch_batch_prepare_inputs(model, batch_size: int) -> None:
    if batch_size <= 1 or getattr(model, '_studio_batch_prepare_inputs_patch', False):
        return
    raw = model.prepare_inputs_for_generation

    def _wrapped(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, tokenizer_output=None, batch_gen_image_info=None, batch_cond_images=None, infer_align_image_size=False, generator=None, **kwargs):
        position_ids = kwargs.get('position_ids')
        target = None
        if input_ids is not None:
            target = int(input_ids.shape[0])
        elif inputs_embeds is not None:
            target = int(inputs_embeds.shape[0])
        elif kwargs.get('images') is not None:
            target = int(kwargs['images'].shape[0])
        if target is not None and position_ids is not None and (getattr(position_ids, 'dim', lambda: 0)() >= 1):
            if int(position_ids.shape[0]) != target:
                kwargs['position_ids'] = _align_tensor_batch(position_ids, target)
        if target is not None and attention_mask is not None and (getattr(attention_mask, 'dim', lambda: 0)() >= 1):
            if int(attention_mask.shape[0]) != target:
                attention_mask = _align_tensor_batch(attention_mask, target)
        return raw(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, tokenizer_output=tokenizer_output, batch_gen_image_info=batch_gen_image_info, batch_cond_images=batch_cond_images, infer_align_image_size=infer_align_image_size, generator=generator, **kwargs)
    model.prepare_inputs_for_generation = types.MethodType(_wrapped, model)
    model._studio_batch_prepare_inputs_patch = True
    print(f'[hunyuan] patched prepare_inputs batch alignment for native batch size={batch_size}', flush=True)

def _make_batch_safe_cache_update(raw_update):

    def _wrapped(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_kwargs = dict(cache_kwargs or {})
        target = int(key_states.shape[0])
        cache_position = cache_kwargs.get('cache_position')
        if cache_position is not None and getattr(cache_position, 'dim', lambda: 0)() >= 1:
            if int(cache_position.shape[0]) != target:
                cache_kwargs['cache_position'] = _align_tensor_batch(cache_position, target)
        layer = self.layers[layer_idx]
        for attr in ('keys', 'values'):
            tensor = getattr(layer, attr, None)
            if tensor is not None and int(tensor.shape[0]) != target:
                setattr(layer, attr, _align_tensor_batch(tensor, target).contiguous())
        return raw_update(self, key_states, value_states, layer_idx, cache_kwargs)
    _wrapped._studio_batch_cache_align = True
    return _wrapped

def _patch_batch_static_cache(model, batch_size: int) -> None:
    if batch_size <= 1:
        return
    mod = sys.modules.get(type(model).__module__)
    cache_cls = getattr(mod, 'HunyuanStaticCache', None) if mod is not None else None
    if cache_cls is None:
        print('[hunyuan] static-cache batch patch skipped: HunyuanStaticCache not found', flush=True)
        return
    raw = getattr(cache_cls, 'update', None)
    if raw is None or getattr(raw, '_studio_batch_cache_align', False):
        return
    cache_cls.update = _make_batch_safe_cache_update(raw)
    print(f'[hunyuan] patched static cache batch alignment for native batch size={batch_size}', flush=True)

def _patch_batch_rope(model, batch_size: int) -> None:
    if batch_size <= 1:
        return
    patched: list[str] = []
    seen: set[int] = set()
    for name, obj in (('', model), ('model', getattr(model, 'model', None))):
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        rope = getattr(obj, 'cached_rope', None)
        if rope is None or getattr(rope, '_studio_batch_rope_proxy', False):
            continue
        setattr(obj, 'cached_rope', _BatchSafeRoPE(rope))
        patched.append(name or type(obj).__name__)
    if patched:
        where = ', '.join(patched)
        print(f'[hunyuan] patched RoPE cache on {where} for CFG-expanded native batch size={batch_size}', flush=True)
    else:
        print('[hunyuan] RoPE batch patch skipped: no unwrapped cached_rope found', flush=True)

def _selftest_batch_rope() -> int:
    import torch

    class _RawRoPE:

        def __call__(self, seq_len, device, rope_image_info=None, position_ids=None):
            cos = torch.arange(2 * seq_len * 4, dtype=torch.float32, device=device).reshape(2, seq_len, 4)
            sin = cos + 1000
            if position_ids is None:
                return (cos, sin)
            head_size = cos.size(-1)
            idx = position_ids.unsqueeze(-1).expand(-1, -1, head_size)
            return (cos.gather(dim=1, index=idx), sin.gather(dim=1, index=idx))

    class _Model:

        def __init__(self):
            self.cached_rope = _RawRoPE()
            self.model = type('_Backbone', (), {})()
    model = _Model()
    pos = torch.zeros((4, 3), dtype=torch.long)
    try:
        model.cached_rope(8, torch.device('cpu'), position_ids=pos)
    except RuntimeError:
        pass
    else:
        print('[hunyuan:test] expected raw RoPE to fail on 2->4 batch mismatch', flush=True)
        return 1
    _patch_batch_rope(model, 2)
    cos, sin = model.cached_rope(8, torch.device('cpu'), position_ids=pos)
    q = torch.zeros((2, 2, 3, 4))
    cos2, sin2 = _align_rotary_batch(q, cos, sin)
    pos2 = _align_tensor_batch(pos, 2)

    class _Layer:

        def __init__(self):
            self.keys = None
            self.values = None

        def lazy_initialization(self, key_states):
            shape = (key_states.shape[0], key_states.shape[1], 8, key_states.shape[3])
            self.keys = torch.zeros(shape, dtype=key_states.dtype)
            self.values = torch.zeros(shape, dtype=key_states.dtype)

    class _Cache:

        def __init__(self):
            self.layers = [_Layer()]
            self.dynamic = False

    def _raw_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = cache_kwargs.get('cache_position')
        if self.layers[layer_idx].keys is None:
            self.layers[layer_idx].lazy_initialization(key_states)
        k_out = self.layers[layer_idx].keys
        v_out = self.layers[layer_idx].values
        batch_size = cache_position.shape[0]
        assert batch_size == k_out.size(0)
        assert batch_size == v_out.size(0)
        assert batch_size == key_states.size(0)
        assert batch_size == value_states.size(0)
        return (k_out, v_out)
    cache = _Cache()
    wrapped_update = _make_batch_safe_cache_update(_raw_update)
    small_k = torch.zeros((2, 1, 3, 4))
    small_v = torch.zeros((2, 1, 3, 4))
    big_k = torch.zeros((4, 1, 3, 4))
    big_v = torch.zeros((4, 1, 3, 4))
    k_small, _ = wrapped_update(cache, small_k, small_v, 0, {'cache_position': pos})
    k_big, _ = wrapped_update(cache, big_k, big_v, 0, {'cache_position': pos})

    class ResBlock(torch.nn.Module):

        def forward(self, x, t):
            scale = t[:, None, None]
            shift = t[:, None, None]
            return x * (1 + scale) + shift

    class _BatchModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.mod = ResBlock()
    bm = _BatchModel()
    x = torch.zeros((2, 3, 4))
    t = torch.zeros((8,))
    try:
        bm.mod(x, t)
    except RuntimeError:
        pass
    else:
        print('[hunyuan:test] expected raw modulation to fail on 2->8 batch mismatch', flush=True)
        return 1
    _patch_batch_modulation_alignment(bm, 2)
    mod_out = bm.mod(x, t)
    ok = cos.shape == (4, 3, 4) and sin.shape == (4, 3, 4) and (cos2.shape == (2, 3, 4)) and (sin2.shape == (2, 3, 4)) and (pos2.shape == (2, 3)) and (k_small.shape[0] == 2) and (k_big.shape[0] == 4) and (mod_out.shape == x.shape) and getattr(model.cached_rope, '_studio_batch_rope_proxy', False)
    print(f"[hunyuan:test] batch RoPE proxy {('PASS' if ok else 'FAIL')} cos={tuple(cos.shape)} aligned={tuple(cos2.shape)} cache={tuple(k_big.shape)}", flush=True)
    return 0 if ok else 1

def _wire_pipeline_progress(model, total_steps, out_dir, img2img_spec: dict | None=None):
    pipe = getattr(model, 'pipeline', None)
    if pipe is None:
        print('  [hunyuan] no .pipeline to wire progress onto (progress bar unavailable)', flush=True)
        return
    PipeClass = type(pipe)
    if getattr(PipeClass, '_studio_progress_wired', False):
        return
    import os as _os
    prev_dir = _os.path.join(out_dir, '.previews')
    state = {'vae_cpu': None, 'logged': False, 'lazy_logged': False}
    every = max(1, int(total_steps) // 4)

    def _cpu_preview(p, step, latents):
        try:
            try:
                from cosmos_windows import mem_guard as _mg
                _free = _mg.free_gb()
                if _free < 4.0:
                    print(f'  [hunyuan] preview skipped: low RAM ({_free:.1f} GB free, need >= 4)', flush=True)
                    return
            except Exception:
                pass
            import torch as _pt, numpy as _np
            from PIL import Image as _Im
            vae = getattr(p, 'vae', None) or getattr(model, 'vae', None)
            if vae is None or latents is None:
                return
            if getattr(model, '_hy_vae_lazy', False) and (not getattr(model, '_hy_vae_loaded', False)):
                if not state['lazy_logged']:
                    state['lazy_logged'] = True
                    print('  [hunyuan] preview skipped: VAE is lazy and will load only for final tiled decode', flush=True)
                return
            if state['vae_cpu'] is None:
                import copy as _copy
                state['vae_cpu'] = _copy.deepcopy(vae).to('cpu').float().eval()
            vcpu = state['vae_cpu']
            lat = latents.detach().to('cpu').float()
            sf = getattr(getattr(vcpu, 'config', None), 'scaling_factor', None)
            if sf:
                lat = lat / sf
            with _pt.no_grad():
                out = vcpu.decode(lat)
            dec = getattr(out, 'sample', None)
            if dec is None:
                dec = out[0] if isinstance(out, (list, tuple)) else out
            img = dec[0]
            while img.dim() > 3:
                img = img[:, 0]
            img = (img.float().clamp(-1, 1) + 1) / 2
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(_np.uint8)
            im = _Im.fromarray(arr)
            im.thumbnail((384, 384))
            _os.makedirs(prev_dir, exist_ok=True)
            fp = _os.path.join(prev_dir, f'hy_prev_{step}.png')
            im.save(fp)
            print(f'[hunyuan] preview {fp}', flush=True)
        except Exception as e:
            if not state['logged']:
                state['logged'] = True
                shp = tuple(latents.shape) if hasattr(latents, 'shape') else None
                print(f'  [hunyuan] preview decode skipped: {type(e).__name__}: {str(e)[:120]} (latent shape={shp})', flush=True)
    import time as _time
    tstate = {'last': None}

    def _callback(p, step, timestep, cbkw):
        try:
            s = int(step) + 1
            now = _time.time()
            dt = now - tstate['last'] if tstate['last'] else None
            tstate['last'] = now
            extra = f' · {dt:.1f}s/step (~{144.0 / dt:.1f} GB/s if stream-bound)' if dt and dt > 0 else ''
            print(f'[hunyuan] step {s}/{total_steps} denoising...{extra}', flush=True)
            if os.environ.get('HY_PREVIEWS') and (s % every == 0 or s == total_steps):
                _cpu_preview(p, s, cbkw.get('latents'))
        except Exception as _e:
            print(f'  [hunyuan] progress callback note (ignored): {_e}', flush=True)
        if img2img_spec is not None:
            try:
                cbkw = _preserve_hunyuan_mask_latents(p, step, cbkw, img2img_spec)
            except Exception as _e:
                print(f'  [hunyuan] img2img mask preservation note (ignored): {_e}', flush=True)
        return cbkw
    _orig = PipeClass.__call__

    def _wrapped(self, *args, **kw):
        restore_img2img = None
        if img2img_spec is not None:
            restore_img2img = _install_hunyuan_img2img_call(self, model, kw, img2img_spec)
        kw.setdefault('callback_on_step_end', _callback)
        kw.setdefault('callback_on_step_end_tensor_inputs', ['latents'])
        try:
            return _orig(self, *args, **kw)
        finally:
            if restore_img2img is not None:
                restore_img2img()
    try:
        PipeClass.__call__ = _wrapped
        PipeClass._studio_progress_wired = True
        print(f'  [hunyuan] progress + CPU previews wired ({total_steps} steps)', flush=True)
    except Exception as e:
        print(f'  [hunyuan] could not wire pipeline progress ({e}); heartbeat only', flush=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='')
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--prompts-file', default=None, help='file of prompts (one per line) for a native batch')
    ap.add_argument('--num', type=int, default=1, help='batch size when --prompt is used')
    ap.add_argument('--out', default=None, help='single-output path')
    ap.add_argument('--out-dir', default=None, help='batch output directory')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--size', default='auto')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--image', default='', help='optional init image for latent image-to-image/edit generation')
    ap.add_argument('--native-edit', action='store_true', help='Use the source as native image conditioning, without latent img2img')
    ap.add_argument('--reference-image', action='append', default=[], help='Additional native identity reference (repeatable)')
    ap.add_argument('--denoise', type=float, default=1.0, help='img2img strength: 0 keeps the source image, 1 is full regeneration')
    ap.add_argument('--mask', default='', help='optional mask for img2img/edit; white=regenerate, black=preserve')
    ap.add_argument('--gpu', default='1', help='nvidia-smi GPU index (the 5090); pinned via PCI_BUS_ID')
    ap.add_argument('--ar-mode', default='fetch-all', choices=['fetch-all', 'fetch-active', 'skip', 'mac'], help='how the 80B autoregressive reasoning pass runs (user 2026-06-04): fetch-all=stream every expert (works today); fetch-active=stream only topk-active experts (full precision, needs the per-layer gate-hook); skip=straight to diffusion; mac=run AR on the Mac peer (--mac-host).')
    ap.add_argument('--mac-host', default='', help='Mac peer host for --ar-mode mac (AR->Mac, diffusion->RTX)')
    ap.add_argument('--selftest-rope', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.selftest_rope:
        return _selftest_batch_rope()
    if not a.prompt and (not a.prompts_file):
        ap.error('one of --prompt or --prompts-file is required')
    prompts = [l.strip() for l in open(a.prompts_file, encoding='utf-8') if l.strip()] if a.prompts_file else [a.prompt] * max(1, int(a.num))
    if not prompts:
        ap.error('batch has no prompts')
    if a.native_edit and (not a.image or len(prompts) != 1 or a.mask):
        ap.error('native edit requires one prompt, a source image and no latent mask')
    if a.reference_image and (not a.native_edit):
        ap.error('reference-image requires native-edit; references must not be silently discarded')
    seeds = [int(a.seed) + i for i in range(len(prompts))]
    same_prompt = a.prompts_file is None
    is_batch = len(prompts) > 1
    if is_batch:
        out_dir = a.out_dir or (os.path.dirname(os.path.abspath(a.out)) if a.out else os.getcwd())
        out_paths = _batch_output_paths(out_dir, prompts, seeds, same_prompt)
    else:
        out_paths = [a.out or os.path.join(a.out_dir or os.getcwd(), f'hunyuan_seed{seeds[0]}.png')]
    if a.image and (not a.native_edit) and (_clamp_denoise(a.denoise) <= 0.0):
        if 'x' not in str(a.size):
            print('[hunyuan] ABORT: img2img denoise=0 needs an explicit --size HxW/WxH to save the source image.', flush=True)
            return 2
        h, w = [int(x) for x in str(a.size).lower().split('x', 1)]
        for fp in out_paths:
            _save_resized_source(a.image, fp, w, h)
            if is_batch:
                print(f'[bgen] saved {fp}', flush=True)
        print(f'RESULT DONE {out_paths[0]}', flush=True)
        return 0
    ar_mode = a.ar_mode
    if ar_mode == 'fetch-active':
        if os.environ.get('HY_FETCH_ACTIVE_GATEHOOK'):
            os.environ['WF_MOE_FETCH_ALL'] = '0'
            os.environ['WF_MOE_FETCH_ACTIVE'] = '1'
            print('[hunyuan] AR mode=fetch-active (gate-hook enabled): streaming topk-active experts', flush=True)
        else:
            print('[hunyuan] ABORT: ar_mode=fetch-active needs the per-layer gate-hook (set HY_FETCH_ACTIVE_GATEHOOK after it is installed).', flush=True)
            return 2
    elif ar_mode == 'mac':
        if not a.mac_host:
            print('[hunyuan] ABORT: ar_mode=mac needs --mac-host (STUDIO_MAC_HOST in the app).', flush=True)
            return 2
        else:
            print('[hunyuan] ABORT: ar_mode=mac is not deployed for HunyuanImage-3 yet; refusing to silently run local fetch-all.', flush=True)
            return 2
    elif ar_mode == 'skip':
        print('[hunyuan] AR mode=skip: will request generate_image to skip the AR reasoning pass if supported.', flush=True)
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from cosmos_windows import mem_guard
    mem_guard.preflight(min_free_gb=8.0, label='HunyuanImage-3 (80B) streamed load')
    mem_guard.start_watchdog(max_process_gb=24.0, min_free_gb=3.0, label='HunyuanImage-3 gen')
    try:
        from cosmos_windows.attention_memeff import enable_memeff_attention
        enable_memeff_attention(allow_math_fallback=True)
        print('[hunyuan] cuDNN-preferring SDPA enabled (SM120 O(N) attention; streaming unaffected)', flush=True)
    except Exception as e:
        print(f'[hunyuan] memeff-attention enable skipped: {e}', flush=True)
    from cosmos_windows.weight_streaming.hunyuan_backend import build_streamed_hunyuan
    print(f'[hunyuan] loading streamed model from {a.model_dir} (157GB on disk, ~15.6GB resident)...', flush=True)
    t0 = time.time()
    import threading
    _loaded = threading.Event()

    def _heartbeat():
        n = 0
        while not _loaded.wait(20):
            n += 20
            print(f'[hunyuan] still loading the 80B model... {n}s elapsed (meta-init is slow)', flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()
    built = build_streamed_hunyuan(a.model_dir)
    _loaded.set()
    model = built[0] if isinstance(built, tuple) else built
    streamer = built[1] if isinstance(built, tuple) and len(built) > 1 else None
    if not os.environ.get('HY_ALLOW_NON_STREAMING') and streamer is not None and (getattr(streamer, 'mode', None) != 3):
        print(f"[hunyuan] ABORT: streamer mode={getattr(streamer, 'mode', '?')} is NOT DirectStorage(3) — a non-streaming load would prestage the 80B in RAM on a 32 GB box. Fix DirectStorage init (COM apartment / dstorage*.dll) and retry. (override with HY_ALLOW_NON_STREAMING=1)", flush=True)
        try:
            streamer.close()
        except Exception:
            pass
        return 3
    print(f"[hunyuan] streaming confirmed: DirectStorage mode={getattr(streamer, 'mode', 'n/a')} (NVMe->GPU, weights never staged in RAM).", flush=True)
    model.eval()
    if hasattr(model, 'load_tokenizer'):
        try:
            model.load_tokenizer(a.model_dir)
        except Exception as e:
            print(f'[hunyuan] load_tokenizer note: {e}', flush=True)
    print(f'[hunyuan] model ready in {time.time() - t0:.0f}s — generating...', flush=True)
    _patch_batch_text_scalars(model, len(prompts))
    _patch_batch_prepare_inputs(model, len(prompts))
    _patch_batch_static_cache(model, len(prompts))
    _patch_batch_rope(model, len(prompts))
    _patch_batch_rotary_alignment(model, len(prompts))
    _patch_batch_modulation_alignment(model, len(prompts))
    img2img_spec = None
    progress_steps = a.steps
    if a.image and (not a.native_edit):
        img2img_spec = {'image_path': a.image, 'mask_path': a.mask or None, 'denoise': _clamp_denoise(a.denoise), 'steps': int(a.steps), 'seed': int(a.seed), 'image_size': a.size}
        progress_steps = _hunyuan_effective_steps(a.steps, _clamp_denoise(a.denoise))
    _wire_pipeline_progress(model, progress_steps, os.path.dirname(os.path.abspath(out_paths[0])), img2img_spec=img2img_spec)
    params = inspect.signature(model.generate_image).parameters
    names = set(params)
    has_var_kw = any((p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))
    kw: dict = {'prompt': prompts if is_batch else prompts[0]}
    if a.native_edit:
        kw['image'] = _native_conditioning_images(a.image, a.reference_image)
        print('[hunyuan] native edit: source and ' + str(len(a.reference_image)) + ' identity references; no latent img2img', flush=True)
    if 'image_size' in names:
        kw['image_size'] = a.size
    if 'seed' in names:
        kw['seed'] = seeds if is_batch else seeds[0]
    step_param = next((k for k in ('diff_infer_steps', 'num_inference_steps', 'steps') if k in names), None)
    if step_param:
        kw[step_param] = a.steps
    elif has_var_kw:
        kw['diff_infer_steps'] = a.steps
    try:
        if hasattr(model, 'generation_config'):
            setattr(model.generation_config, 'diff_infer_steps', int(a.steps))
    except Exception as e:
        print(f'[hunyuan] step override note: could not set generation_config.diff_infer_steps: {e}', flush=True)
    if ar_mode == 'skip':
        skip_param = next((k for k in ('skip_reasoning', 'use_reasoning', 'enable_reasoning', 'bot_task', 'use_meta_prompt', 'think') if k in names), None)
        if skip_param == 'bot_task':
            kw['bot_task'] = 'image'
        elif skip_param in ('use_reasoning', 'enable_reasoning', 'use_meta_prompt', 'think'):
            kw[skip_param] = False
        elif skip_param == 'skip_reasoning':
            kw['skip_reasoning'] = True
        elif has_var_kw:
            kw['bot_task'] = 'image'
            print("[hunyuan] AR skip: passing bot_task='image' via **kwargs (no explicit toggle in signature).", flush=True)
        else:
            print('[hunyuan] AR skip requested but this model build exposes no skip toggle — running the normal with-AR path.', flush=True)
    t1 = time.time()
    _done = threading.Event()

    def _gen_heartbeat():
        n = 0
        while not _done.wait(15):
            n += 15
            print(f'[hunyuan] still diffusing ({a.steps} streamed steps)... {n}s elapsed', flush=True)
    threading.Thread(target=_gen_heartbeat, daemon=True).start()
    print(f'[hunyuan] native batch size={len(prompts)} seeds={seeds[0]}..{seeds[-1]}', flush=True)
    try:
        result = model.generate_image(**kw)
    finally:
        _done.set()
    try:
        if streamer is not None and hasattr(streamer, 'close'):
            streamer.close()
    except Exception as e:
        print(f'[hunyuan] streamer close note: {e}', flush=True)
    try:
        del model
        del built
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass
    try:
        images = _image_candidates(result)
        if len(images) < len(out_paths):
            raise RuntimeError(f'generate_image returned {len(images)} image(s) for {len(out_paths)} requested')
        for img, fp in zip(images, out_paths):
            _save_one_hunyuan_image(img, fp)
            if a.image and a.mask:
                _final_mask_composite(fp, a.image, a.mask)
            if is_batch:
                print(f'[bgen] saved {fp}', flush=True)
    except Exception as e:
        print(f'[hunyuan] ERROR: save failed: {e}', flush=True)
        return 4
    print(f'[hunyuan] generated in {time.time() - t1:.0f}s', flush=True)
    print(f'RESULT DONE {out_paths[0]}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
