# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/ideogram_gen.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import gc
import os
import sys
import time
from pathlib import Path
QUANT_REPOS = {'nf4': 'ideogram-ai/ideogram-4-nf4', 'fp8': 'ideogram-ai/ideogram-4-fp8'}

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

def _pil_to_minus1_1_tensor(image, *, device, dtype):
    import numpy as np
    import torch
    arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t * 2.0 - 1.0).to(device=device, dtype=dtype)

def _encode_ideogram_init_latent(pipe, init_image):
    import torch
    if getattr(pipe, 'autoencoder', None) is None:
        raise RuntimeError('Ideogram VAE is not loaded for img2img encode')
    x = _pil_to_minus1_1_tensor(init_image, device=pipe.device, dtype=pipe.dtype)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=pipe.dtype, enabled=pipe.device.type == 'cuda'):
        ae = pipe.autoencoder.encoder(x)
    ae = ae.float()
    expected_channels = pipe.conditional_transformer.config.in_channels // (pipe.config.patch_size * pipe.config.patch_size)
    if ae.shape[1] == expected_channels * 2:
        ae = ae[:, :expected_channels]
    elif ae.shape[1] != expected_channels:
        raise RuntimeError(f'unexpected Ideogram VAE latent channels: got {ae.shape[1]}, expected {expected_channels}')
    patch = int(pipe.config.patch_size)
    grid_h = ae.shape[-2] // patch
    grid_w = ae.shape[-1] // patch
    if ae.shape[-2] % patch or ae.shape[-1] % patch:
        raise RuntimeError(f'Ideogram VAE latent size {tuple(ae.shape[-2:])} is not divisible by patch {patch}')
    ae = ae.view(ae.shape[0], ae.shape[1], grid_h, patch, grid_w, patch)
    tokens = ae.permute(0, 2, 4, 3, 5, 1).contiguous().view(ae.shape[0], grid_h * grid_w, -1)
    return ((tokens - pipe.latent_shift.float()) / pipe.latent_scale.float()).to(pipe.device)

def _ideogram_mask_tokens(mask_path: str | None, *, width: int, height: int, grid_h: int, grid_w: int, patch: int, device):
    if not mask_path:
        return None
    import numpy as np
    import torch
    from PIL import Image
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f'mask image not found: {mask_path}')
    with Image.open(mask_path) as im:
        mask = _resize_cover(im.convert('L'), width, height)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device)
    t = torch.nn.functional.interpolate(t, size=(grid_h * patch, grid_w * patch), mode='area')
    t = t.view(1, 1, grid_h, patch, grid_w, patch).mean(dim=(3, 5))
    return t.permute(0, 2, 3, 1).contiguous().view(1, grid_h * grid_w, 1).clamp(0.0, 1.0)

def _ideogram_effective_steps(num_steps: int, denoise: float) -> int:
    if denoise <= 0.0:
        return 0
    return max(1, min(int(num_steps), int(__import__('math').ceil(denoise * int(num_steps)))))

def _guidance_schedule_for_steps(preset_name: str, guidance_schedule, num_steps: int, explicit_steps: bool):
    if guidance_schedule is None:
        return None
    try:
        schedule_len = len(guidance_schedule)
    except TypeError:
        return guidance_schedule
    if schedule_len == int(num_steps):
        return guidance_schedule
    if schedule_len <= 1:
        print(f'[ideo] preset {preset_name} guidance schedule has {schedule_len} entries; using constant guidance for requested {num_steps} steps', flush=True)
        return None
    old = [float(v) for v in guidance_schedule]
    new_len = int(num_steps)
    if new_len <= 1:
        return [old[-1]]
    scale = (schedule_len - 1) / (new_len - 1)
    resized: list[float] = []
    for i in range(new_len):
        pos = i * scale
        lo = int(pos)
        hi = min(lo + 1, schedule_len - 1)
        frac = pos - lo
        resized.append(old[lo] * (1.0 - frac) + old[hi] * frac)
    if explicit_steps:
        print(f'[ideo] preset {preset_name} guidance schedule resized {schedule_len}->{new_len} for custom steps', flush=True)
    else:
        print(f'[ideo] preset {preset_name} schedule/step mismatch ({schedule_len} vs {new_len}); resized guidance schedule', flush=True)
    return resized

def _ideogram_img2img(pipe, prompt: str, *, init_image_path: str, mask_path: str | None, denoise: float, height: int, width: int, num_steps: int, guidance_schedule, mu: float, std: float, seed: int | None, raise_on_caption_issues: bool, release_vae_after_encode):
    import math
    import torch
    import ideogram4.pipeline_ideogram4 as _ideo_pipeline
    denoise = _clamp_denoise(denoise)
    init_image = _load_cover_image(init_image_path, width, height)
    if denoise <= 0.0:
        return ([init_image], init_image)
    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = list(prompt)
    if len(prompts) != 1:
        raise ValueError('Ideogram img2img currently supports one prompt/image per run')
    pipe._verify_prompts(prompts, raise_on_issues=raise_on_caption_issues)
    schedule = _ideo_pipeline.get_schedule_for_resolution((height, width), known_mean=mu, std=std)
    step_intervals = _ideo_pipeline.make_step_intervals(num_steps).to(pipe.device)
    if guidance_schedule is not None:
        gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=pipe.device)
        if gw_per_step.shape != (num_steps,):
            raise ValueError(f'guidance_schedule must have shape ({num_steps},), got {tuple(gw_per_step.shape)}')
    else:
        gw_per_step = torch.full((num_steps,), 7.0, dtype=torch.float32, device=pipe.device)
    inputs = pipe._build_inputs(prompts, height=height, width=width)
    batch_size = 1
    num_image_tokens = inputs['num_image_tokens']
    grid_h, grid_w = (inputs['grid_h'], inputs['grid_w'])
    max_text_tokens = inputs['max_text_tokens']
    latent_dim = pipe.conditional_transformer.config.in_channels
    init_z = _encode_ideogram_init_latent(pipe, init_image).float()
    if release_vae_after_encode:
        pipe.autoencoder = None
        gc.collect()
        torch.cuda.empty_cache()
        print('[ideo] lazy VAE enabled: autoencoder released after img2img encode', flush=True)
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator.manual_seed(int(seed))
    noise = torch.randn(init_z.shape, dtype=torch.float32, device=pipe.device, generator=generator)
    active_steps = _ideogram_effective_steps(num_steps, denoise)
    start_i = active_steps - 1
    t_start = float(schedule(step_intervals[start_i + 1].unsqueeze(0)).item())
    z = (1.0 - t_start) * init_z + t_start * noise
    mask_tokens = _ideogram_mask_tokens(mask_path, width=width, height=height, grid_h=grid_h, grid_w=grid_w, patch=int(pipe.config.patch_size), device=pipe.device)
    if mask_tokens is not None:
        z = mask_tokens * z + (1.0 - mask_tokens) * ((1.0 - t_start) * init_z + t_start * noise)
    llm_features = pipe._encode_text(inputs['token_ids'], inputs['text_position_ids'], inputs['indicator'])
    neg_position_ids = inputs['position_ids'][:, max_text_tokens:]
    neg_segment_ids = inputs['segment_ids'][:, max_text_tokens:]
    neg_indicator = inputs['indicator'][:, max_text_tokens:]
    neg_llm_features = torch.zeros(batch_size, num_image_tokens, llm_features.shape[-1], dtype=llm_features.dtype, device=pipe.device)
    text_z_padding = torch.zeros(batch_size, max_text_tokens, latent_dim, dtype=torch.float32, device=pipe.device)
    print(f'[ideo] img2img denoise={denoise:.2f}; running {active_steps}/{num_steps} steps', flush=True)
    for i in range(start_i, -1, -1):
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
        t = torch.full((batch_size,), t_val, dtype=torch.float32, device=pipe.device)
        pos_z = torch.cat([text_z_padding, z], dim=1)
        pos_out = pipe.conditional_transformer(llm_features=llm_features, x=pos_z, t=t, position_ids=inputs['position_ids'], segment_ids=inputs['segment_ids'], indicator=inputs['indicator'])
        pos_v = pos_out[:, max_text_tokens:]
        neg_v = pipe.unconditional_transformer(llm_features=neg_llm_features, x=z, t=t, position_ids=neg_position_ids, segment_ids=neg_segment_ids, indicator=neg_indicator)
        gw_i = gw_per_step[i]
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        z = z + v * (s_val - t_val)
        if mask_tokens is not None:
            orig_at_s = (1.0 - s_val) * init_z + s_val * noise
            z = mask_tokens * z + (1.0 - mask_tokens) * orig_at_s
    return (pipe._decode(z, grid_h=grid_h, grid_w=grid_w), init_image)

def _final_mask_composite(generated, init_image, mask_path: str | None):
    if not mask_path:
        return generated
    from PIL import Image
    if not os.path.isfile(mask_path):
        return generated
    with Image.open(mask_path) as im:
        mask = _resize_cover(im.convert('L'), generated.width, generated.height)
    base = _resize_cover(init_image, generated.width, generated.height)
    return Image.composite(generated.convert('RGB'), base.convert('RGB'), mask)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=0, help="0 = use the sampler preset's num_steps")
    ap.add_argument('--preset', default='V4_QUALITY_48', help='V4_QUALITY_48 | V4_DEFAULT_20 | V4_TURBO_12')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quant', default='nf4', choices=['nf4', 'fp8'])
    ap.add_argument('--weights-repo', default='', help='override the HF repo (defaults by --quant)')
    ap.add_argument('--conditional-index-filename', default='', help='relative conditional transformer index/safetensors path inside weights repo')
    ap.add_argument('--unconditional-index-filename', default='', help='relative unconditional transformer index/safetensors path inside weights repo')
    ap.add_argument('--autoencoder-filename', default='', help='relative VAE/autoencoder safetensors path inside weights repo')
    ap.add_argument('--text-encoder-subfolder', default='', help='relative Qwen3-VL text encoder subfolder inside weights repo')
    ap.add_argument('--tokenizer-subfolder', default='', help='relative tokenizer subfolder inside weights repo')
    ap.add_argument('--image', default='', help='optional init image for latent image-to-image/edit generation')
    ap.add_argument('--denoise', type=float, default=1.0, help='img2img strength: 0 keeps the source image, 1 is full regeneration')
    ap.add_argument('--mask', default='', help='optional mask for img2img/edit; white=regenerate, black=preserve')
    ap.add_argument('--gpu', default='1')
    a = ap.parse_args()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '20')
    os.environ.setdefault('HF_HOME', '')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from cosmos_windows import mem_guard
        mem_guard.preflight(min_free_gb=6.0, label='Ideogram4')
        mem_guard.start_watchdog(max_process_gb=26.0, min_free_gb=3.0, label='Ideogram4')
    except Exception as e:
        print(f'[ideo] mem_guard note: {e}', flush=True)
    try:
        import subprocess as _sp
        q = _sp.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(a.gpu)], capture_output=True, text=True, timeout=15)
        free_mb = int((q.stdout or '0').strip().splitlines()[0])
        need_mb = 17000 if a.quant == 'nf4' else 28000
        print(f'[ideo] VRAM free on GPU {a.gpu}: {free_mb} MB (Ideogram4 {a.quant} needs ~{need_mb} MB)', flush=True)
        if free_mb < need_mb and (not os.environ.get('IDEO_ALLOW_LOW_VRAM')):
            print(f'[ideo] ABORT: only {free_mb} MB free on GPU {a.gpu} — need ~{need_mb} MB. Switch models / free the card and retry. (override: IDEO_ALLOW_LOW_VRAM=1)', flush=True)
            return 5
    except Exception as e:
        print(f'[ideo] VRAM check skipped: {e}', flush=True)
    print('[ideo] importing torch and Ideogram pipeline...', flush=True)
    import torch
    try:
        from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, PRESETS
        import ideogram4.pipeline_ideogram4 as _ideo_pipeline
    except Exception as e:
        print(f'[ideo] ERROR: the ideogram4 package is not importable ({e}', flush=True)
        return 3
    repo = a.weights_repo or QUANT_REPOS[a.quant]
    local_repo = Path(repo)
    if local_repo.exists():
        try:
            _hub_download = _ideo_pipeline.hf_hub_download

            def _local_aware_hf_hub_download(*args, **kwargs):
                repo_id = kwargs.pop('repo_id', args[0] if args else None)
                filename = kwargs.pop('filename', args[1] if len(args) > 1 else None)
                root = Path(str(repo_id))
                if root.exists() and filename:
                    rel = str(filename).replace('\\', '/')
                    candidate = root.joinpath(*[part for part in rel.split('/') if part])
                    if candidate.is_file():
                        return str(candidate)
                    raise _ideo_pipeline.EntryNotFoundError(f'local Ideogram file not found: {candidate}')
                if args:
                    return _hub_download(*args, **kwargs)
                return _hub_download(repo_id=repo_id, filename=filename, **kwargs)
            _ideo_pipeline.hf_hub_download = _local_aware_hf_hub_download
            print(f'[ideo] using local Ideogram weights directory: {local_repo}', flush=True)
        except Exception as e:
            print(f'[ideo] ERROR: could not enable local Ideogram weights from {local_repo}: {e}', flush=True)
            return 3
    if a.preset not in PRESETS:
        print(f'[ideo] ERROR: unknown preset {a.preset!r}; choices: {sorted(PRESETS)}', flush=True)
        return 2
    preset = PRESETS[a.preset]
    explicit_steps = bool(a.steps)
    num_steps = a.steps or preset.num_steps
    guidance_schedule = _guidance_schedule_for_steps(a.preset, preset.guidance_schedule, num_steps, explicit_steps)
    cfg_kwargs = {'weights_repo': repo}
    for cli_name, cfg_name in (('conditional_index_filename', 'conditional_index_filename'), ('unconditional_index_filename', 'unconditional_index_filename'), ('autoencoder_filename', 'autoencoder_filename'), ('text_encoder_subfolder', 'text_encoder_subfolder'), ('tokenizer_subfolder', 'tokenizer_subfolder')):
        value = getattr(a, cli_name, '') or ''
        if value:
            cfg_kwargs[cfg_name] = value.replace('\\', '/')
    try:
        from cosmos_windows.attention_memeff import enable_memeff_attention
        enable_memeff_attention(allow_math_fallback=True)
        print('[ideo] cuDNN-preferring SDPA enabled (SM120 O(N) attention)', flush=True)
    except Exception as e:
        print(f'[ideo] memeff-attention enable skipped: {e}', flush=True)
    print(f'[ideo] loading Ideogram4 {a.quant} from {repo} (first run downloads ~16 GB, then cached)...', flush=True)
    if len(cfg_kwargs) > 1:
        shown = {k: v for k, v in cfg_kwargs.items() if k != 'weights_repo'}
        print(f'[ideo] component overrides: {shown}', flush=True)
    t0 = time.time()
    pipe = Ideogram4Pipeline.from_pretrained(config=Ideogram4PipelineConfig(**cfg_kwargs), device='cuda', dtype=torch.bfloat16)
    print(f'[ideo] model ready in {time.time() - t0:.0f}s — generating {a.width}x{a.height}, {num_steps} steps, preset {a.preset}...', flush=True)
    original_decode = pipe._decode
    autoencoder_weights = _ideo_pipeline.hf_hub_download(repo_id=repo, filename=cfg_kwargs.get('autoencoder_filename', 'vae/diffusion_pytorch_model.safetensors'))
    lazy_vae = os.environ.get('IDEO_LAZY_VAE', '1') != '0'
    if lazy_vae and (not a.image):
        try:
            pipe.autoencoder = None
            gc.collect()
            torch.cuda.empty_cache()
            print('[ideo] lazy VAE enabled: autoencoder released until decode', flush=True)
        except Exception as e:
            print(f'[ideo] lazy VAE note: {e}', flush=True)

    def _decode_with_headroom(z, *, grid_h: int, grid_w: int):
        print('[ideo] denoise complete; freeing text/DiT modules before VAE decode...', flush=True)
        for attr in ('text_encoder', 'text_tokenizer', 'conditional_transformer', 'unconditional_transformer'):
            try:
                setattr(pipe, attr, None)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if getattr(pipe, 'autoencoder', None) is None:
            print('[ideo] lazy-loading VAE for decode...', flush=True)
            pipe.autoencoder = _ideo_pipeline._load_autoencoder(autoencoder_weights, pipe.device, pipe.dtype)
            gc.collect()
            torch.cuda.empty_cache()
        print('[ideo] VAE decode start...', flush=True)
        return original_decode(z, grid_h=grid_h, grid_w=grid_w)
    pipe._decode = _decode_with_headroom
    counter = {'i': 0}
    progress_total = {'n': num_steps}

    def _pre(_mod, _args):
        counter['i'] += 1
        print(f"[ideo] step {counter['i']}/{progress_total['n']} denoising...", flush=True)
    handle = pipe.conditional_transformer.register_forward_pre_hook(_pre)
    init_image = None
    try:
        if a.image:
            effective_steps = _ideogram_effective_steps(num_steps, _clamp_denoise(a.denoise))
            progress_total['n'] = effective_steps or 1
            counter['i'] = 0
            images, init_image = _ideogram_img2img(pipe, a.prompt, init_image_path=a.image, mask_path=a.mask or None, denoise=a.denoise, height=a.height, width=a.width, num_steps=num_steps, guidance_schedule=guidance_schedule, mu=preset.mu, std=preset.std, seed=a.seed, raise_on_caption_issues=False, release_vae_after_encode=lazy_vae)
        else:
            images = pipe(a.prompt, height=a.height, width=a.width, num_steps=num_steps, guidance_schedule=guidance_schedule, mu=preset.mu, std=preset.std, seed=a.seed, raise_on_caption_issues=False)
    finally:
        handle.remove()
    out = a.out
    base, ext = os.path.splitext(out)
    n = 1
    while os.path.exists(out):
        out = f'{base}_{n}{ext}'
        n += 1
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    image = images[0]
    if init_image is not None:
        image = _final_mask_composite(image, init_image, a.mask or None)
    image.save(out)
    print(f'[ideo] saved in {time.time() - t0:.0f}s -> {out}', flush=True)
    print(f'RESULT DONE {out}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
