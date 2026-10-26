# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
MODEL_ID = 'OpenMOSS-Team/MOSS-Image-SFT'
REVISION = 'daf706d2e1c7a6369c17b6d1658671aee6db1ebe'
GITHUB_COMMIT = '1f808b7c4cfd9b72e7d81bbda9b067bc4d306114'
PIPELINE_SHA256 = '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
DEFAULT_MODEL_DIR = ''
SAMPLERS = ('euler', 'er_sde')
SCHEDULES = ('normal', 'karras', 'exponential', 'beta', 'beta57', 'bong_tangent', 'native_beta', 'flow', 'simple')

def missing_components(root):
    root = Path(root)
    missing = []
    for name in ('transformer', 'text_encoder', 'vae', 'scheduler', 'tokenizer'):
        config = 'scheduler_config.json' if name == 'scheduler' else 'tokenizer_config.json' if name == 'tokenizer' else 'config.json'
        if not (root / name / config).is_file():
            missing.append(f'{name} configuration')
        if name in {'transformer', 'text_encoder', 'vae'}:
            directory = root / name
            indices = list(directory.glob('*.safetensors.index.json'))
            try:
                shards = set(json.loads(indices[0].read_text())['weight_map'].values()) if indices else set()
                complete = all(((directory / f).is_file() for f in shards)) if shards else any(directory.glob('*.safetensors'))
            except (OSError, ValueError, KeyError, TypeError):
                complete = False
            if not complete:
                missing.append(f'{name} weights')
    if not (root / 'pipeline_moss_image.py').is_file():
        missing.append('reviewed pipeline_moss_image.py')
    return missing

def load_reviewed_module(root):
    path = Path(root) / 'pipeline_moss_image.py'
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != PIPELINE_SHA256:
        raise ValueError(f'MOSS pipeline SHA256 mismatch: {actual}; expected {PIPELINE_SHA256}')
    spec = importlib.util.spec_from_file_location('_studio_reviewed_moss_image', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def scheduler_choice(sampler, sigmas):
    sampler = str(sampler or 'euler').strip().lower()
    sigmas = str(sigmas or 'normal').strip().lower()
    if sampler in {'default', 'unipc', 'flow', 'flowmatch'}:
        sampler = 'euler'
    if sigmas == 'default':
        sigmas = 'normal'
    if sampler not in SAMPLERS:
        raise ValueError(f'MOSS-Image supports flow Euler and ER-SDE: {SAMPLERS}')
    if sigmas not in SCHEDULES:
        raise ValueError(f'MOSS-Image sigma schedule must be one of {SCHEDULES}')
    return (sampler, sigmas)

def configure_scheduler(native, sampler, sigmas, steps):
    sampler, sigmas = scheduler_choice(sampler, sigmas)
    normal = sigmas in {'normal', 'flow', 'simple'}
    if normal and sampler == 'euler':
        return native
    from .res4lyf_registry import make_flow_scheduler, custom_flow_sigmas
    scheduler, _ = make_flow_scheduler(dict(native.config), sampler=sampler, sigmas=None if normal else sigmas, shift=1 if normal else 5, log_prefix='[MOSS]')
    original_set = scheduler.set_timesteps

    def set_timesteps(num_inference_steps=steps, device=None):
        if normal:
            native.set_timesteps(num_inference_steps, device=device)
            values = native.sigmas[:-1].tolist()
        else:
            values = custom_flow_sigmas(sigmas, num_inference_steps)
        original_set(num_inference_steps if values is None else None, device=device, sigmas=values)
    scheduler.set_timesteps = set_timesteps
    return scheduler

def release_cached_denoiser(cache):
    blocks = cache.pop('blocks', None)
    if blocks is not None:
        blocks.close()
    transformer = cache.pop('transformer', None)
    if transformer is not None:
        transformer.to('meta')
    _collect()

def validate_request(request):
    w, h = (int(request.get('width', 1024)), int(request.get('height', 1024)))
    if min(w, h) <= 0 or w % 16 or h % 16:
        raise ValueError('MOSS width and height must be positive multiples of 16')
    if not 1 <= int(request.get('steps', 50)) <= 1000:
        raise ValueError('MOSS steps must be between 1 and 1000')
    cfg = float(request.get('guidance', 3))
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError('MOSS guidance must be finite and nonnegative')
    if not isinstance(request.get('prompt'), str) or not request['prompt'].strip():
        raise ValueError('MOSS requires a prompt (natural language, tags or layout JSON)')
    if request.get('modality', 't2i') not in {'t2i', 'i2i'}:
        raise ValueError('MOSS supports text-to-image and SDEdit image-to-image')
    if request.get('modality') == 'i2i':
        if not request.get('image_path') or not Path(request['image_path']).is_file():
            raise ValueError('MOSS image-to-image requires an existing base image')
        strength = float(request.get('denoise', 0.6))
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('MOSS denoise strength must be between 0 and 1')

def prepare_request(params, output):
    from .qwen_image21 import validate_loras
    sampler, sigmas = scheduler_choice(params.sampler, params.sigmas)
    request = dict(prompt=params.prompt, negative=params.negative or '', width=int(params.width or 1024), height=int(params.height or 1024), steps=int(params.steps if params.steps is not None else 50), guidance=float(params.guidance if params.guidance is not None else 3), seed=int(params.seed), shift=5, uncond_skip_layers=[22], loras=validate_loras([{'path': l.path, 'weight': l.weight} for l in params.loras or []]), output_path=str(output), sampler=sampler, sigmas=sigmas, modality=getattr(params, 'modality', 't2i'), image_path=getattr(params, 'image_path', None), denoise=float(getattr(params, 'denoise', None) if getattr(params, 'denoise', None) is not None else 0.6))
    validate_request(request)
    return request

def _collect():
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def _empty(factory, dtype):
    import torch
    from accelerate import init_empty_weights
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with init_empty_weights():
            return factory()
    finally:
        torch.set_default_dtype(previous)

def _load_vae(root):
    import torch
    from diffusers import AutoencoderKLFlux2
    from .qwen_direct_load import bind_component
    config = AutoencoderKLFlux2.load_config(str(root / 'vae'), local_files_only=True)
    vae = _empty(lambda: AutoencoderKLFlux2.from_config(config), torch.float32).eval()
    bind_component(vae, root / 'vae', 'MOSS VAE')
    vae.enable_tiling()
    vae.enable_slicing()
    return vae

def pack_vae_latents(vae, z):
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean = vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    std = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    return ((z - mean) / std).float().contiguous()

def encode_image(pipe, vae, request):
    import torch
    from PIL import Image
    with Image.open(request['image_path']) as source:
        pixels = pipe.image_processor.preprocess(source.convert('RGB'), height=request['height'], width=request['width'])
    with torch.inference_mode():
        z = vae.encode(pixels.to(device='cuda:0', dtype=vae.dtype)).latent_dist.mode()
        return pack_vae_latents(vae, z)

def i2i_start(scheduler, z0, strength, steps, generator):
    from cosmos_windows.weight_streaming.sdedit import flow_start_index, flow_noise_latent
    from diffusers.utils.torch_utils import randn_tensor
    scheduler.set_timesteps(steps, device=z0.device)
    full_count = len(scheduler.timesteps)
    start = flow_start_index(full_count, strength)
    sigma = float(scheduler.sigmas[start])
    noise = randn_tensor(z0.shape, generator=generator, device=z0.device, dtype=z0.dtype)
    latents = flow_noise_latent(z0, noise, sigma)
    original_set = scheduler.set_timesteps

    def set_tail(num_inference_steps=steps, device=None):
        original_set(num_inference_steps, device=device)
        scheduler.timesteps = scheduler.timesteps[start:]
        scheduler.sigmas = scheduler.sigmas[start:]
        scheduler.num_inference_steps = len(scheduler.timesteps)
        scheduler.set_begin_index(0)
    scheduler.set_timesteps = set_tail
    return (latents, dict(start_step=start, start_sigma=sigma, remaining_steps=full_count - start, denoise=strength))

def host_ram_admission(*, imports_ready=None):
    import psutil
    if imports_ready is None:
        imports_ready = all((name in sys.modules for name in ('torch', 'transformers', 'diffusers')))
    return dict(free_at_admission_gib=round(psutil.virtual_memory().available / 2 ** 30, 3), imports_ready=imports_ready, weight_staging_limit_mib=256)

def generate_one(root, request, *, streaming, precision='bf16', cache=None, eager_components=False, pack_dir=None):
    ram_admission = host_ram_admission()
    os.environ['LLAMA_DS_FACTORY_STAGING_MB'] = '256'
    import psutil
    import torch
    from transformers import Qwen3_5Config, Qwen3_5Model, AutoTokenizer
    from diffusers import AutoencoderKLFlux2
    from .qwen_direct_load import bind_component
    from .qwen_block_weights import QwenBlockWeights
    from .qwen_lora import LoraStack
    from .h3_weight_io import cap_cuda
    root = Path(root)
    validate_request(request)
    missing = missing_components(root)
    if missing:
        raise RuntimeError('MOSS local checkpoint incomplete: ' + ', '.join(missing))
    code = load_reviewed_module(root)
    retain = cache is not None and (not streaming)
    if not retain or not cache.get('cuda_capped'):
        cap_cuda()
        if retain:
            cache['cuda_capped'] = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    phases = []

    def phase(name):
        torch.cuda.synchronize()
        phases.append(dict(phase=name, seconds=round(time.perf_counter() - started, 3), allocated_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 3), host_rss_gib=round(psutil.Process().memory_info().rss / 2 ** 30, 3), free_host_gib=round(psutil.virtual_memory().available / 2 ** 30, 3)))
        print('[status] MOSS ' + name, flush=True)
    phase('libraries imported')
    try:
        from cosmos_windows.attention_memeff import enable_memeff_attention
        enable_memeff_attention(allow_math_fallback=True)
    except ImportError:
        pass
    prompt_key = (request['prompt'], request.get('negative', ''), float(request.get('guidance', 3)) != 1)
    prompt_cache = cache.setdefault('prompt_cache', {}) if retain else {}
    encoded = prompt_cache.get(prompt_key)
    if retain and encoded is None:
        release_cached_denoiser(cache)
    encoder = cache.get('encoder') if retain else None
    if encoded is not None:
        encoder, tokenizer = (None, None)
    elif encoder is None:
        phase('loading text encoder with DirectStorage')
        config = Qwen3_5Config.from_pretrained(str(root / 'text_encoder'), local_files_only=True)
        config._attn_implementation = 'sdpa'
        encoder = _empty(lambda: Qwen3_5Model(config), torch.bfloat16)
        bind_component(encoder, root / 'text_encoder', 'MOSS text encoder')
        tokenizer = AutoTokenizer.from_pretrained(str(root / 'tokenizer'), local_files_only=True, trust_remote_code=False)
    else:
        tokenizer = cache['tokenizer']
        phase('reusing resident text encoder')
    scheduler = configure_scheduler(code.MOSSImageScheduler.from_pretrained(str(root / 'scheduler'), local_files_only=True), request.get('sampler'), request.get('sigmas'), request['steps'])
    pipe = code.MOSSImagePipeline(transformer=None, scheduler=scheduler, text_encoder=encoder.eval() if encoder is not None else None, tokenizer=tokenizer, vae=None)
    pipe.set_progress_bar_config(disable=True)
    phase('encoding prompt')
    if encoded is None:
        with torch.inference_mode():
            positive, mask = pipe.encode_prompt(request['prompt'])
            negative, negative_mask = pipe.encode_prompt(request.get('negative', '')) if float(request.get('guidance', 3)) != 1 else (None, None)
        if retain:
            if len(prompt_cache) >= 4:
                prompt_cache.pop(next(iter(prompt_cache)))
            prompt_cache[prompt_key] = (positive, mask, negative, negative_mask)
    else:
        positive, mask, negative, negative_mask = encoded
    keep_encoder = eager_components and (not request.get('loras'))
    if not keep_encoder:
        pipe.register_modules(text_encoder=None, tokenizer=None)
        if retain:
            cache.pop('encoder', None)
            cache.pop('tokenizer', None)
        del encoder, tokenizer
    _collect()
    phase('text encoder retained on GPU' if keep_encoder else 'text encoder unloaded')
    vae = cache.get('vae') if retain else None
    is_i2i = request.get('modality') == 'i2i'
    if (eager_components or is_i2i) and vae is None:
        phase('loading eager fp32 VAE before layer streaming')
        vae = _load_vae(root)
        if retain:
            cache['vae'] = vae
    source_latent = None
    if is_i2i:
        phase('encoding base image')
        source_latent = encode_image(pipe, vae, request)
        phase('base image encoded')
    transformer = cache.get('transformer') if retain else None
    reused = transformer is not None
    if reused:
        blocks = cache['blocks']
        phase('reusing resident denoiser')
    else:
        cfg = code.MOSSImageTransformer.load_config(str(root / 'transformer'), local_files_only=True)
        transformer = _empty(lambda: code.MOSSImageTransformer.from_config(cfg), torch.bfloat16).eval()
        staged = streaming or precision == 'int8'
        bind_component(transformer, root / 'transformer', 'MOSS transformer', skip_blocks=staged, block_prefix='layers')
        transformer.rope.to(device='cuda:0')
        blocks = QwenBlockWeights(transformer, root / 'transformer', streaming=streaming, precision=precision, block_prefix='layers', pack_dir=str(pack_dir or root / 'studio-int8')) if staged else None
        if retain:
            cache.update(transformer=transformer, blocks=blocks)
    pipe.register_modules(transformer=transformer)
    if vae is not None:
        pipe.register_modules(vae=vae)
    stack = None
    binds = evictions = 0
    binds_before, evictions_before = (blocks.binds, blocks.evictions) if blocks is not None else (0, 0)
    lora_reports, lora_calls = ([], 0)
    decode_started = False
    try:
        if request.get('loras'):
            stack = LoraStack(transformer, request['loras'], device='cuda:0', fused=True)
            lora_reports = list(stack.reports)
            if any((row.get('skipped_text_encoder_keys') for row in lora_reports)):
                raise ValueError('MOSS LoRAs must target denoiser Linear modules only; text-encoder adapters are not applied')
        phase('denoising ' + ('streamed' if streaming else 'resident') + ' ' + precision)
        generator = torch.Generator('cpu').manual_seed(request['seed'])
        latents = None
        i2i_metadata = {}
        if source_latent is not None:
            latents, i2i_metadata = i2i_start(scheduler, source_latent, request['denoise'], request['steps'], generator)
            del source_latent
        step_count = 0
        original_step = scheduler.step

        def step(*args, **kwargs):
            nonlocal step_count, stack, binds, evictions, blocks, decode_started, lora_calls, vae
            kwargs.setdefault('generator', generator)
            result = original_step(*args, **kwargs)
            step_count += 1
            total = len(scheduler.timesteps)
            print(f'[genjob] step {step_count}/{total}', flush=True)
            if step_count == total:
                if stack is not None:
                    lora_calls = stack.calls
                    if not lora_calls:
                        raise RuntimeError('MOSS LoRA hooks never ran; refusing to save an unapplied adapter result')
                    stack.close()
                    stack = None
                if blocks is not None:
                    binds, evictions = (blocks.binds - binds_before, blocks.evictions - evictions_before)
                    if not retain and (not eager_components):
                        blocks.close()
                        blocks = None
                if not retain and (not eager_components):
                    transformer.to('meta')
                _collect()
                phase(('denoiser retained; ' if retain or eager_components else 'transformer unloaded; ') + 'fp32 VAE decode')
                if vae is None:
                    vae = _load_vae(root)
                    if retain:
                        cache['vae'] = vae
                pipe.register_modules(vae=vae)
                decode_started = True
            return result
        scheduler.step = step
        with torch.inference_mode():
            image = pipe(prompt_embeds=positive, prompt_attention_mask=mask, negative_prompt_embeds=negative, negative_prompt_attention_mask=negative_mask, height=request['height'], width=request['width'], num_inference_steps=request['steps'], guidance_scale=request['guidance'], latents=latents, uncond_skip_layers=[22], generator=generator).images[0]
        phase('VAE decoded')
        output = Path(request['output_path'])
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        result = dict(path=str(output), seconds=round(time.perf_counter() - started, 3), width=image.width, height=image.height, seed=request['seed'], streaming=streaming, precision=precision, weight_loader='DirectStorage', block_binds=binds, block_evictions=evictions, denoiser_reused=reused, resident_cached=retain, load_mode='resident_cached' if retain else 'streaming_nonlazy' if eager_components else 'streaming_lazy' if streaming else 'resident_lazy', hf_revision=REVISION, github_commit=GITHUB_COMMIT, pipeline_sha256=PIPELINE_SHA256, scheduler=type(scheduler).__name__, sampler=request.get('sampler', 'euler'), sigmas=request.get('sigmas', 'normal'), shift=5, uncond_skip_layers=[22], text_encoder_unloaded=not keep_encoder, vae_decode_offload=False, vae_tiling=True, lora_mode='fused-once GPU deltas applied per arriving layer', phases=phases, loras=lora_reports, lora_hook_calls=lora_calls, host_ram_admission=ram_admission, host_import_rss_gib=phases[0]['host_rss_gib'], modality=request.get('modality', 't2i'), image_path=request.get('image_path'), **i2i_metadata, host_peak_rss_gb=round(getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss) / 2 ** 30, 3), gpu=[dict(index=0, name=torch.cuda.get_device_name(0), peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 3))])
        output.with_suffix(output.suffix + '.metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('RESULT DONE ' + str(output), flush=True)
        return result
    finally:
        if stack is not None:
            stack.close()
        if blocks is not None and (not retain):
            blocks.close()
        del pipe, transformer, positive, negative
        _collect()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--stream-weights', action='store_true')
    parser.add_argument('--precision', choices=['bf16', 'int8'], default='bf16')
    parser.add_argument('--cache-pipeline', action='store_true')
    parser.add_argument('--eager-components', action='store_true')
    parser.add_argument('--pack-dir')
    args = parser.parse_args(argv)
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 16384)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'DIFFUSERS_OFFLINE'):
        os.environ[key] = '1'
    cache = {} if args.cache_pipeline else None
    for line in sys.stdin:
        try:
            result = generate_one(args.model_dir, json.loads(line), streaming=args.stream_weights, precision=args.precision, cache=cache, eager_components=args.eager_components, pack_dir=args.pack_dir)
            print('MOSS_JOB_DONE ' + json.dumps(dict(ok=True, **result)), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print('MOSS_JOB_DONE ' + json.dumps(dict(ok=False, error=f'{type(exc).__name__}: {exc}')), flush=True)
            return 1
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
