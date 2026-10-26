# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
MODEL_ID = 'Qwen/Qwen-Image-2.1'
DEFAULT_MODEL_DIR = ''

def transparent_prompt(prompt: str) -> str:
    lead = 'This is an RGBA image with transparency.'
    tail = 'The image has an alpha channel and the background is transparent.'
    text = str(prompt or '').strip()
    return f'{lead} {text} {tail}'.strip()

def region_prompt(prompt: str, regions: list[dict[str, Any]], annotation_picture: int, mask_pictures: list[int]) -> str:
    lines = [str(prompt or '').strip(), f'Picture 1 is the source image. Picture {annotation_picture} is the painted annotation guide.', 'Apply every listed local edit together in one edit pass. Preserve everything outside the marked areas unless an instruction explicitly says otherwise.', 'The annotation and mask pictures are instructions only. Return a clean finished image: never copy guide colours, strokes, circles, mask pixels, tint overlays, or annotation transparency into the result.']
    mask_index = 0
    for region in regions:
        shape = region.get('shape') or {}
        desc = f"region {region.get('id')} ({region.get('color')}, {shape.get('type')})"
        if shape.get('type') == 'mask':
            desc += f', with its separate mask in Picture {mask_pictures[mask_index]}'
            mask_index += 1
        lines.append(f"- {desc}: {region.get('instruction')}")
    return '\n'.join((line for line in lines if line))

def condition_count(request: dict[str, Any]) -> int:
    count = int(bool(request.get('image_path'))) + len(request.get('reference_images') or [])
    regions = request.get('regions') or []
    if regions:
        count += 1
        count += sum((1 for region in regions if (region.get('shape') or {}).get('type') == 'mask'))
    return count

def condition_memory_budget(request: dict[str, Any], model_dir: str | Path=DEFAULT_MODEL_DIR) -> dict[str, Any]:
    from PIL import Image
    from .qwen_comfy_load import components_root
    validate_request(request)
    config_path = components_root(model_dir) / 'transformer' / 'config.json'
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    resolution = max(int(request.get('width', 1024)), int(request.get('height', 1024)))
    paths = [str(request['image_path'])] if request.get('image_path') else []
    paths += [str(p) for p in request.get('reference_images') or []]
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append((path, image.size))
    if request.get('regions'):
        if not sizes:
            raise ValueError('Qwen region instructions require a primary canvas')
        sizes += [('annotation', sizes[0][1])]
        sizes += [('mask', sizes[0][1]) for region in request['regions'] if (region.get('shape') or {}).get('type') == 'mask']
    rows = []
    for path, (width, height) in sizes:
        ratio = width / height
        resized_width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        resized_height = round(math.sqrt(resolution * resolution * ratio) / ratio / 32) * 32
        if min(resized_width, resized_height) <= 0:
            raise ValueError('Qwen condition aspect ratio produces an empty conditioning canvas')
        tokens = resized_width // 16 * (resized_height // 16)
        rows.append({'path': path, 'source_canvas': [width, height], 'conditioning_canvas': [resized_width, resized_height], 'latent_tokens': tokens, 'vlm_image_slots': tokens // 4})
    branches = 2 if float(request.get('true_cfg_scale', 1.0)) > 1 and request.get('negative') else 1
    per_token = 2 * int(config['num_layers']) * int(config['num_attention_heads']) * int(config['attention_head_dim']) * 2
    tokens = sum((row['latent_tokens'] for row in rows))
    return {'conditions': rows, 'condition_count': len(rows), 'condition_latent_tokens': tokens, 'bf16_kv_bytes_per_token': per_token, 'cfg_branches': branches, 'cached_condition_kv_lower_bound_bytes': tokens * per_token * branches, 'model_config': str(config_path), 'model_config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'scope': 'Condition KV lower bound; excludes text, weights, activations and VAE.'}

def validate_request(request: dict[str, Any], *, require_files: bool=True) -> None:
    count = condition_count(request)
    if count > 10:
        raise ValueError(f'Qwen-Image-2.1 accepts at most 10 condition images; request resolves to {count}')
    paths = []
    if request.get('image_path'):
        paths.append(str(request['image_path']))
    paths.extend((str(path) for path in request.get('reference_images') or []))
    for region in request.get('regions') or []:
        shape = region.get('shape') or {}
        if shape.get('type') == 'mask':
            paths.append(str(shape.get('mask_path') or ''))
    if require_files:
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError('condition image(s) not found: ' + ', '.join(missing))
QWEN_SAMPLERS = ('euler', 'er_sde')
QWEN_SIGMAS = ('exponential', 'beta', 'beta57', 'bong_tangent', 'native_beta')
_DEFAULT_SAMPLERS = {'', 'default', 'euler', 'flow', 'unipc'}
_DEFAULT_SIGMAS = {'', 'default', 'flow', 'simple', 'normal', 'karras'}

def scheduler_choice(sampler: str | None, sigmas: str | None) -> tuple[str | None, str | None]:
    s = str(sampler or '').strip().lower()
    g = str(sigmas or '').strip().lower()
    if s in _DEFAULT_SAMPLERS:
        s_out = None
    elif s in QWEN_SAMPLERS:
        s_out = s
    else:
        raise ValueError(f"Qwen-Image-2.1 is a flow-matching model; sampler {sampler!r} is not implemented for it. Use one of: default (Euler), {', '.join(QWEN_SAMPLERS)}")
    if g in _DEFAULT_SIGMAS:
        g_out = None
    elif g in QWEN_SIGMAS:
        g_out = g
    else:
        raise ValueError(f"Qwen-Image-2.1: scheduler/sigmas {sigmas!r} is not implemented for it. Use one of: default, {', '.join(QWEN_SIGMAS)}")
    return (s_out, g_out)

def validate_loras(loras: list[dict[str, Any]] | None, *, require_files: bool=True) -> list[dict[str, Any]]:
    out = []
    for spec in loras or []:
        path = str((spec or {}).get('path') or '').strip()
        if not path:
            continue
        if not path.lower().endswith('.safetensors'):
            raise ValueError(f'Qwen-Image-2.1 loads .safetensors LoRAs only: {path}')
        if require_files and (not Path(path).is_file()):
            raise FileNotFoundError(f'LoRA not found: {path}')
        out.append({'path': path, 'weight': float(spec.get('weight', 1.0))})
    return out

def _rgb(color: str) -> tuple[int, int, int]:
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(color)[:3]
    except ValueError as exc:
        raise ValueError(f'invalid region color {color!r}; use a CSS color name or #RRGGBB') from exc

def prepare_conditions(request: dict[str, Any], output_path: str | Path) -> tuple[list[Any], str, list[str]]:
    from PIL import Image, ImageDraw
    validate_request(request)
    conditions: list[Any] = []
    artifacts: list[str] = []
    source = None
    if request.get('image_path'):
        source = Image.open(request['image_path']).convert('RGBA')
        declared_w = request.get('source_width')
        declared_h = request.get('source_height')
        if declared_w and declared_h and (source.size != (int(declared_w), int(declared_h))):
            raise ValueError(f'source dimensions are {source.width}x{source.height}, but request declared {declared_w}x{declared_h}')
        conditions.append(source)
    for path in request.get('reference_images') or []:
        conditions.append(Image.open(path).convert('RGBA'))
    regions = request.get('regions') or []
    mask_picture_numbers: list[int] = []
    if regions:
        if source is None:
            raise ValueError('region editing requires image_path')
        declared_w = float(request.get('source_width') or source.width)
        declared_h = float(request.get('source_height') or source.height)
        sx, sy = (source.width / declared_w, source.height / declared_h)
        overlay = Image.new('RGBA', source.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        normalized_masks = []
        for region in regions:
            color = _rgb(str(region['color']))
            shape = region['shape']
            if shape['type'] == 'circle':
                cx, cy = (float(shape['cx']) * sx, float(shape['cy']) * sy)
                rx, ry = (float(shape['radius']) * sx, float(shape['radius']) * sy)
                line = max(3, round(min(source.size) / 180))
                draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(*color, 255), width=line)
            elif shape['type'] == 'stroke':
                points = [(float(p['x']) * sx, float(p['y']) * sy) for p in shape['points']]
                width = max(1, round(float(shape['width']) * (sx + sy) / 2))
                draw.line(points, fill=(*color, 230), width=width, joint='curve')
            else:
                mask = Image.open(shape['mask_path']).convert('L')
                if mask.size != source.size:
                    mask = mask.resize(source.size, Image.Resampling.NEAREST)
                tint = Image.new('RGBA', source.size, (*color, 0))
                tint.putalpha(mask.point(lambda value: round(value * 0.55)))
                overlay.alpha_composite(tint)
                normalized_masks.append(Image.merge('RGB', (mask, mask, mask)))
        annotated = Image.alpha_composite(source, overlay)
        annotation_path = Path(output_path).with_suffix('.annotations.png')
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(annotation_path)
        artifacts.append(str(annotation_path))
        conditions.append(annotated)
        annotation_picture = len(conditions)
        for index, mask in enumerate(normalized_masks, start=1):
            mask_path = Path(output_path).with_suffix(f'.region-mask-{index}.png')
            mask.save(mask_path)
            artifacts.append(str(mask_path))
            conditions.append(mask)
            mask_picture_numbers.append(len(conditions))
        prompt = region_prompt(request.get('prompt', ''), regions, annotation_picture, mask_picture_numbers)
    else:
        prompt = str(request.get('prompt') or '')
    if request.get('output_rgba'):
        prompt = transparent_prompt(prompt)
    if len(conditions) > 10:
        raise ValueError(f'Qwen-Image-2.1 accepts at most 10 condition images; prepared {len(conditions)}')
    return (conditions, prompt, artifacts)
ENCODER_STAGED_POLICY = 'request_encoder_then_transformer_blocks_v1'
ENCODER_RESIDENT_POLICY = 'resident_encoder_v1'

def _encoder_staging(request):
    value = request.get('request_encoder_staging', False)
    if type(value) is not bool:
        raise ValueError('request_encoder_staging must be a boolean')
    return value

def _load_text_encoder(root: Path, use_int8: bool):
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from .qwen_direct_load import bind_component
    cfg = Qwen3VLConfig.from_pretrained(str(root / 'text_encoder'), local_files_only=True)
    from .qwen_attention import flash_attention_function
    cfg._attn_implementation = 'flash_attention_2' if flash_attention_function() else 'sdpa'
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights():
            text_encoder = Qwen3VLForConditionalGeneration(cfg)
    finally:
        torch.set_default_dtype(previous_dtype)
    if use_int8:
        from transformers import BitsAndBytesConfig
        from transformers.integrations.bitsandbytes import replace_with_bnb_linear
        text_encoder = replace_with_bnb_linear(text_encoder, modules_to_not_convert=['lm_head'], quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    bind_component(text_encoder, root / 'text_encoder', 'text encoder', quantize_int8=use_int8)
    text_encoder.eval()
    from .qwen_attention import configure_text_encoder
    configure_text_encoder(text_encoder)
    return text_encoder

def _cuda_memory():
    import torch
    return {'allocated_bytes': torch.cuda.memory_allocated(0), 'reserved_bytes': torch.cuda.memory_reserved(0)}

def _collect_cuda():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def _tensor_statistics(tensor, stage):
    import torch
    values = tensor.detach().to(dtype=torch.float32)
    finite = torch.isfinite(values)
    nonfinite = int((~finite).sum().item())
    stats = {'stage': stage, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'reduction_dtype': 'torch.float32', 'std_definition': 'population (correction=0)', 'finite': nonfinite == 0, 'nonfinite_count': nonfinite, 'min': None, 'max': None, 'mean': None, 'std': None}
    if values.numel() and nonfinite == 0:
        reductions = {'min': values.min(), 'max': values.max(), 'mean': values.mean(), 'std': values.std(correction=0)}
        for key, value in reductions.items():
            number = float(value.item())
            stats[key] = number if math.isfinite(number) else None
    return stats

class _RequestOutputDiagnostics:

    def __init__(self, pipe):
        self.pipe = pipe
        self.stats = {'final_scheduler_latents': None, 'vae_decode_output': None}
        self.decode_method = None
        self.had_instance_method = False
        self.instance_method = None

    def _record(self, key, tensor):
        stats = _tensor_statistics(tensor, key)
        self.stats[key] = stats
        print('[status] Qwen tensor diagnostics: ' + json.dumps(stats, sort_keys=True, allow_nan=False), flush=True)

    def final_latents(self, step, latents):
        if step + 1 == self.pipe.num_timesteps and self.stats['final_scheduler_latents'] is None:
            self._record('final_scheduler_latents', latents)

    def begin(self):
        vae = self.pipe.vae
        self.had_instance_method = 'decode' in vae.__dict__
        self.instance_method = vae.__dict__.get('decode')
        self.decode_method = vae.decode
        vae.decode = self._decode

    def _decode(self, *args, **kwargs):
        result = self.decode_method(*args, **kwargs)
        sample = result[0] if isinstance(result, (tuple, list)) else result.sample
        self._record('vae_decode_output', sample)
        return result

    def finish(self):
        if self.decode_method is None:
            return
        if self.had_instance_method:
            self.pipe.vae.decode = self.instance_method
        else:
            del self.pipe.vae.decode
        self.decode_method = self.instance_method = None

class _RequestEncoderLifetime:

    def __init__(self, pipe, expected_encodes):
        self.pipe, self.handle = (pipe, None)
        self.events = []
        self.released = False
        self.expected_encodes, self.encodes = (expected_encodes, 0)
        self.encode_method = None
        self.encode_instance_value = None
        self.had_encode_instance_value = False

    def _event(self, action, before):
        event = {'action': action, 'before': before, 'after': _cuda_memory()}
        self.events.append(event)
        print('[status] Qwen model lifetime: ' + json.dumps(event, sort_keys=True), flush=True)

    def release_encoder(self):
        if self.pipe.text_encoder is None:
            return
        import torch
        torch.cuda.synchronize(0)
        before = _cuda_memory()
        self.pipe.register_modules(text_encoder=None)
        _collect_cuda()
        self.released = True
        self._event('encoder_released', before)

    def begin(self):
        blocks = getattr(self.pipe.transformer, '_studio_block_weights', None)
        if blocks is not None and (not blocks._streaming):
            before = _cuda_memory()
            blocks.evict_resident()
            _collect_cuda()
            self._event('resident_blocks_evicted', before)
        if self.pipe.text_encoder is None:
            before = _cuda_memory()
            self.pipe.register_modules(text_encoder=_load_text_encoder(self.pipe._studio_model_root, self.pipe._studio_text_encoder_8bit))
            self._event('encoder_reloaded', before)
        self.had_encode_instance_value = 'encode_prompt' in self.pipe.__dict__
        self.encode_instance_value = self.pipe.__dict__.get('encode_prompt')
        self.encode_method = self.pipe.encode_prompt
        self.pipe.encode_prompt = self._encode_prompt
        self.handle = self.pipe.transformer.register_forward_pre_hook(self._before_transformer, prepend=True)

    def _encode_prompt(self, *args, **kwargs):
        if self.encodes >= self.expected_encodes:
            raise RuntimeError('Qwen requested unexpected additional prompt encoding')
        encoded = self.encode_method(*args, **kwargs)
        self.encodes += 1
        if self.encodes == self.expected_encodes:
            self.release_encoder()
        return encoded

    def _before_transformer(self, module, args):
        self.handle.remove()
        self.handle = None
        if self.encodes != self.expected_encodes or not self.released:
            raise RuntimeError('Qwen transformer began before all prompt embeddings were materialized')
        self.release_encoder()
        blocks = getattr(module, '_studio_block_weights', None)
        if blocks is not None and (not blocks._streaming):
            before = _cuda_memory()
            blocks.materialize_resident()
            self._event('transformer_blocks_materialized', before)

    def finish(self, failed):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.encode_method is not None:
            if self.had_encode_instance_value:
                self.pipe.encode_prompt = self.encode_instance_value
            else:
                del self.pipe.encode_prompt
            self.encode_method = self.encode_instance_value = None
        self.release_encoder()
        if failed:
            blocks = getattr(self.pipe.transformer, '_studio_block_weights', None)
            if blocks is not None and (not blocks._streaming):
                before = _cuda_memory()
                blocks.evict_resident()
                _collect_cuda()
                self._event('failed_request_blocks_evicted', before)

def _load_pipeline(model_dir: str, offload_dir: str, *, streaming=False, precision='bf16', request_encoder_staging=False):
    import torch
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor
    from .qwen_direct_load import bind_component
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('STUDIO_QWEN_IMAGE21_VRAM_FRACTION', '0.92')), 0)
    from .qwen_comfy_load import bind_qwen21_transformer, components_root, is_single_file_checkpoint
    root = Path(model_dir)
    single_file = is_single_file_checkpoint(root)
    comp_root = components_root(root)
    started = time.perf_counter()
    use_int8 = os.environ.get('STUDIO_QWEN_IMAGE21_TEXT_ENCODER_8BIT', '1') != '0'
    print(f"[status] Qwen mode: transformer={precision}, streaming={streaming}, text_encoder={('int8' if use_int8 else 'bf16')}", flush=True)
    if single_file:
        print(f'[status] Qwen ComfyUI single-file checkpoint {root.name!r}: transformer from the file, other components from {comp_root}', flush=True)
    print('[status] DirectStorage: loading Qwen text encoder to VRAM', flush=True)
    text_encoder = _load_text_encoder(comp_root, use_int8)
    print(f'[status] Text encoder ready after {time.perf_counter() - started:.1f}s', flush=True)
    components = {}
    for name, cls in (('transformer', QwenImage21Transformer2DModel), ('vae', AutoencoderKLQwenImage21)):
        print(f'[status] DirectStorage: loading Qwen {name} to VRAM', flush=True)
        config = cls.load_config(str(comp_root / name), local_files_only=True)
        with init_empty_weights():
            module = cls.from_config(config).to(dtype=torch.bfloat16)
        if single_file and name == 'transformer':
            bind_qwen21_transformer(module, root, device='cuda:0', dtype=torch.bfloat16, streaming=streaming, defer_resident=request_encoder_staging)
        else:
            staged = name == 'transformer' and (streaming or precision == 'int8' or request_encoder_staging)
            bind_component(module, comp_root / name, name, skip_blocks=staged)
            if staged:
                from .qwen_block_weights import QwenBlockWeights
                print('[status] Qwen phase: initializing transformer block streamer', flush=True)
                module._studio_block_weights = QwenBlockWeights(module, comp_root / name, streaming=streaming, precision=precision, defer_resident=request_encoder_staging)
                print('[status] Qwen phase: transformer block streamer ready', flush=True)
        components[name] = module.eval()
    pipe = QwenImage21Pipeline(text_encoder=text_encoder, processor=Qwen3VLProcessor.from_pretrained(str(comp_root / 'processor'), local_files_only=True), scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(comp_root / 'scheduler'), local_files_only=True), **components)
    pipe.set_progress_bar_config(disable=True)
    from .qwen_attention import configure_attention
    configure_attention(pipe.transformer)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe._studio_base_scheduler_config = dict(pipe.scheduler.config)
    pipe._studio_streaming = streaming
    pipe._studio_precision = precision
    pipe._studio_text_encoder_8bit = use_int8
    pipe._studio_model_root = comp_root
    pipe._studio_request_encoder_staging = request_encoder_staging
    pipe._studio_load_seconds = round(time.perf_counter() - started, 3)
    print(f'[status] Qwen weights ready in {pipe._studio_load_seconds:.1f}s; encoding prompt', flush=True)
    return pipe

def _apply_loras(pipe, specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    from .qwen_lora import LoraStack, stack_key
    key = stack_key(specs)
    current = getattr(pipe, '_studio_lora', None)
    if current is not None and getattr(pipe, '_studio_lora_key', None) == key:
        return current
    if current is not None:
        current.close()
        pipe._studio_lora = pipe._studio_lora_key = None
    if not specs:
        return None
    started = time.perf_counter()
    stack = LoraStack(pipe.transformer, specs, device='cuda:0')
    pipe._studio_lora, pipe._studio_lora_key = (stack, key)
    print(f'[status] Qwen LoRA: {len(specs)} adapter(s) on {len(stack.modules)} modules in {time.perf_counter() - started:.1f}s (unfused side path; base weights untouched)', flush=True)
    return stack

def _apply_scheduler(pipe, request: dict[str, Any]):
    from diffusers import FlowMatchEulerDiscreteScheduler
    sampler, sigmas = (request.get('sampler'), request.get('sigmas'))
    base = getattr(pipe, '_studio_base_scheduler_config', None) or dict(pipe.scheduler.config)
    if not sampler and (not sigmas):
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base)
        return None
    from .res4lyf_registry import make_flow_scheduler, custom_flow_sigmas
    pipe.scheduler, _ = make_flow_scheduler(base, sampler=sampler, sigmas=sigmas, log_prefix='[qwen_image21]')
    steps = int(request.get('steps', 40))
    custom = custom_flow_sigmas(sigmas, steps)
    print(f"[status] Qwen scheduler: {type(pipe.scheduler).__name__} sampler={sampler or 'euler'} sigmas={sigmas or 'flow'}", flush=True)
    return custom

def generate_one(pipe, request: dict[str, Any]) -> dict[str, Any]:
    import torch
    encoder_staging = _encoder_staging(request)
    if encoder_staging != getattr(pipe, '_studio_request_encoder_staging', False):
        raise RuntimeError('Qwen encoder lifetime policy changed; use a separate worker')
    output = Path(request['output_path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    print('[status] Qwen phase: preparing condition images', flush=True)
    conditions, prompt, artifacts = prepare_conditions(request, output)
    print(f"[status] Qwen phase: conditions ready, count={len(conditions)}, use_kv_cache={bool(request.get('use_kv_cache', True))}", flush=True)
    loras = validate_loras(request.get('loras'))
    from .two_pass import runtime_settings
    option = runtime_settings(request.get('two_pass'))
    extra = validate_loras(option.loras) if option and option.enabled else []
    stack = _apply_loras(pipe, loras + extra)
    calls_before = stack.calls if stack is not None else 0
    custom_sigmas = _apply_scheduler(pipe, request)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    diagnostics = _RequestOutputDiagnostics(pipe)

    def on_step(_pipe, step, _timestep, callback_kwargs):
        latents = callback_kwargs.get('latents')
        if latents is not None and (not torch.isfinite(latents).all().item()):
            raise RuntimeError(f'Qwen produced non-finite latents at step {step + 1}; no image was saved')
        if latents is not None:
            diagnostics.final_latents(step, latents)
        print(f"[qwen_image21] step {step + 1}/{request.get('steps', 40)}", flush=True)
        return callback_kwargs
    generator_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    alignment = pipe.vae_scale_factor * 2
    width, height = (int(request.get('width', 1024)), int(request.get('height', 1024)))
    if width % alignment or height % alignment:
        print(f'[status] Qwen aligns output dimensions to {alignment}-pixel multiples: {width}x{height} -> {width // alignment * alignment}x{height // alignment * alignment}', flush=True)
    kwargs = {'prompt': prompt, 'negative_prompt': request.get('negative') or None, 'true_cfg_scale': float(request.get('true_cfg_scale', 1.0)), 'height': int(request.get('height', 1024)), 'width': int(request.get('width', 1024)), 'output_resolution': max(int(request.get('width', 1024)), int(request.get('height', 1024))), 'num_inference_steps': int(request.get('steps', 40)), 'generator': torch.Generator(generator_device).manual_seed(int(request.get('seed', 42))), 'use_kv_cache': bool(request.get('use_kv_cache', True)) and (not bool(option and option.enabled)), 'callback_on_step_end': on_step}
    if conditions:
        kwargs['image'] = conditions
    if custom_sigmas is not None:
        kwargs['sigmas'] = custom_sigmas
    print('[status] Qwen phase: pipeline encode/denoise begins', flush=True)
    expected_encodes = 1 + int(kwargs['true_cfg_scale'] > 1 and kwargs['negative_prompt'] is not None)
    lifetime = _RequestEncoderLifetime(pipe, expected_encodes) if encoder_staging else None
    failed = True
    try:
        diagnostics.begin()
        if lifetime is not None:
            lifetime.begin()
        from .two_pass import run_pipeline
        image = run_pipeline(pipe, kwargs, option, lambda specs: _apply_loras(pipe, specs), first_pass=loras).images[0]
        failed = False
    finally:
        try:
            if lifetime is not None:
                lifetime.finish(failed)
        finally:
            diagnostics.finish()
    lora_calls = stack.calls - calls_before if stack is not None else 0
    if stack is not None and lora_calls == 0:
        raise RuntimeError('LoRA hooks were installed but never ran; refusing to save an image as if applied')
    wall = time.perf_counter() - started
    image.save(output)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    gpu = []
    for index in range(torch.cuda.device_count()):
        gpu.append({'index': index, 'name': torch.cuda.get_device_name(index), 'peak_allocated_gb': round(torch.cuda.max_memory_allocated(index) / 2 ** 30, 3), 'peak_reserved_gb': round(torch.cuda.max_memory_reserved(index) / 2 ** 30, 3)})
    result = {'path': str(output), 'seconds': round(wall, 3), 'seed': int(request.get('seed', 42)), 'width': image.width, 'height': image.height, 'mode': image.mode, 'condition_images': len(conditions), 'kv_cache': kwargs['use_kv_cache'], 'text_encoder_8bit': bool(getattr(pipe, '_studio_text_encoder_8bit', False)), 'request_encoder_staging': encoder_staging, 'execution_policy': ENCODER_STAGED_POLICY if encoder_staging else ENCODER_RESIDENT_POLICY, 'model_lifetime_events': lifetime.events if lifetime is not None else [], 'tensor_diagnostics': diagnostics.stats, 'load_seconds': getattr(pipe, '_studio_load_seconds', None), 'weight_loader': 'DirectStorage', 'attention': getattr(pipe.transformer, '_studio_attention', 'unknown'), 'streaming': getattr(pipe, '_studio_streaming', False), 'precision': getattr(pipe, '_studio_precision', 'bf16'), 'block_binds': getattr(getattr(pipe.transformer, '_studio_block_weights', None), 'binds', 0), 'block_evictions': getattr(getattr(pipe.transformer, '_studio_block_weights', None), 'evictions', 0), 'gpu': gpu, 'artifacts': artifacts, 'loras': stack.reports if stack is not None else [], 'lora_hook_calls': lora_calls, 'scheduler': type(pipe.scheduler).__name__, 'sampler': request.get('sampler') or 'euler', 'sigmas': request.get('sigmas') or 'flow', 'two_pass': getattr(pipe, '_studio_two_pass_receipt', None)}
    metrics = output.with_suffix(output.suffix + '.metrics.json')
    metrics.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULT JSON ' + json.dumps(result), flush=True)
    print(f'RESULT DONE {output}', flush=True)
    return result

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream-weights', action='store_true')
    parser.add_argument('--precision', choices=('bf16', 'int8'), default='bf16')
    parser.add_argument('--serve', action='store_true', help='Keep weights resident and read JSON requests on stdin')
    parser.add_argument('--request', help='JSON request manifest')
    parser.add_argument('--model-dir', default=os.environ.get('STUDIO_QWEN_IMAGE21_MODEL', DEFAULT_MODEL_DIR))
    parser.add_argument('--offload-dir', default=os.environ.get('STUDIO_QWEN_IMAGE21_OFFLOAD', ''))
    args = parser.parse_args(argv)
    if args.serve:
        pipe = None
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if pipe is None:
                    print('[status] Loading Qwen-Image-2.1 weights from ' + args.model_dir, flush=True)
                    pipe = _load_pipeline(args.model_dir, args.offload_dir, streaming=args.stream_weights, precision=args.precision, request_encoder_staging=_encoder_staging(request))
                else:
                    print('[status] Reusing resident Qwen-Image-2.1 weights', flush=True)
                result = generate_one(pipe, request)
                print('QWEN_JOB_DONE ' + json.dumps({'ok': True, **result}), flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print('QWEN_JOB_DONE ' + json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
                return 1
        return 0
    if not args.request:
        parser.error('--request or --serve is required')
    document = json.loads(Path(args.request).read_text(encoding='utf-8'))
    requests = document.get('requests') if isinstance(document, dict) else None
    if requests is None:
        requests = [document]
    print('[status] Loading Qwen-Image-2.1 weights from ' + args.model_dir, flush=True)
    pipe = _load_pipeline(args.model_dir, args.offload_dir, streaming=args.stream_weights, precision=args.precision, request_encoder_staging=_encoder_staging(requests[0]))
    print('[status] Qwen-Image-2.1 loaded; preparing generation', flush=True)
    for request in requests:
        generate_one(pipe, request)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
