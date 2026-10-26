# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
QWEN_IMAGE_REPO = os.environ.get('STUDIO_QWEN_IMAGE_REPO', 'Qwen/Qwen-Image')
QWEN3VL_REPO = os.environ.get('STUDIO_QWEN3VL_REPO', 'Qwen/Qwen3-VL-4B-Instruct')
_QWEN3VL_CACHE = os.environ.get('STUDIO_QWEN3VL_CACHE') or str(Path.home() / '.cache' / 'studio_qwen3vl_4b')
KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

def _comfy_models_base() -> Path:
    env = os.environ.get('STUDIO_COMFYUI_MODELS')
    if env and Path(env).is_dir():
        return Path(env)
    if sys.platform == 'win32':
        win_mirror = Path(os.environ.get('STUDIO_KREA2_WIN_MIRROR', ''))
        if win_mirror.is_dir():
            return win_mirror
    cfg = Path.home() / 'Library' / 'Application Support' / 'Comfy Desktop' / 'settings.json'
    try:
        for md in json.loads(cfg.read_text(encoding='utf-8')).get('modelsDirs', []):
            if Path(md).is_dir():
                return Path(md)
    except Exception:
        pass
    return Path.home() / 'ComfyUI-Shared' / 'models'

def krea2_files() -> dict:
    base = _comfy_models_base()
    return {'dit': os.environ.get('STUDIO_KREA2_DIT') or str(base / 'checkpoints' / 'Krea2_Turbo_int8mixed.safetensors'), 'text_encoder': os.environ.get('STUDIO_KREA2_TEXT_ENCODER') or str(base / 'text_encoders' / 'qwen3vl_4b_int8.safetensors'), 'vae': os.environ.get('STUDIO_KREA2_VAE') or str(base / 'vae' / 'qwen_image_vae.safetensors')}

def krea2_available() -> bool:
    f = krea2_files()
    return Path(f['dit']).is_file() and Path(f['text_encoder']).is_file()
KREA2_EDIT_LORA_NAMES = ('krea2_identity_edit_v1_1.safetensors', 'krea2_identity_edit_v1_1_r128.safetensors', 'krea2_identity_edit_v1_1_r64.safetensors', 'krea2_identity_edit_v1.safetensors')
KREA2_EDIT_TEMPLATE_PREFIX = '<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n'
KREA2_EDIT_TEMPLATE_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n'
_KREA2_VISION_START = '<|vision_start|>'
_KREA2_IMAGE_PAD = '<|image_pad|>'
_KREA2_VISION_END = '<|vision_end|>'

def _default_krea2_edit_lora_roots(dit_path: str | None=None) -> list[Path]:
    roots: list[Path] = []
    if dit_path:
        roots.append(Path(dit_path).expanduser().resolve(strict=False).parent / 'loras')
    try:
        roots.append(Path(krea2_files()['dit']).expanduser().resolve(strict=False).parent / 'loras')
    except Exception:
        pass
    roots.extend([Path.home() / 'Documents' / 'models' / 'Krea2' / 'loras', _comfy_models_base() / 'loras'])
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.expanduser().resolve(strict=False))
        if key not in seen:
            seen.add(key)
            out.append(Path(key))
    return out

def list_available_krea_loras() -> list[dict]:
    found: dict[str, dict] = {}
    for root in _default_krea2_edit_lora_roots():
        try:
            if not root.is_dir():
                continue
            for p in sorted(root.glob('*.safetensors')):
                name = p.stem
                if name not in found:
                    found[name] = {'name': name, 'path': str(p)}
        except OSError:
            continue
    return sorted(found.values(), key=lambda r: r['name'].lower())

def resolve_krea2_edit_lora(explicit_path: str | None=None, *, search_roots: list[str | Path] | None=None, dit_path: str | None=None) -> str | None:
    configured = str(explicit_path or '').strip()
    if not configured:
        configured = str(os.environ.get('STUDIO_KREA2_EDIT_LORA') or '').strip()
    if configured:
        path = Path(configured).expanduser().resolve(strict=False)
        if not path.is_file():
            raise RuntimeError(f'Krea 2 Edit LoRA not found: {str(path)!r}')
        if path.suffix.lower() != '.safetensors':
            raise RuntimeError(f"Krea 2 Edit LoRA must be a completed .safetensors file, not {path.name!r}. Wait for the browser's .textClipping download to finish.")
        return str(path)
    roots = [Path(root).expanduser().resolve(strict=False) for root in search_roots] if search_roots is not None else _default_krea2_edit_lora_roots(dit_path)
    for root in roots:
        if not root.is_dir():
            continue
        by_lower = {child.name.lower(): child for child in root.iterdir() if child.is_file()}
        for name in KREA2_EDIT_LORA_NAMES:
            found = by_lower.get(name.lower())
            if found is not None:
                return str(found.resolve())
    for root in roots:
        if not root.is_dir():
            continue
        matches = sorted((path for path in root.iterdir() if path.is_file() and path.suffix.lower() == '.safetensors' and all((word in path.name.lower() for word in ('krea2', 'identity', 'edit')))), key=lambda path: path.name.lower())
        if matches:
            return str(matches[0].resolve())
    return None

def _is_krea2_edit_lora(row: dict) -> bool:
    return 'krea2_identity_edit' in Path(str(row.get('path') or '')).stem.lower()

def ensure_krea2_edit_lora(loras, *, explicit_path: str | None=None, search_roots: list[str | Path] | None=None, dit_path: str | None=None) -> list[dict]:
    stack = [row if isinstance(row, dict) else row.model_dump() for row in loras or [] if isinstance(row, dict) or hasattr(row, 'model_dump')]
    if not explicit_path and any((_is_krea2_edit_lora(row) for row in stack)):
        return stack
    resolved = resolve_krea2_edit_lora(explicit_path, search_roots=search_roots, dit_path=dit_path)
    if not resolved:
        searched = ', '.join((str(path) for path in ([Path(root) for root in search_roots] if search_roots is not None else _default_krea2_edit_lora_roots(dit_path))))
        raise RuntimeError(f"Krea 2 Edit requires the identity-edit LoRA. Put {KREA2_EDIT_LORA_NAMES[0]} in the Krea2 loras folder, select it explicitly, or set STUDIO_KREA2_EDIT_LORA. Searched: {searched or 'no directories'}")
    stack = [row for row in stack if not _is_krea2_edit_lora(row)]
    return [{'path': resolved, 'weight': 1.0, 'name': Path(resolved).stem}, *stack]

def _krea2_grounded_prompt(prompt: str, grids: list[tuple[int, int, int]]) -> str:
    vision_blocks = []
    for grid_t, grid_h, grid_w in grids:
        visual_tokens = int(grid_t) * int(grid_h) * int(grid_w) // 4
        vision_blocks.append(_KREA2_VISION_START + _KREA2_IMAGE_PAD * visual_tokens + _KREA2_VISION_END)
    return KREA2_EDIT_TEMPLATE_PREFIX + ''.join(vision_blocks) + str(prompt or '') + KREA2_EDIT_TEMPLATE_SUFFIX
KREA2_EDIT_RECIPES = ('identity', 'ostris')
OSTRIS_VLM_MAX_PIXELS = 384 * 384
OSTRIS_REF_MAX_PIXELS = 1024 * 1024
OSTRIS_REF_SNAP = 16

def _krea2_fit_area(image, max_pixels: int, snap: int=1):
    import math
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    new_w = max(round(width * scale / snap) * snap, snap)
    new_h = max(round(height * scale / snap) * snap, snap)
    if (new_w, new_h) == (width, height):
        return image
    return image.resize((new_w, new_h), Image.Resampling.BOX)

def _krea2_ostris_prompt(prompt: str, grids: list[tuple[int, int, int]]) -> str:
    blocks = []
    for index, (grid_t, grid_h, grid_w) in enumerate(grids, start=1):
        visual_tokens = int(grid_t) * int(grid_h) * int(grid_w) // 4
        blocks.append(f'Picture {index}: ' + _KREA2_VISION_START + _KREA2_IMAGE_PAD * visual_tokens + _KREA2_VISION_END)
    return KREA2_EDIT_TEMPLATE_PREFIX + ''.join(blocks) + str(prompt or '') + KREA2_EDIT_TEMPLATE_SUFFIX

def _krea2_ostris_edit_forward(transformer, refs: list, *, hidden_states, encoder_hidden_states, timestep, position_ids, encoder_attention_mask=None, attention_kwargs=None, return_dict=True):
    import torch
    import torch.nn.functional as F
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f'position_ids must have shape (sequence_length, 3), got {tuple(position_ids.shape)}')
    if not refs:
        raise ValueError('Krea 2 Ostris edit conditioning requires at least one reference')
    batch_size, target_len, _ = hidden_states.shape
    text_len = encoder_hidden_states.shape[1]
    ref_len = sum((int(tokens.shape[1]) for tokens, _, _ in refs))
    temb = transformer.time_embed(timestep, dtype=hidden_states.dtype)
    step_mod = transformer.time_mod_proj(F.gelu(temb, approximate='tanh'))
    zero = transformer.time_embed(torch.zeros_like(timestep), dtype=hidden_states.dtype)
    zero_mod = transformer.time_mod_proj(F.gelu(zero, approximate='tanh'))
    token_mod = torch.cat([step_mod.expand(batch_size, text_len + target_len, -1), zero_mod.expand(batch_size, ref_len, -1)], dim=1)
    text_attention_mask = attention_mask = None
    if encoder_attention_mask is not None:
        text_attention_mask = encoder_attention_mask[:, None, None, :]
        image_mask = encoder_attention_mask.new_ones((batch_size, target_len + ref_len))
        attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]
    text_states = transformer.txt_in(transformer.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask))
    target_states = transformer.img_in(hidden_states)
    ref_states = [transformer.img_in(tokens.to(hidden_states.dtype)) for tokens, _, _ in refs]
    combined = torch.cat([text_states, target_states, *ref_states], dim=1)
    device = position_ids.device
    ref_positions = []
    for frame, (_, grid_h, grid_w) in enumerate(refs, start=1):
        ids = torch.zeros(grid_h, grid_w, 3, device=device, dtype=position_ids.dtype)
        ids[..., 0] = float(frame)
        ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        ref_positions.append(ids.reshape(grid_h * grid_w, 3))
    rotary = transformer.rotary_emb(torch.cat([position_ids[:text_len], position_ids[-target_len:], *ref_positions], dim=0))
    for block in transformer.transformer_blocks:
        combined = block(combined, token_mod, rotary, attention_mask)
    output = transformer.final_layer(combined[:, text_len:text_len + target_len], temb)
    if not return_dict:
        return (output,)
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    return Transformer2DModelOutput(sample=output)

def _prepare_krea2_grounding_image(image, grounding_px: int=768):
    import numpy as np
    import torch
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    cap = max(0, int(grounding_px or 0))
    if cap and max(width, height) > cap:
        scale = cap / float(max(width, height))
        width = max(32, round(width * scale))
        height = max(32, round(height * scale))
    factor = 32
    width = max(factor, round(width / factor) * factor)
    height = max(factor, round(height / factor) * factor)
    if max(width, height) / min(width, height) > 200:
        raise RuntimeError('Krea 2 Edit grounding image aspect ratio must be below 200:1')
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = np.array(image, dtype=np.float32, copy=True)
    frame = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    frame = (frame - 0.5) / 0.5
    patches = frame.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)
    grid_t, grid_h, grid_w = (1, height // 16, width // 16)
    patches = patches.view(1, grid_t, 2, 3, grid_h // 2, 2, 16, grid_w // 2, 2, 16)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    patches = patches.reshape(grid_t * grid_h * grid_w, 3 * 2 * 16 * 16)
    grid = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    return (patches, grid)

def _krea2_edit_transformer_forward(transformer, source_tokens: list, *, hidden_states, encoder_hidden_states, timestep, position_ids, encoder_attention_mask=None, attention_kwargs=None, return_dict=True):
    import torch
    import torch.nn.functional as F
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f'position_ids must have shape (sequence_length, 3), got {tuple(position_ids.shape)}')
    batch_size, target_len, _ = hidden_states.shape
    text_len = encoder_hidden_states.shape[1]
    refs = list(source_tokens or [])
    if not refs:
        raise ValueError('Krea 2 Edit requires at least one source-token block')
    if any((ref.shape[1] != target_len for ref in refs)):
        raise ValueError('Krea 2 Edit source tokens must be resized to the target token grid')
    temb = transformer.time_embed(timestep, dtype=hidden_states.dtype)
    temb_mod = transformer.time_mod_proj(F.gelu(temb, approximate='tanh'))
    text_attention_mask = None
    attention_mask = None
    if encoder_attention_mask is not None:
        text_attention_mask = encoder_attention_mask[:, None, None, :]
        image_mask = encoder_attention_mask.new_ones((batch_size, target_len * (len(refs) + 1)))
        attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]
    text_states = transformer.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
    text_states = transformer.txt_in(text_states)
    target_states = transformer.img_in(hidden_states)
    ref_states = [transformer.img_in(ref) for ref in refs]
    combined = torch.cat([text_states, *ref_states, target_states], dim=1)
    text_positions = position_ids[:text_len]
    target_positions = position_ids[-target_len:]
    ref_positions = []
    for frame in range(1, len(refs) + 1):
        ids = target_positions.clone()
        ids[:, 0] = float(frame)
        ref_positions.append(ids)
    edit_position_ids = torch.cat([text_positions, *ref_positions, target_positions], dim=0)
    rotary = transformer.rotary_emb(edit_position_ids)
    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            combined = transformer._gradient_checkpointing_func(block, combined, temb_mod, rotary, attention_mask)
        else:
            combined = block(combined, temb_mod, rotary, attention_mask)
    source_len = target_len * len(refs)
    output = transformer.final_layer(combined[:, text_len + source_len:], temb)
    if not return_dict:
        return (output,)
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    return Transformer2DModelOutput(sample=output)

def dequant_sd(sd: dict) -> dict:
    import torch
    out: dict = {}
    for k, v in sd.items():
        if k.endswith('.comfy_quant') or k.endswith('.weight_scale'):
            continue
        scale = sd.get(k + '_scale') if k.endswith('.weight') else None
        if scale is not None:
            w = v.to(torch.float32)
            s = scale.to(torch.float32)
            if s.numel() == w.shape[0]:
                w = w * s.reshape(-1, *[1] * (w.dim() - 1))
            elif s.numel() == 1:
                w = w * s.reshape(())
            out[k] = w.to(torch.bfloat16)
        else:
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
    return out
_BLOCK_MAP = {'attn.wq': 'attn.to_q', 'attn.wk': 'attn.to_k', 'attn.wv': 'attn.to_v', 'attn.gate': 'attn.to_gate', 'attn.wo': 'attn.to_out.0', 'attn.qknorm.qnorm': 'attn.norm_q', 'attn.qknorm.knorm': 'attn.norm_k', 'mlp.gate': 'ff.gate', 'mlp.up': 'ff.up', 'mlp.down': 'ff.down', 'prenorm': 'norm1', 'postnorm': 'norm2'}
_TOP_MAP = {'first': 'img_in', 'tmlp.0': 'time_embed.linear_1', 'tmlp.2': 'time_embed.linear_2', 'tproj.1': 'time_mod_proj', 'txtmlp.0': 'txt_in.norm', 'txtmlp.1': 'txt_in.linear_1', 'txtmlp.3': 'txt_in.linear_2', 'txtfusion.projector': 'text_fusion.projector', 'last.norm': 'final_layer.norm', 'last.linear': 'final_layer.linear'}
_GROUP_PREFIX = {'blocks': 'transformer_blocks', 'txtfusion.layerwise_blocks': 'text_fusion.layerwise_blocks', 'txtfusion.refiner_blocks': 'text_fusion.refiner_blocks'}

def _normalize_dit_source_key(source_key: str) -> str:
    key = str(source_key)
    if key.startswith('lora_unet_'):
        flat = key[len('lora_unet_'):]
        suffix = '.weight' if flat.endswith('.weight') else ''
        flat = flat[:-len(suffix)] if suffix else flat
        for group in sorted(_GROUP_PREFIX, key=len, reverse=True):
            prefix = group.replace('.', '_') + '_'
            if flat.startswith(prefix):
                index, separator, tail = flat[len(prefix):].partition('_')
                if separator and index.isdigit():
                    leaf = next((name for name in _BLOCK_MAP if name.replace('.', '_') == tail), None)
                    if leaf:
                        return f'{group}.{index}.{leaf}{suffix}'
        top = next((name for name in _TOP_MAP if name.replace('.', '_') == flat), None)
        if top:
            return top + suffix
        raise ValueError(f'Unknown flattened Krea LoRA target: {key}')
    for prefix in ('model.diffusion_model.', 'diffusion_model.'):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key

def _diffusers_suffix(diff_base: str, egg_key: str) -> str:
    if egg_key.endswith('.scale'):
        return diff_base + '.weight'
    if egg_key.endswith('.bias'):
        return diff_base + '.bias'
    return diff_base + '.weight'

def _map_dit_key(egg_key: str) -> tuple[str, str | None] | None:
    egg_key = _normalize_dit_source_key(egg_key)
    if egg_key.startswith('last.up') or egg_key.startswith('last.down'):
        return None
    m = re.match('^(blocks|txtfusion\\.layerwise_blocks|txtfusion\\.refiner_blocks)\\.(\\d+)\\.(.+)$', egg_key)
    if m:
        grp, idx, rest = (m.group(1), m.group(2), m.group(3))
        pfx = _GROUP_PREFIX[grp]
        if rest == 'mod.lin':
            return (f'{pfx}.{idx}.scale_shift_table', 'block_mod')
        sub, _ = rest.rsplit('.', 1)
        if sub not in _BLOCK_MAP:
            raise KeyError(f'unmapped Krea2 block submodule {sub!r} (from {egg_key!r})')
        return (_diffusers_suffix(f'{pfx}.{idx}.{_BLOCK_MAP[sub]}', egg_key), None)
    if egg_key == 'last.modulation.lin':
        return ('final_layer.scale_shift_table', None)
    sub, _ = egg_key.rsplit('.', 1)
    if sub not in _TOP_MAP:
        raise KeyError(f'unmapped Krea2 top-level key {egg_key!r}')
    return (_diffusers_suffix(_TOP_MAP[sub], egg_key), None)

def _reshape_mapped_value(value, reshape_kind: str | None):
    if reshape_kind == 'block_mod':
        return value.reshape(6, -1)
    return value

def map_dit_to_diffusers(egg: dict) -> dict:
    out: dict = {}
    for k, v in egg.items():
        mapped = _map_dit_key(k)
        if mapped is None:
            continue
        target, reshape_kind = mapped
        out[target] = _reshape_mapped_value(v, reshape_kind)
    return out

def _dequant_stream(path: str) -> dict:
    import torch
    from safetensors import safe_open
    out: dict = {}
    with safe_open(path, framework='pt') as f:
        keys = set(f.keys())
        for k in keys:
            if k.endswith('.comfy_quant') or k.endswith('.weight_scale'):
                continue
            if k.endswith('.weight') and k + '_scale' in keys:
                w = f.get_tensor(k).to(torch.float32)
                s = f.get_tensor(k + '_scale').to(torch.float32)
                if s.numel() == w.shape[0]:
                    w = w * s.reshape(-1, *[1] * (w.dim() - 1))
                elif s.numel() == 1:
                    w = w * s.reshape(())
                out[k] = w.to(torch.bfloat16)
                del w, s
            else:
                t = f.get_tensor(k)
                out[k] = t.to(torch.bfloat16) if t.is_floating_point() else t
                del t
    return out
_CONVROT_HADAMARD_CACHE: dict[tuple[int, str, object], object] = {}

def _source_quant_config(sf, keys: set[str], weight_key: str) -> dict:
    if not weight_key.endswith('.weight'):
        return {}
    config_key = weight_key[:-len('.weight')] + '.comfy_quant'
    if config_key not in keys:
        return {}
    try:
        raw = bytes(sf.get_tensor(config_key).reshape(-1).tolist()).decode('utf-8')
        import json
        config = json.loads(raw)
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}

def _regular_hadamard(size: int, *, device: str, dtype):
    import math
    import torch
    if size < 4 or size & size - 1 != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f'Krea2 INT8 ConvRot group size must be a power of 4, got {size}')
    cache_key = (size, str(device), dtype)
    cached = _CONVROT_HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device=device, dtype=dtype)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / size ** 0.5
    _CONVROT_HADAMARD_CACHE[cache_key] = h
    return h

def _inverse_convrot_weight(weight, group_size: int):
    import torch
    if weight.ndim != 2:
        raise ValueError(f'Krea2 INT8 ConvRot expects a 2D linear weight, got shape {tuple(weight.shape)}')
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f'Krea2 INT8 ConvRot in_features {in_features} not divisible by group size {group_size}')
    h = _regular_hadamard(group_size, device=str(weight.device), dtype=weight.dtype)
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    return torch.matmul(grouped, h).reshape(out_features, in_features)

def _source_weight_needs_cpu_float_cast(source_dtype) -> bool:
    return str(source_dtype).startswith('torch.float8')

def _krea2_resident_dtype():
    import torch
    raw = str(os.environ.get('STUDIO_KREA2_DTYPE', 'float16')).strip().lower()
    if raw in {'bfloat16', 'bf16'}:
        return torch.bfloat16
    if raw in {'float16', 'fp16', 'half'}:
        return torch.float16
    raise ValueError(f'unsupported STUDIO_KREA2_DTYPE={raw!r}; expected bfloat16/bf16 or float16/fp16')

def _krea2_target_dtype(model, target: str, default_dtype):
    import torch
    module_path, separator, leaf = str(target).rpartition('.')
    if separator and leaf == 'weight':
        try:
            module = model.get_submodule(module_path)
            if type(module).__name__ == 'Krea2RMSNorm':
                return torch.float32
        except Exception:
            pass
    return default_dtype

def _read_source_tensor(sf, keys: set[str], key: str, dtype, device: str):
    import torch
    if key.endswith('.weight') and key + '_scale' in keys:
        quant_config = _source_quant_config(sf, keys, key)
        source_weight = sf.get_tensor(key)
        if _source_weight_needs_cpu_float_cast(source_weight.dtype):
            w = source_weight.to(torch.float32).to(device)
        else:
            w = source_weight.to(device).to(torch.float32)
        s = sf.get_tensor(key + '_scale').to(device=device, dtype=torch.float32)
        if s.numel() == w.shape[0]:
            w = w * s.reshape(-1, *[1] * (w.dim() - 1))
        elif s.numel() == 1:
            w = w * s.reshape(())
        params_config = quant_config.get('params') if isinstance(quant_config.get('params'), dict) else {}
        if quant_config.get('convrot', params_config.get('convrot', False)):
            group_size = int(quant_config.get('convrot_groupsize', params_config.get('convrot_groupsize', 256)))
            w = _inverse_convrot_weight(w, group_size)
        return w.to(dtype)
    tensor = sf.get_tensor(key)
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device)

def _krea2_layer_group(target_key: str) -> tuple[str, int] | None:
    for prefix, label in (('text_fusion.layerwise_blocks.', 'text_layerwise'), ('text_fusion.refiner_blocks.', 'text_refiner'), ('transformer_blocks.', 'transformer')):
        if target_key.startswith(prefix):
            rest = target_key[len(prefix):]
            idx = rest.split('.', 1)[0]
            if idx.isdigit():
                return (label, int(idx))
    return None

class _Krea2TransformerStreamer:

    def __init__(self, model, source_path: str, *, dtype, device: str):
        from safetensors import safe_open
        self.model = model
        self.source_path = str(source_path)
        self.dtype = dtype
        self.device = str(device)
        self._sf_cm = safe_open(self.source_path, framework='pt', device='cpu')
        self._sf = self._sf_cm.__enter__()
        self.keys = set(self._sf.keys())
        self.entries: dict[str, tuple[str, str | None]] = {}
        self.groups: dict[tuple[str, int], list[str]] = {}
        self.globals: list[str] = []
        self._lora_cms: list = []
        self._lora_handles: list = []
        self._lora_by_target: dict[str, list[tuple]] = {}
        self._lora_modules: list = []
        expected = set(model.state_dict().keys())
        unexpected: list[str] = []
        for src_key in sorted(self.keys):
            if src_key.endswith('.comfy_quant') or src_key.endswith('.weight_scale'):
                continue
            mapped = _map_dit_key(_normalize_dit_source_key(src_key))
            if mapped is None:
                continue
            target, reshape_kind = mapped
            if target not in expected:
                unexpected.append(f'{src_key}->{target}')
                continue
            self.entries[target] = (src_key, reshape_kind)
            group = _krea2_layer_group(target)
            if group is None:
                self.globals.append(target)
            else:
                self.groups.setdefault(group, []).append(target)
        missing = sorted(expected - set(self.entries))
        if missing or unexpected:
            raise RuntimeError(f'Krea2 streamed DiT key mismatch: missing={missing[:6]} unexpected={unexpected[:6]}')

    def close(self) -> None:
        for cm in self._lora_cms:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        self._lora_cms.clear()
        self._lora_handles.clear()
        try:
            self._sf_cm.__exit__(None, None, None)
        except Exception:
            pass

    def _read_target(self, target: str):
        src_key, reshape_kind = self.entries[target]
        target_dtype = _krea2_target_dtype(self.model, target, self.dtype)
        value = _read_source_tensor(self._sf, self.keys, src_key, target_dtype, self.device)
        value = _reshape_mapped_value(value, reshape_kind)
        value = self._apply_loras(target, value)
        want = tuple(self.model.state_dict()[target].shape)
        if tuple(value.shape) != want:
            raise RuntimeError(f'Krea2 tensor shape mismatch for {target}: checkpoint {tuple(value.shape)} != module {want}')
        return value

    def _set_target(self, target: str, value) -> None:
        from accelerate.utils import set_module_tensor_to_device
        assign_dtype = _krea2_target_dtype(self.model, target, self.dtype) if getattr(value, 'is_floating_point', lambda: False)() else None
        set_module_tensor_to_device(self.model, target, self.device, value=value, dtype=assign_dtype)

    def _bind_targets(self, targets: list[str]) -> None:
        for target in targets:
            self._set_target(target, self._read_target(target))

    def _evict_targets(self, targets: list[str]) -> None:
        from accelerate.utils import set_module_tensor_to_device
        for target in targets:
            set_module_tensor_to_device(self.model, target, 'meta')

    def materialize_globals(self) -> int:
        self._bind_targets(self.globals)
        return len(self.globals)

    def materialize_layers(self) -> int:
        count = 0
        for targets in self.groups.values():
            self._bind_targets(targets)
            count += len(targets)
        return count

    def materialize_text_fusion(self) -> int:
        count = 0
        for (label, _idx), targets in self.groups.items():
            if label in ('text_layerwise', 'text_refiner'):
                self._bind_targets(targets)
                count += len(targets)
        return count

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
            module._krea2_raw_forward = raw_forward
        for i, block in enumerate(self.model.text_fusion.layerwise_blocks):
            targets = sorted(self.groups.get(('text_layerwise', i), []))
            if targets:
                _wrap(block, targets)
        for i, block in enumerate(self.model.text_fusion.refiner_blocks):
            targets = sorted(self.groups.get(('text_refiner', i), []))
            if targets:
                _wrap(block, targets)
        for i, block in enumerate(self.model.transformer_blocks):
            targets = sorted(self.groups.get(('transformer', i), []))
            if targets:
                _wrap(block, targets)
        self.model._krea2_streamer = self

    def set_loras(self, loras) -> list[dict]:
        from safetensors import safe_open
        for module in self._lora_modules:
            try:
                module._krea2_lora_adapters = []
            except Exception:
                pass
        for cm in self._lora_cms:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        self._lora_cms.clear()
        self._lora_handles.clear()
        self._lora_by_target = {}
        applied: list[dict] = []
        known_targets = set(self.entries)
        for lora in loras or []:
            d = lora if isinstance(lora, dict) else lora.model_dump() if hasattr(lora, 'model_dump') else {}
            path = str(d.get('path') or '').strip()
            if not path:
                continue
            if not Path(path).is_file():
                raise RuntimeError(f'LoRA file not found: {path!r}')
            weight = float(d.get('weight', 1.0) or 1.0)
            raw_name = str(d.get('name') or '').strip() or Path(path).stem
            with safe_open(path, framework='pt', device='cpu') as handle:
                keys = set(handle.keys())
                down_suffix = '.lora_down.weight'
                up_suffix = '.lora_up.weight'
                down_keys = sorted((k for k in keys if k.endswith(down_suffix)))
                a_suffix = '.lora_A.weight'
                b_suffix = '.lora_B.weight'
                a_keys = sorted((k for k in keys if k.endswith(a_suffix)))
                diff_keys = sorted((k for k in keys if k.endswith('.diff') or k.endswith('.diff_b')))
                if not down_keys and (not a_keys) and (not diff_keys):
                    raise RuntimeError(f'streamed Krea2 only supports Comfy down/up, AI-Toolkit A/B, or diff patches; unsupported adapter {path!r}')
                pairs = [(key, key[:-len(down_suffix)] + up_suffix, down_suffix) for key in down_keys] + [(key, key[:-len(a_suffix)] + b_suffix, a_suffix) for key in a_keys]
                for down_key, up_key, pair_suffix in pairs:
                    if up_key not in keys:
                        raise RuntimeError(f'missing LoRA up tensor for {down_key!r}')
                    base = down_key[:-len(pair_suffix)]
                    mapped = _krea2_lora_target(base, known_targets)
                    if mapped is None:
                        continue
                    target, _reshape = mapped
                    if target not in known_targets:
                        raise RuntimeError(f'LoRA target {target!r} (from {down_key!r}) is not in streamed Krea2')
                    down = handle.get_tensor(down_key).to(device=self.device, dtype=self.dtype)
                    up = handle.get_tensor(up_key).to(device=self.device, dtype=self.dtype)
                    self._install_linear_lora(target, down, up, weight)
                for diff_key in diff_keys:
                    base = diff_key[len('diffusion_model.'):] if diff_key.startswith('diffusion_model.') else diff_key
                    if base.endswith('.diff_b'):
                        mapped = _map_dit_key(base[:-len('.diff_b')] + '.bias')
                    else:
                        mapped = _map_dit_key(base[:-len('.diff')] + '.weight')
                    if mapped is None:
                        continue
                    target, _reshape = mapped
                    if target not in known_targets:
                        raise RuntimeError(f'diff-patch target {target!r} (from {diff_key!r}) is not in streamed Krea2')
                    delta = handle.get_tensor(diff_key).to(device=self.device, dtype=self.dtype)
                    self._lora_by_target.setdefault(target, []).append(('diff', delta, None, None, weight))
            kind = 'streamed_ai_toolkit_lora' if a_keys and (not down_keys) else 'streamed_comfy_lora'
            applied.append({'name': raw_name, 'weight': weight, 'path': path, 'kind': kind})
        return applied

    def _install_linear_lora(self, target: str, down, up, strength: float) -> None:
        import torch
        import torch.nn.functional as F
        module_path, leaf = target.rsplit('.', 1)
        if leaf != 'weight':
            raise RuntimeError(f'streamed Krea2 LoRA target must be a weight tensor, got {target!r}')
        module = self.model.get_submodule(module_path)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f'streamed Krea2 LoRA target {module_path!r} is {type(module).__name__}, not Linear')
        if not hasattr(module, '_krea2_base_forward'):
            module._krea2_base_forward = module.forward

            def _forward(input, _module=module):
                out = _module._krea2_base_forward(input)
                for l_down, l_up, l_strength in getattr(_module, '_krea2_lora_adapters', []):
                    out = out + F.linear(F.linear(input, l_down), l_up) * float(l_strength)
                return out
            module.forward = _forward
            self._lora_modules.append(module)
        adapters = getattr(module, '_krea2_lora_adapters', [])
        adapters.append((down, up, float(strength)))
        module._krea2_lora_adapters = adapters

    def _apply_loras(self, target: str, value):
        out = value
        for kind, delta, _unused_a, _unused_b, strength in self._lora_by_target.get(target, []):
            if kind != 'diff':
                continue
            delta = delta.reshape(out.shape)
            out = out + (delta * float(strength)).to(out.dtype)
        return out

def _require_free_memory(need_gb: float=55.0) -> None:
    try:
        need_gb = float(os.environ.get('STUDIO_KREA2_MIN_FREE_GB', need_gb))
    except (TypeError, ValueError):
        pass
    if need_gb <= 0:
        return
    from . import ram_preflight
    ram_preflight.preflight_resident_ram(need_gb, 'Krea-2 native load â€” int8->bf16 DiT + Qwen3-VL text encoder + VAE staged for GPU', env_prefix='KREA2', advice='Close other apps / unload other models, then retry (Krea2 runs ONE image model at a time).')

def _ensure_qwen3vl_assets() -> str:
    cache = Path(_QWEN3VL_CACHE)
    if (cache / 'config.json').is_file() and (cache / 'tokenizer_config.json').is_file():
        return str(cache)
    from huggingface_hub import snapshot_download
    snapshot_download(QWEN3VL_REPO, local_dir=str(cache), allow_patterns=['config.json', 'generation_config.json', 'tokenizer*', '*.json', 'merges.txt', 'vocab*', 'chat_template*'])
    return str(cache)

def _ensure_krea2_classes() -> None:
    from .boogu_krea import _ensure_krea2_installed
    _ensure_krea2_installed()
    try:
        from diffusers import Krea2Pipeline
        return
    except ImportError:
        pass
    import diffusers
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
    diffusers.models.Krea2Transformer2DModel = Krea2Transformer2DModel
    diffusers.Krea2Transformer2DModel = Krea2Transformer2DModel
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
    diffusers.Krea2Pipeline = Krea2Pipeline
_KREA2_EGG_TBLOCK_RE = re.compile('^blocks\\.(\\d+)\\.')

def _convrot_group_from_comfy_quant(cq_tensor) -> int | None:
    if cq_tensor is None:
        return None
    try:
        import json
        if hasattr(cq_tensor, 'reshape'):
            raw = bytes(cq_tensor.reshape(-1).tolist())
        else:
            raw = bytes(cq_tensor)
        cfg = json.loads(raw.decode('utf-8'))
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    params = cfg.get('params') if isinstance(cfg.get('params'), dict) else {}
    if cfg.get('convrot', params.get('convrot', False)):
        return int(cfg.get('convrot_groupsize', params.get('convrot_groupsize', 256)))
    return None

def _dequant_streamed(weight_t, scale_t, dtype, *, convrot_group=None):
    import torch
    w = weight_t.to(torch.float32)
    s = scale_t.to(torch.float32)
    if s.numel() == w.shape[0]:
        w = w * s.reshape(-1, *[1] * (w.dim() - 1))
    elif s.numel() == 1:
        w = w * s.reshape(())
    if convrot_group:
        w = _inverse_convrot_weight(w, int(convrot_group))
    return w.to(dtype)

def _krea2_quantization_groups(path):
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        return {name[:-len('.comfy_quant')] + '.weight': _convrot_group_from_comfy_quant(handle.get_tensor(name)) for name in handle.keys() if name.endswith('.comfy_quant')}

def _krea2_group_transformer_blocks(idx):
    blocks, globs = ({}, [])
    for name, ref in idx.refs.items():
        m = _KREA2_EGG_TBLOCK_RE.match(name)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    ordered = {i: blocks[k] for i, k in enumerate(sorted(blocks))}
    return (ordered, globs)

class _Krea2TBlockIndex:

    def __init__(self, idx):
        self._idx = idx

    def group_by_layer(self, component=''):
        return _krea2_group_transformer_blocks(self._idx)

    def __getattr__(self, k):
        return getattr(self._idx, k)

def _attach_ds_transformer_blocks(tr, dit_path: str, dtype, device: str, n_gpu_buffers: int=3):
    import sys
    import os as _os
    from accelerate.utils import set_module_tensor_to_device
    repo = _os.environ.get('STUDIO_COSMOS_WINDOWS_PATH', '')
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    idx = build_index(dit_path)
    quantization_groups = _krea2_quantization_groups(dit_path)
    sd_keys = set(tr.state_dict().keys())
    streamer = DirectStorageStreamer(_Krea2TBlockIndex(idx), component='', n_gpu_buffers=n_gpu_buffers, device=device, record_compute_done=True)
    if getattr(streamer, 'mode', None) != 3 and _os.environ.get('KREA2_ALLOW_NON_DIRECTSTORAGE') != '1':
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"Krea2 DS streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3) â€” R1 requires the optimised NVMe->GPU path. Set KREA2_ALLOW_NON_DIRECTSTORAGE=1 for diagnostics.")
    if streamer.n != len(tr.transformer_blocks):
        streamer.close()
        raise RuntimeError(f'Krea2 DS block count {streamer.n} != {len(tr.transformer_blocks)} transformer_blocks (egg->diffusers block grouping mismatch)')

    def _targets_for(il):
        out = []
        for r, _ao in streamer.plan[il][0]:
            name = r.name
            if name.endswith('.weight_scale') or name.endswith('.comfy_quant'):
                continue
            mapped = _map_dit_key(name)
            if mapped and mapped[0] in sd_keys:
                out.append(mapped[0])
        return out
    block_targets = {il: _targets_for(il) for il in range(streamer.n)}
    layer_timings = []
    tr._krea2_layer_timings = layer_timings

    def make_pre(il):

        def hook(_mod, _args):
            started = time.perf_counter()
            streamed = streamer.wait_layer(il)
            landed = time.perf_counter()
            for name, t in streamed.items():
                if name.endswith('.weight_scale') or name.endswith('.comfy_quant'):
                    continue
                mapped = _map_dit_key(name)
                if mapped is None or mapped[0] not in sd_keys:
                    continue
                target, reshape_kind = mapped
                scale = streamed.get(name + '_scale')
                if scale is not None:
                    value = _dequant_streamed(t, scale, dtype, convrot_group=quantization_groups.get(name))
                elif t.is_floating_point():
                    value = t.to(dtype)
                else:
                    value = t.clone()
                value = _reshape_mapped_value(value, reshape_kind)
                set_module_tensor_to_device(tr, target, device, value=value, dtype=dtype if value.is_floating_point() else None)
            layer_timings.append({'layer': il, 'bytes': sum((t.numel() * t.element_size() for t in streamed.values())), 'wait_seconds': landed - started, 'bind_submit_seconds': time.perf_counter() - landed})
        return hook

    def make_post(il):

        def hook(_mod, _args, output):
            for target in block_targets[il]:
                set_module_tensor_to_device(tr, target, 'meta')
            streamer.release_layer(il)
            return output
        return hook

    def new_pass(_module, _args):
        layer_timings.clear()
        streamer.new_pass()
    tr.register_forward_pre_hook(new_pass)
    for il, block in enumerate(tr.transformer_blocks):
        block.register_forward_pre_hook(make_pre(il))
        block.register_forward_hook(make_post(il))
    tr._krea2_ds_streamer = streamer
    return streamer

class _Krea2Int8ResidentHandle:

    def __init__(self, resident):
        self._resident = resident

    def close(self):
        try:
            self._resident.clear()
        except Exception:
            pass
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def _attach_int8_resident_transformer_blocks(tr, dit_path: str, dtype, device: str):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    sd_keys = set(tr.state_dict().keys())
    resident: dict = {}
    quantization_groups = _krea2_quantization_groups(dit_path)
    if str(device).startswith('cuda'):
        from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
        from cosmos_windows.weight_streaming.safetensors_index import build_index
        index = _Krea2TBlockIndex(build_index(dit_path))
        transport = DirectStorageStreamer(index, component='', n_gpu_buffers=1, device=device, record_compute_done=True)
        try:
            if transport.mode != 3:
                raise RuntimeError('Krea2 resident weights require DirectStorage mode 3')
            transport.new_pass()
            for block in range(transport.n):
                for name, value in transport.wait_layer(block).items():
                    if not name.endswith('.comfy_quant'):
                        resident[name] = value.clone()
                torch.cuda.synchronize(device)
                transport.release_layer(block)
        finally:
            transport.close()
    else:
        from safetensors import safe_open
        with safe_open(dit_path, framework='pt', device=str(device)) as sf:
            for name in sf.keys():
                if _KREA2_EGG_TBLOCK_RE.match(name) and (not name.endswith('.comfy_quant')):
                    resident[name] = sf.get_tensor(name)
    by_block: dict = {}
    for k in resident:
        if k.endswith('.weight_scale') or k.endswith('.comfy_quant'):
            continue
        m = _KREA2_EGG_TBLOCK_RE.match(k)
        if m:
            by_block.setdefault(int(m.group(1)), []).append(k)

    def _targets_for(il):
        out = []
        for wk in by_block.get(il, []):
            mapped = _map_dit_key(wk)
            if mapped and mapped[0] in sd_keys:
                out.append(mapped[0])
        return out
    block_targets = {il: _targets_for(il) for il in range(len(tr.transformer_blocks))}

    def make_pre(il):
        keys_for = by_block.get(il, [])

        def hook(_m, _a):
            for wk in keys_for:
                mapped = _map_dit_key(wk)
                if mapped is None or mapped[0] not in sd_keys:
                    continue
                target, reshape_kind = mapped
                t = resident[wk]
                scale = resident.get(wk + '_scale')
                if scale is not None:
                    value = _dequant_streamed(t, scale, dtype, convrot_group=quantization_groups.get(wk))
                elif t.is_floating_point():
                    value = t.to(dtype)
                else:
                    value = t.clone()
                value = _reshape_mapped_value(value, reshape_kind)
                set_module_tensor_to_device(tr, target, device, value=value, dtype=dtype if value.is_floating_point() else None)
        return hook

    def make_post(il):

        def hook(_m, _a, output):
            for target in block_targets[il]:
                set_module_tensor_to_device(tr, target, 'meta')
            return output
        return hook
    for il, block in enumerate(tr.transformer_blocks):
        block.register_forward_pre_hook(make_pre(il))
        block.register_forward_hook(make_post(il))
    handle = _Krea2Int8ResidentHandle(resident)
    tr._krea2_int8_resident = handle
    return handle

def _load_transformer(dit_path: str, dtype, *, device: str='cpu', stream_layers: bool=False, int8_resident: bool=False):
    import torch
    from diffusers import Krea2Transformer2DModel
    with torch.device('meta'):
        tr = Krea2Transformer2DModel()
    streamer = _Krea2TransformerStreamer(tr, dit_path, dtype=dtype, device=device)
    n_globals = streamer.materialize_globals()
    if int8_resident and (not stream_layers):
        n_tf = streamer.materialize_text_fusion()
        streamer.close()
        rs = _attach_int8_resident_transformer_blocks(tr, dit_path, dtype, device)
        tr._krea2_streamer = rs
        print(f'[krea2-native] int8-RESIDENT DiT: {n_globals} globals + {n_tf} text-fusion resident, {len(tr.transformer_blocks)} transformer_blocks int8-in-VRAM (dequant per-forward)', flush=True)
    elif stream_layers:
        n_tf = streamer.materialize_text_fusion()
        streamer.close()
        ds = _attach_ds_transformer_blocks(tr, dit_path, dtype, device)
        tr._krea2_streamer = ds
        print(f'[krea2-native] DS-streamed DiT: {n_globals} globals + {n_tf} text-fusion resident, {ds.n} transformer_blocks via DirectStorage (mode={ds.mode})', flush=True)
    else:
        n_layers = streamer.materialize_layers()
        tr._krea2_streamer = streamer
        print(f'[krea2-native] direct DiT load: {n_globals} globals + {n_layers} block tensors to {device}', flush=True)
    return tr.eval()

def _krea2_te_ds_groups(index, expected: set[str], group_bytes: int=512 << 20):
    if group_bytes <= 0:
        raise ValueError('Krea2 text-encoder DS group budget must be positive')
    keys = set(index.refs)
    groups, current, size = ({}, [], 0)
    unexpected = []
    for name in sorted(keys):
        if name.endswith(('.comfy_quant', '.weight_scale')):
            continue
        target = name[len('model.'):] if name.startswith('model.') else name
        if target not in expected:
            unexpected.append(name)
            continue
        names = [name]
        if name.endswith('.weight'):
            names.extend((k for k in (name + '_scale', name[:-len('.weight')] + '.comfy_quant') if k in keys))
        refs = [index.refs[k] for k in names]
        for ref in refs:
            if ref.nbytes < 0:
                raise ValueError(f'Invalid Krea2 text-encoder tensor size: {ref.name}')
            if ref.name.endswith('.comfy_quant') and ref.nbytes > 65536:
                raise ValueError(f'Krea2 quant metadata exceeds 64 KiB: {ref.name}')
        bundle_bytes = sum((ref.nbytes for ref in refs))
        if current and size + bundle_bytes > group_bytes:
            groups[len(groups)] = current
            current, size = ([], 0)
        current.extend(refs)
        size += bundle_bytes
    if current:
        groups[len(groups)] = current
    return (groups, keys, unexpected)

def _bind_krea2_te_views(te, views, keys: set[str], dtype, device: str):
    from accelerate.utils import set_module_tensor_to_device

    class GroupReader:

        def get_tensor(self, name):
            return views[name]
    reader = GroupReader()
    loaded = set()
    for src_key in sorted(views):
        if src_key.endswith(('.comfy_quant', '.weight_scale')):
            continue
        dst_key = src_key[len('model.'):] if src_key.startswith('model.') else src_key
        value = _read_source_tensor(reader, keys, src_key, dtype, device).clone()
        assign_dtype = dtype if value.is_floating_point() else None
        set_module_tensor_to_device(te, dst_key, device, value=value, dtype=assign_dtype)
        loaded.add(dst_key)
        del value
    return loaded

def _load_krea2_component_ds(te, te_path: str, dtype, device: str, expected: set[str], *, label: str):
    import torch
    repo = os.environ.get('STUDIO_COSMOS_WINDOWS_PATH', '')
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    index = build_index(Path(te_path))
    groups, keys, unexpected = _krea2_te_ds_groups(index, expected)
    selected = {name[len('model.'):] if name.startswith('model.') else name for name in keys if not name.endswith(('.comfy_quant', '.weight_scale'))}
    missing = sorted(expected - selected)
    if missing or unexpected:
        raise RuntimeError(f'Krea2 text-encoder load mismatch: missing={missing[:4]} unexpected={unexpected[:4]}')
    if not groups:
        raise RuntimeError('Krea2 text encoder has no loadable tensors')
    streamer = DirectStorageStreamer(index, component='', layer_groups=groups, n_gpu_buffers=1, host_staging_buffers=1, device=device, record_compute_done=True)
    loaded = set()
    try:
        if streamer.mode != 3:
            raise RuntimeError(f'Krea2 text encoder requires DirectStorage mode 3, got {streamer.mode}')
        largest = max((sum((ref.nbytes for ref in refs)) for refs in groups.values()))
        print(f'[krea2-native] {label} DirectStorage mode=3, groups={len(groups)}, largest_group_payload_bytes={largest}, ring_capacity_bytes={streamer.cap}', flush=True)
        streamer.new_pass()
        requested = torch.device(device)
        for group_id in sorted(groups):
            views = streamer.wait_layer(group_id)
            if set(views) != {ref.name for ref in groups[group_id]}:
                raise RuntimeError('Krea2 text-encoder DirectStorage group coverage mismatch')
            for name, value in views.items():
                if value.device.type != 'cuda' or (requested.index is not None and value.device.index != requested.index):
                    raise RuntimeError(f'Krea2 text-encoder tensor on wrong device: {name}: {value.device}')
            loaded.update(_bind_krea2_te_views(te, views, keys, dtype, device))
            torch.cuda.synchronize(device)
            streamer.release_layer(group_id)
            print(f'[krea2-native] {label} DirectStorage: {group_id + 1}/{len(groups)} groups loaded', flush=True)
    finally:
        streamer.close()
    return loaded

def _load_krea2_text_encoder_ds(te, te_path: str, dtype, device: str, expected: set[str]):
    return _load_krea2_component_ds(te, te_path, dtype, device, expected, label='Qwen3-VL')

def _load_text_encoder(te_path: str, dtype, *, device: str='cpu'):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Qwen3VLConfig, Qwen3VLModel
    cfg_dir = _ensure_qwen3vl_assets()
    cfg = Qwen3VLConfig.from_pretrained(cfg_dir)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights():
            te = Qwen3VLModel(cfg)
    finally:
        torch.set_default_dtype(old)
    expected = set(te.state_dict().keys())
    loaded: set[str] = set()
    unexpected: list[str] = []
    if sys.platform == 'win32' and str(device).split(':', 1)[0].lower() == 'cuda':
        loaded = _load_krea2_text_encoder_ds(te, te_path, dtype, device, expected)
    else:
        with safe_open(te_path, framework='pt', device='cpu') as sf:
            keys = set(sf.keys())
            for src_key in sorted(keys):
                if src_key.endswith('.comfy_quant') or src_key.endswith('.weight_scale'):
                    continue
                dst_key = src_key[len('model.'):] if src_key.startswith('model.') else src_key
                if dst_key not in expected:
                    unexpected.append(src_key)
                    continue
                value = _read_source_tensor(sf, keys, src_key, dtype, device)
                assign_dtype = dtype if value.is_floating_point() else None
                set_module_tensor_to_device(te, dst_key, device, value=value, dtype=assign_dtype)
                loaded.add(dst_key)
                del value
    missing = sorted(expected - loaded)
    if missing or unexpected:
        raise RuntimeError(f'Krea2 text-encoder load mismatch: missing={missing[:4]} unexpected={unexpected[:4]}')
    te.to(device=device)
    print(f'[krea2-native] direct Qwen3-VL load: {len(loaded)}/{len(expected)} tensors to {device}', flush=True)
    return (te.eval(), cfg_dir)

def _load_vae(dtype, *, device: str='cpu'):
    from diffusers import AutoencoderKLQwenImage
    if sys.platform != 'win32' or str(device).split(':', 1)[0].lower() != 'cuda':
        return AutoencoderKLQwenImage.from_pretrained(QWEN_IMAGE_REPO, subfolder='vae', torch_dtype=dtype).eval()
    from accelerate import init_empty_weights
    config = AutoencoderKLQwenImage.load_config(QWEN_IMAGE_REPO, subfolder='vae')
    local = Path(QWEN_IMAGE_REPO)
    if local.is_dir():
        checkpoint = local / 'vae' / 'diffusion_pytorch_model.safetensors'
    else:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(QWEN_IMAGE_REPO, filename='vae/diffusion_pytorch_model.safetensors'))
    if not checkpoint.is_file():
        raise RuntimeError(f'Krea2 VAE safetensors not found: {checkpoint}')
    with init_empty_weights():
        vae = AutoencoderKLQwenImage.from_config(config).to(dtype=dtype)
    expected = set(vae.state_dict())
    loaded = _load_krea2_component_ds(vae, str(checkpoint), dtype, device, expected, label='Qwen-Image VAE')
    if loaded != expected:
        raise RuntimeError(f'Krea2 VAE load mismatch: missing={sorted(expected - loaded)[:4]}')
    for name, buffer in list(vae.named_buffers()):
        if buffer.is_meta:
            raise RuntimeError(f'Krea2 VAE unresolved meta buffer: {name}')
        parent_path, _, leaf = name.rpartition('.')
        parent = vae.get_submodule(parent_path) if parent_path else vae
        parent._buffers[leaf] = buffer.to(device=device)
    return vae.eval()

def _configure_krea2_regional_compile(transformer, device: str) -> bool:
    enabled = str(os.environ.get('STUDIO_KREA2_MPS_COMPILE', '1')).strip().lower() in {'1', 'true', 'yes', 'on'}
    if not enabled or str(device).split(':', 1)[0].lower() != 'mps':
        return False
    compile_repeated = getattr(transformer, 'compile_repeated_blocks', None)
    if not callable(compile_repeated):
        print('[krea2-native] regional compile unavailable on this diffusers build', flush=True)
        return False
    try:
        compile_repeated(fullgraph=True, dynamic=True)
        print('[krea2-native] MPS regional compile armed (fullgraph, dynamic)', flush=True)
        return True
    except Exception as exc:
        print(f'[krea2-native] MPS regional compile disabled after setup failure: {exc}', flush=True)
        return False

def _dit_is_bf16(path: str) -> bool:
    try:
        keys = _safetensors_keys(path)
    except Exception:
        return False
    return not any((k.endswith('.weight_scale') or k.endswith('.comfy_quant') for k in keys))

def build_krea2_pipeline(device: str='mps', dtype=None, *, distilled: bool=True, offload: str='none', dit_path: str | None=None):
    import torch
    _ensure_krea2_classes()
    from diffusers import Krea2Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer
    if dtype is None:
        dtype = _krea2_resident_dtype()
    files = krea2_files()
    dit = str(dit_path).strip() if dit_path else files['dit']
    if not Path(dit).is_file():
        raise RuntimeError(f'Krea2 DiT not found: {dit!r} (set STUDIO_KREA2_DIT / STUDIO_COMFYUI_MODELS)')
    if not Path(files['text_encoder']).is_file():
        raise RuntimeError(f"Krea2 text encoder not found: {files['text_encoder']!r}")
    raw_need = 4.0
    try:
        raw_need = float(os.environ.get('STUDIO_KREA2_DIRECT_MIN_FREE_GB', raw_need))
    except (TypeError, ValueError):
        pass
    _require_free_memory(raw_need)
    stream_layers = offload in ('model', 'stream', 'streaming', 'stream_layers', 'streaming_lazy')
    int8_resident = offload in ('int8_resident', 'resident_int8')
    transformer = _load_transformer(dit, dtype, device=device, stream_layers=stream_layers, int8_resident=int8_resident)
    text_encoder, cfg_dir = _load_text_encoder(files['text_encoder'], dtype, device=device)
    vae = _load_vae(dtype, device=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg_dir)
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipe = Krea2Pipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, is_distilled=distilled)
    pipe.vae.to(device=device)
    if not stream_layers and (not int8_resident):
        pipe.transformer.to(device=device)
    pipe._krea2_regional_compile = False
    pipe._krea2_resident_dtype = str(dtype).replace('torch.', '')
    if not stream_layers and (not int8_resident):
        pipe._krea2_regional_compile = _configure_krea2_regional_compile(pipe.transformer, device)
    for _fn in ('enable_tiling', 'enable_slicing'):
        try:
            getattr(pipe.vae, _fn)()
        except Exception:
            pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    if int8_resident:
        pipe._krea2_int8_resident = True
        pipe._int8_resident_device = device
    return pipe

def _safetensors_keys(path: str) -> list:
    from safetensors import safe_open
    with safe_open(path, framework='pt') as f:
        return list(f.keys())

def _egg_diff_target(egg_base: str, suffix: str) -> str:
    import torch
    return next(iter(map_dit_to_diffusers({egg_base + suffix: torch.zeros(1)})))

def _krea2_lora_target(base: str, expected):
    egg = _normalize_dit_source_key(base)
    native = egg[len('transformer.'):] if egg.startswith('transformer.') else egg
    if native + '.weight' in expected:
        return (native + '.weight', None)
    return _map_dit_key(egg + '.weight')

def _apply_diff_patch(pipe, sd: dict, weight: float) -> list:
    import torch
    params = dict(pipe.transformer.named_parameters())
    records = []
    for k, delta in sd.items():
        base = k[len('diffusion_model.'):] if k.startswith('diffusion_model.') else k
        if base.endswith('.diff_b'):
            egg, suffix = (base[:-len('.diff_b')], '.bias')
        elif base.endswith('.diff'):
            egg, suffix = (base[:-len('.diff')], '.weight')
        else:
            continue
        name = _egg_diff_target(egg, suffix)
        p = params.get(name)
        if p is None:
            raise RuntimeError(f'diff-patch target {name!r} (from {k!r}) not found in the Krea2 transformer')
        dl = (delta.reshape(p.shape).to(torch.float32) * float(weight)).to(p.device, p.dtype)
        with torch.no_grad():
            p.data.add_(dl)
        records.append((name, dl))
    return records

def _apply_comfy_lora_patch(pipe, sd: dict, weight: float) -> list:
    import torch
    params = dict(pipe.transformer.named_parameters())
    records = []
    down_suffix = '.lora_down.weight'
    up_suffix = '.lora_up.weight'
    down_keys = sorted((k for k in sd if k.endswith(down_suffix)))
    for down_key in down_keys:
        up_key = down_key[:-len(down_suffix)] + up_suffix
        if up_key not in sd:
            raise RuntimeError(f'missing LoRA up tensor for {down_key!r}')
        base = down_key[:-len(down_suffix)]
        egg = _normalize_dit_source_key(base)
        target = _egg_diff_target(egg, '.weight')
        p = params.get(target)
        if p is None:
            raise RuntimeError(f'LoRA target {target!r} (from {down_key!r}) not found in the Krea2 transformer')
        down = sd[down_key].to(torch.float32)
        up = sd[up_key].to(torch.float32)
        delta = torch.matmul(up, down).reshape(p.shape)
        dl = (delta * float(weight)).to(p.device, p.dtype)
        with torch.no_grad():
            p.data.add_(dl)
        records.append((target, dl))
    diff_sd = {k: v for k, v in sd.items() if k.endswith('.diff') or k.endswith('.diff_b')}
    if diff_sd:
        records += _apply_diff_patch(pipe, diff_sd, weight)
    return records

def _krea2_clear_streamed_loras(transformer) -> None:
    for module in list(getattr(transformer, '_krea2_streamed_lora_modules', None) or []):
        try:
            base_forward = getattr(module, '_krea2_base_forward', None)
            if base_forward is not None:
                module.forward = base_forward
                delattr(module, '_krea2_base_forward')
            for attr in ('_krea2_lora_adapters', '_krea2_linear_patches', '_krea2_lokr_adapters'):
                if hasattr(module, attr):
                    delattr(module, attr)
        except Exception:
            pass
    try:
        transformer._krea2_streamed_lora_modules = []
    except Exception:
        pass

def _krea2_apply_streamed_loras(transformer, loras, device, dtype) -> list:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from .krea2_lokr import full_lokr_bases, load_full_lokr, full_lokr_linear
    _krea2_clear_streamed_loras(transformer)
    applied: list[dict] = []
    if not loras:
        return applied
    sd_keys = set(transformer.state_dict().keys())
    wrapped: list = []
    transformer._krea2_streamed_lora_modules = wrapped

    def _wrap_linear(module):
        if hasattr(module, '_krea2_base_forward'):
            return
        wrapped.append(module)
        module._krea2_base_forward = module.forward

        def _forward(input, _module=module):
            out = _module._krea2_base_forward(input)
            for down, up, strength in getattr(_module, '_krea2_lora_adapters', []):
                down = down.to(input.device, input.dtype)
                up = up.to(input.device, input.dtype)
                out = out + F.linear(F.linear(input, down), up) * strength
            for w1, w2, strength in getattr(_module, '_krea2_lokr_adapters', []):
                out = out + full_lokr_linear(input, w1, w2) * strength
            for leaf, delta, strength in getattr(_module, '_krea2_linear_patches', []):
                delta = delta.to(input.device, input.dtype)
                change = F.linear(input, delta) if leaf == 'weight' else delta
                out = out + change * strength
            return out
        module.forward = _forward
    try:
        for lora in loras:
            d = lora if isinstance(lora, dict) else lora.model_dump() if hasattr(lora, 'model_dump') else {}
            path = str(d.get('path') or '').strip()
            if not path:
                continue
            if not Path(path).is_file():
                raise RuntimeError(f'LoRA file not found: {path!r}')
            weight = float(d.get('weight', 1.0) if d.get('weight') is not None else 1.0)
            raw_name = str(d.get('name') or '').strip() or Path(path).stem
            with safe_open(path, framework='pt', device='cpu') as handle:
                keys = set(handle.keys())
                down_suffix = '.lora_down.weight'
                up_suffix = '.lora_up.weight'
                down_keys = sorted((k for k in keys if k.endswith(down_suffix)))
                a_suffix = '.lora_A.weight'
                b_suffix = '.lora_B.weight'
                a_keys = sorted((k for k in keys if k.endswith(a_suffix)))
                diff_keys = sorted((k for k in keys if k.endswith('.diff') or k.endswith('.diff_b')))
                lokr_bases = full_lokr_bases(keys)
                if not down_keys and (not a_keys) and (not diff_keys) and (not lokr_bases):
                    raise RuntimeError(f'streamed Krea2 supports Comfy down/up, AI-Toolkit A/B, full-factor Linear LoKr and Linear diff patches; unsupported adapter {path!r}')
                pairs = [(key, key[:-len(down_suffix)] + up_suffix, down_suffix) for key in down_keys] + [(key, key[:-len(a_suffix)] + b_suffix, a_suffix) for key in a_keys]
                matched = 0
                for down_key, up_key, pair_suffix in pairs:
                    if up_key not in keys:
                        raise RuntimeError(f'missing LoRA up tensor for {down_key!r}')
                    base = down_key[:-len(pair_suffix)]
                    mapped = _krea2_lora_target(base, sd_keys)
                    if mapped is None:
                        continue
                    target, _reshape = mapped
                    if target not in sd_keys:
                        raise RuntimeError(f'LoRA target {target!r} (from {down_key!r}) is not in streamed Krea2')
                    module_path, leaf = target.rsplit('.', 1)
                    if leaf != 'weight':
                        raise RuntimeError(f'streamed Krea2 LoRA target must be a weight tensor, got {target!r}')
                    module = transformer.get_submodule(module_path)
                    if not isinstance(module, torch.nn.Linear):
                        raise RuntimeError(f'streamed Krea2 LoRA target {module_path!r} is {type(module).__name__}, not Linear')
                    down = handle.get_tensor(down_key).to(device=device, dtype=dtype)
                    up = handle.get_tensor(up_key).to(device=device, dtype=dtype)
                    if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or (down.shape[1] != module.in_features) or (up.shape[0] != module.out_features):
                        raise RuntimeError(f'LoRA tensor shapes {tuple(down.shape)}/{tuple(up.shape)} do not match {target!r} ({module.out_features}, {module.in_features})')
                    matched += 1
                    _wrap_linear(module)
                    adapters = getattr(module, '_krea2_lora_adapters', [])
                    adapters.append((down, up, float(weight)))
                    module._krea2_lora_adapters = adapters
                for base in lokr_bases:
                    try:
                        mapped = _map_dit_key(_normalize_dit_source_key(base) + '.weight')
                    except KeyError as exc:
                        raise RuntimeError(f'LoKr target {base!r} is not in streamed Krea2') from exc
                    if mapped is None or mapped[0] not in sd_keys:
                        raise RuntimeError(f'LoKr target {base!r} is not in streamed Krea2')
                    target, _reshape = mapped
                    module_path, leaf = target.rsplit('.', 1)
                    module = transformer.get_submodule(module_path)
                    if leaf != 'weight' or not isinstance(module, torch.nn.Linear):
                        raise RuntimeError(f'streamed Krea2 LoKr target {target!r} must be Linear')
                    w1, w2 = load_full_lokr(handle, base, module.in_features, module.out_features, device, dtype)
                    _wrap_linear(module)
                    factors = getattr(module, '_krea2_lokr_adapters', [])
                    factors.append((w1, w2, weight))
                    module._krea2_lokr_adapters = factors
                    matched += 1
                for diff_key in diff_keys:
                    suffix = '.diff_b' if diff_key.endswith('.diff_b') else '.diff'
                    leaf = 'bias' if suffix == '.diff_b' else 'weight'
                    egg = _normalize_dit_source_key(diff_key[:-len(suffix)])
                    mapped = _map_dit_key(egg + '.' + leaf)
                    if mapped is None:
                        continue
                    target, reshape_kind = mapped
                    if target not in sd_keys:
                        raise RuntimeError(f'diff-patch target {target!r} (from {diff_key!r}) is not in streamed Krea2')
                    module_path, mapped_leaf = target.rsplit('.', 1)
                    module = transformer.get_submodule(module_path)
                    if not isinstance(module, torch.nn.Linear) or mapped_leaf != leaf:
                        raise RuntimeError(f'streamed Krea2 diff-patch target {target!r} must be a Linear weight/bias')
                    delta = _reshape_mapped_value(handle.get_tensor(diff_key), reshape_kind)
                    want = tuple(getattr(module, leaf).shape)
                    if tuple(delta.shape) != want:
                        raise RuntimeError(f'diff-patch shape {tuple(delta.shape)} does not match {target!r} {want}')
                    delta = delta.to(device=device, dtype=dtype)
                    _wrap_linear(module)
                    patches = getattr(module, '_krea2_linear_patches', [])
                    patches.append((leaf, delta, weight))
                    module._krea2_linear_patches = patches
                    matched += 1
                if not matched:
                    raise RuntimeError(f'No supported linear LoRA targets found in {path!r}')
            kind = 'streamed_ai_toolkit_lora' if a_keys and (not down_keys) else 'streamed_comfy_lora' if down_keys else 'streamed_lokr' if lokr_bases else 'streamed_diff_patch'
            applied.append({'name': raw_name, 'weight': weight, 'path': path, 'kind': kind})
    except Exception:
        _krea2_clear_streamed_loras(transformer)
        raise
    transformer._krea2_streamed_lora_modules = wrapped
    return applied

def apply_loras(pipe, loras) -> list:
    import re
    from safetensors.torch import load_file
    transformer = getattr(pipe, 'transformer', None)
    streamer = getattr(transformer, '__dict__', {}).get('_krea2_streamer') if transformer is not None else None
    if streamer is not None:
        if hasattr(streamer, 'set_loras'):
            return streamer.set_loras(loras)
        dev = getattr(streamer, 'device', None) or getattr(transformer, 'device', None) or 'cuda'
        if str(dev) == 'meta':
            dev = 'cuda'
        dt = getattr(streamer, 'dtype', None) or getattr(transformer, 'dtype', None)
        if dt is None:
            import torch as _torch
            dt = _torch.bfloat16
        return _krea2_apply_streamed_loras(transformer, loras, dev, dt)
    _krea2_clear_streamed_loras(transformer)
    for name, delta in getattr(pipe, '_krea2_diff_patches', None) or []:
        p = dict(pipe.transformer.named_parameters()).get(name)
        if p is not None:
            import torch
            with torch.no_grad():
                p.data.add_(delta.to(p.device, p.dtype), alpha=-1.0)
    pipe._krea2_diff_patches = []
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    names: list[str] = []
    weights: list[float] = []
    applied: list[dict] = []
    diff_records: list = []
    used: set[str] = set()
    for i, lora in enumerate(loras or []):
        d = lora if isinstance(lora, dict) else lora.model_dump() if hasattr(lora, 'model_dump') else {}
        path = str(d.get('path') or '').strip()
        if not path:
            continue
        if not Path(path).is_file():
            raise RuntimeError(f'LoRA file not found: {path!r}')
        weight = float(d.get('weight', 1.0) or 1.0)
        raw = str(d.get('name') or '').strip() or Path(path).stem
        keys = _safetensors_keys(path)
        is_diff = any((k.endswith('.diff') or k.endswith('.diff_b') for k in keys)) and (not any(('lora' in k.lower() for k in keys)))
        is_comfy_lora = any((k.endswith('.lora_down.weight') or k.endswith('.lora_up.weight') for k in keys))
        if is_diff:
            try:
                diff_records += _apply_diff_patch(pipe, load_file(path), weight)
            except Exception as exc:
                raise RuntimeError(f'failed to apply Krea2 diff-patch {raw!r} ({path}): {exc}') from exc
            applied.append({'name': raw, 'weight': weight, 'path': path, 'kind': 'diff'})
            continue
        if is_comfy_lora:
            try:
                diff_records += _apply_comfy_lora_patch(pipe, load_file(path), weight)
            except Exception as exc:
                raise RuntimeError(f'failed to apply Krea2 Comfy LoRA {raw!r} ({path}): {exc}') from exc
            applied.append({'name': raw, 'weight': weight, 'path': path, 'kind': 'comfy_lora'})
            continue
        nm = re.sub('[^0-9A-Za-z_]+', '_', raw)[:48].strip('_') or f'lora{i}'
        base, n = (nm, 1)
        while nm in used:
            nm = f'{base}_{n}'
            n += 1
        used.add(nm)
        try:
            pipe.load_lora_weights(str(Path(path).parent), weight_name=Path(path).name, adapter_name=nm)
        except Exception as exc:
            raise RuntimeError(f'failed to load Krea2 LoRA {raw!r} ({path}): {exc}') from exc
        names.append(nm)
        weights.append(weight)
        applied.append({'name': nm, 'weight': weight, 'path': path, 'kind': 'lora'})
    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except Exception as exc:
            raise RuntimeError(f'failed to activate Krea2 LoRA stack {names}: {exc}') from exc
    pipe._krea2_diff_patches = diff_records
    return applied

def apply_two_pass_loras(pipe, loras) -> list:
    loras = [spec.model_dump() if hasattr(spec, 'model_dump') else spec for spec in loras or []]
    loras = [spec for spec in loras if float(spec.get('weight', 1.0)) != 0]
    transformer = pipe.transformer
    if getattr(transformer, '__dict__', {}).get('_krea2_streamer') is not None:
        return apply_loras(pipe, loras)
    apply_loras(pipe, [])
    parameter = next(transformer.parameters())
    return _krea2_apply_streamed_loras(transformer, loras, parameter.device, parameter.dtype)

def release_krea2_adapters(pipe) -> None:
    transformer = getattr(pipe, 'transformer', None)
    streamer = getattr(transformer, '__dict__', {}).get('_krea2_streamer') if transformer is not None else None
    set_loras = getattr(streamer, 'set_loras', None)
    if callable(set_loras):
        try:
            set_loras([])
        except Exception as exc:
            print(f'[krea2] streamed adapter cleanup warning: {exc}', flush=True)
    if streamer is not None:
        for module in list(getattr(streamer, '_lora_modules', None) or []):
            try:
                base_forward = getattr(module, '_krea2_base_forward', None)
                if base_forward is not None:
                    module.forward = base_forward
                    delattr(module, '_krea2_base_forward')
                if hasattr(module, '_krea2_lora_adapters'):
                    delattr(module, '_krea2_lora_adapters')
            except Exception:
                pass
        try:
            streamer._lora_modules = []
            streamer._lora_by_target = {}
        except Exception:
            pass
        close = getattr(streamer, 'close', None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                print(f'[krea2] streamer close warning: {exc}', flush=True)
    if transformer is not None:
        _krea2_clear_streamed_loras(transformer)
    try:
        pipe._krea2_diff_patches = []
    except Exception:
        pass
    unload = getattr(pipe, 'unload_lora_weights', None)
    if callable(unload):
        try:
            unload()
        except Exception as exc:
            print(f'[krea2] PEFT adapter cleanup warning: {exc}', flush=True)
    modules = getattr(transformer, 'modules', None)
    regions = modules() if callable(modules) else [transformer] if transformer is not None else []
    detached = 0
    for module in regions:
        if getattr(module, '_compiled_call_impl', None) is not None:
            module._compiled_call_impl = None
            detached += 1
    if detached:
        print(f'[krea2] detached {detached} compiled region(s) for model handoff', flush=True)

def _krea2_free_te(pipe):
    import gc
    import torch
    pipe.text_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _krea2_ensure_te(pipe):
    if getattr(pipe, 'text_encoder', None) is not None:
        return
    device = getattr(pipe, '_int8_resident_device', None) or 'cuda:0'
    dtype = pipe.transformer.dtype
    te, _ = _load_text_encoder(krea2_files()['text_encoder'], dtype, device=device)
    pipe.text_encoder = te

def _arm_te_free_hook(pipe):
    state = {'fired': False, 'handle': None}

    def pre_hook(_m, _a):
        if not state['fired']:
            state['fired'] = True
            _krea2_free_te(pipe)
            if state['handle'] is not None:
                state['handle'].remove()
    state['handle'] = pipe.transformer.register_forward_pre_hook(pre_hook)
    return state['handle']

def _synchronize_krea2_device(pipe) -> None:
    try:
        import torch
        execution_device = getattr(pipe, '_execution_device', None)
        device_type = str(getattr(execution_device, 'type', execution_device or '')).split(':', 1)[0]
        if device_type == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

def _report_krea2_stage(on_stage, stage: str, timings: dict) -> None:
    if not on_stage:
        return
    public = {k: v for k, v in timings.items() if not str(k).startswith('_')}
    try:
        on_stage(stage, public)
    except Exception:
        pass

def _compact_krea2_prompt_conditioning(prompt_embeds, prompt_embeds_mask):
    try:
        if prompt_embeds.ndim < 2 or prompt_embeds_mask.ndim != 2:
            return (prompt_embeds, prompt_embeds_mask)
        batch, sequence = prompt_embeds.shape[:2]
        if prompt_embeds_mask.shape != (batch, sequence):
            return (prompt_embeds, prompt_embeds_mask)
        mask = prompt_embeds_mask.bool()
        counts = [int(value) for value in mask.sum(dim=1).detach().cpu().tolist()]
        longest = max(counts, default=0)
        if longest <= 0 or longest >= sequence:
            return (prompt_embeds, prompt_embeds_mask)
        rows = []
        compact_mask = mask.new_zeros((batch, longest))
        for row_index, valid_count in enumerate(counts):
            indices = mask[row_index].nonzero(as_tuple=False).flatten()
            row = prompt_embeds[row_index].index_select(0, indices)
            if valid_count < longest:
                padding = prompt_embeds.new_zeros((longest - valid_count, *prompt_embeds.shape[2:]))
                row = __import__('torch').cat((row, padding), dim=0)
            rows.append(row)
            compact_mask[row_index, :valid_count] = True
        output_mask = None if all((valid_count == longest for valid_count in counts)) else compact_mask
        return (__import__('torch').stack(rows, dim=0), output_mask)
    except Exception:
        return (prompt_embeds, prompt_embeds_mask)

def _align_krea2_cfg_conditioning(positive, positive_mask, negative, negative_mask):
    import torch
    target = max(int(positive.shape[1]), int(negative.shape[1]))

    def pad(embeds, mask):
        sequence = int(embeds.shape[1])
        if mask is None:
            mask = torch.ones((int(embeds.shape[0]), sequence), device=embeds.device, dtype=torch.bool)
        if sequence < target:
            embedding_padding = embeds.new_zeros((int(embeds.shape[0]), target - sequence, *embeds.shape[2:]))
            embeds = torch.cat((embeds, embedding_padding), dim=1)
            mask_padding = mask.new_zeros((int(mask.shape[0]), target - sequence))
            mask = torch.cat((mask, mask_padding), dim=1)
        return (embeds, mask)
    positive, positive_mask = pad(positive, positive_mask)
    negative, negative_mask = pad(negative, negative_mask)
    return (positive, positive_mask, negative, negative_mask)

@contextmanager
def _krea2_stage_timer(pipe, timings: dict | None, on_stage=None, compact_prompt_conditioning: bool=True):
    if timings is None:
        yield
        return
    started = time.perf_counter()
    original_encode = pipe.encode_prompt
    original_decode = pipe.vae.decode
    timings.setdefault('prompt_encode_seconds', 0.0)
    timings.setdefault('denoise_seconds', 0.0)
    timings.setdefault('vae_decode_seconds', 0.0)
    timings.setdefault('vae_decode_calls', 0)
    timings['_denoise_started_at'] = started
    encode_state = {'prompt_metrics_recorded': False}

    def timed_encode(*args, **kwargs):
        _report_krea2_stage(on_stage, 'prompt_encode_start', timings)
        _synchronize_krea2_device(pipe)
        t0 = time.perf_counter()
        result = original_encode(*args, **kwargs)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            before = getattr(result[0], 'shape', (None, None))[1]
            if compact_prompt_conditioning:
                compact_embeds, compact_mask = _compact_krea2_prompt_conditioning(result[0], result[1])
            else:
                compact_embeds, compact_mask = (result[0], result[1])
            result = type(result)((compact_embeds, compact_mask, *result[2:]))
            after = getattr(compact_embeds, 'shape', (None, None))[1]
            if not encode_state['prompt_metrics_recorded'] and isinstance(before, int) and isinstance(after, int):
                timings['prompt_sequence_tokens'] = after
                timings['prompt_padding_tokens_removed'] = max(0, before - after)
                encode_state['prompt_metrics_recorded'] = True
        _synchronize_krea2_device(pipe)
        timings['prompt_encode_seconds'] += time.perf_counter() - t0
        timings['_denoise_started_at'] = time.perf_counter()
        _report_krea2_stage(on_stage, 'prompt_encode_done', timings)
        return result

    def timed_decode(*args, **kwargs):
        _synchronize_krea2_device(pipe)
        decode_started = time.perf_counter()
        denoise_started = timings.get('_denoise_started_at')
        if denoise_started is not None:
            timings['denoise_seconds'] += max(0.0, decode_started - float(denoise_started))
            timings['_denoise_started_at'] = None
        _report_krea2_stage(on_stage, 'vae_decode_start', timings)
        result = original_decode(*args, **kwargs)
        _synchronize_krea2_device(pipe)
        timings['vae_decode_seconds'] += max(0.0, time.perf_counter() - decode_started)
        timings['vae_decode_calls'] += 1
        _report_krea2_stage(on_stage, 'vae_decode_done', timings)
        return result
    pipe.encode_prompt = timed_encode
    pipe.vae.decode = timed_decode
    try:
        yield
    finally:
        _synchronize_krea2_device(pipe)
        pipeline_seconds = max(0.0, time.perf_counter() - started)
        pipe.encode_prompt = original_encode
        pipe.vae.decode = original_decode
        timings.pop('_denoise_started_at', None)
        timings['pipeline_seconds'] = pipeline_seconds
        measured = float(timings.get('prompt_encode_seconds') or 0.0) + float(timings.get('denoise_seconds') or 0.0) + float(timings.get('vae_decode_seconds') or 0.0)
        timings['pipeline_other_seconds'] = max(0.0, pipeline_seconds - measured)

def _call_krea2_step(on_step, step: int, total: int, meta: dict) -> None:
    if not on_step:
        return
    try:
        import inspect
        sig = inspect.signature(on_step)
        params = list(sig.parameters.values())
        rich = any((p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)) or len(params) >= 4
    except Exception:
        rich = False
    try:
        if rich:
            on_step(int(step), int(total), None, meta)
        else:
            on_step(int(step), int(total))
    except Exception:
        pass

def _krea2_step_progress(*, label: str, completed: int, total: int, now: float, denoise_started: float | None, previous_step_at: float | None) -> tuple[dict, float]:
    now = float(now)
    completed = max(0, int(completed))
    total = max(0, int(total))
    if denoise_started is not None:
        started = float(denoise_started)
    elif previous_step_at is not None:
        started = float(previous_step_at)
    else:
        started = now
    previous = float(previous_step_at) if previous_step_at is not None else started
    elapsed = max(0.0, now - started)
    step_seconds = max(0.0, now - previous)
    average = elapsed / max(1, completed)
    return ({'text': str(label), 'step_seconds': round(step_seconds, 3), 'elapsed_seconds': round(elapsed, 3), 's_iter': round(average, 3), 'eta': round(average * max(0, total - completed), 3)}, now)

@contextmanager
def _krea2_safe_mps_vae_tiling(pipe, width: int, height: int):
    vae = pipe.vae
    execution_device = getattr(pipe, '_execution_device', None)
    device_type = str(getattr(execution_device, 'type', execution_device or '')).split(':', 1)[0]
    enabled = device_type == 'mps' and max(int(width), int(height)) > 1024 and callable(getattr(vae, 'enable_tiling', None))
    tiling_attrs = ('use_tiling', 'tile_sample_min_height', 'tile_sample_min_width', 'tile_sample_stride_height', 'tile_sample_stride_width')
    prior = {name: getattr(vae, name) for name in tiling_attrs if hasattr(vae, name)}
    if enabled:
        vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024, tile_sample_stride_height=768, tile_sample_stride_width=768)
    try:
        yield enabled
    finally:
        if enabled:
            if prior.get('use_tiling'):
                vae.enable_tiling(tile_sample_min_height=prior.get('tile_sample_min_height'), tile_sample_min_width=prior.get('tile_sample_min_width'), tile_sample_stride_height=prior.get('tile_sample_stride_height'), tile_sample_stride_width=prior.get('tile_sample_stride_width'))
            else:
                disable_tiling = getattr(vae, 'disable_tiling', None)
                if callable(disable_tiling):
                    disable_tiling()
                elif hasattr(vae, 'use_tiling'):
                    vae.use_tiling = False
            for name, value in prior.items():
                setattr(vae, name, value)

@contextmanager
def _krea2_sda_schedule(pipe, option):
    original = pipe.scheduler
    if option and getattr(option, 'krea_sda_schedule', False):
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(original.config, use_dynamic_shifting=True, time_shift_type='exponential')
    try:
        yield
    finally:
        pipe.scheduler = original

def generate_krea2(pipe, *, prompt: str, negative: str='', width: int=1024, height: int=1024, steps: int=8, guidance: float=0.0, seed: int=42, out_path: str, loras=None, on_step=None, should_cancel=None, stage_timings: dict | None=None, on_stage=None, two_pass=None, baseline_loras=None) -> str:
    import torch
    gated = bool(two_pass and two_pass.enabled)
    if not gated:
        apply_loras(pipe, loras)
    if getattr(pipe, '_krea2_int8_resident', False):
        _krea2_ensure_te(pipe)
        _arm_te_free_hook(pipe)
    gen = torch.Generator(device='cpu').manual_seed(int(seed))
    if should_cancel and should_cancel():
        raise RuntimeError('generation cancelled')
    cb = None
    if on_step:
        step_clock = {'last_at': None}
        callback_started = time.perf_counter()

        def cb(_pipe, step, _ts, cbkw):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            _synchronize_krea2_device(pipe)
            now = time.perf_counter()
            denoise_started = (stage_timings or {}).get('_denoise_started_at', callback_started)
            completed = int(step) + 1
            meta, step_clock['last_at'] = _krea2_step_progress(label='Krea 2 denoising', completed=completed, total=int(steps), now=now, denoise_started=denoise_started, previous_step_at=step_clock['last_at'])
            _call_krea2_step(on_step, completed, int(steps), meta)
            return cbkw
    kw = dict(prompt=prompt, num_inference_steps=int(steps), guidance_scale=float(guidance), width=int(width), height=int(height), generator=gen, callback_on_step_end=cb)
    if guidance and guidance > 0 and negative:
        kw['negative_prompt'] = negative
    with _krea2_sda_schedule(pipe, two_pass), _krea2_safe_mps_vae_tiling(pipe, width, height), _krea2_stage_timer(pipe, stage_timings, on_stage=on_stage, compact_prompt_conditioning=not bool(guidance and guidance > 0)):
        from .two_pass import run_pipeline
        apply_stack = apply_two_pass_loras if gated else apply_loras
        image = run_pipeline(pipe, kw, two_pass, lambda stack: apply_stack(pipe, stack), baseline=baseline_loras or [], first_pass=loras or []).images[0]
        sampling_receipt = {'steps': int(steps), 'scheduler': type(pipe.scheduler).__name__, 'mu': 1.15, 'sigmas': pipe.scheduler.sigmas.detach().cpu().tolist()} if two_pass and getattr(two_pass, 'krea_sda_schedule', False) else None
    save_started = time.perf_counter()
    image.save(str(out_path))
    if sampling_receipt:
        Path(str(out_path) + '.sampling.json').write_text(json.dumps(sampling_receipt, indent=2), encoding='utf-8')
    if two_pass and two_pass.enabled:
        Path(str(out_path) + '.two-pass.json').write_text(json.dumps(pipe._studio_two_pass_receipt, indent=2), encoding='utf-8')
    if stage_timings is not None:
        stage_timings['image_save_seconds'] = max(0.0, time.perf_counter() - save_started)
    return str(out_path)

def _krea2_center_crop(image, width: int, height: int):
    from PIL import Image
    image = image.convert('RGB')
    src_w, src_h = image.size
    target_ratio = float(width) / float(height)
    source_ratio = float(src_w) / float(src_h)
    if source_ratio > target_ratio:
        crop_w = max(1, round(src_h * target_ratio))
        left = (src_w - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, src_h))
    elif source_ratio < target_ratio:
        crop_h = max(1, round(src_w / target_ratio))
        top = (src_h - crop_h) // 2
        image = image.crop((0, top, src_w, top + crop_h))
    return image.resize((int(width), int(height)), Image.Resampling.LANCZOS)

def _krea2_limit_reference_area(image, max_pixels: int=2097152):
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (float(max_pixels) / float(width * height)) ** 0.5
    width = max(16, int(width * scale) // 16 * 16)
    height = max(16, int(height * scale) // 16 * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)

def _encode_krea2_source_latent(pipe, image, *, target_latent_height: int, target_latent_width: int):
    import torch
    import torch.nn.functional as F
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    src_w, src_h = image.size
    scale = int(pipe.vae_scale_factor)
    src_w = max(scale, int(src_w) // scale * scale)
    src_h = max(scale, int(src_h) // scale * scale)
    if image.size != (src_w, src_h):
        from PIL import Image
        image = image.resize((src_w, src_h), Image.Resampling.LANCZOS)
    pixels = pipe.image_processor.preprocess(image, height=src_h, width=src_w).to(device, pipe.vae.dtype)
    mean, std, channels = _vae_norm(pipe.vae, pipe.vae.dtype, device)
    encoded = pipe.vae.encode(pixels.unsqueeze(2))
    if hasattr(encoded, 'latent_dist'):
        distribution = encoded.latent_dist
        latent = distribution.mode() if callable(getattr(distribution, 'mode', None)) else distribution.sample()
    else:
        latent = encoded.latents
    latent = ((latent - mean) / std)[:, :, 0].to(dtype)
    if latent.shape[-2:] != (target_latent_height, target_latent_width):
        latent = F.interpolate(latent.float(), size=(target_latent_height, target_latent_width), mode='bilinear', align_corners=False).to(dtype)
    return pipe._pack_latents(latent, latent.shape[0], channels, target_latent_height, target_latent_width)

def _encode_krea2_grounded_instruction(pipe, prompt: str, images: list, grounding_px: int, *, recipe: str='identity'):
    import torch
    if not images:
        raise ValueError('Krea 2 Edit grounded encoding needs at least one source image')
    device = pipe._execution_device
    pixel_parts = []
    grid_parts = []
    grids: list[tuple[int, int, int]] = []
    for image in images:
        if recipe == 'ostris':
            pixels, grid = _prepare_krea2_grounding_image(_krea2_fit_area(image, OSTRIS_VLM_MAX_PIXELS), 0)
        else:
            pixels, grid = _prepare_krea2_grounding_image(image, grounding_px)
        pixel_parts.append(pixels)
        grid_parts.append(grid)
        grids.append(tuple((int(value) for value in grid[0].tolist())))
    text = (_krea2_ostris_prompt if recipe == 'ostris' else _krea2_grounded_prompt)(prompt, grids)
    tokens = pipe.tokenizer(text, return_tensors='pt')
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device).bool()
    pixels = torch.cat(pixel_parts, dim=0).to(device=device, dtype=pipe.text_encoder.dtype)
    image_grid_thw = torch.cat(grid_parts, dim=0).to(device=device)
    import inspect
    multimodal = {}
    if 'mm_token_type_ids' in inspect.signature(pipe.text_encoder.forward).parameters:
        multimodal['mm_token_type_ids'] = input_ids.eq(pipe.text_encoder.config.image_token_id).long()
    outputs = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixels, image_grid_thw=image_grid_thw, output_hidden_states=True, **multimodal)
    hidden_states = torch.stack([outputs.hidden_states[index] for index in pipe.text_encoder_select_layers], dim=2)
    start = int(pipe.prompt_template_encode_start_idx)
    return (hidden_states[:, start:], attention_mask[:, start:])

def generate_krea2_edit(pipe, *, prompt: str, image_path: str, reference_path: str | None=None, width: int=1024, height: int=1024, steps: int=8, guidance: float=0.0, seed: int=42, grounding_px: int=768, out_path: str, loras=None, on_step=None, should_cancel=None, stage_timings: dict | None=None, on_stage=None, recipe: str='identity') -> str:
    import types
    import torch
    from PIL import Image
    if not image_path or not Path(str(image_path)).is_file():
        raise RuntimeError(f'Krea 2 Edit source image not found: {str(image_path)!r}')
    if reference_path and (not Path(str(reference_path)).is_file()):
        raise RuntimeError(f'Krea 2 Edit second reference not found: {str(reference_path)!r}')
    multiple = int(pipe.vae_scale_factor) * int(pipe.patch_size)
    width = max(multiple, (int(width) + multiple - 1) // multiple * multiple)
    height = max(multiple, (int(height) + multiple - 1) // multiple * multiple)
    if width * height > 2097152:
        raise RuntimeError(f'Krea 2 Edit is limited to 2 MP for reliable source grounding; requested {width}x{height} ({width * height:,} pixels). Choose a smaller matching-aspect canvas.')
    if should_cancel and should_cancel():
        raise RuntimeError('generation cancelled')
    if recipe not in KREA2_EDIT_RECIPES:
        raise RuntimeError(f"Unknown Krea 2 edit recipe {recipe!r}; use one of {', '.join(KREA2_EDIT_RECIPES)}")
    ostris = recipe == 'ostris'
    apply_loras(pipe, loras)
    if getattr(pipe, '_krea2_int8_resident', False):
        _krea2_ensure_te(pipe)
    with Image.open(str(image_path)) as source_image:
        primary = _krea2_fit_area(source_image, OSTRIS_REF_MAX_PIXELS, OSTRIS_REF_SNAP) if ostris else _krea2_center_crop(source_image, width, height)
    sources = [primary]
    if reference_path:
        with Image.open(str(reference_path)) as reference_image:
            sources.append(_krea2_fit_area(reference_image, OSTRIS_REF_MAX_PIXELS, OSTRIS_REF_SNAP) if ostris else _krea2_limit_reference_area(reference_image))
    target_latent_height = height // int(pipe.vae_scale_factor)
    target_latent_width = width // int(pipe.vae_scale_factor)
    source_encode_started = time.perf_counter()
    _report_krea2_stage(on_stage, 'edit_source_encode_start', stage_timings or {})
    with torch.no_grad():
        if ostris:
            scale, patch = (int(pipe.vae_scale_factor), int(pipe.patch_size))
            ostris_refs = []
            for image in sources:
                lat_h, lat_w = (image.height // scale, image.width // scale)
                tokens = _encode_krea2_source_latent(pipe, image, target_latent_height=lat_h, target_latent_width=lat_w)
                ostris_refs.append((tokens, lat_h // patch, lat_w // patch))
            source_tokens = [tokens for tokens, _, _ in ostris_refs]
        else:
            source_tokens = [_encode_krea2_source_latent(pipe, image, target_latent_height=target_latent_height, target_latent_width=target_latent_width) for image in sources]
        _synchronize_krea2_device(pipe)
        if stage_timings is not None:
            stage_timings['source_vae_encode_seconds'] = max(0.0, time.perf_counter() - source_encode_started)
        _report_krea2_stage(on_stage, 'edit_source_encode_done', stage_timings or {})
        grounded_started = time.perf_counter()
        _report_krea2_stage(on_stage, 'prompt_encode_start', stage_timings or {})
        positive, positive_mask = _encode_krea2_grounded_instruction(pipe, prompt, sources, int(grounding_px), recipe=recipe)
        negative = negative_mask = None
        if guidance and guidance > 0:
            negative, negative_mask = _encode_krea2_grounded_instruction(pipe, '', sources, int(grounding_px), recipe=recipe)
            positive, positive_mask, negative, negative_mask = _align_krea2_cfg_conditioning(positive, positive_mask, negative, negative_mask)
        _synchronize_krea2_device(pipe)
        if stage_timings is not None:
            stage_timings['prompt_encode_seconds'] = max(0.0, time.perf_counter() - grounded_started)
            stage_timings['prompt_sequence_tokens'] = int(positive.shape[1])
            stage_timings['prompt_padding_tokens_removed'] = 0
        _report_krea2_stage(on_stage, 'prompt_encode_done', stage_timings or {})
    if getattr(pipe, '_krea2_int8_resident', False):
        _arm_te_free_hook(pipe)
    original_forward = pipe.transformer.forward

    def edit_forward(transformer, **kwargs):
        if ostris:
            return _krea2_ostris_edit_forward(transformer, ostris_refs, **kwargs)
        return _krea2_edit_transformer_forward(transformer, source_tokens, **kwargs)
    pipe.transformer.forward = types.MethodType(edit_forward, pipe.transformer)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    step_clock = {'last_at': None}
    callback_started = time.perf_counter()

    def callback(_pipe, step, _timestep, callback_kwargs):
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        if on_step:
            _synchronize_krea2_device(pipe)
            now = time.perf_counter()
            denoise_started = (stage_timings or {}).get('_denoise_started_at', callback_started)
            completed = int(step) + 1
            meta, step_clock['last_at'] = _krea2_step_progress(label='Krea 2 Edit denoising', completed=completed, total=int(steps), now=now, denoise_started=denoise_started, previous_step_at=step_clock['last_at'])
            _call_krea2_step(on_step, completed, int(steps), meta)
        return callback_kwargs
    call_kwargs = dict(prompt=None, prompt_embeds=positive, prompt_embeds_mask=positive_mask, height=height, width=width, num_inference_steps=int(steps), guidance_scale=float(guidance), generator=generator, callback_on_step_end=callback if on_step or should_cancel else None)
    if negative is not None:
        call_kwargs.update(negative_prompt_embeds=negative, negative_prompt_embeds_mask=negative_mask)
    try:
        with _krea2_safe_mps_vae_tiling(pipe, width, height), _krea2_stage_timer(pipe, stage_timings, on_stage=on_stage, compact_prompt_conditioning=negative is None):
            image = pipe(**call_kwargs).images[0]
    finally:
        pipe.transformer.forward = original_forward
    save_started = time.perf_counter()
    image.save(str(out_path))
    if stage_timings is not None:
        stage_timings['image_save_seconds'] = max(0.0, time.perf_counter() - save_started)
        stage_timings['krea2_edit'] = True
        stage_timings['krea2_edit_references'] = len(sources)
        stage_timings['krea2_edit_grounding_px'] = int(grounding_px)
        stage_timings['krea2_edit_recipe'] = recipe
    return str(out_path)

def _vae_norm(vae, dtype, device):
    import torch
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, dtype)
    return (mean, std, z)

def generate_krea2_batch(pipe, *, prompts: list[str], negative: str='', width: int=1024, height: int=1024, steps: int=8, guidance: float=0.0, seed: int=42, out_paths: list[str], loras=None, on_step=None, should_cancel=None, stage_timings: dict | None=None, on_stage=None) -> list[str]:
    import torch
    clean_prompts = [str(p or '').strip() for p in prompts]
    clean_prompts = [p for p in clean_prompts if p]
    if not clean_prompts:
        raise RuntimeError('Krea2 batch requires at least one non-empty prompt.')
    if len(out_paths) != len(clean_prompts):
        raise RuntimeError('Krea2 batch output path count must match prompt count.')
    apply_loras(pipe, loras)
    if getattr(pipe, '_krea2_int8_resident', False):
        _krea2_ensure_te(pipe)
        _arm_te_free_hook(pipe)
    generators = [torch.Generator(device='cpu').manual_seed(int(seed) + i) for i in range(len(clean_prompts))]
    if should_cancel and should_cancel():
        raise RuntimeError('generation cancelled')
    cb = None
    if on_step:
        step_clock = {'last_at': None}
        callback_started = time.perf_counter()

        def cb(_pipe, step, _ts, cbkw):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            _synchronize_krea2_device(pipe)
            now = time.perf_counter()
            denoise_started = (stage_timings or {}).get('_denoise_started_at', callback_started)
            completed = int(step) + 1
            meta, step_clock['last_at'] = _krea2_step_progress(label='Krea 2 batched denoising', completed=completed, total=int(steps), now=now, denoise_started=denoise_started, previous_step_at=step_clock['last_at'])
            _call_krea2_step(on_step, completed, int(steps), meta)
            return cbkw
    kw = dict(prompt=clean_prompts, num_inference_steps=int(steps), guidance_scale=float(guidance), width=int(width), height=int(height), generator=generators, callback_on_step_end=cb)
    if guidance and guidance > 0 and negative:
        kw['negative_prompt'] = [negative] * len(clean_prompts)
    vae = pipe.vae
    original_decode = vae.decode
    with _krea2_safe_mps_vae_tiling(pipe, width, height), _krea2_stage_timer(pipe, stage_timings, on_stage=on_stage, compact_prompt_conditioning=not bool(guidance and guidance > 0)):
        timed_decode = vae.decode

        def sequential_decode(latents, *decode_args, return_dict=True, **decode_kwargs):
            if int(latents.shape[0]) <= 1:
                return timed_decode(latents, *decode_args, return_dict=return_dict, **decode_kwargs)
            decoded_cpu = []
            for latent in latents.split(1):
                if should_cancel and should_cancel():
                    raise RuntimeError('generation cancelled')
                sample = timed_decode(latent, *decode_args, return_dict=False, **decode_kwargs)[0]
                decoded_cpu.append(sample.to('cpu'))
            combined = torch.cat(decoded_cpu, dim=0)
            if not return_dict:
                return (combined,)
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample=combined)
        vae.decode = sequential_decode
        try:
            images = pipe(**kw).images
        finally:
            vae.decode = timed_decode
    if len(images) != len(clean_prompts):
        raise RuntimeError(f'Krea2 batch returned {len(images)} images for {len(clean_prompts)} prompts.')
    save_started = time.perf_counter()
    saved: list[str] = []
    for image, out_path in zip(images, out_paths):
        image.save(str(out_path))
        saved.append(str(out_path))
    if stage_timings is not None:
        stage_timings['image_save_seconds'] = max(0.0, time.perf_counter() - save_started)
    return saved

def generate_krea2_i2i(pipe, **kwargs) -> str:
    import torch
    tile_size = int(kwargs.get('tile_size') or 0)
    tiling_attrs = ('use_tiling', 'tile_sample_min_height', 'tile_sample_min_width', 'tile_sample_stride_height', 'tile_sample_stride_width')
    prior = {key: getattr(pipe.vae, key) for key in tiling_attrs if hasattr(pipe.vae, key)} if tile_size else {}
    if tile_size:
        overlap = int(kwargs.get('tile_overlap', 256))
        multiple = pipe.vae_scale_factor * pipe.patch_size
        if tile_size % multiple or overlap % multiple or (not 0 <= overlap < tile_size):
            raise ValueError('Krea i2i tile size/overlap must align to the latent grid; overlap must be smaller')
        pipe.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024, tile_sample_stride_height=768, tile_sample_stride_width=768)
    try:
        with torch.no_grad():
            return _generate_krea2_i2i_impl(pipe, **kwargs)
    finally:
        for key, value in prior.items():
            setattr(pipe.vae, key, value)

def _generate_krea2_i2i_impl(pipe, *, prompt: str, image_path: str, mask_path: str | None=None, negative: str='', width: int=1024, height: int=1024, steps: int=8, guidance: float=0.0, strength: float=0.6, seed: int=42, out_path: str, loras=None, on_step=None, should_cancel=None, tile_size: int=0, tile_overlap: int=256) -> str:
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    multiple = pipe.vae_scale_factor * pipe.patch_size
    width = max(multiple, int(width) // multiple * multiple)
    height = max(multiple, int(height) // multiple * multiple)
    gen = torch.Generator(device='cpu').manual_seed(int(seed))
    apply_loras(pipe, loras)
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1, max_sequence_length=512)
    do_cfg = bool(guidance and guidance > 0)
    neg_embeds = neg_mask = None
    if do_cfg:
        neg_embeds, neg_mask = pipe.encode_prompt(prompt=negative or '', device=device, num_images_per_prompt=1, max_sequence_length=512)
    img = Image.open(image_path).convert('RGB').resize((width, height), Image.Resampling.LANCZOS if tile_size else Image.Resampling.BICUBIC)
    px = pipe.image_processor.preprocess(img, height=height, width=width).to(device, pipe.vae.dtype)
    mean, std, z = _vae_norm(pipe.vae, pipe.vae.dtype, device)
    with torch.no_grad():
        enc = pipe.vae.encode(px.unsqueeze(2))
        z_vae = enc.latent_dist.sample(gen) if hasattr(enc, 'latent_dist') else enc.latents
    z_model = ((z_vae - mean) / std).to(dtype)
    lat_h, lat_w = (height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    image_latents = pipe._pack_latents(z_model[:, :, 0], 1, z, lat_h, lat_w)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    seq_len = image_latents.shape[1]
    mu = 1.15 if pipe.config.is_distilled else calculate_shift(seq_len, pipe.scheduler.config.get('base_image_seq_len', 256), pipe.scheduler.config.get('max_image_seq_len', 6400), pipe.scheduler.config.get('base_shift', 0.5), pipe.scheduler.config.get('max_shift', 1.15))
    timesteps, steps = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    t_start = max(0, steps - int(round(steps * float(strength))))
    pipe.scheduler.set_begin_index(t_start)
    run_timesteps = timesteps[t_start:]
    print(f'[krea2-i2i] schedule start {t_start}/{steps}; remaining {len(run_timesteps)}; strength={strength}; canvas={width}x{height}; tile={tile_size}/{tile_overlap}', flush=True)
    noise = randn_tensor(image_latents.shape, generator=gen, device=device, dtype=dtype)
    latents = pipe.scheduler.scale_noise(image_latents, run_timesteps[:1], noise)
    initial_sigma = float(pipe.scheduler.sigmas[t_start])
    sample = latents.reshape(-1)[:4096].float()
    expected_initial = (1 - initial_sigma) * image_latents.reshape(-1)[:4096].float() + initial_sigma * noise.reshape(-1)[:4096].float()
    initial_residual = float((sample - expected_initial).abs().max())
    mask_packed = None
    if mask_path:
        m = Image.open(mask_path).convert('L').resize((lat_w, lat_h))
        mt = torch.from_numpy(np.array(m)).to(device, dtype) / 255.0
        mt = (mt > 0.5).to(dtype)
        mt = mt.view(1, 1, lat_h, lat_w).expand(1, z, lat_h, lat_w)
        mask_packed = pipe._pack_latents(mt, 1, z, lat_h, lat_w)
    grid_h = height // multiple
    grid_w = width // multiple
    position_ids = pipe.prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w, device)
    neg_position_ids = pipe.prepare_position_ids(neg_embeds.shape[1], grid_h, grid_w, device) if do_cfg else None
    total = len(run_timesteps)
    for i, t in enumerate(run_timesteps):
        if should_cancel and should_cancel():
            raise RuntimeError('generation cancelled')
        timestep = (t / pipe.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(latents.dtype)
        if tile_size:
            from .krea_tiled import predict
            noise_pred = predict(pipe, latents, prompt_embeds, timestep, position_ids, prompt_embeds_mask, grid_height=grid_h, grid_width=grid_w, tile=tile_size // multiple, overlap=tile_overlap // multiple, should_cancel=should_cancel)
        else:
            noise_pred = pipe.transformer(hidden_states=latents, encoder_hidden_states=prompt_embeds, timestep=timestep, position_ids=position_ids, encoder_attention_mask=prompt_embeds_mask, return_dict=False)[0]
        if do_cfg:
            if tile_size:
                neg_pred = predict(pipe, latents, neg_embeds, timestep, neg_position_ids, neg_mask, grid_height=grid_h, grid_width=grid_w, tile=tile_size // multiple, overlap=tile_overlap // multiple, should_cancel=should_cancel)
            else:
                neg_pred = pipe.transformer(hidden_states=latents, encoder_hidden_states=neg_embeds, timestep=timestep, position_ids=neg_position_ids, encoder_attention_mask=neg_mask, return_dict=False)[0]
            noise_pred = noise_pred + guidance * (noise_pred - neg_pred)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(dtype)
        if mask_packed is not None:
            if i < total - 1:
                known = pipe.scheduler.scale_noise(image_latents, run_timesteps[i + 1:i + 2], noise)
            else:
                known = image_latents
            latents = mask_packed * latents + (1.0 - mask_packed) * known
        if on_step:
            try:
                on_step(i + 1, total)
            except Exception:
                pass
    latents = pipe._unpack_latents(latents, height, width).to(pipe.vae.dtype)
    latents = latents * std + mean
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type='pil')[0]
    image.save(str(out_path))
    import hashlib, json
    from pathlib import Path
    effective_sigmas = [float(value) for value in pipe.scheduler.sigmas[t_start:]]
    receipt = {'version': 'krea-i2i-execution-v1', 'source_sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(), 'output_sha256': hashlib.sha256(Path(out_path).read_bytes()).hexdigest(), 'width': width, 'height': height, 'seed': seed, 'requested_strength': strength, 'steps': steps, 'start_index': t_start, 'remaining_steps': total, 'timesteps': [float(value) for value in run_timesteps], 'sigmas': effective_sigmas, 'num_train_timesteps': int(pipe.scheduler.config.num_train_timesteps), 'initial_source_coefficient': 1 - effective_sigmas[0], 'initial_noise_coefficient': effective_sigmas[0], 'initial_mixture_max_abs_residual': initial_residual, 'initial_mixture_sample_values': len(sample), 'initial_noise_scope': 'one_global_seeded_latent', 'latent_shape': list(image_latents.shape), 'vae_scale_factor': pipe.vae_scale_factor, 'patch_size': pipe.patch_size, 'scheduler': type(pipe.scheduler).__name__, 'tile_size': tile_size, 'tile_overlap': tile_overlap, 'absolute_tile_positions': bool(tile_size), 'vae_tile_size': getattr(pipe.vae, 'tile_sample_min_height', None) if tile_size else None, 'vae_tile_size_width': getattr(pipe.vae, 'tile_sample_min_width', None) if tile_size else None, 'vae_stride': getattr(pipe.vae, 'tile_sample_stride_height', None) if tile_size else None, 'vae_stride_width': getattr(pipe.vae, 'tile_sample_stride_width', None) if tile_size else None, 'vae_tiling_enabled': bool(getattr(pipe.vae, 'use_tiling', False)) if tile_size else False}
    Path(str(out_path) + '.i2i-receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    return str(out_path)
