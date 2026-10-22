# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_w4a8.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import torch
import torch.nn as nn

def logical_w4a8_meta_state(state, expected):
    if not any((key.endswith('.comfy_quant') for key in state)):
        return state
    if not all((value.is_meta for value in state.values())):
        raise ValueError('W4A8 architecture adaptation requires meta tensors')
    result = {}
    for key, value in state.items():
        if is_w4a8_sidecar(key) or (key in ('latents_mean', 'latents_std') and key not in expected):
            continue
        module = key.removesuffix('.weight')
        if key.endswith('.weight') and module + '.comfy_quant' in state:
            target = expected.get(key)
            if target is None or value.dtype != torch.int8 or value.ndim != 2 or (tuple(target.shape) != (value.shape[0], value.shape[1] * 2)):
                raise ValueError(f'W4A8 logical shape mismatch: {key}')
            for suffix in W4A8_SIDECAR_SUFFIXES[1:]:
                if module + suffix not in state:
                    raise ValueError(f'W4A8 missing {module + suffix}')
            value = torch.empty_like(target, device='meta')
        result[key] = value
    return result

class H3W4A8Linear(nn.Module):

    def __init__(self, in_features, out_features, config, *, bias=False):
        super().__init__()
        self.in_features, self.out_features = (in_features, out_features)
        self.config = dict(config)
        group = int(config.get('group_size', 16))
        for name, shape, dtype in (('weight', (out_features, in_features // 2), torch.int8), ('weight_codebook', (16,), torch.float32), ('weight_s_rel', (out_features, in_features // group), torch.float8_e4m3fn), ('weight_s_channel', (out_features,), torch.float32)):
            self.register_buffer(name, torch.empty(shape, dtype=dtype, device='meta'))
        self.register_buffer('bias', torch.empty(out_features, device='meta', dtype=torch.bfloat16) if bias else None)

    def forward(self, x):
        from .minimax_h3_native import comfy_kitchen
        ck = comfy_kitchen()
        if ck is None:
            raise RuntimeError('Packed W4A8 VAE requires comfy_kitchen')
        return ck.w4a8_int8_linear(x, self.weight, self.weight_s_rel, self.weight_s_channel, codebook=self.weight_codebook, bias=self.bias, group_size=int(self.config.get('group_size', 16)), convrot_groupsize=int(self.config.get('convrot_groupsize', 256)), out_dtype=x.dtype)

def prepare_packed_w4a8_blocks(model, index):
    import os
    if os.environ.get('STUDIO_H3_FUSED_W4A8', '0') != '1':
        return ()
    from .minimax_h3_native import comfy_kitchen
    if comfy_kitchen() is None:
        return ()
    converted = []
    for name, ref in index.refs.items():
        path = name.split('/', 1)[-1]
        if not path.startswith('decoder.transformer_blocks.') or not path.endswith('.comfy_quant'):
            continue
        with open(ref.file, 'rb') as source:
            source.seek(ref.offset)
            config = parse_w4a8_config(source.read(ref.nbytes))
        if not is_w4a8(config):
            continue
        path = path.removesuffix('.comfy_quant')
        target = model.get_submodule(path)
        if not isinstance(target, nn.Linear):
            raise ValueError(f'W4A8 target {path} is not a Linear')
        parent, _, leaf = path.rpartition('.')
        setattr(model.get_submodule(parent), leaf, H3W4A8Linear(target.in_features, target.out_features, config, bias=target.bias is not None))
        converted.append(path)
    model._h3_packed_w4a8_names = frozenset(converted)
    return tuple(converted)

def video_block_tensors(tensors, model):
    packed, dense = ({}, {})
    for name, value in tensors.items():
        path = name.split('/', 1)[-1]
        module_path, _, leaf = path.rpartition('.')
        if module_path in getattr(model, '_h3_packed_w4a8_names', ()):
            if leaf != 'comfy_quant':
                packed[name] = value
        else:
            dense[name] = value
    packed.update(dequantize_w4a8_batch(dense, dtype=torch.bfloat16))
    return packed

def packed_w4a8_weight_saving(component_dir):
    from .safetensors_index import build_index
    index = build_index(component_dir)
    dense, packed = (0, 0)
    for name, ref in index.refs.items():
        path = name.split('/', 1)[-1]
        if not path.startswith('decoder.transformer_blocks.') or not path.endswith('.comfy_quant'):
            continue
        with open(ref.file, 'rb') as source:
            source.seek(ref.offset)
            config = parse_w4a8_config(source.read(ref.nbytes))
        if not is_w4a8(config):
            continue
        prefix = name.removesuffix('comfy_quant')
        weight = index.refs[prefix + 'weight']
        dense += weight.shape[0] * weight.shape[1] * 4
        packed += sum((r.nbytes for key, r in index.refs.items() if key.startswith(prefix) and key != prefix + 'comfy_quant'))
    return max(0, dense - packed - 500000000)
W4A8_FORMAT = 'asym_w4a8_int8'
W4A8_SIDECAR_SUFFIXES = ('.comfy_quant', '.weight_codebook', '.weight_s_rel', '.weight_s_channel')

def parse_w4a8_config(blob) -> dict:
    raw = bytes(blob.reshape(-1).tolist()) if hasattr(blob, 'reshape') else bytes(blob)
    text = raw.decode('utf-8').rstrip('\x00').strip()
    try:
        config = json.loads(text)
    except ValueError as exc:
        raise ValueError(f'W4A8 comfy_quant sidecar is not JSON: {text[:80]!r}') from exc
    if not isinstance(config, dict):
        raise ValueError(f'W4A8 comfy_quant sidecar is not an object: {text[:80]!r}')
    return config

def is_w4a8(config: dict) -> bool:
    return str(config.get('format', '')).strip() == W4A8_FORMAT

def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError(f'W4A8 packed weight must be 2D, got {tuple(packed.shape)}')
    unsigned = packed.view(torch.uint8)
    low = (unsigned & 15).long()
    high = (unsigned >> 4 & 15).long()
    out_features, half = unsigned.shape
    indices = torch.empty(out_features, half * 2, dtype=torch.long, device=packed.device)
    indices[:, 0::2] = low
    indices[:, 1::2] = high
    return indices

def dequantize_w4a8(packed: torch.Tensor, codebook: torch.Tensor, s_rel: torch.Tensor, s_channel: torch.Tensor, config: dict, *, dtype=torch.bfloat16, undo_rotation: bool=True) -> torch.Tensor:
    if not is_w4a8(config):
        raise ValueError(f"not a W4A8 config: {config.get('format')!r}")
    indices = unpack_nibbles(packed)
    codebook = codebook.to(torch.float32).reshape(-1)
    if codebook.numel() != 16:
        raise ValueError(f'W4A8 codebook must hold 16 levels, got {codebook.numel()}')
    weight = codebook.to(indices.device)[indices]
    group_size = int(config.get('group_size', 16))
    out_features, in_features = weight.shape
    if s_rel.ndim != 2 or s_rel.shape[0] != out_features:
        raise ValueError(f'W4A8 per-group scale must be [out, groups], got {tuple(s_rel.shape)}')
    groups = int(s_rel.shape[1])
    if groups * group_size != in_features:
        raise ValueError(f'W4A8 group layout disagrees with the weight: {groups} groups x {group_size} = {groups * group_size}, but in_features is {in_features}')
    weight = weight * s_rel.to(torch.float32).repeat_interleave(group_size, dim=1)
    if s_channel.reshape(-1).numel() != out_features:
        raise ValueError(f'W4A8 per-channel scale must have one entry per output channel ({out_features}), got {s_channel.reshape(-1).numel()}')
    weight = weight * s_channel.to(torch.float32).reshape(out_features, 1)
    if undo_rotation and config.get('convrot'):
        from .minimax_h3_fused import inverse_convrot
        weight = inverse_convrot(weight, int(config.get('convrot_groupsize', 256)))
    elif undo_rotation and 'convrot_groupsize' in config:
        from .minimax_h3_fused import inverse_convrot
        weight = inverse_convrot(weight, int(config['convrot_groupsize']))
    return weight.to(dtype)

def is_w4a8_sidecar(name: str) -> bool:
    return name.endswith(W4A8_SIDECAR_SUFFIXES)

def dequantize_w4a8_batch(tensors: dict, *, dtype=torch.bfloat16, undo_rotation: bool=True) -> dict:
    out: dict = {}
    for name, tensor in tensors.items():
        if is_w4a8_sidecar(name):
            continue
        module = name[:-len('.weight')] if name.endswith('.weight') else name
        blob = tensors.get(f'{module}.comfy_quant')
        if blob is None:
            out[name] = tensor
            continue
        config = parse_w4a8_config(blob)
        if not is_w4a8(config):
            out[name] = tensor
            continue
        missing = [suffix for suffix in ('weight_codebook', 'weight_s_rel', 'weight_s_channel') if f'{module}.{suffix}' not in tensors]
        if missing:
            raise ValueError(f"streamed batch has {name!r} with a W4A8 config but no {', '.join(missing)} -- refusing to bind a quantised weight whose scales did not arrive with it")
        out[name] = dequantize_w4a8(tensor, tensors[f'{module}.weight_codebook'], tensors[f'{module}.weight_s_rel'], tensors[f'{module}.weight_s_channel'], config, dtype=dtype, undo_rotation=undo_rotation)
    return out

def compose_w4a8_video_vae(weights, base_video_vae, out_dir):
    from pathlib import Path
    import os
    import shutil
    weights, base, out_dir = (Path(weights), Path(base_video_vae), Path(out_dir))
    if not weights.is_file():
        raise FileNotFoundError(f'W4A8 video VAE weights are not at {weights}')
    if not (base / 'source' / 'config.json').is_file():
        raise FileNotFoundError(f'base video VAE is incomplete (no source/config.json): {base}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'source').mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.is_file() and item.suffix != '.safetensors':
            target = out_dir / item.name
            if not target.exists():
                shutil.copy2(item, target)
    for item in (base / 'source').iterdir():
        if item.is_file() and item.suffix != '.safetensors':
            target = out_dir / 'source' / item.name
            if not target.exists():
                shutil.copy2(item, target)
    payload = out_dir / 'source' / 'model.safetensors'
    if not payload.exists():
        try:
            os.link(weights, payload)
        except OSError:
            shutil.copy2(weights, payload)
    return out_dir
