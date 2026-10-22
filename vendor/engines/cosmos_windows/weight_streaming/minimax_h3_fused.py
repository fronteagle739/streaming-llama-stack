# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_fused.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Mapping
import torch
_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}
FUSED_TIME_KEYS = ('adaln_t_table', 'adaln_basis', 'adaln_mean')
COMFY_STATE_DICT_PREFIX = 'model.diffusion_model.'
FUSED_REQUIRED_TIME_KEYS = ('adaln_t_table',)

def strip_comfy_prefix(name: str) -> str:
    return name[len(COMFY_STATE_DICT_PREFIX):] if name.startswith(COMFY_STATE_DICT_PREFIX) else name
_QUANT_SIDECAR_SUFFIXES = ('.comfy_quant', '.weight_scale')

def regular_hadamard(size: int, *, device='cpu', dtype=torch.float32) -> torch.Tensor:
    if size < 4 or size & size - 1 != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f'INT8 ConvRot group size must be a power of 4, got {size}')
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device=device, dtype=dtype)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / size ** 0.5
    _HADAMARD_CACHE[key] = h
    return h

def inverse_convrot(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f'INT8 ConvRot expects a 2D linear weight, got {tuple(weight.shape)}')
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f'INT8 ConvRot in_features {in_features} is not divisible by group size {group_size}')
    h = regular_hadamard(group_size, device=str(weight.device), dtype=weight.dtype)
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    return torch.matmul(grouped, h).reshape(out_features, in_features)

def sectioned_qkv_to_interleaved(weight, *, heads: int, head_dim: int):
    rows, cols = weight.shape
    expected = 3 * heads * head_dim
    if rows != expected:
        raise ValueError(f'qkv_proj has {rows} rows, expected 3*{heads}*{head_dim}={expected}; refusing to re-pack a tensor whose head geometry does not match the config')
    return weight.view(3, heads, head_dim, cols).permute(1, 0, 2, 3).reshape(rows, cols)

def parse_comfy_quant(blob: torch.Tensor) -> dict:
    raw = bytes(blob.reshape(-1).tolist())
    text = raw.decode('utf-8').rstrip('\x00').strip()
    try:
        config = json.loads(text)
    except ValueError as exc:
        if not raw or not any(raw):
            raise ValueError(f'comfy_quant sidecar for this tensor is EMPTY ({len(raw)} bytes, all zero): the streamed fetch had not landed when it was read, not a bad checkpoint. The WEIGHT in this batch came through the same wait and is equally suspect') from exc
        raise ValueError(f'comfy_quant sidecar is not JSON: {text[:80]!r}') from exc
    if not isinstance(config, dict):
        raise ValueError(f'comfy_quant sidecar is not an object: {text[:80]!r}')
    return config

def dequantize_int8_convrot(weight: torch.Tensor, scale: torch.Tensor, config: dict, *, dtype=torch.bfloat16, undo_rotation: bool=True) -> torch.Tensor:
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)
    if s.numel() == w.shape[0]:
        w = w * s.reshape(-1, *[1] * (w.dim() - 1))
    elif s.numel() == 1:
        w = w * s.reshape(())
    else:
        raise ValueError(f'INT8 scale has {s.numel()} entries, which is neither per-output-channel ({w.shape[0]}) nor scalar')
    params = config.get('params') if isinstance(config.get('params'), dict) else {}
    if undo_rotation and config.get('convrot', params.get('convrot', False)):
        group = int(config.get('convrot_groupsize', params.get('convrot_groupsize', 256)))
        w = inverse_convrot(w, group)
    return w.to(dtype)

def fused_state_dict(path: str | Path, *, dtype=torch.bfloat16, device='cpu') -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework='pt', device=device) as handle:
        names = set(handle.keys())
        file_quant = file_level_quant_map(handle.metadata())
        for name in sorted(names):
            if name.endswith(_QUANT_SIDECAR_SUFFIXES):
                continue
            module = name[:-len('.weight')] if name.endswith('.weight') else name
            quant_key = f'{module}.comfy_quant'
            scale_key = f'{name}_scale'
            config = None
            if quant_key in names:
                config = parse_comfy_quant(handle.get_tensor(quant_key))
            else:
                config = file_quant.get(strip_comfy_prefix(module))
            if config is not None:
                if scale_key not in names:
                    raise ValueError(f'fused H3 checkpoint {path}: {name!r} is quantised but has no {scale_key!r} — refusing to load a weight whose scale is missing')
                out[strip_comfy_prefix(name)] = dequantize_int8_convrot(handle.get_tensor(name), handle.get_tensor(scale_key), config, dtype=dtype)
            else:
                tensor = handle.get_tensor(name)
                out[strip_comfy_prefix(name)] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    missing = [k for k in FUSED_REQUIRED_TIME_KEYS if k not in out]
    if missing:
        raise ValueError(f'fused H3 checkpoint {path} is missing the rank-8 time tensors {missing} — this does not look like a fused artifact, and loading it would leave adaln_proj unconditioned')
    return out
_FILE_QUANT_METADATA_KEY = '_quantization_metadata'

def file_level_quant_map(metadata: Mapping | None) -> dict:
    if not metadata:
        return {}
    raw = metadata.get(_FILE_QUANT_METADATA_KEY)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{_FILE_QUANT_METADATA_KEY} is not JSON: {str(raw)[:80]!r}') from exc
    layers = parsed.get('layers') if isinstance(parsed, Mapping) else None
    if not isinstance(layers, Mapping):
        return {}
    return {strip_comfy_prefix(str(module)): dict(config) for module, config in layers.items() if isinstance(config, Mapping)}

def checkpoint_quant_map(path) -> dict:
    from safetensors import safe_open
    path = Path(path)
    files = sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
    merged: dict = {}
    for shard in files:
        try:
            with safe_open(str(shard), framework='pt') as handle:
                merged.update(file_level_quant_map(handle.metadata()))
        except (OSError, ValueError):
            continue
    return merged

def is_quant_sidecar(name: str) -> bool:
    return name.endswith(_QUANT_SIDECAR_SUFFIXES)

def dequantize_streamed_batch(tensors: dict[str, torch.Tensor], *, dtype=torch.bfloat16, undo_rotation: bool=True, quant_map: dict | None=None) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if is_quant_sidecar(name):
            continue
        module = name[:-len('.weight')] if name.endswith('.weight') else name
        config_blob = tensors.get(f'{module}.comfy_quant')
        if config_blob is not None:
            try:
                config = parse_comfy_quant(config_blob)
            except ValueError as exc:
                if 'EMPTY' in str(exc):
                    empties, total = ([], 0)
                    for other, blob in tensors.items():
                        try:
                            nonzero = bool(blob.any())
                        except Exception:
                            continue
                        total += 1
                        if not nonzero:
                            empties.append(other)
                    raise ValueError(f"{exc} | BATCH: {len(empties)} of {total} tensors are ALL-ZERO{(' (the whole fetch did not land)' if len(empties) == total else '')}; first few: {sorted(empties)[:6]}") from exc
                raise
        else:
            config = (quant_map or {}).get(strip_comfy_prefix(module))
        if config is None:
            out[name] = tensor
            continue
        scale = tensors.get(f'{name}_scale')
        if scale is None:
            raise ValueError(f'streamed batch has {name!r} with a quant config but no {name}_scale — refusing to bind a quantised weight whose scale did not arrive with it')
        out[name] = dequantize_int8_convrot(tensor, scale, config, dtype=dtype, undo_rotation=undo_rotation)
    return out

def index_looks_fused(names) -> bool:
    wanted = set(FUSED_REQUIRED_TIME_KEYS)
    return wanted <= {strip_comfy_prefix(n.rsplit('/', 1)[-1]) for n in names}

def adaln_coefficients(table: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    if table.ndim != 2:
        raise ValueError(f'adaln_t_table must be 2D, got {tuple(table.shape)}')
    rows = table.shape[0]
    t = timestep.to(torch.float32).reshape(-1)
    if torch.any(t < 0) or torch.any(t > 1):
        raise ValueError('adaln timestep must be normalised to [0, 1] before table lookup')
    pos = t * (rows - 1)
    low = torch.clamp(pos.floor().long(), 0, rows - 1)
    high = torch.clamp(low + 1, 0, rows - 1)
    frac = (pos - low.to(pos.dtype)).unsqueeze(-1)
    table = table.to(torch.float32)
    return table[low] * (1.0 - frac) + table[high] * frac

def reconstruct_time_embedding(coefficients: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return mean.to(torch.float32) + coefficients.to(torch.float32) @ basis.to(torch.float32)
