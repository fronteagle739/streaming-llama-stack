# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_int8.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .minimax_h3_fused import regular_hadamard
_MIN_ROWS_FOR_INT8 = 32

def proj_chunk() -> int:
    return int(os.environ.get('STUDIO_H3_PROJ_CHUNK', '0'))

def rotate_activations(x: torch.Tensor, group_size: int) -> torch.Tensor:
    *lead, in_features = x.shape
    if in_features % group_size:
        raise ValueError(f'activation width {in_features} is not divisible by ConvRot group {group_size}')
    h = regular_hadamard(group_size, device=str(x.device), dtype=x.dtype)
    return (x.reshape(-1, in_features // group_size, group_size) @ h).reshape(*lead, in_features)

def quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return (q, scale.to(torch.float32))

def _int8_matmul(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor | None:
    rows, k = xq.shape
    out_features = wq.shape[0]
    if not xq.is_cuda or k % 8 or out_features % 8:
        return None
    target = max(_MIN_ROWS_FOR_INT8, (rows + 7) // 8 * 8)
    try:
        if target != rows:
            xq = F.pad(xq, (0, 0, 0, target - rows))
        acc = torch._int_mm(xq, wq.t())
    except (RuntimeError, NotImplementedError):
        return None
    return acc[:rows] if target != rows else acc

def _adapter_compute_dtype(value: torch.Tensor) -> torch.dtype:
    if os.environ.get('STUDIO_H3_LORA_BF16', '0') != '1':
        return torch.float32
    return value.dtype if value.dtype in (torch.bfloat16, torch.float16) else torch.float32
_ACTIVE_LORA_PASS = 0

def set_active_lora_pass(index: int) -> None:
    global _ACTIVE_LORA_PASS
    _ACTIVE_LORA_PASS = int(index)

def active_lora_pass() -> int:
    return _ACTIVE_LORA_PASS

def adapter_active(passes) -> bool:
    return not _ACTIVE_LORA_PASS or not passes or _ACTIVE_LORA_PASS in passes

def _adapter_store(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if os.environ.get('STUDIO_H3_LORA_BF16', '0') == '1' and tensor.dtype in (torch.float32, torch.float64):
        return tensor.to(torch.bfloat16)
    return tensor

class H3Int8ConvRotLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, *, bias: bool, group_size: int, device='meta', dtype=torch.bfloat16) -> None:
        super().__init__(in_features, out_features, bias=bias, device='meta', dtype=dtype)
        del self._parameters['weight']
        self.register_buffer('weight', torch.empty(out_features, in_features, dtype=torch.int8, device=device), persistent=True)
        self.register_buffer('weight_scale', torch.empty(out_features, 1, dtype=torch.float32, device=device), persistent=True)
        self.group_size = int(group_size)
        self.compute_dtype = dtype
        self.inference_adapters: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def _rows_forward(self, flat_in: torch.Tensor) -> torch.Tensor:
        rotated = rotate_activations(flat_in.to(torch.float32), self.group_size)
        xq, xs = quantize_per_row(rotated)
        acc = _int8_matmul(xq, self.weight)
        if acc is None:
            w = self.weight.to(torch.float32) * self.weight_scale
            out = rotated @ w.t()
        else:
            out = acc.to(torch.float32) * xs * self.weight_scale.reshape(1, -1)
        return out.to(self.compute_dtype)

    def base_forward(self, value: torch.Tensor) -> torch.Tensor:
        *lead, in_features = value.shape
        flat_in = value.reshape(-1, in_features)
        rows = flat_in.shape[0]
        chunk = proj_chunk()
        if chunk and rows > chunk:
            out = torch.cat([self._rows_forward(flat_in[i:i + chunk]) for i in range(0, rows, chunk)], dim=0)
        else:
            out = self._rows_forward(flat_in)
        out = out.reshape(*lead, self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def add_inference_adapter(self, down: torch.Tensor, up: torch.Tensor, scale: float, passes=()) -> None:
        self.inference_adapters.append((_adapter_store(down), _adapter_store(up), float(scale), frozenset(passes)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base_forward(value)
        _dt = _adapter_compute_dtype(value)
        for down, up, scale, passes in self.inference_adapters:
            if not adapter_active(passes):
                continue
            extra = F.linear(F.linear(value.to(_dt), down.to(_dt)), up.to(_dt))
            result = result + extra.to(result.dtype) * scale
        return result

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, int8_convrot=group{self.group_size}'
_PACKED_LINEAR_CALLS = {'fused_int8': 0, 'float_fallback': 0}

def packed_linear_stats():
    return dict(_PACKED_LINEAR_CALLS)

class H3PackedConvRotLinear(H3Int8ConvRotLinear):

    def dequantize_weight(self) -> torch.Tensor:
        from .minimax_h3_fused import dequantize_int8_convrot
        return dequantize_int8_convrot(self.weight, self.weight_scale, {}, dtype=self.compute_dtype, undo_rotation=False)

    def base_forward(self, value: torch.Tensor) -> torch.Tensor:
        if os.environ.get('STUDIO_H3_FUSED_INT8', '0') == '1' and value.is_cuda and (not torch.is_grad_enabled()):
            from .minimax_h3_native import comfy_kitchen
            ck = comfy_kitchen()
            if ck is not None:
                _PACKED_LINEAR_CALLS['fused_int8'] += 1
                return ck.int8_linear(value, self.weight, self.weight_scale, bias=self.bias, out_dtype=self.compute_dtype, convrot=True, convrot_groupsize=self.group_size)
        _PACKED_LINEAR_CALLS['float_fallback'] += 1
        weight = self.dequantize_weight()
        *lead, width = value.shape
        flat = value.reshape(-1, width)
        chunk = proj_chunk()
        if chunk and flat.shape[0] > chunk:
            rotated = torch.cat([rotate_activations(flat[i:i + chunk], self.group_size) for i in range(0, flat.shape[0], chunk)], dim=0)
        else:
            rotated = rotate_activations(flat, self.group_size)
        out = F.linear(rotated, weight, self.bias)
        return out.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return super().extra_repr() + ', float_matmul=True'

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base_forward(value)
        _dt = _adapter_compute_dtype(value)
        for down, up, scale, passes in self.inference_adapters:
            if not adapter_active(passes):
                continue
            extra = F.linear(F.linear(value.to(_dt), down.to(_dt)), up.to(_dt))
            result = result + extra.to(result.dtype) * scale
        return result

class H3ConvRotLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, *, bias: bool, group_size: int, device='meta', dtype=torch.bfloat16) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.group_size = int(group_size)
        self.inference_adapters: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def base_forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(rotate_activations(value, self.group_size), self.weight, self.bias)

    def add_inference_adapter(self, down: torch.Tensor, up: torch.Tensor, scale: float, passes=()) -> None:
        self.inference_adapters.append((_adapter_store(down), _adapter_store(up), float(scale), frozenset(passes)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base_forward(value)
        _dt = _adapter_compute_dtype(value)
        for down, up, scale, passes in self.inference_adapters:
            if not adapter_active(passes):
                continue
            extra = F.linear(F.linear(value.to(_dt), down.to(_dt)), up.to(_dt))
            result = result + extra.to(result.dtype) * scale
        return result

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, convrot=group{self.group_size}'

def convert_h3_linears_to_convrot(model: nn.Module, module_names, *, group_size: int=256, device='meta', dtype=torch.bfloat16) -> tuple[str, ...]:
    converted: list[str] = []
    for name in sorted(module_names):
        target = model.get_submodule(name)
        if isinstance(target, (H3ConvRotLinear, H3Int8ConvRotLinear)):
            continue
        if not isinstance(target, nn.Linear):
            raise ValueError(f'{name!r} is a {type(target).__name__}, not nn.Linear — refusing to convert it')
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, H3ConvRotLinear(target.in_features, target.out_features, bias=target.bias is not None, group_size=group_size, device=device, dtype=dtype))
        converted.append(name)
    return tuple(converted)

def convert_h3_linears_to_int8(model: nn.Module, module_names, *, group_size: int=256, device='meta', dtype=torch.bfloat16, float_matmul: bool=False) -> tuple[str, ...]:
    converted: list[str] = []
    for name in sorted(module_names):
        target = model.get_submodule(name)
        if isinstance(target, H3Int8ConvRotLinear):
            continue
        if not isinstance(target, nn.Linear):
            raise ValueError(f'{name!r} is a {type(target).__name__}, not nn.Linear — refusing to convert it')
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        linear_type = H3PackedConvRotLinear if float_matmul else H3Int8ConvRotLinear
        setattr(parent, leaf, linear_type(target.in_features, target.out_features, bias=target.bias is not None, group_size=group_size, device=device, dtype=dtype))
        converted.append(name)
    return tuple(converted)

def quantised_module_names(index_names, quant_map=None) -> tuple[str, ...]:
    from .minimax_h3_fused import strip_comfy_prefix
    suffix = '.comfy_quant'
    names = {strip_comfy_prefix(n.split('/', 1)[-1][:-len(suffix)]) for n in index_names if n.endswith(suffix)}
    names.update((strip_comfy_prefix(str(module)) for module in quant_map or {}))
    return tuple(sorted(names))

def dequantized_reference(module: H3Int8ConvRotLinear, value: torch.Tensor) -> torch.Tensor:
    from .minimax_h3_fused import inverse_convrot
    w = module.weight.to(torch.float32) * module.weight_scale
    w = inverse_convrot(w, module.group_size)
    bias = None if module.bias is None else module.bias.to(torch.float32)
    return F.linear(value.to(torch.float32), w, bias)
