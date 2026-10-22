# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_attention.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import asdict, dataclass
import importlib
import os
import re
import sys
from typing import Callable
import torch
SUPPORTED_H3_ATTENTION_BACKENDS = ('sdpa', 'sageattention', 'flash_attn2', 'flash_attn2_int8')

def resolve_h3_attention_backend(backend: str) -> str:
    if backend == 'flash_attn2' and os.environ.get('STUDIO_H3_INT8_ATTENTION', '0') == '1':
        return 'flash_attn2_int8'
    return backend
_INT8_ATTENTION_CALLS = dict(kitchen_int8=0, sage_int8=0, flash_fallback=0)

def int8_attention_stats() -> dict:
    return dict(_INT8_ATTENTION_CALLS)

def reset_int8_attention_stats() -> None:
    for key in _INT8_ATTENTION_CALLS:
        _INT8_ATTENTION_CALLS[key] = 0

class H3Int8VarlenAttention:

    def __init__(self, flash, kernel, *, implementation='kitchen'):
        self.flash = flash
        self.kernel = kernel
        self.implementation = implementation

    def __call__(self, q, k, v, cu_q, cu_k, max_q, max_k, *, dropout_p=0.0, causal=False):
        if torch.is_grad_enabled() or not q.is_cuda or dropout_p or causal:
            result = self.flash(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=dropout_p, causal=causal)
            _INT8_ATTENTION_CALLS['flash_fallback'] += 1
            return result
        if cu_q.numel() != cu_k.numel():
            raise ValueError('Q and K must have the same number of packed segments')
        if cu_q.numel() == 2:
            return self._attend(q, k, v)
        q_bounds, k_bounds = (cu_q.tolist(), cu_k.tolist())
        out = torch.empty_like(q)
        for qa, qb, ka, kb in zip(q_bounds, q_bounds[1:], k_bounds, k_bounds[1:]):
            if qb > qa:
                out[qa:qb] = self._attend(q[qa:qb], k[ka:kb], v[ka:kb])
        return out

    def _attend(self, q, k, v):
        if self.implementation == 'sage' and k.shape[0] < 64:
            cu_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
            cu_k = torch.tensor([0, k.shape[0]], device=k.device, dtype=torch.int32)
            out = self.flash(q, k, v, cu_q, cu_k, q.shape[0], k.shape[0], dropout_p=0.0, causal=False)
            _INT8_ATTENTION_CALLS['flash_fallback'] += 1
            return out[0] if isinstance(out, tuple) else out
        q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        if self.implementation == 'sage':
            out = self.kernel(q, k, v, tensor_layout='HND', is_causal=False)
            _INT8_ATTENTION_CALLS['sage_int8'] += 1
        else:
            out = self.kernel(q, k, v)
            _INT8_ATTENTION_CALLS['kitchen_int8'] += 1
        return out.squeeze(0).transpose(0, 1)

def load_int8_attn_varlen() -> H3Int8VarlenAttention:
    require_h3_attention_backend('flash_attn2_int8')
    implementation = os.environ.get('STUDIO_H3_INT8_ATTENTION_KERNEL', 'kitchen')
    kernel = load_sageattn() if implementation == 'sage' else importlib.import_module('comfy_kitchen.sage_attention').int8_attention
    return H3Int8VarlenAttention(load_flash_attn_varlen(), kernel, implementation=implementation)

def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple((int(part) for part in re.findall('\\d+', value.split('+')[0])[:3]))

@dataclass(frozen=True)
class H3AttentionCapability:
    backend: str
    available: bool
    reason: str | None = None
    python_version: str = ''
    torch_version: str = ''
    triton_version: str = ''
    cuda_version: str = ''
    compute_capability: tuple[int, int] | None = None
    api: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

class H3AttentionBackendUnavailable(RuntimeError):

    def __init__(self, capability: H3AttentionCapability) -> None:
        self.capability = capability
        super().__init__(f"H3 attention backend {capability.backend!r} is unavailable: {capability.reason or 'unknown reason'}")

    def as_dict(self) -> dict:
        return {'code': 'h3_attention_backend_unavailable', 'capability': self.capability.as_dict(), 'requirements': {'python': '>=3.9', 'torch': '>=2.3', 'triton': '>=3.0', 'blackwell_cuda': '>=12.8', 'api': 'sageattention.sageattn'}}

def detect_h3_attention_backend(backend: str, *, import_module: Callable[[str], object]=importlib.import_module, torch_module=torch) -> H3AttentionCapability:
    python_version = '.'.join((str(part) for part in sys.version_info[:3]))
    torch_version = str(getattr(torch_module, '__version__', ''))
    cuda_version = str(getattr(getattr(torch_module, 'version', None), 'cuda', '') or '')
    if backend not in SUPPORTED_H3_ATTENTION_BACKENDS:
        return H3AttentionCapability(backend, False, f'unsupported backend; choose {SUPPORTED_H3_ATTENTION_BACKENDS}', python_version, torch_version, cuda_version=cuda_version)
    if backend == 'sdpa':
        return H3AttentionCapability(backend, True, python_version=python_version, torch_version=torch_version, cuda_version=cuda_version, api='torch.nn.functional.scaled_dot_product_attention')
    if backend == 'flash_attn2_int8':
        flash = detect_h3_attention_backend('flash_attn2', import_module=import_module, torch_module=torch_module)
        reason = flash.reason
        implementation = os.environ.get('STUDIO_H3_INT8_ATTENTION_KERNEL', 'kitchen')
        api = 'comfy_kitchen.sage_attention.int8_attention'
        if implementation == 'sage':
            sage = detect_h3_attention_backend('sageattention', import_module=import_module, torch_module=torch_module)
            reason = reason or sage.reason
            api = sage.api
        elif implementation == 'kitchen':
            try:
                module = import_module('comfy_kitchen.sage_attention')
                if not callable(getattr(module, 'int8_attention', None)) or not module.is_available():
                    reason = reason or 'comfy_kitchen INT8 attention is unavailable on this device'
            except (ImportError, AttributeError, RuntimeError) as exc:
                reason = reason or str(exc)
        else:
            reason = 'STUDIO_H3_INT8_ATTENTION_KERNEL must be kitchen or sage'
        return H3AttentionCapability(backend, reason is None, reason, python_version, torch_version, cuda_version=cuda_version, compute_capability=flash.compute_capability, api=api)
    if backend == 'flash_attn2':
        cc = None
        try:
            if torch_module.cuda.is_available():
                cc = torch_module.cuda.get_device_capability(0)
        except Exception:
            cc = None
        try:
            module = import_module('flash_attn.flash_attn_interface')
            fn = getattr(module, 'flash_attn_varlen_func', None)
        except (ImportError, ModuleNotFoundError) as exc:
            return H3AttentionCapability(backend, False, f'flash_attn is not importable: {exc}', python_version, torch_version, cuda_version=cuda_version, compute_capability=cc)
        if not callable(fn):
            return H3AttentionCapability(backend, False, 'flash_attn.flash_attn_interface.flash_attn_varlen_func is missing', python_version, torch_version, cuda_version=cuda_version, compute_capability=cc)
        return H3AttentionCapability(backend, True, python_version=python_version, torch_version=torch_version, cuda_version=cuda_version, compute_capability=cc, api='flash_attn.flash_attn_interface.flash_attn_varlen_func')
    reason = None
    triton_version = ''
    compute_capability = None
    if sys.version_info < (3, 9):
        reason = 'Python 3.9 or newer is required'
    elif _version_tuple(torch_version) < (2, 3):
        reason = f"PyTorch 2.3 or newer is required (found {torch_version or 'unknown'})"
    try:
        triton = import_module('triton')
        triton_version = str(getattr(triton, '__version__', ''))
    except (ImportError, ModuleNotFoundError) as exc:
        if reason is None:
            reason = f'Triton is not importable: {exc}'
    if reason is None and _version_tuple(triton_version) < (3, 0):
        reason = f"Triton 3.0 or newer is required (found {triton_version or 'unknown'})"
    cuda = getattr(torch_module, 'cuda', None)
    cuda_available = bool(cuda is not None and callable(getattr(cuda, 'is_available', None)) and cuda.is_available())
    if cuda_available:
        compute_capability = tuple((int(part) for part in cuda.get_device_capability()))
        if compute_capability >= (12, 0) and _version_tuple(cuda_version) < (12, 8):
            reason = reason or f"CUDA 12.8 or newer is required on Blackwell (found {cuda_version or 'unknown'})"
    sage_module = None
    try:
        sage_module = import_module('sageattention')
    except (ImportError, ModuleNotFoundError) as exc:
        reason = reason or f'sageattention is not importable: {exc}'
    sageattn = getattr(sage_module, 'sageattn', None) if sage_module is not None else None
    if sage_module is not None and (not callable(sageattn)):
        reason = reason or 'sageattention.sageattn is missing or not callable'
    if reason is None and (not cuda_available):
        reason = 'CUDA is not available'
    return H3AttentionCapability(backend=backend, available=reason is None, reason=reason, python_version=python_version, torch_version=torch_version, triton_version=triton_version, cuda_version=cuda_version, compute_capability=compute_capability, api='sageattention.sageattn' if callable(sageattn) else None)

def require_h3_attention_backend(backend: str) -> H3AttentionCapability:
    capability = detect_h3_attention_backend(backend)
    if not capability.available:
        raise H3AttentionBackendUnavailable(capability)
    return capability

def load_sageattn() -> Callable:
    capability = require_h3_attention_backend('sageattention')
    module = importlib.import_module('sageattention')
    function = getattr(module, 'sageattn')
    if not callable(function):
        raise H3AttentionBackendUnavailable(capability)
    return function

def load_flash_attn_varlen() -> Callable:
    capability = require_h3_attention_backend('flash_attn2')
    module = importlib.import_module('flash_attn.flash_attn_interface')
    function = getattr(module, 'flash_attn_varlen_func', None)
    if not callable(function):
        raise H3AttentionBackendUnavailable(capability)
    return function
