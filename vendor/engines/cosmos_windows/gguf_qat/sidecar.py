# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/sidecar.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gemma4RMSNormLike(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)

class SidecarFFN(nn.Module):

    def __init__(self, hidden: int, d_ff: int=4096, eps: float=1e-06, act: str='gelu_pytorch_tanh', gate_init: float=0.0):
        super().__init__()
        self.hidden = hidden
        self.d_ff = d_ff
        self.norm = Gemma4RMSNormLike(hidden, eps=eps)
        self.gate_up_proj = nn.Linear(hidden, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_init))
        self.last_gate_mean: float = float(torch.sigmoid(torch.tensor(float(gate_init))))
        self.last_gate_max: float = self.last_gate_mean
        if act in ('gelu_pytorch_tanh', 'gelu_tanh'):
            self.act_fn = lambda x: F.gelu(x, approximate='tanh')
        elif act == 'gelu':
            self.act_fn = F.gelu
        elif act == 'silu':
            self.act_fn = F.silu
        else:
            self.act_fn = F.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate_up_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.weight)

    @property
    def gate(self) -> torch.Tensor:
        return torch.tensor(self.last_gate_mean, device=self.gate_proj.weight.device)

    def token_gate(self, h: torch.Tensor) -> torch.Tensor:
        wdtype = self.gate_proj.weight.dtype
        normalized = self.norm(h.to(wdtype))
        gate = torch.sigmoid(self.gate_proj(normalized))
        with torch.no_grad():
            self.last_gate_mean = float(gate.float().mean())
            self.last_gate_max = float(gate.float().max())
        return gate.to(h.dtype)

    def branch(self, h: torch.Tensor) -> torch.Tensor:
        in_dtype = h.dtype
        wdtype = self.gate_up_proj.weight.dtype
        x = self.norm(h.to(wdtype))
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        out = self.down_proj(self.act_fn(gate) * up)
        return out.to(in_dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(h)[0]

    def forward_with_details(self, h: torch.Tensor):
        br = self.branch(h)
        gate = self.token_gate(h).to(br.dtype)
        return (gate * br, br, gate)

class SidecarExpertBank(nn.Module):

    def __init__(self, hidden: int, expert_widths: List[int], *, eps: float=1e-06, act: str='silu', gate_init: float=0.0):
        super().__init__()
        if not expert_widths or any((int(width) < 1 for width in expert_widths)):
            raise ValueError('at least one positive sidecar expert width is required')
        self.experts = nn.ModuleList([SidecarFFN(hidden, d_ff=int(width), eps=eps, act=act, gate_init=gate_init) for width in expert_widths])
        self.register_buffer('enabled_mask', torch.ones(len(self.experts), dtype=torch.bool), persistent=False)
        self.last_gate_tensor = None

    def set_expert_enabled(self, index: int, enabled: bool) -> None:
        self.enabled_mask[int(index)] = bool(enabled)

    def forward_with_details(self, h: torch.Tensor):
        gated_sum = torch.zeros_like(h)
        raw_sum = torch.zeros_like(h)
        gates = []
        for index, expert in enumerate(self.experts):
            if not bool(self.enabled_mask[index]):
                gates.append(torch.zeros((*h.shape[:-1], 1), device=h.device, dtype=h.dtype))
                continue
            gated, raw, gate = expert.forward_with_details(h)
            gated_sum = gated_sum + gated
            raw_sum = raw_sum + raw
            gates.append(gate)
        gate_tensor = torch.cat(gates, dim=-1)
        self.last_gate_tensor = gate_tensor
        return (gated_sum, raw_sum, gate_tensor)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(h)[0]

class SidecarLayerWrapper(nn.Module):

    def __init__(self, orig_layer: nn.Module, sidecar: SidecarFFN):
        super().__init__()
        self.orig_layer = orig_layer
        self.sidecar = sidecar
        self.capture_redundancy = False
        self.last_redundancy: Optional[float] = None
        self.enabled = True

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        h_in = hidden_states
        orig_out = self.orig_layer(hidden_states, *args, **kwargs)
        if isinstance(orig_out, tuple):
            base = orig_out[0]
            rest = orig_out[1:]
        else:
            base = orig_out
            rest = None
        if not self.enabled:
            return orig_out
        gated, branch_raw, _token_gates = self.sidecar.forward_with_details(h_in)
        out = base + gated
        if self.capture_redundancy:
            with torch.no_grad():
                base_delta = (base - h_in).reshape(-1, base.shape[-1]).float()
                br = branch_raw.reshape(-1, branch_raw.shape[-1]).float()
                cs = F.cosine_similarity(br, base_delta, dim=-1)
                self.last_redundancy = float(cs.mean().item())
        if rest is None:
            return out
        return (out, *rest)

def attach_sidecars(model, layer_indices: List[int], d_ff: int=4096, gate_init: float=0.0, device='cuda:0', dtype=torch.float32, eps: Optional[float]=None, act: Optional[str]=None) -> Dict[int, SidecarFFN]:
    inner = model.model if hasattr(model, 'model') else model
    cfg = inner.config
    hidden = cfg.hidden_size
    _eps = eps if eps is not None else getattr(cfg, 'rms_norm_eps', 1e-06)
    _act = act if act is not None else getattr(cfg, 'hidden_activation', 'gelu_pytorch_tanh')
    n_layer = len(inner.layers)
    sidecars: Dict[int, SidecarFFN] = {}
    for i in layer_indices:
        if i < 0:
            i = n_layer + i
        assert 0 <= i < n_layer, f'layer index {i} out of range [0,{n_layer})'
        sc = SidecarFFN(hidden, d_ff=d_ff, eps=_eps, act=_act, gate_init=gate_init)
        sc = sc.to(device=device, dtype=dtype)
        for p in sc.parameters():
            p.requires_grad_(True)
        inner.layers[i] = SidecarLayerWrapper(inner.layers[i], sc)
        sidecars[i] = sc
    return sidecars

def attach_sidecar_banks(model, layer_indices: List[int], expert_widths: List[int], gate_init: float=0.0, device='cuda:0', dtype=torch.float32, eps: Optional[float]=None, act: Optional[str]=None):
    inner = model.model if hasattr(model, 'model') else model
    cfg = inner.config
    hidden = cfg.hidden_size
    _eps = eps if eps is not None else getattr(cfg, 'rms_norm_eps', 1e-06)
    _act = act if act is not None else getattr(cfg, 'hidden_activation', 'silu')
    n_layer = len(inner.layers)
    banks = {}
    for index in layer_indices:
        index = n_layer + index if index < 0 else index
        if not 0 <= index < n_layer:
            raise ValueError(f'layer index {index} out of range [0,{n_layer})')
        bank = SidecarExpertBank(hidden, expert_widths, eps=_eps, act=_act, gate_init=gate_init).to(device=device, dtype=dtype)
        inner.layers[index] = SidecarLayerWrapper(inner.layers[index], bank)
        banks[index] = bank
    return banks

def set_redundancy_capture(sidecars: Dict[int, SidecarFFN], model, on: bool):
    inner = model.model if hasattr(model, 'model') else model
    for layer in inner.layers:
        if isinstance(layer, SidecarLayerWrapper):
            layer.capture_redundancy = on

def set_sidecars_enabled(model, on: bool):
    inner = model.model if hasattr(model, 'model') else model
    for layer in inner.layers:
        if isinstance(layer, SidecarLayerWrapper):
            layer.enabled = on

def collect_redundancy(model) -> Dict[int, float]:
    inner = model.model if hasattr(model, 'model') else model
    out: Dict[int, float] = {}
    for i, layer in enumerate(inner.layers):
        if isinstance(layer, SidecarLayerWrapper) and layer.last_redundancy is not None:
            out[i] = layer.last_redundancy
    return out

def sidecar_state_dict(sidecars: Dict[int, SidecarFFN]) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for idx, sc in sidecars.items():
        for n, p in sc.state_dict().items():
            sd[f'layer{idx}.{n}'] = p.detach().cpu()
    return sd

def trainable_param_count(sidecars: Dict[int, SidecarFFN]) -> int:
    return sum((p.numel() for sc in sidecars.values() for p in sc.parameters()))
