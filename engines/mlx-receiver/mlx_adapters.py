from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
_SIDECAR_KEY = re.compile('^layers\\.(?P<layer>\\d+)\\.experts\\.(?P<expert>\\d+)\\.(?P<part>norm\\.weight|gate_up_proj\\.weight|down_proj\\.weight|gate_proj\\.weight|gate_proj\\.bias)$')

@dataclass(frozen=True)
class MlxLoraArtifact:
    path: Path
    config: dict[str, Any]
    tensors: dict[str, np.ndarray]

@dataclass(frozen=True)
class MlxTokenSidecarArtifact:
    path: Path
    metadata: dict[str, str]
    layers: dict[int, list[dict[str, np.ndarray]]]

def load_mlx_lora(path: str | Path) -> MlxLoraArtifact:
    from safetensors import safe_open
    root = Path(path)
    config = json.loads((root / 'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('schema') != 'studio.mlx_stackable_lora.v1':
        raise ValueError(f'unsupported MLX adapter schema in {root}')
    tensors = {}
    with safe_open(str(root / 'adapters.safetensors'), framework='np') as handle:
        if (handle.metadata() or {}).get('schema') != 'studio.mlx_stackable_lora.v1':
            raise ValueError(f'MLX adapter tensor schema mismatch in {root}')
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    stems = {key[:-len('.lora_a')] for key in tensors if key.endswith('.lora_a')}
    if not stems or any((stem + '.lora_b' not in tensors for stem in stems)):
        raise ValueError(f'MLX adapter contains incomplete LoRA pairs: {root}')
    return MlxLoraArtifact(root, config, tensors)

def load_token_sidecar(path: str | Path) -> MlxTokenSidecarArtifact:
    from safetensors import safe_open
    source = Path(path)
    grouped: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    with safe_open(str(source), framework='np') as handle:
        metadata = dict(handle.metadata() or {})
        if metadata.get('schema') != 'studio.token_gated_sidecar.v1':
            raise ValueError(f'unsupported token-sidecar schema: {source}')
        if metadata.get('architecture') != 'glm-dsa':
            raise ValueError('MLX token sidecars currently require architecture glm-dsa')
        for key in handle.keys():
            match = _SIDECAR_KEY.match(key)
            if match is None:
                raise ValueError(f'unexpected token-sidecar tensor: {key}')
            identity = (int(match.group('layer')), int(match.group('expert')))
            grouped.setdefault(identity, {})[match.group('part')] = handle.get_tensor(key)
    required = {'norm.weight', 'gate_up_proj.weight', 'down_proj.weight', 'gate_proj.weight', 'gate_proj.bias'}
    layers: dict[int, list[dict[str, np.ndarray]]] = {}
    for (layer, expert), tensors in sorted(grouped.items()):
        missing = required - tensors.keys()
        if missing:
            raise ValueError(f'token sidecar layer {layer} expert {expert} missing {sorted(missing)}')
        bank = layers.setdefault(layer, [])
        if expert != len(bank):
            raise ValueError(f'token sidecar layer {layer} has non-contiguous expert index {expert}')
        hidden = tensors['norm.weight'].shape[0]
        gate_up = tensors['gate_up_proj.weight']
        if gate_up.ndim != 2 or gate_up.shape[1] != hidden or gate_up.shape[0] % 2:
            raise ValueError(f'invalid gate/up shape at layer {layer} expert {expert}: {gate_up.shape}')
        width = gate_up.shape[0] // 2
        if tensors['down_proj.weight'].shape != (hidden, width) or tensors['gate_proj.weight'].shape != (1, hidden) or tensors['gate_proj.bias'].shape != (1,):
            raise ValueError(f'invalid sidecar projection shape at layer {layer} expert {expert}')
        bank.append(tensors)
    if not layers:
        raise ValueError(f'token sidecar contains no experts: {source}')
    return MlxTokenSidecarArtifact(source, metadata, layers)

def _resolve_core(model):
    language = getattr(model, 'language_model', model)
    core = getattr(language, 'model', language)
    layers = getattr(core, 'layers', None)
    if layers is None:
        raise ValueError('MLX model exposes no decoder layer list')
    return (language, core, layers)

def attach_mlx_loras(model, paths: list[str]) -> int:
    if not paths:
        return 0
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    class StackedLoRALinear(nn.Module):

        def __init__(self, linear, branches):
            super().__init__()
            self.linear = linear
            self.branches = branches

        def __call__(self, x):
            result = self.linear(x)
            for a, b, scale in self.branches:
                result = result + (x @ a @ b * scale).astype(result.dtype)
            return result
    _language, core, _layers = _resolve_core(model)
    modules = dict(core.named_modules())
    branches: dict[str, list[tuple[Any, Any, float]]] = {}
    for spec in paths:
        artifact = load_mlx_lora(spec)
        scale = float(artifact.config['lora_parameters']['scale'])
        for key, a in artifact.tensors.items():
            if not key.endswith('.lora_a'):
                continue
            stem = key[:-len('.lora_a')]
            b = artifact.tensors[stem + '.lora_b']
            branches.setdefault(stem, []).append((mx.array(a), mx.array(b), scale))
    replacements = []
    for stem, values in branches.items():
        if stem not in modules:
            raise ValueError(f'MLX LoRA target does not exist in model: {stem}')
        replacements.append((stem, StackedLoRALinear(modules[stem], values)))
    core.update_modules(tree_unflatten(replacements))
    return sum((len(values) for values in branches.values()))

def attach_token_sidecars(model, paths: list[str], *, scale: float=1.0, eps: float=1e-06) -> int:
    if not paths:
        return 0
    import mlx.core as mx
    import mlx.nn as nn

    class Expert(nn.Module):

        def __init__(self, tensors):
            super().__init__()
            self.norm = mx.array(tensors['norm.weight'], dtype=mx.float16)
            gate_up = tensors['gate_up_proj.weight']
            width = gate_up.shape[0] // 2
            self.ffn_gate = mx.array(gate_up[:width], dtype=mx.float16)
            self.ffn_up = mx.array(gate_up[width:], dtype=mx.float16)
            self.ffn_down = mx.array(tensors['down_proj.weight'], dtype=mx.float16)
            self.gate = mx.array(tensors['gate_proj.weight'], dtype=mx.float16)
            self.gate_b = mx.array(tensors['gate_proj.bias'], dtype=mx.float16)

        def __call__(self, hidden):
            x = hidden.astype(mx.float32)
            x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
            x = (x * self.norm).astype(mx.float16)
            branch = nn.silu(x @ self.ffn_gate.T) * (x @ self.ffn_up.T) @ self.ffn_down.T
            token_gate = mx.sigmoid(x @ self.gate.T + self.gate_b)
            return (branch * token_gate * scale).astype(hidden.dtype)

    class Bank(nn.Module):

        def __init__(self, experts):
            super().__init__()
            self.experts = experts

        def __call__(self, hidden):
            result = mx.zeros_like(hidden)
            for expert in self.experts:
                result = result + expert(hidden)
            return result

    class WrappedLayer(nn.Module):

        def __init__(self, layer, bank):
            super().__init__()
            self.self_attn = layer.self_attn
            self.mlp = layer.mlp
            self.input_layernorm = layer.input_layernorm
            self.post_attention_layernorm = layer.post_attention_layernorm
            self._studio_token_sidecar_bank = bank

        def __call__(self, x, mask=None, cache=None):
            attention = self.self_attn(self.input_layernorm(x), mask, cache)
            if isinstance(attention, tuple):
                attention = attention[0]
            h = x + attention
            return h + self.mlp(self.post_attention_layernorm(h)) + self._studio_token_sidecar_bank(h)
    _language, _core, layers = _resolve_core(model)
    combined: dict[int, list[dict[str, np.ndarray]]] = {}
    for spec in paths:
        artifact = load_token_sidecar(spec)
        for layer, experts in artifact.layers.items():
            combined.setdefault(layer, []).extend(experts)
    for layer, experts in combined.items():
        if not 0 <= layer < len(layers):
            raise ValueError(f'token-sidecar layer {layer} outside MLX model with {len(layers)} layers')
        layers[layer] = WrappedLayer(layers[layer], Bank([Expert(item) for item in experts]))
    try:
        import mlx_vlm.models.glm_moe_dsa.glm_moe_dsa as glm_dsa

        def sidecar_glm_forward(layer, x, mask=None, cache=None, prev_topk_indices=None):
            r, topk = glm_dsa._glm_attention_forward(layer.self_attn, layer.input_layernorm(x), mask, cache, prev_topk_indices=prev_topk_indices)
            h = x + r
            out = h + layer.mlp(layer.post_attention_layernorm(h))
            bank = getattr(layer, '_studio_token_sidecar_bank', None)
            return (out + bank(h) if bank is not None else out, topk)
        glm_dsa._glm_decoder_layer_forward = sidecar_glm_forward
    except ImportError:
        pass
    return sum((len(items) for items in combined.values()))
