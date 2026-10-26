# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Iterable
_SUFFIXES = (('.lora_A.default.weight', 'A'), ('.lora_B.default.weight', 'B'), ('.lora_A.weight', 'A'), ('.lora_B.weight', 'B'), ('.lora_down.weight', 'A'), ('.lora_up.weight', 'B'), ('.alpha', 'alpha'))
_DOTTED_PREFIXES = ('base_model.model.', 'model.diffusion_model.', 'diffusion_model.', 'transformer.', 'pipe.dit.', 'dit.', 'unet.')
_FLAT_PREFIXES = ('lora_unet_', 'lycoris_', 'lora_transformer_')
_TEXT_ENCODER = re.compile('^(lora_te\\d*_|text_encoder\\.|text_encoder_\\d\\.|te\\.|lora_te_)')
_UNSUPPORTED = ('hada_w1', 'hada_w2', 'lokr_w1', 'lokr_w2', 'dora_scale', '.diff', '.diff_b', '.lora_mid')
_FUSED = {'img_mlp.gate_up': ('img_mlp.gate_layer', 'img_mlp.proj')}
_QWEN_IMAGE_V1_HINTS = ('add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out', 'txt_mlp', 'txt_mod', 'img_mod', 'img_mlp.net.')

class QwenLoraError(ValueError):
    pass

def linear_names(module) -> set[str]:
    import torch
    return {name for name, sub in module.named_modules() if isinstance(sub, torch.nn.Linear)}

def _split_suffix(key: str):
    for suffix, kind in _SUFFIXES:
        if key.endswith(suffix):
            return (key[:-len(suffix)], kind)
    return (None, None)

def _peft_scaling(metadata: dict | None) -> tuple[float | None, dict, dict]:
    if not metadata:
        return (None, {}, {})
    raw = metadata.get('lora_adapter_metadata')
    if not raw:
        return (None, {}, {})
    try:
        meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except ValueError:
        return (None, {}, {})

    def pick(name):
        for key in (f'transformer.{name}', name):
            if key in meta:
                return meta[key]
        return None
    alpha = pick('lora_alpha')
    return (float(alpha) if alpha is not None else None, dict(pick('alpha_pattern') or {}), dict(pick('rank_pattern') or {}))

def map_state_dict(state_dict: dict[str, Any], targets: Iterable[str], metadata: dict | None=None, *, label: str='LoRA') -> tuple[dict[str, list[tuple[Any, Any, float]]], dict[str, Any]]:
    targets = set(targets)
    flat = {name.replace('.', '_'): name for name in targets}
    for fused in _FUSED:
        for name in list(targets):
            if name.endswith(_FUSED[fused][0]):
                virtual = name[:-len(_FUSED[fused][0])] + fused
                flat[virtual.replace('.', '_')] = virtual
    bad = sorted({k for k in state_dict if any((tag in k for tag in _UNSUPPORTED))})
    if bad:
        raise QwenLoraError(f'{label}: LoHa/LoKr/DoRA/full-diff adapters are not supported for Qwen-Image-2.1 (e.g. {bad[0]}); convert it to a plain LoRA')
    parts: dict[str, dict[str, Any]] = {}
    skipped_text: list[str] = []
    unmapped: list[str] = []
    fmt = set()
    for key, value in state_dict.items():
        base, kind = _split_suffix(key)
        if base is None:
            unmapped.append(key)
            continue
        if _TEXT_ENCODER.match(base):
            skipped_text.append(key)
            continue
        name = base
        changed = True
        while changed:
            changed = False
            for prefix in _DOTTED_PREFIXES:
                if name.startswith(prefix):
                    name, changed = (name[len(prefix):], True)
                    fmt.add(prefix.rstrip('.'))
        for prefix in _FLAT_PREFIXES:
            if name.startswith(prefix):
                fmt.add(prefix.rstrip('_'))
                name = flat.get(name[len(prefix):], name)
                break
        if name not in targets and (not any((name.endswith('.' + f) for f in _FUSED))):
            unmapped.append(key)
            continue
        parts.setdefault(name, {})[kind] = value
    if unmapped:
        hint = ''
        if any((h in k for k in unmapped for h in _QWEN_IMAGE_V1_HINTS)):
            hint = ' -- these are Qwen-Image (2025, dual-stream) module names; that LoRA was trained for a different architecture and cannot be applied to Qwen-Image-2.1'
        raise QwenLoraError(f"{label}: {len(unmapped)} key(s) do not map onto Qwen-Image-2.1 (e.g. {', '.join(sorted(unmapped)[:3])}){hint}")
    if not parts:
        raise QwenLoraError(f'{label}: no LoRA weights found for the Qwen-Image-2.1 transformer')
    peft_alpha, alpha_pattern, _ = _peft_scaling(metadata)
    mapped: dict[str, list[tuple[Any, Any, float]]] = {}
    for name, entry in parts.items():
        if 'A' not in entry or 'B' not in entry:
            raise QwenLoraError(f'{label}: {name} has an incomplete A/B pair ({sorted(entry)})')
        A, B = (entry['A'], entry['B'])
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise QwenLoraError(f'{label}: {name} has non-matrix or mismatched LoRA shapes {tuple(A.shape)} / {tuple(B.shape)}')
        rank = int(A.shape[0])
        if 'alpha' in entry:
            scale = float(entry['alpha'].float().item() if hasattr(entry['alpha'], 'float') else entry['alpha']) / rank
        else:
            alpha = next((float(v) for pat, v in alpha_pattern.items() if name.endswith(pat) or re.search(pat, name)), peft_alpha)
            scale = alpha / rank if alpha is not None else 1.0
        fused = next((f for f in _FUSED if name.endswith('.' + f) or name == f), None)
        if fused:
            stem = name[:-len(fused)]
            gate, up = (stem + part for part in _FUSED[fused])
            half = B.shape[0] // 2
            if B.shape[0] % 2 or gate not in targets or up not in targets:
                raise QwenLoraError(f'{label}: cannot split fused {name} (B rows {B.shape[0]})')
            mapped.setdefault(gate, []).append((A, B[:half], scale))
            mapped.setdefault(up, []).append((A, B[half:], scale))
            fmt.add('fused-gate_up')
        else:
            mapped.setdefault(name, []).append((A, B, scale))
    report = {'modules': len(mapped), 'tensors': len(state_dict) - len(skipped_text), 'skipped_text_encoder_keys': len(skipped_text), 'layout': sorted(fmt) or ['diffusers']}
    return (mapped, report)

def load_lora_file(path: str | Path, device: str='cuda:0'):
    from safetensors import safe_open
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise QwenLoraError(f'{path.name}: only .safetensors LoRAs are loaded (no pickle)')
    if not path.is_file():
        raise QwenLoraError(f'LoRA file not found: {path}')
    with safe_open(str(path), framework='pt', device=str(device)) as handle:
        metadata = handle.metadata() or {}
        return ({key: handle.get_tensor(key) for key in handle.keys()}, metadata)

class LoraStack:

    def __init__(self, module, specs: list[dict], *, device='cuda:0', dtype=None, loader=load_lora_file, fused=False):
        import torch
        dtype = dtype or torch.bfloat16
        targets = linear_names(module)
        merged: dict[str, list[tuple[Any, Any]]] = {}
        self.reports = []
        for spec in specs:
            path, strength = (str(spec['path']), float(spec.get('weight', spec.get('strength', 1.0))))
            state, metadata = loader(path, device)
            mapped, report = map_state_dict(state, targets, metadata, label=Path(path).name)
            for name, entries in mapped.items():
                for A, B, scale in entries:
                    merged.setdefault(name, []).append((A.to(device=device, dtype=dtype), (B.to(device=device, dtype=torch.float32) * (scale * strength)).to(dtype)))
            self.reports.append({'path': path, 'weight': strength, **report})
            del state
        self.calls = 0
        self.handles = []
        self.modules = {}
        for name, entries in merged.items():
            A = torch.cat([a for a, _ in entries], dim=0).contiguous()
            B = torch.cat([b for _, b in entries], dim=1).contiguous()
            sub = module.get_submodule(name)
            if A.shape[1] != sub.in_features or B.shape[0] != sub.out_features:
                self.close()
                raise QwenLoraError(f'{name}: LoRA shape {tuple(B.shape)}x{tuple(A.shape)} does not fit Linear({sub.in_features}->{sub.out_features})')
            if fused:
                delta = (B.float() @ A.float()).to(dtype)
                self.modules[name] = delta
                saved = []

                def before(mod, inputs, delta=delta, saved=saved):
                    saved.append(mod.weight)
                    mod._parameters['weight'] = torch.nn.Parameter(mod.weight + delta, requires_grad=False)
                    self.calls += 1

                def after(mod, inputs, output, saved=saved):
                    if saved:
                        mod._parameters['weight'] = saved.pop()
                    return output
                self.handles.append(sub.register_forward_pre_hook(before))
                self.handles.append(sub.register_forward_hook(after, always_call=True))
            else:
                self.modules[name] = (A, B)
                self.handles.append(sub.register_forward_hook(self._hook(A, B)))

    def _hook(self, A, B):
        import torch.nn.functional as F

        def add_delta(mod, inputs, output):
            self.calls += 1
            x = inputs[0]
            return output + F.linear(F.linear(x.to(A.dtype), A), B).to(output.dtype)
        return add_delta

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.modules.clear()

def stack_key(specs: list[dict]) -> tuple:
    out = []
    for spec in specs:
        path = Path(str(spec['path']))
        try:
            stat = path.stat()
            ident = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            ident = None
        out.append((str(path), ident, float(spec.get('weight', spec.get('strength', 1.0)))))
    return tuple(out)
