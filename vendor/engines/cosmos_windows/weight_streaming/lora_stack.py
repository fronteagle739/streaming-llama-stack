# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/lora_stack.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import math
import re
import uuid
COSMOS_LORA_PROTOCOL = 1
DIFFUSERS_LORA_PROTOCOL = 1

def _spec(s):
    if isinstance(s, (list, tuple)):
        path = s[0]
        weight = float(s[1]) if len(s) > 1 else 1.0
        name = s[2] if len(s) > 2 else None
    elif isinstance(s, dict):
        path = s['path']
        weight = float(s.get('weight', 1.0))
        name = s.get('name')
    else:
        path = s.path
        weight = float(getattr(s, 'weight', 1.0))
        name = getattr(s, 'name', None)
    return (path, weight, name)

def _adapter_name(name, fallback):
    import re
    clean = re.sub('[^0-9A-Za-z_]+', '_', str(name or fallback)).strip('_')
    return clean or fallback

def activate_transformer_adapters(transformer, names, weights=None):
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters
    names = list(names)
    weights = [1.0] * len(names) if weights is None else [float(weight) for weight in weights]
    if len(names) != len(weights) or not all((math.isfinite(weight) for weight in weights)):
        raise ValueError('Each adapter must have one finite scalar weight')
    missing = set(names) - set(getattr(transformer, 'peft_config', {}) or {})
    if missing:
        raise ValueError(f'Adapters are not attached: {sorted(missing)}')
    set_weights_and_activate_adapters(transformer, names, weights)

def apply_lora_stack(target, specs, *, low_cpu_mem_usage=None):
    specs = [s for s in specs or [] if s is not None]
    if not specs:
        return ([], [])
    has_pipe_loader = hasattr(target, 'load_lora_weights')
    names, weights, seen = ([], [], {})
    for i, s in enumerate(specs):
        path, weight, name = _spec(s)
        name = _adapter_name(name, f'lora{i}')
        if name in seen:
            name = _adapter_name(f'{name}_{i}', f'lora{i}')
        local = Path(path)
        if not local.is_file():
            raise FileNotFoundError(f'Local LoRA file not found: {local}')
        loader_args = dict(adapter_name=name, weight_name=local.name, local_files_only=True)
        if low_cpu_mem_usage is not None:
            loader_args['low_cpu_mem_usage'] = low_cpu_mem_usage
        if has_pipe_loader:
            target.load_lora_weights(str(local.parent), **loader_args)
        elif hasattr(target, 'load_lora_adapter'):
            target.load_lora_adapter(str(local.parent), **loader_args)
        else:
            raise RuntimeError(f'{type(target).__name__} has no load_lora_weights/load_lora_adapter — cannot stack LoRA (for the custom streamed-base path use gen_with_lora.apply_lora)')
        seen[name] = True
        names.append(name)
        weights.append(weight)
    if has_pipe_loader:
        try:
            target.set_adapters(names, adapter_weights=weights)
        except TypeError:
            target.set_adapters(names, weights=weights)
    else:
        activate_transformer_adapters(target, names, weights)
    print(f'[lora-stack] applied {len(names)} adapter(s): ' + ', '.join((f'{n}@{w:g}' for n, w in zip(names, weights))), flush=True)
    return (names, weights)

def clear_lora_stack(target):
    for m in ('delete_adapters', 'unload_lora_weights'):
        fn = getattr(target, m, None)
        if callable(fn):
            try:
                if m == 'delete_adapters':
                    names = list(getattr(target, 'peft_config', {}) or {})
                    if names:
                        fn(names)
                else:
                    fn()
                return
            except Exception:
                pass

def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def prepare_lora_stack(transformer, specs):
    import torch
    from safetensors import safe_open
    from .lora_family_mapping import adapter_key_plan
    result = []
    for spec in specs or []:
        path, weight, name = _spec(spec)
        if not math.isfinite(weight):
            raise ValueError('LoRA strength must be finite')
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != '.safetensors':
            raise ValueError('LoRA needs an existing local safetensors file')
        pairs, alphas, tensor_map = ({}, {}, {})
        with safe_open(str(path), framework='pt', device='cpu') as source:
            metadata = source.metadata() or {}
            for key in source.keys():
                shape = tuple(source.get_slice(key).get_shape())
                for normalized, shape, rows in adapter_key_plan(transformer, key, shape):
                    match = re.fullmatch('(.+)\\.lora_([AB])\\.default\\.weight', normalized)
                    if not match:
                        if normalized.endswith('.alpha'):
                            value = source.get_tensor(key)
                            if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
                                raise ValueError(f'Invalid LoRA alpha: {key}')
                            module = normalized[:-6]
                            if module in alphas:
                                raise ValueError(f'Duplicate LoRA alpha: {module}')
                            alphas[module] = float(value.item())
                            continue
                        raise ValueError(f'Unsupported LoRA tensor target: {key}')
                    module, side = match.groups()
                    entry = pairs.setdefault(module, {})
                    if side in entry:
                        raise ValueError(f'Duplicate LoRA target: {module}.{side}')
                    entry[side] = {'key': key, 'shape': shape}
                    tensor_map[normalized] = {'source': key, 'rows': rows}
        if not pairs:
            raise ValueError('LoRA contains no compatible linear targets')
        ranks, scales = ({}, {})
        config_path = path.parent / 'adapter_config.json'
        config = {}
        if path.name in {'pytorch_lora_weights.safetensors', 'adapter_model.safetensors'} and config_path.is_file():
            config = json.loads(config_path.read_text(encoding='utf-8'))
            if config.get('use_dora') or config.get('use_rslora') or config.get('bias', 'none') != 'none':
                raise ValueError('This generation path supports plain linear LoRA; DoRA, rank scaling and bias adaptation need their own mapping')
        for module, pair in pairs.items():
            try:
                layer = transformer.get_submodule(module)
            except AttributeError as exc:
                raise ValueError(f'LoRA target is absent from the selected denoiser: {module}') from exc
            if hasattr(layer, 'get_base_layer'):
                layer = layer.get_base_layer()
            if not isinstance(layer, torch.nn.Linear) or set(pair) != {'A', 'B'}:
                raise ValueError(f'LoRA target requires an exact linear A/B pair: {module}')
            a, b = (pair['A']['shape'], pair['B']['shape'])
            if len(a) != 2 or len(b) != 2 or a[0] <= 0 or (a[1] != layer.in_features) or (b != (layer.out_features, a[0])):
                raise ValueError(f'LoRA shape does not match {module}: A={a}, B={b}, base={tuple(layer.weight.shape)}')
            ranks[module] = a[0]
            expected_rank = (config.get('rank_pattern') or {}).get(module, config.get('r', a[0]))
            if int(expected_rank) != a[0]:
                raise ValueError(f'LoRA configuration rank does not match tensor shape: {module}')
            alpha = alphas.get(module, (config.get('alpha_pattern') or {}).get(module, config.get('lora_alpha', metadata.get('lora_alpha', a[0]))))
            scales[module] = float(alpha)
            if not math.isfinite(scales[module]) or scales[module] <= 0:
                raise ValueError(f'LoRA alpha must be finite and positive: {module}')
        if set(alphas) - set(pairs):
            raise ValueError('LoRA alpha names an absent target')
        digest = _file_digest(path)
        identity = hashlib.sha256(json.dumps([digest, weight, ranks, scales], sort_keys=True).encode()).hexdigest()
        result.append({'path': str(path), 'weight': weight, 'name': name, 'sha256': digest, 'identity': identity, 'matched_targets': len(pairs), 'targets': list(pairs), 'rank': next(iter(ranks.values())), 'tensor_map': tensor_map, 'alpha': next(iter(scales.values())), 'rank_pattern': ranks, 'alpha_pattern': scales})
    return result

def preflight_model_loras(model_root, specs, *, transformer_class='Cosmos3OmniTransformer', transformer_cls=None):
    if not specs:
        return []
    import torch
    import diffusers
    config_path = Path(model_root) / 'transformer' / 'config.json'
    if not config_path.is_file():
        raise ValueError("LoRA preflight needs the selected denoiser's local configuration")
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if transformer_cls is not None:
        transformer_class = transformer_cls.__name__
    if config.get('_class_name', transformer_class) != transformer_class:
        raise ValueError('LoRA denoiser configuration does not match the selected family')
    cls = transformer_cls or getattr(diffusers, transformer_class)
    with torch.device('meta'):
        transformer = cls.from_config(config)
    return prepare_lora_stack(transformer, specs)

@contextmanager
def stacked_adapters(transformer, specs, *, device, prepared=None):
    from cosmos_windows.gen_with_lora import apply_lora_stack
    rows = prepared if prepared is not None else prepare_lora_stack(transformer, specs)
    token = 'studio_' + uuid.uuid4().hex
    active = [{**row, 'name': f'{token}_{i}'} for i, row in enumerate(rows) if row['weight'] != 0]
    names = [row['name'] for row in active]
    prior_loaded = getattr(transformer, '_hf_peft_config_loaded', False)
    prior_owned_loaded = '_hf_peft_config_loaded' in transformer.__dict__
    prior_owned_config = 'peft_config' in transformer.__dict__
    prior_trainability = [(name, parameter.requires_grad) for name, parameter in transformer.named_parameters()]
    prior_training = [(module, module.training) for module in transformer.modules()]
    prior_adapters = transformer.active_adapters() if getattr(transformer, 'peft_config', {}) else []
    prior_scales = {name: dict(module.scaling) for name, module in transformer.named_modules() if hasattr(module, 'get_base_layer') and hasattr(module, 'scaling')}
    try:
        if active:
            apply_lora_stack(transformer, active, device=device, strict=True)
        yield rows
    finally:
        attached = [name for name in names if name in (getattr(transformer, 'peft_config', {}) or {})]
        if attached:
            transformer.delete_adapters(attached)
        for module in {module for row in active for module in row['targets']}:
            wrapped = transformer.get_submodule(module)
            if hasattr(wrapped, 'get_base_layer') and (not getattr(wrapped, 'lora_A', {})) and (not getattr(wrapped, 'lora_B', {})):
                parent, _, leaf = module.rpartition('.')
                owner = transformer.get_submodule(parent) if parent else transformer
                setattr(owner, leaf, wrapped.get_base_layer())
        if active and prior_adapters:
            activate_transformer_adapters(transformer, prior_adapters)
            for name, scales in prior_scales.items():
                transformer.get_submodule(name).scaling.update(scales)
        if not getattr(transformer, 'peft_config', {}):
            if prior_owned_loaded:
                transformer._hf_peft_config_loaded = prior_loaded
            elif '_hf_peft_config_loaded' in transformer.__dict__:
                delattr(transformer, '_hf_peft_config_loaded')
            if not prior_owned_config and 'peft_config' in transformer.__dict__:
                delattr(transformer, 'peft_config')
        for name, requires_grad in prior_trainability:
            transformer.get_parameter(name).requires_grad_(requires_grad)
        for module, training in prior_training:
            module.training = training
