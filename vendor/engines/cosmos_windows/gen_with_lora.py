# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gen_with_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, sys

def apply_lora(transformer, adapter_path, rank, targets, device='cuda'):
    import torch
    from peft import LoraConfig
    from safetensors.torch import load_file
    from accelerate.utils import set_module_tensor_to_device
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=targets))
    sd = load_file(adapter_path)
    names = dict(transformer.named_parameters())
    loaded = 0
    for k, v in sd.items():
        if k in names:
            set_module_tensor_to_device(transformer, k, device, value=v.to(torch.float32))
            loaded += 1
    for n, p in list(transformer.named_parameters()):
        if p.device.type == 'meta' and 'lora_' in n:
            set_module_tensor_to_device(transformer, n, device, value=torch.zeros(tuple(p.shape), dtype=torch.float32))
    print(f'[gen-lora] applied adapter: {loaded}/{len(sd)} trained tensors loaded', flush=True)
    return loaded

def _retarget_adapter_key(k, name):
    import re
    k2 = re.sub('(\\.lora_[AB])\\.[^.]+(\\.weight)$', f'\\1.{name}\\2', k)
    if k2 != k:
        return k2
    return re.sub('(\\.lora_[AB])(\\.weight)$', f'\\1.{name}\\2', k)

def _normalize_adapter_key(k, name):
    for prefix in ('diffusion_model.', 'model.diffusion_model.', 'transformer.', 'base_model.model.'):
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    return _retarget_adapter_key(k, name)

def _infer_lora_rank(sd):
    for k, v in sd.items():
        if '.lora_A' in k and getattr(v, 'ndim', 0) == 2:
            return int(v.shape[0])
    return None

def _adapter_name(name, fallback):
    import re
    clean = re.sub('[^0-9A-Za-z_]+', '_', str(name or fallback)).strip('_')
    return clean or fallback

def apply_lora_stack(transformer, specs, rank=None, targets=None, device='cuda', strict=False):
    import torch
    from peft import LoraConfig
    from safetensors.torch import load_file
    from accelerate.utils import set_module_tensor_to_device
    try:
        from cosmos_windows.weight_streaming.lora_stack import _spec
    except Exception:

        def _spec(s):
            if isinstance(s, dict):
                return (s['path'], float(s.get('weight', 1.0)), s.get('name'))
            if isinstance(s, (list, tuple)):
                return (s[0], float(s[1]) if len(s) > 1 else 1.0, s[2] if len(s) > 2 else None)
            return (s.path, float(getattr(s, 'weight', 1.0) or 1.0), getattr(s, 'name', None))
    specs = [s for s in specs or [] if s is not None]
    if not specs:
        return ([], [])
    names, weights, seen = ([], [], set())
    for i, s in enumerate(specs):
        path, weight, name = _spec(s)
        name = _adapter_name(name, f'lora{i}')
        while name in seen:
            name = _adapter_name(f'{name}_{i}', f'lora{i}')
        sd = load_file(path)
        if strict:
            from cosmos_windows.weight_streaming.lora_stack import _file_digest
            if _file_digest(path) != s['sha256']:
                raise ValueError('LoRA content changed after preflight')
            sd = {k: v for k, v in sd.items() if not k.endswith('.alpha')}
            if any((not torch.isfinite(v).all() for v in sd.values())):
                raise ValueError('LoRA weights must be finite')
            if s.get('tensor_map'):
                mapped = {}
                consumed = set()
                for destination, recipe in s['tensor_map'].items():
                    value = sd[recipe['source']]
                    consumed.add(recipe['source'])
                    if recipe['rows'] is not None:
                        start, stop = recipe['rows']
                        value = value[start:stop]
                    mapped[_retarget_adapter_key(destination, name)] = value
                if consumed != set(sd):
                    raise ValueError('LoRA native tensor mapping has incomplete source coverage')
                sd = mapped
        r = (s.get('rank') if isinstance(s, dict) else getattr(s, 'rank', None)) or rank or _infer_lora_rank(sd) or 16
        target_modules = (s.get('targets') if strict else None) or targets or ['to_q', 'to_k', 'to_v', 'to_out']
        transformer.add_adapter(LoraConfig(r=r, lora_alpha=s.get('alpha', r) if strict else r, rank_pattern=s.get('rank_pattern', {}) if strict else {}, alpha_pattern=s.get('alpha_pattern', {}) if strict else {}, init_lora_weights='gaussian', target_modules=target_modules), adapter_name=name)
        tparams = dict(transformer.named_parameters())
        loaded = 0
        for k, v in sd.items():
            tk = k if k in tparams else _normalize_adapter_key(k, name)
            if tk in tparams:
                if strict and tuple(tparams[tk].shape) != tuple(v.shape):
                    raise ValueError(f'LoRA shape changed during attachment: {tk}')
                set_module_tensor_to_device(transformer, tk, device, value=v.to(torch.float32))
                loaded += 1
            elif strict:
                raise ValueError(f'LoRA target was not attached: {k}')
        if strict and loaded != len(sd):
            raise ValueError('LoRA target coverage is incomplete')
        print(f"[gen-lora] stacked adapter '{name}' @{weight:g} r={r}: {loaded}/{len(sd)} tensors", flush=True)
        names.append(name)
        weights.append(weight)
        seen.add(name)
    for n, p in list(transformer.named_parameters()):
        if p.device.type == 'meta' and 'lora_' in n:
            if strict:
                raise ValueError(f'LoRA parameter was not loaded: {n}')
            set_module_tensor_to_device(transformer, n, device, value=torch.zeros(tuple(p.shape), dtype=torch.float32))
    from cosmos_windows.weight_streaming.lora_stack import activate_transformer_adapters
    activate_transformer_adapters(transformer, names, weights)
    print(f"[gen-lora] STACK active: {', '.join((f'{n}@{w:g}' for n, w in zip(names, weights)))}", flush=True)
    return (names, weights)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--out', required=True, help='output DIRECTORY')
    ap.add_argument('--modality', default='t2i')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--no-reasoner', action='store_true', help='adapter was trained WITHOUT --train-reasoner')
    ap.add_argument('--image', default=None, help='seed image for i2v')
    a = ap.parse_args()
    from cosmos_windows.cosmos_cli import Cli, DEFAULTS
    from cosmos_windows.train_lora import _targets
    st = dict(DEFAULTS)
    st.update(model=a.model, width=a.width, height=a.height, num_frames=1, memeff_attn=True)
    cli = Cli(st)
    pipe = cli._get_pipe(need_sound=False)
    apply_lora(pipe.transformer, a.adapter, a.rank, _targets(not a.no_reasoner))
    cli.do(f'/out {a.out}')
    cli.do(f'/set steps {a.steps}')
    cli.generate(a.modality, a.prompt, image_path=a.image)
    print(f'[gen-lora] done -> {a.out}', flush=True)
if __name__ == '__main__':
    sys.exit(main())
