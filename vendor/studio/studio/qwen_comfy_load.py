# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import re
from pathlib import Path
DEFAULT_BASE_REPO = os.environ.get('STUDIO_QWEN_IMAGE21_BASE', '')
_BLOCK = re.compile('^transformer_blocks\\.(\\d+)\\.')

def _block_index(target_key: str) -> int | None:
    match = _BLOCK.match(target_key)
    return int(match.group(1)) if match else None

def is_single_file_checkpoint(model_dir) -> bool:
    p = Path(str(model_dir))
    return p.is_file() and p.suffix.lower() == '.safetensors'

def components_root(model_dir) -> Path:
    p = Path(str(model_dir))
    if is_single_file_checkpoint(p):
        base = Path(DEFAULT_BASE_REPO)
        if not (base / 'model_index.json').is_file():
            raise RuntimeError(f'Qwen-Image-2.1 single-file checkpoint {p} needs a base diffusers repo for its text encoder / VAE / processor / scheduler; set STUDIO_QWEN_IMAGE21_BASE (tried {base}).')
        return base
    return p

def map_comfy_key(src_key: str) -> list[tuple[str, str | None]]:
    if src_key == 'img_mlp.gate_up.weight' or src_key.endswith('.img_mlp.gate_up.weight'):
        stem = src_key[:-len('gate_up.weight')]
        return [(stem + 'gate_layer.weight', 'lo'), (stem + 'proj.weight', 'hi')]
    return [(src_key, None)]

def _comfy_entries(keys) -> dict[str, tuple[str, str | None]]:
    entries: dict[str, tuple[str, str | None]] = {}
    for src_key in keys:
        if src_key.endswith('.weight_scale') or src_key.endswith('.comfy_quant'):
            continue
        for target, half in map_comfy_key(src_key):
            entries[target] = (src_key, half)
    return entries

class QwenComfyBlockWeights:

    def __init__(self, module, model_file, *, device: str='cuda:0', dtype=None, streaming: bool=False, defer_resident: bool=False):
        import torch
        from safetensors import safe_open
        dtype = dtype or torch.bfloat16
        self.module, self.device, self.dtype = (module, device, dtype)
        self.model_file = Path(str(model_file))
        self._streaming = bool(streaming)
        self.binds = self.evictions = 0
        with safe_open(str(self.model_file), framework='pt') as sf:
            keys = set(sf.keys())
        self._keys = keys
        sd = set(module.state_dict().keys())
        groups: dict[int, list[str]] = {}
        for k in keys:
            i = _block_index(k)
            if i is not None:
                groups.setdefault(i, []).append(k)
        if sorted(groups) != list(range(len(module.transformer_blocks))):
            raise RuntimeError(f'Qwen Comfy block indices {sorted(groups)} != {len(module.transformer_blocks)} transformer_blocks')
        self._groups = groups
        self._targets: dict[int, list[tuple[str, str, str | None]]] = {}
        produced: set[str] = set()
        for i, ks in groups.items():
            rows: list[tuple[str, str, str | None]] = []
            for k in ks:
                if k.endswith('.weight_scale') or k.endswith('.comfy_quant'):
                    continue
                for target, half in map_comfy_key(k):
                    if target in sd:
                        rows.append((k, target, half))
                        produced.add(target)
            self._targets[i] = rows
        expected = {n for n in sd if _block_index(n) is not None}
        if produced != expected:
            raise RuntimeError(f'Qwen Comfy block coverage mismatch: missing={sorted(expected - produced)[:6]} extra={sorted(produced - expected)[:6]}')
        self._resident: dict | None = None
        self._sf = None
        self._deferred = bool(defer_resident) and (not self._streaming)
        if not self._streaming and (not self._deferred):
            self.materialize_resident()
        self.handles: list = []
        for i, block in enumerate(module.transformer_blocks):
            self.handles.append(block.register_forward_pre_hook(self._pre(i)))
            self.handles.append(block.register_forward_hook(self._post(i), always_call=True))
        mode = 'per-block disk stream' if self._streaming else 'int8-resident, active block -> bf16'
        print(f'[status] Qwen Comfy blocks: {mode} ({len(self._groups)} blocks)', flush=True)

    def _open(self):
        if self._sf is None:
            from safetensors import safe_open
            self._sf = safe_open(str(self.model_file), framework='pt')
        return self._sf

    def materialize_resident(self) -> None:
        if self._streaming or self._resident is not None:
            return
        sf = self._open()
        resident: dict[int, dict] = {}
        for i, ks in self._groups.items():
            resident[i] = {k: sf.get_tensor(k).to(self.device) for k in ks}
        self._resident = resident
        self._deferred = False

    def evict_resident(self) -> None:
        if self._streaming:
            return
        self._resident = None
        import torch
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def new_pass(self) -> None:
        return None

    def _value_for(self, i: int, src_key: str, dtype):
        if self._resident is not None:
            holder = self._resident[i]
            t = holder[src_key]
            if src_key.endswith('.weight'):
                stem = src_key[:-len('.weight')]
                scale = holder.get(src_key + '_scale')
                if scale is not None:
                    from .krea2_diffusion import _convrot_group_from_comfy_quant, _dequant_streamed
                    cq = holder.get(stem + '.comfy_quant')
                    return _dequant_streamed(t, scale, dtype, convrot_group=_convrot_group_from_comfy_quant(cq))
            return t.to(dtype) if t.is_floating_point() else t.clone()
        from .krea2_diffusion import _read_source_tensor
        return _read_source_tensor(self._open(), self._keys, src_key, dtype, self.device)

    def _pre(self, i: int):

        def bind(_module, _args):
            import torch
            for src_key, target, half in self._targets[i]:
                value = self._value_for(i, src_key, self.dtype)
                if half is not None:
                    cut = value.shape[0] // 2
                    value = value[:cut] if half == 'lo' else value[cut:]
                value = value.contiguous()
                parent_name, _, leaf = target.rpartition('.')
                parent = self.module.get_submodule(parent_name)
                param = torch.nn.Parameter(value, requires_grad=False)
                if leaf in parent._parameters:
                    parent._parameters[leaf] = param
                else:
                    parent._buffers[leaf] = value
            self.binds += 1
        return bind

    def _post(self, i: int):
        names = [target for _s, target, _h in self._targets[i]]

        def evict(_module, _args, output):
            import torch
            for target in names:
                parent_name, _, leaf = target.rpartition('.')
                parent = self.module.get_submodule(parent_name)
                if leaf in parent._parameters:
                    old = parent._parameters[leaf]
                    parent._parameters[leaf] = torch.nn.Parameter(torch.empty_like(old, device='meta'), requires_grad=False)
                elif leaf in parent._buffers:
                    parent._buffers[leaf] = torch.empty_like(parent._buffers[leaf], device='meta')
            self.evictions += 1
            return output
        return evict

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._sf is not None:
            try:
                self._sf.__exit__(None, None, None)
            except Exception:
                pass
            self._sf = None
        self._resident = None

def bind_qwen21_transformer(module, model_file, *, device: str='cuda:0', dtype=None, streaming: bool=False, defer_resident: bool=False):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from .krea2_diffusion import _read_source_tensor
    dtype = dtype or torch.bfloat16
    model_file = Path(str(model_file))
    expected = set(module.state_dict().keys())
    with safe_open(str(model_file), framework='pt') as sf:
        keys = set(sf.keys())
        entries = _comfy_entries(keys)
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        if missing or unexpected:
            raise RuntimeError(f'Qwen-Image-2.1 single-file key mismatch vs QwenImage21Transformer2DModel: missing={missing[:8]} unexpected={unexpected[:8]}')
        bound = 0
        for target in sorted(entries):
            if _block_index(target) is not None:
                continue
            src_key, half = entries[target]
            value = _read_source_tensor(sf, keys, src_key, dtype, device)
            if half is not None:
                cut = value.shape[0] // 2
                value = value[:cut] if half == 'lo' else value[cut:]
            value = value.contiguous()
            if value.is_floating_point():
                value = value.to(dtype)
            set_module_tensor_to_device(module, target, device, value=value)
            bound += 1
    blocks = QwenComfyBlockWeights(module, model_file, device=device, dtype=dtype, streaming=streaming, defer_resident=defer_resident)
    module._studio_block_weights = blocks
    print(f'[status] Qwen ComfyUI transformer: {bound} resident globals + {len(blocks._groups)} blocks from {model_file.name}', flush=True)
    return blocks
