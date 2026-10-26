# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import sys
from pathlib import Path
FLOAT8 = ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz')

def _weight_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob('*.safetensors'))
    if not files or any((f.suffix != '.safetensors' for f in files)):
        raise RuntimeError(f'DIRECT_VRAM_UNAVAILABLE: {path} has no complete .safetensors weights; DirectStorage cannot read pickle/.bin/.pth checkpoints')
    return files

def tensors(path, device):
    path = Path(path)
    files = _weight_files(path)
    if sys.platform == 'win32':
        if not str(device).startswith('cuda'):
            raise RuntimeError(f'DIRECT_VRAM_UNAVAILABLE: Windows loads go straight to a CUDA device, not {device!r}')
        repo = os.environ.get('STUDIO_COSMOS_WINDOWS_PATH', '')
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from cosmos_windows.weight_streaming.safetensors_index import SafetensorsIndex, _index_one_file
        from .h3_weight_io import stream_tensors
        refs = {}
        for f in files:
            for name, ref in _index_one_file(f).items():
                if name in refs:
                    raise RuntimeError(f'DIRECT_VRAM_UNAVAILABLE: duplicate tensor {name} in {path}')
                refs[name] = ref
        yield from stream_tensors(SafetensorsIndex(refs), device)
        return
    from safetensors import safe_open
    for f in files:
        with safe_open(str(f), framework='pt', device='cpu', backend='pread') as reader:
            for name in reader.keys():
                yield (name, reader.get_tensor(name).to(device))

def load_into(module, path, device, dtype=None, *, keep_fp32=()):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    expected = set(module.state_dict())
    keep = set(keep_fp32 or ())
    seen = set()
    for name, value in tensors(path, device):
        if name not in expected or name in seen:
            raise RuntimeError(f'DIRECT_VRAM_UNAVAILABLE: unexpected or duplicate tensor {name!r} in {path}')
        if dtype is not None and value.is_floating_point() and (str(value.dtype).split('.')[-1] not in FLOAT8):
            value = value.to(torch.float32 if keep & set(name.split('.')) else dtype)
        set_module_tensor_to_device(module, name, str(device), value=value, dtype=value.dtype)
        seen.add(name)
        del value
    if expected - seen and hasattr(module, 'tie_weights'):
        module.tie_weights()
    missing = [k for k, t in list(module.named_parameters()) + list(module.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(f'DIRECT_VRAM_UNAVAILABLE: {len(missing)} tensors missing from {path}, e.g. {missing[:3]}')
    module.to(str(device))
    target = torch.device(str(device))
    for key, parameter in list(module.named_parameters()) + list(module.named_buffers()):
        if parameter.device.type != target.type or (target.index is not None and parameter.device.index != target.index):
            raise RuntimeError(f'DIRECT_VRAM_UNAVAILABLE: {key} is on {parameter.device}, not {target} (missing from {path}?)')
    return module.eval()

def empty(build):
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        return build()
