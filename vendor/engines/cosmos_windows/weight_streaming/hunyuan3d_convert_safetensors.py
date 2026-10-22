# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/hunyuan3d_convert_safetensors.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
from typing import Iterable
import torch
from safetensors.torch import save_file

def _load_torch(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    except TypeError:
        return torch.load(str(path), map_location='cpu', weights_only=True)

def _tensor_items(state: dict, *, flatten_sections: bool) -> Iterable[tuple[str, torch.Tensor]]:
    for key, value in state.items():
        if torch.is_tensor(value):
            yield (str(key), value)
        elif flatten_sections and isinstance(value, dict):
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue):
                    yield (f'{key}.{subkey}', subvalue)

def _convert(src: Path, dst: Path, *, flatten_sections: bool=False, force: bool=False) -> str:
    if dst.is_file() and (not force):
        return f'skip existing {dst}'
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f'{dst.name}.tmp.{os.getpid()}')
    print(f'[convert] {src} -> {dst}', flush=True)
    state = _load_torch(src)
    tensors = {}
    for key, tensor in _tensor_items(state, flatten_sections=flatten_sections):
        tensors[key] = tensor if tensor.is_contiguous() else tensor.contiguous()
    if not tensors:
        raise RuntimeError(f'No tensors found in {src}')
    save_file(tensors, str(tmp), metadata={'format': 'pt'})
    os.replace(str(tmp), str(dst))
    del tensors, state
    gc.collect()
    return f'wrote {dst}'

def convert_all(weights: Path, *, force: bool=False) -> list[str]:
    jobs = [(weights / 'hunyuan3d-dit-v2-1' / 'model.fp16.ckpt', weights / 'hunyuan3d-dit-v2-1' / 'model.fp16.safetensors', True), (weights / 'hunyuan3d-vae-v2-1' / 'model.fp16.ckpt', weights / 'hunyuan3d-vae-v2-1' / 'model.fp16.safetensors', False), (weights / 'hunyuan3d-paintpbr-v2-1' / 'unet' / 'diffusion_pytorch_model.bin', weights / 'hunyuan3d-paintpbr-v2-1' / 'unet' / 'diffusion_pytorch_model.safetensors', False), (weights / 'hunyuan3d-paintpbr-v2-1' / 'vae' / 'diffusion_pytorch_model.bin', weights / 'hunyuan3d-paintpbr-v2-1' / 'vae' / 'diffusion_pytorch_model.safetensors', False), (weights / 'hunyuan3d-paintpbr-v2-1' / 'text_encoder' / 'pytorch_model.bin', weights / 'hunyuan3d-paintpbr-v2-1' / 'text_encoder' / 'model.safetensors', False)]
    results = []
    for src, dst, flatten in jobs:
        results.append(_convert(src, dst, flatten_sections=flatten, force=force))
    return results

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='', help='Hunyuan3D-2.1 weight root')
    parser.add_argument('--force', action='store_true', help='rewrite safetensors outputs that already exist')
    args = parser.parse_args()
    for line in convert_all(Path(args.weights), force=bool(args.force)):
        print(line, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
