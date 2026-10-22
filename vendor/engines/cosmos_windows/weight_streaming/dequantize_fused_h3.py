# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/dequantize_fused_h3.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .minimax_h3_fused import dequantize_int8_convrot, is_quant_sidecar, parse_comfy_quant
DEFAULT_SHARD_BYTES = 4 * 1024 ** 3

def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def convert(src: Path, dst_dir: Path, *, dtype=torch.bfloat16, device: str='cpu', shard_bytes: int=DEFAULT_SHARD_BYTES, progress=print) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file
    dst_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_size = 0
    shard_no = 0
    total_out = 0
    n_dequantised = 0

    def flush():
        nonlocal shard, shard_size, shard_no, total_out
        if not shard:
            return
        shard_no += 1
        name = f'model-{shard_no:05d}.safetensors'
        save_file(shard, str(dst_dir / name))
        for k in shard:
            weight_map[k] = name
        total_out += shard_size
        progress(f'  wrote {name}  ({shard_size / 2 ** 30:.2f} GB, {len(shard)} tensors)')
        shard = {}
        shard_size = 0
    with safe_open(str(src), framework='pt', device='cpu') as handle:
        names = sorted(handle.keys())
        quantised = {n[:-len('.comfy_quant')] for n in names if n.endswith('.comfy_quant')}
        progress(f'{len(names)} tensors, {len(quantised)} of them quantised')
        for name in names:
            if is_quant_sidecar(name):
                continue
            module = name[:-len('.weight')] if name.endswith('.weight') else name
            if module in quantised and name.endswith('.weight'):
                w = handle.get_tensor(name).to(device)
                s = handle.get_tensor(f'{name}_scale').to(device)
                cfg = parse_comfy_quant(handle.get_tensor(f'{module}.comfy_quant'))
                out = dequantize_int8_convrot(w, s, cfg, dtype=dtype).cpu()
                n_dequantised += 1
                if n_dequantised % 25 == 0:
                    progress(f'  dequantised {n_dequantised}/{len(quantised)}')
            else:
                t = handle.get_tensor(name)
                out = t.to(dtype) if t.is_floating_point() and t.dtype != torch.float32 else t
            shard[name] = out.contiguous()
            shard_size += _nbytes(out)
            if shard_size >= shard_bytes:
                flush()
    flush()
    index = {'metadata': {'total_size': total_out, 'note': 'dequantised from minimax_h3_fused INT8 ConvRot'}, 'weight_map': weight_map}
    (dst_dir / 'model.safetensors.index.json').write_text(json.dumps(index, indent=1), encoding='utf-8')
    progress(f'done: {len(weight_map)} tensors, {total_out / 2 ** 30:.2f} GB across {shard_no} shards')
    return {'tensors': len(weight_map), 'bytes': total_out, 'shards': shard_no, 'dequantised': n_dequantised}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard-gb', type=float, default=4.0)
    a = ap.parse_args()
    convert(a.src, a.dst, device=a.device, shard_bytes=int(a.shard_gb * 2 ** 30))
if __name__ == '__main__':
    main()
