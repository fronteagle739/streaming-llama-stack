# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/convert_mxfp8_to_bf16.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import struct
import sys
import time
from pathlib import Path
import torch
from safetensors import safe_open
from .wan_backend import dequant_mxfp8
_TORCH_TO_ST = {torch.float64: 'F64', torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16', torch.int64: 'I64', torch.int32: 'I32', torch.int16: 'I16', torch.int8: 'I8', torch.uint8: 'U8', torch.bool: 'BOOL'}

def _raw_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()

def convert(in_file: str, out_file: str, device: str='cuda', block: int=32) -> None:
    in_path, out_path = (Path(in_file), Path(out_file))
    if not in_path.is_file():
        raise FileNotFoundError(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(str(in_path), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        scale_keys = {k for k in keys if k.endswith('.weight_scale')}
        quant_marker_keys = {k for k in keys if k.endswith('.comfy_quant')}
        drop_keys = scale_keys | quant_marker_keys
        out_keys = [k for k in keys if k not in drop_keys]

        def out_dtype(k: str) -> torch.dtype:
            sl = f.get_slice(k)
            src = _st_to_torch(sl.get_dtype())
            if k.endswith('.weight') and k[:-len('.weight')] + '.weight_scale' in scale_keys:
                return torch.bfloat16
            return src
        header: dict = {}
        offset = 0
        plan = []
        for k in out_keys:
            sl = f.get_slice(k)
            shape = list(sl.get_shape())
            odt = out_dtype(k)
            numel = 1
            for s in shape:
                numel *= int(s)
            nbytes = numel * torch.empty(0, dtype=odt).element_size()
            header[k] = {'dtype': _TORCH_TO_ST[odt], 'shape': shape, 'data_offsets': [offset, offset + nbytes]}
            scl = k[:-len('.weight')] + '.weight_scale' if odt == torch.bfloat16 and k.endswith('.weight') else None
            plan.append((k, scl is not None, scl, odt, nbytes))
            offset += nbytes
        header['__metadata__'] = {**{str(a): str(b) for a, b in meta.items()}, 'converted_from': 'mxfp8', 'converter': 'convert_mxfp8_to_bf16'}
        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
        pad = -len(header_json) % 8
        header_json += b' ' * pad
        data_size = offset
        print(f'[mxfp8->bf16] {in_path.name}: {len(out_keys)} tensors, {len(scale_keys)} mxfp8 weights, data {data_size / 1000000000.0:.1f} GB (header {len(header_json)} B)', flush=True)
        t0 = time.time()
        written = 0
        with open(out_path, 'wb') as out:
            out.write(struct.pack('<Q', len(header_json)))
            out.write(header_json)
            for i, (k, is_dq, scl, odt, nbytes) in enumerate(plan):
                t = f.get_tensor(k)
                if is_dq:
                    scale = f.get_tensor(scl)
                    bf = dequant_mxfp8(t.to(device), scale.to(device), block=block, dtype=torch.bfloat16)
                    raw = _raw_bytes(bf)
                    del bf, scale
                else:
                    raw = _raw_bytes(t)
                del t
                if len(raw) != nbytes:
                    raise RuntimeError(f'byte mismatch for {k}: header {nbytes} vs data {len(raw)}')
                out.write(raw)
                written += len(raw)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                if (i + 1) % 100 == 0 or i + 1 == len(plan):
                    print(f'  {i + 1}/{len(plan)}  {written / 1000000000.0:.1f}/{data_size / 1000000000.0:.1f} GB  {written / 1000000000.0 / max(1e-06, time.time() - t0):.2f} GB/s', flush=True)
        print(f'[mxfp8->bf16] wrote {out_path} ({written / 1000000000.0:.1f} GB) in {time.time() - t0:.1f}s', flush=True)

def _st_to_torch(tag: str) -> torch.dtype:
    inv = {v: k for k, v in _TORCH_TO_ST.items()}
    if tag in inv:
        return inv[tag]
    if tag in ('F8_E4M3', 'F8_E5M2'):
        return torch.float8_e4m3fn if tag == 'F8_E4M3' else torch.float8_e5m2
    raise ValueError(f'unsupported safetensors dtype {tag!r}')

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Convert an MXFP8 Wan/Bernini checkpoint to bf16 (fast streaming).')
    ap.add_argument('in_file')
    ap.add_argument('out_file')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--block', type=int, default=32)
    a = ap.parse_args(argv)
    convert(a.in_file, a.out_file, device=a.device, block=a.block)
    return 0
if __name__ == '__main__':
    sys.exit(main())
