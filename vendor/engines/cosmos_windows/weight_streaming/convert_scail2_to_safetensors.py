# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/convert_scail2_to_safetensors.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import importlib.util
import json
import struct
import time
from pathlib import Path
import torch
_TORCH_TO_ST = {torch.float64: 'F64', torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16', torch.int64: 'I64', torch.int32: 'I32', torch.int16: 'I16', torch.int8: 'I8', torch.uint8: 'U8', torch.bool: 'BOOL'}

def _load_upstream_converter(scail_repo: str | Path):
    convert_py = Path(scail_repo) / 'convert.py'
    if not convert_py.is_file():
        raise FileNotFoundError(f'SCAIL convert.py not found: {convert_py}')
    spec = importlib.util.spec_from_file_location('_scail2_upstream_convert', convert_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'could not load converter module from {convert_py}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _target_dtype(dtype_name: str) -> torch.dtype:
    name = (dtype_name or 'bf16').lower()
    if name in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if name in ('fp16', 'float16', 'f16'):
        return torch.float16
    if name in ('fp32', 'float32', 'f32'):
        return torch.float32
    raise ValueError(f'unsupported dtype {dtype_name!r}; use bf16, fp16, or fp32')

def _convert_dtype(t: torch.Tensor, target: torch.dtype) -> torch.Tensor:
    if t.is_floating_point() and t.dtype != target:
        return t.to(target)
    return t

def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()

def _raw_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()

def _align_to(value: int, align: int) -> int:
    if align <= 1:
        return value
    return (value + align - 1) // align * align

def convert_scail2_checkpoint(pt_file: str | Path, out_file: str | Path, *, scail_repo: str | Path='', dtype: str='bf16', align: int=512) -> Path:
    src = Path(pt_file)
    out = Path(out_file)
    if not src.is_file():
        raise FileNotFoundError(src)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    if tmp.exists():
        raise FileExistsError(f'temporary output already exists from a prior run: {tmp}')
    target = _target_dtype(dtype)
    upstream = _load_upstream_converter(scail_repo)
    print(f'[scail-convert] loading mmap checkpoint metadata: {src}', flush=True)
    ckpt = torch.load(str(src), map_location='cpu', mmap=True, weights_only=False)
    state = ckpt['module']
    plan: list[tuple[str, torch.Tensor, torch.dtype, tuple[int, ...], int, int, int]] = []
    seen: set[str] = set()
    for old_key, old_tensor in state.items():
        mapped = upstream.get_new_mappings(old_key, old_tensor)
        for new_key, tensor in mapped.items():
            if new_key in seen:
                raise RuntimeError(f'duplicate converted tensor key: {new_key}')
            seen.add(new_key)
            out_dtype = target if tensor.is_floating_point() else tensor.dtype
            shape = tuple((int(s) for s in tensor.shape))
            nbytes = _tensor_nbytes(shape, out_dtype)
            plan.append((new_key, tensor, out_dtype, shape, nbytes, 0, 0))
    offset = 0
    header: dict[str, object] = {}
    aligned_plan: list[tuple[str, torch.Tensor, torch.dtype, tuple[int, ...], int, int, int]] = []
    for key, tensor, out_dtype, shape, nbytes, _, _ in plan:
        offset = _align_to(offset, align)
        begin, end = (offset, offset + nbytes)
        header[key] = {'dtype': _TORCH_TO_ST[out_dtype], 'shape': list(shape), 'data_offsets': [begin, end]}
        aligned_plan.append((key, tensor, out_dtype, shape, nbytes, begin, end))
        offset = end
    plan = aligned_plan
    header['__metadata__'] = {'source': str(src), 'converted_from': 'zai-org/SCAIL-2 fsdp2_rank_0000_checkpoint.pt', 'converter': 'convert_scail2_to_safetensors', 'dtype': str(target).replace('torch.', '')}
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_json += b' ' * (-(8 + len(header_json)) % max(8, align))
    print(f'[scail-convert] {len(state)} source tensors -> {len(plan)} tensors, {offset / 1000000000.0:.2f} GB data, dtype={target}, align={align}, tmp={tmp}', flush=True)
    t0 = time.time()
    written = 0
    with open(tmp, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header_json)))
        fh.write(header_json)
        cursor = 0
        for i, (key, tensor, out_dtype, shape, nbytes, begin, end) in enumerate(plan, start=1):
            if cursor < begin:
                fh.write(b'\x00' * (begin - cursor))
                cursor = begin
            out_tensor = _convert_dtype(tensor, out_dtype)
            raw = _raw_bytes(out_tensor)
            if len(raw) != nbytes:
                raise RuntimeError(f'byte mismatch for {key}: header {nbytes} vs data {len(raw)}')
            fh.write(raw)
            cursor += len(raw)
            if cursor != end:
                raise RuntimeError(f'cursor mismatch for {key}: cursor {cursor} vs header end {end}')
            written += len(raw)
            del out_tensor, raw
            if i % 50 == 0 or i == len(plan):
                dt = max(time.time() - t0, 1e-06)
                print(f'  {i}/{len(plan)}  {written / 1000000000.0:.2f}/{offset / 1000000000.0:.2f} GB  {written / 1000000000.0 / dt:.2f} GB/s', flush=True)
    tmp.replace(out)
    print(f'[scail-convert] wrote {out} in {time.time() - t0:.1f}s', flush=True)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Convert SCAIL-2 FSDP2 checkpoint to streamable safetensors.')
    ap.add_argument('pt_file')
    ap.add_argument('out_file')
    ap.add_argument('--scail-repo', default='')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--align', type=int, default=512, help='align data section and tensor offsets; 512 keeps Windows DIRECT_IO happy')
    args = ap.parse_args(argv)
    convert_scail2_checkpoint(args.pt_file, args.out_file, scail_repo=args.scail_repo, dtype=args.dtype, align=args.align)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
