# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/repair_h3_turbo_adaln.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path
DEFAULT_BAD_BLOCKS = range(0, 25)
FLIPPED_COLUMNS = slice(4, 8)
ADALN_WEIGHT = 'blocks.%d.adaln_proj.linear.weight'

def read_header(path: Path) -> tuple[dict, int]:
    with open(path, 'rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
    return (header, 8 + length)

def _tensor_names(header: dict, blocks) -> list:
    names = []
    for block in blocks:
        bare = ADALN_WEIGHT % block
        for candidate in (bare, 'model.diffusion_model.' + bare):
            if candidate in header:
                names.append(candidate)
                break
        else:
            raise KeyError(f'{bare} is not in this checkpoint')
    return names

def repair(source: Path, target: Path, *, blocks=DEFAULT_BAD_BLOCKS, dry_run: bool=False) -> dict:
    import torch
    source, target = (Path(source), Path(target))
    header, data_start = read_header(source)
    names = _tensor_names(header, blocks)
    for name in names:
        info = header[name]
        if info['dtype'] != 'BF16':
            raise ValueError(f"{name} is {info['dtype']}, expected BF16 -- refusing to patch bytes whose layout this was not measured against")
        if len(info['shape']) != 2 or info['shape'][1] != 8:
            raise ValueError(f"{name} has shape {info['shape']}, expected [rows, 8] rank-8 coefficients -- this is not the tensor the repair was derived from")
    if dry_run:
        return {'target': str(target), 'tensors': len(names), 'written': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        shutil.copyfile(source, target)
    patched = 0
    with open(target, 'r+b') as handle:
        for name in names:
            start, stop = header[name]['data_offsets']
            handle.seek(data_start + start)
            raw = bytearray(handle.read(stop - start))
            weight = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(header[name]['shape'])
            fixed = weight.clone()
            fixed[:, FLIPPED_COLUMNS] = -fixed[:, FLIPPED_COLUMNS]
            handle.seek(data_start + start)
            handle.write(fixed.contiguous().view(torch.uint8).numpy().tobytes())
            patched += 1
    return {'target': str(target), 'tensors': patched, 'written': True, 'bytes': target.stat().st_size}

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--blocks', default='0-24', help='inclusive block range to flip, e.g. 0-24')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    lo, _, hi = args.blocks.partition('-')
    blocks = range(int(lo), int(hi or lo) + 1)
    result = repair(args.source, args.target, blocks=blocks, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0
if __name__ == '__main__':
    sys.exit(main())
