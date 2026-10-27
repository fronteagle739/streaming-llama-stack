# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/safetensors_index.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
_LAYER_RE = re.compile('(?:^|[/.])layers\\.(\\d+)\\.')
_DTYPE_SIZE = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'U16': 2, 'U32': 4, 'U64': 8, 'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1}

@dataclass(frozen=True)
class TensorRef:
    name: str
    file: str
    offset: int
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

def _read_header(path: Path) -> Tuple[dict, int]:
    with open(path, 'rb') as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f'{path}: truncated safetensors (no 8-byte length)')
        header_len = struct.unpack('<Q', raw_len)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f'{path}: truncated safetensors header ({len(header_bytes)}/{header_len})')
    header = json.loads(header_bytes)
    return (header, 8 + header_len)

def _index_one_file(path: Path) -> Dict[str, TensorRef]:
    header, data_start = _read_header(path)
    refs: Dict[str, TensorRef] = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        begin, end = meta['data_offsets']
        refs[name] = TensorRef(name=name, file=str(path), offset=data_start + begin, nbytes=end - begin, dtype=meta['dtype'], shape=tuple(meta['shape']))
    return refs

class SafetensorsIndex:

    def __init__(self, refs: Dict[str, TensorRef]):
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, name: str) -> TensorRef:
        return self.refs[name]

    def __contains__(self, name: str) -> bool:
        return name in self.refs

    def names(self) -> List[str]:
        return list(self.refs.keys())

    def total_bytes(self) -> int:
        return sum((r.nbytes for r in self.refs.values()))

    def filter_prefix(self, prefix: str) -> Dict[str, TensorRef]:
        return {n: r for n, r in self.refs.items() if n.startswith(prefix)}

    def group_by_layer(self, component: str='transformer') -> Tuple[Dict[int, List[TensorRef]], List[TensorRef]]:
        layers: Dict[int, List[TensorRef]] = {}
        globs: List[TensorRef] = []
        pre = component + '/' if component else ''
        for name, ref in self.refs.items():
            if pre and (not name.startswith(pre)):
                continue
            m = _LAYER_RE.search(name)
            if m:
                layers.setdefault(int(m.group(1)), []).append(ref)
            else:
                globs.append(ref)
        return (layers, globs)

def build_index(path: str | Path, *, recursive: bool=True) -> SafetensorsIndex:
    path = Path(path)
    refs: Dict[str, TensorRef] = {}
    if path.is_file() and path.suffix == '.safetensors':
        return SafetensorsIndex(_index_one_file(path))
    if not path.is_dir():
        raise FileNotFoundError(f'not a safetensors file or directory: {path}')
    shards = sorted(path.rglob('*.safetensors') if recursive else path.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'no .safetensors under {path}')
    for shard in shards:
        comp = shard.parent.relative_to(path).as_posix()
        prefix = '' if comp in ('', '.') else comp + '/'
        for name, ref in _index_one_file(shard).items():
            refs[prefix + name] = ref
    return SafetensorsIndex(refs)
if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else ''
    idx = build_index(model)
    total = idx.total_bytes()
    print(f'model: {model}')
    print(f'tensors: {len(idx)}')
    print(f'total weight bytes: {total / 1000000000.0:.2f} GB')
    comps: Dict[str, List[int]] = {}
    for n, r in idx.refs.items():
        comp = n.split('/', 1)[0] if '/' in n else '(root)'
        comps.setdefault(comp, [0, 0])
        comps[comp][0] += 1
        comps[comp][1] += r.nbytes
    print('components:')
    for c, (cnt, b) in sorted(comps.items(), key=lambda kv: -kv[1][1]):
        print(f'  {c:20s} {cnt:5d} tensors  {b / 1000000000.0:7.2f} GB')
    sample = [n for n in idx.names() if 'block' in n.lower()][:4]
    print('sample block tensors:')
    for n in sample:
        r = idx[n]
        print(f'  {n}  {r.dtype} {r.shape}  off={r.offset} nbytes={r.nbytes}')
