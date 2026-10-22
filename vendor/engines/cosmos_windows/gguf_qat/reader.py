# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/reader.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
_GGUF_PY_CANDIDATES = [os.environ.get('GGUF_PY_DIR', ''), '', '']

def _ensure_gguf_py() -> None:
    try:
        import gguf
        return
    except Exception:
        pass
    for cand in _GGUF_PY_CANDIDATES:
        if cand and os.path.isdir(os.path.join(cand, 'gguf')):
            sys.path.insert(0, cand)
            try:
                import gguf
                return
            except Exception:
                continue
    raise ImportError('gguf-py not importable; set GGUF_PY_DIR or install gguf. Tried: ' + repr(_GGUF_PY_CANDIDATES))
_ensure_gguf_py()
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
QUANT_BLOCK_GEOMETRY: Dict[str, Dict[str, int]] = {'F32': {'block_size': 1, 'type_size': 4}, 'F16': {'block_size': 1, 'type_size': 2}, 'BF16': {'block_size': 1, 'type_size': 2}, 'Q8_0': {'block_size': 32, 'type_size': 34}, 'Q4_0': {'block_size': 32, 'type_size': 18}, 'Q4_K': {'block_size': 256, 'type_size': 144}, 'Q5_K': {'block_size': 256, 'type_size': 176}, 'Q6_K': {'block_size': 256, 'type_size': 210}, 'Q3_K': {'block_size': 256, 'type_size': 110}}

@dataclass
class TensorInfo:
    name: str
    quant: str
    shape: List[int]
    n_elements: int
    n_bytes: int
    data_offset: int

class GGUFTensorReader:

    def __init__(self, path: str, device: str='cuda'):
        self.path = path
        self.device = device
        self._reader = GGUFReader(path, 'r')
        self._mm: np.memmap = self._reader.data
        self.tensors: List[TensorInfo] = []
        self._by_name: Dict[str, TensorInfo] = {}
        for t in self._reader.tensors:
            info = TensorInfo(name=t.name, quant=t.tensor_type.name, shape=[int(x) for x in t.shape], n_elements=int(t.n_elements), n_bytes=int(t.n_bytes), data_offset=int(t.data_offset))
            self.tensors.append(info)
            self._by_name[info.name] = info

    @property
    def fields(self):
        return self._reader.fields

    def meta(self, key: str, default: Any=None) -> Any:
        f = self._reader.fields.get(key)
        if f is None:
            return default
        try:
            v = f.contents()
        except Exception:
            return default
        if isinstance(v, (bytes, bytearray)):
            return v.decode('utf-8', 'replace')
        return v

    def meta_dict(self, prefix: Optional[str]=None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in self._reader.fields:
            if prefix and (not k.startswith(prefix)):
                continue
            out[k] = self.meta(k)
        return out

    def info(self, name: str) -> TensorInfo:
        return self._by_name[name]

    def has(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> List[str]:
        return [t.name for t in self.tensors]

    def _window(self, info: TensorInfo) -> np.ndarray:
        start = info.data_offset
        end = start + info.n_bytes
        if end > self._mm.shape[0]:
            raise ValueError(f'tensor {info.name!r} window [{start}:{end}] exceeds file size {self._mm.shape[0]}')
        return self._mm[start:end]

    def raw_bytes(self, name: str):
        import torch
        info = self._by_name[name]
        win = self._window(info)
        host = np.array(win, dtype=np.uint8)
        t = torch.from_numpy(host).to(self.device, non_blocking=False)
        del host
        return t

    def raw_bytes_pread(self, name: str):
        import torch
        info = self._by_name[name]
        with open(self.path, 'rb', buffering=0) as fh:
            fh.seek(info.data_offset)
            buf = fh.read(info.n_bytes)
        if len(buf) != info.n_bytes:
            raise IOError(f'short read for {name!r}: got {len(buf)} want {info.n_bytes}')
        host = np.frombuffer(buf, dtype=np.uint8)
        t = torch.from_numpy(host.copy()).to(self.device, non_blocking=False)
        del host, buf
        return t

    def raw_bytes_numpy(self, name: str) -> np.ndarray:
        info = self._by_name[name]
        return np.array(self._window(info), dtype=np.uint8)

    def close(self):
        try:
            del self._mm
        except Exception:
            pass
        try:
            del self._reader
        except Exception:
            pass

def quant_geometry(quant: str) -> Dict[str, int]:
    if quant not in QUANT_BLOCK_GEOMETRY:
        raise NotImplementedError(f'no block geometry registered for quant {quant!r}')
    return QUANT_BLOCK_GEOMETRY[quant]
if __name__ == '__main__':
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
    start_rss_guard(limit_gb=6.0, label='reader_selftest')
    path = sys.argv[1] if len(sys.argv) > 1 else ''
    r = GGUFTensorReader(path, device='cuda')
    print('arch:', r.meta('general.architecture'), '| n_tensors:', len(r.tensors))
    for nm in ['token_embd.weight', 'blk.0.attn_q.weight', 'blk.0.ffn_down.weight']:
        if r.has(nm):
            ti = r.info(nm)
            geo = quant_geometry(ti.quant)
            nblocks = ti.n_elements // geo['block_size']
            expect = nblocks * geo['type_size']
            ok = expect == ti.n_bytes
            print(f'  {nm}: {ti.quant} shape={ti.shape} n_bytes={ti.n_bytes} (geom-expect {expect}, match={ok}) offset={ti.data_offset}')
    g = r.raw_bytes('blk.0.attn_q.weight')
    print(f'  fetched blk.0.attn_q.weight to GPU: {tuple(g.shape)} {g.dtype} dev={g.device}')
    print(f'RSS now {current_rss_gb():.3f} GB (peak {peak_rss_gb():.3f} GB)')
    r.close()
    print('reader self-check OK')
