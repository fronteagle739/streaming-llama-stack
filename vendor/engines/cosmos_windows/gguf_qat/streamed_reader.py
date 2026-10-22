# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/streamed_reader.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.reader import _ensure_gguf_py
_ensure_gguf_py()
from gguf import GGUFReader

@dataclass
class STensorInfo:
    name: str
    shard: str
    quant: str
    shape: List[int]
    n_elements: int
    n_bytes: int
    data_offset: int

def shard_paths_from_first(first: str) -> List[str]:
    m = re.search('(.*?)(\\d{5})-of-(\\d{5})\\.gguf$', first)
    if not m:
        return [first]
    base, total = (m.group(1), int(m.group(3)))
    out, missing = ([], [])
    for i in range(1, total + 1):
        p = f'{base}{i:05d}-of-{total:05d}.gguf'
        if os.path.exists(p):
            out.append(p)
        else:
            missing.append(p)
    if missing:
        preview = ', '.join((os.path.basename(p) for p in missing[:3]))
        raise FileNotFoundError(f'incomplete split GGUF: missing {len(missing)}/{total} shard(s): {preview}')
    return out

class StreamedGGUF:

    def __init__(self, first_shard: str, device: str='cuda:0'):
        self.device = device
        self.shards = shard_paths_from_first(first_shard)
        self._meta_reader = GGUFReader(self.shards[0], 'r')
        self._by_name: Dict[str, STensorInfo] = {}
        for p in self.shards:
            rd = GGUFReader(p, 'r')
            for t in rd.tensors:
                self._by_name[t.name] = STensorInfo(name=t.name, shard=p, quant=t.tensor_type.name, shape=[int(x) for x in t.shape], n_elements=int(t.n_elements), n_bytes=int(t.n_bytes), data_offset=int(t.data_offset))
            if p != self.shards[0]:
                del rd
        self._fh: Dict[str, Any] = {}

    def meta(self, key: str, default: Any=None) -> Any:
        f = self._meta_reader.fields.get(key)
        if f is None:
            return default
        try:
            v = f.contents()
        except Exception:
            return default
        if isinstance(v, (bytes, bytearray)):
            return v.decode('utf-8', 'replace')
        return v

    def arch(self) -> str:
        return self.meta('general.architecture')

    def has(self, name: str) -> bool:
        return name in self._by_name

    def info(self, name: str) -> STensorInfo:
        return self._by_name[name]

    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def _handle(self, shard: str):
        fh = self._fh.get(shard)
        if fh is None:
            fh = open(shard, 'rb', buffering=0)
            self._fh[shard] = fh
        return fh
    _READ_CHUNK = 1 << 30

    def raw_bytes_pread(self, name: str):
        import torch
        info = self._by_name[name]
        fh = self._handle(info.shard)
        fh.seek(info.data_offset)
        n = int(info.n_bytes)
        if n <= self._READ_CHUNK:
            buf = fh.read(n)
            if len(buf) != n:
                raise IOError(f'short read for {name!r}: got {len(buf)} want {n}')
            host = np.frombuffer(buf, dtype=np.uint8)
            t = torch.from_numpy(host.copy()).to(self.device, non_blocking=False)
            del host, buf
            return t
        host = np.empty(n, dtype=np.uint8)
        got = 0
        while got < n:
            chunk = fh.read(min(self._READ_CHUNK, n - got))
            if not chunk:
                raise IOError(f'short read for {name!r}: got {got} want {n}')
            m = len(chunk)
            host[got:got + m] = np.frombuffer(chunk, dtype=np.uint8)
            got += m
        t = torch.from_numpy(host).to(self.device, non_blocking=False)
        del host
        return t

    def raw_bytes_slice(self, name: str, byte_offset: int, byte_count: int):
        import torch
        info = self._by_name[name]
        byte_offset, byte_count = (int(byte_offset), int(byte_count))
        if byte_offset < 0 or byte_count < 1 or byte_offset + byte_count > int(info.n_bytes):
            raise ValueError(f'raw slice outside tensor {name}: offset={byte_offset} bytes={byte_count}')
        fh = self._handle(info.shard)
        fh.seek(info.data_offset + byte_offset)
        buf = fh.read(byte_count)
        if len(buf) != byte_count:
            raise IOError(f'short slice read for {name!r}: got {len(buf)} want {byte_count}')
        out = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).copy()).to(self.device, non_blocking=False)
        return out

    def dequant(self, name: str, dtype=None):
        from cosmos_windows.gguf_qat.dequant import dequantize
        info = self._by_name[name]
        raw = self.raw_bytes_pread(name)
        out = dequantize(raw, info.quant, info.shape)
        del raw
        if dtype is not None:
            out = out.to(dtype)
        return out

    def dequant_2d_hf(self, name: str, dtype=None):
        from cosmos_windows.gguf_qat.dequant import dequantize
        info = self._by_name[name]
        assert len(info.shape) == 2, (name, info.shape)
        raw = self.raw_bytes_pread(name)
        flat = dequantize(raw, info.quant, info.shape).reshape(-1)
        w = flat.reshape(int(info.shape[1]), int(info.shape[0]))
        del raw, flat
        if dtype is not None:
            w = w.to(dtype)
        return w

    def close(self):
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass
        self._fh.clear()
        try:
            del self._meta_reader
        except Exception:
            pass
if __name__ == '__main__':
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
    start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '8.0')), label='streamed_reader')
    which = sys.argv[1] if len(sys.argv) > 1 else 'step'
    if which == 'kimi':
        first = ''
        probe = 'blk.1.attn_q_b.weight'
    else:
        first = ''
        probe = 'blk.3.attn_q.weight'
    g = StreamedGGUF(first, device='cuda:0')
    print(f'arch={g.arch()!r} shards={len(g.shards)} tensors={len(g.names())}')
    i = g.info(probe)
    print(f'{probe}: shard={os.path.basename(i.shard)} quant={i.quant} shape={i.shape}')
    w = g.dequant_2d_hf(probe, dtype=None)
    print(f'dequant_2d_hf -> {tuple(w.shape)} {w.dtype} dev={w.device} mean={float(w.float().mean()):.4f}')
    g.close()
    print(f'peak RSS {peak_rss_gb():.3f} GB (current {current_rss_gb():.3f} GB)')
