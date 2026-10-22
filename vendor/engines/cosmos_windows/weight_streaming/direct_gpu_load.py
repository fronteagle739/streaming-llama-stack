# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/direct_gpu_load.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import os
import struct
import torch
_DTYPES = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32, 'F64': torch.float64, 'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool}
for _name, _attr in (('F8_E4M3', 'float8_e4m3fn'), ('F8_E5M2', 'float8_e5m2')):
    if hasattr(torch, _attr):
        _DTYPES[_name] = getattr(torch, _attr)
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
LAST_LOAD_STATS: dict = {}

def read_safetensors_header(path: str) -> tuple[dict, int]:
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f'{path}: not a safetensors file (short header)')
        n, = struct.unpack('<Q', raw)
        if n <= 0 or 8 + n > size:
            raise ValueError(f'{path}: corrupt safetensors header length {n} (file {size} B)')
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    return (header, 8 + n)

def safetensors_keys(path: str) -> list[str]:
    return list(read_safetensors_header(path)[0].keys())

class DirectSafetensorsReader:

    def __init__(self, path: str, device, *, chunk_bytes: int=DEFAULT_CHUNK_BYTES):
        self.path = str(path)
        self.device = torch.device(device)
        self.header, self.data_start = read_safetensors_header(self.path)
        self.is_cuda = self.device.type == 'cuda'
        biggest = max((m['data_offsets'][1] - m['data_offsets'][0] for m in self.header.values()), default=0)
        self.buf_len = max(1, min(max(1, int(chunk_bytes)), biggest))
        self._bufs = None
        self._events = [None, None]
        self._slot = 0
        self._f = open(self.path, 'rb', buffering=0)
        self.bytes_read = 0
        self.tensors_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        if self._f is not None:
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
            self._f.close()
            self._f = None
        self._bufs = None

    def keys(self) -> list:
        return list(self.header.keys())

    @property
    def host_buffer_bytes(self) -> int:
        return 0 if self._bufs is None else sum((b.numel() for b in self._bufs))

    def _buffers(self):
        if self._bufs is None:
            self._bufs = [torch.empty(self.buf_len, dtype=torch.uint8, pin_memory=self.is_cuda) for _ in range(2)]
            self._views = [b.numpy() for b in self._bufs]
        return self._bufs

    def get_tensor(self, name: str, dtype=None):
        if name not in self.header:
            raise KeyError(f'{self.path}: tensor {name!r} not in file')
        meta = self.header[name]
        st = meta['dtype']
        if st not in _DTYPES:
            raise TypeError(f'{self.path}: unsupported safetensors dtype {st} for {name}')
        tdtype, shape = (_DTYPES[st], tuple(meta['shape']))
        begin, end = meta['data_offsets']
        nbytes = end - begin
        if nbytes == 0:
            t = torch.empty(shape, dtype=tdtype, device=self.device)
        else:
            bufs = self._buffers()
            raw = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            self._f.seek(self.data_start + begin)
            pos = 0
            while pos < nbytes:
                n = min(self.buf_len, nbytes - pos)
                slot = self._slot
                if self._events[slot] is not None:
                    self._events[slot].synchronize()
                got = self._f.readinto(memoryview(self._views[slot])[:n])
                if got != n:
                    raise IOError(f'{self.path}: short read in {name} ({got} of {n} bytes)')
                raw[pos:pos + n].copy_(bufs[slot][:n], non_blocking=self.is_cuda)
                if self.is_cuda:
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream(self.device))
                    self._events[slot] = ev
                pos += n
                self._slot ^= 1
            t = raw.view(tdtype).reshape(shape)
        if dtype is not None and t.is_floating_point() and (t.dtype != dtype):
            t = t.to(dtype)
        self.bytes_read += nbytes
        self.tensors_read += 1
        return t

def load_safetensors_direct(path: str, device, *, keys=None, dtype=None, chunk_bytes: int=DEFAULT_CHUNK_BYTES) -> dict:
    with DirectSafetensorsReader(path, device, chunk_bytes=chunk_bytes) as r:
        header = r.header
        names = list(header) if keys is None else list(keys)
        missing = [k for k in names if k not in header]
        if missing:
            raise KeyError(f'{path}: tensors not in file: {missing[:5]}')
        out = {}
        for name in sorted(names, key=lambda k: header[k]['data_offsets'][0]):
            out[name] = r.get_tensor(name, dtype=dtype)
        LAST_LOAD_STATS.clear()
        LAST_LOAD_STATS.update({'path': str(path), 'tensors': r.tensors_read, 'bytes': r.bytes_read, 'host_buffer_bytes': r.host_buffer_bytes, 'device': str(r.device)})
    return {k: out[k] for k in names}
