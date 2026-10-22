# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/backends.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import mmap
import os
from typing import Optional
from .safetensors_index import TensorRef
try:
    import torch
    _ST_TO_TORCH = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16, 'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool}
    for _n in ('float8_e4m3fn', 'float8_e5m2'):
        pass
    if hasattr(torch, 'float8_e4m3fn'):
        _ST_TO_TORCH['F8_E4M3'] = torch.float8_e4m3fn
    if hasattr(torch, 'float8_e5m2'):
        _ST_TO_TORCH['F8_E5M2'] = torch.float8_e5m2
except Exception:
    torch = None
    _ST_TO_TORCH = {}

def _torch_dtype(st_dtype: str):
    if st_dtype not in _ST_TO_TORCH:
        raise ValueError(f'unsupported safetensors dtype {st_dtype} (have {list(_ST_TO_TORCH)})')
    return _ST_TO_TORCH[st_dtype]

def hf_hub_dirs() -> list:
    import os
    out = []
    cands = [os.environ.get('HF_HUB_CACHE') or '', os.path.join(os.environ.get('HF_HOME') or '', 'hub') if os.environ.get('HF_HOME') else '', '', os.path.expanduser('~/.cache/huggingface/hub')]
    for c in cands:
        if c and os.path.isdir(c) and (c not in out):
            out.append(c)
    return out

class PortableBackend:
    name = 'portable'

    def __init__(self, pin_memory: bool=True):
        self.pin_memory = pin_memory
        self._mm_cache: dict[str, mmap.mmap] = {}
        self._fh_cache: dict[str, object] = {}
        self._stage_buf = None

    def _mm(self, path: str) -> mmap.mmap:
        mm = self._mm_cache.get(path)
        if mm is None:
            fh = open(path, 'rb')
            mm = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ)
            self._fh_cache[path] = fh
            self._mm_cache[path] = mm
        return mm

    def _raw_fh(self, path: str):
        key = ('raw', path)
        fh = self._fh_cache.get(key)
        if fh is None:
            fh = open(path, 'rb', buffering=0)
            self._fh_cache[key] = fh
        return fh

    @staticmethod
    def _readinto_all(fh, buf, nbytes: int) -> int:
        mv = memoryview(buf)
        off = 0
        CHUNK = 1 << 30
        while off < nbytes:
            n = fh.readinto(mv[off:off + min(CHUNK, nbytes - off)])
            if not n:
                break
            off += n
        return off

    def read_bytes(self, ref: TensorRef) -> bytes:
        fh = self._raw_fh(ref.file)
        fh.seek(ref.offset)
        buf = bytearray(ref.nbytes)
        got = self._readinto_all(fh, buf, ref.nbytes)
        if got != ref.nbytes:
            raise RuntimeError(f'direct read_bytes short for {ref.name}: {got} != {ref.nbytes}')
        return bytes(buf)
    BIG_TENSOR_BYTES = int(os.environ.get('WS_BIG_TENSOR_BYTES', str(512 << 20)))
    BOUNDED_STAGE_BYTES = int(os.environ.get('WS_STAGE_BUF_BYTES', str(256 << 20)))

    def _read_tensor_bounded(self, ref: 'TensorRef', device, stream) -> 'torch.Tensor':
        dt = _torch_dtype(ref.dtype)
        dst = torch.empty(ref.shape, dtype=dt, device=device)
        flat = dst.view(-1).view(torch.uint8) if dst.numel() else dst
        stage_n = min(self.BOUNDED_STAGE_BYTES, ref.nbytes)
        if self._stage_buf is None or self._stage_buf.numel() < stage_n:
            self._stage_buf = torch.empty(stage_n, dtype=torch.uint8, pin_memory=True)
        fh = self._raw_fh(ref.file)
        fh.seek(ref.offset)
        mv = memoryview(self._stage_buf.numpy())
        off = 0
        while off < ref.nbytes:
            want = min(stage_n, ref.nbytes - off)
            got = fh.readinto(mv[:want])
            if not got:
                raise RuntimeError(f'bounded read short for {ref.name} at offset {off}')
            src = self._stage_buf[:got]
            if stream is not None:
                with torch.cuda.stream(stream):
                    flat[off:off + got].copy_(src, non_blocking=True)
                    stream.synchronize()
            else:
                flat[off:off + got].copy_(src, non_blocking=True)
                torch.cuda.synchronize()
            off += got
        return dst

    def read_tensor(self, ref: TensorRef, device='cuda', stream=None, direct: bool=True) -> 'torch.Tensor':
        if torch is None:
            raise RuntimeError('torch not available')
        dt = _torch_dtype(ref.dtype)
        if direct and str(device) != 'cpu' and (ref.nbytes >= self.BIG_TENSOR_BYTES):
            return self._read_tensor_bounded(ref, device, stream)
        if direct:
            fh = self._raw_fh(ref.file)
            fh.seek(ref.offset)
            buf = bytearray(ref.nbytes)
            got = self._readinto_all(fh, buf, ref.nbytes)
            if got != ref.nbytes:
                raise RuntimeError(f'direct read short for {ref.name}: {got} != {ref.nbytes}')
            host = torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(ref.shape)
            if str(device) == 'cpu':
                return host
        else:
            mm = self._mm(ref.file)
            host_u8 = torch.frombuffer(mm, dtype=torch.uint8, count=ref.nbytes, offset=ref.offset)
            host = host_u8.view(dt).reshape(ref.shape)
            if str(device) == 'cpu':
                return host.clone()
        if self.pin_memory:
            staged = torch.empty(ref.shape, dtype=dt, pin_memory=True)
            staged.copy_(host)
            host = staged
        if stream is not None:
            with torch.cuda.stream(stream):
                return host.to(device, non_blocking=True)
        return host.to(device, non_blocking=self.pin_memory)

    def read_layer(self, refs, device='cuda', stream=None) -> 'dict':
        if torch is None:
            raise RuntimeError('torch not available')
        files = {r.file for r in refs}
        if len(files) != 1:
            return {r.name: self.read_tensor(r, device, stream) for r in refs}
        f = next(iter(files))
        lo = min((r.offset for r in refs))
        hi = max((r.offset + r.nbytes for r in refs))
        fh = self._raw_fh(f)
        fh.seek(lo)
        span = bytearray(hi - lo)
        got = fh.readinto(span)
        if got != hi - lo:
            raise RuntimeError(f'direct read_layer short: {got} != {hi - lo}')
        host = torch.frombuffer(span, dtype=torch.uint8)
        if str(device) == 'cpu':
            return {r.name: host[r.offset - lo:r.offset - lo + r.nbytes].view(_torch_dtype(r.dtype)).reshape(r.shape).clone() for r in refs}
        if self.pin_memory:
            staged = torch.empty(hi - lo, dtype=torch.uint8, pin_memory=True)
            staged.copy_(host)
            host = staged
        if stream is not None:
            with torch.cuda.stream(stream):
                gpu = host.to(device, non_blocking=True)
        else:
            gpu = host.to(device, non_blocking=self.pin_memory)
        out = {}
        for r in refs:
            s = r.offset - lo
            out[r.name] = gpu[s:s + r.nbytes].view(_torch_dtype(r.dtype)).reshape(r.shape)
        return out

    def close(self):
        for mm in self._mm_cache.values():
            try:
                mm.close()
            except Exception:
                pass
        for fh in self._fh_cache.values():
            try:
                fh.close()
            except Exception:
                pass
        self._mm_cache.clear()
        self._fh_cache.clear()

class CuFileBackend:
    name = 'cufile'

    def __init__(self, **kw):
        raise NotImplementedError('CuFileBackend lands in WS2 (kvikio/cuFile). Use PortableBackend for WS1.')

class DirectStorageBackend:
    name = 'directstorage'

    def __init__(self, **kw):
        raise NotImplementedError('DirectStorageBackend lands in WS2 (DirectStorage). Use PortableBackend for WS1.')

def get_backend(kind: str='auto', **kw):
    if kind in ('portable', 'auto'):
        return PortableBackend(**kw)
    if kind == 'cufile':
        return CuFileBackend(**kw)
    if kind == 'directstorage':
        return DirectStorageBackend(**kw)
    raise ValueError(f'unknown backend {kind}')
if __name__ == '__main__':
    import sys
    from safetensors import safe_open
    from .safetensors_index import build_index
    model = sys.argv[1] if len(sys.argv) > 1 else ''
    idx = build_index(model)
    be = PortableBackend(pin_memory=False)
    names = [n for n in idx.names() if n.startswith('transformer/')]
    small = sorted(names, key=lambda n: idx[n].nbytes)[:5]
    import torch as _t
    ok = 0
    for n in small:
        ref = idx[n]
        ours = be.read_tensor(ref, device='cpu')
        shard = ref.file
        bare = n.split('/', 1)[1] if '/' in n else n
        with safe_open(shard, framework='pt', device='cpu') as f:
            theirs = f.get_tensor(bare)
        same = ours.shape == theirs.shape and ours.dtype == theirs.dtype and _t.equal(ours.to(_t.float32), theirs.to(_t.float32))
        print(f'  {bare:55s} {ref.dtype} {tuple(ref.shape)}  match={same}')
        ok += int(same)
    print(f'PORTABLE READ CROSS-CHECK: {ok}/{len(small)} tensors byte-correct vs safetensors')
    be.close()
