# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/ds_streamed_reader.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import contextlib
from typing import Dict, List, Optional, Set
from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF, STensorInfo
from cosmos_windows.gguf_qat.ds_gguf_stream import GGUFDirectStorageStreamer, _layer_index

class DSStreamedGGUF:

    def __init__(self, first_shard: str, device: str='cuda:0', n_gpu_buffers: int=3):
        self.g = StreamedGGUF(first_shard, device=device)
        self.device = device
        self.n_gpu_buffers = n_gpu_buffers
        self.ds = GGUFDirectStorageStreamer(self.g, device=device, n_gpu_buffers=n_gpu_buffers)
        self._name_layer: Dict[str, int] = {}
        for nm in self.g.names():
            il = _layer_index(nm)
            if il is not None:
                self._name_layer[nm] = il
        self._cur_layer: Optional[int] = None
        self._cur_views: Dict[str, 'object'] = {}
        self._pass_active = False
        self.effective_mode = self.ds.mode
        self._suffix = None
        self._suffix_layers = None
        self._active = None

    def meta(self, key, default=None):
        return self.g.meta(key, default)

    def arch(self):
        return self.g.arch()

    def has(self, name):
        return self.g.has(name)

    def info(self, name):
        return self.g.info(name)

    def names(self):
        return self.g.names()

    @property
    def shards(self):
        return self.g.shards

    def ensure_suffix_streamer(self, suffix_layers):
        sl = [int(x) for x in suffix_layers]
        if self._suffix is not None and self._suffix_layers == sl:
            return
        if self._suffix is not None:
            try:
                self._suffix.close()
            except Exception:
                pass
        self._suffix = GGUFDirectStorageStreamer(self.g, device=self.device, n_gpu_buffers=self.n_gpu_buffers, only_layers=sl)
        self._suffix_layers = sl

    def drop_full_streamer(self):
        if self.ds is not None:
            try:
                self.ds.close()
            except Exception:
                pass
            self.ds = None
            import torch
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @contextlib.contextmanager
    def ds_pass(self, suffix: bool=False, reverse: bool=False):
        streamer = self._suffix if suffix and self._suffix is not None else self.ds
        if streamer is None:
            self._active = None
            self._pass_active = False
            yield self
            return
        self._active = streamer
        streamer.new_pass(reverse=reverse)
        self._pass_active = True
        try:
            yield self
        finally:
            self._pass_active = False
            if self._cur_layer is not None:
                self.ds_exit(self._cur_layer)
            self._active = None

    def ds_enter(self, il: int):
        if not self._pass_active or self._active is None:
            return
        self._cur_views = self._active.wait_layer(il)
        self._cur_layer = il

    def ds_exit(self, il: int):
        if not self._pass_active or self._active is None:
            return
        if self._cur_layer == il:
            self._active.release_layer(il)
            self._cur_layer = None
            self._cur_views = {}

    def raw_bytes_pread(self, name: str):
        if self._cur_layer is not None and self._name_layer.get(name) == self._cur_layer and (name in self._cur_views):
            return self._cur_views[name].clone()
        return self.g.raw_bytes_pread(name)

    def raw_bytes_slice(self, name: str, byte_offset: int, byte_count: int):
        return self.g.raw_bytes_slice(name, byte_offset, byte_count)

    def dequant(self, name, dtype=None):
        from cosmos_windows.gguf_qat.dequant import dequantize
        info = self.g._by_name[name]
        raw = self.raw_bytes_pread(name)
        out = dequantize(raw, info.quant, info.shape)
        del raw
        if dtype is not None:
            out = out.to(dtype)
        return out

    def dequant_2d_hf(self, name, dtype=None):
        from cosmos_windows.gguf_qat.dequant import dequantize
        info = self.g._by_name[name]
        assert len(info.shape) == 2, (name, info.shape)
        raw = self.raw_bytes_pread(name)
        flat = dequantize(raw, info.quant, info.shape).reshape(-1)
        w = flat.reshape(int(info.shape[1]), int(info.shape[0]))
        del raw, flat
        if dtype is not None:
            w = w.to(dtype)
        return w

    def close(self):
        for s in (self._suffix, self.ds):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        try:
            self.g.close()
        except Exception:
            pass
