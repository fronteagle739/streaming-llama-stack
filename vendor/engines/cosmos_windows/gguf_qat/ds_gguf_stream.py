# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/ds_gguf_stream.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import ctypes as C
import os
import sys
from typing import Dict, List, Optional
if '' not in sys.path:
    sys.path.insert(0, '')
from cosmos_windows.weight_streaming.ds_backend import load_wf, wf_params, wf_weight_span, WF_MODE_DSTORAGE, WF_MOE_FETCH_ALL, WF_POST_KEEP_FETCHER, _DEFAULT_DLL, _aln, _ALIGN

def _layer_index(name: str) -> Optional[int]:
    if name.startswith('blk.'):
        try:
            return int(name.split('.')[1])
        except (IndexError, ValueError):
            return None
    return None

class GGUFDirectStorageStreamer:

    def __init__(self, g, device: str='cuda:0', n_gpu_buffers: int=3, dll_path: str=_DEFAULT_DLL, only_layers=None):
        import torch
        self.torch = torch
        self.g = g
        self.device = device
        self.lib = load_wf(dll_path)
        lib = self.lib
        for fn, rt, at in [('wf_get_n_gpu_bufs', C.c_int, [C.c_void_p]), ('wf_get_gpu_buf_capacity', C.c_size_t, [C.c_void_p]), ('wf_set_gpu_buf_ptr', None, [C.c_void_p, C.c_int, C.c_void_p]), ('wf_get_effective_mode', C.c_int, [C.c_void_p]), ('wf_end_pass', None, [C.c_void_p]), ('wf_stop', None, [C.c_void_p]), ('wf_request_layer', None, [C.c_void_p, C.c_int])]:
            getattr(lib, fn).restype = rt
            getattr(lib, fn).argtypes = at
        self._has_reverse = hasattr(lib, 'wf_start_reverse')
        if self._has_reverse:
            lib.wf_start_reverse.argtypes = [C.c_void_p]
            lib.wf_start_reverse.restype = None
            self._warned_no_reverse = False
        shards = list(g.shards)
        self.fidx = {f: i for i, f in enumerate(shards)}
        layer_tensors: Dict[int, List[str]] = {}
        for name in g.names():
            il = _layer_index(name)
            if il is None:
                continue
            layer_tensors.setdefault(il, []).append(name)
        all_ids = sorted(layer_tensors.keys())
        if only_layers is not None:
            keep = set((int(x) for x in only_layers))
            self.layer_ids = [il for il in all_ids if il in keep]
            if not self.layer_ids:
                raise ValueError(f'only_layers={sorted(keep)} matched no blk.N layers in the GGUF')
        else:
            self.layer_ids = all_ids
        self.n = len(self.layer_ids)
        self.plan: Dict[int, tuple] = {}
        maxbytes = 0
        for il in self.layer_ids:
            names = sorted(layer_tensors[il])
            off = 0
            sl = []
            for nm in names:
                info = g.info(nm)
                sl.append((nm, off, info.n_bytes))
                off = _aln(off + info.n_bytes)
            self.plan[il] = (sl, off)
            maxbytes = max(maxbytes, off)
        p = wf_params()
        p.model_path = shards[0].encode()
        self._sp = (C.c_char_p * len(shards[1:]))(*[s.encode() for s in shards[1:]]) if len(shards) > 1 else None
        p.split_paths = C.cast(self._sp, C.POINTER(C.c_char_p)) if self._sp else None
        p.n_split_files = len(shards) - 1
        p.io_block_size = 512 * 1024
        p.io_queue_depth = 8
        p.io_threads = 8
        p.io_direct = True
        p.prefetch_depth = max(0, n_gpu_buffers - 1)
        p.n_gpu_buffers = n_gpu_buffers
        p.n_ram_buffers = 2
        p.max_layer_bytes = _aln(maxbytes) + _ALIGN
        p.n_sublayers = 1
        p.gpu_backend = None
        p.cuda_device = None
        p.mode = WF_MODE_DSTORAGE
        p.moe_mode = WF_MOE_FETCH_ALL
        p.post_mode = WF_POST_KEEP_FETCHER
        p.n_layers = self.n
        p.trace = False
        self.ctx = lib.wf_create(C.byref(p))
        if not self.ctx:
            raise RuntimeError('wf_create failed (GGUF DirectStorage streamer)')
        self.mode = lib.wf_get_effective_mode(self.ctx)
        nb = lib.wf_get_n_gpu_bufs(self.ctx)
        cap = lib.wf_get_gpu_buf_capacity(self.ctx)
        self.cap = cap
        self.tbufs = [torch.empty(cap, dtype=torch.uint8, device=device) for _ in range(nb)]
        self.ptr2buf = {}
        for i, b in enumerate(self.tbufs):
            lib.wf_set_gpu_buf_ptr(self.ctx, i, C.c_void_p(b.data_ptr()))
            self.ptr2buf[b.data_ptr()] = b
        self.il_to_wf = {il: k for k, il in enumerate(self.layer_ids)}
        self.wf_to_il = {k: il for il, k in self.il_to_wf.items()}
        for il in self.layer_ids:
            sl, total = self.plan[il]
            arr = (wf_weight_span * len(sl))()
            for j, (nm, ao, nbytes) in enumerate(sl):
                info = g.info(nm)
                s = arr[j]
                s.tensor_name = nm.encode()
                s.file_offset = info.data_offset
                s.nbytes = nbytes
                s.alloc_size = 0
                s.mmap_addr = None
                s.file_idx = self.fidx[info.shard]
                s.is_expert = False
                s.expert_idx = -1
                s.alloc_offset = ao
            lib.wf_register_layer(self.ctx, self.il_to_wf[il], arr, len(sl), total)
        lib.wf_set_layer_major_mode(self.ctx, True)
        lib.wf_start(self.ctx)
        self._started = True

    def new_pass(self, reverse: bool=False):
        self.lib.wf_end_pass(self.ctx)
        if reverse and getattr(self, '_has_reverse', False):
            self.lib.wf_start_reverse(self.ctx)
        else:
            if reverse and (not getattr(self, '_warned_no_reverse', True)):
                print('[DS-GGUF] wf_start_reverse not in DLL — backward recompute falls back to a forward stream (slower, still correct).', flush=True)
                self._warned_no_reverse = True
            self.lib.wf_start(self.ctx)

    def wait_layer(self, il: int) -> Dict[str, 'object']:
        wf_il = self.il_to_wf[il]
        base = self.lib.wf_wait_layer_gpu(self.ctx, wf_il)
        self.lib.wf_cuda_wait_upload(self.ctx, wf_il)
        if not base:
            raise RuntimeError(f'wf_wait_layer_gpu(il={il}) returned null')
        buf = None
        base_off = 0
        for dp, b in self.ptr2buf.items():
            if dp <= base < dp + self.cap:
                buf = b
                base_off = base - dp
                break
        if buf is None:
            raise RuntimeError(f'layer {il} base ptr 0x{base:x} not in torch bufs')
        out = {}
        for nm, ao, nbytes in self.plan[il][0]:
            off = base_off + ao
            out[nm] = buf[off:off + nbytes]
        return out

    def release_layer(self, il: int):
        self.lib.wf_release_layer(self.ctx, self.il_to_wf[il])

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        try:
            if getattr(self, 'ctx', None):
                self.lib.wf_stop(self.ctx)
        except Exception:
            pass
        try:
            if getattr(self, 'ctx', None):
                self.lib.wf_destroy(self.ctx)
        except Exception:
            pass
        self.ctx = None
        self.ptr2buf = {}
        self.tbufs = []
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
if __name__ == '__main__':
    import sys, time
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
    from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
    import torch
    start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '10.0')), label='ds_gguf_bench')
    which = sys.argv[1] if len(sys.argv) > 1 else 'step'
    if which == 'kimi':
        first = ''
    else:
        first = ''
    g = StreamedGGUF(first, device='cuda:0')
    ds = GGUFDirectStorageStreamer(g, device='cuda:0', n_gpu_buffers=3)
    print(f'arch={g.arch()} layers={ds.n} effective_mode={ds.mode} (3=DSTORAGE) buf_cap={ds.cap / 1000000000.0:.2f}GB', flush=True)
    nlayers = int(os.environ.get('BENCH_NLAYERS', str(ds.n)))
    ds.new_pass()
    torch.cuda.synchronize()
    t0 = time.time()
    read = 0
    for il in ds.layer_ids[:nlayers]:
        views = ds.wait_layer(il)
        for nm, v in views.items():
            read += v.numel()
        ds.release_layer(il)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'[DS-GGUF] swept {nlayers} layers, {read / 1000000000.0:.2f}GB in {dt:.2f}s = {read / 1000000000.0 / dt:.3f} GB/s', flush=True)
    print(f'[DS-GGUF] peak RSS {peak_rss_gb():.3f}GB peak VRAM {torch.cuda.max_memory_allocated() / 1000000000.0:.2f}GB', flush=True)
    ds.close()
    g.close()
