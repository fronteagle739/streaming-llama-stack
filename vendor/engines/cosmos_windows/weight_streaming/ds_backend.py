# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/ds_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import ctypes as C
import os
from pathlib import Path
if os.environ.get('STUDIO_WF_OVERLAP') == '1':
    os.environ.setdefault('LLAMA_WF_ASYNC_DRAIN', '1')
    os.environ.setdefault('LLAMA_WF_DS_USE_COPY_STREAM', '1')
WF_MODE_MMAP, WF_MODE_DIRECT_IO, WF_MODE_GDS, WF_MODE_DSTORAGE, WF_MODE_AUTO = (0, 1, 2, 3, 4)
WF_MOE_FETCH_ALL, WF_MOE_FETCH_ACTIVE = (0, 1)
WF_POST_HANDOFF_MMAP, WF_POST_KEEP_FETCHER = (0, 1)

class wf_params(C.Structure):
    _fields_ = [('model_path', C.c_char_p), ('split_paths', C.POINTER(C.c_char_p)), ('n_split_files', C.c_int), ('io_block_size', C.c_size_t), ('io_queue_depth', C.c_int), ('io_threads', C.c_int), ('io_direct', C.c_bool), ('prefetch_depth', C.c_int), ('n_gpu_buffers', C.c_int), ('n_ram_buffers', C.c_int), ('max_layer_bytes', C.c_size_t), ('n_sublayers', C.c_int), ('gpu_backend', C.c_void_p), ('cuda_device', C.c_void_p), ('mode', C.c_int), ('moe_mode', C.c_int), ('post_mode', C.c_int), ('n_layers', C.c_int), ('trace', C.c_bool)]

class wf_stats(C.Structure):
    _fields_ = [('nvme_read_bw_gbps', C.c_float), ('ram_to_gpu_bw_gbps', C.c_float), ('end_to_end_bw_gbps', C.c_float), ('total_nvme_read_ms', C.c_float), ('total_staging_ms', C.c_float), ('total_h2d_ms', C.c_float), ('total_stall_ms', C.c_float), ('total_compute_ms', C.c_float), ('bytes_read_nvme', C.c_size_t), ('bytes_uploaded_gpu', C.c_size_t), ('layers_fetched', C.c_int), ('stall_count', C.c_int), ('io_ops', C.c_int), ('experts_skipped', C.c_int), ('expert_bytes_saved', C.c_size_t), ('cache_hits', C.c_uint64), ('cache_vram_hits', C.c_uint64), ('cache_ram_hits', C.c_uint64), ('cache_misses', C.c_uint64), ('cache_bytes_saved', C.c_size_t), ('expert_prefetch_predicted', C.c_uint64), ('expert_prefetch_hits', C.c_uint64), ('expert_prefetch_misses', C.c_uint64), ('expert_prefetch_wasted_bytes', C.c_size_t), ('gds_active', C.c_bool), ('effective_mode', C.c_int)]

def stats_look_sane(st) -> bool:
    return 0 <= st.effective_mode <= 4 and st.bytes_uploaded_gpu < 1 << 50 and (st.bytes_read_nvme < 1 << 50) and (st.layers_fetched >= 0) and (st.io_ops >= 0) and (st.nvme_read_bw_gbps == st.nvme_read_bw_gbps)

class wf_weight_span(C.Structure):
    _fields_ = [('tensor_name', C.c_char_p), ('file_offset', C.c_uint64), ('nbytes', C.c_size_t), ('alloc_size', C.c_size_t), ('mmap_addr', C.c_void_p), ('file_idx', C.c_int), ('is_expert', C.c_bool), ('expert_idx', C.c_int), ('alloc_offset', C.c_size_t)]
_DEFAULT_DLL = os.environ.get('STUDIO_WF_DLL', '')

def load_wf(dll_path: str=_DEFAULT_DLL):
    dll_path = str(dll_path)
    search = [os.path.dirname(dll_path)]
    cuda_path = os.environ.get('CUDA_PATH') or ''
    search.append(os.path.join(cuda_path, 'bin'))
    search.append(os.path.join(cuda_path, 'bin', 'x64'))
    for sub in ('', 'bin'):
        p = os.path.join(os.path.dirname(os.path.dirname(dll_path)), sub)
        search.append(p)
    for d in search:
        if hasattr(os, 'add_dll_directory') and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
    try:
        C.windll.ole32.CoInitializeEx(None, 0)
    except Exception:
        pass
    bindir = os.path.dirname(dll_path)
    for ds in ('dstoragecore.dll', 'dstorage.dll'):
        try:
            C.CDLL(os.path.join(bindir, ds))
        except OSError:
            pass
    lib = C.CDLL(dll_path)
    lib.wf_gds_available.restype = C.c_bool
    lib.wf_gds_available.argtypes = []
    lib.wf_cuda_get_vram_info.restype = C.c_bool
    lib.wf_cuda_get_vram_info.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    lib.wf_create.restype = C.c_void_p
    lib.wf_create.argtypes = [C.POINTER(wf_params)]
    lib.wf_destroy.argtypes = [C.c_void_p]
    lib.wf_register_layer.argtypes = [C.c_void_p, C.c_int, C.POINTER(wf_weight_span), C.c_int, C.c_size_t]
    lib.wf_set_layer_major_mode.argtypes = [C.c_void_p, C.c_bool]
    lib.wf_start.argtypes = [C.c_void_p]
    if hasattr(lib, 'wf_start_reverse'):
        lib.wf_start_reverse.argtypes = [C.c_void_p]
        lib.wf_start_reverse.restype = None
    lib.wf_wait_layer_gpu.restype = C.c_void_p
    lib.wf_wait_layer_gpu.argtypes = [C.c_void_p, C.c_int]
    lib.wf_release_layer.argtypes = [C.c_void_p, C.c_int]
    lib.wf_get_span_gpu_offset.restype = C.c_size_t
    lib.wf_get_span_gpu_offset.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.wf_is_ds_direct.restype = C.c_bool
    lib.wf_is_ds_direct.argtypes = [C.c_void_p]
    lib.wf_get_cuda_device_index.restype = C.c_int
    lib.wf_get_cuda_device_index.argtypes = [C.c_void_p]
    lib.wf_cuda_wait_upload.argtypes = [C.c_void_p, C.c_int]
    if hasattr(lib, 'wf_get_stats'):
        lib.wf_get_stats.argtypes = [C.c_void_p, C.POINTER(wf_stats)]
        lib.wf_get_stats.restype = None
    if hasattr(lib, 'wf_cuda_record_compute_done'):
        lib.wf_cuda_record_compute_done.argtypes = [C.c_void_p, C.c_int]
        lib.wf_cuda_record_compute_done.restype = None
    if hasattr(lib, 'wf_set_active_experts'):
        lib.wf_set_active_experts.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int]
        lib.wf_set_active_experts.restype = None
    return lib
_ALIGN = 256

def _aln(x):
    return (x + _ALIGN - 1) // _ALIGN * _ALIGN

def enforce_ds_mode(streamer, label: str, env_prefix: str) -> None:
    if getattr(streamer, 'mode', None) != 3 and os.environ.get(f'{env_prefix}_ALLOW_NON_DIRECTSTORAGE') != '1':
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"{label} streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3) — R1 requires the optimised NVMe->GPU path. Set {env_prefix}_ALLOW_NON_DIRECTSTORAGE=1 only for diagnostics.")

class DirectStorageStreamer:

    def __init__(self, index, component='transformer', n_gpu_buffers=3, device='cuda', dll_path=_DEFAULT_DLL, moe_mode=WF_MOE_FETCH_ALL, expert_name_re=None, *, layer_groups=None, host_staging_buffers: int=2, record_compute_done: 'Optional[bool]'=None):
        import torch, re as _re
        self.torch = torch
        self.device = device
        self.moe_mode = moe_mode
        self._rc_override = record_compute_done
        self._record_compute_done = None
        self._erx = _re.compile(expert_name_re) if isinstance(expert_name_re, str) else expert_name_re
        self.lib = load_wf(dll_path)
        lib = self.lib
        if moe_mode == WF_MOE_FETCH_ACTIVE and (not hasattr(lib, 'wf_set_active_experts')):
            raise RuntimeError('FETCH_ACTIVE requested but this llama.dll lacks wf_set_active_experts (rebuild the MoE-capable weight-fetcher) — use WF_MOE_FETCH_ALL')
        for fn, rt, at in [('wf_get_n_gpu_bufs', C.c_int, [C.c_void_p]), ('wf_get_gpu_buf_capacity', C.c_size_t, [C.c_void_p]), ('wf_set_gpu_buf_ptr', None, [C.c_void_p, C.c_int, C.c_void_p]), ('wf_get_effective_mode', C.c_int, [C.c_void_p]), ('wf_end_pass', None, [C.c_void_p]), ('wf_stop', None, [C.c_void_p]), ('wf_request_layer', None, [C.c_void_p, C.c_int])]:
            getattr(lib, fn).restype = rt
            getattr(lib, fn).argtypes = at
        layers = layer_groups if layer_groups is not None else index.group_by_layer(component)[0]
        if not layers:
            raise ValueError(f'no streamable layer groups found for component {component!r}')
        expected_ids = list(range(len(layers)))
        if sorted(layers) != expected_ids:
            raise ValueError(f"DirectStorage layer groups must be contiguous zero-based IDs; got {sorted(layers)[:8]}{('...' if len(layers) > 8 else '')}")
        if host_staging_buffers < 1:
            raise ValueError('host_staging_buffers must be at least 1')
        self.n = len(layers)
        shards = sorted({r.file for ls in layers.values() for r in ls})
        self.fidx = {f: i for i, f in enumerate(shards)}
        self.plan = {}
        self.eidx_of = {}
        maxbytes = 0

        def _eidx(ref):
            if self._erx is None:
                return -1
            m = self._erx.search(ref.name)
            return int(m.group(1)) if m else -1
        for il, refs in layers.items():
            tagged = [(r, _eidx(r)) for r in refs]
            ordered = [r for r, e in tagged if e < 0] + [r for r, e in tagged if e >= 0] if self._erx is not None else refs
            off = 0
            sl = []
            for r in ordered:
                sl.append((r, off))
                off = _aln(off + r.nbytes)
                self.eidx_of[r.name] = _eidx(r)
            self.plan[il] = (sl, off)
            maxbytes = max(maxbytes, off)
        p = wf_params()
        p.model_path = shards[0].encode()
        self._sp = (C.c_char_p * len(shards[1:]))(*[s.encode() for s in shards[1:]]) if len(shards) > 1 else None
        p.split_paths = C.cast(self._sp, C.POINTER(C.c_char_p)) if self._sp else None
        p.n_split_files = len(shards) - 1
        p.io_block_size = 512 * 1024
        p.io_queue_depth = 4
        p.io_threads = 8
        p.io_direct = True
        p.prefetch_depth = max(0, n_gpu_buffers - 1)
        p.n_gpu_buffers = n_gpu_buffers
        p.n_ram_buffers = int(host_staging_buffers)
        p.max_layer_bytes = _aln(maxbytes) + _ALIGN
        p.n_sublayers = 1
        p.gpu_backend = None
        p.cuda_device = None
        p.mode = WF_MODE_DSTORAGE
        p.moe_mode = moe_mode
        p.post_mode = WF_POST_KEEP_FETCHER
        p.n_layers = self.n
        p.trace = os.environ.get('WF_TRACE', '0').strip().lower() in ('1', 'true', 'yes', 'on')
        self.ctx = lib.wf_create(C.byref(p))
        if not self.ctx:
            raise RuntimeError('wf_create failed')
        self.mode = lib.wf_get_effective_mode(self.ctx)
        nb = lib.wf_get_n_gpu_bufs(self.ctx)
        cap = lib.wf_get_gpu_buf_capacity(self.ctx)
        self.cap = cap
        self.tbufs = [torch.empty(cap, dtype=torch.uint8, device=device) for _ in range(nb)]
        self.ptr2buf = {}
        for i, b in enumerate(self.tbufs):
            lib.wf_set_gpu_buf_ptr(self.ctx, i, C.c_void_p(b.data_ptr()))
            self.ptr2buf[b.data_ptr()] = b
        for il, (sl, total) in self.plan.items():
            arr = (wf_weight_span * len(sl))()
            for j, (r, ao) in enumerate(sl):
                s = arr[j]
                s.tensor_name = r.name.encode()
                s.file_offset = r.offset
                s.nbytes = r.nbytes
                s.alloc_size = 0
                s.mmap_addr = None
                s.file_idx = self.fidx[r.file]
                eidx = self.eidx_of.get(r.name, -1)
                s.is_expert = eidx >= 0
                s.expert_idx = eidx
                s.alloc_offset = ao
            lib.wf_register_layer(self.ctx, il, arr, len(sl), total)
        self._boundary = os.environ.get('WF_PASS_BOUNDARY', 'cycle').strip().lower()
        if self._boundary not in ('cycle', 'inline'):
            raise ValueError(f"WF_PASS_BOUNDARY must be 'cycle' or 'inline', got {self._boundary!r}")
        self._layer_major = self._boundary == 'cycle'
        lib.wf_set_layer_major_mode(self.ctx, self._layer_major)
        print(f'[DS] pass-boundary mode = {self._boundary} (layer_major={self._layer_major})', flush=True)
        lib.wf_start(self.ctx)
        self._started = True

    def fetch_stats(self) -> dict:
        if not hasattr(self.lib, 'wf_get_stats') or not self.ctx:
            return {}
        st = wf_stats()
        self.lib.wf_get_stats(self.ctx, C.byref(st))
        if not stats_look_sane(st):
            return {'layout_suspect': True}
        return {name: getattr(st, name) for name, _ in wf_stats._fields_}

    def new_pass(self, reverse: bool=False):
        self._pass_count = getattr(self, '_pass_count', 0) + 1
        if getattr(self, '_boundary', 'cycle') == 'inline':
            print(f'[DS] new_pass #{self._pass_count}: INLINE boundary (DLL resets at layer 0)', flush=True)
            return
        print(f'[DS] new_pass #{self._pass_count}: wf_end_pass + wf_start (pipeline reset)', flush=True)
        self.lib.wf_end_pass(self.ctx)
        if reverse and hasattr(self.lib, 'wf_start_reverse'):
            self.lib.wf_start_reverse(self.ctx)
        else:
            if reverse and (not getattr(self, '_warned_no_reverse', False)):
                print('[DS] wf_start_reverse not in DLL yet — backward falls back to forward stream (rebuild llama-weight-fetcher per TRAINING_ROADMAP.md for the full-speed reverse pass)', flush=True)
                self._warned_no_reverse = True
            self.lib.wf_start(self.ctx)

    def wait_layer(self, il):
        base = self.lib.wf_wait_layer_gpu(self.ctx, il)
        self.lib.wf_cuda_wait_upload(self.ctx, il)
        if not base:
            raise RuntimeError(f'wf_wait_layer_gpu({il}) returned null')
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
        for r, ao in self.plan[il][0]:
            off = base_off + ao
            out[r.name] = buf[off:off + r.nbytes].view(_torch_dtype_for(r.dtype)).reshape(r.shape)
        return out

    def release_layer(self, il):
        if self._record_compute_done is None:
            has_sym = hasattr(self.lib, 'wf_cuda_record_compute_done')
            if self._rc_override is not None:
                if self._rc_override and (not has_sym):
                    raise RuntimeError('record_compute_done=True requested but this llama.dll lacks wf_cuda_record_compute_done — the DS buffer-reuse protection gate is unavailable; rebuild the weight-fetcher DLL or use the pinned transport.')
                self._record_compute_done = bool(self._rc_override) and has_sym
            else:
                _rc_default = '1' if os.environ.get('STUDIO_WF_OVERLAP') == '1' else '0'
                self._record_compute_done = os.environ.get('LLAMA_WF_RECORD_COMPUTE_DONE', _rc_default) == '1' and has_sym
        if self._record_compute_done:
            self.lib.wf_cuda_record_compute_done(self.ctx, il)
        self.lib.wf_release_layer(self.ctx, il)

    def set_active_experts(self, il, expert_ids):
        if self.moe_mode != WF_MOE_FETCH_ACTIVE:
            return
        ids = sorted({int(e) for e in expert_ids})
        arr = (C.c_int * max(1, len(ids)))(*(ids or [-1]))
        self.lib.wf_set_active_experts(self.ctx, il, arr, len(ids))

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

def _torch_dtype_for(st):
    from .backends import _torch_dtype
    return _torch_dtype(st)

def ds_probe(dll_path: str=_DEFAULT_DLL) -> dict:
    lib = load_wf(dll_path)
    gds = bool(lib.wf_gds_available())
    free = C.c_size_t(0)
    total = C.c_size_t(0)
    ok = bool(lib.wf_cuda_get_vram_info(C.byref(free), C.byref(total)))
    return {'dll': dll_path, 'wf_gds_available': gds, 'wf_cuda_get_vram_info_ok': ok, 'vram_free_gb': round(free.value / 1000000000.0, 2) if ok else None, 'vram_total_gb': round(total.value / 1000000000.0, 2) if ok else None}
if __name__ == '__main__':
    import json
    print(json.dumps(ds_probe(), indent=2))
