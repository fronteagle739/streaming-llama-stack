# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_layer_ring.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from cosmos_windows.weight_streaming.safetensors_index import SafetensorsIndex, TensorRef
_MOE_MARKER = 'moe_mlp.W_gate'
_ALIGN = 256

def _aln(n: int, a: int=_ALIGN) -> int:
    return (n + a - 1) // a * a

@dataclass(frozen=True)
class LayerPlan:
    layer_idx: int
    is_moe: bool
    nbytes: int
    packed_nbytes: int
    tensor_count: int
    refs: Tuple[TensorRef, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class Magi2LayerRingPlan:
    layers: Tuple[LayerPlan, ...]
    global_bytes: int
    global_tensor_count: int
    vram_budget_bytes: int
    ring_depth: int
    ring_bytes: int
    component: str = ''

    @property
    def layer_ids(self) -> List[int]:
        return [l.layer_idx for l in self.layers]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def moe_layer_ids(self) -> List[int]:
        return [l.layer_idx for l in self.layers if l.is_moe]

    @property
    def largest_layer_bytes(self) -> int:
        return max((l.packed_nbytes for l in self.layers), default=0)

    @property
    def total_layer_bytes(self) -> int:
        return sum((l.nbytes for l in self.layers))

    def bytes_per_pass(self) -> int:
        return self.total_layer_bytes

    def summary(self) -> Dict[str, object]:
        return {'num_layers': self.num_layers, 'moe_layers': len(self.moe_layer_ids), 'dense_layers': self.num_layers - len(self.moe_layer_ids), 'global_bytes': self.global_bytes, 'global_tensors': self.global_tensor_count, 'largest_layer_bytes': self.largest_layer_bytes, 'total_layer_bytes': self.total_layer_bytes, 'vram_budget_bytes': self.vram_budget_bytes, 'ring_depth': self.ring_depth, 'ring_bytes': self.ring_bytes, 'bytes_per_pass': self.bytes_per_pass()}

def classify_layers(index: SafetensorsIndex, *, component: str='') -> Tuple[Dict[int, List[TensorRef]], List[TensorRef]]:
    return index.group_by_layer(component=component)

def plan_layer_ring(index: SafetensorsIndex, *, vram_budget_bytes: int, component: str='', min_ring_depth: int=2, max_ring_depth: Optional[int]=None) -> Magi2LayerRingPlan:
    layer_refs, globals_ = classify_layers(index, component=component)
    if not layer_refs:
        raise ValueError(f"no per-layer tensors found (component={component!r}); the preview checkpoint must expose 'layers.N.' tensor names")
    layers: List[LayerPlan] = []
    for idx in sorted(layer_refs):
        refs = layer_refs[idx]
        is_moe = any((_MOE_MARKER in r.name for r in refs))
        nbytes = sum((r.nbytes for r in refs))
        packed = sum((_aln(r.nbytes) for r in refs))
        layers.append(LayerPlan(layer_idx=idx, is_moe=is_moe, nbytes=nbytes, packed_nbytes=packed, tensor_count=len(refs), refs=tuple(refs)))
    largest = max((l.packed_nbytes for l in layers))
    depth = vram_budget_bytes // largest if largest else 0
    if depth < min_ring_depth:
        raise MemoryError(f'VRAM ring budget {vram_budget_bytes / 1000000000.0:.2f}GB holds {depth} of the largest layer ({largest / 1000000000.0:.2f}GB) but needs >= {min_ring_depth} to double-buffer. Raise the ring budget or stream a finer granularity (per-expert active set).')
    if max_ring_depth is not None:
        depth = min(depth, max_ring_depth)
    depth = min(depth, len(layers))
    return Magi2LayerRingPlan(layers=tuple(layers), global_bytes=sum((r.nbytes for r in globals_)), global_tensor_count=len(globals_), vram_budget_bytes=vram_budget_bytes, ring_depth=depth, ring_bytes=depth * largest, component=component)

def estimate_layer_stream_vs_compute(plan: 'Magi2LayerRingPlan', *, packed_tokens: int, gbps: float=4.3, tflops: float=120.0, num_heads: int=24, head_dim: int=128, cfg_batch: int=2, moe_topk: int=6, moe_heads: int=12, d_head: int=256, d_expert: int=1280) -> Tuple[float, float]:
    n = packed_tokens
    attn_flops = cfg_batch * 2 * (2 * n * n * head_dim * num_heads)
    moe_flops = cfg_batch * (n * moe_heads * moe_topk) * (2 * d_head * d_expert * 3)
    compute_s = (attn_flops + moe_flops) / (tflops * 1000000000000.0)
    stream_s = plan.largest_layer_bytes / (gbps * 1000000000.0)
    return (stream_s, compute_s)

def choose_ring_regime(plan: 'Magi2LayerRingPlan', *, packed_tokens: int, gbps: float=4.3, **kw) -> Tuple[int, str]:
    stream_s, compute_s = estimate_layer_stream_vs_compute(plan, packed_tokens=packed_tokens, gbps=gbps, **kw)
    if stream_s >= compute_s:
        return (2, f'AUTO ring: I/O-bound (stream {stream_s * 1000.0:.0f}ms/layer >= compute {compute_s * 1000.0:.0f}ms/layer @ {packed_tokens} tok) -> depth-2 (prefetch overlap)')
    return (1, f'AUTO ring: compute-bound (compute {compute_s * 1000.0:.0f}ms/layer > stream {stream_s * 1000.0:.0f}ms/layer @ {packed_tokens} tok) -> depth-1 (reclaim ~{plan.largest_layer_bytes / 1000000000.0:.1f}GB slot for activations)')

@dataclass
class PassStats:
    steps: int = 0
    layers_streamed: int = 0
    bytes_streamed: int = 0
    evictions: int = 0
    prefetch_events: int = 0
    max_resident_layers: int = 0
    modeled_seconds: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)

def dry_run_pass(plan: Magi2LayerRingPlan, *, steps: int=1, gbps: float=6.0, reverse_alternate: bool=False) -> PassStats:
    if steps < 1:
        raise ValueError('steps must be >= 1')
    d = plan.ring_depth
    order = list(range(plan.num_layers))
    st = PassStats(steps=steps)
    bytes_per_layer = [l.nbytes for l in plan.layers]
    for step in range(steps):
        seq = list(reversed(order)) if reverse_alternate and step % 2 == 1 else order
        resident: List[int] = []
        for pos in seq[:d]:
            resident.append(pos)
            st.prefetch_events += 1
            st.layers_streamed += 1
            st.bytes_streamed += bytes_per_layer[pos]
        st.max_resident_layers = max(st.max_resident_layers, len(resident))
        for i, pos in enumerate(seq):
            assert pos in resident, 'ring invariant violated: computing a non-resident layer'
            resident.remove(pos)
            st.evictions += 1
            ahead = i + d
            if ahead < len(seq):
                nxt = seq[ahead]
                resident.append(nxt)
                st.prefetch_events += 1
                st.layers_streamed += 1
                st.bytes_streamed += bytes_per_layer[nxt]
                st.max_resident_layers = max(st.max_resident_layers, len(resident))
    st.modeled_seconds = st.bytes_streamed / (gbps * 1000000000.0) if gbps > 0 else 0.0
    return st

def assert_ring_invariant(plan: Magi2LayerRingPlan, *, steps: int=2) -> PassStats:
    st = dry_run_pass(plan, steps=steps, reverse_alternate=True)
    assert st.max_resident_layers <= plan.ring_depth, f'ring peak occupancy {st.max_resident_layers} > ring_depth {plan.ring_depth} (evict-before-prefetch violated)'
    assert st.layers_streamed == steps * plan.num_layers, 'a layer was skipped or double-streamed'
    assert st.evictions == steps * plan.num_layers, 'evict count != streamed count'
    return st

class FakeLayerSource:

    def read_range(self, ref: TensorRef, byte_off: int, nbytes: int, buf) -> int:
        buf[:nbytes] = b'\x00' * nbytes
        return nbytes

    def read_into(self, ref: TensorRef, buf) -> int:
        return self.read_range(ref, 0, ref.nbytes, buf)

class FileLayerSource:

    def __init__(self) -> None:
        self._handles: Dict[str, object] = {}

    def _handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            h = open(path, 'rb', buffering=0)
            self._handles[path] = h
        return h

    def read_range(self, ref: TensorRef, byte_off: int, nbytes: int, buf) -> int:
        h = self._handle(ref.file)
        h.seek(ref.offset + byte_off)
        got = h.readinto(buf[:nbytes])
        if got != nbytes:
            raise RuntimeError(f'short read for {ref.name}[{byte_off}:{byte_off + nbytes}]: {got} != {nbytes}')
        return got

    def read_into(self, ref: TensorRef, buf) -> int:
        return self.read_range(ref, 0, ref.nbytes, buf)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()
_ST_TO_TORCH = {'F64': 'float64', 'F32': 'float32', 'F16': 'float16', 'BF16': 'bfloat16', 'I64': 'int64', 'I32': 'int32', 'I16': 'int16', 'I8': 'int8', 'U8': 'uint8', 'BOOL': 'bool', 'F8_E4M3': 'float8_e4m3fn', 'F8_E5M2': 'float8_e5m2'}

def _torch_dtype(torch, st: str):
    name = _ST_TO_TORCH.get(st)
    if name is None or not hasattr(torch, name):
        return torch.uint8
    return getattr(torch, name)

class Magi2DirectStorageRing:
    transport = 'directstorage'

    def __init__(self, plan: Magi2LayerRingPlan, *, device: str='cuda', n_gpu_buffers: int=1, dll_path: Optional[str]=None, host_staging_buffers: int=2) -> None:
        import os
        import torch
        from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer, _DEFAULT_DLL, WF_MOE_FETCH_ALL
        self._torch = torch
        n_gpu_buffers = int(os.environ.get('MAGI2_DS_GPU_BUFFERS', n_gpu_buffers))
        self._sync_after_wait = os.environ.get('MAGI2_DS_SYNC_AFTER_WAIT', '0') not in ('0', '', 'false')
        self.plan = plan
        self.device = device
        layers = sorted(plan.layers, key=lambda l: l.layer_idx)
        self._t2l: Dict[int, int] = {}
        self._l2t: Dict[int, int] = {}
        layer_groups: Dict[int, list] = {}
        for tid, lp in enumerate(layers):
            layer_groups[tid] = list(lp.refs)
            self._t2l[tid] = lp.layer_idx
            self._l2t[lp.layer_idx] = tid
        self._stream_checked = False
        self.compute_stream_handle = None
        self._ds = DirectStorageStreamer(None, layer_groups=layer_groups, n_gpu_buffers=n_gpu_buffers, device=device, dll_path=dll_path or _DEFAULT_DLL, moe_mode=WF_MOE_FETCH_ALL, host_staging_buffers=host_staging_buffers, record_compute_done=True)
        self.mode = self._ds.mode
        self.pinned_bytes = 0
        self.ring_vram_bytes = len(getattr(self._ds, 'tbufs', [])) * getattr(self._ds, 'cap', 0)

    def new_pass(self, reverse: bool=False) -> None:
        self._ds.new_pass(reverse=reverse)

    def prefetch(self, layer_idx: int):
        return None

    def wait_layer(self, layer_idx: int) -> Dict[str, object]:
        if not self._stream_checked:
            self._stream_checked = True
            try:
                import torch as _t
                h = _t.cuda.current_stream().cuda_stream
                self.compute_stream_handle = int(h)
                if int(h) != 0:
                    msg = f'[MAGI2-DS] PROTECT-REUSE BROKEN: torch is computing on a NON-DEFAULT CUDA stream (handle {int(h)}), but the DS buffer-reuse gate records compute-done on the LEGACY DEFAULT stream. The DMA thread can refill a slot while its weights are still being read -> finite-but-WRONG weights. Set MAGI2_DS_STREAM_STRICT=0 to proceed anyway (output will be suspect).'
                    print(msg, flush=True)
                    if os.environ.get('MAGI2_DS_STREAM_STRICT', '1').strip().lower() not in ('0', 'false', 'no', 'off'):
                        raise RuntimeError(msg)
            except RuntimeError:
                raise
            except Exception:
                pass
        views = self._ds.wait_layer(self._l2t[layer_idx])
        cur = self._torch.cuda.current_stream()
        if cur.cuda_stream != 0:
            cur.wait_stream(self._torch.cuda.default_stream())
        if self._sync_after_wait:
            self._torch.cuda.synchronize()
        return views

    def evict(self, layer_idx: int) -> None:
        cur = self._torch.cuda.current_stream()
        if cur.cuda_stream != 0:
            self._torch.cuda.default_stream().wait_stream(cur)
        self._ds.release_layer(self._l2t[layer_idx])

    def close(self) -> None:
        self._ds.close()

class Magi2LayerRingStreamer:
    transport = 'pinned'
    DEFAULT_HOST_STAGE_BYTES = 64 * 1024 * 1024

    def __init__(self, plan: Magi2LayerRingPlan, *, device: str='cuda', source: Optional[object]=None, host_staging_buffers: int=4, host_stage_bytes: int=DEFAULT_HOST_STAGE_BYTES, async_read: Optional[bool]=None) -> None:
        import os
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Magi2LayerRingStreamer requires CUDA; use dry_run_pass for planning')
        self.torch = torch
        if async_read is None:
            async_read = os.environ.get('MAGI2_PINNED_ASYNC_READ', '1') not in ('0', '', 'false', 'no')
        self._async_read = bool(async_read)
        self.plan = plan
        self.device = device
        self.source = source if source is not None else FileLayerSource()
        slot = plan.largest_layer_bytes
        self.slot_bytes = slot
        self.gpu_ring = [torch.empty(slot, dtype=torch.uint8, device=device) for _ in range(plan.ring_depth)]
        n_stage = max(2, host_staging_buffers)
        self.host_stage_bytes = int(host_stage_bytes)
        self.host_stage = [torch.empty(self.host_stage_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(n_stage)]
        self.copy_stream = torch.cuda.Stream(device=device)
        self.copy_done = [torch.cuda.Event() for _ in range(plan.ring_depth)]
        self.stage_done = [torch.cuda.Event() for _ in range(n_stage)]
        self.slot_evicted = [torch.cuda.Event() for _ in range(plan.ring_depth)]
        self._stage_cursor = 0
        self._slot_of: Dict[int, int] = {}
        self._free_slots: List[int] = list(range(plan.ring_depth))
        self._pass = 0
        self._packed: Dict[int, List[Tuple[TensorRef, int]]] = {}
        for lp in plan.layers:
            off = 0
            entries: List[Tuple[TensorRef, int]] = []
            for ref in lp.refs:
                entries.append((ref, off))
                off = _aln(off + ref.nbytes)
            self._packed[lp.layer_idx] = entries
        self._reader = None
        self._work_q = None
        self._reader_err: Optional[BaseException] = None
        self._layer_ready: Dict[int, 'object'] = {}
        if self._async_read:
            import queue as _queue
            import threading
            self._threading = threading
            self._work_q = _queue.Queue()
            self._reader = threading.Thread(target=self._reader_loop, name='magi2-pinned-reader', daemon=True)
            self._reader.start()

    def _reader_loop(self) -> None:
        while True:
            item = self._work_q.get()
            if item is None:
                self._work_q.task_done()
                return
            layer_idx, slot = item
            try:
                self._stream_layer_into_slot(layer_idx, slot)
            except BaseException as exc:
                self._reader_err = exc
            finally:
                ev = self._layer_ready.get(layer_idx)
                if ev is not None:
                    ev.set()
                self._work_q.task_done()

    def _stream_layer_into_slot(self, layer_idx: int, slot: int) -> None:
        torch = self.torch
        gpu_slot = self.gpu_ring[slot]
        n_stage = len(self.host_stage)
        self.copy_stream.wait_event(self.slot_evicted[slot])
        for ref, dst in self._packed[layer_idx]:
            src_off = 0
            remaining = ref.nbytes
            base = dst
            while remaining > 0:
                chunk = min(remaining, self.host_stage_bytes)
                si = self._stage_cursor % n_stage
                self._stage_cursor += 1
                self.stage_done[si].synchronize()
                pinned = self.host_stage[si]
                host_view = memoryview(pinned.numpy()).cast('B')
                self.source.read_range(ref, src_off, chunk, host_view[:chunk])
                with torch.cuda.stream(self.copy_stream):
                    gpu_slot[base:base + chunk].copy_(pinned[:chunk], non_blocking=True)
                    self.stage_done[si].record(self.copy_stream)
                base += chunk
                src_off += chunk
                remaining -= chunk
        with torch.cuda.stream(self.copy_stream):
            self.copy_done[slot].record(self.copy_stream)

    @property
    def pinned_bytes(self) -> int:
        return len(self.host_stage) * self.host_stage_bytes

    @property
    def ring_vram_bytes(self) -> int:
        return len(self.gpu_ring) * self.slot_bytes

    def new_pass(self, reverse: bool=False) -> None:
        if self._async_read and self._work_q is not None:
            self._work_q.join()
            if self._reader_err is not None:
                err, self._reader_err = (self._reader_err, None)
                raise RuntimeError(f'pinned reader thread failed: {type(err).__name__}: {err}') from err
        self._layer_ready.clear()
        self._pass += 1
        self._slot_of.clear()
        self._free_slots = list(range(self.plan.ring_depth))

    def prefetch(self, layer_idx: int) -> int:
        if layer_idx in self._slot_of:
            return self._slot_of[layer_idx]
        if not self._free_slots:
            raise RuntimeError('layer ring full; evict a computed layer before prefetching')
        slot = self._free_slots.pop(0)
        self._slot_of[layer_idx] = slot
        if self._async_read:
            ev = self._threading.Event()
            self._layer_ready[layer_idx] = ev
            self._work_q.put((layer_idx, slot))
        else:
            self._stream_layer_into_slot(layer_idx, slot)
        return slot

    def wait_layer(self, layer_idx: int) -> Dict[str, object]:
        torch = self.torch
        slot = self._slot_of[layer_idx]
        if self._async_read:
            ev = self._layer_ready.get(layer_idx)
            if ev is not None:
                ev.wait()
            if self._reader_err is not None:
                err, self._reader_err = (self._reader_err, None)
                raise RuntimeError(f'pinned reader thread failed: {type(err).__name__}: {err}') from err
        torch.cuda.current_stream().wait_event(self.copy_done[slot])
        gpu_slot = self.gpu_ring[slot]
        out: Dict[str, object] = {}
        for ref, off in self._packed[layer_idx]:
            dt = _torch_dtype(torch, ref.dtype)
            view = gpu_slot[off:off + ref.nbytes]
            if dt is not torch.uint8:
                view = view.view(dt)
            if ref.shape:
                view = view.reshape(tuple(ref.shape))
            out[ref.name] = view
        return out

    def wait_layer_buffer(self, layer_idx: int):
        slot = self._slot_of[layer_idx]
        self.torch.cuda.current_stream().wait_event(self.copy_done[slot])
        return self.gpu_ring[slot]

    def evict(self, layer_idx: int) -> None:
        slot = self._slot_of.pop(layer_idx, None)
        if slot is not None:
            self.slot_evicted[slot].record(self.torch.cuda.current_stream())
            self._free_slots.append(slot)
        self._layer_ready.pop(layer_idx, None)

    def close(self) -> None:
        if getattr(self, '_async_read', False) and getattr(self, '_work_q', None) is not None:
            try:
                self._work_q.put(None)
                if self._reader is not None:
                    self._reader.join(timeout=30)
            except Exception:
                pass
        if hasattr(self.source, 'close'):
            self.source.close()

def directstorage_available(dll_path: Optional[str]=None) -> bool:
    try:
        from cosmos_windows.weight_streaming.ds_backend import ds_probe, _DEFAULT_DLL
        info = ds_probe(dll_path or _DEFAULT_DLL)
        return bool(info.get('wf_cuda_get_vram_info_ok'))
    except Exception:
        return False

def build_layer_ring_streamer(plan: Magi2LayerRingPlan, *, device: str='cuda', transport: str='auto', emit=None, allow_fallback: bool=True, n_gpu_buffers: int=1, dll_path: Optional[str]=None, source: Optional[object]=None, require_env: str='MAGI2_REQUIRE_DIRECTSTORAGE', **pinned_kwargs):
    import os
    import sys
    is_windows = sys.platform == 'win32'
    if transport == 'pinned':
        if emit is not None:
            emit.emit('transport_selected', transport='pinned', level='info', reason='explicitly requested', message='layer-ring transport = bounded-pinned ring (portable tier)')
        return Magi2LayerRingStreamer(plan, device=device, source=source, **pinned_kwargs)
    if not is_windows:
        if transport == 'directstorage':
            raise RuntimeError("transport='directstorage' is Windows-only (DirectStorage); use 'auto'/'pinned' — the bounded-pinned ring is the designated portable tier on macOS/Linux.")
        if emit is not None:
            emit.emit('transport_selected', transport='pinned', level='info', reason='portable (no DirectStorage on this platform)', message='layer-ring transport = bounded-pinned portable tier (macOS/Metal / Linux, 256MB fixed host, RSS-guarded)')
        return Magi2LayerRingStreamer(plan, device=device, source=source, **pinned_kwargs)
    env_strict = os.environ.get(require_env, '').strip() in ('1', 'true', 'yes', 'on')
    strict = transport == 'directstorage' or env_strict
    reason = None
    try:
        ring = Magi2DirectStorageRing(plan, device=device, n_gpu_buffers=n_gpu_buffers, dll_path=dll_path)
        if getattr(ring, 'mode', None) == 3:
            if emit is not None:
                emit.emit('transport_selected', transport='directstorage', mode=3, level='info', message='layer-ring transport = TRUE DirectStorage NVMe->GPU DMA (zero pageable host RAM in the weight path)')
            return ring
        reason = f"DirectStorage did not engage (effective mode={getattr(ring, 'mode', '?')})"
        ring.close()
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
    if strict or not allow_fallback:
        raise RuntimeError(f"DirectStorage transport required but unavailable: {reason}. (Set {require_env}=0 / transport='auto' to allow the bounded-pinned fallback.)")
    msg = f'DirectStorage unavailable on Windows -> bounded-pinned host staging ({reason})'
    if emit is not None:
        emit.fallback('layer-ring transport', msg)
    else:
        print(f'[MAGI2-RING] LOUD FALLBACK: {msg}', flush=True)
    return Magi2LayerRingStreamer(plan, device=device, source=source, **pinned_kwargs)

def _time_pass(ring, plan: Magi2LayerRingPlan, steps: int) -> Tuple[int, float]:
    import torch
    ids = plan.layer_ids
    nbytes_by_pos = [lp.nbytes for lp in plan.layers]
    is_pinned = getattr(ring, 'transport', '') == 'pinned'
    d = plan.ring_depth
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for _ in range(steps):
        ring.new_pass()
        if is_pinned:
            for pos in range(min(d, len(ids))):
                ring.prefetch(ids[pos])
            for i, lid in enumerate(ids):
                ring.wait_layer(lid)
                total += nbytes_by_pos[i]
                ring.evict(lid)
                ahead = i + d
                if ahead < len(ids):
                    ring.prefetch(ids[ahead])
        else:
            for i, lid in enumerate(ids):
                ring.wait_layer(lid)
                total += nbytes_by_pos[i]
                ring.evict(lid)
    torch.cuda.synchronize()
    return (total, time.perf_counter() - t0)

def throughput_compare(shard_dir: str, *, device: str='cuda', ring_budget_bytes: Optional[int]=None, steps: int=2) -> Dict[str, object]:
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    index = build_index(shard_dir)
    budget = ring_budget_bytes or 2 * max((sum((_aln(r.nbytes) for r in refs)) for _, refs in index.group_by_layer('')[0].items()))
    plan = plan_layer_ring(index, vram_budget_bytes=budget)
    result: Dict[str, object] = {'shard_dir': shard_dir, 'num_layers': plan.num_layers, 'bytes_per_pass': plan.bytes_per_pass(), 'steps': steps}
    try:
        ds = build_layer_ring_streamer(plan, device=device, transport='directstorage')
        try:
            nb, sec = _time_pass(ds, plan, steps)
            result['ds_mode'] = getattr(ds, 'mode', None)
            result['ds_gbps'] = round(nb / 1000000000.0 / sec, 2) if sec > 0 else None
            result['ds_pinned_bytes'] = getattr(ds, 'pinned_bytes', None)
        finally:
            ds.close()
    except Exception as exc:
        result['ds_error'] = f'{type(exc).__name__}: {exc}'
    pin = build_layer_ring_streamer(plan, device=device, transport='pinned', source=FileLayerSource())
    try:
        nb, sec = _time_pass(pin, plan, steps)
        result['pinned_gbps'] = round(nb / 1000000000.0 / sec, 2) if sec > 0 else None
        result['pinned_bytes'] = getattr(pin, 'pinned_bytes', None)
    finally:
        pin.close()
    return result
