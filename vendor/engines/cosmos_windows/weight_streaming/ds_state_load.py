# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/ds_state_load.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
DS_LOAD_GROUP_BYTES = int(os.environ.get('WS_DS_LOAD_GROUP_BYTES', str(2 << 30)))

def directstorage_usable() -> bool:
    if sys.platform != 'win32':
        return False
    try:
        from cosmos_windows.weight_streaming.ds_backend import ds_probe
        rep = ds_probe()
        return bool(rep.get('ok') or rep.get('available') or rep.get('dll'))
    except Exception:
        return False

def _group_refs(refs: List, group_bytes: int) -> Dict[int, List]:
    groups: Dict[int, List] = {}
    gi, acc = (0, 0)
    for ref in refs:
        if acc and acc + ref.nbytes > group_bytes:
            gi += 1
            acc = 0
        groups.setdefault(gi, []).append(ref)
        acc += ref.nbytes
    return groups

def ds_load_state_dict(source, device: str='cuda', *, name_filter: Optional[Callable[[str], bool]]=None, rename: Optional[Callable[[str], str]]=None, group_bytes: int=DS_LOAD_GROUP_BYTES, emit=None) -> Dict[str, 'object']:
    import torch
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    index = build_index(Path(source)) if not hasattr(source, 'refs') else source
    refs = [r for n, r in index.refs.items() if name_filter is None or name_filter(n)]
    if not refs:
        raise ValueError(f'no tensors selected from {source}')
    groups = _group_refs(refs, group_bytes)
    streamer = DirectStorageStreamer(index, layer_groups=groups, n_gpu_buffers=1, device=device)
    out: Dict[str, object] = {}
    try:
        streamer.new_pass()
        for gi in sorted(groups):
            views = streamer.wait_layer(gi)
            for name, view in views.items():
                out[rename(name) if rename else name] = view.clone()
            streamer.release_layer(gi)
    finally:
        try:
            streamer.close()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if emit is not None:
        emit(event='stage_log', stage='ds_load', message=f'DirectStorage resident load: {len(out)} tensors, {sum((r.nbytes for r in refs)) / 1000000000.0:.2f}GB, ~0 host RSS', tensors=len(out), groups=len(groups))
    return out

def ds_bind_module(module, source, device: str='cuda', *, group_bytes: int=DS_LOAD_GROUP_BYTES, emit=None) -> set:
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    index = build_index(Path(source)) if not hasattr(source, 'refs') else source
    state_keys = set(module.state_dict().keys())
    refs = [r for n, r in index.refs.items() if n in state_keys]
    if not refs:
        raise ValueError(f"no checkpoint tensors match the module's state_dict under {source}")
    groups = _group_refs(refs, group_bytes)
    streamer = DirectStorageStreamer(index, layer_groups=groups, n_gpu_buffers=1, device=device)
    bound: set = set()
    try:
        streamer.new_pass()
        for gi in sorted(groups):
            views = streamer.wait_layer(gi)
            for name, view in views.items():
                if name in state_keys:
                    set_module_tensor_to_device(module, name, device, value=view.clone())
                    bound.add(name)
            streamer.release_layer(gi)
    finally:
        try:
            streamer.close()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if emit is not None:
        emit(event='stage_log', stage='ds_bind', message=f'DirectStorage resident bind: {len(bound)} tensors, {sum((r.nbytes for r in refs)) / 1000000000.0:.2f}GB, ~0 host RSS', tensors=len(bound), groups=len(groups))
    return bound
