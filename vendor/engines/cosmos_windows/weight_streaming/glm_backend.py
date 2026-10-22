# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/glm_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import re
from pathlib import Path
GLM_LOAD_POLICY_PROTOCOL = 1

class _PortableGlmStreamer:

    def __init__(self, index, device):
        from .backends import PortableBackend
        self.blocks, _ = glm_group_by_block(index)
        self.backend = PortableBackend(pin_memory=False)
        self.device, self.n, self.mode = (device, len(self.blocks), 'portable')
        self.closed = False

    def new_pass(self):
        if self.closed:
            raise RuntimeError('GLM block stream is closed')

    def wait_layer(self, index):
        self.new_pass()
        return {ref.name: self.backend.read_tensor(ref, self.device) for ref in self.blocks[index]}

    def release_layer(self, index):
        pass

    def close(self):
        self.closed = True
        close = getattr(self.backend, 'close', None)
        if close:
            close()

def _nonlazy_capacity(layer_refs, global_refs, dtype, device, reserve_bytes, n_gpu_buffers):
    import torch
    from .ds_backend import _ALIGN, _aln
    element_size = torch.empty((), dtype=dtype).element_size()

    def target_bytes(ref):
        return ref.numel * element_size if ref.dtype.startswith(('F', 'BF')) else ref.nbytes
    resident = sum((target_bytes(ref) for rows in layer_refs.values() for ref in rows))
    globals_size = sum((target_bytes(ref) for ref in global_refs))
    raw_max = max((sum((_aln(ref.nbytes) for ref in rows)) for rows in layer_refs.values())) + _ALIGN
    upload = max(2, n_gpu_buffers) * raw_max if str(device).startswith('cuda') else 2 * raw_max
    needed = resident + globals_size + upload + int(reserve_bytes)
    if str(device).startswith('cuda'):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
    else:
        import psutil
        free = psutil.virtual_memory().available
        if str(device).startswith('mps'):
            free = min(free, max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    if needed > free:
        raise ValueError(f'GLM streamed/nonlazy needs {needed} free bytes on {device}; only {free} available. Choose streamed/lazy or a device with more capacity.')
    return {'resident_bytes': resident + globals_size, 'upload_budget_bytes': upload, 'reserve_bytes': int(reserve_bytes), 'required_bytes': needed, 'available_bytes': free}
_GLM_BLOCK_RE = re.compile('(?:^|[/.])transformer_blocks\\.(\\d+)\\.')

def glm_group_by_block(idx):
    blocks, globs = ({}, [])
    for name, ref in idx.refs.items():
        m = _GLM_BLOCK_RE.search(name)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    return (blocks, globs)

class _GlmBlockIndex:

    def __init__(self, idx):
        self._idx = idx

    def group_by_layer(self, component=''):
        return glm_group_by_block(self._idx)

    def __getattr__(self, k):
        return getattr(self._idx, k)

def _transformer_dir(model_dir: str | Path) -> Path:
    d = Path(model_dir)
    return d / 'transformer' if (d / 'transformer').is_dir() else d

def load_meta_glm_transformer(model_dir: str | Path, *, dtype=None):
    import torch
    from accelerate import init_empty_weights
    from diffusers import GlmImageTransformer2DModel
    if dtype is None:
        dtype = torch.float16
    cfg = GlmImageTransformer2DModel.load_config(str(_transformer_dir(model_dir)))
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights():
            model = GlmImageTransformer2DModel.from_config(cfg)
    finally:
        torch.set_default_dtype(old)
    return model

def build_streamed_glm_transformer(model_dir: str | Path, *, device: str='cuda', dtype=None, n_gpu_buffers: int=3, allow_non_directstorage: bool=False, load_policy: str='streaming_lazy', reserve_bytes: int=0, should_cancel=None, trace=None):
    import os as _os
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    if dtype is None:
        dtype = torch.float16
    if load_policy not in {'streaming_lazy', 'streaming_nonlazy'}:
        raise ValueError('Unknown GLM streaming policy')
    if reserve_bytes < 0:
        raise ValueError('GLM memory reserve cannot be negative')

    def check_cancel():
        if should_cancel and should_cancel():
            raise RuntimeError('GLM generation cancelled during weight loading')

    def record(event, **details):
        if trace is not None:
            trace.setdefault('events', []).append({'event': event, **details})
    check_cancel()
    tdir = _transformer_dir(model_dir)
    if not tdir.is_dir():
        raise RuntimeError(f'GLM transformer dir not found under {model_dir!r}')
    model = load_meta_glm_transformer(model_dir, dtype=dtype)
    idx = build_index(str(tdir))
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = glm_group_by_block(idx)
    if len(layer_refs) != len(model.transformer_blocks):
        raise RuntimeError(f'GLM streaming keyset mismatch: index has {len(layer_refs)} transformer_blocks groups, module has {len(model.transformer_blocks)} blocks (regex found {sorted(layer_refs)[:4]}...).')
    missing = sorted({r.name for refs in layer_refs.values() for r in refs} - sd_keys)
    if missing:
        raise RuntimeError(f'GLM streamed block tensors have no module home, e.g. {missing[:8]}')
    absent = sorted(sd_keys - set(idx.refs))
    if absent:
        raise RuntimeError(f'GLM checkpoint lacks required tensors, e.g. {absent[:8]}')
    nonlazy = load_policy == 'streaming_nonlazy'
    if nonlazy:
        budget = _nonlazy_capacity(layer_refs, global_refs, dtype, device, reserve_bytes, n_gpu_buffers)
        if trace is not None:
            trace.update(policy=load_policy, capacity=budget)
    elif trace is not None:
        trace['policy'] = load_policy
    pb = PortableBackend(pin_memory=True)
    n_glob = 0
    for ref in global_refs:
        check_cancel()
        if ref.name not in sd_keys:
            continue
        value = pb.read_tensor(ref, device)
        if value.is_floating_point():
            value = value.to(dtype)
        set_module_tensor_to_device(model, ref.name, device, value=value, dtype=dtype if value.is_floating_point() else None)
        n_glob += 1
    cuda = str(device).startswith('cuda')
    streamer = DirectStorageStreamer(_GlmBlockIndex(idx), component='', n_gpu_buffers=n_gpu_buffers, device=device) if cuda else _PortableGlmStreamer(idx, device)
    if cuda and getattr(streamer, 'mode', None) != 3 and (not (allow_non_directstorage or _os.environ.get('GLM_ALLOW_NON_DIRECTSTORAGE') == '1')):
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"GLM streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3). Set GLM_ALLOW_NON_DIRECTSTORAGE=1 only for diagnostics.")

    def _resolve(path: str) -> str:
        parent, _, leaf = path.rpartition('.')
        try:
            module = model.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(module, 'base_layer') else path
    layer_paths = {il: [r.name for r in refs if r.name in sd_keys] for il, refs in layer_refs.items()}

    def make_bind(il: int):

        def bind():
            check_cancel()
            for name, tensor in streamer.wait_layer(il).items():
                if name not in sd_keys:
                    continue
                tensor = tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype, copy=nonlazy)
                set_module_tensor_to_device(model, _resolve(name), device, value=tensor, dtype=dtype if tensor.is_floating_point() else None)
            record('block_load', block=il)
        return bind

    def make_evict(il: int):

        def evict():
            for path in layer_paths[il]:
                set_module_tensor_to_device(model, _resolve(path), 'meta')
            streamer.release_layer(il)
            record('block_evict', block=il)
        return evict
    if nonlazy:
        try:
            for il in range(len(model.transformer_blocks)):
                make_bind(il)()
                if cuda:
                    torch.cuda.synchronize(torch.device(device))
                streamer.release_layer(il)
            streamer.close()
        except BaseException:
            for path in sd_keys:
                set_module_tensor_to_device(model, _resolve(path), 'meta')
            streamer.close()
            raise

        class ResidentBlocks:
            mode, n = ('prefetched', len(model.transformer_blocks))
            closed = False

            def close(self):
                if not self.closed:
                    for path in sd_keys:
                        set_module_tensor_to_device(model, _resolve(path), 'meta')
                    self.closed = True
                    record('prefetched_blocks_release')

            def new_pass(self):
                if self.closed:
                    raise RuntimeError('GLM prefetched blocks have been released')
                record('resident_forward')
        owner = ResidentBlocks()
        model.register_forward_pre_hook(lambda *_: owner.new_pass())
        model._glm_streamer = owner
        model.eval()
        return (model, owner)
    for il, layer in enumerate(model.transformer_blocks):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(*args, _layer=layer, _forward=cls_forward, _bind=bind, _evict=evict, **kwargs):
            _bind()
            try:
                return _forward(_layer, *args, **kwargs)
            finally:
                _evict()
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
    model.register_forward_pre_hook(lambda _m, _a: streamer.new_pass())
    model._glm_streamer = streamer
    model.eval()
    print(f'[GLM-STREAM] {streamer.n} transformer_blocks streamed, {n_glob} globals resident, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)
    return (model, streamer)
