# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/acestep_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from pathlib import Path

def _load_meta_transformer(model_dir: str):
    from accelerate import init_empty_weights
    from diffusers.models.transformers.ace_step_transformer import AceStepTransformer1DModel
    tdir = str(Path(model_dir) / 'transformer')
    cfg = AceStepTransformer1DModel.load_config(tdir)
    with init_empty_weights(include_buffers=False):
        model = AceStepTransformer1DModel.from_config(cfg)
    return model

def keyset_report(model_dir: str) -> dict:
    from .loader import StreamingWeightLoader
    from .safetensors_index import build_index
    model = _load_meta_transformer(model_dir)
    sd_keys = set(model.state_dict().keys())
    idx = build_index(model_dir)
    layer_refs, global_refs = idx.group_by_layer('transformer')
    idx_paths: set[str] = set()
    for refs in layer_refs.values():
        idx_paths.update((StreamingWeightLoader._param_path(r) for r in refs))
    idx_paths.update((StreamingWeightLoader._param_path(r) for r in global_refs))
    return {'n_layers': len(layer_refs), 'n_idx_paths': len(idx_paths), 'n_state_dict': len(sd_keys), 'missing_in_tree': sorted(idx_paths - sd_keys), 'extra_in_tree': sorted(sd_keys - idx_paths)}

def build_streamed_acestep_transformer(model_dir: str, *, device: str='cuda', dtype=None, n_gpu_buffers: int=3):
    import os
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .loader import StreamingWeightLoader
    from .safetensors_index import build_index
    dtype = dtype or torch.bfloat16
    model = _load_meta_transformer(model_dir)
    idx = build_index(model_dir)
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = idx.group_by_layer('transformer')
    layers = getattr(model, 'layers', None)
    if layers is None:
        raise RuntimeError('ACE-Step transformer has no layers attribute to stream')
    if len(layer_refs) != len(layers):
        raise RuntimeError(f'ACE-Step streaming keyset mismatch: checkpoint has {len(layer_refs)} layer groups, module has {len(layers)} layers.')
    missing = sorted({StreamingWeightLoader._param_path(r) for refs in layer_refs.values() for r in refs} - sd_keys)
    if missing:
        raise RuntimeError(f'ACE-Step streamed tensors have no module home, e.g. {missing[:8]}')
    pb = PortableBackend(pin_memory=True)
    n_globals = 0
    for ref in global_refs:
        path = StreamingWeightLoader._param_path(ref)
        if path not in sd_keys:
            continue
        value = pb.read_tensor(ref, device)
        if value.is_floating_point():
            value = value.to(dtype)
        set_module_tensor_to_device(model, path, device, value=value, dtype=dtype if value.is_floating_point() else None)
        n_globals += 1
    streamer = DirectStorageStreamer(idx, component='transformer', n_gpu_buffers=n_gpu_buffers, device=device)
    if getattr(streamer, 'mode', None) != 3 and os.environ.get('ACESTEP_ALLOW_NON_DIRECTSTORAGE') != '1':
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"ACE-Step streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3). Refusing to count this as native streaming. Set ACESTEP_ALLOW_NON_DIRECTSTORAGE=1 only for diagnostics.")
    layer_paths = {il: [StreamingWeightLoader._param_path(r) for r in refs if StreamingWeightLoader._param_path(r) in sd_keys] for il, refs in layer_refs.items()}

    def make_bind(il: int):

        def bind():
            for name, tensor in streamer.wait_layer(il).items():
                path = name.split('/', 1)[1] if '/' in name else name
                if path not in sd_keys:
                    continue
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                set_module_tensor_to_device(model, path, device, value=tensor, dtype=dtype if tensor.is_floating_point() else None)
        return bind

    def make_evict(il: int):

        def evict():
            for path in layer_paths.get(il, []):
                set_module_tensor_to_device(model, path, 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(layers):
        raw_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(*args, _layer=layer, _raw_forward=raw_forward, _bind=bind, _evict=evict, **kwargs):
            _bind()
            try:
                return _raw_forward(_layer, *args, **kwargs)
            finally:
                _evict()
        layer.forward = wrapped
        layer._acestep_raw_forward = raw_forward
        layer._acestep_bind = bind
        layer._acestep_evict = evict
    model.register_forward_pre_hook(lambda _m, _a: streamer.new_pass())
    model._acestep_streamer = streamer
    model.eval()
    print(f'[ACESTEP-STREAM] streamed transformer: {n_globals} globals resident, {len(layer_refs)} layers bind/evict, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)
    return (model, streamer)

def build_acestep_pipeline(model_dir: str, *, device: str='cuda', dtype=None, stream_weights: bool=True, n_gpu_buffers: int=3):
    import torch
    from diffusers import AceStepPipeline
    dtype = dtype or (torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    streamer = None
    if stream_weights and str(device).startswith('cuda'):
        transformer, streamer = build_streamed_acestep_transformer(model_dir, device=device, dtype=dtype, n_gpu_buffers=n_gpu_buffers)
        pipe = AceStepPipeline.from_pretrained(model_dir, transformer=transformer, torch_dtype=dtype, local_files_only=True)
        for name, component in pipe.components.items():
            if name == 'transformer':
                continue
            if hasattr(component, 'to'):
                component.to(device)
        pipe._acestep_streamer = streamer
    else:
        pipe = AceStepPipeline.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)
        pipe = pipe.to(device)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe

def close_acestep_pipeline(pipe) -> None:
    streamer = getattr(pipe, '_acestep_streamer', None)
    close = getattr(streamer, 'close', None)
    if close is not None:
        close()
