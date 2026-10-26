# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations

def bind_component(module, source, label, *, quantize_int8=False, device='cuda:0', skip_blocks=False, block_prefix='transformer_blocks'):
    import hashlib
    import inspect
    from pathlib import Path
    import torch
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    from cosmos_windows.weight_streaming.ds_state_load import _group_refs
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    index = build_index(source)
    expected = {name for name in module.state_dict() if not (skip_blocks and name.startswith(block_prefix + '.'))}
    refs = [ref for name, ref in index.refs.items() if name in expected]
    if not refs:
        raise RuntimeError(f'No local Qwen {label} tensors match {source}')
    groups = _group_refs(refs, 512 << 20)
    bound = set()
    streamer = DirectStorageStreamer(index, layer_groups=groups, n_gpu_buffers=1, device=device, record_compute_done=True)
    try:
        if streamer.mode != 3:
            raise RuntimeError(f'{label} requires DirectStorage mode 3; CPU staging is refused')
        print(f'[status] Qwen phase: {label} DirectStorage mode={streamer.mode}, GPU buffers=1', flush=True)
        transport_source = Path(inspect.getfile(DirectStorageStreamer))
        print(f'[status] Qwen transport source: {transport_source}; sha256={hashlib.sha256(transport_source.read_bytes()).hexdigest()}', flush=True)
        streamer.new_pass()
        for gi in sorted(groups):
            views = streamer.wait_layer(gi)
            for name, value in views.items():
                parent_path, _, leaf = name.rpartition('.')
                parent = module.get_submodule(parent_path) if parent_path else module
                old = getattr(parent, leaf)
                if quantize_int8 and type(old).__name__ == 'Int8Params':
                    import bitsandbytes as bnb
                    data, scales, _ = bnb.functional.int8_vectorwise_quant(value.to(torch.float16).contiguous())
                    parent._parameters[leaf] = bnb.nn.Int8Params(data, requires_grad=False, has_fp16_weights=False, CB=data, SCB=scales)
                else:
                    owned = value.to(dtype=old.dtype).clone()
                    if leaf in parent._parameters:
                        parent._parameters[leaf] = torch.nn.Parameter(owned, requires_grad=False)
                    else:
                        parent._buffers[leaf] = owned
                bound.add(name)
            torch.cuda.synchronize(device)
            streamer.release_layer(gi)
            print(f'[status] DirectStorage {label}: {gi + 1}/{len(groups)} groups loaded', flush=True)
    finally:
        streamer.close()
        torch.cuda.empty_cache()
    if hasattr(module, 'tie_weights'):
        module.tie_weights()
    missing = expected - bound
    unresolved = [n for n, p in module.named_parameters() if p.is_meta and (not (skip_blocks and n.startswith(block_prefix + '.')))]
    if unresolved or any((n not in bound and n != 'lm_head.weight' for n in missing)):
        raise RuntimeError(f'Incomplete Qwen {label} weights: {sorted(missing)}; meta={unresolved}')
    for name, buffer in list(module.named_buffers()):
        if buffer.is_meta:
            raise RuntimeError(f'Uninitialized Qwen {label} buffer: {name}')
        parent_path, _, leaf = name.rpartition('.')
        parent = module.get_submodule(parent_path) if parent_path else module
        parent._buffers[leaf] = buffer.to(device)
