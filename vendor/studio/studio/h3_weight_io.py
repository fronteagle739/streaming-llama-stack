# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
import sys
from .h3_upscale import require_fast_weights

def cap_cuda(device='cuda:0'):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Native H3 upscale requires the configured CUDA runner; Studio's CPU venv is for tests")
    torch.cuda.set_per_process_memory_fraction(0.98, 0)
    ordinal = torch.device(device).index or 0
    if ordinal:
        torch.cuda.set_per_process_memory_fraction(0.98, ordinal)
    import os
    staging_gb = float(os.environ.get('LLAMA_DS_STAGING_MB', '1300')) / 1024
    free_bytes, total_bytes = torch.cuda.mem_get_info(ordinal)
    if sys.platform == 'win32':
        from .cuda_device_identity import dedicated_free_bytes
        identity = getattr(torch.cuda.get_device_properties(ordinal), 'uuid', None)
        free_bytes = min(free_bytes, dedicated_free_bytes(identity))
    reserve_bytes = (staging_gb + 1) * 1024 ** 3
    if total_bytes <= 0 or free_bytes <= reserve_bytes:
        raise RuntimeError('H3 upscale has insufficient free dedicated VRAM for DirectStorage staging')
    fraction = min(0.98, (free_bytes - reserve_bytes) / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, ordinal)
    current = torch.cuda.get_per_process_memory_fraction(ordinal)
    if current > fraction + 1e-06:
        raise RuntimeError('CUDA allocator cap was not applied')

def weight_index(path):
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    require_fast_weights(path)
    index = build_index(path)
    for ref in index.refs.values():
        if ref.dtype == 'F64':
            raise ValueError(f'Unexpected float64 H3 weight: {ref.name}')
    return index

def stream_tensors(index, device):
    import torch
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer, enforce_ds_mode
    for path in {ref.file for ref in index.refs.values()}:
        require_fast_weights(path)
    groups = {i: [ref] for i, ref in enumerate(index.refs.values())}
    streamer = DirectStorageStreamer(index, component='', n_gpu_buffers=1, device=device, layer_groups=groups, host_staging_buffers=1)
    try:
        enforce_ds_mode(streamer, 'Studio H3 upscale weights', 'MINIMAX_H3')
        if streamer.mode != 3:
            raise RuntimeError('H3 upscale requires DirectStorage mode 3 without fallback')
        for i in groups:
            for name, tensor in streamer.wait_layer(i).items():
                if tensor.device.type != 'cuda':
                    raise RuntimeError(f'H3 weight did not land in VRAM: {name}')
                owned = tensor.clone()
                torch.cuda.synchronize(device)
                yield (name, owned)
            streamer.release_layer(i)
    finally:
        streamer.close()
