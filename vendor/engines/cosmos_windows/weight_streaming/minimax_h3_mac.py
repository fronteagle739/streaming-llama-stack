# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_mac.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class H3MacPreflight:
    platform: str
    mps_available: bool
    checkpoint_complete: bool
    block_count: int
    largest_block_bytes: int
    transport: str = 'bounded_seek_read_tensor_to_mps'
    max_cpu_staging: str = 'one_tensor'
    mmap_weights: bool = False
    pageable_staging: bool = True
    end_to_end_launchable: bool = False
    blocker: str = 'H3 DiT block execution is available on MPS, but Qwen3-VL layer streaming and the video/audio VAE paths still instantiate CUDA DirectStorage transports. PyTorch MPS also cannot provide direct disk-to-Metal DMA.'

def mac_h3_runtime_contract() -> dict:
    return {'backend': 'minimax_h3_mac_mps', 'modalities': ['t2v', 'i2v', 'l2v', 'fflf', 'r2v', 'v2v'], 'variants': {'t2v': 'FL2VA', 'i2v': 'FL2VA', 'l2v': 'FL2VA', 'fflf': 'FL2VA', 'r2v': 'Ref2VA', 'v2v': 'Ref2VA'}, 'weight_transport': {'unit': 'one_dense_dit_block', 'cpu_staging_bound': 'one_tensor', 'mmap': False, 'pageable_staging': True, 'direct_disk_to_metal': False}, 'launchable_stages': ['checkpoint_preflight', 'dit_block_streaming'], 'blocked_stages': ['qwen3_vl_streaming', 'video_vae_streaming', 'audio_vae_streaming'], 'end_to_end_launchable': False}

def h3_variant_for_mode(mode: str) -> str:
    aliases = {'flf2v': 'fflf'}
    mode = aliases.get(str(mode).lower(), str(mode).lower())
    variants = mac_h3_runtime_contract()['variants']
    if mode not in variants:
        raise ValueError(f'unsupported MiniMax H3 mode: {mode!r}')
    return variants[mode]

def mac_h3_preflight(model_dir: str | Path) -> H3MacPreflight:
    import torch
    from .minimax_h3_backend import inspect_h3_transformer
    layout = inspect_h3_transformer(model_dir)
    return H3MacPreflight(platform=sys.platform, mps_available=bool(torch.backends.mps.is_available()), checkpoint_complete=layout.complete, block_count=layout.block_count, largest_block_bytes=layout.largest_block_bytes)

class MPSBoundedBlockStreamer:

    def __init__(self, layer_groups: dict[int, list], *, device: str='mps') -> None:
        import torch
        from .backends import PortableBackend
        if str(device) != 'mps':
            raise ValueError("MPSBoundedBlockStreamer requires device='mps'")
        if not torch.backends.mps.is_available():
            raise RuntimeError('PyTorch MPS is unavailable')
        self.device = device
        self.layer_groups = layer_groups
        self._portable = PortableBackend(pin_memory=False)
        self._bound: dict[str, object] | None = None
        self.bytes_read = 0
        self.layers_read = 0

    def new_pass(self, reverse: bool=False) -> None:
        if reverse:
            raise ValueError('reverse streaming is not implemented for the Mac inference path')
        if self._bound is not None:
            raise RuntimeError('cannot start an H3 pass while a block is still bound')

    def wait_layer(self, block_id: int) -> dict:
        if self._bound is not None:
            raise RuntimeError('only one H3 block may be resident through this transport')
        refs = self.layer_groups[int(block_id)]
        loaded = {}
        for ref in refs:
            loaded[ref.name] = self._portable.read_tensor(ref, device=self.device, direct=True)
            self.bytes_read += int(ref.nbytes)
        self.layers_read += 1
        self._bound = loaded
        return loaded

    def release_layer(self, block_id: int) -> None:
        del block_id
        self._bound = None
        gc.collect()

    def close(self) -> None:
        self._bound = None
        self._portable.close()
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass

def build_streamed_h3_transformer_mps(transformer, index, *, device: str='mps'):
    if str(device) != 'mps':
        raise ValueError("Mac H3 transformer device must be 'mps'")
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .minimax_h3_backend import H3_TRANSFORMER_BLOCKS, group_h3_transformer_blocks, inspect_h3_transformer_index
    blocks, globals_, naming = group_h3_transformer_blocks(index)
    layout = inspect_h3_transformer_index(index)
    if not layout.complete:
        raise ValueError(f'H3 transformer is incomplete: {layout.block_count}/50 blocks; missing {list(layout.missing_block_ids)}; partial {list(layout.incomplete_block_ids)}')
    module_blocks = getattr(transformer, 'blocks', None)
    if module_blocks is None or len(module_blocks) != H3_TRANSFORMER_BLOCKS:
        raise ValueError('Mac H3 transformer must expose 50 source-named blocks')

    def param_path(name: str) -> str:
        return name.split('/', 1)[1] if name.startswith('transformer/') else name
    state_keys = set(transformer.state_dict())
    missing = sorted((param_path(ref.name) for refs in blocks.values() for ref in refs if param_path(ref.name) not in state_keys))
    if missing:
        raise ValueError('H3 checkpoint/model key mismatch before MPS streaming: ' + ', '.join(missing[:8]))
    portable = PortableBackend(pin_memory=False)
    try:
        for ref in globals_:
            path = param_path(ref.name)
            if path in state_keys:
                value = portable.read_tensor(ref, device=device, direct=True)
                set_module_tensor_to_device(transformer, path, device, value=value)
    finally:
        portable.close()
    streamer = MPSBoundedBlockStreamer(blocks, device=device)
    block_paths = {i: [param_path(ref.name) for ref in refs] for i, refs in blocks.items()}
    for block_id, block in enumerate(module_blocks):
        raw_forward = type(block).forward

        def wrapped(*args, _id=block_id, _block=block, _forward=raw_forward, **kwargs):
            loaded = streamer.wait_layer(_id)
            try:
                for name, tensor in loaded.items():
                    set_module_tensor_to_device(transformer, param_path(name), device, value=tensor)
                return _forward(_block, *args, **kwargs)
            finally:
                for path in block_paths[_id]:
                    set_module_tensor_to_device(transformer, path, 'meta')
                streamer.release_layer(_id)
        block.forward = wrapped
        block._h3_raw_forward = raw_forward
    transformer.register_forward_pre_hook(lambda _module, _args, _kwargs=None: streamer.new_pass(), with_kwargs=True)
    transformer.eval()
    return (transformer, streamer)

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='MiniMax H3 bounded-memory Mac/Metal transport')
    parser.add_argument('command', choices=('preflight', 'generate'))
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prompt', default='')
    parser.add_argument('--mode', default='t2v')
    parser.add_argument('--out')
    parser.add_argument('--json-events', action='store_true')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image')
    parser.add_argument('--last-frame')
    parser.add_argument('--frame-index', action='append', default=[])
    parser.add_argument('--reference-image', action='append', default=[])
    parser.add_argument('--source-video')
    parser.add_argument('--reference-audio', action='append', default=[])
    parser.add_argument('--lora', action='append', default=[])
    ns = parser.parse_args(argv)
    result = asdict(mac_h3_preflight(ns.model_dir))
    event = {'event': 'mac_h3_preflight', 'time': time.time(), 'contract': mac_h3_runtime_contract(), **result}
    print(json.dumps(event, sort_keys=True), flush=True)
    if ns.command == 'generate':
        raise RuntimeError(result['blocker'])
    return 0 if result['mps_available'] and result['checkpoint_complete'] else 2
if __name__ == '__main__':
    raise SystemExit(main())
