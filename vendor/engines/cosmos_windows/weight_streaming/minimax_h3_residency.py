# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_residency.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import hashlib
import math
import os
from pathlib import Path
import struct
VRAM_FRACTION = 0.934
VAE_PEAK_BYTES = 7100000000
WORKSPACE_BYTES = 500000000
MEASURED_EXTERNAL_BYTES = {'af8386ef349c7924031b0a64a7d7c160deaa3a004ae1eb7391137101d17d2d59': 9781936128, '9a0facdd92b56577908d4e9b68a365b51288bc15de455ad5bc54e6555076591e': 3615916032}
VAE_384_PEAK_BYTES = 5420000000
REFERENCE_ROW_BYTES = 251465
REFERENCE_ROWS_AT_2048 = 5184
REFERENCE_ROWS_NATIVE = 144

def video_reference_rows(width: int, height: int, duration: float) -> int:
    from .minimax_h3_pipeline import h3_latent_geometry
    geometry = h3_latent_geometry(float(duration), int(width), int(height))
    return int(geometry.video_rows)

def estimate_reference_rows(scales, *, native_res: bool=False) -> int:
    if isinstance(scales, (int, float)):
        raise TypeError('estimate_reference_rows takes one scale PER reference, not a count -- a bare count silently prices every reference at full size')
    total = 0
    for scale in scales:
        scale = float(scale)
        if not 0 < scale <= 1.0:
            raise ValueError(f'reference scale must be in (0, 1]: {scale}')
        if native_res:
            total += REFERENCE_ROWS_NATIVE
        else:
            total += math.ceil(REFERENCE_ROWS_AT_2048 * scale * scale)
    return int(total)
VAE_BYTES_PER_PIXEL_SECOND = (VAE_PEAK_BYTES - VAE_384_PEAK_BYTES) / ((448 * 448 - 384 * 384) * 5.0)
VAE_FIXED_BYTES = VAE_384_PEAK_BYTES - VAE_BYTES_PER_PIXEL_SECOND * 384 * 384 * 5.0

def cuda_memory_snapshot(device=0):
    import torch
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    return dict(total_bytes=total, free_bytes=free, reserved_bytes=reserved, allocated_bytes=torch.cuda.memory_allocated(device), external_bytes=max(0, total - free - reserved))

def tensor_headers(root):
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.glob('*.safetensors'))
    if not files:
        raise ValueError(f'No safetensors in {root}')
    for path in files:
        with path.open('rb') as source:
            length = struct.unpack('<Q', source.read(8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError(f'Oversized safetensors header: {path}')
            header = json.loads(source.read(length))
        yield from ((name, desc) for name, desc in header.items() if name != '__metadata__')

def resident_weight_bytes(model_dir, loras=(), *, packed_adapters=True):
    root = Path(model_dir)
    if (root / 'transformer').is_dir():
        root = root / 'transformer'
    total = 0
    for name, desc in tensor_headers(root):
        if name.endswith('.comfy_quant'):
            continue
        width = 4 if name.endswith('.weight_scale') else 1 if desc['dtype'] == 'I8' else 2
        total += math.prod(desc['shape']) * width
    if (root / 'config.json').is_file():
        from .safetensors_index import build_index
        from .minimax_h3_native import build_meta_h3_native_model
        from .minimax_h3_fused import index_looks_fused, checkpoint_quant_map
        from .minimax_h3_int8 import convert_h3_linears_to_int8, quantised_module_names
        index = build_index(root)
        fused = index_looks_fused(index.refs)
        model = build_meta_h3_native_model(root / 'config.json', fused_adaln=fused)
        if fused:
            convert_h3_linears_to_int8(model, quantised_module_names(index.refs, checkpoint_quant_map(root)), float_matmul=packed_adapters)
        total = sum((t.numel() * t.element_size() for t in model.state_dict().values()))
    for path in loras:
        for name, desc in tensor_headers(path):
            packed = any((marker in name for marker in ('.attn.qkv_proj.', '.attn.out_proj.', '.mlp.fc1.', '.mlp.fc2.')))
            packed = packed and 'token_refiner' not in name and packed_adapters
            raw_width = 2 if desc['dtype'] in ('BF16', 'F16') else 4
            width = raw_width + (0 if packed else 4) if packed_adapters else 4
            total += math.prod(desc['shape']) * width
    return total

def residency_budget(weight_bytes, *, free_bytes, total_bytes, allocated_bytes=0, reserved_bytes=0, external_bytes=None, fraction=VRAM_FRACTION, width=448, height=448, duration=5.0, vae_weight_saving_bytes=0, reference_images=0, reference_rows=0):
    if not 0 < fraction <= VRAM_FRACTION:
        raise ValueError('H3 live allocator fraction must be in (0, 0.934]')
    pixel_seconds = width * height * duration
    vae_peak = math.ceil(VAE_FIXED_BYTES + VAE_BYTES_PER_PIXEL_SECOND * pixel_seconds)
    workspace = math.ceil(WORKSPACE_BYTES * pixel_seconds / (448 * 448 * 5.0))
    vae_peak -= min(max(0, vae_weight_saving_bytes), VAE_384_PEAK_BYTES - WORKSPACE_BYTES)
    from .minimax_h3_vae_decode_config import decode_tile_extra_bytes, decode_tile_size, decode_stack_tiles, decode_reuse_workspace, decode_cache_weights, decode_offload_weights
    tile_extra = decode_tile_extra_bytes(width, height)
    vae_peak += math.ceil(tile_extra * duration / 5.0)
    current_external = max(0, total_bytes - free_bytes - reserved_bytes)
    calibrated = external_bytes is not None
    overhead = max(current_external, external_bytes or 0)
    torch_need = weight_bytes + vae_peak + workspace
    need = torch_need + overhead
    allocator_free = int(total_bytes * fraction) - allocated_bytes
    device_available = total_bytes - allocated_bytes - WORKSPACE_BYTES
    available = min(device_available, allocator_free + overhead)
    reference_bytes = int(REFERENCE_ROW_BYTES * max(0, int(reference_rows)))
    unmodelled = f'{reference_images} reference(s) add {reference_rows:,} rows costing about {reference_bytes / 1000000000.0:.2f} GB, which is NOT in the margin above. The row counts are derived (per-image scale, and clip geometry for a video reference); the per-row BYTE cost is two measured points, so treat this as an UPPER BOUND rather than a guarantee.' if reference_images else ''
    return dict(resident=calibrated and weight_bytes > 0 and (need <= available), reference_images=int(reference_images), reference_rows=int(reference_rows), reference_bytes=reference_bytes, unmodelled=unmodelled, vae_decode_tile=decode_tile_size(), vae_decode_stack=decode_stack_tiles(), vae_decode_reuse=decode_reuse_workspace(), vae_decode_cache=decode_cache_weights(), vae_decode_offload=decode_offload_weights(), vae_tile_extra_bytes=tile_extra, vae_weight_saving_bytes=vae_weight_saving_bytes, weight_bytes=weight_bytes, vae_peak_bytes=vae_peak, workspace_bytes=workspace, external_bytes=overhead, measured_external_bytes=external_bytes, current_external_bytes=current_external, calibrated=calibrated, physical_margin_bytes=WORKSPACE_BYTES, torch_required_bytes=torch_need, required_bytes=need, device_free_bytes=free_bytes, allocator_free_bytes=allocator_free, device_available_bytes=device_available, available_bytes=available, fraction=fraction)

def probe_residency(model_dir, *, loras=(), device=0, width=448, height=448, duration=5.0, fraction=VRAM_FRACTION, reference_images=0, reference_rows=0):
    import torch
    weight_bytes = resident_weight_bytes(model_dir, loras, packed_adapters=os.environ.get('STUDIO_H3_INT8_MATMUL') != '1' or os.environ.get('STUDIO_H3_FUSED_INT8') == '1')
    from .ds_backend import _DEFAULT_DLL
    dll = Path(_DEFAULT_DLL)
    digest = hashlib.sha256(dll.read_bytes()).hexdigest() if dll.is_file() else None
    memory = cuda_memory_snapshot(device)
    saving = 0
    if os.environ.get('STUDIO_H3_FUSED_W4A8', '0') == '1':
        from .minimax_h3_native import comfy_kitchen
        from .minimax_h3_w4a8 import packed_w4a8_weight_saving
        vae_dir = os.environ.get('MINIMAX_H3_VIDEO_VAE_DIR')
        if vae_dir and comfy_kitchen() is not None:
            saving = packed_w4a8_weight_saving(vae_dir)
    budget = residency_budget(weight_bytes, free_bytes=memory['free_bytes'], total_bytes=memory['total_bytes'], allocated_bytes=memory['allocated_bytes'], reserved_bytes=memory['reserved_bytes'], external_bytes=MEASURED_EXTERNAL_BYTES.get(digest), fraction=fraction, width=width, height=height, duration=duration, vae_weight_saving_bytes=saving, reference_images=reference_images, reference_rows=reference_rows)
    return dict(budget, dll=str(dll), dll_sha256=digest)

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--lora', action='append', default=[])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--width', type=int, default=448)
    ap.add_argument('--height', type=int, default=448)
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--reference-native-res', action='store_true', help='references sampled at the clip size instead of a 2048 short edge; worth ~3.8 GB at three references')
    ap.add_argument('--reference-images', type=int, default=0, help='how many reference images the session will condition on; their rows are NOT modelled, so the budget reports that rather than implying a margin')
    ap.add_argument('--reference-videos', type=int, default=0, help="how many VIDEO references the session will condition on. They encode at the CLIP geometry over every frame, so each costs the clip's own video row count rather than a single image's.")
    ap.add_argument('--reference-scale', action='append', default=[], type=float, help='repeat once PER reference: its multiplier of the 2048 short edge. Omit for the released full-size sampling on every reference.')
    args = ap.parse_args()
    scales = list(args.reference_scale)
    if scales and len(scales) != args.reference_images:
        ap.error(f'--reference-scale given {len(scales)} time(s) for {args.reference_images} reference image(s): pass one scale per reference or none at all')
    scales = scales or [1.0] * max(0, args.reference_images)
    print(json.dumps(probe_residency(args.model_dir, loras=args.lora, device=args.gpu, width=args.width, height=args.height, duration=args.duration, reference_images=args.reference_images + args.reference_videos, reference_rows=estimate_reference_rows(scales, native_res=args.reference_native_res) + args.reference_videos * video_reference_rows(args.width, args.height, args.duration))))
if __name__ == '__main__':
    main()
