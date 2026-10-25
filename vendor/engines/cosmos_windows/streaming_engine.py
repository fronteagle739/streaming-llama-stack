# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/streaming_engine.py
# Machine fallback paths cleared; no model data included.
import argparse
import math
import os
import pathlib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

def main():
    ap = argparse.ArgumentParser(description='Blockwise streaming Cosmos3 video (configured CUDA device).')
    ap.add_argument('--model', default='')
    ap.add_argument('--prompt', default='a cinematic drone shot over a coastal road at sunset')
    ap.add_argument('--negative-prompt', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--total-frames', type=int, default=49, help='total frames to stream across all blocks')
    ap.add_argument('--chunk', type=int, default=9, help='frames per block (bounds per-block attention -> VRAM). Use 4k+1.')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--steps', type=int, default=35)
    ap.add_argument('--guidance-scale', type=float, default=6.0)
    ap.add_argument('--flow-shift', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--offload', choices=['model', 'sequential', 'none'], default='model')
    ap.add_argument('--stream-weights', action='store_true', help="WS1+P3 combo: stream the transformer's layers NVMe->GPU on demand (meta-init), so a big model fits AND long video streams in bounded blocks. The full streaming-diffusion target.")
    ap.add_argument('--window', type=int, default=2, help='weight-stream prefetch window (with --stream-weights)')
    ap.add_argument('--weight-backend', choices=['portable', 'directstorage'], default='portable', help="with --stream-weights: 'directstorage' = NVMe->GPU DMA via the wf_* C ABI (~12-15x faster, lower VRAM; Windows). 'portable' = mmap+pinned (cross-platform floor).")
    ap.add_argument('--gpu-index', default=os.environ.get('STUDIO_GPU_INDEX', '0'))
    args = ap.parse_args()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu_index)
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    if args.stream_weights:
        if args.weight_backend == 'directstorage':
            from cosmos_windows.weight_streaming.build_streamed_transformer import build_ds_streamed_transformer
            print(f'Loading {args.model} with DIRECTSTORAGE-streamed transformer (NVMe->GPU DMA)...', flush=True)
            tf, _streamer = build_ds_streamed_transformer(args.model, device='cuda')
        else:
            from cosmos_windows.weight_streaming.build_streamed_transformer import build_streamed_transformer
            print(f'Loading {args.model} with portable WEIGHT-STREAMED transformer (window={args.window})...', flush=True)
            tf, _loader = build_streamed_transformer(args.model, window=args.window, device='cuda')
        pipe = Cosmos3OmniPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, transformer=tf, enable_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
        for nm, comp in pipe.components.items():
            if nm != 'transformer' and isinstance(comp, torch.nn.Module):
                comp.to('cuda')
    else:
        print(f'Loading {args.model} (bf16, offload={args.offload})...', flush=True)
        pipe = Cosmos3OmniPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, enable_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
        if args.offload == 'model':
            pipe.enable_model_cpu_offload()
        elif args.offload == 'sequential':
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to('cuda')
    if (args.chunk - 1) % 4 != 0:
        valid = max(5, (args.chunk - 1) // 4 * 4 + 1)
        print(f'WARN: --chunk {args.chunk} is not 4k+1; using {valid} per block (VAE temporal constraint).', flush=True)
        args.chunk = valid
    import math as _m
    n_blocks = max(1, 1 + _m.ceil(max(0, args.total_frames - args.chunk) / (args.chunk - 1)))
    print(f'Streaming ~{args.total_frames} frames in {n_blocks} block(s) of {args.chunk} (4k+1) @ {args.width}x{args.height}', flush=True)
    all_frames = []
    last_frame = None
    produced = 0
    blk = -1
    while produced < args.total_frames:
        blk += 1
        want = args.chunk
        gen = torch.Generator(device='cuda').manual_seed(args.seed + blk)
        kwargs = dict(prompt=args.prompt, negative_prompt=args.negative_prompt, num_frames=want, height=args.height, width=args.width, fps=args.fps, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, add_resolution_template=False, add_duration_template=False, generator=gen)
        if last_frame is not None:
            kwargs['image'] = last_frame
        t0 = time.perf_counter()
        result = pipe(**kwargs)
        block_frames = list(result.video if hasattr(result, 'video') else result.frames[0])
        if last_frame is not None and len(block_frames) > 1:
            block_frames = block_frames[1:]
        all_frames.extend(block_frames)
        produced += len(block_frames)
        last_frame = block_frames[-1]
        print(f'  block {blk + 1}/{n_blocks}: +{len(block_frames)} frames (total {produced}/{args.total_frames}) in {time.perf_counter() - t0:.1f}s', flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(all_frames, str(out), fps=args.fps, macro_block_size=1)
    print(f'STREAM PASS: wrote {out} ({produced} frames, {n_blocks} blocks)', flush=True)
if __name__ == '__main__':
    main()
