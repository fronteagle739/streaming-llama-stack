# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/generate.py
# Machine fallback paths cleared; no model data included.
import argparse
import os
import pathlib

def main():
    ap = argparse.ArgumentParser(description='Native-Windows Cosmos3 generation (RTX 5090 / diffusers).')
    ap.add_argument('--model', default='', help='Local Cosmos3 model dir (Nano or Super).')
    ap.add_argument('--mode', choices=['t2i', 't2v'], default='t2i')
    ap.add_argument('--prompt', default='a red cube on a wooden table, studio lighting')
    ap.add_argument('--negative-prompt', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--num-frames', type=int, default=1, help='1 for image; >1 (e.g. 17) for video.')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--steps', type=int, default=35)
    ap.add_argument('--guidance-scale', type=float, default=6.0)
    ap.add_argument('--flow-shift', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--offload', choices=['model', 'sequential', 'none'], default='model', help='model=cpu_offload (proven for 720p T2I); sequential=lowest VRAM (video/Super); none=full GPU.')
    ap.add_argument('--gpu-index', default='1', help='CUDA device index AFTER PCI_BUS_ID ordering (5090=1 on this box).')
    args = ap.parse_args()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu_index)
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    print(f'Loading {args.model} (bf16, offload={args.offload})...', flush=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    if args.offload == 'model':
        pipe.enable_model_cpu_offload()
    elif args.offload == 'sequential':
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to('cuda')
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    print(f'Generating {args.mode} {args.width}x{args.height} num_frames={args.num_frames} steps={args.steps}...', flush=True)
    result = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt, num_frames=args.num_frames, height=args.height, width=args.width, fps=args.fps, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, add_resolution_template=False, add_duration_template=False, generator=gen)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 't2i' or args.num_frames == 1:
        img = result.video[0] if hasattr(result, 'video') else result.frames[0][0] if hasattr(result, 'frames') else result.images[0]
        img.save(out)
    else:
        from diffusers.utils import export_to_video
        vid = result.video if hasattr(result, 'video') else result.frames[0] if hasattr(result, 'frames') else result.images
        export_to_video(vid, str(out), fps=args.fps, macro_block_size=1)
    print(f'PASS: wrote {out}', flush=True)
if __name__ == '__main__':
    main()
