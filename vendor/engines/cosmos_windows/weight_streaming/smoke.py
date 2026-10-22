# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/smoke.py
# Machine fallback paths cleared; no model data included.
import os, sys, time
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from cosmos_windows.weight_streaming.build_streamed_transformer import build_streamed_transformer, build_ds_streamed_transformer
MODEL = sys.argv[1] if len(sys.argv) > 1 else ''
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
W = int(sys.argv[3]) if len(sys.argv) > 3 else 256
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
BACKEND = sys.argv[5] if len(sys.argv) > 5 else 'portable'
OUT = os.path.join('', 'ws_smoke_' + os.path.basename(MODEL) + '.png')
print(f'[1] meta transformer + {BACKEND} streaming for {MODEL}', flush=True)
if BACKEND == 'directstorage':
    tf, _ = build_ds_streamed_transformer(MODEL, device='cuda')
else:
    tf, loader = build_streamed_transformer(MODEL, window=2, device='cuda')
print(f'    VRAM after globals: {torch.cuda.memory_allocated() / 1000000000.0:.2f} GB', flush=True)
print('[2] from_pretrained(transformer=tf) — small components only (the big transformer is NOT loaded)', flush=True)
pipe = Cosmos3OmniPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, transformer=tf, enable_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
for name, comp in pipe.components.items():
    if name != 'transformer' and isinstance(comp, torch.nn.Module):
        comp.to('cuda')
print(f'    VRAM after small components: {torch.cuda.memory_allocated() / 1000000000.0:.2f} GB', flush=True)
torch.cuda.reset_peak_memory_stats()
gen = torch.Generator(device='cuda').manual_seed(42)
print(f'[3] generating {W}x{H} T2I, {STEPS} steps, guidance 1.0 via streamed transformer...', flush=True)
t0 = time.perf_counter()
result = pipe(prompt='a red cube on a wooden table', negative_prompt='', num_frames=1, height=H, width=W, num_inference_steps=STEPS, guidance_scale=1.0, add_resolution_template=False, add_duration_template=False, generator=gen)
dt = time.perf_counter() - t0
img = result.video[0] if hasattr(result, 'video') else result.frames[0][0] if hasattr(result, 'frames') else result.images[0]
img.save(OUT)
peak = torch.cuda.max_memory_allocated() / 1000000000.0
print(f'[4] DONE {dt:.1f}s, peak VRAM {peak:.2f} GB -> {OUT}', flush=True)
