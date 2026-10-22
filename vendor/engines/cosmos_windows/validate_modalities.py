# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_modalities.py
# Machine fallback paths cleared; no model data included.
import os, sys, time, traceback
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import numpy as np
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
MODEL = sys.argv[1] if len(sys.argv) > 1 else ''
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
W = int(sys.argv[3]) if len(sys.argv) > 3 else 256
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
NF = 17
OUTDIR = ''
os.makedirs(OUTDIR, exist_ok=True)

def _frames_ok(frames):
    if not frames:
        return (False, 'no frames')
    arrs = [np.asarray(f, dtype=np.float32) for f in frames]
    a = np.stack(arrs)
    if not np.isfinite(a).all():
        return (False, f'non-finite values (shape {a.shape})')
    std = float(a.std())
    if std < 1.0:
        return (False, f'degenerate/flat (std={std:.3f}, shape {a.shape})')
    return (True, f'{len(frames)} frame(s) {frames[0].size} std={std:.1f}')

def run(pipe, name, **kw):
    base = dict(negative_prompt='', height=H, width=W, num_inference_steps=STEPS, guidance_scale=6.0, add_resolution_template=False, add_duration_template=False, generator=torch.Generator(device='cuda').manual_seed(42))
    base.update(kw)
    t0 = time.perf_counter()
    try:
        res = pipe(**base)
        dt = time.perf_counter() - t0
        frames = list(res.video) if not isinstance(res.video, list) else res.video
        ok, msg = _frames_ok(frames)
        if len(frames) == 1:
            frames[0].save(os.path.join(OUTDIR, f'{name}.png'))
        else:
            frames[0].save(os.path.join(OUTDIR, f'{name}_frame0.png'))
            frames[-1].save(os.path.join(OUTDIR, f'{name}_frameN.png'))
        snd = getattr(res, 'sound', None)
        snd_msg = ''
        if snd is not None:
            sok = torch.isfinite(snd).all().item() and float(snd.std()) > 0.0001
            snd_msg = f" | sound {tuple(snd.shape)} std={float(snd.std()):.4f} {('OK' if sok else 'FLAT')}"
            ok = ok and sok
            np.save(os.path.join(OUTDIR, f'{name}_sound.npy'), snd.float().cpu().numpy())
        print(f"  [{('PASS' if ok else 'FAIL')}] {name}: {dt:.1f}s | {msg}{snd_msg}", flush=True)
        return (frames[0] if ok else None, ok)
    except Exception as e:
        print(f'  [FAIL] {name}: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        return (None, False)

def main():
    print(f'[load] {MODEL} (bf16, model-cpu-offload) on RTX 5090...', flush=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
    pipe.enable_model_cpu_offload()
    print(f'  sound_tokenizer present: {pipe.sound_tokenizer is not None}', flush=True)
    print(f'[run] {W}x{H}, {STEPS} steps, {NF}-frame video\n', flush=True)
    results = {}
    img, ok = run(pipe, 't2i', prompt='a red cube on a wooden table, studio lighting', num_frames=1)
    results['T2I'] = ok
    _, ok = run(pipe, 't2v', prompt='a red cube slowly rotating on a wooden table', num_frames=NF, fps=24)
    results['T2V'] = ok
    cond = img if img is not None else Image.fromarray((np.random.RandomState(0).rand(H, W, 3) * 255).astype(np.uint8))
    _, ok = run(pipe, 'i2v', prompt='the cube rolls across the table', image=cond, num_frames=NF, fps=24)
    results['I2V'] = ok
    _, ok = run(pipe, 't2v_sound', prompt='rain falling on a window with thunder', num_frames=NF, fps=24, enable_sound=True)
    results['T2V+sound'] = ok
    print('\n=== MODALITY VALIDATION SUMMARY (diffusers Cosmos3OmniPipeline) ===', flush=True)
    for k, v in results.items():
        print(f"  {k:12s} {('PASS' if v else 'FAIL')}", flush=True)
    allok = all(results.values())
    print(f"\n{('ALL DIFFUSERS MODALITIES PASS' if allok else 'SOME MODALITIES FAILED')} -> {OUTDIR}", flush=True)
    sys.exit(0 if allok else 1)
if __name__ == '__main__':
    main()
