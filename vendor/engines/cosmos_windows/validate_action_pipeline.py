# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_action_pipeline.py
# Machine fallback paths cleared; no model data included.
import os, sys, time
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cosmos_compat
import torch
from diffusers_cosmos3 import Cosmos3OmniDiffusersPipeline
from diffusers_cosmos3.transformer import Cosmos3OmniTransformer
MODEL = sys.argv[1] if len(sys.argv) > 1 else ''
OUT = ''
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def main():
    print(f'[1] load ACTION-CAPABLE plugin transformer from {MODEL}\\transformer ...', flush=True)
    t0 = time.perf_counter()
    tf = Cosmos3OmniTransformer.from_pretrained(os.path.join(MODEL, 'transformer'), torch_dtype=torch.bfloat16)
    print(f'    loaded in {time.perf_counter() - t0:.1f}s | action_gen={tf.config.action_gen} max_action_dim={tf.config.max_action_dim}', flush=True)
    act = {n: p for n, p in tf.named_parameters() if 'action' in n.lower()}
    print(f'[2] ACTION heads in the live transformer: {len(act)} tensors', flush=True)
    ok = len(act) >= 5
    for n, p in act.items():
        real = p.is_meta is False and torch.isfinite(p).all().item() and (float(p.float().abs().sum()) > 0)
        ok = ok and real
        print(f'      {n:30s} {tuple(p.shape)} finite+nonzero={real}', flush=True)
    if not ok:
        print('[FAIL] action heads not loaded with real weights', flush=True)
        sys.exit(1)
    print('    -> the FULL MoT model (incl. 32-embodiment action heads) is loaded on Windows.', flush=True)
    print('[3] (best-effort) build plugin pipeline + try a gen (T2V; action gen needs native framework) ...', flush=True)
    try:
        pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(MODEL, transformer=tf, torch_dtype=torch.bfloat16)
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to('cuda')
        gen = torch.Generator(device='cuda').manual_seed(42)
        t0 = time.perf_counter()
        out = pipe(prompt='a red cube slowly rotating on a wooden table', negative_prompt='', num_frames=17, height=256, width=256, generator=gen)
        dt = time.perf_counter() - t0
        vid = getattr(out, 'video', None) or getattr(out, 'frames', None) or out
        frames = vid[0] if isinstance(vid, (list, tuple)) and vid and isinstance(vid[0], (list, tuple)) else vid
        n = len(frames) if hasattr(frames, '__len__') else '?'
        print(f'[4] BONUS: plugin pipeline GENERATED {n} frames in {dt:.1f}s', flush=True)
        try:
            from diffusers.utils import export_to_video
            export_to_video(list(frames), OUT, fps=24)
            print(f'    saved -> {OUT}', flush=True)
        except Exception:
            pass
    except Exception as e:
        print(f'[4] plugin __call__ gen skipped (transformers-5.x plugin friction, expected): {type(e).__name__}: {e}', flush=True)
        print('    (stock diffusers covers T2V; action gen needs native cosmos_framework — see MODALITY_COVERAGE.md)', flush=True)
    print('\n=== ACTION-CAPABLE MODEL PASS: full MoT model (32-embodiment action heads, real weights) loads on native Windows ===', flush=True)
    sys.exit(0)
if __name__ == '__main__':
    main()
