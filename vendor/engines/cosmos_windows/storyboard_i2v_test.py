# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/storyboard_i2v_test.py
# Machine fallback paths cleared; no model data included.
import sys, time, os
sys.path.insert(0, '')
import torch
from cosmos_windows.cosmos_cli import Cli, DEFAULTS
TIFA = ''
PROMPT = 'A cinematic video of the young woman with long brown hair in a white top and black skirt. First she looks to her left. Then she slowly turns her head to face the camera and smiles confidently. Then she raises her right hand and gives a small wave. Throughout, the camera slowly pushes in toward a close-up. Soft natural lighting, smooth realistic motion, sharp focus, high detail.'

def vram_reset():
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

def vram_peak():
    try:
        return torch.cuda.max_memory_allocated() / 1000000000.0
    except Exception:
        return -1.0

def run(tag, fps, num_frames, w, h, steps, out):
    os.makedirs(out, exist_ok=True)
    st = dict(DEFAULTS)
    st.update(cfg_batch=True, prefetch=True, out=out, fps=fps, num_frames=num_frames, width=w, height=h, seed=7)
    cli = Cli(st)
    cli.do(f'/set steps {steps}')
    cli.do('/set guidance 6')
    vram_reset()
    t0 = time.time()
    p = cli.generate('i2v', PROMPT, image_path=TIFA)
    dt = time.time() - t0
    dur = num_frames / fps
    print(f'[storyboard:{tag}] {w}x{h} nf={num_frames} fps={fps} (~{dur:.1f}s) steps={steps} cfg_batch=on -> {dt:.1f}s peakVRAM={vram_peak():.1f}GB  out={p}', flush=True)
    return p
print('[storyboard] === SMOKE (short — de-risk path + memory before the 5s run) ===', flush=True)
run('smoke', fps=8, num_frames=21, w=448, h=640, steps=16, out='')
print('[storyboard] === FULL 5-second clip ===', flush=True)
run('full', fps=16, num_frames=81, w=448, h=640, steps=35, out='')
print('RESULT DONE', flush=True)
