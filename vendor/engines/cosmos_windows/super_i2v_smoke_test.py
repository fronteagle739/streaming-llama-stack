# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/super_i2v_smoke_test.py
# Machine fallback paths cleared; no model data included.
import sys, time, os
sys.path.insert(0, '')
import torch
from cosmos_windows.cosmos_cli import Cli, DEFAULTS
st = dict(DEFAULTS)
st.update(out='', width=256, height=256, num_frames=17, fps=8, seed=3, cfg_batch=False, prefetch=True)
os.makedirs(st['out'], exist_ok=True)
cli = Cli(st)
cli.do('/model super-i2v')
cli.do('/set steps 6')
cli.do('/set guidance 6')
try:
    torch.cuda.reset_peak_memory_stats()
except Exception:
    pass
t0 = time.time()
p = cli.generate('i2v', 'the young woman slowly turns her head toward the camera and smiles', image_path='')
dt = time.time() - t0
try:
    peak = torch.cuda.max_memory_allocated() / 1000000000.0
except Exception:
    peak = -1.0
print(f'[super-i2v] 64B DirectStorage i2v 256x256 nf=17/fps8/6steps -> {dt:.1f}s peakVRAM={peak:.1f}GB out={p}', flush=True)
print('RESULT', 'DONE' if p else 'FAIL', flush=True)
