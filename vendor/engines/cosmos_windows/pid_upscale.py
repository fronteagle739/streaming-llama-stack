# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/pid_upscale.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

def _stream_child(cmd, cwd, env, phases: int):
    from collections import deque
    tail = deque(maxlen=40)

    def step(n, label):
        print(f'[pid] step {n}/{phases} {label}', flush=True)
    step(1, 'loading PiD model + VAE (first run downloads the ~5GB text encoder, then cached)...')
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    seen = set()
    for raw in proc.stdout:
        line = raw.rstrip('\n')
        if line:
            tail.append(line)
        if 'Processing' in line and 'sample' in line and (2 not in seen):
            seen.add(2)
            step(2, 'model ready — encoding input image...')
        elif 'Clean latent shape' in line and 3 not in seen:
            seen.add(3)
            step(3, 'encoded — running pixel-diffusion upscale...')
        elif '-> ours=' in line and 4 not in seen:
            seen.add(4)
            step(4, 'upscaled — saving result...')
        elif 'Done!' in line and 'saved' in line and (5 not in seen):
            seen.add(5)
            step(5, 'finalizing...')
    rc = proc.wait()
    return (rc, '\n'.join(tail))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='input image to upscale')
    ap.add_argument('--out', required=True, help='output path for the upscaled image')
    ap.add_argument('--backbone', default='flux', help='VAE latent space: flux (=Z-Image) | flux2 | sd3 | sdxl | qwenimage | dinov2 | siglip')
    ap.add_argument('--scale', type=int, default=4, help='super-resolution factor (4 or 8)')
    ap.add_argument('--ckpt-type', default='2kto4k', choices=['2k', '2kto4k'])
    ap.add_argument('--steps', type=int, default=4, help='PiD inference steps (checkpoints are 4-step distilled)')
    ap.add_argument('--prompt', default='', help='optional caption (PiD conditions on it; blank = model default)')
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--pid-repo', default='')
    ap.add_argument('--pid-model-dir', default='')
    a = ap.parse_args()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '20')
    if not (os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')):
        print('[pid] note: no HF_TOKEN in env — first-run text-encoder download may rate-limit/401 if the repo is gated (cached after the first success).', flush=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from cosmos_windows import mem_guard
        mem_guard.preflight(min_free_gb=6.0, label='PiD upscale')
        mem_guard.start_watchdog(max_process_gb=22.0, min_free_gb=3.0, label='PiD upscale')
    except Exception as e:
        print(f'[pid] mem_guard note: {e}', flush=True)
    try:
        import subprocess as _sp
        q = _sp.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(a.gpu)], capture_output=True, text=True, timeout=15)
        free_mb = int((q.stdout or '0').strip().splitlines()[0])
        need_mb = 14000 if a.scale <= 4 else 22000
        print(f'[pid] VRAM free on GPU {a.gpu}: {free_mb} MB (PiD x{a.scale} needs ~{need_mb} MB)', flush=True)
        if free_mb < need_mb and (not os.environ.get('PID_ALLOW_LOW_VRAM')):
            print(f"[pid] ABORT: only {free_mb} MB VRAM free on GPU {a.gpu} — PiD x{a.scale} needs ~{need_mb} MB. The app's gen model is likely resident on this GPU; switch models / wait for it to free, then retry. (override: PID_ALLOW_LOW_VRAM=1)", flush=True)
            return 5
    except Exception as e:
        print(f'[pid] VRAM check skipped: {e}', flush=True)
    if not os.path.isfile(a.image):
        print(f'[pid] ERROR: input image not found: {a.image}', flush=True)
        return 2
    if not os.path.isdir(os.path.join(a.pid_model_dir, 'checkpoints')):
        print(f'[pid] ERROR: PiD checkpoints not found under {a.pid_model_dir}\\checkpoints (download nvidia/PiD first)', flush=True)
        return 2
    outdir = tempfile.mkdtemp(prefix='pid_')
    print(f'[pid] upscaling {os.path.basename(a.image)} x{a.scale} via PiD {a.backbone} ({a.ckpt_type}, {a.steps} steps)...', flush=True)
    cmd = [sys.executable, '-m', 'pid._src.inference.from_clean', '--backbone', a.backbone, '--input_path', a.image, '--degrade_sigmas', '0.0', '--cfg_scale', '1', '--pid_inference_steps', str(a.steps), '--scale', str(a.scale), '--pid_ckpt_type', a.ckpt_type, '--output_dir', outdir]
    cmd += ['--prompt', a.prompt or 'a high quality, sharp, highly detailed photograph']
    link = os.path.join(a.pid_repo, 'checkpoints')
    target = os.path.join(a.pid_model_dir, 'checkpoints')
    if not os.path.exists(link) and os.path.isdir(target):
        try:
            os.system(f'cmd /c mklink /J "{link}" "{target}" >nul 2>&1')
        except Exception as e:
            print(f'[pid] note: could not link checkpoints ({e}); pass --pid-model-dir == --pid-repo if needed', flush=True)
    env = dict(os.environ, PYTHONPATH=a.pid_repo, PYTHONUNBUFFERED='1')
    t0 = time.time()
    rc, tail = _stream_child(cmd, a.pid_repo, env, phases=5)

    def _seg(p: str) -> str:
        return p.replace('/', os.sep)
    cands = [p for p in glob.glob(os.path.join(outdir, '**', '*.*'), recursive=True) if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.webp') and _seg('/input/') not in p and (_seg('/vae_decode/') not in p)]
    if not cands:
        print(f'[pid] FAILED (no output, exit {rc}):\n{tail}', flush=True)
        return 2
    src = max(cands, key=os.path.getmtime)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    shutil.copy(src, a.out)
    print(f'[pid] upscaled in {time.time() - t0:.0f}s -> {a.out}', flush=True)
    print(f'RESULT DONE {a.out}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
