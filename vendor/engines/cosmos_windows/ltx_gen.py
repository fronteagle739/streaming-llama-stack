# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/ltx_gen.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import sys
import time

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-file', default='')
    ap.add_argument('--components-repo', default=os.environ.get('LTX23_COMPONENTS', ''), help='diffusers LTX-2.3 repo/dir with the NON-transformer components (vae/audio_vae/text-enc/vocoder/scheduler/connectors/processor). We stream the transformer separately.')
    ap.add_argument('--image', default=None, help='i2v: conditioning first frame')
    ap.add_argument('--frames', type=int, default=97)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=704)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', default=os.environ.get('STUDIO_GPU_INDEX', '0'))
    ap.add_argument('--guidance', type=float, default=4.0)
    ap.add_argument('--audio-guidance', type=float, default=4.0)
    ap.add_argument('--negative', default='worst quality, blurry, jittery, distorted, watermark, text')
    ap.add_argument('--sampler', default=None, help='Diffusers/RES4LYF sampler name, e.g. res_2s')
    ap.add_argument('--sigmas', default=None, help='Sigma schedule, e.g. beta57 or bong_tangent')
    ap.add_argument('--buffers', type=int, default=3)
    ap.add_argument('--decode-audio', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cosmos_windows import mem_guard
    mem_guard.preflight(min_free_gb=8.0, label='LTX 2.3 (22B) streamed gen')
    proc_cap = max(28.0, min(mem_guard.total_gb() - 4.0, 64.0))
    mem_guard.start_watchdog(max_process_gb=proc_cap, min_free_gb=3.0, label='LTX 2.3 gen')
    from cosmos_windows.weight_streaming.ltx_backend import LTX23_DEFAULT_COMPONENTS_REPO, streamed_ltx_generate
    if not a.components_repo:
        a.components_repo = LTX23_DEFAULT_COMPONENTS_REPO
    if a.image:
        print("[ltx] ABORT: --image/i2v is not yet wired into streamed_ltx_generate's manual denoise loop.", flush=True)
        return 5
    from diffusers.utils import export_to_video
    print(f'[ltx] generating {a.frames}f @ {a.width}x{a.height}, {a.steps} steps...', flush=True)
    t1 = time.time()
    out = streamed_ltx_generate(a.model_file, a.prompt, components_repo=a.components_repo, negative_prompt=a.negative, num_frames=a.frames, height=a.height, width=a.width, num_inference_steps=a.steps, guidance_scale=a.guidance, audio_guidance_scale=a.audio_guidance, frame_rate=float(a.fps), seed=a.seed, n_gpu_buffers=a.buffers, decode_audio=a.decode_audio, sampler_name=a.sampler, sigmas_name=a.sigmas)
    frames = out['video']
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    export_to_video(frames, a.out, fps=a.fps)
    print(f'[ltx] generated in {time.time() - t1:.0f}s', flush=True)
    print(f'RESULT DONE {a.out}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
