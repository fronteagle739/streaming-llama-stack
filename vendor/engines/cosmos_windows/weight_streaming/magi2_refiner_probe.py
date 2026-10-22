# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_refiner_probe.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

def _setup(model_dir: Path):
    from cosmos_windows.weight_streaming.magi2_attention_adapter import install_dependency_shims, install_magi2_attention_intercept
    from cosmos_windows.weight_streaming.magi2_moe_binding import install_moe_kernel_sm120_patch
    os.environ.setdefault('MAGI2_CKPT_ROOT', str(model_dir))
    os.environ.setdefault('MAGI2_DISABLE_MAGI_COMPILE', '1')
    os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
    install_dependency_shims()
    import inference.model.magi2_preview
    import inference.model.magi2_refiner
    import inference.common.arch
    install_magi2_attention_intercept()
    install_moe_kernel_sm120_patch()

def run(model_dir: Path, refiner_steps: int, txt_tokens: int=226, prev_hw=(16, 28), prev_t: int=32, refiner_width: int=1920, refiner_height: int=1088) -> dict:
    import torch
    from cosmos_windows.weight_streaming.magi2_refiner_stage import load_refiner_config, load_refiner_resident, refine_latent
    device = 'cuda'
    dtype = torch.bfloat16
    rep: dict = {'refiner_steps': refiner_steps, 'device': torch.cuda.get_device_name(0)}
    _mf = os.environ.get('MAGI2_MEM_FRACTION')
    if _mf:
        torch.cuda.set_per_process_memory_fraction(float(_mf), 0)
    rcfg = load_refiner_config()
    ac = rcfg.magi2_refiner_arch_config
    ec = rcfg.evaluation_config
    t = time.time()
    refiner = load_refiner_resident(Path(model_dir) / 'refiner', ac, device)
    rep['load_s'] = round(time.time() - t, 1)
    rep['refiner_resident_gb'] = round(torch.cuda.memory_allocated() / 1000000000.0, 2)
    try:
        import psutil
        rep['host_rss_gb_after_load'] = round(psutil.Process(os.getpid()).memory_info().rss / 1000000000.0, 2)
    except Exception:
        rep['host_rss_gb_after_load'] = None
    torch.manual_seed(42)
    h, w = prev_hw
    latent = torch.randn(1, ac.video_in_channels, prev_t, h, w, dtype=torch.float32, device=device)
    audio_latent = torch.zeros(1, 0, 64, dtype=torch.float32, device=device)
    context = torch.randn(1, txt_tokens, ac.text_in_channels, dtype=dtype, device=device)
    null_context = torch.randn(1, txt_tokens, ac.text_in_channels, dtype=dtype, device=device)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad():
        refined, refiner_audio = refine_latent(refiner, ec, context, null_context, latent, audio_latent, refiner_width, refiner_height, refiner_steps, device=device, dtype=dtype)
    torch.cuda.synchronize()
    rep['refine_s'] = round(time.time() - t, 1)
    rep['refine_s_per_step'] = round(rep['refine_s'] / max(refiner_steps, 1), 1)
    rep['refiner_peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1000000000.0, 2)
    rep['refined_shape'] = list(refined.shape)
    rep['refined_finite'] = bool(torch.isfinite(refined).all())
    rep['passed'] = rep['refined_finite']
    return rep

def _main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, default=Path(''))
    p.add_argument('--refiner-steps', type=int, default=2)
    p.add_argument('--refiner-width', type=int, default=1920)
    p.add_argument('--refiner-height', type=int, default=1088)
    args = p.parse_args(argv)
    _setup(args.model_dir)
    rep = run(args.model_dir, args.refiner_steps, refiner_width=args.refiner_width, refiner_height=args.refiner_height)
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep.get('passed') else 1
if __name__ == '__main__':
    sys.path.insert(0, '')
    sys.path.insert(0, os.environ.get('MAGI2_REPO_ROOT', ''))
    raise SystemExit(_main())
