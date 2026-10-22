# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/vram_guard.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os

def cuda_free_mib(device: str='cuda') -> tuple[int, int]:
    import torch
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        return (1 << 60, 1 << 60)
    idx = 0
    if ':' in str(device):
        try:
            idx = int(str(device).split(':', 1)[1])
        except ValueError:
            idx = 0
    free, total = torch.cuda.mem_get_info(idx)
    return (free // (1024 * 1024), total // (1024 * 1024))

def preflight_vram(device: str, need_mib: int, label: str, *, skip_env: str | None=None) -> None:
    skip_names = ['STUDIO_SKIP_VRAM_PREFLIGHT', 'LTX_SKIP_VRAM_PREFLIGHT']
    if skip_env and skip_env not in skip_names:
        skip_names.append(skip_env)
    if any((os.environ.get(n) == '1' for n in skip_names)):
        return
    free_mib, total_mib = cuda_free_mib(device)
    if free_mib < need_mib:
        raise RuntimeError(f"[VRAM-PREFLIGHT] {label}: needs ~{need_mib} MiB free on {device} but only {free_mib} MiB free (of {total_mib} MiB total). Another process is likely holding VRAM on this GPU (external trainer / ComfyUI / another studio job — the driver sees them even though the daemon's job list does not). Free that VRAM or pick a different GPU, then retry. (Override with STUDIO_SKIP_VRAM_PREFLIGHT=1.)")
