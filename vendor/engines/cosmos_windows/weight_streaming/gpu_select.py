# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/gpu_select.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import subprocess

def gpu_free_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,name', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                rows.append((parts[0], int(parts[1]), parts[2] if len(parts) > 2 else '?'))
            except ValueError:
                pass
    return rows

def pick_free_gpu(min_free_gb: float=0.0, set_env: bool=True):
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    rows = gpu_free_mib()
    if not rows:
        idx = '0'
    else:
        idx, free, name = max(rows, key=lambda r: r[1])
        if free < min_free_gb * 1024:
            print(f'[gpu] WARNING: freest card cuda:{idx} ({name}) has only {free / 1024:.1f}GB free (< {min_free_gb}GB requested)', flush=True)
        print(f'[gpu] selected cuda:{idx} = {name} ({free / 1024:.1f}GB free)', flush=True)
    if set_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = idx
    return idx

def assert_gpu_idle(max_used_gb: float=1.0, indices=None):
    wanted = None if indices is None else {str(i) for i in indices}
    busy = []
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,name', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL).decode()
        for line in out.strip().splitlines():
            idx, used, name = [p.strip() for p in line.split(',')][:3]
            if wanted is not None and idx not in wanted:
                continue
            if int(used) > max_used_gb * 1024:
                busy.append((idx, int(used), name))
    except Exception:
        pass
    if busy:
        print('[gpu] note: in-use GPUs: ' + ', '.join((f'cuda:{i}={u / 1024:.1f}GB({n})' for i, u, n in busy)), flush=True)
    return busy
