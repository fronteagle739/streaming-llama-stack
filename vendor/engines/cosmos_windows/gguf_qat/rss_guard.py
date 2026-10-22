# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/rss_guard.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
import threading
import time
_GUARD_STARTED = False
_PEAK_BYTES = 0
_LOCK = threading.Lock()

def _proc():
    import psutil
    return psutil.Process(os.getpid())

def peak_rss_gb() -> float:
    return _PEAK_BYTES / 1024 ** 3

def current_rss_gb() -> float:
    try:
        return _proc().memory_info().rss / 1024 ** 3
    except Exception:
        return 0.0

def _sampler(limit_bytes: int, period_s: float, label: str):
    global _PEAK_BYTES
    p = _proc()
    while True:
        try:
            rss = p.memory_info().rss
        except Exception:
            return
        with _LOCK:
            if rss > _PEAK_BYTES:
                _PEAK_BYTES = rss
        if rss > limit_bytes:
            gb = rss / 1024 ** 3
            lim = limit_bytes / 1024 ** 3
            sys.stderr.write(f'\n[RSS-GUARD] ABORT ({label}): RSS {gb:.3f} GB exceeded limit {lim:.3f} GB. peak={peak_rss_gb():.3f} GB. Hard-exiting to protect the box.\n')
            sys.stderr.flush()
            os._exit(137)
        time.sleep(period_s)

def start_rss_guard(limit_gb: float=6.0, period_s: float=0.5, label: str='gguf_qat') -> None:
    global _GUARD_STARTED
    if _GUARD_STARTED:
        return
    _GUARD_STARTED = True
    try:
        import psutil
    except Exception as e:
        sys.stderr.write(f'[RSS-GUARD] FATAL: psutil unavailable ({e}); refusing to run unguarded.\n')
        os._exit(2)
    limit_bytes = int(limit_gb * 1024 ** 3)
    t = threading.Thread(target=_sampler, args=(limit_bytes, period_s, label), name='rss-guard', daemon=True)
    t.start()
    sys.stderr.write(f'[RSS-GUARD] armed: limit={limit_gb:.2f} GB, sample={period_s}s, label={label}\n')
    sys.stderr.flush()
if __name__ == '__main__':
    start_rss_guard(limit_gb=0.5, period_s=0.1, label='selftest')
    print('armed at 0.5GB; allocating to trigger abort (expected hard-exit 137)...')
    blob = bytearray()
    for _ in range(2000):
        blob += bytearray(4 * 1024 * 1024)
        time.sleep(0.02)
    print('ERROR: guard did NOT fire')
