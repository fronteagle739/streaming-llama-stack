# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import sys
from pathlib import Path
_DEFAULT_COSMOS_REPOS = ('',)
_TRUTHY = ('1', 'true', 'yes', 'on')

def _candidate_repo_paths() -> list[str]:
    cands: list[str] = []
    env = os.environ.get('STUDIO_COSMOS_WINDOWS_PATH', '').strip()
    if env:
        cands.append(env)
    try:
        from .config import settings
        p = str(getattr(settings, 'cosmos_windows_path', '') or '').strip()
        if p:
            cands.append(p)
    except Exception:
        pass
    cands.extend(_DEFAULT_COSMOS_REPOS)
    return cands

def shared_ram_guard():
    try:
        from cosmos_windows.weight_streaming import ram_guard
        return ram_guard
    except Exception:
        pass
    for c in _candidate_repo_paths():
        if c and Path(c).is_dir() and (c not in sys.path):
            sys.path.insert(0, c)
    try:
        from cosmos_windows.weight_streaming import ram_guard
        return ram_guard
    except Exception:
        return None

def free_host_ram_gb() -> float:
    rg = shared_ram_guard()
    if rg is not None:
        return rg.free_host_ram_gb()
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return 0.0

def preflight_resident_ram(need_gb: float, label: str, *, env_prefix: str, advice: str | None=None) -> None:
    try:
        need = float(os.environ.get(f'{env_prefix}_MIN_FREE_RAM_GB', need_gb))
    except (TypeError, ValueError):
        need = float(need_gb)
    skip_env = f'{env_prefix}_SKIP_RAM_PREFLIGHT'
    force_env = f'{env_prefix}_FORCE_RESIDENT'
    rg = shared_ram_guard()
    if rg is not None:
        rg.preflight_host_ram(need, label, skip_env=skip_env, force_env=force_env, advice=advice)
        return

    def _set(name: str) -> bool:
        return os.environ.get(name, '').strip().lower() in _TRUTHY
    if _set('STUDIO_SKIP_RAM_PREFLIGHT') or _set('STUDIO_RAM_PREFLIGHT_FORCE') or _set(skip_env) or _set(force_env):
        return
    free = free_host_ram_gb()
    if free < need:
        tail = advice or 'Use the streaming/staged path (the RAM-bounded route on this box) instead of a resident load.'
        raise RuntimeError(f'[RAM-PREFLIGHT] {label}: needs ~{need:.0f} GB free host RAM for a RESIDENT load but only {free:.1f} GB is free. This box is host-RAM constrained; a resident load here would thrash the pagefile and can hard-hang the machine. {tail} (Override with {force_env}=1 / STUDIO_RAM_PREFLIGHT_FORCE=1.)')

def wait_for_host_ram(min_free_gb: float, *, label: str='launch', timeout_s: float=90.0, poll_s: float=1.0, settle_s: float=4.0, emit=None) -> float:
    if os.environ.get('STUDIO_SKIP_RAM_PREFLIGHT', '').strip().lower() in _TRUTHY or os.environ.get('STUDIO_RAM_PREFLIGHT_FORCE', '').strip().lower() in _TRUTHY:
        return free_host_ram_gb()
    import time as _t
    deadline = _t.monotonic() + max(0.0, float(timeout_s))
    last = best = free_host_ram_gb()
    stable_since = None
    reported = False
    while True:
        free = free_host_ram_gb()
        moving = abs(free - last) > 0.3
        if free > best + 0.3:
            deadline, best = (_t.monotonic() + max(0.0, float(timeout_s)), free)
        if free >= float(min_free_gb):
            if moving or stable_since is None:
                stable_since = _t.monotonic()
            if _t.monotonic() - stable_since >= float(settle_s):
                return free
        else:
            stable_since = None
            if emit is not None and (not reported):
                reported = True
                try:
                    emit({'type': 'gen_status', 'text': f'{label}: waiting for host RAM to clear ({free:.1f} GB free, need {float(min_free_gb):.0f} GB)'})
                except Exception:
                    pass
        last = free
        if _t.monotonic() >= deadline:
            return free
        _t.sleep(max(0.2, float(poll_s)))

def guard_python() -> str:
    return sys.executable

def guarded_argv(argv: list, *, log_path, min_free_ram_gb: float=3.0, interval_s: float=3.0, breaches: int=3) -> list:
    guard = Path(__file__).resolve().parent.parent / 'tools' / 'gpu_run_guard.py'
    if not guard.is_file():
        return list(argv)
    return [guard_python(), str(guard), '--log', str(log_path), '--min-ram-mib', str(int(float(min_free_ram_gb) * 1024)), '--interval', str(float(interval_s)), '--breaches', str(int(breaches)), '--'] + list(argv)

class RamWatchdog:

    def __init__(self, proc, *, min_free_gb: float=2.5, poll_s: float=2.0, breaches: int=3, on_trip=None, terminate=None, paging=None):
        import threading
        from .job_paging import GlobalPagingGrowth
        self.proc, self.min_free_gb, self.poll_s, self.breaches, self.on_trip = (proc, float(min_free_gb), float(poll_s), int(breaches), on_trip)
        self.tripped: str | None = None
        self.terminate = terminate
        self.paging = paging if paging is not None else GlobalPagingGrowth()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='ram-watchdog')

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import time as _t
        strikes, noted = (0, float('-inf'))
        while not self._stop.is_set() and self.proc.poll() is None:
            try:
                growth = self.paging.sample()
                reason = None
            except Exception as exc:
                growth, reason = (0, f'paging telemetry failed ({exc})')
            strikes = strikes + 1 if self.paging.exceeded(growth) else 0
            free = free_host_ram_gb()
            if free < self.min_free_gb and _t.monotonic() - noted >= 30:
                print(f'[ram-watchdog] free host RAM {free:.2f} GB < {self.min_free_gb:.1f} GB (logged, not fatal; only paging stops a job)', flush=True)
                noted = _t.monotonic()
            if strikes >= self.breaches and reason is None:
                reason = f'paging: global pagefile/swap grew by {growth} bytes for {strikes} samples'
            if reason is not None:
                self.tripped = reason
                if self.on_trip is not None:
                    try:
                        self.on_trip(self.tripped)
                    except Exception:
                        pass
                try:
                    if self.terminate is not None:
                        self.terminate()
                    else:
                        from .job_paging import graceful_stop
                        stuck = graceful_stop(self.proc, label='RAM-watchdog worker')
                        if stuck is not None:
                            self.tripped = f'{self.tripped}; STUCK: {stuck}'
                except Exception:
                    pass
                return
            _t.sleep(self.poll_s)
