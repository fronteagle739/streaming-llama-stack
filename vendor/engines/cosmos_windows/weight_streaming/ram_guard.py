# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/ram_guard.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
_GLOBAL_SKIP_ENVS = ('STUDIO_SKIP_RAM_PREFLIGHT',)
_GLOBAL_FORCE_ENVS = ('STUDIO_RAM_PREFLIGHT_FORCE',)
_TRUTHY = ('1', 'true', 'yes', 'on')

def free_host_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        pass
    if sys.platform == 'win32':
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / 1024 ** 3
        except Exception:
            return 0.0
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES') / 1024 ** 3
    except Exception:
        return 0.0

def total_host_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    if sys.platform == 'win32':
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 1024 ** 3
        except Exception:
            return 0.0
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3
    except Exception:
        return 0.0

def _env_set(names) -> bool:
    return any((os.environ.get(n, '').strip().lower() in _TRUTHY for n in names if n))

def preflight_host_ram(need_gb: float, label: str, *, skip_env: str | None=None, force_env: str | None=None, advice: str | None=None) -> None:
    if _env_set(_GLOBAL_SKIP_ENVS) or _env_set(_GLOBAL_FORCE_ENVS):
        return
    if skip_env and os.environ.get(skip_env, '').strip().lower() in _TRUTHY:
        return
    if force_env and os.environ.get(force_env, '').strip().lower() in _TRUTHY:
        return
    free = free_host_ram_gb()
    if free < float(need_gb):
        overrides = 'STUDIO_RAM_PREFLIGHT_FORCE=1'
        if force_env:
            overrides = f'{force_env}=1 / {overrides}'
        tail = f' {advice}' if advice else ' Use the streaming/per-block path (the RAM-bounded route on this box) instead of a resident load.'
        raise RuntimeError(f'[RAM-PREFLIGHT] {label}: needs ~{float(need_gb):.0f} GB free host RAM for a RESIDENT load but only {free:.1f} GB is free. This box is host-RAM constrained; a resident load here would thrash the pagefile and can hard-hang the machine.{tail} (Override to attempt anyway — will page heavily / may OOM — with {overrides}.)')
