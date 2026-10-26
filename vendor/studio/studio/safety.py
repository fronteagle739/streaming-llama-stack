# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Optional
from .config import settings

def ram_abort_floor_gb() -> float:
    try:
        return max(0.25, float(os.environ.get('STUDIO_RAM_ABORT_FLOOR_GB', '1.0')))
    except (TypeError, ValueError):
        return 1.0

def free_ram_gb() -> float:
    try:
        if sys.platform == 'win32':
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullAvailPhys) / 1000000000.0
        else:
            with open('/proc/meminfo', encoding='ascii') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) / 1000000.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1000000000.0
    except Exception:
        return float('inf')

def total_ram_gb() -> float:
    try:
        if sys.platform == 'win32':
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullTotalPhys) / 1000000000.0
        import psutil
        return psutil.virtual_memory().total / 1000000000.0
    except Exception:
        return float('inf')

def enough_ram(min_gb: float) -> tuple[bool, float]:
    free = free_ram_gb()
    return (free >= ram_abort_floor_gb(), free)

def mem_guard_script(repo: Optional[str | Path]=None) -> Optional[Path]:
    root = Path(repo or settings.cosmos_windows_path or '')
    cand = root / 'cosmos_windows' / 'mem_guard.py'
    return cand if cand.is_file() else None

def guard_argv(real_argv: list[str], *, min_gb: float=4.0, repo: Optional[str | Path]=None, python: Optional[str]=None) -> list[str]:
    if not mem_guard_script(repo) or not real_argv:
        return real_argv
    guard = Path(__file__).resolve().parent / 'job_paging.py'
    return [sys.executable, '-I', str(guard), '--log-floor-gb', str(ram_abort_floor_gb()), '--', *real_argv]
