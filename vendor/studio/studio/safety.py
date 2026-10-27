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
    # The pinned job_paging module is a policy library, not a CLI launcher.
    # Stack media already runs in the shared guard's owned process tree. Check
    # that ownership before letting its subprocess inherit the same limits.
    import json
    import psutil
    jobs=(Path(settings.data_root).parent/'jobs').resolve()
    for ancestor in psutil.Process().parents():
        args=ancestor.cmdline()
        module=next((name for name in ('slstack.guard_child','slstack.guard') if name in args),None)
        if not module:continue
        position=args.index(module)
        if position+1>=len(args):continue
        spec_path=Path(args[position+1]).resolve(strict=True)
        if spec_path.parent!=jobs:continue
        spec=json.loads(spec_path.read_text())
        owner=psutil.Process(spec['owner_pid'])
        if abs(owner.create_time()-spec['owner_created'])>.01 or not spec.get('paging_telemetry'):
            raise RuntimeError('Media guard ownership or paging policy differs')
        if sys.platform=='win32':
            import ctypes
            assigned=ctypes.c_int()
            if not ctypes.windll.kernel32.IsProcessInJob(ctypes.c_void_p(-1),None,ctypes.byref(assigned)) or not assigned.value:
                raise RuntimeError('Media worker is outside its Windows owned Job')
        return real_argv
    raise RuntimeError('Launch media through the stack owned process guard')
