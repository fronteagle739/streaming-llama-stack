"""Protect the owned native producer working set before accepting readiness.

The protection function is reused unchanged from a qualified private fork at
543a76c5126f6bcc2400307dd13c82b4407857dd (slstack/combo_ram.py). Only its required Win32 bindings are retained.
"""
import ctypes
from ctypes import wintypes as w
import os
import time
import psutil
from .journal import atomic_json


def kernel_api():
    if os.name != 'nt': raise RuntimeError('Native producer working-set protection requires Windows')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetProcessWorkingSetSizeEx.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(w.DWORD)]
    k.GetProcessWorkingSetSizeEx.restype=w.BOOL
    k.SetProcessWorkingSetSizeEx.argtypes=[w.HANDLE,ctypes.c_size_t,ctypes.c_size_t,w.DWORD]
    k.SetProcessWorkingSetSizeEx.restype=w.BOOL
    k.GetProcessId.argtypes=[w.HANDLE];k.GetProcessId.restype=w.DWORD
    return k


def protect_working_set(job,host_bytes,floor_bytes,aggregate_bytes,*,receipt=None):
    """Protect admitted host pages before bootstrap/readiness of the owned job.

    A hard maximum could force paging; it stays disabled. RSS/commit/floor and
    actual paging remain enforced by the shared guardian.
    https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex
    """
    if any(type(n) is not int or n<=0 for n in (host_bytes,floor_bytes,aggregate_bytes)):
        raise ValueError('Explicit integer working-set/aggregate/floor budgets required')
    controller_rss=psutil.Process().memory_info().rss
    metadata_headroom=64<<20
    if host_bytes+controller_rss+metadata_headroom>aggregate_bytes:
        raise MemoryError('Engine plus controller/state exceeds aggregate resident admission')
    available=psutil.virtual_memory().available
    if host_bytes+floor_bytes>available:
        raise MemoryError('Resident engine working-set admission cannot fit physical RAM')
    # Wait for the new process working set; its Job/RSS/floor budgets already
    # apply before resume. This admission precedes bootstrap and readiness.
    started=time.monotonic();process=psutil.Process(job.pid)
    while process.memory_info().rss<8<<20:
        if job.poll() is not None:raise RuntimeError('Owned process exited before resident admission')
        if time.monotonic()-started>=2:raise TimeoutError('Owned working-set initialization deadline exceeded')
        time.sleep(.001)
    k=kernel_api();handle=job.pi.process
    if k.GetProcessId(handle)!=job.pid:raise ValueError('Suspended process handle/owner PID differs')
    lo,hi,flags=ctypes.c_size_t(),ctypes.c_size_t(),w.DWORD()
    if not k.GetProcessWorkingSetSizeEx(handle,ctypes.byref(lo),ctypes.byref(hi),ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    original=dict(minimum=lo.value,maximum=hi.value,flags=flags.value)
    # HARDWS_MIN_ENABLE=1; HARDWS_MAX_DISABLE=8. The existing locked allocator
    # preserves this flag and retains a larger existing working-set allowance.
    # Keep space between the minimum and the advisory maximum; equal limits
    # are rejected on the target Windows host. The RSS budget stays host_bytes.
    if not k.SetProcessWorkingSetSizeEx(handle,host_bytes,host_bytes+metadata_headroom,1|8):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.GetProcessWorkingSetSizeEx(handle,ctypes.byref(lo),ctypes.byref(hi),ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    if lo.value!=host_bytes or hi.value<host_bytes or not flags.value&1 or flags.value&4:
        raise MemoryError('Windows did not retain the admitted hard-minimum working set')
    row=dict(pid=job.pid,host_budget_bytes=host_bytes,aggregate_budget_bytes=aggregate_bytes,
        controller_rss_bytes=controller_rss,metadata_headroom_bytes=metadata_headroom,
        physical_floor_bytes=floor_bytes,available_before=available,original=original,
        minimum=lo.value,maximum=hi.value,flags=flags.value,initialization_wait_seconds=time.monotonic()-started)
    if receipt is not None:atomic_json(receipt,row)
    return row
