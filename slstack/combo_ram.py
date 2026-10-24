"""Parent-owned Windows shared RAM, physically locked for the whole handoff.

This is an anonymous kernel section, never a filesystem checkpoint. All
transport reads/writes stay <=64KiB. Failure to lock is a visible refusal.
"""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import math
import os
import threading
import time
import uuid
import psutil
from .guard import admit
from .journal import atomic_json

GiB=1024**3
PAGE=65536


def wait_for_host_capacity(host_bytes,floor_bytes,cancelled,*,timeout_s=30,available=None,clock=time.monotonic,sleep=time.sleep):
    """Allow bounded OS RAM reclamation after certified engine retirement."""
    if any(type(n) is not int or n<=0 for n in (host_bytes,floor_bytes)):
        raise ValueError('Explicit integer host/floor budgets required')
    if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or not 0<timeout_s<=30:
        raise ValueError('Host admission settling must be bounded to30seconds')
    available=available or (lambda:psutil.virtual_memory().available)
    started=clock();minimum=None;polls=0;announced=False
    while True:
        free=available();polls+=1
        if type(free) is not int or free<0:raise ValueError('Host admission physical telemetry is invalid')
        minimum=free if minimum is None else min(minimum,free)
        row=dict(host_budget_bytes=host_bytes,physical_floor_bytes=floor_bytes,required_bytes=host_bytes+floor_bytes,
            available_bytes=free,minimum_available_bytes=minimum,seconds=clock()-started,polls=polls,timeout_seconds=timeout_s)
        if cancelled():
            error=RuntimeError('Host admission cancelled before producer launch');error.admission_receipt=row;raise error
        if free>=host_bytes+floor_bytes:return row
        if row['seconds']>=timeout_s:
            error=MemoryError(f"Producer host admission refused after RAM settling: available={free}, required={host_bytes+floor_bytes}")
            error.admission_receipt=row;raise error
        if not announced:
            print('DS4COMBO_HOST_ADMISSION_WAIT '+json.dumps(row),flush=True);announced=True
        sleep(min(.1,timeout_s-row['seconds']))


def kernel_api():
    if os.name!='nt':raise RuntimeError('The one-RTX shared checkpoint requires Windows')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileMappingW.argtypes=[w.HANDLE,ctypes.c_void_p,w.DWORD,w.DWORD,w.DWORD,w.LPCWSTR]
    k.CreateFileMappingW.restype=w.HANDLE
    k.OpenFileMappingW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];k.OpenFileMappingW.restype=w.HANDLE
    k.MapViewOfFile.argtypes=[w.HANDLE,w.DWORD,w.DWORD,w.DWORD,ctypes.c_size_t];k.MapViewOfFile.restype=ctypes.c_void_p
    k.UnmapViewOfFile.argtypes=[ctypes.c_void_p];k.UnmapViewOfFile.restype=w.BOOL
    k.VirtualLock.argtypes=[ctypes.c_void_p,ctypes.c_size_t];k.VirtualLock.restype=w.BOOL
    k.VirtualUnlock.argtypes=[ctypes.c_void_p,ctypes.c_size_t];k.VirtualUnlock.restype=w.BOOL
    k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
    k.GetCurrentProcess.restype=w.HANDLE
    k.GetProcessWorkingSetSize.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessWorkingSetSize.restype=w.BOOL
    k.SetProcessWorkingSetSize.argtypes=[w.HANDLE,ctypes.c_size_t,ctypes.c_size_t];k.SetProcessWorkingSetSize.restype=w.BOOL
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


class LockedRam:
    def __init__(self,capacity,*,session,host_budget_gib,min_available_gib):
        if type(capacity) is not int or not 0<capacity<=2*GiB:raise ValueError('Checkpoint RAM capacity must be explicit and <=2GiB')
        if len(session)!=32 or any(c not in '0123456789abcdef' for c in session):raise ValueError('Exact session nonce required')
        self.capacity=capacity
        self.host_budget_gib=host_budget_gib;self.floor_gib=min_available_gib
        if psutil.Process().memory_info().rss+capacity>host_budget_gib*GiB:
            raise MemoryError('Controller plus locked checkpoint exceeds the explicit host budget')
        admit(capacity/GiB,min_available_gib)
        self.name='Local\\DS4COMBO-'+session+uuid.uuid4().hex
        self.lock=threading.RLock();self.leases=0;self.handle=None;self.address=None;self.locked=False
        self.k=kernel_api()
        try:
            ctypes.set_last_error(0)
            self.handle=self.k.CreateFileMappingW(w.HANDLE(-1),None,4,capacity>>32,capacity&0xffffffff,self.name)
            if not self.handle or ctypes.get_last_error()==183:raise ctypes.WinError(ctypes.get_last_error())
            self.address=self.k.MapViewOfFile(self.handle,6,0,0,capacity)
            if not self.address:raise ctypes.WinError(ctypes.get_last_error())
            process=self.k.GetCurrentProcess();lo,hi=ctypes.c_size_t(),ctypes.c_size_t()
            if not self.k.GetProcessWorkingSetSize(process,ctypes.byref(lo),ctypes.byref(hi)):
                raise ctypes.WinError(ctypes.get_last_error())
            needed=psutil.Process().memory_info().rss+capacity+(32<<20)
            if not self.k.SetProcessWorkingSetSizeEx(process,max(lo.value,needed),max(hi.value,needed+(64<<20)),1|8):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.k.VirtualLock(self.address,capacity):raise ctypes.WinError(ctypes.get_last_error())
            self.locked=True
            if psutil.virtual_memory().available<min_available_gib*GiB:
                raise MemoryError('Available physical RAM floor failed while locking checkpoint')
        except BaseException:
            self.close();raise

    def _range(self,offset,count):
        if not self.address or not self.locked:raise RuntimeError('Checkpoint RAM is closed or not physically locked')
        if type(offset) is not int or type(count) is not int or not 0<=count<=PAGE or not 0<=offset<=self.capacity or count>self.capacity-offset:
            raise ValueError('Checkpoint transfer must be bounded to64KiB and its admitted extent')

    def read(self,offset,count):
        with self.lock:
            self._range(offset,count)
            return ctypes.string_at(self.address+offset,count)

    def write(self,offset,data):
        with self.lock:
            if type(data) is not bytes:raise TypeError('Checkpoint page must be immutable bytes')
            self._range(offset,len(data));ctypes.memmove(self.address+offset,data,len(data))

    def sha256(self,offset,count):
        if type(count) is not int or count<0 or offset<0 or count>self.capacity-offset:raise ValueError('Hash extent exceeds checkpoint RAM')
        digest=hashlib.sha256()
        with self.lease():
            for start in range(0,count,PAGE):digest.update(self.read(offset+start,min(PAGE,count-start)))
        return digest.hexdigest()

    @contextmanager
    def lease(self):
        with self.lock:
            self._range(0,0);self.leases+=1
        try:yield self
        finally:
            with self.lock:self.leases-=1

    def close(self):
        with self.lock:
            if self.leases:raise RuntimeError('Cannot release checkpoint RAM while transport owns it')
            if self.address:
                if self.locked and not self.k.VirtualUnlock(self.address,self.capacity):raise ctypes.WinError(ctypes.get_last_error())
                self.locked=False
                if not self.k.UnmapViewOfFile(self.address):raise ctypes.WinError(ctypes.get_last_error())
                self.address=None
            if self.handle:
                if not self.k.CloseHandle(self.handle):raise ctypes.WinError(ctypes.get_last_error())
                self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
