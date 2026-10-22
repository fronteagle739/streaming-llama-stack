"""Parent-owned Windows shared RAM, physically locked for the whole handoff.

This is an anonymous kernel section, never a filesystem checkpoint. All
transport reads/writes stay <=64KiB. Failure to lock is a visible refusal.
"""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as w
import hashlib
import os
import threading
import uuid
import psutil
from .guard import admit

GiB=1024**3
PAGE=65536


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
    return k


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
            if not self.k.SetProcessWorkingSetSize(process,max(lo.value,needed),max(hi.value,needed+(64<<20))):
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
