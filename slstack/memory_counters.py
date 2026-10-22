"""Read-only OS paging counters; global activity is not process attribution."""
import ctypes
import os
import re
import subprocess
import sys
import time
import psutil


def diagnostic_snapshot():
    result = {'time_ns': time.time_ns(), 'available_physical_bytes': psutil.virtual_memory().available,
              'scope': 'system-wide; other processes may contribute'}
    if os.name == 'nt':
        from ctypes import wintypes as w
        class Raw(ctypes.Structure):
            _fields_ = [('status', w.DWORD), ('stamp', w.FILETIME), ('first', ctypes.c_int64),
                        ('second', ctypes.c_int64), ('count', w.DWORD)]
        pdh = ctypes.WinDLL('pdh')
        pdh.PdhOpenQueryW.argtypes = [w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
        pdh.PdhAddEnglishCounterW.argtypes = [w.HANDLE, w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
        pdh.PdhCollectQueryData.argtypes = [w.HANDLE]
        pdh.PdhGetRawCounterValue.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD), ctypes.POINTER(Raw)]
        pdh.PdhCloseQuery.argtypes = [w.HANDLE]
        query = w.HANDLE()
        status = pdh.PdhOpenQueryW(None, 0, ctypes.byref(query))
        if status:
            return {**result, 'error': 'PdhOpenQuery: ' + hex(status & 0xffffffff)}
        paths = {'pages_input': r'\Memory\Pages Input/sec', 'pages_output': r'\Memory\Pages Output/sec',
                 'page_reads': r'\Memory\Page Reads/sec', 'page_writes': r'\Memory\Page Writes/sec',
                 'paging_file_usage': r'\Paging File(_Total)\% Usage'}
        handles = {}
        try:
            for name, path in paths.items():
                handle = w.HANDLE()
                status = pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(handle))
                if not status:
                    handles[name] = handle
                else:
                    result[name] = {'error': hex(status & 0xffffffff)}
            status = pdh.PdhCollectQueryData(query)
            if status:
                result['collect_error'] = hex(status & 0xffffffff)
            for name, handle in handles.items():
                raw, kind = Raw(), w.DWORD()
                status = pdh.PdhGetRawCounterValue(handle, ctypes.byref(kind), ctypes.byref(raw))
                result[name] = ({'first': int(raw.first), 'second': int(raw.second),
                                 'type': int(kind.value), 'status': int(raw.status)}
                                if not status else {'error': hex(status & 0xffffffff)})
        finally:
            pdh.PdhCloseQuery(query)
    elif sys.platform == 'darwin':
        run = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=10)
        if run.returncode:
            result['vm_stat_error'] = run.returncode
        else:
            result['vm_stat'] = {key.strip(): int(value) for key, value in
                re.findall(r'^([^:\n]+):\s+(\d+)\.', run.stdout, re.MULTILINE)}
            size = re.search(r'page size of (\d+) bytes', run.stdout)
            result['page_size'] = int(size[1]) if size else None
    else:
        swap = psutil.swap_memory()
        result['swap_in_bytes'], result['swap_out_bytes'] = swap.sin, swap.sout
    return result


def snapshot():
    """Guardian-safe Windows totals without the PDH counter-provider service.

    https://learn.microsoft.com/windows/win32/api/psapi/nf-psapi-enumpagefilesw
    https://learn.microsoft.com/windows/win32/api/psapi/ns-psapi-enum_page_file_information
    TotalInUse is allocated pages, not process commitment or disk-I/O proof.
    The fuller PDH diagnostic remains explicitly available outside the guard.
    """
    if os.name!='nt':return diagnostic_snapshot()
    from ctypes import wintypes as w
    import mmap
    class PageFile(ctypes.Structure):
        _fields_=[('cb',w.DWORD),('reserved',w.DWORD),('total',ctypes.c_size_t),
                  ('used',ctypes.c_size_t),('peak',ctypes.c_size_t)]
    callback_type=ctypes.WINFUNCTYPE(w.BOOL,ctypes.c_void_p,ctypes.POINTER(PageFile),w.LPCWSTR)
    used,total,count,invalid=0,0,0,False
    @callback_type
    def collect(context,info,filename):
        nonlocal used,total,count,invalid
        row=info.contents
        if row.cb<ctypes.sizeof(PageFile) or row.used>row.total:
            invalid=True;return False
        used+=int(row.used);total+=int(row.total);count+=1;return True
    api=ctypes.WinDLL('psapi.dll',use_last_error=True,winmode=2048)
    api.EnumPageFilesW.argtypes=[callback_type,ctypes.c_void_p];api.EnumPageFilesW.restype=w.BOOL
    started=time.monotonic()
    if not api.EnumPageFilesW(collect,None) or invalid:
        raise RuntimeError('Cannot read authoritative global paging totals: '+str(ctypes.get_last_error()))
    return dict(time_ns=time.time_ns(),available_physical_bytes=psutil.virtual_memory().available,
                scope='system-wide; other processes may contribute',api='EnumPageFilesW',page_size=mmap.PAGESIZE,
                pagefile_count=count,paging_file_usage=dict(first=used,second=total),
                query_seconds=time.monotonic()-started)
