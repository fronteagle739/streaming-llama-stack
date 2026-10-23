# Derived from Studio backend/studio/job_paging.py at d83aa32382f83027631e7954951560c520f9cf6b.
# Shared implementation retained unchanged below this provenance header.
"""Shared paging policy and bounded, process-attributed paging observations.

Windows PagefileUsage is commit, not actual swap. Anonymous page locations use
NtQueryVirtualMemory's extended working-set ABI, as described by System Informer
phnt/ntmmapi.h (MemoryLocationPagefile). No process memory content is read.
The observer is incremental; RSS/commit/physical-floor checks still run every poll.
"""
from __future__ import annotations
import ctypes,errno,math,mmap,os,sys,time
from pathlib import Path

GLOBAL_PAGING_LIMIT=256_000_000


def paging_policy(*,rss,commit,available,rss_limit,commit_limit,floor,global_growth,job_paged=0):
    if any(not math.isfinite(value) or value < 0 for value in
           (rss,commit,available,rss_limit,commit_limit,floor,global_growth,job_paged)):
        return 'memory/paging telemetry is unavailable or invalid'
    if rss>rss_limit or commit>commit_limit or available<floor:
        return f'RAM budget breached: RSS={rss/1e9:.3f}GB commit={commit/1e9:.3f}GB available={available/1e9:.3f}GB'
    if job_paged:
        return f'job-attributed pagefile/swap pages observed: {job_paged} bytes; refusing job paging'
    if global_growth>GLOBAL_PAGING_LIMIT:
        return f'global pagefile/swap grew by {global_growth} bytes, above256MB pressure limit'
    return None


def pagefile_location(flags):
    """NT invalid-page Location=2, excluding page-table metadata."""
    return not flags&1 and not flags&(1<<21) and (flags>>22)&3==2


class JobPaging:
    """Bounded observation of the owned tree; no system-process enumeration."""
    def __init__(self,max_pages=16384):
        if not 1<=max_pages<=65536:raise ValueError('bounded1..65536 page observation required')
        self.max_pages=max_pages;self.cursors={};self.scanned_pages=0;self.completed_scans=0
        self.last_seconds=0.;self.last_processes=[]
        self.attribution='bounded process pagefile observation'
        if sys.platform=='win32':self._windows_api()
        elif sys.platform=='darwin':
            # PROC_PIDREGIONINFO pages_swapped_out includes compressor pages.
            # It cannot establish actual disk swap for this process.
            self.attribution='UNAVAILABLE: Darwin region counter includes compression; physical footprint and global disk-swap guard active'
        elif not sys.platform.startswith('linux'):raise ValueError('job paging observation unsupported on this platform')

    def _windows_api(self):
        if ctypes.sizeof(ctypes.c_void_p)!=8:raise ValueError('64-bit Windows paging observer required')
        from ctypes import wintypes as w
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];self.kernel.OpenProcess.restype=w.HANDLE
        self.kernel.CloseHandle.argtypes=[w.HANDLE];self.kernel.CloseHandle.restype=w.BOOL
        self.kernel.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)]
        self.kernel.GetExitCodeProcess.restype=w.BOOL
        class Region(ctypes.Structure):
            _fields_=[('base',ctypes.c_void_p),('allocation',ctypes.c_void_p),('allocation_protect',w.DWORD),
                ('partition',w.WORD),('size',ctypes.c_size_t),('state',w.DWORD),('protect',w.DWORD),('kind',w.DWORD)]
        assert ctypes.sizeof(Region)==48
        self.Region=Region
        self.kernel.VirtualQueryEx.argtypes=[w.HANDLE,ctypes.c_void_p,ctypes.POINTER(Region),ctypes.c_size_t]
        self.kernel.VirtualQueryEx.restype=ctypes.c_size_t
        self.query=ctypes.WinDLL('ntdll').NtQueryVirtualMemory
        self.query.argtypes=[w.HANDLE,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        self.query.restype=ctypes.c_long
        class Page(ctypes.Structure):
            _fields_=[('address',ctypes.c_void_p),('flags',ctypes.c_size_t)]
        self.Page=Page

    def windows_pages(self,handle,addresses):
        pages=(self.Page*len(addresses))()
        for item,address in zip(pages,addresses):item.address=address
        status=self.query(handle,None,4,pages,ctypes.sizeof(pages),None)
        if status<0:
            if self._windows_exited(handle):raise ProcessLookupError('owned process exited during paging query')
            raise ValueError(f'owned working-set query failed: NTSTATUS{status&0xffffffff:08x}')
        return [int(page.flags) for page in pages]

    def _windows_exited(self,handle):
        code=ctypes.c_ulong()
        return bool(self.kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value!=259)

    def _windows(self,proc,key,limit):
        handle=self.kernel.OpenProcess(0x0400,False,proc.pid)
        if not handle:raise OSError(ctypes.get_last_error(),'owned process paging handle unavailable')
        address=self.cursors.get(key,0);scanned=found=0
        try:
            for _ in range(256):
                region=self.Region()
                if not self.kernel.VirtualQueryEx(handle,address,ctypes.byref(region),ctypes.sizeof(region)):
                    if self._windows_exited(handle):raise ProcessLookupError('owned process exited during region query')
                    if ctypes.get_last_error()==87:
                        address=0;self.completed_scans+=1;break
                    raise OSError(ctypes.get_last_error(),'owned region query failed')
                end=int(region.base or 0)+region.size
                if end<=address:raise ValueError('non-progressing owned virtual-memory scan')
                eligible=region.state==0x1000 and region.kind in (0x20000,0x1000000)
                if region.state==0x1000 and region.kind==0x40000:
                    info=ctypes.create_string_buffer(48)
                    status=self.query(handle,region.base,7,info,48,None)
                    if status>=0:eligible=bool(int.from_bytes(info.raw[12:16],'little')&8)
                if eligible:
                    count=min((end-address)//mmap.PAGESIZE,limit-scanned)
                    if count:
                        flags=self.windows_pages(handle,range(address,address+count*mmap.PAGESIZE,mmap.PAGESIZE))
                        found+=sum(pagefile_location(flag) and (region.kind!=0x1000000
                            or not flag&((1<<15)|(1<<30))) for flag in flags)*mmap.PAGESIZE
                        scanned+=count;address+=count*mmap.PAGESIZE
                    if scanned>=limit or found:break
                address=end
        finally:self.kernel.CloseHandle(handle)
        self.cursors[key]=address;self.scanned_pages+=scanned
        return found

    def _darwin_api(self):
        u32=ctypes.c_uint32;u64=ctypes.c_uint64
        class Region(ctypes.Structure):
            _fields_=[('protection',u32),('maximum',u32),('inheritance',u32),('flags',u32),('offset',u64)]+[
                (name,u32) for name in ('behavior','wired','tag','resident','private_shared','swapped',
                    'dirtied','references','shadow','share_mode','private_resident','shared_resident','object','depth')]+[
                ('address',u64),('size',u64)]
        assert ctypes.sizeof(Region)==96
        self.Region=Region;self.libproc=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        self.libproc.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,u64,ctypes.c_void_p,ctypes.c_int]
        self.libproc.proc_pidinfo.restype=ctypes.c_int

    def _darwin(self,proc,key):
        address=self.cursors.get(key,0);found=0
        for _ in range(64):
            region=self.Region();ctypes.set_errno(0)
            count=self.libproc.proc_pidinfo(proc.pid,7,address,ctypes.byref(region),ctypes.sizeof(region))
            if not count and ctypes.get_errno() in (0,errno.EINVAL,errno.ESRCH):
                address=0;self.completed_scans+=1;break
            if count!=ctypes.sizeof(region):raise OSError(ctypes.get_errno(),'owned Mac region paging query failed')
            found+=int(region.swapped)*mmap.PAGESIZE
            end=int(region.address)+region.size
            if end<=address:raise ValueError('non-progressing owned Mac paging scan')
            address=end
            if found:break
        self.cursors[key]=address
        return found

    def sample(self,proc):
        import psutil
        begin=time.monotonic();found=0;live=set();self.last_processes=[]
        try:items=[proc]+proc.children(recursive=True)
        except psutil.NoSuchProcess:return 0
        if len(items)>256:raise ValueError('owned paging observer process bound exceeded')
        for item in items:
            try:
                key=(item.pid,item.create_time());live.add(key)
                if sys.platform=='win32':value=self._windows(item,key,max(1,self.max_pages//len(items)))
                elif sys.platform=='darwin':value=0
                else:
                    data=Path(f'/proc/{item.pid}/status').read_text()
                    value=next(int(line.split()[1])*1024 for line in data.splitlines() if line.startswith('VmSwap:'))
                found+=value
                if value:self.last_processes.append(dict(pid=item.pid,paged_bytes=value))
            except (psutil.NoSuchProcess,ProcessLookupError,FileNotFoundError):pass
            except OSError:
                if item.is_running():raise
        self.cursors={key:value for key,value in self.cursors.items() if key in live}
        self.last_seconds=time.monotonic()-begin
        return found


class PagingWatch:
    def __init__(self,read_global,*,warning=None,observer=None):
        self.read_global=read_global;self.lowwater=read_global();self.observer=observer or JobPaging()
        self.warning=warning or (lambda message:print(message,flush=True))
        self.last_warning=float('-inf');self.warned_growth=0;self.max_global_growth=0;self.job_paged=0

    def check(self,proc,rss,commit,*,rss_limit,commit_limit,floor):
        import psutil
        available=psutil.virtual_memory().available;current=self.read_global()
        growth=max(0,current-self.lowwater);self.lowwater=min(self.lowwater,current)
        self.max_global_growth=max(self.max_global_growth,growth)
        failure=paging_policy(rss=rss,commit=commit,available=available,rss_limit=rss_limit,
            commit_limit=commit_limit,floor=floor,global_growth=growth)
        if failure:return failure
        if growth>self.warned_growth and time.monotonic()-self.last_warning>=5:
            self.warning(f'[training-memory] WARNING: global pagefile/swap growth={growth}bytes at/below256MB; job attribution checked separately')
            self.last_warning=time.monotonic()
            self.warned_growth=growth
        self.job_paged=self.observer.sample(proc)
        return paging_policy(rss=rss,commit=commit,available=available,rss_limit=rss_limit,
            commit_limit=commit_limit,floor=floor,global_growth=growth,job_paged=self.job_paged)


def paging_bytes():
    """Actual global pagefile/swap usage; private commitment is not paging."""
    if sys.platform != 'win32':
        import psutil
        return psutil.swap_memory().used
    from ctypes import wintypes
    class Info(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('Reserved', wintypes.DWORD),
                    ('TotalSize', ctypes.c_size_t), ('TotalInUse', ctypes.c_size_t),
                    ('PeakUsage', ctypes.c_size_t)]
    pages=[]
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,ctypes.c_void_p,
        ctypes.POINTER(Info),wintypes.LPCWSTR)
    def collect(context,info,name):
        pages.append(info.contents.TotalInUse)
        return True
    callback=callback_type(collect)
    enum=ctypes.WinDLL('psapi',use_last_error=True).EnumPageFilesW
    enum.argtypes=[callback_type,ctypes.c_void_p];enum.restype=wintypes.BOOL
    if not enum(callback,None):
        raise OSError(ctypes.get_last_error(),'actual pagefile usage measurement failed')
    return sum(pages)*mmap.PAGESIZE


def charged_memory(process):
    """Private commit on Windows; charged physical footprint on Darwin."""
    if sys.platform != 'darwin':
        memory=process.memory_info()
        return getattr(memory,'private',memory.rss)
    class Rusage0(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_ubyte*16)]+[(name,ctypes.c_uint64) for name in (
            'user_time','system_time','pkg_idle_wkups','interrupt_wkups','pageins',
            'wired_size','resident_size','phys_footprint','start','exit')]
    query=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True).proc_pid_rusage
    query.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];query.restype=ctypes.c_int
    info=Rusage0()
    if query(process.pid,0,ctypes.byref(info)):
        raise OSError(ctypes.get_errno(),'owned physical footprint measurement failed')
    return info.phys_footprint
