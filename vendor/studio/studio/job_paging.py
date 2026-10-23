# Derived from Studio backend/studio/job_paging.py at 2774636281ab32bf871497f0a8caae23c6b6cdc7.
# Source body unchanged; only this two-line provenance header added.
"""Shared paging policy and bounded, process-attributed paging observations.

Windows PagefileUsage is commit, not actual swap. Anonymous page locations use
NtQueryVirtualMemory's extended working-set ABI, as described by System Informer
phnt/ntmmapi.h (MemoryLocationPagefile). No process memory content is read.
The observer is incremental; RSS/commit/physical-floor checks still run every poll.
"""
from __future__ import annotations
import ctypes,math,mmap,os,sys,time,threading,uuid,struct
from pathlib import Path

GLOBAL_PAGING_LIMIT=256_000_000
OWNED_PAGING_LIMIT=64*1024**2


class PagefileLedger:
    """Last observed page locations, keyed by exact process birth identity.

    A newly launched, gated job has a zero pre-execution baseline. Never use its
    first (possibly already paged) scan as a free allowance. Rescanning replaces
    addresses; it does not accumulate repeated observations as new growth.
    """
    def __init__(self):self.pages={}

    def replace(self,key,start,flags):
        pages=self.pages.setdefault(key,set())
        for index,flag in enumerate(flags):
            address=start+index*mmap.PAGESIZE
            if pagefile_location(flag):pages.add(address)
            else:pages.discard(address)

    def discard_range(self,key,start,end):
        pages=self.pages.get(key,set())
        pages.difference_update([address for address in pages if start<=address<end])

    def retain(self,live):self.pages={key:value for key,value in self.pages.items() if key in live}

    @property
    def total(self):return sum(map(len,self.pages.values()))*mmap.PAGESIZE


class HardFaultWindow:
    """A rate is usable only when its source attributes actual pagefile reads.

    None explicitly means unavailable; it breaks continuity rather than inventing
    a zero observation. Failed/invalid supported telemetry is fatal.
    """
    def __init__(self):self.since=None

    def check(self,rate,now):
        if rate is None:self.since=None;return None
        if not math.isfinite(rate) or rate<0:return 'pagefile hard-fault telemetry is invalid'
        if rate<=100:self.since=None;return None
        if self.since is None:self.since=now
        if now-self.since>=5:return 'owned pagefile hard faults >100/s sustained for 5 seconds'
        return None


class PagefileFaultEvents:
    """ETW HardFault joined to FileIo_Name, never generic process fault counts.

    https://learn.microsoft.com/windows/win32/etw/pagefault-hardfault
    Unknown file objects break attribution; they are never counted as zero I/O.
    The owner supplies only threads of its verified live process identities.
    """
    def __init__(self):
        self.files={};self.owned_threads={};self.count=0;self.unknown=0

    def name(self,file_object,name):
        if len(self.files)>=65536 and file_object not in self.files:
            raise RuntimeError('pagefile fault file-name attribution bound exceeded')
        self.files[file_object]=name.replace('/','\\').lower()

    def fault(self,thread,file_object):
        if thread not in self.owned_threads:return
        name=self.files.get(file_object)
        if name is None:self.unknown+=1
        elif name.rsplit('\\',1)[-1] in ('pagefile.sys','swapfile.sys'):self.count+=1


class WindowsPagefileFaults(PagefileFaultEvents):
    """One uniquely owned, bounded real-time kernel trace; never adopts a session.

    Requires Windows kernel-ETW access. Refusal is explicitly UNAVAILABLE, not
    an alternate generic-fault counter. No trace file, downloads or elevation.
    Layouts follow the installed Windows SDK evntrace.h (64-bit ABI only).
    """
    def __init__(self):
        super().__init__()
        from ctypes import wintypes as w
        if ctypes.sizeof(ctypes.c_void_p)!=8:raise OSError('64-bit ETW required')
        u32=ctypes.c_uint32;u64=ctypes.c_uint64;ptr=ctypes.c_void_p
        class Guid(ctypes.Structure):_fields_=[('data',ctypes.c_ubyte*16)]
        class Wnode(ctypes.Structure):
            _fields_=[('size',u32),('provider',u32),('context',u64),('timestamp',u64),
                ('guid',Guid),('clock',u32),('flags',u32)]
        class Properties(ctypes.Structure):
            _fields_=[('wnode',Wnode)]+[(name,u32) for name in (
                'buffer_kb','min_buffers','max_buffers','max_file','mode','flush','enable','age',
                'buffers','free','lost','written','log_lost','realtime_lost')]+[
                ('logger_thread',ptr),('file_offset',u32),('name_offset',u32)]
        class Header(ctypes.Structure):
            _fields_=[('size',ctypes.c_uint16),('flags',ctypes.c_uint16),('version',u32),
                ('thread',u32),('process',u32),('timestamp',u64),('guid',Guid),('processor',u64)]
        class Event(ctypes.Structure):
            _fields_=[('header',Header),('instance',u32),('parent',u32),('parent_guid',Guid),
                ('data',ptr),('length',u32),('context',u32)]
        class LogHeader(ctypes.Structure):
            _fields_=[('prefix',ctypes.c_byte*56),('logger',ptr),('file',ptr),
                ('timezone',ctypes.c_byte*172),('boot',u64),('frequency',u64),('start',u64),
                ('flags',u32),('lost',u32)]
        class Log(ctypes.Structure):
            _fields_=[('file',w.LPWSTR),('logger',w.LPWSTR),('time',u64),('read',u32),('mode',u32),
                ('event',Event),('header',LogHeader),('buffer_callback',ptr),('buffer_size',u32),
                ('filled',u32),('lost',u32),('callback',ptr),('kernel',u32),('context',ptr)]
        if (ctypes.sizeof(Properties),ctypes.sizeof(Header),ctypes.sizeof(Event),ctypes.sizeof(LogHeader),ctypes.sizeof(Log))!=(120,48,88,280,448):
            raise RuntimeError('ETW structure ABI differs')
        self.api=ctypes.WinDLL('advapi32',use_last_error=True)
        self.api.StartTraceW.argtypes=[ctypes.POINTER(u64),w.LPCWSTR,ptr];self.api.StartTraceW.restype=u32
        self.api.ControlTraceW.argtypes=[u64,w.LPCWSTR,ptr,u32];self.api.ControlTraceW.restype=u32
        self.api.OpenTraceW.argtypes=[ctypes.POINTER(Log)];self.api.OpenTraceW.restype=u64
        self.api.ProcessTrace.argtypes=[ctypes.POINTER(u64),u32,ptr,ptr];self.api.ProcessTrace.restype=u32
        self.api.CloseTrace.argtypes=[u64];self.api.CloseTrace.restype=u32
        self.name_text='StudioOwnedPaging-'+uuid.uuid4().hex
        self.storage=ctypes.create_string_buffer(ctypes.sizeof(Properties)+1024)
        self.properties=Properties.from_buffer(self.storage)
        p=self.properties;p.wnode.size=len(self.storage);p.wnode.flags=0x20000;p.wnode.clock=1
        p.wnode.guid=Guid((ctypes.c_ubyte*16).from_buffer_copy(uuid.uuid4().bytes_le))
        p.buffer_kb=64;p.min_buffers=2;p.max_buffers=16;p.flush=1
        # NO_PER_PROCESSOR_BUFFERING keeps the bounded pool independent of CPU count.
        p.mode=0x12000100;p.enable=0x2000|0x200 # SYSTEM_LOGGER, REAL_TIME, HARD_FAULTS, DISK_FILE_IO
        p.name_offset=ctypes.sizeof(Properties)
        name_buffer=ctypes.create_unicode_buffer(self.name_text)
        ctypes.memmove(ctypes.addressof(self.storage)+p.name_offset,name_buffer,ctypes.sizeof(name_buffer))
        self.session=u64();self.trace=None;self.thread=None;self.error=None;self.closed=False
        status=self.api.StartTraceW(ctypes.byref(self.session),self.name_text,self.storage)
        if status:raise OSError(status,'kernel pagefile ETW unavailable')
        self.lock=threading.Lock()
        fault_guid=uuid.UUID('3d6fa8d3-fe05-11d0-9dda-00c04fd7ba7c').bytes_le
        file_guid=uuid.UUID('90cbdc39-4a3e-11d1-84f4-0000f80464e3').bytes_le
        @ctypes.WINFUNCTYPE(None,ctypes.POINTER(Event))
        def callback(pointer):
            try:
                e=pointer.contents;kind=e.header.version&255;guid=bytes(e.header.guid.data)
                with self.lock:
                    if guid==file_guid and kind in (0,32,36) and e.length>=10:
                        data=ctypes.string_at(e.data,e.length)
                        self.name(int.from_bytes(data[:8],'little'),data[8:].decode('utf-16-le').rstrip('\0'))
                    elif guid==file_guid and kind==35 and e.length>=8:
                        self.files.pop(ctypes.c_uint64.from_address(e.data).value,None)
                    elif guid==fault_guid and kind==32 and e.length>=40:
                        data=ctypes.string_at(e.data,e.length)
                        self.fault(struct.unpack_from('<I',data,32)[0],struct.unpack_from('<Q',data,24)[0])
            except BaseException as error:self.error=repr(error)
        self.callback=callback
        self.log=Log();self.log.logger=self.name_text;self.log.mode=0x100
        self.log.callback=ctypes.cast(callback,ptr)
        try:
            handle=self.api.OpenTraceW(ctypes.byref(self.log))
            if handle==2**64-1:raise ctypes.WinError(ctypes.get_last_error())
            self.trace=u64(handle)
            def consume():
                status=self.api.ProcessTrace(ctypes.byref(self.trace),1,None,None)
                if not self.closed:self.error='ETW consumer exited: '+str(status)
            self.thread=threading.Thread(target=consume,name='owned-pagefile-etw',daemon=True);self.thread.start()
            self.previous=time.monotonic();self.previous_count=0;self.previous_unknown=0
        except BaseException:
            self.close();raise

    def sample(self,items):
        threads={}
        for process in items:
            try:
                identity=(process.pid,process.create_time())
                for thread in process.threads():threads[thread.id]=identity
            except __import__('psutil').NoSuchProcess:pass
        # EVENT_TRACE_LOGFILE fields alone do not provide live session loss
        # statistics. Query our session once per rate interval, failing closed
        # if an established source loses events or can no longer be queried.
        now=time.monotonic();elapsed=now-self.previous
        if elapsed>=1:
            status=self.api.ControlTraceW(self.session,self.name_text,self.storage,0)
            if status:raise RuntimeError('pagefile ETW statistics query failed: '+str(status))
        with self.lock:
            self.owned_threads=threads
            if self.error:raise RuntimeError('pagefile hard-fault telemetry failed: '+self.error)
            if (self.log.lost or self.log.header.lost or self.properties.lost
                    or self.properties.log_lost or self.properties.realtime_lost):
                raise RuntimeError('pagefile ETW events lost')
            # A full second avoids extrapolating individual sparse events.
            if elapsed<1:return getattr(self,'rate',0.)
            self.rate=None if self.unknown!=self.previous_unknown else (self.count-self.previous_count)/elapsed
            self.previous_count=self.count;self.previous_unknown=self.unknown;self.previous=now
            return self.rate

    def close(self):
        if self.closed:return
        self.closed=True
        if self.session.value:
            status=self.api.ControlTraceW(self.session,self.name_text,self.storage,1)
            if status not in (0,4201):raise OSError(status,'owned ETW session stop failed')
            self.session.value=0
        if self.trace is not None:self.api.CloseTrace(self.trace)
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():raise RuntimeError('owned ETW consumer exit unconfirmed')


def paging_policy(*,rss,commit,available,rss_limit,commit_limit,floor,global_growth,job_paged=0,
                  owned_paging_limit=OWNED_PAGING_LIMIT):
    if any(not math.isfinite(value) or value < 0 for value in
           (rss,commit,available,rss_limit,commit_limit,floor,global_growth,job_paged,owned_paging_limit)):
        return 'memory/paging telemetry is unavailable or invalid'
    if owned_paging_limit>OWNED_PAGING_LIMIT:return 'owned paging allowance exceeds shared maximum'
    if rss>rss_limit or commit>commit_limit or available<floor:
        return f'RAM budget breached: RSS={rss/1e9:.3f}GB commit={commit/1e9:.3f}GB available={available/1e9:.3f}GB'
    if job_paged>owned_paging_limit:
        return f'job-attributed pagefile/swap growth {job_paged} bytes exceeds {owned_paging_limit/1024**2:g} MiB allowance'
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
        self.last_seconds=0.;self.last_processes=[];self.ledger=PagefileLedger()
        self.hard_fault_rate=None;self.fault_source=None
        self.hard_fault_attribution='UNAVAILABLE: pagefile-specific hard-fault events are not available on this platform'
        self.attribution='bounded process pagefile observation'
        if sys.platform=='win32':
            self._windows_api()
            try:
                self.fault_source=WindowsPagefileFaults()
                self.hard_fault_attribution='ETW HardFault joined to pagefile FileIo_Name for verified owned threads'
            except OSError as error:
                if error.errno not in (5,50,1314):raise
                self.hard_fault_attribution='UNAVAILABLE: Windows kernel ETW access refused: '+str(error)
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
                        # Image-shared pages are file-backed, not owned paging.
                        self.ledger.replace(key,address-count*mmap.PAGESIZE,
                            [0 if region.kind==0x1000000 and flag&((1<<15)|(1<<30)) else flag for flag in flags])
                    if scanned>=limit:break
                else:self.ledger.discard_range(key,address,end)
                address=end
        finally:self.kernel.CloseHandle(handle)
        self.cursors[key]=address;self.scanned_pages+=scanned
        return len(self.ledger.pages.get(key,()))*mmap.PAGESIZE

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
        self.ledger.retain(live)
        if self.fault_source:
            self.hard_fault_rate=self.fault_source.sample(items)
            if self.hard_fault_rate is None:
                self.hard_fault_attribution='UNAVAILABLE: owned hard-fault file object has no ETW filename; continuity reset'
        self.last_seconds=time.monotonic()-begin
        return found

    def close(self):
        if self.fault_source:self.fault_source.close()


class PagingWatch:
    def __init__(self,read_global,*,warning=None,observer=None,owned_paging_limit=None):
        self.owned_paging_limit=OWNED_PAGING_LIMIT if owned_paging_limit is None else owned_paging_limit
        if not math.isfinite(self.owned_paging_limit) or not 0<=self.owned_paging_limit<=OWNED_PAGING_LIMIT:
            raise ValueError('owned paging allowance must be finite and between zero and the shared maximum')
        self.read_global=read_global;self.lowwater=read_global();self.observer=observer or JobPaging()
        self.warning=warning or (lambda message:print(message,flush=True))
        self.last_warning=float('-inf');self.warned_growth=0;self.max_global_growth=0;self.job_paged=0
        self.peak_job_paged=0;self.warned_owned=False;self.fault_window=HardFaultWindow()
        self.fault_attribution=getattr(self.observer,'hard_fault_attribution',
            'UNAVAILABLE: observer supplies no pagefile-specific hard-fault telemetry')
        self.hard_fault_rate=None;self.warned_fault_unavailable=False

    def check(self,proc,rss,commit,*,rss_limit,commit_limit,floor):
        import psutil
        available=psutil.virtual_memory().available;current=self.read_global()
        growth=max(0,current-self.lowwater);self.lowwater=min(self.lowwater,current)
        self.max_global_growth=max(self.max_global_growth,growth)
        failure=paging_policy(rss=rss,commit=commit,available=available,rss_limit=rss_limit,
            commit_limit=commit_limit,floor=floor,global_growth=growth,owned_paging_limit=self.owned_paging_limit)
        if failure:return failure
        if growth>self.warned_growth and time.monotonic()-self.last_warning>=5:
            self.warning(f'[training-memory] WARNING: global pagefile/swap growth={growth}bytes at/below256MB; job attribution checked separately')
            self.last_warning=time.monotonic()
            self.warned_growth=growth
        self.job_paged=self.observer.sample(proc)
        self.peak_job_paged=max(self.peak_job_paged,self.job_paged)
        if 0<self.job_paged<=self.owned_paging_limit and not self.warned_owned:
            self.warning(f'[training-memory] WARNING: owned pagefile/swap growth={self.job_paged} bytes within {self.owned_paging_limit/1024**2:g} MiB allowance')
            self.warned_owned=True
        self.hard_fault_rate=getattr(self.observer,'hard_fault_rate',None)
        self.fault_attribution=getattr(self.observer,'hard_fault_attribution',self.fault_attribution)
        if self.hard_fault_rate is None and not self.warned_fault_unavailable:
            self.warning('[training-memory] WARNING: '+self.fault_attribution)
            self.warned_fault_unavailable=True
        fault=self.fault_window.check(self.hard_fault_rate,time.monotonic())
        if fault:return fault
        return paging_policy(rss=rss,commit=commit,available=available,rss_limit=rss_limit,
            commit_limit=commit_limit,floor=floor,global_growth=growth,job_paged=self.job_paged,
            owned_paging_limit=self.owned_paging_limit)

    def details(self):
        return dict(owned_paging_limit_bytes=self.owned_paging_limit,owned_paging_baseline='zero at gated job launch',
            peak_owned_paged_bytes=self.peak_job_paged,pagefile_hard_fault_rate=self.hard_fault_rate,
            pagefile_hard_fault_attribution=self.fault_attribution,
            pagefile_hard_fault_limit_per_second=100,pagefile_hard_fault_duration_seconds=5,
            scan_pages=getattr(self.observer,'scanned_pages',None),
            last_paged_processes=getattr(self.observer,'last_processes',None))

    def close(self):
        close=getattr(self.observer,'close',None)
        if close:close()


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
