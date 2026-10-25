# Derived from Studio backend/studio/dsv41_windows_runner.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal source export; transport and owned process logic unchanged.
from __future__ import annotations
from contextlib import contextmanager, ExitStack
import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import logging
import subprocess
import sys
import time
import uuid
from .dsv41_session_contract import Refused, require

log = logging.getLogger(__name__)

# Studio c0403208a7 (owner rule: never force-kill a process that may hold VRAM or
# DirectStorage resources). Run detached: attach to the owned root's windowless
# console and raise Ctrl-Break there; the helper shields itself with its own handler
# registered after AttachConsole and waits for it to see the event before leaving.
_CTRL_BREAK = """import ctypes,os,sys,threading
from ctypes import wintypes as W
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.FreeConsole()
if not k.AttachConsole(int(sys.argv[1])):os._exit(20+min(ctypes.get_last_error(),50))
seen=threading.Event()
H=ctypes.WINFUNCTYPE(W.BOOL,W.DWORD)
def handler(kind):
    seen.set()
    return 1
keep=H(handler)
k.SetConsoleCtrlHandler.argtypes=[H,W.BOOL]
if not k.SetConsoleCtrlHandler(keep,True):os._exit(11)
if not k.GenerateConsoleCtrlEvent(1,0):os._exit(10)
os._exit(0 if seen.wait(5) else 12)
"""


class StuckProcess(RuntimeError):
    """A resumed process still alive after every graceful step: left running, never terminated."""


def stuck_message(pid, reason):
    return "DSV41 STUCK PROCESS pid=%s holding VRAM, NOT killed; owner action needed (%s)" % (pid, reason)

class _Memory(C.Structure):
    _fields_ = [('length', W.DWORD), ('load', W.DWORD)] + [(name, C.c_ulonglong) for name in ('total', 'available', 'page', 'page_free', 'virtual', 'virtual_free', 'extended')]

class _Performance(C.Structure):
    _fields_ = [('cb', W.DWORD)] + [(name, C.c_size_t) for name in ('commit', 'limit', 'peak', 'physical', 'available', 'cache', 'kernel', 'paged', 'nonpaged', 'page')] + [(name, W.DWORD) for name in ('handles', 'processes', 'threads')]

class _Basic(C.Structure):
    _fields_ = [('process_time', C.c_longlong), ('job_time', C.c_longlong), ('flags', W.DWORD), ('min_ws', C.c_size_t), ('max_ws', C.c_size_t), ('active', W.DWORD), ('affinity', C.c_size_t), ('priority', W.DWORD), ('scheduling', W.DWORD)]

class _Limits(C.Structure):
    _fields_ = [('basic', _Basic), ('io', C.c_ulonglong * 6)] + [(name, C.c_size_t) for name in ('process_memory', 'job_memory', 'peak_process', 'peak_job')]

class _Accounting(C.Structure):
    _fields_ = [(name, C.c_longlong) for name in ('user', 'kernel', 'period_user', 'period_kernel')] + [(name, W.DWORD) for name in ('faults', 'total', 'active', 'terminated')]

class _Startup(C.Structure):
    _fields_ = [('cb', W.DWORD), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR), ('title', W.LPWSTR)] + [(name, W.DWORD) for name in ('x', 'y', 'xsize', 'ysize', 'xchars', 'ychars', 'fill', 'flags')] + [('show', W.WORD), ('reserved_size', W.WORD), ('reserved_data', C.c_void_p), ('stdin', W.HANDLE), ('stdout', W.HANDLE), ('stderr', W.HANDLE)]

class _StartupEx(C.Structure):
    _fields_ = [('startup', _Startup), ('attributes', C.c_void_p)]

class _Process(C.Structure):
    _fields_ = [('process', W.HANDLE), ('thread', W.HANDLE), ('pid', W.DWORD), ('tid', W.DWORD)]

class _Native:

    def __init__(self):
        require(os.name == 'nt' and C.sizeof(C.c_void_p) == 8, 'Windows x64 runner required')
        self.k = C.WinDLL('kernel32', use_last_error=True, winmode=2048)
        signatures = {'CreateFileW': ([W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE], W.HANDLE), 'CloseHandle': ([W.HANDLE], W.BOOL), 'GetFileInformationByHandleEx': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL), 'GetFinalPathNameByHandleW': ([W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD], W.DWORD), 'QueryDosDeviceW': ([W.LPCWSTR, W.LPWSTR, W.DWORD], W.DWORD), 'GlobalMemoryStatusEx': ([C.POINTER(_Memory)], W.BOOL), 'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE), 'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL), 'QueryInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL), 'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL), 'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL), 'TerminateProcess': ([W.HANDLE, W.UINT], W.BOOL), 'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD), 'GetExitCodeProcess': ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL), 'ResumeThread': ([W.HANDLE], W.DWORD), 'CreateProcessW': ([W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p, W.BOOL, W.DWORD, C.c_void_p, W.LPCWSTR, C.POINTER(_StartupEx), C.POINTER(_Process)], W.BOOL), 'InitializeProcThreadAttributeList': ([C.c_void_p, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)], W.BOOL), 'UpdateProcThreadAttribute': ([C.c_void_p, W.DWORD, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p], W.BOOL), 'DeleteProcThreadAttributeList': ([C.c_void_p], None), 'GetCurrentProcess': ([], W.HANDLE), 'GetWindowsDirectoryW': ([W.LPWSTR, W.UINT], W.UINT), 'K32EnumProcessModulesEx': ([W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), W.DWORD], W.BOOL), 'K32GetModuleFileNameExW': ([W.HANDLE, W.HANDLE, W.LPWSTR, W.DWORD], W.DWORD), 'DuplicateHandle': ([W.HANDLE, W.HANDLE, W.HANDLE, C.POINTER(W.HANDLE), W.DWORD, W.BOOL, W.DWORD], W.BOOL)}
        for name, (args, result) in signatures.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = (args, result)

    @staticmethod
    def ok(result):
        if not result:
            raise C.WinError(C.get_last_error())
        return result

    def memory(self):
        value = _Memory()
        value.length = C.sizeof(value)
        self.ok(self.k.GlobalMemoryStatusEx(C.byref(value)))
        return value.available

    def commit(self):
        psapi = C.WinDLL('psapi.dll', use_last_error=True, winmode=2048)
        fn = psapi.GetPerformanceInfo
        fn.argtypes, fn.restype = ([C.POINTER(_Performance), W.DWORD], W.BOOL)
        value = _Performance()
        value.cb = C.sizeof(value)
        self.ok(fn(C.byref(value), C.sizeof(value)))
        return (int(value.commit * value.page), int(value.limit * value.page))

    def _open(self, path, *, directory=False, exclusive=False):
        path = Path(path)
        require(path.is_absolute() and len(path.drive) == 2 and ('..' not in path.parts), 'A physical local absolute path is required')
        require(all((':' not in part for part in path.parts[1:])), 'Alternate data streams refused')
        device = C.create_unicode_buffer(32768)
        self.ok(self.k.QueryDosDeviceW(path.drive, device, len(device)))
        require(device.value.startswith('\\Device\\HarddiskVolume'), 'Aliases/network volumes refused')
        access, share, disposition = (3221225472, 0, 4) if exclusive else (128 if directory else 2147483648, 1, 3)
        flags = 2097152 | (33554432 if directory else 128)
        handle = self.k.CreateFileW(str(path), access, share, None, disposition, flags, None)
        if handle == C.c_void_p(-1).value:
            raise C.WinError(C.get_last_error())
        try:
            info = (W.DWORD * 2)()
            self.ok(self.k.GetFileInformationByHandleEx(handle, 9, info, C.sizeof(info)))
            require(not info[0] & 1024 and bool(info[0] & 16) == directory, 'Reparse or unexpected file type refused')
            final = C.create_unicode_buffer(32768)
            size = self.ok(self.k.GetFinalPathNameByHandleW(handle, final, len(final), 0))
            require(size < len(final) and os.path.normcase(final.value.removeprefix('\\\\?\\')) == os.path.normcase(str(path)), 'Physical path changed or aliases an input')
            return handle
        except BaseException:
            self.k.CloseHandle(handle)
            raise

    @contextmanager
    def directory(self, path):
        handles = []
        try:
            for parent in [*reversed(Path(path).parents), Path(path)]:
                handles.append(self._open(parent, directory=True))
            yield
        finally:
            for handle in reversed(handles):
                self.k.CloseHandle(handle)

    @contextmanager
    def read_lease(self, path):
        import msvcrt
        with self.directory(Path(path).parent):
            handle = self._open(path)
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                self.k.CloseHandle(handle)
                raise
            with os.fdopen(fd, 'rb') as stream:
                yield stream

    @contextmanager
    def exclusive(self, path):
        with self.directory(Path(path).parent):
            handle = self._open(path, exclusive=True)
            try:
                yield
            finally:
                self.k.CloseHandle(handle)

    def gpu(self, ordinal, expected_uuid):
        driver = C.WinDLL('nvcuda.dll', winmode=2048)

        def cuda(name, args, *values):
            fn = getattr(driver, name)
            fn.argtypes, fn.restype = (args, C.c_int)
            require(fn(*values) == 0, 'CUDA identity query failed: ' + name)
        cuda('cuInit', [C.c_uint], 0)
        device = C.c_int()
        cuda('cuDeviceGet', [C.POINTER(C.c_int), C.c_int], C.byref(device), ordinal)
        identifier = (C.c_ubyte * 16)()
        cuda('cuDeviceGetUuid_v2', [C.c_void_p, C.c_int], identifier, device)
        require('GPU-' + str(uuid.UUID(bytes=bytes(identifier))) == expected_uuid, 'CUDA ordinal is not the configured GPU UUID')
        nvml = C.WinDLL('nvml.dll', winmode=2048)

        def call(name, args, *values):
            fn = getattr(nvml, name)
            fn.argtypes, fn.restype = (args, C.c_int)
            require(fn(*values) == 0, 'NVML admission unavailable: ' + name)
        call('nvmlInit_v2', [])
        try:
            handle = C.c_void_p()
            call('nvmlDeviceGetHandleByUUID', [C.c_char_p, C.POINTER(C.c_void_p)], expected_uuid.encode('ascii'), C.byref(handle))
            name = C.create_string_buffer(128)
            call('nvmlDeviceGetName', [C.c_void_p, C.c_char_p, C.c_uint], handle, name, len(name))
            major, minor = C.c_int(), C.c_int()
            call('nvmlDeviceGetCudaComputeCapability', [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)],
                 handle, C.byref(major), C.byref(minor))
            require(major.value * 10 + minor.value >= 120,
                    'Selected CUDA device lacks the compute capability required by this backend')
            memory = (C.c_ulonglong * 3)()
            call('nvmlDeviceGetMemoryInfo', [C.c_void_p, C.c_void_p], handle, memory)
            return int(memory[1])
        finally:
            call('nvmlShutdown', [])

    def suspended(self, argv, env, directory, stdin, stdout, stderr):
        return _Job(self, argv, env, directory, stdin, stdout, stderr)

    def windows_directory(self):
        value = C.create_unicode_buffer(32768)
        count = self.ok(self.k.GetWindowsDirectoryW(value, len(value)))
        require(count < len(value), 'Windows platform root exceeds bound')
        return Path(value.value)

    def modules(self, process):
        modules = (W.HANDLE * 2048)()
        needed = W.DWORD()

        def observation_error(api, index=None):
            error = C.WinError(C.get_last_error())
            error.module_api = api
            error.strerror = api + (' module[' + str(index) + ']' if index is not None else '') + ': ' + error.strerror
            return error
        if not self.k.K32EnumProcessModulesEx(process, modules, C.sizeof(modules), C.byref(needed), 3):
            raise observation_error('K32EnumProcessModulesEx')
        require(0 < needed.value <= C.sizeof(modules) and needed.value % C.sizeof(W.HANDLE) == 0, 'Actual loaded module list exceeds bound')
        paths = []
        for index, handle in enumerate(modules[:needed.value // C.sizeof(W.HANDLE)]):
            value = C.create_unicode_buffer(32768)
            count = self.k.K32GetModuleFileNameExW(process, handle, value, len(value))
            if not count:
                raise observation_error('K32GetModuleFileNameExW', index)
            require(count < len(value), 'Actual loaded module path exceeds bound')
            paths.append(Path(value.value))
        return paths

class _Job:

    def __init__(self, native, argv, env, directory, stdin, stdout, stderr):
        import msvcrt
        self.n, self.pi, self.job = (native, _Process(), None)
        self.assigned = self.resumed = False
        duplicate, attributes, attributes_initialized = ([], None, False)
        try:
            self.job = native.ok(native.k.CreateJobObjectW(None, None))
            limits = _Limits()
            limits.basic.flags = 8192
            native.ok(native.k.SetInformationJobObject(self.job, 9, C.byref(limits), C.sizeof(limits)))
            current = native.k.GetCurrentProcess()
            for fd in (stdin, stdout, stderr):
                handle = W.HANDLE()
                native.ok(native.k.DuplicateHandle(current, msvcrt.get_osfhandle(fd), current, C.byref(handle), 0, True, 2))
                duplicate.append(handle)
            size = C.c_size_t()
            native.k.InitializeProcThreadAttributeList(None, 1, 0, C.byref(size))
            require(0 < size.value < 65536, 'Invalid process attribute-list size')
            attributes = C.create_string_buffer(size.value)
            native.ok(native.k.InitializeProcThreadAttributeList(attributes, 1, 0, C.byref(size)))
            attributes_initialized = True
            handles = (W.HANDLE * 3)(*[h.value for h in duplicate])
            native.ok(native.k.UpdateProcThreadAttribute(attributes, 0, 131074, handles, C.sizeof(handles), None, None))
            si = _StartupEx()
            si.startup.cb, si.startup.flags = (C.sizeof(si), 256)
            si.startup.stdin, si.startup.stdout, si.startup.stderr = [h.value for h in duplicate]
            si.attributes = C.cast(attributes, C.c_void_p)
            block = C.create_unicode_buffer('\x00'.join((f'{k}={v}' for k, v in sorted(env.items(), key=lambda p: p[0].upper()))) + '\x00\x00')
            command = C.create_unicode_buffer(subprocess.list2cmdline(argv))
            flags = 4 | 1024 | 524288 | 134217728 | 16384
            native.ok(native.k.CreateProcessW(argv[0], command, None, None, True, flags, block, str(directory), C.byref(si), C.byref(self.pi)))
            if not native.k.AssignProcessToJobObject(self.job, self.pi.process):
                error = C.get_last_error()
                native.ok(native.k.TerminateProcess(self.pi.process, 125))
                require(native.k.WaitForSingleObject(self.pi.process, 30000) == 0, 'Unassigned suspended child exit unconfirmed')
                raise C.WinError(error)
            self.assigned = True
        except BaseException:
            if self.assigned:
                self.stop()
            self.close()
            raise
        finally:
            if attributes_initialized:
                native.k.DeleteProcThreadAttributeList(attributes)
            for handle in duplicate:
                native.k.CloseHandle(handle)

    @property
    def pid(self):
        return int(self.pi.pid)

    def resume(self):
        require(self.n.k.ResumeThread(self.pi.thread) != 4294967295, 'Owned worker resume failed')
        self.resumed = True

    def modules(self):
        return self.n.modules(self.pi.process)

    def active(self):
        accounting = _Accounting()
        self.n.ok(self.n.k.QueryInformationJobObject(self.job, 1, C.byref(accounting), C.sizeof(accounting), None))
        return accounting.active

    def poll(self):
        status = self.n.k.WaitForSingleObject(self.pi.process, 0)
        require(status in (0, 258), 'Owned root state is unavailable')
        if status == 258:
            return None
        value = W.DWORD()
        self.n.ok(self.n.k.GetExitCodeProcess(self.pi.process, C.byref(value)))
        return int(value.value)

    def interrupt(self):
        done = subprocess.run([sys.executable, '-I', '-c', _CTRL_BREAK, str(self.pid)], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x8, timeout=15)
        if done.returncode:
            raise OSError('Ctrl-Break helper failed with exit code %d' % done.returncode)
        return True

    def _wait_empty(self, seconds):
        deadline = time.monotonic() + seconds
        while self.active() and time.monotonic() < deadline:
            time.sleep(0.05)
        return not self.active()

    def release_kill_on_close(self):
        limits = _Limits()
        limits.basic.flags = 0
        self.n.ok(self.n.k.SetInformationJobObject(self.job, 9, C.byref(limits), C.sizeof(limits)))

    def stop(self, graceful_seconds=0, stuck_seconds=None):
        """Never kills a resumed process (it may hold VRAM): Ctrl-Break, graceful_seconds,
        a second Ctrl-Break, up to stuck_seconds in total, then StuckProcess with the
        process left running and kill-on-close cleared. A never-resumed root ran no code
        and holds nothing: that one is terminated."""
        if not self.active():
            return 'exited'
        if not self.resumed:
            self.n.ok(self.n.k.TerminateJobObject(self.job, 125))
            deadline = time.monotonic() + 30
            while self.active():
                require(time.monotonic() < deadline, 'Unstarted worker exit cannot be confirmed')
                time.sleep(0.05)
            self.n.k.WaitForSingleObject(self.pi.process, 2000)
            return 'terminate-unstarted'
        stuck_seconds = max(graceful_seconds, stuck_seconds if stuck_seconds is not None else graceful_seconds)
        failures = []
        for attempt, window in ((1, graceful_seconds), (2, stuck_seconds - graceful_seconds)):
            try:
                self.interrupt()
            except Exception as exc:
                failures.append('Ctrl-Break %d: %s' % (attempt, repr(exc)[:200]))
            if self._wait_empty(window):
                self.n.k.WaitForSingleObject(self.pi.process, 2000)
                code = self.poll()
                how = 'graceful' if attempt == 1 else 'graceful-late'
                return how if code in (0, 130) else '%s-crashed exit=%s' % (how, code)
        reason = 'alive %ds after Ctrl-Break%s' % (stuck_seconds, ('; ' + '; '.join(failures)) if failures else '')
        self.release_kill_on_close()
        message = stuck_message(self.pid, reason)
        log.critical(message)
        print(message, file=sys.stderr, flush=True)
        raise StuckProcess(message)

    def close(self):
        for handle in (self.pi.thread, self.pi.process, self.job):
            if handle:
                self.n.k.CloseHandle(handle)
        self.pi, self.job = (_Process(), None)
