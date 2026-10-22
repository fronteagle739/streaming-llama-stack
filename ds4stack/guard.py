"""Independent process-tree guardian, Windows kill-on-close Job, RSS and free-RAM checks."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

GiB = 1024 ** 3


def admit(budget_gib, floor_gib):
    ram = psutil.virtual_memory()
    if not 0 < budget_gib * GiB <= ram.available - floor_gib * GiB:
        raise MemoryError(f'Host budget {budget_gib:.2f} GiB cannot fit: available={ram.available / GiB:.2f}, floor={floor_gib:.2f}')
    return ram.available


def kill_tree(proc):
    try:
        root = psutil.Process(proc.pid)
        children = root.children(recursive=True)
        for p in reversed(children):
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.kill()
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    proc.wait(timeout=10)


def windows_job(proc, commit_gib):
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(x, ctypes.c_uint64) for x in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                  'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
    class Extended(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Extended()
    limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
    limits.JobMemoryLimit = int(commit_gib * GiB)
    try:
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, w.HANDLE(int(proc._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        nt = ctypes.WinDLL('ntdll')
        nt.NtResumeProcess.argtypes = [w.HANDLE]
        if nt.NtResumeProcess(w.HANDLE(int(proc._handle))) != 0:
            raise RuntimeError('Could not resume guarded child')
    except BaseException:
        kernel.CloseHandle(job)
        raise
    return kernel, job


def _supervise(spec):
    budget, floor = spec['host_budget_gib'], spec['min_available_gib']
    admit(budget, floor)
    flags = subprocess.CREATE_NO_WINDOW | 4 if os.name == 'nt' else 0  # suspended until Job admission
    start = time.monotonic()
    peak, minimum, reason = 0, psutil.virtual_memory().available, None
    with open(spec['log'], 'ab', buffering=0) as log:
        child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=spec['env'],
                                 stdin=subprocess.PIPE if 'stdin' in spec else subprocess.DEVNULL,
                                 stdout=log, stderr=log, creationflags=flags,
                                 start_new_session=os.name != 'nt')
        job = None
        try:
            if os.name == 'nt':
                job = windows_job(child, spec['job_commit_gib'])
            if 'stdin' in spec:
                child.stdin.write(spec['stdin'].encode())
                child.stdin.close()
            parent = psutil.Process(spec['owner_pid'])
            while child.poll() is None:
                try:
                    if not parent.is_running() or abs(parent.create_time() - spec['owner_created']) > .01:
                        reason = 'owning process exited'
                except psutil.NoSuchProcess:
                    reason = 'owning process exited'
                try:
                    p = psutil.Process(child.pid)
                    rss = sum(x.memory_info().rss for x in [p, *p.children(recursive=True)] if x.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                free = psutil.virtual_memory().available
                peak, minimum = max(peak, rss), min(minimum, free)
                if rss > budget * GiB or free < floor * GiB:
                    reason = f'HOST MEMORY LIMIT: tree RSS={rss}, available={free}'
                if time.monotonic() - start > spec['timeout_s']:
                    reason = 'deadline exceeded'
                if Path(spec['stop']).exists():
                    reason = 'stop requested'
                if reason:
                    kill_tree(child)
                    break
                time.sleep(.05)
            code = child.wait()
        finally:
            if child.poll() is None:
                kill_tree(child)
            if job:
                job[0].CloseHandle(job[1])
    receipt = {'exit_code': code, 'reason': reason, 'peak_tree_rss': peak,
               'min_available': minimum, 'elapsed_s': time.monotonic() - start,
               'host_budget_gib': budget, 'floor_gib': floor,
               'job_commit_gib': spec['job_commit_gib'], 'child_pid': child.pid}
    Path(spec['receipt']).write_text(json.dumps(receipt, indent=2))
    return 0 if code == 0 and reason is None else 1


def supervise(spec):
    if spec.get('gpu'):
        from .lease import gpu_lease
        with gpu_lease(spec['gpu']):
            # The independent guardian owns and releases the hold even if the launcher is killed.
            spec['timeout_s'] = min(spec['timeout_s'], spec['gpu']['lease_minutes'] * 60)
            return _supervise(spec)
    return _supervise(spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    args = parser.parse_args()
    sys.exit(supervise(json.loads(Path(args.spec).read_text())))
