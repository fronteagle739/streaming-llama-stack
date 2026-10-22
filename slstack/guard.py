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
    from .lease import HoldWatcher
    watch = spec.get('hold_watch')
    watcher = HoldWatcher(watch['path'], watch['identity'], mac=watch.get('mac', False)) if watch else None
    if watcher and watcher.reason():
        raise RuntimeError('Yielding before process creation to shared hold: ' + watcher.reason())
    flags = subprocess.CREATE_NO_WINDOW | 4 if os.name == 'nt' else 0  # suspended until Job admission
    start = time.monotonic()
    peak, minimum, reason = 0, psutil.virtual_memory().available, None
    from .memory_counters import snapshot
    paging_before = snapshot() if spec.get('paging_telemetry') else None
    process_counters = {}
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
                    rss = 0
                    for member in [p, *p.children(recursive=True)]:
                        try:
                            info, created = member.memory_info(), member.create_time()
                            rss += info.rss
                            key = f'{member.pid}:{created:.6f}'
                            faults = getattr(info, 'num_page_faults', None)
                            row = process_counters.setdefault(key, {'pid': member.pid, 'created': created,
                                'name': member.name(), 'first_page_faults': faults, 'last_page_faults': faults,
                                'peak_private_commit_bytes': 0})
                            row['last_page_faults'] = faults
                            row['peak_private_commit_bytes'] = max(row['peak_private_commit_bytes'],
                                getattr(info, 'private', getattr(info, 'pagefile', 0)))
                        except psutil.NoSuchProcess:
                            pass
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
                if watcher:
                    try:
                        foreign = watcher.reason()
                        if foreign:
                            reason = 'Yielding to shared hold: ' + foreign
                    except OSError as exc:
                        reason = 'Shared hold observation failed: ' + str(exc)
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
    receipt.update(paging_before=paging_before, paging_after=snapshot() if spec.get('paging_telemetry') else None,
                   process_memory_counters=list(process_counters.values()),
                   paging_interpretation='Process faults include soft and hard faults. Private commit/pagefile usage is not disk I/O. System counters include other processes; these counters alone do not prove core/KV residency.')
    Path(spec['receipt']).write_text(json.dumps(receipt, indent=2))
    return 0 if code == 0 and reason is None else 1


def supervise(spec):
    from .lease import gpu_lease, recover_leases, AdmissionRefused
    try:
        if spec.get('gpu'):
            waiting = spec['gpu'].get('gpu_wait_seconds', 0)
            deadline = time.monotonic() + waiting
            owner = psutil.Process(spec['owner_pid'])
            while True:
                entered = False
                try:
                    with gpu_lease(spec['gpu']) as identity:
                        entered = True
                        spec['hold_watch'] = {'path': spec['gpu'].get('hold_file') or
                            str(Path(spec['runtime_root']) / 'state' / 'gpu.holds'), 'identity': identity}
                        spec['timeout_s'] = min(spec['timeout_s'], spec['gpu']['lease_minutes'] * 60)
                        return _supervise(spec)
                except (AdmissionRefused, PermissionError):
                    # Retry only before owning a lease/starting a child. Engine or cleanup
                    # failures must never cause an automatic second generation.
                    if entered or not waiting or time.monotonic() >= deadline:
                        raise
                    if (Path(spec['stop']).exists() or not owner.is_running() or
                        abs(owner.create_time() - spec['owner_created']) > .01):
                        raise RuntimeError('GPU admission wait stopped by its owner')
                    if psutil.virtual_memory().available < spec['min_available_gib'] * GiB:
                        raise MemoryError('Available physical RAM fell below the admission-wait floor')
                    time.sleep(.5)
        return _supervise(spec)
    finally:
        # If this outer guardian killed an inner GPU guardian, its Windows Job is
        # now closed. Release that recorded orphan rather than leaving an ACTIVE hold.
        recover_leases(spec['runtime_root'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    args = parser.parse_args()
    sys.exit(supervise(json.loads(Path(args.spec).read_text())))
