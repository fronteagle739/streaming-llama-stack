"""Independent process-tree guardian with cooperative retirement and memory checks."""
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
from .progress import Window, activity
from .memory_counters import available_physical_bytes
from .native_broker import vendor
vendor()
from studio.job_guard import OwnedProcesses, OwnedView
from studio.job_paging import PagingWatch, paging_bytes

GiB = 1024 ** 3


from .shutdown import ProgressOwnedProcesses as GracefulOwnedProcesses


class GPUWindowReader:
    """Retry atomic-publication permission races until a whole progress window is idle.

    Keep validating the last complete window (including its deadline and lease)
    while the rest of the guard continues polling RAM, ownership and paging.
    Missing, malformed and persistently unreadable metadata still fail closed.
    """
    def __init__(self, path, seconds=30, *, clock=None):
        self.path = Path(path)
        self.window = None
        self.progress = Window(seconds, clock=clock)
        self.reads = 0
        self.blocked_since = None
        self.permission_errors = 0
        self.transition_retries = 0

    def read(self):
        try:
            raw = self.path.read_bytes()
        except PermissionError:
            self.permission_errors += 1
            if self.blocked_since is None:self.blocked_since = self.progress.clock()
            if self.progress.expired():raise
            return self.window
        if len(raw) > 16384:raise ValueError('GPU window metadata exceeded16KiB')
        window = json.loads(raw)
        if not isinstance(window, dict):raise ValueError('GPU window metadata must be an object')
        self.window = window
        self.blocked_since = None
        self.reads += 1
        self.progress.observe(self.reads)  # even an unchanged complete object is a successful read
        if isinstance(window.get('deadline'), (int, float)):
            self.progress.rising(window['deadline'])  # the publisher's advancing heartbeat
        return window


class WindowPublisherExited(Exception):
    """Normal owned exit raced a final cached request-window observation."""


def gpu_window_observation(reader, spec, child):
    """Read one consistent window/ledger pair without weakening lease checks.

    A producer retirement can publish a new window between the first metadata
    read and the ledger read. Re-read ownership after the ledger; retry only
    when that exact identity changed. A stable missing or foreign hold remains
    visible to the ordinary admission checks below.
    """
    from .lease import active_holds
    def validate(window):
        if not window or not window.get('active'):return
        owner=window.get('owner_pid');target=None
        try:
            candidate=psutil.Process(owner) if owner else None
            if candidate is not None and (owner==child.pid or candidate.ppid()==child.pid):target=candidate
        except psutil.Error:target=None
        if target is None and child.poll() is not None:raise WindowPublisherExited()
        if (window.get('run_id')!=spec['gpu_run_id'] or target is None or
                abs(window.get('owner_created',0)-target.create_time())>.01):
            raise ValueError('GPU window process/run identity differs')
    def identity(window):
        return tuple((window or {}).get(key)for key in ('active','run_id','owner_pid','owner_created','lease_id'))
    window=reader.read()
    for _ in range(4):
        validate(window)
        if not window or not window.get('active') or not window.get('lease_id'):return window,None
        holds=active_holds(Path(spec['gpu_hold_file']).read_text(encoding='utf-8-sig'))
        after=reader.read();validate(after)
        if identity(window)==identity(after):return after,holds
        reader.transition_retries+=1
        window=after
    raise ValueError('GPU request window ownership did not stabilize across four ledger observations')


def gpu_lease_observation(holds, identity, gpu, budget, floor, free, rss):
    """Keep both the foreign-reservation rule and the exact own-hold gate."""
    from .lease import reservation_admission
    own=lambda line:len(line.split())>=3 and line.split()[2].rstrip(';')==identity
    foreign=[line for line in holds if not own(line)]
    blockers,reservation=reservation_admission(foreign,gpu.get('host_budget_gib',budget+.25),floor,available=free,owned_rss=rss)
    if blockers:return 'preempted by foreign ACTIVE hold or RAM reservation: '+repr(blockers),reservation
    if not any(own(line)for line in holds):return 'own ACTIVE hold is no longer present',reservation
    return None,reservation


def admit(budget_gib, floor_gib):
    ram = psutil.virtual_memory()
    available=available_physical_bytes()
    if not 0 < budget_gib * GiB <= available - floor_gib * GiB:
        raise MemoryError(f'Host budget {budget_gib:.2f} GiB cannot fit: available={available / GiB:.2f}, floor={floor_gib:.2f}')
    return available


def kill_tree(proc, spec=None):
    """Legacy name: request cooperative exit and wait before the last force fallback."""
    owned = GracefulOwnedProcesses(proc, spec)
    proof = owned.stop()
    if not proof['confirmed']:
        error = RuntimeError('Owned process cleanup unconfirmed')
        error.owned_cleanup_uncertain = True
        raise error
    proc.wait(timeout=10)
    return proof


def windows_job(proc, commit_gib, *, kill_on_close=True):
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
    limits.BasicLimitInformation.LimitFlags = (0x2000 if kill_on_close else 0) | 0x200
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
    from .paging_limits import owned_paging_limit_bytes
    paging_limit = owned_paging_limit_bytes(spec)
    if paging_limit is not None and not spec.get('paging_telemetry'):
        raise ValueError('An explicit owned-paging limit requires paging telemetry')
    # Engine/GPU jobs use the pinned shared Studio guard. A server that opens its own
    # per-request GPU windows (combo_proxy) or a job bound to a named lease stays on
    # the owned loop below, which checks the window and ledger identity every poll.
    # Persistent servers retain the owned loop's RAM/paging checks. The pinned
    # shared adapter renews its finite no-progress window on tree/log activity.
    if spec.get('timeout_s') is not None and spec.get('paging_telemetry') and not (spec.get('gpu_window_file') or spec.get('gpu_lease_identity') or spec.get('cooperative_stop')):
        from .shared_guard import supervise as shared_supervise
        return shared_supervise({**spec,'timeout_s':min(spec['timeout_s'],1140)})
    if not (spec.get('paging_telemetry') or spec.get('gpu_window_file') or spec.get('gpu_lease_identity') or spec.get('cooperative_stop')):
        return _supervise_plain(spec)
    admit(spec['host_budget_gib'],spec['min_available_gib'])
    warnings=[]
    watcher=PagingWatch(paging_bytes,warning=warnings.append,owned_paging_limit=paging_limit)
    receipt=Path(spec['receipt'])
    try:
        return _supervise_owned(spec,watcher,warnings)
    finally:
        try:
            watcher.close()
        except BaseException as exc:
            if receipt.exists():
                record=json.loads(receipt.read_text())
                record.update(paging_telemetry_closed=False,reason='Paging telemetry cleanup failed: '+repr(exc))
                receipt.write_text(json.dumps(record,indent=2))
            raise
        else:
            if receipt.exists():
                record=json.loads(receipt.read_text())
                record['paging_telemetry_closed']=True
                receipt.write_text(json.dumps(record,indent=2))


def _supervise_plain(spec):
    """Release CPU/build loop: whole-tree RSS, deadline, stop file and shared holds.
    Retire dead build identities between polls so compiler churn stays bounded."""
    budget, floor = spec['host_budget_gib'], max(spec['min_available_gib'],spec.get('runtime_min_free_gib',0))
    admit(budget, spec['min_available_gib'])
    from .lease import HoldWatcher
    watch = spec.get('hold_watch')
    watcher = HoldWatcher(watch['path'], watch['identity'], mac=watch.get('mac', False),
                          task=watch.get('task', 'DS4STACK')) if watch else None
    if watcher and watcher.reason():
        raise RuntimeError('Yielding before process creation to shared hold: ' + watcher.reason())
    flags = subprocess.CREATE_NO_WINDOW | 4 if os.name == 'nt' else 0  # suspended until Job admission
    start = time.monotonic()
    peak, minimum, reason = 0, available_physical_bytes(), None
    from .memory_counters import snapshot, PagefilePressure
    paging_before = snapshot() if spec.get('paging_telemetry') else None
    pressure = PagefilePressure() if spec.get('paging_telemetry') else None
    next_pressure_check = start
    process_counters = {}
    progress, next_progress_check = Window(spec['timeout_s']), start
    with open(spec['log'], 'ab', buffering=0) as log:
        child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=spec['env'],
                                 stdin=subprocess.PIPE if 'stdin' in spec else subprocess.DEVNULL,
                                 stdout=log, stderr=log, creationflags=flags,
                                 start_new_session=os.name != 'nt')
        job = None
        owned = GracefulOwnedProcesses(child,spec)
        cleanup = None
        try:
            if os.name == 'nt':
                job = windows_job(child, spec['job_commit_gib'], kill_on_close=False)
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
                    # Keep live descendants across parent exit without retaining
                    # thousands of dead compiler identities in the owned set.
                    owned.identities={pid:identity for pid,identity in owned.identities.items()
                        if pid==child.pid or identity.process() is not None}
                    owned.retired_identities=[]
                    rss = 0
                    for member in owned.refresh():
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
                            row['peak_working_set_bytes'] = max(row.get('peak_working_set_bytes',0),info.rss)
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    rss = 0
                free = available_physical_bytes()
                peak, minimum = max(peak, rss), min(minimum, free)
                if rss > budget * GiB or free < floor * GiB:
                    reason = f'HOST MEMORY LIMIT: tree RSS={rss}, available={free}'
                if pressure and time.monotonic() >= next_pressure_check:
                    reason = reason or pressure.check()
                    next_pressure_check = time.monotonic() + .5
                if time.monotonic() >= next_progress_check:   # Runtime contract: timeout_s is a NO-PROGRESS window
                    progress.observe(activity(child.pid, spec['log']));next_progress_check = time.monotonic() + min(1, progress.seconds / 4)
                if progress.expired():
                    reason = 'deadline exceeded: no progress for %.1fs (log, I/O, CPU and members unchanged; %.0fs since start)' % (progress.idle(), progress.elapsed())
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
                    spec['stop_reason'] = reason
                    cleanup = owned.stop()
                    break
                time.sleep(.05)
            code = child.poll() if cleanup and not cleanup['confirmed'] else child.wait()
            if code is None:code=1
        finally:
            if cleanup is None:cleanup=owned.stop()
            if job:
                job[0].CloseHandle(job[1])
    receipt = {'exit_code': code, 'reason': reason, 'peak_tree_rss': peak,
               'min_available': minimum, 'elapsed_s': time.monotonic() - start,
               'host_budget_gib': budget, 'floor_gib': floor,
               'job_commit_gib': spec['job_commit_gib'], 'child_pid': child.pid,
               'cleanup':cleanup,'cleanup_confirmed':cleanup['confirmed'],'release_allowed':cleanup['confirmed']}
    receipt.update(paging_before=paging_before, paging_after=snapshot() if spec.get('paging_telemetry') else None,
                   system_pagefile_pressure=pressure.receipt() if pressure else None,
                   process_memory_counters=list(process_counters.values()),
                   paging_interpretation='Process faults include soft and hard faults. Private commit/pagefile usage is not disk I/O. System counters include other processes; these counters alone do not prove core/KV residency.')
    Path(spec['receipt']).write_text(json.dumps(receipt, indent=2))
    if not cleanup['confirmed']:
        error=RuntimeError('Owned cleanup unconfirmed; resource ownership retained')
        error.owned_cleanup_uncertain=True
        raise error
    return 0 if code == 0 and reason is None else 1


def _supervise_owned(spec,paging_watch,paging_warnings):
    if spec.get('cooperative_stop'):
        from .cooperative_stop import validate
        validate(spec['cooperative_stop'])
    budget, floor = spec['host_budget_gib'], max(spec['min_available_gib'],spec.get('runtime_min_free_gib',0))
    admit(budget, spec['min_available_gib'])
    from .lease import HoldWatcher
    watch = spec.get('hold_watch')
    watcher = HoldWatcher(watch['path'], watch['identity'], mac=watch.get('mac', False),
                          task=watch.get('task', 'DS4STACK')) if watch else None
    if watcher and watcher.reason():
        raise RuntimeError('Yielding before process creation to shared hold: ' + watcher.reason())
    flags = subprocess.CREATE_NO_WINDOW | 4 if os.name == 'nt' else 0  # suspended until Job admission
    start = time.monotonic()
    peak, minimum, reason = 0, available_physical_bytes(), None
    from .memory_counters import snapshot, PagefilePressure
    paging_before = snapshot() if spec.get('paging_telemetry') else None
    pressure = PagefilePressure() if spec.get('paging_telemetry') else None
    next_pressure_check = start
    process_counters = {}
    reservation_samples=[];last_reservation_sample=0
    last_paging_sample = 0
    ledger_blocked_since = None
    active_window = None
    window_started = None
    window_reader = GPUWindowReader(spec['gpu_window_file'], spec.get('gpu_window_stall_seconds', 30)) if spec.get('gpu_window_file') else None
    progress = None if spec['timeout_s'] is None else Window(spec['timeout_s'])
    next_progress_check = start
    with open(spec['log'], 'ab', buffering=0) as log:
        child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=spec['env'],
                                 stdin=subprocess.PIPE if 'stdin' in spec else subprocess.DEVNULL,
                                 stdout=log, stderr=log, creationflags=flags,
                                 start_new_session=os.name != 'nt')
        job = None
        if spec.get('cooperative_stop'):
            from .cooperative_stop import CooperativeOwnedProcesses
            owned=CooperativeOwnedProcesses(child,spec)
        else:owned = GracefulOwnedProcesses(child, spec)
        cleanup = None
        try:
            if os.name == 'nt':
                job = windows_job(child, spec['job_commit_gib'],kill_on_close=False)
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
                    members=owned.refresh()
                    rss = commit = 0
                    for member in members:
                        try:
                            info, created = member.memory_info(), member.create_time()
                            rss += info.rss
                            commit += getattr(info,'private',info.rss)
                            key = f'{member.pid}:{created:.6f}'
                            faults = getattr(info, 'num_page_faults', None)
                            row = process_counters.setdefault(key, {'pid': member.pid, 'created': created,
                                'name': member.name(), 'first_page_faults': faults, 'last_page_faults': faults,
                                'peak_private_commit_bytes': 0})
                            row['last_page_faults'] = faults
                            row['peak_private_commit_bytes'] = max(row['peak_private_commit_bytes'],
                                getattr(info, 'private', getattr(info, 'pagefile', 0)))
                            row['peak_working_set_bytes'] = max(row.get('peak_working_set_bytes',0),info.rss)
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    rss = commit = 0;members=[]
                free = available_physical_bytes()
                peak, minimum = max(peak, rss), min(minimum, free)
                if rss > budget * GiB or free < floor * GiB:
                    reason = f'HOST MEMORY LIMIT: tree RSS={rss}, available={free}'
                if pressure and time.monotonic() >= next_pressure_check:
                    reason = reason or pressure.check()
                    next_pressure_check = time.monotonic() + .5
                if progress is not None and time.monotonic() >= next_progress_check:   # Runtime contract: timeout_s is a NO-PROGRESS window
                    progress.observe(activity(child.pid, spec['log']));next_progress_check = time.monotonic() + min(1, progress.seconds / 4)
                if progress is not None and progress.expired():
                    reason = 'deadline exceeded: no progress for %.1fs (log, I/O, CPU and members unchanged; %.0fs since start)' % (progress.idle(), progress.elapsed())
                if Path(spec['stop']).exists():
                    reason = 'stop requested'
                if watcher:
                    try:
                        foreign = watcher.reason()
                        if foreign:
                            reason = 'Yielding to shared hold: ' + foreign
                    except OSError as exc:
                        reason = 'Shared hold observation failed: ' + str(exc)
                identity=spec.get('gpu_lease_identity');window_holds=None
                hold_file=(spec.get('gpu') or {}).get('hold_file')
                if window_reader is not None:
                    try:
                        window,window_holds=gpu_window_observation(window_reader,spec,child)
                        if window is not None and window.get('active'):
                            identity=window['lease_id'];hold_file=spec['gpu_hold_file']
                            if active_window!=identity or window_started is None:active_window=identity;window_started=time.monotonic()   # lease_id is null until the producer's lease is taken
                            # During a read denial the old heartbeat cannot prove a stall.
                            # The reader's independent no-progress window bounds that wait;
                            # fresh metadata still enforces the publisher's own deadline.
                            if window_reader.blocked_since is None and time.time()>=window['deadline']:
                                reason='GPU request window made no progress before its renewed deadline'
                        elif window is not None:active_window=None;window_started=None
                    except WindowPublisherExited:
                        break
                    except FileNotFoundError:
                        if active_window:reason='Active GPU window metadata disappeared'
                    except (ValueError,KeyError,TypeError,OSError) as exc:
                        reason='GPU request window metadata refused: '+str(exc)
                if identity:
                    from .lease import active_holds
                    try:
                        holds = window_holds if window_holds is not None else active_holds(Path(hold_file).read_text(encoding='utf-8-sig'))
                        ledger_blocked_since = None
                        gpu=spec.get('gpu') or {}
                        lease_error,reservation=gpu_lease_observation(holds,identity,gpu,budget,floor,free,rss)
                        if time.monotonic()-last_reservation_sample>=50:
                            reservation_samples.append(reservation);last_reservation_sample=time.monotonic()
                        if lease_error:reason=lease_error
                    except OSError as exc:
                        if ledger_blocked_since is None:ledger_blocked_since=time.monotonic()
                        if time.monotonic()-ledger_blocked_since>=1:
                            reason = 'GPU hold ledger unavailable for1second: ' + str(exc)
                if members and spec.get('paging_telemetry'):
                    failure=paging_watch.check(OwnedView(members),rss,commit,rss_limit=budget*GiB,
                        commit_limit=spec['job_commit_gib']*GiB,floor=floor*GiB)
                    if failure:reason=reason or failure
                if reason:
                    spec['stop_reason'] = reason
                    cleanup=owned.stop()
                    break
                time.sleep(.05)
            code = child.poll() if cleanup and not cleanup['confirmed'] else child.wait()
            if code is None:code=1
        except BaseException as exc:
            reason=repr(exc);cleanup=owned.stop();code=child.poll()
            if code is None:code=1
        finally:
            if child.poll() is None and cleanup is None:
                cleanup=owned.stop();child.wait(timeout=10)
            if job:
                job[0].CloseHandle(job[1])
            if cleanup is None:cleanup=owned.stop()
    receipt = {'exit_code': code, 'reason': reason, 'peak_tree_rss': peak,
               'min_available': minimum, 'elapsed_s': time.monotonic() - start,
               'started_monotonic': start, 'no_progress_window_s': spec['timeout_s'],
               'gpu_window_permission_errors': window_reader.permission_errors if window_reader else 0,
               'gpu_window_transition_retries': window_reader.transition_retries if window_reader else 0,
               'gpu_last_observed_window': {key:(window_reader.window or {}).get(key)for key in ('active','run_id','owner_pid','owner_created','lease_id','deadline')} if window_reader else None,
               'gpu_window_stall_seconds': window_reader.progress.seconds if window_reader else None,
               'host_budget_gib': budget, 'floor_gib': floor,
               'job_commit_gib': spec['job_commit_gib'], 'child_pid': child.pid}
    receipt.update(paging_before=paging_before, paging_after=snapshot() if spec.get('paging_telemetry') else None,
                   system_pagefile_pressure=pressure.receipt() if pressure else None,
                   process_memory_counters=list(process_counters.values()),
                   paging_warnings=paging_warnings,
                   global_paging_growth=paging_watch.max_global_growth,owned_paged_bytes=paging_watch.job_paged,
                   paged_processes=paging_watch.observer.last_processes,
                   paging_scanned_pages=paging_watch.observer.scanned_pages,
                   paging_completed_scans=paging_watch.observer.completed_scans,
                   paging_attribution=paging_watch.observer.attribution,paging_policy=paging_watch.details(),cleanup=cleanup,
                   release_allowed=cleanup['confirmed'],cleanup_confirmed=cleanup['confirmed'],reservation_samples=reservation_samples,
                   paging_interpretation='Process faults include soft and hard faults. Private commit/pagefile usage is not disk I/O. System counters include other processes; these counters alone do not prove core/KV residency.')
    Path(spec['receipt']).write_text(json.dumps(receipt, indent=2))
    if not cleanup['confirmed']:
        error=RuntimeError('Owned cleanup unconfirmed; GPU hold retained')
        error.owned_cleanup_uncertain=True
        raise error
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
                        # Admission creates the default ledger when needed.
                        # Resolve that same file before every child-side check;
                        # an empty value must never become Path('') == '.'.
                        spec['gpu']['hold_file']=str(Path(spec['gpu'].get('hold_file') or
                            Path(spec['runtime_root'])/'state'/'gpu.holds').resolve())
                        spec['hold_watch'] = {'path': spec['gpu'].get('hold_file') or
                            str(Path(spec['runtime_root']) / 'state' / 'gpu.holds'), 'identity': identity,
                            'task': spec['gpu'].get('task', 'DS4STACK')}
                        spec['gpu_lease_identity'] = identity
                        spec['timeout_s'] = spec['gpu']['lease_minutes'] * 60 if spec['timeout_s'] is None else min(spec['timeout_s'], spec['gpu']['lease_minutes'] * 60)
                        return _supervise(spec)
                except (AdmissionRefused, PermissionError):
                    # Retry only before owning a lease/starting a child. Engine or cleanup
                    # failures must never cause an automatic second generation.
                    if entered or not waiting or time.monotonic() >= deadline:
                        raise
                    if (Path(spec['stop']).exists() or not owner.is_running() or
                        abs(owner.create_time() - spec['owner_created']) > .01):
                        raise RuntimeError('GPU admission wait stopped by its owner')
                    if available_physical_bytes() < spec['min_available_gib'] * GiB:
                        raise MemoryError('Available physical RAM fell below the admission-wait floor')
                    time.sleep(.5)
        return _supervise(spec)
    finally:
        # Legacy kill-on-close orphans can be recovered. Cooperative lease
        # records retain ownership because guardian death cannot prove cleanup.
        if not getattr(sys.exc_info()[1], 'owned_cleanup_uncertain', False):
            recover_leases(spec['runtime_root'], (spec.get('gpu') or {}).get('task', spec.get('task', 'DS4STACK')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    args = parser.parse_args()
    from .envvars import restore_secrets
    sys.exit(supervise(restore_secrets(json.loads(Path(args.spec).read_text()))))
