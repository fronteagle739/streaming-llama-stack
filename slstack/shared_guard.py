"""Stack adapter around the pinned Studio ownership and paging implementation."""
import json
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
import psutil
from .journal import atomic_json
from .lease import HoldWatcher
from .native_broker import vendor
from .config import ROOT
from .progress import Window, activity

vendor()
from studio.job_guard import GuardedJob
from studio.job_paging import GLOBAL_PAGING_LIMIT

GiB = 2**30


class CleanupUnconfirmed(RuntimeError):
    owned_cleanup_uncertain = True


class _PagingCloseTracker:
    """Delegate all paging behavior; record completion of the actual close call.

    Absence/entry/failure are False. True requires normal return from the pinned
    watcher's close, with no failed close attempt. A failure remains sticky, as
    does the provider's failure/lease policy; retry cannot erase that evidence.
    """
    __slots__ = ('_watcher', '_record', '_close_attempts', '_close_error', '_close_lock')

    def __init__(self, watcher, record):
        self._watcher = watcher
        self._record = record
        self._close_attempts = 0
        self._close_error = None
        self._close_lock = threading.RLock()
        record.update(paging_telemetry_closed=False,
                      paging_telemetry_close_attempts=0,
                      paging_telemetry_close_error=None)

    def __getattr__(self, name):
        return getattr(self._watcher, name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._watcher, name, value)

    def __delattr__(self, name):
        if name in self.__slots__:
            object.__delattr__(self, name)
        else:
            delattr(self._watcher, name)

    def close(self):
        with self._close_lock:
            self._close_attempts += 1
            self._record.update(paging_telemetry_closed=False,
                                paging_telemetry_close_attempts=self._close_attempts)
            try:
                result = self._watcher.close()
            except BaseException as error:
                self._close_error = repr(error)
                self._record.update(paging_telemetry_close_error=self._close_error,
                                    release_allowed=False)
                raise
            closed = self._close_error is None
            self._record['paging_telemetry_closed'] = closed
            if not closed:
                self._record['release_allowed'] = False
            return result


def _paging_telemetry_closed(job):
    # A canonical bool: no "not failed"/missing-key/cleanup inference can turn
    # an unobserved close into positive telemetry-closure evidence.
    return job.record.get('paging_telemetry_closed') is True


def _watch_live_ownership(job, spec):
    """Preserve the provider monitor while bounding live, rather than past, children."""
    from .shutdown import ProgressOwnedProcesses
    job.owned = ProgressOwnedProcesses.from_owned(job.owned, spec)
    return GuardedJob._watch(job)


def _wait_for_command_receipt(job):
    """Wait without mistaking a receipt published during poll() for its loss."""
    if job.child is None:
        raise RuntimeError('job has not started')
    try:
        while not job.command_status.exists() and not job.cancelled.is_set():
            if job.child.poll() is not None:
                # The bootstrap writes the receipt before it exits. The first
                # exists() can precede that write while poll() follows exit.
                if not job.command_status.exists():
                    job.failure = job.failure or 'owned bootstrap exited before command receipt'
                break
            job.finished.wait(job.poll)
        code = int(job.command_status.read_text()) if job.command_status.exists() else 137
        job.record['command_exit_code'] = code
    except BaseException as error:
        job.failure = job.failure or repr(error)
        job.cancelled.set()
        raise
    finally:
        job.close()
    if job.failure:
        raise RuntimeError(job.failure)
    return code


def supervise(spec):
    from .paging_limits import owned_paging_limit_bytes
    paging_limit = owned_paging_limit_bytes(spec)
    if paging_limit is not None and not spec.get('paging_telemetry'):
        raise ValueError('An explicit owned-paging limit requires paging telemetry')
    for name, pin in {'job_guard.py':'f323a66355e32ef7738c7f905be9ca7b39d2ce3963856de6f7d7c0ea80372068',
                      # The bounded diagnostics retain every
                      # original enforcement statement; pin that actual body.
                      'job_paging.py':'72e613722e3dcb1fbb8ebf8cd948b8ddca3b063606ab494a473808d0b11a31ef'}.items():
        source = (ROOT/'vendor/studio/studio'/name).read_bytes().split(b'\n',2)[2].replace(b'\r\n',b'\n')
        if hashlib.sha256(source).hexdigest() != pin:
            raise RuntimeError('Pinned shared guard source changed: ' + name)
    watch = spec.get('hold_watch')
    holds = HoldWatcher(watch['path'],watch['identity'],mac=watch.get('mac',False),task=watch.get('task','DS4STACK')) if watch else None
    parent = psutil.Process(spec['owner_pid'])
    if abs(parent.create_time()-spec['owner_created']) > .01:
        raise RuntimeError('Owning process identity changed before launch')
    class Adapter(GuardedJob):
        wait = _wait_for_command_receipt

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.watcher=_PagingCloseTracker(self.watcher,self.record)
            self.retirement_lock=threading.RLock()
            self.retirement_proof=None

        def _cooperative_ownership(self):
            # Keep memory limits but prevent handle-close/pipe-loss from
            # bypassing cooperative retirement on either Windows or POSIX.
            if self.kernel_job:
                kernel=self.kernel_job
                limits=kernel.Limits()
                limits.basic.flags=0x200 | 0x100
                limits.process=limits.job=int(self.commit_limit)
                if not kernel.api.SetInformationJobObject(kernel.handle,9,
                        kernel.c.byref(limits),kernel.c.sizeof(limits)):
                    raise kernel.c.WinError(kernel.c.get_last_error())
            if self.child and self.child.stdin and not self.child.stdin.closed:
                self.child.stdin.write(b'H')
                self.child.stdin.flush()
                self.child.stdin.close()

        def start(self,*args,**kwargs):
            result=super().start(*args,**kwargs)
            self._cooperative_ownership()
            return result

        def _watch(self):
            # GuardedJob starts this thread only after establishing ownership.
            # Upgrade before its first refresh; pinned paging/priority/limits remain intact.
            return _watch_live_ownership(self, spec)

        def _stop(self):
            from .shutdown import ProgressOwnedProcesses
            with self.retirement_lock:
                if self.retirement_proof is not None:return self.retirement_proof
                # Preserve the pinned provider's identity ledger and paging checks.
                self._cooperative_ownership()
                owned=ProgressOwnedProcesses.from_owned(self.owned,
                    {**spec,'stop_reason':self.failure or 'shared guarded job retirement'})
                cleanup=owned.stop()
                if cleanup['forced_kill'] and self.kernel_job:
                    # Catch descendants born between ownership polls only after grace.
                    self.kernel_job.stop()
                self.retirement_proof=cleanup
                return cleanup

        def close(self):
            from .shutdown import wait_for_thread,gpu_free_reader
            self.finished.set()
            if self.monitor and self.monitor is not threading.current_thread():
                try:wait_for_thread(self.monitor,self.child,diagnostic=spec['log'],
                    gpu_free=gpu_free_reader(spec.get('gpu_uuid') or (spec.get('gpu') or {}).get('gpu_uuid')))
                except RuntimeError as error:raise CleanupUnconfirmed(str(error)) from error
            return super().close()

        def _priority(self):
            super()._priority()
            if Path(spec['stop']).exists():
                raise RuntimeError('stop requested')
            if not parent.is_running() or abs(parent.create_time()-spec['owner_created']) > .01:
                raise RuntimeError('owning process exited')
            if holds and (reason := holds.reason()):
                raise RuntimeError('Yielding to shared hold: ' + reason)
            if spec.get('runtime_min_free_gib',0):
                from .memory_counters import available_physical_bytes
                if available_physical_bytes()<spec['runtime_min_free_gib']*GiB:
                    raise MemoryError('Runtime physical RAM floor crossed')
            if self.child is not None:
                progress.observe(activity(self.child.pid, spec['log']))
                if progress.expired():
                    raise TimeoutError('Owned operation made no progress within its stall window')
                self.deadline = progress.last + progress.seconds
    receipt = Path(spec['receipt'])
    certificate = receipt.with_suffix('.shared.json')
    host = spec['host_budget_gib'] * GiB
    commitment = spec['job_commit_gib'] * GiB if os.name == 'nt' else host
    progress = Window(spec['timeout_s'])
    job = Adapter(rss_limit=host,commit_limit=host,floor=spec['min_available_gib']*GiB,
                  owned_paging_limit=paging_limit,
                  device_commit_allowance=max(0,commitment-host) if os.name=='nt' else 0,
                  seconds=spec['timeout_s'],receipt=certificate)
    started = time.monotonic()
    code, reason = 1, None
    try:
        with open(spec['log'],'ab',buffering=0) as log:
            job.start([sys.executable,'-B','-m','slstack.guard_child',spec['_spec_path']],
                      cwd=spec['cwd'],env=spec['env'],stdout=log)
            code = job.wait()
    except BaseException as error:
        reason = str(error)
    finally:
        if 'ended' not in job.record:
            job.close()
        certain = job.child is None or bool(job.record.get('release_allowed'))
        record = {'exit_code':code,'reason':reason,'peak_tree_rss':job.record['peak_rss'],
                  'min_available':job.record['minimum_available'],'elapsed_s':time.monotonic()-started,
                  'host_budget_gib':spec['host_budget_gib'],'floor_gib':spec['min_available_gib'],
                  'runtime_min_free_gib':spec.get('runtime_min_free_gib',0),
                  'no_progress_window_s':spec['timeout_s'],
                  'job_commit_gib':commitment/GiB,
                  'child_pid':job.child.pid if job.child else None,
                  'shared_guard_commit':'2774636281ab32bf871497f0a8caae23c6b6cdc7',
                  'stack_retirement_policy':'cooperative request, progress-aware grace, logged force fallback',
                  'shared_guard_certificate':str(certificate),'cleanup_confirmed':certain,
                  'paging_telemetry_closed':_paging_telemetry_closed(job),
                  'paging_policy':job.watcher.details(),
                  'owned_paging':{'attribution':job.watcher.observer.attribution,
                    'observed_paged_bytes':job.watcher.job_paged,
                    'observed_processes':job.watcher.observer.last_processes,
                    'scanned_pages':job.watcher.observer.scanned_pages,
                    'completed_scans':job.watcher.observer.completed_scans},
                  'system_pagefile_pressure':{'maximum_growth_bytes':job.watcher.max_global_growth,
                    'kill_above_bytes':GLOBAL_PAGING_LIMIT,'scope':'global; not process attribution'},
                  'paging_interpretation':'Owned observation is bounded and incremental. Zero observed pages is not an exhaustive zero-I/O proof. Darwin disk-only process attribution is unavailable.'}
        atomic_json(receipt,record)
        if not certain:
            identities = {(item['pid'],item['created']) for item in job.record.get('cleanup',{}).get('identities',[])}
            for file in (Path(spec['runtime_root'])/'state/leases').glob('*.json'):
                row = json.loads(file.read_text())
                if (row.get('guardian_pid'),row.get('guardian_created')) in identities and not row.get('released'):
                    atomic_json(file,{**row,'cleanup_uncertain':True,'cleanup_certificate':str(certificate)})
            raise CleanupUnconfirmed('Owned exit unconfirmed; resource lease must remain held: ' + str(certificate))
    return 0 if code==0 and reason is None else 1
