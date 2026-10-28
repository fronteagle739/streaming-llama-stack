"""Cooperative retirement with a renewable silence window before force is allowed."""
import dataclasses
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil
from .progress import Window
from .native_broker import vendor

vendor()
from studio.job_guard import OwnedProcesses

log = logging.getLogger(__name__)


def gpu_free_reader(identity):
    """Read only the configured GPU, at most once a second, without a CUDA context."""
    if not isinstance(identity,str) or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',identity):return None
    next_sample=0.;last=0;warned=False
    def read():
        nonlocal next_sample,last,warned
        if time.monotonic()<next_sample:return last
        try:
            result=subprocess.run(['nvidia-smi','--id='+identity,'--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=2,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            last=int(float(result.stdout.strip())*(1<<20))
        except (OSError,ValueError,subprocess.SubprocessError):
            if not warned:
                log.warning('Shutdown GPU telemetry unavailable; tracking owned logs and RSS')
                warned=True
        next_sample=time.monotonic()+1
        return last
    return read


class RetirementProgress:
    """Only new diagnostics, retiring members and falling RSS renew shutdown.

    CPU use, polling, HTTP retries and growing allocations are not retirement.
    A caller with device telemetry can also supply increasing free GPU bytes.
    """
    def __init__(self, seconds=60, *, clock=None, diagnostic=None, gpu_free=None):
        self.window = Window(seconds, clock=clock)
        self.diagnostic, self.gpu_free = diagnostic, gpu_free
        self.best_log = self.best_gpu = 0
        self.least_rss = self.least_members = None

    def observe(self, members):
        rss = 0
        for process in members:
            try:rss += process.memory_info().rss
            except psutil.NoSuchProcess:pass
        if self.diagnostic is not None:
            try:
                size = self.diagnostic() if callable(self.diagnostic) else Path(self.diagnostic).stat().st_size
                self.best_log = max(self.best_log, size)
            except OSError:pass
        if self.gpu_free is not None:
            self.best_gpu = max(self.best_gpu, self.gpu_free())
        self.least_rss = rss if self.least_rss is None else min(rss, self.least_rss)
        self.least_members = len(members) if self.least_members is None else min(len(members), self.least_members)
        self.window.observe((self.best_log, self.best_gpu, self.least_rss, self.least_members))


def process_members(process):
    pid = getattr(process, 'pid', None)
    if not isinstance(pid, int):return []
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:return []


def wait_for_exit(process, seconds=60, *, diagnostic=None, gpu_free=None, clock=None, sleep=None):
    """Wait indefinitely while shutdown advances, then allow a full silence window."""
    progress = RetirementProgress(seconds, clock=clock, diagnostic=diagnostic,
        gpu_free=gpu_free or getattr(process,'gpu_free',None))
    sleep = sleep or time.sleep
    while (code := process.poll()) is None:
        waiter=getattr(process,'wait',None)
        if callable(waiter):
            try:return waiter(timeout=.05)
            except subprocess.TimeoutExpired:pass
        progress.observe(process_members(process))
        if progress.window.expired():
            raise subprocess.TimeoutExpired(getattr(process, 'args', 'owned process'), seconds)
        if not callable(waiter):sleep(.05)
    return code


def wait_for_thread(thread, process, seconds=240, *, diagnostic=None,gpu_free=None):
    """A retirement worker gets the same progress policy as its owned process."""
    progress=RetirementProgress(seconds,diagnostic=diagnostic,gpu_free=gpu_free)
    while thread.is_alive():
        progress.observe(process_members(process) if process is not None else [])
        if progress.window.expired():
            raise RuntimeError('Owned retirement worker made no progress for %.1fs' % progress.window.idle())
        thread.join(.05)


def request_interrupt(process):
    """Ctrl-Break on Windows; SIGINT on POSIX (Python runs its finally blocks)."""
    if os.name == 'nt':
        from studio.dsv41_windows_runner import _CTRL_BREAK
        result = subprocess.run([sys.executable, '-I', '-c', _CTRL_BREAK, str(process.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x8, timeout=15)
        if result.returncode:raise OSError('Owned Ctrl-Break request failed: exit %d' % result.returncode)
    else:process.send_signal(signal.SIGINT)


class ProgressOwnedProcesses(OwnedProcesses):
    def __init__(self, child, spec=None):
        super().__init__(child)
        self.spec = spec or {}

    @classmethod
    def from_owned(cls, owned, spec):
        result = cls.__new__(cls)
        result.__dict__.update(owned.__dict__)
        result.spec = spec
        return result

    def request(self, members, done, errors):
        # Let a controller drain its children before signalling those children.
        for process in members[:1]:
            try:request_interrupt(process)
            except psutil.NoSuchProcess:pass
            except Exception as error:errors.append(type(error).__name__)

    def stop(self, grace=60, timeout=10):
        errors, request_errors = [], []
        import threading
        done = threading.Event()
        forced = False; force_reason = None
        progress = RetirementProgress(self.spec.get('shutdown_stall_seconds', grace),
            diagnostic=self.spec.get('log'),gpu_free=gpu_free_reader(
                self.spec.get('gpu_uuid') or (self.spec.get('gpu') or {}).get('gpu_uuid')))
        try:
            members = self.refresh()
            requested_root = members[0].pid if members else None
            if members:self.request(members, done, request_errors)
            while (members := self.refresh()):
                if not self.spec.get('cooperative_stop') and requested_root not in {p.pid for p in members}:
                    requested_root=members[0].pid
                    self.request(members,done,request_errors)
                progress.observe(members)
                if progress.window.expired():
                    force_reason = ('shutdown made no progress for %.1fs after cooperative request '
                        '(log, RSS and owned members unchanged); %s' %
                        (progress.window.idle(), self.spec.get('stop_reason', 'owned retirement requested')))
                    log.critical('FORCE KILL LAST FALLBACK: %s; owned PIDs=%s', force_reason,
                                 [p.pid for p in members])
                    forced = True
                    for process in reversed(members):
                        try:process.kill()
                        except psutil.NoSuchProcess:pass
                    deadline = time.monotonic() + timeout
                    while self.refresh() and time.monotonic() < deadline:time.sleep(.05)
                    break
                time.sleep(.05)
        except Exception as error:errors.append(repr(error))
        finally:done.set()
        try:survivors = [p.pid for p in self.live()]
        except Exception as error:
            errors.append(repr(error));survivors = list(self.identities)
        return dict(confirmed=not survivors and not errors, surviving_owned_pids=survivors,
            errors=errors, identities=[dataclasses.asdict(x) for x in self.identities.values()],
            retired_identities=[dataclasses.asdict(x) for x in self.retired_identities],
            forced_kill=forced, force_reason=force_reason, cooperative=True,
            cooperative_request_errors=request_errors, shutdown_elapsed_s=progress.window.elapsed(),
            shutdown_idle_s=progress.window.idle())
