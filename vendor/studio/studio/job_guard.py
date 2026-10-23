# Derived from Studio backend/studio/job_guard.py at d83aa32382f83027631e7954951560c520f9cf6b.
# Shared implementation retained unchanged below this provenance header.
"""Ownership and cancellation for the shared TRAINFIX paging guard.

All blocking work runs independently of the monitor. Only processes launched
through this guard (and their verified descendants) are eligible for teardown.
The receipt certifies exit before a caller may release its resource lease.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import psutil

from .job_paging import PagingWatch, charged_memory, paging_bytes


@dataclasses.dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    created: float
    executable: str

    @classmethod
    def read(cls, process):
        return cls(process.pid, process.create_time(), os.path.normcase(process.exe()))

    def process(self):
        try:
            process = psutil.Process(self.pid)
            if self == self.read(process) and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                return process
        except psutil.NoSuchProcess:
            pass
        return None


class OwnedProcesses:
    def __init__(self, child):
        # A Popen launch handle, never a name/snapshot difference, grants ownership.
        if not isinstance(child, subprocess.Popen):
            raise TypeError('ownership requires the Popen handle returned by our launch')
        root = psutil.Process(child.pid)
        self.identities = {child.pid: ProcessIdentity.read(root)}

    def refresh(self):
        pending = list(self.identities.values())
        seen = set()
        while pending:
            identity = pending.pop()
            if identity in seen:
                continue
            seen.add(identity)
            parent = identity.process()
            if parent is None:
                continue
            try:
                for process in parent.children():
                    candidate = ProcessIdentity.read(process)
                    if process.ppid() != parent.pid or identity.process() is None:
                        continue
                    previous = self.identities.get(candidate.pid)
                    if previous is not None and previous != candidate:
                        raise RuntimeError('owned PID was reused; refusing to adopt its replacement')
                    self.identities[candidate.pid] = candidate
                    pending.append(candidate)
                    if len(self.identities) > 256:
                        raise RuntimeError('owned process bound exceeded')
            except psutil.NoSuchProcess:
                continue
        return self.live()

    def live(self):
        return [process for identity in self.identities.values()
                if (process := identity.process()) is not None]

    def stop(self, grace=1, timeout=3):
        errors = []
        def live():
            result=[]
            for identity in self.identities.values():
                try:
                    if (process := identity.process()) is not None:
                        result.append(process)
                except Exception as error:
                    errors.append(repr(error))
            return result
        try:
            self.refresh()
        except Exception as error:
            errors.append('final ownership refresh: ' + repr(error))
        for identity in reversed(list(self.identities.values())):
            try:
                if (process := identity.process()) is not None:
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                errors.append(repr(error))
        deadline = time.monotonic() + grace
        while live() and time.monotonic() < deadline:
            time.sleep(.02)
        for identity in reversed(list(self.identities.values())):
            try:
                if (process := identity.process()) is not None:
                    process.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                errors.append(repr(error))
        deadline = time.monotonic() + timeout
        while live() and time.monotonic() < deadline:
            time.sleep(.02)
        survivors = [process.pid for process in live()]
        return {'confirmed': not survivors and not errors, 'surviving_owned_pids': survivors,
                'errors': list(dict.fromkeys(errors)), 'identities': [dataclasses.asdict(x) for x in self.identities.values()]}


def priority_holds(text, owners, *, resource='RTX', own_lease=''):
    """Append ledger; unreleased manual holds never expire by inference."""
    active = {}
    for line in text.splitlines():
        match = re.match(r'^(?:(MAC|RTX)\s+)?(ACTIVE|RELEASED|ENDED|CANCELLED|CLEARED):\s+(\S+)\s*(.*)', line.strip())
        if not match:
            continue
        machine, state, owner, remainder = match.groups()
        machine = machine or 'RTX'
        if machine != resource or (owner not in owners and '*' not in owners):
            continue
        first = remainder.split(';', 1)[0].split()
        lease = first[0] if first and re.fullmatch(r'(?:[\w-]*[a-f0-9]{32}|[a-f0-9-]{36})', first[0]) else ''
        key = machine, owner, lease
        if state == 'ACTIVE':
            active[key] = line
        elif lease:
            active.pop(key, None)
        else:
            active = {key: value for key, value in active.items() if key[:2] != (machine, owner)}
    return [line for line in active.values() if not own_lease or own_lease not in line]


class OwnedView:
    """PagingWatch sees only the already identity-verified owned set."""
    def __init__(self, processes):
        self.root, *self.rest = processes
    def __getattr__(self, name):
        return getattr(self.root, name)
    def children(self, recursive=True):
        return self.rest


class WindowsJob:
    """Kernel ownership contains short-lived descendants between monitor polls."""
    def __init__(self, limit):
        import ctypes as c
        from ctypes import wintypes as w
        self.c = c
        self.api = c.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.api.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        class Basic(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64), ('flags', w.DWORD),
                ('min_ws', c.c_size_t), ('max_ws', c.c_size_t), ('active', w.DWORD),
                ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class Limits(c.Structure):
            _fields_ = [('basic', Basic), ('io', c.c_uint64*6), ('process', c.c_size_t),
                ('job', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Limits()
        limits.basic.flags = 0x2000 | 0x200 | 0x100
        limits.process = limits.job = int(limit)
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            self.api.CloseHandle(self.handle)
            raise c.WinError(c.get_last_error())

    def attach(self, child):
        if not self.api.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise self.c.WinError(self.c.get_last_error())

    def stop(self):
        if not self.api.TerminateJobObject(self.handle, 137):
            raise self.c.WinError(self.c.get_last_error())
        class Accounting(self.c.Structure):
            _fields_ = [('times', self.c.c_int64*4), ('faults', self.c.c_uint32),
                ('total', self.c.c_uint32), ('active', self.c.c_uint32), ('terminated', self.c.c_uint32)]
        deadline = time.monotonic()+3
        while True:
            info = Accounting()
            if not self.api.QueryInformationJobObject(self.handle, 1, self.c.byref(info), self.c.sizeof(info), None):
                raise self.c.WinError(self.c.get_last_error())
            if info.active == 0:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('owned kernel job exit unconfirmed')
            time.sleep(.02)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class GuardedJob:
    def __init__(self, *, rss_limit, commit_limit, floor, seconds, receipt,
                 ledger=None, priority_owners=(), resource='RTX', own_lease='',
                 poll=.1, watcher=None, device_commit_allowance=0):
        values = (rss_limit, commit_limit, floor, seconds, poll)
        if any(not math.isfinite(x) or x <= 0 for x in values) or seconds > 1140 or poll > 1:
            raise ValueError('positive finite limits required; lease <=1140s; poll <=1s')
        if not math.isfinite(device_commit_allowance) or device_commit_allowance<0:
            raise ValueError('device backing allowance must be finite and nonnegative')
        if device_commit_allowance and os.name!='nt':
            raise ValueError('separate device private-commit backing is Windows-only')
        self.host_commit_limit=commit_limit
        self.device_commit_allowance=device_commit_allowance
        self.rss_limit, self.commit_limit, self.floor = rss_limit, commit_limit+device_commit_allowance, floor
        self.seconds, self.poll = seconds, poll
        self.receipt = Path(receipt)
        if self.receipt.exists():
            raise FileExistsError('guard receipts are immutable: ' + str(self.receipt))
        self.ledger = Path(ledger) if ledger else None
        if priority_owners and self.ledger is None:
            raise ValueError('priority owners require an explicit ledger')
        self.priority_owners, self.resource, self.own_lease = priority_owners, resource, own_lease
        self.watcher = watcher or PagingWatch(paging_bytes)
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.child = self.owned = self.monitor = self.kernel_job = None
        self.group_stopped = False
        self.command_status = self.receipt.with_suffix('.command-status.json')
        if self.command_status.exists():
            raise FileExistsError('command status already exists: '+str(self.command_status))
        self.failure = None
        self.record = {'rss_limit': rss_limit, 'host_commit_limit': commit_limit,
                       'device_commit_allowance': device_commit_allowance,
                       'commit_limit': self.commit_limit, 'physical_floor': floor,
                       'deadline_seconds': seconds, 'poll_seconds': poll,
                       'paging_attribution': self.watcher.observer.attribution,
                       'peak_rss': 0, 'peak_commit': 0, 'minimum_available': psutil.virtual_memory().available}

    def _priority(self):
        if self.ledger:
            holds = priority_holds(self.ledger.read_text(encoding='utf-8-sig'), self.priority_owners,
                                   resource=self.resource, own_lease=self.own_lease)
            if holds:
                raise RuntimeError('YIELD TO HIGHER PRIORITY: ' + '\n'.join(holds))

    def start(self, command, *, cwd=None, env=None, stdout=None, new_session=True):
        if self.child is not None:
            raise RuntimeError('a guard owns exactly one launch')
        self._priority()
        vm = psutil.virtual_memory()
        if max(self.rss_limit, self.host_commit_limit) > vm.available - self.floor:
            raise RuntimeError('declared job budget exceeds available physical RAM minus floor')
        self.record.update(started=time.time(), command=list(command), physical_total=vm.total)
        self.owns_session=os.name!='nt' and new_session
        bootstrap = ('import json,os,pathlib,subprocess,sys; '
                     'command=json.loads(sys.argv[1]); '
                     'gate=sys.stdin.buffer.read(1); '
                     'sys.exit(125) if gate!=b"G" else None; '
                     'rc=subprocess.call(command,stdin=subprocess.DEVNULL); '
                     'pathlib.Path(sys.argv[2]).write_text(str(rc)); '
                     'sys.stdin.buffer.read(); sys.exit(rc)')
        self.child = subprocess.Popen([sys.executable, '-B', '-c', bootstrap, json.dumps(list(command)),str(self.command_status)],
            cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            start_new_session=self.owns_session)
        try:
            self.owned = OwnedProcesses(self.child)
            if os.name == 'nt':
                self.kernel_job = WindowsJob(self.commit_limit)
                self.kernel_job.attach(self.child)
            self.monitor = threading.Thread(target=self._watch, name='owned-job-memory-priority', daemon=True)
            self.monitor.start()
            self.child.stdin.write(b'G')
            self.child.stdin.flush()
        except BaseException:
            self.close()
            raise
        return self

    def _watch(self):
        try:
            deadline = time.monotonic() + self.seconds
            while not self.finished.is_set():
                self._priority()  # independent of chat/switch/network reads
                if time.monotonic() >= deadline:
                    raise TimeoutError('owned lease deadline exceeded')
                processes = self.owned.refresh()
                if not processes:
                    return
                rss = commit = 0
                for process in processes:
                    try:
                        memory = process.memory_info()
                        rss += memory.rss
                        commit += charged_memory(process)
                    except psutil.NoSuchProcess:
                        continue
                available = psutil.virtual_memory().available
                self.record['peak_rss'] = max(self.record['peak_rss'], rss)
                self.record['peak_commit'] = max(self.record['peak_commit'], commit)
                self.record['minimum_available'] = min(self.record['minimum_available'], available)
                reason = self.watcher.check(OwnedView(processes), rss, commit,
                    rss_limit=self.rss_limit, commit_limit=self.commit_limit, floor=self.floor)
                if reason:
                    raise RuntimeError(reason)
                self.finished.wait(self.poll)
        except BaseException as error:
            self.failure = repr(error)
            self.cancelled.set()
            self.record['cleanup'] = self._stop()

    def _stop(self):
        kernel_error = None
        if self.kernel_job:
            try:
                self.kernel_job.stop()
            except Exception as error:
                kernel_error = repr(error)
        elif self.owns_session and not self.group_stopped:
            # The bootstrap remains the session/group anchor after its command
            # exits. A verified group grants ownership of descendants between
            # polls, including reparented children; no name/snapshot adoption.
            root=self.owned.identities[self.child.pid]
            try:
                if root.process() is not None and os.getpgid(root.pid)==root.pid:
                    self.record['owned_process_group']=dataclasses.asdict(root)
                    os.killpg(root.pid,signal.SIGKILL)
                    self.group_stopped=True
                else:
                    try:os.killpg(root.pid,0)
                    except ProcessLookupError:pass
                    else:kernel_error='owned group anchor lost; group exit unconfirmed'
            except ProcessLookupError:
                pass
            except Exception as error:
                kernel_error=repr(error)
        cleanup = self.owned.stop()
        if kernel_error:
            cleanup['errors'].append(kernel_error)
            cleanup['confirmed'] = False
        return cleanup

    def wait(self):
        if self.child is None:
            raise RuntimeError('job has not started')
        while not self.command_status.exists() and not self.cancelled.is_set():
            if self.child.poll() is not None:
                self.failure=self.failure or 'owned bootstrap exited before command receipt'
                break
            self.finished.wait(self.poll)
        code = int(self.command_status.read_text()) if self.command_status.exists() else 137
        self.record['command_exit_code']=code
        self.close()
        if self.failure:
            raise RuntimeError(self.failure)
        return code

    def handoff(self, service_pid):
        """End a successful POSIX startup lease after verifying the owned service.

        The caller must first verify service health. Failure never hands off;
        ordinary teardown remains mandatory. Windows job detachment is refused.
        """
        if os.name=='nt':
            raise RuntimeError('Windows kernel job handoff is unsupported')
        self.owned.refresh()
        identity=self.owned.identities.get(service_pid)
        if identity is None or identity.process() is None:
            raise RuntimeError('handoff service is not an exact owned live identity')
        self._priority()
        if self.failure or self.cancelled.is_set():
            raise RuntimeError('failed startup cannot hand off')
        self.finished.set()
        self.monitor.join(timeout=10)
        if self.monitor.is_alive() or self.failure or self.cancelled.is_set():
            raise RuntimeError('startup monitor did not finish successfully')
        if identity.process() is None:
            raise RuntimeError('owned service exited during handoff')
        self.child.stdin.close()
        self.record.update(status='HEALTHY_SERVICE_HANDOFF',handoff=dataclasses.asdict(identity),
            ended=time.time(),failure=None,release_allowed=True,
            global_paging_growth=self.watcher.max_global_growth,
            owned_paged_bytes=self.watcher.job_paged)
        self.receipt.write_text(json.dumps(self.record,indent=2))
        return self.record

    def close(self):
        self.finished.set()
        if self.monitor:
            self.monitor.join(timeout=10)
            if self.monitor.is_alive():
                self.failure = self.failure or 'monitor did not terminate; lease must remain held'
        if self.owned:
            cleanup = self._stop()
            self.record['cleanup'] = cleanup
            if not cleanup['confirmed']:
                self.failure = self.failure or 'owned exit unconfirmed; lease must remain held'
        if self.child:
            if self.owned is None and self.child.poll() is None:
                self.child.terminate()  # exact launch handle, still behind the gate
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.failure = self.failure or 'launch handle exit unconfirmed; lease must remain held'
            if self.child.stdin and not self.child.stdin.closed:
                self.child.stdin.close()
        self.record.update(failure=self.failure, ended=time.time(),
            global_paging_growth=self.watcher.max_global_growth,
            owned_paged_bytes=self.watcher.job_paged,
            release_allowed=bool(self.record.get('cleanup', {}).get('confirmed')) and
                not (self.monitor and self.monitor.is_alive()))
        self.record['status']='PASS' if self.record.get('command_exit_code')==0 and not self.failure else 'FAILED_OR_CANCELLED'
        if self.kernel_job:
            self.kernel_job.close()
            self.kernel_job = None
        self.receipt.write_text(json.dumps(self.record, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rss-gb', type=float, required=True)
    parser.add_argument('--commit-gb', type=float, required=True)
    parser.add_argument('--device-commit-gb', type=float, default=0,
        help='Windows GPU backing only; verify allowance against target VRAM; host RSS/floor stay strict')
    parser.add_argument('--floor-gb', type=float, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--ledger')
    parser.add_argument('--priority-owner', action='append', default=[])
    parser.add_argument('--resource', choices=['RTX', 'MAC'], default='RTX')
    parser.add_argument('--own-lease', default='')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('an explicit command is required')
    job = None
    try:
        job = GuardedJob(rss_limit=args.rss_gb*1e9, commit_limit=args.commit_gb*1e9,
            floor=args.floor_gb*1e9, seconds=args.seconds, receipt=args.receipt,
            ledger=args.ledger, priority_owners=args.priority_owner,
            resource=args.resource, own_lease=args.own_lease,
            device_commit_allowance=args.device_commit_gb*1e9)
        job.start(command)
        return job.wait()
    except Exception as error:
        print('[shared-job-guard] REFUSED/ABORTED: ' + repr(error), file=sys.stderr, flush=True)
        if job and job.child:
            job.close()
        return 137


if __name__ == '__main__':
    sys.exit(main())
