"""Shared short GPU leases. No production service is stopped or adopted."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
import psutil
from .journal import atomic_json


class AdmissionRefused(RuntimeError):
    pass


def recover_leases(runtime_root, task='DS4STACK'):
    """Release only this stack's recorded dead guardians, after their kill-on-close Jobs have closed."""
    root = Path(runtime_root) / 'state' / 'leases'
    if not root.exists():
        return []
    released = []
    for file in root.glob('*.json'):
        row = json.loads(file.read_text())
        if row.get('task') != task or row.get('released') or row.get('cleanup_uncertain'):
            continue
        try:
            proc = psutil.Process(row['guardian_pid'])
            if abs(proc.create_time() - row['guardian_created']) < .01 and proc.is_running():
                continue
        except psutil.NoSuchProcess:
            pass
        # On Windows the guardian handle closes its Job and terminates all GPU descendants.
        # Recovering an active hold never kills an unrelated process or guesses expiry.
        with Path(row['hold_file']).open('a', encoding='utf-8') as stream:
            stream.write(f'\nRELEASED: {task} {row["id"]}; recovered dead guardian {row["guardian_pid"]}; {datetime.now(timezone.utc).isoformat()}\n')
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(file, {**row, 'released': True, 'recovered': True})
        released.append(row['id'])
    return released


def active_holds(text, *, mac=None):
    """Unreleased ledger holds, parsed by the vendored Studio priority_holds.

    mac=False: RTX holds only; mac=True: MAC holds only. The default (None) is the
    RTX admission view: RTX holds plus MAC KIT11* holds, which also block the RTX.
    """
    from .native_broker import vendor
    vendor()
    from studio.job_guard import priority_holds
    if mac:
        return priority_holds(text, ('*',), resource='MAC')
    rtx = priority_holds(text, ('*',), resource='RTX')
    if mac is False:
        return rtx
    return rtx + [line for line in priority_holds(text, ('*',), resource='MAC')
                  if re.match(r'^MAC ACTIVE:\s+KIT11[A-Z0-9]*\s', line)]


def reservation_admission(lines, budget_gib, floor_gib, *, available=None, owned_rss=None):
    """Owner-approved KIT11FIX CPU exception; other/new/larger holds preempt."""
    blocking=[];reserved=0
    for line in lines:
        original_cpu=(line.startswith('ACTIVE: KIT11FIX Windows native plugin build;') and
             'host budget 3 GiB; physical RAM reserve 6 GiB;' in line and
             'no GPU or Studio restart requested.' in line and
             not re.search(r'exclusive|GPU hold',line,re.I))
        # The current qualification names the same CPU-only reservation with
        # a smaller2GiB job cap. Keep the owner's full3+6GiB reservation rather
        # than treating that CPU build as a GPU hold or reducing its allowance.
        # An unrecognised, GPU, or larger reservation still preempts immediately.
        current_cpu=re.fullmatch(
            r'ACTIVE: KIT11FIX [a-z][a-z0-9-]*; expires [0-9T:.+Z-]+; '
            r'Windows-only CPU build/test, host budget ([0-9]+(?:\.[0-9]+)?) GiB, reserve ([0-9]+(?:\.[0-9]+)?) GiB; '
            r'yield RAM for bounded [A-Z][A-Z0-9]* window; no Studio restart',line)
        current_cpu=(current_cpu is not None and
                     0<float(current_cpu[1])<=3 and 0<float(current_cpu[2])<=6)
        if original_cpu or current_cpu:reserved+=9*2**30
        else:blocking.append(line)
    available=psutil.virtual_memory().available if available is None else available
    floor=max(floor_gib,2)*2**30 if reserved else floor_gib*2**30
    # At admission reserve the full budget. During execution the extra
    # outstanding-allocation rule applies to the owner's CPU reservation.
    # With no reservation, the shared guard enforces actual RSS/commit/floor;
    # each subsequent engine/locked-state allocation also has its own admission.
    outstanding=max(0,budget_gib*2**30-(owned_rss or 0))
    required=reserved+floor+(outstanding if reserved or owned_rss is None else 0)
    if available<required:
        blocking.append(f'RAM reservation: available={available}, required={required}, owned_rss={owned_rss}')
    return blocking,dict(available=available,required=required,foreign_reserved_bytes=reserved,
                         own_budget_bytes=budget_gib*2**30,owned_rss=owned_rss,floor_bytes=floor)


def hold_admission(lines, *, budget_gib=0, floor_gib=2, available=None):
    """Explicit CPU-only reservations share the GPU only with reserved RAM intact."""
    blocking, reserved = [], 0.0
    for line in lines:
        budget = re.search(r'host budget ([\d.]+) GiB', line)
        reserve = re.search(r'physical RAM reserve ([\d.]+) GiB', line)
        if (budget and reserve and 'no GPU or Studio restart requested' in line
                and not re.search(r'exclusive|GPU hold', line, re.I)):
            reserved += float(budget[1]) + float(reserve[1])
        else:
            blocking.append(line)
    floor = max(floor_gib, reserved + 2) if reserved else floor_gib
    available = psutil.virtual_memory().available if available is None else available
    if reserved and available < (budget_gib + floor) * 2**30:
        blocking.append(f'CPU reservations require {budget_gib + floor:g} GiB available; '
                        f'available={available / 2**30:.3f} GiB')
    return blocking


class HoldWatcher:
    """Fail closed on a new foreign hold; unchanged ledgers cost one stat call."""
    def __init__(self, path, identity, *, mac=False, task='DS4STACK'):
        self.path, self.identity = Path(path), identity
        self.mac, self.task = mac, task
        self.stamp = None
        self.blocking = None
        self.foreign = []

    def reason(self):
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self.stamp:
            if stat.st_size > 16 * 1024**2:
                raise RuntimeError('Shared hold ledger exceeds its observation budget')
            blocking = []
            for line in active_holds(self.path.read_text(encoding='utf-8-sig'), mac=bool(self.mac)):
                match = re.match(r'^(?:MAC |RTX )?ACTIVE:\s+(\S+)\s+(\S+)', line.strip())
                if not match or (match[1], match[2].rstrip(';')) != (self.task, self.identity):
                    blocking.append(line)
            self.foreign = blocking
            self.stamp = stamp
        self.blocking = '\n'.join(self.foreign if self.mac else hold_admission(self.foreign)) or None
        return self.blocking


@contextmanager
def gpu_lease(cfg):
    task = cfg.get('task', 'DS4STACK')
    if not isinstance(task, str) or re.fullmatch('[A-Z][A-Z0-9]{0,31}', task) is None:
        raise ValueError('GPU lease task must be a bounded coordination identifier')
    hold = Path(cfg['hold_file']) if cfg.get('hold_file') else Path(cfg['runtime_root']) / 'state' / 'gpu.holds'
    parent = cfg.get('_gpu_parent_lease')
    if parent:
        # Only a live process ancestor may delegate its existing reservation.
        # Children never release it or claim another GPU window inside it.
        record = Path(parent)
        row = json.loads(record.read_text(encoding='utf-8'))
        owner = next((p for p in [psutil.Process(), *psutil.Process().parents()]
                      if p.pid == row.get('guardian_pid')), None)
        if (row.get('task') != task or row.get('released') is not False
                or row.get('cleanup_uncertain') or owner is None
                or abs(owner.create_time() - row.get('guardian_created', 0)) >= .01
                or Path(row.get('hold_file', '')).resolve() != hold.resolve()
                or record.name != str(row.get('id')) + '.json'
                or cfg['host_budget_gib'] > row.get('host_budget_gib', 0)
                or cfg['min_available_gib'] < row.get('min_available_gib', float('inf'))
                or datetime.now(timezone.utc) >= datetime.fromisoformat(row['expires'])):
            raise AdmissionRefused('Inherited GPU reservation is not a live compatible ancestor lease')
        lines = active_holds(hold.read_text(encoding='utf-8-sig'))
        own = [line for line in lines if re.match(
            r'^ACTIVE:\s+' + re.escape(task) + r'\s+' + re.escape(row['id']) + r'(?:;|\s|$)', line)]
        foreign = [line for line in lines if line not in own]
        if len(own) != 1 or hold_admission(foreign, budget_gib=cfg['host_budget_gib'],
                                         floor_gib=cfg['min_available_gib']):
            raise AdmissionRefused('Inherited GPU reservation is absent or a foreign hold blocks it')
        yield row['id']
        return
    if not cfg.get('hold_file'):
        hold.parent.mkdir(parents=True, exist_ok=True)
        hold.touch(exist_ok=True)
    # Windows byte-range lock serializes admission among instances of this stack.
    # Other agents' leases are respected via the shared append-only hold file.
    with hold.open('r+', encoding='utf-8-sig') as stream:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            lines = stream.read()
            active = hold_admission(active_holds(lines), budget_gib=cfg['host_budget_gib'],
                                    floor_gib=cfg['min_available_gib'])
            if active:
                raise AdmissionRefused('RTX admission refused: ACTIVE holds exist: ' + '\n'.join(active))
            # WDDM reports every graphics context as a compute client with N/A memory.
            # Check the selected GPU's actual baseline instead of rejecting Explorer/Slack.
            result = subprocess.run(['nvidia-smi', '--id=' + cfg['gpu_uuid'],
                                     '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise RuntimeError('Cannot certify GPU ownership: ' + result.stderr)
            used, free, busy = [int(x.strip()) for x in result.stdout.strip().split(',')]
            if used > cfg['gpu_idle_max_mib'] or free < 30000 or busy > 5:
                raise AdmissionRefused('RTX is not idle enough for exclusive admission; no other process was ejected: ' + result.stdout)
            identity = uuid.uuid4().hex
            expiry = datetime.now(timezone.utc) + timedelta(minutes=cfg['lease_minutes'])
            lease_path = Path(cfg['runtime_root']) / 'state' / 'leases' / (identity + '.json')
            lease_row = {'task': task, 'id': identity, 'hold_file': str(hold),
                         'guardian_pid': os.getpid(), 'guardian_created': psutil.Process().create_time(),
                         'host_budget_gib': cfg['host_budget_gib'],
                         'min_available_gib': cfg['min_available_gib'], 'expires': expiry.isoformat(),
                         'released': False}
            atomic_json(lease_path, lease_row)
            line = f"\nACTIVE: {task} {identity}; exclusive RTX5090; expires {expiry.isoformat()}; host RSS {cfg['host_budget_gib']} GiB; floor {cfg['min_available_gib']} GiB\n"
            stream.seek(0, 2)
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    uncertain=False
    try:
        yield identity
    except BaseException as error:
        uncertain=bool(getattr(error,'owned_cleanup_uncertain',False))
        if uncertain:atomic_json(lease_path,{**lease_row,'cleanup_uncertain':True})
        raise
    finally:
        if not uncertain:
            with hold.open('a', encoding='utf-8') as stream:
                stream.write(f'\nRELEASED: {task} {identity}; {datetime.now(timezone.utc).isoformat()}\n')
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(lease_path, {**lease_row, 'released': True})
