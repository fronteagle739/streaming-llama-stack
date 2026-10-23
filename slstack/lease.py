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
        if row.get('task') != task or row.get('released'):
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


def active_holds(text):
    active = {}
    for line in text.splitlines():
        mac=line.startswith('MAC ')
        if mac and not re.match(r'^MAC (ACTIVE|RELEASED|ENDED):\s+KIT11[A-Z0-9]*\s',line):
            continue
        match = re.match(r'^(?:MAC )?(ACTIVE|RELEASED|ENDED):\s+(\S+)\s+(\S+)', line)
        if match:
            state, task, identity = match.groups()
            key = (mac, task, identity.rstrip(';'))
            if state == 'ACTIVE':
                active[key] = line
            else:
                active.pop(key, None)
    return list(active.values())


@contextmanager
def gpu_lease(cfg):
    task = cfg.get('task', 'DS4STACK')
    if not isinstance(task, str) or re.fullmatch('[A-Z][A-Z0-9]{0,31}', task) is None:
        raise ValueError('GPU lease task must be a bounded coordination identifier')
    hold = Path(cfg['hold_file']) if cfg.get('hold_file') else Path(cfg['runtime_root']) / 'state' / 'gpu.holds'
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
            active = active_holds(lines)
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
            lease_path = Path(cfg['runtime_root']) / 'state' / 'leases' / (identity + '.json')
            lease_row = {'task': task, 'id': identity, 'hold_file': str(hold),
                         'guardian_pid': os.getpid(), 'guardian_created': psutil.Process().create_time(),
                         'released': False}
            atomic_json(lease_path, lease_row)
            expiry = datetime.now(timezone.utc) + timedelta(minutes=cfg['lease_minutes'])
            line = f"\nACTIVE: {task} {identity}; exclusive RTX5090; expires {expiry.isoformat()}; host RSS {cfg['host_budget_gib']} GiB; floor {cfg['min_available_gib']} GiB\n"
            stream.seek(0, 2)
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    try:
        yield identity
    finally:
        with hold.open('a', encoding='utf-8') as stream:
            stream.write(f'\nRELEASED: {task} {identity}; {datetime.now(timezone.utc).isoformat()}\n')
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(lease_path, {**lease_row, 'released': True})
