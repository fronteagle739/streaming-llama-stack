"""Shared short GPU leases. No production service is stopped or adopted."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


def active_holds(text):
    active = {}
    for line in text.splitlines():
        match = re.match(r'^(?:MAC )?(ACTIVE|RELEASED|ENDED):\s+(\S+)\s+(\S+)', line)
        if match:
            state, task, identity = match.groups()
            key = (task, identity.rstrip(';'))
            if state == 'ACTIVE':
                active[key] = line
            else:
                active.pop(key, None)
    return list(active.values())


@contextmanager
def gpu_lease(cfg):
    hold = Path(cfg['hold_file'])
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
                raise RuntimeError('RTX admission refused: ACTIVE holds exist: ' + '\n'.join(active))
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise RuntimeError('Cannot certify GPU ownership: ' + result.stderr)
            if result.stdout.strip():
                raise RuntimeError('GPU already has compute clients; stack will not eject unowned processes: ' + result.stdout)
            identity = uuid.uuid4().hex
            expiry = datetime.now(timezone.utc) + timedelta(minutes=cfg['lease_minutes'])
            line = f"\nACTIVE: DS4STACK {identity}; exclusive RTX5090; expires {expiry.isoformat()}; host RSS {cfg['host_budget_gib']} GiB; floor {cfg['min_available_gib']} GiB\n"
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
            stream.write(f'\nRELEASED: DS4STACK {identity}; {datetime.now(timezone.utc).isoformat()}\n')
            stream.flush()
            os.fsync(stream.fileno())
