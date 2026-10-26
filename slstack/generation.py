"""Studio quality/speed contract plus live, token-boundary sampler settings.

The native decoder reads the atomically published control at every token. It
never needs the occupied session pipe to accept a toolbar change.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import uuid
from .native_broker import vendor

PHASES = ('reasoning', 'answer', 'tool_call', 'code')
SAMPLERS = {'temperature': (0, 2), 'top_p': (0.001, 1), 'top_k': (0, 1024), 'min_p': (0, 1)}


def publish(path, row):
    """Publish once, retrying short Windows reader/antivirus rename races."""
    target = Path(path)
    tmp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.pending')
    with tmp.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(row, f, allow_nan=False);f.flush();os.fsync(f.fileno())
    deadline = time.monotonic() + 2
    while True:
        try:os.replace(tmp, target);return
        except PermissionError:
            if time.monotonic() >= deadline:raise
            time.sleep(.002)


def defaults():
    # Studio's five bias points, with the requested phase ordering. Explicit
    # sampler fields in an API request still override this profile for that turn.
    return dict(revision=1, per_phase=True, value=0.35,
        sampler=dict(temperature=0.6, top_p=0.95, top_k=0, min_p=0.0),
        phases={p: dict(bias=b, temperature=t, top_p=0.95, top_k=0, min_p=0.0)
                for p, b, t in [('reasoning', 1.36125, 1.0), ('answer', .35, .6),
                                 ('tool_call', 0.0, .1), ('code', 0.0, .2)]})


def number(value, lo, hi, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'Setting must be a finite number in [{lo},{hi}]')
    if integer and int(value) != value:raise ValueError('top_k must be an integer')
    return int(value) if integer else float(value)


def validate(row):
    if not isinstance(row, dict) or set(row) != {'revision', 'per_phase', 'value', 'sampler', 'phases'}:
        raise ValueError('Unknown or missing generation settings')
    out = deepcopy(row)
    if type(out['per_phase']) is not bool:raise ValueError('per_phase must be a boolean')
    if type(out['revision']) is not int or out['revision'] < 1:raise ValueError('Invalid revision')
    vendor();from studio.quality_speed_contract import validate_scalar
    out['value'] = validate_scalar(out['value'])
    if not isinstance(out['phases'], dict) or set(out['phases']) != set(PHASES):raise ValueError('Four phases required')
    for sampler in [out['sampler'], *out['phases'].values()]:
        phase = sampler is not out['sampler']
        if not isinstance(sampler, dict) or set(sampler) != set(SAMPLERS) | ({'bias'} if phase else set()):
            raise ValueError('Unknown or missing sampler settings')
        if phase:sampler['bias'] = validate_scalar(sampler['bias'])
        for key, (lo, hi) in SAMPLERS.items():sampler[key] = number(sampler[key], lo, hi, key == 'top_k')
    return out


class Store:
    def __init__(self, cfg):
        self.path = Path(cfg.get('generation_control_file') or Path(cfg['runtime_root']) / 'generation' / 'live.json')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.exclusive():
            if not self.path.exists():publish(self.path, defaults())

    @contextmanager
    def exclusive(self):
        with self.lock:
            # Router and single-engine proxy share this tiny configuration file.
            # Lock publication across processes, independently of generation.
            if os.name == 'nt':
                import msvcrt
                with self.path.with_suffix('.lock').open('a+b') as f:
                    if f.tell() == 0:f.write(b'0');f.flush()
                    f.seek(0);msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    try:yield
                    finally:f.seek(0);msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                with self.path.with_suffix('.lock').open('a+b') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:yield
                    finally:fcntl.flock(f, fcntl.LOCK_UN)

    def get(self):
        deadline = time.monotonic() + 1
        while True:
            try:return validate(json.loads(self.path.read_text(encoding='utf-8')))
            except PermissionError:
                if time.monotonic() >= deadline:raise
                time.sleep(.002)

    def state(self):
        applied = None
        try:
            target = Path(str(self.path) + '.applied.json')
            if target.stat().st_size <= 16384:applied = json.loads(target.read_text(encoding='utf-8'))
        except (OSError, ValueError):pass
        return dict(settings=self.get(), applied=applied, scope='session', applies='next token boundary')

    def update(self, changes):
        if not isinstance(changes, dict) or not set(changes) <= {'per_phase', 'value', 'sampler', 'phases'}:
            raise ValueError('Unknown generation setting')
        with self.exclusive():
            row = self.get()
            for key, value in changes.items():
                if key == 'sampler':row[key].update(value)
                elif key == 'phases':
                    if not isinstance(value, dict) or not set(value) <= set(PHASES):raise ValueError('Unknown phase')
                    for phase, knobs in value.items():row[key][phase].update(knobs)
                else:row[key] = value
            row['revision'] += 1
            row = validate(row);publish(self.path, row)
            return row

    def bias(self, value):
        vendor();from studio.quality_speed_contract import validate as check
        value = check(value)
        if isinstance(value, dict):return self.update(dict(per_phase=True, phases={p:dict(bias=v) for p,v in value.items()}))
        return self.update(dict(per_phase=False, value=value))


def vocabulary(tokenizer, directory):
    """Byte-exact renderer vocabulary, bounded to 16 MiB, supplied once per producer.

    This is tokenizer metadata, never a KV/state/weight spill. Native phase
    tracking reuses ds4_quality_phase.h and actual special token IDs.
    """
    target = Path(directory) / 'generation-vocabulary.bin'
    if not target.exists():
        with target.open('xb') as f:
            f.write(struct.pack('<I', len(tokenizer.tokens)))
            for token in range(len(tokenizer.tokens)):
                raw = tokenizer.token_bytes(token)
                if len(raw) > 65536:raise ValueError('Vocabulary token too large')
                f.write(struct.pack('<I', len(raw)));f.write(raw)
        if target.stat().st_size > 16 << 20:raise ValueError('Vocabulary exceeds 16 MiB')
    return str(target)
