from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
import psutil


def sha256(path, progress=None):
    """A folder hash is a sorted manifest of relative paths and file hashes; no weights copied.
    progress(done_bytes, total_bytes), when given, is called after each block of a file."""
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    if p.is_dir():
        for f in sorted(p.rglob('*')):
            if f.is_file():
                if f.is_symlink():
                    raise ValueError(f'Model manifests do not accept symlinks: {f}')
                h.update(f.relative_to(p).as_posix().encode() + b'\0')
                h.update(bytes.fromhex(sha256(f)))
    else:
        total, done = p.stat().st_size, 0
        for block in _blocks(p):
            h.update(block)  # hashlib releases the GIL here, so the reader keeps the disk busy
            if progress:
                done += len(block)
                progress(done, total)
    return h.hexdigest()


def _blocks(path, block_bytes=8 * 1024 * 1024, depth=4):
    """The file's bytes in order, read on a separate thread so reading overlaps hashing
    (serial read+hash ran at ~1.25 GB/s; the disk and SHA-256 each manage ~2.2-3 GB/s)."""
    import queue
    blocks, stop, failure = queue.Queue(maxsize=depth), threading.Event(), []

    def reader():
        try:
            with open(path, 'rb', buffering=0) as stream:
                while not stop.is_set():
                    block = stream.read(block_bytes)
                    blocks.put(block)
                    if not block:
                        return
        except BaseException as error:
            failure.append(error)
            blocks.put(b'')

    thread = threading.Thread(target=reader, name='sha256-reader', daemon=True)
    thread.start()
    try:
        while True:
            block = blocks.get()
            if not block:
                break
            yield block
    finally:
        stop.set()
        while thread.is_alive():  # unblock a reader waiting on a full queue
            try:
                blocks.get_nowait()
            except queue.Empty:
                thread.join(0.05)
        thread.join()
    if failure:
        raise failure[0]


def atomic_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, p)
    if os.name != 'nt':
        fd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Journal:
    def __init__(self, root):
        self.path = Path(root) / 'state' / 'lora_journal.json'
        self.thread_lock = threading.RLock()
        self._base_hashes = {}

    def base_digest(self, base, *, fresh=False):
        p = Path(base).resolve(strict=True)
        if not p.is_file():
            return sha256(p)
        st = p.stat()
        signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        cached = self._base_hashes.get(str(p))
        if fresh or cached is None or cached[0] != signature:
            digest = sha256(p)
            after = p.stat()
            if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError('Base changed during SHA-256 verification')
            self._base_hashes[str(p)] = (signature, digest)
        return self._base_hashes[str(p)][1]

    @contextmanager
    def locked(self):
        with self.thread_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix('.lock').open('a+b') as lock:
                lock.seek(0)
                lock.write(b'0')
                lock.flush()
                lock.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    yield
                finally:
                    lock.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {'version': 1, 'entries': []}

    def recover(self):
        reports = []
        with self.locked():
            data = self.read()
            for row in data['entries']:
                # Verify every recorded base, including completed merges, before any new operation.
                if self.base_digest(row['base_path']) != row['base_sha256']:
                    raise RuntimeError(f"BASE HASH MISMATCH: {row['base_path']}; refusing all operations")
                if row['status'] == 'detached':
                    row['applied'] = False
                if row['status'] not in ('attaching', 'attached', 'merging'):
                    continue
                try:
                    proc = psutil.Process(row['pid'])
                    alive = abs(proc.create_time() - row['pid_created']) < .01
                except psutil.NoSuchProcess:
                    alive = False
                if alive:
                    raise RuntimeError(f"LoRA journal is owned by live PID {row['pid']}")
                partial = row.get('partial_path')
                # Preserve partial bytes recoverably. Never delete or overwrite an unverified output.
                if partial and Path(partial).exists():
                    original = Path(partial)
                    if original.parent != Path(row['output_path']).parent or not original.name.endswith('.partial-' + row['id']):
                        raise RuntimeError('Invalid partial output ownership in journal')
                    quarantine = original.with_name(original.name + '.recovered-' + uuid.uuid4().hex)
                    original.rename(quarantine)
                    row['recovered_partial'] = str(quarantine)
                row['status'] = 'detached' if row['status'] in ('attaching', 'attached') else 'failed'
                row['applied'] = False
                row['recovered_at'] = datetime.now(timezone.utc).isoformat()
                reports.append({'id': row['id'], 'status': row['status'], 'base_verified': True,
                                'partial': row.get('recovered_partial')})
            atomic_json(self.path, data)
        return reports

    def begin(self, base, loras, status, output=None):
        # Call recover once at command entry, not while another attach is active.
        row = {'id': uuid.uuid4().hex, 'base_path': str(Path(base).resolve()), 'base_sha256': self.base_digest(base),
               'loras': [{'path': str(Path(p).resolve()), 'sha256': sha256(p), 'scale': float(s)} for p, s in loras],
               'output_path': str(Path(output).resolve()) if output else None, 'output_sha256': None,
               'status': status, 'pid': os.getpid(), 'pid_created': psutil.Process().create_time(),
               'time': datetime.now(timezone.utc).isoformat()}
        if output:
            row['partial_path'] = row['output_path'] + '.partial-' + row['id']
        with self.locked():
            data = self.read()
            data['entries'].append(row)
            atomic_json(self.path, data)
        return row

    def update(self, identity, status, **fields):
        with self.locked():
            data = self.read()
            row = next(r for r in data['entries'] if r['id'] == identity)
            if self.base_digest(row['base_path'], fresh=status in ('detached', 'merged', 'failed')) != row['base_sha256']:
                raise RuntimeError('BASE HASH MISMATCH; journal transition refused')
            row.update(fields, status=status, time=datetime.now(timezone.utc).isoformat())
            atomic_json(self.path, data)
        return row
