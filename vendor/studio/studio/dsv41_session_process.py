# Derived from Studio backend/studio/dsv41_session_process.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal source export; transport and owned process logic unchanged.
from __future__ import annotations
import json, os, threading, time
from pathlib import Path
import psutil
from .dsv41_session_contract import MODEL_SHA256, MAX_CONTROL, Refused, checkpoint, command, control, require, session_identity

class Producer:

    def __init__(self, session, argv, env, directory, *, host_bytes, physical_floor, operation_seconds=600, idle_seconds=3600, native=None, cancelled=None, runtime_check=None):
        self.creation_attempted = False
        try:
            self._initialize(session, argv, env, directory, host_bytes=host_bytes, physical_floor=physical_floor, operation_seconds=operation_seconds, idle_seconds=idle_seconds, native=native, cancelled=cancelled, runtime_check=runtime_check)
        except BaseException as exc:
            if not self.creation_attempted:
                exc.owned_cleanup = dict(creation_attempted=False, owned_job_empty=True, owned_process_signaled=True, owned_pid_gone=True)
            raise

    def _initialize(self, session, argv, env, directory, *, host_bytes, physical_floor, operation_seconds, idle_seconds, native, cancelled, runtime_check):
        session_identity(session)
        require(type(host_bytes) is int and host_bytes > 0 and (type(physical_floor) is int) and (physical_floor >= 1 << 30), 'Explicit host budget/floor required')
        require(0 < operation_seconds <= 3600 and 0 < idle_seconds <= 3600, 'Bounded session deadlines required')
        require(psutil.virtual_memory().available >= host_bytes + physical_floor, 'Persistent producer host admission refused')
        if native is None:
            from studio.dsv41_windows_runner import _Native
            native = _Native()
        self.session, self.native = (session, native)
        self.host_bytes, self.floor = (host_bytes, physical_floor)
        self.operation_seconds, self.idle_seconds = (operation_seconds, idle_seconds)
        self.cancelled = cancelled or (lambda: False)
        self.runtime_check = runtime_check
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.job_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sequence = 1
        self.job = None
        self.files = []
        self.threads = []
        self.diagnostic = bytearray()
        self.failure = None
        self.started = self.activity = time.monotonic()
        self.deadline = self.started + operation_seconds
        self.peak_rss = 0
        self.minimum_available = psutil.virtual_memory().available
        self.remaining = 0
        self.transfer = None
        self.closed = False
        self.ready = None
        self.receipt = {}
        try:
            for _ in range(3):
                rd, wr = os.pipe()
                self.files.extend([os.fdopen(rd, 'rb', buffering=0), os.fdopen(wr, 'wb', buffering=0)])
            child_in, self.input, self.output, child_out, self.errors, child_err = self.files
            self.creation_attempted = True
            self.job = native.suspended(argv, env, Path(directory), child_in.fileno(), child_out.fileno(), child_err.fileno())
            child_in.close()
            child_out.close()
            child_err.close()
            self.threads = [threading.Thread(target=self._watch, name='dsv41-budget', daemon=True), threading.Thread(target=self._drain, name='dsv41-stderr', daemon=True)]
            for thread in self.threads:
                thread.start()
            self.job.resume()
            self.ready = self._receive()
            require(self.ready.get('status') == 'ready' and type(self.ready.get('next_sequence')) is int and (self.ready.get('next_sequence') == 1) and (self.ready.get('maximum_prefill_tokens') == 49152) and (self.ready.get('maximum_decode_tokens') == 128) and (self.ready.get('control_host_bytes') == 64 << 20), 'Unexpected persistent producer admission')
            checkpoint(self.ready.get('checkpoint'))
            self.deadline = None
            self.activity = time.monotonic()
            if self.runtime_check:
                self.runtime_check(self)
        except BaseException as exc:
            if self.creation_attempted and self.job is None:
                for file in self.files:
                    if not file.closed:
                        file.close()
                exc.owned_cleanup = dict(creation_attempted=True, owned_job_empty=False, owned_process_signaled=False, owned_pid_gone=False)
                raise
            self._poison(exc)
            self.close()
            exc.owned_cleanup = dict(self.receipt)
            exc.owned_diagnostic = bytes(self.diagnostic)
            raise

    def _poison(self, exc):
        with self.state_lock:
            if self.failure is None:
                self.failure = str(exc)
        with self.job_lock:
            if self.job:
                self.job.stop()

    def _watch(self):
        try:
            while not self.stop_event.wait(0.1):
                require(not self.cancelled(), 'Persistent session cancelled')
                available = psutil.virtual_memory().available
                rss = 0
                try:
                    root = psutil.Process(self.job.pid)
                    rss = sum((p.memory_info().rss for p in [root, *root.children(recursive=True)] if p.is_running()))
                except psutil.NoSuchProcess:
                    pass
                self.peak_rss = max(self.peak_rss, rss)
                self.minimum_available = min(self.minimum_available, available)
                require(available >= self.floor and rss <= self.host_bytes, 'Persistent producer host/floor budget failed')
                now = time.monotonic()
                deadline = self.deadline
                require(now < (deadline if deadline is not None else self.activity + self.idle_seconds), 'Persistent producer deadline expired')
                if self.ready is not None and self.runtime_check:
                    self.runtime_check(self)
        except BaseException as exc:
            self._poison(exc)

    def _drain(self):
        try:
            while (block := self.errors.read(65536)):
                require(len(self.diagnostic) + len(block) <= 8 << 20, 'Persistent producer diagnostic RAM cap exceeded')
                self.diagnostic.extend(block)
        except BaseException as exc:
            self._poison(exc)

    def _receive(self):
        raw = self.output.readline(65537)
        require(raw.endswith(b'\n') and len(raw) <= 65536, 'Persistent producer reply truncated or too large')
        row = control(raw)
        require(row.get('session') == self.session and row.get('model_sha256') == MODEL_SHA256, 'Producer reply identity mismatch')
        require(self.failure is None, self.failure or 'Producer failed')
        return row

    def _write(self, data):
        for off in range(0, len(data), 65536):
            page = memoryview(data)[off:off + 65536]
            while page:
                n = self.input.write(page)
                require(n is not None and n > 0, 'Producer input pipe closed')
                page = page[n:]

    def _enter(self, row):
        require(self.lock.acquire(blocking=False), 'Another session operation is active')
        writing = False
        try:
            require(not self.closed and self.failure is None, 'Session unavailable: ' + str(self.failure))
            require(self.job.poll() is None, 'Persistent producer exited')
            command(row, self.session, self.sequence)
            data = (json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode()
            require(len(data) <= MAX_CONTROL, 'Session control line exceeds bound')
            self.deadline = time.monotonic() + self.operation_seconds
            writing = True
            self._write(data)
        except BaseException as exc:
            if writing:
                self._poison(exc)
            self.lock.release()
            raise

    def _leave(self):
        self.sequence += 1
        self.activity = time.monotonic()
        self.deadline = None
        self.transfer = None
        self.remaining = 0
        self.lock.release()

    def _reply(self):
        reply = self._receive()
        require(type(reply.get('sequence')) is int and reply.get('sequence') == self.sequence and (reply.get('status') == 'complete'), 'Producer reply sequence/status mismatch')
        checkpoint(reply.get('checkpoint'))
        return reply

    def request(self, row):
        require(row.get('operation') not in ['apply_decode_delta', 'export_initial', 'export_delta'], 'Binary operation requires a stream lease')
        self._enter(row)
        try:
            return self._reply()
        except BaseException as exc:
            self._poison(exc)
            raise
        finally:
            self._leave()

    def begin_export(self, row):
        require(row.get('operation') in ['export_initial', 'export_delta'], 'Export command required')
        self._enter(row)
        try:
            meta = self._receive()
            require(meta.get('status') == 'payload' and type(meta.get('sequence')) is int and (meta.get('sequence') == self.sequence), 'Invalid export reply')
            target = checkpoint(meta.get('target'))
            require(target == row['base'], 'Export target differs from requested checkpoint')
            n = meta.get('bytes')
            require(type(n) is int and 0 < n <= 8 << 30, 'Export length exceeds native context bound')
            if row['operation'] == 'export_initial':
                require(meta.get('kind') == 'native_initial' and meta.get('base') is None, 'Initial export format')
            else:
                base = checkpoint(meta.get('base'))
                require(meta.get('kind') == 'artifact_delta_v2_1' and base['context'] == target['context'] and (0 < target['position'] - base['position'] <= 49152), 'Incremental export format')
            self.remaining = n
            self.transfer = 'out'
            return meta
        except BaseException as exc:
            self._poison(exc)
            self._leave()
            raise

    def read_export(self):
        require(self.transfer == 'out' and self.lock.locked(), 'No export lease')
        try:
            if not self.remaining:
                return b''
            wanted = min(self.remaining, 65536)
            parts = []
            received = 0
            while received < wanted:
                block = self.output.read(wanted - received)
                require(block, 'Truncated native export')
                parts.append(block)
                received += len(block)
            self.remaining -= received
            return b''.join(parts)
        except BaseException as exc:
            self._poison(exc)
            raise

    def end_export(self):
        require(self.transfer == 'out' and self.lock.locked(), 'No export lease')
        try:
            require(self.remaining == 0, 'Native export disconnected before exact extent')
        except BaseException as exc:
            self._poison(exc)
            raise
        finally:
            self._leave()

    def begin_decode(self, row):
        require(row.get('operation') == 'apply_decode_delta', 'Decode delta command required')
        self._enter(row)
        self.remaining = row['bytes']
        self.transfer = 'in'

    def write_decode(self, block):
        require(self.transfer == 'in' and self.lock.locked(), 'No decode import lease')
        try:
            require(type(block) is bytes and 0 < len(block) <= 65536 and (len(block) <= self.remaining), 'Decode stream extent exceeded')
            self._write(block)
            self.remaining -= len(block)
        except BaseException as exc:
            self._poison(exc)
            raise

    def end_decode(self):
        require(self.transfer == 'in' and self.lock.locked(), 'No decode import lease')
        try:
            require(self.remaining == 0, 'Decode delta truncated')
            reply = self._reply()
            require(reply.get('peer_synchronized') is True and reply.get('RTX_decode') is False, 'Decode sync not certified')
            return reply
        except BaseException as exc:
            self._poison(exc)
            raise
        finally:
            self._leave()

    def abort_transfer(self):
        self._poison(Refused('Native transfer cancelled; session invalidated'))
        if self.transfer is not None and self.lock.locked():
            self._leave()

    def close(self):
        if self.closed:
            require(self.receipt.get('owned_job_empty') and self.receipt.get('owned_process_signaled') and self.receipt.get('owned_pid_gone'), 'Previous producer cleanup unconfirmed')
            return self.receipt
        self.closed = True
        self.stop_event.set()
        with self.job_lock:
            if self.job:
                self.job.stop()
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(10)
        require(not any((t.is_alive() for t in self.threads)), 'Persistent producer reader did not stop')
        pid = self.job.pid if self.job else None
        end = time.monotonic() + 5
        while self.job is not None and self.job.poll() is None and (time.monotonic() < end):
            time.sleep(0.01)
        exited = self.job is None or self.job.poll() is not None
        empty = self.job is None or self.job.active() == 0
        self.receipt = dict(session=self.session, host_budget=self.host_bytes, physical_floor=self.floor, peak_rss=self.peak_rss, minimum_available_ram=self.minimum_available, failure=self.failure, owned_pid=pid, owned_job_empty=empty, owned_process_signaled=exited)
        with self.job_lock:
            if self.job:
                self.job.close()
                self.job = None
        for file in self.files:
            if not file.closed:
                file.close()
        end = time.monotonic() + 2
        while pid is not None and psutil.pid_exists(pid) and (time.monotonic() < end):
            time.sleep(0.01)
        self.receipt['owned_pid_gone'] = pid is None or not psutil.pid_exists(pid)
        require(empty and exited and self.receipt['owned_pid_gone'], 'Persistent producer exit unconfirmed')
        return self.receipt
