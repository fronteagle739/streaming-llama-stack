# Derived from Studio backend/studio/dsv41_session_process.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal source export; transport retained with optional mode1 startup and resource-admission hooks.
from __future__ import annotations
import json, logging, math, os, threading, time
from pathlib import Path
import psutil
from .dsv41_session_contract import MODEL_SHA256, MAX_CONTROL, MAX_DECODE, MAX_PREFILL, Refused, checkpoint, command, control, negotiate_decode_delta, negotiate_prefill, require, session_identity

log = logging.getLogger(__name__)

def now_tick(owner, period=1.0):
    """True at most once per period for owner (the tree scan is cheaper at 1 Hz than 10 Hz)."""
    now = time.monotonic()
    if now - getattr(owner, '_tick', 0.0) < period:return False
    owner._tick = now;return True


def operation_live(now, deadline, last_progress, stall_seconds):
    """Progress-based liveness (owner: no fixed wall-clock limits that a 262K prompt or a long
    generation would hit). An active operation lives while the producer shows progress (pipe
    bytes, diagnostics, CPU time or I/O of its job tree) within stall_seconds; deadline is only
    the silence bound for an operation that never showed any progress."""
    if deadline is None:return True
    if stall_seconds:return now < max(deadline, last_progress + stall_seconds)
    return now < deadline


class Producer:
    # Owner rule: never force-kill a producer that may hold VRAM. Stop = Ctrl-Break,
    # graceful_seconds, a second Ctrl-Break, up to stuck_seconds; still alive = STUCK,
    # left running, receipt/log marked, GPU work refused (the caller keeps its hold).
    graceful_seconds = 30
    stuck_seconds = 120
    stuck = None

    def _stop_job(self):
        from .dsv41_windows_runner import StuckProcess
        with self.job_lock:
            if self.job and not self.stuck:
                try:
                    self.job.stop(graceful_seconds=self.graceful_seconds, stuck_seconds=self.stuck_seconds)
                except StuckProcess as exc:
                    self.stuck = str(exc)
                    with self.state_lock:
                        self.failure = self.stuck if self.failure is None else self.failure + '; ' + self.stuck

    def __init__(self, session, argv, env, directory, *, host_bytes, physical_floor, operation_seconds=600, idle_seconds=3600, native=None, cancelled=None, runtime_check=None, native_mode=3, bootstrap=None, startup=None, owner_epoch=0, after_resume=None, stall_seconds=None, runtime_min_free_gib=4.0):
        self.creation_attempted = False
        self.stall_seconds = stall_seconds
        self.last_progress = time.monotonic();self._progress_mark = None
        try:
            require(type(runtime_min_free_gib) in (int, float) and math.isfinite(runtime_min_free_gib) and 0 <= runtime_min_free_gib < 2**34,
                    'runtime_min_free_gib must be finite and nonnegative (<2**34); 0 disables it')
            self.runtime_min_free_gib = float(runtime_min_free_gib)
            self.runtime_floor = int(runtime_min_free_gib * (1 << 30))
            self.native_mode = native_mode
            require(type(owner_epoch) is int and 0<=owner_epoch<(1<<64), 'Exact bounded owner epoch required')
            self.owner_epoch = owner_epoch
            require(type(native_mode) is int and native_mode in (1, 3), 'Explicit native session mode must be1 or3')
            require(bootstrap is None or native_mode == 1, 'Native bootstrap requires explicit target-only mode1')
            self.bootstrap = bootstrap
            require(startup is None or callable(startup) and native_mode == 1 and bootstrap is None, 'HTTP-native startup requires explicit target-only mode1 and no pipe bootstrap')
            self.startup = startup
            require(after_resume is None or callable(after_resume), 'Pre-readiness admission must be callable')
            self.after_resume = after_resume
            self.resume_admission = None
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
        # Admission and runtime floors are independent. The runtime available-RAM stop defaults to
        # the known safe 4 GiB; 0 opts into paging-only. Keep the existing paging stop as well.
        from .job_paging import paging_bytes, GLOBAL_PAGING_LIMIT
        self._paging_bytes, self._paging_limit = paging_bytes, GLOBAL_PAGING_LIMIT
        self._paging0 = paging_bytes(); self.paging_growth = 0; self._paging_tick = 0
        self.budget_breaches = 0; self.first_budget_breach = None
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
            if self.startup is not None:
                self.threads.append(threading.Thread(target=self._drain_stdout, name='dsv41-stdout', daemon=True))
            for thread in self.threads:
                thread.start()
            self.job.resume()
            if self.after_resume is not None:
                self.resume_admission = self.after_resume(self.job)
            if self.bootstrap is not None:
                self.bootstrap(self._write)
            if self.startup is None:
                self.ready = self._receive()
                require(self.ready.get('status') == 'ready' and type(self.ready.get('next_sequence')) is int and (self.ready.get('next_sequence') == 1) and (self.ready.get('maximum_prefill_tokens') == 49152) and (type(self.ready.get('maximum_decode_tokens')) is int and self.ready['maximum_decode_tokens'] >= MAX_DECODE) and (self.ready.get('control_host_bytes') == 64 << 20), 'Unexpected persistent producer admission')
                checkpoint(self.ready.get('checkpoint'), self.native_mode)
                # Negotiated decode-delta maximum, published to the peer through the broker's metadata.
                self.ready['maximum_decode_delta_tokens'] = self.decode_delta_max = negotiate_decode_delta(self.ready)
                # LLAMADECODE single sweep: negotiated one-op prefill maximum (49152 for old producers).
                self.ready['maximum_prefill_sweep_tokens'] = self.prefill_max = negotiate_prefill(self.ready)
            else:
                self.ready = self.startup(self)
                require(type(self.ready) is dict and self.ready.get('status') == 'ready', 'HTTP-native readiness was not certified')
            require(not self.owner_epoch or type(self.ready.get('owner_epoch')) is int and self.ready['owner_epoch'] == self.owner_epoch, 'Native process readiness owner epoch differs')
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
            try:
                self._poison(exc)
                self.close()
            except BaseException as cleanup:
                exc.owned_cleanup={'creation_attempted':self.creation_attempted,
                    'owned_job_empty':False,'owned_process_signaled':False,'owned_pid_gone':False,**self.receipt}
                exc.owned_process=self
                exc.owned_cleanup_error=repr(cleanup)
                raise exc from cleanup
            exc.owned_cleanup = dict(self.receipt)
            exc.owned_diagnostic = bytes(self.diagnostic)
            raise

    def _poison(self, exc):
        with self.state_lock:
            if self.failure is None:
                self.failure = str(exc)
        self._stop_job()

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
                if self.stall_seconds and now_tick(self):self._note_tree_progress()
                self.minimum_available = min(self.minimum_available, available)
                if self.runtime_floor and available < self.runtime_floor:
                    reason = 'Persistent producer stopped: available physical RAM %d bytes below runtime_min_free_gib=%g (%d bytes)' % (available, self.runtime_min_free_gib, self.runtime_floor)
                    log.critical(reason)
                    raise Refused(reason)
                if available < self.floor or rss > self.host_bytes:
                    self.budget_breaches += 1
                    if self.first_budget_breach is None:
                        self.first_budget_breach = dict(available=available, rss=rss, floor=self.floor, host_budget=self.host_bytes)
                self._paging_tick += 1
                if self._paging_tick % 10 == 0:   # ~1 Hz
                    self.paging_growth = max(self.paging_growth, self._paging_bytes() - self._paging0)
                    require(self.paging_growth <= self._paging_limit,
                            'Persistent producer stopped: paging detected (pagefile usage rose %d bytes)' % self.paging_growth)
                now = time.monotonic()
                deadline = self.deadline
                if deadline is not None:require(operation_live(now, deadline, self.last_progress, self.stall_seconds), 'Persistent producer made no progress within its stall window')
                else:require(now < self.activity + self.idle_seconds, 'Persistent producer idle deadline expired')
                if self.ready is not None and self.runtime_check:
                    self.runtime_check(self)
        except BaseException as exc:
            self._poison(exc)

    def _drain(self):
        try:
            while (block := self.errors.read(65536)):
                with self.state_lock:
                    require(len(self.diagnostic) + len(block) <= 8 << 20, 'Persistent producer diagnostic RAM cap exceeded')
                    self.diagnostic.extend(block)
                self.last_progress = time.monotonic()
        except BaseException as exc:
            self._poison(exc)

    def _drain_stdout(self):
        try:
            while (block := self.output.read(65536)):
                with self.state_lock:
                    require(len(self.diagnostic) + len(block) <= 8 << 20, 'Native diagnostic RAM cap exceeded')
                    self.diagnostic.extend(block)
                self.last_progress = time.monotonic()
        except BaseException as exc:
            self._poison(exc)

    def _note_tree_progress(self):
        """CPU time and I/O bytes of the producer's job tree; any growth is progress."""
        mark = 0
        try:
            root = psutil.Process(self.job.pid)
            for p in [root, *root.children(recursive=True)]:
                try:
                    t = p.cpu_times();io = p.io_counters()
                    mark += int((t.user + t.system) * 1000) + io.read_bytes + io.write_bytes + io.other_bytes
                except psutil.Error:
                    pass
        except psutil.Error:
            return
        if mark != self._progress_mark:
            self._progress_mark = mark;self.last_progress = time.monotonic()

    death_wait_seconds = 5.0

    def _death(self, message):
        """LLAMADECODE: why the producer stopped answering: its exit code, its final
        LLAMADECODE_PRODUCER_ABORT line (else the stderr tail), so a dead producer
        never surfaces as a bare 'truncated' reply."""
        code = None
        end = time.monotonic() + self.death_wait_seconds
        while True:
            try:
                code = self.job.poll() if self.job is not None else None
            except Exception:
                code = None
            if code is not None or time.monotonic() >= end:
                break
            time.sleep(0.05)
        if code is not None:   # the stderr reader ends at the pipe's EOF: let it collect the last lines
            for thread in getattr(self, 'threads', []):
                if thread.name == 'dsv41-stderr' and thread is not threading.current_thread():
                    thread.join(max(0.0, end - time.monotonic()) + 1.0)
        self.exit_code = code
        with self.state_lock:
            text = bytes(self.diagnostic).decode('utf-8', errors='replace')
            message = self.failure or message
        aborts = [line for line in text.splitlines() if line.startswith('LLAMADECODE_PRODUCER_ABORT ')]
        if code is None:
            parts = [message, 'producer still running (no exit code yet)']
        else:
            parts = [message, 'producer exit code %d (0x%08X)' % (code, code & 0xFFFFFFFF)]
        if aborts:
            parts.append(aborts[-1])
        else:
            tail = [line for line in text.splitlines() if line.strip()][-3:]
            parts.append('no LLAMADECODE_PRODUCER_ABORT line; stderr tail: ' + (' | '.join(tail)[-1200:] if tail else '(empty)'))
        return '; '.join(parts)

    def _receive(self):
        raw = self.output.readline(65537)
        self.last_progress = time.monotonic()
        require(self.failure is None, self.failure or 'Producer failed')
        if not (raw.endswith(b'\n') and len(raw) <= 65536):
            raise Refused(self._death('Persistent producer reply truncated or too large' if len(raw) > 65536 or raw
                                      else 'Persistent producer closed its reply pipe'))
        row = control(raw)
        require(row.get('session') == self.session and row.get('model_sha256') == MODEL_SHA256, 'Producer reply identity mismatch')
        require(not self.owner_epoch or type(row.get('owner_epoch')) is int and row['owner_epoch'] == self.owner_epoch, 'Producer reply owner epoch mismatch')
        if row.get('status') == 'error':   # LLAMADECODE: the producer's abort reply carries its reason
            raise Refused(self._death('Persistent producer aborted (%s, sequence %s): %s' % (row.get('error_kind'), row.get('sequence'), row.get('reason'))))
        require(self.failure is None, self.failure or 'Producer failed')
        return row

    def _write(self, data):
        for off in range(0, len(data), 65536):
            page = memoryview(data)[off:off + 65536]
            while page:
                n = self.input.write(page)
                require(n is not None and n > 0, 'Producer input pipe closed')
                page = page[n:]
                self.last_progress = time.monotonic()

    def _enter(self, row):
        require(self.lock.acquire(blocking=False), 'Another session operation is active')
        writing = False
        try:
            require(not self.closed and self.failure is None, 'Session unavailable: ' + str(self.failure))
            require(self.job.poll() is None, 'Persistent producer exited')
            command(row, self.session, self.sequence, self.native_mode, getattr(self, 'prefill_max', MAX_PREFILL))
            wire = dict(row)
            if self.owner_epoch:
                wire['owner_epoch'] = self.owner_epoch
            data = (json.dumps(wire, separators=(',', ':'), allow_nan=False) + '\n').encode()
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
        checkpoint(reply.get('checkpoint'), self.native_mode)
        return reply

    def request(self, row):
        require(row.get('operation') not in ['apply_decode_delta', 'export_initial', 'export_delta', 'export_local'], 'Binary operation requires a stream lease')
        self._enter(row)
        try:
            return self._reply()
        except BaseException as exc:
            self._poison(exc)
            raise
        finally:
            self._leave()

    # DS4COMBO fast switch: the producer process, CUDA context and model lease stay
    # alive while its device memory and carry RAM are given back to the native engine.
    def suspend(self, base):
        reply = self.request(dict(session=self.session, model_sha256=MODEL_SHA256, sequence=self.sequence, operation='suspend', base=base))
        require(reply.get('suspended') is True, 'Producer suspend was not certified')
        return reply

    def resume(self, contract_sha256, tokens, bootstrap):
        require(callable(bootstrap), 'Resume requires the native base writer')
        self._enter(dict(session=self.session, model_sha256=MODEL_SHA256, sequence=self.sequence, operation='resume',
                         native_base_contract_sha256=contract_sha256, tokens=list(tokens)))
        try:
            bootstrap(self._write)
            reply = self._reply()
            require(reply.get('peer_synchronized') is False and reply.get('prompt_tokens') == len(tokens), 'Producer resume prefill was not certified')
            return reply
        except BaseException as exc:
            self._poison(exc)
            raise
        finally:
            self._leave()

    def begin_export(self, row):
        require(row.get('operation') in ['export_initial', 'export_delta', 'export_local'], 'Export command required')
        self._enter(row)
        try:
            meta = self._receive()
            require(meta.get('status') == 'payload' and type(meta.get('sequence')) is int and (meta.get('sequence') == self.sequence), 'Invalid export reply')
            target = checkpoint(meta.get('target'), self.native_mode)
            require(target == row['base'], 'Export target differs from requested checkpoint')
            n = meta.get('bytes')
            require(type(n) is int and 0 < n <= 8 << 30, 'Export length exceeds native context bound')
            if row['operation'] == 'export_local':
                require(self.ready.get('local_snapshot_version') == 1 and meta.get('kind') == 'local_snapshot_v1' and meta.get('base') is None, 'Local snapshot format')
                require(meta.get('yoco_keep') == self.ready.get('yoco_keep') and meta.get('cache_precision') in ('f32', 'bf16'), 'Local snapshot preset differs')
            elif row['operation'] == 'export_initial':
                require(meta.get('kind') == 'native_initial' and meta.get('base') is None, 'Initial export format')
            else:
                base = checkpoint(meta.get('base'), self.native_mode)
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
        require(row['target']['position'] - row['base']['position'] <= getattr(self, 'decode_delta_max', MAX_DECODE), 'Decode delta exceeds the negotiated maximum')
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
        self._stop_job()
        if self.stuck:
            # Left running: reader threads stay blocked on its pipes and the pipes stay open.
            # Only the Job handle closes (kill-on-close was cleared, so this cannot kill).
            pid = self.job.pid if self.job else None
            self.receipt = dict(session=self.session, host_budget=self.host_bytes, physical_floor=self.floor,
                                peak_rss=self.peak_rss, minimum_available_ram=self.minimum_available, failure=self.failure,
                                owned_pid=pid, owned_job_empty=False, owned_process_signaled=False, owned_pid_gone=False,
                                stuck=self.stuck)
            with self.job_lock:
                if self.job:
                    self.job.close()
                    self.job = None
            raise RuntimeError(self.stuck)
        # A 30 GB VRAM teardown can take >10 s before the pipes close; wait up to 45 s in total.
        deadline = time.monotonic() + 45
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
        require(not any((t.is_alive() for t in self.threads)), 'Persistent producer reader did not stop')
        pid = self.job.pid if self.job else None
        end = time.monotonic() + 5
        while self.job is not None and self.job.poll() is None and (time.monotonic() < end):
            time.sleep(0.01)
        exited = self.job is None or self.job.poll() is not None
        empty = self.job is None or self.job.active() == 0
        self.receipt = dict(session=self.session, host_budget=self.host_bytes, physical_floor=self.floor, runtime_min_free_gib=self.runtime_min_free_gib, peak_rss=self.peak_rss, minimum_available_ram=self.minimum_available, paging_growth=self.paging_growth, budget_breaches=self.budget_breaches, first_budget_breach=self.first_budget_breach, failure=self.failure, owned_pid=pid, owned_job_empty=empty, owned_process_signaled=exited, exit_code=self.job.poll() if self.job is not None else getattr(self, 'exit_code', None))
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
