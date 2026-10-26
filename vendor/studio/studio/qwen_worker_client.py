# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import hashlib
import queue
import threading
import time
import uuid
from pathlib import Path
from collections import deque
from .jobs import managed_popen, kill_process_tree
from .progress import parse_line

class QwenWorker:

    def __init__(self, argv, env, *, log_path=None, completion_prefix='QWEN_JOB_DONE ', event_prefix=None):
        self.completion_prefix = completion_prefix
        self.event_prefix = event_prefix
        self.argv = list(argv)
        self.gpu_environment = {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER') if k in env}
        self.log_lock = threading.Lock()
        self.log_handle = None
        self.log_path = None
        self.log_base = None
        self.request_started = False
        if log_path:
            self._set_log(log_path)
        try:
            self.proc = managed_popen(argv, env_extra=env, isolate_python=True, stdin_pipe=True)
        except BaseException as exc:
            self._log('WORKER_START_ERROR ' + repr(exc))
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None
            raise
        self.lines = queue.Queue()
        self.lock = threading.Lock()

        def read():
            try:
                for line in self.proc.stdout:
                    self._log(line.rstrip())
                    self.lines.put(line.rstrip())
            finally:
                self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def _set_log(self, path, *, new_attempt=False):
        path = Path(path)
        with self.log_lock:
            if self.log_base == str(path) and (not new_attempt):
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            unique = path.with_name(f'{path.stem}.{uuid.uuid4().hex}{path.suffix}')
            handle = unique.open('x', encoding='utf-8')
            if self.log_handle:
                self.log_handle.close()
            self.log_handle, self.log_path, self.log_base = (handle, str(unique), str(path))
            handle.write(json.dumps({'time': time.time(), 'worker_argv': self.argv, 'gpu_environment': self.gpu_environment}) + '\n')
            handle.flush()

    def _log(self, line):
        with self.log_lock:
            if self.log_handle:
                self.log_handle.write(json.dumps({'time': time.time(), 'line': line}) + '\n')
                self.log_handle.flush()

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        if self.alive():
            kill_process_tree(self.proc)
        try:
            self.proc.wait(timeout=15)
        except Exception:
            pass
        if self.proc.stdin:
            self.proc.stdin.close()
        self.reader.join(timeout=5)
        self._log('WORKER_EXIT ' + str(self.proc.poll()))
        with self.log_lock:
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None

    def generate(self, request, on_step=None, should_cancel=None):
        with self.lock:
            if request.get('output_path'):
                self._set_log(Path(request['output_path']).with_suffix('.worker.jsonl'), new_attempt=self.request_started)
            self.request_started = True
            request_bytes = json.dumps(request, sort_keys=True).encode()
            request_snapshot = Path(self.log_path).with_suffix('.request.json') if self.log_path else None
            if request_snapshot:
                with request_snapshot.open('xb') as handle:
                    handle.write(request_bytes)
            self._log('REQUEST ' + json.dumps({'sha256': hashlib.sha256(request_bytes).hexdigest(), 'worker_pid': self.proc.pid, 'width': request.get('width'), 'height': request.get('height'), 'use_kv_cache': request.get('use_kv_cache', True), 'request_path': str(request_snapshot) if request_snapshot else None}))
            tail = deque(maxlen=20)
            try:
                if should_cancel and should_cancel():
                    raise RuntimeError('generation cancelled')
                self.proc.stdin.write(json.dumps(request) + '\n')
                self.proc.stdin.flush()
                while True:
                    if should_cancel and should_cancel():
                        raise RuntimeError('generation cancelled')
                    try:
                        line = self.lines.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if line is None:
                        raise RuntimeError('Qwen worker exited: ' + '\n'.join(tail))
                    tail.append(line)
                    if line.startswith(self.completion_prefix):
                        result = json.loads(line[len(self.completion_prefix):])
                        if not result.get('ok'):
                            raise RuntimeError(result.get('error') or 'Qwen generation failed')
                        result['worker_log'] = self.log_path
                        return result
                    if self.event_prefix and line.startswith(self.event_prefix):
                        event = json.loads(line[len(self.event_prefix):])
                        if on_step:
                            on_step(event['step'], event['total'], event.get('preview'), event.get('meta'))
                        continue
                    event = parse_line(line, 'qwen_image21')
                    if event and on_step:
                        on_step(event.get('step', 0), event.get('total', 0), None, event)
            except BaseException as exc:
                self._log('REQUEST_ERROR ' + repr(exc))
                self.close()
                if isinstance(exc, RuntimeError) and self.log_path:
                    raise RuntimeError(f'{exc}; Qwen worker log: {self.log_path}') from exc
                raise
