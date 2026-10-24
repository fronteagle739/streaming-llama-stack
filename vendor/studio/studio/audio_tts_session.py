# Derived from Studio backend/studio/audio_tts_session.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""One lazily loaded resident TTS child; release only for eviction or shutdown."""
from __future__ import annotations

import atexit
import json
import queue
import subprocess
import threading


class TtsSession:
    def __init__(self, spawn):
        self.spawn = spawn
        self.lock = threading.RLock()
        self.proc = None
        self.key = None
        self.responses = None
        atexit.register(self.close)

    def _stop(self):
        proc, self.proc = self.proc, None
        self.key = None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=2)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)

    def close(self):
        with self.lock:
            self._stop()

    @staticmethod
    def _read(proc, responses):
        try:
            for line in proc.stdout:
                if line.startswith(('RESULT ', 'ERROR ')):
                    responses.put(line)
        finally:
            responses.put(None)

    def call(self, key, args, timeout):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None or key != self.key:
                self._stop()
                self.proc = self.spawn(key)
                self.key = key
                self.responses = queue.Queue()
                threading.Thread(target=self._read, args=(self.proc, self.responses), daemon=True).start()
            try:
                self.proc.stdin.write(json.dumps({'args': args}) + '\n')
                self.proc.stdin.flush()
                line = self.responses.get(timeout=timeout)
                if line is None:
                    raise RuntimeError('Speech worker exited before returning audio.')
                kind, data = line.split(' ', 1)
                result = json.loads(data)
                if kind == 'ERROR':
                    raise RuntimeError(result.get('error', 'Speech worker failed.'))
            except queue.Empty:
                self._stop()
                raise TimeoutError('Speech synthesis timed out.') from None
            except Exception:
                self._stop()
                raise
            return result
