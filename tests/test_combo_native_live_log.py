"""Live native stderr.log mirror (CPU only, fakes, no engine)."""
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path

from slstack.combo_native import LiveLog, Native

ROOT = Path(__file__).resolve().parents[1]


class Process:
    def __init__(self):
        self.diagnostic, self.state_lock = bytearray(), threading.Lock()


class Owner:
    process = None


def wait_for(file, expected, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(file).read_bytes() == expected:
            return True
        time.sleep(0.02)
    return False


class LiveLogTests(unittest.TestCase):
    def test_mirrors_growth_before_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'stderr.log'
            owner = Owner()
            log = LiveLog(owner, file, interval=0.02)
            try:
                time.sleep(0.1)  # no process yet: nothing written, no failure
                self.assertEqual(file.read_bytes(), b'')
                owner.process = Process()
                owner.process.diagnostic.extend(b'DS4COMBO_T1 a\n')
                self.assertTrue(wait_for(file, b'DS4COMBO_T1 a\n'))
                owner.process.diagnostic.extend(b'DS4COMBO_EXPERT_RESTORE_PENDING vram=1 host=2\n')
                self.assertTrue(wait_for(file, b'DS4COMBO_T1 a\nDS4COMBO_EXPERT_RESTORE_PENDING vram=1 host=2\n'))
                self.assertTrue(log.thread.is_alive())
            finally:
                log.stop()
            self.assertFalse(log.thread.is_alive())

    def test_hard_killed_owner_keeps_log(self):
        # The reaper kills the harness process outright; the bytes mirrored
        # before the kill must already be on disk.
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'stderr.log'
            child = textwrap.dedent(f'''
                import sys, threading, time
                sys.path.insert(0, {str(ROOT)!r})
                from slstack.combo_native import LiveLog
                class P: pass
                class O: process = None
                o = O(); p = P(); p.diagnostic = bytearray(); p.state_lock = threading.Lock()
                LiveLog(o, {str(file)!r}, interval=0.02)
                o.process = p
                p.diagnostic.extend(b'before-kill\\n')
                print('ready', flush=True)
                time.sleep(60)
            ''')
            proc = subprocess.Popen([sys.executable, '-c', child], stdout=subprocess.PIPE)
            try:
                self.assertEqual(proc.stdout.readline().strip(), b'ready')
                self.assertTrue(wait_for(file, b'before-kill\n'))
            finally:
                proc.kill()
                proc.wait(10)
                proc.stdout.close()
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(file.read_bytes(), b'before-kill\n')

    def test_close_stops_mirror_and_writes_full_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = Native.__new__(Native)
            n.closed, n.restore_lease, n.leases, n.directory = False, None, ExitStack(), Path(tmp)
            n.cfg = dict(native_graceful_shutdown_seconds=1)
            proof = dict(owned_job_empty=True, owned_process_signaled=True, owned_pid_gone=True)
            job = type('J', (), dict(poll=lambda self: 0))()
            n.process = type('P', (), dict(job=job, close=lambda self: dict(proof)))()
            n.process.diagnostic = bytearray(b'one\n')
            n.process.state_lock = threading.Lock()
            n.live_log = LiveLog(n, Path(tmp) / 'stderr.log', interval=0.02)
            self.assertTrue(wait_for(Path(tmp) / 'stderr.log', b'one\n'))
            n.process.diagnostic.extend(b'two\n')
            n.close()
            self.assertFalse(n.live_log.thread.is_alive())
            self.assertEqual((Path(tmp) / 'stderr.log').read_bytes(), b'one\ntwo\n')


if __name__ == '__main__':
    unittest.main()
