"""Graceful native/producer shutdown verdicts (CPU only, fakes, no engine)."""
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
import tempfile

from slstack.combo_native import Native
from slstack.combo_producer import ComboProducer

PROOF = dict(owned_job_empty=True, owned_process_signaled=True, owned_pid_gone=True)


class Job:
    def __init__(self, code):
        self.code, self.polls = code, 0
    def poll(self):
        # Alive until the owner has asked for the shutdown, then exited.
        return self.code if self.polls else None


def native(tmp, code, diagnostic, *, graceful=True, fail=False):
    n = Native.__new__(Native)
    n.closed, n.restore_lease, n.leases, n.directory = False, None, ExitStack(), Path(tmp)
    n.cfg = dict(native_graceful_shutdown=graceful, native_graceful_shutdown_seconds=1)
    job = Job(code)
    n.process = type('P', (), dict(job=job, diagnostic=diagnostic, close=lambda self: dict(PROOF)))()
    def control(op):
        assert op == 'shutdown'
        if fail: raise RuntimeError('Native HTTP 409')
        job.polls += 1
    n.control = control
    return n


def producer(tmp, code, diagnostic, *, graceful=True, busy=False):
    p = ComboProducer.__new__(ComboProducer)
    p.closed, p.leases, p.directory, p.creation_attempted = False, ExitStack(), Path(tmp), True
    p.cfg = dict(producer_graceful_shutdown=graceful, producer_graceful_shutdown_seconds=1)
    p.shutdown_path, p.shutdown_seconds, p.shutdown_detail, p._exit_code = 'terminate', 0.0, None, None
    p.session = 'a' * 32
    job = Job(code)
    lock = threading.Lock()
    if busy: lock.acquire()
    sent = []
    class Proc:
        closed = False; failure = None; sequence = 7
        def __init__(self): self.job, self.lock, self.diagnostic = job, lock, diagnostic
        def request(self, row): sent.append(row['operation']); job.polls += 1; return dict(status='complete')
        def close(self): return dict(PROOF)
    p.process = Proc()
    return p, sent


DONE_N = b'x\nDS4COMBO_NATIVE_GRACEFUL_SHUTDOWN_COMPLETE engine_ms=1.0 ds_ms=2.0 engine_and_ds_ms=3.0 ds_shutdown=1\n'
DONE_P = b'DS4COMBO_PRODUCER_GRACEFUL_SHUTDOWN_COMPLETE sweep_release_ms=4.0\n'


class NativeGraceful(unittest.TestCase):
    def test_clean_exit_with_marker_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = native(tmp, 0, DONE_N).close()
        self.assertEqual(proof['native_shutdown_path'], 'graceful')
        self.assertFalse(proof['kill_path'])
        self.assertIn('ds_ms=2.0', proof['native_shutdown_detail'])

    def test_exit_without_marker_is_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = native(tmp, 0, b'abort\n').close()
        self.assertEqual(proof['native_shutdown_path'], 'graceful-crashed exit=0')

    def test_nonzero_exit_with_marker_is_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = native(tmp, 3221225477, DONE_N).close()
        self.assertEqual(proof['native_shutdown_path'], 'graceful-crashed exit=3221225477')

    def test_terminate_code_is_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = native(tmp, 125, DONE_N).close()
        self.assertEqual(proof['native_shutdown_path'], 'graceful-crashed exit=125')

    def test_refused_op_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = native(tmp, 0, b'', fail=True).close()
        self.assertTrue(proof['native_shutdown_path'].startswith('graceful-failed-terminate'))
        self.assertTrue(proof['kill_path'])  # still alive when terminated: counted as KILL PATH

class ProducerGraceful(unittest.TestCase):
    def test_close_op_and_marker_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sent = producer(tmp, 0, DONE_P)
            proof = p.close()
        self.assertEqual(sent, ['close'])
        self.assertEqual(proof['producer_shutdown_path'], 'graceful')
        self.assertFalse(proof['kill_path'])
        self.assertIn('sweep_release_ms=4.0', proof['producer_shutdown_detail'])

    def test_crash_in_teardown_is_not_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _ = producer(tmp, 1, DONE_P)
            proof = p.close()
        self.assertEqual(proof['producer_shutdown_path'], 'graceful-crashed exit=1')

    def test_busy_producer_is_terminated_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sent = producer(tmp, 0, DONE_P, busy=True)
            proof = p.close()
        self.assertEqual(sent, [])
        self.assertTrue(proof['producer_shutdown_path'].startswith('terminate:'))
        self.assertTrue(proof['kill_path'])  # a busy, live producer is terminated: KILL PATH


if __name__ == '__main__':
    unittest.main()
