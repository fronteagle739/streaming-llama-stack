"""LLAMADECODE producer abort: a died, aborted or truncated persistent producer yields an error
naming its exit code and its LLAMADECODE_PRODUCER_ABORT reason, never a bare 'truncated'."""
import io
import json
import threading
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import psutil
from slstack.native_broker import vendor

vendor()
from studio.dsv41_session_contract import MODEL_SHA256, Refused
from studio.dsv41_session_process import Producer

SESSION = '6b646ad77b2ceeccb0b803a13443ac11'


class FakeJob:
    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code


def producer(stdout, stderr, code):
    p = Producer.__new__(Producer)
    p.output = io.BytesIO(stdout)
    p.job = FakeJob(code)
    p.diagnostic = bytearray(stderr)
    p.state_lock = threading.Lock()
    p.threads = []
    p.session, p.owner_epoch, p.failure = SESSION, 7, None
    p.death_wait_seconds = 0.05
    return p


ABORT = (b'DS4PERF_PAIRED_FFN_COMPLETE layer=28\n'
         b'LLAMADECODE_PRODUCER_ABORT kind=bad_alloc code=0x00000000 sequence=0 reason=std::bad_alloc: a host allocation failed\n'
         b'Offline producer failed: std::bad_alloc. Existing/unfinished files were preserved.\n')


class ProducerAbort(unittest.TestCase):
    def receive(self, p):
        with self.assertRaises(Refused) as caught:
            p._receive()
        return str(caught.exception)

    def test_died_producer_names_exit_code_and_abort_reason(self):
        text = self.receive(producer(b'', ABORT, 1))
        self.assertIn('closed its reply pipe', text)
        self.assertIn('exit code 1 (0x00000001)', text)
        self.assertIn('LLAMADECODE_PRODUCER_ABORT kind=bad_alloc', text)
        self.assertIn('host allocation failed', text)

    def test_truncated_reply_names_exit_code_and_stderr_tail(self):
        text = self.receive(producer(b'{"status":"rea', b'DS4PERF_HC_BF16_EPILOGUE layer=28 start=36864\n', 0xC0000005))
        self.assertIn('truncated', text)
        self.assertIn('exit code 3221225477 (0xC0000005)', text)
        self.assertIn('no LLAMADECODE_PRODUCER_ABORT line; stderr tail: DS4PERF_HC_BF16_EPILOGUE layer=28', text)

    def test_error_reply_carries_reason(self):
        row = dict(status='error', session=SESSION, model_sha256=MODEL_SHA256, owner_epoch=7, sequence=3, error_kind='exception',
                   error_code=0, reason='LLAMADECODE_TEST_FAULT main_throw at layer 4 on the sweep thread (test-only injected fault)')
        text = self.receive(producer((json.dumps(row) + '\n').encode(), b'LLAMADECODE_PRODUCER_ABORT kind=exception reason=x\n', 1))
        self.assertIn('Persistent producer aborted (exception, sequence 3): LLAMADECODE_TEST_FAULT main_throw at layer 4', text)
        self.assertIn('exit code 1', text)

    def test_still_running_is_said(self):
        text = self.receive(producer(b'', b'', None))
        self.assertIn('still running', text)
        self.assertNotIn('producer exit code', text)

    def test_complete_reply_unchanged(self):
        row = dict(status='complete', session=SESSION, model_sha256=MODEL_SHA256, owner_epoch=7, sequence=1)
        self.assertEqual(producer((json.dumps(row) + '\n').encode(), b'', None)._receive(), row)

    def watch(self, floor=4.0, available=3<<30, paged=0, rss=None):
        p=producer(b'{"status":"rea',b'',None)
        p.job_lock=threading.Lock();p.stuck=None
        p.job.pid=123
        stops=[]
        p.job.stop=lambda **kwargs:stops.append(kwargs)
        p.runtime_min_free_gib=floor;p.runtime_floor=int(floor*(1<<30))
        p.stop_event=SimpleNamespace(wait=Mock(side_effect=[False,True]))
        p.cancelled=lambda:False;p.stall_seconds=None
        p.peak_rss=0;p.minimum_available=10<<30
        p.floor=1<<30;p.host_bytes=8<<30;p.budget_breaches=0;p.first_budget_breach=None
        p._paging_tick=9;p.paging_growth=0;p._paging0=0;p._paging_limit=256000000;p._paging_bytes=lambda:paged
        p.deadline=None;p.activity=float('inf');p.idle_seconds=1;p.ready=None
        root=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss),children=lambda **kwargs:[],is_running=lambda:True)
        lookup=(patch('studio.dsv41_session_process.psutil.Process',return_value=root) if rss is not None else
                patch('studio.dsv41_session_process.psutil.Process',side_effect=psutil.NoSuchProcess(123)))
        with patch('studio.dsv41_session_process.psutil.virtual_memory',return_value=SimpleNamespace(available=available)),lookup:
            p._watch()
        return p,stops

    def test_low_available_ram_stops_gracefully_and_client_gets_reason(self):
        with self.assertLogs('studio.dsv41_session_process',level='CRITICAL') as logs:
            p,stops=self.watch()
        self.assertIn('available physical RAM',logs.output[0])
        self.assertEqual(stops,[dict(graceful_seconds=30,stuck_seconds=120)])
        text=self.receive(p)
        self.assertIn('runtime_min_free_gib=4',text)
        self.assertIn(str(3<<30),text)
        self.assertNotIn('truncated',text)

    def test_zero_disables_free_ram_stop(self):
        p,stops=self.watch(floor=0,available=1)
        self.assertIsNone(p.failure);self.assertEqual(stops,[])

    def test_exact_floor_is_allowed(self):
        p,stops=self.watch(available=4<<30)
        self.assertIsNone(p.failure);self.assertEqual(stops,[])

    def test_paging_stop_stays_enabled_when_ram_floor_disabled(self):
        p,stops=self.watch(floor=0,paged=256000001)
        self.assertIn('paging detected',p.failure)
        self.assertEqual(stops,[dict(graceful_seconds=30,stuck_seconds=120)])

    def test_host_rss_budget_trip_requests_the_same_cooperative_retirement(self):
        p,stops=self.watch(floor=0,available=10<<30,rss=(8<<30)+1)
        self.assertIn('host RSS budget exceeded',p.failure)
        self.assertEqual(stops,[dict(graceful_seconds=30,stuck_seconds=120)])

    def test_abort_file_is_published_before_the_owned_runner_stop(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            p=Producer.__new__(Producer);p.job_lock=threading.Lock();p.stuck=None
            p.abort_file=str(Path(directory)/'guard.abort');p.job=Mock()
            p.job.stop.side_effect=lambda **kwargs:self.assertTrue(Path(p.abort_file).exists())
            p._stop_job()
            p.job.stop.assert_called_once_with(graceful_seconds=30,stuck_seconds=120)


if __name__ == '__main__':
    unittest.main()
