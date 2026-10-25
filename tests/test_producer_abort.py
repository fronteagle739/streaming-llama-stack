"""LLAMADECODE producer abort: a died, aborted or truncated persistent producer yields an error
naming its exit code and its LLAMADECODE_PRODUCER_ABORT reason, never a bare 'truncated'."""
import io
import json
import threading
import unittest
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


if __name__ == '__main__':
    unittest.main()
