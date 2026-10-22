from contextlib import contextmanager
from pathlib import Path
import os
import psutil
import unittest
from unittest.mock import patch, Mock
from slstack.guard import supervise
from slstack.lease import AdmissionRefused


class AdmissionWaitTests(unittest.TestCase):
    def spec(self):
        return {'gpu': {'gpu_wait_seconds': 30, 'lease_minutes': 19},
                'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time(),
                'stop': 'missing-stop-file', 'runtime_root': 'runtime',
                'min_available_gib': .01, 'timeout_s': 2000}

    @patch('slstack.lease.recover_leases')
    @patch('slstack.guard._supervise', return_value=0)
    @patch('slstack.guard.time.sleep')
    def test_waits_without_starting_then_runs_once_under_lease(self, sleep, run, recover):
        calls = []
        @contextmanager
        def lease(cfg):
            calls.append('attempt')
            if len(calls) < 3:
                raise AdmissionRefused('busy')
            yield 'owned'
        with patch('slstack.lease.gpu_lease', lease):
            self.assertEqual(supervise(self.spec()), 0)
        self.assertEqual(len(calls), 3)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0]['timeout_s'], 1140)

    @patch('slstack.lease.recover_leases')
    @patch('slstack.guard._supervise', side_effect=PermissionError('engine failed'))
    def test_failure_after_admission_never_retries_engine(self, run, recover):
        @contextmanager
        def lease(cfg):
            yield 'owned'
        with patch('slstack.lease.gpu_lease', lease), self.assertRaisesRegex(PermissionError, 'engine failed'):
            supervise(self.spec())
        run.assert_called_once()

    @patch('slstack.lease.recover_leases')
    @patch('slstack.guard._supervise')
    def test_default_refuses_immediately_without_engine(self, run, recover):
        @contextmanager
        def lease(cfg):
            raise AdmissionRefused('busy')
            yield
        spec = self.spec(); spec['gpu']['gpu_wait_seconds'] = 0
        with patch('slstack.lease.gpu_lease', lease), self.assertRaises(AdmissionRefused):
            supervise(spec)
        run.assert_not_called()
