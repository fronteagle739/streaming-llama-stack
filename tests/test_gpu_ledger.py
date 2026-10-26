from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import psutil
from slstack.guard import supervise


class MediaLedgerTests(unittest.TestCase):
    def test_default_and_explicit_ledger_are_identical_for_all_guard_stages(self):
        root=Path(tempfile.mkdtemp(prefix='slstack-ledger-proof-')).resolve()
        for configured in ('',None,str(root/'explicit.holds')):
            expected=Path(configured or root/'state'/'gpu.holds').resolve()
            gpu=dict(runtime_root=str(root),hold_file=configured,lease_minutes=1)
            spec=dict(gpu=gpu,runtime_root=str(root),owner_pid=os.getpid(),timeout_s=10)
            @contextmanager
            def lease(cfg):
                self.assertEqual(cfg['hold_file'],configured)
                yield 'proof-identity'
            def child(actual):
                self.assertEqual(Path(actual['gpu']['hold_file']),expected)
                self.assertEqual(Path(actual['hold_watch']['path']),expected)
                self.assertEqual(actual['gpu_lease_identity'],'proof-identity')
                self.assertNotEqual(expected,Path('.').resolve())
                return 0
            with self.subTest(configured=configured),patch('slstack.lease.gpu_lease',lease),patch('slstack.guard._supervise',child),patch('slstack.lease.recover_leases'):
                self.assertEqual(supervise(spec),0)


if __name__=='__main__':unittest.main()
