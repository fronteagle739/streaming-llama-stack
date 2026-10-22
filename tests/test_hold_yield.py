import json
import os
from pathlib import Path
import sys
import time
import unittest
import uuid

from slstack.config import load
from slstack.lease import HoldWatcher
from slstack.process import OwnedProcess


class HoldYieldTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(overrides={'gpu_uuid': 'GPU-test', 'host_budget_gib': .25, 'min_available_gib': 4})
        if os.environ.get('STUDIO_RUNTIME_ROOT'):
            self.cfg['runtime_root'] = os.environ['STUDIO_RUNTIME_ROOT']
        self.directory = Path(self.cfg['runtime_root']) / ('hold-test-' + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.hold = self.directory / 'holds.txt'
        self.hold.write_text('ACTIVE: DS4STACK own; test\n', encoding='utf-8')

    def tearDown(self):
        # Only flat, verified test files. No recursive cleanup operation.
        for file in self.directory.iterdir():
            self.assertTrue(file.is_file())
            file.unlink()
        self.directory.rmdir()

    def test_rtx_and_mac_namespaces_and_release(self):
        rtx, mac = HoldWatcher(self.hold, 'own'), HoldWatcher(self.hold, 'own', mac=True)
        self.assertIsNone(rtx.reason())
        self.hold.write_text('ACTIVE: DS4STACK own; test\nMAC ACTIVE: KIT11 mac; priority\n', encoding='utf-8')
        self.assertIsNone(rtx.reason())
        self.assertIn('KIT11', mac.reason())
        with self.hold.open('a', encoding='utf-8') as out:
            out.write('ACTIVE: DS4COMBO rtx; priority\n')
        self.assertIn('DS4COMBO', rtx.reason())
        with self.hold.open('a', encoding='utf-8') as out:
            out.write('RELEASED: DS4COMBO rtx; done\nMAC RELEASED: KIT11 mac; done\n')
        self.assertIsNone(rtx.reason())
        self.assertIsNone(mac.reason())

    def test_real_owned_child_stops_on_kit11_hold(self):
        ready = self.directory / 'ready'
        cfg = {**self.cfg, 'request_timeout_s': 20,
               'hold_watch': {'path': str(self.hold), 'identity': 'own'}}
        child = 'from pathlib import Path; import sys,time; Path(sys.argv[1]).write_text("ready"); time.sleep(15)'
        job = OwnedProcess(cfg, [sys.executable, '-B', '-c', child, str(ready)], 'hold-yield-test')
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and job.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(ready.exists())
            with self.hold.open('a', encoding='utf-8') as out:
                out.write('ACTIVE: KIT11 priority; test\n')
            with self.assertRaises(RuntimeError):
                job.wait()
            receipt = json.loads(job.receipt.read_text())
            self.assertIn('KIT11', receipt['reason'])
            self.assertLess(receipt['elapsed_s'], 10)
        finally:
            if job.proc.poll() is None:
                job.stop()


if __name__ == '__main__':
    unittest.main()
