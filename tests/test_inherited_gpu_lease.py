import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch
import uuid

from slstack.config import load
from slstack.lease import AdmissionRefused, active_holds, gpu_lease
from slstack.pairing import Pair
from slstack.process import OwnedProcess


class InheritedLeaseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(overrides={'gpu_uuid': 'GPU-test', 'host_budget_gib': .5,
                                   'min_available_gib': 2, 'lease_minutes': 1})
        root = Path(os.environ.get('STUDIO_RUNTIME_ROOT', self.cfg['runtime_root']))
        self.directory = root / ('inherited-lease-test-' + uuid.uuid4().hex)
        self.directory.mkdir()
        self.hold = self.directory / 'holds.txt'
        self.hold.write_text('', encoding='utf-8')
        self.cfg.update(runtime_root=str(self.directory), hold_file=str(self.hold))
        self.probe = patch('slstack.lease.subprocess.run',
                           return_value=Mock(returncode=0, stdout='0, 32000, 0'))

    def reference(self, identity):
        return self.directory / 'state/leases' / (identity + '.json')

    def test_real_descendant_inherits_without_acquiring_or_releasing(self):
        with self.probe, gpu_lease(self.cfg) as identity:
            record = self.reference(identity)
            cfg = {**self.cfg, '_gpu_parent_lease': str(record), 'host_budget_gib': .25}
            config = self.directory / 'child.json'
            config.write_text(json.dumps(cfg))
            code = ('import json,sys; from slstack.lease import gpu_lease; '
                    'cfg=json.load(open(sys.argv[1])); '
                    'lease=gpu_lease(cfg); print(lease.__enter__(),flush=True); '
                    'lease.__exit__(None,None,None)')
            job = OwnedProcess(cfg, [sys.executable, '-B', '-c', code, str(config)],
                               'inherited-lease-child', timeout=10)
            try:
                self.assertEqual(job.wait()['exit_code'], 0)
                self.assertEqual(job.log.read_text().strip(), identity)
            finally:
                if job.proc.poll() is None:
                    job.stop()
            self.assertEqual(len(active_holds(self.hold.read_text())), 1)
            self.assertFalse(json.loads(record.read_text())['released'])
        self.assertEqual(active_holds(self.hold.read_text()), [])

    def test_invalid_owner_budget_and_lifetime_are_refused(self):
        with self.probe, gpu_lease(self.cfg) as identity:
            record = self.reference(identity)
            original = json.loads(record.read_text())
            cfg = {**self.cfg, '_gpu_parent_lease': str(record)}
            variants = [dict(guardian_pid=-1),
                        dict(guardian_created=original['guardian_created'] - 1),
                        dict(released=True), dict(cleanup_uncertain=True),
                        dict(host_budget_gib=.1), dict(min_available_gib=3)]
            for change in variants:
                with self.subTest(change=change):
                    record.write_text(json.dumps({**original, **change}))
                    with self.assertRaises(AdmissionRefused), gpu_lease(cfg):
                        self.fail('Invalid reservation admitted')
            record.write_text(json.dumps(original))

    def test_foreign_hold_or_missing_parent_hold_is_refused(self):
        with self.probe, gpu_lease(self.cfg) as identity:
            cfg = {**self.cfg, '_gpu_parent_lease': str(self.reference(identity))}
            original = self.hold.read_text()
            for ledger in ('', original + '\nACTIVE: KIT11 priority; GPU\n'):
                with self.subTest(ledger=ledger):
                    self.hold.write_text(ledger)
                    with self.assertRaises(AdmissionRefused), gpu_lease(cfg):
                        self.fail('Conflicting reservation admitted')
            self.hold.write_text(original)

    def test_live_ancestor_can_finish_after_estimated_expiry(self):
        with self.probe, gpu_lease(self.cfg) as identity:
            record = self.reference(identity)
            row = json.loads(record.read_text())
            record.write_text(json.dumps({**row, 'expires': '2000-01-01T00:00:00+00:00'}))
            with gpu_lease({**self.cfg, '_gpu_parent_lease': str(record)}) as child:
                self.assertEqual(child, identity)

    def test_pair_keeps_reservation_when_broker_cleanup_is_uncertain(self):
        pair = Pair.__new__(Pair)
        pair.connected = True
        pair.request = Mock(return_value={'ejected': True})
        pair.broker = Mock()
        pair.broker.receipt = self.directory / 'missing-receipt.json'
        pair.lease_context = MagicMock()
        context = pair.lease_context
        with self.assertRaisesRegex(RuntimeError, 'cleanup unconfirmed'):
            pair.close()
        context.__exit__.assert_called_once()
        self.assertTrue(context.__exit__.call_args.args[1].owned_cleanup_uncertain)
        self.assertFalse(pair.connected)


if __name__ == '__main__':
    unittest.main()
