"""The one-time full model SHA-256 runs at install/check, never silently on the first prompt.

CPU only: a small temp file stands in for the GGUF and MODEL_SHA256 is patched to its digest.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from slstack import check as check_module
from slstack import native_broker
from slstack.native_broker import model_receipt, vendor
from slstack.single_runtime import receipt_config

vendor()
import studio.dsv41_session_contract as contract  # noqa: E402

NAME = 'DeepSeek-V4.1-Flash-1017-Q4-v41.gguf'


class ModelReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.model = root / 'models' / NAME
        self.model.parent.mkdir()
        self.model.write_bytes(os.urandom((3 << 20) + 17))   # several 1 MiB hash blocks
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.cfg = {'model': str(self.model), 'runtime_root': str(root / 'runtime')}
        self.target = root.resolve() / 'runtime' / 'single-engine' / 'state' / 'original-model.json'
        patch = mock.patch.object(contract, 'MODEL_SHA256', self.digest)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.temp.cleanup)

    def test_check_path_writes_the_serving_receipt_with_progress(self):
        lines = []
        with mock.patch.object(native_broker.time, 'monotonic', side_effect=(float(i) for i in range(10 ** 6))), \
                mock.patch('builtins.print', lambda *a, **k: lines.append(' '.join(map(str, a)))):
            result = check_module.verify_model_receipt(self.cfg)
        self.assertEqual(result, {'ready': True, 'receipt': str(self.target)})
        data = json.loads(self.target.read_text(encoding='utf-8'))
        stat = self.model.stat()
        self.assertEqual(data, {'receipts': {NAME: {'sha256': self.digest, 'final_identity': {
            'size': stat.st_size, 'dev': stat.st_dev, 'ino': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}}},
            'validation': 'full SHA-256'})
        progress = [line for line in lines if '%' in line]
        self.assertTrue(progress and progress[-1].startswith('Verifying original model SHA-256: 100.0%'), lines)
        self.assertIn('GB/s', progress[-1])
        self.assertIn('of 0.0 GB', progress[-1])

    def test_serving_reuses_the_cached_receipt_without_rehashing(self):
        check_module.verify_model_receipt(self.cfg)
        with mock.patch.object(native_broker, 'sha256', side_effect=AssertionError('rehashed')):
            self.assertEqual(model_receipt(receipt_config(self.cfg)), self.target)

    def test_not_applicable_on_the_paired_mac_or_native_path(self):
        self.assertIsNone(check_module.verify_model_receipt({**self.cfg, 'mac': 'http://mac:1'}))
        self.assertIsNone(check_module.verify_model_receipt({**self.cfg, 'serving': 'native'}))
        self.assertFalse(self.target.exists())

    def test_wrong_model_fails_the_check_row(self):
        with mock.patch.object(contract, 'MODEL_SHA256', '0' * 64), mock.patch('builtins.print'):
            result = check_module.verify_model_receipt(self.cfg)
        self.assertFalse(result['ready'])
        self.assertIn('native handoff refused', result['error'])
        self.assertFalse(self.target.exists())

    def test_missing_receipt_at_serve_time_is_a_blocking_gate_with_status_progress(self):
        # The producer refuses to start without the receipt, so serving hashes first,
        # blocking, and the progress goes to the model status the router reports.
        from slstack.engines import Engines
        engines = Engines.__new__(Engines)
        engines.cfg = {'lora': [], 'runtime_root': self.cfg['runtime_root']}
        engines.events, engines.selected_model = [], 'm'
        single = {**receipt_config(self.cfg), 'env_overrides': {}}
        order = []

        class Launched(Exception):
            pass

        def launch(*args, **kwargs):
            order.append(('launch', self.target.is_file()))
            raise Launched

        with mock.patch('slstack.single_runtime.stack_config', return_value=single), \
                mock.patch('slstack.engines.OwnedProcess', side_effect=launch), mock.patch('builtins.print'):
            with self.assertRaises(Launched):
                engines._ready_single()
        self.assertEqual(order, [('launch', True)])
        messages = [row['message'] for row in engines.events]
        self.assertTrue(messages[0].startswith('Verifying original model SHA-256 (read only'), messages)
        self.assertTrue(messages[-1].startswith('Verifying original model SHA-256: 100.0%'), messages)

        # Second start: the cached receipt is reused, no hash and no status noise.
        engines.events = []
        with mock.patch('slstack.single_runtime.stack_config', return_value=single), \
                mock.patch('slstack.engines.OwnedProcess', side_effect=launch), \
                mock.patch.object(native_broker, 'sha256', side_effect=AssertionError('rehashed')):
            with self.assertRaises(Launched):
                engines._ready_single()
        self.assertEqual(engines.events, [])


if __name__ == '__main__':
    unittest.main()
