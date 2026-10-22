import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from slstack import train_native_prefix as workflow


class NativeTrainingLifecycleTests(unittest.TestCase):
    def request(self):
        return {'cfg': {'model': 'base.gguf', 'lease_minutes': 19},
                'settings': {'expected_base_sha256': 'a' * 64}, 'output': 'adapter'}

    def phase(self):
        return SimpleNamespace(wait=Mock(return_value={'exit_code': 0}), stop=Mock(),
            log=SimpleNamespace(read_text=lambda **kw: 'SLSTACK_TRAIN_PHASE ' + json.dumps({'training_seconds': 3})))

    @patch.object(workflow, 'atomic_json')
    @patch.object(workflow, 'sha256', side_effect=['a' * 64, 'a' * 64])
    @patch.object(workflow, 'OwnedProcess')
    def test_gpu_lease_covers_phase_and_hashes_cover_workflow(self, owned, digest, save):
        phase = self.phase(); owned.return_value = phase
        result = workflow.run(self.request())
        self.assertTrue(result['base_unchanged'])
        self.assertEqual(digest.call_count, 2)
        self.assertTrue(owned.call_args.kwargs['gpu'])
        self.assertEqual(owned.call_args.kwargs['timeout'], 1140)
        sent = json.loads(owned.call_args.kwargs['stdin'])
        self.assertEqual(sent['phase'], 'gpu')
        self.assertEqual(sent['base_sha256_before'], 'a' * 64)
        phase.stop.assert_called_once()

    @patch.object(workflow, 'sha256', side_effect=['a' * 64, 'a' * 64])
    @patch.object(workflow, 'OwnedProcess')
    def test_failed_phase_still_stops_and_checks_base(self, owned, digest):
        phase = self.phase(); phase.wait.side_effect = RuntimeError('phase failed'); owned.return_value = phase
        with self.assertRaisesRegex(RuntimeError, 'phase failed'):
            workflow.run(self.request())
        self.assertEqual(digest.call_count, 2)
        phase.stop.assert_called_once()

    @patch.object(workflow, 'sha256', side_effect=['a' * 64, 'b' * 64])
    @patch.object(workflow, 'OwnedProcess')
    def test_base_mutation_is_refused(self, owned, digest):
        owned.return_value = self.phase()
        with self.assertRaisesRegex(RuntimeError, 'BASE HASH MISMATCH'):
            workflow.run(self.request())
