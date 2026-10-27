"""A GLM graph must not enter the DSV4.1 single-engine state/receipt path."""
from pathlib import Path
import unittest
from unittest.mock import patch

from slstack import single_runtime as runtime
from slstack.check import verify_model_receipt
from slstack.native_broker import vendor

vendor()


class GlmAdmissionTests(unittest.TestCase):
    def test_glm_refused_before_binary_hash_receipt_or_preset(self):
        cfg = {'model': 'fixture.gguf', 'host_budget_gib': 3}
        with patch.object(Path, 'is_file', return_value=True), patch(
                'studio.gguf_meta._read_kv_header', return_value={'general.architecture': 'glm5-next'}), patch(
                'slstack.journal.sha256', side_effect=AssertionError('must not hash the wrong model')):
            for action in (lambda: runtime.stack_config(cfg), lambda: runtime.receipt_config(cfg),
                           lambda: runtime.preset(cfg, 'managed-runtime'),
                           lambda: runtime.serving_config(cfg, runtime_root='runtime', studio_root='vendor',
                               llama_producer='producer', library_directories=[], revision='test')):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'KDA/MLA/indexer'):
                    action()
            result = verify_model_receipt(cfg)
            self.assertFalse(result['ready'])
            self.assertIn('KDA/MLA/indexer', result['error'])

    def test_existing_v41_model_and_missing_model_keep_existing_checks(self):
        with patch.object(Path, 'is_file', return_value=True), patch(
                'studio.gguf_meta._read_kv_header', return_value={'general.architecture': 'deepseek4'}):
            runtime.require_supported_model({'model': 'fixture.gguf'})
        with patch.object(Path, 'is_file', return_value=False), patch(
                'studio.gguf_meta._read_kv_header', side_effect=AssertionError('missing file')):
            runtime.require_supported_model({'model': 'missing.gguf'})


if __name__ == '__main__':
    unittest.main()
