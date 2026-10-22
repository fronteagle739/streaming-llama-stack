import contextlib
import io
import unittest
from slstack.envvars import apply
from slstack.config import load


class EnvironmentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(overrides={'gpu_uuid': 'GPU-test'})

    def test_native_prefill_keeps_core_and_ignores_inherited_eviction(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            env = apply({'DS4_CUDA_PREFILL_EDGE_VRAM': '1'}, self.cfg, 'ds4-win')
        self.assertEqual(env['DS4_CUDA_PREFILL_EDGE_VRAM'], '0')
        self.assertIn('ignored', output.getvalue())

    def test_explicit_core_eviction_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'memory-residency'):
            apply({}, {**self.cfg, 'env_overrides': {'DS4_CUDA_PREFILL_EDGE_VRAM': '1'}}, 'ds4-win')

    def test_mac_native_storage_is_wired_and_cannot_be_disabled(self):
        self.assertEqual(apply({}, self.cfg, 'ds4-mac')['DS4_METAL_WIRED_CORE_KV'], '1')
        with self.assertRaisesRegex(ValueError, 'memory-residency'):
            apply({}, {**self.cfg, 'env_overrides': {'DS4_METAL_WIRED_CORE_KV': '0'}}, 'ds4-mac')
