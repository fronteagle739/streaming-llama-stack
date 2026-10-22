import contextlib
import io
import unittest
from slstack.envvars import apply


class EnvironmentSafetyTests(unittest.TestCase):
    def test_native_prefill_keeps_core_and_ignores_inherited_eviction(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            env = apply({'DS4_CUDA_PREFILL_EDGE_VRAM': '1'}, {}, 'ds4-win')
        self.assertEqual(env['DS4_CUDA_PREFILL_EDGE_VRAM'], '0')
        self.assertIn('ignored', output.getvalue())

    def test_explicit_core_eviction_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'memory-residency'):
            apply({}, {'env_overrides': {'DS4_CUDA_PREFILL_EDGE_VRAM': '1'}}, 'ds4-win')

    def test_mac_native_storage_is_wired_and_cannot_be_disabled(self):
        self.assertEqual(apply({}, {}, 'ds4-mac')['DS4_METAL_WIRED_CORE_KV'], '1')
        with self.assertRaisesRegex(ValueError, 'memory-residency'):
            apply({}, {'env_overrides': {'DS4_METAL_WIRED_CORE_KV': '0'}}, 'ds4-mac')
