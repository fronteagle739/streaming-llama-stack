import contextlib
import io
import unittest
from slstack.envvars import apply
from slstack.config import load


class EnvironmentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(overrides={'gpu_uuid': 'GPU-test'})

    def test_writer_trace_reaches_native_child_without_affecting_other_scopes(self):
        for value in ('0', '1'):
            cfg = {**self.cfg, 'env_overrides': {'DS4_DIRECTION_TRACE': value}}
            self.assertEqual(apply({}, cfg, 'rtx-producer')['DS4_DIRECTION_TRACE'], value)
            self.assertEqual(apply({}, cfg, 'ds4-win')['DS4_DIRECTION_TRACE'], value)
            self.assertNotIn('DS4_DIRECTION_TRACE', apply({'DS4_DIRECTION_TRACE': value}, cfg, 'build'))
        with self.assertRaisesRegex(ValueError, 'memory-residency'):
            apply({}, {**self.cfg, 'env_overrides': {'DS4_DIRECTION_TRACE': '2'}}, 'rtx-producer')

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
