import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from slstack.envvars import apply, validate_table


class EnvironmentContractTests(unittest.TestCase):
    def test_all_source_reads_have_documented_table_rows(self):
        self.assertGreater(validate_table(), 100)

    def test_nonstandard_engine_prefix_does_not_leak_from_shell(self):
        row = dict(var='MINIMAX_TEST_SWITCH', flag=None, scopes=['rtx-llama'])
        with patch('slstack.envvars.rows', return_value=[row]), contextlib.redirect_stderr(io.StringIO()) as out:
            env = apply({**os.environ, 'MINIMAX_TEST_SWITCH': 'unsafe'}, {}, 'rtx-producer')
        actual = subprocess.check_output([sys.executable, '-c', 'import os; print(os.getenv("MINIMAX_TEST_SWITCH", "absent"))'], env=env, text=True)
        self.assertEqual(actual.strip(), 'absent')
        self.assertIn('ignored', out.getvalue())

    def test_literal_json_override_and_scope(self):
        row = dict(var='GGML_TEST_JSON', flag='--ggml-test-json', scopes=['rtx-llama'],
                   defaults_by_scope={'rtx-llama': '{host_budget_bytes}'})
        with patch('slstack.envvars.rows', return_value=[row]):
            cfg = {'host_budget_gib': 2, 'env_overrides': {'GGML_TEST_JSON': '{"size": 12}'}}
            self.assertEqual(apply({}, cfg, 'rtx-producer')['GGML_TEST_JSON'], '{"size": 12}')
            self.assertNotIn('GGML_TEST_JSON', apply({}, cfg, 'image'))
            self.assertEqual(apply({}, {'host_budget_gib': 2}, 'rtx-llama')['GGML_TEST_JSON'], str(2 << 30))

    def test_nonprefix_preset_is_applied_to_real_child(self):
        from slstack.config import load
        cfg = load(overrides={'gpu_uuid': 'GPU-test', 'host_budget_gib': 1, 'min_available_gib': 4})
        env = apply(dict(os.environ), cfg, 'image')
        actual = subprocess.check_output([sys.executable, '-c', 'import os; print(os.environ["CUDA_DEVICE_ORDER"])'], env=env, text=True)
        self.assertEqual(actual.strip(), 'PCI_BUS_ID')

    def test_runtime_path_and_memory_overrides_survive_environment_builder(self):
        from slstack.config import load, environment
        runtime = os.environ.get('STUDIO_RUNTIME_ROOT')
        cfg = load(overrides={'gpu_uuid': 'GPU-test', 'host_budget_gib': 1, 'min_available_gib': 4,
                              **({'runtime_root': runtime} if runtime else {})})
        cfg['env_overrides'] = {'MEM_GUARD_MIN_FREE_GB': '5', 'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'}
        env = environment(cfg, 'image')
        self.assertEqual(env['MEM_GUARD_MIN_FREE_GB'], '5')
        self.assertEqual(env['CUDA_DEVICE_ORDER'], 'FASTEST_FIRST')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertTrue(env['TMP'].startswith(cfg['runtime_root']))
        self.assertAlmostEqual(float(env['MEM_GUARD_MAX_PROC_GB']) * 1e9, 1 << 30)
        # Windows private commitment includes GPU backing; it is separate
        # from the physical RSS cap enforced by the outer guardian.
        self.assertEqual(apply({}, cfg, 'ds4-win')['DS4_HOST_MEMORY_BUDGET_MB'], '45056')

    def test_environment_cannot_override_guard_gpu_or_memory_limits(self):
        from slstack.config import load
        cfg = load(overrides={'gpu_uuid': 'GPU-test', 'host_budget_gib': 1, 'min_available_gib': 4})
        for scope, var, value in [('rtx-llama', 'CUDA_VISIBLE_DEVICES', 'GPU-other'),
                                  ('mac-ds4', 'DS4_NATIVE_SESSION_HOST_BYTES', str(2 << 30)),
                                  ('mac-ds4', 'DS4_NATIVE_SESSION_HOST_FLOOR_BYTES', str(1 << 30)),
                                  ('ds4-win', 'DS4_HOST_MEMORY_BUDGET_MB', '65536'),
                                  ('image', 'MEM_GUARD_DISABLE', '1')]:
            with self.subTest(var=var), self.assertRaises(ValueError):
                apply({}, {**cfg, 'env_overrides': {var: value}}, scope)

    def test_bias_flag_beats_configured_engine_argument(self):
        from slstack.envvars import argv_overrides
        cfg = {'env_overrides': {'DS4_CACHE_AFFINITY_BIAS': '0.35'}}
        for args in [['engine', '--cache-affinity-bias', '0', '--cpu'],
                     ['engine', '--cache-affinity-bias=0', '--cpu']]:
            self.assertEqual(argv_overrides(args, cfg, 'ds4-win'), ['engine', '--cpu'])


if __name__ == '__main__':
    unittest.main()
