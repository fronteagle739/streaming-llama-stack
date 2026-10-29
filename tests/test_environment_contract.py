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
    def test_other22_explicit_controls_reach_the_native_child(self):
        from slstack.config import load
        cfg=load(overrides={'gpu_uuid':'GPU-test'})
        chosen={'GGML_CUDA_DISABLE_GRAPHS':'0','LLAMA_DSV41_CUDA_GRAPHS':'1',
                'LLAMA_DSV41_HC_COLLAPSE_FUSION':'0','LLAMA_DSV41_QAT_TILED':'0',
                'LLAMA_DSV41_ASYNC_LAYER':'1','LLAMA_DSV41_LAYER_FUSION':'2'}
        cfg['env_overrides']=chosen
        actual=apply({},cfg,'rtx-producer')
        for name,value in chosen.items():self.assertEqual(actual.get(name),value)
        cfg['env_overrides']={}
        self.assertEqual(apply({},cfg,'rtx-producer')['GGML_CUDA_DISABLE_GRAPHS'],'1')
        for name in chosen:
            if name!='GGML_CUDA_DISABLE_GRAPHS':self.assertNotIn(name,apply({},cfg,'rtx-producer'))

    def test_pinned_external_evidence_is_retained_but_stale_local_reads_fail(self):
        row=dict(var='PATH',doc='Host path',reads=['engines/current.c:1','engines/mlx-serve/pinned.zig:2'])
        with patch('slstack.envvars.rows',return_value=[row]),patch('slstack.envvars.inventory',return_value={'PATH':['engines/current.c:1']}):
            self.assertEqual(validate_table(),1)
            with patch('slstack.envvars.inventory',return_value={'PATH':['engines/current.c:3']}):
                with self.assertRaisesRegex(ValueError,'stale'):validate_table()

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


    def test_test_fault_is_documented_and_only_explicit(self):
        from slstack.envvars import rows
        row = next(r for r in rows() if r['var'] == 'LLAMADECODE_TEST_FAULT')
        self.assertTrue(row['doc'] and 'TEST-ONLY' in row['doc'] and row['scopes'] == ['rtx-llama'] and not row.get('defaults_by_scope'))
        with patch('slstack.envvars.rows', return_value=[row]):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertNotIn('LLAMADECODE_TEST_FAULT', apply({'LLAMADECODE_TEST_FAULT': 'main_throw@4'}, {}, 'rtx-producer'))
            cfg = {'env_overrides': {'LLAMADECODE_TEST_FAULT': 'worker_throw@4'}}
            self.assertEqual(apply({}, cfg, 'rtx-producer')['LLAMADECODE_TEST_FAULT'], 'worker_throw@4')
            self.assertNotIn('LLAMADECODE_TEST_FAULT', apply({}, {}, 'rtx-producer'))

if __name__ == '__main__':
    unittest.main()
