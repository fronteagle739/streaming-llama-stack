"""The DSV4.1 single-engine launch configurations (scripts/dsv41_single.py and the .ps1 wrappers)."""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dsv41_single', ROOT / 'scripts' / 'dsv41_single.py')
dsv41_single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsv41_single)

GATED_ENV = {'DS4_DS_READ_PARTS': '8', 'LLAMADECODE_VRAM_RESERVE_MIB': '512',
             'LLAMADECODE_ENGRAM_EARLY': 'auto', 'LLAMADECODE_ENGRAM_CACHE_MB': '0',
             'LLAMADECODE_APPEND_DECODE_MAX': '0', 'LLAMADECODE_APPEND_RESIDENT_SWEEP': '1',
             'LLAMA_DSV41_INDEX_FP4': '1', 'DS4_CUDA_SINGLE_GPU_HC_GRAPHS': '1', 'LLAMADECODE_HOST_COPY_STAGING': 'auto',
             'LLAMADECODE_RECENCY_LFU': '1', 'LLAMADECODE_LFU_HALF_LIFE': '16'}


class LaunchConfigTests(unittest.TestCase):
    def test_every_profile_preserves_the_resident_sweep_off_toggle(self):
        for preset in ('262k','1m'):
            for profile in ('accurate','fast','ultra','ultra-yoco'):
                with self.subTest(preset=preset,profile=profile):
                    defaults=dsv41_single.overlay(preset,profile=profile)
                    self.assertEqual(defaults['env_overrides']['LLAMADECODE_APPEND_RESIDENT_SWEEP'],'1')
                    self.assertEqual(defaults['env_overrides']['LLAMA_DSV41_INDEX_FP4'],'1')
                    local={'env_overrides':{'LLAMADECODE_APPEND_RESIDENT_SWEEP':'0','LLAMA_DSV41_INDEX_FP4':'0','DS4_CUDA_SINGLE_GPU_HC_GRAPHS':'0','LLAMADECODE_HOST_COPY_STAGING':'0','LLAMADECODE_RECENCY_LFU':'0'}}
                    cfg=dsv41_single.overlay(preset,local,profile=profile)
                    self.assertEqual(cfg['env_overrides']['LLAMADECODE_APPEND_RESIDENT_SWEEP'],'0')
                    self.assertEqual(cfg['env_overrides']['LLAMA_DSV41_INDEX_FP4'],'0')
                    self.assertEqual(cfg['env_overrides']['DS4_CUDA_SINGLE_GPU_HC_GRAPHS'],'0')
                    self.assertEqual(cfg['env_overrides']['LLAMADECODE_HOST_COPY_STAGING'],'0')
                    self.assertEqual(cfg['env_overrides']['LLAMADECODE_RECENCY_LFU'],'0')
                    self.assertEqual(cfg['env_overrides']['LLAMADECODE_LFU_HALF_LIFE'],'16')
                    self.assertEqual(cfg['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'0')
                    self.assertEqual(local,{'env_overrides':{'LLAMADECODE_APPEND_RESIDENT_SWEEP':'0','LLAMA_DSV41_INDEX_FP4':'0','DS4_CUDA_SINGLE_GPU_HC_GRAPHS':'0','LLAMADECODE_HOST_COPY_STAGING':'0','LLAMADECODE_RECENCY_LFU':'0'}})

    def test_262k_is_f32_growable_single_engine(self):
        cfg = dsv41_single.overlay('262k')
        self.assertEqual((cfg['context'], cfg['ctx_cache_precision'], cfg['ctx_cache_growable']), (262144, 'f32', True))
        self.assertEqual(cfg['serving'], 'single_engine')
        self.assertEqual(cfg['producer_host_reserve_gib'], .5)
        self.assertEqual(cfg['env_overrides'], GATED_ENV)
        self.assertEqual((cfg['producer_prefill_sweep_tokens'], cfg['producer_prefill_fold_cap_tokens'], cfg['producer_host_max_gib']),
                         (49152, 61440, 7.75))   # the gated fold sweep plan

    def test_1m_is_bf16_growable_at_999424(self):
        cfg = dsv41_single.overlay('1m')
        self.assertEqual((cfg['context'], cfg['ctx_cache_precision'], cfg['ctx_cache_growable']), (999424, 'bf16', True))

    def test_local_keys_survive_and_gated_keys_win(self):
        local = {'model': 'M.gguf', 'port': 9000, 'env_overrides': {'FOO': '1', 'LLAMADECODE_ENGRAM_CACHE_MB': '64'},
                 'producer_host_max_gib': 12, 'producer_prefill_sweep_tokens': 98304, 'context': 4096}
        cfg = dsv41_single.overlay('262k', local, model='N.gguf')
        self.assertEqual((cfg['model'], cfg['port'], cfg['context']), ('N.gguf', 9000, 262144))
        self.assertEqual(cfg['env_overrides'], {'FOO': '1', **GATED_ENV, 'LLAMADECODE_ENGRAM_CACHE_MB': '64'})
        self.assertEqual((cfg['producer_host_max_gib'], cfg['producer_prefill_sweep_tokens']), (7.75, 49152))   # gated fold plan wins
        self.assertEqual(local['env_overrides']['LLAMADECODE_ENGRAM_CACHE_MB'], '64')   # input untouched

    def test_resolved_producer_flags_env_and_sweep(self):
        fold = {'--prefill-sweep-tokens': '49152', '--prefill-fold-cap-tokens': '61440', '--yoco-keep': '2541'}
        for preset, flags in (('262k', {'--context': '262144', '--compressed-cache-bf16': '0',
                                        '--ctx-cache-precision': 'f32', '--ctx-cache-growable': '1', **fold}),
                              ('1m', {'--context': '999424', '--compressed-cache-bf16': '1',
                                      '--ctx-cache-precision': 'bf16', '--ctx-cache-growable': '1', **fold})):
            with self.subTest(preset=preset):
                with tempfile.TemporaryDirectory() as runtime:
                    row = dsv41_single.resolved({**dsv41_single.overlay(preset), 'runtime_root': runtime})
                self.assertEqual(row['producer_flags'], flags)
                for key, value in GATED_ENV.items():
                    self.assertEqual(row['producer_env'][key], value)
                self.assertEqual(row['producer_env']['LLAMADECODE_DS4_DECODE'], '1')
                self.assertEqual((row['sweep_plan']['policy'], row['sweep_plan']['primary_sweep_tokens'],
                                  row['sweep_plan']['fold_cap_tokens']), ('fold', 49152, 61440))
                self.assertEqual(row['sweep_plan']['capacity_tokens'], 49152)   # a short prompt keeps the base bank
                self.assertGreater(row['sweep_plan']['max_host_bytes'], row['sweep_plan']['base_host_bytes'])
                self.assertEqual(row['single_engine']['producer_host_reserve_gib'], .5)

    def test_default_cone_reaches_native_args_and_keeps_qualified_routing_boundary(self):
        from slstack.combo_producer import prefill_arguments, routed_prefill_choice
        for preset in ('262k','1m'):
            cfg=dsv41_single.overlay(preset, dict(producer_profile='accurate', yoco_keep=140))
            self.assertEqual((cfg['producer_profile'],cfg['yoco_keep']),('ultra',2541))
            self.assertEqual(cfg.get('producer_chunks_per_pass',1),1)
            for count, routed in ((2048,'1'),(2049,'tail'),(49555,'tail')):
                self.assertEqual(routed_prefill_choice(cfg,count),routed)
                args,plan=prefill_arguments(cfg,count,20*1024**3)
                self.assertEqual(args['--yoco-keep'],'2541')
                self.assertEqual(args['--capture-dspark'],'0')
            with self.assertRaisesRegex(ValueError,'no negotiated remote state'):
                prefill_arguments(cfg,49555,20*1024**3,export_state=True)

    def test_resident_followup_is_an_explicit_local_toggle(self):
        cfg=dsv41_single.overlay('262k',dict(env_overrides={'LLAMADECODE_APPEND_DECODE_MAX':'96'}))
        self.assertEqual(cfg['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'96')

    def test_resident_followup_defaults_on_only_for_ultra_cone(self):
        self.assertEqual(dsv41_single.overlay('262k')['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'0')
        self.assertEqual(dsv41_single.overlay('262k',profile='ultra-yoco')['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'0')
        off=dsv41_single.overlay('262k',dict(env_overrides={'LLAMADECODE_APPEND_DECODE_MAX':'0'}))
        self.assertEqual(off['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'0')

    def test_lfu_and_engram_options_reach_the_producer(self):
        cfg=dsv41_single.overlay('262k',dict(env_overrides={
            'LLAMADECODE_RECENCY_LFU':'0', 'LLAMADECODE_LFU_HALF_LIFE':'1',
            'LLAMADECODE_ENGRAM_READ_ALIGN':'512','LLAMADECODE_ENGRAM_EARLY':'0',
            'LLAMADECODE_ENGRAM_CACHE_MB':'64'}))
        with tempfile.TemporaryDirectory() as runtime:
            resolved=dsv41_single.resolved({**cfg,'runtime_root':runtime})
        for key in ('LLAMADECODE_RECENCY_LFU','LLAMADECODE_LFU_HALF_LIFE',
                    'LLAMADECODE_ENGRAM_READ_ALIGN','LLAMADECODE_ENGRAM_EARLY','LLAMADECODE_ENGRAM_CACHE_MB'):
            self.assertEqual(resolved['producer_env'][key],cfg['env_overrides'][key])

    def test_write_and_dry_run_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / 'my.local.json'
            local.write_text(json.dumps({'runtime_root': folder, 'model': 'M.gguf'}), encoding='utf-8')
            target = Path(folder) / 'start.local.json'
            result = subprocess.run([os.sys.executable, '-B', str(ROOT / 'scripts' / 'dsv41_single.py'), '--preset', '1m',
                                     '--config', str(local), '--write', str(target), '--dry-run'],
                                    capture_output=True, text=True, timeout=60, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), printed['start_config'])
            self.assertEqual(printed['resolved']['single_engine']['context'], 999424)

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell'), 'Windows PowerShell launcher')
    def test_powershell_dry_run_starts_nothing(self):
        result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(ROOT / 'scripts' / 'start-dsv41-single-262k.ps1'), '-DryRun', '-WebUI', '-Model', 'M.gguf'],
                                capture_output=True, text=True, timeout=120, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('bootstrap.py', result.stdout)
        self.assertIn('start --webui', result.stdout)
        self.assertIn('Router / web UI: http://127.0.0.1:', result.stdout)
        written = json.loads((ROOT / 'runtime' / 'dsv41-single-262k-ultra-cone.local.json').read_text(encoding='utf-8'))
        self.assertEqual(written['ctx_cache_precision'], 'f32')
        self.assertEqual(written['yoco_keep'],2541)


class KeptKeyTests(unittest.TestCase):
    def test_reserve_reaches_the_single_engine_config(self):
        from slstack.single_runtime import serving_config
        row = serving_config({'host_budget_gib': 8, 'producer_host_reserve_gib': 1, 'context': 262144},
                             runtime_root='r', studio_root='s', llama_producer='p.exe', library_directories=[], revision='x')
        self.assertEqual(row['producer_host_reserve_gib'], 1)

    def test_stack_single_engine_uses_the_packaged_spread_file_by_default(self):
        from slstack.single_runtime import stack_config
        with tempfile.TemporaryDirectory() as folder:
            producer = Path(folder) / 'llama-dsv41-prefill.exe'
            for name in (producer.name, 'ds4-decode.dll', 'dstorage.dll', 'dstoragecore.dll'):
                (Path(folder) / name).write_bytes(b'x')
            cfg = {**dsv41_single.overlay('262k'), 'llama_producer': str(producer), 'runtime_root': folder,
                   'ds4_url': 'http://127.0.0.1:8089', 'studio_root': 'vendor/studio', 'host_budget_gib': 8,
                   'min_available_gib': 2, 'cache_affinity_spread_file': ''}
            row = stack_config(cfg)
        self.assertEqual(Path(row['cache_affinity_spread_file']), ROOT / 'presets' / 'dsv41-affinity-spreads.txt')
        self.assertEqual((row['context'], row['ctx_cache_precision'], row['ctx_cache_growable']), (262144, 'f32', True))
        self.assertEqual(row['env_overrides']['LLAMADECODE_ENGRAM_CACHE_MB'], '0')
        self.assertEqual(row['producer_host_reserve_gib'], .5)

    def test_stack_single_engine_creates_its_hold_file_on_a_fresh_runtime(self):
        from slstack.single_runtime import stack_config
        with tempfile.TemporaryDirectory() as folder:
            producer = Path(folder) / 'llama-dsv41-prefill.exe'
            for name in (producer.name, 'ds4-decode.dll', 'dstorage.dll', 'dstoragecore.dll'):
                (Path(folder) / name).write_bytes(b'x')
            runtime = Path(folder) / 'runtime'
            cfg = {**dsv41_single.overlay('262k'), 'llama_producer': str(producer), 'runtime_root': str(runtime),
                   'ds4_url': 'http://127.0.0.1:8089', 'studio_root': 'vendor/studio', 'host_budget_gib': 8,
                   'min_available_gib': 2, 'cache_affinity_spread_file': ''}
            row = stack_config(cfg)
            self.assertTrue(Path(row['hold_file']).is_file())
            self.assertEqual(Path(row['hold_file']), runtime.resolve() / 'state' / 'gpu.holds')


class RuntimePythonTests(unittest.TestCase):
    def test_stage_refuses_an_interpreter_other_than_the_runtime_python(self):
        from slstack.single_runtime import require_runtime_python
        with tempfile.TemporaryDirectory() as folder:
            runtime, other = Path(folder) / 'venv-python.exe', Path(folder) / 'system-python.exe'
            for p in (runtime, other):
                p.write_bytes(b'x')
            require_runtime_python({'studio_python': str(runtime)}, executable=str(runtime))
            with self.assertRaisesRegex(ValueError, 'Stage with the runtime interpreter'):
                require_runtime_python({'studio_python': str(runtime)}, executable=str(other))
            with self.assertRaisesRegex(ValueError, 'needs studio_python'):
                require_runtime_python({}, executable=str(runtime))


if __name__ == '__main__':
    unittest.main()
