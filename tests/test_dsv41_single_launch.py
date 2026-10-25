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
             'LLAMADECODE_ENGRAM_EARLY': 'auto', 'LLAMADECODE_ENGRAM_CACHE_MB': '0'}


class LaunchConfigTests(unittest.TestCase):
    def test_262k_is_f32_growable_single_engine(self):
        cfg = dsv41_single.overlay('262k')
        self.assertEqual((cfg['context'], cfg['ctx_cache_precision'], cfg['ctx_cache_growable']), (262144, 'f32', True))
        self.assertEqual(cfg['serving'], 'single_engine')
        self.assertEqual(cfg['producer_host_reserve_gib'], 1)
        self.assertEqual(cfg['env_overrides'], GATED_ENV)
        self.assertNotIn('producer_host_max_gib', cfg)

    def test_1m_is_bf16_growable_at_999424(self):
        cfg = dsv41_single.overlay('1m')
        self.assertEqual((cfg['context'], cfg['ctx_cache_precision'], cfg['ctx_cache_growable']), (999424, 'bf16', True))

    def test_local_keys_survive_and_gated_keys_win(self):
        local = {'model': 'M.gguf', 'port': 9000, 'env_overrides': {'FOO': '1', 'LLAMADECODE_ENGRAM_CACHE_MB': '2048'},
                 'producer_host_max_gib': 12, 'context': 4096}
        cfg = dsv41_single.overlay('262k', local, model='N.gguf')
        self.assertEqual((cfg['model'], cfg['port'], cfg['context']), ('N.gguf', 9000, 262144))
        self.assertEqual(cfg['env_overrides'], {'FOO': '1', **GATED_ENV})
        self.assertNotIn('producer_host_max_gib', cfg)   # single-sweep stays off
        self.assertEqual(local['env_overrides']['LLAMADECODE_ENGRAM_CACHE_MB'], '2048')   # input untouched

    def test_resolved_producer_flags_env_and_sweep(self):
        for preset, flags in (('262k', {'--context': '262144', '--compressed-cache-bf16': '0',
                                        '--ctx-cache-precision': 'f32', '--ctx-cache-growable': '1'}),
                              ('1m', {'--context': '999424', '--compressed-cache-bf16': '1',
                                      '--ctx-cache-precision': 'bf16', '--ctx-cache-growable': '1'})):
            with self.subTest(preset=preset):
                with tempfile.TemporaryDirectory() as runtime:
                    row = dsv41_single.resolved({**dsv41_single.overlay(preset), 'runtime_root': runtime})
                self.assertEqual(row['producer_flags'], flags)
                for key, value in GATED_ENV.items():
                    self.assertEqual(row['producer_env'][key], value)
                self.assertEqual(row['producer_env']['LLAMADECODE_DS4_DECODE'], '1')
                self.assertFalse(row['single_sweep']['enabled'])
                self.assertEqual(row['single_sweep']['capacity_tokens'], 49152)
                self.assertEqual(row['single_sweep']['max_host_bytes'], row['single_sweep']['base_host_bytes'])
                self.assertEqual(row['single_engine']['producer_host_reserve_gib'], 1)

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
        written = json.loads((ROOT / 'runtime' / 'dsv41-single-262k.local.json').read_text(encoding='utf-8'))
        self.assertEqual(written['ctx_cache_precision'], 'f32')


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
        self.assertEqual(row['producer_host_reserve_gib'], 1)

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


if __name__ == '__main__':
    unittest.main()
