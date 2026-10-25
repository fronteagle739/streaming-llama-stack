"""Single-engine managed runtime: installed config, preset and the Studio launch argv."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import single_runtime as sr
from slstack.combo_proxy import native_launch_config, request_worker_config

BASE = dict(gpu_uuid='GPU-1', gpu_idle_max_mib=1024, hold_file='runtime/coordination/DEPLOY_HOLD.md', task='LLAMADECODE',
            lease_minutes=19, request_timeout_s=1140, host_budget_gib=9.25, min_available_gib=1, job_commit_gib=48,
            context=262144, model='models/x.gguf', model_name='DSV4.1-1017', producer_host_budget_gib=6.25,
            producer_profile='ultra', producer_sha256='a' * 64, producer_source_revision='r',
            llama_producer='runtime/llamadecode-r.exe', runtime_root='runtime/runs',
            producer_library_directories=['runtime/ds4-win', 'C:/CUDA/v13.2/bin/x64'],
            cache_affinity_spread_file='runtime/s.txt', model_receipt='runtime/m.json', ds4_command=['native.exe'],
            env_overrides={'CUDA_CACHE_DISABLE': '1'}, host='127.0.0.1', port=18748)
INSTALLED = Path('runtime/0123abcd')
STUDIO_NATIVE = Path('source/ds4_native.py')


class RuntimeTests(unittest.TestCase):
    def test_installed_config_is_self_contained(self):
        cfg = sr.single_config(BASE, INSTALLED, 'rev')
        self.assertEqual(cfg['serving_mode'], 'single_engine')
        self.assertEqual(cfg['env_overrides'], {'CUDA_CACHE_DISABLE': '1', 'LLAMADECODE_DS4_DECODE': '1'})
        self.assertEqual(Path(cfg['llama_producer']), INSTALLED / 'llamadecode-r.exe')
        self.assertEqual([Path(p) for p in cfg['producer_library_directories']], [INSTALLED, Path('C:/CUDA/v13.2/bin/x64')])
        self.assertNotIn('ds4_command', cfg);self.assertEqual(cfg['single_engine_stop_tokens'], [1])
        self.assertEqual(request_worker_config(cfg)['host_budget_gib'], BASE['host_budget_gib'])
        with self.assertRaises(ValueError):sr.single_config(dict(BASE, model='models/model.gguf'), INSTALLED, 'rev')

    def test_preset_points_at_the_runtime(self):
        preset = sr.preset(BASE, INSTALLED)
        self.assertEqual(preset['name'], 'DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine)')
        rtx = preset['rtx'];self.assertEqual(rtx['backend'], 'ds4')
        self.assertEqual(Path(rtx['ds4_binary']), INSTALLED / 'ds4-server.exe')
        self.assertEqual(rtx['ds4_env'], {'LLAMADECODE_DS4_DECODE': '1'})
        self.assertTrue(rtx['ds4_cache_affinity_spread_file'])   # Studio offers the quality-speed tool only then

    def test_ctx_cache_config(self):
        from slstack.combo_producer import ctx_cache
        self.assertEqual(ctx_cache(BASE), ('f32', False))   # absent = today's fixed f32 argv
        self.assertEqual(ctx_cache(dict(BASE, ctx_cache_precision='bf16', ctx_cache_growable=True)), ('bf16', True))
        for bad in (dict(ctx_cache_precision='f16'), dict(ctx_cache_growable=1), dict(ctx_cache_growable=True, producer_keep_alive=True)):
            with self.assertRaises(ValueError):ctx_cache(dict(BASE, **bad))
        big = dict(BASE, context=999424, ctx_cache_precision='bf16', ctx_cache_growable=True,
                   preset_name='DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine, 1M)')
        cfg = sr.single_config(big, INSTALLED, 'rev')
        self.assertEqual((cfg['context'], cfg['ctx_cache_precision'], cfg['ctx_cache_growable']), (999424, 'bf16', True))
        preset = sr.preset(big, INSTALLED)
        self.assertEqual((preset['name'], preset['rtx']['ctx']), (big['preset_name'], 999424))

    def test_host_registration_config(self):
        from slstack.combo_producer import host_registration, host_registration_args
        from slstack.native_profiles import profile
        today = dict(profile('ultra')['arguments'])
        self.assertEqual(host_registration(BASE), 'tile')   # absent = today's per-tile registration
        self.assertEqual(host_registration_args(BASE, dict(today)), today)   # argv unchanged
        self.assertEqual(host_registration_args(dict(BASE, producer_host_registration='tile'), dict(today)), today)
        for mode in ('whole', 'tile-async'):
            args = host_registration_args(dict(BASE, producer_host_registration=mode), dict(today))
            self.assertEqual(args.pop('--host-registration'), mode)
            self.assertEqual(args, today)   # nothing else changes; tile registration stays forced on
            self.assertEqual(args['--tile-host-registration'], '1')
        for bad in ('async', 'Whole', 1, None):
            with self.assertRaises(ValueError):host_registration(dict(BASE, producer_host_registration=bad))
        with self.assertRaises(ValueError):
            host_registration_args(dict(BASE, producer_host_registration='whole'), dict(today, **{'--tile-host-registration': '0'}))
        cfg = sr.single_config(dict(BASE, producer_host_registration='whole'), INSTALLED, 'rev')
        self.assertEqual(cfg['producer_host_registration'], 'whole')   # carried into the installed runtime config

    @unittest.skipUnless(STUDIO_NATIVE.exists(), 'needs the Studio checkout')
    def test_studio_argv_launches_the_single_mode(self):
        cfg = sr.single_config(BASE, INSTALLED, 'rev');preset = sr.preset(BASE, INSTALLED)
        cfg['studio_expected_args'] = sr.normal_argv(STUDIO_NATIVE, preset)
        argv = list(cfg['studio_expected_args']);argv[argv.index('--port') + 1] = '19999'
        launched = native_launch_config(cfg, argv)
        self.assertEqual((launched['port'], launched['serving_mode']), (19999, 'single_engine'))
        with self.assertRaises(ValueError):native_launch_config(cfg, argv + ['--extra'])


if __name__ == '__main__':unittest.main()
