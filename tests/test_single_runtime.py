"""Single-engine managed runtime: installed config, preset and the Studio launch argv."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import single_runtime as sr
from slstack.combo_proxy import native_launch_config, request_worker_config

BASE = dict(gpu_uuid='GPU-1', gpu_idle_max_mib=1024, hold_file='X:/tmp/delegate/GPU_HOLDS.md', task='LLAMADECODE',
            lease_minutes=19, request_timeout_s=1140, host_budget_gib=9.25, min_available_gib=1, job_commit_gib=48,
            context=262144, model='X:/models/x.gguf', model_name='DSV4.1-1017', producer_host_budget_gib=6.25,
            producer_profile='ultra', producer_sha256='a' * 64, producer_source_revision='r',
            llama_producer='Y:\\scratch\\llamadecode\\bin-r\\llamadecode-r.exe', runtime_root='Y:\\scratch\\llamadecode\\runs',
            producer_library_directories=['Y:\\scratch\\ds4combo\\worktree\\engines\\ds4-win', 'C:/CUDA/v13.2/bin/x64'],
            cache_affinity_spread_file='X:/s.txt', model_receipt='Y:/scratch/m.json', ds4_command=['native.exe'],
            env_overrides={'CUDA_CACHE_DISABLE': '1'}, host='127.0.0.1', port=18748)
INSTALLED = Path('X:/StreamingStudio/runtime/ds4/0123abcd')
STUDIO_NATIVE = Path('X:/GitHub/front-eagle-stack/backend/studio/ds4_native.py')


class RuntimeTests(unittest.TestCase):
    def test_installed_config_is_self_contained(self):
        cfg = sr.single_config(BASE, INSTALLED, 'rev')
        self.assertEqual(cfg['serving_mode'], 'single_engine')
        self.assertEqual(cfg['env_overrides'], {'CUDA_CACHE_DISABLE': '1', 'LLAMADECODE_DS4_DECODE': '1'})
        self.assertEqual(Path(cfg['llama_producer']), INSTALLED / 'llamadecode-r.exe')
        self.assertEqual([Path(p) for p in cfg['producer_library_directories']], [INSTALLED, Path('C:/CUDA/v13.2/bin/x64')])
        self.assertNotIn('ds4_command', cfg);self.assertEqual(cfg['single_engine_stop_tokens'], [1])
        self.assertEqual(request_worker_config(cfg)['host_budget_gib'], BASE['host_budget_gib'])
        with self.assertRaises(ValueError):sr.single_config(dict(BASE, model='Y:/scratch/model.gguf'), INSTALLED, 'rev')

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

    @unittest.skipUnless(STUDIO_NATIVE.exists(), 'needs the Studio checkout')
    def test_studio_argv_launches_the_single_mode(self):
        cfg = sr.single_config(BASE, INSTALLED, 'rev');preset = sr.preset(BASE, INSTALLED)
        cfg['studio_expected_args'] = sr.normal_argv(STUDIO_NATIVE, preset)
        argv = list(cfg['studio_expected_args']);argv[argv.index('--port') + 1] = '19999'
        launched = native_launch_config(cfg, argv)
        self.assertEqual((launched['port'], launched['serving_mode']), (19999, 'single_engine'))
        with self.assertRaises(ValueError):native_launch_config(cfg, argv + ['--extra'])


if __name__ == '__main__':unittest.main()
