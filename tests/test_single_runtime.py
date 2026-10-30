"""Single-engine managed runtime: installed config, preset and the Studio launch argv."""
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import single_runtime as sr
from slstack.combo_proxy import native_launch_config, request_worker_config

# Pure configuration paths: these tests never create files. A real temporary
# root may itself contain "scratch", which the installed-config guard rejects.
FIXTURE = Path('managed-runtime-fixture')
BASE = dict(gpu_uuid='GPU-1', gpu_idle_max_mib=1024, hold_file=str(FIXTURE / 'gpu.holds'), task='LLAMADECODE',
            lease_minutes=19, request_timeout_s=1140, host_budget_gib=9.25, min_available_gib=1, job_commit_gib=48,
            context=262144, model=str(FIXTURE / 'models/x.gguf'), model_name='DSV4.1-1017', producer_host_budget_gib=6.25,
            producer_profile='ultra', producer_sha256='a' * 64, producer_source_revision='r',
            llama_producer=str(FIXTURE / 'scratch/bin/llamadecode-r.exe'), runtime_root=str(FIXTURE / 'scratch/runs'),
            producer_library_directories=[str(FIXTURE / 'scratch/engines/ds4-win'), str(FIXTURE / 'cuda/bin')],
            cache_affinity_spread_file=str(FIXTURE / 's.txt'), model_receipt=str(FIXTURE / 'scratch/m.json'), ds4_command=['native.exe'],
            env_overrides={'CUDA_CACHE_DISABLE': '1'}, host='127.0.0.1', port=18748)
INSTALLED = FIXTURE / 'installed/0123abcd'
STUDIO_NATIVE = ROOT.parent / 'front-eagle-stack/backend/studio/ds4_native.py'


class RuntimeTests(unittest.TestCase):
    def test_packed_fp4_extension_and_temp_paths_survive_scratch_retirement(self):
        env=dict(BASE['env_overrides'], LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL=str(FIXTURE/'scratch/bin/native.dll'),
                 LLAMA_DSV41_STORED_MXFP4='1', LLAMADECODE_PREFILL_ROUTED='0',
                 TEMP=str(FIXTURE/'scratch'), TMP=str(FIXTURE/'scratch'), TMPDIR=str(FIXTURE/'scratch'),
                 CUDA_CACHE_PATH=str(FIXTURE/'scratch/cache'))
        cfg=sr.single_config(dict(BASE,env_overrides=env),INSTALLED,'rev')
        self.assertEqual(Path(cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL']),INSTALLED/'native.dll')
        self.assertEqual(cfg['env_overrides']['LLAMA_DSV41_STORED_MXFP4'],'1')
        self.assertEqual(cfg['env_overrides']['LLAMADECODE_PREFILL_ROUTED'],'0')
        self.assertEqual(set(cfg['env_overrides']),set(env)|{'LLAMADECODE_DS4_DECODE'})
        self.assertTrue(all(Path(cfg['env_overrides'][k])==INSTALLED/'runtime/temp' for k in ('TEMP','TMP','TMPDIR')))
        self.assertEqual(sr.extension_files(dict(env_overrides=env)),
                         {'LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL':Path(env['LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL'])})
        with self.assertRaises(ValueError):sr.extension_files({'env_overrides':{'LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL':''}})
        with self.assertRaises(ValueError):sr.extension_files({'env_overrides':{'LLAMA_DSV41_DENSE_FP8_DLL':'wrong.exe'}})

    def test_runtime_floor_config_default_and_cli_override(self):
        from slstack.config import load
        from slstack.__main__ import main
        cfg=load(overrides={'gpu_uuid':'GPU-test'})
        self.assertEqual(cfg['runtime_min_free_gib'],0.5)
        for argv in (['slstack','--runtime-min-free-gib','0','check'],['slstack','check','--runtime-min-free-gib','6.5']):
            with self.subTest(argv=argv),patch.object(sys,'argv',argv),patch('slstack.__main__.load',return_value=cfg) as loaded,patch('slstack.check.check',return_value=[]):
                self.assertEqual(main(),0)
                self.assertEqual(loaded.call_args.args[1]['runtime_min_free_gib'],float(argv[argv.index('--runtime-min-free-gib')+1]))

    def test_installed_config_is_self_contained(self):
        cfg = sr.single_config(BASE, INSTALLED, 'rev')
        self.assertEqual(cfg['serving_mode'], 'single_engine')
        self.assertEqual(cfg['env_overrides'], {'CUDA_CACHE_DISABLE': '1', 'LLAMADECODE_DS4_DECODE': '1'})
        self.assertEqual(Path(cfg['llama_producer']), INSTALLED / 'llamadecode-r.exe')
        self.assertEqual([Path(p) for p in cfg['producer_library_directories']], [INSTALLED, FIXTURE / 'cuda/bin'])
        self.assertNotIn('ds4_command', cfg);self.assertEqual(cfg['single_engine_stop_tokens'], [1])
        self.assertEqual(request_worker_config(cfg)['host_budget_gib'], BASE['host_budget_gib'])
        with self.assertRaises(ValueError):sr.single_config(dict(BASE, model=str(FIXTURE / 'scratch/model.gguf')), INSTALLED, 'rev')

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

    def test_runtime_floor_survives_managed_runtime_and_worker_config(self):
        for floor in (0, 4.0, 6.5):
            with self.subTest(floor=floor):
                cfg = sr.single_config(dict(BASE, runtime_min_free_gib=floor), INSTALLED, 'rev')
                self.assertEqual(cfg['runtime_min_free_gib'], floor)
                self.assertEqual(request_worker_config(cfg)['runtime_min_free_gib'], floor)

    @unittest.skipUnless(STUDIO_NATIVE.exists(), 'needs the Studio checkout')
    def test_studio_argv_launches_the_single_mode(self):
        cfg = sr.single_config(BASE, INSTALLED, 'rev');preset = sr.preset(BASE, INSTALLED)
        cfg['studio_expected_args'] = sr.normal_argv(STUDIO_NATIVE, preset)
        argv = list(cfg['studio_expected_args']);argv[argv.index('--port') + 1] = '19999'
        launched = native_launch_config(cfg, argv)
        self.assertEqual((launched['port'], launched['serving_mode']), (19999, 'single_engine'))
        with self.assertRaises(ValueError):native_launch_config(cfg, argv + ['--extra'])


if __name__ == '__main__':unittest.main()
