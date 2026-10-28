import importlib.util
from pathlib import Path
import unittest
import json
import shutil
import subprocess

from slstack.combo_producer import yoco_keep
from slstack.single_runtime import serving_config

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('yoco_launch',ROOT/'scripts/dsv41_single.py')
launch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class YocoTests(unittest.TestCase):
    def test_default_selects_exact_cone(self):
        cfg=launch.overlay('262k')
        self.assertEqual(yoco_keep(cfg),2541)
        self.assertEqual(launch.resolved(cfg)['producer_flags']['--yoco-keep'],'2541')
        full=launch.overlay('262k',profile='ultra')
        self.assertEqual(yoco_keep(full),0)
        self.assertNotIn('--yoco-keep',launch.resolved(full)['producer_flags'])

    def test_config_accepts_explicit_mac_route_and_refuses_unpaired_native(self):
        from slstack.config import load
        cfg=load(overrides=dict(gpu_uuid='GPU-test',mac='http://example.test:8180',yoco_keep=140,serving='native'))
        self.assertEqual(cfg['yoco_keep'],140)
        with self.assertRaisesRegex(ValueError,'explicitly paired'):
            load(overrides=dict(gpu_uuid='GPU-test',yoco_keep=140,serving='native'))

    def test_config_rejects_malformed_keep_before_launch(self):
        from slstack.config import load
        for keep in (True,'140',139,141):
            with self.subTest(keep=keep),self.assertRaisesRegex(ValueError,'0, 140 or 2541'):
                load(overrides=dict(gpu_uuid='GPU-test',yoco_keep=keep))
        # CLI None means "no override"; explicit JSON null is malformed.
        import tempfile
        config=Path(tempfile.mkdtemp())/'null.local.json'
        config.write_text('{"yoco_keep":null}')
        with self.assertRaisesRegex(ValueError,'0, 140 or 2541'):
            load(config,overrides=dict(gpu_uuid='GPU-test'))

    def test_both_launch_modes_keep_the_flag_and_budgets(self):
        for preset in ('262k','1m'):
            cfg=launch.overlay(preset,yoco_keep=140)
            resolved=launch.resolved(cfg)
            self.assertEqual(resolved['producer_flags']['--yoco-keep'],'140')
            self.assertEqual(cfg['producer_host_max_gib'],7.75)
            self.assertEqual(cfg['producer_prefill_sweep_tokens'],49152)
            self.assertEqual(cfg['producer_prefill_fold_cap_tokens'],61440)
            row=serving_config(cfg|{'host_budget_gib':9.5},runtime_root='r',studio_root='s',
                               llama_producer='p',library_directories=[],revision='test')
            self.assertEqual(row['yoco_keep'],140)
            self.assertEqual(yoco_keep(row),140)

    def test_excludes_remote_routes_and_invalid_values(self):
        for serving in ('combo','alternating','native',None):
            with self.assertRaisesRegex(ValueError,'local single_engine'):
                yoco_keep(dict(yoco_keep=140,serving_mode=serving))
        for keep in (-1,1,139,141,True,'140',None):
            with self.assertRaisesRegex(ValueError,'0, 140 or 2541'):
                yoco_keep(dict(yoco_keep=keep,serving_mode='single_engine'))

    def test_dependency_cone_has_distinct_local_identity_and_no_remote_export(self):
        from slstack.combo_producer import prefill_arguments
        from slstack.config import load
        cfg=load(overrides=dict(gpu_uuid='GPU-test',yoco_keep=2541,context=999424,ctx_cache_precision='bf16',ctx_cache_growable=True))
        self.assertEqual(yoco_keep(cfg),2541)
        args,_=prefill_arguments(cfg,4096,20*(1<<30))
        self.assertEqual(args['--yoco-keep'],'2541')
        with self.assertRaises(ValueError):prefill_arguments(cfg,4096,20*(1<<30),export_state=True)
        with self.assertRaises(ValueError):load(overrides=dict(gpu_uuid='GPU-test',yoco_keep=2541,mac='http://example.test:8180'))

    @unittest.skipUnless(shutil.which('powershell'),'Windows PowerShell launcher')
    def test_wrappers_default_exact_and_allow_approximate_webui(self):
        for preset in ('262k','1m'):
            for approximate in (False,True):
                args=['powershell','-NoProfile','-File',str(ROOT/f'scripts/start-dsv41-single-{preset}-ultra-yoco.ps1'),
                      '-DryRun','-WebUI']+(['-ApproximateYoco'] if approximate else [])
                run=subprocess.run(args,capture_output=True,text=True,timeout=30,cwd=ROOT)
                self.assertEqual(run.returncode,0,run.stderr)
                data,_=json.JSONDecoder().raw_decode(run.stdout.lstrip())
                self.assertEqual(data['resolved']['producer_flags']['--yoco-keep'],'140' if approximate else '2541')
                self.assertIn('start --webui',run.stdout)

    def test_long_live_append_and_edit_reprefill(self):
        from test_combo_single import Harness, ids
        from slstack.combo_single import EOS_ID
        h=Harness([])
        h.engine.cfg['yoco_keep']=140
        h.scripts.append(ids(h.tok,'</think>answer')+[EOS_ID])
        body=dict(messages=[dict(role='user',content='q')],max_tokens=50)
        h.run(body)
        first=h.launched[0]
        h.scripts.append([EOS_ID])
        follow=dict(messages=body['messages']+[dict(role='assistant',content='answer'),
                    dict(role='user',content='long append across the negotiated seven-token RPC capacity')])
        _,_,metrics=h.run(follow)
        self.assertEqual(len(h.launched),2)
        self.assertTrue(first.closed)
        self.assertEqual(first.prefills,[])
        self.assertEqual(metrics['reused_prefix_tokens'],0)
        second=h.launched[1]
        h.scripts.append([EOS_ID])
        h.run(dict(messages=[dict(role='user',content='edited earlier history')]))
        self.assertTrue(second.closed)
        self.assertEqual(len(h.launched),3)


if __name__=='__main__':unittest.main()
