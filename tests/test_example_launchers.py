"""Example dry runs must be safe and preserve the selected execution mode."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('examples_launcher',ROOT/'examples/run_preset.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class Launchers(unittest.TestCase):
    @unittest.skipUnless(shutil.which('powershell'),'PowerShell is needed to execute Windows examples')
    def test_all_single_machine_scripts_are_read_only_dry_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime=Path(folder)/'must-not-be-created'
            local=Path(folder)/'test.local.json'
            local.write_text(json.dumps(dict(model='model.gguf',runtime_root=str(runtime),webui=True,
                                            producer_profile='accurate',yoco_keep=140,api_key='example-placeholder')))
            for file in sorted((ROOT/'examples/single-machine').glob('*.ps1')):
                with self.subTest(script=file.name):
                    run=subprocess.run(['powershell','-NoProfile','-File',str(file),'-Config',str(local),'-DryRun'],
                                       capture_output=True,text=True,timeout=30)
                    self.assertEqual(run.returncode,0,run.stderr)
                    result=json.loads(run.stdout);cfg=result['config']
                    profile=file.stem.removeprefix('dsv41-').removesuffix('-webui')
                    cone=profile in ('ultra-cone','ultra-yoco')
                    self.assertEqual(cfg['producer_profile'],'ultra' if cone else profile)
                    self.assertEqual(cfg['yoco_keep'],2541 if cone else 0)
                    self.assertEqual(cfg['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'96' if cone else '0')
                    self.assertEqual(cfg['webui'],file.stem.endswith('-webui'))
                    self.assertNotIn('example-placeholder',run.stdout)
                    self.assertFalse(runtime.exists())

    @unittest.skipUnless(shutil.which('powershell'),'PowerShell is needed to execute Windows examples')
    def test_legacy_yoco_launchers_require_explicit_approximation(self):
        with tempfile.TemporaryDirectory() as folder:
            local=Path(folder)/'test.local.json';local.write_text('{}')
            for file in sorted((ROOT/'examples/single-machine').glob('*ultra-yoco*.ps1')):
                run=subprocess.run(['powershell','-NoProfile','-File',str(file),'-Config',str(local),
                                    '-ApproximateYoco','-Context','1m','-DryRun'],capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stderr)
                cfg=json.loads(run.stdout)['config']
                self.assertEqual((cfg['producer_profile'],cfg['yoco_keep']),('ultra',140))

    def test_every_dual_recipe_refuses_unresolved_settings_before_start(self):
        with tempfile.TemporaryDirectory() as folder:
            local=Path(folder)/'empty.local.json';local.write_text('{}')
            for recipe in sorted((ROOT/'examples/dual-machine').glob('*/preset.json')):
                for machine in ('mac','rtx'):
                    with self.subTest(preset=recipe.parent.name,machine=machine):
                        args=['--preset',recipe.parent.name,'--machine',machine,'--config',str(local)]
                        with patch.object(launcher.subprocess,'call',side_effect=AssertionError('must not start')):
                            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                                launcher.main(args)
                            self.assertEqual(exc.exception.code,2)
                            out=io.StringIO()
                            with contextlib.redirect_stdout(out):self.assertEqual(launcher.main(args+['--dry-run']),0)
                            self.assertTrue(json.loads(out.getvalue())['missing_local_settings'])


if __name__=='__main__':unittest.main()
