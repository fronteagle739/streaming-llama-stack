import json
from pathlib import Path
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch
from scripts import build_combo_runtime
from scripts.build_combo_runtime import relocate_runtime


class InstallTest(unittest.TestCase):
    def staging_args(self):
        return ['--config','config.json','--stage','stage','--install-root','installed',
            '--launcher','launcher','--studio-native','native.py','--native-baseline','baseline.json',
            '--measurements','quality.json','balanced.json','speed.json',
            '--alternating','reasoning-on.json','reasoning-off.json']

    def test_source_branch_is_required_even_with_environment_override(self):
        output=StringIO()
        with patch.dict('os.environ',{'STACK_BUILD_BRANCH':'release-source'}), \
                patch.object(build_combo_runtime,'stage') as stage, redirect_stderr(output):
            with self.assertRaises(SystemExit) as error:
                build_combo_runtime.main(self.staging_args())
        self.assertEqual(error.exception.code,2)
        self.assertIn('--source-branch',output.getvalue())
        self.assertIn('required',output.getvalue())
        stage.assert_not_called()

    def test_explicit_source_branch_reaches_staging(self):
        with patch.dict('os.environ',{'STACK_BUILD_BRANCH':'different-source'}), \
                patch.object(build_combo_runtime,'stage') as stage:
            build_combo_runtime.main(self.staging_args()+['--source-branch','release-source'])
        self.assertEqual(stage.call_args.args[0].source_branch,'release-source')

    def test_help_documents_source_branch_without_needing_inputs(self):
        output=StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as error:
            build_combo_runtime.main(['--help'])
        self.assertEqual(error.exception.code,0)
        self.assertIn('--source-branch SOURCE_BRANCH',output.getvalue())
        self.assertIn('no environment or default branch',' '.join(output.getvalue().split()))

    def test_source_branch_must_match_and_remain_owned_non_main(self):
        owned='agent'+'/'
        for current,selected in [(owned+'release',owned+'other'),('main','main'),
                (owned+'main',owned+'main'),('release','release')]:
            with self.subTest(current=current,selected=selected), \
                    patch.object(build_combo_runtime,'read',return_value={}), \
                    patch.object(build_combo_runtime,'acceptance'), \
                    patch.object(build_combo_runtime.subprocess,'check_output',return_value=current):
                args=unittest.mock.Mock(source_branch=selected)
                with self.assertRaisesRegex(ValueError,'match --source-branch'):
                    build_combo_runtime.stage(args)

    def test_matching_explicit_branch_proceeds_to_clean_tree_check(self):
        branch='agent'+'/'+'release'
        with patch.dict('os.environ',{'STACK_BUILD_BRANCH':'different-source'}), \
                patch.object(build_combo_runtime,'read',return_value={}), \
                patch.object(build_combo_runtime,'acceptance'), \
                patch.object(build_combo_runtime.subprocess,'check_output',side_effect=[branch,' M file']):
            with self.assertRaisesRegex(ValueError,'Commit and publish'):
                build_combo_runtime.stage(unittest.mock.Mock(source_branch=branch))

    def test_retired_scratch_is_not_a_runtime_dependency(self):
        scratch=Path.cwd()/'qualification-scratch'
        installed=Path.cwd()/'installed-runtime'
        external=Path.cwd()/'cuda-runtime'
        cfg=dict(runtime_root=str(scratch),host_budget_gib=7.75,
            ds4_command=[str(scratch/'candidate.exe'),'--cuda'],
            producer_library_directories=[str(scratch/'worktree/engines/native'),str(external)],
            combo_core_binding=str(scratch/'core.json'))
        original=json.dumps(cfg,sort_keys=True)
        result=relocate_runtime(cfg,installed,'qualified-source')
        self.assertNotIn(str(scratch).replace('\\','/'),json.dumps(result).replace('\\\\','/'))
        self.assertEqual(result['producer_library_directories'],[str(installed/'native'),str(external)])
        self.assertEqual(result['host_budget_gib'],8)
        self.assertEqual(result['combo_worker_host_budget_gib'],7.75)
        self.assertEqual(json.dumps(cfg,sort_keys=True),original)

    def test_profile_producers_run_their_installed_copies(self):
        scratch=Path.cwd()/'qualification-scratch'
        installed=Path.cwd()/'installed-runtime'
        cfg=dict(runtime_root=str(scratch),host_budget_gib=7.75,ds4_command=['candidate.exe'],
            producer_library_directories=[],combo_crossover_profiles=dict(
                ultra=dict(combo_crossover_tokens=3770),
                fast=dict(combo_crossover_tokens=5395,llama_producer=str(scratch/'combo-prefill-fast.exe'),producer_sha256='b'*64)))
        result=relocate_runtime(cfg,installed,'source')
        self.assertEqual(result['combo_crossover_profiles']['fast']['llama_producer'],str(installed/'combo-prefill-fast.exe'))
        self.assertNotIn('llama_producer',result['combo_crossover_profiles']['ultra'])
        self.assertEqual(cfg['combo_crossover_profiles']['fast']['llama_producer'],str(scratch/'combo-prefill-fast.exe'))

    def test_profile_qualification_is_staged_per_profile(self):
        scratch=Path.cwd()/'qualification-scratch'
        installed=Path.cwd()/'installed-runtime'
        cfg=dict(runtime_root=str(scratch),host_budget_gib=7.75,ds4_command=['candidate.exe'],
            producer_library_directories=[],combo_crossover_profiles=dict(
                fast=dict(combo_crossover_tokens=5395,llama_producer=str(scratch/'fast.exe'),producer_sha256='b'*64,
                    combo_qualification_root=str(scratch/'qual-fast'),
                    combo_native_binding=str(scratch/'qual-fast/native-parity-binding.json'))))
        row=relocate_runtime(cfg,installed,'source')['combo_crossover_profiles']['fast']
        self.assertEqual(row['combo_qualification_root'],str(installed/'qualification/fast'))
        self.assertEqual(row['combo_native_binding'],str(installed/'qualification/fast/native-parity-binding.json'))
        self.assertNotIn('combo_ultra_guard',row)

    def test_unknown_scratch_dependency_refuses_staging(self):
        scratch=Path.cwd()/'qualification-scratch'
        cfg=dict(runtime_root=str(scratch),host_budget_gib=7.75,
            ds4_command=['candidate.exe'],producer_library_directories=[],
            env_overrides=dict(UNEXPECTED_FILE=str(scratch/'unpacked-file')))
        with self.assertRaisesRegex(ValueError,'still depends'):
            relocate_runtime(cfg,Path.cwd()/'installed','source')
