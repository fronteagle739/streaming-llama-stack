import json
from pathlib import Path
import unittest
from scripts.build_combo_runtime import relocate_runtime


class InstallTest(unittest.TestCase):
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

    def test_unknown_scratch_dependency_refuses_staging(self):
        scratch=Path.cwd()/'qualification-scratch'
        cfg=dict(runtime_root=str(scratch),host_budget_gib=7.75,
            ds4_command=['candidate.exe'],producer_library_directories=[],
            env_overrides=dict(UNEXPECTED_FILE=str(scratch/'unpacked-file')))
        with self.assertRaisesRegex(ValueError,'still depends'):
            relocate_runtime(cfg,Path.cwd()/'installed','source')
