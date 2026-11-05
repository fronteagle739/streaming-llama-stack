import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dsv41_ram_profile',
    Path(__file__).resolve().parents[1] / 'scripts/dsv41_100k_ram_profile.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Profile(unittest.TestCase):
    def test_launcher_defaults_admit_100k_and_retain_explicit_features(self):
        from scripts.dsv41_single import overlay
        from slstack.combo_producer import prefill_arguments
        for preset in ('262k', '1m'):
            cfg = overlay(preset)
            self.assertEqual(cfg['host_budget_gib'], 10.25)
            self.assertEqual(cfg['min_available_gib'], 4)
            self.assertEqual(cfg['owned_paging_limit_bytes'],0)
            _, plan = prefill_arguments(cfg, 100000, 15 << 30)
            self.assertEqual(plan['sweeps'], [100000])
        local = dict(producer_host_max_gib='auto', producer_prefill_sweep_tokens=49152,
                     producer_prefill_fold_cap_tokens=61440, host_budget_gib=12,
                     producer_host_reserve_gib=5, producer_keep_alive=True,
                     unknown_future_option={'keep': [1]}, temperature=.7,
                     env_overrides={'LLAMA_DSV41_INDEX_FP4': '0', 'DS4_DS_READ_PARTS': '4',
                                    'LLAMADECODE_VRAM_MARGIN_MIB': '512'})
        cfg = overlay('1m', local)
        for key, value in local.items():
            if key != 'env_overrides':
                self.assertEqual(cfg[key], value)
        self.assertEqual(cfg['env_overrides']['LLAMA_DSV41_INDEX_FP4'], '0')
        self.assertEqual(cfg['env_overrides']['DS4_DS_READ_PARTS'], '4')
        self.assertEqual(cfg['env_overrides']['LLAMADECODE_VRAM_MARGIN_MIB'], '512')
        from slstack.single_runtime import serving_config
        serving = serving_config(cfg, runtime_root='r', studio_root='s',
            llama_producer='p', library_directories=[], revision='test')
        self.assertIs(serving['producer_keep_alive'], True)
        # The native planner must reject an unsupported combination, not drop it.
        with self.assertRaisesRegex(ValueError, 'producer_keep_alive'):
            prefill_arguments(serving, 100000, 20 << 30)

    def test_benchmark_defaults_precede_generic_defaults_and_preserve_file(self):
        from unittest.mock import patch
        import tempfile, json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'local.json'
            local = dict(model='x.gguf', producer_host_max_gib='auto', seed=17,
                         env_overrides={'FUTURE_FEATURE': '1'})
            path.write_text(json.dumps(local))
            before = path.read_bytes()
            with patch('slstack.config.load', side_effect=lambda path, overrides: {**local, **overrides}) as loader:
                cfg = module.load_performance_config(path)
            self.assertNotIn('model', loader.call_args.kwargs['overrides'])
            self.assertEqual(cfg['host_budget_gib'], 10.25)
            self.assertEqual(cfg['producer_host_max_gib'], 'auto')
            self.assertEqual(cfg['seed'], 17)
            self.assertEqual(cfg['env_overrides'], {**module.PERFORMANCE_ENV_DEFAULTS,**local['env_overrides']})
            self.assertEqual(path.read_bytes(), before)

    def config(self):
        return dict(context=1000001, producer_profile='ultra', yoco_keep=2541,
            host_budget_gib=7, producer_host_budget_gib=6.25,
            producer_host_max_gib=6.25, combo_worker_host_budget_gib=6.75,
            min_available_gib=4, runtime_min_free_gib=4,
            producer_desktop_reserve_gib=4, producer_host_reserve_gib=4,
            producer_prefill_sweep_tokens='auto', producer_graph_floor_gib=3,
            ctx_cache_precision='bf16', ctx_cache_growable=True,
            env_overrides={'LLAMA_DSV41_FLASHINFER_MXFP4': '1',
                           'LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL': 'x.dll'},
            direction_bank_count=1, unknown_future_option={'keep': [1, 2, 3]})

    def test_admits_single_sweep_preserving_every_nonbudget_field(self):
        original = self.config()
        output, plan = module.profile(original, 15 << 30)
        self.assertEqual(plan['sweeps'], [100000])
        changed = {k for k in output if output[k] != original.get(k)}
        self.assertEqual(changed, {'host_budget_gib', 'producer_host_max_gib',
                                   'combo_worker_host_budget_gib'})
        self.assertEqual(original['host_budget_gib'], 7)
        output['unknown_future_option']['keep'].append(4)
        self.assertEqual(original['unknown_future_option']['keep'], [1, 2, 3])

    def test_insufficient_physical_memory_fails_loudly(self):
        with self.assertRaisesRegex(MemoryError, 'unchanged physical reserve'):
            module.profile(self.config(), 12 << 30)

    def test_existing_larger_limits_and_auto_are_retained(self):
        cfg = dict(self.config(), host_budget_gib=12,
                   combo_worker_host_budget_gib=11.5, producer_host_max_gib='auto')
        output, plan = module.profile(cfg, 16 << 30)
        self.assertEqual(output['host_budget_gib'], 12)
        self.assertEqual(output['producer_host_max_gib'], 'auto')
        self.assertEqual(output['combo_worker_host_budget_gib'], 11.5)
        self.assertEqual(plan['sweeps'], [100000])

    def test_other_profile_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ULTRA'):
            module.profile(dict(self.config(), producer_profile='accurate'), 15 << 30)

    def test_200k_keeps_all_engram_rows_with_same_sweeps(self):
        from slstack.combo_producer import prefill_arguments
        cfg=module.performance_defaults(dict(context=1000001, producer_profile='ultra', yoco_keep=2541))
        _, good=prefill_arguments(cfg,199952,16<<30)
        _, refused=prefill_arguments(dict(cfg,host_budget_gib=8.5),199952,16<<30)
        self.assertEqual(good['sweeps'],[102400,97552])
        self.assertEqual(refused['sweeps'],good['sweeps'])
        self.assertGreater(good['whole_prompt_engram_bytes'],0)
        self.assertEqual(refused['whole_prompt_engram_bytes'],0)
        self.assertIn('host_budget_gib',refused['whole_prompt_engram_refused'])
        self.assertLess(good['host_bytes']+(1<<29),cfg['host_budget_gib']*(1<<30))
        _, proof=module.profile(self.config(),16<<30,199952)
        self.assertGreater(proof['whole_prompt_engram_bytes'],0)

    def test_flags_override_config_without_losing_other_features(self):
        import argparse
        parser=argparse.ArgumentParser()
        module.add_performance_flags(parser)
        args=parser.parse_args(['--host-budget-gib','11','--producer-host-max-gib','auto',
                                '--producer-prefill-sweep-tokens','49152',
                                '--llamadecode-engram-scan-rings','4',
                                '--llamadecode-engram-whole-prompt','0'])
        cfg=self.config()
        updated=module.apply_performance_flags(cfg,args)
        self.assertEqual(updated['host_budget_gib'],11)
        self.assertEqual(updated['combo_worker_host_budget_gib'],10.75)
        self.assertEqual(updated['producer_host_max_gib'],'auto')
        self.assertEqual(updated['producer_prefill_sweep_tokens'],49152)
        self.assertEqual(updated['env_overrides']['LLAMADECODE_ENGRAM_SCAN_RINGS'],'4')
        self.assertEqual(updated['env_overrides']['LLAMADECODE_ENGRAM_WHOLE_PROMPT'],'0')
        self.assertEqual(updated['direction_bank_count'],cfg['direction_bank_count'])
        self.assertEqual(updated['unknown_future_option'],cfg['unknown_future_option'])
        self.assertNotIn('LLAMADECODE_ENGRAM_SCAN_RINGS',cfg['env_overrides'])

    def test_engram_queue_policy_preserves_explicit_options_and_rejects_conflicts(self):
        from slstack.combo_producer import engram_scan_rings
        cfg=dict(producer_engram_scan_rings='auto')
        self.assertEqual(engram_scan_rings(cfg,dict(single_sweep=True)),'4')
        self.assertEqual(engram_scan_rings(cfg,dict(single_sweep=False)),'1')
        explicit=dict(cfg,env_overrides={'LLAMADECODE_ENGRAM_SCAN_RINGS':'8'})
        self.assertEqual(engram_scan_rings(explicit,dict(single_sweep=True)),'8')
        self.assertEqual(engram_scan_rings(dict(producer_engram_scan_rings=2),{}),'2')
        with self.assertRaisesRegex(ValueError,'Conflicting explicit'):
            engram_scan_rings(dict(explicit,producer_engram_scan_rings=2),{})
        with self.assertRaisesRegex(ValueError,'integer1..8'):
            engram_scan_rings(dict(producer_engram_scan_rings=0),{})


if __name__ == '__main__':
    unittest.main()
