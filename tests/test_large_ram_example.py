"""CPU-only config/argv checks for the explicitly blocked large-RAM recipe."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/dual-machine/dsv41-large-ram-untested'
spec = importlib.util.spec_from_file_location('large_ram_example', EXAMPLE / 'launch.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LargeRamExample(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((EXAMPLE / 'config.example.json').read_text())

    def test_preview_uses_existing_ultra_yoco_export_and_broker_argv(self):
        from slstack.combo_producer import prefill_arguments
        paired = json.loads((ROOT / 'examples/dual-machine/dsv41-ultra-yoco/preset.json').read_text())['rtx']
        for key in paired:
            if key not in ('model', 'runtime_root', 'mac', 'pair_token'):
                self.assertEqual(self.cfg['rtx'][key], paired[key], key)
        result = launcher.plan('rtx', self.cfg)
        self.assertFalse(result['launchable'])
        self.assertEqual(result['status'], 'UNTESTED example')
        self.assertEqual(result['proposed_argv'][-3:],
                         ['--config', str(Path('runtime/dsv41-large-ram-untested') / 'rtx.local.json'), 'rtx'])
        self.assertEqual(Path(result['proposed_argv'][2]), ROOT / 'scripts/bootstrap.py')
        flags, _ = prefill_arguments(self.cfg['rtx'], 49787, int(12.5 * 2**30), export_state=True)
        self.assertEqual(result['state_export']['flags'], flags)
        for key, value in {'--capture-dspark': '1', '--yoco-keep': '140', '--yoco-remote-state-version': '1'}.items():
            self.assertEqual(flags[key], value)

    def test_preview_starts_nothing_writes_nothing_and_redacts_both_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = copy.deepcopy(self.cfg)
            runtime = Path(folder) / 'must-not-exist'
            cfg['rtx']['runtime_root'] = str(runtime)
            for machine in ('rtx', 'decode_plan'):
                cfg[machine]['pair_token'] = 'private-test-pairing-value'
            local = Path(folder) / 'large-ram.local.json'
            local.write_text(json.dumps(cfg))
            for machine in ('rtx', 'decode'):
                out = io.StringIO()
                with patch('subprocess.Popen', side_effect=AssertionError('must not start')):
                    with contextlib.redirect_stdout(out):
                        self.assertEqual(launcher.main(['--machine', machine, '--config', str(local), '--dry-run']), 0)
                self.assertNotIn('private-test-pairing-value', out.getvalue())
                result = json.loads(out.getvalue())
                self.assertFalse(result['launchable'])
                self.assertEqual(result['config']['pair_token'], '<redacted>')
                if machine == 'decode':
                    self.assertIsNone(result['proposed_argv'])
                    self.assertEqual(result['config']['backend'], 'cpu')
                    self.assertEqual(result['config']['host_budget_gib'], 600)
                self.assertFalse(runtime.exists())

    def test_launch_refused_even_with_resolved_settings(self):
        for machine in ('rtx', 'decode'):
            with patch.object(Path, 'read_text', side_effect=AssertionError('must refuse before reading')):
                with patch('subprocess.Popen', side_effect=AssertionError('must not start')):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        launcher.main(['--machine', machine])
            self.assertEqual(error.exception.code, 2)

    def test_invalid_budget_and_incompatible_export_are_refused(self):
        for section, key, value in [('rtx', 'yoco_keep', 0), ('decode_plan', 'native_state_mode', 1),
                                    ('rtx', 'host_budget_gib', float('inf')),
                                    ('decode_plan', 'host_budget_gib', -1),
                                    ('decode_plan', 'min_available_gib', float('nan')),
                                    ('rtx', 'runtime_min_free_gib', 0.25)]:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                launcher.plan('rtx', cfg)


if __name__ == '__main__':
    unittest.main()
