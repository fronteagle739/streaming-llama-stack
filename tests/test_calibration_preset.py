import argparse
import os
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

from slstack.envvars import ROOT, add_flags, apply, rows


class CalibrationPresetTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(os.environ.get('SLSTACK_TEST_ROOT', ROOT / 'runtime')) / 'header-fixtures'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.directory / (uuid.uuid4().hex + '.header')
        self.file.write_bytes(b'hello')
        self.row = dict(next(row for row in rows() if row['var'] == 'DS4_CACHE_AFFINITY_SPREAD_FILE'))
        # Published FNV-1a test vector, independent of the implementation.
        self.row['model_signature'] = {'file_size': 5, 'header_fnv64': 'a430d84680aabd0b'}
        self.cfg = {'runtime_root': str(self.directory), 'model': str(self.file)}

    def tearDown(self):
        self.file.unlink()
        if not any(self.directory.iterdir()):
            self.directory.rmdir()

    def selected(self, cfg=None):
        with patch('slstack.envvars.rows', return_value=[self.row]):
            return apply({}, cfg or self.cfg, 'ds4-win')

    def test_matching_model_selects_packaged_numeric_calibration(self):
        self.assertEqual(self.selected()['DS4_CACHE_AFFINITY_SPREAD_FILE'],
                         str(ROOT / 'presets/dsv41-affinity-spreads.txt'))

    def test_same_size_other_header_does_not_select_wrong_calibration(self):
        self.file.write_bytes(b'world')
        self.assertNotIn('DS4_CACHE_AFFINITY_SPREAD_FILE', self.selected())

    def test_explicit_calibration_is_literal_and_overrides_preset_binding(self):
        cfg = {**self.cfg, 'model': str(self.directory / 'another-model'),
               'env_overrides': {'DS4_CACHE_AFFINITY_SPREAD_FILE': 'custom{calibration}.txt'}}
        self.assertEqual(self.selected(cfg)['DS4_CACHE_AFFINITY_SPREAD_FILE'], 'custom{calibration}.txt')

    def test_both_native_calibration_controls_are_cli_flags(self):
        parser = argparse.ArgumentParser()
        add_flags(parser)
        args = parser.parse_args(['--ds4-cache-affinity-spread-file', 'spreads.txt',
                                  '--ds4-cache-affinity-calibrate', 'record.txt'])
        self.assertEqual(args.env_DS4_CACHE_AFFINITY_SPREAD_FILE, 'spreads.txt')
        self.assertEqual(args.env_DS4_CACHE_AFFINITY_CALIBRATE, 'record.txt')


if __name__ == '__main__':
    unittest.main()
