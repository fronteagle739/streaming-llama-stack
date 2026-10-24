"""Graceful abort refuses at operation boundaries; shutdown/close stay allowed (CPU only)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slstack import combo_abort
from slstack.combo_native import Native
from slstack.combo_producer import ComboProducer


class Abort(unittest.TestCase):
    def test_no_file_no_abort(self):
        with mock.patch.dict(os.environ, {'DS4COMBO_ABORT_FILE': ''}):
            combo_abort.check('x')

    def test_file_aborts(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / 'run.abort'; f.write_text('{}')
            with mock.patch.dict(os.environ, {'DS4COMBO_ABORT_FILE': str(f)}):
                with self.assertRaises(combo_abort.AbortRequested): combo_abort.check('x')

    def test_native_shutdown_still_allowed_but_other_ops_refused(self):
        n = Native.__new__(Native)
        n.control_lock = mock.MagicMock(); n.cancelled = lambda: True
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / 'run.abort'; f.write_text('{}')
            with mock.patch.dict(os.environ, {'DS4COMBO_ABORT_FILE': str(f)}):
                with self.assertRaises(combo_abort.AbortRequested): n.control('render')
                # shutdown passes the abort gate (it then meets the existing cancel check)
                with self.assertRaises(RuntimeError) as e: n.control('shutdown')
                self.assertNotIsInstance(e.exception, combo_abort.AbortRequested)

    def test_producer_close_command_allowed(self):
        p = ComboProducer.__new__(ComboProducer)
        p.session = 'a' * 32; p.process = mock.MagicMock(sequence=3)
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / 'run.abort'; f.write_text('{}')
            with mock.patch.dict(os.environ, {'DS4COMBO_ABORT_FILE': str(f)}), \
                 mock.patch.dict('sys.modules', {'studio.dsv41_session_contract': mock.MagicMock(MODEL_SHA256='m')}):
                self.assertEqual(p.command('close')['operation'], 'close')
                with self.assertRaises(combo_abort.AbortRequested): p.command('prefill')


if __name__ == '__main__':
    unittest.main()
