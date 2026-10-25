"""Window publication races must not retire a healthy engine or bypass admission."""
import json
import unittest
from unittest.mock import patch

from slstack.guard import GPUWindowReader


class GPUWindowReaderTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.

    def reader(self, seconds=30):
        return GPUWindowReader('unused-window.json', seconds, clock=lambda: self.now)

    def test_atomic_replace_permission_race_recovers_complete_window(self):
        reader = self.reader()
        first = dict(active=True, lease_id='owned', deadline=100)
        second = dict(active=False, lease_id=None, deadline=101)
        with patch('pathlib.Path.read_bytes', side_effect=[json.dumps(first).encode(),
                PermissionError(13, 'Permission denied'), json.dumps(second).encode()]):
            self.assertEqual(reader.read(), first)
            self.assertEqual(reader.read(), first)  # deadline and lease are not renewed
            self.assertEqual(reader.read(), second)
        self.assertIsNone(reader.blocked_since)
        self.assertEqual(reader.permission_errors, 1)

    def test_denial_before_first_complete_window_is_bounded(self):
        reader = self.reader()
        with patch('pathlib.Path.read_bytes', side_effect=PermissionError(13, 'Permission denied')):
            for self.now in (0, 1, 10, 29.999):
                self.assertIsNone(reader.read())
            self.now = 30
            with self.assertRaises(PermissionError):reader.read()

    def test_repeated_denials_do_not_restart_grace_period(self):
        reader = self.reader()
        with patch('pathlib.Path.read_bytes', side_effect=[b'{"active":true}',
                PermissionError(13, 'Permission denied'), PermissionError(13, 'Permission denied')]):
            reader.read()
            self.now = 29.999
            self.assertTrue(reader.read()['active'])
            self.now = 30
            with self.assertRaises(PermissionError):reader.read()

    def test_recovery_starts_new_bounded_episode(self):
        reader = self.reader()
        denied = PermissionError(13, 'Permission denied')
        with patch('pathlib.Path.read_bytes', side_effect=[denied, b'{"active":false}', denied]):
            self.assertIsNone(reader.read())
            self.now = 29
            reader.read()
            self.now = 58.999
            self.assertFalse(reader.read()['active'])

    def test_successful_unchanged_reads_renew_beyond_total_window(self):
        reader = self.reader()
        with patch('pathlib.Path.read_bytes', return_value=b'{"active":true,"deadline":100}'):
            for self.now in range(0, 121, 20):
                self.assertTrue(reader.read()['active'])
        with patch('pathlib.Path.read_bytes', side_effect=PermissionError(13, 'Permission denied')):
            self.now = 149.999
            self.assertTrue(reader.read()['active'])
            self.now = 150
            with self.assertRaises(PermissionError):reader.read()

    def test_heartbeat_advance_and_custom_progress_window(self):
        reader = self.reader(45)
        with patch('pathlib.Path.read_bytes', side_effect=[b'{"deadline":100}', b'{"deadline":140}']):
            reader.read()
            self.now = 40
            reader.read()
        self.assertEqual(reader.progress.best, 140)
        with patch('pathlib.Path.read_bytes', side_effect=PermissionError(13, 'Permission denied')):
            self.now = 84.999
            self.assertEqual(reader.read()['deadline'], 140)
            self.now = 85
            with self.assertRaises(PermissionError):reader.read()

    def test_missing_corrupt_oversize_and_other_io_fail_immediately(self):
        for value, error in [(FileNotFoundError(), FileNotFoundError), (OSError(5, 'I/O'), OSError),
                             (b'{', ValueError), (b'null', ValueError), (b'[]', ValueError),
                             (b'1', ValueError), (b'"window"', ValueError), (b' ' * 16385, ValueError)]:
            with self.subTest(value=type(value).__name__):
                reader = GPUWindowReader('unused-window.json')
                with patch('pathlib.Path.read_bytes', return_value=b'{"active":true}'):
                    reader.read()
                kwargs = dict(side_effect=value) if isinstance(value, Exception) else dict(return_value=value)
                with patch('pathlib.Path.read_bytes', **kwargs):
                    with self.assertRaises(error):reader.read()


if __name__ == '__main__':unittest.main()
