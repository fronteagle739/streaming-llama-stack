"""Window publication races must not retire a healthy engine or bypass admission."""
import json
import unittest
from unittest.mock import patch

from slstack.guard import GPUWindowReader


class GPUWindowReaderTest(unittest.TestCase):
    def test_atomic_replace_permission_race_recovers_complete_window(self):
        reader = GPUWindowReader('unused-window.json')
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
        reader = GPUWindowReader('unused-window.json')
        with patch('pathlib.Path.read_bytes', side_effect=PermissionError(13, 'Permission denied')):
            with patch('slstack.guard.time.monotonic', side_effect=[10, 10.999, 11]):
                self.assertIsNone(reader.read())
                self.assertIsNone(reader.read())
                with self.assertRaises(PermissionError):reader.read()

    def test_repeated_denials_do_not_restart_grace_period(self):
        reader = GPUWindowReader('unused-window.json')
        with patch('pathlib.Path.read_bytes', side_effect=[b'{"active":true}',
                PermissionError(13, 'Permission denied'), PermissionError(13, 'Permission denied')]):
            reader.read()
            with patch('slstack.guard.time.monotonic', side_effect=[0, 1]):
                self.assertTrue(reader.read()['active'])
                with self.assertRaises(PermissionError):reader.read()

    def test_recovery_starts_new_bounded_episode(self):
        reader = GPUWindowReader('unused-window.json')
        denied = PermissionError(13, 'Permission denied')
        with patch('pathlib.Path.read_bytes', side_effect=[denied, b'{"active":false}', denied]):
            with patch('slstack.guard.time.monotonic', side_effect=[0, 10]):
                self.assertIsNone(reader.read())
                reader.read()
                self.assertFalse(reader.read()['active'])

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
