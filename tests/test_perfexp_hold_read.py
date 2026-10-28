import unittest
from unittest.mock import patch
from slstack.lease import read_hold_text


class HoldReadTests(unittest.TestCase):
    def test_transient_lock_reads_current_ledger(self):
        with patch('slstack.lease.Path.read_text', side_effect=[PermissionError(), 'ACTIVE: peer new']) as read:
            with patch('slstack.lease.time.sleep') as sleep:
                self.assertEqual(read_hold_text('ledger'), 'ACTIVE: peer new')
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(.02)

    def test_persistent_lock_fails_closed_within_one_second(self):
        with patch('slstack.lease.Path.read_text', side_effect=PermissionError()) as read:
            with patch('slstack.lease.time.sleep') as sleep:
                with self.assertRaises(PermissionError):
                    read_hold_text('ledger')
        self.assertEqual(read.call_count, 51)
        self.assertEqual(sleep.call_count, 50)


if __name__ == '__main__':
    unittest.main()
