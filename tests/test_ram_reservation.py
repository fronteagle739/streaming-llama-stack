import unittest
from unittest.mock import patch
from slstack.lease import hold_admission, HoldWatcher


CPU = ('ACTIVE: BUILD reservation; host budget 3 GiB; physical RAM reserve 6 GiB; '
       'no GPU or Studio restart requested.')


class ReservationTests(unittest.TestCase):
    def test_full_budget_and_reservation_fit_before_admission(self):
        self.assertEqual(hold_admission([CPU], budget_gib=3.5, floor_gib=6,
                                       available=int(14.5 * 2**30)), [])
        self.assertTrue(hold_admission([CPU], budget_gib=3.5, floor_gib=6,
                                      available=int(14.49 * 2**30)))

    def test_ambiguous_or_gpu_hold_never_exempted(self):
        for line in [CPU.replace('no GPU or Studio restart requested.', ''),
                     CPU + ' GPU hold', 'ACTIVE: PRIORITY work; exclusive GPU']:
            self.assertEqual(hold_admission([line], available=100 * 2**30), [line])

    def test_reservations_add_and_physical_pressure_is_rechecked(self):
        self.assertTrue(hold_admission([CPU, CPU], available=19 * 2**30))
        watcher = HoldWatcher.__new__(HoldWatcher)
        watcher.path = unittest.mock.Mock()
        watcher.path.stat.return_value.st_mtime_ns = 1
        watcher.path.stat.return_value.st_size = 2
        watcher.stamp = (1, 2)
        watcher.foreign, watcher.mac = [CPU], False
        with patch('slstack.lease.psutil.virtual_memory') as memory:
            memory.return_value.available = 12 * 2**30
            self.assertIsNone(watcher.reason())
            memory.return_value.available = 10 * 2**30
            self.assertIn('CPU reservations', watcher.reason())


if __name__ == '__main__':
    unittest.main()
