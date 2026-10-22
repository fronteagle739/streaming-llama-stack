import unittest
import contextlib
import io

from slstack.memory_counters import PagefilePressure, pagefile_used_bytes


class PagefilePressureTests(unittest.TestCase):
    def samples(self, values):
        sequence = iter(values)
        return PagefilePressure(lambda: next(sequence))

    def test_small_external_growth_warns_without_aborting(self):
        for growth in (600_000, 15_000_000, 256 * 1024**2):
            with self.subTest(growth=growth):
                guard = self.samples([1_000_000_000, 1_000_000_000 + growth])
                with contextlib.redirect_stderr(io.StringIO()) as log:
                    self.assertIsNone(guard.check())
                self.assertIn('WARNING', log.getvalue())
                self.assertIn('not attributed', guard.receipt()['warning'])
                self.assertEqual(guard.maximum_growth, growth)

    def test_pressure_above_limit_aborts(self):
        guard = self.samples([20, 20 + 256 * 1024**2 + 1])
        self.assertIn('SYSTEM PAGEFILE PRESSURE', guard.check())

    def test_fluctuations_do_not_move_the_original_baseline(self):
        guard = self.samples([500_000_000, 400_000_000, 510_000_000, 800_000_000])
        self.assertIsNone(guard.check())
        self.assertIsNone(guard.check())
        self.assertIn('SYSTEM PAGEFILE PRESSURE', guard.check())

    def test_actual_os_sampler_is_bounded_and_nonnegative(self):
        value = pagefile_used_bytes()
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 0)


if __name__ == '__main__':
    unittest.main()
