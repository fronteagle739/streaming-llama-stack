"""Owner rule (no fixed timeouts): waits live while their signal shows progress and fail only
after a whole no-progress window, saying what stopped."""
import sys
import unittest
from pathlib import Path
from slstack.progress import Window, activity
from slstack.combo_release import release_barrier, suspend_barrier


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class WindowTests(unittest.TestCase):
    def test_any_change_renews_and_flat_expires(self):
        c = Clock(); w = Window(10, clock=c)
        for i in range(100): w.observe(i % 7); c.sleep(1)   # changing every second: never expires
        self.assertFalse(w.expired())
        w.observe(6); c.sleep(9.5); w.observe(6); self.assertFalse(w.expired())
        c.sleep(.6); self.assertTrue(w.expired()); self.assertAlmostEqual(w.idle(), 10.1)

    def test_rising_renews_only_on_a_new_maximum(self):
        c = Clock(); w = Window(5, clock=c)
        w.rising(10); c.sleep(4); w.rising(9); c.sleep(1.5)   # a dip is not progress
        self.assertTrue(w.expired())

    def test_window_bounds(self):
        for bad in (0, -1, 86401):
            with self.assertRaises(ValueError): Window(bad)

    def test_activity_sees_this_process_and_its_log(self):
        import os
        size, members, io, cpu = activity(os.getpid(), __file__)
        self.assertEqual(size, Path(__file__).stat().st_size); self.assertGreaterEqual(members, 1)


class _Runtime:
    def close(self): return dict(owned_job_empty=True, owned_process_signaled=True, owned_pid_gone=True)


class _Native:
    def __init__(self, clock, free): self.clock, self.free = clock, free
    def gpu(self, index, uuid): return self.free(self.clock.now)


CFG = dict(gpu_uuid='GPU-test')
G = 1 << 30


class BarrierTests(unittest.TestCase):
    def test_release_waits_past_30s_while_vram_keeps_recovering(self):
        c = Clock()   # 20 GiB freed at 0.25 GiB/s: 80 s, far past the old 30 s cap
        native = _Native(c, lambda t: min(24 * G, 4 * G + int(t * G / 4)))
        row = release_barrier(_Runtime(), native, CFG, 24 * G, lambda: False, sleep=c.sleep, window=Window(30, clock=c))
        self.assertGreaterEqual(row['gpu_free_bytes'], 24 * G - (128 << 20)); self.assertGreater(c.now, 75)   # 128 MiB desktop drift

    def test_release_refuses_after_a_flat_window_with_the_numbers(self):
        c = Clock()   # recovers for 10 s, then stuck 6 GiB short
        native = _Native(c, lambda t: 10 * G + int(min(t, 10) * G / 5))
        with self.assertRaisesRegex(RuntimeError, r'stopped recovering for 30\.\ds \(free=12884901888, baseline=19327352832'):
            release_barrier(_Runtime(), native, CFG, 18 * G, lambda: False, sleep=c.sleep, window=Window(30, clock=c))
        self.assertGreater(c.now, 39.9); self.assertLess(c.now, 40.5)

    def test_suspend_refuses_a_flat_residual(self):
        c = Clock(); native = _Native(c, lambda t: 20 * G)
        with self.assertRaisesRegex(RuntimeError, 'stopped falling'):
            suspend_barrier(_Runtime(), native, CFG, 24 * G, lambda: False, sleep=c.sleep, window=Window(30, clock=c))

    def test_cancel_still_wins(self):
        c = Clock(); native = _Native(c, lambda t: 0)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            release_barrier(_Runtime(), native, CFG, 24 * G, lambda: True, sleep=c.sleep, window=Window(30, clock=c))


if __name__ == '__main__':
    unittest.main()
