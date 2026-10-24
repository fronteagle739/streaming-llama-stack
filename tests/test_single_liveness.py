"""Progress-based liveness on the single-engine path (owner: no fixed wall-clock limit that a
262K-token prompt or a long generation would hit). Simulated clocks; no GPU."""
import json, os, tempfile, threading, time, unittest
from pathlib import Path
from slstack.native_broker import vendor
vendor()
from studio.dsv41_session_process import operation_live, Producer
from slstack.combo_producer import stall_seconds
from slstack.combo_single import SingleProxy

CONTEXT = 262144
STALL = 600.0
OLD_CAP = 1140.0   # the fixed limit this replaces (min(1140, request_timeout_s))


def simulate(duration, tick, stall_at=None, deadline_after=OLD_CAP):
    """Walk a simulated operation second by second; progress every `tick` s until stall_at.
    Returns the second the watchdog would stop it, or None if it completes."""
    start = 0.0;deadline = start + deadline_after;last = start
    t = 0.0
    while t <= duration:
        if (stall_at is None or t < stall_at) and t - last >= tick:last = t
        if not operation_live(t, deadline, last, STALL):return t
        t += 1.0
    return None


class LivenessTest(unittest.TestCase):
    def test_262k_prefill_at_a_slow_rate_is_not_cut_off(self):
        # ULTRA measured 13.63 s + N/2064 tok/s; take a 10x slower worst case: 206 tok/s.
        duration = 13.63 + CONTEXT / 206.4
        self.assertGreater(duration, OLD_CAP)                      # the old fixed cap would kill it
        self.assertFalse(operation_live(OLD_CAP + 1, OLD_CAP, 0.0, None))
        self.assertIsNone(simulate(duration, tick=5))               # progress every 5 s: completes

    def test_long_generation_is_not_cut_off(self):
        self.assertIsNone(simulate(20000 / 6.0, tick=1))           # 20K tokens at 6 tok/s, ~56 min

    def test_a_stalled_operation_still_stops(self):
        stopped = simulate(5000, tick=5, stall_at=1500)
        self.assertIsNotNone(stopped)
        self.assertLessEqual(stopped, 1500 + STALL + 1)

    def test_silent_from_the_start_stops_at_the_silence_bound(self):
        self.assertEqual(simulate(5000, tick=10**9, deadline_after=OLD_CAP), OLD_CAP)

    def test_idle_between_operations_keeps_its_idle_rule(self):
        self.assertTrue(operation_live(10**6, None, 0.0, STALL))

    def test_stall_window_bounds(self):
        self.assertEqual(stall_seconds({}), 600.0)
        with self.assertRaises(ValueError):stall_seconds({'single_engine_stall_seconds': 5})

    def test_tree_cpu_time_counts_as_progress(self):
        p = Producer.__new__(Producer)
        p.job = type('J', (), {'pid': os.getpid()})();p._progress_mark = None;p.last_progress = 0.0
        p._note_tree_progress();first = p.last_progress
        self.assertGreater(first, 0.0)
        end = time.process_time() + .05
        while time.process_time() < end:pass                        # CPU work, no pipe output
        p._note_tree_progress()
        self.assertGreater(p.last_progress, first)


class FakeEngine:
    poison = None;lease_id = 'lease';producer = None


class HeartbeatTest(unittest.TestCase):
    def setUp(self):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = 'x' * 40
        self.dir = tempfile.TemporaryDirectory();self.window = Path(self.dir.name) / 'w.json'
        self.proxy = SingleProxy({'context': CONTEXT, 'model_name': 'm'}, 'run', self.window, engine=FakeEngine())
        self.proxy.stall = 2.0;self.proxy.beat_seconds = .05

    def tearDown(self):self.dir.cleanup()

    def test_window_is_renewed_while_progressing_and_keepalive_is_sent(self):
        beats = []
        self.proxy.window(True, 'lease');first = json.loads(self.window.read_text())['deadline']
        with self.proxy.heartbeat(lambda: beats.append(1)):time.sleep(.4)
        self.assertGreater(json.loads(self.window.read_text())['deadline'], first)
        self.assertGreater(len(beats), 2)

    def test_window_lapses_when_the_producer_stalls(self):
        process = type('P', (), {})();process.last_progress = time.monotonic() - 10
        self.proxy.engine.producer = type('C', (), {'process': process})()
        self.proxy.window(True, 'lease');first = json.loads(self.window.read_text())['deadline']
        beats = []
        with self.proxy.heartbeat(lambda: beats.append(1)):time.sleep(.3)
        self.assertEqual(json.loads(self.window.read_text())['deadline'], first)
        self.assertEqual(beats, [])

    def test_relaunch_inside_a_request_rewrites_the_window_with_the_new_lease(self):
        # GUI run 5d: retire + relaunch in one request left the old lease id in the window and
        # the guardian stopped the server for a 'foreign' hold that was our own new lease.
        from slstack.combo_single import SingleEngine
        engine = self.proxy.engine
        self.proxy.request_active = True;self.proxy.window(True, 'old')
        engine.lease_id = 'new';SingleEngine.lease_changed(engine)
        self.assertEqual(json.loads(self.window.read_text())['lease_id'], 'new')
        self.proxy.request_active = False;engine.lease_id = 'later';SingleEngine.lease_changed(engine)
        self.assertEqual(json.loads(self.window.read_text())['lease_id'], 'new')   # idle: untouched

    def test_262k_prompt_body_fits_the_context_sized_bound(self):
        body = json.dumps({'messages': [{'role': 'user', 'content': 'word ' * CONTEXT}]}).encode()
        self.assertGreater(len(body), 1 << 20)
        self.assertLessEqual(len(body), max(1 << 20, CONTEXT * 32))


if __name__ == '__main__':unittest.main()
