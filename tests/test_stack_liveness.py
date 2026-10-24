"""DS4COMBO2 liveness for the stack's native and two-machine paths (owner: no fixed timers
that a 262K prompt or a long generation would hit; memory floors stay)."""
import json, re, sys, unittest
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SL = ROOT / 'slstack'

class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self, n=-1): return json.dumps(self.payload).encode()

class Liveness(unittest.TestCase):
    def test_native_long_operations_have_no_fixed_socket_timeout(self):
        from slstack.combo_native import Native
        fake = type('F', (), {})(); fake.cfg = {'request_timeout_s': 300}; fake.url = 'http://x'; fake.token = 't'
        seen = []
        def urlopen(request, timeout=None): seen.append(timeout); return FakeResponse({'ok': True})
        with mock.patch('urllib.request.urlopen', urlopen):
            Native.http(fake, '/v1/models', timeout=2)             # short probe keeps its bound
            Native.http(fake, '/v1/quality-speed')                 # default control call: request_timeout_s
            Native.http(fake, '/decode/prefill', {'a': 1}, timeout=None)   # a 262K prefill: no fixed cap
        self.assertEqual(seen, [2, 300, None])

    def test_control_operations_use_the_unbounded_call(self):
        text = (SL / 'combo_native.py').read_text()
        self.assertIn("self.http('/decode/'+operation,row,timeout=None)", text)
        self.assertIn('stall_seconds=stall_seconds(cfg)', text)

    def test_servers_have_no_lifetime_cap(self):
        for name in ('engines.py', 'mac.py'):
            text = (SL / name).read_text()
            self.assertNotRegex(text, r"OwnedProcess\([^)]*timeout=\w*\[?'?lease_minutes", name)
            self.assertIn('unbounded=True', text)
        self.assertIn("'timeout_s': None if unbounded else", (SL / 'process.py').read_text())

    def test_requests_are_not_capped_by_lease_minutes(self):
        proxy = (SL / 'combo_proxy.py').read_text(); broker = (SL / 'native_broker.py').read_text()
        self.assertNotIn("threading.Timer(proxy.cfg['lease_minutes']*60,session.cancel.set)", proxy)
        self.assertNotIn("threading.Timer(cfg['lease_minutes'] * 60, cancel.set)", broker)
        self.assertIn('progress_watch', proxy); self.assertIn('stall_seconds=stall_seconds(cfg)', broker)
        self.assertNotIn("urlopen(request,timeout=proxy.cfg['request_timeout_s'])", proxy)

    def test_non_streaming_chat_is_not_a_fixed_request_length(self):
        from slstack import engines
        seen = []
        def urlopen(request, timeout=None): seen.append(timeout); return FakeResponse({'choices': []})
        with mock.patch('urllib.request.urlopen', urlopen):
            engines.http_json('http://x/v1/chat/completions', {'m': 1}, None)
        self.assertEqual(seen, [None])

    def test_stall_window_is_bounded_and_configurable(self):
        from slstack.combo_producer import stall_seconds
        self.assertEqual(stall_seconds({}), 600)
        with self.assertRaises(ValueError): stall_seconds({'single_engine_stall_seconds': 5})

if __name__ == '__main__':
    unittest.main()
