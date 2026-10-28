"""Single-engine graceful shutdown endpoint (Studio eject / switch-away). Producer mocked; no GPU."""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / 'tests'))
from slstack import combo_single as s
from test_combo_single import Harness, ids, char_tokenizer

TOKEN = 't' * 40


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = TOKEN
        self.h = Harness([ids(char_tokenizer(), 'Hey') + [s.EOS_ID]])
        cfg = dict(context=4096, lease_minutes=19, runtime_root=self.h.dir, model_name='DeepSeek-V4.1-Flash')
        self.proxy = s.SingleProxy(cfg, 'run', None, engine=self.h.engine)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(self.proxy));self.proxy.server = self.server
        self.serving = threading.Thread(target=self.server.serve_forever, kwargs=dict(poll_interval=.05), daemon=True)
        self.serving.start()
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def tearDown(self):
        if self.serving.is_alive():self.server.shutdown()
        self.server.server_close()

    def call(self, path, body, token=TOKEN):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:return json.loads(response.read())

    def test_shutdown_retires_the_producer_releases_the_hold_then_stops_serving(self):
        self.call('/v1/chat/completions', dict(messages=[dict(role='user', content='q')], thinking=False))
        producer = self.h.launched[0];self.assertFalse(producer.closed);self.assertEqual(self.h.leases, ['held'])
        self.assertEqual(self.call('/v1/stack/shutdown', {}), dict(closing=True))
        self.assertTrue(producer.closed);self.assertEqual(self.h.leases, ['held', 'released'])
        self.assertTrue(self.h.engine.stopping.is_set())
        self.serving.join(5);self.assertFalse(self.serving.is_alive())   # serve_forever returned: the process exits 0

    def test_shutdown_needs_the_bearer_token(self):
        with self.assertRaises(urllib.error.HTTPError) as refused:self.call('/v1/stack/shutdown', {}, token='x' * 40)
        self.assertEqual(refused.exception.code, 401);self.assertFalse(self.h.engine.stopping.is_set())

    def test_unconfirmed_release_keeps_serving_and_reports(self):
        self.call('/v1/chat/completions', dict(messages=[dict(role='user', content='q')], thinking=False))
        def stuck():raise RuntimeError('producer did not exit')
        self.h.launched[0].close = stuck
        with self.assertRaises(urllib.error.HTTPError) as refused:self.call('/v1/stack/shutdown', {})
        self.assertEqual(refused.exception.code, 503)
        self.assertFalse(self.h.engine.stopping.is_set());self.assertTrue(self.serving.is_alive())
        self.assertIs(self.h.engine.producer,self.h.launched[0])


if __name__ == '__main__':unittest.main()
