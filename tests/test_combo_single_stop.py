"""Single-engine client stop sequences and client disconnect (producer mocked; no GPU).

The stepping producer mirrors the live-session.inc decode loop: one STEP line per
evaluated token, and a cancel_decode() request stops it at the next token
boundary exactly as at max_tokens (the last returned token is not evaluated)."""
import json
import os
import socket
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / 'tests'))
from slstack import ds4_render as r
from slstack import combo_single as s
from test_combo_single import FakeProducer, Harness, ids, char_tokenizer


class SteppingProducer(FakeProducer):
    step_seconds = .02

    def __init__(self, *args):
        super().__init__(*args);self.cancel = threading.Event();self.cancels = 0;self.decodes = []

    def cancel_decode(self):
        self.cancels += 1;self.cancel.set()

    def decode(self, max_tokens, stop_tokens=()):
        self.cancel.clear()   # the producer clears the cancel file at every decode start
        script = list(self.script.pop(0));out = [];cancelled = False
        for index, token in enumerate(script):
            out.append(token)
            if token in stop_tokens or len(out) == max_tokens:break
            if self.cancel.is_set():cancelled = True;break
            with self.process.state_lock:
                self.process.diagnostic.extend(b'LLAMADECODE_STEP index=%d token=%d position=1 step_ms=1\n' % (index, token))
            self.position += 1;time.sleep(self.step_seconds)
        self.decodes.append(out);evaluated = len(out) - 1
        return dict(tokens=out, stopped=bool(out) and out[-1] in stop_tokens, cancelled=cancelled, evaluated_tokens=evaluated,
                    step_ms=[1.0] * evaluated, decode_ms=float(evaluated))


def stepping_harness(scripts):
    h = Harness(scripts)
    def factory(tokens):
        producer = SteppingProducer(tokens, h.scripts, h.dir);h.launched.append(producer);return producer
    h.engine.producer_factory = factory
    return h


class StopMatcherTests(unittest.TestCase):
    def test_holdback_flush_and_cut(self):
        m = s.StopMatcher(['abc'])
        self.assertEqual([m.feed(b) for b in (b'x', b'a', b'b')], ['', '', 'x'])   # max_len-1 bytes held
        self.assertEqual(m.feed(b'd'), 'a');self.assertIsNone(m.hit);self.assertEqual(m.flush(), 'bd')
        m = s.StopMatcher(['abc'])
        self.assertEqual(m.feed(b'zza') + m.feed(b'bc tail'), 'zz');self.assertEqual(m.hit, 2);self.assertEqual(m.final_text(), 'zz')
        self.assertEqual(m.feed(b'more'), '');self.assertEqual(m.flush(), '')

    def test_earliest_match_wins_and_empty_stops_ignored(self):
        m = s.StopMatcher(['c', 'ab', ''])
        self.assertEqual(m.feed(b'xabc'), 'x');self.assertEqual(m.final_text(), 'x')
        self.assertEqual(r.parse_stop(['', 'a', 3, None, 'b\0c']), ['a', 'b']);self.assertEqual(r.parse_stop('x'), ['x'])
        self.assertEqual(r.parse_stop(7), []);self.assertEqual(r.parse_request(dict(messages=[], stop='Q'))['stops'], ['Q'])

    def test_split_utf8_character_is_held(self):
        m = s.StopMatcher(['Q'])
        self.assertEqual(m.feed(b'\xc3'), '');self.assertEqual(m.feed(b'\xa9'), 'é')


class EngineStopTests(unittest.TestCase):
    body = dict(messages=[dict(role='user', content='q')], thinking=False, max_tokens=200)

    def test_stop_sequence_cancels_the_decode_and_trims(self):
        h = stepping_harness([])
        h.scripts.append(ids(h.tok, 'Hello STOP world and a long tail that must never be decoded') + [s.EOS_ID])
        deltas, final, metrics = h.run(dict(self.body, stop=['STOP']))
        producer = h.launched[0]
        self.assertEqual(final['parsed']['content'].rstrip(), 'Hello');self.assertEqual(final['parsed']['finish'], 'stop')
        self.assertEqual(''.join(t for f, t in deltas if f == 'content'), 'Hello')
        self.assertEqual(producer.cancels, 1);self.assertEqual(len(producer.decodes), 1)
        self.assertLess(len(producer.decodes[0]), 20)   # stopped at a token boundary, not max_tokens/EOS
        self.assertEqual(metrics['completion_tokens'], len('Hello STOP'));self.assertEqual(metrics['decode_ended'], 'stop sequence')
        self.assertEqual(len(h.engine.live), producer.position)   # frontier exact: evaluated tokens only
        self.assertEqual(h.engine.live[-len(producer.decodes[0]) + 1:], producer.decodes[0][:-1])

    def test_stop_on_the_unevaluated_token_at_a_call_limit(self):
        h = stepping_harness([]);limit = s.MAX_DECODE_CALL
        try:
            s.MAX_DECODE_CALL = 4
            h.scripts.extend([ids(h.tok, 'abcd'), ids(h.tok, 'efgh')])
            _, final, metrics = h.run(dict(self.body, stop='cd'))
        finally:s.MAX_DECODE_CALL = limit
        producer = h.launched[0]
        self.assertEqual(final['parsed']['content'], 'ab');self.assertEqual(final['parsed']['finish'], 'stop')
        self.assertEqual(producer.prefills, []);self.assertEqual(len(producer.decodes), 1)   # 'd' never evaluated, no next call
        self.assertEqual(len(h.engine.live), producer.position)

    def test_no_stop_hit_flushes_the_held_tail(self):
        h = stepping_harness([ids(char_tokenizer(), 'abcxy') + [s.EOS_ID]])
        deltas, final, _ = h.run(dict(self.body, stop=['xyz']))
        self.assertEqual(final['parsed']['content'], 'abcxy');self.assertEqual(final['parsed']['finish'], 'stop')
        self.assertEqual(''.join(t for f, t in deltas if f == 'content'), 'abcxy');self.assertEqual(h.launched[0].cancels, 0)

    def test_client_gone_stops_at_the_next_boundary_and_stays_resident(self):
        h = stepping_harness([]);gone = threading.Event();seen = []
        h.scripts.append(ids(h.tok, 'x' * 150) + [s.EOS_ID])
        def emit(kind, *values):
            if kind == 'delta':
                seen.append(values)
                if len(seen) == 5:gone.set()
        metrics = {}
        h.engine.run(dict(self.body), emit, metrics, client_gone=gone.is_set)
        producer = h.launched[0]
        self.assertEqual(producer.cancels, 1);self.assertLess(len(producer.decodes[0]), 20)
        self.assertEqual(metrics['decode_ended'], 'client disconnected');self.assertEqual(len(h.engine.live), producer.position)
        self.assertFalse(producer.closed);self.assertIs(h.engine.producer, producer)
        # The next turn on the same producer still works.
        h.scripts.append(ids(h.tok, 'ok') + [s.EOS_ID])
        _, final, _ = h.run(dict(self.body, messages=[dict(role='user', content='q2')]))
        self.assertEqual(final['parsed']['content'], 'ok')


class HttpDisconnectTests(unittest.TestCase):
    def test_closed_stream_cancels_the_decode(self):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = 't' * 40
        h = stepping_harness([ids(char_tokenizer(), 'y' * 400) + [s.EOS_ID]])
        cfg = dict(context=4096, lease_minutes=19, runtime_root=h.dir, model_name='DeepSeek-V4.1-Flash')
        proxy = s.SingleProxy(cfg, 'run', None, engine=h.engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(proxy))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            body = json.dumps(dict(messages=[dict(role='user', content='q')], stream=True, thinking=False, max_tokens=1000)).encode()
            client = socket.create_connection(server.server_address)
            client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer ' + b't' * 40 +
                           b'\r\nContent-Type: application/json\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
            data = b''
            while data.count(b'"content"') < 3:data += client.recv(4096)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b'\x01\x00\x00\x00\x00\x00\x00\x00');client.close()   # RST
            deadline = time.monotonic() + 20
            while (not h.launched or not h.launched[0].decodes or h.engine.lock.locked()) and time.monotonic() < deadline:time.sleep(.05)
            producer = h.launched[0]
            self.assertEqual(producer.cancels, 1);self.assertLess(len(producer.decodes[0]), 200)
            self.assertEqual(len(h.engine.live), producer.position);self.assertFalse(producer.closed)
            self.assertFalse(h.engine.lock.locked())
        finally:server.shutdown();server.server_close()


if __name__ == '__main__':unittest.main()
