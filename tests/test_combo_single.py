"""Single-engine serving mode (slstack.combo_single) with a fake producer: no GPU."""
import contextlib
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import ds4_render as r
from slstack import combo_single as s
from test_ds4_render import char_tokenizer


class FakeProcess:
    def __init__(self):
        self.diagnostic = bytearray();self.state_lock = threading.Lock();self.ready = dict(maximum_prefill_tokens=7)


class FakeProducer:
    """Records the session ops; decode replays scripted token lists."""
    def __init__(self, tokens, script, directory):
        self.process = FakeProcess();self.position = len(tokens);self.prefills = [];self.qualities = []
        self.script = script;self.directory = directory;self.closed = False;self.launch = list(tokens)
    @property
    def checkpoint(self):return dict(position=self.position)
    def prefill(self, tokens):
        assert 0 < len(tokens) <= 7;self.prefills.append(list(tokens));self.position += len(tokens)
    def quality(self, value):self.qualities.append(value);return dict(value=value)
    def decode(self, max_tokens, stop_tokens=()):
        tokens = list(self.script.pop(0))[:max_tokens]
        stopped = bool(tokens) and tokens[-1] in stop_tokens
        for i, t in enumerate(tokens[:-1]):
            with self.process.state_lock:self.process.diagnostic.extend(b'LLAMADECODE_STEP index=%d token=%d position=1 step_ms=1\n' % (i, t))
        self.position += max(0, len(tokens) - 1)
        return dict(tokens=tokens, stopped=stopped, evaluated_tokens=max(0, len(tokens) - 1), step_ms=[1.0] * max(0, len(tokens) - 1), decode_ms=float(max(0, len(tokens) - 1)))
    def close(self):self.closed = True;return dict(certified=True)


def ids(tok, text):return tok.rendered(text)


class Harness:
    def __init__(self, scripts):
        self.tok = char_tokenizer();self.dir = tempfile.mkdtemp();self.launched = [];self.scripts = scripts;self.leases = []
        cfg = dict(context=4096, lease_minutes=19, model='unused', runtime_root=self.dir)
        @contextlib.contextmanager
        def lease():
            self.leases.append('held');yield 'lease-1';self.leases.append('released')
        def factory(tokens):
            producer = FakeProducer(tokens, self.scripts, self.dir);self.launched.append(producer);return producer
        self.engine = s.SingleEngine(cfg, tokenizer=self.tok, producer_factory=factory, lease_factory=lease)

    def run(self, body):
        deltas = [];final = {};metrics = {}
        def emit(kind, *values):
            if kind == 'delta':deltas.append(values)
            else:final.update(values[0])
        self.engine.run(body, emit, metrics)
        return deltas, final, metrics


class EngineTests(unittest.TestCase):
    def test_resident_multi_turn_appends_only_the_new_suffix(self):
        h = Harness([])
        h.scripts.append(ids(h.tok, 'hm</think>Hi there') + [s.EOS_ID])
        body = dict(messages=[dict(role='user', content='q')], max_tokens=50)
        deltas, final, metrics = h.run(body)
        self.assertEqual(final['parsed']['content'], 'Hi there');self.assertEqual(final['parsed']['reasoning'], 'hm')
        self.assertEqual(final['parsed']['finish'], 'stop');self.assertEqual(metrics['prefix_source'], 'empty')
        self.assertEqual(''.join(t for f, t in deltas if f == 'content'), 'Hi there')
        self.assertEqual(''.join(t for f, t in deltas if f == 'reasoning_content'), 'hm')
        producer = h.launched[0]
        self.assertEqual(len(h.engine.live), producer.position)
        # Turn 2: the client replays the answer; only the new tail is prefilled, in reserved-chunk pieces.
        h.scripts.append(ids(h.tok, '</think>ok') + [s.EOS_ID])
        follow = dict(body, messages=body['messages'] + [dict(role='assistant', content='Hi there', reasoning_content='hm'),
                                                        dict(role='user', content='and again, longer text')])
        _, final, metrics = h.run(follow)
        self.assertEqual(len(h.launched), 1);self.assertIn(metrics['prefix_source'], ('memory-token', 'memory-text'))
        self.assertGreater(metrics['reused_prefix_tokens'], 0)
        self.assertTrue(producer.prefills and all(len(c) <= 7 for c in producer.prefills))
        expected = r.render(follow, h.tok)[1]
        self.assertEqual(h.engine.live[:len(expected)], expected)
        # Turn 3: a different conversation diverges: retire, release the hold, relaunch on the whole prompt.
        h.scripts.append([s.EOS_ID])
        _, final, metrics = h.run(dict(messages=[dict(role='user', content='other chat')]))
        self.assertEqual(len(h.launched), 2);self.assertTrue(producer.closed)
        self.assertEqual(h.leases, ['held', 'released', 'held']);self.assertIn('fresh_reason', metrics)

    def test_length_limit_and_long_decode_carries_the_unevaluated_token(self):
        h = Harness([]);s_max = s.MAX_DECODE_CALL
        try:
            s.MAX_DECODE_CALL = 3
            h.scripts.extend([ids(h.tok, 'abc'), ids(h.tok, 'def'), ids(h.tok, 'g')])
            _, final, metrics = h.run(dict(messages=[dict(role='user', content='q')], max_tokens=7, thinking=False))
        finally:s.MAX_DECODE_CALL = s_max
        self.assertEqual(final['parsed']['content'], 'abcdefg');self.assertEqual(final['parsed']['finish'], 'length')
        self.assertEqual(h.launched[0].prefills, [ids(h.tok, 'c'), ids(h.tok, 'f')])
        self.assertEqual(len(h.engine.live), h.launched[0].position)

    def test_tool_call_finish_and_exact_dsml_replay(self):
        h = Harness([])
        block = r.dsml_tool_calls_text([dict(id='x', name='f', arguments='{"a": 1}')])
        h.scripts.append(ids(h.tok, '</think>Calling.' + block) + [s.EOS_ID])
        tools = [dict(type='function', function=dict(name='f', parameters={}))]
        body = dict(messages=[dict(role='user', content='q')], tools=tools)
        deltas, final, _ = h.run(body)
        parsed = final['parsed']
        self.assertEqual(parsed['finish'], 'tool_calls');self.assertEqual(parsed['content'], 'Calling.')
        self.assertEqual(''.join(t for f, t in deltas if f == 'content'), 'Calling.')
        call = parsed['calls'][0];self.assertEqual(h.engine.tool_memory[call['id']], block)
        h.scripts.append(ids(h.tok, '</think>done') + [s.EOS_ID])
        follow = dict(body, messages=body['messages'] + [
            dict(role='assistant', content='Calling.', reasoning_content='', tool_calls=[dict(id=call['id'], function=dict(name='f', arguments='{"a":1}'))]),
            dict(role='tool', tool_call_id=call['id'], content='42')])
        _, _, metrics = h.run(follow)
        self.assertEqual(len(h.launched), 1);self.assertIn(metrics['prefix_source'], ('memory-token', 'memory-text'))

    def test_quality_applies_once_and_per_phase_picks_the_decode_phase(self):
        h = Harness([[s.EOS_ID], [s.EOS_ID]])
        h.engine.set_quality('max_speed')
        h.run(dict(messages=[dict(role='user', content='q')]))
        self.assertEqual(h.launched[0].qualities, [1.36125])
        h.engine.set_quality(dict(reasoning=0.0, tool_call=0.35, answer=0.9))
        h.run(dict(messages=[dict(role='user', content='q')]))
        self.assertEqual(h.launched[0].qualities, [1.36125, 0.0])

    def test_splitter_never_streams_past_the_final_parse(self):
        split = s.Splitter(True);pieces = []
        for ch in 'abc</thi' + 'nk>Hello ' + '\n\n' + r.DS41_TOOL_CALLS_START + 'x':pieces += split.feed(ch)
        self.assertEqual(''.join(t for f, t in pieces if f == 'reasoning_content'), 'abc')
        self.assertEqual(''.join(t for f, t in pieces if f == 'content'), 'Hello')
        self.assertEqual(split.remainder('content', 'Hello'), '');self.assertEqual(split.remainder('content', 'Hello!'), '!')


class HttpTests(unittest.TestCase):
    def setUp(self):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = 't' * 40
        self.h = Harness([ids(char_tokenizer(), 'Hey') + [s.EOS_ID]])
        cfg = dict(context=4096, lease_minutes=19, runtime_root=self.h.dir, model_name='DeepSeek-V4.1-Flash')
        self.proxy = s.SingleProxy(cfg, 'run', None, engine=self.h.engine)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(self.proxy))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def tearDown(self):self.server.shutdown();self.server.server_close()

    def call(self, path, body=None, token='t' * 40):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:return response.read()

    def test_endpoints(self):
        self.assertEqual(json.loads(self.call('/v1/models'))['data'][0]['id'], 'deepseek-v4.1-flash')
        with self.assertRaises(urllib.error.HTTPError):self.call('/v1/quality-speed', token='bad' * 20)
        state = json.loads(self.call('/v1/quality-speed', dict(value='balanced')))
        self.assertTrue(state['supported']);self.assertEqual(state['value'], 0.35);self.assertTrue(state['deferred'])
        raw = self.call('/v1/chat/completions', dict(messages=[dict(role='user', content='q')], stream=True, thinking=False,
                                                    stream_options=dict(include_usage=True)))
        rows = [json.loads(line[6:]) for line in raw.decode().split('\n') if line.startswith('data: {')]
        self.assertTrue(raw.decode().rstrip().endswith('data: [DONE]'))
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in rows if c['choices']), 'Hey')
        self.assertEqual([c['choices'][0]['finish_reason'] for c in rows if c['choices']][-1], 'stop')
        self.assertEqual(rows[-1]['usage']['completion_tokens'], 3)
        self.assertEqual(self.h.launched[0].qualities, [0.35])
        with self.assertRaises(urllib.error.HTTPError) as refused:self.call('/v1/chat/completions', dict(model='m'))
        self.assertEqual(refused.exception.code, 400)
        self.assertEqual(len(self.h.launched), 1);self.assertFalse(self.h.launched[0].closed)


if __name__ == '__main__':unittest.main()
