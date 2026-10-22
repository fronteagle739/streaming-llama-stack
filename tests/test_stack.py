"""CPU protocol/safety fixtures only. No fixture represents a qualified model output."""
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from ds4stack.config import load, ROOT
from ds4stack.journal import Journal, sha256
from ds4stack.lease import active_holds
from ds4stack.process import OwnedProcess
from ds4stack.router import Stack, handler
from ds4stack.tools import validate, QUALITY, media_schema


class FixtureEngines:
    def __init__(self, replies=None):
        self.lock = threading.RLock()
        self.llm = None
        self.replies = list(replies or [])
        self.requests, self.media_calls = [], []

    def chat(self, body):
        self.requests.append(deepcopy(body))
        return self.replies.pop(0)

    def media(self, kind, params):
        self.media_calls.append((kind, params))
        return {'data': [{'url': '/outputs/fixture.png'}], 'fixture_only': True}


def reply(message, finish='stop'):
    return {'id': 'chatcmpl-TEST', 'object': 'chat.completion', 'model': 'TEST',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


def tool_reply(name, args):
    return reply({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call-TEST', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls')


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path('runtime/fixtures') / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.cfg = load(overrides={'runtime_root': str(self.root), 'host_budget_gib': .5,
                                   'job_commit_gib': .5, 'min_available_gib': 2})


class SchemaTests(Base):
    def test_bias_edges_and_rejections(self):
        for value in (0, .35, .9):
            self.assertEqual(validate(QUALITY, {'value': value})['value'], value)
        for value in (-.01, 1, float('nan'), float('inf'), True, '0.3'):
            with self.assertRaises(ValueError):
                validate(QUALITY, {'value': value})

    def test_media_defaults(self):
        args = validate(media_schema('video'), {'prompt': 'TEST scene'})
        self.assertEqual((args['steps'], args['num_frames'], args['seed']), (20, 17, 42))
        for patch in ({'width': 65}, {'num_frames': 16}, {'fps': 0}, {'steps': 101}, {'prompt': ''}, {'model': 'unowned'}):
            with self.assertRaises(ValueError):
                validate(media_schema('video'), {'prompt': 'TEST', **patch})

    def test_invalid_budget_and_runtime(self):
        with self.assertRaises(ValueError):
            load(overrides={'host_budget_gib': float('nan')})
        with self.assertRaises(ValueError):
            load(overrides={'lease_minutes': 21})
        if os.name == 'nt':
            with self.assertRaises(ValueError):
                load(overrides={'runtime_root': 'runtime/never'})

    def test_shared_holds(self):
        self.assertEqual(active_holds('ACTIVE: DS4WINBIAS2 abc; GPU\nRELEASED: DS4WINBIAS2 abc; done'), [])
        self.assertEqual(len(active_holds('ACTIVE: OTHER task GPU\nACTIVE: DS4STACK abc; GPU\nRELEASED: DS4STACK abc; done')), 1)


class RouterTests(Base):
    def test_model_tool_media_roundtrip(self):
        engine = FixtureEngines([tool_reply('generate_image', {'prompt': 'TEST', 'steps': 7}),
                                 reply({'role': 'assistant', 'content': 'Generated TEST fixture'})])
        result = Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST image, 7 steps'}]})
        self.assertEqual(engine.media_calls[0][1]['steps'], 7)
        self.assertEqual(engine.requests[1]['messages'][-1]['role'], 'tool')
        self.assertIn('/outputs/fixture.png', result['choices'][0]['message']['content'])

    def test_missing_bias_never_reports_success(self):
        engine = FixtureEngines([tool_reply('set_quality_speed', {'value': .35}), reply({'role': 'assistant', 'content': 'Unavailable'})])
        Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST bias'}]})
        outcome = json.loads(engine.requests[1]['messages'][-1]['content'])
        self.assertIn('No setting was changed', outcome['error'])

    def test_external_client_tools_not_executed(self):
        engine = FixtureEngines([tool_reply('client_function', {})])
        result = Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST'}]})
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(engine.media_calls, [])

    def test_tool_loop_bounded(self):
        engine = FixtureEngines([tool_reply('get_quality_speed', {}) for _ in range(8)])
        with self.assertRaisesRegex(RuntimeError, 'iteration limit'):
            Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST'}]})

    def test_mcp_protocol_errors(self):
        stack = Stack(self.cfg, FixtureEngines())
        self.assertEqual(len(stack.mcp({'id': 1, 'method': 'tools/list'})['result']['tools']), 4)
        result = stack.mcp({'id': 2, 'method': 'tools/call', 'params': {'name': 'generate_video', 'arguments': {'prompt': 'TEST', 'fps': 0}}})
        self.assertTrue(result['result']['isError'])
        self.assertIsNone(stack.mcp({'method': 'notifications/initialized'}))

    def test_http_json_sse_mcp_and_path_confinement(self):
        cfg = {**self.cfg, 'webui': True, 'webui_root': str(self.root / 'web')}
        (self.root / 'web').mkdir()
        (self.root / 'web' / 'index.html').write_text('<!doctype html><p>TEST fixture only</p>')
        (self.root / 'secret.txt').write_text('TEST PRIVATE')
        engine = FixtureEngines([reply({'role': 'assistant', 'content': 'TEST SSE'})])
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(Stack(cfg, engine)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        def get(route):
            with urllib.request.urlopen(base + route, timeout=3) as response:
                return response.read()
        def post(route, body):
            req = urllib.request.Request(base + route, json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.read()
        try:
            self.assertIn(b'TEST fixture', get('/'))
            self.assertIn(b'DSV4.1', get('/v1/models'))
            data = post('/v1/chat/completions', {'stream': True, 'messages': [{'role': 'user', 'content': 'TEST'}]})
            self.assertIn(b'data: [DONE]', data)
            self.assertIn(b'TEST SSE', data)
            mcp = json.loads(post('/mcp', {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}))
            self.assertEqual(len(mcp['result']['tools']), 4)
            with self.assertRaises(urllib.error.HTTPError) as exc:
                get('/outputs/%2e%2e/secret.txt')
            self.assertEqual(exc.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class JournalTests(Base):
    def test_dead_attach_recovered_and_base_untouched(self):
        base, lora = self.root / 'base.bin', self.root / 'lora.bin'
        base.write_bytes(b'TEST immutable base')
        lora.write_bytes(b'TEST adapter')
        digest = sha256(base)
        script = 'import sys,time;from ds4stack.journal import Journal;j=Journal(sys.argv[1]);j.begin(sys.argv[2],[(sys.argv[3],.5)],"attaching");print("READY",flush=True);time.sleep(60)'
        proc = subprocess.Popen([sys.executable, '-B', '-c', script, str(self.root), str(base), str(lora)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'READY')
            with self.assertRaisesRegex(RuntimeError, 'live PID'):
                Journal(self.root).recover()
            proc.kill()  # Windows equivalent of kill -9, only our synthetic fixture.
            proc.wait(timeout=5)
            reports = Journal(self.root).recover()
            self.assertEqual(reports[0]['status'], 'detached')
            self.assertEqual(sha256(base), digest)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()

    def test_hash_mismatch_refused_and_partial_preserved(self):
        base, lora = self.root / 'base.bin', self.root / 'lora.bin'
        base.write_bytes(b'TEST base')
        lora.write_bytes(b'TEST adapter')
        j = Journal(self.root)
        row = j.begin(base, [(lora, 1)], 'merging', self.root / 'merged.bin')
        data = j.read()
        data['entries'][0]['pid'] = 2147483647
        j.path.write_text(json.dumps(data))
        partial = Path(row['partial_path'])
        partial.write_bytes(b'TEST partial preserved')
        self.assertEqual(j.recover()[0]['status'], 'failed')
        recovered = Path(j.read()['entries'][0]['recovered_partial'])
        self.assertEqual(recovered.read_bytes(), b'TEST partial preserved')
        base.write_bytes(b'TEST unexpected modification')
        with self.assertRaisesRegex(RuntimeError, 'BASE HASH MISMATCH'):
            j.recover()


class GuardianTests(Base):
    def test_cpu_child_receipt(self):
        child = OwnedProcess(self.cfg, [sys.executable, '-B', '-c', 'print("TEST guarded CPU")'], 'TEST-cpu')
        result = child.wait()
        self.assertEqual(result['exit_code'], 0)
        self.assertLess(result['peak_tree_rss'], .5 * 1024**3)

    def test_deadline_and_child_tree_killed(self):
        child = OwnedProcess(self.cfg, [sys.executable, '-B', '-c', 'import time;time.sleep(30)'], 'TEST-deadline', timeout=.25)
        with self.assertRaisesRegex(RuntimeError, 'deadline exceeded'):
            child.wait()


if __name__ == '__main__':
    unittest.main()
