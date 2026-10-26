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
from slstack.config import load, ROOT
from slstack.journal import Journal, sha256
from slstack.lease import active_holds, recover_leases
from slstack.process import OwnedProcess
from slstack.router import Stack, handler, media_request
from slstack.tools import validate, QUALITY, media_schema
from slstack.lora import parse_lora, output_path, train_lora, merge_lora, unmerge_lora


class FixtureEngines:
    def __init__(self, replies=None):
        self.lock = threading.RLock()
        self.llm = None
        self.replies = list(replies or [])
        self.requests, self.media_calls = [], []

    def chat(self, body):
        self.requests.append(deepcopy(body))
        return self.replies.pop(0)

    def quality(self, value=None):
        raise RuntimeError('Live control unavailable in this explicit CPU fixture')

    def media(self, kind, params):
        self.media_calls.append((kind, params))
        return {'data': [{'url': '/outputs/fixture.png', 'path': 'TEST_PRIVATE_OUTPUT_PATH'}],
                'receipt': 'TEST_PRIVATE_RECEIPT_PATH', 'fixture_only': True}


def reply(message, finish='stop'):
    return {'id': 'chatcmpl-TEST', 'object': 'chat.completion', 'model': 'TEST',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


def tool_reply(name, args):
    return reply({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call-TEST', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls')


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(load()['runtime_root']) / 'fixtures' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.cfg = load(overrides={'runtime_root': str(self.root), 'host_budget_gib': .5,
                                   'job_commit_gib': .5, 'min_available_gib': 2})


class SchemaTests(Base):
    def test_bias_edges_and_rejections(self):
        for value in (0, .35, .9):
            self.assertEqual(validate(QUALITY, {'value': value})['value'], value)
        for value in (-.01, 1.36126, float('nan'), float('inf'), True, '0.3'):
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
        self.assertEqual(Path(load(overrides={'runtime_root': str(self.root)})['runtime_root']), self.root.resolve())

    def test_shared_holds(self):
        self.assertEqual(active_holds('ACTIVE: DS4WINBIAS2 abc; GPU\nRELEASED: DS4WINBIAS2 abc; done'), [])
        self.assertEqual(len(active_holds('ACTIVE: OTHER task GPU\nACTIVE: DS4STACK abc; GPU\nRELEASED: DS4STACK abc; done')), 1)

    def test_dead_guardian_hold_released_once(self):
        hold = self.root / 'fixture-hold.md'
        hold.write_text('ACTIVE: DS4STACK TEST123; fixture\n')
        leases = self.root / 'state' / 'leases'
        leases.mkdir(parents=True)
        (leases / 'TEST123.json').write_text(json.dumps({'task': 'DS4STACK', 'id': 'TEST123',
            'hold_file': str(hold), 'guardian_pid': 2147483647, 'guardian_created': 0, 'released': False}))
        self.assertEqual(recover_leases(self.root), ['TEST123'])
        self.assertEqual(active_holds(hold.read_text()), [])
        self.assertEqual(recover_leases(self.root), [])


class RouterTests(Base):
    def test_openai_image_request_mapping_and_model_ownership(self):
        cfg = {**self.cfg, 'image_model': str(self.root / 'TEST-anima.safetensors'), 'image_arch': 'anima-1.0'}
        mapped = media_request('image', {'prompt': 'TEST', 'model': 'image', 'n': 1, 'size': '256x256', 'response_format': 'url'}, cfg)
        self.assertEqual(mapped, {'prompt': 'TEST', 'width': 256, 'height': 256})
        for patch in ({'n': 2}, {'n': True}, {'model': 'unselected'}, {'size': 'auto'}, {'size': '512x512', 'width': 256}):
            with self.assertRaises(ValueError):
                media_request('image', {'prompt': 'TEST', **patch}, cfg)

    def test_model_tool_media_roundtrip(self):
        engine = FixtureEngines([tool_reply('generate_image', {'prompt': 'TEST', 'steps': 7}),
                                 reply({'role': 'assistant', 'content': 'Generated TEST fixture'})])
        result = Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST image, 7 steps'}]})
        self.assertEqual(engine.media_calls[0][1]['steps'], 7)
        self.assertEqual(engine.requests[1]['messages'][-1]['role'], 'tool')
        self.assertNotIn('TEST_PRIVATE', engine.requests[1]['messages'][-1]['content'])
        self.assertIn('/outputs/fixture.png', result['choices'][0]['message']['content'])

    def test_missing_bias_never_reports_success(self):
        engine = FixtureEngines([tool_reply('set_quality_speed', {'value': .35}), reply({'role': 'assistant', 'content': 'Unavailable'})])
        Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST bias'}]})
        outcome = json.loads(engine.requests[1]['messages'][-1]['content'])
        self.assertIn('unavailable', outcome['error'])

    def test_external_client_tools_not_executed(self):
        engine = FixtureEngines([tool_reply('client_function', {})])
        result = Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST'}]})
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(engine.media_calls, [])

    def test_tool_loop_has_no_fixed_round_limit(self):
        engine = FixtureEngines([tool_reply('get_quality_speed', {}) for _ in range(12)] +
                               [reply({'role':'assistant','content':'Finished'})])
        result=Stack(self.cfg, engine).chat({'messages': [{'role': 'user', 'content': 'TEST'}]})
        self.assertIn('Finished',result['choices'][0]['message']['content'])
        self.assertEqual(len(engine.requests),13)

    def test_mcp_protocol_errors(self):
        stack = Stack(self.cfg, FixtureEngines())
        self.assertEqual(len(stack.mcp({'id': 1, 'method': 'tools/list'})['result']['tools']), 6)
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
            self.assertEqual(len(mcp['result']['tools']), 6)
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
        script = 'import sys,time;from slstack.journal import Journal;j=Journal(sys.argv[1]);j.begin(sys.argv[2],[(sys.argv[3],.5)],"attaching");print("READY",flush=True);time.sleep(60)'
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

    def test_progressing_child_outlives_its_window(self):
        # Owner rule: timeout_s is a no-progress window. A child that keeps logging for 4 s with a
        # 1 s window must finish normally (the old total cap would have killed it at 1 s).
        child = OwnedProcess(self.cfg, [sys.executable, '-B', '-u', '-c',
                                        'import time\nfor i in range(20):print("tick",i,flush=True);time.sleep(.2)'],
                             'TEST-progress', timeout=1)
        self.assertEqual(child.wait()['exit_code'], 0)

    def test_rss_budget_failure_is_loud(self):
        cfg = {**self.cfg, 'host_budget_gib': .08, 'job_commit_gib': .5}
        child = OwnedProcess(cfg, [sys.executable, '-B', '-c', 'import time;a=bytearray(128*1024**2);time.sleep(3)'], 'TEST-memory')
        with self.assertRaisesRegex(RuntimeError, 'HOST MEMORY LIMIT'):
            child.wait()


class LoraTests(Base):
    def test_windows_scale_and_collision(self):
        self.assertEqual(parse_lora(str(self.root / 'adapter.safetensors') + ':.25')[1], .25)
        self.assertEqual(parse_lora(str(self.root / 'adapter.safetensors'))[1], 1)
        p = self.root / 'existing.bin'
        p.write_bytes(b'TEST')
        with self.assertRaises(FileExistsError):
            output_path(self.cfg, p)
        with self.assertRaises(ValueError):
            output_path(self.cfg, self.root.parent / 'escape.bin')

    def test_unsupported_architecture_never_starts(self):
        data = self.root / 'prompts'
        data.mkdir()
        (data / 'TEST.txt').write_text('TEST corpus')
        base = self.root / 'base.gguf'
        base.write_bytes(b'GGUF TEST')
        settings = self.root / 'settings.json'
        settings.write_text(json.dumps({'family': 'gguf_streamed', 'arch': 'unsupported_architecture', 'model': str(base)}))
        with self.assertRaisesRegex(ValueError, 'not supported'):
            train_lora(self.cfg, data, settings, self.root / 'adapter')
        self.assertFalse((self.root / 'jobs').exists())

    def test_permanent_merge_requires_small_test(self):
        with self.assertRaisesRegex(ValueError, '--test'):
            merge_lora(self.cfg, 'missing', [], self.root / 'output', 'missing')

    def test_unmerge_selects_verified_base_without_deleting(self):
        base, lora, merged = [self.root / n for n in ('base.bin', 'lora.bin', 'merged.bin')]
        base.write_bytes(b'TEST base')
        lora.write_bytes(b'TEST adapter')
        merged.write_bytes(b'TEST merged copy')
        j = Journal(self.root)
        row = j.begin(base, [(lora, .5)], 'merging', merged)
        j.update(row['id'], 'merged', output_sha256=sha256(merged))
        result = unmerge_lora(self.cfg, row['id'])
        self.assertEqual(result['model'], str(base.resolve()))
        self.assertTrue(merged.is_file())
        selected = load(overrides={'runtime_root': str(self.root)})
        self.assertEqual(selected['model'], str(base.resolve()))


if __name__ == '__main__':
    unittest.main()
