"""The single-RTX web UI path: streamed single-engine replies, prefill progress and context."""
import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from slstack.config import load
from slstack.engines import Engines
from slstack.router import Stack, handler, server_ref


def event(delta, finish=None, **extra):
    return {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}], **extra}


PROGRESS = {'tokens_total': 2000, 'tokens_done': 1000, 'percent': 50.0, 'sweep': 1, 'sweeps': 1,
            'layer': 20, 'layers': 40, 'elapsed_s': 1.5, 'eta_s': 1.5}


def single_engine(cfg):
    engine = Mock()
    engine.models = None
    engine.single_engine.return_value = True
    engine.lock = threading.RLock()
    engine.chat_events.return_value = iter([
        event({'role': 'assistant'}),
        {'object': 'chat.completion.chunk', 'choices': [], 'prefill_progress': PROGRESS},
        event({'content': 'Hello.'}),
        event({}, 'stop', timings={'predicted_n': 2, 'predicted_ms': 300.0, 'predicted_per_second': 6.67})])
    return engine


def serve(stack):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(stack))
    server.daemon_threads = True
    server_ref[0] = server
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class SingleEngineStreamTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(overrides={'gpu_uuid': 'GPU-test'})

    def test_progress_reaches_the_web_ui_as_prompt_progress(self):
        stack = Stack(self.cfg, single_engine(self.cfg))
        rows = list(stack.chat_events({'messages': [{'role': 'user', 'content': 'Hi'}]}))
        progress = [row for row in rows if 'prompt_progress' in row]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]['choices'], [])
        self.assertEqual(progress[0]['prompt_progress'], {'total': 2000, 'cache': 0, 'processed': 1000, 'time_ms': 1500})
        self.assertEqual(progress[0]['prefill_progress'], PROGRESS)
        self.assertEqual(rows[-1]['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(rows[-1]['timings']['predicted_n'], 2)
        self.assertEqual(rows[-1]['slstack']['route'], 'single-engine')
        self.assertTrue(''.join(r['choices'][0]['delta'].get('content', '') for r in rows if r['choices']).startswith('Hello.'))

    def test_router_streams_the_single_engine_instead_of_buffering(self):
        engine = single_engine(self.cfg)
        server = serve(Stack(self.cfg, engine))
        try:
            body = json.dumps({'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}).encode()
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', data=body,
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                text = response.read().decode()
        finally:
            server.shutdown();server.server_close()
        engine.chat.assert_not_called()
        rows = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
        self.assertTrue(any('prompt_progress' in row for row in rows))
        self.assertFalse(rows[-1]['slstack']['buffered_stream'])
        self.assertTrue(text.rstrip().endswith('data: [DONE]'))

    def test_runtime_ram_abort_reaches_streamed_client_as_an_error(self):
        from studio.dsv41_session_contract import Refused
        reason='Persistent producer stopped: available physical RAM below runtime_min_free_gib=4'
        def failed():
            yield event({'content':'Partial reply'})
            raise Refused(reason)
        engine=single_engine(self.cfg);engine.chat_events.return_value=failed()
        server=serve(Stack(self.cfg,engine))
        try:
            body=json.dumps({'messages':[{'role':'user','content':'Hi'}],'stream':True}).encode()
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=10) as response:text=response.read().decode()
        finally:
            server.shutdown();server.server_close()
        rows=[json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
        self.assertIn(reason,next(row['error']['message'] for row in rows if 'error' in row))
        self.assertTrue(any(reason in choice.get('delta',{}).get('content','') for row in rows for choice in row.get('choices',[])))
        self.assertFalse(any(choice.get('finish_reason')=='stop' for row in rows for choice in row.get('choices',[])))
        self.assertTrue(text.rstrip().endswith('data: [DONE]'))

    def test_props_report_the_single_engine_context(self):
        engine = single_engine(self.cfg)
        server = serve(Stack({**self.cfg, 'context': 999424}, engine))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/props', timeout=10) as response:
                props = json.load(response)
        finally:
            server.shutdown();server.server_close()
        self.assertEqual(props['default_generation_settings']['n_ctx'], 999424)

    def test_engines_stream_from_the_local_single_engine_with_its_token(self):
        seen = {}
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):pass
            def do_POST(self):
                seen['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen['auth'] = self.headers.get('Authorization')
                self.send_response(200);self.send_header('Content-Type', 'text/event-stream');self.send_header('Connection', 'close');self.end_headers()
                for row in (event({'role': 'assistant'}), {'choices': [], 'prefill_progress': PROGRESS}, event({}, 'stop')):
                    self.wfile.write(b'data: ' + json.dumps(row).encode() + b'\n\n')
                self.wfile.write(b': progress\n\ndata: [DONE]\n\n')
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        engines = Engines.__new__(Engines)
        engines.cfg = {**self.cfg, 'ds4_url': f'http://127.0.0.1:{upstream.server_port}', '_control_token': 'tok',
                       'serving': 'single_engine', 'mac': None}
        engines.remote = None
        engines.select = lambda model: None
        engines.ready = lambda: None
        try:
            rows = list(engines.chat_events({'messages': [], 'stream': False}))
        finally:
            upstream.shutdown();upstream.server_close()
        self.assertEqual(seen['auth'], 'Bearer tok')
        self.assertTrue(seen['body']['stream'])
        self.assertEqual(rows[1]['prefill_progress'], PROGRESS)
        self.assertEqual(rows[-1]['choices'][0]['finish_reason'], 'stop')

    def test_native_serving_still_refuses_unpaired_streaming(self):
        engines = Engines.__new__(Engines)
        engines.cfg = {**self.cfg, 'serving': 'native', 'mac': None}
        engines.remote = None
        engines.select = lambda model: None
        engines.ready = lambda: None
        with self.assertRaisesRegex(ValueError, 'paired Mac receiver or the single engine'):
            list(engines.chat_events({'messages': []}))


if __name__ == '__main__':
    unittest.main()
