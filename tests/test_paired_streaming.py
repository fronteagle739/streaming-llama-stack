import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from slstack.config import load
from slstack.router import Stack
from slstack.streaming import UpstreamError, events, send


def event(delta, finish=None, **extra):
    return {'choices':[{'index':0,'delta':delta,'finish_reason':finish}], **extra}


class PairedStreamingTests(unittest.TestCase):
    def test_http_first_event_arrives_before_generation_finishes(self):
        continue_generation = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                def source():
                    yield event({'content':'first'})
                    if not continue_generation.wait(5):
                        raise RuntimeError('Client did not receive the first event during generation')
                    yield event({'content':'second'})
                    yield event({},'stop')
                send(self,source())
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        source = events(f'http://127.0.0.1:{server.server_port}',{},timeout=6)
        try:
            self.assertEqual(next(source)['choices'][0]['delta']['content'],'first')
            continue_generation.set()
            tail = list(source)
            self.assertEqual(tail[0]['choices'][0]['delta']['content'],'second')
            self.assertEqual(tail[-1]['choices'][0]['finish_reason'],'stop')
        finally:
            continue_generation.set()
            source.close()
            server.shutdown()
            server.server_close()
            worker.join(2)

    def test_fragmented_stack_tool_executes_between_streamed_rounds(self):
        cfg = load(overrides={'gpu_uuid':'GPU-test','mac':'http://example.test:8180'})
        engine = Mock()
        engine.models = None
        tool = {'index':0,'id':'call-test','type':'function',
                'function':{'name':'set_quality_speed','arguments':'{"value":'}}
        engine.chat_events.side_effect = [iter([
            event({'reasoning_content':'Considering the requested setting.'}),
            event({'tool_calls':[tool]}),
            event({'tool_calls':[{'index':0,'function':{'arguments':'"balanced"}'}}]},'tool_calls')]),
            iter([event({'content':'Applied.'}),event({},'stop',timings={'predicted_n':3,'predicted_ms':12})])]
        engine.quality.return_value = {'supported':False}
        stack = Stack(cfg,engine)
        original = {'messages':[{'role':'user','content':'Use Balanced'}]}
        source = stack.chat_events(original)
        first = next(source)
        self.assertEqual(first['choices'][0]['delta']['reasoning_content'],'Considering the requested setting.')
        engine.quality.assert_not_called()
        result = list(source)
        engine.quality.assert_called_once_with(.35)
        self.assertEqual(result[-1]['choices'][0]['finish_reason'],'stop')
        self.assertFalse(result[-1]['slstack']['buffered_stream'])
        self.assertEqual(result[-1]['timings']['predicted_n'],3)
        self.assertEqual(len(original['messages']),1)
        second = engine.chat_events.call_args.args[0]
        self.assertTrue(second['stream'])
        self.assertEqual(second['messages'][-1]['role'],'tool')
        self.assertEqual(json.loads(second['messages'][-1]['content']),{'supported':False})


REFUSAL = "[Errno 2] No such file or directory: 'state\\gpu.holds'"


def serve(do_post):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            do_post(self)
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


def post(server):
    request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}', data=b'{}',
                                     headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(request, timeout=5)


class StreamErrorTests(unittest.TestCase):
    def test_refusal_before_content_is_a_plain_http_error(self):
        # The single engine commits 200 + a role chunk, then refuses (the captured live repro).
        def engine(handler):
            handler.send_response(200)
            handler.send_header('Content-Type','text/event-stream')
            handler.send_header('Connection','close')
            handler.end_headers()
            handler.close_connection = True
            handler.wfile.write(b'data: ' + json.dumps(event({'role':'assistant'})).encode() + b'\n\n')
            handler.wfile.write(b'data: ' + json.dumps({'error':{'message':REFUSAL,'type':'single_engine_refused'}}).encode() + b'\n\n')
            handler.wfile.write(b'data: [DONE]\n\n')
        upstream = serve(engine)
        router = serve(lambda handler: send(handler, events(f'http://127.0.0.1:{upstream.server_port}',{},timeout=5)))
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post(router)
            self.assertEqual(caught.exception.code,503)
            self.assertEqual(caught.exception.headers['Content-Type'],'application/json')
            body = json.loads(caught.exception.read())
            self.assertEqual(body,{'error':{'code':503,'message':REFUSAL,'type':'single_engine_refused'}})
        finally:
            for server in (upstream, router):
                server.shutdown()
                server.server_close()

    def test_upstream_http_error_keeps_its_status_and_plain_message(self):
        def engine(handler):
            body = json.dumps({'error':{'message':REFUSAL,'type':'single_engine_refused'}}).encode()
            handler.send_response(503)
            handler.send_header('Content-Type','application/json')
            handler.send_header('Content-Length',str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        upstream = serve(engine)
        try:
            with self.assertRaises(UpstreamError) as caught:
                list(events(f'http://127.0.0.1:{upstream.server_port}',{},timeout=5))
            self.assertEqual((caught.exception.status,str(caught.exception),caught.exception.type),
                             (503,REFUSAL,'single_engine_refused'))
        finally:
            upstream.shutdown()
            upstream.server_close()

    def test_error_after_content_is_shown_as_content_then_error_row(self):
        def source():
            yield event({'role':'assistant'})
            yield event({'content':'Partial'}, id='chatcmpl-x', model='m')
            raise UpstreamError({'message':REFUSAL,'type':'single_engine_refused'})
        router = serve(lambda handler: send(handler, source()))
        try:
            with post(router) as response:
                self.assertEqual(response.status,200)
                rows = [line[6:] for line in response.read().decode().split('\n') if line.startswith('data: ')]
            self.assertEqual(rows[-1],'[DONE]')
            parsed = [json.loads(row) for row in rows[:-1]]
            content = ''.join(r['choices'][0]['delta'].get('content','') for r in parsed if r.get('choices'))
            self.assertEqual(content,'Partial\n\n**Error:** ' + REFUSAL)
            self.assertEqual(parsed[-1],{'error':{'code':500,'message':REFUSAL,'type':'single_engine_refused'}})
            self.assertNotIn("{'message'",content)
        finally:
            router.shutdown()
            router.server_close()


if __name__ == '__main__':
    unittest.main()
