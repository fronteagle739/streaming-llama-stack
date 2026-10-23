import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from slstack.config import load
from slstack.router import Stack
from slstack.streaming import events, send


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


if __name__ == '__main__':
    unittest.main()
