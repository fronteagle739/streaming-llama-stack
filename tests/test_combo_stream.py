import json
import threading
import unittest
from slstack.combo_proxy import relay_stream,conversation_key,native_launch_config


class Response:
    def __init__(self,raw,width):self.raw=raw;self.width=width
    def read1(self,limit):
        block=self.raw[:self.width];self.raw=self.raw[self.width:];return block


class StreamTest(unittest.TestCase):
    def test_normal_native_launch_accepts_assigned_port_but_not_changed_context(self):
        args=['-m','original.gguf','--port','8080','--host','127.0.0.1','--ctx','262144']
        cfg=dict(studio_expected_args=args)
        actual=list(args);actual[3]='18808'
        self.assertEqual(native_launch_config(cfg,actual)['port'],18808)
        actual[-1]='1048576'
        with self.assertRaises(ValueError):native_launch_config(cfg,actual)

    def test_native_bytes_preserved_and_completion_waits_for_checkpoint(self):
        content=b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        final=b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        for width in (1,7,65536):
            emitted=[];completed=threading.Event()
            tail=relay_stream(Response(content+final,width),emitted.append,completed)
            self.assertTrue(completed.is_set())
            self.assertEqual(b''.join(emitted),content)
            self.assertEqual(b''.join(emitted)+tail,content+final)

    def test_no_success_when_native_stream_truncates(self):
        for raw in (b'data: {',b'data: {"choices":[]}\n\n'):
            with self.assertRaises(ValueError):relay_stream(Response(raw,3),lambda b:None,threading.Event())

    def test_conversation_reuses_only_stable_prefix_and_tool_schema(self):
        body=dict(model='test',messages=[dict(role='system',content='system'),dict(role='user',content='first')])
        key=conversation_key(body)
        self.assertEqual(key,conversation_key({**body,'messages':[*body['messages'],dict(role='assistant',content='answer'),dict(role='user',content='next')]}))
        self.assertNotEqual(key,conversation_key({**body,'tools':[dict(type='function',function=dict(name='new'))]}))


if __name__=='__main__':unittest.main()
