"""General decoder reservations end on success, failure and cancellation."""
import threading
from http.server import ThreadingHTTPServer
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import urllib.request

from slstack.general_engine import GeneralEngines


class GeneralRequests(unittest.TestCase):
    def fixture(self,*,fail=False):
        engine=GeneralEngines.__new__(GeneralEngines)
        engine.cfg={'general':{'mode':'streaming'}};engine.lock=threading.RLock()
        engine.select=Mock();engine.remote=None;engine.llm=None
        remote=Mock();job=Mock();received=[]
        def events(body):
            received.append(body)
            yield {'choices':[{'delta':{'content':'first'}}]}
            if fail:raise RuntimeError('decode failed')
            yield {'choices':[{'delta':{'content':'second'}}]}
        remote.events.side_effect=events
        def ready():engine.remote=remote;engine.llm=job
        engine.ready=ready
        return engine,remote,job,received

    def test_sampler_fields_survive_and_success_releases_the_producer(self):
        engine,remote,job,received=self.fixture()
        body=dict(messages=[{'role':'user','content':'test'}],temperature=.8,top_p=.9,
                  top_k=37,min_p=.02,seed=123,max_tokens=256)
        self.assertEqual(len(list(engine.chat_events(body))),2)
        self.assertEqual(received,[body])
        job.stop.assert_called_once();remote.close.assert_called_once()
        self.assertIsNone(engine.llm);self.assertIsNone(engine.remote)

    def test_failure_releases_the_producer(self):
        engine,remote,job,_=self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError,'decode failed'):list(engine.chat_events({}))
        job.stop.assert_called_once();remote.close.assert_called_once()

    def test_browser_cancellation_releases_the_producer(self):
        engine,remote,job,_=self.fixture()
        stream=engine.chat_events({});next(stream);stream.close()
        job.stop.assert_called_once();remote.close.assert_called_once()

    def test_webui_props_report_the_general_preset_context(self):
        from slstack.router import handler
        engine=GeneralEngines.__new__(GeneralEngines)
        engine.cfg={'general':{'mode':'streaming','rtx':{'ctx':65536}}}
        stack=SimpleNamespace(engines=engine,cfg={**engine.cfg,'ds4_command':['server','--ctx','262144'],'webui':True},
                              models=SimpleNamespace(rows=[{}],default='model'))
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(stack))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with urllib.request.urlopen('http://127.0.0.1:'+str(server.server_port)+'/props') as response:
                props=json.load(response)
            self.assertEqual(props['default_generation_settings']['n_ctx'],65536)
            with urllib.request.urlopen('http://127.0.0.1:'+str(server.server_port)+'/slstack-generation.js') as response:
                self.assertEqual(response.read(),b'')
        finally:server.shutdown();server.server_close();thread.join()

    def test_general_route_annotations_distinguish_remote_prefill(self):
        from slstack.router import Stack
        stack=Stack.__new__(Stack)
        stack.cfg={'general':{'mode':'streaming','mac':{'backend':'mlx-receiver'}}}
        result=stack.route_metadata({'route':'single-engine','ultra_handoff':True})
        self.assertEqual((result['route'],result['decode_backend'],result['transport']),
                         ('rtx-prefill-mac-decode','mlx-receiver','kv-token'))
        self.assertFalse(result['ultra_handoff'])
        stack.cfg={'general':{'mode':'remote','mac':{'backend':'omlx'}}}
        self.assertEqual(stack.route_metadata({})['route'],'mac-http')


if __name__=='__main__':unittest.main()
