"""General decoder reservations end on success, failure and cancellation."""
import threading
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import urllib.error
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
                self.assertTrue(response.read().startswith(b'window.slstackNativeGeneration=false;'))
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

    def test_generation_controls_preserve_general_and_paired_contracts(self):
        from slstack.router import handler
        cases=[({},True,'next token boundary'),
               ({'mac':'http://example.test'},True,'next request'),
               ({'general':{'mode':'remote'}},False,None)]
        for cfg,supported,applies in cases:
            with self.subTest(cfg=cfg):
                generation=Mock()
                generation.state.return_value={}
                generation.update.return_value={'revision':1}
                server=ThreadingHTTPServer(('127.0.0.1',0),handler(SimpleNamespace(cfg=cfg,generation=generation)))
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    url='http://127.0.0.1:'+str(server.server_port)+'/slstack/generation'
                    with urllib.request.urlopen(url) as response:state=json.load(response)
                    self.assertEqual(state['supported'],supported)
                    request=urllib.request.Request(url,data=b'{"value":0.22}',headers={'Content-Type':'application/json'})
                    if supported:
                        self.assertEqual(state['applies'],applies)
                        with urllib.request.urlopen(request) as response:updated=json.load(response)
                        self.assertTrue(updated['supported']);self.assertEqual(updated['applies'],applies)
                        generation.update.assert_called_once_with({'value':.22})
                    else:
                        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
                        self.assertEqual(caught.exception.code,400)
                        self.assertIn('OpenAI request sampler',json.load(caught.exception)['error']['message'])
                        caught.exception.close()
                        generation.update.assert_not_called()
                finally:server.shutdown();server.server_close();thread.join()

    @unittest.skipUnless(shutil.which('node'),'Node is needed for the browser request interceptor test')
    def test_browser_preserves_general_samplers_and_normalizes_random_seed(self):
        source=(Path(__file__).resolve().parents[1]/'slstack/generation.js').read_text(encoding='utf-8')
        script='''const fs=require('fs'),vm=require('vm');
const data=JSON.parse(fs.readFileSync(0,'utf8'));let received,clones=0,created=0;
const element=()=>({style:{},append(){},setAttribute(){},addEventListener(){},
 querySelector:()=>element(),querySelectorAll:()=>[]});
const response={ok:true,headers:{get:()=> 'text/event-stream'},clone:()=>{clones++;return {
 body:{getReader:()=>({read:async()=>({done:true}),cancel:async()=>{}})}};}};
const context={URL,clearTimeout,setTimeout,setInterval(){},
 MutationObserver:class {observe(){}},TextDecoder,
 location:{href:'http://127.0.0.1/',origin:'http://127.0.0.1'},
 window:{slstackNativeGeneration:data.native,fetch:async(url,init)=>{received=JSON.parse(init.body);return response;}},
 document:{hidden:true,head:element(),body:element(),addEventListener(){},querySelector:()=>null,
 createElement:()=>{created++;return element();}}};
(async()=>{
 vm.runInNewContext(data.source,context);
 await context.window.fetch('/v1/chat/completions',{body:JSON.stringify(data.body)});
 process.stdout.write(JSON.stringify({body:received,clones,created}));
})().catch(error=>{console.error(error);process.exitCode=1;});'''
        body=dict(temperature=.8,top_p=.9,top_k=37,min_p=.02,seed=-1,max_tokens=256)
        for native in (False,True):
            run=subprocess.run(['node','-e',script],input=json.dumps(dict(source=source,native=native,body=body)),
                               text=True,capture_output=True,check=True,timeout=15)
            result=json.loads(run.stdout);actual=result['body']
            self.assertEqual(result['clones'],int(native))
            self.assertEqual(result['created']>0,native)
            self.assertNotIn('seed',actual)
            self.assertEqual(actual['max_tokens'],256)
            for key in ('temperature','top_p','top_k','min_p'):
                if native:self.assertNotIn(key,actual)
                else:self.assertEqual(actual[key],body[key])


if __name__=='__main__':unittest.main()
