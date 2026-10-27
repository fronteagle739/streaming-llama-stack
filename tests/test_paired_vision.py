"""Projected image transport: exact pixels/embeddings, not placeholder IDs alone."""
import hashlib,json,struct,tempfile,unittest
from pathlib import Path
from slstack.paired_vision import unpack,materialize,MEDIA,MAX_BODY
from slstack.native_broker import vendor
vendor()
from studio.dsv41_session_contract import initial_request,MODEL_SHA256,command

class PairedVisionTests(unittest.TestCase):
    def wire(self,position=1,ids=(11,129264,12),fingerprint=b'a'*32):
        image=struct.pack('<4I',0x49564c53,1,1,5120)+fingerprint+struct.pack('<5120f',*([.25]*5120))
        rows=[dict(position=position,count=1,sha256=hashlib.sha256(image).hexdigest())]
        meta=json.dumps(rows,separators=(',',':')).encode()
        tokens=struct.pack('<%dI'%len(ids),*ids)
        return b'SLVP'+struct.pack('<II',len(tokens),len(meta))+tokens+meta+image

    def test_initial_and_incremental_positions_and_owned_descriptor(self):
        tokens,images=unpack(self.wire())
        self.assertEqual(struct.unpack('<3I',tokens),(11,129264,12))
        with tempfile.TemporaryDirectory() as d:
            desc=materialize(Path(d),images)
            rows=json.loads(desc.read_text())
            self.assertEqual(rows[0]['position'],1)
            self.assertEqual(Path(rows[0]['file']).read_bytes(),images[0]['data'])
            self.assertEqual(materialize(Path(d),images),desc)
        tokens,images=unpack(self.wire(position=101),base=100)
        self.assertEqual(images[0]['position'],101)

    def test_wrong_embedding_digest_marker_overlap_and_bounds_refuse(self):
        corrupt=bytearray(self.wire());corrupt[-1]^=1
        for payload in (bytes(corrupt),self.wire(position=0),self.wire(ids=(11,12,13)),self.wire()+b'x',b'SLVP'):
            with self.subTest(size=len(payload)),self.assertRaises(ValueError):unpack(payload)

    def test_image_inside_replay_tail_and_before_tail_have_same_transport(self):
        for position in (5,899,995):
            ids=[11]*1000;ids[position]=129264
            _,images=unpack(self.wire(position=position,ids=ids))
            self.assertEqual(images[0]['position'],position)

    def test_changed_historical_image_is_not_accepted_as_existing_identity(self):
        _,old=unpack(self.wire())
        with tempfile.TemporaryDirectory() as d:
            materialize(Path(d),old)
            _,changed=unpack(self.wire(fingerprint=b'b'*32))
            with self.assertRaisesRegex(ValueError,'historical'):
                materialize(Path(d),changed,previous=old,base=3)

    def headers(self,wire):
        tokens,_=unpack(wire)
        return {'content-type':MEDIA,'content-length':str(len(wire)),
                'x-dsv41-body-sha256':hashlib.sha256(wire).hexdigest(),
                'x-dsv41-model-sha256':MODEL_SHA256,'x-dsv41-mode':'3','x-dsv41-context':'1000001',
                'x-dsv41-tokens':str(len(tokens)//4),'x-dsv41-tokens-sha256':hashlib.sha256(tokens).hexdigest(),
                'x-dsv41-tier':'ultra','x-dsv41-yoco-keep':'140','x-dsv41-yoco-state-version':'1'}

    def test_authenticated_initial_contract_keeps_yoco_and_projections(self):
        wire=self.wire();headers=self.headers(wire)
        request=initial_request('a'*32,headers,wire)
        self.assertEqual(request['yoco_keep'],140)
        self.assertEqual(request['vision'][0]['position'],1)
        self.assertEqual(request['tokens'],struct.pack('<3I',11,129264,12))
        for field in ('x-dsv41-body-sha256','x-dsv41-tokens-sha256'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                initial_request('a'*32,dict(headers,**{field:'b'*64}),wire)

    def test_real_router_preserves_projection_and_incremental_absolute_position(self):
        from fastapi import FastAPI,Header,HTTPException
        from fastapi.testclient import TestClient
        from unittest.mock import Mock
        from studio.dsv41_session_broker import create_router
        def authorize(x_studio_token:str=Header(default='')):
            if x_studio_token!='test-only-token':raise HTTPException(401)
        broker=Mock();broker.start.return_value={'state':'starting'};broker.enqueue.return_value={'state':'running'}
        app=FastAPI();app.include_router(create_router(broker,authorize));client=TestClient(app)
        wire=self.wire();headers=self.headers(wire)
        url='/dsv41-native-offload/sessions/'+'a'*32
        self.assertEqual(client.post(url,content=wire,headers=headers).status_code,401)
        headers['x-studio-token']='test-only-token'
        self.assertEqual(client.post(url,content=wire,headers=headers).status_code,202)
        self.assertEqual(broker.start.call_args.args[0]['vision'][0]['data'],unpack(wire)[1][0]['data'])
        cp=dict(context=1000001,position=100,mode=3,capture_end=100,capture_length=100,
                tokens_sha256='c'*64,logits_sha256='d'*64,owner_sha256=['e'*64]*49)
        row=dict(session='a'*32,model_sha256=MODEL_SHA256,sequence=1,operation='prefill',base=cp)
        wire=self.wire(position=101);headers.update({'content-length':str(len(wire)),
            'x-dsv41-body-sha256':hashlib.sha256(wire).hexdigest(),'x-dsv41-control':json.dumps(row)})
        self.assertEqual(client.post(url+'/vision',content=wire,headers=headers).status_code,202)
        sent=broker.enqueue.call_args
        self.assertEqual(sent.kwargs['vision'][0]['position'],101)
        self.assertEqual(sent.args[1]['tokens'],[11,129264,12])
        command(sent.args[1],'a'*32,1)

    def test_network_cannot_supply_local_vision_paths(self):
        from studio.dsv41_session_broker import Broker
        broker=Broker(lambda *_:None)
        with self.assertRaisesRegex(ValueError,'Network paths'):
            broker.enqueue('a'*32,dict(operation='prefill',vision_spans_file='foreign.json'))

    def test_mac_launcher_selects_the_existing_native_encoder_without_opening_model(self):
        from slstack.mac import native_vision_path
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'DeepSeek-V4.1-Flash-1017-Q4-v41.gguf'
            encoder=model.with_name('DeepSeek-V4.1-Flash-1017-Vision.gguf')
            model.write_bytes(b'test path only');encoder.write_bytes(b'test encoder path only')
            self.assertEqual(native_vision_path(dict(model=str(model))),str(encoder.resolve()))
            self.assertIsNone(native_vision_path(dict(model=str(Path(d)/'other/model.gguf'))))
            with self.assertRaises(ValueError):native_vision_path(dict(model=str(model),vision_encoder=str(Path(d)/'absent.gguf')))

if __name__=='__main__':unittest.main()
