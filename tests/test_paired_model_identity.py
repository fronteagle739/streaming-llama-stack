import hashlib
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from slstack.native_broker import vendor
vendor()
from studio.dsv41_session_contract import MODEL_SHA256, Refused, initial_request
from studio.dsv41_session_broker import Broker, Record, create_router

FP4='0ba5ae3bdeea8c38501d2a69e6394436b6f99c07b5a531226a3f93249c45e572'
SID='1234567890abcdef1234567890abcdef'


class PairedModelIdentity(unittest.TestCase):
    def headers(self, identity):
        self.tokens=struct.pack('<3I',1,2,3)
        return {'content-type':'application/vnd.studio.dsv41.tokens-le32',
                'content-length':str(len(self.tokens)), 'x-dsv41-model-sha256':identity,
                'x-dsv41-mode':'3','x-dsv41-context':'1000001','x-dsv41-tokens':'3',
                'x-dsv41-tokens-sha256':hashlib.sha256(self.tokens).hexdigest(),
                'x-dsv41-tier':'ultra','x-dsv41-yoco-keep':'2541','x-dsv41-yoco-state-version':'2'}

    def test_http_initial_request_binds_to_trusted_model_and_keeps_yoco(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        for identity in [MODEL_SHA256,FP4]:
            broker=Broker(Mock(),model_sha256=identity)
            broker.start=Mock(return_value={'state':'starting'})
            app=FastAPI();app.include_router(create_router(broker,lambda:None))
            headers=self.headers(identity)
            with TestClient(app) as client:
                reply=client.post('/dsv41-native-offload/sessions/'+SID,headers=headers,content=self.tokens)
                self.assertEqual(reply.status_code,202,reply.text)
                request=broker.start.call_args.args[0]
                self.assertEqual(request['model_sha256'],identity)
                self.assertEqual((request['yoco_keep'],request['yoco_state_version']),(2541,2))
                self.assertEqual(request['tokens'],self.tokens)
                broker.start.reset_mock()
                headers['x-dsv41-model-sha256']='f'*64
                self.assertEqual(client.post('/dsv41-native-offload/sessions/'+SID,headers=headers,content=self.tokens).status_code,409)
                broker.start.assert_not_called()

    def test_default_q4_contract_and_explicit_fp4_refuse_each_others_identity(self):
        headers=self.headers(MODEL_SHA256)
        self.assertEqual(initial_request(SID,headers,self.tokens)['model_sha256'],MODEL_SHA256)
        with self.assertRaises(Refused):initial_request(SID,headers,self.tokens,FP4)
        headers=self.headers(FP4)
        with self.assertRaises(Refused):initial_request(SID,headers,self.tokens)
        with self.assertRaises(Refused):Broker(Mock(),model_sha256='invalid')
        broker=Broker(Mock(),model_sha256=FP4)
        with self.assertRaises(Refused):broker.start({'session':SID,'model_sha256':MODEL_SHA256})
        self.assertIsNone(broker.active)

    def test_commands_retain_session_sequence_and_model_binding(self):
        for identity in [MODEL_SHA256,FP4]:
            broker=Broker(Mock(),model_sha256=identity)
            record=Record({'session':SID});record.state='ready'
            record.producer=SimpleNamespace(sequence=7)
            broker.records[SID]=record
            row=dict(session=SID,model_sha256=identity,sequence=7,operation='close')
            for update in [dict(model_sha256='e'*64),dict(sequence=6),dict(session='2'*32)]:
                with self.assertRaises(Refused):broker._claim(SID,{**row,**update})
                self.assertFalse(record.busy)
            got,producer=broker._claim(SID,row)
            self.assertIs(got,record);self.assertIs(producer,record.producer)
            self.assertTrue(record.busy)


if __name__=='__main__':unittest.main()
