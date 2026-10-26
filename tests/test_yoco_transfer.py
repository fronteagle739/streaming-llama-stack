import hashlib
import struct
import unittest
from unittest.mock import Mock
from slstack.native_broker import vendor
from slstack.mac import Receiver

vendor()
from studio.dsv41_session_contract import initial_request,MODEL_SHA256


class YocoTransferTests(unittest.TestCase):
    def test_explicit_provenance_required_and_old_exact_peer_unchanged(self):
        body=struct.pack('<2I',2,3)
        headers={'content-type':'application/vnd.studio.dsv41.tokens-le32','content-length':str(len(body)),
                 'x-dsv41-model-sha256':MODEL_SHA256,'x-dsv41-mode':'3','x-dsv41-context':'1000001',
                 'x-dsv41-tokens':'2','x-dsv41-tokens-sha256':hashlib.sha256(body).hexdigest(),'x-dsv41-tier':'ultra'}
        self.assertNotIn('yoco_keep',initial_request('a'*32,headers,body))
        for keep,version in [('140','0'),('139','1'),('0','1'),('140','2')]:
            with self.subTest(keep=keep,version=version),self.assertRaises(ValueError):
                initial_request('a'*32,dict(headers,**{'x-dsv41-yoco-keep':keep,'x-dsv41-yoco-state-version':version}),body)
        parsed=initial_request('a'*32,dict(headers,**{'x-dsv41-yoco-keep':'140','x-dsv41-yoco-state-version':'1'}),body)
        self.assertEqual((parsed['yoco_keep'],parsed['yoco_state_version']),(140,1))

    def test_receiver_binds_yoco_to_authenticated_pair(self):
        receiver=Receiver.__new__(Receiver);receiver.close=Mock();receiver.build={'binary_sha256':'c'*64}
        row=dict(producer_sha256='a'*64,producer_source_revision='b'*40,producer_ipv4='127.0.0.1',broker_port=8090,tier='ultra')
        self.assertEqual(receiver.pair(row,'127.0.0.1')['status'],'candidate')
        with self.assertRaises(ValueError):receiver.pair(dict(row,yoco_keep=140),'127.0.0.1')
        accepted=receiver.pair(dict(row,yoco_keep=140,yoco_state_version=1),'127.0.0.1')
        self.assertEqual(accepted['yoco_keep'],140)
        with self.assertRaises(ValueError):receiver.pair(dict(row,yoco_keep=True,yoco_state_version=1),'127.0.0.1')


if __name__=='__main__':unittest.main()
