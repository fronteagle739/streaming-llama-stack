import copy
import importlib.util
from pathlib import Path
import unittest
from slstack.model_compatibility import describe
from tests import test_qfn_state as fixtures

spec=importlib.util.spec_from_file_location('qfn_receiver',Path(__file__).resolve().parents[1]/'examples/dual-machine/qfn/receive.py')
receiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(receiver)


class QfnAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.key=b'k'*32;self.nonce='a'*32
        self.local=describe(fixtures.QfnStateTests().header(),'b'*64)
        self.row=dict(schema='qfn-paired-stream-v1',nonce=self.nonce,producer_binary_sha256='c'*64,
            model_compatibility=describe(fixtures.QfnStateTests().header(),'a'*64),verify_state=False,payload_bytes=1024)
        self.sign()

    def sign(self):
        self.row.pop('hmac',None);self.row['hmac']=receiver.signature(self.key,self.row)

    def test_authenticated_cross_quant_candidate(self):
        receiver.validate_handshake(self.row,self.key,self.nonce,self.local,2048)

    def test_unknown_fields_and_tampering_refused(self):
        for field,value in [('verify_state',True),('payload_bytes',2049),('nonce','f'*32),('ignored_sampler',True)]:
            row=copy.deepcopy(self.row);row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):receiver.validate_handshake(row,self.key,self.nonce,self.local,2048)

    def test_valid_signature_cannot_override_geometry_or_budget(self):
        self.row['payload_bytes']=2049;self.sign()
        with self.assertRaises(ValueError):receiver.validate_handshake(self.row,self.key,self.nonce,self.local,2048)
        self.row['payload_bytes']=1024
        self.row['model_compatibility']['contract']['mtp_alignment']='different'
        from slstack.model_compatibility import digest
        self.row['model_compatibility']['compatibility_sha256']=digest(self.row['model_compatibility']['contract']);self.sign()
        with self.assertRaises(ValueError):receiver.validate_handshake(self.row,self.key,self.nonce,self.local,2048)


if __name__=='__main__':unittest.main()
