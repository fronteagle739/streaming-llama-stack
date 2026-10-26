import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from slstack.combo_contract import validate, VERSION, owner_layout
from slstack.combo_state import verify_payload
from slstack.local_cache import Payload, validate_contract, Cache, VERSION as LOCAL_VERSION


class LocalCacheTests(unittest.TestCase):
    def setUp(self):
        # Retain exact test evidence; workspace retirement requires owner approval.
        self.root=Path(tempfile.mkdtemp(prefix='slstack-cache-proof-'))

    def payload(self):
        pos=3;context=999424;tokens=struct.pack('<3I',2,3,4);logits=b'\0'*(129280*4)
        header=struct.pack('<13I',0x34565344,2,context,1,128,128,context+1,pos,40,512,128,129280,0x413431)
        owners=[];chunks=[]
        for (kind,index),size in owner_layout(pos).items():
            block=b'\0'*size;chunks.append(block);owners.append(dict(kind=kind,index=index,bytes=size,sha256=hashlib.sha256(block).hexdigest()))
        path=self.root/'state.kv';path.write_bytes(header+tokens+logits+b''.join(chunks))
        row=dict(version=VERSION,mode=1,kv_dtype='f32',context=context,position=pos,request_id='1'*32,owner_epoch=1,generation=1,
                 tokens_sha256=hashlib.sha256(tokens).hexdigest(),logits_sha256=hashlib.sha256(logits).hexdigest(),owners=owners,decode_provenance='phase-bias')
        row.update({k:'a'*64 for k in ('model_sha256','tokenizer_sha256','template_sha256','rope_sha256')})
        return path,row

    def test_streamed_payload_verifies_all_regions_without_ram_staging(self):
        path,row=self.payload();proof=verify_payload(Payload(path),row,local=True)
        self.assertEqual(proof['payload_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        with path.open('r+b') as f:f.seek(-1,2);f.write(b'x')
        with self.assertRaises(ValueError):verify_payload(Payload(path),row,local=True)

    def test_yoco_and_precision_provenance_are_required(self):
        _,row=self.payload();row['local_state']=dict(version=LOCAL_VERSION,yoco_keep=140,cache_precision='bf16',producer_sha256='b'*64)
        validate_contract(row)
        with self.assertRaises(ValueError):validate(row)
        for field,value in [('yoco_keep',139),('cache_precision','q4'),('version','unknown')]:
            changed=json.loads(json.dumps(row));changed['local_state'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(changed)

    def test_reader_checks_bounds_and_closes_nested_leases(self):
        path,_=self.payload();reader=Payload(path);self.assertEqual(len(reader.read(0,52)),52)
        with reader.lease():
            with reader.lease():self.assertEqual(reader.read(0,4),b'DSV4')
        self.assertIsNone(reader.stream)
        for offset,count in [(-1,1),(reader.capacity,1),(0,65537)]:
            with self.subTest(offset=offset,count=count),self.assertRaises(ValueError):reader.read(offset,count)


if __name__=='__main__':unittest.main()
