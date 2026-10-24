from contextlib import contextmanager
import hashlib
import struct
import unittest
from slstack.combo_contract import owner_layout
from slstack.combo_state import payload_bytes,contract_from_checkpoint,verify_payload


class RamFixture:
    def __init__(self,raw):self.raw=raw;self.capacity=len(raw)
    def read(self,off,n):assert n<=65536;return bytes(self.raw[off:off+n])
    @contextmanager
    def lease(self):yield self


def state(position):
    layout=owner_layout(position);tokens=struct.pack('<'+'I'*position,*range(position));logits=struct.pack('<f',.25)*129280
    pieces=[struct.pack('<13I',0x34565344,2,262144,1,128,128,262145,position,40,512,128,129280,0x413431),tokens,logits]
    cp=dict(context=262144,position=position,mode=1,capture_end=0,capture_length=0,
            tokens_sha256=hashlib.sha256(tokens).hexdigest(),logits_sha256=hashlib.sha256(logits).hexdigest(),owner_sha256=[])
    for i,(key,count) in enumerate(layout.items()):
        block=bytes([i])*count;pieces.append(block);cp['owner_sha256'].append(hashlib.sha256(block).hexdigest())
    identity={k:'a'*64 for k in ('model_sha256','tokenizer_sha256','template_sha256','rope_sha256')}
    row=contract_from_checkpoint(cp,identity,'1'*32,1,1,'unbiased')
    return RamFixture(bytearray(b''.join(pieces))),row


class StateTest(unittest.TestCase):
    def test_actual_owner_bytes_and_native_headers(self):
        for position in (1,2,127,128,129):
            ram,row=state(position);proof=verify_payload(ram,row,hashlib.sha256(ram.raw).hexdigest())
            self.assertEqual(proof['bytes'],payload_bytes(position))
            self.assertTrue(proof['all_owner_hashes_verified'])

    def test_corrupt_header_tokens_logits_and_owner_refuse(self):
        for where in ('header','token','logit','owner'):
            ram,row=state(2)
            if where=='header':ram.raw[12]=3
            if where=='token':ram.raw[52:56]=struct.pack('<I',129280)
            if where=='logit':ram.raw[60:64]=struct.pack('<I',0x7f800000)
            if where=='owner':ram.raw[-1]^=1
            with self.subTest(where=where),self.assertRaises(ValueError):verify_payload(ram,row)

    def test_refuse_capacity_and_expected_payload_mismatch(self):
        ram,row=state(2);ram.capacity-=1
        with self.assertRaises(MemoryError):verify_payload(ram,row)
        ram.capacity+=1
        with self.assertRaises(ValueError):verify_payload(ram,row,'0'*64)


if __name__=='__main__':unittest.main()
