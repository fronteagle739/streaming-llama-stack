"""Golden Mac payload layout, independent of the C++ writer's implementation."""
import struct
import unittest
import copy
import io
from unittest.mock import patch
from pathlib import Path
from slstack.model_compatibility import describe, model_description, require_compatible


def golden(mtp=True,narrow=False):
    u32=lambda *x:struct.pack('<'+'I'*len(x),*x)
    f32=lambda x:struct.pack('<'+'f'*len(x),*x)
    out=u32(0x34565344,2,16,4,16,16,4,5,5,2,2,16,0x51573802)
    out+=u32(1,2,3,4,5)+f32([.25]*16)+u32(5 if mtp else 0)
    for layer in range(5):
        if layer<3:
            out+=f32([layer*1000+i for i in range(8)])
            out+=f32([layer*1000+c*3+t for t in range(3) for c in range(6)])
        else:
            if layer==4 and not mtp:continue
            half=lambda count:b''.join(struct.pack('<H',0x3c00+layer*16+i) for i in range(count))
            out+=half(10)*2
            out+=half(10) if narrow else f32([struct.unpack('<e',struct.pack('<H',0x3c00+layer*16+i))[0] for i in range(10)])
            out+=half(2)
    out+=f32([c*6+t for t in range(6) for c in range(4)])
    out+=u32(5,4,3,2,1,0,0,0)+u32(0)
    out+=b''.join(u32(i,i,i,0) for i in range(5))
    return out


class QfnStateTests(unittest.TestCase):
    def test_native_golden_fixture(self):
        # A real native fixture is supplied by the guarded build/qualification.
        fixture=Path(__file__).resolve().parents[2]/'state-fixture.bin'
        if not fixture.exists():self.skipTest('Run qfn-engine --state-codec-fixture under the task guardian')
        self.assertEqual(fixture.read_bytes(),golden())

    def test_native_narrow_index_frontier(self):
        fixture=Path(__file__).resolve().parents[2]/'state-fixture-narrow-index.bin'
        if not fixture.exists():self.skipTest('Run native narrow-index codec fixture')
        self.assertEqual(fixture.read_bytes(),golden(narrow=True))
        self.assertEqual(len(golden())-len(golden(narrow=True)),40)

    def test_native_mtp_off_frontier(self):
        fixture=Path(__file__).resolve().parents[2]/'state-fixture-mtp-off.bin'
        if not fixture.exists():self.skipTest('Run the native zero-MTP-frontier fixture')
        self.assertEqual(fixture.read_bytes(),golden(False))

    def header(self):
        required=('block_count','embedding_length','attention.key_length','attention.head_count_kv',
            'ssm.state_size','ssm.time_step_rank','ssm.conv_kernel','full_attention_interval',
            'hyper_connection.count','attention.indexer.key_length','attention.compress_ratios',
            'ple.ngram_size','ple.conv_kernel','ple.eos_token_id')
        meta={'general.architecture':'qwen4exp','tokenizer.ggml.tokens':['a','b']}
        meta.update({'qwen4exp.'+k:1 for k in required})
        return dict(metadata=meta,tensors=[dict(name='blk.0.ssm_a',shape=[2],type=0)])

    def test_qfn_is_independent_and_cross_precision_requires_qualification(self):
        h=self.header();a=describe(h,'a'*64)
        self.assertEqual(a['contract']['family_tag'],0x51573802)
        h['tensors'][0]['type']=8
        self.assertTrue(require_compatible(a,describe(h,'b'*64)))
        self.assertEqual(a['contract']['kv_bits'],16)
        with self.assertRaises(ValueError):describe(h,'b'*64,2541)
        h['metadata']['qwen4exp.ple.ngram_size']=3
        with self.assertRaises(ValueError):require_compatible(a,describe(h,'b'*64))

    def test_incomplete_qfn_never_admitted(self):
        h=self.header();h['metadata'].pop('qwen4exp.ssm.state_size')
        with self.assertRaises(ValueError):describe(h,'a'*64)

    def test_source_metadata_matches_actual_float32_wire_values(self):
        source=self.header();source['metadata']['general.sampling.temperature']=.6
        source['tensors'][0]['operation']='dense'
        receiver=copy.deepcopy(source)
        receiver['metadata']['general.sampling.temperature']=struct.unpack('<f',struct.pack('<f',.6))[0]
        with patch('pathlib.Path.open',return_value=io.BytesIO(b'SLPACK\0\0')), \
             patch('scripts.qfn_mac_convert.plan',return_value=(None,None,source,None)):
            producer=model_description('fixture.slpack','a'*64,0)
        self.assertTrue(require_compatible(producer,describe(receiver,'b'*64)))
        receiver['metadata']['general.sampling.temperature']=.7
        with self.assertRaises(ValueError):require_compatible(producer,describe(receiver,'b'*64))


if __name__=='__main__':unittest.main()
