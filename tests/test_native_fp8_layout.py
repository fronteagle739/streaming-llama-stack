"""Lossless native FP8 byte slots and refusal of canonical interpretation."""
import os,tempfile,types,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from scripts import dsv41_fp8_layout as layout
from scripts import convert_dsv41_gguf as converter
from slstack.gguf_header import validate_dsv41_types

class Source:
    def __init__(self):
        self.values=(np.arange(128*128,dtype=np.uint32)%127).astype(np.uint8).tobytes()
        self.scales=bytes(range(16))
        self.tensors={'dense.weight':self.info('dense.weight'),'dense.scale':self.info('dense.scale')}
    def info(self,name):
        sf=name.endswith('.scale');return dict(dtype='F8_E8M0' if sf else 'F8_E4M3',shape=[4,4] if sf else [128,128],nbytes=16 if sf else 16384)
    def read(self,name):return self.scales if name.endswith('.scale') else self.values

class NativeDenseLayoutTests(unittest.TestCase):
    def test_values_and_every_scale_byte_are_copied_without_quantization(self):
        source=Source();actual=layout.pack_source(source,'dense.weight',(128,128))
        self.assertEqual(actual[:16384],source.values)
        self.assertEqual(actual[16384:16400],source.scales)
        self.assertEqual(actual[16400:],bytes(128*128*34//32-16400))
        self.assertEqual(len(actual),128*128*34//32)

    def test_nonfinite_native_operands_fail_before_writing(self):
        source=Source();source.scales=b'\xff'+source.scales[1:]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):layout.pack_source(source,'dense.weight',(128,128))
        source=Source();source.values=b'\xff'+source.values[1:]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):layout.pack_source(source,'dense.weight',(128,128))

    def test_marker_cannot_be_loaded_as_canonical_q8(self):
        header=dict(metadata={'general.architecture':'deepseek41','general.alignment':16384,
            'deepseek41.expert_layout':'flashinfer-sm120-mxfp8-mxfp4-v1',layout.KEY:layout.LAYOUT},tensors=[])
        with self.assertRaises(ValueError):validate_dsv41_types(header)
        validate_dsv41_types(header,expert_layout=header['metadata']['deepseek41.expert_layout'],dense_layout=layout.LAYOUT)
        header['metadata'][layout.KEY]='unknown'
        with self.assertRaises(ValueError):validate_dsv41_types(header,expert_layout=header['metadata']['deepseek41.expert_layout'],dense_layout='unknown')

    def test_pinned_writer_bypasses_float_quantizer_for_native_dense(self):
        tools=os.environ.get('ANYGGUF_DS4_TOOLS')
        if not tools:self.skipTest('Pinned converter tools required')
        modules=converter.load_tools(tools);q=modules['deepseek41_quantize'];g=modules['glm53_quantize']
        item=g.TensorPlan('blk.0.attn_q_a.weight',(128,128),8,role='dense',source='dense.weight')
        item.offset,item.nbytes=0,128*128*34//32
        class NoQuantizer:
            np=np
            def __init__(self,*args):pass
            def to_f32(self,*args):raise AssertionError('Native FP8 was dequantized')
            def encode(self,*args):raise AssertionError('Native FP8 was requantized')
        source=Source()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'native-dense.gguf'
            args=types.SimpleNamespace(quants_library='unused',imatrix=None,out=str(output),resume=False,threads=1)
            with mock.patch.object(q,'NativeQuantizer',NoQuantizer),mock.patch.object(q.shutil,'disk_usage',return_value=types.SimpleNamespace(free=1<<40)):
                converter.native_writer(q,kernel_layout=True,native_dense=True)(args,[item],[],source)
            h=converter.read_header(output);start=(h['header_bytes']+16383)//16384*16384
            self.assertEqual(output.read_bytes()[start:start+item.nbytes],layout.pack_source(source,'dense.weight',(128,128)))

if __name__=='__main__':unittest.main()
