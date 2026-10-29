"""Independent byte placement, lossless source repack and loader refusals."""
import hashlib
import io
import types
import unittest
import struct
import json
import tempfile
from pathlib import Path
from unittest import mock
import numpy as np
from scripts.dsv41_fp4_layout import FLASHINFER,pack_expert,unpack_expert,requant_chunks,validate_layout
from scripts.convert_dsv41_gguf import write_native_experts
from slstack.gguf_header import validate_dsv41_types
from slstack.gguf_header import read_header
from scripts import dsv41_requant as requant

class NativeLayoutTests(unittest.TestCase):
    def test_codes_signed_zero_and_every_finite_scale_have_exact_byte_positions(self):
        n,k=128,512
        blocks=np.arange(n*k//32*17,dtype=np.uint32).reshape(n,k//32,17).astype(np.uint8)
        blocks[:,:,0]=(np.arange(n*k//32).reshape(n,k//32)%255).astype(np.uint8)
        original=blocks.tobytes();values,scales=pack_expert(original,n,k)
        for column in range(n):
            for kb in range(k//32):
                self.assertEqual(scales[((kb//4)*n+column)*4+kb%4],int(blocks[column,kb,0]))
                codes=list(blocks[column,kb,1:]&15)+list(blocks[column,kb,1:]>>4)
                at=(column*(k//32)+kb)*16
                self.assertEqual(list(values[at:at+16]),[int(codes[2*j])|(int(codes[2*j+1])<<4) for j in range(16)])
        self.assertEqual(unpack_expert(values,scales,n,k),original)

    def test_malformed_geometry_or_scale_never_enters_converter(self):
        for n,k in [(127,128),(128,127),(0,128),(-128,128)]:
            with self.assertRaises(ValueError):pack_expert(b'',n,k)
        raw=bytearray(128*128//32*17);raw[0]=255
        with self.assertRaisesRegex(ValueError,'Nonfinite'):pack_expert(raw,128,128)
        with self.assertRaises(ValueError):validate_layout('unqualified-b12x')

    def test_official_source_repack_has_all_values_before_all_scales(self):
        n=k=128
        packed=np.arange(n*k//2,dtype=np.uint32).astype(np.uint8).reshape(n,k//2)
        scales=np.arange(n*k//32,dtype=np.uint32).astype(np.uint8).reshape(n,k//32)%255
        class Source:
            def info(self,name):return dict(dtype='F8_E8M0' if name.endswith('.scale') else 'I8',shape=list(scales.shape if name.endswith('.scale') else packed.shape))
            def read(self,name):return (scales if name.endswith('.scale') else packed).tobytes()
        out=io.BytesIO();item=types.SimpleNamespace(shape=[k,n,384],source='expert.{expert}.weight')
        write_native_experts(out,item,Source())
        raw=out.getvalue();boundary=384*n*k//2
        self.assertEqual(raw[:boundary],packed.tobytes()*384)
        expected=scales.reshape(n,k//128,4).transpose(1,0,2).tobytes()*384
        self.assertEqual(raw[boundary:],expected)

    def test_requant_emits_one_expert_at_a_time_and_hashes_every_source_byte(self):
        n=k=128;expert_bytes=n*k//256*144
        raw=bytes(range(256))*(384*expert_bytes//256)
        source=io.BytesIO(raw);h=hashlib.sha256();calls=[]
        def convert(data,threads):
            calls.append(len(data));return (bytes([127])+bytes(16))*(n*k//32)
        chunks=list(requant_chunks(source,{'type':12},{'type':39,'shape':[k,n,384]},convert,1,lambda:None,h))
        self.assertEqual(calls,[expert_bytes]*384)
        self.assertEqual(h.hexdigest(),hashlib.sha256(raw).hexdigest())
        self.assertEqual(b''.join(chunks),bytes(384*n*k//2)+bytes([127])*(384*n*k//32))

    def test_canonical_loaders_refuse_packed_and_unknown_layouts(self):
        def header(layout=FLASHINFER):
            return dict(metadata={'general.architecture':'deepseek41','general.alignment':16384,'deepseek41.expert_layout':layout},
                tensors=[dict(name=f'blk.0.ffn_{kind}_exps.weight',type=39,shape=shape,offset=i*16384)
                    for i,(kind,shape) in enumerate([('gate',[5120,2304,384]),('up',[5120,2304,384]),('down',[2304,5120,384])])])
        with self.assertRaisesRegex(ValueError,'matching prefill loader'):validate_dsv41_types(header())
        validate_dsv41_types(header(),expert_layout=FLASHINFER)
        with self.assertRaises(ValueError):validate_dsv41_types(header('unknown'),expert_layout='unknown')
        bad=header();bad['tensors'][0]['offset']=1
        with self.assertRaisesRegex(ValueError,'mismatch'):validate_dsv41_types(bad,expert_layout=FLASHINFER)
        bad=header();bad['tensors'][0]['type']=12
        with self.assertRaisesRegex(ValueError,'mismatch'):validate_dsv41_types(bad,expert_layout=FLASHINFER)
        bad=header();bad['tensors'][0]['shape']=[2304,5120,384]
        with self.assertRaisesRegex(ValueError,'mismatch'):validate_dsv41_types(bad,expert_layout=FLASHINFER)

    def test_full_gguf_conversion_resume_identity_readback_and_dense_preservation(self):
        library=Path(__file__).resolve().parents[2]/'bin/requant-avx2.dll'
        if not library.is_file():
            self.skipTest('Built CPU Q4 requant library required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.gguf';output=root/'packed.gguf'
            meta=(requant.string('general.architecture')+struct.pack('<I',8)+requant.string('deepseek41')+
                  requant.string('general.alignment')+struct.pack('<II',4,16384))
            descriptors=[];payloads=[];offset=0
            q4=struct.pack('<ee',.25,.125)+bytes([1])*12+bytes([0x21])*128
            for kind in ['gate','up','down']:
                payload=q4*(384*128*128//256)
                descriptors.append(requant.string(f'blk.0.ffn_{kind}_exps.weight')+struct.pack('<IQQQIQ',3,128,128,384,12,offset))
                payloads.append(payload+bytes(requant.align(len(payload),16384)-len(payload)))
                offset+=len(payloads[-1])
            dense=bytes(range(34))
            descriptors.append(requant.string('output.weight')+struct.pack('<IQQIQ',2,32,1,8,offset))
            payloads.append(dense+bytes(16384-len(dense)))
            header=b'GGUF'+struct.pack('<IQQ',3,4,2)+meta+b''.join(descriptors)
            source.write_bytes(header+bytes(requant.align(len(header),16384)-len(header))+b''.join(payloads))
            args=types.SimpleNamespace(requant_gguf=source,out=output,runtime_root=root,expert_layout=FLASHINFER,
                quants_library=library,threads=1,hold_file=root/'no-holds',runtime_floor_gib=.5,resume=False,defer_readback=False)
            with mock.patch.object(requant,'gpu_holds',return_value=[]):requant.worker(args)
            h=read_header(output);self.assertEqual(h['metadata']['deepseek41.expert_layout'],FLASHINFER)
            start=requant.align(h['header_bytes'],16384);raw=output.read_bytes()
            tensor=h['tensors'][-1];self.assertEqual(raw[start+tensor['offset']:start+tensor['offset']+34],dense)
            journal=json.loads((root/'requant-native-progress.json').read_text())
            self.assertTrue(journal['readback_verified']);self.assertEqual(journal['sha256'],hashlib.sha256(raw).hexdigest())
            with mock.patch.object(requant,'gpu_holds',return_value=[]),self.assertRaises(FileExistsError):requant.worker(args)

if __name__=='__main__':unittest.main()
