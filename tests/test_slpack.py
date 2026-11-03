"""Bounded independent .slpack and raw-NVFP4 source fixtures.

Run with SLPACK_FIXTURE_ROOT inside the caller's sole task scratch. Fixture
files are retained, never removed automatically. No engine or listener starts.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.slpack import MAGIC,HEADER,ENTRY,HEADER_BYTES,ModelWriter,ModelFile,Stream
from scripts.pack_slpack import NativeWriter,BLOB


class PackFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['SLPACK_FIXTURE_ROOT']).resolve()
        cls.root.mkdir(parents=True,exist_ok=True)

    def fixture(self,name,raw):
        path=self.root/name
        if path.exists():
            self.assertEqual(path.read_bytes(),raw)
        else:
            with path.open('xb') as stream:stream.write(raw)
        return path

    def test_deterministic_envelope_all_payloads_and_parts(self):
        paths=[self.root/('slpack-envelope-'+label+'.slpack') for label in ('a','b')]
        for path in paths:
            if path.exists():
                continue
            writer=ModelWriter(path)
            writer.add_bytes('meta',b'{"unknown_future_field":{"keep":true}}')
            writer.add_parts('gguf',[b'abc',Stream(5,[b'de',b'fgh'])])
            writer.add_bytes('vision/fixture',bytes(range(256)))
            writer.add_bytes('mtp/fixture',bytes(range(255,-1,-1)))
            writer.close()
        a,b=[ModelFile(p) for p in paths]
        self.assertEqual(a.verify_all()['sha256'],b.verify_all()['sha256'])
        self.assertEqual(a.names(),['meta','gguf','vision/fixture','mtp/fixture'])
        self.assertEqual(a.read('gguf'),b'abc'+bytes(4093)+b'defgh')
        with paths[0].open('rb') as stream:
            self.assertEqual(stream.read(12),MAGIC+struct.pack('<I',1))
        self.assertTrue(a.verify_all()['bytes']<9<<20)

    def test_explicit_legacy_reader_and_reserved_fields(self):
        base=self.root/'slpack-envelope-a.slpack'
        if not base.exists():self.test_deterministic_envelope_all_payloads_and_parts()
        raw=bytearray(base.read_bytes())
        raw[:12]=bytes.fromhex('424153414c544d46')+struct.pack('<I',2)
        legacy=self.fixture('slpack-legacy-fixture.bin',bytes(raw))
        with self.assertRaisesRegex(ValueError,'legacy migration'):
            ModelFile(legacy)
        self.assertEqual(ModelFile(legacy,allow_legacy=True).read('vision/fixture'),bytes(range(256)))
        raw=bytearray(base.read_bytes());raw[24]=1
        with self.assertRaisesRegex(ValueError,'Unknown active header'):
            ModelFile(self.fixture('slpack-active-header.slpack',bytes(raw)))
        raw=bytearray(base.read_bytes());struct.pack_into('<Q',raw,64+128+64,2<<20)
        with self.assertRaisesRegex(ValueError,'overlapping'):
            ModelFile(self.fixture('slpack-overlap.slpack',bytes(raw)))
        raw=bytearray(base.read_bytes());raw[64+80]^=1
        damaged=ModelFile(self.fixture('slpack-bad-sha.slpack',bytes(raw)))
        with self.assertRaisesRegex(ValueError,'Section SHA256'):
            damaged.verify_all()

    def test_source_nvfp4_fixture_through_native_writer(self):
        import numpy as np
        package=ROOT/'scripts/slpack_qfn_source'
        sys.path.insert(0,str(package))
        checkpoint=importlib.import_module('checkpoint')
        arrays,header,payload=[],{},bytearray()
        expected=[]
        for role,n,k in [('gate_proj',640,2560),('up_proj',640,2560),('down_proj',2560,640)]:
            codes=np.arange(n*(k//2),dtype=np.uint8).reshape(n,k//2)
            scales=(np.arange(n*(k//16),dtype=np.uint8)%127).reshape(n,k//16)
            expected.append(codes.tobytes()+scales.reshape(n,k//64,4).transpose(1,0,2).tobytes())
            prefix='model.language_model.layers.0.mlp.experts.0.'+role+'.'
            for suffix,dtype,shape,data in [('weight','U8',[n,k//2],codes.tobytes()),
                    ('weight_scale','F8_E4M3',[n,k//16],scales.tobytes()),
                    ('weight_scale_2','F32',[],struct.pack('<f',.25)),
                    ('input_scale','F32',[],struct.pack('<f',.125))]:
                start=len(payload);payload.extend(data)
                header[prefix+suffix]=dict(dtype=dtype,shape=shape,data_offsets=[start,len(payload)])
        encoded=json.dumps(header,separators=(',',':')).encode()
        source=self.fixture('model-00001-of-00001.safetensors',struct.pack('<Q',len(encoded))+encoded+payload)
        config={'text_config':dict(num_hidden_layers=1,num_experts=1,hidden_size=2560,moe_intermediate_size=640)}
        self.fixture('config.json',json.dumps(config).encode())
        ck=checkpoint.Checkpoint(source.parent)
        experts=checkpoint.CheckpointExperts(ck)
        # Call the production source job, preserving source input-scale bytes independently.
        raw=checkpoint._expert_job(next(experts.jobs(1)))
        self.assertEqual(len(raw),BLOB)
        target=self.root/'slpack-source-native-fixture.slpack'
        if not target.exists():
            writer=NativeWriter(target)
            writer.add_chunks('experts',[raw[:257],raw[257:]])
            writer.add_bytes('experts.layout',experts.layout_text())
            writer.add_bytes('experts.input_scale',experts.input_scale_bytes)
            writer.add_bytes('meta',b'{"fixture":true,"preserved_field":{"enabled":true}}')
            writer.close()
        packed=ModelFile(target)
        self.assertEqual(packed.read('experts'),b''.join(expected)+struct.pack('<4f',.25,.25,.25,0))
        self.assertEqual(packed.read('experts.input_scale'),struct.pack('<3f',.125,.125,.125))
        self.assertEqual(json.loads(packed.read('meta'))['preserved_field'],{'enabled':True})
        self.assertTrue(packed.verify_all()['sha256'])

    def test_sampler_defaults_unsupported_active_field_refused(self):
        sys.path.insert(0,str(ROOT/'scripts/slpack_qfn_source'))
        sampling=importlib.import_module('sampling')
        self.assertEqual(sampling.sampling_defaults({'gguf_kv':{'general.sampling.top_p':.949999988079071},'sampling':{'top_p':.95}})['top_p'],.95)
        for metadata in [{'gguf_kv':{'general.sampling.missing':.3}}, {'sampling':{'unimplemented':True}}]:
            with self.assertRaises(ValueError):sampling.sampling_defaults(metadata)


if __name__=='__main__':
    unittest.main()
