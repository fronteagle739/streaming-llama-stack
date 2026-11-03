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
from scripts.slpack import MAGIC,HEADER,ENTRY,HEADER_BYTES,ModelWriter,ModelFile,Stream,VerificationWriter
from scripts.pack_slpack import NativeWriter,BLOB,canonical_metadata,check_source


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
        verifier=NativeWriter(self.root/'no-native-output.slpack',verify_against=target)
        verifier.add_chunks('experts',[raw[:257],raw[257:]])
        verifier.add_bytes('experts.layout',experts.layout_text())
        verifier.add_bytes('experts.input_scale',experts.input_scale_bytes)
        verifier.add_bytes('meta',b'{"fixture":true,"preserved_field":{"enabled":true}}')
        self.assertEqual(verifier.close()['sha256'],packed.verify_all()['sha256'])
        self.assertFalse((self.root/'no-native-output.slpack').exists())

    def test_complete_virtual_hash_and_hc8_readback(self):
        sys.path.insert(0,str(ROOT/'scripts/slpack_qfn_source'))
        convert=importlib.import_module('convert')
        hc8=importlib.import_module('hc8')
        dense=struct.pack('<64H',*([0x3f80]*64))
        index=b'output_hc_down.weight 0 4 0 128 0 0 32 2\n'
        mtp_index=b'hyper_connection.fixture bf16 2 32 0 128\n'
        profile=b'fixture profile with all bytes preserved'
        historical=dict(format='historical',version=2,converter=dict(tool='old/converter',retained=True),
            checkpoint=dict(dir='machine-source',sections_from=dict(path='machine-base',
                meta_sha256='a'*64,profile_export='machine-output.profile',
                profile_runtime_args=['--expert-profile-save','machine-output.profile'],
                section_sha256=dict(profile=hashlib.sha256(profile).hexdigest()),extra={'keep':True})),
            retained_unknown=['sampler',{'vision':True}])
        canonical=canonical_metadata(historical)
        self.assertEqual(canonical['retained_unknown'],historical['retained_unknown'])
        moved=json.loads(json.dumps(historical))
        moved['checkpoint']['dir']='different-source'
        moved['checkpoint']['sections_from']['path']='different-base'
        moved['checkpoint']['sections_from']['profile_export']='different.profile'
        moved['checkpoint']['sections_from']['profile_runtime_args'][1]='different.profile'
        self.assertEqual(canonical_metadata(moved),canonical)
        def emit(writer,metadata):
            job=convert.Job(writer)
            for name,data in [('dense',dense),('dense.index',index),('mtp/dense',dense),
                              ('mtp/dense.index',mtp_index),('profile',profile)]:
                job.bytes(name,data)
            job.hc8()
            job.bytes('meta',(json.dumps(metadata,ensure_ascii=False,separators=(',',':'))+'\n').encode())
            return writer.close()
        reference=self.root/'raw-verification-historical.slpack'
        expected=self.root/'raw-verification-canonical.slpack'
        if not reference.exists():emit(ModelWriter(reference),historical)
        if not expected.exists():emit(ModelWriter(expected),canonical)
        verifier=VerificationWriter(reference,canonical_metadata)
        report=emit(verifier,canonical)
        self.assertEqual(report['sha256'],ModelFile(expected).verify_all()['sha256'])
        self.assertEqual(report['bytes'],expected.stat().st_size)
        self.assertEqual(report['reference_sha256'],ModelFile(reference).verify_all()['sha256'])
        copies=hc8.read_table(ModelFile(expected).read('hc8'))
        self.assertEqual(list(copies),['output_hc_down.weight','mtp.hyper_connection.fixture'])
        for _,_,_,raw in copies.values():
            self.assertEqual(raw[:64],bytes([127]*64))
            self.assertEqual(len(raw),96)
        # Matching regenerated bytes/table do not admit corrupt reference readback.
        damaged=bytearray(reference.read_bytes())
        damaged[ModelFile(reference).section('dense')[0]]^=1
        corrupted=self.fixture('raw-verification-corrupt.slpack',bytes(damaged))
        verifier=VerificationWriter(corrupted,canonical_metadata)
        verifier.add_bytes('dense',dense)
        with self.assertRaisesRegex(ValueError,'readback section SHA256'):
            verifier.readback('dense',0,2)
        invalid=json.loads(json.dumps(historical))
        invalid['checkpoint']['sections_from']['profile_runtime_args'].append('--future-option')
        with self.assertRaisesRegex(ValueError,'Unsupported profile runtime arguments'):
            canonical_metadata(invalid)

    def test_sampler_defaults_unsupported_active_field_refused(self):
        sys.path.insert(0,str(ROOT/'scripts/slpack_qfn_source'))
        sampling=importlib.import_module('sampling')
        self.assertEqual(sampling.sampling_defaults({'gguf_kv':{'general.sampling.top_p':.949999988079071},'sampling':{'top_p':.95}})['top_p'],.95)
        for metadata in [{'gguf_kv':{'general.sampling.missing':.3}}, {'sampling':{'unimplemented':True}}]:
            with self.assertRaises(ValueError):sampling.sampling_defaults(metadata)

    def test_pinned_source_authentication_returns_stable_identities(self):
        raw=b'{"complete_source_metadata":true}\n'
        path=self.fixture('source-auth-config.json',raw)
        recipe={'source':{'files':[dict(name=path.name,bytes=len(raw),
            hash_algorithm='sha256',hash=hashlib.sha256(raw).hexdigest())]}}
        identities=check_source(path.parent,recipe)
        self.assertEqual(list(identities),[str(path)])
        recipe['source']['files'][0]['hash']='0'*64
        with self.assertRaisesRegex(ValueError,'Pinned source hash differs'):
            check_source(path.parent,recipe)


if __name__=='__main__':
    unittest.main()
