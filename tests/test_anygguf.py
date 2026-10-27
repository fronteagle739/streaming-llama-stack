"""CPU/header tests for file integrity, quant types and offline conversion plans."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from slstack.gguf_header import read_header, validate_dsv41_types
from slstack.native_broker import model_receipt, model_digest, vendor
vendor()
from studio.dsv41_session_contract import command, Refused
from studio.dsv41_session_process import Producer

ROOT=Path(__file__).resolve().parents[1]


def gguf(tensors=(), *, architecture='deepseek41'):
    def string(value):
        value=value.encode();return struct.pack('<Q',len(value))+value
    data=b'GGUF'+struct.pack('<IQQ',3,len(tensors),1)
    data+=string('general.architecture')+struct.pack('<I',8)+string(architecture)
    for name,kind in tensors:
        data+=string(name)+struct.pack('<IQIQ',1,256,kind,0)
    return data


class GGUFTests(unittest.TestCase):
    def test_public_quant_layouts_and_named_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'public.gguf'
            for group in ((12,12,12),(16,16,10),(39,39,39)):
                file.write_bytes(gguf([('blk.0.ffn_'+part+'_exps.weight',kind) for part,kind in zip(('gate','up','down'),group)]))
                validate_dsv41_types(read_header(file))
            for kind in (14,999):
                file.write_bytes(gguf([('blk.7.ffn_up_exps.weight',kind)]))
                with self.assertRaisesRegex(ValueError,r'blk.7.ffn_up_exps.weight.*\('+str(kind)+r'\)'):
                    validate_dsv41_types(read_header(file))
            file.write_bytes(gguf([('blk.0.ffn_'+p+'_exps.weight',kind) for p,kind in zip(('gate','up','down'),(16,12,10))]))
            with self.assertRaisesRegex(ValueError,'triplet'):validate_dsv41_types(read_header(file))

    def test_header_budget_and_duplicate_tensor_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'tiny.gguf'
            file.write_bytes(gguf([('output.weight',8)]))
            with self.assertRaisesRegex(ValueError,'budget'):read_header(file,limit=24)
            file.write_bytes(gguf([('output.weight',8),('output.weight',8)]))
            with self.assertRaisesRegex(ValueError,'Invalid GGUF tensor'):read_header(file)
            file.write_bytes(gguf([('output.weight',8)])[:-2])
            with self.assertRaisesRegex(ValueError,'Truncated'):read_header(file)

    def test_unsupported_type_fails_before_any_full_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'unsupported.gguf';file.write_bytes(gguf([('output.weight',999)]))
            with mock.patch('slstack.native_broker.sha256',side_effect=AssertionError('payload hash')):
                with self.assertRaisesRegex(ValueError,'output.weight.*999'):
                    model_receipt(dict(model=str(file),runtime_root=folder))

    def test_receipts_are_for_selected_files_and_reject_stale_or_pinned_digests(self):
        with tempfile.TemporaryDirectory() as folder:
            first=Path(folder)/'public-iq2.gguf';second=Path(folder)/'public-mxfp4.gguf'
            cfg=dict(model=str(first),runtime_root=folder)
            first.write_bytes(gguf([('output.weight',8)])+b'first fixture')
            second.write_bytes(gguf([('output.weight',8)])+b'other fixture')
            with mock.patch('builtins.print'):
                a=model_digest(cfg);b=model_digest(dict(cfg,model=str(second)))
                self.assertNotEqual(a,b)
                self.assertEqual(a,hashlib.sha256(first.read_bytes()).hexdigest())
                with mock.patch('slstack.native_broker.sha256',side_effect=AssertionError('cached hash')):
                    self.assertEqual(b,model_digest(dict(cfg,model=str(second))))
                first.write_bytes(first.read_bytes()+b'changed')
                self.assertNotEqual(a,model_digest(cfg))
                with self.assertRaisesRegex(ValueError,'hash/identity differs'):
                    model_receipt(dict(cfg,model_sha256=a))

    def test_session_commands_and_replies_keep_per_instance_model_identity(self):
        for digest in ('a'*64,'b'*64):
            request=dict(session='1'*32,model_sha256=digest,sequence=1,operation='close')
            command(request,'1'*32,1,model_sha256=digest)
            with self.assertRaises(Refused):command(request,'1'*32,1,model_sha256='c'*64)
            producer=Producer.__new__(Producer);producer.session='1'*32;producer.owner_epoch=0;producer.model_sha256=digest
            producer.failure=None
            producer.output=io.BytesIO(json.dumps(request).encode()+b'\n')
            self.assertEqual(producer._receive(),request)
            producer.output=io.BytesIO(json.dumps(dict(request,model_sha256='c'*64)).encode()+b'\n')
            with self.assertRaises(Refused):producer._receive()


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=os.environ.get('ANYGGUF_DS4_TOOLS')
        if not directory:raise unittest.SkipTest('Set ANYGGUF_DS4_TOOLS to the pinned source-only gguf-tools directory')
        spec=importlib.util.spec_from_file_location('convert_dsv41',ROOT/'scripts/convert_dsv41_gguf.py')
        cls.converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.converter)
        cls.modules=cls.converter.load_tools(directory)
        fixture=json.loads((ROOT/'tests/fixtures/dsv41-source.json').read_text())
        cls.config=fixture['config'];cls.index=dict(weight_map={})
        for pattern,count in fixture['source_patterns'].items():
            for expert in range(count):cls.index['weight_map'][pattern.format(expert=expert)]='absent.safetensors'
        assert len(cls.index['weight_map'])==fixture['indexed_tensors']
        cls.reference=cls.converter.read_json(ROOT/'scripts/dsv41/reference-layout.json')['components']

    def test_all_three_plans_need_only_config_and_index(self):
        with mock.patch.object(self.modules['glm53_quantize'],'SourceDB',side_effect=AssertionError('opened shard')), \
             mock.patch.object(self.modules['deepseek41_quantize'],'NativeQuantizer',side_effect=AssertionError('loaded quantizer')):
            for component,count in (('main',1046),('dspark',81),('vision',266)):
                ref=self.reference[component]['tensors']
                plan=self.converter.make_plan(component,self.modules,self.config,self.index,ref)
                self.assertEqual(len(plan),count)
                self.assertEqual(self.converter.planned_map(plan),ref)

    def test_bad_source_inventory_and_config_cannot_pass_reference(self):
        index=copy.deepcopy(self.index);index['weight_map'].pop('layers.0.ffn.experts.17.w1.weight')
        with self.assertRaisesRegex(ValueError,'Missing indexed source tensor'):
            self.converter.make_plan('main',self.modules,self.config,index,self.reference['main']['tensors'])
        config=copy.deepcopy(self.config);config['text_config']['hidden_size']+=256
        with self.assertRaisesRegex(ValueError,'Planned shape differs'):
            self.converter.make_plan('main',self.modules,config,self.index,self.reference['main']['tensors'])

    def test_source_pin_rejects_modified_converter(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'Pinned antirez/ds4 source required'):
                self.converter.load_tools(folder)


if __name__=='__main__':unittest.main()
