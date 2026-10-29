"""Independent Q4_K decoding/rounding oracle and interrupted file conversion."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
try:
    import numpy as np
except ImportError:
    np=None
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('requant',ROOT/'scripts/dsv41_requant.py')
rq=importlib.util.module_from_spec(spec);spec.loader.exec_module(rq)
LIBRARY=os.environ.get('NATIVEFP4_REQUANT_LIBRARY')

@unittest.skipUnless(np is not None and LIBRARY,'Set NATIVEFP4_REQUANT_LIBRARY to the built CPU adapter')
class RequantTests(unittest.TestCase):
    def test_legacy_dspark_metadata_uses_verified_target_configuration(self):
        values=dict(block_size=5,markov_rank=256,noise_token_id=128799,n_routed_experts=128,num_experts_per_tok=3)
        meta={'general.architecture':'deepseek41-dspark','general.source.url':'fixture','general.source.revision':'a'*40,'deepseek41.dspark.stage_count':3,'deepseek41.dspark.n_layers':3}
        meta.update({'deepseek41.dspark.'+k:v for k,v in values.items()})
        config={'dspark_'+k:v for k,v in values.items()};config.update(dspark_target_layer_ids=[37,38,39],num_hidden_layers=40)
        target=dict(metadata={'general.architecture':'deepseek41','general.source.url':'fixture','general.source.revision':'a'*40,'deepseek41.config':json.dumps(dict(text_config=config))})
        fields=rq.dspark_aliases(dict(metadata=meta),target)
        self.assertEqual(fields['dspark.target_layer_ids'],[37,38,39])
        self.assertEqual(fields['dspark.n_routed_experts'],128)
        with self.assertRaisesRegex(ValueError,'matching target'):rq.dspark_aliases(dict(metadata=meta))
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            rq.dspark_aliases(dict(metadata=dict(meta,**{'dspark.block_size':99})),target)
        bad=dict(target['metadata'],**{'general.source.revision':'b'*40})
        with self.assertRaisesRegex(ValueError,'source identity'):rq.dspark_aliases(dict(metadata=meta),dict(metadata=bad))
    def test_conversion_reservation_filters_only_its_exact_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ledger';path.write_text('reservation')
            own='ACTIVE: NATIVEFP4 own; exclusive configured GPU'
            foreign='ACTIVE: OTHER foreign; exclusive configured GPU'
            with patch.object(rq,'_conversion_hold',('NATIVEFP4','own')):
                with patch.object(rq,'active_holds',return_value=[own,foreign]):
                    self.assertEqual(rq.gpu_holds(path),[foreign])
                path.write_text('own reservation only')
                with patch.object(rq,'active_holds',return_value=[own]):
                    self.assertFalse(rq.gpu_holds(path))
                path.write_text('reservation changed')
                with patch.object(rq,'active_holds',return_value=[]):
                    self.assertTrue(rq.gpu_holds(path))

    def test_unreadable_gpu_ledger_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ledger';path.write_text('')
            with patch.object(Path,'read_text',side_effect=PermissionError('writer lock')):
                self.assertTrue(rq.gpu_holds(path))

    def test_gpu_ledger_cache_observes_a_new_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ledger';path.write_text('')
            with patch.object(rq,'active_holds',side_effect=lambda text:['active'] if text else []) as parse:
                self.assertFalse(rq.gpu_holds(path));self.assertFalse(rq.gpu_holds(path))
                self.assertEqual(parse.call_count,1)
                path.write_text('ACTIVE: timed run')
                self.assertTrue(rq.gpu_holds(path))
                self.assertEqual(parse.call_count,2)

    def test_atomic_receipt_retries_windows_reader_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'state.json'
            original=os.replace
            calls=[]
            def replace(a,b):
                calls.append(1)
                if len(calls)==1:raise PermissionError('temporary reader lock')
                return original(a,b)
            with patch.object(rq.os,'replace',replace),patch.object(rq.time,'sleep'):
                rq.durable_json(target,{'status':'test'})
            self.assertEqual(json.loads(target.read_text()),{'status':'test'})
            self.assertEqual(len(calls),2)

    def test_independent_float32_decoder_and_nearest_code_oracle(self):
        rng=np.random.default_rng(1017)
        raw=rng.integers(0,256,(4096,144),dtype=np.uint8)
        raw[:,:4]=rng.uniform(.00001,.003,(4096,2)).astype('<f2').view(np.uint8).reshape(-1,4)
        raw[0]=0
        scales=np.empty((len(raw),8),np.float32);mins=scales.copy()
        p=raw[:,4:16]
        scales[:,:4]=p[:,:4]&63;mins[:,:4]=p[:,4:8]&63
        scales[:,4:]=(p[:,8:12]&15)|((p[:,:4]>>6)<<4)
        mins[:,4:]=(p[:,8:12]>>4)|((p[:,4:8]>>6)<<4)
        quant=raw[:,16:].reshape(-1,4,32)
        codes=np.stack((quant&15,quant>>4),axis=2).reshape(-1,8,32)
        base=raw[:,:4].copy().view('<f2').astype(np.float32)
        values=(base[:,0,None]*scales)[:,:,None]*codes-(base[:,1,None]*mins)[:,:,None]
        blocks=values.reshape(-1,32)
        maximum=np.max(np.abs(blocks),axis=1)
        exponent=np.zeros(len(blocks),np.uint8)
        positive=maximum>0
        exponent[positive]=(np.floor(np.log2(maximum[positive]))-2+127).astype(np.uint8)
        lut=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
        scale=np.ldexp(np.ones(len(blocks),np.float32),exponent.astype(np.int32)-127)
        distances=np.abs(blocks[:,:,None]-scale[:,None,None]*lut[None,None,:])
        nearest=distances.argmin(axis=2).astype(np.uint8)
        expected=np.empty((len(blocks),17),np.uint8);expected[:,0]=exponent
        expected[:,1:]=nearest[:,:16]|(nearest[:,16:]<<4)
        convert=rq.converter(Path(LIBRARY))
        self.assertEqual(convert(raw.tobytes(),1),expected.tobytes())
        self.assertEqual(convert(raw.tobytes(),4),expected.tobytes())
        with self.assertRaises(ValueError):convert(b'bad',1)

    def test_dspark_resume_and_unchanged_dense_payload(self):
        self.test_durable_resume_and_unchanged_dense_payload('deepseek41-dspark')

    def test_dspark_rejects_unknown_fields_and_missing_experts(self):
        with self.assertRaisesRegex(ValueError,'target architecture'):
            rq.validate_dspark(dict(metadata={'dspark.target_architecture':'deepseek4'},tensors=[]))
        h=dict(metadata={'general.architecture':'deepseek41-dspark'},tensors=[
            dict(name='mtp.0.ffn_gate_exps.weight',type=12,shape=[256,1,1])])
        with self.assertRaisesRegex(ValueError,'Incomplete'):rq.validate_dspark(h)
        h['tensors']=[dict(name='mtp.0.unknown.weight',type=0)]
        with self.assertRaisesRegex(ValueError,'Unsupported'):rq.validate_dspark(h)

    def test_durable_resume_and_unchanged_dense_payload(self,architecture='deepseek41'):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            hold=root/'hold';hold.write_text('')
            names=['blk.0.ffn_'+part+'_exps.weight' for part in ('gate','up','down')]+['output.weight']
            if architecture=='deepseek41-dspark':
                names=[n.replace('blk.0.','mtp.0.') for n in names[:3]]+['mtp.0.main_proj.weight']
            tensors=[dict(name=name,shape=[256,1,1],type=12,offset=i*16384) for i,name in enumerate(names[:3])]
            tensors+=[dict(name=names[3],shape=[32,1],type=8,offset=3*16384)]
            prefix=b'GGUF'+struct.pack('<IQQ',3,4,2)
            prefix+=rq.string('general.architecture')+struct.pack('<I',8)+rq.string(architecture)
            prefix+=rq.string('general.alignment')+struct.pack('<II',4,16384)
            descriptors=b''.join(rq.string(t['name'])+struct.pack('<I',len(t['shape']))+struct.pack('<'+'Q'*len(t['shape']),*t['shape'])+struct.pack('<IQ',t['type'],t['offset']) for t in tensors)
            raw=prefix+descriptors;raw+=bytes(rq.align(len(raw),16384)-len(raw))
            for t in tensors:
                data=bytes(range(34)) if t['type']==8 else bytes(144)
                raw+=data+bytes(16384-len(data))
            source=root/'source.gguf';source.write_bytes(raw)
            args=SimpleNamespace(requant_gguf=source,out=root/'out.gguf',runtime_root=root,
                                 hold_file=hold,quants_library=Path(LIBRARY),resume=False,threads=2,runtime_floor_gib=4)
            with patch('shutil.disk_usage',side_effect=[SimpleNamespace(free=10**12),SimpleNamespace(free=20*2**30)]):
                with self.assertRaisesRegex(ValueError,'20 GiB reserve'):rq.worker(args)
            self.assertEqual(json.loads((root/'requant-progress.json').read_text())['completed'],[])
            args.resume=True
            original=rq.converter
            calls=[]
            def fail_after_first(lib):
                fn=original(lib)
                def run(data,threads):
                    calls.append(1)
                    if len(calls)==2:raise InterruptedError('test interruption')
                    return fn(data,threads)
                return run
            with patch.object(rq,'converter',fail_after_first),patch('shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
                with self.assertRaises(InterruptedError):rq.worker(args)
            self.assertEqual(len(json.loads((root/'requant-progress.json').read_text())['completed']),1)
            args.resume=True
            partial=args.out.with_name(args.out.name+'.partial')
            data_start=rq.align(rq.read_header(partial)['header_bytes'],16384)
            with partial.open('r+b') as f:f.seek(data_start);f.write(b'\x01')
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):rq.worker(args)
            with partial.open('r+b') as f:f.seek(data_start);f.write(b'\x00')
            args.defer_readback=True
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):rq.worker(args)
            h=rq.read_header(args.out)
            self.assertEqual(h['metadata']['deepseek41.speed_proxy'],rq.LABEL)
            if architecture=='deepseek41-dspark':
                self.assertEqual(h['metadata']['dspark.target_architecture'],'deepseek41')
            self.assertEqual([t['type'] for t in h['tensors']],[39,39,39,8])
            start=rq.align(h['header_bytes'],16384)+h['tensors'][-1]['offset']
            with args.out.open('rb') as f:f.seek(start);self.assertEqual(f.read(34),bytes(range(34)))
            self.assertEqual(source.read_bytes(),raw)
            receipt=json.loads((root/'requant-progress.json').read_text())
            self.assertEqual(receipt['status'],'COMPLETE')
            self.assertFalse(receipt['readback_verified'])
            self.assertEqual(receipt['sha256'],hashlib.sha256(args.out.read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
