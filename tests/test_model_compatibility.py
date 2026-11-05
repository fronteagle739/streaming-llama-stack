import copy
import unittest
from slstack.model_compatibility import describe, require_compatible, continuation_kl


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.header = dict(metadata={'general.architecture':'deepseek41',
            'deepseek41.num_hidden_layers':40, 'deepseek41.source_revision':'checkpoint-1017',
            'tokenizer.ggml.tokens':['a','b'], 'tokenizer.ggml.bos_token_id':0},
            tensors=[dict(name='blk.0.ffn_gate_exps.weight',shape=[5120,2304,384],type=12)])

    def test_different_weights_and_storage_types_are_compatible(self):
        a = describe(self.header,'a'*64,2541)
        self.header['tensors'][0]['type']=39
        self.header['metadata']['deepseek41.quantization']='lossy requant speed proxy'
        b = describe(self.header,'b'*64,2541)
        self.assertTrue(require_compatible(a,b))
        self.assertNotEqual(a['quant'], b['quant'])

    def test_shape_tokenizer_checkpoint_and_mode_mismatches_refuse(self):
        a=describe(self.header,'a'*64,2541)
        for field, value in [('tokenizer.ggml.tokens',['a','c']),
                             ('deepseek41.num_hidden_layers',39),
                             ('deepseek41.source_revision','another-checkpoint')]:
            h=copy.deepcopy(self.header);h['metadata'][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_compatible(a,describe(h,'b'*64,2541))
        h=copy.deepcopy(self.header);h['tensors'][0]['shape'][0]=4096
        with self.assertRaises(ValueError):require_compatible(a,describe(h,'b'*64,2541))
        with self.assertRaises(ValueError):require_compatible(a,describe(self.header,'b'*64,140))
        h=copy.deepcopy(a);h['contract']['wire_version']=3
        with self.assertRaises(ValueError):require_compatible(a,h)

    def test_full_vocabulary_kl_checks_extent_and_normalizes(self):
        import struct
        a=struct.pack('<4f',1,2,3,4);b=struct.pack('<4f',2,3,4,5)
        self.assertAlmostEqual(continuation_kl(a,b,2)['mean_nats'],0)
        with self.assertRaises(ValueError):continuation_kl(a,b[:-4],2)
        with self.assertRaises(ValueError):continuation_kl(a,struct.pack('<4f',1,2,float('nan'),4),2)

    def test_file_comparison_excludes_imported_frontier_and_preserves_all_rows(self):
        import struct,tempfile
        from pathlib import Path
        from slstack.model_compatibility import continuation_kl_files
        # Keep the tiny fixture under the guardian's configured temp root;
        # this task's evidence must not be removed during qualification.
        directory=tempfile.mkdtemp(prefix='qfn-rows-')
        a,b=Path(directory)/'control',Path(directory)/'candidate'
        a.write_bytes(struct.pack('<6f',99,0,1,2,3,4))
        b.write_bytes(struct.pack('<4f',2,3,4,5))
        result=continuation_kl_files(a,b,2,1,0)
        self.assertEqual(result['rows'],2)
        self.assertEqual(result['top1'],1)
        self.assertEqual(len(result['per_token_nats']),2)
        self.assertAlmostEqual(result['mean_nats'],0)
        with self.assertRaises(ValueError):continuation_kl_files(a,b,2,0,0)
        b.write_bytes(struct.pack('<4f',2,3,float('nan'),5))
        with self.assertRaises(ValueError):continuation_kl_files(a,b,2,1,0)

    def test_pairing_and_observation_settings_reach_the_mac_engine(self):
        from slstack.config import load, environment
        settings={key:'a'*64 for key in ['DS4_NATIVE_SESSION_MODEL_SHA256',
            'DS4_NATIVE_SESSION_RECEIVER_MODEL_SHA256','DS4_NATIVE_SESSION_COMPATIBILITY_SHA256']}
        settings['DS4_NATIVE_QUALIFICATION_LOGITS']='/explicit/qualification.f32'
        env=environment(load(overrides={'env_overrides':settings}), 'ds4-mac')
        for key,value in settings.items():self.assertEqual(env.get(key),value,key)


if __name__=='__main__':unittest.main()
