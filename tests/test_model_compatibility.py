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


if __name__=='__main__':unittest.main()
