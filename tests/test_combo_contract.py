import copy
import json
import unittest

from slstack.combo_contract import VERSION, owner_layout, parse, validate, same_checkpoint, ram_checkpoint_bytes
from studio.dsv41_session_contract import checkpoint as transport_checkpoint, command, MODEL_SHA256


def checkpoint(position=129):
    row = dict(version=VERSION, mode=1, kv_dtype='f32', context=262144, position=position,
               request_id='1'*32, owner_epoch=1, generation=2, decode_provenance='unbiased')
    row.update({k: 'a'*64 for k in ('model_sha256','tokenizer_sha256','template_sha256',
                'rope_sha256','tokens_sha256','logits_sha256')})
    row['owners'] = [dict(kind=k,index=i,bytes=n,sha256='b'*64) for (k,i),n in owner_layout(position).items()]
    return row


class ContractTest(unittest.TestCase):
    def test_kind_index_survives_order_change(self):
        base=checkpoint();other=copy.deepcopy(base);other['owners'].reverse()
        self.assertTrue(same_checkpoint(other,base))
        self.assertEqual(len(validate(checkpoint(128))['owners']),48)
        self.assertEqual(len(validate(base)['owners']),54)

    def test_contract_refusals(self):
        base=checkpoint()
        changes={'version':'other','mode':3,'model_sha256':'c'*64,'tokenizer_sha256':'c'*64,
                 'template_sha256':'c'*64,'rope_sha256':'c'*64,'kv_dtype':'q4_0',
                 'context':1048576,'request_id':'2'*32,'owner_epoch':2,'position':0,
                 'generation':True,'decode_provenance':'unknown'}
        for key,value in changes.items():
            with self.subTest(key=key):
                changed=copy.deepcopy(base);changed[key]=value
                with self.assertRaises(ValueError):validate(changed,base)

    def test_checkpoint_hashes_and_generation(self):
        base=checkpoint()
        for key,value in [('generation',3),('tokens_sha256','c'*64),('logits_sha256','c'*64),('decode_provenance','manual-bias')]:
            changed=copy.deepcopy(base);changed[key]=value
            with self.assertRaises(ValueError):same_checkpoint(changed,base)
        changed=copy.deepcopy(base);changed['owners'][0]['sha256']='c'*64
        with self.assertRaises(ValueError):same_checkpoint(changed,base)

    def test_owner_refusals(self):
        base=checkpoint()
        for edit in ('duplicate','dspark','extent','missing','nonfinite'):
            changed=copy.deepcopy(base)
            if edit=='duplicate':changed['owners'][1]=changed['owners'][0].copy()
            if edit=='dspark':changed['owners'][0]['kind']='dspark'
            if edit=='extent':changed['owners'][0]['bytes']+=4
            if edit=='missing':changed['owners'].pop()
            if edit=='nonfinite':changed['owners'][0]['bytes']=float('nan')
            with self.subTest(edit=edit),self.assertRaises(ValueError):parse(json.dumps(changed).encode())

    def test_bounded_strict_parser(self):
        with self.assertRaises(ValueError):parse(b'{"mode":1,"mode":3}')
        with self.assertRaises(ValueError):parse(b' '*((1<<20)+1))
        self.assertEqual(parse(json.dumps(checkpoint()).encode())['mode'],1)

    def test_262k_budget(self):
        size=ram_checkpoint_bytes(checkpoint(262143))
        self.assertGreater(size,1_600_000_000)
        self.assertLess(size,1_750_000_000)

    def test_transport_mode_is_explicit(self):
        base=dict(context=262144,position=129,mode=1,capture_end=0,capture_length=0,
                  tokens_sha256='a'*64,logits_sha256='b'*64,owner_sha256=['c'*64]*54)
        self.assertEqual(transport_checkpoint(base,1),base)
        with self.assertRaises(ValueError):transport_checkpoint(base)
        legacy={**base,'mode':3,'capture_end':129,'capture_length':128,'owner_sha256':['c'*64]*55}
        self.assertEqual(transport_checkpoint(legacy),legacy)
        with self.assertRaises(ValueError):transport_checkpoint(legacy,1)
        request=dict(session='1'*32,model_sha256=MODEL_SHA256,sequence=1,operation='save_rewind',base=base)
        self.assertEqual(command(request,'1'*32,1,1),request)
        with self.assertRaises(ValueError):command(request,'1'*32,1)


if __name__=='__main__':unittest.main()
