import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('breakdown',Path(__file__).resolve().parents[1]/'scripts/analyze_dsv41_prefill.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Breakdown(unittest.TestCase):
    def test_hint_io_counts_bytes_without_becoming_demand_wait(self):
        log='''[dsv41-fetch] stage=0 bytes=100 fetch_ms=10
[dsv41-prefill-experts] stage=22 rows=6 bytes=20 fetch_ms=2
[dsv41-byte-ledger] stage=0 tensor=token_embd.weight file_offset=0 bytes=100 source=stage
[dsv41-byte-ledger] stage=22 tensor=blk.21.ffn_up_exps.weight file_offset=100 bytes=20 source=experts
[dsv41-prefill-hint] stage=23 bytes=30 fetch_ms=3.5 begin_epoch_ms=110.5 end_epoch_ms=114.5 exposed_join_ms=0.75
[dsv41-byte-ledger] stage=23 tensor=blk.22.ffn_up_exps.weight file_offset=9007199254740993 bytes=30 source=tail_hint
[dsv41-perf] layer=21 wall_ms=40 acquire_ms=1 compute_ms=30
[dsv41-fetch] stage=0 bytes=100 fetch_ms=10
'''
        result=module.analyze(log);p=result['passes'][0]
        self.assertEqual(p['fetched_bytes'],150)
        self.assertEqual(p['fetch_ms_overlapped'],15.5)
        self.assertEqual(p['routed_fetch_wait_ms'],2)
        self.assertEqual(p['hint_fetched_bytes'],30)
        self.assertEqual(p['hint_fetch_ms_overlapped'],3.5)
        self.assertEqual(p['exposed_hint_join_ms'],0.75)
        self.assertEqual(p['expert_hints'][0]['begin_epoch_ms'],110.5)
        self.assertEqual(p['expert_hints'][0]['end_epoch_ms'],114.5)
        self.assertEqual(p['exposed_acquire_ms'],1)
        self.assertTrue(p['tensor_ledger_reconciles'])
        self.assertEqual(p['tensor_bytes_by_role']['routed_experts'],50)
        self.assertEqual(result['passes'][1]['hint_fetched_bytes'],0)
        self.assertEqual(result['passes'][1]['expert_hints'],[])
        # A partial hinted pass cannot be reconstructed as legacy full stages.
        hinted=module.analyze('[dsv41-fetch] stage=0 bytes=100 fetch_ms=10\n[dsv41-prefill-hint] stage=23 bytes=30 fetch_ms=3 begin_epoch_ms=1 end_epoch_ms=4 exposed_join_ms=0\n')
        module.attach_inventory(hinted,[])
        self.assertEqual(hinted['passes'][0]['fetched_bytes'],130)
        self.assertIsNone(hinted['passes'][0]['tensor_ledger_reconciles'])

    def test_demand_read_default_uses_measured_short_prompt_range(self):
        from slstack.combo_producer import routed_prefill_choice
        for n in (1,2048):self.assertEqual(routed_prefill_choice({},n),'1')
        for n in (2049,8192,32768,200000):self.assertEqual(routed_prefill_choice({},n),'0')
        for profile in ('accurate','fast'):self.assertEqual(routed_prefill_choice(dict(producer_profile=profile),2048),'0')
        self.assertEqual(routed_prefill_choice(dict(env_overrides={'LLAMADECODE_PREFILL_ROUTED':'0'}),2048),'0')
        self.assertEqual(routed_prefill_choice(dict(env_overrides={'LLAMADECODE_PREFILL_ROUTED':'1'}),32768),'1')
        with self.assertRaises(ValueError):routed_prefill_choice(dict(env_overrides={'LLAMADECODE_PREFILL_ROUTED':'auto'}),2048)
        self.assertEqual(routed_prefill_choice(dict(yoco_keep=2541),100000),'tail')
        from scripts.dsv41_single import overlay
        cone=overlay('262k',profile='ultra-cone')
        self.assertEqual((cone['producer_profile'],cone['yoco_keep']),('ultra',2541))

    def test_inventory_and_legacy_pass_multiplicity(self):
        header=dict(metadata={},header_bytes=33,tensors=[
            dict(name='token_embd.weight',type=1,shape=[32,2],offset=0),
            dict(name='blk.0.ffn_up_exps.weight',type=12,shape=[256,1,384],offset=128),
            dict(name='blk.1.engram_embd.weight',type=24,shape=[264,2],offset=128+144*384)])
        rows=module.inventory(header)
        self.assertEqual(rows[0]['file_offset'],64)
        self.assertEqual(rows[1]['bytes'],144*384)
        self.assertIsNone(rows[2]['stage'])
        d=module.analyze('[dsv41-fetch] stage=0 bytes=128 fetch_ms=1\n[dsv41-fetch] stage=0 bytes=128 fetch_ms=1\n')
        module.attach_inventory(d,rows)
        self.assertEqual(d['tensor_read_totals']['token_embd.weight'],dict(bytes=256,read_extents=2))
        self.assertNotIn(rows[2]['tensor'],d['tensor_read_totals'])
        d['passes'][0]['fetches'][0]['bytes']=127
        d['passes'][0]['tensors']=[]
        with self.assertRaises(ValueError):module.attach_inventory(d,rows)

    def test_exact_extents_repeated_reads_and_separate_engram_io(self):
        log='''[dsv41-engram-prefetch] read_bytes=512 requests=1
[dsv41-fetch] stage=0 bytes=20 fetch_ms=1
[dsv41-byte-ledger] stage=0 tensor=token_embd.weight file_offset=9007199254740993 bytes=20 source=stage
[dsv41-fetch] stage=1 bytes=10 fetch_ms=1
[dsv41-byte-ledger] stage=1 tensor=blk.0.attn_norm.weight file_offset=40 bytes=10 source=stage
[dsv41-byte-ledger] stage=1 tensor=blk.0.ffn_up_exps.weight file_offset=100 bytes=30 source=experts first=1 count=3
[dsv41-prefill-experts] stage=1 rows=2 active=3 new=3 resident=3 bytes=30 fetch_ms=2
[dsv41-prefill-experts] stage=1 rows=2 active=3 new=0 resident=3 bytes=0 fetch_ms=0
DS4PERF_LAYER_WALL_LEDGER layer=0 wall_ms=9 acquire_ms=1 executor_wall_ms=7
'''
        result=module.analyze(log);p=result['passes'][0]
        self.assertEqual(p['fetched_bytes'],60)
        self.assertTrue(p['tensor_ledger_reconciles'])
        self.assertEqual(p['tensor_bytes_by_role']['routed_experts'],30)
        self.assertEqual(p['tensors'][0]['file_offset'],9007199254740993)
        self.assertEqual(p['routed_fetch_wait_ms'],2)
        self.assertEqual(p['wall_totals']['executor_wall_ms'],7)
        self.assertEqual(result['engram_reads'][0]['read_bytes'],512)

    def test_fetches_are_counted_once_and_not_added_to_exposed_wait(self):
        log='''[dsv41-fetch] stage=0 bytes=100 fetch_ms=10
[dsv41-fetch] stage=1 bytes=200 fetch_ms=20
[dsv41-perf] layer=0 wall_ms=40 acquire_ms=1 compute_ms=30
[dsv41-fetch] stage=0 bytes=100 fetch_ms=10
[dsv41-fetch] stage=1 bytes=200 fetch_ms=20
[dsv41-perf] layer=0 wall_ms=42 acquire_ms=2 compute_ms=31
DS4PERF_NODE_PROFILE_HEADER
DS4PERF_NODE_PROFILE\t3\t5.0\tMUL_MAT_ID|a=blk.0.ffn_up_exps.weight
DS4PERF_NODE_PROFILE\t1\t4.0\tFLASH_ATTN_EXT_SPARSE|node=sparse
DS4PERF_NODE_PROFILE\t1\t2.0\tRMS_NORM|node=norm
'''
        result=module.analyze(log)
        self.assertEqual([p['fetched_bytes'] for p in result['passes']],[300,300])
        self.assertEqual([p['exposed_acquire_ms'] for p in result['passes']],[1,2])
        self.assertEqual(result['gpu_node_ms'],dict(expert_gemm=5,attention_and_indexer=4,other_gpu=2))
