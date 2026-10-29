"""The optional adapter must fit the existing device envelope before launch."""
import unittest
from slstack.combo_producer import prefill_arguments

class FlashInferAdmissionTests(unittest.TestCase):
    def config(self,setting=None):
        return dict(context=1000001,producer_profile='ultra',producer_host_budget_gib=6.25,
            producer_host_max_gib=7.75,ctx_cache_precision='bf16',ctx_cache_growable=True,
            yoco_keep=140,env_overrides={} if setting is None else {'LLAMA_DSV41_FLASHINFER_MXFP4':setting})

    def test_explicit_off_preserves_the_entire_existing_plan(self):
        for count in [2048,49555,100000]:
            self.assertEqual(prefill_arguments(self.config(),count,24<<30),
                             prefill_arguments(self.config('0'),count,24<<30))

    def test_optional_workspace_repartitions_without_raising_any_device_cap(self):
        original,plan=prefill_arguments(self.config(),100000,24<<30)
        candidate,candidate_plan=prefill_arguments(self.config('1'),100000,24<<30)
        self.assertEqual(plan,candidate_plan)
        changes={key for key in original if original[key]!=candidate[key]}
        self.assertEqual(changes,{'--backend-temporary-reserve-bytes'})
        reserve=int(candidate['--backend-temporary-reserve-bytes'])
        self.assertGreater(int(candidate['--graph-bytes'])-reserve,0)
        # Independently total all five buffers at the largest admitted tile.
        for n,k in [(2304,5120),(5120,2304)]:
            routes=6*4096
            padded_scale_rows=((routes+384*3+127)//128)*128
            buffers=[384*n*k//2,384*n*k//32,routes*k,padded_scale_rows*k//32,routes*n*2]
            rounded=sum((size+255)//256*256 for size in buffers)
            module_headroom=16<<20
            other_temporaries=448<<20
            # ABI3 uses a fixed maximum allocation to avoid legacy-pool growth
            # when gate/up and down request different sizes. Admit its lookahead.
            maximum_routes=6*4096
            scale_rows=((maximum_routes+384*3+127)//128)*128
            fixed_buffers=[384*5120*2304//2,384*5120*2304//32,
                maximum_routes*2304,scale_rows*2304//32,maximum_routes*5120*2]
            fixed_rounded=sum((size+255)//256*256 for size in fixed_buffers)
            self.assertLessEqual(rounded,fixed_rounded)
            pooled=((int(1.05*(fixed_rounded+255))+255)//256)*256
            self.assertLess(pooled+module_headroom+other_temporaries,reserve)

    def test_incompatible_arithmetic_profile_is_rejected_before_engine_creation(self):
        cfg=self.config('1');cfg['producer_profile']='accurate'
        with self.assertRaisesRegex(ValueError,'paired ULTRA'):
            prefill_arguments(cfg,2048,24<<30)

    def test_larger_expert_chunks_preserve_total_device_and_host_ceilings(self):
        base,plan=prefill_arguments(self.config('1'),100000,24<<30)
        for rows,reserve in [('8192',4096),('16384',5120)]:
            cfg=self.config('1');cfg['yoco_keep']=2541
            cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_BATCH_TOKENS']=rows
            candidate,actual_plan=prefill_arguments(cfg,100000,24<<30)
            self.assertEqual(plan,actual_plan)
            self.assertEqual(candidate['--compute-tile'],rows)
            self.assertEqual(candidate['--yoco-keep'],'2541')
            self.assertEqual(int(candidate['--backend-temporary-reserve-bytes']),reserve<<20)
            for key in ['--job-device-bytes','--job-host-bytes','--cache-bytes']:
                self.assertEqual(candidate[key],base[key])
            # A larger legacy expert pool borrows exactly the same graph bytes.
            self.assertEqual(sum(int(candidate[k]) for k in ['--graph-bytes','--expert-workspace-bytes','--provider-device-bytes']),
                             sum(int(base[k]) for k in ['--graph-bytes','--expert-workspace-bytes','--provider-device-bytes']))
            self.assertGreater(int(candidate['--graph-bytes']),reserve<<20)

    def test_unbounded_or_malformed_chunk_is_rejected_before_engine_creation(self):
        for value in ['0','49152','16385','8192.0','bad']:
            cfg=self.config('1');cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_BATCH_TOKENS']=value
            with self.assertRaisesRegex(ValueError,'expert batch'):
                prefill_arguments(cfg,100000,24<<30)

    def test_small_context_preserves_the_cache_and_hidden_bank_allowances(self):
        cfg=self.config('1');cfg['context']=131072
        original,_=prefill_arguments(cfg,100000,24<<30)
        cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_BATCH_TOKENS']='8192'
        candidate,_=prefill_arguments(cfg,100000,24<<30)
        for key in ['--cache-bytes','--hidden-device-bytes']:
            self.assertEqual(candidate[key],original[key])
        self.assertEqual(candidate['--job-device-bytes'],original['--job-device-bytes'])

if __name__=='__main__':unittest.main()
