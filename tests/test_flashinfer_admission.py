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
            self.assertLess(rounded+module_headroom+other_temporaries,reserve)

    def test_incompatible_arithmetic_profile_is_rejected_before_engine_creation(self):
        cfg=self.config('1');cfg['producer_profile']='accurate'
        with self.assertRaisesRegex(ValueError,'paired ULTRA'):
            prefill_arguments(cfg,2048,24<<30)

if __name__=='__main__':unittest.main()
