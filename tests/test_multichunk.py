"""RAM admission, native argv reuse and causal/KV equivalence of grouped passes."""
from collections import deque
from unittest import TestCase, main
from unittest.mock import patch

from slstack.combo_producer import prefill_plan, prefill_arguments, carry_bank_bytes, engram_prompt_bytes
from slstack.multichunk import selection

GIB = 1 << 30
CFG = dict(context=1000001, serving='single_engine', producer_host_budget_gib=6.25,
           producer_host_max_gib=24, host_budget_gib=25, producer_prefill_sweep_tokens=49152,
           producer_prefill_fold_cap_tokens=61440)


class Multichunk(TestCase):
    def test_one_is_the_existing_plan(self):
        for n in (1, 49152, 100000, 200000, 400000):
            self.assertEqual(prefill_plan(CFG, n, 30*GIB),
                             prefill_plan(dict(CFG, producer_chunks_per_pass=1), n, 30*GIB))

    def test_exact_capacity_and_host_charges(self):
        for chunks in (2, 3, 4):
            for n in (1, 49152, 100000, 200000, 400000):
                p = prefill_plan(dict(CFG, producer_chunks_per_pass=chunks), n, 30*GIB)
                self.assertEqual(sum(p['sweeps']), n)
                self.assertTrue(all(0 < s <= p['capacity'] for s in p['sweeps']))
                self.assertEqual(p['sweep_tokens'], chunks*49152)
                self.assertEqual(p['fold_cap'], chunks*49152+12288)
                self.assertEqual(p['carry_delta'], carry_bank_bytes(p['capacity'])-carry_bank_bytes(49152))
                self.assertEqual(p['engram_delta'], engram_prompt_bytes(p['capacity'])-engram_prompt_bytes(49152))
                self.assertEqual(p['host_bytes'] % 65536, 0)
                self.assertFalse(p['device_capacity_changed'])

    def test_auto_respects_physical_ram_outer_budget_and_page_rounding(self):
        for available in (7, 8, 12, 16, 24, 30):
            cfg = dict(CFG, producer_chunks_per_pass='auto')
            p = prefill_plan(cfg, 400000, available*GIB)
            self.assertLessEqual(p['host_bytes'], (available-.5)*GIB)
            self.assertLessEqual(p['host_bytes'], (cfg['host_budget_gib']-.25)*GIB)
            self.assertLessEqual(p['host_bytes'], cfg['producer_host_max_gib']*GIB)
        p = prefill_plan(dict(CFG, host_budget_gib=8, producer_chunks_per_pass='auto'), 400000, 30*GIB)
        self.assertEqual(p['chunks_per_pass'], 1)
        p = prefill_plan(dict(CFG, producer_chunks_per_pass='auto', runtime_min_free_gib=4), 400000, 16*GIB)
        self.assertEqual(p['chunks_per_pass'], 2)
        with self.assertRaises(MemoryError):
            prefill_plan(dict(CFG, producer_chunks_per_pass=4), 400000, 16*GIB)
        with self.assertRaises(MemoryError):
            prefill_plan(dict(CFG, producer_chunks_per_pass='auto'), 400000, 6*GIB)
        # Rounding must not add the last host page above the measured ceiling.
        top = prefill_plan(dict(CFG, producer_chunks_per_pass=2), 100000, 30*GIB)['host_bytes']
        with self.assertRaises(MemoryError):
            prefill_plan(dict(CFG, producer_chunks_per_pass=2), 100000, top+GIB//2-1)

    def test_bad_setting_and_other_serving_modes_fail(self):
        for value in (True, False, 0, 5, 2.0, '2', None, 'AUTO'):
            with self.assertRaises(ValueError): selection(value)
        with self.assertRaises(ValueError):
            prefill_plan(dict(CFG, producer_chunks_per_pass=2, serving='native'), 100000, 30*GIB)
        with self.assertRaises(ValueError):
            prefill_plan(dict(CFG, producer_chunks_per_pass=2, producer_prefill_sweep_tokens=65536), 100000, 30*GIB)

    def test_native_graph_and_working_buffers_stay_fixed(self):
        # The all-rows hidden bank (OPUS22 default) sizes VRAM by capacity; pin it off to compare buffers.
        fixed = dict(CFG, ctx_cache_precision='bf16', ctx_cache_growable=True, producer_hidden_all_rows=False)
        a, _ = prefill_arguments(fixed, 200000, 30*GIB)
        b, p = prefill_arguments(dict(fixed, producer_chunks_per_pass=2), 200000, 30*GIB)
        changed = {'--prefill-chunk-tokens','--job-host-bytes','--carry-host-bytes','--provider-host-bytes',
                   '--engram-prefetch-bytes','--prefill-sweep-tokens','--prefill-fold-cap-tokens'}
        self.assertEqual({k:v for k,v in a.items() if k not in changed}, {k:v for k,v in b.items() if k not in changed})
        self.assertEqual(int(b['--prefill-sweep-tokens']), 98304)
        self.assertEqual(p['sweeps'], [98304, 101696])

    def test_config_survives_single_runtime(self):
        from slstack.single_runtime import serving_config
        row = serving_config(dict(CFG, producer_chunks_per_pass=3), runtime_root='r',studio_root='s',
                             llama_producer='p',library_directories=[],revision='test')
        self.assertEqual(row['producer_chunks_per_pass'], 3)


def fixture(tokens, passes, *, replay=0):
    """Independent scalar transformer with four residual streams, raw KV rings,
    ratio-two pooling across odd boundaries and shared compressed KV.
    Integer math makes an ordering or KV overwrite error bit-observable.
    """
    layers=5; rings=[deque() for _ in range(layers)]; sums=[0]*layers
    pooled=[]; odd=None; hidden={}; visits=[]; offset=0
    for size in passes:
        values=[tuple(t+i for i in range(4)) for t in tokens[offset:offset+size]]
        final=offset+size==len(tokens)
        for layer in range(layers if final or not replay else 3):
            first=max(0,size-replay) if replay and layer>2 else 0
            if replay and layer>2:rings[layer].clear();sums[layer]=0
            for start in range(first,size,8):
                for local in range(start,min(start+8,size)):
                    pos=offset+local; row=values[local]; key=sum(row)
                    if layer==2:
                        if odd is None:odd=key
                        else:pooled.append(odd+key);odd=None
                    ring=rings[layer];ring.append(key);sums[layer]+=key
                    if len(ring)>7:sums[layer]-=ring.popleft()
                    visible=(pos+1)//2
                    selected=pooled[(pos*3)%visible] if visible and layer>=2 else 0
                    values[local]=tuple((x+sums[layer]+selected+layer)%104729 for x in row)
                    visits.append((layer,pos))
        for local in range(size):hidden[offset+local]=values[local]
        offset+=size
    return hidden, rings, pooled, odd, visits


class CausalFixtures(TestCase):
    def plans(self,n,chunks):
        # Scale only the chunk unit, retaining the production grouping and
        # admission code. Native geometry is covered by argv and C++ tests.
        if chunks==1:
            from slstack.combo_producer import fold_sweeps
            return fold_sweeps(n,32,40,40)[0]
        with patch('slstack.combo_producer.LEGACY_SWEEP',32):
            cfg=dict(CFG,producer_prefill_sweep_tokens=32,producer_prefill_fold_cap_tokens=40,
                     producer_chunks_per_pass=chunks)
            from slstack.multichunk import plan
            return plan(cfg,n,30*GIB)['sweeps']

    def test_exact_hidden_kv_pool_and_causal_order_for_one_to_four(self):
        tokens=[(i*17+3)%83 for i in range(139)]
        reference=fixture(tokens,self.plans(len(tokens),1))
        for chunks in (2,3,4):
            actual=fixture(tokens,self.plans(len(tokens),chunks))
            self.assertEqual(actual[:4], reference[:4])
            for layer in range(5):
                self.assertEqual([p for l,p in actual[4] if l==layer],list(range(len(tokens))))
        altered=tokens[:-1]+[tokens[-1]+1]
        for chunks in (1,2,3,4):
            actual=fixture(altered,self.plans(len(tokens),chunks))
            self.assertEqual([actual[0][i] for i in range(len(tokens)-1)],
                             [reference[0][i] for i in range(len(tokens)-1)])

    def test_yoco_final_is_final_logical_pass_not_each_chunk(self):
        tokens=[(i*13+1)%79 for i in range(127)]
        # The toy's exact cone: a final seven-row ring over two replay layers
        # needs 7 + 2*6 = 19 inputs. Group boundaries cannot choose a new tail.
        reference=fixture(tokens,[len(tokens)])
        for chunks in (1,2,3,4):
            passes=self.plans(len(tokens),chunks)
            actual=fixture(tokens,passes,replay=19)
            self.assertEqual(actual[0][len(tokens)-1],reference[0][len(tokens)-1])
            self.assertEqual(list(actual[1][-1]),list(reference[1][-1]))
            self.assertEqual(actual[2:4],reference[2:4])
            for layer in (3,4):
                self.assertEqual([p for l,p in actual[4] if l==layer],list(range(len(tokens)-19,len(tokens))))


if __name__=='__main__':main()
