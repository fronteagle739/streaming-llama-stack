"""LLAMADECODE single sweep: capacity/budget plan, negotiated prefill maximum, near-equal split."""
import unittest
from slstack.native_broker import vendor
from slstack.combo_producer import prefill_plan, hide_passes, hidden_bank_rows, carry_bank_bytes, engram_prompt_bytes

vendor()
from studio.dsv41_session_contract import command, negotiate_prefill, split_prefill, MODEL_SHA256, Refused

CFG = dict(producer_host_budget_gib=6.25, context=262144)
SS = dict(CFG, producer_host_max_gib=12.0)


class PrefillPlan(unittest.TestCase):
    def test_no_raise_without_cap_keeps_today(self):
        for n in (14859, 43327, 49152, 103319):
            p = prefill_plan(CFG, n)
            self.assertEqual((p['capacity'], p['host_bytes'], p['carry_delta'], p['engram_delta']), (49152, int(6.25 * 2**30), 0, 0))

    def test_small_prompts_unchanged_with_cap(self):
        for n in (1, 14859, 43327, 49152):
            p = prefill_plan(SS, n)
            self.assertEqual((p['capacity'], p['host_bytes']), (49152, int(6.25 * 2**30)))
            self.assertTrue(p['single_sweep'])

    def test_103k_is_one_sweep_and_budget_is_exact(self):
        p = prefill_plan(SS, 103319)
        self.assertEqual(p['capacity'], 103319)
        self.assertTrue(p['single_sweep'])
        self.assertEqual(p['carry_delta'], carry_bank_bytes(103319) - carry_bank_bytes(49152))
        self.assertEqual(p['engram_delta'], (103319 - 49152) * 12864)
        self.assertEqual(p['host_bytes'], -(-(int(6.25 * 2**30) + p['carry_delta'] + p['engram_delta']) // 65536) * 65536)
        self.assertLessEqual(p['host_bytes'], 12 * 2**30)

    def test_cap_bounds_capacity(self):
        p = prefill_plan(SS, 224588)
        self.assertFalse(p['single_sweep'])
        self.assertEqual(p['capacity'], 77824)   # cap 107,840 -> near-equal chunk reserved
        self.assertIn('near-equal sweeps of 77824', p['reason'])
        self.assertLessEqual(p['host_bytes'], 12 * 2**30)
        self.assertEqual(prefill_plan(dict(SS, producer_host_max_gib=12.7), 224588)['capacity'], 114688)   # two sweeps
        self.assertEqual(prefill_plan(dict(SS, context=60000), 103319)['capacity'], 59999)

    def test_measured_available_caps_and_pages(self):
        avail = 17324515328   # ss-91k launch
        p = prefill_plan(SS, 93668, avail)
        self.assertTrue(p['single_sweep']);self.assertEqual(p['capacity'], 93668);self.assertEqual(p['capped_by'], 'producer_host_max_gib')   # 1 GiB reserve: 12 GiB max binds first
        self.assertEqual(p['host_bytes'] % 65536, 0)
        self.assertLessEqual(p['host_bytes'], avail - 1 * 2**30)
        q = prefill_plan(SS, 224588, avail)
        self.assertFalse(q['single_sweep']);self.assertEqual(q['capacity'], 77824);self.assertLessEqual(q['host_bytes'], avail - 1 * 2**30)
        r = prefill_plan(SS, 93668, 10 * 2**30)   # too little RAM for any raise: today's 49152 budget
        self.assertEqual((r['capacity'], r['host_bytes']), (49152, int(6.25 * 2**30)))
        for n in (16000, 43327):self.assertEqual(prefill_plan(SS, n, avail)['host_bytes'], int(6.25 * 2**30))

    def test_byte_formulas_match_producer(self):
        self.assertEqual(carry_bank_bytes(49152), 4536143872)
        self.assertEqual(engram_prompt_bytes(49152), 636485632)

    def test_cap_below_base_refused(self):
        with self.assertRaises(ValueError):prefill_plan(dict(CFG, producer_host_max_gib=6), 1000)


HIDE = dict(CFG, producer_host_max_gib=13.0, producer_prefill_hide_tokens=32768)
BIG = 1 << 20


class Hide(unittest.TestCase):
    def test_examples(self):
        h = lambda n, c=BIG: hide_passes(n, c, 32768, 49152)
        self.assertEqual(h(103189), ([32768, 32768, 37653], False, False))
        self.assertEqual(h(100000), ([32768, 32768, 34464], False, False))
        self.assertEqual(h(60176), ([60176], True, False))
        self.assertEqual(h(60176, 55000), ([28672, 31504], False, True))
        self.assertEqual(h(146770), ([49152, 49152, 48466], False, False))
        self.assertEqual(h(70000), ([32768, 37232], False, False))
        self.assertEqual(h(93668), ([45056, 48612], False, False))
        self.assertEqual(h(40000), ([40000], False, False))
        self.assertEqual(h(20000), ([20000], False, False))
        self.assertEqual(hide_passes(103189, BIG, 32768, 65536)[0], [49152, 54037])

    def test_bank_rows_from_hidden_bytes(self):
        self.assertEqual(hidden_bank_rows(2521649916), 49152)   # the profile's --hidden-device-bytes
        self.assertEqual(hidden_bank_rows(0), 49152)
        self.assertEqual(hidden_bank_rows(2 * 2521649916), 98304)

    def test_plan(self):
        p = prefill_plan(HIDE, 103189, None, 49152)
        self.assertEqual((p['policy'], p['sweeps'], p['capacity'], p['hide_tokens'], p['bank_rows']), ('hide', [32768, 32768, 37653], 49152, 32768, 49152))
        p = prefill_plan(HIDE, 60176, None, 49152)
        self.assertEqual((p['sweeps'], p['capacity'], p['folded']), ([60176], 60176, True))
        self.assertEqual(p['carry_delta'], carry_bank_bytes(60176) - carry_bank_bytes(49152))
        p = prefill_plan(dict(HIDE, producer_host_max_gib=6.25), 60176, None, 49152)   # no RAM raise: C = 49,152
        self.assertEqual((p['sweeps'], p['capacity'], p['fold_refused']), ([28672, 31504], 49152, True))
        self.assertIn('do not fit the capacity', p['reason'])

    def test_superseded_fold_keys_select_hide_and_are_ignored(self):
        p = prefill_plan(dict(CFG, producer_host_max_gib=13.0, producer_prefill_sweep_tokens=49152, producer_prefill_fold_cap_tokens=61440), 103189)
        self.assertEqual((p['policy'], p['hide_tokens'], p['sweeps']), ('hide', 32768, [32768, 32768, 37653]))
        self.assertEqual(p['ignored'], dict(producer_prefill_sweep_tokens=49152, producer_prefill_fold_cap_tokens=61440))

    def test_bad_hide_refused(self):
        for bad in (0, -1, 32768.0, 1 << 21):
            with self.assertRaises(ValueError):prefill_plan(dict(HIDE, producer_prefill_hide_tokens=bad), 100000)

    def test_absent_is_today(self):
        for n in (1, 14859, 49152, 60000, 103319, 109000, 224588):
            for cfg in (CFG, SS):
                self.assertNotIn('policy', prefill_plan(cfg, n))

    def test_invariants(self):
        for n in range(1, 400000, 977):
            for c in (49152, 53000, 55000, 60176, 65536, 80000, 120000, BIG):
                for h in (28672, 32768, 36000, 40960):
                    for bank in (32768, 49152, 65536):
                        passes, folded, refused = hide_passes(n, c, h, bank)
                        bp = min(bank, c); k0 = -(-n // bp); k = len(passes)
                        self.assertEqual(sum(passes), n)
                        starts = [sum(passes[:i]) for i in range(k)]
                        self.assertTrue(all(x % 4096 == 0 for x in starts))
                        self.assertTrue(all(x <= c and (folded or x <= bp) for x in passes))
                        self.assertTrue(all(x >= h or n < h * k for x in passes), (n, c, h, bank, passes))
                        self.assertTrue(len(set(passes[:-1])) <= 1)
                        self.assertTrue(k + 1 == k0 and n < h * k0 if folded else k >= k0)


class Negotiation(unittest.TestCase):
    def test_old_and_new_producers(self):
        self.assertEqual(negotiate_prefill(dict(maximum_prefill_tokens=49152)), 49152)        # old producer
        self.assertEqual(negotiate_prefill(dict(maximum_prefill_tokens=49152, maximum_prefill_sweep_tokens=103319)), 103319)
        self.assertEqual(negotiate_prefill(dict(maximum_prefill_tokens=49152, maximum_prefill_sweep_tokens=1 << 22)), 1 << 20)
        self.assertEqual(negotiate_prefill({}), 49152)

    def test_command_bound_is_the_negotiated_maximum(self):
        session = '1' * 32
        base = dict(context=262144, position=1000, mode=1, capture_end=0, capture_length=0, tokens_sha256='0' * 64,
                    logits_sha256='0' * 64, owner_sha256=['0' * 64] * 48)
        row = dict(session=session, model_sha256=MODEL_SHA256, sequence=1, operation='prefill', base=base, tokens=[1] * 60000)
        with self.assertRaises(Refused):command(row, session, 1, 1)                      # default: legacy 49152
        command(row, session, 1, 1, prefill_max=103319)


class Split(unittest.TestCase):
    def test_matches_producer_split(self):
        self.assertEqual(split_prefill(103319, 107840), 103319)
        self.assertEqual(split_prefill(103319, 49152), 36864)
        self.assertEqual(split_prefill(224588, 107840), 77824)
        self.assertEqual(split_prefill(224588, 114688), 114688)
        self.assertEqual(split_prefill(98305, 49152), 36864)
        self.assertEqual(split_prefill(40354, 49152), 40354)

    def test_invariants(self):
        for n in range(1, 400000, 977):
            for cap in (49152, 61440, 107840):
                c = split_prefill(n, cap);k = -(-n // c)
                self.assertTrue(c <= cap and (k == 1 or c % 4096 == 0))
                self.assertLess(c * k - n, 4096 * k if k > 1 else 1)


if __name__ == '__main__':
    unittest.main()
