"""LLAMADECODE single sweep: capacity/budget plan, negotiated prefill maximum, near-equal split."""
import unittest
from slstack.native_broker import vendor
from slstack.combo_producer import prefill_plan, carry_bank_bytes, engram_prompt_bytes

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
        self.assertTrue(p['single_sweep']);self.assertEqual(p['capacity'], 93668);self.assertEqual(p['capped_by'], 'available_ram')
        self.assertEqual(p['host_bytes'] % 65536, 0)
        self.assertLessEqual(p['host_bytes'], avail - 5 * 2**30)
        q = prefill_plan(SS, 224588, avail)
        self.assertFalse(q['single_sweep']);self.assertEqual(q['capacity'], 77824);self.assertLessEqual(q['host_bytes'], avail - 5 * 2**30)
        r = prefill_plan(SS, 93668, 10 * 2**30)   # too little RAM for any raise: today's 49152 budget
        self.assertEqual((r['capacity'], r['host_bytes']), (49152, int(6.25 * 2**30)))
        for n in (16000, 43327):self.assertEqual(prefill_plan(SS, n, avail)['host_bytes'], int(6.25 * 2**30))

    def test_byte_formulas_match_producer(self):
        self.assertEqual(carry_bank_bytes(49152), 4536143872)
        self.assertEqual(engram_prompt_bytes(49152), 636485632)

    def test_cap_below_base_refused(self):
        with self.assertRaises(ValueError):prefill_plan(dict(CFG, producer_host_max_gib=6), 1000)


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
