"""LLAMADECODE single sweep: capacity/budget plan, negotiated prefill maximum, near-equal split."""
import unittest
from slstack.native_broker import vendor
from slstack.combo_producer import prefill_plan, fold_sweeps, carry_bank_bytes, engram_prompt_bytes

vendor()
from studio.dsv41_session_contract import command, negotiate_prefill, split_prefill, MODEL_SHA256, Refused

CFG = dict(producer_host_budget_gib=6.25, context=262144)
SS = dict(CFG, producer_host_max_gib=12.0)


class PrefillPlan(unittest.TestCase):
    def test_compact_carry_record_shrinks_the_bank(self):
        # OPUS22: 51216 B records in 1642496 B pages, against today's 92176 B / 2953216 B.
        self.assertEqual(carry_bank_bytes(100000, True), 4096 + 3125 * 1642496)
        legacy = prefill_plan(dict(SS, producer_host_max_gib=7.75), 100000)
        compact = prefill_plan(dict(SS, producer_host_max_gib=7.75, producer_carry_compact_record=True), 100000)
        self.assertFalse(legacy['single_sweep'])
        self.assertTrue(compact['single_sweep'])
        self.assertEqual(compact['carry_delta'], carry_bank_bytes(100000, True) - carry_bank_bytes(49152))
        with self.assertRaises(ValueError):
            prefill_plan(dict(SS, producer_carry_compact_record=1), 100000)

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

    def test_desktop_reserve_keeps_512mib_free(self):
        # Owner: 512MiB admission reserve; the separate runtime floor stays4GiB.
        avail = 10 * 2**30
        p = prefill_plan(SS, 93668, avail)
        self.assertLessEqual(p['host_bytes'], avail - .5 * 2**30);self.assertEqual(p['capped_by'], 'available_ram')
        self.assertFalse(p['single_sweep'])
        f = prefill_plan(dict(FOLD, producer_host_max_gib=7.75), 103189, avail)
        self.assertLessEqual(f['host_bytes'], avail - .5 * 2**30)
        self.assertEqual(prefill_plan(dict(SS, producer_desktop_reserve_gib=12), 93668, avail)['host_bytes'], int(6.25 * 2**30))   # never below base
        self.assertEqual(prefill_plan(dict(SS, producer_desktop_reserve_gib=2), 93668, 32 * 2**30)['capped_by'], 'producer_host_max_gib')

    def test_measured_available_caps_and_pages(self):
        avail = 17324515328   # ss-91k launch
        SS = dict(globals()['SS'], producer_desktop_reserve_gib=1)   # the pre-desktop-reserve 1 GiB headroom
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


FOLD = dict(CFG, producer_host_max_gib=13.0, producer_prefill_sweep_tokens=49152)


class Fold(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(fold_sweeps(110000, 49152, 1 << 20), ([49152, 60848], False, False))
        self.assertEqual(fold_sweeps(93668, 65536, 1 << 20), ([93668], False, False))
        self.assertEqual(fold_sweeps(30000, 49152, 49152), ([30000], False, False))
        self.assertEqual(fold_sweeps(49152, 49152, 49152), ([49152], False, False))
        self.assertEqual(fold_sweeps(49153, 49152, 1 << 20), ([49153], False, False))
        self.assertEqual(fold_sweeps(98304, 49152, 49152), ([49152, 49152], False, False))
        self.assertEqual(fold_sweeps(230000, 49152, 81000), ([49152] * 3 + [45056, 37488], False, True))

    def test_rebalance_examples(self):
        self.assertEqual(fold_sweeps(103189, 49152, 53000, 61440), ([49152, 28672, 25365], False, True))
        for cap in (54037, 70000, 1 << 20):
            self.assertEqual(fold_sweeps(103189, 49152, cap, 61440), ([49152, 54037], False, False))
        self.assertEqual(fold_sweeps(93668, 49152, 53000, 61440), ([49152, 44516], True, False))
        self.assertEqual(fold_sweeps(150000, 49152, 50000, 61440), ([49152, 49152, 28672, 23024], False, True))

    def test_rebalance_invariants(self):
        for n in range(1, 400000, 977):
            for s, f in ((49152, None), (49152, 61440), (65536, None)):
                for cap in range(s, 2 * s + 4097, 1517):
                    sweeps, tail, rebalance = fold_sweeps(n, s, cap, f)
                    self.assertEqual(sum(sweeps), n)
                    self.assertTrue(all(x <= cap for x in sweeps), (n, s, f, cap, sweeps))
                    starts = [sum(sweeps[:i]) for i in range(len(sweeps))]
                    self.assertTrue(all(x % 4096 == 0 for x in starts))
                    self.assertTrue(len(sweeps) == 1 or min(sweeps) >= 4096, (n, s, f, cap, sweeps))
                    self.assertFalse(tail and rebalance)

    def test_plan_examples(self):
        p = prefill_plan(FOLD, 109000)
        self.assertEqual((p['policy'], p['sweeps'], p['capacity'], p['fold_tail']), ('fold', [49152, 59848], 59848, False))
        self.assertEqual(p['carry_delta'], carry_bank_bytes(59848) - carry_bank_bytes(49152))
        self.assertEqual(p['engram_delta'], (59848 - 49152) * 12864)
        self.assertEqual(p['host_bytes'], -(-(int(6.25 * 2**30) + p['carry_delta'] + p['engram_delta']) // 65536) * 65536)
        p = prefill_plan(FOLD, 60000)
        self.assertEqual((p['sweeps'], p['capacity'], p['single_sweep']), ([60000], 60000, True))
        p = prefill_plan(FOLD, 110000)
        self.assertEqual((p['sweeps'], p['capacity']), ([49152, 60848], 60848))
        p = prefill_plan(FOLD, 16000)
        self.assertEqual((p['sweeps'], p['capacity'], p['host_bytes']), ([16000], 49152, int(6.25 * 2**30)))

    def test_fold_over_ram_cap_rebalances(self):
        p = prefill_plan(dict(FOLD, producer_host_max_gib=7.0), 110000)   # cap below 60,848 rows
        self.assertTrue(p['rebalance'] and not p['fold_tail'])
        self.assertEqual((p['sweeps'], p['capacity']), ([49152, 32768, 28080], 49152))
        self.assertIn('rebalanced into 32768+28080', p['reason'])

    def test_fold_cap(self):
        expect = {103189: [49152, 54037], 93668: [49152, 44516], 60000: [60000], 70000: [49152, 20848],
                  110000: [49152, 60848], 120000: [49152, 49152, 21696]}
        for n, sweeps in expect.items():
            p = prefill_plan(FOLD, n)
            self.assertEqual((p['sweeps'], p['capacity'], p['fold_cap']), (sweeps, max(49152, max(sweeps)), 61440), n)
        p = prefill_plan(FOLD, 93668)
        self.assertTrue(p['fold_tail']);self.assertIn('the fold cap 61440', p['reason'])
        self.assertEqual(prefill_plan(dict(FOLD, producer_prefill_fold_cap_tokens=100000), 93668)['sweeps'], [93668])
        with self.assertRaises(ValueError):prefill_plan(dict(FOLD, producer_prefill_fold_cap_tokens=0), 93668)

    def test_bad_sweep_refused(self):
        for bad in (0, 49000, -4096, 1 << 21, 49152.0):
            with self.assertRaises(ValueError):prefill_plan(dict(FOLD, producer_prefill_sweep_tokens=bad), 100000)
        with self.assertRaises(ValueError):prefill_plan(dict(FOLD, producer_host_max_gib=6.25, producer_prefill_sweep_tokens=65536), 100000)

    def test_absent_is_today(self):
        for n in (1, 14859, 49152, 60000, 103319, 109000, 224588):
            for cfg in (CFG, SS):
                self.assertNotIn('policy', prefill_plan(cfg, n))


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
