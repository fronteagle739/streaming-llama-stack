import unittest
from slstack.bank_gate import check, BankGateFailed


def sweep(tokens, acquire=0.0, banks=True, layers=40):
    out = (b'LLAMADECODE_PREFILL_LAYOUT banks=2 streaming_banks=2 persistent_stages=0 expert_pool_bound=0\n' if banks else b'')
    out += b'[dsv41-sweep] layer=0 begin tokens=%d\n' % tokens
    for n in range(layers):out += b'DS4PERF_LAYER_WALL_LEDGER layer=%d wall_ms=900.0 acquire_ms=%.1f executor_wall_ms=850.0\n' % (n, acquire)
    return out

DECODE = b'LLAMADECODE_DECODE_LAYOUT stages=82\nLLAMADECODE_LAYOUT_SWITCH decode=1 switch_ms=700 same_admission=1\n'
BACK = b'LLAMADECODE_LAYOUT_SWITCH decode=0 switch_ms=800 same_admission=1\n'


class BankGateTest(unittest.TestCase):
    def test_decode_then_large_prefill_with_both_banks_passes(self):
        log = sweep(49152) + DECODE + BACK + sweep(40354)
        r = check(log);self.assertEqual([x['banks'] for x in r], [(2, 2), (2, 2)])

    def test_small_to_large_transition_passes_when_large_waits_nothing(self):
        check(sweep(5015, acquire=350.0) + DECODE + BACK + sweep(43000))

    def test_prefill_after_decode_without_the_bank_layout_fails_loudly(self):
        with self.assertRaises(BankGateFailed):check(sweep(49152) + DECODE + BACK + sweep(40354, banks=False))

    def test_large_prefill_exposing_ds_wait_fails(self):
        with self.assertRaises(BankGateFailed):check(DECODE + BACK + sweep(43000, acquire=300.0))


if __name__ == '__main__':unittest.main()
