import unittest
from slstack.prefill_progress import PrefillProgress, chunk


def sweep_lines(tokens, layers=40, upto=None):
    out = []
    for n in range(layers if upto is None else upto):
        out.append(b'[dsv41-sweep] layer=%d begin tokens=%d\nDS4PERF_X layer=%d\n[dsv41-sweep] layer=%d complete\n' % (n, tokens, n, n))
    return b''.join(out)


class PrefillProgressTest(unittest.TestCase):
    def test_single_sweep_percent_and_eta(self):
        p = PrefillProgress(40354, 49152, started=0.0)
        p.feed(sweep_lines(40354, upto=20))
        s = p.snapshot(27.0)
        self.assertEqual((s['sweep'], s['sweeps'], s['layer'], s['layers']), (1, 1, 20, 40))
        self.assertEqual(s['percent'], 50.0);self.assertEqual(s['eta_s'], 27.0)

    def test_multi_sweep_is_token_weighted_and_monotonic(self):
        p = PrefillProgress(103319, 49152);seen = []
        data = sweep_lines(49152) + sweep_lines(49152) + sweep_lines(5015)
        for i in range(0, len(data), 997):   # arbitrary splits across lines
            p.feed(data[i:i + 997]);seen.append(p.snapshot(1.0)['percent'])
        self.assertEqual(seen, sorted(seen));self.assertEqual(seen[-1], 100.0)
        self.assertEqual(p.snapshot(1.0)['sweeps'], 3)

    def test_262k_prompt_reports_from_the_first_layer(self):
        p = PrefillProgress(262144, 49152);p.feed(sweep_lines(49152, upto=1))
        s = p.snapshot(10.0)
        self.assertGreater(s['percent'], 0);self.assertEqual(s['sweeps'], 6);self.assertIsNotNone(s['eta_s'])

    def test_no_lines_yet(self):
        s = PrefillProgress(1000, 49152).snapshot(5.0)
        self.assertEqual(s['percent'], 0.0);self.assertIsNone(s['eta_s'])

    def test_chunk_is_skippable_by_openai_clients(self):
        c = chunk('id', 'm', 1, {'percent': 1.0})
        self.assertEqual(c['choices'], []);self.assertEqual(c['object'], 'chat.completion.chunk')


if __name__ == '__main__':unittest.main()
