"""Cross-session expert hotlist (CPU only, no engine)."""
import tempfile
import unittest
from pathlib import Path

from slstack.combo_native import global_hotlist, publish_hotlist

LIVE = b'# ds4-expert-cache-snapshot v1 model_size=1 source=live vram=2 host=0\nv 0 1 2 2\nv 1 3 2 2\n'
INHERITED = LIVE.replace(b'source=live', b'source=inherited')


def cfg(root, budget='24GB'):
    return dict(runtime_root=str(root), combo_identity=dict(model_sha256='2cc6bc254f22a0238020e61e931c0cac'),
                ds4_command=['x.exe', '--ssd-streaming-cache-experts', budget])


class Hotlist(unittest.TestCase):
    def test_key_is_model_and_budget(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = global_hotlist(cfg(t)), global_hotlist(cfg(t, '12GB'))
        self.assertNotEqual(a, b)
        self.assertEqual(a.name, 'expert-hotlist-2cc6bc254f22a023-24GB.txt')
        with tempfile.TemporaryDirectory() as t:
            c = cfg(t); c['ds4_command'] += ['--ram-streaming-cache-experts', '6GB']
            self.assertEqual(global_hotlist(c).name, 'expert-hotlist-2cc6bc254f22a023-24GB-ram6GB.txt')

    def test_only_live_lists_are_published(self):
        with tempfile.TemporaryDirectory() as t:
            s, g = Path(t) / 's.txt', Path(t) / 'g.txt'
            s.write_bytes(INHERITED)
            self.assertFalse(publish_hotlist(s, g)); self.assertFalse(g.exists())
            s.write_bytes(LIVE)
            self.assertTrue(publish_hotlist(s, g)); self.assertEqual(g.read_bytes(), LIVE)

    def test_missing_session_list_is_not_published(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(publish_hotlist(Path(t) / 'none.txt', Path(t) / 'g.txt'))
            self.assertFalse(publish_hotlist(None, Path(t) / 'g.txt'))


FREQ = ('# ds4-expert-frequency v1 model_size=9 vram_configured=2 host_configured=1 experts=4\n'
        'f 0 1 10 5 6\nf 0 2 30 5 6\nf 1 3 20 5 6\nf 1 4 1 5 6\n')


class Frequency(unittest.TestCase):
    def test_merge_decays_and_adds(self):
        from slstack.combo_native import merge_frequency, _read_frequency
        with tempfile.TemporaryDirectory() as t:
            s, g = Path(t) / 's.freq', Path(t) / 'g.freq'
            s.write_text(FREQ)
            self.assertTrue(merge_frequency(s, g)); self.assertTrue(merge_frequency(s, g))
            head, rows = _read_frequency(g)
        self.assertEqual(head, (9, 2, 1))
        self.assertAlmostEqual(rows[(0, 2)][0], 30 * 0.5 + 30)

    def test_seed_ranks_vram_then_host_least_frequent_first(self):
        from slstack.combo_native import build_frequency_seed
        with tempfile.TemporaryDirectory() as t:
            g, out = Path(t) / 'g.freq', Path(t) / 'seed.txt'
            g.write_text(FREQ)
            self.assertEqual(build_frequency_seed(g, out), (2, 1))
            lines = out.read_text().splitlines()
        self.assertEqual(lines[0], '# ds4-expert-cache-snapshot v1 model_size=9 source=frequency vram=2 host=1')
        self.assertEqual(lines[1:], ['h 0 1 5 6', 'v 1 3 5 6', 'v 0 2 5 6'])

    def test_no_global_no_seed(self):
        from slstack.combo_native import build_frequency_seed
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(build_frequency_seed(Path(t) / 'none', Path(t) / 'o'))


if __name__ == '__main__':
    unittest.main()
