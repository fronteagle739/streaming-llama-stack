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


if __name__ == '__main__':
    unittest.main()
