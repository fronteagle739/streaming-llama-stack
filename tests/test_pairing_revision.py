"""Pairing build identity: the checkout's CMake build, or an installed producer's recorded revision."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack.pairing import producer_revision

HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
INSTALLED = str(ROOT.parent / 'installed-runtime' / 'llama-dsv41-prefill.exe')


class ProducerRevision(unittest.TestCase):
    def test_installed_producer_uses_recorded_revision(self):
        self.assertEqual(producer_revision({'llama_producer': INSTALLED, 'producer_source_revision': HEAD}), HEAD)

    def test_installed_producer_refuses_changed_source(self):
        first = subprocess.check_output(['git', 'rev-list', '--max-parents=0', 'HEAD'], cwd=ROOT, text=True).split()[0]
        with self.assertRaisesRegex(ValueError, 'source changed'):
            producer_revision({'llama_producer': INSTALLED, 'producer_source_revision': first})

    @unittest.skipIf((ROOT / 'build/llama/common/build-info.cpp').is_file(), 'checkout has a CMake build')
    def test_checkout_build_needs_its_identity(self):
        with self.assertRaisesRegex(ValueError, 'build identity missing'):
            producer_revision({'producer_source_revision': HEAD})


if __name__ == '__main__':
    unittest.main()
