import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
from slstack.config import ROOT
from slstack.journal import Journal, sha256


class JournalIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'runtime' / 'fixtures' / ('journal-integrity-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.base = self.root / 'TEST-base.bin'
        self.base.write_bytes(b'TEST base bytes')

    def test_intermediate_transition_reuses_verified_hash_and_detach_rehashes(self):
        journal = Journal(self.root)
        with patch('slstack.journal.sha256', wraps=sha256) as digest:
            row = journal.begin(self.base, [], 'attaching')
            self.assertEqual(digest.call_count, 1)
            journal.update(row['id'], 'attached')
            self.assertEqual(digest.call_count, 1)
            journal.update(row['id'], 'detached')
            self.assertEqual(digest.call_count, 2)

    def test_detach_rejects_changed_bytes_even_when_size_and_mtime_are_preserved(self):
        journal = Journal(self.root)
        row = journal.begin(self.base, [], 'attaching')
        journal.update(row['id'], 'attached')
        st = self.base.stat()
        self.base.write_bytes(b'TEST evil bytes')
        os.utime(self.base, ns=(st.st_atime_ns, st.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, 'BASE HASH MISMATCH'):
            journal.update(row['id'], 'detached')
        self.assertEqual(journal.read()['entries'][0]['status'], 'attached')


if __name__ == '__main__':
    unittest.main()
