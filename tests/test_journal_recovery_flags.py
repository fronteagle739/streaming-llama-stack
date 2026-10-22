import json
import os
from pathlib import Path
import tempfile
import unittest
from slstack.journal import Journal
from slstack.config import ROOT


class JournalRecoveryFlagTests(unittest.TestCase):
    def test_dead_attachment_and_historical_detach_are_not_applied(self):
        # Preserve this tiny fixture under the task runtime; no cleanup deletion.
        folder = Path(tempfile.mkdtemp(prefix='journal-recovery-', dir=ROOT.parent / 'runtime'))
        base = folder / 'base'; base.write_bytes(b'unchanged small TEST base')
        adapter = folder / 'adapter'; adapter.write_bytes(b'TEST adapter')
        journal = Journal(folder)
        first = journal.begin(base, [(adapter, 1)], 'attached')
        data = journal.read(); data['entries'][0].update(pid=2147483647, applied=True)
        journal.path.write_text(json.dumps(data))
        reports = journal.recover()
        self.assertEqual(reports[0]['status'], 'detached')
        self.assertFalse(journal.read()['entries'][0]['applied'])
        data = journal.read(); data['entries'][0]['applied'] = True
        journal.path.write_text(json.dumps(data))
        journal.recover()
        self.assertFalse(journal.read()['entries'][0]['applied'])
