from pathlib import Path
import json
import unittest
from unittest.mock import patch
import uuid
from slstack.config import ROOT
from scripts.release_check import scan


class ReleaseCheckTests(unittest.TestCase):
    def fixture(self):
        root = ROOT / 'runtime' / 'fixtures' / ('release-check-' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        (root / 'release-secrets.local.json').write_text(json.dumps({'forbidden_literals': ['TEST_AUDIT_LITERAL']}))
        (root / 'code.py').write_text('first line\nvalue = "TEST_AUDIT_LITERAL"\n')
        return root

    def test_exact_audit_literal_is_detected_without_a_password_label(self):
        root = self.fixture()
        with patch('scripts.release_check.subprocess.check_output', return_value=b'code.py\0'):
            found = scan(root)
        self.assertEqual(found, [('code.py', 2, 'forbidden audit literal')])

    def test_local_configuration_cannot_be_tracked(self):
        root = self.fixture()
        with patch('scripts.release_check.subprocess.check_output', return_value=b'release-secrets.local.json\0'):
            found = scan(root)
        self.assertEqual(found, [('release-secrets.local.json', 0, 'local configuration must be untracked')])


if __name__ == '__main__':
    unittest.main()
