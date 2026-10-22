from pathlib import Path
import json
import unittest
from unittest.mock import patch
import uuid
import hashlib
import struct
import zlib
from slstack.config import ROOT
from scripts.release_check import scan, reviewed_png


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

    def test_private_names_are_local_inputs_and_paths_are_scanned(self):
        root = self.fixture()
        (root / 'release-secrets.local.json').write_text(json.dumps({'forbidden_names': ['TEST_PRIVATE_NAME']}))
        (root / 'TEST_PRIVATE_NAME.py').write_text('value = "TEST_PRIVATE_NAME"\n')
        with patch('scripts.release_check.subprocess.check_output', return_value=b'TEST_PRIVATE_NAME.py\0'):
            found = scan(root)
        self.assertEqual({row[2] for row in found}, {'private identity in path', 'private identity'})

    def image(self, root, metadata=False):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        raw = bytes.fromhex('89504e470d0a1a0a')
        raw += chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
        if metadata:
            raw += chunk(b'tEXt', b'Comment\0unreviewed text')
        raw += chunk(b'IDAT', zlib.compress(b'\0\0\0\0')) + chunk(b'IEND', b'')
        path = root / 'examples' / 'proof.png'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_only_exact_reviewed_pixel_png_is_accepted(self):
        root = self.fixture()
        path, digest = self.image(root)
        with patch('scripts.release_check.subprocess.check_output', return_value=b'examples/proof.png\0'):
            self.assertEqual(scan(root)[0][2], 'unreviewed non-text artifact')
            (root / 'release-artifacts.json').write_text(json.dumps({'artifacts': [{
                'path': 'examples/proof.png', 'sha256': digest, 'media_type': 'image/png',
                'review': 'Visually reviewed synthetic pixel fixture.'}]}))
            self.assertEqual(scan(root), [])
            path.write_bytes(path.read_bytes() + b'unreviewed')
            self.assertEqual(scan(root)[0][2], 'unreviewed non-text artifact')

    def test_metadata_is_refused_even_with_matching_hash(self):
        root = self.fixture()
        path, digest = self.image(root, metadata=True)
        self.assertFalse(reviewed_png(path, digest))

    def test_unix_route_is_not_a_machine_path_but_drive_and_home_are(self):
        root = self.fixture()
        (root / 'code.py').write_text("route = '/users/:id/items'\na = '" + "D" + ":/private/file'\nb = '/" + "home/sample/file'\n")
        with patch('scripts.release_check.subprocess.check_output', return_value=b'code.py\0'):
            self.assertEqual(scan(root), [('code.py', 2, 'machine path'), ('code.py', 3, 'machine path')])


if __name__ == '__main__':
    unittest.main()
