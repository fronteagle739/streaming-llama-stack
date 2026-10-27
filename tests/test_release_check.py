from pathlib import Path
import json
import unittest
from unittest.mock import patch
import uuid
import hashlib
import struct
import zlib
import os
import subprocess
from slstack.config import ROOT
from scripts.release_check import scan, reviewed_png, scan_history, scan_index, scan_identities


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

    def test_builtin_identity_and_project_rules_do_not_need_local_inputs(self):
        root = self.fixture()
        person = 'Ed' + 'waardH' + 'olmes' + '-C' + 'ranborne'
        project = 'mc' + '4-integration'
        (root / 'LICENSE').write_text('Copyright Public Contributor\n' + person + '\n' + project)
        with patch('scripts.release_check.subprocess.check_output', return_value=b'LICENSE\0'):
            self.assertEqual(scan(root), [('LICENSE', 2, 'private identity'), ('LICENSE', 3, 'excluded project')])

    def test_paths_on_other_drives_and_example_scope(self):
        root = self.fixture()
        script = 'config = ' + repr('C' + ':/' + 'Users/sample/model.gguf')
        (root / 'code.py').write_text(script)
        (root / 'examples').mkdir()
        (root / 'examples/launch.ps1').write_text(script)
        (root / 'examples/README.md').write_text(script)
        paths = b'code.py\0examples/launch.ps1\0examples/README.md\0'
        with patch('scripts.release_check.subprocess.check_output', return_value=paths):
            self.assertEqual(scan(root), [('code.py', 1, 'machine path'), ('examples\\README.md' if os.name == 'nt' else 'examples/README.md', 1, 'machine path')])

    def test_tracked_build_directories_are_not_exempt(self):
        root = self.fixture()
        (root / 'build').mkdir()
        (root / 'build/leak.py').write_text('TEST_AUDIT_LITERAL')
        with patch('scripts.release_check.subprocess.check_output', return_value=b'build/leak.py\0'):
            self.assertEqual(scan(root)[0][2], 'forbidden audit literal')

    def history_fixture(self):
        root = self.fixture()
        subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
        def git(*args, author='Front Eagle'):
            env = dict(os.environ, GIT_AUTHOR_NAME=author, GIT_COMMITTER_NAME=author,
                       GIT_AUTHOR_EMAIL='fronteagle739@users.noreply.github.com',
                       GIT_COMMITTER_EMAIL='fronteagle739@users.noreply.github.com')
            return subprocess.check_output(['git', '-C', str(root), '-c', 'core.hooksPath=.no-hooks', *args], env=env)
        return root, git

    def test_history_reads_old_blobs_even_after_working_tree_is_clean(self):
        root, git = self.history_fixture()
        git('add', 'code.py');git('commit', '--quiet', '-m', 'Initial snapshot')
        (root / 'code.py').write_text('value = 1\n')
        git('add', 'code.py');git('commit', '--quiet', '-m', 'Clean latest tree')
        findings = scan_history(root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0][2], 'forbidden audit literal')
        self.assertIn(':code.py', findings[0][0])

    def test_index_gate_reads_staged_bytes_when_working_copy_has_been_cleaned(self):
        root, git = self.history_fixture()
        git('add', 'code.py')
        (root / 'code.py').write_text('value = 1\n')
        self.assertEqual(scan_index(root), [('code.py', 2, 'forbidden audit literal')])

    def test_builtin_identity_rule_also_blocks_standalone_aliases(self):
        root = self.fixture()
        names = ['Cran' + 'borne', 'ehol' + 'mes', 'edst' + 'udio']
        (root / 'code.py').write_text('\n'.join(names))
        with patch('scripts.release_check.subprocess.check_output', return_value=b'code.py\0'):
            self.assertEqual([row[1] for row in scan(root)], [1, 2, 3])

    def test_identity_gate_checks_environment_overrides(self):
        root, git = self.history_fixture()
        git('config', 'user.name', 'Front Eagle')
        git('config', 'user.email', 'fronteagle739@users.noreply.github.com')
        with patch.dict(os.environ, {'GIT_AUTHOR_NAME': 'Other Contributor',
                                    'GIT_COMMITTER_NAME': 'Front Eagle',
                                    'GIT_AUTHOR_EMAIL': 'fronteagle739@users.noreply.github.com',
                                    'GIT_COMMITTER_EMAIL': 'fronteagle739@users.noreply.github.com'}):
            self.assertEqual(scan_identities(root), [('GIT_AUTHOR_IDENT', 0, 'commit requires Front Eagle identity')])

    def test_history_requires_both_front_eagle_identities_and_ignores_other_refs(self):
        root, git = self.history_fixture()
        (root / 'code.py').write_text('value = 1\n')
        git('add', 'code.py');git('commit', '--quiet', '-m', 'Clean release')
        release = git('rev-parse', 'HEAD').decode().strip()
        (root / 'code.py').write_text('value = 2\n')
        git('add', 'code.py');git('commit', '--quiet', '-m', 'Development', author='Private Author')
        self.assertEqual(scan_history(root, release), [])
        findings = scan_history(root)
        self.assertEqual(findings, [(git('rev-parse', '--short=12', 'HEAD').decode().strip(), 0, 'history requires Front Eagle author and committer')])


if __name__ == '__main__':
    unittest.main()
