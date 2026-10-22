import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from slstack.config import ROOT
from slstack.journal import sha256
from slstack.mac import setup, source_digest


class MacBuildIntegrityTests(unittest.TestCase):
    def fixture(self):
        root = ROOT / 'runtime' / 'fixtures' / ('mac-build-' + uuid.uuid4().hex)
        directory = root / 'engines' / 'ds4-mac'
        directory.mkdir(parents=True)
        (directory / 'ds4_server.c').write_text('/* TEST source */\n')
        (directory / 'ds4-server').write_bytes(b'TEST old binary')
        (root / 'model.gguf').write_bytes(b'TEST path only')
        runtime = root / 'runtime'
        (runtime / 'state').mkdir(parents=True)
        cfg = {'runtime_root': str(runtime), 'model': str(root / 'model.gguf')}
        record = {'source_sha256': source_digest(directory),
                  'binary_sha256': sha256(directory / 'ds4-server')}
        (runtime / 'state' / 'ds4-build.json').write_text(json.dumps(record))
        return root, directory, cfg

    def test_check_refuses_existing_binary_after_source_content_changes(self):
        root, directory, cfg = self.fixture()
        with patch('slstack.mac.ROOT', root), patch('slstack.mac.sys.platform', 'darwin'), \
             patch('slstack.mac.platform.machine', return_value='arm64'), \
             patch('slstack.mac.subprocess.check_output', return_value='512000\n'), \
             patch('slstack.mac.OwnedProcess') as process:
            self.assertTrue(setup(cfg, check=True)['ready'])
            (directory / 'ds4_server.c').write_text('/* TEST changed source */\n')
            self.assertFalse(setup(cfg, check=True)['ready'])
            process.assert_not_called()

    def test_source_change_during_build_is_refused_before_recording_success(self):
        root, directory, cfg = self.fixture()
        (directory / 'ds4_server.c').write_text('/* TEST new source */\n')
        before = (Path(cfg['runtime_root']) / 'state' / 'ds4-build.json').read_bytes()
        def change_source():
            (directory / 'ds4_server.c').write_text('/* TEST concurrent edit */\n')
            return {'exit_code': 0}
        with patch('slstack.mac.ROOT', root), patch('slstack.mac.sys.platform', 'darwin'), \
             patch('slstack.mac.platform.machine', return_value='arm64'), \
             patch('slstack.mac.subprocess.check_output', return_value='512000\n'), \
             patch('slstack.mac.shutil.which', return_value='TEST tool'), \
             patch('slstack.mac.OwnedProcess') as process:
            process.return_value.wait.side_effect = change_source
            with self.assertRaisesRegex(ValueError, 'source changed'):
                setup(cfg)
        self.assertEqual((Path(cfg['runtime_root']) / 'state' / 'ds4-build.json').read_bytes(), before)
