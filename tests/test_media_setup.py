import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from slstack.config import ROOT
from slstack.install import media_profile, python_path
from slstack import media_assets


class MediaSetupTests(unittest.TestCase):
    def test_marker_cannot_claim_ready_when_import_verification_fails(self):
        root=Path(tempfile.mkdtemp());venv=root/'media-venv'
        python_path(venv).parent.mkdir(parents=True);python_path(venv).write_bytes(b'fixture')
        digest=hashlib.sha256((ROOT/'locks/media-win-py312.txt').read_bytes()).hexdigest()
        (venv/'slstack-installed.json').write_text(json.dumps(dict(lock_sha256=digest)))
        job=Mock();job.log=root/'probe.log'
        job.log.write_text('SLSTACK_MEDIA_RUNTIME '+json.dumps(dict(ready=False,missing=[dict(module='required_pipeline')]))+'\n')
        job.wait.side_effect=RuntimeError('Import failed')
        cfg=dict(runtime_root=str(root),host_budget_gib=9.5)
        with patch('slstack.install.OwnedProcess',return_value=job):
            result=media_profile(cfg,'studio',check=True)
        self.assertFalse(result['ready']);job.stop.assert_called_once()

    def test_setup_keeps_the_working_general_pin_and_studios_qwen_pin_separate(self):
        general=(ROOT/'locks/media-win-py312.txt').read_text()
        qwen=(ROOT/'locks/media-qwen21-win-py312.txt').read_text()
        self.assertIn('b95637a98dda87a679321a2dfde5f166f22a8119',general)
        self.assertIn('transformers==5.9.0',general)
        self.assertIn('80c7ed262aeffbeb43ef13ae04baeb9b84515a69',qwen)
        self.assertIn('transformers==5.17.0',qwen)
        self.assertIn('bitsandbytes==0.50.2',qwen)

    def test_incomplete_asset_transfer_never_becomes_a_ready_cache(self):
        root=Path(tempfile.mkdtemp())
        spec=dict(test=dict(repository='test/repo',revision='a'*40,
            files=[dict(name='tokenizer.json',bytes=10,sha256=hashlib.sha256(b'0123456789').hexdigest())]))
        reply=Mock();reply.__enter__=Mock(return_value=reply);reply.__exit__=Mock(return_value=False)
        reply.read.return_value=b'partial'
        with patch('slstack.media_assets.specifications',return_value=spec),patch('slstack.media_assets.urllib.request.urlopen',return_value=reply):
            with self.assertRaisesRegex(ValueError,'Incomplete media asset transfer'):media_assets.install(root)
            self.assertFalse(media_assets.install(root,check=True)['ready'])
            self.assertFalse((media_assets.directory(root,'test')/'tokenizer.json').exists())


if __name__=='__main__':unittest.main()
