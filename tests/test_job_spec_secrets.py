"""Secret values never reach runtime/jobs files, yet the guarded child still receives them."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from slstack.config import load
from slstack.envvars import is_secret, redact_cfg, split_secrets
from slstack.process import OwnedProcess

SECRET = 'unit-test-secret-value-6f1c2b9a7d'
DIGEST = __import__('hashlib').sha256(SECRET.encode()).hexdigest()
CHILD = ("import hashlib,os;"
         "print('CHILD', hashlib.sha256(os.environ.get('DS4_SERVER_AUTH_TOKEN','').encode()).hexdigest())")


class JobSpecSecretTests(unittest.TestCase):
    def test_names(self):
        for name in ('DS4_SERVER_AUTH_TOKEN', 'HF_TOKEN', 'OPENAI_API_KEY', 'CLIENT_SECRET', 'DB_PASSWORD', 'my_token'):
            self.assertTrue(is_secret(name), name)
        for name in ('PATH', 'CUDA_VISIBLE_DEVICES', 'DS4_COMBO_ENABLED'):
            self.assertFalse(is_secret(name), name)
        public, secrets = split_secrets({'PATH': 'x', 'HF_TOKEN': SECRET})
        self.assertEqual(public, {'PATH': 'x', 'HF_TOKEN': '<redacted>'})
        self.assertEqual(secrets, {'HF_TOKEN': SECRET})
        cfg = redact_cfg({'_control_token': SECRET, 'env_overrides': {'DS4_SERVER_AUTH_TOKEN': SECRET, 'A': '1'},
                          'models': [{'api_key': SECRET}]})
        self.assertNotIn(SECRET, json.dumps(cfg))
        self.assertEqual(cfg['env_overrides']['A'], '1')

    def run_job(self, paging):
        # The token is a ds4-win scoped setting; ds4-win jobs use the shared paging guard.
        with tempfile.TemporaryDirectory() as root:
            cfg = load(None)
            cfg.update(runtime_root=root, host_budget_gib=1, min_available_gib=1, request_timeout_s=60,
                       paging_telemetry=paging, _control_token=SECRET,
                       env_overrides={'DS4_SERVER_AUTH_TOKEN': SECRET})
            stdin = json.dumps({'cfg': redact_cfg(cfg)}) + '\n'
            proc = OwnedProcess(cfg, [sys.executable, '-c', CHILD], 'secret-test', stdin=stdin, scope='ds4-win')
            proc.wait()
            written = [p for p in Path(root, 'jobs').iterdir() if p.is_file()]
            log = proc.log.read_text(errors='replace')
            self.assertIn('CHILD ' + DIGEST, log)
            for path in written:
                self.assertNotIn(SECRET, path.read_text(errors='replace'), path.name)
            spec = json.loads(proc.spec.read_text())
            self.assertEqual(spec['env']['DS4_SERVER_AUTH_TOKEN'], '<redacted>')
            self.assertIn('DS4_SERVER_AUTH_TOKEN', spec['secret_env'])

    def test_guarded_child_receives_secret(self):
        self.run_job(True)

    def test_restore_from_environment(self):
        from unittest.mock import patch
        from slstack.envvars import restore_secrets
        spec = {'env': {'HF_TOKEN': '<redacted>', 'PATH': 'x'}, 'secret_env': ['HF_TOKEN']}
        with patch.dict('os.environ', {'HF_TOKEN': SECRET}):
            self.assertEqual(restore_secrets(spec)['env']['HF_TOKEN'], SECRET)
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(RuntimeError):
            restore_secrets({'env': {}, 'secret_env': ['HF_TOKEN']})


if __name__ == '__main__':
    unittest.main()
