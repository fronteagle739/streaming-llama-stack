import unittest
from unittest.mock import patch

from slstack.__main__ import main


class GuardedCommandTests(unittest.TestCase):
    def test_build_enters_guard_without_touching_model_journal(self):
        cfg = {'runtime_root': 'unused'}
        with patch('sys.argv', ['slstack', 'guarded', '--', 'cmake', '--version']), \
             patch('slstack.__main__.load', return_value=cfg), \
             patch('slstack.__main__.Journal', side_effect=AssertionError('model journal touched')), \
             patch('slstack.lease.recover_leases', side_effect=AssertionError('lease recovery touched')), \
             patch('slstack.process.OwnedProcess') as owned:
            owned.return_value.wait.return_value = {'exit_code': 0}
            self.assertEqual(main(), 0)
            self.assertEqual(owned.call_args.args, (cfg, ['cmake', '--version'], 'guarded'))


if __name__ == '__main__':
    unittest.main()
