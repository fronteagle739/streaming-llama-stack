import unittest
from unittest.mock import patch

from slstack.config import load
from slstack.native_profiles import MODES, profile


class NativeModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from slstack.native_broker import vendor
        vendor()

    def test_each_delivered_mode_preserves_arithmetic_and_locks_storage(self):
        from studio.dsv41_session_profiles import profile as original
        for mode in MODES:
            with self.subTest(mode=mode):
                expected = original(mode)
                actual = profile(mode)
                self.assertEqual(actual['techniques'], expected['techniques'])
                a, b = actual['arguments'].copy(), expected['arguments'].copy()
                for key in ('--resident-pageable-copies', '--tile-host-registration'):
                    a.pop(key)
                    b.pop(key)
                self.assertEqual(a, b)
                self.assertEqual(actual['arguments']['--resident-pageable-copies'], '0')
                self.assertEqual(actual['arguments']['--tile-host-registration'], '1')

    def test_cli_mode_selects_prefill_without_changing_media_mode(self):
        from slstack.__main__ import main
        class Checked(Exception):
            pass
        for mode in MODES:
            def capture(config, overrides):
                self.assertEqual(overrides['native_tier'], mode)
                self.assertNotIn('image_mode', overrides)
                self.assertNotIn('video_mode', overrides)
                raise Checked
            with patch('sys.argv', ['slstack', 'rtx', '--mode', mode]), patch('slstack.__main__.load', side_effect=capture):
                with self.assertRaises(Checked):
                    main()

    def test_unknown_config_mode_refused(self):
        with self.assertRaisesRegex(ValueError, 'mode'):
            load(overrides={'native_tier': 'typo'})


if __name__ == '__main__':
    unittest.main()
