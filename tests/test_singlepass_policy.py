from unittest import TestCase
from slstack.singlepass import arguments, registration


class Registration(TestCase):
    def test_modes_and_unsupported_switch(self):
        for bad in (None, '', 1, 'auto', 'pageable'):
            with self.assertRaises(ValueError):
                registration(bad)
        with self.assertRaisesRegex(ValueError, 'switch arena'):
            arguments(dict(producer_carry_registration='bank', producer_keep_alive=True), {})

    def test_only_registration_changes(self):
        original = {k:'1' for k in ('--pair-ffn-tiles', '--resident-hidden-bf16',
            '--resident-compact-ids', '--resident-metadata', '--direct-carry-budget', '--tile-host-registration')}
        original.update({'--compute-tile':'4096', '--hidden-device-bytes':'2521649916'})
        self.assertEqual(arguments({}, original.copy()), original)
        changed = arguments(dict(producer_carry_registration='bank'), original.copy())
        self.assertEqual(changed, dict(original, **{'--tile-host-registration':'0'}))
        original['--pair-ffn-tiles'] = '0'
        with self.assertRaisesRegex(ValueError, 'paired native'):
            arguments(dict(producer_carry_registration='bank'), original)

    def test_policy_survives_runtime_staging(self):
        from slstack.single_runtime import serving_config
        result = serving_config(dict(producer_carry_registration='bank', host_budget_gib=12),
            runtime_root='runtime', studio_root='studio', llama_producer='producer',
            library_directories=[], revision='test')
        self.assertEqual(result['producer_carry_registration'], 'bank')
