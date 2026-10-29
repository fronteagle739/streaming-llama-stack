import unittest

from slstack.native_broker import check_packed_layout

BRIDGE = {'LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL': 'bridge.dll', 'LLAMA_DSV41_STORED_MXFP4': '1',
          'LLAMADECODE_PREFILL_ROUTED': '0'}


class PackedFp4DecodeGate(unittest.TestCase):
    def test_decode_allowed_with_bridge_stored_and_full_stage_reads(self):
        check_packed_layout(dict(env_overrides=dict(BRIDGE, LLAMADECODE_DS4_DECODE='1')))
        check_packed_layout(dict(env_overrides=dict(BRIDGE)))

    def test_prefill_only_still_requires_decode_off(self):
        check_packed_layout(dict(native_fp4_prefill_only=True, env_overrides=dict(BRIDGE, LLAMADECODE_DS4_DECODE='0')))
        with self.assertRaises(ValueError):
            check_packed_layout(dict(native_fp4_prefill_only=True, env_overrides=dict(BRIDGE, LLAMADECODE_DS4_DECODE='1')))

    def test_missing_bridge_stored_or_routed_reads_refused(self):
        for key, value in (('LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL', ''), ('LLAMA_DSV41_STORED_MXFP4', '0'),
                           ('LLAMADECODE_PREFILL_ROUTED', '1')):
            with self.assertRaises(ValueError):
                check_packed_layout(dict(env_overrides=dict(BRIDGE, **{key: value})))


if __name__ == '__main__':
    unittest.main()
