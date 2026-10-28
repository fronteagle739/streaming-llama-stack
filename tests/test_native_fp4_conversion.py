"""Independent source/GGML layout oracles and pinned writer integration."""
import importlib.util
import http.client
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_converter', ROOT/'scripts/convert_dsv41_gguf.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
spec = importlib.util.spec_from_file_location('source_audit', ROOT/'scripts/audit_dsv41_source.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Source:
    def __init__(self, weights, scales):
        self.weights = np.ascontiguousarray(weights, dtype=np.uint8)
        self.scales = np.ascontiguousarray(scales, dtype=np.uint8)
        self.tensors = {'expert.weight': self.info('expert.weight'), 'expert.scale': self.info('expert.scale')}

    def info(self, name):
        scale = name.endswith('.scale')
        data = self.scales if scale else self.weights
        return {'dtype': 'F8_E8M0' if scale else 'I8', 'shape': list(data.shape), 'nbytes': data.nbytes}

    def read(self, name):
        return (self.scales if name.endswith('.scale') else self.weights).tobytes()


@unittest.skipIf(np is None, 'Optional converter NumPy dependency is unavailable')
class NativePackingTests(unittest.TestCase):
    def test_all_codes_and_finite_exponents_round_trip_with_independent_value_oracle(self):
        # Every packed byte occurs at every position, for each finite E8M0.
        blocks = (np.arange(256, dtype=np.uint16)[:, None] + np.arange(16)[None, :]).astype(np.uint8)
        weights = np.tile(blocks, (255, 1))
        scales = np.repeat(np.arange(255, dtype=np.uint8), 256).reshape(-1, 1)
        source = Source(weights, scales)
        output = np.frombuffer(converter.pack_native_expert(source, 'expert.weight'), np.uint8).reshape(-1, 17)
        np.testing.assert_array_equal(output[:, 0], scales[:, 0])
        decoded_codes = np.concatenate((output[:, 1:] & 15, output[:, 1:] >> 4), axis=1)
        source_codes = np.empty((len(weights), 32), np.uint8)
        source_codes[:, 0::2], source_codes[:, 1::2] = weights & 15, weights >> 4
        np.testing.assert_array_equal(decoded_codes, source_codes)
        restored = decoded_codes[:, 0::2] | (decoded_codes[:, 1::2] << 4)
        np.testing.assert_array_equal(restored, weights)
        source_lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
        ggml_lut = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12])
        expected = np.ldexp(source_lut[source_codes], scales.astype(np.int32) - 127)
        actual = np.ldexp(ggml_lut[decoded_codes].astype(np.float64), output[:, :1].astype(np.int32) - 128)
        np.testing.assert_array_equal(actual, expected)

    def test_multirow_multiblock_layout_and_invalid_scales(self):
        source = Source(np.arange(96).reshape(3, 32), np.arange(6).reshape(3, 2))
        result = np.frombuffer(converter.pack_native_expert(source, 'expert.weight'), np.uint8).reshape(3, 2, 17)
        np.testing.assert_array_equal(result[:, :, 0], source.scales)
        source.scales[1, 1] = 255
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            converter.pack_native_expert(source, 'expert.weight')
        source = Source(np.zeros((2, 16)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, 'per32'):
            converter.pack_native_expert(source, 'expert.weight')


class SourceAuditTests(unittest.TestCase):
    def test_incomplete_http_read_is_retried(self):
        response = mock.MagicMock(status=206, headers={'Content-Range': 'bytes 0-3/9'})
        response.__enter__.return_value = response
        response.read.side_effect = [http.client.IncompleteRead(b'ab', 2), b'abcd']
        with mock.patch.object(audit.urllib.request, 'urlopen', return_value=response) as get, \
             mock.patch.object(audit.time, 'sleep'):
            self.assertEqual(audit.request('https://example.test/shard', 0, 3, 9), b'abcd')
            self.assertEqual(get.call_count, 2)

    def test_range_ignored_truncated_or_wrong_offset_fails_closed(self):
        for status, content_range, body in ((200, None, b'abcd'), (206, 'bytes 1-4/9', b'abcd'),
                                            (206, 'bytes 0-3/9', b'abc'), (206, 'bytes 0-3/9', b'abcde')):
            response = mock.MagicMock(status=status, headers={'Content-Range': content_range})
            response.__enter__.return_value = response
            response.read.return_value = body
            with mock.patch.object(audit.urllib.request, 'urlopen', return_value=response), mock.patch.object(audit.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'metadata request failed'):
                    audit.request('https://example.test/shard', 0, 3, 9)

    def test_metadata_git_identity_is_verified(self):
        with mock.patch.object(audit, 'request', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                audit.fetch_blob({'rfilename': 'config.json', 'size': 2, 'blobId': '0'*40})


@unittest.skipIf(np is None, 'Optional converter NumPy dependency is unavailable')
class PinnedWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ANYGGUF_DS4_TOOLS')
        if not directory:
            raise unittest.SkipTest('Pinned tools required')
        cls.modules = converter.load_tools(directory)

    def test_native_plan_changes_only_120_routed_tensors(self):
        fixture = json.loads((ROOT/'tests/fixtures/dsv41-source.json').read_text())
        index = {'weight_map': {pattern.format(expert=e): 'absent.safetensors'
                               for pattern, count in fixture['source_patterns'].items() for e in range(count)}}
        original = converter.read_json(ROOT/'scripts/dsv41/reference-layout.json')['components']['main']['tensors']
        reference = converter.native_layout(original)
        self.assertEqual(sum(reference[name] != row for name, row in original.items()), 120)
        with mock.patch.object(self.modules['glm53_quantize'], 'SourceDB', side_effect=AssertionError('opened payload')):
            plan = converter.make_plan('main', self.modules, fixture['config'], index, reference)
        self.assertEqual(converter.planned_map(plan), reference)
        self.assertEqual(sum(p.qtype == 39 for p in plan), 120)
        self.assertEqual(len(plan), 1046)

    def test_pinned_streaming_writer_copies_experts_without_float_conversion(self):
        q, g = self.modules['deepseek41_quantize'], self.modules['glm53_quantize']
        source = Source(np.arange(32).reshape(2, 16), np.array([[127], [128]]))
        item = g.TensorPlan('blk.0.ffn_gate_exps.weight', (32, 2, 1), 39,
                            role='expert', source='expert.weight', expert_layer=0, expert_count=1)
        item.offset, item.nbytes = 0, 34
        dense = g.TensorPlan('output.weight', (32, 1), 8, role='output', source='dense.weight')
        dense.offset, dense.nbytes = 16384, 34
        dense_calls = []
        class NoFloatQuantizer:
            np = np
            def __init__(self, library): pass
            def to_f32(self, db, name):
                if name != 'dense.weight': raise AssertionError('Expert was dequantized')
                dense_calls.append(name)
                return 'dense values'
            def encode(self, values, qtype):
                if values != 'dense values' or qtype != 8: raise AssertionError('Expert was requantized')
                return b'Q8'*17
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'native.gguf'
            args = types.SimpleNamespace(quants_library='unused', imatrix=None, out=str(output), resume=False, threads=1)
            with mock.patch.object(q, 'NativeQuantizer', NoFloatQuantizer), \
                 mock.patch.object(q.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1 << 40)):
                writer = converter.native_writer(q)
                writer(args, [item, dense], [], source)
            header = converter.read_header(output)
            self.assertEqual(converter.tensor_map(header)[item.name], {'type': 39, 'shape': [32, 2, 1]})
            raw = output.read_bytes()
            # The pinned writer aligns its data start and every tensor to16KiB.
            self.assertEqual(raw[16384:16384+34], converter.pack_native_expert(source, 'expert.weight'))
            self.assertEqual(raw[32768:32768+34], b'Q8'*17)
            self.assertEqual(dense_calls, ['dense.weight'])
            self.assertFalse(output.with_suffix('.gguf.partial').exists())


class ConversionScratchPolicy(unittest.TestCase):
    def test_model_output_can_use_a_separate_volume_from_guard_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'models'/'native.gguf'
            runtime=root/'scratch'
            args=types.SimpleNamespace(out=output,runtime_root=runtime,host_budget_gib=2,runtime_floor_gib=4)
            with mock.patch('slstack.guard.supervise',return_value=0) as supervise, \
                 mock.patch.object(converter.sys,'argv',['convert','--out',str(output),'--runtime-root',str(runtime),'--plan','already-written.json']):
                self.assertEqual(converter.guarded_conversion(args),0)
            spec=supervise.call_args.args[0]
            self.assertIn(str(output),spec['command'])
            self.assertNotIn('--plan',spec['command'])
            self.assertEqual(spec['command'][-1],'--worker')
            for key in ('receipt','log','stop'):
                self.assertEqual(Path(spec[key]).parent,runtime.resolve())
            for key in ('TEMP','TMP','TMPDIR'):
                self.assertEqual(spec['env'][key],str(runtime.resolve()))
            self.assertEqual(spec['host_budget_gib'],2)
            self.assertEqual(spec['min_available_gib'],4)
            self.assertEqual(spec['timeout_s'],21600)


if __name__ == '__main__':
    unittest.main()
