"""Exercise the actual bounded native loader, without a GPU or model file."""
import json
import os
from pathlib import Path
import struct
import subprocess
import unittest
import uuid


class NativeOutputLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get('SLSTACK_TEST_LORA_READER')
        if not executable:
            raise unittest.SkipTest('Build the native_output_lora_reader test executable first')
        cls.executable = executable
        cls.root = Path(os.environ['SLSTACK_TEST_ROOT']) / 'native-lora-cases'
        cls.root.mkdir(parents=True, exist_ok=True)

    def fixture(self, edit=None, *, values=None, truncate=0):
        header = {'__metadata__': {'slstack_arch': 'deepseek41', 'base_sha256': 'a' * 64,
                                  'rank': '2', 'alpha': '4'},
                  'base_model.model.lm_head.lora_A.weight': {'dtype': 'F32', 'shape': [2, 4], 'data_offsets': [0, 32]},
                  'base_model.model.lm_head.lora_B.weight': {'dtype': 'F32', 'shape': [3, 2], 'data_offsets': [32, 56]}}
        if edit:
            edit(header)
        data = json.dumps(header).encode()
        payload = struct.pack('<Q', len(data)) + data + struct.pack('<14f', *(values or [0.125] * 14))
        target = self.root / (uuid.uuid4().hex + '.safetensors')
        target.write_bytes(payload[:-truncate] if truncate else payload)
        return target

    def read(self, target, digest='a' * 64):
        return subprocess.run([self.executable, str(target), digest, '4', '3'],
                              capture_output=True, text=True, timeout=10,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def test_valid_standard_safetensors(self):
        result = self.read(self.fixture())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('rank=2 alpha=4', result.stdout)

    def test_base_binding(self):
        self.assertNotEqual(self.read(self.fixture(), digest='b' * 64).returncode, 0)

    def test_shape_and_dtype(self):
        for field, value in [('shape', [2, 3]), ('dtype', 'F16')]:
            result = self.read(self.fixture(lambda h: h['base_model.model.lm_head.lora_A.weight'].update({field: value})))
            self.assertNotEqual(result.returncode, 0)

    def test_overlap_and_truncation(self):
        result = self.read(self.fixture(lambda h: h['base_model.model.lm_head.lora_B.weight'].update(data_offsets=[0, 24])))
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(self.read(self.fixture(truncate=4)).returncode, 0)

    def test_unsupported_target_and_rank(self):
        def rename(h):
            h['base_model.model.q_proj.lora_A.weight'] = h.pop('base_model.model.lm_head.lora_A.weight')
        self.assertNotEqual(self.read(self.fixture(rename)).returncode, 0)
        self.assertNotEqual(self.read(self.fixture(lambda h: h['__metadata__'].update(rank='65'))).returncode, 0)

    def test_non_finite_weight(self):
        self.assertNotEqual(self.read(self.fixture(values=[float('nan')] + [0.0] * 13)).returncode, 0)


if __name__ == '__main__':
    unittest.main()
