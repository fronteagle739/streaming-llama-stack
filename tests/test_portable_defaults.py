"""CPU checks for configured storage policy and fail-closed device admission."""
import ast
import ctypes as C
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PortableDefaultsTests(unittest.TestCase):
    def storage(self):
        # Extract the pure helper without creating application state or a token.
        tree = ast.parse((ROOT / 'vendor/studio/studio/config.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_windows_slow_model_drive')
        class PolicyPath(PureWindowsPath):
            def expanduser(self): return self
        namespace = dict(Path=PolicyPath, Optional=__import__('typing').Optional,
                         _paths=SimpleNamespace(is_windows=lambda: True), os=__import__('os'))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<storage-policy>', 'exec'), namespace)
        return namespace['_windows_slow_model_drive']

    def test_no_volume_is_excluded_without_configuration(self):
        slow = self.storage()
        self.assertFalse(slow('C:' + '/models', {}))
        self.assertFalse(slow('I:' + '/models', {}))

    def test_explicit_policy_is_case_insensitive_and_not_shell_dependent(self):
        slow = self.storage()
        with patch.dict('os.environ', {'STUDIO_SLOW_MODEL_VOLUMES': 'Z:'}):
            self.assertTrue(slow('Q:' + '/models', {'STUDIO_SLOW_MODEL_VOLUMES': ' q: ; r:/ '}))
            self.assertFalse(slow('Z:' + '/models', {'STUDIO_SLOW_MODEL_VOLUMES': 'Q:'}))

    def test_device_identity_and_capability_must_both_pass(self):
        from studio.dsv41_windows_runner import _Native as Native
        import uuid
        expected = 'GPU-' + str(uuid.UUID(int=1))
        capability = [12, 0]

        class Function:
            def __init__(self, name): self.name = name
            def __call__(self, *args):
                if self.name == 'cuDeviceGetUuid_v2':
                    for index, value in enumerate(uuid.UUID(int=1).bytes): args[0][index] = value
                if self.name == 'nvmlDeviceGetCudaComputeCapability':
                    C.cast(args[1], C.POINTER(C.c_int))[0] = capability[0]
                    C.cast(args[2], C.POINTER(C.c_int))[0] = capability[1]
                if self.name == 'nvmlDeviceGetMemoryInfo': args[1][1] = 123
                return 0

        class Library:
            def __getattr__(self, name): return Function(name)

        with patch.object(C, 'WinDLL', return_value=Library(), create=True):
            native = Native.__new__(Native)
            self.assertEqual(native.gpu(0, expected), 123)
            with self.assertRaisesRegex(Exception, 'configured GPU UUID'):
                native.gpu(0, 'GPU-' + str(uuid.UUID(int=2)))
            capability[:] = [8, 6]
            with self.assertRaisesRegex(Exception, 'compute capability'):
                native.gpu(0, expected)
