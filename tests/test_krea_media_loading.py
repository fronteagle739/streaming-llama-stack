"""Resident Krea tensors must outlive the temporary DirectStorage ring."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class KreaDirectStorageTests(unittest.TestCase):
    def setUp(self):
        from slstack.native_broker import vendor
        vendor()
        from studio.krea2_diffusion import _Krea2TransformerStreamer
        self.owner = object.__new__(_Krea2TransformerStreamer)
        self.owner.source_path = 'models/krea.safetensors'
        self.owner.device = 'cuda'
        self.owner.entries = {'target.weight': ('source.weight', None)}
        self.owner.keys = {'source.weight', 'source.weight_scale'}
        self.original = Mock()
        self.original.get_tensor.side_effect = AssertionError('CPU checkpoint mmap was used')
        self.owner._sf = self.original

    def exercise(self, failure=False):
        events = []
        value = Mock(); value.clone.return_value = 'owned tensor'
        self.owner._read_target = Mock(side_effect=lambda target: self.owner._sf.get_tensor('source.weight'))
        self.owner._set_target = Mock(side_effect=RuntimeError('bind failed') if failure else
                                     lambda *a: events.append(('bound', a)))
        refs = {k: SimpleNamespace(name=k, nbytes=16) for k in self.owner.keys}
        stream = Mock(mode=3)
        stream.wait_layer.return_value = {k: value for k in refs}
        stream.release_layer.side_effect = lambda *a: events.append('released')
        modules = {
            'torch': SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda *a: events.append('synchronized'))),
            'cosmos_windows.weight_streaming.ds_backend': SimpleNamespace(DirectStorageStreamer=Mock(return_value=stream)),
            'cosmos_windows.weight_streaming.safetensors_index': SimpleNamespace(build_index=lambda *a: SimpleNamespace(refs=refs)),
        }
        with patch.dict(sys.modules, modules), patch('studio.krea2_diffusion.sys.platform', 'win32'):
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'bind failed'):
                    self.owner._bind_targets(['target.weight'])
            else:
                self.owner._bind_targets(['target.weight'])
                self.assertEqual(events, [('bound', ('target.weight', 'owned tensor')), 'synchronized', 'released'])
        self.assertIs(self.owner._sf, self.original)
        self.original.get_tensor.assert_not_called()
        stream.close.assert_called_once()

    def test_gpu_owned_copy_precedes_ring_release(self):
        self.exercise()

    def test_failed_binding_restores_reader_and_closes_streamer(self):
        self.exercise(failure=True)


if __name__ == '__main__':
    unittest.main()
