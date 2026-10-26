"""Real PEFT/meta adapter loading without full model weights or a GPU."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('torch'), 'needs the media runtime')
class AnimaAdapterTests(unittest.TestCase):
    def test_native_adapter_components_materialize_and_base_stays_streamable(self):
        sys.path.insert(0, str(ROOT / 'vendor/engines'))
        import torch
        from peft.tuners.lora import Linear
        from peft import LoraConfig
        from safetensors.torch import save_file
        from diffusers.loaders import AnimaLoraLoaderMixin
        from cosmos_windows.weight_streaming.anima_backend import (
            _anima_binding_target, _materialize_anima_adapters,
        )
        root = ROOT / 'runtime' / ('anima-adapter-' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        path = root / 'adapter.safetensors'
        tensors = {}
        models = {}
        for component, native, internal in (
            ('transformer', 'blocks.0.self_attn.q_proj', 'transformer_blocks.0.attn1.to_q'),
            ('text_conditioner', 'llm_adapter.blocks.0.cross_attn.k_proj', 'blocks.0.cross_attn.k_proj'),
        ):
            model = torch.nn.Module()
            owner = model
            parts = internal.split('.')
            for part in parts[:-1]:
                child = torch.nn.Module(); owner.add_module(part, child); owner = child
            base = torch.nn.Linear(4, 4, bias=False, device='meta')
            layer = Linear(base, adapter_name='test', config=LoraConfig(r=2, lora_alpha=2), r=2, lora_alpha=2)
            owner.add_module(parts[-1], layer)
            models[component] = model
            tensors['diffusion_model.' + native + '.lora_A.weight'] = torch.ones(2, 4) * .25
            tensors['diffusion_model.' + native + '.lora_B.weight'] = torch.ones(4, 2) * .5
        save_file(tensors, str(path))
        pipe = type('Pipe', (AnimaLoraLoaderMixin,), {})()
        for name, model in models.items():setattr(pipe, name, model)
        _materialize_anima_adapters(pipe, [{'path': str(path)}], ['test'], 'cpu')
        for name, model in models.items():
            for key, value in model.named_parameters():
                if 'lora_' in key:
                    self.assertFalse(value.is_meta)
                    self.assertTrue(torch.isfinite(value).all())
                else:self.assertTrue(value.is_meta)
        target = _anima_binding_target(models['transformer'], 'transformer_blocks.0.attn1.to_q.weight')
        self.assertEqual(target, 'transformer_blocks.0.attn1.to_q.base_layer.weight')
        from accelerate.utils import set_module_tensor_to_device
        set_module_tensor_to_device(models['transformer'], target, 'cpu', value=torch.eye(4))
        layer = models['transformer'].get_submodule('transformer_blocks.0.attn1.to_q')
        self.assertTrue(torch.allclose(layer(torch.ones(1, 4)), torch.full((1, 4), 2.0)))
        layer.set_scale('test', .5)
        self.assertTrue(torch.allclose(layer(torch.ones(1, 4)), torch.full((1, 4), 1.5)))

    def test_standalone_child_launch_uses_outer_job_without_studio_daemon(self):
        sys.path.insert(0, str(ROOT / 'vendor/studio'))
        from studio import worker_registry
        with patch.object(worker_registry, '_owner', None), patch.object(worker_registry.subprocess, 'Popen') as spawn:
            worker_registry._spawn([sys.executable, '-c', 'pass'])
        spawn.assert_called_once()


if __name__ == '__main__':unittest.main()
