"""Exercise the worker's real Studio parameter schema without loading GPU weights."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('pydantic'), 'needs the media runtime')
class StudioWorkerSchemaTests(unittest.TestCase):
    def test_image_video_and_explicit_loras_reach_the_typed_backend(self):
        sys.path.insert(0, str(ROOT / 'vendor/studio'))
        from slstack.studio_worker import run
        from studio.models import GenerationParams
        root = ROOT / 'runtime' / ('worker-schema-' + uuid.uuid4().hex)
        cfg = dict(runtime_root=str(root), comfyui_models='', studio_python=sys.executable,
                   image_model=str(root / 'base.safetensors'), image_arch='anima-1.0',
                   image_support_files={}, image_mode='streaming',
                   video_model=str(root / 'video.safetensors'), video_arch='ltx-2.3',
                   video_support_files={}, video_mode='streaming')
        args = dict(prompt='A red cube', negative_prompt='', width=512, height=512,
                    steps=4, guidance=1.0, seed=1)
        adapter = dict(path=str(root / 'adapter.safetensors'), weight=0.7)
        cases = [('image', 'anima-1.0', [adapter]), ('image', 'z-image', []),
                 ('image', 'moss-image', []), ('video', 'ltx-2.3', [])]
        env = dict(STUDIO_DATA_ROOT=str(root / 'studio'), STUDIO_COSMOS_WINDOWS_PATH=str(ROOT / 'vendor/engines'))
        with patch.dict(os.environ, env):
            backend = Mock()
            backend.generate.return_value = SimpleNamespace(model_dump=lambda **kw: {'path': str(root / 'outputs/fixture.png')})
            with (patch('studio.adapter.load_backend', return_value=(backend, False, 'schema boundary')),
                  patch('studio.custom_models.comfy_support_defaults', return_value={})):
                for kind, arch, loras in cases:
                    with self.subTest(kind=kind, arch=arch, loras=bool(loras)):
                        config = dict(cfg, **{kind + '_arch': arch})
                        result = run(dict(cfg=config, kind=kind, params=dict(args, lora_files=loras)))
                        params = backend.generate.call_args.args[0]
                        self.assertIsInstance(params, GenerationParams)
                        self.assertEqual(params.model_support_files['_stack_explicit_loras_only'], '1')
                        self.assertEqual([(x.path, x.weight) for x in params.loras], [(x['path'], x['weight']) for x in loras])
                        self.assertTrue(result['ok'])
                self.assertEqual(backend.unload.call_count, len(cases))


if __name__ == '__main__':unittest.main()
