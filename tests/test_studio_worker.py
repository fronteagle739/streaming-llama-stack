"""Exercise the worker's real Studio parameter schema without loading GPU weights."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]


class MediaFailureTests(unittest.TestCase):
    def test_worker_error_is_preserved_without_injecting_the_guard_receipt(self):
        from slstack.engines import Engines
        root=ROOT/'runtime'/('worker-error-'+uuid.uuid4().hex);root.mkdir(parents=True)
        job=Mock();job.log=root/'image.log'
        job.log.write_text('DS4STACK_RESULT '+json.dumps(dict(ok=False,error='Required model component is missing'))+'\n')
        job.wait.side_effect=RuntimeError('Large guard receipt containing diagnostic state')
        engine=object.__new__(Engines);engine.cfg=dict(studio_python=sys.executable,runtime_root=str(root))
        with patch('slstack.engines.OwnedProcess',return_value=job):
            with self.assertRaisesRegex(RuntimeError,'^Required model component is missing$'):
                engine._media('image',{})
        job.stop.assert_called_once()

    def test_resource_admission_failure_becomes_a_chat_tool_error(self):
        from slstack.router import Stack
        stack=object.__new__(Stack);stack.cfg={};stack.tools=Mock()
        stack.tools.call.side_effect=MemoryError('Not enough available RAM')
        result=stack.tool_result(dict(id='call-1',function=dict(name='generate_image',arguments='{}')),[])
        self.assertEqual(json.loads(result['content']),dict(error='Not enough available RAM'))


@unittest.skipUnless(importlib.util.find_spec('pydantic'), 'needs the media runtime')
class StudioWorkerSchemaTests(unittest.TestCase):
    def test_config_only_zimage_folder_does_not_reach_weight_loading(self):
        from slstack.studio_worker import resolve_checkpoint
        root=ROOT/'runtime'/('zimage-bundle-'+uuid.uuid4().hex);(root/'transformer').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'no transformer weights'):
            resolve_checkpoint(str(root),'z-image',{})
        (root/'transformer/model.safetensors').write_bytes(b'fixture')
        support={};resolved=resolve_checkpoint(str(root),'z-image',support)
        self.assertEqual(Path(resolved),root/'transformer/model.safetensors')
        self.assertEqual(support['config_dir'],str(root))

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
        cases = [('image', 'anima-1.0', [adapter], 'streaming'), ('image', 'z-image', [], 'streaming'),
                 ('image', 'moss-image', [], 'streaming'), ('video', 'ltx-2.3', [], 'streaming'),
                 ('image', 'anima-1.0', [], 'resident'), ('image', 'minimax-h3', [], 'resident'),
                 ('video', 'minimax-h3', [], 'streaming')]
        env = dict(STUDIO_DATA_ROOT=str(root / 'studio'), STUDIO_COSMOS_WINDOWS_PATH=str(ROOT / 'vendor/engines'))
        with patch.dict(os.environ, env):
            backend = Mock()
            backend.generate.return_value = SimpleNamespace(model_dump=lambda **kw: {'path': str(root / 'outputs/fixture.png')})
            def generate(params, on_step=None):
                on_step(1, params.steps)
                on_step(2, params.steps)
                return backend.generate.return_value
            backend.generate.side_effect = generate
            with (patch('studio.adapter.load_backend', return_value=(backend, False, 'schema boundary')),
                  patch('studio.custom_models.comfy_support_defaults', return_value={})):
                for kind, arch, loras, mode in cases:
                    with self.subTest(kind=kind, arch=arch, loras=bool(loras)):
                        config = dict(cfg, **{kind + '_arch': arch, kind + '_mode': mode})
                        from slstack.media_progress import MediaProgress
                        progress = MediaProgress(root, kind, args['steps'])
                        result = run(dict(cfg=config, kind=kind, params=dict(args, lora_files=loras), progress_file=str(progress.path)))
                        self.assertEqual(progress.snapshot()['step'], 2)
                        params = backend.generate.call_args.args[0]
                        self.assertIsInstance(params, GenerationParams)
                        self.assertEqual(params.weight_stream, mode == 'streaming')
                        self.assertEqual(params.load_mode, 'streaming_lazy' if mode == 'streaming' else 'resident_cached')
                        self.assertEqual(params.model_support_files['_stack_explicit_loras_only'], '1')
                        self.assertEqual([(x.path, x.weight) for x in params.loras], [(x['path'], x['weight']) for x in loras])
                        self.assertTrue(result['ok'])
                self.assertEqual(backend.unload.call_count, len(cases))


if __name__ == '__main__':unittest.main()
