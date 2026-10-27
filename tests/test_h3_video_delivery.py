import json
from pathlib import Path
import sys
import tempfile
import struct
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from slstack.config import load, ROOT
from slstack.router import Stack, handler
from slstack import media_tools


class VideoHttpTests(unittest.TestCase):
    def test_browser_ranges_and_full_download_preserve_exact_bytes(self):
        root=Path(tempfile.mkdtemp());(root/'outputs').mkdir()
        payload=bytes(range(256))*1024;(root/'outputs/clip.mp4').write_bytes(payload)
        engine=Mock();engine.models=None
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(Stack(load(overrides=dict(runtime_root=str(root),gpu_uuid='GPU-fixture')),engine)))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}/outputs/clip.mp4'
        try:
            for request,first,last in [('bytes=10-29',10,29),('bytes=262130-',262130,len(payload)-1),('bytes=-20',len(payload)-20,len(payload)-1)]:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':request})) as response:
                    self.assertEqual(response.status,206)
                    self.assertEqual(response.headers['Content-Range'],f'bytes {first}-{last}/{len(payload)}')
                    self.assertEqual(response.read(),payload[first:last+1])
            with urllib.request.urlopen(url) as response:
                self.assertEqual(response.headers['Content-Type'],'video/mp4')
                self.assertEqual(response.headers['Accept-Ranges'],'bytes')
                self.assertEqual(response.read(),payload)
            for value in ('bytes=999999-','bytes=10-2','bytes=-0','bytes=0-1,4-5'):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(urllib.request.Request(url,headers={'Range':value}))
                self.assertEqual(error.exception.code,416)
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_ready_marker_cannot_hide_changed_or_unverified_tools(self):
        root=Path(tempfile.mkdtemp());tools=media_tools.directory(root);tools.mkdir(parents=True)
        for name in ('ffmpeg','ffprobe'):(tools/(name+'.exe')).write_bytes(b'fixture')
        (tools.parent/'installed.json').write_text(json.dumps(dict(archive_sha256=media_tools.SHA256,files={})))
        self.assertFalse(media_tools.install(root,check=True)['ready'])


class H3AdapterTests(unittest.TestCase):
    def test_h3_default_duration_and_explicit_frame_count(self):
        from slstack.tools import Tools
        root=Path(tempfile.mkdtemp());engine=Mock()
        tools=Tools(load(overrides=dict(runtime_root=str(root),gpu_uuid='GPU-fixture',video_arch='minimax-h3')),engine)
        tools.preferences=Mock();tools.preferences.select.return_value=dict(arch='minimax-h3')
        self.assertEqual(tools.schemas['generate_video']['properties']['num_frames']['default'],121)
        tools.call('generate_video',dict(prompt='A red cup'))
        self.assertEqual(engine.media.call_args.args[1]['num_frames'],121)
        tools.call('generate_video',dict(prompt='A red cup',num_frames=17))
        self.assertEqual(engine.media.call_args.args[1]['num_frames'],17)

    def test_chat_prompt_uses_studio_fields_and_preserves_explicit_captions(self):
        from slstack.native_broker import vendor
        vendor()
        from studio.h3_image_mode import build_video_prompt
        prompt=build_video_prompt('A red cup rotates on a table.')
        self.assertTrue(prompt.startswith('integrated_multimodal_description: [Shot 1] A red cup'))
        self.assertIn('overall_soundscape:',prompt)
        self.assertEqual(build_video_prompt(prompt),prompt)
        explicit='integrated_multimodal_description: [Shot 1] A person says <d>[English] Hello</d>\noverall_soundscape: Rain\nnon_diegetic_music: Jazz'
        self.assertEqual(build_video_prompt(explicit),explicit)
    def test_qkv_layout_follows_export_identity_and_explicit_config(self):
        from slstack.native_broker import vendor
        vendor()
        from studio.h3_model_profiles import h3_checkpoint_worker_env
        root=Path(tempfile.mkdtemp());(root/'transformer').mkdir()
        checkpoint=root/'transformer/minimax_h3_turbo_adalnfix.safetensors';checkpoint.write_bytes(b'fixture')
        self.assertEqual(dict(h3_checkpoint_worker_env(root)),{'STUDIO_H3_COMFY_QKV_SECTIONED':'1'})
        (root/'transformer/config.json').write_text(json.dumps(dict(qkv_layout='interleaved')))
        self.assertEqual(dict(h3_checkpoint_worker_env(root)),{'STUDIO_H3_COMFY_QKV_SECTIONED':'0'})
        self.assertEqual(h3_checkpoint_worker_env('models/MATLOWAI'),())

    def test_transformer_index_excludes_nested_composition_caches(self):
        sys.path.insert(0,str(ROOT/'vendor/engines'))
        from cosmos_windows.weight_streaming.safetensors_index import build_index
        root=Path(tempfile.mkdtemp());nested=root/'.h3-composed'/'other'/'transformer';nested.mkdir(parents=True)
        header=json.dumps({'blocks.0.weight':dict(dtype='F16',shape=[2,2],data_offsets=[0,8])}).encode()
        for folder in (root,nested):(folder/'weights.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*8)
        selected=build_index(root,recursive=False)
        self.assertEqual(set(selected.refs),{'blocks.0.weight'})
        self.assertEqual(Path(selected.refs['blocks.0.weight'].file),root/'weights.safetensors')
        self.assertEqual(len(build_index(root).refs),2)

    def test_unowned_worker_cannot_silently_skip_the_shared_guard(self):
        from slstack.native_broker import vendor
        vendor()
        from studio.safety import guard_argv
        with patch('psutil.Process',return_value=SimpleNamespace(parents=lambda:[])):
            with self.assertRaisesRegex(RuntimeError,'owned process guard'):
                guard_argv(['worker'])

    def test_actual_shared_guard_executes_the_nested_command(self):
        from slstack.process import OwnedProcess
        root=Path(tempfile.mkdtemp());cfg=load(overrides=dict(runtime_root=str(root),gpu_uuid='GPU-fixture',host_budget_gib=.5,job_commit_gib=.5,min_available_gib=2,paging_telemetry=True))
        script="from slstack.native_broker import vendor; vendor(); from studio.safety import guard_argv; import subprocess,sys; subprocess.run(guard_argv([sys.executable,'-c',\"print('H3_NESTED_EXECUTED')\"]),check=True)"
        job=OwnedProcess(cfg,[sys.executable,'-B','-c',script],'h3-nested-proof',timeout=30)
        try:
            job.wait()
            self.assertIn('H3_NESTED_EXECUTED',job.log.read_text())
        finally:job.stop()

    def test_complete_single_file_folder_reaches_worker_without_manifest_or_weight_copy(self):
        from slstack.native_broker import vendor
        vendor()
        from studio.adapter import RealCosmosBackend
        from studio.models import GenerationParams
        root=Path(tempfile.mkdtemp());model=root/'h3';(model/'transformer').mkdir(parents=True)
        (model/'transformer/weights.safetensors').write_bytes(b'fixture')
        backend=object.__new__(RealCosmosBackend)
        backend.settings=SimpleNamespace(cosmos_windows_path=str(ROOT/'vendor/engines'),cosmos_python=sys.executable)
        backend.unload=Mock();backend._cleanup_torch=Mock();backend._cuda_device=Mock(return_value='cuda:0')
        backend._maybe_enhance_h3=Mock()
        def generate(argv,*args,**kw):
            Path(argv[argv.index('--out')+1]).write_bytes(b'fixture video')
            return 0,[]
        backend._run_guarded_subprocess=Mock(side_effect=generate)
        params=GenerationParams(model=str(model),model_arch='minimax-h3',prompt='A cup rotating on a table.',
            modality='t2v',width=256,height=256,num_frames=5,steps=3,output_dir=str(root),h3_prompt_enhance=False,h3_attention_backend='sdpa')
        layout=SimpleNamespace(complete=True)
        module=SimpleNamespace(inspect_h3_transformer=Mock(return_value=layout))
        with patch.dict(sys.modules,{'cosmos_windows.weight_streaming.minimax_h3_backend':module}), patch('studio.chat_artifacts.settings',SimpleNamespace(output_dir=root)),patch('studio.safety.guard_argv',side_effect=lambda argv,**kw:argv):
            first=backend._generate_minimax_h3(params)
            second=backend._generate_minimax_h3(params)
        self.assertNotEqual(first.path,second.path)
        self.assertEqual(first.kind,'video')
        self.assertFalse((model/'transformer/model.safetensors.index.json').exists())
        argv=backend._run_guarded_subprocess.call_args.args[0]
        self.assertEqual(argv[argv.index('--attention-backend')+1],'sdpa')

        # Exercise the actual adapter's command builder for both reference paths.
        source=root/'conditioning.png';source.write_bytes(bytes.fromhex('89504e470d0a1a0a'))
        for modality,extra,flag in [('i2v',dict(image_path=str(source)),'--image'),
                                    ('ref2v',dict(reference_images=[str(source)]),'--reference-image')]:
            request=params.model_copy(update=dict(modality=modality,**extra))
            with patch.dict(sys.modules,{'cosmos_windows.weight_streaming.minimax_h3_backend':module}),patch('studio.chat_artifacts.settings',SimpleNamespace(output_dir=root)),patch('studio.safety.guard_argv',side_effect=lambda argv,**kw:argv):
                backend._generate_minimax_h3(request)
            argv=backend._run_guarded_subprocess.call_args.args[0]
            self.assertEqual(argv[argv.index('--mode')+1],modality)
            self.assertEqual(argv[argv.index(flag)+1],str(source))
            if modality=='i2v':self.assertEqual(argv[argv.index('--frame-index')+1],'0')
