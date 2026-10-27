import importlib.util
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch
from slstack.config import ROOT
from slstack.backends import BACKENDS,command,commands
from slstack.native_broker import vendor
from slstack.general_templates import minja_template,template_file

vendor()
from studio.general_launch import stream_arch_id,canonical_streaming_rtx,streaming_prefill_env


class Exports(unittest.TestCase):
    def test_all_backend_builders_and_real_receiver_options(self):
        receiver=(ROOT/'engines/streaming-llama/prefill_llama.cpp/tools/decode-receiver/main.cpp').read_text()
        allowed=set(re.findall(r'arg == "([^"]+)"',receiver))
        for name in BACKENDS:
            with self.subTest(backend=name):
                cfg=dict(backend=name,model_path='models/model',bind_address='127.0.0.1',fa='on',
                         binary='backend-binary',ctx=262144,mtp_mode='off')
                with tempfile.TemporaryDirectory() as temp:
                    argv=command(cfg,runtime=temp,python='decode-python',write_settings=False)
                    self.assertTrue(argv)
                    if name in ('llama-decode','llama-decode-upstream'):
                        self.assertTrue(set(x for x in argv[1:] if x.startswith('-'))<=allowed)
                        self.assertEqual(argv[argv.index('--kv-host')+1],'127.0.0.1')
                        self.assertEqual(argv[argv.index('--token-stream-host')+1],'127.0.0.1')
                        self.assertIn('--flash-attn',argv)
                    if name=='mtp-shim':self.assertEqual(len(commands(cfg,runtime=temp,python='decode-python')),2)

    def test_glm_family_and_flash_are_not_conflated(self):
        for name in ('GLM-5.1','GLM-5.2','GLM-5.3','GLM5.1','GLM5.2','GLM5.3'):
            self.assertEqual(stream_arch_id(name),'glm-dsa')
        self.assertEqual(stream_arch_id('GLM-5.3-Flash'),'glm5-next')
        self.assertEqual(stream_arch_id('', 'models/GLM-5.3-Flash-Q4.gguf'),'glm5-next')
        cfg=canonical_streaming_rtx(dict(arch='glm-dsa',fa='on',ubatch=2048,wf_gpu_bufs=2,dsa_sparse=True))
        self.assertEqual((cfg['fa'],cfg['ubatch']),('off',512))
        env=streaming_prefill_env(cfg)
        self.assertEqual(env['LLAMA_MLA_SPARSE'],'1')
        self.assertEqual(env['LLAMA_DSA_SCORE_CUBLAS'],'1')

    def test_model_template_preserves_control_tokens_and_numbers(self):
        original='[gMASK]<sop>{{ m.content.0.output }} {{ m.tool_calls.0.function.name }} 0.95'
        expected='[gMASK]<sop>{{ m.content[0].output }} {{ m.tool_calls[0].function.name }} 0.95'
        self.assertEqual(minja_template(original),expected)
        with tempfile.TemporaryDirectory() as temp:
            path=template_file(temp,metadata={'tokenizer.chat_template':original})
            self.assertEqual(Path(path).read_text(),expected)
            self.assertEqual(template_file(temp,metadata={'tokenizer.chat_template':original}),path)
            self.assertIsNone(template_file(temp,metadata={'tokenizer.chat_template':'{{ messages }}'}))

    def test_glm_launch_uses_model_metadata_instead_of_kimi_defaults(self):
        from slstack.general_engine import launch_spec
        def string(s):
            b=s.encode();return struct.pack('<Q',len(b))+b
        fields=[('general.architecture',8,string('glm-dsa')),('glm-dsa.block_count',4,struct.pack('<I',79)),
                ('glm-dsa.nextn_predict_layers',4,struct.pack('<I',1)),
                ('tokenizer.chat_template',8,string('[gMASK]<sop>{{ m.content.0.output }}'))]
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)/'model.gguf'
            model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,len(fields))+b''.join(string(k)+struct.pack('<I',kind)+value for k,kind,value in fields))
            cfg=dict(runtime_root=temp,model=str(model),host='127.0.0.1',general=dict(mac=dict(host='127.0.0.1',backend='mlx-receiver')))
            argv,env,_=launch_spec(cfg)
            self.assertIn('glm-dsa',argv)
            self.assertEqual(argv[argv.index('--decode-remote-layers')+1],'78')
            self.assertEqual(Path(argv[argv.index('--chat-template-file')+1]).read_text(),'[gMASK]<sop>{{ m.content[0].output }}')

    def test_omlx_never_imports_global_settings(self):
        from studio.backend_argv import omlx_write_settings
        with tempfile.TemporaryDirectory() as temp:
            with patch('studio.backend_argv.os.path.expanduser',side_effect=AssertionError('unexpected global settings read')):
                target=omlx_write_settings(temp,262144)
            self.assertEqual(json.loads(Path(target).read_text())['sampling']['max_context_window_policy'],262144)

    def test_decode_preflight_checks_paths_and_isolates_omlx(self):
        from slstack.decode import inspect,settings
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)/'model';model.mkdir()
            cfg=dict(runtime_root=temp,decode=dict(backend='omlx',model_path=str(model)))
            result=inspect(cfg)
            self.assertFalse(result['ready'])
            self.assertIn('omlx-venv',result['missing'][0])
            self.assertEqual(settings(cfg,selected='mlx-vlm')['backend'],'mlx-vlm')
            with self.assertRaises(ValueError):settings(dict(runtime_root=temp))
            cfg['decode'].update(backend='llama-server',binary=__import__('sys').executable)
            self.assertTrue(inspect(cfg)['ready'])

    def test_native_sources_resources_and_licenses_survive_export(self):
        spec=importlib.util.spec_from_file_location('backend_audit',ROOT/'scripts/audit_backend_sources.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.audit()
        self.assertEqual(len(result['backends']),len(BACKENDS))
        self.assertTrue(result['all_retained_components_complete'],result)
        self.assertEqual(result['mlx_unpacked_build_inputs'],[])
        self.assertTrue((ROOT/'vendor/mac-mlx/mlx_vlm/models/mimo_v2/_arch_v26.LICENSE').is_file())

if __name__=='__main__':unittest.main()
