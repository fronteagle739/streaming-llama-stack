import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from slstack.discovery import model_info, scan_models
from slstack.preferences import Preferences


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='slstack-discovery-proof-'))

    def checkpoint(self,name,keys):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        header=json.dumps({k:{'dtype':'F16','shape':[2,2],'data_offsets':[0,8]} for k in keys}).encode()
        path.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*8);return path

    def pipeline(self,name,cls):
        path=self.root/name;path.mkdir(parents=True)
        (path/'model_index.json').write_text(json.dumps({'_class_name':cls}));return path

    def test_requested_families_use_structure_not_filenames(self):
        krea=self.checkpoint('unnamed-native.safetensors',['txtfusion.gate.weight','blocks.0.attn.wq.weight'])
        qwen=self.checkpoint('my-finetune.safetensors',['transformer_blocks.0.img_mlp.net.0.proj.weight'])
        moss=self.pipeline('moss','MOSSImagePipeline');h3=self.pipeline('h3','MiniMaxH3Pipeline')
        for path,arch,kinds in [(krea,'krea2-native',['image']),(qwen,'qwen-image-2.1',['image']),
                                 (moss,'moss-image',['image']),(h3,'minimax-h3',['image','video'])]:
            with self.subTest(arch=arch):
                row=model_info(path);self.assertEqual(row['arch'],arch);self.assertEqual(row['kinds'],kinds)

    def test_lora_is_not_a_standalone_model_even_with_override(self):
        lora=self.checkpoint('adapter.safetensors',['transformer_blocks.0.img_mlp.lora_A.weight'])
        self.assertIsNone(model_info(lora,'qwen-image-2.1'))

    def test_folder_without_model_index_and_deep_scan(self):
        root=self.root.joinpath(*['level']*12);path=root/'pack';(path/'transformer').mkdir(parents=True)
        (path/'transformer/config.json').write_text(json.dumps({'_class_name':'MiniMaxH3DiTModel'}))
        row=scan_models(self.root)
        self.assertTrue(any(m['path']==str(path.resolve()) for m in row['models']));self.assertFalse(row['scanning'])

    def test_cancellation_keeps_partial_results(self):
        self.pipeline('moss','MOSSImagePipeline');stop=threading.Event();stop.set()
        row=scan_models(self.root,cancelled=stop);self.assertFalse(row['scanning'])

    def test_manual_architecture_and_settings_remain_persistent(self):
        model=self.checkpoint('unknown.safetensors',['custom.layer.weight'])
        cfg=dict(runtime_root=str(self.root/'runtime'));prefs=Preferences(cfg)
        row=prefs.update({'image_model':str(model),'image_arch':'moss-image'})
        self.assertEqual(row['settings']['image_arch'],'moss-image')
        again=Preferences(dict(cfg));self.assertEqual(again.row['model_arch_overrides'][str(model.resolve())],'moss-image')

    def test_selected_search_root_is_not_silently_narrowed_to_models(self):
        (self.root/'models').mkdir();prefs=Preferences(dict(runtime_root=str(self.root/'runtime')))
        state=prefs.update({'comfyui_models':str(self.root)})
        self.assertEqual(state['settings']['comfyui_models'],str(self.root.resolve()))

    def test_support_components_are_joined_across_search_roots(self):
        bundle=self.root/'configs'/'anima';(bundle/'text_conditioner').mkdir(parents=True)
        (bundle/'text_conditioner/config.json').write_text(json.dumps({'_class_name':'AnimaTextConditioner'}))
        self.checkpoint('weights/text_encoders/qwen_3_06b_base.safetensors',['embed_tokens.weight'])
        self.pipeline('configs/zimage','ZImagePipeline')
        self.checkpoint('weights/text_encoders/qwen_3_4b.safetensors',['embed_tokens.weight'])
        self.checkpoint('weights/vae/ae.safetensors',['decoder.conv_out.weight'])
        row=scan_models([self.root/'configs',self.root/'weights'])
        self.assertEqual(row['support_files']['anima-1.0']['components_repo'],str(bundle))
        self.assertEqual(Path(row['support_files']['anima-1.0']['text_encoder']).name,'qwen_3_06b_base.safetensors')
        self.assertEqual(row['support_files']['z-image']['config_dir'],str(self.root/'configs/zimage'))
        self.assertEqual(Path(row['support_files']['z-image']['vae']).name,'ae.safetensors')

    def test_pipeline_shards_and_config_bundles_are_not_separate_models(self):
        qwen=self.pipeline('qwen','QwenImage21Pipeline')
        self.checkpoint('qwen/transformer/diffusion_pytorch_model-00001-of-00002.safetensors',['transformer_blocks.0.img_mlp.proj.weight'])
        flux=self.pipeline('klein','Flux2KleinKVPipeline')
        configs=self.pipeline('z-configs','ZImagePipeline')
        row=scan_models(self.root)
        self.assertEqual({m['path'] for m in row['models']},{str(qwen),str(flux)})
        self.assertEqual(row['support_files']['qwen-image-2.1']['components_repo'],str(qwen))
        self.assertEqual(row['support_files']['flux2-klein']['components_repo'],str(flux))
        self.assertEqual(row['support_files']['z-image']['config_dir'],str(configs))


if __name__=='__main__':unittest.main()
