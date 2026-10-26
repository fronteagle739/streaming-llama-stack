from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock
from slstack.discovery import scan_models, Scanner
from slstack.lora_discovery import lora_info, compatibility, public_loras
from slstack.preferences import Preferences
from slstack.tools import Tools,media_schema,validate


class AdapterDiscoveryTests(unittest.TestCase):
    def setUp(self):self.root=Path(tempfile.mkdtemp(prefix='slstack-lora-proof-'))

    def file(self,relative,shapes):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        metadata=shapes.pop('__metadata__',{})
        header={k:dict(dtype='F16',shape=v,data_offsets=[0,8]) for k,v in shapes.items()}
        header['__metadata__']=metadata;raw=json.dumps(header).encode()
        path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*8);return path

    def fixtures(self):
        model=self.file('one/model.safetensors',{'transformer_blocks.0.img_mlp.net.0.proj.weight':[16,8]})
        other=self.file('two/other.safetensors',{'transformer_blocks.0.img_mlp.net.0.proj.weight':[32,12]})
        lora=self.file('two/loras/four-step.safetensors',{'transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight':[2,8],
                         'transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_B.weight':[16,2]})
        return model,other,lora

    def test_multiple_roots_find_loras_and_exact_compatible_model(self):
        model,other,lora=self.fixtures();catalog=scan_models([self.root/'one',self.root/'two',self.root/'one'])
        self.assertEqual(len(catalog['models']),2);self.assertEqual(len(catalog['loras']),1)
        listed=public_loras(catalog)
        self.assertEqual(listed[0]['name'],'four-step')
        self.assertEqual([m['name'] for m in listed[0]['compatible_models']],[model.name])
        self.assertFalse(compatibility(catalog['loras'][0],next(m for m in catalog['models'] if m['name']==other.name))[0])

    def test_unknown_and_mismatched_adapters_are_not_attached(self):
        model,_,_=self.fixtures();self.file('two/loras/unknown.safetensors',{'random.layer.lora_A.weight':[2,8]})
        prefs=Preferences(dict(runtime_root=str(self.root/'runtime')))
        prefs.update(dict(comfyui_models=str(self.root/'one'),model_search_roots=[str(self.root/'two')],image_model=str(model)))
        prefs.scanner.thread.join(10);self.assertFalse(prefs.catalog['scanning'])
        listed=prefs.list_loras();self.assertEqual(len(listed),2)
        self.assertEqual(next(r for r in listed if r['name']=='unknown')['compatible_models'],[])
        with self.assertRaisesRegex(ValueError,'incompatible'):prefs.resolve_loras('image',None,[dict(name='unknown',strength=1)])
        with self.assertRaisesRegex(ValueError,'Duplicate'):prefs.resolve_loras('image',None,[dict(name='four-step',strength=.5)]*2)

    def test_strength_and_attachments_are_per_generation_only(self):
        model,_,_=self.fixtures();cfg=dict(runtime_root=str(self.root/'runtime'),image_mode='streaming',video_mode='streaming',bias_mcp_url='')
        engines=Mock();tools=Tools(cfg,engines)
        tools.preferences.update(dict(comfyui_models=str(self.root/'one'),model_search_roots=[str(self.root/'two')],image_model=str(model)))
        tools.preferences.scanner.thread.join(10)
        original=deepcopy(tools.preferences.row)
        tools.call('generate_image',dict(prompt='Scene one',loras=[dict(name='four-step',strength=.7)]))
        self.assertEqual(engines.media.call_args.args[1]['loras'],[dict(name='four-step',strength=.7)])
        resolved=tools.preferences.resolve_loras('image',None,[dict(name='four-step',strength=.7)])
        self.assertEqual(resolved[0]['weight'],.7)
        tools.call('generate_image',dict(prompt='Scene two'))
        self.assertEqual(engines.media.call_args.args[1]['loras'],[])
        self.assertEqual(tools.preferences.row,original)
        response=tools.call('list_loras',dict(model=model.name))
        self.assertEqual(response['loras'][0]['compatible_models'][0]['name'],model.name)
        self.assertNotIn('path',response['loras'][0])
        for value in ([{'name':'four-step','strength':True}],[{'name':'four-step','strength':5}],[{'name':'four-step','unexpected':1}],['four-step']):
            with self.subTest(value=value),self.assertRaises(ValueError):validate(media_schema('image'),dict(prompt='Scene',loras=value))

    def test_saved_tool_toggles_gain_new_tool_without_reset(self):
        cfg=dict(runtime_root=str(self.root/'runtime'));prefs=Preferences(cfg)
        prefs.update(dict(enabled_tools={'generate_image':False}))
        saved=json.loads(prefs.path.read_text());saved['enabled_tools'].pop('list_loras');prefs.path.write_text(json.dumps(saved))
        loaded=Preferences(cfg)
        self.assertTrue(loaded.enabled('list_loras'));self.assertFalse(loaded.enabled('generate_image'))

    def test_moss_step_adapters_and_h3_metadata_are_detected(self):
        moss=self.file('moss-step.safetensors',{'diffusion_model.layers.0.proj.ffn_down.lora_down.weight':[128,8192]})
        h3=self.file('h3-step.safetensors',{'__metadata__':{'model':'MiniMax-H3'},'diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight':[128,5376]})
        self.assertEqual(lora_info(moss)['arches'],['moss-image'])
        self.assertEqual(lora_info(h3)['arches'],['minimax-h3'])


if __name__=='__main__':unittest.main()
