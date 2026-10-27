"""Real image bytes and protocol boundaries; no fixture claims GPU qualification."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import threading
import unittest
from unittest.mock import Mock,patch
import uuid
import zlib
from slstack.config import ROOT,load
from slstack.media_references import References,conditioning
from slstack.tools import media_schema,validate
from slstack.router import Stack


def png(pixel=b'\xff\0\0'):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    return (bytes.fromhex('89504e470d0a1a0a')+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))+
            chunk(b'IDAT',zlib.compress(b'\0'+pixel))+chunk(b'IEND',b''))


class MediaReferenceTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'runtime'/'fixtures'/('media-references-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.cfg=load(overrides={'runtime_root':str(self.root),'gpu_uuid':'GPU-test'})
        self.url='data:image/png;base64,'+base64.b64encode(png()).decode()

    def test_uploaded_pixels_have_stable_ids_and_no_private_paths_in_annotations(self):
        first=References(self.cfg,scoped=True);second=References(self.cfg,scoped=True)
        original=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=self.url)),dict(type='text',text='Animate this')])]
        a,b=deepcopy(original),deepcopy(original);first.annotate(a);second.annotate(b)
        self.assertEqual(a,b)
        identity=next(iter(first.paths));self.assertEqual(Path(first.resolve(identity)).read_bytes(),png())
        self.assertIn(identity,a[0]['content'][1]['text']);self.assertNotIn(str(self.root),str(a))
        self.assertEqual(len(list(first.root.glob('*.png'))),1)
        self.assertEqual(original[0]['content'][1]['text'],'Animate this')

    def test_references_rehydrate_after_restart_but_chat_ids_remain_scoped(self):
        bank=References(self.cfg);identity=bank.add(self.url)
        self.assertEqual(Path(References(self.cfg).resolve(identity)).read_bytes(),png())
        with self.assertRaisesRegex(ValueError,'not in this conversation'):References(self.cfg,scoped=True).resolve(identity)
        Path(bank.resolve(identity)).write_bytes(png(b'\0\xff\0'))
        with self.assertRaisesRegex(ValueError,'identity differs'):bank.resolve(identity)

    def test_generated_url_and_upload_of_the_same_pixels_share_identity(self):
        output=self.root/'outputs';output.mkdir();(output/'sample.png').write_bytes(png())
        bank=References(self.cfg,scoped=True)
        identity=bank.add(self.url)
        self.assertEqual(bank.add('/outputs/sample.png'),identity)
        absolute=f'http://127.0.0.1:{self.cfg["port"]}/outputs/sample.png'
        self.assertEqual(Path(bank.resolve(absolute)).read_bytes(),png())
        messages=[dict(role='assistant',content='![Generated media](/outputs/sample.png)')]
        bank.annotate(messages);self.assertIn(identity,messages[0]['content'])

    def test_remote_urls_local_file_paths_and_output_traversal_are_refused(self):
        bank=References(self.cfg);(self.root/'outputs').mkdir();(self.root/'outside.png').write_bytes(png())
        for value in ('https://example.org/image.png','http://127.0.0.1:1/outputs/sample.png',
                      'file:///private.png','/outputs/../outside.png','/outputs/%2e%2e/outside.png'):
            with self.subTest(value=value),self.assertRaises((ValueError,OSError)):bank.resolve(value)
        with self.assertRaisesRegex(ValueError,'PNG'):bank.add('data:image/png;base64,'+base64.b64encode(b'not an image').decode())

    def test_conditioning_never_drops_inputs_or_silently_changes_the_mode(self):
        source=dict(_source_image_path='source.png',_reference_image_paths=[],denoise=.3)
        row=conditioning('image',source,'anima-1.0','model.safetensors')
        self.assertEqual((row['modality'],row['image_path'],row['denoise']),('i2i','source.png',.3))
        row=conditioning('video',source,'ltx-2.3','model.safetensors')
        self.assertEqual((row['modality'],row['image_path']),('i2v','source.png'))
        refs=dict(_reference_image_paths=['first.png','second.png'])
        row=conditioning('video',refs,'minimax-h3','model')
        self.assertEqual((row['modality'],row['reference_images']),('ref2v',['first.png','second.png']))
        row=conditioning('video',refs,'bernini','model')
        self.assertEqual(row['modality'],'r2v')
        for kind,args,arch in [('image',source,'krea2-turbo'),('video',refs,'ltx-2.3'),
                               ('video',dict(modality='i2v'),'minimax-h3'),
                               ('image',{**source,'modality':'t2i'},'anima-1.0'),
                               ('image',{**source,'_reference_image_paths':['other.png']},'anima-1.0')]:
            with self.subTest(arch=arch,args=args),self.assertRaises(ValueError):conditioning(kind,args,arch,'model')

    def test_reference_only_models_can_be_discovered(self):
        from slstack.discovery import model_info
        model=self.root/'reference-only';model.mkdir()
        (model/'config.json').write_text(json.dumps({'_class_name':'WanImageToVideoPipeline'}))
        row=model_info(model,'domainshuttle')
        self.assertEqual(row['kinds'],['video']);self.assertIn('ref2v',row['modalities'])

    def test_tool_schema_accepts_reference_strings_and_rejects_invalid_strengths_and_modes(self):
        row=validate(media_schema('image'),dict(prompt='Edit this',source_image='image-'+'a'*64,denoise=.25))
        self.assertEqual(row['denoise'],.25)
        row=validate(media_schema('video'),dict(prompt='Animate',modality='ref2v',reference_images=['/outputs/a.png']))
        self.assertEqual(row['reference_images'],['/outputs/a.png'])
        for args in (dict(reference_images=[{}]),dict(reference_images=['a']*10),dict(modality='i2v'),dict(denoise=1.1)):
            with self.assertRaises(ValueError):validate(media_schema('image'),dict(prompt='Edit',**args))

    def test_uploaded_then_generated_image_reaches_sequential_chat_tools(self):
        output=self.root/'outputs';output.mkdir();(output/'result.png').write_bytes(png(b'\0\xff\0'))
        identity='image-'+hashlib.sha256(png()).hexdigest()
        calls=[('generate_image',dict(prompt='Edit',source_image=identity,denoise=.4)),
               ('generate_video',dict(prompt='Animate',source_image='/outputs/result.png',modality='i2v')),
               ('generate_video',dict(prompt='Use both',reference_images=[identity,'/outputs/result.png'],modality='ref2v'))]
        engine=Mock();engine.models=None;engine.lock=threading.RLock()
        engine.media.side_effect=[dict(data=[dict(url='/outputs/result.png')]),dict(data=[dict(url='/outputs/result.mp4')]),dict(data=[dict(url='/outputs/refs.mp4')])]
        replies=[dict(choices=[dict(message=dict(role='assistant',content='',tool_calls=[dict(id=str(i),type='function',function=dict(name=name,arguments=json.dumps(args)))]),finish_reason='tool_calls')]) for i,(name,args) in enumerate(calls)]
        replies.append(dict(choices=[dict(message=dict(role='assistant',content='Done'),finish_reason='stop')]))
        engine.chat.side_effect=replies
        stack=Stack(self.cfg,engine)
        stack.tools.preferences.select=Mock(side_effect=lambda kind,model=None:dict(arch='anima-1.0' if kind=='image' else 'minimax-h3',path='model',id='fixture'))
        result=stack.chat(dict(messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=self.url))])]))
        self.assertIn('Done',result['choices'][0]['message']['content'])
        jobs=[args.args[1] for args in engine.media.call_args_list]
        self.assertEqual([job['_conditioning']['modality'] for job in jobs],['i2i','i2v','ref2v'])
        self.assertEqual(Path(jobs[0]['_source_image_path']).read_bytes(),png())
        self.assertEqual(Path(jobs[1]['_source_image_path']).read_bytes(),png(b'\0\xff\0'))
        self.assertEqual([Path(p).read_bytes() for p in jobs[2]['_reference_image_paths']],[png(),png(b'\0\xff\0')])

    def test_streamed_tool_calls_keep_uploaded_references_and_render_video_inline(self):
        identity='image-'+hashlib.sha256(png()).hexdigest()
        args=dict(prompt='Animate the uploaded subject',source_image=identity,modality='i2v')
        engine=Mock();engine.models=None
        engine.media.return_value=dict(data=[dict(url='/outputs/video.mp4')])
        engine.chat_events.side_effect=[iter([dict(choices=[dict(index=0,delta=dict(tool_calls=[dict(index=0,id='one',type='function',function=dict(name='generate_video',arguments=json.dumps(args)))]),finish_reason='tool_calls')])]),
                                       iter([dict(choices=[dict(index=0,delta=dict(content='Ready'),finish_reason='stop')])])]
        stack=Stack(self.cfg,engine);stack.tools.preferences.select=Mock(return_value=dict(arch='minimax-h3',path='model'))
        rows=list(stack.chat_events(dict(messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=self.url))])])) )
        text=''.join(choice['delta'].get('content','') for row in rows for choice in row.get('choices',[]))
        self.assertIn('![Generated media](/outputs/video.mp4)',text)
        job=engine.media.call_args.args[1]
        self.assertEqual(job['_conditioning']['modality'],'i2v')
        self.assertEqual(Path(job['_conditioning']['image_path']).read_bytes(),png())

    def test_conditioning_is_forwarded_to_the_real_typed_studio_contract(self):
        from slstack.native_broker import vendor
        vendor()
        from slstack.studio_worker import run
        from studio.models import GenerationParams
        import os
        backend=Mock();backend.generate.return_value=Mock(model_dump=lambda **kw:dict(path=str(self.root/'outputs/result.png')))
        cfg={**self.cfg,'studio_python':'fixture-python','image_model':'fixture-model','image_arch':'anima-1.0',
             'image_support_files':{},'video_model':'fixture-model','video_arch':'minimax-h3','video_support_files':{}}
        image=References(self.cfg).resolve(self.url)
        cases=[('image','anima-1.0',dict(_source_image_path=image,modality='i2i',denoise=.4)),
               ('video','minimax-h3',dict(_source_image_path=image,modality='i2v')),
               ('video','minimax-h3',dict(_reference_image_paths=[image],modality='ref2v')),
               ('video','bernini',dict(_reference_image_paths=[image],modality='ref2v'))]
        env=dict(STUDIO_DATA_ROOT=str(self.root/'studio'),STUDIO_COSMOS_WINDOWS_PATH=str(ROOT/'vendor/engines'))
        with patch.dict(os.environ,env),patch('studio.adapter.load_backend',return_value=(backend,False,'contract fixture')),patch('studio.custom_models.comfy_support_defaults',return_value={}):
            for kind,arch,inputs in cases:
                with self.subTest(kind=kind,arch=arch,inputs=inputs):
                    run(dict(cfg={**cfg,kind+'_arch':arch},kind=kind,params=dict(prompt='Conditioned scene',negative_prompt='',width=256,height=256,steps=4,guidance=1,seed=1,**inputs)))
                    params=backend.generate.call_args.args[0]
                    self.assertIsInstance(params,GenerationParams)
                    self.assertEqual(params.modality,'r2v' if arch=='bernini' else inputs['modality'])
                    if inputs.get('_source_image_path'):self.assertEqual(Path(params.image_path).read_bytes(),png())
                    else:self.assertEqual([Path(p).read_bytes() for p in params.reference_images],[png()])
                    self.assertEqual(params.denoise,inputs.get('denoise'))
                    if arch=='minimax-h3' and params.reference_images:
                        self.assertIn('<Picture 1> / <Subject 1>',params.prompt)
                        self.assertTrue(params.prompt.endswith('non_diegetic_music: N/A'))

    def test_reference_caption_keeps_studio_picture_order_and_explicit_bindings(self):
        from slstack.media_references import h3_reference_prompt
        prompt=h3_reference_prompt('A person from <Picture 1> in the style of <Picture 2>.',2)
        self.assertLess(prompt.index('<Picture 1> /'),prompt.index('<Picture 2> /'))
        self.assertEqual(h3_reference_prompt(prompt,2),prompt)
        self.assertEqual(h3_reference_prompt('Unconditioned',0),'Unconditioned')

    def test_unsupported_conditioning_fails_before_the_engine_unloads_or_starts_a_job(self):
        from slstack.tools import Tools
        engine=Mock();tools=Tools(self.cfg,engine)
        tools.preferences.select=Mock(return_value=dict(arch='krea2-turbo',path='model'))
        bank=References(self.cfg,scoped=True);identity=bank.add(self.url)
        with self.assertRaisesRegex(ValueError,'does not support i2i'):
            tools.call('generate_image',dict(prompt='Edit',source_image=identity),references=bank)
        engine.media.assert_not_called();engine.close.assert_not_called()


if __name__=='__main__':unittest.main()
