"""Pixel-bound context and automatic encoder selection regressions."""
import base64
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from slstack.config import configure_vision
from slstack.vision import Images, image_bytes, render_tokens
from slstack.local_cache import Cache, VERSION


class VisionTests(unittest.TestCase):
    def test_embedded_images_have_native_positions_and_no_caption(self):
        images=Images({})
        images.encode=lambda raw,admit:dict(key=raw.hex(),count=3,file='embedding.vision',sha256='a'*64)
        uri='data:image/png;base64,'+base64.b64encode(b'pixels').decode()
        original=dict(messages=[dict(role='user',content=[dict(type='text',text='A'),dict(type='image_url',image_url=dict(url=uri)),dict(type='text',text='B')])])
        clean,blocks,identity=images.prepare(original,lambda:None)
        class Tokenizer:
            def rendered(self,text):return list(text.encode())
        tokens,spans=render_tokens(clean['messages'][0]['content'],Tokenizer(),blocks)
        self.assertEqual(tokens,[65,129264,129264,129264,66])
        self.assertEqual(spans[0]['position'],1)
        self.assertIsInstance(original['messages'][0]['content'],list)
        self.assertTrue(identity)
        changed=deepcopy(original);changed['messages'][0]['content'][1]['image_url']['url']='data:image/png;base64,b3RoZXI='
        self.assertNotEqual(images.prepare(changed,lambda:None)[2],identity)

    def test_data_url_validation_and_local_output_boundary(self):
        root=Path(tempfile.mkdtemp(prefix='slstack-vision-proof-'))
        (root/'outputs').mkdir();(root/'outputs'/'image.png').write_bytes(b'pixels')
        cfg=dict(runtime_root=str(root/'single-engine'),stack_preferences_file=str(root/'settings'/'stack.local.json'))
        self.assertEqual(image_bytes('/outputs/image.png',cfg),b'pixels')
        (root/'private.png').write_bytes(b'private')
        for url in ('/outputs/../private.png','https://example.com/image.png','data:image/svg+xml;base64,YQ==','data:image/png;base64,===='):
            with self.subTest(url=url),self.assertRaises((ValueError,FileNotFoundError)):image_bytes(url,cfg)

    def test_auto_encoder_uses_selected_model_and_requires_capability(self):
        root=Path(tempfile.mkdtemp(prefix='slstack-encoder-proof-'))
        encoder=root/'Model-Vision.gguf';encoder.write_bytes(b'encoder')
        cfg=dict(model=str(root/'Model.gguf'))
        configure_vision(cfg);self.assertNotIn('vision_encoder',cfg)
        cfg['vision_bridge_version']=1;configure_vision(cfg)
        self.assertEqual(cfg['vision_encoder'],str(encoder.resolve()))
        cfg.pop('vision_encoder');cfg['ds4_command']=['server','--vision',str(encoder)]
        configure_vision(cfg);self.assertEqual(cfg['vision_encoder'],str(encoder.resolve()))

    def test_checkpoint_allows_new_image_but_rejects_changed_prefix_pixels(self):
        cfg=dict(context=999424,producer_sha256='b'*64,yoco_keep=140,ctx_cache_precision='bf16')
        cache=Cache(cfg);cache._identity={k:'a'*64 for k in ('model_sha256','tokenizer_sha256','template_sha256','rope_sha256')}
        from tests.test_local_cache import LocalCacheTests
        fixture=LocalCacheTests();fixture.setUp();_,contract=fixture.payload()
        contract['local_state']=dict(version=VERSION,yoco_keep=140,cache_precision='bf16',producer_sha256='b'*64)
        saved=dict(contract=contract,images=[])
        cfg['_vision_spans']=[dict(position=10,count=3,sha256='c'*64)]
        self.assertTrue(cache.compatible(saved))
        cfg['_vision_spans']=[dict(position=1,count=1,sha256='c'*64)]
        saved['images']=[dict(position=1,count=1,sha256='c'*64)]
        self.assertTrue(cache.compatible(saved))
        cfg['_vision_spans'][0]['sha256']='d'*64
        self.assertFalse(cache.compatible(saved))


if __name__=='__main__':unittest.main()
