"""A model's loading choice is shared by image/video and reaches its worker."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from slstack.preferences import Preferences
from slstack.engines import Engines


class MediaLoadingPreferencesTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='slstack-media-modes-'))
        self.models=[dict(id='anima',name='Anima',path=str(self.root/'anima.safetensors'),arch='anima-1.0',kinds=['image']),
                     dict(id='h3',name='H3',path=str(self.root/'h3'),arch='minimax-h3',kinds=['image','video'])]
        self.catalog=dict(models=self.models,support_files={})
        self.scanner=Mock();self.scanner.state.return_value=self.catalog
        self.patch=patch('slstack.preferences.Scanner',return_value=self.scanner)
        self.factory=self.patch.start();self.addCleanup(self.patch.stop)
        self.cfg=dict(runtime_root=str(self.root),image_mode='streaming',video_mode='streaming',
                      studio_python=sys.executable,image_support_files={},video_support_files={})

    def test_per_model_choice_persists_without_changing_other_models_or_rescanning(self):
        prefs=Preferences(dict(self.cfg))
        result=prefs.update(dict(media_load_modes={self.models[0]['path']:'resident'}))
        self.assertEqual(result['media_load_modes'],dict(anima='resident',h3='streaming'))
        self.assertEqual(self.factory.call_count,1)
        reloaded=Preferences(dict(self.cfg))
        self.assertEqual(reloaded.media_mode(self.models[0]['path'],'image'),'resident')
        reloaded.update(dict(media_load_modes={self.models[1]['path']:'resident'}))
        self.assertEqual(reloaded.media_mode(self.models[1]['path'],'image'),'resident')
        self.assertEqual(reloaded.media_mode(self.models[1]['path'],'video'),'resident')
        calls=self.factory.call_count;reloaded.update({})
        self.assertEqual(self.factory.call_count,calls+1)

    def test_invalid_modes_are_rejected_without_writing_preferences(self):
        prefs=Preferences(dict(self.cfg))
        for value in ('auto',True,[], {'x':'guess'}):
            with self.subTest(value=value),self.assertRaises(ValueError):
                prefs.update(dict(media_load_modes=value))
        self.assertFalse(prefs.path.exists())

    def test_effective_mode_reaches_the_selected_worker_and_result(self):
        prefs=Preferences(dict(self.cfg))
        prefs.update(dict(media_load_modes={m['path']:'resident' for m in self.models}))
        output=self.root/'outputs'/'image.png';output.parent.mkdir();output.write_bytes(b'fixture')
        job=Mock();job.log=self.root/'worker.log'
        job.log.write_text('DS4STACK_RESULT '+json.dumps(dict(ok=True,files=[str(output)]))+'\n')
        engine=object.__new__(Engines);engine.cfg=dict(self.cfg);engine.preferences=prefs
        with patch('slstack.engines.OwnedProcess',return_value=job) as run,patch('slstack.media_profiles.python_for',return_value=sys.executable):
            for kind,ident in [('image','anima'),('image','h3'),('video','h3')]:
                result=engine._media(kind,dict(model=ident,prompt='fixture'))
                request=json.loads(run.call_args.kwargs['stdin'])
                self.assertEqual(request['cfg'][kind+'_mode'],'resident')
                self.assertEqual(result['mode'],'resident')


if __name__=='__main__':unittest.main()
