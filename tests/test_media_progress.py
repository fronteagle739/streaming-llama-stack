import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer
from slstack.config import load
from slstack.media_progress import MediaProgress, callback
from slstack.router import Stack, handler
from slstack.vision import Images, has_images


class MediaProgressTests(unittest.TestCase):
    def test_step_timing_excludes_preparation_and_duplicate_notifications(self):
        progress = MediaProgress(tempfile.mkdtemp(), 'image', 4)
        update = callback(progress.path)
        with patch('slstack.media_progress.time') as clock:
            clock.monotonic.side_effect = [1, 100, 101, 102, 104]
            clock.time.return_value = 1000
            update(0, 4)
            update(1, 4)
            self.assertIsNone(progress.snapshot()['s_iter'])
            update(1, 4)
            update(2, 4)
            self.assertEqual(progress.snapshot()['s_iter'], 2)
            update(4, 4)
        state = progress.snapshot()
        self.assertEqual((state['step'], state['total'], state['phase'], state['s_iter']), (4, 4, 'decoding', 1))
        progress.phase('restoring');self.assertTrue(progress.snapshot()['active'])
        progress.phase('complete', active=False);self.assertFalse(progress.snapshot()['active'])
        other = MediaProgress(progress.path.parents[1], 'video', 8)
        self.assertEqual(other.snapshot()['step'], 0)
        self.assertIsNone(other.snapshot()['s_iter'])
        self.assertNotEqual(other.snapshot()['id'], state['id'])

    def test_status_is_readable_while_chat_owns_engine_lock(self):
        engine = Mock();engine.models = None;engine.lock = threading.Lock()
        engine.events = [];engine.media_progress = MediaProgress(tempfile.mkdtemp(), 'image', 20)
        callback(engine.media_progress.path)(3, 20, meta={'s_iter':1.25})
        stack = Stack(load(overrides={'gpu_uuid':'GPU-test'}), engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(stack))
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        try:
            with engine.lock, urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/slstack/status', timeout=2) as reply:
                row = json.load(reply)['media_progress']
            self.assertEqual((row['step'], row['total'], row['s_iter']), (3, 20, 1.25))
            self.assertNotIn('prompt', row);self.assertNotIn('path', row)
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_image_streams_before_continuation_and_is_in_vision_tool_context(self):
        cfg = load(overrides={'gpu_uuid':'GPU-test'})
        cfg.update(vision_bridge_version=1, vision_encoder='selected-encoder.gguf')
        engine = Mock();engine.models = None
        call = dict(index=0, id='image-call', type='function', function=dict(name='generate_image', arguments='{"prompt":"A cup"}'))
        def event(delta, finish=None):return dict(choices=[dict(index=0, delta=delta, finish_reason=finish)])
        engine.chat_events.side_effect = [iter([event({'tool_calls':[call]}, 'tool_calls')]),
                                          iter([event({'content':'Here it is.'}), event({}, 'stop')])]
        stack = Stack(cfg, engine)
        stack.tools.call = Mock(return_value=dict(data=[dict(url='/outputs/image.png')], settings={}))
        source = stack.chat_events(dict(messages=[dict(role='user', content='Make a cup')]))
        first = next(source)
        self.assertIn('stack-tool',first['choices'][0]['delta']['content'])
        self.assertFalse(stack.tools.call.called)
        next(source)  # completed tool activity
        first = next(source)
        self.assertIn('![Generated media](/outputs/image.png)', first['choices'][0]['delta']['content'])
        self.assertEqual(engine.chat_events.call_count, 1)
        tool = engine.chat_events.call_args.args[0]['messages'][-1]
        self.assertEqual(tool['role'], 'tool')
        self.assertEqual(tool['content'][1], dict(type='image_url', image_url=dict(url='/outputs/image.png')))
        images = Images({})
        images.encode = lambda raw, admit:dict(key='pixels', count=3, file='native.vision', sha256='a'*64)
        with patch('slstack.vision.image_bytes', return_value=b'pixels'):
            clean, blocks, identity = images.prepare(dict(messages=[tool]), lambda:None)
        self.assertEqual(blocks[0]['count'], 3)
        self.assertIn(blocks[0]['marker'], clean['messages'][0]['content'])
        tail = list(source)
        self.assertNotIn('![Generated media]', json.dumps(tail))
        self.assertEqual(tail[0]['choices'][0]['delta']['content'], 'Here it is.')

    def test_generated_image_stays_in_context_on_the_next_browser_turn(self):
        body=dict(messages=[dict(role='assistant',content='Here it is.\n\n![Generated media](/outputs/image.png)'),
                            dict(role='user',content='Change the cup to blue')])
        self.assertTrue(has_images(body))
        images=Images({})
        images.encode=lambda raw, admit:dict(key='pixels',count=3,file='native.vision',sha256='a'*64)
        with patch('slstack.vision.image_bytes',return_value=b'pixels'):
            clean,blocks,identity=images.prepare(body,lambda:None)
        self.assertEqual(clean['messages'][0]['role'],'assistant')
        self.assertIn(blocks[0]['marker'],clean['messages'][0]['content'])
        self.assertEqual(body['messages'][0]['content'],'Here it is.\n\n![Generated media](/outputs/image.png)')
        for url in ('https://example.com/image.png','/outputs/video.mp4','/private/image.png'):
            self.assertFalse(has_images(dict(messages=[dict(role='assistant',content=f'![Generated media]({url})')])))


if __name__ == '__main__':unittest.main()
