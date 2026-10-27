"""Exercise the real paired request boundary, including a greedy-loss control."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from slstack.generation import defaults
from slstack.pairing import Pair
from slstack.mac import Receiver


class PairedGenerationTests(unittest.TestCase):
    def body(self):
        return dict(messages=[dict(role='user', content='Remember the image')],
                    temperature=.83, top_p=.87, top_k=37, min_p=.06,
                    presence_penalty=.14, frequency_penalty=.27,
                    repeat_penalty=1.12, repeat_last_n=513,
                    seed=123456, max_tokens=79, quality_speed=.22,
                    logit_bias={'42': -.3}, stop=['END'])

    def assert_samplers(self, expected, actual):
        for key, value in expected.items():
            if key != 'messages':
                self.assertEqual(actual[key], value, key)

    def pair(self):
        pair = Pair.__new__(Pair)
        pair.cfg = dict(request_timeout_s=60)
        pair.url = 'http://127.0.0.1:8180'
        pair.token = 'test-only-placeholder'
        pair.start = Mock()
        pair.request = Mock(return_value={})
        pair.generation = Mock()
        profile = defaults()
        profile.update(per_phase=False, value=.22)
        profile['sampler'].update(temperature=.71, top_p=.89, top_k=29, min_p=.04)
        pair.generation.get.return_value = profile
        return pair

    def test_nondefault_request_reaches_mac_and_negative_greedy_control(self):
        pair = self.pair()
        body = self.body()
        original = deepcopy(body)
        pair.chat(body)
        sent = pair.request.call_args.args[1]
        self.assert_samplers(body, sent)
        self.assertEqual(sent['slstack_generation'], pair.generation.get())
        self.assertEqual(body, original)
        greedy = dict(sent, temperature=0, top_p=1, quality_speed=0)
        with self.assertRaises(AssertionError):
            self.assert_samplers(body, greedy)

    def test_stream_uses_same_request_contract(self):
        pair = self.pair()
        body = self.body()
        with patch('slstack.streaming.events', return_value=iter([{'done': True}])) as events:
            self.assertEqual(list(pair.events(body)), [{'done': True}])
        sent = events.call_args.args[1]
        self.assert_samplers(body, sent)
        self.assertEqual(sent['slstack_generation'], pair.generation.get())
        self.assertTrue(sent['stream'])

    def test_receiver_preserves_all_explicit_settings_to_actual_decoder(self):
        receiver = Receiver.__new__(Receiver)
        receiver.backend = 'ds4'
        receiver.peer = dict(status='qualified')
        receiver.cfg = dict(mac_engine_port=8181)
        receiver.ready = Mock()
        body = {k:v for k,v in self.body().items() if k in
                {'messages','temperature','top_p','top_k','min_p','seed','max_tokens','stop'}}
        body['slstack_generation'] = self.pair().generation.get()
        body['slstack_generation']['value'] = 0
        with patch('slstack.mac.http_json', return_value={}) as http:
            receiver.chat(body)
        self.assert_samplers(body, http.call_args.args[1])
        self.assertEqual(http.call_args.args[0], 'http://127.0.0.1:8181/v1/chat/completions')

    def test_toolbar_defaults_reach_decoder_and_explicit_request_wins(self):
        from slstack.paired_generation import decode_request
        profile = self.pair().generation.get()
        profile['value'] = 0
        sent = decode_request(dict(messages=[], slstack_generation=profile, top_p=.77))
        self.assertEqual(sent['temperature'], .71)
        self.assertEqual(sent['top_p'], .77)
        self.assertEqual(sent['top_k'], 29)
        self.assertEqual(sent['min_p'], .04)

    def test_unsupported_native_knobs_fail_before_engine_launch(self):
        receiver = Receiver.__new__(Receiver)
        receiver.backend = 'ds4'
        receiver.peer = dict(status='qualified')
        receiver.ready = Mock()
        knobs = dict(presence_penalty=.14, frequency_penalty=.27, repeat_penalty=1.12,
                     typical_p=.8, logit_bias={'42': -.3}, samplers=['top_k','temperature'],
                     quality_speed=.22, dynatemp_range=.1, mirostat=2, dry_multiplier=.5)
        for key, value in knobs.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                receiver.chat(dict(messages=[dict(role='user',content='q')], **{key:value}))
        receiver.ready.assert_not_called()

    def test_phase_and_bias_profiles_cannot_silently_fall_back(self):
        from slstack.paired_generation import decode_request
        with self.assertRaisesRegex(ValueError, 'phase-specific'):
            decode_request(dict(slstack_generation=defaults()))
        profile = self.pair().generation.get()
        with self.assertRaisesRegex(ValueError, 'quality/speed bias'):
            decode_request(dict(slstack_generation=profile))

    def test_single_engine_still_sends_nondefault_samplers_to_native_generation(self):
        from test_combo_single import FakeProducer, Harness, ids
        from slstack.combo_single import EOS_ID
        captured = []
        def generation(producer, row):
            captured.append(deepcopy(row))
            return dict(applies='next token boundary')
        with patch.object(FakeProducer, 'generation', generation, create=True):
            h = Harness([])
            h.engine.generation.update(dict(per_phase=False,value=.22))
            h.scripts.append(ids(h.tok, '</think>answer') + [EOS_ID])
            body = {k:v for k,v in self.body().items() if k in
                    {'messages','temperature','top_p','top_k','min_p','seed','max_tokens','stop'}}
            h.run(body)
            expected = {k:body[k] for k in ('temperature','top_p','top_k','min_p')}
            self.assertEqual(captured[0]['overrides'],expected)
            self.assertEqual(captured[0]['seed'],123456)
            self.assertEqual(h.engine.generation.get()['value'],.22)
            h.engine.retire('CPU fixture complete')


if __name__ == '__main__':
    unittest.main()
