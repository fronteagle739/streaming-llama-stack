import unittest
from unittest.mock import patch, Mock

from slstack.config import load
from slstack.engines import Engines
from slstack.pairing import endpoint, conversation_key
from slstack.__main__ import main
from slstack.mac import Receiver


class PairingContractTests(unittest.TestCase):
    def test_endpoint_accepts_only_receiver_origin(self):
        self.assertEqual(endpoint('example.test'), 'http://example.test:8180')
        self.assertEqual(endpoint('http://example.test:9001/'), 'http://example.test:9001')
        for value in ['https://example.test', 'http://user:password@example.test',
                      'http://example.test/?token=private', 'http://example.test/path',
                      'http://example.test/#fragment', 'http://[::1]:8180', 'http://example.test:99999']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                endpoint(value)

    def test_second_turn_reuses_key_but_system_or_initial_prompt_changes_do_not(self):
        body = {'model': 'test', 'messages': [{'role':'system','content':'Be brief'},
                                             {'role':'user','content':'Remember amber'}]}
        first = conversation_key(body)
        continued = {**body, 'messages': body['messages'] + [
            {'role':'assistant','content':'Remembered'}, {'role':'user','content':'Which colour?'}]}
        self.assertEqual(conversation_key(continued), first)
        changed = {**body, 'messages': [{'role':'system','content':'Use rhymes'}, body['messages'][1]]}
        self.assertNotEqual(conversation_key(changed), first)
        self.assertEqual(conversation_key({**body,'prompt_cache_key':'explicit-chat-1'}), 'explicit-chat-1')
        for key in ['', None, 0, 'bad/key', 'x'*65]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                conversation_key({**body, 'prompt_cache_key':key})

    def test_mac_quality_is_unsupported_without_loading_either_engine(self):
        engine = Engines(load(overrides={'gpu_uuid':'GPU-test','mac':'http://example.test:8180'}))
        with patch.object(engine, 'ready', side_effect=AssertionError('engine loaded')):
            self.assertFalse(engine.quality()['supported'])
            self.assertFalse(engine.quality(.35)['phases_supported'])
        self.assertIsNone(engine.llm)
        self.assertIsNone(engine.remote)

    def test_receiver_forwards_same_prefix_key_on_actual_chat_path(self):
        receiver = Receiver.__new__(Receiver)
        receiver.backend, receiver.peer = 'ds4', {'status':'qualified'}
        receiver.cfg = {'request_timeout_s':1}
        receiver.ready = Mock()
        first = {'model':'test','messages':[{'role':'user','content':'Remember amber'}]}
        second = {**first,'messages':first['messages'] + [
            {'role':'assistant','content':'Remembered'}, {'role':'user','content':'Which colour?'}]}
        with patch('slstack.mac.http_json',return_value={'choices':[]}) as upstream:
            receiver.chat(first)
            initial = upstream.call_args.args[1]
            receiver.chat(second)
            continued = upstream.call_args.args[1]
        self.assertEqual(initial['prompt_cache_key'],continued['prompt_cache_key'])
        self.assertTrue(continued['include_previous_reasoning'])
        self.assertEqual(continued['messages'],second['messages'])

    def test_pair_cli_passes_receiver_and_token_without_model_journal(self):
        args = ['slstack', 'pair', '--mac','example.test','--pair-token','TEST_VALUE_ONLY_1234']
        with patch('sys.argv',args), patch('slstack.__main__.load',return_value={}) as config, \
             patch('slstack.pairing.Pair') as peer, \
             patch('slstack.__main__.Journal',side_effect=AssertionError('journal touched')):
            self.assertEqual(main(),0)
            self.assertEqual(config.call_args.args[1]['mac'],'example.test')
            self.assertEqual(config.call_args.args[1]['pair_token'],'TEST_VALUE_ONLY_1234')
            peer.return_value.start.assert_called_once()
            peer.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
