import json
import unittest
from slstack.native_broker import vendor
from slstack.request_limits import chat_body_limit
vendor()
from studio.dsv41_session_contract import control


class RequestLimitTests(unittest.TestCase):
    def test_chat_accepts_image_json_while_native_controls_keep_the_small_limit(self):
        data=json.dumps({'messages':[{'role':'user','content':'a'*(2<<20)}]}).encode()
        with self.assertRaises(ValueError):control(data)
        self.assertEqual(len(control(data,max_bytes=chat_body_limit({'context':4096}))['messages'][0]['content']),2<<20)
        self.assertGreater(chat_body_limit({}),28<<20)
        self.assertEqual(chat_body_limit({'context':999424}),32<<20)
        self.assertEqual(chat_body_limit({'context':10000000}),64<<20)

    def test_larger_chat_bound_preserves_strict_json_validation(self):
        for data in (b'{"x":1,"x":2}',b'{"x":NaN}',b'[]'):
            with self.subTest(data=data),self.assertRaises(ValueError):control(data,max_bytes=32<<20)


if __name__=='__main__':unittest.main()
