import unittest
from slstack.mac import qualification_request, qualification_request_digest


class QualificationPrompt(unittest.TestCase):
    def test_long_prompt_keeps_the_same_eight_row_gate_and_semantic_binding(self):
        messages=[dict(role='user',content='long context text\n'*50000)]
        native=qualification_request(dict(mode='native',messages=messages))
        paired=qualification_request(dict(mode='paired',messages=messages))
        self.assertEqual((native['max_tokens'],native['temperature'],native['logprobs']),(8,0,True))
        self.assertEqual(native['messages'],messages)
        self.assertEqual(qualification_request_digest(native),qualification_request_digest(paired))
        changed={**paired,'messages':[dict(role='user',content='another context')]}
        self.assertNotEqual(qualification_request_digest(native),qualification_request_digest(changed))
        self.assertNotEqual(qualification_request_digest(native),qualification_request_digest({**native,'logit_bias':{'42':100}}))

    def test_default_gate_and_unknown_or_oversized_inputs(self):
        self.assertEqual(qualification_request({'mode':'native'})['messages'][0]['content'],'Repeat exactly READY, with no explanation.')
        for row in [dict(mode='bad'),dict(mode='native',max_tokens=1),dict(mode='native',messages=[]),
                    dict(mode='native',messages=[dict(role='user',content='x',ignored=True)]),
                    dict(mode='native',messages=[dict(role='user',content='x'*(8*1024**2+1))])]:
            with self.assertRaises(ValueError):qualification_request(row)


if __name__=='__main__':unittest.main()
