import json
import hashlib
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from slstack.combo_calibrate import calibrate,CalibrationPending


class CalibrationTest(unittest.TestCase):
    def test_resume_does_not_repeat_completed_pairs_or_qualify_extrapolation(self):
        root=Path(tempfile.mkdtemp(prefix='combo-calibration-'))
        cfg=dict(runtime_root=str(root),combo_identity={'model':'fixture'},native_sha256='native',
                 producer_sha256='producer',provider_schema={},native_prefill_chunk=128,context=262144)
        calls=[]
        def measured(config,body,target,route):
            words=re.search(r'-w(\d+)',target.stem)[1];calls.append((route,target.stem))
            tokens=10+int(words)
            seconds=5+tokens/10 if route=='native' else 20+tokens/200
            result=dict(status='PASS',identity=cfg['combo_identity'],
                request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                **{key:cfg[key] for key in ('native_sha256','producer_sha256','provider_schema','native_prefill_chunk')},
                turns=[dict(total_first_token_seconds=seconds,metrics=dict(prompt_tokens=tokens,prepared_seconds=0),
                            engine_milestones={'100':dict(seconds=seconds)},ultra_prefill_tok_s=200)])
            target.write_text(json.dumps(result));return result
        with patch('slstack.combo_calibrate.require_parity'),patch('slstack.combo_calibrate.measure',side_effect=measured):
            with self.assertRaises(CalibrationPending) as first:calibrate(cfg,root/'result.json')
            self.assertEqual(first.exception.words,64)
            self.assertEqual(len(calls),2)
            with self.assertRaises(CalibrationPending) as second:calibrate(cfg,root/'result.json')
            self.assertEqual(second.exception.words,256)
            self.assertEqual(len(calls),4)
            self.assertEqual(calls.count(('native','crossover-native-w0000')),1)
            result=calibrate(cfg,root/'result.json',pair_budget=20)
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['bracket_verified'])
        self.assertTrue(result['repeated_bracket_verified'])
        self.assertLessEqual(result['bracket'][1]-result['bracket'][0],32)
        self.assertEqual(len(calls),len(set(calls)))


if __name__=='__main__':unittest.main()
