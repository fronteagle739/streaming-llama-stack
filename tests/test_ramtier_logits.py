import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('ramtier_logits',Path(__file__).with_name('ramtier_compare_logits.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class LogitsOracle(unittest.TestCase):
    def test_forced_comparison_refuses_different_inputs_before_reading_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.mkdir();b.mkdir()
            for p,digest in ((a,'a'*64),(b,'b'*64)):
                (p/'result.json').write_text(json.dumps(dict(status='PASS',prompt_tokens=49553,token_sha256=digest)))
            with self.assertRaisesRegex(ValueError,'identity'):module.compare_forced(a,b)

    def test_distribution_shift_and_rounding(self):
        p=np.array([1.,2.,3.])
        self.assertEqual(module.distribution_error(p,p+17)['kl'],0)
        rounded=module.distribution_error(p,p+np.array([1e-5,0,-1e-5]))
        self.assertEqual(rounded['top1_equal'],1)
        self.assertLess(rounded['kl'],1e-9)
        self.assertGreater(module.distribution_error(p,-p)['kl'],.5)

    def test_trace_refuses_partial_nonfinite_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.bin'
            good=struct.pack('<IIIff',32,7,2,1.,2.)
            p.write_bytes(good)
            self.assertEqual(module.read_trace(p)[32][0],7)
            for bad in (good[:-1],good[:5],good+good,struct.pack('<IIIf',32,7,1,float('nan'))):
                p.write_bytes(bad)
                with self.assertRaises(ValueError):module.read_trace(p)


if __name__=='__main__':unittest.main()
