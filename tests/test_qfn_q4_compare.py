import math,struct,unittest
from scripts.qfn_mac_q4_compare import q4

class Q4ComparisonTests(unittest.TestCase):
    def test_zero_blocks_and_signed_peak_scalar_reference(self):
        self.assertEqual(q4(struct.pack('<e',.25)+bytes(32)),struct.pack('<e',-0.)+bytes([0x88]*16))
        values=list(range(-16,16));scale=.25
        raw=struct.pack('<e',scale)+struct.pack('<32b',*values)
        decoded=[v*scale for v in values];peak=max(decoded,key=abs);delta=peak/-8
        codes=[min(15,math.floor(v/delta+8.5)) for v in decoded]
        expected=struct.pack('<e',delta)+bytes(codes[i]|codes[i+16]<<4 for i in range(16))
        self.assertEqual(q4(raw),expected)

    def test_multiple_blocks_keep_boundaries(self):
        a=struct.pack('<e',.5)+struct.pack('<32b',*range(-16,16))
        b=struct.pack('<e',.125)+struct.pack('<32b',*range(15,-17,-1))
        self.assertEqual(q4(a+b),q4(a)+q4(b))
        with self.assertRaises(ValueError):q4(a[:-1])
