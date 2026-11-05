import math
import struct
import unittest
import numpy as np
from scripts.qfn_mac_convert import dequant_nvfp4, value, quantize_q4
from scripts.dsv41_fp4_layout import pack_nvfp4_matrix


class MacConversionTests(unittest.TestCase):
    def test_all_fp4_codes_and_unsigned_scale_boundaries(self):
        # Independent scalar decoding of all codes and representative scale bins.
        for scale_code in (0,1,7,8,55,56,119,126):
            codes=list(range(16))*4
            raw=bytes([scale_code]*4)+bytes(codes[s*16+j]|codes[s*16+j+8]<<4 for s in range(4) for j in range(8))
            actual=dequant_nvfp4(raw,1,64,.125,False).reshape(-1)
            e,m=scale_code//8,scale_code%8
            scale=(m/8*2**-6 if e==0 else (1+m/8)*2**(e-7))*.125
            levels=(0,.5,1,1.5,2,3,4,6)
            expected=[(-1 if c&8 else 1)*levels[c&7]*scale for c in codes]
            np.testing.assert_array_equal(actual,np.array(expected,np.float32))
            # The CUDA-native layout requires N,K divisible by 128.
            matrix=raw*2*128
            packed=pack_nvfp4_matrix(matrix,128,128)
            np.testing.assert_array_equal(np.tile(actual,(128,2)),dequant_nvfp4(packed,128,128,.125,True))

    def test_invalid_scales_fail_instead_of_clamping_or_omitting(self):
        for code in (127,128,255):
            with self.assertRaises(ValueError):dequant_nvfp4(bytes([code]*4)+bytes(32),1,64,1,False)
        for scale in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):dequant_nvfp4(bytes([56]*4)+bytes(32),1,64,scale,False)

    def test_direct_q4_uses_original_float_values_and_signed_peak(self):
        cases=[np.zeros(32,np.float32),np.arange(-16,16,dtype=np.float32)*.125,
            np.linspace(-.0071,.0093,32,dtype=np.float32),np.array([1.,-1.]*16,np.float32)]
        for values in cases:
            peak=float(max(values,key=abs));delta=peak/-8
            inverse=1/delta if delta else 0
            codes=[min(15,math.floor(float(x)*inverse+8.5)) for x in values]
            expected=struct.pack('<e',delta)+bytes(codes[i]|codes[i+16]<<4 for i in range(16))
            self.assertEqual(quantize_q4(values),expected)
        for bad in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):quantize_q4(np.full(32,bad,np.float32))

    def test_metadata_integer_width_and_unicode_preserved(self):
        for n in (0,2**32,2**63-1,-2**40):
            raw=value(n);self.assertEqual(struct.unpack('<Q' if n>=0 else '<q',raw[4:])[0],n)
        raw=value('日本語');self.assertEqual(raw[12:].decode(),'日本語')
        with self.assertRaises(ValueError):value([1,'lost field'])


if __name__=='__main__':unittest.main()
