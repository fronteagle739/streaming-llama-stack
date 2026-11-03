"""Lossless packed NVFP4 against separate logical-value decoders."""
import struct

import numpy as np
import pytest

from scripts.dsv41_fp4_layout import pack_nvfp4_matrix,unpack_nvfp4_matrix,pack_nvfp4_blob


def ue4m3(code):
    code=np.asarray(code,dtype=np.uint8)
    e=(code>>3).astype(np.int32)
    m=(code&7).astype(np.float64)
    return np.where(e==0,np.ldexp(m,-9),np.ldexp(1+m/8,e-7))


def decode_codes(codes):
    return np.asarray([0,.5,1,1.5,2,3,4,6])[codes&7]*np.where(codes&8,-1.,1.)


@pytest.mark.parametrize('n,k',[(128,128),(640,2560),(2560,640)])
def test_independent_logical_values_and_byte_inverse(n,k):
    rng=np.random.default_rng(731)
    source=rng.integers(0,256,(n,k//64,36),dtype=np.uint8)
    source[:,:,:4]%=127
    original_codes=source[:,:,4:].reshape(n,k//64,4,8)
    original_values=np.concatenate((decode_codes(original_codes&15),decode_codes(original_codes>>4)),axis=3)
    original_values=(original_values*ue4m3(source[:,:,:4])[:,:,:,None]).reshape(n,k)
    packed=pack_nvfp4_matrix(source.tobytes(),n,k)
    adjacent=np.frombuffer(packed,dtype=np.uint8,count=n*k//2).reshape(n,k//2)
    codes=np.empty((n,k),dtype=np.uint8)
    codes[:,::2]=adjacent&15;codes[:,1::2]=adjacent>>4
    scales=np.frombuffer(packed,dtype=np.uint8,offset=n*k//2).reshape(k//64,n,4)
    logical_scales=np.empty((n,k//16),dtype=np.uint8)
    for block in range(k//16):logical_scales[:,block]=scales[block//4,:,block%4]
    actual=decode_codes(codes)*np.repeat(ue4m3(logical_scales),16,axis=1)
    np.testing.assert_array_equal(actual,original_values)
    assert unpack_nvfp4_matrix(packed,n,k)==source.tobytes()


def test_full_blob_preserves_every_tail_bit_and_all_scale_bytes():
    # Includes all256 byte patterns, signed zeros, unequal projection globals.
    raw=bytes(range(256))*(2764800//256)+struct.pack('<4f',.03125,.017,.0001,-0.)
    packed=pack_nvfp4_blob(raw)
    assert len(packed)==len(raw)
    assert packed[-16:]==raw[-16:]
    assert pack_nvfp4_blob(packed,inverse=True)==raw


def test_bad_geometry_and_lengths_fail():
    with pytest.raises(ValueError):pack_nvfp4_matrix(bytes(36),640,640)
    with pytest.raises(ValueError):pack_nvfp4_matrix(bytes(36),127,128)
    with pytest.raises(ValueError):unpack_nvfp4_matrix(bytes(36),128,128)
    with pytest.raises(ValueError):pack_nvfp4_blob(bytes(2764800))
