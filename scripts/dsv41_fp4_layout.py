"""Lossless byte layouts for E2M1/E8M0 expert tensors; no float conversion."""
from __future__ import annotations

CANONICAL='ggml-mxfp4'
FLASHINFER='flashinfer-sm120-mxfp8-mxfp4-v1'
LAYOUTS=(CANONICAL,FLASHINFER)
KEY='deepseek41.expert_layout'

def validate_layout(name):
    # A b12x layout must be added with its pinned consuming kernel and an
    # independent round-trip oracle. An unqualified name never falls back.
    if name not in LAYOUTS:raise ValueError('Unsupported expert layout '+str(name))
    return name

def shape(n,k):
    if type(n) is not int or type(k) is not int or n<=0 or n%128 or k<=0 or k%128:
        raise ValueError('Native FP4 requires positive N,K divisible by128')

def pack_expert(canonical,n,k):
    """One canonical expert -> separate kernel values/scales (128B aligned).

    Tensor assembly concatenates ALL experts' values, then ALL their scales.
    Scales pack four K blocks together, with N the next stride. There is no
    additional scale swizzle or weight pre-shuffle for this FlashInfer kernel.
    """
    import numpy as np
    shape(n,k)
    if len(canonical)!=n*k//32*17:raise ValueError('Canonical expert byte count differs')
    blocks=np.frombuffer(canonical,dtype=np.uint8).reshape(n,k//32,17)
    if np.any(blocks[:,:,0]==255):raise ValueError('Nonfinite E8M0 scale')
    codes=blocks[:,:,1:]
    adjacent=np.empty_like(codes)
    adjacent[:,:,:8]=(codes[:,:,0::2]&15)|((codes[:,:,1::2]&15)<<4)
    adjacent[:,:,8:]=(codes[:,:,0::2]>>4)|(codes[:,:,1::2]&240)
    scales=blocks[:,:,0].reshape(n,k//128,4).transpose(1,0,2).copy()
    return adjacent.tobytes(),scales.tobytes()

def unpack_expert(values,scales,n,k):
    """Independent inverse, including signed-zero codes and unchanged scales."""
    import numpy as np
    shape(n,k)
    if len(values)!=n*k//2 or len(scales)!=n*k//32:raise ValueError('Native expert byte count differs')
    a=np.frombuffer(values,dtype=np.uint8).reshape(n,k//32,16)
    codes=np.empty((n,k//32,32),dtype=np.uint8)
    codes[:,:,0::2]=a&15;codes[:,:,1::2]=a>>4
    result=np.empty((n,k//32,17),dtype=np.uint8)
    result[:,:,0]=np.frombuffer(scales,dtype=np.uint8).reshape(k//128,n,4).transpose(1,0,2).reshape(n,k//32)
    result[:,:,1:]=codes[:,:,:16]|(codes[:,:,16:]<<4)
    return result.tobytes()

def requant_chunks(source,old,tensor,convert,threads,idle,input_hash):
    """Read one Q4 expert at a time; retain only the tensor's bounded scales."""
    k,n,experts=tensor['shape']
    shape(n,k)
    if experts!=384 or old['type']!=12 or tensor['type']!=39:
        raise ValueError('Expected384 Q4_K experts -> native FP4')
    scales=bytearray(experts*n*k//32)
    expert_bytes=n*k//256*144
    for expert in range(experts):
        idle();raw=source.read(expert_bytes)
        if len(raw)!=expert_bytes:raise ValueError('Short Q4 expert read')
        input_hash.update(raw)
        values,sf=pack_expert(convert(raw,threads),n,k)
        begin=expert*len(sf);scales[begin:begin+len(sf)]=sf
        yield values
    for begin in range(0,len(scales),16<<20):
        idle();yield memoryview(scales)[begin:begin+(16<<20)]
