"""Bounded nonzero execution probes of all six compiled entry points."""
import importlib.util
import json
from pathlib import Path
import sys
import mlx.core as mx
from smoke import check

path=next(Path(sys.argv[1]).glob('_ext*.so'))
spec=importlib.util.spec_from_file_location('_ext',path)
ext=importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
result=check(ext)
result['constant_contracts']={}
for dtype in (mx.bfloat16,mx.float16):
    sparse=ext.glm_dsa_sparse_mla_attention(
        mx.zeros((1,64,2,512),dtype),mx.zeros((1,64,2,64),dtype),
        mx.ones((1,1,256,512),dtype),mx.zeros((1,1,256,64),dtype),
        mx.broadcast_to(mx.arange(256,dtype=mx.uint32),(1,1,2,256)),1.0)
    exact=ext.glm_dsa_exact_block_attention(
        mx.zeros((1,1,16,256),dtype),mx.zeros((1,1,16,256),dtype),
        mx.ones((1,1,16,256),dtype),mx.ones((1,1,1,2),mx.bool_),
        mx.full((1,1,16,2),255,mx.uint32),1.0)
    moe=ext.glm_moe_weighted_sum(mx.ones((8,1,64),dtype),
        mx.arange(8,dtype=mx.uint32),mx.full((1,8),1/8,mx.float32))
    q8=ext.glm_dsa_q8_vup_flat(mx.ones((1,64,1,512),dtype),
        mx.zeros((64,256,128),mx.uint32),mx.zeros((64,256,8),dtype),
        mx.ones((64,256,8),dtype))
    mx.eval(sparse,exact,moe,q8)
    for name,out,value in [('sparse_mla',sparse,1),('exact_attention',exact,1),
                           ('moe',moe,1),('q8',q8,512)]:
        diff=float(mx.max(mx.abs(out.astype(mx.float32)-value)))
        if diff>0.008: raise RuntimeError(f'{name} wrong values/norm: {diff}')
        result['constant_contracts'][str(dtype)+'.'+name]=dict(shape=out.shape,max_abs=diff)
print(json.dumps(result,indent=2))
