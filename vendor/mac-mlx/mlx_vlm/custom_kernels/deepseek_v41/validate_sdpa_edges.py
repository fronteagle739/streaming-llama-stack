"""Mask, sink, dtype, batch/stride and dispatch-selection gates on actual Metal."""
import argparse
import json
import os
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

from validate_sdpa import gate, load_functions, low_precision_sdpa, metrics, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    env = load_functions(args.root)
    run = env['sparse_attn']
    result = dict(cases=[], sink_rounding=[], mutations={})
    rng = np.random.default_rng(113)
    for dtype in (mx.bfloat16,mx.float16,mx.float32):
        for ratio in (1,2):
            # Each query crosses a real window/compression visibility boundary.
            pos = np.array([0,1,2,3,126,127,128,129])
            b,m,h,d,nw,nc = 2,len(pos),64,512,130,130//ratio
            q = mx.array(rng.normal(size=(b,m*2,h,d*2)).astype('float32'),dtype=dtype)[:,::2,:,::2]
            w = mx.array(rng.normal(size=(b,nw,d)).astype('float32'),dtype=dtype)
            c = mx.array(rng.normal(size=(b,nc,d)).astype('float32'),dtype=dtype)
            wi = np.maximum(pos-127,0)[:,None]+np.arange(128)[None]
            wi = np.where(wi<=pos[:,None],wi,-1)
            ci = np.broadcast_to(np.arange(nc)[None],(m,nc))
            ci = np.where(ci<(pos[:,None]+1)//ratio,ci+nw,-1)
            indices = np.concatenate((wi,ci),axis=-1)
            # Slot order is arbitrary, not a causal timeline. Preserve duplicates.
            indices = indices[:,rng.permutation(indices.shape[-1])]
            indices = np.broadcast_to(indices[None],(b,*indices.shape)).copy()
            indices[1,0] = -1
            indices[0,2,0:2] = 0
            idx = mx.array(indices,dtype=mx.int32)
            sink = mx.array(np.linspace(-80,80,h).astype('float32'))
            # Also exercise mixed source-bank precision; fp32 sinks remain exact.
            for bank_dtype in (dtype,mx.float32):
                banks = (w,c.astype(bank_dtype))
                select('manual'); expected = np.array(run(q,banks,sink,idx,d**-.5).astype(mx.float32))
                for path in ('sdpa','metal'):
                    select(path)
                    actual = np.array(run(q,banks,sink,idx,d**-.5).astype(mx.float32))
                    assert np.count_nonzero(actual[1,0]) == 0
                    record = dict(dtype=str(dtype),compressed_dtype=str(bank_dtype),ratio=ratio,path=path,
                                  **gate(actual,expected,2e-5 if dtype==mx.float32 else .004))
                    result['cases'].append(record)
                # Deliberately bypass the production mask; same selected rows.
                select('sdpa')
                original = mx.fast.scaled_dot_product_attention
                def broken_mask(*a,**kw):
                    kw['mask'] = None
                    return original(*a,**kw)
                mx.fast.scaled_dot_product_attention = broken_mask
                try:
                    broken = np.array(run(q,banks,sink,idx,d**-.5).astype(mx.float32))
                    try:
                        gate(broken,expected,.004)
                    except AssertionError as exc:
                        result['mutations']['removed_mask'] = str(exc)
                    else:
                        raise AssertionError('removed mask survived')
                finally:
                    mx.fast.scaled_dot_product_attention = original
    # Sink-dominated fixture: rounding a learned sink changes the output norm.
    for s in (8.03,32.12,80.24):
        q = mx.zeros((1,1,64,512),dtype=mx.bfloat16).at[...,0].add(s)
        kv = mx.ones((1,1,512),dtype=q.dtype)
        idx = mx.zeros((1,1,1),dtype=mx.int32)
        sink = mx.full((64,),s,dtype=mx.float32)
        select('manual'); expected = np.array(run(q,kv,sink,idx,1.).astype(mx.float32))
        select('sdpa'); actual = np.array(run(q,kv,sink,idx,1.).astype(mx.float32))
        accepted = gate(actual,expected,1e-6)
        low = np.array(low_precision_sdpa(env,q,kv,sink,idx,1.).astype(mx.float32))
        try:
            gate(low,expected,.012)
        except AssertionError as exc:
            caught = str(exc)
        else:
            raise AssertionError('sink rounding survived norm gate')
        result['sink_rounding'].append(dict(sink=s,fp32=accepted,rejected_lowp=metrics(low,expected),caught=caught))
    # Invocation gates prove flags choose their regime, before chunking.
    native = sys.modules['mlx_vlm.custom_kernels.deepseek_v41.sparse_attention']
    orig_sdpa = mx.fast.scaled_dot_product_attention
    orig_metal = native.sparse_attention
    counts = dict(sdpa=0,metal=0)
    def counted_sdpa(*a,**kw):
        counts['sdpa'] += 1
        return orig_sdpa(*a,**kw)
    def counted_metal(*a,**kw):
        counts['metal'] += 1
        return orig_metal(*a,**kw)
    mx.fast.scaled_dot_product_attention = counted_sdpa
    native.sparse_attention = counted_metal
    result['dispatch_selection'] = []
    try:
        for sdpa,metal in [('0','0'),('1','1'),('decode','prefill'),('prefill','decode')]:
            for m in (1,3,257):
                os.environ['DSV41_SDPA_SPARSE_ATTN'] = sdpa
                os.environ['DSV41_METAL_SPARSE_ATTN'] = metal
                counts.update(sdpa=0,metal=0)
                qq=mx.broadcast_to(q,(1,m,64,512))
                ii=mx.broadcast_to(idx,(1,m,1))
                mx.eval(run(qq,kv,sink,ii,1.))
                regime='decode' if m==1 else 'prefill'
                n=(m+255)//256
                expected_counts=dict(sdpa=n if sdpa in ('1',regime) else 0,
                    metal=n if sdpa not in ('1',regime) and metal in ('1',regime) else 0)
                assert counts == expected_counts,(sdpa,metal,m,counts)
                result['dispatch_selection'].append(dict(sdpa=sdpa,metal=metal,m=m,calls=dict(counts)))
    finally:
        mx.fast.scaled_dot_product_attention = orig_sdpa
        native.sparse_attention = orig_metal
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',len(result['cases']),'numerical edge cases; mask, sink, norm and regime gates',flush=True)


if __name__ == '__main__':
    main()
