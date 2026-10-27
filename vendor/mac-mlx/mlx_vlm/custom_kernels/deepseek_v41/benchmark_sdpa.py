"""Rotated-order synthetic benchmark; async batches amortize host synchronization.

No model imports/weights. Counts are Metal encoder dispatches, not Python ops.
These are component service times under a shared GPU, not token latency.
Build --census using scripts/dsv41_s2_dispatch_count.m from this worktree.
"""
import argparse
import ctypes
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from validate_sdpa import load_functions, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--census',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache-limit-mib',type=int,
                        help='Optional allocator stress experiment; default leaves MLX unchanged')
    args=parser.parse_args()
    f=load_functions(args.root)['sparse_attn']
    census=ctypes.CDLL(str(args.census))
    census.s2_install.restype=ctypes.c_char_p
    census.s2_count.restype=ctypes.c_ulonglong
    encoder=census.s2_install().decode()
    assert encoder != 'unsupported'
    a=mx.ones((1024,)); mx.eval(a); mx.synchronize()
    census.s2_reset(); mx.eval(mx.exp(a)); mx.synchronize()
    assert census.s2_count() == 1
    if args.cache_limit_mib is not None:
        mx.set_cache_limit(args.cache_limit_mib*1024*1024)
    rng=np.random.default_rng(5)
    result=dict(method='9 rotated rounds; 10 async calls/sample decode, 2 prefill; '
                'synchronize once per sample, time divided by call count. '
                'Component service time, not single-token latency.',encoder=encoder,
                cache_limit_mib=args.cache_limit_mib,cases=[])
    for m,nw,nc,k in [(1,4096,2048,640),(1,16384,8192,640),(1,4096,0,128),
                      (256,4096,2048,640),(256,16384,8192,640),(256,4096,0,128)]:
        q=mx.array(rng.normal(size=(1,m,64,512)).astype('float32'),dtype=mx.bfloat16)
        w=mx.array(rng.normal(size=(1,nw,512)).astype('float32'),dtype=q.dtype)
        c=mx.array(rng.normal(size=(1,nc,512)).astype('float32'),dtype=q.dtype)
        wi=rng.integers(0,nw,(1,m,128),dtype='int32')
        ci=rng.integers(nw,nw+nc,(1,m,512),dtype='int32') if nc else None
        idx=mx.array(np.concatenate((wi,ci),axis=-1) if nc else wi)
        sink=mx.linspace(-2,3,64); kv=(w,c) if nc else w
        mx.eval(q,w,c,idx,sink)
        paths=['manual','metal','sdpa','compiled']
        samples={p:[] for p in paths}; counts={p:[] for p in paths}
        for p in paths:
            select(p)
            for _ in range(3): mx.eval(f(q,kv,sink,idx,512**-.5))
        for rep in range(9):
            for p in paths[rep%4:]+paths[:rep%4]:
                select(p); mx.synchronize(); census.s2_reset()
                start=time.perf_counter(); n=10 if m==1 else 2
                for _ in range(n): mx.async_eval(f(q,kv,sink,idx,512**-.5))
                mx.synchronize()
                samples[p].append((time.perf_counter()-start)*1000/n)
                counts[p].append(census.s2_count()/n)
        record=dict(shape=[1,m,64,512],banks=[nw,nc],selected=k,
                    paths={p:dict(median_ms=float(np.median(samples[p])),samples_ms=samples[p],
                                  dispatches_per_call=counts[p]) for p in paths})
        result['cases'].append(record)
        print(m,nw,nc,{p:round(record['paths'][p]['median_ms'],4) for p in paths},flush=True)
    result['peak_memory_bytes']=mx.get_peak_memory()
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
