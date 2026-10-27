"""Synthetic F1 gate; run from an isolated copy, never imports/loads a model.

Mac: python validate_sdpa.py --root ROOT --census LIB --out results.json
The source functions are AST-extracted verbatim to avoid package side effects.
"""
import argparse
import ast
import ctypes
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def load_functions(root):
    import mlx.core as mx
    path = root / 'mlx_vlm/models/deepseek_v41/language.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('_gather_kv', '_sparse_attn_chunk', 'sparse_attn')]
    env = dict(mx=mx, os=os, partial=partial, __package__='mlx_vlm.models.deepseek_v41')
    exec(compile(tree, str(path), 'exec'), env)
    name = 'mlx_vlm.custom_kernels.deepseek_v41.sparse_attention'
    spec = importlib.util.spec_from_file_location(name, root / (name.replace('.', '/') + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return env


def select(path):
    os.environ['DSV41_SDPA_SPARSE_ATTN'] = '1' if path == 'sdpa' else '0'
    os.environ['DSV41_METAL_SPARSE_ATTN'] = '1' if path == 'metal' else '0'
    os.environ['DSV41_COMPILE_SPARSE_ATTN'] = '1' if path == 'compiled' else '0'


def metrics(actual, expected):
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    norm = np.linalg.norm(e)
    return dict(max_abs=float(np.max(np.abs(a-e))),
                relative_l2=float(np.linalg.norm(a-e)/max(norm, 1e-30)),
                norm_ratio=float(np.linalg.norm(a)/norm) if norm else None)


def gate(actual, expected, tol=.012):
    m = metrics(actual, expected)
    assert np.isfinite(actual).all(), m
    assert m['max_abs'] <= 2*tol*max(1,float(np.max(np.abs(expected)))), m
    assert m['relative_l2'] <= tol, m
    if m['norm_ratio'] is not None:
        assert abs(m['norm_ratio']-1) <= tol, m
    else:
        assert m['max_abs'] == 0, m
    return m


def low_precision_sdpa(env, q, kv, sink, idx, scale):
    """Rejected experiment: rounds the learned sink and fallback intermediates."""
    import mlx.core as mx
    b,m,h,d = q.shape
    rows = env['_gather_kv'](kv,idx).reshape(b*m,1,idx.shape[-1],d)
    return mx.fast.scaled_dot_product_attention(
        q.reshape(b*m,h,1,d),rows,rows,scale=scale,
        mask=(idx>=0).reshape(b*m,1,1,idx.shape[-1]),
        sinks=sink.astype(mx.result_type(q,rows))).reshape(q.shape).astype(q.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--census', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    import mlx.core as mx
    from importlib.metadata import version
    env = load_functions(args.root)
    attend = env['sparse_attn']
    if args.worker:
        import base64
        import io
        for line in sys.stdin:
            msg = json.loads(line)
            data = np.load(io.BytesIO(base64.b64decode(msg['arrays'])))
            a = {k: mx.array(data[k], dtype=getattr(mx, v)) for k, v in msg['dtypes'].items()}
            select(msg['path'])
            kv = (a['w'], a['c']) if 'c' in a else a['w']
            out = attend(a['q'], kv, a['sink'], a['idx'], msg['scale'])
            mx.eval(out)
            buf = io.BytesIO()
            np.save(buf, np.array(out.astype(mx.float32)))
            print(json.dumps(dict(output=base64.b64encode(buf.getvalue()).decode())), flush=True)
        return
    census = ctypes.CDLL(str(args.census)) if args.census else None
    if census:
        census.s2_install.restype = ctypes.c_char_p
        census.s2_count.restype = ctypes.c_ulonglong
        encoder = census.s2_install().decode()
        assert encoder != 'unsupported'
        a = mx.ones((1024,)); mx.eval(a); mx.synchronize()
        census.s2_reset(); mx.eval(mx.exp(a)); mx.synchronize()
        assert census.s2_count() == 1
    result = dict(mlx=version('mlx'), device=mx.device_info(),
                  encoder=encoder if census else None, cases=[])
    result['forced_fusion_probes'] = []
    for h,l,d in [(64,1,512),(64,256,512),(1,64,512),(64,1,256),(1,64,256)]:
        q = mx.zeros((1,h,l,d),dtype=mx.bfloat16)
        kv = mx.zeros((1,1,640,d),dtype=q.dtype)
        try:
            mx.eval(mx.fast.scaled_dot_product_attention(q,kv,kv,scale=d**-.5,
                    sinks=mx.zeros((h,),dtype=q.dtype),force_fused=True))
            evidence = 'fused kernel available'
        except ValueError as exc:
            evidence = str(exc)
        result['forced_fusion_probes'].append(dict(h=h,l=l,d=d,result=evidence))
    rng = np.random.default_rng(41)
    for b, m, h, d, nw, nc, k in [
            (1, 1, 64, 512, 4096, 2048, 640),
            (1, 1, 64, 512, 16384, 8192, 640),
            (1, 256, 64, 512, 4096, 2048, 640),
            (1, 1, 64, 512, 4096, 0, 128),
            (1, 256, 64, 512, 4096, 0, 128),
            (2, 3, 4, 32, 19, 7, 12),
            (1, 257, 4, 32, 300, 150, 64)]:
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            q = mx.array(rng.normal(size=(b,m,h,d)).astype('float32'), dtype=dtype)
            w = mx.array(rng.normal(size=(b,nw,d)).astype('float32'), dtype=dtype)
            c = mx.array(rng.normal(size=(b,nc,d)).astype('float32'), dtype=dtype)
            sink = mx.array(np.linspace(-2,3,h).astype('float32'))
            wi = rng.integers(0,nw,size=(b,m,min(k,128)),dtype='int32')
            ci = rng.integers(nw,nw+nc,size=(b,m,k-128),dtype='int32') if k>128 else None
            idx = wi if ci is None else np.concatenate((wi,ci),axis=-1)
            idx[...,::17] = -1
            idx = mx.array(idx)
            kv = (w,c) if nc else w
            mx.eval(q,w,c,sink,idx)
            record = dict(shape=[b,m,h,d], banks=[nw,nc], selected=k, dtype=str(dtype), paths={})
            expected = None
            for path in ('manual','metal','sdpa','compiled','sdpa_lowp'):
                select(path)
                run = partial(low_precision_sdpa,env) if path == 'sdpa_lowp' else attend
                out = run(q,kv,sink,idx,d**-.5); mx.eval(out)
                a = np.array(out.astype(mx.float32))
                if expected is None: expected = a
                parity = gate(a, expected, tol=2e-5 if dtype == mx.float32 else .012)
                for _ in range(2): mx.eval(run(q,kv,sink,idx,d**-.5))
                elapsed, counts = [], []
                for _ in range(7):
                    mx.synchronize()
                    if census: census.s2_reset()
                    start = time.perf_counter()
                    mx.eval(run(q,kv,sink,idx,d**-.5)); mx.synchronize()
                    elapsed.append((time.perf_counter()-start)*1000)
                    if census: counts.append(census.s2_count())
                record['paths'][path] = dict(**parity, median_ms=float(np.median(elapsed)),
                                             samples_ms=elapsed, dispatches=counts)
            result['cases'].append(record)
            print(json.dumps(record), flush=True)
            args.out.write_text(json.dumps(result,indent=2)+'\n')
    # Norm-sensitive fault injection must fail, even though argmax is unchanged.
    try:
        gate(expected*np.sqrt(2), expected)
    except AssertionError as exc:
        result['scale_mutation_caught'] = str(exc)
    else:
        raise AssertionError('scale mutation survived')
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
