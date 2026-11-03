"""Off-tree Basalt attention candidate: independent native-F16 CPU oracle.

GPU dispatch is coordinator-only. Complete outputs, holes/repeats/causal masking,
explicit padded strides, optional gate, changed pinned capture inputs,100Kpool.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1].parent
sys.path.insert(0,str(HERE))
import parity as base
import numpy as np
P,I,U64=base.P,base.I,base.U64
D,H,HK=256,24,2

class Desc(C.Structure):
    _pack_=8
    _fields_=[(n,I) for n in ('mode','nt','n_heads','n_kv_heads','n_kv','cap')]+[('scale',C.c_float)]+[(n,U64) for n in (
        'q_head_stride','q_token_stride','gate_head_stride','gate_token_stride','k_head_stride','k_cell_stride',
        'v_head_stride','v_cell_stride','ids_token_stride','out_head_stride','out_token_stride')]

def desc(nt,nkv,cap,mode,scale=.0625):
    return Desc(mode,nt,H,HK,nkv,cap,scale,1040,H*1040+16,1040,H*1040+32,
                528,HK*528+32,544,HK*544+16,(cap+7)*4,1032,H*1032+16)

def span(d,kind):
    if kind in ('q','gate','out'):
        return (d.nt-1)*getattr(d,kind+'_token_stride')+(H-1)*getattr(d,kind+'_head_stride')+D*4
    if kind in ('k','v'):
        return 0 if not d.n_kv else (d.n_kv-1)*getattr(d,kind+'_cell_stride')+(HK-1)*getattr(d,kind+'_head_stride')+D*2
    return (d.nt-1)*d.ids_token_stride+d.cap*4

def view(raw,d,kind):
    if kind in ('q','gate','out'):
        return np.ndarray((d.nt,H,D),np.float32,buffer=raw,strides=(getattr(d,kind+'_token_stride'),getattr(d,kind+'_head_stride'),4))
    if kind in ('k','v'):
        return np.ndarray((d.n_kv,HK,D),np.float16,buffer=raw,strides=(getattr(d,kind+'_cell_stride'),getattr(d,kind+'_head_stride'),2))
    return np.ndarray((d.nt,d.cap),np.int32,buffer=raw,strides=(d.ids_token_stride,4))

def oracle(q,k,v,ids,pos,scale,gate=None):
    out=np.zeros(q.shape,np.float64)
    for t in range(len(q)):
        rows=ids[t];rows=rows[(rows>=0)&(rows<len(k))&(rows<=pos[t])]
        if not len(rows):continue
        for hk in range(HK):
            heads=slice(hk*12,(hk+1)*12)
            scores=q[t,heads].astype(np.float64)@k[rows,hk].astype(np.float64).T*float(np.float32(scale))
            scores-=np.max(scores,axis=1,keepdims=True)
            weights=np.exp(scores);weights/=np.sum(weights,axis=1,keepdims=True)
            out[t,heads]=weights@v[rows,hk].astype(np.float64)
    if gate is not None:
        # Stable double sigmoid is an independent mathematical oracle; GPU uses
        # current native F32 expf/divide expression, so no bitidentity promise.
        g=gate.astype(np.float64);s=np.empty_like(g);nonneg=g>=0
        s[nonneg]=1/(1+np.exp(-g[nonneg]));eg=np.exp(g[~nonneg]);s[~nonneg]=eg/(1+eg)
        out*=s
    return out

def cpu_checks():
    assert C.sizeof(Desc)==120 and Desc.q_head_stride.offset==32
    q=np.ones((2,H,D),np.float32);k=np.zeros((4,HK,D),np.float16)
    v=np.broadcast_to(np.arange(4,dtype=np.float16)[:,None,None],(4,HK,D)).copy()
    ids=np.array([[-1,2,0,2,9,3],[3,2,1,0,-1,9]],np.int32);pos=np.array([2,-1],np.int32)
    y=oracle(q,k,v,ids,pos,.0625)
    assert np.all(y[0]==4/3) and np.count_nonzero(y[1])==0,'Holes, repeats, future/OOB masking'
    g=np.zeros_like(q);assert np.array_equal(oracle(q,k,v,ids,pos,.0625,g),y*.5)
    empty=oracle(q,k,v,ids[:,:0],pos,.0625);assert np.count_nonzero(empty)==0
    d=desc(17,100003,2051,1);assert (d.cap+63)//64*H*(D+2)*4+256==817600
    assert span(d,'k')==100002*d.k_cell_stride+(HK-1)*d.k_head_stride+D*2
    d=desc(2,67,65,1) # host layout proof needs no100Kpool allocation
    for kind in ('q','gate','k','v','ids','out'):
        a=np.zeros(max(1,span(d,kind)),np.uint8);x=view(a,d,kind);x[...] = 1
        assert np.all(x==1)
    return dict(cpu_oracle='PASS: independent GQA softmax, all-entry mask/repeats/gate/zero',abi_size=120,
        decode_scratch_bytes_per_query_cap2048=792832,decode_scratch_bytes_per_query_cap2051=817600,gpu='NOT RUN')

def bind_api(lib):
    return (base.bind(lib,'qfn_basalt_attn_abi',[]),base.bind(lib,'qfn_basalt_attn_prepare',[P]),
        base.bind(lib,'qfn_basalt_attn_workspace_bytes',[C.POINTER(Desc)],U64),
        base.bind(lib,'qfn_basalt_attn_run',[P,P,P,P,P,P,P,C.POINTER(Desc),P,U64,P]))

def abi_checks():
    import os
    home=Path(os.environ.get('CUDA_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'))
    directories=[os.add_dll_directory(str(p)) for p in (home/'bin',home/'bin/x64') if p.is_dir()]
    try:
        path=HERE.parent.parent/'bin/qfn-basalt-nvfp4.dll';before=hashlib.sha256(path.read_bytes()).hexdigest()
        lib=C.CDLL(str(path));abi,_,query,run=bind_api(lib);assert abi()==1
        for mode in (0,1):
            for cap in (0,1,31,32,33,63,64,65,2048,2051):
                d=desc(4,100003,cap,mode);actual=query(C.byref(d))
                expected=0 if mode==0 or cap==0 else 4*4*(((cap+63)//64)*H*(D+2)+64)
                assert actual==expected,(mode,cap,actual,expected)
        for field,value in (('mode',2),('nt',0),('n_heads',23),('n_kv_heads',0),('n_kv',-1),('cap',-1),
            ('scale',float('nan')),('q_head_stride',1025),('out_head_stride',1000),('out_token_stride',16),('ids_token_stride',3),('k_cell_stride',2**64-16)):
            d=desc(4,100003,2051,1);setattr(d,field,value);assert query(C.byref(d))==2**64-1,(field,value)
        d=desc(1,67,65,0)
        assert run(P(257),None,P(256),P(256),P(256),P(256),P(256),C.byref(d),None,0,None)!=0
        assert run(P(256),None,P(256),P(256),P(256),P(256),P(257),C.byref(d),None,0,None)!=0
        d.q_head_stride=0;d.q_token_stride=0;assert query(C.byref(d))==0,'Read broadcasts supported'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        return dict(host_only_abi_query='PASS',dll_sha256=before,no_cuda_context_functions_called=True)
    finally:
        for directory in directories:directory.close()

def gpu_case(c,run,query,nt,cap,nkv,gate_on,scale,qscale):
    ds,hs,bs=len(c.device),len(c.host),c.device_bytes;graph,exe=P(),P()
    try:
        d=desc(nt,nkv,cap,0,scale);rng=np.random.default_rng(234+nt+cap)
        broadcast=nt==4 and cap==64
        if broadcast:
            d.q_head_stride=d.q_token_stride=d.gate_head_stride=d.gate_token_stride=0
            d.k_head_stride=d.v_head_stride=0
        arrays={};ptrs={}
        for kind in ('q','gate','k','v','ids','out'):
            raw=c.pinned((max(1,span(d,kind)),),np.uint8);raw[:]=0xA5
            arrays[kind]=view(raw,d,kind);ptrs[kind]=c.malloc(raw.nbytes)
            arrays[kind+'_raw']=raw
        q,k,v,ids,gate=(arrays[s] for s in ('q','k','v','ids','gate'))
        q[:]=rng.normal(0,qscale,q.shape).astype(np.float32)
        # Bounded assignment staging, no duplicate100Kpool float64 materialization.
        for first in range(0,nkv,1024):
            k[first:first+1024]=rng.normal(0,.2,k[first:first+1024].shape).astype(np.float16)
            v[first:first+1024]=rng.normal(0,.2,v[first:first+1024].shape).astype(np.float16)
        ids[:]=rng.integers(0,max(1,nkv),(nt,cap),dtype=np.int32)
        if cap:
            ids[:,::11]=-1;ids[:,1::13]=nkv+7
            if cap>3:ids[:,3]=ids[:,2] # repeated selection is intentionally counted twice
        pos=c.pinned((nt,),np.int32);pos[:]=max(0,nkv-1);pos[::5]=max(0,nkv//2)
        if nt>1:pos[1]=-1
        dp=c.malloc(pos.nbytes);gate[:]=rng.normal(0,2,gate.shape).astype(np.float32)
        gate[:,0,::3]=np.float32(100);gate[:,0,1::3]=np.float32(-100)
        expected=oracle(q,k,v,ids,pos,d.scale,gate if gate_on else None)
        for kind in ('q','gate','k','v','ids'):
            raw=arrays[kind+'_raw'];c.copy(ptrs[kind],P(raw.ctypes.data),raw.nbytes,1)
        c.copy(dp,P(pos.ctypes.data),pos.nbytes,1)
        c.copy(ptrs['out'],P(arrays['out_raw'].ctypes.data),arrays['out_raw'].nbytes,1)
        modes=[];baseline=None
        for mode in (0,1):
            d.mode=mode;need=query(C.byref(d));assert need!=2**64-1
            work=c.malloc(need) if need else P()
            def invoke():return run(ptrs['q'],ptrs['gate'] if gate_on else None,ptrs['k'],ptrs['v'],ptrs['ids'],dp,ptrs['out'],C.byref(d),work,need,c.stream)
            def read():
                raw=arrays['out_raw'];c.copy(P(raw.ctypes.data),ptrs['out'],raw.nbytes,2);c.sync()
                gaps=np.ndarray((nt,H-1,2),np.uint32,buffer=raw,offset=D*4,strides=(d.out_token_stride,d.out_head_stride,4))
                assert np.all(gaps==0xA5A5A5A5),'Output head-padding canary'
                if nt>1:
                    off=(H-1)*d.out_head_stride+D*4
                    gaps=np.ndarray((nt-1,(d.out_token_stride-off)//4),np.uint32,buffer=raw,offset=off,strides=(d.out_token_stride,4))
                    assert np.all(gaps==0xA5A5A5A5),'Output token-padding canary'
                return arrays['out'].copy()
            c.ok(invoke());got=read();assert np.isfinite(got).all()
            delta=got.astype(np.float64)-expected;rel=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-30))
            maximum=float(np.max(np.abs(delta)));normmax=maximum/max(float(np.max(np.abs(expected))),1e-30)
            assert rel<2e-4 and normmax<1e-2,(nt,cap,mode,rel,normmax)
            if nt>1:assert np.count_nonzero(got[1])==0,'All-invalid query output'
            if cap==0:assert np.count_nonzero(got)==0
            c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(invoke());c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)))
            c.ok(c.cudaGraphInstantiateWithFlags(C.byref(exe),graph,0))
            q[0]*=np.float32(.875)
            if gate_on:gate[0]*=np.float32(.75)
            if nkv:k[0]*=np.float16(.9375);v[0]*=np.float16(.875)
            if cap:ids[0,-1]=0
            pos[0]=max(0,nkv-1)
            for kind in ('q','gate','k','v','ids'):
                raw=arrays[kind+'_raw'];c.copy(ptrs[kind],P(raw.ctypes.data),raw.nbytes,1)
            c.copy(dp,P(pos.ctypes.data),pos.nbytes,1)
            c.ok(c.cudaGraphLaunch(exe,c.stream));replay=read();c.ok(invoke());ordinary=read()
            assert np.array_equal(replay,ordinary),'Changed pinned inputs captured replay'
            c.ok(c.cudaGraphExecDestroy(exe));exe=P();c.ok(c.cudaGraphDestroy(graph));graph=P()
            # Query0 replay is checked again as a standalone query with unchanged
            # device addresses/metadata (all strides remain actual padded strides).
            d.nt=1;c.ok(invoke());one=read()[0];d.nt=nt
            assert np.array_equal(one,ordinary[0]),'Query/batch independence'
            modes.append(dict(mode=mode,workspace_bytes=need,complete_cpu_reference_relative_rms=rel,
                complete_cpu_reference_normalized_max=normmax,absolute_max=maximum,capture_all_outputs='bitwise PASS',
                batch_query0='bitwise PASS',replay_sha256=hashlib.sha256(replay.tobytes()).hexdigest()))
            # Second mode sees the same mutated inputs; recompute a fresh full
            # independent oracle rather than compare with stale pre-capture data.
            expected=oracle(q,k,v,ids,pos,d.scale,gate if gate_on else None)
        return dict(tokens=nt,cap=cap,n_kv=nkv,gate=gate_on,scale=scale,q_scale=qscale,input_broadcast=broadcast,output_padding_canary='PASS',
            complete_reference_values=nt*H*D,modes=modes,requested_device_bytes=c.device_bytes)
    finally:
        if exe.value:c.ok(c.cudaGraphExecDestroy(exe))
        if graph.value:c.ok(c.cudaGraphDestroy(graph))
        for p in reversed(c.device[ds:]):c.ok(c.cudaFree(p))
        for p in reversed(c.host[hs:]):c.ok(c.cudaFreeHost(p))
        del c.device[ds:];del c.host[hs:];c.device_bytes=bs

def gpu_checks(lease):
    marker=Path(lease)
    if not marker.is_file() or not marker.read_text().startswith('ACTIVE QFNBASALT'):raise RuntimeError('Coordinator exclusive GPU lease required')
    path=HERE.parent.parent/'bin/qfn-basalt-nvfp4.dll';before=hashlib.sha256(path.read_bytes()).hexdigest()
    c=base.CUDA();lib=C.CDLL(str(path));assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    abi,prepare,query,run=bind_api(lib);result=dict(dll_sha256_before=before,cases=[])
    try:
        assert abi()==1;c.ok(prepare(c.stream))
        # Combined boundaries cover both32-cell prompt and64-cell decode tails.
        for nt,cap,nkv,gate,scale,qs in ((1,0,67,False,.0625,.1),(1,33,0,True,.0625,.1),(2,1,67,True,.125,.1),
            (4,31,67,False,-.0625,.1),(17,32,101,True,.0625,1e-30),(4,33,101,True,.0,.1),
            (2,63,101,False,.0625,1e4),(4,64,101,True,.0625,.1),(17,65,101,False,.03125,.1),
            (1,2048,100003,True,.0625,.1),(17,2051,100003,True,.0625,.1),(2048,65,100003,True,.0625,.1)):
            r=gpu_case(c,run,query,nt,cap,nkv,gate,scale,qs);result['cases'].append(r)
            base.progress(f'measurement=attentioncandidate T{nt} cap{cap} pool{nkv} completeCPU/capture PASS; next=remaining case/model quality')
    finally:c.close()
    result['dll_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest();assert result['dll_sha256_after']==before
    result['qualification']='Candidate native-F16 arithmetic/masks/capture only; no fullmodel accuracy/performance qualification'
    return result

def main():
    ap=argparse.ArgumentParser(__doc__);g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-only',action='store_true');g.add_argument('--abi-only',action='store_true');g.add_argument('--allow-gpu',action='store_true')
    ap.add_argument('--lease',default=str(ROOT/'runtime/nvfp4-parity.lease'));ap.add_argument('--receipt',default=str(ROOT/'attention-parity.json'));a=ap.parse_args()
    p=Path(a.receipt).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Receipt outside task root')
    sources=[Path(__file__)]+[Path(__file__).parent/n for n in ('qfn_basalt_attention.h','basalt-attention.cu','basalt-attention-prompt.cuh','basalt-attention-decode.cuh')]+[HERE/'parity.py']
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report=dict(source_sha256_before=hashes(),gpu='NOT RUN')
    try:
        report['cpu']=cpu_checks()
        if a.abi_only:report['abi']=abi_checks()
        if a.allow_gpu:report['gpu']=gpu_checks(a.lease)
    except BaseException as exc:report['failure']=repr(exc);raise
    finally:
        report['source_sha256_after']=hashes();report['sources_unchanged']=report['source_sha256_before']==report['source_sha256_after']
        p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    assert report['sources_unchanged']

if __name__=='__main__':main()
