"""Guarded calibrated-static A4 checks; GPU only with coordinator exclusive lease.

Separate unequal gate/up globals, original-H down calibration, all weight tails,
complete zero/scatter/capture checks. Synthetic parity does not qualify model KL.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import parity as base
import numpy as np

ROOT,HERE=base.ROOT,base.HERE
P,I,U64=base.P,base.I,base.U64
E,F,K,BLOB,TAIL=base.E,base.F,base.K,base.BLOB,base.TAIL
SELECTED=(0,15,16,31,127,128,255,256,496,511)


def calibration():
    e=np.arange(512,dtype=np.float32)
    gate=np.float32(.0004)+np.remainder(e,5)*np.float32(.0001)
    up=gate*np.float32(2.25)
    down=(np.float32(1)+np.remainder(e,7))*np.float32(2e-6)
    return np.stack((gate,up,down),axis=1).astype(np.float32)


def a4_static(x,g):
    x=np.asarray(x,np.float32).reshape(-1,16);g=np.float32(g)
    if not np.isfinite(g) or not g>0:raise ValueError('Static calibration must be finite positive F32')
    sd=base.nearest_even(np.max(np.abs(x),axis=1)*(np.float32(1)/g)*np.float32(1/6),base.UE)
    scale=sd*g;inv=np.divide(np.float32(1),scale,out=np.zeros_like(scale),where=scale!=0)
    z=x*inv[:,None]
    return (np.copysign(base.nearest_even(np.abs(z),base.FP4),z)*scale[:,None]).reshape(-1).astype(np.float64)


def reference(x,blob,globals_):
    sg,su,sd,_=np.frombuffer(blob,'<f4',count=4,offset=TAIL)
    g=(base.matrix(blob,F,E,0)@a4_static(x,globals_[0])).astype(np.float32)*sg
    u=(base.matrix(blob,F,E,921600)@a4_static(x,globals_[1])).astype(np.float32)*su
    h=(g/(np.float32(1)+np.exp(-g))*u).astype(np.float32)
    return (base.matrix(blob,E,F,1843200)@a4_static(h,globals_[2]))*sd


def expected_workspace(nt):
    n=min(nt,16384);r=n*K;align=lambda x:(x+255)//256*256
    tiles=(r+63)//64+512
    return sum(map(align,(r*F*4,r*E*4,tiles*20*4608,tiles*64*4,r*4,r*4,513*4,513*4)))


def cpu_checks():
    result=base.cpu_checks();aux=calibration();blob=base.fixture(128)
    x=np.random.default_rng(41).normal(0,.1,E).astype(np.float32)
    assert np.all(aux[:,0]!=aux[:,1]) and aux.shape==(512,3)
    assert np.array_equal(reference(np.zeros(E,np.float32),blob,aux[128]),np.zeros(E))
    actual=reference(x,blob,aux[128]);changed=aux[128].copy();changed[1]=changed[0]
    assert not np.array_equal(actual,reference(x,blob,changed)),'Unequal up calibration must affect the oracle'
    for g in (0,-1,np.nan,np.inf):
        try:a4_static(x,g)
        except ValueError:pass
        else:raise AssertionError('Invalid calibration accepted')
    result.update(calibrated_oracle='PASS',unequal_gate_up='PASS',original_h_down_scale='PASS',
                  expected_workspace_bytes={str(t):expected_workspace(t) for t in (1,2048,16384,16385)})
    return result


def bind_api(lib):
    return (base.bind(lib,'qfn_basalt_nvfp4_calibrated_abi',[]),
            base.bind(lib,'qfn_basalt_nvfp4_prepare',[P]),
            base.bind(lib,'qfn_basalt_nvfp4_workspace_bytes',[I,I,C.c_uint,I],U64),
            base.bind(lib,'qfn_basalt_nvfp4_run_calibrated',[P,P,P,P,U64,C.c_uint,I,I,P,P,U64,P]))


def abi_checks():
    home=Path(__import__('os').environ.get('CUDA_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'))
    dirs=[__import__('os').add_dll_directory(str(p)) for p in (home/'bin',home/'bin/x64') if p.is_dir()]
    try:
        lib=C.CDLL(str(HERE.parent.parent/'bin/qfn-basalt-nvfp4.dll'))
        abi,_,query,run=bind_api(lib);assert abi()==1
        for nt in (1,2048,16384,16385):assert query(nt,K,512,4)==expected_workspace(nt)
        assert query(1,K,513,4)==query(1,K,512,6)==0
        # Mode5 is a new declared capability; invalid-mode coverage moves to6.
        assert base.bind(lib,'qfn_basalt_nvfp4_calibrated_decode_abi',[])()==1
        align=lambda value:(value+255)//256*256
        for nt in (1,2,3,4):
            r=nt*K;tiles=(r+63)//64+r
            expected=sum(map(align,(r*BLOB,r*3*4,r*4,r*F*4,r*E*4,
                tiles*20*4608,tiles*64*4,r*4,r*4,(r+1)*4,(r+1)*4)))
            assert query(nt,K,20002,5)==expected
        assert query(0,K,20002,5)==query(5,K,20002,5)==query(1,K-1,20002,5)==0
        need=query(1,K,512,4)
        # Deliberate argument errors must fail before any CUDA/context access.
        assert run(P(256),P(256),P(256),P(257),BLOB,512,1,K,P(256),P(4096),need,None)!=0
        assert run(P(256),P(256),P(256),None,BLOB,512,1,K,P(256),P(4096),need,None)!=0
        return dict(calibrated_host_abi='PASS',calibrated_abi=abi())
    finally:
        for d in dirs:d.close()


def case(c,run,query,bank,daux,aux,nt,style):
    ds,hs,bs=len(c.device),len(c.host),c.device_bytes;graph,executable=P(),P()
    try:
        x=c.pinned((nt,E),np.float32);ids=c.pinned((nt,K),np.int32)
        x[:]=np.random.default_rng(71).normal(0,.1,x.shape).astype(np.float32)
        if nt>1:x[1]=0
        ids[:]=(np.arange(nt,dtype=np.int32)[:,None]%32)*16+np.arange(K,dtype=np.int32)[None,:]
        ids[0]=SELECTED
        if nt>1:ids[-1]=np.roll(SELECTED,3)
        if style=='hotspot':ids[:]=SELECTED
        first_input=x[0].copy();dx,di=c.malloc(x.nbytes),c.malloc(ids.nbytes)
        outbytes=nt*K*E*4;da,db=c.malloc(outbytes),c.malloc(outbytes)
        wbytes=query(nt,K,512,4);assert wbytes==expected_workspace(nt);ws=c.malloc(wbytes)
        a,b=c.pinned((16,K,E),np.float32),c.pinned((16,K,E),np.float32)
        def invoke(dest):return run(dx,di,bank,daux,BLOB,512,nt,K,dest,ws,wbytes,c.stream)
        def upload():
            for d,src in ((dx,x),(di,ids),(daux,aux)):c.copy(d,P(src.ctypes.data),src.nbytes,1)
        def read(first,n,both=False):
            off=first*K*E*4;size=n*K*E*4
            c.copy(P(a.ctypes.data),P(da.value+off),size,2)
            if both:c.copy(P(b.ctypes.data),P(db.value+off),size,2)
            c.sync()
        upload();c.ok(invoke(da));baseline={};samples=[];zeros=0
        for first in range(0,nt,16):
            n=min(16,nt-first);read(first,n)
            assert np.isfinite(a[:n]).all()
            mask=~np.isin(ids[first:first+n],SELECTED)
            assert np.count_nonzero(a[:n][mask])==0,'Zero-weight expert result'
            zeros+=int(np.count_nonzero(mask))*E
            if first<=1<first+n and nt>1:assert np.count_nonzero(a[1-first])==0,'Zero input result'
            for t in sorted(set((0,nt-1))):
                if not first<=t<first+n:continue
                baseline[t]=a[t-first].copy()
                for k in range(K):
                    slot=int(ids[t,k]);ref=reference(x[t],base.fixture(slot),aux[slot]);got=a[t-first,k].astype(np.float64)
                    diff=got-ref;rel=float(np.linalg.norm(diff)/max(np.linalg.norm(ref),1e-30))
                    worst=float(np.max(np.abs(diff))/max(np.max(np.abs(ref)),1e-30))
                    # Published FP4 matmul plus independent calibrated quantization.
                    assert rel<2e-4 and worst<1e-2,(nt,t,k,slot,rel,worst)
                    samples.append(dict(token=t,route=k,slot=slot,globals=aux[slot].tolist(),relative_rms=rel,normalized_max_abs=worst))
        c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(invoke(da))
        c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)));c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0))
        original_aux=aux.copy();x[:]*=np.float32(.875);aux[:]*=np.array([1.125,.875,.9375],np.float32)
        upload();c.ok(c.cudaGraphLaunch(executable,c.stream));c.ok(invoke(db))
        replay=hashlib.sha256()
        for first in range(0,nt,16):
            n=min(16,nt-first);read(first,n,True)
            assert np.array_equal(a[:n],b[:n]),'Changed-pinned-input/aux captured replay'
            replay.update(a[:n].tobytes())
        x[0]=first_input;aux[:]=original_aux
        sx,si,so=c.malloc(E*4),c.malloc(K*4),c.malloc(K*E*4)
        c.copy(sx,P(x.ctypes.data),E*4,1);c.copy(si,P(ids.ctypes.data),K*4,1)
        c.copy(daux,P(aux.ctypes.data),aux.nbytes,1)
        c.ok(run(sx,si,bank,daux,BLOB,512,1,K,so,ws,wbytes,c.stream))
        c.copy(P(a.ctypes.data),so,K*E*4,2);c.sync()
        assert np.array_equal(a[0],baseline[0]),'Static calibrated token/batch independence'
        return dict(tokens=nt,style=style,mode=4,workspace_bytes=wbytes,samples=samples,
                    exact_zero_weight_values=zeros,zero_input='PASS' if nt>1 else 'not applicable',
                    complete_graph_changed_inputs_and_aux='bitwise PASS',replay_sha256=replay.hexdigest(),
                    token_batch_independence='bitwise PASS',device_peak_requested_bytes=c.device_bytes,
                    compact_groups_covered=len(set((ids//16).reshape(-1).tolist())))
    finally:
        if executable.value:c.ok(c.cudaGraphExecDestroy(executable))
        if graph.value:c.ok(c.cudaGraphDestroy(graph))
        for p in reversed(c.device[ds:]):c.ok(c.cudaFree(p))
        for p in reversed(c.host[hs:]):c.ok(c.cudaFreeHost(p))
        del c.device[ds:];del c.host[hs:];c.device_bytes=bs


def gpu_checks(lease):
    if not Path(lease).is_file() or not Path(lease).read_text().startswith('ACTIVE QFNBASALT'):raise RuntimeError('Coordinator exclusive lease required')
    c=base.CUDA();path=HERE.parent.parent/'bin/qfn-basalt-nvfp4.dll'
    before=hashlib.sha256(path.read_bytes()).hexdigest();lib=C.CDLL(str(path))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    abi,prepare,query,run=bind_api(lib)
    memset=base.bind(c.lib,'cudaMemsetAsync',[P,I,C.c_size_t,P])
    copy2d=base.bind(c.lib,'cudaMemcpy2DAsync',[P,C.c_size_t,P,C.c_size_t,C.c_size_t,C.c_size_t,I,P])
    result=dict(abi=abi(),dll_sha256_before=before,cases=[],qualification='Synthetic calibrated arithmetic only; wholemodel KL/top1/recall pending')
    try:
        assert abi()==1;c.ok(prepare(c.stream))
        bank=c.malloc(512*BLOB);c.ok(memset(bank,0,512*BLOB,c.stream))
        tails=c.pinned((512,4),np.float32);tails[:]=(1,1,1,0)
        c.ok(copy2d(P(bank.value+TAIL),BLOB,P(tails.ctypes.data),16,16,512,1,c.stream))
        stage=c.pinned((BLOB,),np.uint8)
        for slot in SELECTED:
            stage[:]=np.frombuffer(base.fixture(slot),np.uint8);c.copy(P(bank.value+slot*BLOB),P(stage.ctypes.data),BLOB,1);c.sync()
        aux=c.pinned((512,3),np.float32);aux[:]=calibration();daux=c.malloc(aux.nbytes)
        for nt,style in ((1,'spread'),(17,'spread'),(1024,'spread'),(2048,'hotspot'),(16385,'spread')):
            r=case(c,run,query,bank,daux,aux,nt,style);result['cases'].append(r)
            base.progress(f'measurement=calibratedA4 T={nt} style={style} unequalgateup/alltails CPUoracle PASS; completegraph/zero/token invariants PASS; next=remaining case or model quality')
    finally:c.close()
    result['dll_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest();assert result['dll_sha256_after']==before
    return result


def main():
    ap=argparse.ArgumentParser(__doc__);g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-only',action='store_true');g.add_argument('--abi-only',action='store_true');g.add_argument('--allow-gpu',action='store_true')
    ap.add_argument('--lease',default=str(ROOT/'runtime/nvfp4-parity.lease'))
    ap.add_argument('--receipt',default=str(ROOT/'calibrated-parity.json'));args=ap.parse_args()
    receipt=Path(args.receipt).resolve()
    if not receipt.is_relative_to(ROOT):raise ValueError('Receipt must remain under task root')
    paths=[HERE/'calibrated-parity.py',HERE/'parity.py']
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result=dict(test_source_sha256_before=hashes(),gpu='NOT RUN')
    try:
        result['cpu']=cpu_checks()
        if args.abi_only:result['abi']=abi_checks()
        if args.allow_gpu:result['gpu']=gpu_checks(args.lease)
    except BaseException as exc:
        result['failure']=repr(exc)
        if args.allow_gpu:result['gpu']='FAILED/INCOMPLETE'
        raise
    finally:
        result['test_source_sha256_after']=hashes();result['test_sources_unchanged']=result['test_source_sha256_before']==result['test_source_sha256_after']
        receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    assert result['test_sources_unchanged']
    base.progress('measurement=calibratedA4 '+('GPU synthetic PASS' if args.allow_gpu else 'CPU/host check PASS; GPU NOT RUN')+'; next=coordinator review')


if __name__=='__main__':main()
