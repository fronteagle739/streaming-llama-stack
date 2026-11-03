"""Complete-score QSA parity, bounded host staging, same task DLL; GPU opt-in only."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import parity as base

D,H,R,TOP=128,4,4,512
P,I,U=base.P,base.I,base.U64
HERE,ROOT=base.HERE,base.ROOT


def select_cpu(score,pos,top=TOP,ratio=R):
    """Current QFN radix/tie/tail semantics, expressed as a stable CPU ordering."""
    nv=(pos+1)//ratio
    if nv>len(score): raise ValueError("Position exceeds score capacity")
    if nv<=top: chosen=np.arange(nv)
    else:
        keys=np.where(score[:nv]>0,score[:nv],np.float32(0))
        chosen=np.sort(np.lexsort((np.arange(nv),-keys))[:top])
    cells=(chosen[:,None]*ratio+np.arange(ratio)).reshape(-1)
    tail=np.arange(nv*ratio,pos+1)
    out=np.full(top*ratio+ratio-1,-1,np.int32)
    out[:len(cells)]=cells
    out[len(cells):len(cells)+len(tail)]=tail
    return out


def cpu_checks():
    result=base.cpu_checks()
    proof=json.loads((HERE/'QSA_PROVENANCE.json').read_text())
    for rel,key in [('vendor/basalt/src/kernels/cuda/qsa_select.cu','basalt_source_sha256'),
                    (proof['oracle_snapshot'],'oracle_source_sha256')]:
        assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest()==proof[key]
    for scores in (np.zeros(1031,np.float32),np.ones(1031,np.float32),-np.ones(1031,np.float32)):
        ids=select_cpu(scores,1031*4-2)
        assert np.array_equal(ids[:TOP*R],np.arange(TOP*R))
        assert np.array_equal(ids[TOP*R:],np.array([4120,4121,4122]))
    scores=np.zeros(1031,np.float32)
    scores[700]=1
    assert 700*4 in select_cpu(scores,1031*4-1)
    return {**result,'qsa_cpu_selector/provenance':'PASS'}


def abi_checks():
    # Pure host shape/alignment failures are checked before any runtime call.
    directories=[]
    home=Path(__import__('os').environ.get('CUDA_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'))
    for p in (home/'bin',home/'bin/x64'):
        if p.is_dir(): directories.append(__import__('os').add_dll_directory(str(p)))
    try:
        lib=C.CDLL(str(HERE.parent.parent/'bin/qfn-basalt-nvfp4.dll'))
        abi=base.bind(lib,'qfn_basalt_qsa_abi',[])
        run=bind_run(lib)
        assert abi()==1
        strides=[512,2048,0,256,0,16,0,4,0]
        def call(nt=1,kt=0,q=P(4096),stride_values=strides):
            return run(q,P(8192),P(12288),P(16384),nt,1,1,kt,*stride_values,1.,1.,None,0)
        assert call(nt=0)!=0 and call(nt=1048561)!=0 and call(kt=2)!=0
        assert call(q=P(4097))!=0
        overflow=strides.copy();overflow[0]=2**64-4
        assert call(stride_values=overflow)!=0
        return {'qsa_host_admission':'PASS','abi':abi()}
    finally:
        for directory in directories: directory.close()


def bind_run(lib):
    return base.bind(lib,'qfn_basalt_qsa_run',[P,P,P,P,I,I,I,I]+[U]*9+[C.c_float,C.c_float,P,I])


def pinned_view(c,shape,dtype,strides):
    dtype=np.dtype(dtype)
    size=sum((n-1)*s for n,s in zip(shape,strides))+dtype.itemsize
    storage=c.pinned((size,),np.uint8)
    storage[:]=0xA5
    return np.ndarray(shape,dtype,buffer=storage,strides=strides),storage


def run_case(c,run,nt,npool,key_type,style='random',ns=1):
    dstart,hstart,bytes_start=len(c.device),len(c.host),c.device_bytes
    kd=np.dtype(np.float16 if key_type==0 else np.float32)
    qh=(D+4)*4;qt=H*qh+16;qs=nt*qt+16
    kp=(D+8)*kd.itemsize;ks=npool*kp+16
    wt=(H+4)*4;ws=nt*wt+16;ot=(npool+7)*4;oss=nt*ot+16
    strides=[qh,qt,qs,kp,ks,wt,ws,ot,oss]
    graph,executable=P(),P()
    try:
        q,qstore=pinned_view(c,(ns,nt,H,D),np.float32,(qs,qt,qh,4))
        k,kstore=pinned_view(c,(ns,npool,D),kd,(ks,kp,kd.itemsize))
        w,wstore=pinned_view(c,(ns,nt,H),np.float32,(ws,wt,4))
        rng=np.random.default_rng(17)
        q[:]=rng.normal(0,.2,q.shape).astype(np.float32)
        k[:]=rng.normal(0,.1,k.shape).astype(kd)
        w[:]=np.float32(1/np.sqrt(D))
        se,sh=.75,1.25
        if style=='zero': q[:]=0
        elif style=='ties': q[:]=1;k[:]=.125
        elif style=='negative': q[:]=1;k[:]=.125;w[:]=-1
        elif style=='near_cutoff':
            common=rng.normal(0,.1,(D,)).astype(np.float32)
            q[:]=common*2+rng.normal(0,.01,q.shape).astype(np.float32)
            k[:]=(common+rng.normal(0,2e-6,k.shape).astype(np.float32)).astype(kd)
        if nt==17 and style=='random': w[:]=rng.uniform(-.5,.75,w.shape).astype(np.float32)
        dq,dk,dw=c.malloc(qstore.nbytes),c.malloc(kstore.nbytes),c.malloc(wstore.nbytes)
        outsize=(ns-1)*oss+(nt-1)*ot+npool*4
        da,db=c.malloc(outsize),c.malloc(outsize)
        stage_a=c.pinned((32,npool),np.float32);stage_b=c.pinned((32,npool),np.float32)
        copy2d=base.bind(c.lib,'cudaMemcpy2DAsync',[P,C.c_size_t,P,C.c_size_t,C.c_size_t,C.c_size_t,I,P])
        def invoke(mode,dest):
            return run(dq,dk,dw,dest,nt,npool,ns,key_type,*strides,se,sh,c.stream,mode)
        def upload():
            for d,a in ((dq,qstore),(dk,kstore),(dw,wstore)): c.copy(d,P(a.ctypes.data),a.nbytes,1)
        def read_pair(first,s,n):
            offset=s*oss+first*ot
            for dest,src in ((stage_a,da),(stage_b,db)):
                c.ok(copy2d(P(dest.ctypes.data),npool*4,P(src.value+offset),ot,npool*4,n,2,c.stream))
            c.sync()
        upload();c.ok(invoke(0,da));c.ok(invoke(1,db))
        sse=ref_sse=0.;maxabs=maxref=0.;changed=selected_rows=0;cpu_double=[]
        for s in range(ns):
            for first in range(0,nt,32):
                n=min(32,nt-first);read_pair(first,s,n)
                aa,bb=stage_a[:n],stage_b[:n]
                if not np.isfinite(aa).all() or not np.isfinite(bb).all(): raise AssertionError('Nonfinite complete score matrix')
                diff=aa.astype(np.float64)-bb
                sse+=float(np.sum(diff*diff));ref_sse+=float(np.sum(bb.astype(np.float64)**2))
                maxabs=max(maxabs,float(np.max(np.abs(diff))));maxref=max(maxref,float(np.max(np.abs(bb))))
                if style=='zero': assert np.count_nonzero(aa)==np.count_nonzero(bb)==0
                for token in (0,nt-1):
                    if not first<=token<first+n: continue
                    row=token-first
                    # Exercise complete, causal prefix, incomplete-tail and identity selection.
                    for pos in (min(npool*R-1,99999),min(npool*R-2,4098)):
                        ia,ib=select_cpu(aa[row],pos),select_cpu(bb[row],pos)
                        changed+=int(np.count_nonzero(ia!=ib));selected_rows+=1
                        if style in ('zero','ties','negative'): assert np.array_equal(ia,ib)
                        assert np.all(np.diff(ia[ia>=0])>0)
                    if token==0:
                        for b in (0,npool-1):
                            dots=np.sum(q[s,0].astype(np.float64)*k[s,b].astype(np.float64),axis=1)
                            ref=float(np.sum(np.maximum(dots*se,0)*w[s,0])*sh)
                            err=abs(float(bb[row,b])-ref)/max(1.,abs(ref))
                            cpu_double.append(err)
                            if err>1e-5: raise AssertionError(f'Warp vs CPUdouble sample {err}')
        rel=(sse/max(ref_sse,1e-30))**.5
        if rel>1e-5 or maxabs/max(1.,maxref)>1e-4:
            raise AssertionError(f'Complete score numeric gate rel={rel} maxabs={maxabs}')
        c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(invoke(0,da))
        c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)))
        c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0))
        q[:]*=np.float32(.875);w[:]*=np.float32(1.125);upload()
        c.ok(c.cudaGraphLaunch(executable,c.stream));c.ok(invoke(0,db))
        for s in range(ns):
            for first in range(0,nt,32):
                n=min(32,nt-first);read_pair(first,s,n)
                assert np.array_equal(stage_a[:n],stage_b[:n]),'Captured replay differs from ordinary same-input TC scores'
        return {'shape':[ns,nt,npool],'key_type':'F16' if key_type==0 else 'F32','style':style,
                'full_matrix_elements':ns*nt*npool,'complete_relative_rms':rel,'complete_max_abs':maxabs,
                'cpu_double_oracle_sample_relative':cpu_double,'selected_rows_compared':selected_rows,
                'selected_id_positions_different':changed,'capture_changed_pinned_inputs':'bitwise PASS',
                'all_actual_strides_bytes':strides}
    finally:
        if executable.value: c.ok(c.cudaGraphExecDestroy(executable))
        if graph.value: c.ok(c.cudaGraphDestroy(graph))
        for p in reversed(c.device[dstart:]): c.ok(c.cudaFree(p))
        for p in reversed(c.host[hstart:]): c.ok(c.cudaFreeHost(p))
        del c.device[dstart:];del c.host[hstart:];c.device_bytes=bytes_start


def gpu_checks(lease):
    if not Path(lease).is_file() or not Path(lease).read_text().startswith('ACTIVE QFNBASALT'):
        raise RuntimeError('Exclusive coordinator lease must begin ACTIVE QFNBASALT')
    c=base.CUDA();path=HERE.parent.parent/'bin/qfn-basalt-nvfp4.dll'
    before=hashlib.sha256(path.read_bytes()).hexdigest();lib=C.CDLL(str(path))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before,'DLL changed between hash and load'
    abi=base.bind(lib,'qfn_basalt_qsa_abi',[]);prepare=base.bind(lib,'qfn_basalt_qsa_prepare',[P]);run=bind_run(lib)
    result={'abi':abi(),'dll_sha256_before':before,'numerical_policy':'3xTF32; near-cutoff selected sets may differ; full-model quality gates pending','cases':[]}
    try:
        assert abi()==1;c.ok(prepare(c.stream))
        for kt in (0,1):
            for nt,npool,style,ns in ((1,100000,'random',1),(17,1031,'random',2),(2048,100000,'random',1),
                                     (17,1031,'zero',1),(17,1031,'ties',1),(17,1031,'negative',1),(17,1031,'near_cutoff',1)):
                r=run_case(c,run,nt,npool,kt,style,ns);result['cases'].append(r)
                base.progress(f"measurement=QSA complete scores kt={kt} T={nt} pools={npool} style={style} RMS={r['complete_relative_rms']} changedIDpositions={r['selected_id_positions_different']}; next=remaining parity case")
    finally: c.close()
    result['dll_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['dll_sha256_after']==before,'DLL changed during GPU tests'
    return result


def main():
    ap=argparse.ArgumentParser(__doc__);g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-only',action='store_true');g.add_argument('--abi-only',action='store_true');g.add_argument('--allow-gpu',action='store_true')
    ap.add_argument('--lease',default=str(ROOT/'runtime/nvfp4-parity.lease'))
    ap.add_argument('--receipt',default=str(ROOT/'qsa-parity.json'));args=ap.parse_args()
    receipt=Path(args.receipt).resolve()
    if not receipt.is_relative_to(ROOT): raise ValueError('Receipt must stay under task root')
    report={'cpu':cpu_checks(),'gpu':'NOT RUN'}
    if args.abi_only: report['abi']=abi_checks()
    if args.allow_gpu: report['gpu']=gpu_checks(args.lease)
    receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    base.progress('measurement=QSA '+('GPU complete-score PASS' if args.allow_gpu else 'CPU/host admission PASS; GPU NOT RUN')+'; next=coordinator quality/gain review')


if __name__=='__main__': main()
