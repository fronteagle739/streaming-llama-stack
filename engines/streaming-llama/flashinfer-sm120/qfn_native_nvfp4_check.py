"""Minimal packed NVFP4 GPU qualification; invoke through existing gpurun.py.

Requires --allow-gpu, an ACTIVE QFNLEAD coordinator marker, and --receipt under
the explicit workspace. Uses the original parity.py CUDA ctypes helper (12GiB GPU
allocation ceiling), bounded pinned staging, and the original calibrated CPU
oracle. Synthetic arithmetic qualification does not qualify model KL/recall.
No executable, model download, process guard or background worker is created.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np

HERE=Path(__file__).resolve().parent
STACK=HERE.parents[2]
ROOT=STACK.parent
BASE=STACK/'engines/qfn-basalt-nvfp4'
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(STACK/'scripts'))
import parity as base
import dsv41_fp4_layout as layout
spec=importlib.util.spec_from_file_location('qfn_original_calibrated_parity',BASE/'calibrated-parity.py')
calibrated=importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrated)
P,I,U64=base.P,base.I,base.U64
E,F,K,BLOB,TAIL=base.E,base.F,base.K,base.BLOB,base.TAIL
SELECTED=np.array((0,15,16,31,127,128,255,256,496,511),np.int32)
PROGRESS=None
PHASE='not started'
CLEANUP_ERRORS=[]


def progress(message):
    line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' native NVFP4 fixture: '+message
    print(line,flush=True)
    with PROGRESS.open('a',encoding='utf-8') as out:out.write(line+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(slot,all_bytes=False):
    result=bytearray(base.fixture(slot))
    # Nonzero uninterpreted spare tail bytes must also survive the GPU inverse.
    result[TAIL+12:TAIL+16]=np.array([0xA5000000|slot],'<u4').tobytes()
    if all_bytes:
        for role,(n,k) in enumerate(((F,E),(F,E),(E,F))):
            blocks=np.frombuffer(result,np.uint8,count=921600,offset=role*921600).reshape(n,k//64,36)
            blocks[:,:,:4]=np.arange(n*(k//64)*4,dtype=np.uint32).astype(np.uint8).reshape(n,k//64,4)
            blocks[:,:,4:]=np.arange(n*(k//64)*32,dtype=np.uint32).astype(np.uint8).reshape(n,k//64,32)
    return bytes(result)


class Scope:
    """Release each case's allocations before the next case, using base helpers."""
    def __init__(self,c):
        self.c=c;self.ds=len(c.device);self.hs=len(c.host);self.bytes=c.device_bytes
    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb):
        c=self.c;errors=[]
        for kind,pointers,free in (('device',c.device[self.ds:],c.cudaFree),
                                   ('host',c.host[self.hs:],c.cudaFreeHost)):
            for p in reversed(pointers):
                try:c.ok(free(p))
                except BaseException as cleanup:
                    entry=dict(operation='free '+kind,pointer=p.value,failure=repr(cleanup))
                    errors.append(entry);CLEANUP_ERRORS.append(entry)
        del c.device[self.ds:];del c.host[self.hs:];c.device_bytes=self.bytes
        if errors:
            print('CUDA cleanup errors (primary exception preserved): '+json.dumps(errors),flush=True)
            if exc_type is None:raise RuntimeError('CUDA allocation cleanup failed: '+repr(errors[0]))
        return False


def phase(c,label,call):
    """Untimed diagnostic boundaries attribute asynchronous failures exactly."""
    global PHASE
    PHASE=label+' / before sync';progress(PHASE);c.sync()
    PHASE=label+' / launch';c.ok(call())
    PHASE=label+' / after sync';progress(PHASE);c.sync()
    PHASE=label+' / complete';progress(PHASE)


def destroy_graph(c,executable,graph):
    primary=sys.exc_info()[1];errors=[]
    for label,p,free in (('graph executable',executable,c.cudaGraphExecDestroy),
                         ('graph',graph,c.cudaGraphDestroy)):
        if p.value:
            try:c.ok(free(p))
            except BaseException as cleanup:
                entry=dict(operation='destroy '+label,pointer=p.value,failure=repr(cleanup))
                errors.append(entry);CLEANUP_ERRORS.append(entry)
    if errors and primary is None:raise RuntimeError('CUDA graph cleanup failed: '+repr(errors[0]))


def close_cuda(c,primary):
    # Reuse the original helper's exact frees while attempting every cleanup
    # even if a poisoned CUDA context rejects one; retain the FIRST fault.
    scope=Scope(c);scope.ds=scope.hs=scope.bytes=0
    pending=primary
    try:scope.__exit__(type(primary) if primary else None,primary,None)
    except BaseException as cleanup:pending=cleanup
    try:c.ok(c.cudaStreamDestroy(c.stream))
    except BaseException as cleanup:
        CLEANUP_ERRORS.append(dict(operation='destroy stream',failure=repr(cleanup)))
        if pending is None:pending=cleanup
    if primary is None and pending is not None:raise pending


def pinned(c,shape,dtype):
    # The largest individual host buffer is the20MiB T2048 input. Output
    # comparisons use16-token chunks; neither raw bank is ever held on the CPU.
    size=int(np.prod(shape))*np.dtype(dtype).itemsize
    if size>64*2**20:raise MemoryError('Fixture individual pinned-buffer budget64MiB')
    return c.pinned(shape,dtype)


def upload(c,dest,array):c.copy(dest,P(array.ctypes.data),array.nbytes,1)


def api(c):
    native_path=STACK/'bin/qfn-native-nvfp4.dll'
    original_path=STACK/'bin/qfn-basalt-nvfp4.dll'
    native=C.CDLL(str(native_path));old=C.CDLL(str(original_path))
    functions=dict(
        query=base.bind(native,'qfn_native_nvfp4_packed_workspace_bytes',[I,C.c_uint,I],U64),
        prepare=base.bind(native,'qfn_native_nvfp4_prepare',[P]),
        run=base.bind(native,'qfn_native_nvfp4_run',[P,P,P,P,P,C.c_uint,I,I,P,P,U64,P]),
        unpack=base.bind(native,'qfn_native_nvfp4_unpack',[P,P,P,C.c_uint,C.c_uint,P,U64,P,P,P]),
        group_plan=base.bind(native,'qfn_native_nvfp4_group_plan',[P,I,I,P,P,P]),
        old_query=base.bind(old,'qfn_basalt_nvfp4_workspace_bytes',[I,I,C.c_uint,I],U64),
        old_run=base.bind(old,'qfn_basalt_nvfp4_run',[P,P,P,U64,C.c_uint,I,I,P,P,U64,P,I]),
        old_calibrated=base.bind(old,'qfn_basalt_nvfp4_run_calibrated',[P,P,P,P,U64,C.c_uint,I,I,P,P,U64,P]),
        old_decode=base.bind(old,'qfn_basalt_nvfp4_run_calibrated_decode',[P,P,P,P,P,U64,C.c_uint,I,I,P,P,U64,P]),
    )
    assert base.bind(native,'qfn_native_nvfp4_abi',[])()==1
    c.ok(functions['prepare'](c.stream))
    c.sync()
    return functions


def banks_and_inverse(c,f):
    raw=c.malloc(520*BLOB);packed=c.malloc(520*BLOB)
    with Scope(c):
        stage=pinned(c,(BLOB,),np.uint8)
        source_hash=hashlib.sha256()
        for slot in range(520):
            blob=fixture(slot)
            if slot<512:source_hash.update(blob)
            stage[:]=np.frombuffer(blob,np.uint8);upload(c,P(raw.value+slot*BLOB),stage);c.sync()
            stage[:]=np.frombuffer(layout.pack_nvfp4_blob(blob),np.uint8)
            upload(c,P(packed.value+slot*BLOB),stage);c.sync()
        progress('uploaded520 streamed synthetic raw/packed blobs, valid unequal tails; next full512 inverse byte oracle')
        inverse=c.malloc(512*BLOB)
        c.ok(f['unpack'](packed,None,None,512,0,inverse,512*BLOB,None,None,c.stream))
        actual_hash=hashlib.sha256()
        for slot in range(512):
            c.copy(P(stage.ctypes.data),P(inverse.value+slot*BLOB),BLOB,2);c.sync()
            assert stage.tobytes()==fixture(slot),('Full GPU inverse byte mismatch',slot)
            actual_hash.update(stage)
        assert actual_hash.hexdigest()==source_hash.hexdigest()
        full=dict(status='byte-exact PASS',slots=512,bytes=512*BLOB,
                  raw_sha256=source_hash.hexdigest(),all16_tail_bytes=True)
        progress('full512 inverse byte-exact PASS including every16-byte tail; next selected high-slot/all-byte inverse')
        # Layout-only payloads cover every code/scale bit pattern, including
        # nonfinite scale encodings; they are restored before arithmetic tests.
        for slot in range(512,520):
            stage[:]=np.frombuffer(layout.pack_nvfp4_blob(fixture(slot,True)),np.uint8)
            upload(c,P(packed.value+slot*BLOB),stage);c.sync()
        ids=pinned(c,(10,),np.int32);logical=pinned(c,(10,),np.int32)
        ids[:]=(519,0,512,519,511,1,515,128,0,518)
        logical[:]=(7,33,511,8,129,0,4,499,2,128)
        di,dl=c.malloc(ids.nbytes),c.malloc(logical.nbytes)
        dg,dc=c.malloc(ids.nbytes),c.malloc(ids.nbytes)
        upload(c,di,ids);upload(c,dl,logical)
        c.ok(f['unpack'](packed,di,dl,520,len(ids),inverse,512*BLOB,dg,dc,c.stream))
        for r,slot in enumerate(ids):
            c.copy(P(stage.ctypes.data),P(inverse.value+r*BLOB),BLOB,2);c.sync()
            assert stage.tobytes()==fixture(int(slot),int(slot)>=512),('Selected inverse mismatch',r,int(slot))
        maps=pinned(c,(2,10),np.int32)
        c.copy(P(maps[0].ctypes.data),dg,ids.nbytes,2)
        c.copy(P(maps[1].ctypes.data),dc,ids.nbytes,2);c.sync()
        assert np.array_equal(maps[0],np.arange(10,dtype=np.int32))
        assert np.array_equal(maps[1],logical)
        # Optional logical input also has an exact documented physical-ID default.
        c.ok(f['unpack'](packed,di,None,520,len(ids),inverse,512*BLOB,dg,dc,c.stream))
        c.copy(P(maps[1].ctypes.data),dc,ids.nbytes,2);c.sync()
        assert np.array_equal(maps[1],ids)
        for slot in range(512,520):
            restored=layout.pack_nvfp4_blob(fixture(slot))
            stage[:]=np.frombuffer(restored,np.uint8)
            upload(c,P(packed.value+slot*BLOB),stage);c.sync()
            c.copy(P(stage.ctypes.data),P(packed.value+slot*BLOB),BLOB,2);c.sync()
            assert stage.tobytes()==restored,('High packed slot restoration',slot)
        progress('selected inverse PASS; packed slots512..519 restored byte-exact, raw520 bank was never overwritten; next arithmetic')
        selected=dict(status='byte-exact PASS',physical_ids=ids.tolist(),logical_ids=logical.tolist(),
                      duplicated_out_of_order_high_slots=True,all256_code_and_scale_bytes=True,
                      all16_tail_bytes=True,optional_logical_default='exact PASS')
    return raw,packed,dict(full=full,selected=selected)


def compare(c,a,b,nt,ids,exact=False):
    """Compare EVERY output coordinate with bounded staging and F64 sums."""
    with Scope(c):
        ha=pinned(c,(16,K,E),np.float32);hb=pinned(c,(16,K,E),np.float32)
        square=reference_square=0.;max_abs=reference_max=0.;different=0
        sha=hashlib.sha256();zero_seen=nt==1
        for first in range(0,nt,16):
            n=min(16,nt-first);size=n*K*E*4;offset=first*K*E*4
            c.copy(P(ha.ctypes.data),P(a.value+offset),size,2)
            c.copy(P(hb.ctypes.data),P(b.value+offset),size,2);c.sync()
            aa,bb=ha[:n],hb[:n]
            assert np.isfinite(aa).all() and np.isfinite(bb).all(),'Nonfinite complete output'
            different+=int(np.count_nonzero(aa.view(np.uint32)!=bb.view(np.uint32)))
            da=aa.astype(np.float64);db=bb.astype(np.float64);delta=da-db
            square+=float(np.sum(delta*delta));reference_square+=float(np.sum(db*db))
            max_abs=max(max_abs,float(np.max(np.abs(delta))));reference_max=max(reference_max,float(np.max(np.abs(db))))
            sha.update(aa)
            if nt>1 and first<=1<first+n:
                assert np.count_nonzero(aa[1-first])==np.count_nonzero(bb[1-first])==0,'Zero token output'
                zero_seen=True
            if exact:assert np.array_equal(aa.view(np.uint32),bb.view(np.uint32)),'Expected bit-exact complete output'
        rms=(square/max(reference_square,1e-60))**.5
        normalized_max=max_abs/max(reference_max,1e-30)
        assert zero_seen
        assert rms<=1e-3 and normalized_max<=1e-2,('Grouped complete-output arithmetic gate',rms,normalized_max)
        return dict(values=nt*K*E,relative_rms=rms,normalized_max_abs=normalized_max,
                    different_values=different,zero_token='PASS' if nt>1 else 'not applicable',sha256=sha.hexdigest())


def native_case(c,f,raw,packed,nt,equal_gu=False):
    with Scope(c):
        x=pinned(c,(nt,E),np.float32);ids=pinned(c,(nt,K),np.int32)
        aux=pinned(c,(512,3),np.float32)
        x[:]=np.random.default_rng(971+nt).normal(0,.1,x.shape).astype(np.float32)
        if nt>1:x[1]=0
        ids[:]=(np.arange(nt,dtype=np.int32)[:,None]*17+SELECTED[None,:])%512
        if nt==2048:ids[:]=SELECTED # large occupied groups plus502 empty experts
        aux[:]=calibrated.calibration();assert np.all(aux[:,0]!=aux[:,1])
        if equal_gu:
            aux[:,1]=aux[:,0]
            assert aux[:,0].tobytes()==aux[:,1].tobytes()
        dx,di,ds=c.malloc(x.nbytes),c.malloc(ids.nbytes),c.malloc(aux.nbytes)
        da,db,dc=(c.malloc(nt*K*E*4) for _ in range(3))
        nw,ow=f['query'](nt,512,6),f['old_query'](nt,K,512,4)
        assert nw and ow
        work,old_work=c.malloc(nw),c.malloc(ow)
        def upload_all():
            for dest,src in ((dx,x),(di,ids),(ds,aux)):upload(c,dest,src)
        def native(dest):return f['run'](dx,di,None,packed,ds,512,nt,6,dest,work,nw,c.stream)
        def old(dest):return f['old_calibrated'](dx,di,raw,ds,BLOB,512,nt,K,dest,old_work,ow,c.stream)
        progress(f'mode6 T{nt} first physical IDs={ids[0].tolist()}, native workspace={nw}, original workspace={ow}; next isolated native/reference calls')
        upload_all()
        phase(c,f'mode6 T{nt} native baseline',lambda:native(da))
        phase(c,f'mode6 T{nt} original calibrated baseline',lambda:old(db))
        baseline=compare(c,da,db,nt,ids)
        # Independent calibrated CPU arithmetic samples supplement the complete
        # original-kernel comparison; this uses the existing source oracle.
        with Scope(c):
            sample=pinned(c,(K,E),np.float32)
            old_sample=pinned(c,(K,E),np.float32)
            c.copy(P(sample.ctypes.data),da,sample.nbytes,2);c.sync()
            c.copy(P(old_sample.ctypes.data),db,old_sample.nbytes,2);c.sync()
            cpu_errors=[]
            for r in (0,K-1):
                expert=int(ids[0,r]);expected=calibrated.reference(x[0],fixture(expert),aux[expert])
                error=float(np.linalg.norm(sample[r].astype(np.float64)-expected)/max(np.linalg.norm(expected),1e-30))
                original_error=float(np.linalg.norm(old_sample[r].astype(np.float64)-expected)/max(np.linalg.norm(expected),1e-30))
                original_max=float(np.max(np.abs(old_sample[r].astype(np.float64)-expected))/max(np.max(np.abs(expected)),1e-30))
                # Retain the existing calibrated fixture's stronger source
                # oracle gate for the unchanged original reference path.
                assert original_error<2e-4 and original_max<1e-2,('Original calibrated CPU oracle',nt,r,original_error,original_max)
                assert error<=1e-3,('Independent calibrated CPU oracle',nt,r,error)
                cpu_errors.append(dict(route=r,native_relative_rms=error,
                                       original_relative_rms=original_error,original_normalized_max_abs=original_max))
        graph,executable=P(),P()
        try:
            c.sync();c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(native(da))
            c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)))
            c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0))
            x[:]*=np.float32(.875);aux[:]*=np.array([1.125,1.125 if equal_gu else .875,.9375],np.float32)
            upload_all()
            phase(c,f'mode6 T{nt} captured changed input',lambda:c.cudaGraphLaunch(executable,c.stream))
            phase(c,f'mode6 T{nt} ordinary changed input',lambda:native(dc))
            replay=compare(c,da,dc,nt,ids,True)
            assert replay['sha256']!=baseline['sha256'],'Changed input/calibration must affect output'
            phase(c,f'mode6 T{nt} original calibrated changed input',lambda:old(db))
            changed=compare(c,da,db,nt,ids)
        finally:
            destroy_graph(c,executable,graph)
        result=dict(tokens=nt,style='hotspot' if nt==2048 else 'spread',native_workspace=nw,
                    original_workspace=ow,complete_output=baseline,independent_cpu_relative_rms=cpu_errors,
                    captured_vs_ordinary=replay,
                    requested_device_bytes=c.device_bytes)
        result['calibration_relation']='equal-gate-up/independent-down' if equal_gu else 'independent-roles'
        result['changed_input_and_equal_gu_calibration' if equal_gu else 'changed_input_and_unequal_calibration']=changed
        progress(f'mode6 T={nt} complete output RMS={baseline["relative_rms"]:.9g}, changed replay bit-exact; next remaining case')
        return result


def compatibility_case(c,f,raw,packed,mode,nt=4,slots=520,vector=False):
    with Scope(c):
        x=pinned(c,(nt,E),np.float32);ids=pinned(c,(nt,K),np.int32)
        logical=pinned(c,(nt,K),np.int32);aux=pinned(c,(512,3),np.float32)
        x[:]=np.random.default_rng(581+mode).normal(0,.1,x.shape).astype(np.float32)
        if nt>1:x[1]=0
        placement=(519,512,0,511,518,2,519,515,128,514) if slots==520 else (511,508,0,15,510,2,511,500,128,504)
        for t in range(nt):ids[t]=np.roll(placement,t)
        logical[:]=(np.arange(nt,dtype=np.int32)[:,None]*31+np.arange(K,dtype=np.int32)[None,:]*17)%512
        aux[:]=calibrated.calibration()
        dx,di,dl,ds=(c.malloc(a.nbytes) for a in (x,ids,logical,aux))
        da,db,dc=(c.malloc(nt*K*E*4) for _ in range(3))
        reference_mode=2 if vector else 5 if mode==6 else mode
        nw,ow=f['query'](nt,slots,mode),f['old_query'](nt,K,slots,reference_mode)
        assert nw and ow
        if mode==6:
            assert nw==f['query'](nt,512,6)==f['query'](nt,520,6)==f['query'](nt,20001,6)
            assert f['query'](5,520,6)==0,'Long mode6 arbitrary-slot request must fail loudly'
        work,old_work=c.malloc(nw),c.malloc(ow)
        def upload_all():
            for dest,src in ((dx,x),(di,ids),(dl,logical),(ds,aux)):upload(c,dest,src)
        def native(dest=da):return f['run'](dx,di,dl if mode>=5 else None,packed,ds if mode>=5 else None,slots,nt,mode,dest,work,nw,c.stream)
        def old():
            if vector:return f['old_run'](dx,di,raw,BLOB,slots,nt,K,db,old_work,ow,c.stream,2)
            if mode>=5:return f['old_decode'](dx,di,dl,raw,ds,BLOB,slots,nt,K,db,old_work,ow,c.stream)
            return f['old_run'](dx,di,raw,BLOB,slots,nt,K,db,old_work,ow,c.stream,mode)
        upload_all()
        phase(c,f'mode{mode} T{nt} packed compatibility baseline',native)
        phase(c,f'mode{mode} T{nt} original baseline',old)
        baseline=compare(c,da,db,nt,ids,mode!=6)
        graph,executable=P(),P()
        try:
            c.sync();c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(native())
            c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)))
            c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0))
            x[:]*=np.float32(.75)
            if mode>=5:
                logical[:]=(logical+7)%512
                # Change both placement and unequal role calibration on replay.
                ids[:]=ids[:,::-1].copy()
                aux[:]*=np.array([1.125,.875,.9375],np.float32)
            upload_all()
            phase(c,f'mode{mode} T{nt} captured changed input',lambda:c.cudaGraphLaunch(executable,c.stream))
            phase(c,f'mode{mode} T{nt} ordinary changed input',lambda:native(dc))
            captured=compare(c,da,dc,nt,ids,True)
            phase(c,f'mode{mode} T{nt} original changed input',old)
            replay=compare(c,da,db,nt,ids,mode!=6)
            assert replay['sha256']!=baseline['sha256'],'Changed compatibility input must affect output'
        finally:
            destroy_graph(c,executable,graph)
        progress(f'mode{mode} T{nt} physical slots{slots}/max{int(ids.max())} duplicated payloads PASS, native captured/eager bit-exact; reference RMS={replay["relative_rms"]:.9g}; next remaining case')
        return dict(mode=mode,arithmetic='packed-vector-q8_1' if vector else 'original',
                    reference_mode=reference_mode,tokens=nt,physical_slots=slots,physical_ids=ids.tolist(),
                    logical_ids=logical.tolist() if mode>=5 else None,complete_output=baseline,
                    captured_changed_input=replay,captured_vs_ordinary=captured,
                    native_workspace=nw,original_workspace=ow)


def residual_components_cpu(x,global_scale):
    """Independent adjacent FP4 hi/lo quantizer, returning normalized operands.

    Globals stay outside each GEMM, matching the explicit bridge epilogues.
    The low operand quantizes F32 input minus reconstructed high input.
    """
    values=np.asarray(x,np.float32).reshape(-1,16)
    g=np.float32(global_scale)
    def component(v):
        scales=base.nearest_even(np.max(np.abs(v),axis=1)*(np.float32(1)/g)*np.float32(1/6),base.UE)
        inv=np.divide(np.float32(1),scales*g,out=np.zeros_like(scales),where=scales!=0)
        z=v*inv[:,None]
        codes=np.copysign(base.nearest_even(np.abs(z),base.FP4),z)
        return (codes*scales[:,None]).astype(np.float32)
    high=component(values)
    remaining=(values-high*g).astype(np.float32)
    low=component(remaining)
    return high.reshape(-1).astype(np.float64),low.reshape(-1).astype(np.float64)


def residual_reference_cpu(x,blob,globals_):
    """Two F32 products, F32 add, then independent activation/weight globals."""
    tails=np.frombuffer(blob,'<f4',count=4,offset=TAIL)
    def product(v,role,rows,cols,offset):
        high,low=residual_components_cpu(v,globals_[role])
        weights=base.matrix(blob,rows,cols,offset)
        hi=(weights@high).astype(np.float32)
        lo=(weights@low).astype(np.float32)
        return (((hi+lo).astype(np.float32)*np.float32(globals_[role]))*tails[role]).astype(np.float32)
    gate=product(x,0,F,E,0)
    up=product(x,1,F,E,921600)
    hidden=(gate/(np.float32(1)+np.exp(-gate))*up).astype(np.float32)
    return product(hidden,2,E,F,1843200).astype(np.float64)


def residual_case(c,f,packed,nt,slots):
    """All-coordinate safety plus independent arithmetic samples and replay.

    CPU samples permit RMS2e-3/max2e-2 relative to the composed FP4 oracle:
    FP32 GEMM accumulation differences can move a subsequent FP4 rounding
    boundary. This is a component gate, not a native-Q8/model quality claim.
    """
    with Scope(c):
        x=pinned(c,(nt,E),np.float32);ids=pinned(c,(nt,K),np.int32)
        logical=pinned(c,(nt,K),np.int32);aux=pinned(c,(512,3),np.float32)
        x[:]=np.random.default_rng(923+nt+slots).normal(0,.1,x.shape).astype(np.float32)
        if nt>1:x[1]=0
        placement=(519,512,0,511,518,2,519,515,128,514) if slots==520 else (511,508,0,15,510,2,511,500,128,504)
        for t in range(nt):ids[t]=np.roll(placement,t)
        logical[:]=(np.arange(nt,dtype=np.int32)[:,None]*31+np.arange(K,dtype=np.int32)[None,:]*17)%512
        aux[:]=calibrated.calibration()
        assert np.all(aux[:,0]!=aux[:,1])
        dx,di,dl,ds=(c.malloc(a.nbytes) for a in (x,ids,logical,aux))
        expected_guard=pinned(c,(256,),np.uint8);expected_guard[:]=0xA5
        guard_stage=pinned(c,(256,),np.uint8);guards=[]
        def guarded(size):
            storage=c.malloc(size+512)
            for at in (storage.value,storage.value+256+size):
                pointer=P(at);upload(c,pointer,expected_guard);guards.append(pointer)
            return P(storage.value+256)
        nw=f['query'](nt,slots,6)
        assert nw and nw==f['query'](nt,512,6)==f['query'](nt,520,6)
        work=guarded(nw);da,dc=(guarded(nt*K*E*4) for _ in range(2))
        stage=pinned(c,(nt,K,E),np.float32)
        def upload_all():
            for dest,src in ((dx,x),(di,ids),(dl,logical),(ds,aux)):upload(c,dest,src)
        def native(dest):return f['run'](dx,di,dl,packed,ds,slots,nt,6,dest,work,nw,c.stream)
        def check_guards():
            for pointer in guards:
                c.copy(P(guard_stage.ctypes.data),pointer,256,2);c.sync()
                assert np.array_equal(guard_stage,expected_guard),'Residual output/workspace boundary overwritten'
        def inspect(dest,zero=False,oracle=False):
            c.copy(P(stage.ctypes.data),dest,stage.nbytes,2);c.sync()
            assert np.isfinite(stage).all(),'Residual complete output is nonfinite'
            if zero:assert np.count_nonzero(stage)==0,'Residual zero input must produce exact zero'
            if nt>1:assert np.count_nonzero(stage[1])==0,'Residual zero token must produce exact zero'
            samples=[]
            if oracle:
                for token in sorted({0,nt-1}):
                    for route in (0,K-1):
                        physical=int(ids[token,route]);canonical=int(logical[token,route])
                        expected=residual_reference_cpu(x[token],fixture(physical),aux[canonical])
                        actual=stage[token,route].astype(np.float64);difference=actual-expected
                        rms=float(np.linalg.norm(difference)/max(np.linalg.norm(expected),1e-30))
                        maximum=float(np.max(np.abs(difference))/max(np.max(np.abs(expected)),1e-30))
                        assert rms<=2e-3 and maximum<=2e-2,('Independent residual CPU oracle',nt,slots,token,route,rms,maximum)
                        samples.append(dict(token=token,route=route,physical_id=physical,logical_id=canonical,
                                            relative_rms=rms,normalized_max_abs=maximum))
            check_guards()
            return dict(values=stage.size,sha256=hashlib.sha256(stage.tobytes()).hexdigest(),
                        all_finite=True,guard_bytes_intact=True,cpu_samples=samples)
        upload_all()
        refused=[]
        for mode in (2,5):
            assert f['query'](nt,slots,mode)==0,('Residual flag ignored by workspace query',mode)
            status=f['run'](dx,di,dl,packed,ds,slots,nt,mode,da,work,nw,c.stream)
            assert status!=0,('Residual flag ignored by non-mode6 run',mode)
            refused.append(dict(mode=mode,workspace_bytes=0,run_status=status))
        phase(c,f'residual mode6 T{nt} slots{slots} baseline',lambda:native(da))
        baseline=inspect(da,oracle=True)
        saved=x.copy();x[:]=0;upload(c,dx,x)
        phase(c,f'residual mode6 T{nt} slots{slots} all-zero input',lambda:native(dc))
        zero=inspect(dc,zero=True);x[:]=saved;upload(c,dx,x)
        graph,executable=P(),P()
        try:
            c.sync();c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(native(da))
            c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)))
            c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0))
            x[:]*=np.float32(.75);aux[:]*=np.array([1.125,.875,.9375],np.float32)
            ids[:]=ids[:,::-1].copy();logical[:]=(logical+7)%512
            upload_all()
            phase(c,f'residual mode6 T{nt} slots{slots} captured changed inputs',lambda:c.cudaGraphLaunch(executable,c.stream))
            phase(c,f'residual mode6 T{nt} slots{slots} eager changed inputs',lambda:native(dc))
            replay=compare(c,da,dc,nt,ids,True)
            changed=inspect(da,oracle=True)
            assert changed['sha256']!=baseline['sha256'],'Changed residual inputs/calibration/maps must affect output'
        finally:destroy_graph(c,executable,graph)
        progress(f'mode6 residual T{nt} slots{slots} complete outputs/zero/canaries/CPU oracle PASS, changed eager/capture byte-exact; next remaining residual case')
        return dict(tokens=nt,physical_slots=slots,native_workspace=nw,physical_ids=ids.tolist(),
                    logical_ids=logical.tolist(),baseline=baseline,zero_input=zero,changed=changed,
                    captured_vs_eager=replay,unsupported_short_modes_refused=refused,
                    cpu_tolerance=dict(relative_rms=2e-3,normalized_max_abs=2e-2))


def grouping_case(c,f,nt,pattern):
    """Actual CUDA old/new arrays, checked against a stable CPU permutation."""
    with Scope(c):
        tokens=np.arange(nt,dtype=np.int32)[:,None]
        routes=np.arange(K,dtype=np.int32)[None,:]
        if pattern=='permuted':values=(tokens*37+routes*51)%512
        elif pattern=='skew':values=(routes+tokens%K)%K
        elif pattern=='sparse':values=480+(tokens*7+routes*3)%32
        else:raise ValueError(pattern)
        assert np.all(np.diff(np.sort(values,axis=1),axis=1)>0),'Fixture requires distinct routes'
        ids=pinned(c,(nt,K),np.int32);ids[:]=values
        di=c.malloc(ids.nbytes);upload(c,di,ids)
        expected_order=np.argsort(values.reshape(-1),kind='stable').astype(np.int32)
        counts=np.bincount(values.reshape(-1),minlength=512)
        expected_bounds=np.concatenate(([0],np.cumsum(counts))).astype(np.int32)
        actual_order=pinned(c,(nt*K,),np.int32);actual_bounds=pinned(c,(513,),np.int32)
        # Two device guards surround each output; unwritten values use a sentinel.
        do=c.malloc(actual_order.nbytes+8);db=c.malloc(actual_bounds.nbytes+8)
        orders=[];boundaries=[];guard=np.array([0x1234abcd],np.int32)
        for parallel in (0,1):
            c.copy(do,P(guard.ctypes.data),4,1);c.copy(P(do.value+actual_order.nbytes+4),P(guard.ctypes.data),4,1)
            c.copy(db,P(guard.ctypes.data),4,1);c.copy(P(db.value+actual_bounds.nbytes+4),P(guard.ctypes.data),4,1)
            actual_order[:]=-1;actual_bounds[:]=-1
            upload(c,P(do.value+4),actual_order);upload(c,P(db.value+4),actual_bounds)
            phase(c,f'grouping T{nt} {pattern} parallel{parallel}',
                  lambda:f['group_plan'](di,nt,parallel,P(do.value+4),P(db.value+4),c.stream))
            c.copy(P(actual_order.ctypes.data),P(do.value+4),actual_order.nbytes,2)
            c.copy(P(actual_bounds.ctypes.data),P(db.value+4),actual_bounds.nbytes,2);c.sync()
            assert actual_order.tobytes()==expected_order.tobytes(),('Stable grouping order mismatch',nt,pattern,parallel)
            assert actual_bounds.tobytes()==expected_bounds.tobytes(),('Grouping bounds mismatch',nt,pattern,parallel)
            orders.append(actual_order.tobytes());boundaries.append(actual_bounds.tobytes())
            for start,size in ((do,actual_order.nbytes),(db,actual_bounds.nbytes)):
                for offset in (0,size+4):
                    got=np.zeros(1,np.int32);c.copy(P(got.ctypes.data),P(start.value+offset),4,2);c.sync()
                    assert got[0]==guard[0],('Grouping output guard changed',nt,pattern,parallel,offset)
        assert orders[0]==orders[1] and boundaries[0]==boundaries[1]
        return dict(tokens=nt,pattern=pattern,status='byte-exact PASS',
                    routes=nt*K,order_sha256=hashlib.sha256(orders[0]).hexdigest(),
                    bounds_sha256=hashlib.sha256(boundaries[0]).hexdigest(),
                    cpu_stable_oracle=True,output_guards=True)


def main():
    global ROOT,PROGRESS
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--progress-log',type=Path,required=True)
    parser.add_argument('--allow-gpu',action='store_true',required=True)
    parser.add_argument('--lease',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--residual-self-test',action='store_true',
                        help='Explicit short mode6 hi+residual component arm; requires residual flag1')
    parser.add_argument('--vector-self-test',action='store_true',
                        help='DSV packed-vector Q8_1 arm vs original native-Q8, T1..4 and high/duplicate slots')
    parser.add_argument('--grouping-self-test',action='store_true',
                        help='Compare actual warp/CTA256 CUDA order and bounds against a stable CPU oracle')
    parser.add_argument('--share-gu-input-self-test',action='store_true',
                        help='Equal gate/up globals, eager/captured changes; compare full output hashes across explicit policy0/1 receipts')
    args=parser.parse_args();ROOT=args.workspace.resolve();PROGRESS=args.progress_log.resolve();receipt=args.receipt.resolve();lease=args.lease.resolve()
    residual_flag=os.environ.get('QFN_NATIVE_NVFP4_DECODE_RESIDUAL','0')
    vector_flag=os.environ.get('QFN_NATIVE_NVFP4_DECODE_VECTOR','0')
    shared_gu_flag=os.environ.get('QFN_NATIVE_NVFP4_SHARE_GU_INPUT','0')
    if shared_gu_flag not in ('0','1'):raise ValueError('Shared gate/up input policy must be exactly0/1')
    if sum((args.vector_self_test,args.residual_self_test,args.grouping_self_test,args.share_gu_input_self_test))>1:raise ValueError('Choose one component arm')
    if shared_gu_flag=='1' and not any((args.vector_self_test,args.residual_self_test,args.grouping_self_test,args.share_gu_input_self_test)):
        raise ValueError('The original unequal-calibration long suite requires shared-input policy0; use the separate equal-GU component arm for policy1')
    if vector_flag!=('1' if args.vector_self_test else '0'):
        raise ValueError('--vector-self-test requires QFN_NATIVE_NVFP4_DECODE_VECTOR=1; other arms require0')
    if residual_flag!=('1' if args.residual_self_test else '0'):
        raise ValueError('--residual-self-test requires QFN_NATIVE_NVFP4_DECODE_RESIDUAL=1; original full suite requires0')
    if not receipt.is_relative_to(ROOT):raise ValueError('Receipt must remain under '+str(ROOT))
    if not lease.is_relative_to(ROOT) or not lease.is_file() or not lease.read_text().startswith(('ACTIVE QFNLEAD','ACTIVE QFNBASALT')):
        raise RuntimeError('Existing coordinator ACTIVE QFNLEAD exclusive GPU lease marker required')
    sources=[Path(__file__),HERE/'qfn_native_nvfp4_bridge.cu',HERE/'qfn_native_nvfp4_bridge.h',HERE/'qfn_packed_vector.cuh',
             HERE/'qfn-native-nvfp4-kernel.py',HERE/'export_qfn_nvfp4.py',
             BASE/'parity.py',BASE/'calibrated-parity.py',STACK/'scripts/dsv41_fp4_layout.py',
             STACK/'bin/qfn-native-nvfp4.dll',STACK/'bin/qfn-basalt-nvfp4.dll']
    hashes=lambda:{str(p.relative_to(STACK)):digest(p) for p in sources}
    result=dict(status='IN-PROGRESS',qualification='Synthetic layout/arithmetic only; whole-model KL/top1/recall pending',
                host_fixture_data_budget_bytes=1536*2**20,gpu_allocation_budget_bytes=12*2**30,
                short_mode6_fused_mid=os.environ.get('QFN_NATIVE_NVFP4_FUSED_MID','0'),
                source_sha256_before=hashes(),native_cases=[],compatibility_cases=[])
    if args.residual_self_test:
        result.update(test_arm='residual-self-test',short_mode6_residual=residual_flag,residual_cases=[])
    if args.vector_self_test:
        result.update(test_arm='vector-self-test',short_mode6_vector=vector_flag,vector_cases=[])
    if args.grouping_self_test:
        result.update(test_arm='grouping-self-test',grouping_cases=[],
                      grouping_policy=os.environ.get('QFN_NATIVE_NVFP4_GROUP_PARALLEL','0'))
    if args.share_gu_input_self_test:
        result.update(test_arm='share-gu-input-self-test',shared_gu_policy=shared_gu_flag,shared_gu_cases=[])
    c=None;primary=None
    def save():receipt.write_text(json.dumps(result,indent=2)+'\n')
    try:
        save();c=base.CUDA();f=api(c)
        if args.share_gu_input_self_test and shared_gu_flag=='1':
            refused=[]
            for mode in range(6):
                assert f['query'](40,512,mode)==0,('Shared-GU long unsupported mode was accepted',mode)
                refused.append(mode)
            result['unsupported_long_modes_refused']=refused;save()
        if args.grouping_self_test:
            for nt in (1,4,63,64,1696,2048,4096,16384):
                for pattern in ('permuted','skew','sparse'):
                    result['grouping_cases'].append(grouping_case(c,f,nt,pattern));save()
        else:
            raw,packed,result['inverse']=banks_and_inverse(c,f);save()
        if args.share_gu_input_self_test:
            for nt in (40,1696,2048):
                result['shared_gu_cases'].append(native_case(c,f,raw,packed,nt,equal_gu=True));save()
            # Short requests continue to support independently calibrated roles.
            for nt in (1,4):
                result['compatibility_cases'].append(compatibility_case(c,f,raw,packed,6,nt));save()
        elif args.vector_self_test:
            for slots in (520,512):
                for nt in (1,2,3,4):
                    result['vector_cases'].append(compatibility_case(c,f,raw,packed,6,nt,slots,True));save()
        elif args.residual_self_test:
            for slots in (520,512):
                for nt in (1,4):result['residual_cases'].append(residual_case(c,f,packed,nt,slots));save()
        elif not args.grouping_self_test:
            for nt in (1,40,2048):result['native_cases'].append(native_case(c,f,raw,packed,nt));save()
            for mode in (2,5):result['compatibility_cases'].append(compatibility_case(c,f,raw,packed,mode));save()
            for nt in (1,2,3,4):result['compatibility_cases'].append(compatibility_case(c,f,raw,packed,6,nt));save()
            for nt in (1,4):result['compatibility_cases'].append(compatibility_case(c,f,raw,packed,6,nt,512));save()
        result['status']='PASS'
    except BaseException as exc:
        primary=exc
        result['status']='FAILED/INCOMPLETE';result['failure']=repr(exc)
        result['failure_phase']=PHASE;result['first_failure_traceback']=traceback.format_exc();raise
    finally:
        try:
            if c is not None:close_cuda(c,primary)
            result['cuda_allocations_released']=not CLEANUP_ERRORS
        except BaseException as cleanup:
            if primary is None:
                result['status']='FAILED/INCOMPLETE';result['failure']=repr(cleanup)
                result['failure_phase']='CUDA cleanup';result['first_failure_traceback']=traceback.format_exc()
            raise
        finally:
            result['cuda_cleanup_errors']=CLEANUP_ERRORS
            if CLEANUP_ERRORS:
                result['cuda_allocations_released']=False
                if result['status']=='PASS':result['status']='FAILED/INCOMPLETE'
            result['source_sha256_after']=hashes()
            result['sources_unchanged']=result['source_sha256_before']==result['source_sha256_after']
            if not result['sources_unchanged']:result['status']='FAILED/INCOMPLETE'
            save();print(json.dumps(result,indent=2),flush=True)
    assert result['sources_unchanged'],'Source/DLL changed during fixture'
    progress(('shared-GU component arm PASS (cross-policy exact comparison still required)' if args.share_gu_input_self_test else 'grouping component checks PASS' if args.grouping_self_test else 'packed-vector component checks PASS' if args.vector_self_test else 'residual component checks PASS' if args.residual_self_test else 'all layout/mode6/mode2/mode5 checks PASS')+
             '; latest complete receipt='+str(receipt)+'; next parent whole-model100K/recall')


if __name__=='__main__':main()
