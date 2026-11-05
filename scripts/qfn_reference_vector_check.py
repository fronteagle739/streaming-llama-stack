"""Guarded CPU-oracle/GPU qualification of explicit reference vector policies.

Uses the actual GPU lease and shared Studio paging guard; no model is loaded.
Every runtime/library/output path and both memory budgets are explicit flags.
Checks both original-weight F32 and converted Q8/Q4 rounding when selected.
"""
import argparse,ctypes as C,datetime,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--hold-file',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpu-threads',type=int,choices=range(1,17),default=1);p.add_argument('--host-budget-gib',type=float,required=True);p.add_argument('--floor-gib',type=float,required=True);p.add_argument('--commit-gib',type=float,required=True);p.add_argument('--guard-worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--output',type=Path,required=True);p.add_argument('--fused-mid',type=int,choices=[0,1],required=True);p.add_argument('--vector-policy',type=int,choices=[2,3,4],required=True);a=p.parse_args()
s=Path(__file__).resolve().parents[1];sys.path.insert(0,str(s))
if not a.guard_worker:
    from slstack.paging_limits import shared_owned_paging_limit_bytes
    from slstack.config import load
    from slstack.process import OwnedProcess
    if a.output.exists():raise ValueError('Use a fresh evidence path for each fixture')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=load(overrides=dict(runtime_root=str(a.runtime_root),task='QfnPair',hold_file=str(a.hold_file),gpu_uuid=a.gpu_uuid,host_budget_gib=a.host_budget_gib,job_commit_gib=a.commit_gib,min_available_gib=a.floor_gib,owned_paging_limit_bytes=shared_owned_paging_limit_bytes(),lease_minutes=3,request_timeout_s=180,gpu_idle_max_mib=1024))
    job=OwnedProcess(cfg,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--guard-worker'],'qfn-reference-vector-fixture',gpu=True,telemetry=True,scope='studio')
    proof=job.wait();a.output.with_suffix('.guard.json').write_text(json.dumps(proof,indent=2))
    assert proof['exit_code']==0 and proof['release_allowed']
    print(a.output.read_text());raise SystemExit(0)
import psutil
if not any('slstack.guard' in process.cmdline() for process in psutil.Process().parents()):
    raise RuntimeError('Fixture worker requires its actual independent guardian')
os.environ['OPENBLAS_NUM_THREADS']=str(a.cpu_threads);os.environ['OMP_NUM_THREADS']=str(a.cpu_threads)
os.environ['QFN_NATIVE_NVFP4_DECODE_VECTOR']=str(a.vector_policy)
os.environ['QFN_NATIVE_NVFP4_FUSED_MID']=str(a.fused_mid)
os.environ['QFN_NATIVE_NVFP4_DECODE_RESIDUAL']='0'
r=a.library.resolve().parent.parent;sys.path.insert(0,str(s/'engines/qfn-basalt-nvfp4'));sys.path.insert(0,str(s/'scripts'))
import numpy as np
import parity as b
from dsv41_fp4_layout import pack_nvfp4_blob
P,I,U64=b.P,b.I,b.U64
def sha(file):
    with file.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()
inputs=[a.library.resolve(),Path(__file__).resolve(),s/'engines/qfn-basalt-nvfp4/parity.py',s/'scripts/dsv41_fp4_layout.py']
inputs.extend(s/'engines/streaming-llama/flashinfer-sm120'/name for name in ('qfn_native_nvfp4_bridge.cu','qfn_native_nvfp4_bridge.h','qfn_packed_vector.cuh'))
before={str(file):sha(file) for file in inputs}
c=b.CUDA();report=dict(status='FAIL',vector_policy=a.vector_policy,fused_mid=a.fused_mid,cases=[])
def rounded(matrix,global_scale):
    if a.vector_policy==2:return matrix,global_scale
    values=(matrix.astype(np.float32)*global_scale).reshape(-1,32)
    if a.vector_policy==3:
        delta=np.abs(values).max(axis=1)/np.float32(127)
        inverse=np.divide(np.float32(1),delta,out=np.zeros_like(delta),where=delta!=0)
        scaled=values*inverse[:,None];codes=np.sign(scaled)*np.floor(np.abs(scaled)+np.float32(.5))
    else:
        peak=values[np.arange(len(values)),np.argmax(np.abs(values),axis=1)]
        delta=peak/np.float32(-8)
        inverse=np.divide(np.float32(1),delta,out=np.zeros_like(delta),where=delta!=0)
        codes=np.minimum(15,np.floor(values*inverse[:,None]+np.float32(8.5)))-8
    result=(delta.astype(np.float16).astype(np.float32)[:,None]*codes).reshape(matrix.shape)
    return result.astype(np.float64),np.float32(1)
try:
    lib=C.CDLL(str(a.library.resolve()))
    query=b.bind(lib,'qfn_native_nvfp4_packed_workspace_bytes',[I,C.c_uint,I],U64)
    prepare=b.bind(lib,'qfn_native_nvfp4_prepare',[P])
    run=b.bind(lib,'qfn_native_nvfp4_run',[P,P,P,P,P,C.c_uint,I,I,P,P,U64,P])
    c.ok(prepare(c.stream));c.sync()
    bank=c.malloc(10*b.BLOB)
    def device(array):
        ptr=c.malloc(array.nbytes);c.copy(ptr,P(array.ctypes.data),array.nbytes,1);c.sync();return ptr
    for slot in range(10):
        packed=np.frombuffer(pack_nvfp4_blob(b.fixture(slot)),np.uint8)
        c.copy(P(bank.value+slot*b.BLOB),P(packed.ctypes.data),packed.nbytes,1);c.sync()
    aux=np.ones((512,3),np.float32);da=device(aux)
    for nt in (1,2,3,4):
        x=np.random.default_rng(761+nt).normal(0,.2,(nt,b.E)).astype(np.float32)
        ids=np.stack([np.roll(np.arange(10,dtype=np.int32),t) for t in range(nt)])
        logical=(ids+71).astype(np.int32)
        dx,di,dl=device(x),device(ids),device(logical)
        nw=query(nt,10,6);assert 0<nw<64*2**20,nw;work=c.malloc(nw)
        shape=(nt,10,b.E);actual=np.zeros(shape,np.float32);nbytes=actual.nbytes
        guarded=np.full(nbytes+512,0xa5,np.uint8);storage=device(guarded);out=P(storage.value+256)
        def call():return run(dx,di,dl,bank,da,10,nt,6,out,work,nw,c.stream)
        c.ok(call());c.sync();c.copy(P(actual.ctypes.data),out,nbytes,2);c.sync()
        expected=np.zeros(shape,np.float32)
        for slot in range(10):
            blob=b.fixture(slot);sg,su,sd,_=np.frombuffer(blob,'<f4',count=4,offset=b.TAIL)
            gate=b.matrix(blob,b.F,b.E,0);up=b.matrix(blob,b.F,b.E,921600);down=b.matrix(blob,b.E,b.F,1843200)
            gate,sg=rounded(gate,sg);up,su=rounded(up,su);down,sd=rounded(down,sd)
            g=(x.astype(np.float64)@gate.T).astype(np.float32)*sg
            u=(x.astype(np.float64)@up.T).astype(np.float32)*su
            hidden=(g/(np.float32(1)+np.exp(-g))*u).astype(np.float32)
            y=(hidden.astype(np.float64)@down.T).astype(np.float32)*sd
            for t in range(nt):expected[t,int(np.flatnonzero(ids[t]==slot)[0])]=y[t]
        delta=actual.astype(np.float64)-expected
        rms=float(np.linalg.norm(delta)/np.linalg.norm(expected));maximum=float(np.max(np.abs(delta))/np.max(np.abs(expected)))
        assert np.isfinite(actual).all() and rms<2e-5 and maximum<1e-4,(nt,rms,maximum)
        c.copy(P(guarded.ctypes.data),storage,len(guarded),2);c.sync()
        assert np.all(guarded[:256]==0xa5) and np.all(guarded[-256:]==0xa5),'Output extent overrun'
        eager=actual.tobytes();graph=P();executable=P()
        try:
            c.ok(c.cudaStreamBeginCapture(c.stream,0));c.ok(call());c.ok(c.cudaStreamEndCapture(c.stream,C.byref(graph)))
            c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0));c.ok(c.cudaGraphLaunch(executable,c.stream));c.sync()
            c.copy(P(actual.ctypes.data),out,nbytes,2);c.sync();assert eager==actual.tobytes(),'Captured arithmetic differs'
        finally:
            if executable.value:c.ok(c.cudaGraphExecDestroy(executable))
            if graph.value:c.ok(c.cudaGraphDestroy(graph))
        x[:]=0;c.copy(dx,P(x.ctypes.data),x.nbytes,1);c.ok(call());c.sync();c.copy(P(actual.ctypes.data),out,nbytes,2);c.sync();assert np.all(actual==0),'Zero input differs'
        report['cases'].append(dict(tokens=nt,relative_rms=rms,normalized_max_abs=maximum,captured_equals_eager=True,output_canaries=True,zero_input=True))
    assert before=={str(file):sha(file) for file in inputs},'Fixture inputs changed while running'
    report['inputs_sha256']=before
    report['status']='PASS'
finally:
    c.close();report['device_and_host_allocations_freed']=True
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
