"""Explicitly lossy Q4_K -> MXFP4 speed proxy; never a native model.

Bounded CPU streaming, durable tensor-boundary resume, cooperative read pauses.
Dense/Engram/metadata payloads are copied without quantization.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from slstack.gguf_header import read_header, tensor_map, validate_dsv41_types
from slstack.lease import active_holds
from scripts.dsv41_fp4_layout import CANONICAL,FLASHINFER,KEY,validate_layout,requant_chunks

LABEL='Q4\u2192MXFP4 requant (lossy), speed proxy for native FP4'
SUFFIXES=('_gate_exps.weight','_up_exps.weight','_down_exps.weight')
SIZES={0:(1,4),1:(1,2),8:(32,34),12:(256,144),24:(1,1),39:(32,17)}
_hold_cache={}
_conversion_hold=None

def align(n,a):return (n+a-1)//a*a
def string(s):
    value=s.encode('utf8');return struct.pack('<Q',len(value))+value
def nbytes(t):
    block,size=SIZES[t['type']]
    count=math.prod(t['shape'])
    if count%block:raise ValueError('Unaligned tensor '+t['name'])
    return count//block*size

def layout(source,expert_layout=CANONICAL):
    validate_layout(expert_layout)
    h=validate_dsv41_types(read_header(source))
    a=h['metadata'].get('general.alignment',32)
    if a<32 or a>65536 or a&(a-1):raise ValueError('Unsupported alignment')
    tensors=[];offset=0
    for old in h['tensors']:
        t=dict(old)
        if t['name'].endswith(SUFFIXES):
            if t['type']!=12:raise ValueError('Expected Q4_K experts')
            t['type']=39
        t['offset']=offset;offset+=align(nbytes(t),a);tensors.append(t)
    desc_size=sum(8+len(t['name'].encode())+4+8*len(t['shape'])+4+8 for t in tensors)
    with source.open('rb') as f:prefix=f.read(h['header_bytes']-desc_size)
    key='deepseek41.speed_proxy'
    if key in h['metadata']:raise ValueError('Input already has speed-proxy marker')
    if KEY in h['metadata']:raise ValueError('Source already has a kernel-native expert layout')
    prefix=prefix[:16]+struct.pack('<Q',len(h['metadata'])+1+(expert_layout!=CANONICAL))+prefix[24:]
    prefix+=string(key)+struct.pack('<I',8)+string(LABEL)
    if expert_layout!=CANONICAL:
        if a<128:raise ValueError('Kernel-native FP4 requires alignment>=128')
        prefix+=string(KEY)+struct.pack('<I',8)+string(expert_layout)
    desc=b''.join(string(t['name'])+struct.pack('<I',len(t['shape']))+
                  struct.pack('<'+'Q'*len(t['shape']),*t['shape'])+
                  struct.pack('<IQ',t['type'],t['offset']) for t in tensors)
    raw=prefix+desc
    raw+=bytes(align(len(raw),a)-len(raw))
    return h,tensors,raw,offset+len(raw),align(h['header_bytes'],a),a

def converter(library):
    lib=ctypes.CDLL(str(library))
    fn=lib.requant_q4_mxfp4
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int]
    fn.restype=ctypes.c_int
    def convert(raw,threads):
        if len(raw)%144:raise ValueError('Unaligned Q4_K chunk')
        result=ctypes.create_string_buffer(len(raw)//144*136)
        if fn(raw,result,len(raw)//144,threads):raise ValueError('Nonfinite/out-of-range expert block')
        return result.raw
    return convert

def durable_json(path,state):
    # Windows readers may briefly omit FILE_SHARE_DELETE. Retain one owned
    # pending receipt and retry its atomic rename, rather than losing a run.
    pending=path.with_name(path.name+'.next')
    with pending.open('w',encoding='utf8') as f:
        json.dump(state,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    deadline=time.monotonic()+5
    while True:
        try:os.replace(pending,path);return
        except PermissionError:
            if time.monotonic()>deadline:raise
            time.sleep(.05)

def gpu_holds(path):
    try:
        stat=path.stat();stamp=(stat.st_mtime_ns,stat.st_size)
        if stat.st_size>16<<20:return ['GPU ledger exceeds its 16 MiB observation budget; conversion paused']
        cached=_hold_cache.get(path)
        if cached is not None and cached[0]==stamp:holds=cached[1]
        else:
            holds=active_holds(path.read_text(encoding='utf-8-sig'))
            _hold_cache[path]=(stamp,holds)
        if _conversion_hold:
            task,identity=_conversion_hold
            own=[line for line in holds if line.split()[:3]==['ACTIVE:',task,identity+';']]
            if len(own)!=1:return ['Conversion reservation is absent; conversion paused']
            holds=[line for line in holds if line not in own]
        return holds
    except OSError:
        # A concurrent ledger writer may temporarily exclude readers on
        # Windows. Unknown ownership must pause all source reads.
        return ['GPU ledger temporarily unreadable; conversion paused']

def worker(args):
    import shutil
    import psutil
    waiting=False
    while gpu_holds(args.hold_file):
        if (args.runtime_root/'requant.stop').exists():raise InterruptedError('Cooperative conversion stop')
        if not waiting:print('WAIT_GPU: no source reads until the shared GPU ledger is clear',flush=True);waiting=True
        time.sleep(.25)
    expert_layout=getattr(args,'expert_layout',CANONICAL)
    h,tensors,header,total,data_start,a=layout(args.requant_gguf,expert_layout)
    src_stat=args.requant_gguf.stat()
    identity=dict(path=str(args.requant_gguf.resolve()),size=src_stat.st_size,mtime_ns=src_stat.st_mtime_ns,
                  header_sha256=hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest(),
                  library_sha256=hashlib.sha256(args.quants_library.read_bytes()).hexdigest(),
                  output=str(args.out.resolve()),expected_bytes=total)
    if expert_layout!=CANONICAL:identity['expert_layout']=expert_layout
    journal=args.runtime_root/('requant-progress.json' if expert_layout==CANONICAL else 'requant-native-progress.json')
    partial=args.out.with_name(args.out.name+'.partial')
    if args.out.exists():raise FileExistsError('Refusing existing completed output')
    state={'identity':identity,'label':LABEL,'status':'RUNNING','completed':[],'started':time.time()}
    if journal.exists():
        if not args.resume:raise FileExistsError('Resume receipt exists; use --resume')
        state=json.loads(journal.read_text())
        if state['identity']!=identity:raise ValueError('Resume identity changed')
        if not partial.exists():raise ValueError('Resume partial missing')
    elif partial.exists():raise FileExistsError('Unowned partial exists')
    remaining=total-(partial.stat().st_size if partial.exists() else 0)
    if shutil.disk_usage(args.out.parent).free<remaining+20*2**30:raise ValueError('Output volume needs output plus 20 GiB reserve')
    convert=converter(args.quants_library)
    def checkpoint(**kw):
        state.update(kw,updated=time.time());durable_json(journal,state)
    def idle():
        while True:
            if (args.runtime_root/'requant.stop').exists():raise InterruptedError('Cooperative conversion stop')
            if psutil.virtual_memory().available<args.runtime_floor_gib*2**30:raise MemoryError('Physical RAM floor breached')
            holds=gpu_holds(args.hold_file)
            if not holds:return
            if state.get('status')!='PAUSED_GPU_HOLD' or state.get('holds')!=holds:
                checkpoint(status='PAUSED_GPU_HOLD',holds=holds)
            time.sleep(.25)
    checkpoint()
    with args.requant_gguf.open('rb',buffering=0) as source,partial.open('r+b' if partial.exists() else 'x+b',buffering=0) as output:
        whole_hash=hashlib.sha256()
        if state['completed']:
            # Do not trust an interrupted or externally altered partial.
            output.seek(0)
            idle()
            observed_header=output.read(len(header))
            if observed_header!=header:raise ValueError('Resume header changed')
            whole_hash.update(observed_header)
            for row in state['completed']:
                t=tensors[row['index']];output.seek(len(header)+t['offset'])
                digest=hashlib.sha256();left=nbytes(t)
                while left:
                    idle()
                    block=output.read(min(left,16<<20))
                    if not block:raise ValueError('Truncated completed tensor')
                    digest.update(block);whole_hash.update(block);left-=len(block)
                if digest.hexdigest()!=row['output_sha256']:raise ValueError('Completed tensor hash mismatch')
                padding=output.read(align(nbytes(t),a)-nbytes(t))
                if padding!=bytes(align(nbytes(t),a)-nbytes(t)):raise ValueError('Completed tensor padding changed')
                whole_hash.update(padding)
        else:
            output.write(header);output.flush();os.fsync(output.fileno())
            whole_hash.update(header)
        for index in range(len(state['completed']),len(tensors)):
            old,t=h['tensors'][index],tensors[index]
            source.seek(data_start+old['offset']);output.seek(len(header)+t['offset'])
            left=nbytes(old);done=0;input_hash=hashlib.sha256();output_hash=hashlib.sha256()
            started=time.monotonic();last=started
            def chunks():
                nonlocal left
                while left:
                    idle()
                    size=min(left,65536*144 if t['type']==39 else 16<<20)
                    raw=source.read(size)
                    if len(raw)!=size:raise ValueError('Short source tensor read')
                    input_hash.update(raw);left-=size
                    yield convert(raw,args.threads) if t['type']==39 else raw
            converted=(requant_chunks(source,old,t,convert,args.threads,idle,input_hash)
                if expert_layout==FLASHINFER and t['type']==39 else chunks())
            for block in converted:
                idle()
                if shutil.disk_usage(args.out.parent).free<len(block)+20*2**30:raise ValueError('Output volume reached the 20 GiB reserve')
                output.write(block);output_hash.update(block);whole_hash.update(block);done+=len(block)
                if time.monotonic()-last>5:
                    checkpoint(status='RUNNING',tensor=t['name'],tensor_index=index,tensor_bytes=done,
                               output_position=output.tell(),expected_bytes=total)
                    print(f'{index}/{len(tensors)} {t["name"]} {done}/{nbytes(t)} bytes',flush=True);last=time.monotonic()
            if done!=nbytes(t):raise ValueError('Tensor output size mismatch')
            padding=bytes(align(done,a)-done)
            output.write(padding);whole_hash.update(padding);output.flush();os.fsync(output.fileno())
            state['completed'].append(dict(index=index,name=t['name'],input_sha256=input_hash.hexdigest(),
                                          output_sha256=output_hash.hexdigest(),seconds=time.monotonic()-started))
            checkpoint(status='RUNNING',tensor=t['name'],tensor_index=index,tensor_bytes=done,output_position=output.tell())
        if output.tell()!=total or partial.stat().st_size!=total:raise ValueError('Final output size mismatch')
    expected={t['name']:dict(type=t['type'],shape=t['shape']) for t in tensors}
    idle()
    if tensor_map(read_header(partial))!=expected:raise ValueError('Converted header mismatch')
    digest=whole_hash
    if not getattr(args,'defer_readback',False):
        digest=hashlib.sha256()
        checkpoint(status='HASHING_OUTPUT')
        with partial.open('rb',buffering=0) as f:
            while True:
                idle()
                block=f.read(16<<20)
                if not block:break
                digest.update(block)
        if digest.digest()!=whole_hash.digest():raise ValueError('Output readback differs from written-byte hash')
    partial.rename(args.out)
    checkpoint(status='COMPLETE',sha256=digest.hexdigest(),readback_verified=not getattr(args,'defer_readback',False),
               hash_scope='full file bytes including header and padding; streaming write hash; completed prefix read-verified on resume',finished=time.time())
    print(json.dumps({'status':'COMPLETE','sha256':digest.hexdigest(),'bytes':total,'label':LABEL}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requant-gguf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--hold-file',type=Path,required=True)
    p.add_argument('--quants-library',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--expert-layout',choices=(CANONICAL,FLASHINFER),default=CANONICAL,
                   help='Explicit kernel-native byte layout; lossy Q4 speed proxy remains unshipped')
    p.add_argument('--host-budget-gib',type=float,required=True)
    p.add_argument('--runtime-floor-gib',type=float,default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--defer-readback',action='store_true',help='Hash all bytes during writing; require stored-byte readback during the later verified copy before timing')
    p.add_argument('--conversion-lease',type=Path,help='Existing live ancestor reservation held through conversion and readback')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not 1<=args.threads<=16 or not math.isfinite(args.host_budget_gib) or not 0<args.host_budget_gib<=8 or not math.isfinite(args.runtime_floor_gib) or args.runtime_floor_gib<.5:
        p.error('Require threads 1..16, finite host budget <=8 GiB, finite RAM floor >=0.5 GiB')
    if args.requant_gguf.resolve()==args.out.resolve():p.error('Source is read-only; distinct output required')
    args.runtime_root.mkdir(parents=True,exist_ok=True);args.out.parent.mkdir(parents=True,exist_ok=True)
    import psutil
    if args.worker:
        if os.environ.get('SLSTACK_REQUANT_PARENT')!=str(psutil.Process().ppid()):raise ValueError('Internal worker only')
        with (args.runtime_root/'requant.lock').open('a+b') as lock:
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.conversion_lease:
                from slstack.lease import gpu_lease
                row=json.loads(args.conversion_lease.read_text(encoding='utf-8'))
                cfg=dict(task=row['task'],hold_file=str(args.hold_file),
                         _gpu_parent_lease=str(args.conversion_lease),
                         host_budget_gib=args.host_budget_gib,min_available_gib=args.runtime_floor_gib)
                with gpu_lease(cfg) as identity:
                    global _conversion_hold
                    _conversion_hold=(row['task'],identity)
                    worker(args)
            else:worker(args)
        return 0
    deadline=time.monotonic()+21600
    reported=0
    while psutil.virtual_memory().available<(args.host_budget_gib+args.runtime_floor_gib)*2**30:
        if time.monotonic()>deadline or (args.runtime_root/'requant.stop').exists():raise MemoryError('Cannot admit host budget plus physical floor')
        if time.monotonic()-reported>30:
            print('WAIT_RAM: conversion has not started; waiting for budget plus physical floor',flush=True);reported=time.monotonic()
        time.sleep(.25)
    from slstack.guard import supervise
    env=dict(os.environ,SLSTACK_REQUANT_PARENT=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1',
             TEMP=str(args.runtime_root),TMP=str(args.runtime_root),TMPDIR=str(args.runtime_root))
    receipt=supervise(dict(command=[sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],
        cwd=str(ROOT),env=env,host_budget_gib=args.host_budget_gib,job_commit_gib=args.host_budget_gib,
        min_available_gib=args.runtime_floor_gib,timeout_s=21600,owner_pid=os.getpid(),
        owner_created=psutil.Process().create_time(),runtime_root=str(args.runtime_root),
        stop=str(args.runtime_root/'requant.stop'),log=str(args.runtime_root/'requant.log'),
        receipt=str(args.runtime_root/'requant-memory.json')))
    return receipt if isinstance(receipt,int) else (receipt.get('exit_code') or bool(receipt.get('reason')))

if __name__=='__main__':sys.exit(main())
