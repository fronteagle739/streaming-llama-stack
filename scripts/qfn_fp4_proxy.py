"""Explicitly lossy, CPU-only QFN Q2_0 -> DSV FlashInfer MXFP4 proxy.

Run under the existing memory guard; write/verify also require its exclusive
shared I/O lease. One expert in RAM, one writer, durable layer-boundary resume.
Original GGUF/dense/MTP/vision/tokenizer payloads and metadata are preserved.
"""
import argparse
import ctypes as C
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dsv41_fp4_layout import FLASHINFER,pack_expert,unpack_expert
from scripts.dsv41_requant import durable_json

SOURCE_SHA='dee4ae542817d73e4f7a56c448ef97cb7bb47ef92b8b2c5b7721ec0e8c25f76c'
OLD_STRIDE=1382400
TENSOR_BYTES=870400
STRIDE=2611200
LAYERS=48
EXPERTS=512
ALIGN=2<<20
CHUNK=16<<20
SHAPES={'gate':(640,2560),'up':(640,2560),'down':(2560,640)}
PROGRESS=None

def progress(doing,latest,next_step):
    line=f'{dt.datetime.now(dt.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
    print(line,flush=True)
    with PROGRESS.open('a',encoding='utf8') as f:f.write(line+'\n')

def align(x):return (x+ALIGN-1)//ALIGN*ALIGN
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()

def header(path):
    with path.open('rb',buffering=0) as f:raw=f.read(ALIGN)
    magic,version,count,size=struct.unpack_from('<8sIIQ',raw)
    if magic!=b'BASALTMF' or version!=1 or count!=18 or size!=path.stat().st_size:raise ValueError('Invalid basalt header')
    rows=[];end=ALIGN;names=set()
    for i in range(count):
        name,offset,length,digest=struct.unpack_from('<64sQQ32s',raw,64+i*128)
        name=name.rstrip(b'\0').decode('ascii')
        if name in names or offset<end or offset%ALIGN or offset+length>size:raise ValueError('Invalid section '+name)
        names.add(name);end=offset+length
        rows.append(dict(name=name,offset=offset,bytes=length,sha256=digest.hex()))
    if end!=size:raise ValueError('Unexpected file tail')
    return raw,rows

def read_section(path,row):
    with path.open('rb',buffering=0) as f:f.seek(row['offset']);raw=f.read(row['bytes'])
    if len(raw)!=row['bytes'] or sha(raw)!=row['sha256']:raise ValueError('Descriptor hash differs '+row['name'])
    return raw

def layout_spec():
    tensors={}
    for i,(name,(n,k)) in enumerate(SHAPES.items()):
        tensors[name]=dict(n=n,k=k,offset=i*TENSOR_BYTES,bytes=TENSOR_BYTES,
                          values_offset=i*TENSOR_BYTES,values_bytes=n*k//2,
                          scales_offset=i*TENSOR_BYTES+n*k//2,scales_bytes=n*k//32,
                          values_shape=[n,k//2],scales_shape=[k//128,n,4])
    return dict(version=1,expert_format='mxfp4',expert_layout=FLASHINFER,
                container_layout='qfn-expert-major-gate-up-down-values-scales-v1',
                ggml_type_id=39,layers=LAYERS,experts_per_layer=EXPERTS,
                expert_stride=STRIDE,layer_stride=EXPERTS*STRIDE,tensors=tensors,
                nibble_order='adjacent K: even low nibble, odd high nibble',
                scale_order='[K/128][N][4] C contiguous; E8M0 per 32 weights',
                scale_value='2^(e-127), E2M1={0,.5,1,1.5,2,3,4,6} with sign bit',
                format_precedence='qfn_fp4_proxy.storage and experts.layout v6 describe ONLY experts; original tensors describe unchanged gguf; mtp stays original Q2_0')

def plan(source):
    raw,rows=header(source);by={r['name']:r for r in rows}
    if by['experts']['bytes']!=LAYERS*EXPERTS*OLD_STRIDE:raise ValueError('Q2 expert geometry differs')
    original_layout=read_section(source,by['experts.layout']).decode()
    lines=original_layout.splitlines()
    if not lines[0].startswith('# experts v5'):raise ValueError('Expected original experts v5')
    data=[list(map(int,line.split())) for line in lines[1:] if line.strip()]
    if data!=[[i,42,42,i*EXPERTS*OLD_STRIDE,OLD_STRIDE] for i in range(LAYERS)]:raise ValueError('Unexpected source expert layout')
    meta=json.loads(read_section(source,by['meta']))
    if any(k in meta for k in ('qfn_fp4_proxy','expert_format','expert_layout','speed_proxy','source_sha256')):raise ValueError('Already marked/conflicting metadata')
    spec=layout_spec()
    additions=dict(expert_format='mxfp4',expert_layout=FLASHINFER,speed_proxy=True,source_sha256=SOURCE_SHA,
                   qfn_fp4_proxy=dict(label='LOSSY SPEED-PROXY; not a default/release model',source_sha256=SOURCE_SHA,
                       source_experts_sha256=by['experts']['sha256'],source_experts_layout=original_layout,
                       storage=spec,quantizer='ggml quantize_row_mxfp4_ref; strict first-code ties; fused Q2 lookup verified against F32 reference',
                       unchanged_sections=[r['name'] for r in rows if r['name'] not in ('experts','experts.layout','meta')]))
    new_meta={**meta,**additions}
    if any(new_meta[k]!=v for k,v in meta.items()):raise ValueError('Original metadata changed')
    layout=('# experts v6 qfn-expert-major-gate-up-down-values-scales-v1 '+FLASHINFER+'\n'+
            '\n'.join(f'{i} 39 39 {i*EXPERTS*STRIDE} {STRIDE}' for i in range(LAYERS))+'\n').encode()
    replacements={'meta':encoded(new_meta),'experts.layout':layout}
    planned=[];end=ALIGN
    for old in rows:
        name=old['name'];size=LAYERS*EXPERTS*STRIDE if name=='experts' else len(replacements[name]) if name in replacements else old['bytes']
        row=dict(old,offset=align(end),bytes=size)
        if name in replacements:row['sha256']=sha(replacements[name])
        elif name=='experts':row['sha256']='00'*32
        planned.append(row);end=row['offset']+size
    h=bytearray(raw);struct.pack_into('<Q',h,16,end)
    for i,row in enumerate(planned):struct.pack_into('<64sQQ32s',h,64+i*128,row['name'].encode(),row['offset'],row['bytes'],bytes.fromhex(row['sha256']))
    result=dict(source_sha256=SOURCE_SHA,source_bytes=source.stat().st_size,source_header_sha256=sha(raw),
                source_mtime_ns=source.stat().st_mtime_ns,
                expert_bytes=LAYERS*EXPERTS*STRIDE,other_section_bytes=sum(r['bytes'] for r in planned if r['name']!='experts'),
                target_bytes=end,source_sections=rows,target_sections=planned,storage=spec,
                descriptor_diff={'experts.layout':{'old':original_layout,'new':layout.decode()},'meta':{'added_fields':additions,'modified_original_fields':[]}},
                dense_optional=dense_budget(source,by))
    return result,bytes(h),replacements

def dense_budget(source,by):
    # Dense index v3 stores all logical tensor shapes, including gguf aliases.
    text=read_section(source,by['dense.index']).decode();metadata=json.loads(read_section(source,by['meta']))
    counts={};total=0;eligible=0;matrix_elements=0;q8_bytes=0;fp8_scales=0;matrix_count=0;unaligned=[]
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        p=line.split();tensor=metadata['tensors'][p[0]];ne=tensor['shape'];kind=tensor['type_id']
        count=1
        for n in ne:
            if n:count*=n
        counts[str(kind)]=counts.get(str(kind),0)+count
        total+=count
        if len(ne)==2 and ne[0]%128==0 and ne[1]%128==0:eligible+=count
        if len(ne)==2 and ne[1]>1:
            matrix_count+=1;matrix_elements+=count
            if ne[0]%32:unaligned.append(dict(name=p[0],shape=ne,source_type=kind))
            q8_bytes+=((ne[0]+31)//32)*ne[1]*34
            fp8_scales+=((ne[0]+31)//32)*((ne[1]+31)//32)
    return dict(logical_elements=total,logical_elements_by_source_ggml_type=counts,
                matrix_count=matrix_count,matrix_elements=matrix_elements,
                hypothetical_matrix_q8_0_padded_bytes=q8_bytes,q8_unaligned_matrices=unaligned,
                hypothetical_matrix_fp8_values_bytes=matrix_elements,
                fp8_matrix_32x32_scale_bytes=fp8_scales,
                geometry_128x128_elements=eligible,converted=False,
                kernel_availability='QFN has existing Q8_0 ggml kernels; DSV native FP8 dense bridge exists but QFN has no metadata dispatch for it. Dense is unchanged; no conversion in this task.')

class Native:
    def __init__(self,path):
        self.lib=C.CDLL(str(path.resolve()))
        self.lib.qfn_dequant.argtypes=[C.c_void_p,C.c_void_p,C.c_int64]
        self.lib.qfn_quant.argtypes=[C.c_void_p,C.c_void_p,C.c_int64]
        self.lib.qfn_convert.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_int]
        self.lib.qfn_convert.restype=C.c_int
    def convert(self,raw,threads):
        out=C.create_string_buffer(len(raw)//18*34)
        if self.lib.qfn_convert(raw,out,len(raw)//18,threads):raise ValueError('Nonfinite or invalid expert block')
        return out.raw

def dequant_q2(raw):
    import numpy as np
    a=np.frombuffer(raw,dtype=np.uint8).reshape(-1,18)
    d=a[:,:2].copy().view('<f2').reshape(-1).astype(np.float32)
    codes=((a[:,2:,None]>>np.array([0,2,4,6],dtype=np.uint8))&3).reshape(-1,64)
    return ((codes.astype(np.float32)-1)*d[:,None]).reshape(-1)

def decode_fp4(values,scales,n,k):
    import numpy as np
    raw=unpack_expert(values,scales,n,k)
    a=np.frombuffer(raw,dtype=np.uint8).reshape(-1,17)
    code=np.concatenate((a[:,1:]&15,a[:,1:]>>4),axis=1)
    table=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
    return (table[code]*np.ldexp(np.ones(len(a),dtype=np.float32),a[:,0].astype(np.int32)-127)[:,None]).reshape(-1)

def prove(args,info,native):
    import numpy as np
    # Exhaust all finite f16 scales, with deterministic 2-bit code patterns.
    halves=np.arange(65536,dtype='<u2');halves=halves[(halves&0x7c00)!=0x7c00]
    blocks=np.empty((len(halves),18),dtype=np.uint8);blocks[:,:2]=halves.view('u1').reshape(-1,2)
    rng=np.random.default_rng(20261101);blocks[:,2:]=rng.integers(0,256,size=(len(halves),16),dtype=np.uint8)
    raw=blocks.tobytes();x=dequant_q2(raw);cpu=np.empty_like(x)
    native.lib.qfn_dequant(raw,cpu.ctypes.data,len(x))
    if not np.array_equal(cpu.view('u4'),x.view('u4')):raise ValueError('Finite-half Q2 CPU proof failed')
    fused=native.convert(raw,args.threads);reference=C.create_string_buffer(len(fused))
    native.lib.qfn_quant(x.ctypes.data,reference,len(x))
    if fused!=reference.raw:raise ValueError('Finite-half GGML quantizer proof failed')
    # A malformed/nonfinite source must fail loudly, never emit substitutes.
    bad=bytes([0,124])+bytes(16);out=C.create_string_buffer(34)
    if native.lib.qfn_convert(bad,out,1,args.threads)==0:raise ValueError('Nonfinite source accepted')
    progress('synthetic quantizer proof',f'{len(halves)} finite f16 scales/{len(x)} weights CPU exact; MXFP4 bytes exact; nonfinite rejected','real expert samples')
    results=[];expertrow=next(r for r in info['source_sections'] if r['name']=='experts')
    samples=[(0,0),(23,255),(47,511)]
    with args.source.open('rb',buffering=0) as f:
        for layer,expert in samples:
            f.seek(expertrow['offset']+(layer*EXPERTS+expert)*OLD_STRIDE);raw=f.read(OLD_STRIDE)
            begin=time.monotonic()
            for i,(name,(n,k)) in enumerate(SHAPES.items()):
                part=raw[i*460800:(i+1)*460800];x=dequant_q2(part)
                cpu=np.empty_like(x);native.lib.qfn_dequant(part,cpu.ctypes.data,len(x))
                if not np.array_equal(cpu.view('u4'),x.view('u4')):raise ValueError('Q2_0 GGML CPU mismatch')
                canonical=native.convert(part,args.threads)
                reference=C.create_string_buffer(len(canonical));native.lib.qfn_quant(x.ctypes.data,reference,len(x))
                if canonical!=reference.raw:raise ValueError('Fused conversion differs from GGML F32 reference')
                values,scales=pack_expert(canonical,n,k)
                if unpack_expert(values,scales,n,k)!=canonical:raise ValueError('DSV layout roundtrip failed')
                y=decode_fp4(values,scales,n,k)
                rms=float(np.sqrt(np.mean((y.astype('f8')-x)**2))/np.sqrt(np.mean(x.astype('f8')**2)))
                results.append(dict(layer=layer,expert=expert,tensor=name,relative_rms=rms,max_abs=float(np.max(np.abs(y-x))),
                                    q2_cpu_bit_exact=True,ggml_quantizer_byte_exact=True,canonical_sha256=sha(canonical)))
            progress('sample proof',f'layer{layer}/expert{expert} all3 tensors CPU/quantizer/layout PASS in{time.monotonic()-begin:.3f}s','next sample/write')
    return dict(samples=results,finite_half_scales=len(halves),synthetic_weights=len(halves)*64,
                synthetic_cpu_bit_exact=True,synthetic_quantizer_byte_exact=True,nonfinite_rejected=True,
                storage=info['storage'],native_library_sha256=sha(args.library.read_bytes()))

def check_runtime(args,heavy=False):
    import psutil
    if psutil.virtual_memory().available<6*2**30:raise MemoryError('Available physical RAM below 6GiB')
    if psutil.Process().memory_info().rss>4*2**30:raise MemoryError('4GiB RSS budget exceeded')
    if (args.runtime/'STOP').exists():raise InterruptedError('Owner cooperative STOP')
    if heavy:
        if getattr(args,'timing_marker',None) and args.timing_marker.exists():
            with args.timing_marker.open(encoding='utf-8-sig') as marker:
                released=marker.read(1024).split()[:1]==['RELEASED']
            if not released:
                raise InterruptedError('Coordinator timing marker active; retain durable checkpoint')
        from slstack.lease import active_holds,read_hold_text
        holds=active_holds(read_hold_text(args.hold_file))
        if len(holds)!=1 or not holds[0].startswith('ACTIVE: QFNFP4A '):raise RuntimeError('Missing exclusive QFNFP4A heavy-I/O lease: '+str(holds))

def pad(f,end):
    while f.tell()<end:f.write(bytes(min(CHUNK,end-f.tell())))

def write_pack(args,info,h,replacements,native):
    args.out.parent.mkdir(parents=True,exist_ok=True)
    journal=args.runtime/'pack-checkpoint.json';partial=args.out.with_name(args.out.name+'.partial')
    identity=dict(source_sha256=SOURCE_SHA,plan_sha256=sha(encoded(info)),library_sha256=sha(args.library.read_bytes()))
    state=dict(identity=identity,completed_sections=[],layers=[],started=time.time())
    if args.out.exists():raise FileExistsError('Completed output exists; use verify')
    if journal.exists():
        if not args.resume:raise FileExistsError('Checkpoint exists; use --resume')
        state=json.loads(journal.read_text())
        if state['identity']!=identity or not partial.exists():raise ValueError('Resume source/layout/library identity changed')
    elif partial.exists():raise FileExistsError('Unowned partial exists')
    remaining=info['target_bytes']-(partial.stat().st_size if partial.exists() else 0)
    if shutil.disk_usage(args.out.parent).free<remaining+50_000_000_000:raise OSError('Need complete target plus 50GB reserve')
    check_runtime(args,True)
    with args.source.open('rb',buffering=0) as src,partial.open('r+b' if partial.exists() else 'x+b',buffering=0) as dst:
        if state['completed_sections'] or state['layers']:
            # Rehash every durable completed extent before accepting resume.
            for r in state['completed_sections']+state['layers']:
                dst.seek(r['offset']);dig=hashlib.sha256();left=r['bytes']
                while left:
                    check_runtime(args,True);b=dst.read(min(left,CHUNK))
                    if not b:raise ValueError('Short resume extent')
                    dig.update(b);left-=len(b)
                if dig.hexdigest()!=r['sha256']:raise ValueError('Resume extent hash differs')
        dst.seek(0);dst.write(h)
        for old,new in zip(info['source_sections'],info['target_sections']):
            name=old['name']
            if any(r['name']==name for r in state['completed_sections']):continue
            dst.seek(ALIGN if not state['completed_sections'] else state['completed_sections'][-1]['offset']+state['completed_sections'][-1]['bytes'])
            pad(dst,new['offset']);digest=hashlib.sha256()
            if name=='experts':
                source_digest=hashlib.sha256()
                for layer in range(LAYERS):
                    if layer<len(state['layers']):
                        dst.seek(new['offset']+layer*EXPERTS*STRIDE);left=EXPERTS*STRIDE
                        while left:
                            check_runtime(args,True);b=dst.read(min(left,CHUNK));digest.update(b);left-=len(b)
                        src.seek(old['offset']+layer*EXPERTS*OLD_STRIDE);left=EXPERTS*OLD_STRIDE
                        while left:
                            check_runtime(args,True);b=src.read(min(left,CHUNK))
                            if not b:raise ValueError('Short resumed source layer')
                            source_digest.update(b);left-=len(b)
                        continue
                    src.seek(old['offset']+layer*EXPERTS*OLD_STRIDE);dst.seek(new['offset']+layer*EXPERTS*STRIDE)
                    ld=hashlib.sha256();source_ld=hashlib.sha256()
                    for expert in range(EXPERTS):
                        check_runtime(args,True)
                        if shutil.disk_usage(args.out.parent).free<info['target_bytes']-dst.tell()+50_000_000_000:raise OSError('Disk reserve breached')
                        raw=src.read(OLD_STRIDE)
                        if len(raw)!=OLD_STRIDE:raise ValueError('Short Q2 expert')
                        source_ld.update(raw);source_digest.update(raw)
                        for i,(part,(n,k)) in enumerate(SHAPES.items()):
                            canonical=native.convert(raw[i*460800:(i+1)*460800],args.threads)
                            values,scales=pack_expert(canonical,n,k)
                            dst.write(values);dst.write(scales);ld.update(values);ld.update(scales);digest.update(values);digest.update(scales)
                    dst.flush();os.fsync(dst.fileno())
                    state['layers'].append(dict(layer=layer,offset=new['offset']+layer*EXPERTS*STRIDE,bytes=EXPERTS*STRIDE,sha256=ld.hexdigest(),source_sha256=source_ld.hexdigest()))
                    durable_json(journal,state)
                    progress('requantized layer',f'{layer+1}/48; output experts{(layer+1)*EXPERTS*STRIDE}B','next layer/unchanged sections')
                new['sha256']=digest.hexdigest()
                if source_digest.hexdigest()!=old['sha256']:raise ValueError('Source experts SHA differs')
            elif name in replacements:
                data=replacements[name];dst.write(data);digest.update(data)
            else:
                src.seek(old['offset']);left=old['bytes']
                while left:
                    check_runtime(args,True);data=src.read(min(left,CHUNK))
                    if not data:raise ValueError('Short source section')
                    dst.write(data);digest.update(data);left-=len(data)
                if digest.hexdigest()!=old['sha256']:raise ValueError('Source section SHA differs '+name)
            dst.flush();os.fsync(dst.fileno())
            entry=dict(new)
            entry['sha256']=digest.hexdigest()
            state['completed_sections'].append(entry);durable_json(journal,state)
            progress('section written',f'{name} {new["bytes"]}B; file position{dst.tell()}','next section/final readback')
        # A fully durable resumed pack leaves the cursor after the header.
        # Extents were rehashed above; validate actual EOF, not that cursor.
        if dst.seek(0,os.SEEK_END)!=info['target_bytes']:raise ValueError('Output size differs')
        for row in info['target_sections']:
            row['sha256']=next(r['sha256'] for r in state['completed_sections'] if r['name']==row['name'])
    return partial,state

def verify(args,info,path,finalize=False):
    h,rows=header(path)
    if path.stat().st_size!=info['target_bytes']:raise ValueError('Readback size differs')
    expected=info['target_sections'];whole=hashlib.sha256();digests=[hashlib.sha256() for _ in rows]
    with path.open('rb',buffering=0) as f:
        pos=0;last=time.monotonic()
        while True:
            check_runtime(args,True);data=f.read(CHUNK)
            if not data:break
            whole.update(data);end=pos+len(data)
            for r,d in zip(rows,digests):
                lo=max(pos,r['offset']);hi=min(end,r['offset']+r['bytes'])
                if hi>lo:d.update(memoryview(data)[lo-pos:hi-pos])
            pos=end
            if time.monotonic()-last>30:progress('sequential written-file readback',f'{pos}/{info["target_bytes"]}B','finish18 hashes');last=time.monotonic()
    observed=[]
    for actual,want,d in zip(rows,expected,digests):
        for k in ('name','offset','bytes'):
            if actual[k]!=want[k]:raise ValueError('Readback extent differs '+str(actual))
        got=d.hexdigest()
        if (finalize or want['sha256']!='00'*32) and got!=want['sha256']:raise ValueError('Readback SHA differs '+want['name'])
        if not finalize and got!=actual['sha256']:raise ValueError('Header SHA differs '+want['name'])
        observed.append(dict(actual,sha256=got,source_sha256=next(r['sha256'] for r in info['source_sections'] if r['name']==actual['name']),changed=actual['name'] in ('experts','experts.layout','meta')))
    if finalize:
        patched=bytearray(h)
        for i,r in enumerate(observed):struct.pack_into('<32s',patched,64+128*i+80,bytes.fromhex(r['sha256']))
        with path.open('r+b',buffering=0) as f:f.write(patched);f.flush();os.fsync(f.fileno())
        # Full second read validates final header, complete file and all hashes.
        return verify(args,info,path,False)
    # Read descriptors again and verify original fields recursively by plan identity.
    by={r['name']:r for r in rows};oldby={r['name']:r for r in info['source_sections']}
    oldmeta=json.loads(read_section(args.source,oldby['meta']));newmeta=json.loads(read_section(path,by['meta']))
    if any(newmeta.get(k)!=v for k,v in oldmeta.items()):raise ValueError('Original metadata field differs')
    if newmeta['qfn_fp4_proxy']['storage']!=layout_spec():raise ValueError('Storage descriptor differs')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=whole.hexdigest(),sections=observed,
                immutable_sections_verified=15,original_metadata_fields_preserved=True,storage=layout_spec())

def main():
    global PROGRESS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','prove','write','verify'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--source-receipt',type=Path)
    p.add_argument('--timing-marker',type=Path)
    p.add_argument('--progress-log',type=Path,required=True);p.add_argument('--hold-file',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=range(1,5),default=4);p.add_argument('--resume',action='store_true')
    args=p.parse_args();PROGRESS=args.progress_log;args.runtime.mkdir(parents=True,exist_ok=True)
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    check_runtime(args);started=time.monotonic();info,h,replacements=plan(args.source)
    if args.mode in ('prove','write'):
        if args.source_receipt is None:raise ValueError('Verified whole-source --source-receipt required')
        admission=json.loads(args.source_receipt.read_text())
        if not (admission.get('verified') and admission.get('sha256')==SOURCE_SHA and admission.get('size')==info['source_bytes'] and
                Path(admission['path']).resolve()==args.source.resolve() and admission.get('sections')==18 and
                admission.get('guard_exit')==0 and admission.get('cleanup_confirmed')):raise ValueError('Whole-source guarded admission differs')
    durable_json(args.runtime/'plan.json',info)
    progress('exact size budget',f'target={info["target_bytes"]}B experts={info["expert_bytes"]}B other sections={info["other_section_bytes"]}B Dfree={shutil.disk_usage(args.runtime).free}B','plan/prove/leased write')
    if args.mode=='prove':durable_json(args.runtime/'proof.json',prove(args,info,Native(args.library)))
    elif args.mode=='write':
        proof=json.loads((args.runtime/'proof.json').read_text())
        if proof['native_library_sha256']!=sha(args.library.read_bytes()):raise ValueError('Proof library differs')
        partial,state=write_pack(args,info,h,replacements,Native(args.library))
        report=verify(args,info,partial,True)
        # Confirm independent errors on the written artifact, not only RAM output.
        report['roundtrip']=prove_written(args,info,partial)
        if args.source.stat().st_mtime_ns!=info['source_mtime_ns']:raise ValueError('Read-only source changed during conversion')
        os.replace(partial,args.out);report['path']=str(args.out)
        report.update(elapsed_seconds=time.monotonic()-started,source_sha256=SOURCE_SHA)
        durable_json(args.runtime/'readback.json',report)
    elif args.mode=='verify':durable_json(args.runtime/'verify.json',verify(args,info,args.out))
    progress(args.mode+' finished',f'elapsed={time.monotonic()-started:.3f}s','review evidence/commit/checkpoint')

def prove_written(args,info,path):
    import numpy as np
    results=[];old=next(r for r in info['source_sections'] if r['name']=='experts');new=next(r for r in info['target_sections'] if r['name']=='experts')
    with args.source.open('rb',buffering=0) as src,path.open('rb',buffering=0) as dst:
        for layer,expert in [(0,0),(23,255),(47,511)]:
            index=layer*EXPERTS+expert;src.seek(old['offset']+index*OLD_STRIDE);a=src.read(OLD_STRIDE)
            dst.seek(new['offset']+index*STRIDE);b=dst.read(STRIDE)
            for i,(name,(n,k)) in enumerate(SHAPES.items()):
                x=dequant_q2(a[i*460800:(i+1)*460800]);part=b[i*TENSOR_BYTES:(i+1)*TENSOR_BYTES]
                y=decode_fp4(part[:n*k//2],part[n*k//2:],n,k)
                error=float(np.sqrt(np.mean((y.astype('f8')-x)**2))/np.sqrt(np.mean(x.astype('f8')**2)))
                results.append(dict(layer=layer,expert=expert,tensor=name,relative_rms=error,max_abs=float(np.max(np.abs(y-x)))))
    progress('disk FP4 sample roundtrip',f'{len(results)} tensors independently decoded; RMS range{min(r["relative_rms"] for r in results):.9f}..{max(r["relative_rms"] for r in results):.9f}','final receipt')
    return results

if __name__=='__main__':main()
