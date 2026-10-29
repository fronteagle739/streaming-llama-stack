"""E19: pinned official components and native operators, identical resident inputs.

Run inside the shared GPU lease and paging/RSS guard. This is a component
comparison, not a model accuracy or end-to-end throughput qualification.
"""
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scratch-root',type=Path,required=True)
parser.add_argument('--native',type=Path,required=True)
parser.add_argument('--rows',type=int,nargs='+',default=[2541,65536])
parser.add_argument('--components',nargs='+',default=['norm','hc_split','dense','index','attention'])
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--attention-variant',choices=('0','2'),default='0')
parser.add_argument('--cubin',type=Path)
args=parser.parse_args()
root=args.scratch_root.resolve()
sys.path[:0]=[str(root/'tilelang-deps'),str(root/'python-packages')]
os.environ.update(TILELANG_CACHE_DIR=str(root/'tilelang-cache'),TVM_FFI_CACHE_DIR=str(root/'tvm-cache'),MAX_JOBS='2')
os.environ['LLAMA_DSV41_TILELANG_ATTENTION']=args.attention_variant
if args.attention_variant=='2':
    assert args.cubin and args.cubin.resolve().is_relative_to(root)
    os.environ['LLAMA_DSV41_TILELANG_CUBIN']=str(args.cubin.resolve())
upstream=Path(__file__).with_name('upstream')
manifest=json.loads((upstream/'manifest.json').read_text())
for item in manifest['files']:
    file=upstream/Path(item['path']).name
    assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'],file.name
sys.path.insert(0,str(upstream))
import torch
import tilelang
import kernel
import model

torch.set_num_threads(2)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('cuda')
torch.manual_seed(19)
assert torch.cuda.get_device_capability()==(12,0)
search_paths=[args.native.resolve().parent,Path(os.environ['CUDA_PATH'])/'bin',Path(os.environ['CUDA_PATH'])/'bin/x64']
dll_directories=[os.add_dll_directory(str(p)) for p in search_paths if p.is_dir()] if os.name=='nt' else []
dll=C.CDLL(str(args.native.resolve()))
dll.e19_create.argtypes=[C.c_int]*4+[C.POINTER(C.c_void_p)]
dll.e19_create.restype=C.c_void_p
dll.e19_error.restype=C.c_char_p
dll.e19_run.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_float)]
dll.e19_copy.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_size_t]
dll.e19_destroy.argtypes=[C.c_void_p]

class Native:
    def __init__(self,kind,rows,keys,start,inputs):
        self.inputs=inputs
        assert all(x.is_cuda and x.is_contiguous() for x in inputs)
        torch.cuda.synchronize()
        pointers=(C.c_void_p*len(inputs))(*(x.data_ptr() for x in inputs))
        self.handle=dll.e19_create(kind,rows,keys,start,pointers)
        if not self.handle:raise RuntimeError(dll.e19_error().decode())
    def time(self,iterations=3):
        ms=C.c_float()
        if dll.e19_run(self.handle,iterations,C.byref(ms)):raise RuntimeError(dll.e19_error().decode())
        return ms.value
    def outputs(self,shapes):
        result=[]
        for i,shape in enumerate(shapes):
            out=torch.empty(shape,dtype=torch.float32)
            if dll.e19_copy(self.handle,i,out.data_ptr(),out.numel()*4):raise RuntimeError(dll.e19_error().decode())
            result.append(out)
        return result
    def close(self):
        if self.handle:dll.e19_destroy(self.handle);self.handle=None

record=dict(status='RUNNING',scope='component-only; synthetic operands, not checkpoint accuracy',
    source_revision=manifest['revision'],torch=torch.__version__,tilelang=tilelang.__version__,
    device=torch.cuda.get_device_name(),native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),
    input_preload='native owns D2D copies made before timers; official inputs already resident',attention_variant=args.attention_variant,results=[])
def publish():args.output.write_text(json.dumps(record,indent=2)+'\n')
def timed(fn,iterations=3):
    fn();torch.cuda.synchronize()
    measurements=[]
    for _ in range(3):
        first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(iterations):result=fn()
        last.record();last.synchronize()
        measurements.append(first.elapsed_time(last)/iterations)
    return statistics.median(measurements)
def quality(a,b):
    # Bounded rows for host export; all elements checked for finiteness on device.
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta=(a.float()-b.float())
    return dict(relative_l2=float((delta.double().square().sum()/b.double().square().sum().clamp_min(1e-30)).sqrt()),
        max_abs=float(delta.abs().max()),bit_exact=bool(torch.equal(a.float(),b.float())))
def compare(name,rows,fn,kind,inputs,shapes,keys=0,start=0,**extra):
    native=Native(kind,rows,keys,start,inputs)
    try:
        official_ms=timed(fn)
        expected=fn();expected=[expected] if isinstance(expected,torch.Tensor) else list(expected)
        native.time(1)
        actual=native.outputs(shapes)
        checks=[quality(a,b) for a,b in zip(actual,expected)]
        native_ms=statistics.median(native.time() for _ in range(3))
        row=dict(component=name,rows=rows,official_ms=official_ms,native_ms=native_ms,official_speedup=native_ms/official_ms,quality=checks,**extra)
        record['results'].append(row);publish();print(json.dumps(row),flush=True)
        return row
    finally:native.close()

publish()
try:
    with torch.inference_mode():
        for rows in args.rows:
            if not 0<rows<=65536:raise ValueError('Unadmitted row count')
            for component in args.components:
                print('BEGIN',component,rows,flush=True)
                if component=='norm':
                    x=torch.randn((1,rows,5120));w=torch.randn(5120)
                    layer=model.RMSNorm(5120,1e-20);layer.weight.copy_(w)
                    compare(component,rows,lambda:layer(x),0,[x.float(),w.float()],[(1,rows,5120)])
                    del x,w,layer
                elif component in ('hc_split','hc_native_fused','hc_native_parity'):
                    mixes=torch.randn((1,rows,24),dtype=torch.float32)
                    scale=torch.tensor([.1,.2,.3],dtype=torch.float32);base=torch.randn(24,dtype=torch.float32)
                    pre=torch.empty((rows,4),dtype=torch.float32);post=torch.empty_like(pre);comb=torch.empty((rows,4,4),dtype=torch.float32)
                    compiled=kernel.hc_split_sinkhorn_kernel(4,20,1e-6)
                    def run_hc():
                        compiled(mixes.view(rows,24),scale,base,pre,post,comb)
                        return pre.view(1,rows,4),post.view(1,rows,4),comb.view(1,rows,4,4)
                    if rows<=4096 or component in ('hc_native_fused','hc_native_parity'):
                        kind=7 if component=='hc_native_parity' else 6 if component=='hc_native_fused' else 5
                        shapes=[(1,rows,4),(1,rows,4),(1,rows,4,4)]
                        compare(component,rows,run_hc,kind,[mixes,scale,base],shapes,official_preallocated=True)
                        if component=='hc_native_parity' and rows<=4096:
                            original=Native(5,rows,0,0,[mixes,scale,base]);fused=Native(7,rows,0,0,[mixes,scale,base])
                            try:
                                original.time(1);fused.time(1)
                                checks=[quality(a,b) for a,b in zip(fused.outputs(shapes),original.outputs(shapes))]
                                record['results'].append(dict(component='hc_parity_oracle',rows=rows,quality=checks))
                                publish();print('PARITY',json.dumps(checks),flush=True)
                            finally:original.close();fused.close()
                    else:
                        official_ms=timed(run_hc)
                        parts=[]
                        for first in range(0,rows,4096):
                            count=min(4096,rows-first)
                            def run_hc_tile():
                                compiled(mixes[:,first:first+count].view(count,24),scale,base,pre[first:first+count],post[first:first+count],comb[first:first+count])
                                return pre[first:first+count].view(1,count,4),post[first:first+count].view(1,count,4),comb[first:first+count].view(1,count,4,4)
                            parts.append(compare('hc_split_tile',count,run_hc_tile,5,[mixes[:,first:first+count].contiguous(),scale,base],
                                [(1,count,4),(1,count,4),(1,count,4,4)],parent_rows=rows,first=first))
                        record['results'].append(dict(component=component,rows=rows,official_ms=official_ms,
                            native_ms=sum(p['native_ms'] for p in parts),native_query_tile=4096,official_query_tile=rows,
                            native_schedule='sum of independently timed native microtiles',quality=[p['quality'] for p in parts]))
                        publish()
                        del parts
                    del mixes,scale,base,pre,post,comb,compiled
                elif component=='dense':
                    # Same BF16 operand values. This isolates dense GEMM; no Q8/FP8 weight conversion claim.
                    x=torch.randn((rows,5120));w=torch.randn((1280,5120))*.01
                    compare(component,rows,lambda:torch.nn.functional.linear(x,w),2,[x.float(),w],[(rows,1280)],keys=1280,weight_format='same BF16; FP8 quantization excluded')
                    del x,w
                elif component=='index':
                    keys=32768 if rows==65536 else 100000
                    key=torch.randint(-6,7,(1,keys,128),dtype=torch.int32).to(torch.bfloat16)
                    kernel.fp4_act_quant(key,32,True)
                    parts=[]
                    for first in range(0,rows,4096):
                        count=min(4096,rows-first)
                        query=torch.randint(-6,7,(1,count,32,128),dtype=torch.int32).to(torch.bfloat16)
                        kernel.fp4_act_quant(query,32,True)
                        weights=torch.rand((1,count,32)).to(torch.bfloat16)
                        visible=torch.full((count,),keys,dtype=torch.int32)
                        def index_reference():
                            result=torch.empty((1,count,keys),dtype=torch.bfloat16)
                            for offset in range(0,count,128):
                                score=torch.einsum('bshd,btd->bsht',query[:,offset:offset+128],key)
                                # Native scorer includes1/sqrt(128*32). Apply
                                # that factor to the official head weights too.
                                result[:,offset:offset+128]=(score.relu_()*(weights[:,offset:offset+128]/64).unsqueeze(-1)).sum(dim=2)
                            return result
                        parts.append(compare('index_tile',count,index_reference,3,
                            [query.float(),weights.float(),key.float(),visible],[(1,count,keys)],keys=keys,
                            parent_rows=rows,first=first,official_query_tile=128,native_query_tile=count,score_scale=1/64))
                        del query,weights,visible
                    record['results'].append(dict(component=component,rows=rows,keys=keys,
                        official_ms=sum(p['official_ms'] for p in parts),native_ms=sum(p['native_ms'] for p in parts),
                        schedule='sum of independently timed microtiles; no full-layer claim',
                        unmodified_head_score_bytes=rows*32*keys*2,
                        official_tiling_required=True,quality=[p['quality'] for p in parts]))
                    del key,parts
                    publish()
                elif component=='attention':
                    keys=32769 if rows==65536 else 100001
                    # KV and queries are representable in BF16 and F16; attention inputs match exactly.
                    comp=(torch.randint(-16,17,(keys,512),dtype=torch.int32).float()/32).to(torch.bfloat16)
                    sink=torch.randn(64,dtype=torch.float32)
                    parts=[]
                    sparse=kernel.sparse_attn_kernel(16,512,512**-.5)
                    for first in range(0,rows,4096):
                        count=min(4096,rows-first);start=100000-rows+first
                        query=torch.randn((1,count,64,512))
                        raw=(torch.randint(-16,17,(count+127,512),dtype=torch.int32).float()/32).to(torch.bfloat16)
                        kv=torch.cat([raw,comp]).unsqueeze(0)
                        qi=torch.arange(count,dtype=torch.int32)[:,None]
                        selected=1+(qi*997+torch.arange(512,dtype=torch.int32)[None,:]*13)%(keys-1)
                        ids=torch.cat([qi+torch.arange(128,dtype=torch.int32)[None,:],selected+count+127],dim=1).unsqueeze(0)
                        records=torch.zeros(((count+3)//4,2048,2),dtype=torch.int32)
                        record_view=records.view(-1,512,2)
                        record_view[:count,:,0]=selected
                        record_view[:count,:,1]=(1<<(torch.arange(count,dtype=torch.int32)%4))[:,None]
                        counts=torch.full(((count+3)//4,),2048,dtype=torch.int32)
                        def sparse_reference():
                            result=torch.empty_like(query)
                            for head in range(0,64,16):
                                group=query[:,:,head:head+16].contiguous();out=torch.empty_like(group)
                                sparse(group,kv,out,sink[head:head+16],ids)
                                result[:,:,head:head+16]=out
                            return result
                        parts.append(compare('attention_tile',count,sparse_reference,4,
                            [query.float(),raw.float(),comp.float(),records,counts,sink],[(1,count,64,512)],
                            keys=keys,start=start,parent_rows=rows,first=first,official_head_group=16,num_stages=2))
                        del query,raw,kv,qi,selected,ids,records,record_view,counts
                    record['results'].append(dict(component=component,rows=rows,keys=keys,
                        official_ms=sum(p['official_ms'] for p in parts),native_ms=sum(p['native_ms'] for p in parts),
                        schedule='sum of independently timed microtiles; no full-layer claim',quality=[p['quality'] for p in parts]))
                    del comp,sink,parts,sparse
                    publish()
                else:raise ValueError(component)
                torch.cuda.synchronize();torch.cuda.empty_cache()
    record['status']='PASS'
except BaseException as exc:
    record.update(status='FAIL',error=repr(exc));raise
finally:publish()
