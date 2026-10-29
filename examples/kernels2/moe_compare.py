"""Official FP4 MoE vs the existing ABI4 SM120 FlashInfer adapter, same real weights.

Keep the streaming adapter owned by the native-FP4 work unchanged. Native MoE
uses16K-or-smaller expert chunks; this does not claim one64K grouped invocation.
"""
from pathlib import Path
import argparse,ctypes as C,json,os,runpy,sys,time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scratch-root',type=Path,required=True)
p.add_argument('--source-receipt',type=Path,required=True)
p.add_argument('--bridge',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--chunk',type=int,choices=(4096,8192,16384),default=16384)
args=p.parse_args();root=args.scratch_root.resolve()
saved_argv=sys.argv
sys.argv=['official_layer','--scratch-root',str(root),'--source-receipt',str(args.source_receipt),
    '--output',str(root/'moe-preload.json'),'--preload-only']
scope=runpy.run_path(str(Path(__file__).with_name('official_layer.py')),run_name='e19_official_preload')
sys.argv=saved_argv
torch,layer=scope['torch'],scope['layer'];ffn=layer.ffn
dirs=[args.bridge.resolve().parent,Path(os.environ['CUDA_PATH'])/'bin/x64']
dll_dirs=[os.add_dll_directory(str(d)) for d in dirs if d.exists()]
dll=C.CDLL(str(args.bridge.resolve()))
assert dll.slstack_flashinfer_mxfp4_abi()==4
call=dll.slstack_flashinfer_mxfp4
call.argtypes=[C.c_void_p]*6+[C.c_int]*3+[C.c_int64]*3+[C.c_void_p,C.c_uint64,C.c_void_p]
call.restype=C.c_int
def align(n):return (n+255)//256*256
def workspace_bytes(routes,n,k):
    sf_rows=(routes+384*3+127)//128*128
    return align(384*n*k//2)+align(384*n*k//32)+align(routes*k)+align(sf_rows*k//32)+align(routes*n*2)
record=dict(status='RUNNING',scope='same checkpoint FP4 bytes and BF16 inputs; native16K expert chunks',
    source_shard_sha256=scope['receipt']['sha256'],native_chunk=args.chunk,results=[])
def publish():args.output.write_text(json.dumps(record,indent=2)+'\n')
publish()
native_weights={}
with torch.inference_mode():
    for projection,n,k in [('w1',2304,5120),('w3',2304,5120),('w2',5120,2304)]:
        packed=torch.empty((384,n,k//32,17),dtype=torch.uint8)
        for e in range(384):
            w=getattr(ffn.experts[e],projection)
            paired=w.weight.view(torch.uint8).reshape(n,k//32,16)
            codes=torch.stack((paired&15,paired>>4),dim=-1).reshape(n,k//32,32)
            packed[e,:,:,0]=w.scale.view(torch.uint8)
            packed[e,:,:,1:]=codes[:,:,:16]|(codes[:,:,16:]<<4)
        native_weights[projection]=packed
        print('PACKED',projection,packed.numel(),flush=True)
    del packed,paired,codes
    torch.cuda.synchronize();torch.cuda.empty_cache()
    work_size=max(workspace_bytes(args.chunk*6,n,k) for n,k in [(2304,5120),(5120,2304)])
    workspace=torch.empty(work_size,dtype=torch.uint8)
    record['workspace_bytes']=work_size
    def project(weight,x,gather,scatter,bounds,n,k,rows):
        out=torch.empty((rows,6,n),dtype=torch.float32)
        pointers=[weight.data_ptr(),x.data_ptr(),gather.data_ptr(),scatter.data_ptr(),bounds.data_ptr(),out.data_ptr()]
        status=call(*pointers,rows*6,n,k,k,n,n*6,workspace.data_ptr(),work_size,torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError('FlashInfer ABI4 failure: '+str(status))
        return out
    def native(x):
        result=torch.empty_like(x)
        # Router and every gather/sort/scatter are within the native timed call.
        for first in range(0,len(x),args.chunk):
            inp=x[first:first+args.chunk];rows=len(inp)
            weights,indices=ffn.gate(inp)
            order=torch.argsort(indices.flatten(),stable=True).to(torch.int32)
            gather=(order//6).to(torch.int32)
            counts=torch.bincount(indices.flatten(),minlength=384)
            bounds=torch.cat([torch.zeros(1,dtype=torch.int32),counts.cumsum(0).to(torch.int32)])
            inp32=inp.float()
            gate=project(native_weights['w1'],inp32,gather,order,bounds,2304,5120,rows)
            up=project(native_weights['w3'],inp32,gather,order,bounds,2304,5120,rows)
            mid=(torch.nn.functional.silu(gate.clamp(max=10))*up.clamp(-10,10)*weights[:,:,None]).bfloat16().float()
            del gate,up,inp32
            down=project(native_weights['w2'],mid.reshape(-1,2304),order,order,bounds,5120,2304,rows)
            del mid
            combine=torch.zeros((rows,5120),dtype=torch.float32)
            expert_order=indices.argsort(dim=-1)
            rowids=torch.arange(rows)
            for slot in range(6):combine+=down[rowids,expert_order[:,slot]]
            combine+=ffn.shared_experts(inp)
            result[first:first+rows]=combine.bfloat16()
        return result
    try:
        for rows in [2541,65536]:
            torch.manual_seed(19);x=torch.randn((rows,5120));outputs=[];measurements=[]
            for name,fn in [('official',ffn),('native',native)]:
                print('WARMUP',name,rows,flush=True);warm=fn(x);torch.cuda.synchronize();del warm
                start=time.perf_counter();out=fn(x);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                outputs.append(out);measurements.append(elapsed)
                print('MEASURED',name,rows,elapsed,flush=True)
            a,b=outputs;delta=(a.float()-b.float())
            check=dict(relative_l2=float((delta.double().square().sum()/a.double().square().sum()).sqrt()),
                max_abs=float(delta.abs().max()),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                bit_exact=bool(torch.equal(a,b)))
            record['results'].append(dict(rows=rows,official_ms=measurements[0],native_ms=measurements[1],quality=check,
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated()))
            publish();print(json.dumps(record['results'][-1]),flush=True)
            del x,outputs,a,b,delta,out;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        record['status']='PASS'
    except BaseException as exc:record.update(status='FAIL',error=repr(exc));raise
    finally:publish()
