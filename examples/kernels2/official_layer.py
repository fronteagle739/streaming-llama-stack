"""Run one actual pinned DeepSeek Block with its weights preloaded on SM120.

The input residual is synthetic and reproducible. This is not a model-logit or
recall qualification. The only forward adapters are16-head sparse attention
groups and row-tiling of HC post's otherwise oversized broadcast intermediate.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import os
import struct
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scratch-root',type=Path,required=True)
p.add_argument('--source-receipt',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--rows',type=int,nargs='+',default=[2541,65536])
p.add_argument('--preload-only',action='store_true')
args=p.parse_args();root=args.scratch_root.resolve()
sys.path[:0]=[str(root/'tilelang-deps'),str(root/'python-packages'),str(Path(__file__).with_name('upstream'))]
os.environ.update(TILELANG_CACHE_DIR=str(root/'tilelang-cache'),TVM_FFI_CACHE_DIR=str(root/'tvm-cache'),MAX_JOBS='2')
os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
import torch
import kernel
import model

torch.set_num_threads(2);torch.set_default_device('cuda');torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_per_process_memory_fraction(.94);torch.manual_seed(19)
receipt=json.loads(args.source_receipt.read_text());source=Path(receipt['source']);before=source.stat()
assert receipt['verified'] and before.st_size==receipt['bytes'] and before.st_mtime_ns==receipt['mtime_ns']
assert receipt['revision']=='1bae0e1088993b9b82de08f59de5be5a34e42170' and receipt['layer']==0
with source.open('rb') as stream:
    length=struct.unpack('<Q',stream.read(8))[0];assert length<2**20
    header=json.loads(stream.read(length))
types={'F32':torch.float32,'BF16':torch.bfloat16,'F8_E4M3':torch.float8_e4m3fn,
       'F8_E8M0':torch.float8_e8m0fnu,'I8':torch.int8,'U8':torch.uint8}
def read(name):
    item=header[name];first,last=item['data_offsets']
    assert last-first<=256*2**20
    with source.open('rb') as stream:
        stream.seek(8+length+first);raw=bytearray(stream.read(last-first))
    assert len(raw)==last-first
    return torch.frombuffer(raw,dtype=types[item['dtype']]).reshape(item['shape']).to('cuda')

config=json.loads(Path(__file__).with_name('upstream').joinpath('config.json').read_text())
config.update(max_batch_size=1,max_seq_len=max(args.rows))
layer=model.Block(0,model.ModelArgs(**config)).eval()
loaded=[]
with torch.inference_mode():
    for name,parameter in layer.named_parameters():
        source_name='layers.0.'+name
        if name=='attn.wo_a.weight':
            w=read(source_name);s=read(source_name.replace('weight','scale'))
            w=(w.unflatten(0,(-1,32)).unflatten(-1,(-1,32)).float()*s[:,None,:,None].float()).flatten(2,3).flatten(0,1).bfloat16()
        else:
            w=read(source_name)
            if parameter.dtype==torch.float4_e2m1fn_x2:w=w.view(torch.float4_e2m1fn_x2)
        assert w.shape==parameter.shape and w.dtype==parameter.dtype,(name,w.shape,parameter.shape,w.dtype,parameter.dtype)
        parameter.copy_(w);loaded.append(name);del w
torch.cuda.synchronize()
weight_bytes=sum(t.numel()*t.element_size() for t in layer.parameters())
record=dict(status='RUNNING',source_revision=receipt['revision'],source_shard_sha256=receipt['sha256'],
    layer=0,weight_bytes=weight_bytes,parameter_count=len(loaded),input='seed19 synthetic BF16 residual',
    scope='actual checkpoint layer, preloaded weights; no comparison of whole-model quality',
    adapters=['sparse head groups16 with unchanged per-head arithmetic','HC post row tiles1024; original method within each tile'],results=[])
def publish():args.output.write_text(json.dumps(record,indent=2)+'\n')
publish()
sparse=kernel.sparse_attn_kernel(16,512,512**-.5)
events=[];recording=False
def measured(category,fn):
    @functools.wraps(fn)
    def run(*a,**kw):
        if not recording:return fn(*a,**kw)
        first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        first.record();result=fn(*a,**kw);last.record();events.append((category,first,last));return result
    return run
def sparse_grouped(q,kv,sink,ids,scale):
    assert scale==512**-.5
    out=torch.empty_like(q)
    for head in range(0,64,16):
        group=q[:,:,head:head+16].contiguous();result=torch.empty_like(group)
        sparse(group,kv,result,sink[head:head+16],ids)
        out[:,:,head:head+16]=result
    return out
model.sparse_attn=measured('attention',sparse_grouped)
post=layer.hc_post
def post_tiled(x,residual,post_mix,comb):
    if x.shape[1]<=1024:return post(x,residual,post_mix,comb)
    result=torch.empty_like(residual)
    for first in range(0,x.shape[1],1024):
        sl=slice(first,first+1024)
        result[:,sl]=post(x[:,sl],residual[:,sl],post_mix[:,sl],comb[:,sl])
    return result
layer.hc_post=post_tiled
for method in ['hc_pre','hc_post','hc_mixes']:
    setattr(layer,method,measured('norms_hc',getattr(layer,method)))
for name,module in layer.named_modules():
    if isinstance(module,model.Expert):
        module.forward=measured('experts' if name.startswith('ffn.experts.') else 'shared_expert',module.forward)
    elif isinstance(module,model.Gate):module.forward=measured('router',module.forward)
    elif isinstance(module,model.RMSNorm):module.forward=measured('norms_hc',module.forward)
    elif isinstance(module,model.Linear) and not name.startswith(('ffn.experts.','ffn.shared_experts.')):
        module.forward=measured('dense',module.forward)
try:
    with torch.inference_mode():
        # Independent adapter check at a ragged size before using tiled HC at64K.
        x=torch.randn((1,2541,5120));residual=torch.randn((1,2541,4,5120))
        pm=torch.randn((1,2541,4),dtype=torch.float32);comb=torch.randn((1,2541,4,4),dtype=torch.float32)
        assert torch.equal(post(x,residual,pm,comb),post_tiled(x,residual,pm,comb))
        record['hc_post_adapter_bit_exact']=True;del x,residual,pm,comb
        for rows in ([] if args.preload_only else args.rows):
            assert rows in (2541,65536)
            x=torch.randn((1,rows,4,5120));pre=model.make_identity_pre_mix(x,4)
            print('WARMUP',rows,flush=True)
            warm=layer(x,0,pre,None);torch.cuda.synchronize();del warm
            # Compilation and all weight I/O precede this timer.
            events.clear();recording=True;started=time.perf_counter()
            output=layer(x,0,pre,None);torch.cuda.synchronize()
            wall=time.perf_counter()-started;recording=False
            assert all(torch.isfinite(t).all() for t in output)
            stages={}
            for category,first,last in events:stages[category]=stages.get(category,0)+first.elapsed_time(last)
            # Events are disjoint leaf/operator scopes; residual includes launch gaps and unclassified ops.
            stages['other_wall_ms']=wall*1000-sum(stages.values())
            record['results'].append(dict(rows=rows,wall_ms=wall*1000,stages_ms=stages,
                indexer_ms=0,engram_ms=0,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                unmodified_hc_broadcast_bytes=rows*16*5120*4,finite=True))
            print(json.dumps(record['results'][-1]),flush=True);publish()
            del output,x,pre;events.clear();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    assert source.stat().st_mtime_ns==before.st_mtime_ns and source.stat().st_size==before.st_size
    record['status']='PASS'
except BaseException as exc:record.update(status='FAIL',error=repr(exc));raise
finally:publish()
