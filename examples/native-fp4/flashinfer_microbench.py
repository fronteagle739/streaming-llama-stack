import json
import os
from pathlib import Path
import runpy
import time
ROOT=Path(os.environ['NATIVEFP4_BENCH_ROOT'])
scope=runpy.run_path(str(Path(__file__).with_name('probe_flashinfer.py')))
torch,kernel,EpiMethod=(scope[name] for name in ('torch','kernel','EpiMethod'))
import cutlass.cute as cute
props=torch.cuda.get_device_properties(0)
rows=[]
record={'status':'RUNNING','source_commit':'1d622fdb1b053fc532069914bc9a6b498bd5f722',
        'fixture':'hash32-v1',
        'torch':torch.__version__,'scale_granularity_A':32,'scale_granularity_B':32,
        'scope':'Synthetic token-packed grouped GEMMs; balanced M, not an observed routing histogram; activation quantization and layout preparation excluded',
        'rows':rows}
target=ROOT/'flashinfer-microbench.json'
def publish():target.write_text(json.dumps(record,indent=2))
def hashes(expert, row, column):
    x=(((expert+1)*0x9e3779b9)^((row+1)*0x85ebca6b)^((column+1)*0xc2b2ae35))&0xffffffff
    x=((x^(x>>16))*0x7feb352d)&0xffffffff
    x=((x^(x>>15))*0x846ca68b)&0xffffffff
    return x^(x>>16)
def scalar_hash(expert,row,column):
    x=(((expert+1)*0x9e3779b9)^((row+1)*0x85ebca6b)^((column+1)*0xc2b2ae35))&0xffffffff
    x=((x^(x>>16))*0x7feb352d)&0xffffffff
    x=((x^(x>>15))*0x846ca68b)&0xffffffff
    return x^(x>>16)
publish()
try:
    for experts,m,n,k in ((1,128,2304,5120),(6,1,2304,5120),(6,1,5120,2304),
                          (384,64,2304,5120),(384,64,5120,2304),
                          (384,768,2304,5120),(384,768,5120,2304)):
        total=experts*m
        a=torch.empty((total,k),device='cuda',dtype=torch.float8_e4m3fn)
        b=torch.empty((experts,n,k//2),device='cuda',dtype=torch.uint8)
        columns=torch.arange(k,device='cuda',dtype=torch.int64)[None,:]
        a_rows=torch.arange(m,device='cuda',dtype=torch.int64)[:,None]
        b_rows=torch.arange(n,device='cuda',dtype=torch.int64)[:,None]
        for expert in range(experts):
            a[expert*m:(expert+1)*m]=((hashes(expert,a_rows,columns)%5)-2).to(torch.float8_e4m3fn)
            b[expert]=((hashes(expert,b_rows,columns[:,::2])&15)|((hashes(expert,b_rows,columns[:,1::2])&15)<<4)).to(torch.uint8)
        del columns,a_rows,b_rows
        # Four native E8M0/block32 scales packed in one int32: 1,2,4,1.
        sf_word=0x7f81807f
        sf_rows=((total+experts*(kernel.SF_M_ALIGN-1)+127)//128)*128
        sa=torch.zeros((k//128,sf_rows),device='cuda',dtype=torch.int32)
        for expert in range(experts):
            first=((expert*m+expert*(kernel.SF_M_ALIGN-1))//kernel.SF_M_ALIGN)*kernel.SF_M_ALIGN
            for offset in range(min(3,m)):
                phase=(expert+offset)%3
                word=sum((127+(value+phase)%3)<<(8*i) for i,value in enumerate((0,1,2,0)))
                if word>=2**31:word-=2**32
                sa[:,first+offset:first+m:3]=word
        sb=torch.full((experts,k//128,((n+127)//128)*128),sf_word,device='cuda',dtype=torch.int32)
        out=torch.empty((total,n),device='cuda',dtype=torch.bfloat16)
        out.fill_(float('nan'))
        offsets=torch.arange(0,(experts+1)*m,m,device='cuda',dtype=torch.int32)
        tile=(64,128,128)
        cfg=kernel.make_cfg(tile,2,epi=EpiMethod.R2G_WG)
        program=kernel.CuteDslSm120MoeMxfp8Mxfp4Grouped(cfg,props.multi_processor_count)
        args=kernel.make_args(a,sa,b,sb,out,offsets,cfg.epi,tile)
        print('COMPILE',experts,m,n,k,flush=True)
        compiled=cute.compile(program,*args)
        compiled(*args)
        torch.cuda.synchronize()
        codebook=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float64)
        scale=torch.tensor([1,2,4,1],dtype=torch.float64).repeat(k//128).repeat_interleave(32)
        max_error=0.;checked=0;exact=True;checks=[]
        for expert in sorted(set((0,experts//2,experts-1))):
            mi=sorted(set((0,m//2,m-1)))
            ni=sorted(set((0,n//2,n-1)))
            ai=torch.tensor([expert*m+i for i in mi],device='cuda')
            bi=torch.tensor(ni,device='cuda')
            a_scales=torch.stack([torch.tensor([2.**((value+expert+i)%3) for value in (0,1,2,0)],dtype=torch.float64).repeat(k//128).repeat_interleave(32) for i in mi])
            ar=a[ai].float().cpu().double()*a_scales
            bp=b[expert,bi].cpu()
            codes=torch.stack((bp&15,bp>>4),dim=-1).reshape(len(ni),k).long()
            expected_codes=torch.tensor([[scalar_hash(expert,j,column)&15 for column in range(k)] for j in ni])
            if not torch.equal(codes,expected_codes):raise AssertionError('Weight fixture differs from independent scalar hash')
            expected_a=torch.tensor([[scalar_hash(expert,i,column)%5-2 for column in range(k)] for i in mi],dtype=torch.float64)
            if not torch.equal(a[ai].float().cpu().double(),expected_a):raise AssertionError('Activation fixture differs from scalar hash')
            br=codebook[codes]*scale
            reference_bf16=(ar@br.T).to(torch.bfloat16)
            actual_bf16=out[ai][:,bi].cpu()
            reference=reference_bf16.double()
            actual=actual_bf16.double()
            error=((actual-reference).abs()/reference.abs().clamp_min(1)).max().item()
            max_error=max(max_error,error);checked+=actual.numel();exact &= torch.equal(actual_bf16.view(torch.int16),reference_bf16.view(torch.int16))
            checks.append({'expert':expert,'mi':mi,'ni':ni,'actual':actual.tolist(),'reference':reference.tolist()})
        (ROOT/f'flashinfer-check-{experts}-{m}-{n}-{k}.json').write_text(json.dumps(checks,indent=2))
        if not exact:raise AssertionError(f'Grouped GEMM sampled exact double oracle failed: scaled error={max_error}')
        # Tiny-M rotates identical banks beyond L2; large grouped banks already exceed it.
        bank_count=max(1,((128<<20)+b.numel()-1)//b.numel())
        banks=[b]+[b.clone() for _ in range(bank_count-1)]
        eager_args=[kernel.make_args(a,sa,bank,sb,out,offsets,cfg.epi,tile) for bank in banks]
        for i in range(4):compiled(*eager_args[i%bank_count])
        torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(20):compiled(*eager_args[i%bank_count])
        end.record();end.synchronize()
        eager_ms=start.elapsed_time(end)/20
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            graph_args=[kernel.make_args(a,sa,bank,sb,out,offsets,cfg.epi,tile) for bank in banks]
            compiled(*graph_args[0])
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        count=max(4,bank_count)
        with torch.cuda.graph(graph,stream=stream):
            for i in range(count):compiled(*graph_args[i%bank_count])
        graph.replay();torch.cuda.synchronize()
        start.record()
        for _ in range(10):graph.replay()
        end.record();end.synchronize()
        ms=start.elapsed_time(end)/(10*count)
        if not torch.equal(out[ai][:,bi].cpu().view(torch.int16),reference_bf16.view(torch.int16)):raise AssertionError('Post-graph reference changed')
        flops=2*total*n*k
        logical_bytes=total*k+experts*n*k/2+(total+experts*n)*k/32+2*total*n
        row={'experts':experts,'M_per_expert':m,'N':n,'K':k,'tile':list(tile),'stages':2,
             'activation_sf_rows':sf_rows,'activation_sf_alignment':kernel.SF_M_ALIGN,
             'sampled_double_reference_elements':checked,'bit_exact_on_samples':exact,'max_scaled_error':max_error,
             'graph_ms':ms,'eager_ms':eager_ms,'TFLOPS':flops/ms/1e9,'logical_effective_GB_s':logical_bytes/ms/1e6,
             'weight_banks':bank_count,'weight_bank_bytes':b.numel()*bank_count}
        rows.append(row);publish();print(json.dumps(row),flush=True)
        del graph,compiled,args,eager_args,graph_args,banks,a,b,sa,sb,out,offsets,program,cfg
        torch.cuda.empty_cache()
    record.update(status='PASS',finished=time.time())
except BaseException as error:
    record.update(status='FAIL',error=repr(error),finished=time.time())
    raise
finally:publish()
