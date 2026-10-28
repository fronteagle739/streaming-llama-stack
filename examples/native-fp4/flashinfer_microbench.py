import json
import os
from pathlib import Path
import runpy
import time
import argparse
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
parser=argparse.ArgumentParser()
parser.add_argument('--route-cases',type=Path)
parser.add_argument('--output',type=Path)
options=parser.parse_args()
target=options.output or ROOT/'flashinfer-microbench.json'
cases=[(e,[m]*e,n,k,'balanced') for e,m,n,k in ((1,128,2304,5120),(6,1,2304,5120),(6,1,5120,2304),
       (384,64,2304,5120),(384,64,5120,2304),(384,768,2304,5120),(384,768,5120,2304))]
if options.route_cases:
    cases=[]
    for case in json.loads(options.route_cases.read_text())['cases']:
        counts=case['counts']
        if len(counts)!=384 or any(type(m) is not int or not 0<=m<=4096 for m in counts) or sum(counts)!=case['rows']*6:
            raise ValueError('Invalid observed routing counts')
        cases.extend((384,counts,n,k,case['label']) for n,k in [(2304,5120),(5120,2304)])
    record['scope']='Observed graph routing counts; synthetic representable hash32-v1 operands, activation quantization and packing excluded'
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
    for experts,counts,n,k,case_label in cases:
        m=max(counts);total=sum(counts);starts=[0]
        for count in counts:starts.append(starts[-1]+count)
        a=torch.empty((total,k),device='cuda',dtype=torch.float8_e4m3fn)
        b=torch.empty((experts,n,k//2),device='cuda',dtype=torch.uint8)
        columns=torch.arange(k,device='cuda',dtype=torch.int64)[None,:]
        a_rows=torch.arange(m,device='cuda',dtype=torch.int64)[:,None]
        b_rows=torch.arange(n,device='cuda',dtype=torch.int64)[:,None]
        for expert in range(experts):
            a[starts[expert]:starts[expert+1]]=((hashes(expert,a_rows[:counts[expert]],columns)%5)-2).to(torch.float8_e4m3fn)
            b[expert]=((hashes(expert,b_rows,columns[:,::2])&15)|((hashes(expert,b_rows,columns[:,1::2])&15)<<4)).to(torch.uint8)
        del columns,a_rows,b_rows
        # Four native E8M0/block32 scales packed in one int32: 1,2,4,1.
        sf_word=0x7f81807f
        sf_rows=((total+experts*(kernel.SF_M_ALIGN-1)+127)//128)*128
        sa=torch.zeros((k//128,sf_rows),device='cuda',dtype=torch.int32)
        for expert in range(experts):
            first=((starts[expert]+expert*(kernel.SF_M_ALIGN-1))//kernel.SF_M_ALIGN)*kernel.SF_M_ALIGN
            for offset in range(min(3,counts[expert])):
                phase=(expert+offset)%3
                word=sum((127+(value+phase)%3)<<(8*i) for i,value in enumerate((0,1,2,0)))
                if word>=2**31:word-=2**32
                sa[:,first+offset:first+counts[expert]:3]=word
        sb=torch.full((experts,k//128,((n+127)//128)*128),sf_word,device='cuda',dtype=torch.int32)
        out=torch.empty((total,n),device='cuda',dtype=torch.bfloat16)
        out.fill_(float('nan'))
        offsets=torch.tensor(starts,device='cuda',dtype=torch.int32)
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
        if not torch.isfinite(out).all().item():raise AssertionError('Nonfinite or unwritten full output')
        active=[e for e in range(experts) if counts[e]]
        sampled=sorted(set((active[0],active[len(active)//2],active[-1],counts.index(m))))
        for expert in sampled:
            mi=sorted(set((0,counts[expert]//2,counts[expert]-1)))
            ni=sorted(set((0,n//2,n-1)))
            ai=torch.tensor([starts[expert]+i for i in mi],device='cuda')
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
        used_weight_bytes=sum(m>0 for m in counts)*n*k//2
        bank_count=max(1,((128<<20)+used_weight_bytes-1)//used_weight_bytes)
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
        active_experts=sum(m>0 for m in counts)
        logical_bytes=total*k+used_weight_bytes+(total+active_experts*n)*k/32+2*total*n
        row={'experts':experts,'M_per_expert':m if len(set(counts))==1 else None,'maximum_M':m,
             'M_counts':counts,'case':case_label,'N':n,'K':k,'tile':list(tile),'stages':2,
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
