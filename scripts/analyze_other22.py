"""Analyze CUDA kernel/copy intervals; summed families can overlap other resources.

Input is an Nsight SQLite export and a matched native producer trace with
LLAMADECODE_TILE_TIMING=1. The envelope runs from first tile host preparation
through final tile completion, including deferred layer certification;
prompt-only time and launch-to-frontier TTFT remain separate measurements.
Host phase fields are nested wall timers, not CPU utilization or additive work.
"""
import argparse,json,re,sqlite3
from bisect import bisect_left
from pathlib import Path
p=argparse.ArgumentParser(description='Rank Nsight kernel families against native QPC tile intervals; no per-node synchronization.')
p.add_argument('database',type=Path);p.add_argument('native_trace',type=Path)
p.add_argument('measurement',type=Path);p.add_argument('--output',required=True,type=Path)
args=p.parse_args();name=args.measurement.stem
measurement=json.loads(args.measurement.read_text())
guard=json.loads(args.measurement.with_suffix('.guard.json').read_text())
if measurement.get('status')!='PASS' or guard.get('exit_code')!=0 or guard.get('reason') or not guard.get('cleanup_confirmed'):
    raise ValueError('Refusing a failed/rejected measurement or unconfirmed process cleanup')
text=args.native_trace.read_text(errors='replace')
db=sqlite3.connect('file:'+args.database.resolve().as_posix()+'?mode=ro',uri=True)
origin=db.execute('select startTime from ANALYSIS_DETAILS').fetchone()[0]
async_windows=re.findall(r'OTHER22_ASYNC_LAYER_CERTIFIED.*begin_epoch_ms=([\d.]+) end_epoch_ms=([\d.]+)',text)
windows=[(float(a)*1e6-origin,float(b)*1e6-origin) for a,b in (async_windows or re.findall(r'\[dsv41-compute-timeline\].*begin_epoch_ms=([\d.]+) end_epoch_ms=([\d.]+)',text))]
if not windows:raise ValueError('No native compute timeline windows')
left,right=min(a for a,b in windows),max(b for a,b in windows)
host_windows=[(float(a)*1e6-origin,float(b)*1e6-origin) for a,b in re.findall(r'\[dsv41-host-timeline\].*begin_epoch_ms=([\d.]+) end_epoch_ms=([\d.]+)',text)]
if host_windows:
    left=min(left,min(a for a,b in host_windows));right=max(right,max(b for a,b in host_windows))
def merged(xs):
    out=[]
    for a,b in sorted(xs):
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out
def duration(xs):return sum(b-a for a,b in merged(xs))/1e9
rows=db.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=? order by k.start',(left,right)).fetchall()
if not rows:raise ValueError('CUDA trace does not intersect native QPC compute windows')
owners=db.execute('select distinct globalPid from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(left,right)).fetchall()
if len(owners)!=1:raise ValueError('Expected exactly one CUDA process in the native compute envelope')
global_pid=owners[0][0]
starts=[a for a,b,s in rows]
if any(bisect_left(starts,a)==len(starts) or starts[bisect_left(starts,a)]>b for a,b in windows):
    raise ValueError('At least one native compute tile has no CUDA kernel coverage')
if async_windows:
    submitted=re.findall(r'LLAMADECODE_TILE_TIMING layer=(\d+) tile=(\d+) rows=(\d+)',text)
    certified=re.findall(r'OTHER22_ASYNC_TILE_CERTIFIED layer=(\d+) start=(\d+) count=(\d+)',text)
    if sorted(submitted)!=sorted(certified):raise ValueError('Asynchronous CUDA timing events did not certify every submitted tile')
groups={}
def family(s):
    if 'dsv41_layer_norm_kernel' in s:return 'Fused HC collapse / RMS / weight / BF16'
    if 'hc_collapse' in s:return 'HC collapse'
    if 'dsv41_index_fp4' in s:return 'FP4 index score / certificate'
    if 'native_sparse_attn' in s or 'tilelang_' in s:return 'Attention'
    if any(v in s for v in ['moecute','gather(','mid_pack(','reduce(','mm_ids_helper','inverse','scale_rows']):return 'Expert incl packing'
    if 'gemm_bf16_' in s or 'dsa_accum_scores' in s:return 'BF16 GEMM / score epilogue'
    if 'mul_mat_q<' in s or 'gemm_f16_' in s or 'sgemm_' in s or 'fixup' in s:return 'FP16 / Q8 / F32 GEMM incl fixup'
    if 'topk' in s or 'top_k' in s:return 'Top K'
    if 'hc_expand' in s or 'hc_channels' in s:return 'HC expand'
    if 'qat' in s or 'quantize' in s:return 'QAT / quantization'
    if 'concat' in s:return 'Concat'
    if 'cpy_' in s or 'cast_vector' in s:return 'Cast / copy'
    if 'rope' in s:return 'RoPE'
    if 'rms_norm' in s:return 'RMS norm'
    return 'Remaining kernels'
for a,b,s in rows:groups[family(s)]=groups.get(family(s),0)+(b-a)/1e9
top=db.execute('select s.value,sum(k.end-k.start)/1e9,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=? group by k.demangledName order by sum(k.end-k.start) desc',(left,right)).fetchall()
apis=db.execute('select s.value,sum(k.end-k.start)/1e9,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId where k.start>=? and k.end<=? group by k.nameId order by sum(k.end-k.start) desc limit 20',(left,right)).fetchall()
copies=db.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(left,right)).fetchall()
tiles=[]
for line in text.splitlines():
    if line.startswith('LLAMADECODE_TILE_TIMING '):tiles.append(dict((k,float(v)) for k,v in re.findall(r'(\w+)=([\d.]+)',line)))
host={k[:-3]+'_seconds':sum(t.get(k,0) for t in tiles)/1000 for k in tiles[0] if k.endswith('_ms')}
layer_rows=[dict((k,float(v)) for k,v in re.findall(r'(\w+)=([\d.]+)',line))
    for line in text.splitlines() if line.startswith('DS4PERF_LAYER_WALL_LEDGER ')]
layer_totals={k[:-3]+'_seconds':sum(t.get(k,0) for t in layer_rows)/1000 for k in layer_rows[0] if k.endswith('_ms')} if layer_rows else {}
fetched_bytes=sum(map(int,re.findall(r'\[dsv41-fetch\] stage=\d+ bytes=(\d+)',text)))
result=dict(name=name,prompt_seconds=measurement['prompt_prefill_seconds'],
    clock='QPC: SQLite startTime + relative nanoseconds; native steady milliseconds',
    first_to_last_compute_envelope_seconds=(right-left)/1e9,kernel_count=len(rows),
    kernel_busy_union_seconds=duration((a,b) for a,b,s in rows),kernel_duration_sum_seconds=sum((b-a)/1e9 for a,b,s in rows),
    copy_busy_union_seconds=duration(copies),kernel_or_copy_union_seconds=duration([(a,b) for a,b,s in rows]+copies),
    diagnostic_warnings=db.execute('select severity,text from DIAGNOSTIC_EVENT where severity>=2 and (globalPid=? or globalPid is null)',(global_pid,)).fetchall(),
    ranked_kernel_categories=sorted(groups.items(),key=lambda x:-x[1]),top_kernels=top,api_duration_sums=apis,
    tile_host_seconds=host,tile_count=len(tiles),layer_ledger_seconds=layer_totals,
    layer_passes=len(layer_rows),fetched_bytes=fetched_bytes)
result['gpu_idle_seconds']=(right-left)/1e9-result['kernel_or_copy_union_seconds']
result['envelope_scope']='first tile host preparation to last tile completion, including deferred layer certification when enabled'
result['copy_kind_totals']=db.execute('select e.label,count(*),sum(m.bytes),sum(m.end-m.start)/1e9 from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.end<=? group by m.copyKind',(left,right)).fetchall()
result['host_wait_apis']=db.execute("select s.value,count(*),sum(k.end-k.start)/1e9 from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId where k.start>=? and k.end<=? and (s.value like '%Synchronize%' or s.value like '%Graph%Destroy%' or s.value like '%GraphInstantiate%') group by k.nameId order by sum(k.end-k.start) desc",(left,right)).fetchall()
if async_windows:result['asynchronous_certified_tiles']=len(certified)
if async_windows:result['asynchronous_tile_gpu_event_seconds']=sum(map(float,re.findall(r'OTHER22_ASYNC_TILE_CERTIFIED .*gpu_ms=([\d.]+)',text)))/1000
if any('dropped' in text.lower() or 'lost event' in text.lower() for severity,text in result['diagnostic_warnings']):
    raise ValueError('Nsight reports incomplete event collection')
result['qualified_for_clean_ranking']=not any(severity>=3 for severity,message in result['diagnostic_warnings'])
result['trace_quality']='CLEAN_WITH_RETAINED_WARNINGS' if result['qualified_for_clean_ranking'] else 'DIAGNOSTIC_ONLY_METADATA_ERRORS'
args.output.write_text(json.dumps(result,indent=2)+'\n')
db.close()
print(json.dumps({k:v for k,v in result.items() if k not in ('top_kernels','api_duration_sums')},indent=2))
print('API',json.dumps(apis[:8],indent=2))
