"""Join Nsight CUDA kernel intervals to the native QPC provider timeline.

Nsight timestamps are relative to ANALYSIS_DETAILS.startTime (nanoseconds on
the Windows QPC clock). Native steady-clock timestamps are QPC milliseconds.
CUDA kernel busy is a union, not a sum of concurrent kernel durations. Host
intervals are envelopes, including waits; thread CPU counters, when present,
are reported separately and do not locate scheduler slices within an envelope.
"""
import argparse
from bisect import bisect_left,bisect_right
import csv
import json
from pathlib import Path
import sqlite3
from coneall_timeline import summarize,merged


def overlap(intervals,left,right):
    return sum(max(0,min(end,right)-max(begin,left)) for begin,end in intervals if begin<right and end>left)


def clipped(intervals,left,right):
    return [(max(begin,left),min(end,right)) for begin,end in intervals if begin<right and end>left]


def indexed_overlap(intervals,left,right):
    begins,ends=intervals
    start=bisect_right(ends,left);stop=bisect_left(begins,right)
    return sum(min(ends[i],right)-max(begins[i],left) for i in range(start,stop))


def analyze(database,native_text,pid):
    result=summarize(native_text)
    db=sqlite3.connect(str(database))
    origin,stop=db.execute('select startTime,stopTime from ANALYSIS_DETAILS').fetchone()
    global_pid=next(row[0] for row in db.execute('select distinct globalPid from CUPTI_ACTIVITY_KIND_KERNEL')
                    if (row[0]>>24)&0xffffff==pid)
    resources={}
    for name,table in (('gpu_kernel','CUPTI_ACTIVITY_KIND_KERNEL'),('gpu_copy','CUPTI_ACTIVITY_KIND_MEMCPY')):
        resources[name]=merged(((begin+origin)/1e6,(end+origin)/1e6)
            for begin,end in db.execute('select start,end from '+table+' where globalPid=? order by start',(global_pid,)))
    # A missing/misaligned trace must never become a zero-GPU-time report.
    dispatch=[(r['begin_ms'],r['end_ms']) for r in result['events'] if r['resource']=='dispatch']
    if not resources['gpu_kernel'] or not any(overlap(resources['gpu_kernel'],a,b)>0 for a,b in dispatch):
        raise ValueError('QPC-aligned CUDA kernels do not intersect native dispatch')
    result['clock']=dict(alignment='QPC absolute: SQLite startTime + relative ns; native steady ms',
                         origin_ns=origin,stop_ns=stop)
    result['diagnostic_warnings']=[row[0] for row in db.execute(
        'select distinct text from DIAGNOSTIC_EVENT where globalPid=? and severity>=2',(global_pid,))]
    result['gpu_kernel_count']=db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=?',(global_pid,)).fetchone()[0]
    indexed={key:([a for a,b in value],[b for a,b in value]) for key,value in resources.items()}
    for row in result['histogram']:
        windows=merged((r['begin_ms'],r['end_ms']) for r in result['events']
            if r['resource']=='dispatch' and r['sweep']==row['sweep'] and r['layer']==row['layer'])
        for key in resources:row[key+'_busy_ms']=sum(indexed_overlap(indexed[key],a,b) for a,b in windows)
    for row in result['passes']:
        events=[r for r in result['events'] if r['sweep']==row['sweep']]
        left=min(r['begin_ms'] for r in events if r['resource']=='fetch');right=left+row['window_ms']
        row.update(begin_ms=left,end_ms=right)
        for key in resources:row[key+'_busy_ms']=indexed_overlap(indexed[key],left,right)
        row['gpu_kernel_utilization_percent']=row['gpu_kernel_busy_ms']/row['window_ms']*100
    left=result['passes'][0]['begin_ms'];right=result['passes'][-1]['end_ms']
    result['top_kernels']=[dict(name=name,total_ms=ns/1e6,count=count) for name,ns,count in db.execute(
        'select s.value,sum(k.end-k.start),count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName '
        'where k.globalPid=? and k.start+?>=? and k.end+?<=? group by k.demangledName order by sum(k.end-k.start) desc limit 30',
        (global_pid,origin,int(left*1e6),origin,int(right*1e6)))]
    db.close()
    return result,resources


def plot(result,resources,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    left=result['passes'][0]['begin_ms'];right=result['passes'][-1]['end_ms']
    fig,axes=plt.subplots(2,1,figsize=(13,6),height_ratios=(2.5,1.4),constrained_layout=True)
    lanes=[('DS fetch / routed fetch',merged((r['begin_ms'],r['end_ms']) for r in result['events'] if r['resource'] in ('fetch','routed','hint')),'#167c80'),
           ('CUDA kernels',resources['gpu_kernel'],'#4f62c0'),('CUDA copies',resources['gpu_copy'],'#e29d35'),
           ('Host tile envelope',merged((r['begin_ms'],r['end_ms']) for r in result['events'] if r['resource']=='host'),'#9a78a6')]
    for y,(label,intervals,color) in enumerate(lanes):
        # One-ms display bins only. Exact DS idle intervals remain in the CSV.
        bins=np.zeros(int(right-left)+2,dtype=bool)
        for a,b in clipped(intervals,left,right):bins[max(0,int(a-left)):min(len(bins),int(b-left)+1)]=True
        edges=np.diff(np.r_[False,bins,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
        axes[0].broken_barh([(a/1000,(b-a)/1000) for a,b in zip(starts,ends)],(y-.3,.6),facecolors=color)
    axes[0].set(yticks=range(4),yticklabels=[r[0] for r in lanes],xlabel='Seconds from first fetch submission (1 ms display bins)',xlim=(0,(right-left)/1000))
    axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.25)
    for row in result['passes'][1:]:axes[0].axvline((row['begin_ms']-left)/1000,color='#555555',linestyle='--',linewidth=1)
    x=range(len(result['histogram']))
    axes[1].bar(x,[r['gpu_kernel_busy_ms']/1000 for r in result['histogram']],color='#4f62c0',label='CUDA kernel busy')
    axes[1].plot(x,[r['routed_wait_ms']/1000 for r in result['histogram']],color='#c0453c',label='Routed wait')
    axes[1].set(xlabel='Sweep × 40 layers (skipped layers retained)',ylabel='Seconds',xticks=[0,20,39,40,60,79])
    axes[1].legend(loc='upper left',ncol=2);axes[1].grid(axis='y',alpha=.25)
    fig.suptitle('100K exact cone: DirectStorage, CUDA and host envelopes\nDiagnostic capture; host envelopes include waiting. Resource clocks overlap.')
    fig.savefig(path.with_suffix('.svg'));fig.savefig(path.with_suffix('.png'),dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('database',type=Path);p.add_argument('log',type=Path)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
    result,resources=analyze(args.database,args.log.read_text(encoding='utf-8',errors='replace'),args.pid)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for key in ('histogram','idle_intervals','events'):
        with (args.out/(key+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(result[key][0]));writer.writeheader();writer.writerows(result[key])
    if args.plot:plot(result,resources,args.out/'timeline')
    print(json.dumps(dict(passes=result['passes'],top_kernels=result['top_kernels'][:8],warnings=result['diagnostic_warnings']),indent=2))


if __name__=='__main__':main()
