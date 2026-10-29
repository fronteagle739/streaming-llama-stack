"""Export strict-provider fetch/dispatch/host timelines without node profiling.

The dispatch envelope includes CPU launch gaps, stream waits and synchronization;
it is not a claim of continuously busy GPU hardware. Tile CUDA events supply
elapsed stream time, also including waits. Fetch envelopes include submission,
fence and completion overhead. No clocks here are additive across resources.
"""
import argparse
import csv
import json
from pathlib import Path
from analyze_dsv41_prefill import analyze, fields


def merged(intervals):
    result=[]
    for begin,end in sorted(intervals):
        if end<begin:raise ValueError('Timeline clock moved backwards')
        if result and begin<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([begin,end])
    return result


def summarize(text):
    measured=[p for p in analyze(text)['passes'] if p['layers']]
    groups=[]
    for line in text.splitlines():
        if '[dsv41-fetch-timeline]' in line:
            row=fields(line)
            if row['stage']==0:groups.append(dict(fetch=[],routed=[],hint=[],dispatch=[],host=[],tiles=[]))
            if groups:groups[-1]['fetch'].append(row)
        elif groups:
            for marker,key in (('[dsv41-prefill-experts]','routed'),('[dsv41-prefill-hint]','hint'),('[dsv41-compute-timeline]','dispatch'),
                               ('[dsv41-host-timeline]','host'),('LLAMADECODE_TILE_TIMING layer=','tiles')):
                if marker in line:groups[-1][key].append(fields(line));break
    groups=[g for g in groups if g['dispatch']]
    if len(groups)!=len(measured):raise ValueError('Fetch passes and layer passes do not reconcile')
    output=[];histogram=[];idle=[];events=[]
    for sweep,(group,p) in enumerate(zip(groups,measured)):
        all_intervals=[]
        for key in ('fetch','routed','hint','dispatch','host'):
            for r in group[key]:
                begin=r.get('submit_epoch_ms',r.get('begin_epoch_ms'))
                end=r.get('fence_epoch_ms',r.get('end_epoch_ms'))
                if begin is None:continue
                events.append(dict(sweep=sweep,resource=key,layer=r.get('layer',r.get('stage',0)-1),
                                   tile=r.get('start',''),begin_ms=begin,end_ms=end,duration_ms=end-begin))
                if key in ('fetch','routed','hint'):all_intervals.append((begin,end))
        intervals=merged(all_intervals)
        begin=min(r['submit_epoch_ms'] for r in group['fetch'])
        end=max([r['end_epoch_ms'] for r in group['dispatch']]+[r['fence_epoch_ms'] for r in group['fetch']])
        cursor=begin
        for left,right in intervals:
            if left>cursor:idle.append(dict(sweep=sweep,begin_ms=cursor,end_ms=left,duration_ms=left-cursor))
            cursor=max(cursor,right)
        if cursor<end:idle.append(dict(sweep=sweep,begin_ms=cursor,end_ms=end,duration_ms=end-cursor))
        busy=sum(b-a for a,b in intervals)
        for layer in range(40):
            perf=next((r for r in p['layers'] if r['layer']==layer),{})
            fetches=[r for r in p['fetches'] if r['stage']==layer+1]
            routes=[r for r in p['expert_tiles'] if r['stage']==layer+1]
            hints=[r for r in p['expert_hints'] if r['stage']==layer+1]
            tiles=[r for r in group['tiles'] if r['layer']==layer]
            hosts=[r for r in group['host'] if r['layer']==layer]
            histogram.append(dict(sweep=sweep,layer=layer,executed=bool(perf),
                fetch_bytes=sum(r['bytes'] for r in fetches+routes+hints),
                fetch_ms=sum(r['fetch_ms'] for r in fetches+routes+hints),
                hint_bytes=sum(r['bytes'] for r in hints),hint_fetch_ms=sum(r['fetch_ms'] for r in hints),
                exposed_hint_join_ms=sum(r['exposed_join_ms'] for r in hints),
                layer_acquire_ms=perf.get('acquire_ms',0),routed_wait_ms=sum(r['fetch_ms'] for r in routes),
                compute_dispatch_sync_ms=perf.get('compute_ms',0),prepare_ms=perf.get('prepare_ms',0),
                upload_ms=perf.get('upload_ms',0),wall_ms=perf.get('wall_ms',0),
                cuda_stream_elapsed_ms=sum(r['gpu_ms'] for r in tiles) if tiles else None,
                registration_ms=sum(r['host_register_ms'] for r in tiles) if tiles else None,
                host_thread_cpu_ms=sum(r['thread_cpu_ms'] for r in hosts) if hosts and all(r.get('thread_cpu_ms',-1)>=0 for r in hosts) else None,
                host_graph_ms=sum(r['host_graph_ms'] for r in tiles) if tiles else None,
                host_reserve_ms=sum(r['host_reserve_ms'] for r in tiles) if tiles else None,
                host_launch_ms=sum(r['host_launch_ms'] for r in tiles) if tiles else None,
                host_sync_ms=sum(r['host_sync_ms'] for r in tiles) if tiles else None))
        output.append(dict(sweep=sweep,window_ms=end-begin,fetch_envelope_busy_ms=busy,
            stream_utilization_percent=100*busy/(end-begin),
            stream_idle_intervals=sum(r['sweep']==sweep for r in idle),
            routed_timestamps_complete=all('begin_epoch_ms' in r for r in group['routed']),
            **{k:p[k] for k in ('fetched_bytes','exposed_acquire_ms','routed_fetch_wait_ms','hint_fetched_bytes','exposed_hint_join_ms','totals')}))
    return dict(passes=output,histogram=histogram,idle_intervals=idle,events=events,
        interpretation=__doc__,window='First stage submission through last layer dispatch/head fetch fence; excludes model open, bulk export, final head compute and inter-pass admission. Exact prompt timing is reported separately.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.log.read_text(encoding='utf-8',errors='replace'))
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for key in ('histogram','idle_intervals','events'):
        with (args.out/(key+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            rows=result[key]
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['sweep'])
            writer.writeheader();writer.writerows(rows)
    print(json.dumps(result['passes'],indent=2))


if __name__=='__main__':main()
