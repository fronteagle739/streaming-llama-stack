import json,re
from pathlib import Path
import sys
r=Path(sys.argv.pop(1)).resolve();rows=[]
for label,count in [(x.rsplit(':',1)[0],int(x.rsplit(':',1)[1])) for x in sys.argv[1:]]:
 pair=[]
 for arm in ('control','candidate'):
  d=json.loads((r/f'{label}-{arm}-{count}.json').read_text());assert d['status']=='PASS'
  log=(r/f'{label}-{arm}-{count}-progress.log').read_text();lines=[x for x in log.splitlines() if x.startswith('LLAMADECODE_TILE_TIMING ')]
  tiles=[{k:float(v) for k,v in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)',x)} for x in lines]
  keys=('host_prepare_ms','host_bind_ms','host_register_ms','host_graph_ms','host_reserve_ms','host_upload_ms','host_readback_ms','host_cleanup_ms','compute_ms','gpu_ms')
  native=json.loads((Path(d['native_directory'])/'producer.json').read_text())
  p=dict(arm=arm,prompt_seconds=d['prompt_prefill_seconds'],prompt_tok_s=d['prompt_tok_s'],ttft_seconds=d['ttft_seconds'],tile_count=len(tiles),tile_phase_ms={k:sum(x[k] for x in tiles) for k in keys},native_phases_ms=native['phase_ms'],register_count=log.count('DS4PERF_TILE_HOST_REGISTRATION '),staging_alloc_count=log.count('PERFEXP_HOST_STAGING_ALLOC '),staging_bind_count=log.count('PERFEXP_HOST_STAGING_BIND '),guard=json.loads((r/f'{label}-{arm}-guard.json').read_text()),producer_sha256=d['producer_sha256'])
  assert p['guard']['exit_code']==0 and p['guard']['reason'] is None and p['guard']['cleanup_confirmed']
  # Exclude decode-sized later allocations: fixed ownership covers each logical prefill.
  pair.append(p)
 c=json.loads((r/f'{label}-control-{count}.json').read_text());n=json.loads((r/f'{label}-candidate-{count}.json').read_text())
 for stage in ('ready','frontier','recall','forced'):
  assert c[stage]['checkpoint']==n[stage]['checkpoint'],stage
 assert c['recall']['tokens']==n['recall']['tokens']
 gates=json.loads((r/f'{label}-{count}-summary.json').read_text())['gates'];assert gates['frontier']['bit_exact'] and gates['forced']['bit_exact']
 row=dict(experiment=label,count=count,arms=pair,gates=gates,prompt_throughput_ratio=pair[0]['prompt_seconds']/pair[1]['prompt_seconds'])
 print(json.dumps(row));rows.append(row)
(r/'E18-results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
