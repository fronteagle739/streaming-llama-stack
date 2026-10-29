import json,re,statistics,sys
from pathlib import Path
r=Path(sys.argv.pop(1)).resolve()
out=[]
for experiment in sys.argv[1:]:
 rows=[]
 for arm in ('control','candidate'):
  d=json.loads((r/f'{experiment}-{arm}-8192.json').read_text());assert d['status']=='PASS'
  begin=d['warmup']['checkpoint']['position'];end=d['decode']['checkpoint']['position'];assert end-begin==127
  h=[];ds=[]
  for line in (r/f'{experiment}-{arm}-8192-progress.log').read_text(errors='replace').splitlines():
   if not line.startswith(('PERFEXP_HC_GRAPH_TOKEN ','LLAMADECODE_DS4X_TOKEN ')):continue
   x={k:float(v) if '.' in v else int(v) for k,v in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)',line)}
   if not begin<=x['position']<end:continue
   (h if line.startswith('PERFEXP') else ds).append(x)
  assert len(h)==len(ds)==127
  assert all(x['calls']==38 and x['enabled']==int(arm=='candidate') and x['replays']==(38 if arm=='candidate' else 0) for x in h)
  sums={k:sum(x[k] for x in ds) for k in ('ds_bytes','ds_ms','l1_hits','l2_hits','misses','decode_ms')}
  assert sums['l1_hits']+sums['l2_hits']+sums['misses']==127*240
  rows.append(dict(arm=arm,prompt_prefill_seconds=json.loads((Path(d['native_directory'])/'producer.json').read_text())['phase_ms']['embedding_40_layers_head']/1000,ttft_seconds=d['ttft_seconds'],decode=dict(tok_s=127/(sum(d['decode']['step_ms'])/1000),seconds=sum(d['decode']['step_ms'])/1000,evaluated_tokens=127),hc_encode_ms=sum(x['encode_ms'] for x in h),hc_median_encode_ms=statistics.median(x['encode_ms'] for x in h),witness=dict(tokens=127,calls_per_token=38,replays_per_token=38 if arm=='candidate' else 0),ledger=sums,producer_sha256=d['producer_sha256'],decode_dll_sha256=d['decode_dll_sha256']))
 gate=json.loads((r/f'{experiment}-8192-summary.json').read_text())['gates'];assert gate['frontier']['bit_exact'] and gate['forced']['bit_exact']
 c=json.loads((r/f'{experiment}-control-8192.json').read_text());n=json.loads((r/f'{experiment}-candidate-8192.json').read_text())
 for key in ('recall','warmup','decode'):
  assert c[key]['tokens']==n[key]['tokens']
  assert c[key]['checkpoint']==n[key]['checkpoint']
 x=dict(experiment=experiment,rows=rows,gates=gate,decode_throughput_ratio=rows[1]['decode']['tok_s']/rows[0]['decode']['tok_s'],hc_encode_saved_ms=rows[0]['hc_encode_ms']-rows[1]['hc_encode_ms'],decode_saved_ms=rows[0]['ledger']['decode_ms']-rows[1]['ledger']['decode_ms'])
 out.append(x);print(json.dumps(x))
(r/'E10-qualified-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
