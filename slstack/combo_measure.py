"""Guarded hardware measurements using the actual native OpenAI stream/logs."""
from __future__ import annotations
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from .combo_session import ComboSession
from .journal import atomic_json


def require_parity(cfg):
    root=Path(cfg['runtime_root'])
    native=json.loads((root/'native-parity.json').read_text())
    ultra=json.loads((root/'ultra-parity.json').read_text())
    if native.get('status')!='PASS' or any(native.get(k)!=384 for k in ('roundtrip_byte_exact','rewind_byte_exact')):
        raise ValueError('Mandatory native full-vector parity is incomplete')
    if ultra.get('status')!='PASS' or any(ultra.get(k)!=cfg[k] for k in ('native_sha256','producer_sha256')):
        raise ValueError('Mandatory ULTRA import parity is absent or from different binaries')
    for key in ('comparison','appended_comparison'):
        if ultra.get(key,{}).get('vectors')!=384:raise ValueError('Full and native-base append comparisons are both mandatory')
    return dict(native=native,ultra=ultra)


def milestones(raw):
    rows={}
    for values in re.findall(rb'DS4COMBO_DECODE_PROGRESS tokens=(\d+) decode_seconds=([0-9.]+) cumulative_tok_s=([0-9.]+) warm500_tok_s=([0-9.]+)',raw):
        count=int(values[0])
        if count in rows:raise ValueError('Multiple native decode requests share this measurement log')
        rows[count]=dict(tokens=count,seconds=float(values[1]),tok_s=float(values[2]),warm500_tok_s=float(values[3]))
    return {str(k):v for k,v in rows.items()}


def stream_completion(native,body):
    body={**body,'stream':True,'stream_options':{'include_usage':True}}
    request=urllib.request.Request(native.url+'/v1/chat/completions',data=json.dumps(body,allow_nan=False,separators=(',',':')).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+native.token})
    started=time.monotonic();first=None;content=[];reasoning=[];usage=None;size=0;finish=None
    with urllib.request.urlopen(request,timeout=native.cfg['request_timeout_s']) as response:
        for line in response:
            size+=len(line)
            if size>8<<20 or len(line)>65536:raise ValueError('Native measurement stream exceeded its8MiB metadata budget')
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]':break
            row=json.loads(data)
            if row.get('error'):raise RuntimeError('Native stream error: '+str(row['error']))
            if row.get('usage'):usage=row['usage']
            for choice in row.get('choices',[]):
                delta=choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content'):
                    if first is None:first=time.monotonic()
                if delta.get('content'):content.append(delta['content'])
                if delta.get('reasoning_content'):reasoning.append(delta['reasoning_content'])
                if choice.get('finish_reason'):finish=choice['finish_reason']
    if first is None:raise ValueError('Native stream produced no answer or reasoning token')
    message=dict(role='assistant',content=''.join(content))
    if reasoning:message['reasoning_content']=''.join(reasoning)
    return dict(message=message,usage=usage,finish_reason=finish,
                native_api_first_token_seconds=first-started,stream_seconds=time.monotonic()-started)


def measure(cfg,body,output,route=None,followups=()):
    require_parity(cfg)
    # The controller must bind CUDA visibility before its first identity query.
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    session=ComboSession(cfg,qualification_route=route)
    proof=dict(status='RUNNING',native_sha256=cfg['native_sha256'],producer_sha256=cfg['producer_sha256'],
               provider_schema=cfg['provider_schema'],identity=cfg['combo_identity'],context=cfg['context'],
               native_prefill_chunk=cfg['native_prefill_chunk'],turns=[])
    current=body
    try:
        for index in range(1+len(followups)):
            started=time.monotonic()
            with session.prepare(current) as (native,rendered,metrics):
                result=stream_completion(native,rendered)
                result['total_first_token_seconds']=metrics['prepared_seconds']+result['native_api_first_token_seconds']
                directory=native.directory
            result['metrics']=metrics
            result['wall_seconds']=time.monotonic()-started
            result['native_directory']=str(directory)
            result['engine_milestones']=milestones((directory/'stderr.log').read_bytes())
            if metrics.get('ultra_directory'):
                producer=json.loads((Path(metrics['ultra_directory'])/'producer.json').read_text())
                result['producer_receipt']=producer
                duration=producer.get('phase_ms',{}).get('embedding_40_layers_head')
                if duration and duration>0:result['ultra_prefill_tok_s']=metrics['new_tokens']/(duration/1000)
            proof['turns'].append(result);atomic_json(output,proof)
            if index<len(followups):
                spec=followups[index]
                current={**body,'messages':[*(current['messages']),result['message'],dict(role='user',content=spec['content'])],
                         'max_tokens':spec.get('max_tokens',128),'include_previous_reasoning':spec.get('include_previous_reasoning',True)}
        proof['status']='PASS'
    except BaseException as exc:
        proof['status']='FAIL';proof['error']=repr(exc);raise
    finally:
        session.close();atomic_json(output,proof)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--case',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--route',choices=['auto','native','ultra'],default='auto')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());case=json.loads(Path(args.case).read_text())
    measure(cfg,case['body'],args.output,None if args.route=='auto' else args.route,case.get('followups',[]))
