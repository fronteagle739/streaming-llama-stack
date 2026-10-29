"""Summarize native prefill logs without equating overlapped fetch time to wait.

Usage: python scripts/analyze_dsv41_prefill.py stderr.log --output summary.json
Layer acquire_ms includes exposed wait plus binding/synchronization. GPU node
events require LLAMA_CUDA_OP_PROFILE=1 and LLAMA_CUDA_OP_PROFILE_NODES=1; those
diagnostic runs are distinct from uninstrumented throughput measurements.
"""
import argparse
import json
from pathlib import Path
import re
import math
import sys


def inventory(header):
    """Payload extents only; bounds come from the existing bounded GGUF reader."""
    sizes = {0:(1,4),1:(1,2),8:(32,34),10:(256,84),12:(256,144),
             16:(256,66),24:(1,1),30:(1,2),39:(32,17)}
    alignment=header['metadata'].get('general.alignment',32)
    if not isinstance(alignment,int) or alignment<=0 or alignment&(alignment-1):
        raise ValueError('Invalid GGUF alignment')
    start=(header['header_bytes']+alignment-1)//alignment*alignment
    rows=[]
    for t in header['tensors']:
        if t['type'] not in sizes:raise ValueError('Unsupported ledger tensor type '+str(t['type']))
        block,size=sizes[t['type']]
        if t['shape'][0]%block:raise ValueError('GGUF block does not divide tensor row')
        nbytes=math.prod(t['shape'])//block*size
        kind=role(t['name'])
        match=re.match(r'blk\.(\d+)\.',t['name'])
        stage=None if kind=='engram_table' else 0 if kind=='embedding' else 41 if kind=='head' else int(match[1])+1 if match else None
        if stage is None and kind!='engram_table':raise ValueError('Unclassified weight '+t['name'])
        rows.append(dict(tensor=t['name'],role=kind,stage=stage,file_offset=start+t['offset'],bytes=nbytes))
    return rows


def attach_inventory(result, rows):
    """Reconstruct legacy full-stage receipts only when their bytes reconcile.

    This is explicitly distinguished from native per-extent instrumentation.
    Never infer which experts routed from a partial stage byte count.
    """
    result['model_inventory']=rows
    for p in result['passes']:
        if p['tensors'] or p['expert_tiles'] or p['expert_hints']:continue
        reconstructed=[]
        for fetch in p['fetches']:
            tensors=[dict(t,source='legacy_full_stage_inventory') for t in rows if t['stage']==fetch['stage']]
            if sum(t['bytes'] for t in tensors)!=fetch['bytes']:
                raise ValueError('Legacy stage bytes do not reconcile with GGUF; native extent ledger required')
            reconstructed.extend(tensors)
        p['tensors']=reconstructed
        p['tensor_bytes_by_role']={r:sum(t['bytes'] for t in reconstructed if t['role']==r) for r in sorted({t['role'] for t in reconstructed})}
        p['tensor_bytes']=sum(p['tensor_bytes_by_role'].values())
        p['tensor_ledger_reconciles']=p['tensor_bytes']==p['fetched_bytes']
    result['tensor_read_totals']={}
    for p in result['passes']:
        for t in p['tensors']:
            total=result['tensor_read_totals'].setdefault(t['tensor'],dict(read_extents=0,bytes=0))
            total['read_extents']+=1;total['bytes']+=t['bytes']
    return result


def fields(line):
    return {key: float(value) if '.' in value else int(value)
            for key, value in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)', line)}


def role(name):
    if 'engram_embd' in name: return 'engram_table'
    if '_exps.weight' in name: return 'routed_experts'
    if name == 'token_embd.weight': return 'embedding'
    if name in ('output.weight', 'output_norm.weight'): return 'head'
    return 'dense_and_other'


def analyze(text):
    passes = []
    nodes = []
    engram_reads = []
    for line in text.splitlines():
        if '[dsv41-fetch] stage=0 ' in line:
            passes.append(dict(fetched_bytes=0, fetch_ms_overlapped=0, layers=[], wall_layers=[], fetches=[], tensors=[], expert_tiles=[], expert_hints=[]))
        if '[dsv41-fetch]' in line and passes:
            row = fields(line)
            passes[-1]['fetched_bytes'] += int(row['bytes'])
            passes[-1]['fetch_ms_overlapped'] += row['fetch_ms']
            passes[-1]['fetches'].append(row)
        if '[dsv41-prefill-experts]' in line and passes:
            row = fields(line)
            passes[-1]['expert_tiles'].append(row)
            passes[-1]['fetched_bytes'] += row['bytes']
            passes[-1]['fetch_ms_overlapped'] += row['fetch_ms']
        if '[dsv41-prefill-hint]' in line and passes:
            row = fields(line)
            passes[-1]['expert_hints'].append(row)
            passes[-1]['fetched_bytes'] += int(row['bytes'])
            passes[-1]['fetch_ms_overlapped'] += row['fetch_ms']
        if '[dsv41-byte-ledger]' in line and passes:
            row = fields(line)
            row.update(re.findall(r'(tensor|source)=(\S+)',line))
            row['role'] = role(row['tensor'])
            passes[-1]['tensors'].append(row)
        if '[dsv41-engram-prefetch]' in line and 'read_bytes=' in line:
            engram_reads.append(fields(line))
        if '[dsv41-perf] layer=' in line and passes:
            row = fields(line)
            row['layer'] = int(row['layer'])
            passes[-1]['layers'].append(row)
        if 'DS4PERF_LAYER_WALL_LEDGER layer=' in line and passes:
            passes[-1]['wall_layers'].append(fields(line))
        if line.startswith('DS4PERF_NODE_PROFILE_HEADER'):
            nodes = []  # a final cumulative dump supersedes earlier dumps
        elif line.startswith('DS4PERF_NODE_PROFILE\t'):
            _, count, ms, key = line.split('\t', 3)
            op = key.split('|', 1)[0]
            if op == 'MUL_MAT_ID':
                stage = 'expert_gemm'
            elif any(word in key for word in ('FLASH_ATTN', 'INDEX', 'indexer', '.attn_', 'dsv41_attention')):
                stage = 'attention_and_indexer'
            else:
                stage = 'other_gpu'
            nodes.append(dict(stage=stage, count=int(count), gpu_ms=float(ms), key=key))
    for p in passes:
        p['totals'] = {key: sum(row[key] for row in p['layers'])
                       for key in (p['layers'][0] if p['layers'] else []) if key != 'layer'}
        p['exposed_acquire_ms'] = p['totals'].get('acquire_ms', 0)
        p['wall_totals'] = {key: sum(row[key] for row in p['wall_layers'])
                            for key in (p['wall_layers'][0] if p['wall_layers'] else []) if key != 'layer'}
        p['routed_fetch_wait_ms'] = sum(row['fetch_ms'] for row in p['expert_tiles'])
        p['hint_fetched_bytes'] = sum(int(row['bytes']) for row in p['expert_hints'])
        p['hint_fetch_ms_overlapped'] = sum(row['fetch_ms'] for row in p['expert_hints'])
        p['exposed_hint_join_ms'] = sum(row['exposed_join_ms'] for row in p['expert_hints'])
        p['tensor_bytes_by_role'] = {r: sum(t['bytes'] for t in p['tensors'] if t['role']==r)
                                     for r in sorted({t['role'] for t in p['tensors']})}
        p['tensor_bytes'] = sum(p['tensor_bytes_by_role'].values())
        p['tensor_ledger_reconciles'] = p['tensor_bytes']==p['fetched_bytes'] if p['tensors'] else None
    gpu = {stage: sum(n['gpu_ms'] for n in nodes if n['stage'] == stage)
           for stage in ('expert_gemm', 'attention_and_indexer', 'other_gpu')}
    return dict(passes=passes, engram_reads=engram_reads, gpu_node_ms=gpu if nodes else None,
                gpu_node_count=len(nodes),
                interpretation='fetch_ms_overlapped is not exposed wait; acquire_ms includes wait and binding. '
                               'GPU node events exclude CPU prepare, transfer, waiting and head startup overhead. '
                               'Attention includes sparse attention, indexer and named attention projections. '
                               'Engram row reads are separate I/O, may precede a pass and are not layer-weight bytes. '
                               'Routed fetch wait occurs inside executor wall; do not add it again to layer wall. '
                               'Hint transfer time is separate overlapped I/O; only its cumulative release/acquire join is exposed wait. '
                               'Hint bytes count toward fetched_bytes and native extent reconciliation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model',type=Path,help='Optional bounded GGUF inventory and legacy full-stage reconciliation')
    args = parser.parse_args()
    data=analyze(args.log.read_text(encoding='utf-8', errors='replace'))
    if args.model:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from slstack.gguf_header import read_header
        attach_inventory(data,inventory(read_header(args.model)))
    result = json.dumps(data, indent=2)+'\n'
    if args.output: args.output.write_text(result, encoding='utf-8')
    else: print(result, end='')


if __name__ == '__main__': main()
