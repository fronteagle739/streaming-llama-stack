"""Difference cumulative CUDA snapshots and report host phases per layer.

Use diagnostic runs with LLAMA_CUDA_OP_PROFILE{,_NODES}=1 and
LLAMADECODE_TILE_TIMING=1. Host and GPU clocks overlap and must not be added.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def fields(line):
    return {k: float(v) for k, v in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', line)}


def stage(key):
    op = key.split('|', 1)[0]
    if op == 'MUL_MAT_ID':
        return 'expert_gemm_gpu_ms'
    if op in ('CPY', 'SET_ROWS') and any(s in key for s in
            ('dsv41_compressed_l', 'dsv41_index_l', 'dsv41_raw_l', 'dsv41_pool_')):
        return 'cache_publication_gpu_ms'
    if any(s in key.lower() for s in ('flash_attn', 'index', '.attn_', 'dsv41_attention')):
        return 'attention_indexer_gpu_ms'
    return 'other_gpu_ms'


def analyze(lines):
    layers = defaultdict(lambda: defaultdict(float))
    previous, snapshot = {}, {}
    spilled = {}
    registrations = 0
    pass_index = -1
    for line in lines:
        if '[dsv41-fetch] stage=0 ' in line:
            pass_index += 1
        if line.startswith('DS4PERF_TILE_HOST_REGISTRATION '):
            registrations += 1
        if line.startswith('DS4PERF_NODE_PROFILE_HEADER'):
            snapshot = {}
        elif line.startswith('DS4PERF_NODE_PROFILE\t'):
            _, count, ms, key = line.rstrip('\r\n').split('\t', 3)
            snapshot[key] = float(ms)
        elif line.startswith('SINGLEPASS_LAYER_NODE_SNAPSHOT '):
            layer = int(fields(line)['layer'])
            row = layers[pass_index, layer]
            if previous.keys()-snapshot.keys():
                raise ValueError('CUDA cumulative snapshot lost previously observed nodes')
            for key, value in snapshot.items():
                delta = value - previous.get(key, 0)
                if delta < -1e-5:
                    raise ValueError('CUDA cumulative profile moved backwards')
                row[stage(key)] += max(0, delta)
            previous = snapshot.copy()
        elif line.startswith('LLAMADECODE_TILE_TIMING '):
            f = fields(line);row = layers[pass_index, int(f.pop('layer'))]
            row['tiles'] += 1
            row['registered_tiles'] += int(f.get('host_register_ms', 0) > 0)
            for k, v in f.items():
                if k.endswith('_ms'):
                    row[k] += v
        elif line.startswith('LLAMADECODE_PREFILL_SPILL_TILE '):
            f = fields(line);row = layers[pass_index, int(f['layer'])]
            row['spill_h2d_host_ms'] += f['h2d_ms']
            row['spill_d2h_host_ms'] += f['d2h_ms']
            rows = int(f['count']);layer = int(f['layer'])
            row['spill_tiles'] += 1
            row['spill_d2h_bytes'] += rows*40976
            row['spill_h2d_bytes'] += rows*40976 if layer else 0
            spilled[pass_index, layer, int(f['start'])] = rows
        elif line.startswith('DS4PERF_RESIDENT_IDS_COMPLETE '):
            f = fields(line);layer = int(f['layer'])
            rows = spilled.pop((pass_index, layer, int(f['start'])), None)
            if rows is not None and layer:
                # Actual compact widths from the executor, not a fixed guess.
                query_source = layer in (2, 8, 14, 20, 24, 28, 32, 36)
                ids = (f['selected_width'] if layer >= 2 and not query_source else 0)
                ids += f['candidate_width'] if layer > 20 else 0
                layers[pass_index, layer]['spill_h2d_bytes'] += rows*ids*4
        elif line.startswith('DS4PERF_LAYER_WALL_LEDGER '):
            f = fields(line);row = layers[pass_index, int(f.pop('layer'))]
            for k, v in f.items():
                row[k] += v
    result = [dict(pass_index=p, layer=l, **dict(v)) for (p, l), v in sorted(layers.items())]
    totals = defaultdict(float)
    for row in result:
        for key, value in row.items():
            if key not in ('pass_index', 'layer'):
                totals[key] += value
    return dict(layers=result, totals=dict(totals), registration_calls=registrations,
        spill_id_witnesses_missing=len(spilled), interpretation=
        'All times are ms. GPU node times are from synchronizing diagnostics, not throughput runs. '
        'host_register_ms is inside host_bind_ms and host_prepare_ms. '
        'host_sync_ms is inside compute_ms. Spill copy walls are inside upload/readback. '
        'Cache publication includes named cache writes; unnamed writes remain other GPU. '
        'Spill bytes count the native BF16 hidden/F32 pre contract and witnessed compact widths, excluding common seed/ID mirrors. '
        'The first layer snapshot also contains seed operations. Never add overlapping clocks.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path);p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.trace.open(encoding='utf-8', errors='replace') as source:
        result = analyze(source)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
