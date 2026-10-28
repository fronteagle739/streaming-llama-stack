"""Summarize a separate, synchronized CUDA profiling run without host paths."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def summarize(measurement, trace):
    row = json.loads(measurement.read_text(encoding='utf-8'))
    if row.get('status') != 'PASS' or row.get('profiled') is not True or row.get('decode'):
        raise ValueError('A successful prefill-only profiling measurement is required')
    nodes = []
    pending = None
    ledger = defaultdict(float)
    fetched = layers = 0
    with trace.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            # Dumps are cumulative. Use only the final complete node table.
            if line.startswith('DS4PERF_NODE_PROFILE_HEADER\t'):
                pending = []
            elif line.startswith('DS4PERF_NODE_PROFILE\t'):
                if pending is None:
                    raise ValueError('CUDA node profile lacks its header')
                parts = line.rstrip().split('\t', 3)
                pending.append((int(parts[1]), float(parts[2]), parts[3]))
            elif line.startswith('DS4PERF_NODE_PROFILE_END\t'):
                expected = int(re.search(r'rows=(\d+)', line)[1])
                if pending is None or len(pending) != expected:
                    raise ValueError('Truncated CUDA node profile')
                nodes = pending
            elif 'DS4PERF_LAYER_WALL_LEDGER ' in line:
                layers += 1
                for key, value in re.findall(r'(\w+)=([-0-9.]+)', line):
                    if key != 'layer':
                        ledger[key] += float(value)
            elif '[dsv41-fetch] ' in line:
                fetched += int(re.search(r'bytes=(\d+)', line)[1])
    if not nodes or not layers:
        raise ValueError('Missing complete node or layer profile')
    gpu = defaultdict(float)
    expert_flops = 0
    for count, ms, description in nodes:
        op = description.split('|', 1)[0]
        if op == 'MUL_MAT_ID' and '_exps.weight' in description:
            category = 'expert_gemm'
            shape = [int(x) for x in re.search(r'\|dst=([0-9,]+)', description)[1].split(',')]
            k = int(re.search(r'\|ashape=(\d+),', description)[1])
            expert_flops += count * 2 * k * shape[0] * shape[1] * shape[2] * shape[3]
        elif op.startswith('FLASH_ATTN'):
            category = 'sparse_attention_core'
        elif op in ('DSV41_INDEX_SCORE', 'DSV41_CANDIDATE_BLOCKS', 'DSV41_SELECTED_MASK') or (
                op == 'TOP_K' and 'dsv41_index_scores' in description):
            category = 'indexer_core'
        else:
            category = 'other_gpu'
        gpu[category] += ms / 1000
    wall = row['prefill_seconds']
    wait = ledger['acquire_ms'] / 1000
    residual = wall - wait - sum(gpu.get(key, 0) for key in (
        'expert_gemm', 'sparse_attention_core', 'indexer_core'))
    if residual < 0:
        raise ValueError('Stage counters exceed measured wall')
    return dict(status='PASS', arm=row['arm'], label=row['label'],
                prompt_tokens=row['prompt_tokens'], sweeps=row['plan']['sweeps'],
                diagnostic_only=True, prefill_seconds=wall,
                stage_seconds=dict(stream_wait=wait, expert_gemm=gpu['expert_gemm'],
                                   sparse_attention_core=gpu['sparse_attention_core'],
                                   indexer_core=gpu['indexer_core'], other_wall=residual),
                gpu_seconds=dict(gpu), layer_count=layers, fetched_bytes=fetched,
                routed_expert_flops=expert_flops,
                routed_effective_tflops=expert_flops / gpu['expert_gemm'] / 1e12,
                nominal_5090_dense_fp4_tflops=1676,
                fp4_roof_ratio_reference_only=(expert_flops / gpu['expert_gemm'] / 1e12 / 1676
                                              if row['arm'] == 'proxy' else None),
                useful_fetch_bytes_per_wall_second=fetched / wall,
                layer_ledger_ms=dict(ledger),
                definitions=dict(stream_wait='Sum of exposed per-layer acquire wall time; transfer-duration sums overlap compute.',
                                 expert_gemm='MUL_MAT_ID nodes whose weight origin is a routed expert.',
                                 sparse_attention_core='FLASH_ATTN nodes; projection, RoPE and packing remain in other.',
                                 indexer_core='Index scores, their top-k, candidate blocks and selected masks.',
                                 other_wall='Wall residual including other GPU work, CPU scheduling, carry/Engram work and startup.'),
                caveat='CUDA profiling synchronizes operations. These diagnostic walls are not headline throughput or an unsynchronized overlap decomposition.',
                producer_sha256=row['producer_sha256'], decode_library_sha256=row['decode_library_sha256'],
                model_sha256=row['model_sha256'], input_sha256=row['input_sha256'],
                measurement_sha256=hashlib.sha256(measurement.read_bytes()).hexdigest(),
                trace_sha256=hashlib.file_digest(trace.open('rb'), 'sha256').hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.measurement, args.trace), indent=2) + '\n', encoding='utf-8')
