"""Compare sampled logits only while both WebUI runs have identical inputs.

Each directory contains logits.bin, TURN-native.txt and TURN-proof.json.
Reports excluded samples after the first token divergence, rather than
mistaking distribution drift from different prefixes for kernel error.
"""
import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np


def read_trace(path):
    rows = {}
    with Path(path).open('rb') as stream:
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError('Truncated logits header')
            position, token, count = struct.unpack('<III', header)
            if not 1 <= count <= 129280 or token >= 129280:
                raise ValueError('Invalid logits geometry')
            payload = stream.read(count * 4)
            if len(payload) != count * 4:
                raise ValueError('Truncated logits payload')
            values = np.frombuffer(payload, dtype='<f4').astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite logits')
            if position in rows:
                raise ValueError('Duplicate position in logits trace')
            rows[position] = token, values
    return rows


def distribution_error(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError('Vocabulary mismatch')
    def log_prob(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())
    p, q = log_prob(reference), log_prob(candidate)
    return dict(kl=max(0., float((np.exp(p) * (p-q)).sum())),
                top1_equal=int(reference.argmax() == candidate.argmax()),
                max_abs_logit=float(np.abs(reference-candidate).max()))


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    a, b = read_trace(reference/'logits.bin'), read_trace(candidate/'logits.bin')
    result = {}; previous_equal = True
    for turn in ('recall-50k', 'followup', 'max-quality'):
        proofs = [json.loads((p/(turn+'-proof.json')).read_text(encoding='utf-8'))
                  for p in (reference, candidate)]
        if proofs[0]['input_sha256'] != proofs[1]['input_sha256']:
            raise ValueError('Prompt mismatch')
        steps = []
        for p in (reference, candidate):
            native = (p/(turn+'-native.txt')).read_text(encoding='utf-8')
            steps.append({int(pos)-1: int(token) for token, pos in re.findall(
                r'LLAMADECODE_STEP index=\d+ token=(\d+) position=(\d+)', native)})
        positions = sorted(set(steps[0]) | set(steps[1]))
        mismatch = next((p for p in positions if steps[0].get(p) != steps[1].get(p)), float('inf'))
        samples = []
        if previous_equal:
            for p in positions:
                if p >= mismatch or p not in a or p not in b:
                    continue
                if a[p][0] != b[p][0]:
                    raise ValueError('Trace input token mismatch')
                samples.append(distribution_error(a[p][1], b[p][1]))
        result[turn] = dict(samples=len(samples), identical_prefix_steps=sum(p<mismatch for p in positions) if previous_equal else 0,
            first_different_input=None if mismatch==float('inf') else mismatch,
            mean_kl=float(np.mean([s['kl'] for s in samples])) if samples else None,
            max_kl=max((s['kl'] for s in samples), default=None),
            top1_agreement=float(np.mean([s['top1_equal'] for s in samples])) if samples else None,
            max_abs_logit=max((s['max_abs_logit'] for s in samples), default=None),
            answer_equal=proofs[0]['answer']==proofs[1]['answer'])
        previous_equal &= proofs[0]['answer']==proofs[1]['answer']
    return result


def compare_forced(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    proofs=[json.loads((p/'result.json').read_text(encoding='utf-8')) for p in (reference,candidate)]
    if (any(p.get('status')!='PASS' for p in proofs) or
            not re.fullmatch('[0-9a-f]{64}',proofs[0].get('token_sha256','')) or
            proofs[0]['token_sha256']!=proofs[1].get('token_sha256') or
            proofs[0]['prompt_tokens']!=proofs[1].get('prompt_tokens')):
        raise ValueError('Forced input identity or run status differs')
    result={}
    for phase,key,filename in (('quality','reply','logits.f32'),('speed','speed_reply','speed-logits.f32')):
        count=proofs[0][key]['vectors']
        if not 1<=count<=4096 or count!=proofs[1][key]['vectors']:
            raise ValueError('Forced vector count differs')
        files=[p/filename for p in (reference,candidate)]
        if any(p.stat().st_size!=count*129280*4 for p in files):
            raise ValueError('Forced logits extent differs')
        arrays=[np.memmap(p,dtype='<f4',mode='r',shape=(count,129280)) for p in files]
        errors=[]
        for i in range(count):
            rows=[np.array(a[i],dtype=np.float64) for a in arrays]
            if any(not np.isfinite(row).all() for row in rows):
                raise ValueError('Nonfinite forced logits')
            errors.append(distribution_error(*rows))
        warm=[]
        for p in proofs:
            times=p[key]['step_ms'][100:]
            if not times:raise ValueError('Too few warm forced steps')
            warm.append(1000*len(times)/sum(times))
        result[phase]=dict(vectors=count,mean_kl=float(np.mean([e['kl'] for e in errors])),
            max_kl=max(e['kl'] for e in errors),top1_agreement=float(np.mean([e['top1_equal'] for e in errors])),
            max_abs_logit=max(e['max_abs_logit'] for e in errors),reference_warm_tok_s=warm[0],candidate_warm_tok_s=warm[1],
            raw_slowdown_percent=100*(warm[0]/warm[1]-1))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference');parser.add_argument('candidate')
    parser.add_argument('--forced',action='store_true',help='Compare existing decode_forced raw vectors and input receipts')
    args=parser.parse_args()
    print(json.dumps((compare_forced if args.forced else compare)(args.reference,args.candidate),indent=2))
