"""Guarded GLM graph regression matrix; synthetic weights, never a speed benchmark.

Generate the three GGUF fixtures with glm5-next/make_tiny_glm5next.py first.
The supplied stack config sets host/commit budgets, free-RAM floor and temp roots.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from slstack.config import load
from slstack.lease import active_holds
from slstack.process import OwnedProcess


def logits(path, rows):
    raw = path.read_bytes()
    shape = tuple(np.frombuffer(raw[:8], dtype='<i4'))
    if shape != (rows, 256):
        raise ValueError('Unexpected synthetic logits dimensions: ' + str(shape))
    result = np.frombuffer(raw[8:], dtype='<f4').reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite GLM logits')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'models', 'binary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    output = args.output.resolve()
    if not output.is_relative_to(Path(cfg['runtime_root']).resolve()):
        raise ValueError('Test output must remain under runtime_root')
    if cfg['job_commit_gib'] > cfg['host_budget_gib']:
        raise ValueError('CPU checks require commit cap no larger than host budget')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for model in ('up-full', 'up-shared', 'ds4-full'):
        for prompt in (64, 320):
            for ubatch, restore in ((64, 0), (16, 0), (16, 1)):
                if cfg.get('hold_file'):
                    holds = active_holds(Path(cfg['hold_file']).read_text(encoding='utf-8-sig'))
                    if holds:
                        raise RuntimeError('Yielding to active engine holds: ' + '\n'.join(holds))
                target = output / f'{model}-{prompt}-{ubatch}-{restore}.bin'
                command = [str(args.binary.resolve()), str(args.models.resolve() / (model + '.gguf')),
                           str(target), str(prompt), str(ubatch), '8', '0', '1234', '0', str(restore)]
                job = OwnedProcess(cfg, command, 'glm53-cpu')
                receipt = job.wait()
                actual = logits(target, prompt + 8)
                reference = logits(output / f'{model}-{prompt}-64-0.bin', prompt + 8)
                if not np.array_equal(actual, reference):
                    raise ValueError('Microbatch or restored-state logits changed')
                layout_diff = 0.0
                if model == 'ds4-full':
                    reference = logits(output / f'up-full-{prompt}-{ubatch}-{restore}.bin', prompt + 8)
                    layout_diff = float(np.abs(actual - reference).max())
                    # A_log is transformed with the backend's exp, while the upstream
                    # fixture uses NumPy's exp. Windows SIMD can differ by a few ULPs.
                    if layout_diff >= 1e-4 or not np.array_equal(actual.argmax(1), reference.argmax(1)):
                        raise ValueError('Upstream/ds4 tensor-layout parity failed')
                row = dict(model=model, prompt=prompt, ubatch=ubatch, restore=bool(restore),
                           rows=prompt + 8, same_layout_max_abs_diff=0.0, layout_max_abs_diff=layout_diff,
                           peak_rss=receipt['peak_tree_rss'], min_available=receipt['min_available'],
                           guardian_receipt=str(job.receipt))
                cases.append(row)
                print('PASS ' + json.dumps(row), flush=True)
    (output / 'results.json').write_text(json.dumps(dict(status='PASS', cases=cases), indent=2), encoding='utf-8')
    print(f'PASS: {len(cases)} CPU graph cases; real-model serving remains unqualified')


if __name__ == '__main__':
    main()
