"""Guarded tiny-fixture CPU/CUDA numerical matrix; never a model benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from slstack.config import load, environment
from slstack.lease import active_holds
from slstack.process import OwnedProcess


def logits(path, rows):
    raw = path.read_bytes()
    if tuple(np.frombuffer(raw[:8], dtype='<i4')) != (rows, 256):
        raise ValueError('Unexpected synthetic logits dimensions')
    value = np.frombuffer(raw[8:], dtype='<f4').reshape(rows, 256)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite logits')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'models', 'cpu-binary', 'cuda-binary', 'cpu-ops',
                 'cuda-ops', 'cpu-kda', 'cuda-kda', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cuda-library-directory', type=Path)
    args = parser.parse_args()
    cfg = load(args.config)
    if not (0 < cfg['job_commit_gib'] <= cfg['host_budget_gib'] <= 3):
        raise ValueError('Set explicit host/commit caps no larger than 3 GiB')
    if cfg['min_available_gib'] < 4 or not cfg.get('gpu_uuid') or not cfg.get('hold_file'):
        raise ValueError('Set a 4 GiB physical floor, selected GPU and shared hold ledger')
    output = args.output.resolve()
    if not output.is_relative_to(Path(cfg['runtime_root']).resolve()):
        raise ValueError('Output must remain under runtime_root')
    for name in ('cpu_binary', 'cuda_binary', 'cpu_ops', 'cuda_ops', 'cpu_kda', 'cuda_kda'):
        if not getattr(args, name).is_file():
            raise ValueError('Missing binary: ' + name)
    if args.cuda_library_directory:
        os.environ['PATH'] = str(args.cuda_library_directory.resolve()) + os.pathsep + os.environ.get('PATH', '')
    # Presence, even with value "0", disables capture in this backend.
    os.environ.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    output.mkdir(parents=True, exist_ok=False)
    results = dict(scalar=[], graphs=[], comparisons=[], tolerance=1e-2)

    def run(label, command, cuda=False, graphs=False):
        holds = active_holds(Path(cfg['hold_file']).read_text(encoding='utf-8-sig'))
        if holds:
            raise RuntimeError('Yielding to active engine holds: ' + '\n'.join(holds))
        env = {**cfg['env_overrides'], 'GGML_CUDA_GRAPH_TRACE': '1'}
        env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
        if not graphs:
            env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
        conf = {**cfg, 'env_overrides': env}
        # The producer profile deliberately forces capture off. Standalone
        # backend checks use the native engine profile, with the same memory
        # floor and GPU lease. Verify the resolved child env before launching.
        child_env = environment(conf, 'ds4-win')
        if ('GGML_CUDA_DISABLE_GRAPHS' not in child_env) != graphs:
            raise ValueError('Launcher profile overrides the requested graph arm')
        if child_env.get('GGML_CUDA_GRAPH_TRACE') != '1':
            raise ValueError('Capture witness is absent from the child environment')
        job = OwnedProcess(conf, command, label,
                           gpu=cuda, scope='ds4-win', telemetry=False)
        receipt = job.wait()
        log = job.log.read_text(errors='replace')
        return dict(peak_rss=receipt['peak_tree_rss'], min_available=receipt['min_available'],
                    guardian_receipt=str(job.receipt)), log

    for backend, graphs in (('cpu', False), ('cuda', False), ('cuda', True)):
        for operation, count in (('ops', 49), ('kda', 10)):
            binary = getattr(args, backend + '_' + operation)
            receipt, log = run('glm53-' + operation, [str(binary.resolve())] +
                               (['--cuda'] if backend == 'cuda' else []), backend == 'cuda', graphs)
            prefix = 'OP backend=' if operation == 'ops' else 'KDA backend='
            lines = [line for line in log.splitlines() if line.startswith(prefix)]
            if len(lines) != count or any(not line.endswith(' PASS') for line in lines):
                raise ValueError('Incomplete scalar matrix')
            launches = log.count('CUDA_GRAPH_EXEC mode=launch')
            if backend == 'cuda' and operation == 'ops' and bool(launches) != graphs:
                raise ValueError('CUDA graph arm did not engage')
            results['scalar'].append(dict(backend=backend, graphs=graphs, operation=operation,
                                          cases=lines, graph_launch_markers=launches, **receipt))
            print(f'PASS {backend} graphs={graphs} {operation}: {count} scalar cases', flush=True)

    for model in ('ds4-full', 'up-full', 'up-shared'):
        for prompt in (64, 65, 320):
            reference = None
            prior = {}
            arms = [('cpu', 16, True, False, False)] + [
                ('cuda', ub, restore, graphs, strict)
                for strict in (False, True) for graphs in (False, True)
                for ub, restore in ((64, False), (16, False), (16, True))]
            for backend, ub, restore, graphs, strict in arms:
                label = f'{model}-{prompt}-{backend}-{ub}-{int(restore)}-{int(graphs)}-{int(strict)}'
                target = output / (label + '.bin')
                binary = args.cuda_binary if backend == 'cuda' else args.cpu_binary
                command = [str(binary.resolve()), str(args.models.resolve() / (model + '.gguf')),
                           str(target), str(prompt), str(ub), '8', '0', '1234',
                           '99' if backend == 'cuda' else '0', str(int(restore)), '-', str(int(strict))]
                receipt, log = run('glm53-graph', command, backend == 'cuda', graphs)
                marker = 'graphs_disabled=unset' if graphs else 'graphs_disabled=1'
                if marker not in log or (strict and 'PROBE F32 engaged' not in log):
                    raise ValueError('Precision/capture environment arm did not engage')
                actual = logits(target, prompt + 8)
                if reference is None:
                    reference = actual
                error = float(np.abs(actual - reference).max())
                same_top = int((actual.argmax(1) == reference.argmax(1)).sum())
                # Same threshold as the pre-existing compare_logits.py. Every
                # top token must agree; do not tune tolerance to a failing case.
                if error >= results['tolerance'] or same_top != prompt + 8:
                    raise ValueError(f'Numerical comparison failed: {label} max_abs={error} top={same_top}')
                if backend == 'cuda':
                    key = (ub, graphs, strict)
                    if restore:
                        if not np.array_equal(actual, prior[key]):
                            raise ValueError('Restored-state logits changed: ' + label)
                    else:
                        prior[key] = actual
                results['graphs'].append(dict(model=model, prompt=prompt, backend=backend, ubatch=ub,
                    restore=restore, graphs=graphs, strict=strict, max_abs=error,
                    top_agree=same_top, rows=prompt + 8, **receipt))
                # Compare capture/default/F32 and microbatch arms directly, too.
                if backend == 'cuda' and ub == 16 and not restore:
                    wide = logits(output / f'{model}-{prompt}-cuda-64-0-{int(graphs)}-{int(strict)}.bin', prompt+8)
                    batch_error = float(np.abs(actual-wide).max())
                    if batch_error >= results['tolerance'] or not np.array_equal(actual.argmax(1), wide.argmax(1)):
                        raise ValueError('Microbatch comparison failed: ' + label)
                    results['comparisons'].append(dict(label=label, microbatch_max_abs=batch_error))
                (output / 'results.json').write_text(json.dumps(results, indent=2))
                print(f'PASS {label} max_abs={error:.8g} top={same_top}/{prompt+8}', flush=True)
    print(f"PASS: {len(results['graphs'])} graph runs; synthetic numerical acceptance only", flush=True)


if __name__ == '__main__':
    main()
