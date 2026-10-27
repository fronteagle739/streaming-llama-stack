"""Guarded scalar KDA checks on CPU and CUDA; no trained model or speed claim."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from slstack.config import load
from slstack.lease import active_holds
from slstack.process import OwnedProcess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cpu-binary', 'cuda-binary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cuda-library-directory', type=Path)
    args = parser.parse_args()
    cfg = load(args.config)
    if not (0 < cfg['job_commit_gib'] <= cfg['host_budget_gib'] <= 3):
        raise ValueError('Scalar checks require explicit host/commit limits no larger than 3 GiB')
    if cfg['min_available_gib'] < 4 or not cfg.get('gpu_uuid') or not cfg.get('hold_file'):
        raise ValueError('Set a 4 GiB physical floor, selected GPU UUID and shared hold_file')
    output = args.output.resolve()
    if not output.is_relative_to(Path(cfg['runtime_root']).resolve()):
        raise ValueError('Output must remain under runtime_root')
    for binary in (args.cpu_binary, args.cuda_binary):
        if not binary.is_file():
            raise ValueError('Missing scalar check binary')
    if args.cuda_library_directory:
        library = args.cuda_library_directory.resolve()
        if not library.is_dir():
            raise ValueError('CUDA library directory does not exist')
        os.environ['PATH'] = str(library) + os.pathsep + os.environ.get('PATH', '')
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for kind, binary in (('cpu', args.cpu_binary), ('cuda', args.cuda_binary)):
        holds = active_holds(Path(cfg['hold_file']).read_text(encoding='utf-8-sig'))
        if holds:
            raise RuntimeError('Yielding to active engine holds: ' + '\n'.join(holds))
        command = [str(binary.resolve())] + (['--cuda'] if kind == 'cuda' else [])
        job = OwnedProcess(cfg, command, 'glm53-kda-' + kind, gpu=kind == 'cuda')
        receipt = job.wait()
        lines = [line for line in job.log.read_text().splitlines() if line.startswith('KDA backend=')]
        expected_backend = 'CUDA' if kind == 'cuda' else 'CPU'
        if len(lines) != 10 or any(not line.endswith(' PASS') or
                not line.startswith('KDA backend=' + expected_backend) for line in lines):
            raise RuntimeError('Incomplete scalar matrix or incorrect backend placement')
        results.append(dict(backend=kind, cases=lines, guardian_receipt=str(job.receipt),
                            peak_rss=receipt['peak_tree_rss'], min_available=receipt['min_available']))
        print(f'PASS: {kind}, {len(lines)} scalar cases', flush=True)
    (output / 'results.json').write_text(json.dumps(dict(status='PASS', cases=results), indent=2))


if __name__ == '__main__':
    main()
