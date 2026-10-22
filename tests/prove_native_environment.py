"""Exercise actual native bias parsing on CPU, without a model or GPU job."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slstack.config import load, path
from slstack.envvars import add_flags
from slstack.journal import atomic_json
from slstack.process import OwnedProcess


CHILD = '''import json,os,subprocess,sys
print(json.dumps({"received_bias":os.environ["DS4_CACHE_AFFINITY_BIAS"]}),flush=True)
raise SystemExit(subprocess.run(sys.argv[1:]).returncode)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--expect', choices=['model-open', 'cuda-required'], required=True)
    parser.add_argument('--receipt', required=True)
    add_flags(parser)
    args = parser.parse_args()
    cfg = load(args.config)
    cfg.setdefault('env_overrides', {}).update({key[4:]: value for key, value in vars(args).items()
                                               if key.startswith('env_') and value is not None})
    binary = path('engines/ds4-win/ds4-server.exe')
    # Include a conflicting configured engine argument. The table flag must
    # win. A missing relative model stops the CPU control before inference.
    command = [sys.executable, '-B', '-c', CHILD, str(binary), '--cpu', '--ctx', '64',
               '--cache-affinity-bias', '0', '--model', 'slstack-env-proof-no-model.gguf']
    assert not path('slstack-env-proof-no-model.gguf').exists()
    proc = OwnedProcess(cfg, command, 'native-env-proof', scope='ds4-win', timeout=20)
    try:
        proc.wait()
    except RuntimeError:
        pass
    receipt = json.loads(proc.receipt.read_text())
    log = proc.log.read_text(encoding='utf-8', errors='replace')
    assert receipt['exit_code'] != 0 and receipt['reason'] is None, (receipt, log)
    received = json.loads(log.splitlines()[0])
    assert received['received_bias'] == cfg['env_overrides']['DS4_CACHE_AFFINITY_BIAS'], log
    refused = 'cache affinity requires serialized single-session GPU CUDA SSD streaming' in log
    assert refused == (args.expect == 'cuda-required'), log
    if args.expect == 'model-open':
        assert 'slstack-env-proof-no-model.gguf' in log and 'HOST_MEMORY_ADMITTED' in log, log
    atomic_json(Path(args.receipt), {'status': 'PASS', 'expected': args.expect,
                'role': 'actual native CPU startup; no model/GPU inference', 'received': received,
                'child_stdout_stderr': log, 'exit_code': receipt['exit_code'],
                'host_budget_gib': cfg['host_budget_gib'], 'floor_gib': cfg['min_available_gib'],
                'peak_rss': receipt['peak_tree_rss'], 'minimum_available_ram': receipt['min_available'],
                'elapsed_s': receipt['elapsed_s']})
    print(f'PASS: native CPU startup {args.expect}; bias{received["received_bias"]}')


if __name__ == '__main__':
    main()
