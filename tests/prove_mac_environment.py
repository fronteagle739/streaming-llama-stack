"""Run the actual Metal factory under launcher-built environment and RSS guard."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slstack.config import load
from slstack.envvars import add_flags
from slstack.journal import atomic_json
from slstack.process import OwnedProcess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--expect', choices=['admitted', 'refused'], required=True)
    parser.add_argument('--receipt', required=True)
    add_flags(parser)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('This proof requires the actual macOS Metal runtime')
    cfg = load(args.config)
    cfg.setdefault('env_overrides', {}).update({key[4:]: value for key, value in vars(args).items()
                                               if key.startswith('env_') and value is not None})
    proc = OwnedProcess(cfg, [str(Path(args.binary).resolve()), args.expect], 'mac-env-proof',
                        scope='mac-ds4', timeout=20)
    receipt = proc.wait()
    log = proc.log.read_text(encoding='utf-8', errors='replace')
    received = next(json.loads(line) for line in log.splitlines() if line.startswith('{'))
    assert received['host_bytes'] == cfg['env_overrides']['DS4_NATIVE_SESSION_HOST_BYTES'], log
    assert received['wired'] == '1', log
    assert 'MAC_FACTORY_' + args.expect.upper() in log, log
    atomic_json(Path(args.receipt), {'status': 'PASS', 'expected': args.expect,
                'role': 'actual native Metal owned/wired allocation factory; no model inference',
                'received': received, 'child_stdout_stderr': log, 'exit_code': receipt['exit_code'],
                'host_budget_gib': cfg['host_budget_gib'], 'floor_gib': cfg['min_available_gib'],
                'peak_rss': receipt['peak_tree_rss'], 'minimum_available_ram': receipt['min_available'],
                'elapsed_s': receipt['elapsed_s']})
    print(f'PASS: Mac factory {args.expect}; host budget{received["host_bytes"]} bytes')


if __name__ == '__main__':
    main()
