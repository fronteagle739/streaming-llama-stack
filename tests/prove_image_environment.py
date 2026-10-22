"""Real child proof for the reused image runtime's in-process memory watchdog.

Uses no model or GPU. A floor above physical RAM must exit137; the admitted
floor must keep the child alive. The outer guardian remains active in both cases.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slstack.config import load
from slstack.envvars import add_flags
from slstack.journal import atomic_json
from slstack.process import OwnedProcess


CHILD = '''import json,os,time
from cosmos_windows import mem_guard
print(json.dumps({"received_floor_gb":os.environ["MEM_GUARD_MIN_FREE_GB"],"received_cap_gb":os.environ["MEM_GUARD_MAX_PROC_GB"]}),flush=True)
mem_guard.start_watchdog(poll_seconds=.05,label="image-environment-proof")
time.sleep(.25)
print("IMAGE_GUARD_ADMITTED",flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--expect', choices=['admitted', 'refused'], required=True)
    parser.add_argument('--receipt', required=True)
    add_flags(parser)
    args = parser.parse_args()
    cfg = load(args.config)
    cfg.setdefault('env_overrides', {}).update({key[4:]: value for key, value in vars(args).items()
                                               if key.startswith('env_') and value is not None})
    proc = OwnedProcess(cfg, [sys.executable, '-B', '-c', CHILD], 'image-env-proof', scope='image', timeout=20)
    try:
        proc.wait()
    except RuntimeError:
        pass
    receipt = json.loads(proc.receipt.read_text())
    log = proc.log.read_text(encoding='utf-8', errors='replace')
    expected = 0 if args.expect == 'admitted' else 137
    assert receipt['exit_code'] == expected and receipt['reason'] is None, (receipt, log)
    assert ('IMAGE_GUARD_ADMITTED' in log) == (args.expect == 'admitted'), log
    assert ('MEM_GUARD_TRIP' in log) == (args.expect == 'refused'), log
    atomic_json(Path(args.receipt), {'status': 'PASS', 'expected': args.expect,
                'role': 'reused image runtime memory watchdog; no model/GPU', 'child_stdout_stderr': log,
                'exit_code': receipt['exit_code'], 'host_budget_gib': cfg['host_budget_gib'],
                'floor_gib': cfg['min_available_gib'], 'peak_rss': receipt['peak_tree_rss'],
                'minimum_available_ram': receipt['min_available'], 'elapsed_s': receipt['elapsed_s']})
    print(f'PASS: image runtime watchdog {args.expect}, exit{receipt["exit_code"]}')


if __name__ == '__main__':
    main()
