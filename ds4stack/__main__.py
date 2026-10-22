import argparse
import json
import sys
from .config import load, ROOT
from .journal import Journal


def main():
    parser = argparse.ArgumentParser(prog='ds4stack')
    parser.add_argument('--config')
    parser.add_argument('--host-budget-gib', type=float)
    parser.add_argument('--min-available-gib', type=float)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    guarded = sub.add_parser('guarded')
    guarded.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load(args.config, {'host_budget_gib': args.host_budget_gib, 'min_available_gib': args.min_available_gib})
    Journal(cfg['runtime_root']).recover()
    if args.command == 'check':
        from .check import check
        rows = check(cfg)
        print(json.dumps(rows, indent=2))
        return 0 if all(row['ok'] for row in rows) else 1
    if args.command == 'guarded':
        from .process import OwnedProcess
        command = args.argv[1:] if args.argv and args.argv[0] == '--' else args.argv
        print(json.dumps(OwnedProcess(cfg, command, 'guarded', cwd=ROOT).wait(), indent=2))
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f'ds4stack: {exc}', file=sys.stderr)
        sys.exit(1)
