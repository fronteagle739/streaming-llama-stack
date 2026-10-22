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
    start = sub.add_parser('start')
    start.add_argument('--webui', action='store_true', default=None)
    start.add_argument('--model')
    start.add_argument('--image-model')
    start.add_argument('--video-model')
    start.add_argument('--image-arch')
    start.add_argument('--video-arch')
    start.add_argument('--comfyui-models')
    start.add_argument('--mode', choices=['streaming', 'resident'])
    start.add_argument('--image-mode', choices=['streaming', 'resident'])
    start.add_argument('--video-mode', choices=['streaming', 'resident'])
    start.add_argument('--lora', action='append')
    guarded = sub.add_parser('guarded')
    guarded.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ('config', 'command', 'argv', 'mode') and v is not None}
    if getattr(args, 'mode', None):
        overrides['image_mode'] = overrides['video_mode'] = args.mode
    cfg = load(args.config, overrides)
    Journal(cfg['runtime_root']).recover()
    if args.command == 'check':
        from .check import check
        rows = check(cfg)
        print(json.dumps(rows, indent=2))
        return 0 if all(row['ok'] for row in rows) else 1
    if args.command == 'start':
        from .check import check
        rows = check(cfg)
        failures = [row for row in rows if not row['ok']]
        if failures:
            print(json.dumps(failures, indent=2), file=sys.stderr)
            return 1
        from .router import serve
        serve(cfg)
        return 0
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
