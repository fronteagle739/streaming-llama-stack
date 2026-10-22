import argparse
import json
import sys
from pathlib import Path
from .config import load, ROOT
from .journal import Journal


def main():
    parser = argparse.ArgumentParser(prog='slstack')
    from .envvars import add_flags
    add_flags(parser)
    parser.add_argument('--config')
    parser.add_argument('--host-budget-gib', type=float)
    parser.add_argument('--min-available-gib', type=float)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    installer = sub.add_parser('install')
    for option in ('image', 'video', 'all', 'check', 'sdk'):
        installer.add_argument('--' + option, action='store_true')
    sub.add_parser('mcp')
    sub.add_parser('serve-internal', help=argparse.SUPPRESS)
    train = sub.add_parser('train_lora')
    train.add_argument('--data-folder', required=True)
    train.add_argument('--settings', required=True)
    train.add_argument('--output', required=True)
    merge = sub.add_parser('merge_lora')
    merge.add_argument('--base', required=True)
    merge.add_argument('--lora', action='append', required=True)
    merge.add_argument('--output', required=True)
    merge.add_argument('--exporter', required=True)
    merge.add_argument('--test', action='store_true')
    unmerge = sub.add_parser('unmerge_lora')
    unmerge.add_argument('--entry', required=True)
    prompts = sub.add_parser('run_prompts')
    prompts.add_argument('--folder', required=True)
    prompts.add_argument('--output', required=True)
    prompts.add_argument('--endpoint')
    prompts.add_argument('--model')
    prompts.add_argument('--max-tokens', type=int, default=256)
    start = sub.add_parser('start', aliases=['rtx'])
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
    guarded.add_argument('--cwd', default=str(ROOT))
    guarded.add_argument('argv', nargs=argparse.REMAINDER)
    for child in dict.fromkeys(sub.choices.values()):
        add_flags(child, suppress=True)
    args = parser.parse_args()
    config_keys = {'host_budget_gib', 'min_available_gib', 'model', 'webui', 'image_model', 'video_model',
                   'image_arch', 'video_arch', 'comfyui_models', 'image_mode', 'video_mode'}
    if args.command in ('start', 'rtx'):
        config_keys.add('lora')
    overrides = {k: v for k, v in vars(args).items() if k in config_keys and v is not None}
    if getattr(args, 'mode', None):
        overrides.setdefault('image_mode', args.mode)
        overrides.setdefault('video_mode', args.mode)
    cfg = load(args.config, overrides)
    cfg.setdefault('env_overrides', {}).update({k[4:]: v for k, v in vars(args).items()
                                              if k.startswith('env_') and v is not None})
    if args.command == 'install':
        from .install import install
        result = install(cfg, image=args.image, video=args.video, all_engines=args.all,
                         check=args.check, sdk_only=args.sdk)
        print(json.dumps(result, indent=2))
        return 0 if result['sdk']['ready'] and all(row['ready'] for row in result['engines']) else 1
    from .lease import recover_leases
    for identity in recover_leases(cfg['runtime_root']):
        print('Released dead GPU guardian hold: ' + identity, file=sys.stderr)
    reports = Journal(cfg['runtime_root']).recover() if args.command not in ('check', 'start', 'rtx') else []
    for report in reports:
        print('Recovered stale journal: ' + json.dumps(report), file=sys.stderr)
    if args.command == 'check':
        from .check import check
        rows = check(cfg)
        print(json.dumps(rows, indent=2))
        return 0 if all(row['ok'] or not row['required'] for row in rows) else 1
    if args.command in ('start', 'rtx'):
        from .check import check
        rows = check(cfg)
        failures = [row for row in rows if not row['ok'] and row['required']]
        for warning in [row for row in rows if not row['ok'] and not row['required']]:
            print('Unavailable optional capability: ' + json.dumps(warning), file=sys.stderr)
        if failures:
            print(json.dumps(failures, indent=2), file=sys.stderr)
            return 1
        from .process import OwnedProcess
        from .journal import atomic_json
        effective = Path(cfg['runtime_root']) / 'effective-config.json'
        atomic_json(effective, cfg)
        proc = OwnedProcess(cfg, [sys.executable, '-B', '-m', 'slstack', '--config', str(effective), 'serve-internal'],
                            'router', timeout=86400)
        print(f'Router starting on http://{cfg["host"]}:{cfg["port"]}; log: {proc.log}', flush=True)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.stop()
        return 0
    if args.command == 'serve-internal':
        from .guard import admit
        admit(cfg['host_budget_gib'], cfg['min_available_gib'])
        from .router import serve
        serve(cfg)
        return 0
    if args.command == 'mcp':
        from .guard import admit
        admit(cfg['host_budget_gib'], cfg['min_available_gib'])
        from .router import Stack
        stack = Stack(cfg)
        try:
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    result = stack.mcp(request)
                    if result is not None:
                        print(json.dumps(result), flush=True)
                except (ValueError, TypeError) as exc:
                    print(json.dumps({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': str(exc)}}), flush=True)
        finally:
            stack.engines.close()
        return 0
    if args.command == 'train_lora':
        from .lora import train_lora
        print(json.dumps(train_lora(cfg, args.data_folder, args.settings, args.output), indent=2))
        return 0
    if args.command == 'merge_lora':
        from .lora import merge_lora, parse_lora
        print(json.dumps(merge_lora(cfg, args.base, [parse_lora(x) for x in args.lora], args.output,
                                   args.exporter, test=args.test), indent=2))
        return 0
    if args.command == 'unmerge_lora':
        from .lora import unmerge_lora
        print(json.dumps(unmerge_lora(cfg, args.entry), indent=2))
        return 0
    if args.command == 'run_prompts':
        from .prompts import run_prompts
        print(json.dumps(run_prompts(cfg, args.folder, args.output, args.endpoint, args.model, args.max_tokens), indent=2))
        return 0
    if args.command == 'guarded':
        from .process import OwnedProcess
        command = args.argv[1:] if args.argv and args.argv[0] == '--' else args.argv
        print(json.dumps(OwnedProcess(cfg, command, 'guarded', cwd=args.cwd).wait(), indent=2))
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError, MemoryError) as exc:
        print(f'slstack: {exc}', file=sys.stderr)
        sys.exit(1)
