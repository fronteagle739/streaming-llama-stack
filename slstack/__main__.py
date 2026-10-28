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
    parser.add_argument('--runtime-min-free-gib', type=float, help='Runtime available physical RAM floor (default 0.5 GiB; 0 disables it)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    mlx_setup = sub.add_parser('setup-mlx', help='Install the retained MLX-VLM fork and native GLM kernels')
    mlx_setup.add_argument('--check', action='store_true')
    mlx_setup.add_argument('--no-native-glm', action='store_true')
    mlx_setup.add_argument('--wheelhouse', help='Use a local wheel directory without accessing a package index')
    from .backends import BACKENDS
    backend_setup = sub.add_parser('setup-backend', help='Install a pinned decoder and its build dependencies')
    backend_setup.add_argument('--backend', choices=[*BACKENDS,'all'], required=True)
    backend_setup.add_argument('--check', action='store_true')
    backend_setup.add_argument('--fetch-only', action='store_true')
    backend_setup.add_argument('--build-only', action='store_true',help='Compile without GPU smoke checks; verification stays pending')
    backend_setup.add_argument('--offline', action='store_true')
    backend_setup.add_argument('--cache-dir')
    backend_setup.add_argument('--wheelhouse')
    backend_setup.add_argument('--jobs',type=int,choices=(1,2),default=2)
    decoder = sub.add_parser('decode', help='Start a configured decoder on its serving host')
    decoder.add_argument('--model')
    decoder.add_argument('--backend', choices=list(BACKENDS))
    decoder.add_argument('--check', action='store_true')
    sub.add_parser('decode-internal', help=argparse.SUPPRESS)
    installer = sub.add_parser('install')
    for option in ('image', 'video', 'all', 'check', 'sdk'):
        installer.add_argument('--' + option, action='store_true')
    sub.add_parser('mcp')
    sub.add_parser('serve-internal', help=argparse.SUPPRESS)
    sub.add_parser('mac-internal', help=argparse.SUPPRESS)
    sub.add_parser('broker-internal', help=argparse.SUPPRESS)
    pair = sub.add_parser('pair')
    pair.add_argument('--mac', required=True)
    pair.add_argument('--pair-token')
    pair.add_argument('--model')
    for name in ('mac', 'mac-setup'):
        child = sub.add_parser(name)
        child.add_argument('--model')
        child.add_argument('--pair-token', help='Optional shared token for either-machine-first startup')
        child.add_argument('--mtp-model')
        child.add_argument('--backend', choices=['ds4', 'mlx-serve'], default=None)
        child.add_argument('--port', dest='mac_port', type=int)
        child.add_argument('--engine-port', dest='mac_engine_port', type=int)
        child.add_argument('--zig')
        child.add_argument('--check', action='store_true')
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
    start.add_argument('--models', help='Comma-separated local model paths')
    start.add_argument('--resident', help='One or two registered model names/paths; normal switching until dual residency lands')
    start.add_argument('--image-model')
    start.add_argument('--video-model')
    start.add_argument('--image-arch')
    start.add_argument('--video-arch')
    start.add_argument('--comfyui-models')
    start.add_argument('--mac', help='Native Mac receiver endpoint')
    start.add_argument('--pair-token', help='Token printed by the Mac; stored in the private runtime')
    start.add_argument('--mode', choices=['accurate', 'fast', 'ultra', 'hyper'], dest='native_tier',
                       help='RTX prefill arithmetic profile (default: accurate)')
    start.add_argument('--image-mode', choices=['streaming', 'resident'])
    start.add_argument('--video-mode', choices=['streaming', 'resident'])
    start.add_argument('--lora', action='append')
    start.add_argument('--serving', choices=['single_engine', 'native'],
                       help='Single-RTX serving: single_engine (default; ULTRA producer prefill + ds4 decode in one process) or native (ds4-win server)')
    guarded = sub.add_parser('guarded')
    guarded.add_argument('--cwd', default=str(ROOT))
    guarded.add_argument('argv', nargs=argparse.REMAINDER)
    for child in dict.fromkeys(sub.choices.values()):
        child.add_argument('--config', default=argparse.SUPPRESS)
        child.add_argument('--host-budget-gib', type=float, default=argparse.SUPPRESS)
        child.add_argument('--min-available-gib', type=float, default=argparse.SUPPRESS)
        child.add_argument('--runtime-min-free-gib', type=float, default=argparse.SUPPRESS)
        add_flags(child, suppress=True)
    args = parser.parse_args()
    config_keys = {'host_budget_gib', 'min_available_gib', 'runtime_min_free_gib', 'model', 'webui', 'image_model', 'video_model',
                   'image_arch', 'video_arch', 'comfyui_models', 'image_mode', 'video_mode',
                   'mtp_model', 'mac_port', 'mac_engine_port', 'backend', 'zig', 'models', 'resident', 'native_tier',
                   'mac', 'pair_token'}
    if args.command in ('start', 'rtx'):
        config_keys.update({'lora', 'serving'})
    overrides = {k: v for k, v in vars(args).items() if k in config_keys and v is not None}
    cfg = load(args.config, overrides)
    if args.command == 'rtx' and cfg.get('mac') and args.webui is None:
        cfg['webui'] = True
    cfg.setdefault('env_overrides', {}).update({k[4:]: v for k, v in vars(args).items()
                                              if k.startswith('env_') and v is not None})
    if args.command == 'decode':
        from .decode import run
        result = run(cfg, model=args.model, selected=args.backend, check=args.check)
        print(json.dumps(result, indent=2))
        return 0 if result['ready'] else 1
    if args.command == 'setup-backend':
        from .backend_setup import setup
        names=list(BACKENDS) if args.backend=='all' else [args.backend]
        results=[]
        for name in names:
            try:
                results.append(setup(cfg,name,check_only=args.check,fetch_only=args.fetch_only,build_only=args.build_only,
                                     cache_dir=args.cache_dir,offline=args.offline,wheelhouse=args.wheelhouse,jobs=args.jobs))
            except (OSError,ValueError,RuntimeError) as exc:results.append(dict(backend=name,built=False,ready=False,error=str(exc)))
        print(json.dumps(results,indent=2))
        if args.check:return 0 if all(r.get('ready') for r in results) else 1
        return 0 if all(r.get('downloaded') is not None or r.get('built') or r.get('ready') for r in results) else 1
    if args.command == 'decode-internal':
        from .decode import serve
        serve(cfg)
        return 0
    if args.command == 'setup-mlx':
        from .mlx_setup import setup
        result = setup(cfg, check=args.check, native_glm=not args.no_native_glm, wheelhouse=args.wheelhouse)
        print(json.dumps(result, indent=2))
        return 0 if result['ready'] else 1
    if args.command == 'install':
        from .install import install
        result = install(cfg, image=args.image, video=args.video, all_engines=args.all,
                         check=args.check, sdk_only=args.sdk)
        print(json.dumps(result, indent=2))
        return 0 if (result['sdk']['ready'] and all(row['ready'] for row in result['engines'])
                     and result.get('build', {}).get('ready', True)
                     and result.get('model_receipt', {}).get('ready', True)) else 1
    if args.command in ('mac', 'mac-setup'):
        from .mac import setup, budget
        from .process import OwnedProcess
        from .journal import atomic_json
        backend = cfg.get('backend') or 'ds4'
        build = setup(cfg, backend, check=args.check, zig=cfg.get('zig'))
        print(json.dumps(build), flush=True)
        if args.command == 'mac-setup' or args.check:
            return 0 if build['ready'] else 1
        if args.host_budget_gib is not None:
            cfg['mac_host_budget_gib'] = args.host_budget_gib
        launch = budget(cfg, 400 if backend == 'ds4' else 128)
        launch['mac_build'] = build
        launch['backend'] = backend
        effective = Path(cfg['runtime_root']) / 'mac-effective.local.json'
        atomic_json(effective, launch)
        proc = OwnedProcess(launch, [sys.executable, '-B', '-m', 'slstack', '--config', str(effective), 'mac-internal'],
                            'mac-receiver', timeout=86400)
        try:
            from .mac import wait_for_receiver
            wait_for_receiver(launch, proc)
            proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            if proc.proc.poll() is None:
                proc.stop()
        return 0
    if args.command == 'mac-internal':
        from .mac import serve
        serve(cfg, cfg['backend'], cfg['mac_build'])
        return 0
    if args.command == 'broker-internal':
        from .native_broker import serve
        serve(cfg)
        return 0
    if args.command == 'pair':
        from .pairing import Pair
        peer = Pair(cfg)
        try:
            peer.start()
            print('Pairing and native top-1 handshake passed', flush=True)
        finally:
            peer.close()
        return 0
    if args.command == 'guarded':
        # Build/install subprocesses do not use model state. In particular, do
        # not hash a large model outside the child guardian before a build.
        from .process import OwnedProcess
        command = args.argv[1:] if args.argv and args.argv[0] == '--' else args.argv
        print(json.dumps(OwnedProcess(cfg, command, 'guarded', cwd=args.cwd).wait(), indent=2))
        return 0
    from .lease import recover_runtime_leases
    for identity in recover_runtime_leases(cfg['runtime_root'], cfg.get('task', 'DS4STACK')):
        print('Released dead GPU guardian hold: ' + identity, file=sys.stderr)
    reports = Journal(cfg['runtime_root']).recover() if args.command not in ('check', 'start', 'rtx', 'serve-internal') else []
    for report in reports:
        print('Recovered stale journal: ' + json.dumps(report), file=sys.stderr)
    if args.command == 'check':
        from .check import check
        rows = check(cfg)
        print(json.dumps(rows, indent=2))
        return 0 if all(row['ok'] or not row['required'] for row in rows) else 1
    if args.command in ('start', 'rtx'):
        from .check import check
        rows = check(cfg, verify_journal=False)
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
            # Graceful first: the router closes the native engine, which drains DirectStorage.
            import urllib.request
            try:
                urllib.request.urlopen(urllib.request.Request(f'http://{cfg["host"]}:{cfg["port"]}/v1/stack/shutdown',
                                                              data=b'{}', headers={'Content-Type': 'application/json'}),
                                       timeout=10).read()
                from .shutdown import wait_for_exit,gpu_free_reader
                wait_for_exit(proc.proc, 120, diagnostic=proc.log,gpu_free=gpu_free_reader(cfg.get('gpu_uuid')))
            except Exception as exc:
                print('Graceful router stop failed: ' + repr(exc), file=sys.stderr)
            if proc.proc.poll() is None:
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


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError, MemoryError) as exc:
        print(f'slstack: {exc}', file=sys.stderr)
        sys.exit(1)
