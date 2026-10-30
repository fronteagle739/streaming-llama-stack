"""Merge a launcher config (config/<mode>.json + optional config/<mode>.local.json) for one
machine side and write the JSON that a packer bakes into the launcher. Python 3.9+.

    python -B launcher/app/bake.py --mode single|dual --side rtx|mac --python <exe> --out <file>
"""
import argparse
import json
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1]


def merge(base, local):
    result = dict(base)
    for key, value in local.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def placeholders(value, prefix=''):
    if isinstance(value, dict):
        return [n for k, v in value.items() for n in placeholders(v, prefix + '.' + k if prefix else k)]
    if isinstance(value, list):
        return [n for i, v in enumerate(value) for n in placeholders(v, f'{prefix}[{i}]')]
    return [prefix] if isinstance(value, str) and value.startswith('SET_') else []


def bake(mode, side, python, stack_root=None):
    config = LAUNCHER / 'config'
    data = json.loads((config / (mode + '.json')).read_text(encoding='utf-8'))
    local = config / (mode + '.local.json')
    if local.is_file():
        data = merge(data, json.loads(local.read_text(encoding='utf-8-sig')))
    shared = {k: v for k, v in data.items() if k not in ('rtx', 'mac')}
    cfg = merge(data[side], {k: v for k, v in shared.items() if k != 'pair_token'})
    if 'pair_token' in shared:
        cfg['stack']['pair_token'] = shared['pair_token']
    cfg.update(mode=mode, side=side, python=python,
               stack_root=str(Path(stack_root or cfg.get('stack_root') or LAUNCHER.parent).resolve()))
    root = cfg['runtime_root']
    cfg['stack'].setdefault('runtime_root', root)
    if 'ui' in cfg:
        cfg['ui'].setdefault('runtime_root', root)
        if mode == 'dual':
            cfg['ui']['general'].setdefault('api_key', cfg['stack']['pair_token'])
    missing = placeholders(cfg)
    if missing:
        raise SystemExit('Set these in launcher/config/%s.local.json: %s' % (mode, ', '.join(missing)))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('single', 'dual'), required=True)
    parser.add_argument('--side', choices=('rtx', 'mac'), required=True)
    parser.add_argument('--python', required=True, help='Python 3.12 that runs the stack on this machine')
    parser.add_argument('--stack-root', help='Stack checkout the launcher runs (default: this repo)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = bake(args.mode, args.side, args.python, args.stack_root)
    Path(args.out).write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print('Baked %s (%s/%s): stack %s, python %s' % (cfg['title'], args.mode, args.side, cfg['stack_root'], args.python))


if __name__ == '__main__':
    main()
