"""DSV4.1 single-engine launch configurations (262K and 1M) for `slstack start`.

scripts/start-dsv41-single-262k.ps1 and -1m.ps1 call this file. It writes one config file that
`slstack start --config` loads on top of config.example.json: your own local config (paths, ports,
budgets), then the gated single-engine settings below. A local key replaces the example's key
rather than merging, so env_overrides is merged here.

    python -B scripts/dsv41_single.py --preset 262k [--config my.local.json] [--model M.gguf] --write runtime/x.local.json
    python -B scripts/dsv41_single.py --preset 1m --config my.local.json --dry-run      (prints the resolved settings)

Both presets: serving single_engine (streaming-llama ULTRA prefill + ds4-decode.dll decode in one
producer), DirectStorage read parts 8, 512 MiB VRAM reserve beside the L2 expert tier, the default
quality/speed setting, single-sweep prefill off (no producer_host_max_gib: 49152-token sweeps),
producer_host_reserve_gib 1, Engram early load auto with no Engram row cache.
  262k: context 262144, f32 context cache, growable; the L2 tier takes its full NVML offer.
  1m:   context 999424, bf16 context cache, growable (the largest growable bf16 context that fits
        the producer's admitted cache: 2 MiB rounding on 8 ranges refuses 1,000,001).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMON_ENV = {
    'DS4_DS_READ_PARTS': '8',                 # DirectStorage split reads (the engines' default)
    'LLAMADECODE_VRAM_RESERVE_MIB': '512',    # physical VRAM kept free beside the L2 tier (default)
    'LLAMADECODE_ENGRAM_EARLY': 'auto',       # default
    'LLAMADECODE_ENGRAM_CACHE_MB': '0',       # no row cache: unset takes all admission headroom on this revision
}
PRESETS = {
    '262k': dict(context=262144, ctx_cache_precision='f32', ctx_cache_growable=True,
                 preset_name='DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine, 262K)'),
    '1m': dict(context=999424, ctx_cache_precision='bf16', ctx_cache_growable=True,
               preset_name='DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine, 1M)'),
}
# Keys that would change the gated configuration; a local config may not set them for these presets.
REMOVED = ('producer_host_max_gib', 'producer_keep_alive')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}


def overlay(preset, local=None, model=None):
    """The config file for `slstack start --config`: the local keys, then the preset."""
    if preset not in PRESETS:
        raise ValueError('preset must be one of ' + ', '.join(PRESETS))
    result = {k: v for k, v in dict(local or {}).items() if k not in REMOVED}
    if model:
        result['model'] = model
    env = dict(result.get('env_overrides') or {})
    env.update(COMMON_ENV)
    result.update(PRESETS[preset], serving='single_engine', producer_host_reserve_gib=1, env_overrides=env)
    return result


def resolved(cfg):
    """What the single engine will run with, without model files or a built producer."""
    from slstack import config as stack_config
    from slstack.single_runtime import serving_config, context
    from slstack.combo_producer import ctx_cache, prefill_plan
    from slstack.envvars import apply
    merged = json.loads((ROOT / 'config.example.json').read_text(encoding='utf-8'))
    merged.update(cfg)
    root = Path(merged['runtime_root']).expanduser()
    merged['runtime_root'] = str(root if root.is_absolute() else (ROOT / root).resolve())
    base = {**merged, 'producer_profile': merged.get('producer_profile', 'ultra'),
            'producer_host_budget_gib': merged.get('producer_host_budget_gib', 6.25), 'context': context(merged)}
    single = serving_config(base, runtime_root=Path(merged['runtime_root']) / 'single-engine',
                            studio_root=stack_config.path(merged['studio_root']),
                            llama_producer=stack_config.path(merged['llama_producer']), library_directories=[],
                            revision='dry-run', host_budget_gib=merged['host_budget_gib'] - .25,
                            spread_file=stack_config.path(merged.get('cache_affinity_spread_file') or 'presets/dsv41-affinity-spreads.txt'))
    precision, growable = ctx_cache(single)
    plan = prefill_plan(single, 2048)
    flags = {'--context': str(single['context']), '--compressed-cache-bf16': '1' if precision == 'bf16' else '0'}
    if precision != 'f32' or growable:
        flags.update({'--ctx-cache-precision': precision, '--ctx-cache-growable': str(int(growable))})
    env = apply({}, single, 'rtx-producer')
    keep = ('LLAMADECODE_', 'DS4_DS_READ_PARTS', 'LLAMA_DSV41_')
    return dict(
        serving=merged.get('serving'), model=merged.get('model') or '(set "model" in your config or pass -Model)',
        router=f"http://{merged['host']}:{merged['port']}", engine=merged['ds4_url'],
        llama_producer=str(stack_config.path(merged['llama_producer'])),
        single_engine={k: single.get(k) for k in ('context', 'ctx_cache_precision', 'ctx_cache_growable',
                       'producer_profile', 'producer_host_budget_gib', 'producer_host_reserve_gib',
                       'host_budget_gib', 'combo_worker_host_budget_gib', 'cache_affinity_spread_file')},
        single_sweep=dict(enabled='producer_host_max_gib' in single, capacity_tokens=plan['capacity'],
                          max_host_bytes=plan['max_host_bytes'], base_host_bytes=plan['base_host_bytes']),
        producer_flags=flags,
        producer_env={k: v for k, v in sorted(env.items()) if k.startswith(keep)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', required=True, choices=sorted(PRESETS))
    parser.add_argument('--config', help='Your local config (paths, ports, budgets); optional')
    parser.add_argument('--model', help='DSV4.1 GGUF path (overrides "model" in --config)')
    parser.add_argument('--write', help='Write the start config here')
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved settings')
    args = parser.parse_args(argv)
    cfg = overlay(args.preset, read(args.config), args.model)
    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    if args.dry_run:
        print(json.dumps(dict(preset=args.preset, start_config=cfg, resolved=resolved(cfg)), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
