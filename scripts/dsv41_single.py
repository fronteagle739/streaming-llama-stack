"""DSV4.1 single-engine launch configurations (262K and 1M) for `slstack start`.

scripts/start-dsv41-single-262k.ps1 and -1m.ps1 call this file. It writes one config file that
`slstack start --config` loads on top of config.example.json: your own local config (paths, ports,
budgets), then the gated single-engine settings below. A local key replaces the example's key
rather than merging, so env_overrides is merged here.

    python -B scripts/dsv41_single.py --preset 262k [--config my.local.json] [--model M.gguf] --write runtime/x.local.json
    python -B scripts/dsv41_single.py --preset 1m --config my.local.json --dry-run      (prints the resolved settings)

Both presets: serving single_engine (ULTRA exact-cone prefill + ds4-decode.dll decode in one
producer), DirectStorage read parts 8, 512 MiB VRAM reserve beside the L2 expert tier, the default
quality/speed setting, the fold sweep plan (49152-token primary sweeps, the remainder folded into the
last sweep up to 61,440 tokens, else its own sweep),
producer_host_reserve_gib 0.5, Engram early load auto with no Engram row cache.
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
}
DEFAULT_ENV = {
    'LLAMADECODE_ENGRAM_EARLY': 'auto',       # default
    'LLAMADECODE_ENGRAM_CACHE_MB': '0',       # no row cache: unset takes all admission headroom on this revision
    'LLAMADECODE_APPEND_DECODE_MAX': '0',     # E2c resident appends change logits; explicit toggle only
    'LLAMADECODE_RECENCY_LFU': '1',          # E8 exact placement policy, faster in both measured repetitions
    'LLAMADECODE_LFU_HALF_LIFE': '16',       # qualified cadence; one-token decay was effectively neutral
    'LLAMADECODE_APPEND_RESIDENT_SWEEP': '1',
    'LLAMADECODE_HOST_COPY_STAGING': 'auto',
    'DS4_CUDA_SINGLE_GPU_HC_GRAPHS': '1',
    'LLAMA_DSV41_INDEX_FP4': '1',
}
PRESETS = {
    '262k': dict(context=262144, ctx_cache_precision='f32', ctx_cache_growable=True,
                 preset_name='DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine, 262K)'),
    '1m': dict(context=999424, ctx_cache_precision='bf16', ctx_cache_growable=True,
               preset_name='DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine, 1M)'),
}
# Keys that would change the gated configuration; a local config may not set them for these presets.
REMOVED = ('producer_host_max_gib', 'producer_keep_alive', 'producer_prefill_sweep_tokens', 'producer_prefill_fold_cap_tokens')
# The gated fold sweep plan (a qualified reference measurement, 2026-10-25): primary sweep S* tokens, the
# remainder folded into the last sweep while it stays <= the fold cap, else its own sweep. The host max
# is only the ceiling for the largest sweep's bank (capacity <= the fold cap).
# OPUS22 round 6 (owner: single stream by default): 'auto' = one sweep up to the host-capped capacity
# (100K fits 7.75 GiB with the compact carry record), else near-equal sweeps under that cap.
FOLD = dict(producer_prefill_sweep_tokens='auto', producer_prefill_fold_cap_tokens=61440, producer_host_max_gib=7.75)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}


def overlay(preset, local=None, model=None, yoco_keep=None, vision=None, profile=None):
    """The config file for `slstack start --config`: the local keys, then the preset."""
    if preset not in PRESETS:
        raise ValueError('preset must be one of ' + ', '.join(PRESETS))
    result = {k: v for k, v in dict(local or {}).items() if k not in REMOVED}
    if model:
        result['model'] = model
    env = dict(result.get('env_overrides') or {})
    env.update(COMMON_ENV)
    for key,value in DEFAULT_ENV.items():env.setdefault(key,value)
    result.update(PRESETS[preset], serving='single_engine', producer_host_reserve_gib=.5, env_overrides=env, **FOLD)
    from slstack.multichunk import selection
    chunks=selection(result.get('producer_chunks_per_pass',1))
    if chunks!=1 and local and 'producer_host_max_gib' in local:
        result['producer_host_max_gib']=local['producer_host_max_gib']
    if profile is None and yoco_keep is None:
        profile='ultra-cone'
    if profile is not None:
        if profile not in ('accurate','fast','ultra','ultra-yoco','ultra-cone'):raise ValueError('Unknown single-engine quality profile')
        result['producer_profile']='ultra' if profile in ('ultra-yoco','ultra-cone') else profile
        expected=2541 if profile=='ultra-cone' else 140 if profile=='ultra-yoco' else 0
        if yoco_keep is not None and yoco_keep!=expected:raise ValueError('YOCO selection conflicts with the quality profile')
        yoco_keep=expected
        result['preset_name']='DSV4.1 single engine / '+profile+' / '+preset
    if yoco_keep is not None:
        if type(yoco_keep) is not int or yoco_keep not in (0,140,2541):raise ValueError('yoco_keep must be 0, 140 or 2541')
        result['yoco_keep']=yoco_keep
    if vision:result['vision_bridge_version']=1
    # Qualified Q4 exact-cone carry lifetime: use the unused growable-cache
    # reservation during initial prefill, then restore it before cache growth.
    # Keep N=1: the optimized two-sweep control is faster than grouped N=2.
    if (preset=='1m' and result.get('producer_profile','ultra')=='ultra' and
            result.get('yoco_keep')==2541 and not result.get('vision_bridge_version') and
            result.get('producer_fixed_direct_open',True) and
            env.get('LLAMADECODE_DECODE_COMPACT','1')!='0'):
        env.setdefault('LLAMADECODE_SINGLEPASS_VRAM','1')
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
    if single.get('producer_prefill_sweep_tokens'):   # the fold sweep plan's producer flags
        flags.update({'--prefill-sweep-tokens': str(plan['sweep_tokens']),
                      '--prefill-fold-cap-tokens': str(plan['fold_cap'])})
    from slstack.combo_producer import yoco_keep
    keep_yoco=yoco_keep(single)
    if keep_yoco:flags['--yoco-keep']=str(keep_yoco)
    env = apply({}, single, 'rtx-producer')
    keep = ('LLAMADECODE_', 'DS4_DS_READ_PARTS', 'LLAMA_DSV41_', 'DS4_CUDA_SINGLE_GPU_HC_GRAPHS')
    return dict(
        serving=merged.get('serving'), model=merged.get('model') or '(set "model" in your config or pass -Model)',
        router=f"http://{merged['host']}:{merged['port']}", engine=merged['ds4_url'],
        llama_producer=str(stack_config.path(merged['llama_producer'])),
        single_engine={k: single.get(k) for k in ('context', 'ctx_cache_precision', 'ctx_cache_growable',
                       'producer_profile', 'producer_host_budget_gib', 'producer_host_reserve_gib',
                       'host_budget_gib', 'combo_worker_host_budget_gib', 'cache_affinity_spread_file')},
        sweep_plan=dict(policy=plan.get('policy') or 'split', primary_sweep_tokens=plan.get('sweep_tokens'),
                        fold_cap_tokens=plan.get('fold_cap'), capacity_tokens=plan['capacity'],
                        max_host_bytes=plan['max_host_bytes'], base_host_bytes=plan['base_host_bytes']),
        producer_flags=flags,
        producer_env={k: v for k, v in sorted(env.items()) if k.startswith(keep)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', required=True, choices=sorted(PRESETS))
    parser.add_argument('--profile',choices=('accurate','fast','ultra','ultra-yoco','ultra-cone'))
    parser.add_argument('--chunks-per-pass', choices=('auto','1','2','3','4'), help='Chunks sharing each layer weight lease; explicit RAM budget required')
    parser.add_argument('--config', help='Your local config (paths, ports, budgets); optional')
    parser.add_argument('--yoco-keep', type=int, choices=(0,140,2541), default=None, help='Bounded replay: 0 full, 140 approximate, 2541 exact dependency cone (local only)')
    parser.add_argument('--vision',action='store_true',default=None,help='Enable native image prefill and auto-detect the matching vision GGUF (requires the vision-capable producer)')
    parser.add_argument('--model', help='DSV4.1 GGUF path (overrides "model" in --config)')
    parser.add_argument('--write', help='Write the start config here')
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved settings')
    args = parser.parse_args(argv)
    local=read(args.config)
    if args.chunks_per_pass is not None:
        local['producer_chunks_per_pass'] = 'auto' if args.chunks_per_pass == 'auto' else int(args.chunks_per_pass)
    cfg = overlay(args.preset, local, args.model, args.yoco_keep, args.vision,args.profile)
    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    if args.dry_run:
        print(json.dumps(dict(preset=args.preset, start_config=cfg, resolved=resolved(cfg)), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
