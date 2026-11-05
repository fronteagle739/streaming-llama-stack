"""DSV4.1 single-engine launch configurations (262K and 1M) for `slstack start`.

scripts/start-dsv41-single-262k.ps1 and -1m.ps1 call this file. It writes one config file that
`slstack start --config` loads on top of config.example.json: your own local config (paths, ports,
budgets), the selected profile and the full-performance defaults below. A local key replaces the example's key
rather than merging, so env_overrides is merged here.

    python -B scripts/dsv41_single.py --preset 262k [--config my.local.json] [--model M.gguf] --write runtime/x.local.json
    python -B scripts/dsv41_single.py --preset 1m --config my.local.json --dry-run      (prints the resolved settings)

Both presets: serving single_engine (ULTRA exact-cone prefill + ds4-decode.dll decode in one
producer), DirectStorage read parts 8, 512 MiB VRAM reserve beside the L2 expert tier, the default
quality/speed setting, automatic RAM-admitted sweeps (one sweep for 100K with
the default 10.25 GiB host cap / 7.75 GiB producer maximum), 4 GiB physical reserve,
and whole-prompt Engram admission at 200K. Needs 14.25 GiB available RAM;
larger explicit budgets retain their own admission requirement.
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
    'LLAMADECODE_VRAM_RESERVE_MIB': '128',    # physical VRAM kept free beside the L2 tier (default)
    'LLAMADECODE_VRAM_MARGIN_MIB': '64',      # L2 growth margin beside the reserve (default; 128+0 breached)
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
    # Graph floor 3.0 GiB: measured planned graph peak 3.244 GiB at 200K;
    # the hidden bank then holds every 102,400-row sweep at 1M context (no spill).
    '1m': dict(context=999424, ctx_cache_precision='bf16', ctx_cache_growable=True, producer_graph_floor_gib=3.0,
               preset_name='DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine, 1M)'),
}
# Automatic sweep defaults live in dsv41_100k_ram_profile.PERFORMANCE_DEFAULTS.
# Explicit local fold/sweep/cap settings retain their normal planner semantics.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}


def overlay(preset, local=None, model=None, yoco_keep=None, vision=None, profile=None):
    """Local settings and full RAM defaults, followed by the selected profile."""
    if preset not in PRESETS:
        raise ValueError('preset must be one of ' + ', '.join(PRESETS))
    from scripts.dsv41_100k_ram_profile import performance_defaults
    result = performance_defaults(local or {})
    if model:
        result['model'] = model
    env = dict(result.get('env_overrides') or {})
    for key,value in {**COMMON_ENV, **DEFAULT_ENV}.items():env.setdefault(key,value)
    result.update(PRESETS[preset], serving='single_engine', env_overrides=env)
    from slstack.multichunk import selection
    selection(result.get('producer_chunks_per_pass',1))
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
    from slstack.performance_contract import seal
    from scripts.dsv41_100k_ram_profile import enforce_release_budgets
    generic = json.loads((ROOT/'config.example.json').read_text(encoding='utf-8'))
    result = {**generic, **result}
    enforce_release_budgets(result)
    return seal(result, 'dsv41')


def resolved(cfg):
    """What the single engine will run with, without model files or a built producer."""
    from slstack import config as stack_config
    from slstack.single_runtime import serving_config, context
    from slstack.combo_producer import ctx_cache, prefill_plan, engram_scan_rings
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
    rings=engram_scan_rings(single,plan)
    if rings is not None:env['LLAMADECODE_ENGRAM_SCAN_RINGS']=rings
    keep = ('LLAMADECODE_', 'DS4_DS_READ_PARTS', 'LLAMA_DSV41_', 'DS4_CUDA_SINGLE_GPU_HC_GRAPHS',
            'OPENBLAS_NUM_THREADS', 'GGML_', 'CUDA_MODULE_LOADING')
    return dict(
        serving=merged.get('serving'), model=merged.get('model') or '(set "model" in your config or pass -Model)',
        router=f"http://{merged['host']}:{merged['port']}", engine=merged['ds4_url'],
        llama_producer=str(stack_config.path(merged['llama_producer'])),
        single_engine={k: single.get(k) for k in ('context', 'ctx_cache_precision', 'ctx_cache_growable',
                       'producer_profile', 'producer_host_budget_gib', 'producer_host_reserve_gib',
                       'host_budget_gib', 'combo_worker_host_budget_gib', 'cache_affinity_spread_file',
                       'producer_engram_scan_rings')},
        sweep_plan=dict(policy=plan.get('policy') or 'split', primary_sweep_tokens=plan.get('sweep_tokens'),
                        fold_cap_tokens=plan.get('fold_cap'), capacity_tokens=plan['capacity'],
                        max_host_bytes=plan['max_host_bytes'], base_host_bytes=plan['base_host_bytes']),
        producer_flags=flags,
        producer_env={k: v for k, v in sorted(env.items()) if k.startswith(keep)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from scripts.dsv41_100k_ram_profile import add_performance_flags, apply_performance_flags
    add_performance_flags(parser)
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
    local=apply_performance_flags(read(args.config), args)
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
