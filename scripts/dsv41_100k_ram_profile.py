"""Prepare the measured DSV4.1 100K/200K RAM profile for normal runtime staging.

This writes a new configuration; it never edits a preset or deploys a runtime.
All non-budget fields, sampler settings, writers and native binary pins survive.
Normal strict admission, paging observation and physical-memory floors still apply.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slstack.combo_producer import prefill_arguments


PERFORMANCE_DEFAULTS = dict(
    host_budget_gib=10.25, combo_worker_host_budget_gib=10.0,
    producer_host_budget_gib=6.25, producer_host_max_gib=7.75,
    min_available_gib=4, runtime_min_free_gib=4,
    producer_desktop_reserve_gib=4, producer_host_reserve_gib=4,
    producer_prefill_sweep_tokens='auto', producer_prefill_fold_cap_tokens=61440,
    producer_graph_floor_gib=3.0, producer_engram_scan_rings='auto', owned_paging_limit_bytes=0,
    producer_warmup_one_token=False, producer_warmup_host_reserve_mib=512,
    producer_warmup_defer_prompt_data=True,
    producer_fixed_direct_open=True)

ENGRAM_DEFAULTS = dict(LLAMADECODE_ENGRAM_WHOLE_PROMPT='1', LLAMADECODE_ENGRAM_SCAN='auto',
    LLAMADECODE_ENGRAM_SCAN_KIB='1024', LLAMADECODE_ENGRAM_SCAN_GAP_KIB='64',
    LLAMADECODE_ENGRAM_SCAN_DEPTH='64')
PERFORMANCE_ENV_DEFAULTS = {**ENGRAM_DEFAULTS, 'OPENBLAS_NUM_THREADS': '1'}


def performance_defaults(configuration):
    """Full measured RAM/sweep defaults; retain every explicit local setting.

    Default admission needs 14.25 GiB available physical RAM (10.25 + 4 reserve).
    The 200K whole-prompt Engram load needs 8.727 GiB plus controller room;
    the former 8.5 GiB aggregate cap refused it despite admitting both sweeps.
    Smaller explicit budgets remain supported and can require more sweeps.
    Model arithmetic, pack selection, sampling and optional features are untouched.
    """
    result = deepcopy(configuration)
    if 'host_budget_gib' in configuration and 'combo_worker_host_budget_gib' not in configuration:
        result['combo_worker_host_budget_gib'] = configuration['host_budget_gib'] - .25
    for key, value in PERFORMANCE_DEFAULTS.items():
        result.setdefault(key, value)
    env=result.setdefault('env_overrides',{})
    for key,value in PERFORMANCE_ENV_DEFAULTS.items():env.setdefault(key,value)
    return result


def enforce_release_budgets(cfg, experiment=False):
    """Do not let an unrecorded old local file recreate the deployed cap regression."""
    if cfg.get('performance_manifest') is not None:
        from slstack.performance_contract import validate
        validate(cfg, 'dsv41')
    if experiment or cfg.get('performance_experiment'):
        return
    minima={'host_budget_gib':10.25,'combo_worker_host_budget_gib':10.0,'producer_host_max_gib':7.75}
    lowered=[k for k,v in minima.items() if cfg.get(k) not in (None,'auto') and float(cfg[k])<v]
    if lowered:
        raise ValueError('Unrecorded lower DSV performance caps: '+str(lowered)+
                         '; use --performance-experiment for an intentional alternate profile')


def load_performance_config(path, *, experiment=False):
    """Apply DSV defaults before generic stack defaults, preserving the file."""
    from slstack.config import load
    local = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    enforce_release_budgets(local, experiment)
    inherited = {key: value for key, value in PERFORMANCE_DEFAULTS.items() if key not in local}
    result=load(path, overrides=inherited)
    env=result.setdefault('env_overrides',{})
    for key,value in PERFORMANCE_ENV_DEFAULTS.items():env.setdefault(key,value)
    return result


def add_performance_flags(parser):
    """Expose the same engine table and RAM/sweep controls in every DSV script."""
    from slstack.envvars import add_flags
    add_flags(parser)
    parser.add_argument('--performance-experiment',action='store_true',
                        help='Explicitly record an alternate RAM profile; never label it as the release recipe')
    for key in ('host_budget_gib', 'combo_worker_host_budget_gib', 'producer_host_budget_gib',
                'producer_graph_floor_gib', 'min_available_gib', 'runtime_min_free_gib',
                'producer_desktop_reserve_gib', 'producer_host_reserve_gib'):
        parser.add_argument('--'+key.replace('_','-'), type=float)
    parser.add_argument('--producer-host-max-gib', help='Finite positive GiB or auto')
    parser.add_argument('--producer-prefill-sweep-tokens', help='Positive aligned count or auto')
    parser.add_argument('--producer-prefill-fold-cap-tokens', type=int)
    parser.add_argument('--owned-paging-limit-bytes',type=int,
                        help='Owned paging observation limit in bytes; the full profile defaults to0')
    parser.add_argument('--producer-warmup-one-token', type=int, choices=(0,1),
                        help='Native short warm-up before the different timed prompt in the same process')
    parser.add_argument('--producer-warmup-host-reserve-mib',type=int,
                        help='Explicit transient host bill retained while the warm request retires; required with warm-up')
    parser.add_argument('--producer-fixed-direct-open',type=int,choices=(0,1))
    parser.add_argument('--producer-warmup-defer-prompt-data',type=int,choices=(0,1),
                        help='Start the measured prompt Engram reads after warm-up; early prefetch stays active inside its own request')
    parser.add_argument('--producer-engram-scan-rings', choices=('auto',*[str(n) for n in range(1,9)]),
                        help='auto: four queues for a single sweep, one for multiple sweeps; explicit engine flag wins over auto')


def apply_performance_flags(configuration, args):
    result=deepcopy(configuration)
    values=vars(args)
    if values.get('performance_experiment'):
        result['performance_experiment']=True
    for key in PERFORMANCE_DEFAULTS:
        value=values.get(key)
        if value is not None:
            if key=='producer_host_max_gib' and value!='auto':value=float(value)
            if key=='producer_prefill_sweep_tokens' and value!='auto':value=int(value)
            if key=='producer_engram_scan_rings' and value!='auto':value=int(value)
            result[key]=value
    if values.get('host_budget_gib') is not None and values.get('combo_worker_host_budget_gib') is None:
        result['combo_worker_host_budget_gib']=result['host_budget_gib']-.25
    selected={key[4:]:value for key,value in values.items() if key.startswith('env_') and value is not None}
    if selected:
        result.setdefault('env_overrides',{}).update(selected)
    margin=result.get('env_overrides',{}).get('SLSTACK_DECODE_VRAM_MARGIN_MIB')
    if margin is not None:
        if not str(margin).isascii() or not str(margin).isdigit() or not 0<=int(margin)<=4096:
            raise ValueError('SLSTACK_DECODE_VRAM_MARGIN_MIB must be an integer in 0..4096')
        result['env_overrides']['LLAMADECODE_VRAM_MARGIN_MIB']=str(margin)
    if values.get('producer_warmup_one_token') is not None:
        result['producer_warmup_one_token'] = bool(values['producer_warmup_one_token'])
        result['producer_fixed_direct_open'] = not result['producer_warmup_one_token']
    if values.get('producer_warmup_host_reserve_mib') is not None:
        result['producer_warmup_host_reserve_mib']=values['producer_warmup_host_reserve_mib']
    if values.get('producer_fixed_direct_open') is not None:
        result['producer_fixed_direct_open']=bool(values['producer_fixed_direct_open'])
    if values.get('producer_warmup_defer_prompt_data') is not None:
        result['producer_warmup_defer_prompt_data']=bool(values['producer_warmup_defer_prompt_data'])
    return result


def profile(configuration, available_bytes, prompt_tokens=100000):
    if configuration.get('producer_profile') != 'ultra':
        raise ValueError('This measured profile requires the DSV4.1 ULTRA producer')
    result = deepcopy(configuration)
    # A 6.25 GiB maximum forces two sweeps for 100K and an extra 157.46 GB
    # of full-layer reads. 7.75 GiB admits the existing exact 100K planner.
    for key, minimum in {'host_budget_gib': 10.25,
                         'combo_worker_host_budget_gib': 10.0,
                         'producer_host_max_gib': 7.75}.items():
        if key == 'producer_host_max_gib' and result.get(key) == 'auto':
            continue
        result[key] = max(float(result.get(key, 0)), minimum)
    required = (result['host_budget_gib'] + result['min_available_gib']) * 2**30
    if available_bytes < required:
        raise MemoryError(f'100K/200K profile needs {required / 2**30:.2f} GiB available '
                          f'including the unchanged physical reserve; '
                          f'available {available_bytes / 2**30:.2f} GiB')
    _, plan = prefill_arguments(result, prompt_tokens, available_bytes)
    if prompt_tokens==100000 and (not plan['single_sweep'] or plan['sweeps'] != [100000]):
        raise MemoryError(f'Existing guarded planner cannot admit one 100K sweep: {plan}')
    if prompt_tokens>100000 and plan.get('whole_prompt_engram_refused'):
        raise MemoryError(f'Full-performance Engram admission refused: {plan}')
    return result, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prompt-tokens', type=int, default=100000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output; existing configuration is retained')
    import psutil
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    output, plan = profile(cfg, psutil.virtual_memory().available, args.prompt_tokens)
    with args.output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), plan=plan,
                          capability_reductions=[]), indent=2))


if __name__ == '__main__':
    main()
