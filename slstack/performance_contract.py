"""Per-model launch contracts and immutable evidence for published measurements."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

DSV_REQUIRED = ('host_budget_gib', 'combo_worker_host_budget_gib',
    'producer_host_budget_gib', 'producer_host_max_gib', 'producer_engram_scan_rings')
QFN_REQUIRED = ('host_budget_gib', 'qfn_host_budget_gib', 'qfn_engram_scan_rings', 'qfn_tile')
COMMON = ('min_available_gib', 'runtime_min_free_gib', 'job_commit_gib',
          'owned_paging_limit_bytes', 'context', 'yoco_keep', 'native_techniques',
          'native_fp4_prefill_only', 'vision_bridge_version',
          'ctx_cache_precision', 'ctx_cache_growable', 'model_settings', 'performance_experiment')
ENV_PREFIXES = ('LLAMA', 'GGML', 'DS4', 'QFN', 'OPENBLAS_', 'OMP_', 'CUDA_MODULE_', 'SLSTACK_DECODE_')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def engine_environment(env):
    # Preserve numeric *_TOKENS controls while keeping actual credentials out
    # of committed measurement evidence.
    return {k: ('sha256:'+digest(str(v)) if k.endswith(('_TOKEN','_KEY','_SECRET','_PASSWORD')) else str(v))
            for k,v in env.items() if k.startswith(ENV_PREFIXES)}


def settings(cfg, model):
    if model not in ('dsv41', 'qfn'):
        raise ValueError('A model-specific performance contract is required')
    required = DSV_REQUIRED if model == 'dsv41' else QFN_REQUIRED
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(f'{model} missing explicit performance settings: {missing}')
    for key in required:
        value = cfg[key]
        if value == 'auto' and key in ('producer_host_max_gib', 'producer_engram_scan_rings'):
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{model} invalid {key}: {value!r}')
    rings = cfg['producer_engram_scan_rings' if model == 'dsv41' else 'qfn_engram_scan_rings']
    if model=='qfn' and type(cfg['qfn_tile']) is not int:
        raise ValueError('QFN tile must be an explicit integer; fractional tiles cannot be truncated')
    if rings != 'auto' and (type(rings) is not int or not 1 <= rings <= 8):
        raise ValueError('Engram scan rings must be an integer in 1..8 (DSV also supports auto)')
    operational = {'producer_sha256', 'producer_source_revision', 'producer_library_directories',
                   'qfn_engine', 'qfn_vocab_gguf', 'qfn_task', 'qfn_preload'}
    keys = set(required) | set(COMMON) | {k for k in cfg if k.startswith(('producer_', 'qfn_')) and k not in operational}
    values = {key: deepcopy(cfg[key]) for key in sorted(keys) if key in cfg}
    # The request worker reserves the controller's 0.25 GiB before launch.
    # Record the aggregate and worker separately in either process.
    if model == 'dsv41':
        if cfg['host_budget_gib'] not in (cfg['combo_worker_host_budget_gib'], cfg['combo_worker_host_budget_gib'] + .25):
            raise ValueError('DSV aggregate/worker budget reservation differs')
        values['host_budget_gib'] = cfg['combo_worker_host_budget_gib'] + .25
        if not cfg.get('performance_experiment'):
            minima={'host_budget_gib':10.25,'combo_worker_host_budget_gib':10,'producer_host_max_gib':7.75}
            lowered=[k for k,v in minima.items() if values[k]!='auto' and values[k]<v]
            if lowered:
                raise ValueError('DSV performance settings drift below release caps: '+str(lowered)+
                                 '; use --performance-experiment to record an intentional alternate profile')
    values['env_overrides'] = engine_environment(cfg.get('env_overrides', {}))
    explicit = values['env_overrides'].get('LLAMADECODE_ENGRAM_SCAN_RINGS')
    if explicit is not None and rings != 'auto' and explicit != str(rings):
        raise ValueError('Per-model queue flag conflicts with explicit native queue flag')
    return values


def seal(cfg, model):
    """Record an intentional script selection; an existing contract cannot be silently resealed."""
    result = deepcopy(cfg)
    if result.get('performance_manifest') is not None:
        validate(result, model)
        return result
    values = settings(result, model)
    result['performance_manifest'] = dict(schema=1, model=model, settings=values,
                                          sha256=digest(values))
    return result


def validate(cfg, model):
    actual = settings(cfg, model)
    manifest = cfg.get('performance_manifest')
    if not isinstance(manifest, dict) or manifest.get('schema') != 1 or manifest.get('model') != model:
        raise ValueError(f'{model}: missing recorded performance manifest; generate one with the model launch script')
    if digest(manifest.get('settings')) != manifest.get('sha256'):
        raise ValueError('Recorded performance manifest digest differs')
    if actual != manifest['settings']:
        changed = sorted(k for k in actual.keys() | manifest['settings'].keys()
                         if actual.get(k) != manifest['settings'].get(k))
        raise ValueError(f'{model} performance settings drift: {changed}')
    return manifest


def record(path, cfg, model, *, argv, environment, fixture_sha256, plan=None):
    manifest = validate(cfg, model)
    row = dict(schema=1, model=model, settings_manifest=manifest, argv=list(argv),
        environment=engine_environment(environment),
        fixture_sha256=fixture_sha256, producer_sha256=cfg.get('producer_sha256'),
        model_sha256=cfg.get('model_sha256'), plan=plan)
    row['sha256'] = digest(row)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
    return row


def regression(baseline, current):
    """Only compare matched second-prompt runs; tolerance comes from repeated controls."""
    identity = ('fixture_sha256', 'prompt_tokens', 'settings_sha256', 'protocol', 'pack_sha256','clock','prompt_data_policy')
    if any(baseline.get(k) != current.get(k) or baseline.get(k) is None for k in identity):
        raise ValueError('Regression gate requires matched fixture, length, settings, pack and protocol')
    if baseline['protocol'] != 'warm-short-then-different-prompt-same-process':
        raise ValueError('Regression gate requires the warm second-prompt protocol')
    if baseline['clock']!='native-second-prompt-open-and-run' or baseline['prompt_data_policy']!='after-warm-up':
        raise ValueError('Regression gate requires full second-prompt accounting and deferred prompt data')
    controls = baseline['control_tok_s']
    if len(controls) < 2 or any(not math.isfinite(x) or x <= 0 for x in controls):
        raise ValueError('At least two positive repeated controls are required')
    threshold = min(controls)
    observed = current['prefill_tok_s']
    if not math.isfinite(observed) or observed < threshold:
        raise RuntimeError(f'Prefill regression: {observed:.2f} tok/s below repeated-control floor {threshold:.2f}')
    return dict(status='PASS', observed_tok_s=observed, control_floor_tok_s=threshold,
                spread_tok_s=max(controls)-min(controls))


def baseline_from_controls(controls):
    from .combo_release import certified
    if len(controls)<2:raise ValueError('At least two completed warm controls are required')
    identity=('fixture_sha256','prompt_tokens','settings_sha256','pack_sha256','protocol','clock','prompt_data_policy')
    first=controls[0]
    for row in controls:
        if row.get('status')!='PASS' or not certified(row.get('cleanup',{})):
            raise ValueError('A failed or uncleared control cannot qualify a baseline')
        cleanup=row['cleanup']
        if (cleanup.get('failure') or cleanup.get('exit_code',0)!=0 or
            cleanup.get('peak_owned_paged_bytes',0) or cleanup.get('owned_paged_bytes',0) or cleanup.get('budget_breaches',0)):
            raise ValueError('A control with a guard failure cannot qualify a baseline')
        warm=row.get('warm_measurement',{})
        if not warm.get('enabled') or not warm.get('same_process_and_CUDA_backend') or not warm.get('next_prompt_data_deferred'):
            raise ValueError('Native same-process warm witness is required')
        if any(row.get(k)!=first.get(k) or row.get(k) is None for k in identity):
            raise ValueError('Warm controls differ in fixture, length, settings, pack or protocol')
        if row['settings_receipt']['producer_sha256']!=first['settings_receipt']['producer_sha256']:
            raise ValueError('Repeated baseline controls must use the same producer binary')
    result={k:first[k] for k in identity}
    result.update(control_tok_s=[c['prefill_tok_s'] for c in controls],
                  producer_sha256=first['settings_receipt']['producer_sha256'])
    regression(result,{**result,'prefill_tok_s':min(result['control_tok_s'])})
    return result
