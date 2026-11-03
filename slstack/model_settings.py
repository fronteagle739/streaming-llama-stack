"""Model-specific named router settings; explicit numeric requests remain available."""
from copy import deepcopy
import json
import math
from pathlib import Path
from .config import ROOT

MODES = ('max_quality', 'balanced', 'max_speed')


def load(value):
    if isinstance(value, dict):
        row = deepcopy(value)
    else:
        path = Path(value)
        row = json.loads((path if path.is_absolute() else ROOT / path).read_text(encoding='utf-8-sig'))
    if row.get('version') != 1 or not row.get('model_key'):
        raise ValueError('Model settings require version 1 and model_key')
    bias = row['bias']
    for key in ('min', 'max', *MODES):
        v = bias[key]
        if type(v) not in (int, float) or not math.isfinite(v):
            raise ValueError('Model bias settings must be finite numbers: ' + key)
    if not 0 <= bias['min'] <= bias['max'] or bias['max_quality'] != 0:
        raise ValueError('Invalid bias range or nonzero Max quality')
    if not all(bias['min'] <= bias[k] <= bias['max'] for k in MODES):
        raise ValueError('Named bias falls outside the model range')
    return row


def named_bias(row, mode):
    if mode not in MODES:
        raise ValueError('Model mode must be ' + ', '.join(MODES))
    return float(row['bias'][mode])


def apply(cfg, family):
    """Resolve one model profile without overriding explicit local engine options."""
    if not cfg.get('model_settings'):
        return cfg
    row = load(cfg['model_settings'])
    if not row['model_key'].startswith(family + '-'):
        raise ValueError('Model settings belong to a different engine family')
    result = deepcopy(cfg)
    result['model_settings'] = row
    mode = cfg.get('model_mode')
    if mode is not None:
        bias = named_bias(row, mode)
        if family == 'qfn':
            result['qfn_quality_speed'] = mode
        else:
            result['generation_initial_bias'] = bias
    env = dict(result.get('env_overrides') or {})
    engine, cache = row['engine'], row['cache']
    allowed_engine = {'tile','read_parts','prefill','decode'} | ({'mtp','indexer','attention'} if family == 'qfn' else {'cone_keep'})
    allowed_cache = {'half_life','seed_lfu_max'} if family == 'qfn' else {'half_life','restore_hot_experts'}
    if set(engine) - allowed_engine or set(cache) - allowed_cache:
        raise ValueError('Unsupported model engine/cache settings: ' + ', '.join(sorted((set(engine)-allowed_engine) | (set(cache)-allowed_cache))))
    spreads = row['router_spreads'].get('file', row['router_spreads'].get('path'))
    if spreads and not row['router_spreads'].get('reference_only', False):
        path = Path(spreads)
        path = path if path.is_absolute() else ROOT / path
        if family == 'qfn':
            env.setdefault('QFN_AFFINITY_SPREADS', str(path))
        else:
            if not result.get('cache_affinity_spread_file'):
                result['cache_affinity_spread_file'] = str(path)
    env.setdefault('DS4_DS_READ_PARTS', str(engine['read_parts']))
    if family == 'qfn':
        for source, target in (('tile', 'qfn_tile'), ('prefill', 'qfn_nvfp4_mode'),
                               ('decode', 'qfn_nvfp4_decode_mode'), ('indexer', 'qfn_qsa_indexer'), ('attention','qfn_qsa_attention')):
            if source in engine:
                result.setdefault(target, engine[source])
        env.setdefault('QFN_MTP', str(engine['mtp']))
        env.setdefault('QFN_LFU_HALF_LIFE', str(cache['half_life']))
        env.setdefault('QFN_SEED_LFU_MAX', str(cache['seed_lfu_max']))
    else:
        if engine['decode'] != 'ds4-decode':
            raise ValueError('DSV named launchers require the existing ds4-decode path')
        prefill = engine['prefill']
        if prefill not in ('accurate','fast','ultra','ultra-cone','ultra-yoco'):
            raise ValueError('Unknown DSV prefill profile in model settings')
        result.setdefault('producer_profile','ultra' if prefill in ('ultra-cone','ultra-yoco') else prefill)
        result.setdefault('yoco_keep',engine['cone_keep'])
        result.setdefault('producer_compute_tile',engine['tile'])
        result.setdefault('expert_cache_restore',cache['restore_hot_experts'])
        env.setdefault('LLAMADECODE_LFU_HALF_LIFE', str(cache['half_life']))
    result['env_overrides'] = env
    return result
