"""Fail-closed admission for the native Mac decoder's actual sampler support.

Keep the complete caller request. Core knobs are parsed by ds4-server; richer
llama-server knobs and cache-affinity bias need native support before admission.
"""
from .generation import SAMPLERS, number, validate


def decode_request(body):
    result = dict(body)
    profile = result.get('slstack_generation')
    if profile is not None:
        profile = validate(profile)
        if profile['per_phase']:
            raise ValueError('Mac ds4 does not yet apply phase-specific generation profiles; paired release gate is blocked')
        if profile['value'] != 0:
            raise ValueError('Mac ds4 does not implement the requested quality/speed bias; no setting was silently dropped')
        for key, value in profile['sampler'].items():
            result.setdefault(key, value)
    for key, (lo, hi) in SAMPLERS.items():
        if key in result:
            result[key] = number(result[key], lo, hi, key == 'top_k')
    if 'seed' in result:
        seed = result['seed']
        if type(seed) is not int or not 0 <= seed < 2**53:
            raise ValueError('seed must be an integer in [0,2^53)')
    # ds4-server currently ignores these JSON fields. A request must not appear
    # successful while silently running another sampling distribution.
    neutral = dict(presence_penalty=0, frequency_penalty=0, repeat_penalty=1,
                   repetition_penalty=1, repeat_last_n=64, penalty_last_n=64,
                   typical_p=1, tfs_z=1, top_n_sigma=-1, xtc_probability=0,
                   dynatemp_range=0, mirostat=0, dry_multiplier=0,
                   logit_bias={}, samplers=[], quality_speed=0)
    # Auxiliary knobs have no effect while their controlling sampler is neutral.
    for key, value in result.items():
        if key in neutral and value != neutral[key]:
            raise ValueError('Mac ds4 does not implement requested sampler '+key+'; no greedy fallback is allowed')
    return result
