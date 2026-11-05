"""Normalize the WebUI profile for the Mac's token-boundary sampler.

Explicit request knobs override a profile. Phase knobs stay separate so a
reasoning/answer transition cannot accidentally freeze the first temperature.
"""
from .generation import SAMPLERS, number, validate


def decode_request(body,*,vocab=129280,top_k_max=None,bias_max=None):
    result = dict(body)
    profile = result.get('slstack_generation')
    if profile is not None:
        profile = validate(profile,vocab=top_k_max,bias_max=bias_max)
        if profile['per_phase']:
            result['slstack_phase_samplers'] = profile['phases']
            result.setdefault('quality_speed', {p:v['bias'] for p,v in profile['phases'].items()})
        else:
            for key, value in profile['sampler'].items():
                result.setdefault(key, value)
            result.setdefault('quality_speed', profile['value'])
    if 'quality_speed' in result:
        from .native_broker import vendor
        vendor()
        from studio.quality_speed_contract import validate as quality
        value=result['quality_speed']
        if bias_max is None:result['quality_speed']=quality(value)
        else:
            from studio.quality_speed_contract import PHASES,validate_scalar
            scalar=lambda x:validate_scalar(x) if isinstance(x,str) else number(x,0,bias_max)
            if isinstance(value,dict):
                if set(value)==set(PHASES)-{'code'}:value={**value,'code':value['answer']}
                if set(value)!=set(PHASES):raise ValueError('Four quality phases required')
                result['quality_speed']={k:scalar(v) for k,v in value.items()}
            else:result['quality_speed']=scalar(value)
    limits={**SAMPLERS,**({'top_k':(0,top_k_max)} if top_k_max is not None else {})}
    for key, (lo, hi) in limits.items():
        if key in result:
            result[key] = number(result[key], lo, hi, key == 'top_k')
    if 'seed' in result:
        seed = result['seed']
        if type(seed) is not int or not 0 <= seed < 2**53:
            raise ValueError('seed must be an integer in [0,2^53)')
    for key in ('presence_penalty','frequency_penalty'):
        if key in result: result[key] = number(result[key], -2, 2)
    for key in ('repeat_penalty','repetition_penalty'):
        if key in result: result[key] = number(result[key], .001, 100)
    for key in ('repeat_last_n','penalty_last_n'):
        if key in result: result[key] = number(result[key], -1, 1048576, True)
    if 'logit_bias' in result:
        biases = result['logit_bias']
        if not isinstance(biases, dict) or len(biases)>1024: raise ValueError('logit_bias requires at most1024 token entries')
        for token,value in biases.items():
            if not isinstance(token,str) or not token.isascii() or not token.isdigit() or not 0<=int(token)<vocab:
                raise ValueError('logit_bias token outside native vocabulary')
            number(value,-100,100)
    for key,limits in dict(typical_p=(0,1),typ_p=(0,1),top_n_sigma=(-1,100),xtc_probability=(0,1),
                           xtc_threshold=(0,1),dynatemp_range=(0,2),dynatemp_exponent=(.001,100),
                           dry_multiplier=(0,100),dry_base=(1,100)).items():
        if key in result:result[key]=number(result[key],*limits)
    for key,lo in (('dry_allowed_length',0),('dry_penalty_last_n',-1)):
        if key in result:result[key]=number(result[key],lo,1048576,True)
    if 'samplers' in result:
        order=result['samplers']
        known={'top_k','typ_p','top_p','min_p','temperature','xtc','dry','penalties','top_n_sigma'}
        if not isinstance(order,list) or any(not isinstance(s,str) or s not in known for s in order) or len(set(order))!=len(order):
            raise ValueError('samplers must name distinct native sampler stages')
    # Non-WebUI custom algorithms must never be silently discarded.
    neutral = dict(tfs_z=1,mirostat=0)
    # Auxiliary knobs have no effect while their controlling sampler is neutral.
    for key, value in result.items():
        if key in neutral and value != neutral[key]:
            raise ValueError('Mac ds4 does not implement requested sampler '+key+'; no greedy fallback is allowed')
    return result
