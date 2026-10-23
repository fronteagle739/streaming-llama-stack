# Pure existing Studio quality/speed validator export; source SHA256 83a0536a9b10d402d2ac8f8d48a736e366d8d337ee3156e1534d3562ff5a3e76
import math

NAMED = {'max_quality': 0.0, 'balanced': 0.35, 'fast': 0.65, 'faster': 0.9, 'max_speed': 1.36125}

def validate_scalar(value):
    if isinstance(value, str):
        if value not in NAMED:
            raise ValueError('use max_quality, balanced, fast, faster, max_speed, or a number in [0,1.36125]')
        value = NAMED[value]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (not 0 <= value <= 1.36125):
        raise ValueError('Quality vs speed must be a finite number in [0,1.36125]')
    return float(value)
PHASES = ('reasoning', 'tool_call', 'answer', 'code')

def validate(value):
    if isinstance(value, dict):
        if set(value) == set(PHASES) - {'code'}:
            value = {**value, 'code': value['answer']}
        if set(value) != set(PHASES):
            raise ValueError('per-phase value requires reasoning, tool_call, answer and optional code')
        return {phase: validate_scalar(value[phase]) for phase in PHASES}
    return validate_scalar(value)

def setting_arguments(args, current):
    if set(args) == {'value'}:
        return validate(args['value'])
    if not args or not set(args).issubset(PHASES):
        raise ValueError('use value alone, or reasoning/tool_call/answer/code settings')
    base = validate(current) if isinstance(current, dict) else {p: current for p in PHASES}
    base.update({p: validate_scalar(v) for p, v in args.items()})
    return validate(base)
