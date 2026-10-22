# Derived from Studio backend/studio/quality_speed.py at99facdf8a83970793f6141ea267039f18f6733b6.

# Only the delivered schema and pure validators are exported; stack supplies transport.

import math

NAMED = {"max_quality": 0.0, "balanced": 0.35, "fast": 0.65, "faster": 0.9, "max_speed": 1.36125}

DESCRIPTION = 'Set Quality vs speed (0–1.36125 or max_quality, balanced, fast, faster, max_speed); use value or optional reasoning/tool_call/answer/code settings. Modes above Balanced are faster but may produce worse answers.'

VALUE_SCHEMA = {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1.36125},
                            {"type": "string", "enum": list(NAMED)}]}

SPECS = [
    {"name": "set_quality_speed", "label": "Quality vs speed", "category": "agent", "danger": False,
     "description": DESCRIPTION,
     "parameters": {"type": "object", "properties": {"value": VALUE_SCHEMA,
                     **{p: VALUE_SCHEMA for p in ('reasoning','tool_call','answer','code')}},
                    "additionalProperties": False,
                    "anyOf": [{"required": [p]} for p in ('value','reasoning','tool_call','answer','code')]}},
    {"name": "get_quality_speed", "label": "Get quality vs speed", "category": "agent", "danger": False,
     "description": "Read this chat’s Quality vs speed setting and phase settings.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]

def validate_scalar(value):
    if isinstance(value, str):
        if value not in NAMED:
            raise ValueError("use max_quality, balanced, fast, faster, max_speed, or a number in [0,1.36125]")
        value = NAMED[value]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1.36125:
        raise ValueError("Quality vs speed must be a finite number in [0,1.36125]")
    return float(value)

PHASES = ('reasoning', 'tool_call', 'answer', 'code')

def validate(value):
    if isinstance(value, dict):
        if set(value) == set(PHASES)-{'code'}:
            value={**value,'code':value['answer']}
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
    base.update({p: validate_scalar(v) for p,v in args.items()})
    return validate(base)
