# Derived from Studio backend/studio/quality_speed.py; source SHA256 fbd105933dfbd4f0b736153f0aa2808cedeab81a7a2c2f3d3187f3f59adcffea
# Only the delivered tool schema and validator are reused. Transport is supplied by stack glue.
import math
NAMED = {'max_quality': 0.0, 'balanced': 0.35, 'max_speed': 0.9}
DESCRIPTION = "Quality vs speed for calibrated ds4-win. 0/max_quality preserves original routing; 0.35/balanced trades some accuracy for speed; 0.9/max_speed favors resident experts. Current 384-token benchmark post-warm rates: 6.03 / 10.30 / 14.43 tok/s respectively; KL 0.000 / 0.104 / 0.359 over 383 decode predictions including 100 unbiased warm-up tokens. These are prompt-specific measurements, not guaranteed rates. Prefer max_quality for code/careful reasoning, speed for drafts, or follow the user's choice. Applies at the next decode token, persists for this chat. Warm-up 100 and ramp 8 still apply. Adaptive mode overrides manual settings and is never disabled."
SPECS = [{'name': 'set_quality_speed', 'label': 'Quality vs speed', 'category': 'agent', 'danger': False, 'description': DESCRIPTION, 'parameters': {'type': 'object', 'properties': {'value': {'anyOf': [{'type': 'number', 'minimum': 0, 'maximum': 0.9}, {'type': 'string', 'enum': list(NAMED)}]}}, 'required': ['value'], 'additionalProperties': False}}, {'name': 'get_quality_speed', 'label': 'Get quality vs speed', 'category': 'agent', 'danger': False, 'description': "Return this chat's Quality vs speed value, applied engine value, recent tok/s and adaptive status. " + DESCRIPTION, 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}]

def validate(value):
    if isinstance(value, str):
        if value not in NAMED:
            raise ValueError('use max_quality, balanced, max_speed, or a number in [0,0.9]')
        value = NAMED[value]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (not 0 <= value <= 0.9):
        raise ValueError('Quality vs speed must be a finite number in [0,0.9]')
    return float(value)
