from __future__ import annotations
import json
import math
from .engines import http_json
from .native_broker import vendor

vendor()
from studio import quality_speed

QUALITY = quality_speed.SPECS[0]['parameters']


def media_schema(kind):
    props = {'prompt': {'type': 'string', 'minLength': 1, 'maxLength': 65536, 'description': 'Required generation prompt'},
             'negative_prompt': {'type': 'string', 'maxLength': 65536, 'default': ''},
             'width': {'type': 'integer', 'minimum': 64, 'maximum': 2048, 'multipleOf': 32, 'default': 512},
             'height': {'type': 'integer', 'minimum': 64, 'maximum': 2048, 'multipleOf': 32, 'default': 512},
             'steps': {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 20},
             'guidance': {'type': 'number', 'minimum': 0, 'maximum': 30, 'default': 4},
             'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2147483647, 'default': 42}}
    if kind == 'video':
        props.update(num_frames={'type': 'integer', 'minimum': 5, 'maximum': 129, 'default': 17,
                                 'description': 'Must equal 4*k+1; default 17'},
                     fps={'type': 'integer', 'minimum': 1, 'maximum': 60, 'default': 24})
    return {'type': 'object', 'properties': props, 'required': ['prompt'], 'additionalProperties': False}


def validate(schema, args):
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be an object')
    if set(args) - set(schema['properties']):
        raise ValueError('Unknown tool settings: ' + ', '.join(sorted(set(args) - set(schema['properties']))))
    result = {}
    for key, prop in schema['properties'].items():
        if key not in args:
            if key in schema.get('required', []):
                raise ValueError(f'Missing {key}')
            if 'default' not in prop:
                continue
        value = args.get(key, prop.get('default'))
        typ = prop['type']
        valid = ((typ == 'integer' and type(value) is int) or
                 (typ == 'number' and type(value) in (int, float) and math.isfinite(value)) or
                 (typ == 'string' and isinstance(value, str)))
        if not valid:
            raise ValueError(f'{key} must be {typ}')
        if typ == 'string':
            if not prop.get('minLength', 0) <= len(value) <= prop.get('maxLength', 65536):
                raise ValueError(f'{key} length outside valid range')
        elif not prop.get('minimum', -math.inf) <= value <= prop.get('maximum', math.inf):
            raise ValueError(f'{key} outside valid range')
        if 'multipleOf' in prop and value % prop['multipleOf']:
            raise ValueError(f'{key} must be a multiple of {prop["multipleOf"]}')
        result[key] = value
    if 'num_frames' in result and (result['num_frames'] - 1) % 4:
        raise ValueError('num_frames must equal 4*k+1')
    return result


class Tools:
    def __init__(self, cfg, engines):
        self.cfg, self.engines = cfg, engines
        empty = {'type': 'object', 'properties': {}, 'additionalProperties': False}
        self.schemas = {'set_quality_speed': QUALITY, 'get_quality_speed': empty,
                        'generate_image': media_schema('image'), 'generate_video': media_schema('video')}

    def catalog(self):
        descriptions = {row['name']: row['description'] for row in quality_speed.SPECS}
        rows = []
        for name, schema in self.schemas.items():
            description = descriptions.get(name)
            if description is None:
                kind = name.removeprefix('generate_')
                description = f'Generate {kind} with the startup-selected Studio model in {self.cfg[kind + "_mode"]} mode. Media is displayed inline automatically; confirm briefly without file paths or job receipts. Only one GPU job runs at a time. Backend may reject unsupported model settings. Settings: '
                description += '; '.join(f'{key}: {json.dumps(value)}' for key, value in schema['properties'].items())
            rows.append({'name': name, 'description': description, 'inputSchema': schema})
        return rows

    def openai(self):
        # Do not spend native prefill on tools whose engine/model is unavailable.
        # MCP still publishes the fixed interface and returns explicit capability errors.
        rows = [row for row in self.catalog() if
                (row['name'] in ('set_quality_speed', 'get_quality_speed')) or
                (row['name'] == 'generate_image' and (self.cfg['image_model'] or self.cfg['comfyui_models'])) or
                (row['name'] == 'generate_video' and (self.cfg['video_model'] or self.cfg['comfyui_models']))]
        return [{'type': 'function', 'function': {'name': row['name'], 'description': row['description'],
                                                'parameters': row['inputSchema']}} for row in rows]

    def call(self, name, args):
        if name not in self.schemas:
            raise ValueError('Unknown stack tool: ' + name)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be an object')
        quality = name in ('set_quality_speed', 'get_quality_speed')
        if quality and name == 'get_quality_speed' and args:
            raise ValueError('get_quality_speed takes no arguments')
        checked = quality_speed.setting_arguments(args, 0) if name == 'set_quality_speed' else None
        if name in ('set_quality_speed', 'get_quality_speed') and not self.cfg.get('bias_mcp_url'):
            if name == 'get_quality_speed':
                return self.engines.quality()
            if set(args) == {'value'}:
                value = checked
            else:
                state = self.engines.quality()
                if state.get('supported') is False:
                    return state
                current = state.get('phases') if state.get('per_phase') else state.get('value', 0)
                value = quality_speed.setting_arguments(args, current)
            return self.engines.quality(value)
        if not quality:
            args = validate(self.schemas[name], args)
        if name.startswith('generate_'):
            return self.engines.media(name.removeprefix('generate_'), args)
        url = self.cfg['bias_mcp_url']
        if not url:
            raise RuntimeError('DS4WINBIAS2 has not delivered a live quality/speed MCP service; set bias_mcp_url when available. No setting was changed.')
        # JSON-RPC transport to the existing MCP implementation; do not duplicate bias engine logic.
        reply = http_json(url, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                               'params': {'name': name, 'arguments': args}}, timeout=30)
        if reply.get('error'):
            raise RuntimeError(str(reply['error']))
        result = reply.get('result')
        if not isinstance(result, dict) or result.get('isError'):
            raise RuntimeError('Upstream bias tool failed: ' + str(result))
        return result
