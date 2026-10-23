# Derived from Studio backend/studio/dsv41_session_contract.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import hashlib, json, re, struct
MODEL_SHA256 = '2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f'
MAX_CONTROL = 1 << 20
CONTEXT = 1000001
MAX_PREFILL = 49152
MAX_DECODE = 128

class Refused(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise Refused(message)

def integer(value, lo, hi):
    require(type(value) is int and lo <= value <= hi, 'Invalid exact integer')
    return value

def hex_value(value, size):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(size) + '}', value) is not None, 'Invalid identity digest')
    return value

def session_identity(value):
    hex_value(value, 32)
    require(int(value[:16], 16) > 0, 'Zero session routing fingerprint')
    return value

def keys(row, expected):
    require(type(row) is dict and set(row) == set(expected), 'Unknown or missing session metadata')

def control(data: bytes):
    require(type(data) is bytes and 0 < len(data) <= MAX_CONTROL, 'Session control body limit')

    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'Duplicate JSON key')
            out[k] = v
        return out

    def bad_constant(value):
        raise Refused('Nonfinite JSON number')
    try:
        row = json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (ValueError, UnicodeError, RecursionError) as e:
        raise Refused('Invalid session JSON') from e
    require(type(row) is dict, 'Session control must be an object')
    pending = [(row, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        require(depth <= 16 and nodes <= 200000, 'Session JSON complexity limit')
        if isinstance(value, (list, dict)):
            pending.extend(((v, depth + 1) for v in (value.values() if isinstance(value, dict) else value)))
    return row

def checkpoint(row, mode=3):
    keys(row, ['context', 'position', 'mode', 'capture_end', 'capture_length', 'tokens_sha256', 'logits_sha256', 'owner_sha256'])
    ctx = integer(row['context'], 2, 1048576)
    pos = integer(row['position'], 1, ctx - 1)
    require(type(mode) is int and mode in (1, 3), 'Unsupported native session mode')
    require(integer(row['mode'], mode, mode) == mode, 'Native session mode mismatch')
    if mode == 3:
        require(integer(row['capture_end'], 1, ctx - 1) == pos and integer(row['capture_length'], 1, 128) == min(pos, 128), 'Incomplete native DSpark frontier')
    else:
        require(integer(row['capture_end'], 0, 0) == 0 and integer(row['capture_length'], 0, 0) == 0, 'Target-only state cannot contain a DSpark capture')
    hex_value(row['tokens_sha256'], 64)
    hex_value(row['logits_sha256'], 64)
    owners = row['owner_sha256']
    require(type(owners) is list and len(owners) == (55 if pos % 2 else 49) - (mode == 1), 'Native owner geometry differs')
    for value in owners:
        hex_value(value, 64)
    return row

def command(row, session, sequence, mode=3):
    require(type(row) is dict, 'Session command object required')
    session_identity(session)
    require(row.get('session') == session and row.get('model_sha256') == MODEL_SHA256, 'Session/model mismatch')
    require(integer(row.get('sequence'), 1, (1 << 64) - 1) == sequence, 'Replayed or skipped session sequence')
    op = row.get('operation')
    common = ['session', 'model_sha256', 'sequence', 'operation']
    extra = {'checkpoint': [], 'close': [], 'export_initial': ['base'], 'export_delta': ['base'], 'acknowledge': ['base'], 'save_rewind': ['base'], 'rewind': ['base', 'target'], 'prefill': ['base', 'tokens'], 'apply_decode_delta': ['base', 'target', 'bytes'],
             # DS4COMBO fast switch: keep-alive producer device suspend / resume onto a new native base.
             'suspend': ['base'], 'resume': ['native_base_contract_sha256', 'tokens']}
    require(type(op) is str and op in extra, 'Unknown session operation')
    keys(row, common + extra[op])
    if 'base' in row:
        checkpoint(row['base'], mode)
    if op == 'rewind':
        target = checkpoint(row['target'], mode)
        base = row['base']
        require(target['context'] == base['context'] and target['mode'] == base['mode'] and (0 < target['position'] < base['position']), 'Rewind target must be an earlier same-context checkpoint')
    if op == 'prefill':
        tokens = row['tokens']
        require(type(tokens) is list and 0 < len(tokens) <= MAX_PREFILL, 'Prefill token extent')
        for token in tokens:
            integer(token, 0, 129279)
        require(row['base']['position'] + len(tokens) < row['base']['context'], 'Prefill context exhausted')
    if op == 'resume':
        tokens = row['tokens']
        digest = row['native_base_contract_sha256']
        require(type(digest) is str and len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest), 'Resume base contract SHA-256 required')
        require(type(tokens) is list and 0 < len(tokens) <= MAX_PREFILL, 'Resume prefill token extent')
        for token in tokens:
            integer(token, 0, 129279)
    if op == 'apply_decode_delta':
        target = checkpoint(row['target'], mode)
        base = row['base']
        require(target['context'] == base['context'] and 0 < target['position'] - base['position'] <= MAX_DECODE, 'Decode delta direction/extent')
        integer(row['bytes'], (264 if mode == 3 else 256) + 4 + 129280 * 4, 64 << 20)
    return row

def initial_request(session, headers, body):
    session_identity(session)
    require(type(body) is bytes and 0 < len(body) <= MAX_PREFILL * 4 and (len(body) % 4 == 0), 'Initial token extent')
    require(type(headers) is dict and all((type(k) is str and type(v) is str for k, v in headers.items())), 'Invalid request headers')
    h = {k.lower(): v for k, v in headers.items()}
    require(len(h) == len(headers), 'Duplicate request header')
    require(h.get('content-type') == 'application/vnd.studio.dsv41.tokens-le32' and h.get('x-dsv41-model-sha256') == MODEL_SHA256 and (h.get('x-dsv41-mode') == '3'), 'Native model/mode/media mismatch')
    for key in ['content-length', 'x-dsv41-context', 'x-dsv41-tokens']:
        require(re.fullmatch('[1-9][0-9]{0,9}', h.get(key, '')) is not None, 'Invalid request length')
    require(int(h['x-dsv41-context']) == CONTEXT and int(h['x-dsv41-tokens']) * 4 == len(body) == int(h['content-length']), 'Native capacity/length mismatch')
    require(hex_value(h.get('x-dsv41-tokens-sha256'), 64) == hashlib.sha256(body).hexdigest(), 'Initial token SHA mismatch')
    require(all((t[0] < 129280 for t in struct.iter_unpack('<I', body))), 'Token outside native vocabulary')
    tier = h.get('x-dsv41-tier')
    require(tier in ['accurate', 'fast', 'ultra', 'hyper'], 'Explicit qualified tier required')
    result = dict(session=session, context=CONTEXT, tier=tier, tokens=body, tokens_sha256=h['x-dsv41-tokens-sha256'])
    if 'x-dsv41-techniques' in h:
        wire = h['x-dsv41-techniques'].encode('utf-8')
        require(len(wire) <= 4096, 'Technique header too large')
        from .dsv41_session_profiles import techniques
        result['techniques'] = techniques(tier, control(wire))
    return result
