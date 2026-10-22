"""Versioned target-only checkpoint identity, using the existing strict parser.

Bytes are the original native F32 owners. Identity does not permit weight
conversion, lower precision KV, a shifted DSpark ordinal or token-only reuse.
"""
from __future__ import annotations

from .config import ROOT
import sys

studio_root = str(ROOT / 'vendor' / 'studio')
if studio_root not in sys.path:
    sys.path.insert(0, studio_root)
from studio.dsv41_session_contract import control, require, integer, hex_value, keys, session_identity

VERSION = 'dsv41-cuda-target-state/1'
DTYPE_BITS = {'f32': 32, 'bf16': 16, 'f16': 16, 'q8_0': 8}
IDENTITY_FIELDS = ('version', 'mode', 'model_sha256', 'tokenizer_sha256',
                   'template_sha256', 'rope_sha256', 'kv_dtype', 'context')
FIELDS = (*IDENTITY_FIELDS, 'position', 'request_id', 'owner_epoch', 'generation',
          'tokens_sha256', 'logits_sha256', 'owners', 'decode_provenance')


def owner_layout(position):
    """Same kind/index/extent contract as the existing native span planner."""
    result = {('raw', i): min(position, 128) * 512 * 4 for i in range(40)}
    for i in range(4):
        live = position // (2 if i < 3 else 1)
        result['compressed', i] = live * 512 * 4
        result['index', i] = live * 128 * 4
        if i < 3 and position % 2:
            result['pool_kv', i] = result['pool_score', i] = 512 * 4
    return result


def validate(row, expected=None):
    keys(row, FIELDS)
    require(row['version'] == VERSION and type(row['mode']) is int and row['mode'] == 1,
            'Target-only state contract version/mode mismatch')
    for name in ('model_sha256', 'tokenizer_sha256', 'template_sha256', 'rope_sha256',
                 'tokens_sha256', 'logits_sha256'):
        hex_value(row[name], 64)
    require(type(row['kv_dtype']) is str and row['kv_dtype'] in DTYPE_BITS,
            'KV dtype must be explicitly qualified at 8 bit or higher')
    context = integer(row['context'], 2, 262144)
    position = integer(row['position'], 1, context - 1)
    session_identity(row['request_id'])
    integer(row['owner_epoch'], 1, (1 << 64) - 1)
    integer(row['generation'], 1, (1 << 64) - 1)
    require(type(row['decode_provenance']) is str and row['decode_provenance'] in
            ('unbiased', 'manual-bias', 'phase-bias'), 'Missing decode provenance')
    layout = owner_layout(position)
    require(type(row['owners']) is list and len(row['owners']) == len(layout),
            'Target-only owner count mismatch')
    seen = set()
    for owner in row['owners']:
        keys(owner, ('kind', 'index', 'bytes', 'sha256'))
        require(type(owner['kind']) is str, 'Invalid owner kind')
        index = integer(owner['index'], 0, 39)
        key = owner['kind'], index
        require(key in layout and key not in seen, 'Unknown/duplicate owner kind/index')
        require(integer(owner['bytes'], 0, 1 << 32) == layout[key], 'Owner extent mismatch')
        hex_value(owner['sha256'], 64)
        seen.add(key)
    if expected is not None:
        require(all(row[name] == expected[name] for name in IDENTITY_FIELDS),
                'Model/tokenizer/template/RoPE/KV/context identity mismatch')
        require((row['request_id'], row['owner_epoch']) ==
                (expected['request_id'], expected['owner_epoch']), 'Request ownership mismatch')
    return row


def parse(raw, expected=None):
    return validate(control(raw), expected)


def same_checkpoint(actual, expected):
    validate(expected)
    validate(actual, expected)
    require(all(actual[name] == expected[name] for name in
                ('position', 'generation', 'tokens_sha256', 'logits_sha256', 'decode_provenance')),
            'Checkpoint position/generation/token/logit identity mismatch')
    def keyed(row):
        return {(x['kind'], x['index']): (x['bytes'], x['sha256']) for x in row['owners']}
    require(keyed(actual) == keyed(expected), 'Actual owner byte hashes mismatch')
    return True


def ram_checkpoint_bytes(row):
    validate(row)
    # Native body and immutable token/logit buffers, plus bounded control pages.
    return sum(x['bytes'] for x in row['owners']) + 4 * row['position'] + 129280 * 4 + 65536
