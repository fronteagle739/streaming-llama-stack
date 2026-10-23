"""Measured-input port of Kimi's compute_stream_crossover_tokens.

Source: llama-context.cpp, same floor+n/stream versus n/native equation and
ceil/clamp semantics. A production route requires matching measured receipts.
"""
import math


def crossover(stream_floor_ms,stream_tok_s,native_tok_s,minimum=100):
    if any(not math.isfinite(v) or v<=0 for v in (stream_floor_ms,stream_tok_s,native_tok_s)) or native_tok_s>=stream_tok_s:
        raise ValueError('Measured crossover unavailable; handoff admission refused')
    denominator=1/native_tok_s-1/stream_tok_s
    tokens=(stream_floor_ms/1000)/denominator
    if not math.isfinite(tokens) or tokens<=0:raise ValueError('Measured crossover is invalid')
    return min(2147483647,math.ceil(max(tokens,minimum)))


def qualified_threshold(cfg):
    import json
    from pathlib import Path
    row=json.loads(Path(cfg['combo_crossover_receipt']).read_text())
    if row.get('status')!='PASS' or row.get('native_sha256')!=cfg['native_sha256'] or row.get('producer_sha256')!=cfg['producer_sha256']:
        raise ValueError('Measured crossover receipt is absent, unqualified or from different binaries')
    if row.get('identity')!=cfg['combo_identity'] or row.get('context')!=cfg['context']:
        raise ValueError('Measured crossover identity/context differs')
    if row.get('provider_schema')!=cfg['provider_schema']:
        raise ValueError('Measured crossover quality/speed provider schema differs')
    if row.get('native_prefill_chunk')!=cfg['native_prefill_chunk']:
        raise ValueError('Measured crossover native workspace cap differs')
    result=crossover(row['stream_floor_ms'],row['stream_tok_s'],row['native_tok_s'],row.get('minimum_tokens',100))
    if result!=row.get('crossover_tokens') or row.get('bracket_verified') is not True or row.get('repeated_bracket_verified') is not True:
        raise ValueError('Measured crossover has no verified native/ULTRA bracket')
    return result
