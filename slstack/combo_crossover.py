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


def owner_crossover(r_native,r_stream,switch_seconds,native_intercept_s=0.0,stream_fixed_s=0.0):
    """Owner method: the smallest integer N >= 1 with T_native(N) > S + T_stream(N), where
    T_native(N) = native_intercept_s + N/R_native and T_stream(N) = stream_fixed_s + N/R_stream.
    Both intercepts default to 0, which is the original N/R_native > S + N/R_stream."""
    if any(not math.isfinite(v) or v<=0 for v in (r_native,r_stream,switch_seconds)) or r_native>=r_stream:
        raise ValueError('Measured crossover unavailable; handoff admission refused')
    if any(not math.isfinite(v) or v<0 for v in (native_intercept_s,stream_fixed_s)):
        raise ValueError('Measured crossover intercepts must be finite and non-negative')
    def native_slower(n):
        return native_intercept_s+n/r_native>switch_seconds+stream_fixed_s+n/r_stream
    gap=switch_seconds+stream_fixed_s-native_intercept_s
    n=max(1,math.ceil(gap/(1/r_native-1/r_stream)))
    while not native_slower(n):n+=1
    while n>1 and native_slower(n-1):n-=1
    return n


def qualified_threshold(cfg):
    # Per-launcher/preset N*: carried in the active config with its measured inputs.
    if 'combo_crossover_tokens' in cfg:
        row=cfg['combo_crossover_inputs']
        if row.get('producer_profile')!=cfg.get('producer_profile','ultra'):raise ValueError('Crossover N* belongs to another streaming preset')
        n=owner_crossover(row['r_native_tok_s'],row['r_stream_tok_s'],row['switch_seconds'],
                          row.get('native_intercept_s',0.0),row.get('stream_fixed_s',0.0))
        if n!=cfg['combo_crossover_tokens']:raise ValueError('Preset crossover N* does not match its measured inputs')
        return n
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
