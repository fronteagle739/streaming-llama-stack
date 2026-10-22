# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/streaming_flag.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse

def add_streaming_arg(ap: argparse.ArgumentParser, *, with_aliases: bool=True) -> None:
    ap.add_argument('--streaming', choices=['on', 'off'], default=None, help="ONE toggle for streaming training. 'on' = forward AND backward layer-streamed (DirectStorage / per-layer re-stream; ~1 layer base resident — the big-model path). 'off' = resident (whole frozen base on GPU; the prior QLoRA/fp8/bf16 behaviour). Omitted = the trainer's default.")
    if with_aliases:
        g = ap.add_argument_group('deprecated streaming aliases (map onto --streaming)')
        g.add_argument('--streamed', action='store_true', help='DEPRECATED alias for --streaming on (Cosmos §4b DS-reuse base).')
        g.add_argument('--streamed-reverse', dest='streamed_reverse', action='store_true', help='DEPRECATED alias for --streaming on (Cosmos §4c reverse-ds backward).')
        g.add_argument('--resident', action='store_true', help='DEPRECATED alias for --streaming off (force resident training).')

def resolve_streaming(args: argparse.Namespace, default: bool) -> bool:
    val = getattr(args, 'streaming', None)
    if val is not None:
        return val == 'on'
    if getattr(args, 'streamed', False) or getattr(args, 'streamed_reverse', False):
        return True
    if getattr(args, 'resident', False):
        return False
    return bool(default)
