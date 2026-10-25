"""Owner gate: every prefill streams through BOTH banks.

Decode repurposes the producer's two direct banks (bank 0 becomes the expert cache, bank 1
holds resident dense weights). The next prefill must switch back (LLAMADECODE_LAYOUT_SWITCH
decode=0), log LLAMADECODE_PREFILL_LAYOUT with both banks streaming, and show per-layer acquire
waits of about zero for a large prompt (the warm 40K append measured 0.1 s over 40 layers).
check() raises BankGateFailed so a single-banked prefill fails loudly."""
from __future__ import annotations
import re

LAYOUT = re.compile(rb'LLAMADECODE_PREFILL_LAYOUT banks=(\d+) streaming_banks=(\d+)')
SWITCH = re.compile(rb'LLAMADECODE_LAYOUT_SWITCH decode=(\d)')
# The only allowed single-banked prefill: the plan's explicit long-context VRAM fallback, logged
# with its reason and numbers (owner rule; shrink L2/expert cache first).
SINGLE = re.compile(rb'LLAMADECODE_PREFILL_SINGLE_BANK reason=(\S+) kv_bytes=(\d+) free_bytes=(\d+)')
BEGIN = re.compile(rb'\[dsv41-sweep\] layer=0 begin tokens=(\d+)')
LEDGER = re.compile(rb'DS4PERF_LAYER_WALL_LEDGER layer=(\d+) wall_ms=([0-9.]+) acquire_ms=([0-9.]+)')


class BankGateFailed(RuntimeError):
    pass


def sweeps(log):
    """[(layout_banks, tokens, [(layer, wall_ms, acquire_ms)])] for each prefill sweep in log order."""
    out = [];banks = None;decode = False;fallback = None
    for line in bytes(log).splitlines():
        if (m := SWITCH.search(line)):decode = m.group(1) == b'1';banks = None if decode else banks
        elif (m := SINGLE.search(line)):fallback = dict(reason=m.group(1).decode(), kv_bytes=int(m.group(2)), free_bytes=int(m.group(3)))
        elif (m := LAYOUT.search(line)):banks = (int(m.group(1)), int(m.group(2)))
        elif (m := BEGIN.search(line)):
            out.append(dict(banks=banks, fallback=fallback if banks == (1, 1) else None, after_decode_switch=not decode, tokens=int(m.group(1)), layers=[]))
        elif (m := LEDGER.search(line)) and out:out[-1]['layers'].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def check(log, large_tokens=34000, max_wait_fraction=.02):
    """Raise unless every sweep ran with two streaming banks, and large sweeps waited ~0 on DS."""
    report = []
    for n, s in enumerate(sweeps(log)):
        if s['banks'] == (1, 1) and s['fallback']:
            pass   # planned long-context fallback; reported below
        elif s['banks'] != (2, 2):
            raise BankGateFailed(f'Prefill sweep {n} ({s["tokens"]} tokens) ran without both streaming banks: {s["banks"]}')
        rest = [l for l in s['layers'] if l[0] > 0]
        wall = sum(l[1] for l in rest);wait = sum(l[2] for l in rest)
        if s['banks'] == (2, 2) and s['tokens'] >= large_tokens and wall and wait / wall > max_wait_fraction:
            raise BankGateFailed(f'Prefill sweep {n} ({s["tokens"]} tokens) exposed {wait / 1000:.1f} s DS wait over {wall / 1000:.1f} s')
        report.append(dict(sweep=n, tokens=s['tokens'], banks=s['banks'], single_bank_fallback=s['fallback'],
                           layers=len(s['layers']), wait_s=wait / 1000, wall_s=wall / 1000))
    return report
